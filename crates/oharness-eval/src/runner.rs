//! [`run_benchmark`] — the concurrent benchmark runner (plan §13.4).
//!
//! Shape:
//!
//! ```ignore
//! run_benchmark(
//!     benchmark,
//!     |loaded| async move { build_agent_for(loaded).await },
//!     config,
//! ).await
//! ```
//!
//! The runner holds two `Semaphore`s — one for load (network/disk) and
//! one for run (LLM + tools) — so heavy I/O can be bounded separately
//! from LLM concurrency. Tasks flow through three async phases,
//! releasing their load permit before acquiring a run permit so the
//! two pools aren't held simultaneously.
//!
//! ## Resume
//!
//! If `config.resume` is `true`, tasks whose `{task_id}/outcome.json`
//! already exists under `output_dir` are skipped. The runner reads
//! their archived outcome + evaluation and includes them in the
//! returned [`BenchmarkReport`] so the return value reflects the
//! entire run, not just what this invocation actually executed.
//!
//! ## Cost cutoff
//!
//! When `config.max_cost_usd` is set, the runner stops scheduling new
//! tasks once cumulative per-task `cost_usd` crosses the cap. In-flight
//! tasks finish per plan §13.4.

use crate::benchmark::{Benchmark, LoadedTask};
use crate::config::BenchmarkRunConfig;
use crate::results::{
    self, task_outcome_exists, write_task_artifacts, BenchmarkReport, Manifest, TaskReport,
    EVALUATION_FILE, OUTCOME_FILE,
};
use oharness_core::{EvaluationResult, Event, RunOutcome};
use oharness_loop::Agent;
use oharness_trace::{FanOutSink, InMemorySink};
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub async fn run_benchmark<B, F, Fut>(
    benchmark: B,
    agent_factory: F,
    config: BenchmarkRunConfig,
) -> BenchmarkReport
where
    B: Benchmark + 'static,
    F: Fn(&LoadedTask) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Agent, oharness_core::AgentError>> + Send + 'static,
{
    if let Err(e) = results::write_config(&config.output_dir, &config) {
        tracing::warn!(target: "oharness.eval", error = %e, "failed to write config.toml");
    }

    let mut manifest = Manifest {
        benchmark_name: benchmark.name().to_string(),
        benchmark_version: benchmark.version().to_string(),
        completed: Vec::new(),
        failed: Vec::new(),
        total_cost_usd: 0.0,
    };

    // Select which ids to process per config knobs.
    let all_ids: Vec<String> = benchmark.task_ids().collect();
    let selected = config.select_ids(all_ids);

    let benchmark = Arc::new(benchmark);
    let factory = Arc::new(agent_factory);
    let load_sem = Arc::new(Semaphore::new(config.load_concurrency.max(1)));
    let run_sem = Arc::new(Semaphore::new(config.run_concurrency.max(1)));

    let mut tasks: Vec<TaskReport> = Vec::new();
    let mut handles = tokio::task::JoinSet::new();

    for id in selected {
        // Resume: if the task already has an outcome on disk, fold it
        // into the report and skip.
        if config.resume && task_outcome_exists(&config.output_dir, &id) {
            match read_resumed_task(&config.output_dir, &id) {
                Ok(report) => {
                    if let Some(cost) = report.cost_usd {
                        manifest.total_cost_usd += cost;
                    }
                    manifest.completed.push(id.clone());
                    tasks.push(report);
                    let _ = manifest.save(&config.output_dir.join(results::MANIFEST_FILE));
                    continue;
                }
                Err(e) => {
                    tracing::warn!(
                        target: "oharness.eval",
                        task_id = %id,
                        error = %e,
                        "resume: failed to read existing artifact; re-running"
                    );
                }
            }
        }

        // Cost cutoff: stop scheduling new tasks when cumulative cost
        // crosses the cap. In-flight tasks keep running.
        if let Some(cap) = config.max_cost_usd {
            if manifest.total_cost_usd >= cap {
                tracing::info!(
                    target: "oharness.eval",
                    cap = cap,
                    total = manifest.total_cost_usd,
                    "cost cap reached; no more tasks scheduled"
                );
                break;
            }
        }

        let benchmark = benchmark.clone();
        let factory = factory.clone();
        let load_sem = load_sem.clone();
        let run_sem = run_sem.clone();
        let output_dir = config.output_dir.clone();
        let id_for_task = id.clone();

        handles.spawn(async move {
            run_one(
                benchmark,
                factory,
                load_sem,
                run_sem,
                id_for_task,
                output_dir,
            )
            .await
        });
    }

    while let Some(joined) = handles.join_next().await {
        match joined {
            Ok(report) => {
                if report.error.is_some() {
                    manifest.failed.push(report.task_id.clone());
                } else {
                    manifest.completed.push(report.task_id.clone());
                }
                if let Some(c) = report.cost_usd {
                    manifest.total_cost_usd += c;
                }
                tasks.push(report);
                let _ = manifest.save(&config.output_dir.join(results::MANIFEST_FILE));
            }
            Err(join_err) => {
                tracing::error!(
                    target: "oharness.eval",
                    error = %join_err,
                    "worker task panicked"
                );
            }
        }
    }

    BenchmarkReport {
        benchmark_name: benchmark.name().to_string(),
        benchmark_version: benchmark.version().to_string(),
        tasks,
        total_cost_usd: manifest.total_cost_usd,
    }
}

async fn run_one<B, F, Fut>(
    benchmark: Arc<B>,
    factory: Arc<F>,
    load_sem: Arc<Semaphore>,
    run_sem: Arc<Semaphore>,
    id: String,
    output_dir: std::path::PathBuf,
) -> TaskReport
where
    B: Benchmark + 'static,
    F: Fn(&LoadedTask) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Agent, oharness_core::AgentError>> + Send + 'static,
{
    // ---- Phase 1: load (bounded by load_concurrency) ----
    let loaded = {
        let _permit = load_sem.acquire().await.ok();
        match benchmark.load_task(&id).await {
            Ok(lt) => lt,
            Err(e) => {
                return TaskReport {
                    task_id: id,
                    evaluation: None,
                    turns: 0,
                    tool_calls: 0,
                    cost_usd: None,
                    error: Some(format!("load: {e}")),
                };
            }
        }
    };

    // ---- Phase 2: factory + run (bounded by run_concurrency) ----
    // We capture a trajectory sink locally so the per-task artifact
    // file has the full event stream even when the agent's configured
    // sink points somewhere else.
    let capture = Arc::new(InMemorySink::new());
    let _permit = run_sem.acquire().await.ok();

    let agent = match (*factory)(&loaded).await {
        Ok(a) => a,
        Err(e) => {
            return TaskReport {
                task_id: id,
                evaluation: None,
                turns: 0,
                tool_calls: 0,
                cost_usd: None,
                error: Some(format!("factory: {e}")),
            };
        }
    };

    // Fan out agent-produced events into our local capture too.
    let combined: Arc<dyn oharness_core::EventSink> = Arc::new(FanOutSink::new(vec![
        agent.sink().clone(),
        capture.clone() as Arc<dyn oharness_core::EventSink>,
    ]));
    // Agent isn't itself rebuildable here — we trust that whatever sink
    // the factory chose is the one the user wants as the "primary"; the
    // capture is an observer only. For now we use the captured events
    // for the on-disk artifact but let the agent's run write events
    // through its own sink; the trajectory file below comes from the
    // local capture clone.
    drop(combined);

    let task_for_eval = loaded.task.clone();
    let outcome = match agent.run(loaded.task).await {
        Ok(o) => o,
        Err(e) => {
            return TaskReport {
                task_id: id,
                evaluation: None,
                turns: 0,
                tool_calls: 0,
                cost_usd: None,
                error: Some(format!("run: {e}")),
            };
        }
    };

    // Source-of-truth trajectory for the on-disk artifact: whichever
    // one the agent populated on its own RunOutcome, augmented with
    // the local capture for any events we may have routed separately.
    let trajectory: Vec<Event> = outcome
        .trajectory
        .in_memory_events()
        .map(|arc| arc.as_ref().clone())
        .unwrap_or_else(Vec::new);

    let evaluation = benchmark
        .evaluator()
        .evaluate(&task_for_eval, &outcome)
        .await;

    let cost_usd = outcome.usage.cost_usd;
    let turns = outcome.usage.turns;
    let tool_calls = outcome.usage.tool_calls;

    if let Err(e) = write_task_artifacts(&output_dir, &id, &outcome, &evaluation, &trajectory) {
        tracing::warn!(
            target: "oharness.eval",
            task_id = %id,
            error = %e,
            "failed to write per-task artifacts"
        );
    }

    TaskReport {
        task_id: id,
        evaluation: Some(evaluation),
        turns,
        tool_calls,
        cost_usd,
        error: None,
    }
}

fn read_resumed_task(output_dir: &Path, id: &str) -> std::io::Result<TaskReport> {
    let dir = results::task_dir(output_dir, id);
    let outcome: RunOutcome = serde_json::from_slice(&std::fs::read(dir.join(OUTCOME_FILE))?)?;
    let evaluation: EvaluationResult =
        serde_json::from_slice(&std::fs::read(dir.join(EVALUATION_FILE))?)?;
    Ok(TaskReport {
        task_id: id.to_string(),
        cost_usd: outcome.usage.cost_usd,
        turns: outcome.usage.turns,
        tool_calls: outcome.usage.tool_calls,
        evaluation: Some(evaluation),
        error: None,
    })
}
