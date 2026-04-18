//! End-to-end tests for `run_benchmark`.
//!
//! Uses an in-memory benchmark + a scripted LLM so the runner can
//! exercise its full loop without network / filesystem dependencies
//! beyond `tempfile`.

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Usage,
};
use oharness_eval::{
    run_benchmark, AlwaysPassEvaluator, BenchmarkRunConfig, InMemoryBenchmark, InMemoryTask,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use std::sync::Arc;
use tempfile::tempdir;

struct OneShotLlm;

#[async_trait]
impl Llm for OneShotLlm {
    fn name(&self) -> &str {
        "scripted"
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }
    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "msg".into(),
            model: ModelId::new("m"),
            content: vec![Content::text("done")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }
    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

fn sample_benchmark() -> InMemoryBenchmark {
    InMemoryBenchmark::new(
        "demo",
        "demo-1",
        vec![
            InMemoryTask::new("t-0", Task::new("first")),
            InMemoryTask::new("t-1", Task::new("second")),
            InMemoryTask::new("t-2", Task::new("third")),
        ],
        Arc::new(AlwaysPassEvaluator),
    )
}

/// Factory body. Takes no arguments — in the tests below it's always
/// called via `|_lt| build_agent()`, which satisfies the runner's
/// `Fn(&LoadedTask) -> impl Future` bound without borrowing from
/// `_lt` into the returned future (that would make it non-'static).
async fn build_agent() -> Result<Agent, oharness_core::AgentError> {
    Agent::builder()
        .with_llm(Arc::new(OneShotLlm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()
}

#[tokio::test]
async fn runs_every_task_and_writes_artifacts() {
    let dir = tempdir().unwrap();
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf());
    let report = run_benchmark(sample_benchmark(), |_lt| build_agent(), config).await;

    assert_eq!(report.tasks.len(), 3);
    assert!(report.tasks.iter().all(|t| t.error.is_none()));
    assert!((report.pass_at_1() - 1.0).abs() < 1e-9);

    // Each task's artifacts landed under {dir}/{task_id}/.
    for task in &report.tasks {
        let task_dir = dir.path().join(&task.task_id);
        assert!(
            task_dir.join("outcome.json").exists(),
            "outcome.json missing for {}",
            task.task_id
        );
        assert!(task_dir.join("evaluation.json").exists());
        assert!(task_dir.join("trajectory.jsonl").exists());
    }
    // Run-level config + manifest.
    assert!(dir.path().join("config.toml").exists());
    assert!(dir.path().join("manifest.json").exists());
}

#[tokio::test]
async fn filter_limits_scheduled_tasks() {
    let dir = tempdir().unwrap();
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf()).with_filter("t-1");
    let report = run_benchmark(sample_benchmark(), |_lt| build_agent(), config).await;
    assert_eq!(report.tasks.len(), 1);
    assert_eq!(report.tasks[0].task_id, "t-1");
}

#[tokio::test]
async fn sample_n_takes_prefix() {
    let dir = tempdir().unwrap();
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf()).with_sample_n(2);
    let report = run_benchmark(sample_benchmark(), |_lt| build_agent(), config).await;
    assert_eq!(report.tasks.len(), 2);
    let ids: Vec<_> = report.tasks.iter().map(|t| t.task_id.clone()).collect();
    // Order of completion isn't guaranteed, but the set must be the
    // first two ids.
    assert!(ids.contains(&"t-0".to_string()));
    assert!(ids.contains(&"t-1".to_string()));
    assert!(!ids.contains(&"t-2".to_string()));
}

#[tokio::test]
async fn resume_skips_already_completed_tasks() {
    let dir = tempdir().unwrap();
    // First run — complete everything.
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf());
    let first = run_benchmark(sample_benchmark(), |_lt| build_agent(), config).await;
    assert_eq!(first.tasks.len(), 3);

    // Second run with resume=true — runner should read back the
    // on-disk outcomes instead of re-running.
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf()).with_resume(true);
    let second = run_benchmark(
        sample_benchmark(),
        // Factory will panic if invoked — resume must avoid the factory
        // entirely for all three tasks.
        |_lt| async move {
            panic!("resume must not invoke the factory");
        },
        config,
    )
    .await;
    assert_eq!(second.tasks.len(), 3);
    for t in &second.tasks {
        assert!(t.evaluation.as_ref().expect("eval").passed);
    }
}

#[tokio::test]
async fn factory_errors_surface_as_skipped_tasks() {
    let dir = tempdir().unwrap();
    let config = BenchmarkRunConfig::new(dir.path().to_path_buf());
    let report = run_benchmark(
        sample_benchmark(),
        |_lt| async move {
            Err(oharness_core::AgentError::Configuration(
                "synthetic factory failure".into(),
            ))
        },
        config,
    )
    .await;
    assert_eq!(report.tasks.len(), 3);
    for t in &report.tasks {
        assert!(t.evaluation.is_none());
        assert!(
            t.error.as_ref().is_some_and(|e| e.starts_with("factory:")),
            "expected factory error, got {:?}",
            t.error
        );
    }
    assert_eq!(report.pass_at_1(), 0.0);
}
