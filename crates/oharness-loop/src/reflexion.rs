//! [`run_reflexion`] — multi-episode wrapper that threads reflections
//! from a [`Reflector`] into the next episode via the agent's
//! [`ReflectionInjector`]. Not a [`Loop`] impl itself (plan §12.2 /
//! §12.6). Behind the `reflexion` feature.
//!
//! The evaluator is typed as [`oharness_core::TaskEvaluator`] — the
//! same trait used by `oharness_eval::run_benchmark`. Any evaluator
//! implementation round-trips freely between reflexion and benchmark
//! runs.

use crate::agent::Agent;
use oharness_core::{AgentError, Episode, OwnedEpisode, Reflection, Task, TaskEvaluator};
use oharness_critic::Reflector;
use std::sync::Arc;

/// Run `max_episodes` attempts of the agent on `task`, threading
/// reflections produced by `reflector` into subsequent episodes via
/// the agent's [`ReflectionInjector`]. Returns one [`OwnedEpisode`]
/// per attempt. Stops early when `evaluator.evaluate(..).passed` is
/// `true`.
///
/// `Err(AgentError::Configuration)` is returned immediately — **before
/// any episode runs** — if the agent wasn't built with
/// `.with_reflection_injector(..)`.
///
/// Plan §12.6 sketch:
/// - start each iteration by calling `injector.set_reflections(prior)`
/// - run `agent.run(task.clone())`
/// - evaluate
/// - call `reflector.reflect(&episode)`; if Some, append to prior and
///   emit `reflection.generated { index, text }`
/// - stop if `evaluation.passed`
pub async fn run_reflexion(
    agent: &Agent,
    task: Task,
    evaluator: Arc<dyn TaskEvaluator>,
    reflector: Arc<dyn Reflector>,
    max_episodes: u32,
) -> Result<Vec<OwnedEpisode>, AgentError> {
    let Some(injector) = agent.injector() else {
        return Err(AgentError::Configuration(
            "run_reflexion requires an agent built with .with_reflection_injector(...)".into(),
        ));
    };

    let mut reflections: Vec<Reflection> = Vec::new();
    let mut out: Vec<OwnedEpisode> = Vec::new();

    for i in 0..max_episodes {
        injector.set_reflections(reflections.clone());
        injector.bump_episode();

        let outcome = agent.run(task.clone()).await?;
        let evaluation = evaluator.evaluate(&task, &outcome).await;

        let episode = Episode {
            index: i,
            task: &task,
            outcome: &outcome,
            evaluation: &evaluation,
            prior_reflections: &reflections,
        };

        let should_stop = evaluation.passed;

        // Call the reflector and materialize the owned episode before
        // mutating `reflections` — `episode` holds an immutable borrow
        // on the current reflection list.
        let maybe_reflection = reflector.reflect(&episode).await;
        let owned = episode.to_owned();
        if let Some(reflection) = maybe_reflection {
            // Emit reflection.generated so downstream tools / replay
            // can observe when the reflector actually produced a note.
            // Not emitted when the reflector returns None.
            emit_reflection_generated(agent, i, &reflection);
            reflections.push(reflection);
        }
        out.push(owned);

        if should_stop {
            break;
        }
    }

    Ok(out)
}

fn emit_reflection_generated(agent: &Agent, episode_index: u32, reflection: &Reflection) {
    use oharness_core::event::{EventKind, SchemaVersion};
    use oharness_core::{Event, RunId, SpanId};
    use serde_json::json;

    // We don't share a ScopedEmitter with the agent's run — the run
    // owns the sequence counter for its own trajectory. For the
    // inter-episode reflection-generated event, we just emit onto the
    // raw sink with a synthetic envelope. Readers that care about
    // precise seq ordering should look for this event between
    // successive runs by its run_id / episode_index.
    let event = Event {
        v: SchemaVersion::CURRENT,
        seq: 0, // synthetic — this event lives between agent runs
        run_id: RunId::new(),
        timestamp: Some(time::OffsetDateTime::now_utc()),
        span_id: SpanId::from("reflexion"),
        parent: None,
        kind: EventKind::ReflectionGenerated(json!({
            "episode_index": episode_index,
            "text": reflection.text,
            "metadata": reflection.metadata,
        })),
        redactions: Vec::new(),
    };
    agent.sink().emit(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentBuilder;
    use async_trait::async_trait;
    use oharness_core::{
        CompletionReason, CompletionResponse, Content, EvaluationResult, LlmCapabilities,
        MetadataMap, ModelId, ResourceUsage, RunOutcome, StopReason, Task, Termination,
        TrajectoryHandle, Usage,
    };
    use oharness_critic::{shipped::NullReflector, ReflectionInjector};
    use oharness_llm::{ChunkStream, CompletionRequest, Llm, LlmError};

    fn dummy_outcome() -> RunOutcome {
        RunOutcome {
            run_id: oharness_core::RunId::new(),
            task_id: None,
            termination: Termination::Completed {
                reason: CompletionReason::EndTurn,
            },
            final_messages: Vec::new(),
            trajectory: TrajectoryHandle::in_memory(Vec::new()),
            usage: ResourceUsage::default(),
            per_model_usage: Default::default(),
            started_at: time::OffsetDateTime::now_utc(),
            finished_at: time::OffsetDateTime::now_utc(),
            agent_state: MetadataMap::new(),
        }
    }

    struct OneTurnLlm;
    #[async_trait]
    impl Llm for OneTurnLlm {
        fn name(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                id: "x".into(),
                model: ModelId::new("m"),
                content: vec![Content::text("ok")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
        async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            Err(LlmError::Unsupported("stream"))
        }
    }

    struct AlwaysPass;
    #[async_trait]
    impl TaskEvaluator for AlwaysPass {
        async fn evaluate(&self, _: &Task, _: &RunOutcome) -> EvaluationResult {
            EvaluationResult::pass()
        }
    }

    struct AlwaysFail;
    #[async_trait]
    impl TaskEvaluator for AlwaysFail {
        async fn evaluate(&self, _: &Task, _: &RunOutcome) -> EvaluationResult {
            EvaluationResult::fail()
        }
    }

    struct EmptyTools;
    #[async_trait]
    impl oharness_tools::ToolSet for EmptyTools {
        fn specs(&self) -> &[oharness_core::ToolSpec] {
            &[]
        }
        async fn execute(
            &self,
            _name: &str,
            _input: serde_json::Value,
            _ctx: &oharness_tools::context::ToolContext,
        ) -> oharness_tools::toolset::ToolOutcome {
            oharness_tools::toolset::ToolOutcome::error("no-op", false)
        }
    }

    #[tokio::test]
    async fn run_reflexion_errors_when_no_injector_configured() {
        let agent = AgentBuilder::default()
            .with_llm(Arc::new(OneTurnLlm))
            .with_tools(Arc::new(EmptyTools))
            .build()
            .expect("agent");
        let result = run_reflexion(
            &agent,
            Task::new("t"),
            Arc::new(AlwaysPass),
            Arc::new(NullReflector),
            3,
        )
        .await;
        match result {
            Err(AgentError::Configuration(msg)) => assert!(msg.contains("reflection_injector")),
            other => panic!("expected Configuration, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_reflexion_stops_on_pass() {
        let injector = Arc::new(ReflectionInjector::new());
        let agent = AgentBuilder::default()
            .with_llm(Arc::new(OneTurnLlm))
            .with_tools(Arc::new(EmptyTools))
            .with_reflection_injector(injector.clone())
            .build()
            .expect("agent");
        let episodes = run_reflexion(
            &agent,
            Task::new("t"),
            Arc::new(AlwaysPass),
            Arc::new(NullReflector),
            5,
        )
        .await
        .expect("reflexion ok");
        // First episode passes → loop stops after 1 episode.
        assert_eq!(episodes.len(), 1);
        assert!(episodes[0].evaluation.passed);
    }

    #[tokio::test]
    async fn run_reflexion_runs_all_episodes_when_nothing_passes() {
        let injector = Arc::new(ReflectionInjector::new());
        let agent = AgentBuilder::default()
            .with_llm(Arc::new(OneTurnLlm))
            .with_tools(Arc::new(EmptyTools))
            .with_reflection_injector(injector.clone())
            .build()
            .expect("agent");
        let episodes = run_reflexion(
            &agent,
            Task::new("t"),
            Arc::new(AlwaysFail),
            Arc::new(NullReflector),
            3,
        )
        .await
        .expect("reflexion ok");
        assert_eq!(episodes.len(), 3);
        for ep in &episodes {
            assert!(!ep.evaluation.passed);
        }
    }

    // Dummy helper so the outcome-only test code above stays quiet about
    // unused items when features vary.
    #[allow(dead_code)]
    fn _unused_outcome() -> RunOutcome {
        dummy_outcome()
    }
}
