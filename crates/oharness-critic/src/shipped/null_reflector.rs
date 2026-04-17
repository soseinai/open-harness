use crate::reflector::Reflector;
use async_trait::async_trait;
use oharness_core::{Episode, Reflection};

/// A reflector that always returns `None`. Useful for debugging,
/// benchmarks that want to measure reflexion overhead against a no-op
/// baseline, and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullReflector;

#[async_trait]
impl Reflector for NullReflector {
    fn name(&self) -> &str {
        "null"
    }

    async fn reflect(&self, _episode: &Episode<'_>) -> Option<Reflection> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::{
        CompletionReason, EvaluationResult, MetadataMap, ResourceUsage, RunOutcome, Task,
        Termination, TrajectoryHandle,
    };
    use time::OffsetDateTime;

    fn dummy_episode<'a>(
        task: &'a Task,
        outcome: &'a RunOutcome,
        eval: &'a EvaluationResult,
        prior: &'a [Reflection],
    ) -> Episode<'a> {
        Episode {
            index: 0,
            task,
            outcome,
            evaluation: eval,
            prior_reflections: prior,
        }
    }

    #[tokio::test]
    async fn null_reflector_returns_none() {
        let reflector = NullReflector;
        let task = Task::new("t");
        let outcome = RunOutcome {
            run_id: oharness_core::RunId::new(),
            task_id: task.id.clone(),
            termination: Termination::Completed {
                reason: CompletionReason::EndTurn,
            },
            final_messages: Vec::new(),
            trajectory: TrajectoryHandle::in_memory(Vec::new()),
            usage: ResourceUsage::default(),
            per_model_usage: Default::default(),
            started_at: OffsetDateTime::now_utc(),
            finished_at: OffsetDateTime::now_utc(),
            agent_state: MetadataMap::new(),
        };
        let eval = EvaluationResult::fail();
        let ep = dummy_episode(&task, &outcome, &eval, &[]);
        assert!(reflector.reflect(&ep).await.is_none());
        assert_eq!(reflector.name(), "null");
    }
}
