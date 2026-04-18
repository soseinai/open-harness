//! [`InMemoryBenchmark`] — a fixture that implements [`Benchmark`] over
//! a hand-rolled task list. Intended for tests, tutorials, and
//! "harness-on-harness" smoke runs where no real dataset exists yet.

use crate::benchmark::{Benchmark, BenchmarkError, LoadedTask};
use async_trait::async_trait;
use oharness_core::{EvaluationResult, RunOutcome, Task, TaskEvaluator};
use std::sync::Arc;

/// Single-task fixture entry. The evaluator is invoked with the full
/// [`RunOutcome`]; for static pass/fail entries, use an
/// [`AlwaysPassEvaluator`] or similar simple impl below.
pub struct InMemoryTask {
    pub id: String,
    pub task: Task,
}

impl InMemoryTask {
    pub fn new(id: impl Into<String>, task: Task) -> Self {
        Self {
            id: id.into(),
            task,
        }
    }
}

pub struct InMemoryBenchmark {
    name: String,
    version: String,
    tasks: Vec<InMemoryTask>,
    evaluator: Arc<dyn TaskEvaluator>,
}

impl InMemoryBenchmark {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        tasks: Vec<InMemoryTask>,
        evaluator: Arc<dyn TaskEvaluator>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tasks,
            evaluator,
        }
    }
}

#[async_trait]
impl Benchmark for InMemoryBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn task_count(&self) -> Option<usize> {
        Some(self.tasks.len())
    }

    fn task_ids(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        Box::new(self.tasks.iter().map(|t| t.id.clone()))
    }

    async fn load_task(&self, id: &str) -> Result<LoadedTask, BenchmarkError> {
        let task = self
            .tasks
            .iter()
            .find(|t| t.id == id)
            .ok_or_else(|| BenchmarkError::TaskNotFound(id.to_string()))?
            .task
            .clone();
        Ok(LoadedTask::new(task))
    }

    fn evaluator(&self) -> Arc<dyn TaskEvaluator> {
        self.evaluator.clone()
    }
}

// ======================================================================
// Cheap shared evaluators
// ======================================================================

/// Evaluator that always marks the outcome as pass. Useful for
/// runner-level smoke tests that exercise the plumbing without caring
/// about evaluation logic.
pub struct AlwaysPassEvaluator;

#[async_trait]
impl TaskEvaluator for AlwaysPassEvaluator {
    async fn evaluate(&self, _task: &Task, _outcome: &RunOutcome) -> EvaluationResult {
        EvaluationResult::pass()
    }
}

/// Evaluator that always marks the outcome as fail.
pub struct AlwaysFailEvaluator;

#[async_trait]
impl TaskEvaluator for AlwaysFailEvaluator {
    async fn evaluate(&self, _task: &Task, _outcome: &RunOutcome) -> EvaluationResult {
        EvaluationResult::fail()
    }
}
