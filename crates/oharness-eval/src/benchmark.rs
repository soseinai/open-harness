//! [`Benchmark`] trait and supporting types (plan §13.2).
//!
//! A benchmark is a named collection of tasks paired with an evaluator.
//! It owns:
//!
//! - a stable *name* and *version* — used in results directory manifests
//!   so runs against different dataset revisions don't quietly overwrite
//!   each other.
//! - a cheap *id enumeration* ([`Benchmark::task_ids`]) — enumerates
//!   without touching network / disk beyond listing the dataset.
//! - an expensive *load* step ([`Benchmark::load_task`]) — does the
//!   real I/O (git clone, container setup, workspace staging).
//! - an *evaluator* — shared across all tasks in the benchmark.

use async_trait::async_trait;
use oharness_core::{Task, TaskEvaluator};
use oharness_tools::context::Workspace;
use std::sync::Arc;

#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Stable identifier of the benchmark itself — e.g.
    /// `"swe-bench-lite"`, `"gaia"`. Used as a directory prefix and on
    /// reports.
    fn name(&self) -> &str;

    /// Dataset version the adapter is currently pointing at —
    /// e.g. `"swe-bench-lite-1.2.1"`. A bumped version invalidates
    /// resume-from-disk runs against older snapshots (the manifest
    /// records this).
    fn version(&self) -> &str;

    /// Total task count if cheaply knowable, else `None`. Used by the
    /// runner for progress reporting only.
    fn task_count(&self) -> Option<usize>;

    /// Enumerate every task id in dataset order. Must be cheap — no
    /// I/O beyond listing the dataset is allowed here.
    fn task_ids(&self) -> Box<dyn Iterator<Item = String> + Send + '_>;

    /// Load the expensive per-task setup: fetch data, clone the repo,
    /// stage a workspace. Errors here surface as skipped tasks in the
    /// report rather than aborting the run.
    async fn load_task(&self, id: &str) -> Result<LoadedTask, BenchmarkError>;

    /// The shared evaluator used to grade every task in this
    /// benchmark. Returned as `Arc` so the runner can clone it across
    /// worker tasks without re-instantiation.
    fn evaluator(&self) -> Arc<dyn TaskEvaluator>;
}

/// A task loaded and ready to run. Holds an optional scratch
/// [`Workspace`] the agent's tools can latch onto — benchmark adapters
/// that need a per-task filesystem (SWE-bench, MBPP) populate it; pure
/// prompt-response benchmarks leave it `None`.
pub struct LoadedTask {
    pub task: Task,
    pub workspace: Option<Arc<Workspace>>,
}

impl LoadedTask {
    pub fn new(task: Task) -> Self {
        Self {
            task,
            workspace: None,
        }
    }

    pub fn with_workspace(mut self, workspace: Arc<Workspace>) -> Self {
        self.workspace = Some(workspace);
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("benchmark I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("benchmark decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("benchmark: task `{0}` not found")]
    TaskNotFound(String),
    #[error("benchmark: {0}")]
    Other(String),
}
