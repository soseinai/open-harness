//! Benchmark runner for open-harness (plan §13).
//!
//! - [`TaskEvaluator`](oharness_core::TaskEvaluator) re-exported from
//!   core — benchmark adapters implement it directly; `run_reflexion`
//!   in `oharness-loop` consumes the same trait, so the two round-trip
//!   without adapters.
//! - [`Benchmark`] trait + [`LoadedTask`] + [`BenchmarkError`] — the
//!   shape an adapter crate (e.g. `oharness-bench-swe`) implements.
//! - [`BenchmarkRunConfig`] with concurrency / filter / sample / shard
//!   / resume knobs.
//! - [`run_benchmark`] — the runner. Async from day one (plan §13.4)
//!   because real factories pull auth / build middleware stacks /
//!   touch disk.
//! - [`BenchmarkReport`] with aggregate statistics.
//! - [`in_memory`] — an `InMemoryBenchmark` fixture for tests and
//!   harness-on-harness smoke runs.
//!
//! [`Workspace`](oharness_tools::context::Workspace) is re-exported from
//! `oharness-tools` rather than re-defined here; it was already built
//! there for M1a tool scoping and owns its own cleanup machinery.

pub mod benchmark;
pub mod config;
pub mod in_memory;
pub mod results;
pub mod runner;

pub use benchmark::{Benchmark, BenchmarkError, LoadedTask};
pub use config::{BenchmarkRunConfig, Shard};
pub use in_memory::{AlwaysFailEvaluator, AlwaysPassEvaluator, InMemoryBenchmark, InMemoryTask};
pub use results::{BenchmarkReport, TaskReport};
pub use runner::run_benchmark;

pub use oharness_core::{EvaluationResult, TaskEvaluator};
pub use oharness_tools::context::Workspace;
