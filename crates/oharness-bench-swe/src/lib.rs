//! SWE-bench adapter — loads the [SWE-bench
//! Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
//! dataset as an [`oharness_eval::Benchmark`] and grades agent runs by
//! applying the per-task test patch, running the project's test
//! command, and checking the `FAIL_TO_PASS` / `PASS_TO_PASS` invariants
//! described by SWE-bench (plan §13.6 / docs/remaining-work.md §3.4).
//!
//! ## What this ships
//!
//! - [`SweBenchInstance`] — deserializes one task's JSON record.
//! - [`SweBenchLite::from_jsonl`] — loads a dataset dump from disk
//!   (one record per line). Users download the dump themselves;
//!   fetching from HuggingFace directly is a later enhancement.
//! - [`SweBenchLite`] implements `Benchmark`. `load_task` shells out
//!   to `git clone` + `git checkout` to stage a per-task
//!   `Workspace` whose `path` is the cloned repo at the task's
//!   `base_commit`.
//! - [`SweBenchEvaluator`] runs `git apply test_patch`, then executes
//!   a test command (default `pytest`, overridable), then parses the
//!   output for `PASSED` / `FAILED` markers and compares against the
//!   `FAIL_TO_PASS` / `PASS_TO_PASS` id sets.
//!
//! ## What this does NOT ship (M2 gate remains open)
//!
//! The M2 completion gate per plan §21.1 is "get ≥5 SWE-bench-lite
//! tasks passing end-to-end with a real LLM". That's an eval
//! *campaign* — not a coding task — because it needs a live LLM with
//! API budget, Python environments per repo (ideally in Docker), and
//! patience for inherent flakiness in some tasks. The adapter here is
//! the prerequisite plumbing; kicking the full run is a follow-up
//! session against a real provider.

pub mod dataset;
pub mod evaluator;
pub mod pytest;

pub use dataset::{SweBenchInstance, SweBenchLite};
pub use evaluator::SweBenchEvaluator;
pub use pytest::{parse_pytest_output, PytestResults};
