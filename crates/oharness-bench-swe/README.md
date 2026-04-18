# oharness-bench-swe

SWE-bench (lite + full) adapter for
[open-harness](https://github.com/aishfenton/open-harness).

## What's in here

- **`SweBenchInstance` + `SweBenchLite::from_jsonl(path)`** —
  JSONL dataset loader. Users download the dataset dump (from
  HuggingFace) themselves for now; runtime fetch is a later
  enhancement.
- **`SweBench` (impl `Benchmark`)** — stages each task's repo by
  shelling out to `git clone` + `git checkout` into a per-task
  `Workspace` at `{clone_root}/{instance_id}/`.
- **`SweBenchEvaluator`** (impl `TaskEvaluator`) — applies the
  test patch via `git apply`, runs a configurable test command
  (defaults to `pytest -v --tb=short --no-header`), parses
  `PASSED` / `FAILED` outcomes, and grades against the
  `FAIL_TO_PASS` / `PASS_TO_PASS` id sets. `EvaluationResult.details`
  carries the outcome map + a 4KB tail of raw test output for
  post-run inspection.

## Quickstart

```rust
use oharness_bench_swe::{SweBench, SweBenchEvaluator, SweBenchLite};
use oharness_eval::{run_benchmark, BenchmarkRunConfig};
use std::sync::Arc;

let dataset = SweBenchLite::from_jsonl("swe-bench-lite.jsonl")?;
let bench = SweBench::new(dataset, "/tmp/clones");
let evaluator = Arc::new(SweBenchEvaluator::default());

let results = run_benchmark(
    &bench,
    move |_task| build_agent(),  // provides Llm + tools scoped to task.workspace
    evaluator,
    BenchmarkRunConfig { sample_n: Some(5), ..Default::default() },
).await?;
```

Per-task environment setup (Python venvs, conda envs, Docker)
is configured via the evaluator's `test_command` — see
the crate docs.

## License

Dual-licensed under MIT or Apache-2.0.
