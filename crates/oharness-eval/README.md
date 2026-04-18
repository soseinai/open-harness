# oharness-eval

Benchmark runner + `TaskEvaluator` entry points for
[open-harness](https://github.com/aishfenton/open-harness).

## What's in here

- **`TaskEvaluator` re-export** — the trait itself lives in
  `oharness-core` (so both the loop crate and benchmark crates
  can consume it without cycles); this crate re-exports it for
  convenience.
- **`Benchmark` trait** — `load(..) -> LoadedTask` +
  per-task metadata. Implemented by adapters like
  [`oharness-bench-swe`](https://crates.io/crates/oharness-bench-swe).
- **`LoadedTask`** — a `Task` paired with an optional `Workspace`
  (scratch dir for tool-scoped runs).
- **`BenchmarkRunConfig`** — concurrency pools (load + run),
  cost/time caps, filter predicates, sampling.
- **`run_benchmark`** — the top-level driver. Takes a benchmark,
  an agent factory, a task evaluator, and a config; writes a
  results directory (per-task `evaluation.json` + aggregated
  summary).

## Quickstart

```rust
use oharness_eval::{run_benchmark, BenchmarkRunConfig};
use std::sync::Arc;

let results = run_benchmark(
    &my_benchmark,           // impl Benchmark
    move |_task| { build_agent() },  // agent factory
    Arc::new(my_evaluator),  // impl TaskEvaluator
    BenchmarkRunConfig { sample_n: Some(5), ..Default::default() },
).await?;
```

## License

Dual-licensed under MIT or Apache-2.0.
