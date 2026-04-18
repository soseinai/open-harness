# oharness-trace

Event sinks, trajectory writer/reader, and `ReplayLlm` for
[open-harness](https://github.com/aishfenton/open-harness).

## What's in here

- **Event sinks**:
  - `InMemorySink` — captures events into a `Vec<Event>`; useful
    for tests and small demos.
  - `FileSink` — async JSONL writer backed by a bounded mpsc
    channel; the production-grade sink for long runs. Use
    `flush().await` to drain cleanly.
  - `FanOutSink` — forward every event to multiple sinks. The
    `Agent` uses this internally to keep both the user's sink and
    an internal `InMemorySink` for `RunOutcome.trajectory`.
- **JSONL reader** — `read_events(path)` streams events back out
  for post-run analysis.
- **Tracing middleware** — `RequestTracer`, `StreamTracer`,
  `ToolTracer` emit the canonical `llm.request` / `llm.response`
  / `tool.call.*` events. `Agent::run` wires these automatically.
- **`ReplayLlm`** — re-drive an agent against a recorded
  trajectory. `ReplayMode::Positional` (default) pairs the Nth
  live request with the Nth recorded response; `::Strict` adds
  canonical-JSON input comparison with `DriftPolicy::{WarnAndContinue, Fail}`.

## Quickstart — record and replay

```rust
use oharness_trace::{FileSink, ReplayLlm, ReplayMode, DriftPolicy};

let sink = Arc::new(FileSink::to_path("run.jsonl").await?);
// ... build agent with `.with_event_sink(sink.clone())` and run ...
sink.flush().await?;

let replay = ReplayLlm::from_path("run.jsonl", ReplayMode::Positional, DriftPolicy::default()).await?;
// Drop `replay` into a fresh agent — same inputs, same outputs, no API call.
```

See the `replay_trajectory` example in `oharness-loop/examples/`.

## License

Dual-licensed under MIT or Apache-2.0.
