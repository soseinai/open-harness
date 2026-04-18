# open-harness

**Kernel-style research framework for LLM agent loops.** Pairs
with `lm-eval-harness` — small, sharp extension points rather
than a broad LangChain-style integration surface.

Targets agent researchers publishing new techniques, eval/safety
teams running agents on benchmarks (SWE-bench, τ-bench), and
practitioners shipping production agents. If you want a batteries
-included "I'll wire up 50 tools for you" framework, this is not
that library.

## What's shipped

| Area                     | Status                                                                 |
|--------------------------|------------------------------------------------------------------------|
| Core event schema (v1.0) | ✅ JSON-Schema exported, CI-checked for drift                          |
| Providers                | ✅ Anthropic, OpenAI, OpenRouter, Ollama, vLLM                         |
| Streaming                | ✅ SSE-based chunk streams                                             |
| Middleware               | ✅ `RequestLayer`, `ResponseLayer`, `FullLayer`, `ChunkObserver/Transformer` |
| Budgets                  | ✅ Token, step, cost, time, composite                                   |
| Memory policies          | ✅ Passthrough, truncate-after-tokens, elide-tool-results              |
| Critics + Reflexion      | ✅ Trait + composites; shipped `LlmJudgeCritic`, `TestCritic`, `RegexDenyCritic`; `run_reflexion` helper |
| Conversation loops       | ✅ `ConversationLoop` + scripted / LLM-driven user simulators          |
| Replay                   | ✅ `ReplayLlm` (positional + strict modes, drift policies)             |
| Benchmarks               | ✅ `Benchmark` trait + `oharness-bench-swe` adapter                    |
| Python bindings          | ✅ `oharness` wheel via `maturin` — 9 of 10 plan §14.2 traits live     |
| Examples                 | ✅ 11 runnable examples in `oharness-loop/examples/` (target: 15)      |
| Docs                     | 🟡 Front-door complete; per-subsystem docs in progress                 |

Full milestone plan: [`docs/remaining-work.md`](docs/remaining-work.md).
Design spec (locked 2026-04-17): [`docs/open-harness-plan.md`](docs/open-harness-plan.md).

## Hello, agent — in 10 lines

```rust
use oharness_core::Task;
use oharness_loop::{Agent, ReactLoop};
use oharness_providers::AnthropicLlm;
use oharness_tools::fs::FsToolSet;
use std::sync::Arc;

let agent = Agent::builder()
    .with_llm(Arc::new(AnthropicLlm::from_env()?))  // ANTHROPIC_API_KEY
    .with_tools(Arc::new(FsToolSet::new()))
    .with_loop(Box::new(ReactLoop::new()))
    .with_max_turns(10)
    .build()?;

let outcome = agent.run(Task::new("list what's in the current directory")).await?;
println!("{:?}", outcome.termination);
```

No real API key? Swap `AnthropicLlm::from_env()?` for a scripted
`Llm` — see `crates/oharness-loop/examples/hello_scripted.rs`.

## Docs

- **[`docs/quickstart.md`](docs/quickstart.md)** — first agent in 5 minutes.
- **[`docs/concepts.md`](docs/concepts.md)** — mental model (Task → Agent → Loop → Event).
- **[`docs/philosophy.md`](docs/philosophy.md)** — design principles, non-goals.
- **[`docs/security.md`](docs/security.md)** — trust model + deployment guidance.
- **[`docs/pricing.md`](docs/pricing.md)** — maintaining the pricing table.
- **[`RELEASE.md`](RELEASE.md)** — maintainer release procedure.
- **[`CHANGELOG.md`](CHANGELOG.md)** — user-facing change log.
- **[`CHANGELOG-schema.md`](CHANGELOG-schema.md)** — event-schema governance.

Each publishable crate also has its own `README.md` rendered on crates.io.

## Workspace layout

```
oharness-core       pure types, event schema, context traits; serde only
oharness-llm        Llm trait + middleware helper traits
oharness-providers  Anthropic / OpenAI / OpenRouter / Ollama / vLLM adapters
oharness-tools      ToolSet trait + bash + fs tool kits
oharness-memory     Passthrough / TruncateAfterTokens / ElideToolResults
oharness-trace      EventSink impls + FileSink + InMemorySink + ReplayLlm
oharness-budget     Token / Step / Cost / Time / Composite budgets
oharness-critic     Critic / Reflector + CompositeCritic + shipped impls
oharness-loop       Agent + ReactLoop + ConversationLoop + run_reflexion
oharness-eval       Benchmark trait + run_benchmark driver
oharness-bench-swe  SWE-bench (lite + full) adapter
oharness-py         Python bindings (maturin; ships to PyPI as `oharness`)
```

Dependency DAG is one-way top-to-bottom — plan §3.1.

## Examples

`cargo run --example <name> -p oharness-loop` runs any of the 11
shipped examples. All use a scripted `Llm` so they work without
API keys. `just examples` smoke-runs them in CI.

| Example                      | Covers                                              |
|------------------------------|-----------------------------------------------------|
| `hello_scripted`             | Minimum viable agent (one turn, no tools)           |
| `react_with_tools`           | Multi-turn ReAct with real tool dispatch            |
| `custom_critic`              | Implement `Critic` from scratch; `Reject` verdict   |
| `self_refine`                | Critic `Revise` → in-place turn rewrite             |
| `llm_judge_critic`           | Shipped `LlmJudgeCritic` + SCORE-based threshold    |
| `budget_enforcement`         | `BudgetMiddleware` + tight token cap                |
| `custom_middleware`          | `RequestLayer` + `ResponseLayer` + `FullLayer`      |
| `custom_memory_policy`       | Implement `MemoryPolicy` from scratch               |
| `replay_trajectory`          | Record → JSONL → `ReplayLlm` round-trip             |
| `reflexion_run`              | `run_reflexion` with a `NudgeReflector`             |
| `multi_agent_conversation`   | `ConversationLoop` + `ScriptedUserSimulator`        |

## Requirements

- Rust 1.82+ (stable).
- Tokio runtime (the framework is tokio-only — plan §0.4).
- For `oharness-py`: Python 3.10+ and `maturin`; see
  `crates/oharness-py/README.md`.

## Development

```bash
just ci            # fmt-check + clippy + test + examples + schema-check
just examples      # run all shipped examples
just schema-export # regenerate the JSON Schema baseline
just python-check  # opt-in cargo check for the pyo3 crate
```

## License

Dual-licensed under **MIT** or **Apache-2.0** at your option. See
`LICENSE-MIT` and `LICENSE-APACHE` at the workspace root.
