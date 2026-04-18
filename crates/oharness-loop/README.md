# oharness-loop

`Agent`, `Loop` trait, and shipped loop implementations for
[open-harness](https://github.com/aishfenton/open-harness) — the
user-facing orchestration layer.

## What's in here

- **`Agent` + `AgentBuilder`** — the top-level "I want to run an
  agent" entry point. Threads LLM, tools, memory, event sink,
  budget, critics, and a chosen `Loop` into a run.
- **`Loop` trait** — a loop owns the turn-taking policy. Shipped:
  - **`ReactLoop`** (feature `react`, default-on) — the classic
    reason-act-observe loop with tool dispatch.
  - **`ConversationLoop`** (feature `conversation`) — alternates
    assistant turns with a `UserSimulator`.
- **`run_reflexion`** (feature `reflexion`) — multi-episode
  wrapper that threads `Reflector`-produced notes into the next
  episode via `ReflectionInjector`.
- **`UserSimulator` trait + `ScriptedUserSimulator` + `LlmUserSimulator`**
  — the user-side model for conversational loops.

## Quickstart — first agent in 10 lines

```rust
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_core::Task;
use std::sync::Arc;

let agent = Agent::builder()
    .with_llm(my_llm)                          // impl Llm
    .with_tools(Arc::new(FsToolSet::new()))
    .with_loop(Box::new(ReactLoop::new()))
    .with_max_turns(10)
    .build()?;

let outcome = agent.run(Task::new("say hello")).await?;
```

See `examples/hello_scripted.rs` for a runnable version. The
`examples/` directory ships 11 runnable examples covering tools,
critics, budgets, replay, reflexion, custom middleware, custom
memory policies, and multi-agent conversation.

## Feature flags

- `react` (default) — `ReactLoop`.
- `conversation` — `ConversationLoop`.
- `reflexion` — `run_reflexion` helper.
- `llm-judge` — forwards to `oharness-critic/llm-judge` so the
  `llm_judge_critic` example builds.

## License

Dual-licensed under MIT or Apache-2.0.
