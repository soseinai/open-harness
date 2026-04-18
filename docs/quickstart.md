# Quickstart

*Your first agent in 5 minutes.*

Prerequisites: Rust 1.82+. No API key needed for §1–2; §3 uses
Anthropic (you can substitute any provider the library supports).

## 1. Add the dependencies

```toml
[dependencies]
oharness-core = "0.1"
oharness-loop = "0.1"
oharness-tools = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
async-trait = "0.1"
```

(Once v1.0 ships to crates.io. Until then, use
`git = "https://github.com/aishfenton/open-harness"` path deps.)

## 2. Run an agent with no API key

Every example in this library starts with a **scripted `Llm`** —
a one-file stub that returns a canned response. This lets you
exercise the whole loop without any real provider wiring. Here's
the smallest possible run:

```rust
use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId,
    StopReason, Task, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use std::sync::Arc;

/// A scripted `Llm` that returns one canned response.
struct HelloLlm;

#[async_trait]
impl Llm for HelloLlm {
    fn name(&self) -> &str { "hello" }
    fn capabilities(&self) -> LlmCapabilities { LlmCapabilities::default() }

    async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "m1".into(),
            model: ModelId::new("hello-example"),
            content: vec![Content::text("Hi!")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    async fn stream(&self, _: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::builder()
        .with_llm(Arc::new(HelloLlm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()?;

    let outcome = agent.run(Task::new("say hello")).await?;
    println!("Termination: {:?}", outcome.termination);
    Ok(())
}
```

Run it:

```bash
cargo run
```

You should see `Termination: Completed { reason: EndTurn }`.
That's a complete agent loop: LLM → tool availability
negotiation → turn-taking → termination decision → event
capture.

The runnable version of this pattern lives at
`crates/oharness-loop/examples/hello_scripted.rs`.

## 3. Swap in a real provider

Once the scripted version runs, swap the LLM for a real one:

```toml
[dependencies]
oharness-providers = { version = "0.1", features = ["anthropic"] }
```

```rust
use oharness_providers::AnthropicLlm;

let llm = Arc::new(AnthropicLlm::from_env()?);  // reads ANTHROPIC_API_KEY
let agent = Agent::builder()
    .with_llm(llm)
    .with_tools(Arc::new(FsToolSet::new()))
    .with_loop(Box::new(ReactLoop::new()))
    .with_max_turns(10)
    .build()?;

let outcome = agent
    .run(Task::new("list what's in the current directory"))
    .await?;
```

Everything else is identical. Same loop, same tools, same
`RunOutcome` shape. Providers are fully interchangeable — that's
the point of the `Llm` trait.

Other shipped adapters (each behind its own feature flag):

```toml
oharness-providers = { version = "0.1", features = ["openai"] }
# Also: "openrouter", "ollama", "vllm"
```

## 4. Capture a trajectory

The run already captured every event — it's in
`outcome.trajectory`. To persist it to disk:

```rust
use oharness_trace::FileSink;

let sink = Arc::new(FileSink::to_path("run.jsonl").await?);
let agent = Agent::builder()
    .with_llm(llm)
    .with_tools(Arc::new(FsToolSet::new()))
    .with_event_sink(sink.clone())     // <-- add this
    .with_loop(Box::new(ReactLoop::new()))
    .build()?;

let outcome = agent.run(Task::new("…")).await?;
sink.flush().await?;  // drain the writer task
```

The file at `run.jsonl` is a line-per-event JSONL document — open
it in `jq`, `less`, or a paper-supplement analysis script.

## 5. Replay the trajectory

Once recorded, a trajectory re-drives an agent without the
original provider:

```rust
use oharness_trace::{DriftPolicy, ReplayLlm, ReplayMode};

let replay = ReplayLlm::from_path(
    "run.jsonl",
    ReplayMode::Positional,
    DriftPolicy::default(),
).await?;

let replay_agent = Agent::builder()
    .with_llm(Arc::new(replay))
    .with_tools(Arc::new(FsToolSet::new()))
    .with_loop(Box::new(ReactLoop::new()))
    .build()?;

let replay_outcome = replay_agent.run(Task::new("…")).await?;
// replay_outcome.final_messages == outcome.final_messages
```

`ReplayMode::Positional` pairs the Nth live `llm.request` with
the Nth recorded `llm.response` — no byte-for-byte input
comparison. Use `::Strict` + `DriftPolicy::Fail` when you need
canonical equality (e.g., regression tests of your prompt
construction).

## 6. Where to go next

You now have the core loop. The next layers — tools with real
side effects, critics that block bad turns, budgets that cap
cost, memory policies that manage context, reflexion that
iterates across episodes — all compose the same way.

The 11 runnable examples in `crates/oharness-loop/examples/`
each focus on one of these. Start with:

- **`react_with_tools`** — multi-turn ReAct with actual tool dispatch.
- **`custom_critic`** — your own `Critic` implementation.
- **`budget_enforcement`** — cap a run at N tokens.
- **`custom_middleware`** — `RequestLayer` + `ResponseLayer` + `FullLayer` composed.
- **`reflexion_run`** — multi-episode learning from prior attempts.

All use scripted LLMs so they run without API keys. `cargo run
--example <name> -p oharness-loop`.

For the full mental model of what every piece is and how they
compose, read [`docs/concepts.md`](concepts.md). For the design
principles behind those pieces, read
[`docs/philosophy.md`](philosophy.md).
