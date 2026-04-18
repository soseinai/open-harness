# Concepts

*The mental model. Read once, reference as needed.*

This document explains what every piece of open-harness is and
how they fit together. It's the prose counterpart to the plan's
type-definition walkthrough — less detail, more narrative.

## The pipeline

```
  Task ─▶ Agent ─▶ Loop ─▶ RunOutcome
                   │
                   ▼
                 Events ─▶ EventSink ─▶ trajectory.jsonl
```

Read left-to-right: a `Task` goes into an `Agent`, the `Agent`
delegates turn-taking to a `Loop`, and the `Loop` produces a
`RunOutcome`. Along the way, every phase boundary emits an
`Event` to the configured `EventSink`. The event stream is the
primary artifact; the `RunOutcome` is a convenience summary.

## The types you handle

### `Task` (in `oharness-core`)

The thing you want the agent to do. Minimal shape:

```rust
pub struct Task {
    pub instruction: String,
    // + optional id, attachments, metadata
}
```

`Task::new("tell me a joke")` is the common case. Benchmarks
supply richer `Task` values with ids, test specifications, and
reference inputs.

### `Agent` + `AgentBuilder` (in `oharness-loop`)

The top-level "run an agent" entry point. Wires your choices
(LLM, tools, memory, loop, critics, budget, event sink) into a
single runnable unit.

```rust
let agent = Agent::builder()
    .with_llm(Arc::new(my_llm))          // impl Llm
    .with_tools(Arc::new(my_tools))      // impl ToolSet
    .with_loop(Box::new(ReactLoop::new()))// impl Loop
    .with_max_turns(10)
    .build()?;
```

`Agent` is deliberately a **thin orchestrator** — it doesn't
decide turn-taking policy. That's the `Loop`'s job.

### `Loop` (in `oharness-loop`)

The turn-taking policy. Shipped:

- **`ReactLoop`** — classic reason-act-observe. The assistant
  emits `tool_use` blocks; the loop dispatches them via `ToolSet`
  and threads the `tool_result` back. Terminates on `EndTurn` or
  max turns.
- **`ConversationLoop`** — alternates assistant + user turns.
  The user side is a `UserSimulator` (scripted, LLM-driven, or
  custom). Terminates when the simulator emits `EndConversation`.
- **`run_reflexion`** — not a `Loop` impl; a helper that drives a
  `Loop` across multiple episodes, threading `Reflection` notes
  between them.

Writing your own `Loop` is legitimate — it's a one-method trait.
Most users stick with `ReactLoop`.

### `RunOutcome` (in `oharness-core`)

What comes back from `agent.run(task).await`:

```rust
pub struct RunOutcome {
    pub run_id: RunId,
    pub termination: Termination,    // Completed | Truncated | Failed
    pub final_messages: Vec<Message>,
    pub trajectory: TrajectoryHandle,
    pub usage: ResourceUsage,        // tokens, tool calls, …
    // + per-model usage, timestamps, agent state
}
```

The `termination` field tells you why the run stopped
(`EndTurn`, max turns, budget exceeded, critic rejected, …).

## The traits you extend

These are the user-facing extension points. Writing any of them
is how you add behavior to open-harness.

### `Llm` (`oharness-llm`)

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> LlmCapabilities;
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError>;
    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError>;
}
```

Both `complete` and `stream` are required. Providers without
streaming return `Err(LlmError::Unsupported("stream"))`. The
distinction is honest — it lets the loop know what's available
without surprise runtime errors.

The shipped providers (`AnthropicLlm`, `OpenAiLlm`, etc.) are
canonical references.

### `ToolSet` (`oharness-tools`)

```rust
#[async_trait]
pub trait ToolSet: Send + Sync {
    fn specs(&self) -> &[ToolSpec];
    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutcome;
}
```

`specs()` returns the tool metadata the LLM sees in
`CompletionRequest.tools` (name, description, input JSON schema).
`execute()` runs a specific tool by name with the LLM-supplied
input.

`ToolContext` threads cross-cutting concerns: `EventSink`,
`BudgetHandle`, `Cancellation`, `ApprovalChannel`, `Workspace`.
Tools use these when they matter; most read just
`ctx.workspace_path()` and call it a day.

### `Critic` (`oharness-critic`)

```rust
#[async_trait]
pub trait Critic: Send + Sync {
    fn name(&self) -> &str;
    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict;
}
```

A critic inspects the most recent assistant turn and emits:
- **`Accept`** / **`AcceptWithNote(..)`** — continue.
- **`Reject { reason }`** — terminate the run.
- **`Revise { replacement, reason }`** — swap the turn in place;
  the loop continues. Capped at `revision_depth_cap` (default 3).
- **`Abort { reason }`** — hard stop with an error.

Multiple critics compose via `CompositeCritic` +
`AggregationPolicy` (`FirstReject`, `AllMustAccept`,
`MajorityVote`, `Weighted`).

### `MemoryPolicy` (`oharness-memory`)

```rust
#[async_trait]
pub trait MemoryPolicy: Send + Sync {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError>;
}
```

Sits between the full conversation state and what the LLM sees
on the next turn. Shipped: `Passthrough`, `TruncateAfterTokens`,
`ElideToolResults`. Writing your own is a ~20-line impl — see
`custom_memory_policy.rs`.

### `Reflector` (`oharness-critic`)

```rust
#[async_trait]
pub trait Reflector: Send + Sync {
    fn name(&self) -> &str;
    async fn reflect(&self, episode: &Episode<'_>) -> Option<Reflection>;
}
```

Runs between `run_reflexion` episodes. Given an `Episode` (prior
run + evaluation + prior reflections), produces an optional
`Reflection` that feeds the next episode's system prompt via
`ReflectionInjector`.

### `UserSimulator` (`oharness-loop`)

```rust
#[async_trait]
pub trait UserSimulator: Send + Sync {
    fn name(&self) -> &str;
    async fn initial_message(&self, task: &Task) -> Result<String, UserError>;
    async fn respond(&self, conversation: ConversationView<'_>, task: &Task)
        -> Result<UserAction, UserError>;
}
```

Drives the user side of a `ConversationLoop`. Shipped:
`ScriptedUserSimulator` (pre-written script) and
`LlmUserSimulator` (persona + prompt template on a judge LLM).

### `TaskEvaluator` (`oharness-core`)

```rust
#[async_trait]
pub trait TaskEvaluator: Send + Sync {
    async fn evaluate(&self, task: &Task, outcome: &RunOutcome) -> EvaluationResult;
}
```

Scores a completed run. Used by `oharness-eval`'s
`run_benchmark` and by `run_reflexion`. Lives in core (not in
the eval crate) to dodge loop → eval → loop dep cycles.

## Middleware composition (`oharness-llm`)

The `Llm` trait has three middleware shapes you can wrap around
a provider. Each wrapper itself implements `Llm`, so the whole
chain is a drop-in replacement:

- **`RequestLayer`** — sync, mutate `CompletionRequest` in place.
  E.g., stamp a request-id, inject metadata, redact PII from
  the outgoing prompt.
- **`ResponseLayer`** — sync, mutate `CompletionResponse` in
  place. E.g., strip secrets from the response text. Has a
  `stream_mode()` hook that decides behaviour when wrapped
  around `stream()`: `WarnAndSkip` (default), `Error`, or
  `SilentSkip`.
- **`FullLayer`** — async, wrap the whole `complete()` / `stream()`
  call via `BoxFuture`. E.g., timing, retries, caching, rate
  limiting. Two methods (`around_complete` + `around_stream`)
  because they have different retry semantics.

Composition via `LlmExt`:

```rust
let llm = provider
    .with_request_layer(stamp)
    .with_response_layer(redactor)
    .with_full_layer(timer);
```

The canonical references: `BudgetMiddleware` in
`oharness-budget`, `RequestTracer` / `StreamTracer` / `ToolTracer`
in `oharness-trace`, `PromptCaching` in `oharness-providers`.
See `custom_middleware.rs` for a from-scratch example.

## Events + trajectory

Every phase boundary emits an `Event`. The v1.0 schema has 36
`EventKind` variants; the machine-readable spec lives at
`crates/oharness-core/schema/events-v1.0.json`.

```rust
pub struct Event {
    pub v: SchemaVersion,        // "1.0"
    pub seq: u64,                // monotonic within run
    pub run_id: RunId,           // UUID
    pub timestamp: OffsetDateTime,
    pub span_id: SpanId,         // open/close pair identifier
    pub parent: Option<SpanId>,  // nested spans
    pub kind: EventKind,         // tagged payload
    pub redactions: Vec<Redaction>,
}
```

### Where events come from

- **Lifecycle**: `meta`, `run.{started,finished}`,
  `turn.{started,finished,revised}`. Emitted by the `Loop`.
- **LLM**: `llm.{request,response,retry,failed}`,
  `llm.stream.chunk`. Emitted by `RequestTracer` /
  `StreamTracer` middleware (wired by `Agent::run`).
- **Tools**: `tool.call.{started,finished,failed}`,
  `tool.approval.{requested,decided}`. Emitted by `ToolTracer`.
- **Memory**: `memory.{evicted,summarized,retrieved}`. Emitted
  by memory policies when they mangle the view.
- **Budget**: `budget.exceeded`. Emitted by budget middleware.
- **Critic**: `critic.{assessed,rejected,revised,failed}`.
- **Reflection**: `reflection.{generated,injected}`.
- **User**: `human.{interrupt,inject}`,
  `user.simulated.{message,ended}`, `user.log`.

### `EventSink` (`oharness-core`)

The abstraction every crate consumes:

```rust
pub trait EventSink: Send + Sync {
    fn emit(&self, event: Event);
    fn try_emit(&self, event: Event) -> Result<(), Event>;
}
```

Shipped implementations (in `oharness-trace`):

- **`InMemorySink`** — `Vec<Event>` buffer. Tests, small demos.
- **`FileSink`** — async JSONL writer backed by a bounded mpsc
  channel. The production sink. `sink.flush().await` drains
  before the program exits.
- **`FanOutSink`** — forward to multiple sinks. `Agent::run`
  wires this automatically so every run populates
  `RunOutcome.trajectory` regardless of the user's choice.

Schema governance (plan §19): any change to `EventKind` or its
payloads requires a `CHANGELOG-schema.md` entry and a
`SchemaVersion::CURRENT` bump. CI verifies the committed
`events-v1.0.json` matches a fresh export on every build (`just
schema-check`).

## Budgets (`oharness-budget`)

`BudgetHandle` is the trait; `BudgetMiddleware` wraps an `Llm`
to enforce it. Shipped handles:

- `TokenBudget` — input / output / input+output caps.
- `StepBudget` — number of `complete()` calls.
- `CostBudget` — USD, via a `PricingTable`.
- `TimeBudget` — wall-clock.
- `CompositeBudget` — any child trip trips the composite.

When a cap trips, `BudgetMiddleware::complete()` returns
`LlmError::Provider(BudgetExceeded)`, which the loop converts
to `Termination::Failed { category: Llm }`. The budget handle's
snapshot API gives per-task telemetry independent of the
trajectory events.

## Benchmarks (`oharness-eval` + adapters)

`Benchmark` trait + `run_benchmark` driver. A `Benchmark` loads
tasks (with optional per-task `Workspace` scratch dirs) and
hands them to an agent factory; the runner evaluates each with
a `TaskEvaluator` and writes a results directory.

Concrete adapter: `oharness-bench-swe` (SWE-bench lite + full).
τ-bench, GAIA, and others are user-land crates that implement
the `Benchmark` trait.

## The crate DAG

One-way top-to-bottom:

```
oharness-core       (zero IO; everyone depends on this)
     ↓
oharness-llm ────── oharness-tools ──── oharness-memory
     │                  │                    │
     ├─ oharness-providers                   │
     ├─ oharness-trace ──────────────────────┤
     ├─ oharness-budget                      │
     └─ oharness-critic                      │
           │                                 │
           └───────────────► oharness-loop ◄─┘
                                  │
                                  ├─ oharness-eval
                                  │     │
                                  │     └─ oharness-bench-swe
                                  │
                                  └─ oharness-py (via PyO3)
```

Plan §3.1 is the authoritative dependency rule set; summarised:
no crate above the line imports anything below it.

## What's next to read

- [`docs/philosophy.md`](philosophy.md) — why the library is shaped this way.
- [`docs/quickstart.md`](quickstart.md) — first agent in 5 minutes.
- [`docs/security.md`](security.md) — trust model + deployment guidance.
- [`docs/open-harness-plan.md`](open-harness-plan.md) — the full v1 spec (~1900 lines, locked 2026-04-17).
- Per-crate `README.md` files — crate-level reference for each published crate.
