# open-harness — Implementation Plan

**Status: LOCKED (v1 — 2026-04-17).** Design-frozen spec. Went through three rounds of critical review; all blocking and structural issues resolved. Further changes should go through a revision PR rather than ad-hoc edits, so reviewers can see what's being reopened and why.

Comprehensive specification capturing all design decisions for the `open-harness` research framework.

---

## 1. Project identity

| Property | Value |
|---|---|
| Name | `open-harness` |
| Crate prefix | `oharness-` (e.g. `oharness-core`, `oharness-llm`) |
| Python package | `oharness` |
| CLI binary | `openh` (verbose `open-harness` also accepted) |
| Env var prefix | `OHARNESS_` |
| License | Dual MIT / Apache-2.0 |
| Schema version at launch | `1.0` |

## 2. Philosophy

**"open-harness"** = scaffolding between LLM and task. Pairs with `lm-eval-harness`. Target audiences (priority order):

1. **Agent researchers** publishing new techniques (Reflexion++, novel memory, planners)
2. **Eval/safety teams** running agents on benchmarks (SWE-bench, τ-bench, GAIA)
3. **Practitioners** shipping production agents (Tower for agents)

Explicitly NOT: LangChain/LlamaIndex-style broad integration surface. A **kernel** — small core, sharp extension points.

### Design principles

- **Small kernel, big periphery.** `Llm` + `ToolSet` + `Loop` stay tiny. Everything else composes.
- **Data-oriented boundaries.** Every phase boundary emits a serializable event.
- **Composition over configuration.** Middleware stacks, not knob explosions.
- **No surprise orchestration.** Library never picks algorithms; defaults are minimal.
- **Deterministic when possible, instrumented always.** Record/replay first-class.
- **Rust core, polyglot surface.** Python bindings non-negotiable.
- **Provider honesty.** Don't flatten provider capabilities into lossy common denominator.
- **Fail loud, not silent.** Construction-time errors for incompatibilities; no silent no-ops.

---

## 3. Crate layout

Cargo workspace. **One-way dependency DAG** — no crate above the line imports anything below it.

```
oharness-core           # pure types, event schema, context traits; serde only
oharness-llm            # Llm trait + middleware helper traits + complete_from_stream
oharness-providers      # feature-gated provider adapters
oharness-tools          # ToolSet trait + contributed tool kits
oharness-memory         # pluggable memory/context strategies
oharness-trace          # EventSink implementations, trajectory writer/reader, replayer, tracing middleware
oharness-budget         # BudgetHandle implementations + enforcement middleware
oharness-critic         # Critic + Reflector traits + composites
oharness-loop           # Agent + Loop trait + ReactLoop + ConversationLoop + ReflexionLoop
oharness-eval           # Benchmark trait + runner + benchmark adapters
oharness-py             # PyO3 bindings (optional)
oharness-cli            # binary: openh
```

**Context-plumbing traits live in `oharness-core`**: `EventSink`, `ApprovalChannel`, `BudgetHandle`, and the cancellation wrapper. *Implementations* (file sinks, budget trackers, CLI approval prompts) live in their respective crates. This keeps the DAG clean: every crate that needs to thread these references depends only on core.

### Dependency rules

- `oharness-core` depends only on `serde`, `thiserror`, `uuid`, `time`, `tokio-util` (for `CancellationToken`) (no IO-doing code, but trait defs referencing async are fine)
- `oharness-llm` depends on `oharness-core` + `reqwest`, `tokio`, `futures`
- `oharness-tools` depends on `oharness-core` only — uses the context traits from core, not from trace/budget
- `oharness-memory` depends on `oharness-core` + `oharness-llm` (`MemoryPolicy::Summarize` calls an `Llm`)
- `oharness-trace` depends on `oharness-core` + `oharness-llm` (tracing middleware wraps `Llm`)
- `oharness-budget` depends on `oharness-core` + `oharness-llm` (budget middleware wraps `Llm`)
- `oharness-critic` depends on `oharness-core` + `oharness-llm`
- `oharness-loop` depends on `oharness-core`, `oharness-llm`, `oharness-tools`, optionally `oharness-memory`, `oharness-critic`, `oharness-budget`, `oharness-trace`
- `oharness-eval` depends on `oharness-loop` + everything

---

## 4. `oharness-core`

The zero-IO foundation. Everyone depends on this.

### 4.1 Identifiers

```rust
pub struct RunId(pub Uuid);
pub struct SpanId(pub String);      // e.g., "llm-0", "tool-3"
pub struct ModelId(pub String);     // e.g., "anthropic/claude-opus-4-7"
```

### 4.2 Messages & content

```rust
pub enum Message {
    System    { content: String, meta: Map<String, Value> },
    User      { content: Vec<Content>, meta: Map<String, Value> },
    Assistant { content: Vec<Content>, stop_reason: Option<StopReason>,
                meta: Map<String, Value> },
}

pub enum Content {
    Text(String),
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, output: ToolOutput, is_error: bool },
    Thinking(String),               // extended thinking blocks
    Image(ImageRef),
    Document(DocumentRef),
    Audio(AudioRef),
    Citation(CitationRef),
}

pub struct ToolOutput {
    pub content: Vec<Content>,      // tools can return text + images + more
    pub truncated: bool,
}
```

`meta: Map<String, Value>` on every message and the `extensions` pattern on content references is the **universal research annotation hook** — attach per-block metadata without forking the enum.

### 4.3 Task

```rust
pub struct Task {
    pub id: Option<String>,
    pub instruction: String,
    pub attachments: Vec<Attachment>,
    pub metadata: Map<String, Value>,     // namespaced keys: reverse-DNS convention
}

pub enum Attachment {
    Text   { name: String, content: String },
    File   { name: String, path: PathBuf },
    Inline { name: String, bytes: Vec<u8>, mime: String },
    Url    { url: Url, mime_hint: Option<String> },
}
```

Task is **pure data**. No behavior, no predicates, no closures. Serde-serializable. Success predicates live in `TaskEvaluator`.

### 4.4 RunOutcome

```rust
pub struct RunOutcome {
    pub run_id: RunId,
    pub task_id: Option<String>,

    pub termination: Termination,
    pub final_messages: Vec<Message>,         // LLM's view of conversation

    pub trajectory: TrajectoryHandle,         // lazy event stream
    pub usage: ResourceUsage,
    pub per_model_usage: Map<ModelId, ResourceUsage>,

    pub started_at: SystemTime,
    pub finished_at: SystemTime,

    pub agent_state: Map<String, Value>,      // opaque, agent-specific
}

pub enum Termination {
    Completed  { reason: CompletionReason },     // EndTurn, StopSequence
    Truncated  { limit: TruncationLimit },       // MaxTurns, Budget, Timeout
    Failed     { error: RunError, at_turn: u32 },
    Interrupted{ reason: InterruptionReason },   // User, ApprovalDenied, Cancellation
}

pub struct ResourceUsage {
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub tokens_cache_read: u64,
    pub tokens_cache_create: u64,
    pub cost_usd: Option<f64>,
    pub wall_clock: Duration,
    pub turns: u32,
    pub tool_calls: u32,
}
```

**Rules:**
- Mid-run failure returns `Ok(RunOutcome)` with `Termination::Failed`. `AgentError` (the outer `Result::Err`) is for "couldn't start."
- No derived convenience fields beyond `usage`. Use `trajectory` queries instead.
- `TrajectoryHandle` is a reference — serializes as path/URI, not inlined events.

### 4.5 Capabilities

```rust
pub struct LlmCapabilities {
    pub streaming: bool,
    pub prompt_caching: bool,
    pub parallel_tool_use: bool,
    pub vision: bool,
    pub thinking: bool,
    pub structured_output: bool,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
}
```

Returned **by value** from `Llm::capabilities()`.

### 4.6 Context-plumbing traits

Traits that thread through every subsystem. Defined here so `oharness-tools`, `oharness-memory`, `oharness-loop`, etc. can depend only on `oharness-core`. Concrete implementations live in the appropriate downstream crate.

```rust
/// Emit events to wherever the harness routes them.
/// Implementations provide BOTH methods explicitly — no defaults — because their
/// semantics are fundamentally different (one blocks, one returns). The split
/// exists so `try_emit` never silently blocks.
pub trait EventSink: Send + Sync {
    /// Blocking-as-specified emit. Behavior on a full channel:
    /// - Default shipped sinks: `try_send` first; on `Full`, the sink spawns a
    ///   `spawn_blocking` task that blocks on `send`. This preserves the
    ///   "block on backpressure" invariant without stalling a tokio worker
    ///   thread — the `spawn_blocking` call moves the block to the blocking pool.
    /// - `NullSink`: discards unconditionally.
    /// - Call from `Drop`: callers SHOULD use `try_emit` instead; `emit` from
    ///   a `Drop` is legal but risks blocking shutdown.
    fn emit(&self, event: Event);

    /// Non-blocking variant. Returns `Err(event)` immediately if the channel
    /// is full so the caller can decide policy (drop, buffer locally, log).
    /// Never blocks, never spawns.
    fn try_emit(&self, event: Event) -> Result<(), Event>;
}

/// Budget state accessible from tools and middleware.
#[async_trait]
pub trait BudgetHandle: Send + Sync {
    async fn check(&self, request: BudgetRequest) -> BudgetDecision;
    async fn consume(&self, amount: BudgetAmount);
    fn snapshot(&self) -> BudgetSnapshot;
}

/// Approval channel for tool calls needing human/external OK.
#[async_trait]
pub trait ApprovalChannel: Send + Sync {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse;
}

pub struct ApprovalRequest {
    pub tool_name: String,
    pub input: Value,
    pub reason: String,
}

pub enum ApprovalResponse {
    Allow,
    Deny(String),
}

/// Re-exports `tokio_util::sync::CancellationToken` as `oharness_core::Cancellation`
/// to keep a stable import path even if we swap the underlying primitive later.
pub type Cancellation = tokio_util::sync::CancellationToken;
```

**Why `EventSink::emit` is sync (not async):** emission must never await. Implementations push onto a bounded `mpsc` channel; a dedicated writer task drains. Synchronous `emit` means tools/memory/middleware can emit from non-async contexts (e.g., from `Drop`).

**How backpressure is handled without stalling the tokio runtime:** a naive blocking `send` from an async context could park a tokio worker thread, deadlocking other tasks. Shipped sinks instead: (1) call `try_send` first; (2) if `Full`, use `tokio::task::spawn_blocking` to move the blocking `send` onto the blocking thread pool. The emitter still blocks — preserving the "block on backpressure" invariant — but the block is on a blocking-pool thread, not a worker. Default channel size is 10,000 events (tuneable via `OHARNESS_EVENT_BUFFER`), generous enough that the fallback rarely fires outside pathological workloads.

### 4.7 Event schema

**Envelope** — every event has the same shape:

```rust
pub struct Event {
    pub v: SchemaVersion,          // "1.0" at launch
    pub seq: u64,                  // monotonic per run
    pub run_id: RunId,
    pub timestamp: Option<SystemTime>,  // None in replay output
    pub span_id: SpanId,
    pub parent: Option<u64>,       // parent span's opening seq
    pub kind: EventKind,
    pub redactions: Vec<String>,   // JSON paths in payload that were scrubbed
}
```

**Spans via pairs, not wrappers**: each span emits an open event and a close event sharing a `span_id`. The close carries the outcome. Enables streaming; supports post-hoc tree reconstruction via `parent`.

**Event catalog (v1.0)** — `EventKind` discriminated union:

| Category | Kinds |
|---|---|
| Lifecycle | `meta` (always first), `run.started`, `run.finished`, `turn.started`, `turn.finished`, `turn.revised` |
| LLM | `llm.request`, `llm.response`, `llm.stream.chunk`, `llm.retry`, `llm.failed` |
| Tool | `tool.call.started`, `tool.call.finished`, `tool.call.failed`, `tool.approval.requested`, `tool.approval.decided` |
| Memory | `memory.evicted`, `memory.summarized`, `memory.retrieved` |
| Budget | `budget.exceeded` |
| Policy | `policy.input.checked`, `policy.output.checked`, `policy.blocked` |
| Planner | `planner.proposed`, `planner.revised`, `planner.committed` |
| Critic | `critic.assessed`, `critic.rejected`, `critic.revised`, `critic.failed` |
| Reflector | `reflection.generated`, `reflection.injected` |
| Human | `human.interrupt`, `human.inject` |
| User sim | `user.simulated.message`, `user.simulated.ended` |
| Escape | `user.log` — requires `namespace: String` field (reverse-DNS) |

**Event payload notes:**
- `turn.revised { original_seq: u64, replacement_seq: u64, reason: String }` — fires when a critic's `Revise` verdict replaces a turn. Ties old to new so trajectories make revision visible.
- `critic.assessed` — fires on every critic call, payload includes verdict summary. Always present.
- `critic.rejected` — fires on `Reject` verdict (strictly more specific than `assessed`). Payload includes the rejection reason.
- `critic.revised` — fires on `Revise` verdict. Payload includes revision reason. The subsequent `turn.revised` carries the replacement linkage.
- `critic.failed` — fires when a critic panics or errors. Required event: fail-open behavior MUST emit this so positional replay can detect drift (a critic that failed-open during record but succeeds during replay is a silent divergence).
- `reflection.generated { episode_index, text_excerpt, metadata }` — emitted by `run_reflexion` when a `Reflector` produces a new reflection.
- `reflection.injected { episode_index, reflection_count }` — emitted by `ReflectionInjector` middleware when reflections are prepended into an outgoing request.

**Rules:**
- `meta` event is always first. Payload: `schema_version`, `harness_version`, `task_snapshot`, `llm_capabilities` (the `LlmCapabilities` of the LLM at record time — consumed by `ReplayLlm` to surface faithful capabilities; §9.6).
- Unknown event types preserved by consumers (forward-compat contract).
- Unknown payload fields preserved.
- `user.log` requires namespace; namespaces MUST NOT start with built-in category prefixes: `run.`, `turn.`, `llm.`, `tool.`, `memory.`, `budget.`, `policy.`, `planner.`, `critic.`, `reflection.`, `human.`, `user.simulated.`, `meta.`. Violations rejected at event construction (a `Result`-returning constructor).
- `budget.checked` is NOT an event — budget state lives in `RunOutcome.usage`. Spammy.
- `critic.failed` fires on BOTH record and replay paths, so positional replay can detect when critic behavior diverged.

**Versioning:** Semver on schema. Additive = minor, breaking = major. Support N-1 always, warn N-2, error older (post-v1). Rust types source of truth, JSON Schema exported as build artifact.

### 4.8 ConversationView

Read-only view over the conversation, post memory-policy mangling. What the LLM saw.

```rust
pub struct ConversationView<'a> {
    messages: &'a [Message],
    // may wrap additional filtering/transformation
}

impl<'a> ConversationView<'a> {
    pub fn messages(&self) -> &[Message];
    pub fn last_assistant(&self) -> Option<&Message>;
    pub fn user_visible(&self) -> Vec<Message>;    // strips tool blocks — for UserSimulator
    pub fn token_estimate(&self) -> u32;
}
```

`user_visible()` filtering logic lives in core since it's useful beyond `ConversationLoop` (any information-asymmetric multi-agent scenario).

### 4.9 Shared supporting types

```rust
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub cache_hints: CacheHints,
    pub extensions: Map<String, Value>,   // namespaced: "anthropic.thinking", etc.
}

pub struct CompletionResponse {
    pub id: String,
    pub model: ModelId,
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence(String),
    ToolUse,
    Refusal,
    Error(String),
}

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,            // JSON Schema
}

pub struct CacheHints {
    pub breakpoints: Vec<CacheBreakpoint>,
}
```

---

## 5. `oharness-llm`

### 5.1 The `Llm` trait

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> LlmCapabilities;

    async fn complete(&self, req: CompletionRequest)
        -> Result<CompletionResponse, LlmError>;

    async fn stream(&self, req: CompletionRequest)
        -> Result<BoxStream<'static, Result<Chunk, LlmError>>, LlmError>;
}
```

**Both methods required.** No defaults. Providers that don't support streaming return `Err(LlmError::Unsupported("stream"))`.

### 5.2 `Chunk`

Normalized across providers with escape hatch:

```rust
pub enum Chunk {
    MessageStart { id: String, model: ModelId },

    BlockStart { index: u32, kind: BlockStartKind },
    TextDelta     { index: u32, text: String },
    ToolUseDelta  { index: u32, partial_json: String },   // raw deltas, not accumulated
    ThinkingDelta { index: u32, text: String },
    BlockStop     { index: u32 },

    StopReason { reason: StopReason },
    Usage      { usage: Usage },
    MessageStop,

    Raw { provider: String, event: Value },              // escape hatch
}

pub enum BlockStartKind {
    Text,
    ToolUse { name: String, id: String },
    Thinking,
}
```

`StopReason` and `Usage` are **separate chunk variants** from `MessageStop` — providers may emit mid-stream usage updates.

### 5.3 `LlmError`

```rust
pub enum LlmError {
    Unsupported(&'static str),
    Authentication,
    RateLimited    { retry_after: Option<Duration> },
    ContextTooLong { max: u32, got: u32 },
    Cancelled,
    Network(io::Error),
    MalformedResponse(String),
    Provider(Box<dyn Error + Send + Sync>),
}
```

Structured enough for retry middleware: `RateLimited` + transient `Network` retry with backoff; `ContextTooLong` needs memory-policy action; `Authentication` fatal.

### 5.4 Helper: `complete_from_stream`

```rust
pub async fn complete_from_stream<S>(stream: S) -> Result<CompletionResponse, LlmError>
where S: Stream<Item = Result<Chunk, LlmError>> + Unpin;
```

Free function in `oharness-llm`. Providers sharing implementation between `complete()` and `stream()` use this internally.

### 5.5 Middleware contract

**`Llm` itself is the middleware contract.** Every middleware implements `Llm` wrapping another `Llm`. ToolSet middleware is symmetric.

**Two composition methods — infallible and fallible:**

```rust
pub trait LlmExt: Llm + Sized {
    /// Infallible composition. Layer must be of a type that always succeeds.
    fn with_layer<L: InfallibleLlmLayer<Self>>(self, layer: L) -> L::Output;

    /// Fallible composition. Layer may reject the inner Llm at construction
    /// (e.g., capability mismatch). Fluent chain uses `?`.
    fn try_with_layer<L: LlmLayer<Self>>(self, layer: L) -> Result<L::Output, LayerError>;
}

impl<T: Llm> LlmExt for T {}

/// Every layer implements this. If construction can fail (capability mismatch), the
/// layer type implements `LlmLayer` only. If it can never fail, it also implements
/// `InfallibleLlmLayer` which provides a trivial `with_layer` path.
pub trait LlmLayer<Inner: Llm> {
    type Output: Llm;
    fn wrap(self, inner: Inner) -> Result<Self::Output, LayerError>;
}

pub trait InfallibleLlmLayer<Inner: Llm>: LlmLayer<Inner> {
    fn wrap_infallible(self, inner: Inner) -> Self::Output;
}
```

**When to use which:**

- **`with_layer`** for layers that work on any `Llm`: `RateLimiter`, `Tracing`, `RequestLayer`-based redaction, etc. No `?` noise in the fluent chain.
- **`try_with_layer`** for layers with capability requirements: `PromptCaching::anthropic()` fails if the provider doesn't report `capabilities().prompt_caching`. Uses `?`.

```rust
let llm = AnthropicLlm::from_env()?
    .try_with_layer(PromptCaching::automatic())?     // capability-gated
    .with_layer(Tracing::new(sink.clone()))           // always succeeds
    .with_layer(RetryOnRateLimit::default())          // always succeeds
    .with_layer(RateLimiter::per_minute(60));         // always succeeds
```

Mixed chains stay readable. Safe layers don't pollute with `?`.

### 5.6 Middleware helper traits

Five traits for common shapes. Each has a blanket `Llm` impl via a wrapper type.

```rust
pub trait RequestLayer: Send + Sync {
    fn on_request(&self, req: &mut CompletionRequest);
}

pub trait ResponseLayer: Send + Sync {
    fn on_response(&self, res: &mut CompletionResponse);
}

/// Around-hook for middleware that wraps entire calls. Because `complete` and `stream`
/// have different return types and different retry semantics (restart-the-stream vs
/// re-issue-the-call), two methods rather than one generic `around`. Default impls
/// delegate to `around_complete` / `around_stream` so simple layers need only override
/// one if they only care about one mode.
#[async_trait]
pub trait FullLayer: Send + Sync {
    async fn around_complete(
        &self,
        req: CompletionRequest,
        call: BoxFuture<'_, Result<CompletionResponse, LlmError>>,
    ) -> Result<CompletionResponse, LlmError> {
        call.await
    }

    async fn around_stream(
        &self,
        req: CompletionRequest,
        call: BoxFuture<'_, Result<BoxStream<'static, Result<Chunk, LlmError>>, LlmError>>,
    ) -> Result<BoxStream<'static, Result<Chunk, LlmError>>, LlmError> {
        call.await
    }
}

pub trait ChunkObserver: Send + Sync {
    fn on_chunk(&self, chunk: &Chunk);             // observe only; no drop, no mutate
}

pub trait ChunkTransformer: Send + Sync {
    fn on_chunk(&self, chunk: Chunk) -> Option<Chunk>;   // None = drop
}
```

**Why `FullLayer` is two methods, not one generic `around<T, F>`:**
A generic `T` over both `CompletionResponse` and `BoxStream<...>` cannot express the different semantics each needs. A retry layer must restart a *stream* on `RateLimited` (open a new HTTP connection), not wrap a `BoxStream` in `?` and return it — which a generic `T` signature would force. Splitting into `around_complete` and `around_stream` makes each mode's semantics explicit.

**No `SymmetricFullLayer` abstraction.** Layers whose complete/stream logic is largely similar (e.g., retry) implement both methods with similar bodies — the actual semantics *always* differ enough (retry complete = re-call; retry stream = new HTTP connection + re-subscribe) that a shared `around<T>` would be misleading. The duplication is ~20 LOC per layer and buys correctness.

### 5.6.1 `ResponseLayer` streaming behavior

`ResponseLayer::on_response` operates on a `CompletionResponse` — a concept that doesn't exist mid-stream. Each `ResponseLayer` declares how it handles streams:

```rust
pub trait ResponseLayer: Send + Sync {
    fn on_response(&self, res: &mut CompletionResponse);

    /// How this layer behaves when wrapped around an `Llm` whose `stream()` is called.
    /// Default: WarnAndSkip. Layers that must operate on streams implement ChunkTransformer
    /// instead, or override to Error.
    fn stream_mode(&self) -> ResponseLayerStreamMode {
        ResponseLayerStreamMode::WarnAndSkip
    }
}

pub enum ResponseLayerStreamMode {
    /// Emit a `policy.layer_skipped_on_stream { layer_name }` event once per run
    /// and pass chunks through unchanged. Audible default — a silent pass-through
    /// would be a correctness trap for layers like redaction.
    WarnAndSkip,
    /// Return `LlmError::Unsupported("response_layer_on_stream")` from `stream()`.
    /// Use when the layer's invariants cannot be satisfied by streaming.
    Error,
    /// Explicitly acknowledge the layer is no-op on streams. For layers that only
    /// exist to observe (e.g., post-flight logging of final response details) —
    /// rare, since ChunkObserver is usually the right choice.
    SilentSkip,
}
```

The `WarnAndSkip` default means a redaction `ResponseLayer` wrapped around a provider whose caller eventually uses `stream()` produces a visible `policy.layer_skipped_on_stream` event in the trajectory. Reviewers will see it. Layers that genuinely work on both modes (e.g., a usage counter) set `SilentSkip`. Layers that must reject streaming set `Error`.

### 5.6.2 Wrapper types

Wrapper types (in `oharness-llm`):
- `WithRequestLayer<L, R>`: blanket `impl<L: Llm, R: RequestLayer> Llm` — applies `on_request` in both `complete()` and `stream()`.
- `WithResponseLayer<L, R>`: applies `on_response` inside `complete()`. In `stream()`, consults `layer.stream_mode()` and acts accordingly (emit warning event, error, or silent skip).
- `WithFullLayer<L, F>`: dispatches to `around_complete` / `around_stream` respectively.
- `WithChunkObserver<L, O>`: calls `on_chunk` on every chunk yielded from `stream()`. On `complete()`, passes through unobserved — observers are explicitly streaming-only. If a layer wants both, it implements `ResponseLayer` + `ChunkObserver` separately, or implements `Llm` directly.
- `WithChunkTransformer<L, T>`: applies `on_chunk` to each chunk yielded from `stream()`. `None` returns skip the chunk. On `complete()`, passes through unchanged.

**Capability propagation:** default `capabilities()` delegates to inner. Layers that affect capabilities (fallback, caching) override.

**Escape hatch:** for middleware that doesn't fit any helper (speculative sampling, fallback-between-providers, **or middleware that needs multiple hook sites simultaneously with shared state, e.g., `BudgetMiddleware`**), implement `Llm` directly. ~30 LOC per technique.

### 5.7 Ordering recommendations (documented, not enforced)

| Position | Typical layers | Why |
|---|---|---|
| Innermost | Prompt caching | Operates on final request |
| Middle-inner | Tracing | Retries become visible events |
| Middle | Retry, fallback | Operates on post-mutation picture |
| Outermost | Rate limiter, approval gate | User-visible protection |

Tracing counterintuitively goes inner, so retry attempts are individually logged.

---

## 6. `oharness-providers`

Feature-gated provider adapters. Each provider is a feature flag; no provider enabled by default in this crate (consumers opt in).

Providers to ship:

| Provider | Feature flag | Priority |
|---|---|---|
| Anthropic | `anthropic` | v1 (day 1) |
| OpenAI (chat completions + responses API) | `openai` | v1 |
| OpenRouter | `openrouter` | v1 |
| Ollama | `ollama` | v1 |
| vLLM (OpenAI-compatible endpoint) | `vllm` | v1 |
| AWS Bedrock | `bedrock` | v1.1 |
| Google Vertex | `vertex` | v1.1 |
| Gemini direct | `gemini` | v1.1 |

**Anthropic capabilities to preserve:** prompt caching, extended thinking, parallel tool use, vision, PDF documents.

**OpenAI capabilities to preserve:** structured outputs, parallel tool use, vision, o-series reasoning.

Each adapter converts to/from Anthropic-shaped `Message` types (our canonical representation). `Chunk::Raw` escape hatch for provider events not yet normalized.

---

## 7. `oharness-tools`

### 7.1 The `ToolSet` trait

```rust
#[async_trait]
pub trait ToolSet: Send + Sync {
    fn specs(&self) -> &[ToolSpec];
    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext)
        -> ToolOutcome;
}

pub enum ToolOutcome {
    Success(ToolOutput),
    ExecutionError { message: String, recoverable: bool },
    Denied { reason: String },                  // gated by policy
    Cancelled,
}
```

### 7.2 `ToolContext`

Threaded through every `execute()` call. Carries cross-cutting concerns.

```rust
pub struct ToolContext {
    pub events: Arc<dyn EventSink>,
    pub budget: Arc<dyn BudgetHandle>,
    pub cancellation: CancellationToken,
    pub approval: Arc<dyn ApprovalChannel>,
    pub workspace: Option<Arc<Workspace>>,
    pub extensions: Map<String, Value>,         // namespaced, like request extensions
}
```

### 7.3 Tool schemas

JSON Schema always round-trips. Hand-written JSON escape hatch always available.

**Optional `schemars` integration** behind `derive-schema` feature:

```rust
#[cfg(feature = "derive-schema")]
#[derive(ToolInput)]
pub struct FsWriteInput {
    pub path: String,
    pub content: String,
}
```

### 7.4 ToolSet middleware

Symmetric to `Llm` middleware. `ToolSet` is the contract; implement `ToolSet` wrapping another.

**Helper traits:**

```rust
pub trait ToolPolicy: Send + Sync {
    async fn decide(&self, name: &str, input: &Value, ctx: &ToolContext) -> Decision;
}

pub enum Decision { Allow, Deny(String), AskUser }

pub trait ToolRequestLayer: Send + Sync {
    fn on_call(&self, name: &str, input: &mut Value);
}

pub trait ToolResponseLayer: Send + Sync {
    fn on_result(&self, name: &str, outcome: &mut ToolOutcome);
}
```

Wrapper types: `ApprovalGate<T, P>`, `WithToolRequestLayer<T, R>`, `WithToolResponseLayer<T, R>`.

### 7.5 Bundled tool kits

All feature-gated in `oharness-tools`:

| Kit | Feature | Contents |
|---|---|---|
| `bash` | `bash` | Shell execution, with sandboxing helpers |
| `fs` | `fs` | read/write/list/glob, scoped to `Workspace` |
| `http` | `http` | GET/POST with allowlist policy |
| `python-sandbox` | `python-sandbox` | Subprocess-based Python exec |
| `mcp-bridge` | `mcp` | Consume MCP servers as `ToolSet` |

### 7.6 MCP integration (v1)

**Consume** MCP servers: `McpToolSet::connect(url)` returns a `ToolSet`. Free interoperability with the MCP ecosystem.

**Expose** as MCP server: deferred to v1.1+. Needs session-model design first.

---

## 8. `oharness-memory`

### 8.1 `MemoryPolicy` trait

```rust
pub trait MemoryPolicy: Send + Sync {
    async fn transform(&self, conversation: ConversationView<'_>, ctx: &MemoryContext)
        -> Result<Vec<Message>, MemoryError>;
}

pub struct MemoryContext {
    pub events: Arc<dyn EventSink>,
    pub token_budget: u32,
}

pub enum MemoryError {
    RetrieverTimeout,
    SummarizerFailed(LlmError),
    Configuration(String),
}
```

Memory policies run before each LLM call: input is the full conversation, output is what gets sent. Policies emit `memory.evicted` / `memory.summarized` / `memory.retrieved` events.

### 8.2 Shipped policies

| Policy | Description |
|---|---|
| `Passthrough` | No-op; send everything |
| `TruncateAfterTokens(n)` | Drop oldest messages until under token budget |
| `ElideToolResults` | Replace old tool results with `[elided: ...]` placeholders (current ought behavior) |
| `Summarize { llm, threshold }` | LLM-compressed summary of old messages |
| `HierarchicalSummary` | Rolling summary tree |
| `Rag { retriever }` | Retrieval-augmented from external store |

**Key seam:** `Retriever` trait abstracts the store. `oharness-memory` doesn't ship vector DBs — users bring Pinecone/Qdrant/LanceDB via `Retriever` impls.

```rust
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<RetrievedItem>, RetrieverError>;
}
```

---

## 9. `oharness-trace`

### 9.1 EventSink implementations

The `EventSink` trait is defined in `oharness-core` (§4.6). This crate provides the concrete implementations.

**Backpressure policy: block the emitter.** Dropping events silently corrupts trajectories — unacceptable default. Config knob `OHARNESS_EVENT_OVERFLOW=drop` for users who prefer not to be slowed.

### 9.2 Shipped sinks

- `FileSink::to_path(path)` — JSONL writer, spawns a tokio writer task
- `FileSink::to_path_gz(path)` — gzipped JSONL
- `InMemorySink` — for tests; keeps all events in a Vec
- `FanOutSink` — sends to multiple sinks
- `NullSink` — drops everything (for benchmarking the harness itself; also the default in `AgentBuilder`)

### 9.3 Trajectory file format

JSONL (newline-delimited JSON), optionally gzipped. `{run_id}.jsonl[.gz]`.

- First line is `meta` event
- Last line on success is `run.finished`
- File ends without `run.finished` → partial trajectory (recoverable, useful)
- grep/jq-friendly
- Concatenatable across runs

**Payload externalization** at 64KB threshold (configurable):

```json
{"type":"tool.call.finished","payload":{
  "output":{"ref":"payloads/sha256-abc123.bin","size":1048576,"hash":"sha256:abc123","mime":"text/plain"}
}}
```

Sidecar directory: `{run_id}.payloads/`.

**Write ordering (mandatory for readers tailing the file):**
1. Writer computes SHA-256 of the externalized payload.
2. Writer writes `{run_id}.payloads/sha256-<hash>.bin` and **fsyncs** it.
3. Only then writes the JSONL line referencing it.

Violating this order allows a tailing reader (another `openh trace view --follow` process, for example) to encounter a dangling `ref`. Readers must tolerate the missing-sidecar case with a `PayloadNotFound` warning, but writers must never create the window. Tests cover this invariant.

Run IDs are UUIDs — concurrent runs writing to the same output directory never collide. Two runs *with the same run id* is a caller error, detected at sink construction (fails if the target file exists).

### 9.4 `TrajectoryHandle`

```rust
pub struct TrajectoryHandle {
    source: TrajectorySource,
}

enum TrajectorySource {
    File(PathBuf),
    InMemory(Arc<Vec<Event>>),
    Stream(Box<dyn Fn() -> BoxStream<'static, Result<Event, TrajectoryError>> + Send + Sync>),
}

impl TrajectoryHandle {
    pub fn events(&self) -> BoxStream<'static, Result<Event, TrajectoryError>>;
    pub async fn load_all(&self) -> Result<Vec<Event>, TrajectoryError>;
    pub fn summarize(&self) -> TrajectorySummary;  // counts, no content
}
```

Serializes as path/URI reference — **not** inlined. In-memory handles error on serialization unless materialized.

### 9.5 Tracing middleware

Plain middleware living in `oharness-trace`. Instances:

- `RequestTracer` (for `Llm::complete`)
- `StreamTracer` (for `Llm::stream`, implements `ChunkObserver`)
- `ToolTracer` (for `ToolSet`)

All hold an `Arc<dyn EventSink>` from construction.

### 9.6 Replayer

```rust
pub struct ReplayLlm {
    trajectory: TrajectoryHandle,
    llm_capabilities: LlmCapabilities,        // read from the `meta` event's `llm_capabilities` payload field
    mode: ReplayMode,
    on_drift: DriftPolicy,
}

pub enum ReplayMode {
    Positional,         // default: Nth llm.request in loop pairs with Nth recorded response
    Strict,             // inputs must match byte-for-byte
}

pub enum DriftPolicy {
    WarnAndContinue,    // default
    Fail,
}

impl Llm for ReplayLlm {
    fn name(&self) -> &str { "replay" }

    /// Returns the capabilities recorded in the trajectory's `meta` event.
    /// A trajectory captured from a streaming-capable Anthropic provider replays
    /// with `streaming: true`, even though the ReplayLlm produces chunks from
    /// recorded events rather than a live connection. This means capability-gated
    /// middleware (e.g., `PromptCaching::anthropic()`) can be applied over a
    /// ReplayLlm without construction errors — though the layer's effect is usually
    /// moot since requests are replayed positionally rather than re-sent.
    fn capabilities(&self) -> LlmCapabilities { self.llm_capabilities.clone() }

    // complete() and stream() reproduce recorded responses / chunks
}
```

`ReplayLlm` and `ReplayToolSet` implement `Llm` / `ToolSet` respectively. Deterministic across machines, zero API cost. The `meta` event's payload includes `llm_capabilities: LlmCapabilities` (part of the schema contract per §4.7) specifically so replay can faithfully surface them.

### 9.7 Trajectory CLI

```
openh trace info <path>           # summary: counts, token totals, errors
openh trace view <path>           # pretty-printed event stream
openh trace grep <path> <pattern> # filter events
openh trace diff <a> <b>          # structural diff between two runs
```

---

## 10. `oharness-budget`

### 10.1 `BudgetHandle` reference

The `BudgetHandle` trait is defined in `oharness-core` (§4.6). This crate provides the concrete `BudgetHandle` implementations listed below, plus the supporting value types:

```rust
pub struct BudgetAmount {
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub cost_usd: f64,
    pub wall_clock: Duration,
    pub steps: u32,
}

pub enum BudgetDecision {
    Allow,
    Deny { reason: String },
}

pub struct BudgetRequest { /* estimate of the call being considered */ }
pub struct BudgetSnapshot { /* current state, for logging/inspection */ }

/// Error wrapper used when budget middleware short-circuits an LLM call. Wrapped in
/// `LlmError::Provider` so downstream handlers can distinguish budget exhaustion from
/// other provider errors (via downcast) while keeping the error type stable.
#[derive(Debug, thiserror::Error)]
#[error("budget exceeded: {reason}")]
pub struct BudgetExceeded {
    pub reason: String,
}
```

### 10.2 Shipped budgets

- `TokenBudget::input_plus_output(n)` — hard cap
- `CostBudget::usd(n)` — requires pricing data per model
- `StepBudget::turns(n)` — max turns
- `TimeBudget::wall_clock(d)` — max wall-clock duration
- `CompositeBudget` — multiple budgets, any-exceeded = deny

### 10.3 Middleware

`BudgetMiddleware` implements `Llm` directly (the escape hatch per §5.6.2), not via a helper-trait wrapper. It needs to participate in multiple hook sites with **shared state** (the budget counter) — pre-call checks, post-call consumption on `complete()`, and per-chunk observation on `stream()` — which no single helper trait can express.

```rust
pub struct BudgetMiddleware<L> {
    inner: L,
    budget: Arc<dyn BudgetHandle>,
    pricing: Arc<PricingTable>,
}

#[async_trait]
impl<L: Llm> Llm for BudgetMiddleware<L> {
    fn name(&self) -> &str { self.inner.name() }
    fn capabilities(&self) -> LlmCapabilities { self.inner.capabilities() }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let pre_check = self.budget.check(BudgetRequest::from_request(&req, &self.pricing)).await;
        if let BudgetDecision::Deny { reason } = pre_check {
            return Err(LlmError::Provider(Box::new(BudgetExceeded { reason })));
        }
        let res = self.inner.complete(req).await?;
        self.budget.consume(BudgetAmount::from_response(&res, &self.pricing)).await;
        Ok(res)
    }

    async fn stream(&self, req: CompletionRequest)
        -> Result<BoxStream<'static, Result<Chunk, LlmError>>, LlmError>
    {
        // Pre-check
        let pre_check = self.budget.check(BudgetRequest::from_request(&req, &self.pricing)).await;
        if let BudgetDecision::Deny { reason } = pre_check {
            return Err(LlmError::Provider(Box::new(BudgetExceeded { reason })));
        }

        let inner_stream = self.inner.stream(req).await?;
        let budget = self.budget.clone();
        let pricing = self.pricing.clone();

        // Wrap the stream: observe Chunk::Usage, short-circuit on exceed
        let wrapped = inner_stream.then(move |chunk_result| {
            let budget = budget.clone();
            let pricing = pricing.clone();
            async move {
                if let Ok(Chunk::Usage { usage }) = &chunk_result {
                    budget.consume(BudgetAmount::from_usage(usage, &pricing)).await;
                    if let BudgetDecision::Deny { reason } = budget.check(BudgetRequest::empty()).await {
                        return Err(LlmError::Provider(Box::new(BudgetExceeded { reason })));
                    }
                }
                chunk_result
            }
        }).boxed();

        Ok(wrapped)
    }
}
```

Why directly: a single `BudgetMiddleware` value needs to observe state across all three hook sites (pre, post-complete, per-chunk). Composing a `FullLayer` + a separate `ChunkObserver` would produce two wrapper values that don't share the budget counter — or require a hacky `Arc<Mutex<_>>` dance the helper traits don't support. Direct `impl Llm` is cleaner and correct.

Tools call `ctx.budget.check()` / `.consume()` themselves for tool-level costs — `BudgetHandle` is a context trait (§4.6), available via `ToolContext.budget`.

### 10.4 Model pricing

Shipped pricing table per model for cost calculation. Must be updatable without library bump:

```rust
pub struct PricingTable {
    models: Map<ModelId, ModelPricing>,
}

impl PricingTable {
    pub fn builtin() -> Self;                    // ships with common models
    pub fn load_from(path: &Path) -> Result<Self, _>;
    pub fn override_model(&mut self, id: ModelId, pricing: ModelPricing);
}
```

---

## 11. `oharness-critic`

### 11.1 `Critic` trait

```rust
#[async_trait]
pub trait Critic: Send + Sync {
    fn name(&self) -> &str;
    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict;
}

pub struct AssessmentContext<'a> {
    pub task: &'a Task,
    pub conversation: ConversationView<'a>,    // post-memory-policy view
    pub latest_turn: &'a AssistantTurn,
    pub trajectory: TrajectoryView<'a>,        // read-only peek at events-so-far
}

pub enum CriticVerdict {
    Accept,
    AcceptWithNote(String),
    Reject { reason: String },
    Revise { replacement: AssistantTurn, reason: String },
    Abort  { reason: String },
}
```

**Supporting types** (both defined in `oharness-core`, re-used by `Critic`, `Loop`, `Reflector`):

```rust
/// A single assistant turn as produced by the loop. Distinct from `Message::Assistant`
/// because it bundles turn metadata the critic/loop need.
pub struct AssistantTurn {
    pub turn_index: u32,
    pub span_id: SpanId,                       // ties to llm.request/response events
    pub message: Message,                      // always Message::Assistant
    pub tool_calls: Vec<ToolCall>,             // parsed from message content for convenience
    pub usage: Usage,
    pub stop_reason: StopReason,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Read-only, lifetime-scoped view into the trajectory being built. Backed by the
/// in-memory tail of the event stream. Distinct from `TrajectoryHandle` (§9.4): the
/// handle is a post-run reference; `TrajectoryView` is a mid-run peek. A critic MUST
/// NOT mutate the trajectory — only read for context.
pub struct TrajectoryView<'a> {
    events: &'a [Event],
}

impl<'a> TrajectoryView<'a> {
    pub fn events(&self) -> &[Event];
    pub fn events_of_kind(&self, prefix: &str) -> impl Iterator<Item = &Event>;
    pub fn turn_count(&self) -> u32;
    pub fn to_handle(&self) -> TrajectoryHandle;   // copy into an in-memory handle
}
```

**Rules:**
- `Revise` **replaces** the assistant turn (not append). Loop emits `turn.revised { original_seq, replacement_seq, reason }` tying the two.
- Revision depth capped (default 3, configurable); exceeding → convert to `Reject`.
- Critic failures are **fail-open** (treated as `Accept` with a `critic.failed` event). `critic.failed` is a required event (not just log output) so positional replay can detect when critic behavior diverged between record and replay.

### 11.2 `CriticTrigger`

Agent config, not on the trait.

```rust
pub enum CriticTrigger {
    AfterAssistant,            // default
    AfterToolResult,
    AfterEveryNTurns(u32),
    OnDemand,
}
```

### 11.3 `CompositeCritic`

```rust
pub struct CompositeCritic {
    critics: Vec<Box<dyn Critic>>,
    policy: AggregationPolicy,
}

pub enum AggregationPolicy {
    FirstReject,                      // sequential, short-circuit on non-Accept
    AllMustAccept,                    // parallel-safe; any non-Accept wins
    MajorityVote,                     // runs all, majority decides
    Weighted(Vec<f32>),               // weighted voting
}
```

**Critics are independent** — no shared state, no visibility into each other. `AllMustAccept` runs in parallel via `tokio::join_all`. `FirstReject` is sequential by nature.

### 11.4 `Reflector` trait

```rust
#[async_trait]
pub trait Reflector: Send + Sync {
    fn name(&self) -> &str;
    async fn reflect(&self, episode: &Episode<'_>) -> Option<Reflection>;
}

/// Borrowed view passed to `Reflector::reflect`. Fields borrow from the
/// caller's locals during a single `run_reflexion` iteration.
pub struct Episode<'a> {
    pub index: u32,
    pub task: &'a Task,
    pub outcome: &'a RunOutcome,
    pub evaluation: &'a EvaluationResult,
    pub prior_reflections: &'a [Reflection],
}

/// Owned counterpart of `Episode<'a>` for storage after the iteration's locals drop.
/// Returned from `run_reflexion` to the caller.
pub struct OwnedEpisode {
    pub index: u32,
    pub task: Task,
    pub outcome: RunOutcome,
    pub evaluation: EvaluationResult,
    pub prior_reflections: Vec<Reflection>,
}

impl<'a> Episode<'a> {
    pub fn to_owned(&self) -> OwnedEpisode;
}

pub struct Reflection {
    pub text: String,
    pub metadata: Map<String, Value>,
    pub created_at: SystemTime,
}
```

**Always called**, returns `Option<Reflection>` — reflector gates internally.

### 11.5 `ReflectionInjector` middleware

Lives in `oharness-critic`. Wraps `Llm` via `RequestLayer`. Prepends reflections as a system-prompt suffix (or first-user-message prefix, configurable). Emits `reflection.injected { episode_index, reflection_count }` on each invocation.

Core Agent/Loop stays unaware of reflection concept — reflections become a `Vec<Reflection>` threaded through middleware configuration. `run_reflexion` reconfigures the injector's reflections between episodes.

### 11.6 Shipped critic/reflector implementations

- `LlmJudgeCritic { judge: Arc<dyn Llm>, rubric: String, threshold: f32 }`
- `TestCritic { workspace: PathBuf, cmd: Vec<String> }` — runs command, check exit status
- `RegexDenyCritic` — output regex-matching block
- `ConstitutionalCritic` — principle-based revision
- `LlmReflector { llm: Arc<dyn Llm>, template: String }`
- `NullReflector` — returns `None` always (debugging)

---

## 12. `oharness-loop`

### 12.1 `Loop` trait

```rust
#[async_trait]
pub trait Loop: Send + Sync {
    async fn run(
        &self,
        task: Task,
        ctx: &LoopContext,
    ) -> Result<RunOutcome, AgentError>;
}

pub struct LoopContext {
    pub llm: Arc<dyn Llm>,
    pub tools: Arc<dyn ToolSet>,
    pub memory: Arc<dyn MemoryPolicy>,
    pub critics: Option<Arc<CompositeCritic>>,
    pub critic_trigger: CriticTrigger,
    pub events: Arc<dyn EventSink>,
    pub budget: Arc<dyn BudgetHandle>,
    pub cancellation: CancellationToken,
    pub approval: Arc<dyn ApprovalChannel>,
    pub revision_depth_cap: u32,
    pub max_turns: u32,
}
```

### 12.2 Shipped loops

- **`ReactLoop`** — default. Thought → Action → Observation cycle. ~200 LOC.
- **`ConversationLoop<U: UserSimulator>`** — alternates agent + user simulator.
- **`ReflexionLoop`** — multi-episode wrapper; uses `TaskEvaluator` + `Reflector`. Not a `Loop` impl itself but a function that invokes an inner `Loop` repeatedly.

Deferred to v1.1+:
- `PlanExecuteLoop`
- `TreeOfThoughtsLoop`
- `GraphOfThoughtsLoop`

### 12.3 `UserSimulator` trait

```rust
#[async_trait]
pub trait UserSimulator: Send + Sync {
    fn name(&self) -> &str;
    async fn initial_message(&self, task: &Task) -> Result<String, UserError>;
    async fn respond(&self, conversation: ConversationView<'_>, task: &Task)
        -> Result<UserAction, UserError>;
}

pub enum UserAction {
    Say(String),
    EndConversation,
}
```

**Rules:**
- Simulator receives a `ConversationView` — calling `.user_visible()` on it strips tool calls and internal reasoning. Simulators that want stricter or looser filters can compose their own views on top.
- `task` passed every `respond()` call (cheap, enables context-dependent personas).
- Simulator errors → `Termination::Failed { reason: "user_simulator_error" }` (not `EndConversation`).
- Primary termination = simulator's `EndConversation`; secondary = `max_turns`.

### 12.4 Shipped simulators

- `LlmUserSimulator { llm: Arc<dyn Llm>, persona: String, prompt_template: String }`
- `ScriptedUserSimulator` — replays a fixed sequence (for tests/eval reproducibility)

### 12.5 `Agent` assembly

```rust
pub struct Agent {
    llm: Arc<dyn Llm>,
    tools: Arc<dyn ToolSet>,
    memory: Arc<dyn MemoryPolicy>,
    critics: Option<Arc<CompositeCritic>>,
    critic_trigger: CriticTrigger,
    loop_impl: Box<dyn Loop>,
    events: Arc<dyn EventSink>,
    budget: Arc<dyn BudgetHandle>,
    reflection_injector: Option<Arc<ReflectionInjector>>,  // None if not configured
    config: AgentConfig,
}

impl Agent {
    pub fn builder() -> AgentBuilder;
    pub async fn run(&self, task: Task) -> Result<RunOutcome, AgentError>;
    pub async fn run_conversation(&self, messages: Vec<Message>) -> Result<RunOutcome, AgentError>;

    /// Event sink accessor. Used by `run_reflexion` to emit `reflection.generated`
    /// events between episodes without re-threading the sink through a separate path.
    pub fn sink(&self) -> &Arc<dyn EventSink> { &self.events }

    /// Reflection injector accessor. Returns `None` if the agent wasn't built with
    /// `.with_reflection_injector(...)`. `run_reflexion` uses this to reconfigure
    /// injected reflections between episodes; if `None`, reflections have no effect.
    pub fn injector(&self) -> Option<&Arc<ReflectionInjector>> { self.reflection_injector.as_ref() }
}
```

`AgentBuilder::with_reflection_injector()` wires a `ReflectionInjector` into the LLM middleware stack AND stashes a handle on the `Agent` for `run_reflexion`'s later reconfiguration. Building an agent without a reflection injector and passing it to `run_reflexion` is a configuration error caught at the start of the reflexion loop.

`run(Task)` is the **documented path**. `run_conversation` is the escape hatch.

`AgentBuilder` preconfigures:
- **`NullSink` by default** — no file writes from a library call. Users opt in via `.with_event_sink(FileSink::to_path(...))` or the CLI (`openh run` enables file tracing).
- `ReactLoop` as default
- `Passthrough` memory policy default
- No critics default
- `NullBudget` default (unlimited)

**Why `NullSink` by default, not a file sink:** a library call that writes to cwd is a footgun in CI, pytest runners, embedded use, and servers. The CLI turns on file tracing because it has a known output directory; the library doesn't assume one. The doc quickstart for tracing is one extra line (`.with_event_sink(FileSink::to_path("run.jsonl")?)`).

### 12.6 `run_reflexion` helper

```rust
pub async fn run_reflexion(
    agent: &Agent,
    task: Task,
    evaluator: Arc<dyn TaskEvaluator>,
    reflector: Arc<dyn Reflector>,
    max_episodes: u32,
) -> Vec<OwnedEpisode>;
```

Returns `Vec<OwnedEpisode>` — the borrowed `Episode<'a>` is what's passed into `Reflector::reflect` per iteration; owned versions are returned to the caller for storage. Each episode runs the full agent; reflections threaded into next via `ReflectionInjector` middleware. Emits `reflection.generated` after each successful reflection and `reflection.injected` each time the injector prepends reflections into an outgoing request.

Returns `Err(AgentError::Configuration)` immediately if the agent was not built with `.with_reflection_injector(...)` — caught before any episode runs.

```rust
// Sketch of run_reflexion's body (for clarity, not part of the public API):
let injector = agent.injector()
    .ok_or(AgentError::Configuration(
        "run_reflexion requires an agent built with .with_reflection_injector()".into()
    ))?;

let mut reflections: Vec<Reflection> = Vec::new();
let mut out: Vec<OwnedEpisode> = Vec::new();
for i in 0..max_episodes {
    injector.set_reflections(reflections.clone());          // reconfigure middleware
    let outcome = agent.run(task.clone()).await?;
    let eval = evaluator.evaluate(&task, &outcome).await;

    let borrowed = Episode {
        index: i, task: &task, outcome: &outcome,
        evaluation: &eval, prior_reflections: &reflections,
    };
    let should_stop = eval.passed;
    if let Some(r) = reflector.reflect(&borrowed).await {
        agent.sink().emit(Event::reflection_generated(i, &r));
        reflections.push(r);
    }
    out.push(borrowed.to_owned());
    if should_stop { break; }
}
```

---

## 13. `oharness-eval`

### 13.1 `TaskEvaluator` trait

```rust
#[async_trait]
pub trait TaskEvaluator: Send + Sync {
    async fn evaluate(&self, task: &Task, outcome: &RunOutcome) -> EvaluationResult;
}

pub struct EvaluationResult {
    pub score: f64,                      // 0.0..1.0 typical, arbitrary allowed
    pub passed: bool,
    pub details: Map<String, Value>,
}
```

### 13.2 `Benchmark` trait

```rust
#[async_trait]
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &'static str;
    fn version(&self) -> &str;               // dataset version (e.g., "swe-bench-lite-1.2.1")
    fn task_count(&self) -> Option<usize>;

    fn task_ids(&self) -> Box<dyn Iterator<Item = String> + Send + '_>;
    async fn load_task(&self, id: &str) -> Result<LoadedTask, BenchmarkError>;

    fn evaluator(&self) -> Arc<dyn TaskEvaluator>;
}

pub struct LoadedTask {
    pub task: Task,
    pub workspace: Option<Workspace>,
}

pub struct Workspace {
    pub path: PathBuf,
    cleanup: Option<WorkspaceCleanup>,
}

/// Cleanup can be sync (tempdir removal) or async (kill container, release worktree).
/// On Drop, sync cleanups run inline; async cleanups are spawned on the current tokio
/// runtime as best-effort (panic if no runtime). Benchmarks that need guaranteed
/// cleanup call `Workspace::teardown().await` explicitly before drop.
pub enum WorkspaceCleanup {
    Sync(Box<dyn FnOnce() + Send>),
    Async(BoxFuture<'static, ()>),
}

impl Workspace {
    pub async fn teardown(mut self);   // explicit, awaitable
}
```

**Rules:**
- `task_ids` is cheap enumeration — no I/O beyond listing the dataset
- `load_task` does expensive setup (git clone, container setup, etc.)
- `Workspace` cleans up on Drop (RAII)

### 13.3 `BenchmarkRunConfig`

```rust
pub struct BenchmarkRunConfig {
    pub output_dir: PathBuf,
    pub run_concurrency: usize,          // default: 8
    pub load_concurrency: usize,         // default: 4 (network/disk-bound)
    pub max_cost_usd: Option<f64>,
    pub filter: Option<String>,          // glob pattern on task id
    pub sample_n: Option<usize>,
    pub shard: Option<(usize, usize)>,   // (index, total) for manual sharding
    pub resume: bool,                    // skip already-completed
}
```

### 13.4 Runner

```rust
pub async fn run_benchmark<B: Benchmark, F, Fut>(
    benchmark: B,
    agent_factory: F,
    config: BenchmarkRunConfig,
) -> BenchmarkReport
where
    F: Fn(&LoadedTask) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Agent, AgentError>> + Send;
```

Async from day one — real factories need to pull auth, load reflections, construct Llm stacks with middleware that may do I/O at construction. Sync-only factories wrap a value in `async { Ok(agent) }`.

**Rules:**
- `agent_factory` receives `&LoadedTask` — agent can be configured per-instance (e.g., tools scoped to `workspace.path`)
- Separate concurrency pools for load vs run
- Rate-limit coordination across workers via `Arc<dyn Llm>` shared across all agents (middleware handles)
- Max-cost cutoff: in-flight runs finish, no new ones started
- Resume: check `output_dir` for existing `{task_id}/outcome.json`, skip those

### 13.5 Results directory layout

```
results/
├── config.toml              # snapshot of run config
├── manifest.json            # completed ids, rolling cost
├── {task_id}/
│   ├── outcome.json         # serialized RunOutcome
│   ├── trajectory.jsonl
│   ├── evaluation.json      # EvaluationResult
│   └── payloads/            # externalized event payloads
└── ...
```

This is the paper-supplement artifact. Portable, replayable, diffable.

### 13.6 Benchmark adapters

Ship as separate crates depending on `oharness-eval`:

- `oharness-bench-swe` — SWE-bench-lite, SWE-bench-full
- `oharness-bench-tau` — τ-bench (conversational, uses `ConversationLoop`)
- `oharness-bench-gaia` — GAIA (web-based; needs browser tool)
- `oharness-bench-agent` — AgentBench (as licensing permits)

### 13.7 CLI commands

```
openh bench list                            # list built-in benchmarks
openh bench run <benchmark>                 # run benchmark
    --agents <config.toml>                  # agent factory config
    --out <dir>
    --concurrency <n>
    --max-cost <usd>
    --filter <glob>
    --sample-n <n>
    --shard <i/n>
openh bench resume <results-dir>
openh bench report <results-dir>            # aggregate pass@1, cost, turns
openh bench diff <dir-a> <dir-b>           # side-by-side comparison
```

---

## 14. `oharness-py` (Python bindings)

### 14.1 API shape

**Blocking-first** with async variants:

```python
from oharness import Agent, Anthropic, Task

agent = Agent(
    llm=Anthropic.from_env(),
    tools=default_tools(),
).with_tracing("run.jsonl")

result = agent.run(Task("Fix the bug in auth.py"))     # blocking
# async variant:
result = await agent.arun(Task("..."))
```

### 14.2 Implementable in Python

Rust traits exposed as Python ABCs. Python implementations are driven from the Rust loop.

**Python-facing API (user writes this):**

```python
from oharness.base import Critic, CriticVerdict, AssessmentContext

class MyReflexionCritic(Critic):
    def name(self) -> str:
        return "my-reflexion"

    async def assess(self, ctx: AssessmentContext) -> CriticVerdict:
        if ...: return CriticVerdict.reject("reason")
        return CriticVerdict.accept()

agent = Agent(...).with_critic(MyReflexionCritic())
```

**Rust-side adapter (what the library ships internally):**

```rust
#[pyclass]
struct PyCritic { inner: PyObject }  // holds the Python instance

#[async_trait]
impl Critic for PyCritic {
    fn name(&self) -> &str { &self.cached_name }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        let py_ctx: Py<PyAny> = Python::with_gil(|py| convert_ctx(py, ctx));
        // Call the Python async method, bridge via pyo3-async-runtimes:
        let coro = Python::with_gil(|py| {
            self.inner.call_method1(py, "assess", (py_ctx,))
        }).unwrap();
        let fut = Python::with_gil(|py| {
            pyo3_async_runtimes::tokio::into_future(coro.bind(py))
        }).unwrap();
        match fut.await {
            Ok(verdict_obj) => Python::with_gil(|py| parse_verdict(py, verdict_obj)),
            Err(pyerr) => {
                // Convert to CriticVerdict::Accept with a critic.failed event
                // (fail-open per §11.1 rules); include pyerr message in the event
                CriticVerdict::Accept
            }
        }
    }
}
```

**Known constraints (explicit, not handwaved):**

1. **Async bridging requires `pyo3-async-runtimes`** (successor to `pyo3-asyncio`). Python `async def` methods return coroutines; Rust futures have to drive them via this bridge. Adds a dependency but is the standard solution.

2. **`Python::with_gil` per call.** Each method call acquires the GIL briefly to marshal arguments and call the Python method. For `Critic::assess` (once per turn) this is fine. For high-frequency calls (per-chunk transformers), Python implementations are **not recommended** — we document this as a rough rule: `on_chunk`-level Python traits will work but cost ~microseconds of GIL acquisition per chunk.

3. **`Llm::stream` Python implementations are possible but costly.** Each yielded chunk crosses the FFI boundary, re-acquires GIL. Acceptable for research prototypes (<100 chunks/sec), not for production-scale streaming. Documented explicitly in the Python docs.

4. **`Send + Sync` via opaque `Py<PyAny>`.** Python objects are technically `Send` via GIL-protected access — `pyo3` models this correctly. But a Python impl that captures non-GIL-safe state in a closure is on the user; we surface this as a documented caveat, not a hard guarantee.

5. **Exception translation.** Python exceptions in a trait method become `LlmError::Provider` (or the trait's equivalent error). The Rust adapter catches all Python errors and translates; the Python traceback is preserved as a string inside the error.

**Priority order for Python-implementable traits:**

| Trait | Python support | Rationale |
|---|---|---|
| `Llm::complete` | **v1** | Once per turn; cheap. Must ship with v1 so Python users of `MemoryPolicy::Summarize` aren't forced to bring a Rust Llm. |
| `Critic` | **v1** | Low-frequency, high research value (Reflexion, self-refine iterations) |
| `Reflector` | **v1** | Even lower frequency; obvious need |
| `TaskEvaluator` | **v1** | Runs once per task; trivial |
| `UserSimulator` | **v1** | Per-turn frequency, acceptable |
| `MemoryPolicy` | **v1** | Once per LLM call, acceptable. Depends on Python-implementable `Llm` so `Summarize` works end-to-end from Python. |
| `ToolSet` | **v1.1** | Per-tool-call; acceptable but needs solid schema marshalling story |
| `Llm::stream` | **v1.2** | Per-chunk GIL crossing is the worst case; document limitations, ship without if needed |
| `RequestLayer` / `ResponseLayer` | **v1.1** | Once per call; cheap |
| `ChunkObserver` / `ChunkTransformer` | **deferred** | Per-chunk GIL overhead; discourage |

The "killer move" is **Critic + Reflector + Memory + Evaluator implementable from Python** — that covers almost every published 2023-era agent technique. Streaming in Python is a stretch goal, not a headline feature.

### 14.3 Packaging

- `oharness` on PyPI
- Maturin-based build
- Wheels for macOS (arm64, x86_64), Linux (x86_64, aarch64), Windows (x86_64)
- Python 3.10+

### 14.4 Convenience layer

- `oharness.benchmarks.swe_bench_lite()` — shortcut loader
- `oharness.trajectories.load(path)` — pandas-able trajectory reader
- `from oharness.llm import openai, anthropic, openrouter, ollama`

---

## 15. CLI (`oharness-cli`)

Binary: `openh`.

```
openh run <task-file.json>                  # run a single task
    --agents <config.toml>
    --out <dir>

openh replay <trajectory.jsonl>             # re-run agent from trajectory
    --strict | --positional

openh trace info <path>
openh trace view <path>
openh trace grep <path> <pattern>
openh trace diff <a> <b>

openh bench <subcommands above>

openh schema export                         # emit JSON Schema for event schema
openh schema validate <trajectory>          # validate against schema version
```

---

## 16. Cross-cutting concerns

### 16.1 Feature flags

Feature-gating philosophy: **default is minimal**. Users opt in to providers, tools, benchmarks.

Per-crate feature summary:

| Crate | Default features | Optional |
|---|---|---|
| `oharness-core` | (none) | `schemars-export` |
| `oharness-llm` | (none) | `derive-schema` |
| `oharness-providers` | (none — nothing enabled) | `anthropic`, `openai`, `openrouter`, `ollama`, `vllm`, `bedrock`, `vertex`, `gemini` |
| `oharness-tools` | (none) | `bash`, `fs`, `http`, `python-sandbox`, `mcp` |
| `oharness-memory` | `passthrough`, `truncate`, `elide` | `summarize`, `rag` |
| `oharness-trace` | `file-sink`, `jsonl` | `gzip`, `opentelemetry` |
| `oharness-budget` | `token`, `step` | `cost`, `wall-clock` |
| `oharness-critic` | (none) | `llm-judge`, `test-runner`, `regex-deny`, `constitutional` |
| `oharness-loop` | `react` | `conversation`, `reflexion` |
| `oharness-eval` | (none) | per-benchmark features |

### 16.2 Error handling

Every crate exports a `Error` type (e.g., `LlmError`, `ToolError`). `thiserror`-based. No `anyhow` in public APIs; allowed internally.

Top-level `AgentError` wraps everything:

```rust
pub enum AgentError {
    Llm(LlmError),
    Tool(ToolError),
    Memory(MemoryError),
    Budget(BudgetExceeded),
    Configuration(String),
    Cancelled,
}
```

### 16.3 Tracing / logging

Internal logging via `tracing` crate (separate from our event schema — `tracing` is for library debugging, `oharness-trace` is for research observability).

### 16.4 Async runtime

Tokio only. Not runtime-generic. Explicit dependency — keeps PyO3 bindings straightforward.

### 16.5 Serialization

- JSON via `serde_json` for events, task files, results
- TOML via `toml` for config files
- YAML deferred — add only if users ask

### 16.6 Determinism

Where possible:
- Seed internal RNG (sampling strategies, retry jitter)
- Document that LLM determinism depends on provider (temperature=0, seed param where supported)
- Replay mode is fully deterministic given a trajectory

---

## 17. Testing strategy

### 17.1 Unit tests

- `oharness-core`: serialization round-trip for every type; event schema versioning tests
- Each provider: mock HTTP server validating request format and response parsing
- Each helper middleware: blanket impl correctness (both complete and stream wrapped)
- `CompositeCritic`: aggregation policy correctness, parallel execution safety

### 17.2 Integration tests

- `RecordedLlm` (from `oharness-trace`) to run deterministic end-to-end tests
- Each shipped `Loop` tested against recorded trajectories
- `BenchmarkRunConfig` scenarios: resume, shard, max-cost cutoff

### 17.3 Example-as-test

Every example in `examples/` is run in CI. If examples break, build fails.

### 17.4 Schema compatibility tests

Tests that read trajectories from prior minor versions and verify they still load.

---

## 18. Documentation

### 18.1 Top-level

- README — positioning, quickstart, 10-line example
- `docs/philosophy.md` — design principles
- `docs/quickstart.md` — first agent in 5 minutes
- `docs/concepts.md` — Task, Agent, Loop, Middleware, Event, etc.

### 18.2 Per-subsystem

- `docs/llm.md` — providers, middleware, writing custom layers
- `docs/tools.md` — tool schemas, ToolContext, writing tools
- `docs/memory.md` — policies, writing custom policies
- `docs/events.md` — schema reference, versioning, replay
- `docs/critics.md` — Critic/Reflector, composition patterns
- `docs/benchmarks.md` — writing a Benchmark, running evals
- `docs/python.md` — Python bindings, implementing traits in Python

### 18.3 Examples directory

Minimum 15 runnable examples covering:
- Hello world (10 lines)
- ReAct with tool use
- Self-refine (critic)
- Reflexion (evaluator + reflector)
- Constitutional AI
- Prompt caching
- Budget enforcement
- Replay a trajectory
- Custom middleware (RequestLayer, ResponseLayer, FullLayer, ChunkTransformer)
- Custom critic
- Custom memory policy
- Multi-agent conversation
- Speculative sampling (full-control middleware)
- SWE-bench-lite runner
- τ-bench runner

### 18.4 Paper-supplement template

`docs/paper-supplement-template.md` — how to include trajectories + config in a paper supplement for reproducibility.

---

## 19. Event schema governance

### 19.1 Versioning

- Source of truth: Rust types in `oharness-core::events`
- JSON Schema exported as build artifact: `oharness-core/schema/events-v{major}.{minor}.json`
- Semver: additive = minor, breaking = major
- Support N-1 always, warn N-2, error older (post-v1)

### 19.2 Schema changes

Any PR touching event types must:
1. Bump schema version in `SchemaVersion::CURRENT`
2. Update JSON Schema export
3. Add migration code if breaking
4. Add a compat test reading prior version's trajectories
5. Document change in `CHANGELOG-schema.md`

### 19.3 The `v1.0 → v1.1` story

To earn trust, we commit publicly: **from v1.0 forward, trajectories written by any v1.x remain readable by all future v1.x viewers.** Breaking changes bump to v2. Researchers can cite a trajectory with confidence.

---

## 20. Open items / deferred decisions

### 20.1 Deferred to v1.1+

- `PlanExecuteLoop`, `ToTLoop`, `GraphOfThoughtsLoop`
- Exposing tools as MCP server
- Additional providers: Bedrock, Vertex, Gemini direct
- Additional benchmarks beyond SWE-lite and τ-bench
- OpenTelemetry sink in `oharness-trace`
- LangChain/LlamaIndex adapter (`LangChainLlm` wrapper)
- `Sampler` strategy trait (BestOf, SelfConsistency) — treated as middleware for now

### 20.2 Open questions (not blocking v1.0)

- Whether `Agent::run` should ever auto-retry on transient errors at the loop level (vs relying on LLM-layer retry middleware). Lean: no — retry is middleware territory.
- Semantics of `ConversationView` when memory policy does RAG — does the evicted content "leak" into the retrieved items? Design for v1.1.
- How to reconcile `Budget` across a multi-episode Reflexion run: shared or per-episode? Lean: user picks via `run_reflexion` config.
- Approval channel transport (CLI, web, Slack, MCP sampling) — design one clean abstraction before shipping integrations.
- Distributed execution for benchmark runners — deferred.
- `LlmError::Network(io::Error)` is not `Clone` — retry middleware that wants to log-and-rethrow needs to stringify. Revisit if it bites.
- `ReactLoop::with_prompt(...)` — expose the default prompt as a constructor parameter so ablations don't require forking. Minor API addition, deferred to M1b polish.

### 20.3 Known refactors between milestones

- **M1a → M1b: `ReactLoop` event emission.** In M1a the loop emits events directly via its `Arc<dyn EventSink>` (no middleware stack). In M1b tracing middleware (`RequestTracer`, `StreamTracer`, `ToolTracer`) takes over for LLM and tool events; the loop still emits lifecycle events (`run.started/finished`, `turn.started/finished`, `turn.revised`) directly. This is a meaningful internal refactor, not a pure addition. Planned, not a surprise.

---

## 21. Implementation order

Dependency DAG (rough layers):

```
Layer 0: oharness-core                    ← start here
Layer 1: oharness-llm (trait only, no providers)
         oharness-tools (trait only)
         oharness-memory (Passthrough + Truncate)
Layer 2: oharness-trace (sink + trajectory + replayer)
         oharness-budget
         oharness-providers (Anthropic first, OpenAI second)
Layer 3: oharness-critic
         oharness-loop (Agent + ReactLoop)
Layer 4: oharness-eval (Benchmark trait + runner)
         oharness-loop (ConversationLoop + ReflexionLoop)
Layer 5: oharness-py (bindings)
         oharness-cli
         Benchmark adapters
Layer 6: Documentation + examples
```

### 21.1 Milestones

**M1a — "Minimum viable agent"**  
`oharness-core`, `oharness-llm` (trait only), `oharness-providers` (Anthropic, **non-streaming only**), `oharness-tools` (trait + `bash` + `fs`), `oharness-memory` (Passthrough + Truncate + ElideToolResults), `oharness-trace` (FileSink, TrajectoryHandle — no replayer yet), `oharness-loop` (Agent + ReactLoop). Runs a basic tool-use agent end-to-end with full trajectory capture. No streaming, no middleware stack, no critics. The goal: prove the one-way DAG holds in practice and produce the first real trajectory file.

**M1b — "Middleware-complete"**  
Anthropic streaming + prompt caching. Full middleware system: `with_layer`/`try_with_layer`, all five helper traits (`RequestLayer`, `ResponseLayer`, `FullLayer`, `ChunkObserver`, `ChunkTransformer`). `oharness-trace` replayer with positional + strict modes. `oharness-budget` (all four budget types + middleware). Tracing middleware replaces ad-hoc event emission from M1a's loop. OpenAI provider added. Goal: equivalent to what `ought-agent` does today, but with the kernel in place.

**M2 — "Research-grade"**  
`oharness-critic` (Critic + Reflector + CompositeCritic). `oharness-loop` adds `ConversationLoop` and `ReflexionLoop`. `oharness-eval` (Benchmark trait + runner + results-dir format). SWE-bench-lite adapter. Reflexion runs end-to-end on a synthetic task. At least one real benchmark run produces a publishable trajectory set.

**M3 — "Polyglot"**  
`oharness-py` — Python bindings for `Critic`, `Reflector`, `TaskEvaluator`, `UserSimulator`, `MemoryPolicy` (the "v1" rows in §14.2). Maturin build pipeline + wheels for macOS/Linux/Windows. End-to-end example: implement a `Critic` in Python, run a full Rust-loop agent against it.

**M4 — "Production artifact"**  
Everything required to claim v1.0:
- JSON Schema export automated in CI (§19.2)
- Schema compat test suite (§17.4) reading every prior version
- Pricing table maintenance mechanism documented and usable (§10.4)
- Security review of `bash` and (if shipping) `python-sandbox` tools
- MCP consume implementation hardened (timeouts, reconnection, per-server sandboxing)
- Examples (§18.3) all building and running in CI
- Publish crates to crates.io + `oharness` to PyPI
- Blog post + τ-bench adapter + first external user

Then: **M1b's scope is large** — streaming alone is a 2-week lift, middleware composition another 2. If M1b drifts, split further rather than cramming.

### 21.2 Non-goals for v1.0

- Vector DBs / embedding providers
- Prompt template DSL (Jinja, LMQL)
- Chain/graph YAML config language
- Built-in UI (separate optional project)
- Single-prompt-style blessing (CoT, ReAct prompt text — each loop has ~30 LOC of default prompt, easy to fork)

---

## Appendix A — Summary of design decisions

A condensed checklist of the calls made during design:

- **Name / identity:** `open-harness`, crate prefix `oharness-`, CLI `openh`, env `OHARNESS_`, dual MIT/Apache-2.0
- **Task:** pure data, untyped (no `Task<T>`), attachments reference-or-inline, no budget, no environment (use metadata + tools)
- **TaskEvaluator:** separate trait from Task, paired by `Benchmark`
- **RunOutcome:** trajectory as handle (lazy), `final_messages` convenience, mid-run failure = `Termination::Failed` not `Err`, pre-computed usage scalars, no derived conveniences beyond usage
- **Event schema:** envelope with seq/run_id/span_id/parent/redactions, spans via open/close pairs, JSONL format, `meta` first, unknown preserved, positional replay default, block on backpressure, `user.log` requires namespace (no shadowing built-in prefixes); `turn.revised`, `critic.revised`, `critic.failed`, `reflection.generated/injected` all first-class events
- **Context traits in core:** `EventSink`, `BudgetHandle`, `ApprovalChannel`, `Cancellation` live in `oharness-core`; implementations live in the appropriate downstream crate. Keeps the DAG clean.
- **Llm trait:** both `complete` and `stream` no defaults, `BoxStream`, `Chunk` normalized with `Raw` escape hatch, `LlmCapabilities` by value, structured `LlmError`, partial-JSON deltas, provider extensions via namespaced `extensions` map
- **Middleware:** `Llm` is the contract, `with_layer` (infallible) + `try_with_layer` (fallible, capability-gated). Helper traits: `RequestLayer`, `ResponseLayer`, `FullLayer` (two methods: `around_complete` + `around_stream`, NOT one generic `around<T>`), `ChunkObserver`, `ChunkTransformer`. Tracing as plain middleware, ordering docs-only
- **ToolSet:** symmetric middleware, `ToolContext` with extensions map, schemars optional
- **Memory:** `MemoryPolicy::transform` returns `Result<Vec<Message>, MemoryError>`; policies emit events
- **Critic:** `CriticVerdict` with Replace semantics for Revise, `CriticTrigger::AfterAssistant` default, revision depth cap, fail-open on errors with **required `critic.failed` event** so positional replay detects drift, independent critics in composite (parallelizable)
- **Reflector:** always called, returns `Option<Reflection>`. `Episode<'a>` borrowed during iteration; `OwnedEpisode` returned from `run_reflexion`. Injected via middleware, not core loop
- **Benchmark:** lightweight `task_ids` + async `load_task`, `Workspace` with sync-or-async cleanup on Drop (explicit `teardown().await` for guarantees), async `agent_factory`, results directory as artifact, separate concurrency for load/run, max-cost lets in-flight finish
- **Conversational tasks:** `ConversationLoop<U: UserSimulator>`, simulator takes `ConversationView` (not raw `&[Message]`), user-sim errors → `Termination::Failed`
- **AgentBuilder default sink:** `NullSink` (not a file sink) — library calls don't write to cwd. File tracing is opt-in or CLI-enabled.
- **Python:** blocking-first (`run` + `arun`), v1 implementable traits: Critic, Reflector, TaskEvaluator, UserSimulator, MemoryPolicy. `ToolSet` and `Llm::complete` in v1.1; `Llm::stream` deferred or documented as slow due to per-chunk GIL. Uses `pyo3-async-runtimes` for async bridging
- **File trajectory write ordering:** payload sidecar written + fsync'd BEFORE the JSONL line referencing it. Readers warn on dangling refs; writers must not create them
- **Milestones:** M1 split into M1a (non-streaming, no middleware) + M1b (streaming + middleware + replay). M4 includes schema CI, compat tests, security review of sandboxed tools, MCP hardening

---

*End of plan.*
