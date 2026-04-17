# open-harness — remaining work (handover)

Written 2026-04-17, immediately after M1a landed (commit `eb4b03c`). This file is
a self-contained handover for an agent with no prior context. Read `README.md`
and this document, then start on M1b (§2 below).

---

## 0. Orientation (read first)

### 0.1 What this project is

A **kernel-style research framework** for LLM agent loops (think `lm-eval-harness`
for agents, not LangChain). The design spec lives alongside this file at
`docs/open-harness-plan.md` (also mirrored in the upstream `ought` repo's git
history for provenance). It was locked v1 on 2026-04-17 after three review
rounds. That document is the source of truth; this handover is secondary. Read
the plan's §21 (implementation order) and §20 (open items / deferred decisions)
before writing any code.

### 0.2 What's already built (M1a)

7 workspace crates, all clippy-clean, single integration smoke test green:

| Crate | Status | Notes |
|---|---|---|
| `oharness-core` | complete | All types from plan §4. Includes `ScopedEmitter` — see §0.4 below. |
| `oharness-llm` | **trait-only** | `Llm` + `Chunk` + `complete_from_stream`. No middleware helpers. |
| `oharness-tools` | complete (M1a kits) | `ToolSet`, `ToolContext`, `bash`, `fs`. No middleware helpers yet. |
| `oharness-memory` | M1a policies | `Passthrough`, `TruncateAfterTokens`, `ElideToolResults`. |
| `oharness-trace` | partial | `FileSink`, `InMemorySink`, `FanOutSink`, JSONL reader. **No tracing middleware, no replayer.** |
| `oharness-providers` | Anthropic only | `complete()` only; `stream()` returns `Unsupported`. No caching, no other providers. |
| `oharness-loop` | M1a `ReactLoop` | Emits events directly via `ScopedEmitter`. **Will be refactored in M1b.** |

Not present at all: `oharness-budget`, `oharness-critic`, `oharness-eval`,
`oharness-py`, `oharness-cli`.

### 0.3 Build / test / lint commands

```bash
cd /Users/aishfenton/src/open-harness

cargo build                        # full workspace build
cargo test --workspace             # all tests (currently 1)
cargo clippy --workspace --all-targets    # workspace lints; warnings are errors
```

The workspace sets `warnings = "deny"` (rust) and `clippy::all = { level = "deny" }`,
so build errors on any lint. Fix, don't `#[allow]` — but if you must, attach a
one-line comment justifying why the lint is wrong for your case (the
`clippy::result_large_err` in `oharness-core/src/context.rs` is the example
already in tree).

### 0.4 Architectural decisions you should NOT re-debate

These are already settled (either in the locked plan or by M1a's
implementation). Do not re-open without a strong reason:

- **Tokio only.** No runtime-generic code. PyO3 bindings depend on this.
- **Event envelope is `{v, seq, run_id, timestamp, span_id, parent, kind,
  redactions}`** with `kind` flattened via serde tag/content into the envelope.
  See `oharness-core/src/event.rs`.
- **Spans are open/close event pairs**, not wrappers. The close event carries
  the outcome.
- **`ScopedEmitter`** (in `oharness-core/src/context.rs`) wraps
  `Arc<dyn EventSink>` with a `RunId` and atomic `seq` counter. `MemoryContext`
  and `LoopContext` carry a `ScopedEmitter`, not a raw sink. `ToolContext` still
  carries a raw `Arc<dyn EventSink>` because tools don't self-emit in the
  current design (the loop brackets tool calls with `tool.call.*` events). The
  plan's literal struct definitions in §4.6/§7.2/§8.1 showed
  `events: Arc<dyn EventSink>` on every context — we upgraded to
  `ScopedEmitter` where subsystems actually emit. This is a pragmatic deviation
  you should preserve.
- **`Llm::complete` and `Llm::stream` are both required** — no defaults.
  Providers without streaming return `Err(LlmError::Unsupported("stream"))`.
- **`FanOutSink` in `Agent::run`** captures every event into a local
  `InMemorySink` alongside the user's configured sink, so
  `RunOutcome.trajectory` is always populated. If you keep this pattern, note
  it doubles event cloning — fine for M1a, may want to revisit in M4.
- **`AgentBuilder` default sink is `NullSink`**, not a file sink. A library
  call must not write to cwd.
- **Schema version is `1.0`.** Never bump casually — see §19 of the plan for
  the governance process. The `SchemaVersion::CURRENT` constant in
  `oharness-core/src/event.rs` is the one place to update.

### 0.5 Known gotchas from building M1a

1. **`time` crate features.** `time = { features = ["serde", "formatting",
   "parsing", "macros"] }` in workspace deps. Without `formatting`+`parsing`,
   `time::serde::rfc3339` silently disappears and you get confusing errors.
2. **`#[serde(tag = "foo")]` on an enum conflicts with a variant field also
   named `foo`.** Tripped in `Chunk` — renamed tag to `chunk` and field to
   `start`. Remember this when adding new enums.
3. **`async-trait` + lifetime on `BoxFuture<'_, ...>`.** Needs explicit lifetime
   bounds when the future borrows from `self`. Relevant for `FullLayer` in M1b.
4. **Clippy `await_holding_lock`.** Don't hold a `std::sync::Mutex` across an
   `await`. Take the value out first, drop the guard, then await — see the
   pattern in `oharness-tools/src/context.rs::Workspace::teardown`.
5. **`tokio::sync::mpsc` + sync-to-async.** The `FileSink` uses `try_send`
   first, falls back to `spawn_blocking` + `blocking_send` on full. Do NOT call
   `blocking_send` directly from a tokio worker — it deadlocks. The pattern in
   `oharness-trace/src/file_sink.rs` is the reference.
6. **`Arc<dyn EventSink>::try_emit` returns `Result<(), Event>`.** The `Err`
   variant is large (~296 bytes) and clippy complains; the trait has an
   explicit `#[allow(clippy::result_large_err)]`. The ownership hand-back is
   load-bearing — keep it.
7. **Feature-flagged re-exports.** `oharness-memory/src/lib.rs` gates re-exports
   on features; `oharness-loop/src/agent.rs` initially had the same pattern but
   it's simpler to unconditionally depend on `Passthrough` since it's a default
   feature. Prefer simple.

---

## 1. Repo hygiene before any milestone work

These are quality-of-life improvements worth doing as one small PR before M1b
proper, **in order**:

1. **Add `justfile` or top-level CI-equivalent command.** `ought` has one;
   `open-harness` doesn't yet. A `just ci` that runs
   `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
   && cargo test --workspace` is enough. Don't over-engineer.
2. **Add `rustfmt.toml`.** Default `edition = "2021"` fmt is fine. The existing
   code was formatted by Claude Code; `cargo fmt` will produce small diffs on
   first run — expected.
3. **Decide: remote + PR flow or local-only.** The repo is currently on local
   branch `main` with no remote. If the user wants a GitHub repo, `gh repo
   create aishfenton/open-harness --public --source=. --push` — but **ask
   first**, this is a visibility decision.
4. **Add `CHANGELOG.md` and `CHANGELOG-schema.md`.** Plan §19.2 mandates the
   schema changelog. Better to start empty and add entries per schema version
   than to retrofit.
5. ~~**Add the design-spec file into this repo.**~~ Done 2026-04-17 — the plan
   lives at `docs/open-harness-plan.md` in this repo. Keep it in sync with any
   upstream copies via revision PRs only.

---

## 2. M1b — "Middleware-complete" (the next milestone)

Plan §21.1 calls M1b's scope large: "streaming alone is a 2-week lift,
middleware composition another 2. If M1b drifts, split further rather than
cramming." **Split it.** Concretely:

### 2.1 M1b-α: Anthropic streaming

**Target:** `AnthropicLlm::stream()` actually returns a chunk stream instead of
`Unsupported`.

**Approach:**
1. Anthropic's streaming endpoint is the same `POST /v1/messages` with
   `"stream": true`. Response is SSE (server-sent events). Use `reqwest` with
   `.send().await?.bytes_stream()` to get a raw `Stream<Item = Bytes>`.
2. Parse SSE framing yourself — one line at a time, separated by `\n\n`, with
   `event:` and `data:` prefixes. Don't pull in `eventsource-client` unless you
   absolutely need it; adds a dep.
3. Translate Anthropic event names to `Chunk` variants:
   - `message_start` → `Chunk::MessageStart`
   - `content_block_start` → `Chunk::BlockStart` with `BlockStartKind::*`
   - `content_block_delta` (text) → `Chunk::TextDelta`
   - `content_block_delta` (input_json_delta) → `Chunk::ToolUseDelta`
   - `content_block_delta` (thinking_delta) → `Chunk::ThinkingDelta`
   - `content_block_stop` → `Chunk::BlockStop`
   - `message_delta` (usage) → `Chunk::Usage`
   - `message_stop` → `Chunk::MessageStop`
   - everything else → `Chunk::Raw { provider: "anthropic", event: <value> }`
4. Flip `capabilities().streaming` to `true`.
5. Add an integration test with a mocked HTTP server (use `wiremock` in
   dev-deps) feeding a recorded SSE response — don't hit real Anthropic in CI.

**Deliverable:** Streaming works end-to-end. `complete_from_stream` in
`oharness-llm` should round-trip the stream to the same response
`complete()` produces (property test: request X, assert
`complete(X).content == complete_from_stream(stream(X)).content`).

### 2.2 M1b-β: Middleware helper traits + `LlmExt`

Plan §5.5 and §5.6 are the spec. The key shapes:

```rust
// In oharness-llm/src/layer.rs (new module)
pub trait RequestLayer: Send + Sync {
    fn on_request(&self, req: &mut CompletionRequest);
}
pub trait ResponseLayer: Send + Sync {
    fn on_response(&self, res: &mut CompletionResponse);
    fn stream_mode(&self) -> ResponseLayerStreamMode { ResponseLayerStreamMode::WarnAndSkip }
}
#[async_trait]
pub trait FullLayer: Send + Sync {
    async fn around_complete<'a>(&'a self, req: CompletionRequest,
        call: BoxFuture<'a, Result<CompletionResponse, LlmError>>)
        -> Result<CompletionResponse, LlmError> { call.await }
    async fn around_stream<'a>(&'a self, req: CompletionRequest,
        call: BoxFuture<'a, Result<ChunkStream, LlmError>>)
        -> Result<ChunkStream, LlmError> { call.await }
}
pub trait ChunkObserver: Send + Sync {
    fn on_chunk(&self, chunk: &Chunk);
}
pub trait ChunkTransformer: Send + Sync {
    fn on_chunk(&self, chunk: Chunk) -> Option<Chunk>;
}
```

Each has a wrapper type implementing `Llm`:

```rust
pub struct WithRequestLayer<L, R> { inner: L, layer: R }
impl<L: Llm, R: RequestLayer> Llm for WithRequestLayer<L, R> { ... }
// ... same for the other four.
```

And the composition entry point:

```rust
pub trait LlmLayer<Inner: Llm> {
    type Output: Llm;
    fn wrap(self, inner: Inner) -> Result<Self::Output, LayerError>;
}
pub trait InfallibleLlmLayer<Inner: Llm>: LlmLayer<Inner> { ... }

pub trait LlmExt: Llm + Sized {
    fn with_layer<L: InfallibleLlmLayer<Self>>(self, layer: L) -> L::Output { ... }
    fn try_with_layer<L: LlmLayer<Self>>(self, layer: L) -> Result<L::Output, LayerError> { ... }
}
```

**Gotchas from the plan:**
- `FullLayer` is **two methods** (`around_complete` / `around_stream`), not
  one generic `around<T>`. Plan §5.5 explains why — don't re-unify them.
- **No `SymmetricFullLayer`** (plan §5.6.2 note). Don't introduce one.
- `ResponseLayer::stream_mode()` default is `WarnAndSkip`, and that event
  emission (`policy.layer_skipped_on_stream`) is part of the user-visible
  contract.
- Blanket `impl<L: Llm, R: RequestLayer> Llm for WithRequestLayer<L, R>`
  etc. don't conflict because each wrapper is a distinct type. But a single
  layer type that wants to be *both* `RequestLayer` and `ResponseLayer` must
  be wrapped twice (once for each role). The convenience methods
  `with_request_layer` / `with_response_layer` on `LlmExt` exist for that.

**Convenience methods on `LlmExt`** (not in plan but helpful):
```rust
fn with_request_layer<R: RequestLayer>(self, r: R) -> WithRequestLayer<Self, R>;
fn with_response_layer<R: ResponseLayer>(self, r: R) -> WithResponseLayer<Self, R>;
// ... etc for each helper trait
```

### 2.3 M1b-γ: `oharness-budget` crate

Plan §10. The interesting part is `BudgetMiddleware` — it's **not** a
`FullLayer`; it implements `Llm` directly because it needs shared state across
the pre-check, complete-post, and per-chunk hooks. The sketch is in §10.3; copy
it. Add `BudgetExceeded` where plan §10.1 says.

Feature flags per plan §16.1: default `token`, `step`; optional `cost`
(requires pricing table), `wall-clock`.

### 2.4 M1b-δ: Tracing middleware + ReactLoop refactor

Plan §9.5 says tracing middleware (`RequestTracer`, `StreamTracer`,
`ToolTracer`) lives in `oharness-trace`. Plan §20.3 explicitly flags
**M1a→M1b ReactLoop refactor**: today the loop emits `llm.request/response`
and `tool.call.*` events directly; in M1b the tracing middleware does it, and
the loop only emits lifecycle events (`run.*`, `turn.*`, `turn.revised`).

This is a meaningful internal change, not pure addition. Do it carefully:
1. Build `RequestTracer` (an `LlmLayer` that implements both `RequestLayer` and
   something that wraps `complete()` to emit response events) + `StreamTracer`
   (a `ChunkObserver`) + `ToolTracer` (a `ToolSet` wrapper).
2. Change `Agent::run` to wrap the user's `Llm` with tracing middleware
   *before* building the loop context.
3. Remove `llm.request` / `llm.response` / `tool.call.*` emissions from
   `oharness-loop/src/react.rs`. Keep lifecycle events.
4. Update the smoke test in `oharness-loop/tests/react_smoke.rs` — expected
   event set doesn't change, but the emitters do. The test should still pass.

### 2.5 M1b-ε: `ReplayLlm`

Plan §9.6. In `oharness-trace`, new module `replay`:

```rust
pub struct ReplayLlm {
    trajectory: TrajectoryHandle,
    llm_capabilities: LlmCapabilities,  // read from meta event
    mode: ReplayMode,                    // Positional | Strict
    on_drift: DriftPolicy,
}
impl Llm for ReplayLlm { ... }
```

The `meta` event's `llm_capabilities` field (plan §4.7, already emitted by
`ReactLoop::run`) is specifically there so replay can faithfully surface them.

**Positional mode:** the Nth `llm.request` event in the replayed loop pairs
with the Nth recorded `llm.response`. No input comparison.

**Strict mode:** recorded request bytes must match the current request's JSON
serialization byte-for-byte. On mismatch, emit `critic.failed`-style drift
event per `DriftPolicy` — `WarnAndContinue` (default) or `Fail`.

**stream():** reconstruct `Chunk`s from `llm.stream.chunk` events in the
trajectory (those are raw JSON payloads; you'll need a round-trip).

### 2.6 M1b providers

Per plan §6: add OpenAI next (highest demand), then OpenRouter, Ollama, vLLM.
All share the OpenAI `chat/completions` shape with provider-specific quirks.
Factor shared code into an `openai_compatible` module.

### 2.7 M1b-ζ: Prompt caching

Plan §6 / §5.7: `PromptCaching::anthropic()` is a `try_with_layer` layer —
fails construction if `inner.capabilities().prompt_caching == false`. Wires
`cache_control` breakpoints into the Anthropic request body. Currently
`AnthropicLlm.capabilities.prompt_caching = false` — flip to `true` when this
lands (and only then).

---

## 3. M2 — "Research-grade"

Plan §21.1. Three pieces:

### 3.1 `oharness-critic`

Plan §11. Contains:
- `Critic` trait + `CriticVerdict` (§11.1)
- `AssessmentContext`, `AssistantTurn`, `TrajectoryView` — the last two need
  to be **added to `oharness-core`** since they're shared (§11.1 "Supporting
  types... defined in oharness-core"). The M1a codebase hasn't defined them
  yet; it was a deferral.
- `CriticTrigger` (§11.2)
- `CompositeCritic` with `AggregationPolicy` (§11.3)
- `Reflector` trait + `Episode<'a>` + `OwnedEpisode` + `Reflection` (§11.4) —
  also add `Episode`/`OwnedEpisode` to `oharness-core`.
- `ReflectionInjector` middleware (§11.5) — a `RequestLayer` living here that
  prepends reflections.
- Shipped impls (§11.6): `LlmJudgeCritic`, `TestCritic`, `RegexDenyCritic`,
  `ConstitutionalCritic`, `LlmReflector`, `NullReflector`.

**Key behavior:** `critic.failed` is a **required event** — fail-open
behavior MUST emit it so positional replay can detect drift (plan §11.1 and
§4.7). The event is already declared in `EventKind::CriticFailed` — just wire
it.

### 3.2 `oharness-loop` additions

- `ConversationLoop<U: UserSimulator>` (§12.3–12.4): alternates agent + user
  simulator. Simulator receives `ConversationView` (not raw messages).
  Simulator errors → `Termination::Failed { reason: "user_simulator_error" }`,
  not `EndConversation`.
- `run_reflexion` helper function (§12.6). Not a `Loop` impl — a function
  that invokes an inner `Loop` repeatedly and threads reflections via
  `ReflectionInjector::set_reflections()` between episodes. The sketch in
  §12.6 of the plan is the reference; follow it.
- `Agent::injector()` accessor + `reflection_injector` field on `Agent`
  (§12.5). M1a skipped these — add now.

### 3.3 `oharness-eval`

Plan §13. Contains:
- `TaskEvaluator` + `EvaluationResult` (§13.1)
- `Benchmark` + `LoadedTask` + `Workspace` (§13.2). **Note:** `Workspace` is
  already in `oharness-tools/src/context.rs` for M1a tool scoping. Move it to
  `oharness-core` (per the plan's natural home) or re-export — don't
  duplicate.
- `BenchmarkRunConfig` (§13.3)
- `run_benchmark` runner (§13.4)
- Results directory layout (§13.5)

### 3.4 First real benchmark: SWE-bench-lite

Plan §13.6. Separate crate `oharness-bench-swe`. Loads the SWE-bench-lite
dataset (HuggingFace-hosted), constructs per-task `Workspace`es (git clone
the patch base into a temp dir), defines `TaskEvaluator` that runs the
project's tests after the agent's patch. Get at least **5 tasks** passing end-
to-end — that's the M2 gate.

---

## 4. M3 — "Polyglot" (Python bindings)

Plan §14. Crate `oharness-py`. Uses `pyo3` + `pyo3-async-runtimes`.

**v1 traits implementable from Python** (plan §14.2 priority table):
- `Critic`, `Reflector`, `TaskEvaluator`, `UserSimulator`, `MemoryPolicy`,
  `Llm::complete`

Deferred: `ToolSet` (v1.1), `Llm::stream` (v1.2 or never), `RequestLayer` /
`ResponseLayer` (v1.1), `ChunkObserver` / `ChunkTransformer` (per-chunk GIL
cost — discouraged).

**Packaging** (plan §14.3): maturin-based. Wheels for macOS arm64+x86_64,
Linux x86_64+aarch64, Windows x86_64. Python 3.10+. Publish to PyPI as
`oharness`.

The adapter pattern sketched in §14.2 of the plan (`PyCritic`, etc.) is the
reference. Do **not** try to auto-generate adapters — write them by hand per
trait; ~50 LOC each.

---

## 5. M4 — "Production artifact" (v1.0 release gate)

Plan §21.1. Each item is a gate to v1.0:

- **JSON Schema export in CI** (plan §19.2). Script emits
  `oharness-core/schema/events-v1.0.json` on each build; CI fails if the
  committed schema differs from the freshly generated one.
- **Schema compat tests** (plan §17.4). Store recorded v1.0 trajectories in
  `testdata/trajectories/v1.0/`, verify they load cleanly in every future
  v1.x.
- **Pricing table maintenance** (plan §10.4). Document how to add a new
  model's pricing without a library bump (runtime `load_from(path)` or
  `override_model`).
- **Security review of `bash` / `python-sandbox`**. `bash` tool is currently
  unsandboxed — audit. `python-sandbox` isn't shipped; if M4 adds it, it must
  be sandboxed via subprocess isolation (maybe `firejail` on Linux?). This is
  a "defer until decided" — ship without if review uncovers issues.
- **MCP `consume` hardening**. Plan §7.6 — timeouts, reconnection, per-server
  sandboxing. Not implemented in M1a.
- **Examples in CI**. Plan §18.3 lists 15 target examples. Every one runs in
  CI as `cargo build --example X && ./target/debug/examples/X`.
- **Publish**: `oharness-*` crates to crates.io, `oharness` to PyPI.
- **Blog post + τ-bench adapter + first external user**. Marketing / adoption
  gate.

---

## 6. Cross-cutting work (applies to any milestone)

### 6.1 Testing strategy reminders (plan §17)

- **Unit tests**: serialization round-trip for every public type. M1a has
  ~zero; add them as you touch each module. Aim for coverage you'd trust in a
  schema compat test down the road.
- **Integration tests**: use `InMemorySink` + a scripted `Llm` (the pattern
  is already in `crates/oharness-loop/tests/react_smoke.rs`). Add one per
  shipped `Loop`.
- **Example-as-test**: every file in `examples/` is run in CI.
- **Mocked HTTP for providers**: `wiremock`. Never hit live APIs in CI.

### 6.2 Documentation targets (plan §18)

Build these as you go, not at the end:

- `README.md` — positioning, quickstart, 10-line example. (Currently stub.)
- `docs/philosophy.md` — design principles (plan §2).
- `docs/quickstart.md` — first agent in 5 minutes.
- `docs/concepts.md` — Task, Agent, Loop, Middleware, Event.
- `docs/llm.md`, `docs/tools.md`, `docs/memory.md`, `docs/events.md`,
  `docs/critics.md`, `docs/benchmarks.md`, `docs/python.md` — per-subsystem.

### 6.3 Event schema governance (plan §19)

Every PR that touches `EventKind` or its payloads:
1. Decide: additive (minor bump) or breaking (major — starts v2+).
2. Bump `SchemaVersion::CURRENT` in `oharness-core/src/event.rs`.
3. Update JSON Schema export (once that's in place).
4. Add a compat test reading prior-version trajectories.
5. Document in `CHANGELOG-schema.md`.

Until the CI schema export exists, at minimum: update `CURRENT`, add a test,
and note it in the PR description.

### 6.4 Open questions from plan §20.2 (not blocking)

Re-surface these only if they're actively blocking your work:

- Loop-level auto-retry on transient errors (plan leans no — retry is
  middleware territory).
- `ConversationView` semantics with RAG-based memory.
- Budget sharing across multi-episode reflexion runs.
- Approval channel transport abstraction (CLI vs web vs Slack vs MCP
  sampling).
- Distributed benchmark execution.
- `LlmError::Network(io::Error)` is not `Clone` — retry layers must
  stringify.
- `ReactLoop::with_prompt(...)` — expose default prompt as constructor
  param for ablations.

---

## 7. Things a new agent should NOT do

- **Do not unify `FullLayer::around_complete` / `around_stream` into one
  generic method.** Plan §5.5 and §5.6.2 explain why. The duplication is a
  feature.
- **Do not change `EventSink::emit` to be async.** Sync emit is required so
  tools/memory/middleware can emit from non-async contexts (including `Drop`).
  The backpressure model is solved via `spawn_blocking` fallback — see plan
  §4.6 and `oharness-trace/src/file_sink.rs`.
- **Do not add framework-generic async runtime abstraction.** Tokio-only is
  intentional.
- **Do not publish crates to crates.io until M4.** Names are reserved (via
  name availability check at design-lock), but publishing before v1.0 locks
  us in.
- **Do not create documentation files or architectural decision records
  without the user asking.** The plan is the spec; this handover is the plan
  for remaining work; that's enough. Add per-subsystem docs only when
  building that subsystem.

---

## 8. Recommended starting point

When you pick this up:

1. Read the full plan at `docs/open-harness-plan.md` in this repo (~1890 lines).
2. Re-read §0.4 and §0.5 of this file for decisions and gotchas.
3. Run `cargo build && cargo test --workspace && cargo clippy --workspace
   --all-targets` to confirm M1a still works cleanly.
4. Start with **M1b-α (Anthropic streaming)** — it's the smallest standalone
   piece and unblocks M1b-ε (replayer). Open a PR (after deciding on
   remote/PR flow per §1). Keep PRs small: one M1b letter per PR.

Good luck.
