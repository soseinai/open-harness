# Changelog

All notable changes to the open-harness workspace crates are tracked here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once the v1.0 release gate (see `docs/remaining-work.md` §5 — M4) is reached.

Event-schema changes are tracked separately in [CHANGELOG-schema.md](./CHANGELOG-schema.md).

## [Unreleased]

### Added
- M1a minimum-viable agent (commit `eb4b03c`): 7 workspace crates (`oharness-core`,
  `oharness-llm`, `oharness-providers`, `oharness-tools`, `oharness-memory`,
  `oharness-trace`, `oharness-loop`). `ReactLoop` with Anthropic `complete()`
  provider, `bash` + `fs` tools, three memory policies (`Passthrough`,
  `TruncateAfterTokens`, `ElideToolResults`), `FileSink` / `InMemorySink` /
  `FanOutSink` trace sinks, and JSONL trajectory reader.
- Design spec at `docs/open-harness-plan.md` and M1b+ handover at
  `docs/remaining-work.md` (commit `f035de8`).
- Repo hygiene: `rustfmt.toml`, `justfile` (`just ci`), this changelog, and
  `CHANGELOG-schema.md`.
- **Fix**: convert `Content::Text` and `Content::Thinking` from newtype
  variants (`Text(String)`, `Thinking(String)`) to struct variants
  (`Text { text }`, `Thinking { thinking }`). Serde rejects tagged newtype
  variants wrapping a primitive, so every `llm.request` / `llm.response`
  event payload was silently dropped from the JSONL trajectory on
  serialization (`file_sink.rs` warn-and-skipped the error). The on-the-wire
  JSON shape is unchanged (`{"type":"text","text":"..."}`), so no schema
  version bump is required. Constructors `Content::text(..)` and new
  `Content::thinking(..)` keep the ergonomic call sites short. 7 new
  round-trip unit tests cover every `Content` variant plus full
  `Event::LlmRequest` / `Event::LlmResponse` envelopes.
- **M1b-δ**: tracing middleware + `ReactLoop` refactor. `oharness-trace`
  gains three types:
  - `RequestTracer` wraps `Arc<dyn Llm>`, implementing `Llm`. Emits
    `llm.request` before `complete()` / `stream()` and `llm.response` or
    `llm.failed` after `complete()`. For `stream()` it wraps each chunk
    with an inline emission that produces `llm.stream.chunk` events, so
    the streaming path never depends on the loop re-implementing the
    decoder.
  - `StreamTracer` is a standalone `ChunkObserver` that emits
    `llm.stream.chunk` events. Users composing their own middleware chain
    attach it via `LlmExt::with_chunk_observer`.
  - `ToolTracer` wraps `Arc<dyn ToolSet>`, implementing `ToolSet`. Emits
    `tool.call.started` before `execute()` and `tool.call.finished` /
    `tool.call.failed` after. Reads `tool_use_id` from
    `ToolContext.extensions["oharness.tool_use_id"]` — the new
    `TOOL_USE_ID_KEY` constant exposes this contract for other loop
    implementations.
  `Agent::run` now wraps the user's LLM and tool set in `RequestTracer` /
  `ToolTracer` before building `LoopContext`, and `ReactLoop` no longer
  emits `llm.*` or `tool.*` events itself — it only emits lifecycle
  events (`meta`, `run.*`, `turn.*`, `budget.exceeded`). The smoke test
  still sees the same event set (now from tracers instead of the loop),
  as per plan §20.3. 6 tracer unit tests (complete/response pairs,
  failure path, stream chunks, standalone observer, tool
  started/finished, tool execution-error failure) alongside the existing
  integration coverage.
- **M1b-γ**: new `oharness-budget` crate (plan §10). Concrete
  `BudgetHandle` implementations — `TokenBudget::input_plus_output`,
  `StepBudget::turns`, `CostBudget::usd` (feature `cost`),
  `TimeBudget::wall_clock` (feature `wall-clock`), and `CompositeBudget`
  (any-child-denies). `PricingTable` + `ModelPricing` with `builtin()`,
  `load_from(path)` and `override_model(..)` so pricing updates don't
  require a library bump. `BudgetMiddleware` implements `Llm` directly
  (plan §5.6.2 / §10.3) to thread one shared counter through pre-check,
  post-`complete` consume, and per-chunk observe on `stream`; consumes
  *deltas* between successive `Chunk::Usage` reports so multi-emission
  providers (like Anthropic) aren't double-counted. `BudgetExceeded` is
  wrapped in `LlmError::Provider` for `downcast_ref`-based detection.
  34 tests (8 feature-independent + 19 default + 9 under `cost`/
  `wall-clock`). Default features: `token`, `step`; optional:
  `cost`, `wall-clock`.
- **M1b-β**: middleware helper traits + fluent composition in `oharness-llm`.
  Five helper traits (`RequestLayer`, `ResponseLayer`, `FullLayer`,
  `ChunkObserver`, `ChunkTransformer`) each get a wrapper type
  (`WithRequestLayer`, …) that implements `Llm`. `ResponseLayer` streaming
  behaviour is configurable via `ResponseLayerStreamMode`
  (`WarnAndSkip` / `Error` / `SilentSkip`). `FullLayer` is intentionally
  two methods (`around_complete` / `around_stream`) rather than a generic
  `around<T>` so retry semantics stay explicit per plan §5.5.
  Bespoke layers implement `LlmLayer<Inner>` (fallible) or
  `InfallibleLlmLayer<Inner>` (infallible); `LlmExt` adds
  `with_layer` / `try_with_layer` plus direct convenience methods
  (`with_request_layer`, `with_response_layer`, `with_full_layer`,
  `with_chunk_observer`, `with_chunk_transformer`). 15 unit tests cover
  each role plus a mixed-chain smoke. `tracing` added as a direct
  dependency of `oharness-llm` for the `WarnAndSkip` log.
- **Fix**: convert `Content::Text` and `Content::Thinking` from newtype
  Events parser (no new runtime dependency — only `wiremock` added as a
  dev-dep for the fixture-backed integration tests). Anthropic events
  (`message_start`, `content_block_{start,delta,stop}`, `message_delta`,
  `message_stop`, `ping`, `error`) translate to `Chunk` variants; unknown
  event types and delta types (e.g. `signature_delta`) pass through as
  `Chunk::Raw { provider: "anthropic", .. }`. `LlmCapabilities::streaming`
  flips to `true`. 15 SSE/decoder unit tests and 3 mocked-endpoint
  integration tests (chunk sequence, `complete()` vs. `complete_from_stream`
  round-trip, capability flag).
