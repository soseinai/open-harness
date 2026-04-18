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
- **M2 part 3 — `oharness-eval` crate** (plan §13). Final slice of the
  M2 trait surface: `TaskEvaluator` moves to `oharness-core` (so it
  can be shared between the loop's `run_reflexion` and the benchmark
  runner without a crate cycle); the new `oharness-eval` crate ships
  the benchmark contract and a concurrent runner. SWE-bench-lite
  lands in its own adapter crate (`oharness-bench-swe`) later — this
  slice only ships the plumbing, plus an `InMemoryBenchmark` fixture
  for tests.
  New in `oharness-core`:
  - `TaskEvaluator` trait (`async fn evaluate(&self, &Task,
    &RunOutcome) -> EvaluationResult`). Previously duplicated as
    `oharness-loop::ReflexionEvaluator`; that local trait is gone
    and `run_reflexion` now takes `Arc<dyn TaskEvaluator>` directly.
  New `oharness-eval` crate:
  - `Benchmark` trait (`name`/`version`/`task_count`/`task_ids`/
    `load_task`/`evaluator`) + `LoadedTask { task, workspace:
    Option<Arc<Workspace>> }` + `BenchmarkError`. `Workspace` is
    re-exported from `oharness-tools` rather than duplicated.
  - `BenchmarkRunConfig` — `output_dir`, `run_concurrency` (default 8),
    `load_concurrency` (default 4), `max_cost_usd`, `filter`
    (substring), `sample_n` (prefix), `shard { index, total }`,
    `resume`. Serde-serializable (snapshotted as `config.toml` in the
    output dir). `select_ids(..)` helper composes filter → shard →
    sample deterministically.
  - `run_benchmark(benchmark, agent_factory, config)` — concurrent
    runner using two separate `tokio::sync::Semaphore`s for
    load vs run. Async factory (`Fn(&LoadedTask) ->
    impl Future<Output = Result<Agent, AgentError>>`) per plan
    §13.4. Load / factory / run errors surface as per-task
    `TaskReport` entries with `error: Some(..)` rather than
    aborting the whole run. Max-cost cutoff stops scheduling new
    tasks once cumulative cost crosses the cap; in-flight tasks
    finish. `resume` reads back outcomes from disk and folds them
    into the returned `BenchmarkReport` so the return value always
    reflects the full run.
  - Results directory layout per plan §13.5:
    `{output_dir}/config.toml`, `manifest.json`, and per task
    `{task_id}/{outcome.json, trajectory.jsonl, evaluation.json}`.
    `outcome.json` swaps the in-memory trajectory handle for a
    file-backed one pointing at `trajectory.jsonl` before serializing
    (in-memory handles refuse to serialize by design — plan §9.4).
  - `BenchmarkReport` + `TaskReport` with `pass_at_1()` helper.
  - `InMemoryBenchmark` + `AlwaysPassEvaluator` + `AlwaysFailEvaluator`
    fixtures for tests, tutorials, and harness-on-harness smoke runs.
  15 new tests: 10 unit (config knob composition, manifest
  round-trip, id sanitization, pass_at_1 on mixed results) and 5
  integration (runs every task and writes all artifacts, filter
  limits scheduled tasks, sample_n takes prefix, resume skips
  already-completed tasks without invoking the factory, factory
  errors surface as skipped tasks with `factory:` prefix). 202
  tests total on `--all-features` (was 187).

  Still outstanding: SWE-bench-lite adapter crate (the actual M2
  completion gate per plan §21.1) — that's its own piece of work
  requiring HuggingFace dataset loading + per-task git workspace
  staging.
- **M2 part 2 — loop integration** (plan §12). Wires the critic +
  reflector trait surface from M2 part 1 into the ReactLoop and ships
  a `ConversationLoop` + `run_reflexion` helper. Loop integration is the
  second of three M2 parts; part 3 (`oharness-eval` + SWE-bench-lite) is
  still outstanding and gates M2 overall.
  - `LoopContext` grows `critics: Option<Arc<CompositeCritic>>` and
    `critic_trigger: CriticTrigger`. `ReactLoop` invokes the critic
    after each assistant turn (only `AfterAssistant` is wired;
    `AfterToolResult` / `AfterEveryNTurns` / `OnDemand` deferred) and
    dispatches on `CriticVerdict`:
    - `Accept` / `AcceptWithNote` emit `critic.assessed` and continue.
    - `Reject` emits `critic.rejected` and terminates with
      `Termination::Failed { category: Critic }`.
    - `Revise` emits `critic.revised` + `turn.revised` (linking the
      replacement to the original via `TurnRevisedPayload`), swaps
      the in-history assistant message for the replacement, and
      re-invokes the critic up to `revision_depth_cap` before
      converting to `Reject` per plan §11.1.
    - `Abort` emits `critic.rejected { abort: true }` and terminates.
  - `RunErrorCategory` gains `Critic`.
  - `Agent` + `AgentBuilder` grow `critics`, `critic_trigger`, and
    `reflection_injector` fields, plus `agent.injector()` /
    `agent.critics()` / `agent.critic_trigger()` accessors per plan
    §12.5. `AgentBuilder::with_critics(..)`,
    `.with_critic_trigger(..)`, and `.with_reflection_injector(..)`
    land with sensible defaults (no critics, `AfterAssistant` trigger,
    no injector).
  - New `UserSimulator` trait + `UserAction` (`Say` /
    `EndConversation`) + `UserError` (plan §12.3). Shipped impls:
    - `ScriptedUserSimulator::new(script)` — replays a fixed sequence.
      First entry is the initial user message; subsequent responses
      walk the remaining entries; exhausted script returns
      `EndConversation`.
    - `LlmUserSimulator::new(llm, persona, prompt_template)` — drives
      a user LLM with `{persona}` / `{task}` substitutions; parses a
      configurable end sentinel (default `<end>`, case-insensitive)
      to decide `EndConversation`.
  - New `ConversationLoop<U: UserSimulator>` (feature `conversation`).
    Alternates agent assistant turns with simulator responses.
    Simulator errors surface as `Termination::Failed { category:
    UserSimulator }` — **never** as `EndConversation` per plan §12.3.
    Emits `user.simulated.message` on each simulator utterance and
    `user.simulated.ended` once on the terminating turn.
  - New `run_reflexion(agent, task, evaluator, reflector,
    max_episodes)` helper (feature `reflexion`). Returns
    `Result<Vec<OwnedEpisode>, AgentError>`. Short-circuits with
    `AgentError::Configuration` before any episode runs if the agent
    wasn't built with `.with_reflection_injector(..)` per plan §12.6.
    Threads reflections via `ReflectionInjector::set_reflections(..)`
    between episodes, emits `reflection.generated` after each reflection
    that materialized, stops on `evaluation.passed`. A local
    `ReflexionEvaluator` trait stands in for the future
    `oharness-eval::TaskEvaluator` — when the eval crate lands it will
    re-export the same shape.
  - 13 new tests: 2 critic integration end-to-end (accepting critic
    completes the run + emits `critic.assessed`; rejecting critic
    fails with `RunErrorCategory::Critic` + emits `critic.rejected`),
    1 conversation-loop end-to-end (scripted simulator drives 2-turn
    conversation, `user.simulated.{message,ended}` events present),
    5 `UserSimulator` unit tests (scripted order, empty-script error,
    LLM user initial message, LLM user say + end-sentinel paths,
    LLM error propagates through `UserError::Llm`), and 3
    `run_reflexion` unit tests (missing injector → `Configuration`
    error, evaluator passes → stops after 1 episode, evaluator fails
    → all `max_episodes` run). 187 tests total on `--all-features`
    (was 174).
- **M2 — `oharness-critic` crate** (plan §11). First slice of the
  research-grade milestone: the critic / reflector trait surface plus
  shipped implementations, with no loop integration yet. The loop-side
  work (ConversationLoop, `run_reflexion`, `Agent::injector()` accessor)
  is M2 part 2; `oharness-eval` + SWE-bench-lite is M2 part 3.
  New types in `oharness-core` (since they're shared by critic, eval,
  and the eventual reflexion loop — dependency-cleanness):
  - `AssistantTurn` + `ToolCall` — bundles a completed assistant turn
    with its span id, parsed tool calls, usage, and stop reason.
    `AssistantTurn::new(..)` auto-extracts `ToolCall`s from the message
    content.
  - `TrajectoryView<'a>` — read-only mid-run peek at the event slice,
    with `turn_count()` and `to_handle()`. Distinct from the post-run
    `TrajectoryHandle`.
  - `EvaluationResult { score, passed, details }` — `pass()` / `fail()` /
    `scored(f)` constructors, used by both critic Episode and the
    upcoming eval crate.
  - `Episode<'a>` + `OwnedEpisode` + `Reflection` — the
    `run_reflexion` iteration record.
  New `oharness-critic` crate (10 modules):
  - `Critic` trait + `CriticVerdict` (Accept / AcceptWithNote / Reject /
    Revise / Abort) + `AssessmentContext<'a>` + `CriticTrigger`
    (AfterAssistant default, AfterToolResult, AfterEveryNTurns, OnDemand).
  - `CompositeCritic` + four aggregation policies: `FirstReject`
    (sequential short-circuit), `AllMustAccept` (parallel,
    first-non-accept wins), `MajorityVote`, `Weighted(Vec<f32>)`.
    Parallel policies fan out via `futures::join_all`.
  - `Reflector` trait — always invoked per episode, returns
    `Option<Reflection>` so the reflector gates internally.
  - `ReflectionInjector` (`RequestLayer`) — threads accumulated
    reflections into the next `CompletionRequest`, either as a system
    suffix (default, appends to `req.system`) or as a prefix onto the
    first user message's first text block. `set_reflections(..)` lets
    `run_reflexion` swap the reflection list between episodes without
    rebuilding middleware. Emits `reflection.injected { episode_index,
    reflection_count, placement }` when a `ScopedEmitter` is attached.
  - Shipped impls:
    - `NullReflector` (always returns None).
    - `LlmReflector` (calls any `Arc<dyn Llm>` with a `{task} {score}
      {passed} {prior_reflections}` templated prompt; returns None on
      empty text or LLM error).
    - `RegexDenyCritic` (feature `regex-deny`; rejects if any pattern
      matches the assistant's rendered text).
    - `TestCritic` (feature `test-runner`; runs an external command via
      `tokio::process`, accepts on exit 0, rejects with the last 2KB of
      stderr otherwise).
    - `LlmJudgeCritic` (feature `llm-judge`; prompts a judge LLM with
      a rubric, parses a `SCORE: <float>` line, accepts ≥ threshold;
      fails open on parse error or LLM error).
  - `ConstitutionalCritic` deliberately deferred — its principle-based
    revision flow has richer config than this M2 slice needs.
  41 new unit tests cover each piece: aggregation policies (all four),
  parallel behavior, empty-composite accept, mismatched-weights panic
  guard, `ReflectionInjector` placements and ordered rendering,
  `parse_score` edge cases, `TestCritic` exit codes / spawn failure /
  empty command, `LlmJudgeCritic` fail-open paths. 174 tests total
  on `--all-features` (was 129).
- **OpenAI-compatible variants** (plan §6 — finishes the v1 provider
  roster alongside Anthropic and OpenAI). `OpenAiLlm` refactored to
  support the knobs these variants need without forking its code:
  - `api_key: Option<String>` — Ollama and no-auth vLLM deployments
    produce an adapter with no `Authorization` header at all.
  - `name: String` field plus `with_name(..)` builder — trajectory
    events now identify the specific provider (`"openrouter"`,
    `"ollama"`, `"vllm"`) rather than always saying `"openai"`.
  - `extra_headers: Vec<(String, String)>` plus `with_extra_header(..)`
    — OpenRouter's optional `HTTP-Referer` / `X-Title` attribution
    headers ride on this.
  - `without_auth(..)` constructor and a single `build_request` helper
    that conditionally wires bearer auth + extra headers.
  Three factories land in a new `openai_compatible` module:
  - `OpenRouter::from_env(model)` (reads `OPENROUTER_API_KEY`),
    `OpenRouter::new(api_key, model)`,
    `OpenRouter::from_env_with_attribution(model, referer, title)`.
    Targets `https://openrouter.ai/api/v1/chat/completions`.
  - `Ollama::local(model)` defaults to
    `http://localhost:11434/v1/chat/completions`; `Ollama::at(url, model)`
    for custom endpoints. No auth.
  - `Vllm::at(url, model)` (no auth), `Vllm::at_with_key(url, key, model)`
    (bearer auth).
  Each sits behind its own feature flag (`openrouter`, `ollama`, `vllm`)
  that transitively enables `openai`. 8 new unit tests cover factory
  name/URL behavior and `from_env` missing-key errors; 4 new wiremock
  integration tests in `tests/openai_compatible_wire.rs` prove the
  auth/header wiring hits the wire: bearer-auth present on
  OpenRouter + attribution headers forwarded, `authorization` header
  explicitly absent on Ollama and no-auth vLLM, bearer-auth present on
  keyed vLLM. 129 tests total on `--all-features` (was 117).
- **OpenAI Chat Completions adapter** (plan §6). New `openai` feature on
  `oharness-providers`; `OpenAiLlm::from_env()` reads `OPENAI_API_KEY` and
  defaults to `gpt-4o`. Both `complete()` and `stream()` hit
  `POST /v1/chat/completions`. Streaming auto-sets
  `stream_options: {include_usage: true}` so `BudgetMiddleware` can count
  tokens on the streaming path.
  Translation notes:
  - `Content::ToolUse` assistant blocks collapse onto OpenAI's
    `tool_calls` array; `function.arguments` is a JSON-encoded **string**
    per OpenAI's schema.
  - `Content::ToolResult` blocks on user messages expand into separate
    `role: "tool"` messages carrying `tool_call_id` — a single canonical
    user message with N tool results produces N wire messages.
  - `Content::Thinking` is dropped (o-series reasoning stays internal on
    Chat Completions).
  - Streaming reserves block index 0 for text and allocates 1.. for tool
    calls in registration order, since OpenAI's
    `choices[].delta.tool_calls[i].index` is a per-message counter
    rather than our block index.
  - `finish_reason` maps: `stop`→`EndTurn`, `length`→`MaxTokens`,
    `tool_calls`/`function_call`→`ToolUse`, `content_filter`→`Refusal`,
    unknown→`Error(raw)`.
  - `MessageStop` is synthesized at stream close (OpenAI's `[DONE]`
    sentinel is a wire-level terminator; readers of the canonical
    `Chunk` stream expect an explicit stop).
  Capabilities: `streaming: true`, `parallel_tool_use: true`,
  `vision: true`, `structured_output: true`, `thinking: false`,
  `prompt_caching: false` (automatic server-side prefix-hit discount
  isn't addressable via the request shape). 18 unit tests cover wire
  translation both directions, SSE framing, the delta-decoder state
  machine (text/tool-call registration, continuation, finish-reason
  block-close ordering, usage chunk, `[DONE]` sentinel) and
  `finish_reason` mapping; 4 wiremock integration tests cover the
  chunk sequence, `complete()` vs `complete_from_stream` round-trip
  equivalence, capability advertisement, and that the streaming request
  body always carries `stream_options.include_usage`.
- **M1b-ζ**: Anthropic prompt caching. `LlmCapabilities::prompt_caching`
  flips to `true` on `AnthropicLlm` and the `wire_messages` encoder
  honours `CompletionRequest.cache_hints`: each `CacheBreakpoint` marks
  the last content block of its target message with Anthropic's
  `cache_control: {"type": "ephemeral", "ttl": "5m" | "1h"}` marker
  (`CacheTtl::Short` → 5m, `CacheTtl::Long` → 1h, `None` → 5m default).
  `PromptCaching::anthropic()` is exposed as an `LlmLayer` that fails
  construction (`try_with_layer`) when
  `inner.capabilities().prompt_caching == false` — a construction-time
  check so a `ReplayLlm` built from a non-caching trajectory, or any
  non-Anthropic provider, can't be paired with this layer by mistake.
  `CacheTtl` is now re-exported at the `oharness-core` crate root for
  downstream ergonomics. 9 new unit tests (ttl short/long/default,
  no-op when hints empty, multi-block last-block targeting, capability
  advertise, layer accepts caching LLM, layer rejects non-caching LLM,
  factory round-trip through `PromptCaching::anthropic()`).
- **M1b-ε**: `ReplayLlm` replays a recorded trajectory as an `Llm`
  implementation (plan §9.6). Two modes:
  - `ReplayMode::Positional` (default): Nth live `complete()` / `stream()`
    returns the Nth recorded response. No input comparison.
  - `ReplayMode::Strict`: the incoming `CompletionRequest` must serialize
    byte-for-byte identically to the recorded one. Mismatch emits a
    `critic.failed`-shaped drift event (when a drift emitter is attached)
    and `DriftPolicy` decides whether to continue with the recorded
    response (`WarnAndContinue`, default) or surface an
    `LlmError::Provider(ReplayDriftError)` (`Fail`).
  Capabilities are read from the trajectory's `meta` event so
  capability-gated middleware (e.g. the eventual `PromptCaching`) can
  still wrap a `ReplayLlm` cleanly. `stream()` reconstructs `Chunk`s from
  the `llm.stream.chunk` events that sat between successive recorded
  `llm.request`s. Constructors: `from_events`, `from_path`, `from_handle`.
  11 unit tests (positional + strict, capabilities, ran-off-end,
  recorded-failure replay, drift-emitter wiring, stream reconstruction,
  missing-meta rejection) and a full record-then-replay integration test
  (`oharness-loop/tests/replay_roundtrip.rs`) that verifies a live
  `ReactLoop` run's final messages, turn count, and tool-call count all
  match when the same task is re-run against a `ReplayLlm` built from the
  captured trajectory.
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
