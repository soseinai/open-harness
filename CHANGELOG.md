# Changelog

All notable changes to the open-harness workspace crates are tracked here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once the v1.0 release gate (see `docs/remaining-work.md` §5 — M4) is reached.

Event-schema changes are tracked separately in [CHANGELOG-schema.md](./CHANGELOG-schema.md).

## [Unreleased]

### Changed
- **Python examples moved to a top-level project**
  (`python-examples/`). Previously the 11 example `.py` files
  lived at `crates/oharness-py/examples/` — nested inside the
  pyo3 crate alongside Rust source. That framing made them feel
  like internal dev scripts of a Rust crate. The new layout is
  a **vanilla Python project** with its own `pyproject.toml`
  and `README.md`, importing `oharness` as a library the way a
  real Python consumer would.
  - **`python-examples/pyproject.toml`** declares
    `oharness-examples` as a project depending on `oharness`.
    The `[tool.uv.sources]` entry points at the local
    `crates/oharness-py` path so `uv sync` builds + installs
    the wheel via maturin automatically. Once `oharness` ships
    to PyPI (post-v1.0), the `tool.uv.sources` block comes out
    and consumers install from the index.
  - **`crates/oharness-py/pyproject.toml`** now exists
    alongside `Cargo.toml` — PEP 621 metadata + maturin build
    backend, the standard pyo3 layout. Needed so downstream
    projects (and the examples) can resolve `oharness` as a
    regular Python distribution.
  - **Smoke-run recipe rewrite** — `just python-examples` now
    runs `uv sync` in `python-examples/` (which invokes maturin
    transparently via the path dependency) then
    `uv run python <example>.py` for each of the 11. No more
    manual venv bootstrap, no more `maturin develop` wrangling.
    First run is ~10s (warm cargo cache) or ~60s (cold); reruns
    are near-instant.
  - **Docs**: new `python-examples/README.md` documents the
    vanilla Python workflow (uv sync + uv run, or plain
    `pip install -e ../crates/oharness-py`). `oharness-py/README.md`
    points at the new location. Top-level `README.md` status
    table splits "Rust examples" from "Python examples".
  - **`.gitignore`**: `python-examples/.venv` and
    `python-examples/.python-version` are per-machine and
    ignored; `python-examples/uv.lock` IS committed (standard
    practice for reproducibility).

### Added
- **M3 part 5: `oharness-py` orchestration surface + 10 Python
  examples.** Closes the plan §14 v1 gate — Python users can now
  drive the full `Agent` loop end-to-end, not just write
  extension points that Rust code consumes. The adapter side
  (M3 parts 1–4) ships nine user-written trait bridges; this
  part ships the orchestration side.
  - **New pyclasses** (no `Py*` prefix in Python — they're
    first-class bindings, not adapter bridges): `Agent`,
    `AgentBuilder`, `Task`, `ReactLoop`, `ConversationLoop`,
    `FsToolSet`, `InMemorySink`, `FileSink`, `ReplayLlm`,
    `TokenBudget`, `BudgetMiddleware`, `LayeredLlm`,
    `LlmJudgeCritic`, `ReflectionInjector`, `CompositeCritic`,
    `ScriptedUserSimulator`. Plus one module-level function:
    `run_reflexion`.
  - **`AgentBuilder` surface**: `.with_llm(..)`,
    `.with_tools(..)`, `.with_loop(..)`, `.with_memory(..)`,
    `.with_critics(..)`, `.with_event_sink(..)`,
    `.with_reflection_injector(..)`, `.with_max_turns(..)`.
    Accepts any of the shipped wrapper types polymorphically —
    e.g., `with_llm` takes `PyLlm` / `ReplayLlm` /
    `LayeredLlm` / `BudgetMiddleware`; `with_loop` takes
    `ReactLoop` or `ConversationLoop`; `with_event_sink` takes
    `InMemorySink` or `FileSink`.
  - **Sync `.run()` with GIL-release**: `PyAgent.run(task)`
    blocks from Python's perspective but internally uses
    `py.allow_threads(..)` to release the GIL for the duration,
    letting Python-defined adapters re-acquire it via
    `Python::with_gil` in their `spawn_blocking` tasks. A
    shared tokio runtime (via `OnceLock`) amortises the
    runtime-construction cost across calls.
  - **Wire shape**: every `.run()` / `run_reflexion` / trimmed
    outcome return goes through `OutcomeWire` / `EpisodeWire`
    to dodge `TrajectoryHandle::Serialize`'s in-memory error.
    Consumers who want the raw trajectory attach a `FileSink`
    and read the JSONL file directly.
  - **10 runnable examples** in `crates/oharness-py/examples/`,
    mirroring 10 of the 11 Rust examples: `hello_scripted.py`,
    `react_with_tools.py`, `custom_critic.py`,
    `budget_enforcement.py`, `custom_middleware.py`,
    `custom_memory_policy.py`, `replay_trajectory.py`,
    `llm_judge_critic.py`, `reflexion_run.py`,
    `multi_agent_conversation.py`. Plus `self_refine.py` as a
    deferred stub — `CriticVerdict::Revise` requires a full
    `AssistantTurn` round-trip across the GIL that's out of
    scope for v1 (documented in the stub and the README).
  - **`just python-examples` recipe** — builds the wheel via
    `maturin develop --release` in a local `.venv` and runs
    every example. Opt-in (NOT in `just ci`), same as
    `just python-check`.
  - **Library fixes** surfaced while wiring the orchestration
    surface:
    - `impl<T: Llm + ?Sized> Llm for Arc<T>` added to
      `oharness-llm/src/llm.rs`. Symmetric with the
      `Arc<T>: RequestLayer` / `Arc<T>: ResponseLayer` impls
      from M4 examples batch 1. Without this, generic
      middleware like `BudgetMiddleware<Arc<dyn Llm>>`
      wouldn't compile.
    - `impl<T: Critic + ?Sized> Critic for Arc<T>` added to
      `oharness-critic/src/critic.rs`. Lets shipped critics
      that live behind an `Arc` (e.g., `LlmJudgeCritic`) drop
      straight into `CompositeCritic.push(Box::new(arc))`
      without an adapter shim.
    - `PyTaskEvaluator::evaluate` (M3 part 1) was serialising
      the raw `RunOutcome` — broken on in-memory trajectories
      since the start, but the Python-side `FinishedEvaluator`
      in `reflexion_run.py` is the first caller to exercise
      the path. Now uses `OutcomeWire` like `PyReflector`
      already does.
  - **Cargo.toml** gains `oharness-trace` + `oharness-budget`
    as deps, and activates `oharness-loop`'s `reflexion` +
    `conversation` features + `oharness-critic`'s `llm-judge`
    feature so the orchestration surface has everything it
    needs.
  - **README rewrite** — full orchestration section with a
    10-line "Hello agent" snippet + a table of the 10 shipped
    examples. `.gitignore` gains `crates/oharness-py/.venv/`.

### Changed
- **Plan §18.3 revised — examples-in-CI target: 15 → 11.** The
  original 15-example list was aspirational; M4 batches 1 + 2
  shipped 11 that cover the full extension surface users need
  to see. The four cut items each had a "what we'd demonstrate
  isn't shipped yet" problem rather than an authoring cost, so
  they're tracked as post-v1.0 follow-ups with per-item
  rationale in the revised §18.3:
  - **Constitutional AI** — `ConstitutionalCritic` isn't a
    concrete type yet; `LlmJudgeCritic` (example #5) covers
    the same shape with a principles-as-rubric string.
  - **Prompt caching** — needs a `wiremock`-based Anthropic
    fixture; `PromptCaching::anthropic()` layer is tested in
    `oharness-providers/tests/` already, so the incremental
    coverage is modest.
  - **Speculative sampling** — plan lists it but no canonical
    speculative-sampling layer ships; writing "the" example
    would require the framework to pick an opinion, which is
    out-of-scope for v1.0.
  - **τ-bench runner** — `oharness-bench-tau` adapter doesn't
    exist yet (parallel to `oharness-bench-swe`, future work).
  - `SWE-bench-lite runner` stays on the plan but is
    explicitly build-only in CI (needs live LLM + dataset
    download); the `oharness-bench-swe` integration test
    covers the adapter plumbing.
  - `docs/remaining-work.md §5` M4 gate list updated to mark
    the examples gate closed with a pointer to the rationale.

### Added
- **M4 docs pass — front-door docs** (plan §18.2). Closes the
  user-facing documentation gate for v1.0. Four new docs land:
  - **`README.md`** — rewritten from a stale M1a-era stub into
    a proper front door. Positioning + target audiences, a
    "what's shipped" status table, a 10-line Hello-agent
    snippet, a per-crate layout, a table of the 11 shipped
    runnable examples with one-line summaries, and pointers
    into the rest of the docs.
  - **`docs/philosophy.md`** — promotes plan §2's design
    principles into a standalone document with concrete
    examples from the shipped code. Explains why this library
    is a **kernel** rather than an integration surface (the
    LangChain-vs-this comparison), the three target audiences
    in priority order, and the eight design principles
    (small-kernel, data-oriented boundaries, composition over
    configuration, no surprise orchestration, deterministic+
    instrumented, Rust core with Python surface, provider
    honesty, fail loud). Closes with a "what open-harness is
    NOT" section that's explicit about the five most common
    false expectations.
  - **`docs/quickstart.md`** — the "first agent in 5 minutes"
    walkthrough. Six steps: add deps, run an agent with no API
    key via a scripted Llm, swap in a real provider, capture
    a trajectory to JSONL, replay it via `ReplayLlm`, then
    pointers to the 11 examples for next-layer topics (tools,
    critics, budgets, middleware, memory, reflexion).
  - **`docs/concepts.md`** — the mental model. The pipeline
    diagram (Task → Agent → Loop → RunOutcome → Events →
    EventSink → trajectory.jsonl), per-type walkthroughs of
    everything a user handles (`Task`, `Agent`, `Loop`,
    `RunOutcome`), the eight user-facing traits you extend
    (`Llm`, `ToolSet`, `Critic`, `MemoryPolicy`, `Reflector`,
    `UserSimulator`, `TaskEvaluator`, plus the middleware
    helpers), middleware composition, events + trajectory
    shape, budgets, benchmarks, and the crate DAG.
  - Cross-links wired throughout — every doc links to the
    others, and the README links to all four plus the
    pre-existing `security.md`, `pricing.md`, `RELEASE.md`,
    `CHANGELOG-schema.md`.
  - Plan §18.2's remaining targets (`docs/llm.md`,
    `docs/tools.md`, `docs/memory.md`, `docs/events.md`,
    `docs/critics.md`, `docs/benchmarks.md`, `docs/python.md`
    per-subsystem reference docs) are a future batch — the
    per-crate `README.md` files shipped in the previous
    "publish prep" commit cover the same ground at a lower
    density, and `concepts.md` covers the cross-cutting shape.

### Added
- **M4 security audit — `docs/security.md` + BashTool hardening**
  (plan §21.1 / remaining-work §5 gate). The audit closes the
  plan's "security review of `bash`" M4 gate. Three concrete
  library mitigations landed alongside the audit document.
  - **`docs/security.md`** — the trust model, per-tool risk
    inventory, mitigation status, deployer recommendations,
    deferred hardening (post-v1.0), and security-reporting
    process. The document is explicit about what open-harness
    *is* (a research framework, not a sandbox) and what it's
    *not* (safe against adversarial LLM output without an
    external isolation layer). Includes a per-risk table for
    `BashToolSet` + `FsToolSet` with severity / status columns
    users can audit in one glance.
  - **`BashTool::with_env_allowlist(..)`** — opt-in env-var
    filtering. When set, the subprocess starts with a cleared
    environment and only the named variables are copied over
    from the parent. Default (None) preserves full-env
    inheritance so existing callers keep working. Recommended
    for eval / CI: `["PATH", "HOME", "USER", "SHELL", "LANG"]`
    — enough for most tools, nothing sensitive. Hides
    `*_API_KEY`, `AWS_*`, `ANTHROPIC_*`, SSH agent sockets,
    etc. from the subprocess.
  - **Cancellation during execution** — the bash tool now
    races the child process against `ctx.cancellation` via
    `tokio::select!`. Previously the cancellation token was
    checked once at the start and never again; a long-running
    command couldn't be stopped by the agent loop. Now a
    `cancellation.cancel()` call mid-execution returns
    `ToolOutcome::Cancelled` in under a second.
  - **Windows / platform note** — `/bin/bash` is still
    hard-coded; Windows behaviour is explicitly called out in
    `docs/security.md §2.1`. A `cmd.exe /c` variant is a
    future addition, tracked in the deferred-hardening
    section.
  - 6 new unit tests in `oharness-tools/src/bash.rs`: happy
    path, timeout no-longer-leaks, cancellation-interrupts-running,
    env allowlist hides secrets, no-allowlist inherits env,
    64KiB output truncation. 239 workspace tests on
    --all-features (was 233; +6 new).

### Fixed
- **BashTool timeout leaked the subprocess** — when
  `timeout(..)` fired, the future was dropped but tokio's
  `Command::kill_on_drop(false)` default meant the spawned
  `/bin/bash` kept running in the background. A `sleep 3600 &`
  with `timeout_secs = 1` would survive the agent run
  entirely. `kill_on_drop(true)` is now set so the child
  receives `SIGKILL` when the enclosing future drops.
- **Bash subprocess inherited parent stdio** — the rewrite
  for cancellation uncovered that `Command::spawn` without
  explicit `Stdio::piped()` inherits the parent's stdout /
  stderr (unlike `Command::output()` which pipes
  automatically). Output was leaking into the harness's own
  stdio and not being captured. Fixed by explicitly piping
  both stdout and stderr, and nulling stdin so commands like
  `cat` don't block on input that never arrives.

### Added
- **M4 publish prep** — crates.io metadata polish + per-crate
  READMEs + release procedure. Gets the workspace to a
  legitimately releasable state (plan §21.1 "Publish" gate).
  - **Per-crate metadata** — every publishable crate's
    `Cargo.toml` now carries `description`, `keywords` (≤5),
    `categories` (from crates.io's category list), `readme =
    "README.md"`, plus `homepage` inherited from a new
    `[workspace.package].homepage = "..."` entry.
  - **Per-crate `README.md`** — 11 new READMEs (one per
    publishable crate; `oharness-py` already had one). Each is
    ~40 LOC and follows the same shape: what the crate is, what's
    in it, a minimal quickstart, relevant feature flags,
    dual-license note. These are what crates.io renders on the
    crate's page.
  - **`RELEASE.md`** — maintainer-facing release procedure.
    Pre-flight checks (CHANGELOG, `just ci`, `just examples`,
    schema drift), the linearised publish order (11 crates in
    topological dependency order because cargo can't dry-run
    downstream crates until upstream is published), GitHub tag
    + release notes, `maturin build` + `twine upload` for the
    Python wheel, post-flight version-bump housekeeping, and a
    troubleshooting list of the errors publish-prep actually
    surfaces (missing `description`, bad `readme` path, license
    mismatches).
  - **Dry-run validated**: `cargo publish --dry-run` passes
    end-to-end for `oharness-core` (the root of the dep graph;
    all others are blocked by the unpublished-path-deps cargo
    limitation). `cargo package --list` confirms all 11 crates
    package cleanly: total file counts land at 10–29 per crate
    (the 29 is `oharness-loop` with its 11 runnable examples).
  - `oharness-py` intentionally retains `publish = false` —
    it ships via PyPI as the `oharness` wheel, not crates.io.
    The Python package will be published separately via
    `maturin build --release` + `twine upload` per
    `RELEASE.md §3`.
- **M4 examples-in-CI batch 2** (plan §18.3 / remaining-work §5).
  Five more runnable examples in
  `crates/oharness-loop/examples/`, each smoke-run in `just ci`
  via the `just examples` recipe. Example count now **11 of 15**
  per plan §18.3; the remaining four (`prompt_caching`,
  `speculative_sampling`, `swe_bench_runner`, `tau_bench_runner`)
  need either a mocked Anthropic HTTP endpoint, a yet-unwritten
  crate, or a live-LLM eval campaign — less mechanical than the
  rest.
  - **`self_refine`** — a `ProofreadHedges` critic emits
    `CriticVerdict::Revise { replacement, reason }` when it
    detects hedge phrases in an assistant turn. The loop swaps
    the assistant message in place, emits `critic.revised` +
    `turn.revised` events, and continues — no LLM re-dispatch.
    Shows the in-place-revision path that's distinct from
    `Reject` (terminates) and from middleware-driven retries
    (re-hits the model). Default `revision_depth_cap = 3`.
  - **`llm_judge_critic`** — wraps the shipped
    `oharness_critic::shipped::LlmJudgeCritic`. A scripted judge
    LLM returns `SCORE: 0.87`, the critic compares against a
    0.75 threshold, and emits `AcceptWithNote`. Feature-gated
    behind `oharness-loop/llm-judge` (forwards to
    `oharness-critic/llm-judge`) so the default build stays
    small. The `ConstitutionalCritic` the plan §11.6 mentions is
    deferred library-side; `LlmJudgeCritic` with a
    principles-as-rubric string is the same shape and available
    today.
  - **`custom_middleware`** — composes three custom layers via
    `LlmExt`:
    - `RequestIdStamp: RequestLayer` — stamps a counter-based
      request-id into `req.extensions` (the reverse-DNS metadata
      map where `anthropic.*` / `openai.*` extensions live).
    - `RedactSecrets: ResponseLayer` — replaces `sk-live-*`
      tokens in every text block of the response.
    - `Timer: FullLayer` — async `BoxFuture`-wrapping timer that
      logs before + after every `complete()` call with elapsed
      duration. Streaming side is a pass-through (timing an
      active stream wants `ChunkObserver`, not `FullLayer`
      wrapping).
    Chain composition: `.with_request_layer(..).with_response_layer(..).with_full_layer(..)`.
    Each wrapper itself implements `Llm`, so the whole chain is
    a drop-in replacement for any `Arc<dyn Llm>`.
  - **`custom_memory_policy`** — implements the `MemoryPolicy`
    trait from scratch as `KeepLastN`. Preserves leading system
    messages, drops everything non-system except the last `n`
    entries. The trivial base case of token-budget management.
    Surfaces the `ConversationView` +
    `MemoryContext { events: ScopedEmitter, token_budget }`
    shape that all policies see.
  - **`multi_agent_conversation`** — `ConversationLoop` driven
    by `ScriptedUserSimulator` (3 pre-written user utterances).
    Alternates between a scripted assistant LLM and the
    simulator. When the user script runs out, the simulator
    emits `UserAction::EndConversation` → `Termination::Completed
    { EndTurn }`. Prints the full interleaved transcript.
    Feature-gated behind `oharness-loop/conversation`.
    Simulator errors are promoted to
    `Termination::Failed { category: UserSimulator }` — silent
    fall-to-end would hide simulator bugs in research logs.
  - `oharness-loop/Cargo.toml` gains two
    feature-forwarding flags: `llm-judge =
    ["oharness-critic/llm-judge"]` and (already existed)
    `conversation = []` / `reflexion = []` pattern. The three
    gated examples use `required-features = [...]` with local
    feature names, since cargo doesn't accept `crate/feature`
    syntax there.
  - `just examples` grows 5 new `cargo run` invocations + 2
    extra `cargo build --features ...` lines
    (`oharness-loop/conversation`, `oharness-loop/llm-judge`)
    so feature-gated build regressions can't hide behind the
    default-feature build. All 11 examples now smoke-run on
    every `just ci` invocation.
  - 233 workspace tests on `--all-features` still green (no new
    tests this commit — the examples themselves are the tests).

### Fixed
- **`FileSink::flush` deadlock with outstanding `Arc` clones**
  (discovered while writing the `replay_trajectory` example in M4
  examples batch 1). The writer task drained events via
  `while let Some(event) = rx.recv().await`, which only returns
  `None` when *every* `Sender<Event>` has dropped. Since
  `flush(&self)` cannot drop `self.tx`, any caller holding an
  `Arc<FileSink>` clone at flush time would hang forever on the
  awaited `JoinHandle`.
  - **Fix**: added an internal `tokio::sync::oneshot` close
    channel. The writer loop now `select!`s between `rx.recv()`
    and `&mut close_rx`; when the close signal fires (sender
    dropped), the writer drains any remaining queued events via
    `rx.try_recv()` and finalises the file. `flush()` takes the
    sender out of an internal `Mutex<Option<_>>`, drops it to
    fire the signal, then awaits the writer as before. Idempotent
    — second `flush()` call is a no-op `Ok(())`.
  - **Regression test**: `flush_completes_with_outstanding_arc_clones`
    wraps a 2-second `tokio::time::timeout` around the flush call
    while deliberately holding an `Arc<FileSink>` clone alive; the
    fixed version completes in single-digit milliseconds, the
    old version hung indefinitely.
  - Two more tests: `flush_is_idempotent` (back-to-back calls
    must both return `Ok`) and `emit_after_flush_is_warned_not_panicked`
    (post-flush emits hit the existing `TrySendError::Closed`
    warn-drop path without panicking).
  - `crates/oharness-loop/examples/replay_trajectory.rs` reverts
    to the natural `FileSink::to_path(...) + sink.flush().await`
    flow. The earlier workaround (capture into `InMemorySink`,
    then serialize to disk by hand with `serde_json::to_string` +
    `writeln!`) is gone; the example is ~15 LOC shorter and
    matches what the `FileSink` docstring has always promised.
  - `oharness-trace/Cargo.toml` gains `time` as a regular dep
    (was transitively available but not explicitly listed) and
    a `[dev-dependencies]` section with `tokio` (macros + time)
    + `tempfile = "3"` for the new tests.
  - 233 workspace tests on `--all-features` (was 230; +3 new
    `FileSink` tests).

### Added
- **M4 examples-in-CI batch 1** (plan §18.3 / remaining-work §5). Five
  new runnable examples land in `crates/oharness-loop/examples/`,
  each built + smoke-run in `just ci` via the `just examples`
  recipe. Plan §18.3 specifies 15 examples as the M4 gate; this
  commit brings the count from 1 (just `hello_scripted`) to 6.
  All five use a scripted `Llm` so CI runs them with no API key,
  no network, and no cost — swap in `AnthropicLlm` / `OpenAiLlm`
  and nothing else changes.
  - **`react_with_tools`** — scripted multi-turn ReAct run that
    actually dispatches a tool call (`fs_list` via `FsToolSet`)
    and threads the result back. The canonical "ReAct + tool use"
    demo sibling to `hello_scripted`.
  - **`custom_critic`** — `NoHedgingCritic` implements the `Critic`
    trait from scratch (scans assistant text blocks for hedge
    phrases, emits `CriticVerdict::Reject`). Shows what the `Reject`
    surface looks like at the loop layer: `Termination::Failed {
    category: Critic }` + a `critic.rejected` event on the
    trajectory. The template for user-written critics.
  - **`budget_enforcement`** — `BudgetMiddleware::new(inner,
    Arc<TokenBudget>)` caps a run at 50 input+output tokens; the
    scripted LLM's 300-token response trips the pre-call check,
    yielding `Termination::Failed { category: Llm }` before the
    call actually dispatches. Shows the budget `.snapshot()` API
    for per-task telemetry independent of the trajectory events.
  - **`replay_trajectory`** — records an agent run into an
    `InMemorySink`, writes the captured events to a JSONL file
    (the on-disk format external tooling consumes), rebuilds the
    same agent configuration against `ReplayLlm::from_events(...,
    ReplayMode::Positional, DriftPolicy::default())`, re-runs,
    and asserts every `RunOutcome` field matches. Demonstrates
    bit-for-bit reproducibility without provider API keys or
    dollars — the paper-supplement workflow.
  - **`reflexion_run`** — `run_reflexion` over 5 max episodes with
    a `NudgeReflector` that emits "be concrete, say 'done!'" and a
    `FinishedEvaluator` that fails unless the assistant's final
    message contains "done". Scripts three LLM responses (two
    hedging, one that says "done!"); the loop iterates exactly 3
    episodes before the evaluator passes and stops the sweep.
    Each episode's `prior_reflections.len()` prints as evidence
    the notes feed forward via `ReflectionInjector`. Gated behind
    the `reflexion` feature via `required-features` in
    `Cargo.toml`, so `cargo run --example reflexion_run -p
    oharness-loop --features reflexion` is the invocation.
  - Two library fixes surfaced by writing the examples:
    - `impl<T: RequestLayer + ?Sized> RequestLayer for Arc<T>` (+
      symmetric `ResponseLayer` impl). The docs on
      `Agent::with_reflection_injector` promised
      `LlmExt::with_request_layer(injector.clone())` would work
      with `Arc<ReflectionInjector>`, but no blanket impl existed.
      Four-line fix in `crates/oharness-llm/src/layer.rs`; now the
      documented idiom compiles.
    - `justfile` `examples:` recipe grows the five `cargo run`
      invocations + a second `cargo build --examples
      --features oharness-loop/reflexion` line to catch any
      feature-gated build regression that the default-feature
      build would miss.
  - **Spawned as a follow-up** (separate chip in the task sidebar):
    `FileSink::flush` deadlocks if any `Arc<FileSink>` clone is
    alive when flush is called, because the underlying writer
    task only exits when all senders drop. The `replay_trajectory`
    example had to work around this by using `InMemorySink` +
    `serde_json::to_string` to disk; once the flush fix lands,
    the example will revert to `FileSink::to_path` + `flush()`.
  230 workspace tests on `--all-features` still green.
- **M3 part 4: `oharness-py` adapters for `RequestLayer` +
  `ResponseLayer`** (plan §14.2). Middleware traits complete the
  v1 Python surface — nine of the ten plan-§14.2 traits are now
  live. Only `Llm::stream` + `ChunkObserver`/`ChunkTransformer`
  remain deferred (both blocked on GIL-vs-streaming design
  questions).
  - `PyRequestLayer(py_obj, name=...)` — expects `on_request(req_json:
    str) -> str`. Python returns a full-shape `CompletionRequest`
    JSON; the Rust side replaces the outgoing request in place
    with the deserialized result.
  - `PyResponseLayer(py_obj, name=..., stream_mode=...)` — expects
    `on_response(res_json: str) -> str`. Python returns a
    full-shape `CompletionResponse` JSON; in-place replacement
    same as above. `stream_mode` is a string argument:
    - `"warn_and_skip"` (default) — log once per wrapper, pass
      chunks through unchanged.
    - `"error"` — `stream()` returns `LlmError::Unsupported`.
    - `"silent_skip"` — pass chunks through without logging.
    Other values raise `ValueError` at construction.
  - **Sync-in-async note** — unlike the seven async adapters,
    `RequestLayer` / `ResponseLayer` are sync traits (`fn
    on_request(&self, req: &mut CompletionRequest)`). The Python
    call happens **synchronously under the GIL** from inside the
    async `complete()` / `stream()` task. Fine for cheap layers
    (redaction, header injection, metadata merging); users who
    need heavy Python work should compose it outside the layer
    chain. Using `tokio::task::spawn_blocking` here would require
    blocking the current task on a `JoinHandle`, which defeats
    the point.
  - **Fail-open on errors** — any Python exception, bad JSON, or
    bad shape logs to stderr (`PyRequestLayer(name): ...`) and
    leaves the request / response unchanged. A broken layer must
    not crash the run; the unmodified value still reaches the
    next stage.
  - **`ResponseLayer::name()` caveat** — the trait method returns
    `&'static str`, which our adapter's owned `String` name
    doesn't satisfy, so the trait falls back to the default
    (type name). The user-supplied name is still exposed via
    `__repr__` and is useful for Python-side inspection. A future
    API-break could widen the trait method to `&str`.
  - `#[pymodule]` registers `PyRequestLayer` + `PyResponseLayer`.
    README adds `inject-request-id` + `redact-secrets` examples
    with full wire notes and the sync-in-async / fail-open
    caveats. Scope table flips `RequestLayer` / `ResponseLayer`
    from `⏳ v1.1` to `✅ v1`.
  - 230 workspace tests on `--all-features` still green.
- **M3 part 3: `oharness-py` adapter for `ToolSet`** (plan §14.2).
  Python-side classes can now ship as first-class tools inside
  Rust-driven agent runs. Seven of the ten plan-§14.2 traits are
  now live; three remain deferred.
  - `PyToolSet(py_obj, specs_json, name=...)` — expects
    `execute(name: str, input_json: str, ctx_json: str) -> str`
    returning a JSON-encoded `ToolOutcome`. **Specs are fixed at
    construction time** (passed in as a JSON array of `ToolSpec`,
    deserialized once, stored as `Vec<ToolSpec>`, returned by
    `specs()` as a slice). This avoids round-tripping through
    Python on every turn just to enumerate tools — the loop reads
    `specs()` once per request.
  - **Wire shapes for the `execute` return value** (snake_case
    tagged union, matching `WireToolOutcome`):
    - `{"outcome":"success","output":{...ToolOutput...}}` —
      full-fidelity success.
    - `{"outcome":"success_text","text":"..."}` — convenience
      variant equivalent to `success` with a single text
      `ToolOutput` block; handy for the common "tool returns one
      string" case.
    - `{"outcome":"execution_error","message":"...","recoverable":false}`
    - `{"outcome":"denied","reason":"..."}`
    - `{"outcome":"cancelled"}`
  - **`ToolContextWire`** is trimmed: only `workspace_path`
    (optional) + `extensions` (reverse-DNS metadata map) cross the
    boundary. `EventSink`, `BudgetHandle`, `Cancellation`,
    `ApprovalChannel` are Rust-runtime types that can't usefully
    be serialized to Python in v1 (same pattern as
    `PyMemoryPolicy`'s `ScopedEmitter` handling).
  - **Error handling**: any Python exception, malformed JSON, or
    bad-shape response is promoted to `ToolOutcome::ExecutionError
    { recoverable: false }` with the bridge error as the message.
    The loop sees the failure via `tool.call.failed`; the agent
    continues and sees the error as the tool result. Unlike
    `PyMemoryPolicy` errors (fatal for the turn), tool errors are
    recoverable signals the agent can reason about.
  - **Python-side utility**: `toolset.tool_names()` returns the
    list of registered tool names for quick inspection.
  - `#[pymodule]` registers `PyToolSet`; `oharness-py/Cargo.toml`
    gains `oharness-tools` as a path dep. README grows a
    `reverse`-tool example, full wire-shape reference, and the
    scope table flips `ToolSet` from `⏳ v1.1` to `✅ v1`.
  - Remaining deferred per plan §14.2: `Llm::stream` (v1.2+),
    `Request/ResponseLayer` (v1.1), `ChunkObserver` /
    `ChunkTransformer` (discouraged by per-chunk GIL cost).
  - 230 workspace tests on `--all-features` still green.
- **M3 part 2: `oharness-py` adapters for Reflector / UserSimulator /
  MemoryPolicy** (plan §14.2). Three more traits join the v1
  adapter surface; the JSON-wire + `tokio::task::spawn_blocking`
  pattern from part 1 carries over verbatim.
  - `PyReflector(callable, name=...)` — expects `reflect(episode_json:
    str) -> Optional[str]`. Python returns `None` (or the literal
    `"null"`) to skip the episode, or a JSON `{"text", "metadata"}`
    to emit a [`Reflection`]. `created_at` is stamped on the Rust
    side so Python authors don't have to emit valid RFC-3339. Errors
    `eprintln!` and return `None` — a broken reflector must not
    break the reflexion sweep.
  - `EpisodeWire`: the episode passed into Python is a trimmed view
    carrying `index`, `task`, `outcome` (without the
    `TrajectoryHandle` — in-memory handles refuse to serialize, and
    file handles are useless to Python anyway), `evaluation`,
    `prior_reflections`. `OutcomeWire` keeps `run_id`, `task_id`,
    `termination`, `final_messages`, `usage`.
  - `PyUserSimulator(callable, name=...)` — two methods, one per
    trait entry point:
    - `initial_message(task_json: str) -> str` — bare string, the
      first user turn.
    - `respond(conversation_json: str, task_json: str) -> str` —
      returns `{"action": "say", "message": "..."}` or
      `{"action": "end_conversation"}`.
    - **Not fail-open**: simulator errors promote to
      `UserError::Other`, which the `ConversationLoop` turns into
      `Termination::Failed { reason: "user_simulator_error" }`.
      Unlike critics, hiding simulator bugs behind a silent
      `EndConversation` would break eval reproducibility.
  - `PyMemoryPolicy(callable, name=...)` — expects
    `transform(conversation_json: str, ctx_json: str) -> str`
    returning a JSON `Vec<Message>`. The `ctx_json` carries only
    `{"token_budget": N}` — **Python memory policies cannot emit
    `memory.*` events in v1** (the `ScopedEmitter` doesn't cross the
    boundary). Documented as a known limitation; future work may
    grow a return-side events channel. Errors promote to
    `MemoryError::Configuration` (treated as fatal for the turn) —
    a corrupted context window is worse than a failed run.
  - `oharness-py/Cargo.toml` gains `oharness-memory` + `oharness-loop`
    as path deps (for the trait definitions only — no runners
    linked).
  - `#[pymodule]` registers `PyReflector`, `PyUserSimulator`,
    `PyMemoryPolicy` alongside the originals. `__version__` still
    tracks crate version.
  - `README.md` grows usage examples for each of the three new
    adapters and the v1 scope table flips the three from `⏳` to
    `✅`. Remaining deferred per plan §14.2: `Llm::stream`
    (v1.2+), `ToolSet` (v1.1), `Request/ResponseLayer` (v1.1),
    `ChunkObserver`/`ChunkTransformer` (discouraged by per-chunk
    GIL cost).
  230 workspace tests on `--all-features` still green — the
  bindings ship no tests yet (a Python interpreter is required to
  meaningfully exercise them; that's a future task once a
  pytest-based harness lands).
- **M3 part 1: `oharness-py` Python bindings scaffold** (plan §14). A
  new crate at `crates/oharness-py/` ships the adapter pattern that
  lets Python code plug into Rust-side agent runs. Imported as
  `import oharness`; built via `maturin develop --release`.
  - **Workspace isolation**: the crate is intentionally **excluded
    from workspace membership** (`Cargo.toml` top-level `exclude =
    ["crates/oharness-py"]`). `maturin develop` needs Python headers
    which aren't available on every CI runner, so `just ci` skips
    this crate by design. Path deps from `oharness-py` into the
    workspace still resolve. Contributors working on the bindings
    run `just python-check` (opt-in) to lint the Rust side without
    needing maturin.
  - **pyo3 0.24 with `abi3-py310` + `extension-module`**: the stable
    ABI for Python 3.10+ means pyo3 bundles the ABI stubs, so
    `cargo check` / `cargo clippy` work without `python-devel`
    headers; the extension-module feature skips linking libpython
    (the host interpreter resolves symbols at load time).
  - **Three adapters, one pattern** — each wraps a Python object
    implementing one method; the wire type between Rust and Python
    is always a **JSON-encoded string** (not a structured `dict`),
    so the serde codec on the Rust side stays canonical:
    - `PyLlm(callable, name=...)` — expects `complete(req_json: str)
      -> str` returning a `CompletionResponse` JSON. Implements
      `oharness_llm::Llm::complete`; `stream()` returns
      `LlmError::Unsupported("stream")` (streaming from Python is
      v1.2+ per plan §14.2). The sync Python method runs under
      `tokio::task::spawn_blocking` so blocking IO in Python
      doesn't stall the async runtime.
    - `PyCritic(callable, name=...)` — expects `assess(ctx_json:
      str) -> str` returning a verdict JSON with shapes
      `{"verdict":"accept"}`, `{"verdict":"accept_with_note","note":"..."}`,
      `{"verdict":"reject","reason":"..."}`, or
      `{"verdict":"abort","reason":"..."}`. Implements
      `oharness_critic::Critic::assess`. **Fail-open**: any Python
      exception or JSON decode error returns `AcceptWithNote` with
      the error message, matching plan §11.1. `revise` is
      intentionally unsupported from Python — the replacement
      `AssistantTurn` shape is non-trivial, so Python critics emit
      `reject` and let the loop's retry path regenerate.
    - `PyTaskEvaluator(callable)` — expects `evaluate(task_json:
      str, outcome_json: str) -> str` returning a
      `TaskEvaluationResult` JSON (`{score, passed, details}`).
      Implements `oharness_core::TaskEvaluator::evaluate`.
  - **GIL discipline**: `PyObjectExt::clone_ref_unbound_gil()`
    (implemented via `Python::with_gil`) is used to ship
    `Arc<PyObject>` handles across `spawn_blocking` boundaries
    without holding the GIL on the async side.
  - **`#[pymodule] fn oharness(m: &Bound<'_, PyModule>)`** registers
    the three classes plus `__version__` (matches crate version).
    `PyBridgeError` (via `thiserror`) maps internal failures to
    Python exceptions cleanly.
  - **Docs**: `crates/oharness-py/README.md` with maturin build
    steps, a full end-to-end example for each of the three
    adapters, a priority table per plan §14.2 showing what's live
    (`Llm::complete`, `Critic::assess`, `TaskEvaluator::evaluate`)
    and what's deferred (`Reflector`, `UserSimulator`,
    `MemoryPolicy`, `Llm::stream`, `ToolSet`,
    `RequestLayer`/`ResponseLayer`, per-chunk observers).
  - **`just python-check` recipe** — opt-in, NOT part of `just ci`.
    Runs `cargo check` + `cargo clippy --all-targets -- -D
    warnings` from `crates/oharness-py/`. Contributors touching the
    bindings run this; CI stays green without Python.
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
- **M4 plumbing part 2**: JSON Schema export for the trajectory
  `Event` envelope (plan §19.2). The committed file
  `crates/oharness-core/schema/events-v1.0.json` is the canonical
  machine-readable shape; CI diffs a fresh export against it and
  fails on any divergence.
  - New optional `schemars` workspace dep (non-default).
    `oharness-core` grows a `schemars-export` cargo feature gating
    the JsonSchema derives. Default builds pay nothing for
    `schemars`.
  - `schemars::JsonSchema` derived via `#[cfg_attr(feature =
    "schemars-export", derive(schemars::JsonSchema))]` on every
    Event-reachable type: `Event`, `EventKind` (with all 36
    variants), 13 payload structs, `Message` / `Content` (+
    `ToolOutput` / `ImageRef` / `DocumentRef` / `AudioRef` /
    `CitationRef`), `Task` / `Attachment`, `CompletionRequest` /
    `CompletionResponse` / `StopReason` / `ToolSpec` / `CacheHints` /
    `CacheBreakpoint` / `CacheTtl` / `Usage`, `LlmCapabilities`.
    Non-derivable field types (`OffsetDateTime`, `Url`, `PathBuf`)
    carry `#[schemars(with = "String")]` so the schema reports them
    as strings (matching serde's wire shape).
  - Manual `JsonSchema` impls for `SchemaVersion` (string with a
    `^\d+\.\d+$` pattern constraint, matching its custom Serialize)
    and `RunId` (UUID-format string). `SpanId` / `ModelId` use
    `#[schemars(transparent)]` to inherit the inner `String` schema.
  - `examples/export_schema.rs` regenerates the baseline and writes
    `schema/events-v1.0.json`. The default stub-main (when feature
    isn't enabled) `eprintln!`s the right cargo invocation and
    exits non-zero — no silent no-op.
  - `tests/schema_up_to_date.rs` loads the committed baseline via
    `include_str!`, regenerates in-memory, compares. On mismatch it
    prints a context-annotated diff (3 lines before/after the first
    divergent line) + the correct `just schema-export` command —
    no hand-holding required to regenerate.
  - `just schema-check` / `just schema-export` recipes + `just ci`
    gains `schema-check` as a mandatory stage.
  - `CHANGELOG-schema.md` adds an "Unreleased" section noting the
    export is live and the governance rule: any schema-affecting
    change MUST come with a `SchemaVersion::CURRENT` bump + a
    matching changelog entry.
  230 tests total on `--all-features` (was 229; +1 drift test).
- **M4 plumbing part 1**: schema-compat test, first example, pricing
  docs. Three of the M4 gates land in this commit; JSON-Schema export
  via `schemars`, examples-in-CI at the 15-file scale, and crates.io
  publish prep are separate chunks.
  - `crates/oharness-core/testdata/trajectories/v1.0/smoke.jsonl` —
    an authoritative v1.0 trajectory fixture with 11 events covering
    `meta`, `run.{started,finished}`, `turn.{started,finished}`,
    `llm.{request,response,failed}`, and
    `tool.call.{started,finished}`. Generated via
    `crates/oharness-core/examples/gen_v1_fixture.rs` (the generator
    stays in-tree as documentation for how to rebuild the baseline
    if v1.0 ever legitimately needs regeneration).
  - `crates/oharness-core/tests/v1_compat.rs` — 6 tests that load
    the fixture and verify shape invariants (deserializes cleanly,
    starts with `meta`, carries expected `EventKind` variants, seqs
    are monotonic, one shared `run_id`, all events declare schema
    v1.0). Per plan §17.4 this is the lower bound: prior-version
    fixtures MUST continue to parse when the schema bumps to v1.1 /
    v2.0.
  - `crates/oharness-loop/examples/hello_scripted.rs` — the "first
    agent in 10 lines" entry-point example. Scripted LLM + FsToolSet
    + default ReactLoop; no real API, no cost. Prints termination +
    final assistant message.
  - `just examples` target added; `just ci` now builds `--examples`
    across the workspace and smoke-runs the safe ones (today just
    `hello_scripted`; the fixture generator is built-but-not-run so
    CI doesn't rewrite the committed testdata).
  - `docs/pricing.md` documents the three paths for updating
    `BudgetMiddleware` pricing without a library bump —
    `with_pricing(..)` at runtime, `PricingTable::load_from(path)`
    via JSON file, or a PR against `PricingTable::builtin()`.
  229 tests total on `--all-features` (was 223; +6 compat).
- **Workspace scoping through the loop** — the shipped `FsToolSet` and
  `BashTool` were already workspace-scoped, but the loop was
  hardcoding `ToolContext.workspace` to `None` on every tool call, so
  the scoping never kicked in. Fixes the bug by threading a workspace
  handle through `Agent` → `LoopContext` → `ToolContext`:
  - `LoopContext` gains `workspace: Option<Arc<Workspace>>`.
  - `Agent` + `AgentBuilder` gain a matching field and the new
    `.with_workspace(Arc<Workspace>)` builder method plus a
    `Agent::workspace()` accessor.
  - `react.rs::execute_tool_calls` populates
    `ToolContext.workspace` from `ctx.workspace` (instead of always
    `None`).
  - `ReactLoop` now scopes `fs_read` / `fs_write` / `fs_list` / `bash`
    to `Agent`'s workspace automatically; agents without a workspace
    fall back to cwd (unchanged M1a behaviour).
  3 new end-to-end tests in `workspace_scoping.rs`: `fs_read` reads
  inside the workspace, `fs_read` refuses `../`-escapes with a
  non-recoverable tool error (and no secret leakage), and
  agents without a workspace still read from cwd. This unblocks
  step 2 of the SWE-bench run-it recipe in
  `docs/remaining-work.md §3.4` — benchmark factories just call
  `.with_workspace(loaded.workspace.unwrap())` now.
- **M2 part 4 — `oharness-bench-swe` adapter crate** (plan §13.6). The
  first real-benchmark adapter plumbing: dataset types, git-workspace
  staging, patch apply, and FAIL_TO_PASS / PASS_TO_PASS grading. The
  ≥5-passing M2 completion gate is NOT hit by this commit — it's an
  eval campaign requiring a live LLM, per-repo Python envs, and money;
  see docs/remaining-work.md §3.4 for the run-it recipe.
  - `SweBenchInstance` deserializes one dataset record. `FAIL_TO_PASS`
    / `PASS_TO_PASS` accept both SCREAMING_CASE (canonical) and
    snake_case (for hand-authored fixtures) on the way in and emit
    SCREAMING_CASE on the way out.
  - `SweBenchLite::from_jsonl(path)` loads a dataset dump from disk
    (one JSON record per line). HF-hub fetch deferred.
  - `Benchmark` impl: `load_task` shells out to `git clone` +
    `git checkout base_commit` in `{clone_root}/{instance_id}/`,
    returns a `LoadedTask` whose `Workspace.path` is the cloned repo.
    The `Task.instruction` bundles the problem statement with an
    orientation blurb; the full instance record is stashed on
    `Task.metadata["swe-bench.instance"]` so critics / reflectors /
    the evaluator can pull any field they need.
  - `SweBenchEvaluator` implements `TaskEvaluator`:
    `git apply test_patch`, runs a configurable test command
    (default `pytest -v --tb=short --no-header`; override via
    `.with_test_command(..)` for stubs / different runners),
    parses per-test outcomes, grades pass iff all FAIL_TO_PASS +
    PASS_TO_PASS ids pass. `details` carries the outcome map, the
    specific ids that regressed / went missing, and a 4KB tail of
    raw pytest output for post-run inspection.
  - Hand-rolled pytest parser: forgiving of `-v` output with color,
    ignores short-summary failure recap lines, respects last-seen
    status for retry-plugin scenarios.
  - Dataset-side path heuristic: `repo` field passes through when it
    looks like a URL (`http://`, `https://`, `git@`) or an absolute
    local path (`/tmp/upstream.git` — used by the test fixture);
    otherwise treated as a GitHub `owner/name` slug and expanded to
    `https://github.com/owner/name.git`.
  - 16 unit tests (dataset loader, URL heuristic, id sanitization,
    task-metadata stashing, pytest parser edge cases, grading logic)
    + 2 synthetic-fixture integration tests that spin up a local git
    repo and drive the full benchmark → evaluator flow using `cat` as
    the test command (no pytest / python required on the host).
    220 tests total on `--all-features` (was 202).
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
