# Philosophy

*Why this library exists, what shape it takes, and what it
explicitly is not.*

## The kernel stance

`open-harness` is a **kernel** for agent loops: a small core with
sharp extension points, not a broad integration layer.

The comparison:

| If you want…                                   | Reach for…            |
|------------------------------------------------|-----------------------|
| "Wire up 50 tools, 20 LLMs, and a dozen apps." | LangChain / LlamaIndex|
| "A canonical eval loop I can extend with a new technique." | `open-harness` (this library) |
| "Just call an LLM."                            | the provider's SDK    |

This isn't a beauty contest. LangChain serves a different
audience well. `open-harness` serves researchers and eval teams
who need a small, predictable surface they can build on without
fighting the framework's opinions.

## Target audiences, in priority order

1. **Agent researchers** — publishing Reflexion++, novel memory,
   planners, critics, new orchestration strategies. Care about
   reproducibility, trajectory inspection, and clean extension
   points.
2. **Eval / safety teams** — running agents on SWE-bench, τ-bench,
   GAIA. Care about cost control, deterministic replay,
   per-task isolation, and rigorous result tracking.
3. **Practitioners** — shipping production agents. Care about
   observability, streaming, backpressure, and not being
   surprised by the framework's defaults.

## Design principles

These are the architectural commitments every piece of the
library is judged against. They're plan §2's eight principles,
expanded with concrete examples from the shipped code.

### 1. Small kernel, big periphery

`Llm` has two methods. `ToolSet` has two. `Loop` has one
(`run`). `Critic` has one (`assess`). `MemoryPolicy` has one
(`transform`). `Reflector` has one (`reflect`).

Everything else composes:
- Middleware wraps `Llm` via `LlmExt::with_layer(..)`.
- Critics compose via `CompositeCritic + AggregationPolicy`.
- Budgets compose via `CompositeBudget`.
- Tools compose by... not composing — you write a new `ToolSet`
  that wraps multiple concrete tool collections.

Resist the urge to grow trait surface. Ship a helper crate
instead.

### 2. Data-oriented boundaries

Every phase boundary emits a **serializable event**. There are
36 `EventKind` variants in the v1.0 schema; every subsystem
(LLM, tools, memory, budget, critic, reflexion, user sim)
announces its state changes through the same `EventSink`
abstraction.

The consequence: a completed run is fully described by its
trajectory JSONL. `ReplayLlm` re-drives an agent against that
JSONL with byte-for-byte fidelity. Paper-supplement reproduction
works offline, without the original provider's API key.

The plan §19 governance process treats the schema as a first-class
interface — any change requires a CHANGELOG-schema.md entry and
`SchemaVersion::CURRENT` bump, and CI diffs the committed
`events-v1.0.json` against a fresh export on every build.

### 3. Composition over configuration

The library never picks an algorithm for you. You get:

- **Middleware stacks**: `.with_request_layer(..).with_response_layer(..).with_full_layer(..)`.
- **Critic chains**: `CompositeCritic::new(..).push(...).push(...)`.
- **Budget composition**: `CompositeBudget::new().with(TokenBudget::...)`.

What you don't get:
- A `Config::smart_defaults()` that wires a canonical stack.
- An "auto-retry" flag on `Llm::complete()`.
- A `ReactLoopOptions { enable_critics, enable_reflexion, memory_policy, … }` god-struct.

When a knob becomes load-bearing (like `revision_depth_cap`), it
lives on `AgentConfig` — a plain data struct with named fields,
not a builder chain on the agent itself.

### 4. No surprise orchestration

The library is boringly literal. `ReactLoop` does reason-act-
observe. `ConversationLoop` alternates user + assistant.
`run_reflexion` iterates episodes with `ReflectionInjector`
between them. None of these second-guess your intent — if you
wanted a loop to auto-retry on transient errors, that's
middleware territory, not loop territory.

### 5. Deterministic when possible, instrumented always

The trajectory is the primary artifact, not a debug log. Two
consequences:

- **Every event has a seq counter and run_id**. A trajectory is
  a DAG the replayer reconstructs.
- **`FanOutSink` is wired into `Agent::run`**. Even if you hand
  the agent an `InMemorySink`, the run still captures every
  event into `RunOutcome.trajectory`. That's the path `ReplayLlm`
  consumes.

Practitioners who want raw performance can pass `NullSink` and
skip the `FanOutSink` wiring — but the default is audit-first.

### 6. Rust core, polyglot surface

The core is Rust because we need the type safety, the async
model, and the zero-overhead composition. Python bindings are
**non-negotiable** — that's the 90% of the research surface.

The Python bindings (`oharness-py`) implement nine of the ten
plan §14.2 traits via a consistent adapter pattern: JSON wire +
`tokio::task::spawn_blocking` for sync Python. The tenth
(`Llm::stream`) is deferred to a later milestone because async
streaming across the GIL is a genuine research problem.

### 7. Provider honesty

Anthropic supports prompt caching. OpenAI doesn't (yet).
OpenRouter exposes a rich model zoo. Ollama runs locally.

We **don't flatten these differences** into a lossy common
denominator. Each provider's `LlmCapabilities` declares what it
supports. `PromptCaching::anthropic()` construction **fails
loudly** if wrapped around an `Llm` whose `capabilities.prompt_caching == false`. That's a compile-time-adjacent
error surface, not a silent no-op.

### 8. Fail loud, not silent

Things that are errors in `open-harness`:

- Attaching `PromptCaching` to a provider that doesn't support it.
- Constructing an `Agent` with no `with_tools()` *and* a
  `ReactLoop` that requires tools.
- A `UserSimulator` that errors (promotes to
  `Termination::Failed { category: UserSimulator }`, never
  silently `EndConversation`).
- A critic that crashes (the wrapper emits `critic.failed` so
  positional replay can detect drift; the loop continues but
  the event is permanent).
- A `MemoryPolicy` that returns an error mid-turn (fatal; not
  silently passing through the raw conversation).

Things that are **not** errors:
- A `ToolSet::execute` returning `ExecutionError` — the agent
  sees the error as the tool result and decides what to do.
- An LLM returning a truncated response via `stop_reason:
  max_tokens` — that's a valid response; surface it, don't
  fake-retry.

The principle: if the caller could reasonably recover, return
an error they can match on. If they couldn't, fail loudly and
instrument the failure so post-mortems are possible.

## What `open-harness` is NOT

In roughly descending order of "you'd be surprised":

- **Not a sandbox.** `BashToolSet` runs `/bin/bash -c <command>`
  against your filesystem. See `docs/security.md` — wrap in a
  container for anything beyond local dev.
- **Not a LangChain alternative.** It doesn't wrap every SaaS
  API in a chain node. Those integrations belong in user-land
  crates that implement `ToolSet` for each SaaS.
- **Not runtime-agnostic.** Tokio is the only supported async
  runtime. `pyo3-asyncio` + other runtimes were considered and
  rejected at design lock.
- **Not an agent marketplace.** No prompt library, no "agent
  registry", no community prompts repo. If you want that, build
  it on top — everything you need is composable.
- **Not auto-magical.** You wire the pieces. The defaults are
  minimal; the docs walk through them.

## Reading the rest

- [`docs/quickstart.md`](quickstart.md) — first agent in 5 minutes.
- [`docs/concepts.md`](concepts.md) — Task / Agent / Loop / Event mental model.
- [`docs/open-harness-plan.md`](open-harness-plan.md) — the full design spec (v1, locked 2026-04-17).
- [`docs/remaining-work.md`](remaining-work.md) — milestone-by-milestone roadmap.
