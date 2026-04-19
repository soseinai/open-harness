# `oharness` — Python bindings for open-harness

Plan §14. End-to-end Python-driven agent runs — Python users
write adapters (Llm, Critic, MemoryPolicy, …) and/or drive the
full `Agent` loop from Python. Sync API; `async def` adapters
land in a later milestone.

> **Status: v1 — full orchestration surface live.** Nine
> adapter classes (`PyLlm`, `PyCritic`, `PyTaskEvaluator`,
> `PyReflector`, `PyUserSimulator`, `PyMemoryPolicy`, `PyToolSet`,
> `PyRequestLayer`, `PyResponseLayer`) plus twelve orchestration
> classes (`Agent`, `AgentBuilder`, `Task`, `ReactLoop`,
> `ConversationLoop`, `FsToolSet`, `InMemorySink`, `FileSink`,
> `ReplayLlm`, `TokenBudget`, `BudgetMiddleware`,
> `LayeredLlm`, `LlmJudgeCritic`, `ReflectionInjector`,
> `CompositeCritic`, `ScriptedUserSimulator`) plus one
> module-level function (`run_reflexion`).
>
> Deferred per plan §14.2: `Llm::stream` (async streaming
> across the GIL), `ChunkObserver` / `ChunkTransformer`
> (per-chunk GIL cost), `CriticVerdict::Revise` from Python
> (full `AssistantTurn` round-trip needs more design), and
> `FullLayer` from Python (`BoxFuture` wrapping doesn't
> round-trip cleanly).

## Build

The crate lives outside the Cargo workspace (path deps still resolve)
so a CI runner without Python headers doesn't block on it. Build with
[maturin](https://www.maturin.rs):

```bash
cd crates/oharness-py
pipx install maturin  # or `uv tool install maturin`, etc.
maturin develop --release
```

This produces an importable Python package named `oharness` in the
current Python environment.

### Checking from Rust only

`cargo check` / `cargo clippy` work without Python because
`pyo3`'s `abi3-py310` feature bundles the ABI stubs:

```bash
cd crates/oharness-py
cargo check
cargo clippy --all-targets -- -D warnings
```

The root `just ci` skips this crate intentionally — CI runners
without Python headers would fail. Use `just python-check` to run the
Rust-side check opt-in.

## Adapter pattern

Nine adapter classes live on the Python side; each wraps a Python
object implementing one method (or two, for `PyUserSimulator`).
The wire type between Rust and Python is always a JSON-encoded
string — deliberately not a structured `dict`, because serde on the
Rust side already has the canonical codec.

### `PyLlm`

```python
import oharness
import json

class MyLlm:
    def complete(self, req_json: str) -> str:
        req = json.loads(req_json)
        # ... compute a CompletionResponse from req ...
        return json.dumps({
            "id": "msg_1",
            "model": "my-model",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 42, "tokens_output": 4},
        })

llm = oharness.PyLlm(MyLlm(), name="my-llm")
```

Notes:
- `stream()` returns `LlmError::Unsupported("stream")` for now —
  streaming from Python is v1.2+ per plan §14.2.
- The sync `complete` method runs under `tokio::task::spawn_blocking`
  on the Rust side, so Python code blocking on IO doesn't stall the
  async runtime.

### `PyCritic`

```python
import oharness
import json

class NoHedging:
    def assess(self, ctx_json: str) -> str:
        ctx = json.loads(ctx_json)
        # Inspect ctx["latest_turn"], ctx["task"], ctx["turn_index"].
        if "not sure" in json.dumps(ctx).lower():
            return json.dumps({"verdict": "reject", "reason": "hedging"})
        return json.dumps({"verdict": "accept"})

critic = oharness.PyCritic(NoHedging(), name="no-hedging")
```

Valid verdict shapes:

```json
{"verdict": "accept"}
{"verdict": "accept_with_note", "note": "..."}
{"verdict": "reject", "reason": "..."}
{"verdict": "abort",  "reason": "..."}
```

`revise` is intentionally NOT supported from Python in v1 — the
replacement `AssistantTurn` shape is non-trivial. Python critics that
want to rewrite turns should emit `reject` and let the loop's retry
path handle regeneration.

**Fail-open:** on any exception in Python or JSON decode error on the
Rust side, the adapter returns `AcceptWithNote` with the error message
— consistent with plan §11.1's fail-open policy.

### `PyTaskEvaluator`

```python
import oharness
import json

class ContainsHello:
    def evaluate(self, task_json: str, outcome_json: str) -> str:
        outcome = json.loads(outcome_json)
        passed = any(
            "hello" in block.get("text", "").lower()
            for msg in outcome["final_messages"]
            if msg.get("role") == "assistant"
            for block in msg.get("content", [])
            if block.get("type") == "text"
        )
        return json.dumps({
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "details": {"evaluator": "contains_hello"},
        })

evaluator = oharness.PyTaskEvaluator(ContainsHello())
```

### `PyReflector`

```python
import oharness
import json

class SummarizeLowScoreFailures:
    def reflect(self, episode_json: str):
        ep = json.loads(episode_json)
        if ep["evaluation"]["passed"]:
            return None           # no reflection on success
        score = ep["evaluation"]["score"]
        return json.dumps({
            "text": f"Episode {ep['index']} failed (score={score:.2f}). "
                    f"Last termination: {ep['outcome']['termination']}.",
            "metadata": {"source": "py", "score": score},
        })

reflector = oharness.PyReflector(SummarizeLowScoreFailures(), name="low-score")
```

Notes:
- Return `None` (or the literal string `"null"`) to emit no
  reflection for the current episode.
- The episode wire carries `task`, `outcome` (without the trajectory
  handle — in-memory handles can't serialize, and file handles are
  useless to Python), `evaluation`, `prior_reflections`, `index`.
- `created_at` on the `Reflection` is stamped on the Rust side, so
  Python only needs to supply `text` + optional `metadata`.
- Errors (exception, malformed JSON) `eprintln!` and return `None` —
  a broken reflector should not break the reflexion sweep.

### `PyUserSimulator`

```python
import oharness
import json

class HelpfulUser:
    def initial_message(self, task_json: str) -> str:
        task = json.loads(task_json)
        return task["instruction"]

    def respond(self, conversation_json: str, task_json: str) -> str:
        messages = json.loads(conversation_json)
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"),
            None,
        )
        if last_assistant and any(
            "done" in (b.get("text") or "").lower()
            for b in last_assistant.get("content", [])
            if b.get("type") == "text"
        ):
            return json.dumps({"action": "end_conversation"})
        return json.dumps({"action": "say", "message": "keep going"})

user = oharness.PyUserSimulator(HelpfulUser(), name="helpful")
```

Action wire shapes:

```json
{"action": "say", "message": "..."}
{"action": "end_conversation"}
```

**Not fail-open.** Unlike critics, simulator errors are promoted to
`UserError::Other`, which the `ConversationLoop` turns into
`Termination::Failed { reason: "user_simulator_error" }`. Hiding
simulator bugs behind a silent `EndConversation` would break eval
reproducibility, so the loop refuses to do it.

### `PyMemoryPolicy`

```python
import oharness
import json

class KeepLastN:
    def __init__(self, n: int = 8):
        self.n = n

    def transform(self, conversation_json: str, ctx_json: str) -> str:
        messages = json.loads(conversation_json)
        # Preserve any leading system message + the last N non-system messages.
        head = [m for m in messages[:1] if m.get("role") == "system"]
        tail = [m for m in messages if m.get("role") != "system"][-self.n:]
        return json.dumps(head + tail)

policy = oharness.PyMemoryPolicy(KeepLastN(8), name="keep-last-8")
```

Notes:
- `ctx_json` carries `{"token_budget": N}` and nothing else.
- **Python memory policies cannot emit `memory.evicted` /
  `memory.summarized` / `memory.retrieved` events** in v1 — the
  `ScopedEmitter` doesn't cross the boundary. Future work may grow
  a return-side `events` channel so Python policies can surface
  telemetry.
- Errors promote to `MemoryError::Configuration`, which the loop
  treats as fatal for the turn. A broken memory policy must NOT
  silently pass the raw conversation through — corrupted context
  windows are worse than a failed run.

### `PyToolSet`

```python
import oharness
import json

SPECS = [
    {
        "name": "reverse",
        "description": "Reverse a string.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

class StringTools:
    def execute(self, name: str, input_json: str, ctx_json: str) -> str:
        args = json.loads(input_json)
        if name == "reverse":
            return json.dumps({
                "outcome": "success_text",
                "text": args["text"][::-1],
            })
        return json.dumps({
            "outcome": "execution_error",
            "message": f"unknown tool: {name}",
            "recoverable": False,
        })

toolset = oharness.PyToolSet(StringTools(), json.dumps(SPECS), name="string-tools")
```

Wire shapes for the `execute` return value:

```json
{"outcome": "success", "output": {"content": [{"type":"text","text":"..."}], "truncated": false}}
{"outcome": "success_text", "text": "..."}
{"outcome": "execution_error", "message": "...", "recoverable": false}
{"outcome": "denied", "reason": "..."}
{"outcome": "cancelled"}
```

Notes:
- **Specs are fixed at construction time.** The Rust loop reads
  `specs()` once per request when assembling the
  `CompletionRequest`, so round-tripping through Python on every
  turn just to list tools would be wasteful. Rebuild the
  `PyToolSet` between runs if you need a different spec set.
- `ctx_json` carries `workspace_path` (optional) + `extensions` (a
  reverse-DNS metadata map). `EventSink`, `BudgetHandle`,
  `Cancellation`, `ApprovalChannel` are Rust-runtime types that
  can't usefully be exposed to Python in v1.
- `success_text` is a shorthand — it's equivalent to `success`
  with a single text `ToolOutput` block. Handy for the common
  "tool returns one string" case.
- Errors (exception, malformed JSON, bad shape) turn into
  `ExecutionError { recoverable: false }`. The loop will see the
  failure via `tool.call.failed` — the agent continues, it just
  sees the error as the tool result.
- `toolset.tool_names()` is exposed for quick Python-side
  inspection.

### `PyRequestLayer`

```python
import oharness
import json

class InjectRequestId:
    def __init__(self):
        self.counter = 0

    def on_request(self, req_json: str) -> str:
        req = json.loads(req_json)
        req.setdefault("metadata", {})["x-request-id"] = f"req-{self.counter}"
        self.counter += 1
        return json.dumps(req)

layer = oharness.PyRequestLayer(InjectRequestId(), name="inject-req-id")
```

Notes:
- The layer is called **synchronously under the GIL** from inside
  the async `complete()` / `stream()` task. Fine for cheap work
  (redaction, header injection, metadata merging); don't put heavy
  Python here — wrap your `PyLlm` with slow middleware in Python
  instead.
- Python must return a full-shape `CompletionRequest` JSON. The
  Rust side replaces the outgoing request in place with the
  deserialized result.
- **Fail-open**: exception, bad JSON, or bad shape logs to stderr
  (`PyRequestLayer(name): ...`) and leaves the request unchanged.
  A broken layer should not crash the run.

### `PyResponseLayer`

```python
import oharness
import json

class RedactSecrets:
    def on_response(self, res_json: str) -> str:
        res = json.loads(res_json)
        for block in res.get("content", []):
            if block.get("type") == "text":
                block["text"] = block["text"].replace("sk-live-", "sk-live-REDACTED-")
        return json.dumps(res)

layer = oharness.PyResponseLayer(
    RedactSecrets(),
    name="redact-live-keys",
    stream_mode="warn_and_skip",  # or "error" / "silent_skip"
)
```

Notes:
- `stream_mode` picks the behaviour when the layer is wrapped
  around `stream()`:
  - `"warn_and_skip"` (default) — log once per wrapper, pass
    chunks through unchanged.
  - `"error"` — `stream()` returns
    `LlmError::Unsupported("response_layer_on_stream")`. Use when
    your layer's invariants can't be satisfied by streaming (e.g.,
    the redaction has to see the whole response to decide).
  - `"silent_skip"` — pass chunks through without logging. Rare;
    usually you want `"warn_and_skip"` so misconfiguration is
    audible.
- Same sync-in-async caveat as `PyRequestLayer` — keep layers
  cheap.
- Same fail-open semantics — a broken layer leaves the response
  unchanged.

## Orchestration surface

The adapter pattern above lets you **write** extension points in
Python. To **drive** a full agent from Python, use the
orchestration classes — Rust types wrapped as first-class Python
bindings (no `Py*` prefix; they're ergonomic Python classes, not
adapter bridges):

- **`Agent` / `AgentBuilder`** — `oharness.Agent.builder()...build()`
  fluent construction, `.run(task)` returns a JSON-serialised
  `RunOutcome`.
- **`Task`** — minimum-shape task with an instruction string.
- **Loops**: `ReactLoop`, `ConversationLoop`.
- **Sinks**: `InMemorySink` (for tests / inspection),
  `FileSink` (JSONL trajectory writer; call `.flush()` before
  exit).
- **Tools**: `FsToolSet` (shipped). Your own `PyToolSet` (the
  adapter) also works everywhere `FsToolSet` does.
- **Critics**: `CompositeCritic(name, policy)` wraps one or more
  critics with an aggregation policy
  (`"first_reject"` / `"all_must_accept"` / `"majority_vote"`).
  `LlmJudgeCritic(judge, rubric, threshold)` is the shipped
  SCORE-based judge.
- **Middleware**: `LayeredLlm(inner, request_layers=[...],
  response_layers=[...])` composes arbitrary Python-defined
  layers around any Llm.
- **Budgets**: `TokenBudget.input_plus_output(cap)` +
  `BudgetMiddleware(inner, budget)`.
- **Replay**: `ReplayLlm.from_path("run.jsonl")` reads a
  recorded trajectory and re-drives an agent without the
  original provider.
- **Reflexion**: `ReflectionInjector()` + the module-level
  `oharness.run_reflexion(agent, task, evaluator, reflector,
  max_episodes=N)` function. The agent must have been built
  with `.with_reflection_injector(injector)` for `run_reflexion`
  to find the handle.
- **Conversation**: `ScriptedUserSimulator([msg, msg, ...])`
  feeds a `ConversationLoop` with a pre-written user side.

### Hello agent — the 10-line version

```python
import json
import oharness


class HelloLlm:
    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "m", "model": "hello",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 1, "tokens_output": 2},
        })


agent = (oharness.Agent.builder()
    .with_llm(oharness.PyLlm(HelloLlm()))
    .with_tools(oharness.FsToolSet())
    .with_loop(oharness.ReactLoop())
    .build())
outcome = json.loads(agent.run(oharness.Task("say hello")))
print(outcome["termination"])
```

## Examples

`crates/oharness-py/examples/` ships 10 runnable examples + 1
stub, mirroring the 11 Rust examples in
`crates/oharness-loop/examples/`. All use scripted LLMs so they
run without API keys. Build + run via `just python-examples`
(requires a `.venv` at `crates/oharness-py/.venv/` — see the
recipe for bootstrap).

| Example                          | Covers                                                |
|----------------------------------|-------------------------------------------------------|
| `hello_scripted.py`              | Minimum viable agent (one turn, no tools)             |
| `react_with_tools.py`            | Multi-turn ReAct with real `FsToolSet` dispatch       |
| `custom_critic.py`               | Implement `Critic` from scratch; `reject` verdict     |
| `self_refine.py`                 | Stub — `CriticVerdict::Revise` not exposed from Python|
| `llm_judge_critic.py`            | Shipped `LlmJudgeCritic` + SCORE-based threshold      |
| `budget_enforcement.py`          | `BudgetMiddleware` + tight token cap                  |
| `custom_middleware.py`           | `PyRequestLayer` + `PyResponseLayer` via `LayeredLlm` |
| `custom_memory_policy.py`        | Implement `MemoryPolicy` from scratch (keep-last-N)   |
| `replay_trajectory.py`           | Record JSONL → `ReplayLlm` round-trip                 |
| `reflexion_run.py`               | `run_reflexion` + `ReflectionInjector` over episodes  |
| `multi_agent_conversation.py`    | `ConversationLoop` + `ScriptedUserSimulator`          |

## What's next

Plan §14.2 priority table:

| Trait | Status |
|---|---|
| `Llm::complete`     | ✅ v1 |
| `Critic::assess`    | ✅ v1 |
| `TaskEvaluator::evaluate` | ✅ v1 |
| `Reflector::reflect` | ✅ v1 |
| `UserSimulator`     | ✅ v1 |
| `MemoryPolicy::transform` | ✅ v1 |
| `ToolSet::execute`  | ✅ v1 |
| `RequestLayer`      | ✅ v1 |
| `ResponseLayer`     | ✅ v1 |
| `Llm::stream`       | ⏳ v1.2+ |
| `ChunkObserver` / `ChunkTransformer` | ⏳ per-chunk GIL cost; discouraged |

Each follows the same pattern: JSON wire + `tokio::task::spawn_blocking`
for sync Python.
