# `oharness` — Python bindings for open-harness

Plan §14. Lets Python code plug `Llm` / `Critic` / `TaskEvaluator`
implementations into Rust-side agent runs.

> **Status: v1 — seven adapters live.** `Llm`, `Critic`,
> `TaskEvaluator`, `Reflector`, `UserSimulator`, `MemoryPolicy`,
> and `ToolSet` all ship with their Python shim. End-to-end
> Python-driven agent runs, `async def` Python methods, and the
> remaining traits (`RequestLayer` / `ResponseLayer`,
> `ChunkObserver` / `ChunkTransformer`) land in follow-up
> milestones per plan §14.2.

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

Seven adapter classes live on the Python side; each wraps a Python
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
| `Llm::stream`       | ⏳ v1.2+ |
| `RequestLayer` / `ResponseLayer` | ⏳ v1.1 |
| `ChunkObserver` / `ChunkTransformer` | ⏳ per-chunk GIL cost; discouraged |

Each follows the same pattern: JSON wire + `tokio::task::spawn_blocking`
for sync Python.
