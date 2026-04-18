# `oharness` — Python bindings for open-harness

Plan §14. Lets Python code plug `Llm` / `Critic` / `TaskEvaluator`
implementations into Rust-side agent runs.

> **Status: v1 scaffold.** The adapter pattern is complete; end-to-end
> Python-driven agent runs, `async def` Python methods, and the
> remaining traits (`ToolSet`, `MemoryPolicy`, `Reflector`,
> `UserSimulator`, `RequestLayer` / `ResponseLayer`) land in
> follow-up milestones per plan §14.2.

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

Three adapter classes live on the Python side; each wraps a Python
object implementing a single method. The wire type between Rust and
Python is always a JSON-encoded string — deliberately not a structured
`dict`, because serde on the Rust side already has the canonical
codec.

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

## What's next

Plan §14.2 priority table:

| Trait | Status |
|---|---|
| `Llm::complete`     | ✅ v1 |
| `Critic::assess`    | ✅ v1 |
| `TaskEvaluator::evaluate` | ✅ v1 |
| `Reflector`         | ⏳ v1 |
| `UserSimulator`     | ⏳ v1 |
| `MemoryPolicy`      | ⏳ v1 |
| `Llm::stream`       | ⏳ v1.2+ |
| `ToolSet`           | ⏳ v1.1 |
| `RequestLayer` / `ResponseLayer` | ⏳ v1.1 |
| `ChunkObserver` / `ChunkTransformer` | ⏳ per-chunk GIL cost; discouraged |

Each follows the same pattern: JSON wire + `tokio::task::spawn_blocking`
for sync Python.
