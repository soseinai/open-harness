# oharness-examples

Runnable Python examples for [open-harness](https://github.com/aishfenton/open-harness).

This is a **vanilla Python project** — its only special feature
is that it depends on `oharness`, the Python binding for
open-harness. Everything else is standard
`pyproject.toml` + `uv` / `pip` machinery.

## Setup

```bash
cd python-examples
uv sync
```

That's it. `uv sync` reads `pyproject.toml`, notices the
`[tool.uv.sources]` entry pointing at the local
`crates/oharness-py/` directory, invokes maturin to build the
Rust wheel, and installs `oharness` into `python-examples/.venv/`.
Takes ~10 seconds on a warm cargo cache; ~60 seconds cold.

Once `oharness` is released to PyPI, the `tool.uv.sources` entry
goes away and `uv sync` pulls the prebuilt wheel like any other
dependency.

### If you don't have uv

```bash
# From this directory:
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/pip install -e ../crates/oharness-py
```

`pip install -e` invokes maturin the same way uv does — the
editable install points at the local source, so a `maturin
develop` re-run after editing the Rust side picks up the
change.

## Running

```bash
# From python-examples/:
uv run python hello_scripted.py

# Or with the venv activated:
source .venv/bin/activate
python hello_scripted.py
```

Every example uses a scripted `Llm` — no API keys, no network,
deterministic output.

## The 11 examples

| File                          | Covers                                                |
|-------------------------------|-------------------------------------------------------|
| `hello_scripted.py`           | Minimum viable agent (one turn, no tools)             |
| `react_with_tools.py`         | Multi-turn ReAct with `fs_list` tool dispatch         |
| `custom_critic.py`            | Implement `Critic` from scratch; `reject` verdict     |
| `self_refine.py`              | Stub — `Revise` verdict not exposed from Python (v1)  |
| `llm_judge_critic.py`         | Shipped `LlmJudgeCritic` + SCORE-based threshold      |
| `budget_enforcement.py`       | `BudgetMiddleware` + tight token cap                  |
| `custom_middleware.py`        | `PyRequestLayer` + `PyResponseLayer` via `LayeredLlm` |
| `custom_memory_policy.py`     | Implement `MemoryPolicy` from scratch (keep-last-N)   |
| `replay_trajectory.py`        | Record JSONL → `ReplayLlm` round-trip                 |
| `reflexion_run.py`            | `run_reflexion` + `ReflectionInjector` over episodes  |
| `multi_agent_conversation.py` | `ConversationLoop` + `ScriptedUserSimulator`          |

`self_refine.py` prints a docstring explaining why the pattern
isn't exposed from Python in v1 (see
`crates/oharness-py/README.md` for the design rationale).

## Running all of them

From the repo root:

```bash
just python-examples
```

That rebuilds the wheel via `uv sync` and runs every example in
sequence. Useful as a smoke test after touching either the
Python examples or the Rust bindings.

## Project structure

This project is deliberately **flat** — no `src/` layout, no
sub-packages, just one `.py` file per example. Each file is
self-contained and stands alone; readers should be able to grab
one, drop it into their own project, and see the full pattern
without hunting through shared modules.

## License

Dual-licensed under MIT or Apache-2.0 (matches the workspace).
