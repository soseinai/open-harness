"""self_refine — NOT YET AVAILABLE from Python.

The Rust-side `self_refine` example demonstrates a critic
emitting `CriticVerdict::Revise { replacement, reason }`. The
loop swaps the assistant message in place and continues — no LLM
re-dispatch.

**This pattern is not exposed from Python in v1.** Python
critics can emit four verdicts:

- `accept`
- `accept_with_note` (with a `note` field)
- `reject` (with a `reason` field — terminates the run)
- `abort`  (with a `reason` field — hard stop)

The fifth variant `revise` requires handing the loop a full
replacement `AssistantTurn` struct with content, span id, usage,
and stop reason. Round-tripping that shape across the GIL is
non-trivial, so it was deferred at the M3 design stage.

If you need critic-driven self-refinement from Python today,
your options are:

1. Emit `reject` instead and rebuild the agent with an
   improved prompt. You lose the single-run semantics but gain
   reproducibility.
2. Use `run_reflexion` (see `reflexion_run.py`) — episode-level
   feedback via a `Reflector`. Coarser-grained but fully
   supported.
3. Write the refiner as Rust-side middleware. The Rust
   `self_refine.rs` example is ~160 LOC and copy-pasteable.

For background on the design decision, see
`crates/oharness-py/README.md` §PyCritic and plan §14.2's
priority table.
"""

import sys


def main() -> None:
    print(__doc__)
    print("Exit: no runnable code in this example.")
    sys.exit(0)


if __name__ == "__main__":
    main()
