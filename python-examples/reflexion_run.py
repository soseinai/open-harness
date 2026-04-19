"""reflexion_run — multi-episode run_reflexion from Python.

The pattern (plan §11.4 / §12.6):

1. Run the agent on a task.
2. Score the outcome with a `TaskEvaluator`. If it passes, stop.
3. Ask a `Reflector` to produce a note.
4. The note is injected into the next episode's system prompt
   via `ReflectionInjector` — a RequestLayer that prepends a
   `"Reflections from prior attempts:\\n..."` block.
5. Repeat up to `max_episodes`.

Scripts three LLM responses: first two are vague ("still
thinking…") so the evaluator fails and the reflector emits
notes; the third says "done!" so the loop stops. The injector
is threaded through both the `LayeredLlm` (so the LLM sees
accumulated reflections) and the agent builder (so
`run_reflexion` can find it between episodes).
"""

import itertools
import json

import oharness


class CyclingLlm:
    """Returns the next scripted response each call, wrapping
    around when exhausted. Because `complete` is called once per
    agent turn (not once per episode), each episode picks up
    whatever the cursor lands on.
    """

    def __init__(self) -> None:
        self._responses = [
            "I'm still thinking — let me gather context.",
            "I need more time to consider.",
            "Task complete — done!",
        ]
        self._cursor = 0

    def complete(self, req_json: str) -> str:
        idx = self._cursor % len(self._responses)
        self._cursor += 1
        return json.dumps({
            "id": f"msg_{idx}",
            "model": "reflexion-example",
            "content": [{"type": "text", "text": self._responses[idx]}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 5, "tokens_output": 5},
        })


class FinishedEvaluator:
    """`evaluate(task_json, outcome_json) -> str` — pass iff the
    final assistant text contains 'done'."""

    def evaluate(self, task_json: str, outcome_json: str) -> str:
        outcome = json.loads(outcome_json)
        ok = any(
            "done" in block.get("text", "").lower()
            for msg in outcome.get("final_messages", [])
            if msg.get("role") == "assistant"
            for block in msg.get("content", [])
            if block.get("type") == "text"
        )
        return json.dumps({
            "score": 1.0 if ok else 0.0,
            "passed": ok,
            "details": {},
        })


class NudgeReflector:
    """`reflect(episode_json) -> Optional[str]` — emit a canned
    "be concrete" note. Real reflectors call an LLM for a
    pointed analysis (see `LlmReflector` on the Rust side)."""

    def reflect(self, episode_json: str):
        ep = json.loads(episode_json)
        return json.dumps({
            "text": (
                f"Episode {ep['index']} didn't finish. Be concrete — "
                "say 'done!' when the task is complete."
            ),
            "metadata": {},
        })


def main() -> None:
    # The injector is the middleware the reflector feeds into.
    # Build one, share it with both the LLM's layer stack AND the
    # agent — `run_reflexion` needs the latter to locate it
    # between episodes.
    injector = oharness.ReflectionInjector()

    base_llm = oharness.PyLlm(CyclingLlm(), name="cycling")
    layered = oharness.LayeredLlm(base_llm, request_layers=[injector])

    agent = (
        oharness.Agent.builder()
        .with_llm(layered)
        .with_tools(oharness.FsToolSet())
        .with_loop(oharness.ReactLoop())
        .with_reflection_injector(injector)
        .with_max_turns(1)
        .build()
    )

    episodes_json = oharness.run_reflexion(
        agent,
        oharness.Task("finish the task"),
        oharness.PyTaskEvaluator(FinishedEvaluator()),
        oharness.PyReflector(NudgeReflector(), name="nudge"),
        max_episodes=5,
    )
    episodes = json.loads(episodes_json)

    print(f"Episodes run: {len(episodes)}")
    for i, ep in enumerate(episodes):
        ev = ep["evaluation"]
        print(
            f"  episode {i}: passed={ev['passed']} "
            f"score={ev['score']:.2f} "
            f"reflections_seen={len(ep['prior_reflections'])}"
        )

    last = episodes[-1]
    assert last["evaluation"]["passed"], "expected a passing final episode"
    print("Final episode passed ✔")


if __name__ == "__main__":
    main()
