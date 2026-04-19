"""custom_critic — implement the `Critic` trait from scratch.

Shows:
1. What implementing `Critic` looks like end-to-end (one method,
   one verdict JSON string).
2. How a `reject` verdict surfaces at the loop layer: the run
   terminates with `Termination::Failed { category: Critic }`.

Python can emit four verdicts: `accept`, `accept_with_note`,
`reject`, `abort`. The `revise` variant (in-place turn rewrite)
is not currently supported from Python — see `self_refine.py`
for the trade-off, or `oharness-loop/examples/self_refine.rs`
for the Rust version.
"""

import json

import oharness

HEDGES = ("i'm not sure", "i am not sure", "maybe", "possibly")


class NoHedgingCritic:
    """Reject any assistant turn containing hedge phrases.

    The Python class gets an `assess(ctx_json) -> str` method;
    `ctx_json` is a trimmed `AssessmentContext` carrying
    `task`, `latest_turn` (the just-emitted assistant message),
    and `turn_index`. Returns a verdict JSON.
    """

    def assess(self, ctx_json: str) -> str:
        ctx = json.loads(ctx_json)
        turn = ctx.get("latest_turn") or {}
        content = turn.get("content", [])
        text_blocks = [
            b.get("text", "") for b in content
            if b.get("type") == "text"
        ]
        joined = " ".join(text_blocks).lower()
        for hedge in HEDGES:
            if hedge in joined:
                return json.dumps({
                    "verdict": "reject",
                    "reason": f"response hedges: found '{hedge}'",
                })
        return json.dumps({"verdict": "accept"})


class ScriptedHedgeLlm:
    """Scripted LLM that hedges on its one and only turn."""

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "msg_1",
            "model": "scripted-hedger",
            "content": [{"type": "text", "text": "I'm not sure what you're asking."}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 8, "tokens_output": 9},
        })


def main() -> None:
    # A CompositeCritic wraps one or more critics with an
    # aggregation policy. Even a single critic goes through the
    # composite so the policy is explicit.
    critics = oharness.CompositeCritic("hedge-guard", "first_reject")
    critics.push(oharness.PyCritic(NoHedgingCritic(), name="no-hedging"))

    sink = oharness.InMemorySink()
    agent = (
        oharness.Agent.builder()
        .with_llm(oharness.PyLlm(ScriptedHedgeLlm(), name="scripted"))
        .with_tools(oharness.FsToolSet())
        .with_event_sink(sink)
        .with_loop(oharness.ReactLoop())
        .with_critics(critics)
        .with_max_turns(3)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("figure it out")))

    # Reject on turn 1 → run fails with category=Critic.
    termination = outcome["termination"]
    print(f"Termination: {termination}")
    if termination.get("kind") == "failed":
        err = termination.get("error", {})
        print(f"Critic message: {err.get('message')}")
        print(f"Category: {err.get('category')}")

    # The trajectory carries a `critic.rejected` event. Events
    # are shaped as `{v, seq, run_id, timestamp, span_id, type,
    # payload, ...}` — `type` is the event kind tag.
    events = json.loads(sink.events_json())
    rejections = sum(1 for e in events if e.get("type") == "critic.rejected")
    print(f"critic.rejected events: {rejections}")


if __name__ == "__main__":
    main()
