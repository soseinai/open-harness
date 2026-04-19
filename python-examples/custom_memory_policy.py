"""custom_memory_policy — implement the MemoryPolicy trait from Python.

A memory policy sits between the conversation state and the LLM:
on each turn, the loop hands the policy the full conversation
and the policy returns the `Vec<Message>` the LLM will actually
see.

This example ships `KeepLastN`, which preserves a leading system
message and drops everything non-system except the last `n`
entries. Trivial but typical — exactly what you'd write the
first time you hit context-length issues.
"""

import json

import oharness


class KeepLastN:
    """Keep leading system messages + last N non-system messages."""

    def __init__(self, n: int = 3) -> None:
        self.n = n

    def transform(self, conversation_json: str, ctx_json: str) -> str:
        messages = json.loads(conversation_json)
        systems = [m for m in messages if m.get("role") == "system"]
        non_systems = [m for m in messages if m.get("role") != "system"]
        kept_tail = non_systems[-self.n:]
        return json.dumps(systems + kept_tail)


class ReplyAgainLlm:
    """Scripted LLM that always responds with a counter — one
    response per turn. Lets us grow the conversation long enough
    for the memory policy to kick in."""

    def __init__(self) -> None:
        self._turn = 0

    def complete(self, req_json: str) -> str:
        # Inspect the messages the LLM actually saw — a real
        # policy would drop old tool results, etc. Here we just
        # let the policy run and observe the message count.
        req = json.loads(req_json)
        visible = len(req.get("messages", []))
        self._turn += 1
        return json.dumps({
            "id": f"msg_{self._turn}",
            "model": "replier",
            "content": [{
                "type": "text",
                "text": f"Turn {self._turn}, saw {visible} messages.",
            }],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 5, "tokens_output": 5},
        })


def main() -> None:
    # KeepLastN with n=3 — with each ReactLoop turn producing
    # one user + one assistant pair, the policy will start
    # kicking in after a few turns.
    policy = oharness.PyMemoryPolicy(KeepLastN(3), name="keep-last-3")

    agent = (
        oharness.Agent.builder()
        .with_llm(oharness.PyLlm(ReplyAgainLlm(), name="replier"))
        .with_tools(oharness.FsToolSet())
        .with_memory(policy)
        .with_loop(oharness.ReactLoop())
        .with_max_turns(4)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("chat with me")))

    print(f"Termination: {outcome['termination']}")
    print(f"Turns: {outcome['usage']['turns']}")
    for msg in outcome["final_messages"]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "text":
                print(f"Assistant: {block['text']}")

    # Show the final conversation length — with a ReactLoop
    # running max_turns=4 and n=3, the tail is bounded regardless
    # of how long the conversation grew.
    print(f"Final conversation length: {len(outcome['final_messages'])}")


if __name__ == "__main__":
    main()
