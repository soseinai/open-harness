"""multi_agent_conversation — ConversationLoop + ScriptedUserSimulator.

Where `ReactLoop` drives a single agent answering one user task,
`ConversationLoop` alternates the assistant's replies with a
user simulator's follow-ups. When the simulator emits
`end_conversation`, the loop terminates with
`Termination::Completed`.

Production setups typically replace the user side with an LLM-
driven simulator; here we script both sides for a
deterministic, no-API-key demo.
"""

import itertools
import json

import oharness


class ScriptedAssistant:
    """One canned reply per user turn."""

    def __init__(self) -> None:
        self._responses = iter([
            (
                "Sure — for unit testing, the built-in `#[test]` attribute "
                "covers most cases; for BDD-ish feel, try `rstest`."
            ),
            (
                "`criterion` is the canonical benchmarking crate, not a "
                "unit-test framework. Use it alongside `#[test]`."
            ),
            "You're welcome! Happy testing.",
        ])

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "msg",
            "model": "scripted",
            "content": [{"type": "text", "text": next(self._responses)}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 5, "tokens_output": 10},
        })


def main() -> None:
    # ScriptedUserSimulator returns the first script entry as the
    # initial message, then each subsequent entry on `respond`.
    # When exhausted, it emits `end_conversation`.
    user = oharness.ScriptedUserSimulator([
        "Hi! Can you help me pick a library for unit testing in Rust?",
        "How does it compare to `criterion`?",
        "Thanks, that's helpful.",
    ], name="inquisitive-user")

    assistant = oharness.PyLlm(ScriptedAssistant(), name="scripted-assistant")
    conv_loop = oharness.ConversationLoop(
        user,
        system_prompt="You are a helpful, concise Rust library guide.",
    )

    sink = oharness.InMemorySink()
    agent = (
        oharness.Agent.builder()
        .with_llm(assistant)
        .with_tools(oharness.FsToolSet())
        .with_event_sink(sink)
        .with_loop(conv_loop)
        .with_max_turns(10)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("Rust library recommendations")))
    print(f"Termination: {outcome['termination']}")
    print(f"Turns: {outcome['usage']['turns']}")

    print("\nTranscript:")
    for msg in outcome["final_messages"]:
        role = msg.get("role")
        parts = [
            b.get("text", "") for b in msg.get("content", [])
            if b.get("type") == "text"
        ]
        text = " ".join(parts) if parts else (msg.get("content") or "")
        print(f"  [{role}] {text}")


if __name__ == "__main__":
    main()
