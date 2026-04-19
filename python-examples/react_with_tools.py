"""react_with_tools — scripted multi-turn ReAct with real tool dispatch.

Sibling to `hello_scripted.py`: turn 1 emits a `tool_use` block
for `fs_list`, the loop dispatches it against the shipped
`FsToolSet`, the tool result threads back, turn 2 produces a
final text turn.

The LLM is scripted (no network, no cost, no API key), so this
runs identically on every CI machine. Swap in a real provider
and everything else stays the same.
"""

import itertools
import json

import oharness


class ScriptedLlm:
    """Two-response script. First call returns a `fs_list` tool
    call; second call returns a text summary.
    """

    def __init__(self) -> None:
        self._responses = iter([
            {
                "id": "msg_001",
                "model": "scripted-tools-example",
                "content": [
                    {
                        "type": "text",
                        "text": "Let me list the current directory to see what's here.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "fs_list",
                        "input": {"path": "."},
                    },
                ],
                "stop_reason": {"kind": "tool_use"},
                "usage": {"tokens_input": 12, "tokens_output": 40},
            },
            {
                "id": "msg_002",
                "model": "scripted-tools-example",
                "content": [{
                    "type": "text",
                    "text": (
                        "I can see the repository layout. There's a `crates/` "
                        "directory, which is the Cargo workspace root."
                    ),
                }],
                "stop_reason": {"kind": "end_turn"},
                "usage": {"tokens_input": 80, "tokens_output": 30},
            },
        ])

    def complete(self, req_json: str) -> str:
        return json.dumps(next(self._responses))


def main() -> None:
    llm = oharness.PyLlm(ScriptedLlm(), name="scripted")
    # InMemorySink captures the trajectory so we can inspect it
    # at the end. Real runs use FileSink (see replay_trajectory.py).
    sink = oharness.InMemorySink()

    agent = (
        oharness.Agent.builder()
        .with_llm(llm)
        .with_tools(oharness.FsToolSet())
        .with_event_sink(sink)
        .with_loop(oharness.ReactLoop())
        .with_max_turns(5)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("inspect the repo")))

    print(f"Termination: {outcome['termination']}")
    print(
        f"Turns: {outcome['usage']['turns']} | "
        f"tool calls: {outcome['usage']['tool_calls']} | "
        f"tokens in/out: {outcome['usage']['tokens_input']}/{outcome['usage']['tokens_output']}"
    )
    for msg in outcome["final_messages"]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "text":
                print(f"Assistant: {block['text']}")

    print(f"Trajectory events captured: {sink.len()}")


if __name__ == "__main__":
    main()
