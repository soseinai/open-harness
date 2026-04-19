"""replay_trajectory — record a JSONL trajectory, then replay it
against `ReplayLlm`.

This is the paper-supplement reproducibility path: a recorded
trajectory re-drives an agent with byte-for-byte fidelity, no
provider API key required.

`ReplayMode.positional` (used here) pairs the Nth live
`llm.request` with the Nth recorded `llm.response`. Use
`"strict"` + `drift="fail"` for canonical-JSON equality checks.
"""

import itertools
import json
import tempfile
from pathlib import Path

import oharness


class ScriptedLlm:
    """Two-turn script: tool call, then summary."""

    def __init__(self) -> None:
        self._responses = iter([
            {
                "id": "msg_001",
                "model": "scripted-replay",
                "content": [
                    {"type": "text", "text": "Let me look."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "fs_list",
                        "input": {"path": "."},
                    },
                ],
                "stop_reason": {"kind": "tool_use"},
                "usage": {"tokens_input": 10, "tokens_output": 5},
            },
            {
                "id": "msg_002",
                "model": "scripted-replay",
                "content": [{"type": "text", "text": "Found a crates/ directory."}],
                "stop_reason": {"kind": "end_turn"},
                "usage": {"tokens_input": 20, "tokens_output": 6},
            },
        ])

    def complete(self, req_json: str) -> str:
        return json.dumps(next(self._responses))


def last_assistant_text(messages):
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "text":
                return block["text"]
    return None


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trajectory.jsonl"

        # --- phase 1: record a live run into a FileSink.
        print(f"[phase 1] live run → {path}")
        sink = oharness.FileSink(str(path))
        live_agent = (
            oharness.Agent.builder()
            .with_llm(oharness.PyLlm(ScriptedLlm(), name="scripted"))
            .with_tools(oharness.FsToolSet())
            .with_event_sink(sink)
            .with_loop(oharness.ReactLoop())
            .with_max_turns(5)
            .build()
        )
        live = json.loads(live_agent.run(oharness.Task("look around")))
        # Drain the writer task before reading the file back.
        sink.flush()
        print(
            f"  termination: {live['termination']['kind']}, "
            f"turns: {live['usage']['turns']}"
        )

        # --- phase 2: replay from the recorded trajectory file.
        print("[phase 2] replay")
        replay = oharness.ReplayLlm.from_path(
            str(path), mode="positional", drift="warn_and_continue"
        )
        replay_agent = (
            oharness.Agent.builder()
            .with_llm(replay)
            .with_tools(oharness.FsToolSet())
            .with_loop(oharness.ReactLoop())
            .with_max_turns(5)
            .build()
        )
        replay_out = json.loads(replay_agent.run(oharness.Task("look around")))
        print(
            f"  termination: {replay_out['termination']['kind']}, "
            f"turns: {replay_out['usage']['turns']}, "
            f"final: {last_assistant_text(replay_out['final_messages'])!r}"
        )

        # --- phase 3: assert match.
        assert live["usage"]["turns"] == replay_out["usage"]["turns"]
        assert live["usage"]["tool_calls"] == replay_out["usage"]["tool_calls"]
        assert live["termination"]["kind"] == replay_out["termination"]["kind"] == "completed"
        assert (
            last_assistant_text(live["final_messages"])
            == last_assistant_text(replay_out["final_messages"])
        )
        print("[phase 3] replay output matches live run ✔")


if __name__ == "__main__":
    main()
