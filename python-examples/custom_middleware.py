"""custom_middleware — compose custom RequestLayer + ResponseLayer.

Shows two of the three middleware shapes users write most often:

1. `RequestLayer` — sync, mutate CompletionRequest. Here: stamp a
   request-id into `extensions` (the reverse-DNS metadata map).
2. `ResponseLayer` — sync, mutate CompletionResponse. Here:
   redact `sk-live-*` secrets from every text block.

FullLayer isn't exposed from Python — its `BoxFuture` wrapping
contract doesn't round-trip across the GIL cleanly. For
before/after hooks you get the same observation surface with a
RequestLayer + ResponseLayer pair; for true around-logic (retry,
caching) you'd implement it in Rust-side middleware.

Composition happens via `LayeredLlm(inner, request_layers=[...],
response_layers=[...])`. The result is itself an Llm — hand it
to `.with_llm(..)` like any other.
"""

import itertools
import json

import oharness


class RequestIdStamp:
    """Stamp a monotonically-increasing request-id into the
    outgoing request's extensions map."""

    def __init__(self) -> None:
        self._counter = itertools.count()

    def on_request(self, req_json: str) -> str:
        req = json.loads(req_json)
        req.setdefault("extensions", {})["example.request_id"] = (
            f"req-{next(self._counter)}"
        )
        print(f"[RequestIdStamp] stamped {req['extensions']['example.request_id']}")
        return json.dumps(req)


class RedactSecrets:
    """Redact fake 'sk-live-*' tokens in response text blocks."""

    def on_response(self, res_json: str) -> str:
        res = json.loads(res_json)
        changed = False
        for block in res.get("content", []):
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if "sk-live-" in text:
                block["text"] = text.replace("sk-live-", "sk-live-REDACTED-")
                changed = True
        if changed:
            print("[RedactSecrets] redacted secret in response")
        return json.dumps(res)


class LeakyLlm:
    """Scripted LLM that leaks a fake key in its response."""

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "msg_1",
            "model": "middleware-example",
            "content": [{
                "type": "text",
                "text": "All set. My API key is sk-live-1234567890abc — please keep it safe.",
            }],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 7, "tokens_output": 20},
        })


def main() -> None:
    base = oharness.PyLlm(LeakyLlm(), name="leaky")
    wrapped = oharness.LayeredLlm(
        base,
        request_layers=[oharness.PyRequestLayer(RequestIdStamp(), name="req-id")],
        response_layers=[oharness.PyResponseLayer(RedactSecrets(), name="redactor")],
    )

    agent = (
        oharness.Agent.builder()
        .with_llm(wrapped)
        .with_tools(oharness.FsToolSet())
        .with_loop(oharness.ReactLoop())
        .with_max_turns(2)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("test middleware composition")))
    for msg in outcome["final_messages"]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "text":
                print(f"Assistant (post-redaction): {block['text']}")
    print(f"Termination: {outcome['termination']}")


if __name__ == "__main__":
    main()
