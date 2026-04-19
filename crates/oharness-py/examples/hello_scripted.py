"""hello_scripted — the "first Python agent in 10 lines" example.

Builds a minimal agent wired with a scripted `Llm` (no real API
calls, no cost), the shipped `FsToolSet`, and the default
`ReactLoop`. Runs a single turn and prints the termination + final
assistant message.

Build + install the `oharness` wheel first:

    cd crates/oharness-py
    maturin develop --release

Then run:

    python examples/hello_scripted.py
"""

import json

import oharness


class HelloLlm:
    """A scripted `Llm` that returns one canned response.

    The Python class just needs a `complete(req_json) -> str`
    method. `req_json` is a JSON-encoded `CompletionRequest`
    (which this example doesn't need to inspect); the return is a
    JSON-encoded `CompletionResponse`.

    Real adapters (`AnthropicLlm`, `OpenAiLlm`, …) slot in here
    without any other changes to the agent.
    """

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "hello-1",
            "model": "scripted-example",
            "content": [{
                "type": "text",
                "text": (
                    "Hello from open-harness! Running against a scripted LLM; "
                    "the trajectory of this run is captured via the default "
                    "middleware stack."
                ),
            }],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 3, "tokens_output": 20},
        })


def main() -> None:
    llm = oharness.PyLlm(HelloLlm(), name="hello")
    agent = (
        oharness.Agent.builder()
        .with_llm(llm)
        .with_tools(oharness.FsToolSet())
        .with_loop(oharness.ReactLoop())
        .with_max_turns(3)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("say hello")))

    print(f"Termination: {outcome['termination']}")
    print(
        f"Turns: {outcome['usage']['turns']}, "
        f"tool calls: {outcome['usage']['tool_calls']}"
    )
    for msg in outcome["final_messages"]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "text":
                print(f"Assistant: {block['text']}")


if __name__ == "__main__":
    main()
