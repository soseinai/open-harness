"""budget_enforcement — cap a run's token usage with BudgetMiddleware.

`BudgetMiddleware` is a plain `Llm` wrapper: wrap any provider,
get pre-call + post-call accounting for free. When the cap
trips, the underlying `complete()` call returns an error, which
the agent loop converts to `Termination::Failed { category:
Llm }`.

The pre-call check is the first line of defense — it rejects
before the call is dispatched, so no tokens are actually spent
when the budget denies.
"""

import json

import oharness


class ChattyLlm:
    """Scripted LLM that always claims to spend 100/200 tokens.
    We fake usage for determinism; real providers report actual
    usage on the CompletionResponse.
    """

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "msg_1",
            "model": "chatty-model",
            "content": [{
                "type": "text",
                "text": "Sure, here is a very long response…",
            }],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 100, "tokens_output": 200},
        })


def main() -> None:
    # Tight cap — guaranteed to trip on the first call.
    budget = oharness.TokenBudget.input_plus_output(50)
    bounded = oharness.BudgetMiddleware(
        oharness.PyLlm(ChattyLlm(), name="chatty"),
        budget,
    )

    sink = oharness.InMemorySink()
    agent = (
        oharness.Agent.builder()
        .with_llm(bounded)
        .with_tools(oharness.FsToolSet())
        .with_event_sink(sink)
        .with_loop(oharness.ReactLoop())
        .with_max_turns(5)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("hello")))
    termination = outcome["termination"]
    print(f"Termination: {termination}")
    if termination.get("kind") == "failed":
        err = termination.get("error", {})
        print(f"  category: {err.get('category')}")
        print(f"  message : {err.get('message')}")

    # Budget handle exposes a snapshot for per-task telemetry.
    snap = json.loads(budget.snapshot_json())
    consumed = snap["consumed"]
    remaining = snap.get("remaining") or {}
    print(
        f"Budget snapshot: consumed in/out {consumed['tokens_input']}/"
        f"{consumed['tokens_output']} — remaining in/out "
        f"{remaining.get('tokens_input', 'unbounded')}/"
        f"{remaining.get('tokens_output', 'unbounded')}"
    )
    print(f"Trajectory events captured: {sink.len()}")


if __name__ == "__main__":
    main()
