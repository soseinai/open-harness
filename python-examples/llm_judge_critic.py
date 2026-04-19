"""llm_judge_critic — shipped LlmJudgeCritic + SCORE threshold.

Uses a second LLM as a grader: it receives the task, the
assistant's response, and a rubric; it replies with a
`SCORE: <0..1>` line; the critic parses that and compares to a
threshold. Above → AcceptWithNote. Below → Reject.

Constitutional-AI style critics (where the rubric encodes
principles) are the natural next step — same shape, just pass a
principles-as-rubric string.
"""

import json

import oharness


class StudentLlm:
    """The "student" being graded. One response."""

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "student_1",
            "model": "student-model",
            "content": [{
                "type": "text",
                "text": (
                    "The capital of France is Paris. It's the country's "
                    "largest city and sits on the river Seine."
                ),
            }],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 8, "tokens_output": 25},
        })


class ScriptedJudge:
    """Judge LLM. Scripted to return SCORE: 0.87.

    Real deployments point this at a stronger model (GPT-4 /
    Claude Opus) with the judging prompt the critic generates
    internally.
    """

    def __init__(self, score_line: str = "SCORE: 0.87") -> None:
        self.score_line = score_line

    def complete(self, req_json: str) -> str:
        return json.dumps({
            "id": "judge_1",
            "model": "judge-model",
            "content": [{"type": "text", "text": self.score_line}],
            "stop_reason": {"kind": "end_turn"},
            "usage": {"tokens_input": 0, "tokens_output": 0},
        })


RUBRIC = """\
Award 1.0 for a correct, complete answer.
Award 0.7 for a correct but partial answer.
Award 0.0 for an incorrect answer.
Do not reward verbose filler.
"""


def main() -> None:
    judge = oharness.PyLlm(ScriptedJudge("SCORE: 0.87"), name="scripted-judge")
    critic = oharness.LlmJudgeCritic(
        judge, RUBRIC, threshold=0.75, name="judge-paris"
    )

    critics = oharness.CompositeCritic("judge-chain", "first_reject")
    critics.push(critic)

    sink = oharness.InMemorySink()
    agent = (
        oharness.Agent.builder()
        .with_llm(oharness.PyLlm(StudentLlm(), name="student"))
        .with_tools(oharness.FsToolSet())
        .with_event_sink(sink)
        .with_loop(oharness.ReactLoop())
        .with_critics(critics)
        .with_max_turns(3)
        .build()
    )

    outcome = json.loads(agent.run(oharness.Task("What is the capital of France?")))
    print(f"Termination: {outcome['termination']}")

    # Find the critic.assessed event and dump its payload.
    events = json.loads(sink.events_json())
    for ev in events:
        if ev.get("type") == "critic.assessed":
            print(f"critic.assessed payload: {ev.get('payload')}")


if __name__ == "__main__":
    main()
