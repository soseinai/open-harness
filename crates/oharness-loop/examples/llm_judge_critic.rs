//! `llm_judge_critic` — shipped `LlmJudgeCritic` + judge LLM scoring
//! against a rubric.
//!
//! Uses a second LLM as a grader: it receives the task, the assistant's
//! response, and a rubric; it replies with a `SCORE: <0..1>` line; the
//! critic parses that and compares to a threshold. Above threshold →
//! `AcceptWithNote` carrying the score. Below → `Reject` with a
//! reason.
//!
//! This is the pattern the plan calls "LLM-as-judge" in §11.6.
//! Constitutional-AI style critics — where the rubric encodes
//! principles — are the natural next step (plan defers
//! `ConstitutionalCritic` to a later milestone, but the same
//! `LlmJudgeCritic` machinery supports that shape: just pass a
//! principles-as-rubric string).
//!
//! Run with:
//!
//! ```bash
//! cargo run --example llm_judge_critic -p oharness-loop --features llm-judge
//! ```

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Usage,
};
use oharness_critic::{shipped::LlmJudgeCritic, AggregationPolicy, CompositeCritic};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::Arc;

// ---------------------------------------------------------------------
// Target LLM — the "student" being graded. One response.
// ---------------------------------------------------------------------

struct StudentLlm;

#[async_trait]
impl Llm for StudentLlm {
    fn name(&self) -> &str {
        "student"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "student_1".into(),
            model: ModelId::new("student-model"),
            content: vec![Content::text(
                "The capital of France is Paris. It's the country's largest city and \
                 sits on the river Seine.",
            )],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 8,
                tokens_output: 25,
                ..Default::default()
            },
        })
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

// ---------------------------------------------------------------------
// Judge LLM — scripted to return `SCORE: 0.87`. Real deployments
// hand this to a stronger model (GPT-4 / Claude Opus) with the
// judging prompt the critic generates.
// ---------------------------------------------------------------------

struct ScriptedJudge {
    score_line: String,
}

#[async_trait]
impl Llm for ScriptedJudge {
    fn name(&self) -> &str {
        "scripted-judge"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "judge_1".into(),
            model: ModelId::new("judge-model"),
            content: vec![Content::text(&self.score_line)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The judge's rubric — plain English guidance. The critic
    // renders it into a full judge prompt internally.
    const RUBRIC: &str = "\
        Award 1.0 for a correct, complete answer.\n\
        Award 0.7 for a correct but partial answer.\n\
        Award 0.0 for an incorrect answer.\n\
        Do not reward verbose filler.";

    let judge: Arc<dyn Llm> = Arc::new(ScriptedJudge {
        score_line: "SCORE: 0.87".into(),
    });

    let critic = LlmJudgeCritic::new(judge, RUBRIC, /*threshold*/ 0.75).with_name("judge-paris");

    let critics = Arc::new(
        CompositeCritic::new("judge-chain", AggregationPolicy::FirstReject).push(Box::new(critic)),
    );

    let sink = Arc::new(InMemorySink::new());
    let agent = Agent::builder()
        .with_llm(Arc::new(StudentLlm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_critics(critics)
        .with_max_turns(3)
        .build()?;

    let outcome = agent
        .run(Task::new("What is the capital of France?"))
        .await?;

    println!("Termination: {:?}", outcome.termination);

    // `critic.assessed` event carries the judge's verdict — find it
    // and print the note/score.
    for event in sink.events() {
        if let EventKind::CriticAssessed(payload) = &event.kind {
            println!("critic.assessed payload: {payload}");
        }
    }

    Ok(())
}
