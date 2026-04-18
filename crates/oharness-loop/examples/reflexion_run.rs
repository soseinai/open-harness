//! `reflexion_run` — multi-episode agent loop that threads
//! reflections from one attempt into the next via
//! [`run_reflexion`].
//!
//! The pattern (plan §11.4 / §12.6):
//!
//! 1. Run the agent on a task.
//! 2. Score the outcome with a `TaskEvaluator`. If it passes, stop.
//! 3. Ask a `Reflector` to produce a note about what went wrong.
//! 4. The note is injected into the next episode's system prompt
//!    via `ReflectionInjector` — a shipped `RequestLayer` that
//!    prepends a `"Reflections from prior attempts:\n..."` block.
//! 5. Repeat up to `max_episodes` times.
//!
//! This example scripts three responses: the first two are vague
//! ("still thinking…") so the evaluator fails and the reflector
//! emits notes; the third responds with "done!" so the evaluator
//! passes and the loop stops. A tiny illustrative assertion at the
//! end verifies the stop-on-pass semantics.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example reflexion_run -p oharness-loop --features reflexion
//! ```

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, Episode, EvaluationResult, LlmCapabilities,
    ModelId, Reflection, RunOutcome, StopReason, Task, TaskEvaluator, Usage,
};
use oharness_critic::{ReflectionInjector, Reflector};
use oharness_llm::{ChunkStream, Llm, LlmError, LlmExt};
use oharness_loop::{run_reflexion, Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------
// Scripted LLM — returns response i per turn, wraps around if exhausted.
// ---------------------------------------------------------------------

struct CyclingLlm {
    responses: Vec<CompletionResponse>,
    cursor: AtomicU32,
}

#[async_trait]
impl Llm for CyclingLlm {
    fn name(&self) -> &str {
        "cycling"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst) as usize;
        Ok(self.responses[idx % self.responses.len()].clone())
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

fn text_response(text: &str) -> CompletionResponse {
    CompletionResponse {
        id: "msg".into(),
        model: ModelId::new("reflexion-example"),
        content: vec![Content::text(text)],
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            tokens_input: 5,
            tokens_output: 5,
            ..Default::default()
        },
    }
}

// ---------------------------------------------------------------------
// TaskEvaluator — "did the assistant actually finish?" A real
// evaluator would run tests, check a diff, or call an LLM judge.
// ---------------------------------------------------------------------

struct FinishedEvaluator;

#[async_trait]
impl TaskEvaluator for FinishedEvaluator {
    async fn evaluate(&self, _task: &Task, outcome: &RunOutcome) -> EvaluationResult {
        let ok = outcome.final_messages.iter().any(|m| {
            let oharness_core::Message::Assistant { content, .. } = m else {
                return false;
            };
            content.iter().any(|c| matches!(c, Content::Text { text } if text.to_ascii_lowercase().contains("done")))
        });
        if ok {
            EvaluationResult::pass()
        } else {
            EvaluationResult::fail()
        }
    }
}

// ---------------------------------------------------------------------
// Reflector — a canned "be more concrete" note. Real reflectors call
// an LLM to produce a pointed analysis; the shipped `LlmReflector` is
// the reference.
// ---------------------------------------------------------------------

struct NudgeReflector;

#[async_trait]
impl Reflector for NudgeReflector {
    fn name(&self) -> &str {
        "nudge"
    }

    async fn reflect(&self, ep: &Episode<'_>) -> Option<Reflection> {
        Some(Reflection::new(format!(
            "Episode {} didn't finish. Be concrete — say 'done!' when the task is complete.",
            ep.index
        )))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The injector is the middleware the reflector feeds into. Build
    // one, share it with both the LLM's layer stack AND the agent via
    // `.with_reflection_injector(..)` — `run_reflexion` needs the
    // latter to locate it between episodes.
    let injector = Arc::new(ReflectionInjector::new());

    // Script: two hedging responses, then "done!". Because `complete`
    // is called once per agent turn (not once per episode), each
    // episode picks up the response at the current cursor — that's
    // what drives the "first two fail, third passes" behavior.
    let llm = CyclingLlm {
        responses: vec![
            text_response("I'm still thinking — let me gather context."),
            text_response("I need more time to consider."),
            text_response("Task complete — done!"),
        ],
        cursor: AtomicU32::new(0),
    };
    let llm_with_reflections = Arc::new(llm.with_request_layer(injector.clone()));

    let agent = Agent::builder()
        .with_llm(llm_with_reflections)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_reflection_injector(injector)
        .with_max_turns(1)
        .build()?;

    let episodes = run_reflexion(
        &agent,
        Task::new("finish the task"),
        Arc::new(FinishedEvaluator),
        Arc::new(NudgeReflector),
        5,
    )
    .await?;

    println!("Episodes run: {}", episodes.len());
    for (i, ep) in episodes.iter().enumerate() {
        println!(
            "  episode {}: passed={} score={:.2} reflections_seen={}",
            i,
            ep.evaluation.passed,
            ep.evaluation.score,
            ep.prior_reflections.len(),
        );
    }

    // Loop should stop the moment an episode passes — so the final
    // episode is the one that passed.
    let last = episodes.last().expect("at least one episode ran");
    assert!(last.evaluation.passed);
    println!("Final episode passed ✔");

    Ok(())
}
