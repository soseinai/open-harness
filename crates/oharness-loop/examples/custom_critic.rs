//! `custom_critic` — implement the `Critic` trait from scratch and
//! attach it to an agent.
//!
//! Shows:
//! 1. What implementing `Critic` looks like end-to-end (it's tiny —
//!    one trait method, one verdict value).
//! 2. How a `Reject` verdict surfaces at the loop layer: the run
//!    terminates with `Termination::Failed { category: Critic }` and
//!    a `critic.rejected` event lands in the trajectory.
//!
//! The shipped critics (`LlmJudgeCritic`, `TestCritic`,
//! `RegexDenyCritic`, `ConstitutionalCritic`) all follow the same
//! pattern; this example is the template for writing your own.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_critic -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Termination, Usage,
};
use oharness_critic::{
    AggregationPolicy, AssessmentContext, CompositeCritic, Critic, CriticVerdict,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::Arc;

// ---------------------------------------------------------------------
// Custom critic: rejects any assistant turn that hedges.
//
// Real critics care about more than one phrase — the pattern is the
// same: look at `ctx.latest_turn.message`, inspect the text blocks,
// emit a verdict.
// ---------------------------------------------------------------------

struct NoHedgingCritic;

#[async_trait]
impl Critic for NoHedgingCritic {
    fn name(&self) -> &str {
        "no-hedging"
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        // The `AssistantTurn` carries `message`, which is a
        // `Message::Assistant { content, .. }`. We scan the text
        // blocks for hedge-words.
        let oharness_core::Message::Assistant { content, .. } = &ctx.latest_turn.message else {
            // Not an assistant turn — nothing to assess. Accept.
            return CriticVerdict::Accept;
        };

        let joined_text = content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();

        const HEDGES: &[&str] = &["i'm not sure", "i am not sure", "maybe", "possibly"];
        for hedge in HEDGES {
            if joined_text.contains(hedge) {
                return CriticVerdict::Reject {
                    reason: format!("response hedges: found '{hedge}'"),
                };
            }
        }
        CriticVerdict::Accept
    }
}

// ---------------------------------------------------------------------
// Scripted LLM that hedges on its one and only turn — so the critic
// fires. Swap in a confident response to see the `Accept` path.
// ---------------------------------------------------------------------

struct ScriptedHedgeLlm;

#[async_trait]
impl Llm for ScriptedHedgeLlm {
    fn name(&self) -> &str {
        "scripted"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "msg_1".into(),
            model: ModelId::new("scripted-hedger"),
            content: vec![Content::text("I'm not sure what you're asking.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 8,
                tokens_output: 9,
                ..Default::default()
            },
        })
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `CompositeCritic` is the entry point — a single critic still
    // goes through it so the aggregation policy is explicit. Use
    // `AggregationPolicy::FirstReject` for fail-fast behavior, or
    // `AllMustAccept` / `MajorityVote` for voting semantics.
    let critics = Arc::new(
        CompositeCritic::new("hedge-guard", AggregationPolicy::FirstReject)
            .push(Box::new(NoHedgingCritic)),
    );

    let sink = Arc::new(InMemorySink::new());
    let agent = Agent::builder()
        .with_llm(Arc::new(ScriptedHedgeLlm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_critics(critics)
        .with_max_turns(3)
        .build()?;

    let outcome = agent.run(Task::new("figure it out")).await?;

    // The critic rejects on turn 1 → run fails with category=Critic.
    match &outcome.termination {
        Termination::Failed { error, .. } => {
            println!("Termination: Failed (category={:?})", error.category);
            println!("Critic message: {}", error.message);
        }
        other => println!("Termination: {other:?}"),
    }

    // The trajectory carries a `critic.rejected` event so downstream
    // analysis (or Reflexion) can see exactly what the critic saw.
    let rejections = sink
        .events()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::CriticRejected(_)))
        .count();
    println!("critic.rejected events: {rejections}");
    Ok(())
}
