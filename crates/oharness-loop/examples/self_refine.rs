//! `self_refine` — critic-driven in-place revision (`CriticVerdict::Revise`).
//!
//! Where `custom_critic` emits `Reject` and terminates the run,
//! self-refine emits **`Revise { replacement, reason }`** — the
//! critic hands the loop a corrected `AssistantTurn`, and the loop
//! swaps the assistant message in place, emits
//! `critic.revised` + `turn.revised` events, and continues. The
//! loop caps revision depth at `AgentConfig::revision_depth_cap`
//! (default 3); past the cap, `Revise` converts to `Reject`.
//!
//! The pattern lets a critic act like a "proofreader" without
//! re-hitting the LLM — cheap, deterministic, and replayable. It's
//! distinct from `Reflexion` (next-episode feedback) and from a
//! retry layer that regenerates from the model (that's plan §5.5
//! middleware territory).
//!
//! Run with:
//!
//! ```bash
//! cargo run --example self_refine -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    AssistantTurn, CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message,
    MetadataMap, ModelId, StopReason, Task, Termination, Usage,
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
// ProofreadHedges — rewrites hedge phrases into a confident version.
//
// Real-world analogues: strip PII, normalize terminology, collapse
// whitespace — anything the critic can fix mechanically without
// going back to the model.
// ---------------------------------------------------------------------

struct ProofreadHedges;

#[async_trait]
impl Critic for ProofreadHedges {
    fn name(&self) -> &str {
        "proofread-hedges"
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        let Message::Assistant { content, .. } = &ctx.latest_turn.message else {
            return CriticVerdict::Accept;
        };

        // Build the rewritten text blocks. If nothing changed, Accept.
        let mut any_change = false;
        let rewritten: Vec<Content> = content
            .iter()
            .map(|c| match c {
                Content::Text { text } => {
                    let cleaned = replace_hedges(text);
                    if cleaned != *text {
                        any_change = true;
                    }
                    Content::Text { text: cleaned }
                }
                other => other.clone(),
            })
            .collect();

        if !any_change {
            return CriticVerdict::Accept;
        }

        // Construct the replacement AssistantTurn. The loop swaps it
        // into the conversation and continues.
        let replacement_msg = Message::Assistant {
            content: rewritten,
            stop_reason: Some(ctx.latest_turn.stop_reason.clone()),
            meta: MetadataMap::new(),
        };
        let replacement = AssistantTurn::new(
            ctx.latest_turn.turn_index,
            ctx.latest_turn.span_id.clone(),
            replacement_msg,
            ctx.latest_turn.usage.clone(),
            ctx.latest_turn.stop_reason.clone(),
        );

        CriticVerdict::Revise {
            replacement,
            reason: "removed hedge phrases".into(),
        }
    }
}

fn replace_hedges(text: &str) -> String {
    // Minimal illustrative rewrite — a real impl would use proper
    // tokenization / regex. This is enough to exercise the Revise
    // wiring.
    text.replace("I'm not sure,", "")
        .replace("I'm not sure", "")
        .replace("maybe", "")
        .replace("  ", " ")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------
// Scripted LLM that hedges. One call, one response.
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
            content: vec![Content::text("I'm not sure, but maybe the answer is 42.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 10,
                tokens_output: 12,
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
    let critics = Arc::new(
        CompositeCritic::new("proofreader", AggregationPolicy::FirstReject)
            .push(Box::new(ProofreadHedges)),
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

    let outcome = agent.run(Task::new("tell me a number")).await?;

    // The critic's Revise verdict rewrote the turn in place; the run
    // completed normally (not Failed).
    println!("Termination: {:?}", outcome.termination);
    assert!(matches!(outcome.termination, Termination::Completed { .. }));

    // Print the final assistant text — should be the rewritten
    // (hedge-free) version, not the original.
    if let Some(Message::Assistant { content, .. }) = outcome.final_messages.last() {
        for c in content {
            if let Content::Text { text } = c {
                println!("Final assistant text: {text:?}");
            }
        }
    }

    // Count revision events — the loop emits these per Revise verdict.
    let events = sink.events();
    let critic_revised = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::CriticRevised(_)))
        .count();
    let turn_revised = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::TurnRevised(_)))
        .count();
    println!("critic.revised events: {critic_revised}");
    println!("turn.revised events: {turn_revised}");

    Ok(())
}
