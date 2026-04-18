//! `hello_scripted` — the "first agent in 10 lines" example.
//!
//! Builds a minimal [`Agent`] wired with a scripted Llm (no real API
//! calls, no cost), the shipped `fs` tool kit, and the default
//! `ReactLoop`. Runs a single turn and prints the resulting
//! termination + final assistant message.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example hello_scripted -p oharness-loop
//! ```
//!
//! Intent: this is the entry-point demo users read first, so it
//! deliberately avoids any of the larger pieces (critics, budgets,
//! trajectory files, middleware, reflexion). Those have their own
//! examples in the same directory.

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use std::sync::Arc;

/// A scripted `Llm` that returns a one-turn canned response. Real
/// adapters (`AnthropicLlm`, `OpenAiLlm`, …) slot in here without
/// any other changes to the agent.
struct ScriptedLlm;

#[async_trait]
impl Llm for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "hello".into(),
            model: ModelId::new("scripted-example"),
            content: vec![Content::text(
                "Hello from open-harness! Running against a scripted LLM; the \
                 trajectory of this run is captured via the default middleware \
                 stack.",
            )],
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
    let agent = Agent::builder()
        .with_llm(Arc::new(ScriptedLlm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()?;

    let outcome = agent.run(Task::new("say hello")).await?;

    println!("Termination: {:?}", outcome.termination);
    println!(
        "Turns: {}, tool calls: {}",
        outcome.usage.turns, outcome.usage.tool_calls
    );
    if let Some(oharness_core::Message::Assistant { content, .. }) = outcome.final_messages.last() {
        for c in content {
            if let Content::Text { text } = c {
                println!("Assistant: {text}");
            }
        }
    }
    Ok(())
}
