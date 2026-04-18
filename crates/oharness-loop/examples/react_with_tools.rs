//! `react_with_tools` — multi-turn ReAct agent that actually calls a tool.
//!
//! Sibling example to `hello_scripted`. Where `hello_scripted` is the
//! "minimum viable run" (one turn, no tool use), this one shows the
//! canonical ReAct loop: the assistant emits a `tool_use` block, the
//! loop dispatches it to a real [`FsToolSet`], the tool result is
//! threaded back into the conversation, and the agent produces a
//! final text turn.
//!
//! The LLM is scripted (no network, no cost, no API key), so this
//! runs identically on every CI machine. Swap in `AnthropicLlm` /
//! `OpenAiLlm` and everything else stays the same.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example react_with_tools -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    Task, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// A two-response script: first the assistant calls `fs_list`, then
/// it summarizes the result as a text turn.
struct ScriptedLlm {
    responses: Vec<CompletionResponse>,
    cursor: AtomicU32,
}

#[async_trait]
impl Llm for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst) as usize;
        self.responses
            .get(idx)
            .cloned()
            .ok_or(LlmError::Unsupported("script exhausted"))
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let script = vec![
        // Turn 1 — assistant decides to look around.
        CompletionResponse {
            id: "msg_001".into(),
            model: ModelId::new("scripted-tools-example"),
            content: vec![
                Content::text("Let me list the current directory to see what's here."),
                Content::ToolUse {
                    id: "tu_1".into(),
                    name: "fs_list".into(),
                    input: json!({"path": "."}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                tokens_input: 12,
                tokens_output: 40,
                ..Default::default()
            },
        },
        // Turn 2 — assistant reacts to the tool result.
        CompletionResponse {
            id: "msg_002".into(),
            model: ModelId::new("scripted-tools-example"),
            content: vec![Content::text(
                "I can see the repository layout. There's a `crates/` \
                 directory, which is the Cargo workspace root.",
            )],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 80,
                tokens_output: 30,
                ..Default::default()
            },
        },
    ];

    let llm = Arc::new(ScriptedLlm {
        responses: script,
        cursor: AtomicU32::new(0),
    });

    // InMemorySink captures the trajectory so we can inspect the
    // event trace at the end.
    let sink = Arc::new(InMemorySink::new());

    let agent = Agent::builder()
        .with_llm(llm)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()?;

    let outcome = agent.run(Task::new("inspect the repo")).await?;

    println!("Termination: {:?}", outcome.termination);
    println!(
        "Turns: {} | tool calls: {} | tokens in/out: {}/{}",
        outcome.usage.turns,
        outcome.usage.tool_calls,
        outcome.usage.tokens_input,
        outcome.usage.tokens_output,
    );
    if let Some(Message::Assistant { content, .. }) = outcome.final_messages.last() {
        for c in content {
            if let Content::Text { text } = c {
                println!("Assistant: {text}");
            }
        }
    }

    // Show how many trajectory events were captured — full shape is
    // JSONL via `FileSink` in real runs.
    println!("Trajectory events captured: {}", sink.events().len());

    Ok(())
}
