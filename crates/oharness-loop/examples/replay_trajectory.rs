//! `replay_trajectory` — run an agent, record every event to a
//! JSONL trajectory file, then replay that file via [`ReplayLlm`]
//! and verify the re-drive matches the original.
//!
//! This is the "scientific" path: you get bit-for-bit reproducibility
//! of a recorded run without needing the underlying provider's API
//! key, network, or dollars. Replay is how debuggers / post-mortems
//! / paper-supplement reproductions work.
//!
//! `ReplayMode::Positional` (used here) pairs the Nth live
//! `llm.request` in the replay loop with the Nth recorded
//! `llm.response` — no byte-for-byte input comparison, so the replay
//! tolerates minor non-determinism in request shape. `ReplayMode::Strict`
//! adds canonical-JSON equality and emits `critic.failed` on drift,
//! controlled via `DriftPolicy::WarnAndContinue` (default) or
//! `DriftPolicy::Fail`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example replay_trajectory -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    Task, Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::{DriftPolicy, FileSink, ReplayLlm, ReplayMode};
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

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

fn script() -> Vec<CompletionResponse> {
    vec![
        CompletionResponse {
            id: "msg_001".into(),
            model: ModelId::new("scripted-replay-example"),
            content: vec![
                Content::text("Let me look."),
                Content::ToolUse {
                    id: "tu_1".into(),
                    name: "fs_list".into(),
                    input: json!({"path": "."}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                tokens_input: 10,
                tokens_output: 5,
                ..Default::default()
            },
        },
        CompletionResponse {
            id: "msg_002".into(),
            model: ModelId::new("scripted-replay-example"),
            content: vec![Content::text("Found a crates/ directory.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 20,
                tokens_output: 6,
                ..Default::default()
            },
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------
    // Phase 1: live run that records every event straight to a JSONL
    // trajectory file — the on-disk format external tooling (`jq`,
    // paper-supplement analysis scripts, `ReplayLlm::from_path`)
    // consumes.
    // -----------------------------------------------------------------
    let tempdir = tempfile::tempdir()?;
    let path = tempdir.path().join("trajectory.jsonl");
    println!("[phase 1] live run → {}", path.display());

    let sink = Arc::new(FileSink::to_path(&path).await?);
    let live_agent = Agent::builder()
        .with_llm(Arc::new(ScriptedLlm {
            responses: script(),
            cursor: AtomicU32::new(0),
        }))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()?;
    let live_outcome = live_agent.run(Task::new("look around")).await?;

    // Drain the writer task so every queued event is on disk before
    // `ReplayLlm::from_path` reads the file. `flush()` runs cleanly
    // against the shared `Arc<FileSink>` — the close signal lets the
    // writer exit without waiting for every `Arc` clone to drop.
    sink.flush().await?;

    println!(
        "  termination: {:?}, turns: {}",
        live_outcome.termination, live_outcome.usage.turns,
    );

    // -----------------------------------------------------------------
    // Phase 2: replay directly from the recorded file. Under the hood
    // `from_path` streams the JSONL and cherry-picks `llm.response`
    // events into the replay queue.
    // -----------------------------------------------------------------
    println!("[phase 2] replay");
    let replay =
        ReplayLlm::from_path(&path, ReplayMode::Positional, DriftPolicy::default()).await?;
    let replay_agent = Agent::builder()
        .with_llm(Arc::new(replay))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()?;
    let replay_outcome = replay_agent.run(Task::new("look around")).await?;

    println!(
        "  termination: {:?}, turns: {}, final: {}",
        replay_outcome.termination,
        replay_outcome.usage.turns,
        last_assistant_text(&replay_outcome.final_messages).unwrap_or_else(|| "<none>".into()),
    );

    // -----------------------------------------------------------------
    // Phase 3: assert the replay matches the live run.
    // -----------------------------------------------------------------
    assert_eq!(replay_outcome.usage.turns, live_outcome.usage.turns);
    assert_eq!(
        replay_outcome.usage.tool_calls,
        live_outcome.usage.tool_calls
    );
    assert!(matches!(
        (&live_outcome.termination, &replay_outcome.termination),
        (Termination::Completed { .. }, Termination::Completed { .. }),
    ));
    assert_eq!(
        last_assistant_text(&live_outcome.final_messages),
        last_assistant_text(&replay_outcome.final_messages),
    );
    println!("[phase 3] replay output matches live run ✔");

    Ok(())
}

fn last_assistant_text(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|m| match m {
        Message::Assistant { content, .. } => content.iter().find_map(|c| match c {
            Content::Text { text } => Some(text.clone()),
            _ => None,
        }),
        _ => None,
    })
}
