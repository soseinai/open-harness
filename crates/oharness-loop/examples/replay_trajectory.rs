//! `replay_trajectory` — run an agent, then replay the recorded
//! trajectory against [`ReplayLlm`] and verify the re-drive matches
//! the original.
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
//! The example also demonstrates writing the captured trajectory to a
//! JSONL file alongside the in-memory round-trip — that's the
//! on-disk format `oharness_trace::jsonl::read_events` + external
//! tools (`jq`, paper-supplement analysis scripts) consume.
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
use oharness_trace::{DriftPolicy, InMemorySink, ReplayLlm, ReplayMode};
use serde_json::json;
use std::io::Write;
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
    // Phase 1: live run that captures every event into an InMemorySink.
    // -----------------------------------------------------------------
    println!("[phase 1] live run");
    let sink = Arc::new(InMemorySink::new());
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
    let events = sink.events();
    println!(
        "  termination: {:?}, turns: {}, events: {}",
        live_outcome.termination,
        live_outcome.usage.turns,
        events.len(),
    );

    // -----------------------------------------------------------------
    // (Optional) write the captured trajectory to a JSONL file, the
    // on-disk format external tooling consumes. Uses plain std::io —
    // `FileSink` is the production-grade sink but requires careful
    // drop semantics to flush, so we keep the example self-contained
    // by serializing the captured events directly.
    // -----------------------------------------------------------------
    let tempdir = tempfile::tempdir()?;
    let path = tempdir.path().join("trajectory.jsonl");
    {
        let mut f = std::fs::File::create(&path)?;
        for event in &events {
            let line = serde_json::to_string(event)?;
            writeln!(f, "{line}")?;
        }
    }
    println!(
        "[phase 1.5] wrote {} events → {}",
        events.len(),
        path.display()
    );

    // -----------------------------------------------------------------
    // Phase 2: replay directly from the captured events (same result
    // as reading back from disk via `ReplayLlm::from_path`).
    // -----------------------------------------------------------------
    println!("[phase 2] replay");
    let replay = ReplayLlm::from_events(events, ReplayMode::Positional, DriftPolicy::default())?;
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
