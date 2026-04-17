//! End-to-end smoke test for the M1a agent loop.
//!
//! Uses a scripted `Llm` (no real API calls) + a `fs_read` tool. Verifies that the
//! loop drives a tool call, threads the result back, and produces the expected
//! trajectory shape.

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
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
            .ok_or(LlmError::Unsupported("ran off the end of the script"))
    }
    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::test]
async fn scripted_tool_call_roundtrip() {
    // Script: turn 1 returns a tool_use for fs_list on ".", turn 2 returns text + EndTurn.
    let tool_use_response = CompletionResponse {
        id: "msg_001".into(),
        model: ModelId::new("scripted-test"),
        content: vec![
            Content::text("Let me look around."),
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
    };
    let final_response = CompletionResponse {
        id: "msg_002".into(),
        model: ModelId::new("scripted-test"),
        content: vec![Content::text("Done.")],
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            tokens_input: 20,
            tokens_output: 2,
            ..Default::default()
        },
    };
    let llm = Arc::new(ScriptedLlm {
        responses: vec![tool_use_response, final_response],
        cursor: AtomicU32::new(0),
    });

    let sink = Arc::new(InMemorySink::new());
    let agent = Agent::builder()
        .with_llm(llm)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()
        .expect("agent build");

    let outcome = agent
        .run(Task::new("inspect the repo"))
        .await
        .expect("run ok");

    assert!(matches!(outcome.termination, Termination::Completed { .. }));
    assert_eq!(outcome.usage.turns, 2);
    assert_eq!(outcome.usage.tool_calls, 1);

    let events = sink.events();

    // First event must be meta.
    assert!(
        matches!(events[0].kind, EventKind::Meta(_)),
        "first event must be Meta"
    );
    // We should have run.started and run.finished bracketing.
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::RunStarted(_))));
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::RunFinished(_))));
    // At least one llm.request/response pair and at least one tool.call.started.
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::LlmRequest(_))));
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::LlmResponse(_))));
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::ToolCallStarted(_))));
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::ToolCallFinished(_))));
}
