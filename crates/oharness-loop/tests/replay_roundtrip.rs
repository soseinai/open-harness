//! Record a trajectory from a live ReactLoop, then replay it via
//! [`ReplayLlm`] and assert the replay produces the same final messages +
//! termination shape (plan §9.6 / docs/remaining-work.md §2.5).

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::{DriftPolicy, InMemorySink, ReplayLlm, ReplayMode};
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

fn script() -> Vec<CompletionResponse> {
    vec![
        CompletionResponse {
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
        },
        CompletionResponse {
            id: "msg_002".into(),
            model: ModelId::new("scripted-test"),
            content: vec![Content::text("Done.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 20,
                tokens_output: 2,
                ..Default::default()
            },
        },
    ]
}

#[tokio::test]
async fn replay_matches_live_outcome() {
    // ---- record ----
    let llm_live: Arc<dyn Llm> = Arc::new(ScriptedLlm {
        responses: script(),
        cursor: AtomicU32::new(0),
    });
    let live_sink = Arc::new(InMemorySink::new());
    let live_agent = Agent::builder()
        .with_llm(llm_live)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(live_sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()
        .expect("live agent build");

    let live_outcome = live_agent
        .run(Task::new("inspect the repo"))
        .await
        .expect("live run ok");
    assert!(matches!(
        live_outcome.termination,
        Termination::Completed { .. }
    ));
    let live_trajectory = live_sink.events();

    // ---- replay ----
    let replay = ReplayLlm::from_events(
        live_trajectory.clone(),
        ReplayMode::Positional,
        DriftPolicy::default(),
    )
    .expect("replay init");
    let replay_sink = Arc::new(InMemorySink::new());
    let replay_agent = Agent::builder()
        .with_llm(Arc::new(replay))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(replay_sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()
        .expect("replay agent build");

    let replay_outcome = replay_agent
        .run(Task::new("inspect the repo"))
        .await
        .expect("replay run ok");

    // ---- compare ----
    assert_eq!(replay_outcome.usage.turns, live_outcome.usage.turns);
    assert_eq!(
        replay_outcome.usage.tool_calls,
        live_outcome.usage.tool_calls
    );
    assert!(matches!(
        replay_outcome.termination,
        Termination::Completed { .. }
    ));

    // Final assistant message text matches.
    let live_last = last_assistant_text(&live_outcome.final_messages);
    let replay_last = last_assistant_text(&replay_outcome.final_messages);
    assert_eq!(live_last, replay_last);
    assert_eq!(replay_last.as_deref(), Some("Done."));
}

fn last_assistant_text(messages: &[oharness_core::Message]) -> Option<String> {
    messages.iter().rev().find_map(|m| match m {
        oharness_core::Message::Assistant { content, .. } => content.iter().find_map(|c| match c {
            Content::Text { text } => Some(text.clone()),
            _ => None,
        }),
        _ => None,
    })
}
