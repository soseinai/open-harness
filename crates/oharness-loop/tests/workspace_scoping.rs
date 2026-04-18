//! End-to-end tests for workspace scoping: an `Agent` built with
//! `.with_workspace(..)` must propagate that workspace into every
//! tool-call's `ToolContext`, so the shipped `fs` / `bash` tools
//! resolve paths relative to it (and refuse paths that escape).

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::context::Workspace;
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tempfile::tempdir;

/// A scripted LLM that emits one tool_use turn, then one EndTurn turn.
/// The tool_use's input is constructor-provided so each test can aim
/// `fs_read` at the path it wants to probe.
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

fn scripted_for_fs_read(path: &str) -> Arc<dyn Llm> {
    Arc::new(ScriptedLlm {
        responses: vec![
            CompletionResponse {
                id: "msg_1".into(),
                model: ModelId::new("scripted"),
                content: vec![
                    Content::text("Let me check the file."),
                    Content::ToolUse {
                        id: "tu_1".into(),
                        name: "fs_read".into(),
                        input: json!({ "path": path }),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            },
            CompletionResponse {
                id: "msg_2".into(),
                model: ModelId::new("scripted"),
                content: vec![Content::text("Done.")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ],
        cursor: AtomicU32::new(0),
    })
}

/// Extract the text content of the most recent User message — which is
/// the synthesized `tool_result` message the loop appended from the
/// single tool call this script made.
fn last_tool_result_text(outcome: &oharness_core::RunOutcome) -> String {
    outcome
        .final_messages
        .iter()
        .rev()
        .find_map(|m| match m {
            oharness_core::Message::User { content, .. } => content.iter().find_map(|c| match c {
                Content::ToolResult { output, .. } => output.content.iter().find_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                }),
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn fs_read_resolves_relative_to_attached_workspace() {
    let work = tempdir().expect("workspace tempdir");
    // Seed the workspace with a known file.
    let file_path = work.path().join("hello.txt");
    std::fs::write(&file_path, "greetings from the workspace").unwrap();

    let ws = Arc::new(Workspace::new(work.path().to_path_buf()));
    let agent = Agent::builder()
        .with_llm(scripted_for_fs_read("hello.txt"))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_workspace(ws)
        .with_event_sink(Arc::new(InMemorySink::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("inspect")).await.expect("run ok");
    assert!(matches!(outcome.termination, Termination::Completed { .. }));
    let text = last_tool_result_text(&outcome);
    assert_eq!(
        text, "greetings from the workspace",
        "fs_read should have resolved `hello.txt` inside the workspace"
    );
}

#[tokio::test]
async fn fs_read_rejects_paths_escaping_workspace() {
    let work = tempdir().expect("workspace tempdir");
    let outside = tempdir().expect("outside tempdir");
    // Seed a file OUTSIDE the workspace; agent must not be able to
    // reach it via `../` tricks.
    std::fs::write(outside.path().join("secret.txt"), "SHHH").unwrap();

    // Craft a path that uses `..` to escape the workspace root.
    let outside_rel = format!(
        "../{}/secret.txt",
        outside.path().file_name().unwrap().to_string_lossy()
    );
    // Both workspace and outside live under the system tempdir, so
    // this relative path actually resolves to the outside file — the
    // kind of escape fs.rs's normalize + prefix check rejects.

    let ws = Arc::new(Workspace::new(work.path().to_path_buf()));
    let agent = Agent::builder()
        .with_llm(scripted_for_fs_read(&outside_rel))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_workspace(ws)
        .with_event_sink(Arc::new(InMemorySink::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("probe")).await.expect("run ok");
    let text = last_tool_result_text(&outcome);
    assert!(
        text.contains("error:") && text.contains("escapes workspace root"),
        "fs_read should have rejected the escape attempt; got: {text:?}"
    );
    // Secret content must not appear.
    assert!(
        !text.contains("SHHH"),
        "escape attempt leaked secret content: {text:?}"
    );
}

#[tokio::test]
async fn fs_read_without_workspace_falls_back_to_cwd() {
    // Sanity-check the no-workspace path: agents built without
    // `.with_workspace(..)` should continue to work, resolving paths
    // relative to cwd (the existing M1a behaviour).
    let agent = Agent::builder()
        .with_llm(scripted_for_fs_read("Cargo.toml"))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(Arc::new(InMemorySink::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(3)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("probe")).await.expect("run ok");
    let text = last_tool_result_text(&outcome);
    assert!(
        text.starts_with("[package]") || text.contains("oharness-loop"),
        "expected the repo's Cargo.toml from cwd, got: {text:?}"
    );
}
