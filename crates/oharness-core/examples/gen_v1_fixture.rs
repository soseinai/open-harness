use oharness_core::event::{
    EventKind, LlmFailedPayload, LlmRequestPayload, LlmResponsePayload, MetaPayload,
    RunFinishedPayload, RunStartedPayload, SchemaVersion, ToolCallFinishedPayload,
    ToolCallStartedPayload, TurnFinishedPayload, TurnPayload,
};
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, Event, LlmCapabilities, Message, MetadataMap,
    ModelId, RunId, StopReason, Task, Usage,
};
use std::fs::File;
use std::io::Write;

fn main() {
    let run_id = RunId::new();
    let capabilities = LlmCapabilities {
        streaming: true,
        prompt_caching: true,
        parallel_tool_use: true,
        vision: true,
        thinking: true,
        structured_output: false,
        max_context_tokens: 200_000,
        max_output_tokens: 8192,
    };
    let task = Task::new("inspect the repo").with_id("smoke-1");

    let events = vec![
        Event::new(
            0,
            run_id,
            "run-0",
            EventKind::Meta(MetaPayload {
                schema_version: SchemaVersion::CURRENT,
                harness_version: "0.1.0".into(),
                task_snapshot: task.clone(),
                llm_capabilities: capabilities.clone(),
            }),
        ),
        Event::new(
            1,
            run_id,
            "run-0",
            EventKind::RunStarted(RunStartedPayload {
                extra: MetadataMap::new(),
            }),
        ),
        Event::new(
            2,
            run_id,
            "turn-0",
            EventKind::TurnStarted(TurnPayload { turn_index: 0 }),
        ),
        Event::new(
            3,
            run_id,
            "llm-0",
            EventKind::LlmRequest(LlmRequestPayload {
                request: CompletionRequest {
                    messages: vec![Message::user_text("inspect the repo")],
                    tools: vec![],
                    system: Some("You are an agent.".into()),
                    max_tokens: Some(1024),
                    temperature: None,
                    stop_sequences: vec![],
                    cache_hints: Default::default(),
                    extensions: MetadataMap::new(),
                },
                provider: Some("anthropic".into()),
            }),
        ),
        Event::new(
            4,
            run_id,
            "llm-0",
            EventKind::LlmResponse(LlmResponsePayload {
                response: CompletionResponse {
                    id: "msg_01".into(),
                    model: ModelId::new("claude-sonnet-4-5"),
                    content: vec![
                        Content::text("Let me check."),
                        Content::ToolUse {
                            id: "tu_1".into(),
                            name: "fs_list".into(),
                            input: serde_json::json!({"path": "."}),
                        },
                    ],
                    stop_reason: StopReason::ToolUse,
                    usage: Usage {
                        tokens_input: 42,
                        tokens_output: 18,
                        ..Default::default()
                    },
                },
            }),
        ),
        Event::new(
            5,
            run_id,
            "tool-0",
            EventKind::ToolCallStarted(ToolCallStartedPayload {
                tool_name: "fs_list".into(),
                tool_use_id: "tu_1".into(),
                input: serde_json::json!({"path": "."}),
            }),
        ),
        Event::new(
            6,
            run_id,
            "tool-0",
            EventKind::ToolCallFinished(ToolCallFinishedPayload {
                tool_name: "fs_list".into(),
                tool_use_id: "tu_1".into(),
                output: serde_json::json!([{"type": "text", "text": "Cargo.toml\nsrc/"}]),
                truncated: false,
            }),
        ),
        Event::new(
            7,
            run_id,
            "turn-0",
            EventKind::TurnFinished(TurnFinishedPayload {
                turn_index: 0,
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    tokens_input: 42,
                    tokens_output: 18,
                    ..Default::default()
                },
                tool_calls: 1,
            }),
        ),
        // Second turn: llm.failed path, for coverage.
        Event::new(
            8,
            run_id,
            "turn-1",
            EventKind::TurnStarted(TurnPayload { turn_index: 1 }),
        ),
        Event::new(
            9,
            run_id,
            "llm-1",
            EventKind::LlmFailed(LlmFailedPayload {
                reason: "rate limited".into(),
            }),
        ),
        Event::new(
            10,
            run_id,
            "run-0",
            EventKind::RunFinished(RunFinishedPayload {
                termination: "Completed { reason: EndTurn }".into(),
                turns: 1,
                tool_calls: 1,
                extra: MetadataMap::new(),
            }),
        ),
    ];

    let path = "crates/oharness-core/testdata/trajectories/v1.0/smoke.jsonl";
    let mut file = File::create(path).unwrap();
    for event in events {
        let line = serde_json::to_vec(&event).unwrap();
        file.write_all(&line).unwrap();
        file.write_all(b"\n").unwrap();
    }
    println!("wrote {path}");
}
