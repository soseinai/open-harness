//! Integration tests for `OpenAiLlm::complete()` and `stream()` via a
//! wiremock-backed endpoint. Never hits the real OpenAI API.

#![cfg(feature = "openai")]

use futures::StreamExt;
use oharness_core::{CompletionRequest, Content, Message};
use oharness_llm::{complete_from_stream, Chunk, Llm};
use oharness_providers::OpenAiLlm;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// SSE stream shaped like a real Chat Completions response with one text
/// delta + one tool call. Follows OpenAI's canonical per-chunk shape:
/// first chunk has `role: "assistant"`, subsequent chunks carry `content`
/// or `tool_calls` deltas, last chunk has `finish_reason`, a separate
/// "usage" chunk lands when `include_usage` is set, and the stream ends
/// with `data: [DONE]`.
const SSE_BODY: &str = "\
data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}

data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}

data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}

data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}

data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}

data: [DONE]

";

fn non_streaming_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2}
    })
}

async fn setup_server() -> MockServer {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(SSE_BODY.as_bytes().to_vec(), "text/event-stream"),
        )
        .with_priority(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(non_streaming_body()))
        .with_priority(5)
        .mount(&server)
        .await;

    server
}

fn build_request() -> CompletionRequest {
    CompletionRequest::new(vec![Message::user_text("Say hello")])
}

fn make_llm(server: &MockServer) -> OpenAiLlm {
    OpenAiLlm::new("test-key", "gpt-4o")
        .with_base_url(format!("{}/v1/chat/completions", server.uri()))
}

fn chunk_label(c: &Chunk) -> &'static str {
    match c {
        Chunk::MessageStart { .. } => "message_start",
        Chunk::BlockStart { .. } => "block_start",
        Chunk::TextDelta { .. } => "text_delta",
        Chunk::ToolUseDelta { .. } => "tool_use_delta",
        Chunk::ThinkingDelta { .. } => "thinking_delta",
        Chunk::BlockStop { .. } => "block_stop",
        Chunk::StopReason { .. } => "stop_reason",
        Chunk::Usage { .. } => "usage",
        Chunk::MessageStop => "message_stop",
        Chunk::Raw { .. } => "raw",
    }
}

#[tokio::test]
async fn stream_emits_expected_chunk_sequence() {
    let server = setup_server().await;
    let llm = make_llm(&server);

    let mut stream = llm.stream(build_request()).await.expect("stream opens");

    let mut kinds = Vec::new();
    let mut text = String::new();
    while let Some(result) = stream.next().await {
        let chunk = result.expect("chunk ok");
        kinds.push(chunk_label(&chunk));
        if let Chunk::TextDelta { text: t, .. } = &chunk {
            text.push_str(t);
        }
    }

    assert_eq!(text, "Hello world");
    // MessageStart → BlockStart(Text) → two TextDeltas → BlockStop →
    // StopReason → Usage → MessageStop (synthesized at stream end).
    assert_eq!(
        kinds,
        vec![
            "message_start",
            "block_start",
            "text_delta",
            "text_delta",
            "block_stop",
            "stop_reason",
            "usage",
            "message_stop",
        ]
    );
}

#[tokio::test]
async fn complete_from_stream_matches_complete() {
    let server = setup_server().await;
    let llm = make_llm(&server);

    let via_complete = llm.complete(build_request()).await.expect("complete");
    let stream = llm.stream(build_request()).await.expect("stream opens");
    let via_stream = complete_from_stream(stream).await.expect("drain stream");

    assert_eq!(via_complete.id, via_stream.id);
    assert_eq!(
        serde_json::to_value(&via_complete.content).expect("ser complete"),
        serde_json::to_value(&via_stream.content).expect("ser stream"),
    );
    assert_eq!(via_complete.stop_reason, via_stream.stop_reason);
    assert_eq!(
        via_complete.usage.tokens_input,
        via_stream.usage.tokens_input
    );
    assert_eq!(
        via_complete.usage.tokens_output,
        via_stream.usage.tokens_output
    );
    assert_eq!(via_stream.content.len(), 1);
    assert!(matches!(&via_stream.content[0], Content::Text { text } if text == "Hello world"));
}

#[tokio::test]
async fn capabilities_advertise_streaming_and_structured_output() {
    let llm = OpenAiLlm::new("test-key", "gpt-4o");
    let caps = llm.capabilities();
    assert!(caps.streaming);
    assert!(caps.structured_output);
    // Anthropic-style cache_control doesn't apply to Chat Completions.
    assert!(!caps.prompt_caching);
}

#[tokio::test]
async fn request_body_sets_stream_options_include_usage_when_streaming() {
    // Mount a mock that ONLY matches when stream_options.include_usage is
    // true — if the adapter forgets to set it, the request 404s.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(SSE_BODY.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let llm = make_llm(&server);
    let mut stream = llm.stream(build_request()).await.expect("stream opens");
    while let Some(c) = stream.next().await {
        c.expect("chunk");
    }
}
