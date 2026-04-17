//! Integration tests for `AnthropicLlm::stream()` using a wiremock-backed
//! endpoint. Never hits the real Anthropic API — fixtures live inline.

use futures::StreamExt;
use oharness_core::{CompletionRequest, Content, Message};
use oharness_llm::{complete_from_stream, Chunk, Llm};
use oharness_providers::AnthropicLlm;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Representative Anthropic SSE stream: message_start → block_start → ping →
/// two text deltas → block_stop → message_delta (stop_reason + usage) →
/// message_stop. Uses literal `\n\n` frame delimiters with no trailing garbage.
const SSE_BODY: &str = "\
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01ABC\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: ping
data: {\"type\":\"ping\"}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":15}}

event: message_stop
data: {\"type\":\"message_stop\"}

";

fn non_streaming_body() -> serde_json::Value {
    json!({
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello world"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 25, "output_tokens": 15},
    })
}

async fn setup_server() -> MockServer {
    let server = MockServer::start().await;

    // Stream variant — matched first because it has lower priority (tighter match).
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(SSE_BODY.as_bytes().to_vec(), "text/event-stream"),
        )
        .with_priority(1)
        .mount(&server)
        .await;

    // Non-streaming fallback: serve the JSON equivalent.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(non_streaming_body()))
        .with_priority(5)
        .mount(&server)
        .await;

    server
}

fn build_request() -> CompletionRequest {
    CompletionRequest::new(vec![Message::user_text("Say hello")])
}

fn make_llm(server: &MockServer) -> AnthropicLlm {
    AnthropicLlm::new("test-key", "claude-sonnet-4-5")
        .with_base_url(format!("{}/v1/messages", server.uri()))
}

/// `Content` currently lacks `PartialEq` (and `#[serde(tag = "type")]` newtype
/// variants block `serde_json::to_value`), so the streaming vs. non-streaming
/// round-trip test compares content manually on the variants this fixture
/// exercises.
fn content_equal(a: &[Content], b: &[Content]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|pair| match pair {
        (Content::Text(x), Content::Text(y)) => x == y,
        (Content::Thinking(x), Content::Thinking(y)) => x == y,
        (
            Content::ToolUse {
                id: xi,
                name: xn,
                input: xp,
            },
            Content::ToolUse {
                id: yi,
                name: yn,
                input: yp,
            },
        ) => xi == yi && xn == yn && xp == yp,
        _ => false,
    })
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
    assert_eq!(
        kinds,
        vec![
            "message_start",
            "usage",
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
    assert!(
        content_equal(&via_complete.content, &via_stream.content),
        "content differs: complete={:?} stream={:?}",
        via_complete.content,
        via_stream.content
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

    // Surface-level sanity: content is a single text block "Hello world".
    assert_eq!(via_stream.content.len(), 1);
    assert!(matches!(&via_stream.content[0], Content::Text(t) if t == "Hello world"));
}

#[tokio::test]
async fn capabilities_advertise_streaming() {
    let llm = AnthropicLlm::new("test-key", "claude-sonnet-4-5");
    assert!(llm.capabilities().streaming);
}
