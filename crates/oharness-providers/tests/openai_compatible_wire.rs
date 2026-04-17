//! Integration tests for the OpenAI-compatible variants. Verify each
//! factory's auth + base-URL + extra-header behavior actually lands on
//! the wire — a shape-only unit test can't catch a missing `bearer_auth()`
//! call or a swapped-in URL.

#![cfg(all(feature = "openrouter", feature = "ollama", feature = "vllm"))]

use oharness_core::{CompletionRequest, Message};
use oharness_llm::Llm;
use oharness_providers::{Ollama, OpenRouter, Vllm};
use serde_json::json;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_body() -> serde_json::Value {
    json!({
        "id": "id-1",
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    })
}

fn req() -> CompletionRequest {
    CompletionRequest::new(vec![Message::user_text("hi")])
}

// ---------- OpenRouter: bearer auth present, attribution headers forwarded ----------

#[tokio::test]
async fn openrouter_sends_bearer_auth_and_attribution_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test-key"))
        .and(header("HTTP-Referer", "https://example.com/agent"))
        .and(header("X-Title", "test-agent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body()))
        .mount(&server)
        .await;

    let llm = OpenRouter::new("sk-test-key", "anthropic/claude-sonnet-4-5")
        .with_base_url(format!("{}/v1/chat/completions", server.uri()))
        .with_extra_header("HTTP-Referer", "https://example.com/agent")
        .with_extra_header("X-Title", "test-agent");

    assert_eq!(llm.name(), "openrouter");
    llm.complete(req()).await.expect("openrouter complete");
}

// ---------- Ollama: NO authorization header, no auth leak ----------

#[tokio::test]
async fn ollama_does_not_send_authorization_header() {
    let server = MockServer::start().await;
    // Primary mount: match the request only if it does NOT carry an
    // `authorization` header. wiremock doesn't have a "header absent"
    // matcher directly, so layer a catch-all mount with higher priority
    // (numerically lower) that matches when the auth header IS present —
    // if Ollama ever leaks auth, this mount fires and the primary doesn't.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header_exists("authorization"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string("authorization should not be sent"),
        )
        .with_priority(1) // tried first
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body()))
        .with_priority(5)
        .mount(&server)
        .await;

    let llm = Ollama::at(format!("{}/v1/chat/completions", server.uri()), "llama3.2");
    assert_eq!(llm.name(), "ollama");

    // Succeeds only if the request landed on the fallback mount (no auth).
    llm.complete(req()).await.expect("ollama complete");
}

// ---------- vLLM: no-auth mode + keyed mode ----------

#[tokio::test]
async fn vllm_at_sends_no_authorization() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header_exists("authorization"))
        .respond_with(
            ResponseTemplate::new(400).set_body_string("authorization should not be sent"),
        )
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body()))
        .with_priority(5)
        .mount(&server)
        .await;

    let llm = Vllm::at(
        format!("{}/v1/chat/completions", server.uri()),
        "meta-llama/Llama-3.1-8B-Instruct",
    );
    llm.complete(req()).await.expect("vllm (no auth) complete");
}

#[tokio::test]
async fn vllm_at_with_key_sends_bearer_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer vllm-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body()))
        .mount(&server)
        .await;

    let llm = Vllm::at_with_key(
        format!("{}/v1/chat/completions", server.uri()),
        "vllm-secret",
        "meta-llama/Llama-3.1-8B-Instruct",
    );
    llm.complete(req()).await.expect("vllm (keyed) complete");
}
