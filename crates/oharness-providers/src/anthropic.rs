//! Anthropic adapter — non-streaming completion only (M1a).
//!
//! Converts to/from Anthropic's Messages API (2023-06-01 version string). Vision,
//! documents, parallel tool use, and extended thinking are all expressible via the
//! canonical `Message` / `Content` shape this harness uses.
//!
//! Streaming (`stream()`) returns `LlmError::Unsupported("stream")` in M1a.

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    ToolSpec, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use serde::Deserialize;
use serde_json::{Value, json};
use std::env;
use std::time::Duration;

const ANTHROPIC_API: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct AnthropicLlm {
    http: reqwest::Client,
    api_key: String,
    model: ModelId,
    base_url: String,
    timeout: Duration,
    capabilities: LlmCapabilities,
}

impl AnthropicLlm {
    /// Construct from the `ANTHROPIC_API_KEY` environment variable, defaulting to
    /// `claude-sonnet-4-5`. Fails if the env var is missing.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LlmError::Authentication)?;
        Ok(Self::new(api_key, "claude-sonnet-4-5"))
    }

    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("oharness-providers/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client build");
        let model = ModelId::new(model.into());
        Self {
            http,
            api_key: api_key.into(),
            model,
            base_url: ANTHROPIC_API.to_string(),
            timeout: Duration::from_secs(120),
            capabilities: LlmCapabilities {
                // M1a does not ship streaming.
                streaming: false,
                prompt_caching: false,
                parallel_tool_use: true,
                vision: true,
                thinking: true,
                structured_output: false,
                max_context_tokens: 200_000,
                max_output_tokens: 8192,
            },
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }
}

#[async_trait]
impl Llm for AnthropicLlm {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = to_wire_request(&self.model, &req);
        let resp = self
            .http
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .map_err(reqwest_to_llm_err)?;

        let status = resp.status();
        let text = resp.text().await.map_err(reqwest_to_llm_err)?;

        if !status.is_success() {
            return Err(classify_http_error(status, &text));
        }

        let parsed: WireResponse = serde_json::from_str(&text)
            .map_err(|e| LlmError::MalformedResponse(format!("decode: {e}: {text}")))?;
        Ok(from_wire_response(parsed))
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

// ---------- request translation ----------

fn to_wire_request(model: &ModelId, req: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": model.as_str(),
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": wire_messages(&req.messages),
    });

    if let Some(sys) = &req.system {
        body["system"] = Value::String(sys.clone());
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = json!(temp);
    }
    if !req.stop_sequences.is_empty() {
        body["stop_sequences"] = json!(req.stop_sequences);
    }
    if !req.tools.is_empty() {
        body["tools"] = wire_tools(&req.tools);
    }
    // Known Anthropic extension: `anthropic.thinking = { type: "enabled", budget_tokens: n }`.
    if let Some(thinking) = req.extensions.get("anthropic.thinking") {
        body["thinking"] = thinking.clone();
    }
    body
}

fn wire_messages(messages: &[Message]) -> Value {
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        match m {
            Message::System { .. } => {
                // Anthropic puts system content outside `messages`. Callers that
                // rely on `CompletionRequest.system` instead of a system Message
                // get the correct behaviour; legacy system Messages are skipped.
            }
            Message::User { content, .. } => {
                out.push(json!({
                    "role": "user",
                    "content": wire_content(content),
                }));
            }
            Message::Assistant { content, .. } => {
                out.push(json!({
                    "role": "assistant",
                    "content": wire_content(content),
                }));
            }
        }
    }
    Value::Array(out)
}

fn wire_content(content: &[Content]) -> Value {
    let blocks: Vec<Value> = content
        .iter()
        .map(|c| match c {
            Content::Text(t) => json!({"type": "text", "text": t}),
            Content::ToolUse { id, name, input } => {
                json!({"type": "tool_use", "id": id, "name": name, "input": input})
            }
            Content::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => {
                // Anthropic expects content blocks inside the tool_result.
                let inner: Vec<Value> = output
                    .content
                    .iter()
                    .map(|c| match c {
                        Content::Text(t) => json!({"type": "text", "text": t}),
                        // Only text passes through in M1a; richer types land with
                        // vision/document support in later milestones.
                        _ => json!({"type": "text", "text": "[unsupported content block]"}),
                    })
                    .collect();
                json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": inner,
                    "is_error": *is_error,
                })
            }
            Content::Thinking(t) => {
                json!({"type": "thinking", "thinking": t})
            }
            // Vision/audio/document/citation — round-trip as text stubs for M1a.
            Content::Image(_) | Content::Document(_) | Content::Audio(_) | Content::Citation(_) => {
                json!({"type": "text", "text": "[unsupported content block type for M1a]"})
            }
        })
        .collect();
    Value::Array(blocks)
}

fn wire_tools(tools: &[ToolSpec]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect(),
    )
}

// ---------- response translation ----------

#[derive(Debug, Deserialize)]
struct WireResponse {
    id: String,
    model: String,
    content: Vec<WireBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    stop_sequence: Option<String>,
    usage: WireUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum WireBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

fn from_wire_response(w: WireResponse) -> CompletionResponse {
    let content: Vec<Content> = w
        .content
        .into_iter()
        .filter_map(|b| match b {
            WireBlock::Text { text } => Some(Content::Text(text)),
            WireBlock::ToolUse { id, name, input } => Some(Content::ToolUse { id, name, input }),
            WireBlock::Thinking { thinking } => Some(Content::Thinking(thinking)),
            WireBlock::Other => None,
        })
        .collect();

    let stop_reason = match w.stop_reason.as_deref() {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence(w.stop_sequence.unwrap_or_default()),
        Some("tool_use") => StopReason::ToolUse,
        Some("refusal") => StopReason::Refusal,
        Some(other) => StopReason::Error(other.to_string()),
        None => StopReason::EndTurn,
    };

    CompletionResponse {
        id: w.id,
        model: ModelId::new(w.model),
        content,
        stop_reason,
        usage: Usage {
            tokens_input: w.usage.input_tokens,
            tokens_output: w.usage.output_tokens,
            tokens_cache_read: w.usage.cache_read_input_tokens,
            tokens_cache_create: w.usage.cache_creation_input_tokens,
        },
    }
}

// ---------- error mapping ----------

fn reqwest_to_llm_err(e: reqwest::Error) -> LlmError {
    if e.is_timeout() {
        return LlmError::RateLimited { retry_after: None };
    }
    LlmError::Network(std::io::Error::other(e.to_string()))
}

fn classify_http_error(status: reqwest::StatusCode, body: &str) -> LlmError {
    match status.as_u16() {
        401 | 403 => LlmError::Authentication,
        429 => LlmError::RateLimited { retry_after: None },
        400 => {
            // Often context-too-long; the body mentions it explicitly.
            if body.to_lowercase().contains("context") && body.to_lowercase().contains("long") {
                LlmError::ContextTooLong { max: 0, got: 0 }
            } else {
                LlmError::MalformedResponse(format!("{status}: {body}"))
            }
        }
        _ => LlmError::MalformedResponse(format!("HTTP {status}: {body}")),
    }
}
