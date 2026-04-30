//! Anthropic adapter (M1b-α: streaming enabled).
//!
//! Converts to/from Anthropic's Messages API (2023-06-01 version string). Vision,
//! documents, parallel tool use, and extended thinking are all expressible via the
//! canonical `Message` / `Content` shape this harness uses.
//!
//! Streaming is implemented as a Server-Sent Events parser (no third-party SSE
//! client). Each decoded Anthropic event is translated to zero or more `Chunk`
//! values; unrecognised event and delta types are passed through verbatim as
//! `Chunk::Raw { provider: "anthropic", .. }`.

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, StreamExt};
use oharness_core::{
    CacheHints, CacheTtl, CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message,
    ModelId, StopReason, ToolSpec, Usage,
};
use oharness_llm::{BlockStartKind, Chunk, ChunkStream, LayerError, Llm, LlmError, LlmLayer};
use serde::Deserialize;
use serde_json::{json, Value};
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
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| LlmError::Authentication)?;
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
                streaming: true,
                prompt_caching: true,
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

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let mut body = to_wire_request(&self.model, &req);
        body["stream"] = Value::Bool(true);

        let resp = self
            .http
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .map_err(reqwest_to_llm_err)?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(reqwest_to_llm_err)?;
            return Err(classify_http_error(status, &text));
        }

        Ok(chunk_stream_from_response(resp))
    }
}

// ---------- request translation ----------

fn to_wire_request(model: &ModelId, req: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": model.as_str(),
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": wire_messages(&req.messages, &req.cache_hints),
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

fn wire_messages(messages: &[Message], cache_hints: &CacheHints) -> Value {
    // Pre-index breakpoints so each message knows its target TTL (if any).
    // Per plan §4.9 `CacheBreakpoint.message_index` is inclusive — the last
    // content block of that message receives Anthropic's `cache_control`
    // marker, declaring the prefix up through that block cacheable.
    let mut marks: std::collections::HashMap<usize, Option<CacheTtl>> =
        std::collections::HashMap::new();
    for bp in &cache_hints.breakpoints {
        marks.insert(bp.message_index, bp.ttl);
    }

    let mut out: Vec<Value> = Vec::new();
    for (idx, m) in messages.iter().enumerate() {
        match m {
            Message::System { .. } => {
                // Anthropic puts system content outside `messages`. Callers that
                // rely on `CompletionRequest.system` instead of a system Message
                // get the correct behaviour; legacy system Messages are skipped.
            }
            Message::User { content, .. } | Message::Assistant { content, .. } => {
                let mut blocks = wire_content_blocks(content);
                if let Some(&ttl) = marks.get(&idx) {
                    apply_cache_control(&mut blocks, ttl);
                }
                let role = if matches!(m, Message::User { .. }) {
                    "user"
                } else {
                    "assistant"
                };
                out.push(json!({
                    "role": role,
                    "content": Value::Array(blocks),
                }));
            }
        }
    }
    Value::Array(out)
}

fn apply_cache_control(blocks: &mut [Value], ttl: Option<CacheTtl>) {
    let Some(last) = blocks.last_mut() else {
        return;
    };
    let ttl_str = match ttl {
        Some(CacheTtl::Long) => "1h",
        Some(CacheTtl::Short) | None => "5m",
    };
    if let Some(obj) = last.as_object_mut() {
        obj.insert(
            "cache_control".to_string(),
            json!({ "type": "ephemeral", "ttl": ttl_str }),
        );
    }
}

fn wire_content_blocks(content: &[Content]) -> Vec<Value> {
    content
        .iter()
        .map(|c| match c {
            Content::Text { text } => json!({"type": "text", "text": text}),
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
                        Content::Text { text } => json!({"type": "text", "text": text}),
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
            Content::Thinking { thinking } => {
                json!({"type": "thinking", "thinking": thinking})
            }
            // Vision/audio/document/citation — round-trip as text stubs for M1a.
            Content::Image(_) | Content::Document(_) | Content::Audio(_) | Content::Citation(_) => {
                json!({"type": "text", "text": "[unsupported content block type for M1a]"})
            }
        })
        .collect()
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
            WireBlock::Text { text } => Some(Content::Text { text }),
            WireBlock::ToolUse { id, name, input } => Some(Content::ToolUse { id, name, input }),
            WireBlock::Thinking { thinking } => Some(Content::Thinking { thinking }),
            WireBlock::Other => None,
        })
        .collect();

    let stop_reason = w
        .stop_reason
        .as_deref()
        .map(|raw| map_stop_reason(raw, w.stop_sequence.clone()))
        .unwrap_or(StopReason::EndTurn);

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

// ---------- shared helpers ----------

fn map_stop_reason(raw: &str, stop_sequence: Option<String>) -> StopReason {
    match raw {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence(stop_sequence.unwrap_or_default()),
        "tool_use" => StopReason::ToolUse,
        "refusal" => StopReason::Refusal,
        other => StopReason::Error(other.to_string()),
    }
}

fn parse_usage(v: Option<&Value>) -> Usage {
    let Some(v) = v else {
        return Usage::default();
    };
    Usage {
        tokens_input: v.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
        tokens_output: v.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
        tokens_cache_read: v
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        tokens_cache_create: v
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

// ---------- SSE framing ----------

#[derive(Debug)]
struct SseFrame {
    /// Optional `event:` field. The Anthropic stream always sets it, but the
    /// authoritative discriminator is the `type` field inside `data`.
    #[allow(dead_code)]
    event: Option<String>,
    data: String,
}

/// Find the end of the first complete SSE frame in `buf` (`\n\n` or `\r\n\r\n`).
/// Returns `(frame_body_len, total_consumed_len)` — the frame body is
/// `&buf[..frame_body_len]`, and `total_consumed_len` includes the terminator.
fn find_frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        if buf[i] == b'\n' {
            if i + 1 < buf.len() && buf[i + 1] == b'\n' {
                return Some((i, i + 2));
            }
            if i + 2 < buf.len() && buf[i + 1] == b'\r' && buf[i + 2] == b'\n' {
                return Some((i, i + 3));
            }
        }
        i += 1;
    }
    None
}

fn parse_frame_body(bytes: &[u8]) -> Option<SseFrame> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut event = None;
    let mut data = String::new();
    let mut data_seen = false;
    for raw in text.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => event = Some(value.to_string()),
            "data" => {
                if data_seen {
                    data.push('\n');
                }
                data.push_str(value);
                data_seen = true;
            }
            _ => {}
        }
    }
    if !data_seen && event.is_none() {
        return None;
    }
    Some(SseFrame { event, data })
}

/// Extract one SSE frame from `buf`, removing consumed bytes on success.
fn extract_sse_frame(buf: &mut Vec<u8>) -> Option<SseFrame> {
    let (body_len, total) = find_frame_end(buf)?;
    let frame = parse_frame_body(&buf[..body_len]);
    buf.drain(..total);
    // Empty / comment-only frames return None from parse_frame_body; keep trying.
    frame
}

// ---------- event → Chunk decoder ----------

#[derive(Default)]
struct ChunkDecoder {
    /// Input-token count from `message_start.message.usage`; Anthropic's
    /// `message_delta.usage` only carries the updated output count, so we
    /// forward the initial figures along with each usage chunk.
    initial_usage: Option<Usage>,
}

fn missing(field: &str) -> LlmError {
    LlmError::MalformedResponse(format!("anthropic sse: missing field `{field}`"))
}

fn decode_frame(frame: &SseFrame, state: &mut ChunkDecoder) -> Result<Vec<Chunk>, LlmError> {
    let payload: Value = serde_json::from_str(&frame.data).map_err(|e| {
        LlmError::MalformedResponse(format!("anthropic sse decode: {e}: {}", frame.data))
    })?;

    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| missing("type"))?
        .to_string();

    match event_type.as_str() {
        "message_start" => {
            let msg = payload.get("message").ok_or_else(|| missing("message"))?;
            let id = msg
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("message.id"))?
                .to_string();
            let model = msg
                .get("model")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("message.model"))?
                .to_string();
            let usage = parse_usage(msg.get("usage"));
            state.initial_usage = Some(usage.clone());
            Ok(vec![
                Chunk::MessageStart {
                    id,
                    model: ModelId::new(model),
                },
                Chunk::Usage { usage },
            ])
        }
        "content_block_start" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| missing("index"))? as u32;
            let block = payload
                .get("content_block")
                .ok_or_else(|| missing("content_block"))?;
            let kind = block
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("content_block.type"))?;
            let start = match kind {
                "text" => BlockStartKind::Text,
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| missing("content_block.id"))?
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| missing("content_block.name"))?
                        .to_string();
                    BlockStartKind::ToolUse { name, id }
                }
                "thinking" => BlockStartKind::Thinking,
                _ => {
                    return Ok(vec![Chunk::Raw {
                        provider: "anthropic".to_string(),
                        event: payload,
                    }]);
                }
            };
            Ok(vec![Chunk::BlockStart { index, start }])
        }
        "content_block_delta" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| missing("index"))? as u32;
            let delta = payload.get("delta").ok_or_else(|| missing("delta"))?;
            let dtype = delta
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("delta.type"))?;
            let chunk = match dtype {
                "text_delta" => {
                    let text = delta
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    Chunk::TextDelta { index, text }
                }
                "input_json_delta" => {
                    let partial_json = delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    Chunk::ToolUseDelta {
                        index,
                        partial_json,
                    }
                }
                "thinking_delta" => {
                    let text = delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    Chunk::ThinkingDelta { index, text }
                }
                _ => Chunk::Raw {
                    provider: "anthropic".to_string(),
                    event: payload,
                },
            };
            Ok(vec![chunk])
        }
        "content_block_stop" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| missing("index"))? as u32;
            Ok(vec![Chunk::BlockStop { index }])
        }
        "message_delta" => {
            let mut out = Vec::new();
            if let Some(delta) = payload.get("delta") {
                if let Some(raw) = delta.get("stop_reason").and_then(Value::as_str) {
                    let seq = delta
                        .get("stop_sequence")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    out.push(Chunk::StopReason {
                        reason: map_stop_reason(raw, seq),
                    });
                }
            }
            if let Some(u) = payload.get("usage") {
                let delta_usage = parse_usage(Some(u));
                let merged = merge_usage(state.initial_usage.as_ref(), &delta_usage);
                out.push(Chunk::Usage { usage: merged });
            }
            Ok(out)
        }
        "message_stop" => Ok(vec![Chunk::MessageStop]),
        "ping" => Ok(Vec::new()),
        "error" => Err(classify_sse_error(payload.get("error"))),
        _ => Ok(vec![Chunk::Raw {
            provider: "anthropic".to_string(),
            event: payload,
        }]),
    }
}

fn merge_usage(initial: Option<&Usage>, delta: &Usage) -> Usage {
    let mut out = initial.cloned().unwrap_or_default();
    // Only overwrite non-zero fields from the delta so missing-field semantics
    // line up with the Anthropic spec (message_delta.usage may carry only
    // `output_tokens`).
    if delta.tokens_input > 0 {
        out.tokens_input = delta.tokens_input;
    }
    if delta.tokens_output > 0 {
        out.tokens_output = delta.tokens_output;
    }
    if delta.tokens_cache_read > 0 {
        out.tokens_cache_read = delta.tokens_cache_read;
    }
    if delta.tokens_cache_create > 0 {
        out.tokens_cache_create = delta.tokens_cache_create;
    }
    out
}

fn classify_sse_error(err: Option<&Value>) -> LlmError {
    let (err_type, err_msg) = err
        .map(|e| {
            (
                e.get("type").and_then(Value::as_str).unwrap_or("unknown"),
                e.get("message").and_then(Value::as_str).unwrap_or(""),
            )
        })
        .unwrap_or(("unknown", ""));
    match err_type {
        "authentication_error" | "permission_error" => LlmError::Authentication,
        "rate_limit_error" | "overloaded_error" => LlmError::RateLimited { retry_after: None },
        _ => LlmError::MalformedResponse(format!("anthropic sse error ({err_type}): {err_msg}")),
    }
}

// ---------- stream driver ----------

fn chunk_stream_from_response(resp: reqwest::Response) -> ChunkStream {
    let (mut tx, rx) = mpsc::channel::<Result<Chunk, LlmError>>(32);
    tokio::spawn(async move {
        let mut bytes_stream = Box::pin(resp.bytes_stream());
        let mut buffer: Vec<u8> = Vec::new();
        let mut decoder = ChunkDecoder::default();

        loop {
            // Drain all complete frames currently in the buffer.
            while let Some(frame) = extract_sse_frame(&mut buffer) {
                match decode_frame(&frame, &mut decoder) {
                    Ok(chunks) => {
                        for c in chunks {
                            if tx.send(Ok(c)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }

            match bytes_stream.next().await {
                Some(Ok(bytes)) => buffer.extend_from_slice(bytes.as_ref()),
                Some(Err(e)) => {
                    let _ = tx
                        .send(Err(LlmError::Network(std::io::Error::other(e.to_string()))))
                        .await;
                    return;
                }
                None => {
                    // Some servers omit the trailing blank line; best-effort flush.
                    if !buffer.is_empty() {
                        buffer.extend_from_slice(b"\n\n");
                        if let Some(frame) = extract_sse_frame(&mut buffer) {
                            match decode_frame(&frame, &mut decoder) {
                                Ok(chunks) => {
                                    for c in chunks {
                                        if tx.send(Ok(c)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                }
                            }
                        }
                    }
                    return;
                }
            }
        }
    });
    rx.boxed()
}

// ---------- prompt caching layer ----------

/// `PromptCaching::anthropic()` layer returned by the factory at
/// [`crate::caching::PromptCaching::anthropic`].
///
/// Anthropic's request body already reads [`CacheHints`] directly in
/// `wire_messages` and tags the targeted content block with
/// `cache_control`. This layer is therefore a runtime no-op — its job is
/// the *construction-time* capability check (plan §5.7): it refuses to
/// wrap an `Llm` whose `capabilities().prompt_caching` is `false`, which
/// catches misconfigurations early (e.g., a `ReplayLlm` built from a
/// trajectory whose meta event reported `prompt_caching: false`, or any
/// non-Anthropic provider paired with this layer by mistake).
pub struct AnthropicPromptCaching;

impl<L: Llm> LlmLayer<L> for AnthropicPromptCaching {
    type Output = L;

    fn wrap(self, inner: L) -> Result<L, LayerError> {
        if !inner.capabilities().prompt_caching {
            return Err(LayerError::MissingCapability {
                layer: "PromptCaching::anthropic",
                capability: "prompt_caching",
            });
        }
        Ok(inner)
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_frame_single() {
        let mut buf = b"event: ping\ndata: {\"type\":\"ping\"}\n\n".to_vec();
        let frame = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(frame.event.as_deref(), Some("ping"));
        assert_eq!(frame.data, "{\"type\":\"ping\"}");
        assert!(buf.is_empty());
    }

    #[test]
    fn extract_frame_crlf() {
        let mut buf = b"event: ping\r\ndata: {}\r\n\r\n".to_vec();
        let frame = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(frame.event.as_deref(), Some("ping"));
        assert_eq!(frame.data, "{}");
        assert!(buf.is_empty());
    }

    #[test]
    fn extract_frame_multiple() {
        let mut buf = b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n".to_vec();
        let f1 = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(f1.event.as_deref(), Some("a"));
        assert_eq!(f1.data, "1");
        let f2 = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(f2.event.as_deref(), Some("b"));
        assert_eq!(f2.data, "2");
        assert!(extract_sse_frame(&mut buf).is_none());
    }

    #[test]
    fn extract_frame_partial_waits() {
        let mut buf = b"event: a\ndata: 1".to_vec();
        assert!(extract_sse_frame(&mut buf).is_none());
        buf.extend_from_slice(b"\n\n");
        let f = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(f.data, "1");
    }

    #[test]
    fn extract_frame_multiline_data_joined() {
        let mut buf = b"data: line1\ndata: line2\n\n".to_vec();
        let f = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(f.data, "line1\nline2");
    }

    #[test]
    fn extract_frame_ignores_comment() {
        let mut buf = b": heartbeat\nevent: ping\ndata: {}\n\n".to_vec();
        let f = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(f.event.as_deref(), Some("ping"));
    }

    fn frame(event: &str, data: &str) -> SseFrame {
        SseFrame {
            event: Some(event.to_string()),
            data: data.to_string(),
        }
    }

    #[test]
    fn decode_message_start_emits_start_and_usage() {
        let f = frame(
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5","usage":{"input_tokens":7,"output_tokens":1}}}"#,
        );
        let mut state = ChunkDecoder::default();
        let chunks = decode_frame(&f, &mut state).unwrap();
        assert_eq!(chunks.len(), 2);
        match &chunks[0] {
            Chunk::MessageStart { id, model } => {
                assert_eq!(id, "msg_1");
                assert_eq!(model.as_str(), "claude-sonnet-4-5");
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }
        match &chunks[1] {
            Chunk::Usage { usage } => {
                assert_eq!(usage.tokens_input, 7);
                assert_eq!(usage.tokens_output, 1);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        assert!(state.initial_usage.is_some());
    }

    #[test]
    fn decode_block_start_variants() {
        let mut state = ChunkDecoder::default();
        let text = decode_frame(
            &frame(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            text[0],
            Chunk::BlockStart {
                index: 0,
                start: BlockStartKind::Text
            }
        ));

        let tool = decode_frame(
            &frame(
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"fs_list","input":{}}}"#,
            ),
            &mut state,
        )
        .unwrap();
        match &tool[0] {
            Chunk::BlockStart {
                index: 1,
                start: BlockStartKind::ToolUse { name, id },
            } => {
                assert_eq!(name, "fs_list");
                assert_eq!(id, "t1");
            }
            other => panic!("expected ToolUse BlockStart, got {other:?}"),
        }

        let thinking = decode_frame(
            &frame(
                "content_block_start",
                r#"{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":""}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            thinking[0],
            Chunk::BlockStart {
                index: 2,
                start: BlockStartKind::Thinking
            }
        ));
    }

    #[test]
    fn decode_block_start_unknown_passes_through_raw() {
        let mut state = ChunkDecoder::default();
        let out = decode_frame(
            &frame(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"image","source":{}}}"#,
            ),
            &mut state,
        )
        .unwrap();
        match &out[0] {
            Chunk::Raw { provider, event } => {
                assert_eq!(provider, "anthropic");
                assert_eq!(event["type"], "content_block_start");
            }
            other => panic!("expected Raw, got {other:?}"),
        }
    }

    #[test]
    fn decode_content_deltas() {
        let mut state = ChunkDecoder::default();
        let txt = decode_frame(
            &frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &txt[0],
            Chunk::TextDelta { index: 0, text } if text == "Hi"
        ));

        let ijson = decode_frame(
            &frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &ijson[0],
            Chunk::ToolUseDelta { index: 1, partial_json } if partial_json == "{\"a\":"
        ));

        let think = decode_frame(
            &frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &think[0],
            Chunk::ThinkingDelta { index: 2, text } if text == "hmm"
        ));
    }

    #[test]
    fn decode_signature_delta_passes_through_raw() {
        let mut state = ChunkDecoder::default();
        let out = decode_frame(
            &frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert!(matches!(&out[0], Chunk::Raw { .. }));
    }

    #[test]
    fn decode_message_delta_merges_usage() {
        let mut state = ChunkDecoder::default();
        // Prime initial_usage from a message_start.
        decode_frame(
            &frame(
                "message_start",
                r#"{"type":"message_start","message":{"id":"m","model":"claude-sonnet-4-5","usage":{"input_tokens":12,"output_tokens":1}}}"#,
            ),
            &mut state,
        )
        .unwrap();

        let out = decode_frame(
            &frame(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":42}}"#,
            ),
            &mut state,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(
            &out[0],
            Chunk::StopReason {
                reason: StopReason::EndTurn
            }
        ));
        match &out[1] {
            Chunk::Usage { usage } => {
                assert_eq!(usage.tokens_input, 12);
                assert_eq!(usage.tokens_output, 42);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn decode_message_stop_and_ping() {
        let mut state = ChunkDecoder::default();
        let stop = decode_frame(
            &frame("message_stop", r#"{"type":"message_stop"}"#),
            &mut state,
        )
        .unwrap();
        assert!(matches!(stop.as_slice(), [Chunk::MessageStop]));

        let ping = decode_frame(&frame("ping", r#"{"type":"ping"}"#), &mut state).unwrap();
        assert!(ping.is_empty());
    }

    #[test]
    fn decode_error_event_returns_err() {
        let mut state = ChunkDecoder::default();
        let err = decode_frame(
            &frame(
                "error",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
            ),
            &mut state,
        )
        .unwrap_err();
        assert!(matches!(err, LlmError::RateLimited { .. }));

        let auth = decode_frame(
            &frame(
                "error",
                r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#,
            ),
            &mut state,
        )
        .unwrap_err();
        assert!(matches!(auth, LlmError::Authentication));
    }

    #[test]
    fn decode_unknown_event_type_is_raw() {
        let mut state = ChunkDecoder::default();
        let out = decode_frame(
            &frame("something_new", r#"{"type":"something_new","v":1}"#),
            &mut state,
        )
        .unwrap();
        assert!(matches!(&out[0], Chunk::Raw { .. }));
    }

    // ---------- prompt caching ----------

    use oharness_core::{
        CacheBreakpoint, CacheHints as CacheHintsInner, CacheTtl as CacheTtlInner,
    };

    fn user_msg(text: &str) -> oharness_core::Message {
        oharness_core::Message::user_text(text)
    }

    #[test]
    fn wire_messages_applies_short_cache_control() {
        let msgs = vec![user_msg("one"), user_msg("two")];
        let hints = CacheHintsInner {
            breakpoints: vec![CacheBreakpoint {
                message_index: 1,
                ttl: Some(CacheTtlInner::Short),
            }],
        };
        let body = wire_messages(&msgs, &hints);
        // message 0 should have no cache_control
        assert!(body[0]["content"][0].get("cache_control").is_none());
        // message 1's last (only) block has cache_control with ttl 5m
        assert_eq!(
            body[1]["content"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "5m"})
        );
    }

    #[test]
    fn wire_messages_applies_long_cache_control() {
        let msgs = vec![user_msg("x")];
        let hints = CacheHintsInner {
            breakpoints: vec![CacheBreakpoint {
                message_index: 0,
                ttl: Some(CacheTtlInner::Long),
            }],
        };
        let body = wire_messages(&msgs, &hints);
        assert_eq!(body[0]["content"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn wire_messages_default_ttl_is_5m() {
        let msgs = vec![user_msg("x")];
        let hints = CacheHintsInner {
            breakpoints: vec![CacheBreakpoint {
                message_index: 0,
                ttl: None,
            }],
        };
        let body = wire_messages(&msgs, &hints);
        assert_eq!(body[0]["content"][0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn wire_messages_noop_when_no_hints() {
        let msgs = vec![user_msg("x"), user_msg("y")];
        let body = wire_messages(&msgs, &CacheHintsInner::default());
        assert!(body[0]["content"][0].get("cache_control").is_none());
        assert!(body[1]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn wire_messages_tags_last_block_of_multi_block_message() {
        use oharness_core::Content;
        let msgs = vec![oharness_core::Message::User {
            content: vec![Content::text("first"), Content::text("second")],
            meta: Default::default(),
        }];
        let hints = CacheHintsInner {
            breakpoints: vec![CacheBreakpoint {
                message_index: 0,
                ttl: Some(CacheTtlInner::Short),
            }],
        };
        let body = wire_messages(&msgs, &hints);
        // First block: no marker. Second (last): carries cache_control.
        assert!(body[0]["content"][0].get("cache_control").is_none());
        assert_eq!(
            body[0]["content"][1]["cache_control"],
            json!({"type": "ephemeral", "ttl": "5m"})
        );
    }

    #[test]
    fn anthropic_capabilities_advertise_prompt_caching() {
        let llm = AnthropicLlm::new("test-key", "claude-sonnet-4-5");
        assert!(llm.capabilities().prompt_caching);
    }

    #[test]
    fn prompt_caching_layer_wraps_cache_capable_llm() {
        use oharness_llm::LlmExt;
        let llm = AnthropicLlm::new("test-key", "claude-sonnet-4-5");
        let _wrapped = llm
            .try_with_layer(AnthropicPromptCaching)
            .expect("anthropic supports caching");
    }

    #[test]
    fn prompt_caching_layer_rejects_non_cache_capable_llm() {
        // Stub whose capabilities report prompt_caching == false.
        struct NonCachingStub;
        #[async_trait]
        impl Llm for NonCachingStub {
            fn name(&self) -> &str {
                "stub"
            }
            fn capabilities(&self) -> LlmCapabilities {
                LlmCapabilities::default()
            }
            async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, LlmError> {
                unreachable!()
            }
            async fn stream(&self, _: CompletionRequest) -> Result<ChunkStream, LlmError> {
                unreachable!()
            }
        }
        use oharness_llm::LlmExt;
        match NonCachingStub.try_with_layer(AnthropicPromptCaching) {
            Ok(_) => panic!("layer should have rejected stub without caching capability"),
            Err(LayerError::MissingCapability { layer, capability }) => {
                assert_eq!(layer, "PromptCaching::anthropic");
                assert_eq!(capability, "prompt_caching");
            }
            Err(other) => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn prompt_caching_factory_returns_anthropic_layer() {
        use oharness_llm::LlmExt;
        let llm = AnthropicLlm::new("test-key", "claude-sonnet-4-5");
        // `PromptCaching::anthropic()` is the stable public entry point.
        let _wrapped = llm
            .try_with_layer(crate::caching::PromptCaching::anthropic())
            .expect("anthropic supports caching");
    }
}
