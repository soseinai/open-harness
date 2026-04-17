//! OpenAI Chat Completions adapter (plan §6).
//!
//! Targets `POST /v1/chat/completions`. Chat Completions is still the most
//! widely-supported OpenAI API shape and is the basis for the OpenRouter /
//! vLLM / Ollama OpenAI-compatible endpoints we'll add next. The newer
//! Responses API will land as a separate adapter (`OpenAiResponsesLlm`)
//! when we need its reasoning-block story.
//!
//! Translation notes (canonical `Message` shape → OpenAI wire format):
//!
//! - `Content::ToolUse` blocks in assistant messages collapse onto OpenAI's
//!   `tool_calls` array. Arguments are serialized as a **JSON string** (per
//!   OpenAI's schema), not an object.
//! - `Content::ToolResult` blocks on user messages expand to separate
//!   `role: "tool"` messages carrying `tool_call_id`. A single canonical
//!   user message with N tool results produces N wire messages.
//! - `Content::Thinking` is dropped — OpenAI's o-series models handle
//!   reasoning internally and do not surface it on the Chat Completions
//!   API surface.
//! - Streaming deltas reserve block index `0` for text output and allocate
//!   `1..` for tool calls in registration order; OpenAI's `choices[].delta
//!   .tool_calls[i].index` is a per-message counter, not our block index,
//!   so we keep a translation map.

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, StreamExt};
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    ToolSpec, Usage,
};
use oharness_llm::{BlockStartKind, Chunk, ChunkStream, Llm, LlmError};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

const OPENAI_API: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_MODEL: &str = "gpt-4o";

pub struct OpenAiLlm {
    http: reqwest::Client,
    api_key: String,
    model: ModelId,
    base_url: String,
    timeout: Duration,
    capabilities: LlmCapabilities,
}

impl OpenAiLlm {
    /// Construct from the `OPENAI_API_KEY` environment variable, defaulting to
    /// `gpt-4o`. Fails if the env var is missing.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| LlmError::Authentication)?;
        Ok(Self::new(api_key, DEFAULT_MODEL))
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
            base_url: OPENAI_API.to_string(),
            timeout: Duration::from_secs(120),
            capabilities: LlmCapabilities {
                streaming: true,
                // OpenAI doesn't expose the Anthropic-style `cache_control`
                // request extension on Chat Completions. Automatic prompt
                // caching (prefix-hit discount) exists server-side but has
                // no request-shape opt-in.
                prompt_caching: false,
                parallel_tool_use: true,
                vision: true,
                thinking: false,
                structured_output: true,
                max_context_tokens: 128_000,
                max_output_tokens: 4096,
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
impl Llm for OpenAiLlm {
    fn name(&self) -> &str {
        "openai"
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = to_wire_request(&self.model, &req, false);
        let resp = self
            .http
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
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
            .map_err(|e| LlmError::MalformedResponse(format!("openai decode: {e}: {text}")))?;
        from_wire_response(parsed)
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let body = to_wire_request(&self.model, &req, true);
        let resp = self
            .http
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
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

fn to_wire_request(model: &ModelId, req: &CompletionRequest, streaming: bool) -> Value {
    let mut body = json!({
        "model": model.as_str(),
        "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": wire_messages(&req.messages, req.system.as_deref()),
    });
    if streaming {
        body["stream"] = Value::Bool(true);
        // Ask the final chunk to carry usage so BudgetMiddleware can count
        // tokens on the streaming path.
        body["stream_options"] = json!({ "include_usage": true });
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = json!(temp);
    }
    if !req.stop_sequences.is_empty() {
        body["stop"] = json!(req.stop_sequences);
    }
    if !req.tools.is_empty() {
        body["tools"] = wire_tools(&req.tools);
    }
    body
}

fn wire_messages(messages: &[Message], system_override: Option<&str>) -> Value {
    let mut out: Vec<Value> = Vec::new();
    if let Some(sys) = system_override {
        out.push(json!({"role": "system", "content": sys}));
    }

    for m in messages {
        match m {
            Message::System { content, .. } => {
                out.push(json!({"role": "system", "content": content}));
            }
            Message::User { content, .. } => {
                // Split off ToolResult blocks into their own `role: "tool"`
                // messages; everything else coalesces into one user message.
                let mut non_tool_results: Vec<&Content> = Vec::new();
                for c in content {
                    if let Content::ToolResult {
                        tool_use_id,
                        output,
                        is_error,
                    } = c
                    {
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tool_result_text(output, *is_error),
                        }));
                    } else {
                        non_tool_results.push(c);
                    }
                }
                if !non_tool_results.is_empty() {
                    out.push(json!({
                        "role": "user",
                        "content": user_content_field(&non_tool_results),
                    }));
                }
            }
            Message::Assistant { content, .. } => {
                out.push(assistant_wire_message(content));
            }
        }
    }

    Value::Array(out)
}

fn tool_result_text(output: &oharness_core::ToolOutput, is_error: bool) -> String {
    let body = output
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if is_error {
        format!("ERROR: {body}")
    } else {
        body
    }
}

fn user_content_field(parts: &[&Content]) -> Value {
    // All-text fast path: collapse into a plain string (matches most live
    // traffic and is what OpenAI's UI exposes).
    if parts.iter().all(|c| matches!(c, Content::Text { .. })) {
        let joined = parts
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Value::String(joined);
    }
    // Otherwise emit a content-parts array. Vision / unsupported types fall
    // through to text stubs for now — richer vision support lands with the
    // image type work.
    let parts: Vec<Value> = parts
        .iter()
        .map(|c| match c {
            Content::Text { text } => json!({"type": "text", "text": text}),
            Content::Image(_) => json!({"type": "text", "text": "[image content — TODO vision]"}),
            Content::Document(_) | Content::Audio(_) | Content::Citation(_) => {
                json!({"type": "text", "text": "[unsupported content block]"})
            }
            Content::Thinking { .. } | Content::ToolUse { .. } | Content::ToolResult { .. } => {
                // Shouldn't reach here — filtered upstream.
                json!({"type": "text", "text": "[internal: unexpected block]"})
            }
        })
        .collect();
    Value::Array(parts)
}

fn assistant_wire_message(content: &[Content]) -> Value {
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for c in content {
        match c {
            Content::Text { text: t } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            Content::ToolUse { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        // OpenAI requires arguments as a JSON-encoded string.
                        "arguments": input.to_string(),
                    }
                }));
            }
            Content::Thinking { .. } => {
                // Not carried on Chat Completions.
            }
            _ => {
                // Images / documents / audio / citations in assistant
                // messages are rare; drop for M1b adapter rather than
                // insert placeholder text that could confuse the model.
            }
        }
    }

    let mut msg = json!({ "role": "assistant" });
    if !text.is_empty() {
        msg["content"] = Value::String(text);
    } else if tool_calls.is_empty() {
        msg["content"] = Value::String(String::new());
    } else {
        // OpenAI accepts `content: null` when tool_calls is present.
        msg["content"] = Value::Null;
    }
    if !tool_calls.is_empty() {
        msg["tool_calls"] = Value::Array(tool_calls);
    }
    msg
}

fn wire_tools(tools: &[ToolSpec]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
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
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireAssistantMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireAssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunctionCall,
}

#[derive(Debug, Deserialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

fn from_wire_response(w: WireResponse) -> Result<CompletionResponse, LlmError> {
    let Some(choice) = w.choices.into_iter().next() else {
        return Err(LlmError::MalformedResponse(
            "openai: response had no choices".to_string(),
        ));
    };

    let mut content: Vec<Content> = Vec::new();
    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            content.push(Content::text(text));
        }
    }
    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::String(tc.function.arguments));
            content.push(Content::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    let stop_reason = map_finish_reason(choice.finish_reason.as_deref());
    let usage = w.usage.unwrap_or_default();

    Ok(CompletionResponse {
        id: w.id,
        model: ModelId::new(w.model),
        content,
        stop_reason,
        usage: Usage {
            tokens_input: usage.prompt_tokens,
            tokens_output: usage.completion_tokens,
            ..Default::default()
        },
    })
}

fn map_finish_reason(raw: Option<&str>) -> StopReason {
    match raw {
        Some("stop") | None => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::Refusal,
        Some(other) => StopReason::Error(other.to_string()),
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
            let lower = body.to_lowercase();
            if lower.contains("context") && lower.contains("length") {
                LlmError::ContextTooLong { max: 0, got: 0 }
            } else {
                LlmError::MalformedResponse(format!("{status}: {body}"))
            }
        }
        _ => LlmError::MalformedResponse(format!("HTTP {status}: {body}")),
    }
}

// ---------- SSE framing (shared pattern with the Anthropic adapter) ----------

#[derive(Debug)]
struct SseFrame {
    data: String,
}

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
    let mut data = String::new();
    let mut data_seen = false;
    for raw in text.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(value) = line
            .strip_prefix("data:")
            .map(|s| s.strip_prefix(' ').unwrap_or(s))
        {
            if data_seen {
                data.push('\n');
            }
            data.push_str(value);
            data_seen = true;
        }
        // `event:` / `id:` / `retry:` are ignored — Chat Completions only
        // uses `data:`.
    }
    if !data_seen {
        return None;
    }
    Some(SseFrame { data })
}

fn extract_sse_frame(buf: &mut Vec<u8>) -> Option<SseFrame> {
    let (body_len, total) = find_frame_end(buf)?;
    let frame = parse_frame_body(&buf[..body_len]);
    buf.drain(..total);
    frame
}

// ---------- SSE event → Chunk decoder ----------

#[derive(Default)]
struct StreamState {
    message_started: bool,
    /// Has `BlockStart { index: 0, Text }` been emitted? We wait for the
    /// first content delta rather than emit proactively — responses that
    /// are pure tool_calls shouldn't carry an empty text block.
    text_block_emitted: bool,
    /// OpenAI `tool_calls[i].index` → our `Chunk::BlockStart.index`.
    tool_blocks: HashMap<u32, u32>,
    /// Next block index for a freshly-seen tool call. Reserves `0` for
    /// text, so this starts at `1`.
    next_tool_block: u32,
    /// Have we emitted StopReason for this stream? Guards against OpenAI
    /// sending a `finish_reason` in one chunk and `usage` in the next.
    stop_emitted: bool,
}

impl StreamState {
    fn new() -> Self {
        Self {
            next_tool_block: 1,
            ..Default::default()
        }
    }
}

fn decode_frame(frame: &SseFrame, state: &mut StreamState) -> Result<Vec<Chunk>, LlmError> {
    if frame.data.trim() == "[DONE]" {
        return Ok(Vec::new());
    }
    let payload: Value = serde_json::from_str(&frame.data).map_err(|e| {
        LlmError::MalformedResponse(format!("openai sse decode: {e}: {}", frame.data))
    })?;

    let mut out = Vec::new();

    // MessageStart: first chunk we see.
    if !state.message_started {
        let id = payload
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.push(Chunk::MessageStart {
            id,
            model: ModelId::new(model),
        });
        state.message_started = true;
    }

    // Process choice[0] delta and finish_reason.
    if let Some(choice) = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
    {
        if let Some(delta) = choice.get("delta") {
            // Text content delta.
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    if !state.text_block_emitted {
                        out.push(Chunk::BlockStart {
                            index: 0,
                            start: BlockStartKind::Text,
                        });
                        state.text_block_emitted = true;
                    }
                    out.push(Chunk::TextDelta {
                        index: 0,
                        text: text.to_string(),
                    });
                }
            }
            // Tool-call deltas.
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in tool_calls {
                    let oai_idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                    let our_idx = match state.tool_blocks.get(&oai_idx) {
                        Some(i) => *i,
                        None => {
                            // First time seeing this tool call — must have
                            // id + function.name to open a BlockStart.
                            let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            let our_idx = state.next_tool_block;
                            state.next_tool_block += 1;
                            state.tool_blocks.insert(oai_idx, our_idx);
                            out.push(Chunk::BlockStart {
                                index: our_idx,
                                start: BlockStartKind::ToolUse {
                                    name: name.to_string(),
                                    id: id.to_string(),
                                },
                            });
                            our_idx
                        }
                    };
                    if let Some(args) = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        if !args.is_empty() {
                            out.push(Chunk::ToolUseDelta {
                                index: our_idx,
                                partial_json: args.to_string(),
                            });
                        }
                    }
                }
            }
        }

        if let Some(raw) = choice.get("finish_reason").and_then(Value::as_str) {
            if !state.stop_emitted {
                // Close any open blocks so complete_from_stream can assemble
                // content in block-index order.
                if state.text_block_emitted {
                    out.push(Chunk::BlockStop { index: 0 });
                }
                let mut tool_indices: Vec<u32> = state.tool_blocks.values().copied().collect();
                tool_indices.sort_unstable();
                for idx in tool_indices {
                    out.push(Chunk::BlockStop { index: idx });
                }
                out.push(Chunk::StopReason {
                    reason: map_finish_reason(Some(raw)),
                });
                state.stop_emitted = true;
            }
        }
    }

    // Usage (arrives on the final chunk when `stream_options.include_usage`
    // is set).
    if let Some(u) = payload.get("usage") {
        let usage = Usage {
            tokens_input: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            tokens_output: u
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            ..Default::default()
        };
        out.push(Chunk::Usage { usage });
    }

    Ok(out)
}

// ---------- stream driver ----------

fn chunk_stream_from_response(resp: reqwest::Response) -> ChunkStream {
    let (mut tx, rx) = mpsc::channel::<Result<Chunk, LlmError>>(32);
    tokio::spawn(async move {
        let mut bytes_stream = Box::pin(resp.bytes_stream());
        let mut buffer: Vec<u8> = Vec::new();
        let mut state = StreamState::new();

        loop {
            loop {
                let Some(frame) = extract_sse_frame(&mut buffer) else {
                    break;
                };
                match decode_frame(&frame, &mut state) {
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
                    if !buffer.is_empty() {
                        buffer.extend_from_slice(b"\n\n");
                        if let Some(frame) = extract_sse_frame(&mut buffer) {
                            if let Ok(chunks) = decode_frame(&frame, &mut state) {
                                for c in chunks {
                                    if tx.send(Ok(c)).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    // Emit MessageStop once the upstream closes. OpenAI's
                    // Chat Completions stream has no dedicated
                    // `message_stop`-like sentinel beyond `[DONE]`; readers
                    // of our Chunk stream expect a terminal MessageStop so
                    // `complete_from_stream` knows when to break.
                    let _ = tx.send(Ok(Chunk::MessageStop)).await;
                    return;
                }
            }
        }
    });
    rx.boxed()
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::message::ToolOutput;

    // ---------- wire_messages ----------

    #[test]
    fn wire_messages_system_override_prepends_system() {
        let msgs = vec![Message::user_text("hi")];
        let out = wire_messages(&msgs, Some("be helpful"));
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], "be helpful");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn wire_messages_plain_user_text_uses_string_content() {
        let msgs = vec![Message::user_text("hello world")];
        let out = wire_messages(&msgs, None);
        assert_eq!(out[0]["content"], "hello world");
    }

    #[test]
    fn wire_messages_tool_result_becomes_tool_role_message() {
        let msgs = vec![Message::User {
            content: vec![Content::ToolResult {
                tool_use_id: "call_1".into(),
                output: ToolOutput::text("result body"),
                is_error: false,
            }],
            meta: Default::default(),
        }];
        let out = wire_messages(&msgs, None);
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["tool_call_id"], "call_1");
        assert_eq!(out[0]["content"], "result body");
    }

    #[test]
    fn wire_messages_tool_result_error_is_prefixed() {
        let msgs = vec![Message::User {
            content: vec![Content::ToolResult {
                tool_use_id: "call_1".into(),
                output: ToolOutput::text("boom"),
                is_error: true,
            }],
            meta: Default::default(),
        }];
        let out = wire_messages(&msgs, None);
        assert_eq!(out[0]["content"], "ERROR: boom");
    }

    #[test]
    fn wire_messages_assistant_with_text_and_tool_calls() {
        let msgs = vec![Message::Assistant {
            content: vec![
                Content::text("let me check"),
                Content::ToolUse {
                    id: "call_1".into(),
                    name: "fs_list".into(),
                    input: json!({"path": "."}),
                },
            ],
            stop_reason: Some(StopReason::ToolUse),
            meta: Default::default(),
        }];
        let out = wire_messages(&msgs, None);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[0]["content"], "let me check");
        let calls = &out[0]["tool_calls"];
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "fs_list");
        // arguments must be a JSON *string*, not a nested object
        let args = calls[0]["function"]["arguments"].as_str().unwrap();
        let decoded: Value = serde_json::from_str(args).unwrap();
        assert_eq!(decoded, json!({"path": "."}));
    }

    #[test]
    fn wire_messages_assistant_with_only_tool_calls_has_null_content() {
        let msgs = vec![Message::Assistant {
            content: vec![Content::ToolUse {
                id: "call_1".into(),
                name: "foo".into(),
                input: json!({}),
            }],
            stop_reason: Some(StopReason::ToolUse),
            meta: Default::default(),
        }];
        let out = wire_messages(&msgs, None);
        assert!(out[0]["content"].is_null());
        assert!(!out[0]["tool_calls"][0]["id"].is_null());
    }

    // ---------- from_wire_response ----------

    #[test]
    fn from_wire_response_maps_text_and_tool_calls() {
        let wire: WireResponse = serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "fs_list", "arguments": "{\"path\":\".\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        }))
        .unwrap();
        let res = from_wire_response(wire).unwrap();
        assert_eq!(res.id, "chatcmpl-1");
        assert_eq!(res.content.len(), 2);
        assert!(matches!(&res.content[0], Content::Text { text } if text == "thinking..."));
        assert!(matches!(
            &res.content[1],
            Content::ToolUse { id, name, .. } if id == "call_1" && name == "fs_list"
        ));
        assert!(matches!(res.stop_reason, StopReason::ToolUse));
        assert_eq!(res.usage.tokens_input, 5);
        assert_eq!(res.usage.tokens_output, 3);
    }

    #[test]
    fn from_wire_response_errors_on_empty_choices() {
        let wire: WireResponse = serde_json::from_value(json!({
            "id": "chatcmpl-empty",
            "model": "gpt-4o",
            "choices": []
        }))
        .unwrap();
        match from_wire_response(wire) {
            Err(LlmError::MalformedResponse(msg)) => assert!(msg.contains("no choices")),
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn map_finish_reason_covers_known_variants() {
        assert!(matches!(
            map_finish_reason(Some("stop")),
            StopReason::EndTurn
        ));
        assert!(matches!(
            map_finish_reason(Some("length")),
            StopReason::MaxTokens
        ));
        assert!(matches!(
            map_finish_reason(Some("tool_calls")),
            StopReason::ToolUse
        ));
        assert!(matches!(
            map_finish_reason(Some("content_filter")),
            StopReason::Refusal
        ));
        if let StopReason::Error(s) = map_finish_reason(Some("weird")) {
            assert_eq!(s, "weird");
        } else {
            panic!("expected Error");
        }
        assert!(matches!(map_finish_reason(None), StopReason::EndTurn));
    }

    // ---------- SSE framing ----------

    #[test]
    fn extract_sse_frame_single_data_line() {
        let mut buf = b"data: {\"a\":1}\n\n".to_vec();
        let frame = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(frame.data, "{\"a\":1}");
    }

    #[test]
    fn extract_sse_frame_handles_done_sentinel() {
        let mut buf = b"data: [DONE]\n\n".to_vec();
        let frame = extract_sse_frame(&mut buf).unwrap();
        assert_eq!(frame.data, "[DONE]");
    }

    // ---------- stream decoder ----------

    fn frame(data: &str) -> SseFrame {
        SseFrame {
            data: data.to_string(),
        }
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

    #[test]
    fn decode_emits_message_start_once() {
        let mut state = StreamState::new();
        let first = decode_frame(
            &frame(r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#),
            &mut state,
        )
        .unwrap();
        assert!(matches!(first[0], Chunk::MessageStart { .. }));

        let second = decode_frame(
            &frame(r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#),
            &mut state,
        )
        .unwrap();
        assert!(!second
            .iter()
            .any(|c| matches!(c, Chunk::MessageStart { .. })));
    }

    #[test]
    fn decode_text_delta_emits_block_start_then_delta() {
        let mut state = StreamState::new();
        let chunks = decode_frame(
            &frame(r#"{"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}"#),
            &mut state,
        )
        .unwrap();
        let labels: Vec<_> = chunks.iter().map(chunk_label).collect();
        assert_eq!(labels, vec!["message_start", "block_start", "text_delta"]);
    }

    #[test]
    fn decode_tool_call_registers_block_on_first_delta() {
        let mut state = StreamState::new();
        let chunks = decode_frame(
            &frame(
                r#"{"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc_1","type":"function","function":{"name":"foo","arguments":"{\"a\":"}}]}}]}"#,
            ),
            &mut state,
        )
        .unwrap();
        let labels: Vec<_> = chunks.iter().map(chunk_label).collect();
        assert_eq!(
            labels,
            vec!["message_start", "block_start", "tool_use_delta"]
        );
        match &chunks[1] {
            Chunk::BlockStart {
                index,
                start: BlockStartKind::ToolUse { name, id },
            } => {
                assert_eq!(*index, 1); // text reserved at 0
                assert_eq!(name, "foo");
                assert_eq!(id, "tc_1");
            }
            other => panic!("expected tool ToolUse BlockStart, got {other:?}"),
        }
    }

    #[test]
    fn decode_tool_call_continuation_reuses_block_index() {
        let mut state = StreamState::new();
        // Open the tool-call block.
        decode_frame(
            &frame(
                r#"{"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc_1","type":"function","function":{"name":"foo","arguments":"{"}}]}}]}"#,
            ),
            &mut state,
        )
        .unwrap();
        // Continuation with only arguments.
        let chunks = decode_frame(
            &frame(
                r#"{"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a\":1}"}}]}}]}"#,
            ),
            &mut state,
        )
        .unwrap();
        let labels: Vec<_> = chunks.iter().map(chunk_label).collect();
        // No BlockStart — we've already seen this tool call.
        assert_eq!(labels, vec!["tool_use_delta"]);
        if let Chunk::ToolUseDelta { index, .. } = &chunks[0] {
            assert_eq!(*index, 1);
        } else {
            panic!("expected ToolUseDelta");
        }
    }

    #[test]
    fn decode_finish_reason_emits_block_stops_and_stop_reason() {
        let mut state = StreamState::new();
        // Open text + tool blocks.
        decode_frame(
            &frame(r#"{"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}"#),
            &mut state,
        )
        .unwrap();
        decode_frame(
            &frame(
                r#"{"id":"c1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#,
            ),
            &mut state,
        )
        .unwrap();
        // Finish.
        let chunks = decode_frame(
            &frame(
                r#"{"id":"c1","model":"m","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            ),
            &mut state,
        )
        .unwrap();
        let labels: Vec<_> = chunks.iter().map(chunk_label).collect();
        assert_eq!(labels, vec!["block_stop", "block_stop", "stop_reason"]);
    }

    #[test]
    fn decode_usage_chunk_forwards() {
        let mut state = StreamState::new();
        // Typical final chunk shape when include_usage is set.
        let chunks = decode_frame(
            &frame(r#"{"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#),
            &mut state,
        )
        .unwrap();
        let usage = chunks
            .iter()
            .find_map(|c| {
                if let Chunk::Usage { usage } = c {
                    Some(usage)
                } else {
                    None
                }
            })
            .expect("usage chunk");
        assert_eq!(usage.tokens_input, 10);
        assert_eq!(usage.tokens_output, 5);
    }

    #[test]
    fn decode_done_sentinel_is_noop() {
        let mut state = StreamState::new();
        let chunks = decode_frame(&frame("[DONE]"), &mut state).unwrap();
        assert!(chunks.is_empty());
    }
}
