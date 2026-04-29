//! OpenAI Codex provider backed by ChatGPT OAuth credentials.
//!
//! This is intentionally separate from the normal OpenAI Platform adapter:
//! it talks to ChatGPT's Codex Responses endpoint and expects a short-lived
//! OAuth access token plus the ChatGPT account id embedded in that token.

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, StreamExt};
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    ToolOutput, ToolSpec, Usage,
};
use oharness_llm::{complete_from_stream, BlockStartKind, Chunk, ChunkStream, Llm, LlmError};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::str;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const SCOPE: &str = "openid profile email offline_access";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const CALLBACK_PORT: u16 = 1455;
const DEFAULT_CALLBACK_HOST: &str = "127.0.0.1";
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_ORIGINATOR: &str = "ought";

/// Request auth for the ChatGPT Codex backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCodexAuth {
    pub access_token: String,
    pub account_id: String,
}

/// Persistable OAuth credentials for ChatGPT/Codex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCodexCredentials {
    pub access: String,
    pub refresh: String,
    /// Unix epoch milliseconds at which the access token expires.
    pub expires: u64,
    #[serde(alias = "accountId")]
    pub account_id: String,
}

impl OpenAiCodexCredentials {
    pub fn auth(&self) -> OpenAiCodexAuth {
        OpenAiCodexAuth {
            access_token: self.access.clone(),
            account_id: self.account_id.clone(),
        }
    }

    pub fn is_expired(&self) -> bool {
        now_ms().saturating_add(60_000) >= self.expires
    }
}

/// PKCE authorization state created before opening the browser.
#[derive(Debug, Clone)]
pub struct OpenAiCodexAuthorization {
    pub verifier: String,
    pub state: String,
    pub url: String,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiCodexOAuthError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("token exchange failed with HTTP {status}: {body}")]
    TokenExchange { status: StatusCode, body: String },
    #[error("token response missing {0}")]
    MissingTokenField(&'static str),
    #[error("authorization callback did not include a code")]
    MissingCode,
    #[error("authorization state mismatch")]
    StateMismatch,
    #[error("failed to decode ChatGPT account id from access token")]
    MissingAccountId,
}

/// OAuth helper for OpenAI Codex/ChatGPT credentials.
#[derive(Debug, Clone)]
pub struct OpenAiCodexOAuthClient {
    http: reqwest::Client,
    callback_host: String,
    originator: String,
}

impl Default for OpenAiCodexOAuthClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiCodexOAuthClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            callback_host: DEFAULT_CALLBACK_HOST.to_string(),
            originator: DEFAULT_ORIGINATOR.to_string(),
        }
    }

    pub fn with_callback_host(mut self, callback_host: impl Into<String>) -> Self {
        self.callback_host = callback_host.into();
        self
    }

    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    pub fn create_authorization(&self) -> OpenAiCodexAuthorization {
        let verifier = create_verifier();
        let challenge = pkce_challenge(&verifier);
        let state = Uuid::new_v4().simple().to_string();
        let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("valid OpenAI authorize URL");
        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", CLIENT_ID)
            .append_pair("redirect_uri", REDIRECT_URI)
            .append_pair("scope", SCOPE)
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &state)
            .append_pair("id_token_add_organizations", "true")
            .append_pair("codex_cli_simplified_flow", "true")
            .append_pair("originator", &self.originator);

        OpenAiCodexAuthorization {
            verifier,
            state,
            url: url.to_string(),
        }
    }

    pub fn start_callback_server(
        &self,
        state: impl Into<String>,
    ) -> std::io::Result<OpenAiCodexCallbackServer> {
        OpenAiCodexCallbackServer::start(self.callback_host.clone(), state.into())
    }

    pub async fn exchange_authorization_code(
        &self,
        code: &str,
        verifier: &str,
    ) -> Result<OpenAiCodexCredentials, OpenAiCodexOAuthError> {
        let response = self
            .http
            .post(TOKEN_URL)
            .form(&[
                ("grant_type", "authorization_code"),
                ("client_id", CLIENT_ID),
                ("code", code),
                ("code_verifier", verifier),
                ("redirect_uri", REDIRECT_URI),
            ])
            .send()
            .await?;

        token_response_to_credentials(response).await
    }

    pub async fn refresh_access_token(
        &self,
        refresh_token: &str,
    ) -> Result<OpenAiCodexCredentials, OpenAiCodexOAuthError> {
        let response = self
            .http
            .post(TOKEN_URL)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", CLIENT_ID),
            ])
            .send()
            .await?;

        token_response_to_credentials(response).await
    }

    pub fn parse_authorization_input(
        input: &str,
        expected_state: &str,
    ) -> Result<String, OpenAiCodexOAuthError> {
        let value = input.trim();
        if value.is_empty() {
            return Err(OpenAiCodexOAuthError::MissingCode);
        }

        if let Ok(url) = reqwest::Url::parse(value) {
            return code_from_params(url.query_pairs(), expected_state);
        }

        if let Some((code, state)) = value.split_once('#') {
            if state != expected_state {
                return Err(OpenAiCodexOAuthError::StateMismatch);
            }
            return Ok(code.to_string());
        }

        if value.contains("code=") {
            let url = reqwest::Url::parse(&format!("http://localhost/?{value}"))
                .map_err(|_| OpenAiCodexOAuthError::MissingCode)?;
            return code_from_params(url.query_pairs(), expected_state);
        }

        Ok(value.to_string())
    }
}

/// One-shot localhost OAuth callback server.
pub struct OpenAiCodexCallbackServer {
    rx: std_mpsc::Receiver<Result<String, OpenAiCodexOAuthError>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl OpenAiCodexCallbackServer {
    fn start(host: String, state: String) -> std::io::Result<Self> {
        let listener = TcpListener::bind((host.as_str(), CALLBACK_PORT))?;
        let (tx, rx) = std_mpsc::channel();
        let handle = thread::spawn(move || {
            let result = handle_one_callback(listener, &state);
            let _ = tx.send(result);
        });
        Ok(Self {
            rx,
            handle: Some(handle),
        })
    }

    pub fn wait_for_code(mut self) -> Result<String, OpenAiCodexOAuthError> {
        let result = self
            .rx
            .recv()
            .unwrap_or(Err(OpenAiCodexOAuthError::MissingCode));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        result
    }
}

/// ChatGPT Codex Responses API adapter.
pub struct OpenAiCodexLlm {
    http: reqwest::Client,
    auth: OpenAiCodexAuth,
    model: ModelId,
    base_url: String,
    timeout: Duration,
    originator: String,
    capabilities: LlmCapabilities,
}

impl OpenAiCodexLlm {
    pub fn new(auth: OpenAiCodexAuth, model: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("oharness-providers/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client build");
        Self {
            http,
            auth,
            model: ModelId::new(model.into()),
            base_url: DEFAULT_CODEX_BASE_URL.to_string(),
            timeout: Duration::from_secs(120),
            originator: DEFAULT_ORIGINATOR.to_string(),
            capabilities: LlmCapabilities {
                streaming: true,
                prompt_caching: true,
                parallel_tool_use: true,
                vision: true,
                thinking: true,
                structured_output: true,
                max_context_tokens: 272_000,
                max_output_tokens: 128_000,
            },
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    fn codex_url(&self) -> String {
        let normalized = self.base_url.trim_end_matches('/');
        if normalized.ends_with("/codex/responses") {
            normalized.to_string()
        } else if normalized.ends_with("/codex") {
            format!("{normalized}/responses")
        } else {
            format!("{normalized}/codex/responses")
        }
    }

    fn build_request(&self, body: &Value) -> reqwest::RequestBuilder {
        self.http
            .post(self.codex_url())
            .bearer_auth(&self.auth.access_token)
            .header("chatgpt-account-id", &self.auth.account_id)
            .header("originator", &self.originator)
            .header("OpenAI-Beta", "responses=experimental")
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .json(body)
    }
}

#[async_trait]
impl Llm for OpenAiCodexLlm {
    fn name(&self) -> &str {
        "openai-codex"
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        complete_from_stream(self.stream(req).await?).await
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let body = to_wire_request(&self.model, &req);
        let resp = self
            .build_request(&body)
            .send()
            .await
            .map_err(reqwest_to_llm_err)?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(reqwest_to_llm_err)?;
            return Err(classify_http_error(status, &text));
        }

        Ok(chunk_stream_from_response(resp, self.model.clone()))
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

async fn token_response_to_credentials(
    response: reqwest::Response,
) -> Result<OpenAiCodexCredentials, OpenAiCodexOAuthError> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(OpenAiCodexOAuthError::TokenExchange { status, body });
    }

    let token: TokenResponse = response.json().await?;
    let access = token
        .access_token
        .ok_or(OpenAiCodexOAuthError::MissingTokenField("access_token"))?;
    let refresh = token
        .refresh_token
        .ok_or(OpenAiCodexOAuthError::MissingTokenField("refresh_token"))?;
    let expires_in = token
        .expires_in
        .ok_or(OpenAiCodexOAuthError::MissingTokenField("expires_in"))?;
    let account_id =
        account_id_from_access_token(&access).ok_or(OpenAiCodexOAuthError::MissingAccountId)?;

    Ok(OpenAiCodexCredentials {
        access,
        refresh,
        expires: now_ms().saturating_add(expires_in.saturating_mul(1000)),
        account_id,
    })
}

fn handle_one_callback(
    listener: TcpListener,
    state: &str,
) -> Result<String, OpenAiCodexOAuthError> {
    let (mut stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = [0_u8; 8192];
    let n = stream.read(&mut buf)?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let request_line = request.lines().next().unwrap_or_default();
    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or(OpenAiCodexOAuthError::MissingCode)?;
    let url = reqwest::Url::parse(&format!("http://localhost{target}"))
        .map_err(|_| OpenAiCodexOAuthError::MissingCode)?;
    let result = code_from_params(url.query_pairs(), state);

    match &result {
        Ok(_) => write_callback_response(
            &mut stream,
            200,
            "OpenAI authentication completed. You can close this window.",
        )?,
        Err(OpenAiCodexOAuthError::StateMismatch) => {
            write_callback_response(&mut stream, 400, "State mismatch.")?;
        }
        Err(_) => {
            write_callback_response(&mut stream, 400, "Missing authorization code.")?;
        }
    }

    result
}

fn write_callback_response(
    stream: &mut std::net::TcpStream,
    status: u16,
    message: &str,
) -> std::io::Result<()> {
    let reason = if status == 200 { "OK" } else { "Bad Request" };
    let body = format!("<!doctype html><meta charset=\"utf-8\"><p>{message}</p>");
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn code_from_params<'a, I>(params: I, expected_state: &str) -> Result<String, OpenAiCodexOAuthError>
where
    I: IntoIterator<Item = (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>)>,
{
    let mut code = None;
    let mut state = None;
    for (key, value) in params {
        match key.as_ref() {
            "code" => code = Some(value.into_owned()),
            "state" => state = Some(value.into_owned()),
            _ => {}
        }
    }
    if state.as_deref() != Some(expected_state) {
        return Err(OpenAiCodexOAuthError::StateMismatch);
    }
    code.ok_or(OpenAiCodexOAuthError::MissingCode)
}

fn create_verifier() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

fn pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64_url_no_pad(&digest)
}

fn account_id_from_access_token(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64_url_decode(payload).ok()?;
    let json: Value = serde_json::from_slice(&bytes).ok()?;
    json.get(JWT_CLAIM_PATH)?
        .get("chatgpt_account_id")?
        .as_str()
        .map(str::to_string)
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        }
    }
    out
}

fn base64_url_decode(input: &str) -> Result<Vec<u8>, ()> {
    let mut values = Vec::with_capacity(input.len());
    for b in input.bytes() {
        let value = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            b'=' => continue,
            _ => return Err(()),
        };
        values.push(value);
    }
    if values.len() % 4 == 1 {
        return Err(());
    }
    let output_len = values.len() * 3 / 4;
    while values.len() % 4 != 0 {
        values.push(0);
    }

    let mut out = Vec::with_capacity(output_len);
    for chunk in values.chunks(4) {
        out.push((chunk[0] << 2) | (chunk[1] >> 4));
        out.push((chunk[1] << 4) | (chunk[2] >> 2));
        out.push((chunk[2] << 6) | chunk[3]);
    }
    out.truncate(output_len);
    Ok(out)
}

fn now_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn to_wire_request(model: &ModelId, req: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": model.as_str(),
        "store": false,
        "stream": true,
        "input": wire_input(&req.messages),
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(system) = &req.system {
        body["instructions"] = Value::String(system.clone());
    }
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(req.tools.iter().map(wire_tool).collect());
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = json!(temp);
    }

    body
}

fn wire_input(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut msg_index = 0_u32;
    for msg in messages {
        match msg {
            Message::System { content, .. } => out.push(json!({
                "role": "system",
                "content": content,
            })),
            Message::User { content, .. } => {
                let mut user_parts = Vec::new();
                for block in content {
                    match block {
                        Content::Text { text } => {
                            user_parts.push(json!({ "type": "input_text", "text": text }));
                        }
                        Content::ToolResult {
                            tool_use_id,
                            output,
                            ..
                        } => {
                            let (call_id, _) = split_tool_id(tool_use_id);
                            out.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": tool_output_text(output),
                            }));
                        }
                        _ => {}
                    }
                }
                if !user_parts.is_empty() {
                    out.push(json!({ "role": "user", "content": user_parts }));
                }
            }
            Message::Assistant { content, .. } => {
                for block in content {
                    match block {
                        Content::Text { text } => {
                            out.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": format!("msg_{msg_index}"),
                            }));
                            msg_index += 1;
                        }
                        Content::ToolUse { id, name, input } => {
                            let (call_id, item_id) = split_tool_id(id);
                            out.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": name,
                                "arguments": input.to_string(),
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    out
}

fn wire_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    })
}

fn tool_output_text(output: &ToolOutput) -> String {
    let text = output
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        "(non-text tool output omitted)".to_string()
    } else {
        text
    }
}

fn split_tool_id(id: &str) -> (String, String) {
    if let Some((call_id, item_id)) = id.split_once('|') {
        (
            normalize_id_part(call_id),
            normalize_function_item_id(item_id),
        )
    } else {
        let call_id = normalize_id_part(id);
        let item_id = normalize_function_item_id(&call_id);
        (call_id, item_id)
    }
}

fn normalize_function_item_id(id: &str) -> String {
    let normalized = normalize_id_part(id);
    if normalized.starts_with("fc_") {
        normalized
    } else {
        normalize_id_part(&format!("fc_{normalized}"))
    }
}

fn normalize_id_part(id: &str) -> String {
    let mut out = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect::<String>();
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "id".to_string()
    } else {
        out
    }
}

fn reqwest_to_llm_err(e: reqwest::Error) -> LlmError {
    if e.is_timeout() || e.is_connect() {
        LlmError::Network(std::io::Error::other(e.to_string()))
    } else {
        LlmError::provider(e)
    }
}

fn classify_http_error(status: StatusCode, text: &str) -> LlmError {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => LlmError::Authentication,
        StatusCode::TOO_MANY_REQUESTS => LlmError::RateLimited { retry_after: None },
        StatusCode::BAD_REQUEST if text.contains("context") => {
            LlmError::ContextTooLong { max: 0, got: 0 }
        }
        _ => {
            LlmError::MalformedResponse(format!("openai-codex HTTP {}: {}", status.as_u16(), text))
        }
    }
}

#[derive(Default)]
struct CodexStreamState {
    current_text: Option<u32>,
    current_tool: Option<(u32, String)>,
    next_index: u32,
    saw_tool: bool,
    started: bool,
}

fn chunk_stream_from_response(resp: reqwest::Response, fallback_model: ModelId) -> ChunkStream {
    let (mut tx, rx) = mpsc::channel::<Result<Chunk, LlmError>>(32);
    tokio::spawn(async move {
        let mut bytes_stream = Box::pin(resp.bytes_stream());
        let mut buffer = Vec::new();
        let mut state = CodexStreamState::default();

        loop {
            loop {
                let Some(frame) = extract_sse_frame(&mut buffer) else {
                    break;
                };
                match decode_frame(&frame, &mut state, &fallback_model) {
                    Ok(chunks) => {
                        for chunk in chunks {
                            if tx.send(Ok(chunk)).await.is_err() {
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
                    if !state.started {
                        let _ = tx
                            .send(Ok(Chunk::MessageStart {
                                id: String::new(),
                                model: fallback_model,
                            }))
                            .await;
                    }
                    let _ = tx.send(Ok(Chunk::MessageStop)).await;
                    return;
                }
            }
        }
    });
    rx.boxed()
}

fn extract_sse_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pos = buffer.windows(2).position(|w| w == b"\n\n")?;
    let frame = buffer.drain(..pos).collect::<Vec<_>>();
    buffer.drain(..2);
    Some(frame)
}

fn decode_frame(
    frame: &[u8],
    state: &mut CodexStreamState,
    fallback_model: &ModelId,
) -> Result<Vec<Chunk>, LlmError> {
    let text = str::from_utf8(frame)
        .map_err(|e| LlmError::MalformedResponse(format!("openai-codex utf8: {e}")))?;
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");
    if data.trim().is_empty() || data.trim() == "[DONE]" {
        return Ok(Vec::new());
    }

    let value: Value = serde_json::from_str(&data).map_err(|e| {
        LlmError::MalformedResponse(format!("openai-codex SSE decode: {e}: {data}"))
    })?;
    let mut chunks = vec![Chunk::Raw {
        provider: "openai-codex".to_string(),
        event: value.clone(),
    }];
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match event_type {
        "response.created" => {
            state.started = true;
            let response = value.get("response").unwrap_or(&Value::Null);
            let id = response
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let model = response
                .get("model")
                .and_then(Value::as_str)
                .map_or_else(|| fallback_model.clone(), ModelId::new);
            chunks.push(Chunk::MessageStart { id, model });
        }
        "response.output_item.added" => {
            if let Some(item) = value.get("item") {
                decode_output_item_added(item, state, &mut chunks);
            }
        }
        "response.output_text.delta" | "response.refusal.delta" => {
            if let (Some(index), Some(delta)) = (
                state.current_text,
                value.get("delta").and_then(Value::as_str),
            ) {
                chunks.push(Chunk::TextDelta {
                    index,
                    text: delta.to_string(),
                });
            }
        }
        "response.function_call_arguments.delta" => {
            if let (Some((index, partial)), Some(delta)) = (
                state.current_tool.as_mut(),
                value.get("delta").and_then(Value::as_str),
            ) {
                partial.push_str(delta);
                chunks.push(Chunk::ToolUseDelta {
                    index: *index,
                    partial_json: delta.to_string(),
                });
            }
        }
        "response.function_call_arguments.done" => {
            if let (Some((index, partial)), Some(arguments)) = (
                state.current_tool.as_mut(),
                value.get("arguments").and_then(Value::as_str),
            ) {
                if let Some(delta) = arguments.strip_prefix(partial.as_str()) {
                    if !delta.is_empty() {
                        chunks.push(Chunk::ToolUseDelta {
                            index: *index,
                            partial_json: delta.to_string(),
                        });
                    }
                    *partial = arguments.to_string();
                }
            }
        }
        "response.output_item.done" => {
            if let Some(item) = value.get("item") {
                decode_output_item_done(item, state, &mut chunks);
            }
        }
        "response.completed" | "response.done" | "response.incomplete" => {
            let response = value.get("response").unwrap_or(&Value::Null);
            if let Some(usage) = decode_usage(response.get("usage")) {
                chunks.push(Chunk::Usage { usage });
            }
            chunks.push(Chunk::StopReason {
                reason: decode_stop_reason(response, state.saw_tool),
            });
            chunks.push(Chunk::MessageStop);
        }
        "response.failed" => {
            let msg = value
                .pointer("/response/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Codex response failed");
            return Err(LlmError::MalformedResponse(msg.to_string()));
        }
        "error" => {
            let msg = value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Codex stream error");
            return Err(LlmError::MalformedResponse(msg.to_string()));
        }
        _ => {}
    }

    Ok(chunks)
}

fn decode_output_item_added(item: &Value, state: &mut CodexStreamState, chunks: &mut Vec<Chunk>) {
    match item.get("type").and_then(Value::as_str).unwrap_or_default() {
        "message" => {
            let index = state.next_index;
            state.next_index += 1;
            state.current_text = Some(index);
            chunks.push(Chunk::BlockStart {
                index,
                start: BlockStartKind::Text,
            });
        }
        "function_call" => {
            let index = state.next_index;
            state.next_index += 1;
            state.saw_tool = true;
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("call");
            let item_id = item.get("id").and_then(Value::as_str).unwrap_or(call_id);
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let id = format!("{call_id}|{item_id}");
            let initial_arguments = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            state.current_tool = Some((index, initial_arguments.clone()));
            chunks.push(Chunk::BlockStart {
                index,
                start: BlockStartKind::ToolUse { name, id },
            });
            if !initial_arguments.is_empty() {
                chunks.push(Chunk::ToolUseDelta {
                    index,
                    partial_json: initial_arguments,
                });
            }
        }
        _ => {}
    }
}

fn decode_output_item_done(item: &Value, state: &mut CodexStreamState, chunks: &mut Vec<Chunk>) {
    match item.get("type").and_then(Value::as_str).unwrap_or_default() {
        "message" => {
            if let Some(index) = state.current_text.take() {
                chunks.push(Chunk::BlockStop { index });
            }
        }
        "function_call" => {
            if let Some((index, partial)) = state.current_tool.take() {
                if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                    if let Some(delta) = arguments.strip_prefix(partial.as_str()) {
                        if !delta.is_empty() {
                            chunks.push(Chunk::ToolUseDelta {
                                index,
                                partial_json: delta.to_string(),
                            });
                        }
                    }
                }
                chunks.push(Chunk::BlockStop { index });
            }
        }
        _ => {}
    }
}

fn decode_usage(value: Option<&Value>) -> Option<Usage> {
    let usage = value?;
    let input_total = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Some(Usage {
        tokens_input: input_total.saturating_sub(cached),
        tokens_output: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        tokens_cache_read: cached,
        tokens_cache_create: 0,
    })
}

fn decode_stop_reason(response: &Value, saw_tool: bool) -> StopReason {
    match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => StopReason::MaxTokens,
        Some("failed") | Some("cancelled") => StopReason::Error(
            response
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Codex response failed")
                .to_string(),
        ),
        _ if saw_tool => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::message::ToolOutput;

    #[test]
    fn pkce_challenge_uses_url_safe_base64_without_padding() {
        let challenge = pkce_challenge("abc");
        assert_eq!(challenge, "ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0");
    }

    #[test]
    fn parses_account_id_from_access_token() {
        let payload = json!({
            JWT_CLAIM_PATH: {
                "chatgpt_account_id": "acct_123"
            }
        });
        let token = format!(
            "e30.{}.sig",
            base64_url_no_pad(payload.to_string().as_bytes())
        );
        assert_eq!(
            account_id_from_access_token(&token),
            Some("acct_123".to_string())
        );
    }

    #[test]
    fn wire_input_turns_tool_results_into_function_outputs() {
        let messages = vec![Message::User {
            content: vec![Content::ToolResult {
                tool_use_id: "call_1|fc_1".into(),
                output: ToolOutput::text("done"),
                is_error: false,
            }],
            meta: Default::default(),
        }];
        let out = wire_input(&messages);
        assert_eq!(out[0]["type"], "function_call_output");
        assert_eq!(out[0]["call_id"], "call_1");
        assert_eq!(out[0]["output"], "done");
    }

    #[test]
    fn decode_tool_call_streams_arguments() {
        let mut state = CodexStreamState::default();
        let model = ModelId::new("gpt-5.2-codex");
        let frame = br#"data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"write_test","arguments":""}}"#;
        let chunks = decode_frame(frame, &mut state, &model).unwrap();
        assert!(matches!(
            chunks.get(1),
            Some(Chunk::BlockStart {
                start: BlockStartKind::ToolUse { name, id },
                ..
            }) if name == "write_test" && id == "call_1|fc_1"
        ));

        let chunks = decode_frame(
            br#"data: {"type":"response.function_call_arguments.delta","delta":"{\"clause_id\""}"#,
            &mut state,
            &model,
        )
        .unwrap();
        assert!(matches!(
            chunks.get(1),
            Some(Chunk::ToolUseDelta { partial_json, .. }) if partial_json == "{\"clause_id\""
        ));
    }
}
