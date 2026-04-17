//! Shared completion types (§4.9).

use crate::MetadataMap;
use crate::ids::ModelId;
use crate::message::{Content, Message};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "CacheHints::is_empty")]
    pub cache_hints: CacheHints,
    /// Provider extensions, reverse-DNS-namespaced (e.g. `anthropic.thinking`).
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub extensions: MetadataMap,
}

impl CompletionRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            system: None,
            max_tokens: None,
            temperature: None,
            stop_sequences: Vec::new(),
            cache_hints: CacheHints::default(),
            extensions: MetadataMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: ModelId,
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Reason a model stopped producing output. `Error(String)` wraps provider-specific
/// error reasons that don't map to a known variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence(String),
    ToolUse,
    Refusal,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's input. Never transformed — round-trips verbatim.
    pub input_schema: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheHints {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub breakpoints: Vec<CacheBreakpoint>,
}

impl CacheHints {
    pub fn is_empty(&self) -> bool {
        self.breakpoints.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheBreakpoint {
    /// Message index at which to mark a cache point (inclusive upper bound of the
    /// cacheable prefix).
    pub message_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    Short,
    Long,
}

/// Token accounting for a single completion. Fields default to zero so providers
/// that don't report cache figures can leave them untouched.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub tokens_input: u64,
    pub tokens_output: u64,
    #[serde(default)]
    pub tokens_cache_read: u64,
    #[serde(default)]
    pub tokens_cache_create: u64,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.tokens_input += other.tokens_input;
        self.tokens_output += other.tokens_output;
        self.tokens_cache_read += other.tokens_cache_read;
        self.tokens_cache_create += other.tokens_cache_create;
    }
}
