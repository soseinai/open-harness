//! Message & content types (§4.2).

use crate::completion::StopReason;
use crate::MetadataMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use url::Url;

/// A conversation message. Three roles; assistant turns carry a stop reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        meta: MetadataMap,
    },
    User {
        content: Vec<Content>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        meta: MetadataMap,
    },
    Assistant {
        content: Vec<Content>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<StopReason>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        meta: MetadataMap,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            meta: MetadataMap::new(),
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![Content::Text(text.into())],
            meta: MetadataMap::new(),
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![Content::Text(text.into())],
            stop_reason: None,
            meta: MetadataMap::new(),
        }
    }
}

/// A single content block inside a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        output: ToolOutput,
        #[serde(default)]
        is_error: bool,
    },
    /// Extended-thinking blocks (Anthropic).
    Thinking(String),
    Image(ImageRef),
    Document(DocumentRef),
    Audio(AudioRef),
    Citation(CitationRef),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }
}

/// Structured output of a tool call. Tools can return rich content (text, images, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: Vec<Content>,
    #[serde(default)]
    pub truncated: bool,
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content: vec![Content::Text(s.into())],
            truncated: false,
        }
    }
}

/// Reference to an image — inline bytes, URL, or file path. Research annotations live
/// in `extensions` (reverse-DNS namespaced).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageRef {
    Url {
        url: Url,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    File {
        path: PathBuf,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    Inline {
        mime: String,
        bytes: Vec<u8>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentRef {
    Url {
        url: Url,
        mime: Option<String>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    File {
        path: PathBuf,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    Inline {
        mime: String,
        bytes: Vec<u8>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AudioRef {
    Url {
        url: Url,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    File {
        path: PathBuf,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
    Inline {
        mime: String,
        bytes: Vec<u8>,
        #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
        extensions: MetadataMap,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationRef {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quoted_text: Option<String>,
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub extensions: MetadataMap,
}
