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
            content: vec![Content::text(text)],
            meta: MetadataMap::new(),
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![Content::text(text)],
            stop_reason: None,
            meta: MetadataMap::new(),
        }
    }
}

/// A single content block inside a message.
///
/// All variants are struct-shaped so the enum round-trips cleanly through
/// serde's `#[serde(tag = "type")]` representation — serde refuses to
/// serialize tagged newtype variants that wrap a primitive, which would
/// otherwise silently break `llm.request` / `llm.response` event payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
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
    Thinking {
        thinking: String,
    },
    Image(ImageRef),
    Document(DocumentRef),
    Audio(AudioRef),
    Citation(CitationRef),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    pub fn thinking(s: impl Into<String>) -> Self {
        Self::Thinking { thinking: s.into() }
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
            content: vec![Content::text(s)],
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip(content: &Content) -> Content {
        let bytes = serde_json::to_vec(content).expect("serialize Content");
        serde_json::from_slice::<Content>(&bytes).expect("deserialize Content")
    }

    #[test]
    fn text_serializes_as_tagged_struct() {
        let c = Content::text("hello");
        let v = serde_json::to_value(&c).expect("to_value");
        assert_eq!(v, json!({"type": "text", "text": "hello"}));
        assert!(matches!(round_trip(&c), Content::Text { text } if text == "hello"));
    }

    #[test]
    fn thinking_serializes_as_tagged_struct() {
        let c = Content::thinking("hmm");
        let v = serde_json::to_value(&c).expect("to_value");
        assert_eq!(v, json!({"type": "thinking", "thinking": "hmm"}));
        assert!(matches!(round_trip(&c), Content::Thinking { thinking } if thinking == "hmm"));
    }

    #[test]
    fn tool_use_round_trips() {
        let c = Content::ToolUse {
            id: "tu_1".into(),
            name: "fs_list".into(),
            input: json!({"path": "."}),
        };
        let v = serde_json::to_value(&c).expect("to_value");
        assert_eq!(v["type"], "tool_use");
        assert_eq!(v["id"], "tu_1");
        assert_eq!(v["name"], "fs_list");
        assert_eq!(v["input"], json!({"path": "."}));
        assert!(matches!(round_trip(&c), Content::ToolUse { id, .. } if id == "tu_1"));
    }

    #[test]
    fn tool_result_round_trips() {
        let c = Content::ToolResult {
            tool_use_id: "tu_1".into(),
            output: ToolOutput::text("ok"),
            is_error: false,
        };
        let bytes = serde_json::to_vec(&c).expect("serialize");
        let back: Content = serde_json::from_slice(&bytes).expect("deserialize");
        match back {
            Content::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => {
                assert_eq!(tool_use_id, "tu_1");
                assert!(!is_error);
                assert!(!output.truncated);
                assert!(
                    matches!(&output.content[..], [Content::Text { text }] if text == "ok"),
                    "output content: {:?}",
                    output.content
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn message_with_mixed_content_round_trips() {
        let msg = Message::Assistant {
            content: vec![
                Content::text("Here's what I found:"),
                Content::ToolUse {
                    id: "tu_1".into(),
                    name: "fs_list".into(),
                    input: json!({"path": "."}),
                },
            ],
            stop_reason: Some(StopReason::ToolUse),
            meta: MetadataMap::new(),
        };
        let bytes = serde_json::to_vec(&msg).expect("serialize Message");
        let back: Message = serde_json::from_slice(&bytes).expect("deserialize Message");
        match back {
            Message::Assistant { content, .. } => assert_eq!(content.len(), 2),
            other => panic!("expected Assistant, got {other:?}"),
        }
    }
}
