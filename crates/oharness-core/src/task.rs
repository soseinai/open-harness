//! Task and Attachment types (§4.3).

use crate::MetadataMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use url::Url;

/// Pure data description of a task. No behaviour, no closures. Success predicates
/// live in `TaskEvaluator`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    pub instruction: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,

    /// Reverse-DNS-namespaced keys. Unknown keys round-trip.
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub metadata: MetadataMap,
}

impl Task {
    pub fn new(instruction: impl Into<String>) -> Self {
        Self {
            id: None,
            instruction: instruction.into(),
            attachments: Vec::new(),
            metadata: MetadataMap::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_attachment(mut self, a: Attachment) -> Self {
        self.attachments.push(a);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attachment {
    Text { name: String, content: String },
    File { name: String, path: PathBuf },
    Inline { name: String, mime: String, bytes: Vec<u8> },
    Url { url: Url, mime_hint: Option<String> },
}
