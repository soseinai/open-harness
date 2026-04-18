//! Identifier newtypes (§4.1).

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Identifies a single agent run. UUIDs are chosen for collision-freedom across concurrent
/// runs writing to the same output directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(pub Uuid);

#[cfg(feature = "schemars-export")]
impl schemars::JsonSchema for RunId {
    fn schema_name() -> String {
        "RunId".into()
    }
    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        // UUID v4, rendered as the canonical 36-char hex form.
        schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::String.into()),
            format: Some("uuid".into()),
            ..Default::default()
        }
        .into()
    }
}

impl RunId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Identifies a span (open/close event pair). Strings rather than UUIDs because they
/// are meant to be human-readable: `llm-0`, `tool-3`, `run-0`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars-export", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars-export", schemars(transparent))]
#[serde(transparent)]
pub struct SpanId(pub String);

impl SpanId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<&str> for SpanId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for SpanId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&String> for SpanId {
    fn from(s: &String) -> Self {
        Self(s.clone())
    }
}

/// Provider-qualified model identifier, e.g. `anthropic/claude-opus-4-7`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars-export", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars-export", schemars(transparent))]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        Self(s)
    }
}
