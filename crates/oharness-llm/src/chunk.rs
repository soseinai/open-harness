//! Normalized streaming chunks (§5.2). `Chunk::Raw` is the escape hatch for
//! provider-specific events we don't translate.

use oharness_core::{ModelId, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "chunk", rename_all = "snake_case")]
pub enum Chunk {
    MessageStart {
        id: String,
        model: ModelId,
    },
    BlockStart {
        index: u32,
        #[serde(flatten)]
        start: BlockStartKind,
    },
    TextDelta {
        index: u32,
        text: String,
    },
    /// Partial-JSON deltas — **not** accumulated. Consumers concatenate themselves.
    ToolUseDelta {
        index: u32,
        partial_json: String,
    },
    ThinkingDelta {
        index: u32,
        text: String,
    },
    BlockStop {
        index: u32,
    },
    StopReason {
        reason: StopReason,
    },
    Usage {
        usage: Usage,
    },
    MessageStop,
    /// Provider-specific event passed through verbatim.
    Raw {
        provider: String,
        event: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "block", rename_all = "snake_case")]
pub enum BlockStartKind {
    Text,
    ToolUse { name: String, id: String },
    Thinking,
}
