//! `RunOutcome` and related types (§4.4) plus `AgentError` (§16.2).

use crate::MetadataMap;
use crate::completion::Usage;
use crate::ids::{ModelId, RunId};
use crate::trajectory::TrajectoryHandle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use time::OffsetDateTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutcome {
    pub run_id: RunId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,

    pub termination: Termination,

    /// Final conversation state as the LLM saw it.
    pub final_messages: Vec<crate::message::Message>,

    /// Lazy reference to the event stream. Never inlined on serialization.
    pub trajectory: TrajectoryHandle,

    pub usage: ResourceUsage,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub per_model_usage: HashMap<ModelId, ResourceUsage>,

    #[serde(with = "time::serde::rfc3339")]
    pub started_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub finished_at: OffsetDateTime,

    /// Agent-specific opaque state (e.g. planner scratch, memory summaries).
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub agent_state: MetadataMap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Termination {
    Completed {
        reason: CompletionReason,
    },
    Truncated {
        limit: TruncationLimit,
    },
    Failed {
        error: RunError,
        at_turn: u32,
    },
    Interrupted {
        reason: InterruptionReason,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionReason {
    EndTurn,
    StopSequence(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationLimit {
    MaxTurns(u32),
    Budget(String),
    Timeout,
    MaxTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptionReason {
    User,
    ApprovalDenied(String),
    Cancellation,
}

/// In-run failure captured on the outcome (distinct from `AgentError`, which is
/// returned when the run couldn't even start).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunError {
    pub category: RunErrorCategory,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunErrorCategory {
    Llm,
    Tool,
    Memory,
    Budget,
    UserSimulator,
    Other,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub tokens_cache_read: u64,
    pub tokens_cache_create: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(with = "duration_seconds")]
    pub wall_clock: Duration,
    pub turns: u32,
    pub tool_calls: u32,
}

impl ResourceUsage {
    pub fn add_usage(&mut self, u: &Usage) {
        self.tokens_input += u.tokens_input;
        self.tokens_output += u.tokens_output;
        self.tokens_cache_read += u.tokens_cache_read;
        self.tokens_cache_create += u.tokens_cache_create;
    }
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(d.as_secs_f64())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

/// Top-level error for "couldn't start the run." Mid-run failures are reported via
/// `Termination::Failed` on a successful `Ok(RunOutcome)`.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("llm error: {0}")]
    Llm(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("memory error: {0}")]
    Memory(String),
    #[error("budget exceeded: {0}")]
    Budget(String),
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}
