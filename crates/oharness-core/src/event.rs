//! Event schema (§4.7). The JSONL format is the source of truth for trajectory files.

use crate::MetadataMap;
use crate::capabilities::LlmCapabilities;
use crate::completion::{CompletionRequest, CompletionResponse, StopReason, Usage};
use crate::context::NamespaceError;
use crate::ids::{RunId, SpanId};
use crate::task::Task;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

/// Semver-tagged schema version. Additive changes bump minor; breaking changes bump
/// major (and become v2+). See §19.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl SchemaVersion {
    pub const CURRENT: SchemaVersion = SchemaVersion { major: 1, minor: 0 };
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl Serialize for SchemaVersion {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SchemaVersion {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let (maj, min) = s
            .split_once('.')
            .ok_or_else(|| serde::de::Error::custom("schema version must be `MAJOR.MINOR`"))?;
        let major = maj.parse().map_err(serde::de::Error::custom)?;
        let minor = min.parse().map_err(serde::de::Error::custom)?;
        Ok(SchemaVersion { major, minor })
    }
}

/// Top-level event envelope. Every event — lifecycle, LLM, tool, memory, etc. — is
/// wrapped in this struct. Spans are represented by two events sharing a `span_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub v: SchemaVersion,
    pub seq: u64,
    pub run_id: RunId,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "time::serde::rfc3339::option"
    )]
    pub timestamp: Option<OffsetDateTime>,
    pub span_id: SpanId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// Flattens `type` and `payload` fields into the envelope. Unknown types deserialize
    /// as `EventKind::Unknown` carrying the raw payload — a forward-compat contract.
    #[serde(flatten)]
    pub kind: EventKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redactions: Vec<String>,
}

impl Event {
    pub fn new(
        seq: u64,
        run_id: RunId,
        span_id: impl Into<SpanId>,
        kind: EventKind,
    ) -> Self {
        Self {
            v: SchemaVersion::CURRENT,
            seq,
            run_id,
            timestamp: Some(OffsetDateTime::now_utc()),
            span_id: span_id.into(),
            parent: None,
            kind,
            redactions: Vec::new(),
        }
    }

    pub fn with_parent(mut self, parent: u64) -> Self {
        self.parent = Some(parent);
        self
    }
}

/// Discriminated event catalog. `type` field is the serde tag; `payload` holds the
/// variant's data. New variants are additive per schema versioning rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
#[non_exhaustive]
pub enum EventKind {
    #[serde(rename = "meta")]
    Meta(MetaPayload),

    #[serde(rename = "run.started")]
    RunStarted(RunStartedPayload),
    #[serde(rename = "run.finished")]
    RunFinished(RunFinishedPayload),

    #[serde(rename = "turn.started")]
    TurnStarted(TurnPayload),
    #[serde(rename = "turn.finished")]
    TurnFinished(TurnFinishedPayload),
    #[serde(rename = "turn.revised")]
    TurnRevised(TurnRevisedPayload),

    #[serde(rename = "llm.request")]
    LlmRequest(LlmRequestPayload),
    #[serde(rename = "llm.response")]
    LlmResponse(LlmResponsePayload),
    #[serde(rename = "llm.stream.chunk")]
    LlmStreamChunk(Value),
    #[serde(rename = "llm.retry")]
    LlmRetry(LlmRetryPayload),
    #[serde(rename = "llm.failed")]
    LlmFailed(LlmFailedPayload),

    #[serde(rename = "tool.call.started")]
    ToolCallStarted(ToolCallStartedPayload),
    #[serde(rename = "tool.call.finished")]
    ToolCallFinished(ToolCallFinishedPayload),
    #[serde(rename = "tool.call.failed")]
    ToolCallFailed(ToolCallFailedPayload),
    #[serde(rename = "tool.approval.requested")]
    ToolApprovalRequested(Value),
    #[serde(rename = "tool.approval.decided")]
    ToolApprovalDecided(Value),

    #[serde(rename = "memory.evicted")]
    MemoryEvicted(Value),
    #[serde(rename = "memory.summarized")]
    MemorySummarized(Value),
    #[serde(rename = "memory.retrieved")]
    MemoryRetrieved(Value),

    #[serde(rename = "budget.exceeded")]
    BudgetExceeded(Value),

    #[serde(rename = "policy.input.checked")]
    PolicyInputChecked(Value),
    #[serde(rename = "policy.output.checked")]
    PolicyOutputChecked(Value),
    #[serde(rename = "policy.blocked")]
    PolicyBlocked(Value),

    #[serde(rename = "planner.proposed")]
    PlannerProposed(Value),
    #[serde(rename = "planner.revised")]
    PlannerRevised(Value),
    #[serde(rename = "planner.committed")]
    PlannerCommitted(Value),

    #[serde(rename = "critic.assessed")]
    CriticAssessed(Value),
    #[serde(rename = "critic.rejected")]
    CriticRejected(Value),
    #[serde(rename = "critic.revised")]
    CriticRevised(Value),
    #[serde(rename = "critic.failed")]
    CriticFailed(Value),

    #[serde(rename = "reflection.generated")]
    ReflectionGenerated(Value),
    #[serde(rename = "reflection.injected")]
    ReflectionInjected(Value),

    #[serde(rename = "human.interrupt")]
    HumanInterrupt(Value),
    #[serde(rename = "human.inject")]
    HumanInject(Value),

    #[serde(rename = "user.simulated.message")]
    UserSimulatedMessage(Value),
    #[serde(rename = "user.simulated.ended")]
    UserSimulatedEnded(Value),

    /// Escape hatch for user-defined events. Namespace MUST NOT start with a built-in
    /// category prefix. Construct via `EventKind::user_log`.
    #[serde(rename = "user.log")]
    UserLog(UserLogPayload),

    /// Forward-compat fallback: unknown event type preserved verbatim.
    #[serde(other)]
    Unknown,
}

/// Namespaces reserved for built-in categories — `user.log` namespaces may not collide.
pub const RESERVED_NAMESPACE_PREFIXES: &[&str] = &[
    "run.",
    "turn.",
    "llm.",
    "tool.",
    "memory.",
    "budget.",
    "policy.",
    "planner.",
    "critic.",
    "reflection.",
    "human.",
    "user.simulated.",
    "meta.",
];

impl EventKind {
    /// Construct a `user.log` event. Returns an error if the namespace is empty or
    /// collides with a built-in category prefix.
    pub fn user_log(
        namespace: impl Into<String>,
        data: Value,
    ) -> Result<Self, NamespaceError> {
        let namespace = namespace.into();
        if namespace.is_empty() {
            return Err(NamespaceError::Empty);
        }
        // A namespace collides if it equals or is prefixed by a reserved category.
        for reserved in RESERVED_NAMESPACE_PREFIXES {
            let bare = reserved.trim_end_matches('.');
            if namespace == bare || namespace.starts_with(reserved) {
                return Err(NamespaceError::BuiltinCollision(namespace));
            }
        }
        Ok(EventKind::UserLog(UserLogPayload { namespace, data }))
    }
}

// -- payload structs ----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaPayload {
    pub schema_version: SchemaVersion,
    pub harness_version: String,
    pub task_snapshot: Task,
    pub llm_capabilities: LlmCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStartedPayload {
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub extra: MetadataMap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFinishedPayload {
    pub termination: String,
    pub turns: u32,
    pub tool_calls: u32,
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub extra: MetadataMap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnPayload {
    pub turn_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnFinishedPayload {
    pub turn_index: u32,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub tool_calls: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRevisedPayload {
    pub original_seq: u64,
    pub replacement_seq: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequestPayload {
    pub request: CompletionRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponsePayload {
    pub response: CompletionResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRetryPayload {
    pub attempt: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmFailedPayload {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallStartedPayload {
    pub tool_name: String,
    pub tool_use_id: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFinishedPayload {
    pub tool_name: String,
    pub tool_use_id: String,
    pub output: Value,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFailedPayload {
    pub tool_name: String,
    pub tool_use_id: String,
    pub reason: String,
    #[serde(default)]
    pub recoverable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLogPayload {
    pub namespace: String,
    #[serde(flatten)]
    pub data: Value,
}

/// Shim alias so callers that want to pattern-match on `EventPayload::X` can import
/// the payload types through one re-export. (Not strictly needed; kept for ergonomics.)
pub type EventPayload = Value;

/// Errors encountered constructing events (currently: namespace collisions).
pub type EventConstructionError = NamespaceError;
