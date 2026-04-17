//! Core types, event schema, and context-plumbing traits for open-harness.
//!
//! Zero-IO foundation. Every other crate in the workspace depends on this one.

pub mod capabilities;
pub mod completion;
pub mod context;
pub mod event;
pub mod ids;
pub mod message;
pub mod outcome;
pub mod task;
pub mod trajectory;
pub mod view;

pub use capabilities::LlmCapabilities;
pub use completion::{
    CacheBreakpoint, CacheHints, CompletionRequest, CompletionResponse, StopReason, ToolSpec,
    Usage,
};
pub use context::{
    ApprovalChannel, ApprovalRequest, ApprovalResponse, BudgetAmount, BudgetDecision,
    BudgetHandle, BudgetRequest, BudgetSnapshot, Cancellation, EventSink, NamespaceError,
    NullApprovalChannel, NullBudget, NullSink, ScopedEmitter, SharedSink,
};
pub use event::{Event, EventConstructionError, EventKind, EventPayload, SchemaVersion};
pub use ids::{ModelId, RunId, SpanId};
pub use message::{
    AudioRef, CitationRef, Content, DocumentRef, ImageRef, Message, ToolOutput,
};
pub use outcome::{
    AgentError, CompletionReason, InterruptionReason, ResourceUsage, RunError, RunErrorCategory,
    RunOutcome, Termination, TruncationLimit,
};
pub use task::{Attachment, Task};
pub use trajectory::{TrajectoryError, TrajectoryHandle, TrajectorySummary};
pub use view::ConversationView;

/// Re-export of `serde_json::Map<String, serde_json::Value>` — the standard "metadata bag"
/// type used on messages, events, tasks, etc.
pub type MetadataMap = serde_json::Map<String, serde_json::Value>;
