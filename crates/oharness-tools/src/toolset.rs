//! `ToolSet` trait (§7.1) and `ToolOutcome` / `ToolError` types.

use crate::context::ToolContext;
use async_trait::async_trait;
use oharness_core::ToolSpec;
use oharness_core::message::ToolOutput;
use serde_json::Value;

#[async_trait]
pub trait ToolSet: Send + Sync {
    /// Tool specifications the LLM sees in `CompletionRequest.tools`.
    fn specs(&self) -> &[ToolSpec];

    /// Execute a tool by name. `name` is guaranteed to be one returned from `specs()`
    /// by the caller; implementations return `ToolOutcome::ExecutionError` if that
    /// invariant is violated.
    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutcome;
}

#[derive(Debug, Clone)]
pub enum ToolOutcome {
    Success(ToolOutput),
    ExecutionError { message: String, recoverable: bool },
    Denied { reason: String },
    Cancelled,
}

impl ToolOutcome {
    pub fn success_text(s: impl Into<String>) -> Self {
        ToolOutcome::Success(ToolOutput::text(s))
    }

    pub fn error(message: impl Into<String>, recoverable: bool) -> Self {
        ToolOutcome::ExecutionError {
            message: message.into(),
            recoverable,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool `{0}` not found")]
    NotFound(String),
    #[error("invalid input for tool `{name}`: {reason}")]
    InvalidInput { name: String, reason: String },
    #[error("tool execution: {0}")]
    Execution(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}
