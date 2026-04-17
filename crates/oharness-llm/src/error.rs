//! `LlmError` (§5.3) — structured enough for retry middleware to dispatch.

use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Feature not supported by this provider (e.g., `stream` on a sync-only adapter).
    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    #[error("authentication failure")]
    Authentication,

    #[error("rate limited (retry_after = {retry_after:?})")]
    RateLimited { retry_after: Option<Duration> },

    #[error("context too long: max={max}, got={got}")]
    ContextTooLong { max: u32, got: u32 },

    #[error("cancelled")]
    Cancelled,

    #[error("network: {0}")]
    Network(#[from] std::io::Error),

    #[error("malformed response: {0}")]
    MalformedResponse(String),

    #[error("provider error: {0}")]
    Provider(Box<dyn std::error::Error + Send + Sync>),
}

impl LlmError {
    pub fn provider<E: std::error::Error + Send + Sync + 'static>(e: E) -> Self {
        LlmError::Provider(Box::new(e))
    }
}

/// Construction-time error returned from `LlmLayer::wrap` when a capability
/// requirement isn't met.
#[derive(Debug, thiserror::Error)]
pub enum LayerError {
    #[error("layer `{layer}` requires capability `{capability}`, which the inner Llm does not expose")]
    MissingCapability {
        layer: &'static str,
        capability: &'static str,
    },
    #[error("layer `{layer}` rejected inner Llm: {reason}")]
    Rejected {
        layer: &'static str,
        reason: String,
    },
}
