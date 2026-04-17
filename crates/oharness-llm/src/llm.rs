//! The `Llm` trait (§5.1). Both `complete` and `stream` are required — providers
//! without streaming return `Err(LlmError::Unsupported("stream"))`.

use crate::error::LlmError;
use crate::ChunkStream;
use async_trait::async_trait;
use oharness_core::{CompletionRequest, CompletionResponse, LlmCapabilities};

#[async_trait]
pub trait Llm: Send + Sync {
    /// Short, user-facing name — used in tracing and error messages.
    fn name(&self) -> &str;

    /// Provider capabilities, returned **by value**. Middleware can override these
    /// when they affect the capabilities the stack exposes (caching, fallback, …).
    fn capabilities(&self) -> LlmCapabilities;

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError>;
}
