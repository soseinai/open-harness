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

/// Forward-through impl so `Arc<dyn Llm>` (and `Arc<T>` for any
/// concrete `T: Llm`) satisfies the `Llm` bound itself. This is the
/// canonical shared-handle idiom — callers holding an
/// `Arc<dyn Llm>` can wrap it in middleware like `BudgetMiddleware`
/// without unwrapping the concrete type first. Symmetric with the
/// `impl<T: RequestLayer + ?Sized> RequestLayer for Arc<T>` pair in
/// `layer.rs`.
#[async_trait]
impl<T: Llm + ?Sized> Llm for std::sync::Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn capabilities(&self) -> LlmCapabilities {
        (**self).capabilities()
    }
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        (**self).complete(req).await
    }
    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        (**self).stream(req).await
    }
}
