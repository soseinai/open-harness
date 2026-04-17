//! `MemoryPolicy` trait (§8.1) and supporting types.

use async_trait::async_trait;
use oharness_core::{ConversationView, Message, ScopedEmitter};

#[async_trait]
pub trait MemoryPolicy: Send + Sync {
    /// Transform the conversation before the next LLM call. Input is the full
    /// history; output is what the LLM sees. Policies should emit `memory.evicted`
    /// / `memory.summarized` / `memory.retrieved` events when mangling the view.
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError>;
}

pub struct MemoryContext {
    pub events: ScopedEmitter,
    pub token_budget: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("retriever timed out")]
    RetrieverTimeout,
    #[error("summarizer failed: {0}")]
    SummarizerFailed(String),
    #[error("memory configuration: {0}")]
    Configuration(String),
}
