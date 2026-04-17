//! No-op memory policy.

use crate::policy::{MemoryContext, MemoryError, MemoryPolicy};
use async_trait::async_trait;
use oharness_core::{ConversationView, Message};

#[derive(Debug, Default, Clone)]
pub struct Passthrough;

#[async_trait]
impl MemoryPolicy for Passthrough {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        _ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError> {
        Ok(conversation.messages().to_vec())
    }
}
