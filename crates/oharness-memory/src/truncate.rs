//! Drop the oldest messages until the token estimate is under the configured cap.
//! System messages are preserved. A `memory.evicted` event is emitted describing
//! the number and approximate token count of dropped messages.

use crate::policy::{MemoryContext, MemoryError, MemoryPolicy};
use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{ConversationView, Message};
use serde_json::json;

#[derive(Debug, Clone, Copy)]
pub struct TruncateAfterTokens {
    pub max_tokens: u32,
}

impl TruncateAfterTokens {
    pub fn new(max_tokens: u32) -> Self {
        Self { max_tokens }
    }
}

#[async_trait]
impl MemoryPolicy for TruncateAfterTokens {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError> {
        let cap = self.max_tokens.min(ctx.token_budget.max(self.max_tokens));
        let messages = conversation.messages();
        let original = conversation.token_estimate();
        if original <= cap {
            return Ok(messages.to_vec());
        }

        let mut head: Vec<Message> = Vec::new();
        let mut tail: Vec<Message> = Vec::new();
        for m in messages {
            match m {
                Message::System { .. } => head.push(m.clone()),
                _ => tail.push(m.clone()),
            }
        }

        let mut dropped = 0usize;
        while !tail.is_empty() {
            let combined: Vec<Message> = head.iter().chain(tail.iter()).cloned().collect();
            let est = ConversationView::new(&combined).token_estimate();
            if est <= cap {
                break;
            }
            tail.remove(0);
            dropped += 1;
        }

        let final_msgs: Vec<Message> = head.into_iter().chain(tail).collect();

        if dropped > 0 {
            let payload = json!({
                "dropped": dropped,
                "cap_tokens": cap,
                "estimate_before": original,
                "estimate_after": ConversationView::new(&final_msgs).token_estimate(),
            });
            let _ = ctx
                .events
                .try_emit("memory-0", EventKind::MemoryEvicted(payload), None);
        }

        Ok(final_msgs)
    }
}
