//! Replace old tool results with short placeholders so the context footprint of
//! long tool outputs stops growing. The latest N tool results are preserved verbatim.

use crate::policy::{MemoryContext, MemoryError, MemoryPolicy};
use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::message::{Content, ToolOutput};
use oharness_core::{ConversationView, Message};
use serde_json::json;

#[derive(Debug, Clone, Copy)]
pub struct ElideToolResults {
    /// How many most-recent tool-result blocks to keep verbatim. Older ones are
    /// replaced with a stub that preserves the `tool_use_id` so the conversation
    /// shape (tool_use → tool_result pairing) remains valid.
    pub keep_recent: usize,
}

impl Default for ElideToolResults {
    fn default() -> Self {
        Self { keep_recent: 3 }
    }
}

impl ElideToolResults {
    pub fn new(keep_recent: usize) -> Self {
        Self { keep_recent }
    }
}

#[async_trait]
impl MemoryPolicy for ElideToolResults {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError> {
        let messages = conversation.messages();

        // Count tool_result blocks across the conversation; elide the oldest
        // (total - keep_recent) of them.
        let mut tool_result_positions: Vec<(usize, usize)> = Vec::new(); // (msg_index, content_index)
        for (mi, m) in messages.iter().enumerate() {
            if let Message::User { content, .. } = m {
                for (ci, c) in content.iter().enumerate() {
                    if matches!(c, Content::ToolResult { .. }) {
                        tool_result_positions.push((mi, ci));
                    }
                }
            }
        }

        let total = tool_result_positions.len();
        if total <= self.keep_recent {
            return Ok(messages.to_vec());
        }
        let elide_up_to = total - self.keep_recent;

        let mut out: Vec<Message> = messages.to_vec();
        let mut elided = 0usize;
        for (mi, ci) in tool_result_positions.into_iter().take(elide_up_to) {
            if let Some(Message::User { content, .. }) = out.get_mut(mi) {
                if let Some(slot) = content.get_mut(ci) {
                    if let Content::ToolResult {
                        tool_use_id,
                        is_error,
                        ..
                    } = slot
                    {
                        *slot = Content::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            output: ToolOutput {
                                content: vec![Content::text(
                                    "[elided: output summarised for context length]",
                                )],
                                truncated: true,
                            },
                            is_error: *is_error,
                        };
                        elided += 1;
                    }
                }
            }
        }

        if elided > 0 {
            let _ = ctx.events.try_emit(
                "memory-0",
                EventKind::MemoryEvicted(json!({
                    "elided_tool_results": elided,
                    "kept_recent": self.keep_recent,
                })),
                None,
            );
        }

        Ok(out)
    }
}
