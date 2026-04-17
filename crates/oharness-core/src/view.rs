//! `ConversationView` (§4.8). Read-only view over the conversation post memory-policy
//! mangling — "what the LLM saw."

use crate::message::{Content, Message};

pub struct ConversationView<'a> {
    messages: &'a [Message],
}

impl<'a> ConversationView<'a> {
    pub fn new(messages: &'a [Message]) -> Self {
        Self { messages }
    }

    pub fn messages(&self) -> &[Message] {
        self.messages
    }

    pub fn last_assistant(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|m| matches!(m, Message::Assistant { .. }))
    }

    /// Strips tool-use/tool-result content blocks. Useful for `UserSimulator`
    /// implementations that should only see human-visible conversation.
    pub fn user_visible(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter_map(|m| match m {
                Message::System { .. } => None,
                Message::User { content, meta } => {
                    let filtered: Vec<Content> = content
                        .iter()
                        .filter(|c| {
                            !matches!(c, Content::ToolUse { .. } | Content::ToolResult { .. })
                        })
                        .cloned()
                        .collect();
                    if filtered.is_empty() {
                        None
                    } else {
                        Some(Message::User {
                            content: filtered,
                            meta: meta.clone(),
                        })
                    }
                }
                Message::Assistant {
                    content,
                    stop_reason,
                    meta,
                } => {
                    let filtered: Vec<Content> = content
                        .iter()
                        .filter(|c| {
                            !matches!(
                                c,
                                Content::ToolUse { .. }
                                    | Content::ToolResult { .. }
                                    | Content::Thinking { .. }
                            )
                        })
                        .cloned()
                        .collect();
                    if filtered.is_empty() {
                        None
                    } else {
                        Some(Message::Assistant {
                            content: filtered,
                            stop_reason: stop_reason.clone(),
                            meta: meta.clone(),
                        })
                    }
                }
            })
            .collect()
    }

    /// Cheap heuristic — 4 chars per token. Useful only for budget estimates, not
    /// for anything the LLM charges for.
    pub fn token_estimate(&self) -> u32 {
        let chars: usize = self
            .messages
            .iter()
            .map(|m| match m {
                Message::System { content, .. } => content.len(),
                Message::User { content, .. } | Message::Assistant { content, .. } => content
                    .iter()
                    .map(|c| match c {
                        Content::Text { text } => text.len(),
                        Content::Thinking { thinking } => thinking.len(),
                        Content::ToolUse { input, .. } => input.to_string().len(),
                        Content::ToolResult { output, .. } => output
                            .content
                            .iter()
                            .map(|inner| match inner {
                                Content::Text { text } => text.len(),
                                _ => 0,
                            })
                            .sum(),
                        Content::Image(_)
                        | Content::Document(_)
                        | Content::Audio(_)
                        | Content::Citation(_) => 0,
                    })
                    .sum(),
            })
            .sum();
        (chars / 4) as u32
    }
}
