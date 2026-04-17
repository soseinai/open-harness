//! `complete_from_stream` (§5.4). Providers that share implementation between
//! `complete()` and `stream()` use this helper to reduce a chunk stream back to a
//! `CompletionResponse`.

use crate::chunk::{BlockStartKind, Chunk};
use crate::error::LlmError;
use futures::{Stream, StreamExt};
use oharness_core::{CompletionResponse, Content, ModelId, StopReason, Usage};
use serde_json::Value;

/// Drain a chunk stream into a single `CompletionResponse`. Useful for providers that
/// implement streaming natively and want to derive `complete()` from it.
pub async fn complete_from_stream<S>(mut stream: S) -> Result<CompletionResponse, LlmError>
where
    S: Stream<Item = Result<Chunk, LlmError>> + Unpin,
{
    let mut response = CompletionResponse {
        id: String::new(),
        model: ModelId::new(""),
        content: Vec::new(),
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    };

    // Per-block accumulators, keyed by block index.
    let mut text_blocks: Vec<(u32, String)> = Vec::new();
    let mut tool_blocks: Vec<(u32, String, String, String)> = Vec::new(); // (index, id, name, partial_json)
    let mut thinking_blocks: Vec<(u32, String)> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        match chunk {
            Chunk::MessageStart { id, model } => {
                response.id = id;
                response.model = model;
            }
            Chunk::BlockStart { index, start } => match start {
                BlockStartKind::Text => text_blocks.push((index, String::new())),
                BlockStartKind::ToolUse { name, id } => {
                    tool_blocks.push((index, id, name, String::new()))
                }
                BlockStartKind::Thinking => thinking_blocks.push((index, String::new())),
            },
            Chunk::TextDelta { index, text } => {
                if let Some(slot) = text_blocks.iter_mut().find(|(i, _)| *i == index) {
                    slot.1.push_str(&text);
                }
            }
            Chunk::ToolUseDelta {
                index,
                partial_json,
            } => {
                if let Some(slot) = tool_blocks.iter_mut().find(|(i, _, _, _)| *i == index) {
                    slot.3.push_str(&partial_json);
                }
            }
            Chunk::ThinkingDelta { index, text } => {
                if let Some(slot) = thinking_blocks.iter_mut().find(|(i, _)| *i == index) {
                    slot.1.push_str(&text);
                }
            }
            Chunk::BlockStop { .. } => {}
            Chunk::StopReason { reason } => response.stop_reason = reason,
            Chunk::Usage { usage } => response.usage = usage,
            Chunk::MessageStop => break,
            Chunk::Raw { .. } => { /* pass through; raw events don't shape the response */ }
        }
    }

    // Assemble content in block-index order.
    let mut content: Vec<(u32, Content)> = Vec::new();
    for (idx, text) in text_blocks {
        content.push((idx, Content::Text(text)));
    }
    for (idx, id, name, partial) in tool_blocks {
        let input = if partial.trim().is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&partial).unwrap_or(Value::String(partial))
        };
        content.push((idx, Content::ToolUse { id, name, input }));
    }
    for (idx, text) in thinking_blocks {
        content.push((idx, Content::Thinking(text)));
    }
    content.sort_by_key(|(i, _)| *i);
    response.content = content.into_iter().map(|(_, c)| c).collect();

    Ok(response)
}
