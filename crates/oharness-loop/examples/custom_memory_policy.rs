//! `custom_memory_policy` — implement the `MemoryPolicy` trait from
//! scratch.
//!
//! Memory policies sit between the conversation state and the LLM:
//! on each turn, the loop hands the policy a `ConversationView` of
//! everything so far and the policy returns the `Vec<Message>` the
//! LLM will actually see. The shipped policies cover the common
//! cases:
//!
//! - `Passthrough` — identity, no mangling.
//! - `TruncateAfterTokens` — token-budget-aware tail-drop.
//! - `ElideToolResults` — collapse old tool results into stubs.
//!
//! This example ships a fourth: `KeepLastN`, which preserves a
//! leading system message and drops everything non-system except
//! the last N entries. Trivial but typical — exactly what you'd
//! write the first time you hit context-length issues.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_memory_policy -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_core::{ConversationView, Message, NullSink, RunId, ScopedEmitter};
use oharness_memory::{MemoryContext, MemoryError, MemoryPolicy};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

// ---------------------------------------------------------------------
// KeepLastN — drop everything non-system except the last N messages.
//
// A real impl would probably tag messages with turn indices and drop
// by age; this version operates positionally on the message vector
// for simplicity.
// ---------------------------------------------------------------------

struct KeepLastN {
    n: usize,
}

#[async_trait]
impl MemoryPolicy for KeepLastN {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        _ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError> {
        let all = conversation.messages();

        // Split: any leading system messages (keep them), and the
        // rest (drop all but the last N).
        let (systems, rest): (Vec<&Message>, Vec<&Message>) = all
            .iter()
            .partition(|m| matches!(m, Message::System { .. }));

        let keep_tail_start = rest.len().saturating_sub(self.n);
        let kept_tail = &rest[keep_tail_start..];

        Ok(systems
            .into_iter()
            .chain(kept_tail.iter().copied())
            .cloned()
            .collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a synthetic conversation with 6 messages: 1 system + 5
    // turns. The policy keeps the system + the last 3 non-system
    // messages, dropping the two oldest non-system entries.
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user_text("first user turn"),
        Message::assistant_text("first assistant reply"),
        Message::user_text("second user turn"),
        Message::assistant_text("second assistant reply"),
        Message::user_text("third user turn (the latest)"),
    ];

    let policy = KeepLastN { n: 3 };
    let view = ConversationView::new(&messages);
    // The policy receives a `MemoryContext` wrapping the run's event
    // sink (policies can emit `memory.evicted` / `memory.summarized`
    // / `memory.retrieved` events) plus a token budget. The example
    // wires a `NullSink` for brevity — real runs pass the agent's
    // scoped emitter through.
    let ctx = MemoryContext {
        events: ScopedEmitter::new(
            Arc::new(NullSink) as Arc<dyn oharness_core::EventSink>,
            RunId::new(),
            Arc::new(AtomicU64::new(0)),
        ),
        token_budget: 4_000,
    };

    let transformed = policy.transform(view, &ctx).await?;

    println!("Before ({} messages):", messages.len());
    for (i, m) in messages.iter().enumerate() {
        println!("  {i}: {}", describe(m));
    }
    println!("After  ({} messages):", transformed.len());
    for (i, m) in transformed.iter().enumerate() {
        println!("  {i}: {}", describe(m));
    }

    assert_eq!(transformed.len(), 4, "system + last 3 non-system");
    assert!(
        matches!(transformed[0], Message::System { .. }),
        "leading system preserved"
    );
    println!("\nPolicy preserved the system prompt and kept the last 3 non-system messages ✔");

    Ok(())
}

fn describe(m: &Message) -> String {
    match m {
        Message::System { content, .. } => format!("[system] {content:?}"),
        Message::User { content, .. } => format!("[user] {}", flatten(content)),
        Message::Assistant { content, .. } => format!("[assistant] {}", flatten(content)),
    }
}

fn flatten(content: &[oharness_core::Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            oharness_core::Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
