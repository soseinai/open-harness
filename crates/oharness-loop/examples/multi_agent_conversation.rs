//! `multi_agent_conversation` — alternate agent + user turns via
//! [`ConversationLoop`].
//!
//! Where `ReactLoop` drives a single agent-side agent answering one
//! user task, `ConversationLoop` alternates the assistant's replies
//! with a [`UserSimulator`]'s follow-ups until the simulator emits
//! `UserAction::EndConversation`.
//!
//! Both sides are scripted here:
//! - the assistant LLM returns canned responses per turn, and
//! - the user simulator is `ScriptedUserSimulator`, replaying a
//!   pre-written list of user utterances. When the script is
//!   exhausted, the simulator signals `EndConversation`.
//!
//! Production setups typically replace the user side with
//! `LlmUserSimulator` (a persona-driven user LLM), or a handwritten
//! simulator built on top of the `UserSimulator` trait.
//!
//! Termination semantics (plan §12.3):
//! - `UserAction::EndConversation` → `Termination::Completed { EndTurn }`
//! - `ctx.max_turns` exceeded → `Termination::Truncated { MaxTurns }`
//! - Simulator errors → `Termination::Failed { category: UserSimulator }`
//!   (never `EndConversation` — silent-fall-to-end would hide
//!   simulator bugs in research logs).
//!
//! Run with:
//!
//! ```bash
//! cargo run --example multi_agent_conversation -p oharness-loop --features conversation
//! ```

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason,
    Task, Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ConversationLoop, ScriptedUserSimulator};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct ScriptedAssistant {
    responses: Vec<CompletionResponse>,
    cursor: AtomicU32,
}

#[async_trait]
impl Llm for ScriptedAssistant {
    fn name(&self) -> &str {
        "scripted-assistant"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst) as usize;
        self.responses
            .get(idx)
            .cloned()
            .ok_or(LlmError::Unsupported("assistant script exhausted"))
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

fn text_response(model: &str, text: &str) -> CompletionResponse {
    CompletionResponse {
        id: "msg".into(),
        model: ModelId::new(model),
        content: vec![Content::text(text)],
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            tokens_input: 5,
            tokens_output: 10,
            ..Default::default()
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // User side — a 3-turn script. The first entry is the initial
    // message; subsequent entries are replies to the assistant.
    // When exhausted, the simulator emits EndConversation and the
    // loop terminates with Completed.
    let user = ScriptedUserSimulator::new([
        "Hi! Can you help me pick a library for unit testing in Rust?",
        "How does it compare to `criterion`?",
        "Thanks, that's helpful.",
    ]);

    // Assistant side — one canned reply per user turn.
    let assistant_script = vec![
        text_response(
            "scripted",
            "Sure — for unit testing, the built-in `#[test]` attribute + \
             `assert_eq!` covers most cases; for BDD-ish feel, try \
             `rstest`.",
        ),
        text_response(
            "scripted",
            "`criterion` is the canonical benchmarking crate, not a unit-test \
             framework. Use it alongside `#[test]` for micro-benchmarks.",
        ),
        text_response("scripted", "You're welcome! Happy testing."),
    ];

    let assistant: Arc<dyn Llm> = Arc::new(ScriptedAssistant {
        responses: assistant_script,
        cursor: AtomicU32::new(0),
    });

    let sink = Arc::new(InMemorySink::new());
    let agent = Agent::builder()
        .with_llm(assistant)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ConversationLoop::new(user).with_system_prompt(
            "You are a helpful, concise Rust library guide.",
        )))
        .with_max_turns(10)
        .build()?;

    let outcome = agent.run(Task::new("Rust library recommendations")).await?;

    // Simulator ran out → EndConversation → Completed termination.
    assert!(matches!(outcome.termination, Termination::Completed { .. }));
    println!("Termination: {:?}", outcome.termination);
    println!("Turns: {}", outcome.usage.turns);

    // Print the full transcript as the simulator + assistant
    // produced it.
    println!("\nTranscript:");
    for msg in &outcome.final_messages {
        match msg {
            Message::System { content, .. } => println!("  [system] {content}"),
            Message::User { content, .. } => println!("  [user] {}", flatten(content)),
            Message::Assistant { content, .. } => println!("  [assistant] {}", flatten(content)),
        }
    }

    Ok(())
}

fn flatten(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}
