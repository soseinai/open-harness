//! End-to-end test for the `ConversationLoop` with a `ScriptedUserSimulator`.
//! Behind the `conversation` feature flag.

#![cfg(feature = "conversation")]

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ConversationLoop, ScriptedUserSimulator};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct EchoLlm(AtomicU32);

#[async_trait]
impl Llm for EchoLlm {
    fn name(&self) -> &str {
        "echo"
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }
    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResponse {
            id: format!("assistant-{n}"),
            model: ModelId::new("echo"),
            content: vec![Content::text(format!("ack {n}"))],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }
    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::test]
async fn conversation_loop_terminates_when_simulator_ends() {
    let sink = Arc::new(InMemorySink::new());
    // Script: "hello" is the initial message; after the assistant's first
    // ack the simulator responds with "one more"; after the next ack the
    // script is exhausted → EndConversation.
    let simulator = ScriptedUserSimulator::new(["hello", "one more"]);
    let agent = Agent::builder()
        .with_llm(Arc::new(EchoLlm(AtomicU32::new(0))))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ConversationLoop::new(simulator)))
        .with_max_turns(10)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("chat with me")).await.expect("run ok");
    assert!(matches!(outcome.termination, Termination::Completed { .. }));
    // At least 2 assistant turns (one after each user utterance) →
    // turns >= 2.
    assert!(outcome.usage.turns >= 2);

    // user.simulated.message appears at least once; user.simulated.ended
    // appears exactly once.
    let events = sink.events();
    let simulated_msgs = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::UserSimulatedMessage(_)))
        .count();
    let simulated_ends = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::UserSimulatedEnded(_)))
        .count();
    assert!(simulated_msgs >= 1);
    assert_eq!(simulated_ends, 1);
}
