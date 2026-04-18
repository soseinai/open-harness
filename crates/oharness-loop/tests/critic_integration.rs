//! End-to-end tests exercising the React loop's critic integration:
//! - Accepting critic lets the run complete normally.
//! - Rejecting critic terminates the run with
//!   `Termination::Failed { category: Critic }`.

use async_trait::async_trait;
use oharness_core::event::EventKind;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, RunErrorCategory,
    StopReason, Task, Termination, Usage,
};
use oharness_critic::{
    AggregationPolicy, AssessmentContext, CompositeCritic, Critic, CriticVerdict,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct ScriptedLlm {
    responses: Vec<CompletionResponse>,
    cursor: AtomicU32,
}

#[async_trait]
impl Llm for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted"
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }
    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst) as usize;
        self.responses
            .get(idx)
            .cloned()
            .ok_or(LlmError::Unsupported("ran off end"))
    }
    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

fn single_turn_llm() -> Arc<dyn Llm> {
    Arc::new(ScriptedLlm {
        responses: vec![CompletionResponse {
            id: "msg_1".into(),
            model: ModelId::new("scripted"),
            content: vec![Content::text("Done.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }],
        cursor: AtomicU32::new(0),
    })
}

// ---------- Shipped critic stubs ----------

struct AlwaysAccept;
#[async_trait]
impl Critic for AlwaysAccept {
    fn name(&self) -> &str {
        "always-accept"
    }
    async fn assess(&self, _: &AssessmentContext<'_>) -> CriticVerdict {
        CriticVerdict::Accept
    }
}

struct AlwaysReject;
#[async_trait]
impl Critic for AlwaysReject {
    fn name(&self) -> &str {
        "always-reject"
    }
    async fn assess(&self, _: &AssessmentContext<'_>) -> CriticVerdict {
        CriticVerdict::Reject {
            reason: "test reject".into(),
        }
    }
}

#[tokio::test]
async fn accepting_critic_lets_run_complete() {
    let sink = Arc::new(InMemorySink::new());
    let critics = Arc::new(
        CompositeCritic::new("accept-chain", AggregationPolicy::FirstReject)
            .push(Box::new(AlwaysAccept)),
    );
    let agent = Agent::builder()
        .with_llm(single_turn_llm())
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_critics(critics)
        .with_max_turns(5)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("t")).await.expect("run ok");
    assert!(matches!(outcome.termination, Termination::Completed { .. }));

    // critic.assessed event present on the trajectory.
    let events = sink.events();
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::CriticAssessed(_))));
}

#[tokio::test]
async fn rejecting_critic_fails_run_with_critic_category() {
    let sink = Arc::new(InMemorySink::new());
    let critics = Arc::new(
        CompositeCritic::new("reject-chain", AggregationPolicy::FirstReject)
            .push(Box::new(AlwaysReject)),
    );
    let agent = Agent::builder()
        .with_llm(single_turn_llm())
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_critics(critics)
        .with_max_turns(5)
        .build()
        .expect("agent build");

    let outcome = agent.run(Task::new("t")).await.expect("run ok");
    match outcome.termination {
        Termination::Failed { error, .. } => {
            assert!(matches!(error.category, RunErrorCategory::Critic));
            assert!(error.message.contains("test reject"));
        }
        other => panic!("expected Failed, got {other:?}"),
    }

    // critic.rejected event on the trajectory.
    let events = sink.events();
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, EventKind::CriticRejected(_))));
}
