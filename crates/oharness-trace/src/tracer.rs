//! Tracing middleware (plan §9.5).
//!
//! Three types move event emission out of the loop and onto the provider /
//! tool boundaries:
//!
//! - [`RequestTracer`] wraps an `Arc<dyn Llm>`, implementing [`Llm`] itself.
//!   Emits `llm.request` before `complete()` / `stream()`, and `llm.response`
//!   (or `llm.failed`) after `complete()`. Per-chunk emission on `stream()` is
//!   delegated to an internal [`StreamTracer`] so the streaming path never
//!   depends on the loop re-implementing the decoder.
//! - [`StreamTracer`] is a [`ChunkObserver`]. Emits one `llm.stream.chunk`
//!   event per chunk. Users composing their own middleware chain attach it
//!   via `LlmExt::with_chunk_observer`.
//! - [`ToolTracer`] wraps an `Arc<dyn ToolSet>`, implementing [`ToolSet`].
//!   Emits `tool.call.started` before `execute()` and `tool.call.finished` /
//!   `tool.call.failed` after.
//!
//! All three tracers share one [`ScopedEmitter`] so the events they produce
//! land in the same sequenced stream as the loop's lifecycle events
//! (`meta`, `run.*`, `turn.*`). Parent-span ids:
//!
//! - `RequestTracer` uses `llm-<N>` where `N` is an atomic per-tracer counter.
//!   The `llm.response` / `llm.failed` close event carries the `llm.request`
//!   open event's `seq` as its `parent`, so readers can pair them without
//!   relying on loop-side context.
//! - `ToolTracer` uses `tool-<N>` with the same pattern.
//!
//! Tool-call emissions need a `tool_use_id` that the LLM assigned. The loop
//! passes it to the tracer via `ToolContext.extensions[TOOL_USE_ID_KEY]`;
//! this is documented as part of the internal contract between `ReactLoop`
//! and `ToolTracer` (see [`TOOL_USE_ID_KEY`] below).

use async_trait::async_trait;
use futures::StreamExt;
use oharness_core::event::{
    EventKind, LlmFailedPayload, LlmRequestPayload, LlmResponsePayload, ToolCallFailedPayload,
    ToolCallFinishedPayload, ToolCallStartedPayload,
};
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ScopedEmitter, ToolSpec,
};
use oharness_llm::{Chunk, ChunkObserver, ChunkStream, Llm, LlmError};
use oharness_tools::context::ToolContext;
use oharness_tools::toolset::{ToolOutcome, ToolSet};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// The reverse-DNS-namespaced key the loop uses to pass the current
/// `tool_use_id` into [`ToolContext::extensions`] so [`ToolTracer`] can
/// populate its payloads. Kept public so other loop implementations can
/// honor the same contract.
pub const TOOL_USE_ID_KEY: &str = "oharness.tool_use_id";

// ======================================================================
// RequestTracer — Llm wrapper
// ======================================================================

pub struct RequestTracer {
    inner: Arc<dyn Llm>,
    emitter: ScopedEmitter,
    call_counter: AtomicU32,
}

impl RequestTracer {
    pub fn new(inner: Arc<dyn Llm>, emitter: ScopedEmitter) -> Self {
        Self {
            inner,
            emitter,
            call_counter: AtomicU32::new(0),
        }
    }

    fn next_span(&self) -> String {
        let n = self.call_counter.fetch_add(1, Ordering::SeqCst);
        format!("llm-{n}")
    }
}

#[async_trait]
impl Llm for RequestTracer {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.inner.capabilities()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let span = self.next_span();
        let req_seq = self.emitter.emit(
            span.as_str(),
            EventKind::LlmRequest(LlmRequestPayload {
                request: req.clone(),
                provider: Some(self.inner.name().to_string()),
            }),
            None,
        );

        match self.inner.complete(req).await {
            Ok(res) => {
                self.emitter.emit(
                    span.as_str(),
                    EventKind::LlmResponse(LlmResponsePayload {
                        response: res.clone(),
                    }),
                    Some(req_seq),
                );
                Ok(res)
            }
            Err(e) => {
                self.emitter.emit(
                    span.as_str(),
                    EventKind::LlmFailed(LlmFailedPayload {
                        reason: e.to_string(),
                    }),
                    Some(req_seq),
                );
                Err(e)
            }
        }
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let span = self.next_span();
        self.emitter.emit(
            span.as_str(),
            EventKind::LlmRequest(LlmRequestPayload {
                request: req.clone(),
                provider: Some(self.inner.name().to_string()),
            }),
            None,
        );

        let upstream = match self.inner.stream(req).await {
            Ok(s) => s,
            Err(e) => {
                self.emitter.emit(
                    span.as_str(),
                    EventKind::LlmFailed(LlmFailedPayload {
                        reason: e.to_string(),
                    }),
                    None,
                );
                return Err(e);
            }
        };

        let emitter = self.emitter.clone();
        let span_owned = span;
        let mapped = upstream.map(move |item| {
            if let Ok(chunk) = &item {
                emit_stream_chunk(&emitter, span_owned.as_str(), chunk);
            }
            item
        });
        Ok(mapped.boxed())
    }
}

fn emit_stream_chunk(emitter: &ScopedEmitter, span: &str, chunk: &Chunk) {
    let payload = serde_json::to_value(chunk).unwrap_or_else(|e| {
        tracing::warn!(target: "oharness.trace", error = %e, "failed to serialize chunk");
        Value::Null
    });
    emitter.emit(span, EventKind::LlmStreamChunk(payload), None);
}

// ======================================================================
// StreamTracer — ChunkObserver
// ======================================================================

/// A standalone [`ChunkObserver`] that emits `llm.stream.chunk` events for
/// each chunk it sees. Use via `LlmExt::with_chunk_observer` when assembling
/// a custom middleware chain; `RequestTracer::stream` performs the same
/// emission inline for the default `Agent::run` wiring.
pub struct StreamTracer {
    emitter: ScopedEmitter,
    span: String,
}

impl StreamTracer {
    pub fn new(emitter: ScopedEmitter) -> Self {
        Self {
            emitter,
            span: "llm-stream".to_string(),
        }
    }

    pub fn with_span(mut self, span: impl Into<String>) -> Self {
        self.span = span.into();
        self
    }
}

impl ChunkObserver for StreamTracer {
    fn on_chunk(&self, chunk: &Chunk) {
        emit_stream_chunk(&self.emitter, self.span.as_str(), chunk);
    }
}

// ======================================================================
// ToolTracer — ToolSet wrapper
// ======================================================================

pub struct ToolTracer {
    inner: Arc<dyn ToolSet>,
    emitter: ScopedEmitter,
    call_counter: AtomicU32,
}

impl ToolTracer {
    pub fn new(inner: Arc<dyn ToolSet>, emitter: ScopedEmitter) -> Self {
        Self {
            inner,
            emitter,
            call_counter: AtomicU32::new(0),
        }
    }

    fn next_span(&self) -> String {
        let n = self.call_counter.fetch_add(1, Ordering::SeqCst);
        format!("tool-{n}")
    }
}

#[async_trait]
impl ToolSet for ToolTracer {
    fn specs(&self) -> &[ToolSpec] {
        self.inner.specs()
    }

    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutcome {
        let tool_use_id = ctx
            .extensions
            .get(TOOL_USE_ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let span = self.next_span();
        let start_seq = self.emitter.emit(
            span.as_str(),
            EventKind::ToolCallStarted(ToolCallStartedPayload {
                tool_name: name.to_string(),
                tool_use_id: tool_use_id.clone(),
                input: input.clone(),
            }),
            None,
        );

        let outcome = self.inner.execute(name, input, ctx).await;
        let close_kind = match &outcome {
            ToolOutcome::Success(output) => {
                let repr = Value::Array(
                    output
                        .content
                        .iter()
                        .map(|c| match c {
                            Content::Text { text } => json!({"type": "text", "text": text}),
                            _ => json!({"type": "other"}),
                        })
                        .collect(),
                );
                EventKind::ToolCallFinished(ToolCallFinishedPayload {
                    tool_name: name.to_string(),
                    tool_use_id,
                    output: repr,
                    truncated: output.truncated,
                })
            }
            ToolOutcome::ExecutionError {
                message,
                recoverable,
            } => EventKind::ToolCallFailed(ToolCallFailedPayload {
                tool_name: name.to_string(),
                tool_use_id,
                reason: message.clone(),
                recoverable: *recoverable,
            }),
            ToolOutcome::Denied { reason } => EventKind::ToolCallFailed(ToolCallFailedPayload {
                tool_name: name.to_string(),
                tool_use_id,
                reason: format!("denied: {reason}"),
                recoverable: false,
            }),
            ToolOutcome::Cancelled => EventKind::ToolCallFailed(ToolCallFailedPayload {
                tool_name: name.to_string(),
                tool_use_id,
                reason: "cancelled".to_string(),
                recoverable: false,
            }),
        };

        self.emitter
            .emit(span.as_str(), close_kind, Some(start_seq));
        outcome
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemorySink;
    use futures::stream;
    use oharness_core::event::EventKind;
    use oharness_core::message::ToolOutput;
    use oharness_core::{Message, ModelId, RunId, StopReason, Usage};
    use oharness_llm::{BlockStartKind, Chunk};
    use std::sync::atomic::AtomicU64;

    fn emitter_with_sink() -> (ScopedEmitter, Arc<InMemorySink>) {
        let sink: Arc<InMemorySink> = Arc::new(InMemorySink::new());
        let seq = Arc::new(AtomicU64::new(0));
        let emitter = ScopedEmitter::new(sink.clone(), RunId::new(), seq);
        (emitter, sink)
    }

    struct ScriptedLlm {
        succeed: bool,
        chunks: std::sync::Mutex<Option<Vec<Result<Chunk, LlmError>>>>,
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
            if self.succeed {
                Ok(CompletionResponse {
                    id: "r".into(),
                    model: ModelId::new("m"),
                    content: vec![Content::text("ok")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            } else {
                Err(LlmError::Authentication)
            }
        }
        async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            let chunks = self.chunks.lock().unwrap().take().unwrap_or_default();
            Ok(stream::iter(chunks).boxed())
        }
    }

    fn req() -> CompletionRequest {
        CompletionRequest::new(vec![Message::user_text("hi")])
    }

    // ---------- RequestTracer ----------

    #[tokio::test]
    async fn request_tracer_emits_request_and_response_on_complete() {
        let (emitter, sink) = emitter_with_sink();
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            succeed: true,
            chunks: std::sync::Mutex::new(None),
        });
        let tracer = RequestTracer::new(llm, emitter);
        tracer.complete(req()).await.unwrap();

        let events = sink.events();
        let kinds: Vec<_> = events.iter().map(|e| event_label(&e.kind)).collect();
        assert_eq!(kinds, vec!["llm.request", "llm.response"]);
        // response references request via parent.
        assert_eq!(events[1].parent, Some(events[0].seq));
    }

    #[tokio::test]
    async fn request_tracer_emits_failed_on_error() {
        let (emitter, sink) = emitter_with_sink();
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            succeed: false,
            chunks: std::sync::Mutex::new(None),
        });
        let tracer = RequestTracer::new(llm, emitter);
        let _ = tracer.complete(req()).await;

        let events = sink.events();
        let kinds: Vec<_> = events.iter().map(|e| event_label(&e.kind)).collect();
        assert_eq!(kinds, vec!["llm.request", "llm.failed"]);
    }

    #[tokio::test]
    async fn request_tracer_emits_request_then_chunks_on_stream() {
        let (emitter, sink) = emitter_with_sink();
        let chunks = vec![
            Ok(Chunk::MessageStart {
                id: "msg".into(),
                model: ModelId::new("m"),
            }),
            Ok(Chunk::BlockStart {
                index: 0,
                start: BlockStartKind::Text,
            }),
            Ok(Chunk::TextDelta {
                index: 0,
                text: "hi".into(),
            }),
            Ok(Chunk::MessageStop),
        ];
        let llm: Arc<dyn Llm> = Arc::new(ScriptedLlm {
            succeed: true,
            chunks: std::sync::Mutex::new(Some(chunks)),
        });
        let tracer = RequestTracer::new(llm, emitter);

        let mut s = tracer.stream(req()).await.unwrap();
        while let Some(result) = s.next().await {
            result.unwrap();
        }

        let events = sink.events();
        let kinds: Vec<_> = events.iter().map(|e| event_label(&e.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                "llm.request",
                "llm.stream.chunk",
                "llm.stream.chunk",
                "llm.stream.chunk",
                "llm.stream.chunk",
            ]
        );
    }

    // ---------- StreamTracer ----------

    #[test]
    fn stream_tracer_emits_one_event_per_chunk() {
        let (emitter, sink) = emitter_with_sink();
        let tracer = StreamTracer::new(emitter);
        tracer.on_chunk(&Chunk::MessageStart {
            id: "m".into(),
            model: ModelId::new("mdl"),
        });
        tracer.on_chunk(&Chunk::MessageStop);

        let events = sink.events();
        assert_eq!(events.len(), 2);
        assert!(events
            .iter()
            .all(|e| matches!(e.kind, EventKind::LlmStreamChunk(_))));
    }

    // ---------- ToolTracer ----------

    struct StubTools;
    #[async_trait]
    impl ToolSet for StubTools {
        fn specs(&self) -> &[ToolSpec] {
            &[]
        }
        async fn execute(&self, _name: &str, _input: Value, _ctx: &ToolContext) -> ToolOutcome {
            ToolOutcome::Success(ToolOutput::text("result"))
        }
    }

    struct FailingTools;
    #[async_trait]
    impl ToolSet for FailingTools {
        fn specs(&self) -> &[ToolSpec] {
            &[]
        }
        async fn execute(&self, _name: &str, _input: Value, _ctx: &ToolContext) -> ToolOutcome {
            ToolOutcome::error("boom", true)
        }
    }

    fn tool_ctx_with_id(id: &str) -> ToolContext {
        let mut ctx = ToolContext::null();
        ctx.extensions
            .insert(TOOL_USE_ID_KEY.to_string(), json!(id));
        ctx
    }

    #[tokio::test]
    async fn tool_tracer_emits_started_and_finished() {
        let (emitter, sink) = emitter_with_sink();
        let tracer = ToolTracer::new(Arc::new(StubTools), emitter);
        let outcome = tracer
            .execute("stub", json!({"x": 1}), &tool_ctx_with_id("tu_1"))
            .await;
        assert!(matches!(outcome, ToolOutcome::Success(_)));

        let events = sink.events();
        let kinds: Vec<_> = events.iter().map(|e| event_label(&e.kind)).collect();
        assert_eq!(kinds, vec!["tool.call.started", "tool.call.finished"]);
        assert_eq!(events[1].parent, Some(events[0].seq));

        // tool_use_id round-trip
        if let EventKind::ToolCallStarted(p) = &events[0].kind {
            assert_eq!(p.tool_use_id, "tu_1");
        } else {
            panic!("expected ToolCallStarted");
        }
    }

    #[tokio::test]
    async fn tool_tracer_emits_failed_on_execution_error() {
        let (emitter, sink) = emitter_with_sink();
        let tracer = ToolTracer::new(Arc::new(FailingTools), emitter);
        let _ = tracer
            .execute("fails", json!({}), &tool_ctx_with_id("tu_2"))
            .await;

        let events = sink.events();
        let kinds: Vec<_> = events.iter().map(|e| event_label(&e.kind)).collect();
        assert_eq!(kinds, vec!["tool.call.started", "tool.call.failed"]);
    }

    fn event_label(kind: &EventKind) -> &'static str {
        match kind {
            EventKind::Meta(_) => "meta",
            EventKind::RunStarted(_) => "run.started",
            EventKind::RunFinished(_) => "run.finished",
            EventKind::TurnStarted(_) => "turn.started",
            EventKind::TurnFinished(_) => "turn.finished",
            EventKind::TurnRevised(_) => "turn.revised",
            EventKind::LlmRequest(_) => "llm.request",
            EventKind::LlmResponse(_) => "llm.response",
            EventKind::LlmStreamChunk(_) => "llm.stream.chunk",
            EventKind::LlmRetry(_) => "llm.retry",
            EventKind::LlmFailed(_) => "llm.failed",
            EventKind::ToolCallStarted(_) => "tool.call.started",
            EventKind::ToolCallFinished(_) => "tool.call.finished",
            EventKind::ToolCallFailed(_) => "tool.call.failed",
            EventKind::ToolApprovalRequested(_) => "tool.approval.requested",
            EventKind::ToolApprovalDecided(_) => "tool.approval.decided",
            EventKind::MemoryEvicted(_) => "memory.evicted",
            EventKind::MemorySummarized(_) => "memory.summarized",
            EventKind::MemoryRetrieved(_) => "memory.retrieved",
            EventKind::BudgetExceeded(_) => "budget.exceeded",
            EventKind::PolicyInputChecked(_) => "policy.input.checked",
            EventKind::PolicyOutputChecked(_) => "policy.output.checked",
            EventKind::PolicyBlocked(_) => "policy.blocked",
            EventKind::PlannerProposed(_) => "planner.proposed",
            EventKind::PlannerRevised(_) => "planner.revised",
            EventKind::PlannerCommitted(_) => "planner.committed",
            EventKind::CriticAssessed(_) => "critic.assessed",
            EventKind::CriticRejected(_) => "critic.rejected",
            EventKind::CriticRevised(_) => "critic.revised",
            EventKind::CriticFailed(_) => "critic.failed",
            EventKind::ReflectionGenerated(_) => "reflection.generated",
            EventKind::ReflectionInjected(_) => "reflection.injected",
            EventKind::HumanInterrupt(_) => "human.interrupt",
            EventKind::HumanInject(_) => "human.inject",
            EventKind::UserSimulatedMessage(_) => "user.simulated.message",
            EventKind::UserSimulatedEnded(_) => "user.simulated.ended",
            EventKind::UserLog(_) => "user.log",
            _ => "unknown",
        }
    }
}
