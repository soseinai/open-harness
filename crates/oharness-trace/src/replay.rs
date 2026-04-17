//! [`ReplayLlm`] — replays a trajectory as an `Llm` implementation
//! (plan §9.6 / docs/remaining-work.md §2.5).
//!
//! The Nth `complete()` call returns the Nth recorded `llm.response` event
//! (or replays a recorded `llm.failed` error). `stream()` rebuilds a
//! chunk-by-chunk stream from the `llm.stream.chunk` events that sit
//! between the Nth and (N+1)th recorded `llm.request`.
//!
//! Two replay modes:
//!
//! - [`ReplayMode::Positional`] (default): no input comparison. The replay is
//!   strictly sequential — the caller's `CompletionRequest` is ignored
//!   except that its existence advances the counter.
//! - [`ReplayMode::Strict`]: the incoming request's canonical-JSON bytes
//!   must match the recorded `llm.request.request` payload byte-for-byte.
//!   On mismatch a `critic.failed` drift event is emitted (if a drift
//!   emitter was attached) and the [`DriftPolicy`] decides whether to
//!   continue with the recorded response or error out.

use async_trait::async_trait;
use futures::stream;
use futures::StreamExt;
use oharness_core::event::{
    Event, EventKind, LlmFailedPayload, LlmRequestPayload, LlmResponsePayload,
};
use oharness_core::{
    CompletionRequest, CompletionResponse, LlmCapabilities, ScopedEmitter, TrajectoryError,
    TrajectoryHandle,
};
use oharness_llm::{Chunk, ChunkStream, Llm, LlmError};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// How `ReplayLlm` matches replayed calls against recorded ones.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode {
    /// Nth live call → Nth recorded call. No input comparison.
    #[default]
    Positional,
    /// Nth live call must equal the Nth recorded request byte-for-byte
    /// (after canonical JSON serialization). Mismatch triggers the
    /// configured [`DriftPolicy`].
    Strict,
}

/// What to do when [`ReplayMode::Strict`] detects a request drift.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DriftPolicy {
    /// Emit a `critic.failed`-shaped drift event (if a drift emitter is
    /// attached) and return the recorded response anyway.
    #[default]
    WarnAndContinue,
    /// Emit the drift event and return an `LlmError::Provider(ReplayDriftError)`.
    Fail,
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("trajectory has no `meta` event; cannot read capabilities")]
    NoMetaEvent,
    #[error("trajectory cannot be loaded from the given source: {0}")]
    UnsupportedSource(&'static str),
    #[error(transparent)]
    Trajectory(#[from] TrajectoryError),
}

#[derive(Debug, thiserror::Error)]
#[error("replay drift at request #{index}: {reason}")]
pub struct ReplayDriftError {
    pub index: usize,
    pub reason: String,
}

/// Internal: one paired call. Carries the request payload (for Strict
/// comparison) plus the recorded terminal event — either a successful
/// `LlmResponse` or a failure we'll replay as `LlmError::Provider`.
#[derive(Debug, Clone)]
struct ReplayedCall {
    recorded_request: Value,
    outcome: ReplayedOutcome,
    chunks: Vec<Value>,
}

#[derive(Debug, Clone)]
enum ReplayedOutcome {
    Response(CompletionResponse),
    Failed(String),
    MissingClose,
}

pub struct ReplayLlm {
    name: String,
    capabilities: LlmCapabilities,
    calls: Vec<ReplayedCall>,
    mode: ReplayMode,
    on_drift: DriftPolicy,
    cursor: AtomicUsize,
    drift_emitter: Option<ScopedEmitter>,
}

impl ReplayLlm {
    /// Build a replay over an already-loaded set of events.
    pub fn from_events(
        events: Vec<Event>,
        mode: ReplayMode,
        on_drift: DriftPolicy,
    ) -> Result<Self, ReplayError> {
        let capabilities = events
            .iter()
            .find_map(|e| match &e.kind {
                EventKind::Meta(m) => Some(m.llm_capabilities.clone()),
                _ => None,
            })
            .ok_or(ReplayError::NoMetaEvent)?;

        let calls = pair_calls(&events);
        Ok(Self {
            name: "replay".to_string(),
            capabilities,
            calls,
            mode,
            on_drift,
            cursor: AtomicUsize::new(0),
            drift_emitter: None,
        })
    }

    /// Build a replay by reading a JSONL trajectory file.
    pub async fn from_path(
        path: impl AsRef<Path>,
        mode: ReplayMode,
        on_drift: DriftPolicy,
    ) -> Result<Self, ReplayError> {
        let events = crate::reader::read_events(path.as_ref()).await?;
        Self::from_events(events, mode, on_drift)
    }

    /// Build a replay from a [`TrajectoryHandle`]. In-memory handles use
    /// their captured events; file handles load from disk.
    pub async fn from_handle(
        handle: &TrajectoryHandle,
        mode: ReplayMode,
        on_drift: DriftPolicy,
    ) -> Result<Self, ReplayError> {
        if let Some(events) = handle.in_memory_events() {
            return Self::from_events((**events).clone(), mode, on_drift);
        }
        if let Some(path) = handle.path() {
            return Self::from_path(path, mode, on_drift).await;
        }
        Err(ReplayError::UnsupportedSource("unknown"))
    }

    /// Attach an emitter that will receive `critic.failed` drift events.
    /// Optional — drift events are quietly dropped when no emitter is set.
    pub fn with_drift_emitter(mut self, emitter: ScopedEmitter) -> Self {
        self.drift_emitter = Some(emitter);
        self
    }

    /// Override the replay's `name()`. Default: `"replay"`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn take_next_call(&self) -> Result<&ReplayedCall, LlmError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.calls.get(idx).ok_or_else(|| {
            LlmError::Provider(Box::new(ReplayDriftError {
                index: idx,
                reason: format!(
                    "ran past end of trajectory: {} recorded call(s) available",
                    self.calls.len()
                ),
            }))
        })
    }

    fn check_drift(&self, idx: usize, live: &Value, recorded: &Value) -> Result<(), LlmError> {
        if self.mode != ReplayMode::Strict {
            return Ok(());
        }
        if live == recorded {
            return Ok(());
        }

        let reason = drift_reason(live, recorded);
        if let Some(em) = &self.drift_emitter {
            em.emit(
                "replay",
                EventKind::CriticFailed(json!({
                    "source": "replay.drift",
                    "index": idx,
                    "reason": reason,
                })),
                None,
            );
        }

        match self.on_drift {
            DriftPolicy::WarnAndContinue => {
                tracing::warn!(
                    target: "oharness.replay",
                    index = idx,
                    %reason,
                    "replay request drift (continuing with recorded response)",
                );
                Ok(())
            }
            DriftPolicy::Fail => Err(LlmError::Provider(Box::new(ReplayDriftError {
                index: idx,
                reason,
            }))),
        }
    }
}

#[async_trait]
impl Llm for ReplayLlm {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let idx = self.cursor.load(Ordering::SeqCst);
        let call = self.take_next_call()?;

        if self.mode == ReplayMode::Strict {
            let live = serde_json::to_value(&req).map_err(|e| {
                LlmError::MalformedResponse(format!("replay: encode live request: {e}"))
            })?;
            self.check_drift(idx, &live, &call.recorded_request)?;
        }

        match &call.outcome {
            ReplayedOutcome::Response(r) => Ok(r.clone()),
            ReplayedOutcome::Failed(reason) => Err(replay_failure(reason)),
            ReplayedOutcome::MissingClose => Err(LlmError::MalformedResponse(format!(
                "replay: recorded call #{idx} has no llm.response / llm.failed close event"
            ))),
        }
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let idx = self.cursor.load(Ordering::SeqCst);
        let call = self.take_next_call()?;

        if self.mode == ReplayMode::Strict {
            let live = serde_json::to_value(&req).map_err(|e| {
                LlmError::MalformedResponse(format!("replay: encode live request: {e}"))
            })?;
            self.check_drift(idx, &live, &call.recorded_request)?;
        }

        // If the recorded call failed pre-stream, replay the error immediately.
        if let ReplayedOutcome::Failed(reason) = &call.outcome {
            return Err(replay_failure(reason));
        }

        // Decode chunk payloads upfront; errors become stream errors.
        let mut decoded: Vec<Result<Chunk, LlmError>> = Vec::with_capacity(call.chunks.len());
        for (i, raw) in call.chunks.iter().enumerate() {
            match serde_json::from_value::<Chunk>(raw.clone()) {
                Ok(c) => decoded.push(Ok(c)),
                Err(e) => decoded.push(Err(LlmError::MalformedResponse(format!(
                    "replay: decode chunk #{i} at call #{idx}: {e}"
                )))),
            }
        }

        Ok(stream::iter(decoded).boxed())
    }
}

fn replay_failure(reason: &str) -> LlmError {
    LlmError::Provider(Box::new(ReplayDriftError {
        index: 0,
        reason: format!("recorded failure: {reason}"),
    }))
}

fn drift_reason(live: &Value, recorded: &Value) -> String {
    // Give the user something more useful than "JSON differs": surface the
    // top-level keys that disagree.
    if let (Some(a), Some(b)) = (live.as_object(), recorded.as_object()) {
        let mut mismatches = Vec::new();
        for key in a.keys().chain(b.keys()) {
            if a.get(key) != b.get(key) {
                mismatches.push(key.as_str());
            }
        }
        mismatches.sort();
        mismatches.dedup();
        if !mismatches.is_empty() {
            return format!("request fields differ: {}", mismatches.join(", "));
        }
    }
    "request differs".to_string()
}

/// Walk the event stream, pairing each `llm.request` with the first
/// subsequent `llm.response` / `llm.failed` and collecting any
/// `llm.stream.chunk` events that sit between them.
fn pair_calls(events: &[Event]) -> Vec<ReplayedCall> {
    let mut calls: Vec<ReplayedCall> = Vec::new();
    let mut current: Option<ReplayedCall> = None;

    for event in events {
        match &event.kind {
            EventKind::LlmRequest(LlmRequestPayload { request, .. }) => {
                if let Some(prev) = current.take() {
                    // The previous request had no close event in stream —
                    // store it with `MissingClose` so replay surfaces it.
                    calls.push(prev);
                }
                let recorded_request = serde_json::to_value(request).unwrap_or(Value::Null);
                current = Some(ReplayedCall {
                    recorded_request,
                    outcome: ReplayedOutcome::MissingClose,
                    chunks: Vec::new(),
                });
            }
            EventKind::LlmResponse(LlmResponsePayload { response }) => {
                if let Some(call) = current.as_mut() {
                    call.outcome = ReplayedOutcome::Response(response.clone());
                }
            }
            EventKind::LlmFailed(LlmFailedPayload { reason }) => {
                if let Some(call) = current.as_mut() {
                    call.outcome = ReplayedOutcome::Failed(reason.clone());
                }
            }
            EventKind::LlmStreamChunk(payload) => {
                if let Some(call) = current.as_mut() {
                    call.chunks.push(payload.clone());
                }
            }
            _ => {}
        }
    }
    if let Some(last) = current {
        calls.push(last);
    }
    calls
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemorySink;
    use oharness_core::event::{MetaPayload, SchemaVersion};
    use oharness_core::{Content, ModelId, RunId, StopReason, Task, Usage};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    fn mk_event(seq: u64, kind: EventKind) -> Event {
        Event::new(seq, RunId::new(), "span", kind)
    }

    fn caps_meta() -> Event {
        mk_event(
            0,
            EventKind::Meta(MetaPayload {
                schema_version: SchemaVersion::CURRENT,
                harness_version: "0.0.0".into(),
                task_snapshot: Task::new("t"),
                llm_capabilities: LlmCapabilities {
                    streaming: true,
                    prompt_caching: false,
                    parallel_tool_use: false,
                    vision: false,
                    thinking: false,
                    structured_output: false,
                    max_context_tokens: 0,
                    max_output_tokens: 0,
                },
            }),
        )
    }

    fn sample_request(msg: &str) -> CompletionRequest {
        CompletionRequest::new(vec![oharness_core::Message::user_text(msg)])
    }

    fn sample_response(id: &str, text: &str) -> CompletionResponse {
        CompletionResponse {
            id: id.into(),
            model: ModelId::new("m"),
            content: vec![Content::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    fn events_two_calls() -> Vec<Event> {
        vec![
            caps_meta(),
            mk_event(
                1,
                EventKind::LlmRequest(LlmRequestPayload {
                    request: sample_request("one"),
                    provider: Some("scripted".into()),
                }),
            ),
            mk_event(
                2,
                EventKind::LlmResponse(LlmResponsePayload {
                    response: sample_response("r1", "first"),
                }),
            ),
            mk_event(
                3,
                EventKind::LlmRequest(LlmRequestPayload {
                    request: sample_request("two"),
                    provider: Some("scripted".into()),
                }),
            ),
            mk_event(
                4,
                EventKind::LlmResponse(LlmResponsePayload {
                    response: sample_response("r2", "second"),
                }),
            ),
        ]
    }

    // ---------- capabilities + missing meta ----------

    #[test]
    fn from_events_rejects_trajectory_without_meta() {
        let events = vec![mk_event(
            0,
            EventKind::LlmRequest(LlmRequestPayload {
                request: sample_request("x"),
                provider: None,
            }),
        )];
        match ReplayLlm::from_events(events, ReplayMode::Positional, DriftPolicy::default()) {
            Err(ReplayError::NoMetaEvent) => {}
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn capabilities_read_from_meta() {
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap();
        assert!(r.capabilities().streaming);
    }

    // ---------- positional complete ----------

    #[tokio::test]
    async fn positional_returns_recorded_responses_in_order() {
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap();
        let first = r.complete(sample_request("anything")).await.unwrap();
        let second = r.complete(sample_request("still anything")).await.unwrap();
        assert_eq!(first.id, "r1");
        assert_eq!(second.id, "r2");
    }

    #[tokio::test]
    async fn running_off_the_end_errors() {
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap();
        r.complete(sample_request("a")).await.unwrap();
        r.complete(sample_request("b")).await.unwrap();
        match r.complete(sample_request("c")).await {
            Err(LlmError::Provider(e)) => {
                let downcast = e.downcast_ref::<ReplayDriftError>().unwrap();
                assert!(downcast.reason.contains("ran past"));
            }
            other => panic!("expected Provider(ReplayDriftError), got {other:?}"),
        }
    }

    // ---------- strict mode ----------

    #[tokio::test]
    async fn strict_passes_when_requests_match() {
        let r = ReplayLlm::from_events(events_two_calls(), ReplayMode::Strict, DriftPolicy::Fail)
            .unwrap();
        let res = r.complete(sample_request("one")).await.unwrap();
        assert_eq!(res.id, "r1");
    }

    #[tokio::test]
    async fn strict_warn_and_continue_returns_recorded_response() {
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Strict,
            DriftPolicy::WarnAndContinue,
        )
        .unwrap();
        let res = r
            .complete(sample_request("something else entirely"))
            .await
            .unwrap();
        assert_eq!(res.id, "r1");
    }

    #[tokio::test]
    async fn strict_fail_surfaces_error_on_drift() {
        let r = ReplayLlm::from_events(events_two_calls(), ReplayMode::Strict, DriftPolicy::Fail)
            .unwrap();
        match r.complete(sample_request("wrong message")).await {
            Err(LlmError::Provider(e)) => {
                let downcast = e.downcast_ref::<ReplayDriftError>().unwrap();
                assert!(downcast.reason.contains("messages"));
            }
            other => panic!("expected Provider(ReplayDriftError), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn strict_emits_critic_failed_to_drift_emitter() {
        let sink = Arc::new(InMemorySink::new());
        let emitter = ScopedEmitter::new(
            sink.clone() as Arc<dyn oharness_core::EventSink>,
            RunId::new(),
            Arc::new(AtomicU64::new(0)),
        );
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Strict,
            DriftPolicy::WarnAndContinue,
        )
        .unwrap()
        .with_drift_emitter(emitter);

        let _ = r.complete(sample_request("different")).await.unwrap();
        let events = sink.events();
        assert!(events
            .iter()
            .any(|e| matches!(e.kind, EventKind::CriticFailed(_))));
    }

    // ---------- recorded failures ----------

    #[tokio::test]
    async fn recorded_llm_failed_replays_as_error() {
        let events = vec![
            caps_meta(),
            mk_event(
                1,
                EventKind::LlmRequest(LlmRequestPayload {
                    request: sample_request("boom"),
                    provider: None,
                }),
            ),
            mk_event(
                2,
                EventKind::LlmFailed(LlmFailedPayload {
                    reason: "authentication".into(),
                }),
            ),
        ];
        let r =
            ReplayLlm::from_events(events, ReplayMode::Positional, DriftPolicy::default()).unwrap();
        match r.complete(sample_request("ignored")).await {
            Err(LlmError::Provider(e)) => {
                let downcast = e.downcast_ref::<ReplayDriftError>().unwrap();
                assert!(downcast.reason.contains("authentication"));
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    // ---------- stream reconstruction ----------

    fn events_with_stream() -> Vec<Event> {
        let chunks = [
            Chunk::MessageStart {
                id: "msg".into(),
                model: ModelId::new("m"),
            },
            Chunk::TextDelta {
                index: 0,
                text: "hi".into(),
            },
            Chunk::MessageStop,
        ];
        let mut events = vec![
            caps_meta(),
            mk_event(
                1,
                EventKind::LlmRequest(LlmRequestPayload {
                    request: sample_request("stream"),
                    provider: None,
                }),
            ),
        ];
        let mut seq = 2;
        for chunk in &chunks {
            events.push(mk_event(
                seq,
                EventKind::LlmStreamChunk(serde_json::to_value(chunk).unwrap()),
            ));
            seq += 1;
        }
        events
    }

    #[tokio::test]
    async fn stream_reconstructs_recorded_chunks() {
        let r = ReplayLlm::from_events(
            events_with_stream(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap();
        let mut s = r.stream(sample_request("stream")).await.unwrap();
        let mut seen = Vec::new();
        while let Some(c) = s.next().await {
            seen.push(c.unwrap());
        }
        assert_eq!(seen.len(), 3);
        assert!(matches!(seen[0], Chunk::MessageStart { .. }));
        assert!(matches!(&seen[1], Chunk::TextDelta { text, .. } if text == "hi"));
        assert!(matches!(seen[2], Chunk::MessageStop));
    }

    #[test]
    fn name_defaults_to_replay_and_can_be_overridden() {
        let r = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap();
        assert_eq!(r.name(), "replay");
        let named = ReplayLlm::from_events(
            events_two_calls(),
            ReplayMode::Positional,
            DriftPolicy::default(),
        )
        .unwrap()
        .with_name("custom");
        assert_eq!(named.name(), "custom");
    }
}
