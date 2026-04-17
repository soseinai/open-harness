//! Context-plumbing traits (§4.6). Defined here so downstream crates can depend only
//! on `oharness-core` when threading observability / budget / approval / cancellation.

use crate::event::{Event, EventKind};
use crate::ids::{RunId, SpanId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Emit events to wherever the harness routes them.
///
/// Implementations provide BOTH methods explicitly — no defaults — because their
/// semantics differ fundamentally. `emit` blocks on backpressure; `try_emit` never
/// blocks. The split exists so callers can pick the right one at the call site.
///
/// NOTE: `try_emit` returns the event on failure so callers can retry or log. The
/// `Err` variant is larger than clippy likes; we suppress the lint because the
/// ownership hand-back is load-bearing to the contract.
#[allow(clippy::result_large_err)]
pub trait EventSink: Send + Sync {
    /// Blocking-as-specified emit.
    ///
    /// - Default shipped sinks: `try_send` first; on `Full`, block via
    ///   `tokio::task::spawn_blocking` so a tokio worker thread is not stalled.
    /// - `NullSink`: discards unconditionally.
    /// - Call from `Drop`: callers SHOULD use `try_emit` instead.
    fn emit(&self, event: Event);

    /// Non-blocking variant. Returns `Err(event)` immediately on a full channel so the
    /// caller can decide policy (drop, buffer locally, log). Never blocks, never spawns.
    fn try_emit(&self, event: Event) -> Result<(), Event>;
}

/// A sink that silently discards events. Default for `AgentBuilder` — a library call
/// must not write to `cwd` implicitly.
#[derive(Debug, Default, Clone)]
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: Event) {}
    fn try_emit(&self, _event: Event) -> Result<(), Event> {
        Ok(())
    }
}

/// Budget state accessible from tools and middleware.
#[async_trait]
pub trait BudgetHandle: Send + Sync {
    async fn check(&self, request: BudgetRequest) -> BudgetDecision;
    async fn consume(&self, amount: BudgetAmount);
    fn snapshot(&self) -> BudgetSnapshot;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetAmount {
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub cost_usd: f64,
    #[serde(with = "crate::context::duration_seconds")]
    pub wall_clock: Duration,
    pub steps: u32,
}

/// A pre-call estimate supplied to `BudgetHandle::check`. Fields are all optional —
/// callers populate what they can estimate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetRequest {
    pub estimated_input_tokens: Option<u64>,
    pub estimated_output_tokens: Option<u64>,
    pub estimated_cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum BudgetDecision {
    Allow,
    Deny { reason: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub consumed: BudgetAmount,
    pub remaining: Option<BudgetAmount>,
}

/// A no-op budget that always allows. Default in `AgentBuilder`.
#[derive(Debug, Default, Clone)]
pub struct NullBudget;

#[async_trait]
impl BudgetHandle for NullBudget {
    async fn check(&self, _request: BudgetRequest) -> BudgetDecision {
        BudgetDecision::Allow
    }
    async fn consume(&self, _amount: BudgetAmount) {}
    fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot::default()
    }
}

/// Approval channel for tool calls needing human/external OK.
#[async_trait]
pub trait ApprovalChannel: Send + Sync {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub tool_name: String,
    pub input: Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApprovalResponse {
    Allow,
    Deny(String),
}

/// A default approval channel that allows everything. Suitable for non-interactive
/// contexts; CLI/UI surfaces provide interactive variants.
#[derive(Debug, Default, Clone)]
pub struct NullApprovalChannel;

#[async_trait]
impl ApprovalChannel for NullApprovalChannel {
    async fn request(&self, _req: ApprovalRequest) -> ApprovalResponse {
        ApprovalResponse::Allow
    }
}

/// Stable alias for the cancellation primitive. Lets us swap the underlying type if
/// `tokio-util`'s token is ever replaced without churning every `use` site.
pub type Cancellation = tokio_util::sync::CancellationToken;

/// Error returned when event-payload construction rejects an invalid namespace for
/// `user.log` events (§4.7).
#[derive(Debug, thiserror::Error)]
pub enum NamespaceError {
    #[error("`user.log` namespace `{0}` collides with a built-in event-category prefix")]
    BuiltinCollision(String),
    #[error("`user.log` namespace must be non-empty")]
    Empty,
}

/// Convenient alias: `Arc`-wrapped sink, the type every subsystem receives.
pub type SharedSink = Arc<dyn EventSink>;

/// Run-scoped event emitter. Subsystems (tools, memory, middleware) receive one of
/// these so they don't need to know about run_id or the monotonic seq counter — they
/// just call `.emit(span_id, kind)`.
///
/// The plan's context types show `events: Arc<dyn EventSink>` alone; in practice the
/// envelope needs a `seq` and `run_id` that the loop owns. This wrapper closes that
/// gap without changing the `EventSink` trait.
#[derive(Clone)]
pub struct ScopedEmitter {
    sink: Arc<dyn EventSink>,
    run_id: RunId,
    seq: Arc<AtomicU64>,
}

impl ScopedEmitter {
    pub fn new(sink: Arc<dyn EventSink>, run_id: RunId, seq: Arc<AtomicU64>) -> Self {
        Self { sink, run_id, seq }
    }

    pub fn sink(&self) -> &Arc<dyn EventSink> {
        &self.sink
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst)
    }

    /// Build and emit a stamped event (blocks on backpressure per `EventSink::emit`).
    /// Returns the event's `seq` so spans can be linked.
    pub fn emit(&self, span_id: impl Into<SpanId>, kind: EventKind, parent: Option<u64>) -> u64 {
        let seq = self.next_seq();
        let mut event = Event::new(seq, self.run_id, span_id, kind);
        event.parent = parent;
        self.sink.emit(event);
        seq
    }

    /// Non-blocking emit. On backpressure, the event is dropped and `Err(seq)` is
    /// returned so the caller can log or retry.
    pub fn try_emit(
        &self,
        span_id: impl Into<SpanId>,
        kind: EventKind,
        parent: Option<u64>,
    ) -> Result<u64, u64> {
        let seq = self.next_seq();
        let mut event = Event::new(seq, self.run_id, span_id, kind);
        event.parent = parent;
        match self.sink.try_emit(event) {
            Ok(()) => Ok(seq),
            Err(_) => Err(seq),
        }
    }
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(d.as_secs_f64())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}
