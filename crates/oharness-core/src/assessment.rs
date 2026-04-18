//! Shared critic / reflector support types (plan §11.1, §11.4, §13.1).
//!
//! These live in core — not in `oharness-critic` — because the critic, loop,
//! reflector, and eval crates all touch them. Keeping them central prevents a
//! dependency tangle (plan §11.1 / §11.4 / §3.3 / §6 all explicitly call this
//! out) and lets consumer crates pattern-match on the concrete types without
//! re-exporting through four different APIs.

use crate::completion::{StopReason, Usage};
use crate::event::Event;
use crate::ids::SpanId;
use crate::message::Message;
use crate::outcome::RunOutcome;
use crate::task::Task;
use crate::trajectory::TrajectoryHandle;
use crate::MetadataMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

// ======================================================================
// AssistantTurn + ToolCall (plan §11.1)
// ======================================================================

/// A single completed assistant turn as the loop hands it off to a critic or
/// captures it for reflexion. Distinct from `Message::Assistant` because it
/// bundles the turn metadata the critic and loop consumers both need —
/// turn index, span id tying to `llm.request` / `llm.response` events,
/// parsed tool calls, token usage, and the stop reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantTurn {
    pub turn_index: u32,
    pub span_id: SpanId,
    /// Always `Message::Assistant { .. }`.
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub stop_reason: StopReason,
}

impl AssistantTurn {
    /// Build from an already-assembled assistant message. `span_id` ties the
    /// turn to its `llm.request` / `llm.response` events; the loop produces
    /// one per turn via `RequestTracer`'s `llm-<N>` span.
    pub fn new(
        turn_index: u32,
        span_id: impl Into<SpanId>,
        message: Message,
        usage: Usage,
        stop_reason: StopReason,
    ) -> Self {
        let tool_calls = tool_calls_from_message(&message);
        Self {
            turn_index,
            span_id: span_id.into(),
            message,
            tool_calls,
            usage,
            stop_reason,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

fn tool_calls_from_message(message: &Message) -> Vec<ToolCall> {
    use crate::message::Content;
    let Message::Assistant { content, .. } = message else {
        return Vec::new();
    };
    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolUse { id, name, input } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            _ => None,
        })
        .collect()
}

// ======================================================================
// TrajectoryView (plan §11.1)
// ======================================================================

/// Read-only lifetime-scoped view into the trajectory being built mid-run.
/// Backed by the in-memory tail of the event stream. Distinct from
/// [`TrajectoryHandle`] — the handle is a *post-run* reference used to load
/// completed runs, `TrajectoryView` is a *mid-run* peek passed to critics.
///
/// Critics MUST NOT mutate the trajectory; this type enforces that by
/// exposing only read accessors.
pub struct TrajectoryView<'a> {
    events: &'a [Event],
}

impl<'a> TrajectoryView<'a> {
    pub fn new(events: &'a [Event]) -> Self {
        Self { events }
    }

    pub fn events(&self) -> &[Event] {
        self.events
    }

    /// Count of completed turns in the view — one per `turn.finished` event.
    pub fn turn_count(&self) -> u32 {
        use crate::event::EventKind;
        self.events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TurnFinished(_)))
            .count() as u32
    }

    /// Copy the view's events into an in-memory [`TrajectoryHandle`]. Useful
    /// for critics / reflectors that want to pass the view downstream to an
    /// API that takes a handle (e.g., [`crate::TrajectoryHandle::in_memory`]
    /// via `ReplayLlm`).
    pub fn to_handle(&self) -> TrajectoryHandle {
        TrajectoryHandle::in_memory(self.events.to_vec())
    }
}

// ======================================================================
// EvaluationResult (plan §13.1)
// ======================================================================

/// Result of a [`Task`] evaluation. Lives in core — not in `oharness-eval` —
/// because `Episode` / reflexion references it, and the `TaskEvaluator`
/// trait in the eval crate is the producer but not the only consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// 0.0..1.0 is the typical range; arbitrary values are allowed so
    /// evaluators can report whatever metric makes sense for the benchmark.
    pub score: f64,
    /// Binary pass/fail. For graded benchmarks a threshold on `score`
    /// populates this; for strict benchmarks it's the authoritative signal.
    pub passed: bool,
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub details: MetadataMap,
}

/// Scores a completed [`RunOutcome`] against the [`Task`] that produced
/// it. Lives in core because it sits at the seam between the loop (which
/// drives runs), the critic crate (which uses it for reflexion), and the
/// eval crate (which drives benchmarks and aggregates across many
/// outcomes). Keeping the trait central dodges a `loop → eval → loop`
/// dependency cycle.
#[async_trait]
pub trait TaskEvaluator: Send + Sync {
    async fn evaluate(&self, task: &Task, outcome: &RunOutcome) -> EvaluationResult;
}

impl EvaluationResult {
    pub fn pass() -> Self {
        Self {
            score: 1.0,
            passed: true,
            details: MetadataMap::new(),
        }
    }

    pub fn fail() -> Self {
        Self {
            score: 0.0,
            passed: false,
            details: MetadataMap::new(),
        }
    }

    /// Score-based constructor. `passed` is derived via a 0.5 threshold
    /// unless you override `details` downstream.
    pub fn scored(score: f64) -> Self {
        Self {
            score,
            passed: score >= 0.5,
            details: MetadataMap::new(),
        }
    }

    pub fn with_details(mut self, details: MetadataMap) -> Self {
        self.details = details;
        self
    }
}

// ======================================================================
// Episode / OwnedEpisode / Reflection (plan §11.4)
// ======================================================================

/// A single iteration within a `run_reflexion` sweep. Borrows from the
/// caller's locals to avoid a `clone()` on every reflection pass; use
/// [`Episode::to_owned`] when you need to retain the value past the
/// iteration's lifetime (e.g., appending to a returned `Vec<OwnedEpisode>`).
pub struct Episode<'a> {
    pub index: u32,
    pub task: &'a Task,
    pub outcome: &'a RunOutcome,
    pub evaluation: &'a EvaluationResult,
    pub prior_reflections: &'a [Reflection],
}

impl<'a> Episode<'a> {
    pub fn to_owned(&self) -> OwnedEpisode {
        OwnedEpisode {
            index: self.index,
            task: self.task.clone(),
            outcome: self.outcome.clone(),
            evaluation: self.evaluation.clone(),
            prior_reflections: self.prior_reflections.to_vec(),
        }
    }
}

/// Owned counterpart of [`Episode<'a>`] for storage after the iteration's
/// locals drop — e.g., the return value from `run_reflexion`.
#[derive(Debug, Clone)]
pub struct OwnedEpisode {
    pub index: u32,
    pub task: Task,
    pub outcome: RunOutcome,
    pub evaluation: EvaluationResult,
    pub prior_reflections: Vec<Reflection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub text: String,
    #[serde(default, skip_serializing_if = "MetadataMap::is_empty")]
    pub metadata: MetadataMap,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

impl Reflection {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            metadata: MetadataMap::new(),
            created_at: OffsetDateTime::now_utc(),
        }
    }

    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Content;
    use crate::MetadataMap;

    #[test]
    fn assistant_turn_extracts_tool_calls_from_message() {
        let msg = Message::Assistant {
            content: vec![
                Content::text("Let me check."),
                Content::ToolUse {
                    id: "tu_1".into(),
                    name: "fs_list".into(),
                    input: serde_json::json!({"path": "."}),
                },
                Content::ToolUse {
                    id: "tu_2".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"cmd": "ls"}),
                },
            ],
            stop_reason: Some(StopReason::ToolUse),
            meta: MetadataMap::new(),
        };
        let turn = AssistantTurn::new(0, "span-0", msg, Usage::default(), StopReason::ToolUse);
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].name, "fs_list");
        assert_eq!(turn.tool_calls[1].id, "tu_2");
    }

    #[test]
    fn trajectory_view_turn_count_matches_finished_events() {
        use crate::event::{EventKind, TurnFinishedPayload, TurnPayload};
        use crate::ids::RunId;
        let run = RunId::new();
        let events = vec![
            Event::new(
                0,
                run,
                "turn-0",
                EventKind::TurnStarted(TurnPayload { turn_index: 0 }),
            ),
            Event::new(
                1,
                run,
                "turn-0",
                EventKind::TurnFinished(TurnFinishedPayload {
                    turn_index: 0,
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                    tool_calls: 0,
                }),
            ),
            Event::new(
                2,
                run,
                "turn-1",
                EventKind::TurnStarted(TurnPayload { turn_index: 1 }),
            ),
            Event::new(
                3,
                run,
                "turn-1",
                EventKind::TurnFinished(TurnFinishedPayload {
                    turn_index: 1,
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                    tool_calls: 0,
                }),
            ),
        ];
        let view = TrajectoryView::new(&events);
        assert_eq!(view.turn_count(), 2);
        assert_eq!(view.events().len(), 4);
    }

    #[test]
    fn evaluation_result_constructors() {
        assert!(EvaluationResult::pass().passed);
        assert_eq!(EvaluationResult::pass().score, 1.0);
        assert!(!EvaluationResult::fail().passed);
        assert!(EvaluationResult::scored(0.7).passed);
        assert!(!EvaluationResult::scored(0.4).passed);
    }

    #[test]
    fn reflection_round_trips_through_serde() {
        let r = Reflection::new("next time, check the imports first");
        let bytes = serde_json::to_vec(&r).unwrap();
        let back: Reflection = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.text, r.text);
    }
}
