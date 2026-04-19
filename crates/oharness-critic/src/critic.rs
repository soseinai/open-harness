//! [`Critic`] trait, [`CriticVerdict`], [`AssessmentContext`], and the
//! [`CriticTrigger`] config enum (plan §11.1, §11.2).

use async_trait::async_trait;
use oharness_core::{AssistantTurn, ConversationView, Task, TrajectoryView};
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait Critic: Send + Sync {
    /// Short, stable identifier used in trajectory events — e.g.
    /// `"regex-deny"`, `"tests"`, `"llm-judge"`.
    fn name(&self) -> &str;

    /// Inspect `ctx` and return a verdict for the just-completed turn.
    /// Implementations should be fast and side-effect-free beyond any
    /// explicit process invocation (e.g. `TestCritic` running a shell
    /// command).
    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict;
}

/// Forward-through impl so `Arc<dyn Critic>` (and `Arc<T>` for any
/// concrete `T: Critic`) satisfies the `Critic` bound. This is the
/// standard shared-handle idiom — callers holding an `Arc<MyCritic>`
/// can drop it straight into `CompositeCritic.push(Box::new(arc))`
/// without an adapter shim. Symmetric with the `Arc<T>: Llm` /
/// `Arc<T>: RequestLayer` / `Arc<T>: ResponseLayer` impls elsewhere.
#[async_trait]
impl<T: Critic + ?Sized> Critic for std::sync::Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        (**self).assess(ctx).await
    }
}

/// The slice of state a critic gets to see. Borrowed — nothing to keep
/// after `assess` returns.
pub struct AssessmentContext<'a> {
    pub task: &'a Task,
    /// Post-memory-policy view of the conversation. Matches what the LLM
    /// saw on this turn.
    pub conversation: ConversationView<'a>,
    pub latest_turn: &'a AssistantTurn,
    pub trajectory: TrajectoryView<'a>,
}

impl<'a> AssessmentContext<'a> {
    pub fn new(
        task: &'a Task,
        conversation: ConversationView<'a>,
        latest_turn: &'a AssistantTurn,
        trajectory: TrajectoryView<'a>,
    ) -> Self {
        Self {
            task,
            conversation,
            latest_turn,
            trajectory,
        }
    }
}

/// Outcome of `Critic::assess`.
///
/// Per plan §11.1:
/// - [`CriticVerdict::Revise`] **replaces** the assistant turn (not
///   append). The loop emits `turn.revised { original_seq, replacement_seq,
///   reason }` tying the two. Revision depth is capped (default 3,
///   configurable); exceeding the cap converts Revise → Reject.
/// - Critic failures are **fail-open**: the loop treats a panicking /
///   errored critic as `Accept` and emits `critic.failed` — a *required*
///   event so positional replay can detect critic behavior drift.
#[derive(Debug, Clone)]
pub enum CriticVerdict {
    Accept,
    AcceptWithNote(String),
    Reject {
        reason: String,
    },
    Revise {
        replacement: oharness_core::AssistantTurn,
        reason: String,
    },
    Abort {
        reason: String,
    },
}

impl CriticVerdict {
    /// `true` for `Accept` and `AcceptWithNote`. Used by aggregation
    /// policies to count "acceptances" without pattern-matching every site.
    pub fn is_accepting(&self) -> bool {
        matches!(
            self,
            CriticVerdict::Accept | CriticVerdict::AcceptWithNote(_)
        )
    }

    pub fn is_rejecting(&self) -> bool {
        matches!(self, CriticVerdict::Reject { .. })
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, CriticVerdict::Abort { .. })
    }
}

/// When the loop invokes the critic. Agent-level config (plan §11.2) —
/// not a trait method, since a single critic might run under any trigger
/// depending on how the loop was configured.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CriticTrigger {
    /// After each assistant turn. The default.
    #[default]
    AfterAssistant,
    /// After each `user`-role tool-result message is threaded back in.
    AfterToolResult,
    /// Every `n` assistant turns.
    AfterEveryNTurns(u32),
    /// Never automatic — driven by external callers (e.g. a UI button).
    OnDemand,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_classification_helpers() {
        assert!(CriticVerdict::Accept.is_accepting());
        assert!(CriticVerdict::AcceptWithNote("n".into()).is_accepting());
        assert!(CriticVerdict::Reject {
            reason: "no".into()
        }
        .is_rejecting());
        assert!(!CriticVerdict::Reject {
            reason: "no".into()
        }
        .is_accepting());
        assert!(CriticVerdict::Abort { reason: "x".into() }.is_terminal());
        assert!(!CriticVerdict::Accept.is_terminal());
    }

    #[test]
    fn trigger_round_trips_through_serde() {
        let t = CriticTrigger::AfterEveryNTurns(5);
        let s = serde_json::to_string(&t).unwrap();
        let back: CriticTrigger = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn trigger_default_is_after_assistant() {
        assert_eq!(CriticTrigger::default(), CriticTrigger::AfterAssistant);
    }
}
