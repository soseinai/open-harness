//! [`CompositeCritic`] — fans an `assess` call across multiple child critics
//! and reduces via an [`AggregationPolicy`] (plan §11.3).

use crate::critic::{AssessmentContext, Critic, CriticVerdict};
use async_trait::async_trait;
use futures::future::join_all;

pub struct CompositeCritic {
    name: String,
    critics: Vec<Box<dyn Critic>>,
    policy: AggregationPolicy,
}

/// How verdicts from multiple critics combine into the composite's single
/// verdict.
#[derive(Debug, Clone)]
pub enum AggregationPolicy {
    /// Sequential, short-circuits on the first non-accepting verdict.
    /// The surviving verdict wins.
    FirstReject,
    /// Parallel. Every critic must return an accepting verdict; any
    /// non-Accept wins (the first one encountered in child-order).
    AllMustAccept,
    /// Parallel. Accept if more than half of critics accept.
    MajorityVote,
    /// Parallel. Each critic carries a weight (parallel to `critics`);
    /// accept if the sum of accepting critics' weights exceeds half the
    /// total weight. The weight vector length must match the critic
    /// count.
    Weighted(Vec<f32>),
}

impl CompositeCritic {
    pub fn new(name: impl Into<String>, policy: AggregationPolicy) -> Self {
        Self {
            name: name.into(),
            critics: Vec::new(),
            policy,
        }
    }

    pub fn push(mut self, critic: Box<dyn Critic>) -> Self {
        self.critics.push(critic);
        self
    }

    pub fn len(&self) -> usize {
        self.critics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.critics.is_empty()
    }
}

#[async_trait]
impl Critic for CompositeCritic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        match &self.policy {
            AggregationPolicy::FirstReject => {
                // Sequential — short-circuit on the first non-Accept.
                for critic in &self.critics {
                    let v = critic.assess(ctx).await;
                    if !v.is_accepting() {
                        return v;
                    }
                }
                CriticVerdict::Accept
            }
            AggregationPolicy::AllMustAccept => {
                let verdicts = run_parallel(&self.critics, ctx).await;
                verdicts
                    .into_iter()
                    .find(|v| !v.is_accepting())
                    .unwrap_or(CriticVerdict::Accept)
            }
            AggregationPolicy::MajorityVote => {
                let verdicts = run_parallel(&self.critics, ctx).await;
                let accepting = verdicts.iter().filter(|v| v.is_accepting()).count();
                if accepting * 2 > verdicts.len() {
                    CriticVerdict::Accept
                } else {
                    // Surface the first non-accepting verdict as the
                    // composite's reason; fall back to a synthetic one
                    // if every verdict happened to accept (shouldn't
                    // happen given the branch condition).
                    verdicts.into_iter().find(|v| !v.is_accepting()).unwrap_or(
                        CriticVerdict::Reject {
                            reason: format!(
                                "majority vote: {accepting}/{} accepted",
                                self.critics.len()
                            ),
                        },
                    )
                }
            }
            AggregationPolicy::Weighted(weights) => {
                assert_eq!(
                    weights.len(),
                    self.critics.len(),
                    "Weighted aggregation: weights vector length must equal critic count \
                     (got {} weights for {} critics)",
                    weights.len(),
                    self.critics.len()
                );
                let verdicts = run_parallel(&self.critics, ctx).await;
                let total: f32 = weights.iter().sum();
                let accepting_weight: f32 = weights
                    .iter()
                    .zip(verdicts.iter())
                    .filter(|(_, v)| v.is_accepting())
                    .map(|(w, _)| *w)
                    .sum();
                if accepting_weight * 2.0 > total {
                    CriticVerdict::Accept
                } else {
                    verdicts.into_iter().find(|v| !v.is_accepting()).unwrap_or(
                        CriticVerdict::Reject {
                            reason: format!(
                                "weighted vote: {accepting_weight:.2}/{total:.2} accepting"
                            ),
                        },
                    )
                }
            }
        }
    }
}

async fn run_parallel(
    critics: &[Box<dyn Critic>],
    ctx: &AssessmentContext<'_>,
) -> Vec<CriticVerdict> {
    // Each critic borrows ctx immutably for the duration of its
    // `assess` future, so we can simply collect futures here.
    let futs = critics.iter().map(|c| c.assess(ctx));
    join_all(futs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::{
        AssistantTurn, ConversationView, Message, StopReason, Task, TrajectoryView, Usage,
    };

    fn ctx_with<'a>(turn: &'a AssistantTurn, task: &'a Task) -> AssessmentContext<'a> {
        AssessmentContext::new(
            task,
            ConversationView::new(&[]),
            turn,
            TrajectoryView::new(&[]),
        )
    }

    fn sample_turn() -> AssistantTurn {
        AssistantTurn::new(
            0,
            "span-0",
            Message::assistant_text("hello"),
            Usage::default(),
            StopReason::EndTurn,
        )
    }

    // Stub critics for the aggregation tests.
    struct Always(CriticVerdict);
    #[async_trait]
    impl Critic for Always {
        fn name(&self) -> &str {
            "always"
        }
        async fn assess(&self, _: &AssessmentContext<'_>) -> CriticVerdict {
            self.0.clone()
        }
    }

    #[tokio::test]
    async fn first_reject_returns_first_rejection() {
        let composite = CompositeCritic::new("c", AggregationPolicy::FirstReject)
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no".into(),
            })))
            // Third critic should never be reached — its rejection would
            // have different text, proving short-circuit.
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no (second)".into(),
            })));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(matches!(
            verdict,
            CriticVerdict::Reject { ref reason } if reason == "no"
        ));
    }

    #[tokio::test]
    async fn all_must_accept_rejects_on_any_non_accept() {
        let composite = CompositeCritic::new("c", AggregationPolicy::AllMustAccept)
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "bad".into(),
            })))
            .push(Box::new(Always(CriticVerdict::Accept)));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(matches!(verdict, CriticVerdict::Reject { .. }));
    }

    #[tokio::test]
    async fn all_must_accept_accepts_when_every_child_accepts() {
        let composite = CompositeCritic::new("c", AggregationPolicy::AllMustAccept)
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::AcceptWithNote("ok".into()))));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_accepting());
    }

    #[tokio::test]
    async fn majority_vote_accepts_when_most_accept() {
        let composite = CompositeCritic::new("c", AggregationPolicy::MajorityVote)
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no".into(),
            })));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_accepting());
    }

    #[tokio::test]
    async fn majority_vote_rejects_when_most_reject() {
        let composite = CompositeCritic::new("c", AggregationPolicy::MajorityVote)
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no1".into(),
            })))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no2".into(),
            })));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_rejecting());
    }

    #[tokio::test]
    async fn weighted_accepts_when_accepting_weight_exceeds_half() {
        let composite = CompositeCritic::new("c", AggregationPolicy::Weighted(vec![0.75, 0.25]))
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "no".into(),
            })));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_accepting());
    }

    #[tokio::test]
    async fn weighted_rejects_when_accepting_weight_does_not_exceed_half() {
        let composite = CompositeCritic::new("c", AggregationPolicy::Weighted(vec![0.25, 0.75]))
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Reject {
                reason: "blocked".into(),
            })));
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_rejecting());
    }

    #[tokio::test]
    async fn empty_composite_accepts_by_default() {
        let composite = CompositeCritic::new("c", AggregationPolicy::FirstReject);
        let task = Task::new("t");
        let turn = sample_turn();
        let verdict = composite.assess(&ctx_with(&turn, &task)).await;
        assert!(verdict.is_accepting());
    }

    #[tokio::test]
    #[should_panic(expected = "Weighted aggregation")]
    async fn weighted_mismatched_weights_panics() {
        let composite = CompositeCritic::new("c", AggregationPolicy::Weighted(vec![1.0]))
            .push(Box::new(Always(CriticVerdict::Accept)))
            .push(Box::new(Always(CriticVerdict::Accept)));
        let task = Task::new("t");
        let turn = sample_turn();
        let _ = composite.assess(&ctx_with(&turn, &task)).await;
    }
}
