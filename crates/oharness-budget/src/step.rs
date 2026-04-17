//! `StepBudget` — hard cap on the number of completed LLM calls (plan §10.2).

use async_trait::async_trait;
use oharness_core::{BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, BudgetSnapshot};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct StepBudget {
    cap_steps: u32,
    consumed_steps: AtomicU32,
    consumed_other: std::sync::Mutex<BudgetAmount>,
}

impl StepBudget {
    pub fn turns(cap: u32) -> Self {
        Self {
            cap_steps: cap,
            consumed_steps: AtomicU32::new(0),
            consumed_other: std::sync::Mutex::new(BudgetAmount::default()),
        }
    }
}

#[async_trait]
impl BudgetHandle for StepBudget {
    async fn check(&self, _request: BudgetRequest) -> BudgetDecision {
        let consumed = self.consumed_steps.load(Ordering::Relaxed);
        // `check` is called pre-call — deny if consuming one more step
        // would exceed the cap.
        if consumed >= self.cap_steps {
            BudgetDecision::Deny {
                reason: format!(
                    "step budget: {consumed} step(s) used of {} cap",
                    self.cap_steps
                ),
            }
        } else {
            BudgetDecision::Allow
        }
    }

    async fn consume(&self, amount: BudgetAmount) {
        if amount.steps > 0 {
            self.consumed_steps
                .fetch_add(amount.steps, Ordering::Relaxed);
        }
        let mut c = self.consumed_other.lock().expect("step budget mutex");
        c.tokens_input = c.tokens_input.saturating_add(amount.tokens_input);
        c.tokens_output = c.tokens_output.saturating_add(amount.tokens_output);
        c.cost_usd += amount.cost_usd;
        c.wall_clock = c.wall_clock.saturating_add(amount.wall_clock);
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let mut consumed = self
            .consumed_other
            .lock()
            .expect("step budget mutex")
            .clone();
        consumed.steps = self.consumed_steps.load(Ordering::Relaxed);
        let remaining = BudgetAmount {
            steps: self.cap_steps.saturating_sub(consumed.steps),
            ..Default::default()
        };
        BudgetSnapshot {
            consumed,
            remaining: Some(remaining),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_allows_until_cap_reached() {
        let b = StepBudget::turns(2);
        assert!(matches!(
            b.check(BudgetRequest::default()).await,
            BudgetDecision::Allow
        ));
        b.consume(BudgetAmount {
            steps: 1,
            ..Default::default()
        })
        .await;
        assert!(matches!(
            b.check(BudgetRequest::default()).await,
            BudgetDecision::Allow
        ));
        b.consume(BudgetAmount {
            steps: 1,
            ..Default::default()
        })
        .await;
        let d = b.check(BudgetRequest::default()).await;
        assert!(matches!(d, BudgetDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn snapshot_reports_remaining_steps() {
        let b = StepBudget::turns(5);
        b.consume(BudgetAmount {
            steps: 2,
            ..Default::default()
        })
        .await;
        let s = b.snapshot();
        assert_eq!(s.consumed.steps, 2);
        assert_eq!(s.remaining.unwrap().steps, 3);
    }
}
