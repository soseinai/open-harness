//! `TimeBudget` — hard cap on wall-clock time since construction
//! (plan §10.2).
//!
//! Construction records an `Instant`; `check` rejects when `now - start` has
//! exceeded the cap. `consume`'s `wall_clock` field is *not* load-bearing for
//! this budget (it's derived from a single monotonic start), but is still
//! accumulated into the snapshot for observability.

use async_trait::async_trait;
use oharness_core::{BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, BudgetSnapshot};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct TimeBudget {
    cap: Duration,
    started_at: Instant,
    consumed_non_clock: Mutex<BudgetAmount>,
}

impl TimeBudget {
    pub fn wall_clock(cap: Duration) -> Self {
        Self {
            cap,
            started_at: Instant::now(),
            consumed_non_clock: Mutex::new(BudgetAmount::default()),
        }
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

#[async_trait]
impl BudgetHandle for TimeBudget {
    async fn check(&self, _request: BudgetRequest) -> BudgetDecision {
        let elapsed = self.elapsed();
        if elapsed >= self.cap {
            BudgetDecision::Deny {
                reason: format!("time budget: elapsed {elapsed:?} >= cap {:?}", self.cap),
            }
        } else {
            BudgetDecision::Allow
        }
    }

    async fn consume(&self, amount: BudgetAmount) {
        let mut c = self.consumed_non_clock.lock().expect("time budget mutex");
        c.tokens_input = c.tokens_input.saturating_add(amount.tokens_input);
        c.tokens_output = c.tokens_output.saturating_add(amount.tokens_output);
        c.cost_usd += amount.cost_usd;
        c.steps = c.steps.saturating_add(amount.steps);
        // wall_clock is handled by the monotonic start; ignore the passed value.
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let mut consumed = self
            .consumed_non_clock
            .lock()
            .expect("time budget mutex")
            .clone();
        consumed.wall_clock = self.elapsed();
        let remaining = BudgetAmount {
            wall_clock: self.cap.saturating_sub(consumed.wall_clock),
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
    async fn check_allows_when_under_cap() {
        let b = TimeBudget::wall_clock(Duration::from_secs(3600));
        assert!(matches!(
            b.check(BudgetRequest::default()).await,
            BudgetDecision::Allow
        ));
    }

    #[tokio::test]
    async fn check_denies_when_over_cap() {
        // Nanosecond cap guarantees elapsed >= cap on any modern machine.
        let b = TimeBudget::wall_clock(Duration::from_nanos(1));
        // Force a tiny sleep so elapsed advances past the cap deterministically.
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(matches!(
            b.check(BudgetRequest::default()).await,
            BudgetDecision::Deny { .. }
        ));
    }
}
