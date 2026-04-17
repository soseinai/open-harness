//! `CostBudget` — hard cap on cumulative USD cost (plan §10.2).
//!
//! Cost comes from `BudgetAmount::cost_usd`, which is populated by the
//! middleware via [`crate::amount::amount_from_response`] /
//! [`crate::amount::amount_from_usage`] using a `PricingTable`. Models with
//! no pricing entry contribute `0.0` — the pricing table emits a
//! `tracing::warn!` when that happens so silent under-counting is visible.

use async_trait::async_trait;
use oharness_core::{BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, BudgetSnapshot};
use std::sync::Mutex;

pub struct CostBudget {
    cap_usd: f64,
    consumed: Mutex<BudgetAmount>,
}

impl CostBudget {
    pub fn usd(cap: f64) -> Self {
        Self {
            cap_usd: cap,
            consumed: Mutex::new(BudgetAmount::default()),
        }
    }
}

#[async_trait]
impl BudgetHandle for CostBudget {
    async fn check(&self, request: BudgetRequest) -> BudgetDecision {
        let spent = {
            let c = self.consumed.lock().expect("cost budget mutex");
            c.cost_usd
        };
        let projected = spent + request.estimated_cost_usd.unwrap_or(0.0);
        if projected > self.cap_usd {
            BudgetDecision::Deny {
                reason: format!(
                    "cost budget: projected ${projected:.4} > cap ${:.4}",
                    self.cap_usd
                ),
            }
        } else {
            BudgetDecision::Allow
        }
    }

    async fn consume(&self, amount: BudgetAmount) {
        let mut c = self.consumed.lock().expect("cost budget mutex");
        c.tokens_input = c.tokens_input.saturating_add(amount.tokens_input);
        c.tokens_output = c.tokens_output.saturating_add(amount.tokens_output);
        c.cost_usd += amount.cost_usd;
        c.wall_clock = c.wall_clock.saturating_add(amount.wall_clock);
        c.steps = c.steps.saturating_add(amount.steps);
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let c = self.consumed.lock().expect("cost budget mutex").clone();
        let remaining = BudgetAmount {
            cost_usd: (self.cap_usd - c.cost_usd).max(0.0),
            ..Default::default()
        };
        BudgetSnapshot {
            consumed: c,
            remaining: Some(remaining),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_denies_when_projected_exceeds_cap() {
        let b = CostBudget::usd(1.00);
        b.consume(BudgetAmount {
            cost_usd: 0.90,
            ..Default::default()
        })
        .await;
        let d = b
            .check(BudgetRequest {
                estimated_cost_usd: Some(0.15),
                ..Default::default()
            })
            .await;
        assert!(matches!(d, BudgetDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn check_allows_when_projected_within_cap() {
        let b = CostBudget::usd(1.00);
        b.consume(BudgetAmount {
            cost_usd: 0.50,
            ..Default::default()
        })
        .await;
        assert!(matches!(
            b.check(BudgetRequest {
                estimated_cost_usd: Some(0.20),
                ..Default::default()
            })
            .await,
            BudgetDecision::Allow
        ));
    }

    #[tokio::test]
    async fn snapshot_reports_remaining_dollars() {
        let b = CostBudget::usd(5.0);
        b.consume(BudgetAmount {
            cost_usd: 3.25,
            ..Default::default()
        })
        .await;
        let s = b.snapshot();
        assert!((s.consumed.cost_usd - 3.25).abs() < 1e-9);
        assert!((s.remaining.unwrap().cost_usd - 1.75).abs() < 1e-9);
    }
}
