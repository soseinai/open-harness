//! `TokenBudget` — hard cap on the sum of input + output tokens consumed
//! over the lifetime of the handle (plan §10.2).

use async_trait::async_trait;
use oharness_core::{BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, BudgetSnapshot};
use std::sync::Mutex;

pub struct TokenBudget {
    cap_input_plus_output: u64,
    consumed: Mutex<BudgetAmount>,
}

impl TokenBudget {
    /// Hard cap on `tokens_input + tokens_output`.
    pub fn input_plus_output(cap: u64) -> Self {
        Self {
            cap_input_plus_output: cap,
            consumed: Mutex::new(BudgetAmount::default()),
        }
    }

    fn total(&self) -> u64 {
        let c = self.consumed.lock().expect("token budget mutex");
        c.tokens_input + c.tokens_output
    }
}

#[async_trait]
impl BudgetHandle for TokenBudget {
    async fn check(&self, request: BudgetRequest) -> BudgetDecision {
        let projected = self.total()
            + request.estimated_input_tokens.unwrap_or(0)
            + request.estimated_output_tokens.unwrap_or(0);
        if projected > self.cap_input_plus_output {
            BudgetDecision::Deny {
                reason: format!(
                    "token budget: projected {projected} > cap {}",
                    self.cap_input_plus_output
                ),
            }
        } else {
            BudgetDecision::Allow
        }
    }

    async fn consume(&self, amount: BudgetAmount) {
        let mut c = self.consumed.lock().expect("token budget mutex");
        c.tokens_input = c.tokens_input.saturating_add(amount.tokens_input);
        c.tokens_output = c.tokens_output.saturating_add(amount.tokens_output);
        c.cost_usd += amount.cost_usd;
        c.wall_clock = c.wall_clock.saturating_add(amount.wall_clock);
        c.steps = c.steps.saturating_add(amount.steps);
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let c = self.consumed.lock().expect("token budget mutex").clone();
        BudgetSnapshot {
            consumed: c,
            // `BudgetAmount::remaining` has no clean mapping for a combined
            // input+output cap — omit rather than lie.
            remaining: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_allows_under_cap() {
        let b = TokenBudget::input_plus_output(1000);
        assert!(matches!(
            b.check(BudgetRequest {
                estimated_input_tokens: Some(400),
                estimated_output_tokens: Some(400),
                ..Default::default()
            })
            .await,
            BudgetDecision::Allow
        ));
    }

    #[tokio::test]
    async fn check_denies_over_cap() {
        let b = TokenBudget::input_plus_output(1000);
        b.consume(BudgetAmount {
            tokens_input: 900,
            ..Default::default()
        })
        .await;
        let d = b
            .check(BudgetRequest {
                estimated_input_tokens: Some(50),
                estimated_output_tokens: Some(100),
                ..Default::default()
            })
            .await;
        assert!(matches!(d, BudgetDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn consume_accumulates() {
        let b = TokenBudget::input_plus_output(10_000);
        b.consume(BudgetAmount {
            tokens_input: 100,
            tokens_output: 50,
            ..Default::default()
        })
        .await;
        b.consume(BudgetAmount {
            tokens_input: 25,
            tokens_output: 75,
            ..Default::default()
        })
        .await;
        let s = b.snapshot();
        assert_eq!(s.consumed.tokens_input, 125);
        assert_eq!(s.consumed.tokens_output, 125);
    }
}
