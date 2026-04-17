//! `CompositeBudget` — fans `check` / `consume` / `snapshot` across any
//! number of inner `BudgetHandle`s. Any single child returning `Deny`
//! denies the whole composite (plan §10.2).

use async_trait::async_trait;
use oharness_core::{BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, BudgetSnapshot};
use std::sync::Arc;

pub struct CompositeBudget {
    children: Vec<Arc<dyn BudgetHandle>>,
}

impl CompositeBudget {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn push(mut self, child: Arc<dyn BudgetHandle>) -> Self {
        self.children.push(child);
        self
    }

    pub fn from_children(children: Vec<Arc<dyn BudgetHandle>>) -> Self {
        Self { children }
    }
}

impl Default for CompositeBudget {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BudgetHandle for CompositeBudget {
    async fn check(&self, request: BudgetRequest) -> BudgetDecision {
        for child in &self.children {
            // Clone per child — each may annotate differently.
            if let BudgetDecision::Deny { reason } = child.check(request.clone()).await {
                return BudgetDecision::Deny { reason };
            }
        }
        BudgetDecision::Allow
    }

    async fn consume(&self, amount: BudgetAmount) {
        for child in &self.children {
            child.consume(amount.clone()).await;
        }
    }

    fn snapshot(&self) -> BudgetSnapshot {
        // Summed consumed; remaining = None because combining remainders from
        // heterogeneous budgets is not meaningful.
        let mut combined = BudgetAmount::default();
        for child in &self.children {
            let s = child.snapshot();
            combined.tokens_input = combined
                .tokens_input
                .saturating_add(s.consumed.tokens_input);
            combined.tokens_output = combined
                .tokens_output
                .saturating_add(s.consumed.tokens_output);
            combined.cost_usd += s.consumed.cost_usd;
            combined.wall_clock = combined.wall_clock.saturating_add(s.consumed.wall_clock);
            combined.steps = combined.steps.saturating_add(s.consumed.steps);
        }
        BudgetSnapshot {
            consumed: combined,
            remaining: None,
        }
    }
}

#[cfg(all(test, feature = "token", feature = "step"))]
mod tests {
    use super::*;
    use crate::StepBudget;
    use crate::TokenBudget;

    #[tokio::test]
    async fn denies_if_any_child_denies() {
        let tokens: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(100));
        let steps: Arc<dyn BudgetHandle> = Arc::new(StepBudget::turns(1));
        let composite = CompositeBudget::new()
            .push(tokens.clone())
            .push(steps.clone());

        // Step budget trips first after one step.
        steps
            .consume(BudgetAmount {
                steps: 1,
                ..Default::default()
            })
            .await;
        assert!(matches!(
            composite.check(BudgetRequest::default()).await,
            BudgetDecision::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn allows_when_every_child_allows() {
        let tokens: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(10_000));
        let steps: Arc<dyn BudgetHandle> = Arc::new(StepBudget::turns(10));
        let composite = CompositeBudget::new().push(tokens).push(steps);
        assert!(matches!(
            composite.check(BudgetRequest::default()).await,
            BudgetDecision::Allow
        ));
    }

    #[tokio::test]
    async fn consume_fans_out() {
        let tokens: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(10_000));
        let steps: Arc<dyn BudgetHandle> = Arc::new(StepBudget::turns(10));
        let composite = CompositeBudget::new()
            .push(tokens.clone())
            .push(steps.clone());
        composite
            .consume(BudgetAmount {
                tokens_input: 50,
                tokens_output: 50,
                steps: 1,
                ..Default::default()
            })
            .await;
        assert_eq!(tokens.snapshot().consumed.tokens_input, 50);
        assert_eq!(steps.snapshot().consumed.steps, 1);
    }
}
