# oharness-budget

Budget middleware (token, step, cost, time) for
[open-harness](https://github.com/aishfenton/open-harness).

`BudgetMiddleware` is a plain `Llm` wrapper: wrap any provider,
get pre-call + post-call accounting for free. When the cap trips,
the underlying call returns
`LlmError::Provider(BudgetExceeded)`, which the agent loop
converts to `Termination::Failed { category: Llm }`.

## Shipped budget handles

| Handle             | Feature       | Default | What it caps                                                      |
|--------------------|---------------|---------|-------------------------------------------------------------------|
| `TokenBudget`      | `token`       | ✅      | Input + output tokens (configurable: input-only, output-only, or both). |
| `StepBudget`       | `step`        | ✅      | Number of `complete()` calls.                                     |
| `CostBudget`       | `cost`        |         | Dollars, via a `PricingTable`.                                    |
| `TimeBudget`       | `wall-clock`  |         | Wall-clock duration across the run.                               |
| `CompositeBudget`  | (always)      | ✅      | Combine multiple handles; any child trip trips the composite.     |

All implement the `BudgetHandle` trait.

## Quickstart

```rust
use oharness_budget::{BudgetMiddleware, TokenBudget};
use oharness_core::BudgetHandle;
use std::sync::Arc;

let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(100_000));
let bounded = Arc::new(BudgetMiddleware::new(my_llm, budget.clone()));
// Pass `bounded` to `AgentBuilder::with_llm(...)` as you would any Llm.
```

See the `budget_enforcement` example in `oharness-loop/examples/`.

## License

Dual-licensed under MIT or Apache-2.0.
