# Model pricing maintenance

`BudgetMiddleware` uses a `PricingTable` (from
[`oharness-budget`](../crates/oharness-budget)) to compute per-call USD
cost, which then feeds the shared `BudgetHandle`. Published pricing
changes without library bumps; this doc is the contract for how to keep
the table current.

---

## Shape

Pricing is per model, expressed as *USD per million tokens*:

```rust
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}
```

`PricingTable` holds a `HashMap<ModelId, ModelPricing>`. Models not in
the table yield `cost_for(..) == 0.0` and emit a
`tracing::warn!("pricing unknown; cost reported as 0", model)` once per
call — silent undercounting would be a correctness trap, so the warn
is noisy by design.

---

## Built-in defaults

`PricingTable::builtin()` ships starter entries for commonly-used
models. Consult
[`oharness-budget/src/pricing.rs`](../crates/oharness-budget/src/pricing.rs)
for the current list. These are **starting points, not source of
truth** — published list prices change, and the library tracks them on
a best-effort basis only.

---

## Updating pricing without a library bump

Three paths, in order of operational preference.

### 1. Override at runtime via `with_pricing(..)`

The operator's preferred path. Builds a fresh table in-process, no
disk dependency, no restart dance.

```rust
use oharness_budget::{ModelPricing, PricingTable, BudgetMiddleware};
use oharness_core::ModelId;
use std::sync::Arc;

let mut pricing = PricingTable::builtin();
pricing.override_model(
    ModelId::new("claude-sonnet-4-7"),
    ModelPricing::new(3.0, 15.0).with_cache(0.3, 3.75),
);
let middleware = BudgetMiddleware::new(inner_llm, budget.clone())
    .with_pricing(Arc::new(pricing));
```

### 2. Load from a JSON file

For deployments that want prices to be config-managed rather than
code-managed:

```rust
let pricing = PricingTable::load_from(Path::new("/etc/oharness/pricing.json"))?;
```

File shape is a top-level JSON object keyed by model id:

```json
{
  "claude-sonnet-4-5": {
    "input_per_million": 3.0,
    "output_per_million": 15.0,
    "cache_read_per_million": 0.3,
    "cache_write_per_million": 3.75
  },
  "gpt-4o": {
    "input_per_million": 2.5,
    "output_per_million": 10.0,
    "cache_read_per_million": 0.0,
    "cache_write_per_million": 0.0
  }
}
```

`cache_*_per_million` default to `0.0` if omitted, which matches the
typical OpenAI-compatible deployment where Anthropic-style cache
control doesn't apply.

### 3. Pull-request the built-in table

When a price is stable enough to bake into the library (a newly-
released model, a permanent tier discount), open a PR against
[`oharness-budget/src/pricing.rs`](../crates/oharness-budget/src/pricing.rs)
updating `PricingTable::builtin()`. This bumps the library version on
next release; downstream consumers pick it up automatically.

---

## When pricing is unknown

`cost_for(..)` returns `0.0` for unknown models and logs a
`tracing::warn!` once. `CostBudget` treats `0.0` as "under budget";
that's the correct behavior for models we haven't priced (no signal to
deny), but operators who care should either:

- attach a pricing table via option 1 or 2 above, or
- wrap `BudgetMiddleware` with a request/response layer that rejects
  requests targeting unpriced models.

The warn log carries the model id so operators can cross-reference
which models are still unpriced after a deploy.

---

## CI note

Pricing isn't schema-checked (unlike the event schema — see
`crates/oharness-core/testdata/trajectories/v1.0/` and
`crates/oharness-core/tests/v1_compat.rs`). A stale entry in
`PricingTable::builtin()` is a correctness issue surfaced by the
`tracing::warn!` log, not a CI failure.
