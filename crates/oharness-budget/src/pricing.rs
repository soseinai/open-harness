//! Per-model pricing table used by `CostBudget` and the cost component of
//! `BudgetAmount` (plan §10.4).
//!
//! Prices are per million tokens. Callers populate the table at runtime
//! (`load_from`, `override_model`) so adding a new model's pricing does not
//! require a library bump — this is a hard constraint for M4.

use oharness_core::{ModelId, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    #[serde(default)]
    pub cache_read_per_million: f64,
    #[serde(default)]
    pub cache_write_per_million: f64,
}

impl ModelPricing {
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }
    }

    pub fn with_cache(mut self, read_per_million: f64, write_per_million: f64) -> Self {
        self.cache_read_per_million = read_per_million;
        self.cache_write_per_million = write_per_million;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct PricingTable {
    models: HashMap<ModelId, ModelPricing>,
}

#[derive(Debug, thiserror::Error)]
pub enum PricingLoadError {
    #[error("pricing table read error: {0}")]
    Io(#[from] std::io::Error),
    #[error("pricing table parse error: {0}")]
    Parse(#[from] serde_json::Error),
}

impl PricingTable {
    pub fn empty() -> Self {
        Self::default()
    }

    /// A small, opinionated default. Callers should treat this as a *starting
    /// point* — published Anthropic pricing changes; use `load_from` or
    /// `override_model` to keep it current without a library bump. Numbers
    /// below reflect published list pricing as of the crate's ship date and
    /// are USD per million tokens.
    pub fn builtin() -> Self {
        let mut t = Self::default();
        t.override_model(
            ModelId::new("claude-haiku-4-5"),
            ModelPricing::new(1.0, 5.0).with_cache(0.1, 1.25),
        );
        t.override_model(
            ModelId::new("claude-sonnet-4-5"),
            ModelPricing::new(3.0, 15.0).with_cache(0.3, 3.75),
        );
        t.override_model(
            ModelId::new("claude-opus-4"),
            ModelPricing::new(15.0, 75.0).with_cache(1.5, 18.75),
        );
        t.override_model(
            ModelId::new("claude-opus-4-7"),
            ModelPricing::new(15.0, 75.0).with_cache(1.5, 18.75),
        );
        t
    }

    pub fn load_from(path: &Path) -> Result<Self, PricingLoadError> {
        let bytes = std::fs::read(path)?;
        let models: HashMap<String, ModelPricing> = serde_json::from_slice(&bytes)?;
        let models = models
            .into_iter()
            .map(|(k, v)| (ModelId::new(k), v))
            .collect();
        Ok(Self { models })
    }

    pub fn override_model(&mut self, id: ModelId, pricing: ModelPricing) {
        self.models.insert(id, pricing);
    }

    pub fn pricing_for(&self, model: &ModelId) -> Option<&ModelPricing> {
        self.models.get(model)
    }

    /// USD cost of the given `Usage` on the given model. Returns `0.0` if the
    /// model is unknown and emits a single `tracing::warn!` — a silent 0
    /// would be a correctness trap for cost budgets.
    pub fn cost_for(&self, model: &ModelId, usage: &Usage) -> f64 {
        let Some(p) = self.pricing_for(model) else {
            tracing::warn!(
                target: "oharness.budget.pricing",
                model = %model.as_str(),
                "pricing unknown; cost reported as 0",
            );
            return 0.0;
        };
        let per_m = |tokens: u64, rate: f64| (tokens as f64) * rate / 1_000_000.0;
        per_m(usage.tokens_input, p.input_per_million)
            + per_m(usage.tokens_output, p.output_per_million)
            + per_m(usage.tokens_cache_read, p.cache_read_per_million)
            + per_m(usage.tokens_cache_create, p.cache_write_per_million)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_for_known_model() {
        let mut t = PricingTable::empty();
        t.override_model(ModelId::new("m"), ModelPricing::new(2.0, 6.0));
        let usage = Usage {
            tokens_input: 1_000_000,
            tokens_output: 500_000,
            ..Default::default()
        };
        assert!((t.cost_for(&ModelId::new("m"), &usage) - (2.0 + 3.0)).abs() < 1e-9);
    }

    #[test]
    fn cost_for_unknown_model_is_zero() {
        let t = PricingTable::empty();
        let usage = Usage {
            tokens_input: 1_000_000,
            ..Default::default()
        };
        assert_eq!(t.cost_for(&ModelId::new("unknown"), &usage), 0.0);
    }

    #[test]
    fn cache_lines_are_accounted_for() {
        let mut t = PricingTable::empty();
        t.override_model(
            ModelId::new("m"),
            ModelPricing::new(0.0, 0.0).with_cache(0.5, 1.0),
        );
        let usage = Usage {
            tokens_cache_read: 2_000_000,
            tokens_cache_create: 1_000_000,
            ..Default::default()
        };
        assert!((t.cost_for(&ModelId::new("m"), &usage) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn override_model_replaces_entry() {
        let mut t = PricingTable::empty();
        t.override_model(ModelId::new("m"), ModelPricing::new(1.0, 2.0));
        t.override_model(ModelId::new("m"), ModelPricing::new(4.0, 8.0));
        let p = t.pricing_for(&ModelId::new("m")).unwrap();
        assert_eq!(p.input_per_million, 4.0);
        assert_eq!(p.output_per_million, 8.0);
    }

    #[test]
    fn builtin_has_common_anthropic_models() {
        let t = PricingTable::builtin();
        assert!(t.pricing_for(&ModelId::new("claude-sonnet-4-5")).is_some());
        assert!(t.pricing_for(&ModelId::new("claude-haiku-4-5")).is_some());
        assert!(t.pricing_for(&ModelId::new("claude-opus-4")).is_some());
    }
}
