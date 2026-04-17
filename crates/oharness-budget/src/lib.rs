//! Budget middleware and shipped [`BudgetHandle`] implementations (plan §10).
//!
//! The [`BudgetHandle`] trait itself lives in `oharness-core`; this crate
//! provides the concrete implementations (`TokenBudget`, `StepBudget`,
//! `CostBudget`, `TimeBudget`, `CompositeBudget`), a [`PricingTable`] for
//! cost calculations, and [`BudgetMiddleware`] — an `Llm` wrapper that
//! pre-checks, post-consumes on `complete()`, and observes `Chunk::Usage`
//! events on `stream()`.
//!
//! `BudgetMiddleware` implements [`oharness_llm::Llm`] directly (the escape
//! hatch per plan §5.6.2) rather than via a helper-trait wrapper, because it
//! needs to thread the same budget counter through all three hook sites.
//!
//! Feature flags per plan §16.1: `token` and `step` are on by default;
//! `cost` and `wall-clock` opt-in.
//!
//! [`BudgetHandle`]: oharness_core::BudgetHandle

pub mod amount;
pub mod composite;
pub mod error;
pub mod middleware;
pub mod pricing;

#[cfg(feature = "cost")]
pub mod cost;
#[cfg(feature = "step")]
pub mod step;
#[cfg(feature = "wall-clock")]
pub mod time;
#[cfg(feature = "token")]
pub mod token;

pub use amount::{amount_from_response, amount_from_usage, budget_request_from};
pub use composite::CompositeBudget;
pub use error::BudgetExceeded;
pub use middleware::BudgetMiddleware;
pub use pricing::{ModelPricing, PricingTable};

#[cfg(feature = "cost")]
pub use cost::CostBudget;
#[cfg(feature = "step")]
pub use step::StepBudget;
#[cfg(feature = "wall-clock")]
pub use time::TimeBudget;
#[cfg(feature = "token")]
pub use token::TokenBudget;
