//! Shipped [`Critic`] and [`Reflector`] implementations.
//!
//! All impls behind cargo features — the trait crate stays dep-free by
//! default. See the module-level docs on each type for the feature flag.

// NullReflector and LlmReflector don't need extra deps beyond oharness-llm,
// which is already a required dep of this crate. They're always available.
mod null_reflector;
pub use null_reflector::NullReflector;

mod llm_reflector;
pub use llm_reflector::LlmReflector;

#[cfg(feature = "regex-deny")]
mod regex_deny;
#[cfg(feature = "regex-deny")]
pub use regex_deny::RegexDenyCritic;

#[cfg(feature = "test-runner")]
mod test_critic;
#[cfg(feature = "test-runner")]
pub use test_critic::TestCritic;

#[cfg(feature = "llm-judge")]
mod llm_judge;
#[cfg(feature = "llm-judge")]
pub use llm_judge::LlmJudgeCritic;
