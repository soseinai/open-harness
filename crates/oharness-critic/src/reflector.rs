//! [`Reflector`] trait (plan §11.4).
//!
//! A reflector runs once per episode in a `run_reflexion` sweep. It receives
//! a borrowed [`oharness_core::Episode`] and optionally emits a
//! [`oharness_core::Reflection`] that the next episode will see via the
//! [`crate::ReflectionInjector`] middleware. Always invoked — the reflector
//! itself decides whether to produce a reflection (hence the `Option`
//! return type).

use async_trait::async_trait;
use oharness_core::{Episode, Reflection};

#[async_trait]
pub trait Reflector: Send + Sync {
    /// Short, stable identifier — e.g. `"null"`, `"llm"`.
    fn name(&self) -> &str;

    /// Inspect the finished episode and optionally produce a reflection.
    /// Returning `None` skips reflexion injection for the next iteration.
    async fn reflect(&self, episode: &Episode<'_>) -> Option<Reflection>;
}
