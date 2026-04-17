//! Critics and reflectors for open-harness (plan §11).
//!
//! A [`Critic`] inspects a just-completed assistant turn and returns a
//! [`CriticVerdict`] — accept, reject, revise, or abort. The loop is
//! responsible for invoking critics and acting on their verdicts; this
//! crate only ships the trait surface and the shipped implementations.
//!
//! A [`Reflector`] inspects a finished episode (task + outcome +
//! evaluation) and optionally emits a [`oharness_core::Reflection`] for
//! the next iteration of a `run_reflexion` loop. Reflections are threaded
//! back into subsequent calls via the [`ReflectionInjector`] middleware,
//! which is a `RequestLayer` that prepends reflections as a system-prompt
//! suffix or first-user-message prefix.
//!
//! Shipped impls are behind feature flags so the trait crate itself stays
//! dep-free:
//!
//! | impl | feature |
//! |------|---------|
//! | [`shipped::NullReflector`] | — (always available) |
//! | [`shipped::LlmReflector`] | — |
//! | [`shipped::RegexDenyCritic`] | `regex-deny` |
//! | [`shipped::TestCritic`] | `test-runner` |
//! | [`shipped::LlmJudgeCritic`] | `llm-judge` |
//!
//! `ConstitutionalCritic` is deliberately deferred — its principle-based
//! revision flow has more configuration surface than M2 needs right now.

pub mod composite;
pub mod critic;
pub mod injector;
pub mod reflector;
pub mod shipped;

pub use composite::{AggregationPolicy, CompositeCritic};
pub use critic::{AssessmentContext, Critic, CriticTrigger, CriticVerdict};
pub use injector::{ReflectionInjector, ReflectionPlacement};
pub use reflector::Reflector;
