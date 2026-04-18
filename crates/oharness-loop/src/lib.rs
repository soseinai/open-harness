//! Agent + `Loop` trait + shipped loops.
//!
//! Shipped loops are feature-gated:
//! - `react` (default): [`ReactLoop`] — thought → action → observation.
//! - `conversation`: [`ConversationLoop`] — alternates agent + user simulator.
//! - `reflexion`: [`run_reflexion`] helper (not a `Loop` impl).

pub mod agent;
pub mod config;
pub mod loop_trait;
pub mod user_simulator;

#[cfg(feature = "react")]
pub mod react;

#[cfg(feature = "conversation")]
pub mod conversation;

#[cfg(feature = "reflexion")]
pub mod reflexion;

pub use agent::{Agent, AgentBuilder};
pub use config::AgentConfig;
pub use loop_trait::{Loop, LoopContext};
pub use user_simulator::{
    LlmUserSimulator, ScriptedUserSimulator, UserAction, UserError, UserSimulator,
};

#[cfg(feature = "react")]
pub use react::ReactLoop;

#[cfg(feature = "conversation")]
pub use conversation::ConversationLoop;

#[cfg(feature = "reflexion")]
pub use reflexion::run_reflexion;

pub use oharness_core::AgentError;
