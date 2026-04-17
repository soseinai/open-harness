//! Agent + `Loop` trait + `ReactLoop`.
//!
//! M1a: `ReactLoop` emits events directly via its `ScopedEmitter` — no middleware
//! tracing yet (that replaces this in M1b per §20.3).

pub mod agent;
pub mod config;
pub mod loop_trait;

#[cfg(feature = "react")]
pub mod react;

pub use agent::{Agent, AgentBuilder};
pub use config::AgentConfig;
pub use loop_trait::{Loop, LoopContext};

#[cfg(feature = "react")]
pub use react::ReactLoop;

pub use oharness_core::AgentError;
