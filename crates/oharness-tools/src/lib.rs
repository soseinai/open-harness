//! `ToolSet` trait and bundled tool kits.

pub mod context;
pub mod toolset;

#[cfg(feature = "bash")]
pub mod bash;
#[cfg(feature = "fs")]
pub mod fs;

pub use context::{ToolContext, Workspace};
pub use toolset::{ToolError, ToolOutcome, ToolSet};

pub use oharness_core::ToolSpec;
pub use oharness_core::message::ToolOutput;
