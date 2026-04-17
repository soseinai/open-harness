//! Pluggable memory/context strategies.
//!
//! M1a ships: `Passthrough`, `TruncateAfterTokens`, `ElideToolResults`.
//! `Summarize`, `HierarchicalSummary`, `Rag` land in M1b+.

pub mod policy;

#[cfg(feature = "elide")]
pub mod elide;
#[cfg(feature = "passthrough")]
pub mod passthrough;
#[cfg(feature = "truncate")]
pub mod truncate;

pub use policy::{MemoryContext, MemoryError, MemoryPolicy};

#[cfg(feature = "elide")]
pub use elide::ElideToolResults;
#[cfg(feature = "passthrough")]
pub use passthrough::Passthrough;
#[cfg(feature = "truncate")]
pub use truncate::TruncateAfterTokens;
