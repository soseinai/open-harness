//! `EventSink` implementations, trajectory writer/reader.
//!
//! M1a scope: `FileSink` (JSONL), `InMemorySink`, `FanOutSink`. `NullSink` is
//! re-exported from `oharness-core` since it's trivial. Tracing middleware and
//! `ReplayLlm` land in M1b.

pub mod fanout;
pub mod file_sink;
pub mod in_memory;
pub mod reader;
pub mod tracer;

pub use fanout::FanOutSink;
pub use file_sink::FileSink;
pub use in_memory::InMemorySink;
pub use reader::{read_events, read_events_streaming};
pub use tracer::{RequestTracer, StreamTracer, ToolTracer, TOOL_USE_ID_KEY};

pub use oharness_core::NullSink;
