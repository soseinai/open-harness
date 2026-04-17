//! `Llm` trait, error types, streaming chunks, and the `complete_from_stream` helper.
//!
//! M1a scope: trait + error + chunk + stream helper only. The full middleware helper
//! traits (`RequestLayer`, `ResponseLayer`, `FullLayer`, `ChunkObserver`,
//! `ChunkTransformer`) and `LlmExt` fluent composition land in M1b per §21.1.

pub mod chunk;
pub mod error;
pub mod layer;
pub mod llm;
pub mod stream;

pub use chunk::{BlockStartKind, Chunk};
pub use error::{LayerError, LlmError};
pub use layer::{
    ChunkObserver, ChunkTransformer, FullLayer, InfallibleLlmLayer, LlmExt, LlmLayer, RequestLayer,
    ResponseLayer, ResponseLayerStreamMode, WithChunkObserver, WithChunkTransformer, WithFullLayer,
    WithRequestLayer, WithResponseLayer,
};
pub use llm::Llm;
pub use stream::complete_from_stream;

pub use oharness_core::{
    CompletionRequest, CompletionResponse, LlmCapabilities, ModelId, StopReason, ToolSpec, Usage,
};

/// Boxed stream of chunk results — the public return type of `Llm::stream`.
pub type ChunkStream = futures::stream::BoxStream<'static, Result<Chunk, LlmError>>;
