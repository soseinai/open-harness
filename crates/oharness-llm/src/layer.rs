//! Middleware helper traits and wrapper types (§5.5, §5.6).
//!
//! Five helper traits cover the common middleware shapes:
//!
//! - [`RequestLayer`]: mutate the outgoing request.
//! - [`ResponseLayer`]: mutate the `CompletionResponse` returned by `complete()`.
//!   Streaming behaviour is configurable via [`ResponseLayerStreamMode`].
//! - [`FullLayer`]: wrap an entire `complete()` or `stream()` call. Two methods
//!   rather than a generic `around<T>` — `complete` and `stream` have
//!   different retry semantics (plan §5.5).
//! - [`ChunkObserver`]: observe every chunk yielded from `stream()`.
//! - [`ChunkTransformer`]: transform (or drop) every chunk yielded from
//!   `stream()`.
//!
//! Each helper trait has a corresponding wrapper (`WithRequestLayer`, …) that
//! implements [`Llm`]. Users compose via [`LlmExt`] convenience methods
//! (`with_request_layer`, `with_response_layer`, …) or — for bespoke layer
//! types like rate-limiters or prompt caching — via [`LlmExt::with_layer`] /
//! [`LlmExt::try_with_layer`] together with a manual [`LlmLayer`] impl.
//!
//! Helper traits are deliberately distinct so a single layer type can
//! implement several roles; composition is one-role-at-a-time, which also
//! dodges blanket-impl conflicts between `RequestLayer → LlmLayer` and
//! `FullLayer → LlmLayer`.

use crate::chunk::Chunk;
use crate::error::{LayerError, LlmError};
use crate::llm::Llm;
use crate::ChunkStream;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::StreamExt;
use oharness_core::{CompletionRequest, CompletionResponse, LlmCapabilities};
use std::sync::atomic::{AtomicBool, Ordering};

// ======================================================================
// Helper traits
// ======================================================================

pub trait RequestLayer: Send + Sync {
    fn on_request(&self, req: &mut CompletionRequest);
}

/// A shared layer handle forwards to the inner type. This lets users
/// share one layer instance between the LLM middleware stack and an
/// external holder — canonically, `ReflectionInjector` between the
/// request-layer chain and `Agent::with_reflection_injector`. Without
/// this blanket impl, `with_request_layer(Arc::clone(&injector))`
/// wouldn't compile.
impl<T: RequestLayer + ?Sized> RequestLayer for std::sync::Arc<T> {
    fn on_request(&self, req: &mut CompletionRequest) {
        (**self).on_request(req)
    }
}

/// Mutate `CompletionResponse` values returned by `complete()`. Streaming
/// behaviour is configured via [`ResponseLayer::stream_mode`].
pub trait ResponseLayer: Send + Sync {
    fn on_response(&self, res: &mut CompletionResponse);

    /// How this layer behaves when wrapped around `stream()`.
    ///
    /// Default: [`ResponseLayerStreamMode::WarnAndSkip`] — an audible default
    /// so layers like redaction don't silently fail on streams. Layers that
    /// genuinely operate on streams implement [`ChunkTransformer`] instead;
    /// layers that must reject streaming override to
    /// [`ResponseLayerStreamMode::Error`].
    fn stream_mode(&self) -> ResponseLayerStreamMode {
        ResponseLayerStreamMode::WarnAndSkip
    }

    /// Short identifier used in the warning log for `WarnAndSkip`. Override
    /// to a more informative string when helpful.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Symmetric to the `RequestLayer` impl for `Arc<T>`. Forwards all
/// three methods so shared response layers (e.g., a redaction layer
/// held by a supervisor process) work seamlessly with
/// `LlmExt::with_response_layer(shared.clone())`.
impl<T: ResponseLayer + ?Sized> ResponseLayer for std::sync::Arc<T> {
    fn on_response(&self, res: &mut CompletionResponse) {
        (**self).on_response(res)
    }
    fn stream_mode(&self) -> ResponseLayerStreamMode {
        (**self).stream_mode()
    }
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseLayerStreamMode {
    /// Log a `tracing::warn!("policy.layer_skipped_on_stream", layer = ..)`
    /// once per wrapper (and per run once M1b-δ wires an event sink), then
    /// pass chunks through unchanged.
    WarnAndSkip,
    /// Return `LlmError::Unsupported("response_layer_on_stream")` from
    /// `stream()`. Use when the layer's invariants cannot be satisfied by
    /// streaming (plan §5.6.1).
    Error,
    /// Explicitly acknowledge the layer is a no-op on streams. Rare — prefer
    /// `ChunkObserver` for streaming-side observation.
    SilentSkip,
}

/// Wrap an entire `complete()` / `stream()` call. Two methods (not one
/// generic `around<T>`) because the two modes have different retry
/// semantics: retrying `complete()` re-issues the call, retrying `stream()`
/// opens a fresh HTTP connection and re-subscribes (plan §5.5).
#[async_trait]
pub trait FullLayer: Send + Sync {
    async fn around_complete<'a>(
        &'a self,
        _req: CompletionRequest,
        call: BoxFuture<'a, Result<CompletionResponse, LlmError>>,
    ) -> Result<CompletionResponse, LlmError> {
        call.await
    }

    async fn around_stream<'a>(
        &'a self,
        _req: CompletionRequest,
        call: BoxFuture<'a, Result<ChunkStream, LlmError>>,
    ) -> Result<ChunkStream, LlmError> {
        call.await
    }
}

/// Observe every chunk yielded from `stream()`. Observers cannot mutate or
/// drop chunks — for that, use [`ChunkTransformer`].
pub trait ChunkObserver: Send + Sync {
    fn on_chunk(&self, chunk: &Chunk);
}

/// Transform every chunk yielded from `stream()`. Returning `None` drops the
/// chunk entirely.
pub trait ChunkTransformer: Send + Sync {
    fn on_chunk(&self, chunk: Chunk) -> Option<Chunk>;
}

// ======================================================================
// Wrapper types
// ======================================================================

macro_rules! default_capabilities {
    ($inner:expr) => {
        $inner.capabilities()
    };
}

pub struct WithRequestLayer<L: Llm, R: RequestLayer> {
    inner: L,
    layer: R,
}

impl<L: Llm, R: RequestLayer> WithRequestLayer<L, R> {
    pub fn new(inner: L, layer: R) -> Self {
        Self { inner, layer }
    }
}

#[async_trait]
impl<L: Llm, R: RequestLayer> Llm for WithRequestLayer<L, R> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        default_capabilities!(self.inner)
    }

    async fn complete(&self, mut req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.layer.on_request(&mut req);
        self.inner.complete(req).await
    }

    async fn stream(&self, mut req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        self.layer.on_request(&mut req);
        self.inner.stream(req).await
    }
}

pub struct WithResponseLayer<L: Llm, R: ResponseLayer> {
    inner: L,
    layer: R,
    /// Emit the `WarnAndSkip` notice only once per wrapper, not per chunk.
    warned: AtomicBool,
}

impl<L: Llm, R: ResponseLayer> WithResponseLayer<L, R> {
    pub fn new(inner: L, layer: R) -> Self {
        Self {
            inner,
            layer,
            warned: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl<L: Llm, R: ResponseLayer> Llm for WithResponseLayer<L, R> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        default_capabilities!(self.inner)
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut res = self.inner.complete(req).await?;
        self.layer.on_response(&mut res);
        Ok(res)
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        match self.layer.stream_mode() {
            ResponseLayerStreamMode::WarnAndSkip => {
                // TODO(M1b-δ): once a tracing middleware passes an event sink
                // through, emit `policy.layer_skipped_on_stream` as a real
                // event. Until then this is a `tracing::warn!` only.
                if !self.warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        target: "oharness.policy",
                        event = "layer_skipped_on_stream",
                        layer = %self.layer.name(),
                        "response layer skipped on streaming call",
                    );
                }
                self.inner.stream(req).await
            }
            ResponseLayerStreamMode::SilentSkip => self.inner.stream(req).await,
            ResponseLayerStreamMode::Error => {
                Err(LlmError::Unsupported("response_layer_on_stream"))
            }
        }
    }
}

pub struct WithFullLayer<L: Llm, F: FullLayer> {
    inner: L,
    layer: F,
}

impl<L: Llm, F: FullLayer> WithFullLayer<L, F> {
    pub fn new(inner: L, layer: F) -> Self {
        Self { inner, layer }
    }
}

#[async_trait]
impl<L: Llm, F: FullLayer> Llm for WithFullLayer<L, F> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        default_capabilities!(self.inner)
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let call_req = req.clone();
        let call: BoxFuture<'_, _> = Box::pin(self.inner.complete(call_req));
        self.layer.around_complete(req, call).await
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let call_req = req.clone();
        let call: BoxFuture<'_, _> = Box::pin(self.inner.stream(call_req));
        self.layer.around_stream(req, call).await
    }
}

pub struct WithChunkObserver<L: Llm, O: ChunkObserver + 'static> {
    inner: L,
    observer: std::sync::Arc<O>,
}

impl<L: Llm, O: ChunkObserver + 'static> WithChunkObserver<L, O> {
    pub fn new(inner: L, observer: O) -> Self {
        Self {
            inner,
            observer: std::sync::Arc::new(observer),
        }
    }
}

#[async_trait]
impl<L: Llm, O: ChunkObserver + 'static> Llm for WithChunkObserver<L, O> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        default_capabilities!(self.inner)
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Observers are streaming-only (plan §5.6.2).
        self.inner.complete(req).await
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let upstream = self.inner.stream(req).await?;
        let observer = self.observer.clone();
        let mapped = upstream.map(move |item| {
            if let Ok(chunk) = &item {
                observer.on_chunk(chunk);
            }
            item
        });
        Ok(mapped.boxed())
    }
}

pub struct WithChunkTransformer<L: Llm, T: ChunkTransformer + 'static> {
    inner: L,
    transformer: std::sync::Arc<T>,
}

impl<L: Llm, T: ChunkTransformer + 'static> WithChunkTransformer<L, T> {
    pub fn new(inner: L, transformer: T) -> Self {
        Self {
            inner,
            transformer: std::sync::Arc::new(transformer),
        }
    }
}

#[async_trait]
impl<L: Llm, T: ChunkTransformer + 'static> Llm for WithChunkTransformer<L, T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        default_capabilities!(self.inner)
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // Transformers are streaming-only (plan §5.6.2).
        self.inner.complete(req).await
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        let upstream = self.inner.stream(req).await?;
        let transformer = self.transformer.clone();
        let mapped = upstream.filter_map(move |item| {
            let transformer = transformer.clone();
            async move {
                match item {
                    Ok(chunk) => transformer.on_chunk(chunk).map(Ok),
                    Err(e) => Some(Err(e)),
                }
            }
        });
        Ok(mapped.boxed())
    }
}

// ======================================================================
// Composition: LlmLayer / InfallibleLlmLayer / LlmExt
// ======================================================================

/// A bespoke layer type that produces a new `Llm` by wrapping an inner one.
///
/// Helper-trait layers don't implement `LlmLayer` directly — doing so would
/// create overlapping blanket impls. Instead, use the
/// `with_{request,response,full,chunk_observer,chunk_transformer}_layer`
/// methods on [`LlmExt`]. Custom layers (tracing, retry, prompt caching, …)
/// implement `LlmLayer` or [`InfallibleLlmLayer`] themselves.
pub trait LlmLayer<Inner: Llm> {
    type Output: Llm;
    fn wrap(self, inner: Inner) -> Result<Self::Output, LayerError>;
}

/// Marker extension for layers whose `wrap` never fails. Enables the
/// `with_layer` method (no `?` in the fluent chain).
pub trait InfallibleLlmLayer<Inner: Llm>: LlmLayer<Inner> {
    fn wrap_infallible(self, inner: Inner) -> Self::Output;
}

/// Fluent composition for `Llm` implementations (plan §5.5).
pub trait LlmExt: Llm + Sized {
    /// Compose with a bespoke layer whose construction is always infallible.
    fn with_layer<L>(self, layer: L) -> L::Output
    where
        L: InfallibleLlmLayer<Self>,
    {
        layer.wrap_infallible(self)
    }

    /// Compose with a bespoke layer that may reject the inner `Llm`
    /// (e.g., capability mismatch). Use `?` in the fluent chain.
    fn try_with_layer<L>(self, layer: L) -> Result<L::Output, LayerError>
    where
        L: LlmLayer<Self>,
    {
        layer.wrap(self)
    }

    fn with_request_layer<R: RequestLayer>(self, layer: R) -> WithRequestLayer<Self, R> {
        WithRequestLayer::new(self, layer)
    }

    fn with_response_layer<R: ResponseLayer>(self, layer: R) -> WithResponseLayer<Self, R> {
        WithResponseLayer::new(self, layer)
    }

    fn with_full_layer<F: FullLayer>(self, layer: F) -> WithFullLayer<Self, F> {
        WithFullLayer::new(self, layer)
    }

    fn with_chunk_observer<O: ChunkObserver + 'static>(
        self,
        observer: O,
    ) -> WithChunkObserver<Self, O> {
        WithChunkObserver::new(self, observer)
    }

    fn with_chunk_transformer<T: ChunkTransformer + 'static>(
        self,
        transformer: T,
    ) -> WithChunkTransformer<Self, T> {
        WithChunkTransformer::new(self, transformer)
    }
}

impl<T: Llm> LlmExt for T {}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Chunk;
    use futures::stream;
    use oharness_core::{Content, Message, ModelId, StopReason, Usage};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    // ---------- fixtures ----------

    struct StubLlm {
        name: &'static str,
        caps: LlmCapabilities,
        complete_calls: Arc<AtomicUsize>,
        stream_calls: Arc<AtomicUsize>,
        last_system: std::sync::Mutex<Option<String>>,
    }

    impl StubLlm {
        fn new() -> Self {
            Self {
                name: "stub",
                caps: LlmCapabilities {
                    streaming: true,
                    prompt_caching: false,
                    parallel_tool_use: false,
                    vision: false,
                    thinking: false,
                    structured_output: false,
                    max_context_tokens: 0,
                    max_output_tokens: 0,
                },
                complete_calls: Arc::new(AtomicUsize::new(0)),
                stream_calls: Arc::new(AtomicUsize::new(0)),
                last_system: std::sync::Mutex::new(None),
            }
        }

        fn canned_response() -> CompletionResponse {
            CompletionResponse {
                id: "msg_stub".into(),
                model: ModelId::new("stub"),
                content: vec![Content::text("hi")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }
        }

        fn canned_chunks() -> Vec<Chunk> {
            vec![
                Chunk::MessageStart {
                    id: "msg_stub".into(),
                    model: ModelId::new("stub"),
                },
                Chunk::TextDelta {
                    index: 0,
                    text: "hi".into(),
                },
                Chunk::MessageStop,
            ]
        }
    }

    #[async_trait]
    impl Llm for StubLlm {
        fn name(&self) -> &str {
            self.name
        }
        fn capabilities(&self) -> LlmCapabilities {
            self.caps.clone()
        }
        async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            self.complete_calls.fetch_add(1, Ordering::Relaxed);
            *self.last_system.lock().unwrap() = req.system.clone();
            Ok(Self::canned_response())
        }
        async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            self.stream_calls.fetch_add(1, Ordering::Relaxed);
            *self.last_system.lock().unwrap() = req.system.clone();
            let chunks = Self::canned_chunks()
                .into_iter()
                .map(Ok::<_, LlmError>)
                .collect::<Vec<_>>();
            Ok(stream::iter(chunks).boxed())
        }
    }

    fn req() -> CompletionRequest {
        CompletionRequest::new(vec![Message::user_text("hi")])
    }

    // ---------- RequestLayer ----------

    struct SetSystem(&'static str);
    impl RequestLayer for SetSystem {
        fn on_request(&self, req: &mut CompletionRequest) {
            req.system = Some(self.0.to_string());
        }
    }

    #[tokio::test]
    async fn request_layer_mutates_complete() {
        let stub = StubLlm::new();
        let wrapped = stub.with_request_layer(SetSystem("YOU ARE HELPFUL"));
        wrapped.complete(req()).await.unwrap();
        assert_eq!(
            wrapped.inner.last_system.lock().unwrap().as_deref(),
            Some("YOU ARE HELPFUL")
        );
    }

    #[tokio::test]
    async fn request_layer_mutates_stream() {
        let stub = StubLlm::new();
        let wrapped = stub.with_request_layer(SetSystem("SYSMSG"));
        let mut s = wrapped.stream(req()).await.unwrap();
        // drain
        while let Some(result) = s.next().await {
            result.unwrap();
        }
        assert_eq!(
            wrapped.inner.last_system.lock().unwrap().as_deref(),
            Some("SYSMSG")
        );
    }

    // ---------- ResponseLayer ----------

    struct RewriteId(&'static str);
    impl ResponseLayer for RewriteId {
        fn on_response(&self, res: &mut CompletionResponse) {
            res.id = self.0.to_string();
        }
        fn name(&self) -> &'static str {
            "RewriteId"
        }
    }

    #[tokio::test]
    async fn response_layer_mutates_complete() {
        let stub = StubLlm::new();
        let wrapped = stub.with_response_layer(RewriteId("msg_rewritten"));
        let res = wrapped.complete(req()).await.unwrap();
        assert_eq!(res.id, "msg_rewritten");
    }

    #[tokio::test]
    async fn response_layer_warn_and_skip_passes_stream_through() {
        let stub = StubLlm::new();
        let wrapped = stub.with_response_layer(RewriteId("msg_rewritten"));
        let mut s = wrapped.stream(req()).await.unwrap();
        let mut n = 0;
        while let Some(result) = s.next().await {
            result.unwrap();
            n += 1;
        }
        assert_eq!(n, 3);
    }

    struct RejectStream;
    impl ResponseLayer for RejectStream {
        fn on_response(&self, _: &mut CompletionResponse) {}
        fn stream_mode(&self) -> ResponseLayerStreamMode {
            ResponseLayerStreamMode::Error
        }
        fn name(&self) -> &'static str {
            "RejectStream"
        }
    }

    #[tokio::test]
    async fn response_layer_error_mode_rejects_stream() {
        let stub = StubLlm::new();
        let wrapped = stub.with_response_layer(RejectStream);
        match wrapped.stream(req()).await {
            Err(LlmError::Unsupported("response_layer_on_stream")) => {}
            Err(other) => panic!("wrong error variant: {other:?}"),
            Ok(_) => panic!("should have errored"),
        }
    }

    // ---------- FullLayer ----------

    struct CountAround {
        complete_calls: Arc<AtomicUsize>,
        stream_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl FullLayer for CountAround {
        async fn around_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            call: BoxFuture<'a, Result<CompletionResponse, LlmError>>,
        ) -> Result<CompletionResponse, LlmError> {
            self.complete_calls.fetch_add(1, Ordering::Relaxed);
            call.await
        }

        async fn around_stream<'a>(
            &'a self,
            _req: CompletionRequest,
            call: BoxFuture<'a, Result<ChunkStream, LlmError>>,
        ) -> Result<ChunkStream, LlmError> {
            self.stream_calls.fetch_add(1, Ordering::Relaxed);
            call.await
        }
    }

    #[tokio::test]
    async fn full_layer_around_complete_invoked() {
        let complete_calls = Arc::new(AtomicUsize::new(0));
        let stream_calls = Arc::new(AtomicUsize::new(0));
        let stub = StubLlm::new();
        let wrapped = stub.with_full_layer(CountAround {
            complete_calls: complete_calls.clone(),
            stream_calls: stream_calls.clone(),
        });
        wrapped.complete(req()).await.unwrap();
        assert_eq!(complete_calls.load(Ordering::Relaxed), 1);
        assert_eq!(stream_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn full_layer_around_stream_invoked() {
        let complete_calls = Arc::new(AtomicUsize::new(0));
        let stream_calls = Arc::new(AtomicUsize::new(0));
        let stub = StubLlm::new();
        let wrapped = stub.with_full_layer(CountAround {
            complete_calls: complete_calls.clone(),
            stream_calls: stream_calls.clone(),
        });
        let mut s = wrapped.stream(req()).await.unwrap();
        while let Some(r) = s.next().await {
            r.unwrap();
        }
        assert_eq!(complete_calls.load(Ordering::Relaxed), 0);
        assert_eq!(stream_calls.load(Ordering::Relaxed), 1);
    }

    // ---------- ChunkObserver ----------

    struct CountChunks(Arc<AtomicUsize>);
    impl ChunkObserver for CountChunks {
        fn on_chunk(&self, _chunk: &Chunk) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn chunk_observer_sees_every_chunk() {
        let count = Arc::new(AtomicUsize::new(0));
        let stub = StubLlm::new();
        let wrapped = stub.with_chunk_observer(CountChunks(count.clone()));
        let mut s = wrapped.stream(req()).await.unwrap();
        while let Some(r) = s.next().await {
            r.unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn chunk_observer_does_not_see_complete() {
        let count = Arc::new(AtomicUsize::new(0));
        let stub = StubLlm::new();
        let wrapped = stub.with_chunk_observer(CountChunks(count.clone()));
        wrapped.complete(req()).await.unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    // ---------- ChunkTransformer ----------

    struct DropTextDeltas;
    impl ChunkTransformer for DropTextDeltas {
        fn on_chunk(&self, chunk: Chunk) -> Option<Chunk> {
            if matches!(chunk, Chunk::TextDelta { .. }) {
                None
            } else {
                Some(chunk)
            }
        }
    }

    #[tokio::test]
    async fn chunk_transformer_can_drop_chunks() {
        let stub = StubLlm::new();
        let wrapped = stub.with_chunk_transformer(DropTextDeltas);
        let s = wrapped.stream(req()).await.unwrap();
        let collected: Vec<Chunk> = s.map(|r| r.unwrap()).collect().await;
        // MessageStart + MessageStop, TextDelta dropped.
        assert_eq!(collected.len(), 2);
        assert!(matches!(collected[0], Chunk::MessageStart { .. }));
        assert!(matches!(collected[1], Chunk::MessageStop));
    }

    // ---------- LlmLayer / InfallibleLlmLayer / try_with_layer ----------

    struct RequiresStreamingLayer;
    impl<Inner: Llm> LlmLayer<Inner> for RequiresStreamingLayer {
        type Output = Inner;
        fn wrap(self, inner: Inner) -> Result<Self::Output, LayerError> {
            if inner.capabilities().streaming {
                Ok(inner)
            } else {
                Err(LayerError::MissingCapability {
                    layer: "RequiresStreaming",
                    capability: "streaming",
                })
            }
        }
    }

    struct AlwaysOkLayer;
    impl<Inner: Llm> LlmLayer<Inner> for AlwaysOkLayer {
        type Output = Inner;
        fn wrap(self, inner: Inner) -> Result<Self::Output, LayerError> {
            Ok(inner)
        }
    }
    impl<Inner: Llm> InfallibleLlmLayer<Inner> for AlwaysOkLayer {
        fn wrap_infallible(self, inner: Inner) -> Self::Output {
            inner
        }
    }

    #[tokio::test]
    async fn try_with_layer_succeeds_when_capability_present() {
        let stub = StubLlm::new();
        let _wrapped = stub
            .try_with_layer(RequiresStreamingLayer)
            .expect("present");
    }

    #[tokio::test]
    async fn try_with_layer_fails_when_capability_missing() {
        let mut stub = StubLlm::new();
        stub.caps.streaming = false;
        match stub.try_with_layer(RequiresStreamingLayer) {
            Ok(_) => panic!("should have failed"),
            Err(LayerError::MissingCapability { capability, .. }) => {
                assert_eq!(capability, "streaming");
            }
            Err(other) => panic!("wrong error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn with_layer_composes_infallibly() {
        let stub = StubLlm::new();
        // Fluent chain compiles without `?`.
        let wrapped = stub
            .with_layer(AlwaysOkLayer)
            .with_request_layer(SetSystem("sys"))
            .with_response_layer(RewriteId("msg_chain"));
        let res = wrapped.complete(req()).await.unwrap();
        assert_eq!(res.id, "msg_chain");
    }

    // ---------- Capability propagation ----------

    #[tokio::test]
    async fn capabilities_delegate_to_inner() {
        let stub = StubLlm::new();
        let wrapped = stub.with_request_layer(SetSystem("x"));
        assert!(wrapped.capabilities().streaming);
    }

    // ---------- Mixed chain smoke ----------

    #[tokio::test]
    async fn mixed_chain_applies_all_layers_in_order() {
        let count = Arc::new(AtomicUsize::new(0));
        let stub = StubLlm::new();
        let wrapped = stub
            .with_request_layer(SetSystem("system"))
            .with_response_layer(RewriteId("msg_final"))
            .with_chunk_observer(CountChunks(count.clone()));

        // complete(): RequestLayer mutates, ResponseLayer rewrites, observer
        // does nothing (streaming-only).
        let res = wrapped.complete(req()).await.unwrap();
        assert_eq!(res.id, "msg_final");
        assert_eq!(count.load(Ordering::Relaxed), 0);

        // stream(): RequestLayer mutates, observer counts, ResponseLayer
        // WarnAndSkip passes through.
        let mut s = wrapped.stream(req()).await.unwrap();
        while let Some(r) = s.next().await {
            r.unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }
}
