//! `custom_middleware` — compose three custom layers around an LLM.
//!
//! Covers the three middleware shapes users write most often:
//!
//! 1. **`RequestLayer`** — sync, mutate `CompletionRequest` in place.
//!    Used here to stamp a request-id into `metadata`.
//! 2. **`ResponseLayer`** — sync, mutate `CompletionResponse` in
//!    place. Used here to redact a secret-looking token from every
//!    text block.
//! 3. **`FullLayer`** — async, wrap the whole `complete()` /
//!    `stream()` call. Used here as a duration timer that logs
//!    before/after every call. The shape with `BoxFuture` is the
//!    standard async-trait dance — once you've seen it, bespoke
//!    wrappers (rate limiters, retries, caching) drop straight in.
//!
//! Composition via `LlmExt`:
//!
//! ```text
//! inner
//!     .with_request_layer(RequestIdStamp::new())
//!     .with_response_layer(RedactSecrets)
//!     .with_full_layer(Timer)
//! ```
//!
//! Each wrapper itself implements `Llm`, so the whole chain is a
//! drop-in replacement for any `Arc<dyn Llm>`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example custom_middleware -p oharness-loop
//! ```

use async_trait::async_trait;
use futures::future::BoxFuture;
use oharness_core::{
    CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId, StopReason, Task,
    Usage,
};
use oharness_llm::{ChunkStream, FullLayer, Llm, LlmError, LlmExt, RequestLayer, ResponseLayer};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------
// 1. RequestLayer — stamp a monotonically increasing request-id.
// ---------------------------------------------------------------------

struct RequestIdStamp {
    counter: AtomicU64,
}

impl RequestIdStamp {
    fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl RequestLayer for RequestIdStamp {
    fn on_request(&self, req: &mut CompletionRequest) {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        // `extensions` is a reverse-DNS metadata map — the canonical
        // home for namespaced request annotations. (`anthropic.*` and
        // `openai.*` live here too.)
        req.extensions.insert(
            "example.request_id".into(),
            serde_json::json!(format!("req-{id}")),
        );
        eprintln!("[RequestIdStamp] stamped req-{id}");
    }
}

// ---------------------------------------------------------------------
// 2. ResponseLayer — redact a fake "secret" token from every text block.
// ---------------------------------------------------------------------

struct RedactSecrets;

impl ResponseLayer for RedactSecrets {
    fn on_response(&self, res: &mut CompletionResponse) {
        for block in res.content.iter_mut() {
            if let Content::Text { text } = block {
                if text.contains("sk-live-") {
                    *text = text.replace("sk-live-", "sk-live-REDACTED-");
                    eprintln!("[RedactSecrets] redacted secret in response");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// 3. FullLayer — wrap the whole `complete()`/`stream()` call. A timing
//    layer: log before + after + the elapsed duration. The async-trait
//    + BoxFuture shape is the canonical way to layer around an
//    existing future.
// ---------------------------------------------------------------------

struct Timer;

#[async_trait]
impl FullLayer for Timer {
    async fn around_complete<'a>(
        &'a self,
        _req: CompletionRequest,
        call: BoxFuture<'a, Result<CompletionResponse, LlmError>>,
    ) -> Result<CompletionResponse, LlmError> {
        let start = Instant::now();
        eprintln!("[Timer] complete() starting");
        let result = call.await;
        eprintln!("[Timer] complete() finished in {:?}", start.elapsed());
        result
    }

    async fn around_stream<'a>(
        &'a self,
        _req: CompletionRequest,
        call: BoxFuture<'a, Result<ChunkStream, LlmError>>,
    ) -> Result<ChunkStream, LlmError> {
        // Streaming side uses the default passthrough here — timing
        // an active stream wants `ChunkObserver`, not FullLayer
        // wrapping. Left as a no-op to keep the example focused.
        call.await
    }
}

// ---------------------------------------------------------------------
// Scripted inner LLM — the target the middleware wraps.
// ---------------------------------------------------------------------

struct ScriptedLlm;

#[async_trait]
impl Llm for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "msg_1".into(),
            model: ModelId::new("middleware-example"),
            content: vec![Content::text(
                "All set. My API key is sk-live-1234567890abc — please keep it safe.",
            )],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 7,
                tokens_output: 20,
                ..Default::default()
            },
        })
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compose the three layers around the scripted LLM. The order
    // matters: layers closer to `inner` see the request after outer
    // layers have mutated it, and the response before outer layers
    // mutate it. Here Timer wraps Response wraps Request — so the
    // timing measurement includes the inner `complete` + the
    // redaction pass.
    let llm = ScriptedLlm
        .with_request_layer(RequestIdStamp::new())
        .with_response_layer(RedactSecrets)
        .with_full_layer(Timer);

    let agent = Agent::builder()
        .with_llm(Arc::new(llm))
        .with_tools(Arc::new(FsToolSet::new()))
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(2)
        .build()?;

    let outcome = agent.run(Task::new("test middleware composition")).await?;

    if let Some(oharness_core::Message::Assistant { content, .. }) = outcome.final_messages.last() {
        for c in content {
            if let Content::Text { text } = c {
                println!("Assistant (post-redaction): {text}");
            }
        }
    }
    println!("Termination: {:?}", outcome.termination);
    Ok(())
}
