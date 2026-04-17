//! `BudgetMiddleware` — an `Llm` wrapper that threads a shared
//! `BudgetHandle` counter through three hook sites:
//!
//! 1. **Pre-call**: estimate tokens/cost from the `CompletionRequest` and
//!    call [`BudgetHandle::check`]. Deny short-circuits to
//!    [`BudgetExceeded`] wrapped in [`LlmError::Provider`].
//! 2. **Post-`complete`**: call [`BudgetHandle::consume`] with actual usage
//!    and wall-clock from the response.
//! 3. **Per-chunk on `stream`**: observe [`Chunk::Usage`] updates and
//!    consume the *delta* since the previous update (Anthropic emits usage
//!    multiple times as a running counter — a naïve sum would double-count).
//!    A single `steps: 1` and final wall-clock increment land at
//!    [`Chunk::MessageStop`].
//!
//! `BudgetMiddleware` implements [`Llm`] directly rather than via a
//! [`oharness_llm::FullLayer`] / [`oharness_llm::ChunkObserver`] pair
//! because those wrappers cannot share the budget counter (plan §5.6.2,
//! §10.3).

use crate::amount::{amount_from_response, amount_from_usage, budget_request_from};
use crate::error::BudgetExceeded;
use crate::pricing::PricingTable;
use async_trait::async_trait;
use futures::StreamExt;
use oharness_core::{
    BudgetAmount, BudgetDecision, BudgetHandle, BudgetRequest, CompletionRequest,
    CompletionResponse, LlmCapabilities, ModelId, Usage,
};
use oharness_llm::{Chunk, ChunkStream, Llm, LlmError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct BudgetMiddleware<L: Llm> {
    inner: L,
    budget: Arc<dyn BudgetHandle>,
    pricing: Arc<PricingTable>,
}

impl<L: Llm> BudgetMiddleware<L> {
    pub fn new(inner: L, budget: Arc<dyn BudgetHandle>) -> Self {
        Self {
            inner,
            budget,
            pricing: Arc::new(PricingTable::empty()),
        }
    }

    pub fn with_pricing(mut self, pricing: Arc<PricingTable>) -> Self {
        self.pricing = pricing;
        self
    }

    pub fn budget(&self) -> &Arc<dyn BudgetHandle> {
        &self.budget
    }
}

fn budget_error(reason: impl Into<String>) -> LlmError {
    LlmError::Provider(Box::new(BudgetExceeded::new(reason)))
}

async fn pre_check(
    budget: &Arc<dyn BudgetHandle>,
    req: &CompletionRequest,
    pricing: &PricingTable,
) -> Result<(), LlmError> {
    match budget.check(budget_request_from(req, pricing)).await {
        BudgetDecision::Allow => Ok(()),
        BudgetDecision::Deny { reason } => Err(budget_error(reason)),
    }
}

#[async_trait]
impl<L: Llm> Llm for BudgetMiddleware<L> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.inner.capabilities()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        pre_check(&self.budget, &req, &self.pricing).await?;
        let started_at = Instant::now();
        let res = self.inner.complete(req).await?;
        let amount = amount_from_response(&res, started_at.elapsed(), &self.pricing);
        self.budget.consume(amount).await;
        Ok(res)
    }

    async fn stream(&self, req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        pre_check(&self.budget, &req, &self.pricing).await?;

        let started_at = Instant::now();
        let inner_stream = self.inner.stream(req).await?;
        let budget = self.budget.clone();
        let pricing = self.pricing.clone();
        let state = Arc::new(Mutex::new(StreamState::default()));
        let stop_consumed = Arc::new(Mutex::new(false));

        let wrapped = inner_stream.then(move |chunk_result| {
            let budget = budget.clone();
            let pricing = pricing.clone();
            let state = state.clone();
            let stop_consumed = stop_consumed.clone();
            async move {
                handle_chunk(
                    chunk_result,
                    &budget,
                    &pricing,
                    &state,
                    started_at,
                    &stop_consumed,
                )
                .await
            }
        });

        Ok(wrapped.boxed())
    }
}

#[derive(Default)]
struct StreamState {
    model: Option<ModelId>,
    /// Running cumulative usage the upstream has reported so far. We consume
    /// *deltas* against this running count so multi-emission providers don't
    /// get double-counted.
    last_usage: Usage,
}

async fn handle_chunk(
    chunk_result: Result<Chunk, LlmError>,
    budget: &Arc<dyn BudgetHandle>,
    pricing: &PricingTable,
    state: &Arc<Mutex<StreamState>>,
    started_at: Instant,
    stop_consumed: &Arc<Mutex<bool>>,
) -> Result<Chunk, LlmError> {
    let chunk = chunk_result?;

    match &chunk {
        Chunk::MessageStart { model, .. } => {
            state.lock().expect("budget stream state").model = Some(model.clone());
        }
        Chunk::Usage { usage } => {
            let (model, delta) = {
                let mut s = state.lock().expect("budget stream state");
                let model = s.model.clone().unwrap_or_else(|| ModelId::new(""));
                let delta = delta_usage(&s.last_usage, usage);
                s.last_usage = usage.clone();
                (model, delta)
            };
            let amount = amount_from_usage(&delta, &model, pricing);
            budget.consume(amount).await;
            if let BudgetDecision::Deny { reason } = budget.check(BudgetRequest::default()).await {
                return Err(budget_error(reason));
            }
        }
        Chunk::MessageStop => {
            // Consume the step + wall-clock exactly once. Guard against
            // providers that emit `MessageStop` multiple times. Take the
            // flag value out of the lock BEFORE awaiting — a MutexGuard
            // cannot be held across `.await`.
            let should_consume = {
                let mut flag = stop_consumed.lock().expect("budget stop flag");
                if *flag {
                    false
                } else {
                    *flag = true;
                    true
                }
            };
            if should_consume {
                budget
                    .consume(BudgetAmount {
                        steps: 1,
                        wall_clock: started_at.elapsed(),
                        ..Default::default()
                    })
                    .await;
            }
        }
        _ => {}
    }

    Ok(chunk)
}

fn delta_usage(running: &Usage, current: &Usage) -> Usage {
    Usage {
        tokens_input: current.tokens_input.saturating_sub(running.tokens_input),
        tokens_output: current.tokens_output.saturating_sub(running.tokens_output),
        tokens_cache_read: current
            .tokens_cache_read
            .saturating_sub(running.tokens_cache_read),
        tokens_cache_create: current
            .tokens_cache_create
            .saturating_sub(running.tokens_cache_create),
    }
}

#[cfg(all(test, feature = "token"))]
mod tests {
    use super::*;
    use crate::TokenBudget;
    use async_trait::async_trait;
    use futures::stream;
    use oharness_core::{Content, Message, StopReason};
    use oharness_llm::{BlockStartKind, Chunk};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ScriptedLlm {
        complete_response: Mutex<Option<CompletionResponse>>,
        stream_chunks: Mutex<Option<Vec<Result<Chunk, LlmError>>>>,
        complete_calls: AtomicUsize,
    }

    impl ScriptedLlm {
        fn new(_model: &str) -> Self {
            Self {
                complete_response: Mutex::new(None),
                stream_chunks: Mutex::new(None),
                complete_calls: AtomicUsize::new(0),
            }
        }

        fn with_complete(self, res: CompletionResponse) -> Self {
            *self.complete_response.lock().unwrap() = Some(res);
            self
        }

        fn with_stream(self, chunks: Vec<Result<Chunk, LlmError>>) -> Self {
            *self.stream_chunks.lock().unwrap() = Some(chunks);
            self
        }
    }

    #[async_trait]
    impl Llm for ScriptedLlm {
        fn name(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities {
                streaming: true,
                prompt_caching: false,
                parallel_tool_use: false,
                vision: false,
                thinking: false,
                structured_output: false,
                max_context_tokens: 0,
                max_output_tokens: 0,
            }
        }
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            self.complete_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self
                .complete_response
                .lock()
                .unwrap()
                .clone()
                .expect("complete_response"))
        }
        async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            let chunks = self
                .stream_chunks
                .lock()
                .unwrap()
                .take()
                .expect("stream_chunks (scripted stream is one-shot)");
            Ok(stream::iter(chunks).boxed())
        }
    }

    fn req_text(s: &str) -> CompletionRequest {
        CompletionRequest::new(vec![Message::user_text(s)])
    }

    fn completion_response(model: &str, input: u64, output: u64) -> CompletionResponse {
        CompletionResponse {
            id: "r".into(),
            model: ModelId::new(model),
            content: vec![Content::text("ok")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: input,
                tokens_output: output,
                ..Default::default()
            },
        }
    }

    fn scripted_chunks(model: &str) -> Vec<Result<Chunk, LlmError>> {
        vec![
            Ok(Chunk::MessageStart {
                id: "msg".into(),
                model: ModelId::new(model),
            }),
            Ok(Chunk::Usage {
                usage: Usage {
                    tokens_input: 25,
                    tokens_output: 1,
                    ..Default::default()
                },
            }),
            Ok(Chunk::BlockStart {
                index: 0,
                start: BlockStartKind::Text,
            }),
            Ok(Chunk::TextDelta {
                index: 0,
                text: "hi".into(),
            }),
            Ok(Chunk::BlockStop { index: 0 }),
            Ok(Chunk::Usage {
                // Running total: same input, final output. The middleware
                // should consume only the delta, not double-count input.
                usage: Usage {
                    tokens_input: 25,
                    tokens_output: 15,
                    ..Default::default()
                },
            }),
            Ok(Chunk::MessageStop),
        ]
    }

    // ---------- complete() path ----------

    #[tokio::test]
    async fn complete_pre_check_deny_short_circuits() {
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(0));
        let stub = ScriptedLlm::new("m").with_complete(completion_response("m", 10, 5));
        let wrapped = BudgetMiddleware::new(stub, budget.clone());
        match wrapped.complete(req_text("hello")).await {
            Err(LlmError::Provider(e)) => assert!(e.downcast_ref::<BudgetExceeded>().is_some()),
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("should have denied"),
        }
        // Inner Llm was not called.
        assert_eq!(wrapped.inner.complete_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn complete_consumes_after_call() {
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(10_000));
        let stub = ScriptedLlm::new("m").with_complete(completion_response("m", 100, 50));
        let wrapped = BudgetMiddleware::new(stub, budget.clone());
        wrapped.complete(req_text("hello")).await.unwrap();
        let s = budget.snapshot();
        assert_eq!(s.consumed.tokens_input, 100);
        assert_eq!(s.consumed.tokens_output, 50);
        assert_eq!(s.consumed.steps, 1);
    }

    // ---------- stream() path ----------

    #[tokio::test]
    async fn stream_pre_check_deny_short_circuits() {
        // `ChunkStream` doesn't impl `Debug`, so `expect_err` can't be used;
        // match on the Result directly.
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(0));
        let stub = ScriptedLlm::new("m").with_stream(scripted_chunks("m"));
        let wrapped = BudgetMiddleware::new(stub, budget.clone());
        match wrapped.stream(req_text("hello")).await {
            Err(LlmError::Provider(e)) => assert!(e.downcast_ref::<BudgetExceeded>().is_some()),
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("should have denied"),
        }
    }

    #[tokio::test]
    async fn stream_consumes_delta_not_sum_over_multiple_usage_chunks() {
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(10_000));
        let stub = ScriptedLlm::new("m").with_stream(scripted_chunks("m"));
        let wrapped = BudgetMiddleware::new(stub, budget.clone());
        let mut s = wrapped.stream(req_text("hello")).await.unwrap();
        let mut count = 0;
        while let Some(result) = s.next().await {
            result.unwrap();
            count += 1;
        }
        assert!(count > 0);

        let snap = budget.snapshot();
        // Final totals: input=25, output=15 (NOT 50, NOT 16).
        assert_eq!(snap.consumed.tokens_input, 25);
        assert_eq!(snap.consumed.tokens_output, 15);
        assert_eq!(snap.consumed.steps, 1);
    }

    #[tokio::test]
    async fn stream_short_circuits_when_usage_exceeds_mid_stream() {
        // Cap permits only initial (25 input + 1 output) = 26. Second usage
        // would push to 25 + 15 = 40 total, breaching the cap.
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(30));
        let stub = ScriptedLlm::new("m").with_stream(scripted_chunks("m"));
        let wrapped = BudgetMiddleware::new(stub, budget.clone());
        let mut s = wrapped.stream(req_text("hello")).await.unwrap();

        let mut terminated_with_budget_err = false;
        while let Some(result) = s.next().await {
            if let Err(LlmError::Provider(e)) = &result {
                if e.downcast_ref::<BudgetExceeded>().is_some() {
                    terminated_with_budget_err = true;
                    break;
                }
            }
            // otherwise pass through ok chunks
            result.expect("non-budget error in test");
        }
        assert!(terminated_with_budget_err);
    }

    // ---------- capability + name delegation ----------

    #[tokio::test]
    async fn name_and_capabilities_delegate_to_inner() {
        let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(10_000));
        let stub = ScriptedLlm::new("m");
        let wrapped = BudgetMiddleware::new(stub, budget);
        assert_eq!(wrapped.name(), "scripted");
        assert!(wrapped.capabilities().streaming);
    }
}
