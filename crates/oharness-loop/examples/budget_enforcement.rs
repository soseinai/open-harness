//! `budget_enforcement` — cap a run's token usage with a
//! [`BudgetMiddleware`] and show what happens when the cap trips.
//!
//! `BudgetMiddleware` is a plain `Llm` wrapper: wrap any provider,
//! get pre-call + post-call accounting for free. When the cap trips,
//! the underlying `Llm::complete` call returns
//! `LlmError::Provider(BudgetExceeded)`, which the agent loop
//! converts to `Termination::Failed { category: Llm }`. The pre-call
//! check is the first line of defense — it rejects before the call
//! is dispatched, so no tokens are actually spent when the budget
//! denies. Post-call accounting updates the consumed counters from
//! the real `CompletionResponse.usage`.
//!
//! The shipped budget types — `TokenBudget`, `StepBudget`,
//! `CostBudget`, `TimeBudget`, `CompositeBudget` — all satisfy the
//! same `BudgetHandle` trait and compose the same way.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example budget_enforcement -p oharness-loop
//! ```

use async_trait::async_trait;
use oharness_budget::{BudgetMiddleware, TokenBudget};
use oharness_core::{
    BudgetHandle, CompletionRequest, CompletionResponse, Content, LlmCapabilities, ModelId,
    StopReason, Task, Termination, Usage,
};
use oharness_llm::{ChunkStream, Llm, LlmError};
use oharness_loop::{Agent, ReactLoop};
use oharness_tools::fs::FsToolSet;
use oharness_trace::InMemorySink;
use std::sync::Arc;

// A scripted LLM that returns a suspiciously big response — enough
// to blow past a tight input+output token cap. Real providers
// report usage in the `CompletionResponse.usage` field; we fake it
// here for determinism.
struct ChattyLlm;

#[async_trait]
impl Llm for ChattyLlm {
    fn name(&self) -> &str {
        "chatty"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(CompletionResponse {
            id: "msg_1".into(),
            model: ModelId::new("chatty-model"),
            content: vec![Content::text("Sure, here is a very long response…")],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                tokens_input: 100,
                tokens_output: 200,
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
    // Tight cap so we're guaranteed to trip it.
    let budget: Arc<dyn BudgetHandle> = Arc::new(TokenBudget::input_plus_output(50));

    // Wrap the LLM with BudgetMiddleware. The wrapper itself impls
    // `Llm`, so it composes with every other middleware in the
    // workspace (`RequestTracer`, `PromptCaching`, response layers,
    // …) without further ceremony.
    let bounded_llm = Arc::new(BudgetMiddleware::new(ChattyLlm, budget.clone()));

    let sink = Arc::new(InMemorySink::new());
    let agent = Agent::builder()
        .with_llm(bounded_llm)
        .with_tools(Arc::new(FsToolSet::new()))
        .with_event_sink(sink.clone())
        .with_loop(Box::new(ReactLoop::new()))
        .with_max_turns(5)
        .build()?;

    let outcome = agent.run(Task::new("hello")).await?;

    match &outcome.termination {
        Termination::Failed { error, .. } => {
            println!(
                "Termination: Failed (category={:?})\n  message: {}",
                error.category, error.message
            );
        }
        other => println!("Termination: {other:?}"),
    }

    // The budget handle carries a live snapshot. Useful for
    // per-task telemetry independent of the trajectory events.
    let snapshot = budget.snapshot();
    let remaining_in = snapshot
        .remaining
        .as_ref()
        .map(|r| r.tokens_input.to_string())
        .unwrap_or_else(|| "unbounded".into());
    let remaining_out = snapshot
        .remaining
        .as_ref()
        .map(|r| r.tokens_output.to_string())
        .unwrap_or_else(|| "unbounded".into());
    println!(
        "Budget snapshot: consumed {}/{} input tokens, {}/{} output tokens (remaining)",
        snapshot.consumed.tokens_input,
        remaining_in,
        snapshot.consumed.tokens_output,
        remaining_out,
    );

    println!("Trajectory events captured: {}", sink.events().len());
    Ok(())
}
