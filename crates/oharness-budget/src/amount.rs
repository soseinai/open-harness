//! Helpers to project `CompletionRequest` / `CompletionResponse` / per-chunk
//! `Usage` values onto the `BudgetAmount` / `BudgetRequest` shapes that the
//! [`oharness_core::BudgetHandle`] API expects.

use crate::pricing::PricingTable;
use oharness_core::{
    BudgetAmount, BudgetRequest, CompletionRequest, CompletionResponse, Content, Message, ModelId,
    Usage,
};
use std::time::Duration;

/// Rough pre-call estimate. Input tokens are guessed via a 4-chars-per-token
/// heuristic over message text (plan §4.8 uses the same figure for
/// `ConversationView::token_estimate`); output tokens come from
/// `req.max_tokens` if set; cost is left `None` because it depends on model
/// choice, which the caller will supply to `BudgetMiddleware`.
pub fn budget_request_from(req: &CompletionRequest, _pricing: &PricingTable) -> BudgetRequest {
    let chars: usize = req
        .system
        .as_ref()
        .map(String::len)
        .unwrap_or(0)
        .saturating_add(
            req.messages
                .iter()
                .map(|m| match m {
                    Message::System { content, .. } => content.len(),
                    Message::User { content, .. } | Message::Assistant { content, .. } => {
                        content.iter().map(text_len).sum()
                    }
                })
                .sum(),
        );
    BudgetRequest {
        estimated_input_tokens: Some((chars / 4) as u64),
        estimated_output_tokens: req.max_tokens.map(u64::from),
        estimated_cost_usd: None,
        label: None,
    }
}

fn text_len(c: &Content) -> usize {
    match c {
        Content::Text { text } => text.len(),
        Content::Thinking { thinking } => thinking.len(),
        Content::ToolUse { input, .. } => input.to_string().len(),
        Content::ToolResult { output, .. } => output.content.iter().map(text_len).sum(),
        Content::Image(_) | Content::Document(_) | Content::Audio(_) | Content::Citation(_) => 0,
    }
}

/// Project a fully-decoded `CompletionResponse` onto a `BudgetAmount`. Always
/// counts `steps: 1` — the response represents one completed call.
pub fn amount_from_response(
    res: &CompletionResponse,
    wall_clock: Duration,
    pricing: &PricingTable,
) -> BudgetAmount {
    BudgetAmount {
        tokens_input: res.usage.tokens_input,
        tokens_output: res.usage.tokens_output,
        cost_usd: pricing.cost_for(&res.model, &res.usage),
        wall_clock,
        steps: 1,
    }
}

/// Project a per-chunk `Chunk::Usage` payload. `steps` is **0** — each chunk
/// update contributes to the same enclosing call; the call itself is counted
/// once by the post-stream accounting in `BudgetMiddleware`.
pub fn amount_from_usage(usage: &Usage, model: &ModelId, pricing: &PricingTable) -> BudgetAmount {
    BudgetAmount {
        tokens_input: usage.tokens_input,
        tokens_output: usage.tokens_output,
        cost_usd: pricing.cost_for(model, usage),
        wall_clock: Duration::ZERO,
        steps: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_request_estimates_from_text() {
        let req = CompletionRequest {
            messages: vec![Message::user_text("hello world")], // 11 chars
            tools: Vec::new(),
            system: Some("you are helpful".into()), // 15 chars
            max_tokens: Some(1024),
            temperature: None,
            stop_sequences: Vec::new(),
            cache_hints: Default::default(),
            extensions: Default::default(),
        };
        let br = budget_request_from(&req, &PricingTable::empty());
        // (15 + 11) / 4 = 6
        assert_eq!(br.estimated_input_tokens, Some(6));
        assert_eq!(br.estimated_output_tokens, Some(1024));
        assert!(br.estimated_cost_usd.is_none());
    }

    #[test]
    fn amount_from_response_counts_one_step_and_computes_cost() {
        let mut pricing = PricingTable::empty();
        pricing.override_model(ModelId::new("m"), ModelPricing::new(2.0, 4.0));
        let res = CompletionResponse {
            id: "r".into(),
            model: ModelId::new("m"),
            content: Vec::new(),
            stop_reason: oharness_core::StopReason::EndTurn,
            usage: Usage {
                tokens_input: 1_000_000,
                tokens_output: 500_000,
                ..Default::default()
            },
        };
        let amt = amount_from_response(&res, Duration::from_millis(250), &pricing);
        assert_eq!(amt.tokens_input, 1_000_000);
        assert_eq!(amt.tokens_output, 500_000);
        assert_eq!(amt.steps, 1);
        assert_eq!(amt.wall_clock, Duration::from_millis(250));
        assert!((amt.cost_usd - (2.0 + 2.0)).abs() < 1e-9);
    }

    #[test]
    fn amount_from_usage_does_not_bump_steps() {
        let pricing = PricingTable::empty();
        let usage = Usage {
            tokens_input: 10,
            tokens_output: 5,
            ..Default::default()
        };
        let amt = amount_from_usage(&usage, &ModelId::new("m"), &pricing);
        assert_eq!(amt.steps, 0);
    }

    use crate::pricing::ModelPricing;
}
