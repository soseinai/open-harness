//! `LlmJudgeCritic` — prompts a judge LLM with a rubric and parses a
//! numeric score from its response. Behind the `llm-judge` feature.

use crate::critic::{AssessmentContext, Critic, CriticVerdict};
use async_trait::async_trait;
use oharness_core::{CompletionRequest, Content, Message};
use oharness_llm::Llm;
use std::sync::Arc;

pub struct LlmJudgeCritic {
    judge: Arc<dyn Llm>,
    rubric: String,
    threshold: f32,
    name: String,
}

impl LlmJudgeCritic {
    /// Build a judge critic. `rubric` is the scoring guidance shown to
    /// the judge; `threshold` (0.0..1.0) decides the accept/reject
    /// cutoff.
    pub fn new(judge: Arc<dyn Llm>, rubric: impl Into<String>, threshold: f32) -> Self {
        Self {
            judge,
            rubric: rubric.into(),
            threshold,
            name: "llm-judge".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

fn assistant_text(message: &Message) -> String {
    let Message::Assistant { content, .. } = message else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_judge_prompt(rubric: &str, task_instruction: &str, assistant_output: &str) -> String {
    format!(
        "You are a strict grader. Evaluate the assistant's response to the task \
         against this rubric:\n\n{rubric}\n\n\
         Task:\n{task_instruction}\n\n\
         Assistant response:\n{assistant_output}\n\n\
         Respond with a single line of the form `SCORE: <number>` where <number> \
         is between 0 and 1. Do not include any other output."
    )
}

/// Parse the judge's response text for a `SCORE: <float>` line. Returns
/// `None` when the response contains no such line. Case-insensitive on
/// the `SCORE:` token; trims whitespace; clamps to `[0, 1]`.
fn parse_score(text: &str) -> Option<f32> {
    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        let Some(rest) = lower.strip_prefix("score:") else {
            continue;
        };
        let value = rest.trim();
        if let Ok(n) = value.parse::<f32>() {
            return Some(n.clamp(0.0, 1.0));
        }
    }
    None
}

#[async_trait]
impl Critic for LlmJudgeCritic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        let prompt = render_judge_prompt(
            &self.rubric,
            &ctx.task.instruction,
            &assistant_text(&ctx.latest_turn.message),
        );
        let req = CompletionRequest::new(vec![Message::user_text(prompt)]);
        let res = match self.judge.complete(req).await {
            Ok(r) => r,
            Err(e) => {
                // Fail-open on judge errors so the loop keeps moving;
                // the loop's fail-open wrapper emits `critic.failed` so
                // replay can still detect the divergence.
                tracing::warn!(
                    target: "oharness.critic.llm_judge",
                    error = %e,
                    "LlmJudgeCritic.complete failed; returning Accept",
                );
                return CriticVerdict::AcceptWithNote(format!(
                    "llm-judge: judge error ({e}); defaulting to Accept"
                ));
            }
        };

        let text = assistant_text_response(&res.content);
        let Some(score) = parse_score(&text) else {
            return CriticVerdict::AcceptWithNote(format!(
                "llm-judge: no `SCORE:` line parsed from judge output (text: {})",
                truncate(&text, 200)
            ));
        };

        if score >= self.threshold {
            CriticVerdict::AcceptWithNote(format!(
                "llm-judge: score {score:.3} >= threshold {:.3}",
                self.threshold
            ))
        } else {
            CriticVerdict::Reject {
                reason: format!(
                    "llm-judge: score {score:.3} < threshold {:.3}",
                    self.threshold
                ),
            }
        }
    }
}

fn assistant_text_response(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oharness_core::{
        AssistantTurn, CompletionResponse, ConversationView, LlmCapabilities, ModelId, StopReason,
        Task, TrajectoryView, Usage,
    };
    use oharness_llm::{ChunkStream, LlmError};
    use std::sync::Mutex;

    struct Scripted(Mutex<Option<CompletionResponse>>);
    #[async_trait]
    impl Llm for Scripted {
        fn name(&self) -> &str {
            "scripted-judge"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            self.0
                .lock()
                .unwrap()
                .take()
                .ok_or(LlmError::Unsupported("one-shot"))
        }
        async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            Err(LlmError::Unsupported("stream"))
        }
    }

    fn score_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            id: "judge".into(),
            model: ModelId::new("judge-model"),
            content: vec![Content::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    async fn assess_with(critic: &LlmJudgeCritic) -> CriticVerdict {
        let task = Task::new("respond politely");
        let msg = Message::assistant_text("greetings");
        let turn = AssistantTurn::new(0, "span", msg, Usage::default(), StopReason::EndTurn);
        let ctx = AssessmentContext::new(
            &task,
            ConversationView::new(&[]),
            &turn,
            TrajectoryView::new(&[]),
        );
        critic.assess(&ctx).await
    }

    #[test]
    fn parse_score_basic() {
        assert_eq!(parse_score("SCORE: 0.8"), Some(0.8));
        assert_eq!(parse_score("score: 1.0"), Some(1.0));
        assert_eq!(parse_score("noise\nSCORE: 0.2\nother"), Some(0.2));
    }

    #[test]
    fn parse_score_clamps_out_of_range() {
        assert_eq!(parse_score("SCORE: 1.5"), Some(1.0));
        assert_eq!(parse_score("SCORE: -0.2"), Some(0.0));
    }

    #[test]
    fn parse_score_returns_none_when_missing() {
        assert_eq!(parse_score("the answer is great"), None);
        assert_eq!(parse_score(""), None);
    }

    #[tokio::test]
    async fn accept_above_threshold() {
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(Some(score_response("SCORE: 0.92")))));
        let critic = LlmJudgeCritic::new(llm, "be polite", 0.8);
        assert!(assess_with(&critic).await.is_accepting());
    }

    #[tokio::test]
    async fn reject_below_threshold() {
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(Some(score_response("SCORE: 0.3")))));
        let critic = LlmJudgeCritic::new(llm, "be polite", 0.8);
        assert!(assess_with(&critic).await.is_rejecting());
    }

    #[tokio::test]
    async fn accept_with_note_when_no_score_parsed() {
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(Some(score_response(
            "the agent was fine",
        )))));
        let critic = LlmJudgeCritic::new(llm, "be polite", 0.8);
        let v = assess_with(&critic).await;
        // Fails open — no SCORE: line => AcceptWithNote.
        assert!(matches!(v, CriticVerdict::AcceptWithNote(_)));
    }

    #[tokio::test]
    async fn accept_with_note_when_judge_errors() {
        // One-shot scripted LLM with no response queued → first call
        // errors.
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(None)));
        let critic = LlmJudgeCritic::new(llm, "be polite", 0.8);
        let v = assess_with(&critic).await;
        assert!(matches!(v, CriticVerdict::AcceptWithNote(_)));
    }
}
