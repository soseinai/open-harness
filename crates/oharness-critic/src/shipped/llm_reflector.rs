//! `LlmReflector` — calls an LLM with a templated prompt, returns the
//! response text as the reflection body.

use crate::reflector::Reflector;
use async_trait::async_trait;
use oharness_core::{CompletionRequest, Content, Episode, Message, Reflection};
use oharness_llm::Llm;
use std::sync::Arc;

/// Reflector that prompts an LLM for a reflection. The template is a
/// plain-string format with `{key}` placeholders:
///
/// - `{task}` — `task.instruction`
/// - `{termination}` — `Debug` rendering of `outcome.termination`
/// - `{turns}` — `outcome.usage.turns`
/// - `{score}` — `evaluation.score`
/// - `{passed}` — `evaluation.passed`
/// - `{prior_reflections}` — numbered, one-per-line rendering of
///   previous reflections (or `"(none)"` on the first episode)
///
/// Unknown placeholders are left verbatim so users can embed raw `{x}`
/// tokens in prose without escaping, as long as they don't collide with
/// the known keys.
pub struct LlmReflector {
    llm: Arc<dyn Llm>,
    template: String,
    name: String,
}

impl LlmReflector {
    pub fn new(llm: Arc<dyn Llm>, template: impl Into<String>) -> Self {
        Self {
            llm,
            template: template.into(),
            name: "llm".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Default template that asks the model to critique the failed
    /// attempt and return a short, actionable reflection. Useful for
    /// bootstrapping — callers who want full control supply their own.
    pub fn default_template() -> &'static str {
        "The agent attempted the following task:\n\n\
         {task}\n\n\
         Outcome: {termination} (turns={turns}, score={score}, passed={passed}).\n\n\
         Prior reflections:\n{prior_reflections}\n\n\
         Write a single-paragraph reflection that names one concrete thing the \
         next attempt should do differently. Be specific; do not restate the task."
    }
}

fn render_template(template: &str, episode: &Episode<'_>) -> String {
    let prior = if episode.prior_reflections.is_empty() {
        "(none)".to_string()
    } else {
        episode
            .prior_reflections
            .iter()
            .enumerate()
            .map(|(i, r)| format!("  {}. {}", i + 1, r.text))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut out = template.to_string();
    for (key, value) in [
        ("{task}", episode.task.instruction.as_str().to_string()),
        (
            "{termination}",
            format!("{:?}", episode.outcome.termination),
        ),
        ("{turns}", episode.outcome.usage.turns.to_string()),
        ("{score}", format!("{:.4}", episode.evaluation.score)),
        ("{passed}", episode.evaluation.passed.to_string()),
        ("{prior_reflections}", prior),
    ] {
        out = out.replace(key, &value);
    }
    out
}

#[async_trait]
impl Reflector for LlmReflector {
    fn name(&self) -> &str {
        &self.name
    }

    async fn reflect(&self, episode: &Episode<'_>) -> Option<Reflection> {
        let prompt = render_template(&self.template, episode);
        let req = CompletionRequest::new(vec![Message::user_text(prompt)]);
        match self.llm.complete(req).await {
            Ok(res) => extract_text(&res.content).map(Reflection::new),
            Err(e) => {
                tracing::warn!(
                    target: "oharness.critic.reflector",
                    error = %e,
                    "LlmReflector.complete failed; skipping reflection",
                );
                None
            }
        }
    }
}

fn extract_text(content: &[Content]) -> Option<String> {
    let joined = content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if joined.trim().is_empty() {
        None
    } else {
        Some(joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oharness_core::{
        CompletionReason, CompletionResponse, EvaluationResult, LlmCapabilities, MetadataMap,
        ModelId, ResourceUsage, RunOutcome, StopReason, Task, Termination, TrajectoryHandle, Usage,
    };
    use oharness_llm::{ChunkStream, LlmError};
    use std::sync::Mutex;

    struct Scripted(Mutex<Option<CompletionResponse>>);
    #[async_trait]
    impl Llm for Scripted {
        fn name(&self) -> &str {
            "scripted"
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

    fn dummy_outcome() -> RunOutcome {
        RunOutcome {
            run_id: oharness_core::RunId::new(),
            task_id: Some("task-1".to_string()),
            termination: Termination::Completed {
                reason: CompletionReason::EndTurn,
            },
            final_messages: Vec::new(),
            trajectory: TrajectoryHandle::in_memory(Vec::new()),
            usage: ResourceUsage {
                turns: 4,
                ..Default::default()
            },
            per_model_usage: Default::default(),
            started_at: time::OffsetDateTime::now_utc(),
            finished_at: time::OffsetDateTime::now_utc(),
            agent_state: MetadataMap::new(),
        }
    }

    #[tokio::test]
    async fn llm_reflector_returns_response_text() {
        let response = CompletionResponse {
            id: "r".into(),
            model: ModelId::new("m"),
            content: vec![Content::text("Next time, run tests first.")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(Some(response))));
        let reflector = LlmReflector::new(llm, LlmReflector::default_template());
        let task = Task::new("run the tests");
        let outcome = dummy_outcome();
        let eval = EvaluationResult::fail();
        let ep = Episode {
            index: 0,
            task: &task,
            outcome: &outcome,
            evaluation: &eval,
            prior_reflections: &[],
        };
        let reflection = reflector.reflect(&ep).await.expect("reflection");
        assert_eq!(reflection.text, "Next time, run tests first.");
    }

    #[tokio::test]
    async fn llm_reflector_returns_none_on_empty_text() {
        let response = CompletionResponse {
            id: "r".into(),
            model: ModelId::new("m"),
            content: vec![Content::text("")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(Some(response))));
        let reflector = LlmReflector::new(llm, "x");
        let task = Task::new("t");
        let outcome = dummy_outcome();
        let eval = EvaluationResult::pass();
        let ep = Episode {
            index: 0,
            task: &task,
            outcome: &outcome,
            evaluation: &eval,
            prior_reflections: &[],
        };
        assert!(reflector.reflect(&ep).await.is_none());
    }

    #[tokio::test]
    async fn llm_reflector_returns_none_on_llm_error() {
        let llm: Arc<dyn Llm> = Arc::new(Scripted(Mutex::new(None)));
        let reflector = LlmReflector::new(llm, "x");
        let task = Task::new("t");
        let outcome = dummy_outcome();
        let eval = EvaluationResult::fail();
        let ep = Episode {
            index: 0,
            task: &task,
            outcome: &outcome,
            evaluation: &eval,
            prior_reflections: &[],
        };
        assert!(reflector.reflect(&ep).await.is_none());
    }

    #[test]
    fn render_template_substitutes_known_placeholders() {
        let task = Task::new("fix the bug");
        let outcome = dummy_outcome();
        let eval = EvaluationResult::scored(0.75);
        let ep = Episode {
            index: 0,
            task: &task,
            outcome: &outcome,
            evaluation: &eval,
            prior_reflections: &[],
        };
        let rendered = render_template(
            "task={task} turns={turns} score={score} passed={passed}",
            &ep,
        );
        assert_eq!(
            rendered,
            "task=fix the bug turns=4 score=0.7500 passed=true"
        );
    }

    #[test]
    fn render_template_numbers_prior_reflections() {
        let task = Task::new("t");
        let outcome = dummy_outcome();
        let eval = EvaluationResult::fail();
        let prior = vec![Reflection::new("alpha"), Reflection::new("beta")];
        let ep = Episode {
            index: 2,
            task: &task,
            outcome: &outcome,
            evaluation: &eval,
            prior_reflections: &prior,
        };
        let rendered = render_template("{prior_reflections}", &ep);
        assert!(rendered.contains("1. alpha"));
        assert!(rendered.contains("2. beta"));
    }
}
