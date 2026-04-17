//! `RegexDenyCritic` — reject if the assistant output matches any of the
//! configured regex patterns (e.g. "no `eval()`", "no credit-card numbers").
//! Behind the `regex-deny` feature.

use crate::critic::{AssessmentContext, Critic, CriticVerdict};
use async_trait::async_trait;
use oharness_core::{Content, Message};

pub struct RegexDenyCritic {
    name: String,
    patterns: Vec<regex::Regex>,
}

impl RegexDenyCritic {
    /// Build a critic that rejects any assistant turn whose rendered text
    /// matches any of `patterns`.
    pub fn new<I, S>(name: impl Into<String>, patterns: I) -> Result<Self, regex::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let compiled = patterns
            .into_iter()
            .map(|p| regex::Regex::new(p.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name: name.into(),
            patterns: compiled,
        })
    }

    fn pattern_strs(&self) -> Vec<&str> {
        self.patterns.iter().map(|p| p.as_str()).collect()
    }
}

fn extract_text(message: &Message) -> String {
    let Message::Assistant { content, .. } = message else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            Content::Thinking { thinking } => Some(thinking.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl Critic for RegexDenyCritic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        let text = extract_text(&ctx.latest_turn.message);
        for pat in &self.patterns {
            if pat.is_match(&text) {
                return CriticVerdict::Reject {
                    reason: format!("regex-deny: matched `{}`", pat.as_str()),
                };
            }
        }
        CriticVerdict::Accept
    }
}

impl std::fmt::Debug for RegexDenyCritic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexDenyCritic")
            .field("name", &self.name)
            .field("patterns", &self.pattern_strs())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::{
        AssistantTurn, ConversationView, Message, StopReason, Task, TrajectoryView, Usage,
    };

    fn assess_with_text(critic: &RegexDenyCritic, assistant_text: &str) -> CriticVerdict {
        let task = Task::new("t");
        let msg = Message::assistant_text(assistant_text);
        let turn = AssistantTurn::new(0, "span", msg, Usage::default(), StopReason::EndTurn);
        let ctx = AssessmentContext::new(
            &task,
            ConversationView::new(&[]),
            &turn,
            TrajectoryView::new(&[]),
        );
        futures::executor::block_on(critic.assess(&ctx))
    }

    #[test]
    fn accepts_text_without_forbidden_patterns() {
        let critic = RegexDenyCritic::new("c", ["eval\\(", "exec\\("]).unwrap();
        let v = assess_with_text(&critic, "let x = 1 + 2;");
        assert!(v.is_accepting());
    }

    #[test]
    fn rejects_text_matching_first_pattern() {
        let critic = RegexDenyCritic::new("c", ["eval\\(", "exec\\("]).unwrap();
        match assess_with_text(&critic, "running eval(x) here") {
            CriticVerdict::Reject { reason } => assert!(reason.contains("eval")),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn rejects_text_matching_later_pattern() {
        let critic = RegexDenyCritic::new("c", ["eval\\(", "exec\\("]).unwrap();
        match assess_with_text(&critic, "exec('rm -rf /')") {
            CriticVerdict::Reject { reason } => assert!(reason.contains("exec")),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn invalid_pattern_surfaces_regex_error() {
        match RegexDenyCritic::new("c", ["[unclosed"]) {
            Err(_) => {}
            Ok(_) => panic!("should have failed to compile regex"),
        }
    }
}
