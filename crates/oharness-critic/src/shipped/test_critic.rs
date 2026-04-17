//! `TestCritic` — runs an external command (typically a test runner) and
//! reports a verdict based on its exit status. Behind the `test-runner`
//! feature.

use crate::critic::{AssessmentContext, Critic, CriticVerdict};
use async_trait::async_trait;
use std::path::PathBuf;

pub struct TestCritic {
    name: String,
    cmd: Vec<String>,
    working_dir: Option<PathBuf>,
}

impl TestCritic {
    /// Build a test critic that runs `cmd` in the given working
    /// directory. Exit 0 → `Accept`, non-zero → `Reject { reason }` with
    /// the last 2KB of stderr for context.
    pub fn new<S>(name: impl Into<String>, cmd: impl IntoIterator<Item = S>) -> Self
    where
        S: Into<String>,
    {
        Self {
            name: name.into(),
            cmd: cmd.into_iter().map(Into::into).collect(),
            working_dir: None,
        }
    }

    pub fn with_working_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }
}

#[async_trait]
impl Critic for TestCritic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn assess(&self, _ctx: &AssessmentContext<'_>) -> CriticVerdict {
        let Some((program, args)) = self.cmd.split_first() else {
            return CriticVerdict::Reject {
                reason: "TestCritic: empty command".to_string(),
            };
        };

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        if let Some(wd) = &self.working_dir {
            cmd.current_dir(wd);
        }
        cmd.kill_on_drop(true);

        let output = match cmd.output().await {
            Ok(o) => o,
            Err(e) => {
                return CriticVerdict::Reject {
                    reason: format!("TestCritic: spawn failed: {e}"),
                };
            }
        };

        if output.status.success() {
            CriticVerdict::Accept
        } else {
            let tail = last_bytes(&output.stderr, 2048);
            let code = output
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            CriticVerdict::Reject {
                reason: format!(
                    "TestCritic: `{}` exited {code}\nstderr tail:\n{}",
                    self.cmd.join(" "),
                    String::from_utf8_lossy(tail),
                ),
            }
        }
    }
}

fn last_bytes(bytes: &[u8], max: usize) -> &[u8] {
    if bytes.len() <= max {
        bytes
    } else {
        &bytes[bytes.len() - max..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::{
        AssistantTurn, ConversationView, Message, StopReason, Task, TrajectoryView, Usage,
    };

    fn dummy_ctx<'a>(task: &'a Task, turn: &'a AssistantTurn) -> AssessmentContext<'a> {
        AssessmentContext::new(
            task,
            ConversationView::new(&[]),
            turn,
            TrajectoryView::new(&[]),
        )
    }

    fn sample_turn() -> AssistantTurn {
        AssistantTurn::new(
            0,
            "span",
            Message::assistant_text("x"),
            Usage::default(),
            StopReason::EndTurn,
        )
    }

    #[tokio::test]
    async fn test_critic_accepts_on_exit_zero() {
        let critic = TestCritic::new("tests", ["true"]);
        let task = Task::new("t");
        let turn = sample_turn();
        let v = critic.assess(&dummy_ctx(&task, &turn)).await;
        assert!(v.is_accepting());
    }

    #[tokio::test]
    async fn test_critic_rejects_on_nonzero_exit() {
        let critic = TestCritic::new("tests", ["false"]);
        let task = Task::new("t");
        let turn = sample_turn();
        let v = critic.assess(&dummy_ctx(&task, &turn)).await;
        assert!(v.is_rejecting());
    }

    #[tokio::test]
    async fn test_critic_rejects_with_stderr_tail() {
        let critic = TestCritic::new("tests", ["sh", "-c", "echo boom >&2; exit 1"]);
        let task = Task::new("t");
        let turn = sample_turn();
        match critic.assess(&dummy_ctx(&task, &turn)).await {
            CriticVerdict::Reject { reason } => assert!(reason.contains("boom")),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_critic_rejects_when_spawn_fails() {
        let critic = TestCritic::new("tests", ["/this/definitely/does/not/exist-xyz"]);
        let task = Task::new("t");
        let turn = sample_turn();
        match critic.assess(&dummy_ctx(&task, &turn)).await {
            CriticVerdict::Reject { reason } => assert!(reason.contains("spawn")),
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_critic_rejects_empty_command() {
        let critic: TestCritic = TestCritic::new("tests", std::iter::empty::<String>());
        let task = Task::new("t");
        let turn = sample_turn();
        match critic.assess(&dummy_ctx(&task, &turn)).await {
            CriticVerdict::Reject { reason } => assert!(reason.contains("empty")),
            other => panic!("expected Reject, got {other:?}"),
        }
    }
}
