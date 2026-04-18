//! `bash` tool kit (§7.5). Runs shell commands in the workspace directory.
//!
//! M1a: straightforward subprocess exec with stdout/stderr capture and optional
//! timeout. Not sandboxed — callers should pair this with `ApprovalChannel` or a
//! custom `ToolPolicy` for anything beyond local research. The full threat
//! model, what's mitigated here vs. what isn't, and deployer
//! recommendations live in `docs/security.md`.

use crate::context::ToolContext;
use crate::toolset::{ToolOutcome, ToolSet};
use async_trait::async_trait;
use oharness_core::message::{Content, ToolOutput};
use oharness_core::ToolSpec;
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Command;

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// A single-tool `ToolSet` exposing `bash`.
pub struct BashTool {
    name: String,
    timeout: Duration,
    /// Optional env-var allowlist. When `Some(names)`, the
    /// subprocess starts with a cleared environment and only the
    /// listed variables are copied over from the parent. When
    /// `None`, the subprocess inherits the full parent environment
    /// (backwards-compatible default). See `docs/security.md` §3.3
    /// for recommended allowlists.
    env_allowlist: Option<Vec<String>>,
    specs: Vec<ToolSpec>,
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new("bash")
    }
}

impl BashTool {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let specs = vec![ToolSpec {
            name: name.clone(),
            description: "Execute a shell command via `/bin/bash -c <command>`. Returns \
                          combined stdout/stderr. Commands run in the configured \
                          workspace directory, or the current directory if no workspace \
                          is set. Output is truncated at 64KiB."
                .to_string(),
            input_schema: default_schema(),
        }];
        Self {
            name,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            env_allowlist: None,
            specs,
        }
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Restrict the subprocess environment to the listed variables.
    /// The child starts with no env, then the allowlisted names are
    /// copied over from the parent's env (silently skipped if
    /// unset). Recommended for eval / CI / untrusted-LLM work:
    ///
    /// ```ignore
    /// let tool = BashTool::default()
    ///     .with_env_allowlist(["PATH", "HOME", "USER", "SHELL", "LANG"]);
    /// ```
    ///
    /// This hides `*_API_KEY`, `AWS_*`, `ANTHROPIC_*`, SSH agent
    /// sockets, and similar secrets from the subprocess. Without
    /// an allowlist the subprocess inherits the full environment.
    pub fn with_env_allowlist<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.env_allowlist = Some(names.into_iter().map(Into::into).collect());
        self
    }
}

#[async_trait]
impl ToolSet for BashTool {
    fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutcome {
        if name != self.name {
            return ToolOutcome::error(format!("tool `{name}` not handled by BashTool"), false);
        }
        if ctx.cancellation.is_cancelled() {
            return ToolOutcome::Cancelled;
        }

        let parsed: BashInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolOutcome::error(format!("invalid bash input: {e}"), false),
        };

        let mut cmd = Command::new("/bin/bash");
        cmd.arg("-c").arg(&parsed.command);
        if let Some(ws) = ctx.workspace_path() {
            cmd.current_dir(ws);
        }

        // Env-allowlist filtering. Default (`None`) keeps the
        // full parent env for backwards-compat; an allowlist
        // clears env first, then copies over the named vars.
        if let Some(names) = &self.env_allowlist {
            cmd.env_clear();
            for name in names {
                if let Ok(val) = std::env::var(name) {
                    cmd.env(name, val);
                }
            }
        }

        // Capture stdout + stderr via pipes (`Command::spawn` by
        // default inherits the parent's, which leaks output into
        // the harness's own stdio).
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // Don't let the subprocess read from the parent stdin —
        // some commands (e.g. `cat`) would otherwise block waiting
        // for input that never arrives.
        cmd.stdin(Stdio::null());

        // Ensure the child dies if the enclosing future is dropped —
        // e.g. on timeout, cancellation, or the caller drop-completing
        // a task that spawned the tool call. Without this, a timed-out
        // bash command leaks a background process. Audit notes:
        // `docs/security.md` §3.1.
        cmd.kill_on_drop(true);

        let timeout_dur = parsed
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);

        // Run the child, racing three things:
        // 1. the child finishing normally,
        // 2. the timeout firing,
        // 3. `ctx.cancellation` being signalled.
        // Because `kill_on_drop(true)` is set, dropping the Child on
        // any of (2)/(3) sends SIGKILL; we don't need a manual kill.
        let cancellation = ctx.cancellation.clone();
        let output = {
            let child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => return ToolOutcome::error(format!("bash spawn: {e}"), true),
            };
            tokio::select! {
                res = child.wait_with_output() => match res {
                    Ok(o) => o,
                    Err(e) => return ToolOutcome::error(format!("bash: {e}"), true),
                },
                _ = tokio::time::sleep(timeout_dur) => {
                    // child drops here; kill_on_drop reaps it.
                    return ToolOutcome::error(
                        format!("bash: timed out after {}s", timeout_dur.as_secs()),
                        true,
                    );
                }
                _ = cancellation.cancelled() => {
                    return ToolOutcome::Cancelled;
                }
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code();

        let mut combined = String::new();
        if !stdout.is_empty() {
            combined.push_str("STDOUT:\n");
            combined.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push_str("\n\n");
            }
            combined.push_str("STDERR:\n");
            combined.push_str(&stderr);
        }
        let (combined, truncated) = if combined.len() > MAX_OUTPUT_BYTES {
            (
                format!(
                    "{}\n\n[truncated at {MAX_OUTPUT_BYTES} bytes]",
                    &combined[..MAX_OUTPUT_BYTES]
                ),
                true,
            )
        } else {
            (combined, false)
        };

        let tail = match code {
            Some(0) => String::new(),
            Some(c) => format!("\n\n[exit code: {c}]"),
            None => "\n\n[exit: killed by signal]".to_string(),
        };

        ToolOutcome::Success(ToolOutput {
            content: vec![Content::text(format!("{combined}{tail}"))],
            truncated,
        })
    }
}

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

fn default_schema() -> Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Optional per-call timeout in seconds.",
                        "minimum": 1
                    }
                },
                "additionalProperties": false
            })
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn context() -> ToolContext {
        ToolContext::null()
    }

    fn outcome_text(outcome: &ToolOutcome) -> String {
        match outcome {
            ToolOutcome::Success(output) => output
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            ToolOutcome::ExecutionError { message, .. } => message.clone(),
            ToolOutcome::Denied { reason } => reason.clone(),
            ToolOutcome::Cancelled => String::from("<cancelled>"),
        }
    }

    #[tokio::test]
    async fn happy_path_captures_stdout() {
        let tool = BashTool::default();
        let outcome = tool
            .execute("bash", json!({"command": "echo hello world"}), &context())
            .await;
        assert!(matches!(outcome, ToolOutcome::Success(_)));
        let text = outcome_text(&outcome);
        assert!(text.contains("hello world"), "missing stdout: {text}");
    }

    /// Regression: pre-audit, a timed-out bash command leaked a
    /// background process because `timeout()` dropped the child
    /// without killing it. With `kill_on_drop(true)`, the child
    /// dies when the future drops — verified by the timeout
    /// returning in ~1s even though the command asked for 30s.
    #[tokio::test]
    async fn timeout_kills_subprocess_not_leaks_it() {
        let tool = BashTool::default().with_timeout(Duration::from_secs(1));
        let start = Instant::now();
        let outcome = tool
            .execute("bash", json!({"command": "sleep 30"}), &context())
            .await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "bash did not return promptly on timeout: took {elapsed:?}"
        );
        match outcome {
            ToolOutcome::ExecutionError { message, .. } => {
                assert!(message.contains("timed out"), "{message}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    /// Cancellation via `ctx.cancellation` fires mid-execution
    /// (not just pre-spawn) — the tool polls the token concurrently
    /// via `tokio::select!`.
    #[tokio::test]
    async fn cancellation_interrupts_running_command() {
        let tool = BashTool::default().with_timeout(Duration::from_secs(30));
        let mut ctx = ToolContext::null();
        let token = ctx.cancellation.clone();
        // Cancel after 200ms — mid-sleep.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            token.cancel();
        });
        // Reassign to capture the token's cancel trigger — the
        // cancellation token shared with the background task IS
        // the same token ctx holds.
        ctx.cancellation = ctx.cancellation.clone();

        let start = Instant::now();
        let outcome = tool
            .execute("bash", json!({"command": "sleep 30"}), &ctx)
            .await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "cancellation was not prompt: took {elapsed:?}"
        );
        assert!(matches!(outcome, ToolOutcome::Cancelled), "got {outcome:?}");
    }

    /// Env allowlist: when set, the subprocess starts with a
    /// cleared env and only the listed vars propagate. Use a
    /// distinctive var that the parent sets but we don't allow —
    /// `env` should NOT show it in the child's output.
    #[tokio::test]
    async fn env_allowlist_hides_unlisted_vars() {
        // Set a parent env var the subprocess shouldn't see.
        // Safety note: modifying the process env isn't ideal in
        // parallel tests, but the child observes a snapshot at
        // spawn. Other tests don't read this key.
        // SAFETY: in Rust 2024+ `std::env::set_var` is unsafe; on
        // our pinned toolchain it's still safe.
        std::env::set_var("OHARNESS_BASH_TEST_SECRET", "should-not-leak");

        let tool = BashTool::default().with_env_allowlist(["PATH", "HOME"]);
        let outcome = tool
            .execute("bash", json!({"command": "env"}), &context())
            .await;
        let text = outcome_text(&outcome);
        assert!(
            !text.contains("OHARNESS_BASH_TEST_SECRET"),
            "secret env var leaked through allowlist: {text}"
        );

        // Clean up.
        std::env::remove_var("OHARNESS_BASH_TEST_SECRET");
    }

    /// Without an allowlist, the subprocess inherits the full
    /// env (backwards-compatible default).
    #[tokio::test]
    async fn no_allowlist_inherits_env() {
        std::env::set_var("OHARNESS_BASH_PASSTHROUGH", "visible");

        let tool = BashTool::default();
        let outcome = tool
            .execute("bash", json!({"command": "env"}), &context())
            .await;
        let text = outcome_text(&outcome);
        assert!(
            text.contains("OHARNESS_BASH_PASSTHROUGH"),
            "expected env var to passthrough without allowlist: {text}"
        );

        std::env::remove_var("OHARNESS_BASH_PASSTHROUGH");
    }

    /// Output over 64KiB is truncated, and the `truncated` flag
    /// is set on the ToolOutput.
    #[tokio::test]
    async fn large_output_is_truncated_flagged() {
        let tool = BashTool::default();
        // Produce ~200KB of output — well past the 64KB cap.
        let outcome = tool
            .execute(
                "bash",
                json!({"command": "yes foo | head -c 200000"}),
                &context(),
            )
            .await;
        match outcome {
            ToolOutcome::Success(output) => {
                assert!(output.truncated, "truncated flag not set");
                let text = output
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert!(text.contains("truncated at"), "missing truncation marker");
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }
}
