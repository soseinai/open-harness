//! `bash` tool kit (§7.5). Runs shell commands in the workspace directory.
//!
//! M1a: straightforward subprocess exec with stdout/stderr capture and optional
//! timeout. Not sandboxed — callers should pair this with `ApprovalChannel` or a
//! custom `ToolPolicy` for anything beyond local research.

use crate::context::ToolContext;
use crate::toolset::{ToolOutcome, ToolSet};
use async_trait::async_trait;
use oharness_core::message::{Content, ToolOutput};
use oharness_core::ToolSpec;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// A single-tool `ToolSet` exposing `bash`.
pub struct BashTool {
    name: String,
    timeout: Duration,
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
            specs,
        }
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
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

        let timeout_dur = parsed
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);

        let exec = async {
            let output = cmd.output().await?;
            Ok::<_, std::io::Error>(output)
        };

        let output = match timeout(timeout_dur, exec).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return ToolOutcome::error(format!("bash: {e}"), true),
            Err(_) => {
                return ToolOutcome::error(
                    format!("bash: timed out after {}s", timeout_dur.as_secs()),
                    true,
                );
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
