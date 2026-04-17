//! `fs` tool kit (§7.5). read/write/list scoped to a `Workspace` (or cwd if none).
//!
//! All paths resolve relative to the workspace root; absolute paths that escape the
//! workspace are rejected with `ExecutionError { recoverable: false }`.

use crate::context::ToolContext;
use crate::toolset::{ToolOutcome, ToolSet};
use async_trait::async_trait;
use oharness_core::ToolSpec;
use oharness_core::message::{Content, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MAX_READ_BYTES: u64 = 1024 * 1024; // 1MiB

/// Bundle of small fs tools: `fs_read`, `fs_write`, `fs_list`.
pub struct FsToolSet {
    specs: Vec<ToolSpec>,
}

impl Default for FsToolSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FsToolSet {
    pub fn new() -> Self {
        Self {
            specs: vec![
                ToolSpec {
                    name: "fs_read".to_string(),
                    description: "Read a UTF-8 text file relative to the workspace root. \
                                  Returns the file's contents (max 1MiB)."
                        .to_string(),
                    input_schema: read_schema(),
                },
                ToolSpec {
                    name: "fs_write".to_string(),
                    description: "Write UTF-8 text to a file relative to the workspace root. \
                                  Overwrites if the file exists; creates parent directories \
                                  as needed."
                        .to_string(),
                    input_schema: write_schema(),
                },
                ToolSpec {
                    name: "fs_list".to_string(),
                    description: "List entries in a directory relative to the workspace root."
                        .to_string(),
                    input_schema: list_schema(),
                },
            ],
        }
    }
}

#[async_trait]
impl ToolSet for FsToolSet {
    fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    async fn execute(&self, name: &str, input: Value, ctx: &ToolContext) -> ToolOutcome {
        if ctx.cancellation.is_cancelled() {
            return ToolOutcome::Cancelled;
        }
        let root = ctx
            .workspace_path()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        match name {
            "fs_read" => do_read(input, &root).await,
            "fs_write" => do_write(input, &root).await,
            "fs_list" => do_list(input, &root).await,
            other => ToolOutcome::error(format!("unknown fs tool `{other}`"), false),
        }
    }
}

async fn do_read(input: Value, root: &Path) -> ToolOutcome {
    #[derive(Deserialize)]
    struct ReadInput {
        path: String,
    }
    let parsed: ReadInput = match serde_json::from_value(input) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::error(format!("invalid fs_read input: {e}"), false),
    };

    let resolved = match resolve(root, &parsed.path) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::error(e, false),
    };

    let metadata = match tokio::fs::metadata(&resolved).await {
        Ok(m) => m,
        Err(e) => return ToolOutcome::error(format!("fs_read stat: {e}"), true),
    };
    if !metadata.is_file() {
        return ToolOutcome::error(format!("fs_read: `{}` is not a file", parsed.path), false);
    }
    if metadata.len() > MAX_READ_BYTES {
        return ToolOutcome::error(
            format!(
                "fs_read: `{}` is {} bytes; max {MAX_READ_BYTES}",
                parsed.path,
                metadata.len()
            ),
            false,
        );
    }

    match tokio::fs::read_to_string(&resolved).await {
        Ok(contents) => ToolOutcome::Success(ToolOutput {
            content: vec![Content::Text(contents)],
            truncated: false,
        }),
        Err(e) => ToolOutcome::error(format!("fs_read: {e}"), true),
    }
}

async fn do_write(input: Value, root: &Path) -> ToolOutcome {
    #[derive(Deserialize)]
    struct WriteInput {
        path: String,
        content: String,
    }
    let parsed: WriteInput = match serde_json::from_value(input) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::error(format!("invalid fs_write input: {e}"), false),
    };

    let resolved = match resolve(root, &parsed.path) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::error(e, false),
    };

    if let Some(parent) = resolved.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            return ToolOutcome::error(format!("fs_write mkdir: {e}"), true);
        }
    }

    match tokio::fs::write(&resolved, parsed.content.as_bytes()).await {
        Ok(()) => ToolOutcome::success_text(format!(
            "wrote {} bytes to {}",
            parsed.content.len(),
            parsed.path
        )),
        Err(e) => ToolOutcome::error(format!("fs_write: {e}"), true),
    }
}

async fn do_list(input: Value, root: &Path) -> ToolOutcome {
    #[derive(Deserialize)]
    struct ListInput {
        #[serde(default = "default_dot")]
        path: String,
    }
    fn default_dot() -> String {
        ".".to_string()
    }
    let parsed: ListInput = match serde_json::from_value(input) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::error(format!("invalid fs_list input: {e}"), false),
    };

    let resolved = match resolve(root, &parsed.path) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::error(e, false),
    };

    let mut entries = match tokio::fs::read_dir(&resolved).await {
        Ok(r) => r,
        Err(e) => return ToolOutcome::error(format!("fs_list: {e}"), true),
    };

    let mut names: Vec<String> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry
            .file_type()
            .await
            .map(|ft| ft.is_dir())
            .unwrap_or(false);
        names.push(if is_dir { format!("{name}/") } else { name });
    }
    names.sort();

    ToolOutcome::success_text(names.join("\n"))
}

/// Resolve a user-provided path relative to `root`, rejecting anything that escapes
/// the workspace (after canonicalization).
fn resolve(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let candidate = root.join(rel);
    // We don't canonicalize (the target may not exist yet for write), so instead
    // normalize and then check the prefix.
    let normalized = normalize(&candidate);
    if !normalized.starts_with(root) {
        return Err(format!(
            "path `{rel}` escapes workspace root `{}`",
            root.display()
        ));
    }
    Ok(normalized)
}

/// Simple path normalization (no I/O): collapses `..` / `.` components.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in p.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    out
}

fn read_schema() -> Value {
    static S: OnceLock<Value> = OnceLock::new();
    S.get_or_init(|| {
        json!({
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace root."}
            },
            "additionalProperties": false
        })
    })
    .clone()
}

fn write_schema() -> Value {
    static S: OnceLock<Value> = OnceLock::new();
    S.get_or_init(|| {
        json!({
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace root."},
                "content": {"type": "string", "description": "UTF-8 text to write."}
            },
            "additionalProperties": false
        })
    })
    .clone()
}

fn list_schema() -> Value {
    static S: OnceLock<Value> = OnceLock::new();
    S.get_or_init(|| {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to workspace root (default `.`)."}
            },
            "additionalProperties": false
        })
    })
    .clone()
}
