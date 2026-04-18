//! SWE-bench dataset types + loader.
//!
//! The record shape matches the canonical SWE-bench JSON schema:
//!
//! ```ignore
//! {
//!   "instance_id": "astropy__astropy-12907",
//!   "repo": "astropy/astropy",
//!   "base_commit": "abc123…",
//!   "patch": "…gold patch text…",
//!   "test_patch": "…test-only patch enabling FAIL_TO_PASS tests…",
//!   "problem_statement": "…",
//!   "hints_text": "…",
//!   "version": "4.3",
//!   "environment_setup_commit": "…",
//!   "FAIL_TO_PASS": ["astropy/wcs/tests/test_wcs.py::test_foo"],
//!   "PASS_TO_PASS": ["…", "…"]
//! }
//! ```
//!
//! Only `instance_id`, `repo`, `base_commit`, `test_patch`,
//! `problem_statement`, `FAIL_TO_PASS`, and `PASS_TO_PASS` are used at
//! runtime; the rest are passed through via `Task::metadata` so
//! downstream critics / reflectors can read them.

use crate::evaluator::SweBenchEvaluator;
use async_trait::async_trait;
use oharness_core::{MetadataMap, Task, TaskEvaluator};
use oharness_eval::{Benchmark, BenchmarkError, LoadedTask, Workspace};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Supported SWE-bench dataset versions. The adapter's behavior is
/// identical across `Lite` and `Full`; this enum only participates in
/// the `Benchmark::name()` / `version()` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweBenchVariant {
    Lite,
    Full,
}

impl SweBenchVariant {
    pub fn name(&self) -> &'static str {
        match self {
            SweBenchVariant::Lite => "swe-bench-lite",
            SweBenchVariant::Full => "swe-bench-full",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweBenchInstance {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    #[serde(default)]
    pub patch: String,
    pub test_patch: String,
    pub problem_statement: String,
    #[serde(default)]
    pub hints_text: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub environment_setup_commit: String,
    #[serde(rename = "FAIL_TO_PASS", alias = "fail_to_pass", default)]
    pub fail_to_pass: Vec<String>,
    #[serde(rename = "PASS_TO_PASS", alias = "pass_to_pass", default)]
    pub pass_to_pass: Vec<String>,
}

impl SweBenchInstance {
    /// Render the task instruction shown to the agent — the problem
    /// statement prefixed by a short orientation. Kept deterministic so
    /// trajectory replays are stable.
    pub fn task_instruction(&self) -> String {
        format!(
            "You are working in a clone of {repo} at commit {commit}. \
             Resolve the following issue by editing files in the \
             current workspace. Tests in FAIL_TO_PASS must go from \
             failing to passing, and tests in PASS_TO_PASS must remain \
             passing.\n\n# Problem\n\n{problem}",
            repo = self.repo,
            commit = self.base_commit,
            problem = self.problem_statement,
        )
    }

    /// Stash the full instance record in `Task::metadata` under
    /// `swe-bench.instance` so critics / reflectors can read it.
    pub fn to_task(&self) -> Task {
        let mut task = Task::new(self.task_instruction()).with_id(self.instance_id.clone());
        // Non-fatal if serialization fails — metadata is best-effort.
        if let Ok(val) = serde_json::to_value(self) {
            if let Some(obj) = val.as_object() {
                let mut md = MetadataMap::new();
                md.insert(
                    "swe-bench.instance".to_string(),
                    serde_json::Value::Object(obj.clone()),
                );
                task.metadata = md;
            }
        }
        task
    }
}

// ======================================================================
// SweBenchLite
// ======================================================================

pub struct SweBenchLite {
    variant: SweBenchVariant,
    dataset_version: String,
    instances: Vec<SweBenchInstance>,
    /// Scratch root for per-task checkouts. Each task clones into
    /// `{clone_root}/{instance_id}/`.
    clone_root: PathBuf,
    /// Evaluator config shared by every task. Exposed here rather than
    /// stashed on each `LoadedTask` because every task evaluates the
    /// same way; per-task differences ride on the `SweBenchInstance`
    /// that's already in `Task::metadata`.
    evaluator: Arc<SweBenchEvaluator>,
}

impl SweBenchLite {
    /// Load instances from a JSONL dump — one JSON record per line,
    /// matching the shape returned by HuggingFace's dataset viewer.
    ///
    /// Users download the dump themselves; runtime fetching from the
    /// HuggingFace datasets API is a later enhancement.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self, BenchmarkError> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let mut instances = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let instance: SweBenchInstance = serde_json::from_str(line).map_err(|e| {
                BenchmarkError::Other(format!(
                    "line {}: {e} (first 200 chars: {})",
                    i + 1,
                    &line[..line.len().min(200)]
                ))
            })?;
            instances.push(instance);
        }
        Ok(Self::from_instances(instances))
    }

    /// Build directly from already-loaded instances. Handy for tests
    /// and for users who load the dataset through a different channel.
    pub fn from_instances(instances: Vec<SweBenchInstance>) -> Self {
        let clone_root = std::env::temp_dir().join("oharness-swe-bench");
        Self {
            variant: SweBenchVariant::Lite,
            dataset_version: "swe-bench-lite-unversioned".to_string(),
            instances,
            clone_root,
            evaluator: Arc::new(SweBenchEvaluator::default()),
        }
    }

    pub fn with_variant(mut self, variant: SweBenchVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn with_dataset_version(mut self, version: impl Into<String>) -> Self {
        self.dataset_version = version.into();
        self
    }

    pub fn with_clone_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.clone_root = root.into();
        self
    }

    pub fn with_evaluator(mut self, evaluator: SweBenchEvaluator) -> Self {
        self.evaluator = Arc::new(evaluator);
        self
    }

    pub fn instances(&self) -> &[SweBenchInstance] {
        &self.instances
    }
}

#[async_trait]
impl Benchmark for SweBenchLite {
    fn name(&self) -> &str {
        self.variant.name()
    }

    fn version(&self) -> &str {
        &self.dataset_version
    }

    fn task_count(&self) -> Option<usize> {
        Some(self.instances.len())
    }

    fn task_ids(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        Box::new(self.instances.iter().map(|i| i.instance_id.clone()))
    }

    async fn load_task(&self, id: &str) -> Result<LoadedTask, BenchmarkError> {
        let instance = self
            .instances
            .iter()
            .find(|i| i.instance_id == id)
            .ok_or_else(|| BenchmarkError::TaskNotFound(id.to_string()))?;

        std::fs::create_dir_all(&self.clone_root)?;
        // `{clone_root}/{instance_id}/` — sanitise because some SWE-bench
        // instance ids contain `/`.
        let per_task_dir = self
            .clone_root
            .join(sanitize_instance_id(&instance.instance_id));

        // Idempotent: if the directory already exists (e.g. from a
        // previous run), start from scratch so we pick up the correct
        // base_commit.
        if per_task_dir.exists() {
            std::fs::remove_dir_all(&per_task_dir)?;
        }
        std::fs::create_dir_all(&per_task_dir)?;

        clone_and_checkout(&instance.repo, &instance.base_commit, &per_task_dir).await?;

        let workspace_path = per_task_dir.clone();
        let workspace = Workspace::new(workspace_path).with_sync_cleanup(move || {
            // Best-effort cleanup. Failures just leak the dir — not
            // fatal for benchmark runs, which typically run out of a
            // one-shot scratch area anyway.
            let _ = std::fs::remove_dir_all(&per_task_dir);
        });

        Ok(LoadedTask::new(instance.to_task()).with_workspace(Arc::new(workspace)))
    }

    fn evaluator(&self) -> Arc<dyn TaskEvaluator> {
        self.evaluator.clone()
    }
}

async fn clone_and_checkout(
    repo: &str,
    base_commit: &str,
    dest: &Path,
) -> Result<(), BenchmarkError> {
    let url = github_clone_url(repo);
    let clone_status = tokio::process::Command::new("git")
        .args(["clone", "--quiet", &url, "."])
        .current_dir(dest)
        .kill_on_drop(true)
        .status()
        .await
        .map_err(|e| BenchmarkError::Other(format!("git clone spawn: {e}")))?;
    if !clone_status.success() {
        return Err(BenchmarkError::Other(format!(
            "git clone `{url}` failed ({})",
            status_code_str(&clone_status)
        )));
    }

    let checkout_status = tokio::process::Command::new("git")
        .args(["checkout", "--quiet", base_commit])
        .current_dir(dest)
        .kill_on_drop(true)
        .status()
        .await
        .map_err(|e| BenchmarkError::Other(format!("git checkout spawn: {e}")))?;
    if !checkout_status.success() {
        return Err(BenchmarkError::Other(format!(
            "git checkout `{base_commit}` failed ({})",
            status_code_str(&checkout_status)
        )));
    }
    Ok(())
}

fn github_clone_url(repo: &str) -> String {
    // SWE-bench repo slugs are `owner/name`; default remote is GitHub.
    // Users who want to mirror to an internal Gerrit / self-hosted
    // Git can subclass by calling `from_instances` and then editing
    // the SweBenchInstance.repo fields to full URLs. Absolute local
    // paths also pass through verbatim — used by the test fixture,
    // handy for mirror-setups that want offline runs.
    if repo.starts_with("http://")
        || repo.starts_with("https://")
        || repo.starts_with("git@")
        || repo.starts_with('/')
        || std::path::Path::new(repo).is_absolute()
    {
        repo.to_string()
    } else {
        format!("https://github.com/{repo}.git")
    }
}

fn sanitize_instance_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn status_code_str(status: &std::process::ExitStatus) -> String {
    match status.code() {
        Some(c) => format!("exit {c}"),
        None => "killed by signal".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_clone_url_handles_slug_and_url_forms() {
        assert_eq!(
            github_clone_url("astropy/astropy"),
            "https://github.com/astropy/astropy.git"
        );
        assert_eq!(
            github_clone_url("https://example.com/fork/astropy.git"),
            "https://example.com/fork/astropy.git"
        );
        assert_eq!(
            github_clone_url("git@github.com:astropy/astropy.git"),
            "git@github.com:astropy/astropy.git"
        );
        // Absolute local paths pass through — test fixtures use this.
        assert_eq!(github_clone_url("/tmp/upstream.git"), "/tmp/upstream.git");
    }

    #[test]
    fn sanitize_instance_id_replaces_unsafe_chars() {
        assert_eq!(
            sanitize_instance_id("astropy__astropy-12907"),
            "astropy__astropy-12907"
        );
        assert_eq!(sanitize_instance_id("owner/name-42"), "owner_name-42");
    }

    #[test]
    fn task_instruction_contains_repo_commit_problem() {
        let instance = SweBenchInstance {
            instance_id: "id".into(),
            repo: "owner/name".into(),
            base_commit: "deadbeef".into(),
            patch: String::new(),
            test_patch: String::new(),
            problem_statement: "the bug description".into(),
            hints_text: String::new(),
            version: String::new(),
            environment_setup_commit: String::new(),
            fail_to_pass: vec![],
            pass_to_pass: vec![],
        };
        let s = instance.task_instruction();
        assert!(s.contains("owner/name"));
        assert!(s.contains("deadbeef"));
        assert!(s.contains("the bug description"));
    }

    #[test]
    fn to_task_stashes_instance_in_metadata() {
        let instance = SweBenchInstance {
            instance_id: "id-1".into(),
            repo: "r".into(),
            base_commit: "abc".into(),
            patch: String::new(),
            test_patch: String::new(),
            problem_statement: "p".into(),
            hints_text: String::new(),
            version: String::new(),
            environment_setup_commit: String::new(),
            fail_to_pass: vec!["test_foo".into()],
            pass_to_pass: vec![],
        };
        let task = instance.to_task();
        assert_eq!(task.id.as_deref(), Some("id-1"));
        let md = &task.metadata["swe-bench.instance"];
        assert_eq!(md["instance_id"], "id-1");
        assert_eq!(md["FAIL_TO_PASS"], serde_json::json!(["test_foo"]));
    }

    #[test]
    fn from_jsonl_skips_blank_lines_and_reports_parse_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        let valid = serde_json::json!({
            "instance_id": "id-1",
            "repo": "r",
            "base_commit": "abc",
            "test_patch": "",
            "problem_statement": "p",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        });
        let content = format!("\n{}\n\n", valid);
        std::fs::write(&path, content).unwrap();
        let ds = SweBenchLite::from_jsonl(&path).unwrap();
        assert_eq!(ds.instances().len(), 1);

        // Now an invalid line.
        std::fs::write(&path, "{not json}").unwrap();
        match SweBenchLite::from_jsonl(&path) {
            Err(BenchmarkError::Other(msg)) => assert!(msg.contains("line 1")),
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected parse error"),
        }
    }
}
