//! [`SweBenchEvaluator`] — applies the instance's `test_patch`, runs a
//! configurable test command (default `pytest`), parses per-test
//! outcomes, and grades against `FAIL_TO_PASS` / `PASS_TO_PASS`.
//!
//! Evaluation procedure per task (plan §13.6 / SWE-bench scoring):
//!
//! 1. Locate the workspace. We rely on the agent having edited files
//!    in place inside the `LoadedTask.workspace` that this crate
//!    staged during `Benchmark::load_task`. The evaluator doesn't
//!    have direct access to that `Arc<Workspace>` from `RunOutcome`
//!    alone, so we re-derive the workspace path from the
//!    evaluator's `clone_root` + sanitised `instance_id`. That
//!    mirrors the staging convention in `dataset.rs`.
//! 2. `git apply test_patch` inside the workspace to enable the
//!    FAIL_TO_PASS tests.
//! 3. Run the configured test command with the workspace as the
//!    working directory.
//! 4. Parse stdout via [`crate::parse_pytest_output`].
//! 5. Grade: pass iff every FAIL_TO_PASS id is `PASSED`/`XPASS` AND
//!    every PASS_TO_PASS id is `PASSED`/`XPASS`. Missing ids count
//!    as failure.
//!
//! `EvaluationResult.details` carries the full parsed outcome map so
//! downstream reflectors / reports can surface which specific tests
//! regressed.

use crate::dataset::SweBenchInstance;
use crate::pytest::{parse_pytest_output, PytestResults};
use async_trait::async_trait;
use oharness_core::{EvaluationResult, MetadataMap, RunOutcome, Task, TaskEvaluator};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub struct SweBenchEvaluator {
    /// Command + args used to run tests. Default: `["pytest", "-v",
    /// "--tb=short", "--no-header"]`. Override via
    /// [`SweBenchEvaluator::with_test_command`] for environments
    /// where `pytest` isn't on PATH or for tests that use a stub.
    test_command: Vec<String>,

    /// Root directory the adapter cloned repos into. The evaluator
    /// re-derives per-task workspace paths as
    /// `clone_root.join(sanitize(instance_id))`. Mirrors the staging
    /// convention in [`crate::SweBenchLite::load_task`].
    clone_root: PathBuf,
}

impl Default for SweBenchEvaluator {
    fn default() -> Self {
        Self {
            test_command: vec![
                "pytest".to_string(),
                "-v".to_string(),
                "--tb=short".to_string(),
                "--no-header".to_string(),
            ],
            clone_root: std::env::temp_dir().join("oharness-swe-bench"),
        }
    }
}

impl SweBenchEvaluator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_test_command<I, S>(mut self, cmd: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.test_command = cmd.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_clone_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.clone_root = root.into();
        self
    }

    fn workspace_for(&self, instance_id: &str) -> PathBuf {
        self.clone_root.join(sanitize_instance_id(instance_id))
    }
}

#[async_trait]
impl TaskEvaluator for SweBenchEvaluator {
    async fn evaluate(&self, task: &Task, _outcome: &RunOutcome) -> EvaluationResult {
        let Some(instance_value) = task.metadata.get("swe-bench.instance") else {
            return failing("task metadata missing `swe-bench.instance`");
        };
        let instance: SweBenchInstance = match serde_json::from_value(instance_value.clone()) {
            Ok(i) => i,
            Err(e) => return failing(&format!("decode swe-bench.instance: {e}")),
        };

        let workspace = self.workspace_for(&instance.instance_id);
        if !workspace.exists() {
            return failing(&format!(
                "evaluator workspace not found at {} — did the benchmark stage this task?",
                workspace.display()
            ));
        }

        if let Err(e) = apply_test_patch(&workspace, &instance.test_patch).await {
            return failing(&format!("apply test_patch: {e}"));
        }

        let output = match run_test_command(&self.test_command, &workspace).await {
            Ok(o) => o,
            Err(e) => return failing(&format!("run tests: {e}")),
        };

        let results = parse_pytest_output(&output);
        grade(&instance, &results, &output)
    }
}

async fn apply_test_patch(workspace: &Path, patch: &str) -> std::io::Result<()> {
    if patch.trim().is_empty() {
        return Ok(());
    }
    use tokio::io::AsyncWriteExt;

    let mut child = tokio::process::Command::new("git")
        .args(["apply", "--whitespace=nowarn", "-"])
        .current_dir(workspace)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(patch.as_bytes()).await?;
        stdin.flush().await?;
        drop(stdin);
    }

    let output = child.wait_with_output().await?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "git apply exited {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

async fn run_test_command(cmd: &[String], workspace: &Path) -> std::io::Result<String> {
    let Some((program, args)) = cmd.split_first() else {
        return Err(std::io::Error::other("empty test command"));
    };
    let output = tokio::process::Command::new(program)
        .args(args)
        .current_dir(workspace)
        .kill_on_drop(true)
        .output()
        .await?;
    // Concat stdout + stderr — pytest writes progress to stdout but
    // some environments (notably `pytest-xdist`) route status lines
    // through stderr.
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(combined)
}

fn grade(
    instance: &SweBenchInstance,
    results: &PytestResults,
    raw_output: &str,
) -> EvaluationResult {
    let f2p_pass = results.all_passed(&instance.fail_to_pass);
    let p2p_pass = results.all_passed(&instance.pass_to_pass);
    let passed = f2p_pass && p2p_pass;

    let fail_to_pass_missing: Vec<&str> = instance
        .fail_to_pass
        .iter()
        .filter(|t| !results.passed(t))
        .map(String::as_str)
        .collect();
    let pass_to_pass_missing: Vec<&str> = instance
        .pass_to_pass
        .iter()
        .filter(|t| !results.passed(t))
        .map(String::as_str)
        .collect();

    let score = if passed { 1.0 } else { 0.0 };

    let mut details = MetadataMap::new();
    details.insert("fail_to_pass_passed".into(), json!(f2p_pass));
    details.insert("pass_to_pass_passed".into(), json!(p2p_pass));
    details.insert("fail_to_pass_missing".into(), json!(fail_to_pass_missing));
    details.insert("pass_to_pass_missing".into(), json!(pass_to_pass_missing));
    details.insert(
        "outcomes".into(),
        Value::Object(
            results
                .outcomes
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect(),
        ),
    );
    // Truncated raw output for auditability. Cap at 4KB to keep
    // evaluation.json manageable.
    let tail_cap = 4096;
    let raw_tail = if raw_output.len() > tail_cap {
        raw_output[raw_output.len() - tail_cap..].to_string()
    } else {
        raw_output.to_string()
    };
    details.insert("pytest_output_tail".into(), Value::String(raw_tail));

    EvaluationResult {
        score,
        passed,
        details,
    }
}

fn failing(reason: &str) -> EvaluationResult {
    let mut details = MetadataMap::new();
    details.insert("error".into(), Value::String(reason.to_string()));
    EvaluationResult {
        score: 0.0,
        passed: false,
        details,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_instance() -> SweBenchInstance {
        SweBenchInstance {
            instance_id: "id-1".into(),
            repo: "r/r".into(),
            base_commit: "abc".into(),
            patch: String::new(),
            test_patch: String::new(),
            problem_statement: "p".into(),
            hints_text: String::new(),
            version: String::new(),
            environment_setup_commit: String::new(),
            fail_to_pass: vec!["tests/a.py::test_one".into()],
            pass_to_pass: vec!["tests/a.py::test_two".into()],
        }
    }

    #[test]
    fn grade_passes_when_both_sets_pass() {
        let output = "\
tests/a.py::test_one PASSED
tests/a.py::test_two PASSED
";
        let results = parse_pytest_output(output);
        let eval = grade(&sample_instance(), &results, output);
        assert!(eval.passed);
        assert_eq!(eval.score, 1.0);
    }

    #[test]
    fn grade_fails_when_f2p_test_missing() {
        let output = "tests/a.py::test_two PASSED\n";
        let results = parse_pytest_output(output);
        let eval = grade(&sample_instance(), &results, output);
        assert!(!eval.passed);
        let missing = eval.details.get("fail_to_pass_missing").unwrap();
        assert_eq!(missing, &json!(["tests/a.py::test_one"]));
    }

    #[test]
    fn grade_fails_when_p2p_regresses() {
        let output = "\
tests/a.py::test_one PASSED
tests/a.py::test_two FAILED
";
        let results = parse_pytest_output(output);
        let eval = grade(&sample_instance(), &results, output);
        assert!(!eval.passed);
    }

    #[test]
    fn workspace_for_matches_sanitization_convention() {
        let e = SweBenchEvaluator::default().with_clone_root("/tmp/x");
        assert_eq!(
            e.workspace_for("astropy__astropy-12907"),
            PathBuf::from("/tmp/x/astropy__astropy-12907")
        );
        assert_eq!(
            e.workspace_for("owner/repo-42"),
            PathBuf::from("/tmp/x/owner_repo-42")
        );
    }

    #[tokio::test]
    async fn evaluate_returns_failing_when_workspace_missing() {
        let e = SweBenchEvaluator::default().with_clone_root("/tmp/this-does-not-exist-please");
        let instance = sample_instance();
        let task = instance.to_task();
        // Use a cheap dummy outcome — evaluator doesn't inspect it.
        let outcome = RunOutcome {
            run_id: oharness_core::RunId::new(),
            task_id: task.id.clone(),
            termination: oharness_core::Termination::Completed {
                reason: oharness_core::CompletionReason::EndTurn,
            },
            final_messages: Vec::new(),
            trajectory: oharness_core::TrajectoryHandle::in_memory(Vec::new()),
            usage: oharness_core::ResourceUsage::default(),
            per_model_usage: Default::default(),
            started_at: time::OffsetDateTime::now_utc(),
            finished_at: time::OffsetDateTime::now_utc(),
            agent_state: MetadataMap::new(),
        };
        let result = e.evaluate(&task, &outcome).await;
        assert!(!result.passed);
        assert!(result
            .details
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("workspace not found"));
    }
}
