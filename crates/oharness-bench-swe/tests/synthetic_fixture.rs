//! End-to-end integration test exercising the SWE-bench adapter's
//! full flow against a synthetic local git repo + a stub test
//! command. Does NOT require pytest / python / network — just `git`,
//! which the adapter already depends on.
//!
//! Flow:
//! 1. Create a bare "upstream" git repo with two commits:
//!    commit `base` (code.py returns 1) and commit `after_fix`
//!    (code.py returns 42). We never actually check out
//!    `after_fix`; it exists only so the checkout step exercises
//!    real SHA resolution.
//! 2. Build a SweBenchInstance whose `test_patch` adds
//!    `tests/test_answer.py` that prints
//!    `tests/test_answer.py::test_x PASSED` when run, via a stub
//!    `test_command` that just `cat`s the file.
//! 3. Invoke `Benchmark::load_task` — that git-clones our synthetic
//!    upstream into a tempdir and checks out `base`.
//! 4. Invoke the evaluator — which `git apply test_patch`s the test
//!    file, runs the stub command, and grades FAIL_TO_PASS against
//!    the parsed output.
//!
//! This proves the plumbing (dataset types → staging → patch apply →
//! test runner → grading) works end-to-end without the real Python
//! environment the M2 gate needs.

use oharness_bench_swe::{SweBenchEvaluator, SweBenchInstance, SweBenchLite};
use oharness_core::{
    CompletionReason, MetadataMap, ResourceUsage, RunId, RunOutcome, Termination, TrajectoryHandle,
};
use oharness_eval::Benchmark;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;
use time::OffsetDateTime;

fn git(dir: &Path, args: &[&str]) {
    // Keep commits stable across hosts by pinning author/committer.
    let status = Command::new("git")
        .args(args)
        .env("GIT_AUTHOR_NAME", "test")
        .env("GIT_AUTHOR_EMAIL", "test@example.com")
        .env("GIT_COMMITTER_NAME", "test")
        .env("GIT_COMMITTER_EMAIL", "test@example.com")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .current_dir(dir)
        .status()
        .expect("git spawn");
    assert!(status.success(), "git {args:?} failed ({status:?})");
}

fn write(path: impl AsRef<Path>, content: &str) {
    std::fs::write(path, content).expect("write");
}

fn create_upstream_repo(root: &Path) -> String {
    // Working directory that we'll push from.
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    git(&work, &["init", "--quiet", "--initial-branch=main"]);
    git(&work, &["config", "user.name", "test"]);
    git(&work, &["config", "user.email", "test@example.com"]);
    write(work.join("code.py"), "def answer():\n    return 1\n");
    git(&work, &["add", "code.py"]);
    git(&work, &["commit", "-q", "-m", "base"]);

    // Capture the SHA for the base commit.
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&work)
        .output()
        .expect("rev-parse");
    let base = String::from_utf8(out.stdout).unwrap().trim().to_string();

    // Add a second commit so the base isn't trivially HEAD of an
    // advanced branch — nothing in the test actually uses this, but
    // it makes the checkout step exercise real SHA resolution.
    write(work.join("code.py"), "def answer():\n    return 42\n");
    git(&work, &["add", "code.py"]);
    git(&work, &["commit", "-q", "-m", "after_fix"]);

    // Bare clone so `git clone` against a local directory path works.
    let bare = root.join("upstream.git");
    git(
        root,
        &[
            "clone",
            "--quiet",
            "--bare",
            work.to_str().unwrap(),
            bare.to_str().unwrap(),
        ],
    );

    // Return the URL (path) to the bare repo + the base commit SHA.
    let _ = base;
    // Rebuild the base SHA from the bare clone so both paths agree.
    let out = Command::new("git")
        .args(["rev-parse", "main~1"])
        .current_dir(&bare)
        .output()
        .expect("rev-parse base");
    let base = String::from_utf8(out.stdout).unwrap().trim().to_string();
    bare.to_str().unwrap().to_string() + "\n" + &base
}

fn dummy_outcome() -> RunOutcome {
    RunOutcome {
        run_id: RunId::new(),
        task_id: None,
        termination: Termination::Completed {
            reason: CompletionReason::EndTurn,
        },
        final_messages: Vec::new(),
        trajectory: TrajectoryHandle::in_memory(Vec::new()),
        usage: ResourceUsage::default(),
        per_model_usage: Default::default(),
        started_at: OffsetDateTime::now_utc(),
        finished_at: OffsetDateTime::now_utc(),
        agent_state: MetadataMap::new(),
    }
}

#[tokio::test]
async fn full_flow_against_synthetic_repo() {
    let fixture = tempdir().expect("fixture tempdir");
    let two = create_upstream_repo(fixture.path());
    let mut parts = two.splitn(2, '\n');
    let upstream_url = parts.next().unwrap().to_string();
    let base_commit = parts.next().unwrap().to_string();

    // A minimal `test_patch` that adds tests/test_answer.py with two
    // fake pytest-shaped lines. The stub "test command" just `cat`s
    // the file, so the evaluator sees pytest-v-like output.
    let test_patch = r#"diff --git a/tests/test_answer.py b/tests/test_answer.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/tests/test_answer.py
@@ -0,0 +1,2 @@
+tests/test_answer.py::test_pass PASSED
+tests/test_answer.py::test_keep PASSED
"#;

    let instance = SweBenchInstance {
        instance_id: "synth-1".into(),
        repo: upstream_url,
        base_commit,
        patch: String::new(),
        test_patch: test_patch.to_string(),
        problem_statement: "fix the thing".into(),
        hints_text: String::new(),
        version: String::new(),
        environment_setup_commit: String::new(),
        fail_to_pass: vec!["tests/test_answer.py::test_pass".into()],
        pass_to_pass: vec!["tests/test_answer.py::test_keep".into()],
    };

    let clone_root = fixture.path().join("clones");
    let evaluator = SweBenchEvaluator::new()
        .with_clone_root(clone_root.clone())
        // Stub test runner: `cat tests/test_answer.py` produces the
        // pytest-shaped lines we baked into the patch above. No python
        // required.
        .with_test_command(["cat", "tests/test_answer.py"]);

    let benchmark = SweBenchLite::from_instances(vec![instance.clone()])
        .with_clone_root(clone_root.clone())
        .with_evaluator(evaluator);

    assert_eq!(benchmark.name(), "swe-bench-lite");
    assert_eq!(benchmark.task_count(), Some(1));

    let loaded = benchmark
        .load_task("synth-1")
        .await
        .expect("load_task against synthetic upstream");
    let workspace_path = loaded
        .workspace
        .as_ref()
        .expect("benchmark stages a workspace")
        .path
        .clone();
    assert!(workspace_path.join("code.py").exists());

    // File should contain the base-commit text (returns 1), not the
    // second commit's (returns 42) — proves base_commit checkout.
    let contents = std::fs::read_to_string(workspace_path.join("code.py")).unwrap();
    assert!(
        contents.contains("return 1"),
        "expected base_commit checkout, got: {contents:?}"
    );

    // Evaluate against a dummy outcome — the evaluator only uses the
    // task's metadata + on-disk workspace.
    let result = benchmark
        .evaluator()
        .evaluate(&loaded.task, &dummy_outcome())
        .await;

    assert!(
        result.passed,
        "expected passing evaluation, got details: {:?}",
        result.details
    );
    assert_eq!(result.score, 1.0);
    assert_eq!(result.details["fail_to_pass_passed"], true);
    assert_eq!(result.details["pass_to_pass_passed"], true);
}

#[tokio::test]
async fn evaluator_reports_failure_when_required_tests_missing() {
    let fixture = tempdir().expect("fixture tempdir");
    let two = create_upstream_repo(fixture.path());
    let mut parts = two.splitn(2, '\n');
    let upstream_url = parts.next().unwrap().to_string();
    let base_commit = parts.next().unwrap().to_string();

    // test_patch adds a file that reports *different* test ids than the
    // instance's FAIL_TO_PASS expects — grade should fail.
    let test_patch = r#"diff --git a/tests/test_other.py b/tests/test_other.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/tests/test_other.py
@@ -0,0 +1 @@
+tests/test_other.py::test_unrelated PASSED
"#;

    let instance = SweBenchInstance {
        instance_id: "synth-2".into(),
        repo: upstream_url,
        base_commit,
        patch: String::new(),
        test_patch: test_patch.to_string(),
        problem_statement: "fix".into(),
        hints_text: String::new(),
        version: String::new(),
        environment_setup_commit: String::new(),
        fail_to_pass: vec!["tests/test_answer.py::test_missing".into()],
        pass_to_pass: vec![],
    };

    let clone_root = fixture.path().join("clones");
    let evaluator = SweBenchEvaluator::new()
        .with_clone_root(clone_root.clone())
        .with_test_command(["cat", "tests/test_other.py"]);

    let benchmark = SweBenchLite::from_instances(vec![instance])
        .with_clone_root(clone_root)
        .with_evaluator(evaluator);

    let loaded = benchmark.load_task("synth-2").await.unwrap();
    let result = benchmark
        .evaluator()
        .evaluate(&loaded.task, &dummy_outcome())
        .await;

    assert!(!result.passed);
    let missing = result
        .details
        .get("fail_to_pass_missing")
        .and_then(|v| v.as_array())
        .expect("fail_to_pass_missing list");
    assert_eq!(missing[0], "tests/test_answer.py::test_missing");
}
