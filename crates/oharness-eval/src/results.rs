//! Results directory layout writer (plan §13.5).
//!
//! Produces the paper-supplement artifact for a single benchmark run:
//!
//! ```text
//! {output_dir}/
//! ├── config.toml              # snapshot of BenchmarkRunConfig
//! ├── manifest.json            # completed ids + rolling cost
//! ├── {task_id}/
//! │   ├── outcome.json         # serialized RunOutcome
//! │   ├── trajectory.jsonl
//! │   └── evaluation.json
//! └── …
//! ```
//!
//! The manifest is written after each task completes, so resume works
//! even if the runner crashes mid-run.

use crate::config::BenchmarkRunConfig;
use oharness_core::{EvaluationResult, Event, RunOutcome, TrajectoryHandle};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const CONFIG_FILE: &str = "config.toml";
pub const OUTCOME_FILE: &str = "outcome.json";
pub const TRAJECTORY_FILE: &str = "trajectory.jsonl";
pub const EVALUATION_FILE: &str = "evaluation.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub benchmark_name: String,
    pub benchmark_version: String,
    /// Completed task ids. Order is completion order, not dataset
    /// order — that lets a reader spot long-running tail tasks.
    pub completed: Vec<String>,
    /// Failed task ids (load or evaluation error).
    pub failed: Vec<String>,
    /// Rolling cost in USD — sum of per-task `ResourceUsage.cost_usd`.
    pub total_cost_usd: f64,
}

impl Manifest {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes)
    }
}

pub fn write_config(output_dir: &Path, config: &BenchmarkRunConfig) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(CONFIG_FILE);
    let toml = toml::to_string_pretty(config).map_err(std::io::Error::other)?;
    std::fs::write(path, toml)
}

pub fn task_dir(output_dir: &Path, task_id: &str) -> PathBuf {
    output_dir.join(sanitize_id(task_id))
}

pub fn write_task_artifacts(
    output_dir: &Path,
    task_id: &str,
    outcome: &RunOutcome,
    evaluation: &EvaluationResult,
    trajectory: &[Event],
) -> std::io::Result<()> {
    let dir = task_dir(output_dir, task_id);
    std::fs::create_dir_all(&dir)?;

    // Write the trajectory JSONL first so the outcome.json can reference
    // it by path. `RunOutcome` serializes its `trajectory` field via
    // [`TrajectoryHandle`], and an in-memory handle refuses to serialize
    // by design (plan §9.4) — so we swap to a file-backed handle
    // pointing at the just-written JSONL before serializing the outcome.
    let trajectory_path = dir.join(TRAJECTORY_FILE);
    let mut jsonl = Vec::with_capacity(trajectory.len() * 128);
    for event in trajectory {
        let mut line = serde_json::to_vec(event)?;
        line.push(b'\n');
        jsonl.extend_from_slice(&line);
    }
    std::fs::write(&trajectory_path, jsonl)?;

    let mut outcome_on_disk = outcome.clone();
    outcome_on_disk.trajectory = TrajectoryHandle::from_path(&trajectory_path);
    let outcome_bytes = serde_json::to_vec_pretty(&outcome_on_disk)?;
    std::fs::write(dir.join(OUTCOME_FILE), outcome_bytes)?;

    let eval_bytes = serde_json::to_vec_pretty(evaluation)?;
    std::fs::write(dir.join(EVALUATION_FILE), eval_bytes)?;
    Ok(())
}

pub fn task_outcome_exists(output_dir: &Path, task_id: &str) -> bool {
    task_dir(output_dir, task_id).join(OUTCOME_FILE).exists()
}

/// Task ids are surfaced verbatim in the results directory layout;
/// some benchmarks use paths / URLs / arbitrary text. Be conservative:
/// replace anything that isn't alphanumeric / dot / underscore / hyphen
/// with `_`. The manifest still carries the original id.
fn sanitize_id(id: &str) -> String {
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

// ======================================================================
// BenchmarkReport — runner return value
// ======================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark_name: String,
    pub benchmark_version: String,
    /// Per-task reports, in completion order.
    pub tasks: Vec<TaskReport>,
    pub total_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskReport {
    pub task_id: String,
    /// `None` on load / factory errors that short-circuited before
    /// evaluation.
    pub evaluation: Option<EvaluationResult>,
    pub turns: u32,
    pub tool_calls: u32,
    pub cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl BenchmarkReport {
    /// Fraction of runs that passed, over the total number of tasks
    /// attempted (including ones that errored, which count as !pass).
    pub fn pass_at_1(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        let passed = self
            .tasks
            .iter()
            .filter(|t| t.evaluation.as_ref().is_some_and(|e| e.passed))
            .count();
        passed as f64 / self.tasks.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn sanitize_id_replaces_unsafe_chars() {
        assert_eq!(sanitize_id("abc/def"), "abc_def");
        assert_eq!(sanitize_id("swe-bench/task_42"), "swe-bench_task_42");
        assert_eq!(sanitize_id("simple-id.v2"), "simple-id.v2");
    }

    #[test]
    fn manifest_round_trips_through_disk() {
        let dir = tempdir().unwrap();
        let manifest = Manifest {
            benchmark_name: "b".into(),
            benchmark_version: "1".into(),
            completed: vec!["a".into(), "b".into()],
            failed: vec![],
            total_cost_usd: 1.25,
        };
        let path = dir.path().join(MANIFEST_FILE);
        manifest.save(&path).unwrap();
        let back = Manifest::load(&path).unwrap();
        assert_eq!(back.completed, manifest.completed);
        assert_eq!(back.total_cost_usd, 1.25);
    }

    #[test]
    fn pass_at_1_on_empty_report_is_zero() {
        let r = BenchmarkReport {
            benchmark_name: "b".into(),
            benchmark_version: "1".into(),
            tasks: vec![],
            total_cost_usd: 0.0,
        };
        assert_eq!(r.pass_at_1(), 0.0);
    }

    #[test]
    fn pass_at_1_counts_passed() {
        let r = BenchmarkReport {
            benchmark_name: "b".into(),
            benchmark_version: "1".into(),
            tasks: vec![
                TaskReport {
                    task_id: "a".into(),
                    evaluation: Some(EvaluationResult::pass()),
                    turns: 1,
                    tool_calls: 0,
                    cost_usd: None,
                    error: None,
                },
                TaskReport {
                    task_id: "b".into(),
                    evaluation: Some(EvaluationResult::fail()),
                    turns: 1,
                    tool_calls: 0,
                    cost_usd: None,
                    error: None,
                },
                TaskReport {
                    task_id: "c".into(),
                    evaluation: None,
                    turns: 0,
                    tool_calls: 0,
                    cost_usd: None,
                    error: Some("load".into()),
                },
            ],
            total_cost_usd: 0.0,
        };
        assert!((r.pass_at_1() - 1.0 / 3.0).abs() < 1e-9);
    }
}
