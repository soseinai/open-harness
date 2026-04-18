//! [`BenchmarkRunConfig`] (plan §13.3).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Knobs for a single `run_benchmark` invocation. Serde-serializable so
/// a snapshot lands in `{output_dir}/config.toml` alongside the run's
/// trajectories — the paper-supplement artifact per plan §13.5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    pub output_dir: PathBuf,
    /// Parallel-run concurrency. Plan §13.4 default: 8.
    #[serde(default = "default_run_concurrency")]
    pub run_concurrency: usize,
    /// Parallel-load concurrency (network/disk-bound; typically lower
    /// than run concurrency). Plan §13.4 default: 4.
    #[serde(default = "default_load_concurrency")]
    pub load_concurrency: usize,
    /// Stop starting new tasks once cumulative cost crosses this
    /// threshold. In-flight runs finish (plan §13.4 rule). `None`
    /// disables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost_usd: Option<f64>,
    /// Glob-ish filter on task id. When `Some`, only tasks whose id
    /// contains `filter` (case-sensitive substring match) run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Subsample to the first `n` (post-filter) tasks. Deterministic —
    /// just takes the prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_n: Option<usize>,
    /// Manual sharding: run tasks whose index `i` satisfies
    /// `i % shard.total == shard.index`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<Shard>,
    /// When `true`, skip tasks whose `outcome.json` already exists
    /// under `output_dir/{task_id}/` — enables mid-run resume.
    #[serde(default)]
    pub resume: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Shard {
    pub index: usize,
    pub total: usize,
}

fn default_run_concurrency() -> usize {
    8
}
fn default_load_concurrency() -> usize {
    4
}

impl BenchmarkRunConfig {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            run_concurrency: default_run_concurrency(),
            load_concurrency: default_load_concurrency(),
            max_cost_usd: None,
            filter: None,
            sample_n: None,
            shard: None,
            resume: false,
        }
    }

    pub fn with_run_concurrency(mut self, n: usize) -> Self {
        self.run_concurrency = n;
        self
    }

    pub fn with_load_concurrency(mut self, n: usize) -> Self {
        self.load_concurrency = n;
        self
    }

    pub fn with_max_cost_usd(mut self, usd: f64) -> Self {
        self.max_cost_usd = Some(usd);
        self
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    pub fn with_sample_n(mut self, n: usize) -> Self {
        self.sample_n = Some(n);
        self
    }

    pub fn with_shard(mut self, index: usize, total: usize) -> Self {
        self.shard = Some(Shard { index, total });
        self
    }

    pub fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    /// Apply `filter`, `sample_n`, and `shard` to a stream of task ids
    /// (in dataset order). Returns the ids the runner should actually
    /// process, in the order they'll be scheduled.
    pub fn select_ids<I>(&self, ids: I) -> Vec<String>
    where
        I: IntoIterator<Item = String>,
    {
        // Apply filter, then shard, then sample — order matches plan
        // §13.3 rule: "deterministic — just takes the prefix" for
        // sampling, after sharding has already thinned the set.
        let filtered = ids
            .into_iter()
            .enumerate()
            .filter(|(_, id)| self.filter.as_deref().is_none_or(|f| id.contains(f)));
        let sharded: Vec<String> = match self.shard {
            Some(Shard { index, total }) if total > 0 => filtered
                .filter(|(i, _)| i % total == index)
                .map(|(_, id)| id)
                .collect(),
            _ => filtered.map(|(_, id)| id).collect(),
        };
        match self.sample_n {
            Some(n) => sharded.into_iter().take(n).collect(),
            None => sharded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_plan() {
        let cfg = BenchmarkRunConfig::new("/tmp/results");
        assert_eq!(cfg.run_concurrency, 8);
        assert_eq!(cfg.load_concurrency, 4);
        assert!(cfg.max_cost_usd.is_none());
        assert!(!cfg.resume);
    }

    #[test]
    fn select_ids_passes_everything_through_when_no_knobs() {
        let cfg = BenchmarkRunConfig::new("/tmp/r");
        let ids: Vec<_> = (0..5).map(|i| format!("id-{i}")).collect();
        let out = cfg.select_ids(ids.clone());
        assert_eq!(out, ids);
    }

    #[test]
    fn select_ids_applies_substring_filter() {
        let cfg = BenchmarkRunConfig::new("/tmp/r").with_filter("even");
        let ids = vec![
            "odd-1".to_string(),
            "even-2".to_string(),
            "odd-3".to_string(),
            "even-4".to_string(),
        ];
        let out = cfg.select_ids(ids);
        assert_eq!(out, vec!["even-2", "even-4"]);
    }

    #[test]
    fn select_ids_applies_sharding_across_original_indices() {
        // 6 ids across 2 shards. Shard 0 picks indices 0, 2, 4.
        let cfg = BenchmarkRunConfig::new("/tmp/r").with_shard(0, 2);
        let ids: Vec<_> = (0..6).map(|i| format!("id-{i}")).collect();
        let out = cfg.select_ids(ids);
        assert_eq!(out, vec!["id-0", "id-2", "id-4"]);
    }

    #[test]
    fn select_ids_takes_prefix_on_sample_n() {
        let cfg = BenchmarkRunConfig::new("/tmp/r").with_sample_n(3);
        let ids: Vec<_> = (0..10).map(|i| format!("id-{i}")).collect();
        let out = cfg.select_ids(ids);
        assert_eq!(out, vec!["id-0", "id-1", "id-2"]);
    }

    #[test]
    fn select_ids_composes_filter_then_shard_then_sample() {
        let cfg = BenchmarkRunConfig::new("/tmp/r")
            .with_filter("keep")
            .with_shard(0, 2)
            .with_sample_n(2);
        // Filter keeps "keep-*" entries with their *original* enumeration
        // indices, then shard 0 of 2 keeps even-index entries, then
        // sample takes the first two.
        let ids = vec![
            "keep-a".to_string(), // orig idx 0 — kept by filter, idx 0
            "drop-x".to_string(),
            "keep-b".to_string(), // orig idx 2
            "drop-y".to_string(),
            "keep-c".to_string(), // orig idx 4
            "keep-d".to_string(), // orig idx 5
        ];
        let out = cfg.select_ids(ids);
        // filter keeps all "keep-*" with enumeration indices 0,2,4,5 →
        // shard 0 of 2 keeps idx % 2 == 0 → 0, 2, 4 → "keep-a",
        // "keep-b", "keep-c" → sample 2 → first two.
        assert_eq!(out, vec!["keep-a", "keep-b"]);
    }
}
