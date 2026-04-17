//! `TrajectoryHandle` (§9.4) lives here in core so `RunOutcome` can carry it without
//! every downstream crate depending on `oharness-trace`. The concrete *sources* a
//! handle can point at (file, in-memory, stream) are enumerated here; the trace
//! crate supplies the machinery that reads them.

use crate::event::Event;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Read-only reference to a trajectory. Serializes as a path/URI — never inlines
/// the event stream. In-memory handles error on serialization unless materialized
/// to a file first.
#[derive(Clone)]
pub struct TrajectoryHandle {
    source: TrajectorySource,
}

#[derive(Clone)]
pub enum TrajectorySource {
    File(PathBuf),
    InMemory(Arc<Vec<Event>>),
}

impl std::fmt::Debug for TrajectoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            TrajectorySource::File(p) => write!(f, "TrajectoryHandle::File({})", p.display()),
            TrajectorySource::InMemory(v) => {
                write!(f, "TrajectoryHandle::InMemory(len={})", v.len())
            }
        }
    }
}

impl TrajectoryHandle {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            source: TrajectorySource::File(path.into()),
        }
    }

    pub fn in_memory(events: Vec<Event>) -> Self {
        Self {
            source: TrajectorySource::InMemory(Arc::new(events)),
        }
    }

    pub fn source(&self) -> &TrajectorySource {
        &self.source
    }

    /// Returns `Some(path)` for file-backed handles, `None` for in-memory.
    pub fn path(&self) -> Option<&std::path::Path> {
        match &self.source {
            TrajectorySource::File(p) => Some(p),
            TrajectorySource::InMemory(_) => None,
        }
    }

    pub fn summarize(&self) -> TrajectorySummary {
        match &self.source {
            TrajectorySource::File(p) => TrajectorySummary {
                event_count: None,
                path: Some(p.clone()),
                in_memory: false,
            },
            TrajectorySource::InMemory(v) => TrajectorySummary {
                event_count: Some(v.len()),
                path: None,
                in_memory: true,
            },
        }
    }

    /// In-memory handles return their events inline. File handles require callers
    /// to read via `oharness-trace`; this helper is not available here to keep
    /// core IO-free.
    pub fn in_memory_events(&self) -> Option<&Arc<Vec<Event>>> {
        match &self.source {
            TrajectorySource::InMemory(v) => Some(v),
            TrajectorySource::File(_) => None,
        }
    }
}

impl Serialize for TrajectoryHandle {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match &self.source {
            TrajectorySource::File(p) => {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("type", "file")?;
                m.serialize_entry("path", p)?;
                m.end()
            }
            TrajectorySource::InMemory(_) => Err(serde::ser::Error::custom(
                "in-memory TrajectoryHandle cannot be serialized; materialize to a file first",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for TrajectoryHandle {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(rename = "type")]
            kind: String,
            path: Option<PathBuf>,
        }
        let w = Wire::deserialize(d)?;
        match w.kind.as_str() {
            "file" => {
                let p = w
                    .path
                    .ok_or_else(|| serde::de::Error::custom("file trajectory missing `path`"))?;
                Ok(TrajectoryHandle::from_path(p))
            }
            other => Err(serde::de::Error::custom(format!(
                "unknown trajectory source `{other}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySummary {
    pub event_count: Option<usize>,
    pub path: Option<PathBuf>,
    pub in_memory: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum TrajectoryError {
    #[error("trajectory I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("trajectory JSON decode: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("payload sidecar missing: {0}")]
    PayloadNotFound(String),
    #[error("unsupported trajectory source for this operation")]
    Unsupported,
}
