//! JSONL trajectory reader. Used by `TrajectoryHandle::File(...)` consumers.

use oharness_core::{Event, TrajectoryError};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Load every event from a JSONL trajectory file.
pub async fn read_events(path: &Path) -> Result<Vec<Event>, TrajectoryError> {
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut out = Vec::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(trimmed)?;
        out.push(event);
    }
    Ok(out)
}

/// Stream events one at a time. Caller drives iteration; useful for tailing huge
/// trajectory files.
pub async fn read_events_streaming<F>(
    path: &Path,
    mut each: F,
) -> Result<(), TrajectoryError>
where
    F: FnMut(Event) -> bool,
{
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(trimmed)?;
        if !each(event) {
            break;
        }
    }
    Ok(())
}
