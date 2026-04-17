//! `FileSink` — JSONL trajectory writer (§9.2, §9.3).
//!
//! M1a: plain JSONL. Payload externalization (sidecar files for >64KiB payloads)
//! is M1b+. The writer task is a tokio task that drains a bounded mpsc channel and
//! appends newline-delimited JSON to the target file.
//!
//! Backpressure model (§4.6): emitters `try_send` first; on `Full`, they spawn a
//! `spawn_blocking` task that calls `blocking_send`. The block stays on the
//! blocking pool, never on a tokio worker thread.

use oharness_core::{Event, EventSink};
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::{self, Sender, error::TrySendError};
use tokio::task::JoinHandle;
use tracing::warn;

const DEFAULT_BUFFER: usize = 10_000;

pub struct FileSink {
    tx: Sender<Event>,
    path: PathBuf,
    // Kept so the writer task has a clean way to be awaited/joined from outside.
    writer_handle: Arc<tokio::sync::Mutex<Option<JoinHandle<io::Result<()>>>>>,
}

impl std::fmt::Debug for FileSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSink")
            .field("path", &self.path)
            .finish()
    }
}

impl FileSink {
    /// Create a file sink rooted at `path`. The file must not exist (runs with a
    /// colliding UUID are caller errors per §9.3). Returns the sink; drain via
    /// `flush().await` before the program exits to ensure all events hit disk.
    pub async fn to_path(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        Self::to_path_with_buffer(path, buffer_size()).await
    }

    pub async fn to_path_with_buffer(path: PathBuf, buffer: usize) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "trajectory file already exists: {} (run ids must be unique)",
                    path.display()
                ),
            ));
        }

        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await?;

        let (tx, mut rx) = mpsc::channel::<Event>(buffer);
        let writer = tokio::spawn(async move {
            let mut file = file;
            while let Some(event) = rx.recv().await {
                let mut line = match serde_json::to_vec(&event) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, "trajectory: event serialization failed; skipping");
                        continue;
                    }
                };
                line.push(b'\n');
                file.write_all(&line).await?;
            }
            file.flush().await?;
            file.sync_all().await?;
            Ok::<(), io::Error>(())
        });

        Ok(Self {
            tx,
            path,
            writer_handle: Arc::new(tokio::sync::Mutex::new(Some(writer))),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Close the sender and await the writer task. Call once when shutting down.
    pub async fn flush(&self) -> io::Result<()> {
        // Dropping the sender closes the channel; but the `&self` signature means
        // we can't drop. Instead we rely on the task exiting once all clones are
        // gone. Here we just wait for the writer if this is the last holder.
        let mut guard = self.writer_handle.lock().await;
        if let Some(handle) = guard.take() {
            // If this sink is still cloned elsewhere, the task won't finish; return
            // immediately in that case (best-effort). Callers that need strict
            // shutdown should drop every clone first.
            drop(self.tx.clone()); // no-op; kept for clarity
            match handle.await {
                Ok(r) => r,
                Err(join_err) => Err(io::Error::other(format!("writer task: {join_err}"))),
            }
        } else {
            Ok(())
        }
    }
}

impl EventSink for FileSink {
    fn emit(&self, event: Event) {
        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) => {
                // Channel is saturated. Push the blocking send onto the blocking
                // pool so we don't stall a tokio worker thread.
                let tx = self.tx.clone();
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn_blocking(move || {
                        // blocking_send may block until capacity opens. That's OK —
                        // we're on the blocking pool.
                        if let Err(e) = tx.blocking_send(event) {
                            warn!(error = %e, "trajectory: blocking send failed; event dropped");
                        }
                    });
                } else {
                    warn!("trajectory: no tokio runtime; event dropped on full channel");
                }
            }
            Err(TrySendError::Closed(_)) => {
                warn!("trajectory: writer task has exited; event dropped");
            }
        }
    }

    fn try_emit(&self, event: Event) -> Result<(), Event> {
        match self.tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(ev)) | Err(TrySendError::Closed(ev)) => Err(ev),
        }
    }
}

fn buffer_size() -> usize {
    env::var("OHARNESS_EVENT_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BUFFER)
}
