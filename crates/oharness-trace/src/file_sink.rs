//! `FileSink` — JSONL trajectory writer (§9.2, §9.3).
//!
//! M1a: plain JSONL. Payload externalization (sidecar files for >64KiB payloads)
//! is M1b+. The writer task is a tokio task that drains a bounded mpsc channel and
//! appends newline-delimited JSON to the target file.
//!
//! Backpressure model (§4.6): emitters `try_send` first; on `Full`, they spawn a
//! `spawn_blocking` task that calls `blocking_send`. The block stays on the
//! blocking pool, never on a tokio worker thread.
//!
//! Shutdown: `flush()` sends a `oneshot` close signal that the writer
//! task `select!`s on alongside the event channel. The writer drains
//! any still-buffered events via `try_recv` before finalising the
//! file. This lets `flush()` run cleanly against a shared
//! `Arc<FileSink>` — no need for every caller to drop their clone
//! first.

use oharness_core::{Event, EventSink};
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::{self, error::TrySendError, Sender};
use tokio::sync::{oneshot, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::warn;

const DEFAULT_BUFFER: usize = 10_000;

pub struct FileSink {
    tx: Sender<Event>,
    path: PathBuf,
    // One-shot close signal the writer listens for. Wrapped in an
    // `Option<Mutex<...>>` so `flush(&self)` can take the sender out,
    // drop it to fire the signal, and fail silently on second call.
    close_tx: TokioMutex<Option<oneshot::Sender<()>>>,
    // Kept so the writer task has a clean way to be awaited/joined
    // from outside. Also wrapped in an Option so flush() is idempotent.
    writer_handle: Arc<TokioMutex<Option<JoinHandle<io::Result<()>>>>>,
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
        let (close_tx, mut close_rx) = oneshot::channel::<()>();
        let writer = tokio::spawn(async move {
            let mut file = file;
            loop {
                tokio::select! {
                    // Biased so the close branch doesn't starve event writes
                    // when both are ready — we still drain buffered events
                    // after the signal, just not ahead of it.
                    maybe = rx.recv() => {
                        match maybe {
                            Some(event) => write_event(&mut file, event).await?,
                            // All senders dropped — equivalent to a close.
                            None => break,
                        }
                    }
                    _ = &mut close_rx => {
                        // Close requested. Drain anything already queued so
                        // the caller's `emit()` calls that landed before
                        // `flush()` actually hit disk, then finalise.
                        while let Ok(event) = rx.try_recv() {
                            write_event(&mut file, event).await?;
                        }
                        break;
                    }
                }
            }
            file.flush().await?;
            file.sync_all().await?;
            Ok::<(), io::Error>(())
        });

        Ok(Self {
            tx,
            path,
            close_tx: TokioMutex::new(Some(close_tx)),
            writer_handle: Arc::new(TokioMutex::new(Some(writer))),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Close the sink and await the writer task. Safe to call against
    /// a shared `Arc<FileSink>` — the close signal is an internal
    /// one-shot the writer selects on, so outstanding clones of the
    /// `Sender` don't block shutdown.
    ///
    /// Idempotent: the second and subsequent calls are no-ops
    /// returning `Ok(())`. Events emitted after `flush()` returns
    /// will see `TrySendError::Closed` and be warn-dropped — the
    /// sink is single-shutdown.
    pub async fn flush(&self) -> io::Result<()> {
        // Step 1: fire the close signal. Dropping the sender is
        // enough — the writer receives `Err(RecvError)` and treats
        // it the same as a normal close.
        if let Some(sender) = self.close_tx.lock().await.take() {
            drop(sender);
        }

        // Step 2: await the writer. Idempotent via Option::take.
        let mut guard = self.writer_handle.lock().await;
        if let Some(handle) = guard.take() {
            match handle.await {
                Ok(r) => r,
                Err(join_err) => Err(io::Error::other(format!("writer task: {join_err}"))),
            }
        } else {
            Ok(())
        }
    }
}

/// Serialize + append one event to the writer file. Logs and
/// skips on serialisation error rather than failing the whole
/// writer task — one bad event shouldn't drop subsequent writes.
async fn write_event(file: &mut tokio::fs::File, event: Event) -> io::Result<()> {
    let mut line = match serde_json::to_vec(&event) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "trajectory: event serialization failed; skipping");
            return Ok(());
        }
    };
    line.push(b'\n');
    file.write_all(&line).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::event::{EventKind, SchemaVersion, UserLogPayload};
    use oharness_core::{RunId, SpanId};
    use serde_json::json;
    use std::time::Duration;

    fn sample_event(seq: u64, run: RunId) -> Event {
        Event {
            v: SchemaVersion::CURRENT,
            seq,
            run_id: run,
            timestamp: Some(time::OffsetDateTime::now_utc()),
            span_id: SpanId::from("test"),
            parent: None,
            kind: EventKind::UserLog(UserLogPayload {
                namespace: "test".into(),
                data: json!({"ix": seq}),
            }),
            redactions: Vec::new(),
        }
    }

    /// Regression test — `flush()` must complete even when the
    /// caller is still holding `Arc<FileSink>` clones. Before the
    /// close-signal fix, this deadlocked: the writer task only
    /// exited when *every* `Sender<Event>` was dropped, and
    /// `flush(&self)` can't drop `self.tx`.
    #[tokio::test]
    async fn flush_completes_with_outstanding_arc_clones() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("traj.jsonl");
        let sink = Arc::new(FileSink::to_path(&path).await.expect("sink"));

        let run = RunId::new();
        // Hold a clone alive while we flush — the original deadlock scenario.
        let holder = Arc::clone(&sink);
        holder.emit(sample_event(0, run));
        holder.emit(sample_event(1, run));

        // A 2-second timeout is a hard upper bound; the fix completes
        // in single-digit milliseconds. If this test hangs, the close
        // signal regressed.
        let flush_fut = async { sink.flush().await };
        tokio::time::timeout(Duration::from_secs(2), flush_fut)
            .await
            .expect("flush timed out — close-signal regression?")
            .expect("flush returned Err");

        // The clone is still alive; dropping it afterwards must not panic.
        drop(holder);
        // And the file should have both events on disk.
        let contents = tokio::fs::read_to_string(&path).await.expect("read");
        assert_eq!(
            contents.lines().count(),
            2,
            "events missing from file: {contents:?}"
        );
    }

    /// `flush()` is idempotent — second call should no-op cleanly.
    #[tokio::test]
    async fn flush_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("traj.jsonl");
        let sink = FileSink::to_path(&path).await.expect("sink");
        sink.flush().await.expect("first flush");
        sink.flush().await.expect("second flush (idempotent)");
    }

    /// Events emitted after `flush()` completes are dropped with a
    /// warning (writer task is gone), not panicked on.
    #[tokio::test]
    async fn emit_after_flush_is_warned_not_panicked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("traj.jsonl");
        let sink = FileSink::to_path(&path).await.expect("sink");
        sink.flush().await.expect("flush");
        // Should not panic — the writer is gone so the event drops,
        // but emit() handles that gracefully.
        sink.emit(sample_event(0, RunId::new()));
    }
}
