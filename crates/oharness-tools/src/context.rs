//! `ToolContext` and `Workspace` (§7.2).

use oharness_core::{
    ApprovalChannel, BudgetHandle, Cancellation, EventSink, MetadataMap, NullApprovalChannel,
    NullBudget, NullSink,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Threaded through every `ToolSet::execute` call. Carries cross-cutting concerns —
/// event emission, budget, cancellation, approval, workspace scoping.
pub struct ToolContext {
    pub events: Arc<dyn EventSink>,
    pub budget: Arc<dyn BudgetHandle>,
    pub cancellation: Cancellation,
    pub approval: Arc<dyn ApprovalChannel>,
    pub workspace: Option<Arc<Workspace>>,
    /// Reverse-DNS-namespaced extras — like request extensions.
    pub extensions: MetadataMap,
}

impl ToolContext {
    /// Null-everything context, useful for tests and M1a smoke paths.
    pub fn null() -> Self {
        Self {
            events: Arc::new(NullSink),
            budget: Arc::new(NullBudget),
            cancellation: Cancellation::new(),
            approval: Arc::new(NullApprovalChannel),
            workspace: None,
            extensions: MetadataMap::new(),
        }
    }

    pub fn workspace_path(&self) -> Option<&Path> {
        self.workspace.as_ref().map(|w| w.path.as_path())
    }
}

/// A scratch directory + optional cleanup strategy. Benchmarks use this to hand
/// per-task directories to tools.
pub struct Workspace {
    pub path: PathBuf,
    /// Cleanup policy; run on `teardown()` or on `Drop` (sync cleanups only run
    /// inline; async cleanups on Drop are best-effort).
    cleanup: std::sync::Mutex<Option<WorkspaceCleanup>>,
}

impl Workspace {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            cleanup: std::sync::Mutex::new(None),
        }
    }

    pub fn with_sync_cleanup(
        mut self,
        f: impl FnOnce() + Send + 'static,
    ) -> Self {
        self.cleanup = std::sync::Mutex::new(Some(WorkspaceCleanup::Sync(Box::new(f))));
        self
    }

    /// Drop the workspace, running its cleanup.
    pub async fn teardown(self) {
        let cleanup = {
            let mut guard = self.cleanup.lock().unwrap();
            guard.take()
        };
        if let Some(cleanup) = cleanup {
            match cleanup {
                WorkspaceCleanup::Sync(f) => f(),
                WorkspaceCleanup::Async(fut) => fut.await,
            }
        }
    }
}

impl std::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workspace").field("path", &self.path).finish()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.cleanup.lock() {
            if let Some(cleanup) = guard.take() {
                match cleanup {
                    WorkspaceCleanup::Sync(f) => f(),
                    // On drop, async cleanups are best-effort. If there's a runtime,
                    // we spawn; otherwise we silently drop the work. Benchmarks that
                    // require guaranteed async cleanup call `teardown().await`.
                    WorkspaceCleanup::Async(fut) => {
                        if let Ok(handle) = tokio::runtime::Handle::try_current() {
                            handle.spawn(fut);
                        }
                    }
                }
            }
        }
    }
}

pub enum WorkspaceCleanup {
    Sync(Box<dyn FnOnce() + Send>),
    Async(futures::future::BoxFuture<'static, ()>),
}
