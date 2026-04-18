//! The `Loop` trait + `LoopContext` (§12.1).

use async_trait::async_trait;
use oharness_core::{
    AgentError, ApprovalChannel, BudgetHandle, Cancellation, RunOutcome, ScopedEmitter, Task,
};
use oharness_critic::{CompositeCritic, CriticTrigger};
use oharness_llm::Llm;
use oharness_memory::MemoryPolicy;
use oharness_tools::context::Workspace;
use oharness_tools::ToolSet;
use std::sync::Arc;

#[async_trait]
pub trait Loop: Send + Sync {
    async fn run(&self, task: Task, ctx: &LoopContext) -> Result<RunOutcome, AgentError>;
}

pub struct LoopContext {
    pub llm: Arc<dyn Llm>,
    pub tools: Arc<dyn ToolSet>,
    pub memory: Arc<dyn MemoryPolicy>,
    /// Optional critic (usually a [`CompositeCritic`] for multi-critic
    /// setups). `None` disables all critic invocations regardless of
    /// trigger setting.
    pub critics: Option<Arc<CompositeCritic>>,
    pub critic_trigger: CriticTrigger,
    pub events: ScopedEmitter,
    pub budget: Arc<dyn BudgetHandle>,
    pub cancellation: Cancellation,
    pub approval: Arc<dyn ApprovalChannel>,
    /// Optional per-run workspace. Propagated into
    /// `ToolContext.workspace` for every tool call so the shipped
    /// `fs` / `bash` tools (which already consult
    /// `ToolContext.workspace_path()`) scope filesystem access to
    /// this directory. `None` leaves tools unscoped (they fall back
    /// to cwd).
    pub workspace: Option<Arc<Workspace>>,
    pub revision_depth_cap: u32,
    pub max_turns: u32,
}
