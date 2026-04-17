//! The `Loop` trait + `LoopContext` (§12.1).

use async_trait::async_trait;
use oharness_core::{
    AgentError, ApprovalChannel, BudgetHandle, Cancellation, RunOutcome, ScopedEmitter, Task,
};
use oharness_llm::Llm;
use oharness_memory::MemoryPolicy;
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
    pub events: ScopedEmitter,
    pub budget: Arc<dyn BudgetHandle>,
    pub cancellation: Cancellation,
    pub approval: Arc<dyn ApprovalChannel>,
    pub revision_depth_cap: u32,
    pub max_turns: u32,
}
