//! `Agent` assembly + `AgentBuilder` (§12.5).

use crate::config::AgentConfig;
use crate::loop_trait::{Loop, LoopContext};
use oharness_core::{
    AgentError, ApprovalChannel, BudgetHandle, Cancellation, EventSink, NullApprovalChannel,
    NullBudget, NullSink, RunId, RunOutcome, ScopedEmitter, SharedSink, Task, TrajectoryHandle,
};
use oharness_llm::Llm;
use oharness_memory::{MemoryPolicy, Passthrough};
use oharness_tools::ToolSet;
use oharness_trace::InMemorySink;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

pub struct Agent {
    llm: Arc<dyn Llm>,
    tools: Arc<dyn ToolSet>,
    memory: Arc<dyn MemoryPolicy>,
    loop_impl: Box<dyn Loop>,
    events: Arc<dyn EventSink>,
    budget: Arc<dyn BudgetHandle>,
    approval: Arc<dyn ApprovalChannel>,
    config: AgentConfig,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    pub fn llm(&self) -> &Arc<dyn Llm> {
        &self.llm
    }

    pub fn tools(&self) -> &Arc<dyn ToolSet> {
        &self.tools
    }

    pub fn sink(&self) -> &Arc<dyn EventSink> {
        &self.events
    }

    pub async fn run(&self, task: Task) -> Result<RunOutcome, AgentError> {
        let run_id = RunId::new();
        let seq = Arc::new(AtomicU64::new(0));

        // Always fan out into an in-memory capture so we can populate the returned
        // TrajectoryHandle. The user's configured sink still sees every event too.
        let capture = InMemorySink::new();
        let fan: Arc<dyn EventSink> = Arc::new(FanOut {
            a: self.events.clone(),
            b: Arc::new(capture.clone()),
        });
        let emitter = ScopedEmitter::new(fan, run_id, seq);

        let loop_ctx = LoopContext {
            llm: self.llm.clone(),
            tools: self.tools.clone(),
            memory: self.memory.clone(),
            events: emitter,
            budget: self.budget.clone(),
            cancellation: Cancellation::new(),
            approval: self.approval.clone(),
            revision_depth_cap: self.config.revision_depth_cap,
            max_turns: self.config.max_turns,
        };

        let mut outcome = self.loop_impl.run(task, &loop_ctx).await?;
        outcome.run_id = run_id;
        outcome.trajectory = TrajectoryHandle::in_memory(capture.events());
        Ok(outcome)
    }
}

#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Arc<dyn Llm>>,
    tools: Option<Arc<dyn ToolSet>>,
    memory: Option<Arc<dyn MemoryPolicy>>,
    loop_impl: Option<Box<dyn Loop>>,
    events: Option<SharedSink>,
    budget: Option<Arc<dyn BudgetHandle>>,
    approval: Option<Arc<dyn ApprovalChannel>>,
    config: AgentConfig,
}

impl AgentBuilder {
    pub fn with_llm(mut self, llm: Arc<dyn Llm>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn with_tools(mut self, tools: Arc<dyn ToolSet>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_memory(mut self, memory: Arc<dyn MemoryPolicy>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_loop(mut self, l: Box<dyn Loop>) -> Self {
        self.loop_impl = Some(l);
        self
    }

    pub fn with_event_sink(mut self, sink: SharedSink) -> Self {
        self.events = Some(sink);
        self
    }

    pub fn with_budget(mut self, budget: Arc<dyn BudgetHandle>) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_approval(mut self, approval: Arc<dyn ApprovalChannel>) -> Self {
        self.approval = Some(approval);
        self
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_max_turns(mut self, n: u32) -> Self {
        self.config.max_turns = n;
        self
    }

    pub fn build(self) -> Result<Agent, AgentError> {
        let llm = self
            .llm
            .ok_or_else(|| AgentError::Configuration("llm is required".into()))?;
        let tools = self
            .tools
            .ok_or_else(|| AgentError::Configuration("tools is required".into()))?;
        // Passthrough default keeps surprise-free behavior.
        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(Passthrough) as Arc<dyn MemoryPolicy>);

        let loop_impl = match self.loop_impl {
            Some(l) => l,
            #[cfg(feature = "react")]
            None => Box::new(crate::react::ReactLoop::default()),
            #[cfg(not(feature = "react"))]
            None => {
                return Err(AgentError::Configuration(
                    "loop is required (no default without `react` feature)".into(),
                ));
            }
        };

        let events = self.events.unwrap_or_else(|| Arc::new(NullSink));
        let budget = self.budget.unwrap_or_else(|| Arc::new(NullBudget));
        let approval = self
            .approval
            .unwrap_or_else(|| Arc::new(NullApprovalChannel));

        Ok(Agent {
            llm,
            tools,
            memory,
            loop_impl,
            events,
            budget,
            approval,
            config: self.config,
        })
    }
}

struct FanOut {
    a: Arc<dyn EventSink>,
    b: Arc<dyn EventSink>,
}

impl EventSink for FanOut {
    fn emit(&self, event: oharness_core::Event) {
        self.a.emit(event.clone());
        self.b.emit(event);
    }
    fn try_emit(&self, event: oharness_core::Event) -> Result<(), oharness_core::Event> {
        // Best-effort: attempt both; return error if either refuses.
        let e1 = self.a.try_emit(event.clone());
        let e2 = self.b.try_emit(event);
        match (e1, e2) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(ev), _) | (_, Err(ev)) => Err(ev),
        }
    }
}
