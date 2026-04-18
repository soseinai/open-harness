//! `Agent` assembly + `AgentBuilder` (§12.5).

use crate::config::AgentConfig;
use crate::loop_trait::{Loop, LoopContext};
use oharness_core::{
    AgentError, ApprovalChannel, BudgetHandle, Cancellation, EventSink, NullApprovalChannel,
    NullBudget, NullSink, RunId, RunOutcome, ScopedEmitter, SharedSink, Task, TrajectoryHandle,
};
use oharness_critic::{CompositeCritic, CriticTrigger, ReflectionInjector};
use oharness_llm::Llm;
use oharness_memory::{MemoryPolicy, Passthrough};
use oharness_tools::context::Workspace;
use oharness_tools::ToolSet;
use oharness_trace::{InMemorySink, RequestTracer, ToolTracer};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub struct Agent {
    llm: Arc<dyn Llm>,
    tools: Arc<dyn ToolSet>,
    memory: Arc<dyn MemoryPolicy>,
    loop_impl: Box<dyn Loop>,
    events: Arc<dyn EventSink>,
    budget: Arc<dyn BudgetHandle>,
    approval: Arc<dyn ApprovalChannel>,
    critics: Option<Arc<CompositeCritic>>,
    critic_trigger: CriticTrigger,
    /// Optional scratch [`Workspace`]. Propagated into
    /// `LoopContext.workspace` and from there into every
    /// `ToolContext` the loop constructs, so the shipped `fs` /
    /// `bash` tools scope to the agent's workspace rather than cwd.
    /// Benchmark adapters (e.g. `oharness-bench-swe`) populate this
    /// from their `LoadedTask.workspace` in the agent factory.
    workspace: Option<Arc<Workspace>>,
    /// Handle stashed for `run_reflexion` — `None` if the builder wasn't
    /// given a [`ReflectionInjector`]. The injector itself (if present)
    /// is also wired into the Llm middleware stack before the agent's
    /// run begins; this field is just the accessor to let
    /// [`run_reflexion`](crate::reflexion) swap its reflection list
    /// between episodes.
    reflection_injector: Option<Arc<ReflectionInjector>>,
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

    /// Returns the agent's [`ReflectionInjector`], or `None` if the
    /// builder wasn't given one. `run_reflexion` uses this accessor to
    /// reconfigure injected reflections between episodes; building an
    /// agent without an injector and then passing it to `run_reflexion`
    /// is a configuration error caught before any episode runs.
    pub fn injector(&self) -> Option<&Arc<ReflectionInjector>> {
        self.reflection_injector.as_ref()
    }

    pub fn critics(&self) -> Option<&Arc<CompositeCritic>> {
        self.critics.as_ref()
    }

    pub fn critic_trigger(&self) -> CriticTrigger {
        self.critic_trigger
    }

    pub fn workspace(&self) -> Option<&Arc<Workspace>> {
        self.workspace.as_ref()
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

        // M1b-δ: wrap the user's Llm + ToolSet in the tracing middleware so
        // llm.* and tool.* events are emitted at the provider / tool boundary
        // instead of from inside the loop. See docs/remaining-work.md §2.4.
        let traced_llm: Arc<dyn Llm> =
            Arc::new(RequestTracer::new(self.llm.clone(), emitter.clone()));
        let traced_tools: Arc<dyn ToolSet> =
            Arc::new(ToolTracer::new(self.tools.clone(), emitter.clone()));

        let loop_ctx = LoopContext {
            llm: traced_llm,
            tools: traced_tools,
            memory: self.memory.clone(),
            critics: self.critics.clone(),
            critic_trigger: self.critic_trigger,
            events: emitter,
            budget: self.budget.clone(),
            cancellation: Cancellation::new(),
            approval: self.approval.clone(),
            workspace: self.workspace.clone(),
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
    critics: Option<Arc<CompositeCritic>>,
    critic_trigger: Option<CriticTrigger>,
    reflection_injector: Option<Arc<ReflectionInjector>>,
    workspace: Option<Arc<Workspace>>,
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

    /// Attach a critic. Typically a [`CompositeCritic`] wrapping one or
    /// more [`oharness_critic::Critic`] implementations; single-critic
    /// setups can construct a composite with one child under
    /// `AggregationPolicy::FirstReject`.
    pub fn with_critics(mut self, critics: Arc<CompositeCritic>) -> Self {
        self.critics = Some(critics);
        self
    }

    pub fn with_critic_trigger(mut self, trigger: CriticTrigger) -> Self {
        self.critic_trigger = Some(trigger);
        self
    }

    /// Attach a [`ReflectionInjector`] for use with
    /// [`run_reflexion`](crate::reflexion). The injector is *not* wired
    /// into the LLM middleware stack automatically — users wire it with
    /// `LlmExt::with_request_layer(injector.clone())` before passing the
    /// LLM to `.with_llm(..)`. This stash is so `run_reflexion` can find
    /// the injector later and swap its reflection list between episodes.
    pub fn with_reflection_injector(mut self, injector: Arc<ReflectionInjector>) -> Self {
        self.reflection_injector = Some(injector);
        self
    }

    /// Attach a [`Workspace`] that every tool call in this agent's run
    /// will be scoped to. The shipped `fs` / `bash` tools respect
    /// `ToolContext::workspace_path()` — without a workspace attached,
    /// they fall back to cwd (surprise-prone for research runs).
    /// Benchmark adapters populate this from their `LoadedTask.workspace`
    /// in the agent factory.
    pub fn with_workspace(mut self, workspace: Arc<Workspace>) -> Self {
        self.workspace = Some(workspace);
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
            critics: self.critics,
            critic_trigger: self.critic_trigger.unwrap_or_default(),
            reflection_injector: self.reflection_injector,
            workspace: self.workspace,
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
