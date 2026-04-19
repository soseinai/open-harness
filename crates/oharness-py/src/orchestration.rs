//! Orchestration bindings — expose `Agent`, `AgentBuilder`, `Task`,
//! shipped `Loop` impls, shipped `ToolSet` impls, and shipped
//! `EventSink` impls so Python can drive a full run end-to-end.
//!
//! Naming convention: adapter classes (for user-written Llm /
//! Critic / etc. types) keep the `Py*` prefix because they're the
//! "Python adapter for <Rust trait>". Orchestration classes drop
//! the prefix in Python — they're first-class bindings, not
//! adapters. So Python sees `oharness.Agent`, `oharness.Task`,
//! `oharness.ReactLoop` etc.
//!
//! Internally these types hold `Option<T>` or `Arc<T>` so the
//! builder chain can take-and-replace without moving out of a
//! pyclass. If a user calls `builder.with_llm(..)` twice with the
//! same wrapper, it still works — we clone the internals rather
//! than consume them.

use crate::{
    EpisodeWire, OutcomeWire, PyBridgeError, PyCritic, PyLlm, PyMemoryPolicy, PyReflector,
    PyRequestLayer, PyResponseLayer, PyTaskEvaluator, PyToolSet as UserPyToolSet,
};
use async_trait::async_trait;
use oharness_budget::{BudgetMiddleware, TokenBudget};
use oharness_core::{
    AgentError, BudgetHandle, CompletionRequest, CompletionResponse, EventSink, LlmCapabilities,
    RunOutcome, Task, TaskEvaluator,
};
use oharness_critic::shipped::LlmJudgeCritic;
use oharness_critic::{
    AggregationPolicy, AssessmentContext, CompositeCritic, Critic, CriticVerdict,
    ReflectionInjector, Reflector,
};
use oharness_llm::{ChunkStream, Llm, LlmError, LlmExt};
use oharness_loop::{
    run_reflexion as rust_run_reflexion, Agent as RustAgent, AgentBuilder, ConversationLoop, Loop,
    ReactLoop, ScriptedUserSimulator,
};
use oharness_tools::fs::FsToolSet;
use oharness_tools::ToolSet;
use oharness_trace::{DriftPolicy, FileSink, InMemorySink, ReplayLlm, ReplayMode};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Lazy-constructed tokio runtime used by `Agent.run()` when
/// called synchronously from Python. One shared multi-threaded
/// runtime across all agent runs in the process; cheaper than
/// spinning a fresh one per `run()` call.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("oharness: failed to build tokio runtime")
    })
}

// ======================================================================
// PyTask — wraps oharness_core::Task.
// ======================================================================

/// A task to run an agent against. The minimum shape is an
/// instruction string; optional fields (id, metadata,
/// attachments) will be added as the Python surface grows.
#[pyclass(name = "Task")]
pub struct PyTask {
    pub(crate) inner: Task,
}

#[pymethods]
impl PyTask {
    #[new]
    fn new(instruction: String) -> Self {
        Self {
            inner: Task::new(instruction),
        }
    }

    fn __repr__(&self) -> String {
        format!("Task(instruction={:?})", self.inner.instruction)
    }

    /// The task's instruction text.
    #[getter]
    fn instruction(&self) -> &str {
        &self.inner.instruction
    }

    /// JSON-serialise the full task (instruction + id +
    /// metadata + attachments). Symmetric with `from_json`.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Task: encode: {e}")))
    }

    /// Reconstruct a task from its JSON form. Useful when a
    /// benchmark / harness hands tasks through as JSON.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: Task = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("Task: decode: {e}")))?;
        Ok(Self { inner })
    }
}

// ======================================================================
// PyReactLoop — shipped ReactLoop wrapper. Consumed by `Agent.builder()
// .with_loop(..)`.
// ======================================================================

/// The shipped reason-act-observe loop. Created empty,
/// consumed when passed to `Agent.builder().with_loop(..)` —
/// the builder takes ownership of the underlying Rust struct.
/// A subsequent `with_loop(..)` call on the same Python object
/// raises `RuntimeError`.
#[pyclass(name = "ReactLoop")]
pub struct PyReactLoop {
    pub(crate) inner: std::sync::Mutex<Option<ReactLoop>>,
}

#[pymethods]
impl PyReactLoop {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(ReactLoop::new())),
        }
    }

    fn __repr__(&self) -> &'static str {
        "ReactLoop()"
    }
}

// ======================================================================
// PyFsToolSet — shipped FsToolSet wrapper.
// ======================================================================

/// The shipped filesystem `ToolSet` (`fs_list`, `fs_read`,
/// `fs_write`, `fs_stat`). Constructor takes no arguments; the
/// tools respect the agent's `workspace_path` when scoped.
#[pyclass(name = "FsToolSet")]
pub struct PyFsToolSet {
    pub(crate) inner: Arc<FsToolSet>,
}

#[pymethods]
impl PyFsToolSet {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(FsToolSet::new()),
        }
    }

    fn __repr__(&self) -> &'static str {
        "FsToolSet()"
    }

    /// Names of the tools this set exposes.
    fn tool_names(&self) -> Vec<String> {
        self.inner.specs().iter().map(|s| s.name.clone()).collect()
    }
}

// ======================================================================
// PyInMemorySink — shipped InMemorySink wrapper.
// ======================================================================

/// Event sink that captures every emitted event into an
/// in-process `Vec`. Useful for tests and small demos; the
/// shared clone the agent uses writes the same list the
/// Python-side holder reads.
#[pyclass(name = "InMemorySink")]
pub struct PyInMemorySink {
    pub(crate) inner: Arc<InMemorySink>,
}

#[pymethods]
impl PyInMemorySink {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemorySink::new()),
        }
    }

    fn __repr__(&self) -> &'static str {
        "InMemorySink()"
    }

    /// Number of events captured so far.
    fn len(&self) -> usize {
        self.inner.events().len()
    }

    /// JSON-serialise every captured event as a JSON array
    /// string. Python users typically parse this with
    /// `json.loads` and inspect per-event.
    fn events_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.events())
            .map_err(|e| PyRuntimeError::new_err(format!("events: encode: {e}")))
    }
}

// ======================================================================
// PyCompositeCritic — shipped CompositeCritic wrapper.
// ======================================================================

/// Chain multiple critics with an aggregation policy
/// (`"first_reject"`, `"all_must_accept"`, `"majority_vote"`).
/// Use `.push(critic)` to add a PyCritic, then hand the result
/// to `Agent.builder().with_critics(..)`.
///
/// Weighted voting isn't exposed yet — pass a custom
/// `CompositeCritic` from Rust if you need it.
#[pyclass(name = "CompositeCritic")]
pub struct PyCompositeCritic {
    pub(crate) inner: std::sync::Mutex<Option<CompositeCritic>>,
}

#[pymethods]
impl PyCompositeCritic {
    /// Construct an empty composite. `name` identifies the
    /// composite in trajectory events; `policy` is one of
    /// `"first_reject"`, `"all_must_accept"`, `"majority_vote"`.
    #[new]
    #[pyo3(signature = (name, policy = "first_reject".to_string()))]
    fn new(name: String, policy: String) -> PyResult<Self> {
        let policy = match policy.as_str() {
            "first_reject" => AggregationPolicy::FirstReject,
            "all_must_accept" => AggregationPolicy::AllMustAccept,
            "majority_vote" => AggregationPolicy::MajorityVote,
            other => {
                return Err(PyValueError::new_err(format!(
                    "CompositeCritic: unknown aggregation policy {other:?} — \
                     expected one of \"first_reject\", \"all_must_accept\", \
                     \"majority_vote\""
                )));
            }
        };
        Ok(Self {
            inner: std::sync::Mutex::new(Some(CompositeCritic::new(name, policy))),
        })
    }

    /// Append a critic to the chain. Accepts either a
    /// `PyCritic` (wrapping a user-written Python class) or a
    /// `LlmJudgeCritic` (shipped). Returns the composite for
    /// chaining.
    fn push<'py>(
        slf: PyRefMut<'py, Self>,
        critic: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let boxed = extract_critic(&critic)?;
        let mut guard = slf
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("CompositeCritic: lock: {e}")))?;
        let Some(inner) = guard.take() else {
            return Err(PyRuntimeError::new_err(
                "CompositeCritic: already consumed by .build(); construct a new one",
            ));
        };
        *guard = Some(inner.push(boxed));
        drop(guard);
        Ok(slf)
    }
}

// ======================================================================
// PyAgentBuilder + PyAgent — the top-level run surface.
// ======================================================================

/// Fluent builder for a `PyAgent`. Constructed via
/// `Agent.builder()`. Every `.with_*` call mutates the builder
/// and returns it for chaining; `.build()` consumes the builder
/// and produces a runnable `Agent`.
///
/// The builder mirrors the Rust `AgentBuilder` surface closely;
/// the Python side just shifts the "take ownership" semantics
/// into an internal `Option` so Python can re-use the same
/// `builder` variable between calls.
#[pyclass(name = "AgentBuilder")]
pub struct PyAgentBuilder {
    inner: Option<AgentBuilder>,
}

#[pymethods]
impl PyAgentBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(AgentBuilder::default()),
        }
    }

    fn with_llm<'py>(
        mut slf: PyRefMut<'py, Self>,
        llm: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let llm_arc = extract_llm(&llm)?;
        let builder = take_inner(&mut slf.inner, "with_llm")?;
        slf.inner = Some(builder.with_llm(llm_arc));
        Ok(slf)
    }

    /// Accepts either a shipped `FsToolSet` or a user-written
    /// `PyToolSet`. Internally the builder holds `Arc<dyn
    /// ToolSet>`, so both paths land in the same slot.
    fn with_tools<'py>(
        mut slf: PyRefMut<'py, Self>,
        tools: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let tools_arc = extract_toolset(tools)?;
        let builder = take_inner(&mut slf.inner, "with_tools")?;
        slf.inner = Some(builder.with_tools(tools_arc));
        Ok(slf)
    }

    /// Attach a `Loop` to the agent. Accepts either a
    /// `ReactLoop` or a `ConversationLoop` (the two shipped loop
    /// impls). The loop is consumed; a second `.with_loop(..)`
    /// call on the same Python handle raises `RuntimeError`.
    fn with_loop<'py>(
        mut slf: PyRefMut<'py, Self>,
        loop_: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let boxed = extract_loop(&loop_)?;
        let builder = take_inner(&mut slf.inner, "with_loop")?;
        slf.inner = Some(builder.with_loop(boxed));
        Ok(slf)
    }

    fn with_memory<'py>(
        mut slf: PyRefMut<'py, Self>,
        memory: PyRef<'_, PyMemoryPolicy>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        let cloned = PyMemoryPolicy {
            py_obj: memory.py_obj.clone_ref(py),
            name: memory.name.clone(),
        };
        let mem_arc: Arc<dyn oharness_memory::MemoryPolicy> = Arc::new(cloned);
        let builder = take_inner(&mut slf.inner, "with_memory")?;
        slf.inner = Some(builder.with_memory(mem_arc));
        Ok(slf)
    }

    fn with_critics<'py>(
        mut slf: PyRefMut<'py, Self>,
        critics: PyRef<'_, PyCompositeCritic>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut guard = critics
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("CompositeCritic: lock: {e}")))?;
        let Some(composite) = guard.take() else {
            return Err(PyRuntimeError::new_err(
                "CompositeCritic: already consumed; construct a new one",
            ));
        };
        drop(guard);
        let builder = take_inner(&mut slf.inner, "with_critics")?;
        slf.inner = Some(builder.with_critics(Arc::new(composite)));
        Ok(slf)
    }

    /// Attach an `EventSink`. Accepts either a shipped
    /// `InMemorySink` (for tests / small demos) or a `FileSink`
    /// (for production JSONL persistence).
    fn with_event_sink<'py>(
        mut slf: PyRefMut<'py, Self>,
        sink: Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let sink_arc = extract_event_sink(&sink)?;
        let builder = take_inner(&mut slf.inner, "with_event_sink")?;
        slf.inner = Some(builder.with_event_sink(sink_arc));
        Ok(slf)
    }

    fn with_max_turns<'py>(
        mut slf: PyRefMut<'py, Self>,
        n: u32,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let builder = take_inner(&mut slf.inner, "with_max_turns")?;
        slf.inner = Some(builder.with_max_turns(n));
        Ok(slf)
    }

    /// Stash a `ReflectionInjector` on the agent so
    /// `run_reflexion` can find it between episodes. The
    /// injector must also be wired into the LLM stack via a
    /// `LayeredLlm(inner, request_layers=[injector_as_layer])`
    /// — but `ReflectionInjector` isn't a `PyRequestLayer`; see
    /// the `reflexion_run.py` example for the exact pattern
    /// (usually: construct the injector, layer it into the LLM
    /// inside a `LayeredLlm`, and also hand the same injector
    /// here).
    fn with_reflection_injector<'py>(
        mut slf: PyRefMut<'py, Self>,
        injector: PyRef<'_, PyReflectionInjector>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let builder = take_inner(&mut slf.inner, "with_reflection_injector")?;
        slf.inner = Some(builder.with_reflection_injector(injector.inner.clone()));
        Ok(slf)
    }

    /// Finalise the builder into a runnable `Agent`. Fails
    /// loudly if a required slot wasn't provided (e.g., no LLM
    /// or no loop).
    fn build(&mut self) -> PyResult<PyAgent> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AgentBuilder: already consumed by .build()"))?;
        let agent = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Agent build: {e}")))?;
        Ok(PyAgent {
            inner: Arc::new(agent),
        })
    }
}

/// A built agent ready to run. Hold it across multiple
/// `run(task)` calls if you want the same LLM / tools /
/// middleware composition for each run.
#[pyclass(name = "Agent")]
pub struct PyAgent {
    pub(crate) inner: Arc<RustAgent>,
}

#[pymethods]
impl PyAgent {
    /// Entry point that mirrors the Rust `Agent::builder()`
    /// idiom — `oharness.Agent.builder().with_llm(..)...build()`.
    #[staticmethod]
    fn builder() -> PyAgentBuilder {
        PyAgentBuilder::new()
    }

    /// Run the agent against `task`. Blocks synchronously from
    /// Python's perspective — internally spins a shared tokio
    /// runtime and releases the GIL for the duration so
    /// Python-defined adapters can re-acquire it via
    /// `Python::with_gil`.
    ///
    /// Returns a JSON-serialised `RunOutcome` — parse with
    /// `json.loads`. Common fields: `termination`, `usage`,
    /// `final_messages`, `run_id`.
    fn run(&self, py: Python<'_>, task: PyRef<'_, PyTask>) -> PyResult<String> {
        let agent = self.inner.clone();
        let task_clone = task.inner.clone();
        py.allow_threads(move || {
            let outcome: Result<RunOutcome, AgentError> =
                shared_runtime().block_on(agent.run(task_clone));
            let outcome = outcome.map_err(|e| {
                PyRuntimeError::new_err(format!("Agent::run: {e}"))
            })?;
            // Serialize via OutcomeWire so we skip the in-memory
            // `TrajectoryHandle` (which refuses Serialize on the
            // in-memory variant). Consumers who need the raw
            // trajectory should pass their own FileSink via
            // `.with_event_sink(..)` — the JSONL file on disk is
            // the canonical shape anyway.
            let wire = OutcomeWire::from(&outcome);
            serde_json::to_string(&wire)
                .map_err(|e| PyRuntimeError::new_err(format!("RunOutcome: encode: {e}")))
        })
    }
}

// ======================================================================
// PyFileSink — shipped FileSink wrapper (JSONL trajectory writer).
// ======================================================================

/// JSONL trajectory writer backed by a bounded mpsc channel.
/// Writes events to a file path you specify. Call `.flush()`
/// before the program exits to drain the writer task.
///
/// The sink itself is safe to hand to both the agent and keep
/// a Python-side reference — `flush()` signals completion via
/// an internal close channel (fix landed in the M4 security
/// pass).
#[pyclass(name = "FileSink")]
pub struct PyFileSink {
    pub(crate) inner: Arc<FileSink>,
}

#[pymethods]
impl PyFileSink {
    /// Open a new JSONL trajectory file at `path`. The file
    /// must not already exist (each run writes its own). Raises
    /// `RuntimeError` on I/O error.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let sink = py.allow_threads(|| {
            shared_runtime()
                .block_on(FileSink::to_path(path))
                .map_err(|e| PyRuntimeError::new_err(format!("FileSink: {e}")))
        })?;
        Ok(Self {
            inner: Arc::new(sink),
        })
    }

    /// Drain the writer task — blocks until every queued event
    /// has hit disk. Idempotent; safe to call more than once.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(move || {
            shared_runtime()
                .block_on(inner.flush())
                .map_err(|e| PyRuntimeError::new_err(format!("FileSink.flush: {e}")))
        })
    }

    /// The on-disk path the sink is writing to.
    #[getter]
    fn path(&self) -> String {
        self.inner.path().display().to_string()
    }

    fn __repr__(&self) -> String {
        format!("FileSink(path={:?})", self.inner.path().display().to_string())
    }
}

// ======================================================================
// PyReplayLlm — shipped ReplayLlm wrapper.
// ======================================================================

/// Replay a recorded trajectory as if it were a live `Llm`.
/// Constructors read from a JSONL file (`.from_path(path)`) or
/// from an events-array JSON string (`.from_events_json(s)`).
///
/// `mode` and `drift` default to positional replay with
/// warn-and-continue behaviour — the common case for
/// reproducible runs. For strict byte-for-byte input matching
/// use `mode="strict"`.
#[pyclass(name = "ReplayLlm")]
pub struct PyReplayLlm {
    pub(crate) inner: Arc<ReplayLlm>,
}

#[pymethods]
impl PyReplayLlm {
    /// Load a trajectory from a JSONL file.
    ///
    /// `mode`: `"positional"` (default) or `"strict"`.
    /// `drift`: `"warn_and_continue"` (default) or `"fail"`.
    #[staticmethod]
    #[pyo3(signature = (path, mode = "positional".to_string(), drift = "warn_and_continue".to_string()))]
    fn from_path(
        py: Python<'_>,
        path: PathBuf,
        mode: String,
        drift: String,
    ) -> PyResult<Self> {
        let mode = parse_replay_mode(&mode)?;
        let drift = parse_drift_policy(&drift)?;
        let replay = py.allow_threads(move || {
            shared_runtime()
                .block_on(ReplayLlm::from_path(path, mode, drift))
                .map_err(|e| PyRuntimeError::new_err(format!("ReplayLlm: {e}")))
        })?;
        Ok(Self {
            inner: Arc::new(replay),
        })
    }

    /// Load a trajectory from a JSON string (an array of events,
    /// matching `InMemorySink.events_json()`'s output).
    #[staticmethod]
    #[pyo3(signature = (events_json, mode = "positional".to_string(), drift = "warn_and_continue".to_string()))]
    fn from_events_json(
        events_json: &str,
        mode: String,
        drift: String,
    ) -> PyResult<Self> {
        let events: Vec<oharness_core::Event> = serde_json::from_str(events_json)
            .map_err(|e| PyValueError::new_err(format!("ReplayLlm: decode events: {e}")))?;
        let mode = parse_replay_mode(&mode)?;
        let drift = parse_drift_policy(&drift)?;
        let replay = ReplayLlm::from_events(events, mode, drift)
            .map_err(|e| PyRuntimeError::new_err(format!("ReplayLlm: {e}")))?;
        Ok(Self {
            inner: Arc::new(replay),
        })
    }

    fn __repr__(&self) -> &'static str {
        "ReplayLlm(...)"
    }
}

fn parse_replay_mode(s: &str) -> PyResult<ReplayMode> {
    match s {
        "positional" => Ok(ReplayMode::Positional),
        "strict" => Ok(ReplayMode::Strict),
        other => Err(PyValueError::new_err(format!(
            "ReplayLlm: unknown mode {other:?} — expected \"positional\" or \"strict\""
        ))),
    }
}

fn parse_drift_policy(s: &str) -> PyResult<DriftPolicy> {
    match s {
        "warn_and_continue" => Ok(DriftPolicy::WarnAndContinue),
        "fail" => Ok(DriftPolicy::Fail),
        other => Err(PyValueError::new_err(format!(
            "ReplayLlm: unknown drift policy {other:?} — expected \"warn_and_continue\" or \"fail\""
        ))),
    }
}

// ======================================================================
// PyTokenBudget — shipped TokenBudget wrapper.
// ======================================================================

/// Token-based `BudgetHandle`. Cap on input tokens, output
/// tokens, or the sum. Trip the cap → the underlying
/// `Llm::complete()` returns
/// `LlmError::Provider(BudgetExceeded)`, which the loop
/// converts to `Termination::Failed`.
#[pyclass(name = "TokenBudget")]
pub struct PyTokenBudget {
    pub(crate) inner: Arc<TokenBudget>,
}

#[pymethods]
impl PyTokenBudget {
    /// Cap total tokens (input + output).
    #[staticmethod]
    fn input_plus_output(cap: u64) -> Self {
        Self {
            inner: Arc::new(TokenBudget::input_plus_output(cap)),
        }
    }

    /// JSON-serialised snapshot: `{consumed, remaining}`. Parse
    /// with `json.loads` to inspect.
    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(format!("snapshot: encode: {e}")))
    }

    fn __repr__(&self) -> &'static str {
        "TokenBudget(...)"
    }
}

// ======================================================================
// PyBudgetMiddleware — wraps an Llm with a BudgetHandle.
// ======================================================================

/// Wrap an Llm with budget accounting. Constructor takes a
/// Python-facing Llm wrapper (PyLlm / ReplayLlm / another
/// BudgetMiddleware) plus a budget handle.
///
/// The result implements `Llm`, so you can pass it to
/// `Agent.builder().with_llm(..)` — or wrap it again with more
/// middleware.
#[pyclass(name = "BudgetMiddleware")]
pub struct PyBudgetMiddleware {
    pub(crate) inner: Arc<dyn Llm>,
}

#[pymethods]
impl PyBudgetMiddleware {
    #[new]
    fn new(inner_llm: Bound<'_, PyAny>, budget: PyRef<'_, PyTokenBudget>) -> PyResult<Self> {
        let llm = extract_llm(&inner_llm)?;
        let budget_handle: Arc<dyn BudgetHandle> = budget.inner.clone();
        let wrapped = BudgetMiddleware::new(llm, budget_handle);
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    fn __repr__(&self) -> &'static str {
        "BudgetMiddleware(...)"
    }
}

// ======================================================================
// PyLayeredLlm — compose request + response layers around an Llm.
// ======================================================================

/// Wrap an Llm with zero or more `RequestLayer` / `ResponseLayer`
/// layers. Layers are applied in the order given — outermost
/// layer seen first by the caller.
///
/// `FullLayer` is intentionally not exposed from Python; its
/// `BoxFuture`-wrapping contract doesn't round-trip well across
/// the GIL. For before/after hooks, compose `RequestLayer` +
/// `ResponseLayer` — you get the same observation surface.
#[pyclass(name = "LayeredLlm")]
pub struct PyLayeredLlm {
    pub(crate) inner: Arc<dyn Llm>,
}

#[pymethods]
impl PyLayeredLlm {
    /// Build a new layered Llm.
    ///
    /// `request_layers` and `response_layers` accept Python lists
    /// of `PyRequestLayer` / `PyResponseLayer` instances. Both
    /// default to `None` (no layers).
    #[new]
    #[pyo3(signature = (inner, request_layers = None, response_layers = None))]
    fn new(
        inner: Bound<'_, PyAny>,
        request_layers: Option<Bound<'_, PyList>>,
        response_layers: Option<Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let py = inner.py();
        let mut llm: Arc<dyn Llm> = extract_llm(&inner)?;

        if let Some(layers) = request_layers {
            for layer in layers.iter() {
                // Accept either a PyRequestLayer (user-written) or
                // a ReflectionInjector (shipped RequestLayer impl).
                if let Ok(req_layer) = layer.extract::<PyRef<'_, PyRequestLayer>>() {
                    let cloned = PyRequestLayer {
                        py_obj: req_layer.py_obj.clone_ref(py),
                        name: req_layer.name.clone(),
                    };
                    llm = Arc::new(llm.with_request_layer(cloned));
                } else if let Ok(injector) =
                    layer.extract::<PyRef<'_, PyReflectionInjector>>()
                {
                    llm = Arc::new(llm.with_request_layer(injector.inner.clone()));
                } else {
                    return Err(PyValueError::new_err(
                        "request_layers entries must be oharness.PyRequestLayer \
                         or oharness.ReflectionInjector instances",
                    ));
                }
            }
        }

        if let Some(layers) = response_layers {
            for layer in layers.iter() {
                let res_layer = layer.extract::<PyRef<'_, PyResponseLayer>>().map_err(|_| {
                    PyValueError::new_err(
                        "response_layers entries must be oharness.PyResponseLayer instances",
                    )
                })?;
                let cloned = PyResponseLayer {
                    py_obj: res_layer.py_obj.clone_ref(py),
                    name: res_layer.name.clone(),
                    stream_mode: res_layer.stream_mode,
                };
                llm = Arc::new(llm.with_response_layer(cloned));
            }
        }

        Ok(Self { inner: llm })
    }

    fn __repr__(&self) -> &'static str {
        "LayeredLlm(...)"
    }
}

// ======================================================================
// PyLlmJudgeCritic — shipped LlmJudgeCritic wrapper.
// ======================================================================

/// Prompt a judge LLM with a rubric and parse a numeric score
/// from its response (`SCORE: <0..1>`). Scores ≥ threshold →
/// `AcceptWithNote`; below → `Reject`.
///
/// The critic itself is `Clone + Send + Sync` via its
/// `Arc<dyn Llm>` judge handle — so we can hand the same
/// instance to `CompositeCritic.push(..)` multiple times if
/// needed (though a single push is the common case).
#[pyclass(name = "LlmJudgeCritic")]
pub struct PyLlmJudgeCritic {
    pub(crate) inner: Arc<LlmJudgeCritic>,
}

#[pymethods]
impl PyLlmJudgeCritic {
    /// Build a judge critic.
    ///
    /// - `judge`: any Llm-wrapping pyclass (PyLlm / ReplayLlm /
    ///   LayeredLlm / BudgetMiddleware) — the critic calls it to
    ///   render + grade each turn.
    /// - `rubric`: plain-English scoring guidance; shown to the
    ///   judge as part of its prompt.
    /// - `threshold`: accept ↔ reject cutoff in `[0.0, 1.0]`.
    /// - `name`: identifier used in `critic.assessed` events.
    #[new]
    #[pyo3(signature = (judge, rubric, threshold, name = "llm-judge".to_string()))]
    fn new(
        judge: Bound<'_, PyAny>,
        rubric: String,
        threshold: f32,
        name: String,
    ) -> PyResult<Self> {
        let judge_llm = extract_llm(&judge)?;
        let critic = LlmJudgeCritic::new(judge_llm, rubric, threshold).with_name(name);
        Ok(Self {
            inner: Arc::new(critic),
        })
    }

    fn __repr__(&self) -> &'static str {
        "LlmJudgeCritic(...)"
    }
}

// ======================================================================
// PyReflectionInjector — shipped ReflectionInjector wrapper.
// ======================================================================

/// The middleware that carries reflections produced by a
/// `Reflector` into the next episode's system prompt. Create
/// once, share the same instance with both the agent builder
/// (`.with_reflection_injector(..)`) and the LLM's request-layer
/// chain.
///
/// Internally holds an `Arc<ReflectionInjector>` — cloning the
/// pyclass is cheap (Arc bump). `run_reflexion` uses this
/// handle to call `set_reflections` / `bump_episode` between
/// episodes.
#[pyclass(name = "ReflectionInjector")]
#[derive(Clone)]
pub struct PyReflectionInjector {
    pub(crate) inner: Arc<ReflectionInjector>,
}

#[pymethods]
impl PyReflectionInjector {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(ReflectionInjector::new()),
        }
    }

    /// Number of reflections currently staged for injection.
    fn reflection_count(&self) -> usize {
        self.inner.reflection_count()
    }

    fn __repr__(&self) -> &'static str {
        "ReflectionInjector()"
    }
}

// ======================================================================
// PyScriptedUserSimulator — shipped ScriptedUserSimulator wrapper.
// ======================================================================

/// A `UserSimulator` that replays a fixed sequence of user
/// utterances. The first entry is returned from
/// `initial_message`; each subsequent `respond` call returns the
/// next entry. When the script is exhausted, the simulator
/// emits `EndConversation`.
///
/// Useful for tests, reproducible evaluation runs, and
/// conversation-loop smoke paths with no LLM on the user side.
#[pyclass(name = "ScriptedUserSimulator")]
pub struct PyScriptedUserSimulator {
    pub(crate) inner: std::sync::Mutex<Option<ScriptedUserSimulator>>,
}

#[pymethods]
impl PyScriptedUserSimulator {
    #[new]
    #[pyo3(signature = (script, name = "scripted-user".to_string()))]
    fn new(script: Vec<String>, name: String) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(
                ScriptedUserSimulator::new(script).with_name(name),
            )),
        }
    }

    fn __repr__(&self) -> &'static str {
        "ScriptedUserSimulator(...)"
    }
}

// ======================================================================
// PyConversationLoop — shipped ConversationLoop wrapper.
// ======================================================================

/// Alternates assistant turns with a `UserSimulator`. Constructed
/// with a simulator (scripted here; `LlmUserSimulator` wrapper
/// can follow when needed). Optionally carries a system prompt.
///
/// Like `ReactLoop`, the inner `ConversationLoop` is boxed up as
/// a `dyn Loop` once handed to `AgentBuilder.with_loop(..)` — so
/// subsequent `.with_loop(..)` calls on the same Python handle
/// raise `RuntimeError`.
#[pyclass(name = "ConversationLoop")]
pub struct PyConversationLoop {
    pub(crate) inner:
        std::sync::Mutex<Option<ConversationLoop<ScriptedUserSimulator>>>,
}

#[pymethods]
impl PyConversationLoop {
    /// Construct from a `ScriptedUserSimulator`. The simulator
    /// is consumed — don't re-use the same `ScriptedUserSimulator`
    /// across multiple conversation loops.
    ///
    /// `system_prompt` is optional; defaults to `None`. If set,
    /// the loop prepends it to every turn's `CompletionRequest`.
    #[new]
    #[pyo3(signature = (simulator, system_prompt = None))]
    fn new(
        simulator: PyRef<'_, PyScriptedUserSimulator>,
        system_prompt: Option<String>,
    ) -> PyResult<Self> {
        let mut guard = simulator
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("ScriptedUserSimulator: lock: {e}")))?;
        let Some(sim) = guard.take() else {
            return Err(PyRuntimeError::new_err(
                "ScriptedUserSimulator: already consumed; construct a new one",
            ));
        };
        let mut loop_ = ConversationLoop::new(sim);
        if let Some(prompt) = system_prompt {
            loop_ = loop_.with_system_prompt(prompt);
        }
        Ok(Self {
            inner: std::sync::Mutex::new(Some(loop_)),
        })
    }

    fn __repr__(&self) -> &'static str {
        "ConversationLoop(...)"
    }
}

// ======================================================================
// Helpers
// ======================================================================

fn take_inner(slot: &mut Option<AgentBuilder>, method: &'static str) -> PyResult<AgentBuilder> {
    slot.take().ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "AgentBuilder: .{method}(..) called after .build() already consumed the builder"
        ))
    })
}

/// Accept either a user-written `PyCritic` or a shipped
/// `PyLlmJudgeCritic`; produce a `Box<dyn Critic>` suitable for
/// `CompositeCritic.push`. Relies on the blanket `Arc<T>: Critic`
/// impl in `oharness-critic/src/critic.rs` so the shipped
/// `Arc<LlmJudgeCritic>` handle can cross the trait-object
/// boundary without a shim.
fn extract_critic(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Critic>> {
    let py = obj.py();
    if let Ok(c) = obj.extract::<PyRef<'_, PyCritic>>() {
        let cloned = PyCritic {
            py_obj: c.py_obj.clone_ref(py),
            name: c.name.clone(),
        };
        return Ok(Box::new(cloned));
    }
    if let Ok(j) = obj.extract::<PyRef<'_, PyLlmJudgeCritic>>() {
        return Ok(Box::new(j.inner.clone()));
    }
    Err(PyValueError::new_err(
        "expected oharness.PyCritic or oharness.LlmJudgeCritic",
    ))
}

/// Accept a user-written `PyTaskEvaluator` and produce
/// `Arc<dyn TaskEvaluator>`. The adapter's internal `py_obj`
/// handle is cloned under the GIL.
fn extract_evaluator(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn TaskEvaluator>> {
    let py = obj.py();
    let ev = obj.extract::<PyRef<'_, PyTaskEvaluator>>().map_err(|_| {
        PyValueError::new_err("expected oharness.PyTaskEvaluator")
    })?;
    let cloned = PyTaskEvaluator {
        py_obj: ev.py_obj.clone_ref(py),
    };
    Ok(Arc::new(cloned))
}

/// Accept a user-written `PyReflector` and produce
/// `Arc<dyn Reflector>`.
fn extract_reflector(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Reflector>> {
    let py = obj.py();
    let r = obj.extract::<PyRef<'_, PyReflector>>().map_err(|_| {
        PyValueError::new_err("expected oharness.PyReflector")
    })?;
    let cloned = PyReflector {
        py_obj: r.py_obj.clone_ref(py),
        name: r.name.clone(),
    };
    Ok(Arc::new(cloned))
}

// ======================================================================
// Module-level `run_reflexion` function.
// ======================================================================

/// Multi-episode wrapper that threads `Reflection` notes from a
/// `Reflector` into subsequent episodes via the agent's
/// `ReflectionInjector`. Returns a JSON-serialised list of
/// `OwnedEpisode` records — parse with `json.loads`.
///
/// The agent must have been built with
/// `.with_reflection_injector(..)`. If not, raises
/// `RuntimeError` immediately.
///
/// Usage:
///
/// ```python
/// episodes_json = oharness.run_reflexion(
///     agent, task, evaluator, reflector, max_episodes=5,
/// )
/// ```
///
/// `evaluator` is a `PyTaskEvaluator`; `reflector` is a
/// `PyReflector`.
#[pyfunction(name = "run_reflexion")]
#[pyo3(signature = (agent, task, evaluator, reflector, max_episodes = 5))]
pub fn py_run_reflexion(
    py: Python<'_>,
    agent: PyRef<'_, PyAgent>,
    task: PyRef<'_, PyTask>,
    evaluator: Bound<'_, PyAny>,
    reflector: Bound<'_, PyAny>,
    max_episodes: u32,
) -> PyResult<String> {
    let agent_arc = agent.inner.clone();
    let task_clone = task.inner.clone();
    let evaluator_arc = extract_evaluator(&evaluator)?;
    let reflector_arc = extract_reflector(&reflector)?;

    py.allow_threads(move || {
        let episodes = shared_runtime()
            .block_on(rust_run_reflexion(
                &agent_arc,
                task_clone,
                evaluator_arc,
                reflector_arc,
                max_episodes,
            ))
            .map_err(|e| PyRuntimeError::new_err(format!("run_reflexion: {e}")))?;
        // Map each OwnedEpisode → EpisodeWire (trims the
        // trajectory handle from the outcome), then JSON-serialize
        // the whole list.
        let wire: Vec<EpisodeWire<'_>> = episodes
            .iter()
            .map(|ep| EpisodeWire {
                index: ep.index,
                task: &ep.task,
                outcome: OutcomeWire::from(&ep.outcome),
                evaluation: &ep.evaluation,
                prior_reflections: &ep.prior_reflections,
            })
            .collect();
        serde_json::to_string(&wire)
            .map_err(|e| PyRuntimeError::new_err(format!("run_reflexion: encode: {e}")))
    })
}

/// Accept any of the Llm-wrapping pyclasses exposed by this
/// crate and produce a unified `Arc<dyn Llm>`. Used by the
/// `AgentBuilder.with_llm(..)` path (which wants any Llm-ish
/// thing) and by `BudgetMiddleware(inner, ..)` (which wraps an
/// Llm with budget accounting).
///
/// Supported wrappers:
/// - `PyLlm` — user-written Llm (Python class).
/// - `PyReplayLlm` — `ReplayLlm` for trajectory replay.
/// - `PyBudgetMiddleware` — budget-wrapped Llm; composition by
///   handing a wrapped llm back.
pub(crate) fn extract_llm(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Llm>> {
    let py = obj.py();
    if let Ok(py_llm) = obj.extract::<PyRef<'_, PyLlm>>() {
        let cloned = PyLlm {
            py_obj: py_llm.py_obj.clone_ref(py),
            name: py_llm.name.clone(),
            capabilities: py_llm.capabilities.clone(),
        };
        return Ok(Arc::new(cloned) as Arc<dyn Llm>);
    }
    if let Ok(replay) = obj.extract::<PyRef<'_, PyReplayLlm>>() {
        return Ok(replay.inner.clone() as Arc<dyn Llm>);
    }
    if let Ok(budget) = obj.extract::<PyRef<'_, PyBudgetMiddleware>>() {
        return Ok(budget.inner.clone() as Arc<dyn Llm>);
    }
    if let Ok(layered) = obj.extract::<PyRef<'_, PyLayeredLlm>>() {
        return Ok(layered.inner.clone() as Arc<dyn Llm>);
    }
    Err(PyValueError::new_err(
        "expected one of oharness.PyLlm / oharness.ReplayLlm / \
         oharness.LayeredLlm / oharness.BudgetMiddleware",
    ))
}

/// Accept either a shipped `InMemorySink` or `FileSink` and
/// produce `Arc<dyn EventSink>`. Both types already hold their
/// state behind an `Arc`, so extraction is a cheap bump.
fn extract_event_sink(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn EventSink>> {
    if let Ok(mem) = obj.extract::<PyRef<'_, PyInMemorySink>>() {
        return Ok(mem.inner.clone() as Arc<dyn EventSink>);
    }
    if let Ok(fs) = obj.extract::<PyRef<'_, PyFileSink>>() {
        return Ok(fs.inner.clone() as Arc<dyn EventSink>);
    }
    Err(PyValueError::new_err(
        "with_event_sink expects an oharness.InMemorySink or oharness.FileSink",
    ))
}

/// Accept either a shipped `ReactLoop` or `ConversationLoop`
/// and take ownership, producing a `Box<dyn Loop>` for the
/// agent builder.
fn extract_loop(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Loop>> {
    if let Ok(react) = obj.extract::<PyRef<'_, PyReactLoop>>() {
        let mut guard = react
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("ReactLoop: lock: {e}")))?;
        let Some(inner) = guard.take() else {
            return Err(PyRuntimeError::new_err(
                "ReactLoop: already consumed; construct a new ReactLoop()",
            ));
        };
        drop(guard);
        return Ok(Box::new(inner));
    }
    if let Ok(conv) = obj.extract::<PyRef<'_, PyConversationLoop>>() {
        let mut guard = conv
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("ConversationLoop: lock: {e}")))?;
        let Some(inner) = guard.take() else {
            return Err(PyRuntimeError::new_err(
                "ConversationLoop: already consumed; construct a new one",
            ));
        };
        drop(guard);
        return Ok(Box::new(inner));
    }
    Err(PyValueError::new_err(
        "with_loop expects an oharness.ReactLoop or oharness.ConversationLoop",
    ))
}

/// Accept either a shipped `FsToolSet` (Python-side) or a
/// user-written `PyToolSet` wrapping a Python class. Internally
/// both produce `Arc<dyn ToolSet>`.
fn extract_toolset(tools: Bound<'_, PyAny>) -> PyResult<Arc<dyn ToolSet>> {
    if let Ok(fs) = tools.extract::<PyRef<'_, PyFsToolSet>>() {
        return Ok(fs.inner.clone() as Arc<dyn ToolSet>);
    }
    if let Ok(user) = tools.extract::<PyRef<'_, UserPyToolSet>>() {
        let py = tools.py();
        let cloned = UserPyToolSet {
            py_obj: user.py_obj.clone_ref(py),
            specs: user.specs.clone(),
            name: user.name.clone(),
        };
        return Ok(Arc::new(cloned) as Arc<dyn ToolSet>);
    }
    Err(PyValueError::new_err(
        "with_tools expects an oharness.FsToolSet or oharness.PyToolSet, \
         got something else",
    ))
}

// ======================================================================
// Shim so PyCritic impls Critic on the Rust side after being
// cloned into a CompositeCritic child. The existing adapters.rs
// impl Critic for PyCritic (and friends) handles this already —
// we just rely on it.
// ======================================================================
#[async_trait]
trait _EnsureCriticImpl: Critic {}
impl _EnsureCriticImpl for PyCritic {}

// Suppress unused-import warnings for trait-path bounds that the
// type system needs to infer but the lint can't see.
#[allow(dead_code)]
fn _type_checks(
    _: AssessmentContext<'_>,
    _: CriticVerdict,
    _: CompletionRequest,
    _: CompletionResponse,
    _: LlmCapabilities,
    _: ChunkStream,
    _: LlmError,
) {
}

// Bridge the PyBridgeError type from lib.rs — so downstream
// commits (B/C/D) can report errors via the same enum.
#[allow(dead_code)]
pub(crate) type _Bridge = PyBridgeError;
