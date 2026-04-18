//! Python bindings for open-harness (plan §14).
//!
//! Exposes wrapper types that let Python users plug their own trait
//! implementations into Rust-side agent runs:
//!
//! - [`PyLlm`]            — `complete(req_json: str) -> str`
//! - [`PyCritic`]         — `assess(ctx_json: str) -> str`
//! - [`PyTaskEvaluator`]  — `evaluate(task_json: str, outcome_json: str) -> str`
//! - [`PyReflector`]      — `reflect(episode_json: str) -> Optional[str]`
//! - [`PyUserSimulator`]  — `initial_message(task_json: str) -> str`
//!   plus `respond(conversation_json: str, task_json: str) -> str`
//! - [`PyMemoryPolicy`]   — `transform(conversation_json: str, ctx_json: str) -> str`
//! - [`PyToolSet`]        — `execute(name: str, input_json: str, ctx_json: str) -> str`
//!   (specs fixed at construction time)
//! - [`PyRequestLayer`]   — `on_request(req_json: str) -> str` (sync, in-place mutate)
//! - [`PyResponseLayer`]  — `on_response(res_json: str) -> str` (sync, in-place mutate)
//!
//! All wire types cross the Rust↔Python boundary as JSON-encoded
//! strings. The Python side implements a duck-typed class with the
//! named method(s); the Rust side serializes arguments with serde,
//! calls the Python method under the GIL (for the async adapters,
//! wrapped in `tokio::task::spawn_blocking` so the async runtime
//! stays responsive; for the sync `Request/ResponseLayer` adapters,
//! called directly under the GIL — layers are expected to be cheap),
//! and deserializes the returned string.
//!
//! ## v1 scope vs. later (plan §14.2)
//!
//! v1 (this crate, as it ships now): `Llm::complete`, `Critic::assess`,
//! `TaskEvaluator::evaluate`, `Reflector::reflect`, `UserSimulator`,
//! `MemoryPolicy::transform`, `ToolSet::execute`, `RequestLayer`,
//! `ResponseLayer`. Sync Python side (async Python is v1.1).
//!
//! Deferred: `Llm::stream`, `ChunkObserver` / `ChunkTransformer`.
//! Streaming from Python is an open research problem (GIL + async);
//! per-chunk observers are discouraged by per-chunk GIL cost.
//!
//! ## Build
//!
//! ```bash
//! cd crates/oharness-py
//! maturin develop --release
//! ```
//!
//! Then in Python:
//!
//! ```python
//! import oharness
//!
//! class MyLlm:
//!     def complete(self, req_json: str) -> str:
//!         # req_json is a CompletionRequest as JSON
//!         # return a CompletionResponse as JSON
//!         return '{"id":"r","model":"m","content":[],"stop_reason":{"kind":"end_turn"},"usage":{"tokens_input":0,"tokens_output":0}}'
//!
//! llm = oharness.PyLlm(MyLlm())
//! ```
//!
//! The `oharness::PyLlm` handle implements `oharness_llm::Llm` on the
//! Rust side and can be dropped into any Rust call that takes
//! `Arc<dyn Llm>`.

use async_trait::async_trait;
use oharness_core::{
    CompletionRequest, CompletionResponse, ConversationView, Episode, EvaluationResult,
    LlmCapabilities, Message, Reflection, RunOutcome, Task, TaskEvaluator, ToolOutput, ToolSpec,
};
use oharness_critic::{AssessmentContext, Critic, CriticVerdict, Reflector};
use oharness_llm::{
    ChunkStream, Llm, LlmError, RequestLayer, ResponseLayer, ResponseLayerStreamMode,
};
use oharness_loop::{UserAction, UserError, UserSimulator};
use oharness_memory::{MemoryContext, MemoryError, MemoryPolicy};
use oharness_tools::{ToolContext, ToolOutcome, ToolSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Serialize;

// ======================================================================
// PyLlm — wraps a Python `Llm`-like object.
// ======================================================================

/// Rust handle to a Python `Llm` implementation. Calls the Python
/// object's `complete(req_json: str) -> str` method; JSON-encodes /
/// decodes on the Rust side. `stream()` always returns
/// `LlmError::Unsupported("stream")` — async streaming from Python is
/// a later milestone.
#[pyclass]
pub struct PyLlm {
    py_obj: PyObject,
    name: String,
    capabilities: LlmCapabilities,
}

#[pymethods]
impl PyLlm {
    #[new]
    #[pyo3(signature = (py_obj, name = "python".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self {
            py_obj,
            name,
            capabilities: LlmCapabilities::default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("PyLlm(name={:?})", self.name)
    }
}

#[async_trait]
impl Llm for PyLlm {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let req_json = serde_json::to_string(&req).map_err(map_json_err)?;
        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let response_json = tokio::task::spawn_blocking(move || -> Result<String, LlmError> {
            Python::with_gil(|py| {
                let result = py_obj
                    .call_method1(py, "complete", (req_json,))
                    .map_err(map_py_err)?;
                result
                    .extract::<String>(py)
                    .map_err(|e| LlmError::MalformedResponse(format!("PyLlm.complete: {e}")))
            })
        })
        .await
        .map_err(|e| LlmError::Provider(Box::new(PyBridgeError::JoinError(e.to_string()))))??;

        serde_json::from_str(&response_json).map_err(|e| {
            LlmError::MalformedResponse(format!(
                "PyLlm.complete returned non-CompletionResponse JSON: {e}"
            ))
        })
    }

    async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
        Err(LlmError::Unsupported("stream (Python v1.0 is complete-only)"))
    }
}

// ======================================================================
// PyCritic — wraps a Python `Critic`-like object.
// ======================================================================

/// Rust handle to a Python `Critic` implementation. Calls the Python
/// object's `assess(ctx_json: str) -> str` method. The returned string
/// must be a JSON-serialized
/// [`oharness_critic::CriticVerdict`] — e.g. `{"verdict": "accept"}` or
/// `{"verdict": "reject", "reason": "..."}`. The simpler verdict
/// spellings are documented below.
#[pyclass]
pub struct PyCritic {
    py_obj: PyObject,
    name: String,
}

#[pymethods]
impl PyCritic {
    #[new]
    #[pyo3(signature = (py_obj, name = "python-critic".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self { py_obj, name }
    }

    fn __repr__(&self) -> String {
        format!("PyCritic(name={:?})", self.name)
    }
}

/// Python-side `assess` return shape. Documented here (not just in a
/// Python README) because the contract lives on the Rust deserialize.
///
/// ```json
/// {"verdict": "accept"}
/// {"verdict": "accept_with_note", "note": "..."}
/// {"verdict": "reject", "reason": "..."}
/// {"verdict": "abort",  "reason": "..."}
/// ```
///
/// `revise` is intentionally NOT supported from Python in v1 — the
/// replacement `AssistantTurn` has more shape (span ids, tool-call
/// extraction) than is worth serde-ing across the boundary for this
/// slice. Python critics that want to rewrite turns should emit
/// `reject` with a descriptive reason and let the loop's retry path
/// produce a fresh generation.
#[derive(serde::Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
enum WireVerdict {
    Accept,
    AcceptWithNote { note: String },
    Reject { reason: String },
    Abort { reason: String },
}

#[async_trait]
impl Critic for PyCritic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn assess(&self, ctx: &AssessmentContext<'_>) -> CriticVerdict {
        // Build a JSON view of the AssessmentContext that the Python
        // side can pattern-match against without us handing out
        // lifetime-borrowed types.
        let wire = AssessmentView {
            task: ctx.task.clone(),
            latest_turn: serde_json::to_value(&ctx.latest_turn.message).unwrap_or_default(),
            turn_index: ctx.latest_turn.turn_index,
        };
        let ctx_json = match serde_json::to_string(&wire) {
            Ok(s) => s,
            Err(e) => return fail_open(&format!("PyCritic: encode ctx: {e}")),
        };

        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let verdict_json_res =
            tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
                Python::with_gil(|py| {
                    let out = py_obj
                        .call_method1(py, "assess", (ctx_json,))
                        .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                    out.extract::<String>(py)
                        .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
                })
            })
            .await;

        let verdict_json = match verdict_json_res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return fail_open(&format!("PyCritic: {e}")),
            Err(e) => return fail_open(&format!("PyCritic: join error: {e}")),
        };

        let wire: WireVerdict = match serde_json::from_str(&verdict_json) {
            Ok(v) => v,
            Err(e) => {
                return fail_open(&format!(
                    "PyCritic: decode verdict: {e} (raw: {verdict_json})"
                ))
            }
        };

        match wire {
            WireVerdict::Accept => CriticVerdict::Accept,
            WireVerdict::AcceptWithNote { note } => CriticVerdict::AcceptWithNote(note),
            WireVerdict::Reject { reason } => CriticVerdict::Reject { reason },
            WireVerdict::Abort { reason } => CriticVerdict::Abort { reason },
        }
    }
}

/// Critics fail open per plan §11.1: errors on the critic path are
/// turned into `AcceptWithNote` with a descriptive note, so
/// `critic.failed` emission downstream can still detect the
/// divergence without stalling the loop.
fn fail_open(msg: &str) -> CriticVerdict {
    CriticVerdict::AcceptWithNote(msg.to_string())
}

#[derive(Serialize)]
struct AssessmentView {
    task: Task,
    latest_turn: serde_json::Value,
    turn_index: u32,
}

// ======================================================================
// PyTaskEvaluator — wraps a Python `TaskEvaluator`-like object.
// ======================================================================

/// Rust handle to a Python `TaskEvaluator`. Calls
/// `evaluate(task_json: str, outcome_json: str) -> str`; the response
/// is a JSON-serialized [`EvaluationResult`] (fields
/// `score: f64`, `passed: bool`, optional `details: object`). Errors
/// turn into a failing `EvaluationResult` with the error message in
/// `details["error"]` — the runner treats it as "this task didn't
/// score", consistent with `oharness_eval`'s handling of load /
/// factory errors.
#[pyclass]
pub struct PyTaskEvaluator {
    py_obj: PyObject,
}

#[pymethods]
impl PyTaskEvaluator {
    #[new]
    fn new(py_obj: PyObject) -> Self {
        Self { py_obj }
    }

    fn __repr__(&self) -> &'static str {
        "PyTaskEvaluator(...)"
    }
}

#[async_trait]
impl TaskEvaluator for PyTaskEvaluator {
    async fn evaluate(&self, task: &Task, outcome: &RunOutcome) -> EvaluationResult {
        let task_json = match serde_json::to_string(task) {
            Ok(s) => s,
            Err(e) => return eval_error(&format!("encode task: {e}")),
        };
        let outcome_json = match serde_json::to_string(outcome) {
            Ok(s) => s,
            Err(e) => return eval_error(&format!("encode outcome: {e}")),
        };

        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let result_json_res =
            tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
                Python::with_gil(|py| {
                    let out = py_obj
                        .call_method1(py, "evaluate", (task_json, outcome_json))
                        .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                    out.extract::<String>(py)
                        .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
                })
            })
            .await;

        let result_json = match result_json_res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return eval_error(&format!("{e}")),
            Err(e) => return eval_error(&format!("join error: {e}")),
        };

        match serde_json::from_str::<EvaluationResult>(&result_json) {
            Ok(r) => r,
            Err(e) => eval_error(&format!("decode EvaluationResult: {e} (raw: {result_json})")),
        }
    }
}

fn eval_error(msg: &str) -> EvaluationResult {
    use oharness_core::MetadataMap;
    let mut details = MetadataMap::new();
    details.insert("error".into(), serde_json::Value::String(msg.to_string()));
    EvaluationResult {
        score: 0.0,
        passed: false,
        details,
    }
}

// ======================================================================
// PyReflector — wraps a Python `Reflector`-like object.
// ======================================================================

/// Rust handle to a Python `Reflector` implementation. Calls the Python
/// object's `reflect(episode_json: str) -> Optional[str]` method. The
/// episode is serialized as a compact view (task / outcome summary /
/// evaluation / prior reflections) — notably *without* the trajectory
/// handle, since in-memory `TrajectoryHandle`s refuse to serialize and
/// file-backed ones would need Python to re-read the JSONL anyway.
///
/// Python should return one of:
/// - `None` — no reflection emitted this episode.
/// - a JSON string `"null"` — same as `None`.
/// - a JSON string `{"text": "...", "metadata": {...}}` — produces a
///   [`Reflection`] with the given text + optional metadata.
///
/// Any error on the Python side (exception, malformed JSON, bad
/// shape) logs via `eprintln!` and returns `None` — consistent with
/// the reflector contract that a bad reflector shouldn't break the
/// reflexion sweep.
#[pyclass]
pub struct PyReflector {
    py_obj: PyObject,
    name: String,
}

#[pymethods]
impl PyReflector {
    #[new]
    #[pyo3(signature = (py_obj, name = "python-reflector".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self { py_obj, name }
    }

    fn __repr__(&self) -> String {
        format!("PyReflector(name={:?})", self.name)
    }
}

/// Wire shape for the episode passed into Python. Flattens the
/// borrowed [`Episode`] fields into an owned, serde-friendly view. The
/// `outcome` field is a trimmed [`RunOutcome`] mirror that omits the
/// `trajectory` handle — in-memory handles refuse to serialize (they'd
/// blow up the call), and reflectors written in Python don't have a
/// good way to consume a file path anyway.
#[derive(Serialize)]
struct EpisodeWire<'a> {
    index: u32,
    task: &'a Task,
    outcome: OutcomeWire<'a>,
    evaluation: &'a EvaluationResult,
    prior_reflections: &'a [Reflection],
}

#[derive(Serialize)]
struct OutcomeWire<'a> {
    run_id: oharness_core::RunId,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<&'a String>,
    termination: &'a oharness_core::Termination,
    final_messages: &'a [Message],
    usage: &'a oharness_core::ResourceUsage,
}

impl<'a> From<&'a RunOutcome> for OutcomeWire<'a> {
    fn from(o: &'a RunOutcome) -> Self {
        Self {
            run_id: o.run_id,
            task_id: o.task_id.as_ref(),
            termination: &o.termination,
            final_messages: &o.final_messages,
            usage: &o.usage,
        }
    }
}

/// Optional Python-side return payload — `{"text", "metadata"}`. The
/// full `Reflection` (including `created_at`) is reconstructed on the
/// Rust side so Python authors don't have to emit valid RFC-3339.
#[derive(serde::Deserialize)]
struct WireReflection {
    text: String,
    #[serde(default)]
    metadata: oharness_core::MetadataMap,
}

#[async_trait]
impl Reflector for PyReflector {
    fn name(&self) -> &str {
        &self.name
    }

    async fn reflect(&self, episode: &Episode<'_>) -> Option<Reflection> {
        let wire = EpisodeWire {
            index: episode.index,
            task: episode.task,
            outcome: episode.outcome.into(),
            evaluation: episode.evaluation,
            prior_reflections: episode.prior_reflections,
        };
        let episode_json = match serde_json::to_string(&wire) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("PyReflector({}): encode episode: {e}", self.name);
                return None;
            }
        };

        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let result_res =
            tokio::task::spawn_blocking(move || -> Result<Option<String>, PyBridgeError> {
                Python::with_gil(|py| {
                    let out = py_obj
                        .call_method1(py, "reflect", (episode_json,))
                        .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                    // Accept either a str or None. `extract::<Option<String>>`
                    // happily handles both.
                    out.extract::<Option<String>>(py)
                        .map_err(|e| PyBridgeError::PythonCall(format!("extract: {e}")))
                })
            })
            .await;

        let maybe_json = match result_res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                eprintln!("PyReflector({}): {e}", self.name);
                return None;
            }
            Err(e) => {
                eprintln!("PyReflector({}): join error: {e}", self.name);
                return None;
            }
        };

        let json_str = match maybe_json {
            Some(s) => s,
            None => return None,
        };
        // A literal `"null"` return also means None — symmetric with
        // the Python-None case, so authors can be sloppy about which
        // they emit.
        if json_str.trim() == "null" {
            return None;
        }

        match serde_json::from_str::<WireReflection>(&json_str) {
            Ok(w) => {
                let mut r = Reflection::new(w.text);
                r.metadata = w.metadata;
                Some(r)
            }
            Err(e) => {
                eprintln!(
                    "PyReflector({}): decode reflection: {e} (raw: {json_str})",
                    self.name
                );
                None
            }
        }
    }
}

// ======================================================================
// PyUserSimulator — wraps a Python `UserSimulator`-like object.
// ======================================================================

/// Rust handle to a Python `UserSimulator` implementation. Calls two
/// Python methods:
///
/// - `initial_message(task_json: str) -> str` — returns the first user
///   message as a bare string.
/// - `respond(conversation_json: str, task_json: str) -> str` — returns
///   a JSON-encoded next-action.
///
/// Action wire shapes:
///
/// ```json
/// {"action": "say", "message": "..."}
/// {"action": "end_conversation"}
/// ```
///
/// Any Python exception or JSON shape error is promoted to a
/// `UserError::Other` — the `ConversationLoop` then terminates with
/// `Termination::Failed { reason: "user_simulator_error" }`. Simulators
/// intentionally do NOT fail-open (unlike critics): hiding simulator
/// bugs behind `EndConversation` would break eval reproducibility.
#[pyclass]
pub struct PyUserSimulator {
    py_obj: PyObject,
    name: String,
}

#[pymethods]
impl PyUserSimulator {
    #[new]
    #[pyo3(signature = (py_obj, name = "python-user".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self { py_obj, name }
    }

    fn __repr__(&self) -> String {
        format!("PyUserSimulator(name={:?})", self.name)
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum WireUserAction {
    Say { message: String },
    EndConversation,
}

#[async_trait]
impl UserSimulator for PyUserSimulator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn initial_message(&self, task: &Task) -> Result<String, UserError> {
        let task_json =
            serde_json::to_string(task).map_err(|e| UserError::Other(format!("encode task: {e}")))?;
        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let res = tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
            Python::with_gil(|py| {
                let out = py_obj
                    .call_method1(py, "initial_message", (task_json,))
                    .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                out.extract::<String>(py)
                    .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
            })
        })
        .await;

        match res {
            Ok(Ok(s)) => Ok(s),
            Ok(Err(e)) => Err(UserError::Other(e.to_string())),
            Err(e) => Err(UserError::Other(format!("join: {e}"))),
        }
    }

    async fn respond(
        &self,
        conversation: ConversationView<'_>,
        task: &Task,
    ) -> Result<UserAction, UserError> {
        let conv_json = serde_json::to_string(conversation.messages())
            .map_err(|e| UserError::Other(format!("encode conversation: {e}")))?;
        let task_json =
            serde_json::to_string(task).map_err(|e| UserError::Other(format!("encode task: {e}")))?;

        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let res = tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
            Python::with_gil(|py| {
                let out = py_obj
                    .call_method1(py, "respond", (conv_json, task_json))
                    .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                out.extract::<String>(py)
                    .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
            })
        })
        .await;

        let action_json = match res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(UserError::Other(e.to_string())),
            Err(e) => return Err(UserError::Other(format!("join: {e}"))),
        };

        let wire: WireUserAction = serde_json::from_str(&action_json).map_err(|e| {
            UserError::Other(format!(
                "decode UserAction: {e} (raw: {action_json})"
            ))
        })?;
        Ok(match wire {
            WireUserAction::Say { message } => UserAction::Say(message),
            WireUserAction::EndConversation => UserAction::EndConversation,
        })
    }
}

// ======================================================================
// PyMemoryPolicy — wraps a Python `MemoryPolicy`-like object.
// ======================================================================

/// Rust handle to a Python `MemoryPolicy` implementation. Calls
/// `transform(conversation_json: str, ctx_json: str) -> str`; the
/// returned string is a JSON array of [`Message`] — the transformed
/// conversation the LLM will see on the next turn.
///
/// The `ctx` wire carries only `token_budget` — the `ScopedEmitter`
/// doesn't cross the boundary, so **Python memory policies cannot
/// emit `memory.evicted` / `memory.summarized` / `memory.retrieved`
/// events**. This is a documented v1 limitation; future work may grow
/// a return-side "events" channel so Python policies can surface
/// telemetry.
///
/// Any error on the Python side is promoted to
/// `MemoryError::Configuration`, which the loop treats as fatal for
/// the turn — unlike critics, a broken memory policy must not
/// silently pass the raw conversation through.
#[pyclass]
pub struct PyMemoryPolicy {
    py_obj: PyObject,
    name: String,
}

#[pymethods]
impl PyMemoryPolicy {
    #[new]
    #[pyo3(signature = (py_obj, name = "python-memory".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self { py_obj, name }
    }

    fn __repr__(&self) -> String {
        format!("PyMemoryPolicy(name={:?})", self.name)
    }
}

#[derive(Serialize)]
struct MemoryContextWire {
    token_budget: u32,
}

#[async_trait]
impl MemoryPolicy for PyMemoryPolicy {
    async fn transform(
        &self,
        conversation: ConversationView<'_>,
        ctx: &MemoryContext,
    ) -> Result<Vec<Message>, MemoryError> {
        let conv_json = serde_json::to_string(conversation.messages()).map_err(|e| {
            MemoryError::Configuration(format!("encode conversation: {e}"))
        })?;
        let ctx_json = serde_json::to_string(&MemoryContextWire {
            token_budget: ctx.token_budget,
        })
        .map_err(|e| MemoryError::Configuration(format!("encode ctx: {e}")))?;

        let name = self.name.clone();
        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let res = tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
            Python::with_gil(|py| {
                let out = py_obj
                    .call_method1(py, "transform", (conv_json, ctx_json))
                    .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                out.extract::<String>(py)
                    .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
            })
        })
        .await;

        let messages_json = match res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                return Err(MemoryError::Configuration(format!(
                    "PyMemoryPolicy({name}): {e}"
                )))
            }
            Err(e) => {
                return Err(MemoryError::Configuration(format!(
                    "PyMemoryPolicy({name}): join: {e}"
                )))
            }
        };

        serde_json::from_str::<Vec<Message>>(&messages_json).map_err(|e| {
            MemoryError::Configuration(format!(
                "PyMemoryPolicy({name}): decode Vec<Message>: {e} (raw: {messages_json})"
            ))
        })
    }
}

// ======================================================================
// PyToolSet — wraps a Python `ToolSet`-like object.
// ======================================================================

/// Rust handle to a Python `ToolSet` implementation. Calls
/// `execute(name: str, input_json: str, ctx_json: str) -> str` for
/// each tool invocation; the returned string is a JSON-encoded
/// [`ToolOutcome`].
///
/// **Specs are fixed at construction time** — they're passed in as a
/// JSON array and stored as owned `Vec<ToolSpec>`. This avoids
/// round-tripping through Python on every turn just to ask "what
/// tools do you have?" (the loop reads `specs()` once per request
/// when assembling the `CompletionRequest`). If you need dynamic
/// specs, rebuild the `PyToolSet` between runs.
///
/// ## Wire shapes
///
/// Input to Python's `execute`:
/// - `name`: the tool name (one of the specs' names).
/// - `input_json`: JSON-encoded tool input (whatever shape the
///   `input_schema` describes — tools validate their own inputs).
/// - `ctx_json`: a trimmed [`ToolContext`] view — only
///   `workspace_path` (optional string) and `extensions` (metadata
///   map) cross the boundary. `EventSink`, `BudgetHandle`,
///   `Cancellation`, `ApprovalChannel` are Rust-runtime types that
///   can't usefully be exposed to Python in v1.
///
/// Output from Python's `execute`:
///
/// ```json
/// {"outcome": "success", "output": {"content": [{"type":"text","text":"..."}], "truncated": false}}
/// {"outcome": "execution_error", "message": "...", "recoverable": false}
/// {"outcome": "denied", "reason": "..."}
/// {"outcome": "cancelled"}
/// ```
///
/// As a convenience, bare-string success is also accepted:
/// `{"outcome": "success_text", "text": "..."}` is equivalent to
/// the full `{"outcome": "success", "output": {...}}` form with a
/// single text block.
///
/// ## Error handling
///
/// Any Python exception, bad JSON, or bad shape is promoted to
/// [`ToolOutcome::ExecutionError`] with `recoverable: false` and the
/// bridge error as the message. This lets the loop see the failure
/// (via `tool.call.failed`) without crashing the run.
#[pyclass]
pub struct PyToolSet {
    py_obj: PyObject,
    specs: Vec<ToolSpec>,
    name: String,
}

#[pymethods]
impl PyToolSet {
    /// Build a `PyToolSet` from a Python object and a JSON array of
    /// [`ToolSpec`]s. A typical construction:
    ///
    /// ```python
    /// specs = json.dumps([{
    ///     "name": "echo",
    ///     "description": "Echo the input.",
    ///     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
    /// }])
    /// toolset = oharness.PyToolSet(MyToolSet(), specs, name="python-tools")
    /// ```
    ///
    /// Raises `ValueError` if `specs_json` doesn't deserialize as
    /// `Vec<ToolSpec>`.
    #[new]
    #[pyo3(signature = (py_obj, specs_json, name = "python-toolset".to_string()))]
    fn new(py_obj: PyObject, specs_json: &str, name: String) -> PyResult<Self> {
        let specs: Vec<ToolSpec> = serde_json::from_str(specs_json).map_err(|e| {
            PyValueError::new_err(format!(
                "PyToolSet: specs_json must be a JSON array of ToolSpec: {e}"
            ))
        })?;
        Ok(Self {
            py_obj,
            specs,
            name,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyToolSet(name={:?}, tools={})",
            self.name,
            self.specs.len()
        )
    }

    /// Names of the tools this set exposes. Useful from Python for
    /// quick inspection.
    fn tool_names(&self) -> Vec<String> {
        self.specs.iter().map(|s| s.name.clone()).collect()
    }
}

/// Trimmed `ToolContext` view that crosses the boundary. Rust-runtime
/// types (`EventSink` / `BudgetHandle` / `Cancellation` /
/// `ApprovalChannel`) don't serialize; dropping them is consistent
/// with the `PyMemoryPolicy` approach for `ScopedEmitter`. A
/// Python-side "observability" channel may land in a future revision
/// — for now, Python tools are essentially stateless.
#[derive(Serialize)]
struct ToolContextWire<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_path: Option<String>,
    extensions: &'a oharness_core::MetadataMap,
}

/// Python-side `execute` return shape. Snake-case externally-tagged
/// enum — matches the documented wire in the `PyToolSet` doc.
#[derive(serde::Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum WireToolOutcome {
    Success { output: ToolOutput },
    /// Convenience variant for the very common "single text block"
    /// success case. Python users can write
    /// `{"outcome":"success_text","text":"..."}` instead of
    /// assembling the full `ToolOutput` JSON.
    SuccessText { text: String },
    ExecutionError {
        message: String,
        #[serde(default)]
        recoverable: bool,
    },
    Denied { reason: String },
    Cancelled,
}

impl From<WireToolOutcome> for ToolOutcome {
    fn from(w: WireToolOutcome) -> Self {
        match w {
            WireToolOutcome::Success { output } => ToolOutcome::Success(output),
            WireToolOutcome::SuccessText { text } => {
                ToolOutcome::Success(ToolOutput::text(text))
            }
            WireToolOutcome::ExecutionError {
                message,
                recoverable,
            } => ToolOutcome::ExecutionError {
                message,
                recoverable,
            },
            WireToolOutcome::Denied { reason } => ToolOutcome::Denied { reason },
            WireToolOutcome::Cancelled => ToolOutcome::Cancelled,
        }
    }
}

#[async_trait]
impl ToolSet for PyToolSet {
    fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    async fn execute(
        &self,
        name: &str,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> ToolOutcome {
        let input_json = match serde_json::to_string(&input) {
            Ok(s) => s,
            Err(e) => {
                return ToolOutcome::ExecutionError {
                    message: format!("PyToolSet: encode input: {e}"),
                    recoverable: false,
                };
            }
        };

        let ctx_wire = ToolContextWire {
            workspace_path: ctx
                .workspace_path()
                .map(|p| p.to_string_lossy().into_owned()),
            extensions: &ctx.extensions,
        };
        let ctx_json = match serde_json::to_string(&ctx_wire) {
            Ok(s) => s,
            Err(e) => {
                return ToolOutcome::ExecutionError {
                    message: format!("PyToolSet: encode ctx: {e}"),
                    recoverable: false,
                };
            }
        };

        let py_obj = self.py_obj.clone_ref_unbound_gil();
        let name_owned = name.to_string();
        let set_name = self.name.clone();
        let res = tokio::task::spawn_blocking(move || -> Result<String, PyBridgeError> {
            Python::with_gil(|py| {
                let out = py_obj
                    .call_method1(py, "execute", (name_owned, input_json, ctx_json))
                    .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
                out.extract::<String>(py)
                    .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
            })
        })
        .await;

        let outcome_json = match res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                return ToolOutcome::ExecutionError {
                    message: format!("PyToolSet({set_name}): {e}"),
                    recoverable: false,
                };
            }
            Err(e) => {
                return ToolOutcome::ExecutionError {
                    message: format!("PyToolSet({set_name}): join: {e}"),
                    recoverable: false,
                };
            }
        };

        match serde_json::from_str::<WireToolOutcome>(&outcome_json) {
            Ok(w) => w.into(),
            Err(e) => ToolOutcome::ExecutionError {
                message: format!(
                    "PyToolSet({set_name}): decode outcome: {e} (raw: {outcome_json})"
                ),
                recoverable: false,
            },
        }
    }
}

// ======================================================================
// PyRequestLayer — wraps a Python `RequestLayer`-like object.
// ======================================================================

/// Rust handle to a Python `RequestLayer` implementation. Calls the
/// Python object's `on_request(req_json: str) -> str` method;
/// deserializes the returned string and replaces the outgoing
/// `CompletionRequest` in place.
///
/// ## Sync-in-async blocking
///
/// Unlike the six async adapters, `RequestLayer` is a sync trait:
/// `fn on_request(&self, req: &mut CompletionRequest)`. The layer
/// still runs inside an async `complete()` / `stream()` call, so
/// the Python call happens **synchronously under the GIL** from the
/// async task's poll. This is fine for cheap layers — redaction,
/// header injection, metadata merging, request-id stamping. For
/// heavy Python work, wrap your `PyLlm` *outside* the layer
/// composition (the layer should stay fast).
///
/// ## Fail-open on errors
///
/// Any exception on the Python side, bad JSON, or bad shape logs
/// via `eprintln!` and leaves the request unchanged. A broken layer
/// should not crash the run — the unmodified request still reaches
/// the underlying LLM.
#[pyclass]
pub struct PyRequestLayer {
    py_obj: PyObject,
    name: String,
}

#[pymethods]
impl PyRequestLayer {
    #[new]
    #[pyo3(signature = (py_obj, name = "python-request-layer".to_string()))]
    fn new(py_obj: PyObject, name: String) -> Self {
        Self { py_obj, name }
    }

    fn __repr__(&self) -> String {
        format!("PyRequestLayer(name={:?})", self.name)
    }
}

impl RequestLayer for PyRequestLayer {
    fn on_request(&self, req: &mut CompletionRequest) {
        let req_json = match serde_json::to_string(req) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "PyRequestLayer({}): encode request: {e}; leaving unchanged",
                    self.name
                );
                return;
            }
        };

        let result = Python::with_gil(|py| -> Result<String, PyBridgeError> {
            let out = self
                .py_obj
                .call_method1(py, "on_request", (req_json,))
                .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
            out.extract::<String>(py)
                .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
        });

        let new_json = match result {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "PyRequestLayer({}): {e}; leaving request unchanged",
                    self.name
                );
                return;
            }
        };

        match serde_json::from_str::<CompletionRequest>(&new_json) {
            Ok(r) => *req = r,
            Err(e) => {
                eprintln!(
                    "PyRequestLayer({}): decode request: {e} (raw: {new_json}); \
                     leaving request unchanged",
                    self.name
                );
            }
        }
    }
}

// ======================================================================
// PyResponseLayer — wraps a Python `ResponseLayer`-like object.
// ======================================================================

/// Rust handle to a Python `ResponseLayer` implementation. Calls the
/// Python object's `on_response(res_json: str) -> str` method;
/// deserializes the returned string and replaces the incoming
/// `CompletionResponse` in place.
///
/// ## `stream_mode`
///
/// `ResponseLayer` has a `stream_mode()` hook that decides what
/// happens when the layer is wrapped around `stream()`. Python
/// users choose via a string argument at construction:
/// - `"warn_and_skip"` (default) — log once per wrapper, pass
///   chunks through unchanged.
/// - `"error"` — `stream()` returns `LlmError::Unsupported`.
/// - `"silent_skip"` — pass chunks through without logging.
///
/// ## Sync-in-async blocking
///
/// Same caveat as [`PyRequestLayer`] — called synchronously under
/// the GIL from inside the async `complete()` task. Keep layers
/// cheap.
///
/// ## Fail-open on errors
///
/// Any exception on the Python side, bad JSON, or bad shape logs
/// via `eprintln!` and leaves the response unchanged.
#[pyclass]
pub struct PyResponseLayer {
    py_obj: PyObject,
    name: String,
    stream_mode: ResponseLayerStreamMode,
}

#[pymethods]
impl PyResponseLayer {
    /// Construct a Python response layer. `stream_mode` picks the
    /// behaviour when wrapped around `stream()`:
    /// `"warn_and_skip"` (default), `"error"`, or `"silent_skip"`.
    /// Raises `ValueError` on any other string.
    #[new]
    #[pyo3(signature = (
        py_obj,
        name = "python-response-layer".to_string(),
        stream_mode = "warn_and_skip".to_string(),
    ))]
    fn new(py_obj: PyObject, name: String, stream_mode: String) -> PyResult<Self> {
        let stream_mode = match stream_mode.as_str() {
            "warn_and_skip" => ResponseLayerStreamMode::WarnAndSkip,
            "error" => ResponseLayerStreamMode::Error,
            "silent_skip" => ResponseLayerStreamMode::SilentSkip,
            other => {
                return Err(PyValueError::new_err(format!(
                    "PyResponseLayer: stream_mode must be one of \
                     \"warn_and_skip\" / \"error\" / \"silent_skip\", got {other:?}"
                )));
            }
        };
        Ok(Self {
            py_obj,
            name,
            stream_mode,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyResponseLayer(name={:?}, stream_mode={:?})",
            self.name, self.stream_mode
        )
    }
}

impl ResponseLayer for PyResponseLayer {
    fn on_response(&self, res: &mut CompletionResponse) {
        let res_json = match serde_json::to_string(res) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "PyResponseLayer({}): encode response: {e}; leaving unchanged",
                    self.name
                );
                return;
            }
        };

        let result = Python::with_gil(|py| -> Result<String, PyBridgeError> {
            let out = self
                .py_obj
                .call_method1(py, "on_response", (res_json,))
                .map_err(|e| PyBridgeError::PythonCall(format!("{e}")))?;
            out.extract::<String>(py)
                .map_err(|e| PyBridgeError::PythonCall(format!("extract str: {e}")))
        });

        let new_json = match result {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "PyResponseLayer({}): {e}; leaving response unchanged",
                    self.name
                );
                return;
            }
        };

        match serde_json::from_str::<CompletionResponse>(&new_json) {
            Ok(r) => *res = r,
            Err(e) => {
                eprintln!(
                    "PyResponseLayer({}): decode response: {e} (raw: {new_json}); \
                     leaving response unchanged",
                    self.name
                );
            }
        }
    }

    fn stream_mode(&self) -> ResponseLayerStreamMode {
        self.stream_mode
    }

    // The trait's `name()` returns `&'static str`; our adapter holds
    // an owned String so we can't return it directly. Fall back to
    // the default (type name) — the user-supplied name is still
    // useful as an attribute and in `__repr__`, just not in the
    // `tracing::warn!` that fires once on `WarnAndSkip`.
    // A future API-break could widen the trait method to `&str`.
}

// ======================================================================
// Error helpers shared across the adapters
// ======================================================================

#[derive(Debug, thiserror::Error)]
enum PyBridgeError {
    #[error("python call: {0}")]
    PythonCall(String),
    #[error("tokio join error: {0}")]
    JoinError(String),
}

fn map_py_err(e: PyErr) -> LlmError {
    LlmError::Provider(Box::new(PyBridgeError::PythonCall(format!("{e}"))))
}

fn map_json_err(e: serde_json::Error) -> LlmError {
    LlmError::MalformedResponse(format!("serde_json: {e}"))
}

/// Small shim: `PyObject::clone_ref` requires a `Python` token, which
/// we don't always want to take when constructing background work.
/// This helper clones the reference without re-acquiring the GIL on
/// call-sites that just need an owning handle. Under the hood pyo3
/// exposes `Py<T>::clone_ref(py)` for this; the helper wraps it so the
/// three adapters don't each re-acquire the GIL.
trait PyObjectExt {
    fn clone_ref_unbound_gil(&self) -> PyObject;
}

impl PyObjectExt for PyObject {
    fn clone_ref_unbound_gil(&self) -> PyObject {
        Python::with_gil(|py| self.clone_ref(py))
    }
}

// ======================================================================
// #[pymodule]
// ======================================================================

#[pymodule]
fn oharness(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLlm>()?;
    m.add_class::<PyCritic>()?;
    m.add_class::<PyTaskEvaluator>()?;
    m.add_class::<PyReflector>()?;
    m.add_class::<PyUserSimulator>()?;
    m.add_class::<PyMemoryPolicy>()?;
    m.add_class::<PyToolSet>()?;
    m.add_class::<PyRequestLayer>()?;
    m.add_class::<PyResponseLayer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
