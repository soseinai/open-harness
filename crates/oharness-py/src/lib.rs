//! Python bindings for open-harness (plan §14).
//!
//! Exposes three wrapper types that let Python users plug their own
//! `Llm` / `Critic` / `TaskEvaluator` implementations into Rust-side
//! agent runs:
//!
//! - [`PyLlm`]   — `complete(req_json: str) -> str`
//! - [`PyCritic` ] — `assess(ctx_json: str) -> str`
//! - [`PyTaskEvaluator`] — `evaluate(task_json: str, outcome_json: str) -> str`
//!
//! All wire types cross the Rust↔Python boundary as JSON-encoded
//! strings. The Python side implements a duck-typed class with the
//! named method; the Rust side serializes arguments with serde, calls
//! the Python method under the GIL (wrapped in
//! `tokio::task::spawn_blocking` so the async runtime stays
//! responsive), and deserializes the returned string.
//!
//! ## v1 scope vs. later (plan §14.2)
//!
//! v1 (this crate, as it ships now): `Llm::complete`, `Critic::assess`,
//! `TaskEvaluator::evaluate`. Sync Python side (async Python is v1.1).
//!
//! Deferred: `Llm::stream`, `ToolSet`, `MemoryPolicy`, `Reflector`,
//! `UserSimulator`, `RequestLayer` / `ResponseLayer`, `ChunkObserver` /
//! `ChunkTransformer`. Each follows the same adapter pattern when it
//! lands — the scaffold here is the template.
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
    CompletionRequest, CompletionResponse, EvaluationResult, LlmCapabilities, RunOutcome, Task,
    TaskEvaluator,
};
use oharness_critic::{AssessmentContext, Critic, CriticVerdict};
use oharness_llm::{ChunkStream, Llm, LlmError};
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
// Error helpers shared across the three adapters
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
