//! The sentinel error returned when a budget is exhausted.
//!
//! `BudgetMiddleware` wraps this in `LlmError::Provider` so downstream
//! handlers can `downcast_ref::<BudgetExceeded>()` to distinguish budget
//! exhaustion from any other provider error (plan §10.1).

#[derive(Debug, thiserror::Error)]
#[error("budget exceeded: {reason}")]
pub struct BudgetExceeded {
    pub reason: String,
}

impl BudgetExceeded {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}
