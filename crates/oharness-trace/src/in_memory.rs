//! In-memory event sink — keeps every emitted event in a `Vec`. Primarily for tests.

use oharness_core::{Event, EventSink};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default, Clone)]
pub struct InMemorySink {
    events: Arc<Mutex<Vec<Event>>>,
}

impl InMemorySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of events so far. Returns a cloned Vec so callers can iterate
    /// without holding the lock.
    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    /// Number of events captured.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.lock().unwrap().is_empty()
    }
}

impl EventSink for InMemorySink {
    fn emit(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }

    fn try_emit(&self, event: Event) -> Result<(), Event> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }
}
