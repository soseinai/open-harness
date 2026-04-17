//! Fan-out sink — dispatches each event to multiple inner sinks.

use oharness_core::{Event, EventSink};
use std::sync::Arc;

pub struct FanOutSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl FanOutSink {
    pub fn new(sinks: Vec<Arc<dyn EventSink>>) -> Self {
        Self { sinks }
    }

    pub fn add(&mut self, sink: Arc<dyn EventSink>) {
        self.sinks.push(sink);
    }
}

impl EventSink for FanOutSink {
    fn emit(&self, event: Event) {
        for sink in &self.sinks {
            sink.emit(event.clone());
        }
    }

    fn try_emit(&self, event: Event) -> Result<(), Event> {
        // Any inner backpressure returns the event (best-effort semantics).
        for sink in &self.sinks {
            if sink.try_emit(event.clone()).is_err() {
                return Err(event);
            }
        }
        Ok(())
    }
}
