//! [`ReflectionInjector`] — the middleware bridge that turns accumulated
//! [`Reflection`]s into prompt material the next LLM call will see
//! (plan §11.5).
//!
//! Implements [`oharness_llm::RequestLayer`]. Prepends reflections either
//! as a system-prompt suffix (default — benign, doesn't disturb few-shot
//! exemplars) or as a prefix on the first user message (useful when you
//! don't have a system prompt to latch onto).
//!
//! State (the current reflection list + episode counter) lives behind a
//! `Mutex` so `run_reflexion` can call [`set_reflections`] between
//! episodes without rebuilding the Llm middleware stack.
//!
//! Emits `reflection.injected { episode_index, reflection_count }` when
//! an [`oharness_core::ScopedEmitter`] is attached via
//! [`ReflectionInjector::with_emitter`].

use oharness_core::event::EventKind;
use oharness_core::{CompletionRequest, Content, Message, Reflection, ScopedEmitter};
use oharness_llm::RequestLayer;
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// Where the injector places the rendered reflection block inside the
/// outgoing `CompletionRequest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReflectionPlacement {
    /// Append to `req.system`. If `system` is `None`, the injector
    /// populates it with the reflection block. Default.
    #[default]
    SystemSuffix,
    /// Prefix onto the first text content block of the first user
    /// message. Useful when a provider / wrapper has no system-prompt
    /// channel.
    FirstUserPrefix,
}

pub struct ReflectionInjector {
    reflections: Mutex<Vec<Reflection>>,
    placement: ReflectionPlacement,
    emitter: Option<ScopedEmitter>,
    episode_index: AtomicU32,
}

impl ReflectionInjector {
    pub fn new() -> Self {
        Self {
            reflections: Mutex::new(Vec::new()),
            placement: ReflectionPlacement::default(),
            emitter: None,
            episode_index: AtomicU32::new(0),
        }
    }

    pub fn with_placement(mut self, placement: ReflectionPlacement) -> Self {
        self.placement = placement;
        self
    }

    /// Attach an emitter so each injection logs
    /// `reflection.injected { episode_index, reflection_count }`. Optional
    /// — without an emitter the injector is silent on the trajectory.
    pub fn with_emitter(mut self, emitter: ScopedEmitter) -> Self {
        self.emitter = Some(emitter);
        self
    }

    /// Replace the current reflection list. `run_reflexion` calls this
    /// between episodes; it also resets the "warned once" surface for
    /// future logging policies.
    pub fn set_reflections(&self, reflections: Vec<Reflection>) {
        let mut guard = self.reflections.lock().expect("reflection injector mutex");
        *guard = reflections;
    }

    /// Increment the episode counter used in emitted events. Call at the
    /// start of each reflexion iteration.
    pub fn bump_episode(&self) {
        self.episode_index.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reflection_count(&self) -> usize {
        self.reflections
            .lock()
            .expect("reflection injector mutex")
            .len()
    }
}

impl Default for ReflectionInjector {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestLayer for ReflectionInjector {
    fn on_request(&self, req: &mut CompletionRequest) {
        // Snapshot the reflections under lock so we don't hold the guard
        // during mutation of the request.
        let snapshot: Vec<Reflection> = {
            let guard = self.reflections.lock().expect("reflection injector mutex");
            if guard.is_empty() {
                return;
            }
            guard.clone()
        };

        let block = render_reflections(&snapshot);
        match self.placement {
            ReflectionPlacement::SystemSuffix => match &mut req.system {
                Some(sys) => {
                    sys.push_str("\n\n");
                    sys.push_str(&block);
                }
                slot @ None => {
                    *slot = Some(block);
                }
            },
            ReflectionPlacement::FirstUserPrefix => {
                for msg in req.messages.iter_mut() {
                    if let Message::User { content, .. } = msg {
                        // Mutate the first text block in place; if there
                        // isn't one, prepend a fresh text block to the
                        // user message.
                        if let Some(first_text) = content.iter_mut().find_map(|c| match c {
                            Content::Text { text } => Some(text),
                            _ => None,
                        }) {
                            let original = std::mem::take(first_text);
                            *first_text = format!("{block}\n\n{original}");
                        } else {
                            content.insert(0, Content::text(block.clone()));
                        }
                        break;
                    }
                }
            }
        }

        if let Some(em) = &self.emitter {
            let episode_index = self.episode_index.load(Ordering::Relaxed);
            em.emit(
                "reflection",
                EventKind::ReflectionInjected(json!({
                    "episode_index": episode_index,
                    "reflection_count": snapshot.len(),
                    "placement": format!("{:?}", self.placement),
                })),
                None,
            );
        }
    }
}

fn render_reflections(reflections: &[Reflection]) -> String {
    // Header plus numbered bullets. The header signals to models that
    // this is out-of-band guidance rather than part of the user's task.
    let mut s = String::from("# Reflections from prior attempts\n");
    for (i, r) in reflections.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, r.text));
    }
    s.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_core::{CompletionRequest, Message};

    fn sample_req() -> CompletionRequest {
        CompletionRequest::new(vec![Message::user_text("do the thing")])
    }

    fn sample_reflection(text: &str) -> Reflection {
        Reflection::new(text)
    }

    #[test]
    fn empty_reflections_are_a_noop() {
        let inj = ReflectionInjector::new();
        let mut req = sample_req();
        let before = req.system.clone();
        inj.on_request(&mut req);
        assert_eq!(req.system, before);
    }

    #[test]
    fn system_suffix_populates_absent_system() {
        let inj = ReflectionInjector::new();
        inj.set_reflections(vec![sample_reflection("check imports")]);
        let mut req = sample_req();
        inj.on_request(&mut req);
        let sys = req.system.expect("system set");
        assert!(sys.contains("Reflections"));
        assert!(sys.contains("check imports"));
    }

    #[test]
    fn system_suffix_appends_to_existing_system() {
        let inj = ReflectionInjector::new();
        inj.set_reflections(vec![sample_reflection("no eval()")]);
        let mut req = sample_req();
        req.system = Some("Be concise.".to_string());
        inj.on_request(&mut req);
        let sys = req.system.expect("system set");
        assert!(sys.starts_with("Be concise."));
        assert!(sys.contains("no eval()"));
    }

    #[test]
    fn first_user_prefix_prepends_to_first_text_block() {
        let inj = ReflectionInjector::new().with_placement(ReflectionPlacement::FirstUserPrefix);
        inj.set_reflections(vec![sample_reflection("mind edge cases")]);
        let mut req = sample_req();
        inj.on_request(&mut req);
        match &req.messages[0] {
            Message::User { content, .. } => match &content[0] {
                Content::Text { text } => {
                    assert!(text.contains("mind edge cases"));
                    assert!(text.trim_end().ends_with("do the thing"));
                }
                other => panic!("expected Text, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }

    #[test]
    fn set_reflections_replaces_prior_state() {
        let inj = ReflectionInjector::new();
        inj.set_reflections(vec![sample_reflection("one"), sample_reflection("two")]);
        assert_eq!(inj.reflection_count(), 2);
        inj.set_reflections(vec![sample_reflection("three")]);
        assert_eq!(inj.reflection_count(), 1);
    }

    #[test]
    fn bump_episode_increments_counter() {
        let inj = ReflectionInjector::new();
        inj.bump_episode();
        inj.bump_episode();
        assert_eq!(inj.episode_index.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn numbered_rendering_matches_reflection_order() {
        let inj = ReflectionInjector::new();
        inj.set_reflections(vec![
            sample_reflection("alpha"),
            sample_reflection("beta"),
            sample_reflection("gamma"),
        ]);
        let mut req = sample_req();
        inj.on_request(&mut req);
        let sys = req.system.expect("system");
        let alpha_pos = sys.find("alpha").unwrap();
        let beta_pos = sys.find("beta").unwrap();
        let gamma_pos = sys.find("gamma").unwrap();
        assert!(alpha_pos < beta_pos);
        assert!(beta_pos < gamma_pos);
    }
}
