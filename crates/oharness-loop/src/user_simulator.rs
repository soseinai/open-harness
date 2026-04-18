//! [`UserSimulator`] trait and shipped simulator implementations
//! (plan §12.3, §12.4).
//!
//! A simulator stands in for a human user during a [`ConversationLoop`]
//! run: it produces an initial message from the task, then responds to
//! each assistant turn with either [`UserAction::Say`] (send a follow-up)
//! or [`UserAction::EndConversation`] (terminate the loop).
//!
//! Simulators receive a [`oharness_core::ConversationView`] — not raw
//! messages — so they can call `.user_visible()` to strip internal
//! reasoning / tool calls / tool results and respond only to the
//! human-facing thread. Simulators that want stricter or looser views
//! can compose their own.

use async_trait::async_trait;
use oharness_core::{CompletionRequest, ConversationView, Message, Task};
use oharness_llm::Llm;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[async_trait]
pub trait UserSimulator: Send + Sync {
    fn name(&self) -> &str;

    /// Produce the first user message from the task. Called once at the
    /// start of a conversation loop.
    async fn initial_message(&self, task: &Task) -> Result<String, UserError>;

    /// Respond to the current conversation. Return
    /// [`UserAction::Say`] with the next user message, or
    /// [`UserAction::EndConversation`] to terminate the loop.
    async fn respond(
        &self,
        conversation: ConversationView<'_>,
        task: &Task,
    ) -> Result<UserAction, UserError>;
}

/// What the simulator wants to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserAction {
    Say(String),
    EndConversation,
}

/// Simulator-side errors. Any error here is promoted to
/// `Termination::Failed { reason: "user_simulator_error" }` by the
/// [`ConversationLoop`] — simulators cannot silently fall back to
/// [`UserAction::EndConversation`] since that would hide bugs.
#[derive(Debug, thiserror::Error)]
pub enum UserError {
    #[error("user simulator: {0}")]
    Other(String),
    #[error("user simulator llm: {0}")]
    Llm(#[from] oharness_llm::LlmError),
}

// ======================================================================
// ScriptedUserSimulator (plan §12.4)
// ======================================================================

/// A simulator that replays a fixed sequence of user utterances. The
/// first entry is returned from [`UserSimulator::initial_message`]; each
/// subsequent call to [`UserSimulator::respond`] returns the next entry,
/// [`UserAction::Say`]-wrapped, until the script is exhausted — at which
/// point the simulator returns [`UserAction::EndConversation`].
///
/// Useful for tests, reproducible evaluation runs, and
/// conversation-loop smoke paths where no LLM should be involved on the
/// user side.
pub struct ScriptedUserSimulator {
    script: Vec<String>,
    cursor: AtomicUsize,
    name: String,
}

impl ScriptedUserSimulator {
    pub fn new(script: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let script = script.into_iter().map(Into::into).collect::<Vec<_>>();
        Self {
            script,
            cursor: AtomicUsize::new(0),
            name: "scripted-user".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl UserSimulator for ScriptedUserSimulator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn initial_message(&self, task: &Task) -> Result<String, UserError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.script
            .get(idx)
            .cloned()
            .ok_or_else(|| UserError::Other(format!("empty script (task={})", task.instruction)))
    }

    async fn respond(
        &self,
        _conversation: ConversationView<'_>,
        _task: &Task,
    ) -> Result<UserAction, UserError> {
        let idx = self.cursor.fetch_add(1, Ordering::SeqCst);
        match self.script.get(idx) {
            Some(msg) => Ok(UserAction::Say(msg.clone())),
            None => Ok(UserAction::EndConversation),
        }
    }
}

// ======================================================================
// LlmUserSimulator (plan §12.4)
// ======================================================================

/// A simulator that drives a user LLM with a persona + template. Each
/// `respond` call builds a [`CompletionRequest`] whose system prompt is
/// the rendered `prompt_template` (with `{persona}` and `{task}`
/// substituted) and whose user message is the serialized conversation
/// so far.
///
/// The simulator looks for a terminal sentinel — by default `<end>` —
/// in the LLM's response to decide whether to return
/// [`UserAction::EndConversation`]. The sentinel is matched
/// case-insensitively and stripped from the surrounding text.
pub struct LlmUserSimulator {
    llm: Arc<dyn Llm>,
    persona: String,
    prompt_template: String,
    end_sentinel: String,
    name: String,
}

impl LlmUserSimulator {
    /// A reasonable default template — users with opinionated
    /// simulators should supply their own.
    pub fn default_template() -> &'static str {
        "You are role-playing a user with this persona:\n\n{persona}\n\n\
         The user's underlying task is:\n\n{task}\n\n\
         Respond to the assistant's most recent turn as the user would. \
         Keep responses short. When the task is fully resolved, include \
         the literal token `<end>` anywhere in your reply to end the \
         conversation. Do not prefix your reply with `USER:` or any role \
         label."
    }

    pub fn new(
        llm: Arc<dyn Llm>,
        persona: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        Self {
            llm,
            persona: persona.into(),
            prompt_template: prompt_template.into(),
            end_sentinel: "<end>".to_string(),
            name: "llm-user".to_string(),
        }
    }

    pub fn with_end_sentinel(mut self, sentinel: impl Into<String>) -> Self {
        self.end_sentinel = sentinel.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn render_system(&self, task: &Task) -> String {
        self.prompt_template
            .replace("{persona}", &self.persona)
            .replace("{task}", &task.instruction)
    }
}

#[async_trait]
impl UserSimulator for LlmUserSimulator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn initial_message(&self, task: &Task) -> Result<String, UserError> {
        // Cheap: the task's instruction is already a user-shaped prompt.
        // Callers that want a more elaborate kickoff can subclass /
        // wrap — for the default, we just replay the instruction.
        Ok(task.instruction.clone())
    }

    async fn respond(
        &self,
        conversation: ConversationView<'_>,
        task: &Task,
    ) -> Result<UserAction, UserError> {
        let transcript = render_transcript(conversation);
        let mut req = CompletionRequest::new(vec![Message::user_text(transcript)]);
        req.system = Some(self.render_system(task));
        let res = self.llm.complete(req).await?;
        let text = res
            .content
            .iter()
            .filter_map(|c| match c {
                oharness_core::Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let text_lower = text.to_ascii_lowercase();
        let sentinel_lower = self.end_sentinel.to_ascii_lowercase();
        if text_lower.contains(&sentinel_lower) {
            // Sentinel present — end the conversation. Any surrounding
            // text is dropped; users who want a trailing message should
            // emit it on a previous turn rather than glue it to `<end>`.
            let _ = strip_case_insensitive(&text, &self.end_sentinel);
            Ok(UserAction::EndConversation)
        } else {
            Ok(UserAction::Say(text))
        }
    }
}

fn render_transcript(view: ConversationView<'_>) -> String {
    let mut out = String::new();
    for m in view.user_visible() {
        match m {
            Message::System { content, .. } => {
                out.push_str("SYSTEM: ");
                out.push_str(&content);
                out.push('\n');
            }
            Message::User { content, .. } => {
                out.push_str("USER: ");
                out.push_str(&flatten_text(&content));
                out.push('\n');
            }
            Message::Assistant { content, .. } => {
                out.push_str("ASSISTANT: ");
                out.push_str(&flatten_text(&content));
                out.push('\n');
            }
        }
    }
    out
}

fn flatten_text(content: &[oharness_core::Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            oharness_core::Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_case_insensitive(haystack: &str, needle: &str) -> String {
    // Case-insensitive single-pass strip. Simple and correct for small
    // inputs; we don't expect the sentinel to appear many times.
    let hl = haystack.to_ascii_lowercase();
    let nl = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if hl[i..].starts_with(&nl) {
            i += needle.len();
        } else {
            let ch = haystack[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oharness_core::{
        CompletionResponse, Content, LlmCapabilities, Message, ModelId, StopReason, Task, Usage,
    };
    use oharness_llm::{ChunkStream, LlmError};
    use std::sync::Mutex;

    // ---------- ScriptedUserSimulator ----------

    #[tokio::test]
    async fn scripted_returns_initial_then_sequenced_responses() {
        let sim = ScriptedUserSimulator::new(["hi", "more please", "thanks"]);
        let task = Task::new("chat");
        let first = sim.initial_message(&task).await.unwrap();
        assert_eq!(first, "hi");

        let empty: Vec<Message> = Vec::new();
        let v = ConversationView::new(&empty);
        match sim.respond(v, &task).await.unwrap() {
            UserAction::Say(s) => assert_eq!(s, "more please"),
            other => panic!("expected Say, got {other:?}"),
        }
        let v = ConversationView::new(&empty);
        match sim.respond(v, &task).await.unwrap() {
            UserAction::Say(s) => assert_eq!(s, "thanks"),
            other => panic!("expected Say, got {other:?}"),
        }
        // Exhausted -> EndConversation.
        let v = ConversationView::new(&empty);
        assert_eq!(
            sim.respond(v, &task).await.unwrap(),
            UserAction::EndConversation
        );
    }

    #[tokio::test]
    async fn scripted_empty_script_errors_on_initial_message() {
        let sim: ScriptedUserSimulator = ScriptedUserSimulator::new(std::iter::empty::<String>());
        match sim.initial_message(&Task::new("t")).await {
            Err(UserError::Other(msg)) => assert!(msg.contains("empty script")),
            other => panic!("expected Err(UserError::Other), got {other:?}"),
        }
    }

    // ---------- LlmUserSimulator ----------

    struct OneShot(Mutex<Option<CompletionResponse>>);
    #[async_trait]
    impl Llm for OneShot {
        fn name(&self) -> &str {
            "one-shot-user"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            self.0
                .lock()
                .unwrap()
                .take()
                .ok_or(LlmError::Unsupported("one-shot"))
        }
        async fn stream(&self, _req: CompletionRequest) -> Result<ChunkStream, LlmError> {
            Err(LlmError::Unsupported("stream"))
        }
    }

    fn text_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            id: "u".into(),
            model: ModelId::new("m"),
            content: vec![Content::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    #[tokio::test]
    async fn llm_user_initial_message_replays_task_instruction() {
        let llm: Arc<dyn Llm> = Arc::new(OneShot(Mutex::new(None)));
        let sim = LlmUserSimulator::new(llm, "friendly user", LlmUserSimulator::default_template());
        let task = Task::new("help me debug this bug");
        assert_eq!(
            sim.initial_message(&task).await.unwrap(),
            "help me debug this bug"
        );
    }

    #[tokio::test]
    async fn llm_user_respond_emits_say_on_plain_response() {
        let llm: Arc<dyn Llm> =
            Arc::new(OneShot(Mutex::new(Some(text_response("that's helpful")))));
        let sim = LlmUserSimulator::new(llm, "user", LlmUserSimulator::default_template());
        let empty: Vec<Message> = Vec::new();
        match sim
            .respond(ConversationView::new(&empty), &Task::new("t"))
            .await
            .unwrap()
        {
            UserAction::Say(s) => assert_eq!(s, "that's helpful"),
            other => panic!("expected Say, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn llm_user_respond_emits_end_on_sentinel() {
        let llm: Arc<dyn Llm> = Arc::new(OneShot(Mutex::new(Some(text_response("done <end>")))));
        let sim = LlmUserSimulator::new(llm, "user", LlmUserSimulator::default_template());
        let empty: Vec<Message> = Vec::new();
        assert_eq!(
            sim.respond(ConversationView::new(&empty), &Task::new("t"))
                .await
                .unwrap(),
            UserAction::EndConversation
        );
    }

    #[tokio::test]
    async fn llm_user_respond_is_case_insensitive_on_sentinel() {
        let llm: Arc<dyn Llm> = Arc::new(OneShot(Mutex::new(Some(text_response("<END>")))));
        let sim = LlmUserSimulator::new(llm, "user", LlmUserSimulator::default_template());
        let empty: Vec<Message> = Vec::new();
        assert_eq!(
            sim.respond(ConversationView::new(&empty), &Task::new("t"))
                .await
                .unwrap(),
            UserAction::EndConversation
        );
    }

    #[tokio::test]
    async fn llm_user_respond_errors_on_llm_error() {
        // One-shot with no response → first respond errors.
        let llm: Arc<dyn Llm> = Arc::new(OneShot(Mutex::new(None)));
        let sim = LlmUserSimulator::new(llm, "user", LlmUserSimulator::default_template());
        let empty: Vec<Message> = Vec::new();
        match sim
            .respond(ConversationView::new(&empty), &Task::new("t"))
            .await
        {
            Err(UserError::Llm(_)) => {}
            other => panic!("expected Err(UserError::Llm), got {other:?}"),
        }
    }
}
