//! [`ConversationLoop`] — alternates between the agent's LLM turns and a
//! [`UserSimulator`] (plan §12.2–12.4). Behind the `conversation` feature.
//!
//! Contract:
//! - Primary termination = simulator's [`UserAction::EndConversation`]
//!   (maps to `Termination::Completed { reason: EndTurn }`).
//! - Secondary termination = `ctx.max_turns` exceeded
//!   (`Termination::Truncated { limit: MaxTurns }`).
//! - Simulator errors map to `Termination::Failed { error:
//!   RunError { category: UserSimulator, .. } }` — **never** to
//!   `EndConversation`. Silent-fall-to-end would hide simulator bugs in
//!   research logs.
//!
//! For M2 part 2 this loop deliberately does NOT thread tool calls: its
//! job is to exercise the user-side of a conversation with whatever the
//! agent's LLM replies with. Tool-enabled conversational agents can
//! layer a later conversational loop on top of the React loop; that's
//! a separate piece of loop engineering.

use crate::loop_trait::{Loop, LoopContext};
use crate::user_simulator::{UserAction, UserError, UserSimulator};
use async_trait::async_trait;
use oharness_core::event::{
    EventKind, MetaPayload, RunFinishedPayload, RunStartedPayload, TurnFinishedPayload, TurnPayload,
};
use oharness_core::{
    AgentError, CompletionRequest, ConversationView, Message, MetadataMap, ResourceUsage, RunError,
    RunErrorCategory, RunOutcome, StopReason, Task, Termination, TrajectoryHandle, TruncationLimit,
};
use serde_json::json;
use time::OffsetDateTime;

pub struct ConversationLoop<U: UserSimulator> {
    simulator: U,
    system_prompt: Option<String>,
}

impl<U: UserSimulator> ConversationLoop<U> {
    pub fn new(simulator: U) -> Self {
        Self {
            simulator,
            system_prompt: None,
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

#[async_trait]
impl<U: UserSimulator> Loop for ConversationLoop<U> {
    async fn run(&self, task: Task, ctx: &LoopContext) -> Result<RunOutcome, AgentError> {
        let started_at = OffsetDateTime::now_utc();
        let start_instant = std::time::Instant::now();

        // ---- meta + run.started ----
        let capabilities = ctx.llm.capabilities();
        ctx.events.emit(
            "run-0",
            EventKind::Meta(MetaPayload {
                schema_version: oharness_core::event::SchemaVersion::CURRENT,
                harness_version: env!("CARGO_PKG_VERSION").to_string(),
                task_snapshot: task.clone(),
                llm_capabilities: capabilities.clone(),
            }),
            None,
        );
        let run_open_seq = ctx.events.emit(
            "run-0",
            EventKind::RunStarted(RunStartedPayload {
                extra: MetadataMap::new(),
            }),
            None,
        );

        // ---- initial user message (from the simulator) ----
        let initial = match self.simulator.initial_message(&task).await {
            Ok(s) => s,
            Err(e) => {
                return finish(
                    ctx,
                    run_open_seq,
                    failed(RunErrorCategory::UserSimulator, e),
                    Vec::new(),
                    ResourceUsage::default(),
                    Default::default(),
                    started_at,
                    start_instant,
                    task,
                );
            }
        };
        let mut messages: Vec<Message> = vec![Message::user_text(initial)];
        let mut usage_totals = ResourceUsage::default();
        let mut per_model: std::collections::HashMap<oharness_core::ModelId, ResourceUsage> =
            std::collections::HashMap::new();
        let mut termination: Option<Termination> = None;
        let mut turn_index: u32 = 0;

        while termination.is_none() {
            if turn_index >= ctx.max_turns {
                termination = Some(Termination::Truncated {
                    limit: TruncationLimit::MaxTurns(ctx.max_turns),
                });
                break;
            }
            if ctx.cancellation.is_cancelled() {
                termination = Some(Termination::Interrupted {
                    reason: oharness_core::InterruptionReason::Cancellation,
                });
                break;
            }

            let turn_span = format!("turn-{turn_index}");
            let turn_open_seq = ctx.events.emit(
                &turn_span,
                EventKind::TurnStarted(TurnPayload { turn_index }),
                Some(run_open_seq),
            );

            // ---- LLM turn ----
            let mut req = CompletionRequest::new(messages.clone());
            req.system = self.system_prompt.clone();
            let response = match ctx.llm.complete(req).await {
                Ok(r) => r,
                Err(e) => {
                    termination = Some(Termination::Failed {
                        error: RunError {
                            category: RunErrorCategory::Llm,
                            message: e.to_string(),
                        },
                        at_turn: turn_index,
                    });
                    break;
                }
            };
            usage_totals.add_usage(&response.usage);
            per_model
                .entry(response.model.clone())
                .or_default()
                .add_usage(&response.usage);

            let assistant_msg = Message::Assistant {
                content: response.content.clone(),
                stop_reason: Some(response.stop_reason.clone()),
                meta: MetadataMap::new(),
            };
            messages.push(assistant_msg);

            ctx.events.emit(
                &turn_span,
                EventKind::TurnFinished(TurnFinishedPayload {
                    turn_index,
                    stop_reason: response.stop_reason.clone(),
                    usage: response.usage.clone(),
                    tool_calls: 0,
                }),
                Some(turn_open_seq),
            );
            usage_totals.turns += 1;

            if matches!(response.stop_reason, StopReason::Error(_)) {
                termination = Some(Termination::Failed {
                    error: RunError {
                        category: RunErrorCategory::Llm,
                        message: format!("{:?}", response.stop_reason),
                    },
                    at_turn: turn_index,
                });
                break;
            }

            // ---- User simulator response ----
            let view = ConversationView::new(&messages);
            let action = self.simulator.respond(view, &task).await;
            match action {
                Ok(UserAction::Say(text)) => {
                    ctx.events.emit(
                        &turn_span,
                        EventKind::UserSimulatedMessage(json!({
                            "simulator": self.simulator.name(),
                            "text": text,
                        })),
                        Some(turn_open_seq),
                    );
                    messages.push(Message::user_text(text));
                }
                Ok(UserAction::EndConversation) => {
                    ctx.events.emit(
                        &turn_span,
                        EventKind::UserSimulatedEnded(json!({
                            "simulator": self.simulator.name(),
                            "at_turn": turn_index,
                        })),
                        Some(turn_open_seq),
                    );
                    termination = Some(Termination::Completed {
                        reason: oharness_core::CompletionReason::EndTurn,
                    });
                    break;
                }
                Err(e) => {
                    termination = Some(failed(RunErrorCategory::UserSimulator, e));
                    break;
                }
            }

            turn_index += 1;
        }

        finish(
            ctx,
            run_open_seq,
            termination.unwrap_or(Termination::Completed {
                reason: oharness_core::CompletionReason::EndTurn,
            }),
            messages,
            usage_totals,
            per_model,
            started_at,
            start_instant,
            task,
        )
    }
}

fn failed(category: RunErrorCategory, err: UserError) -> Termination {
    Termination::Failed {
        error: RunError {
            category,
            message: err.to_string(),
        },
        at_turn: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(
    ctx: &LoopContext,
    run_open_seq: u64,
    termination: Termination,
    messages: Vec<Message>,
    mut usage_totals: ResourceUsage,
    per_model: std::collections::HashMap<oharness_core::ModelId, ResourceUsage>,
    started_at: OffsetDateTime,
    start_instant: std::time::Instant,
    task: Task,
) -> Result<RunOutcome, AgentError> {
    let finished_at = OffsetDateTime::now_utc();
    usage_totals.wall_clock = start_instant.elapsed();
    ctx.events.emit(
        "run-0",
        EventKind::RunFinished(RunFinishedPayload {
            termination: format!("{termination:?}"),
            turns: usage_totals.turns,
            tool_calls: usage_totals.tool_calls,
            extra: MetadataMap::new(),
        }),
        Some(run_open_seq),
    );
    Ok(RunOutcome {
        run_id: ctx.events.run_id(),
        task_id: task.id.clone(),
        termination,
        final_messages: messages,
        trajectory: TrajectoryHandle::in_memory(Vec::new()),
        usage: usage_totals,
        per_model_usage: per_model,
        started_at,
        finished_at,
        agent_state: MetadataMap::new(),
    })
}
