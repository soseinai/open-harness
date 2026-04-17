//! The default ReAct-style loop: thought → action → observation.
//!
//! Emits all lifecycle / llm / tool events directly via `ScopedEmitter`. M1a
//! doesn't have the tracing-middleware layer from §9.5 yet — this loop stands in.

use crate::loop_trait::{Loop, LoopContext};
use async_trait::async_trait;
use oharness_core::event::{
    EventKind, LlmFailedPayload, LlmRequestPayload, LlmResponsePayload, MetaPayload,
    RunFinishedPayload, RunStartedPayload, ToolCallFailedPayload, ToolCallFinishedPayload,
    ToolCallStartedPayload, TurnFinishedPayload, TurnPayload,
};
use oharness_core::{
    AgentError, BudgetRequest, CompletionRequest, CompletionResponse, Content, ConversationView,
    Message, MetadataMap, ResourceUsage, RunError, RunErrorCategory, RunOutcome, StopReason, Task,
    Termination, TrajectoryHandle, TruncationLimit,
};
use oharness_memory::policy::MemoryContext;
use oharness_tools::context::ToolContext;
use oharness_tools::toolset::ToolOutcome;
use serde_json::{json, Value};
use time::OffsetDateTime;

pub struct ReactLoop {
    system_prompt: Option<String>,
}

impl Default for ReactLoop {
    fn default() -> Self {
        Self {
            system_prompt: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
        }
    }
}

impl ReactLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn without_system_prompt(mut self) -> Self {
        self.system_prompt = None;
        self
    }
}

#[async_trait]
impl Loop for ReactLoop {
    async fn run(&self, task: Task, ctx: &LoopContext) -> Result<RunOutcome, AgentError> {
        let started_at = OffsetDateTime::now_utc();
        let start_instant = std::time::Instant::now();

        // ---- meta event (always first) ----
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

        // ---- initial conversation ----
        let mut messages: Vec<Message> = Vec::new();
        let user_text = build_user_text(&task);
        messages.push(Message::user_text(user_text));

        let tools_specs = ctx.tools.specs().to_vec();
        let mut usage_totals = ResourceUsage::default();
        let mut per_model: std::collections::HashMap<oharness_core::ModelId, ResourceUsage> =
            std::collections::HashMap::new();

        let mut termination: Option<Termination> = None;
        let mut turn_index: u32 = 0;
        let mut tool_call_counter: u32 = 0;

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

            // ---- memory policy transform ----
            let mem_ctx = MemoryContext {
                events: ctx.events.clone(),
                token_budget: capabilities.max_context_tokens,
            };
            let transformed = match ctx
                .memory
                .transform(ConversationView::new(&messages), &mem_ctx)
                .await
            {
                Ok(m) => m,
                Err(e) => {
                    termination = Some(Termination::Failed {
                        error: RunError {
                            category: RunErrorCategory::Memory,
                            message: e.to_string(),
                        },
                        at_turn: turn_index,
                    });
                    break;
                }
            };

            // ---- budget pre-check ----
            let pre_budget = ctx
                .budget
                .check(BudgetRequest {
                    estimated_input_tokens: Some(
                        ConversationView::new(&transformed).token_estimate() as u64,
                    ),
                    ..Default::default()
                })
                .await;
            if let oharness_core::BudgetDecision::Deny { reason } = pre_budget {
                ctx.events.emit(
                    &turn_span,
                    EventKind::BudgetExceeded(json!({"reason": reason})),
                    Some(turn_open_seq),
                );
                termination = Some(Termination::Truncated {
                    limit: TruncationLimit::Budget(reason),
                });
                break;
            }

            // ---- build + emit llm.request ----
            let mut req = CompletionRequest::new(transformed);
            req.tools = tools_specs.clone();
            req.system = self.system_prompt.clone();

            let llm_span = format!("llm-{turn_index}");
            let llm_open_seq = ctx.events.emit(
                &llm_span,
                EventKind::LlmRequest(LlmRequestPayload {
                    request: req.clone(),
                    provider: Some(ctx.llm.name().to_string()),
                }),
                Some(turn_open_seq),
            );

            // ---- LLM call ----
            let response = match ctx.llm.complete(req).await {
                Ok(r) => r,
                Err(e) => {
                    ctx.events.emit(
                        &llm_span,
                        EventKind::LlmFailed(LlmFailedPayload {
                            reason: e.to_string(),
                        }),
                        Some(llm_open_seq),
                    );
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

            // ---- consume budget + update totals ----
            usage_totals.add_usage(&response.usage);
            per_model
                .entry(response.model.clone())
                .or_default()
                .add_usage(&response.usage);

            ctx.budget
                .consume(oharness_core::BudgetAmount {
                    tokens_input: response.usage.tokens_input,
                    tokens_output: response.usage.tokens_output,
                    cost_usd: 0.0,
                    wall_clock: std::time::Duration::ZERO,
                    steps: 1,
                })
                .await;

            ctx.events.emit(
                &llm_span,
                EventKind::LlmResponse(LlmResponsePayload {
                    response: response.clone(),
                }),
                Some(llm_open_seq),
            );

            // ---- append assistant message ----
            let assistant_msg = Message::Assistant {
                content: response.content.clone(),
                stop_reason: Some(response.stop_reason.clone()),
                meta: MetadataMap::new(),
            };
            messages.push(assistant_msg.clone());

            // ---- tool execution if any ----
            let tool_calls_in_turn = execute_tool_calls(
                &response,
                ctx,
                &mut messages,
                &turn_span,
                turn_open_seq,
                &mut tool_call_counter,
            )
            .await;

            // ---- turn.finished ----
            ctx.events.emit(
                &turn_span,
                EventKind::TurnFinished(TurnFinishedPayload {
                    turn_index,
                    stop_reason: response.stop_reason.clone(),
                    usage: response.usage.clone(),
                    tool_calls: tool_calls_in_turn,
                }),
                Some(turn_open_seq),
            );

            usage_totals.turns += 1;
            usage_totals.tool_calls += tool_calls_in_turn;

            // ---- termination decision ----
            match response.stop_reason {
                StopReason::EndTurn => {
                    termination = Some(Termination::Completed {
                        reason: oharness_core::CompletionReason::EndTurn,
                    });
                }
                StopReason::StopSequence(s) => {
                    termination = Some(Termination::Completed {
                        reason: oharness_core::CompletionReason::StopSequence(s),
                    });
                }
                StopReason::MaxTokens => {
                    termination = Some(Termination::Truncated {
                        limit: TruncationLimit::MaxTokens,
                    });
                }
                StopReason::Refusal => {
                    termination = Some(Termination::Completed {
                        reason: oharness_core::CompletionReason::EndTurn,
                    });
                }
                StopReason::ToolUse => {
                    // Continue to next turn; tool_results already appended.
                    turn_index += 1;
                    continue;
                }
                StopReason::Error(e) => {
                    termination = Some(Termination::Failed {
                        error: RunError {
                            category: RunErrorCategory::Llm,
                            message: e,
                        },
                        at_turn: turn_index,
                    });
                }
            }
        }

        let termination = termination.unwrap_or(Termination::Completed {
            reason: oharness_core::CompletionReason::EndTurn,
        });

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
            // Trajectory reference is supplied by the Agent wrapper; the loop itself
            // doesn't know where events were routed. Agent overwrites this with the
            // correct handle (file path or in-memory snapshot) before returning.
            trajectory: TrajectoryHandle::in_memory(Vec::new()),
            usage: usage_totals,
            per_model_usage: per_model,
            started_at,
            finished_at,
            agent_state: MetadataMap::new(),
        })
    }
}

async fn execute_tool_calls(
    response: &CompletionResponse,
    ctx: &LoopContext,
    messages: &mut Vec<Message>,
    turn_span: &str,
    turn_parent_seq: u64,
    tool_call_counter: &mut u32,
) -> u32 {
    let mut results: Vec<Content> = Vec::new();
    let mut count = 0u32;

    for block in &response.content {
        if let Content::ToolUse { id, name, input } = block {
            count += 1;
            let span = format!("tool-{}", *tool_call_counter);
            *tool_call_counter += 1;

            let tool_start = ctx.events.emit(
                &span,
                EventKind::ToolCallStarted(ToolCallStartedPayload {
                    tool_name: name.clone(),
                    tool_use_id: id.clone(),
                    input: input.clone(),
                }),
                Some(turn_parent_seq),
            );

            let tool_ctx = ToolContext {
                events: ctx.events.sink().clone(),
                budget: ctx.budget.clone(),
                cancellation: ctx.cancellation.clone(),
                approval: ctx.approval.clone(),
                workspace: None,
                extensions: MetadataMap::new(),
            };

            let outcome = ctx.tools.execute(name, input.clone(), &tool_ctx).await;
            match outcome {
                ToolOutcome::Success(output) => {
                    let output_repr: Value = Value::Array(
                        output
                            .content
                            .iter()
                            .map(|c| match c {
                                Content::Text { text } => json!({"type": "text", "text": text}),
                                _ => json!({"type": "other"}),
                            })
                            .collect(),
                    );
                    ctx.events.emit(
                        &span,
                        EventKind::ToolCallFinished(ToolCallFinishedPayload {
                            tool_name: name.clone(),
                            tool_use_id: id.clone(),
                            output: output_repr,
                            truncated: output.truncated,
                        }),
                        Some(tool_start),
                    );
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output,
                        is_error: false,
                    });
                }
                ToolOutcome::ExecutionError {
                    message,
                    recoverable,
                } => {
                    ctx.events.emit(
                        &span,
                        EventKind::ToolCallFailed(ToolCallFailedPayload {
                            tool_name: name.clone(),
                            tool_use_id: id.clone(),
                            reason: message.clone(),
                            recoverable,
                        }),
                        Some(tool_start),
                    );
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output: oharness_core::message::ToolOutput::text(format!(
                            "error: {message}"
                        )),
                        is_error: true,
                    });
                }
                ToolOutcome::Denied { reason } => {
                    ctx.events.emit(
                        &span,
                        EventKind::ToolCallFailed(ToolCallFailedPayload {
                            tool_name: name.clone(),
                            tool_use_id: id.clone(),
                            reason: format!("denied: {reason}"),
                            recoverable: false,
                        }),
                        Some(tool_start),
                    );
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output: oharness_core::message::ToolOutput::text(format!(
                            "denied: {reason}"
                        )),
                        is_error: true,
                    });
                }
                ToolOutcome::Cancelled => {
                    ctx.events.emit(
                        &span,
                        EventKind::ToolCallFailed(ToolCallFailedPayload {
                            tool_name: name.clone(),
                            tool_use_id: id.clone(),
                            reason: "cancelled".to_string(),
                            recoverable: false,
                        }),
                        Some(tool_start),
                    );
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output: oharness_core::message::ToolOutput::text("cancelled"),
                        is_error: true,
                    });
                }
            }
        }
    }

    if !results.is_empty() {
        messages.push(Message::User {
            content: results,
            meta: MetadataMap::new(),
        });
    }
    let _ = turn_span;
    count
}

fn build_user_text(task: &Task) -> String {
    let mut s = task.instruction.clone();
    for att in &task.attachments {
        s.push_str("\n\n");
        match att {
            oharness_core::Attachment::Text { name, content } => {
                s.push_str(&format!("# attachment: {name}\n{content}"));
            }
            oharness_core::Attachment::File { name, path } => {
                s.push_str(&format!("# attachment: {name} (file: {})", path.display()));
            }
            oharness_core::Attachment::Inline { name, mime, bytes } => {
                s.push_str(&format!(
                    "# attachment: {name} ({mime}, {} bytes)",
                    bytes.len()
                ));
            }
            oharness_core::Attachment::Url { url, .. } => {
                s.push_str(&format!("# attachment: {url}"));
            }
        }
    }
    s
}

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are an agent running inside the open-harness research framework. You have \
     access to the tools listed in the `tools` field. Think step by step, call tools \
     to gather evidence and make changes, and respond with plain text when you've \
     completed the task. Stop calling tools once the task is done.";
