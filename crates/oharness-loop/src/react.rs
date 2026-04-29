//! The default ReAct-style loop: thought → action → observation.
//!
//! Emits only lifecycle events (`meta`, `run.*`, `turn.*`, `budget.exceeded`).
//! `llm.*` and `tool.*` events are produced by `RequestTracer` and
//! `ToolTracer` in `oharness-trace`, wired in by `Agent::run`
//! (see docs/remaining-work.md §2.4 — M1b-δ refactor).

use crate::loop_trait::{Loop, LoopContext};
use async_trait::async_trait;
use oharness_core::event::{
    EventKind, MetaPayload, RunFinishedPayload, RunStartedPayload, TurnFinishedPayload,
    TurnPayload, TurnRevisedPayload,
};
use oharness_core::{
    AgentError, AssistantTurn, BudgetRequest, CompletionRequest, CompletionResponse, Content,
    ConversationView, Message, MetadataMap, ResourceUsage, RunError, RunErrorCategory, RunOutcome,
    StopReason, Task, Termination, TrajectoryHandle, TrajectoryView, TruncationLimit,
};
use oharness_critic::{AssessmentContext, Critic, CriticTrigger, CriticVerdict};
use oharness_llm::complete_from_stream;
use oharness_memory::policy::MemoryContext;
use oharness_tools::context::ToolContext;
use oharness_tools::toolset::ToolOutcome;
use oharness_trace::TOOL_USE_ID_KEY;
use serde_json::json;
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

            // ---- build + issue LLM call (llm.* events are emitted by
            //      RequestTracer wrapping ctx.llm) ----
            let mut req = CompletionRequest::new(transformed);
            req.tools = tools_specs.clone();
            req.system = self.system_prompt.clone();

            let response =
                match complete_with_optional_streaming(ctx, req, capabilities.streaming).await {
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

            // ---- append assistant message ----
            let mut assistant_msg = Message::Assistant {
                content: response.content.clone(),
                stop_reason: Some(response.stop_reason.clone()),
                meta: MetadataMap::new(),
            };
            let mut effective_stop = response.stop_reason.clone();
            let mut effective_usage = response.usage.clone();
            messages.push(assistant_msg.clone());

            // ---- critic invocation (plan §11) ----
            // Only AfterAssistant is wired for M2 part 2; other triggers
            // (AfterToolResult, AfterEveryNTurns, OnDemand) are deferred.
            if matches!(ctx.critic_trigger, CriticTrigger::AfterAssistant) {
                if let Some(critic) = &ctx.critics {
                    match run_critic_after_assistant(
                        critic.as_ref(),
                        &task,
                        &messages,
                        turn_index,
                        &assistant_msg,
                        &effective_usage,
                        &effective_stop,
                        &turn_span,
                        turn_open_seq,
                        ctx,
                    )
                    .await
                    {
                        CriticOutcome::Continue {
                            effective_message,
                            effective_usage: new_usage,
                            effective_stop: new_stop,
                        } => {
                            // Swap the in-history assistant message with the
                            // (possibly revised) final version.
                            if let Some(last) = messages.last_mut() {
                                *last = effective_message.clone();
                            }
                            assistant_msg = effective_message;
                            effective_usage = new_usage;
                            effective_stop = new_stop;
                        }
                        CriticOutcome::Terminate { error } => {
                            termination = Some(Termination::Failed {
                                error,
                                at_turn: turn_index,
                            });
                            break;
                        }
                    }
                }
            }
            let _ = assistant_msg; // silence unused-assignment warning when no revision

            // ---- tool execution if any ----
            // Use the *effective* response — a revise verdict may have
            // swapped the tool_use blocks the agent actually intended.
            let effective_response = CompletionResponse {
                id: response.id.clone(),
                model: response.model.clone(),
                content: extract_content_from_message(messages.last()),
                stop_reason: effective_stop.clone(),
                usage: effective_usage.clone(),
            };
            let tool_calls_in_turn =
                execute_tool_calls(&effective_response, ctx, &mut messages).await;

            // ---- turn.finished ----
            ctx.events.emit(
                &turn_span,
                EventKind::TurnFinished(TurnFinishedPayload {
                    turn_index,
                    stop_reason: effective_stop.clone(),
                    usage: effective_usage.clone(),
                    tool_calls: tool_calls_in_turn,
                }),
                Some(turn_open_seq),
            );

            usage_totals.turns += 1;
            usage_totals.tool_calls += tool_calls_in_turn;

            // ---- termination decision ----
            match effective_stop {
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

async fn complete_with_optional_streaming(
    ctx: &LoopContext,
    req: CompletionRequest,
    streaming: bool,
) -> Result<CompletionResponse, oharness_llm::LlmError> {
    if streaming {
        let stream = ctx.llm.stream(req).await?;
        complete_from_stream(stream).await
    } else {
        ctx.llm.complete(req).await
    }
}

/// Result of a single critic cycle for one assistant turn. Produced by
/// [`run_critic_after_assistant`], consumed by the loop body.
///
/// `Continue` always carries the effective assistant state — the same
/// values that went in if the critic accepted first-pass, or the final
/// revision if the critic revised-then-accepted. This lets the loop
/// consume one canonical `(message, usage, stop)` triple regardless of
/// whether a revision happened.
enum CriticOutcome {
    Continue {
        effective_message: Message,
        effective_usage: oharness_core::Usage,
        effective_stop: StopReason,
    },
    Terminate {
        error: RunError,
    },
}

/// Invoke the critic on the freshly-appended assistant turn and resolve
/// the verdict. Re-invokes the critic on each revision up to
/// [`LoopContext::revision_depth_cap`] — once exceeded, the most recent
/// `Revise` verdict is converted to `Reject` per plan §11.1.
#[allow(clippy::too_many_arguments)]
async fn run_critic_after_assistant(
    critic: &oharness_critic::CompositeCritic,
    task: &Task,
    messages: &[Message],
    turn_index: u32,
    initial_message: &Message,
    initial_usage: &oharness_core::Usage,
    initial_stop: &StopReason,
    turn_span: &str,
    turn_open_seq: u64,
    ctx: &LoopContext,
) -> CriticOutcome {
    let mut current_message = initial_message.clone();
    let mut current_usage = initial_usage.clone();
    let mut current_stop = initial_stop.clone();

    for depth in 0..=ctx.revision_depth_cap {
        let original_seq = turn_open_seq;
        let trajectory_tail: Vec<oharness_core::Event> = Vec::new(); // loop does not re-read events
        let turn = AssistantTurn::new(
            turn_index,
            turn_span,
            current_message.clone(),
            current_usage.clone(),
            current_stop.clone(),
        );
        let assess_ctx = AssessmentContext::new(
            task,
            ConversationView::new(messages),
            &turn,
            TrajectoryView::new(&trajectory_tail),
        );

        let verdict = critic.assess(&assess_ctx).await;
        match verdict {
            CriticVerdict::Accept => {
                ctx.events.emit(
                    turn_span,
                    EventKind::CriticAssessed(json!({
                        "critic": critic.name(),
                        "verdict": "accept",
                        "revision_depth": depth,
                    })),
                    Some(turn_open_seq),
                );
                return CriticOutcome::Continue {
                    effective_message: current_message,
                    effective_usage: current_usage,
                    effective_stop: current_stop,
                };
            }
            CriticVerdict::AcceptWithNote(note) => {
                ctx.events.emit(
                    turn_span,
                    EventKind::CriticAssessed(json!({
                        "critic": critic.name(),
                        "verdict": "accept_with_note",
                        "note": note,
                        "revision_depth": depth,
                    })),
                    Some(turn_open_seq),
                );
                return CriticOutcome::Continue {
                    effective_message: current_message,
                    effective_usage: current_usage,
                    effective_stop: current_stop,
                };
            }
            CriticVerdict::Reject { reason } => {
                ctx.events.emit(
                    turn_span,
                    EventKind::CriticRejected(json!({
                        "critic": critic.name(),
                        "reason": reason,
                        "revision_depth": depth,
                    })),
                    Some(turn_open_seq),
                );
                return CriticOutcome::Terminate {
                    error: RunError {
                        category: RunErrorCategory::Critic,
                        message: format!("critic `{}` rejected turn: {reason}", critic.name()),
                    },
                };
            }
            CriticVerdict::Abort { reason } => {
                ctx.events.emit(
                    turn_span,
                    EventKind::CriticRejected(json!({
                        "critic": critic.name(),
                        "reason": reason,
                        "abort": true,
                        "revision_depth": depth,
                    })),
                    Some(turn_open_seq),
                );
                return CriticOutcome::Terminate {
                    error: RunError {
                        category: RunErrorCategory::Critic,
                        message: format!("critic `{}` aborted run: {reason}", critic.name()),
                    },
                };
            }
            CriticVerdict::Revise {
                replacement,
                reason,
            } => {
                if depth >= ctx.revision_depth_cap {
                    // Cap exceeded — convert to Reject per plan §11.1.
                    ctx.events.emit(
                        turn_span,
                        EventKind::CriticRejected(json!({
                            "critic": critic.name(),
                            "reason": format!(
                                "revision depth cap ({}) exceeded: {reason}",
                                ctx.revision_depth_cap
                            ),
                            "revision_depth": depth,
                        })),
                        Some(turn_open_seq),
                    );
                    return CriticOutcome::Terminate {
                        error: RunError {
                            category: RunErrorCategory::Critic,
                            message: format!(
                                "critic `{}`: revision depth cap ({}) exceeded",
                                critic.name(),
                                ctx.revision_depth_cap
                            ),
                        },
                    };
                }
                // Emit critic.revised (critic's declaration) + turn.revised
                // (loop's view of the swap), per plan §11.1.
                let critic_revised_seq = ctx.events.emit(
                    turn_span,
                    EventKind::CriticRevised(json!({
                        "critic": critic.name(),
                        "reason": reason,
                        "revision_depth": depth,
                    })),
                    Some(turn_open_seq),
                );
                ctx.events.emit(
                    turn_span,
                    EventKind::TurnRevised(TurnRevisedPayload {
                        original_seq,
                        replacement_seq: critic_revised_seq,
                        reason,
                    }),
                    Some(turn_open_seq),
                );
                current_message = replacement.message;
                current_usage = replacement.usage;
                current_stop = replacement.stop_reason;
                // Loop again — the revised turn is itself assessed.
                continue;
            }
        }
    }

    // Unreachable: loop returns in every branch except Revise, and Revise
    // checks `depth >= cap` to bail out. Defensive fallback:
    CriticOutcome::Continue {
        effective_message: current_message,
        effective_usage: current_usage,
        effective_stop: current_stop,
    }
}

/// Pull the `Content` vec out of an `Assistant` message. Used to build
/// the post-revision `CompletionResponse` fed into
/// `execute_tool_calls` — the critic may have changed which tool-use
/// blocks the agent issues.
fn extract_content_from_message(msg: Option<&Message>) -> Vec<Content> {
    match msg {
        Some(Message::Assistant { content, .. }) => content.clone(),
        _ => Vec::new(),
    }
}

async fn execute_tool_calls(
    response: &CompletionResponse,
    ctx: &LoopContext,
    messages: &mut Vec<Message>,
) -> u32 {
    let mut results: Vec<Content> = Vec::new();
    let mut count = 0u32;

    for block in &response.content {
        if let Content::ToolUse { id, name, input } = block {
            count += 1;

            // tool.call.* events are emitted by the ToolTracer wrapping
            // ctx.tools. Pass the tool_use_id via ctx.extensions so the
            // tracer can populate its payloads (see tracer::TOOL_USE_ID_KEY).
            let mut extensions = MetadataMap::new();
            extensions.insert(TOOL_USE_ID_KEY.to_string(), json!(id));
            let tool_ctx = ToolContext {
                events: ctx.events.sink().clone(),
                budget: ctx.budget.clone(),
                cancellation: ctx.cancellation.clone(),
                approval: ctx.approval.clone(),
                // Thread the loop's workspace into every tool call so
                // the shipped `fs` / `bash` tools scope to it instead
                // of cwd. Benchmark adapters populate this by building
                // the agent with `.with_workspace(loaded.workspace)`.
                workspace: ctx.workspace.clone(),
                extensions,
            };

            let outcome = ctx.tools.execute(name, input.clone(), &tool_ctx).await;
            match outcome {
                ToolOutcome::Success(output) => {
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output,
                        is_error: false,
                    });
                }
                ToolOutcome::ExecutionError {
                    message,
                    recoverable: _,
                } => {
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output: oharness_core::message::ToolOutput::text(format!(
                            "error: {message}"
                        )),
                        is_error: true,
                    });
                }
                ToolOutcome::Denied { reason } => {
                    results.push(Content::ToolResult {
                        tool_use_id: id.clone(),
                        output: oharness_core::message::ToolOutput::text(format!(
                            "denied: {reason}"
                        )),
                        is_error: true,
                    });
                }
                ToolOutcome::Cancelled => {
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
