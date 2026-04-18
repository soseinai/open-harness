//! v1.0 trajectory compatibility test (plan §17.4 / §19).
//!
//! Loads a hand-rolled v1.0 trajectory fixture from
//! `testdata/trajectories/v1.0/smoke.jsonl` and verifies it
//! deserializes cleanly under the current code. When the schema
//! eventually bumps to v1.1 / v2.0, this test stays as the lower
//! bound — prior-version fixtures MUST continue to parse, per the
//! §19 governance rule.
//!
//! The fixture was generated from
//! `crates/oharness-core/examples/gen_v1_fixture.rs`; that generator
//! stays in-tree as documentation for how to rebuild the canonical
//! baseline if v1.0 itself ever needs regeneration (it shouldn't).

use oharness_core::event::EventKind;
use oharness_core::{Event, SchemaVersion};

const FIXTURE: &str = include_str!("../testdata/trajectories/v1.0/smoke.jsonl");

fn load_fixture() -> Vec<Event> {
    FIXTURE
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Event>(line)
                .unwrap_or_else(|e| panic!("failed to deserialize line: {e}\nline: {line}"))
        })
        .collect()
}

#[test]
fn v1_0_fixture_deserializes() {
    let events = load_fixture();
    assert!(!events.is_empty(), "fixture is empty");
}

#[test]
fn v1_0_fixture_starts_with_meta_event() {
    let events = load_fixture();
    match &events[0].kind {
        EventKind::Meta(payload) => {
            assert_eq!(payload.schema_version, SchemaVersion::CURRENT);
            assert_eq!(payload.task_snapshot.instruction, "inspect the repo");
        }
        other => panic!("expected first event to be Meta, got {other:?}"),
    }
}

#[test]
fn v1_0_fixture_carries_expected_event_kinds() {
    let events = load_fixture();
    let kinds: Vec<&'static str> = events.iter().map(event_label).collect();

    // We don't require exact order — just presence — so the fixture
    // can be extended over time without churning this check.
    for expected in [
        "meta",
        "run.started",
        "run.finished",
        "turn.started",
        "turn.finished",
        "llm.request",
        "llm.response",
        "llm.failed",
        "tool.call.started",
        "tool.call.finished",
    ] {
        assert!(
            kinds.contains(&expected),
            "fixture missing expected kind `{expected}` (got: {kinds:?})"
        );
    }
}

#[test]
fn v1_0_fixture_seqs_are_monotonic() {
    let events = load_fixture();
    for window in events.windows(2) {
        assert!(
            window[1].seq > window[0].seq,
            "seq {} → {} is not strictly increasing",
            window[0].seq,
            window[1].seq
        );
    }
}

#[test]
fn v1_0_fixture_events_share_one_run_id() {
    let events = load_fixture();
    let first = events[0].run_id;
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.run_id, first, "event #{i} has a different run_id");
    }
}

#[test]
fn v1_0_fixture_declares_schema_version_1_0() {
    let events = load_fixture();
    for e in &events {
        assert_eq!(e.v, SchemaVersion::CURRENT);
    }
}

fn event_label(e: &Event) -> &'static str {
    match &e.kind {
        EventKind::Meta(_) => "meta",
        EventKind::RunStarted(_) => "run.started",
        EventKind::RunFinished(_) => "run.finished",
        EventKind::TurnStarted(_) => "turn.started",
        EventKind::TurnFinished(_) => "turn.finished",
        EventKind::TurnRevised(_) => "turn.revised",
        EventKind::LlmRequest(_) => "llm.request",
        EventKind::LlmResponse(_) => "llm.response",
        EventKind::LlmStreamChunk(_) => "llm.stream.chunk",
        EventKind::LlmRetry(_) => "llm.retry",
        EventKind::LlmFailed(_) => "llm.failed",
        EventKind::ToolCallStarted(_) => "tool.call.started",
        EventKind::ToolCallFinished(_) => "tool.call.finished",
        EventKind::ToolCallFailed(_) => "tool.call.failed",
        EventKind::ToolApprovalRequested(_) => "tool.approval.requested",
        EventKind::ToolApprovalDecided(_) => "tool.approval.decided",
        EventKind::MemoryEvicted(_) => "memory.evicted",
        EventKind::MemorySummarized(_) => "memory.summarized",
        EventKind::MemoryRetrieved(_) => "memory.retrieved",
        EventKind::BudgetExceeded(_) => "budget.exceeded",
        EventKind::PolicyInputChecked(_) => "policy.input.checked",
        EventKind::PolicyOutputChecked(_) => "policy.output.checked",
        EventKind::PolicyBlocked(_) => "policy.blocked",
        EventKind::PlannerProposed(_) => "planner.proposed",
        EventKind::PlannerRevised(_) => "planner.revised",
        EventKind::PlannerCommitted(_) => "planner.committed",
        EventKind::CriticAssessed(_) => "critic.assessed",
        EventKind::CriticRejected(_) => "critic.rejected",
        EventKind::CriticRevised(_) => "critic.revised",
        EventKind::CriticFailed(_) => "critic.failed",
        EventKind::ReflectionGenerated(_) => "reflection.generated",
        EventKind::ReflectionInjected(_) => "reflection.injected",
        EventKind::HumanInterrupt(_) => "human.interrupt",
        EventKind::HumanInject(_) => "human.inject",
        EventKind::UserSimulatedMessage(_) => "user.simulated.message",
        EventKind::UserSimulatedEnded(_) => "user.simulated.ended",
        EventKind::UserLog(_) => "user.log",
        _ => "unknown",
    }
}
