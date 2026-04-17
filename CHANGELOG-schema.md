# Event Schema Changelog

Tracks changes to the trajectory event schema defined in
[`oharness-core/src/event.rs`](./crates/oharness-core/src/event.rs). This file is
mandated by the design spec §19.2 (event schema governance).

Every PR that touches `EventKind` or its payloads MUST add an entry here and
bump `SchemaVersion::CURRENT` accordingly.

Versioning:
- **Additive** changes (new event kinds, new optional fields) → minor bump.
- **Breaking** changes (renamed/removed fields, changed semantics) → major bump.
  A major bump starts a new `v2+` line and requires compat tests against prior
  `v1.x` trajectories (see design spec §17.4).

## [1.0] — 2026-04-17 (M1a)

Initial schema. Envelope: `{v, seq, run_id, timestamp, span_id, parent, kind,
redactions}` with `kind` flattened via serde `tag`/`content`.

`EventKind` variants declared (most payloads are still `serde_json::Value` in
M1a and will gain typed payloads as each subsystem lands):

- **Lifecycle**: `meta`, `run.started`, `run.finished`
- **Turn**: `turn.started`, `turn.finished`, `turn.revised`
- **LLM**: `llm.request`, `llm.response`, `llm.stream.chunk`, `llm.retry`,
  `llm.failed`
- **Tools**: `tool.call.started`, `tool.call.finished`, `tool.call.failed`,
  `tool.approval.requested`, `tool.approval.decided`
- **Memory**: `memory.evicted`, `memory.summarized`, `memory.retrieved`
- **Budget**: `budget.exceeded`
- **Policy**: `policy.input.checked`, `policy.output.checked`, `policy.blocked`
- **Planner**: `planner.proposed`, `planner.revised`, `planner.committed`
- **Critic**: `critic.assessed`, `critic.rejected`, `critic.revised`,
  `critic.failed`
- **Reflection**: `reflection.generated`, `reflection.injected`
- **Human / simulated user**: `human.interrupt`, `human.inject`,
  `user.simulated.message`, `user.simulated.ended`
- **User-emitted**: `user.log`
