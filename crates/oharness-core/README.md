# oharness-core

Core types, event schema, and context-plumbing traits for
[open-harness](https://github.com/aishfenton/open-harness) — the
kernel-style research framework for LLM agent loops.

`oharness-core` is the foundation crate: every other crate in the
workspace depends on it. It deliberately carries **no IO, no async
executor, no provider integrations** — it's serde types + traits
other crates implement.

## What's in here

- **Event schema** (`Event`, `EventKind`, `SchemaVersion`) — the
  trajectory format. Governed by `CHANGELOG-schema.md`; the JSON
  Schema lives at `schema/events-v1.0.json`.
- **Message / Content** — `Message::{User,Assistant,System}`,
  `Content::{Text,ToolUse,ToolResult,Image,…}` — the building
  blocks every provider adapter speaks.
- **Task / RunOutcome / Termination** — the top-level shape an
  agent run takes in and produces.
- **AssistantTurn / AssessmentContext / Episode** — critic /
  reflector-facing views.
- **TaskEvaluator trait** — lives here (not in `oharness-eval`) so
  the loop crate can take evaluators without pulling the eval
  crate.
- **ScopedEmitter + EventSink** — the event-emission abstraction
  every other crate consumes.

## When to use this crate directly

- You're writing a trait implementation (`Llm`, `Critic`,
  `MemoryPolicy`, `ToolSet`, `Reflector`, `UserSimulator`,
  `TaskEvaluator`) and need only the plain data types.
- You're consuming a trajectory file programmatically (use
  `oharness-trace` for the JSONL reader; deserialize into `Event`
  types from here).

Most users pull in `oharness-loop` and get this transitively.

## License

Dual-licensed under MIT or Apache-2.0 (see `LICENSE-MIT` /
`LICENSE-APACHE` at the workspace root).
