# open-harness

Kernel-style research framework for agent loops, memory, planning, tools, safety gates. Pairs with `lm-eval-harness`. Explicitly **not** a LangChain-style integration surface.

Workspace layout:

```
oharness-core           # pure types, event schema, context traits; serde only
oharness-llm            # Llm trait + middleware helper traits
oharness-providers      # feature-gated provider adapters (Anthropic first)
oharness-tools          # ToolSet trait + contributed tool kits
oharness-memory         # pluggable memory/context strategies
oharness-trace          # EventSink implementations, trajectory writer
oharness-loop           # Agent + Loop trait + ReactLoop
```

See [`docs/open-harness-plan.md`](docs/open-harness-plan.md) for the design spec (v1, locked 2026-04-17) and [`docs/remaining-work.md`](docs/remaining-work.md) for the milestone-by-milestone handover plan.

## Status

**M1a — minimum viable agent.** Non-streaming Anthropic + ReactLoop + FileSink + core types. Next: M1b (streaming + middleware + replayer).

## License

Dual-licensed under MIT or Apache-2.0 at your option.
