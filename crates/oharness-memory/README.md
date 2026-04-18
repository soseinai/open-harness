# oharness-memory

Pluggable memory/context strategies for
[open-harness](https://github.com/aishfenton/open-harness).

A memory policy sits between the conversation state and the LLM:
on each turn, the loop hands the policy a `ConversationView` of
everything so far and the policy returns the `Vec<Message>` the
LLM will actually see.

## Shipped policies

| Name                   | Feature          | Default | What it does                                           |
|------------------------|------------------|---------|--------------------------------------------------------|
| `Passthrough`          | `passthrough`    | ✅      | Identity — no mangling.                                |
| `TruncateAfterTokens`  | `truncate`       | ✅      | Drop from the head until under a token cap.            |
| `ElideToolResults`     | `elide`          | ✅      | Replace old tool results with short `[elided]` stubs.  |

All implement the `MemoryPolicy` trait.

## Quickstart

```rust
use oharness_memory::TruncateAfterTokens;
use std::sync::Arc;

let memory = Arc::new(TruncateAfterTokens::new(4_000));
// Pass to your Agent via `.with_memory(memory)`.
```

## Writing your own

The `custom_memory_policy` example in `oharness-loop/examples/`
ships a 30-line `KeepLastN` policy built on the trait.

## License

Dual-licensed under MIT or Apache-2.0.
