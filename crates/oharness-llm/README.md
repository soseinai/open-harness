# oharness-llm

`Llm` trait + middleware helper traits for
[open-harness](https://github.com/aishfenton/open-harness).

This crate defines the provider-facing interface plus the
middleware composition shapes. Concrete providers live in
[`oharness-providers`](https://crates.io/crates/oharness-providers);
bespoke middleware (budget, tracing, caching, replay) lives in
[`oharness-budget`](https://crates.io/crates/oharness-budget),
[`oharness-trace`](https://crates.io/crates/oharness-trace), etc.

## What's in here

- **`Llm` trait** — `complete()` + `stream()`, both required. A
  provider without streaming returns
  `LlmError::Unsupported("stream")` rather than stubbing the
  trait.
- **Middleware helper traits**:
  - `RequestLayer` — sync, mutate outgoing `CompletionRequest`.
  - `ResponseLayer` — sync, mutate incoming `CompletionResponse`
    (with a `stream_mode()` hook controlling behaviour on
    streams).
  - `FullLayer` — async, wrap the whole `complete()` / `stream()`
    call via `BoxFuture`. Two methods (not one generic
    `around<T>`) because `complete` and `stream` have different
    retry semantics.
  - `ChunkObserver` / `ChunkTransformer` — per-chunk hooks for
    streaming.
- **`LlmExt`** — fluent extension trait: `.with_request_layer(..)`,
  `.with_response_layer(..)`, `.with_full_layer(..)`,
  `.with_layer(..)` / `.try_with_layer(..)` for bespoke layers.
- **`LlmLayer` / `InfallibleLlmLayer`** — the composition type
  that wraps an inner `Llm`. Bespoke layers (prompt caching, rate
  limiters) implement these.
- **`Chunk`** — per-chunk streaming event type.

## Quickstart — middleware composition

```rust
use oharness_llm::LlmExt;
use oharness_providers::AnthropicLlm;

let llm = AnthropicLlm::from_env()?
    .with_request_layer(my_header_stamp)
    .with_response_layer(my_redactor)
    .with_full_layer(my_timer);
```

See the `custom_middleware` example in `oharness-loop/examples/`.

## License

Dual-licensed under MIT or Apache-2.0.
