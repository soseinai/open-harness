# oharness-providers

LLM provider adapters for
[open-harness](https://github.com/aishfenton/open-harness) — each
feature-gated so you only pay for the providers you use.

## Shipped adapters

| Provider      | Feature         | Default | Streaming | Notes                              |
|---------------|-----------------|---------|-----------|------------------------------------|
| Anthropic     | `anthropic`     | ✅      | ✅ SSE    | `messages` endpoint                |
| OpenAI        | `openai`        |         | ✅ SSE    | Chat Completions                   |
| OpenAI Codex  | `openai-codex`  |         | ✅ SSE    | ChatGPT/Codex OAuth + Responses    |
| OpenRouter    | `openrouter`    |         | ✅ SSE    | Thin factory on top of `openai`    |
| Ollama        | `ollama`        |         | ✅ SSE    | Local, no API key                  |
| vLLM          | `vllm`          |         | ✅ SSE    | Self-hosted, OpenAI-compatible     |

All adapters implement the [`Llm`](https://crates.io/crates/oharness-llm)
trait.

## Quickstart

```rust
use oharness_providers::AnthropicLlm;
use oharness_core::{CompletionRequest, Message};

let llm = AnthropicLlm::from_env()?;  // ANTHROPIC_API_KEY
let res = llm
    .complete(CompletionRequest::new(vec![Message::user_text("hi")]))
    .await?;
println!("{:?}", res.content);
```

## Feature flags

```toml
[dependencies]
oharness-providers = { version = "0.1", default-features = false, features = ["anthropic"] }
```

Defaults to `anthropic`. Disable by setting `default-features = false`.

## License

Dual-licensed under MIT or Apache-2.0.
