//! Provider-specific prompt-caching layers (plan §5.7).
//!
//! `PromptCaching` is a zero-sized factory namespace; each provider that
//! supports caching exposes a construction function (e.g.
//! [`PromptCaching::anthropic`]) returning a concrete layer that can be
//! attached via `LlmExt::try_with_layer` — the capability check is what
//! makes this `try_` rather than infallible.

pub struct PromptCaching;

impl PromptCaching {
    /// Layer for Anthropic's `cache_control` extension. Fails construction
    /// if `inner.capabilities().prompt_caching == false`. Runtime
    /// behaviour is a passthrough — the AnthropicLlm request encoder
    /// reads `CompletionRequest.cache_hints` directly.
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> crate::anthropic::AnthropicPromptCaching {
        crate::anthropic::AnthropicPromptCaching
    }
}
