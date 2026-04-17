//! Feature-gated LLM provider adapters.
//!
//! M1a ships the Anthropic adapter with **non-streaming only** (`complete()`).
//! Streaming, prompt caching, and additional providers (OpenAI, OpenRouter, Ollama,
//! vLLM) land in M1b.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

pub mod caching;

#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicLlm, AnthropicPromptCaching};
pub use caching::PromptCaching;
#[cfg(feature = "openai")]
pub use openai::OpenAiLlm;
