//! OpenAI-compatible provider factories.
//!
//! OpenRouter, Ollama, and self-hosted vLLM all speak the OpenAI Chat
//! Completions wire protocol — they differ only in:
//!
//! - base URL (`https://openrouter.ai/api/v1`, `http://localhost:11434/v1`,
//!   or a user-supplied vLLM endpoint)
//! - whether an `Authorization: Bearer …` header is required (OpenRouter
//!   yes; Ollama no; vLLM sometimes)
//! - environment-variable conventions for API keys
//! - extra HTTP headers (OpenRouter's optional `HTTP-Referer` / `X-Title`
//!   attribution pair)
//! - the provider name that should land in `llm.name()` so trajectory
//!   events stay identifiable
//!
//! Each factory here returns a pre-configured [`crate::OpenAiLlm`]. The
//! concrete `OpenAiLlm` type means users can still reach for the full
//! `with_*` builder surface (`with_timeout`, `with_extra_header`,
//! `with_capabilities`, …) if they need to tune anything further.

use crate::openai::OpenAiLlm;
use oharness_llm::LlmError;
use std::env;

// ======================================================================
// OpenRouter
// ======================================================================

/// Factory namespace for [OpenRouter](https://openrouter.ai)'s
/// OpenAI-compatible endpoint.
#[cfg(feature = "openrouter")]
pub struct OpenRouter;

#[cfg(feature = "openrouter")]
impl OpenRouter {
    /// Base URL of the OpenRouter Chat Completions endpoint.
    pub const BASE_URL: &'static str = "https://openrouter.ai/api/v1/chat/completions";

    /// Construct from the `OPENROUTER_API_KEY` environment variable.
    ///
    /// `model` should be the OpenRouter-qualified name, e.g.
    /// `"anthropic/claude-sonnet-4-5"` or `"openai/gpt-4o"`.
    pub fn from_env(model: impl Into<String>) -> Result<OpenAiLlm, LlmError> {
        let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::Authentication)?;
        Ok(Self::new(api_key, model))
    }

    /// Construct with an explicit API key.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> OpenAiLlm {
        OpenAiLlm::new(api_key, model)
            .with_base_url(Self::BASE_URL)
            .with_name("openrouter")
    }

    /// Same as [`OpenRouter::from_env`] but also sets OpenRouter's optional
    /// `HTTP-Referer` and `X-Title` attribution headers.
    pub fn from_env_with_attribution(
        model: impl Into<String>,
        referer: impl Into<String>,
        title: impl Into<String>,
    ) -> Result<OpenAiLlm, LlmError> {
        let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::Authentication)?;
        Ok(Self::new(api_key, model)
            .with_extra_header("HTTP-Referer", referer)
            .with_extra_header("X-Title", title))
    }
}

// ======================================================================
// Ollama
// ======================================================================

/// Factory namespace for [Ollama](https://ollama.com)'s local
/// OpenAI-compatible endpoint.
#[cfg(feature = "ollama")]
pub struct Ollama;

#[cfg(feature = "ollama")]
impl Ollama {
    /// Default local URL Ollama exposes.
    pub const LOCAL_URL: &'static str = "http://localhost:11434/v1/chat/completions";

    /// Point at the default local Ollama daemon. No authentication header
    /// is sent — Ollama's compatibility endpoint doesn't expect one.
    ///
    /// `model` is the raw Ollama model name, e.g. `"llama3.2"` or
    /// `"qwen2.5:7b-instruct"`.
    pub fn local(model: impl Into<String>) -> OpenAiLlm {
        Self::at(Self::LOCAL_URL, model)
    }

    /// Point at an Ollama-compatible endpoint at the given URL. No auth.
    pub fn at(url: impl Into<String>, model: impl Into<String>) -> OpenAiLlm {
        OpenAiLlm::without_auth(model)
            .with_base_url(url)
            .with_name("ollama")
    }
}

// ======================================================================
// vLLM
// ======================================================================

/// Factory namespace for self-hosted [vLLM](https://github.com/vllm-project/vllm)
/// deployments exposing the OpenAI-compatible Chat Completions endpoint.
#[cfg(feature = "vllm")]
pub struct Vllm;

#[cfg(feature = "vllm")]
impl Vllm {
    /// Point at an unauthenticated vLLM instance.
    pub fn at(url: impl Into<String>, model: impl Into<String>) -> OpenAiLlm {
        OpenAiLlm::without_auth(model)
            .with_base_url(url)
            .with_name("vllm")
    }

    /// Point at a vLLM instance fronted by a bearer-token auth shim.
    pub fn at_with_key(
        url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> OpenAiLlm {
        OpenAiLlm::new(api_key, model)
            .with_base_url(url)
            .with_name("vllm")
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oharness_llm::Llm;

    #[cfg(feature = "openrouter")]
    #[test]
    fn openrouter_factory_sets_name_and_url() {
        let llm = OpenRouter::new("sk-test", "anthropic/claude-sonnet-4-5");
        assert_eq!(llm.name(), "openrouter");
        // Verify the BASE_URL constant round-trips cleanly through construction.
        // We can't peek into private fields, but the factory chains
        // `with_base_url` last so this is a shape check on the public API.
        assert_eq!(
            OpenRouter::BASE_URL,
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[cfg(feature = "openrouter")]
    #[test]
    fn openrouter_from_env_missing_key_is_authentication_error() {
        // Ensure the env var isn't present for this probe.
        std::env::remove_var("OPENROUTER_API_KEY");
        match OpenRouter::from_env("anthropic/claude-sonnet-4-5") {
            Err(LlmError::Authentication) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("should have failed without OPENROUTER_API_KEY"),
        }
    }

    #[cfg(feature = "openrouter")]
    #[test]
    fn openrouter_from_env_with_attribution_requires_key() {
        std::env::remove_var("OPENROUTER_API_KEY");
        match OpenRouter::from_env_with_attribution(
            "anthropic/claude-sonnet-4-5",
            "https://example.com",
            "my-agent",
        ) {
            Err(LlmError::Authentication) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("should have failed without OPENROUTER_API_KEY"),
        }
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_local_sets_name() {
        let llm = Ollama::local("llama3.2");
        assert_eq!(llm.name(), "ollama");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_local_url_constant() {
        assert_eq!(
            Ollama::LOCAL_URL,
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_at_accepts_custom_url() {
        let llm = Ollama::at("http://10.0.0.5:8080/v1/chat/completions", "mistral:7b");
        assert_eq!(llm.name(), "ollama");
    }

    #[cfg(feature = "vllm")]
    #[test]
    fn vllm_at_sets_name_without_auth() {
        let llm = Vllm::at(
            "http://vllm.internal:8000/v1/chat/completions",
            "meta-llama/Llama-3.1-8B-Instruct",
        );
        assert_eq!(llm.name(), "vllm");
    }

    #[cfg(feature = "vllm")]
    #[test]
    fn vllm_at_with_key_sets_name() {
        let llm = Vllm::at_with_key(
            "http://vllm.internal:8000/v1/chat/completions",
            "secret",
            "meta-llama/Llama-3.1-8B-Instruct",
        );
        assert_eq!(llm.name(), "vllm");
    }
}
