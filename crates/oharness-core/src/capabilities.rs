//! `LlmCapabilities` (§4.5). Returned by value from `Llm::capabilities()`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars-export", derive(schemars::JsonSchema))]
pub struct LlmCapabilities {
    pub streaming: bool,
    pub prompt_caching: bool,
    pub parallel_tool_use: bool,
    pub vision: bool,
    pub thinking: bool,
    pub structured_output: bool,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
}
