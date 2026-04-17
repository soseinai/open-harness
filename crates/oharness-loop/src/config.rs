//! Agent-level configuration knobs. Many correspond to fields on `LoopContext`.

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_turns: u32,
    pub revision_depth_cap: u32,
    /// Default token budget hint for memory policies. Matches `LlmCapabilities.max_context_tokens`
    /// when available; otherwise this floor.
    pub default_token_budget: u32,
    /// If true, the CLI-facing flag `--system-prompt` (or builder
    /// `.with_system_prompt()`) prepends a system message; if false the loop only
    /// sends what the caller supplies via the initial task.
    pub include_default_system_prompt: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 30,
            revision_depth_cap: 3,
            default_token_budget: 180_000,
            include_default_system_prompt: true,
        }
    }
}
