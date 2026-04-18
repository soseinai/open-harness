# open-harness dev commands.
# Run `just ci` before pushing — matches what CI will eventually enforce.

default: ci

# Lint, format-check, test, smoke-run examples, and verify the
# committed event schema is up-to-date.
ci: fmt-check clippy test examples schema-check

# Fail if any file isn't rustfmt-clean.
fmt-check:
    cargo fmt --all --check

# Apply rustfmt in-place.
fmt:
    cargo fmt --all

# Deny-warnings clippy over all crates and targets.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run the full workspace test suite.
test:
    cargo test --workspace

# Build the workspace (dev profile).
build:
    cargo build --workspace

# Smoke-run the example binaries that are safe to invoke in CI (no
# network, no disk-dirtying side effects). Tool-style examples like
# `gen_v1_fixture` and `export_schema` are built-but-not-run — they
# mutate on-disk fixtures and are only run manually when the schema
# legitimately changes.
#
# Most user-facing examples live in `oharness-loop/examples/`. They
# wire the shipped subsystems (tool calls, critics, budgets, replay,
# reflexion) against a scripted `Llm` so CI runs them without any
# API keys, network, or cost.
examples:
    cargo build --workspace --examples
    cargo build --workspace --examples --features oharness-loop/reflexion
    cargo build --workspace --examples --features oharness-loop/conversation
    cargo build --workspace --examples --features oharness-loop/llm-judge
    cargo run -p oharness-loop --example hello_scripted
    cargo run -p oharness-loop --example react_with_tools
    cargo run -p oharness-loop --example custom_critic
    cargo run -p oharness-loop --example budget_enforcement
    cargo run -p oharness-loop --example replay_trajectory
    cargo run -p oharness-loop --example reflexion_run --features reflexion
    cargo run -p oharness-loop --example self_refine
    cargo run -p oharness-loop --example llm_judge_critic --features llm-judge
    cargo run -p oharness-loop --example custom_middleware
    cargo run -p oharness-loop --example custom_memory_policy
    cargo run -p oharness-loop --example multi_agent_conversation --features conversation

# Verify the committed Event JSON Schema matches a fresh export
# (plan §19.2). Runs the drift test under the `schemars-export`
# feature; a mismatch means someone touched the wire format without
# regenerating `schema/events-v1.0.json` and, if appropriate,
# bumping `SchemaVersion::CURRENT`.
schema-check:
    cargo test -p oharness-core --features schemars-export --test schema_up_to_date

# Regenerate `crates/oharness-core/schema/events-v1.0.json` from the
# current type graph. Run this after any intentional schema change,
# then commit the updated file alongside the code change + a
# CHANGELOG-schema.md entry.
schema-export:
    cargo run -p oharness-core --example export_schema --features schemars-export

# Rust-side lint/check for the `oharness-py` pyo3 crate. Opt-in (NOT
# part of `just ci`) because, while the `abi3-py310` feature lets
# `cargo check` / `cargo clippy` run without Python headers, a full
# `maturin develop` build still needs them. Keep CI runners happy
# and let contributors working on the bindings run this explicitly.
# A full build uses `maturin develop --release` from that crate dir.
python-check:
    cd crates/oharness-py && cargo check
    cd crates/oharness-py && cargo clippy --all-targets -- -D warnings
