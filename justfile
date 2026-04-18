# open-harness dev commands.
# Run `just ci` before pushing — matches what CI will eventually enforce.

default: ci

# Lint, format-check, test, and smoke-run examples across the whole workspace.
ci: fmt-check clippy test examples

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
# `gen_v1_fixture` are built-but-not-run — they mutate on-disk
# fixtures and are only run manually when the schema legitimately
# changes.
examples:
    cargo build --workspace --examples
    cargo run -p oharness-loop --example hello_scripted
