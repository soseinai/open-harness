# open-harness dev commands.
# Run `just ci` before pushing — matches what CI will eventually enforce.

default: ci

# Lint, format-check, and test the whole workspace.
ci: fmt-check clippy test

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
