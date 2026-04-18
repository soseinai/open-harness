# oharness-tools

`ToolSet` trait + bundled tool kits (bash, filesystem) for
[open-harness](https://github.com/aishfenton/open-harness).

## What's in here

- **`ToolSet` trait** — the primary interface for tool collections
  agents can call.
- **`ToolContext`** — cross-cutting context threaded through every
  `execute()` call: event sink, budget, cancellation, approval
  channel, workspace scoping.
- **`Workspace`** — a scratch directory with optional cleanup.
  Benchmarks hand these to per-task runs so tools operate inside a
  scoped root.
- **Shipped tool kits**:
  - `FsToolSet` (feature `fs`, default-on): `fs_list`, `fs_read`,
    `fs_write`, `fs_stat`. Respects `ToolContext::workspace_path()`
    when set.
  - `BashToolSet` (feature `bash`, default-on): `bash` executes a
    shell command. **Currently unsandboxed** — review your
    threat model before exposing to untrusted code.

## Quickstart

```rust
use oharness_tools::fs::FsToolSet;
use std::sync::Arc;

let tools = Arc::new(FsToolSet::new());
// Pass to your Agent via `.with_tools(tools)`.
```

## Feature flags

```toml
[dependencies]
oharness-tools = { version = "0.1", default-features = false, features = ["fs"] }
```

Both `bash` and `fs` are on by default; disable the ones you
don't need.

## License

Dual-licensed under MIT or Apache-2.0.
