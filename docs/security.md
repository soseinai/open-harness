# open-harness — security & trust model

This document records the threat model, known risks, and
recommended mitigations for running open-harness agents. It's the
deliverable from the plan §21.1 / remaining-work §5 M4 security
audit gate.

The audit focuses on the shipped tool kits — everything else in
the workspace is pure data / trait code or HTTP provider
adapters that inherit `reqwest`'s threat model.

## 1. Trust model (read this first)

**open-harness is a research framework. It is not a sandbox.**

The library runs code produced by an LLM against your
filesystem, your network, and your shell. Any agent run is a
trusted execution of the LLM's output. If you can't trust the
LLM's output, you must add an external isolation layer — a
container, `firejail`, `bubblewrap`, a VM, or a remote execution
service.

Concretely:

- Tools emit `ToolOutcome` based on real system side effects.
  `BashToolSet` shells out. `FsToolSet` reads and writes files.
- `ToolContext::workspace_path` is a **convenience**, not a
  boundary. It sets the CWD for a subprocess or the fs-tool
  default root; it does not prevent a process from accessing
  absolute paths, `..` escapes, or symlinks.
- Environment variables (`ANTHROPIC_API_KEY`, `AWS_*`, SSH
  agent sockets, etc.) are inherited by default.
- Outbound network access is unrestricted.

**Who this is safe for, as shipped:**

- Developers running agents locally against their own code.
- CI systems where the agent is a known, audited prompt and the
  tools operate on ephemeral working copies.
- Benchmark evaluation inside an already-isolated container (one
  container per task — the SWE-bench reference setup).

**Who this is NOT safe for, as shipped:**

- Multi-tenant production hosting.
- Running against adversarial LLM endpoints.
- Exposing to untrusted users who can influence the prompt.
- Any context where the LLM's output can touch user-owned data
  it shouldn't.

## 2. Per-tool risk inventory

### 2.1 `BashToolSet`

Path: `crates/oharness-tools/src/bash.rs`.

Executes `/bin/bash -c <command>` as a subprocess. Caller-facing
interface is a JSON `{ command, timeout_secs? }` contract.

| Risk                                   | Severity | Status                                                                                 |
|----------------------------------------|----------|----------------------------------------------------------------------------------------|
| Arbitrary command execution            | High     | **Inherent to the tool.** Documented trust model; no mitigation at library level.       |
| `workspace_path` is not a boundary     | High     | Documented. Callers that need real isolation wrap in a container / firejail / bubblewrap.|
| Environment variable exfiltration      | Medium   | **Mitigated**: opt-in env allowlist via `BashTool::with_env_allowlist(..)`. Default: inherit all. See §3.  |
| Subprocess outlives timeout (leak)     | Medium   | **Fixed**: `kill_on_drop(true)` — the child process dies when the Future drops. See §3.|
| No cancellation during execution       | Medium   | **Fixed**: `tokio::select!` polls `ctx.cancellation` + the child concurrently. See §3.  |
| Unbounded memory on huge stdout        | Low      | 64KiB output cap; `cmd.output()` still buffers until the cap — a 10GB-producing process still OOMs before the cap fires. Accepted risk; callers who need streaming write their own `ToolSet`. |
| No Windows support                     | Low      | Documented. `/bin/bash` is hard-coded; on Windows, the tool returns an execution error. A Windows-native variant (`cmd.exe /c`) is a future addition. |
| No syscall / file-access policy        | Accepted | Would require platform-specific sandboxing (seccomp, pledge, App Sandbox). Out of scope for v1.0; use a container. |

### 2.2 `FsToolSet`

Path: `crates/oharness-tools/src/fs/`.

Exposes `fs_list`, `fs_read`, `fs_write`, `fs_stat`.

| Risk                                   | Severity | Status                                                                                 |
|----------------------------------------|----------|----------------------------------------------------------------------------------------|
| Path traversal (`../../etc/passwd`)    | Medium   | Respects `ToolContext::workspace_path()` if set — but does not reject absolute paths or `..` escapes. Callers that need real scoping wrap in a container. Documented. |
| Symlink following                      | Low      | Default OS behaviour; a well-placed symlink inside the workspace can point anywhere. Callers control the workspace contents. |
| Large file reads                       | Low      | `fs_read` has a 256KiB default cap (check module docs for the current value); truncated reads are flagged. |

## 3. Mitigations landed in this audit

The audit was opportunistic — it shipped library changes that
reduce concrete risks without claiming to make the tool a
sandbox.

### 3.1 `kill_on_drop(true)` on bash subprocesses

`tokio::process::Command::kill_on_drop(true)` is now set so that
when the enclosing `timeout(..)` future drops the child, tokio
sends `SIGKILL` to the process. Previously a timed-out bash
command would leak into the background; a
`sleep 3600 &` running with `timeout_secs = 1` would survive the
agent run.

### 3.2 Cancellation during execution

The bash tool now polls `ctx.cancellation` via `tokio::select!`
concurrently with the subprocess. A cooperating parent can
cancel a running command, not just a pending one.

### 3.3 Environment allowlist

`BashTool::with_env_allowlist(vec!["PATH", "HOME", ...])` opts
in to env-var-filtered execution. If no allowlist is set, the
subprocess inherits the full parent environment (matching
current behaviour — existing callers keep working).

Recommended for eval work / CI: set an allowlist to
`["PATH", "HOME", "USER", "SHELL", "LANG"]` — enough for most
tools, nothing sensitive.

### 3.4 Explicit Windows unsupported signal

On Windows the tool already fails (no `/bin/bash`), but the
error was cryptic. The trust-model doc above flags it
explicitly.

## 4. Recommendations for deployers

For anything beyond local-dev use, **wrap open-harness in an
isolation layer**:

1. **Container per run** — Docker, Podman, or nerdctl running
   an image with only the tools you want the LLM to reach.
   Mount the workspace read-write, mount nothing else.
2. **`firejail` (Linux)** — a user-mode sandbox. Run your
   harness binary under `firejail --private --noroot --net=none
   --seccomp` for a reasonable default.
3. **`bubblewrap` (Linux)** — lower-level than firejail; what
   Flatpak uses. Good for custom policies.
4. **Per-task ephemeral VM** — the strongest isolation; also
   the slowest. Used by some SWE-bench reference setups.

The library's `ApprovalChannel` (declared in
`oharness-core/src/context.rs`, default is `NullApprovalChannel`
— pass-through) is also a deferral lever: wire a
human-in-the-loop approval channel, and every bash command
requires explicit OK before execution. This is the
out-of-the-box mitigation the library provides; it's not a
substitute for isolation, but it is a useful defence-in-depth.

## 5. Deferred hardening (post-v1.0)

These are known improvements that are not in v1.0 scope:

- **`python-sandbox` tool** (plan §7.5) — when it lands, it
  must run the child via `subprocess` isolation (`firejail` on
  Linux, or per-task Docker on macOS/Windows). The bare
  `bash`-equivalent semantics are a non-starter.
- **MCP `consume` hardening** (plan §7.6) — MCP client isn't
  shipped yet; when it is, per-server sandboxing, timeouts, and
  reconnection are required before the tool is default-on.
- **Platform-specific sandboxing** — seccomp filters on Linux,
  `sandbox-exec` on macOS, Windows Job Objects. Opt-in flag on
  `BashTool`. Meaningful work; scheduled for a later milestone.

## 6. Reporting security issues

Found a vulnerability? Open a private security advisory on the
GitHub repository. Do not file it as a public issue until a fix
is staged and released.
