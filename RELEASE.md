# open-harness — release procedure

This document records the publish procedure for the crates.io +
PyPI release surface. It's targeted at maintainers running the
v1.0 release; for day-to-day development, see
[`docs/remaining-work.md`](docs/remaining-work.md).

## 0. Pre-flight

Before tagging a release:

1. **CHANGELOG is up to date.** `CHANGELOG.md` has an `[Unreleased]`
   section that will be renamed to the target version. Event-schema
   changes (if any) are also in `CHANGELOG-schema.md`, matching
   `SchemaVersion::CURRENT` in `oharness-core`.
2. **CI is green.** `just ci` on the current `main` head.
3. **Examples run.** `just examples` — all 11 runnable examples
   smoke-run clean.
4. **Schema export matches.** `just schema-check` — the committed
   `schema/events-v1.0.json` equals a fresh regeneration.
5. **Version bumps staged.** Workspace `[workspace.package].version`
   in the root `Cargo.toml` matches every crate's published
   version (they all inherit via `version.workspace = true`). The
   `[workspace.dependencies]` block has matching `version = "..."`
   entries — bump those too, they're what consumers pin to.

## 1. Publish order (crates.io)

Crates must be published in **topological dependency order**.
cargo's `--dry-run` can't validate downstream crates until their
path-deps are uploaded, so there's no shortcut here — publish one,
wait for the index to propagate, publish the next.

```
oharness-core
  └─> oharness-llm
        ├─> oharness-providers
        ├─> oharness-tools ─────┐
        ├─> oharness-memory ────┤
        ├─> oharness-critic ────┤
        ├─> oharness-budget ────┤
        │                       │
        │   oharness-trace <────┤ (also depends on llm)
        │                       │
        │   oharness-loop <─────┤ (uses trace, memory, critic, tools)
        │                       │
        │   oharness-eval <─────┘ (uses loop, tools, trace)
        │
        └─> oharness-bench-swe  (uses eval, tools)
```

Concrete sequence (linearised, safe against cargo's index
propagation):

1. `cargo publish -p oharness-core`
2. `cargo publish -p oharness-llm`
3. `cargo publish -p oharness-tools`
4. `cargo publish -p oharness-memory`
5. `cargo publish -p oharness-trace`
6. `cargo publish -p oharness-budget`
7. `cargo publish -p oharness-critic`
8. `cargo publish -p oharness-providers`
9. `cargo publish -p oharness-loop`
10. `cargo publish -p oharness-eval`
11. `cargo publish -p oharness-bench-swe`

Allow 30–60 seconds between each for the crates.io index to
update; `cargo publish` will fail with "no matching package" if
you move too fast.

`oharness-py` is **not** on crates.io (`publish = false` in its
`Cargo.toml`) — it ships via PyPI. See §3 below.

## 2. Tag + push

```bash
git tag -a v1.0.0 -m "open-harness v1.0.0"
git push origin v1.0.0
```

GitHub Release notes: paste the newly-renamed `[1.0.0]` block
from `CHANGELOG.md`.

## 3. Publish the Python wheel

```bash
cd crates/oharness-py
maturin build --release --out dist/
# Build wheels for each supported Python version / platform; in
# CI this is a maturin-action matrix (macOS arm64+x86_64, Linux
# x86_64+aarch64, Windows x86_64; Python 3.10+).
twine upload dist/*.whl
```

Verify on PyPI that the release appears under
`https://pypi.org/project/oharness/`.

## 4. Post-flight

1. **Rename `[Unreleased]` → `[<version>] — <date>`** in
   `CHANGELOG.md` and `CHANGELOG-schema.md`; open a new
   `[Unreleased]` block.
2. **Bump the workspace version to the next dev version** (e.g.
   `0.2.0-dev.0` after releasing `0.1.0`).
3. **Announce** — Rust users' forum, Hacker News, the plan's
   "first external user" outreach list.

## Troubleshooting

- **"description field missing"**: the workspace-level
  `description` default isn't inherited automatically; each crate
  sets its own `description = "..."`.
- **"readme points to a file that doesn't exist"**: every
  crate has `readme = "README.md"` in its `[package]` block, and
  a matching `README.md` in its directory. If you split a crate
  into multiple Cargo packages (e.g. renaming), copy the README
  first.
- **"license file not found"**: crates use
  `license = "MIT OR Apache-2.0"` (SPDX string), which crates.io
  validates against the dual-license LICENSE-MIT / LICENSE-APACHE
  files at the workspace root. No `license-file` key is set.
- **Publish fails mid-sequence**: crates.io publishes are
  idempotent for a given (name, version) pair. Fix the offending
  crate, bump its version, update downstream crates' version
  pins, and resume.

## Crates.io names reserved

All `oharness-*` crate names and `oharness` on PyPI were reserved
via name-availability checks at design lock (2026-04-17). Nothing
else should squat on them.
