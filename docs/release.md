# Release process

*How to cut a new open-harness release, and how the automation works.*

Releases publish all eleven workspace crates to crates.io. The
process is automated by `.github/workflows/release.yml`; the
fallback manual procedure lives in [`RELEASE.md`](../RELEASE.md)
at the repo root and is rarely needed.

## Doing a release

1. **Pre-flight on `main`** — confirm the items in
   [`RELEASE.md` §0](../RELEASE.md): CHANGELOG up to date, `just
   ci` green, `just examples` green, `just schema-check` clean.
2. **Trigger the workflow** — GitHub → **Actions → Release →
   Run workflow** → enter the new semver (e.g. `0.2.0` or
   `0.2.0-rc1`) → **Run workflow**.
3. **Watch the two runs.** Each release produces *two* runs in
   the Actions tab, labelled `· 1/2` and `· 2/2`:
   - **1/2 (bump + tag)** — the bot bumps `Cargo.toml`, commits
     to `main`, tags `vX.Y.Z`, and pushes both. ~30s.
   - **2/2 (publish to crates.io)** — fired automatically by the
     tag push. Publishes the eleven crates in topological order,
     sleeping 45 s between each so the crates.io index has time
     to propagate. ~10 min total.
4. **Post-flight** — follow [`RELEASE.md` §4](../RELEASE.md):
   rename `[Unreleased]` → `[X.Y.Z] — <date>` in `CHANGELOG.md`
   (and `CHANGELOG-schema.md` if the wire format changed), open
   a fresh `[Unreleased]` block.

That's it. No `cargo publish` calls from a laptop, no version
strings to hand-edit.

## How the workflow is wired

The release workflow has **two trigger paths** that together
form one logical operation:

```
┌─ Path 1: workflow_dispatch ──────────────────────────────┐
│  You click "Run workflow" with version=X.Y.Z.            │
│                                                          │
│  bump-and-tag job:                                       │
│   • mints a token from sosein-release-bot GitHub App     │
│   • cargo-edit `set-version --workspace X.Y.Z`           │
│   • commits Cargo.toml + Cargo.lock as the bot           │
│   • tags vX.Y.Z and pushes both to main                  │
└──────────────────────────────────────────────────────────┘
                            │
                            │  tag push fires the second run
                            ▼
┌─ Path 2: push tags ["v*"] ───────────────────────────────┐
│  Either fired by path 1 above, OR by                     │
│  `git tag vX.Y.Z && git push origin vX.Y.Z` from a       │
│  laptop (skips the bump step entirely).                  │
│                                                          │
│  publish-crates job:                                     │
│   • verifies tag matches workspace version               │
│   • cargo publish each of the 11 crates in topological   │
│     order (RELEASE.md §1), sleeping 45 s between each    │
└──────────────────────────────────────────────────────────┘
```

Why split into two paths instead of one big job? Because
publish-crates needs to run on the *committed and tagged* code,
not the in-memory bump from a single job — the tag push is what
guarantees the published artefacts match a real, immutable git
ref.

## Prerequisites (one-time setup)

The workflow assumes the following are already configured in
the `soseinai` GitHub organisation:

- **`CARGO_REGISTRY_TOKEN`** — org-level secret with publish
  rights on all `oharness-*` crates. Used by path 2.
- **`APP_ID` and `APP_PRIVATE_KEY`** — org-level secrets for
  the `sosein-release-bot` GitHub App. Used by path 1 to push
  the bump commit + tag.
- **`sosein-release-bot` App installed on `soseinai/open-harness`**
  with `contents: write` permission. Without this install, path
  1 cannot mint a token for this repo.
- **(Optional) Branch protection bypass for the bot** — if `main`
  has a "require PR for changes" ruleset, add the bot to its
  bypass list so it can push the bump commit directly. Without
  this, path 1 will fail on `git push origin main`.

The same App + secrets back the release flow in the sibling
`ought` repo, so they're already provisioned at the org level.

## Escape hatches and recovery

- **Tag from a laptop.** `git tag v0.2.0 && git push origin
  v0.2.0` skips path 1 entirely and goes straight to publish.
  Useful when the workspace version is already correct (e.g.
  recovering from a partial publish).
- **Publish failed mid-sequence.** crates.io publishes are
  idempotent for `(name, version)` pairs that already uploaded.
  Cause one to fail, fix the offending crate, bump just that
  crate's version (or bump the whole workspace), update
  downstream `[workspace.dependencies]` pins, and re-trigger.
  The workflow doesn't have built-in resume; it'll re-attempt
  the earlier crates and skip them as already-published.
- **Wrong version pushed.** crates.io versions are immutable
  once uploaded. Yank with `cargo yank --version X.Y.Z -p
  <crate>`, bump to `X.Y.Z+1`, re-release. There is no "undo
  publish" — confirm the version field in the workflow input
  before clicking **Run**.
- **Tag exists but you want to re-run publish.** Delete the tag
  locally and remotely (`git push --delete origin vX.Y.Z`) and
  re-push it. Path 2 fires again on the new tag push.

## When this won't work

- **`oharness-py` is not published from this workflow.** It
  ships separately to PyPI via `maturin` (see [`RELEASE.md`
  §3](../RELEASE.md)). Per its `Cargo.toml`, `publish = false`
  prevents accidental upload to crates.io.
- **No pre-flight checks in the workflow itself.** The workflow
  trusts that you ran `just ci` locally on `main` before
  triggering it. If you want a hard gate, run the [CI
  workflow](../.github/workflows/ci.yml) on the latest `main`
  commit and confirm green before triggering Release.
