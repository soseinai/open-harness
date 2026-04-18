# oharness-critic

`Critic` / `Reflector` traits, composition, and shipped impls for
[open-harness](https://github.com/aishfenton/open-harness).

## What's in here

- **`Critic` trait** — `assess(ctx) -> CriticVerdict`. Verdicts:
  - `Accept` / `AcceptWithNote(..)`
  - `Reject { reason }` — terminates the run.
  - `Revise { replacement, reason }` — in-place turn rewrite;
    the loop swaps the message and continues (up to
    `revision_depth_cap`, default 3).
  - `Abort { reason }` — hard stop.
- **`CompositeCritic`** — chain multiple critics with an
  `AggregationPolicy`: `FirstReject`, `AllMustAccept`,
  `MajorityVote`, `Weighted(..)`.
- **`Reflector` trait** — `reflect(episode) -> Option<Reflection>`.
  Runs between `run_reflexion` episodes.
- **`ReflectionInjector`** — `RequestLayer` that prepends
  accumulated reflections to the next episode's system prompt.
- **Shipped impls** (all feature-gated):
  - `LlmJudgeCritic` (`llm-judge`) — prompt a judge LLM with a
    rubric, parse `SCORE: <0..1>`, compare to threshold.
  - `TestCritic` (`test-runner`) — run unit tests as the
    acceptance criterion.
  - `RegexDenyCritic` (`regex-deny`) — reject on regex match.
  - `NullReflector`, `LlmReflector` (always) — reflexion-side
    counterparts.

## Quickstart

```rust
use oharness_critic::{AggregationPolicy, CompositeCritic};
use std::sync::Arc;

let critics = Arc::new(
    CompositeCritic::new("chain", AggregationPolicy::FirstReject)
        .push(Box::new(my_critic)),
);
// Pass to `AgentBuilder::with_critics(critics)`.
```

See the `custom_critic`, `self_refine`, and `llm_judge_critic`
examples in `oharness-loop/examples/`.

## License

Dual-licensed under MIT or Apache-2.0.
