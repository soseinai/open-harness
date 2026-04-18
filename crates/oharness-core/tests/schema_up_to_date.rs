//! Fail CI when the committed `events-v1.0.json` schema diverges from
//! a fresh export (plan §19.2).
//!
//! This is the auto-diff check that catches accidental schema drift:
//! any change to an `#[serde(...)]` attribute on an Event-reachable
//! type, any added / removed variant, any renamed field — all surface
//! as a schema delta and fail this test. Regenerate via:
//!
//! ```bash
//! cargo run -p oharness-core --example export_schema --features schemars-export
//! ```
//!
//! then commit the updated `schema/events-v1.0.json`. Per plan §19
//! schema governance, regeneration is also the point at which you
//! bump `SchemaVersion::CURRENT` and add a `CHANGELOG-schema.md`
//! entry — the test itself doesn't enforce that half; reviewers do.
//!
//! Gated on `schemars-export` so default-feature CI runs don't need
//! the schemars dep. `just ci` drives the `schemars-export`-on
//! variant through the `schema` recipe.

#![cfg(feature = "schemars-export")]

use oharness_core::Event;

const COMMITTED: &str = include_str!("../schema/events-v1.0.json");

#[test]
fn committed_schema_matches_fresh_export() {
    let schema = schemars::schema_for!(Event);
    let fresh = serde_json::to_string_pretty(&schema).expect("serialize fresh schema");
    let fresh = format!("{fresh}\n");

    if fresh == COMMITTED {
        return;
    }

    // Produce a useful diff summary for CI logs without pulling in a
    // diff crate: show the first divergent line + its context.
    let fresh_lines: Vec<&str> = fresh.lines().collect();
    let committed_lines: Vec<&str> = COMMITTED.lines().collect();
    let mut first_diff = None;
    for (i, (a, b)) in fresh_lines.iter().zip(committed_lines.iter()).enumerate() {
        if a != b {
            first_diff = Some(i);
            break;
        }
    }
    let first_diff = first_diff.unwrap_or(committed_lines.len().min(fresh_lines.len()));
    let lo = first_diff.saturating_sub(3);
    let hi = (first_diff + 4).min(fresh_lines.len().max(committed_lines.len()));

    let mut report = String::from(
        "events-v1.0.json is out of date. \
         Regenerate with:\n  \
         cargo run -p oharness-core --example export_schema \
         --features schemars-export\n\n\
         First diverging line:\n",
    );
    for i in lo..hi {
        let c = committed_lines.get(i).copied().unwrap_or("<MISSING>");
        let f = fresh_lines.get(i).copied().unwrap_or("<MISSING>");
        report.push_str(&format!(
            "  line {i:>4}\n    committed: {c}\n    fresh    : {f}\n"
        ));
    }
    report.push_str(&format!(
        "\nSizes: committed={} lines, fresh={} lines.",
        committed_lines.len(),
        fresh_lines.len()
    ));

    panic!("{report}");
}
