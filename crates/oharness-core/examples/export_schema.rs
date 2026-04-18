//! Regenerate the canonical JSON Schema for the trajectory `Event`
//! envelope and write it to `crates/oharness-core/schema/events-v1.0.json`.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p oharness-core --example export_schema --features schemars-export
//! ```
//!
//! CI then diffs the committed file against a fresh export (see
//! `crates/oharness-core/tests/schema_up_to_date.rs`); a mismatch
//! means the schema drifted without a governance-approved bump.
//!
//! Only runs when the `schemars-export` feature is on so the default
//! build doesn't need to pull in schemars.

#[cfg(not(feature = "schemars-export"))]
fn main() {
    eprintln!(
        "export_schema requires the `schemars-export` feature:\n  \
         cargo run -p oharness-core --example export_schema \
         --features schemars-export"
    );
    std::process::exit(2);
}

#[cfg(feature = "schemars-export")]
fn main() -> std::io::Result<()> {
    use oharness_core::Event;
    use std::path::PathBuf;

    let schema = schemars::schema_for!(Event);
    let json = serde_json::to_string_pretty(&schema).expect("serialize schema");
    // Trailing newline so `diff` / editors stay happy.
    let json = format!("{json}\n");

    let out: PathBuf = [env!("CARGO_MANIFEST_DIR"), "schema", "events-v1.0.json"]
        .iter()
        .collect();
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, json.as_bytes())?;
    println!("wrote {}", out.display());
    Ok(())
}
