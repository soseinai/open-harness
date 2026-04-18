//! Minimal pytest-output parser used by [`crate::SweBenchEvaluator`].
//!
//! Parses `pytest -v` output lines shaped like
//!
//! ```text
//! tests/test_foo.py::test_something PASSED                          [ 23%]
//! tests/test_foo.py::test_other    FAILED                           [ 47%]
//! ```
//!
//! SWE-bench grades by comparing per-test-id outcomes against the
//! instance's `FAIL_TO_PASS` and `PASS_TO_PASS` sets, so the parser
//! only needs to surface the id → outcome map; summary / traceback
//! lines are ignored.

use std::collections::HashMap;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PytestResults {
    /// Test id → outcome. Outcome is the raw keyword as pytest
    /// reported it: `"PASSED"`, `"FAILED"`, `"ERROR"`, `"SKIPPED"`,
    /// `"XFAIL"`, `"XPASS"`.
    pub outcomes: HashMap<String, String>,
}

impl PytestResults {
    pub fn passed(&self, test_id: &str) -> bool {
        self.outcomes
            .get(test_id)
            .map(|o| o == "PASSED" || o == "XPASS")
            .unwrap_or(false)
    }

    /// `true` if every id in `required` passed. Missing ids count as
    /// `false` (no silent acceptance).
    pub fn all_passed(&self, required: &[String]) -> bool {
        required.iter().all(|t| self.passed(t))
    }
}

/// Parse pytest output for per-test outcomes.
///
/// The parser is deliberately forgiving:
/// - It accepts both `<id> PASSED` and `<id>  PASSED [ 42%]` shapes.
/// - It ignores everything it can't classify (summary lines,
///   tracebacks, warnings, color codes pytest emits with `-v`).
/// - It respects the *last* status a given test id appears with, so
///   re-runs (retry plugins etc.) settle on the final outcome.
pub fn parse_pytest_output(output: &str) -> PytestResults {
    const OUTCOMES: &[&str] = &["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"];
    let mut outcomes: HashMap<String, String> = HashMap::new();
    for line in output.lines() {
        let line = strip_ansi(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Common pytest -v line shape: `<id> <STATUS>` possibly followed
        // by `[NN%]` or a one-line error summary.
        for status in OUTCOMES {
            if let Some(idx) = line.find(status) {
                // The id is whatever came before the status token
                // (trim trailing whitespace).
                let id = line[..idx].trim_end();
                // Skip short-summary lines that live under the
                // `=========== short test summary info ===========`
                // header: they start with the status keyword, not the
                // id. E.g. `FAILED tests/test_foo.py::test_bar`.
                if id.is_empty() {
                    continue;
                }
                // Guard: if the id doesn't look like a pytest node
                // (must contain `::`), skip.
                if !id.contains("::") {
                    continue;
                }
                outcomes.insert(id.to_string(), (*status).to_string());
                break;
            }
        }
    }
    PytestResults { outcomes }
}

fn strip_ansi(s: &str) -> String {
    // Small hand-rolled ANSI escape stripper — pytest's -v output with
    // color enabled wraps status tokens in escape sequences that
    // otherwise confuse the `line.find(status)` search. We only strip
    // CSI sequences (ESC [ … final-byte); good enough for pytest's
    // output in practice.
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1B && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Skip past the CSI final byte (any letter in @-~).
            i += 2;
            while i < bytes.len() {
                let b = bytes[i];
                i += 1;
                if (0x40..=0x7E).contains(&b) {
                    break;
                }
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_pytest_v_output() {
        let output = "\
tests/test_foo.py::test_alpha PASSED                                    [ 20%]
tests/test_foo.py::test_beta  FAILED                                    [ 40%]
tests/test_bar.py::test_gamma ERROR                                     [ 60%]
tests/test_foo.py::test_delta SKIPPED                                   [ 80%]
tests/test_foo.py::test_xf    XFAIL                                     [100%]
";
        let results = parse_pytest_output(output);
        assert_eq!(results.outcomes.len(), 5);
        assert_eq!(
            results
                .outcomes
                .get("tests/test_foo.py::test_alpha")
                .unwrap(),
            "PASSED"
        );
        assert_eq!(
            results
                .outcomes
                .get("tests/test_foo.py::test_beta")
                .unwrap(),
            "FAILED"
        );
        assert_eq!(
            results
                .outcomes
                .get("tests/test_bar.py::test_gamma")
                .unwrap(),
            "ERROR"
        );
    }

    #[test]
    fn passed_helper_classifies_correctly() {
        let output = "\
tests/a.py::test_pass PASSED
tests/a.py::test_xpass XPASS
tests/a.py::test_fail FAILED
";
        let r = parse_pytest_output(output);
        assert!(r.passed("tests/a.py::test_pass"));
        assert!(r.passed("tests/a.py::test_xpass"));
        assert!(!r.passed("tests/a.py::test_fail"));
        assert!(!r.passed("tests/a.py::test_missing"));
    }

    #[test]
    fn all_passed_returns_false_when_any_missing_or_failing() {
        let output = "tests/a.py::test_one PASSED\ntests/a.py::test_two PASSED";
        let r = parse_pytest_output(output);
        assert!(r.all_passed(&["tests/a.py::test_one".into(), "tests/a.py::test_two".into()]));
        assert!(!r.all_passed(&[
            "tests/a.py::test_one".into(),
            "tests/a.py::test_missing".into()
        ]));
    }

    #[test]
    fn short_summary_lines_are_ignored() {
        // Short summary section lines start with the status token and
        // the node id follows — those don't match our "id before
        // status" parser and should be ignored.
        let output = "\
=========== short test summary info ===========
FAILED tests/a.py::test_one
FAILED tests/a.py::test_two
=========== 2 failed in 0.1s ===========
";
        let r = parse_pytest_output(output);
        // Short-summary FAILED lines shouldn't produce entries — the
        // parser needs id-then-status.
        assert!(r.outcomes.is_empty());
    }

    #[test]
    fn strip_ansi_removes_csi_sequences() {
        let colored = "\x1B[31mtests/a.py::test_red\x1B[0m \x1B[32mPASSED\x1B[0m";
        let out = strip_ansi(colored);
        assert_eq!(out, "tests/a.py::test_red PASSED");
    }

    #[test]
    fn parses_colored_output() {
        let output = "\x1B[32mtests/a.py::test_one\x1B[0m \x1B[32mPASSED\x1B[0m                         [100%]";
        let r = parse_pytest_output(output);
        assert_eq!(r.outcomes.get("tests/a.py::test_one").unwrap(), "PASSED");
    }
}
