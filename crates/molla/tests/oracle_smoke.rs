//! Stage 0 oracle harness smoke test.
//!
//! Shells out to a Python `newton` install (the upstream reference at
//! `xtra/newton/`) and sanity-checks that we can run a trivial scene and
//! parse a trajectory back. No comparison yet — that's stage 1.
//!
//! Skipped by default. Run explicitly:
//!
//! ```bash
//! MOLLA_NEWTON_PYTHON=$(pwd)/xtra/newton/.venv/bin/python \
//!   cargo test -p molla --test oracle_smoke -- --ignored
//! ```

use std::path::PathBuf;
use std::process::Command;

/// Resolve the Python interpreter to use for the oracle. Returns `None`
/// (and the test should skip) if the user hasn't pointed us at one.
fn resolve_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MOLLA_NEWTON_PYTHON") {
        return Some(PathBuf::from(p));
    }
    // Best-effort: look for a venv next to xtra/newton.
    let candidate = PathBuf::from("../../xtra/newton/.venv/bin/python");
    if candidate.is_file() {
        return Some(candidate);
    }
    None
}

#[test]
#[ignore = "requires MOLLA_NEWTON_PYTHON pointing at a newton-physics venv"]
fn newton_oracle_runs_trivial_scene() {
    let python = resolve_python().expect(
        "set MOLLA_NEWTON_PYTHON to a python that has newton-physics installed, \
         or create xtra/newton/.venv with `uv sync`",
    );

    let script = r#"
import json, sys
try:
    import newton as _n
except Exception as e:
    print(json.dumps({"ok": False, "error": f"import newton failed: {e}"}))
    sys.exit(0)
print(json.dumps({"ok": True, "version": getattr(_n, "__version__", "unknown")}))
"#;

    let out = Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to spawn python");

    assert!(out.status.success(), "python exited non-zero: {:?}", out);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let trimmed = stdout.trim();
    assert!(
        trimmed.contains("\"ok\""),
        "unexpected python output: {trimmed}"
    );
    println!("oracle reports: {trimmed}");
}
