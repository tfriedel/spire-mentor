//! G-play (env half): the Rust env driver reproduces `ppo.e2e.env.E2EEnv`
//! decision by decision on the recorded fixtures.
//!
//! Fixtures: `ppo.e2e.export_wb fixtures` writes `<seed>.jsonl.gz` (header,
//! one line per stored decision, terminal). The directory is `$E2E_FIXTURES`
//! or `~/runs/e2e-wb/fixtures`; the test skips with a message when it is
//! absent. Each seed is replayed with the fixture's `chosen` class; at every
//! decision the env's view must equal the record field for field, and the
//! run must end with the recorded outcome and decision count.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sts2_e2e::env::{Decision, E2eEnv, Step};

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("E2E_FIXTURES").map_or_else(
        || {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("runs/e2e-wb/fixtures"))
                .unwrap_or_default()
        },
        PathBuf::from,
    );
    dir.is_dir().then_some(dir)
}

fn read_lines(path: &Path) -> Vec<Value> {
    let f = std::fs::File::open(path).expect("open fixture");
    let r = std::io::BufReader::new(flate2::read::GzDecoder::new(f));
    r.lines()
        .map(|l| serde_json::from_str(&l.expect("read fixture line")).expect("fixture JSON"))
        .collect()
}

/// The env's view of a decision, in the fixture's field names.
fn view(d: &Decision) -> Vec<(&'static str, Value)> {
    vec![
        ("kind", json!(d.kind.as_str())),
        ("dtype", json!(d.dtype)),
        ("act", json!(d.act)),
        ("floor", json!(d.floor)),
        ("kinds", json!(d.kinds)),
        ("keys", json!(d.keys)),
        ("member_class", json!(d.classes)),
        ("members", json!(d.members)),
        ("actions", json!(d.actions)),
        ("obs_actions", json!(d.obs_actions)),
        ("sel", d.sel.to_value()),
        ("event_name", json!(d.event_name)),
        ("turn", json!(d.turn())),
    ]
}

/// The first field-level difference, with context.
fn diff(field: &str, got: &Value, want: &Value) -> String {
    if let (Value::Array(g), Value::Array(w)) = (got, want) {
        for (i, (a, b)) in g.iter().zip(w).enumerate() {
            if a != b {
                return format!("{field}[{i}]:\n    rust:   {a}\n    python: {b}");
            }
        }
        return format!("{field}: length rust {} vs python {}", g.len(), w.len());
    }
    format!("{field}:\n    rust:   {got}\n    python: {want}")
}

/// Replay one seed; `Ok(decisions)` or the first divergence.
fn replay(path: &Path) -> Result<usize, String> {
    let lines = read_lines(path);
    let seed = lines[0]["seed"].as_i64().ok_or("header without seed")?;
    let mut env = E2eEnv::new(seed);
    let mut s = env.reset();
    for (n, rec) in lines[1..].iter().enumerate() {
        if rec.get("terminal").is_some() {
            let Step::Terminal(t) = &s else {
                return Err(format!(
                    "seed {seed}: python ended after {n} decisions, rust has another"
                ));
            };
            let want_outcome = rec["outcome"].as_str().unwrap_or("");
            let want_won = rec["won"].as_bool().unwrap_or(false);
            let want_n = rec["decisions"].as_u64().unwrap_or(0);
            if t.outcome != want_outcome || t.won != want_won || t.rows as u64 != want_n {
                return Err(format!(
                    "seed {seed}: terminal rust ({}, won {}, {} rows, err {:?}) vs python ({want_outcome}, won {want_won}, {want_n})",
                    t.outcome, t.won, t.rows, t.error
                ));
            }
            return Ok(n);
        }
        let d = match &s {
            Step::Decision(d) => d,
            Step::Terminal(t) => {
                return Err(format!(
                    "seed {seed}: rust ended at decision {n} ({} err {:?}, floor {}), python continues (dtype {}, floor {})",
                    t.outcome, t.error, t.floor, rec["dtype"], rec["floor"]
                ))
            }
        };
        for (field, got) in view(d) {
            let want = rec.get(field).cloned().unwrap_or(Value::Null);
            if got != want {
                return Err(format!(
                    "seed {seed} decision {n} (row {}, {} dtype {}, act {}, floor {}, turn {:?}): {}",
                    d.row,
                    d.kind.as_str(),
                    d.dtype,
                    d.act,
                    d.floor,
                    d.turn(),
                    diff(field, &got, &want)
                ));
            }
        }
        let chosen = rec["chosen"].as_u64().ok_or("decision without chosen")?;
        s = env.step(usize::try_from(chosen).map_err(|e| e.to_string())?);
    }
    Err(format!("seed {seed}: fixture has no terminal line"))
}

#[test]
fn env_matches_python_fixtures() {
    let Some(dir) = fixture_dir() else {
        eprintln!("SKIP: no e2e fixtures (set E2E_FIXTURES or create ~/runs/e2e-wb/fixtures)");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read fixture dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".jsonl.gz"))
        .collect();
    files.sort();
    if let Ok(only) = std::env::var("E2E_SEED") {
        files.retain(|p| p.to_string_lossy().contains(&only));
    }
    if files.is_empty() {
        eprintln!("SKIP: no *.jsonl.gz in {}", dir.display());
        return;
    }
    let threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<(PathBuf, Result<usize, String>)>> =
        std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(p) = files.get(i) else { break };
                // A gz still being written fails to decode: report, don't panic.
                let r = std::panic::catch_unwind(|| replay(p))
                    .unwrap_or_else(|_| Err(format!("{}: panicked", p.display())));
                results.lock().unwrap().push((p.clone(), r));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut decisions = 0;
    let mut bad = Vec::new();
    for (p, r) in &results {
        match r {
            Ok(n) => decisions += n,
            Err(e) => bad.push(format!("{}: {e}", p.display())),
        }
    }
    eprintln!(
        "env parity: {}/{} seeds identical, {decisions} decisions checked",
        results.len() - bad.len(),
        results.len()
    );
    assert!(bad.is_empty(), "divergences:\n{}", bad.join("\n\n"));
}
