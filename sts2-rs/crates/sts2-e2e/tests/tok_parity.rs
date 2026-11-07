//! G-tok: the Rust tokenizer (`tokens::encode`) reproduces
//! `ppo.e2e.tokens.encode(d, rare=None, priv=False)` bit for bit on the
//! recorded fixtures.
//!
//! Fixtures: `ppo.e2e.export_wb fixtures` writes `<seed>.jsonl.gz`; each
//! decision line carries Python's policy row (`rowb`: `cat`, `cont` as
//! float16 bits, `sec`, `tok_flags`, `cls_tok`, `cls_kind`, `pool_*`,
//! `mem_*`) and its `row_scalars`. The directory is `$E2E_FIXTURES` or
//! `~/runs/e2e-wb/fixtures` (skipped with a message when absent;
//! `E2E_SEED=<substring>` narrows it). Each seed is replayed through the
//! Rust env with the recorded `chosen` class; every decision's row must
//! equal Python's array for array, bit for bit. The first divergence of a
//! seed names the decision, field, token, column and both values.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use sts2_e2e::env::{E2eEnv, Step};
use sts2_e2e::schema::{cat_name, cont_name, section_name, N_CAT, N_CONT};
use sts2_e2e::tokens::{encode, TokCache, TokRow};

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

fn data(rowb: &Value, key: &str) -> Vec<i64> {
    rowb[key]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("rowb.{key}.data"))
        .iter()
        .map(|x| {
            x.as_i64()
                .unwrap_or_else(|| panic!("rowb.{key}: non-integer"))
        })
        .collect()
}

fn ints<T: Copy + Into<i64>>(v: &[T]) -> Vec<i64> {
    v.iter().map(|&x| x.into()).collect()
}

fn f16_str(bits: i64) -> String {
    let b = u16::try_from(bits).unwrap_or(0);
    format!("{} (0x{b:04x})", half::f16::from_bits(b))
}

/// The first difference between Rust's row and Python's, or `None`.
fn compare(got: &TokRow, rowb: &Value, scalars: &Value) -> Option<String> {
    let sec = data(rowb, "sec");
    let n_tok = sec.len();
    if got.sec.len() != n_tok {
        let names = |s: &[i64]| -> Vec<&str> {
            s.iter()
                .map(|&x| section_name(u8::try_from(x).unwrap_or(255)))
                .collect()
        };
        return Some(format!(
            "n_tok: rust {} vs python {n_tok}\n    rust sections:   {:?}\n    python sections: {:?}",
            got.sec.len(),
            names(&ints(&got.sec)),
            names(&sec)
        ));
    }
    let tok_sec = |t: usize| section_name(u8::try_from(sec[t]).unwrap_or(255));
    let cat = data(rowb, "cat");
    for (i, (&g, &w)) in got.cat.iter().zip(&cat).enumerate() {
        if i64::from(g) != w {
            let (t, c) = (i / N_CAT, i % N_CAT);
            return Some(format!(
                "cat token {t} ({}) column {c} ({}): rust {g} vs python {w}",
                tok_sec(t),
                cat_name(c)
            ));
        }
    }
    let cont = data(rowb, "cont");
    for (i, (&g, &w)) in got.cont.iter().zip(&cont).enumerate() {
        if i64::from(g) != w {
            let (t, c) = (i / N_CONT, i % N_CONT);
            return Some(format!(
                "cont token {t} ({}) column {c} ({}): rust {} vs python {}",
                tok_sec(t),
                cont_name(c),
                f16_str(i64::from(g)),
                f16_str(w)
            ));
        }
    }
    let fields: [(&str, Vec<i64>); 10] = [
        ("sec", ints(&got.sec)),
        ("tok_flags", ints(&got.tok_flags)),
        ("cls_tok", ints(&got.cls_tok)),
        ("cls_kind", ints(&got.cls_kind)),
        ("pool_tok", ints(&got.pool_tok)),
        ("pool_vid", ints(&got.pool_vid)),
        ("pool_w", ints(&got.pool_w)),
        ("mem_tok", ints(&got.mem_tok)),
        ("mem_class", ints(&got.mem_class)),
        ("cat_len", vec![i64::try_from(got.cat.len()).unwrap_or(-1)]),
    ];
    for (name, g) in fields {
        let w = if name == "cat_len" {
            vec![i64::try_from(cat.len()).unwrap_or(-1)]
        } else {
            data(rowb, name)
        };
        if g != w {
            let at = g.iter().zip(&w).position(|(a, b)| a != b);
            return Some(format!(
                "{name}: first difference at {at:?} (lengths rust {} python {})\n    rust:   {g:?}\n    python: {w:?}",
                g.len(),
                w.len()
            ));
        }
    }
    let n = |k: &str| scalars[k].as_i64().unwrap_or(-1);
    let sc: [(&str, i64); 8] = [
        ("is_combat", i64::from(got.is_combat)),
        ("dtype", i64::from(got.dtype)),
        ("n_classes", i64::try_from(got.cls_tok.len()).unwrap_or(-1)),
        ("n_tok", i64::try_from(got.n_tok()).unwrap_or(-1)),
        ("n_mem", i64::try_from(got.mem_tok.len()).unwrap_or(-1)),
        ("n_pool", i64::try_from(got.pool_vid.len()).unwrap_or(-1)),
        ("sctx_lo", i64::try_from(got.sctx_lo).unwrap_or(-1)),
        ("n_sctx", i64::try_from(got.n_sctx).unwrap_or(-1)),
    ];
    for (k, g) in sc {
        if g != n(k) {
            return Some(format!("row scalar {k}: rust {g} vs python {}", n(k)));
        }
    }
    None
}

struct SeedResult {
    decisions: usize,
    tok_time: Duration,
}

/// Replay one seed; `Ok` or the first divergence.
fn replay(path: &Path) -> Result<SeedResult, String> {
    let lines = read_lines(path);
    let seed = lines[0]["seed"].as_i64().ok_or("header without seed")?;
    let mut env = E2eEnv::new(seed);
    let mut cache = TokCache::new();
    let mut s = env.reset();
    let mut tok_time = Duration::ZERO;
    for (n, rec) in lines[1..].iter().enumerate() {
        if rec.get("terminal").is_some() {
            return Ok(SeedResult {
                decisions: n,
                tok_time,
            });
        }
        let Step::Decision(d) = &s else {
            return Err(format!("seed {seed}: rust env ended at decision {n}"));
        };
        let t0 = Instant::now();
        let row = encode(d, env.rm(), &mut cache);
        tok_time += t0.elapsed();
        let row = row.map_err(|e| format!("seed {seed} decision {n}: encode failed: {e}"))?;
        if let Some(diff) = compare(&row, &rec["rowb"], &rec["row_scalars"]) {
            return Err(format!(
                "seed {seed} decision {n} ({} dtype {} {}, act {}, floor {}, turn {:?}): {diff}",
                d.kind.as_str(),
                d.dtype,
                sts2_e2e::contracts::dtype_name(d.dtype),
                d.act,
                d.floor,
                d.turn()
            ));
        }
        let chosen = rec["chosen"].as_u64().ok_or("decision without chosen")?;
        s = env.step(usize::try_from(chosen).map_err(|e| e.to_string())?);
    }
    Err(format!("seed {seed}: fixture has no terminal line"))
}

#[test]
fn tokens_match_python_fixtures() {
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
    let results: std::sync::Mutex<Vec<(PathBuf, Result<SeedResult, String>)>> =
        std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(p) = files.get(i) else { break };
                let r = std::panic::catch_unwind(|| replay(p))
                    .unwrap_or_else(|_| Err(format!("{}: panicked", p.display())));
                results.lock().unwrap().push((p.clone(), r));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut decisions = 0;
    let mut tok_time = Duration::ZERO;
    let mut bad = Vec::new();
    for (p, r) in &results {
        match r {
            Ok(s) => {
                decisions += s.decisions;
                tok_time += s.tok_time;
            }
            Err(e) => bad.push(format!("{}: {e}", p.display())),
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let per = tok_time.as_secs_f64() * 1e3 / decisions.max(1) as f64;
    eprintln!(
        "tok parity: {}/{} seeds bit-identical, {decisions} decisions, {per:.3} ms/decision encode",
        results.len() - bad.len(),
        results.len()
    );
    assert!(bad.is_empty(), "divergences:\n{}", bad.join("\n\n"));
}
