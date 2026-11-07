//! G-net: the Rust policy forward reproduces torch's logits on every decision
//! of every `ppo.e2e.export_wb fixtures` file (`E2E_FIXTURES`, default
//! `~/runs/e2e-wb/fixtures`) with the weights in `E2E_WEIGHTS` (default
//! `~/runs/e2e-wb/export/ironclad_e2e_d1250_ema0999.e2ew`): max |Δlogit| <
//! 1e-4 and argmax == the fixture's `chosen` (or a float tie: Python's margin
//! over the Rust pick is within 2x the tolerance). Skips with a message when either
//! input is missing.
//!
//! The forward is ~10 ms/decision optimised and ~100x that in an unoptimised
//! build, and the fixtures hold ~24k decisions, so the files are checked on
//! worker threads and an unoptimised build (debug assertions on) checks every
//! `E2E_NET_STRIDE`-th decision (default 64) instead of all of them. An
//! optimised build checks every decision: `cargo nextest run -p sts2-e2e
//! --release net_parity`.

mod fixture;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use sts2_e2e::net::{argmax, E2eNet};

/// Max |Δlogit| allowed against torch fp32 (CPU).
const TOL: f32 = 1e-4;

#[derive(Default)]
struct Tally {
    n: usize,
    worst: f32,
    worst_at: String,
    bad_argmax: Vec<String>,
    ties: usize,
}

fn check_file(net: &E2eNet, f: &PathBuf, stride: usize) -> Tally {
    let name = f.file_name().unwrap().to_string_lossy().into_owned();
    let mut t = Tally::default();
    for dec in fixture::read_fixture(f).into_iter().step_by(stride) {
        let got = net
            .try_logits(&dec.row)
            .unwrap_or_else(|e| panic!("{name} #{}: {e}", dec.i));
        assert_eq!(
            got.len(),
            dec.logits.len(),
            "{name} #{}: class count",
            dec.i
        );
        for (a, b) in got.iter().zip(&dec.logits) {
            let e = (a - b).abs();
            if e.is_nan() || e > t.worst {
                t.worst = if e.is_nan() { f32::INFINITY } else { e };
                t.worst_at = format!("{name} #{}", dec.i);
            }
        }
        if argmax(&got) != dec.chosen {
            // A different pick is a failure only when Python's own margin
            // over the Rust pick exceeds the logit tolerance: below it the
            // two picks are a float tie (fp32 summation order decides).
            let margin = dec.logits[dec.chosen] - dec.logits[argmax(&got)];
            if margin > 2.0 * TOL {
                t.bad_argmax
                    .push(format!("{name} #{} (margin {margin:e})", dec.i));
            } else {
                t.ties += 1;
            }
        }
        t.n += 1;
    }
    t
}

#[test]
fn net_parity_all_fixtures() {
    let wp = fixture::weights_path();
    let Ok(bytes) = std::fs::read(&wp) else {
        eprintln!(
            "SKIP net_parity: no weights at {} (E2E_WEIGHTS)",
            wp.display()
        );
        return;
    };
    let files = fixture::fixture_files(&fixture::fixtures_dir());
    if files.is_empty() {
        eprintln!(
            "SKIP net_parity: no *.jsonl.gz under {} (E2E_FIXTURES)",
            fixture::fixtures_dir().display()
        );
        return;
    }
    let stride = std::env::var("E2E_NET_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) { 64 } else { 1 })
        .max(1);
    let net = E2eNet::from_bytes(&bytes).expect("load weights");
    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .clamp(1, 8);
    let next = AtomicUsize::new(0);
    let parts: Vec<Tally> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(f) = files.get(i) else { break };
                        out.push(check_file(&net, f, stride));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut all = Tally::default();
    for t in parts {
        all.n += t.n;
        if t.worst > all.worst || all.worst_at.is_empty() {
            all.worst = t.worst;
            all.worst_at = t.worst_at;
        }
        all.bad_argmax.extend(t.bad_argmax);
        all.ties += t.ties;
    }
    eprintln!(
        "net_parity: {} decisions (stride {stride}) in {} files, worst |dlogit| {:.3e} at {}, \
         argmax mismatches {}, float ties {}",
        all.n,
        files.len(),
        all.worst,
        all.worst_at,
        all.bad_argmax.len(),
        all.ties
    );
    assert!(all.n > 0, "fixtures held no decisions");
    assert!(
        all.worst < TOL,
        "worst |dlogit| {:e} at {} >= {TOL:e}",
        all.worst,
        all.worst_at
    );
    assert!(
        all.bad_argmax.is_empty(),
        "argmax != chosen at {:?}",
        all.bad_argmax
    );
}
