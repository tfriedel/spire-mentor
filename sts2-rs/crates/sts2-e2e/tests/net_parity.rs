//! G-net: the Rust policy forward reproduces torch's logits on every decision
//! of every `ppo.e2e.export_wb fixtures` file (`E2E_FIXTURES`, default
//! `~/runs/e2e-wb/fixtures`) with the weights in `E2E_WEIGHTS` (default
//! `~/runs/e2e-wb/export/ironclad_e2e_d5250_ema0999.e2ew`): max |Δlogit| <
//! 1e-4 and argmax == the fixture's `chosen` (or a float tie: Python's margin
//! over the Rust pick is within 2x the tolerance). Skips with a message when either
//! input is missing.
//!
//! G-probe (GH #1222): when the fixtures carry the probe's `estimate` (written
//! with `export_wb fixtures --probe`), the weights must carry the probe too,
//! and [`E2eNet::evaluate`]'s estimate must match: |Δp| < `TOL_P` for the run
//! and fight probabilities, |ΔHP| < `TOL_HP` for every HP quantile.
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
/// Max |Δp| of the probe's probabilities.
const TOL_P: f32 = 1e-3;
/// Max |ΔHP| of the probe's HP quantiles.
const TOL_HP: f32 = 1e-2;

#[derive(Default)]
struct Tally {
    n: usize,
    worst: f32,
    worst_at: String,
    bad_argmax: Vec<String>,
    ties: usize,
    n_est: usize,
    worst_p: f32,
    worst_hp: f32,
    est_errors: Vec<String>,
}

fn check_file(net: &E2eNet, f: &PathBuf, stride: usize) -> Tally {
    let name = f.file_name().unwrap().to_string_lossy().into_owned();
    let mut t = Tally::default();
    for dec in fixture::read_fixture(f).into_iter().step_by(stride) {
        let ev = net
            .evaluate(&dec.row)
            .unwrap_or_else(|e| panic!("{name} #{}: {e}", dec.i));
        if let Some(want) = &dec.estimate {
            match &ev.estimate {
                None => t
                    .est_errors
                    .push(format!("{name} #{}: weights carry no probe", dec.i)),
                Some(est) => {
                    let num = |v: &serde_json::Value| v.as_f64().map(|x| x as f32);
                    let mut dp = (est.p_run - num(&want["p_run"]).unwrap_or(f32::NAN)).abs();
                    match (est.p_fight, num(&want["p_fight"])) {
                        (Some(a), Some(b)) => dp = dp.max((a - b).abs()),
                        (None, None) => {}
                        _ => t
                            .est_errors
                            .push(format!("{name} #{}: p_fight presence differs", dec.i)),
                    }
                    let wq: Vec<f32> = want["hp_loss"]
                        .as_array()
                        .map(|a| a.iter().filter_map(num).collect())
                        .unwrap_or_default();
                    if wq.len() != est.hp_loss.len() {
                        t.est_errors
                            .push(format!("{name} #{}: hp quantile count differs", dec.i));
                    }
                    let dh = wq
                        .iter()
                        .zip(&est.hp_loss)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if dp.is_nan() || dp > t.worst_p {
                        t.worst_p = if dp.is_nan() { f32::INFINITY } else { dp };
                    }
                    if dh.is_nan() || dh > t.worst_hp {
                        t.worst_hp = if dh.is_nan() { f32::INFINITY } else { dh };
                    }
                    t.n_est += 1;
                }
            }
        }
        let got = ev.logits;
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
        all.n_est += t.n_est;
        all.worst_p = all.worst_p.max(t.worst_p);
        all.worst_hp = all.worst_hp.max(t.worst_hp);
        all.est_errors.extend(t.est_errors);
    }
    eprintln!(
        "net_parity: probe estimates checked {}, worst |dp| {:.3e}, worst |dHP| {:.3e}",
        all.n_est, all.worst_p, all.worst_hp
    );
    assert!(
        all.est_errors.is_empty(),
        "probe: {:?}",
        &all.est_errors[..all.est_errors.len().min(5)]
    );
    assert!(
        all.worst_p < TOL_P,
        "probe worst |dp| {:e} >= {TOL_P:e}",
        all.worst_p
    );
    assert!(
        all.worst_hp < TOL_HP,
        "probe worst |dHP| {:e} >= {TOL_HP:e}",
        all.worst_hp
    );
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
