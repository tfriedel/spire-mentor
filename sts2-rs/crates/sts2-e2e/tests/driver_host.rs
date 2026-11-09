//! A host that owns the run (the workbench) reproduces [`E2eEnv`] with
//! [`Driver::peek`] + its own applies: same decisions, same keys, same end.
//!
//! The chooser is the shipped student (`sts2-pycli/data/e2e/*.e2ew`, greedy)
//! when the weights are in the tree, else a deterministic stand-in (class
//! `(row * 7 + 3) % n`, which dies in the first fights). The host also asks "what would you do here?"
//! on a CLONE of its driver before every real peek (the workbench's
//! auto-action / champion pick), applies forced steps and combat decisions
//! through `apply_with`, and applies every other out-of-combat decision
//! behind the driver's back with `note_external` (a human's move).

use std::path::PathBuf;

use serde_json::Value;
use sts2_core::run_manager::RunManager;
use sts2_e2e::env::{new_run, take_run_action, ApplyKind, Decision, Driver, E2eEnv, Next, Step};
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::{encode, TokCache};

fn weights() -> Option<E2eNet> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../sts2-pycli/data/e2e/ironclad_e2e_d2500_avg5_ema0999.e2ew");
    let bytes = std::fs::read(p).ok()?;
    Some(E2eNet::from_bytes(&bytes).expect("shipped weights load"))
}

fn choose(net: Option<&E2eNet>, d: &Decision, rm: &RunManager, cache: &mut TokCache) -> usize {
    match net {
        Some(net) => net.act_greedy(&encode(d, rm, cache).expect("tokenize").into()),
        None => (d.row * 7 + 3) % d.n_classes(),
    }
}

fn sig(d: &Decision) -> Sig {
    (
        d.row,
        d.dtype,
        d.keys.clone(),
        d.members.clone(),
        d.sel.to_value(),
    )
}

/// The env's own run: decision signatures and the terminal outcome.
type Sig = (usize, u8, Vec<String>, Vec<Vec<usize>>, Value);

fn env_run(net: Option<&E2eNet>, seed: i64, max: usize) -> (Vec<Sig>, String) {
    let mut env = E2eEnv::new(seed);
    let mut cache = TokCache::new();
    let mut out = Vec::new();
    let mut s = env.reset();
    loop {
        match s {
            Step::Terminal(t) => return (out, t.outcome.to_string()),
            Step::Decision(d) => {
                out.push(sig(&d));
                if out.len() >= max {
                    return (out, "stopped".into());
                }
                let c = choose(net, &d, env.rm(), &mut cache);
                s = env.step(c);
            }
        }
    }
}

/// The host's run over the same seed.
fn host_run(net: Option<&E2eNet>, seed: i64, max: usize) -> (Vec<Sig>, String) {
    let mut rm = new_run(seed);
    let mut cache = TokCache::new();
    let mut drv = Driver::new(seed);
    let mut out = Vec::new();
    loop {
        // A recommendation on a clone must not disturb the real driver.
        let mut probe = drv.clone();
        let _ = probe.peek(&mut rm);
        match drv.peek(&mut rm) {
            Next::Terminal(t) => return (out, t.outcome.to_string()),
            Next::Forced(f) => {
                drv.apply_with(&mut rm, &f.action, f.kind, take_run_action)
                    .expect("forced step");
            }
            Next::Decision(d) => {
                out.push(sig(&d));
                if out.len() >= max {
                    return (out, "stopped".into());
                }
                let c = choose(net, &d, &rm, &mut cache);
                let (action, kind) = d.class_action(c).expect("class action");
                if kind == ApplyKind::Combat || d.row % 2 == 0 {
                    drv.apply(&mut rm, &action, kind).expect("apply");
                } else {
                    let mark = Driver::grant_mark(&rm);
                    take_run_action(&mut rm, &action).expect("take");
                    drv.note_external(&rm, mark, false);
                }
            }
        }
    }
}

#[test]
fn host_peek_apply_reproduces_the_env() {
    let net = weights();
    let seeds: &[i64] = if net.is_some() {
        &[170_000_000, 170_000_002]
    } else {
        &[170_000_000, 170_000_002, 170_000_007, 5, 77]
    };
    for &seed in seeds {
        let (want, want_end) = env_run(net.as_ref(), seed, 2000);
        let (got, got_end) = host_run(net.as_ref(), seed, 2000);
        assert!(
            want.len() > if net.is_some() { 300 } else { 15 } && want_end != "error",
            "seed {seed}: {} decisions, end {want_end}",
            want.len()
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "seed {seed}: decision {i} differs");
        }
        assert_eq!(got.len(), want.len(), "seed {seed}: decision count");
        assert_eq!(got_end, want_end, "seed {seed}: end");
        eprintln!("seed {seed}: {} decisions, {want_end}", want.len());
    }
}
