//! G-play / G-wr: the Rust student plays whole runs on its own (env →
//! tokenizer → net → argmax) and, given a Python panel, checks every run's
//! chosen-class sequence against it.
//!
//! ```text
//! cargo run --release -p sts2-e2e --example play -- WEIGHTS SEED_START N [THREADS] [PY_PANEL.jsonl]
//! ```
//!
//! `PY_PANEL.jsonl` is `verify_wr.py`'s output (one row per seed with
//! `seed`, `won`, `chosen`). Prints one JSON row per seed to stdout and a
//! summary to stderr: win rate, ms per decision split into env / tokenize /
//! net, and the seeds whose sequences differ from Python's (first differing
//! decision).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sts2_e2e::env::{E2eEnv, Step};
use sts2_e2e::net::{E2eNet, PolicyRow};
use sts2_e2e::tokens::{encode, TokCache};

struct RunOut {
    seed: i64,
    won: bool,
    outcome: &'static str,
    chosen: Vec<usize>,
    env_s: f64,
    tok_s: f64,
    net_s: f64,
}

fn play(net: &E2eNet, seed: i64) -> RunOut {
    let mut env = E2eEnv::new(seed);
    let mut cache = TokCache::new();
    let (mut env_s, mut tok_s, mut net_s) = (0.0, 0.0, 0.0);
    let mut chosen = Vec::new();
    let t = Instant::now();
    let mut step = env.reset();
    env_s += t.elapsed().as_secs_f64();
    loop {
        match step {
            Step::Terminal(term) => {
                return RunOut {
                    seed,
                    won: term.won,
                    outcome: term.outcome,
                    chosen,
                    env_s,
                    tok_s,
                    net_s,
                };
            }
            Step::Decision(d) => {
                let t0 = Instant::now();
                let row: PolicyRow = encode(&d, env.rm(), &mut cache)
                    .unwrap_or_else(|e| panic!("seed {seed}: tokenize: {e}"))
                    .into();
                let t1 = Instant::now();
                let c = net.act_greedy(&row);
                let t2 = Instant::now();
                chosen.push(c);
                step = env.step(c);
                let t3 = Instant::now();
                tok_s += (t1 - t0).as_secs_f64();
                net_s += (t2 - t1).as_secs_f64();
                env_s += (t3 - t2).as_secs_f64();
            }
        }
    }
}

fn read_panel(path: &str) -> BTreeMap<i64, (bool, Vec<usize>)> {
    let text = std::fs::read_to_string(path).expect("read panel");
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).expect("panel json");
        let seed = v["seed"].as_i64().expect("seed");
        let won = v["won"].as_bool().unwrap_or(false);
        let ch = v["chosen"]
            .as_array()
            .expect("chosen")
            .iter()
            .map(|x| usize::try_from(x.as_u64().expect("class")).expect("usize"))
            .collect();
        out.insert(seed, (won, ch));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "usage: play WEIGHTS SEED_START N [THREADS] [PY_PANEL.jsonl]"
    );
    let bytes = std::fs::read(&args[1]).expect("read weights");
    let net = Arc::new(E2eNet::from_bytes(&bytes).expect("weights"));
    let start: i64 = args[2].parse().expect("seed start");
    let n: i64 = args[3].parse().expect("n");
    let threads: usize = args.get(4).map_or(8, |s| s.parse().expect("threads"));
    let panel = args.get(5).map(|p| Arc::new(read_panel(p)));

    let next = Arc::new(AtomicI64::new(start));
    let results = Arc::new(Mutex::new(Vec::<RunOut>::new()));
    let wall = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (net, next, results) = (Arc::clone(&net), Arc::clone(&next), Arc::clone(&results));
            std::thread::spawn(move || loop {
                let seed = next.fetch_add(1, Ordering::Relaxed);
                if seed >= start + n {
                    break;
                }
                let r = play(&net, seed);
                results.lock().expect("lock").push(r);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker");
    }
    let mut rs = Arc::try_unwrap(results)
        .ok()
        .expect("sole owner")
        .into_inner()
        .expect("lock");
    rs.sort_by_key(|r| r.seed);

    let (mut wins, mut decisions) = (0usize, 0usize);
    let (mut env_s, mut tok_s, mut net_s) = (0.0, 0.0, 0.0);
    let (mut same, mut diff, mut missing, mut py_wins) = (0usize, Vec::new(), 0usize, 0usize);
    for r in &rs {
        wins += usize::from(r.won);
        decisions += r.chosen.len();
        env_s += r.env_s;
        tok_s += r.tok_s;
        net_s += r.net_s;
        println!(
            "{}",
            serde_json::json!({"seed": r.seed, "won": r.won, "outcome": r.outcome, "decisions": r.chosen.len()})
        );
        if let Some(p) = &panel {
            match p.get(&r.seed) {
                None => missing += 1,
                Some((pw, pc)) => {
                    py_wins += usize::from(*pw);
                    if *pc == r.chosen {
                        same += 1;
                    } else {
                        let at = pc
                            .iter()
                            .zip(&r.chosen)
                            .position(|(a, b)| a != b)
                            .unwrap_or(pc.len().min(r.chosen.len()));
                        diff.push((r.seed, at, pc.len(), r.chosen.len()));
                    }
                }
            }
        }
    }
    let nd = decisions.max(1) as f64;
    let summary = serde_json::json!({
        "runs": rs.len(),
        "wins": wins,
        "win_rate": wins as f64 / rs.len().max(1) as f64,
        "decisions": decisions,
        "ms_per_decision": {"env": 1e3 * env_s / nd, "tokenize": 1e3 * tok_s / nd, "net": 1e3 * net_s / nd},
        "wall_s": wall.elapsed().as_secs_f64(),
        "threads": threads,
        "panel": panel.as_ref().map(|_| serde_json::json!({
            "identical": same, "differ": diff.len(), "missing": missing, "python_wins": py_wins,
            "first_diffs": diff.iter().take(20).map(|(s, at, lp, lr)| serde_json::json!({"seed": s, "at": at, "py_len": lp, "rs_len": lr})).collect::<Vec<_>>(),
        })),
    });
    eprintln!("{summary}");
}
