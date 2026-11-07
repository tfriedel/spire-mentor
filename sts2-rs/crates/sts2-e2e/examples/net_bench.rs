//! ms per decision of the e2e policy forward over every fixture decision,
//! single thread:
//!
//! ```text
//! cargo run --release -p sts2-e2e --example net_bench
//! ```
//!
//! Same `E2E_WEIGHTS` / `E2E_FIXTURES` as `tests/net_parity.rs`; `E2E_BENCH_N`
//! limits the run to the first N decisions.

#[path = "../tests/fixture/mod.rs"]
mod fixture;

use std::time::Instant;

use sts2_e2e::net::E2eNet;

fn main() {
    let bytes = std::fs::read(fixture::weights_path()).expect("weights (E2E_WEIGHTS)");
    let t0 = Instant::now();
    let net = E2eNet::from_bytes(&bytes).expect("load weights");
    let load = t0.elapsed();
    let mut rows = Vec::new();
    for f in fixture::fixture_files(&fixture::fixtures_dir()) {
        rows.extend(fixture::read_fixture(&f).into_iter().map(|d| d.row));
    }
    assert!(!rows.is_empty(), "no fixture decisions (E2E_FIXTURES)");
    // `E2E_BENCH_N`: time only the first N decisions (quick iteration).
    if let Some(k) = std::env::var("E2E_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        rows.truncate(k);
    }
    let toks: usize = rows.iter().map(|r| r.sec.len()).sum();
    // Warm up, then time every decision once.
    let mut sink = 0f32;
    for r in rows.iter().take(50) {
        sink += net.logits(r)[0];
    }
    let t = Instant::now();
    for r in &rows {
        sink += net.logits(r)[0];
    }
    let el = t.elapsed();
    println!(
        "load {:.1} ms; {} decisions (mean {:.1} tokens): {:.3} ms/decision, total {:.2} s (sink {sink})",
        load.as_secs_f64() * 1e3,
        rows.len(),
        toks as f64 / rows.len() as f64,
        el.as_secs_f64() * 1e3 / rows.len() as f64,
        el.as_secs_f64(),
    );
}
