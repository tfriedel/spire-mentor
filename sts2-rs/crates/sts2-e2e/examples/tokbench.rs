//! Tokenizer cost on random play: the Rust env plays `N` runs from `SEED`
//! with seeded uniform class choices; prints ms per decision for the env and
//! the tokenizer (a profiling harness for the training worker's encode).
//!
//! ```text
//! cargo run --release -p sts2-e2e --example tokbench -- SEED N
//! ```

use std::time::Instant;

use sts2_e2e::env::{E2eEnv, Step};
use sts2_e2e::tokens::{encode, TokCache};

#[allow(clippy::many_single_char_names)]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seed: i64 = a.get(1).map_or(1_300_000_000, |s| s.parse().expect("SEED"));
    let n: i64 = a.get(2).map_or(20, |s| s.parse().expect("N"));
    let (mut env_s, mut tok_s, mut rows, mut toks) = (0.0, 0.0, 0usize, 0usize);
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for s in seed..seed + n {
        let mut env = E2eEnv::new(s);
        let mut cache = TokCache::new();
        let t = Instant::now();
        let mut step = env.reset();
        env_s += t.elapsed().as_secs_f64();
        while let Step::Decision(d) = step {
            let t0 = Instant::now();
            let row = encode(&d, env.rm(), &mut cache).expect("tokenize");
            let t1 = Instant::now();
            rows += 1;
            toks += row.n_tok();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            #[allow(clippy::cast_possible_truncation)]
            let c = (x % d.n_classes() as u64) as usize;
            step = env.step(c);
            tok_s += (t1 - t0).as_secs_f64();
            env_s += t1.elapsed().as_secs_f64();
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let r = rows as f64;
    println!(
        "rows {rows} tokens/row {:.1} env {:.4} ms/row tokenize {:.4} ms/row",
        toks as f64 / r,
        1e3 * env_s / r,
        1e3 * tok_s / r
    );
}
