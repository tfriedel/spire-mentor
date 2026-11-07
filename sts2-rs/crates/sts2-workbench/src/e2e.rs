//! The `e2e` brain's pure core: the e2e Ironclad student (`sts2-e2e`) as a
//! workbench brain — one transformer for every decision of a run, combat
//! included, no search.
//!
//! - [`load`]: the `.e2ew` weights the host fetched (`dist/e2e/`
//!   [`E2E_FILE`]), checked against the build the student was trained on.
//! - [`E2eState`]: the env driver state ([`Driver`]) and the tokenizer's
//!   per-run cache. The session keeps one per node (`Session::e2e_step`,
//!   `session_e2e.rs`), so seek / rewind / restore resume exactly.
//! - [`decide`]: tokenize + forward at one decision, with the softmax over
//!   the merged classes and the time each half took.
//! - [`selfplay`]: a whole run on a fresh [`E2eEnv`] with no UI
//!   (`POST /api/e2e/selfplay`), for speed and parity checks against the
//!   native `sts2-e2e/examples/play.rs`.

use std::rc::Rc;

use serde_json::{json, Value};
use sts2_core::run_manager::RunManager;
pub use sts2_e2e::env::{ApplyKind, Decision, Driver, GrantMark, Next};
use sts2_e2e::env::{E2eEnv, Step};
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::{self, TokCache};

/// The weights file the host serves under `/e2e/` (the static build copies
/// `sts2-pycli/data/e2e/<this>`).
pub const E2E_FILE: &str = "ironclad_e2e_d1250_ema0999.e2ew";
/// The only character the student plays.
pub const CHARACTER: &str = "Ironclad";
/// The build it was trained on (`sts2_e2e::env::GAME_VERSION`).
pub const GAME_VERSION: &str = sts2_e2e::env::GAME_VERSION;
/// Its training ascension, the start path's default.
pub const DEFAULT_ASCENSION: u8 = sts2_e2e::env::ASCENSION;
/// The status the brain answers when no net is loaded: the worker fetches
/// the weights on demand and retries the call once.
pub const NO_NET_STATUS: u16 = 424;

/// A loaded student, shared by the workbench and its live session.
pub type Net = Rc<E2eNet>;

/// A millisecond clock (`performance.now()` in the browser).
pub type MsClock = fn() -> f64;

/// Load the student from `.e2ew` bytes. Refuses a file stamped for another
/// game build.
///
/// # Errors
///
/// A malformed file, or a `game_version` stamp other than [`GAME_VERSION`].
pub fn load(bytes: &[u8]) -> Result<Net, String> {
    let net = E2eNet::from_bytes(bytes)?;
    match net.header().game_version.as_deref() {
        Some(v) if v != GAME_VERSION => Err(format!(
            "e2e weights are stamped for {v}, the student plays {GAME_VERSION}"
        )),
        _ => Ok(Rc::new(net)),
    }
}

/// `no Instinct net loaded: <reason>`.
#[must_use]
pub fn missing_net_error(reason: Option<&str>) -> String {
    match reason {
        Some(r) => format!("no Instinct net loaded: {r}"),
        None => format!("no Instinct net loaded ({E2E_FILE} not fetched yet)"),
    }
}

/// A monotonic millisecond clock for hosts that give none (native:
/// `Instant` since the first call; wasm: 0).
#[must_use]
pub fn fallback_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// The student's per-run state at one position: the env driver and the
/// tokenizer cache (both cheap to clone: the cache shares its map read).
#[derive(Clone, Debug)]
pub struct E2eState {
    pub driver: Driver,
    pub cache: TokCache,
}

impl E2eState {
    /// The state at a run's start.
    #[must_use]
    pub fn new(seed: i64) -> Self {
        Self {
            driver: Driver::new(seed),
            cache: TokCache::new(),
        }
    }
}

/// How the env applies what it does at a position, read off its [`Next`]:
/// `None` at the end of the run.
#[must_use]
pub fn kind_of(next: &Next) -> Option<ApplyKind> {
    match next {
        Next::Forced(f) => Some(f.kind),
        Next::Decision(d) if d.is_combat() => Some(ApplyKind::Combat),
        Next::Decision(_) => Some(ApplyKind::Run { strict: true }),
        Next::Terminal(_) => None,
    }
}

/// One decision's answer.
#[derive(Clone, Debug)]
pub struct Choice {
    /// The greedy merged class.
    pub class: usize,
    /// Softmax of the logits, per class.
    pub probs: Vec<f64>,
    /// Milliseconds spent tokenizing / in the forward.
    pub ms_tokenize: f64,
    pub ms_net: f64,
}

/// Softmax in f64 (max-subtracted).
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits
        .iter()
        .map(|&x| (f64::from(x) - f64::from(m)).exp())
        .collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

/// Tokenize decision `d` at `rm` and run the student (greedy).
///
/// # Errors
///
/// The tokenizer or the forward refused the decision.
pub fn decide(
    net: &E2eNet,
    d: &Decision,
    rm: &RunManager,
    cache: &mut TokCache,
    clock: MsClock,
) -> Result<Choice, String> {
    let t0 = clock();
    let row = tokens::encode(d, rm, cache)?;
    let t1 = clock();
    let logits = net.try_logits(&row.into())?;
    let t2 = clock();
    Ok(Choice {
        class: sts2_e2e::net::argmax(&logits),
        probs: softmax(&logits),
        ms_tokenize: t1 - t0,
        ms_net: t2 - t1,
    })
}

/// `POST /api/e2e/selfplay`: the student plays seed `seed` on a fresh
/// [`E2eEnv`] (the campaign run at `ascension`) for at most `max_decisions`
/// decisions. `chosen` is the class sequence, comparable to
/// `sts2-e2e/examples/play.rs`.
#[must_use]
pub fn selfplay(
    net: &E2eNet,
    seed: i64,
    ascension: u8,
    max_decisions: usize,
    clock: MsClock,
) -> Value {
    let t_start = clock();
    let mut env = E2eEnv::over_run(sts2_e2e::env::new_run_at(seed, ascension), seed);
    let mut cache = TokCache::new();
    let mut chosen: Vec<usize> = Vec::new();
    let (mut ms_tok, mut ms_net) = (0.0, 0.0);
    let mut step = env.reset();
    let (outcome, won, error, floor, act) = loop {
        match step {
            Step::Terminal(t) => break (t.outcome.to_string(), t.won, t.error, t.floor, t.act),
            Step::Decision(d) => {
                if chosen.len() >= max_decisions {
                    let rs = &env.rm().run_state;
                    break ("stopped".into(), false, None, rs.total_floor, rs.act_index);
                }
                match decide(net, &d, env.rm(), &mut cache, clock) {
                    Ok(c) => {
                        ms_tok += c.ms_tokenize;
                        ms_net += c.ms_net;
                        chosen.push(c.class);
                        step = env.step(c.class);
                    }
                    Err(e) => {
                        let rs = &env.rm().run_state;
                        break ("error".into(), false, Some(e), rs.total_floor, rs.act_index);
                    }
                }
            }
        }
    };
    let ms_total = clock() - t_start;
    let n = chosen.len();
    let per = |ms: f64| {
        if n == 0 {
            0.0
        } else {
            ms / f64::from(u32::try_from(n).unwrap_or(u32::MAX))
        }
    };
    json!({
        "seed": seed,
        "ascension": ascension,
        "won": won,
        "outcome": outcome,
        "error": error,
        "floor": floor,
        "act": act,
        "decisions": n,
        "chosen": chosen,
        "ms_total": ms_total,
        "ms_per_decision": per(ms_total),
        "ms_tokenize_per_decision": per(ms_tok),
        "ms_net_per_decision": per(ms_net),
        "ms_env_per_decision": per(ms_total - ms_tok - ms_net),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[1] > p[2] && p[2] > p[0]);
    }

    #[test]
    fn load_refuses_garbage() {
        assert!(load(b"not a weights file").is_err());
    }
}
