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
pub const E2E_FILE: &str = "ironclad_e2e_d2500_avg5_ema0999.e2ew";
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

/// How [`play`] ended.
struct Played {
    outcome: String,
    won: bool,
    error: Option<String>,
    floor: i32,
    act: u8,
    chosen: Vec<usize>,
    ms_tok: f64,
    ms_net: f64,
}

/// Play `env` to its end (or `max_decisions`), answering each decision with
/// `pick(logits)`.
fn play(
    net: &E2eNet,
    env: &mut E2eEnv,
    max_decisions: usize,
    clock: MsClock,
    mut pick: impl FnMut(&[f32]) -> usize,
) -> Played {
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
                let t0 = clock();
                let logits = tokens::encode(&d, env.rm(), &mut cache)
                    .map(|row| (row, clock()))
                    .and_then(|(row, t1)| net.try_logits(&row.into()).map(|l| (l, t1)));
                match logits {
                    Ok((logits, t1)) => {
                        ms_tok += t1 - t0;
                        ms_net += clock() - t1;
                        let class = pick(&logits);
                        chosen.push(class);
                        step = env.step(class);
                    }
                    Err(e) => {
                        let rs = &env.rm().run_state;
                        break ("error".into(), false, Some(e), rs.total_floor, rs.act_index);
                    }
                }
            }
        }
    };
    Played {
        outcome,
        won,
        error,
        floor,
        act,
        chosen,
        ms_tok,
        ms_net,
    }
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
    let p = play(net, &mut env, max_decisions, clock, sts2_e2e::net::argmax);
    let ms_total = clock() - t_start;
    let n = p.chosen.len();
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
        "won": p.won,
        "outcome": p.outcome,
        "error": p.error,
        "floor": p.floor,
        "act": p.act,
        "decisions": n,
        "chosen": p.chosen,
        "ms_total": ms_total,
        "ms_per_decision": per(ms_total),
        "ms_tokenize_per_decision": per(p.ms_tok),
        "ms_net_per_decision": per(p.ms_net),
        "ms_env_per_decision": per(ms_total - p.ms_tok - p.ms_net),
    })
}

// ── seed-check attempts ───────────────────────────────────────────────────

/// Which attempt sequence an attempt belongs to: the seed check (attempt 0
/// greedy, then sampled) or the win-rate measurement (all sampled). They are
/// seeded apart, so they never share attempts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Seq {
    Check,
    Rate,
}

impl Seq {
    /// `"check"` / `"rate"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Seq::Check => "check",
            Seq::Rate => "rate",
        }
    }

    /// Parse `"check"` / `"rate"`.
    ///
    /// # Errors
    ///
    /// Anything else.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "check" => Ok(Seq::Check),
            "rate" => Ok(Seq::Rate),
            other => Err(format!("seq must be \"check\" or \"rate\", not {other:?}")),
        }
    }
}

/// The sampling temperature of attempt `k` (0 = greedy).
#[must_use]
pub fn attempt_temp(seq: Seq, k: u32) -> f32 {
    match (seq, k) {
        (Seq::Check, 0) => 0.0,
        _ => 1.0,
    }
}

/// SplitMix64: a small seeded PRNG (the browser has no other).
#[derive(Clone, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)` (53 bits).
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// The PRNG seed of attempt `k` of `seq` on the run with master seed `master`.
#[must_use]
pub fn attempt_rng_seed(master: u64, seq: Seq, k: u32) -> u64 {
    let salt = match seq {
        Seq::Check => 0xC4EC_0000_0000_0001_u64,
        Seq::Rate => 0x2A7E_0000_0000_0002_u64,
    };
    SplitMix64(master ^ salt ^ u64::from(k).wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

/// A class drawn from `softmax(logits / temp)`; `temp == 0` is argmax.
#[must_use]
pub fn sample_class(logits: &[f32], temp: f32, rng: &mut SplitMix64) -> usize {
    if temp <= 0.0 || logits.len() <= 1 {
        return sts2_e2e::net::argmax(logits);
    }
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let p = softmax(&scaled);
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (i, pi) in p.iter().enumerate() {
        acc += pi;
        if u < acc {
            return i;
        }
    }
    p.len() - 1
}

/// One seed-check attempt.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Attempt {
    pub seq: Seq,
    pub k: u32,
    pub temp: f32,
    pub won: bool,
    /// `won`, `lost`, `capped`, `error` or `stopped` (hit `max_decisions`).
    pub outcome: String,
    pub error: Option<String>,
    pub floor: i32,
    pub act: u8,
    pub decisions: usize,
    pub classes: Vec<usize>,
    pub hp: i32,
    /// The encounter the run ended in, by display name (else the room
    /// type); `None` for a win.
    pub death_room: Option<String>,
}

/// Attempt `k` of `seq` on game seed `seed` (Ironclad A10, the student's
/// run): greedy or sampled per [`attempt_temp`], seeded per
/// [`attempt_rng_seed`], so an attempt is a pure function of its inputs.
///
/// # Errors
///
/// A bad seed (see `sts2_e2e::env::canonical_game_seed`).
pub fn attempt(
    net: &E2eNet,
    seed: &str,
    ascension: u8,
    seq: Seq,
    k: u32,
    max_decisions: usize,
    clock: MsClock,
) -> Result<Attempt, String> {
    let rm = sts2_e2e::env::new_run_from_game_seed(seed, ascension)?;
    let master = rm.run_state.rng.seed;
    let mut env = E2eEnv::over_run(rm, 0);
    let temp = attempt_temp(seq, k);
    let mut rng = SplitMix64(attempt_rng_seed(master, seq, k));
    let p = play(net, &mut env, max_decisions, clock, |l| {
        sample_class(l, temp, &mut rng)
    });
    let rm = env.rm();
    let death_room = (!p.won).then(|| {
        sts2_wire::emit::run_manager::combat_encounter_id(rm).map_or_else(
            || {
                rm.run_state
                    .current_room
                    .map_or_else(|| "unknown".to_string(), |r| format!("{:?}", r.room_type))
            },
            crate::seedcheck::encounter_name,
        )
    });
    Ok(Attempt {
        seq,
        k,
        temp,
        won: p.won,
        outcome: p.outcome,
        error: p.error,
        floor: p.floor,
        act: p.act,
        decisions: p.chosen.len(),
        classes: p.chosen,
        hp: rm.run_state.current_hp,
        death_room,
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

    /// The shipped net, or `None` (the test passes with a note) when this
    /// checkout has no copy.
    fn shipped_net() -> Option<Net> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../sts2-pycli/data/e2e")
            .join(E2E_FILE);
        let Ok(b) = std::fs::read(&p) else {
            eprintln!("skipping: no {}", p.display());
            return None;
        };
        Some(load(&b).expect("the shipped net loads"))
    }

    #[test]
    fn sample_class_t0_is_argmax() {
        let mut rng = SplitMix64(1);
        assert_eq!(sample_class(&[0.1, 2.0, -1.0], 0.0, &mut rng), 1);
    }

    #[test]
    fn sample_class_follows_softmax() {
        let mut rng = SplitMix64(7);
        let logits = [0.0_f32, 3.0_f32.ln()];
        let n = 20_000;
        let ones = (0..n)
            .filter(|_| sample_class(&logits, 1.0, &mut rng) == 1)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let share = ones as f64 / f64::from(n);
        assert!((share - 0.75).abs() < 0.02, "{share}");
    }

    #[test]
    fn attempt_rng_seed_separates_sequences() {
        let m = 0x1234_5678_9ABC_DEF0;
        assert_ne!(
            attempt_rng_seed(m, Seq::Check, 5),
            attempt_rng_seed(m, Seq::Rate, 5)
        );
        assert_ne!(
            attempt_rng_seed(m, Seq::Check, 5),
            attempt_rng_seed(m, Seq::Check, 6)
        );
        assert_eq!(
            attempt_rng_seed(m, Seq::Rate, 5),
            attempt_rng_seed(m, Seq::Rate, 5)
        );
        assert_eq!(attempt_temp(Seq::Check, 0), 0.0);
        assert_eq!(attempt_temp(Seq::Check, 1), 1.0);
        assert_eq!(attempt_temp(Seq::Rate, 0), 1.0);
    }

    #[test]
    fn attempt_is_deterministic() {
        let Some(net) = shipped_net() else { return };
        let a = attempt(
            &net,
            "1SNVTCEXKR0Q",
            10,
            Seq::Check,
            3,
            usize::MAX,
            fallback_ms,
        )
        .unwrap();
        let b = attempt(
            &net,
            "1snvtcexkr0q",
            10,
            Seq::Check,
            3,
            usize::MAX,
            fallback_ms,
        )
        .unwrap();
        assert_eq!(a.classes, b.classes);
        assert_eq!(a.won, b.won);
        assert!(a.decisions > 0);
        assert_eq!(a.won, a.death_room.is_none());
    }

    /// The shipped net's greedy attempt 0 on the two reference seeds. d1250
    /// (`~/runs/seedcheck/ref/REFERENCE.md`, `ppo.e2e.solve`) won both; the
    /// d2500 avg5 net (2026-11-08) wins `1SNVTCEXKR0G` and dies to an act-2
    /// elite on floor 22 of `1SNVTCEXKR0Q`. Pinned natively (no Python
    /// cross-check for this net yet): a change means the net or engine moved.
    #[test]
    fn greedy_attempt_matches_reference() {
        let Some(net) = shipped_net() else { return };
        for (seed, won, floor) in [("1SNVTCEXKR0Q", false, 22), ("1SNVTCEXKR0G", true, 46)] {
            let a = attempt(&net, seed, 10, Seq::Check, 0, usize::MAX, fallback_ms).unwrap();
            assert_eq!((a.won, a.floor), (won, floor), "{seed}: {}", a.outcome);
            assert_eq!(a.temp, 0.0);
        }
    }

    #[test]
    fn bad_seed_is_an_error() {
        let Some(net) = shipped_net() else { return };
        assert!(attempt(&net, "AB-C", 10, Seq::Check, 0, 10, fallback_ms).is_err());
    }
}
