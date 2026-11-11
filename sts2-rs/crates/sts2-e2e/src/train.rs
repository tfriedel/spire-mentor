//! `ppo.e2e.env.E2EEnv` as the training worker drives it (#1097): one segment
//! (a whole run, or a practice segment stopping at its first OOC row) with
//! the env's rewards, counters, practice captures, F9 sampling and action
//! log, bit-for-bit with the Python env.
//!
//! [`crate::env::E2eEnv`] reproduces only the decision stream (inference).
//! This module is a separate, line-by-line port of the Python env's control
//! flow, because the training env's bookkeeping hangs off exactly where the
//! Python ends a fight, assigns a reward or counts a forced step:
//!
//! * `_RunEnv._advance` / `_take` / `step` → [`TrainEnv::advance`],
//!   [`TrainEnv::take`], [`TrainEnv::run_step`];
//! * `E2EEnv._drive` / `_combat` / `_combat_apply` / `_ooc` →
//!   [`TrainEnv::drive`], [`TrainEnv::combat`], [`TrainEnv::combat_apply`],
//!   [`TrainEnv::ooc`];
//! * `_emit`, `_terminal`, `_cap`, `_error`, `_start_fight`, `_end_fight`,
//!   `_on_move`, `_f9_step` → the methods of the same names.
//!
//! What stays in Python (`ppo.e2e.rsenv`): the practice queues, the F9
//! check bodies (run on the states [`TrainEnv::drain_f9`] hands over), the
//! run fingerprint, the action log sink, the construction-time build checks.

use std::collections::BTreeMap;

use serde_json::Value;
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::GameVersion;
use sts2_wire::emit::names::phase_str;
use sts2_wire::emit::run_manager as emit_rm;

use crate::contracts::{self as ct, ak, dt, Selection};
use crate::env::{
    action_name, available, classify_surface, combat_to_rm, current_event_name,
    ensure_engine_installed, fight_room, is_combat, mask_rejectable, merged_classes, observe,
    round_of, schema, snapshot_of, take_run_action, CombatView, Decision, DecisionKind, OocView,
    ENCODABLE, MAX_RUN_STEPS,
};
use crate::intent_log::IntentLog;
use crate::pyjson::{as_int, get, truthy};
use crate::snapshot;

/// `R_WIN`, `R_BOSS`, `R_FLOOR` (env.py).
pub const R_WIN: f64 = 1.0;
/// A boss fight won before the run's end.
pub const R_BOSS: f64 = 0.05;
/// Per new floor.
pub const R_FLOOR: f64 = 0.002;

/// How a segment ends (`Segment.end`) and what the env records on the way.
#[derive(Clone, Debug)]
pub struct SegmentSpec {
    /// `end == "first_ooc"`: stop at the first OOC row (a value-only
    /// bootstrap row).
    pub first_ooc: bool,
    /// `cfg.on_capture` is set: hand a practice capture to the host on
    /// every Move into an elite or boss node.
    pub capture: bool,
    /// F9: `random.Random(f9_seed)` draws once per applied action, a draw
    /// below `f9_rate` hands the post-action state to the host.
    pub f9_seed: u64,
    /// `InvariantChecker.sample_rate`.
    pub f9_rate: f64,
    /// `EnvConfig`'s caps (`max_run_steps`, `max_fight_turns`, ...).
    pub caps: Caps,
    /// F9-sampled states kept between two drains ([`F9_QUEUE_MAX`] by
    /// default); the hits past it are counted `f9/dropped`.
    pub f9_queue_max: usize,
    /// The build every fight must simulate (`factory.check_fight`, plan
    /// §3.1); `None` skips the check.
    pub game_version: Option<GameVersion>,
}

/// `EnvConfig`'s segment caps; [`Default`] is env.py's defaults.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    /// `max_run_steps`: all steps, forced ones included.
    pub run_steps: u64,
    /// `max_fight_turns`.
    pub fight_turns: i64,
    /// `max_fight_decisions`.
    pub fight_decisions: usize,
    /// `max_run_decisions`.
    pub run_decisions: usize,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            run_steps: MAX_RUN_STEPS,
            fight_turns: ct::MAX_FIGHT_TURNS,
            fight_decisions: ct::MAX_FIGHT_DECISIONS,
            run_decisions: ct::MAX_RUN_DECISIONS,
        }
    }
}

/// The default bound on F9-sampled states held between two drains. The Python env checks a
/// sampled state on the spot; this env queues a whole `RunManager` per hit
/// until `step` returns, so one call that drives a hung fight to the step
/// cap would otherwise hold ~`f9_rate × run_steps` clones (2,000 at the
/// defaults). Hits past the bound are counted `f9/dropped`, not kept; the
/// draw stream is unchanged.
pub const F9_QUEUE_MAX: usize = 32;

/// A practice capture (`Capture`): the state before a Move into an elite or
/// boss node.
#[derive(Clone, Debug)]
pub struct Capture {
    /// A `contracts.CELLS` name.
    pub cell: String,
    /// The run before the Move.
    pub snap: RunManager,
    /// The Move action.
    pub action: Value,
    /// `run.steps` before the Move's increment.
    pub steps: u64,
    /// Rows stored so far.
    pub rows: usize,
    /// The action log so far.
    pub idx: Vec<i64>,
}

/// The segment's end (`E2ETerminal`, minus what the host adds: the
/// fingerprint, the segment, the F9 counts).
#[derive(Clone, Debug)]
pub struct TrainTerminal {
    /// `won`, `lost`, `capped`, `bootstrap` or `error`.
    pub outcome: &'static str,
    /// The run was won.
    pub won: bool,
    /// The cap that ended a capped segment.
    pub cap: Option<&'static str>,
    /// The reward assigned at the end (to the last row).
    pub reward: f64,
    /// Sum of the rows' rewards.
    pub total_reward: f64,
    /// Reward accrued before the first row.
    pub unassigned_reward: f64,
    /// Stored rows.
    pub rows: usize,
    /// All steps, forced included.
    pub steps: u64,
    /// `rm.total_floor()` at the end.
    pub floor: i64,
    /// `rm.act_index` at the end.
    pub act: i64,
    /// Bosses beaten before the run's end.
    pub bosses: i64,
    /// The run reached the act-3 second boss.
    pub reached_boss2: bool,
    /// `_final_counts()`: the counters plus `forced/<type>`.
    pub counts: BTreeMap<String, i64>,
    /// The error's type (`SchemaError`, `RuntimeError`, `NoActions`, ...).
    pub etype: Option<String>,
    /// The error's message.
    pub error: Option<String>,
}

/// What `reset` / `step` returns.
#[derive(Debug)]
pub enum TrainStep {
    /// A stored row: the reward assigned at its emit, whether it is the
    /// segment's bootstrap row. The decision (the tokenizer's input) is
    /// [`TrainEnv::decision`].
    Decision {
        /// `E2EDecision.reward`.
        reward: f64,
        /// `E2EDecision.bootstrap`.
        bootstrap: bool,
    },
    /// The segment ended.
    Terminal(TrainTerminal),
}

/// An error with the Python exception type the env would count
/// (`error/<type>`).
#[derive(Debug)]
struct Fail {
    etype: &'static str,
    msg: String,
}

type F<T> = Result<T, Fail>;

/// Why `reset` / `step` raised instead of returning a row or a terminal.
#[derive(Debug)]
pub enum StepError {
    /// A misuse: no pending decision, a bootstrap row, an index out of
    /// range (the Python raises `RuntimeError` / `IndexError`).
    Usage(String),
    /// An error the Python env propagates rather than ending the segment
    /// on: any error while `reset` advances to the first row (`run.reset()`
    /// runs outside `_drive`'s handler), and a build mismatch anywhere
    /// (`ends_segment` is false for `GameVersionMismatch`).
    Raise {
        /// The Python exception type.
        etype: &'static str,
        /// Its message.
        msg: String,
    },
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(m) => f.write_str(m),
            Self::Raise { etype, msg } => write!(f, "{etype}: {msg}"),
        }
    }
}

impl Fail {
    /// `ends_segment(ex)` is false: the env raises it.
    fn propagates(&self) -> bool {
        self.etype == "GameVersionMismatch"
    }

    fn raise(self) -> StepError {
        StepError::Raise {
            etype: self.etype,
            msg: self.msg,
        }
    }

    fn runtime(msg: impl Into<String>) -> Self {
        Self {
            etype: "RuntimeError",
            msg: msg.into(),
        }
    }
}

/// A helper's `String` error: `SchemaError: ...` is the schema's.
impl From<String> for Fail {
    fn from(msg: String) -> Self {
        let etype = if msg.starts_with("SchemaError") {
            "SchemaError"
        } else {
            "RuntimeError"
        };
        Self { etype, msg }
    }
}

/// The run stepper's output (`RunDecision` / `CombatTurn` / `Terminal`).
enum RunOut {
    Decision {
        phase: &'static str,
        surface: Option<&'static str>,
        actions: Vec<Value>,
        obs: Value,
    },
    Combat,
    Terminal {
        outcome: &'static str,
        won: bool,
    },
}

/// `_ooc`'s result: a row or an end, or the run stepper's next output.
enum OocOut {
    Done(TrainStep),
    Next(RunOut),
}

#[derive(Clone, Debug)]
struct Fight {
    seq: usize,
    room: &'static str,
    at: (u8, Option<(i32, i32)>),
    ilog: IntentLog,
    round: i64,
    rows: usize,
}

/// The stored row awaiting its action: the decision itself (moved, not
/// copied: the host reads it through [`TrainEnv::decision`]).
#[derive(Debug)]
struct Pending {
    bootstrap: bool,
    d: Decision,
}

/// One training segment over its own run.
#[derive(Debug)]
pub struct TrainEnv {
    rm: RunManager,
    seed: i64,
    asc: i64,
    spec: SegmentSpec,
    // _RunEnv
    steps: u64,
    grant_picked: i64,
    forced: BTreeMap<String, i64>,
    run_pending: Option<Vec<Value>>,
    // E2EEnv
    rows: usize,
    rewards: Vec<f64>,
    total_reward: f64,
    unassigned_reward: f64,
    pending_r: f64,
    max_floor: i64,
    bosses: i64,
    reached_boss2: bool,
    act3_bosses: i64,
    fight: Option<Fight>,
    fights: usize,
    log_idx: Vec<i64>,
    counts: BTreeMap<String, i64>,
    pending: Option<Pending>,
    done: bool,
    terminal: Option<TrainTerminal>,
    // hooks
    f9_rng: PyRandom,
    f9_states: Vec<RunManager>,
    captures: Vec<Capture>,
    /// `reset` advanced to its first output: a panic from here on ends the
    /// segment (`_drive`'s handler) instead of propagating.
    started: bool,
}

fn bump(c: &mut BTreeMap<String, i64>, k: impl Into<String>, n: i64) {
    *c.entry(k.into()).or_insert(0) += n;
}

impl TrainEnv {
    /// The env over `rm` (a fresh campaign run, or a practice start).
    /// `seed` is the run's seed (`E2EEnv.seed`).
    #[must_use]
    pub fn new(rm: RunManager, seed: i64, spec: SegmentSpec) -> Self {
        ensure_engine_installed();
        let max_floor = i64::from(rm.run_state.total_floor);
        let asc = i64::from(rm.run_state.ascension_level);
        let f9_rng = PyRandom::new(spec.f9_seed);
        Self {
            rm,
            seed,
            asc,
            spec,
            steps: 0,
            grant_picked: 0,
            forced: BTreeMap::new(),
            run_pending: None,
            rows: 0,
            rewards: Vec::new(),
            total_reward: 0.0,
            unassigned_reward: 0.0,
            pending_r: 0.0,
            max_floor,
            bosses: 0,
            reached_boss2: false,
            act3_bosses: 0,
            fight: None,
            fights: 0,
            log_idx: Vec::new(),
            counts: BTreeMap::new(),
            pending: None,
            done: false,
            terminal: None,
            f9_rng,
            f9_states: Vec::new(),
            captures: Vec::new(),
            started: false,
        }
    }

    /// The stored row awaiting its action (the last [`TrainStep::Decision`]).
    #[must_use]
    pub fn decision(&self) -> Option<&Decision> {
        self.pending.as_ref().map(|p| &p.d)
    }

    /// The live run.
    #[must_use]
    pub const fn rm(&self) -> &RunManager {
        &self.rm
    }

    /// The env's seed.
    #[must_use]
    pub const fn seed(&self) -> i64 {
        self.seed
    }

    /// Stored rows so far.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// `reset` got past `run.reset()` (see [`StepError::Raise`]).
    #[must_use]
    pub const fn started(&self) -> bool {
        self.started
    }

    /// The segment ended.
    #[must_use]
    pub const fn done(&self) -> bool {
        self.done
    }

    /// The terminal, once the segment ended.
    #[must_use]
    pub const fn terminal(&self) -> Option<&TrainTerminal> {
        self.terminal.as_ref()
    }

    /// The action log (`log_idx`): the member index of every stored
    /// decision's applied action.
    #[must_use]
    pub fn log_idx(&self) -> &[i64] {
        &self.log_idx
    }

    /// The raw counters (`env.counts`, without `forced/<type>`).
    #[must_use]
    pub const fn counts(&self) -> &BTreeMap<String, i64> {
        &self.counts
    }

    /// The captures since the last drain, in order.
    pub fn drain_captures(&mut self) -> Vec<Capture> {
        std::mem::take(&mut self.captures)
    }

    /// The F9-sampled post-action states since the last drain, in order.
    pub fn drain_f9(&mut self) -> Vec<RunManager> {
        std::mem::take(&mut self.f9_states)
    }

    // ── public: reset / step ────────────────────────────────────────────

    /// `reset()`: advance to the first stored row.
    ///
    /// # Errors
    ///
    /// [`StepError::Raise`]: the advance to the run stepper's first output
    /// failed (`E2EEnv.reset` calls `run.reset()` outside `_drive`, so the
    /// Python raises and the worker's `MAX_START_FAILS` counts it), or a
    /// build mismatch.
    pub fn reset(&mut self) -> Result<TrainStep, StepError> {
        let out = self.advance().map_err(Fail::raise)?;
        self.started = true;
        self.drive(Ok(out))
    }

    /// `step(choice)`: apply merged class `choice` (its first member).
    ///
    /// # Errors
    ///
    /// [`StepError::Usage`]: no pending decision, a bootstrap row, or
    /// `choice` out of range (the Python raises these outside the env's
    /// error handling); [`StepError::Raise`]: a build mismatch.
    pub fn step(&mut self, choice: usize) -> Result<TrainStep, StepError> {
        let ai = {
            let p = self
                .pending
                .as_ref()
                .ok_or_else(|| StepError::Usage("step() without a pending decision".into()))?;
            p.d.members
                .get(choice)
                .ok_or_else(|| StepError::Usage(format!("class {choice} out of range")))?[0]
        };
        self.step_action(ai)
    }

    /// `step_action(ai)`: apply member `ai` of the pending decision.
    ///
    /// # Errors
    ///
    /// As [`Self::step`].
    pub fn step_action(&mut self, ai: usize) -> Result<TrainStep, StepError> {
        let p = self
            .pending
            .as_ref()
            .ok_or_else(|| StepError::Usage("step_action() without a pending decision".into()))?;
        if p.bootstrap {
            return Err(StepError::Usage(
                "a bootstrap row takes no action: the segment has ended".into(),
            ));
        }
        if ai >= p.d.actions.len() {
            return Err(StepError::Usage(format!(
                "action {ai} out of range 0..{}",
                p.d.actions.len().saturating_sub(1)
            )));
        }
        let p = self.pending.take().expect("checked above");
        self.log_idx.push(i64::try_from(ai).unwrap_or(i64::MAX));
        let out = if p.d.is_combat() {
            combat_to_rm(&p.d.actions[ai])
                .map_err(Fail::from)
                .and_then(|a| self.combat_apply(&a))
                .map(|()| RunOut::Combat)
        } else {
            self.run_step(ai)
        };
        self.drive(out)
    }

    /// End the segment as an error raised outside the env's own checks (a
    /// Rust panic the host caught mid-step: `etype` `PanicException`), as
    /// `_drive`'s `except` does.
    pub fn fail(&mut self, etype: &'static str, msg: String) -> TrainStep {
        self.pending = None;
        self.error(Fail { etype, msg })
    }

    // ── the run stepper (`_RunEnv`) ─────────────────────────────────────

    /// `_RunEnv._terminal`.
    fn run_terminal(&self, outcome: &'static str) -> RunOut {
        let won = self.rm.run_state.player_won;
        RunOut::Terminal {
            outcome: if won { "won" } else { outcome },
            won,
        }
    }

    /// `_RunEnv._advance("net")`.
    fn advance(&mut self) -> F<RunOut> {
        loop {
            if self.rm.is_over() {
                return Ok(self.run_terminal("lost"));
            }
            if self.steps >= self.spec.caps.run_steps {
                return Ok(self.run_terminal("max_steps"));
            }
            let phase = phase_str(self.rm.phase());
            if phase == "event" {
                sts2_content::events::dispatch::bind_pending_event_callback(&mut self.rm.run_state);
            }
            let actions = available(&self.rm);
            if actions.is_empty() {
                return Ok(self.run_terminal("no_actions"));
            }
            let actions = mask_rejectable(&self.rm, phase, actions);
            if is_combat(&actions) {
                return Ok(RunOut::Combat);
            }
            if let Some(a) = actions
                .iter()
                .find(|a| !ENCODABLE.contains(&action_name(a)))
            {
                return Err(Fail::runtime(format!(
                    "unencodable action kind {:?} (phase {phase:?})",
                    action_name(a)
                )));
            }
            let surface = classify_surface(phase, &actions);
            if actions.len() == 1 {
                let name = self.forced_name(&actions);
                bump(&mut self.forced, name, 1);
                let a = actions[0].clone();
                self.take(&a)?;
                self.steps += 1;
                continue;
            }
            let obs = observe(&mut self.rm, phase, surface, self.seed, self.asc);
            self.run_pending = Some(actions.clone());
            return Ok(RunOut::Decision {
                phase,
                surface,
                actions,
                obs,
            });
        }
    }

    /// `_forced_name`: the decision type of a single-option step.
    fn forced_name(&self, actions: &[Value]) -> String {
        match self.classify(actions, &Value::Null) {
            Ok((d, _)) => ct::dtype_name(d).to_owned(),
            Err(_) => format!(
                "unknown_kind:{}",
                actions.first().map_or("", |a| action_name(a))
            ),
        }
    }

    /// env.py `classify_ooc(actions, rm, obs)`.
    fn classify(&self, actions: &[Value], obs: &Value) -> Result<(u8, Vec<u8>), String> {
        let has = |n: &str| actions.iter().any(|a| action_name(a) == n);
        let event = if has("event_choice") {
            current_event_name(&self.rm)
        } else {
            None
        };
        let grant_n = has("grant_deck_card").then(|| emit_rm::card_grant_options(&self.rm).len());
        let bundles =
            get(obs, "card_reward_bundles").is_some_and(truthy) && has("pick_card_reward");
        ct::classify_ooc(actions, event.as_deref(), bundles, grant_n).map_err(schema)
    }

    /// `_RunEnv._take`: grant bookkeeping, the capture hook on a Move, the
    /// F9 hook.
    fn take(&mut self, action: &Value) -> F<&'static str> {
        let name = action_name(action);
        if name == "grant_deck_card" {
            let idx = get(action, "idx")
                .filter(|v| truthy(v))
                .and_then(as_int)
                .unwrap_or(0);
            let n = i64::try_from(emit_rm::card_grant_options(&self.rm).len()).unwrap_or(0);
            if idx < n {
                self.grant_picked += 1;
            }
        } else {
            self.grant_picked = 0;
        }
        let res = if self.spec.capture && name == "move" {
            let before = self.rm.clone();
            let res = take_run_action(&mut self.rm, action)?;
            self.on_move(before, action);
            res
        } else {
            take_run_action(&mut self.rm, action)?
        };
        self.f9_step();
        Ok(res)
    }

    /// `_RunEnv.step(idx)`: the pending OOC actions' `idx`, strictly.
    fn run_step(&mut self, idx: usize) -> F<RunOut> {
        let actions = self
            .run_pending
            .take()
            .ok_or_else(|| Fail::runtime("step() without a pending decision"))?;
        let a = actions
            .get(idx)
            .ok_or_else(|| Fail::runtime(format!("candidate idx {idx} out of range")))?
            .clone();
        let res = self.take(&a)?;
        if matches!(
            res,
            "shop_purchase_failed" | "shop_remove_card_failed" | "invalid_for_phase"
        ) {
            return Err(Fail::runtime(format!("engine rejected action {a}: {res}")));
        }
        self.steps += 1;
        self.advance()
    }

    // ── the drive loop (`E2EEnv`) ───────────────────────────────────────

    /// `_drive(out)`, with its error handling: an error ends the segment,
    /// except one `ends_segment` lets through.
    fn drive(&mut self, out: F<RunOut>) -> Result<TrainStep, StepError> {
        match out.and_then(|o| self.drive_inner(o)) {
            Ok(s) => Ok(s),
            Err(f) if f.propagates() => {
                self.pending = None;
                Err(f.raise())
            }
            Err(f) => Ok(self.error(f)),
        }
    }

    fn drive_inner(&mut self, mut out: RunOut) -> F<TrainStep> {
        loop {
            match out {
                RunOut::Terminal { outcome, won } => return Ok(self.terminal_(outcome, won, None)),
                RunOut::Decision {
                    phase,
                    surface,
                    actions,
                    obs,
                } => match self.ooc(phase, surface, actions, obs)? {
                    OocOut::Done(s) => return Ok(s),
                    OocOut::Next(o) => out = o,
                },
                RunOut::Combat => {
                    if let Some(s) = self.combat()? {
                        return Ok(s);
                    }
                    out = self.advance()?;
                }
            }
        }
    }

    /// `_combat`: forced combat steps until a stored row (`Some`), or until
    /// the run leaves the fight (`None`: resume the run stepper).
    #[allow(clippy::too_many_lines)]
    fn combat(&mut self) -> F<Option<TrainStep>> {
        loop {
            if self.rm.is_over() {
                if self.fight.is_some() {
                    let won = self.rm.run_state.player_won;
                    self.end_fight(won);
                }
                return Ok(None);
            }
            let run_acts = available(&self.rm);
            if !is_combat(&run_acts) {
                if self.fight.is_some() && self.rm.phase() != Phase::Combat {
                    self.end_fight(true);
                }
                return Ok(None);
            }
            if self.steps >= self.spec.caps.run_steps {
                return Ok(Some(self.cap("steps")));
            }
            let cs = self.rm.combat.as_ref().ok_or_else(|| {
                Fail::runtime("the run offers combat actions but has no combat state")
            })?;
            let legal = match sts2_wire::emit::legal_actions(cs) {
                Value::Array(a) => a,
                _ => Vec::new(),
            };
            if legal.is_empty() {
                return Err(Fail::runtime(
                    "the run offers combat actions but the fight has none",
                ));
            }
            if self.fight.is_none() {
                self.start_fight()?;
            }
            let at = (self.rm.run_state.act_index, emit_rm::map_coord(&self.rm));
            if self.fight.as_ref().is_some_and(|f| f.at != at) {
                self.end_fight(true);
                self.start_fight()?;
            }
            let cs = self.rm.combat.as_ref().expect("checked above");
            let bv = snapshot::belief_view(cs);
            let d_bv = snapshot::state_dict(&bv)?;
            let rnd = round_of(&d_bv);
            if self.fight.as_ref().is_some_and(|f| rnd < f.round) {
                bump(&mut self.counts, "round_reset", 1);
                self.end_fight(true);
                self.start_fight()?;
            }
            let f = self.fight.as_mut().expect("a fight is tracked");
            f.round = rnd;
            if rnd > self.spec.caps.fight_turns {
                return Ok(Some(self.cap("turns")));
            }
            if legal.len() == 1 && f.ilog.last_round == Some(rnd) {
                bump(&mut self.counts, "forced/combat", 1);
                let a = combat_to_rm(&legal[0])?;
                self.combat_apply(&a)?;
                continue;
            }
            let room = f.room;
            let snap = f.ilog.observe(snapshot_of(&bv, &d_bv, room));
            if legal.len() == 1 {
                bump(&mut self.counts, "forced/combat", 1);
                let a = combat_to_rm(&legal[0])?;
                self.combat_apply(&a)?;
                continue;
            }
            let cs = self.rm.combat.as_ref().expect("checked above");
            let cc = match merged_classes(cs, &legal, &bv, &d_bv) {
                Ok(cc) => cc,
                Err(e) => {
                    let f = Fail::from(e);
                    if f.etype == "SchemaError" {
                        bump(&mut self.counts, "unknown", 1);
                    }
                    return Err(f);
                }
            };
            if cc.members.len() == 1 {
                bump(&mut self.counts, "forced/combat_merged", 1);
                let a = combat_to_rm(&legal[cc.members[0][0]])?;
                self.combat_apply(&a)?;
                continue;
            }
            self.check_caps(legal.len(), cc.members.len(), cc.dtype)?;
            let (seq, fight_rows) = {
                let f = self.fight.as_ref().expect("a fight is tracked");
                (f.seq, f.rows)
            };
            if fight_rows >= self.spec.caps.fight_decisions {
                return Ok(Some(self.cap("fight_decisions")));
            }
            if self.rows >= self.spec.caps.run_decisions {
                return Ok(Some(self.cap("run_decisions")));
            }
            if legal.len() > cc.members.len() {
                bump(&mut self.counts, "merged_rows", 1);
                bump(
                    &mut self.counts,
                    "merged_actions",
                    i64::try_from(legal.len() - cc.members.len()).unwrap_or(0),
                );
            }
            let sel = if matches!(cc.dtype, dt::COMBAT_CHOICE | dt::CARD_GRANT) {
                ct::combat_selection(sts2_wire::emit::pending_choice(&bv).as_ref(), &cc.kinds)
            } else {
                Selection::default()
            };
            let d = Decision {
                row: self.rows,
                kind: DecisionKind::Combat,
                dtype: cc.dtype,
                kinds: cc.kinds,
                actions: legal,
                obs_actions: cc.obs_actions,
                classes: cc.classes,
                members: cc.members,
                keys: cc.keys,
                sel,
                event_name: None,
                act: self.rm.run_state.act_index,
                floor: self.rm.run_state.total_floor,
                combat: Some(CombatView {
                    encounter_id: emit_rm::combat_encounter_id(&self.rm),
                    bv,
                    d_bv,
                    snap,
                    room,
                    fight: seq,
                    turn: rnd,
                    fight_row: fight_rows,
                }),
                ooc: None,
            };
            let s = self.emit(d, true, false);
            if let Some(f) = self.fight.as_mut() {
                f.rows += 1;
            }
            return Ok(Some(s));
        }
    }

    /// `_combat_apply`: one combat action with the env's bookkeeping.
    fn combat_apply(&mut self, action: &Value) -> F<()> {
        take_run_action(&mut self.rm, action)?;
        self.steps += 1;
        self.f9_step();
        if self.rm.is_over() {
            if self.fight.is_some() {
                let won = self.rm.run_state.player_won;
                self.end_fight(won);
            }
        } else if self.rm.phase() != Phase::Combat && self.fight.is_some() {
            self.end_fight(true);
        }
        Ok(())
    }

    /// `_ooc`: a stored OOC row, or the lone merged class applied.
    fn ooc(
        &mut self,
        phase: &'static str,
        surface: Option<&'static str>,
        actions: Vec<Value>,
        obs: Value,
    ) -> F<OocOut> {
        let classified = self.classify(&actions, &obs).and_then(|(dtype, kinds)| {
            let payloads = ct::ooc_payloads(&obs, &actions, &kinds).map_err(schema)?;
            let keys = ct::ooc_class_keys(&kinds, &payloads).map_err(schema)?;
            Ok((dtype, kinds, keys))
        });
        let (dtype, kinds, keys) = match classified {
            Ok(x) => x,
            Err(e) => {
                bump(&mut self.counts, "unknown", 1);
                return Err(Fail::from(e));
            }
        };
        let (classes, members) = ct::classes_from_keys(&keys);
        let name = ct::dtype_name(dtype);
        if members.len() == 1 {
            bump(&mut self.counts, format!("forced/{name}_merged"), 1);
            return Ok(OocOut::Next(self.run_step(members[0][0])?));
        }
        self.check_caps(actions.len(), members.len(), dtype)?;
        let fine = fine_surface(&actions);
        if fine == "unknown" {
            bump(&mut self.counts, "fine_unknown", 1);
        }
        let sel = if dtype == dt::CARD_GRANT {
            Selection {
                min: self
                    .rm
                    .run_state
                    .pending_card_grants
                    .first()
                    .map_or(0, |g| i64::try_from(g.min_pick).unwrap_or(i64::MAX)),
                max: i64::try_from(emit_rm::card_grant_options(&self.rm).len()).unwrap_or(0),
                picked: self.grant_picked,
                can_skip: kinds.contains(&ak::GRANT_CONFIRM),
            }
        } else {
            Selection::default()
        };
        let event_name = if matches!(dtype, dt::EVENT | dt::NEOW) {
            current_event_name(&self.rm)
        } else {
            None
        };
        let n_actions = actions.len();
        let n_members = members.len();
        let d = Decision {
            row: self.rows,
            kind: DecisionKind::Ooc,
            dtype,
            kinds,
            obs_actions: actions.clone(),
            actions,
            classes,
            members,
            keys,
            sel,
            event_name,
            act: self.rm.run_state.act_index,
            floor: self.rm.run_state.total_floor,
            combat: None,
            ooc: Some(OocView {
                phase,
                surface,
                run_obs: obs,
            }),
        };
        if self.spec.first_ooc {
            bump(&mut self.counts, "rows/bootstrap", 1);
            return Ok(OocOut::Done(self.emit_bootstrap(d)));
        }
        if self.rows >= self.spec.caps.run_decisions {
            return Ok(OocOut::Done(self.cap("run_decisions")));
        }
        bump(&mut self.counts, format!("fine/{fine}"), 1);
        if n_actions > n_members {
            bump(&mut self.counts, "merged_rows", 1);
            bump(
                &mut self.counts,
                "merged_actions",
                i64::try_from(n_actions - n_members).unwrap_or(0),
            );
        }
        Ok(OocOut::Done(self.emit(d, true, false)))
    }

    /// `check_caps` with the `assert/<what>` counters.
    fn check_caps(&mut self, n_members: usize, n_classes: usize, dtype: u8) -> F<()> {
        let mut over = Vec::new();
        if n_members > ct::MAX_MEMBERS {
            over.push("members");
        }
        if n_classes > ct::MAX_OPTIONS {
            over.push("classes");
        }
        if over.is_empty() {
            return Ok(());
        }
        for w in &over {
            bump(&mut self.counts, format!("assert/{w}"), 1);
        }
        Err(Fail {
            etype: "SchemaError",
            msg: format!(
                "{}: {n_members} members / {n_classes} classes exceed {} / {}",
                ct::dtype_name(dtype),
                ct::MAX_MEMBERS,
                ct::MAX_OPTIONS
            ),
        })
    }

    // ── rows and reward ─────────────────────────────────────────────────

    fn accrue_floors(&mut self) {
        let f = i64::from(self.rm.run_state.total_floor);
        if f > self.max_floor {
            #[allow(clippy::cast_precision_loss)]
            let n = (f - self.max_floor) as f64;
            self.pending_r += R_FLOOR * n;
            self.max_floor = f;
        }
    }

    fn assign_pending(&mut self) -> f64 {
        self.accrue_floors();
        let r = self.pending_r;
        self.pending_r = 0.0;
        if self.rows > 0 {
            if let Some(last) = self.rewards.last_mut() {
                *last += r;
            }
            self.total_reward += r;
        } else {
            self.unassigned_reward += r;
        }
        r
    }

    /// `_emit`: a stored row.
    fn emit(&mut self, d: Decision, count: bool, bootstrap: bool) -> TrainStep {
        let r = self.assign_pending();
        self.rewards.push(0.0);
        self.rows += 1;
        if count {
            bump(
                &mut self.counts,
                format!("rows/{}", ct::dtype_name(d.dtype)),
                1,
            );
        }
        self.pending = Some(Pending { bootstrap, d });
        TrainStep::Decision {
            reward: r,
            bootstrap,
        }
    }

    /// `_emit_bootstrap`: the segment's last row, value-only.
    fn emit_bootstrap(&mut self, d: Decision) -> TrainStep {
        let s = self.emit(d, false, true);
        self.done = true;
        let t = self.make_terminal("bootstrap", false, None, 0.0, None);
        self.terminal = Some(t);
        s
    }

    fn final_counts(&self) -> BTreeMap<String, i64> {
        let mut c = self.counts.clone();
        for (k, v) in &self.forced {
            bump(&mut c, format!("forced/{k}"), *v);
        }
        c
    }

    fn make_terminal(
        &self,
        outcome: &'static str,
        won: bool,
        cap: Option<&'static str>,
        reward: f64,
        err: Option<(&'static str, String)>,
    ) -> TrainTerminal {
        let (etype, error) = match err {
            Some((t, m)) => (Some(t.to_owned()), Some(m)),
            None => (None, None),
        };
        TrainTerminal {
            outcome,
            won,
            cap,
            reward,
            total_reward: self.total_reward,
            unassigned_reward: self.unassigned_reward,
            rows: self.rows,
            steps: self.steps,
            floor: i64::from(self.rm.run_state.total_floor),
            act: i64::from(self.rm.run_state.act_index),
            bosses: self.bosses,
            reached_boss2: self.reached_boss2,
            counts: self.final_counts(),
            etype,
            error,
        }
    }

    /// `_terminal`.
    fn terminal_(
        &mut self,
        outcome: &'static str,
        won: bool,
        err: Option<(&'static str, String)>,
    ) -> TrainStep {
        if outcome == "no_actions" {
            let msg = format!(
                "no legal action at act {}, floor {}, phase {:?}, step {}",
                self.rm.run_state.act_index,
                self.rm.run_state.total_floor,
                phase_str(self.rm.phase()),
                self.steps
            );
            return self.error(Fail {
                etype: "NoActions",
                msg,
            });
        }
        if self.fight.is_some() {
            let pw = self.rm.run_state.player_won;
            self.end_fight(won && pw);
        }
        if outcome == "max_steps" {
            return self.cap("steps");
        }
        if won {
            self.pending_r += R_WIN;
        }
        let r = self.assign_pending();
        self.done = true;
        let out = if won {
            "won"
        } else if outcome == "error" {
            "error"
        } else {
            "lost"
        };
        bump(&mut self.counts, format!("outcome/{out}"), 1);
        let t = self.make_terminal(out, won, None, r, err);
        self.terminal = Some(t.clone());
        TrainStep::Terminal(t)
    }

    /// `_error`: the segment ends as an error, `error/<type>` counted.
    fn error(&mut self, f: Fail) -> TrainStep {
        bump(&mut self.counts, "error", 1);
        bump(&mut self.counts, format!("error/{}", f.etype), 1);
        self.fight = None;
        self.terminal_("error", false, Some((f.etype, f.msg)))
    }

    /// `_cap`: capped, no boss reward.
    fn cap(&mut self, reason: &'static str) -> TrainStep {
        self.fight = None;
        let r = self.assign_pending();
        self.done = true;
        bump(&mut self.counts, format!("cap/{reason}"), 1);
        bump(&mut self.counts, "outcome/capped", 1);
        let t = self.make_terminal("capped", false, Some(reason), r, None);
        self.terminal = Some(t.clone());
        TrainStep::Terminal(t)
    }

    // ── fights ──────────────────────────────────────────────────────────

    /// `_start_fight`, with `factory.check_fight` (plan §3.1).
    fn start_fight(&mut self) -> F<()> {
        if let Some(want) = self.spec.game_version {
            let got = self.rm.combat.as_ref().map(|cs| cs.game_version);
            if got != Some(want) {
                return Err(Fail {
                    etype: "GameVersionMismatch",
                    msg: format!(
                        "combat state simulates {:?}, expected {:?}",
                        got.map(|g| g.to_string()),
                        want.to_string()
                    ),
                });
            }
        }
        let room = fight_room(&self.rm);
        let act = self.rm.run_state.act_index;
        self.fight = Some(Fight {
            seq: self.fights,
            room,
            at: (act, emit_rm::map_coord(&self.rm)),
            ilog: IntentLog::new(),
            round: 0,
            rows: 0,
        });
        self.fights += 1;
        bump(&mut self.counts, format!("fights/{room}"), 1);
        if room == "boss" && act >= 3 {
            self.act3_bosses += 1;
            if self.act3_bosses >= 2 {
                self.reached_boss2 = true;
            }
        }
        Ok(())
    }

    fn end_fight(&mut self, won: bool) {
        let Some(f) = self.fight.take() else {
            return;
        };
        if won {
            bump(&mut self.counts, format!("fights_won/{}", f.room), 1);
        }
        if won && f.room == "boss" && !(self.rm.is_over() && self.rm.run_state.player_won) {
            self.pending_r += R_BOSS;
            self.bosses += 1;
        }
    }

    // ── practice capture and F9 ─────────────────────────────────────────

    /// `_on_move` (only with `spec.capture`).
    fn on_move(&mut self, before: RunManager, action: &Value) {
        let room = get(&emit_rm::current_room(&self.rm), "room_type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let act = self.rm.run_state.act_index;
        if act > 3 {
            bump(&mut self.counts, "capture/act4", 1);
            return;
        }
        let cell = match room.as_deref() {
            Some("elite") => format!("ELITE_ACT{act}"),
            Some("boss") => {
                if act >= 3 {
                    let m = emit_rm::full_map(&self.rm);
                    let coord = |v: Option<&Value>| {
                        v.filter(|x| truthy(x)).map(|x| {
                            (
                                get(x, "col").and_then(as_int),
                                get(x, "row").and_then(as_int),
                            )
                        })
                    };
                    let sb = coord(get(&m, "second_boss"));
                    let cur = coord(get(&m, "current_coord"));
                    if sb.is_some() && cur.is_some() && sb == cur {
                        bump(&mut self.counts, "capture/boss2_skipped", 1);
                        return;
                    }
                }
                if act == 3 {
                    "BOSS_ACT3_DOUBLE".to_owned()
                } else {
                    format!("BOSS_ACT{act}")
                }
            }
            _ => return,
        };
        bump(&mut self.counts, format!("capture/{cell}"), 1);
        self.captures.push(Capture {
            cell,
            snap: before,
            action: action.clone(),
            steps: self.steps,
            rows: self.rows,
            idx: self.log_idx.clone(),
        });
    }

    /// `_f9_step`: one draw per applied action; a hit hands the state over.
    fn f9_step(&mut self) {
        if self.f9_rng.random() < self.spec.f9_rate {
            if self.f9_states.len() < self.spec.f9_queue_max {
                self.f9_states.push(self.rm.clone());
            } else {
                bump(&mut self.counts, "f9/dropped", 1);
            }
        }
    }
}

/// `encoder.fine_surface` (the table lives in `sts2-vocab`, shared with
/// `sts2-runai`'s encoder).
fn fine_surface(actions: &[Value]) -> &'static str {
    sts2_vocab::fine_surface(actions.iter().map(action_name))
}

pub use sts2_core::py_random::PyRandom;
