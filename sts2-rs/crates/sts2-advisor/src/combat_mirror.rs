//! The combat mirror: one live engine state per fight, re-synchronized from
//! the game on every decision (GH #1426 step 1).
//!
//! Port of the `CombatMirror` class in `sts2-pycli/python/combat_mirror.py`,
//! everything but the search: seed / fork / drop, the reconcile (parked
//! verify, the in-flight skips, the rebuild + hidden-state transplant, the
//! OBSERVE reseed and its salvage), committing the world's actions, and the
//! choice-screen path (locator resolution, grading, injection, answering).
//! The reasons behind each rule moved here with it; the Python class keeps
//! the public contract in its docstrings, and its tests hold this port to it.
//!
//! No I/O, like the rest of the crate. A divergence builds the forensic
//! report and queues it ([`CombatMirror::take_divergences`]); the host writes
//! it to the log and calls its own hook. Under [`Policy::Strict`] the call
//! that diverged also returns [`MirrorError::Diverged`]; under
//! [`Policy::Observe`] the mirror is tainted and dropped (and, inside a
//! reconcile, re-seeded from the world) instead.

use std::collections::BTreeSet;

use serde_json::{json, Map, Value};
use sts2_core::CombatState;

use crate::mirror::{self, Desync};
use crate::mirror_state::{self, Carry};
use crate::py::{self, get, list, obj, truthy};
use crate::search::RecommendConfig;

/// What a divergence does: `Strict` raises, `Observe` records, drops the
/// mirror and taints the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Strict,
    Observe,
}

impl Policy {
    /// The `MirrorPolicy` value.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Observe => "observe",
        }
    }

    /// Parse a `MirrorPolicy` value.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "strict" => Some(Self::Strict),
            "observe" => Some(Self::Observe),
            _ => None,
        }
    }
}

/// `MirrorStats`: the per-mirror counters (see the Python class for what
/// each one measures).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct Stats {
    pub decisions: u64,
    pub reconciles: u64,
    pub desyncs: u64,
    pub desync_fields: u64,
    pub choice_mismatches: u64,
    pub reshuffle_skips: u64,
    pub unreported_fields: u64,
    pub injections: u64,
    pub unparked_injections: u64,
    pub cursor_drifts: u64,
    pub parked_verifies: u64,
    pub in_flight_skips: u64,
    pub choice_in_flight_skips: u64,
    pub teammate_adoptions: u64,
    pub searches: u64,
    pub seeds: u64,
    pub drops: u64,
}

/// `round(x, 3)`.
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// `a / b` per 1000, 0 when nothing ran.
#[allow(clippy::cast_precision_loss)]
fn per_1000(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        1000.0 * a as f64 / b as f64
    }
}

impl Stats {
    /// `MirrorStats.as_dict()`.
    #[must_use]
    pub fn as_value(&self) -> Value {
        json!({
            "decisions": self.decisions,
            "reconciles": self.reconciles,
            "desyncs": self.desyncs,
            "desync_fields": self.desync_fields,
            "choice_mismatches": self.choice_mismatches,
            "reshuffle_skips": self.reshuffle_skips,
            "unreported_fields": self.unreported_fields,
            "injections": self.injections,
            "unparked_injections": self.unparked_injections,
            "cursor_drifts": self.cursor_drifts,
            "parked_verifies": self.parked_verifies,
            "in_flight_skips": self.in_flight_skips,
            "choice_in_flight_skips": self.choice_in_flight_skips,
            "teammate_adoptions": self.teammate_adoptions,
            "searches": self.searches,
            "seeds": self.seeds,
            "drops": self.drops,
            "desync_rate_per_1000": round3(per_1000(self.desyncs, self.reconciles)),
            "choice_mismatch_rate_per_1000":
                round3(per_1000(self.choice_mismatches, self.injections)),
        })
    }
}

/// One recorded divergence: the message, the diverged paths (what the
/// host's `on_divergence` hook receives) and the forensic report (what it
/// appends to the divergence log, after adding its own timestamp).
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    pub message: String,
    pub paths: Vec<String>,
    pub report: Value,
}

/// Why a mirror call did not return normally.
#[derive(Debug, Clone)]
pub enum MirrorError {
    /// A divergence under [`Policy::Strict`] (`MirrorDivergence`).
    Diverged(Divergence),
    /// The call needs a mirrored fight and there is none.
    NoState,
    /// The wire did not parse (`state_from_dict`'s error).
    Wire(sts2_wire::WireError),
    /// The wire parsed into a state that breaks the engine's invariants.
    Invariant(String),
    /// A `ValueError` from the engine surface: an action that does not
    /// decode, an unknown choice kind / relic / power id, a state that does
    /// not serialize.
    Value(String),
    /// A `KeyError`: a search row the hand cannot map.
    Key(String),
}

impl std::fmt::Display for MirrorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Diverged(d) => f.write_str(&d.message),
            Self::NoState => f.write_str("mirror has no state: seed() a fight first"),
            Self::Wire(e) => write!(f, "{e}"),
            Self::Invariant(m) | Self::Value(m) | Self::Key(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MirrorError {}

/// A mirrored engine state plus what the Python `CombatState` wrapper keeps
/// beside it.
#[derive(Debug, Clone)]
pub struct Mirrored {
    pub state: CombatState,
    /// The wire's `rng.seed` (0 when absent); only the wrapper's repr reads it.
    pub seed: u64,
    /// Whether every Attack intent on the wire carried its damage.
    pub intent_damage_known: bool,
}

/// `state_from_dict(wire)`: parse, build, and check the invariants.
///
/// # Errors
///
/// The wire does not parse, or builds a state that breaks its invariants.
pub fn build_state(wire: &Value) -> Result<Mirrored, MirrorError> {
    let parsed = sts2_wire::parse_json(wire).map_err(MirrorError::Wire)?;
    let intent_damage_known = parsed.intent_damage_known;
    let mut state = sts2_wire::state_from_dict_impl(&parsed).map_err(MirrorError::Wire)?;
    mirror_state::engine_max_energy(&mut state);
    mirror_state::reprime_unshown_amounts(&mut state);
    mirror_state::rampage_growth_as_var(&mut state);
    let seed = get(get(wire, "rng"), "seed").as_u64().unwrap_or(0);
    state.assert_invariants().map_err(|why| {
        MirrorError::Invariant(format!(
            "$: state_from_dict produced inconsistent state: {why}"
        ))
    })?;
    Ok(Mirrored {
        state,
        seed,
        intent_damage_known,
    })
}

/// `state_to_dict(state)`.
fn to_value(state: &CombatState) -> Result<Value, MirrorError> {
    sts2_wire::emit::combat_state_to_value(state).map_err(|e| MirrorError::Value(e.to_string()))
}

/// Every option of a parked choice names the pile slot it sits in
/// (`pending_choice` emits `pile` / `index` for a card in a pile).
fn options_are_located(parked: &Value) -> bool {
    let cards = list(parked, "cards");
    !cards.is_empty()
        && cards
            .iter()
            .all(|c| !get(c, "pile").is_null() && !get(c, "index").is_null())
}

/// Does the producer carry the run's own dice (`rng.run_seed`)? Then the
/// world owns the RNG cursors.
fn world_owns_rng(wire: &Value) -> bool {
    get(wire, "rng")
        .as_object()
        .is_some_and(|r| r.contains_key("run_seed"))
}

/// `sts2_hint.adapter.TURN_HISTORY_KEYS`.
const TURN_HISTORY_KEYS: [&str; 2] = ["cards_exhausted_this_turn", "card_block_gains_this_turn"];

/// `_turn_history_disclosed`: the wire carries our turn's exhausts and card
/// block gains (GH #989).
fn turn_history_disclosed(wire: &Value) -> bool {
    let player = obj(wire, "player");
    TURN_HISTORY_KEYS.iter().all(|k| player.contains_key(*k))
}

/// `_teammate_reachable`: a field a co-op teammate's moves can change.
fn teammate_reachable(path: &str) -> bool {
    path.starts_with("$.encounter") || path.starts_with("$.allies")
}

/// The hidden state a rebuild from `wire` carries over.
fn carry_for(wire: &Value, enemy_slots: Vec<Option<usize>>) -> Carry {
    Carry {
        enemy_slots: Some(enemy_slots),
        rng: !world_owns_rng(wire),
        relic_counters: mirror::undisclosed_relic_counters(wire),
        power_counters: mirror::undisclosed_power_counters(wire),
        turn_history: !turn_history_disclosed(wire),
    }
}

/// `str(desync)`.
#[must_use]
pub fn desync_str(d: &Desync) -> String {
    format!(
        "{}: mirror={} world={}",
        d.path,
        py::repr(&d.mirror),
        py::repr(&d.world)
    )
}

/// The first six desyncs, `"; "`-joined, as a message lists them.
fn first_six(desyncs: &[Desync]) -> String {
    desyncs
        .iter()
        .take(6)
        .map(desync_str)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Desyncs as report rows.
fn desync_rows(desyncs: &[Desync]) -> Value {
    Value::Array(
        desyncs
            .iter()
            .map(|d| json!({"path": d.path, "mirror": d.mirror, "world": d.world}))
            .collect(),
    )
}

fn paths_of(desyncs: &[Desync]) -> Vec<String> {
    desyncs.iter().map(|d| d.path.clone()).collect()
}

/// Build a `Map` from `(key, value)` pairs.
/// Where each pinned card sits in the draw pile (`draw_pile_knowledge`).
fn pin_positions(state: &CombatState, pins: &[sts2_core::CardUid]) -> Vec<usize> {
    pins.iter()
        .filter_map(|uid| state.player.draw.iter().position(|d| d == uid))
        .collect()
}

fn map<const N: usize>(pairs: [(&str, Value); N]) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}

/// One live engine state per fight, re-synchronized from reality.
#[derive(Debug, Clone)]
pub struct CombatMirror {
    pub policy: Policy,
    /// Free-form tag written into every forensic record.
    pub label: String,
    pub stats: Stats,
    /// Set once the mirror recorded a divergence under OBSERVE.
    pub tainted: bool,
    /// Why the mirror was last dropped (`""` for a clean drop).
    pub drop_reason: String,
    /// How many other players share this fight (co-op).
    pub teammates: u32,
    /// The diff the last reconcile adopted under [`Self::teammates`].
    pub last_adopted: Vec<Desync>,
    /// Did any advance since the last reconcile consume the shuffle stream?
    pub crossed_shuffle: bool,
    /// The options a just-answered choice picked, armed for one reconcile.
    pub choice_in_flight: Option<Vec<Value>>,

    state: Option<Mirrored>,
    /// Bumped whenever the mirrored state changes or is replaced, so a host
    /// that hands out a copy of it knows when the copy went stale.
    state_generation: u64,
    /// Bumped whenever [`Self::wire`] is replaced by the call's wire.
    wire_generation: u64,
    last_dropped: Option<Mirrored>,
    last_dropped_parked: Option<Value>,
    parked: Option<Value>,
    last_unparked: Option<Value>,
    last_wire: Option<Value>,
    last_action: Option<Value>,
    last_result: Option<String>,
    last_drift: Vec<Desync>,
    last_injection: Option<Value>,
    open_injection: Option<Value>,
    just_seeded: bool,
    damage_trace_on: bool,
    last_damage_trace: Vec<Value>,
    damage_trace_dropped: u64,
    divergences: Vec<Divergence>,
    /// Enemy uids renamed by rebuilds since the last
    /// [`Self::take_enemy_remap`]: `(uid then, uid now)` (GH #1651).
    enemy_remap: Option<Vec<(u32, u32)>>,
}

/// `then → mid` followed by `mid → now`: `then → now`, for the enemies both
/// maps carry.
fn compose_remap(first: &[(u32, u32)], second: &[(u32, u32)]) -> Vec<(u32, u32)> {
    first
        .iter()
        .filter_map(|&(a, b)| second.iter().find(|&&(m, _)| m == b).map(|&(_, c)| (a, c)))
        .collect()
}

impl CombatMirror {
    /// The enemy uid renames since the last call (`(uid then, uid now)`), or
    /// `None` when no rebuild renamed one: an intent log keyed by uid follows
    /// its enemies through them (GH #1651).
    pub fn take_enemy_remap(&mut self) -> Option<Vec<(u32, u32)>> {
        self.enemy_remap.take()
    }

    fn note_enemy_remap(&mut self, pairs: &[(u32, u32)]) {
        let renamed = pairs.iter().any(|(a, b)| a != b);
        self.enemy_remap = match self.enemy_remap.take() {
            Some(prev) => Some(compose_remap(&prev, pairs)),
            None if renamed => Some(pairs.to_vec()),
            None => None,
        };
    }

    /// A mirror with no fight. `damage_trace` arms the engine's damage-source
    /// trace across the mirror's own steps (GH #368.1).
    #[must_use]
    pub fn new(policy: Policy, label: impl Into<String>, damage_trace: bool) -> Self {
        Self {
            policy,
            label: label.into(),
            stats: Stats::default(),
            tainted: false,
            drop_reason: String::new(),
            teammates: 0,
            last_adopted: Vec::new(),
            crossed_shuffle: false,
            choice_in_flight: None,
            state: None,
            state_generation: 0,
            wire_generation: 0,
            last_dropped: None,
            last_dropped_parked: None,
            parked: None,
            last_unparked: None,
            last_wire: None,
            last_action: None,
            last_result: None,
            last_drift: Vec::new(),
            last_injection: None,
            open_injection: None,
            just_seeded: false,
            damage_trace_on: damage_trace,
            last_damage_trace: Vec::new(),
            damage_trace_dropped: 0,
            divergences: Vec::new(),
            enemy_remap: None,
        }
    }

    // -- accessors ---------------------------------------------------------

    /// Is a fight mirrored?
    #[must_use]
    pub const fn alive(&self) -> bool {
        self.state.is_some()
    }

    /// The mirrored state, `None` before a seed / after a drop.
    #[must_use]
    pub const fn mirrored(&self) -> Option<&Mirrored> {
        self.state.as_ref()
    }

    /// See the field.
    #[must_use]
    pub const fn state_generation(&self) -> u64 {
        self.state_generation
    }

    /// See the field.
    #[must_use]
    pub const fn wire_generation(&self) -> u64 {
        self.wire_generation
    }

    /// The last wire reality handed the mirror.
    #[must_use]
    pub const fn wire(&self) -> Option<&Value> {
        self.last_wire.as_ref()
    }

    /// The choice the engine derived on its own, kept across a reconcile
    /// until an injection grades it.
    #[must_use]
    pub const fn parked_choice(&self) -> Option<&Value> {
        self.parked.as_ref()
    }

    /// The divergences recorded since the last call, oldest first.
    pub fn take_divergences(&mut self) -> Vec<Divergence> {
        std::mem::take(&mut self.divergences)
    }

    /// Count one search (the host runs it on a copy of the state).
    pub const fn note_search(&mut self) {
        self.stats.searches += 1;
    }

    /// The mirrored state, for a test that needs a state no wire can say.
    #[cfg(test)]
    pub(crate) fn state_mut_for_test(&mut self) -> Option<&mut CombatState> {
        self.mirrored_mut().ok()
    }

    fn mirrored_mut(&mut self) -> Result<&mut CombatState, MirrorError> {
        self.state_generation += 1;
        self.state
            .as_mut()
            .map(|m| &mut m.state)
            .ok_or(MirrorError::NoState)
    }

    fn set_wire(&mut self, wire: &Value) {
        self.last_wire = Some(wire.clone());
        self.wire_generation += 1;
    }

    // -- lifecycle ---------------------------------------------------------

    /// Start mirroring a fight from a full combat state.
    ///
    /// # Errors
    ///
    /// The wire does not build a state.
    pub fn seed(&mut self, wire: &Value) -> Result<(), MirrorError> {
        // The corpses the board reports removed (GH #1716); a board a
        // carried fight already gave its corpses is left as it is.
        let seeded = mirror::with_reported_corpses(wire, wire.clone());
        self.state = Some(build_state(&seeded)?);
        self.state_generation += 1;
        self.parked = None;
        self.crossed_shuffle = false;
        self.set_wire(wire);
        self.last_action = None;
        self.last_result = None;
        self.last_drift.clear();
        self.last_injection = None;
        self.open_injection = None;
        self.choice_in_flight = None;
        self.take_damage_trace();
        self.stats.seeds += 1;
        // Wire #1 lists enemies in spawn order and later wires in C#'s
        // slot-sorted order: the first reconcile cannot grade the re-sort.
        self.just_seeded = true;
        Ok(())
    }

    /// A detached copy of this fight, for work that must not touch it. Stats
    /// are not shared. `None` when nothing is mirrored.
    #[must_use]
    pub fn fork(&self, label: &str) -> Option<Self> {
        let state = self.state.clone()?;
        let mut twin = Self::new(
            self.policy,
            if label.is_empty() {
                format!("{}-fork", self.label)
            } else {
                label.to_owned()
            },
            self.damage_trace_on,
        );
        twin.state = Some(state);
        twin.parked.clone_from(&self.parked);
        twin.crossed_shuffle = self.crossed_shuffle;
        twin.last_wire.clone_from(&self.last_wire);
        twin.tainted = self.tainted;
        Some(twin)
    }

    /// Forget the current fight. Idempotent. The dropped state and parked
    /// choice are stashed for [`Self::reseed_after_observe`]'s salvage.
    pub fn drop_fight(&mut self, reason: &str) {
        if let Some(state) = self.state.take() {
            self.stats.drops += 1;
            self.last_dropped = Some(state);
            self.last_dropped_parked = self.parked.take();
            self.state_generation += 1;
        }
        self.parked = None;
        self.crossed_shuffle = false;
        self.open_injection = None;
        self.choice_in_flight = None;
        reason.clone_into(&mut self.drop_reason);
    }

    // -- reconcile ---------------------------------------------------------

    /// Re-impose every observable from reality; return the desyncs. Seeds
    /// the fight when nothing is mirrored yet.
    ///
    /// # Errors
    ///
    /// A divergence under STRICT; a wire that does not build a state.
    #[allow(clippy::too_many_lines)]
    pub fn reconcile(&mut self, wire: &Value) -> Result<Vec<Desync>, MirrorError> {
        self.stats.decisions += 1;
        let Some(carried) = self.state.as_ref().map(|m| &m.state) else {
            self.seed(wire)?;
            return Ok(Vec::new());
        };
        self.stats.reconciles += 1;
        self.last_adopted.clear();

        // A card whose OnPlay opened a screen has the REST of itself suspended
        // in `pending_play`: AfterCardPlayed, replays, disposal. No wire
        // dialect expresses that (the parser refuses a dict that even claims
        // to), so REPLACING the state here silently deletes the tail and the
        // pick resolves into nothing. Verify against the parked mirror and
        // keep it: reality still wins on every observable, because a mismatch
        // is reported exactly as it would be otherwise; what it does not do
        // is throw away the half-executed card in order to say so.
        //
        // Oracle Silent-16, traced: play -> [reconcile rebuilds, tail gone] ->
        // inject -> choose -> Nibbit block mirror 5 / world 1, with
        // `combat_targets` mirror 0 / world 1 (the world ran a tail that
        // targeted; the mirror ran nothing).
        //
        // A mid-ENEMY-turn suspension (KnowledgeDemon's CURSE_OF_KNOWLEDGE
        // screen) must be kept for the same reason: a rebuild cannot represent
        // the parked enemy-turn walk (the wire has no `pending_*_resume`), so
        // re-seeding would delete the rest of the enemy turn and the mirror
        // would sit one whole turn behind after the choice resolves (the
        // entire `$.round`/`$.phase` divergence family on the v27wr sweep).
        //
        // Any OTHER suspended turn-transition walk (Well-Laid Plans' retain
        // select inside the end-of-turn flush, a Stampede auto-play screen,
        // suspended turn setup / draw) is unreconstructable for the same
        // reason; a rebuild lands a full round behind (55 of the 230
        // player-side v28wr events: mirror round N-1, energy 0 vs the world's
        // refilled 3, plus run 1058's 40-event one-turn-lag fight).
        //
        // A DRINK suspended on its own card-select is the same class: C#
        // `PotionModel.OnUseWrapper` awaits `OnUse` (the choose-a-card screen
        // included) before firing `Hook.AfterPotionUsed`, so the engine parks
        // the deferred dispatch on `pending_potion_hook_resume`. Rebuilding
        // deleted both the deferred hook (ReptileTrinket's +3 temporary
        // Strength, BeltBuckle) and the parked choice. Deliberately NOT folded
        // into `walk_parked`: a drink happens inside the player's own play
        // phase, which both sides agree on, so `$.phase` / `$.side` stay
        // compared. Only a turn-transition walk earns that excusal.
        let has_play = carried.pending_play.is_some();
        let curse_parked = !has_play
            && sts2_wire::emit::pending_choice(carried)
                .is_some_and(|c| py::str_of(get(&c, "kind")) == "monster_curse_select");
        let walk_parked = !curse_parked && mirror_state::has_suspended_walk(carried);
        let potion_parked =
            !curse_parked && !walk_parked && carried.pending_potion_hook_resume.is_some();
        let world_rng = world_owns_rng(wire);
        if has_play || curse_parked || walk_parked || potion_parked {
            self.stats.parked_verifies += 1;
            let predicted = to_value(carried)?;
            // Round-trip the wire before diffing, exactly as the rebuild path
            // does. Comparing against the RAW wire diffs the engine dialect
            // against the producer's (`CLUMSY` vs `curse:Clumsy`, an intent's
            // `times` as None vs 1); none of those are divergences, and every
            // one fired as a false STRICT crash against the DLL.
            //
            // The reshuffle blind spot and the unreported-field skip apply for
            // the rebuild path's reasons, and more so: a card that reshuffles
            // the draw pile and THEN opens a screen lands here, and diffing
            // `$.deck.*` would be a false crash on the one path that cannot
            // recover by re-seeding. `crossed_shuffle` is NOT cleared here:
            // clearing it is the rebuild path's privilege, earned by ADOPTING
            // the world's piles; a parked verify keeps the mirror's own.
            if self.crossed_shuffle {
                self.stats.reshuffle_skips += 1;
            }
            let mut unreported = mirror::unreported_paths(wire);
            self.stats.unreported_fields += unreported.len() as u64;
            if curse_parked || walk_parked {
                // Representation, not divergence: C# reports a blocking
                // mid-transition screen as a player-side play phase, the
                // engine the suspended walk it is (EnemyPhase / a parked turn
                // boundary, same round). `$.round` stays compared: the walk
                // must not lag a whole turn.
                unreported.insert("$.phase".into());
                unreported.insert("$.side".into());
            }
            // The world read with the counters the board withholds (Void
            // Form's free plays): a false `$.deck.hand` cost diff here
            // reseeded, which deleted the suspended card's tail, its
            // AfterCardPlayed with it, and left Void Form's count one short
            // for the rest of the turn (Spire Bench seeds 180000322,
            // 180000516, 180000590, 180001456, 180000358, 180000884,
            // 180001345: True Grit / Discovery played past the free plays).
            let mut world_state = build_state(wire)?.state;
            mirror_state::carry_player_power_counters(
                &mut world_state,
                carried,
                &mirror::undisclosed_power_counters(wire),
            );
            let world = to_value(&world_state)?;
            let parked_desyncs = mirror::diff_observables(
                &predicted,
                &world,
                self.crossed_shuffle.then_some("$.deck."),
                &unreported,
            );
            // Same accounting as the rebuild path: a drift is recorded and
            // attached to this decision's report, never raised.
            if world_rng {
                self.last_drift = mirror::cursor_drift(&predicted, wire);
                self.stats.cursor_drifts += self.last_drift.len() as u64;
            } else {
                self.last_drift.clear();
            }
            if !parked_desyncs.is_empty() {
                self.stats.desyncs += 1;
                self.stats.desync_fields += parked_desyncs.len() as u64;
                let what = if curse_parked {
                    "the enemy turn was suspended on a curse screen"
                } else {
                    "a card was suspended mid-play"
                };
                self.diverge(
                    format!(
                        "{} deterministic observable(s) diverged while {what}: {}",
                        parked_desyncs.len(),
                        first_six(&parked_desyncs)
                    ),
                    "desync",
                    Some(wire.clone()),
                    paths_of(&parked_desyncs),
                    Some(predicted),
                    map([
                        ("mid_play", Value::Bool(!curse_parked)),
                        ("curse_parked", Value::Bool(curse_parked)),
                        ("desyncs", desync_rows(&parked_desyncs)),
                    ]),
                )?;
                // Reseed, then report, as the rebuild path does. Returning the
                // reseed's nothing instead threw the report away (live run
                // 001-ironclad-a8-0824-2138 crashed on it).
                self.reseed_after_observe(wire)?;
                return Ok(parked_desyncs);
            }
            self.set_wire(wire);
            return Ok(Vec::new());
        }

        // World-in-flight skip. The LIVE producer can snapshot MID-CARD-
        // RESOLUTION: SimCombatHandler.PlayCardAndWaitAsync returns as soon as
        // energy or hand size moves, which is when the play STARTS, so the
        // next board can show the played card still in `play_pile` with its
        // effects not yet landed. The mirror resolved the same play
        // atomically, so nothing on that board is gradable (the play_pile +
        // powers / discard + play_pile family, 59 of the 296 live2 records).
        // Rebuilding from it is worse than skipping: it adopts a play_pile
        // card that can never resume. Keep the carried mirror and grade at
        // the next stable wire. A legitimately suspended play cannot land
        // here: `pending_play` routed it to the parked verify above.
        // `crossed_shuffle` survives: only a rebuild that ADOPTS the world's
        // piles may clear it.
        if truthy(py::mget(obj(wire, "deck"), "play_pile")) {
            self.stats.in_flight_skips += 1;
            self.set_wire(wire);
            self.last_drift.clear();
            return Ok(Vec::new());
        }

        // Diff BEFORE transplanting: the transplant moves only hidden state,
        // which the diff ignores, and when it refuses (mismatched pile or
        // roster lengths) the field list is what explains why.
        let predicted = to_value(carried)?;
        // The corpses the game's board dropped stay in their slots (GH #1656),
        // and so do the ones it reports that the carried fight missed (GH
        // #1716): without them a missed kill renumbered the survivors with
        // nothing aligned, the rename went unnoted, and the reseed's rename
        // was applied to the carried uids (Spire Bench seeds 180000392,
        // 180000817, 180001103, 180001314).
        let spliced =
            mirror::with_reported_corpses(wire, mirror::with_carried_corpses(wire, &predicted));
        let mut fresh = build_state(&spliced)?;
        // The one hidden state the diff needs: a counter that sets an
        // observable (Void Form's free plays set every hand card's cost).
        // Diffed bare, each reconcile past the free plays reported
        // `$.deck.hand` and reseeded, and the reseed drops every enemy-side
        // carry: TestSubject's MULTI_CLAW escalation (its PAINFUL_STABS
        // counter) restarted at 0 (seeds 180000590, 180001456, 180000358,
        // 180000322).
        mirror_state::carry_player_power_counters(
            &mut fresh.state,
            carried,
            &mirror::undisclosed_power_counters(&spliced),
        );
        // An advance that consumed the shuffle stream means the world rolled
        // its own order for whatever was reshuffled: the PILE paths are
        // unverifiable, not wrong. Nothing else is, and skipping the whole
        // diff (as this once did) dropped 13.5% of reconciles out of the
        // desync denominator; the blind spot is scoped to `$.deck.*`.
        let skip_prefix = self.crossed_shuffle.then_some("$.deck.");
        if self.crossed_shuffle {
            self.stats.reshuffle_skips += 1;
        }
        let mut unreported = mirror::unreported_paths(wire);
        self.stats.unreported_fields += unreported.len() as u64;
        if self.just_seeded {
            unreported.insert("$.encounter.roster".into());
            self.just_seeded = false;
        }
        let mut desyncs = mirror::diff_observables(
            &predicted,
            &to_value(&fresh.state)?,
            skip_prefix,
            &unreported,
        );

        // Choice-effect in-flight skip: the play channel's twin on the CHOICE
        // channel. `SimCardSelector.GetSelectedCards` returns the moment the
        // pick arrives, so the next board can show the screen closed and its
        // grant still in transit: the mirror holding the picked card in hand
        // (or at cost 0) against a world that has neither yet. Deferring is
        // not merely tidier: under OBSERVE the report RESEEDS from the
        // half-applied board, deleting the grant for good and manufacturing
        // a mirror-BEHIND echo one decision later (16 of the 49 pure
        // `$.deck.hand` records in `mirror_live3`). A grant the world never
        // made still diverges at the next wire, because nothing re-arms it.
        let pending_grant = self.choice_in_flight.take();
        if mirror::choice_effect_in_flight(&desyncs, pending_grant.as_deref().unwrap_or(&[])) {
            self.stats.choice_in_flight_skips += 1;
            self.set_wire(wire);
            self.last_drift.clear();
            return Ok(Vec::new());
        }

        // When the producer carries the run's own dice, `fresh` already holds
        // the WORLD's streams and carrying the mirror's would replace known
        // truth with drift. The drift itself is recorded, not raised: the
        // cursor is hidden state, not one of the observables STRICT is
        // defined over, and a drift is a lead, not yet a board divergence.
        let drift = if world_rng {
            let drift = mirror::cursor_drift(&predicted, wire);
            self.stats.cursor_drifts += drift.len() as u64;
            drift
        } else {
            Vec::new()
        };

        let slots = mirror_state::alignment_in_fresh_order(
            &fresh.state,
            &spliced,
            &mirror::enemy_alignment(&spliced, &predicted),
        );
        let pairs = mirror_state::enemy_uid_pairs(&fresh.state, carried, &slots);
        mirror_state::carry_corpse_intents(&mut fresh.state, carried, &slots);
        let carry = carry_for(&spliced, slots);
        if let Err(exc) = mirror_state::transplant_hidden(&mut fresh.state, carried, carry) {
            let diverged_on = if desyncs.is_empty() {
                String::new()
            } else {
                format!(" — diverged on: {}", first_six(&desyncs))
            };
            let paths = if desyncs.is_empty() {
                vec!["$.transplant".to_owned()]
            } else {
                paths_of(&desyncs)
            };
            self.diverge(
                format!("transplant refused the reconstruction: {exc}{diverged_on}"),
                "transplant",
                Some(wire.clone()),
                paths,
                Some(predicted),
                map([
                    ("error", Value::String(exc.clone())),
                    ("desyncs", desync_rows(&desyncs)),
                ]),
            )?;
            self.reseed_after_observe(wire)?;
            if desyncs.is_empty() {
                desyncs.push(Desync {
                    path: "$.transplant".into(),
                    mirror: Value::Null,
                    world: Value::String(exc),
                });
            }
            return Ok(desyncs);
        }

        // The carried AI may name a move the board's own did not.
        mirror_state::reprime_unshown_amounts(&mut fresh.state);
        mirror_state::rampage_growth_as_var(&mut fresh.state);
        self.state = Some(fresh);
        self.state_generation += 1;
        self.crossed_shuffle = false;
        self.set_wire(wire);
        self.last_drift = drift;
        self.note_enemy_remap(&pairs);

        if !desyncs.is_empty()
            && self.teammates > 0
            && desyncs.iter().all(|d| teammate_reachable(&d.path))
        {
            // Co-op: every diff is on the enemies, which a teammate moves
            // between our decisions. Keep the adopted board.
            self.stats.teammate_adoptions += 1;
            self.last_adopted = std::mem::take(&mut desyncs);
        }

        if !desyncs.is_empty() {
            self.stats.desyncs += 1;
            self.stats.desync_fields += desyncs.len() as u64;
            self.diverge(
                format!(
                    "{} deterministic observable(s) diverged: {}",
                    desyncs.len(),
                    first_six(&desyncs)
                ),
                "desync",
                Some(wire.clone()),
                paths_of(&desyncs),
                Some(predicted),
                map([("desyncs", desync_rows(&desyncs))]),
            )?;
            self.reseed_after_observe(wire)?;
        }
        // Close the window whether or not it was reported.
        self.take_damage_trace();
        Ok(desyncs)
    }

    /// Restart from reality after an OBSERVE divergence, salvaging the
    /// undisclosed hidden state and the parked choice from the dropped
    /// state. No-op while a fight is mirrored (i.e. under STRICT).
    ///
    /// # Errors
    ///
    /// The wire does not build a state.
    pub fn reseed_after_observe(&mut self, wire: &Value) -> Result<(), MirrorError> {
        if self.alive() {
            return Ok(());
        }
        let stale = self.last_dropped.take();
        let stale_parked = self.last_dropped_parked.take();
        // The dropped fight still knows the corpses the board left out.
        let spliced = match stale.as_ref().map(|m| to_value(&m.state)) {
            Some(Ok(carried)) => mirror::with_carried_corpses(wire, &carried),
            _ => wire.clone(),
        };
        // Where the dropped fight missed a kill the board reports (GH #1716).
        let spliced = mirror::with_reported_corpses(wire, spliced);
        self.seed(&spliced)?;
        let (Some(stale), Some(fresh)) = (stale, self.state.as_mut()) else {
            return Ok(());
        };
        // One slot per enemy of the rebuilt roster, the spliced corpses
        // included: sized to the bare wire, a kill the board dropped made
        // the salvage refuse and wiped every card's hidden vars (GH #1772).
        let carry = carry_for(wire, vec![None; fresh.state.enemy_uids.len()]);
        // The bare reseed lost ALL carried hidden state, including what the
        // wire cannot disclose: relic latches (ArtOfWar's attacks-played bit
        // re-armed and over-granted +1 energy on v27wr), history counters
        // (Spite / TearAsunder), per-card vars. The divergence did not prove
        // those wrong, and re-zeroing them manufactures the NEXT divergence.
        // Enemy-side carries stay dropped (`None` slots): a mispredicted
        // board most often means the enemy model was the wrong part.
        // Best-effort by definition: a refused salvage leaves the bare reseed.
        let _ = mirror_state::transplant_hidden(&mut fresh.state, &stale.state, carry);
        let mut pairs = None;
        if let Ok(carried) = to_value(&stale.state) {
            let slots = mirror_state::alignment_in_fresh_order(
                &fresh.state,
                &spliced,
                &mirror::enemy_alignment(&spliced, &carried),
            );
            mirror_state::carry_corpse_intents(&mut fresh.state, &stale.state, &slots);
            // The appliers of the player's applier-gated debuffs are player-
            // side state naming an enemy: the all-`None` salvage above cannot
            // map them, and without one Dampen/Shrink/Hex/Constrict outlive
            // their caster for the rest of the fight.
            mirror_state::carry_power_appliers(&mut fresh.state, &stale.state, &slots);
            pairs = Some(mirror_state::enemy_uid_pairs(
                &fresh.state,
                &stale.state,
                &slots,
            ));
        }
        mirror_state::reprime_unshown_amounts(&mut fresh.state);
        mirror_state::rampage_growth_as_var(&mut fresh.state);
        self.state_generation += 1;
        if let Some(stale_parked) = stale_parked {
            // The parked choice is the same salvage class: only its kind lets
            // the next injection resolve locators for a deterministic screen
            // (25 of the 32 v34wr `unlocatable_injection` records were a parked
            // choice discarded just before it was needed). Re-locate each
            // option by identity against the mirror's OWN piles (the locators
            // index the state that resolves them, and the two dialects
            // differ: `DECAY` vs `curse:Decay`); one not found is dropped so
            // the injection reports a genuine `unlocatable_option`.
            let seeded = to_value(&fresh.state).unwrap_or_else(|_| wire.clone());
            self.parked = mirror::relocate_parked(&stale_parked, &seeded);
        }
        // The reseed renumbers the enemies too: an intent log keyed by uid
        // follows them from the dropped state (GH #1695). Without this it kept
        // the rename the refused rebuild noted, onto a numbering the reseed
        // then replaced.
        if let Some(pairs) = pairs {
            self.note_enemy_remap(&pairs);
        }
        Ok(())
    }

    // -- damage-source forensics (GH #368.1) -------------------------------

    fn arm_damage_trace(&self) {
        if self.damage_trace_on {
            sts2_core::damage_trace::set_armed(true);
        }
    }

    fn accumulate_damage_trace(&mut self) {
        if !self.damage_trace_on {
            return;
        }
        self.last_damage_trace
            .extend(sts2_core::damage_trace::take().into_iter().map(|e| {
                json!({
                    "seq": e.seq,
                    "source": e.source,
                    "source_stack": e.source_stack,
                    "dealer": e.dealer,
                    "target_uid": e.target_uid,
                    "target_id": e.target_id,
                    "final_damage": e.final_damage,
                    "blocked": e.blocked,
                    "hp_loss": e.hp_loss,
                    "hp_before": e.hp_before,
                    "hp_after": e.hp_after,
                    "turn": e.turn,
                })
            }));
        self.damage_trace_dropped += u64::from(sts2_core::damage_trace::dropped());
        sts2_core::damage_trace::set_armed(false);
    }

    fn take_damage_trace(&mut self) -> (Vec<Value>, u64) {
        (
            std::mem::take(&mut self.last_damage_trace),
            std::mem::take(&mut self.damage_trace_dropped),
        )
    }

    // -- advance -----------------------------------------------------------

    /// Commit one action the world already took; the engine's step result
    /// (`advanced`, `pending_choice`, `combat_won`, `combat_lost`,
    /// `invalid_target`, `illegal_play`).
    ///
    /// # Errors
    ///
    /// No fight; an action that does not decode; under STRICT, an action the
    /// engine rejects.
    pub fn advance(&mut self, action: &Value) -> Result<&'static str, MirrorError> {
        if self.state.is_none() {
            return Err(MirrorError::NoState);
        }
        // Any committed action ends the previous choice's grace period.
        self.choice_in_flight = None;
        let decoded = mirror_state::parse_bridge_action(action).map_err(MirrorError::Value)?;
        let state = self.mirrored_mut()?;
        let before = state.run_rng.shuffle.counter();
        // The trace is armed only across the mirror's own step, never across
        // the driver's search.
        self.arm_damage_trace();
        let state = self.mirrored_mut()?;
        let result = mirror_state::step_result_name(sts2_core::apply(state, decoded));
        let crossed = state.run_rng.shuffle.counter() != before;
        self.accumulate_damage_trace();
        if crossed {
            self.crossed_shuffle = true;
        }
        self.last_action = Some(action.clone());
        self.last_result = Some(result.to_owned());

        if matches!(result, "invalid_target" | "illegal_play") {
            self.diverge(
                format!(
                    "engine rejected an action the world committed: {}",
                    py::repr(action)
                ),
                "invalid_action",
                self.last_wire.clone(),
                vec!["$.action".into()],
                None,
                map([("action", action.clone())]),
            )?;
            return Ok(result);
        }
        self.parked = if result == "pending_choice" {
            self.state
                .as_ref()
                .and_then(|m| sts2_wire::emit::pending_choice(&m.state))
        } else {
            None
        };
        Ok(result)
    }

    // -- choices -----------------------------------------------------------

    /// Open the choice the WORLD is showing, and grade the engine's own;
    /// the graded differences.
    ///
    /// # Errors
    ///
    /// No fight; an unknown kind; under STRICT, a divergence.
    #[allow(clippy::too_many_lines)]
    pub fn inject_choice(
        &mut self,
        kind: &str,
        options: Vec<Value>,
        min_select: i64,
        max_select: i64,
        can_skip: bool,
        origin: Value,
    ) -> Result<Vec<Desync>, MirrorError> {
        if self.state.is_none() {
            return Err(MirrorError::NoState);
        }
        let parked = self.parked.take();
        // A screen that never opens leaves nothing to answer.
        self.open_injection = None;
        let mut options = options;

        // A generative KIND whose parked options (of that same kind) sit in a
        // pile is not a generated offer: Heirloom Hammer clones one of the colorless HAND
        // cards through `AddGeneratedToHand` (heirloom_hammer.rs). Its options
        // are located like any hand screen's, or they tokenize as pile
        // "generated" with no hand slot (seed 180000982).
        let generative = mirror_state::choice_kind_is_generative(kind).ok_or_else(|| {
            MirrorError::Value(format!(
                "choice_kind_is_generative: unknown choice kind {kind:?}"
            ))
        })? && !parked
            .as_ref()
            .is_some_and(|p| get(p, "kind").as_str() == Some(kind) && options_are_located(p));
        let engine_derived = parked.clone().unwrap_or(Value::Null);
        let mut graded = Vec::new();
        if let (Some(p), false) = (&parked, generative) {
            if !self.resolve_locators(&mut options, p, kind)? {
                return Ok(Vec::new());
            }
            graded = grade_choice(p, kind, &options, min_select, max_select);
        } else if !generative
            && options
                .iter()
                .any(|o| get(o, "pile").is_null() || get(o, "index").is_null())
        {
            // No parked choice to locate against, and the driver only knows
            // card NAMES. Every deterministic resolver bails on a missing
            // `selected_card_uid`, and `ChoiceKind::CardSelect` is an explicit
            // no-op placeholder: the world would upgrade / exhaust / discard a
            // card while the mirror applied NOTHING, with nothing graded and
            // nothing raised. Reachable whenever the mirror was re-seeded
            // rather than parked.
            self.diverge(
                format!(
                    "cannot answer a deterministic {kind}: the world's options carry no \
                     locator and the engine has no parked choice to resolve them against \
                     — injecting would resolve to a silent no-op"
                ),
                "unlocatable_injection",
                self.last_wire.clone(),
                vec!["$.choice.injection".into()],
                None,
                map([
                    ("choice_kind", Value::String(kind.into())),
                    ("options", Value::Array(options)),
                    ("engine_derived", engine_derived),
                ]),
            )?;
            return Ok(Vec::new());
        }

        // The engine parks a skippable choose-a-card screen as pick one with
        // a skip (`request_card_choice(kind, options, 1, 1, true)`), where the
        // world's arity reads "pick none" (GH #1672): the student reads the
        // engine's minimum (RUNCTX `R_SEL_MIN`).
        let engine_min = if can_skip
            && min_select <= 0
            && max_select >= 1
            && get(&origin, "source").as_str() == Some("FromChooseACardScreen")
        {
            1
        } else {
            min_select
        };
        let opened = match (u8::try_from(engine_min), u8::try_from(max_select)) {
            (Ok(lo), Ok(hi)) => {
                let state = self.mirrored_mut()?;
                mirror_state::open_injected_choice(
                    state,
                    kind,
                    &options,
                    lo,
                    hi,
                    can_skip,
                    parked.is_some(),
                )
            }
            _ => Err(format!(
                "open_injected_choice: arity ({min_select}, {max_select}) out of range"
            )),
        };
        // `replace_parked` when sitting on the engine's own derived list: the
        // world's options win, and clearing the choice this way leaves
        // `pending_play` intact, which is why the mirror was kept parked.
        if let Err(exc) = opened {
            self.diverge(
                format!("engine refused the world's option list: {exc}"),
                "injection",
                self.last_wire.clone(),
                vec!["$.choice.injection".into()],
                None,
                map([
                    ("choice_kind", Value::String(kind.into())),
                    ("options", Value::Array(options)),
                    ("engine_derived", engine_derived),
                    ("error", Value::String(exc)),
                ]),
            )?;
            return Ok(Vec::new());
        }

        self.stats.injections += 1;
        let injection = json!({
            "kind": kind,
            "options": options,
            "min_select": min_select,
            "max_select": max_select,
            "engine_derived": get(&engine_derived, "cards"),
            "picks": null,
        });
        self.open_injection = Some(injection.clone());
        self.last_injection = Some(injection);
        if parked.is_none() {
            self.stats.unparked_injections += 1;
            self.last_unparked = Some(json!({
                "kind": kind,
                "options": options.iter().map(|o| get(o, "id").clone()).collect::<Vec<_>>(),
                "min_select": min_select,
                "max_select": max_select,
                "origin": origin,
                "after_action": self.last_action,
                "after_result": self.last_result,
            }));
        } else {
            self.last_unparked = None;
        }
        if !graded.is_empty() {
            // Per-injection, not per-reconcile: not `stats.desyncs`.
            self.stats.choice_mismatches += 1;
            self.stats.desync_fields += graded.len() as u64;
            self.diverge(
                format!(
                    "engine derived a different {kind} than the world showed: {}",
                    first_six(&graded)
                ),
                "choice_mismatch",
                self.last_wire.clone(),
                paths_of(&graded),
                None,
                map([
                    ("choice_kind", Value::String(kind.into())),
                    ("options", Value::Array(options)),
                    ("engine_derived", engine_derived),
                    ("desyncs", desync_rows(&graded)),
                ]),
            )?;
        }
        Ok(graded)
    }

    /// A screen the world opened from a play the mirror never saw start:
    /// the world's board holds a card in `play_pile` while the carried fight
    /// has nothing in play, no play suspended, no choice parked and no walk
    /// suspended. Seeds
    /// from the world's board then, and says so.
    ///
    /// Imbued autoplays its card on turn 1 after the opening draw
    /// (`Imbued.AfterAutoPrePlayPhaseEntered`). A mirror seeded on Toolbox's
    /// combat-start screen (`Toolbox.BeforeHandDraw`) never drew, and
    /// [`Self::reconcile`]'s in-flight skip kept that pre-draw board under
    /// the autoplayed Splash's screen: the hand in the draw pile, Splash
    /// missing from play (seed 180001961).
    ///
    /// # Errors
    ///
    /// The world's board does not build.
    pub fn adopt_unseen_play(&mut self, wire: &Value) -> Result<bool, MirrorError> {
        let Some(m) = self.state.as_ref() else {
            return Ok(false);
        };
        if self.parked.is_some()
            || !m.state.player.play.is_empty()
            || m.state.pending_play.is_some()
            || m.state.pending_choice.is_some()
            || mirror_state::has_suspended_walk(&m.state)
            || !truthy(py::mget(obj(wire, "deck"), "play_pile"))
        {
            return Ok(false);
        }
        self.seed(wire)?;
        Ok(true)
    }

    /// Fill in each option's `pile` / `index` from the engine's own choice,
    /// one-to-one on `(id, upgraded)` in the world's order. False when the
    /// mirror diverged (under STRICT that is an error instead).
    fn resolve_locators(
        &mut self,
        options: &mut [Value],
        parked: &Value,
        kind: &str,
    ) -> Result<bool, MirrorError> {
        let available = list(parked, "cards");
        let mut used = BTreeSet::new();
        for i in 0..options.len() {
            let opt = &options[i];
            if !get(opt, "pile").is_null() && !get(opt, "index").is_null() {
                continue;
            }
            let id = get(opt, "id").clone();
            let upgraded = truthy(get(opt, "upgraded"));
            let found = available.iter().enumerate().position(|(j, c)| {
                !used.contains(&j)
                    && py::eq(get(c, "id"), &id)
                    && truthy(get(c, "upgraded")) == upgraded
                    && !get(c, "pile").is_null()
            });
            let Some(j) = found else {
                self.diverge(
                    format!(
                        "the world offers {} (upgraded={}) as {kind} option {i}, which the \
                         engine's own choice does not contain",
                        py::repr(&id),
                        if upgraded { "True" } else { "False" }
                    ),
                    "unlocatable_option",
                    self.last_wire.clone(),
                    vec!["$.choice.options".into()],
                    None,
                    map([
                        ("choice_kind", Value::String(kind.into())),
                        ("options", Value::Array(options.to_vec())),
                        ("engine_derived", parked.clone()),
                    ]),
                )?;
                return Ok(false);
            };
            used.insert(j);
            if let Some(o) = options[i].as_object_mut() {
                o.insert("pile".into(), get(&available[j], "pile").clone());
                o.insert("index".into(), get(&available[j], "index").clone());
            }
        }
        Ok(true)
    }

    /// Park the rest of turn setup behind the open turn-start screen a relic
    /// raised. False when nothing is mirrored or the engine declined.
    ///
    /// # Errors
    ///
    /// An unknown relic id.
    pub fn arm_turn_start_resume(&mut self, after_relic: &str) -> Result<bool, MirrorError> {
        if self.state.is_none() {
            return Ok(false);
        }
        mirror_state::arm_turn_start_resume(self.mirrored_mut()?, after_relic)
            .map_err(MirrorError::Value)
    }

    /// The power twin of [`Self::arm_turn_start_resume`] (GH #355).
    ///
    /// # Errors
    ///
    /// An unknown power id.
    pub fn arm_turn_start_resume_after_power(
        &mut self,
        after_power: &str,
    ) -> Result<bool, MirrorError> {
        if self.state.is_none() {
            return Ok(false);
        }
        mirror_state::arm_turn_start_resume_after_power(self.mirrored_mut()?, after_power)
            .map_err(MirrorError::Value)
    }

    /// Answer the open choice and confirm it closed; the final step result.
    ///
    /// # Errors
    ///
    /// No fight; a pick that does not decode; under STRICT, no choice open
    /// or a choice that survives its answer.
    pub fn advance_through_choice(
        &mut self,
        selection: &[i64],
    ) -> Result<&'static str, MirrorError> {
        let Some(m) = self.state.as_ref() else {
            return Err(MirrorError::NoState);
        };
        // Consumed whether or not the answer lands.
        let injected = self.open_injection.take();
        let Some(open_choice) = sts2_wire::emit::pending_choice(&m.state) else {
            self.diverge(
                "advance_through_choice with no choice open".into(),
                "no_choice",
                self.last_wire.clone(),
                vec!["$.choice.absent".into()],
                None,
                map([("selection", json!(selection))]),
            )?;
            return Ok("advanced");
        };

        if let Some(Value::Object(li)) = self.last_injection.as_mut() {
            li.insert("picks".into(), json!(selection));
        }
        let mut result = self.advance(&json!({"action": "choose", "selection": selection}))?;
        // A partial multi-pick accumulates and re-opens the same choice; the
        // world resolved in one shot, so finalize.
        let max_select = match get(&open_choice, "max_select") {
            Value::Null => 1,
            v => py::int_of(v),
        };
        let still_open = |me: &Self| {
            me.state
                .as_ref()
                .and_then(|m| sts2_wire::emit::pending_choice(&m.state))
        };
        if still_open(self).is_some()
            && max_select > 1
            && i64::try_from(selection.len()).unwrap_or(i64::MAX) < max_select
        {
            result = self.advance(&json!({"action": "choose", "selection": []}))?;
        }
        // An answer that met the choice's minimum closed it. A choice open
        // after that is the next one: a card played twice (Throwing Axe, a
        // replay) opens its screen again, and the world shows that screen
        // next. Only an answer short of the minimum leaves the same choice
        // waiting for picks the world never made.
        let min_select = match get(&open_choice, "min_select") {
            Value::Null => 1,
            v => py::int_of(v),
        };
        let short = i64::try_from(selection.len()).unwrap_or(i64::MAX) < min_select;
        if let Some(open) = still_open(self).filter(|_| short) {
            // The engine expects more picks than the world made.
            self.diverge(
                format!(
                    "choice still open after selection {}",
                    py::repr(&json!(selection))
                ),
                "unclean_choice",
                self.last_wire.clone(),
                vec!["$.choice.unclean".into()],
                None,
                map([("selection", json!(selection)), ("still_open", open)]),
            )?;
        }
        // Arm the one-board grace period for the grant this pick made.
        let opts = injected
            .as_ref()
            .map_or(&[][..], |inj| list(inj, "options"));
        let named: Vec<Value> = selection
            .iter()
            .filter_map(|&i| usize::try_from(i).ok().and_then(|i| opts.get(i)).cloned())
            .collect();
        self.choice_in_flight = (!named.is_empty()).then_some(named);
        Ok(result)
    }

    // -- search ------------------------------------------------------------

    /// `CombatMirror.search`: one honest search on a clone of the mirror, the
    /// raw `mcts_recommend` result.
    ///
    /// A live mirror on the belief draw representation is a divergence (only
    /// the search's private clone may carry it): recorded, and an empty
    /// result. A clairvoyant profile is refused outright: the mirror holds
    /// the true draw order.
    ///
    /// # Errors
    ///
    /// No fight; a clairvoyant or unknown profile ([`MirrorError::Value`]);
    /// under STRICT, the belief leak.
    pub fn search(&mut self, cfg: &RecommendConfig) -> Result<Value, MirrorError> {
        let m = self.state.as_ref().ok_or(MirrorError::NoState)?;
        if m.state.player.deferred_draw_order {
            let knowledge = json!({
                "pinned_top": pin_positions(&m.state, &m.state.player.pinned_top),
                "pinned_bottom": pin_positions(&m.state, &m.state.player.pinned_bottom),
                "deferred_draw_order": true,
            });
            let wire = self.last_wire.clone();
            self.diverge(
                "the live mirror is on the belief draw representation \
                 (deferred_draw_order=True) — only the private search clone may be"
                    .into(),
                "belief_leak",
                wire,
                vec!["$.deck.deferred_draw_order".into()],
                None,
                map([("draw_pile_knowledge", knowledge)]),
            )?;
            return Ok(json!({"actions": [], "pv": [], "sims": 0}));
        }
        if cfg.hidden_info == "clairvoyant" {
            return Err(MirrorError::Value(format!(
                "CombatMirror.search: refusing hidden_info={}. The mirror holds the TRUE \
                 draw order and RNG cursors, so a clairvoyant root would plan against cards \
                 the player cannot see. Use 'beliefdraw' (the deployed honest profile), \
                 'drawchance', or 'rootscramble'.",
                py::repr(&json!(cfg.hidden_info))
            )));
        }
        let (state, known) = (m.state.clone(), m.intent_damage_known);
        self.recommend(&state, known, cfg)
    }

    /// One search over `state`, counted in the stats.
    fn recommend(
        &mut self,
        state: &CombatState,
        intent_damage_known: bool,
        cfg: &RecommendConfig,
    ) -> Result<Value, MirrorError> {
        self.note_search();
        crate::search::recommend(state, intent_damage_known, cfg).map_err(MirrorError::Value)
    }

    /// `CombatMirror.search_choice_picks`: answer the open choice, one search
    /// per pick on a multi-pick screen (the engine decomposes it
    /// sequentially), on a PRIVATE clone; the mirror itself is not advanced.
    /// `hand` is the wire's `deck.hand`, the frame the tags are read in.
    /// Distinct option indices in pick order; empty means skip.
    ///
    /// # Errors
    ///
    /// As [`Self::search`]; a row the hand cannot map ([`MirrorError::Key`]).
    pub fn search_choice_picks(
        &mut self,
        cfg: &RecommendConfig,
        hand: &[Value],
        n_options: i64,
        min_select: i64,
        max_select: i64,
    ) -> Result<Vec<i64>, MirrorError> {
        use crate::combat_advice::{choice_picks_from_tags, next_choice_pick_from_tags, NextPick};
        let tags = |result: &Value| {
            let (tags, dropped) =
                crate::combat_advice::tags_from_result(result, hand).map_err(|e| {
                    let (crate::combat_advice::Unmapped::Unmappable(m)
                    | crate::combat_advice::Unmapped::Unknown(m)) = e;
                    MirrorError::Key(m)
                })?;
            if let Some(first) = dropped.first() {
                // `tags_from_result`'s note, as the Python one printed it.
                eprintln!(
                    "[hint] dropped {} unmappable action row(s) from the ranking, kept {}: {first}",
                    dropped.len(),
                    tags.len()
                );
            }
            Ok(tags)
        };
        if max_select <= 1 {
            let result = self.search(cfg)?;
            return Ok(choice_picks_from_tags(
                &tags(&result)?,
                n_options,
                min_select,
                max_select,
            ));
        }
        // The guards belong to the live mirror; run them once, then walk.
        let mut result = self.search(cfg)?;
        let m = self.state.as_ref().ok_or(MirrorError::NoState)?;
        let (mut working, known) = (m.state.clone(), m.intent_damage_known);
        let mut picks: Vec<i64> = Vec::new();
        let count = |p: &Vec<i64>| i64::try_from(p.len()).unwrap_or(i64::MAX);
        while count(&picks) < max_select {
            let next = next_choice_pick_from_tags(
                &tags(&result)?,
                n_options,
                &picks,
                count(&picks) >= min_select,
            );
            let Some(NextPick::Option(next)) = next else {
                break;
            };
            picks.push(next);
            if count(&picks) >= max_select {
                break;
            }
            // Condition the next search on the pick; the engine auto-finalizes
            // on the pick that reaches `max_select`.
            let choose = mirror_state::parse_bridge_action(
                &json!({"action": "choose", "selection": [next]}),
            )
            .map_err(MirrorError::Value)?;
            sts2_core::apply(&mut working, choose);
            if sts2_wire::emit::pending_choice(&working).is_none() {
                break;
            }
            result = self.recommend(&working, known, cfg)?;
        }
        if count(&picks) < min_select {
            let rest: Vec<i64> = (0..n_options).filter(|i| !picks.contains(i)).collect();
            picks.extend(rest);
            picks.truncate(usize::try_from(min_select).unwrap_or(0));
        }
        Ok(picks)
    }

    // -- divergence --------------------------------------------------------

    /// Record a divergence, then fail (STRICT) or drop the mirror (OBSERVE).
    ///
    /// `mirror_wire` must be the mirror's PREDICTION, the state before
    /// reality replaced it; `None` serializes the current state.
    ///
    /// # Errors
    ///
    /// [`MirrorError::Diverged`] under STRICT.
    pub fn diverge(
        &mut self,
        message: String,
        kind: &str,
        wire: Option<Value>,
        paths: Vec<String>,
        mirror_wire: Option<Value>,
        extra: Map<String, Value>,
    ) -> Result<(), MirrorError> {
        let mut report = map([
            ("label", Value::String(self.label.clone())),
            ("kind", Value::String(kind.into())),
            ("message", Value::String(message.clone())),
            ("policy", Value::String(self.policy.name().into())),
            ("stats", self.stats.as_value()),
            (
                "last_action",
                self.last_action.clone().unwrap_or(Value::Null),
            ),
            ("world_wire", wire.unwrap_or(Value::Null)),
        ]);
        let (trace, trace_dropped) = self.take_damage_trace();
        if !trace.is_empty() {
            report.insert("damage_trace".into(), Value::Array(trace));
            if trace_dropped > 0 {
                // Never let a truncated trace read as a complete one.
                report.insert("damage_trace_dropped".into(), json!(trace_dropped));
            }
        }
        if let Some(injection) = &self.last_injection {
            report.insert("last_injection".into(), injection.clone());
        }
        if !self.last_drift.is_empty() {
            report.insert(
                "cursor_drift".into(),
                self.last_drift
                    .iter()
                    .map(|d| json!({"stream": d.path, "mirror": d.mirror, "world": d.world}))
                    .collect(),
            );
        }
        if let Some(unparked) = &self.last_unparked {
            report.insert("last_unparked_choice".into(), unparked.clone());
        }
        let mirror_wire = match (mirror_wire, &self.state) {
            (Some(v), _) => v,
            (None, Some(m)) => to_value(&m.state).unwrap_or_else(|e| {
                // Forensics must never mask the divergence.
                report.insert("mirror_wire_error".into(), Value::String(e.to_string()));
                Value::Null
            }),
            (None, None) => Value::Null,
        };
        report.insert("mirror_wire".into(), mirror_wire);
        report.insert("paths".into(), json!(paths));
        report.extend(extra);
        let divergence = Divergence {
            message,
            paths,
            report: Value::Object(report),
        };
        self.divergences.push(divergence.clone());
        if self.policy == Policy::Strict {
            return Err(MirrorError::Diverged(divergence));
        }
        self.tainted = true;
        self.drop_fight(&divergence.message);
        Ok(())
    }
}

/// `(id, upgraded, pile, index)`: an option's board position.
fn option_key(o: &Value) -> Value {
    json!([
        get(o, "id"),
        truthy(get(o, "upgraded")),
        get(o, "pile"),
        get(o, "index")
    ])
}

/// Python's tuple order over [`option_key`]s.
fn key_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    py::as_list(a)
        .iter()
        .zip(py::as_list(b))
        .map(|(x, y)| py::cmp(x, y))
        .find(|o| o.is_ne())
        .unwrap_or_else(|| py::as_list(a).len().cmp(&py::as_list(b).len()))
}

/// `_grade_choice`: the engine's own derived choice against the world's, by
/// kind, arity and the option multiset (a concrete board position each).
fn grade_choice(
    parked: &Value,
    kind: &str,
    options: &[Value],
    min_select: i64,
    max_select: i64,
) -> Vec<Desync> {
    let arity = |key: &str| match get(parked, key) {
        Value::Null => 0,
        v => py::int_of(v),
    };
    let mut out = Vec::new();
    if get(parked, "kind").as_str() != Some(kind) {
        out.push(Desync {
            path: "$.choice.kind".into(),
            mirror: get(parked, "kind").clone(),
            world: Value::String(kind.into()),
        });
    }
    if arity("min_select") != min_select {
        out.push(Desync {
            path: "$.choice.min_select".into(),
            mirror: get(parked, "min_select").clone(),
            world: json!(min_select),
        });
    }
    if arity("max_select") != max_select {
        out.push(Desync {
            path: "$.choice.max_select".into(),
            mirror: get(parked, "max_select").clone(),
            world: json!(max_select),
        });
    }
    let mut mine: Vec<Value> = list(parked, "cards").iter().map(option_key).collect();
    let mut theirs: Vec<Value> = options.iter().map(option_key).collect();
    mine.sort_by(key_cmp);
    theirs.sort_by(key_cmp);
    let (mine, theirs) = (Value::Array(mine), Value::Array(theirs));
    if !py::eq(&mine, &theirs) {
        out.push(Desync {
            path: "$.choice.options".into(),
            mirror: mine,
            world: theirs,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grading_ignores_order_and_names_each_disagreement() {
        let parked = json!({
            "kind": "exhaust_select", "min_select": 1, "max_select": 1,
            "cards": [
                {"id": "STRIKE", "upgraded": false, "pile": "hand", "index": 0},
                {"id": "DEFEND", "upgraded": false, "pile": "hand", "index": 1},
            ],
        });
        let same = [
            json!({"id": "DEFEND", "upgraded": false, "pile": "hand", "index": 1}),
            json!({"id": "STRIKE", "upgraded": false, "pile": "hand", "index": 0}),
        ];
        assert!(grade_choice(&parked, "exhaust_select", &same, 1, 1).is_empty());
        let paths: Vec<String> = grade_choice(&parked, "discard_select", &same[..1], 1, 2)
            .into_iter()
            .map(|d| d.path)
            .collect();
        assert_eq!(
            paths,
            ["$.choice.kind", "$.choice.max_select", "$.choice.options"]
        );
    }

    #[test]
    fn stats_rates_are_per_thousand_and_zero_when_idle() {
        let mut stats = Stats::default();
        assert_eq!(stats.as_value()["desync_rate_per_1000"], json!(0.0));
        stats.reconciles = 3;
        stats.desyncs = 1;
        assert_eq!(stats.as_value()["desync_rate_per_1000"], json!(333.333));
    }

    #[test]
    fn a_mirror_without_a_fight_refuses_to_advance() {
        let mut m = CombatMirror::new(Policy::Strict, "t", false);
        assert!(matches!(
            m.advance(&json!({"action": "end_turn"})),
            Err(MirrorError::NoState)
        ));
        assert!(!m.arm_turn_start_resume("GAMBLING_CHIP").unwrap());
        assert!(m.fork("").is_none());
    }

    /// GH #1656 (and #1651): the game drops an ordinary corpse from its
    /// board. The rebuild keeps it in its slot, as the engine the student
    /// trained on does, so no survivor moves or is renamed.
    #[test]
    fn a_corpse_the_board_dropped_keeps_its_slot() {
        let deck: Vec<String> = ["StrikeIronclad", "DefendIronclad"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut wire = to_value(&cs).expect("wire");
        let enc = wire["encounter"].as_array().expect("encounter").clone();
        assert_eq!(enc.len(), 3, "three rats");
        // The first rat is dead; the mirror still holds its corpse.
        wire["encounter"][0]["hp"] = json!(0);
        wire["encounter"][0]["is_dead"] = json!(true);
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let before: Vec<u32> = m
            .mirrored()
            .expect("fight")
            .state
            .enemy_uids
            .iter()
            .map(|u| u.0.get())
            .collect();
        assert!(m.take_enemy_remap().is_none(), "seeding renames nothing");

        // The game's next board has dropped the corpse.
        let mut world = wire.clone();
        world["encounter"]
            .as_array_mut()
            .expect("encounter")
            .remove(0);
        m.reconcile(&world).expect("reconciles");
        let fight = &m.mirrored().expect("fight").state;
        let after: Vec<u32> = fight.enemy_uids.iter().map(|u| u.0.get()).collect();
        assert_eq!(
            after, before,
            "the corpse keeps slot 0, the survivors keep theirs"
        );
        assert!(
            fight.is_removed_corpse(fight.enemy_uids[0]),
            "slot 0 is the corpse"
        );
        assert!(m.take_enemy_remap().is_none(), "nothing was renamed");
    }

    /// Spire Bench seeds 180000392 / 180000817: the world killed the first
    /// of three same-kind enemies while the mirror had it alive; the board
    /// dropped it and reported it in `removed_roster`. Its uid stays the
    /// corpse's and the survivors keep theirs, so the intent log keyed by uid
    /// is not shifted onto the wrong creatures.
    #[test]
    fn a_kill_the_mirror_missed_renames_no_survivor() {
        let deck: Vec<String> = ["StrikeIronclad", "DefendIronclad"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let wire = to_value(&cs).expect("wire");
        assert_eq!(wire["encounter"].as_array().expect("encounter").len(), 3);
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let before: Vec<u32> = m
            .mirrored()
            .expect("fight")
            .state
            .enemy_uids
            .iter()
            .map(|u| u.0.get())
            .collect();
        assert!(m.take_enemy_remap().is_none());

        // The world killed rat 0 (the mirror has it alive) and dropped it.
        let mut corpse = wire["encounter"][0].clone();
        corpse["hp"] = json!(0);
        corpse["is_dead"] = json!(true);
        corpse["removed_after_death"] = json!(true);
        let mut world = wire.clone();
        let enc = world["encounter"].as_array_mut().expect("encounter");
        enc.remove(0);
        let survivors = enc.clone();
        world["removed_roster"] = Value::Array(std::iter::once(corpse).chain(survivors).collect());
        let _ = m.reconcile(&world).expect("reconciles under OBSERVE");
        let fight = &m.mirrored().expect("fight").state;
        let after: Vec<u32> = fight.enemy_uids.iter().map(|u| u.0.get()).collect();
        assert_eq!(
            after, before,
            "the corpse keeps slot 0, the survivors keep theirs"
        );
        assert!(
            fight.is_removed_corpse(fight.enemy_uids[0]),
            "slot 0 is the corpse"
        );
        let renamed: Vec<(u32, u32)> = m
            .take_enemy_remap()
            .unwrap_or_default()
            .into_iter()
            .filter(|(a, b)| a != b)
            .collect();
        assert!(renamed.is_empty(), "no uid moved: {renamed:?}");
    }

    /// GH #1669: the game's board shows a dead enemy with no telegraph; the
    /// corpse keeps the one it died with, as in the engine the student
    /// trained on.
    #[test]
    fn a_corpse_keeps_the_telegraph_it_died_with() {
        let deck: Vec<String> = vec!["StrikeIronclad".into()];
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let wire = to_value(&cs).expect("wire");
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let intent_of = |m: &CombatMirror| {
            let st = &m.mirrored().expect("fight").state;
            let uid = st.enemy_uids[0];
            st.creatures
                .iter()
                .find(|c| c.id == uid.0)
                .and_then(|c| c.intent)
        };
        let died_with = intent_of(&m);
        assert!(died_with.is_some(), "the rat telegraphs");
        // The mirror plays the kill: its rat dies with that telegraph.
        if let Some(st) = m.state.as_mut() {
            let uid = st.state.enemy_uids[0];
            if let Some(c) = st.state.creatures.iter_mut().find(|c| c.id == uid.0) {
                c.current_hp = 0;
            }
        }
        // The board right after the kill: the rat dead, no telegraph.
        let mut world = wire.clone();
        world["encounter"][0]["hp"] = json!(0);
        world["encounter"][0]["is_dead"] = json!(true);
        world["encounter"][0]["intents"] = json!([]);
        m.reconcile(&world).expect("reconciles");
        assert_eq!(intent_of(&m), died_with, "after the kill");
        // Then the board drops the corpse.
        world["encounter"]
            .as_array_mut()
            .expect("encounter")
            .remove(0);
        m.reconcile(&world).expect("reconciles");
        assert_eq!(intent_of(&m), died_with, "after the board dropped it");
    }

    /// GH #1672: the game's skippable choose-a-card screen reads "pick none";
    /// the mirror parks it as the engine does, pick one with a skip.
    #[test]
    fn a_skippable_choose_a_card_screen_parks_as_pick_one_with_a_skip() {
        let deck: Vec<String> = vec!["StrikeIronclad".into()];
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&to_value(&cs).expect("wire")).expect("seeds");
        let options: Vec<Value> = (0..3)
            .map(|_| json!({"id": sts2_workbench::combat_import::registry_card_id("STRIKE_IRONCLAD", "t").expect("id"), "upgraded": false}))
            .collect();
        let origin = json!({"source": "FromChooseACardScreen"});
        m.inject_choice("add_generated_free_to_hand", options, 0, 1, true, origin)
            .expect("injects");
        let pc = m
            .mirrored()
            .expect("fight")
            .state
            .pending_choice
            .clone()
            .expect("parked");
        assert_eq!((pc.min_select, pc.max_select, pc.can_skip), (1, 1, true));
    }

    fn five_strikes() -> CombatState {
        let deck: Vec<String> = vec!["StrikeIronclad".into(); 10];
        sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds")
    }

    /// Seed 180000982: Heirloom Hammer parks an `AddGeneratedToHand` over
    /// HAND cards. The world's options get the parked hand locators, so the
    /// student reads them in hand (`choice_pile`, `hand_slot`), not as
    /// generated cards.
    #[test]
    fn a_generative_kind_over_hand_cards_is_located() {
        let cs = five_strikes();
        assert!(cs.player.hand.len() >= 2, "an opening hand");
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&to_value(&cs).expect("wire")).expect("seeds");
        let hand = m.mirrored().expect("fight").state.player.hand.clone();
        let at = |i: usize| json!({"id": "STRIKE_IRONCLAD", "upgraded": false, "pile": "hand", "index": i});
        m.parked = Some(json!({
            "type": "card_select", "kind": "add_generated_to_hand",
            "cards": [at(0), at(1)], "min_select": 1, "max_select": 1, "can_skip": false,
        }));
        let options = vec![json!({"id": "STRIKE_IRONCLAD", "upgraded": false}); 2];
        let origin = json!({"source": "FromHand", "source_model": "HEIRLOOM_HAMMER"});
        m.inject_choice("add_generated_to_hand", options, 1, 1, false, origin)
            .expect("injects");
        let pc = m
            .mirrored()
            .expect("fight")
            .state
            .pending_choice
            .clone()
            .expect("open");
        let uids: Vec<_> = pc.options.iter().map(|o| o.card_uid).collect();
        assert_eq!(uids, vec![Some(hand[0]), Some(hand[1])]);
        assert!(pc
            .options
            .iter()
            .all(|o| o.source_pile == Some(sts2_core::cards::PileKind::Hand)));
    }

    /// A truly generated offer keeps no locator.
    #[test]
    fn a_generated_offer_stays_unlocated() {
        let cs = five_strikes();
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&to_value(&cs).expect("wire")).expect("seeds");
        m.parked = Some(json!({
            "type": "card_select", "kind": "add_generated_free_to_hand",
            "cards": [{"id": "FINESSE", "upgraded": false}], "min_select": 1, "max_select": 1,
            "can_skip": true,
        }));
        let options = vec![json!({"id": "FINESSE", "upgraded": false})];
        m.inject_choice(
            "add_generated_free_to_hand",
            options,
            1,
            1,
            true,
            Value::Null,
        )
        .expect("injects");
        let pc = m
            .mirrored()
            .expect("fight")
            .state
            .pending_choice
            .clone()
            .expect("open");
        assert_eq!(pc.options[0].card_uid, None);
    }

    /// Seed 180001961: the world's board shows a card in play (Imbued's
    /// turn-1 autoplay of Splash) that the carried fight never played. The
    /// in-flight skip keeps the carried board; adopting the unseen play seeds
    /// from the world's.
    #[test]
    fn a_play_the_mirror_never_saw_is_adopted() {
        let cs = five_strikes();
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        let seeded = to_value(&cs).expect("wire");
        m.seed(&seeded).expect("seeds");
        let mut world = seeded;
        let in_play = world["deck"]["hand"]
            .as_array_mut()
            .expect("hand")
            .remove(0);
        world["deck"]["play_pile"] = json!([in_play]);
        m.reconcile(&world).expect("reconciles");
        let before = m.mirrored().expect("fight").state.player.hand.len();
        assert_eq!(
            before,
            cs.player.hand.len(),
            "the in-flight skip kept the carried hand"
        );
        assert!(m.adopt_unseen_play(&world).expect("adopts"));
        let after = &m.mirrored().expect("fight").state.player;
        assert_eq!((after.hand.len(), after.play.len()), (before - 1, 1));
        // Nothing more to adopt once the mirror holds the world's board.
        assert!(!m.adopt_unseen_play(&world).expect("no-op"));
    }

    /// A choice the engine parked itself is never re-seeded away.
    #[test]
    fn a_parked_choice_is_not_adopted_over() {
        let cs = five_strikes();
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        let mut world = to_value(&cs).expect("wire");
        m.seed(&world).expect("seeds");
        m.parked = Some(json!({"kind": "exhaust_from_hand", "cards": []}));
        let in_play = world["deck"]["hand"]
            .as_array_mut()
            .expect("hand")
            .remove(0);
        world["deck"]["play_pile"] = json!([in_play]);
        assert!(!m.adopt_unseen_play(&world).expect("no-op"));
    }

    /// GH #1675: a damage wake stuns the Lagavulin Matriarch into SLASH; the
    /// game names that stun `STUNNED`, and a rebuild keeps it, so the
    /// forecast reads the slash next and no further sleep.
    #[test]
    fn a_woken_matriarch_rebuilds_stunned_into_her_slash() {
        let deck: Vec<String> = vec!["StrikeIronclad".into()];
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "LAGAVULIN_MATRIARCH_BOSS",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut wire = to_value(&cs).expect("wire");
        wire["encounter"][0]["block"] = json!(0);
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let forecast = |m: &CombatMirror| -> Vec<String> {
            let fc = sts2_wire::emit::combat::enemy_intent_forecast(
                &m.mirrored().expect("fight").state,
                3,
            )
            .expect("forecast");
            fc[0]
                .as_array()
                .expect("steps")
                .iter()
                .map(|s| s["move_id"].as_str().unwrap_or("").to_owned())
                .collect()
        };
        // The hit wakes her: stunned this turn, then the slash.
        m.advance(&json!({"action": "play", "card_index": 0, "target_index": 0}))
            .expect("plays");
        let woken = forecast(&m);
        assert_eq!(woken.first().map(String::as_str), Some("SLASH"));
        // The host's board: the game's own name for the stun, and its move
        // machine (`AppendMoveMachineState`): the stun resumes SLASH_MOVE.
        let mut world = to_value(&m.mirrored().expect("fight").state).expect("wire");
        world["encounter"][0]["move_state"] = json!("STUNNED");
        world["encounter"][0]["ai"] = json!({
            "log": ["SLEEP_MOVE"], "performed_first_move": true, "performed": false,
            "forced_next": "SLASH_MOVE",
        });
        m.reconcile(&world).expect("reconciles");
        assert_eq!(forecast(&m), woken, "reconciled");
        let mut fresh = CombatMirror::new(Policy::Observe, "t", false);
        fresh.seed(&world).expect("seeds");
        assert_eq!(forecast(&fresh), woken, "seeded from the board");
    }

    /// GH #1657: the game shows a buff without a number, so the board says
    /// 0; the rebuilt fight telegraphs the amount the move declares, as the
    /// engine the student trained on does (Nibbit HISS: +3 Strength at A10).
    #[test]
    fn a_buff_shown_without_a_number_gets_its_moves_amount() {
        let deck: Vec<String> = vec!["StrikeIronclad".into()];
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "NIBBITS_WEAK",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut cs = cs;
        for c in &mut cs.creatures {
            if let Some(ai) = c.ai.as_mut() {
                ai.force_current_state_unlogged("HISS");
                c.intent = ai.current_move().intents.first().copied();
            }
        }
        let declared = cs
            .creatures
            .iter()
            .find_map(|c| c.ai.as_ref().map(|_| c.intent))
            .flatten();
        assert!(
            matches!(declared, Some(sts2_core::Intent::Buff { amount, .. }) if amount > 0),
            "{declared:?}"
        );
        let mut wire = to_value(&cs).expect("wire");
        // What the game's board shows: a buff, no number.
        for i in wire["encounter"][0]["intents"]
            .as_array_mut()
            .expect("intents")
        {
            i.as_object_mut().expect("intent").remove("amount");
        }
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let seeded = m
            .mirrored()
            .expect("fight")
            .state
            .creatures
            .iter()
            .find_map(|c| c.ai.as_ref().map(|_| c.intent))
            .flatten();
        assert_eq!(seeded, declared, "seeded");
        m.reconcile(&wire).expect("reconciles");
        let rebuilt = m
            .mirrored()
            .expect("fight")
            .state
            .creatures
            .iter()
            .find_map(|c| c.ai.as_ref().map(|_| c.intent))
            .flatten();
        assert_eq!(rebuilt, declared, "rebuilt");
    }

    /// GH #1668: the game grows a played Rampage's base damage
    /// (`Rampage.cs:54`); the engine the student trained on keeps the
    /// printed 10 and the growth in `RampageDamage`.
    #[test]
    fn a_grown_rampage_is_held_as_the_engine_holds_it() {
        use sts2_core::cards::{CardId, VarKey};
        let deck: Vec<String> = vec!["Rampage".into()];
        let mut cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "NIBBITS_WEAK",
            10,
            None,
            None,
        )
        .expect("builds");
        for c in &mut cs.cards {
            if c.card_id == CardId::Rampage {
                c.base_damage = 15;
            }
        }
        let wire = to_value(&cs).expect("wire");
        let held = |m: &CombatMirror| {
            m.mirrored()
                .expect("fight")
                .state
                .cards
                .iter()
                .find(|c| c.card_id == CardId::Rampage)
                .map(|c| (c.base_damage, c.combat_var(VarKey::RampageDamage)))
        };
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        assert_eq!(held(&m), Some((10, 5)), "seeded");
        m.reconcile(&wire).expect("reconciles");
        assert_eq!(held(&m), Some((10, 5)), "rebuilt");
    }

    /// GH #1772: an OBSERVE reseed after a kill the board dropped still
    /// salvages the cards' hidden vars.
    #[test]
    fn a_reseed_after_a_dropped_kill_keeps_the_cards_hidden_vars() {
        use sts2_core::cards::VarKey;
        let deck: Vec<String> = ["StrikeIronclad", "DefendIronclad"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut wire = to_value(&cs).expect("wire");
        wire["encounter"][0]["hp"] = json!(0);
        wire["encounter"][0]["is_dead"] = json!(true);
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        {
            let st = &mut m.state.as_mut().expect("fight").state;
            let uid = st.player.hand[0];
            let card = st.cards.iter_mut().find(|c| c.uid == uid).expect("card");
            card.set_combat_var(VarKey::DampenDowngradedMarker, 1);
        }
        // The game's next board: the first rat died and is gone, and the
        // player's HP disagrees, so the mirror reseeds.
        let mut world = wire.clone();
        world["encounter"]
            .as_array_mut()
            .expect("encounter")
            .remove(0);
        world["player"]["hp"] = json!(70);
        let _ = m.reconcile(&world);
        assert!(!m.take_divergences().is_empty(), "the HP desync reseeds");
        let st = &m.mirrored().expect("reseeded").state;
        let uid = st.player.hand[0];
        let card = st.cards.iter().find(|c| c.uid == uid).expect("card");
        assert_eq!(card.combat_var(VarKey::DampenDowngradedMarker), 1);
    }

    /// An OBSERVE reseed keeps the applier of the player's Dampen, so the
    /// debuff still ends when its caster dies (corpus r6 seed 180001115:
    /// a Flail Knight HP desync reseeded the Knights fight, Magi Knight's
    /// Dampen outlived the Knight, and the world's re-upgraded cards broke
    /// the next card_select's injection).
    #[test]
    fn a_reseed_keeps_the_appliers_of_the_players_debuffs() {
        use sts2_core::PowerKind;
        let deck: Vec<String> = ["StrikeIronclad", "DefendIronclad"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let wire = to_value(&cs).expect("wire");
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        let dampen_applier = |st: &CombatState| {
            let player = st.player.creature_uid.expect("player creature");
            st.creatures
                .iter()
                .find(|c| c.id == player.0)
                .map(|c| c.powers.clone())
                .unwrap_or_default()
                .into_iter()
                .chain(st.player.powers.iter().copied())
                .find_map(|pu| {
                    st.powers
                        .get_by_id(pu.0)
                        .filter(|p| p.kind == PowerKind::Dampen)
                        .map(|p| p.applier)
                })
        };
        {
            let st = &mut m.state.as_mut().expect("fight").state;
            let player = st.player.creature_uid.expect("player creature");
            let caster = st.enemy_uids[1];
            sts2_core::card_effects::apply_power_to_creature_with_applier(
                st,
                player,
                PowerKind::Dampen,
                1,
                Some(caster),
            );
            assert_eq!(dampen_applier(st), Some(Some(caster)), "armed");
        }
        // The game's next board shows the Dampen and an HP the mirror did not
        // predict, so the mirror reseeds.
        let mut world = to_value(&m.mirrored().expect("fight").state).expect("wire");
        world["player"]["hp"] = json!(70);
        let _ = m.reconcile(&world);
        assert!(!m.take_divergences().is_empty(), "the HP desync reseeds");
        let st = &m.mirrored().expect("reseeded").state;
        assert_eq!(
            dampen_applier(st),
            Some(Some(st.enemy_uids[1])),
            "the reseeded Dampen still names its caster"
        );
    }

    /// A fight under Void Form (2 free plays), its hand, and the mirror
    /// seeded on it. The rebuild reads `cardsPlayedThisTurn` from nowhere:
    /// the board never shows it.
    fn void_form_fight() -> CombatMirror {
        let deck: Vec<String> = [
            "Discovery",
            "DefendIronclad",
            "DefendIronclad",
            "DefendIronclad",
            "DefendIronclad",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "TWO_TAILED_RATS_NORMAL",
            10,
            None,
            None,
        )
        .expect("builds");
        let mut wire = to_value(&cs).expect("wire");
        wire["player"]["powers"] = json!([{"name": "VOID_FORM", "amount": 2}]);
        let mut m = CombatMirror::new(Policy::Observe, "t", false);
        m.seed(&wire).expect("seeds");
        m
    }

    /// Play the first hand card of `id`; the engine's step result.
    fn play_first(m: &mut CombatMirror, id: sts2_core::cards::CardId) -> &'static str {
        let st = &m.mirrored().expect("fight").state;
        let slot = st
            .player
            .hand
            .iter()
            .position(|uid| st.cards.iter().any(|c| c.uid == *uid && c.card_id == id))
            .expect("in hand");
        m.advance(&json!({"action": "play", "card_index": slot, "target_index": null}))
            .expect("plays")
    }

    /// The game's board of the mirrored fight: every power counter it does
    /// not report left out, as the host leaves Void Form's.
    fn board_without_power_counters(m: &CombatMirror) -> Value {
        let mut world = to_value(&m.mirrored().expect("fight").state).expect("wire");
        for p in world["player"]["powers"].as_array_mut().expect("powers") {
            p.as_object_mut().expect("power").remove("counter");
        }
        world
    }

    /// Spire Bench seeds 180000322, 180000516, 180000590, 180001456,
    /// 180000358: a choice-opening card played under Void Form past its free
    /// plays parks the card's tail. Verified against a bare rebuild, every
    /// hand card read free, `$.deck.hand` diffed, and the reseed deleted the
    /// suspended card, its AfterCardPlayed with it: Void Form's count stayed
    /// one short for the rest of the turn.
    #[test]
    fn a_card_parked_past_void_forms_free_plays_survives_the_verify() {
        use sts2_core::cards::CardId;
        let mut m = void_form_fight();
        let played = |m: &CombatMirror| {
            sts2_core::combat_loop::void_form_cards_played_this_turn(
                &m.mirrored().expect("fight").state,
            )
        };
        play_first(&mut m, CardId::DefendIronclad);
        play_first(&mut m, CardId::DefendIronclad);
        assert_eq!(played(&m), 2, "the free plays are spent");
        assert_eq!(play_first(&mut m, CardId::Discovery), "pending_choice");
        assert!(m.mirrored().expect("fight").state.pending_play.is_some());
        let desyncs = m
            .reconcile(&board_without_power_counters(&m))
            .expect("reconciles");
        assert!(desyncs.is_empty(), "{desyncs:?}");
        assert!(m.take_divergences().is_empty(), "no reseed");
        assert!(
            m.mirrored().expect("fight").state.pending_play.is_some(),
            "the suspended Discovery survives the verify"
        );
    }

    /// The rebuild path's diff, same cause: past the free plays a bare
    /// rebuild read the hand free and reseeded at every reconcile, and the
    /// reseed drops every enemy-side carry (TestSubject's PAINFUL_STABS
    /// escalation restarted at 0: seeds 180000590, 180001456, 180000358).
    #[test]
    fn a_rebuild_past_void_forms_free_plays_reads_the_hand_at_cost() {
        use sts2_core::cards::CardId;
        let mut m = void_form_fight();
        // The first reconcile after a seed excuses the roster; get it done.
        let world = board_without_power_counters(&m);
        assert!(m.reconcile(&world).expect("reconciles").is_empty());
        play_first(&mut m, CardId::DefendIronclad);
        play_first(&mut m, CardId::DefendIronclad);
        let desyncs = m
            .reconcile(&board_without_power_counters(&m))
            .expect("reconciles");
        assert!(desyncs.is_empty(), "{desyncs:?}");
        assert!(m.take_divergences().is_empty(), "no reseed");
        assert_eq!(
            sts2_core::combat_loop::void_form_cards_played_this_turn(
                &m.mirrored().expect("fight").state
            ),
            2
        );
    }

    #[test]
    fn observe_divergence_taints_drops_and_queues_the_report() {
        let mut m = CombatMirror::new(Policy::Observe, "lbl", false);
        m.diverge(
            "boom".into(),
            "desync",
            None,
            vec!["$.player.hp".into()],
            None,
            Map::new(),
        )
        .unwrap();
        assert!(m.tainted);
        assert_eq!(m.drop_reason, "boom");
        let divs = m.take_divergences();
        assert_eq!(divs.len(), 1);
        assert_eq!(divs[0].paths, ["$.player.hp"]);
        assert_eq!(divs[0].report["label"], json!("lbl"));
        assert_eq!(divs[0].report["policy"], json!("observe"));

        let mut strict = CombatMirror::new(Policy::Strict, "s", false);
        let err = strict
            .diverge("bad".into(), "desync", None, vec![], None, Map::new())
            .unwrap_err();
        assert!(matches!(err, MirrorError::Diverged(d) if d.message == "bad"));
        assert!(!strict.tainted);
    }
}
