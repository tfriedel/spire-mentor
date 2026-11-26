//! Instinct in a fight (GH #1426; was `InstinctAdvisor._advise_combat`,
//! `_combat_plan` and `realgame_instinct.combat_state_for`).
//!
//! The student answers a `combat_action` board on the grafted run, standing
//! in the fight the search would build (the fight mirror's carried state
//! when it follows this fight), and draws its greedy rollout through the
//! turn as the plan.

use serde_json::{json, Value};
use sts2_core::parity::combat_state_to_bridge_state;
use sts2_core::run_manager::RunManager;
use sts2_core::runstate::RoomType;
use sts2_core::CombatState;
use sts2_e2e::env::{ApplyKind, Driver, Next};
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::TokCache;

use crate::advisor_mirror::AdvisorMirror;
use crate::board::{self, BoardError};
use crate::board_targets::BoardTargets;
use crate::combat_advice::instinct_combat_advice;
use crate::combat_mirror::{build_state, MirrorError, Mirrored};
use crate::headless;
use crate::i18n::tr;
use crate::instinct::{combat_form, evaluate, Evaluated};
use crate::py::get;
use crate::search::RecommendConfig;
use crate::turn_plan::{first_step, same_move, PlanWalk, MAX_PLAN_STEPS};

/// The character a request that names none is rebuilt as.
pub const INSTINCT_CHARACTER: &str = "Ironclad";

/// Why Instinct has no answer for a fight decision.
#[derive(Debug)]
pub enum InstinctCombatError {
    /// The player-facing reason (`InstinctUnavailable`); the search answers.
    Unavailable(String),
    /// The board has no engine fight.
    Board(BoardError),
    /// The rebuilt wire does not build a state.
    State(MirrorError),
}

impl std::fmt::Display for InstinctCombatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(m) => f.write_str(m),
            Self::Board(e) => write!(f, "{e}"),
            Self::State(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for InstinctCombatError {}

/// `_room_kind`: the board's room as `debug_enter_combat` names it.
#[must_use]
pub fn room_kind(state: &Value) -> RoomType {
    match crate::py::str_of(get(state, "room_type"))
        .to_lowercase()
        .as_str()
    {
        "elite" => RoomType::Elite,
        "boss" => RoomType::Boss,
        _ => RoomType::Monster,
    }
}

/// `combat_state_for`: the engine fight behind a `combat_action` board, as
/// the search builds it, and its wire. The mirror's carried state when it
/// follows this fight and is not parked on a choice; feeding the mirror the
/// board also keeps it in step while Instinct, not the search, answers.
///
/// # Errors
///
/// A board with no engine fight, a rebuild that does not build.
pub fn combat_fight(
    state: &Value,
    cfg: &RecommendConfig,
    mirror: Option<&mut AdvisorMirror>,
) -> Result<(CombatState, Value), InstinctCombatError> {
    combat_fight_remapped(state, cfg, mirror).map(|(fight, wire, _)| (fight, wire))
}

/// [`combat_fight`], plus the enemy uid renames the mirror's rebuilds made
/// since the last board (GH #1651): the intent log follows them.
///
/// # Errors
///
/// As [`combat_fight`].
pub fn combat_fight_remapped(
    state: &Value,
    cfg: &RecommendConfig,
    mirror: Option<&mut AdvisorMirror>,
) -> Result<(CombatState, Value, Option<Vec<(u32, u32)>>), InstinctCombatError> {
    let run_state = get(state, "run_state");
    let character =
        headless::character_display_name(get(run_state, "character")).unwrap_or(INSTINCT_CHARACTER);
    let (wire, _warnings) = board::combat_wire(
        state,
        Some(character),
        board::run_ascension(run_state, cfg.ascension),
        cfg.combat_id,
        cfg.seed,
    )
    .map_err(InstinctCombatError::Board)?;
    if let Some(m) = mirror {
        if m.on_combat_board(&wire, state) {
            let remap = m.take_enemy_remap();
            if let Some(s) = m
                .live_state()
                .filter(|s| sts2_wire::emit::pending_choice(s).is_none())
            {
                return Ok((s.clone(), wire, remap));
            }
        }
    }
    let Mirrored { state: fight, .. } = build_state(&wire).map_err(InstinctCombatError::State)?;
    Ok((fight, wire, None))
}

/// What Instinct answered for a fight decision.
#[derive(Debug, Clone)]
pub struct InstinctCombat {
    pub advice: Value,
    /// The win probe's read of the position, when the weights carry one.
    pub estimate: Option<Value>,
    /// Why the turn plan was left out (`instinct: turn plan failed`).
    pub plan_failed: Option<String>,
}

/// `enter_live_combat`: install the live fight on the grafted run, naming
/// its encounter when the mod sent one (`encounter_id`, on the board or on an
/// in-combat screen's embedded `combat` board): the student reads it (RUNCTX
/// `X_ID_AUX`, GH #1197), and that cell alone can move a pick. An event
/// fight's id stages that event's fight (GH #1477). An id the engine has no
/// fight for leaves the run without one; the returned note says so.
pub fn enter_live_combat(
    rm: &mut RunManager,
    mut fight: CombatState,
    state: &Value,
    room: RoomType,
) -> Option<String> {
    let id =
        [state, get(state, "combat")]
            .into_iter()
            .find_map(|src| match get(src, "encounter_id") {
                Value::String(id) if !id.is_empty() => Some(id.clone()),
                _ => None,
            });
    // Named first: a bad id writes nothing, and entering the fight leaves the
    // encounter alone.
    let note = id.and_then(|id| {
        sts2_wire::graft::set_combat_encounter(rm, &id)
            .err()
            .map(|e| format!("instinct: debug_enter_combat: {e}"))
    });
    if rm
        .run_state
        .event_combat
        .as_ref()
        .is_some_and(|r| r.encounter == sts2_core::events::EventEncounter::DenseVegetation)
    {
        relabel_dense_vegetation_wrigglers(&mut fight);
    }
    return_stolen_cards(rm, &fight);
    rm.debug_enter_combat(fight, room);
    note
}

/// C# `SwipePower.Steal` takes a stolen card out of the deck at once; the
/// engine keeps it there until the fight ends and leaves it out of the deck it
/// shows. The game's deck lost it already, so the rebuild hands each card a
/// thief holds back to the deck, or it is counted out twice (GH #1694).
fn return_stolen_cards(rm: &mut RunManager, fight: &CombatState) {
    for s in fight.swipe_stolen.iter().filter(|s| !s.reclaimed) {
        let spec = sts2_wire::graft::CardSpec {
            id: s.card_id.to_python_name().to_string(),
            upgraded: s.upgraded,
            enchantment: None,
        };
        if let Ok(card) = sts2_wire::graft::build_card(&spec, rm.game_version) {
            rm.run_state.master_deck.push(card);
        }
    }
}

/// The engine spawns the Dense Vegetation event's four Wrigglers as their own
/// species (C# `DenseVegetationEventEncounter`: no spawn-turn stun); the game
/// names them Wrigglers, so a fight rebuilt from its board does too. The
/// student reads the species the engine would spawn (GH #1688).
fn relabel_dense_vegetation_wrigglers(fight: &mut CombatState) {
    for c in &mut fight.creatures {
        if c.species == Some(sts2_core::MonsterKind::Wriggler) {
            c.species = Some(sts2_core::MonsterKind::DenseVegetationWriggler);
        }
    }
}

/// `_advise_combat`: the student's answer for a `combat_action` board on the
/// grafted run `rm` (`graft_run`, entered into the fight here), with this
/// run's `driver`. The fight is entered as [`room_kind`] before the student
/// reads it, so an [`InstinctCombatError::Unavailable`] still stood in it.
///
/// # Errors
///
/// As [`combat_fight`]; the student cannot read the position or its answer
/// does not map onto the live hand ([`InstinctCombatError::Unavailable`]).
/// `log` collects the info lines on the way.
#[allow(clippy::too_many_arguments)]
pub fn advise_combat(
    net: &E2eNet,
    driver: &mut Driver,
    cache: &TokCache,
    rm: &mut RunManager,
    state: &Value,
    cfg: &RecommendConfig,
    mirror: Option<&mut AdvisorMirror>,
    log: &mut Vec<String>,
) -> Result<InstinctCombat, InstinctCombatError> {
    let (fight, wire, remap) = combat_fight_remapped(state, cfg, mirror)?;
    if let Some(pairs) = remap {
        driver.remap_enemies(&pairs);
    }
    // Before the fight moves into the run: a mirror's carried fight keeps
    // corpses the game removed, so its enemy slots are not the board's.
    let targets = BoardTargets::of(&fight);
    if let Some(note) = enter_live_combat(rm, fight, state, room_kind(state)) {
        log.push(note);
    }
    let expect = ["COMBAT_TURN".to_string(), "COMBAT_CHOICE".to_string()];
    let Evaluated { pick, estimate } =
        evaluate(net, driver, cache, rm, &expect).map_err(InstinctCombatError::Unavailable)?;
    let mut advice = instinct_combat_advice(&pick, state, &wire, &targets)
        .map_err(InstinctCombatError::Unavailable)?;
    let best = get(&advice, "best").clone();
    // A drawing must never cost the advice.
    let mut plan_failed = None;
    match combat_plan(net, driver, cache, rm, state, &wire, &targets, &pick, &best) {
        Ok(plan) if !plan.is_empty() => advice["plan"] = Value::Array(plan),
        Ok(_) => {}
        Err(why) => plan_failed = Some(why),
    }
    Ok(InstinctCombat {
        advice,
        estimate,
        plan_failed,
    })
}

fn apply_kind(name: &str) -> ApplyKind {
    if name == "combat" {
        ApplyKind::Combat
    } else {
        ApplyKind::Run { strict: true }
    }
}

const fn kind_name(kind: ApplyKind) -> &'static str {
    match kind {
        ApplyKind::Combat => "combat",
        ApplyKind::Run { .. } => "run",
    }
}

/// `_combat_plan`: the student's greedy rollout through this turn.
///
/// Played on a clone of the grafted run with a copy of this run's driver,
/// the student deciding every later move exactly as it would live; mapped
/// onto the current hand and cut by [`PlanWalk`]'s rules (a draw, a choice,
/// an unmappable move end it). Its first step is `best`; when `best` is not
/// the student's own first move (the mod's veto took it), the plan is
/// `best` alone.
///
/// # Errors
///
/// The student could not evaluate a later position, or named a class the
/// decision does not have: the advice goes without a plan.
#[allow(clippy::too_many_arguments)]
pub fn combat_plan(
    net: &E2eNet,
    driver: &Driver,
    cache: &TokCache,
    rm: &RunManager,
    state: &Value,
    wire: &Value,
    targets: &BoardTargets,
    pick: &crate::instinct::Pick,
    best: &Value,
) -> Result<Vec<Value>, String> {
    if !best.is_object() {
        return Ok(Vec::new());
    }
    let lone = first_step(best, state);
    let alone = || lone.clone().into_iter().collect::<Vec<_>>();
    let dtype = get(&pick.decision, "decision_type").as_str();
    if !matches!(dtype, Some("COMBAT_TURN" | "FORCED")) {
        return Ok(alone());
    }
    let mut walk = PlanWalk::new(wire, state).with_targets(targets.clone());
    let mut sim = rm.clone();
    let mut probe = driver.clone();
    let mut probe_cache = cache.clone();
    for ply in 0..MAX_PLAN_STEPS {
        let (run_action, kind) = match probe.peek(&mut sim) {
            Next::Forced(f) => {
                if f.kind != ApplyKind::Combat {
                    break;
                }
                (f.action, kind_name(f.kind))
            }
            Next::Decision(d) => {
                if get(&d.to_value(), "decision_type").as_str() != Some("COMBAT_TURN") {
                    break;
                }
                let class = if ply == 0 {
                    pick.best
                } else {
                    let ev = sts2_e2e::tokens::encode_for(net, &d, &sim, &mut probe_cache)
                        .and_then(|row| net.evaluate(&row.into()))?;
                    sts2_e2e::net::argmax(&ev.logits)
                };
                let (action, kind) = d.class_action(class)?;
                (action, kind_name(kind))
            }
            Next::Terminal(_) => break,
        };
        let action = combat_form(&run_action);
        let Some(step) = walk.to_step(&action) else {
            break;
        };
        if ply == 0 && !same_move(&step, best) {
            break;
        }
        walk.push(step);
        if walk.done {
            break;
        }
        let before = sim
            .combat
            .as_ref()
            .map_or_else(|| json!({}), combat_state_to_bridge_state);
        if probe
            .apply(&mut sim, &run_action, apply_kind(kind))
            .is_err()
        {
            walk.steps.pop();
            break;
        }
        let Some(after) = sim.combat.as_ref() else {
            break; // the fight ended on this move
        };
        let result = sts2_wire::emit::pending_choice(after).map(|_| "pending_choice");
        walk.advance(
            &action,
            &before,
            &combat_state_to_bridge_state(after),
            result,
        );
        if walk.done {
            break;
        }
    }
    Ok(if walk.steps.is_empty() {
        alone()
    } else {
        walk.steps
    })
}

/// What Instinct picked with on an in-fight choice: its first pick's
/// decision and class probabilities (for the option rows), why it could not
/// pick, and the info lines on the way.
#[derive(Debug, Default)]
pub struct ChoicePicks {
    pub first: Option<(Value, Vec<f64>)>,
    /// Each pick of the last picking, in pick order: the option and the
    /// net's probability for it at its OWN step, given the picks before it
    /// (1.0 for a forced one). A multi-card selection's rows read these
    /// (GH #1770); the first step's distribution only ranks pick 1.
    pub steps: Vec<(i64, f64)>,
    /// The step where the net chose to stop (`done`): its decision and
    /// class probabilities, how likely each card left out was to be taken
    /// next. `None` when the picking ran out of cards or picks instead.
    pub stop: Option<(Value, Vec<f64>)>,
    pub failed: Vec<String>,
    pub log: Vec<String>,
}

/// A picker error as `(Python class name, message)`, the way the mirror's
/// decline quotes it.
pub type PickerError = (String, String);

fn unavailable(picks: &mut ChoicePicks, why: String) -> PickerError {
    picks.failed.push(why.clone());
    ("InstinctUnavailable".into(), why)
}

/// `_advise_combat_select`'s picker: the student's picks on the parked
/// `fork`, one pick at a time as the engine decomposes a multi-pick, on the
/// run `graft` rebuilds from `state`, entered into `fork` as `room`.
/// `driver_for(seed)` is a copy of this run's driver.
///
/// # Errors
///
/// No run to rebuild (`InstinctUnavailable`, also kept in
/// [`ChoicePicks::failed`]), a move the engine refuses (`ValueError`), a
/// position the student cannot evaluate (`RuntimeError`).
#[allow(clippy::too_many_arguments)]
pub fn pick_choice(
    net: Option<&E2eNet>,
    driver_for: &mut dyn FnMut(i64) -> Result<(Driver, TokCache), PickerError>,
    state: &Value,
    version: &str,
    room: RoomType,
    fork: &CombatState,
    max_select: i64,
    picks_out: &mut ChoicePicks,
) -> Result<Vec<i64>, PickerError> {
    use crate::graft::{graft_run, GraftError};
    let mut rm = match graft_run(state, version) {
        Ok((rm, _notes)) => rm,
        Err(GraftError::Unavailable(m)) => return Err(unavailable(picks_out, m)),
        Err(GraftError::Value(m)) => return Err(("ValueError".into(), m)),
    };
    if let Some(note) = enter_live_combat(&mut rm, fork.clone(), state, room) {
        picks_out.log.push(note);
    }
    let Some(net) = net else {
        return Err(unavailable(picks_out, tr("no Instinct net loaded").into()));
    };
    let (mut probe, mut cache) = driver_for(i64::from(rm.seed))?;
    student_choice_picks(net, &mut probe, &mut cache, &mut rm, max_select, picks_out)
}

/// The student's picks on the in-fight choice `rm` stands on, one pick at a
/// time as the engine decomposes a multi-pick (the native run's
/// COMBAT_CHOICE decisions). `probe` and `cache` are a copy of the run's
/// driver.
///
/// # Errors
///
/// A move the engine refuses (`ValueError`), a position the student cannot
/// evaluate (`RuntimeError`).
pub fn student_choice_picks(
    net: &E2eNet,
    probe: &mut Driver,
    cache: &mut TokCache,
    rm: &mut RunManager,
    max_select: i64,
    picks_out: &mut ChoicePicks,
) -> Result<Vec<i64>, PickerError> {
    let value_err = |m: String| ("ValueError".to_string(), m);
    let mut picks: Vec<i64> = Vec::new();
    picks_out.steps.clear();
    picks_out.stop = None;
    for _ in 0..=max_select.max(1) {
        match probe.peek(rm) {
            Next::Forced(f) => {
                let option = get(&f.action, "option");
                if option.is_i64() && i64::try_from(picks.len()).unwrap_or(i64::MAX) < max_select {
                    let option = option.as_i64().unwrap_or(0);
                    picks.push(option);
                    picks_out.steps.push((option, 1.0));
                }
                let kind = match f.kind {
                    ApplyKind::Combat => ApplyKind::Combat,
                    ApplyKind::Run { .. } => ApplyKind::Run { strict: false },
                };
                probe.apply(rm, &f.action, kind).map_err(value_err)?;
                // A forced pick that fills the screen ends it: the next
                // COMBAT_CHOICE is a LATER screen (a second Burning Pact),
                // and asking the net there dumped a spurious row and set the
                // first pick from it (seed 180000859).
                if i64::try_from(picks.len()).unwrap_or(i64::MAX) >= max_select {
                    break;
                }
            }
            Next::Decision(d) => {
                let dec = d.to_value();
                if get(&dec, "decision_type").as_str() != Some("COMBAT_CHOICE") {
                    break;
                }
                let ev = sts2_e2e::tokens::encode_for(net, &d, rm, cache)
                    .and_then(|row| net.evaluate(&row.into()))
                    .map_err(|m| ("RuntimeError".to_string(), m))?;
                let ci = sts2_e2e::net::argmax(&ev.logits);
                let probs = sts2_e2e::net::softmax(&ev.logits);
                if picks_out.first.is_none() {
                    picks_out.first = Some((dec.clone(), probs.clone()));
                }
                let selection: Vec<i64> = crate::py::list(
                    crate::py::as_list(get(&dec, "classes"))
                        .get(ci)
                        .and_then(|c| crate::py::as_list(get(c, "actions")).first())
                        .unwrap_or(&Value::Null),
                    "selection",
                )
                .iter()
                .map(crate::py::int_of)
                .collect();
                if selection.is_empty() {
                    picks_out.stop = Some((dec, probs));
                    break; // done / skip
                }
                let p = probs.get(ci).copied().unwrap_or(0.0);
                picks_out.steps.extend(selection.iter().map(|&i| (i, p)));
                picks.extend(selection);
                let (action, kind) = d.class_action(ci).map_err(value_err)?;
                // The kind crosses by name, as the Python driver applied it.
                probe
                    .apply(rm, &action, apply_kind(kind_name(kind)))
                    .map_err(value_err)?;
                if i64::try_from(picks.len()).unwrap_or(i64::MAX) >= max_select {
                    break;
                }
            }
            Next::Terminal(_) => break,
        }
    }
    if max_select > 0 {
        picks.truncate(usize::try_from(max_select).unwrap_or(usize::MAX));
    }
    picks_out.steps.truncate(picks.len());
    Ok(picks)
}

/// `_annotate_choice_probs`: the first pick's class probabilities on the
/// choice's option rows. A class's members are interchangeable options (two
/// identical cards); each option row carries its class's probability.
pub fn annotate_choice_probs(advice: &mut Value, decision: &Value, probs: &[f64]) {
    let mut by_option: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
    for cls in crate::py::list(decision, "classes") {
        let ci = usize::try_from(crate::py::int_of(get(cls, "class"))).unwrap_or(usize::MAX);
        let p = probs.get(ci).copied().unwrap_or(0.0);
        for a in crate::py::list(cls, "actions") {
            for i in crate::py::list(a, "selection") {
                by_option.entry(crate::py::int_of(i)).or_insert(p);
            }
        }
    }
    if let Some(options) = advice.get_mut("options").and_then(Value::as_array_mut) {
        for (pos, opt) in options.iter_mut().enumerate() {
            let Some(p) = i64::try_from(pos).ok().and_then(|k| by_option.get(&k)) else {
                continue;
            };
            let r = json!(crate::py::round(*p, 4));
            opt["visit_share"] = r.clone();
            opt["prob"] = r;
        }
    }
}

/// A multi-card selection's rows (GH #1770). The net picks such a set one
/// card at a time, so the first step's distribution ranks only the first
/// card: on Gambler's Brew it put 100% on both Defends (one class, either
/// copy) and ~0% on the other three cards it went on to pick, which the
/// panel then labelled "Best". Each picked row carries instead the net's
/// probability for that pick at its own step, and each card left out the
/// probability the net would have taken it rather than stop (the `done`
/// step's class probability), when the net stopped by choice.
pub fn annotate_set_probs(
    advice: &mut Value,
    steps: &[(i64, f64)],
    stop: Option<&(Value, Vec<f64>)>,
) {
    let mut by_option: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
    for &(option, p) in steps {
        by_option.entry(option).or_insert(p);
    }
    if let Some((decision, probs)) = stop {
        for cls in crate::py::list(decision, "classes") {
            let ci = usize::try_from(crate::py::int_of(get(cls, "class"))).unwrap_or(usize::MAX);
            let p = probs.get(ci).copied().unwrap_or(0.0);
            for a in crate::py::list(cls, "actions") {
                for i in crate::py::list(a, "selection") {
                    by_option.entry(crate::py::int_of(i)).or_insert(p);
                }
            }
        }
    }
    let picked: std::collections::BTreeSet<i64> = steps.iter().map(|&(o, _)| o).collect();
    if let Some(options) = advice.get_mut("options").and_then(Value::as_array_mut) {
        for (pos, opt) in options.iter_mut().enumerate() {
            let Ok(k) = i64::try_from(pos) else { continue };
            // A row the answer does not take shows no pick probability.
            if picked.contains(&k) && !crate::py::truthy(get(opt, "is_best")) {
                continue;
            }
            let Some(p) = by_option.get(&k) else { continue };
            let r = json!(crate::py::round(*p, 4));
            opt["visit_share"] = r.clone();
            opt["prob"] = r;
        }
    }
}

/// The note on an advice Instinct picked on the fight mirror's fork.
pub const CHOICE_NOTE: &str = "Instinct (e2e net, no search) · choice";

/// `_advise_combat_select`: an in-combat `card_select` (Burning Pact, a
/// Skill Potion's offer). The fight mirror re-parks a fork of the carried
/// fight on the world's choice (the replay and injection the search uses,
/// [`AdvisorMirror::answer_card_select`]) and Instinct picks on it
/// ([`pick_choice`]), its first pick's class probabilities on the option
/// rows. `log` collects the info lines on the way.
///
/// # Errors
///
/// The player-facing reason Instinct cannot answer: why Instinct itself
/// could not pick, else that the mirror cannot stand on this choice.
#[allow(clippy::too_many_arguments)]
pub fn advise_choice(
    net: Option<&E2eNet>,
    driver_for: &mut dyn FnMut(i64) -> Result<(Driver, TokCache), PickerError>,
    mirror: &mut AdvisorMirror,
    state: &Value,
    cfg: &RecommendConfig,
    sims: Option<u32>,
    version: &str,
    wire_for: &mut crate::advisor_mirror::WireFor<'_>,
    log: &mut Vec<String>,
) -> Result<Value, String> {
    let room = room_kind(get(state, "combat"));
    let mut picks = ChoicePicks::default();
    let mut picker = |fork: &CombatState, a: crate::card_select::Arity| {
        pick_choice(
            net,
            driver_for,
            state,
            version,
            room,
            fork,
            a.max_select,
            &mut picks,
        )
    };
    let advice = mirror.answer_card_select(
        state,
        Some(cfg),
        sims,
        Some(&mut picker),
        Some(tr(CHOICE_NOTE)),
        wire_for,
    );
    log.append(&mut picks.log);
    finish_choice(advice, picks, crate::advisor_mirror::takes_several(state))
}

/// The answer to an in-fight choice from what the mirror answered and what
/// Instinct picked with. The mirror swallows a picker's failure into a bare
/// decline, so the reason is Instinct's own when it has one. A selection
/// that takes `several` cards shows each pick's own probability
/// ([`annotate_set_probs`]); a single pick, the first step's distribution.
///
/// # Errors
///
/// The mirror did not answer: why Instinct could not pick, else that the
/// mirror cannot stand on this choice.
pub fn finish_choice(
    advice: Option<Value>,
    picks: ChoicePicks,
    several: bool,
) -> Result<Value, String> {
    let Some(mut advice) = advice else {
        return Err(picks
            .failed
            .into_iter()
            .next()
            .unwrap_or_else(|| tr("the fight mirror cannot stand on this choice").to_string()));
    };
    if let Some((decision, probs)) = &picks.first {
        if several {
            annotate_set_probs(&mut advice, &picks.steps, picks.stop.as_ref());
        } else {
            annotate_choice_probs(&mut advice, decision, probs);
        }
    }
    Ok(advice)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GH #1688: the Dense Vegetation event's Wrigglers are their own
    /// species in the engine; other fights' Wrigglers stay Wrigglers.
    #[test]
    fn the_dense_vegetation_fight_reads_the_engines_wrigglers() {
        use sts2_core::MonsterKind as K;
        let species = |fight_id: &str| {
            let mut rm = crate::graft::tests::reference_run(1);
            let mut fight = CombatState::default();
            fight.creatures.push(sts2_core::Creature {
                id: std::num::NonZeroU32::new(2).expect("non-zero"),
                current_hp: 20,
                max_hp: 20,
                block: 0,
                powers: smallvec::SmallVec::new(),
                species: Some(K::Wriggler),
                monster_id: sts2_core::MonsterId("Wriggler"),
                intent: None,
                intent_history: sts2_core::IntentHistory::default(),
                ai: None,
            });
            let _ = enter_live_combat(
                &mut rm,
                fight,
                &json!({"encounter_id": fight_id}),
                RoomType::Monster,
            );
            rm.combat
                .as_ref()
                .and_then(|f| f.creatures.last())
                .and_then(|c| c.species)
        };
        assert_eq!(
            species("DENSE_VEGETATION_EVENT_ENCOUNTER"),
            Some(K::DenseVegetationWriggler)
        );
        assert_eq!(species(""), Some(K::Wriggler));
    }

    /// GH #1694: the game's deck already lost a stolen card; the engine
    /// counts it out itself, so the rebuild gives it back.
    #[test]
    fn a_card_the_thief_holds_goes_back_to_the_rebuilt_deck() {
        let deck_after = |reclaimed: bool| {
            let mut rm = crate::graft::tests::reference_run(1);
            let before = rm.run_state.master_deck.len();
            let mut fight = CombatState::default();
            let thief = sts2_core::CreatureUid(std::num::NonZeroU32::new(2).expect("non-zero"));
            let card = sts2_wire::state_dict::card_id_from_python_name("BASH").expect("card");
            fight.record_swipe_steal(thief, card, false);
            if reclaimed {
                fight.swipe_stolen[0].reclaimed = true;
            }
            let _ = enter_live_combat(&mut rm, fight, &json!({}), RoomType::Monster);
            rm.run_state.master_deck.len() - before
        };
        assert_eq!(deck_after(false), 1);
        assert_eq!(deck_after(true), 0);
    }

    #[test]
    fn the_room_is_the_boards_and_anything_else_is_a_monster_room() {
        assert_eq!(room_kind(&json!({"room_type": "Elite"})), RoomType::Elite);
        assert_eq!(room_kind(&json!({"room_type": "boss"})), RoomType::Boss);
        assert_eq!(room_kind(&json!({"room_type": "Event"})), RoomType::Monster);
        assert_eq!(room_kind(&json!({})), RoomType::Monster);
    }

    #[test]
    fn a_pick_instinct_cannot_make_says_why() {
        let mut picks = ChoicePicks::default();
        let mut driver_for = |_seed: i64| -> Result<(Driver, TokCache), PickerError> {
            unreachable!("no run to drive")
        };
        let fork = CombatState::default();
        let err = pick_choice(
            None,
            &mut driver_for,
            &json!({}),
            "v0.111.0",
            RoomType::Monster,
            &fork,
            1,
            &mut picks,
        )
        .unwrap_err();
        assert_eq!(err.0, "InstinctUnavailable");
        assert_eq!(picks.failed, vec![err.1.clone()]);
        // The mirror's bare decline gives way to Instinct's own reason.
        assert_eq!(finish_choice(None, picks, false).unwrap_err(), err.1);
        assert_eq!(
            finish_choice(None, ChoicePicks::default(), false).unwrap_err(),
            "the fight mirror cannot stand on this choice"
        );
    }

    #[test]
    fn the_first_picks_class_probabilities_land_on_its_options() {
        let decision = json!({"classes": [
            {"class": 0, "actions": [{"selection": [0, 2]}]},
            {"class": 1, "actions": [{"selection": [1]}, {"selection": [2]}]},
        ]});
        let picks = ChoicePicks {
            first: Some((decision, vec![0.723_456, 0.276_544])),
            ..ChoicePicks::default()
        };
        let advice = json!({"options": [{}, {}, {}, {}]});
        let got = finish_choice(Some(advice), picks, false).unwrap();
        let probs: Vec<&Value> = got["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| &o["prob"])
            .collect();
        assert_eq!(
            probs,
            [&json!(0.7235), &json!(0.2765), &json!(0.7235), &Value::Null]
        );
        assert_eq!(got["options"][0]["visit_share"], json!(0.7235));
    }

    /// GH #1770, Gambler's Brew in `wb-eb8f0e387af30b15`: the net took both
    /// Defends (first step: one class, 99.94%), then Strike+, Poor Sleep and
    /// Strike, which the first step gave ~0%. Each picked row shows its own
    /// step's probability, the card left out the stop step's, and no picked
    /// row reads near zero.
    #[test]
    fn a_multi_card_selects_rows_carry_each_picks_own_step() {
        // Hand: Backstab, Strike+, Defend, Poor Sleep, Defend, Strike.
        let first = json!({"classes": [
            {"class": 0, "actions": [{"selection": [0]}]},
            {"class": 1, "actions": [{"selection": [1]}]},
            {"class": 2, "actions": [{"selection": [2]}, {"selection": [4]}]},
            {"class": 3, "actions": [{"selection": [3]}]},
            {"class": 4, "actions": [{"selection": [5]}]},
            {"class": 5, "actions": [{"selection": []}]},
        ]});
        let stop = json!({"classes": [
            {"class": 0, "actions": [{"selection": [0]}]},
            {"class": 1, "actions": [{"selection": []}]},
        ]});
        let picks = ChoicePicks {
            first: Some((first, vec![0.0, 0.0, 0.9994, 0.0, 0.0006, 0.0])),
            steps: vec![(2, 0.9994), (4, 0.97), (3, 0.91), (5, 0.8), (1, 0.66)],
            stop: Some((stop, vec![0.002, 0.998])),
            ..ChoicePicks::default()
        };
        let best = [false, true, true, true, true, true];
        let advice =
            json!({"options": best.iter().map(|b| json!({"is_best": b})).collect::<Vec<_>>()});
        let probs = |got: &Value| -> Vec<Value> {
            got["options"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o["prob"].clone())
                .collect()
        };
        let got = finish_choice(Some(advice), picks, true).unwrap();
        assert_eq!(
            probs(&got),
            [
                json!(0.002),
                json!(0.66),
                json!(0.9994),
                json!(0.91),
                json!(0.97),
                json!(0.8)
            ]
        );
    }
}
