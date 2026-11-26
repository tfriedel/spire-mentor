//! One combat decision's advice (GH #1426; was
//! `realgame_advisor.compute_combat_advice`).
//!
//! One full search for one `advice_request` wrapping a combat board: the
//! board as the engine's fight, the search (on the fight the advisor's
//! mirror carries when it carries this one, else on a fresh rebuild), the
//! advice the HUD shows, its regret rows and the turn plan.

use serde_json::Value;
use sts2_core::CombatState;

use crate::advisor_mirror::{AdvisorMirror, Level, On};
use crate::board::{self, BoardError};
use crate::board_targets::BoardTargets;
use crate::combat_advice::{self as ca, Unmapped};
use crate::combat_mirror::{build_state, MirrorError, Mirrored};
use crate::display_names::Names;
use crate::headless::{self, HeadlessError};
use crate::py::{self, get, truthy};
use crate::search::{self, RecommendConfig};
use crate::turn_plan;

/// The turn plan's re-searches run at an eighth of the decision's budget,
/// never below this: they only order the rest of the turn, and they share
/// the plan's time cap.
pub const PLAN_MIN_SIMS: u32 = 500;
pub const PLAN_SIMS_DIVISOR: u32 = 8;

/// What one decision's search produced.
#[derive(Debug, Clone)]
pub struct CombatAdvice {
    /// The wire advice object.
    pub advice: Value,
    /// The fight the search ran on, so a caller can run companion searches
    /// against the very same board.
    pub fight: Mirrored,
    /// The resolved config: the run's character and ascension, this
    /// decision's budget, `c_puct` left to the budget rule.
    pub cfg: RecommendConfig,
    /// The search's regret rows, crowned the way the advice is.
    pub rows: Vec<Value>,
}

/// Why a decision has no advice. Each names the Python exception it was.
#[derive(Debug)]
pub enum AdviseError {
    /// The board has no headless form (`ValueError`, or the version refusal).
    Headless(HeadlessError),
    /// The board has no engine fight.
    Board(BoardError),
    /// The rebuilt wire does not build a state (`state_from_dict`'s errors).
    State(MirrorError),
    /// The search refused its config (`ValueError`).
    Search(String),
    /// A search row of a kind outside the wire contract (`KeyError`).
    Unmapped(String),
}

impl std::fmt::Display for AdviseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Headless(e) => write!(f, "{e}"),
            Self::Board(e) => write!(f, "{e}"),
            Self::State(e) => write!(f, "{e}"),
            Self::Search(m) | Self::Unmapped(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for AdviseError {}

/// The decision's config: `cfg` with the run's character and ascension,
/// this decision's budget, and `c_puct` cleared.
///
/// `sims` varies per decision (boss vs regular) while a pinned `c_puct` is
/// chosen once at construction; clearing it hands it to
/// [`RecommendConfig::resolved_c_puct`], which reads it off the budget this
/// decision actually gets.
#[must_use]
pub fn decision_cfg(state: &Value, cfg: &RecommendConfig, sims: u32) -> RecommendConfig {
    let run_state = get(state, "run_state");
    let character = headless::character_display_name(get(run_state, "character"))
        .map(str::to_string)
        .or_else(|| cfg.character.clone());
    RecommendConfig {
        character,
        ascension: board::run_ascension(run_state, cfg.ascension),
        sims,
        c_puct: None,
        ..cfg.clone()
    }
}

/// `compute_combat_advice`. `log` collects what the advisor logs on the
/// way (the board import's warnings, a failed mirror search); the mirror
/// keeps its own log. `plan_budget_s` caps the turn plan's re-searches
/// ([`turn_plan::PLAN_BUDGET_S`] live; infinite in a replay).
///
/// # Errors
///
/// A board with no headless form or no engine fight, a rebuild that does
/// not build, a config the search refuses, a search row outside the wire
/// contract.
pub fn compute_combat_advice(
    state: &Value,
    cfg: &RecommendConfig,
    sims: u32,
    plan_budget_s: f64,
    mirror: Option<&mut AdvisorMirror>,
    names: &Names,
    log: &mut Vec<(Level, String)>,
) -> Result<CombatAdvice, AdviseError> {
    let call_cfg = decision_cfg(state, cfg, sims);
    let headless = headless::to_headless_combat_play(state).map_err(AdviseError::Headless)?;
    let (wire, warnings) = board::combat_wire(
        state,
        call_cfg.character.as_deref(),
        call_cfg.ascension,
        call_cfg.combat_id,
        call_cfg.seed,
    )
    .map_err(AdviseError::Board)?;
    log.extend(
        warnings
            .into_iter()
            .map(|w| (Level::Info, format!("adapter: {w}"))),
    );

    let mut live = None;
    if let Some(m) = mirror {
        if m.on_combat_board(&wire, state) {
            live = Some(m);
        }
    }
    if let Some(m) = live.as_deref_mut() {
        // The world is asking for a COMBAT action, so it is not showing a
        // selection, and a mirror parked on one answers a question nobody
        // asked: every action its search offers is a `choose`, which has no
        // executable form here. A card that opens a selection reports LATE,
        // so following it parks the mirror on a choice the world already
        // resolved; a reshuffle on the way to the screen hides the stale
        // hand from `reconcile`, and the park outlives every board after
        // (Silent+Prepared+ at Aeonglass, 2026-08-25). Dropping costs this
        // fight its carried state; keeping it costs the run.
        if m.live_state()
            .is_some_and(|s| sts2_wire::emit::pending_choice(s).is_some())
        {
            m.drop_fight("parked on a selection the world has finished");
            live = None;
        }
    }
    let mut searched: Option<(Mirrored, Value)> = None;
    if let Some(m) = live {
        // Carried state wins: the rebuild's board on every observable plus
        // the per-turn counters no wire carries (GH #370).
        match search_live(m, &call_cfg) {
            Ok(found) => searched = Some(found),
            Err(e) => {
                // Advice must always render: fall through to the rebuild.
                log.push((
                    Level::Warning,
                    format!("advisor mirror search failed, rebuilding: {e}"),
                ));
                m.drop_fight("search failed");
            }
        }
    }
    let (fight, result) = if let Some(found) = searched {
        found
    } else {
        let fight = build_state(&wire).map_err(AdviseError::State)?;
        let result = search::recommend(&fight.state, fight.intent_damage_known, &call_cfg)
            .map_err(AdviseError::Search)?;
        (fight, result)
    };

    let mut research = plan_research(&call_cfg, fight.intent_damage_known);
    let (advice, rows) = advice_from_result(
        state,
        &wire,
        &headless,
        &fight.state,
        &result,
        sims,
        names,
        Some(&mut research),
        plan_budget_s,
    )?;
    Ok(CombatAdvice {
        advice,
        fight,
        cfg: call_cfg,
        rows,
    })
}

/// The advice and regret rows for a search `result` over `fight` (the
/// engine fight of `wire`, the board `state` imports as): ranked, vetoed,
/// labelled, with the turn plan, which `research` continues past the PV
/// for at most `plan_budget_s` seconds.
///
/// # Errors
///
/// A search row outside the wire contract.
#[allow(clippy::too_many_arguments)]
pub fn advice_from_result(
    state: &Value,
    wire: &Value,
    headless: &Value,
    fight: &CombatState,
    result: &Value,
    sims: u32,
    names: &Names,
    research: Option<&mut dyn FnMut(&CombatState) -> Option<Vec<Value>>>,
    plan_budget_s: f64,
) -> Result<(Value, Vec<Value>), AdviseError> {
    let hand = py::list(get(wire, "deck"), "hand");
    // The search ranked the engine fight, whose enemy slots can still hold
    // a corpse the game removed (a mirror's carried fight): its rows name
    // the board's slots from here on (`board_targets`).
    let targets = BoardTargets::of(fight);
    let engine_result = result;
    let result = &targets.board_result(engine_result);
    let (tags, dropped) = ca::tags_from_result(result, hand).map_err(|e| {
        AdviseError::Unmapped(match e {
            Unmapped::Unmappable(m) | Unmapped::Unknown(m) => m,
        })
    })?;
    if let Some(first) = dropped.first() {
        eprintln!(
            "[hint] dropped {} unmappable action row(s) from the ranking, kept {}: {first}",
            dropped.len(),
            tags.len()
        );
    }
    // The PV stays in engine slots (the turn plan replays it on `fight`);
    // its labels name the board's.
    let pv: Vec<Value> = py::list(engine_result, "pv").to_vec();
    let pv_board: Vec<Value> = pv
        .iter()
        .take(4)
        .map_while(|a| targets.board_action(a))
        .collect();
    let labels = ca::pv_labels(&pv_board, state);
    let mut advice = ca::build_advice(
        tags,
        state,
        i64::from(sims),
        Some(&labels),
        get(result, "pick"),
        names,
    );
    let rows = ca::combat_rows(result, headless, &mut advice);
    let best = get(&advice, "best").clone();
    let plan = turn_plan::search_plan(fight, &pv, wire, state, &best, research, plan_budget_s);
    if !plan.is_empty() {
        advice["plan"] = Value::Array(plan);
    }
    Ok((advice, rows))
}

/// The live mirror's search, and the fight it searched.
fn search_live(m: &mut AdvisorMirror, cfg: &RecommendConfig) -> Result<(Mirrored, Value), String> {
    // A failure reads as the Python view's `MirrorDivergence` did.
    let viewed = |e: MirrorError| match e {
        MirrorError::Diverged(_) | MirrorError::NoState => format!("[mirror advisor] {e}"),
        other => other.to_string(),
    };
    let mirror = m
        .mirror_mut(On::Live)
        .ok_or_else(|| viewed(MirrorError::NoState))?;
    let fight = mirror
        .mirrored()
        .cloned()
        .ok_or_else(|| viewed(MirrorError::NoState))?;
    let result = mirror.search(cfg).map_err(viewed)?;
    Ok((fight, result))
}

/// The turn plan's re-search config: the decision's at an eighth of its
/// budget, never below [`PLAN_MIN_SIMS`].
#[must_use]
pub fn plan_cfg(call_cfg: &RecommendConfig) -> RecommendConfig {
    RecommendConfig {
        sims: PLAN_MIN_SIMS.max(call_cfg.sims / PLAN_SIMS_DIVISOR),
        c_puct: None,
        ..call_cfg.clone()
    }
}

/// The turn plan's next-move search: the same honest search at an eighth of
/// the budget, from the fight as the plan has left it; its line, pick first.
/// `None` when the search fails.
pub fn plan_research(
    call_cfg: &RecommendConfig,
    intent_damage_known: bool,
) -> impl FnMut(&CombatState) -> Option<Vec<Value>> {
    let cfg = plan_cfg(call_cfg);
    move |fight: &CombatState| {
        let result = search::recommend(fight, intent_damage_known, &cfg).ok()?;
        let best = py::list(&result, "actions")
            .iter()
            .find(|r| truthy(get(r, "is_best")));
        let pick = match best {
            // `r["action"]` on a row without one raised.
            Some(r) => r.get("action")?.clone(),
            None => Value::Null,
        };
        if !pick.is_object() {
            return Some(Vec::new());
        }
        let pv = py::list(&result, "pv");
        Some(match pv.first() {
            Some(first) if py::eq(first, &pick) => pv.to_vec(),
            _ => vec![pick],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_decision_config_takes_the_run_and_the_budget() {
        let cfg = RecommendConfig {
            c_puct: Some(3.0),
            character: Some("Silent".into()),
            ascension: 4,
            ..RecommendConfig::default()
        };
        let state = json!({"run_state": {"character": "IRONCLAD", "ascension": 10}});
        let c = decision_cfg(&state, &cfg, 4000);
        assert_eq!(
            (c.character.as_deref(), c.ascension, c.sims, c.c_puct),
            (Some("Ironclad"), 10, 4000, None)
        );
        let c = decision_cfg(&json!({"run_state": null}), &cfg, 8);
        assert_eq!((c.character.as_deref(), c.ascension), (Some("Silent"), 4));
    }

    #[test]
    fn the_plan_research_runs_at_an_eighth_of_the_budget_never_below_the_floor() {
        let at = |sims| {
            let cfg = RecommendConfig {
                sims,
                c_puct: Some(2.0),
                ..RecommendConfig::default()
            };
            let p = plan_cfg(&cfg);
            (p.sims, p.c_puct)
        };
        assert_eq!(at(4000), (PLAN_MIN_SIMS, None));
        assert_eq!(at(8000), (1000, None));
    }
}
