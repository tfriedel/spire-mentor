//! The turn plan: this turn's moves, in order, in the CURRENT hand's
//! indices (GH #1426 step 3; was `realgame_turn_plan.py`).
//!
//! `advice.plan` (PROTOCOL.md §5) lets the HUD number the cards to play this
//! turn 1, 2, 3. Both brains produce a line of engine actions -- the search
//! its principal variation, Instinct a greedy rollout of itself -- but every
//! action after the first indexes a hand that exists only after the earlier
//! ones were played. This module walks such a line on a copy of the fight
//! and maps each action back to the hand the player is looking at now,
//! keeping only what it can map with certainty:
//!
//! * a play's `card_index` is translated through the engine->mod hand map of
//!   the request (`wire.deck.hand[i].stable_idx`), which is shifted by every
//!   card the line has already played;
//! * after every step the engine's hand must be exactly the previous hand
//!   minus the played card (same cards, same order). A card that draws,
//!   generates, discards, upgrades or reorders ends the plan AFTER itself:
//!   it is still certain, the next one is not;
//! * a step that opens a choice, wins or loses the fight ends the plan after
//!   itself; a move the walk cannot map (a choice, an orb verb, an index the
//!   map does not cover, an engine refusal) ends it before itself;
//! * `end_turn` is the last step when the line reaches it;
//! * the enemy roster must keep its length (`target_index` addresses it).
//!
//! Potions use the remap `best` uses: the engine's potion slot is dense over
//! the held potions, the mod's is the belt slot.

use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use sts2_core::parity::combat_state_to_bridge_state;
use sts2_core::CombatState;
use sts2_workbench::enemy_labels::{enemy_labels, labels_from_names};

use crate::board_targets::BoardTargets;
use crate::i18n::{tr, trf};
use crate::mirror_state::{parse_bridge_action, step_result_name};
use crate::py;

/// A defensive bound on one turn's plan (a real turn ends long before).
pub const MAX_PLAN_STEPS: usize = 12;

/// Wall-clock cap on the re-searches that extend a plan past its PV. A plan
/// is a drawing: once spent, the plan simply ends where it got to. A search
/// starts only when the last one's duration still fits, so the cap holds to
/// within the variance of one search. A non-finite budget has no cap:
/// replays use it, because a plan cut by the clock depends on the machine's
/// load (GH #1784: a golden replay lost a plan step under gate load).
pub const PLAN_BUDGET_S: f64 = 0.5;

/// The keys a plan step and `advice.best` share.
const MOVE_KEYS: [&str; 5] = ["action", "card_index", "target_index", "ally_index", "slot"];

/// Python's `isinstance(i, int) and 0 <= i < n`, as an index.
fn index_in(v: &Value, n: usize) -> Option<usize> {
    let i = v.as_i64()?;
    usize::try_from(i).ok().filter(|&i| i < n)
}

/// The engine hand as comparable identities (`CombatState.get_state`).
fn hand_key(state: &Value) -> Vec<[Value; 4]> {
    py::list(state, "hand")
        .iter()
        .filter(|c| c.is_object())
        .map(|c| ["id", "base_damage", "base_block", "type"].map(|k| py::get(c, k).clone()))
        .collect()
}

fn same_hand(a: &[[Value; 4]], b: &[[Value; 4]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.iter().zip(y).all(|(l, r)| py::eq(l, r)))
}

fn roster_len(state: &Value) -> usize {
    py::list(state, "enemies").len()
}

/// `str(d.get(a) or d.get(b) or fallback)`.
fn name_or(d: &Value, keys: &[&str], fallback: impl FnOnce() -> String) -> String {
    keys.iter()
        .map(|k| py::get(d, k))
        .find(|v| py::truthy(v))
        .map_or_else(fallback, py::str_of)
}

/// The label of teammate `allies[index]` (`enemy_labels.ally_label`): their
/// character, numbered when two share one; every teammate counted.
pub(crate) fn ally_label(allies: &[Value], index: &Value) -> Option<String> {
    let i = index_in(index, allies.len())?;
    let names: Vec<String> = allies
        .iter()
        .enumerate()
        .map(|(n, a)| name_or(a, &["character"], || trf("teammate {n}", &[("n", &n)])))
        .collect();
    labels_from_names(&names).into_iter().nth(i)
}

/// The label of `enemies[index]` (`enemy_labels.enemy_label`), positional.
pub(crate) fn enemy_label(enemies: &[Value], index: &Value) -> Option<String> {
    let i = index_in(index, enemies.len())?;
    enemy_labels(enemies).into_iter().nth(i)
}

/// A plan step named against the request's board (`state`: the mod's).
#[must_use]
pub fn step_label(state: &Value, step: &Value) -> String {
    let action = py::get(step, "action");
    let mut label = match action.as_str() {
        Some("end_turn") => return tr("End turn").into(),
        Some("play") => {
            let hand = py::list(state, "hand");
            let card = index_in(py::get(step, "card_index"), hand.len())
                .map_or(&Value::Null, |i| &hand[i]);
            name_or(card, &["name", "id"], || "?".into())
        }
        Some("potion") => {
            let slot = py::get(step, "slot");
            let potion = py::list(state, "potions")
                .iter()
                .find(|p| p.is_object() && py::eq(py::get(p, "slot"), slot))
                .unwrap_or(&Value::Null);
            name_or(potion, &["name", "id"], || {
                trf("potion {n}", &[("n", &py::str_of(slot))])
            })
        }
        _ => {
            let name = if py::truthy(action) {
                py::str_of(action)
            } else {
                "?".into()
            };
            return name.replace('_', " ");
        }
    };
    let target = if step.get("ally_index").is_some_and(|v| !v.is_null()) {
        ally_label(py::list(state, "allies"), py::get(step, "ally_index"))
    } else {
        enemy_label(py::list(state, "enemies"), py::get(step, "target_index"))
    };
    if let Some(target) = target {
        label.push_str(" → ");
        label.push_str(&target);
    }
    label
}

/// Whether a plan step is the move `advice.best` names.
#[must_use]
pub fn same_move(step: &Value, best: &Value) -> bool {
    MOVE_KEYS
        .iter()
        .all(|k| py::eq(py::get(step, k), py::get(best, k)))
}

/// `best` as a plan step (when the line cannot follow it further).
#[must_use]
pub fn first_step(best: &Value, state: &Value) -> Option<Value> {
    let Value::Object(b) = best else {
        return None;
    };
    if !matches!(
        b.get("action").and_then(Value::as_str),
        Some("play" | "potion" | "end_turn")
    ) {
        return None;
    }
    let mut step: Map<String, Value> = MOVE_KEYS
        .iter()
        .filter_map(|k| {
            b.get(*k)
                .filter(|v| !v.is_null())
                .map(|v| ((*k).into(), v.clone()))
        })
        .collect();
    let label = step_label(state, &Value::Object(step.clone()));
    step.insert("label".into(), label.into());
    Some(Value::Object(step))
}

/// Maps one line of engine actions onto the request's hand, step by step.
#[derive(Debug, Clone)]
pub struct PlanWalk {
    /// The mod's `combat_action` payload (labels, potion slots).
    state: Value,
    /// Engine hand index -> mod hand index (`wire.deck.hand[].stable_idx`).
    hand_map: Vec<Option<i64>>,
    /// Engine (dense) potion slot -> the mod's belt slot.
    potion_map: Vec<i64>,
    /// Engine enemy slot -> the mod's `enemies[]` index.
    targets: BoardTargets,
    pub steps: Vec<Value>,
    pub done: bool,
}

impl PlanWalk {
    /// `wire`: the engine wire dict the fight was built from; `state`: the
    /// mod's `combat_action` payload.
    #[must_use]
    pub fn new(wire: &Value, state: &Value) -> Self {
        let hand_map = py::list(py::get(wire, "deck"), "hand")
            .iter()
            .map(|c| py::get(c, "stable_idx").as_i64())
            .collect();
        let potion_map = py::list(state, "potions")
            .iter()
            .filter(|p| p.is_object() && !py::truthy(py::get(p, "empty")))
            .filter_map(|p| py::get(p, "slot").as_i64())
            .collect();
        Self {
            state: state.clone(),
            hand_map,
            potion_map,
            targets: BoardTargets::default(),
            steps: Vec::new(),
            done: false,
        }
    }

    /// The walk over engine fight `fight`, whose enemy slots can differ
    /// from the board's (`board_targets`).
    #[must_use]
    pub fn on_fight(self, fight: &CombatState) -> Self {
        self.with_targets(BoardTargets::of(fight))
    }

    /// The walk with the engine -> board enemy slots `targets`.
    #[must_use]
    pub fn with_targets(mut self, targets: BoardTargets) -> Self {
        self.targets = targets;
        self
    }

    /// `action` (engine combat form) as a plan step, or `None`.
    #[must_use]
    pub fn to_step(&self, action: &Value) -> Option<Value> {
        let mut step = Map::new();
        match py::get(action, "action").as_str() {
            Some("end_turn") => {
                step.insert("action".into(), "end_turn".into());
                return Some(Value::Object(step));
            }
            Some("play") => {
                let ei = index_in(py::get(action, "card_index"), self.hand_map.len())?;
                let card = self.hand_map[ei]?;
                step.insert("action".into(), "play".into());
                step.insert("card_index".into(), card.into());
            }
            Some("potion" | "drink_potion") => {
                let d = index_in(py::get(action, "slot"), self.potion_map.len())?;
                step.insert("action".into(), "potion".into());
                step.insert("slot".into(), self.potion_map[d].into());
            }
            _ => return None,
        }
        let target = py::get(action, "target_index");
        if !target.is_null() {
            let board = self.targets.to_board(py::int_of(target))?;
            step.insert("target_index".into(), board.into());
        }
        let ally = py::get(action, "ally_index");
        if !ally.is_null() {
            step.insert("ally_index".into(), py::int_of(ally).into());
        }
        Some(Value::Object(step))
    }

    /// Label `step` and append it; ends the walk on `end_turn` or the cap.
    pub fn push(&mut self, mut step: Value) {
        let label = step_label(&self.state, &step);
        if let Value::Object(m) = &mut step {
            m.insert("label".into(), label.into());
        }
        let end = py::get(&step, "action").as_str() == Some("end_turn");
        self.steps.push(step);
        if end || self.steps.len() >= MAX_PLAN_STEPS {
            self.done = true;
        }
    }

    /// Account for `action` having moved the fight `before` -> `after`
    /// (engine `get_state` dicts); ends the walk unless the next step is
    /// still certain.
    pub fn advance(&mut self, action: &Value, before: &Value, after: &Value, result: Option<&str>) {
        if matches!(
            result,
            Some("pending_choice" | "combat_won" | "combat_lost")
        ) || roster_len(after) != roster_len(before)
        {
            self.done = true;
            return;
        }
        let mut expected = hand_key(before);
        match py::get(action, "action").as_str() {
            Some("play") => {
                let ei = usize::try_from(py::int_of(py::get(action, "card_index"))).ok();
                match ei {
                    Some(ei) if ei < expected.len() && ei < self.hand_map.len() => {
                        expected.remove(ei);
                        if !same_hand(&hand_key(after), &expected) {
                            self.done = true;
                            return;
                        }
                        self.hand_map.remove(ei);
                    }
                    _ => self.done = true,
                }
            }
            Some("potion" | "drink_potion") => {
                if !same_hand(&hand_key(after), &expected) {
                    self.done = true;
                    return;
                }
                let slot = usize::try_from(py::int_of(py::get(action, "slot"))).ok();
                match slot {
                    Some(s) if s < self.potion_map.len() => {
                        self.potion_map.remove(s);
                    }
                    _ => self.done = true,
                }
            }
            _ => self.done = true,
        }
    }
}

/// The search's plan for this turn.
///
/// Follows the principal variation, then -- the honest search's PV is one
/// ply deep, because its deeper nodes keep no realized state to name an
/// action against -- `research(fight)`: a search from the fight as the plan
/// has left it, returning its line (its pick first), or `None` when it
/// failed. Each step is applied to a clone of `cs` (never `cs` itself) and
/// mapped/cut by [`PlanWalk`]. When the PV does not open with `best` (the PV
/// follows the highest-Q child; the engine's pick rule or the mod's veto
/// can crown another), the plan is `best` alone.
pub fn search_plan(
    cs: &CombatState,
    pv: &[Value],
    wire: &Value,
    state: &Value,
    best: &Value,
    mut research: Option<&mut dyn FnMut(&CombatState) -> Option<Vec<Value>>>,
    budget_s: f64,
) -> Vec<Value> {
    if !best.is_object() {
        return Vec::new();
    }
    let lone = first_step(best, state);
    let mut sim = cs.clone();
    let mut walk = PlanWalk::new(wire, state).on_fight(cs);
    let mut line: std::collections::VecDeque<Value> =
        pv.iter().filter(|a| a.is_object()).cloned().collect();
    let budget = budget_s
        .is_finite()
        .then(|| Duration::from_secs_f64(budget_s.max(0.0)));
    let t0 = Instant::now();
    let mut last_search = Duration::ZERO;
    let mut ply = 0usize;
    while !walk.done {
        if line.is_empty() {
            let Some(research) = research.as_deref_mut().filter(|_| ply > 0) else {
                break;
            };
            if budget.is_some_and(|b| t0.elapsed() + last_search > b) {
                break;
            }
            let t_search = Instant::now();
            let Some(found) = research(&sim) else {
                break;
            };
            line = found.into_iter().filter(Value::is_object).collect();
            last_search = t_search.elapsed();
            if line.is_empty() {
                break;
            }
        }
        let Some(action) = line.pop_front() else {
            break;
        };
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
        let before = combat_state_to_bridge_state(&sim);
        let Ok(parsed) = parse_bridge_action(&action) else {
            walk.steps.pop();
            break;
        };
        let result = step_result_name(sts2_core::apply(&mut sim, parsed));
        if matches!(result, "invalid_target" | "illegal_play") {
            walk.steps.pop();
            break;
        }
        let after = combat_state_to_bridge_state(&sim);
        walk.advance(&action, &before, &after, Some(result));
        ply += 1;
    }
    if walk.steps.is_empty() {
        return lone.into_iter().collect();
    }
    walk.steps
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn board() -> Value {
        json!({
            "hand": [{"name": "Strike"}, {"id": "DEFEND"}, {}],
            "enemies": [{"name": "Rat"}, {"name": "Rat"}, {"name": "Louse", "is_dead": true}],
            "allies": [{"character": "Silent"}, {"character": "Silent"}, {}],
            "potions": [{"slot": 0, "empty": true}, {"slot": 2, "name": "Fire Potion"}],
        })
    }

    #[test]
    fn labels_name_the_card_and_its_target() {
        let s = board();
        let label = |step| step_label(&s, &step);
        assert_eq!(label(json!({"action": "end_turn"})), "End turn");
        assert_eq!(
            label(json!({"action": "play", "card_index": 0, "target_index": 1})),
            "Strike → Rat #2"
        );
        assert_eq!(label(json!({"action": "play", "card_index": 1})), "DEFEND");
        assert_eq!(label(json!({"action": "play", "card_index": 2})), "?");
        assert_eq!(
            label(json!({"action": "play", "card_index": 9, "target_index": 2})),
            "? → Louse"
        );
        assert_eq!(
            label(json!({"action": "potion", "slot": 2, "ally_index": 1})),
            "Fire Potion → Silent #2"
        );
        assert_eq!(
            label(json!({"action": "potion", "slot": 5, "ally_index": 2})),
            "potion 5 → teammate 2"
        );
        assert_eq!(label(json!({"action": "choose_option"})), "choose option");
        assert_eq!(label(json!({})), "?");
    }

    #[test]
    fn steps_map_engine_indices_to_the_mods() {
        let wire = json!({"deck": {"hand": [{"stable_idx": 4}, {"stable_idx": 2}, {}]}});
        let walk = PlanWalk::new(&wire, &board());
        let step = |a| walk.to_step(&a);
        assert_eq!(
            step(json!({"action": "play", "card_index": 0, "target_index": 1})),
            Some(json!({"action": "play", "card_index": 4, "target_index": 1}))
        );
        assert_eq!(
            step(json!({"action": "play", "card_index": 2})),
            None,
            "no stable_idx"
        );
        assert_eq!(step(json!({"action": "play", "card_index": 3})), None);
        // Dense engine slot 0 is the first held potion: belt slot 2.
        assert_eq!(
            step(json!({"action": "drink_potion", "slot": 0})),
            Some(json!({"action": "potion", "slot": 2}))
        );
        assert_eq!(step(json!({"action": "choose", "index": 0})), None);
    }

    #[test]
    fn a_hand_that_changes_beyond_the_played_card_ends_the_walk() {
        let wire = json!({"deck": {"hand": [{"stable_idx": 0}, {"stable_idx": 1}]}});
        let before = json!({"hand": [{"id": "A"}, {"id": "B"}], "enemies": [{}]});
        let play0 = json!({"action": "play", "card_index": 0});

        let mut walk = PlanWalk::new(&wire, &board());
        walk.advance(
            &play0,
            &before,
            &json!({"hand": [{"id": "B"}], "enemies": [{}]}),
            Some("advanced"),
        );
        assert!(!walk.done);
        assert_eq!(
            walk.to_step(&play0),
            Some(json!({"action": "play", "card_index": 1}))
        );

        let mut drew = PlanWalk::new(&wire, &board());
        drew.advance(
            &play0,
            &before,
            &json!({"hand": [{"id": "B"}, {"id": "C"}], "enemies": [{}]}),
            None,
        );
        assert!(drew.done);

        let mut spawned = PlanWalk::new(&wire, &board());
        spawned.advance(
            &play0,
            &before,
            &json!({"hand": [{"id": "B"}], "enemies": [{}, {}]}),
            None,
        );
        assert!(spawned.done);
    }

    #[test]
    fn first_step_keeps_the_move_keys_and_labels_it() {
        let best =
            json!({"action": "play", "card_index": 0, "target_index": null, "label": "x", "q": 1});
        assert_eq!(
            first_step(&best, &board()),
            Some(json!({"action": "play", "card_index": 0, "label": "Strike"}))
        );
        assert_eq!(first_step(&json!({"action": "choose"}), &board()), None);
        assert!(same_move(
            &json!({"action": "play", "card_index": 0, "label": "a"}),
            &best
        ));
    }
}
