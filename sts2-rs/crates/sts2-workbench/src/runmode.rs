//! Run mode (M3): the native run agent behind "play to end" / autobattle,
//! the RunAction → workbench-verb translation, and the pure cores of
//! `workbench/autobattle.py` and `workbench/wpa_review.py`.
//!
//! * [`RunAgent`] is `sts2_rs.HeuristicRunAgent` — the deployed
//!   `MctsCombatPolicy<HeuristicPolicy>` stepped one decision at a time on
//!   the live run, emitting the same per-step record. Combat is the frozen
//!   MCTS; the heuristic meta covers every other surface. Single-threaded
//!   (`threads` is accepted and ignored on both hosts).
//! * [`to_workbench_verb`] / [`resolve_workbench_verb`] port
//!   `run_policy.py`'s translation of a raw `RunAction` dict into the C#
//!   verb vocabulary the timeline labels and the action endpoint speak.
//! * [`hash_seed`] is `run_stepper.hash_seed`: `run_N` → `N`, anything else
//!   folds to a 31-bit int.
//! * The autobattle helpers decide which states stop for the human and diff
//!   two run-level states ("since your last stop"); the WPA helpers turn the
//!   recorded node values into a curve and ranked turning points.
//!
//! The orchestration (the step loop, the branch nodes, the verdict) lives on
//! [`crate::session::Session`].

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_runai::policy::{CombatBudget, HeuristicPolicy, MctsCombatPolicy};
use sts2_wire::emit;

use crate::game::Game;
use crate::rmx;

// ── seeds ─────────────────────────────────────────────────────────────────

/// `run_stepper.hash_seed`: `run_1`/`run_2` → `1`/`2`; any other string
/// folds to a 31-bit int (`h = (h * 131 + ord(ch)) & 0x7FFF_FFFF`).
#[must_use]
pub fn hash_seed(seed: &str) -> i32 {
    if let Some(tail) = seed.strip_prefix("run_") {
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            // Python's int() is unbounded; the agent takes an i32. A tail
            // beyond i32 is a caller error in Python too (OverflowError).
            if let Ok(n) = tail.parse::<i32>() {
                return n;
            }
        }
    }
    let mut h: i64 = 0;
    for ch in seed.chars() {
        h = (h * 131 + i64::from(u32::from(ch))) & 0x7FFF_FFFF;
    }
    i32::try_from(h).unwrap_or(0)
}

// ── RunAction → workbench verb ────────────────────────────────────────────

/// `_DEDICATED_VERB_KINDS`: action kinds with a workbench verb of their own.
const DEDICATED_VERB_KINDS: &[&str] = &[
    "move",
    "play_card",
    "end_turn",
    "drink_potion",
    "combat_resolve_choice",
    "pick_card_reward",
    "shop_buy_card",
    "shop_buy_relic",
    "shop_buy_potion",
    "smith_card",
    "remove_deck_card",
    "shop_remove_card",
    "rest_option",
    "event_choice",
    "pick_boss_relic",
    "leave_shop",
    "upgrade_deck_card",
    "transform_deck_card",
    "confirm_deck_selection",
    "enchant_deck_card",
    "grant_deck_card",
];

/// `_PENDING_SELECT_KINDS`.
const PENDING_SELECT_KINDS: &[&str] = &[
    "upgrade_deck_card",
    "transform_deck_card",
    "confirm_deck_selection",
    "enchant_deck_card",
    "grant_deck_card",
];

/// `map_move_options`: `col → child node` for the current map decision,
/// from graph connectivity (the offered moves are the current node's
/// children; the act-start decision has no current coord and offers the
/// start node's children).
#[must_use]
pub fn map_move_options(rm: &RunManager) -> BTreeMap<i64, Value> {
    let full = emit::run_manager::full_map(rm);
    let rows = full.get("rows").and_then(Value::as_array);
    let mut node_at: BTreeMap<(i64, i64), &Value> = BTreeMap::new();
    if let Some(rows) = rows {
        for row in rows.iter().filter_map(Value::as_array) {
            for n in row {
                if let (Some(c), Some(r)) = (n["col"].as_i64(), n["row"].as_i64()) {
                    node_at.insert((c, r), n);
                }
            }
        }
    }
    let coord = emit::run_manager::map_coord(rm);
    let mut current = coord.and_then(|(c, r)| node_at.get(&(i64::from(c), i64::from(r))).copied());
    if current.is_none() {
        current = rows
            .and_then(|r| r.first())
            .and_then(Value::as_array)
            .and_then(|r| r.first());
    }
    let mut out = BTreeMap::new();
    let children = current
        .and_then(|c| c.get("children"))
        .and_then(Value::as_array);
    for ch in children.into_iter().flatten() {
        let (Some(c), Some(r)) = (ch["col"].as_i64(), ch["row"].as_i64()) else {
            continue;
        };
        let node = node_at
            .get(&(c, r))
            .map_or_else(|| ch.clone(), |n| (*n).clone());
        out.insert(c, node);
    }
    out
}

/// `to_workbench_verb`: a raw `RunAction` dict → the workbench's C#-shaped
/// `(verb, args)`. Must run BEFORE `take_action` (the map row / bundle
/// sub-state are pre-move). A kind with no dedicated verb round-trips as
/// `(kind, payload)`; [`resolve_workbench_verb`] remaps those to a
/// `choose_option` position.
#[must_use]
pub fn to_workbench_verb(rm: &RunManager, action: &Value) -> (String, Value) {
    let kind = action.get("action").and_then(Value::as_str).unwrap_or("");
    let field = |k: &str| action.get(k).cloned().unwrap_or(Value::Null);
    let indices = |k: &str| match action.get(k) {
        Some(Value::Number(n)) => json!(n.to_string()),
        Some(Value::Null) | None => json!("None"),
        Some(other) => json!(other.to_string()),
    };
    match kind {
        "move" => {
            let col = field("col");
            let node = col
                .as_i64()
                .and_then(|c| map_move_options(rm).remove(&c))
                .unwrap_or_else(|| json!({}));
            let row = node.get("row").cloned().unwrap_or(Value::Null);
            ("select_map_node".into(), json!({"col": col, "row": row}))
        }
        "play_card" => (
            "play_card".into(),
            json!({"card_index": field("hand_idx"), "target_index": field("target_index")}),
        ),
        "end_turn" => ("end_turn".into(), json!({})),
        "drink_potion" => (
            "use_potion".into(),
            json!({"potion_index": field("potion_slot"), "target_index": field("target_index")}),
        ),
        // The e2e brain's multi-pick in-combat selection (no `RunAction`).
        "combat_resolve_choice_selection" => {
            let picks: Vec<String> = action
                .get("selection")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(ToString::to_string)
                .collect();
            (
                "select_cards".into(),
                json!({"indices": format!("[{}]", picks.join(", "))}),
            )
        }
        "combat_resolve_choice" => match action.get("option") {
            None | Some(Value::Null) => ("skip_select".into(), json!({})),
            Some(_) => ("select_cards".into(), json!({"indices": indices("option")})),
        },
        "pick_card_reward" => match action.get("idx") {
            None | Some(Value::Null) => ("skip_card_reward".into(), json!({})),
            Some(idx) => {
                if emit::run_manager::card_reward_bundles(rm).is_some() {
                    ("select_bundle".into(), json!({"bundle_index": idx}))
                } else {
                    ("select_card_reward".into(), json!({"card_index": idx}))
                }
            }
        },
        "shop_buy_card" => ("buy_card".into(), json!({"card_index": field("idx")})),
        "shop_buy_relic" => ("buy_relic".into(), json!({"relic_index": field("idx")})),
        "shop_buy_potion" => ("buy_potion".into(), json!({"potion_index": field("idx")})),
        "confirm_deck_selection" => ("skip_select".into(), json!({})),
        "smith_card" | "remove_deck_card" => {
            ("select_cards".into(), json!({"indices": indices("idx")}))
        }
        k if PENDING_SELECT_KINDS.contains(&k) => {
            ("select_cards".into(), json!({"indices": indices("idx")}))
        }
        "shop_remove_card" => (
            "select_cards".into(),
            json!({"indices": indices("deck_idx")}),
        ),
        "rest_option" | "event_choice" | "pick_boss_relic" => (
            "choose_option".into(),
            json!({"option_index": field("idx")}),
        ),
        "leave_shop" => ("leave_room".into(), json!({})),
        other => {
            let mut payload = Map::new();
            if let Some(m) = action.as_object() {
                for (k, v) in m {
                    if k != "action" {
                        payload.insert(k.clone(), v.clone());
                    }
                }
            }
            (other.to_string(), Value::Object(payload))
        }
    }
}

/// `resolve_workbench_verb`: [`to_workbench_verb`] with the option-list
/// fallback — a kind without a dedicated verb becomes the `choose_option`
/// whose index is the action's position in `acts` (the order the backend
/// renders the options in); `None` when it is not in the list.
#[must_use]
pub fn resolve_workbench_verb(
    rm: &RunManager,
    action: &RunAction,
    acts: &[RunAction],
) -> Option<(String, Value)> {
    let wire = rmx::action_value(action);
    let kind = wire.get("action").and_then(Value::as_str).unwrap_or("");
    if DEDICATED_VERB_KINDS.contains(&kind) {
        return Some(to_workbench_verb(rm, &wire));
    }
    let idx = acts.iter().position(|a| a == action)?;
    Some(("choose_option".into(), json!({"option_index": idx})))
}

// ── the run agent ─────────────────────────────────────────────────────────

/// One `HeuristicRunAgent.step` record.
#[derive(Debug, Clone)]
pub struct StepRecord {
    pub phase: &'static str,
    pub is_combat: bool,
    pub over: bool,
    pub won: bool,
    pub action: RunAction,
    /// `{"actions": [score dicts], "recommended": action}` when MCTS ran.
    pub combat_search: Option<Value>,
    /// `format!("{result:?}")` of the engine's `take_action`.
    pub result: String,
}

/// `sts2_rs.HeuristicRunAgent`: the tier-1 heuristic meta over MCTS combat,
/// stepped one decision at a time. Carries its MCTS RNG and per-run
/// decision counter across calls, so pausing and resuming a run keeps
/// searching from where it stopped.
#[derive(Debug)]
pub struct RunAgent {
    policy: MctsCombatPolicy<HeuristicPolicy>,
}

impl RunAgent {
    /// `HeuristicRunAgent(seed, sims, boss_sims)`; `threads` is ignored by
    /// both hosts (the combat search is single-threaded).
    #[must_use]
    pub fn new(seed: i32, sims: u32, boss_sims: u32) -> Self {
        let mut budget = CombatBudget::eval_cheap();
        budget.sims = sims;
        budget.boss_sims = boss_sims;
        Self {
            policy: MctsCombatPolicy::new(HeuristicPolicy::new(), budget, seed),
        }
    }

    /// Retarget the combat budget mid-run without resetting the RNG or the
    /// decision counter (`set_budget`).
    pub fn set_budget(&mut self, sims: Option<u32>, boss_sims: Option<u32>) {
        let mut budget = self.policy.budget();
        if let Some(v) = sims.filter(|v| *v > 0) {
            budget.sims = v;
        }
        if let Some(v) = boss_sims.filter(|v| *v > 0) {
            budget.boss_sims = v;
        }
        self.policy.set_budget(budget);
    }

    /// `(sims, boss_sims)` in force.
    #[must_use]
    pub fn budget(&self) -> (u32, u32) {
        let b = self.policy.budget();
        (b.sims, b.boss_sims)
    }

    /// Apply one decision to the game's run and return the step record, or
    /// `None` when the policy has no legal action to take.
    pub fn step(&mut self, game: &mut Game) -> Option<StepRecord> {
        let rm = game.native_run()?;
        let actions = rm.get_available_actions();
        if actions.is_empty() {
            return None;
        }
        let phase = rm.phase();
        let is_combat = phase == Phase::Combat;
        let idx = self.policy.choose(rm, &actions)?;
        let action = actions[idx];
        // Capture the combat search AFTER `choose` (it ran MCTS) but BEFORE
        // `take_action` (which advances the state). Only when MCTS ran.
        let combat_search = if is_combat {
            self.policy.last_combat_search().map(|rec| {
                let best_idx = rec
                    .outcomes
                    .iter()
                    .position(|o| o.action == rec.chosen)
                    .unwrap_or(0);
                let list: Vec<Value> = rec
                    .outcomes
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        emit::mcts::action_score_value(
                            &o.action,
                            o.visits,
                            o.wins,
                            o.losses,
                            o.ties,
                            o.value_sum,
                            o.hp_sum,
                            o.hp_sq_sum,
                            o.hp_n,
                            false,
                            false,
                            i == best_idx,
                        )
                    })
                    .collect();
                json!({
                    "actions": list,
                    "recommended": emit::combat_action_to_value(&rec.chosen),
                })
            })
        } else {
            None
        };
        let result = game.take_native(action);
        let rm = game.native_run()?;
        Some(StepRecord {
            phase: rmx::phase_str_of(phase),
            is_combat,
            over: rm.is_over(),
            won: rm.run_state.player_won,
            action,
            combat_search,
            result: format!("{result:?}"),
        })
    }
}

// ── autobattle (pure core of workbench/autobattle.py) ─────────────────────

/// Decision kinds that hand control to the human in an autobattle.
pub const STOP_DECISIONS: &[&str] = &[
    "bundle_select",
    "card_reward",
    "card_select",
    "event_choice",
    "map_select",
    "rest_site",
    "shop",
    "treasure",
];

/// Decisions that never stop.
pub const NEVER_STOP: &[&str] = &["combat_play", "game_over"];

/// How close two options' policy probabilities must be for a disagreement
/// to be a near-tie rather than a miss.
pub const NEAR_TIE_PROB_MARGIN: f64 = 0.05;

/// `verdict_band`: `matched` / `near_tie` / `differed`.
#[must_use]
pub fn verdict_band(agreed: bool, margin: Option<f64>) -> &'static str {
    if agreed {
        return "matched";
    }
    match margin {
        Some(m) if m <= NEAR_TIE_PROB_MARGIN => "near_tie",
        _ => "differed",
    }
}

/// `is_stop_point`: a run-level decision the human should make. A state
/// without a recognisable `decision` is auto-played rather than stopping.
#[must_use]
pub fn is_stop_point(state: &Value) -> bool {
    match state.get("decision").and_then(Value::as_str) {
        Some(d) if !NEVER_STOP.contains(&d) => STOP_DECISIONS.contains(&d),
        _ => false,
    }
}

/// `_card_key`: display name plus upgrade state.
fn card_key(card: &Value) -> (String, bool) {
    let name = ["name", "id", "card_id"]
        .iter()
        .find_map(|k| card.get(*k).filter(|v| crate::render::py::truthy(v)))
        .map_or_else(|| "?".to_string(), crate::render::py::to_str);
    (
        name,
        card.get("upgraded").is_some_and(crate::render::py::truthy),
    )
}

type Counter<K> = BTreeMap<K, i64>;

fn count<K: Ord>(items: impl Iterator<Item = K>) -> Counter<K> {
    let mut c = BTreeMap::new();
    for k in items {
        *c.entry(k).or_insert(0) += 1;
    }
    c
}

/// `Counter` subtraction: positive counts only.
fn minus<K: Ord + Clone>(a: &Counter<K>, b: &Counter<K>) -> Counter<K> {
    let mut out = BTreeMap::new();
    for (k, &n) in a {
        let d = n - b.get(k).copied().unwrap_or(0);
        if d > 0 {
            out.insert(k.clone(), d);
        }
    }
    out
}

/// `_deck_delta`: multiset diff of two master decks, folding upgrades into
/// their own bucket. Returns `(added, removed, upgraded)`, each sorted.
fn deck_delta(prev: &[Value], cur: &[Value]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let before = count(prev.iter().map(card_key));
    let after = count(cur.iter().map(card_key));
    let mut gained = minus(&after, &before);
    let mut lost = minus(&before, &after);
    let mut upgraded: Vec<String> = Vec::new();
    let keys: Vec<(String, bool)> = gained.keys().cloned().collect();
    for key in keys {
        let (name, is_upgraded) = &key;
        if !is_upgraded {
            continue;
        }
        let n = gained[&key];
        let plain = (name.clone(), false);
        let paired = n.min(lost.get(&plain).copied().unwrap_or(0));
        if paired > 0 {
            for _ in 0..paired {
                upgraded.push(name.clone());
            }
            *gained.get_mut(&key).expect("key") -= paired;
            *lost.get_mut(&plain).expect("plain") -= paired;
        }
    }
    let names = |counter: &Counter<(String, bool)>| -> Vec<String> {
        let mut out = Vec::new();
        for ((name, is_upgraded), &n) in counter {
            if n <= 0 {
                continue;
            }
            let label = if *is_upgraded {
                format!("{name}+")
            } else {
                name.clone()
            };
            for _ in 0..n {
                out.push(label.clone());
            }
        }
        out.sort();
        out
    };
    upgraded.sort();
    (names(&gained), names(&lost), upgraded)
}

fn named_items(items: Option<&Value>) -> Counter<String> {
    count(
        items
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|i| {
                ["name", "id"]
                    .iter()
                    .find_map(|k| i.get(*k).filter(|v| crate::render::py::truthy(v)))
                    .map_or_else(|| "?".to_string(), crate::render::py::to_str)
            }),
    )
}

/// `_gained_lost`: `(gained, lost)` display names for a relic / potion tray.
fn gained_lost(prev: Option<&Value>, cur: Option<&Value>) -> (Vec<String>, Vec<String>) {
    let before = named_items(prev);
    let after = named_items(cur);
    let elements = |c: Counter<String>| -> Vec<String> {
        let mut out = Vec::new();
        for (k, n) in c {
            for _ in 0..n {
                out.push(k.clone());
            }
        }
        out.sort();
        out
    };
    (
        elements(minus(&after, &before)),
        elements(minus(&before, &after)),
    )
}

fn int_or_none(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_i64)
}

fn signed(delta: i64) -> String {
    if delta < 0 {
        format!("−{}", delta.unsigned_abs())
    } else {
        format!("+{delta}")
    }
}

/// `context_diff(...).to_dict()`: everything that changed between the
/// previous stop and this one, with the rendered `lines`. `prev` is `None`
/// at the first stop (absolutes only).
#[must_use]
pub fn context_diff(prev: Option<&Value>, cur: &Value, autoplayed: u64) -> Value {
    let null = Value::Null;
    let prev_player = prev.and_then(|p| p.get("player")).unwrap_or(&null);
    let cur_player = cur.get("player").unwrap_or(&null);
    let hp = int_or_none(cur_player.get("hp"));
    let max_hp = int_or_none(cur_player.get("max_hp"));
    let prev_hp = prev.and_then(|_| int_or_none(prev_player.get("hp")));
    let hp_delta = match (hp, prev_hp) {
        (Some(h), Some(p)) => Some(h - p),
        _ => None,
    };
    let gold = int_or_none(cur_player.get("gold"));
    let prev_gold = prev.and_then(|_| int_or_none(prev_player.get("gold")));
    let gold_delta = match (gold, prev_gold) {
        (Some(g), Some(p)) => Some(g - p),
        _ => None,
    };
    let empty: Vec<Value> = Vec::new();
    let deck_of = |p: &Value| -> Vec<Value> {
        p.get("deck")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let (added, removed, upgraded, relics_gained, relics_lost, potions_gained, potions_lost) =
        if prev.is_none() {
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![])
        } else {
            let (a, r, u) = deck_delta(&deck_of(prev_player), &deck_of(cur_player));
            let (rg, rl) = gained_lost(prev_player.get("relics"), cur_player.get("relics"));
            let (pg, pl) = gained_lost(prev_player.get("potions"), cur_player.get("potions"));
            (a, r, u, rg, rl, pg, pl)
        };
    let deck_size = cur_player
        .get("deck")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .len();

    // `render_lines`: most context-carrying first; unchanged quantities omitted.
    let mut lines: Vec<String> = Vec::new();
    if !upgraded.is_empty() {
        lines.push(format!("Upgraded {}", upgraded.join(", ")));
    }
    if !added.is_empty() {
        lines.push(format!("Added {}", added.join(", ")));
    }
    if !removed.is_empty() {
        lines.push(format!("Removed {}", removed.join(", ")));
    }
    if !relics_gained.is_empty() {
        lines.push(format!("Relic {}", relics_gained.join(", ")));
    }
    if !relics_lost.is_empty() {
        lines.push(format!("Lost relic {}", relics_lost.join(", ")));
    }
    if !potions_gained.is_empty() {
        lines.push(format!("Potion {}", potions_gained.join(", ")));
    }
    if !potions_lost.is_empty() {
        lines.push(format!("Used potion {}", potions_lost.join(", ")));
    }
    if let Some(h) = hp {
        let mut line = format!("HP {h}");
        if let Some(m) = max_hp {
            line.push_str(&format!("/{m}"));
        }
        if let Some(d) = hp_delta.filter(|d| *d != 0) {
            line.push_str(&format!(" ({})", signed(d)));
        }
        lines.push(line);
    }
    if let Some(g) = gold {
        let mut line = format!("Gold {g}");
        if let Some(d) = gold_delta.filter(|d| *d != 0) {
            line.push_str(&format!(" ({})", signed(d)));
        }
        lines.push(line);
    }
    if autoplayed > 0 {
        let s = if autoplayed == 1 { "" } else { "s" };
        lines.push(format!("{autoplayed} decision{s} auto-played"));
    }

    json!({
        "from_floor": prev.and_then(|p| int_or_none(p.get("floor"))),
        "to_floor": int_or_none(cur.get("floor")),
        "autoplayed": autoplayed,
        "hp": hp,
        "max_hp": max_hp,
        "hp_delta": hp_delta,
        "gold": gold,
        "gold_delta": gold_delta,
        "deck_size": deck_size,
        "deck": {"added": added, "removed": removed, "upgraded": upgraded},
        "relics_gained": relics_gained,
        "relics_lost": relics_lost,
        "potions_gained": potions_gained,
        "potions_lost": potions_lost,
        "lines": lines,
    })
}

// ── WPA review (pure core of workbench/wpa_review.py) ─────────────────────

/// How many turning points to surface by default.
pub const WPA_DEFAULT_TOP_K: usize = 3;
/// Ignore swings below this magnitude (Φ units) by default.
pub const WPA_DEFAULT_MIN_SWING: f64 = 0.02;

/// The caveats stamped into every payload.
pub const WPA_CAVEATS: [&str; 3] = [
    "The outlook score is the AI's shaped-return estimate of the run — a relative \
     signal, not a win percentage.",
    "A swing mixes your choice with the luck revealed after it (what the shop stocked, \
     what the reward offered, how the fight rolled) — use “verify” to check whether the \
     choice itself mattered.",
    "Judgments are relative to the AI's own play style; a line that is weak for the AI \
     may still be fine for a human plan.",
];

/// Python `round(x, 4)`: exact-value rounding, ties to even.
#[must_use]
pub fn round4(f: f64) -> f64 {
    format!("{f:.4}").parse().unwrap_or(f)
}

/// One scored decision on the run's value curve (`WpaPoint`).
#[derive(Debug, Clone)]
struct WpaPoint {
    node_id: Value,
    idx: i64,
    decision: String,
    floor: Value,
    act: Value,
    value: f64,
    delta_prev: f64,
    taken_label: String,
}

impl WpaPoint {
    fn to_value(&self) -> Value {
        json!({
            "node_id": self.node_id,
            "idx": self.idx,
            "decision": self.decision,
            "floor": self.floor,
            "act": self.act,
            "value": self.value,
            "delta_prev": self.delta_prev,
            "taken_label": self.taken_label,
        })
    }
}

/// `build_points`: rows without a usable `value` are skipped.
fn build_points(rows: &[Value]) -> Vec<WpaPoint> {
    let mut points: Vec<WpaPoint> = Vec::new();
    let mut prev: Option<f64> = None;
    for row in rows {
        let Some(value) = row.get("value").and_then(Value::as_f64) else {
            continue;
        };
        let taken = row
            .get("taken_label")
            .filter(|v| crate::render::py::truthy(v))
            .map_or_else(|| "?".to_string(), crate::render::py::to_str);
        points.push(WpaPoint {
            node_id: row.get("node_id").cloned().unwrap_or(Value::Null),
            idx: row
                .get("idx")
                .and_then(Value::as_i64)
                .unwrap_or(i64::try_from(points.len()).unwrap_or(i64::MAX)),
            decision: row
                .get("decision")
                .map_or_else(|| "unknown".to_string(), crate::render::py::to_str),
            floor: row.get("floor").cloned().unwrap_or(Value::Null),
            act: row.get("act").cloned().unwrap_or(Value::Null),
            value,
            delta_prev: prev.map_or(0.0, |p| value - p),
            taken_label: taken,
        });
        prev = Some(value);
    }
    points
}

/// `turning_points`: decisions ranked by the outlook swing they preceded.
fn turning_points(points: &[WpaPoint], top_k: usize, min_swing: f64) -> Vec<Value> {
    let mut tps: Vec<(f64, i64, Value)> = Vec::new();
    for i in 0..points.len().saturating_sub(1) {
        let p = &points[i];
        let next = &points[i + 1];
        let swing = next.value - p.value;
        if swing.abs() < min_swing {
            continue;
        }
        let direction = if swing < 0.0 { "dropped" } else { "rose" };
        let where_ = match p.floor.as_i64() {
            Some(f) => format!("floor {f}"),
            None if !p.floor.is_null() => format!("floor {}", crate::render::py::to_str(&p.floor)),
            None => format!("step {}", p.idx),
        };
        let gap = next.idx - p.idx;
        let span = if gap > 1 {
            " (spans the fight that followed)"
        } else {
            ""
        };
        let rounded = round4(swing);
        tps.push((
            rounded,
            p.idx,
            json!({
                "node_id": p.node_id,
                "idx": p.idx,
                "decision": p.decision,
                "floor": p.floor,
                "act": p.act,
                "taken_label": p.taken_label,
                "value_before": round4(p.value),
                "value_after": round4(next.value),
                "swing": rounded,
                "message": format!(
                    "{where_} ({}): after “{}” your outlook {direction} {swing:+.3} ({:.3} → {:.3}){span}",
                    p.decision, p.taken_label, p.value, next.value
                ),
            }),
        ));
    }
    tps.sort_by(|a, b| {
        (-a.0.abs())
            .partial_cmp(&(-b.0.abs()))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    tps.into_iter().take(top_k).map(|t| t.2).collect()
}

/// `wpa_review.review`: the full payload from scored decision rows —
/// `available: false` with `reason_if_empty` when fewer than two decisions
/// carry a value.
#[must_use]
pub fn wpa_review(
    rows: &[Value],
    top_k: usize,
    min_swing: f64,
    won: Option<bool>,
    reason_if_empty: &str,
) -> Value {
    let points = build_points(rows);
    if points.len() < 2 {
        return json!({
            "available": false,
            "reason": reason_if_empty,
            "points": points.iter().map(WpaPoint::to_value).collect::<Vec<_>>(),
            "turning_points": [],
            "summary": {},
            "caveats": WPA_CAVEATS,
        });
    }
    let tps = turning_points(&points, top_k, min_swing);
    let deltas: Vec<f64> = points[1..].iter().map(|p| p.delta_prev).collect();
    let fold = |init: f64, pick: fn(f64, f64) -> f64| deltas.iter().copied().fold(init, pick);
    let mut summary = json!({
        "n_points": points.len(),
        "first_value": round4(points[0].value),
        "final_value": round4(points[points.len() - 1].value),
        "biggest_drop": if deltas.is_empty() { 0.0 } else { round4(fold(f64::INFINITY, f64::min)) },
        "biggest_gain": if deltas.is_empty() { 0.0 } else { round4(fold(f64::NEG_INFINITY, f64::max)) },
        "n_turning_points": tps.len(),
    });
    if let Some(w) = won {
        summary["won"] = json!(w);
    }
    json!({
        "available": true,
        "points": points.iter().map(WpaPoint::to_value).collect::<Vec<_>>(),
        "turning_points": tps,
        "summary": summary,
        "caveats": WPA_CAVEATS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_seed_matches_python() {
        assert_eq!(hash_seed("run_1"), 1);
        assert_eq!(hash_seed("run_42"), 42);
        // "abc": ((97*131 + 98)*131 + 99) & 0x7FFFFFFF
        assert_eq!(hash_seed("abc"), ((97 * 131 + 98) * 131 + 99) & 0x7FFF_FFFF);
        assert_eq!(hash_seed("run_"), hash_seed_slow("run_"));
        assert_eq!(hash_seed(""), 0);
    }

    fn hash_seed_slow(s: &str) -> i32 {
        let mut h: i64 = 0;
        for ch in s.chars() {
            h = (h * 131 + i64::from(u32::from(ch))) & 0x7FFF_FFFF;
        }
        i32::try_from(h).unwrap()
    }

    #[test]
    fn stop_points() {
        assert!(is_stop_point(&json!({"decision": "card_reward"})));
        assert!(!is_stop_point(&json!({"decision": "combat_play"})));
        assert!(!is_stop_point(&json!({"decision": "game_over"})));
        assert!(!is_stop_point(&json!({"decision": 3})));
        assert!(!is_stop_point(&json!({})));
    }

    #[test]
    fn verdict_bands() {
        assert_eq!(verdict_band(true, None), "matched");
        assert_eq!(verdict_band(false, None), "differed");
        assert_eq!(verdict_band(false, Some(0.05)), "near_tie");
        assert_eq!(verdict_band(false, Some(0.051)), "differed");
    }

    #[test]
    fn deck_delta_folds_upgrades() {
        let prev = vec![
            json!({"name": "Strike", "upgraded": false}),
            json!({"name": "Strike", "upgraded": false}),
            json!({"name": "Bash", "upgraded": false}),
        ];
        let cur = vec![
            json!({"name": "Strike", "upgraded": true}),
            json!({"name": "Bash", "upgraded": false}),
            json!({"name": "Inflame", "upgraded": false}),
        ];
        let (added, removed, upgraded) = deck_delta(&prev, &cur);
        assert_eq!(added, vec!["Inflame"]);
        assert_eq!(removed, vec!["Strike"]);
        assert_eq!(upgraded, vec!["Strike"]);
    }

    #[test]
    fn context_diff_first_stop_is_absolutes_only() {
        let cur = json!({"floor": 3, "player": {"hp": 70, "max_hp": 80, "gold": 99,
            "deck": [{"name": "Strike"}], "relics": [{"name": "Burning Blood"}]}});
        let d = context_diff(None, &cur, 2);
        assert_eq!(d["from_floor"], Value::Null);
        assert_eq!(d["to_floor"], json!(3));
        assert_eq!(d["deck"]["added"], json!([]));
        assert_eq!(d["hp_delta"], Value::Null);
        assert_eq!(
            d["lines"],
            json!(["HP 70/80", "Gold 99", "2 decisions auto-played"])
        );
    }

    #[test]
    fn context_diff_lines_read_what_moved() {
        let prev = json!({"floor": 3, "player": {"hp": 70, "max_hp": 80, "gold": 99,
            "deck": [{"name": "Strike"}], "relics": [], "potions": [{"name": "Fire Potion"}]}});
        let cur = json!({"floor": 5, "player": {"hp": 52, "max_hp": 80, "gold": 120,
            "deck": [{"name": "Strike"}, {"name": "Inflame"}],
            "relics": [{"name": "Bag of Preparation"}], "potions": []}});
        let d = context_diff(Some(&prev), &cur, 1);
        assert_eq!(
            d["lines"],
            json!([
                "Added Inflame",
                "Relic Bag of Preparation",
                "Used potion Fire Potion",
                "HP 52/80 (−18)",
                "Gold 120 (+21)",
                "1 decision auto-played"
            ])
        );
        assert_eq!(d["hp_delta"], json!(-18));
        assert_eq!(d["deck_size"], json!(2));
    }

    #[test]
    fn wpa_review_ranks_turning_points() {
        let rows = vec![
            json!({"node_id": 0, "idx": 0, "decision": "event_choice", "floor": 0, "act": 1, "value": 0.5, "taken_label": "a"}),
            json!({"node_id": 3, "idx": 3, "decision": "map_select", "floor": 1, "act": 1, "value": 0.45, "taken_label": "b"}),
            json!({"node_id": 9, "idx": 9, "decision": "card_reward", "floor": 2, "act": 1, "value": 0.6, "taken_label": "c"}),
            json!({"node_id": 12, "idx": 10, "decision": "shop", "floor": 3, "act": 1, "value": 0.61, "taken_label": "d"}),
        ];
        let out = wpa_review(&rows, 3, 0.02, Some(false), "none");
        assert_eq!(out["available"], json!(true));
        assert_eq!(out["summary"]["n_points"], json!(4));
        assert_eq!(out["summary"]["won"], json!(false));
        let tps = out["turning_points"].as_array().unwrap();
        assert_eq!(tps.len(), 2);
        assert_eq!(tps[0]["idx"], json!(3));
        assert_eq!(tps[0]["swing"], json!(0.15));
        assert_eq!(
            tps[0]["message"],
            json!("floor 1 (map_select): after “b” your outlook rose +0.150 (0.450 → 0.600) (spans the fight that followed)")
        );
        assert_eq!(tps[1]["idx"], json!(0));
        assert!(tps[1]["message"]
            .as_str()
            .unwrap()
            .contains("dropped -0.050"));
        let none = wpa_review(&rows[..1], 3, 0.02, None, "nothing");
        assert_eq!(none["available"], json!(false));
        assert_eq!(none["reason"], json!("nothing"));
        assert_eq!(none["caveats"].as_array().unwrap().len(), 3);
    }
}
