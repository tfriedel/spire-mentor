//! The live advisor's per-fight engine mirror (GH #1426; was
//! `realgame_advisor_mirror.AdvisorMirror`).
//!
//! The why of each rule (following a human's plays, the end-of-turn catch
//! up, the three replays that re-park a fork on a screen, the multi-pick
//! memory, the stale-report echo) stays on the Python wrapper's docstrings,
//! which the advisor's readers know and whose tests hold this port to them.
//!
//! Every method is total: a failure drops the mirror and the caller falls
//! back to the stateless path. Nothing here does I/O: what the Python class
//! logged is queued in [`AdvisorMirror::take_log`], and the divergences its
//! mirrors recorded in [`AdvisorMirror::take_divergences`], for the host to
//! write.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};
use sts2_core::CombatState;

use crate::board_targets::BoardTargets;
use crate::card_select::{self, StatePicker};
use crate::combat_mirror::{CombatMirror, Divergence, MirrorError, Policy};
use crate::i18n::{tr, tr_dyn, trf};
use crate::mirror_state::{parse_bridge_action, step_result_name};
use crate::py::{self, get, list, truthy};
use crate::search::RecommendConfig;

/// The park inside the end-of-turn flush (`_RETAIN_PARK`).
const RETAIN_PARK: &str = "retain_at_turn_end";
/// The choice kinds [`AdvisorMirror::replay_end_turn`] answers.
const END_TURN_PARKS: [&str; 2] = [RETAIN_PARK, "discard_from_hand"];

/// The counters the run log prints, in the Python dict's order.
const STAT_KEYS: [&str; 12] = [
    "seeds",
    "drops",
    "reconciles",
    "desyncs",
    "replays_attempted",
    "replays_landed",
    "searched_selects",
    "followed",
    "unfollowable",
    "turns_followed",
    "stale_reports_skipped",
    "teammate_adopted",
];

/// A log line the Python class wrote, for the host's logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warning,
}

/// Which mirror a replay works on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum On {
    /// The fight's own mirror.
    Live,
    /// The throwaway fork an answer is worked out on.
    Fork,
}

/// A card as the pair that survives a shrinking offer list.
type Identity = (String, bool);

/// A multi-pick selection's searched picks (`remember_picks`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Picks {
    source: String,
    offer: Vec<Identity>,
    plan: Vec<Identity>,
    /// Who planned it, for the later picks' note ("combat MCTS", Instinct's
    /// [`crate::instinct_combat::CHOICE_NOTE`]).
    head: String,
    /// The plan's [`set_line`], repeated on each later pick.
    set_line: Option<String>,
}

/// The note head of a pick the combat MCTS planned.
const SEARCH_HEAD: &str = "combat MCTS";

/// A selection that may take more than one card (`max_select`, alias
/// `max_count`, above 1): "any number", "up to N", "exactly N".
#[must_use]
pub fn takes_several(state: &Value) -> bool {
    let v = state
        .get("max_select")
        .unwrap_or_else(|| get(state, "max_count"));
    truthy(v) && py::int_of(v) > 1
}

/// The set a multi-card selection's answer takes, for its note (GH #1770):
/// the selection's verb, how many, and which, in pick order -- "Replace 2:
/// Defend, Defend". An answer to such a screen names its whole set, not
/// only the first card, and says so when it takes none ("Replace none").
/// `None` on a single-card selection, whose `best` says it all.
#[must_use]
pub fn set_line(state: &Value, labels: &[String], picks: &[i64]) -> Option<String> {
    if !takes_several(state) {
        return None;
    }
    let verb = tr_dyn(crate::hand_select::verb_of(state));
    let names: Vec<&str> = picks
        .iter()
        .filter_map(|&i| usize::try_from(i).ok().and_then(|i| labels.get(i)))
        .map(String::as_str)
        .collect();
    Some(if names.is_empty() {
        trf("{verb} none", &[("verb", &verb)])
    } else {
        let names: Vec<String> = names.iter().map(|n| (*n).to_string()).collect();
        trf(
            "{verb} {n}: {cards}",
            &[
                ("verb", &verb),
                ("n", &names.len()),
                ("cards", &crate::i18n::join_list(&names, ", ")),
            ],
        )
    })
}

/// The answer to an in-combat selection from its planned `picks` (option
/// positions into `cards`, in pick order): every planned card `is_best`,
/// the first one `best` (a skip when there is none), and `head` plus the
/// [`set_line`] as the note.
#[must_use]
pub fn picked_set_answer(
    cards: &[Value],
    picks: &[i64],
    head: &str,
    set_line: Option<&str>,
) -> Value {
    let labels = card_labels(cards);
    let options: Vec<Value> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let at = i64::try_from(i).unwrap_or(-1);
            json!({"label": label, "is_best": picks.contains(&at), "index": card_index(cards, i), "action": "choose"})
        })
        .collect();
    let note = match set_line {
        Some(line) => format!("{head} · {line}"),
        None => head.to_string(),
    };
    let best = match picks
        .first()
        .and_then(|&b| usize::try_from(b).ok())
        .filter(|&b| b < options.len())
    {
        Some(b) => {
            json!({"action": "choose", "index": options[b]["index"], "label": labels[b]})
        }
        // A skippable selection the search answered with nothing.
        None => json!({"action": "skip", "label": tr("select none")}),
    };
    json!({"note": note, "options": options, "best": best})
}

/// The host's board import for the answer path: the screen's embedded
/// combat, the character (display name), the ascension, the combat id and
/// the seed → the engine wire, or the Python exception as `(class name,
/// message)`. The advisor's own is [`crate::board::combat_wire`]; the Python
/// host passes `realgame_advisor_mirror.combat_wire`, which logs the
/// import's warnings.
pub type WireFor<'a> =
    dyn FnMut(&Value, Option<&str>, i64, i64, u64) -> Result<Value, (String, String)> + 'a;

/// `AdvisorMirror`: the advisor's per-fight [`CombatMirror`], or nothing.
#[derive(Debug)]
pub struct AdvisorMirror {
    pub enabled: bool,
    damage_trace: bool,
    mirror: Option<CombatMirror>,
    fork: Option<CombatMirror>,
    key: Option<(Value, Value)>,
    /// The last combat advice's `best.target_index`, in board space (GH
    /// #1918): the target a killing play reports as null.
    last_advice_target: Option<Value>,
    stats: BTreeMap<&'static str, i64>,
    pub declines: BTreeMap<String, i64>,
    pub desync_paths: BTreeMap<String, i64>,
    pub adopted_paths: BTreeMap<String, i64>,
    said: BTreeSet<String>,
    stale_report: Option<(Identity, u64)>,
    generation: u64,
    picks: Option<Picks>,
    log: Vec<(Level, String)>,
    divergences: Vec<Divergence>,
}

/// `_INDEX_RE.sub("[]", path)`: list indices folded.
fn fold_indices(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut rest = path;
    while let Some(open) = rest.find('[') {
        match rest[open..].find(']') {
            Some(close) => {
                out.push_str(&rest[..open]);
                out.push_str("[]");
                rest = &rest[open + close + 1..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// `_enemy_hp`: the living enemies' HP, in order.
#[must_use]
pub fn enemy_hp(wire: &Value) -> Vec<Value> {
    list(wire, "encounter")
        .iter()
        .filter(|e| !truthy(get(e, "is_dead")))
        .map(|e| get(e, "hp").clone())
        .collect()
}

fn same_values(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| py::eq(x, y))
}

/// `_selected_count`: how many cards the screen already has selected.
#[must_use]
pub fn selected_count(state: &Value) -> i64 {
    let v = get(state, "selected_count");
    let n = match v {
        Value::Bool(b) => i64::from(*b),
        Value::Number(_) => py::int_of(v),
        Value::String(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    };
    n.max(0)
}

/// `_card_labels`.
fn card_labels(cards: &[Value]) -> Vec<String> {
    cards
        .iter()
        .enumerate()
        .map(|(i, c)| {
            for key in ["name", "id"] {
                let v = get(c, key);
                if truthy(v) {
                    return py::str_of(v);
                }
            }
            trf("card {n}", &[("n", &i)])
        })
        .collect()
}

/// `int(cards[i].get("index", i))`.
fn card_index(cards: &[Value], i: usize) -> Value {
    cards
        .get(i)
        .and_then(|c| c.get("index"))
        .map_or_else(|| json!(i), |v| json!(py::int_of(v)))
}

/// `_confirm_selection_answer`: confirm the cards already selected.
#[must_use]
pub fn confirm_selection_answer(state: &Value) -> Value {
    let cards = list(state, "cards");
    let options: Vec<Value> = card_labels(cards)
        .into_iter()
        .enumerate()
        .map(|(i, label)| {
            json!({"label": label, "is_best": false, "index": card_index(cards, i), "action": "choose"})
        })
        .collect();
    json!({
        "note": trf("{n} pick(s) done · confirm", &[("n", &selected_count(state))]),
        "options": options,
        "best": {"action": "skip", "label": tr("confirm")},
    })
}

/// `_identity`.
fn identity(card: &Value) -> Identity {
    if !card.is_object() {
        return (String::new(), false);
    }
    let id = get(card, "id");
    let name = if truthy(id) { id } else { get(card, "name") };
    let id = if truthy(name) {
        py::str_of(name)
    } else {
        String::new()
    };
    (id.to_uppercase(), truthy(get(card, "upgraded")))
}

/// `str(v or "").strip().upper()`.
fn upper_name(v: &Value) -> String {
    if truthy(v) {
        py::str_of(v).trim().to_uppercase()
    } else {
        String::new()
    }
}

/// `isinstance(v, int)` (a `bool` is one).
fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

/// `probe.apply_step(action)` on a clone.
fn apply_step(state: &mut CombatState, action: &Value) -> Result<&'static str, String> {
    let decoded = parse_bridge_action(action)?;
    Ok(step_result_name(sts2_core::apply(state, decoded)))
}

fn state_dict(state: &CombatState) -> Result<Value, String> {
    sts2_wire::emit::combat_state_to_value(state).map_err(|e| e.to_string())
}

/// A failed mirror call as the Python wrapper raised it.
fn mirror_exc(label: &str, e: &MirrorError) -> String {
    match e {
        MirrorError::Diverged(_) | MirrorError::NoState => format!("[mirror {label}] {e}"),
        other => other.to_string(),
    }
}

const fn diverged(e: &MirrorError) -> bool {
    matches!(e, MirrorError::Diverged(_) | MirrorError::NoState)
}

impl AdvisorMirror {
    /// A mirror with no fight. `damage_trace` arms each fight's damage
    /// trace (`STS2_MIRROR_DAMAGE_TRACE`).
    #[must_use]
    pub fn new(enabled: bool, damage_trace: bool) -> Self {
        Self {
            enabled,
            damage_trace,
            mirror: None,
            fork: None,
            key: None,
            last_advice_target: None,
            stats: STAT_KEYS.iter().map(|k| (*k, 0)).collect(),
            declines: BTreeMap::new(),
            desync_paths: BTreeMap::new(),
            adopted_paths: BTreeMap::new(),
            said: BTreeSet::new(),
            stale_report: None,
            generation: 0,
            picks: None,
            log: Vec::new(),
            divergences: Vec::new(),
        }
    }

    /// Bumped on every seed: a new fight is a new mirror.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// The counters, as the Python `stats` dict.
    #[must_use]
    pub fn stats(&self) -> Value {
        Value::Object(
            STAT_KEYS
                .iter()
                .map(|k| ((*k).to_string(), json!(self.stats[k])))
                .collect(),
        )
    }

    fn bump(&mut self, key: &'static str, by: i64) {
        *self.stats.entry(key).or_insert(0) += by;
    }

    fn say(&mut self, level: Level, msg: String) {
        self.log.push((level, msg));
    }

    /// The log lines since the last call, oldest first.
    pub fn take_log(&mut self) -> Vec<(Level, String)> {
        std::mem::take(&mut self.log)
    }

    /// The divergences every mirror recorded since the last call.
    pub fn take_divergences(&mut self) -> Vec<Divergence> {
        self.collect_divergences();
        std::mem::take(&mut self.divergences)
    }

    fn collect_divergences(&mut self) {
        for m in [self.mirror.as_mut(), self.fork.as_mut()]
            .into_iter()
            .flatten()
        {
            self.divergences.extend(m.take_divergences());
        }
    }

    /// `decline`: count (and, once per reason, announce) a decline.
    pub fn decline(&mut self, reason: &str) {
        *self.declines.entry(reason.to_string()).or_insert(0) += 1;
        if self.said.insert(reason.to_string()) {
            self.say(
                Level::Info,
                format!("searched card_select declined ({reason}) — first of this kind"),
            );
        }
    }

    /// `live`: the mirror, iff one is seeded and has not been dropped.
    pub fn live(&mut self) -> Option<&mut CombatMirror> {
        self.mirror.as_mut().filter(|m| m.alive())
    }

    /// The carried fight's enemy uid renames since the last call
    /// ([`CombatMirror::take_enemy_remap`], GH #1651).
    pub fn take_enemy_remap(&mut self) -> Option<Vec<(u32, u32)>> {
        self.mirror
            .as_mut()
            .and_then(CombatMirror::take_enemy_remap)
    }

    /// The live mirror's state.
    #[must_use]
    pub fn live_state(&self) -> Option<&CombatState> {
        self.mirror
            .as_ref()
            .and_then(CombatMirror::mirrored)
            .map(|m| &m.state)
    }

    /// The live mirror or the fork, alive or not.
    pub fn mirror_mut(&mut self, on: On) -> Option<&mut CombatMirror> {
        match on {
            On::Live => self.mirror.as_mut(),
            On::Fork => self.fork.as_mut(),
        }
    }

    /// Fork the live fight (the throwaway an answer is worked out on);
    /// whether there was one to fork.
    pub fn fork_live(&mut self, label: &str) -> bool {
        self.drop_fork("replaced");
        self.fork = self.mirror.as_ref().and_then(|m| m.fork(label));
        self.fork.is_some()
    }

    /// Drop the fork, keeping what it recorded.
    pub fn drop_fork(&mut self, reason: &str) {
        if let Some(mut f) = self.fork.take() {
            f.drop_fight(reason);
            self.divergences.extend(f.take_divergences());
        }
    }

    fn target(&mut self, on: On) -> Option<&mut CombatMirror> {
        match on {
            On::Live => self.mirror.as_mut(),
            On::Fork => self.fork.as_mut(),
        }
        .filter(|m| m.alive())
    }

    /// The mod's `target_index` (an index into the board's `enemies[]`) as
    /// the `on` mirror's engine slot (`board_targets`): the carried fight
    /// keeps corpses the game removed. Null stays null; an index with no
    /// enemy behind it passes through, for the engine to refuse.
    fn engine_target(&self, on: On, board: &Value) -> Value {
        let Some(t) = as_int(board) else {
            return Value::Null;
        };
        let slot = self
            .target_state(on)
            .and_then(|s| BoardTargets::of(s).to_engine(t))
            .unwrap_or(t);
        json!(slot)
    }

    fn target_state(&self, on: On) -> Option<&CombatState> {
        match on {
            On::Live => self.mirror.as_ref(),
            On::Fork => self.fork.as_ref(),
        }
        .and_then(CombatMirror::mirrored)
        .map(|m| &m.state)
    }

    /// `drop`: forget the fight.
    pub fn drop_fight(&mut self, reason: &str) {
        if let Some(mut m) = self.mirror.take() {
            self.bump("drops", 1);
            m.drop_fight(reason);
            self.divergences.extend(m.take_divergences());
        }
        self.key = None;
        self.picks = None;
    }

    /// `on_rewind`: a `snapshot_restored` / `combat_undone` of type `kind`.
    pub fn on_rewind(&mut self, kind: &Value) {
        self.stale_report = None;
        self.drop_fight(&format!("rewind: {}", py::str_of(kind)));
    }

    /// `on_combat_board`: seed or reconcile for one combat decision; whether
    /// the mirror is carrying this fight.
    pub fn on_combat_board(&mut self, wire: &Value, state: &Value) -> bool {
        if !self.enabled {
            return false;
        }
        let key = (get(state, "act").clone(), get(state, "floor").clone());
        let same = self
            .key
            .as_ref()
            .is_some_and(|k| py::eq(&k.0, &key.0) && py::eq(&k.1, &key.1));
        if self.mirror.is_some() && !same {
            let note = self.note();
            self.say(Level::Info, format!("advisor {note}"));
            self.stale_report = None;
            self.drop_fight("new fight");
        }
        if self.mirror.is_none() {
            self.mirror = Some(CombatMirror::new(
                Policy::Observe,
                "advisor",
                self.damage_trace,
            ));
            self.key = Some(key);
            self.bump("seeds", 1);
            self.generation += 1;
        }
        self.catch_up_turns(wire);
        let teammates = {
            let n = get(state, "player_count");
            let n = if truthy(n) { py::int_of(n) } else { 1 };
            u32::try_from((n - 1).max(0)).unwrap_or(u32::MAX)
        };
        let Some(m) = self.mirror.as_mut() else {
            return false;
        };
        m.teammates = teammates;
        let desyncs = match m.reconcile(wire) {
            Ok(d) => d,
            Err(e) => {
                let msg = mirror_exc("advisor", &e);
                if diverged(&e) {
                    self.say(
                        Level::Info,
                        format!("advisor mirror dropped on reconcile: {msg}"),
                    );
                    self.drop_fight("reconcile refused");
                } else {
                    self.say(
                        Level::Warning,
                        format!("advisor mirror reconcile failed: {msg}"),
                    );
                    self.drop_fight("reconcile error");
                }
                return false;
            }
        };
        let adopted: Vec<String> = m
            .last_adopted
            .iter()
            .map(|d| fold_indices(&d.path))
            .collect();
        self.bump("reconciles", 1);
        self.bump("desyncs", i64::try_from(desyncs.len()).unwrap_or(0));
        for d in &desyncs {
            *self.desync_paths.entry(fold_indices(&d.path)).or_insert(0) += 1;
        }
        self.bump(
            "teammate_adopted",
            i64::try_from(adopted.len()).unwrap_or(0),
        );
        for path in adopted {
            *self.adopted_paths.entry(path).or_insert(0) += 1;
        }
        self.collect_divergences();
        self.live().is_some()
    }

    /// `_catch_up_turns`: end the mirror's turn while the world is on a
    /// later round, at most 3.
    fn catch_up_turns(&mut self, wire: &Value) {
        let Some(want) = as_int(get(wire, "round")) else {
            return;
        };
        let Some(m) = self.live() else {
            return;
        };
        let Some(have) = m
            .mirrored()
            .and_then(|s| state_dict(&s.state).ok())
            .and_then(|d| as_int(get(&d, "round")))
        else {
            return;
        };
        if have >= want {
            return;
        }
        if want - have > 3 {
            self.drop_fight(&format!("round gap too wide to follow ({have} -> {want})"));
            return;
        }
        for _ in 0..want - have {
            let Some(m) = self.mirror.as_mut() else {
                return;
            };
            match m.advance(&json!({"action": "end_turn"})) {
                Err(e) => {
                    let msg = mirror_exc("advisor", &e);
                    if diverged(&e) {
                        self.say(
                            Level::Info,
                            format!("advisor mirror dropped ending its turn: {msg}"),
                        );
                        self.drop_fight("turn end diverged");
                    } else {
                        self.say(
                            Level::Warning,
                            format!("advisor mirror could not end its turn: {msg}"),
                        );
                        self.drop_fight("turn end error");
                    }
                    return;
                }
                Ok(result) => {
                    self.bump("turns_followed", 1);
                    if matches!(result, "combat_won" | "combat_lost") {
                        self.drop_fight(&format!("mirror ended the fight early ({result})"));
                        return;
                    }
                }
            }
        }
    }

    /// Advance the live mirror by `action`, the follow's error handling.
    fn follow(&mut self, action: &Value, what: &str) {
        let Some(m) = self.mirror.as_mut() else {
            return;
        };
        match m.advance(action) {
            Ok(_) => self.bump("followed", 1),
            Err(e) => {
                let msg = mirror_exc("advisor", &e);
                if diverged(&e) {
                    self.say(
                        Level::Info,
                        format!("advisor mirror dropped following {what}: {msg}"),
                    );
                    self.drop_fight("following diverged");
                } else {
                    self.say(
                        Level::Warning,
                        format!("advisor mirror could not follow {what}: {msg}"),
                    );
                    self.drop_fight("follow error");
                }
            }
        }
        self.collect_divergences();
    }

    /// Remember the last combat advice's board target (`best.target_index`);
    /// null (an untargeted advice) is stored as nothing.
    pub fn set_advice_target(&mut self, t: Value) {
        self.last_advice_target = if t.is_null() { None } else { Some(t) };
    }

    /// GH #1918: the mod reports `target_index: null` for a card that killed
    /// its target (the target has left the board). The mirror's board is the
    /// pre-play one, so the target is still legal: recover an ENGINE slot
    /// from the advised target, else the only legal one, else the first.
    /// `Null` for a card with no legal target (not targeted: unchanged).
    fn recover_card_target(&mut self, on: On, hand_idx: usize) -> Value {
        let legal: Vec<i64> = match self.target_state(on) {
            Some(s) => sts2_wire::emit::legal_actions(s)
                .as_array()
                .into_iter()
                .flatten()
                .filter(|a| {
                    get(a, "action") == "play"
                        && as_int(get(a, "card_index")) == i64::try_from(hand_idx).ok()
                })
                .filter_map(|a| as_int(get(a, "target_index")))
                .collect(),
            None => return Value::Null,
        };
        let Some(&first) = legal.first() else {
            return Value::Null;
        };
        if let Some(t) = self.last_advice_target.clone() {
            let slot = self.engine_target(on, &t);
            if as_int(&slot).is_some_and(|s| legal.contains(&s)) {
                return slot;
            }
        }
        if legal.len() > 1 {
            self.say(
                Level::Debug,
                format!("advisor mirror: null target, no advice to recover it from; guessing slot {first}"),
            );
        }
        json!(first)
    }

    /// `on_world_action`: advance the mirror through an action the world
    /// just committed (a `human_action`).
    pub fn on_world_action(&mut self, msg: &Value) {
        if self.live().is_none() || get(msg, "decision_type") != "combat_action" {
            return;
        }
        if get(msg, "action") == "end_turn" {
            return;
        }
        let card = get(msg, "card");
        let drink = get(msg, "potion");
        if drink.is_object() && truthy(get(drink, "id")) {
            let src = upper_name(get(drink, "id"));
            match self.drink_action(
                On::Live,
                &src,
                get(msg, "target_index"),
                get(msg, "ally_index"),
            ) {
                None => {
                    self.drop_fight(&format!("{src} not in the mirror's belt to follow"));
                    self.bump("unfollowable", 1);
                }
                Some(action) => self.follow(&action, &src),
            }
            return;
        }
        if !card.is_object() || !truthy(get(card, "id")) {
            self.drop_fight(&format!(
                "unfollowable action: {}",
                py::str_of(get(msg, "action"))
            ));
            self.bump("unfollowable", 1);
            return;
        }
        let hand = match self.live_state().map(state_dict) {
            Some(Ok(d)) => list(get(&d, "deck"), "hand").to_vec(),
            Some(Err(e)) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: cannot read hand to follow: {e}"),
                );
                self.drop_fight("hand unreadable");
                return;
            }
            None => return,
        };
        let want = (
            py::str_of(get(card, "id")).to_uppercase(),
            truthy(get(card, "upgraded")),
        );
        if self.expect_stale_report(&want) {
            return;
        }
        let Some(idx) = hand.iter().position(|c| {
            let id = c.get("id").map_or_else(String::new, py::str_of);
            id.to_uppercase() == want.0 && truthy(get(c, "upgraded")) == want.1
        }) else {
            self.drop_fight(&format!("{} not in the mirror's hand to follow", want.0));
            self.bump("unfollowable", 1);
            return;
        };
        let target = if get(msg, "target_index").is_null() {
            self.recover_card_target(On::Live, idx)
        } else {
            self.engine_target(On::Live, get(msg, "target_index"))
        };
        let mut action = json!({"action": "play", "card_index": idx, "target_index": target});
        if let Some(ally) = as_int(get(msg, "ally_index")) {
            action["ally_index"] = json!(ally);
        }
        self.follow(&action, &want.0);
    }

    /// `_expect_stale_report`: is this play report the echo of a replay?
    fn expect_stale_report(&mut self, want: &Identity) -> bool {
        match &self.stale_report {
            Some((card, generation)) if card == want && *generation != self.generation => {
                self.stale_report = None;
                self.bump("stale_reports_skipped", 1);
                true
            }
            _ => false,
        }
    }

    /// `source_is_replayable`: can a replay possibly work for this source?
    pub fn source_is_replayable(&mut self, source_model: &Value, source_kind: &Value) -> bool {
        let src = upper_name(source_model);
        let kind = if truthy(source_kind) {
            py::str_of(source_kind).trim().to_lowercase()
        } else {
            String::new()
        };
        if src.is_empty() {
            self.decline("wire named no source_model");
            return false;
        }
        if !kind.is_empty() && kind != "card" && kind != "potion" {
            self.decline(&format!("{src} is a {kind}, not a card or a potion"));
            return false;
        }
        true
    }

    /// `_drink_action`: the engine's own drink of potion `src`, or `None`.
    fn drink_action(&mut self, on: On, src: &str, target: &Value, ally: &Value) -> Option<Value> {
        let state = self.target_state(on)?;
        let belt = match state_dict(state) {
            Ok(d) => list(get(&d, "player"), "potions").to_vec(),
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: cannot read the belt: {e}"),
                );
                return None;
            }
        };
        let slots: Vec<i64> = belt
            .iter()
            .enumerate()
            .filter(|(_, p)| upper_name(p) == src)
            .map(|(i, _)| i64::try_from(i).unwrap_or(-1))
            .collect();
        if slots.is_empty() {
            return None;
        }
        let ally = as_int(ally).map_or(Value::Null, |a| json!(a));
        let state = self.target_state(on)?;
        let legal = sts2_wire::emit::legal_actions(state);
        let mut action = legal
            .as_array()
            .into_iter()
            .flatten()
            .find(|a| {
                let kind = get(a, "action");
                (kind == "drink_potion" || kind == "potion")
                    && slots.contains(&a.get("slot").map_or(-1, py::int_of))
                    && py::eq(get(a, "ally_index"), &ally)
            })?
            .clone();
        if !target.is_null() {
            action["target_index"] = self.engine_target(on, target);
        }
        Some(action)
    }

    /// Advance the replay's mirror; the replay's error handling (`what` names
    /// the replay: `drinking X` / `drink replay`).
    fn commit(
        &mut self,
        on: On,
        action: &Value,
        dropped_msg: &str,
        failed_msg: &str,
        diverged_reason: &str,
        error_reason: &str,
    ) -> Option<&'static str> {
        let m = self.target(on)?;
        let out = m.advance(action);
        self.collect_divergences();
        match out {
            Ok(r) => Some(r),
            Err(e) => {
                let msg = mirror_exc(
                    if on == On::Fork {
                        "advisor-answer"
                    } else {
                        "advisor"
                    },
                    &e,
                );
                if diverged(&e) {
                    self.say(Level::Info, format!("{dropped_msg}: {msg}"));
                    self.drop_fight(diverged_reason);
                } else {
                    self.say(Level::Warning, format!("{failed_msg}: {msg}"));
                    self.drop_fight(error_reason);
                }
                None
            }
        }
    }

    /// `replay_opening_drink`: re-drink the potion that opened this screen.
    pub fn replay_opening_drink(
        &mut self,
        on: On,
        source_model: &Value,
        world_wire: &Value,
        target_index: &Value,
    ) -> bool {
        let src = upper_name(source_model);
        let Some(state) = self.target_state(on).filter(|_| self.target_alive(on)) else {
            self.decline("no live mirror");
            return false;
        };
        if sts2_wire::emit::pending_choice(state).is_some() {
            return true;
        }
        if src.is_empty() {
            self.decline("wire named no source_model");
            return false;
        }
        let Some(action) = self.drink_action(on, &src, target_index, &Value::Null) else {
            self.decline(&format!("{src} is not drinkable in the mirror"));
            return false;
        };
        self.bump("replays_attempted", 1);
        let Some(mut probe) = self.target_state(on).cloned() else {
            return false;
        };
        let parked = match apply_step(&mut probe, &action) {
            Ok(p) => p,
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: {src} could not be drunk: {e}"),
                );
                self.decline(&format!("{src} could not be drunk in the mirror"));
                return false;
            }
        };
        if parked != "pending_choice" {
            self.decline(&format!("{src} drunk but parked no choice ({parked})"));
            return false;
        }
        match state_dict(&probe) {
            Ok(d) if same_values(&enemy_hp(&d), &enemy_hp(world_wire)) => {}
            Ok(_) => {
                self.decline(&format!("{src} drunk but the board did not match"));
                return false;
            }
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: cannot compare boards for {src}: {e}"),
                );
                return false;
            }
        }
        let Some(result) = self.commit(
            on,
            &action,
            &format!("advisor mirror dropped drinking {src}"),
            &format!("advisor mirror drink replay of {src} failed"),
            "drink replay diverged",
            "drink replay error",
        ) else {
            return false;
        };
        if result != "pending_choice" {
            self.decline(&format!("{src} drunk but parked no choice ({result})"));
            return false;
        }
        self.bump("replays_landed", 1);
        true
    }

    fn target_alive(&self, on: On) -> bool {
        match on {
            On::Live => self.mirror.as_ref(),
            On::Fork => self.fork.as_ref(),
        }
        .is_some_and(CombatMirror::alive)
    }

    /// `remember_picks`: keep a MULTI-pick search's picks for the screens
    /// still to come.
    pub fn remember_picks(&mut self, source_model: &Value, offered: &[Value], picks: &[i64]) {
        self.remember_planned_picks(source_model, offered, picks, tr(SEARCH_HEAD), None);
    }

    /// [`Self::remember_picks`], with who planned them and the plan's
    /// [`set_line`] for the later picks' note.
    pub fn remember_planned_picks(
        &mut self,
        source_model: &Value,
        offered: &[Value],
        picks: &[i64],
        head: &str,
        set_line: Option<String>,
    ) {
        self.picks = None;
        if picks.len() < 2 {
            return;
        }
        let ids: Vec<Identity> = picks
            .iter()
            .filter_map(|&i| usize::try_from(i).ok().and_then(|i| offered.get(i)))
            .map(identity)
            .collect();
        if ids.len() < 2 {
            return;
        }
        self.picks = Some(Picks {
            source: upper_name(source_model),
            offer: offered.iter().map(identity).collect(),
            plan: ids,
            head: head.to_string(),
            set_line,
        });
    }

    /// `next_remembered_pick`: the next searched pick for this selection,
    /// as an index into `offered`; forgets the selection when the offer is
    /// not exactly the original minus the plan's first picks.
    pub fn next_remembered_pick(
        &mut self,
        source_model: &Value,
        offered: &[Value],
    ) -> Option<usize> {
        let held = self.picks.clone()?;
        let now_source = upper_name(source_model);
        let forget = |me: &mut Self| {
            me.picks = None;
            None
        };
        if !now_source.is_empty() && now_source != held.source {
            return forget(self);
        }
        let mut expected = held.offer.clone();
        let now: Vec<Identity> = offered.iter().map(identity).collect();
        let Some(taken) = expected.len().checked_sub(now.len()) else {
            return forget(self);
        };
        if taken >= held.plan.len() {
            return forget(self);
        }
        for ident in &held.plan[..taken] {
            match expected.iter().position(|e| e == ident) {
                Some(i) => {
                    expected.remove(i);
                }
                None => return forget(self),
            }
        }
        let mut a = expected;
        let mut b = now.clone();
        a.sort();
        b.sort();
        if a != b {
            return forget(self);
        }
        let want = &held.plan[taken];
        match now.iter().position(|n| n == want) {
            Some(i) => Some(i),
            None => forget(self),
        }
    }

    /// `_probe_turn_end`: end the turn on a clone; `(park kind, living enemy
    /// HP)` or the decline reason (both `None`: an unreadable board).
    fn probe_turn_end(&mut self, on: On) -> (Option<(String, Vec<Value>)>, Option<String>) {
        let Some(mut probe) = self.target_state(on).cloned() else {
            return (None, Some("the turn would not end in the mirror".into()));
        };
        let parked = match apply_step(&mut probe, &json!({"action": "end_turn"})) {
            Ok(p) => p,
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: the turn would not end: {e}"),
                );
                return (None, Some("the turn would not end in the mirror".into()));
            }
        };
        if parked != "pending_choice" {
            return (
                None,
                Some(format!("the turn ended but parked no choice ({parked})")),
            );
        }
        let kind = sts2_wire::emit::pending_choice(&probe)
            .map(|pc| get(&pc, "kind").clone())
            .filter(truthy)
            .map_or_else(String::new, |k| py::str_of(&k));
        if !END_TURN_PARKS.contains(&kind.as_str()) {
            let shown = if kind.is_empty() {
                "an unnamed choice"
            } else {
                &kind
            };
            return (None, Some(format!("the turn parked on {shown}")));
        }
        match state_dict(&probe) {
            Ok(d) => (Some((kind, enemy_hp(&d))), None),
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: cannot read the probe's board: {e}"),
                );
                (None, None)
            }
        }
    }

    /// `replay_end_turn`: end the fork's turn to reach a screen the END OF
    /// TURN opened.
    pub fn replay_end_turn(&mut self, on: On, world_wire: &Value) -> bool {
        let Some(state) = self.target_state(on).filter(|_| self.target_alive(on)) else {
            self.decline("no live mirror");
            return false;
        };
        if sts2_wire::emit::pending_choice(state).is_some() {
            return true;
        }
        if state.pending_play.is_some() {
            self.decline("a card is mid-play, so the turn is not ending");
            return false;
        }
        self.bump("replays_attempted", 1);
        let (probed, reason) = self.probe_turn_end(on);
        let Some((mut kind, mut mine)) = probed else {
            if let Some(r) = reason {
                self.decline(&r);
            }
            return false;
        };
        let theirs = enemy_hp(world_wire);
        if !same_values(&mine, &theirs) && kind == RETAIN_PARK {
            self.say(
                Level::Info,
                format!(
                    "advisor mirror: end-turn board proof failed at the retain park, mirror enemy hp {} vs world {} -- reconciling the drift and re-probing",
                    py::repr(&Value::Array(mine.clone())),
                    py::repr(&Value::Array(theirs.clone()))
                ),
            );
            let out = self.target(on).map(|m| m.reconcile(world_wire));
            self.collect_divergences();
            match out {
                Some(Err(e)) => {
                    let msg = mirror_exc("advisor-answer", &e);
                    if diverged(&e) {
                        self.say(
                            Level::Info,
                            format!("advisor mirror: the retain-park reconcile diverged: {msg}"),
                        );
                    } else {
                        self.say(
                            Level::Warning,
                            format!("advisor mirror: the retain-park reconcile failed: {msg}"),
                        );
                    }
                }
                None => {}
                Some(Ok(_)) => {
                    let (probed, reason) = self.probe_turn_end(on);
                    let Some(again) = probed else {
                        if let Some(r) = reason {
                            self.decline(&r);
                        }
                        return false;
                    };
                    (kind, mine) = again;
                }
            }
        }
        let _ = kind;
        if !same_values(&mine, &theirs) {
            self.say(
                Level::Info,
                format!(
                    "advisor mirror: end-turn board proof failed, mirror enemy hp {} vs world {}",
                    py::repr(&Value::Array(mine)),
                    py::repr(&Value::Array(theirs))
                ),
            );
            self.decline("the turn ended but the board did not match");
            return false;
        }
        let Some(result) = self.commit(
            on,
            &json!({"action": "end_turn"}),
            "advisor mirror dropped ending its turn to answer",
            "advisor mirror end-turn replay failed",
            "end-turn replay diverged",
            "end-turn replay error",
        ) else {
            return false;
        };
        if result != "pending_choice" {
            self.decline(&format!("the turn ended but parked no choice ({result})"));
            return false;
        }
        self.bump("replays_landed", 1);
        true
    }

    /// `replay_opening_play`: play the card that opened a select screen, so
    /// the engine parks it, and prove the play against the world's board.
    #[allow(clippy::too_many_lines)]
    pub fn replay_opening_play(
        &mut self,
        on: On,
        source_model: &Value,
        world_wire: &Value,
        source_kind: &Value,
        target_index: &Value,
        upgraded: &Value,
    ) -> bool {
        let src = upper_name(source_model);
        let kind = if truthy(source_kind) {
            py::str_of(source_kind).trim().to_lowercase()
        } else {
            String::new()
        };
        let Some(state) = self.target_state(on).filter(|_| self.target_alive(on)) else {
            self.decline("no live mirror");
            return false;
        };
        if sts2_wire::emit::pending_choice(state).is_some() {
            return true;
        }
        if !self.source_is_replayable(source_model, source_kind) {
            return false;
        }
        let Some(state) = self.target_state(on) else {
            return false;
        };
        let before = match state_dict(state) {
            Ok(d) => d,
            Err(e) => {
                self.say(
                    Level::Debug,
                    format!("advisor mirror: cannot read state for replay: {e}"),
                );
                return false;
            }
        };
        let deck = get(&before, "deck");
        let hand = list(deck, "hand").to_vec();
        let id_of = |c: &Value| {
            c.get("id")
                .map_or_else(String::new, py::str_of)
                .to_uppercase()
        };
        let candidates: Vec<i64> = hand
            .iter()
            .enumerate()
            .filter(|(_, c)| id_of(c) == src)
            .map(|(i, _)| i64::try_from(i).unwrap_or(-1))
            .collect();
        if candidates.is_empty() {
            let elsewhere = ["draw_pile", "discard_pile", "exhaust_pile", "play_pile"]
                .iter()
                .any(|k| list(deck, k).iter().any(|c| id_of(c) == src));
            self.decline(&if elsewhere {
                format!("{src} is in the fight but not in hand")
            } else {
                format!("{src} is not a card")
            });
            return false;
        }
        let want = enemy_hp(world_wire);
        let legal_all = sts2_wire::emit::legal_actions(state);
        let mut legal: Vec<Value> = legal_all
            .as_array()
            .into_iter()
            .flatten()
            .filter(|a| {
                get(a, "action") == "play"
                    && candidates.contains(&a.get("card_index").map_or(-1, py::int_of))
            })
            .cloned()
            .collect();
        if legal.is_empty() {
            return false;
        }
        let upgraded_at = |a: &Value| {
            usize::try_from(py::int_of(get(a, "card_index")))
                .ok()
                .and_then(|i| hand.get(i))
                .is_some_and(|c| truthy(get(c, "upgraded")))
        };
        let mut told = kind == "card" && (!target_index.is_null() || !upgraded.is_null());
        if told {
            let want_target = self.engine_target(on, target_index);
            let named: Vec<Value> = legal
                .iter()
                .filter(|a| {
                    py::eq(get(a, "target_index"), &want_target)
                        && (upgraded.is_null() || upgraded_at(a) == truthy(upgraded))
                })
                .cloned()
                .collect();
            if let Some(first) = named.into_iter().next() {
                legal = vec![first];
            } else {
                self.decline(&format!(
                    "{src} play named by the wire is not legal in the mirror"
                ));
                told = false;
            }
        }
        self.bump("replays_attempted", 1);
        let mut matches: Vec<((Value, bool), Value)> = Vec::new();
        for action in &legal {
            let key = (get(action, "target_index").clone(), upgraded_at(action));
            if matches
                .iter()
                .any(|(k, _)| py::eq(&k.0, &key.0) && k.1 == key.1)
            {
                continue;
            }
            let Some(mut probe) = self.target_state(on).cloned() else {
                return false;
            };
            if apply_step(&mut probe, action).ok() != Some("pending_choice") {
                continue;
            }
            if let Ok(d) = state_dict(&probe) {
                if same_values(&enemy_hp(&d), &want) {
                    matches.push((key, action.clone()));
                }
            }
            if matches.len() > 1 {
                break;
            }
        }
        if matches.len() != 1 {
            if told {
                self.decline(&format!("{src} replayed but the board did not match"));
                self.drop_fight("replay did not reproduce the world board");
                return false;
            }
            let what = if matches.is_empty() {
                "no board matched"
            } else {
                "ambiguous"
            };
            self.decline(&format!("{what} replay of {src}"));
            return false;
        }
        let played = matches.remove(0).1;
        let Some(result) = self.commit(
            on,
            &played,
            &format!("advisor mirror dropped replaying {src}"),
            &format!("advisor mirror replay of {src} failed"),
            "replay diverged",
            "replay error",
        ) else {
            return false;
        };
        if result != "pending_choice" || self.live().is_none() {
            self.decline(&format!("{src} replayed but parked no choice ({result})"));
            return false;
        }
        self.bump("replays_landed", 1);
        self.stale_report = Some(((src, upgraded_at(&played)), self.generation));
        true
    }

    /// `_remembered_pick_answer`: pick 2+ of a selection, the planned pick.
    /// Pick 2+ of a remembered plan: `remembered` (an index into this
    /// screen's `cards`, from [`Self::next_remembered_pick`]) is `best`, and
    /// the plan's other cards still in the offer are `is_best` with it, so
    /// the rest of the set stays marked. The note keeps who planned it, says
    /// which pick this is, and repeats the plan's [`set_line`] (GH #1770).
    fn remembered_pick_answer(&mut self, state: &Value, remembered: usize) -> Value {
        let cards = list(state, "cards");
        self.bump("searched_selects", 1);
        let (head, line, taken, total, rest) = match &self.picks {
            Some(p) => {
                let taken = p.offer.len().saturating_sub(cards.len());
                (
                    p.head.clone(),
                    p.set_line.clone(),
                    taken,
                    p.plan.len(),
                    p.plan.get(taken..).unwrap_or_default().to_vec(),
                )
            }
            None => (tr(SEARCH_HEAD).to_string(), None, 1, 2, Vec::new()),
        };
        // The rest of the plan as positions in this offer, `remembered`
        // first: each identity claims one card it has not claimed yet.
        let mut picks: Vec<i64> = vec![i64::try_from(remembered).unwrap_or(-1)];
        let mut claimed: Vec<usize> = vec![remembered];
        let mut rest = rest.into_iter();
        let _ = rest.next(); // the identity `remembered` already is
        for want in rest {
            if let Some(i) =
                (0..cards.len()).find(|&i| !claimed.contains(&i) && identity(&cards[i]) == want)
            {
                claimed.push(i);
                picks.push(i64::try_from(i).unwrap_or(-1));
            }
        }
        let head = trf(
            "{head} · pick {k} of {total}",
            &[("head", &head), ("k", &(taken + 1)), ("total", &total)],
        );
        picked_set_answer(cards, &picks, &head, line.as_deref())
    }

    /// `answer_card_select`: an IN-COMBAT `card_select` answered by the
    /// combat MCTS (or `picker`) on a fork of the carried fight re-parked on
    /// the world's choice, or `None` (counted as a decline).
    #[allow(clippy::too_many_lines)]
    pub fn answer_card_select(
        &mut self,
        state: &Value,
        cfg: Option<&RecommendConfig>,
        sims: Option<u32>,
        picker: Option<&mut StatePicker<'_>>,
        picker_note: Option<&str>,
        wire_for: &mut WireFor<'_>,
    ) -> Option<Value> {
        let cfg = cfg?;
        if get(state, "combat").is_null() {
            self.decline("no embedded combat (deck screen)");
            return None;
        }
        let cards = list(state, "cards").to_vec();
        if selected_count(state) > 0 {
            if let Some(r) = self.next_remembered_pick(get(state, "source_model"), &cards) {
                return Some(self.remembered_pick_answer(state, r));
            }
            return Some(confirm_selection_answer(state));
        }
        if self.live().is_none() {
            self.decline("no live mirror at the screen");
            return None;
        }
        if let Some(r) = self.next_remembered_pick(get(state, "source_model"), &cards) {
            return Some(self.remembered_pick_answer(state, r));
        }
        let live_state = self.live_state()?;
        let already_parked = sts2_wire::emit::pending_choice(live_state).is_some();
        let source_kind = get(state, "source_kind");
        let skind = if truthy(source_kind) {
            py::str_of(source_kind).trim().to_lowercase()
        } else {
            String::new()
        };
        let end_of_turn = !already_parked && skind == "power" && live_state.pending_play.is_none();
        let hook_sourced =
            !already_parked && matches!(skind.as_str(), "relic" | "affliction" | "enchantment");
        if !already_parked
            && !end_of_turn
            && !hook_sourced
            && !self.source_is_replayable(get(state, "source_model"), source_kind)
        {
            return None;
        }
        let fork = self.mirror.as_ref().and_then(|m| m.fork("advisor-answer"));
        let Some(fork) = fork else {
            self.decline("could not fork the mirror to answer");
            return None;
        };
        self.fork = Some(fork);
        let out = self.answer_on_fork(
            state,
            cfg,
            sims,
            picker,
            wire_for,
            (already_parked, hook_sourced, end_of_turn),
            &skind,
        );
        if let Some(mut fork) = self.fork.take() {
            fork.drop_fight("answered a card_select");
            self.divergences.extend(fork.take_divergences());
        }
        let picks = match out {
            Ok(p) => p,
            Err((type_name, msg)) => {
                self.say(
                    Level::Warning,
                    format!("searched card_select failed, falling back: {msg}"),
                );
                self.decline(&format!("the search could not run ({type_name})"));
                return None;
            }
        };
        let Some(picks) = picks else {
            self.decline(&format!(
                "injection dropped the fork ({})",
                py::str_of(get(state, "screen"))
            ));
            return None;
        };
        let labels = card_labels(&cards);
        if labels.is_empty() {
            self.decline("screen offered no labelled cards");
            return None;
        }
        self.bump("searched_selects", 1);
        // A selection of several cards names its whole set and count in the
        // note, whichever brain planned it (GH #1770).
        let line = set_line(state, &labels, &picks);
        self.remember_planned_picks(
            get(state, "source_model"),
            &cards,
            &picks,
            picker_note.unwrap_or(tr(SEARCH_HEAD)),
            line.clone(),
        );
        let head = picker_note.map_or_else(
            || {
                let mut n = trf(
                    "{head} · {n} pick(s) searched",
                    &[("head", &tr(SEARCH_HEAD)), ("n", &picks.len())],
                );
                if picks.len() > 1 {
                    n.push_str(tr(" (one search per pick)"));
                }
                n
            },
            str::to_string,
        );
        Some(picked_set_answer(&cards, &picks, &head, line.as_deref()))
    }

    /// The fork half of [`Self::answer_card_select`]: re-park the fork on
    /// the world's choice and pick on it. `Ok(None)`: the injection dropped
    /// the fork; `Err`: the Python exception's class name.
    #[allow(clippy::too_many_arguments)]
    fn answer_on_fork(
        &mut self,
        state: &Value,
        cfg: &RecommendConfig,
        sims: Option<u32>,
        picker: Option<&mut StatePicker<'_>>,
        wire_for: &mut WireFor<'_>,
        (already_parked, hook_sourced, end_of_turn): (bool, bool, bool),
        skind: &str,
    ) -> Result<Option<Vec<i64>>, (String, String)> {
        let combat = get(state, "combat");
        let run_state = match get(state, "run_state") {
            r if truthy(r) => r,
            _ => get(combat, "run_state"),
        };
        let character = crate::headless::character_display_name(get(run_state, "character"))
            .map(str::to_string)
            .or_else(|| cfg.character.clone());
        let wire = wire_for(
            combat,
            character.as_deref(),
            crate::board::run_ascension(run_state, cfg.ascension),
            cfg.combat_id,
            cfg.seed,
        )?;
        if already_parked || hook_sourced {
            // The fork inherited the park, or the hook already fired where
            // it stands: straight to the injection.
        } else if end_of_turn {
            self.replay_end_turn(On::Fork, &wire);
        } else if skind == "potion" {
            self.replay_opening_drink(
                On::Fork,
                get(state, "source_model"),
                &wire,
                get(state, "source_target_index"),
            );
        } else {
            self.replay_opening_play(
                On::Fork,
                get(state, "source_model"),
                &wire,
                get(state, "source_kind"),
                get(state, "source_target_index"),
                get(state, "source_upgraded"),
            );
        }
        let Some(mut fork) = self.fork.take() else {
            return Err(("MirrorDivergence".into(), "the fork is gone".into()));
        };
        let mut warnings = Vec::new();
        let out =
            card_select::pick_card_select_for(&mut fork, state, cfg, sims, picker, &mut warnings);
        self.fork = Some(fork);
        for w in warnings {
            self.say(Level::Info, format!("adapter: {w}"));
        }
        let (_, picks) = out.map_err(|e| (e.py_type(), e.py_message()))?;
        Ok(picks)
    }

    /// `note`: the one-line state summary for the run log.
    #[must_use]
    pub fn note(&self) -> String {
        let s = |k: &str| self.stats.get(k).copied().unwrap_or(0);
        let why = self
            .declines
            .iter()
            .map(|(k, v)| format!("{k} x{v}"))
            .collect::<Vec<_>>()
            .join(", ");
        let top = |m: &BTreeMap<String, i64>| {
            let mut v: Vec<(&String, &i64)> = m.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            v.into_iter()
                .take(6)
                .map(|(k, v)| format!("{k} x{v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let paths = top(&self.desync_paths);
        let adopted = top(&self.adopted_paths);
        let mut out = format!(
            "mirror: {} reconciles, {} desyncs, {} plays + {} turns followed ({} not), {}/{} replays, {} searched selects",
            s("reconciles"),
            s("desyncs"),
            s("followed"),
            s("turns_followed"),
            s("unfollowable"),
            s("replays_landed"),
            s("replays_attempted"),
            s("searched_selects"),
        );
        if s("stale_reports_skipped") != 0 {
            out.push_str(&format!(", {} echoes skipped", s("stale_reports_skipped")));
        }
        for (label, text) in [
            ("declined", why),
            ("desynced", paths),
            ("teammate-adopted", adopted),
        ] {
            if !text.is_empty() {
                out.push_str(&format!(" | {label}: {text}"));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_fold_and_the_note_reads_like_the_python_one() {
        assert_eq!(fold_indices("$.encounter[Nibbit#0].hp"), "$.encounter[].hp");
        assert_eq!(
            fold_indices("$.deck.hand[2].cost[x"),
            "$.deck.hand[].cost[x"
        );
        let mut m = AdvisorMirror::new(true, false);
        m.decline("b");
        m.decline("a");
        m.decline("a");
        assert_eq!(
            m.note(),
            "mirror: 0 reconciles, 0 desyncs, 0 plays + 0 turns followed (0 not), 0/0 replays, 0 searched selects | declined: a x2, b x1"
        );
        assert_eq!(m.take_log().len(), 2, "each reason is announced once");
    }

    #[test]
    fn a_card_mid_play_blocks_the_end_turn_replay() {
        // The engine never comes to rest with a play suspended and no choice
        // open; the guard is for the case `answer_card_select`'s gate cannot
        // see, so the state is made by hand.
        let mut deck = vec!["BURNING_PACT".to_string()];
        deck.extend(std::iter::repeat_n("STRIKE_IRONCLAD".to_string(), 4));
        let state = sts2_wire::scenario::build_scenario_state(
            1,
            "Ironclad",
            &deck,
            70,
            80,
            &[],
            &[],
            "SHRINKER_BEETLE_WEAK",
            0,
            None,
            None,
        )
        .expect("builds");
        let wire = sts2_wire::emit::combat_state_to_value(&state).expect("emits");
        let mut m = AdvisorMirror::new(true, false);
        assert!(m.on_combat_board(&wire, &json!({"act": 1, "floor": 3})));
        let live = m
            .mirror_mut(On::Live)
            .and_then(CombatMirror::state_mut_for_test)
            .unwrap();
        let pact = sts2_wire::emit::legal_actions(live)
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["action"] == "play")
            .cloned()
            .unwrap();
        assert_eq!(apply_step(live, &pact), Ok("pending_choice"));
        assert!(live.pending_play.is_some());
        live.pending_choice = None;

        assert!(!m.replay_end_turn(On::Live, &json!({})));
        assert!(m
            .declines
            .contains_key("a card is mid-play, so the turn is not ending"));
        assert_eq!(m.stats()["replays_attempted"], 0);
    }

    /// GH #1918: a killing play reports `target_index: null`; the advice's
    /// target backfills it and the mirror keeps the fight.
    fn rats_mirror(deck: &[&str]) -> (AdvisorMirror, Value) {
        let deck: Vec<String> = deck.iter().map(|s| (*s).to_owned()).collect();
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
        let wire = sts2_wire::emit::combat_state_to_value(&cs).expect("emits");
        let mut m = AdvisorMirror::new(true, false);
        assert!(m.on_combat_board(&wire, &json!({"act": 1, "floor": 3})));
        (m, wire)
    }

    fn enemy_hps(m: &AdvisorMirror) -> Vec<i64> {
        let wire = sts2_wire::emit::combat_state_to_value(m.live_state().unwrap()).unwrap();
        wire["encounter"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["hp"].as_i64().unwrap())
            .collect()
    }

    fn play_msg(id: &str, target: Value) -> Value {
        json!({"decision_type": "combat_action", "action": "play",
               "card": {"id": id, "upgraded": false}, "target_index": target})
    }

    #[test]
    fn a_null_target_on_a_killing_play_is_backfilled_from_the_advice() {
        let (mut m, _) = rats_mirror(&["STRIKE_IRONCLAD"; 5]);
        let before = enemy_hps(&m);
        assert!(before.len() >= 2, "a multi-enemy board");
        m.set_advice_target(json!(1));
        m.on_world_action(&play_msg("STRIKE_IRONCLAD", Value::Null));
        assert!(m.live().is_some(), "the fight is kept");
        assert_eq!(m.stats()["drops"], 0);
        let after = enemy_hps(&m);
        assert!(after[1] < before[1], "the advised enemy was hit");
        assert_eq!(after[0], before[0]);
        assert_eq!(after[2], before[2]);
    }

    #[test]
    fn a_null_target_on_a_card_that_is_not_targeted_stays_null() {
        let (mut m, _) = rats_mirror(&["DEFEND_IRONCLAD"; 5]);
        m.set_advice_target(json!(1));
        let before = enemy_hps(&m);
        m.on_world_action(&play_msg("DEFEND_IRONCLAD", Value::Null));
        assert!(m.live().is_some(), "no spurious drop");
        assert_eq!(m.stats()["drops"], 0);
        assert_eq!(enemy_hps(&m), before, "no target was invented");
        assert_eq!(m.recover_card_target(On::Live, 0), Value::Null);
    }

    /// Gambler's Brew's hand in `wb-eb8f0e387af30b15` (GH #1770).
    fn brew_hand() -> Vec<Value> {
        [
            "Backstab",
            "Strike+",
            "Defend",
            "Poor Sleep",
            "Defend",
            "Strike",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let id = match *name {
                "Strike+" | "Strike" => "STRIKE_SILENT",
                "Defend" => "DEFEND_SILENT",
                other => other,
            };
            json!({"name": name, "id": id, "upgraded": *name == "Strike+", "index": i})
        })
        .collect()
    }

    fn brew_state(cards: &[Value], selected: i64) -> Value {
        json!({
            "cards": cards, "min_select": 0, "max_select": 999_999_999,
            "prompt_key": "GAMBLERS_BREW.selectionScreenPrompt",
            "source_model": "GAMBLERS_BREW", "selected_count": selected,
            "combat": {},
        })
    }

    /// GH #1770: a select of several cards names the whole set and its
    /// count; a single-card select keeps its note.
    #[test]
    fn a_multi_card_select_names_its_set_and_count() {
        let cards = brew_hand();
        let state = brew_state(&cards, 0);
        let labels = card_labels(&cards);
        let line = set_line(&state, &labels, &[2, 4]);
        assert_eq!(line.as_deref(), Some("Replace 2: Defend, Defend"));
        let got = picked_set_answer(&cards, &[2, 4], "Instinct · choice", line.as_deref());
        assert_eq!(
            got["note"],
            json!("Instinct · choice · Replace 2: Defend, Defend")
        );
        let best: Vec<bool> = got["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["is_best"].as_bool().unwrap())
            .collect();
        assert_eq!(best, [false, false, true, false, true, false]);
        assert_eq!(
            got["best"],
            json!({"action": "choose", "index": 2, "label": "Defend"})
        );
        assert_eq!(
            set_line(&state, &labels, &[]).as_deref(),
            Some("Replace none")
        );
        let single = json!({"max_select": 1, "prompt_key": "TO_EXHAUST"});
        assert_eq!(set_line(&single, &labels, &[2]), None);
    }

    /// GH #1770: pick 2 of a planned set keeps who planned it, says which
    /// pick it is, and keeps the rest of the set marked.
    #[test]
    fn a_later_pick_of_a_planned_set_keeps_the_set() {
        let cards = brew_hand();
        let mut m = AdvisorMirror::new(true, false);
        let picks = [2, 1, 3, 4, 5];
        let line = set_line(&brew_state(&cards, 0), &card_labels(&cards), &picks);
        m.remember_planned_picks(
            &json!("GAMBLERS_BREW"),
            &cards,
            &picks,
            "Instinct (e2e net, no search) · choice",
            line,
        );
        // The first Defend left the hand; four cards of the set remain.
        let mut after = cards.clone();
        after.remove(2);
        let cfg = RecommendConfig::default();
        let mut wire_for = |_: &Value, _: Option<&str>, _: i64, _: i64, _: u64| {
            Err::<Value, _>(("Unreachable".to_string(), "no wire".to_string()))
        };
        let got = m
            .answer_card_select(
                &brew_state(&after, 1),
                Some(&cfg),
                None,
                None,
                None,
                &mut wire_for,
            )
            .expect("a remembered pick");
        assert_eq!(
            got["note"],
            json!(
                "Instinct (e2e net, no search) · choice · pick 2 of 5 · \
                 Replace 5: Defend, Strike+, Poor Sleep, Defend, Strike"
            )
        );
        assert_eq!(
            got["best"],
            json!({"action": "choose", "index": 1, "label": "Strike+"})
        );
        let best: Vec<bool> = got["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["is_best"].as_bool().unwrap())
            .collect();
        assert_eq!(best, [false, true, true, true, true]);
    }

    #[test]
    fn a_remembered_plan_follows_the_shrinking_offer() {
        let mut m = AdvisorMirror::new(true, false);
        let offer = [json!({"id": "A"}), json!({"id": "B"}), json!({"id": "C"})];
        m.remember_picks(&json!("Src"), &offer, &[2, 0]);
        assert_eq!(m.next_remembered_pick(&Value::Null, &offer), Some(2));
        let after = [json!({"id": "A"}), json!({"id": "B"})];
        assert_eq!(m.next_remembered_pick(&Value::Null, &after), Some(0));
        assert_eq!(m.next_remembered_pick(&json!("Other"), &after), None);
        assert_eq!(
            m.next_remembered_pick(&Value::Null, &after),
            None,
            "forgotten"
        );
    }
}
