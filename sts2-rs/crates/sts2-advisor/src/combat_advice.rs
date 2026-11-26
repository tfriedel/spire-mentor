//! A fight's `advice` from a ranking (GH #1426; was split across
//! `sts2_hint/recommender.py`, `realgame_advisor.py` and
//! `realgame_instinct.combat_advice`): the search's rows (or Instinct's
//! classes) mapped to the mod's stable indices, the plays the live game
//! has already refused vetoed, and the wire `advice` built from what is
//! left (PROTOCOL.md section 5).
//!
//! Stable indices: a card play is the card's C# `index` (`>= 0`), then
//! [`END_TURN`], potion slot `s` is `POTION_BASE - s`, [`CHANNEL`] and
//! [`EVOKE`], and option `i` of a pending choice is `CHOOSE_BASE - i`
//! ([`CHOOSE_SKIP`] the empty-selection finalize).

use serde_json::{json, Map, Value};

use crate::board_targets::BoardTargets;
use crate::display_names::Names;
use crate::i18n::{is_zh, join_list, tr, tr_dyn, trf};
use crate::instinct::Pick;
use crate::py;
use crate::turn_plan::{ally_label, enemy_label};

pub const END_TURN: i64 = -1;
/// Potion slot `s` -> `POTION_BASE - s`.
pub const POTION_BASE: i64 = -100;
pub const CHANNEL: i64 = -200;
pub const EVOKE: i64 = -201;
/// Choose option `i` -> `CHOOSE_BASE - i`.
pub const CHOOSE_BASE: i64 = -300;
/// The empty-selection FINALIZE / SKIP of a pending choice.
pub const CHOOSE_SKIP: i64 = -299;

/// `sts2_hint.calibration`: `p_cal = sigmoid(INTERCEPT + SLOPE *
/// logit(p_raw))`, raw values clipped to `[EPS, 1 - EPS]` first (the search
/// reports exact 0 and 1, certainty it did not measure).
#[must_use]
pub fn calibrated_p_win(raw: f64) -> f64 {
    const INTERCEPT: f64 = 2.4748;
    const SLOPE: f64 = 0.9319;
    const EPS: f64 = 1e-3;
    let p = raw.clamp(EPS, 1.0 - EPS);
    let z = INTERCEPT + SLOPE * (p / (1.0 - p)).ln();
    1.0 / (1.0 + (-z).exp())
}

/// One action's hint summary (`HintTag`).
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub stable_idx: i64,
    pub win_prob: f64,
    pub bucket_label: String,
    /// The search's own pick (or, when a caller dropped that row, the
    /// engine's next choice among the survivors: [`crown_engine_heir`]).
    pub is_best: bool,
    pub visits: i64,
    pub q_saturated: bool,
    pub low_confidence: bool,
    pub target_index: Option<i64>,
    pub q: f64,
    pub wins: i64,
    pub losses: i64,
    pub ties: i64,
    pub ally_index: Option<i64>,
    /// The engine's preference order (GH #1022): 0 is its pick.
    pub pick_rank: Option<i64>,
    pub in_band: bool,
    /// Which input tag this is, so a caller can tell a surviving tag from a
    /// re-crowned or substituted one (Python compared objects by identity).
    pub origin: Option<usize>,
}

impl Tag {
    /// A tag with the defaults `HintTag` has.
    #[must_use]
    pub fn new(stable_idx: i64, win_prob: f64, bucket_label: &str, is_best: bool) -> Self {
        Self {
            stable_idx,
            win_prob,
            bucket_label: bucket_label.into(),
            is_best,
            visits: 0,
            q_saturated: false,
            low_confidence: false,
            target_index: None,
            q: 0.0,
            wins: 0,
            losses: 0,
            ties: 0,
            ally_index: None,
            pick_rank: None,
            in_band: false,
            origin: None,
        }
    }

    /// `HintTag._asdict()`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json!({
            "stable_idx": self.stable_idx,
            "win_prob": self.win_prob,
            "bucket_label": self.bucket_label,
            "is_best": self.is_best,
            "visits": self.visits,
            "q_saturated": self.q_saturated,
            "low_confidence": self.low_confidence,
            "target_index": self.target_index,
            "q": self.q,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "ally_index": self.ally_index,
            "pick_rank": self.pick_rank,
            "in_band": self.in_band,
        })
    }

    /// A `HintTag._asdict()` back as a tag (missing fields take defaults).
    #[must_use]
    pub fn from_value(v: &Value, origin: usize) -> Self {
        let opt = |k| {
            py::get(v, k)
                .as_i64()
                .or_else(|| py::get(v, k).as_f64().map(|f| f as i64))
        };
        let num = |k| py::get(v, k).as_f64().unwrap_or(0.0);
        let flag = |k| py::truthy(py::get(v, k));
        Self {
            stable_idx: py::int_of(py::get(v, "stable_idx")),
            win_prob: num("win_prob"),
            bucket_label: py::str_of(py::get(v, "bucket_label")),
            is_best: flag("is_best"),
            visits: py::int_of(py::get(v, "visits")),
            q_saturated: flag("q_saturated"),
            low_confidence: flag("low_confidence"),
            target_index: opt("target_index"),
            q: num("q"),
            wins: py::int_of(py::get(v, "wins")),
            losses: py::int_of(py::get(v, "losses")),
            ties: py::int_of(py::get(v, "ties")),
            ally_index: opt("ally_index"),
            pick_rank: opt("pick_rank"),
            in_band: flag("in_band"),
            origin: Some(origin),
        }
    }
}

/// Why an action has no stable index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unmapped {
    /// A play indexing past this world's hand: recoverable, the row is
    /// dropped (GH #360).
    Unmappable(String),
    /// A shape outside the wire contract: adapter drift, not recoverable.
    Unknown(String),
}

/// `int(v)`, or `None` where Python raises.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `int(x) if x is not None else None`.
fn opt_int(v: &Value) -> Option<i64> {
    if v.is_null() {
        None
    } else {
        py_int(v)
    }
}

/// `_stable_idx_for_action`: one engine legal-action dict as its stable
/// index.
///
/// # Errors
///
/// [`Unmapped`].
pub fn stable_idx_for_action(action: &Value, hand: &[Value]) -> Result<i64, Unmapped> {
    let Some(kind) = action.get("action") else {
        return Err(Unmapped::Unknown("'action'".into()));
    };
    let bad = || {
        Unmapped::Unknown(format!(
            "recommend(): unknown sts2-rs action kind {}",
            py::repr(kind)
        ))
    };
    match kind.as_str() {
        Some("play") => {
            let ci = py_int(py::get(action, "card_index")).ok_or_else(bad)?;
            let card = usize::try_from(ci)
                .ok()
                .and_then(|i| hand.get(i))
                .ok_or_else(|| {
                    Unmapped::Unmappable(format!(
                        "recommend(): play action card_index={ci} out of range for hand of size {}",
                        hand.len()
                    ))
                })?;
            py_int(py::get(card, "stable_idx")).ok_or_else(bad)
        }
        Some("end_turn") => Ok(END_TURN),
        Some("potion" | "drink_potion") => {
            Ok(POTION_BASE - py_int(py::get(action, "slot")).ok_or_else(bad)?)
        }
        Some("channel") => Ok(CHANNEL),
        Some("evoke") => Ok(EVOKE),
        Some("choose") => {
            // `selection` is authoritative: it tells the empty-selection
            // FINALIZE/SKIP from "choose option 0" (both have `index == 0`).
            match action.get("selection") {
                Some(Value::Null) | None => Ok(CHOOSE_BASE
                    - action
                        .get("index")
                        .map_or(Some(0), py_int)
                        .ok_or_else(bad)?),
                Some(sel) => match py::as_list(sel).first() {
                    None => Ok(CHOOSE_SKIP),
                    Some(first) => Ok(CHOOSE_BASE - py_int(first).ok_or_else(bad)?),
                },
            }
        }
        _ => Err(bad()),
    }
}

/// `_bucket_label`: the `WIN_PROB_BUCKETS` row whose lower bound `p_win`
/// reaches.
#[must_use]
pub fn bucket_label(p_win: f64) -> &'static str {
    let rows = sts2_mcts::hint::WIN_PROB_BUCKETS;
    rows.iter()
        .find(|(lo, _)| p_win >= f64::from(*lo))
        .or_else(|| rows.last())
        .map_or("unknown", |(_, l)| l)
}

/// `crown_engine_heir`: when no tag is the pick any more, crown the
/// engine's own next choice among the survivors (the lowest `pick_rank`;
/// the first tag when none carries one), heir first.
#[must_use]
pub fn crown_engine_heir(tags: Vec<Tag>) -> Vec<Tag> {
    if tags.is_empty() || tags.iter().any(|t| t.is_best) {
        return tags;
    }
    let heir_at = tags
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.pick_rank.map(|r| (r, i)))
        .min_by_key(|&(r, _)| r)
        .map_or(0, |(_, i)| i);
    let mut rest = tags;
    let mut heir = rest.remove(heir_at);
    heir.is_best = true;
    // A re-crowned tag is a new value, not the input one.
    heir.origin = None;
    std::iter::once(heir).chain(rest).collect()
}

/// `tags_from_result`: a raw `mcts_recommend` result as tags, the pick
/// first, then descending win probability. Returns the tags and the
/// unmappable rows dropped (never silently: the caller logs them).
///
/// # Errors
///
/// A row of a kind outside the wire contract ([`Unmapped::Unknown`]).
pub fn tags_from_result(
    result: &Value,
    hand: &[Value],
) -> Result<(Vec<Tag>, Vec<String>), Unmapped> {
    let mut out = Vec::new();
    let mut dropped = Vec::new();
    for entry in py::list(result, "actions") {
        let action = py::get(entry, "action");
        let p_win = py::get(entry, "p_win").as_f64().unwrap_or(0.0);
        let stable_idx = match stable_idx_for_action(action, hand) {
            Ok(s) => s,
            Err(Unmapped::Unmappable(m)) => {
                dropped.push(m);
                continue;
            }
            Err(e) => return Err(e),
        };
        let q = match entry.get("q") {
            Some(q) => q.as_f64().unwrap_or(0.0),
            None => entry.get("value").and_then(Value::as_f64).unwrap_or(0.0),
        };
        let int0 = |k: &str| entry.get(k).and_then(py_int).unwrap_or(0);
        let mut t = Tag::new(
            stable_idx,
            p_win,
            bucket_label(p_win),
            py::truthy(py::get(entry, "is_best")),
        );
        t.visits = int0("visits");
        t.q_saturated = py::truthy(py::get(entry, "q_saturated"));
        t.low_confidence = py::truthy(py::get(entry, "low_confidence"));
        t.target_index = opt_int(py::get(action, "target_index"));
        t.q = q;
        t.wins = int0("wins");
        t.losses = int0("losses");
        t.ties = int0("ties");
        t.ally_index = opt_int(py::get(action, "ally_index"));
        t.pick_rank = opt_int(py::get(entry, "pick_rank"));
        t.in_band = py::truthy(py::get(entry, "in_band"));
        t.origin = Some(out.len());
        out.push(t);
    }
    let mut out = crown_engine_heir(out);
    out.sort_by(|a, b| {
        (!a.is_best)
            .cmp(&!b.is_best)
            .then(b.win_prob.total_cmp(&a.win_prob))
    });
    Ok((out, dropped))
}

/// `choice_picks_from_tags`: option indices to answer a pending-choice
/// screen with, best-first. The FINALIZE/SKIP tag is a stop signal (honoured
/// once `min_select` picks are in), no index repeats, at most `max_select`
/// picks, padded with the lowest unused indices up to `min_select`.
#[must_use]
pub fn choice_picks_from_tags(
    tags: &[Tag],
    n_options: i64,
    min_select: i64,
    max_select: i64,
) -> Vec<i64> {
    let mut picks: Vec<i64> = Vec::new();
    let mut stop_when_legal = false;
    let len = |p: &Vec<i64>| p.len() as i64;
    for t in tags {
        if t.stable_idx == CHOOSE_SKIP {
            if len(&picks) >= min_select {
                break;
            }
            stop_when_legal = true;
            continue;
        }
        if t.stable_idx > CHOOSE_BASE {
            continue;
        }
        let idx = CHOOSE_BASE - t.stable_idx;
        if !(0..n_options).contains(&idx) || picks.contains(&idx) {
            continue;
        }
        picks.push(idx);
        if len(&picks) >= max_select || (stop_when_legal && len(&picks) >= min_select) {
            break;
        }
    }
    if len(&picks) < min_select {
        let extra: Vec<i64> = (0..n_options).filter(|i| !picks.contains(i)).collect();
        picks.extend(extra);
        picks.truncate(usize::try_from(min_select).unwrap_or(0));
    }
    picks
}

/// `next_choice_pick_from_tags`'s answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextPick {
    Option(i64),
    /// The search ranked FINALIZE above every remaining option.
    Finalize,
}

/// `next_choice_pick_from_tags`: the single best next pick of an open
/// choice, `None` when no choose row is usable.
#[must_use]
pub fn next_choice_pick_from_tags(
    tags: &[Tag],
    n_options: i64,
    taken: &[i64],
    allow_finalize: bool,
) -> Option<NextPick> {
    for t in tags {
        if t.stable_idx == CHOOSE_SKIP {
            if allow_finalize {
                return Some(NextPick::Finalize);
            }
            continue;
        }
        if t.stable_idx > CHOOSE_BASE {
            continue;
        }
        let idx = CHOOSE_BASE - t.stable_idx;
        if (0..n_options).contains(&idx) && !taken.contains(&idx) {
            return Some(NextPick::Option(idx));
        }
    }
    None
}

/// `str(d.get(a) or d.get(b) or fallback)`.
fn name_or(d: &Value, keys: &[&str], fallback: impl FnOnce() -> String) -> String {
    keys.iter()
        .map(|k| py::get(d, k))
        .find(|v| py::truthy(v))
        .map_or_else(fallback, py::str_of)
}

/// The belt's potions (`state.potions` without the empty slots).
fn held_potions(state: &Value) -> Vec<&Value> {
    py::list(state, "potions")
        .iter()
        .filter(|p| !py::truthy(py::get(p, "empty")))
        .collect()
}

fn is_potion(idx: i64) -> bool {
    POTION_BASE - 5 < idx && idx <= POTION_BASE
}

/// `" → <target>"` for a move at `ally` (a teammate) or `target` (an enemy).
fn target_suffix(state: &Value, ally: &Value, target: &Value) -> String {
    let named = if ally.is_null() {
        enemy_label(py::list(state, "enemies"), target)
    } else {
        ally_label(py::list(state, "allies"), ally)
    };
    named.map(|t| format!(" → {t}")).unwrap_or_default()
}

/// `_tag_label`: an option's label, card or potion name plus its target.
#[must_use]
pub fn tag_label(tag: &Tag, state: &Value) -> String {
    let i = tag.stable_idx;
    let label = if i == END_TURN {
        return tr("End turn").into();
    } else if i >= 0 {
        let hand = py::list(state, "hand");
        usize::try_from(i)
            .ok()
            .and_then(|k| hand.get(k))
            .map_or_else(
                || "?".into(),
                |c| name_or(c, &["name", "id"], || "?".into()),
            )
    } else if is_potion(i) {
        let dense = POTION_BASE - i;
        let held = held_potions(state);
        usize::try_from(dense)
            .ok()
            .and_then(|k| held.get(k))
            .map(|p| py::get(p, "name"))
            .filter(|n| py::truthy(n))
            .map_or_else(|| trf("potion {n}", &[("n", &dense)]), py::str_of)
    } else if i == CHOOSE_SKIP {
        return tr("skip").into();
    } else if i <= CHOOSE_BASE {
        // A pick out of a pending choice, named against the world's own
        // offered list when the screen carries one.
        let k = CHOOSE_BASE - i;
        let offered = py::list(state, "cards");
        if let Some(c) = usize::try_from(k)
            .ok()
            .and_then(|k| offered.get(k))
            .filter(|c| c.is_object())
        {
            let named = ["name", "id"]
                .iter()
                .map(|f| py::get(c, f))
                .find(|v| py::truthy(v));
            if let Some(n) = named {
                return py::str_of(n);
            }
        }
        return trf("choose option {k}", &[("k", &k)]);
    } else {
        return trf("action {i}", &[("i", &i)]);
    };
    let ally = tag.ally_index.map_or(Value::Null, Value::from);
    let target = tag.target_index.map_or(Value::Null, Value::from);
    label + &target_suffix(state, &ally, &target)
}

/// `_pv_labels`: the principal variation named without lying about the
/// deeper plies (only ply 1 indexes the hand the player sees).
#[must_use]
pub fn pv_labels(pv: &[Value], state: &Value) -> Vec<String> {
    pv.iter()
        .enumerate()
        .map(|(depth, action)| {
            if !action.is_object() {
                return py::str_of(action);
            }
            let kind = action.get("action").map_or_else(|| "?".into(), py::str_of);
            if kind == "end_turn" {
                return tr("end turn").into();
            }
            if depth > 0 {
                return action_words(&kind);
            }
            let hand = py::list(state, "hand");
            let card = py::get(action, "card_index")
                .as_i64()
                .filter(|_| kind == "play")
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| hand.get(i));
            let label = card.map_or_else(
                || action_words(&kind),
                |c| name_or(c, &["name", "id"], || "?".into()),
            );
            label
                + &target_suffix(
                    state,
                    py::get(action, "ally_index"),
                    py::get(action, "target_index"),
                )
        })
        .collect()
}

/// An engine action verb (`end_turn`, `use_potion`) in words.
pub(crate) fn action_words(kind: &str) -> String {
    if !is_zh() {
        return kind.replace('_', " ");
    }
    match kind {
        "play" => "打出",
        "potion" | "use_potion" | "drink_potion" => "药水",
        "choose" => "选择",
        "skip" => "跳过",
        "channel" => "生成",
        "evoke" => "激发",
        "end_turn" => "结束回合",
        _ => return kind.replace('_', " "),
    }
    .into()
}

/// `_best_to_action`: the executable `best` object in the mod's takeover
/// dialect (a potion's dense slot remapped to the belt slot).
#[must_use]
pub fn best_to_action(tag: &Tag, state: &Value) -> Option<Value> {
    let mut best = Map::new();
    let i = tag.stable_idx;
    if i == END_TURN {
        return Some(json!({"action": "end_turn"}));
    }
    if i >= 0 {
        best.insert("action".into(), json!("play"));
        best.insert("card_index".into(), json!(i));
    } else if is_potion(i) {
        let dense = POTION_BASE - i;
        let held = held_potions(state);
        let slot = usize::try_from(dense)
            .ok()
            .and_then(|k| held.get(k))
            .and_then(|p| p.get("slot"))
            .and_then(Value::as_i64)?;
        best.insert("action".into(), json!("potion"));
        best.insert("slot".into(), json!(slot));
    } else {
        return None;
    }
    if let Some(t) = tag.target_index {
        best.insert("target_index".into(), json!(t));
    }
    if let Some(a) = tag.ally_index {
        best.insert("ally_index".into(), json!(a));
    }
    Some(Value::Object(best))
}

/// `option_move`: what a combat option row does, in `best`'s dialect.
#[must_use]
pub fn option_move(tag: &Tag, state: &Value) -> Value {
    if let Some(m) = best_to_action(tag, state) {
        let mut out = json!({"action": m["action"]});
        if let Some(slot) = m.get("slot") {
            out["slot"] = slot.clone();
        }
        return out;
    }
    let i = tag.stable_idx;
    let action = match i {
        CHOOSE_SKIP => "skip",
        _ if i <= CHOOSE_BASE => "choose",
        _ if is_potion(i) => "potion", // a potion the belt no longer shows
        CHANNEL => "channel",
        EVOKE => "evoke",
        _ => "unknown",
    };
    json!({"action": action})
}

/// `_veto_text`: `hand[2]:EnergyCostTooHigh` as `Bash (energy cost too high)`.
#[must_use]
pub fn veto_text(reason: &str, state: &Value, names: &Names) -> String {
    let parsed = reason.strip_prefix("hand[").and_then(|r| {
        let (digits, rest) = r.split_once("]:")?;
        (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && !rest.contains('\n'))
            .then(|| (digits.parse::<usize>().ok(), rest))
    });
    let Some((i, why)) = parsed else {
        return reason.into();
    };
    let hand = py::list(state, "hand");
    let name = i.and_then(|k| hand.get(k)).map_or_else(
        || trf("card {n}", &[("n", &i.unwrap_or(0))]),
        |c| names.card_name(c),
    );
    if is_zh() {
        let reasons: Vec<&str> = why
            .split(',')
            .map(|r| match r.trim() {
                "HasUnplayableKeyword" => "不能被打出",
                "BlockedByHook" => "被效果阻止",
                "BlockedByCardLogic" => "条件未满足",
                "EnergyCostTooHigh" => "能量不足",
                "StarCostTooHigh" => "辉星不足",
                "NoLivingAllies" => "没有存活的队友",
                other => other,
            })
            .collect();
        return format!("{name}（{}）", reasons.join("、"));
    }
    let mut spaced = String::new();
    for (k, ch) in why.chars().enumerate() {
        if k > 0 && ch.is_ascii_uppercase() {
            spaced.push(' ');
        }
        spaced.push(ch);
    }
    format!("{name} ({})", spaced.to_lowercase())
}

/// `wire_unplayable`: hand indices the game has already declared
/// unplayable, and why (absent or malformed legality reads as playable).
#[must_use]
pub fn wire_unplayable(state: &Value) -> Vec<(i64, String)> {
    py::list(state, "hand")
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_object() && c.get("playable") == Some(&Value::Bool(false)))
        .map(|(i, c)| {
            let r = py::get(c, "unplayable_reason");
            (
                i as i64,
                if py::truthy(r) {
                    py::str_of(r)
                } else {
                    "unplayable".into()
                },
            )
        })
        .collect()
}

/// `veto_unplayable`: drop the ranked plays the live game has already said
/// it will refuse; the crown moves to the engine's next choice. Everything
/// vetoed ends the turn. Returns the kept tags and one reason per veto.
#[must_use]
pub fn veto_unplayable(tags: Vec<Tag>, state: &Value) -> (Vec<Tag>, Vec<String>) {
    let veto = wire_unplayable(state);
    let why = |t: &Tag| {
        if t.stable_idx >= 0 {
            veto.iter()
                .find(|(i, _)| *i == t.stable_idx)
                .map(|(_, r)| r)
        } else {
            None
        }
    };
    if veto.is_empty() {
        return (tags, Vec::new());
    }
    let reasons: Vec<String> = tags
        .iter()
        .filter_map(|t| why(t).map(|r| format!("hand[{}]:{r}", t.stable_idx)))
        .collect();
    if reasons.is_empty() {
        return (tags, Vec::new());
    }
    let kept: Vec<Tag> = tags.iter().filter(|t| why(t).is_none()).cloned().collect();
    if kept.is_empty() {
        // Ending the turn is what the mod's refusal escape hatch would reach
        // anyway, three wasted requests later: reach it now.
        let mut end = tags[0].clone();
        end.stable_idx = END_TURN;
        end.target_index = None;
        end.is_best = true;
        end.origin = None;
        return (vec![end], reasons);
    }
    (crown_engine_heir(kept), reasons)
}

/// `_round_opt`.
fn round_opt(v: &Value) -> Value {
    if v.is_null() {
        Value::Null
    } else {
        json!(py::round(v.as_f64().unwrap_or(0.0), 4))
    }
}

/// `build_advice`: the wire `advice` from ranked tags. `pick` (the search
/// result's explanation of its pick, GH #1022) rides along only while the
/// advice still recommends that pick.
#[must_use]
pub fn build_advice(
    tags: Vec<Tag>,
    state: &Value,
    sims: i64,
    pv: Option<&[String]>,
    pick: &Value,
    names: &Names,
) -> Value {
    if tags.is_empty() {
        return json!({"note": tr("no legal actions")});
    }
    // Number the input: `pick` rides along only while the advice recommends
    // the very tag the engine crowned.
    let tags: Vec<Tag> = tags
        .into_iter()
        .enumerate()
        .map(|(i, mut t)| {
            t.origin = Some(i);
            t
        })
        .collect();
    let engine_best = tags.iter().find(|t| t.is_best).and_then(|t| t.origin);
    let (tags, vetoed) = veto_unplayable(tags, state);
    let best = tags[0].clone();
    let total: i64 = tags.iter().map(|t| t.visits).sum();
    let total = if total == 0 { 1 } else { total };
    let options: Vec<Value> = tags
        .iter()
        .map(|t| {
            let mut o = Map::new();
            o.insert("stable_idx".into(), json!(t.stable_idx));
            o.insert(
                "card_index".into(),
                if t.stable_idx >= 0 {
                    json!(t.stable_idx)
                } else {
                    Value::Null
                },
            );
            o.insert("target_index".into(), json!(t.target_index));
            if let Some(a) = t.ally_index {
                o.insert("ally_index".into(), json!(a));
            }
            o.extend(py::as_obj(&option_move(t, state)).clone());
            o.insert("label".into(), json!(tag_label(t, state)));
            o.insert("p_win".into(), json!(py::round(t.win_prob, 4)));
            o.insert("q".into(), json!(py::round(t.q, 4)));
            o.insert("visits".into(), json!(t.visits));
            o.insert(
                "visit_share".into(),
                json!(py::round(t.visits as f64 / total as f64, 4)),
            );
            o.insert("wins".into(), json!(t.wins));
            o.insert("losses".into(), json!(t.losses));
            o.insert("ties".into(), json!(t.ties));
            o.insert("bucket".into(), json!(t.bucket_label));
            o.insert("is_best".into(), json!(t.is_best));
            o.insert("low_confidence".into(), json!(t.low_confidence));
            o.insert("in_band".into(), json!(t.in_band));
            if let Some(r) = t.pick_rank {
                o.insert("pick_rank".into(), json!(r));
            }
            Value::Object(o)
        })
        .collect();
    let mut note = trf("MCTS {sims} sims · beliefdraw", &[("sims", &sims)]);
    if !vetoed.is_empty() {
        // Loud on purpose: each is a live disagreement between the mirror
        // and the game.
        let named: Vec<String> = vetoed
            .iter()
            .take(3)
            .map(|v| veto_text(v, state, names))
            .collect();
        note.push_str(&trf(
            " · vetoed {n} the game calls unplayable ({cards})",
            &[("n", &vetoed.len()), ("cards", &join_list(&named, ", "))],
        ));
    }
    let mut advice = json!({
        "note": note,
        // The HUD's "est. win": the search's own number reads far too low.
        "p_win": py::round(calibrated_p_win(best.win_prob), 4),
        "p_win_raw": py::round(best.win_prob, 4),
        "low_confidence": best.low_confidence || best.q_saturated,
        "options": options,
    });
    if let Some(mut b) = best_to_action(&best, state) {
        b["label"] = json!(tag_label(&best, state));
        advice["best"] = b;
    } else {
        // A ranked search with nothing executable behind it: the mod would
        // wait for a `best` it never gets.
        let n = trf(
            " · no executable form for {label}",
            &[("label", &tag_label(&best, state))],
        );
        advice["note"] = json!(format!(
            "{}{n}",
            advice["note"].as_str().unwrap_or_default()
        ));
    }
    if let Some(pv) = pv.filter(|p| !p.is_empty()) {
        advice["pv"] = json!(pv);
    }
    if py::truthy(pick) && engine_best.is_some() && best.origin == engine_best {
        advice["pick"] = json!({
            "rule": py::get(pick, "rule"),
            "reason": match py::get(pick, "reason") {
                Value::String(s) => json!(tr_dyn(s)),
                other => other.clone(),
            },
            "gap_q": round_opt(py::get(pick, "gap_q")),
            "gap_z": round_opt(py::get(pick, "gap_z")),
            "gap_hp": round_opt(py::get(pick, "gap_hp")),
        });
    }
    advice
}

/// Instinct's fight `advice` (`realgame_instinct.combat_advice`), in the
/// search's option shape: each merged class one row named by its
/// representative, mapped through the same stable indices; a class the game
/// would refuse hands the crown to the student's next choice. `targets`
/// maps the engine fight's enemy slots onto the board's (`board_targets`):
/// a class aimed at a corpse the game removed has no mod form.
///
/// # Errors
///
/// The player-facing reason Instinct cannot answer here.
pub fn instinct_combat_advice(
    pick: &Pick,
    state: &Value,
    wire: &Value,
    targets: &BoardTargets,
) -> Result<Value, String> {
    let hand = py::list(py::get(wire, "deck"), "hand");
    let mut order: Vec<usize> = (0..pick.probs.len()).collect();
    order.sort_by(|&a, &b| pick.probs[b].total_cmp(&pick.probs[a]));
    let rank = |ci: usize| order.iter().position(|&c| c == ci).map(|r| r as i64);
    let mut tags: Vec<Tag> = Vec::new();
    for cls in py::list(&pick.decision, "classes") {
        let ci = usize::try_from(py::int_of(py::get(cls, "class"))).unwrap_or(usize::MAX);
        let Some(action) = py::as_list(py::get(cls, "actions"))
            .first()
            .filter(|a| a.is_object())
            .and_then(|a| targets.board_action(a))
        else {
            continue;
        };
        let action = &action;
        let Ok(stable_idx) = stable_idx_for_action(action, hand) else {
            continue; // a class with no mod form
        };
        let p = pick.probs.get(ci).copied().unwrap_or(0.0);
        let mut t = Tag::new(stable_idx, p, "", ci == pick.best);
        t.target_index = opt_int(py::get(action, "target_index"));
        t.ally_index = opt_int(py::get(action, "ally_index"));
        t.pick_rank = rank(ci);
        t.origin = Some(tags.len());
        tags.push(t);
    }
    if tags.is_empty() {
        return Err(tr("none of Instinct's moves map onto this board").into());
    }
    let (mut tags, vetoed) = veto_unplayable(tags, state);
    if tags.is_empty() {
        return Err(tr("the game refuses every move Instinct ranks").into());
    }
    tags.sort_by_key(|t| t.pick_rank.unwrap_or(1 << 30));
    let best_at = tags.iter().position(|t| t.is_best).unwrap_or(0);
    let best = tags[best_at].clone();
    let options: Vec<Value> = tags
        .iter()
        .enumerate()
        .map(|(k, t)| {
            let mut o = Map::new();
            o.insert("stable_idx".into(), json!(t.stable_idx));
            o.insert(
                "card_index".into(),
                if t.stable_idx >= 0 {
                    json!(t.stable_idx)
                } else {
                    Value::Null
                },
            );
            o.insert("target_index".into(), json!(t.target_index));
            if let Some(a) = t.ally_index {
                o.insert("ally_index".into(), json!(a));
            }
            o.extend(py::as_obj(&option_move(t, state)).clone());
            o.insert("label".into(), json!(tag_label(t, state)));
            o.insert("visit_share".into(), json!(py::round(t.win_prob, 4)));
            o.insert("prob".into(), json!(py::round(t.win_prob, 4)));
            o.insert("is_best".into(), json!(k == best_at));
            o.insert("pick_rank".into(), json!(t.pick_rank));
            Value::Object(o)
        })
        .collect();
    let mut advice =
        json!({"note": crate::instinct::note_for(pick, best.win_prob), "options": options});
    let mut b =
        best_to_action(&best, state).ok_or(tr("Instinct's pick has no executable form here"))?;
    b["label"] = json!(tag_label(&best, state));
    advice["best"] = b;
    if !vetoed.is_empty() {
        let n = trf(
            " · vetoed {n} the game calls unplayable",
            &[("n", &vetoed.len())],
        );
        advice["note"] = json!(format!(
            "{}{n}",
            advice["note"].as_str().unwrap_or_default()
        ));
    }
    Ok(advice)
}

/// A row's or an option's identity: `(stable_idx, target_index, ally_index)`.
fn row_key(d: &Value) -> [Value; 3] {
    ["stable_idx", "target_index", "ally_index"].map(|k| py::get(d, k).clone())
}

fn same_key(a: &[Value; 3], b: &[Value; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| py::eq(x, y))
}

/// `combat_rows`: the search's per-action regret rows
/// (`rows_from_mcts`, labelled against the search's own `headless` board),
/// crowned the way `advice` is, and the advice's options enriched in place.
///
/// * `is_best` moves to the advice's best option: the unplayable veto can
///   re-crown the advice, and rows still crowning the refused card would
///   grade the player against a move nobody could make.
/// * Each row takes the advice option's label (names the advisor resolves
///   against the wire board), and the options gain the rows'
///   `mean_final_hp` / `hp_mean` / `hp_sd` / `hp_n`.
#[must_use]
pub fn combat_rows(result: &Value, headless: &Value, advice: &mut Value) -> Vec<Value> {
    let mut rows = sts2_workbench::advisors::rows_from_mcts(result, headless);
    // The first row per key wins.
    let find =
        |rows: &[Value], key: &[Value; 3]| rows.iter().position(|r| same_key(&row_key(r), key));
    let mut best_key: Option<[Value; 3]> = None;
    let mut best_idx: Option<Value> = None;
    if let Some(options) = advice.get_mut("options").and_then(Value::as_array_mut) {
        for o in options.iter_mut().filter(|o| o.is_object()) {
            let key = row_key(o);
            if best_key.is_none() && py::truthy(py::get(o, "is_best")) {
                best_key = Some(key.clone());
                best_idx = Some(py::get(o, "stable_idx").clone());
            }
            let Some(i) = find(&rows, &key) else { continue };
            let label = py::get(o, "label").clone();
            if py::truthy(&label) {
                rows[i]["action"] = label;
            }
            let row = &rows[i];
            let mean = py::get(row, "mean_final_hp").clone();
            o["mean_final_hp"] = mean.clone();
            // The HUD's name for the same number (PROTOCOL.md §5): expected
            // HP at the END OF THE FIGHT through this move.
            if !mean.is_null() {
                o["hp_mean"] = json!(py::round(mean.as_f64().unwrap_or(0.0), 1));
            }
            o["hp_sd"] = py::get(row, "hp_sd").clone();
            o["hp_n"] = row.get("hp_n").cloned().unwrap_or(json!(0));
        }
    }
    if let Some(key) = best_key {
        let crown = find(&rows, &key).or_else(|| {
            let idx = best_idx.unwrap_or(Value::Null);
            rows.iter()
                .position(|r| py::eq(py::get(r, "stable_idx"), &idx))
        });
        if let Some(c) = crown {
            for (i, r) in rows.iter_mut().enumerate() {
                r["is_best"] = json!(i == c);
            }
            let win = |r: &Value| py::get(r, "win_prob").as_f64().unwrap_or(0.0);
            rows.sort_by(|a, b| {
                let ab = a["is_best"] == json!(true);
                let bb = b["is_best"] == json!(true);
                bb.cmp(&ab).then(
                    win(b)
                        .partial_cmp(&win(a))
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_indices_follow_the_sentinels() {
        let hand = [json!({"stable_idx": 4})];
        let s = |a: Value| stable_idx_for_action(&a, &hand);
        assert_eq!(s(json!({"action": "play", "card_index": 0})), Ok(4));
        assert!(matches!(
            s(json!({"action": "play", "card_index": 1})),
            Err(Unmapped::Unmappable(_))
        ));
        assert_eq!(s(json!({"action": "drink_potion", "slot": 2})), Ok(-102));
        assert_eq!(
            s(json!({"action": "choose", "selection": []})),
            Ok(CHOOSE_SKIP)
        );
        assert_eq!(s(json!({"action": "choose", "selection": [3]})), Ok(-303));
        assert_eq!(s(json!({"action": "choose", "index": 1})), Ok(-301));
        assert!(matches!(
            s(json!({"action": "warp"})),
            Err(Unmapped::Unknown(_))
        ));
    }

    #[test]
    fn the_heir_is_the_engines_next_choice() {
        let mut a = Tag::new(1, 0.9, "", false);
        a.pick_rank = Some(2);
        let mut b = Tag::new(2, 0.1, "", false);
        b.pick_rank = Some(1);
        let out = crown_engine_heir(vec![a, b]);
        assert_eq!((out[0].stable_idx, out[0].is_best), (2, true));
    }

    #[test]
    fn choice_picks_stop_on_a_legal_finalize() {
        let t = |i| Tag::new(i, 0.5, "", false);
        let tags = [t(-300), t(CHOOSE_SKIP), t(-301)];
        assert_eq!(choice_picks_from_tags(&tags, 3, 1, 3), vec![0]);
        assert_eq!(choice_picks_from_tags(&tags, 3, 2, 3), vec![0, 1]);
        assert_eq!(
            next_choice_pick_from_tags(&tags, 3, &[0], true),
            Some(NextPick::Finalize)
        );
    }

    #[test]
    fn calibration_lifts_the_search_number() {
        assert!(calibrated_p_win(0.5) > 0.9);
        assert!(calibrated_p_win(0.0) > 0.0);
    }
}
