//! An in-combat `card_select` answered off the fight's mirror (GH #1426; was
//! `combat_mirror.recognize_card_select` / `skip_arity_from_world` and
//! `realgame_mcts.pick_card_select_mirrored`).
//!
//! The why of each rule stays on the Python wrappers, whose docstrings the
//! drivers' readers know: the recognition of the screens the engine never
//! opened (generative offers, KnowledgeDemon's curse pick, Gambling Chip's
//! mulligan, the turn-start powers), and the skip arity of
//! `FromChooseACardScreen`. Values are Python's: see [`crate::py`].

use serde_json::{json, Value};

use sts2_core::CombatState;

use crate::board::BoardError;
use crate::combat_mirror::{CombatMirror, MirrorError};
use crate::headless::HeadlessError;
use crate::py::{self, get, list, truthy};
use crate::search::RecommendConfig;

/// What the world's turn-start hook batch is blocked inside: a `"relic"` or
/// a `"power"`, and its id (`GAMBLING_CHIP`, `TYRANNY`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnStartBlock {
    pub source: &'static str,
    pub id: String,
}

/// `_GENERATIVE_SOURCES`: `source_model` → the engine's generative kind.
pub const GENERATIVE_SOURCES: [(&str, &str); 8] = [
    ("Toolbox", "add_generated_to_hand"),
    ("Quasar", "add_generated_to_hand"),
    ("Discovery", "add_generated_free_to_hand"),
    ("Splash", "add_generated_free_to_hand"),
    ("AttackPotion", "add_generated_free_to_hand"),
    ("SkillPotion", "add_generated_free_to_hand"),
    ("PowerPotion", "add_generated_free_to_hand"),
    ("ColorlessPotion", "add_generated_free_to_hand"),
];

/// `_TURN_START_POWER_SOURCES`: `source_model` → (kind, power id).
const TURN_START_POWER_SOURCES: [(&str, (&str, &str)); 2] = [
    ("TyrannyPower", ("exhaust_from_hand", "TYRANNY")),
    (
        "ToolsOfTheTradePower",
        ("discard_from_hand", "TOOLS_OF_THE_TRADE"),
    ),
];

/// Relics whose `AfterPlayerTurnStart` awaits a hand select on any turn:
/// `source_model` (the model's `Id.Entry`; a class name matches too) → (kind,
/// relic id). v0.111.0 Toasty Mittens exhausts one card each turn
/// (ToastyMittens.cs, GH #1696).
const TURN_START_RELIC_SOURCES: [(&str, (&str, &str)); 1] =
    [("TOASTY_MITTENS", ("exhaust_from_hand", "TOASTY_MITTENS"))];

/// `TOASTY_MITTENS` and `ToastyMittens` alike.
fn source_key(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_uppercase()
}

fn lookup_relic_source(src: &str) -> Option<(&'static str, &'static str)> {
    let key = source_key(src);
    TURN_START_RELIC_SOURCES
        .iter()
        .find(|(k, _)| source_key(k) == key)
        .map(|(_, v)| *v)
}

/// A [`GENERATIVE_SOURCES`] kind by source name, as the world names it
/// (`Splash`) or as the registry does (`SPLASH`, `CARD.SPLASH`).
fn lookup_generative(src: &str) -> Option<&'static str> {
    let src = src.strip_prefix("CARD.").unwrap_or(src);
    if src.is_empty() {
        return None;
    }
    let key = source_key(src);
    GENERATIVE_SOURCES
        .iter()
        .find(|(k, _)| source_key(k) == key)
        .map(|(_, v)| *v)
}

/// A screen the world named no source for, opened by the generative card
/// in play: an autoplay (Imbued's turn-1 Splash, seed 180001961) captures no
/// `source_model`. Without it the screen's shape read as Toolbox's offer
/// (round 1, Toolbox owned): `add_generated_to_hand` instead of Splash's
/// free card.
fn generative_in_play(src: &str, wire: &Value) -> Option<&'static str> {
    if !src.is_empty() {
        return None;
    }
    list(get(wire, "deck"), "play_pile")
        .iter()
        .rev()
        .find_map(|c| lookup_generative(&py::str_of(get(c, "id"))))
}

fn lookup<T: Copy>(table: &[(&str, T)], key: &str) -> Option<T> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// `skip_arity_from_world`: `(can_skip, min_select)`, both normalized.
#[must_use]
pub fn skip_arity_from_world(
    cancelable: &Value,
    source: &Value,
    min_select: i64,
    max_select: i64,
) -> (bool, i64) {
    if source != "FromChooseACardScreen" || cancelable.is_null() {
        return (min_select <= 0, min_select);
    }
    let can_skip = truthy(cancelable);
    if !can_skip && min_select <= 0 && max_select >= 1 {
        return (can_skip, 1);
    }
    (can_skip, min_select)
}

/// `str(v or "")`.
fn str_or_empty(v: &Value) -> String {
    if truthy(v) {
        py::str_of(v)
    } else {
        String::new()
    }
}

/// `str(a.get(k1) or a.get(k2) or "")`.
fn str_of_either(d: &Value, k1: &str, k2: &str) -> String {
    let a = get(d, k1);
    if truthy(a) {
        return py::str_of(a);
    }
    str_or_empty(get(d, k2))
}

/// The registry id of a headless card, as `adapter.registry_card_id`.
fn registry_id(c: &Value, path: &str) -> Result<String, String> {
    sts2_workbench::combat_import::registry_card_id(&py::str_of(get(c, "id")), path)
        .map_err(|e| e.to_string())
}

/// The board's hand as `(registry id, upgraded)`.
fn hand_ids(board: &Value) -> Result<Vec<(String, bool)>, String> {
    list(board, "hand")
        .iter()
        .map(|c| Ok((registry_id(c, "$.hand")?, truthy(get(c, "upgraded")))))
        .collect()
}

/// An option's `(id, upgraded)`.
fn option_key(o: &Value) -> (String, bool) {
    (py::str_of(get(o, "id")), truthy(get(o, "upgraded")))
}

/// `_locate_options_in_hand`: stamp each option with a distinct hand slot,
/// in hand order; `Ok(false)` when one finds none (the stamping so far
/// stays).
///
/// # Errors
///
/// A hand card the registry does not know.
fn locate_options_in_hand(options: &mut [Value], board: &Value) -> Result<bool, String> {
    let hand = hand_ids(board)?;
    let mut used = vec![false; hand.len()];
    let mut ok = true;
    for o in options.iter_mut() {
        let want = option_key(o);
        match (0..hand.len()).find(|&i| !used[i] && hand[i] == want) {
            Some(i) => {
                o["pile"] = json!("hand");
                o["index"] = json!(i);
                used[i] = true;
            }
            None => ok = false,
        }
    }
    Ok(ok)
}

/// `(board.player or {}).get("relics", [])`, or `None` where Python's loop
/// over it would raise.
fn board_relics(board: &Value) -> Option<&[Value]> {
    let player = get(board, "player");
    if !truthy(player) {
        return Some(&[]);
    }
    match player.get("relics") {
        None => Some(&[]),
        Some(Value::Array(a)) => Some(a),
        Some(_) => None,
    }
}

/// `recognize_card_select`: the kind to inject and the turn-start block to
/// arm behind it. Fills the options' hand locators where it recognizes a
/// hand screen.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn recognize_card_select(
    parked: Option<&Value>,
    options: &mut [Value],
    board: &Value,
    wire: &Value,
    min_select: i64,
    source_model: &Value,
    prompt_key: &Value,
    source: &Value,
) -> (String, Option<TurnStartBlock>) {
    let n = options.len();
    let mut kind = match parked {
        Some(p) if truthy(p) => py::str_of(get(p, "kind")),
        _ => "card_select".to_string(),
    };
    let mut block = None;
    if parked.is_none() {
        let src = str_or_empty(source_model);
        if let Some(k) = lookup_generative(&src).or_else(|| generative_in_play(&src, wire)) {
            kind = k.to_string();
        } else if let Some((power_kind, power_id)) = lookup(&TURN_START_POWER_SOURCES, &src) {
            if locate_options_in_hand(options, board).unwrap_or(false) {
                kind = power_kind.to_string();
                block = Some(TurnStartBlock {
                    source: "power",
                    id: power_id.to_string(),
                });
            }
        } else if let Some((relic_kind, relic_id)) = lookup_relic_source(&src) {
            if locate_options_in_hand(options, board).unwrap_or(false) {
                kind = relic_kind.to_string();
                block = Some(TurnStartBlock {
                    source: "relic",
                    id: relic_id.to_string(),
                });
            }
        } else if n == 2
            && py::str_of(get(&options[0], "id")) == "DISINTEGRATION"
            && list(wire, "encounter")
                .iter()
                .any(|e| str_of_either(e, "id", "kind") == "KnowledgeDemon")
        {
            kind = "monster_curse_select".to_string();
        } else if str_or_empty(prompt_key)
            .eq_ignore_ascii_case("CHOICES_PARADOX.selectionScreenPrompt")
        {
            // Choices Paradox's turn-1 offer: five generated Retain cards,
            // one may match a deck card (`listener_choices_paradox_turn_start`,
            // GH #1756).
            kind = "add_generated_retained_to_hand".to_string();
        } else if let Some(generated) = generative_by_shape(
            options,
            board,
            wire,
            py::str_of(source) == "FromChooseACardScreen",
        ) {
            kind = generated.to_string();
        }
    }
    if parked.is_none() && kind == "card_select" {
        if let Some((k, b)) = gambling_chip(options, board, min_select, prompt_key) {
            kind = k.to_string();
            block = b;
        }
    }
    (kind, block)
}

/// A screen whose options are generated: in no pile, or offered on a
/// choose-a-card screen, which C# opens only over generated cards (Toolbox,
/// Discovery, the colour potions, Splash, ...). A generated option can match
/// a card the player holds: Toolbox offering a Scrawl to a deck with one
/// (GH #1746). Paid for only from Toolbox's combat-start offer.
fn generative_by_shape(
    options: &[Value],
    board: &Value,
    wire: &Value,
    choose_a_card_screen: bool,
) -> Option<&'static str> {
    let deck = get(wire, "deck");
    let in_any_pile: Vec<(String, bool)> = ["hand", "draw_pile", "discard_pile", "exhaust_pile"]
        .iter()
        .flat_map(|k| list(deck, k))
        .map(option_key)
        .collect();
    let in_a_pile = options.iter().any(|o| in_any_pile.contains(&option_key(o)));
    if options.is_empty() || (in_a_pile && !choose_a_card_screen) {
        return None;
    }
    let relics = board_relics(board)?;
    let toolbox = relics.iter().any(|r| {
        str_of_either(r, "id", "name")
            .to_uppercase()
            .contains("TOOLBOX")
    });
    let round_one = py::int_of(get(board, "round")) <= 1;
    Some(if round_one && toolbox {
        "add_generated_to_hand"
    } else {
        "add_generated_free_to_hand"
    })
}

/// Gambling Chip's turn-1 mulligan: round 1, skippable, the options exactly
/// the whole hand.
fn gambling_chip(
    options: &mut [Value],
    board: &Value,
    min_select: i64,
    prompt_key: &Value,
) -> Option<(&'static str, Option<TurnStartBlock>)> {
    let mut hand = hand_ids(board).ok()?;
    let mut opts: Vec<(String, bool)> = options.iter().map(option_key).collect();
    opts.sort();
    let mut relic_ids: Vec<String> = board_relics(board)?
        .iter()
        .map(|r| str_of_either(r, "id", "name"))
        .collect();
    relic_ids.sort();
    relic_ids.dedup();
    let is_gambling = |r: &String| r.to_uppercase().contains("GAMBLING");
    let pk = str_or_empty(prompt_key);
    hand.sort();
    // The whole hand; or, on the relic's own prompt, part of it: a card a
    // later turn-start hook adds while the screen is open (Crossbow's free
    // Attack, `AfterSideTurnStart`) is in hand but not on offer (GH #1758).
    let offered = if pk.to_uppercase().contains("GAMBLING_CHIP") {
        let mut rest = hand.clone();
        opts.iter().all(|o| {
            rest.iter()
                .position(|h| h == o)
                .map(|i| rest.remove(i))
                .is_some()
        })
    } else {
        opts == hand
    };
    let shape = py::int_of(get(board, "round")) <= 1
        && min_select == 0
        && offered
        && relic_ids.iter().any(is_gambling);
    if !shape {
        return None;
    }
    let block = relic_ids
        .iter()
        .find(|r| is_gambling(r))
        .filter(|_| pk.is_empty() || pk.to_uppercase().contains("GAMBLING_CHIP"))
        .map(|r| TurnStartBlock {
            source: "relic",
            id: r.clone(),
        });
    // The locators are best-effort: a miss keeps the kind, as Python's did.
    let _ = locate_options_in_hand(options, board);
    Some(("discard_from_hand_and_draw", block))
}

/// The selection's arity once the mirror is on the world's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arity {
    pub n_options: i64,
    pub min_select: i64,
    pub max_select: i64,
}

/// `state.get(a, state.get(b)) or default`, as an int.
fn count_of(state: &Value, a: &str, b: &str, default: i64) -> i64 {
    let v = state.get(a).unwrap_or_else(|| get(state, b));
    if truthy(v) {
        py::int_of(v)
    } else {
        default
    }
}

/// The first half of `pick_card_select_mirrored`: put the mirror on the
/// world's choice. Reconcile against `wire` (the screen's own board, the
/// `headless` form beside it), recognize the screen, inject the world's
/// options and arm a blocked turn start behind them.
///
/// # Errors
///
/// A card the registry does not know ([`MirrorError::Invariant`]); under
/// STRICT, a divergence.
pub fn prepare_card_select(
    mirror: &mut CombatMirror,
    state: &Value,
    wire: &Value,
    headless: &Value,
) -> Result<Arity, MirrorError> {
    mirror.reconcile(wire)?;
    mirror.adopt_unseen_play(wire)?;
    let parked = mirror.parked_choice().cloned();
    let mut options = list(state, "cards")
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = sts2_workbench::combat_import::registry_card_id(
                &py::str_of(get(c, "id")),
                &format!("$.cards[{i}]"),
            )
            .map_err(|e| MirrorError::Invariant(e.to_string()))?;
            Ok(json!({
                "id": id,
                "upgraded": truthy(get(c, "upgraded")) || truthy(get(c, "upgrade_level")),
            }))
        })
        .collect::<Result<Vec<Value>, MirrorError>>()?;
    let n = i64::try_from(options.len()).unwrap_or(i64::MAX);
    let raw_min = count_of(state, "min_count", "min_select", 0);
    let raw_max = count_of(state, "max_count", "max_select", 1);
    let min_select = raw_min.min(n).max(0);
    let max_select = raw_max.min(n).max(min_select);
    let (can_skip, min_select) = skip_arity_from_world(
        get(state, "cancelable"),
        get(state, "source"),
        min_select,
        max_select,
    );
    let (kind, block) = recognize_card_select(
        parked.as_ref(),
        &mut options,
        headless,
        wire,
        min_select,
        get(state, "source_model"),
        get(state, "prompt_key"),
        get(state, "source"),
    );
    let origin = json!({
        "prompt_key": get(state, "prompt_key"),
        "source": get(state, "source"),
        "source_model": get(state, "source_model"),
    });
    mirror.inject_choice(&kind, options, min_select, max_select, can_skip, origin)?;
    if let Some(b) = block {
        if b.source == "power" {
            mirror.arm_turn_start_resume_after_power(&b.id)?;
        } else {
            mirror.arm_turn_start_resume(&b.id)?;
        }
    }
    Ok(Arity {
        n_options: n,
        min_select,
        max_select,
    })
}

/// The bridge's own handle for option `i` (`cards[i]["index"]`, else `i`).
fn bridge_index(cards: &[Value], i: i64) -> i64 {
    usize::try_from(i)
        .ok()
        .and_then(|u| cards.get(u))
        .and_then(|c| c.get("index"))
        .map_or(i, py::int_of)
}

/// The second half of `pick_card_select_mirrored`: the bridge action for
/// the picks, and the picks to advance the mirror by once the world takes
/// them. `picked` is `None` when the mirror dropped on the way to the choice:
/// the minimum legal selection then, and no picks (nothing to advance).
#[must_use]
pub fn card_select_answer(
    cards: &[Value],
    min_select: i64,
    picked: Option<Vec<i64>>,
) -> (Value, Option<Vec<i64>>) {
    let indices = |p: &[i64]| -> Value { p.iter().map(|&i| bridge_index(cards, i)).collect() };
    let Some(mut picked) = picked else {
        let picked: Vec<i64> = (0..min_select).collect();
        if picked.is_empty() {
            return (json!({"action": "skip"}), None);
        }
        return (
            json!({"action": "choose", "indices": indices(&picked)}),
            None,
        );
    };
    if picked.is_empty() && min_select <= 0 {
        return (json!({"action": "skip"}), Some(Vec::new()));
    }
    if picked.is_empty() {
        picked = vec![0];
    }
    (
        json!({"action": "choose", "indices": indices(&picked)}),
        Some(picked),
    )
}

/// Who picks once the mirror is on the world's choice instead of the combat
/// MCTS: the mirror's state and the arity → the option positions, or the
/// picker's exception as `(Python class name, message)`.
pub type StatePicker<'a> =
    dyn FnMut(&CombatState, Arity) -> Result<Vec<i64>, (String, String)> + 'a;

/// Why [`pick_card_select_for`] could not answer.
#[derive(Debug)]
pub enum PickError {
    /// The screen's board has no headless form.
    Headless(HeadlessError),
    /// The screen's board has no engine fight.
    Board(BoardError),
    /// The mirror refused (a divergence under STRICT, an unknown card).
    Mirror(MirrorError),
    /// The picker raised: `(Python class name, message)`.
    Picker(String, String),
}

impl PickError {
    /// The Python exception's class name.
    #[must_use]
    pub fn py_type(&self) -> String {
        match self {
            Self::Headless(HeadlessError::NotCombat(_)) => "ValueError".into(),
            Self::Headless(HeadlessError::GameVersion(_)) => "UnsupportedGameVersionError".into(),
            Self::Board(e) => e.py_type().into(),
            Self::Mirror(e) => mirror_error_type(e).into(),
            Self::Picker(name, _) => name.clone(),
        }
    }

    /// The Python exception's message.
    #[must_use]
    pub fn py_message(&self) -> String {
        match self {
            Self::Headless(e) => e.to_string(),
            Self::Board(e) => e.to_string(),
            Self::Mirror(e @ (MirrorError::Diverged(_) | MirrorError::NoState)) => {
                format!("[mirror advisor-answer] {e}")
            }
            Self::Mirror(e) => e.to_string(),
            Self::Picker(_, msg) => msg.clone(),
        }
    }
}

/// The Python class name of a failed mirror call.
#[must_use]
pub fn mirror_error_type(e: &MirrorError) -> &'static str {
    match e {
        MirrorError::Diverged(_) | MirrorError::NoState => "MirrorDivergence",
        MirrorError::Wire(w) if w.is_unsupported() => "UnsupportedStateError",
        MirrorError::Invariant(_) => "UnsupportedStateError",
        MirrorError::Wire(_) | MirrorError::Value(_) => "ValueError",
        MirrorError::Key(_) => "KeyError",
    }
}

/// `realgame_mcts.pick_card_select_mirrored`: answer a MID-COMBAT
/// `card_select` off the fight's mirror. The screen's embedded board is
/// rebuilt at the run's character and ascension (`cfg` otherwise),
/// `sims` overrides the budget, and `picker` (`None`: the combat MCTS)
/// picks once the mirror is on the world's choice. The board import's
/// warnings are appended to `warnings`.
///
/// # Errors
///
/// A board with no fight, the mirror's refusal, the picker's exception.
pub fn pick_card_select_for(
    mirror: &mut CombatMirror,
    state: &Value,
    cfg: &RecommendConfig,
    sims: Option<u32>,
    picker: Option<&mut StatePicker<'_>>,
    warnings: &mut Vec<String>,
) -> Result<(Value, Option<Vec<i64>>), PickError> {
    let run_state = get(state, "run_state");
    let mut call_cfg = cfg.clone();
    if let Some(name) = crate::headless::character_display_name(get(run_state, "character")) {
        call_cfg.character = Some(name.to_string());
    }
    call_cfg.ascension = crate::board::run_ascension(run_state, cfg.ascension);
    if let Some(sims) = sims {
        call_cfg.sims = sims;
    }
    let combat = get(state, "combat");
    let headless = crate::headless::to_headless_combat_play(combat).map_err(PickError::Headless)?;
    let (wire, w) = crate::board::combat_wire(
        combat,
        call_cfg.character.as_deref(),
        call_cfg.ascension,
        call_cfg.combat_id,
        call_cfg.seed,
    )
    .map_err(PickError::Board)?;
    warnings.extend(w);
    let arity = prepare_card_select(mirror, state, &wire, &headless).map_err(PickError::Mirror)?;
    let cards = list(state, "cards");
    if !mirror.alive() {
        return Ok(card_select_answer(cards, arity.min_select, None));
    }
    let picked = match picker {
        Some(p) => {
            let state = &mirror
                .mirrored()
                .ok_or(PickError::Mirror(MirrorError::NoState))?
                .state;
            p(state, arity).map_err(|(name, msg)| PickError::Picker(name, msg))?
        }
        None => mirror
            .search_choice_picks(
                &call_cfg,
                list(get(&wire, "deck"), "hand"),
                arity.n_options,
                arity.min_select,
                arity.max_select,
            )
            .map_err(PickError::Mirror)?,
    };
    Ok(card_select_answer(cards, arity.min_select, Some(picked)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mandatory_choose_a_card_screen_takes_one() {
        let src = json!("FromChooseACardScreen");
        assert_eq!(skip_arity_from_world(&json!(false), &src, 0, 1), (false, 1));
        assert_eq!(skip_arity_from_world(&json!(true), &src, 0, 1), (true, 0));
        assert_eq!(skip_arity_from_world(&Value::Null, &src, 0, 1), (true, 0));
        assert_eq!(
            skip_arity_from_world(&json!(false), &json!("FromHand"), 1, 1),
            (false, 1)
        );
    }

    #[test]
    fn named_and_shaped_screens_are_recognized() {
        let wire = json!({"deck": {"hand": [{"id": "STRIKE_IRONCLAD"}]}, "encounter": []});
        let board = json!({"round": 1, "hand": [], "player": {"relics": [{"id": "TOOLBOX"}]}});
        let mut opts = vec![json!({"id": "FINESSE", "upgraded": false})];
        let named = recognize_card_select(
            None,
            &mut opts,
            &board,
            &wire,
            0,
            &json!("Splash"),
            &Value::Null,
            &Value::Null,
        );
        assert_eq!(named, ("add_generated_free_to_hand".into(), None));
        let shaped = recognize_card_select(
            None,
            &mut opts,
            &board,
            &wire,
            0,
            &Value::Null,
            &Value::Null,
            &Value::Null,
        );
        assert_eq!(shaped.0, "add_generated_to_hand");
        let parked = json!({"kind": "put_back"});
        let kept = recognize_card_select(
            Some(&parked),
            &mut opts,
            &board,
            &wire,
            0,
            &json!("Splash"),
            &Value::Null,
            &Value::Null,
        );
        assert_eq!(kept.0, "put_back");
    }

    /// Seed 180001961: Imbued autoplays Splash on turn 1 of a Toolbox
    /// fight. The world names no source; the card in play is Splash, whose
    /// offer is free, not Toolbox's.
    #[test]
    fn an_unnamed_screen_takes_the_generative_card_in_play() {
        let board = json!({"round": 1, "hand": [], "player": {"relics": [{"id": "TOOLBOX"}]}});
        let screen = json!("FromChooseACardScreen");
        let mut opts = vec![
            json!({"id": "POKE", "upgraded": true}),
            json!({"id": "HEGEMONY", "upgraded": true}),
        ];
        let splash = json!({"deck": {"play_pile": [{"id": "SPLASH"}]}, "encounter": []});
        let recognized = recognize_card_select(
            None,
            &mut opts,
            &board,
            &splash,
            0,
            &Value::Null,
            &Value::Null,
            &screen,
        );
        assert_eq!(recognized.0, "add_generated_free_to_hand");
        // Toolbox's own combat-start offer: nothing in play.
        let toolbox = json!({"deck": {}, "encounter": []});
        let shaped = recognize_card_select(
            None,
            &mut opts,
            &board,
            &toolbox,
            0,
            &Value::Null,
            &Value::Null,
            &screen,
        );
        assert_eq!(shaped.0, "add_generated_to_hand");
        // A named source wins over the card in play.
        let named = recognize_card_select(
            None,
            &mut opts,
            &board,
            &splash,
            0,
            &json!("Toolbox"),
            &Value::Null,
            &screen,
        );
        assert_eq!(named.0, "add_generated_to_hand");
    }

    /// GH #1746: Toolbox offered a Scrawl to a deck holding one. The
    /// choose-a-card screen still offers generated cards.
    #[test]
    fn a_choose_a_card_offer_matching_a_held_card_is_generated() {
        let wire = json!({"deck": {"draw_pile": [{"id": "SCRAWL"}]}, "encounter": []});
        let board = json!({"round": 1, "hand": [], "player": {"relics": [{"id": "TOOLBOX"}]}});
        let mut opts = vec![
            json!({"id": "SPLASH", "upgraded": false}),
            json!({"id": "SEEKER_STRIKE", "upgraded": false}),
            json!({"id": "SCRAWL", "upgraded": false}),
        ];
        let screen = json!("FromChooseACardScreen");
        let recognized = recognize_card_select(
            None,
            &mut opts,
            &board,
            &wire,
            1,
            &Value::Null,
            &json!("TO_OBTAIN"),
            &screen,
        );
        assert_eq!(recognized.0, "add_generated_to_hand");
        let from_pile = recognize_card_select(
            None,
            &mut opts,
            &board,
            &wire,
            1,
            &Value::Null,
            &Value::Null,
            &json!("FromCombatPile"),
        );
        assert_eq!(
            from_pile.0, "card_select",
            "a pile screen stays deterministic"
        );
    }

    /// GH #1758: Gambling Chip's offer leaves out a card Crossbow added
    /// while its screen was open.
    #[test]
    fn gambling_chip_offers_the_hand_it_collected() {
        let wire = json!({"deck": {"hand": [{"id": "IRON_WAVE"}, {"id": "BASH"}, {"id": "BLUDGEON"}]}, "encounter": []});
        let board = json!({"round": 1,
            "hand": [{"id": "IRON_WAVE"}, {"id": "BASH"}, {"id": "BLUDGEON"}],
            "player": {"relics": [{"id": "GAMBLING_CHIP"}, {"id": "CROSSBOW"}]}});
        let recognize = |prompt: Value| {
            let mut opts = vec![
                json!({"id": "IRON_WAVE", "upgraded": false}),
                json!({"id": "BASH", "upgraded": false}),
            ];
            recognize_card_select(
                None,
                &mut opts,
                &board,
                &wire,
                0,
                &Value::Null,
                &prompt,
                &json!("FromHand"),
            )
            .0
        };
        assert_eq!(
            recognize(json!("GAMBLING_CHIP.selectionScreenPrompt")),
            "discard_from_hand_and_draw"
        );
        assert_eq!(
            recognize(Value::Null),
            "card_select",
            "without the prompt, the whole hand only"
        );
    }

    /// GH #1756: Choices Paradox's offer is the engine's retained
    /// generation, even when an offered card is in the deck.
    #[test]
    fn choices_paradox_offers_generated_retained_cards() {
        let wire = json!({"deck": {"draw_pile": [{"id": "PERFECTED_STRIKE"}]}, "encounter": []});
        let board =
            json!({"round": 1, "hand": [], "player": {"relics": [{"id": "CHOICES_PARADOX"}]}});
        let mut opts = vec![
            json!({"id": "PERFECTED_STRIKE", "upgraded": false}),
            json!({"id": "BARRICADE", "upgraded": false}),
        ];
        let recognized = recognize_card_select(
            None,
            &mut opts,
            &board,
            &wire,
            1,
            &Value::Null,
            &json!("CHOICES_PARADOX.selectionScreenPrompt"),
            &json!("FromSimpleGrid"),
        );
        assert_eq!(recognized.0, "add_generated_retained_to_hand");
    }

    #[test]
    fn the_answer_maps_picks_to_the_bridge_handles() {
        let cards = [json!({"index": 7}), json!({"index": 9})];
        assert_eq!(
            card_select_answer(&cards, 0, Some(vec![1])),
            (json!({"action": "choose", "indices": [9]}), Some(vec![1]))
        );
        assert_eq!(
            card_select_answer(&cards, 0, Some(vec![])),
            (json!({"action": "skip"}), Some(vec![]))
        );
        assert_eq!(card_select_answer(&cards, 1, Some(vec![])).1, Some(vec![0]));
        assert_eq!(
            card_select_answer(&cards, 1, None),
            (json!({"action": "choose", "indices": [7]}), None)
        );
    }
}
