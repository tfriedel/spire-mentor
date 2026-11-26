//! The mod's `combat_action` board in the headless `combat_play` dialect
//! (GH #1426; was `realgame_adapter.to_headless_combat_play`).
//!
//! The engine never reads this dialect any more: the fight is built from the
//! mod's board in one pass (`sts2_workbench::combat_import`). What still
//! reads it is the advisor itself: the regret rows are labelled against its
//! hand and belt (`rows_from_mcts`), and the in-combat card select
//! recognizes its screens from its hand, round and relics. Values are
//! Python's: see [`crate::py`].

use serde_json::{json, Map, Value};

use crate::py::{self, as_list, get, truthy};

/// Why a board has no headless form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadlessError {
    /// Not a `combat_action` board (`ValueError` in Python).
    NotCombat(String),
    /// A game build the advisor does not support.
    GameVersion(String),
}

impl std::fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCombat(m) | Self::GameVersion(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for HeadlessError {}

/// `CHARACTER_ID_TO_DISPLAY`.
const CHARACTER_ID_TO_DISPLAY: [(&str, &str); 6] = [
    ("IRONCLAD", "Ironclad"),
    ("SILENT", "Silent"),
    ("DEFECT", "Defect"),
    ("NECROBINDER", "Necrobinder"),
    ("REGENT", "Regent"),
    ("FOOL", "Fool"),
];

/// `canonical_maps.ORB_TITLE_TO_ID`, as `(title, registry id)`.
const ORB_TITLE_TO_ID: [(&str, &str); 6] = [
    ("Dark", "DARK"),
    ("Frost", "FROST"),
    ("Glass", "GLASS"),
    ("Lightning", "LIGHTNING"),
    ("Plasma", "PLASMA"),
    ("Orb Slot", "EMPTY"),
];

/// `character_display_name`: the bridge's `run_state.character`
/// (`IRONCLAD`) in the Title-Case form the adapter takes, or `None`.
#[must_use]
pub fn character_display_name(character_id: &Value) -> Option<&'static str> {
    if !truthy(character_id) {
        return None;
    }
    let up = py::str_of(character_id).to_uppercase();
    CHARACTER_ID_TO_DISPLAY
        .iter()
        .find(|(id, _)| *id == up)
        .map(|(_, name)| *name)
}

/// `d.get(key, default)`: the default only when the key is ABSENT (a
/// present `null` stays `null`).
fn get_or(d: &Value, key: &str, default: Value) -> Value {
    d.get(key).cloned().unwrap_or(default)
}

/// `isinstance(v, int) and not isinstance(v, bool)`.
fn is_int(v: &Value) -> bool {
    matches!(v, Value::Number(n) if n.is_i64() || n.is_u64())
}

/// `isinstance(v, int)`, `bool` included.
fn is_int_or_bool(v: &Value) -> bool {
    is_int(v) || v.is_boolean()
}

/// `(d.get(key) or {}).get("cards") or []`.
fn pile<'a>(d: &'a Value, key: &str) -> &'a [Value] {
    let p = get(d, key);
    if truthy(p) {
        as_list(get(p, "cards"))
    } else {
        &[]
    }
}

/// `d.get(key) or []` over a list.
fn list_or_empty<'a>(d: &'a Value, key: &str) -> &'a [Value] {
    as_list(get(d, key))
}

/// Python's `str.title()`.
fn py_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_cased = false;
    for ch in s.chars() {
        let cased = ch.is_lowercase() || ch.is_uppercase();
        if cased {
            if prev_cased {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
        } else {
            out.push(ch);
        }
        prev_cased = cased;
    }
    out
}

/// `_normalize_affliction_name`.
fn normalize_affliction_name(name: &str) -> String {
    match name.strip_suffix(".title") {
        Some(stem) => py_title(&stem.replace('_', " ")).replace(' ', ""),
        None => name.to_string(),
    }
}

/// `_translate_power`.
fn translate_power(power: &Value) -> Value {
    let mut out = json!({
        "id": get(power, "id"),
        "name": get(power, "name"),
        "amount": get_or(power, "amount", json!(0)),
        "display_amount": get(power, "display_amount"),
        "vars": get(power, "vars"),
    });
    for key in ["applier", "player_target", "target"] {
        let v = get(power, key);
        if !v.is_null() {
            out[key] = v.clone();
        }
    }
    out
}

fn translate_powers(powers: &[Value]) -> Value {
    Value::Array(powers.iter().map(translate_power).collect())
}

/// `_keywords_for_hand_card`.
fn keywords_for_hand_card(card: &Value) -> Vec<Value> {
    let mut kws = Vec::new();
    for key in ["retain", "ethereal", "innate"] {
        if truthy(get(card, key)) {
            kws.push(json!(key));
        }
    }
    if let Value::String(reason) = get(card, "unplayable_reason") {
        if reason.contains("HasUnplayableKeyword") {
            kws.push(json!("unplayable"));
        }
    }
    kws
}

/// `_translate_card`.
fn translate_card(card: &Value, index: usize, hand: bool) -> Value {
    let mut out = json!({
        "index": index,
        "id": get(card, "id"),
        "name": get(card, "name"),
        "type": get(card, "type"),
        "cost": card.get("effective_cost").cloned().unwrap_or_else(|| get(card, "cost").clone()),
        "upgraded": truthy(get(card, "upgrade_level")) || truthy(get(card, "upgraded")),
    });
    for key in [
        "base_cost",
        "persistent_cost",
        "played_cost",
        "stable_cost",
        "star_cost",
        "persistent_star_cost",
    ] {
        let v = get(card, key);
        if is_int(v) {
            out[key] = v.clone();
        }
    }
    if let Value::Object(stats) = get(card, "stats") {
        let clean: Map<String, Value> = stats
            .iter()
            .filter(|(_, v)| is_int(v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !clean.is_empty() {
            out["stats"] = Value::Object(clean);
        }
    }
    if let Value::String(ench) = get(card, "enchantment") {
        if !ench.is_empty() {
            out["enchantment"] = json!(ench);
            let amount = get(card, "enchantment_amount");
            if is_int(amount) {
                out["enchantment_amount"] = amount.clone();
            }
        }
    }
    if let Value::String(aff) = get(card, "affliction") {
        if !aff.is_empty() {
            let first = aff.split(" / ").next().unwrap_or("").trim();
            out["affliction"] = json!(normalize_affliction_name(first));
            let stacks = card
                .get("affliction_stacks")
                .unwrap_or_else(|| get(card, "affliction_amount"));
            if is_int_or_bool(stacks) {
                out["affliction_stacks"] = stacks.clone();
            }
        }
    }
    if let Value::Object(vars) = get(card, "combat_vars") {
        if !vars.is_empty() {
            out["combat_vars"] = Value::Object(vars.clone());
        }
    }
    let replays = get(card, "base_replay_count");
    if is_int(replays) && truthy(replays) {
        out["base_replay_count"] = replays.clone();
    }
    if hand {
        let kws = keywords_for_hand_card(card);
        if !kws.is_empty() {
            out["keywords"] = Value::Array(kws);
        }
    }
    out
}

/// `_translate_pile`: the cards, numbered from `*next`, which moves past them.
fn translate_pile(cards: &[Value], next: &mut usize, hand: bool) -> Value {
    let out = cards
        .iter()
        .enumerate()
        .map(|(i, c)| translate_card(c, *next + i, hand))
        .collect();
    *next += cards.len();
    Value::Array(out)
}

/// `_translate_relic`.
fn translate_relic(relic: &Value) -> Value {
    let stack = get(relic, "stack_count");
    let vars = if is_int_or_bool(stack) && py::int_of(stack) > 1 {
        json!({"stack_count": stack})
    } else {
        Value::Null
    };
    let mut out = json!({"id": get(relic, "id"), "name": get(relic, "name"), "vars": vars});
    let counter = get(relic, "counter");
    let rid = get(relic, "id").as_str().unwrap_or("");
    let rid = rid.strip_prefix("RELIC.").unwrap_or(rid);
    if is_int(counter) {
        // The game's own number, in the engine's terms (GH #1684).
        let shown = py::int_of(counter);
        let id = rid;
        out["counter"] = sts2_e2e::snapshot::engine_counter(id, shown)
            .map_or_else(|| counter.clone(), |c| json!(c));
    } else {
        // No counter shown: for a relic counting down what is left, that is
        // its end (`adapter._ENGINE_COUNTER_UNSHOWN` twin, GH #1829).
        let id = get(relic, "id").as_str().unwrap_or("");
        let id = id.strip_prefix("RELIC.").unwrap_or(id);
        if let Some(c) = sts2_e2e::snapshot::engine_counter_unshown(id) {
            out["counter"] = json!(c);
        }
    }
    for key in ["is_wax", "is_melted"] {
        if truthy(get(relic, key)) {
            out[key] = json!(true);
        }
    }
    out
}

fn translate_relics(relics: &[Value]) -> Value {
    Value::Array(relics.iter().map(translate_relic).collect())
}

/// `_translate_intent`.
fn translate_intent(intent: &Value) -> Value {
    let mut out = json!({"type": get(intent, "intent_type")});
    for key in ["damage", "hits"] {
        let v = get(intent, key);
        if !v.is_null() {
            out[key] = v.clone();
        }
    }
    out
}

/// `_translate_enemy`.
fn translate_enemy(enemy: &Value, index: usize) -> Value {
    let move_state = if truthy(get(enemy, "move_state")) {
        get(enemy, "move_state")
    } else {
        get(enemy, "next_move_id")
    };
    let mut out = json!({
        "index": index,
        "id": get(enemy, "id"),
        "name": get(enemy, "name"),
        "hp": get(enemy, "hp"),
        "max_hp": get(enemy, "max_hp"),
        "block": get_or(enemy, "block", json!(0)),
        "is_dead": !truthy(&get_or(enemy, "is_alive", json!(true))),
        "powers": translate_powers(list_or_empty(enemy, "powers")),
        "intents": Value::Array(list_or_empty(enemy, "intents").iter().map(translate_intent).collect()),
        "move_state": move_state,
    });
    if let Value::String(s) = get(enemy, "fabricator_last_spawned") {
        if !s.is_empty() {
            out["fabricator_last_spawned"] = json!(s);
        }
    }
    for key in ["curse_of_knowledge_counter", "combat_id"] {
        let v = get(enemy, key);
        if is_int(v) {
            out[key] = v.clone();
        }
    }
    let ai = get(enemy, "ai");
    if ai.is_object() {
        out["ai"] = ai.clone();
    }
    out
}

/// `_adapter._id_key`.
fn id_key(raw: &str) -> String {
    raw.trim().replace('_', "").to_uppercase()
}

/// `_orb_type_stem`.
fn orb_type_stem(orb: &Value) -> Value {
    let raw_id = get(orb, "id");
    if let Value::String(raw) = raw_id {
        let stem = raw.trim();
        if !stem.is_empty() {
            let stem = stem
                .strip_suffix("_ORB")
                .or_else(|| stem.strip_suffix("Orb"))
                .unwrap_or(stem);
            let key = id_key(stem);
            if let Some((title, _)) = ORB_TITLE_TO_ID
                .iter()
                .rev()
                .find(|(_, id)| id_key(id) == key)
            {
                return json!(title);
            }
        }
    }
    if let Value::String(name) = get(orb, "name") {
        if !name.trim().is_empty() {
            return json!(name);
        }
    }
    if raw_id.is_string() {
        raw_id.clone()
    } else {
        json!("")
    }
}

/// `_translate_orb`.
fn translate_orb(orb: &Value) -> Value {
    json!({
        "type": orb_type_stem(orb),
        "passive": get_or(orb, "passive_value", json!(0)),
        "evoke": get_or(orb, "evoke_value", json!(0)),
    })
}

/// `_translate_osty`.
fn translate_osty(pets: &[Value], character_id: &Value) -> Option<Value> {
    let osty = pets.iter().find(|p| {
        let id = get(p, "id");
        let id = if truthy(id) {
            py::str_of(id)
        } else {
            String::new()
        };
        id.to_uppercase() == "OSTY"
    });
    let Some(osty) = osty else {
        let ch = if truthy(character_id) {
            py::str_of(character_id)
        } else {
            String::new()
        };
        return (ch.to_uppercase() == "NECROBINDER")
            .then(|| json!({"alive": false, "hp": 0, "max_hp": 0, "block": 0}));
    };
    Some(json!({
        "alive": truthy(&get_or(osty, "is_alive", json!(false))),
        "hp": get_or(osty, "hp", json!(0)),
        "max_hp": get_or(osty, "max_hp", json!(0)),
        "block": get_or(osty, "block", json!(0)),
    }))
}

/// The v0.113.0 Wuppy pet (`WUPPY` in the board's `pets`), in the Osty's
/// headless shape; `None` without one. `sts2_hint.adapter.translate` carries
/// it to the wire's `$.player.wuppy`.
fn translate_wuppy(pets: &[Value]) -> Option<Value> {
    let wuppy = pets.iter().find(|p| {
        let id = get(p, "id");
        truthy(id) && py::str_of(id).to_uppercase() == "WUPPY"
    })?;
    Some(json!({
        "alive": truthy(&get_or(wuppy, "is_alive", json!(false))),
        "hp": get_or(wuppy, "hp", json!(0)),
        "max_hp": get_or(wuppy, "max_hp", json!(0)),
        "block": get_or(wuppy, "block", json!(0)),
    }))
}

/// `int(v or 0)`.
fn int_or_zero(v: &Value) -> i64 {
    if truthy(v) {
        py::int_of(v)
    } else {
        0
    }
}

/// One co-op teammate (PROTOCOL.md `allies`).
fn translate_ally(a: &Value, next: &mut usize) -> Value {
    let mut ally = json!({
        "hp": int_or_zero(get(a, "hp")),
        "max_hp": int_or_zero(get(a, "max_hp")),
        "block": int_or_zero(get(a, "block")),
    });
    if let Value::Array(hand) = get(a, "hand") {
        ally["hand"] = translate_pile(hand, next, true);
        ally["character"] = json!(character_display_name(get(a, "character")));
        ally["energy"] = json!(int_or_zero(get(a, "energy")));
        ally["max_energy"] = json!(int_or_zero(get(a, "max_energy")));
        ally["powers"] = translate_powers(list_or_empty(a, "powers"));
        ally["ended_turn"] = json!(truthy(get(a, "ended_turn")));
        for key in ["draw_pile", "discard_pile", "exhaust_pile", "play_pile"] {
            let p = get(a, key);
            let cards = if truthy(p) {
                get(p, "cards")
            } else {
                &Value::Null
            };
            if let Value::Array(cards) = cards {
                ally[key] = translate_pile(cards, next, false);
            }
        }
        if let Value::Array(relics) = get(a, "relics") {
            ally["relics"] = translate_relics(relics);
        }
        if is_int_or_bool(get(a, "stars")) {
            ally["stars"] = get(a, "stars").clone();
        }
        if get(a, "potions").is_array() {
            ally["potions"] = get(a, "potions").clone();
        }
        if get(a, "hp_lost_this_turn") == &json!(true) {
            ally["hp_lost_this_turn"] = json!(true);
        }
        for key in [
            "unblocked_hits_this_combat",
            "powered_hits_this_turn",
            "cards_played_this_turn",
            "cards_exhausted_this_combat",
            "cards_exhausted_this_turn",
            "card_block_gains_this_turn",
            "extra_turns_taken_this_combat",
            "cards_drawn_this_combat",
            "cards_generated_this_combat",
        ] {
            if truthy(get(a, key)) {
                ally[key] = get(a, key).clone();
            }
        }
        if let Value::Array(orbs) = get(a, "orbs") {
            ally["orbs"] = Value::Array(orbs.iter().map(translate_orb).collect());
            ally["orb_slots"] = get_or(a, "orb_capacity", json!(orbs.len()));
        }
        if let Value::Array(pets) = get(a, "pets") {
            if let Some(osty) = translate_osty(pets, get(a, "character")) {
                ally["osty"] = osty;
            }
        }
    }
    if !get(a, "net_id").is_null() {
        ally["net_id"] = json!(py::str_of(get(a, "net_id")));
    }
    ally
}

/// `{run_seed, counters}` when both are there (`bool_seed`: whether a
/// `bool` counts as an int seed, as the co-op block's check lets it).
fn rng_block(rng: &Value, bool_seed: bool) -> Option<Value> {
    let seed = get(rng, "run_seed");
    let seed_ok = if bool_seed {
        is_int_or_bool(seed)
    } else {
        is_int(seed)
    };
    let counters = get(rng, "counters");
    (rng.is_object() && seed_ok && counters.as_object().is_some_and(|c| !c.is_empty()))
        .then(|| json!({"run_seed": seed, "counters": counters}))
}

/// `to_headless_combat_play`: a bridge `combat_action` board in the headless
/// `combat_play` dialect.
///
/// # Errors
///
/// [`HeadlessError::NotCombat`] for any other board,
/// [`HeadlessError::GameVersion`] for a build the advisor cannot model.
pub fn to_headless_combat_play(state: &Value) -> Result<Value, HeadlessError> {
    if get(state, "type") != "combat_action" {
        return Err(HeadlessError::NotCombat(format!(
            "to_headless_combat_play: expected state['type'] == 'combat_action', got {}",
            py::repr(get(state, "type"))
        )));
    }
    let empty = json!({});
    let player = if truthy(get(state, "player")) {
        get(state, "player")
    } else {
        &empty
    };
    let run_state = if truthy(get(state, "run_state")) {
        get(state, "run_state")
    } else {
        &empty
    };

    let mut next = 0usize;
    let hand = translate_pile(list_or_empty(state, "hand"), &mut next, true);
    let draw = translate_pile(pile(state, "draw_pile"), &mut next, false);
    let discard = translate_pile(pile(state, "discard_pile"), &mut next, false);
    let exhaust = translate_pile(pile(state, "exhaust_pile"), &mut next, false);
    let play = translate_pile(pile(state, "play_pile"), &mut next, false);
    let master_deck = translate_pile(list_or_empty(run_state, "master_deck"), &mut next, false);

    let enemies: Vec<Value> = list_or_empty(state, "enemies")
        .iter()
        .enumerate()
        .map(|(i, e)| translate_enemy(e, i))
        .collect();

    let room_type = match get(state, "room_type") {
        Value::String(s) if !s.is_empty() => s.clone(),
        _ => "Combat".into(),
    };
    let energy = get_or(player, "energy", json!(0));
    let potions = if truthy(get(state, "potions")) {
        get(state, "potions").clone()
    } else {
        json!([])
    };
    let mut player_out = json!({
        "hp": get(player, "hp"),
        "max_hp": get(player, "max_hp"),
        "block": get_or(player, "block", json!(0)),
        "relics": translate_relics(list_or_empty(run_state, "relics")),
        "potions": potions,
        "gold": get(run_state, "gold"),
    });
    for key in [
        "cards_played_this_turn",
        "bound_card_played_this_turn",
        "hp_lost_this_turn",
        "unblocked_hits_this_combat",
        "powered_hits_this_turn",
        "cards_exhausted_this_combat",
        "cards_exhausted_this_turn",
        "card_block_gains_this_turn",
        "extra_turns_taken_this_combat",
        "cards_drawn_this_combat",
        "cards_generated_this_combat",
        "discards_this_turn",
    ] {
        player_out[key] = get(player, key).clone();
    }
    let slot_count = get(run_state, "potion_slot_count");
    if is_int(slot_count) {
        player_out["potion_slots"] = slot_count.clone();
    }

    let mut out = json!({
        "type": "decision",
        "decision": "combat_play",
        "context": {"room_type": room_type},
        "round": get_or(state, "round", json!(1)),
        "energy": energy.clone(),
        "max_energy": player.get("max_energy").cloned().unwrap_or(energy),
        "hand": hand,
        "enemies": enemies,
        "player": player_out,
        "player_powers": translate_powers(list_or_empty(player, "powers")),
        "draw_pile": draw,
        "discard_pile": discard,
        "exhaust_pile": exhaust,
        "play_pile": play,
        "master_deck": master_deck,
    });
    let act = get(state, "act");
    if is_int(act) {
        out["act"] = act.clone();
    }
    let floor = get(state, "floor");
    if is_int(floor) {
        out["context"]["total_floor"] = floor.clone();
    }
    if let Value::String(mode) = get(run_state, "game_mode") {
        match mode.trim().to_lowercase().as_str() {
            "dread" => {
                out["context"]["game_mode"] = json!("dread");
                if let Value::String(d) = get(run_state, "dominion") {
                    if !d.is_empty() {
                        out["context"]["dominion"] = json!(d);
                    }
                }
            }
            "blitz" => out["context"]["game_mode"] = json!("blitz"),
            _ => {}
        }
    }

    let allies: Vec<Value> = list_or_empty(state, "allies")
        .iter()
        .filter(|a| a.is_object())
        .map(|a| translate_ally(a, &mut next))
        .collect();
    if !allies.is_empty() {
        out["allies"] = Value::Array(allies);
        if !get(player, "net_id").is_null() {
            out["net_id"] = json!(py::str_of(get(player, "net_id")));
        }
        if is_int_or_bool(get(player, "seat_index")) {
            out["seat_index"] = get(player, "seat_index").clone();
        }
    }

    if let Some(rng) = rng_block(get(state, "rng"), false) {
        out["rng"] = rng;
    }
    if let Some(rng) = rng_block(get(state, "coop_rng"), true) {
        out["coop_rng"] = rng;
    }
    let stars = get(player, "stars");
    if is_int(stars) {
        out["stars"] = stars.clone();
    }
    if let Some(orbs) = state.get("orbs").filter(|o| !o.is_null()) {
        let orbs = as_list(orbs);
        out["orbs"] = Value::Array(orbs.iter().map(translate_orb).collect());
        out["orb_slots"] = get_or(state, "orb_capacity", json!(orbs.len()));
    }
    let pets = list_or_empty(state, "pets");
    if let Some(osty) = translate_osty(pets, get(run_state, "character")) {
        out["osty"] = osty;
    }
    if let Some(wuppy) = translate_wuppy(pets) {
        out["wuppy"] = wuppy;
    }
    if let Some(version) = game_version(state)? {
        out["game_version"] = json!(version);
    }
    Ok(out)
}

/// `envelope_game_version`, `None` for the legacy build (v0.107.1): the
/// build named on the envelope, a hotfix as its base build.
///
/// # Errors
///
/// [`HeadlessError::GameVersion`] for a build outside the supported set.
pub fn game_version(msg: &Value) -> Result<Option<&'static str>, HeadlessError> {
    use sts2_core::game_version::GameVersion;
    match get(msg, "game_version") {
        Value::Null => Ok(None),
        Value::String(s) if s.trim().is_empty() => Ok(None),
        Value::String(s) => match s.parse::<GameVersion>() {
            Ok(GameVersion::V0_107_1) => Ok(None),
            Ok(v) => Ok(Some(v.as_str())),
            Err(_) => Err(unsupported_game_version(get(msg, "game_version"))),
        },
        other => Err(unsupported_game_version(other)),
    }
}

/// `UnsupportedGameVersionError(raw)`: written for the player, since the
/// mod renders it in the HUD as the advice note.
fn unsupported_game_version(raw: &Value) -> HeadlessError {
    use sts2_core::game_version::GameVersion;
    let supported: Vec<&str> = GameVersion::ALL.iter().map(|v| v.as_str()).collect();
    HeadlessError::GameVersion(crate::i18n::trf(
        "Spire Mentor's advisor does not support game version {raw} (supported: {supported}). Update the advisor, or switch the game to a supported build in Steam (Properties > Betas).",
        &[("raw", &py::repr(raw)), ("supported", &supported.join(", "))],
    ))
}

#[cfg(test)]
mod tests {

    /// GH #1684: the game counts Sword of Stone's elites up, the engine down;
    /// the student reads the game's number back off the engine's.
    #[test]
    fn sword_of_stone_reads_the_games_count_into_the_engines() {
        let r = translate_relic(
            &json!({"id": "SWORD_OF_STONE", "name": "Sword of Stone", "counter": 1}),
        );
        assert_eq!(r["counter"], json!(4), "four elites to go");
        assert_eq!(
            sts2_e2e::snapshot::display_counter("SWORD_OF_STONE", 4),
            Some(1)
        );
        let other = translate_relic(&json!({"id": "PEN_NIB", "name": "Pen Nib", "counter": 7}));
        assert_eq!(other["counter"], json!(7));
    }

    /// A melted Kusarigama keeps the count it melted with, in C# and in the
    /// engine.
    #[test]
    fn a_melted_kusarigama_keeps_its_count() {
        let melted = translate_relic(&json!(
            {"id": "KUSARIGAMA", "name": "k", "counter": 1, "is_wax": true, "is_melted": true}
        ));
        assert_eq!(melted["counter"], json!(1));
        let live = translate_relic(&json!({"id": "KUSARIGAMA", "name": "k", "counter": 1}));
        assert_eq!(live["counter"], json!(1));
    }

    /// GH #1829: C# hides Wongo's ticket counter past its payout; read as 0,
    /// the ticket showed a fresh 5 for the rest of the run.
    #[test]
    fn a_hidden_ticket_counter_reads_as_paid_out() {
        let r =
            translate_relic(&json!({"id": "WONGOS_MYSTERY_TICKET", "name": "t", "counter": null}));
        assert_eq!(r["counter"], json!(-1));
        assert_eq!(
            sts2_e2e::snapshot::display_counter("WONGOS_MYSTERY_TICKET", -1),
            None
        );
        let c = translate_relic(&json!({"id": "SILVER_CRUCIBLE", "name": "c"}));
        assert_eq!(c["counter"], json!(3));
        let other = translate_relic(&json!({"id": "PEN_NIB", "name": "Pen Nib"}));
        assert!(other.get("counter").is_none());
    }

    use super::*;

    #[test]
    fn a_board_translates_and_anything_else_is_refused() {
        let board = json!({
            "type": "combat_action",
            "player": {"energy": 3, "hp": 70},
            "hand": [{"id": "STRIKE_IRONCLAD", "upgrade_level": 1, "retain": true}],
            "draw_pile": {"cards": [{"id": "BASH", "affliction": "TAINTED.title"}]},
            "orbs": [{"id": "FROST_ORB"}],
            "run_state": {"character": "NECROBINDER"},
        });
        let out = to_headless_combat_play(&board).unwrap();
        assert_eq!(out["max_energy"], 3);
        assert_eq!(out["hand"][0]["upgraded"], true);
        assert_eq!(out["hand"][0]["keywords"], json!(["retain"]));
        assert_eq!(out["draw_pile"][0]["index"], 1);
        assert_eq!(out["draw_pile"][0]["affliction"], "Tainted");
        assert_eq!(out["orbs"][0]["type"], "Frost");
        assert_eq!(out["osty"]["alive"], false);
        assert!(matches!(
            to_headless_combat_play(&json!({"type": "card_select"})),
            Err(HeadlessError::NotCombat(_))
        ));
        let refused = game_version(&json!({"game_version": "v0.99.0"})).unwrap_err();
        assert_eq!(
            refused.to_string(),
            "Spire Mentor's advisor does not support game version 'v0.99.0' (supported: \
             v0.107.1, v0.111.0, v0.113.0). Update the advisor, or switch the game to a \
             supported build in Steam (Properties > Betas)."
        );
    }
}
