//! The `combat_play` state view, plus the small dict builders every screen
//! shares (owned relics / potions, deck cards, intents, powers).
//!
//! Port of the module-level half of
//! `sts2-pycli/python/workbench/backends/rust_game.py` (`build_combat_view`,
//! `_combat_card_dict`, `_stolen_card_dict`, `_owned_entry`, `_intent`,
//! `_power`, `_top_level_osty`, `_treasure_reveal`, `_action_label`,
//! `_act_name`, `_boss_encounter_title`). Every function builds
//! `serde_json::Value`s with exactly the keys the Python emits — key
//! presence is part of the wire contract the SPA and the golden-parity
//! tests read, so an optional field is OMITTED (never `null`) wherever the
//! Python omitted it, and `null` wherever the Python wrote `None`.

use serde_json::{json, Map, Value};
use sts2_core::run_manager::RunAction;

use crate::display_maps::{loc_id, monster_title, power_title};
use crate::rmx;

/// `_ACT_NAMES` + `_act_name`: display name of the act at 1-based
/// `act_index`, or `None` for an index this engine has no act for.
///
/// Act 1 has TWO models and the name says WHICH (GH #544): `Overgrowth`
/// (default) and `Underdocks`. These strings are what the C# oracle reports
/// through `act_name`, and — because the frontend keys every painted region
/// off `act_name` — they are also the asset-directory names.
#[must_use]
pub fn act_name(act_index: u8, act1: &str) -> Option<&'static str> {
    match act_index {
        1 => Some(if act1 == "underdocks" {
            "Underdocks"
        } else {
            "Overgrowth"
        }),
        2 => Some("Hive"),
        3 => Some("Glory"),
        _ => None,
    }
}

/// `_BOSS_KIND_ENCOUNTER_NAME`: boss encounters the map labels by their
/// common encounter name rather than by the fight's lead monster (the Queen
/// fight's lead is the Torch Head Amalgam it spawns alongside her, The Kin's a
/// follower).
#[must_use]
pub fn boss_kind_encounter_name(kind: &str) -> Option<&'static str> {
    match kind {
        "TorchHeadAmalgam" => Some("Queen"),
        // The Kin's lead is a follower (roster in C# spawn order, priest last).
        "KinFollower" => Some("The Kin"),
        _ => None,
    }
}

/// `_boss_encounter_title`: `QUEEN_BOSS` -> `Queen`,
/// `LAGAVULIN_MATRIARCH_BOSS` -> `Lagavulin Matriarch`.
#[must_use]
pub fn boss_encounter_title(encounter_id: &str) -> String {
    let stem = encounter_id.strip_suffix("_BOSS").unwrap_or(encounter_id);
    stem.split('_')
        .filter(|w| !w.is_empty())
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Python `str.capitalize()`: first char upper, the rest lower.
#[must_use]
pub fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut out: String = first.to_uppercase().collect();
            out.push_str(&chars.as_str().to_lowercase());
            out
        }
    }
}

/// Python `str.title()`: every run of letters starts upper, the rest lower.
#[must_use]
pub fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_is_cased = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_is_cased {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_is_cased = true;
        } else {
            out.push(c);
            prev_is_cased = false;
        }
    }
    out
}

/// `_action_label`: human label for a raw engine action on the generic
/// option screens (treasure / relic reward / boss relic).
#[must_use]
pub fn action_label(a: &RunAction) -> String {
    let v = rmx::action_value(a);
    let kind = v.get("action").and_then(Value::as_str).unwrap_or("?");
    let idx = v.get("idx").cloned().unwrap_or(Value::Null);
    match kind {
        "pick_relic_reward" => "Take relic".to_string(),
        "skip_relic" => "Skip".to_string(),
        "collect_treasure" => "Open".to_string(),
        "pick_boss_relic" => format!("Relic {}", py_str(&idx)),
        "sacrifice_card_reward" => "Sacrifice".to_string(),
        "smith_card" => format!("Upgrade card {}", py_str(&idx)),
        other => title_case(&other.replace('_', " ")),
    }
}

/// Python `str(value)` for the scalar JSON values the labels interpolate
/// (`None`, ints, strings).
#[must_use]
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// `_owned_entry(raw_id, None)`: owned relic / potion dict whose `name` and
/// `description` are loc keys for the enrichment step to resolve.
#[must_use]
pub fn owned_entry(raw_id: &str) -> Value {
    let cid = loc_id(raw_id);
    json!({
        "id": cid,
        "name": format!("{cid}.title"),
        "description": format!("{cid}.description"),
    })
}

/// A master-deck / option card dict (`_deck_card`, `_deck_subset_cards`,
/// the `_fill_card_*` rows, `fill_combat_choice`'s rows): `index` is
/// whatever index space the screen's action resolves on.
#[must_use]
pub fn deck_card(index: i64, raw_id: &str, upgraded: bool) -> Value {
    let cid = loc_id(raw_id);
    json!({
        "index": index,
        "id": cid,
        "name": format!("{cid}.title"),
        "description": format!("{cid}.description"),
        "upgraded": upgraded,
        "type": "card",
    })
}

/// A reward / shop card dict without an `upgraded` flag (`_fill_card_reward`,
/// `_fill_bundle_select` rows, minus the bundle's missing `index`).
#[must_use]
pub fn offer_card(index: Option<i64>, raw_id: &str) -> Value {
    let cid = loc_id(raw_id);
    let mut m = Map::new();
    if let Some(i) = index {
        m.insert("index".into(), json!(i));
    }
    m.insert("id".into(), json!(cid));
    m.insert("name".into(), json!(format!("{cid}.title")));
    m.insert("description".into(), json!(format!("{cid}.description")));
    m.insert("type".into(), json!("card"));
    Value::Object(m)
}

/// `_combat_card_dict`: frontend hand / pile card from a wire card.
#[must_use]
pub fn combat_card(index: i64, wire: &Value, can_play: bool, target_type: &str) -> Value {
    let cid = loc_id(wire.get("id").and_then(Value::as_str).unwrap_or(""));
    let mut stats = Map::new();
    if let Some(d) = wire.get("base_damage").filter(|v| truthy(v)) {
        stats.insert("damage".into(), d.clone());
    }
    if let Some(b) = wire.get("base_block").filter(|v| truthy(v)) {
        stats.insert("block".into(), b.clone());
    }
    let mut out = Map::new();
    out.insert("index".into(), json!(index));
    out.insert("id".into(), json!(cid));
    out.insert("name".into(), json!(format!("{cid}.title")));
    out.insert("description".into(), json!(format!("{cid}.description")));
    out.insert("stats".into(), Value::Object(stats));
    out.insert(
        "cost".into(),
        wire.get("cost_override").cloned().unwrap_or(json!(0)),
    );
    // The card's PRINTED cost, next to the effective one above (the frame
    // colours the numeral off the difference). Python `wire.get("base_cost")`:
    // absent -> None.
    out.insert(
        "base_cost".into(),
        wire.get("base_cost").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "type".into(),
        wire.get("kind").cloned().unwrap_or(json!("card")),
    );
    out.insert(
        "upgraded".into(),
        wire.get("upgraded").cloned().unwrap_or(json!(false)),
    );
    out.insert("can_play".into(), json!(can_play));
    out.insert("target_type".into(), json!(target_type));
    // Carried ONLY when the engine sent one: an omitted key is how the ~530
    // cards without such a value say "I have no such number".
    if let Some(cv) = wire.get("calculated_value").filter(|v| !v.is_null()) {
        out.insert("calculated_value".into(), cv.clone());
    }
    if let Some(aff) = wire.get("affliction").filter(|v| truthy(v)) {
        out.insert("affliction".into(), aff.clone());
        if let Some(stacks) = wire.get("affliction_stacks").filter(|v| truthy(v)) {
            out.insert("affliction_amount".into(), stacks.clone());
        }
    }
    Value::Object(out)
}

/// `_stolen_card_dict`: a card a `SwipePower` thief is holding hostage.
#[must_use]
pub fn stolen_card(wire: &Value) -> Value {
    let cid = loc_id(wire.get("id").and_then(Value::as_str).unwrap_or(""));
    json!({
        "index": -1,
        "id": cid,
        "name": format!("{cid}.title"),
        "description": format!("{cid}.description"),
        "upgraded": truthy(wire.get("upgraded").unwrap_or(&Value::Null)),
        "type": "card",
    })
}

/// `_intent`: wire `{kind, amount, times}` -> C# `{type, damage, hits}`,
/// with the EFFECTIVE telegraph as `damage` and the base kept alongside only
/// when the two differ. `modifiers` forwarded verbatim.
#[must_use]
pub fn intent(wire: &Value) -> Value {
    let mut out = Map::new();
    out.insert(
        "type".into(),
        wire.get("kind").cloned().unwrap_or(json!("Unknown")),
    );
    if let Some(base) = wire.get("amount").filter(|v| !v.is_null()) {
        let effective = wire
            .get("effective_amount")
            .filter(|v| !v.is_null())
            .unwrap_or(base);
        out.insert("damage".into(), effective.clone());
        if effective != base {
            out.insert("base_damage".into(), base.clone());
        }
    }
    if let Some(times) = wire.get("times").filter(|v| !v.is_null()) {
        out.insert("hits".into(), times.clone());
    }
    if let Some(mods) = wire.get("modifiers").filter(|v| truthy(v)) {
        out.insert("modifiers".into(), mods.clone());
    }
    Value::Object(out)
}

/// `_power`: display title + registry id + amounts.
#[must_use]
pub fn power(wire: &Value) -> Value {
    let amount = wire.get("amount").cloned().unwrap_or(json!(0));
    let raw_name = wire.get("name").and_then(Value::as_str).unwrap_or("");
    json!({
        "name": power_title(raw_name),
        "id": raw_name,
        "amount": amount,
        "display_amount": wire.get("display_amount").cloned().unwrap_or(amount),
    })
}

/// `_top_level_osty`: lift the wire `player.osty` to the top-level shape,
/// naming a living pet so its sprite resolves.
#[must_use]
pub fn top_level_osty(wire: Option<&Value>) -> Option<Value> {
    let wire = wire.filter(|v| truthy(v))?;
    if truthy(wire.get("alive").unwrap_or(&Value::Null)) {
        let mut m = wire.as_object().cloned().unwrap_or_default();
        m.insert("name".into(), json!("Osty"));
        Some(Value::Object(m))
    } else {
        Some(wire.clone())
    }
}

/// Python truthiness for the JSON values these builders test.
#[must_use]
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `build_combat_view`: the canonical `combat_play` state dict from a wire
/// combat state (`emit::combat_state_to_value`), the legal-action list and
/// the per-hand previews.
///
/// `legal` is the run backend's `{action: "play_card", hand_idx,
/// target_index}` list (the native `{action: "play", card_index}` shape is
/// accepted too, as in Python).
#[must_use]
pub fn build_combat_view(
    wire: &Value,
    character: &str,
    legal: &[Value],
    damage_preview: &[Value],
    block_preview: &[Value],
) -> Value {
    // Legality + targeting per hand position.
    let mut playable: Vec<i64> = Vec::new();
    let mut targeted: Vec<(i64, bool)> = Vec::new();
    for a in legal {
        let kind = a.get("action").and_then(Value::as_str).unwrap_or("");
        if kind != "play" && kind != "play_card" {
            continue;
        }
        let raw_ci = a
            .get("card_index")
            .filter(|v| !v.is_null())
            .or_else(|| a.get("hand_idx"))
            .filter(|v| !v.is_null());
        let Some(ci) = raw_ci.and_then(Value::as_i64) else {
            continue;
        };
        if !playable.contains(&ci) {
            playable.push(ci);
        }
        let has_target = a.get("target_index").is_some_and(|t| !t.is_null());
        match targeted.iter_mut().find(|(i, _)| *i == ci) {
            Some(slot) => slot.1 = slot.1 || has_target,
            None => targeted.push((ci, has_target)),
        }
    }
    let is_targeted = |i: i64| targeted.iter().any(|(c, t)| *c == i && *t);

    let player_wire = wire.get("player").cloned().unwrap_or(Value::Null);
    let energy = player_wire
        .get("energy")
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let empty = Vec::new();
    let deck = wire.get("deck").cloned().unwrap_or(Value::Null);
    let hand_wire = deck.get("hand").and_then(Value::as_array).unwrap_or(&empty);
    let mut hand = Vec::with_capacity(hand_wire.len());
    for (i, c) in hand_wire.iter().enumerate() {
        let idx = i as i64;
        let tt = if is_targeted(idx) { "AnyEnemy" } else { "None" };
        let can_play = playable.contains(&idx);
        let mut card = combat_card(idx, c, can_play, tt);
        let cm = card.as_object_mut().expect("card dict");
        if !can_play {
            // WHY the card cannot be played: a resource shortfall reddens the
            // cost numeral; everything else shows the unplayable slash.
            let cost = cm.get("cost").and_then(Value::as_i64);
            let reason = match cost {
                Some(c) if c > energy => "cost",
                _ => "blocked",
            };
            cm.insert("unplayable_reason".into(), json!(reason));
        }
        if let Some(dp) = damage_preview.get(i).filter(|v| truthy(v)) {
            // JSON object keys are strings; normalize so the frontend can look
            // up `damage_preview[String(enemyIndex)]`.
            let mut norm = Map::new();
            if let Some(obj) = dp.as_object() {
                for (k, v) in obj {
                    norm.insert(k.clone(), json!(v.as_i64().unwrap_or(0)));
                }
            }
            cm.insert("damage_preview".into(), Value::Object(norm));
        }
        if let Some(bp) = block_preview.get(i).filter(|v| !v.is_null()) {
            let blk = bp.as_i64().unwrap_or(0);
            let printed = cm
                .get("stats")
                .and_then(|s| s.get("block"))
                .and_then(Value::as_i64);
            if printed != Some(blk) {
                cm.insert("block_preview".into(), json!(blk));
            }
        }
        hand.push(card);
    }

    let mut enemies = Vec::new();
    for (pos, c) in wire
        .get("encounter")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .enumerate()
    {
        if truthy(c.get("is_dead").unwrap_or(&Value::Null)) {
            continue;
        }
        let kind = c.get("kind").and_then(Value::as_str).unwrap_or("");
        let mut e = Map::new();
        e.insert("index".into(), json!(pos));
        e.insert("uid".into(), c.get("uid").cloned().unwrap_or(Value::Null));
        e.insert("kind".into(), json!(kind));
        e.insert("name".into(), json!(monster_title(kind)));
        e.insert("hp".into(), c.get("hp").cloned().unwrap_or(Value::Null));
        e.insert(
            "max_hp".into(),
            c.get("max_hp").cloned().unwrap_or(Value::Null),
        );
        e.insert("block".into(), c.get("block").cloned().unwrap_or(json!(0)));
        e.insert(
            "intents".into(),
            Value::Array(
                c.get("intents")
                    .and_then(Value::as_array)
                    .unwrap_or(&empty)
                    .iter()
                    .map(intent)
                    .collect(),
            ),
        );
        e.insert(
            "powers".into(),
            Value::Array(
                c.get("powers")
                    .and_then(Value::as_array)
                    .unwrap_or(&empty)
                    .iter()
                    .map(power)
                    .collect(),
            ),
        );
        e.insert(
            "move_state".into(),
            c.get("move_state").cloned().unwrap_or(Value::Null),
        );
        let stolen: Vec<Value> = c
            .get("stolen_cards")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(stolen_card)
            .collect();
        if !stolen.is_empty() {
            e.insert("stolen_cards".into(), Value::Array(stolen));
        }
        enemies.push(Value::Object(e));
    }

    let pile = |key: &str| -> Vec<Value> {
        deck.get(key)
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(|c| combat_card(-1, c, false, "None"))
            .collect()
    };
    let draw = pile("draw_pile");
    let discard = pile("discard_pile");
    let exhaust = pile("exhaust_pile");

    let relics: Vec<Value> = player_wire
        .get("relics")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .map(|r| owned_entry(wire_id(r)))
        .collect();
    let potions: Vec<Value> = player_wire
        .get("potions")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .filter(|p| truthy(p))
        .map(|p| owned_entry(wire_id(p)))
        .collect();

    let mut state = Map::new();
    state.insert("decision".into(), json!("combat_play"));
    state.insert(
        "round".into(),
        wire.get("round").cloned().unwrap_or(json!(0)),
    );
    state.insert(
        "energy".into(),
        player_wire.get("energy").cloned().unwrap_or(json!(0)),
    );
    state.insert(
        "max_energy".into(),
        player_wire.get("max_energy").cloned().unwrap_or(json!(0)),
    );
    state.insert(
        "player".into(),
        json!({
            "name": character,
            "hp": player_wire.get("hp").cloned().unwrap_or(Value::Null),
            "max_hp": player_wire.get("max_hp").cloned().unwrap_or(Value::Null),
            "block": player_wire.get("block").cloned().unwrap_or(json!(0)),
            "relics": relics,
            "potions": potions,
            "deck": [],
        }),
    );
    state.insert(
        "player_powers".into(),
        Value::Array(
            player_wire
                .get("powers")
                .and_then(Value::as_array)
                .unwrap_or(&empty)
                .iter()
                .map(power)
                .collect(),
        ),
    );
    state.insert("enemies".into(), Value::Array(enemies));
    state.insert("hand".into(), Value::Array(hand));
    state.insert("draw_pile_count".into(), json!(draw.len()));
    state.insert("discard_pile_count".into(), json!(discard.len()));
    state.insert("exhaust_pile_count".into(), json!(exhaust.len()));
    state.insert("draw_pile".into(), Value::Array(draw));
    state.insert("discard_pile".into(), Value::Array(discard));
    state.insert("exhaust_pile".into(), Value::Array(exhaust));

    if let Some(osty) = top_level_osty(player_wire.get("osty")) {
        state.insert("osty".into(), osty);
    }
    if let Some(orbs) = player_wire.get("orbs") {
        state.insert("orbs".into(), orbs.clone());
        state.insert(
            "orb_slots".into(),
            player_wire.get("orb_slots").cloned().unwrap_or(json!(0)),
        );
    }
    state.insert(
        "stars".into(),
        player_wire.get("stars").cloned().unwrap_or(json!(0)),
    );
    Value::Object(state)
}

/// A wire relic / potion entry is either `{id: ...}` or a bare id string.
fn wire_id(v: &Value) -> &str {
    match v {
        Value::String(s) => s.as_str(),
        Value::Object(o) => o.get("id").and_then(Value::as_str).unwrap_or(""),
        _ => "",
    }
}

/// `_treasure_reveal`: the "what's inside" payload for a chest just walked
/// into, else `None`. Fires only on the action that ENTERED the room.
#[must_use]
pub fn treasure_reveal(
    before: Option<&RoomSnapshot>,
    after: &RoomSnapshot,
    act: u8,
) -> Option<Value> {
    let before = before?;
    let room = after.room.as_ref()?;
    if room.get("room_type").and_then(Value::as_str) != Some("treasure") {
        return None;
    }
    if let Some(was) = before.room.as_ref() {
        if was.get("room_type").and_then(Value::as_str) == Some("treasure")
            && was.get("map_coord") == room.get("map_coord")
        {
            return None; // already in this chest room — not a fresh entry
        }
    }
    let gained: Vec<&String> = after
        .relics
        .iter()
        .filter(|r| !before.relics.contains(r))
        .collect();
    let relic = gained.first().map(|r| owned_entry(r));
    Some(json!({
        "kind": "treasure",
        "act": act,
        "gold": i64::from(after.gold) - i64::from(before.gold),
        "relic": relic,
    }))
}

/// `RustGame._room_snapshot`: gold / owned relics / current room, for the
/// treasure-reveal diff.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomSnapshot {
    pub gold: i32,
    pub relics: Vec<String>,
    /// `current_room()` dict (`{room_type, map_coord}`) or `None`.
    pub room: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_names_match_python_table() {
        assert_eq!(act_name(1, "overgrowth"), Some("Overgrowth"));
        assert_eq!(act_name(1, "underdocks"), Some("Underdocks"));
        assert_eq!(act_name(1, "roll"), Some("Overgrowth"));
        assert_eq!(act_name(2, "overgrowth"), Some("Hive"));
        assert_eq!(act_name(3, "overgrowth"), Some("Glory"));
        assert_eq!(act_name(4, "overgrowth"), None);
    }

    #[test]
    fn boss_titles() {
        assert_eq!(boss_encounter_title("QUEEN_BOSS"), "Queen");
        assert_eq!(
            boss_encounter_title("LAGAVULIN_MATRIARCH_BOSS"),
            "Lagavulin Matriarch"
        );
        assert_eq!(boss_encounter_title("THE_KIN_BOSS"), "The Kin");
    }

    #[test]
    fn title_case_matches_python() {
        assert_eq!(title_case("pick card reward"), "Pick Card Reward");
        assert_eq!(title_case("rest option"), "Rest Option");
        assert_eq!(title_case("x2 hello"), "X2 Hello");
    }

    #[test]
    fn intent_keeps_base_only_when_it_differs() {
        let v = intent(&json!({"kind": "Attack", "amount": 6, "effective_amount": 9, "times": 2}));
        assert_eq!(
            v,
            json!({"type": "Attack", "damage": 9, "base_damage": 6, "hits": 2})
        );
        let v = intent(&json!({"kind": "Attack", "amount": 6, "effective_amount": 6}));
        assert_eq!(v, json!({"type": "Attack", "damage": 6}));
        let v = intent(&json!({"kind": "Buff"}));
        assert_eq!(v, json!({"type": "Buff"}));
    }

    #[test]
    fn treasure_reveal_fires_only_on_entry() {
        let before = RoomSnapshot {
            gold: 10,
            relics: vec!["RELIC.A".into()],
            room: Some(json!({"room_type": "monster", "map_coord": [1, 2]})),
        };
        let after = RoomSnapshot {
            gold: 35,
            relics: vec!["RELIC.A".into(), "RELIC.B".into()],
            room: Some(json!({"room_type": "treasure", "map_coord": [1, 3]})),
        };
        let reveal = treasure_reveal(Some(&before), &after, 1).expect("reveal");
        assert_eq!(reveal["gold"], json!(25));
        assert_eq!(reveal["relic"]["id"], json!("B"));
        // Standing in the same chest room again: no reveal.
        assert!(treasure_reveal(Some(&after), &after, 1).is_none());
        assert!(treasure_reveal(None, &after, 1).is_none());
    }
}
