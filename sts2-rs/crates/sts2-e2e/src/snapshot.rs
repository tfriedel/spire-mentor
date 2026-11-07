//! The honest combat snapshot: `ppo.combat_ctx.snapshot` (v0.111.0 tables).
//!
//! [`snapshot_from_parts`] turns `state_to_dict(bv)` (`bv` = the belief view,
//! [`belief_view`]) and `bv`'s previews and pending choice into the JSON the
//! student reads, through the same whitelist as Python. [`obs_actions`]
//! re-indexes `choose` selections from the true state's option order into
//! `bv`'s. [`choose_outcome_key`] is the canonical JSON of the snapshot after
//! a `choose` (the env's merge key for choices).
//!
//! Dict key order: Python builds these dicts in insertion order; a
//! `serde_json::Map` here is sorted by key. Canonical JSON (`pyjson::dumps`)
//! sorts keys, so the merge keys are unaffected; a consumer that iterates a
//! dict (the tokenizer) must not rely on insertion order.

use serde_json::{json, Map, Value};
use sts2_core::CombatState;

use crate::pyjson::{self, as_int, get, get_null, get_or, list_or_empty, truthy};

/// `CARD_KEYS`: the card fields a snapshot keeps.
pub const CARD_KEYS: [&str; 12] = [
    "id",
    "kind",
    "upgraded",
    "base_cost",
    "base_damage",
    "base_block",
    "calculated_value",
    "flags",
    "enchantment",
    "enchantment_amount",
    "affliction",
    "affliction_stacks",
];
const PLAYER_KEYS: [&str; 9] = [
    "hp",
    "max_hp",
    "block",
    "energy",
    "max_energy",
    "stars",
    "potion_slots",
    "cards_played_this_turn",
    "bound_card_played_this_turn",
];
const POWER_KEYS: [&str; 4] = ["name", "amount", "display_amount", "counter"];
const INTENT_KEYS: [&str; 4] = ["kind", "amount", "effective_amount", "times"];
/// The fight rooms a snapshot may name.
pub const ROOMS: [&str; 3] = ["monster", "elite", "boss"];

/// v0.111.0 `SHOW_COUNTER_RELICS` (`static_tables_v0111.py`).
const SHOW_COUNTER_RELICS_V0111: [&str; 35] = [
    "BOOK_OF_FIVE_RINGS",
    "BRILLIANT_SCARF",
    "EMBER_TEA",
    "FAKE_HAPPY_FLOWER",
    "FISHING_ROD",
    "GALACTIC_DUST",
    "GIRYA",
    "HAPPY_FLOWER",
    "IRON_CLUB",
    "JOSS_PAPER",
    "KUNAI",
    "KUSARIGAMA",
    "LASTING_CANDY",
    "LETTER_OPENER",
    "METRONOME",
    "NUNCHAKU",
    "ORNAMENTAL_FAN",
    "PAELS_FLESH",
    "PAELS_LEGION",
    "PAELS_TOOTH",
    "PAELS_WING",
    "PENDULUM",
    "PEN_NIB",
    "POCKETWATCH",
    "POLLINOUS_CORE",
    "PUMPKIN_CANDLE",
    "SHURIKEN",
    "SILVER_CRUCIBLE",
    "STONE_CALENDAR",
    "SWORD_OF_STONE",
    "TOY_BOX",
    "TUNING_FORK",
    "VELVET_CHOKER",
    "WINGED_BOOTS",
    "WONGOS_MYSTERY_TICKET",
];

/// The counter C# shows on `relic_id` for engine counter `c` (v0.111.0
/// `RELIC_COUNTER_DISPLAY_V0111`), or `None` when no counter is shown.
#[must_use]
pub fn display_counter(relic_id: &str, c: i64) -> Option<i64> {
    if !SHOW_COUNTER_RELICS_V0111.contains(&relic_id) {
        return None;
    }
    let m = |n: i64| Some(c.rem_euclid(n));
    match relic_id {
        // PENDULUM: the engine stores TurnsSeen already wrapped at Turns (3)
        // since main's #1143, so the display is the counter itself.
        "GIRYA" | "PUMPKIN_CANDLE" | "VELVET_CHOKER" | "PENDULUM" => Some(c),
        "EMBER_TEA" => Some(c.max(0)),
        "BRILLIANT_SCARF" => (c < 5).then_some(c),
        "METRONOME" => (c < 7).then_some(c),
        "PEN_NIB" | "NUNCHAKU" | "TUNING_FORK" | "GALACTIC_DUST" => m(10),
        "HAPPY_FLOWER" | "FISHING_ROD" | "KUNAI" | "SHURIKEN" | "ORNAMENTAL_FAN" | "KUSARIGAMA"
        | "LETTER_OPENER" => m(3),
        "FAKE_HAPPY_FLOWER" | "BOOK_OF_FIVE_RINGS" | "JOSS_PAPER" => m(5),
        "POLLINOUS_CORE" | "IRON_CLUB" => m(4),
        "LASTING_CANDY" => m(2),
        "TOY_BOX" => (c < 15).then(|| c.rem_euclid(3)),
        "POCKETWATCH" => Some(c & 0xFF),
        "SWORD_OF_STONE" => Some(5 - c),
        "SILVER_CRUCIBLE" => ((c & 0xFF) < 3).then(|| 3 - (c & 0xFF)),
        "WONGOS_MYSTERY_TICKET" => (0..5).contains(&c).then(|| 5 - c),
        "PAELS_LEGION" => (c > 0).then_some(c),
        // PAELS_FLESH, STONE_CALENDAR, PAELS_TOOTH, PAELS_WING, WINGED_BOOTS:
        // the displayed number is not in the engine counter.
        _ => None,
    }
}

/// `snapshot._card`: a card dict reduced to `CARD_KEYS` plus its effective
/// `cost`, flags sorted.
#[must_use]
pub fn card(c: &Value) -> Value {
    let mut out = Map::new();
    for k in CARD_KEYS {
        if let Some(v) = get(c, k) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    let cost = match get(c, "cost_override") {
        Some(v) => v.clone(),
        None => get_or(c, "base_cost", json!(0)),
    };
    out.insert("cost".to_owned(), cost);
    if let Some(Value::Array(flags)) = out.get("flags") {
        let mut f = flags.clone();
        f.sort_by(py_cmp);
        out.insert("flags".to_owned(), Value::Array(f));
    }
    Value::Object(out)
}

/// Python's `<` on the values a sort here sees (strings, numbers).
fn py_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Number(x), Value::Number(y)) => x
            .as_f64()
            .partial_cmp(&y.as_f64())
            .unwrap_or(std::cmp::Ordering::Equal),
        _ => pyjson::dumps(a).cmp(&pyjson::dumps(b)),
    }
}

/// `snapshot.multiset`: `[[card, count], ...]` sorted by canonical JSON.
#[must_use]
pub fn multiset(cards: Vec<Value>) -> Value {
    let mut groups: std::collections::BTreeMap<String, (Value, i64)> =
        std::collections::BTreeMap::new();
    for c in cards {
        let k = pyjson::dumps(&c);
        groups.entry(k).or_insert((c, 0)).1 += 1;
    }
    Value::Array(
        groups
            .into_values()
            .map(|(c, n)| Value::Array(vec![c, json!(n)]))
            .collect(),
    )
}

fn power(p: &Value) -> Value {
    let mut out = Map::new();
    for k in POWER_KEYS {
        if let Some(v) = get(p, k) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    Value::Object(out)
}

fn relic(r: &Value) -> Value {
    let id = get_null(r, "id");
    let mut out = Map::new();
    out.insert("id".to_owned(), id.clone());
    let counter = get(r, "counter")
        .filter(|v| truthy(v))
        .and_then(as_int)
        .unwrap_or(0);
    if let Some(shown) = id.as_str().and_then(|s| display_counter(s, counter)) {
        out.insert("counter".to_owned(), json!(shown));
    }
    if get(r, "is_wax").is_some_and(truthy) {
        out.insert("is_wax".to_owned(), json!(true));
    }
    if get(r, "is_melted").is_some_and(truthy) {
        out.insert("is_melted".to_owned(), json!(true));
    }
    Value::Object(out)
}

fn intent(i: &Value) -> Value {
    let mut out = Map::new();
    for k in INTENT_KEYS {
        if let Some(v) = get(i, k) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    Value::Object(out)
}

fn enemy(m: &Value) -> Value {
    let mut out = Map::new();
    out.insert("eid".to_owned(), get_null(m, "uid"));
    out.insert("kind".to_owned(), get_null(m, "kind"));
    out.insert("hp".to_owned(), get_or(m, "hp", json!(0)));
    out.insert("max_hp".to_owned(), get_or(m, "max_hp", json!(1)));
    out.insert("block".to_owned(), get_or(m, "block", json!(0)));
    out.insert(
        "powers".to_owned(),
        Value::Array(list_or_empty(m, "powers").iter().map(power).collect()),
    );
    out.insert(
        "intents".to_owned(),
        Value::Array(list_or_empty(m, "intents").iter().map(intent).collect()),
    );
    out.insert(
        "is_dead".to_owned(),
        json!(get(m, "is_dead").is_some_and(truthy)),
    );
    out.insert(
        "removed_after_death".to_owned(),
        json!(get(m, "removed_after_death").is_some_and(truthy)),
    );
    let stolen = list_or_empty(m, "stolen_cards");
    if !stolen.is_empty() {
        out.insert(
            "stolen".to_owned(),
            Value::Array(
                stolen
                    .iter()
                    .map(|s| {
                        json!({
                            "id": get_null(s, "id"),
                            "upgraded": get(s, "upgraded").is_some_and(truthy),
                        })
                    })
                    .collect(),
            ),
        );
    }
    Value::Object(out)
}

fn choice(pc: Option<&Value>) -> Value {
    let Some(pc) = pc else {
        return Value::Null;
    };
    let mut opts = Vec::new();
    for o in list_or_empty(pc, "cards") {
        let pile = get_or(o, "pile", json!("generated"));
        let mut opt = Map::new();
        opt.insert("id".to_owned(), get_null(o, "id"));
        opt.insert(
            "upgraded".to_owned(),
            json!(get(o, "upgraded").is_some_and(truthy)),
        );
        opt.insert("type".to_owned(), get_null(o, "type"));
        if pile == json!("hand") {
            if let Some(i) = get(o, "index") {
                opt.insert("hand_index".to_owned(), i.clone());
            }
        }
        opt.insert("pile".to_owned(), pile);
        opts.push(Value::Object(opt));
    }
    json!({
        "kind": get_null(pc, "kind"),
        "min_select": get_or(pc, "min_select", json!(0)),
        "max_select": get_or(pc, "max_select", json!(0)),
        "can_skip": get(pc, "can_skip").is_some_and(truthy),
        "options": opts,
    })
}

/// `snapshot_from_parts(d, dmg, blk, choice, room)`: the honest snapshot of
/// `d = state_to_dict(bv)` with `bv`'s damage / block previews and pending
/// choice. No `ehist` (the fight's [`crate::intent_log::IntentLog`] adds it).
///
/// # Panics
///
/// If `room` is not one of [`ROOMS`] (Python raises `ValueError`).
#[must_use]
pub fn snapshot_from_parts(
    d: &Value,
    dmg: &[std::collections::BTreeMap<usize, i32>],
    blk: &[Option<i32>],
    choice_v: Option<&Value>,
    room: &str,
) -> Value {
    assert!(ROOMS.contains(&room), "room {room:?} not in {ROOMS:?}");
    let p = get_null(d, "player");
    let mut player = Map::new();
    for k in PLAYER_KEYS {
        if let Some(v) = get(&p, k) {
            player.insert(k.to_owned(), v.clone());
        }
    }
    player.entry("stars").or_insert(json!(0));
    player.entry("potion_slots").or_insert(json!(0));
    let deck = get_null(d, "deck");
    let mut hand = Vec::new();
    for (i, c) in list_or_empty(&deck, "hand").iter().enumerate() {
        let mut hc = card(c);
        let mut dm = Map::new();
        if let Some(pv) = dmg.get(i) {
            for (k, v) in pv {
                dm.insert(k.to_string(), json!(v));
            }
        }
        if let Value::Object(m) = &mut hc {
            m.insert("dmg".to_owned(), Value::Object(dm));
            m.insert(
                "blk".to_owned(),
                blk.get(i)
                    .copied()
                    .flatten()
                    .map_or(Value::Null, |b| json!(b)),
            );
        }
        hand.push(hc);
    }
    let draw = list_or_empty(&deck, "draw_pile");
    let idx_list = |k: &str| -> Vec<usize> {
        list_or_empty(&deck, k)
            .iter()
            .filter_map(|v| v.as_u64().and_then(|u| usize::try_from(u).ok()))
            .collect()
    };
    let top = idx_list("pinned_top");
    let bot = idx_list("pinned_bottom");
    let pinned: std::collections::BTreeSet<usize> = top.iter().chain(bot.iter()).copied().collect();
    let cards_of = |k: &str| -> Vec<Value> { list_or_empty(&deck, k).iter().map(card).collect() };
    let potions: Vec<Value> = list_or_empty(&p, "potions")
        .iter()
        .filter(|x| truthy(x))
        .map(|x| match x {
            Value::String(_) => x.clone(),
            other => Value::String(pyjson::dumps(other)),
        })
        .collect();
    json!({
        "player": player,
        "powers": list_or_empty(&p, "powers").iter().map(power).collect::<Vec<_>>(),
        "relics": list_or_empty(&p, "relics").iter().map(relic).collect::<Vec<_>>(),
        "potions": potions,
        "hand": hand,
        "draw_unknown": multiset(
            draw.iter().enumerate().filter(|(i, _)| !pinned.contains(i)).map(|(_, c)| card(c)).collect()
        ),
        "draw_top": top.iter().filter_map(|&i| draw.get(i)).map(card).collect::<Vec<_>>(),
        "draw_bottom": bot.iter().filter_map(|&i| draw.get(i)).map(card).collect::<Vec<_>>(),
        "discard": multiset(cards_of("discard_pile")),
        "exhaust": multiset(cards_of("exhaust_pile")),
        "play": cards_of("play_pile"),
        "enemies": list_or_empty(d, "encounter").iter().map(enemy).collect::<Vec<_>>(),
        "round": get(d, "round").and_then(as_int).unwrap_or(0),
        "is_elite": get(d, "is_elite").is_some_and(truthy),
        "room": room,
        "is_boss": room == "boss",
        "choice": choice(choice_v),
    })
}

/// `cs.belief_view(0)`: a clone after `belief_root(0)`.
#[must_use]
pub fn belief_view(cs: &CombatState) -> CombatState {
    let mut out = cs.clone();
    sts2_core::blindfold::belief_root(&mut out, 0);
    out
}

/// `state_to_dict(state)`.
///
/// # Errors
///
/// The emitter's refusal (a broken state).
pub fn state_dict(state: &CombatState) -> Result<Value, String> {
    sts2_wire::emit::combat_state_to_value(state).map_err(|e| e.to_string())
}

/// `honest_snapshot(bv, room)`: [`snapshot_from_parts`] over `bv`'s own reads.
///
/// # Errors
///
/// `state_to_dict` refused the state.
pub fn honest_snapshot(bv: &CombatState, room: &str) -> Result<Value, String> {
    let d = state_dict(bv)?;
    let pc = sts2_wire::emit::pending_choice(bv);
    Ok(snapshot_from_parts(
        &d,
        &sts2_wire::emit::card_damage_preview(bv),
        &sts2_wire::emit::card_block_preview(bv),
        pc.as_ref(),
        room,
    ))
}

/// `choose_outcome_key(c)`: the canonical JSON of the honest snapshot of
/// `c.belief_view(0)` in a `monster` room.
///
/// # Errors
///
/// `state_to_dict` refused the state.
pub fn choose_outcome_key(c: &CombatState) -> Result<String, String> {
    Ok(pyjson::dumps(&honest_snapshot(&belief_view(c), "monster")?))
}

fn opt_key(o: &Value) -> String {
    let pile = get_or(o, "pile", json!("generated"));
    let hand_index = if get(o, "pile") == Some(&json!("hand")) {
        get_null(o, "index")
    } else {
        Value::Null
    };
    pyjson::dumps(&json!({
        "id": get_null(o, "id"),
        "upgraded": get(o, "upgraded").is_some_and(truthy),
        "pile": pile,
        "hand_index": hand_index,
    }))
}

/// `obs_actions(actions, cs_choice, bv_choice)`: the true state's legal
/// actions with every `choose` selection re-indexed into `bv`'s option order
/// by visible content (identical options matched first-unused).
///
/// # Errors
///
/// The two choices list different options.
pub fn obs_actions(
    actions: &[Value],
    cs_choice: Option<&Value>,
    bv_choice: Option<&Value>,
) -> Result<Vec<Value>, String> {
    let (Some(cc), Some(bc)) = (cs_choice, bv_choice) else {
        return Ok(actions.to_vec());
    };
    let cs_keys: Vec<String> = list_or_empty(cc, "cards").iter().map(opt_key).collect();
    let bv_keys: Vec<String> = list_or_empty(bc, "cards").iter().map(opt_key).collect();
    if cs_keys == bv_keys {
        return Ok(actions.to_vec());
    }
    let (mut a, mut b) = (cs_keys.clone(), bv_keys.clone());
    a.sort();
    b.sort();
    if a != b {
        return Err("true and belief choices list different options".to_owned());
    }
    let mut remap = vec![usize::MAX; cs_keys.len()];
    let mut used = vec![false; bv_keys.len()];
    for (i, k) in cs_keys.iter().enumerate() {
        if let Some(j) = (0..bv_keys.len()).find(|&j| !used[j] && &bv_keys[j] == k) {
            remap[i] = j;
            used[j] = true;
        }
    }
    let mut out = Vec::with_capacity(actions.len());
    for a in actions {
        if get(a, "action") == Some(&json!("choose")) {
            let sel: Vec<usize> = list_or_empty(a, "selection")
                .iter()
                .map(|x| {
                    as_int(x)
                        .and_then(|i| usize::try_from(i).ok())
                        .and_then(|i| remap.get(i).copied())
                        .filter(|&j| j != usize::MAX)
                        .ok_or_else(|| format!("choose selection {x} has no belief twin"))
                })
                .collect::<Result<_, _>>()?;
            let mut m = a.as_object().cloned().unwrap_or_default();
            m.insert("index".to_owned(), json!(sel.first().copied().unwrap_or(0)));
            m.insert("selection".to_owned(), json!(sel));
            out.push(Value::Object(m));
        } else {
            out.push(a.clone());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_actions_remaps_choose_by_visible_content() {
        let a = json!({"id": "STRIKE_IRONCLAD", "upgraded": false, "pile": "draw", "index": 3});
        let b = json!({"id": "BASH", "upgraded": true, "pile": "draw", "index": 0});
        let cs = json!({"cards": [a, b]});
        let bv = json!({"cards": [b, a]});
        let acts = vec![
            json!({"action": "choose", "index": 0, "selection": [0]}),
            json!({"action": "choose", "index": 1, "selection": [1]}),
            json!({"action": "choose", "index": 0, "selection": []}),
        ];
        let out = obs_actions(&acts, Some(&cs), Some(&bv)).unwrap();
        assert_eq!(
            out[0],
            json!({"action": "choose", "index": 1, "selection": [1]})
        );
        assert_eq!(
            out[1],
            json!({"action": "choose", "index": 0, "selection": [0]})
        );
        assert_eq!(
            out[2],
            json!({"action": "choose", "index": 0, "selection": []})
        );
        let other = json!({"cards": [b, b]});
        assert!(obs_actions(&acts, Some(&cs), Some(&other)).is_err());
    }

    #[test]
    fn display_counter_follows_v0111_rules() {
        assert_eq!(display_counter("PEN_NIB", 23), Some(3));
        assert_eq!(display_counter("POCKETWATCH", 0x0305), Some(5));
        assert_eq!(display_counter("TOY_BOX", 14), Some(2));
        assert_eq!(display_counter("TOY_BOX", 15), None);
        assert_eq!(display_counter("DIAMOND_DIADEM", 2), None);
        assert_eq!(display_counter("BURNING_BLOOD", 2), None);
        assert_eq!(display_counter("SWORD_OF_STONE", 5), Some(0));
    }
}
