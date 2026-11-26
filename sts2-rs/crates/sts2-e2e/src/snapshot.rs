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

/// The powers C# leaves on a creature when it dies: the ones whose
/// `ShouldPowerBeRemovedAfterOwnerDeath()` is `false`
/// (`Creature.RemoveAllPowersAfterDeath`, `CreatureCmd.Kill`).
const KEPT_AFTER_DEATH: [&str; 6] = [
    "ADAPTABLE",
    "DIE_FOR_YOU",
    "MINION",
    "PAINFUL_STABS",
    "REATTACH",
    "STEAM_ERUPTION",
];

/// GH #1791: a dead enemy shows only the powers C# keeps after death. The
/// engine keeps every power on a corpse (its NN-slot retention policy), so
/// the env showed powers the game has already removed. On by default since
/// a paired eval found it neutral for the d4000 net (IC A10 DEV-1000, 78.5%
/// vs 79.2%, 32 vs 39 discordant seeds); `STS2_E2E_CORPSE_POWERS=keep`
/// restores the engine's form.
static STRIP_CORPSE_POWERS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn strip_corpse_powers() -> bool {
    use std::sync::atomic::Ordering;
    match STRIP_CORPSE_POWERS.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("STS2_E2E_CORPSE_POWERS").map_or(true, |v| v != "keep");
            STRIP_CORPSE_POWERS.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// Overrides `STS2_E2E_CORPSE_POWERS` for this process (tests, bindings).
pub fn set_strip_corpse_powers(on: bool) {
    STRIP_CORPSE_POWERS.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// A dead enemy's powers as the game shows them (see [`strip_corpse_powers`]).
/// An Illusion carrier is left alone: its death is a revive, which the
/// engine already strips for.
fn corpse_powers(m: &Value, powers: &[Value]) -> Vec<Value> {
    let dead = get(m, "is_dead").is_some_and(truthy);
    let name = |p: &Value| {
        get(p, "name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    if !dead || !strip_corpse_powers() || powers.iter().any(|p| name(p) == "ILLUSION") {
        return powers.to_vec();
    }
    powers
        .iter()
        .filter(|p| KEPT_AFTER_DEATH.contains(&name(p).as_str()))
        .cloned()
        .collect()
}
/// GH #1835: a dead enemy's attack intents as the game shows them. C# drops
/// every power but [`KEPT_AFTER_DEATH`] when a creature dies, so its
/// telegraph no longer folds the Strength / Weak / Shrink it died with; the
/// engine's corpse keeps them, and its wire's `effective_amount` did. Each
/// corpse's attack intents are re-previewed on a copy of the fight with the
/// dropped powers taken off it. `None` where nothing changes (flag off, no
/// such corpse), so the common row clones nothing.
#[must_use]
pub fn with_corpse_intents(bv: &sts2_core::CombatState, d: &Value) -> Option<Value> {
    if !strip_corpse_powers() {
        return None;
    }
    let mut fight: Option<sts2_core::CombatState> = None;
    let mut out: Option<Value> = None;
    for (i, m) in list_or_empty(d, "encounter").iter().enumerate() {
        if !get(m, "is_dead").is_some_and(truthy) {
            continue;
        }
        let Some(id) = get(m, "uid")
            .and_then(Value::as_u64)
            .and_then(|u| u32::try_from(u).ok())
            .and_then(std::num::NonZeroU32::new)
        else {
            continue;
        };
        let Some(c) = sts2_core::creature_by_id(&bv.creatures, id) else {
            continue;
        };
        let names: Vec<(sts2_core::PowerUid, &str)> = c
            .powers
            .iter()
            .filter_map(|u| bv.powers.get_by_id(u.0).map(|p| (*u, p.kind.python_name())))
            .collect();
        if names.iter().any(|(_, n)| *n == "ILLUSION") {
            continue;
        }
        let dropped: Vec<sts2_core::PowerUid> = names
            .iter()
            .filter(|(_, n)| !KEPT_AFTER_DEATH.contains(n))
            .map(|(u, _)| *u)
            .collect();
        let intents = list_or_empty(m, "intents");
        if dropped.is_empty()
            || !intents
                .iter()
                .any(|it| get(it, "effective_amount").is_some())
        {
            continue;
        }
        let f = fight.get_or_insert_with(|| bv.clone());
        if let Some(fc) = sts2_core::creature_by_id_mut(&mut f.creatures, id) {
            fc.powers.retain(|u| !dropped.contains(u));
        }
        let o = out.get_or_insert_with(|| d.clone());
        for (k, it) in intents.iter().enumerate() {
            let Some(amount) = get(it, "amount").and_then(Value::as_i64) else {
                continue;
            };
            if get(it, "effective_amount").is_none() {
                continue;
            }
            let eff = sts2_core::combat_loop::preview_damage_to_player(
                f,
                sts2_core::hooks::CreatureRef::Enemy(sts2_core::CreatureUid(id)),
                i32::try_from(amount).unwrap_or(0),
                sts2_core::hooks::ValueProps::MOVE,
                None,
            );
            o["encounter"][i]["intents"][k]["effective_amount"] = json!(eff);
        }
    }
    out
}

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

/// v0.113.0's set: v0.111.0's plus Byrdpip and Wuppy (GH #1535).
const SHOW_COUNTER_RELICS_V0113_ADDED: [&str; 2] = ["BYRDPIP", "WUPPY"];

/// The counter C# shows on `relic_id` for engine counter `c` (v0.111.0
/// `RELIC_COUNTER_DISPLAY_V0111`), or `None` when no counter is shown.
#[must_use]
pub fn display_counter(relic_id: &str, c: i64) -> Option<i64> {
    let later = crate::build::later();
    let shown = SHOW_COUNTER_RELICS_V0111.contains(&relic_id)
        || (later && SHOW_COUNTER_RELICS_V0113_ADDED.contains(&relic_id));
    if !shown {
        return None;
    }
    let m = |n: i64| Some(c.rem_euclid(n));
    match relic_id {
        // PENDULUM: the engine stores TurnsSeen already wrapped at Turns (3)
        // since main's #1143, so the display is the counter itself.
        "GIRYA" | "PUMPKIN_CANDLE" | "VELVET_CHOKER" | "PENDULUM" => Some(c),
        "EMBER_TEA" => Some(c.max(0)),
        // v0.113.0: `CardsVar(6)` (`BrilliantScarf.cs:34`).
        "BRILLIANT_SCARF" => (c < if later { 6 } else { 5 }).then_some(c),
        "BYRDPIP" => Some(c.min(6)),
        "WUPPY" => Some(c),
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

/// The engine counter behind the counter C# shows (`display_counter`'s
/// inverse), where the two differ: the game's own number, as the mod and the
/// headless host send it, read into the engine's. `None` where they agree.
///
/// Sword of Stone counts elites down in the engine (5 to go) and up in C#
/// (`ElitesDefeated`), so a game-side 0 read as is showed 5 (GH #1684).
#[must_use]
pub fn engine_counter(relic_id: &str, shown: i64) -> Option<i64> {
    match relic_id {
        "SWORD_OF_STONE" => Some(5 - shown),
        // C# shows what is left (`5 - CombatsFinished`, `3 - uses`); the
        // engine counts what is done. Read as is, a fresh Wongo's ticket (5)
        // showed no counter at all (Spire Bench corpus, 31 runs).
        "WONGOS_MYSTERY_TICKET" => Some(5 - shown),
        "SILVER_CRUCIBLE" => Some(3 - shown),
        _ => None,
    }
}

/// The engine counter of a relic the game shows *without* a counter, where
/// that is not the pickup value (`None` elsewhere). Wongo's ticket hides its
/// counter once five combats are done (`ShowCounter => DisplayAmount > 0`),
/// which is past its payout: the engine's paid-out mark. Silver Crucible
/// hides it once its three uses are spent. Read as 0, a paid-out ticket
/// showed a fresh 5 for the rest of the run (Spire Bench corpus, GH #1829).
#[must_use]
pub fn engine_counter_unshown(relic_id: &str) -> Option<i64> {
    match relic_id {
        "WONGOS_MYSTERY_TICKET" => Some(-1),
        "SILVER_CRUCIBLE" => Some(3),
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
        Value::Array(
            corpse_powers(m, list_or_empty(m, "powers"))
                .iter()
                .map(power)
                .collect(),
        ),
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

/// The snapshot's six pile keys of `deck` (`state_to_dict(bv)["deck"]`), each
/// card through `card_of`: `draw_unknown` (a multiset of the unpinned draw
/// pile), `draw_top`, `draw_bottom`, `discard`, `exhaust` (multisets) and
/// `play`.
fn piles(deck: &Value, card_of: impl Fn(&Value) -> Value) -> Map<String, Value> {
    let draw = list_or_empty(deck, "draw_pile");
    let idx_list = |k: &str| -> Vec<usize> {
        list_or_empty(deck, k)
            .iter()
            .filter_map(|v| v.as_u64().and_then(|u| usize::try_from(u).ok()))
            .collect()
    };
    let top = idx_list("pinned_top");
    let bot = idx_list("pinned_bottom");
    let pinned: std::collections::BTreeSet<usize> = top.iter().chain(bot.iter()).copied().collect();
    let cards_of =
        |k: &str| -> Vec<Value> { list_or_empty(deck, k).iter().map(&card_of).collect() };
    let mut out = Map::new();
    out.insert(
        "draw_unknown".to_owned(),
        multiset(
            draw.iter()
                .enumerate()
                .filter(|(i, _)| !pinned.contains(i))
                .map(|(_, c)| card_of(c))
                .collect(),
        ),
    );
    let at = |idx: &[usize]| -> Value {
        Value::Array(
            idx.iter()
                .filter_map(|&i| draw.get(i))
                .map(&card_of)
                .collect(),
        )
    };
    out.insert("draw_top".to_owned(), at(&top));
    out.insert("draw_bottom".to_owned(), at(&bot));
    out.insert("discard".to_owned(), multiset(cards_of("discard_pile")));
    out.insert("exhaust".to_owned(), multiset(cards_of("exhaust_pile")));
    out.insert("play".to_owned(), Value::Array(cards_of("play_pile")));
    out
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
    let mut piles = piles(&deck, card);
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
        "draw_unknown": piles.remove("draw_unknown"),
        "draw_top": piles.remove("draw_top"),
        "draw_bottom": piles.remove("draw_bottom"),
        "discard": piles.remove("discard"),
        "exhaust": piles.remove("exhaust"),
        "play": piles.remove("play"),
        "enemies": list_or_empty(d, "encounter").iter().map(enemy).collect::<Vec<_>>(),
        "round": get(d, "round").and_then(as_int).unwrap_or(0),
        "is_elite": get(d, "is_elite").is_some_and(truthy),
        "room": room,
        "is_boss": room == "boss",
        "choice": choice(choice_v),
    })
}

/// `live_snapshot(snap, bv, d_bv)`: a copy of `snap` with the live keys
/// ([`add_live`]). The tokenizer makes it on a row that carries the "chars"
/// event, and only there (GH #1197): a row without the event pays for none
/// of the engine walks.
#[must_use]
pub fn live_snapshot(snap: &Value, bv: &CombatState, d_bv: &Value) -> Value {
    let mut out = snap.clone();
    add_live(&mut out, bv, d_bv);
    out
}

/// The pile keys of `snap` rebuilt with each card's `pile_flags` (a pile
/// card's Sly / Sly this turn / Retain this turn, GH #1197) in its `flags`,
/// as a hand card carries them; a multiset entry splits where the flags
/// differ. A no-op when no pile card has `pile_flags`.
fn live_piles(snap: &mut Value, d_bv: &Value) {
    const PILES: [&str; 4] = ["draw_pile", "discard_pile", "exhaust_pile", "play_pile"];
    let deck = get_null(d_bv, "deck");
    let has = PILES.iter().any(|k| {
        list_or_empty(&deck, k)
            .iter()
            .any(|c| get(c, "pile_flags").is_some_and(truthy))
    });
    if !has {
        return;
    }
    let card_of = |c: &Value| -> Value {
        if !get(c, "pile_flags").is_some_and(truthy) {
            return card(c);
        }
        let mut fl = list_or_empty(c, "flags").to_vec();
        fl.extend(list_or_empty(c, "pile_flags").iter().cloned());
        let mut c2 = c.clone();
        c2["flags"] = Value::Array(fl);
        card(&c2)
    };
    if let Value::Object(m) = snap {
        m.extend(piles(&deck, card_of));
    }
}

/// `add_live(snap, bv, d_bv)`: the live keys of a "chars" row (GH #1197),
/// set in place on `snap`. Per hand card: `dmg_live` (`{"<enemy index>":
/// [damage per hit, hits]}`, [`sts2_wire::emit::card_damage_live`]),
/// `gen_live` (`{"flat", "per_energy", "kills"}` when the play generates
/// damage, plus `"by_x"` (a list by the X spent) for an X card whose damage
/// is not linear in X,
/// [`sts2_wire::emit::card_gen_live`]), `blk_live`
/// ([`sts2_wire::emit::card_block_live`], null = no block) and
/// `star_cost_live` (copied from `d_bv`'s hand card when it has one). The
/// piles carry each card's `pile_flags` in its `flags`.
///
/// Also the top-level `end_live` (`{"block", "dies", "self_loss", "buffer",
/// "remnant"}`, [`sts2_wire::emit::end_turn_live`]) for
/// [`crate::tactical::hp_loss_live`] and [`crate::tactical::dies_before_acting`].
///
/// Only on a "chars" row, through [`live_snapshot`]: [`snapshot_from_parts`],
/// [`choose_outcome_key`] (the merge key) and every other consumer of the
/// honest snapshot stay as they were.
pub fn add_live(snap: &mut Value, bv: &CombatState, d_bv: &Value) {
    let dmg = sts2_wire::emit::card_damage_live(bv);
    let gen = sts2_wire::emit::card_gen_live(bv);
    let blk = sts2_wire::emit::card_block_live(bv);
    let d_hand = get(d_bv, "deck").map_or(&[][..], |d| list_or_empty(d, "hand"));
    let Some(Value::Array(hand)) = snap.get_mut("hand") else {
        return;
    };
    for (i, hc) in hand.iter_mut().enumerate() {
        let Value::Object(m) = hc else { continue };
        let mut dm = Map::new();
        if let Some(pv) = dmg.get(i) {
            for (k, (d, h)) in pv {
                dm.insert(k.to_string(), json!([d, f64::from(*h)]));
            }
        }
        m.insert("dmg_live".to_owned(), Value::Object(dm));
        if let Some(g) = gen.get(i).and_then(Option::as_ref) {
            let row = |r: &std::collections::BTreeMap<usize, f64>| -> Value {
                Value::Object(r.iter().map(|(k, v)| (k.to_string(), json!(v))).collect())
            };
            let mut gl =
                json!({"flat": row(&g.flat), "per_energy": row(&g.per_energy), "kills": g.kills});
            if !g.by_x.is_empty() {
                gl["by_x"] = Value::Array(g.by_x.iter().map(row).collect());
            }
            m.insert("gen_live".to_owned(), gl);
        }
        if let Some(sc) = d_hand.get(i).and_then(|c| get(c, "star_cost_live")) {
            m.insert("star_cost_live".to_owned(), sc.clone());
        }
        m.insert(
            "blk_live".to_owned(),
            blk.get(i)
                .copied()
                .flatten()
                .map_or(Value::Null, |b| json!(b)),
        );
    }
    live_piles(snap, d_bv);
    let end = sts2_wire::emit::end_turn_live(bv);
    if let Value::Object(m) = snap {
        m.insert(
            "end_live".to_owned(),
            json!({
                "block": end.block,
                "dies": end.dies,
                "self_loss": end.self_loss,
                "buffer": end.buffer,
                "remnant": end.remnant,
            }),
        );
    }
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
    let d = with_corpse_intents(bv, &d).unwrap_or(d);
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

    /// GH #1791: with the flag on, a corpse keeps only what C# keeps.
    #[test]
    fn a_corpse_shows_only_the_powers_the_game_keeps() {
        let m = json!({"kind": "X", "is_dead": true, "powers": [
            {"name": "VULNERABLE", "amount": 2}, {"name": "MINION", "amount": 1}]});
        let names = |e: &Value| -> Vec<String> {
            e["powers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["name"].as_str().unwrap().to_owned())
                .collect()
        };
        set_strip_corpse_powers(false);
        assert_eq!(names(&enemy(&m)), ["VULNERABLE", "MINION"]);
        set_strip_corpse_powers(true);
        assert_eq!(names(&enemy(&m)), ["MINION"]);
        let alive =
            json!({"kind": "X", "is_dead": false, "powers": [{"name": "VULNERABLE", "amount": 2}]});
        assert_eq!(names(&enemy(&alive)), ["VULNERABLE"]);
        set_strip_corpse_powers(false);
    }

    /// GH #1835: a corpse's telegraph no longer folds the Weak it died with
    /// (C# drops the power at death); a living enemy's still does.
    #[test]
    fn a_corpses_intent_drops_the_weak_it_died_with() {
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            5,
            "Ironclad",
            &vec!["STRIKE_IRONCLAD".to_owned(); 10],
            70,
            80,
            &[],
            &[],
            "SLIMES_WEAK",
            10,
            None,
            None,
        )
        .expect("scenario");
        let mut d = state_dict(&cs).expect("dict");
        for (k, dead) in [(0, true), (1, false)] {
            let e = &mut d["encounter"][k];
            e["is_dead"] = json!(dead);
            if dead {
                e["hp"] = json!(0);
            }
            e["powers"] = json!([{"name": "WEAK", "amount": 2}]);
            e["intents"] = json!([{"kind": "Attack", "amount": 8}]);
        }
        let cs = sts2_wire::state_from_json(&d).expect("rebuild");
        let d = state_dict(&cs).expect("dict");
        let eff = |d: &Value, k: usize| d["encounter"][k]["intents"][0]["effective_amount"].clone();
        assert_eq!((eff(&d, 0), eff(&d, 1)), (json!(6), json!(6)));
        set_strip_corpse_powers(true);
        let out = with_corpse_intents(&cs, &d).expect("a corpse with a dropped power");
        assert_eq!((eff(&out, 0), eff(&out, 1)), (json!(8), json!(6)));
        set_strip_corpse_powers(false);
        assert!(with_corpse_intents(&cs, &d).is_none());
    }

    /// The env's row snapshot (`env::snapshot_of`) of a fresh v0.111.0 fight
    /// with a ten-copy deck. Mirrors `_live_fight` in
    /// `tests/e2e/test_e2e_tactical.py`.
    fn live_fight(character: &str, card: &str, relic: &str) -> (CombatState, Value) {
        let cs = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            5,
            character,
            &vec![card.to_owned(); 10],
            70,
            80,
            &[relic.to_owned()],
            &[],
            "SLIMES_WEAK",
            10,
            None,
            None,
        )
        .expect("scenario");
        let bv = belief_view(&cs);
        let d = state_dict(&bv).expect("dict");
        let snap = live_snapshot(&crate::env::snapshot_of(&bv, &d, "monster"), &bv, &d);
        (cs, snap)
    }

    /// Twin of `test_live_piles_split_on_pile_flags`: a pile card's
    /// `pile_flags` reach its `flags`, and the multiset entry splits.
    #[test]
    fn live_piles_split() {
        let strike =
            json!({"id": "STRIKE_SILENT", "kind": "card", "upgraded": false, "base_cost": 1});
        let mut sly = strike.clone();
        sly["pile_flags"] = json!(["sly_turn"]);
        let d_bv = json!({"deck": {"draw_pile": [], "discard_pile": [strike, sly, strike],
            "exhaust_pile": [], "play_pile": []}});
        let mut snap = json!({"discard": "honest"});
        live_piles(&mut snap, &d_bv);
        let mut got = snap["discard"].as_array().expect("multiset").clone();
        got.sort_by_key(|e| e[1].as_i64());
        let plain = json!({"id": "STRIKE_SILENT", "kind": "card", "upgraded": false, "base_cost": 1, "cost": 1});
        let mut flagged = plain.clone();
        flagged["flags"] = json!(["sly_turn"]);
        assert_eq!(got, vec![json!([flagged, 1]), json!([plain, 2])]);
        let mut honest = json!({"discard": "honest"});
        live_piles(
            &mut honest,
            &json!({"deck": {"discard_pile": [strike, strike]}}),
        );
        assert_eq!(honest, json!({"discard": "honest"}));
    }

    fn playable(cs: &CombatState) -> Vec<i64> {
        match sts2_wire::emit::legal_actions(cs) {
            Value::Array(a) => a
                .iter()
                .filter(|a| get(a, "action").and_then(Value::as_str) == Some("play"))
                .filter_map(|a| get(a, "card_index").and_then(Value::as_i64))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Twin of `test_bound_live_keys_on_a_real_fight`.
    #[test]
    fn bound_live_keys_on_a_real_fight() {
        let (cs, snap) = live_fight("Ironclad", "FIEND_FIRE", "BURNING_BLOOD");
        let hand = list_or_empty(&snap, "hand");
        let living: Vec<String> = list_or_empty(&snap, "enemies")
            .iter()
            .enumerate()
            .filter(|(_, m)| !get(m, "is_dead").is_some_and(truthy))
            .map(|(j, _)| j.to_string())
            .collect();
        assert!(living.len() >= 2 && hand.len() >= 3);
        #[allow(clippy::cast_precision_loss)]
        let others = (hand.len() - 1) as f64;
        for c in hand {
            let dl = c["dmg_live"].as_object().expect("dmg_live");
            assert_eq!(dl.keys().cloned().collect::<Vec<_>>(), living);
            assert!(dl.values().all(|v| v[1].as_f64() == Some(others)));
            assert!(c.get("gen_live").is_none() && c.get("star_cost_live").is_none());
        }
        let md = crate::tactical::max_damage_live(&snap, &playable(&cs));
        for j in &living {
            let v = &hand[0]["dmg_live"][j];
            let want = v[0].as_f64().expect("dmg") * v[1].as_f64().expect("hits");
            assert_eq!(md[j.parse::<usize>().expect("index")], want);
        }

        let (_, snap) = live_fight("Silent", "BLADE_DANCE", "RING_OF_THE_SNAKE");
        for c in list_or_empty(&snap, "hand") {
            assert_eq!(c["dmg_live"], json!({}));
            let want: Map<String, Value> =
                living.iter().map(|j| (j.clone(), json!(12.0))).collect();
            assert_eq!(c["gen_live"]["flat"], Value::Object(want));
        }

        let (cs, snap) = live_fight("Necrobinder", "END_OF_DAYS", "BOUND_PHYLACTERY");
        let enemies = list_or_empty(&snap, "enemies");
        let doomable: Vec<usize> = enemies
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                !get(m, "is_dead").is_some_and(truthy) && m["hp"].as_i64().is_some_and(|h| h <= 29)
            })
            .map(|(j, _)| j)
            .collect();
        assert!(!doomable.is_empty());
        let md = crate::tactical::max_damage_live(&snap, &playable(&cs));
        for c in list_or_empty(&snap, "hand") {
            assert_eq!(c["gen_live"]["kills"], json!(doomable));
        }
        for j in doomable {
            let m = &enemies[j];
            let ehp = m["hp"].as_f64().expect("hp") + m["block"].as_f64().unwrap_or(0.0);
            assert_eq!(md[j], ehp);
        }

        let (_, snap) = live_fight("Regent", "FALLING_STAR", "DIVINE_RIGHT");
        for c in list_or_empty(&snap, "hand") {
            assert_eq!(c["star_cost_live"], json!(2));
        }

        // Tempest: Cracked Core's Lightning in a 3-slot belt, 3 energy. Only X=3
        // overflows, evoking that Lightning (`gen_live.by_x`, by energy spent).
        let (cs, snap) = live_fight("Defect", "TEMPEST", "CRACKED_CORE");
        let energy = snap["player"]["energy"].as_u64().expect("energy");
        assert_eq!(energy, 3);
        let md = crate::tactical::max_damage_live(&snap, &playable(&cs));
        for c in list_or_empty(&snap, "hand") {
            let by_x = c["gen_live"]["by_x"].as_array().expect("by_x");
            assert_eq!(by_x.len(), 4);
            assert!(by_x[..3].iter().all(|r| r == &json!({})));
            let last = by_x[3].as_object().expect("row");
            assert_eq!(last.keys().cloned().collect::<Vec<_>>(), living);
            for (j, v) in last {
                assert_eq!(
                    md[j.parse::<usize>().expect("index")],
                    v.as_f64().expect("v")
                );
            }
        }
    }

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

    /// Every number C# can show reads back to an engine counter that shows
    /// it again: `engine_counter` inverts `display_counter`.
    #[test]
    fn every_shown_relic_counter_reads_back_to_itself() {
        for id in SHOW_COUNTER_RELICS_V0111 {
            for c in 0..40 {
                let Some(shown) = display_counter(id, c) else {
                    continue;
                };
                let back = engine_counter(id, shown).unwrap_or(shown);
                assert_eq!(
                    display_counter(id, back),
                    Some(shown),
                    "{id}: engine {c} shows {shown}"
                );
            }
        }
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
