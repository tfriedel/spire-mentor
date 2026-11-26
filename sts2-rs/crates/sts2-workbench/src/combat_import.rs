//! A live-game fight (the SpireMentor `combat_action` board) → the engine wire
//! dict `sts2_wire::state_from_json` rebuilds a `CombatState` from. GH #708.
//!
//! The port of the desktop workbench's two-stage translation, composed into
//! one: `realgame_adapter.to_headless_combat_play` (mod wire → the headless
//! `combat_play` dialect) and `sts2_hint.adapter.translate` (headless →
//! engine wire). The intermediate dialect is not materialised — every rule
//! of both stages is applied here directly to the mod's objects — and
//! `tests/state_import_combat_parity.rs` pins the result against the Python
//! chain's answer for every real capture under
//! `sts2-pycli/tests/fixtures/realgame_adapter/`.
//!
//! The live advisor runs it too (`sts2_rs.combat_wire_from_bridge`, GH #1426),
//! so it must track every field the Python chain gains, co-op seats and game
//! modes included. `sts2-pycli/tests/test_board_import_parity.py` holds the
//! two to the same wire on every board the Python tests build and on a
//! recorded live run; add a field to both, then re-run
//! `scripts/harvest_adapter_boards.py`.
//!
//! Fail-loud vs soft-drop, exactly as the desktop lane (and worth stating,
//! because #708's bar is "name what cannot be reconstructed"): an unknown
//! **monster**, an unknown **orb**, an intent with no type, or a mid-play
//! board the engine refuses are ERRORS — the fight is not rebuilt and the
//! caller says so. An unknown **card**, **relic**, **power** or **potion**
//! is DROPPED from the rebuilt board; the desktop counted those into a
//! tally nobody read, here every drop is a warning the dialog shows.
//!
//! Everything reads the embedded localization (`data.rs`) and the engine's
//! own registries — no Python, no files — so it runs in the wasm.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde_json::{json, Map, Value};
use sts2_wire::state_dict::registry_names_for;

use crate::data;
use crate::display_maps::{MONSTER_TITLE_ALIASES, MONSTER_TITLE_TO_ID};
use crate::localization::py_title;

/// `sts2_hint.errors.UnsupportedStateError`: a `<path>: <reason>` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedState(pub String);

impl std::fmt::Display for UnsupportedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedState {}

fn unsupported<T>(msg: impl Into<String>) -> Result<T, UnsupportedState> {
    Err(UnsupportedState(msg.into()))
}

/// `sts2_hint.schema.WIRE_VERSION`.
const WIRE_VERSION: i64 = 1;

/// `canonical_maps.STATUS_CARD_TITLES`: status / curse titles that travel as
/// the synthetic `status:<Title>` / `curse:<Title>` wire ids.
const STATUS_CARD_TITLES: [&str; 18] = [
    "Burn",
    "Wound",
    "Slimed",
    "Dazed",
    "Void",
    "Shame",
    "Pain",
    "Decay",
    "Doubt",
    "Injury",
    "Normality",
    "Parasite",
    "Regret",
    "Writhe",
    "Clumsy",
    "Curse of the Bell",
    "Necronomicurse",
    "Pride",
];

/// `sts2_hint.adapter._AFFLICTION_WIRE_NAMES`.
const AFFLICTION_WIRE_NAMES: [&str; 9] = [
    "Bound",
    "Devoured",
    "Entangled",
    "Galvanized",
    "Hexed",
    "Ringing",
    "Smog",
    "Weighted",
    "Tainted",
];

/// `canonical_maps.ORB_TITLE_TO_ID`.
const ORB_TITLE_TO_ID: [(&str, &str); 6] = [
    ("Dark", "DARK"),
    ("Frost", "FROST"),
    ("Glass", "GLASS"),
    ("Lightning", "LIGHTNING"),
    ("Plasma", "PLASMA"),
    ("Orb Slot", "EMPTY"),
];

/// `sts2_hint.adapter._POWER_VAR_TO_ENGINE_FIELD`: (engine power id, C#
/// `DynamicVar`) → the engine field holding the same quantity.
const POWER_VAR_TO_ENGINE_FIELD: [((&str, &str), &str); 9] = [
    (("CRIMSON_MANTLE", "SelfDamage"), "counter"),
    (("INFERNO", "SelfDamage"), "counter"),
    (("SKITTISH", "HasGainedBlockThisTurn"), "counter"),
    (("SURROUNDED", "Facing"), "counter"),
    (("TORIC_TOUGHNESS", "Block"), "counter"),
    (("POSSESS_STRENGTH", "Stolen"), "counter"),
    (("POSSESS_SPEED", "Stolen"), "counter"),
    (("THE_BOMB", "Damage"), "counter"),
    // Test Subject's Multi Claw growth (GH #1897, the mod since #1906).
    (("PAINFUL_STABS", "ExtraMultiClawCount"), "counter"),
];

// ── registries and title maps (cached once per process) ─────────────────

fn registry(kind: &str) -> &'static BTreeSet<String> {
    static CACHE: OnceLock<BTreeMap<&'static str, BTreeSet<String>>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        // `card_any` / `power_any`: every build's ids, which the wire parser
        // accepts. `card` is the PPO vocabulary (frozen at v0.111.0) and
        // `power` the v0.107.1 universe; checking a carried id against those
        // dropped every v0.113.0 card and power from the board. The title
        // maps stay pruned against the narrow kinds, as in the Python.
        [
            "card",
            "card_any",
            "relic",
            "power",
            "power_any",
            "potion",
            "monster",
            "orb",
        ]
        .into_iter()
        .map(|k| {
            (
                k,
                registry_names_for(k)
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
            )
        })
        .collect()
    });
    all.get(kind).expect("known registry kind")
}

/// `_harvest_title_map` + `_prune_against_registry`: `<ID>.title` rows of a
/// loc table inverted to `title → ID` (first registration wins), pruned to
/// the ids the engine knows.
fn title_map(loc: &data::LocMap, kind: &str) -> BTreeMap<String, String> {
    let known = registry(kind);
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in loc {
        let Some(id) = key.strip_suffix(".title") else {
            continue;
        };
        if !known.contains(id) {
            continue;
        }
        out.entry(value.clone()).or_insert_with(|| id.to_string());
    }
    out
}

fn relic_title_to_id() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| title_map(&data::data().loc_relics, "relic"))
}

fn potion_title_to_id() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| title_map(&data::data().loc_potions, "potion"))
}

fn power_title_to_id() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        crate::display_maps::power_title_to_id()
            .into_iter()
            .collect()
    })
}

/// `MONSTER_TITLE_TO_ID` + its aliases, and the kinds a raw loc-key leak may
/// resolve to (`_KNOWN_MONSTER_KINDS`: the map's values plus the three
/// Decimillipede segment variants that share one title).
fn monster_title_to_id() -> &'static BTreeMap<&'static str, &'static str> {
    static MAP: OnceLock<BTreeMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        MONSTER_TITLE_TO_ID
            .iter()
            .chain(MONSTER_TITLE_ALIASES.iter())
            .map(|(t, id)| (*t, *id))
            .collect()
    })
}

fn known_monster_kinds() -> &'static BTreeSet<&'static str> {
    static SET: OnceLock<BTreeSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        let mut set: BTreeSet<&'static str> = monster_title_to_id().values().copied().collect();
        set.extend([
            "DecimillipedeSegmentFront",
            "DecimillipedeSegmentMiddle",
            "DecimillipedeSegmentBack",
        ]);
        set
    })
}

/// `sts2_hint.adapter._id_key`: a model id with underscores dropped and
/// upper-cased. C# `Id.Entry` (`SHRINKER_BEETLE`, `BATTLE_FRIEND_V1`) and the
/// engine's CamelCase monster kinds (`ShrinkerBeetle`) meet here (GH #1423).
fn id_key(raw: &str) -> String {
    raw.trim().replace('_', "").to_uppercase()
}

/// `_monster_kind_from_wire`'s table: `id_key(kind)` → registry monster kind.
fn monster_by_id_key() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        registry("monster")
            .iter()
            .map(|k| (id_key(k), k.clone()))
            .collect()
    })
}

/// `_STATUS_TITLE_BY_ID_KEY`: `id_key` of a status/curse card id → its
/// `STATUS_CARD_TITLES` title (`CURSEOFTHEBELL` → `Curse of the Bell`).
fn status_title_by_id(card_id: &str) -> Option<&'static str> {
    static MAP: OnceLock<BTreeMap<String, &'static str>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        STATUS_CARD_TITLES
            .iter()
            .map(|t| (id_key(&t.replace(' ', "")), *t))
            .collect()
    });
    map.get(&id_key(card_id.rsplit('.').next().unwrap_or(card_id)))
        .copied()
}

/// `_potion_id_from_wire`'s table: `id_key(id)` → registry potion id.
fn potion_by_id_key() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        registry("potion")
            .iter()
            .map(|k| (id_key(k), k.clone()))
            .collect()
    })
}

// ── small helpers ────────────────────────────────────────────────────────

fn str_of(v: &Value) -> Option<&str> {
    v.as_str()
}

/// A JSON integer (a bool is NOT an int — `isinstance(v, int) and not bool`).
fn int_of(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

/// The run's master seed: a full `ulong` in v0.111.0 (GH #788), so read as
/// u64 — `int_of`'s i64 view would drop every seed above `i64::MAX`. A
/// negative integer is a 32-bit seed written signed: its `uint` view. Same
/// reading as `state_import::rng_block`.
#[allow(clippy::cast_sign_loss)]
fn run_seed_of(v: Option<&Value>) -> Option<u64> {
    match v {
        Some(Value::Number(n)) => match n.as_u64() {
            Some(u) => Some(u),
            None => n.as_i64().map(|i| u64::from(i as u32)),
        },
        _ => None,
    }
}

fn get<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| !v.is_null())
}

fn arr<'a>(obj: &'a Value, key: &str) -> &'a [Value] {
    obj.get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str.title()` on the words of a SCREAMING_SNAKE stem, joined without
/// spaces: `TAINTED_POWER` → `TaintedPower`.
fn pascal_of_snake(stem: &str) -> String {
    py_title(&stem.replace('_', " ")).replace(' ', "")
}

/// `_normalize_affliction_name`: `"<NAME>.title"` → `"Name"`, else as is.
fn normalize_affliction_name(name: &str) -> String {
    match name.strip_suffix(".title") {
        Some(stem) => pascal_of_snake(stem),
        None => name.to_string(),
    }
}

/// The English half of a bilingual `"English / Other"` title.
fn english_half(s: &str) -> &str {
    s.split_once(" / ").map_or(s, |(eng, _)| eng).trim()
}

/// `_registry_id_from_leaked`: a leaked `<STEM>.title` loc key → a registry
/// id, optionally minus a kind suffix.
fn registry_id_from_leaked(name: &str, kind: &str, suffixes: &[&str]) -> Option<String> {
    let stem = name.strip_suffix(".title").unwrap_or(name).to_uppercase();
    let reg = registry(kind);
    if reg.contains(&stem) {
        return Some(stem);
    }
    for s in suffixes {
        if let Some(bare) = stem.strip_suffix(s) {
            if reg.contains(bare) {
                return Some(bare.to_string());
            }
        }
    }
    None
}

// ── the sub-element translators ──────────────────────────────────────────

/// `_resolve_card_id` for the bridge's bare ids (a `CARD.` prefix is
/// stripped; `_CARD` / `_STATUS` registry suffixes are tried after the bare
/// form). A slash form (`IRONCLAD/STRIKE`) — the headless dialect — is
/// accepted too, character-suffixed candidate first.
fn resolve_card_id(c_id: &str, path: &str) -> Result<String, UnsupportedState> {
    let known = registry("card_any");
    let candidates: Vec<String> = if let Some((prefix, entry)) = c_id.split_once('/') {
        let entry_u = entry.to_uppercase();
        let prefix_u = prefix.to_uppercase();
        let mut c = Vec::new();
        if [
            "IRONCLAD",
            "SILENT",
            "DEFECT",
            "NECROBINDER",
            "REGENT",
            "FOOL",
        ]
        .contains(&prefix_u.as_str())
        {
            c.push(format!("{entry_u}_{prefix_u}"));
        }
        c.push(entry_u.clone());
        c.push(format!("{entry_u}_CARD"));
        c.push(format!("{entry_u}_STATUS"));
        c
    } else {
        let mut bare = c_id.to_uppercase();
        if let Some(rest) = bare.strip_prefix("CARD.") {
            bare = rest.to_string();
        }
        vec![
            bare.clone(),
            format!("{bare}_CARD"),
            format!("{bare}_STATUS"),
        ]
    };
    for cand in &candidates {
        if known.contains(cand) {
            return Ok(cand.clone());
        }
    }
    // Python's `repr` quoting, so the message reads as the desktop adapter's.
    let listed: Vec<String> = candidates.iter().map(|c| format!("'{c}'")).collect();
    unsupported(format!(
        "{path}: card id '{c_id}' doesn't map to any of [{}] in sts2_rs card registry",
        listed.join(", ")
    ))
}

/// The canonical engine registry id for a bridge / headless card id
/// (`sts2_hint.adapter.registry_card_id`): the id a `card_select` names
/// outside any pile resolves the way a pile card's does.
///
/// # Errors
///
/// No candidate is in the registry.
pub fn registry_card_id(c_id: &str, path: &str) -> Result<String, UnsupportedState> {
    resolve_card_id(c_id, path)
}

/// The character a board's cards belong to, as its display name
/// (`"Ironclad"`), for a lane whose run state names none
/// (`sts2_hint.adapter._infer_class_from_cards`): the first card in hand →
/// draw → discard → exhaust whose id carries a character prefix
/// (`IRONCLAD/STRIKE`) or suffix (`STRIKE_IRONCLAD`).
#[must_use]
pub fn infer_character(board: &Value) -> Option<&'static str> {
    const CHARACTERS: [(&str, &str); 6] = [
        ("IRONCLAD", "Ironclad"),
        ("SILENT", "Silent"),
        ("DEFECT", "Defect"),
        ("NECROBINDER", "Necrobinder"),
        ("REGENT", "Regent"),
        ("FOOL", "Fool"),
    ];
    let no_cards: Vec<Value> = Vec::new();
    let pile = |key: &str| -> &Vec<Value> {
        let v = if key == "hand" {
            board.get("hand")
        } else {
            board.get(key).and_then(|p| p.get("cards"))
        };
        v.and_then(Value::as_array).unwrap_or(&no_cards)
    };
    for key in ["hand", "draw_pile", "discard_pile", "exhaust_pile"] {
        for card in pile(key) {
            let Some(cid) = card.get("id").and_then(Value::as_str) else {
                continue;
            };
            let prefix = if let Some((prefix, _)) = cid.split_once('/') {
                prefix.to_uppercase()
            } else {
                let upper = cid.to_uppercase();
                let bare = upper.strip_prefix("CARD.").unwrap_or(&upper);
                bare.rsplit_once('_')
                    .map_or(String::new(), |(_, s)| s.to_owned())
            };
            if let Some((_, name)) = CHARACTERS.iter().find(|(p, _)| *p == prefix) {
                return Some(*name);
            }
        }
    }
    None
}

/// One bridge Card object → a wire `CardSpec` (`realgame_adapter._translate_card`
/// composed with `sts2_hint.adapter._translate_card`).
fn translate_card(
    card: &Value,
    stable_idx: i64,
    hand: bool,
    path: &str,
) -> Result<Value, UnsupportedState> {
    let c_id = get(card, "id")
        .and_then(str_of)
        .ok_or_else(|| UnsupportedState(format!("{path}.id: required key missing")))?;
    let type_raw = get(card, "type").and_then(str_of).unwrap_or("");
    let type_lower = type_raw.to_lowercase();
    // The intermediate dialect's `upgraded`: `bool(upgrade_level) or bool(upgraded)`.
    let upgraded = truthy(card.get("upgrade_level")) || truthy(card.get("upgraded"));

    let mut spec = Map::new();
    if type_lower == "status" || type_lower == "curse" {
        // The id first: the name is localized (GH #1423).
        let eng = status_title_by_id(c_id).unwrap_or_else(|| {
            english_half(
                get(card, "affliction")
                    .and_then(str_of)
                    .or_else(|| get(card, "name").and_then(str_of))
                    .unwrap_or(""),
            )
        });
        let wire_id = if STATUS_CARD_TITLES.contains(&eng) {
            format!("{type_lower}:{eng}")
        } else {
            resolve_card_id(c_id, path)?
        };
        spec.insert("id".into(), json!(wire_id));
        spec.insert("stable_idx".into(), json!(stable_idx));
        spec.insert("kind".into(), json!(type_lower));
        spec.insert("upgraded".into(), json!(upgraded));
    } else {
        spec.insert("id".into(), json!(resolve_card_id(c_id, path)?));
        spec.insert("stable_idx".into(), json!(stable_idx));
        spec.insert("kind".into(), json!("card"));
        spec.insert("upgraded".into(), json!(upgraded));
    }

    // The resolved cost: `effective_cost`, else `cost` (the intermediate
    // dialect's `cost`), emitted as `cost_override` when it is an int.
    let resolved = card
        .get("effective_cost")
        .filter(|v| !v.is_null())
        .or_else(|| card.get("cost"));
    if let Some(v) = int_of(resolved) {
        spec.insert("cost_override".into(), json!(v));
    }
    // The layered energy costs, then their STAR twins (C# `CurrentStarCost` /
    // `GetStarCostThisCombat`): `SetToFreeThisTurn` zeroes both currencies, so
    // without the star pair a Splash'd Regent star card imports free in energy
    // but at its printed star cost, unplayable for a star-less character.
    for key in [
        "base_cost",
        "persistent_cost",
        "played_cost",
        "stable_cost",
        "star_cost",
        "persistent_star_cost",
    ] {
        if let Some(v) = int_of(card.get(key)) {
            spec.insert(key.into(), json!(v));
        }
    }
    // `stats.damage` / `stats.block` → `base_damage` / `base_block`.
    if let Some(stats) = card.get("stats").and_then(Value::as_object) {
        if let Some(d) = int_of(stats.get("damage")) {
            spec.insert("base_damage".into(), json!(d));
        }
        if let Some(b) = int_of(stats.get("block")) {
            spec.insert("base_block".into(), json!(b));
        }
    }
    // Hand-only keyword flags: the static card properties, never the
    // dynamic `playable` (see `realgame_adapter._keywords_for_hand_card`).
    if hand {
        let mut flags: Vec<&str> = Vec::new();
        if truthy(card.get("retain")) {
            flags.push("retain");
        }
        if truthy(card.get("ethereal")) {
            flags.push("ethereal");
        }
        if truthy(card.get("innate")) {
            flags.push("innate");
        }
        if get(card, "unplayable_reason")
            .and_then(str_of)
            .is_some_and(|r| r.contains("HasUnplayableKeyword"))
        {
            flags.push("unplayable");
        }
        if !flags.is_empty() {
            spec.insert("flags".into(), json!(flags));
        }
    }
    if let Some(ench) = get(card, "enchantment")
        .and_then(str_of)
        .filter(|s| !s.is_empty())
    {
        spec.insert("enchantment".into(), json!(english_half(ench)));
        if let Some(a) = int_of(card.get("enchantment_amount")) {
            spec.insert("enchantment_amount".into(), json!(a));
        }
    }
    if let Some(aff) = get(card, "affliction")
        .and_then(str_of)
        .filter(|s| !s.is_empty())
    {
        let eng = normalize_affliction_name(english_half(aff));
        if AFFLICTION_WIRE_NAMES.contains(&eng.as_str()) {
            spec.insert("affliction".into(), json!(eng));
            let stacks = card
                .get("affliction_stacks")
                .filter(|v| !v.is_null())
                .or_else(|| card.get("affliction_amount"));
            if let Some(s) = int_of(stacks) {
                spec.insert("affliction_stacks".into(), json!(s));
            }
        } else if spec["kind"] == "card" {
            return unsupported(format!("{path}: unknown card affliction {aff:?}"));
        }
    }
    if !type_raw.is_empty() {
        spec.insert("type".into(), json!(type_raw));
    }
    // Per-instance combat scratch the mod disclosed (the engine's `VarKey`
    // wire names, e.g. Bolas' `return_if_played_round`), integers only.
    if let Some(vars) = card.get("combat_vars").and_then(Value::as_object) {
        let clean: Map<String, Value> = vars
            .iter()
            .filter_map(|(k, v)| int_of(Some(v)).map(|n| (k.clone(), json!(n))))
            .collect();
        if !clean.is_empty() {
            spec.insert("combat_vars".into(), Value::Object(clean));
        }
    }
    // Extra plays the instance gained this combat (C# `BaseReplayCount`:
    // Soldier's Stew, Hidden Gem); absent means none.
    if let Some(n) = int_of(card.get("base_replay_count")).filter(|n| *n != 0) {
        spec.insert("base_replay_count".into(), json!(n));
    }
    Ok(Value::Object(spec))
}

/// `_translate_relic` (both stages): a registry-valid `id`, else the title
/// map, else a leaked loc key; `counter` / `is_wax` / `is_melted` forwarded.
fn translate_relic(relic: &Value, path: &str) -> Result<Value, UnsupportedState> {
    let mut relic_id: Option<String> = get(relic, "id")
        .and_then(str_of)
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty() && registry("relic").contains(s));
    if relic_id.is_none() {
        let name = get(relic, "name")
            .and_then(str_of)
            .ok_or_else(|| UnsupportedState(format!("{path}.name: required key missing")))?;
        relic_id = relic_title_to_id()
            .get(name)
            .cloned()
            .or_else(|| registry_id_from_leaked(name, "relic", &[]));
        if relic_id.is_none() {
            return unsupported(format!("{path}: unknown relic title {name:?}"));
        }
    }
    let mut out = Map::new();
    out.insert("id".into(), json!(relic_id));
    out.insert(
        "counter".into(),
        int_of(relic.get("counter")).map_or(Value::Null, |c| json!(c)),
    );
    if truthy(relic.get("is_wax")) {
        out.insert("is_wax".into(), json!(true));
    }
    if truthy(relic.get("is_melted")) {
        out.insert("is_melted".into(), json!(true));
    }
    Ok(Value::Object(out))
}

/// `_translate_potion`: the belt slot's potion as a registry id, `None` for
/// an empty slot; an unknown potion is a soft drop (an empty slot).
fn translate_potion(slot: &Value, path: &str, warnings: &mut Vec<String>) -> Value {
    let Some(obj) = slot.as_object() else {
        return Value::Null;
    };
    if let Some(id) = obj
        .get("id")
        .and_then(Value::as_str)
        .and_then(|s| potion_by_id_key().get(&id_key(s)))
    {
        return json!(id);
    }
    let Some(name) = obj.get("name").and_then(Value::as_str) else {
        return Value::Null;
    };
    if let Some(id) = potion_title_to_id().get(name) {
        return json!(id);
    }
    warnings.push(format!(
        "{path}: unknown potion title {name:?} — the slot was left empty"
    ));
    Value::Null
}

/// `_translate_power` (both stages): a registry id, the id minus `_POWER`,
/// the title map, then a leaked loc key; `vars` through the explicit table;
/// `display_amount` as `display`.
fn translate_power(power: &Value, path: &str) -> Result<Value, UnsupportedState> {
    let name = get(power, "name")
        .and_then(str_of)
        .ok_or_else(|| UnsupportedState(format!("{path}.name: required key missing")))?;
    let reg = registry("power_any");
    let raw_id = get(power, "id").and_then(str_of).filter(|s| !s.is_empty());
    let mut pid: Option<String> = match raw_id {
        Some(id) if reg.contains(id) => Some(id.to_string()),
        Some(id)
            if id
                .strip_suffix("_POWER")
                .is_some_and(|bare| reg.contains(bare)) =>
        {
            id.strip_suffix("_POWER").map(str::to_string)
        }
        _ => power_title_to_id().get(name).cloned(),
    };
    if pid.is_none() {
        pid = registry_id_from_leaked(name, "power", &["_POWER"]);
    }
    let Some(pid) = pid else {
        return unsupported(format!("{path}: unknown power name {name:?}"));
    };
    let amount = match power.get("amount") {
        None | Some(Value::Null) => 0,
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        Some(other) => {
            return unsupported(format!("{path}.amount: non-integer power amount {other}"))
        }
    };
    let mut out = Map::new();
    out.insert("name".into(), json!(pid));
    out.insert("amount".into(), json!(amount));
    if let Some(vars) = power.get("vars").and_then(Value::as_object) {
        for (var_name, v) in vars {
            let target = POWER_VAR_TO_ENGINE_FIELD
                .iter()
                .find(|((p, var), _)| *p == pid && *var == var_name.as_str())
                .map(|(_, t)| *t);
            if let (Some(t), Some(n)) = (target, int_of(Some(v))) {
                out.insert(t.into(), json!(n));
            }
        }
    }
    match power.get("display_amount") {
        Some(Value::Number(n)) => {
            if let Some(d) = n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)) {
                out.insert("display".into(), json!(d));
            }
        }
        Some(Value::Bool(b)) => {
            out.insert("display".into(), json!(i64::from(*b)));
        }
        Some(Value::String(s)) => {
            if let Ok(d) = s.trim().parse::<i64>() {
                out.insert("display".into(), json!(d));
            }
        }
        _ => {}
    }
    // The player who applied it (GH #1004), as the mod's net id, which
    // `resolve_power_appliers` turns into a seat once every seat is known.
    // Imitation Learning's seat is the teammate it copies (`player_target`,
    // GH #992); its `applier` is its own holder, so it is never read. A
    // monster's per-player instance (The Insatiable's Sandpit, Aeonglass's
    // Withering Presence) has no player applier; its seat is the player it
    // targets (C# `PowerModel.Target`, the mod's `target`, GH #1813).
    let applier = if pid == "IMITATION_LEARNING" {
        get(power, "player_target")
    } else {
        get(power, "applier").or_else(|| get(power, "target"))
    };
    if let Some(net_id) = applier {
        out.insert("applier".into(), json!(py_str(net_id)));
    }
    Ok(Value::Object(out))
}

/// Python's `str()` of a JSON scalar (a net id: a string or a number).
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        other => other.to_string(),
    }
}

/// `_translate_intent` (both stages): `intent_type` is the kind; `damage` →
/// `amount`, `hits` → `times`.
fn translate_intent(intent: &Value, path: &str) -> Result<Value, UnsupportedState> {
    let kind = get(intent, "intent_type")
        .and_then(str_of)
        .ok_or_else(|| UnsupportedState(format!("{path}.type: required value is null")))?;
    let mut out = Map::new();
    out.insert("kind".into(), json!(kind));
    if let Some(d) = int_of(intent.get("damage")) {
        out.insert("amount".into(), json!(d));
    }
    if let Some(h) = int_of(intent.get("hits")) {
        out.insert("times".into(), json!(h));
    }
    Ok(Value::Object(out))
}

/// `_SPECIMEN_TEMPLATE` + `_LOC_TEMPLATE_VAR`: strip `#C{Count}…` and any
/// `{var}` from a templated title.
fn bare_species_title(title: &str) -> String {
    let mut s = title.to_string();
    // `\s*#\S*\{[^}]+\}.*$`
    if let Some(hash) = s.find('#') {
        let tail = &s[hash..];
        if let Some(open) = tail.find('{') {
            if tail[open..].contains('}') && !tail[..open].contains(char::is_whitespace) {
                s.truncate(hash);
            }
        }
    }
    // `\s*\{[^}]+\}`
    let mut out = String::new();
    let mut rest = s.as_str();
    while let Some(open) = rest.find('{') {
        if let Some(close_rel) = rest[open..].find('}') {
            out.push_str(rest[..open].trim_end());
            rest = &rest[open + close_rel + 1..];
        } else {
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// `re.sub(r"\d+", "{Count}", name)`.
fn fold_digits(name: &str) -> String {
    let mut out = String::new();
    let mut in_digits = false;
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !in_digits {
                out.push_str("{Count}");
                in_digits = true;
            }
        } else {
            in_digits = false;
            out.push(ch);
        }
    }
    out
}

/// `_translate_enemy` (both stages) minus the lookup: the monster kind from
/// the bridge's `id` (`SHRINKER_BEETLE`, `_monster_kind_from_wire`), else from
/// its title (`MONSTER_TITLE_TO_ID`), a `{Count}`-templated title, the bare
/// species title, or a leaked `<ENTRY>.name` loc key. The id comes first
/// because the `name` is in the game's language (GH #1423).
fn resolve_monster_kind(enemy: &Value, path: &str) -> Result<String, UnsupportedState> {
    if let Some(kind) = get(enemy, "id")
        .and_then(str_of)
        .and_then(|id| monster_by_id_key().get(&id_key(id)))
    {
        return Ok(kind.clone());
    }
    let name = get(enemy, "name")
        .and_then(str_of)
        .ok_or_else(|| UnsupportedState(format!("{path}.name: required key missing")))?;
    let map = monster_title_to_id();
    if let Some(id) = map.get(name) {
        return Ok((*id).to_string());
    }
    if let Some(id) = map.get(fold_digits(name).as_str()) {
        return Ok((*id).to_string());
    }
    // `_bare_species_title_to_id`: the templated entries stripped of their
    // template, first registration wins.
    let bare_wanted = name;
    for (title, id) in MONSTER_TITLE_TO_ID {
        let bare = bare_species_title(title);
        if !bare.is_empty() && bare != *title && bare == bare_wanted {
            return Ok((*id).to_string());
        }
    }
    if let Some(entry) = name.strip_suffix(".name") {
        if !entry.is_empty()
            && entry
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        {
            let parts: Vec<&str> = entry.split('_').collect();
            for k in (1..=parts.len()).rev() {
                let candidate: String = parts[..k].iter().map(|p| py_capitalize(p)).collect();
                if known_monster_kinds().contains(candidate.as_str()) {
                    return Ok(candidate);
                }
            }
        }
    }
    unsupported(format!("{path}: unknown monster name {name:?}"))
}

/// `str.capitalize()`.
fn py_capitalize(s: &str) -> String {
    crate::localization::py_capitalize(s)
}

fn translate_enemy(
    enemy: &Value,
    idx: usize,
    path: &str,
    warnings: &mut Vec<String>,
) -> Result<Value, UnsupportedState> {
    let kind = resolve_monster_kind(enemy, path)?;
    let intents = arr(enemy, "intents")
        .iter()
        .enumerate()
        .map(|(i, it)| translate_intent(it, &format!("{path}.intents[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut powers = Vec::new();
    for (i, p) in arr(enemy, "powers").iter().enumerate() {
        let ppath = format!("{path}.powers[{i}]");
        match translate_power(p, &ppath) {
            Ok(v) => powers.push(v),
            Err(e) => warnings.push(format!(
                "dropped an enemy power the engine does not model ({e})"
            )),
        }
    }
    let hp = int_of(enemy.get("hp"))
        .ok_or_else(|| UnsupportedState(format!("{path}.hp: required key missing")))?;
    let max_hp = int_of(enemy.get("max_hp"))
        .ok_or_else(|| UnsupportedState(format!("{path}.max_hp: required key missing")))?;
    let mut out = Map::new();
    out.insert("kind".into(), json!(kind));
    out.insert("uid".into(), json!(idx as i64 + 1));
    out.insert("hp".into(), json!(hp));
    out.insert("max_hp".into(), json!(max_hp));
    out.insert(
        "block".into(),
        json!(int_of(enemy.get("block")).unwrap_or(0)),
    );
    out.insert("powers".into(), Value::Array(powers));
    out.insert("intents".into(), Value::Array(intents));
    out.insert("intent_history".into(), json!([]));
    // `is_dead = not is_alive` (default alive).
    let alive = match enemy.get("is_alive") {
        None | Some(Value::Null) => true,
        Some(v) => truthy(Some(v)),
    };
    out.insert("is_dead".into(), json!(!alive));
    out.insert("is_escaped".into(), json!(false));
    let move_state = get(enemy, "move_state")
        .filter(|v| truthy(Some(v)))
        .or_else(|| get(enemy, "next_move_id").filter(|v| truthy(Some(v))))
        .cloned()
        .unwrap_or(Value::Null);
    out.insert("move_state".into(), move_state);
    if let Some(s) = get(enemy, "fabricator_last_spawned")
        .and_then(str_of)
        .filter(|s| !s.is_empty())
    {
        out.insert("fabricator_last_spawned".into(), json!(s));
    }
    if let Some(c) = int_of(enemy.get("curse_of_knowledge_counter")) {
        out.insert("curse_of_knowledge_counter".into(), json!(c));
    }
    // The move machine's state (#1181): forwarded verbatim, as the Python
    // adapters do.
    if let Some(ai) = get(enemy, "ai").filter(|v| v.is_object()) {
        out.insert("ai".into(), ai.clone());
    }
    Ok(Value::Object(out))
}

/// `_orb_type_stem` + `_translate_orb`: the `id` (`LIGHTNING_ORB` or
/// `LightningOrb`) wins; the rendered `name` is in the game's language, so it
/// is only the fallback (GH #1423).
fn translate_orb(orb: &Value, path: &str) -> Result<Value, UnsupportedState> {
    let by_id = get(orb, "id")
        .and_then(str_of)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|raw| {
            let stem = raw
                .strip_suffix("_ORB")
                .or_else(|| raw.strip_suffix("Orb"))
                .unwrap_or(raw);
            let key = id_key(stem);
            ORB_TITLE_TO_ID.iter().find(|(_, id)| id_key(id) == key)
        });
    let entry = by_id.or_else(|| {
        let name = get(orb, "name").and_then(str_of)?;
        ORB_TITLE_TO_ID.iter().find(|(t, _)| *t == name)
    });
    let Some((_, id)) = entry else {
        let shown = get(orb, "name")
            .and_then(str_of)
            .filter(|s| !s.trim().is_empty())
            .or_else(|| get(orb, "id").and_then(str_of))
            .unwrap_or("");
        return unsupported(format!("{path}: unknown orb type {shown:?}"));
    };
    Ok(json!({
        "type": id,
        "passive": int_of(orb.get("passive_value")).unwrap_or(0),
        "evoke": int_of(orb.get("evoke_value")).unwrap_or(0),
    }))
}

/// `_translate_osty` (both stages): Osty by pet id, the Necrobinder's
/// explicit "no Osty" sentinel, a dead Osty keeping its `max_hp`.
fn translate_osty(pets: &[Value], character_id: &str, warnings: &mut Vec<String>) -> Option<Value> {
    translate_osty_besides(pets, character_id, &[], warnings)
}

/// [`translate_osty`], not warning about the pet ids in `modelled` (the
/// seated player's Wuppy, which [`translate_wuppy`] carries over).
fn translate_osty_besides(
    pets: &[Value],
    character_id: &str,
    modelled: &[&str],
    warnings: &mut Vec<String>,
) -> Option<Value> {
    let osty = pets.iter().find(|p| {
        get(p, "id")
            .and_then(str_of)
            .is_some_and(|id| id.to_uppercase() == "OSTY")
    });
    for p in pets {
        let id = get(p, "id").and_then(str_of).unwrap_or("");
        if id.to_uppercase() != "OSTY" && !modelled.contains(&id.to_uppercase().as_str()) {
            let name = get(p, "name").and_then(str_of).unwrap_or(id);
            warnings.push(format!(
                "the pet {name:?} has no engine representation and was left out of the fight"
            ));
        }
    }
    let Some(osty) = osty else {
        if character_id.eq_ignore_ascii_case("NECROBINDER") {
            return Some(json!({"alive": false, "hp": 0, "max_hp": 0, "block": 0}));
        }
        return None;
    };
    let alive = truthy(osty.get("is_alive"));
    let max_hp = int_of(osty.get("max_hp")).unwrap_or(0);
    let block = int_of(osty.get("block")).unwrap_or(0);
    if !alive {
        return Some(json!({"alive": false, "hp": 0, "max_hp": max_hp, "block": block}));
    }
    Some(json!({
        "alive": true,
        "hp": int_of(osty.get("hp")).unwrap_or(0),
        "max_hp": max_hp,
        "block": block,
    }))
}

/// The v0.113.0 Wuppy pet (`WUPPY` in the board's `pets`) as the wire's
/// `$.player.wuppy`, in the Osty's shape: a dead Wuppy stays in the fight
/// (`WuppysLoyaltyPower.ShouldCreatureBeRemovedFromCombatAfterDeath`), so it
/// keeps its `max_hp` and only `hp` is forced to 0. `None` without one.
fn translate_wuppy(pets: &[Value]) -> Option<Value> {
    let wuppy = pets.iter().find(|p| {
        get(p, "id")
            .and_then(str_of)
            .is_some_and(|id| id.to_uppercase() == "WUPPY")
    })?;
    let alive = truthy(wuppy.get("is_alive"));
    let max_hp = int_of(wuppy.get("max_hp")).unwrap_or(0);
    let block = int_of(wuppy.get("block")).unwrap_or(0);
    let hp = if alive {
        int_of(wuppy.get("hp")).unwrap_or(0)
    } else {
        0
    };
    Some(json!({"alive": alive, "hp": hp, "max_hp": max_hp, "block": block}))
}

/// A pile's cards translated with sequential `stable_idx`, soft-dropping
/// what the registry does not know (each drop a warning).
fn translate_pile(
    cards: &[Value],
    next_idx: &mut i64,
    hand: bool,
    path: &str,
    warnings: &mut Vec<String>,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(cards.len());
    for (i, card) in cards.iter().enumerate() {
        let idx = *next_idx;
        *next_idx += 1;
        match translate_card(card, idx, hand, &format!("{path}[{i}]")) {
            Ok(v) => out.push(v),
            Err(e) => {
                let name = get(card, "name")
                    .and_then(str_of)
                    .or_else(|| get(card, "id").and_then(str_of))
                    .unwrap_or("?");
                warnings.push(format!("dropped the card {name:?} from {path} ({e})"));
            }
        }
    }
    out
}

fn pile_cards<'a>(board: &'a Value, key: &str) -> &'a [Value] {
    board
        .get(key)
        .and_then(|p| p.get("cards"))
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// `_copy_seat_history`: a seat's History counts a rebuilt board cannot redo
/// (GH #989, #812). Zero is omitted, except for the two turn counts, whose
/// presence says the producer disclosed them (the live mirror lets reality
/// win on them). `truthy_only` is the ally leg, whose first stage copies a
/// count only when it is truthy.
fn copy_seat_history(src: &Value, out: &mut Map<String, Value>, truthy_only: bool) {
    for key in [
        "unblocked_hits_this_combat",
        "cards_exhausted_this_combat",
        "cards_exhausted_this_turn",
        "card_block_gains_this_turn",
        "extra_turns_taken_this_combat",
        "cards_drawn_this_combat",
        "cards_generated_this_combat",
        "discards_this_turn",
    ] {
        let Some(n) = int_of(src.get(key)) else {
            continue;
        };
        if truthy_only && n == 0 {
            continue;
        }
        let turn_key = matches!(
            key,
            "cards_exhausted_this_turn" | "card_block_gains_this_turn"
        );
        if n > 0 || turn_key {
            out.insert(key.into(), json!(n.max(0)));
        }
    }
    if let Some(hits) = src.get("powered_hits_this_turn").and_then(Value::as_array) {
        let rows: Vec<Value> = hits
            .iter()
            .filter(|h| h.is_object() && truthy(h.get("hits")))
            .map(|h| json!({"enemy": py_int(h.get("enemy")), "hits": py_int(h.get("hits"))}))
            .collect();
        if !rows.is_empty() {
            out.insert("powered_hits_this_turn".into(), Value::Array(rows));
        }
    }
}

/// Python's `int()` of a JSON number (a float truncates toward zero).
#[allow(clippy::cast_possible_truncation)]
fn py_int(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

/// `cards_played_this_turn`: the five per-type buckets, clamped at 0.
fn played_buckets(v: Option<&Value>) -> Option<Value> {
    let cppt = v.and_then(Value::as_array)?;
    (cppt.len() == 5).then(|| {
        json!(cppt
            .iter()
            .map(|n| py_int(Some(n)).max(0))
            .collect::<Vec<_>>())
    })
}

/// An orb belt's capacity: `orb_capacity`, else the belt's length — and
/// none at all when the mod sent the key as something other than an int.
fn orb_slots_of(seat: &Value, n_orbs: usize) -> Option<i64> {
    match seat.get("orb_capacity") {
        None => Some(n_orbs as i64),
        Some(v) => int_of(Some(v)),
    }
}

/// The mod's character id (`IRONCLAD`) → the engine class (`ironclad`).
fn class_of_character_id(id: &str) -> Option<&'static str> {
    match id.to_uppercase().as_str() {
        "IRONCLAD" => Some("ironclad"),
        "SILENT" => Some("silent"),
        "DEFECT" => Some("defect"),
        "NECROBINDER" => Some("necrobinder"),
        "REGENT" => Some("regent"),
        "FOOL" => Some("fool"),
        _ => None,
    }
}

/// A teammate (`$.allies[i]`, PROTOCOL.md): HP and block, plus — when the
/// mod sent their hand — the whole seat the engine plays out
/// (`sts2_core::coop`). Their cards are numbered after ours, so no
/// `stable_idx` collides. Returns the ally and its net id.
fn translate_ally(
    ally: &Value,
    i: usize,
    next_idx: &mut i64,
    warnings: &mut Vec<String>,
) -> Result<(Value, Option<String>), UnsupportedState> {
    let path = format!("$.allies[{i}]");
    let mut out = Map::new();
    out.insert("hp".into(), json!(py_int(get(ally, "hp"))));
    out.insert("max_hp".into(), json!(py_int(get(ally, "max_hp"))));
    out.insert("block".into(), json!(py_int(get(ally, "block"))));
    let net_id = get(ally, "net_id").map(py_str);
    let Some(hand_raw) = ally.get("hand").and_then(Value::as_array) else {
        return Ok((Value::Object(out), net_id));
    };
    let character = get(ally, "character").and_then(str_of).unwrap_or("");
    if let Some(class) = class_of_character_id(character) {
        out.insert("class".into(), json!(class));
    }
    out.insert("energy".into(), json!(py_int(get(ally, "energy"))));
    out.insert("max_energy".into(), json!(py_int(get(ally, "max_energy"))));
    let mut powers = Vec::new();
    for (j, p) in arr(ally, "powers").iter().enumerate() {
        match translate_power(p, &format!("{path}.powers[{j}]")) {
            Ok(v) => powers.push(v),
            Err(e) => warnings.push(format!(
                "dropped a teammate power the engine does not model ({e})"
            )),
        }
    }
    out.insert("powers".into(), Value::Array(powers));
    let hand = translate_pile(hand_raw, next_idx, true, &format!("{path}.hand"), warnings);
    out.insert("hand".into(), Value::Array(hand));
    out.insert("ended_turn".into(), json!(truthy(ally.get("ended_turn"))));
    // The rest of the seat (newer mods): what a recorded fight is replayed
    // with as a whole co-op fight.
    for key in ["draw_pile", "discard_pile", "exhaust_pile", "play_pile"] {
        if let Some(cards) = ally
            .get(key)
            .and_then(|p| p.get("cards"))
            .and_then(Value::as_array)
        {
            let pile = translate_pile(cards, next_idx, false, &format!("{path}.{key}"), warnings);
            out.insert(key.into(), Value::Array(pile));
        }
    }
    if let Some(relics_raw) = ally.get("relics").and_then(Value::as_array) {
        let mut relics = Vec::new();
        for (j, r) in relics_raw.iter().enumerate() {
            match translate_relic(r, &format!("{path}.relics[{j}]")) {
                Ok(v) => relics.push(v),
                Err(e) => warnings.push(format!("dropped a teammate relic ({e})")),
            }
        }
        out.insert("relics".into(), Value::Array(relics));
    }
    if let Some(stars) = int_of(ally.get("stars")) {
        out.insert("stars".into(), json!(stars));
    }
    if ally.get("hp_lost_this_turn") == Some(&Value::Bool(true)) {
        out.insert("hp_lost_this_turn".into(), json!(true));
    }
    copy_seat_history(ally, &mut out, true);
    if let Some(b) = played_buckets(
        ally.get("cards_played_this_turn")
            .filter(|v| truthy(Some(v))),
    ) {
        out.insert("cards_played_this_turn".into(), b);
    }
    if let Some(orbs_raw) = ally.get("orbs").and_then(Value::as_array) {
        let orbs = orbs_raw
            .iter()
            .enumerate()
            .map(|(j, o)| translate_orb(o, &format!("{path}.orbs[{j}]")))
            .collect::<Result<Vec<_>, _>>()?;
        out.insert("orbs".into(), Value::Array(orbs));
        if let Some(slots) = orb_slots_of(ally, orbs_raw.len()) {
            out.insert("orb_slots".into(), json!(slots));
        }
    }
    // Their Osty (GH #1017), only when the mod sent their pets.
    if let Some(pets) = ally.get("pets").and_then(Value::as_array) {
        if let Some(osty) = translate_osty(pets, character, warnings) {
            out.insert("osty".into(), osty);
        }
    }
    if let Some(belt) = ally.get("potions").and_then(Value::as_array) {
        let potions: Vec<Value> = belt
            .iter()
            .enumerate()
            .map(|(j, p)| translate_potion(p, &format!("{path}.potions[{j}]"), warnings))
            .collect();
        out.insert("potions".into(), Value::Array(potions));
    }
    Ok((Value::Object(out), net_id))
}

/// `_is_whole_seat`: a teammate the engine can play for a whole fight — a
/// hand, the piles it draws and reshuffles from, and its relics.
fn is_whole_seat(ally: &Value) -> bool {
    ["hand", "draw_pile", "discard_pile", "relics"]
        .iter()
        .all(|k| ally.get(*k).is_some_and(Value::is_array))
}

/// `_resolve_power_appliers`: every power's `applier` net id → its
/// `applier_seat` (0 us, `i + 1` for `$.allies[i]`), GH #1004. An applier the
/// board does not name, or any applier solo, is dropped: the engine then
/// reads the power as ours.
fn resolve_power_appliers(out: &mut Value, seats: &BTreeMap<String, i64>) {
    let fix = |powers: Option<&mut Value>| {
        for p in powers.and_then(Value::as_array_mut).into_iter().flatten() {
            let Some(obj) = p.as_object_mut() else {
                continue;
            };
            if let Some(Value::String(net_id)) = obj.remove("applier") {
                if seats.len() > 1 {
                    if let Some(seat) = seats.get(&net_id) {
                        obj.insert("applier_seat".into(), json!(seat));
                    }
                }
            }
        }
    };
    fix(out.pointer_mut("/player/powers"));
    for key in ["encounter", "allies"] {
        if let Some(list) = out.get_mut(key).and_then(Value::as_array_mut) {
            for entry in list {
                fix(entry.get_mut("powers"));
            }
        }
    }
}

/// `_BLITZ_FIGHT_ASCENSION`: a Blitz level (0..5) → the ascension a fight
/// thresholds (C# `AscensionHelper.GetLevelForBlitz`; only the two enemy
/// levels act in a fight). Twin of `sts2_core::ascension::combat_ascension`.
const BLITZ_FIGHT_ASCENSION: [u8; 6] = [0, 0, 0, 8, 9, 9];

/// `$.game_version` for the wire: the build, normalized (a hotfix is its
/// base build), or `None` for v0.107.1 and a board that names no build.
fn wire_game_version(board: &Value) -> Result<Option<&'static str>, UnsupportedState> {
    use sts2_core::game_version::GameVersion;
    match board.get("game_version") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => match s.parse::<GameVersion>() {
            Ok(GameVersion::V0_107_1) => Ok(None),
            Ok(v) => Ok(Some(v.as_str())),
            Err(e) => unsupported(format!("$.game_version: {e}")),
        },
        Some(other) => unsupported(format!("$.game_version: must be a string, got {other}")),
    }
}

/// The engine wire dict for a SpireMentor `combat_action` board, plus the
/// warnings for everything dropped on the way.
///
/// `character` is the display name (`"Ironclad"`) and `ascension` the run's
/// — both from the paste's `run_state`, exactly what the desktop lane passes
/// (`combat_id` 0 and search `seed` 0, as there).
///
/// # Errors
///
/// An unknown monster or orb, an intent with no type, a card list that is
/// not a list — anything the desktop adapter would have raised
/// `UnsupportedStateError` for.
pub fn combat_wire_from_bridge(
    board: &Value,
    character: &str,
    ascension: u8,
) -> Result<(Value, Vec<String>), UnsupportedState> {
    if !board.is_object() {
        return unsupported("$: the fight is not a JSON object");
    }
    let mut warnings: Vec<String> = Vec::new();
    let empty = json!({});
    let player = board
        .get("player")
        .filter(|v| v.is_object())
        .unwrap_or(&empty);
    let run_state = board
        .get("run_state")
        .filter(|v| v.is_object())
        .unwrap_or(&empty);

    let class = match character {
        "Ironclad" => "ironclad",
        "Silent" => "silent",
        "Defect" => "defect",
        "Necrobinder" => "necrobinder",
        "Regent" => "regent",
        "Fool" => "fool",
        other => return unsupported(format!("$.player.class: unknown character {other:?}")),
    };

    // Piles, `stable_idx` sequential across hand → draw → discard → exhaust →
    // play → master deck (a hand card's index is its hand position).
    let mut next_idx = 0i64;
    let hand = translate_pile(
        arr(board, "hand"),
        &mut next_idx,
        true,
        "$.hand",
        &mut warnings,
    );
    let draw = translate_pile(
        pile_cards(board, "draw_pile"),
        &mut next_idx,
        false,
        "$.draw_pile",
        &mut warnings,
    );
    let discard = translate_pile(
        pile_cards(board, "discard_pile"),
        &mut next_idx,
        false,
        "$.discard_pile",
        &mut warnings,
    );
    let exhaust = translate_pile(
        pile_cards(board, "exhaust_pile"),
        &mut next_idx,
        false,
        "$.exhaust_pile",
        &mut warnings,
    );
    let play = translate_pile(
        pile_cards(board, "play_pile"),
        &mut next_idx,
        false,
        "$.play_pile",
        &mut warnings,
    );
    let master_deck = translate_pile(
        arr(run_state, "master_deck"),
        &mut next_idx,
        false,
        "$.master_deck",
        &mut warnings,
    );

    // Enemies (hard failures), their powers soft.
    let mut encounter = Vec::new();
    for (i, e) in arr(board, "enemies").iter().enumerate() {
        encounter.push(translate_enemy(
            e,
            i,
            &format!("$.enemies[{i}]"),
            &mut warnings,
        )?);
    }

    // Relics (soft), potions (soft), player powers (soft).
    let mut relics = Vec::new();
    for (i, r) in arr(run_state, "relics").iter().enumerate() {
        match translate_relic(r, &format!("$.player.relics[{i}]")) {
            Ok(v) => relics.push(v),
            Err(e) => {
                let name = get(r, "name")
                    .and_then(str_of)
                    .or_else(|| get(r, "id").and_then(str_of))
                    .unwrap_or("?");
                warnings.push(format!("dropped the relic {name:?} ({e})"));
            }
        }
    }
    let potions: Vec<Value> = arr(board, "potions")
        .iter()
        .enumerate()
        .map(|(i, p)| translate_potion(p, &format!("$.player.potions[{i}]"), &mut warnings))
        .collect();
    let mut powers = Vec::new();
    for (i, p) in arr(player, "powers").iter().enumerate() {
        match translate_power(p, &format!("$.player_powers[{i}]")) {
            Ok(v) => powers.push(v),
            Err(e) => warnings.push(format!(
                "dropped a player power the engine does not model ({e})"
            )),
        }
    }

    let round = int_of(board.get("round")).unwrap_or(1);
    let energy = int_of(player.get("energy")).unwrap_or(0);
    let max_energy = int_of(player.get("max_energy")).unwrap_or(energy);
    let hp = int_of(player.get("hp"))
        .ok_or_else(|| UnsupportedState("$.player.hp: required value is null".into()))?;
    let max_hp = int_of(player.get("max_hp"))
        .ok_or_else(|| UnsupportedState("$.player.max_hp: required value is null".into()))?;

    let mut player_out = Map::new();
    player_out.insert("class".into(), json!(class));
    player_out.insert("hp".into(), json!(hp));
    player_out.insert("max_hp".into(), json!(max_hp));
    player_out.insert(
        "block".into(),
        json!(int_of(player.get("block")).unwrap_or(0)),
    );
    player_out.insert("energy".into(), json!(energy));
    player_out.insert("max_energy".into(), json!(max_energy));
    player_out.insert("energy_regen_per_turn".into(), json!(max_energy));
    player_out.insert("relics".into(), Value::Array(relics));
    player_out.insert("powers".into(), Value::Array(powers));
    player_out.insert("potions".into(), Value::Array(potions));
    // Belt capacity — `realgame_adapter` copies `run_state.potion_slot_count`
    // to `player.potion_slots`, which `translate` forwards when it is a
    // positive int (GH #893).
    if let Some(slots) = int_of(run_state.get("potion_slot_count")).filter(|n| *n > 0) {
        player_out.insert("potion_slots".into(), json!(slots));
    }
    if let Some(gold) = int_of(run_state.get("gold")) {
        player_out.insert("gold".into(), json!(gold));
    }
    // An EMPTY belt is still a belt: `orbs` present (even `[]`) → both keys.
    if let Some(orbs_raw) = board.get("orbs").and_then(Value::as_array) {
        let orbs = orbs_raw
            .iter()
            .enumerate()
            .map(|(i, o)| translate_orb(o, &format!("$.orbs[{i}]")))
            .collect::<Result<Vec<_>, _>>()?;
        player_out.insert("orbs".into(), Value::Array(orbs));
        if let Some(slots) = orb_slots_of(board, orbs_raw.len()) {
            player_out.insert("orb_slots".into(), json!(slots));
        }
    }
    if let Some(stars) = int_of(player.get("stars")) {
        player_out.insert("stars".into(), json!(stars));
    }
    let character_id = get(run_state, "character").and_then(str_of).unwrap_or("");
    if let Some(osty) =
        translate_osty_besides(arr(board, "pets"), character_id, &["WUPPY"], &mut warnings)
    {
        player_out.insert("osty".into(), osty);
    }
    if let Some(wuppy) = translate_wuppy(arr(board, "pets")) {
        player_out.insert("wuppy".into(), wuppy);
    }
    if let Some(cppt) = player
        .get("cards_played_this_turn")
        .and_then(Value::as_array)
    {
        if cppt.len() == 5 {
            let mut buckets: Vec<i64> = cppt
                .iter()
                .map(|n| int_of(Some(n)).unwrap_or(0).max(0))
                .collect();
            // The game counts a play when it starts, the engine when it
            // finishes: a card still resolving (the play pile) is not
            // counted yet (`adapter.py` twin).
            for c in pile_cards(board, "play_pile") {
                let k = match c.get("type").and_then(Value::as_str) {
                    Some("Attack") => 0,
                    Some("Skill") => 1,
                    Some("Power") => 2,
                    Some("Curse") => 3,
                    Some("Status") => 4,
                    _ => continue,
                };
                buckets[k] = (buckets[k] - 1).max(0);
            }
            player_out.insert("cards_played_this_turn".into(), json!(buckets));
        }
    }
    if player.get("bound_card_played_this_turn") == Some(&Value::Bool(true)) {
        player_out.insert("bound_card_played_this_turn".into(), json!(true));
    }
    // Spite's "lost HP this turn", then the History counts (GH #989).
    if player.get("hp_lost_this_turn") == Some(&Value::Bool(true)) {
        player_out.insert("hp_lost_this_turn".into(), json!(true));
    }
    copy_seat_history(player, &mut player_out, false);

    let mut deck = Map::new();
    deck.insert("draw_pile".into(), Value::Array(draw));
    deck.insert("discard_pile".into(), Value::Array(discard));
    deck.insert("hand".into(), Value::Array(hand));
    deck.insert("exhaust_pile".into(), Value::Array(exhaust));
    deck.insert("master_deck".into(), Value::Array(master_deck));
    if !play.is_empty() {
        deck.insert("play_pile".into(), Value::Array(play));
    }

    // `$.rng`: the search seed (0), the ascension, and the run's own dice —
    // forwarded all-or-nothing (a seed without counters is worse than none).
    // A Dread run (GH #1346) names its mode and the act's dominion; a Blitz
    // run (GH #1347) its mode, and its ascension is a Blitz level, converted
    // to the one a fight thresholds.
    let game_mode = get(run_state, "game_mode")
        .and_then(str_of)
        .map(|m| m.trim().to_lowercase())
        .unwrap_or_default();
    let fight_ascension = if game_mode == "blitz" {
        BLITZ_FIGHT_ASCENSION[usize::from(ascension).min(BLITZ_FIGHT_ASCENSION.len() - 1)]
    } else {
        ascension
    };
    let mut rng = Map::new();
    rng.insert("seed".into(), json!(0));
    rng.insert("ascension".into(), json!(fight_ascension));
    if let Some(raw) = board.get("rng").and_then(Value::as_object) {
        if let (Some(run_seed), Some(counters)) = (
            run_seed_of(raw.get("run_seed")),
            raw.get("counters").and_then(Value::as_object),
        ) {
            let clean: Map<String, Value> = counters
                .iter()
                .filter_map(|(k, v)| int_of(Some(v)).map(|n| (k.clone(), json!(n))))
                .collect();
            if !clean.is_empty() {
                rng.insert("run_seed".into(), json!(run_seed));
                rng.insert("counters".into(), Value::Object(clean));
            }
        }
    }

    let room_type = get(board, "room_type").and_then(str_of);
    let is_elite = room_type == Some("Elite");
    let mut out = json!({
        "wire_version": WIRE_VERSION,
        "phase": "PlayerPlayPhase",
        "round": round,
        "side": "Player",
        "is_elite": is_elite,
        "player": Value::Object(player_out),
        "deck": Value::Object(deck),
        "encounter": encounter,
        "turn": round,
        "combat_id": 0,
        "rng": Value::Object(rng),
        "active_attack": Value::Null,
        "pending_choice": Value::Null,
        "pending_play": Value::Null,
    });
    // Each optional field only off its default, as `state_to_dict` emits
    // them: the build (GH #800), the act (1-based) and a boss room (GH #989),
    // the run-wide floor v0.113.0's TheTower scales with (GH #1337).
    if let Some(v) = wire_game_version(board)? {
        out["game_version"] = json!(v);
    }
    if let Some(act) = int_of(board.get("act")).filter(|a| *a > 1) {
        out["act_index"] = json!(act);
    }
    if room_type == Some("Boss") {
        out["is_boss"] = json!(true);
    }
    if let Some(floor) = int_of(board.get("floor")).filter(|f| *f > 0) {
        out["total_floor"] = json!(floor);
    }
    if game_mode == "dread" {
        out["game_mode"] = json!("dread");
        if let Some(d) = get(run_state, "dominion")
            .and_then(str_of)
            .filter(|d| !d.is_empty())
        {
            out["dominion"] = json!(d);
        }
    } else if game_mode == "blitz" {
        out["game_mode"] = json!("blitz");
    }

    // Co-op teammates, in the mod's order (an `AnyAlly` card lands on the
    // first living one). Omitted when solo.
    let mut seats: BTreeMap<String, i64> = BTreeMap::new();
    let mut allies = Vec::new();
    for (i, a) in arr(board, "allies")
        .iter()
        .filter(|a| a.is_object())
        .enumerate()
    {
        let (ally, net_id) = translate_ally(a, i, &mut next_idx, &mut warnings)?;
        if let Some(net_id) = net_id {
            seats.insert(net_id, i as i64 + 1);
        }
        allies.push(ally);
    }
    if !allies.is_empty() {
        if let Some(net_id) = get(player, "net_id") {
            seats.insert(py_str(net_id), 0);
        }
        // Every teammate a whole seat (a current mod): a whole co-op fight
        // in which they keep drawing and playing (GH #1046), with our place
        // in the run's player order.
        let whole = allies.iter().all(is_whole_seat);
        out["allies"] = Value::Array(allies);
        if whole {
            out["coop_full"] = json!(true);
            if let Some(seat) = int_of(player.get("seat_index")).filter(|s| *s != 0) {
                out["coop_seat_index"] = json!(seat);
            }
        }
    }
    resolve_power_appliers(&mut out, &seats);
    Ok((out, warnings))
}

/// The paste's `room_type` as the engine's [`sts2_core::runstate::RoomType`]
/// (a fight is a Monster room unless the mod said Elite / Boss).
#[must_use]
pub fn room_type_of(board: &Value) -> sts2_core::runstate::RoomType {
    match get(board, "room_type").and_then(str_of) {
        Some("Elite") => sts2_core::runstate::RoomType::Elite,
        Some("Boss") => sts2_core::runstate::RoomType::Boss,
        _ => sts2_core::runstate::RoomType::Monster,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_character_is_read_off_the_first_prefixed_card() {
        let board = json!({
            "hand": [{"id": "BASH"}, {"id": 7}],
            "draw_pile": {"cards": [{"id": "CARD.DEFEND_SILENT"}]},
            "discard_pile": {"cards": [{"id": "IRONCLAD/STRIKE"}]},
        });
        assert_eq!(infer_character(&board), Some("Silent"));
        assert_eq!(infer_character(&json!({"hand": [{"id": "BASH"}]})), None);
        assert_eq!(infer_character(&json!({})), None);
    }

    #[test]
    fn an_unknown_card_id_is_reported_in_pythons_words() {
        assert_eq!(
            registry_card_id("CARD.INFERNO", "$").unwrap(),
            "INFERNO_CARD"
        );
        assert_eq!(
            registry_card_id("NOPE", "$.cards[0]").unwrap_err().0,
            "$.cards[0]: card id 'NOPE' doesn't map to any of ['NOPE', 'NOPE_CARD', \
             'NOPE_STATUS'] in sts2_rs card registry"
        );
    }

    #[test]
    fn card_ids_resolve_through_the_registry_suffixes() {
        assert_eq!(
            resolve_card_id("STRIKE_IRONCLAD", "$").unwrap(),
            "STRIKE_IRONCLAD"
        );
        assert_eq!(resolve_card_id("CARD.BASH", "$").unwrap(), "BASH");
        assert_eq!(
            resolve_card_id("DEMON_FORM", "$").unwrap(),
            "DEMON_FORM_CARD"
        );
        assert_eq!(
            resolve_card_id("IRONCLAD/STRIKE", "$").unwrap(),
            "STRIKE_IRONCLAD"
        );
        assert!(resolve_card_id("NOT_A_CARD", "$").is_err());
    }

    #[test]
    fn monster_titles_and_leaks_resolve() {
        let e = |name: &str, id: &str| json!({"name": name, "id": id});
        assert_eq!(
            resolve_monster_kind(&e("Shrinker Beetle", "SHRINKER_BEETLE"), "$").unwrap(),
            "ShrinkerBeetle"
        );
        assert_eq!(
            resolve_monster_kind(&e("Test Subject #C8", ""), "$").unwrap(),
            "TestSubject"
        );
        assert_eq!(
            resolve_monster_kind(&e("Test Subject", ""), "$").unwrap(),
            "TestSubject"
        );
        assert_eq!(
            resolve_monster_kind(&e("DECIMILLIPEDE_SEGMENT_FRONT.name", ""), "$").unwrap(),
            "DecimillipedeSegmentFront"
        );
        // The bridge id wins over a localized name (GH #1423), and tells apart
        // kinds that share one title.
        assert_eq!(
            resolve_monster_kind(&e("收缩甲虫", "SHRINKER_BEETLE"), "$").unwrap(),
            "ShrinkerBeetle"
        );
        assert_eq!(
            resolve_monster_kind(&e("战斗伙伴", "BATTLE_FRIEND_V1"), "$").unwrap(),
            "BattleFriendV1"
        );
        assert_eq!(
            resolve_monster_kind(&e("Decimillipede", "DECIMILLIPEDE_SEGMENT_BACK"), "$").unwrap(),
            "DecimillipedeSegmentBack"
        );
        assert_eq!(
            resolve_monster_kind(&e("???", "FUZZY_WURM_CRAWLER"), "$").unwrap(),
            "FuzzyWurmCrawler"
        );
        assert!(resolve_monster_kind(&e("???", "NOPE"), "$").is_err());
    }

    #[test]
    fn orbs_and_status_cards_resolve_by_id_under_a_localized_name() {
        for (id, want) in [
            ("LIGHTNING_ORB", "LIGHTNING"),
            ("LightningOrb", "LIGHTNING"),
            ("FROST_ORB", "FROST"),
            ("EMPTY_ORB", "EMPTY"),
        ] {
            let orb = json!({"id": id, "name": "闪电", "passive_value": 3, "evoke_value": 8});
            assert_eq!(translate_orb(&orb, "$").unwrap()["type"], want, "{id}");
        }
        assert_eq!(
            translate_orb(&json!({"name": "Frost"}), "$").unwrap()["type"],
            "FROST"
        );
        assert!(translate_orb(&json!({"id": "NOPE_ORB", "name": "闪电"}), "$").is_err());
        assert_eq!(status_title_by_id("DECAY"), Some("Decay"));
        assert_eq!(
            status_title_by_id("CARD.CURSE_OF_THE_BELL"),
            Some("Curse of the Bell")
        );
        assert_eq!(status_title_by_id("STRIKE_IRONCLAD"), None);
        // Potions match the id spelling-blind, like `_potion_id_from_wire`.
        let mut warnings = Vec::new();
        for id in ["FIRE_POTION", "FirePotion"] {
            let slot = json!({"id": id, "name": "火焰药水"});
            assert_eq!(translate_potion(&slot, "$", &mut warnings), "FIRE_POTION");
        }
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn helpers_match_python() {
        assert_eq!(fold_digits("Test Subject #C8"), "Test Subject #C{Count}");
        assert_eq!(bare_species_title("Test Subject #C{Count}"), "Test Subject");
        assert_eq!(normalize_affliction_name("TAINTED.title"), "Tainted");
        assert_eq!(normalize_affliction_name("Bound"), "Bound");
        assert_eq!(english_half("Bound / Gebunden"), "Bound");
    }

    #[test]
    fn powers_prefer_the_registry_id_then_the_title() {
        let p = translate_power(
            &json!({"id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2}),
            "$",
        )
        .unwrap();
        assert_eq!(p["name"], "VULNERABLE");
        assert_eq!(p["amount"], 2);
        let p = translate_power(
            &json!({"id": "CRIMSON_MANTLE", "name": "Crimson Mantle", "amount": 3,
            "vars": {"SelfDamage": 2}, "display_amount": 3}),
            "$",
        )
        .unwrap();
        assert_eq!(p["counter"], 2);
        assert_eq!(p["display"], 3);
        // GH #1836: Possess Speed's stolen Dexterity, sent as `Stolen`.
        let p = translate_power(
            &json!({"id": "POSSESS_SPEED_POWER", "name": "Possess Speed", "amount": 1,
            "vars": {"Stolen": 6}}),
            "$",
        )
        .unwrap();
        assert_eq!(p["counter"], 6);
        // Each The Bomb's damage (`TheBombPower.SetDamage`), which the engine
        // keeps on `counter`: rebuilt at 0, a second Bomb dealt nothing.
        let p = translate_power(
            &json!({"id": "THE_BOMB_POWER", "name": "The Bomb", "amount": 3,
            "display_amount": 3, "vars": {"Damage": 40}}),
            "$",
        )
        .unwrap();
        assert_eq!(p["counter"], 40);
        // GH #1906: Test Subject's Multi Claw growth, sent by the mod as a
        // Painful Stabs var, as `adapter.py` already maps it (#1897).
        let p = translate_power(
            &json!({"id": "PAINFUL_STABS_POWER", "name": "Painful Stabs", "amount": 1,
            "vars": {"ExtraMultiClawCount": 2}}),
            "$",
        )
        .unwrap();
        assert_eq!(p["counter"], 2);
        assert!(translate_power(&json!({"id": "X", "name": "Not A Power"}), "$").is_err());
    }

    /// GH #1906: the seat-history counts the mod now sends reach the board,
    /// as `adapter._copy_seat_history` keeps them.
    #[test]
    fn the_newer_seat_history_counts_are_kept() {
        let src = json!({"cards_drawn_this_combat": 9, "cards_generated_this_combat": 2,
            "discards_this_turn": 3});
        let mut out = Map::new();
        copy_seat_history(&src, &mut out, false);
        assert_eq!(out.get("cards_drawn_this_combat"), Some(&json!(9)));
        assert_eq!(out.get("cards_generated_this_combat"), Some(&json!(2)));
        assert_eq!(out.get("discards_this_turn"), Some(&json!(3)));
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn relics_potions_orbs() {
        let r = translate_relic(&json!({"id": "burning_blood", "name": "Wax Burning Blood", "counter": null, "is_wax": true}), "$").unwrap();
        assert_eq!(
            r,
            json!({"id": "BURNING_BLOOD", "counter": null, "is_wax": true})
        );
        let r = translate_relic(
            &json!({"id": "NOPE", "name": "Burning Blood", "counter": 3}),
            "$",
        )
        .unwrap();
        assert_eq!(r["id"], "BURNING_BLOOD");
        assert_eq!(r["counter"], 3);
        assert!(translate_relic(&json!({"id": "NOPE", "name": "Not A Relic"}), "$").is_err());
        let mut w = Vec::new();
        assert_eq!(
            translate_potion(&json!({"slot": 0, "empty": true}), "$", &mut w),
            Value::Null
        );
        assert_eq!(
            translate_potion(
                &json!({"slot": 0, "id": "FIRE_POTION", "name": "Fire Potion"}),
                "$",
                &mut w
            ),
            json!("FIRE_POTION")
        );
        assert_eq!(
            translate_potion(&json!({"slot": 0, "id": "X", "name": "Nope"}), "$", &mut w),
            Value::Null
        );
        assert_eq!(w.len(), 1);
        assert_eq!(translate_orb(&json!({"id": "LIGHTNING_ORB", "name": "Lightning", "passive_value": 3, "evoke_value": 8}), "$").unwrap()["type"], "LIGHTNING");
        assert_eq!(
            translate_orb(
                &json!({"id": "FrostOrb", "passive_value": 2, "evoke_value": 5}),
                "$"
            )
            .unwrap()["type"],
            "FROST"
        );
        assert!(translate_orb(&json!({"id": "FIRE_ORB"}), "$").is_err());
    }

    #[test]
    fn a_minimal_board_becomes_an_engine_state() {
        let board = json!({
            "type": "combat_action", "round": 2, "room_type": "Elite",
            "player": {"hp": 50, "max_hp": 80, "block": 3, "energy": 2, "max_energy": 3,
                "powers": [{"id": "STRENGTH_POWER", "name": "Strength", "amount": 1}]},
            "hand": [{"id": "STRIKE_IRONCLAD", "name": "Strike", "type": "Attack", "cost": 1, "effective_cost": 1, "upgrade_level": 1, "retain": true},
                     {"id": "BURN", "name": "Burn", "type": "Status", "cost": -2}],
            "draw_pile": {"cards": [{"id": "BASH", "type": "Attack", "cost": 2, "effective_cost": 2}]},
            "discard_pile": {"cards": []}, "exhaust_pile": {"cards": []},
            "enemies": [{"id": "NIBBIT", "name": "Nibbit", "is_alive": true, "hp": 20, "max_hp": 30, "block": 0,
                "powers": [], "next_move_id": "BITE_MOVE",
                "intents": [{"kind": "SingleAttackIntent", "intent_type": "Attack", "damage": 6, "hits": 1}]}],
            "potions": [{"slot": 0, "empty": true}],
            "pets": [{"id": "BYRDPIP", "name": "Byrdpip"}],
            "run_state": {"gold": 120, "character": "IRONCLAD", "ascension": 5,
                "potion_slot_count": 2,
                "relics": [{"id": "BURNING_BLOOD", "name": "Burning Blood", "counter": null}],
                "master_deck": [{"id": "STRIKE_IRONCLAD", "type": "Attack", "cost": 1}]},
            "rng": {"run_seed": 12345, "counters": {"shuffle": 3, "monster_ai": 1}},
        });
        let (wire, warnings) = combat_wire_from_bridge(&board, "Ironclad", 5).unwrap();
        assert_eq!(wire["is_elite"], true);
        assert_eq!(wire["player"]["gold"], 120);
        assert_eq!(
            wire["player"]["potion_slots"], 2,
            "the A5 board's Tight Belt"
        );
        assert_eq!(wire["player"]["powers"][0]["name"], "STRENGTH");
        assert_eq!(wire["deck"]["hand"][0]["flags"], json!(["retain"]));
        assert_eq!(wire["deck"]["hand"][0]["upgraded"], true);
        assert_eq!(wire["deck"]["hand"][1]["id"], "status:Burn");
        assert_eq!(wire["deck"]["draw_pile"][0]["stable_idx"], 2);
        assert_eq!(wire["deck"]["master_deck"][0]["stable_idx"], 3);
        assert_eq!(wire["encounter"][0]["kind"], "Nibbit");
        assert_eq!(
            wire["encounter"][0]["intents"][0],
            json!({"kind": "Attack", "amount": 6, "times": 1})
        );
        assert_eq!(wire["encounter"][0]["move_state"], "BITE_MOVE");
        assert_eq!(wire["rng"]["run_seed"], 12345);
        assert_eq!(wire["rng"]["ascension"], 5);
        // A v0.111.0 master seed is a full u64 (GH #788): one above i64::MAX
        // keeps the run's dice instead of silently dropping the block.
        let mut wide = board.clone();
        wide["rng"]["run_seed"] = json!(u64::MAX - 7);
        let (w, _) = combat_wire_from_bridge(&wide, "Ironclad", 5).unwrap();
        assert_eq!(w["rng"]["run_seed"], json!(u64::MAX - 7));
        assert_eq!(w["rng"]["counters"]["shuffle"], 3);
        // A signed 32-bit seed is its `uint` view, as in the run graft.
        wide["rng"]["run_seed"] = json!(-2);
        let (w, _) = combat_wire_from_bridge(&wide, "Ironclad", 5).unwrap();
        assert_eq!(w["rng"]["run_seed"], json!(u64::from(u32::MAX - 1)));
        assert!(wire["player"].get("osty").is_none());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("Byrdpip"));
        assert_eq!(room_type_of(&board), sts2_core::runstate::RoomType::Elite);
        // The engine accepts it.
        let cs = sts2_wire::state_from_json(&wire).expect("engine parses the wire");
        assert_eq!(cs.player.current_hp, 50);
    }

    #[test]
    fn a_wuppy_pet_becomes_the_engines_wuppy() {
        // GH #1496: the v0.113.0 Wuppy pet rides `$.player.wuppy` (no warning),
        // and the engine rebuilds it with its WuppysLoyaltyPower.
        let board = |pet: Value| {
            json!({
                "type": "combat_action", "round": 1, "game_version": "v0.113.0",
                "player": {"hp": 50, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
                "hand": [], "draw_pile": {"cards": []}, "discard_pile": {"cards": []},
                "exhaust_pile": {"cards": []},
                "enemies": [{"id": "NIBBIT", "name": "Nibbit", "is_alive": true, "hp": 20, "max_hp": 30,
                    "block": 0, "powers": [], "next_move_id": "BITE_MOVE", "intents": []}],
                "potions": [], "pets": [pet],
                // No Wuppy relic: a v0.113.0-only relic is still dropped from
                // a live board (the `relic` registry is the embedding's), and
                // the pet does not need it.
                "run_state": {"gold": 0, "character": "IRONCLAD", "ascension": 0,
                    "relics": [], "master_deck": []},
            })
        };
        let live = board(
            json!({"id": "WUPPY", "name": "Wuppy", "is_alive": true, "hp": 17, "max_hp": 30, "block": 2}),
        );
        let (wire, warnings) = combat_wire_from_bridge(&live, "Ironclad", 0).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(wire["player"].get("osty").is_none());
        assert_eq!(
            wire["player"]["wuppy"],
            json!({"alive": true, "hp": 17, "max_hp": 30, "block": 2})
        );
        let cs = sts2_wire::state_from_json(&wire).expect("engine parses the wire");
        let pet = cs
            .creatures
            .iter()
            .find(|c| c.species == Some(sts2_core::MonsterKind::Wuppy))
            .expect("Wuppy rebuilt");
        assert_eq!((pet.current_hp, pet.max_hp, pet.block), (17, 30, 2));
        assert!(!cs.enemy_uids.contains(&sts2_core::CreatureUid(pet.id)));

        let dead = board(
            json!({"id": "WUPPY", "name": "Wuppy", "is_alive": false, "hp": 0, "max_hp": 30}),
        );
        let (wire, _) = combat_wire_from_bridge(&dead, "Ironclad", 0).unwrap();
        assert_eq!(
            wire["player"]["wuppy"],
            json!({"alive": false, "hp": 0, "max_hp": 30, "block": 0})
        );
    }
}
