//! Ids and display names for everything a sandbox position can mention — the
//! browser twin of `sts2-pycli/python/simlab/catalog.py`.
//!
//! The engine registries are the source of truth for *which* ids exist
//! (`registry_names_for`); the game's own localization (bundled in
//! [`crate::localization::Loc`]) supplies the words a person reads, and cards
//! carry their own name/type/cost/rarity on the registry `CardSpec`. Where a
//! datum is absent everything degrades to a title-cased id, exactly as the
//! desktop catalog does on a trimmed deploy.
//!
//! `monster_moves` / `list_encounters` / `list_registry_names` are thin
//! formatters over the same `sts2_content` tables the Python oracle reads; the
//! correctness-critical construction (`build_scenario_state`) is shared through
//! [`sts2_wire::scenario`], so the sandbox stays byte-parallel with the oracle.

use crate::localization::Loc;
use crate::simlab::{camel_to_upper_snake, title_case, ORB_BASE};
use serde_json::{json, Map, Value};
use sts2_content::monster_dispatch::{build_ai_with_ascension, MonsterSpawnArgs, SpawnPosition};
use sts2_core::monster_ai::MonsterState;
use sts2_core::GameVersion;
use sts2_wire::emit::combat::bare_intent_to_value;
use sts2_wire::state_dict::{monster_kind_from_python_name, registry_names_for};

/// Per-character starters: max HP, starter relic, starter deck (registry ids),
/// orb slots. Ported from `catalog.py::STARTERS`.
struct Starter {
    hp: i64,
    relic: &'static str,
    orb_slots: i64,
    deck: fn() -> Vec<String>,
}

fn rep(id: &str, n: usize) -> Vec<String> {
    vec![id.to_string(); n]
}

fn ironclad_deck() -> Vec<String> {
    [
        rep("STRIKE_IRONCLAD", 5),
        rep("DEFEND_IRONCLAD", 4),
        vec!["BASH".into()],
    ]
    .concat()
}
fn silent_deck() -> Vec<String> {
    [
        rep("STRIKE_SILENT", 5),
        rep("DEFEND_SILENT", 5),
        vec!["NEUTRALIZE".into(), "SURVIVOR".into()],
    ]
    .concat()
}
fn defect_deck() -> Vec<String> {
    [
        rep("STRIKE_DEFECT", 4),
        rep("DEFEND_DEFECT", 4),
        vec!["ZAP".into(), "DUALCAST".into()],
    ]
    .concat()
}
fn regent_deck() -> Vec<String> {
    [
        rep("STRIKE_REGENT", 4),
        rep("DEFEND_REGENT", 4),
        vec!["FALLING_STAR".into(), "VENERATE".into()],
    ]
    .concat()
}
fn necro_deck() -> Vec<String> {
    [
        rep("STRIKE_NECROBINDER", 4),
        rep("DEFEND_NECROBINDER", 4),
        vec!["BODYGUARD".into(), "UNLEASH".into()],
    ]
    .concat()
}

fn starter(character: &str) -> Starter {
    match character {
        "Silent" => Starter {
            hp: 70,
            relic: "RING_OF_THE_SNAKE",
            orb_slots: 0,
            deck: silent_deck,
        },
        "Defect" => Starter {
            hp: 75,
            relic: "CRACKED_CORE",
            orb_slots: 3,
            deck: defect_deck,
        },
        "Regent" => Starter {
            hp: 75,
            relic: "DIVINE_RIGHT",
            orb_slots: 0,
            deck: regent_deck,
        },
        "Necrobinder" => Starter {
            hp: 66,
            relic: "BOUND_PHYLACTERY",
            orb_slots: 0,
            deck: necro_deck,
        },
        _ => Starter {
            hp: 80,
            relic: "BURNING_BLOOD",
            orb_slots: 0,
            deck: ironclad_deck,
        },
    }
}

/// The starter row for `character` (PascalCase): `(hp, deck, relic, orbSlots)`.
#[must_use]
pub fn starter_for(character: &str) -> (i64, Vec<String>, &'static str, i64) {
    let s = starter(character);
    (s.hp, (s.deck)(), s.relic, s.orb_slots)
}

/// Sorted canonical names for a registry kind (card/relic/potion/power/monster).
pub fn list_registry_names(kind: &str) -> Result<Vec<String>, String> {
    registry_names_for(kind)
}

/// Every `Move` state of `kind`'s AI, `[{id, initial, intent}]` — the shared
/// engine move machine, formatted for the sandbox's intent dropdown and the
/// position builder's forced-move lookup.
pub fn monster_moves(kind: &str, ascension: i64) -> Result<Value, String> {
    let species =
        monster_kind_from_python_name(kind).ok_or_else(|| format!("unknown monster {kind:?}"))?;
    let position = SpawnPosition {
        alone: true,
        front: true,
        back: false,
        index: 0,
    };
    let asc = u8::try_from(ascension.clamp(0, 255)).unwrap_or(0);
    let ai = build_ai_with_ascension(species, MonsterSpawnArgs::default(), position, asc);
    let mut rows = Vec::new();
    for (id, st) in ai.states.iter() {
        if let MonsterState::Move(mv) = st {
            let intent = mv
                .intents
                .first()
                .copied()
                .map_or(Value::Null, bare_intent_to_value);
            rows.push(json!({"id": id, "initial": ai.initial == *id, "intent": intent}));
        }
    }
    Ok(Value::Array(rows))
}

/// The encounter catalog: every act table, `[{id, display, tier, act, act_name,
/// monsters}]`. Ported from `list_encounters_impl`.
#[must_use]
pub fn list_encounters() -> Value {
    use sts2_content::encounters::{
        act1_overgrowth, act1_underdocks, act2_hive, act3_glory, EncounterDef, EncounterTier,
    };
    fn tier_name(t: EncounterTier) -> &'static str {
        match t {
            EncounterTier::Weak => "Weak",
            EncounterTier::Normal => "Normal",
            EncounterTier::Elite => "Elite",
            EncounterTier::Boss => "Boss",
            _ => "Unknown",
        }
    }
    let tables: [(&[EncounterDef], u8, &str); 4] = [
        (act1_overgrowth::ACT1_ENCOUNTERS, 1, "Overgrowth"),
        (act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS, 1, "Underdocks"),
        (act2_hive::ACT2_ENCOUNTERS, 2, "Hive"),
        (act3_glory::ACT3_ENCOUNTERS, 3, "Glory"),
    ];
    let mut out = Vec::new();
    for (table, act, act_name) in tables {
        for def in table {
            let monsters: Vec<String> = def
                .canonical_roster()
                .iter()
                .map(|(k, _)| k.python_name().to_string())
                .collect();
            out.push(json!({
                "id": def.id,
                "display": def.display,
                "tier": tier_name(def.tier),
                "act": act,
                "act_name": act_name,
                "monsters": monsters,
            }));
        }
    }
    Value::Array(out)
}

// ---------------------------------------------------------------------------
// Display names (embedded localization → title-case fallback)
// ---------------------------------------------------------------------------

/// The display loc of game build `version` (GH #877: a v0.111.0 sandbox
/// names its cards with v0.111.0 text).
fn loc(version: GameVersion) -> &'static Loc {
    Loc::embedded_for(version)
}

fn loc_or_title(l: &Loc, key: &str, ident: &str) -> String {
    l.get(key).map_or_else(|| title_case(ident), str::to_string)
}

/// A card's display name: `{id}.title` from localization, else the same key
/// minus a `_CARD` suffix (`DEMON_FORM_CARD` → `DEMON_FORM`), else title-case —
/// the join rules from `catalog.py::_card_rows`. `CardSpec::name` is the UPPER
/// registry id, not a word, so it is not used here.
fn card_display_name(l: &Loc, id: &str) -> String {
    if let Some(n) = l.get(&format!("{id}.title")) {
        return n.to_string();
    }
    let base = id.strip_suffix("_CARD").unwrap_or(id);
    l.get(&format!("{base}.title"))
        .map_or_else(|| title_case(base), str::to_string)
}

fn card_rows(l: &Loc) -> Vec<Value> {
    let mut ids = registry_names_for("card").unwrap_or_default();
    ids.sort();
    ids.into_iter()
        .map(|id| {
            let (ty, rarity, cost) = match sts2_content::names::resolve_card_name(&id, None) {
                Ok(spec) => (
                    sts2_wire::emit::names::card_type_python_name(spec.card_type).to_string(),
                    format!("{:?}", spec.rarity),
                    Value::from(i64::from(spec.cost)),
                ),
                Err(_) => (String::new(), String::new(), Value::Null),
            };
            let name = card_display_name(l, &id);
            json!({"id": id, "name": name, "type": ty, "rarity": rarity, "cost": cost, "character": ""})
        })
        .collect()
}

fn named_rows(l: &Loc, kind: &str, suffix: &str) -> Vec<Value> {
    let mut ids = registry_names_for(kind).unwrap_or_default();
    ids.sort();
    ids.into_iter()
        .map(|id| {
            let key = format!("{id}{suffix}.title");
            json!({"id": id, "name": loc_or_title(l, &key, &id)})
        })
        .collect()
}

fn monster_rows(l: &Loc) -> Vec<Value> {
    let mut ids = registry_names_for("monster").unwrap_or_default();
    ids.sort();
    ids.into_iter()
        .map(|kind| {
            let snake = camel_to_upper_snake(&kind);
            let name = loc_or_title(l, &format!("{snake}.name"), &kind);
            let moves = monster_moves(&kind, 0).unwrap_or(Value::Array(Vec::new()));
            let move_rows: Vec<Value> = moves
                .as_array()
                .map(|ms| {
                    ms.iter()
                        .map(|mv| {
                            let mid = mv.get("id").and_then(Value::as_str).unwrap_or("");
                            let bare = mid.strip_suffix("_MOVE").unwrap_or(mid);
                            let title = l
                                .get(&format!("{snake}.moves.{mid}.title"))
                                .or_else(|| l.get(&format!("{snake}.moves.{bare}.title"))).map_or_else(|| title_case(bare), str::to_string);
                            json!({
                                "id": mid,
                                "title": title,
                                "initial": mv.get("initial").and_then(Value::as_bool).unwrap_or(false),
                                "intent": mv.get("intent").cloned().unwrap_or(Value::Null),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            json!({"id": kind, "name": name, "moves": move_rows})
        })
        .collect()
}

fn character_rows() -> Vec<Value> {
    crate::simlab::CHARACTERS
        .iter()
        .map(|&c| {
            let (hp, deck, relic, orb_slots) = starter_for(c);
            let display = match c {
                "Ironclad" => "The Ironclad",
                "Silent" => "The Silent",
                "Defect" => "The Defect",
                "Regent" => "The Regent",
                "Necrobinder" => "The Necrobinder",
                other => other,
            };
            json!({"id": c, "name": display, "hp": hp, "relic": relic, "deck": deck, "orbSlots": orb_slots})
        })
        .collect()
}

fn orb_rows() -> Vec<Value> {
    ORB_BASE
        .iter()
        .map(|(k, p, e)| json!({"id": k, "name": k, "passive": p, "evoke": e}))
        .collect()
}

/// The full sandbox catalog (default build's names). Sweep-only fields
/// (`strategies`) are present but empty in the browser build, where the
/// distribution sweeps are desktop-only.
#[must_use]
pub fn catalog() -> Value {
    catalog_for(GameVersion::default())
}

/// [`catalog`] named with game build `version`'s loc
/// (`GET /api/simlab/catalog?game_version=…`, GH #877).
#[must_use]
pub fn catalog_for(version: GameVersion) -> Value {
    let l = loc(version);
    json!({
        "characters": character_rows(),
        "cards": card_rows(l),
        "relics": named_rows(l, "relic", ""),
        "potions": named_rows(l, "potion", ""),
        "powers": named_rows(l, "power", "_POWER"),
        "orbs": orb_rows(),
        "monsters": monster_rows(l),
        "encounters": list_encounters(),
        "strategies": Value::Array(Vec::new()),
    })
}

/// Display name for one id — the catalog-backed `name_of` the labelers use
/// (default build's names).
#[must_use]
pub fn name_of(kind: &str, ident: &str) -> String {
    name_of_for(GameVersion::default(), kind, ident)
}

/// [`name_of`] in game build `version`'s loc.
#[must_use]
pub fn name_of_for(version: GameVersion, kind: &str, ident: &str) -> String {
    let l = loc(version);
    let key = match kind {
        "card" => return card_display_name(l, ident),
        "relic" | "potion" => format!("{ident}.title"),
        "power" => format!("{ident}_POWER.title"),
        "monster" => format!("{}.name", camel_to_upper_snake(ident)),
        "encounter" | "event" => format!("{ident}.title"),
        _ => return title_case(ident),
    };
    loc_or_title(l, &key, ident)
}

/// The SimLab `/api/meta` payload. In the browser the sweep engine is
/// desktop-only, so the sweep-facing lists are minimal; the sandbox reads the
/// full catalog instead.
#[must_use]
pub fn meta() -> Value {
    let mut cards = registry_names_for("card").unwrap_or_default();
    cards.sort();
    let mut monsters = registry_names_for("monster").unwrap_or_default();
    monsters.sort();
    let encounters: Vec<Value> = list_encounters()
        .as_array()
        .map(|es| es.iter().filter_map(|e| e.get("id").cloned()).collect())
        .unwrap_or_default();
    let mut deck_presets = Map::new();
    for (name, cards) in crate::simlab_sweep::deck_presets() {
        deck_presets.insert(name.to_string(), json!(cards));
    }
    json!({
        "deck_presets": deck_presets,
        "strategies": crate::simlab_sweep::STRATEGIES,
        "encounters": encounters,
        "cards": cards,
        "monsters": monsters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_every_registry_the_editor_offers() {
        let cat = catalog();
        for key in [
            "cards",
            "relics",
            "potions",
            "powers",
            "monsters",
            "encounters",
        ] {
            assert!(
                !cat[key].as_array().unwrap().is_empty(),
                "catalog.{key} is empty"
            );
        }
        // Five playable characters, each with a starter deck.
        assert_eq!(cat["characters"].as_array().unwrap().len(), 5);
        assert!(!cat["characters"][0]["deck"].as_array().unwrap().is_empty());
    }

    #[test]
    fn cards_carry_registry_name_type_and_cost() {
        let cat = catalog();
        let bash = cat["cards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == json!("BASH"))
            .expect("BASH is in the registry");
        assert_eq!(bash["name"], json!("Bash"));
        assert_eq!(bash["type"], json!("Attack"));
        assert_eq!(bash["cost"], json!(2));
    }

    #[test]
    fn monster_moves_returns_the_shared_move_machine() {
        let moves = monster_moves("Mawler", 0).expect("Mawler has moves");
        let arr = moves.as_array().unwrap();
        assert!(!arr.is_empty());
        assert!(arr.iter().any(|m| m["initial"] == json!(true)));
        assert!(monster_moves("NotAMonster", 0).is_err());
    }

    #[test]
    fn starter_for_matches_the_character_table() {
        assert_eq!(starter_for("Ironclad").0, 80);
        assert_eq!(starter_for("Silent").0, 70);
        assert_eq!(starter_for("Defect").3, 3); // orb slots
                                                // Unknown → Ironclad default.
        assert_eq!(starter_for("Nobody").2, "BURNING_BLOOD");
    }
}
