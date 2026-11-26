//! The run graft: the live run behind a mod message rebuilt as an engine
//! `RunManager` (GH #1426; was `realgame_instinct.graft_run`).
//!
//! The same steps, in the same load-bearing order, as the workbench's state
//! import (`Game::start_imported`: place on the map, empty the deck, relics,
//! deck, potions, HP/gold, RNG cursors last), on a run built the way the
//! Instinct student's campaign builds its own (`mode="agent"`, interactive
//! potion swap on: `sts2_e2e::env::new_run_at`), so the engine offers exactly
//! the decisions the student was trained on.

use serde_json::{json, Value};
use sts2_core::map::{ActMap, MapCoord, MapPointType};
use sts2_core::run_manager::RunManager;
use sts2_core::runstate::engine_total_floor;
use sts2_wire::run_builder::{build_run, RunOptions};
use sts2_workbench::game::map_disagreements;
use sts2_workbench::state_import::{run_snapshot, RunSnapshot};

use crate::i18n::{tr, trf};
use crate::py;

/// The act whose map the Spoils Map quest card replaces (`SpoilsActIndex =
/// 1`, 0-based, `SpoilsMap.cs:47-50`).
pub const SPOILS_MAP_ACT: i64 = 2;

/// The note when a co-op run stands on the nearest node of a map that is
/// not the screen's.
pub const COOP_MAP_APPROX: &str = "co-op: placed on the nearest node of a map that differs";

/// Why a graft failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraftError {
    /// No run to rebuild, or the engine refuses it: the message is
    /// player-facing (`InstinctUnavailable`).
    Unavailable(String),
    /// A malformed message the Python graft let escape as `ValueError`.
    Value(String),
}

type Result<T> = std::result::Result<T, GraftError>;

/// The seed a seedless rebuild is built from. Nothing the student reads comes
/// from it: the act map and its bosses are the screen's, and no RNG cursor is
/// restored (a referee keeps its dice).
pub const SEEDLESS_SEED: &str = "SEEDLESS";

/// Does the message ask for a rebuild without the run's seed? A Spire Bench
/// observation has none: the run is rebuilt from what a player sees
/// (`"seedless": true`, with the act map), never from the dice.
pub fn seedless(state: &Value) -> bool {
    state.get("seedless").and_then(Value::as_bool) == Some(true)
}

fn point_type(name: &str) -> Option<MapPointType> {
    Some(match name {
        "Unknown" => MapPointType::Unknown,
        "Monster" => MapPointType::Monster,
        "Elite" => MapPointType::Elite,
        "Boss" => MapPointType::Boss,
        "Shop" => MapPointType::Shop,
        "RestSite" => MapPointType::RestSite,
        "Treasure" => MapPointType::Treasure,
        "Ancient" => MapPointType::Ancient,
        _ => return None,
    })
}

fn xy(v: &Value) -> Option<(i32, i32)> {
    let c = i32::try_from(v.get("col")?.as_i64()?).ok()?;
    let r = i32::try_from(v.get("row")?.as_i64()?).ok()?;
    Some((c, r))
}

/// The act map a screen shows (the mod's flat `nodes`, or headless `rows`, of
/// nodes with `type` and `children`; `boss`; and on an A10 final act the
/// second boss node, as `second_boss` or a Boss node above the boss) as an
/// engine `ActMap`,
/// laid out the way the generator lays its own: the start node at (3, 0),
/// typed `Ancient`, leading to every node of row 1; the boss on the row
/// after the last room row.
///
/// # Errors
///
/// A node or edge the screen's map does not describe.
pub fn import_act_map(fm: &Value) -> std::result::Result<ActMap, String> {
    let (bc, br) = xy(py::get(fm, "boss")).ok_or("the map has no boss node")?;
    let mut map = ActMap::new(br - 1);
    let start = map.get_or_create(3, 0);
    map.point_mut(start).point_type = MapPointType::Ancient;
    map.start = Some(start);
    let boss = map.get_or_create(bc, br);
    map.point_mut(boss).point_type = MapPointType::Boss;
    map.boss = Some(boss);
    let nodes: Vec<&Value> = if py::list(fm, "nodes").is_empty() {
        py::list(fm, "rows").iter().flat_map(py::as_list).collect()
    } else {
        py::list(fm, "nodes").iter().collect()
    };
    for n in &nodes {
        let (c, r) = xy(n).ok_or("a map node has no coordinate")?;
        let ty = py::get(n, "type").as_str().unwrap_or("");
        let ty = point_type(ty).ok_or_else(|| format!("unknown map node type {ty:?}"))?;
        let i = map.get_or_create(c, r);
        map.point_mut(i).point_type = ty;
    }
    // The A10 second boss: its own key, or a Boss node above the boss.
    let second = xy(py::get(fm, "second_boss"))
        .or_else(|| nodes.iter().filter_map(|n| xy(n)).find(|&(_, r)| r > br));
    if let Some((sc, sr)) = second {
        let i = map.get_or_create(sc, sr);
        map.point_mut(i).point_type = MapPointType::Boss;
        map.second_boss = Some(i);
    }
    for n in &nodes {
        let (c, r) = xy(n).ok_or("a map node has no coordinate")?;
        let from = map.get_or_create(c, r);
        if r == 1 {
            map.add_child(start, from);
        }
        for child in py::list(n, "children") {
            let (cc, cr) = xy(child).ok_or("a map edge has no coordinate")?;
            if map.get_point(MapCoord::new(cc, cr)).is_none() {
                return Err(format!(
                    "({c}, {r}) leads to ({cc}, {cr}), which is not on the map"
                ));
            }
            let to = map.get_or_create(cc, cr);
            map.add_child(from, to);
        }
    }
    if let Some(sb) = map.second_boss {
        if !map.children_idx_of(boss).contains(&sb) {
            map.add_child(boss, sb);
        }
    }
    Ok(map)
}

/// The engine's own `&'static` id of a boss encounter the screen names.
pub(crate) fn boss_encounter(id: &Value) -> Result<Option<&'static str>> {
    let Some(id) = id.as_str().filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    sts2_core::run_room_gen::all_boss_encounter_ids()
        .into_iter()
        .find(|b| *b == id)
        .map(Some)
        .ok_or_else(|| {
            unavailable(trf(
                "the engine knows no boss encounter {id}",
                &[("id", &format!("{id:?}"))],
            ))
        })
}

/// Put the screen's act map on `rm` (advanced to `act`), with the bosses the
/// screen names. A second boss the screen does not name yet stays unnamed:
/// the student is not shown what the player cannot see.
fn install_screen_map(rm: &mut RunManager, act: i64, state: &Value) -> Result<()> {
    let fm = py::get(state, "map");
    if !py::truthy(fm) || (py::list(fm, "rows").is_empty() && py::list(fm, "nodes").is_empty()) {
        return Err(unavailable(tr(
            "a rebuild without the seed needs the act map",
        )));
    }
    let _ = enter_act(rm, act);
    let map = import_act_map(fm)
        .map_err(|e| unavailable(trf("the act map cannot be read ({e})", &[("e", &e)])))?;
    let boss = boss_encounter(py::get(py::get(fm, "boss"), "id"))?;
    let second = boss_encounter(py::get(py::get(fm, "second_boss"), "id"))?;
    let idx = usize::try_from(act - 1)
        .map_err(|_| unavailable(trf("act {act} is out of range", &[("act", &act)])))?;
    let rooms = rm.run_state.generated_rooms.get_mut(idx).ok_or_else(|| {
        unavailable(trf(
            "the engine has no rooms for act {act}",
            &[("act", &act)],
        ))
    })?;
    if let Some(b) = boss {
        rooms.boss_id = b;
    }
    rooms.second_boss_id = second;
    rm.run_state.current_act_map = Some(map);
    Ok(())
}

fn unavailable(msg: impl Into<String>) -> GraftError {
    GraftError::Unavailable(msg.into())
}

/// A multiplayer run (`player_count` > 1): AI partners, Steam co-op.
#[must_use]
pub fn is_coop(state: &Value) -> bool {
    let n = py::get(state, "player_count");
    let n = if py::truthy(n) { n } else { &Value::Null };
    match n {
        Value::Null => false,
        Value::Bool(b) => i64::from(*b) > 1,
        Value::Number(x) => x
            .as_i64()
            .or_else(|| x.as_f64().map(|f| f as i64))
            .is_some_and(|v| v > 1),
        Value::String(s) => s.trim().parse::<i64>().is_ok_and(|v| v > 1),
        _ => false,
    }
}

/// `_room_kind`: `monster` / `elite` / `boss`, `monster` for anything else.
fn room_kind(src: &Value) -> &'static str {
    let raw = py::get(src, "room_type");
    let raw = if py::truthy(raw) {
        py::str_of(raw).to_lowercase()
    } else {
        String::new()
    };
    match raw.as_str() {
        "elite" => "elite",
        "boss" => "boss",
        _ => "monster",
    }
}

/// The room the mod says the player stands in (`current_room`, the game's
/// `RoomType` name), in the engine's `set_room_type` spelling. `None` when
/// the message has none, or names one a map node cannot be (`Map`,
/// `Unassigned`).
#[must_use]
pub fn live_room(state: &Value) -> Option<&'static str> {
    match state.get("current_room").and_then(Value::as_str)? {
        "Monster" => Some("monster"),
        "Elite" => Some("elite"),
        "Boss" => Some("boss"),
        "Event" => Some("event"),
        "Shop" => Some("shop"),
        "RestSite" => Some("rest_site"),
        "Treasure" => Some("treasure"),
        _ => None,
    }
}

/// The fight's room kind, if the request has one.
fn graft_room(state: &Value) -> Option<&'static str> {
    [state, py::get(state, "combat")]
        .into_iter()
        .find(|src| src.is_object() && py::truthy(py::get(src, "room_type")))
        .map(room_kind)
}

/// `realgame_instinct_screens.card_spec`: a wire Card as a graft card spec.
#[must_use]
pub fn card_spec(card: &Value) -> Value {
    let level = match py::get(card, "upgrade_level") {
        Value::Null => json!(i64::from(py::truthy(py::get(card, "upgraded")))),
        v => v.clone(),
    };
    let level = if py::truthy(&level) {
        py::int_of(&level)
    } else {
        0
    };
    let mut spec = json!({"id": py::get(card, "id"), "upgraded": level});
    let ench = py::get(card, "enchantment");
    if py::truthy(ench) {
        let amount = py::get(card, "enchantment_amount");
        let amount = if py::truthy(amount) {
            py::int_of(amount)
        } else {
            1
        };
        spec["enchantment"] = ench.clone();
        spec["enchantment_amount"] = json!(amount);
    }
    // Mad Science's per-card type and rider (Tinker Time's picks).
    let rider = py::get(card, "tinker_time_rider");
    if py::truthy(rider) {
        spec["tinker_time_rider"] = rider.clone();
        spec["type"] = py::get(card, "type").clone();
    }
    spec
}

fn coord(c: i64, r: i64) -> MapCoord {
    MapCoord::new(c as i32, r as i32)
}

fn place_on_map(
    rm: &mut RunManager,
    act: i64,
    col: i64,
    row: i64,
    visited: Vec<MapCoord>,
    act_floor: i64,
    total: i64,
) -> std::result::Result<(), String> {
    let act =
        u8::try_from(act).map_err(|_| format!("debug_place_on_map: act {act} out of range"))?;
    rm.debug_place_on_map(
        act,
        coord(col, row),
        visited,
        act_floor as i32,
        total as i32,
    )
    .map_err(|e| format!("debug_place_on_map: {e}"))
}

/// Advance `rm` to `act` and return the engine's map of it:
/// `debug_place_on_map` walks the acts (generating each map off its own map
/// RNG) before it checks the node, so an off-map coordinate advances the
/// run and is then refused.
fn enter_act(rm: &mut RunManager, act: i64) -> Value {
    let _ = place_on_map(rm, act, -1, -1, Vec::new(), 0, 0);
    let fm = sts2_wire::emit::run_manager::full_map(rm);
    if py::truthy(&fm) {
        fm
    } else {
        json!({})
    }
}

/// `int(d.get(k) or default)`.
fn int_or(d: &Value, key: &str, default: i64) -> i64 {
    let v = py::get(d, key);
    if py::truthy(v) {
        py::int_of(v)
    } else {
        default
    }
}

/// Place `rm` on the engine-map node of `act` nearest `(col, row)`.
fn place_nearest(
    rm: &mut RunManager,
    act: i64,
    col: i64,
    row: i64,
    total_floor: i64,
) -> Result<()> {
    // The failed placement already advanced the run to `act` and generated
    // its map, so the engine's own act map is readable here.
    let full = sts2_wire::emit::run_manager::full_map(rm);
    let mut nodes: Vec<(i64, i64)> = py::list(&full, "rows")
        .iter()
        .flat_map(py::as_list)
        .map(|n| (py::int_of(py::get(n, "col")), py::int_of(py::get(n, "row"))))
        .collect();
    nodes.sort_by_key(|&(c, r)| ((r - row).abs(), (c - col).abs()));
    for (c, r) in nodes {
        if place_on_map(rm, act, c, r, Vec::new(), r + 1, total_floor).is_ok() {
            return Ok(());
        }
    }
    Err(unavailable(tr(
        "no node of the engine's map takes this co-op position",
    )))
}

/// Node types a map-less graft prefers for a room kind, best first.
fn room_node_types(room: Option<&str>) -> &'static [&'static str] {
    match room {
        Some("monster") => &["Monster", "Unknown"],
        Some("elite") => &["Elite", "Unknown"],
        _ => &[],
    }
}

/// Place `rm` on a node of `act` in the row the game `floor` stands on,
/// of the room's kind when the row has one. Returns the act-local row.
fn place_by_floor(rm: &mut RunManager, act: i64, floor: i64, room: Option<&str>) -> Result<i64> {
    let total = engine_total_floor(floor, act);
    let mut before = 0;
    let mut fm = json!({});
    for a in 1..=act {
        fm = enter_act(rm, a);
        if a < act {
            before += int_or(
                py::get(&fm, "boss"),
                "row",
                py::list(&fm, "rows").len() as i64,
            );
        }
    }
    let mut row = (total - before).max(0);
    let boss = py::get(&fm, "boss").clone();
    let boss_row = int_or(&boss, "row", py::list(&fm, "rows").len() as i64);
    let second = py::get(&fm, "second_boss").clone();
    if room == Some("boss") || row >= boss_row {
        let pt = if row > boss_row && !py::get(&second, "row").is_null() {
            &second
        } else {
            &boss
        };
        if !py::get(pt, "row").is_null() {
            let r = py::int_of(py::get(pt, "row"));
            place_on_map(
                rm,
                act,
                py::int_of(py::get(pt, "col")),
                r,
                Vec::new(),
                r + 1,
                total,
            )
            .map_err(GraftError::Value)?;
            return Ok(r);
        }
        row = boss_row - 1;
    }
    let prefer = room_node_types(room);
    let mut nodes: Vec<&Value> = py::list(&fm, "rows")
        .iter()
        .flat_map(py::as_list)
        .filter(|n| {
            let r = n.get("row").map_or(-1, py::int_of);
            r == row
        })
        .collect();
    nodes.sort_by_key(|n| {
        py::get(n, "type")
            .as_str()
            .and_then(|t| prefer.iter().position(|p| *p == t))
            .unwrap_or(prefer.len())
    });
    let picks: Vec<i64> = nodes
        .iter()
        .map(|n| py::int_of(py::get(n, "col")))
        .collect();
    for col in picks {
        if place_on_map(rm, act, col, row, Vec::new(), row + 1, total).is_ok() {
            return Ok(row);
        }
    }
    Err(unavailable(trf(
        "no node of act {act}'s map stands on floor {floor}",
        &[("act", &act), ("floor", &floor)],
    )))
}

fn new_run(snap: &RunSnapshot, state: &Value, version: &str, spoils: bool) -> Result<RunManager> {
    // A seedless rebuild cannot roll the act-1 variant from the seed: the
    // screen names it (`act1`).
    let act1 = if seedless(state) {
        state.get("act1").and_then(Value::as_str).unwrap_or("roll")
    } else {
        "roll"
    };
    let (mut rm, _) = build_run(&RunOptions {
        game_version: Some(version),
        string_seed: Some(&snap.seed),
        character: &snap.character,
        ascension_level: snap.ascension,
        neow_start: false,
        mode: Some("agent"),
        interactive_potion_swap: true,
        act1,
        ..RunOptions::default()
    })
    .map_err(|e| {
        unavailable(trf(
            "the engine cannot rebuild this run ({e})",
            &[("e", &e)],
        ))
    })?;
    if is_coop(state) {
        // The game builds co-op act maps one room shorter, from the same map
        // RNG (`ActModel.GetNumberOfRooms(isMultiplayer)`): the engine then
        // builds exactly the screen's map (19 of 19 recorded co-op acts).
        rm.debug_use_coop_maps()
            .map_err(|e| GraftError::Value(format!("debug_use_coop_maps: {e}")))?;
    }
    if spoils {
        // The Spoils Map quest card swaps act 2's map for the hourglass one
        // as the act is built (`SpoilsMap.ModifyGeneratedMap`), and the deck
        // is restored only after placement: put the card in first, or act 2
        // is built as a standard map.
        sts2_wire::graft::set_master_deck(&mut rm, &json!([{"id": "SPOILS_MAP", "upgraded": 0}]))
            .map_err(GraftError::Value)?;
    }
    Ok(rm)
}

/// A `?` node stands in the room it turned out to be (GH #1646). Placement
/// types the room from the map node, so a resolved `?` read as `unknown`,
/// while the run the student was trained on stands in the event, fight,
/// shop or treasure the `?` rolled (`current_room.room_type`, RUNCTX
/// `X_ROOM`). The mod names that room (`current_room`); a typed node is
/// left as its own room.
fn resolve_question_mark(rm: &mut RunManager, state: &Value) -> Result<()> {
    let on_question_mark = rm
        .run_state
        .current_room
        .is_some_and(|r| r.room_type == sts2_core::runstate::RoomType::Unknown);
    if let (true, Some(room)) = (on_question_mark, live_room(state)) {
        sts2_wire::graft::set_room_type(rm, room).map_err(GraftError::Value)?;
    }
    Ok(())
}

/// Rebuild the live run behind `state` (a mod message) as an engine
/// `RunManager` on game build `version`.
///
/// Returns `(rm, notes)`: the run, parked in `map_choice` at the player's
/// node, and what could not be restored.
///
/// # Errors
///
/// [`GraftError::Unavailable`]: no run to rebuild, or the engine refuses it.
#[allow(clippy::too_many_lines)]
pub fn graft_run(state: &Value, version: &str) -> Result<(RunManager, Vec<String>)> {
    let mut notes: Vec<String> = Vec::new();
    let Value::Object(msg) = state else {
        return Err(unavailable(tr(
            "no run to rebuild (the message is not a dict)",
        )));
    };
    let seedless = seedless(state);
    let mut owned;
    let msg = if seedless {
        owned = msg.clone();
        // `seedless_seed`: another placeholder, when the caller looks for dice
        // that roll what the screen shows (an event's rolled options).
        let placeholder = msg
            .get("seedless_seed")
            .and_then(Value::as_str)
            .unwrap_or(SEEDLESS_SEED)
            .to_owned();
        owned.insert("seed".into(), Value::String(placeholder));
        owned.remove("rng");
        &owned
    } else {
        msg
    };
    let snap = run_snapshot(msg, &mut notes)
        .map_err(|e| unavailable(trf("no run to rebuild ({why})", &[("why", &e.0)])))?;
    if seedless {
        // Expected, not a loss: a referee keeps its dice.
        notes.retain(|n| !n.starts_with("No RNG cursors"));
    }
    let deck = py::list(py::get(state, "run_state"), "master_deck");
    let spoils_quest = deck.iter().any(|c| {
        c.is_object() && {
            let id = py::get(c, "id");
            let id = if py::truthy(id) {
                py::str_of(id)
            } else {
                String::new()
            };
            id.to_uppercase().ends_with("SPOILS_MAP")
        }
    });
    let visited: Vec<MapCoord> = snap
        .visited()
        .into_iter()
        .map(|(c, r)| coord(i64::from(c), i64::from(r)))
        .collect();
    let total_floor = engine_total_floor(snap.floor, snap.act);

    let mut rm = new_run(&snap, state, version, spoils_quest)?;
    if seedless {
        install_screen_map(&mut rm, snap.act, state)?;
    }
    if let Some((_, 0)) = snap.coord() {
        // The act's start node (the Ancient): the engine's run stands at the
        // act entrance there, not on the node (GH #1654).
        let act = u8::try_from(snap.act)
            .map_err(|_| GraftError::Value(format!("act {} out of range", snap.act)))?;
        rm.debug_park_at_act_entrance(act, total_floor as i32)
            .map_err(|e| unavailable(trf("the map does not match this run ({e})", &[("e", &e)])))?;
        // The run's first map opens in Neow's room, and the run the student
        // was trained on is still in that event room (no node) there; a
        // later act's entrance stands in none (GH #1646).
        // The game's room there is `Map` once the map opens (GH #1664), so
        // only a room the map cannot name (`live_room`) keeps it out.
        if act == 1 && live_room(state).is_none_or(|r| r == "event") {
            sts2_wire::graft::set_room_type(&mut rm, "event").map_err(GraftError::Value)?;
        }
    } else if let Some((col, row)) = snap.coord() {
        let (col, row) = (i64::from(col), i64::from(row));
        let place = |rm: &mut RunManager| {
            place_on_map(
                rm,
                snap.act,
                col,
                row,
                visited.clone(),
                row + 1,
                total_floor,
            )
        };
        let placed: std::result::Result<(), String> = match place(&mut rm).and_then(|()| {
            if !spoils_quest
                && snap.act == SPOILS_MAP_ACT
                && !map_disagreements(&rm, &snap).is_empty()
            {
                Err("act 2's standard map is not the screen's".to_string())
            } else {
                Ok(())
            }
        }) {
            Ok(()) => Ok(()),
            Err(first) if seedless || spoils_quest || snap.act != SPOILS_MAP_ACT => Err(first),
            Err(first) => {
                // The quest is done (the card left the deck) but its act
                // keeps the hourglass map: rebuild with the card for the
                // map, and keep that only if it is the screen's.
                let mut spoils_rm = new_run(&snap, state, version, true)?;
                match place(&mut spoils_rm) {
                    Err(e) => Err(e),
                    Ok(()) if !map_disagreements(&spoils_rm, &snap).is_empty() => Err(first),
                    Ok(()) => {
                        rm = spoils_rm;
                        Ok(())
                    }
                }
            }
        };
        match placed {
            Ok(()) => {
                notes.extend(map_disagreements(&rm, &snap));
                resolve_question_mark(&mut rm, state)?;
            }
            Err(e) if !is_coop(state) => {
                return Err(unavailable(trf(
                    "the map does not match this run ({e})",
                    &[("e", &e)],
                )));
            }
            Err(_) => {
                // Co-op, and the co-op map still differs: stand on the
                // nearest node of the engine's map in the same act, so act,
                // floor and the room's place in the act stay right.
                place_nearest(&mut rm, snap.act, col, row, total_floor)?;
                notes.push(COOP_MAP_APPROX.into());
            }
        }
    } else {
        // Fight requests carry no map graph (GH #1212): place by the
        // screen's act and floor, or the student reads every fight as the
        // first room of the run.
        let row = place_by_floor(&mut rm, snap.act, snap.floor, graft_room(state))?;
        notes.push(format!("no map position: placed by floor on row {row}"));
    }

    // Deck emptied BEFORE the relics (on-obtain upgrades must not chew on
    // it), then set whole -- ids, upgrades AND enchantments -- with no hook
    // firing.
    rm.run_state.master_deck.clear();
    // The relics are set whole too. Their pickup hooks still run (relic
    // state, counters), but some rewrite the relic list itself, which the live
    // list already shows: Touch of Orobas refines the first starter-rarity
    // relic (a held Divine Destiny became Circlet; the fresh run's own
    // starter, left in place, became a second Divine Destiny). So the list is
    // put back to the live one, in its order, after the hooks.
    rm.run_state.relics.clear();
    // The map is the screen's, placed above. A pickup hook that lays out a
    // map (Golden Compass's `GoldenPathAct` is the act it is picked up in)
    // would roll it again, for the act the screen stands in.
    let placed_map = rm.run_state.current_act_map.clone();
    for relic in &snap.relics {
        sts2_wire::graft::force_add_relic(&mut rm, relic).map_err(GraftError::Value)?;
    }
    let live: Vec<_> = snap
        .relics
        .iter()
        .filter_map(|r| {
            sts2_content::names::relic_id_from_str_any_build(r.strip_prefix("RELIC.").unwrap_or(r))
        })
        .collect();
    rm.run_state
        .relic_counters
        .retain(|id, _| live.contains(id));
    // The counters the board shows (Book of Five Rings' cards toward its
    // heal), in the engine's terms. Left at the pickup value, the rebuilt run
    // fired them at other times than the game's. Taken only where the engine
    // value shows back as the same number; Toy Box and Pocketwatch keep more
    // than they show.
    for (name, shown) in &snap.relic_counters {
        let Some(id) = sts2_content::names::relic_id_from_str_any_build(name) else {
            continue;
        };
        if matches!(name.as_str(), "TOY_BOX" | "POCKETWATCH") {
            continue;
        }
        let engine = sts2_e2e::snapshot::engine_counter(name, *shown).unwrap_or(*shown);
        if sts2_e2e::snapshot::display_counter(name, engine) != Some(*shown) {
            continue;
        }
        if let Ok(v) = i32::try_from(engine) {
            rm.run_state.relic_counters.insert(id, v);
        }
    }
    // Obtaining a relic takes it out of the relic pools (C# `RelicCmd.Obtain`):
    // left in, a relic the run gives away comes back on the next pull
    // (Ranwid's relic trade, GH #1760).
    for &id in &live {
        rm.run_state.remove_relic_from_grab_bags(id);
    }
    rm.run_state.relics = live;
    // A wax relic Toy Box melted is still held but does nothing (C#
    // `RelicModel.IsMelted`, the screen's `melted`). Left active, a melted
    // Shovel or Girya added Dig / Lift the game's rest site does not show;
    // the rest-site screen then read the missing button as used this visit
    // and offered the student a Leave the game has not (Spire Bench seeds
    // 180000913, 180000818, 180000916).
    rm.run_state.melted_relics = py::list(py::get(state, "run_state"), "relics")
        .iter()
        .filter(|r| py::truthy(py::get(r, "melted")) || py::truthy(py::get(r, "is_melted")))
        .filter_map(|r| {
            let id = py::str_of(py::get(r, "id"));
            sts2_content::names::relic_id_from_str_any_build(
                id.strip_prefix("RELIC.").unwrap_or(&id),
            )
        })
        .collect();
    rm.run_state.current_act_map = placed_map;
    // Golden Compass's act, which no screen names: this one when the screen's
    // map is its single path, an earlier one otherwise (none of a later
    // act's maps is then the golden path).
    if rm
        .run_state
        .relics
        .contains(&sts2_core::RelicId::GoldenCompass)
    {
        let golden = rm
            .run_state
            .current_act_map
            .as_ref()
            .is_some_and(|m| m.points.len() > 2 && m.points.iter().all(|p| p.coord.col == 3));
        if golden {
            let act = i32::from(rm.run_state.act_index);
            rm.run_state
                .relic_counters
                .insert(sts2_core::RelicId::GoldenCompass, act);
        } else {
            rm.run_state
                .relic_counters
                .remove(&sts2_core::RelicId::GoldenCompass);
        }
    }
    // The pickup hooks queue the choices the live run already made (Lead
    // Paperweight's choose screen): left queued, the first one surfaces in
    // place of the next real screen (GH #1183).
    rm.debug_clear_pending_choices();
    let specs: Vec<Value> = deck
        .iter()
        .filter(|c| c.is_object())
        .map(card_spec)
        .collect();
    let missing = sts2_wire::graft::set_master_deck(&mut rm, &Value::Array(specs))
        .map_err(GraftError::Value)?;
    if !missing.is_empty() {
        notes.push(format!("deck cards not built: {}", missing.join(", ")));
    }
    sts2_wire::graft::set_potion_state(&mut rm, &snap.potions, &[]).map_err(GraftError::Value)?;
    // The belt the screen shows (`potion_slot_count`, else one entry per
    // slot, empty ones included). A pickup hook above may have grown it:
    // Large Capsule obtains two relics off the rebuilt run's own pulls, a
    // Potion Belt among them on some seeds (GH #1848).
    let slots = if snap.potion_slots > 0 {
        snap.potion_slots
    } else {
        py::list(py::get(state, "run_state"), "potions").len() as i64
    };
    if let Ok(slots) = i32::try_from(slots) {
        if slots > 0 {
            rm.run_state.max_potion_slots = slots;
        }
    }
    let clamp = |v: i64| i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX });
    rm.debug_set_player(
        Some(clamp(snap.hp)),
        Some(clamp(snap.max_hp)),
        Some(clamp(snap.gold)),
    );
    if let Some(rng) = &snap.rng {
        let counters: std::result::Result<Vec<(String, u32)>, String> = rng
            .counters
            .iter()
            .map(|(k, v)| {
                u32::try_from(*v)
                    .map(|v| (k.clone(), v))
                    .map_err(|_| format!("cursor {k:?} = {v} is out of range"))
            })
            .collect();
        match counters.and_then(|c| sts2_wire::graft::set_rng_counters(&mut rm, rng.run_seed, &c)) {
            Err(e) => notes.push(format!("RNG cursors not restored ({e})")),
            Ok(ignored) => {
                let run_level: Vec<String> = ignored
                    .into_iter()
                    .filter(|n| !n.starts_with("combat_"))
                    .collect();
                if !run_level.is_empty() {
                    notes.push(format!("RNG streams not carried: {}", run_level.join(", ")));
                }
            }
        }
    }
    Ok((rm, notes))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn floors_and_card_specs_read_like_python() {
        assert_eq!(engine_total_floor(17, 1), 16);
        assert_eq!(engine_total_floor(18, 2), 16);
        assert_eq!(engine_total_floor(0, 0), 0);
        assert_eq!(
            card_spec(&json!({"id": "BASH", "upgraded": true})),
            json!({"id": "BASH", "upgraded": 1})
        );
        assert_eq!(
            card_spec(&json!({"id": "X", "upgrade_level": 2, "enchantment": "SHARP"})),
            json!({"id": "X", "upgraded": 2, "enchantment": "SHARP", "enchantment_amount": 1})
        );
        assert!(is_coop(&json!({"player_count": 2})));
        assert!(!is_coop(&json!({"player_count": "x"})));
        assert_eq!(
            graft_room(&json!({"combat": {"room_type": "Elite"}})),
            Some("elite")
        );
    }

    #[test]
    fn a_message_without_a_run_is_unavailable() {
        let e = graft_run(&json!({"type": "advice_request"}), "v0.111.0").unwrap_err();
        assert!(matches!(e, GraftError::Unavailable(m) if m.starts_with("no run to rebuild")));
    }

    /// A run's act map as the agent sends it: the mod's map (flat `nodes`
    /// without the start, `current`, `visited`, `boss`) with the boss ids the
    /// screen shows.
    pub(crate) fn screen_map(rm: &RunManager) -> Value {
        let fm = sts2_wire::emit::run_manager::full_map(rm);
        let nodes: Vec<Value> = py::list(&fm, "rows")
            .iter()
            .flat_map(py::as_list)
            .filter(|n| n["row"] != json!(0))
            .map(|n| json!({"col": n["col"], "row": n["row"], "type": n["type"], "children": n["children"]}))
            .collect();
        json!({
            "nodes": nodes,
            "current": fm["current_coord"],
            "visited": [],
            "boss": fm["boss"],
            "second_boss": fm["second_boss"],
        })
    }

    pub(crate) fn reference_run(act: i64) -> RunManager {
        let (mut rm, _) = build_run(&RunOptions {
            game_version: Some("v0.111.0"),
            string_seed: Some("REFERENCE"),
            character: "Ironclad",
            ascension_level: 10,
            neow_start: false,
            mode: Some("agent"),
            interactive_potion_swap: true,
            act1: "underdocks",
            ..RunOptions::default()
        })
        .expect("reference run");
        let _ = enter_act(&mut rm, act);
        rm
    }

    pub(crate) fn seedless_message(fm: &Value, act: i64) -> Value {
        let mut fm = fm.clone();
        let first = py::list(&fm, "nodes")
            .iter()
            .find(|n| n["row"] == json!(1))
            .expect("a row-1 node")
            .clone();
        fm["current"] = json!({"col": first["col"], "row": first["row"]});
        json!({
            "seedless": true,
            "act1": "underdocks",
            "act": act,
            "floor": 1,
            "map": fm,
            "run_state": {
                "character": "IRONCLAD", "ascension": 10,
                "hp": 60, "max_hp": 80, "gold": 99,
                "master_deck": [{"id": "BASH", "upgraded": 0}],
                "relics": [{"id": "BURNING_BLOOD"}], "potions": [],
            },
        })
    }

    fn rows_of(fm: &Value) -> Value {
        let mut v: Vec<String> = py::list(fm, "nodes")
            .iter()
            .map(ToString::to_string)
            .collect();
        v.sort();
        json!(v)
    }

    /// GH #1848: the rebuilt belt is the screen's, whatever the relics'
    /// pickup hooks did. Large Capsule obtains two relics off the rebuilt
    /// run's pulls, on some placeholder seeds a Potion Belt (+2 slots).
    #[test]
    fn a_rebuild_keeps_the_screens_belt_size() {
        // The relics' pickup effects are content callbacks.
        sts2_e2e::env::ensure_engine_installed();
        let reference = screen_map(&reference_run(1));
        let mut msg = seedless_message(&reference, 1);
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "LARGE_CAPSULE"}]);
        msg["run_state"]["potions"] =
            json!([{"slot": 0, "empty": true, "id": null}, {"slot": 1, "empty": true, "id": null}]);
        for n in 1..=300 {
            msg["seedless_seed"] = json!(format!("{SEEDLESS_SEED}-v{n}"));
            let (rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
            assert_eq!(rm.run_state.max_potion_slots, 2, "placeholder v{n}");
        }
    }

    #[test]
    fn a_seedless_rebuild_stands_on_the_screens_map_with_its_boss() {
        // Another seed's map: nothing of it can come from the placeholder.
        let reference = screen_map(&reference_run(1));
        let (rm, notes) = graft_run(&seedless_message(&reference, 1), "v0.111.0").expect("graft");
        let rebuilt = screen_map(&rm);
        assert_eq!(rows_of(&rebuilt), rows_of(&reference), "notes: {notes:?}");
        assert_eq!(rebuilt["boss"]["id"], reference["boss"]["id"]);
        assert_eq!(
            rebuilt["current"],
            seedless_message(&reference, 1)["map"]["current"]
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    /// GH #1654: on an act's start node (the Ancient, row 0) the game has
    /// not entered the act's map yet, and the run the student was trained on
    /// stands at the act entrance there: no node, act floor 0, nothing
    /// visited. Placed ON the start node at act floor 1 instead, the student
    /// read an act's first map choice differently (81% vs 11% for the same
    /// column on one seed).
    #[test]
    fn on_an_acts_start_node_the_rebuild_stands_at_the_act_entrance() {
        for act in 1..=3 {
            let reference = reference_run(act);
            let start = sts2_wire::emit::run_manager::full_map(&reference)["rows"][0][0].clone();
            let mut msg = seedless_message(&screen_map(&reference), act);
            msg["map"]["current"] = json!({"col": start["col"], "row": 0});
            msg["map"]["visited"] = json!([]);
            // The game counts the Ancient as a floor; the engine does not.
            let game_floor = i64::from(reference.run_state.total_floor) + act;
            msg["floor"] = json!(game_floor);
            let (rm, notes) = graft_run(&msg, "v0.111.0").expect("graft");
            let r = &rm.run_state;
            assert_eq!(i64::from(r.act_index), act, "{notes:?}");
            assert_eq!(r.act_floor, 0, "act {act}");
            assert_eq!(r.total_floor, reference.run_state.total_floor, "act {act}");
            assert!(
                r.visited_map_coords.is_empty(),
                "act {act}: {:?}",
                r.visited_map_coords
            );
            assert!(
                r.current_room.as_ref().and_then(|x| x.map_coord).is_none(),
                "act {act}: {:?}",
                r.current_room
            );
            assert_eq!(
                rm.phase(),
                sts2_core::run_manager::Phase::MapChoice,
                "act {act}"
            );
            assert_eq!(rows_of(&screen_map(&rm)), rows_of(&msg["map"]), "act {act}");
        }
    }

    /// Golden Compass lays out only the act it was picked up in (its
    /// `GoldenPathAct`) as one path. The rebuild runs each relic's pickup
    /// hook, and that hook took the act the screen stands in, so an act-3
    /// screen got a golden path and the student no map choice at all
    /// (Spire Bench seed 170000008: "Instinct sees 1 paths, the map shows
    /// 2" on every act-3 map).
    #[test]
    fn golden_compass_leaves_a_later_acts_map_as_the_screen_shows() {
        sts2_content::relics::dispatch::install_default_registry();
        let reference = screen_map(&reference_run(3));
        let mut msg = seedless_message(&reference, 3);
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "GOLDEN_COMPASS"}]);
        let (rm, notes) = graft_run(&msg, "v0.111.0").expect("graft");
        assert_eq!(
            rows_of(&screen_map(&rm)),
            rows_of(&reference),
            "notes: {notes:?}"
        );
        let mut later = rm.run_state.clone();
        later.generate_map();
        let regen = later.current_act_map.as_ref().expect("map");
        assert!(
            regen.points.len() > 18,
            "act 3 is not the golden-path act: {} points",
            regen.points.len()
        );
    }

    #[test]
    fn the_second_boss_is_named_only_when_the_screen_names_it() {
        let reference = screen_map(&reference_run(3));
        assert!(
            reference["second_boss"]["id"].is_string(),
            "A10 act 3 has a second boss"
        );
        let mut hidden = reference.clone();
        hidden["second_boss"]
            .as_object_mut()
            .expect("node")
            .remove("id");
        let (rm, _) = graft_run(&seedless_message(&hidden, 3), "v0.111.0").expect("graft");
        let rebuilt = screen_map(&rm);
        assert_eq!(
            rebuilt["second_boss"]["row"],
            reference["second_boss"]["row"]
        );
        assert!(
            rebuilt["second_boss"].get("id").is_none(),
            "{}",
            rebuilt["second_boss"]
        );
        let (rm, _) = graft_run(&seedless_message(&reference, 3), "v0.111.0").expect("graft");
        assert_eq!(
            screen_map(&rm)["second_boss"]["id"],
            reference["second_boss"]["id"]
        );
    }

    /// GH #1646: a `?` node is the room it turned out to be. The native run
    /// stands in that room (`current_room.room_type`, the student's RUNCTX
    /// `X_ROOM`); the map only says `Unknown`, the mod's `current_room` (the
    /// game's `RoomType`) says what it became.
    #[test]
    fn a_question_mark_node_is_the_room_it_turned_out_to_be() {
        use sts2_core::runstate::RoomType;
        let reference = screen_map(&reference_run(1));
        let node = |ty: &str| {
            py::list(&reference, "nodes")
                .iter()
                .find(|n| n["type"] == json!(ty))
                .unwrap_or_else(|| panic!("a {ty} node"))
                .clone()
        };
        let room_at = |n: &Value, live: Option<&str>| {
            let mut msg = seedless_message(&reference, 1);
            msg["map"]["current"] = json!({"col": n["col"], "row": n["row"]});
            msg["floor"] = json!(py::int_of(&n["row"]) + 1);
            if let Some(live) = live {
                msg["current_room"] = json!(live);
            }
            let (rm, notes) = graft_run(&msg, "v0.111.0").expect("graft");
            let room = rm.run_state.current_room.expect("a room");
            assert_eq!(
                room.map_coord,
                Some(coord(py::int_of(&n["col"]), py::int_of(&n["row"]))),
                "{notes:?}"
            );
            room.room_type
        };
        let unknown = node("Unknown");
        assert_eq!(room_at(&unknown, Some("Event")), RoomType::Event);
        assert_eq!(room_at(&unknown, Some("Monster")), RoomType::Monster);
        assert_eq!(room_at(&unknown, Some("Treasure")), RoomType::Treasure);
        assert_eq!(room_at(&unknown, Some("Shop")), RoomType::Shop);
        // No live room (a mod before the field), or none the map can be in.
        assert_eq!(room_at(&unknown, None), RoomType::Unknown);
        assert_eq!(room_at(&unknown, Some("Map")), RoomType::Unknown);
        // A typed node is its own room whatever the message says.
        assert_eq!(room_at(&node("Monster"), Some("Event")), RoomType::Monster);
    }

    /// GH #1646: the first map of the run opens in Neow's room. The run the
    /// student was trained on is still in that event room there (no map
    /// node); the game too (`current_room` "Event"). The later acts'
    /// entrances stand in no room in the engine, whatever the game names.
    #[test]
    fn the_first_map_of_the_run_stands_in_neows_room() {
        use sts2_core::runstate::RoomType;
        for act in 1..=2 {
            let reference = reference_run(act);
            let start = sts2_wire::emit::run_manager::full_map(&reference)["rows"][0][0].clone();
            let mut msg = seedless_message(&screen_map(&reference), act);
            msg["map"]["current"] = json!({"col": start["col"], "row": 0});
            msg["map"]["visited"] = json!([]);
            msg["floor"] = json!(i64::from(reference.run_state.total_floor) + act);
            msg["current_room"] = json!("Event");
            let (rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
            let room = rm.run_state.current_room;
            if act == 1 {
                let room = room.expect("Neow's room");
                assert_eq!((room.room_type, room.map_coord), (RoomType::Event, None));
            } else {
                assert!(room.is_none(), "act {act}: {room:?}");
            }
        }
    }

    /// GH #1664: once the map opens, the game's room is `Map`, not Neow's
    /// `Event`; the run the student was trained on is still in Neow's room.
    #[test]
    fn the_first_map_stands_in_neows_room_while_the_game_names_the_map() {
        use sts2_core::runstate::RoomType;
        let reference = reference_run(1);
        let start = sts2_wire::emit::run_manager::full_map(&reference)["rows"][0][0].clone();
        let mut msg = seedless_message(&screen_map(&reference), 1);
        msg["map"]["current"] = json!({"col": start["col"], "row": 0});
        msg["map"]["visited"] = json!([]);
        msg["floor"] = json!(i64::from(reference.run_state.total_floor) + 1);
        msg["current_room"] = json!("Map");
        let (rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
        let room = rm.run_state.current_room.expect("Neow's room");
        assert_eq!((room.room_type, room.map_coord), (RoomType::Event, None));
    }

    /// GH #1646: a relic only a later build has (v0.111.0's Dowsing Rod,
    /// v0.113.0's Wuppy) is held by the rebuilt run. The relic names
    /// resolved against the v0.107.1 universe only, so the graft dropped it
    /// ("this engine does not model") and the student read one relic token
    /// fewer than the native run for the rest of the run.
    #[test]
    fn a_relic_of_a_later_build_survives_the_graft() {
        for (version, relic) in [("v0.111.0", "DOWSING_ROD"), ("v0.113.0", "WUPPY")] {
            let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
            msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": relic}]);
            let (rm, notes) = graft_run(&msg, version).expect("graft");
            let held: Vec<&str> = rm
                .run_state
                .relics
                .iter()
                .map(|r| r.to_python_name())
                .collect();
            assert_eq!(held, ["BURNING_BLOOD", relic], "{version}: {notes:?}");
            assert!(
                !notes.iter().any(|n| n.contains("does not model")),
                "{notes:?}"
            );
        }
    }

    /// A relic's counter the board shows is the rebuilt run's: Book of Five
    /// Rings at 4 heals on the next card added, as the game's does (Spire
    /// Bench seed 170000009: Grave of the Forgotten's Decay was the fifth
    /// card). A relic the board shows no counter for keeps its pickup value.
    #[test]
    fn a_relic_counter_the_board_shows_carries_over() {
        use sts2_core::RelicId;
        let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
        msg["run_state"]["relics"] = json!([
            {"id": "BURNING_BLOOD"},
            {"id": "BOOK_OF_FIVE_RINGS", "counter": 4},
            {"id": "SWORD_OF_STONE", "counter": 2},
        ]);
        let (rm, notes) = graft_run(&msg, "v0.111.0").expect("graft");
        let c = &rm.run_state.relic_counters;
        assert_eq!(c.get(&RelicId::BookOfFiveRings), Some(&4), "{notes:?}");
        // The game counts Sword of Stone's elites up, the engine down.
        assert_eq!(c.get(&RelicId::SwordOfStone), Some(&3), "{notes:?}");
    }

    /// GH #1760: a held relic is out of the rebuilt run's relic pools.
    #[test]
    fn a_held_relic_is_out_of_the_relic_pools() {
        use sts2_core::RelicId;
        let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "BAG_OF_PREPARATION"}]);
        let (rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
        let rs = &rm.run_state;
        let pooled = rs.relic_grab_bag.contains(&RelicId::BagOfPreparation)
            || rs
                .relic_grab_bag_by_rarity
                .values()
                .any(|b| b.contains(&RelicId::BagOfPreparation));
        assert!(!pooled, "a held relic stays pooled");
    }

    /// A melted wax relic does nothing in the rebuilt run: a melted Shovel
    /// adds no Dig to the rest site (Spire Bench seed 180000913).
    #[test]
    fn a_melted_wax_relic_is_inert_in_the_rebuild() {
        sts2_e2e::env::ensure_engine_installed();
        let dig = |melted: bool| {
            let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
            msg["run_state"]["relics"] = json!([
                {"id": "BURNING_BLOOD"},
                {"id": "SHOVEL", "melted": melted},
            ]);
            let (mut rm, _) = graft_run(&msg, "v0.111.0").expect("graft");
            sts2_wire::graft::enter_rest_site(&mut rm, &[])
                .expect("rest site")
                .iter()
                .any(|id| id.eq_ignore_ascii_case("dig"))
        };
        assert!(dig(false), "a live Shovel digs");
        assert!(!dig(true), "a melted Shovel does not");
    }

    #[test]
    fn a_seedless_rebuild_needs_the_map() {
        let mut msg = seedless_message(&screen_map(&reference_run(1)), 1);
        msg.as_object_mut().expect("msg").remove("map");
        let e = graft_run(&msg, "v0.111.0").unwrap_err();
        assert!(matches!(e, GraftError::Unavailable(m) if m.contains("act map")));
    }
}
