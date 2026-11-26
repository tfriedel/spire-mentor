//! Parse a state pasted out of the live game into a position the browser
//! workbench can rebuild — the port of `python/workbench/state_import.py`.
//!
//! The SpireMentor mod stamps every outbound message with the whole run: the
//! string seed, the act and floor, the full `run_state` (deck / relics /
//! potions / gold / HP / ascension / character), the act map graph including
//! its visited coords and the current node, and `rng` — the run's RNG cursor
//! for every stream. That is enough to rebuild the run position; this module
//! is the plumbing.
//!
//! What comes in — deliberately tolerant, because the JSON reaches the user
//! by whatever route is to hand: the mod's "Copy state" payload (an
//! `advice_request`), any raw wire message, one record out of an advisor log
//! (`{step, ts, type, data}`), or a whole pasted JSONL blob (the newest
//! record carrying a state wins).
//!
//! What comes out — a [`RunSnapshot`] for the run lane, and, when the paste
//! was taken during a fight, the fight itself ([`RunSnapshot::combat`]: the
//! engine wire dict `crate::combat_import` builds from the mod's board, the
//! port of the desktop's `realgame_adapter` + `sts2_hint.adapter` chain).
//! The combat lane grafts that board onto the rebuilt run (GH #708). A board
//! the adapter cannot translate (an unknown monster, a mid-play capture)
//! still imports as the run position, and the warning says why.
//!
//! Nothing is guessed. A field the paste does not carry becomes a warning,
//! because a half-restored state that *looks* whole is the expensive failure
//! here — an import with no `rng` block shares only its seed, and every
//! downstream roll belongs to a different run.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::i18n::{tr, trf};

/// The warning for a fight the adapter refused (`{error}`: why). SimLab
/// picks it out of the warnings by this template.
pub(crate) const FIGHT_NOT_REBUILT: &str =
    "The fight could not be rebuilt ({error}). Copy at a decision point rather than mid-animation.";
/// The warnings for what the run lane left out; SimLab carries them with the
/// board (matched by template, in either language).
pub(crate) const LEFT_OUT: [&str; 3] = [
    "{n} deck card(s) this engine does not know were left out: {list}",
    "{n} relic(s) this engine does not model were left out: {list}",
    "{n} potion(s) this engine does not model were left out: {list}",
];

/// A pasted state the importer cannot turn into a position. The message is
/// shown to the user verbatim, so it says what is wrong with the paste and
/// what to paste instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateImportError(pub String);

impl std::fmt::Display for StateImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StateImportError {}

fn err<T>(msg: impl Into<String>) -> Result<T, StateImportError> {
    Err(StateImportError(msg.into()))
}

/// Wire message types that describe a decision the player was facing.
/// Anything else (`pong`, `error`, `run_complete`) is not a position.
const DECISION_SURFACES: &[&str] = &[
    "combat_action",
    "map_select",
    "card_select",
    "card_reward",
    "card_bundle",
    "crystal_sphere",
    "rest_site",
    "shop",
    "event",
    "treasure",
    "boss_relic",
    "reward_screen",
];

/// Log-envelope keys that mark a `{step, ts, type, data}` wrapper (the
/// advisor's `AdvisorLog` and the workbench's `GameLogger` share it).
const LOG_ENVELOPE_MARKERS: &[&str] = &["step", "ts"];

/// `run_state.character` (`"IRONCLAD"`) → the display name `RunManager` wants.
const CHARACTER_ID_TO_DISPLAY: &[(&str, &str)] = &[
    ("IRONCLAD", "Ironclad"),
    ("SILENT", "Silent"),
    ("DEFECT", "Defect"),
    ("NECROBINDER", "Necrobinder"),
    ("REGENT", "Regent"),
];

fn character_display_name(character_id: &str) -> Option<&'static str> {
    let up = character_id.to_uppercase();
    CHARACTER_ID_TO_DISPLAY
        .iter()
        .find(|(id, _)| *id == up)
        .map(|(_, name)| *name)
}

/// `{run_seed, counters}` off the envelope — the run's RNG cursor per stream,
/// in the wire's own order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngBlock {
    /// `RunRngSet.Seed`: a `uint` in v0.107.1, a `ulong` in v0.111.0.
    pub run_seed: u64,
    pub counters: Vec<(String, i64)>,
}

/// The fight a paste carried, ready for the engine: the wire dict
/// `sts2_wire::state_from_json` rebuilds the `CombatState` from, and the
/// room kind the mod reported (`"Combat"` / `"Elite"` / `"Boss"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedCombat {
    pub wire: Value,
    pub room_type: String,
}

impl ImportedCombat {
    /// The engine's room kind for the post-combat reward chain.
    #[must_use]
    pub fn room_kind(&self) -> sts2_core::runstate::RoomType {
        match self.room_type.as_str() {
            "Elite" => sts2_core::runstate::RoomType::Elite,
            "Boss" => sts2_core::runstate::RoomType::Boss,
            _ => sts2_core::runstate::RoomType::Monster,
        }
    }
}

/// The run-level position, normalized to ids the engine registry knows.
///
/// Serializable so a session can persist it: `seek`, a reload and a named
/// save all rebuild an imported run through the graft, not as a fresh run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub seed: String,
    /// Display name (`"Ironclad"`).
    pub character: String,
    /// Wire id (`"IRONCLAD"`) — what the mod sent.
    pub character_id: String,
    pub ascension: u8,
    pub act: i64,
    pub floor: i64,
    pub hp: i64,
    pub max_hp: i64,
    pub gold: i64,
    /// `(registry_card_id, upgraded)` per master-deck card, in engine order.
    pub deck: Vec<(String, bool)>,
    pub relics: Vec<String>,
    /// `(registry relic id, counter)` for each relic the board shows a
    /// counter on, the game's number (`counter`, C# `DisplayAmount`).
    pub relic_counters: Vec<(String, i64)>,
    pub potions: Vec<String>,
    pub potion_slots: i64,
    /// The act map graph as the mod serialized it, or `None`.
    pub map: Option<Value>,
    pub rng: Option<RngBlock>,
    /// The fight to graft onto the run (the combat lane), or `None`: a paste
    /// off combat, a board the adapter refused, or the run lane chosen. Absent
    /// on persisted snapshots from before GH #708.
    #[serde(default)]
    pub combat: Option<ImportedCombat>,
}

impl RunSnapshot {
    /// The node the player is standing on as `(col, row)`, if known.
    #[must_use]
    pub fn coord(&self) -> Option<(i32, i32)> {
        let cur = self.map.as_ref()?.get("current")?.as_object()?;
        Some((int_field(cur, "col")?, int_field(cur, "row")?))
    }

    /// Every coord this run has entered this act, in the world's own order.
    #[must_use]
    pub fn visited(&self) -> Vec<(i32, i32)> {
        let Some(raw) = self
            .map
            .as_ref()
            .and_then(|m| m.get("visited"))
            .and_then(Value::as_array)
        else {
            return Vec::new();
        };
        raw.iter()
            .filter_map(Value::as_object)
            .filter_map(|e| Some((int_field(e, "col")?, int_field(e, "row")?)))
            .collect()
    }
}

fn int_field(m: &Map<String, Value>, key: &str) -> Option<i32> {
    // JSON integers only — a bool is not a coord (serde keeps them apart).
    m.get(key)?.as_i64().and_then(|v| i32::try_from(v).ok())
}

/// One pasted position: the run, and the fight if there was one.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedState {
    /// The decision the player was facing (`"combat_action"`, `"shop"`, …).
    pub surface: String,
    /// The run; `run.combat` is the fight when the paste carried one the
    /// adapter could translate.
    pub run: RunSnapshot,
    /// The paste carried a combat board (whether or not it translated — a
    /// refused board is a warning, and `has_combat()` is then false).
    pub had_combat: bool,
    /// Which paste shape was recognized — shown back to the user.
    pub source: String,
    pub warnings: Vec<String>,
}

impl ImportedState {
    /// Whether the combat lane is available: the fight translated.
    #[must_use]
    pub fn has_combat(&self) -> bool {
        self.run.combat.is_some()
    }

    /// What this import reproduces exactly, and what it does not. The UI
    /// renders this next to the board; every claim is one the graft can
    /// stand behind.
    #[must_use]
    pub fn fidelity(&self) -> Value {
        let mut exact = vec![
            tr("seed"),
            tr("character"),
            tr("ascension"),
            tr("deck"),
            tr("relics"),
            tr("potions"),
            tr("gold"),
            tr("hp"),
        ];
        if self.run.map.is_some() {
            exact.push(tr("map + position"));
        }
        if self.has_combat() {
            exact.push(tr("combat board (hand, piles, powers, intents)"));
        }
        json!({
            "exact": exact,
            "rng_restored": self.run.rng.is_some(),
            "warnings": self.warnings.clone(),
            "surface": self.surface,
            "source": self.source,
        })
    }

    /// The preview the import dialog shows before the user commits — which
    /// run this is, where it is, and what will and will not reproduce.
    #[must_use]
    pub fn preview(&self) -> Value {
        let r = &self.run;
        let has_combat = self.has_combat();
        json!({
            "surface": self.surface,
            "source": self.source,
            "has_combat": has_combat,
            "lanes": if has_combat { json!(["combat", "run"]) } else { json!(["run"]) },
            "run": {
                "seed": r.seed,
                "character": r.character,
                "ascension": r.ascension,
                "act": r.act,
                "floor": r.floor,
                "hp": r.hp,
                "max_hp": r.max_hp,
                "gold": r.gold,
                "deck_size": r.deck.len(),
                "relics": r.relics.clone(),
                "potions": r.potions.clone(),
                "coord": r.coord().map_or(Value::Null, |(c, row)| json!([c, row])),
            },
            "fidelity": self.fidelity(),
        })
    }
}

// ── paste sniffing ─────────────────────────────────────────────────────────

/// Every JSON value the paste contains, oldest first. One object, a JSON
/// array, or a JSONL blob — the three shapes a copied log arrives in.
/// Unparseable lines are skipped rather than fatal: a paste that clipped a
/// line in half at one end should still import.
fn load_candidates(text: &str) -> Result<Vec<Value>, StateImportError> {
    let stripped = text.trim();
    if stripped.is_empty() {
        return err(tr("Nothing to import — the paste is empty."));
    }
    if let Ok(value) = serde_json::from_str::<Value>(stripped) {
        return Ok(match value {
            Value::Array(items) => items,
            other => vec![other],
        });
    }
    let mut out = Vec::new();
    for line in stripped.lines() {
        let line = line.trim().trim_end_matches(',');
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            out.push(v);
        }
    }
    if out.is_empty() {
        return err(tr(
            "That does not look like JSON. Paste one wire message, one log line, or a whole JSONL log.",
        ));
    }
    Ok(out)
}

/// Strip a log envelope off `value`, or `None` if it is not a message.
/// `AdvisorLog` and `GameLogger` both write `{step, ts, type, data}`; the
/// message is `data`. Nested wrappers are peeled but never more than a couple
/// of levels — a deeper nest is a different file format.
fn unwrap(value: &Value) -> Option<&Map<String, Value>> {
    let mut cur = value;
    for _ in 0..3 {
        let obj = cur.as_object()?;
        let has_marker = LOG_ENVELOPE_MARKERS.iter().any(|k| obj.contains_key(*k));
        match obj.get("data") {
            Some(payload) if payload.is_object() && has_marker => cur = payload,
            _ => return Some(obj),
        }
    }
    cur.as_object()
}

/// The decision `msg` describes, or `None` if it describes none.
/// `advice_request` wraps another message type and names it in `advice_for`.
fn surface_of(msg: &Map<String, Value>) -> Option<String> {
    if let Some(advice_for) = msg.get("advice_for").and_then(Value::as_str) {
        if !advice_for.is_empty() {
            return Some(advice_for.to_string());
        }
    }
    let kind = msg.get("type").and_then(Value::as_str)?;
    if DECISION_SURFACES.contains(&kind) || kind == "run_snapshot" {
        return Some(kind.to_string());
    }
    None
}

/// Is `value` already an `sts2_rs.state_to_dict` wire dict? The tell is the
/// nested `deck` object: the mod's dialect keeps its piles at the top level.
fn is_engine_wire(value: &Value) -> bool {
    value.get("deck").is_some_and(Value::is_object)
        && value.get("player").is_some_and(Value::is_object)
}

fn is_headless_combat_play(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("decision")
        && value.get("decision").and_then(Value::as_str) == Some("combat_play")
}

/// `_combat_source`: the `combat_action`-shaped payload inside `msg`, if it
/// has one — the message itself, or the board a mid-play `card_select`
/// embeds (`SimCombatHandler.BuildEmbeddedCombatState`).
fn combat_source(msg: &Map<String, Value>, surface: &str) -> Option<Value> {
    if surface == "combat_action" {
        return Some(Value::Object(msg.clone()));
    }
    msg.get("combat").filter(|v| v.is_object()).cloned()
}

/// `_combat_wire`: translate the live board to the engine wire, or warn and
/// give up — losing the fight is worth saying out loud, never a crash.
fn combat_wire(
    board: &Value,
    run: &RunSnapshot,
    warnings: &mut Vec<String>,
) -> Option<ImportedCombat> {
    match crate::combat_import::combat_wire_from_bridge(board, &run.character, run.ascension) {
        Ok((wire, notes)) => {
            warnings.extend(notes);
            let room_type = board
                .get("room_type")
                .and_then(Value::as_str)
                .unwrap_or("Combat")
                .to_string();
            Some(ImportedCombat { wire, room_type })
        }
        Err(e) => {
            warnings.push(trf(FIGHT_NOT_REBUILT, &[("error", &e)]));
            None
        }
    }
}

// ── wire → registry ids ─────────────────────────────────────────────────────

/// `(registry_id, upgraded)` per master-deck card. Cards the engine has no
/// entry for are dropped WITH a warning: a deck one card short changes every
/// draw, and a counterfactual on it answers a different question.
fn deck_from_wire(cards: &[Value], warnings: &mut Vec<String>) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (i, raw) in cards.iter().enumerate() {
        let Some(obj) = raw.as_object() else { continue };
        let raw_id = obj.get("id").and_then(Value::as_str).unwrap_or("").trim();
        if raw_id.is_empty() {
            dropped.push(format!("#{i}"));
            continue;
        }
        // An id may spell its own upgrade (`STRIKE_IRONCLAD+1`). Fold the
        // suffix into the flag; `force_add_card` takes the base id.
        let (base, plus_upgraded) = match raw_id.split_once('+') {
            Some((b, _)) => (b, true),
            None => (raw_id, false),
        };
        let Some(cid) = sts2_content::names::card_id_from_str(base) else {
            dropped.push(raw_id.to_string());
            continue;
        };
        let upgraded = plus_upgraded
            || obj
                .get("upgrade_level")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                > 0
            || obj
                .get("upgraded")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        out.push((cid.to_python_name().to_string(), upgraded));
    }
    if !dropped.is_empty() {
        dropped.sort();
        dropped.dedup();
        let head: Vec<&str> = dropped.iter().take(6).map(String::as_str).collect();
        warnings.push(trf(
            LEFT_OUT[0],
            &[("n", &dropped.len()), ("list", &head.join(", "))],
        ));
    }
    out
}

/// Registry ids off the wire's own `id` field, dropping what we lack. `id` is
/// the canonical `Id.Entry` for both relics and potions.
/// The counters the board shows on its relics, by registry id.
fn relic_counters_from_wire(entries: &[Value]) -> Vec<(String, i64)> {
    entries
        .iter()
        .filter_map(|raw| {
            let counter = raw.get("counter").filter(|v| !v.is_boolean())?.as_i64()?;
            let id = raw.get("id").and_then(Value::as_str)?.trim();
            let id = id.strip_prefix("RELIC.").unwrap_or(id);
            let relic = sts2_content::names::relic_id_from_str_any_build(id)?;
            Some((relic.to_python_name().to_string(), counter))
        })
        .collect()
}

fn ids_from_wire(
    entries: &[Value],
    template: &'static str,
    resolve: impl Fn(&str) -> Option<String>,
    warnings: &mut Vec<String>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for raw in entries {
        let Some(obj) = raw.as_object() else { continue };
        if obj.get("empty").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        // The registry id (`entry`) where the producer sends one beside a
        // display-name `id`: the names do not all resolve ("Clarity
        // Extract" is CLARITY; GH #1679).
        let raw_id = obj
            .get("entry")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .or_else(|| obj.get("id").and_then(Value::as_str))
            .unwrap_or("")
            .trim();
        if raw_id.is_empty() {
            continue;
        }
        match resolve(raw_id) {
            Some(id) => out.push(id),
            None => dropped.push(raw_id.to_uppercase()),
        }
    }
    if !dropped.is_empty() {
        dropped.sort();
        dropped.dedup();
        warnings.push(trf(
            template,
            &[("n", &dropped.len()), ("list", &dropped.join(", "))],
        ));
    }
    out
}

/// `{run_seed, counters}` off the envelope, or `None`. Forwarded
/// all-or-nothing: a seed without cursors rebuilds every stream at count 0,
/// which destroys more lineage than it restores.
fn rng_block(msg: &Map<String, Value>) -> Option<RngBlock> {
    let raw = msg.get("rng")?.as_object()?;
    let run_seed = match raw.get("run_seed") {
        // A v0.111.0 master seed is a full `ulong` (GH #788). A negative
        // integer is a 32-bit seed written signed: its `uint` view.
        #[allow(clippy::cast_sign_loss)]
        Some(Value::Number(n)) if n.is_u64() || n.is_i64() => match n.as_u64() {
            Some(u) => u,
            None => u64::from(n.as_i64()? as u32),
        },
        _ => return None,
    };
    let counters_obj = raw.get("counters")?.as_object()?;
    let mut counters: Vec<(String, i64)> = Vec::new();
    for (k, v) in counters_obj {
        // Integers only — a JSON bool is not a counter.
        if let Value::Number(n) = v {
            if let Some(i) = n.as_i64() {
                counters.push((k.clone(), i));
            }
        }
    }
    if counters.is_empty() {
        return None;
    }
    counters.sort();
    Some(RngBlock { run_seed, counters })
}

pub fn run_snapshot(
    msg: &Map<String, Value>,
    warnings: &mut Vec<String>,
) -> Result<RunSnapshot, StateImportError> {
    let rs = msg
        .get("run_state")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            StateImportError(
                tr("This message carries no run_state, so there is no run to rebuild. Copy a state while a run is in progress.")
                    .into(),
            )
        })?;
    let character_id = rs.get("character").and_then(Value::as_str).unwrap_or("");
    if character_id.is_empty() {
        return err(tr(
            "This message names no character, so it was copied outside a run.",
        ));
    }
    let Some(display) = character_display_name(character_id) else {
        return err(trf(
            "Unknown character {character} — this engine has no port.",
            &[("character", &format!("{character_id:?}"))],
        ));
    };
    let seed = msg.get("seed").and_then(Value::as_str).unwrap_or("").trim();
    if seed.is_empty() {
        return err(tr(
            "This message carries no run seed, so the map and every roll would belong to a different run.",
        ));
    }

    let rng = rng_block(msg);
    if rng.is_none() {
        warnings.push(
            tr("No RNG cursors in this paste — the map and deck are exact, but the next card reward, shop and event will be this engine's rolls, not the live run's.")
                .into(),
        );
    }
    let map = match msg.get("map") {
        Some(Value::Object(m)) if m.get("error").is_none() => Some(Value::Object(m.clone())),
        _ => {
            warnings.push(
                tr("No map graph in this paste — the run will start at the act entrance instead of where you were standing.")
                    .into(),
            );
            None
        }
    };

    let empty: Vec<Value> = Vec::new();
    let arr = |key: &str| -> &[Value] {
        rs.get(key)
            .and_then(Value::as_array)
            .map_or(&empty[..], |v| &v[..])
    };
    let uint = |m: &Map<String, Value>, key: &str| m.get(key).and_then(Value::as_i64).unwrap_or(0);

    Ok(RunSnapshot {
        seed: seed.to_string(),
        character: display.to_string(),
        character_id: character_id.to_string(),
        ascension: u8::try_from(uint(rs, "ascension").max(0)).unwrap_or(0),
        act: msg.get("act").and_then(Value::as_i64).unwrap_or(1),
        floor: msg.get("floor").and_then(Value::as_i64).unwrap_or(0),
        hp: uint(rs, "hp"),
        max_hp: uint(rs, "max_hp"),
        gold: uint(rs, "gold"),
        deck: deck_from_wire(arr("master_deck"), warnings),
        combat: None,
        relics: ids_from_wire(
            arr("relics"),
            LEFT_OUT[1],
            |s| {
                sts2_content::names::relic_id_from_str_any_build(
                    s.strip_prefix("RELIC.").unwrap_or(s),
                )
                .map(|r| r.to_python_name().to_string())
            },
            warnings,
        ),
        relic_counters: relic_counters_from_wire(arr("relics")),
        potions: ids_from_wire(
            arr("potions"),
            LEFT_OUT[2],
            |s| sts2_content::names::potion_kind_from_str(s).map(|p| p.python_name().to_string()),
            warnings,
        ),
        potion_slots: uint(rs, "potion_slot_count"),
        map,
        rng,
    })
}

// ── entry point ──────────────────────────────────────────────────────────────

/// Turn pasted JSON into a run position the workbench can rebuild.
///
/// # Errors
/// [`StateImportError`] when the paste is not JSON, carries no run, or carries
/// no decision this importer recognizes. The message is user-facing.
pub fn parse_pasted_state(text: &str) -> Result<ImportedState, StateImportError> {
    let candidates = load_candidates(text)?;

    // Newest-first: a pasted log's LAST state is the one on screen.
    let mut last_error: Option<StateImportError> = None;
    for value in candidates.iter().rev() {
        if is_engine_wire(value) {
            return err(tr(
                "That is an engine wire dict, not a live-game state. It has no run attached (no seed, deck or map), so there is no run to rebuild — copy the state from the mod instead.",
            ));
        }
        let Some(msg) = unwrap(value) else { continue };
        if is_headless_combat_play(&Value::Object(msg.clone())) {
            return err(tr(
                "That is a headless combat_play decision, which carries no run (no seed, deck or map). Copy the state from the mod instead.",
            ));
        }
        let Some(surface) = surface_of(msg) else {
            continue;
        };
        let mut warnings: Vec<String> = Vec::new();
        let run = match run_snapshot(msg, &mut warnings) {
            Ok(r) => r,
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        };
        let mut run = run;
        let board = combat_source(msg, &surface);
        let had_combat = board.is_some();
        if let Some(board) = &board {
            run.combat = combat_wire(board, &run, &mut warnings);
        }
        let source = if msg.get("advice_for").is_some() {
            "advice_request".to_string()
        } else {
            msg.get("type")
                .and_then(Value::as_str)
                .unwrap_or("wire message")
                .to_string()
        };
        return Ok(ImportedState {
            surface,
            run,
            had_combat,
            source,
            warnings,
        });
    }

    if let Some(e) = last_error {
        return Err(e);
    }
    err(tr(
        "No game state in that paste. Expected a SpireMentor wire message (advice_request / combat_action / a screen), an advisor log line, or a whole advisor log.",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every count stands, a melted Kusarigama's too (the engine keeps it).
    #[test]
    fn a_melted_per_turn_count_stands() {
        let got = relic_counters_from_wire(&[
            json!({"id": "KUSARIGAMA", "counter": 2, "is_wax": true, "is_melted": true}),
            json!({"id": "ORNAMENTAL_FAN", "counter": 1}),
            json!({"id": "JOSS_PAPER", "counter": 4, "is_wax": true, "is_melted": true}),
        ]);
        assert_eq!(
            got,
            vec![
                ("KUSARIGAMA".to_string(), 2),
                ("ORNAMENTAL_FAN".to_string(), 1),
                ("JOSS_PAPER".to_string(), 4),
            ]
        );
    }

    fn wire() -> Value {
        json!({
            "type": "advice_request",
            "advice_for": "combat_action",
            "seed": "42",
            "act": 1,
            "floor": 3,
            "combat": {"hand": []},
            "run_state": {
                "character": "IRONCLAD",
                "ascension": 0,
                "hp": 55,
                "max_hp": 80,
                "gold": 217,
                "potion_slot_count": 3,
                "master_deck": [
                    {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                    {"id": "STRIKE_IRONCLAD", "upgrade_level": 1},
                    {"id": "BASH", "upgrade_level": 0},
                ],
                "relics": [{"id": "BURNING_BLOOD"}, {"id": "ANCHOR"}],
                "potions": [{"slot": 0, "id": "FIRE_POTION"}, {"slot": 1, "empty": true}],
            },
        })
    }

    /// A real (minimal) board, as the mod's `combat_action` carries it.
    fn board() -> Value {
        json!({
            "round": 1,
            "player": {"hp": 55, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
            "hand": [{"id": "STRIKE_IRONCLAD", "name": "Strike", "type": "Attack", "cost": 1, "effective_cost": 1}],
            "draw_pile": {"cards": [{"id": "BASH", "type": "Attack", "cost": 2, "effective_cost": 2}]},
            "discard_pile": {"cards": []},
            "exhaust_pile": {"cards": []},
            "enemies": [{"id": "NIBBIT", "name": "Nibbit", "is_alive": true, "hp": 20, "max_hp": 30,
                "block": 0, "powers": [], "next_move_id": "BITE_MOVE",
                "intents": [{"intent_type": "Attack", "damage": 6}]}],
            "potions": [{"slot": 0, "id": "FIRE_POTION", "name": "Fire Potion"}, {"slot": 1, "empty": true}],
        })
    }

    /// GH #1679: the headless host names a potion by its title, which does
    /// not always resolve; its registry `entry` does.
    #[test]
    fn a_potion_resolves_by_its_registry_entry_over_its_title() {
        let mut msg = wire();
        msg["run_state"]["potions"] =
            json!([{"slot": 0, "id": "Clarity Extract", "entry": "CLARITY"}]);
        let mut warnings = Vec::new();
        let snap = run_snapshot(msg.as_object().unwrap(), &mut warnings).unwrap();
        assert_eq!(snap.potions, vec!["CLARITY".to_string()], "{warnings:?}");
    }

    /// A potion only a later build has (v0.111.0's Ambergris, which Neow's
    /// Sacrifice gives; the Fool's v0.113.0 potions) is held, not left out
    /// (Spire Bench seeds 170000047 and 170000058).
    #[test]
    fn a_later_builds_potion_is_held() {
        let mut msg = wire();
        let fool = sts2_core::potions::PotionKind::V0113_ONLY[0].python_name();
        msg["run_state"]["potions"] = json!([
            {"slot": 0, "id": "Ambergris", "entry": "AMBERGRIS"},
            {"slot": 1, "id": fool, "entry": fool},
        ]);
        let mut warnings = Vec::new();
        let snap = run_snapshot(msg.as_object().unwrap(), &mut warnings).unwrap();
        assert_eq!(
            snap.potions,
            vec!["AMBERGRIS".to_string(), fool.to_string()],
            "{warnings:?}"
        );
    }

    #[test]
    fn a_combat_paste_carries_the_fight_and_offers_both_lanes() {
        let mut msg = wire();
        msg["combat"] = board();
        msg["advice_for"] = json!("card_select");
        let imp = parse_pasted_state(&msg.to_string()).unwrap();
        assert!(imp.had_combat);
        assert!(imp.has_combat(), "{:?}", imp.warnings);
        let combat = imp.run.combat.as_ref().unwrap();
        assert_eq!(combat.wire["player"]["hp"], 55);
        assert_eq!(combat.wire["encounter"][0]["kind"], "Nibbit");
        assert_eq!(combat.room_type, "Combat");
        let p = imp.preview();
        assert_eq!(p["has_combat"], json!(true));
        assert_eq!(p["lanes"], json!(["combat", "run"]));
        assert!(imp.fidelity()["exact"]
            .as_array()
            .unwrap()
            .contains(&json!("combat board (hand, piles, powers, intents)")));
        // A `combat_action` message IS the board.
        let mut top = wire();
        for (k, v) in board().as_object().unwrap() {
            top[k] = v.clone();
        }
        top["type"] = json!("combat_action");
        top.as_object_mut().unwrap().remove("advice_for");
        top.as_object_mut().unwrap().remove("combat");
        let imp = parse_pasted_state(&top.to_string()).unwrap();
        assert_eq!(imp.source, "combat_action");
        assert!(imp.has_combat(), "{:?}", imp.warnings);
    }

    #[test]
    fn parses_a_combat_paste_as_a_run_and_says_why_the_fight_did_not_rebuild() {
        // `wire()` embeds an empty board (no player): the fight is refused
        // with a reason, the run lane still imports.
        let imp = parse_pasted_state(&wire().to_string()).unwrap();
        assert_eq!(imp.surface, "combat_action");
        assert_eq!(imp.source, "advice_request");
        assert!(imp.had_combat);
        assert!(!imp.has_combat());
        assert_eq!(imp.run.character, "Ironclad");
        assert_eq!(imp.run.seed, "42");
        assert_eq!(imp.run.hp, 55);
        assert_eq!(imp.run.gold, 217);
        // three deck cards, the second upgraded
        assert_eq!(imp.run.deck.len(), 3);
        assert!(imp.run.deck[1].1, "the +1 Strike is upgraded");
        assert!(!imp.run.deck[0].1);
        assert!(
            imp.warnings
                .iter()
                .any(|w| w.contains("could not be rebuilt")),
            "{:?}",
            imp.warnings
        );
        let p = imp.preview();
        assert_eq!(p["has_combat"], json!(false));
        assert_eq!(p["lanes"], json!(["run"]));
    }

    #[test]
    fn says_what_it_cannot_reproduce() {
        let imp = parse_pasted_state(&wire().to_string()).unwrap();
        assert!(imp.run.rng.is_none());
        let joined = imp.warnings.join(" ");
        assert!(joined.contains("RNG cursors"), "{joined}");
        assert!(joined.contains("map graph"), "{joined}");
        let f = imp.fidelity();
        assert_eq!(f["rng_restored"], json!(false));
        assert!(!f["exact"]
            .as_array()
            .unwrap()
            .contains(&json!("map + position")));
    }

    #[test]
    fn unwraps_a_log_record() {
        let rec = json!({"step": 12, "ts": 1787796140.7, "type": "state", "data": wire()});
        let imp = parse_pasted_state(&rec.to_string()).unwrap();
        assert_eq!(imp.surface, "combat_action");
    }

    #[test]
    fn a_pasted_log_takes_the_newest_position() {
        let early = wire();
        let mut late = wire();
        late["run_state"]["hp"] = json!(41);
        let blob = format!(
            "{}\n{}",
            json!({"step": 0, "ts": 0.0, "type": "state", "data": early}),
            json!({"step": 1, "ts": 0.0, "type": "state", "data": late}),
        );
        assert_eq!(parse_pasted_state(&blob).unwrap().run.hp, 41);
    }

    #[test]
    fn a_clipped_line_does_not_sink_the_paste() {
        let blob = format!("{{\"step\": 0, \"ts\": 0.0, \"typ\n{}", wire());
        assert_eq!(parse_pasted_state(&blob).unwrap().surface, "combat_action");
    }

    #[test]
    fn a_screen_paste_carries_no_fight() {
        let msg = json!({
            "type": "advice_request",
            "advice_for": "shop",
            "seed": "42",
            "act": 1,
            "floor": 5,
            "run_state": wire()["run_state"],
        });
        let imp = parse_pasted_state(&msg.to_string()).unwrap();
        assert_eq!(imp.surface, "shop");
        assert!(!imp.had_combat);
        assert!(!imp.has_combat());
        assert!(!imp
            .warnings
            .iter()
            .any(|w| w.contains("could not be rebuilt")));
    }

    #[test]
    fn restores_the_rng_block_and_map_when_present() {
        let mut msg = wire();
        msg["rng"] = json!({"run_seed": 7, "counters": {"rewards": 3, "shops": 1}});
        msg["map"] = json!({
            "current": {"col": 2, "row": 1},
            "visited": [{"col": 2, "row": 1}],
            "nodes": [],
        });
        let imp = parse_pasted_state(&msg.to_string()).unwrap();
        assert_eq!(imp.run.rng.as_ref().unwrap().run_seed, 7);
        assert_eq!(imp.run.coord(), Some((2, 1)));
        assert_eq!(imp.run.visited(), vec![(2, 1)]);
        assert!(imp.fidelity()["exact"]
            .as_array()
            .unwrap()
            .contains(&json!("map + position")));
    }

    #[test]
    fn rejects_an_empty_paste() {
        assert!(parse_pasted_state("   ").is_err());
        assert!(parse_pasted_state("not json").is_err());
    }

    #[test]
    fn rejects_a_headless_decision() {
        let msg = json!({"type": "decision", "decision": "combat_play", "legal": []});
        let e = parse_pasted_state(&msg.to_string()).unwrap_err();
        assert!(e.0.contains("headless"), "{e}");
    }

    #[test]
    fn a_message_with_no_run_state_is_an_error_message() {
        let msg = json!({"type": "shop", "seed": "42"});
        let e = parse_pasted_state(&msg.to_string()).unwrap_err();
        assert!(
            e.0.contains("run_state") || e.0.contains("No game state"),
            "{e}"
        );
    }

    #[test]
    fn drops_unknown_cards_and_relics_with_a_warning() {
        let mut msg = wire();
        msg["run_state"]["master_deck"] = json!([
            {"id": "STRIKE_IRONCLAD"},
            {"id": "NOT_A_REAL_CARD"},
        ]);
        msg["run_state"]["relics"] = json!([{"id": "BURNING_BLOOD"}, {"id": "NOT_A_RELIC"}]);
        let imp = parse_pasted_state(&msg.to_string()).unwrap();
        assert_eq!(imp.run.deck.len(), 1);
        assert!(imp.warnings.iter().any(|w| w.contains("deck card(s)")));
        assert!(imp.warnings.iter().any(|w| w.contains("relic(s)")));
    }
}
