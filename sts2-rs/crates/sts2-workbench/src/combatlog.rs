//! Render the engine's combat event log the way the game's log panel does.
//!
//! Port of `sts2-pycli/python/workbench/combatlog.py`.
//!
//! The engine (`sts2_core::event_log`, exposed as `RunManager.combat_log()`)
//! records ids, not names: `STRIKE_IRONCLAD` dealt 6 to `Nibbit`. Localization
//! lives here, so this module is where a row becomes a line a human reads —
//! "The Ironclad dealt 6 damage to Nibbit using Strike".
//!
//! Two things it deliberately does NOT do:
//!
//! * It does not return a formatted string only. Each display row keeps the
//!   actor, target, amount and source as separate fields (plus a `text`
//!   fallback) so the UI can colour the names the way the game does without
//!   re-parsing a sentence.
//! * It does not reproduce the engine's C#-bug-compatible wording. `text` on
//!   the engine row says "discarded" for a DRAW because C#'s own
//!   `CardDrawnEntry.Description` does (that rendering exists to diff against
//!   the world-side `[HISTTRACE]` stream). A human-facing log says "drew".
//!
//! By default the output is the GAME's log: hits, block, powers, moves,
//! potions and orbs. Card plays, draws, discards, exhausts and energy are
//! recorded by the engine but hidden — the game does not show them, and the
//! card that mattered already rides along as each row's `source` — except for
//! an AUTO-played card (Stampede, Havoc, Mayhem, …), whose start is shown as a
//! `card_auto_played` row (see [`AUTO_PLAYED_KIND`], GH #1155). Pass
//! `verbose=true` for the full tape, or `keep=[...]` to un-hide named kinds
//! and nothing else — which is what the workbench's timeline slices do for
//! `power_flash`, a VFX signal the board needs on the wire but no reader
//! should ever see.

use std::fmt;

use serde_json::{Map, Value};

use crate::render::py;

/// Row kinds hidden unless `verbose`, or named individually in `keep`. The
/// card play frames are what give the visible rows their `source`, so
/// dropping them loses nothing.
const VERBOSE_ONLY: &[&str] = &[
    "card_play_started",
    "card_play_finished",
    "card_drawn",
    "card_discarded",
    "card_exhausted",
    "energy_spent",
    // A power TRIGGERING is not a combat-log LINE -- the game's own log has no
    // such row. It rides this log only so the workbench can draw
    // `power_flash_vfx`, and a busy fight produces dozens, which would bury
    // everything the log exists to show.
    "power_flash",
];

/// The display kind an auto-played `card_play_started` row is rendered as.
///
/// The game shows an auto-played card: it flies out of the hand (or pile) to
/// the play area and resolves on screen. The workbench's step label names only
/// the card the PLAYER chose (or "End turn"), so an auto-play — Stampede at end
/// of turn, Havoc / Cascade / Mayhem / Hellraiser / Howl from Beyond /
/// Distilled Chaos … — had no visible trace except its consequences (GH #1155).
/// Unlike every other card-play row this one is never hidden.
///
/// Every display row produced INSIDE an auto-play's frame (this row, the
/// damage / block / powers it causes, a nested auto-play) also carries
/// `"auto_played": true`, so the history ledger's compact mode — which
/// otherwise keeps only the enemy's rows — can keep them too. The key is
/// absent, not `false`, on every other row.
pub const AUTO_PLAYED_KIND: &str = "card_auto_played";

/// `str.capitalize()`: first char upper, the rest lower.
fn capitalize(w: &str) -> String {
    let mut chars = w.chars();
    match chars.next() {
        Some(first) => {
            let mut out: String = first.to_uppercase().collect();
            out.push_str(&chars.as_str().to_lowercase());
            out
        }
        None => String::new(),
    }
}

/// `BUTT_MOVE` -> `Butt Move`. The last-resort display name.
#[must_use]
pub fn prettify(raw: &str) -> String {
    raw.replace('-', "_")
        .split('_')
        .filter(|w| !w.is_empty())
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(" ")
}

/// An id -> display-name resolver.
pub type Resolver = Box<dyn Fn(&str) -> String>;

fn default_resolver() -> Resolver {
    Box::new(|raw: &str| prettify(raw))
}

/// Id -> display-name resolvers, one per id namespace.
///
/// Each defaults to [`prettify`], so the module is usable (and testable) with
/// no localization loaded at all; the workbench passes real resolvers.
pub struct LogNames {
    pub card: Resolver,
    pub monster: Resolver,
    pub power: Resolver,
    pub potion: Resolver,
    pub orb: Resolver,
    pub move_: Resolver,
    pub character: Resolver,
    /// Extra fields merged into every display row (unused today — the Python
    /// dataclass declares it and never reads it either; kept so the shape
    /// stays extensible without changing call sites).
    pub extra: Map<String, Value>,
}

impl Default for LogNames {
    fn default() -> Self {
        Self {
            card: default_resolver(),
            monster: default_resolver(),
            power: default_resolver(),
            potion: default_resolver(),
            orb: default_resolver(),
            move_: default_resolver(),
            character: default_resolver(),
            extra: Map::new(),
        }
    }
}

impl fmt::Debug for LogNames {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogNames")
            .field("extra", &self.extra)
            .finish_non_exhaustive()
    }
}

/// `str(row.get(key) or "?")` — the id an engine row names, or `"?"`.
fn id_or_q(row: &Value, key: &str) -> String {
    py::get(row, key)
        .filter(|v| py::truthy(v))
        .map_or_else(|| "?".to_string(), py::to_str)
}

/// `(display name, side, creature uid)` for a serialized `LogCreature`.
///
/// The uid is the join key the VFX layer needs: it is `c.id.get()`, the same
/// `CreatureUid` the board's encounter entries carry, so a per-hit row can be
/// pinned to one board slot. Names cannot do that — two Twig Slimes in one
/// encounter share a display name.
fn creature(
    who: Option<&Value>,
    names: &LogNames,
) -> (Option<String>, Option<&'static str>, Option<i64>) {
    let Some(who) = who.filter(|w| py::truthy(w)) else {
        return (None, None, None);
    };
    let raw = id_or_q(who, "id");
    // `int(uid) if isinstance(uid, int) else None` — bools are ints in Python.
    let uid = py::get(who, "uid").and_then(py::as_int);
    if py::get(who, "kind").and_then(Value::as_str) == Some("player") {
        (Some((names.character)(&raw)), Some("player"), uid)
    } else {
        (Some((names.monster)(&raw)), Some("enemy"), uid)
    }
}

/// Display name of the card / move a row is attributed to.
fn source_name(source: Option<&Value>, names: &LogNames) -> Option<String> {
    let source = source.filter(|s| py::truthy(s))?;
    if py::get(source, "kind").and_then(Value::as_str) == Some("card") {
        let name = (names.card)(&id_or_q(source, "id"));
        return Some(if py::truthy_opt(py::get(source, "upgraded")) {
            format!("{name}+")
        } else {
            name
        });
    }
    Some((names.move_)(&id_or_q(source, "move_id")))
}

/// The raw, joinable identity of the card or move a row came from.
///
/// [`source_name`] renders this same dict for humans; the VFX binding table
/// needs the STABLE id, which prettifying destroys ("BUG_STING" -> "Bug
/// Sting"). Both are emitted: the UI keeps reading `source`, the effect layer
/// reads these.
///
/// A monster's `move_id` alone is not unique across monsters (e.g. `THRASH`
/// exists on both Crusher and BowlbugNectar), so `source_monster` carries the
/// raw, un-prettified monster id alongside it as the other half of the VFX
/// binding key. It is left uncased — PascalCase (`"Crusher"`) is the join key
/// on both the Rust and the binding-table side.
fn source_ids(source: Option<&Value>, base: &mut Map<String, Value>) {
    let mut put = |k: &str, v: Value| {
        base.insert(k.to_string(), v);
    };
    let Some(source) = source.filter(|s| py::truthy(s)) else {
        for k in [
            "source_kind",
            "source_card",
            "source_upgraded",
            "source_move_id",
            "source_monster",
        ] {
            put(k, Value::Null);
        }
        return;
    };
    let kind = py::get(source, "kind");
    let is_card = kind.and_then(Value::as_str) == Some("card");
    let str_if_truthy = |k: &str| {
        py::get(source, k)
            .filter(|v| py::truthy(v))
            .map_or(Value::Null, |v| Value::String(py::to_str(v)))
    };
    put("source_kind", kind.cloned().unwrap_or(Value::Null));
    put(
        "source_card",
        if is_card {
            str_if_truthy("id")
        } else {
            Value::Null
        },
    );
    put(
        "source_upgraded",
        if is_card {
            Value::Bool(py::truthy_opt(py::get(source, "upgraded")))
        } else {
            Value::Null
        },
    );
    put("source_move_id", str_if_truthy("move_id"));
    put("source_monster", str_if_truthy("monster"));
}

/// Turn engine event rows into display rows.
///
/// * `rows`: `RunManager.combat_log()` output, oldest first.
/// * `names`: id -> display-name resolvers. Defaults to prettified ids.
/// * `verbose`: keep ALL the card-movement / energy / play rows the game's
///   own log hides — the whole of `VERBOSE_ONLY`, in one switch.
/// * `keep`: un-hide only these `VERBOSE_ONLY` kinds, leaving the rest
///   hidden. This is the knob for a consumer that needs ONE hidden kind and
///   none of the others: the workbench's timeline slices need `power_flash`
///   (the board's VFX layer reads it off the node) but emphatically not the
///   card tape, which would bloat every saved run and force the client to
///   re-hide six kinds it never asked for. Ignored for kinds that are not
///   hidden to begin with.
/// * `player_id`: the run's character id (`"IRONCLAD"`), for the rows that
///   are implicitly about the player and carry no creature of their own —
///   draws, generated cards, potions, orbs, energy. Rows that DO name a
///   creature (hits, block, powers) use the id the engine stamped on them, so
///   those read right either way; without this the implicit ones would say
///   "Player" while the explicit ones say "The Ironclad".
///
/// Returns one display row per kept event:
///
/// ```text
/// {"kind", "round", "side", "verb", "text",
///  "actor", "actor_side", "actor_uid", "target", "target_side", "target_uid",
///  "amount", "noun", "source",
///  # The raw, joinable identity behind ``source`` — see `source_ids`.
///  "source_kind", "source_card", "source_upgraded", "source_move_id",
///  "source_monster",
///  # ``damage`` rows only:
///  "blocked", "hp_after"}
/// ```
///
/// Absent fields are `null`. `text` is always present and always renders the
/// whole line, so a UI that does not special-case a kind can still show it.
#[must_use]
pub fn format_combat_log(
    rows: &[Value],
    names: Option<&LogNames>,
    verbose: bool,
    keep: &[&str],
    player_id: &str,
) -> Vec<Value> {
    let default_names = LogNames::default();
    let names = names.unwrap_or(&default_names);
    let mut out = Vec::new();
    // One flag per open card-play frame: is this play (or a play that
    // spawned it) an auto-play? Mirrors the engine's own open-source stack
    // (`event_log.rs` `track_source`): a play opens a frame, its finish closes
    // it, a monster move clears it.
    let mut frames: Vec<bool> = Vec::new();
    for row in rows {
        let kind = py::get(row, "kind")
            .filter(|v| py::truthy(v))
            .map_or_else(String::new, py::to_str);
        let auto_start =
            kind == "card_play_started" && py::truthy_opt(py::get_raw(row, "auto_played"));
        match kind.as_str() {
            "monster_move" => frames.clear(),
            "card_play_started" => {
                let parent = frames.last().copied().unwrap_or(false);
                frames.push(parent || auto_start);
            }
            _ => {}
        }
        let in_auto = frames.last().copied().unwrap_or(false);
        if kind == "card_play_finished" {
            frames.pop();
        }
        // An AUTO-played card's start is not hidden: the step label names the
        // card the player chose, never the one Stampede / Havoc / Mayhem pulled
        // (GH #1155), so without this row the play is invisible.
        let shown = if auto_start {
            AUTO_PLAYED_KIND
        } else {
            kind.as_str()
        };
        if VERBOSE_ONLY.contains(&shown) && !verbose && !keep.contains(&shown) {
            continue;
        }
        if let Some(mut display) = format_row(row, shown, names, player_id) {
            if in_auto {
                display.insert("auto_played".to_string(), Value::Bool(true));
            }
            out.push(Value::Object(display));
        }
    }
    out
}

/// `str(x)` for an optional display name (`None` prints as Python's `None`).
fn show(name: Option<&String>) -> &str {
    name.map_or("None", String::as_str)
}

fn opt_string(s: Option<String>) -> Value {
    s.map_or(Value::Null, Value::String)
}

fn opt_side(s: Option<&'static str>) -> Value {
    s.map_or(Value::Null, |s| Value::String(s.to_string()))
}

fn opt_uid(u: Option<i64>) -> Value {
    u.map_or(Value::Null, Value::from)
}

/// The display row under construction; the `put_*` helpers mirror the
/// Python `base |= {...}` updates.
struct Row(Map<String, Value>);

impl Row {
    fn put(&mut self, k: &str, v: Value) {
        self.0.insert(k.to_string(), v);
    }

    fn put_actor(&mut self, name: Option<String>, side: Option<&'static str>, uid: Option<i64>) {
        self.put("actor", opt_string(name));
        self.put("actor_side", opt_side(side));
        self.put("actor_uid", opt_uid(uid));
    }

    fn put_target(&mut self, name: Option<String>, side: Option<&'static str>, uid: Option<i64>) {
        self.put("target", opt_string(name));
        self.put("target_side", opt_side(side));
        self.put("target_uid", opt_uid(uid));
    }
}

/// Render one engine row, or `None` for a kind we do not display.
fn format_row(
    row: &Value,
    kind: &str,
    names: &LogNames,
    player_id: &str,
) -> Option<Map<String, Value>> {
    let mut base = Map::new();
    let mut set = |k: &str, v: Value| {
        base.insert(k.to_string(), v);
    };
    set("kind", Value::String(kind.to_string()));
    set(
        "round",
        py::get_raw(row, "round").cloned().unwrap_or(Value::Null),
    );
    set(
        "side",
        py::get_raw(row, "side").cloned().unwrap_or(Value::Null),
    );
    // The verb this row reads with. The UI composes coloured spans from the
    // fields, and several kinds have two shapes ("applied … to" vs "gained")
    // that it cannot tell apart from actor/target alone — a self-applied power
    // has actor == target and still reads "applied". Deciding it once, here,
    // keeps that logic in one place.
    for k in [
        "verb",
        "actor",
        "actor_side",
        "actor_uid",
        "target",
        "target_side",
        "target_uid",
        "amount",
        "noun",
    ] {
        set(k, Value::Null);
    }
    let source = source_name(py::get(row, "source"), names);
    set("source", opt_string(source.clone()));
    source_ids(py::get(row, "source"), &mut base);

    // The player is the actor for every player-side row; rows that name their
    // own actor (damage, powers, moves) overwrite this below.
    let player = (names.character)(player_id);

    let card_name = || {
        let name = (names.card)(&id_or_q(row, "card"));
        if py::truthy_opt(py::get(row, "upgraded")) {
            format!("{name}+")
        } else {
            name
        }
    };
    let amount = |key: &str| py::int_or_zero(py::get(row, key));

    let mut base = Row(base);
    let text_str = |s: String| Value::String(s);
    let verb_str = |s: &str| Value::String(s.to_string());

    match kind {
        "damage" => {
            let (dealer, dealer_side, dealer_uid) = creature(py::get(row, "dealer"), names);
            let (target, target_side, target_uid) = creature(py::get(row, "receiver"), names);
            let amount = amount("unblocked");
            base.put_actor(dealer.clone(), dealer_side, dealer_uid);
            base.put_target(target.clone(), target_side, target_uid);
            base.put("amount", Value::from(amount));
            // The block split and resulting HP, for the VFX layer: a hit that
            // block ate entirely plays the game's `block_hit`, not the target's
            // material impact, and `hp_after == 0` is the only death signal the
            // log has (`event_log.rs` has no death entry).
            base.put(
                "blocked",
                Value::from(py::int_or_zero(py::get(row, "blocked"))),
            );
            base.put(
                "hp_after",
                py::get_raw(row, "hp_after").cloned().unwrap_or(Value::Null),
            );
            base.put(
                "verb",
                verb_str(if dealer.is_none() { "took" } else { "dealt" }),
            );
            let text = match &dealer {
                None => format!("{} took {amount} damage.", show(target.as_ref())),
                Some(dealer) => {
                    let using = match &source {
                        Some(s) if !s.is_empty() => format!(" using {s}"),
                        _ => String::new(),
                    };
                    format!(
                        "{dealer} dealt {amount} damage to {}{using}.",
                        show(target.as_ref())
                    )
                }
            };
            base.put("text", text_str(text));
        }

        "block" => {
            let (actor, side, actor_uid) = creature(py::get(row, "receiver"), names);
            let amount = amount("amount");
            base.put_actor(actor.clone(), side, actor_uid);
            base.put("amount", Value::from(amount));
            base.put("noun", verb_str("Block"));
            base.put("verb", verb_str("gained"));
            base.put(
                "text",
                text_str(format!("{} gained {amount} Block.", show(actor.as_ref()))),
            );
        }

        "power_flash" => {
            let (target, target_side, target_uid) = creature(py::get(row, "creature"), names);
            let power = (names.power)(&id_or_q(row, "power"));
            base.put_target(target.clone(), target_side, target_uid);
            base.put("noun", verb_str(&power));
            base.put("verb", verb_str("triggered"));
            base.put(
                "text",
                text_str(format!("{}'s {power} triggered.", show(target.as_ref()))),
            );
        }

        "power" => {
            let (applier, applier_side, applier_uid) = creature(py::get(row, "applier"), names);
            let (target, target_side, target_uid) = creature(py::get(row, "receiver"), names);
            let amount = amount("amount");
            let power = (names.power)(&id_or_q(row, "power"));
            // `applier or target` — an empty display name is falsy in Python.
            let actor = match &applier {
                Some(a) if !a.is_empty() => Some(a.clone()),
                _ => target.clone(),
            };
            let actor_side = applier_side.or(target_side);
            let actor_uid = if applier.is_some() {
                applier_uid
            } else {
                target_uid
            };
            base.put_actor(actor, actor_side, actor_uid);
            base.put_target(target.clone(), target_side, target_uid);
            base.put("amount", Value::from(amount));
            base.put("noun", verb_str(&power));
            base.put(
                "verb",
                verb_str(if applier.is_none() {
                    "gained"
                } else {
                    "applied"
                }),
            );
            let text = match &applier {
                None => format!("{} gained {amount} {power}.", show(target.as_ref())),
                Some(applier) => {
                    format!(
                        "{applier} applied {amount} {power} to {}.",
                        show(target.as_ref())
                    )
                }
            };
            base.put("text", text_str(text));
        }

        "monster_move" => {
            let (actor, side, actor_uid) = creature(py::get(row, "monster"), names);
            let mv = (names.move_)(&id_or_q(row, "move_id"));
            base.put_actor(actor.clone(), side, actor_uid);
            base.put("noun", verb_str(&mv));
            base.put("verb", verb_str("used"));
            base.put(
                "text",
                text_str(format!("{} used {mv}.", show(actor.as_ref()))),
            );
        }

        "potion" => {
            let potion = (names.potion)(&id_or_q(row, "potion"));
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&potion));
            base.put("verb", verb_str("drank"));
            base.put("text", text_str(format!("{player} drank {potion}.")));
        }

        "card_generated" => {
            let name = card_name();
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&name));
            base.put("verb", verb_str("generated"));
            base.put("text", text_str(format!("{player} generated {name}.")));
        }

        "orb_channeled" | "orb_evoked" => {
            let orb = (names.orb)(&id_or_q(row, "orb"));
            let verb = if kind == "orb_channeled" {
                "channeled"
            } else {
                "evoked"
            };
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&orb));
            base.put("verb", verb_str(verb));
            base.put("text", text_str(format!("{player} {verb} {orb}.")));
        }

        "card_auto_played" => {
            let name = card_name();
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&name));
            base.put("verb", verb_str("auto-played"));
            // The raw card identity, for the board's auto-play card flash
            // (the `source_*` fields name the PARENT play, e.g. Havoc).
            base.put("card_id", Value::String(id_or_q(row, "card")));
            base.put(
                "card_upgraded",
                Value::Bool(py::truthy_opt(py::get(row, "upgraded"))),
            );
            let via = match &source {
                Some(s) if !s.is_empty() => format!(" via {s}"),
                _ => String::new(),
            };
            base.put(
                "text",
                text_str(format!("{player} auto-played {name}{via}.")),
            );
        }

        // ── verbose-only kinds ──────────────────────────────────────────────
        "card_play_started" | "card_play_finished" => {
            let name = card_name();
            let verb = if kind == "card_play_finished" {
                "played"
            } else {
                "started playing"
            };
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&name));
            base.put("verb", verb_str(verb));
            base.put("text", text_str(format!("{player} {verb} {name}.")));
        }

        "card_drawn" | "card_discarded" | "card_exhausted" => {
            let name = card_name();
            // "drew", not C#'s copy-pasted "discarded" — see the module docs.
            let verb = match kind {
                "card_drawn" => "drew",
                "card_discarded" => "discarded",
                _ => "exhausted",
            };
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("noun", verb_str(&name));
            base.put("verb", verb_str(verb));
            base.put("text", text_str(format!("{player} {verb} {name}.")));
        }

        "energy_spent" => {
            let amount = amount("amount");
            base.put_actor(Some(player.clone()), Some("player"), None);
            base.put("amount", Value::from(amount));
            base.put("verb", verb_str("spent"));
            base.put("text", text_str(format!("{player} spent {amount} Energy.")));
        }

        // An unknown kind is a newer engine talking to an older workbench.
        // Show the engine's own line rather than dropping the event on the
        // floor.
        _ => {
            let text = py::get(row, "text").filter(|t| py::truthy(t))?;
            base.put("text", Value::String(py::to_str(text)));
        }
    }
    Some(base.0)
}

#[cfg(test)]
mod tests {
    //! Port of `tests/workbench/test_combatlog.py` (every case, same names)
    //! plus the formatter halves of `test_bridge_tree_log.py` and
    //! `test_timeline_details.py`.

    use super::*;
    use serde_json::json;

    fn lookup(table: &'static [(&'static str, &'static str)]) -> Resolver {
        Box::new(move |i: &str| {
            table
                .iter()
                .find(|(k, _)| *k == i)
                .map_or_else(|| i.to_string(), |(_, v)| (*v).to_string())
        })
    }

    fn names() -> LogNames {
        LogNames {
            card: lookup(&[
                ("STRIKE_IRONCLAD", "Strike"),
                ("DEFEND_IRONCLAD", "Defend"),
                ("SLIMED", "Slimed"),
                ("BASH", "Bash"),
                ("HAVOC", "Havoc"),
            ]),
            monster: Box::new(|i: &str| i.to_string()),
            power: lookup(&[("HELLRAISER", "Hellraiser")]),
            move_: lookup(&[("BUTT", "Butt Move")]),
            character: lookup(&[("IRONCLAD", "The Ironclad"), ("PLAYER", "The Ironclad")]),
            orb: Box::new(|i: &str| capitalize(i)),
            potion: Box::new(|i: &str| i.to_string()),
            extra: Map::new(),
        }
    }

    fn player() -> Value {
        json!({"kind": "player", "id": "IRONCLAD", "uid": null})
    }

    fn nibbit() -> Value {
        json!({"kind": "monster", "id": "Nibbit", "uid": 1})
    }

    fn strike_source() -> Value {
        json!({"kind": "card", "id": "STRIKE_IRONCLAD", "upgraded": false})
    }

    fn row(kind: &str, fields: Value) -> Value {
        let mut r = json!({"kind": kind, "round": 1, "side": "player", "source": null});
        for (k, v) in fields.as_object().unwrap() {
            r[k] = v.clone();
        }
        r
    }

    fn damage(overrides: Value) -> Value {
        let mut fields = json!({
            "dealer": player(),
            "receiver": nibbit(),
            "unblocked": 6,
            "blocked": 0,
            "hp_before": 20,
            "hp_after": 14,
            "source": strike_source(),
        });
        for (k, v) in overrides.as_object().unwrap() {
            fields[k] = v.clone();
        }
        row("damage", fields)
    }

    fn fmt(rows: &[Value]) -> Vec<Value> {
        format_combat_log(rows, Some(&names()), false, &[], "PLAYER")
    }

    fn one(rows: &[Value]) -> Value {
        let mut out = fmt(rows);
        assert_eq!(out.len(), 1, "{out:?}");
        out.pop().unwrap()
    }

    #[test]
    fn test_empty_log_renders_nothing() {
        assert!(format_combat_log(&[], None, false, &[], "PLAYER").is_empty());
    }

    #[test]
    fn test_a_hit_names_dealer_amount_target_and_card() {
        let line = one(&[damage(json!({}))]);
        assert_eq!(
            line["text"],
            "The Ironclad dealt 6 damage to Nibbit using Strike."
        );
        // The UI colours the names, so it needs them apart from the sentence.
        assert_eq!(
            (&line["actor"], &line["actor_side"]),
            (&json!("The Ironclad"), &json!("player"))
        );
        assert_eq!(
            (&line["target"], &line["target_side"]),
            (&json!("Nibbit"), &json!("enemy"))
        );
        assert_eq!(
            (&line["amount"], &line["source"]),
            (&json!(6), &json!("Strike"))
        );
    }

    #[test]
    fn test_a_hit_with_no_open_card_omits_the_using_clause() {
        let line = one(&[damage(json!({"source": null}))]);
        assert_eq!(line["text"], "The Ironclad dealt 6 damage to Nibbit.");
    }

    #[test]
    fn test_an_upgraded_source_card_keeps_its_plus() {
        let upgraded = json!({"kind": "card", "id": "STRIKE_IRONCLAD", "upgraded": true});
        let line = one(&[damage(json!({"source": upgraded}))]);
        assert_eq!(line["source"], "Strike+");
        assert!(line["text"].as_str().unwrap().ends_with("using Strike+."));
    }

    #[test]
    fn test_source_less_damage_reads_took() {
        let line = one(&[damage(json!({"dealer": null, "receiver": player()}))]);
        assert_eq!(line["text"], "The Ironclad took 6 damage.");
    }

    #[test]
    fn test_block_gain() {
        let line = one(&[row("block", json!({"receiver": player(), "amount": 5}))]);
        assert_eq!(line["text"], "The Ironclad gained 5 Block.");
    }

    #[test]
    fn test_power_applied_names_applier_and_target() {
        let r = row(
            "power",
            json!({"applier": player(), "receiver": player(), "power": "HELLRAISER", "amount": 1}),
        );
        let line = one(&[r]);
        assert_eq!(
            line["text"],
            "The Ironclad applied 1 Hellraiser to The Ironclad."
        );
    }

    #[test]
    fn test_power_without_an_applier_reads_gained() {
        let r = row(
            "power",
            json!({"applier": null, "receiver": nibbit(), "power": "HELLRAISER", "amount": 2}),
        );
        let line = one(&[r]);
        assert_eq!(line["text"], "Nibbit gained 2 Hellraiser.");
    }

    #[test]
    fn test_monster_move() {
        let r = row(
            "monster_move",
            json!({"side": "enemy", "monster": nibbit(), "move_id": "BUTT"}),
        );
        let line = one(&[r]);
        assert_eq!(line["text"], "Nibbit used Butt Move.");
        assert_eq!(line["actor_side"], "enemy");
    }

    #[test]
    fn test_orb_rows() {
        let rows = [
            row("orb_channeled", json!({"orb": "LIGHTNING"})),
            row("orb_evoked", json!({"orb": "LIGHTNING"})),
        ];
        let texts: Vec<Value> = fmt(&rows).into_iter().map(|r| r["text"].clone()).collect();
        assert_eq!(
            texts,
            vec![
                json!("The Ironclad channeled Lightning."),
                json!("The Ironclad evoked Lightning.")
            ]
        );
    }

    #[test]
    fn test_the_games_log_hides_plays_draws_energy_and_power_flashes() {
        // `power_flash` is in this list for a reason the others are not: it is
        // hidden by DEFAULT and un-hidden only by an explicit `keep`, which is
        // the entire premise of that parameter. Nothing else pins it -- every
        // other surviving test asserts the row is PRESENT, so deleting
        // `power_flash` from `VERBOSE_ONLY` once left the whole workbench suite
        // green (979 passed). The tripwire that used to carry this invariant
        // was correctly removed with the hand-mirror it guarded; this row is
        // where it lands now.
        let rows = [
            row("energy_spent", json!({"amount": 1})),
            row(
                "card_play_started",
                json!({"card": "STRIKE_IRONCLAD", "upgraded": false}),
            ),
            row(
                "card_drawn",
                json!({"card": "DEFEND_IRONCLAD", "upgraded": false, "from_hand_draw": true}),
            ),
            damage(json!({})),
            row(
                "power_flash",
                json!({"creature": nibbit(), "power": "THORNS"}),
            ),
            row(
                "card_play_finished",
                json!({"card": "STRIKE_IRONCLAD", "upgraded": false}),
            ),
        ];
        let kinds: Vec<Value> = fmt(&rows).into_iter().map(|r| r["kind"].clone()).collect();
        assert_eq!(
            kinds,
            vec![json!("damage")],
            "the game's panel shows the hit, not the bookkeeping"
        );
    }

    /// GH #1155: Stampede auto-plays Bash at end of turn. The step is labelled
    /// "End turn", so the auto-play row is the only place the card shows.
    #[test]
    fn test_an_auto_played_card_is_shown_and_its_rows_are_tagged() {
        let bash = json!({"kind": "card", "id": "BASH", "upgraded": false});
        let rows = [
            row(
                "card_play_started",
                json!({"card": "BASH", "upgraded": false, "auto_played": true, "energy_spent": 0}),
            ),
            damage(json!({"source": bash.clone()})),
            row(
                "power",
                json!({"applier": null, "receiver": nibbit(), "power": "VULNERABLE",
                       "amount": 2, "source": bash.clone()}),
            ),
            row(
                "card_play_finished",
                json!({"card": "BASH", "upgraded": false, "source": bash}),
            ),
            row(
                "monster_move",
                json!({"side": "enemy", "monster": nibbit(), "move_id": "BUTT"}),
            ),
        ];
        let out = fmt(&rows);
        let kinds: Vec<Value> = out.iter().map(|r| r["kind"].clone()).collect();
        assert_eq!(
            kinds,
            vec![
                json!("card_auto_played"),
                json!("damage"),
                json!("power"),
                json!("monster_move")
            ]
        );
        assert_eq!(out[0]["text"], "The Ironclad auto-played Bash.");
        assert_eq!(out[0]["verb"], "auto-played");
        assert_eq!(out[0]["noun"], "Bash");
        assert_eq!(out[0]["card_id"], "BASH");
        assert_eq!(out[0]["card_upgraded"], false);
        for r in &out[..3] {
            assert_eq!(r["auto_played"], true, "{r}");
        }
        // Absent, not false, outside an auto-play: the enemy's turn is not
        // part of it.
        assert!(out[3].get("auto_played").is_none(), "{}", out[3]);
    }

    /// A manual Havoc auto-plays the top card: the parent's rows are the
    /// player's own play (untagged), the child's are tagged, and the child's
    /// line names what pulled it.
    #[test]
    fn test_a_nested_auto_play_names_its_parent_and_only_it_is_tagged() {
        let havoc = json!({"kind": "card", "id": "HAVOC", "upgraded": false});
        let rows = [
            row(
                "card_play_started",
                json!({"card": "HAVOC", "upgraded": false, "auto_played": false}),
            ),
            row(
                "card_play_started",
                json!({"card": "STRIKE_IRONCLAD", "upgraded": true, "auto_played": true,
                       "source": havoc.clone()}),
            ),
            damage(json!({})),
            row(
                "card_play_finished",
                json!({"card": "STRIKE_IRONCLAD", "upgraded": true, "source": strike_source()}),
            ),
            row(
                "block",
                json!({"receiver": player(), "amount": 3, "source": havoc.clone()}),
            ),
            row(
                "card_play_finished",
                json!({"card": "HAVOC", "upgraded": false, "source": havoc}),
            ),
        ];
        let out = fmt(&rows);
        let kinds: Vec<Value> = out.iter().map(|r| r["kind"].clone()).collect();
        assert_eq!(
            kinds,
            vec![json!("card_auto_played"), json!("damage"), json!("block")]
        );
        assert_eq!(
            out[0]["text"],
            "The Ironclad auto-played Strike+ via Havoc."
        );
        assert_eq!(out[0]["auto_played"], true);
        assert_eq!(out[1]["auto_played"], true);
        assert!(out[2].get("auto_played").is_none(), "{}", out[2]);
    }

    #[test]
    fn test_verbose_keeps_them_and_says_drew_not_discarded() {
        let rows = [
            row("energy_spent", json!({"amount": 1})),
            row(
                "card_drawn",
                json!({"card": "DEFEND_IRONCLAD", "upgraded": false, "from_hand_draw": true}),
            ),
        ];
        let texts: Vec<Value> = format_combat_log(&rows, Some(&names()), true, &[], "PLAYER")
            .into_iter()
            .map(|r| r["text"].clone())
            .collect();
        // The engine's own `text` says "discarded" here, bug-compatibly with
        // C#. A human-facing log must not.
        assert_eq!(
            texts,
            vec![
                json!("The Ironclad spent 1 Energy."),
                json!("The Ironclad drew Defend.")
            ]
        );
    }

    #[test]
    fn test_keep_un_hides_only_the_named_kind() {
        // `keep` is the narrow alternative to `verbose`. The workbench's
        // timeline slices need exactly one hidden kind on the wire —
        // `power_flash`, which the board's VFX layer reads off the node — and
        // none of the card tape, which would ride into every saved run's JSONL
        // and force the client to re-hide six kinds it never asked for. So this
        // asserts BOTH halves: the named kind arrives, and its neighbours in
        // `VERBOSE_ONLY` still do not.
        let rows = [
            row("energy_spent", json!({"amount": 1})),
            row(
                "card_drawn",
                json!({"card": "DEFEND_IRONCLAD", "upgraded": false, "from_hand_draw": true}),
            ),
            damage(json!({})),
            row(
                "power_flash",
                json!({"creature": nibbit(), "power": "THORNS"}),
            ),
        ];
        let kinds: Vec<Value> =
            format_combat_log(&rows, Some(&names()), false, &["power_flash"], "PLAYER")
                .into_iter()
                .map(|r| r["kind"].clone())
                .collect();
        assert_eq!(kinds, vec![json!("damage"), json!("power_flash")]);
    }

    #[test]
    fn test_keep_is_inert_for_a_kind_that_was_never_hidden() {
        // A kind outside `VERBOSE_ONLY` is shown either way — naming it in
        // `keep` must not change anything, least of all duplicate it. The
        // second row is what makes this test able to fail: with only the
        // `damage` row it passed even with `keep` deleted from the filter
        // expression entirely, since it then guarded nothing but duplication.
        // Naming an unhidden kind must also leave its HIDDEN neighbours hidden
        // — `keep` un-hides what it names, not everything alongside it.
        let rows = [
            damage(json!({})),
            row(
                "card_drawn",
                json!({"card": "DEFEND_IRONCLAD", "upgraded": false}),
            ),
            row(
                "power_flash",
                json!({"creature": nibbit(), "power": "THORNS"}),
            ),
        ];
        let kinds: Vec<Value> =
            format_combat_log(&rows, Some(&names()), false, &["damage"], "PLAYER")
                .into_iter()
                .map(|r| r["kind"].clone())
                .collect();
        assert_eq!(kinds, vec![json!("damage")]);
    }

    #[test]
    fn test_an_unknown_kind_falls_back_to_the_engine_text() {
        let r = row("something_new", json!({"text": "IRONCLAD did a new thing"}));
        let line = one(&[r]);
        assert_eq!(line["text"], "IRONCLAD did a new thing");
    }

    #[test]
    fn test_an_unknown_kind_with_no_text_is_dropped() {
        assert!(fmt(&[row("something_new", json!({}))]).is_empty());
    }

    #[test]
    fn test_ids_are_prettified_when_no_resolvers_are_given() {
        let r = row(
            "monster_move",
            json!({"monster": nibbit(), "move_id": "BUTT_MOVE"}),
        );
        let mut out = format_combat_log(&[r], None, false, &[], "PLAYER");
        assert_eq!(out.len(), 1);
        let line = out.pop().unwrap();
        assert_eq!(line["text"], "Nibbit used Butt Move.");
    }

    #[test]
    fn test_a_generated_card_is_shown_like_the_game_does() {
        // A monster's Slimed / Wound lands here: the move row above says what
        // the enemy did, this says what it left behind.
        let r = row(
            "card_generated",
            json!({"card": "SLIMED", "upgraded": false, "pile": "discard", "by_player": false}),
        );
        let line = one(&[r]);
        assert_eq!(line["text"], "The Ironclad generated Slimed.");
        assert_eq!(line["verb"], "generated");
    }

    #[test]
    fn test_a_move_source_carries_the_raw_move_and_monster_ids() {
        // Two monsters (Crusher, BowlbugNectar) both have a THRASH move, so
        // source_move_id alone is not a unique VFX-binding key —
        // source_monster is the other half.
        let move_source = json!({"kind": "move", "monster": "Crusher", "move_id": "THRASH"});
        let line = one(&[damage(json!({"source": move_source}))]);
        assert_eq!(line["source_kind"], "move");
        assert_eq!(line["source_move_id"], "THRASH");
        assert_eq!(line["source_monster"], "Crusher");
        assert_eq!(line["source_card"], Value::Null);
        assert_eq!(line["source_upgraded"], Value::Null);
        // The display name is unchanged — this is additive.
        assert_eq!(line["source"], "THRASH");
    }

    #[test]
    fn test_a_card_source_carries_id_and_upgrade_but_no_monster() {
        let line = one(&[damage(json!({}))]);
        assert_eq!(line["source_kind"], "card");
        assert_eq!(line["source_card"], "STRIKE_IRONCLAD");
        assert_eq!(line["source_upgraded"], false);
        assert_eq!(line["source_move_id"], Value::Null);
        assert_eq!(line["source_monster"], Value::Null);
        // The display name is unchanged — this is additive.
        assert_eq!(line["source"], "Strike");
    }

    #[test]
    fn test_an_absent_source_leaves_the_raw_id_fields_none() {
        let line = one(&[damage(json!({"source": null}))]);
        for k in [
            "source_kind",
            "source_card",
            "source_upgraded",
            "source_move_id",
            "source_monster",
        ] {
            assert_eq!(line[k], Value::Null, "{k}");
        }
    }

    // ── test_bridge_tree_log.py / test_timeline_details.py (formatter parts) ─

    #[test]
    fn test_raw_game_rows_become_display_rows() {
        // The mod ships the game's own history rows; with the run's
        // `player_id` they become the same display rows a simulated run
        // produces, so one panel serves both.
        let raw = json!({
            "kind": "damage",
            "round": 1,
            "side": "player",
            "text": "IRONCLAD dealt 6 damage to NIBBIT",
            "source": {"kind": "card", "id": "STRIKE_IRONCLAD", "upgraded": false},
            "dealer": {"kind": "player", "id": "IRONCLAD", "uid": null},
            "receiver": {"kind": "monster", "id": "Nibbit", "uid": null},
            "unblocked": 6,
            "blocked": 0,
        });
        let mut out = format_combat_log(&[raw], Some(&names()), false, &[], "IRONCLAD");
        assert_eq!(out.len(), 1);
        let r = out.pop().unwrap();
        assert_eq!(
            r["text"],
            "The Ironclad dealt 6 damage to Nibbit using Strike."
        );
        assert_eq!(
            (&r["actor_side"], &r["target_side"]),
            (&json!("player"), &json!("enemy"))
        );
        // No `hp_after` on the wire → null, and the uid-less creatures too.
        assert_eq!(r["hp_after"], Value::Null);
        assert_eq!(r["target_uid"], Value::Null);
    }

    #[test]
    fn test_an_unknown_row_kind_still_shows_the_games_own_sentence() {
        // Forward compatibility: the mod maps entry types the workbench has no
        // renderer for to `kind: "other"` and always sends the game's `text`.
        let raw =
            json!({"kind": "other", "round": 2, "side": "enemy", "text": "NIBBIT summoned 1"});
        let out = format_combat_log(&[raw], Some(&names()), false, &[], "IRONCLAD");
        assert_eq!(out[0]["text"], "NIBBIT summoned 1");
        assert_eq!(out[0]["round"], 2);
        assert_eq!(out[0]["side"], "enemy");
    }

    #[test]
    fn a_full_damage_row_has_exactly_the_documented_keys() {
        let line = one(&[damage(json!({}))]);
        let mut keys: Vec<&str> = line
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        let mut expected = vec![
            "kind",
            "round",
            "side",
            "verb",
            "actor",
            "actor_side",
            "actor_uid",
            "target",
            "target_side",
            "target_uid",
            "amount",
            "noun",
            "source",
            "source_kind",
            "source_card",
            "source_upgraded",
            "source_move_id",
            "source_monster",
            "blocked",
            "hp_after",
            "text",
        ];
        expected.sort_unstable();
        assert_eq!(keys, expected);
        assert_eq!(line["target_uid"], 1);
        assert_eq!(line["actor_uid"], Value::Null);
        assert_eq!(line["hp_after"], 14);
        assert_eq!(line["blocked"], 0);
    }

    #[test]
    fn prettify_matches_python_capitalize_join() {
        assert_eq!(prettify("BUTT_MOVE"), "Butt Move");
        assert_eq!(prettify("bug-sting"), "Bug Sting");
        assert_eq!(prettify("__A__b_"), "A B");
        assert_eq!(prettify(""), "");
    }
}
