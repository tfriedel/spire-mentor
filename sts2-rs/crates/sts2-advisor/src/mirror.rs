//! The combat mirror's comparison of its prediction against reality.
//!
//! Port of the pure half of `sts2-pycli/python/combat_mirror.py` (GH #1426
//! step 1): what is observable on a wire, how two wires' observables differ,
//! which enemy is which across the two worlds, and what a reconcile must
//! carry over because the wire does not say. The mirror itself is
//! [`crate::combat_mirror`]; the Python module's functions call these through
//! `sts2_rs`.
//!
//! The *why* of the public functions' rules (enemy identity keys, the
//! in-flight grant, relocation, the undisclosed counters) stays on the
//! Python wrappers in `combat_mirror.py`, whose docstrings the advisor's
//! readers know and whose tests hold this port to them; the private keys'
//! reasons are here. Values are Python's: see [`crate::py`].

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::py::{self, as_list, as_obj, get, list, mget, obj, str_of, truthy};

/// One observable on which the mirror and the world disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct Desync {
    /// `$.player.hp`, `$.encounter[Nibbit#0].intents`, ...
    pub path: String,
    /// The mirror's value (`Null` when the mirror has no such path).
    pub mirror: Value,
    /// The world's value (`Null` when the world has no such path).
    pub world: Value,
}

/// The per-enemy observable suffixes [`observables`] emits.
pub const ENEMY_FIELDS: [&str; 5] = ["hp", "block", "is_dead", "powers", "intents"];

/// The deck piles [`observables`] compares, in wire spelling.
const PILES: [&str; 5] = [
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "play_pile",
];

/// Streams whose read count a combat decision can move (`_COMBAT_RNG_STREAMS`).
pub const COMBAT_RNG_STREAMS: [&str; 8] = [
    "shuffle",
    "monster_ai",
    "combat_card_generation",
    "combat_potion_generation",
    "combat_card_selection",
    "combat_energy_costs",
    "combat_targets",
    "combat_orbs",
];

/// Powers whose counter `state_from_dict` derives from the wire, so a carry
/// must not override it (`_undisclosed_power_counters`'s `derived`).
const DERIVED_POWER_COUNTERS: [&str; 7] = [
    "SLOW",
    "WITHERING_PRESENCE",
    "AUTOMATION",
    "PANACHE",
    "ORBIT",
    "JUGGLING",
    "HARDENED_SHELL",
];

/// Wire deck-pile key -> the engine's locator pile name, in the order a
/// relocation searches them.
const WIRE_PILE_TO_LOCATOR: [(&str, &str); 4] = [
    ("hand", "hand"),
    ("draw_pile", "draw"),
    ("discard_pile", "discard"),
    ("exhaust_pile", "exhaust"),
];

/// Powers as a sorted `[name, amount]` list.
///
/// Sorted because the arena's power order is an implementation detail: two
/// states with the same powers applied in a different order are the same
/// board, and flagging that as a DESYNC would bury the real ones.
fn powers(raw: &[Value]) -> Value {
    let mut out: Vec<(String, i64)> = raw
        .iter()
        .map(|p| (str_of(get(p, "name")), py::int_of(get(p, "amount"))))
        .collect();
    out.sort();
    Value::Array(out.into_iter().map(|(n, a)| json!([n, a])).collect())
}

/// The Necrobinder pet's `[alive, hp, max_hp, block]`, or `Null` when there
/// is no pet.
///
/// `max_hp` is included deliberately: the Osty GROWS across a combat, and a
/// mirror holding a freshly re-summoned 1-HP pet against a grown one is the
/// exact failure this key exists to catch.
fn osty(raw: &Value) -> Value {
    if !truthy(raw) {
        return Value::Null;
    }
    json!([
        get(raw, "alive"),
        get(raw, "hp"),
        get(raw, "max_hp"),
        get(raw, "block")
    ])
}

/// A card's identity for diff purposes.
///
/// Includes `cost_override` and the `enchantment` (name *and* stack: a
/// Nimble 2 and a Nimble 4 Defend block differently), since a Snecko'd hand
/// and a clean one are genuinely different boards, but not the per-instance
/// `combat_vars` blob, which is opaque on both sides.
fn card_key(c: &Value) -> Value {
    let mut flags: Vec<Value> = list(c, "flags").to_vec();
    flags.sort_by(py::cmp);
    json!([
        get(c, "id"),
        truthy(get(c, "upgraded")),
        get(c, "cost_override"),
        get(c, "enchantment"),
        get(c, "enchantment_amount"),
        flags,
    ])
}

/// An enemy intent's identity: everything the wire actually carries.
///
/// `kind` (not `type`) is what `state_to_dict` emits; both sides of a diff
/// pass through that emitter, so reading the adapter's spelling would
/// silently compare `None` to `None` and mask every intent divergence, one of
/// the highest-signal fields there is.
///
/// `amount`/`times` are compared for **Attack only**. A telegraph is an icon
/// plus, for attacks, the damage: the engine's `Intent::Buff` carries a
/// `PowerKind` and stack count that no wire dialect transmits, so
/// `parse_intent` rebuilds every Buff as `{Strength, 0}`. Diffing those would
/// flag every reconstruction ever made while telling us nothing; the real
/// buff comes from the AI's move when it performs, not from the telegraph.
///
/// The damage compared is the **baked** one, what the icon actually paints.
/// A real wire reports C# `AttackIntent.GetSingleDamage`, which has already
/// run the dealer's Strength/Weak/Shrink and the player's Vulnerable through
/// `ModifyDamage`; the engine's `Intent::Attack` stores the raw base from the
/// move table, because that is what `unbake_wire_intent` reconstructs on the
/// way back in. Comparing those two directly diffs raw against baked and
/// flags every modified board as a divergence: on the v17 corpus, 161 of 213
/// unequal attack telegraphs were exactly that, with Strength the dominant
/// driver. `state_to_dict` therefore emits `effective_amount` alongside
/// `amount`; prefer it when present so both sides are compared in the same
/// units. The world wire has no such key and falls back to `amount`, which is
/// already baked.
///
/// `times` is normalised to 1 when absent. `state_to_dict` (`state_dict.rs`)
/// emits `times` ONLY for `Intent::MultiAttack`, while the bridge always
/// emits `times: 1`. Left raw, that compares `None` against `1` and flags
/// EVERY single-hit attack telegraph when a stored wire pair is re-diffed. A
/// missing `times` means one hit on both sides (C# `AttackIntent.Repeats`
/// defaults to 0 and `SingleAttackIntent` paints one icon), so fold the two
/// spellings together.
fn intent_key(it: &Value) -> Value {
    let kind = get(it, "kind");
    if kind.as_str() == Some("Attack") {
        let baked = get(it, "effective_amount");
        let amount = if baked.is_null() {
            get(it, "amount")
        } else {
            baked
        };
        let times = get(it, "times");
        let times = if times.is_null() { &json!(1) } else { times };
        return json!([kind, amount, times]);
    }
    json!([kind])
}

/// `enemy_slot_key`: `"<kind>#<nth>"` for the living, `"<kind>#dead<nth>"`
/// for corpses, numbered within each namespace in list order.
#[must_use]
pub fn enemy_slot_key(enemy: &Value, index: usize, encounter: &[Value]) -> String {
    let kind = str_of(get(enemy, "kind"));
    let dead = truthy(get(enemy, "is_dead"));
    let nth = encounter[..index.min(encounter.len())]
        .iter()
        .filter(|e| str_of(get(e, "kind")) == kind && truthy(get(e, "is_dead")) == dead)
        .count();
    if dead {
        format!("{kind}#dead{nth}")
    } else {
        format!("{kind}#{nth}")
    }
}

/// `observables`: the deterministic half of a wire, as `{path: value}`.
#[must_use]
pub fn observables(wire: &Value) -> BTreeMap<String, Value> {
    // `get` on a missing or non-object value is `Null`, so this is `wire.get("player") or {}`.
    let player = get(wire, "player");
    let deck = get(wire, "deck");
    let mut out = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        out.insert(k.to_owned(), v);
    };
    put("$.phase", get(wire, "phase").clone());
    put("$.round", get(wire, "round").clone());
    put("$.side", get(wire, "side").clone());
    put("$.player.hp", get(player, "hp").clone());
    put("$.player.block", get(player, "block").clone());
    put("$.player.energy", get(player, "energy").clone());
    put("$.player.powers", powers(list(player, "powers")));
    put(
        "$.player.potions",
        Value::Array(list(player, "potions").to_vec()),
    );
    put(
        "$.player.orbs",
        Value::Array(
            list(player, "orbs")
                .iter()
                .map(|o| json!([get(o, "type"), get(o, "passive"), get(o, "evoke")]))
                .collect(),
        ),
    );
    // The Necrobinder's pet. It absorbs damage aimed at the player
    // (`DieForYouPower`), so a mirror whose Osty disagrees with the game
    // plans against a different amount of soak: as deterministic an
    // observable as player block. Uncompared, a reconstruction that dropped
    // the pet could only surface indirectly, as `$.player.hp`.
    put("$.player.osty", osty(get(player, "osty")));
    // `play_pile` is here because the wire round-trips it and a mid-combat
    // `card_select` is opened BY a card sitting in it. Leave it out and a
    // mirror holding a card in flight against a world reporting none diffs
    // clean: the divergence would be structurally unreportable.
    for pile in PILES {
        put(
            &format!("$.deck.{pile}"),
            Value::Array(list(deck, pile).iter().map(card_key).collect()),
        );
    }

    let encounter = list(wire, "encounter");
    let keys: Vec<String> = encounter
        .iter()
        .enumerate()
        .map(|(i, e)| enemy_slot_key(e, i, encounter))
        .collect();
    // Enemies are keyed by IDENTITY, not by slot (see `enemy_slot_key`).
    // The roster is POSITIONAL, not sorted: C# keeps `_enemies` in
    // encounter-slot order and every random-target effect indexes it
    // (`HittableEnemies`), so order IS an observable. Sorting made a
    // re-ordered board invisible: 18 v28wr enemy-hp events sat on silently
    // re-ordered rosters where Juggernaut's random target hit the wrong
    // creature.
    //
    // Where a CORPSE sits is not an observable. C# keeps dead-but-reviving
    // creatures in their slots, but the headless producer serializes alive
    // and dead into separate lists (`enemies` + `dead_enemies`) that the
    // adapter re-concatenates dead-last, while the engine wire and the live
    // bridge ship corpses inline at their spawn slots. Comparing those
    // interleavings verbatim flagged every post-death Decimillipede /
    // Parafright board (44/158 of the d1 corpus). So: alive first, corpses
    // after, each in list order (a stable sort on `is_dead`); the keys
    // number the living and the dead apart, so they are unchanged by it.
    let roster: Vec<Value> = (0..encounter.len())
        .filter(|&i| !truthy(get(&encounter[i], "is_dead")))
        .chain((0..encounter.len()).filter(|&i| truthy(get(&encounter[i], "is_dead"))))
        .map(|i| Value::String(keys[i].clone()))
        .collect();
    put("$.encounter.roster", Value::Array(roster));
    for (e, key) in encounter.iter().zip(&keys) {
        put(&format!("$.encounter[{key}].hp"), get(e, "hp").clone());
        put(
            &format!("$.encounter[{key}].block"),
            get(e, "block").clone(),
        );
        put(
            &format!("$.encounter[{key}].is_dead"),
            get(e, "is_dead").clone(),
        );
        put(
            &format!("$.encounter[{key}].powers"),
            powers(list(e, "powers")),
        );
        // A corpse the game removed telegraphs nothing on screen. The
        // engine's corpse keeps the intent it died with, and one spliced
        // back into a rebuilt board (`with_carried_corpses`) re-bakes it
        // without the Strength it had, so comparing it flagged every later
        // decision of the fight. A corpse that stays on the board (a
        // reviver) does telegraph, and is compared.
        let intents = if truthy(get(e, "is_dead")) && truthy(get(e, "removed_after_death")) {
            Vec::new()
        } else {
            list(e, "intents").iter().map(intent_key).collect()
        };
        put(
            &format!("$.encounter[{key}].intents"),
            Value::Array(intents),
        );
    }
    out
}

/// `_enemy_keys`: every enemy's identity key.
fn enemy_keys(wire: &Value) -> BTreeSet<String> {
    let enc = list(wire, "encounter");
    enc.iter()
        .enumerate()
        .map(|(i, e)| enemy_slot_key(e, i, enc))
        .collect()
}

/// `_enemy_is_dead`: whether the enemy named `key` is a corpse.
fn enemy_is_dead(wire: &Value, key: &str) -> bool {
    let enc = list(wire, "encounter");
    enc.iter()
        .enumerate()
        .find(|(i, e)| enemy_slot_key(e, *i, enc) == key)
        .is_some_and(|(_, e)| truthy(get(e, "is_dead")))
}

/// `diff_observables`: every observable on which the two wires disagree,
/// sorted by path. A corpse only the mirror still holds (the game reaped it)
/// is not a difference. `skip_prefix` and `skip_paths` drop paths.
#[must_use]
pub fn diff_observables(
    mirror_wire: &Value,
    world_wire: &Value,
    skip_prefix: Option<&str>,
    skip_paths: &BTreeSet<String>,
) -> Vec<Desync> {
    let mut mine = observables(mirror_wire);
    let theirs = observables(world_wire);
    let world_keys = enemy_keys(world_wire);
    // An enemy the mirror still has and the world does not is a corpse the
    // game REAPED. The engine keeps it so its revive machinery can find it: a
    // documented modelling difference, not a misprediction. A mirror-only
    // enemy that is still ALIVE, or a world-only enemy, survives into the
    // roster comparison and is reported.
    let reaped: BTreeSet<String> = enemy_keys(mirror_wire)
        .into_iter()
        .filter(|k| !world_keys.contains(k) && enemy_is_dead(mirror_wire, k))
        .collect();
    if !reaped.is_empty() {
        for k in &reaped {
            for f in ENEMY_FIELDS {
                mine.remove(&format!("$.encounter[{k}].{f}"));
            }
        }
        // Filter, don't re-sort: both rosters are already alive-first, and
        // sorting only the mirror's flagged every post-reap board whose
        // survivors aren't in lexical order.
        if let Some(Value::Array(roster)) = mine.get_mut("$.encounter.roster") {
            roster.retain(|k| k.as_str().is_none_or(|k| !reaped.contains(k)));
        }
    }
    let paths: BTreeSet<&String> = mine.keys().chain(theirs.keys()).collect();
    paths
        .into_iter()
        .filter(|p| !skip_paths.contains(*p) && skip_prefix.is_none_or(|s| !p.starts_with(s)))
        .filter_map(|p| {
            let (m, w) = (
                mine.get(p).unwrap_or(&Value::Null),
                theirs.get(p).unwrap_or(&Value::Null),
            );
            (!py::eq(m, w)).then(|| Desync {
                path: p.clone(),
                mirror: m.clone(),
                world: w.clone(),
            })
        })
        .collect()
}

/// `_cursor_drift`: combat RNG streams the two sides read a different number
/// of times. The mirror's counts are `rng_counters`, the world's
/// `rng.counters`; a stream either side does not report is skipped.
#[must_use]
pub fn cursor_drift(mirror_wire: &Value, world_wire: &Value) -> Vec<Desync> {
    let mine = obj(mirror_wire, "rng_counters");
    let theirs = as_obj(get(get(world_wire, "rng"), "counters"));
    COMBAT_RNG_STREAMS
        .iter()
        .filter_map(|name| {
            let (a, b) = (mget(mine, name), mget(theirs, name));
            (!a.is_null() && !b.is_null() && !py::eq(a, b)).then(|| Desync {
                path: format!("$.rng.counters.{name}"),
                mirror: a.clone(),
                world: b.clone(),
            })
        })
        .collect()
}

/// `unreported_paths`: observable paths the world's producer does not emit
/// at all (an absent `play_pile` means unknown-or-empty, not comparable).
#[must_use]
pub fn unreported_paths(raw_world_wire: &Value) -> BTreeSet<String> {
    if obj(raw_world_wire, "deck").contains_key("play_pile") {
        BTreeSet::new()
    } else {
        BTreeSet::from(["$.deck.play_pile".to_owned()])
    }
}

/// `wire_enemy_index`: `{identity key -> slot}`.
#[must_use]
pub fn wire_enemy_index(wire: &Value) -> BTreeMap<String, usize> {
    let enc = list(wire, "encounter");
    enc.iter()
        .enumerate()
        .map(|(i, e)| (enemy_slot_key(e, i, enc), i))
        .collect()
}

/// How many living enemies of each kind a wire reports.
fn living_by_kind(wire: &Value) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for e in list(wire, "encounter") {
        if !truthy(get(e, "is_dead")) {
            *out.entry(str_of(get(e, "kind"))).or_insert(0) += 1;
        }
    }
    out
}

/// `_unalignable_kinds`: kinds whose living counts differ between the two
/// worlds, so their `#nth` numbering cannot be trusted to name the same
/// creature.
#[must_use]
pub fn unalignable_kinds(world_wire: &Value, mirror_wire: &Value) -> BTreeSet<String> {
    let (world, mirror) = (living_by_kind(world_wire), living_by_kind(mirror_wire));
    world
        .keys()
        .chain(mirror.keys())
        .filter(|k| world.get(*k).copied().unwrap_or(0) != mirror.get(*k).copied().unwrap_or(0))
        .cloned()
        .collect()
}

/// `_enemy_alignment`: for each of the world's enemies, the carried mirror's
/// slot for it, or `None` (no counterpart, or its kind is unalignable).
#[must_use]
pub fn enemy_alignment(world_wire: &Value, mirror_wire: &Value) -> Vec<Option<usize>> {
    let carried = wire_enemy_index(mirror_wire);
    let enc = list(world_wire, "encounter");
    let blocked = unalignable_kinds(world_wire, mirror_wire);
    let mut out: Vec<Option<usize>> = enc
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if blocked.contains(&str_of(get(e, "kind"))) {
                None
            } else {
                carried.get(&enemy_slot_key(e, i, enc)).copied()
            }
        })
        .collect();
    pair_living_by_hp(enc, list(mirror_wire, "encounter"), &mut out);
    align_missed_kills(enc, list(mirror_wire, "encounter"), &mut out);
    out
}

/// Re-pair the enemies of a kind that died in the world while the carried
/// fight still has them alive (a kill the mirror missed). The world board
/// here carries the corpses the game reports removed (`removed_roster`,
/// GH #1716), each in its slot, so it holds a `#dead` entry the carried
/// fight numbers as living: their living counts differ, the kind is
/// unalignable, and every one of them lost its carried state, its uid with
/// it. The intent log then followed the next rebuild's uid shift onto the
/// wrong creatures (Spire Bench seeds 180000392 / 180000817: the dead
/// Scroll's history went to its neighbour; 180001103: The Lost's dropped).
///
/// * The kind has as many entries in both (no summon): a corpse keeps its
///   slot, so the kind's list order names the same creatures.
/// * Every carried one of the kind died and the world has more: the world's
///   living ones are summons, its corpses the carried ones (an Axebot's
///   Stock respawn, `StockPower.AfterDeath`; seed 180001314: the respawn
///   took its dead owner's place and history).
///
/// Anything else stays as the `#nth` pass left it.
fn align_missed_kills(world: &[Value], carried: &[Value], out: &mut [Option<usize>]) {
    let dead = |e: &Value| truthy(get(e, "is_dead"));
    let kinds: BTreeSet<String> = world
        .iter()
        .filter(|e| dead(e))
        .map(|e| str_of(get(e, "kind")))
        .collect();
    for kind in kinds {
        let of = |src: &[Value]| -> Vec<usize> {
            (0..src.len())
                .filter(|&i| str_of(get(&src[i], "kind")) == kind)
                .collect()
        };
        let (ws, cs) = (of(world), of(carried));
        let w_dead = ws.iter().filter(|&&i| dead(&world[i])).count();
        let c_dead = cs.iter().filter(|&&i| dead(&carried[i])).count();
        if w_dead <= c_dead {
            continue;
        }
        if ws.len() == cs.len() {
            for (&w, &c) in ws.iter().zip(&cs) {
                out[w] = Some(c);
            }
        } else if w_dead == cs.len() {
            let mut was = cs.iter().copied();
            for &w in &ws {
                out[w] = if dead(&world[w]) { was.next() } else { None };
            }
        }
    }
}

/// Re-pair the living enemies of one kind by HP where it is unambiguous.
/// The `#nth` numbering assumes both worlds list a kind in the same order,
/// but a summon can land in front of its summoner on the game's board
/// (Two-Tailed Rat, GH #1825) while the mirror lists it behind: each
/// creature then took the other's hidden state and move history. An HP
/// (and max HP) that exactly one world enemy and exactly one carried enemy
/// of the kind show pins that pair; the rest keep the `#nth` order among
/// themselves.
fn pair_living_by_hp(world: &[Value], carried: &[Value], out: &mut [Option<usize>]) {
    let hp = |e: &Value| (py::int_of(get(e, "hp")), py::int_of(get(e, "max_hp")));
    let living = |e: &Value| !truthy(get(e, "is_dead"));
    let kinds: BTreeSet<String> = world
        .iter()
        .filter(|e| living(e))
        .map(|e| str_of(get(e, "kind")))
        .collect();
    for kind in kinds {
        let ws: Vec<usize> = (0..world.len())
            .filter(|&i| living(&world[i]) && str_of(get(&world[i], "kind")) == kind)
            .collect();
        if ws.len() < 2 || ws.iter().any(|&i| out[i].is_none()) {
            continue;
        }
        let cs: Vec<usize> = (0..carried.len())
            .filter(|&i| living(&carried[i]) && str_of(get(&carried[i], "kind")) == kind)
            .collect();
        if cs.len() != ws.len() {
            continue;
        }
        let unique = |list: &[usize], src: &[Value], key| {
            list.iter().filter(|&&i| hp(&src[i]) == key).count() == 1
        };
        let mut pinned: Vec<Option<usize>> = vec![None; ws.len()];
        for (k, &w) in ws.iter().enumerate() {
            let key = hp(&world[w]);
            if unique(&ws, world, key) && unique(&cs, carried, key) {
                pinned[k] = cs.iter().copied().find(|&c| hp(&carried[c]) == key);
            }
        }
        let mut rest = cs.iter().copied().filter(|c| !pinned.contains(&Some(*c)));
        for (k, &w) in ws.iter().enumerate() {
            out[w] = pinned[k].or_else(|| rest.next());
        }
    }
}

/// The world's board with the carried fight's corpses it dropped spliced
/// back in, each just before the first enemy that followed it in the carried
/// fight (GH #1656). The game removes an ordinary corpse from its board; the
/// engine keeps it in its slot (`removed_after_death`, stable NN slots). A
/// rebuild from the bare board loses it, so every later enemy moves up a
/// slot and is renumbered: a layout the student never trained on. The
/// world's own entries keep their order; `uid`s are renumbered by position,
/// as a fresh board numbers them.
#[must_use]
pub fn with_carried_corpses(world_wire: &Value, mirror_wire: &Value) -> Value {
    let carried = list(mirror_wire, "encounter");
    let slots = enemy_alignment(world_wire, mirror_wire);
    let missing: Vec<usize> = (0..carried.len())
        .filter(|i| truthy(get(&carried[*i], "is_dead")) && !slots.contains(&Some(*i)))
        .collect();
    if missing.is_empty() {
        return world_wire.clone();
    }
    let world = list(world_wire, "encounter");
    // Each corpse goes after the world enemy that preceded it latest in the
    // carried fight, then before the next one that followed it. On a board
    // in carried order that is "before the first that followed it"; a summon
    // the game puts in front (Two-Tailed Rat) followed a corpse that stays
    // behind it (GH #1837).
    let mut before: Vec<Vec<usize>> = vec![Vec::new(); world.len() + 1];
    for i in missing {
        let anchor = slots
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.filter(|s| *s < i).map(|s| (s, k)))
            .max()
            .map_or(0, |(_, k)| k + 1);
        let at = (anchor..world.len())
            .find(|&k| slots[k].is_some_and(|s| s > i))
            .unwrap_or(world.len());
        before[at].push(i);
    }
    let mut out: Vec<Value> = Vec::with_capacity(world.len() + carried.len());
    for (k, e) in world.iter().enumerate() {
        out.extend(before[k].iter().map(|&i| carried[i].clone()));
        out.push(e.clone());
    }
    out.extend(before[world.len()].iter().map(|&i| carried[i].clone()));
    for (i, e) in out.iter_mut().enumerate() {
        if let Some(m) = e.as_object_mut() {
            m.insert("uid".into(), Value::from(i + 1));
        }
    }
    let mut spliced = world_wire.clone();
    spliced["encounter"] = Value::Array(out);
    spliced
}

/// How many removed corpses a roster holds.
fn removed_corpses(encounter: &[Value]) -> usize {
    encounter
        .iter()
        .filter(|e| truthy(get(e, "is_dead")) && truthy(get(e, "removed_after_death")))
        .count()
}

/// `seeded` with the corpses the board itself reports removed (GH #1716):
/// the world wire's `removed_roster`, its board with each enemy that died
/// and left it put back in place. `seeded` is the board a fight is about to
/// be seeded from, with whatever corpses a carried fight gave it. A carried
/// fight knows its own corpses and wins; the report fills in where it holds
/// more of them, which is a fight seeded with nothing carried (an enemy that
/// died before the first decision) or one whose carried state missed a kill.
#[must_use]
pub fn with_reported_corpses(world_wire: &Value, seeded: Value) -> Value {
    let roster = list(world_wire, "removed_roster");
    if removed_corpses(roster) <= removed_corpses(list(&seeded, "encounter")) {
        return seeded;
    }
    with_carried_corpses(world_wire, &json!({ "encounter": roster }))
}

/// Relics whose undisclosed counter `state_from_dict` derives from the wire
/// (`sts2_wire::state_dict`): Red Skull from HP, Toy Box as spent, and,
/// where the board carries `cards_played_this_turn`, Ripple Basin's Attacks
/// and Brilliant Scarf's plays this turn. A carry must not override them:
/// the carried mirror's Ripple Basin count is the mirror's own, and a mirror
/// that missed an Attack the board counts gave no turn-end block (Spire Bench
/// seeds 180000177 / 180000614: the board counted 1 Attack, the carry put 0
/// back, real S_HP_LOSS_END 4 HP over native).
fn derived_relic_counter(wire: &Value, id: &str) -> bool {
    match id {
        "RED_SKULL" | "TOY_BOX" => true,
        "RIPPLE_BASIN" | "BRILLIANT_SCARF" => {
            mget(obj(wire, "player"), "cards_played_this_turn").is_array()
        }
        _ => false,
    }
}

/// `_undisclosed_relic_counters`: relics the wire named without a `counter`,
/// minus those `state_from_dict` derives.
#[must_use]
pub fn undisclosed_relic_counters(wire: &Value) -> Vec<String> {
    as_list(mget(obj(wire, "player"), "relics"))
        .iter()
        .filter(|r| r.is_object() && truthy(get(r, "id")) && get(r, "counter").is_null())
        .map(|r| str_of(get(r, "id")))
        .filter(|id| !derived_relic_counter(wire, id))
        .collect()
}

/// `_undisclosed_power_counters`: powers on either side the wire named
/// without a `counter`, minus those `state_from_dict` derives. Sorted, unique.
#[must_use]
pub fn undisclosed_power_counters(wire: &Value) -> Vec<String> {
    let sides = std::iter::once(list(get(wire, "player"), "powers"))
        .chain(list(wire, "encounter").iter().map(|e| list(e, "powers")));
    let mut out = BTreeSet::new();
    for powers in sides {
        for p in powers {
            if p.is_object() && truthy(get(p, "name")) && get(p, "counter").is_null() {
                let name = str_of(get(p, "name"));
                if !DERIVED_POWER_COUNTERS.contains(&name.as_str()) {
                    out.insert(name);
                }
            }
        }
    }
    out.into_iter().collect()
}

/// `_relocate_parked`: re-point a stale parked choice's options at the
/// reseeded piles, matching `(id, upgraded)` one-to-one in option order.
/// An option found nowhere is dropped; `None` when none survives.
///
/// Each option is looked for in the pile it was parked from FIRST (at its
/// old index, then anywhere in that pile), and only then in the others.
/// The choice is still open, so its cards have not moved: a Headbutt
/// (`put_back`) offers DISCARD cards, and a search in fixed hand -> draw
/// order re-pointed every option that also had a copy in hand or draw at
/// that copy, so the student read the wrong `choice_pile` / `hand_slot` and
/// the pick moved the wrong card (GH corpus-compare ACT_CHOOSE class).
#[must_use]
pub fn relocate_parked(parked: &Value, seeded: &Value) -> Option<Value> {
    let deck = get(seeded, "deck");
    let mut used: BTreeSet<(&str, usize)> = BTreeSet::new();
    let mut cards = Vec::new();
    for opt in list(parked, "cards") {
        let (id, up) = (get(opt, "id"), truthy(get(opt, "upgraded")));
        let home = get(opt, "pile").as_str();
        let old_index = get(opt, "index")
            .as_u64()
            .and_then(|i| usize::try_from(i).ok());
        let matches = |loc: &str, idx: usize, wc: &Value| {
            !used.contains(&(loc, idx))
                && wc.is_object()
                && py::eq(get(wc, "id"), id)
                && truthy(get(wc, "upgraded")) == up
        };
        let in_pile = |key: &str, loc: &'static str| {
            let pile = list(deck, key);
            old_index
                .filter(|_| home == Some(loc))
                .and_then(|i| pile.get(i).map(|wc| (i, wc)))
                .filter(|(i, wc)| matches(loc, *i, wc))
                .map(|(i, _)| (loc, i))
                .or_else(|| {
                    pile.iter()
                        .enumerate()
                        .find_map(|(idx, wc)| matches(loc, idx, wc).then_some((loc, idx)))
                })
        };
        let found = WIRE_PILE_TO_LOCATOR
            .iter()
            .filter(|(_, loc)| home == Some(*loc))
            .chain(
                WIRE_PILE_TO_LOCATOR
                    .iter()
                    .filter(|(_, loc)| home != Some(*loc)),
            )
            .find_map(|(key, loc)| in_pile(key, loc));
        let Some(found) = found else { continue };
        used.insert(found);
        let mut card: Map<String, Value> = as_obj(opt).clone();
        card.insert("pile".into(), json!(found.0));
        card.insert("index".into(), json!(found.1));
        cards.push(Value::Object(card));
    }
    if cards.is_empty() {
        return None;
    }
    let mut out = as_obj(parked).clone();
    out.insert("cards".into(), Value::Array(cards));
    Some(Value::Object(out))
}

/// `_choice_effect_in_flight`: is this diff exactly the just-picked card's
/// grant not yet landed in the world's hand? Only a lone `$.deck.hand` diff
/// qualifies, every differing card must be one the pick named, and the
/// mirror must be the side that is ahead (holds the card, or the same card
/// at a strictly lower cost).
#[must_use]
pub fn choice_effect_in_flight(desyncs: &[Desync], picked: &[Value]) -> bool {
    let [only] = desyncs else { return false };
    if picked.is_empty() || only.path != "$.deck.hand" {
        return false;
    }
    let (extra, missing) = multiset_difference(as_list(&only.mirror), as_list(&only.world));
    if extra.is_empty() || missing.len() > extra.len() {
        return false;
    }
    let named: Vec<(String, bool)> = picked
        .iter()
        .map(|o| (str_of(get(o, "id")), truthy(get(o, "upgraded"))))
        .collect();
    let field = |c: &Value, i: usize| as_list(c).get(i).cloned().unwrap_or(Value::Null);
    let is_named = |c: &Value| {
        named.iter().any(|(id, up)| {
            field(c, 0).as_str() == Some(id.as_str()) && py::eq(&field(c, 1), &Value::Bool(*up))
        })
    };
    if !extra.iter().chain(&missing).all(|c| is_named(c)) {
        return false;
    }
    missing.iter().all(|c| {
        extra.iter().any(|e| {
            py::eq(&field(e, 0), &field(c, 0))
                && py::eq(&field(e, 1), &field(c, 1))
                && matches!((field(e, 2).as_i64(), field(c, 2).as_i64()), (Some(a), Some(b)) if a < b)
        })
    })
}

/// `Counter(a) - Counter(b)` and `Counter(b) - Counter(a)`, by Python value.
fn multiset_difference<'a>(a: &'a [Value], b: &'a [Value]) -> (Vec<&'a Value>, Vec<&'a Value>) {
    let mut left: Vec<&Value> = a.iter().collect();
    let mut right = Vec::new();
    for v in b {
        if let Some(i) = left.iter().position(|x| py::eq(x, v)) {
            left.remove(i);
        } else {
            right.push(v);
        }
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ripple Basin's count is derived from the board's Attack bucket when
    /// the board has one, so the carried mirror's count must not replace it.
    #[test]
    fn a_derived_relic_counter_is_not_carried() {
        let relics = json!([
            {"id": "RIPPLE_BASIN", "counter": null},
            {"id": "RED_SKULL", "counter": null},
            {"id": "LIZARD_TAIL", "counter": null},
        ]);
        let with_bucket = json!({"player": {
            "relics": relics, "cards_played_this_turn": [1, 0, 0, 0, 0, 0],
        }});
        assert_eq!(
            undisclosed_relic_counters(&with_bucket),
            vec!["LIZARD_TAIL".to_owned()]
        );
        let without = json!({"player": {"relics": relics}});
        assert_eq!(
            undisclosed_relic_counters(&without),
            vec!["RIPPLE_BASIN".to_owned(), "LIZARD_TAIL".to_owned()],
        );
    }

    fn corpse_test_enemy(kind: &str, dead: bool) -> Value {
        json!({"kind": kind, "is_dead": dead, "uid": 0})
    }

    fn hp_enemy(kind: &str, hp: i64, dead: bool) -> Value {
        json!({"kind": kind, "hp": hp, "max_hp": 40, "is_dead": dead, "uid": 0})
    }

    /// Spire Bench seeds 180000392 / 180000817: the world killed the first
    /// of three Scrolls the mirror still had alive. With the reported corpse
    /// in its slot, each Scroll is the carried one in the same place.
    #[test]
    fn a_kill_the_mirror_missed_keeps_every_slot_aligned() {
        let carried = json!({"encounter": [
            hp_enemy("ScrollOfBiting", 37, false),
            hp_enemy("ScrollOfBiting", 33, false),
            hp_enemy("ScrollOfBiting", 36, false),
        ]});
        let world = json!({"encounter": [
            hp_enemy("ScrollOfBiting", 0, true),
            hp_enemy("ScrollOfBiting", 9, false),
            hp_enemy("ScrollOfBiting", 25, false),
        ]});
        assert_eq!(
            enemy_alignment(&world, &carried),
            vec![Some(0), Some(1), Some(2)]
        );
        // Seed 180001103: a lone enemy of its kind.
        let carried = json!({"encounter": [
            hp_enemy("TheLost", 11, false), hp_enemy("TheForgotten", 111, false),
        ]});
        let world = json!({"encounter": [
            hp_enemy("TheLost", 0, true), hp_enemy("TheForgotten", 111, false),
        ]});
        assert_eq!(enemy_alignment(&world, &carried), vec![Some(0), Some(1)]);
    }

    /// Seed 180001314: the mirror's Axebot died in the world and its Stock
    /// respawn stands ahead of the corpse. The corpse is the carried Axebot;
    /// the respawn is new.
    #[test]
    fn a_respawn_is_not_the_creature_it_replaced() {
        let carried = json!({"encounter": [hp_enemy("Axebot", 3, false)]});
        let world = json!({"encounter": [
            hp_enemy("Axebot", 88, false), hp_enemy("Axebot", 0, true),
        ]});
        assert_eq!(enemy_alignment(&world, &carried), vec![None, Some(0)]);
    }

    /// GH #1656: a corpse the board dropped goes back before the first enemy
    /// that followed it in the carried fight; the board's own order stands.
    #[test]
    fn a_dropped_corpse_is_spliced_back_into_its_slot() {
        let carried = json!({"encounter": [
            corpse_test_enemy("A", false), corpse_test_enemy("B", true), corpse_test_enemy("C", false), corpse_test_enemy("D", true),
        ]});
        let world =
            json!({"encounter": [corpse_test_enemy("A", false), corpse_test_enemy("C", false)]});
        let spliced = with_carried_corpses(&world, &carried);
        let enc = list(&spliced, "encounter");
        let got: Vec<(String, bool, i64)> = enc
            .iter()
            .map(|e| {
                (
                    str_of(get(e, "kind")),
                    truthy(get(e, "is_dead")),
                    get(e, "uid").as_i64().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("A".into(), false, 1),
                ("B".into(), true, 2),
                ("C".into(), false, 3),
                ("D".into(), true, 4)
            ]
        );
        // Nothing dropped: the board is used as it is.
        assert_eq!(with_carried_corpses(&carried, &carried), carried);
    }

    fn enemy(kind: &str, dead: bool, hp: i64) -> Value {
        json!({"kind": kind, "is_dead": dead, "hp": hp, "block": 0, "powers": [], "intents": []})
    }

    fn board(enemies: Vec<Value>) -> Value {
        json!({"phase": "PlayerTurn", "round": 1, "player": {"hp": 50}, "deck": {}, "encounter": enemies})
    }

    #[test]
    fn same_kind_enemies_are_numbered_alive_and_dead_apart() {
        let enc = [
            enemy("Nibbit", true, 0),
            enemy("Nibbit", false, 9),
            enemy("Nibbit", false, 4),
        ];
        let keys: Vec<String> = (0..3).map(|i| enemy_slot_key(&enc[i], i, &enc)).collect();
        assert_eq!(keys, ["Nibbit#dead0", "Nibbit#0", "Nibbit#1"]);
    }

    #[test]
    fn a_corpse_only_the_mirror_kept_is_not_a_difference() {
        let mirror = board(vec![enemy("Nibbit", true, 0), enemy("Wurm", false, 30)]);
        let world = board(vec![enemy("Wurm", false, 30)]);
        assert!(diff_observables(&mirror, &world, None, &BTreeSet::new()).is_empty());
        let alive_only_in_mirror = board(vec![enemy("Nibbit", false, 3), enemy("Wurm", false, 30)]);
        let paths: Vec<String> =
            diff_observables(&alive_only_in_mirror, &world, None, &BTreeSet::new())
                .into_iter()
                .map(|d| d.path)
                .collect();
        assert!(
            paths.contains(&"$.encounter.roster".to_owned()),
            "{paths:?}"
        );
    }

    /// A corpse the board dropped is spliced back from the carried fight
    /// (`with_carried_corpses`), and the rebuilt corpse has lost the Strength
    /// its telegraph was baked with. Its intent is on no screen, so that is
    /// not a divergence: it dropped the hidden enemy state of every later
    /// decision of the fight (Spire Bench seed 170000251).
    #[test]
    fn a_removed_corpse_s_telegraph_is_not_an_observable() {
        let dead = |removed: bool, v: Value| {
            board(vec![
                json!({"kind": "TheLost", "is_dead": true, "hp": 0, "intents": [v],
                       "removed_after_death": removed}),
                enemy("TheForgotten", false, 102),
            ])
        };
        let corpse = |v: Value| dead(true, v);
        // A reviver's corpse stays on the board and telegraphs its return.
        let reviver = |n: i64| dead(false, json!({"kind": "Attack", "amount": n}));
        assert_eq!(
            diff_observables(&reviver(7), &reviver(5), None, &BTreeSet::new())[0].path,
            "$.encounter[TheLost#dead0].intents"
        );
        let mirror =
            corpse(json!({"kind": "Attack", "amount": 5, "effective_amount": 7, "times": 2}));
        let world = corpse(json!({"kind": "Attack", "amount": 5, "times": 2}));
        assert!(diff_observables(&mirror, &world, None, &BTreeSet::new()).is_empty());
        // A living enemy's telegraph still is one.
        let alive = |n: i64| {
            board(vec![
                json!({"kind": "TheLost", "hp": 9, "intents": [{"kind": "Attack", "amount": n}]}),
            ])
        };
        assert_eq!(
            diff_observables(&alive(7), &alive(5), None, &BTreeSet::new())[0].path,
            "$.encounter[TheLost#0].intents"
        );
    }

    /// GH #1716: a board that reports a removed corpse is seeded with it in
    /// place; a board whose carried fight already put its corpses back is
    /// left alone, so no corpse appears twice.
    #[test]
    fn a_reported_corpse_is_seeded_once() {
        let removed = |kind: &str| json!({"kind": kind, "is_dead": true, "uid": 0, "removed_after_death": true});
        let mut world =
            json!({"encounter": [corpse_test_enemy("A", false), corpse_test_enemy("C", false)]});
        // No report: unchanged.
        assert_eq!(with_reported_corpses(&world, world.clone()), world);
        world["removed_roster"] = json!([
            corpse_test_enemy("A", false),
            removed("B"),
            corpse_test_enemy("C", false),
        ]);
        let seeded = with_reported_corpses(&world, world.clone());
        let kinds = |v: &Value| -> Vec<(String, bool)> {
            list(v, "encounter")
                .iter()
                .map(|e| (str_of(get(e, "kind")), truthy(get(e, "is_dead"))))
                .collect()
        };
        let want = vec![
            ("A".to_owned(), false),
            ("B".to_owned(), true),
            ("C".to_owned(), false),
        ];
        assert_eq!(kinds(&seeded), want);
        // Seeded again from that board (a carried fight's corpses): no second B.
        assert_eq!(kinds(&with_reported_corpses(&world, seeded.clone())), want);
    }

    #[test]
    fn the_roster_is_alive_first_and_otherwise_positional() {
        let corpse_inline = board(vec![enemy("Para", true, 0), enemy("Obscura", false, 40)]);
        let corpse_last = board(vec![enemy("Obscura", false, 40), enemy("Para", true, 0)]);
        assert!(diff_observables(&corpse_inline, &corpse_last, None, &BTreeSet::new()).is_empty());
        let a = board(vec![enemy("Spectral", false, 40), enemy("Magi", false, 40)]);
        let b = board(vec![enemy("Magi", false, 40), enemy("Spectral", false, 40)]);
        assert_eq!(
            diff_observables(&a, &b, None, &BTreeSet::new())[0].path,
            "$.encounter.roster"
        );
    }

    #[test]
    fn an_attack_compares_its_baked_damage_and_a_missing_times_is_one() {
        let it = |v: Value| board(vec![json!({"kind": "Nibbit", "intents": [v]})]);
        let mirror = it(json!({"kind": "Attack", "amount": 6, "effective_amount": 9}));
        let world = it(json!({"kind": "Attack", "amount": 9, "times": 1}));
        assert!(diff_observables(&mirror, &world, None, &BTreeSet::new()).is_empty());
        let buff = it(json!({"kind": "Buff", "amount": 3}));
        let other_buff = it(json!({"kind": "Buff", "amount": 0}));
        assert!(diff_observables(&buff, &other_buff, None, &BTreeSet::new()).is_empty());
    }

    #[test]
    fn skips_drop_paths() {
        let a = json!({"player": {"hp": 1, "block": 3}});
        let b = json!({"player": {"hp": 2, "block": 4}});
        let skip = BTreeSet::from(["$.player.hp".to_owned()]);
        let d = diff_observables(&a, &b, None, &skip);
        assert_eq!(
            d.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
            ["$.player.block"]
        );
        assert!(diff_observables(&a, &b, Some("$.player."), &BTreeSet::new()).is_empty());
    }

    /// GH #1825: the game lists a summoned Two-Tailed Rat in front of its
    /// summoner, the mirror behind it; each keeps its own counterpart.
    #[test]
    fn same_kind_enemies_pair_by_hp_across_a_reorder() {
        let mirror = board(vec![
            enemy("TwoTailedRat", true, 0),
            enemy("TwoTailedRat", false, 10),
            enemy("TwoTailedRat", false, 19),
        ]);
        let world = board(vec![
            enemy("TwoTailedRat", false, 19),
            enemy("TwoTailedRat", false, 10),
        ]);
        assert_eq!(enemy_alignment(&world, &mirror), [Some(2), Some(1)]);
        // Equal HP leaves the `#nth` order.
        let mirror = board(vec![enemy("Nibbit", false, 9), enemy("Nibbit", false, 9)]);
        assert_eq!(enemy_alignment(&mirror, &mirror), [Some(0), Some(1)]);
    }

    /// GH #1837: rat #5 is summoned in front of the game's board after #1
    /// and #3 died; the carried fight lists it last. Corpse #3 stays behind
    /// #2, where the engine keeps it, not in front of the summon.
    #[test]
    fn a_corpse_stays_behind_a_summon_put_in_front() {
        let carried = board(vec![
            enemy("TwoTailedRat", true, 0),
            enemy("TwoTailedRat", false, 21),
            enemy("TwoTailedRat", true, 0),
            enemy("TwoTailedRat", false, 19),
            enemy("TwoTailedRat", false, 22),
        ]);
        let world = board(vec![
            enemy("TwoTailedRat", false, 22),
            enemy("TwoTailedRat", false, 21),
            enemy("TwoTailedRat", false, 19),
        ]);
        let enc = list(&with_carried_corpses(&world, &carried), "encounter").to_vec();
        let hp: Vec<i64> = enc.iter().map(|e| py::int_of(get(e, "hp"))).collect();
        assert_eq!(hp, [0, 22, 21, 0, 19]);
    }

    #[test]
    fn a_shifted_kind_is_not_aligned() {
        let mirror = board(vec![
            enemy("Nibbit", false, 5),
            enemy("Nibbit", false, 9),
            enemy("Wurm", false, 3),
        ]);
        let world = board(vec![enemy("Nibbit", false, 9), enemy("Wurm", false, 3)]);
        assert_eq!(enemy_alignment(&world, &mirror), [None, Some(2)]);
    }

    #[test]
    fn the_pending_grant_is_deferred_only_when_the_mirror_is_ahead() {
        let card = |cost: Value| json!(["SHIV", false, cost, null, null, []]);
        let pick = [json!({"id": "SHIV", "upgraded": false})];
        let ahead = Desync {
            path: "$.deck.hand".into(),
            mirror: json!([card(Value::Null)]),
            world: json!([]),
        };
        assert!(choice_effect_in_flight(&[ahead], &pick));
        let behind = Desync {
            path: "$.deck.hand".into(),
            mirror: json!([]),
            world: json!([card(Value::Null)]),
        };
        assert!(!choice_effect_in_flight(&[behind], &pick));
        let free = Desync {
            path: "$.deck.hand".into(),
            mirror: json!([card(json!(0))]),
            world: json!([card(json!(1))]),
        };
        assert!(choice_effect_in_flight(&[free], &pick));
    }

    #[test]
    fn a_parked_option_is_relocated_once_per_card() {
        let seeded =
            json!({"deck": {"hand": [{"id": "STRIKE"}], "discard_pile": [{"id": "STRIKE"}]}});
        let parked =
            json!({"kind": "x", "cards": [{"id": "STRIKE"}, {"id": "STRIKE"}, {"id": "DECAY"}]});
        let out = relocate_parked(&parked, &seeded).unwrap();
        let locs: Vec<(String, u64)> = as_list(&out["cards"])
            .iter()
            .map(|c| {
                (
                    c["pile"].as_str().unwrap().to_owned(),
                    c["index"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(locs, [("hand".to_owned(), 0), ("discard".to_owned(), 0)]);
        assert!(relocate_parked(&json!({"cards": [{"id": "DECAY"}]}), &seeded).is_none());
    }

    #[test]
    fn a_parked_option_stays_in_the_pile_it_was_offered_from() {
        // Headbutt's put_back offers the DISCARD pile; the same ids also sit
        // in hand and draw. The reseed must not re-point them there.
        let seeded = json!({"deck": {
            "hand": [{"id": "UNRELENTING"}],
            "draw_pile": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}],
            "discard_pile": [
                {"id": "DEFEND_IRONCLAD"},
                {"id": "STRIKE_IRONCLAD"},
                {"id": "UNRELENTING"},
                {"id": "DEFEND_IRONCLAD"},
            ],
        }});
        let parked = json!({"kind": "put_back", "cards": [
            {"id": "DEFEND_IRONCLAD", "pile": "discard", "index": 0},
            {"id": "STRIKE_IRONCLAD", "pile": "discard", "index": 1},
            {"id": "UNRELENTING", "pile": "discard", "index": 2},
            {"id": "DEFEND_IRONCLAD", "pile": "discard", "index": 3},
        ]});
        let out = relocate_parked(&parked, &seeded).unwrap();
        let locs: Vec<(String, u64)> = as_list(&out["cards"])
            .iter()
            .map(|c| {
                (
                    c["pile"].as_str().unwrap().to_owned(),
                    c["index"].as_u64().unwrap(),
                )
            })
            .collect();
        let d = |i| ("discard".to_owned(), i);
        assert_eq!(locs, [d(0), d(1), d(2), d(3)]);
    }
}
