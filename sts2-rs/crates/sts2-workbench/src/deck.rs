//! Deck grading (M4): port of `workbench/deck_analysis.py` — the three-axis
//! (Damage / Block / Velocity) Monte-Carlo grade over the two static tables
//! the bundle carries (`card_db.json`, `card_tags.json`).
//!
//! Bit-exact with the Python: the rollouts draw from a port of CPython's
//! `random.Random` (MT19937 seeded the way `Random(int)` seeds, `shuffle`
//! through `_randbelow_with_getrandbits`), the play policy is the same
//! first-maximal pick, and every reported number is rounded like Python's
//! `round(x, 1)` (exact, ties to even).
//!
//! The tables are per game build (GH #877): every entry point has a
//! `*_for(version, …)` form reading that build's tables
//! ([`crate::data::data_for`]); the plain form is the default build's.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use sts2_core::GameVersion;

use crate::data::data_for;

pub const DEFAULT_ENERGY: i64 = 3;
pub const DEFAULT_HAND: i64 = 5;
pub const DEFAULT_SAMPLES: i64 = 200;
const SIM_TURNS: i64 = 7;
const SIM_ENEMIES_SOLO: i64 = 1;
const SIM_ENEMIES_GROUP: i64 = 3;
const MAX_PLAYS_PER_TURN: i64 = 40;
const DMG_SCALE: f64 = 40.0;
const BLK_SCALE: f64 = 20.0;
const VEL_SCALE: f64 = 16.0;
const LEAN_DECK: f64 = 15.0;
const LENSES: [(&str, f64, f64, f64); 3] = [
    ("hallway", 0.60, 0.20, 0.20),
    ("boss", 0.34, 0.33, 0.33),
    ("scaling", 0.20, 0.40, 0.40),
];
const DEAD_TYPES: [&str; 2] = ["Curse", "Status"];

pub use sts2_core::py_random::PyRandom;

// ── the tables ───────────────────────────────────────────────────────────

fn card_db(version: GameVersion) -> &'static BTreeMap<String, Value> {
    &data_for(version).deck_card_db
}

fn card_tags(version: GameVersion) -> &'static BTreeMap<String, Value> {
    &data_for(version).deck_card_tags
}

/// `_card_id`: a deck entry's loc-key (a `"<KEY>.title"` suffix and any
/// upgrade marker dropped).
#[must_use]
pub fn card_id(card: &Value) -> String {
    let cid = match card {
        Value::String(s) => s.clone(),
        Value::Object(m) => ["id", "card_id", "name"]
            .iter()
            .find_map(|k| m.get(*k).and_then(Value::as_str).filter(|s| !s.is_empty()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    };
    let cid = cid.rsplit('.').next().unwrap_or("").to_string();
    if cid.contains('+') {
        cid.trim_end_matches('+')
            .split('+')
            .next()
            .unwrap_or("")
            .to_string()
    } else {
        cid
    }
}

/// One deck entry merged with its static stats + tags (`_resolved`).
#[derive(Debug, Clone)]
struct Card {
    title: String,
    ctype: Option<String>,
    rarity: Option<String>,
    cost: i64,
    cost_x: bool,
    damage: i64,
    hits: i64,
    block: i64,
    draw: i64,
    energy: i64,
    strength: i64,
    dexterity: i64,
    per_turn: bool,
    exhaust: bool,
    unplayable: bool,
    aoe: bool,
    upgraded: bool,
    damage_role: String,
    block_role: String,
    velocity: Vec<String>,
}

fn int_of(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n
            .as_i64()
            .unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(crate::render::py::truthy)
}

/// `x or y` on ints: a zero / missing upgraded stat falls back.
fn or_int(up: Option<&Value>, base: i64) -> i64 {
    let u = int_of(up);
    if u != 0 {
        u
    } else {
        base
    }
}

fn resolved(card: &Value, version: GameVersion) -> Card {
    let cid = card_id(card);
    let empty = Map::new();
    let db = card_db(version)
        .get(&cid)
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let tags = card_tags(version)
        .get(&cid)
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let upgraded = card.as_object().is_some_and(|m| truthy(m.get("upgraded")));
    let mut dmg = int_of(db.get("damage"));
    let mut blk = int_of(db.get("block"));
    let mut draw = int_of(db.get("draw"));
    let mut energy = int_of(db.get("energy"));
    let mut strength = int_of(db.get("strength"));
    let mut dexterity = int_of(db.get("dexterity"));
    if upgraded {
        dmg = or_int(db.get("damage_up"), dmg);
        blk = or_int(db.get("block_up"), blk);
        draw = or_int(db.get("draw_up"), draw);
        energy = or_int(db.get("energy_up"), energy);
        strength = or_int(db.get("strength_up"), strength);
        dexterity = or_int(db.get("dexterity_up"), dexterity);
    }
    let ctype = db
        .get("type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            card.as_object()
                .and_then(|m| m.get("type"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        });
    let hits = match db.get("hits") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(1),
        _ => 1,
    };
    Card {
        title: db
            .get("title")
            .and_then(Value::as_str)
            .map_or_else(|| cid.clone(), str::to_string),
        ctype,
        rarity: db.get("rarity").and_then(Value::as_str).map(str::to_string),
        cost: int_of(db.get("cost")),
        cost_x: truthy(db.get("cost_x")),
        damage: dmg,
        hits,
        block: blk,
        draw,
        energy,
        strength,
        dexterity,
        per_turn: truthy(db.get("per_turn")),
        exhaust: truthy(db.get("exhaust")),
        unplayable: truthy(db.get("unplayable")),
        aoe: truthy(db.get("aoe")),
        upgraded,
        damage_role: tags
            .get("damage_role")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_string(),
        block_role: tags
            .get("block_role")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_string(),
        velocity: tags
            .get("velocity")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn is_dead(c: &Card) -> bool {
    c.ctype.as_deref().is_some_and(|t| DEAD_TYPES.contains(&t))
}

fn effective_cost(c: &Card, energy_left: i64) -> i64 {
    if c.cost_x {
        energy_left
    } else {
        c.cost.max(0)
    }
}

fn card_priority(c: &Card, objective: &str) -> i64 {
    if c.ctype.as_deref() == Some("Power") || c.strength != 0 || c.dexterity != 0 {
        return 4;
    }
    if c.draw != 0 || c.energy != 0 {
        return 3;
    }
    if objective == "damage" && c.damage > 0 {
        return 2;
    }
    if objective == "block" && c.block > 0 {
        return 2;
    }
    1
}

#[derive(Default)]
struct Combat {
    strength: i64,
    dexterity: i64,
    str_per_turn: i64,
    blk_per_turn: i64,
}

fn apply_card(c: &Card, st: &mut Combat, enemies: i64) -> (i64, i64, i64, i64) {
    let mut dmg = 0;
    if c.damage > 0 {
        let per_hit = c.damage + st.strength;
        let targets = if c.aoe { enemies } else { 1 };
        dmg = per_hit.max(0) * c.hits * targets;
    }
    let block = if c.block > 0 {
        c.block + st.dexterity
    } else {
        0
    };
    if c.strength != 0 {
        if c.per_turn {
            st.str_per_turn += c.strength;
        } else {
            st.strength += c.strength;
        }
    }
    if c.dexterity != 0 && !c.per_turn {
        st.dexterity += c.dexterity;
    }
    (dmg, block, c.draw, c.energy)
}

struct RolloutOut {
    damage: f64,
    block: f64,
    plays: f64,
    draw_limited: i64,
    energy_limited: i64,
}

#[allow(clippy::too_many_arguments)]
fn rollout(
    deck: &[Card],
    objective: &str,
    enemies: i64,
    turns: i64,
    base_energy: i64,
    base_hand: i64,
    rng: &mut PyRandom,
    no_engine: bool,
) -> RolloutOut {
    // Cards are referenced by deck index; equal cards are interchangeable.
    let mut draw_pile: Vec<usize> = (0..deck.len()).collect();
    rng.shuffle(&mut draw_pile);
    let mut discard: Vec<usize> = Vec::new();
    let mut st = Combat::default();
    let (mut dmg_total, mut blk_total, mut plays_total) = (0_i64, 0_i64, 0_i64);
    let (mut draw_limited, mut energy_limited) = (0_i64, 0_i64);

    fn draw_cards(
        hand: &mut Vec<usize>,
        k: i64,
        draw_pile: &mut Vec<usize>,
        discard: &mut Vec<usize>,
        rng: &mut PyRandom,
    ) {
        for _ in 0..k {
            if draw_pile.is_empty() {
                if discard.is_empty() {
                    return;
                }
                draw_pile.append(discard);
                rng.shuffle(draw_pile);
            }
            if let Some(c) = draw_pile.pop() {
                hand.push(c);
            }
        }
    }

    for _ in 0..turns {
        st.strength += st.str_per_turn;
        blk_total += st.blk_per_turn;
        let mut energy = base_energy;
        let mut hand: Vec<usize> = Vec::new();
        draw_cards(&mut hand, base_hand, &mut draw_pile, &mut discard, rng);
        let mut plays_this_turn = 0;
        while plays_this_turn < MAX_PLAYS_PER_TURN {
            // `max(playable, key=priority)`: the FIRST card (hand order) with
            // the highest priority among the playable ones.
            let mut best: Option<(usize, i64)> = None;
            for (pos, &ci) in hand.iter().enumerate() {
                let c = &deck[ci];
                if is_dead(c) || c.unplayable || effective_cost(c, energy) > energy {
                    continue;
                }
                let p = card_priority(c, objective);
                if best.is_none_or_lt(p) {
                    best = Some((pos, p));
                }
            }
            let Some((pos, _)) = best else {
                let live_in_hand = hand
                    .iter()
                    .any(|&ci| !is_dead(&deck[ci]) && !deck[ci].unplayable);
                if live_in_hand {
                    energy_limited += 1;
                } else if energy > 0 {
                    draw_limited += 1;
                }
                break;
            };
            let ci = hand.remove(pos);
            let card = &deck[ci];
            energy -= effective_cost(card, energy);
            let (d, b, mut drew, mut gained) = apply_card(card, &mut st, enemies);
            dmg_total += d;
            blk_total += b;
            if no_engine {
                drew = 0;
                gained = 0;
            }
            energy += gained;
            plays_total += 1;
            plays_this_turn += 1;
            if drew != 0 {
                draw_cards(&mut hand, drew, &mut draw_pile, &mut discard, rng);
            }
            if !card.exhaust && card.ctype.as_deref() != Some("Power") {
                discard.push(ci);
            }
        }
        discard.extend(
            hand.iter()
                .copied()
                .filter(|&ci| deck[ci].ctype.as_deref() != Some("Power") && !deck[ci].exhaust),
        );
    }
    let t = turns as f64;
    RolloutOut {
        damage: dmg_total as f64 / t,
        block: blk_total as f64 / t,
        plays: plays_total as f64 / t,
        draw_limited,
        energy_limited,
    }
}

trait BestSoFar {
    fn is_none_or_lt(&self, p: i64) -> bool;
}
impl BestSoFar for Option<(usize, i64)> {
    fn is_none_or_lt(&self, p: i64) -> bool {
        match self {
            None => true,
            Some((_, best)) => p > *best,
        }
    }
}

struct Sim {
    damage_solo: f64,
    damage_group: f64,
    block: f64,
    plays: f64,
    plays_base: f64,
    draw_limited: f64,
    energy_limited: f64,
}

#[allow(clippy::too_many_arguments)]
fn simulate(
    deck: &[Card],
    enemies_solo: i64,
    enemies_group: i64,
    turns: i64,
    base_energy: i64,
    base_hand: i64,
    samples: i64,
    seed: u64,
) -> Sim {
    if !deck.iter().any(|c| !is_dead(c)) {
        let b = base_energy as f64;
        return Sim {
            damage_solo: 0.0,
            damage_group: 0.0,
            block: 0.0,
            plays: b,
            plays_base: b,
            draw_limited: 0.0,
            energy_limited: 0.0,
        };
    }
    let mut rng = PyRandom::new(seed);
    let n = samples.max(1);
    let (mut d_solo, mut d_group, mut blk, mut plays, mut plays_base) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut draw_lim, mut energy_lim) = (0.0_f64, 0.0_f64);
    for _ in 0..n {
        let a = rollout(
            deck,
            "damage",
            enemies_solo,
            turns,
            base_energy,
            base_hand,
            &mut rng,
            false,
        );
        let g = rollout(
            deck,
            "damage",
            enemies_group,
            turns,
            base_energy,
            base_hand,
            &mut rng,
            false,
        );
        let b = rollout(
            deck,
            "block",
            enemies_solo,
            turns,
            base_energy,
            base_hand,
            &mut rng,
            false,
        );
        let pb = rollout(
            deck,
            "damage",
            enemies_solo,
            turns,
            base_energy,
            base_hand,
            &mut rng,
            true,
        );
        d_solo += a.damage;
        d_group += g.damage;
        blk += b.block;
        plays += a.plays;
        plays_base += pb.plays;
        draw_lim += a.draw_limited as f64;
        energy_lim += a.energy_limited as f64;
    }
    let nf = n as f64;
    Sim {
        damage_solo: d_solo / nf,
        damage_group: d_group / nf,
        block: blk / nf,
        plays: plays / nf,
        plays_base: plays_base / nf,
        draw_limited: draw_lim / nf,
        energy_limited: energy_lim / nf,
    }
}

/// Python `round(x, 1)`: exact, ties to even.
fn round1(f: f64) -> f64 {
    format!("{f:.1}").parse().unwrap_or(f)
}

fn round2(f: f64) -> f64 {
    format!("{f:.2}").parse().unwrap_or(f)
}

fn saturate(raw: f64, scale: f64) -> f64 {
    round1(100.0 * (1.0 - (-raw.max(0.0) / scale).exp()))
}

/// `analyze_deck`: the full report (`{"empty": true}` for an empty deck),
/// over the default build's tables.
#[must_use]
pub fn analyze_deck(deck: &[Value], energy: i64, hand: i64, samples: i64, seed: u64) -> Value {
    analyze_deck_for(GameVersion::default(), deck, energy, hand, samples, seed)
}

/// `analyze_deck(…, game_version=version)`: [`analyze_deck`] over build
/// `version`'s tables.
#[must_use]
pub fn analyze_deck_for(
    version: GameVersion,
    deck: &[Value],
    energy: i64,
    hand: i64,
    samples: i64,
    seed: u64,
) -> Value {
    let cards: Vec<Card> = deck.iter().map(|c| resolved(c, version)).collect();
    let n = cards.len();
    if n == 0 {
        return json!({"empty": true});
    }
    let sim = simulate(
        &cards,
        SIM_ENEMIES_SOLO,
        SIM_ENEMIES_GROUP,
        SIM_TURNS,
        energy,
        hand,
        samples,
        seed,
    );
    let dmg_blend = 0.5 * sim.damage_solo + 0.5 * sim.damage_group;
    let blk_pt = sim.block;
    let plays = sim.plays;
    let extra_plays = (plays - sim.plays_base).max(0.0);

    let dmg_scalers: Vec<&Card> = cards
        .iter()
        .filter(|c| c.damage_role == "scaling" || c.damage_role == "both")
        .collect();
    let blk_scalers: Vec<&Card> = cards
        .iter()
        .filter(|c| c.block_role == "scaling" || c.block_role == "both")
        .collect();
    let attacks: Vec<&Card> = cards
        .iter()
        .filter(|c| c.ctype.as_deref() == Some("Attack"))
        .collect();
    let aoe_attacks: Vec<&Card> = attacks.iter().copied().filter(|c| c.aoe).collect();

    // Insertion-ordered like the Python dict.
    let mut vel_counts: Vec<(String, i64)> = Vec::new();
    for c in &cards {
        for tag in &c.velocity {
            if let Some(e) = vel_counts.iter_mut().find(|(k, _)| k == tag) {
                e.1 += 1;
            } else {
                vel_counts.push((tag.clone(), 1));
            }
        }
    }
    let vel_cards = cards.iter().filter(|c| !c.velocity.is_empty()).count();
    let dead: Vec<&Card> = cards.iter().filter(|c| is_dead(c)).collect();
    let basics: Vec<&Card> = cards
        .iter()
        .filter(|c| {
            c.rarity.as_deref() == Some("Basic")
                && matches!(c.ctype.as_deref(), Some("Attack" | "Skill"))
                && !c.upgraded
        })
        .collect();

    let count = |k: &str| vel_counts.iter().find(|(t, _)| t == k).map_or(0, |e| e.1) as f64;
    let mut vel_raw = 16.0 * extra_plays
        + 6.0 * count("extra_play")
        + 6.0 * count("cost_reduction")
        + 5.0 * count("tutor");
    let dilution = (LEAN_DECK / n as f64).min(1.0);
    vel_raw *= dilution;
    let damage_score = saturate(dmg_blend, DMG_SCALE);
    let block_score = saturate(blk_pt, BLK_SCALE);
    let velocity_score = saturate(vel_raw, VEL_SCALE);
    let volume = round1((damage_score * block_score * velocity_score).powf(1.0 / 3.0));

    let mut lenses = Map::new();
    for (name, wd, wb, wv) in LENSES {
        lenses.insert(
            name.to_string(),
            json!(round1(
                wd * damage_score + wb * block_score + wv * velocity_score
            )),
        );
    }

    let draw_lim = sim.draw_limited;
    let energy_lim = sim.energy_limited;
    let bottleneck = if draw_lim < 0.25 && energy_lim < 0.25 {
        "none"
    } else if energy_lim >= draw_lim {
        "energy"
    } else {
        "draw"
    };

    let mut curve: Vec<(String, i64)> = Vec::new();
    let mut by_type: Vec<(String, i64)> = Vec::new();
    let mut by_rarity: Vec<(String, i64)> = Vec::new();
    let bump = |v: &mut Vec<(String, i64)>, k: String| {
        if let Some(e) = v.iter_mut().find(|(t, _)| *t == k) {
            e.1 += 1;
        } else {
            v.push((k, 1));
        }
    };
    for c in &cards {
        if !is_dead(c) {
            let key = if c.cost_x {
                "X".to_string()
            } else {
                c.cost.to_string()
            };
            bump(&mut curve, key);
        }
    }
    for c in &cards {
        bump(
            &mut by_type,
            c.ctype.clone().unwrap_or_else(|| "?".to_string()),
        );
        bump(
            &mut by_rarity,
            c.rarity.clone().unwrap_or_else(|| "?".to_string()),
        );
    }
    let obj = |v: Vec<(String, i64)>| -> Value {
        let mut m = Map::new();
        for (k, n) in v {
            m.insert(k, json!(n));
        }
        Value::Object(m)
    };
    let titles = |v: &[&Card]| -> Vec<String> { v.iter().map(|c| c.title.clone()).collect() };

    let warnings = warnings(
        velocity_score,
        blk_pt,
        &dmg_scalers,
        &blk_scalers,
        &aoe_attacks,
        &attacks,
        &dead,
        &basics,
        n,
        bottleneck,
    );

    json!({
        "deck_size": n,
        "assumptions": {"energy": energy, "hand": hand, "samples": samples},
        "scores": {
            "damage": damage_score,
            "block": block_score,
            "velocity": velocity_score,
            "volume": volume,
        },
        "lenses": lenses,
        "damage": {
            "per_turn": round1(dmg_blend),
            "single_target": round1(sim.damage_solo),
            "vs_group": round1(sim.damage_group),
            "scaling_cards": titles(&dmg_scalers),
            "aoe_attacks": titles(&aoe_attacks),
            "attack_count": attacks.len(),
        },
        "block": {
            "per_turn": round1(blk_pt),
            "scaling_cards": titles(&blk_scalers),
        },
        "velocity": {
            "tags": obj(vel_counts),
            "engine_cards": vel_cards,
            "cards_per_turn": round1(plays),
            "extra_per_turn": round1(extra_plays),
            "dilution": round2(dilution),
            "bottleneck": bottleneck,
            "draw_limited_turns": round1(draw_lim),
            "energy_limited_turns": round1(energy_lim),
        },
        "composition": {
            "by_type": obj(by_type),
            "by_rarity": obj(by_rarity),
            "energy_curve": obj(curve),
            "dead_cards": dead.len(),
            "unupgraded_basics": basics.len(),
        },
        "warnings": warnings,
    })
}

#[allow(clippy::too_many_arguments)]
fn warnings(
    vel_s: f64,
    blk_per_turn: f64,
    dmg_scalers: &[&Card],
    blk_scalers: &[&Card],
    aoe: &[&Card],
    attacks: &[&Card],
    dead: &[&Card],
    basics: &[&Card],
    n: usize,
    bottleneck: &str,
) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    fn add(out: &mut Vec<Value>, sev: &str, text: &str) {
        out.push(json!({"severity": sev, "text": text}));
    }
    {
        let mut add = |sev: &str, text: &str| add(&mut out, sev, text);
        if !attacks.is_empty() && dmg_scalers.is_empty() {
            add(
                "warn",
                "No damage scaling — you may stall out against high-HP bosses.",
            );
        }
        if blk_scalers.is_empty() && blk_per_turn < 10.0 {
            add(
                "warn",
                "Thin on block and no block engine — big hits will land in Act 2+.",
            );
        }
        if !attacks.is_empty() && aoe.is_empty() {
            add(
                "info",
                "No AOE — multi-enemy fights (Gremlins, hallways) will be slow.",
            );
        }
        if vel_s < 25.0 {
            add(
                "info",
                "Low velocity — few draw/energy cards; you'll often draw the wrong hand.",
            );
        }
        if bottleneck == "energy" {
            add(
                "info",
                "Energy-limited — cards stranded in hand; more energy helps, more draw won't.",
            );
        } else if bottleneck == "draw" {
            add(
                "info",
                "Draw-limited — hand empties with energy to spare; add card draw, not energy.",
            );
        }
        if n > 0 && (dead.len() + basics.len()) as f64 / n as f64 > 0.45 {
            add(
                "warn",
                "Deck clogged with basics/curses — prioritise removals.",
            );
        }
        if n >= 35 && vel_s < 45.0 {
            add(
                "info",
                "Large deck with low velocity — consistency suffers as it bloats.",
            );
        }
    }
    if out.is_empty() {
        out.push(json!({"severity": "info", "text": "No structural gaps flagged — keep an eye on your next 5-20 turns."}));
    }
    out
}

const AXIS_KEYS: [&str; 4] = ["damage", "block", "velocity", "volume"];

fn scores_of(report: &Value) -> Value {
    match report.get("scores") {
        Some(s) if s.is_object() => s.clone(),
        _ => json!({"damage": 0.0, "block": 0.0, "velocity": 0.0, "volume": 0.0}),
    }
}

/// `reward_preview`: the deck's scores, then the deck + each candidate with
/// the paired-seed deltas (default build's tables).
#[must_use]
pub fn reward_preview(deck: &[Value], candidates: &[Value], samples: i64, seed: u64) -> Value {
    reward_preview_for(GameVersion::default(), deck, candidates, samples, seed)
}

/// [`reward_preview`] over build `version`'s tables.
#[must_use]
pub fn reward_preview_for(
    version: GameVersion,
    deck: &[Value],
    candidates: &[Value],
    samples: i64,
    seed: u64,
) -> Value {
    let base = scores_of(&analyze_deck_for(
        version,
        deck,
        DEFAULT_ENERGY,
        DEFAULT_HAND,
        samples,
        seed,
    ));
    let mut out = Vec::new();
    for cand in candidates {
        let mut with: Vec<Value> = deck.to_vec();
        with.push(cand.clone());
        let sc = scores_of(&analyze_deck_for(
            version,
            &with,
            DEFAULT_ENERGY,
            DEFAULT_HAND,
            samples,
            seed,
        ));
        let mut delta = Map::new();
        for k in AXIS_KEYS {
            let d = sc[k].as_f64().unwrap_or(0.0) - base[k].as_f64().unwrap_or(0.0);
            delta.insert(k.to_string(), json!(round1(d)));
        }
        out.push(json!({
            "index": cand.get("index").cloned().unwrap_or(Value::Null),
            "scores": sc,
            "delta": delta,
        }));
    }
    json!({"base": base, "candidates": out})
}

/// `deck_signature`: an order-independent `(card id, upgraded)` fingerprint.
#[must_use]
pub fn deck_signature(deck: &[Value]) -> Vec<(String, bool)> {
    let mut sig: Vec<(String, bool)> = deck
        .iter()
        .map(|c| {
            (
                card_id(c),
                c.as_object().is_some_and(|m| truthy(m.get("upgraded"))),
            )
        })
        .collect();
    sig.sort();
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_cpython() {
        // Python: random.Random(0).getrandbits(32) → 3626764237, then 1654615998.
        let mut r = PyRandom::new(0);
        assert_eq!(r.getrandbits(32), 3_626_764_237);
        assert_eq!(r.getrandbits(32), 1_654_615_998);
        // random.Random(12345).getrandbits(32) → 1789368711
        let mut r = PyRandom::new(12345);
        assert_eq!(r.getrandbits(32), 1_789_368_711);
        // A seed above 32 bits keys by two chunks: Random(1 << 40) → 445128065.
        let mut r = PyRandom::new(1 << 40);
        assert_eq!(r.getrandbits(32), 445_128_065);
    }

    #[test]
    fn shuffle_matches_cpython() {
        // Python: x = list(range(10)); random.Random(0).shuffle(x) → [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        let mut x: Vec<usize> = (0..10).collect();
        PyRandom::new(0).shuffle(&mut x);
        assert_eq!(x, vec![7, 8, 1, 5, 3, 4, 2, 0, 9, 6]);
    }

    #[test]
    fn card_ids_normalize_like_python() {
        // `cid.split(".")[-1]` keeps the LAST segment — the Python oracle's
        // own quirk (deck entries carry a bare `id`, so it rarely matters).
        assert_eq!(card_id(&json!("STRIKE_IRONCLAD.title")), "title");
        assert_eq!(
            card_id(&json!({"id": "STRIKE_IRONCLAD", "name": "STRIKE_IRONCLAD.title"})),
            "STRIKE_IRONCLAD"
        );
        assert_eq!(card_id(&json!({"id": "BASH+"})), "BASH");
        assert_eq!(card_id(&json!({"name": "X+1"})), "X");
        assert_eq!(card_id(&json!({})), "");
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn empty_deck_and_a_starter_deck() {
        assert_eq!(analyze_deck(&[], 3, 5, 200, 0), json!({"empty": true}));
        let deck: Vec<Value> = ["STRIKE_IRONCLAD"; 5]
            .iter()
            .chain(["DEFEND_IRONCLAD"; 4].iter())
            .chain(["BASH"].iter())
            .map(|id| json!({"id": *id, "upgraded": false}))
            .collect();
        let r = analyze_deck(&deck, 3, 5, 200, 0);
        assert_eq!(r["deck_size"], json!(10));
        assert!(r["scores"]["damage"].as_f64().unwrap() > 0.0);
        assert!(r["warnings"].as_array().unwrap().len() >= 2);
        let sig = deck_signature(&deck);
        assert_eq!(sig[0], ("BASH".to_string(), false));
        let p = reward_preview(&deck, &[json!({"index": 2, "id": "INFLAME"})], 30, 0);
        assert_eq!(p["candidates"][0]["index"], json!(2));
        assert!(p["candidates"][0]["delta"]["damage"].is_number());
    }
}
