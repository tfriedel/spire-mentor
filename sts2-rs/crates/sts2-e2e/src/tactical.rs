//! `ppo.e2e.tactical`: what this turn can still do, from the honest snapshot
//! only. Per enemy the knapsack damage bound of the playable hand
//! (`max_damage`) and whether it kills (`killable`); the HP lost if the turn
//! ended now (`hp_loss`).
//!
//! A card's hit count is the content prior's `HITS` column
//! (`content_prior_v0111.tsv`, embedded as `data/hits_v0111.tsv`).

use std::collections::HashMap;

use serde_json::Value;

use crate::pyjson::{get, list_or_empty, truthy};
use crate::sctx::{py_int, py_str, pysum};
use crate::vocab::{self, Reg};

const HITS_TSV: &str = include_str!("../data/hits_v0111.tsv");
const HITS_TSV_V0113: &str = include_str!("../data/hits_v0113.tsv");
const HITS_TSV_V0114: &str = include_str!("../data/hits_v0114.tsv");

fn hits_table() -> &'static HashMap<u16, f32> {
    static H: crate::build::PerBuild<HashMap<u16, f32>> = crate::build::PerBuild::new();
    // A lookup table built once (a few hundred cards).
    #[allow(clippy::disallowed_methods)]
    H.get(|| {
        let mut m = HashMap::new();
        let tsv = crate::build::pick(HITS_TSV, HITS_TSV_V0113, HITS_TSV_V0114);
        for line in tsv.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, v) = line.split_once('\t').expect("hits tsv");
            // `content_prior.parse`: `float(val)` stored in a float32 table.
            #[allow(clippy::cast_possible_truncation)]
            let x = v.parse::<f64>().expect("hits value") as f32;
            m.insert(id.parse().expect("hits id"), x);
        }
        m
    })
}

/// `hits_of(card_id)`: a card's printed hit count (1 without a repeat).
#[must_use]
pub fn hits_of(card_id: &str) -> i64 {
    let vid = vocab::load().id(Reg::Card, Some(card_id));
    let h = f64::from(hits_table().get(&vid).copied().unwrap_or(0.0)) * 5.0;
    // Python's `round()`: half to even.
    #[allow(clippy::cast_possible_truncation)]
    let r = h.round_ties_even() as i64;
    r.max(1)
}

fn fnum(v: Option<&Value>) -> f64 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        _ => 0.0,
    }
}

/// `_attack_total(i)`: an Attack / DeathBlow intent's `effective_amount`
/// (else `amount`) times `times`.
#[must_use]
pub fn attack_total(i: &Value) -> f64 {
    if !matches!(
        get(i, "kind").and_then(Value::as_str),
        Some("Attack" | "DeathBlow")
    ) {
        return 0.0;
    }
    let eff = get(i, "effective_amount").map_or_else(
        || get(i, "amount").cloned().unwrap_or(Value::from(0)),
        Clone::clone,
    );
    let eff = if truthy(&eff) { fnum(Some(&eff)) } else { 0.0 };
    let times = get(i, "times").filter(|v| truthy(v));
    eff * times.map_or(1.0, |v| fnum(Some(v)))
}

/// `hp_loss(snap)`: unblocked incoming attack damage if the turn ended now.
#[must_use]
pub fn hp_loss(snap: &Value) -> f64 {
    hp_loss_with(snap, 0.0, 0.0)
}

/// `hp_loss(snap, end_block=..., ally_hp=...)`: the HP the player loses to
/// attack intents if the turn ended now, less the block the orbs add first
/// ([`end_turn_block`]) and the HP of a living Osty ([`ally_hp_of`]).
#[must_use]
pub fn hp_loss_with(snap: &Value, end_block: f64, ally_hp: f64) -> f64 {
    let mut inc = 0.0;
    for m in list_or_empty(snap, "enemies") {
        if get(m, "is_dead").is_some_and(truthy) {
            continue;
        }
        inc += pysum(list_or_empty(m, "intents").iter().map(attack_total));
    }
    let block = get(snap, "player").and_then(|p| get(p, "block"));
    let block = block.filter(|b| truthy(b)).map_or(0.0, |b| fnum(Some(b)));
    let unblocked = (inc - block - end_block).max(0.0);
    (unblocked - ally_hp).max(0.0)
}

/// `end_turn_block(player)`: the block the Frost orbs' passives add at the end
/// of the player's turn (`player` = `state_to_dict(bv)["player"]`).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn end_turn_block(player: &Value) -> f64 {
    list_or_empty(player, "orbs")
        .iter()
        .filter(|o| get(o, "type").and_then(Value::as_str) == Some("Frost"))
        .map(|o| int_or0(get(o, "passive")))
        .sum::<i64>() as f64
}

/// `ally_hp_of(player)`: the HP of the player's living Osty, else 0.
#[must_use]
pub fn ally_hp_of(player: &Value) -> f64 {
    let Some(o) = get(player, "osty").filter(|o| truthy(o)) else {
        return 0.0;
    };
    if !get(o, "alive").is_some_and(truthy) {
        return 0.0;
    }
    get(o, "hp")
        .filter(|v| truthy(v))
        .map_or(0.0, |v| fnum(Some(v)))
}

/// `doomed(snap)`: per enemy, 1.0 if it is alive and its Doom is at least
/// its HP (`DoomPower.IsOwnerDoomed`).
#[must_use]
pub fn doomed(snap: &Value) -> Vec<f64> {
    list_or_empty(snap, "enemies")
        .iter()
        .map(|m| {
            let doom = pysum(
                list_or_empty(m, "powers")
                    .iter()
                    .filter(|p| get(p, "name").and_then(Value::as_str) == Some("DOOM"))
                    .map(|p| {
                        get(p, "amount")
                            .filter(|v| truthy(v))
                            .map_or(0.0, |v| fnum(Some(v)))
                    }),
            );
            let hp = get(m, "hp")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            let alive = !get(m, "is_dead").is_some_and(truthy);
            f64::from(u8::from(alive && hp > 0.0 && doom >= hp))
        })
        .collect()
}

fn int_or0(v: Option<&Value>) -> i64 {
    v.filter(|x| truthy(x)).and_then(py_int).unwrap_or(0)
}

/// `max_damage(snap, playable)`: per enemy, the knapsack damage bound.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn max_damage(snap: &Value, playable: &[i64]) -> Vec<f64> {
    let enemies = list_or_empty(snap, "enemies");
    let n = enemies.len();
    let energy = int_or0(get(snap, "player").and_then(|p| get(p, "energy"))).max(0);
    let eu = usize::try_from(energy).unwrap_or(0);
    let mut fixed = vec![0.0f64; n];
    let mut items: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut xcards: Vec<Vec<f64>> = Vec::new();
    for (i, c) in list_or_empty(snap, "hand").iter().enumerate() {
        if !i64::try_from(i).is_ok_and(|ii| playable.contains(&ii)) {
            continue;
        }
        let Some(dmg) = get(c, "dmg")
            .filter(|d| truthy(d))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let h = hits_of(&get(c, "id").map_or_else(|| "None".to_owned(), py_str)) as f64;
        let mut row = vec![0.0f64; n];
        for (k, v) in dmg {
            let Ok(j) = k.parse::<i64>() else { continue };
            let v = fnum(Some(v));
            if (0..i64::try_from(n).unwrap_or(0)).contains(&j) && v > 0.0 {
                row[usize::try_from(j).unwrap_or(0)] = v * h;
            }
        }
        if row.iter().all(|x| *x == 0.0) {
            continue;
        }
        if int_or0(get(c, "base_cost")) < 0 {
            xcards.push(row.iter().map(|x| x / h).collect());
            continue;
        }
        let cost = int_or0(get(c, "cost")).max(0);
        if cost == 0 {
            for (a, b) in fixed.iter_mut().zip(&row) {
                *a += b;
            }
        } else if cost <= energy {
            items.push((usize::try_from(cost).unwrap_or(0), row));
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut dp = vec![0.0f64; eu + 1];
    for j in 0..n {
        dp.fill(0.0);
        for (cost, row) in &items {
            let v = row[j];
            if v <= 0.0 {
                continue;
            }
            for e in (*cost..=eu).rev() {
                let t = dp[e - cost] + v;
                if t > dp[e] {
                    dp[e] = t;
                }
            }
        }
        let mut best = dp[eu];
        for xc in &xcards {
            let per = xc[j];
            if per > 0.0 {
                for (e, d) in dp.iter().enumerate() {
                    let cand = d + per * (eu - e) as f64;
                    // Python's `max(best, cand)` keeps `best` on ties.
                    if cand > best {
                        best = cand;
                    }
                }
            }
        }
        out.push(fixed[j] + best);
    }
    out
}

/// One hand card's live rows per enemy (`_live_rows`).
struct LiveRows {
    /// `dmg_live`: `dmg x hits`.
    att: Vec<f64>,
    /// `gen_live.flat`: damage the play adds beyond its own attack.
    flat: Vec<f64>,
    /// `gen_live.per_energy`: generated damage per energy of an X-cost card.
    per: Vec<f64>,
    /// `gen_live.kills`: the enemy's HP + block where the play's Doom kills it.
    kill: Vec<f64>,
    /// `gen_live.by_x`: by the energy an X-cost play spends (the stars for a
    /// star-X card), its whole X damage per enemy when that is not linear in
    /// X (Tempest, Chemical X, Heavenly Drill, Multi-Cast); it replaces the
    /// linear term. Empty otherwise.
    byx: Vec<Vec<f64>>,
}

/// A `{"<enemy index>": x}` map read into a per-enemy row (positive values only).
fn index_row(m: Option<&Value>, n: usize, mut f: impl FnMut(&Value) -> Option<f64>) -> Vec<f64> {
    let mut row = vec![0.0f64; n];
    if let Some(m) = m.and_then(Value::as_object) {
        for (k, v) in m {
            let Ok(j) = k.parse::<usize>() else { continue };
            if j < n {
                if let Some(x) = f(v).filter(|x| *x > 0.0) {
                    row[j] = x;
                }
            }
        }
    }
    row
}

/// `_live_rows(c, enemies)`.
fn live_rows(c: &Value, enemies: &[Value]) -> LiveRows {
    let n = enemies.len();
    let att = index_row(get(c, "dmg_live"), n, |v| {
        let d = fnum(v.get(0));
        // Python: `if v[0] > 0: float(v[0]) * float(v[1])`.
        (d > 0.0).then(|| d * fnum(v.get(1)))
    });
    let gen = get(c, "gen_live");
    let flat = index_row(gen.and_then(|g| get(g, "flat")), n, |v| Some(fnum(Some(v))));
    let per = index_row(gen.and_then(|g| get(g, "per_energy")), n, |v| {
        Some(fnum(Some(v)))
    });
    let mut kill = vec![0.0f64; n];
    for j in gen.map_or(&[][..], |g| list_or_empty(g, "kills")) {
        let Some(j) = j.as_u64().and_then(|j| usize::try_from(j).ok()) else {
            continue;
        };
        if j < n && !get(&enemies[j], "is_dead").is_some_and(truthy) {
            let m = &enemies[j];
            let hp = get(m, "hp")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            let bl = get(m, "block")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            kill[j] = hp + bl;
        }
    }
    let byx = gen
        .map_or(&[][..], |g| list_or_empty(g, "by_x"))
        .iter()
        .map(|m| index_row(Some(m), n, |v| Some(fnum(Some(v)))))
        .collect();
    LiveRows {
        att,
        flat,
        per,
        kill,
        byx,
    }
}

/// `max_damage_live(snap, playable)`: per enemy, the live knapsack damage
/// bound (GH #1197) over a 2-D budget of energy and stars; see the Python
/// twin's docstring. [`max_damage`] is untouched.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn max_damage_live(snap: &Value, playable: &[i64]) -> Vec<f64> {
    let enemies = list_or_empty(snap, "enemies");
    let n = enemies.len();
    let player = get(snap, "player");
    let energy = int_or0(player.and_then(|p| get(p, "energy"))).max(0);
    let stars = int_or0(player.and_then(|p| get(p, "stars"))).max(0);
    let eu = usize::try_from(energy).unwrap_or(0);
    let su = usize::try_from(stars).unwrap_or(0);
    let mut fixed = vec![0.0f64; n];
    let mut items: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    let mut xe: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> = Vec::new();
    let mut xs: Vec<(usize, Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> = Vec::new();
    for (i, c) in list_or_empty(snap, "hand").iter().enumerate() {
        if !i64::try_from(i).is_ok_and(|ii| playable.contains(&ii)) {
            continue;
        }
        let r = live_rows(c, enemies);
        let cost = int_or0(get(c, "cost")).max(0);
        let star_cost = int_or0(get(c, "star_cost_live"));
        let any = |v: &[f64]| v.iter().any(|x| *x != 0.0);
        if int_or0(get(c, "base_cost")) < 0 {
            let pe: Vec<f64> = r.att.iter().zip(&r.per).map(|(a, b)| a + b).collect();
            if any(&pe) || any(&r.flat) || r.byx.iter().any(|t| any(t)) {
                xe.push((pe, r.flat, r.byx));
            }
            continue;
        }
        if star_cost < 0 {
            let ps: Vec<f64> = if stars > 0 {
                r.att.iter().map(|a| a / stars as f64).collect()
            } else {
                vec![0.0; n]
            };
            if cost <= energy && (any(&ps) || any(&r.flat) || r.byx.iter().any(|t| any(t))) {
                xs.push((usize::try_from(cost).unwrap_or(0), ps, r.flat, r.byx));
            }
            continue;
        }
        let row: Vec<f64> = (0..n)
            .map(|j| (r.att[j] + r.flat[j]).max(r.kill[j]))
            .collect();
        if !any(&row) {
            continue;
        }
        let star_cost = star_cost.max(0);
        if cost == 0 && star_cost == 0 {
            for (a, b) in fixed.iter_mut().zip(&row) {
                *a += b;
            }
        } else if cost <= energy && star_cost <= stars {
            items.push((
                usize::try_from(cost).unwrap_or(0),
                usize::try_from(star_cost).unwrap_or(0),
                row,
            ));
        }
    }
    let w = su + 1;
    let mut out = Vec::with_capacity(n);
    let mut dp = vec![0.0f64; (eu + 1) * w];
    for j in 0..n {
        dp.fill(0.0);
        for (cost, star_cost, row) in &items {
            let v = row[j];
            if v <= 0.0 {
                continue;
            }
            for e in (*cost..=eu).rev() {
                for s in (*star_cost..=su).rev() {
                    let t = dp[(e - cost) * w + s - star_cost] + v;
                    if t > dp[e * w + s] {
                        dp[e * w + s] = t;
                    }
                }
            }
        }
        let mut best = dp[eu * w + su];
        for (pe, flat, byx) in &xe {
            let (p, b) = (pe[j], flat[j]);
            // `by_x[k][j]`, when the card has a table: its whole X damage at k
            // energy spent (0 past the table), replacing `p x k`.
            let table = !byx.is_empty();
            let at_x = |k: usize| byx.get(k).map_or(0.0, |r| r[j]);
            let live = if table {
                b > 0.0 || byx.iter().any(|r| r[j] > 0.0)
            } else {
                p > 0.0 || b > 0.0
            };
            if live {
                for e in 0..=eu {
                    let lin = if table {
                        at_x(eu - e)
                    } else {
                        p * (eu - e) as f64
                    };
                    let cand = dp[e * w + su] + lin + b;
                    if cand > best {
                        best = cand;
                    }
                }
            }
        }
        for (cost, ps, flat, byx) in &xs {
            let (p, b) = (ps[j], flat[j]);
            // As above, by the stars spent.
            let table = !byx.is_empty();
            let at_x = |k: usize| byx.get(k).map_or(0.0, |r| r[j]);
            let live = if table {
                b > 0.0 || byx.iter().any(|r| r[j] > 0.0)
            } else {
                p > 0.0 || b > 0.0
            };
            if live {
                for s in 0..=su {
                    let lin = if table {
                        at_x(su - s)
                    } else {
                        p * (su - s) as f64
                    };
                    let cand = dp[(eu - cost) * w + s] + lin + b;
                    if cand > best {
                        best = cand;
                    }
                }
            }
        }
        out.push(fixed[j] + best);
    }
    out
}

// ---- end of turn (GH #1197). The engine walks the deterministic end of the
// turn (`sts2_wire::emit::end_turn_live`, the snapshot's `end_live`); these do
// the per-hit arithmetic on it. [`hp_loss_with`] stays as it was for the rows
// without the "chars" event.

fn power_amount(powers: &[Value], name: &str) -> f64 {
    pysum(
        powers
            .iter()
            .filter(|p| get(p, "name").and_then(Value::as_str) == Some(name))
            .map(|p| {
                get(p, "amount")
                    .filter(|v| truthy(v))
                    .map_or(0.0, |v| fnum(Some(v)))
            }),
    )
}

/// The index of the unmelted relic `rid` in the player's relic order.
fn relic_index(snap: &Value, rid: &str) -> Option<usize> {
    list_or_empty(snap, "relics").iter().position(|r| {
        get(r, "id").and_then(Value::as_str) == Some(rid)
            && !get(r, "is_melted").is_some_and(truthy)
    })
}

/// `dies_before_acting(snap, player)`: per enemy, 1.0 if it is alive and
/// certain to die before its move resolves (`end_live.dies`): the
/// deterministic end-of-turn damage, then Poison at its turn start. `player`
/// is unused; the engine read the orbs.
#[must_use]
pub fn dies_before_acting(snap: &Value, _player: &Value) -> Vec<f64> {
    let dies: Vec<i64> = get(snap, "end_live")
        .filter(|e| truthy(e))
        .map_or(&[][..], |e| list_or_empty(e, "dies"))
        .iter()
        .filter_map(py_int)
        .collect();
    list_or_empty(snap, "enemies")
        .iter()
        .enumerate()
        .map(|(j, m)| {
            let hp = get(m, "hp")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            let flagged = i64::try_from(j).is_ok_and(|j| dies.contains(&j));
            f64::from(u8::from(
                flagged && !get(m, "is_dead").is_some_and(truthy) && hp > 0.0,
            ))
        })
        .collect()
}

/// `player_doomed(snap)`: 1.0 if the player's Doom is at least its HP
/// (`DoomPower.AfterSideTurnEnd`: it dies at the end of its own turn).
#[must_use]
pub fn player_doomed(snap: &Value) -> f64 {
    let hp = get(snap, "player")
        .and_then(|p| get(p, "hp"))
        .filter(|v| truthy(v))
        .map_or(0.0, |v| fnum(Some(v)));
    let doom = power_amount(list_or_empty(snap, "powers"), "DOOM");
    f64::from(u8::from(hp > 0.0 && doom >= hp))
}

/// `hp_loss_live(snap, player)`: the HP the player loses if the turn ended
/// now, on a "chars" row: `end_live.self_loss`, then hit by hit as
/// `CreatureCmd.Damage` resolves each (Osty, Intangible, the
/// `ModifyHpLostAfterOsty` relics in relic order, Buffer); see the Python
/// twin's docstring.
#[must_use]
pub fn hp_loss_live(snap: &Value, player: &Value) -> f64 {
    let num = |v: Option<&Value>| v.filter(|v| truthy(v)).map_or(0.0, |v| fnum(Some(v)));
    let end = get(snap, "end_live").filter(|e| truthy(e));
    let mut block = num(get(snap, "player").and_then(|p| get(p, "block")));
    block += num(end.and_then(|e| get(e, "block")));
    let mut osty = ally_hp_of(player);
    let powers = list_or_empty(snap, "powers");
    let mut buffer = match end.and_then(|e| get(e, "buffer")) {
        Some(v) => fnum(Some(v)),
        None => power_amount(powers, "BUFFER"),
    };
    let intangible = power_amount(powers, "INTANGIBLE") > 0.0;
    let rod = relic_index(snap, "TUNGSTEN_ROD");
    let rem_at = relic_index(snap, "BEATING_REMNANT");
    let mut left = match (end.and_then(|e| get(e, "remnant")), rem_at) {
        (Some(v), Some(_)) if !v.is_null() => Some(fnum(Some(v))),
        _ => None,
    };
    let rod_first = rod.is_some_and(|r| rem_at.is_none_or(|m| r < m));
    let dies = dies_before_acting(snap, player);
    let mut loss = num(end.and_then(|e| get(e, "self_loss")));
    for (j, m) in list_or_empty(snap, "enemies").iter().enumerate() {
        if get(m, "is_dead").is_some_and(truthy) || dies[j] != 0.0 {
            continue;
        }
        for i in list_or_empty(m, "intents") {
            if !matches!(
                get(i, "kind").and_then(Value::as_str),
                Some("Attack" | "DeathBlow")
            ) {
                continue;
            }
            let eff = get(i, "effective_amount").map_or_else(
                || get(i, "amount").cloned().unwrap_or(Value::from(0)),
                Clone::clone,
            );
            let d = if truthy(&eff) { fnum(Some(&eff)) } else { 0.0 };
            // Python: `int(i.get("times") or 1)`.
            #[allow(clippy::cast_possible_truncation)]
            let times = get(i, "times")
                .filter(|v| truthy(v))
                .map_or(1, |v| fnum(Some(v)) as i64);
            for _ in 0..times {
                let b = block.min(d);
                block -= b;
                let mut u = d - b;
                if osty > 0.0 {
                    let s = u.min(osty);
                    osty -= s;
                    u -= s;
                }
                if intangible && u >= 1.0 {
                    u = 1.0;
                }
                if rod_first {
                    u = (u - 1.0).max(0.0);
                }
                if let Some(l) = left {
                    u = u.min(l);
                }
                if rod.is_some() && !rod_first {
                    u = (u - 1.0).max(0.0);
                }
                if buffer > 0.0 && u > 0.0 {
                    u = 0.0;
                    buffer -= 1.0;
                }
                if let Some(l) = left.as_mut() {
                    *l = (*l - u).max(0.0);
                }
                loss += u;
            }
        }
    }
    loss
}

/// `tactical(snap, actions)`: `(max_damage, killable, hp_loss)`, unscaled.
#[must_use]
pub fn tactical(snap: &Value, actions: &[Value]) -> (Vec<f64>, Vec<f64>, f64) {
    tactical_with(snap, actions, 0.0, 0.0)
}

/// `tactical(snap, actions, end_block=..., ally_hp=...)`.
#[must_use]
pub fn tactical_with(
    snap: &Value,
    actions: &[Value],
    end_block: f64,
    ally_hp: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let playable: Vec<i64> = actions
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("play"))
        .filter_map(|a| get(a, "card_index").and_then(py_int))
        .collect();
    let md = max_damage(snap, &playable);
    let kill = killable(snap, &md);
    (md, kill, hp_loss_with(snap, end_block, ally_hp))
}

/// `tactical_live(snap, actions, player)`: [`tactical`] on a "chars" row
/// (GH #1197): `(max_damage_live, killable, hp_loss_live)` of a
/// [`crate::snapshot::live_snapshot`], unscaled.
#[must_use]
pub fn tactical_live(snap: &Value, actions: &[Value], player: &Value) -> (Vec<f64>, Vec<f64>, f64) {
    let playable: Vec<i64> = actions
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("play"))
        .filter_map(|a| get(a, "card_index").and_then(py_int))
        .collect();
    let md = max_damage_live(snap, &playable);
    let kill = killable(snap, &md);
    (md, kill, hp_loss_live(snap, player))
}

/// Per enemy, 1.0 if it is alive and `md` reaches its HP + block.
fn killable(snap: &Value, md: &[f64]) -> Vec<f64> {
    list_or_empty(snap, "enemies")
        .iter()
        .enumerate()
        .map(|(j, m)| {
            let alive = !get(m, "is_dead").is_some_and(truthy);
            let hp = get(m, "hp")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            let bl = get(m, "block")
                .filter(|v| truthy(v))
                .map_or(0.0, |v| fnum(Some(v)));
            let ehp = hp + bl;
            f64::from(u8::from(alive && ehp > 0.0 && md[j] >= ehp))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits() {
        assert_eq!(hits_of("STRIKE_IRONCLAD"), 1);
        assert_eq!(hits_of("TWIN_STRIKE"), 2);
        assert_eq!(hits_of("NO_SUCH_CARD"), 1);
    }

    fn snap(block: i64, enemies: Value) -> Value {
        serde_json::json!({"player": {"block": block, "energy": 0}, "hand": [], "enemies": enemies})
    }

    fn attacker(hp: i64, dmg: i64, times: i64, powers: Value) -> Value {
        serde_json::json!({
            "hp": hp, "block": 0, "is_dead": false, "powers": powers,
            "intents": [{"kind": "Attack", "amount": dmg, "effective_amount": dmg, "times": times}],
        })
    }

    /// `CreatureCmd.Damage`: the owner's block first, then a living Osty takes
    /// the unblocked damage and the overkill passes on to the owner.
    #[test]
    fn osty_takes_the_unblocked_damage_first() {
        let s = snap(
            5,
            serde_json::json!([attacker(30, 6, 3, serde_json::json!([]))]),
        );
        assert!((hp_loss(&s) - 13.0).abs() < 1e-9);
        let alive = serde_json::json!({"osty": {"alive": true, "hp": 8, "max_hp": 10}});
        let dead = serde_json::json!({"osty": {"alive": false, "hp": 0, "max_hp": 10}});
        assert!((ally_hp_of(&alive) - 8.0).abs() < 1e-9);
        assert!(ally_hp_of(&dead).abs() < 1e-9);
        assert!((hp_loss_with(&s, 0.0, ally_hp_of(&alive)) - 5.0).abs() < 1e-9);
        assert!((hp_loss_with(&s, 0.0, 20.0)).abs() < 1e-9);
    }

    /// `FrostOrb.Passive` at `OrbQueue.BeforeTurnEnd`: block before the enemies act.
    #[test]
    fn frost_passives_block_before_the_enemies_act() {
        let p = serde_json::json!({"orbs": [
            {"type": "Frost", "passive": 2, "evoke": 5},
            {"type": "Lightning", "passive": 3, "evoke": 8},
            {"type": "Frost", "passive": 4, "evoke": 7},
        ]});
        assert!((end_turn_block(&p) - 6.0).abs() < 1e-9);
        let s = snap(
            5,
            serde_json::json!([attacker(30, 6, 3, serde_json::json!([]))]),
        );
        assert!((hp_loss_with(&s, end_turn_block(&p), 0.0) - 7.0).abs() < 1e-9);
    }

    /// `DoomPower.IsOwnerDoomed`: `CurrentHp <= Amount`.
    #[test]
    fn doomed_is_doom_at_least_hp() {
        let doom = |n: i64| serde_json::json!([{"name": "DOOM", "amount": n}]);
        let mut dead = attacker(9, 1, 1, doom(50));
        dead["is_dead"] = Value::Bool(true);
        let s = snap(
            0,
            serde_json::json!([
                attacker(10, 1, 1, doom(10)),
                attacker(10, 1, 1, doom(9)),
                attacker(
                    10,
                    1,
                    1,
                    serde_json::json!([{"name": "WEAK", "amount": 99}])
                ),
                dead,
            ]),
        );
        assert_eq!(doomed(&s), vec![1.0, 0.0, 0.0, 0.0]);
    }

    // ---- `max_damage_live` (GH #1197). The cases mirror the `bound_` tests
    // of `tests/e2e/test_e2e_tactical.py` one for one.

    fn live_snap(energy: i64, stars: i64, hand: Value, enemies: Value) -> Value {
        serde_json::json!({
            "player": {"energy": energy, "stars": stars, "block": 0},
            "hand": hand,
            "enemies": enemies,
        })
    }

    fn foe(hp: i64, block: i64) -> Value {
        serde_json::json!({"hp": hp, "block": block, "is_dead": false, "intents": []})
    }

    fn all(n: usize) -> Vec<i64> {
        (0..i64::try_from(n).expect("small")).collect()
    }

    #[test]
    fn bound_body_slam_counts() {
        // The stored `base_damage` is 0, so the face preview `dmg` is empty.
        let hand = serde_json::json!([
            {"id": "BODY_SLAM", "cost": 1, "base_cost": 1, "dmg": {}, "dmg_live": {"0": [12, 1.0]}},
        ]);
        let s = live_snap(1, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage(&s, &all(1)), vec![0.0]);
        assert_eq!(max_damage_live(&s, &all(1)), vec![12.0]);
    }

    #[test]
    fn bound_fiend_fire_hits() {
        let hand = serde_json::json!([
            {"id": "FIEND_FIRE", "cost": 2, "base_cost": 2, "dmg": {"0": 7}, "dmg_live": {"0": [7, 4.0]}},
            {"id": "STRIKE_IRONCLAD", "cost": 1, "base_cost": 1, "dmg": {"0": 6}, "dmg_live": {"0": [6, 1.0]}},
        ]);
        let s = live_snap(2, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(2)), vec![28.0]);
        // Not playable: no damage.
        assert_eq!(max_damage_live(&s, &[1]), vec![6.0]);
    }

    #[test]
    fn bound_star_budget_limits() {
        let hand = serde_json::json!([
            {"id": "FALLING_STAR", "cost": 0, "base_cost": 0, "star_cost_live": 2, "base_star_cost": 2,
             "dmg": {"0": 8}, "dmg_live": {"0": [8, 1.0]}},
            {"id": "METEOR_SHOWER", "cost": 0, "base_cost": 0, "star_cost_live": 2, "base_star_cost": 2,
             "dmg": {"0": 14}, "dmg_live": {"0": [14, 1.0]}},
        ]);
        let enemies = serde_json::json!([foe(99, 0)]);
        // Three stars pay for one of the two: the bigger one.
        let s = live_snap(3, 3, hand.clone(), enemies.clone());
        assert_eq!(max_damage_live(&s, &all(2)), vec![14.0]);
        let s = live_snap(3, 4, hand.clone(), enemies.clone());
        assert_eq!(max_damage_live(&s, &all(2)), vec![22.0]);
        // Star-X (Stardust) spends whatever stars are left, 5 per star here.
        let mut hand = hand;
        hand.as_array_mut().expect("list").push(serde_json::json!(
            {"id": "STARDUST", "cost": 0, "base_cost": 0, "star_cost_live": -1, "base_star_cost": -1,
             "dmg": {"0": 5}, "dmg_live": {"0": [5, 3.0]}}
        ));
        let s = live_snap(0, 3, hand.clone(), enemies.clone());
        // Meteor Shower (14) + one star of Stardust (5) beats Stardust x3 (15).
        assert_eq!(max_damage_live(&s, &all(3)), vec![19.0]);
        // No stars: neither the star cards nor Stardust.
        let s = live_snap(3, 0, hand, enemies);
        assert_eq!(max_damage_live(&s, &all(3)), vec![0.0]);
    }

    #[test]
    fn bound_osty_dead_zero() {
        // The face preview scores Poke with the player's modifiers; the live
        // map is empty while the Osty is dead.
        let hand = serde_json::json!([
            {"id": "POKE", "cost": 1, "base_cost": 1, "dmg": {"0": 9}, "dmg_live": {}},
        ]);
        let s = live_snap(1, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage(&s, &all(1)), vec![9.0]);
        assert_eq!(max_damage_live(&s, &all(1)), vec![0.0]);
    }

    #[test]
    fn bound_barrage_orbs() {
        let hand = serde_json::json!([
            {"id": "BARRAGE", "cost": 1, "base_cost": 1, "dmg": {"0": 5, "1": 5},
             "dmg_live": {"0": [5, 3.0], "1": [5, 3.0]}},
            {"id": "WHIRLWIND", "cost": 0, "base_cost": -1, "dmg": {"0": 5, "1": 5},
             "dmg_live": {"0": [5, 1.0], "1": [5, 1.0]}},
        ]);
        let s = live_snap(3, 0, hand, serde_json::json!([foe(40, 0), foe(40, 0)]));
        // Barrage x3 (15) + Whirlwind x2 (10) beats Whirlwind x3 (15).
        assert_eq!(max_damage_live(&s, &all(2)), vec![25.0, 25.0]);
    }

    #[test]
    fn bound_end_of_days_doom_kill() {
        let hand = serde_json::json!([
            {"id": "END_OF_DAYS", "cost": 3, "base_cost": 3, "dmg": {}, "dmg_live": {},
             "gen_live": {"flat": {}, "per_energy": {}, "kills": [0]}},
            {"id": "STRIKE_NECROBINDER", "cost": 1, "base_cost": 1, "dmg": {"0": 6, "1": 6},
             "dmg_live": {"0": [6, 1.0], "1": [6, 1.0]}},
        ]);
        let enemies = serde_json::json!([foe(20, 5), foe(50, 0)]);
        // Lethal on the enemy its Doom kills: HP + block.
        let s = live_snap(3, 0, hand.clone(), enemies.clone());
        assert_eq!(max_damage_live(&s, &all(2)), vec![25.0, 6.0]);
        // Unaffordable: no kill.
        let s = live_snap(2, 0, hand, enemies);
        assert_eq!(max_damage_live(&s, &all(2)), vec![6.0, 6.0]);
    }

    #[test]
    fn bound_blade_dance_shivs() {
        let hand = serde_json::json!([
            {"id": "BLADE_DANCE", "cost": 1, "base_cost": 1, "dmg": {}, "dmg_live": {},
             "gen_live": {"flat": {"0": 12.0, "1": 12.0}, "per_energy": {}, "kills": []}},
            {"id": "STRIKE_SILENT", "cost": 1, "base_cost": 1, "dmg": {"0": 6, "1": 6},
             "dmg_live": {"0": [6, 1.0], "1": [6, 1.0]}},
        ]);
        let enemies = serde_json::json!([foe(40, 0), foe(40, 0)]);
        let s = live_snap(1, 0, hand.clone(), enemies.clone());
        assert_eq!(max_damage_live(&s, &all(2)), vec![12.0, 12.0]);
        let s = live_snap(2, 0, hand, enemies);
        assert_eq!(max_damage_live(&s, &all(2)), vec![18.0, 18.0]);
    }

    #[test]
    fn bound_dualcast_evoke() {
        let hand = serde_json::json!([
            {"id": "DUALCAST", "cost": 1, "base_cost": 1, "dmg": {}, "dmg_live": {},
             "gen_live": {"flat": {"0": 16.0}, "per_energy": {}, "kills": []}},
        ]);
        let s = live_snap(1, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(1)), vec![16.0]);
        // Multi-Cast+: X evokes plus one, so per energy and a flat evoke.
        let hand = serde_json::json!([
            {"id": "MULTI_CAST", "cost": 0, "base_cost": -1, "upgraded": true, "dmg": {}, "dmg_live": {},
             "gen_live": {"flat": {"0": 8.0}, "per_energy": {"0": 8.0}, "kills": []}},
        ]);
        let s = live_snap(3, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(1)), vec![32.0]);
    }

    /// Tempest: its overflow evokes by the energy spent (`gen_live.by_x`), not
    /// linear in X; the knapsack reads the table at the energy the X card
    /// gets. Mirrors `test_bound_tempest_by_x` in `test_e2e_tactical.py`.
    #[test]
    fn bound_tempest_by_x() {
        let tempest = |up: bool| {
            serde_json::json!({"id": "TEMPEST", "cost": 0, "base_cost": -1, "upgraded": up,
             "dmg": {}, "dmg_live": {},
             "gen_live": {"flat": {}, "per_energy": {}, "kills": [],
                          "by_x": [{}, {}, {"0": 8.0}, {"0": 16.0}]}})
        };
        let strike = |d: i64| {
            serde_json::json!({"id": "STRIKE_DEFECT", "cost": 1, "base_cost": 1, "dmg": {},
             "dmg_live": {"0": [d, 1.0]}})
        };
        // X=3 (16) beats Strike + X=2 (7 + 8).
        let hand = serde_json::json!([strike(7), tempest(false)]);
        let s = live_snap(3, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(2)), vec![16.0]);
        // Strike + X=2 (9 + 8) beats X=3.
        let hand = serde_json::json!([strike(9), tempest(false)]);
        let s = live_snap(3, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(2)), vec![17.0]);
        // Alone, at X=0 the table's first entry counts.
        let hand = serde_json::json!([{"id": "TEMPEST", "cost": 0, "base_cost": -1, "dmg": {},
             "dmg_live": {}, "gen_live": {"flat": {}, "per_energy": {}, "kills": [],
                                          "by_x": [{"0": 8.0}]}}]);
        let s = live_snap(0, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(1)), vec![8.0]);
    }

    /// An X card with a `by_x` table (Chemical X, Heavenly Drill's doubling):
    /// the table is its whole X damage, replacing `dmg_live x X`; a star-X
    /// card indexes it by the stars spent. Mirrors
    /// `test_bound_x_table_replaces_the_linear_term` in `test_e2e_tactical.py`.
    #[test]
    fn bound_x_table_replaces_the_linear_term() {
        let table = serde_json::json!([{"0": 10.0}, {"0": 15.0}, {"0": 20.0}]);
        let hand = serde_json::json!([{"id": "WHIRLWIND", "cost": 0, "base_cost": -1, "dmg": {},
             "dmg_live": {"0": [5, 1.0]},
             "gen_live": {"flat": {}, "per_energy": {}, "kills": [], "by_x": table}}]);
        let s = live_snap(2, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(1)), vec![20.0]);
        // A 1-cost Strike and X=1 (6 + 15) beat X=2 (20).
        let hand = serde_json::json!([
            {"id": "STRIKE_IRONCLAD", "cost": 1, "base_cost": 1, "dmg": {},
             "dmg_live": {"0": [6, 1.0]}},
            {"id": "WHIRLWIND", "cost": 0, "base_cost": -1, "dmg": {},
             "dmg_live": {"0": [5, 1.0]},
             "gen_live": {"flat": {}, "per_energy": {}, "kills": [], "by_x": table}}]);
        let s = live_snap(2, 0, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(2)), vec![21.0]);
        // Star-X: by the stars spent.
        let hand = serde_json::json!([{"id": "STARDUST", "cost": 0, "base_cost": 0,
             "star_cost_live": -1, "dmg": {}, "dmg_live": {"0": [5, 2.0]},
             "gen_live": {"flat": {}, "per_energy": {}, "kills": [], "by_x": table}}]);
        let s = live_snap(0, 2, hand, serde_json::json!([foe(40, 0)]));
        assert_eq!(max_damage_live(&s, &all(1)), vec![20.0]);
    }

    #[test]
    fn bound_undeclared_unchanged() {
        // The old bound on a fixed hand, with and without the live keys.
        let plain = serde_json::json!([
            {"id": "TWIN_STRIKE", "cost": 1, "base_cost": 1, "dmg": {"0": 5}},
            {"id": "WHIRLWIND", "cost": 0, "base_cost": -1, "dmg": {"0": 5}},
            {"id": "ANGER", "cost": 0, "base_cost": 0, "dmg": {"0": 6}},
            {"id": "STRIKE_IRONCLAD", "cost": 1, "base_cost": 1, "dmg": {"0": 6}},
        ]);
        let s = live_snap(3, 0, plain.clone(), serde_json::json!([foe(100, 0)]));
        assert_eq!(max_damage(&s, &[0, 1, 2]), vec![26.0]);
        let mut live = plain;
        for c in live.as_array_mut().expect("list") {
            c["dmg_live"] = serde_json::json!({"0": [50, 9.0]});
            c["star_cost_live"] = serde_json::json!(0);
            c["gen_live"] =
                serde_json::json!({"flat": {"0": 99.0}, "per_energy": {}, "kills": [0]});
        }
        let s = live_snap(3, 0, live, serde_json::json!([foe(100, 0)]));
        assert_eq!(max_damage(&s, &[0, 1, 2]), vec![26.0]);
    }

    // ---- end of turn (GH #1197). The cases mirror the end-of-turn tests of
    // `tests/e2e/test_e2e_tactical.py` one for one; the mechanics cases build
    // a real engine state, so `end_live` comes from `end_turn_live`.

    /// `_end_fight`: a v0.111.0 Defect fight (SLIMES_WEAK, enemies at
    /// 12/11/36 HP) edited through the wire dict; the env's snapshot of it
    /// and the player dict.
    #[derive(Default)]
    struct EndFight {
        hp: Option<[i64; 3]>,
        block: Option<[i64; 3]>,
        enemy_powers: Vec<(usize, &'static str, i64)>,
        dead: Vec<usize>,
        orbs: Vec<(&'static str, i64, i64)>,
        relics: Vec<&'static str>,
        powers: Vec<(&'static str, i64)>,
        player_block: i64,
        relic_counters: Vec<(&'static str, i64)>,
    }

    impl EndFight {
        fn build(&self) -> (Value, Value) {
            let (_, snap, p) = self.build_with_state();
            (snap, p)
        }

        /// [`Self::build`] and the belief-view state the snapshot is of.
        fn build_with_state(&self) -> (sts2_core::CombatState, Value, Value) {
            use serde_json::json;
            let cs = sts2_wire::scenario::build_scenario_state_for_version(
                sts2_core::GameVersion::V0_111_0,
                5,
                "Defect",
                &vec!["ZAP".to_owned(); 10],
                70,
                80,
                &["CRACKED_CORE".to_owned()],
                &[],
                "SLIMES_WEAK",
                10,
                None,
                None,
            )
            .expect("scenario");
            let mut d = crate::snapshot::state_dict(&cs).expect("dict");
            let enc = d["encounter"].as_array_mut().expect("encounter");
            for (j, m) in enc.iter_mut().enumerate() {
                if let Some(hp) = self.hp {
                    m["hp"] = json!(hp[j]);
                }
                if let Some(b) = self.block {
                    m["block"] = json!(b[j]);
                }
                m["powers"] = Value::Array(
                    self.enemy_powers
                        .iter()
                        .filter(|(k, _, _)| *k == j)
                        .map(|(_, n, a)| json!({"name": n, "amount": a}))
                        .collect(),
                );
                if self.dead.contains(&j) {
                    m["hp"] = json!(0);
                    m["is_dead"] = json!(true);
                }
            }
            let p = &mut d["player"];
            p["orbs"] = Value::Array(
                self.orbs
                    .iter()
                    .map(|(t, ps, e)| json!({"type": t, "passive": ps, "evoke": e}))
                    .collect(),
            );
            p["relics"] = Value::Array(
                self.relics
                    .iter()
                    .map(|r| {
                        let mut d = json!({"id": r});
                        if let Some((_, c)) = self.relic_counters.iter().find(|(k, _)| k == r) {
                            d["counter"] = json!(c);
                        }
                        d
                    })
                    .collect(),
            );
            p["powers"] = Value::Array(
                self.powers
                    .iter()
                    .map(|(n, a)| json!({"name": n, "amount": a}))
                    .collect(),
            );
            p["block"] = json!(self.player_block);
            let cs = sts2_wire::state_from_json(&d).expect("state");
            let bv = crate::snapshot::belief_view(&cs);
            let d_bv = crate::snapshot::state_dict(&bv).expect("dict");
            let snap = crate::snapshot::live_snapshot(
                &crate::env::snapshot_of(&bv, &d_bv, "monster"),
                &bv,
                &d_bv,
            );
            let p = d_bv["player"].clone();
            (bv, snap, p)
        }
    }

    fn atk(dmg: i64, times: i64) -> Value {
        serde_json::json!({"kind": "Attack", "amount": dmg, "effective_amount": dmg, "times": times})
    }

    fn end_enemy(hp: i64, intents: Value, dead: bool) -> Value {
        serde_json::json!({"hp": hp, "block": 0, "intents": intents, "is_dead": dead})
    }

    fn end_snap(
        enemies: Value,
        block: i64,
        hp: i64,
        powers: &[(&str, i64)],
        relics: &[&str],
        end_live: Option<Value>,
    ) -> Value {
        let mut s = serde_json::json!({
            "player": {"energy": 0, "block": block, "hp": hp},
            "powers": powers.iter().map(|(n, a)| serde_json::json!({"name": n, "amount": a})).collect::<Vec<_>>(),
            "relics": relics.iter().map(|r| serde_json::json!({"id": r})).collect::<Vec<_>>(),
            "hand": [],
            "enemies": enemies,
        });
        if let Some(e) = end_live {
            s["end_live"] = e;
        }
        s
    }

    #[test]
    fn poison_lethal_flags_enemy() {
        // Unblockable: the 5 block does not save the 12-HP enemy from 12 Poison.
        let (snap, p) = EndFight {
            block: Some([5, 0, 0]),
            enemy_powers: vec![(0, "POISON", 12)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(dies_before_acting(&snap, &p), vec![1.0, 0.0, 0.0]);
        let (snap, p) = EndFight {
            enemy_powers: vec![(0, "POISON", 11)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(dies_before_acting(&snap, &p), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn poison_accelerant_counts() {
        let (snap, p) = EndFight {
            enemy_powers: vec![(0, "POISON", 7)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(dies_before_acting(&snap, &p), vec![0.0, 0.0, 0.0]);
        // Accelerant 1: two ticks, 7 + 6 = 13 >= 12.
        let (snap, p) = EndFight {
            enemy_powers: vec![(0, "POISON", 7)],
            powers: vec![("ACCELERANT", 1)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(dies_before_acting(&snap, &p), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn poisoned_enemy_attack_dropped() {
        let enemies = serde_json::json!([
            end_enemy(5, serde_json::json!([atk(6, 1)]), false),
            end_enemy(30, serde_json::json!([atk(5, 1)]), false),
        ]);
        let s = end_snap(
            enemies,
            0,
            50,
            &[],
            &[],
            Some(serde_json::json!({"block": 0, "dies": [0]})),
        );
        let none = serde_json::json!({});
        assert_eq!(dies_before_acting(&s, &none), vec![1.0, 0.0]);
        assert_eq!(hp_loss_live(&s, &none), 5.0);
        assert_eq!(hp_loss(&s), 11.0);
    }

    #[test]
    fn glass_kills_all() {
        let mut f = EndFight {
            hp: Some([4, 4, 36]),
            block: Some([0, 1, 0]),
            orbs: vec![("Glass", 4, 8)],
            ..EndFight::default()
        };
        let (snap, p) = f.build();
        assert_eq!(dies_before_acting(&snap, &p), vec![1.0, 0.0, 0.0]);
        // Gold-Plated Cables: the front Glass again, one lower (4 + 3).
        f.relics = vec!["GOLD_PLATED_CABLES"];
        let (snap, p) = f.build();
        assert_eq!(dies_before_acting(&snap, &p), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn lightning_single_enemy() {
        let mut f = EndFight {
            hp: Some([3, 3, 3]),
            orbs: vec![("Lightning", 3, 8)],
            ..EndFight::default()
        };
        let (snap, p) = f.build();
        assert_eq!(dies_before_acting(&snap, &p), vec![0.0, 0.0, 0.0]);
        f.dead = vec![0, 1];
        let (snap, p) = f.build();
        assert_eq!(dies_before_acting(&snap, &p), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn gold_plated_cables_double_frost() {
        let mut f = EndFight {
            orbs: vec![("Frost", 2, 5), ("Frost", 2, 5)],
            ..EndFight::default()
        };
        let (snap, _) = f.build();
        assert_eq!(snap["end_live"]["block"], serde_json::json!(4));
        f.relics = vec!["GOLD_PLATED_CABLES"];
        let (mut snap, p) = f.build();
        assert_eq!(snap["end_live"]["block"], serde_json::json!(6));
        for m in snap["enemies"].as_array_mut().expect("enemies") {
            m["intents"] = serde_json::json!([atk(4, 2)]);
        }
        // 24 incoming, 6 block before the enemies act.
        assert_eq!(hp_loss_live(&snap, &p), 18.0);
    }

    #[test]
    fn plating_orichalcum_cloak_clasp_block() {
        let mut f = EndFight {
            relics: vec![
                "ORICHALCUM",
                "FAKE_ORICHALCUM",
                "CLOAK_CLASP",
                "RIPPLE_BASIN",
            ],
            powers: vec![("PLATING", 4)],
            ..EndFight::default()
        };
        let (snap, _) = f.build();
        let n = i64::try_from(list_or_empty(&snap, "hand").len()).expect("small");
        assert!(n > 0);
        assert_eq!(
            snap["end_live"]["block"],
            serde_json::json!(4 + 6 + 3 + n + 4)
        );
        // Orichalcum and its fake latch only on 0 block.
        f.player_block = 5;
        let (mut snap, p) = f.build();
        assert_eq!(snap["end_live"]["block"], serde_json::json!(4 + n + 4));
        snap["enemies"] =
            serde_json::json!([end_enemy(30, serde_json::json!([atk(20, 1)]), false)]);
        #[allow(clippy::cast_precision_loss)]
        let want = (20 - 5 - (4 + n + 4)).max(0) as f64;
        assert_eq!(hp_loss_live(&snap, &p), want);
    }

    #[test]
    fn buffer_prevents_loss() {
        let enemies = serde_json::json!([end_enemy(30, serde_json::json!([atk(6, 2)]), false)]);
        let none = serde_json::json!({});
        let s = end_snap(enemies.clone(), 0, 50, &[("BUFFER", 1)], &[], None);
        assert_eq!(hp_loss_live(&s, &none), 6.0);
        // A fully blocked hit does not spend the Buffer.
        let s = end_snap(enemies.clone(), 6, 50, &[("BUFFER", 1)], &[], None);
        assert_eq!(hp_loss_live(&s, &none), 0.0);
        let s = end_snap(enemies, 6, 50, &[], &[], None);
        assert_eq!(hp_loss_live(&s, &none), 6.0);
    }

    #[test]
    fn osty_then_intangible_order() {
        let osty = serde_json::json!({"osty": {"alive": true, "hp": 2, "max_hp": 10}});
        // Intangible caps each telegraphed hit at 1: the Osty soaks two, the
        // player loses 1 a hit after it.
        let enemies = serde_json::json!([end_enemy(30, serde_json::json!([atk(1, 4)]), false)]);
        let s = end_snap(enemies, 0, 50, &[("INTANGIBLE", 1)], &[], None);
        assert_eq!(hp_loss_live(&s, &osty), 2.0);
        // An uncapped hit: the Osty takes 4, the overkill 6 is capped to 1 on
        // the player (after the Osty), then Tungsten Rod takes 1 off.
        let osty = serde_json::json!({"osty": {"alive": true, "hp": 4, "max_hp": 10}});
        let enemies = serde_json::json!([end_enemy(30, serde_json::json!([atk(10, 1)]), false)]);
        let s = end_snap(enemies.clone(), 0, 50, &[("INTANGIBLE", 1)], &[], None);
        assert_eq!(hp_loss_live(&s, &osty), 1.0);
        let s = end_snap(
            enemies.clone(),
            0,
            50,
            &[("INTANGIBLE", 1)],
            &["TUNGSTEN_ROD"],
            None,
        );
        assert_eq!(hp_loss_live(&s, &osty), 0.0);
        let s = end_snap(enemies, 0, 50, &[], &["TUNGSTEN_ROD"], None);
        assert_eq!(hp_loss_live(&s, &osty), 5.0);
        assert_eq!(hp_loss_live(&s, &serde_json::json!({})), 9.0);
    }

    #[test]
    fn self_damage_buffer_and_block() {
        use serde_json::json;
        // Constrict (`AfterSideTurnEnd`) hits the player before the enemies act.
        let (mut snap, p) = EndFight {
            powers: vec![("CONSTRICT", 3)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(snap["end_live"]["block"], json!(0));
        assert_eq!(snap["end_live"]["self_loss"], json!(3));
        snap["enemies"] = json!([]);
        assert_eq!(hp_loss_live(&snap, &p), 3.0);
        // Through block first: the enemies then see 5 - 3.
        let (mut snap, p) = EndFight {
            powers: vec![("CONSTRICT", 3)],
            player_block: 5,
            ..EndFight::default()
        }
        .build();
        assert_eq!(snap["end_live"]["block"], json!(-3));
        assert_eq!(snap["end_live"]["self_loss"], json!(0));
        snap["enemies"] = json!([end_enemy(30, json!([atk(6, 1)]), false)]);
        assert_eq!(hp_loss_live(&snap, &p), 4.0);
        // It spends the Buffer stack, so the attack lands in full.
        let (mut snap, p) = EndFight {
            powers: vec![("CONSTRICT", 3), ("BUFFER", 1)],
            ..EndFight::default()
        }
        .build();
        assert_eq!(snap["end_live"]["self_loss"], json!(0));
        assert_eq!(snap["end_live"]["buffer"], json!(0));
        snap["enemies"] = json!([end_enemy(30, json!([atk(6, 1)]), false)]);
        assert_eq!(hp_loss_live(&snap, &p), 6.0);
    }

    #[test]
    fn beating_remnant_relic_order() {
        use serde_json::json;
        let (snap, _) = EndFight {
            relics: vec!["BEATING_REMNANT"],
            ..EndFight::default()
        }
        .build();
        assert_eq!(snap["end_live"]["remnant"], json!(20));
        let enemies = json!([end_enemy(30, json!([atk(10, 3)]), false)]);
        let live = json!({"block": 0, "dies": [], "self_loss": 0, "buffer": 0, "remnant": 12});
        let none = json!({});
        // The budget caps the HP lost: 10, 2, 0.
        let s = end_snap(
            enemies.clone(),
            0,
            50,
            &[],
            &["BEATING_REMNANT"],
            Some(live.clone()),
        );
        assert_eq!(hp_loss_live(&s, &none), 12.0);
        // Remnant before Tungsten Rod: 10 - 1, 3 - 1, 1 - 1.
        let s = end_snap(
            enemies.clone(),
            0,
            50,
            &[],
            &["BEATING_REMNANT", "TUNGSTEN_ROD"],
            Some(live.clone()),
        );
        assert_eq!(hp_loss_live(&s, &none), 11.0);
        // Rod first: 9, then min(9, 3), then 0.
        let s = end_snap(
            enemies.clone(),
            0,
            50,
            &[],
            &["TUNGSTEN_ROD", "BEATING_REMNANT"],
            Some(live.clone()),
        );
        assert_eq!(hp_loss_live(&s, &none), 12.0);
        // The self-damage counts on top of the hits.
        let mut l = live;
        l["self_loss"] = json!(4);
        l["remnant"] = Value::Null;
        let s = end_snap(enemies, 0, 50, &[], &[], Some(l));
        assert_eq!(hp_loss_live(&s, &none), 34.0);
    }

    /// The engine as the oracle: end the turn for real (phase one, phase
    /// two, the enemies' turn start and moves) and compare the HP the player
    /// lost with `hp_loss_live`, over the relic orders of Beating Remnant and
    /// Tungsten Rod, Constrict and Buffer.
    #[test]
    fn hp_loss_live_matches_the_engine() {
        use sts2_core::combat_loop::{
            end_player_turn_phase_one, end_player_turn_phase_two, enemy_turn_exec, enemy_turn_start,
        };
        use sts2_core::StepResult;
        let cases: Vec<EndFight> = vec![
            EndFight::default(),
            EndFight {
                relics: vec!["BEATING_REMNANT", "TUNGSTEN_ROD"],
                relic_counters: vec![("BEATING_REMNANT", 17)],
                ..EndFight::default()
            },
            EndFight {
                relics: vec!["TUNGSTEN_ROD", "BEATING_REMNANT"],
                relic_counters: vec![("BEATING_REMNANT", 17)],
                ..EndFight::default()
            },
            EndFight {
                powers: vec![("CONSTRICT", 3), ("BUFFER", 1)],
                ..EndFight::default()
            },
            EndFight {
                powers: vec![("CONSTRICT", 2), ("BUFFER", 2)],
                relics: vec!["BEATING_REMNANT"],
                relic_counters: vec![("BEATING_REMNANT", 15)],
                player_block: 1,
                ..EndFight::default()
            },
            EndFight {
                relics: vec!["ORICHALCUM", "TUNGSTEN_ROD"],
                powers: vec![("CONSTRICT", 4)],
                ..EndFight::default()
            },
        ];
        let mut lost = Vec::new();
        for (k, f) in cases.iter().enumerate() {
            let (mut s, snap, p) = f.build_with_state();
            let hp0 = s.player.current_hp;
            for step in [
                end_player_turn_phase_one,
                end_player_turn_phase_two,
                enemy_turn_start,
                enemy_turn_exec,
            ] {
                assert!(matches!(step(&mut s), StepResult::Advanced), "case {k}");
            }
            let real = f64::from(hp0 - s.player.current_hp);
            assert_eq!(hp_loss_live(&snap, &p), real, "case {k}");
            lost.push(real);
        }
        // Not vacuous: the relic order changes the loss (2 against 3).
        assert_eq!(lost, vec![5.0, 2.0, 3.0, 5.0, 0.0, 2.0]);
    }

    #[test]
    fn player_doom_flag() {
        let e = serde_json::json!([]);
        assert_eq!(
            player_doomed(&end_snap(e.clone(), 0, 10, &[("DOOM", 10)], &[], None)),
            1.0
        );
        assert_eq!(
            player_doomed(&end_snap(e.clone(), 0, 11, &[("DOOM", 10)], &[], None)),
            0.0
        );
        assert_eq!(player_doomed(&end_snap(e, 0, 10, &[], &[], None)), 0.0);
    }

    #[test]
    fn old_hp_loss_unchanged() {
        let enemies = serde_json::json!([
            end_enemy(5, serde_json::json!([atk(6, 3)]), false),
            end_enemy(0, serde_json::json!([atk(9, 1)]), true),
        ]);
        let plain = end_snap(enemies.clone(), 5, 50, &[], &[], None);
        assert_eq!(hp_loss_with(&plain, 2.0, 3.0), 8.0);
        let live = end_snap(
            enemies,
            5,
            50,
            &[("BUFFER", 3), ("INTANGIBLE", 1)],
            &["TUNGSTEN_ROD"],
            Some(serde_json::json!({"block": 9, "dies": [0]})),
        );
        assert_eq!(hp_loss_with(&live, 2.0, 3.0), 8.0);
    }
}
