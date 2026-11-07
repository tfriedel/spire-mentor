//! `ppo.e2e.tactical`: what this turn can still do, from the honest snapshot
//! only. Per enemy the knapsack damage bound of the playable hand
//! (`max_damage`) and whether it kills (`killable`); the HP lost if the turn
//! ended now (`hp_loss`).
//!
//! A card's hit count is the content prior's `HITS` column
//! (`content_prior_v0111.tsv`, embedded as `data/hits_v0111.tsv`).

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::pyjson::{get, list_or_empty, truthy};
use crate::sctx::{py_int, py_str, pysum};
use crate::vocab::{self, Reg};

const HITS_TSV: &str = include_str!("../data/hits_v0111.tsv");

fn hits_table() -> &'static HashMap<u16, f32> {
    static H: OnceLock<HashMap<u16, f32>> = OnceLock::new();
    // A lookup table built once (a few hundred cards).
    #[allow(clippy::disallowed_methods)]
    H.get_or_init(|| {
        let mut m = HashMap::new();
        for line in HITS_TSV.lines() {
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
    let mut inc = 0.0;
    for m in list_or_empty(snap, "enemies") {
        if get(m, "is_dead").is_some_and(truthy) {
            continue;
        }
        inc += pysum(list_or_empty(m, "intents").iter().map(attack_total));
    }
    let block = get(snap, "player").and_then(|p| get(p, "block"));
    let block = block.filter(|b| truthy(b)).map_or(0.0, |b| fnum(Some(b)));
    (inc - block).max(0.0)
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

/// `tactical(snap, actions)`: `(max_damage, killable, hp_loss)`, unscaled.
#[must_use]
pub fn tactical(snap: &Value, actions: &[Value]) -> (Vec<f64>, Vec<f64>, f64) {
    let playable: Vec<i64> = actions
        .iter()
        .filter(|a| get(a, "action").and_then(Value::as_str) == Some("play"))
        .filter_map(|a| get(a, "card_index").and_then(py_int))
        .collect();
    let md = max_damage(snap, &playable);
    let kill = list_or_empty(snap, "enemies")
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
        .collect();
    (md, kill, hp_loss(snap))
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
}
