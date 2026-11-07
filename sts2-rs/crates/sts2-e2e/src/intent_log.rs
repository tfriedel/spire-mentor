//! `ppo.combat_ctx.intent_log.IntentLog`: the per-fight log of each enemy's
//! displayed head intent, built from honest snapshots.
//!
//! On the first snapshot of each new `round` it appends, per enemy (keyed by
//! the opaque `eid`), the displayed head intent; [`IntentLog::history`] gives,
//! aligned with the snapshot's enemies, up to [`MAX_HIST`] earlier entries
//! (most recent first) with `lag` = current round − their round. An enemy's
//! history resets when the species under its `eid` changes.
//!
//! The log only records on a round's first snapshot, so a consumer that
//! drives it must show it the first combat step of every round (the env does,
//! forced steps included).

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::pyjson::{self, as_int, get, get_null, list_or_empty};

/// Earlier entries kept per enemy in `ehist`.
pub const MAX_HIST: usize = 3;
/// The head intent kind of an enemy with no intent.
pub const NONE_KIND: &str = "NONE";

/// `head_intent(enemy)`: the displayed head intent (`kind`, and `amount`,
/// `effective_amount`, `times` when present), or `{"kind": "NONE"}`.
#[must_use]
pub fn head_intent(enemy: &Value) -> Value {
    let ints = list_or_empty(enemy, "intents");
    let Some(i) = ints.first() else {
        return json!({ "kind": NONE_KIND });
    };
    let mut out = Map::new();
    out.insert(
        "kind".to_owned(),
        get(i, "kind").cloned().unwrap_or_else(|| json!("Unknown")),
    );
    for k in ["amount", "effective_amount", "times"] {
        if let Some(v) = get(i, k) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    Value::Object(out)
}

/// Displayed-intent history of one fight.
#[derive(Clone, Debug, Default)]
pub struct IntentLog {
    /// The round of the last snapshot recorded (`None` before the first).
    pub last_round: Option<i64>,
    log: HashMap<String, Vec<(i64, Value)>>,
    species: HashMap<String, Value>,
    /// Species changes under one `eid`.
    pub resets: u32,
}

/// `eid` as a map key (the canonical JSON of the value: ints and strings
/// never collide).
fn eid_key(m: &Value) -> String {
    pyjson::dumps(&get_null(m, "eid"))
}

fn round_of(snap: &Value) -> i64 {
    get(snap, "round").and_then(as_int).unwrap_or(0)
}

impl IntentLog {
    /// A fresh log (a new fight).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the snapshot's intents if it is the first of a new round.
    pub fn update(&mut self, snap: &Value) {
        let rnd = round_of(snap);
        if self.last_round == Some(rnd) {
            return;
        }
        self.last_round = Some(rnd);
        for m in list_or_empty(snap, "enemies") {
            let eid = eid_key(m);
            let kind = get_null(m, "kind");
            if self.species.get(&eid).is_some_and(|k| *k != kind) {
                self.log.insert(eid.clone(), Vec::new());
                self.resets += 1;
            }
            self.species.insert(eid.clone(), kind);
            self.log.entry(eid).or_default().push((rnd, head_intent(m)));
        }
    }

    /// Per enemy of `snap` (same order): up to [`MAX_HIST`] earlier entries,
    /// most recent first, each `{"lag": .., **intent}`.
    #[must_use]
    pub fn history(&self, snap: &Value) -> Value {
        let rnd = round_of(snap);
        let mut out = Vec::new();
        for m in list_or_empty(snap, "enemies") {
            let eid = eid_key(m);
            let mut hist = Vec::new();
            if self.species.get(&eid) == Some(&get_null(m, "kind")) {
                let past: Vec<&(i64, Value)> = self
                    .log
                    .get(&eid)
                    .map(|l| l.iter().filter(|(r, _)| *r < rnd).collect())
                    .unwrap_or_default();
                let start = past.len().saturating_sub(MAX_HIST);
                for (r, i) in past[start..].iter().rev() {
                    let mut e = Map::new();
                    e.insert("lag".to_owned(), json!(rnd - r));
                    if let Value::Object(im) = i {
                        for (k, v) in im {
                            e.insert(k.clone(), v.clone());
                        }
                    }
                    hist.push(Value::Object(e));
                }
            }
            out.push(Value::Array(hist));
        }
        Value::Array(out)
    }

    /// [`Self::update`], then set `snap["ehist"]`; returns `snap`.
    #[must_use]
    pub fn observe(&mut self, mut snap: Value) -> Value {
        self.update(&snap);
        let h = self.history(&snap);
        if let Value::Object(m) = &mut snap {
            m.insert("ehist".to_owned(), h);
        }
        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(round: i64, kind: &str, amount: i64) -> Value {
        json!({"round": round, "enemies": [
            {"eid": 7, "kind": kind, "intents": [{"kind": "Attack", "amount": amount, "times": 1}]},
        ]})
    }

    #[test]
    fn history_lags_and_resets_on_species_change() {
        let mut log = IntentLog::new();
        for r in 1..=5 {
            let _ = log.observe(snap(r, "Louse", r * 10));
        }
        // A second snapshot in round 5 records nothing new.
        let s = log.observe(snap(5, "Louse", 99));
        assert_eq!(
            s["ehist"],
            json!([[
                {"lag": 1, "kind": "Attack", "amount": 40, "times": 1},
                {"lag": 2, "kind": "Attack", "amount": 30, "times": 1},
                {"lag": 3, "kind": "Attack", "amount": 20, "times": 1},
            ]])
        );
        let s = log.observe(snap(6, "Slime", 1));
        assert_eq!(s["ehist"], json!([[]]));
        assert_eq!(log.resets, 1);
    }
}
