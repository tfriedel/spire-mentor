//! The per-run scratch both live lanes keep (GH #1426; was
//! `realgame_ppo.apply_session_context` and the shop-removal carry).
//!
//! A JSON object, so the Python lanes keep their `session` dict: the act
//! graph cache (`last_map`: `[act, map]`), the player's map position
//! (`map_pos`: `[col, row]`), the shop's removal pick
//! (`shop_remove_target`: `[card id, upgraded]`) and the card-skip flag
//! (`card_reward_skipped`). Other keys pass through untouched.

use serde_json::{json, Map, Value};

use crate::py::{self, get, list, truthy};

/// `realgame_ppo._card_id`: a wire card's id in the training vocab's
/// spelling (`RAGE` -> `RAGE_CARD`), or `None` without one.
#[must_use]
pub fn card_id(c: &Value) -> Option<String> {
    let cid = get(c, "id");
    truthy(cid).then(|| sts2_runai::ppo::wire::vocab_card_id(&py::str_of(cid)))
}

fn upgraded(c: &Value) -> bool {
    truthy(get(c, "upgrade_level")) || truthy(get(c, "upgraded"))
}

/// `int(cur.get("row", -1))`, or `None` where Python raises.
fn row_of(cur: &Value) -> Option<i64> {
    match cur.get("row") {
        None => Some(-1),
        Some(v) => int_or_raise(v),
    }
}

fn int_or_raise(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        #[allow(clippy::cast_possible_truncation)]
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `apply_session_context`: fold the per-run scratch into one decision's
/// wire state, and update the scratch from it. Returns the map to patch
/// into the state (`{**state, "map": …}`) where the message arrived without
/// an act graph and the cache has this act's.
///
/// ~15% of live `map_select` messages arrive with an EMPTY map graph, and
/// older mods sent the graph only on `map_select` (76% of live net
/// decisions had no map tokens, 2026-07-12 audit); the act's graph is
/// immutable, so the last good one per act is cached and patched in, with
/// `current` moved to the last position (which also drives `act_floor`).
#[must_use]
pub fn apply_session_context(state: &Value, session: &mut Map<String, Value>) -> Option<Value> {
    let msg_type = get(state, "type");
    let mut patched = None;
    let m = get(state, "map");
    let has_nodes = truthy(get(m, "nodes"));
    let act = get(state, "act").clone();
    let cached_for = |session: &Map<String, Value>| -> Option<Value> {
        let cached = session.get("last_map").filter(|c| truthy(c))?;
        let pair = py::as_list(cached);
        (pair.len() == 2 && py::eq(&pair[0], &act)).then(|| pair[1].clone())
    };
    if msg_type.as_str() == Some("map_select") {
        if has_nodes {
            session.insert("last_map".into(), json!([act, m]));
        } else {
            patched = cached_for(session);
        }
    } else if has_nodes {
        // A new-dialect bridge attaches the act graph to every run-level
        // message: refresh the cache and the position tracker, so a message
        // that still arrives bare keeps the same view.
        let prev = session.get("last_map").cloned();
        session.insert("last_map".into(), json!([act, m]));
        if let Some(prev) = prev.filter(|p| !p.is_null()) {
            let prev_act = py::as_list(&prev).first().cloned().unwrap_or(Value::Null);
            if !py::eq(&prev_act, &act) {
                // Fresh act: the old position belongs to the old graph.
                session.remove("map_pos");
            }
        }
        let cur = get(m, "current");
        if cur.is_object() {
            // `int(...)` raising leaves the position as it was.
            match row_of(cur) {
                Some(row) if row > 0 => {
                    if let (Some(col), Some(row)) = (
                        cur.get("col").and_then(int_or_raise),
                        cur.get("row").and_then(int_or_raise),
                    ) {
                        session.insert("map_pos".into(), json!([col, row]));
                    }
                }
                // The pre-act row-0 node is not a position: act start.
                Some(0) => {
                    session.remove("map_pos");
                }
                _ => {}
            }
        }
    } else if let Some(map) = cached_for(session) {
        let mut map = match map {
            Value::Object(o) => o,
            _ => Map::new(),
        };
        if let Some(pos) = session.get("map_pos").filter(|p| !p.is_null()) {
            let pos = py::as_list(pos);
            if pos.len() == 2 {
                map.insert(
                    "current".into(),
                    json!({"col": py::int_of(&pos[0]), "row": py::int_of(&pos[1])}),
                );
            }
        }
        patched = Some(Value::Object(map));
    }
    // The card-skip flag only bridges a card_reward -> the reward_screen
    // that re-appears; any other decision leaves that context.
    if !matches!(msg_type.as_str(), Some("reward_screen" | "card_reward")) {
        session.remove("card_reward_skipped");
    }
    // The shop-removal target only bridges a shop answer -> the card_select
    // that follows it.
    if msg_type.as_str() != Some("card_select") {
        session.remove("shop_remove_target");
    }
    patched
}

/// `stash_shop_removal_target`: remember which deck card the SHOP surface
/// chose to remove (`picked` is the shop's `shop_remove_card` candidate).
pub fn stash_shop_removal_target(
    session: &mut Map<String, Value>,
    run_state: &Value,
    picked: &Value,
) {
    if get(picked, "action").as_str() != Some("shop_remove_card") {
        return;
    }
    let deck = list(run_state, "master_deck");
    let di = picked.get("deck_idx").map_or(-1, py::int_of);
    if let Some(c) = usize::try_from(di).ok().and_then(|i| deck.get(i)) {
        session.insert(
            "shop_remove_target".into(),
            json!([card_id(c), upgraded(c)]),
        );
    }
}

/// `shop_removal_carry_index`: which option on THIS removal screen is the
/// card the shop already picked (`n_responses`: the decision's replies).
/// CONSUMES the stash whether or not the card is found. A miss returns the
/// warning naming the screen and the deck behind it (GH #592).
#[must_use]
pub fn shop_removal_carry_index(
    state: &Value,
    session: &mut Map<String, Value>,
    n_responses: usize,
) -> (Option<usize>, Option<String>) {
    let Some(target) = session
        .remove("shop_remove_target")
        .filter(|t| !t.is_null())
    else {
        return (None, None);
    };
    let offered = list(state, "cards");
    for (i, c) in offered.iter().enumerate() {
        let key = json!([card_id(c), upgraded(c)]);
        if py::eq(&key, &target) && i < n_responses {
            return (Some(i), None);
        }
    }
    let tag = |c: &Value| {
        let mut name = card_id(c).unwrap_or_else(|| "None".into());
        if upgraded(c) {
            name.push('+');
        }
        match c.get("is_removable") {
            Some(Value::Bool(false)) => name.push_str("(fixed)"),
            None | Some(Value::Null) => name.push_str("(?)"),
            _ => {}
        }
        name
    };
    let deck = list(get(state, "run_state"), "master_deck");
    let t = py::as_list(&target);
    let t_id = t.first().map_or_else(|| "None".to_string(), py::str_of);
    let t_up = t.get(1).is_some_and(truthy);
    let warning = format!(
        "shop removal target {t_id}{} is not on the removal screen ({} offered: {}); \
         the deck behind it was ({}: {}); the screen will re-decide",
        if t_up { "+" } else { "" },
        offered.len(),
        offered.iter().map(tag).collect::<Vec<_>>().join(", "),
        deck.len(),
        deck.iter().map(tag).collect::<Vec<_>>().join(", "),
    );
    (None, Some(warning))
}

/// [`apply_session_context`] for a message that is not the live decision:
/// the coach projecting the map path of a choice the player already made.
/// It reads the session and leaves it as it is. Folding a past message in
/// moved the live map position back (and could drop a pending card skip or
/// shop removal target) at whatever moment the coach's thread got there,
/// so a served run's screen advice depended on thread timing (GH #1784).
#[must_use]
pub fn view_with_session_context(state: &Value, session: &Map<String, Value>) -> Option<Value> {
    apply_session_context(state, &mut session.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_past_message_reads_the_session_without_moving_the_live_position() {
        let mut session = Map::new();
        let graph = |row: i64| json!({"nodes": [{"col": 1, "row": row}], "current": {"col": 1, "row": row}});
        let _ = apply_session_context(
            &json!({"type": "event", "act": 1, "map": graph(5)}),
            &mut session,
        );
        let live = session.clone();
        let past = json!({"type": "map_select", "act": 1, "map": graph(2)});
        let _ = view_with_session_context(&past, &session);
        assert_eq!(session, live);
        let bare = json!({"type": "map_select", "act": 1});
        let map = view_with_session_context(&bare, &session).unwrap();
        assert_eq!(map["current"], json!({"col": 1, "row": 5}));
    }

    #[test]
    fn a_bare_message_gets_the_cached_act_graph_at_the_last_position() {
        let mut session = Map::new();
        let graph = json!({"nodes": [{"col": 1, "row": 1}], "current": {"col": 1, "row": 1}});
        let first = json!({"type": "event", "act": 1, "map": graph});
        let _ = apply_session_context(&first, &mut session);
        assert_eq!(session["map_pos"], json!([1, 1]));
        let bare = json!({"type": "shop", "act": 1});
        let map = apply_session_context(&bare, &mut session).unwrap();
        assert_eq!(map["current"], json!({"col": 1, "row": 1}));
        let next_act = json!({"type": "shop", "act": 2});
        assert!(apply_session_context(&next_act, &mut session).is_none());
    }

    #[test]
    fn the_shop_pick_is_carried_once() {
        let mut session = Map::new();
        let run_state =
            json!({"master_deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH", "upgraded": true}]});
        stash_shop_removal_target(
            &mut session,
            &run_state,
            &json!({"action": "shop_remove_card", "deck_idx": 1}),
        );
        let screen =
            json!({"cards": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH", "upgrade_level": 1}]});
        assert_eq!(
            shop_removal_carry_index(&screen, &mut session, 2),
            (Some(1), None)
        );
        assert_eq!(
            shop_removal_carry_index(&screen, &mut session, 2),
            (None, None)
        );
    }
}
