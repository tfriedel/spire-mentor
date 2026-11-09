//! The seed checker's game-side core (`seedcheck.html`): a game seed's
//! preview (Neow offers, act 1, bosses) and the named "winning line" of an
//! attempt, for the text verdict. Attempts themselves are
//! [`crate::e2e::attempt`]; run logs are [`run_log`].
//!
//! Everything is Ironclad on v0.111.0, the e2e student's run
//! (`sts2_e2e::env::new_run_from_game_seed`), at the ascension the page picks
//! (the student trained at A10).

use serde_json::{json, Value};
use sts2_core::map::MapCoord;
use sts2_core::run_manager::RunManager;
use sts2_core::GameVersion;
use sts2_e2e::env::{new_run_from_game_seed, E2eEnv, Step};
use sts2_wire::emit::run_manager as emit_rm;

use crate::display_maps::loc_id;
use crate::e2e::{attempt_temp, Seq};
use crate::game::ModeSeed;
use crate::runlog::{write_run_log, RunLog};
use crate::session::{Session, SessionError};
use crate::simlab_catalog::name_of_for;

/// Game seeds the game deals are this long (`SeedHelper.seedDefaultLength`).
pub const GAME_SEED_LENGTH: usize = 12;

/// An encounter id's display name in v0.111.0 (`VANTOM_BOSS` → `Vantom`).
#[must_use]
pub fn encounter_name(id: &str) -> String {
    nm("encounter", id)
}

/// The display name of a card / relic / encounter / event id in v0.111.0.
fn nm(kind: &str, raw: &str) -> String {
    let id = match kind {
        "encounter" => raw.to_string(),
        _ => loc_id(raw),
    };
    name_of_for(GameVersion::V0_111_0, kind, &id)
}

fn neow_ids(rm: &RunManager) -> Vec<String> {
    emit_rm::neow_offers(rm).map_or_else(Vec::new, |(a, b, c)| vec![a, b, c])
}

/// `POST /api/seedcheck/preview {seed}`: what the seed deals, so a mistyped
/// seed is visible before any attempt is played.
///
/// # Errors
///
/// A bad seed.
pub fn preview(seed: &str, ascension: u8) -> Result<Value, String> {
    let canonical = sts2_e2e::env::canonical_game_seed(seed)?;
    let rm = new_run_from_game_seed(&canonical, ascension)?;
    let bosses: Vec<Vec<String>> = emit_rm::generated_rooms(&rm)
        .as_array()
        .map(|acts| {
            acts.iter()
                .map(|r| {
                    ["boss", "second_boss"]
                        .iter()
                        .filter_map(|k| r[*k].as_str())
                        .map(|id| nm("encounter", id))
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(json!({
        "seed": canonical,
        "length_ok": canonical.chars().count() == GAME_SEED_LENGTH,
        "neow": neow_ids(&rm).iter().map(|id| nm("relic", id)).collect::<Vec<_>>(),
        "act1": crate::simlab::title_case(emit_rm::act1(&rm)),
        "bosses": bosses,
        "character": crate::e2e::CHARACTER,
        "ascension": ascension,
        "game_version": crate::e2e::GAME_VERSION,
    }))
}

/// Map raw action indices (a `ppo.e2e.solve` proof) to the merged classes
/// [`E2eEnv::step`] takes.
///
/// # Errors
///
/// A bad seed, an index no class contains, or a log that outlives the run.
pub fn classes_from_actions(
    seed: &str,
    ascension: u8,
    log_idx: &[usize],
) -> Result<Vec<usize>, String> {
    let mut env = E2eEnv::over_run(new_run_from_game_seed(seed, ascension)?, 0);
    let mut step = env.reset();
    let mut out = Vec::with_capacity(log_idx.len());
    for (n, &ai) in log_idx.iter().enumerate() {
        let Step::Decision(d) = step else {
            return Err(format!("the run ended at step {n} of {}", log_idx.len()));
        };
        let class = d
            .members
            .iter()
            .position(|m| m.contains(&ai))
            .ok_or_else(|| format!("step {n}: action {ai} is in no class"))?;
        out.push(class);
        step = env.step(class);
    }
    Ok(out)
}

fn idx_of(a: &Value, key: &str) -> Option<usize> {
    a.get(key)
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
}

fn shop_item(rm: &RunManager, list: &str, idx: usize) -> Option<String> {
    emit_rm::shop_inventory(rm)
        .get(list)?
        .as_array()?
        .iter()
        .find(|e| idx_of(e, "idx") == Some(idx))?
        .get("id")?
        .as_str()
        .map(str::to_string)
}

/// One route entry: the map node at `c` (its room type, its place in its row
/// counted from the left, `pos` of `of`), the direction from `prev` (`left` /
/// `straight` / `right`; none at an act's first node) and how many nodes the
/// move could have reached.
fn node_entry(rm: &RunManager, prev: Option<MapCoord>, c: MapCoord) -> Value {
    let m = emit_rm::full_map(rm);
    let rows: Vec<&Vec<Value>> = m["rows"]
        .as_array()
        .map(|r| r.iter().filter_map(Value::as_array).collect())
        .unwrap_or_default();
    let at = |co: MapCoord| -> Option<&Value> {
        rows.get(usize::try_from(co.row).ok()?)?
            .iter()
            .find(|n| n["col"].as_i64() == Some(i64::from(co.col)))
    };
    let (room, pos, of) = match usize::try_from(c.row)
        .ok()
        .and_then(|r| rows.get(r))
        .filter(|_| at(c).is_some())
    {
        Some(nodes) => {
            let mut cols: Vec<i64> = nodes.iter().filter_map(|n| n["col"].as_i64()).collect();
            cols.sort_unstable();
            let pos = cols
                .iter()
                .position(|&x| x == i64::from(c.col))
                .map_or(0, |p| p + 1);
            let room = at(c)
                .and_then(|n| n["type"].as_str())
                .unwrap_or("Unknown")
                .to_string();
            (room, pos, cols.len())
        }
        None => ("Boss".to_string(), 1, 1),
    };
    let dir = prev
        .filter(|_| room != "Boss")
        .map(|p| match c.col.cmp(&p.col) {
            std::cmp::Ordering::Less => "left",
            std::cmp::Ordering::Equal => "straight",
            std::cmp::Ordering::Greater => "right",
        });
    let options = prev.map_or(of, |p| {
        at(p)
            .and_then(|n| n["children"].as_array())
            .map_or(1, Vec::len)
    });
    json!({
        "room": room, "pos": pos, "of": of, "dir": dir, "options": options,
        "floor": Value::Null, "act": Value::Null, "name": Value::Null, "kind": Value::Null, "notes": [],
    })
}

/// Bring `route` up to the nodes the run has visited (`visited_map_coords`,
/// cleared each act): one entry per new node, the previous one stamped with
/// the HP it was left on. Covers forced moves too, which no decision shows.
fn sync_route(
    rm: &RunManager,
    route: &mut Vec<Value>,
    seen: &mut (u8, usize),
    relics: &mut Vec<String>,
    floor: i64,
    act: i64,
) {
    let rs = &rm.run_state;
    if rs.act_index != seen.0 {
        *seen = (rs.act_index, 0);
    }
    let left = route.len();
    let v = &rs.visited_map_coords;
    if v.len() > seen.1 {
        let newest = v.len() - 1;
        for i in seen.1..v.len() {
            // The node left (and any passed through with no decision) ends
            // on the HP the run arrives at the next one with.
            if let Some(last) = route.last_mut() {
                last["hp"] = json!(rs.current_hp);
                last["max_hp"] = json!(rs.max_hp);
            }
            let prev = i.checked_sub(1).map(|j| v[j]);
            let mut e = node_entry(rm, prev, v[i]);
            e["floor"] = json!(floor - i64::try_from(newest - i).unwrap_or(0));
            e["act"] = json!(act);
            route.push(e);
        }
        seen.1 = v.len();
    }
    // Relics that arrived without a choice naming them (a chest, a combat
    // reward taken automatically, an event): on a chest passed through just
    // now, else on the node the run was standing on.
    let now = emit_rm::relics(rm);
    let target = (left..route.len())
        .rev()
        .find(|&i| route[i]["room"] == "Treasure")
        .or_else(|| left.checked_sub(1));
    if let Some(entry) = target.and_then(|i| route.get_mut(i)) {
        let mut have = relics.clone();
        for r in &now {
            if let Some(i) = have.iter().position(|h| h == r) {
                have.swap_remove(i);
                continue;
            }
            let name = nm("relic", r);
            if let Some(notes) = entry["notes"].as_array_mut() {
                if !notes
                    .iter()
                    .any(|n| n.as_str().is_some_and(|t| t.contains(&name)))
                {
                    notes.push(json!(format!("got {name}")));
                }
            }
        }
    }
    *relics = now;
}

/// `POST /api/seedcheck/summary {seed, ascension, classes}`: replay an
/// attempt and name its run-level choices (Neow, events, cards, shops,
/// campfires, boss relics), the route floor by floor (`route`: which map node,
/// what was there, what was chosen, HP on leaving) and how it ended.
///
/// # Errors
///
/// A bad seed, or classes that do not fit the run.
#[allow(clippy::too_many_lines)]
pub fn line_summary(seed: &str, ascension: u8, classes: &[usize]) -> Result<Value, String> {
    let mut env = E2eEnv::over_run(new_run_from_game_seed(seed, ascension)?, 0);
    let mut step = env.reset();
    let (mut neow_pick, mut boss_relics, mut events) = (Value::Null, Vec::new(), Vec::new());
    let (mut cards, mut shop, mut rests) = (Vec::new(), Vec::new(), Vec::new());
    let mut route: Vec<Value> = Vec::new();
    let mut seen: (u8, usize) = (u8::MAX, 0);
    let mut relic_ids = emit_rm::relics(env.rm());
    let note = |route: &mut Vec<Value>, text: String| {
        if let Some(last) = route.last_mut() {
            if let Some(n) = last["notes"].as_array_mut() {
                n.push(json!(text));
            }
        }
    };
    for (n, &class) in classes.iter().enumerate() {
        let Step::Decision(d) = step else {
            return Err(format!("the run ended at step {n} of {}", classes.len()));
        };
        let member = *d
            .members
            .get(class)
            .and_then(|m| m.first())
            .ok_or_else(|| format!("step {n}: class {class} out of range"))?;
        sync_route(
            env.rm(),
            &mut route,
            &mut seen,
            &mut relic_ids,
            i64::from(d.floor),
            i64::from(d.act),
        );
        if let Some(last) = route.last_mut() {
            if d.combat.is_some() && last["name"].is_null() {
                if let Some(id) = emit_rm::combat_encounter_id(env.rm()) {
                    last["name"] = json!(nm("encounter", id));
                    last["kind"] = json!("fight");
                }
            }
        }
        if d.combat.is_none() {
            let a = &d.actions[member];
            let rm = env.rm();
            let floor = d.floor;
            match a.get("action").and_then(Value::as_str).unwrap_or_default() {
                "event_choice" => {
                    let i = idx_of(a, "idx").unwrap_or(0);
                    let event = rm
                        .run_state
                        .current_event
                        .map(|id| id.python_name().to_string())
                        .or_else(|| d.event_name.clone())
                        .unwrap_or_default();
                    let option = emit_rm::event_options(rm)
                        .and_then(|o| o.get(i).map(|(_, t)| t.clone()))
                        .unwrap_or_else(|| format!("option {i}"));
                    if event == "Neow" {
                        if let Some(id) = neow_ids(rm).get(i) {
                            neow_pick = json!(nm("relic", id));
                        }
                    } else {
                        let name = nm("event", &event);
                        if let Some(last) = route.last_mut() {
                            if last["name"].is_null() {
                                last["name"] = json!(name);
                                last["kind"] = json!("event");
                            }
                        }
                        note(&mut route, option.clone());
                        events.push(json!({"floor": floor, "event": name, "option": option}));
                    }
                }
                "pick_card_reward" => {
                    if let Some(i) = idx_of(a, "idx") {
                        if let Some(id) = emit_rm::card_reward_option_ids(rm).get(i) {
                            let card = nm("card", id);
                            note(&mut route, format!("took {card}"));
                            cards.push(json!({"floor": floor, "card": card}));
                        }
                    } else {
                        note(&mut route, "skipped the card".to_string());
                    }
                }
                "pick_relic_reward" => {
                    if let Some(r) = rm.run_state.relic_rewards.last() {
                        let relic = nm("relic", &format!("RELIC.{}", r.relic_id.to_python_name()));
                        note(&mut route, format!("took {relic}"));
                    }
                }
                "skip_relic" => note(&mut route, "skipped the relic".to_string()),
                "collect_treasure" => note(&mut route, "opened the chest".to_string()),
                "pick_boss_relic" => {
                    let i = idx_of(a, "idx").unwrap_or(0);
                    if let Some(id) = emit_rm::boss_relic_options(rm).get(i) {
                        let relic = nm("relic", id);
                        note(&mut route, format!("boss relic: {relic}"));
                        boss_relics.push(json!(relic));
                    }
                }
                verb @ ("shop_buy_card" | "shop_buy_relic" | "shop_buy_potion") => {
                    let (list, kind) = match verb {
                        "shop_buy_card" => ("cards", "card"),
                        "shop_buy_relic" => ("relics", "relic"),
                        _ => ("potions", "potion"),
                    };
                    if let Some(id) = idx_of(a, "idx").and_then(|i| shop_item(rm, list, i)) {
                        let item = nm(kind, &id);
                        note(&mut route, format!("bought {item}"));
                        shop.push(json!({"floor": floor, "bought": item}));
                    }
                }
                "shop_remove_card" => {
                    if let Some(id) =
                        idx_of(a, "deck_idx").and_then(|i| emit_rm::master_deck(rm).get(i).cloned())
                    {
                        let card = nm("card", &id);
                        note(&mut route, format!("removed {card}"));
                        shop.push(json!({"floor": floor, "removed": card}));
                    }
                }
                verb @ ("remove_deck_card" | "upgrade_deck_card" | "transform_deck_card") => {
                    if let Some(id) = idx_of(a, "idx")
                        .or_else(|| idx_of(a, "deck_idx"))
                        .and_then(|i| emit_rm::master_deck(rm).get(i).cloned())
                    {
                        let what = match verb {
                            "remove_deck_card" => "removed",
                            "upgrade_deck_card" => "upgraded",
                            _ => "transformed",
                        };
                        note(&mut route, format!("{what} {}", nm("card", &id)));
                    }
                }
                "rest_option" => {
                    let i = idx_of(a, "idx").unwrap_or(0);
                    if let Some((opt, _)) = emit_rm::rest_site_options(rm).get(i) {
                        // A smith is named by the card it upgrades (`smith_card`).
                        match opt.as_str() {
                            "SMITH" => {}
                            "HEAL" => note(&mut route, "rested".to_string()),
                            other => note(&mut route, crate::simlab::title_case(other)),
                        }
                        rests.push(json!({"floor": floor, "choice": opt}));
                    }
                }
                "smith_card" => {
                    if let Some(id) =
                        idx_of(a, "idx").and_then(|i| emit_rm::smith_options(rm).get(i).cloned())
                    {
                        let card = nm("card", &id);
                        note(&mut route, format!("smithed {card}"));
                        if let Some(last) = rests.last_mut() {
                            last["card"] = json!(card);
                        }
                    }
                }
                _ => {}
            }
        }
        step = env.step(class);
    }
    let Step::Terminal(t) = step else {
        return Err(format!(
            "the run did not end after {} classes",
            classes.len()
        ));
    };
    let rm = env.rm();
    let rs = &rm.run_state;
    sync_route(
        rm,
        &mut route,
        &mut seen,
        &mut relic_ids,
        i64::from(t.floor),
        i64::from(t.act),
    );
    if let Some(last) = route.last_mut() {
        last["hp"] = json!(rs.current_hp);
        last["max_hp"] = json!(rs.max_hp);
    }
    let death = (!t.won).then(|| {
        json!({
            "floor": t.floor,
            "act": t.act,
            "room": emit_rm::combat_encounter_id(rm).map(|id| nm("encounter", id)),
        })
    });
    Ok(json!({
        "won": t.won,
        "outcome": t.outcome,
        "hp": rs.current_hp,
        "max_hp": rs.max_hp,
        "floor": t.floor,
        "ascension": ascension,
        "neow_pick": neow_pick,
        "boss_relics": boss_relics,
        "events": events,
        "cards_taken": cards,
        "shop": shop,
        "rests": rests,
        "route": route,
        "deck": emit_rm::master_deck(rm).iter().map(|id| nm("card", id)).collect::<Vec<_>>(),
        "relics": emit_rm::relics(rm).iter().map(|id| nm("relic", id)).collect::<Vec<_>>(),
        "death": death,
    }))
}

/// `POST /api/seedcheck/log`: replay attempt `k` of `seq` (its merged
/// `classes`) as a live e2e run on `sess` — a fresh session with the net
/// loaded — and write it as a normal run log. The session seed is
/// `game:<SEED>`, so the replay viewer rebuilds the same run; the meta record
/// carries `seedcheck: {seq, k, temp, ascension}`.
///
/// # Errors
///
/// A bad seed, no net (424), a class the run refuses, or classes that do not
/// end the run.
pub fn run_log(
    sess: &mut Session,
    seed: &str,
    ascension: u8,
    classes: &[usize],
    seq: Seq,
    k: u32,
    now: &str,
) -> Result<RunLog, SessionError> {
    let canonical =
        sts2_e2e::env::canonical_game_seed(seed).map_err(|e| SessionError::new(400, e))?;
    sess.set_e2e_setup(true);
    sess.start_run_mode_seed(
        crate::e2e::CHARACTER,
        &ModeSeed::Game(canonical),
        ascension,
        "e2e",
        "rust",
        1000,
        4000,
        1,
        None,
    )?;
    let mut used = 0;
    loop {
        let r = sess.e2e_step_with(classes.get(used).copied())?;
        if let Some(e) = r.get("error").and_then(Value::as_str) {
            return Err(SessionError::new(409, format!("replay step {used}: {e}")));
        }
        if r["done"].as_bool().unwrap_or(false) {
            break;
        }
        if r["debug"]["forced"] == Value::Bool(false) {
            if used >= classes.len() {
                // The attempt stopped before the run ended (a decision cap):
                // do not play on greedily past what was recorded.
                return Err(SessionError::new(
                    409,
                    format!(
                        "the run did not end within the {} recorded classes",
                        classes.len()
                    ),
                ));
            }
            used += 1;
        }
    }
    if used != classes.len() {
        return Err(SessionError::new(
            409,
            format!("the run ended after {used} of {} classes", classes.len()),
        ));
    }
    let mut log =
        write_run_log(sess, now).ok_or_else(|| SessionError::new(409, "no run to log"))?;
    let (head, rest) = log.text.split_once('\n').unwrap_or((log.text.as_str(), ""));
    let mut meta: Value =
        serde_json::from_str(head).map_err(|e| SessionError::new(500, e.to_string()))?;
    meta["data"]["seedcheck"] =
        json!({"seq": seq.as_str(), "k": k, "temp": attempt_temp(seq, k), "ascension": ascension});
    log.text = format!("{meta}\n{rest}");
    Ok(log)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `ppo.e2e.solve` proof for the Reddit seed (raw action indices).
    fn proof() -> (String, Vec<usize>) {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sts2-e2e/tests/data/seedcheck_1SNVTCEXKR0Q_proof.json");
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(p).expect("fixture")).expect("json");
        let log = doc["log_idx"]
            .as_array()
            .expect("log_idx")
            .iter()
            .map(|v| usize::try_from(v.as_u64().expect("index")).expect("fits"))
            .collect();
        (doc["seed"].as_str().expect("seed").to_string(), log)
    }

    fn shipped_net() -> Option<crate::e2e::Net> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../sts2-pycli/data/e2e")
            .join(crate::e2e::E2E_FILE);
        std::fs::read(&p)
            .ok()
            .map(|b| crate::e2e::load(&b).expect("the shipped net loads"))
    }

    #[test]
    fn run_log_of_proof_parses_and_wins() {
        let Some(net) = shipped_net() else { return };
        let (seed, log) = proof();
        let classes = classes_from_actions(&seed, 10, &log).unwrap();
        let mut sess = Session::new_for(GameVersion::V0_111_0);
        sess.set_e2e(Some(net.clone()), None);
        let out = run_log(
            &mut sess,
            &seed,
            10,
            &classes,
            Seq::Check,
            0,
            "2026-11-08T12:00:00Z",
        )
        .unwrap();
        assert_eq!(
            std::path::Path::new(&out.filename).extension(),
            Some("jsonl".as_ref())
        );
        let store = crate::runlog::ReplayStore::parse(&out.filename, &out.text).expect("parses");
        let meta: Value = serde_json::from_str(out.text.lines().next().unwrap()).unwrap();
        assert_eq!(meta["data"]["seed"], "game:1SNVTCEXKR0Q");
        assert_eq!(meta["data"]["seedcheck"]["seq"], "check");
        assert!(out.text.contains("\"won\":true") || out.text.contains("\"victory\":true"));
        // The replay viewer rebuilds the run from its identity.
        let ident = store.run_identity().expect("an identity");
        assert_eq!(ident.seed, "game:1SNVTCEXKR0Q");
        assert!(crate::replay::Rebuilt::start(&store).is_ok());
    }

    #[test]
    fn mode_seed_round_trips() {
        for s in ["123", "game:1SNVTCEXKR0Q"] {
            assert_eq!(ModeSeed::parse(s).unwrap().session_string(), s);
        }
        assert!(ModeSeed::parse("game:").is_none());
        assert!(ModeSeed::parse("run_1").is_none());
    }

    #[test]
    fn preview_reddit_seed() {
        let p = preview("1snvtcexkr0q", 10).unwrap();
        assert_eq!(p["seed"], "1SNVTCEXKR0Q");
        assert_eq!(p["length_ok"], true);
        let neow: Vec<&str> = p["neow"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(
            neow.contains(&"Arcane Scroll") && neow.contains(&"Lava Rock"),
            "{neow:?}"
        );
        assert_eq!(p["bosses"][0], json!(["Vantom"]));
        assert_eq!(p["bosses"][2].as_array().unwrap().len(), 2);
        assert_eq!(preview("ABC", 10).unwrap()["length_ok"], false);
        assert!(preview("AB-C", 10).is_err());
        // Below A10 the act 3 boss is single.
        assert_eq!(
            preview("1SNVTCEXKR0Q", 0).unwrap()["bosses"][2]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn summary_of_proof_wins() {
        let (seed, log) = proof();
        let classes = classes_from_actions(&seed, 10, &log).unwrap();
        let s = line_summary(&seed, 10, &classes).unwrap();
        assert_eq!(s["won"], true, "{s}");
        assert!(s["death"].is_null());
        let neow = preview(&seed, 10).unwrap()["neow"].clone();
        assert!(neow.as_array().unwrap().contains(&s["neow_pick"]), "{s}");
        assert_eq!(s["boss_relics"].as_array().unwrap().len(), 2, "{s}");
        assert!(!s["deck"].as_array().unwrap().is_empty());
        // The route: one entry per floor walked, in order, ending on the
        // act 3 bosses with the final HP.
        let route = s["route"].as_array().unwrap();
        let floors: Vec<i64> = route.iter().map(|e| e["floor"].as_i64().unwrap()).collect();
        assert!(floors.windows(2).all(|w| w[0] < w[1]), "{floors:?}");
        assert_eq!(*floors.last().unwrap(), s["floor"].as_i64().unwrap());
        for e in route {
            assert!(e["pos"].as_u64().unwrap() >= 1, "{e}");
            assert!(e["pos"].as_u64() <= e["of"].as_u64(), "{e}");
            assert!(e["room"].as_str().is_some_and(|r| !r.is_empty()), "{e}");
            assert!(e["hp"].is_number(), "{e}");
        }
        let combats = route
            .iter()
            .filter(|e| e["room"] == "Monster" || e["room"] == "Elite")
            .count();
        assert!(combats > 5, "{route:?}");
        assert!(
            route
                .iter()
                .filter(|e| e["room"] == "Monster")
                .all(|e| e["name"].is_string()),
            "{route:?}"
        );
        assert_eq!(route.last().unwrap()["hp"], s["hp"]);
        let first = &route[0];
        assert!(first["dir"].is_null(), "{first}");
    }
}
