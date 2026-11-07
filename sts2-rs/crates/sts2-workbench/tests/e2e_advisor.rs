//! The e2e student as the ADVISOR of a hand-played run (`advisor: "e2e"`):
//! the human makes every move through `/api/live/action` (no e2e step ever
//! runs), and at every position `POST /api/live/e2e-advice` — and the hint /
//! policy panel / auto-action built on it — equals what `sts2_e2e`'s own
//! driver + net give on the same position. The reference is an independent
//! `Driver` loop over a clone of the run's start (`E2eEnv`'s loop), which
//! applies the same moves; the session only sees the human's verbs.
#![cfg(feature = "e2e")]

use std::collections::BTreeMap;

use serde_json::{json, Value};
use sts2_core::run_manager::RunManager;
use sts2_e2e::env::{Driver, Next};
use sts2_e2e::tokens::TokCache;
use sts2_workbench::api::{Config, Workbench};
use sts2_workbench::{e2e, rmx, runmode};

fn new_wb() -> Workbench {
    Workbench::new(Config {
        assets_base: "https://assets.test".to_string(),
        ..Config::default()
    })
}

fn weights() -> Option<Vec<u8>> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../sts2-pycli/data/e2e")
        .join(Workbench::e2e_file());
    std::fs::read(p).ok()
}

fn call(wb: &mut Workbench, method: &str, route: &str, body: Value) -> Value {
    match wb.call(method, route, &body) {
        Ok(v) => v,
        Err(e) => panic!("{method} {route} -> {} {}", e.status, e.detail),
    }
}

fn start(wb: &mut Workbench, seed: &str) {
    call(
        wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": seed, "ascension": 0,
               "act1": "overgrowth", "game_version": "v0.111.0"}),
    );
}

/// The workbench verb a UI sends for env run-action `a` at `rm`.
fn verb_of(rm: &RunManager, a: &Value) -> (String, Value) {
    let acts = rm.get_available_actions();
    match rmx::action_from_value(a) {
        Ok(ra) => runmode::resolve_workbench_verb(rm, &ra, &acts)
            .unwrap_or_else(|| runmode::to_workbench_verb(rm, a)),
        Err(_) => runmode::to_workbench_verb(rm, a),
    }
}

fn active_id(wb: &Workbench) -> u64 {
    wb.session().unwrap().active_node().unwrap().id
}

/// The advice's probability per class.
fn probs_by_class(adv: &Value) -> BTreeMap<u64, f64> {
    adv["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| (o["class"].as_u64().unwrap(), o["prob"].as_f64().unwrap()))
        .collect()
}

struct Played {
    /// Advice per node, as first seen (`probs_by_class`).
    seen: BTreeMap<u64, BTreeMap<u64, f64>>,
    combat_decisions: usize,
    card_rewards: usize,
    graded_e2e: usize,
    actions: usize,
}

/// Play `n` human moves from the session's live position, mirrored on the
/// reference driver `(rm, drv, cache)`. While `advise` is on, every position's
/// advice is checked against the reference net. The human takes the
/// student's pick except at some decisions, where it takes the next class
/// (so the run is the human's, not the student's).
#[allow(clippy::too_many_arguments)]
fn play(
    wb: &mut Workbench,
    net: &e2e::Net,
    rm: &mut RunManager,
    drv: &mut Driver,
    cache: &mut TokCache,
    n: usize,
    advise: bool,
    out: &mut Played,
) {
    let mut decisions = 0usize;
    for _ in 0..n {
        let next = drv.peek(rm);
        let node = active_id(wb);
        let (action, kind) = match next {
            Next::Terminal(t) => panic!("the run ended early: {t:?}"),
            Next::Forced(f) => {
                if advise {
                    let adv = call(wb, "POST", "/api/live/e2e-advice", json!({}));
                    assert_eq!(adv["available"], json!(true), "{adv}");
                    assert_eq!(adv["forced"], json!(true));
                    let (verb, args) = verb_of(rm, &f.action);
                    assert_eq!(adv["pick"]["action"], json!(verb));
                    assert_eq!(adv["pick"]["args"], args);
                }
                (f.action, f.kind)
            }
            Next::Decision(d) => {
                let c = e2e::decide(net, &d, rm, cache, e2e::fallback_ms).expect("decide");
                // Off combat every other decision, in combat every sixth: a
                // human who disagrees but still wins the first fights.
                let deviate = if d.is_combat() {
                    decisions % 6 == 3
                } else {
                    decisions % 2 == 1
                };
                let human = if deviate {
                    (c.class + 1) % d.n_classes()
                } else {
                    c.class
                };
                decisions += 1;
                if d.is_combat() {
                    out.combat_decisions += 1;
                }
                if wb.session().unwrap().active_node().unwrap().decision == "card_reward" {
                    out.card_rewards += 1;
                }
                if advise {
                    let adv = call(wb, "POST", "/api/live/e2e-advice", json!({}));
                    assert_eq!(adv["available"], json!(true), "{adv}");
                    assert_eq!(adv["node_id"], json!(node));
                    let got = probs_by_class(&adv);
                    assert_eq!(got.len(), d.n_classes(), "node {node}");
                    for (i, p) in c.probs.iter().enumerate() {
                        let g = got[&(i as u64)];
                        assert!((g - p).abs() < 1e-9, "node {node} class {i}: {g} vs {p}");
                    }
                    let pick = adv["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|o| o["is_best"] == json!(true))
                        .unwrap();
                    assert_eq!(pick["class"], json!(c.class));
                    out.seen.insert(node, got);
                    // The surfaces built on the same evaluation.
                    let decision = wb
                        .session()
                        .unwrap()
                        .active_node()
                        .unwrap()
                        .decision
                        .clone();
                    if decision == "combat_play" {
                        let hint = call(wb, "POST", "/api/live/hint", json!({"config": {}}));
                        assert_eq!(hint["advisor"], json!("e2e"));
                        assert_eq!(hint["scores"], adv["scores"]);
                        let best: Vec<&Value> = hint["scores"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|s| s["is_best"] == json!(true))
                            .collect();
                        assert_eq!(best.len(), 1);
                    } else if decision != "game_over" {
                        let panel = call(wb, "POST", "/api/live/ppo-policy", json!({}));
                        assert_eq!(panel["policy"], json!("e2e"));
                        assert_eq!(panel["options"].as_array().unwrap().len(), d.n_classes());
                    }
                    let rec = call(wb, "POST", "/api/live/auto-action", json!({}));
                    assert_eq!(rec["action"], pick["action"]);
                    assert_eq!(rec["args"], pick["args"]);
                }
                d.class_action(human).expect("class")
            }
        };
        // The human's move, as the UI sends it.
        let (verb, args) = verb_of(rm, &action);
        if std::env::var("E2E_ADV_TRACE").is_ok() {
            eprintln!("{} {action} -> {verb} {args}", out.actions);
        }
        drv.apply(rm, &action, kind).expect("reference apply");
        let r = call(
            wb,
            "POST",
            "/api/live/action",
            json!({"action": verb, "args": args}),
        );
        assert!(r["error"].is_null(), "{verb} {args}: {}", r["error"]);
        out.actions += 1;
        if advise && !matches!(kind, sts2_e2e::env::ApplyKind::Combat) {
            if let Ok(g) = wb.call("POST", "/api/live/last-choice-grade", &json!({})) {
                if g["policy"] == json!("e2e") {
                    out.graded_e2e += 1;
                }
            }
        }
    }
}

#[test]
fn the_e2e_advisor_follows_a_hand_played_run() {
    let Some(bytes) = weights() else {
        eprintln!("skipped: no shipped e2e weights");
        return;
    };
    let net = e2e::load(&bytes).expect("load");
    let mut wb = new_wb();
    wb.load_e2e(&bytes).expect("load_e2e");
    start(&mut wb, "E2EADVISOR");
    let sess = wb.session().unwrap();
    let mut rm = sess.game().unwrap().native_run().unwrap().clone();
    let mut drv = Driver::new(i64::from(runmode::hash_seed(&sess.seed)));
    let mut cache = TokCache::new();
    let mut out = Played {
        seen: BTreeMap::new(),
        combat_decisions: 0,
        card_rewards: 0,
        graded_e2e: 0,
        actions: 0,
    };

    // The human plays on alone first (search advisor): no student state is
    // kept, so turning the advisor on rebuilds it by shadow replay.
    play(
        &mut wb, &net, &mut rm, &mut drv, &mut cache, 12, false, &mut out,
    );
    let cfg = call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"advisor": "e2e"}),
    );
    assert_eq!(cfg["advisor"], json!("e2e"));
    play(
        &mut wb, &net, &mut rm, &mut drv, &mut cache, 70, true, &mut out,
    );
    assert!(
        out.combat_decisions > 5,
        "a fight: {}",
        out.combat_decisions
    );
    assert!(out.card_rewards > 0, "a card reward");
    assert!(
        out.graded_e2e > 0,
        "the choice coach graded against the student"
    );

    // Seek back into the run, then forward to the leaf: the cached / rebuilt
    // evaluations are the ones first seen.
    let leaf = active_id(&wb);
    let path: Vec<u64> = wb
        .session()
        .unwrap()
        .active_path()
        .unwrap()
        .iter()
        .map(|n| n.id)
        .collect();
    let back = *path
        .iter()
        .rev()
        .skip(25)
        .find(|id| out.seen.contains_key(id))
        .expect("an advised node 25 back");
    call(&mut wb, "POST", "/api/live/seek", json!({"node_id": back}));
    let adv = call(&mut wb, "POST", "/api/live/e2e-advice", json!({}));
    assert_eq!(probs_by_class(&adv), out.seen[&back]);
    call(&mut wb, "POST", "/api/live/seek", json!({"node_id": leaf}));

    // A reload knows nothing of the student's states: the advice at the
    // leaf is rebuilt from the run's start and still matches the reference.
    let save = wb.build_save("adv", "2026-11-06T00:00:00Z").expect("save");
    let mut fresh = new_wb();
    fresh.load_e2e(&bytes).unwrap();
    fresh
        .restore_save(&save, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .expect("restore");
    call(
        &mut fresh,
        "POST",
        "/api/live/run-mode",
        json!({"advisor": "e2e"}),
    );
    let mut seen_after = Played {
        seen: BTreeMap::new(),
        combat_decisions: 0,
        card_rewards: 0,
        graded_e2e: 0,
        actions: 0,
    };
    play(
        &mut fresh,
        &net,
        &mut rm,
        &mut drv,
        &mut cache,
        10,
        true,
        &mut seen_after,
    );
}

#[test]
fn the_e2e_advisor_is_ironclad_v0111_only_and_asks_for_its_net() {
    let mut wb = new_wb();
    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Silent", "seed": "X", "game_version": "v0.111.0"}),
    );
    let err = wb
        .call("POST", "/api/live/run-mode", &json!({"advisor": "e2e"}))
        .expect_err("silent");
    assert_eq!(err.status, 400);
    assert!(err.detail.contains("Ironclad"), "{}", err.detail);

    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "X", "game_version": "v0.107.1"}),
    );
    let err = wb
        .call("POST", "/api/live/run-mode", &json!({"advisor": "e2e"}))
        .expect_err("v0.107.1");
    assert_eq!(err.status, 400);
    assert!(err.detail.contains("v0.111.0"), "{}", err.detail);
    let err = wb
        .call("POST", "/api/live/run-mode", &json!({"advisor": "nope"}))
        .expect_err("bad name");
    assert_eq!(err.status, 400);

    start(&mut wb, "X");
    let cfg = call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"advisor": "e2e"}),
    );
    assert_eq!(cfg["advisor"], json!("e2e"));
    // No net yet: every advised surface answers 424 (the host fetches and
    // retries), nothing falls back to the search silently.
    for route in [
        "/api/live/e2e-advice",
        "/api/live/auto-action",
        "/api/live/ppo-policy",
    ] {
        let err = wb.call("POST", route, &json!({})).expect_err(route);
        assert_eq!(err.status, 424, "{route}");
    }
    // Back to search: the run-mode reply drops the key (default bytes).
    let cfg = call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"advisor": "search"}),
    );
    assert!(cfg.get("advisor").is_none());
}

/// Play to end (`run-mode-step`) on a hand-played run plays on as its
/// advisor: under the e2e advisor every step is the student's, fights
/// included, never the combat MCTS; back on the search it is the search's.
#[test]
fn play_to_end_under_the_e2e_advisor_plays_the_student() {
    let Some(bytes) = weights() else {
        eprintln!("skipped: no shipped e2e weights");
        return;
    };
    let mut wb = new_wb();
    wb.load_e2e(&bytes).expect("load_e2e");
    call(&mut wb, "POST", "/api/advisor", json!({"advisor": "e2e"}));
    start(&mut wb, "E2EPLAYON");
    let mut combat = 0;
    for i in 0..80 {
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        assert!(out["error"].is_null(), "step {i}: {}", out["error"]);
        assert_eq!(
            out["debug"]["brain"],
            json!("e2e"),
            "step {i}: {}",
            out["debug"]
        );
        if out["debug"]["is_combat"] == json!(true) {
            combat += 1;
        }
    }
    assert!(combat > 10, "80 steps from the start include a fight");
    call(
        &mut wb,
        "POST",
        "/api/advisor",
        json!({"advisor": "search"}),
    );
    let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
    assert_ne!(out["debug"]["brain"], json!("e2e"));
}

/// A watch run keeps the brain it was started with, whatever the advisor
/// preference says.
#[test]
fn a_watch_run_keeps_its_brain_under_the_e2e_advisor() {
    let Some(bytes) = weights() else {
        return;
    };
    let mut wb = new_wb();
    wb.load_e2e(&bytes).expect("load_e2e");
    call(&mut wb, "POST", "/api/advisor", json!({"advisor": "e2e"}));
    call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"character": "Ironclad", "seed": "E2EWATCH", "ascension": 0,
               "game_version": "v0.111.0", "sims": 20, "boss_sims": 20}),
    );
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "heuristic", "combat_sims": 20, "combat_boss_sims": 20}),
    );
    assert_eq!(wb.session().unwrap().advisor, "e2e");
    let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
    assert_ne!(out["debug"]["brain"], json!("e2e"));
}

#[test]
fn the_hosts_advisor_preference_follows_every_run_it_can_advise() {
    let mut wb = new_wb();
    let out = call(&mut wb, "POST", "/api/advisor", json!({"advisor": "e2e"}));
    assert_eq!(out, json!({"advisor": "e2e", "live": null}));
    start(&mut wb, "PREF");
    assert_eq!(wb.session().unwrap().advisor, "e2e");
    // A run the student cannot read keeps the search.
    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Defect", "seed": "PREF", "game_version": "v0.111.0"}),
    );
    assert_eq!(wb.session().unwrap().advisor, "search");
    start(&mut wb, "PREF2");
    let out = call(
        &mut wb,
        "POST",
        "/api/advisor",
        json!({"advisor": "search"}),
    );
    assert_eq!(out, json!({"advisor": "search", "live": "search"}));
    assert_eq!(wb.session().unwrap().advisor, "search");
    let err = wb
        .call("POST", "/api/advisor", &json!({"advisor": "mcts"}))
        .expect_err("unknown");
    assert_eq!(err.status, 400);
}
