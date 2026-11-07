//! The `e2e` brain over `Workbench::call`: the net is host-loaded on demand
//! (424 until then), `start-faithful {e2e: true}` forces the student's run,
//! every run-mode step is the student's, `POST /api/e2e/selfplay` matches a
//! native `E2eEnv` run, and a save restores into a session that keeps going.
#![cfg(feature = "e2e")]

use serde_json::{json, Value};
use sts2_core::GameVersion;
use sts2_e2e::env::{E2eEnv, Step};
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::{encode, TokCache};
use sts2_workbench::api::{Config, Workbench};

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

#[test]
fn without_weights_the_brain_answers_424_and_capabilities_say_e2e() {
    let mut wb = new_wb();
    let caps = call(&mut wb, "GET", "/api/capabilities", Value::Null);
    assert_eq!(caps["features"]["e2e"], json!(true));
    assert!(!wb.has_e2e());
    let err = wb
        .call("POST", "/api/e2e/selfplay", &json!({"seed": 1}))
        .expect_err("no net");
    assert_eq!(err.status, 424);
    assert!(wb.load_e2e(b"garbage").is_err());

    // The run starts without the net (it is the student's run, forced
    // Ironclad v0.111.0 A10); the first step asks for the weights.
    let start = call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"e2e": true, "character": "Silent", "seed": "run_3", "game_version": "v0.107.1"}),
    );
    assert!(start["error"].is_null());
    let rm = wb
        .session()
        .and_then(|s| s.game())
        .and_then(|g| g.native_run())
        .unwrap();
    assert_eq!(rm.game_version, GameVersion::V0_111_0);
    assert_eq!(rm.ascension_level, 10);
    let err = wb
        .call("POST", "/api/live/run-mode-step", &json!({}))
        .expect_err("no net yet");
    assert_eq!(err.status, 424);
    wb.set_e2e_error("fetch failed: 404");
    let err = wb
        .call("POST", "/api/live/run-mode-step", &json!({}))
        .expect_err("still no net");
    assert!(err.detail.contains("fetch failed"), "{}", err.detail);
}

#[test]
fn run_mode_validates_the_brain_name() {
    let mut wb = new_wb();
    call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"seed": "run_1"}),
    );
    let cfg = call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "e2e"}),
    );
    assert_eq!(cfg["off_combat_brain"], json!("e2e"));
    let err = wb
        .call(
            "POST",
            "/api/live/run-mode",
            &json!({"off_combat_brain": "nope"}),
        )
        .expect_err("bad brain");
    assert_eq!(err.status, 400);
    assert!(err.detail.contains("e2e"), "{}", err.detail);
}

#[test]
fn selfplay_matches_the_native_env() {
    let Some(bytes) = weights() else {
        eprintln!("skipped: no shipped e2e weights");
        return;
    };
    let mut wb = new_wb();
    let loaded = wb.load_e2e(&bytes).expect("load");
    assert_eq!(loaded["header"]["game_version"], json!("v0.111.0"));
    let out = call(
        &mut wb,
        "POST",
        "/api/e2e/selfplay",
        json!({"seed": 170_000_002}),
    );

    let net = E2eNet::from_bytes(&bytes).unwrap();
    let mut env = E2eEnv::new(170_000_002);
    let mut cache = TokCache::new();
    let mut chosen = Vec::new();
    let mut s = env.reset();
    let won = loop {
        match s {
            Step::Terminal(t) => break t.won,
            Step::Decision(d) => {
                let c = net.act_greedy(&encode(&d, env.rm(), &mut cache).unwrap().into());
                chosen.push(c);
                s = env.step(c);
            }
        }
    };
    assert_eq!(out["chosen"], json!(chosen));
    assert_eq!(out["won"], json!(won));
    assert_eq!(out["decisions"], json!(chosen.len()));
    assert!(out["ms_per_decision"].as_f64().unwrap() > 0.0);

    let capped = call(
        &mut wb,
        "POST",
        "/api/e2e/selfplay",
        json!({"seed": 170_000_002, "max_decisions": 10}),
    );
    assert_eq!(capped["decisions"], json!(10));
    assert_eq!(capped["outcome"], json!("stopped"));
}

#[test]
fn an_e2e_run_saves_and_restores() {
    let Some(bytes) = weights() else {
        return;
    };
    let mut wb = new_wb();
    wb.load_e2e(&bytes).unwrap();
    call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"e2e": true, "seed": "170000000", "ascension": 10}),
    );
    let mut fights = 0;
    for _ in 0..60 {
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        assert!(out["error"].is_null(), "{}", out["error"]);
        assert_eq!(out["debug"]["brain"], json!("e2e"));
        if out["debug"]["is_combat"] == json!(true) {
            fights += 1;
        }
    }
    assert!(fights > 0);
    let payload = wb.build_save("e2e", "2026-11-06T00:00:00Z").expect("save");
    assert_eq!(payload["e2e_setup"], json!(true));
    let before = call(&mut wb, "GET", "/api/live/state", Value::Null);
    let next = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));

    let mut fresh = new_wb();
    fresh.load_e2e(&bytes).unwrap();
    let restored = fresh
        .restore_save(&payload, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .expect("restore");
    assert_eq!(restored["decision"], before["decision"]);
    let again = call(&mut fresh, "POST", "/api/live/run-mode-step", json!({}));
    assert_eq!(again["debug"]["action"], next["debug"]["action"]);
    assert_eq!(again["search"]["scores"], next["search"]["scores"]);
}
