//! GH #801: the workbench's game-version selector over `Workbench::call`.
//!
//! A run can be started (interactive, faithful, imported) in any supported
//! build; the build travels with the run through every rebuild path (seek,
//! save -> restore, persist -> import, run log -> "play from here"), and the
//! default build's payloads never name it.

use serde_json::{json, Value};
use sts2_core::GameVersion;
use sts2_workbench::api::{Config, Workbench};

fn new_wb() -> Workbench {
    Workbench::new(Config {
        assets_base: "https://assets.test".to_string(),
        ..Config::default()
    })
}

fn call(wb: &mut Workbench, method: &str, route: &str, body: Value) -> Value {
    match wb.call(method, route, &body) {
        Ok(v) => v,
        Err(e) => panic!("{method} {route} -> {} {}", e.status, e.detail),
    }
}

/// The live run's engine-side build and hashed master seed.
fn live_run(wb: &Workbench) -> (GameVersion, u64) {
    let rm = wb
        .session()
        .and_then(|s| s.game())
        .and_then(|g| g.native_run())
        .expect("a live run");
    assert_eq!(rm.game_version, rm.run_state.game_version);
    (rm.game_version, rm.run_state.rng.seed)
}

/// Take the first enabled option `steps` times (Neow, then the map...).
fn advance(wb: &mut Workbench, mut step: Value, steps: usize) -> Value {
    for _ in 0..steps {
        let state = &step["state"];
        let (action, args) = if step["decision"] == "card_select" {
            ("select_cards".to_string(), json!({"indices": "0"}))
        } else {
            let opts = state["options"].as_array().cloned().unwrap_or_default();
            match opts.iter().find(|o| o["enabled"].as_bool().unwrap_or(true)) {
                Some(o) => (o["action"].as_str().unwrap().to_string(), o["args"].clone()),
                None => ("proceed".to_string(), json!({})),
            }
        };
        let resp = call(
            wb,
            "POST",
            "/api/live/action",
            json!({"action": action, "args": args}),
        );
        assert!(
            resp["error"].is_null(),
            "{action} rejected: {}",
            resp["error"]
        );
        step = resp["step"].clone();
    }
    step
}

#[test]
fn capabilities_list_the_supported_builds() {
    let mut wb = new_wb();
    let caps = call(&mut wb, "GET", "/api/capabilities", Value::Null);
    // A public snapshot has no non-default bundles, so only the default is
    // offered there (#855); each build with a bundle is listed, oldest first.
    #[allow(unused_mut)]
    let mut want = vec!["v0.107.1"];
    #[cfg(workbench_game_data_v0_111_0)]
    want.push("v0.111.0");
    #[cfg(workbench_game_data_v0_113_0)]
    want.push("v0.113.0");
    assert_eq!(caps["game_versions"], json!(want));
    // A new run starts on the new-run default when its bundle is embedded.
    let default = if cfg!(workbench_game_data_v0_111_0) {
        "v0.111.0"
    } else {
        "v0.107.1"
    };
    assert_eq!(caps["default_game_version"], json!(default));
}

#[test]
fn an_unknown_build_is_a_400_naming_the_supported_ones() {
    let mut wb = new_wb();
    for (route, body) in [
        (
            "/api/live/start",
            json!({"seed": "TEST", "game_version": "v0.999.0"}),
        ),
        (
            "/api/live/start-faithful",
            json!({"seed": "run_1", "game_version": "0.111.0"}),
        ),
        (
            "/api/live/import-state",
            json!({"payload": "{}", "game_version": 111}),
        ),
    ] {
        let e = wb.call("POST", route, &body).unwrap_err();
        assert_eq!(e.status, 400, "{route}: {}", e.detail);
        assert!(
            e.detail.contains("v0.107.1") && e.detail.contains("v0.111.0"),
            "{route}: {}",
            e.detail
        );
    }
    // Nothing was started.
    assert!(wb.session().is_none());
}

#[test]
fn a_v0111_run_hashes_its_seed_with_that_build() {
    let mut wb = new_wb();
    let old = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "TEST", "act1": "overgrowth", "game_version": "v0.107.1"}),
    );
    assert!(
        old["state"].get("game_version").is_none(),
        "a legacy-build state is unnamed"
    );
    let (v_old, seed_old) = live_run(&wb);
    assert_eq!(v_old, GameVersion::V0_107_1);
    assert_eq!(seed_old, GameVersion::V0_107_1.run_seed_hash("TEST"));

    let explicit = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "TEST", "act1": "overgrowth",
               "game_version": "v0.107.1"}),
    );
    assert_eq!(explicit, old, "an explicit default is the default");

    let new = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "TEST", "act1": "overgrowth",
               "game_version": "v0.111.0"}),
    );
    assert_eq!(new["state"]["game_version"], json!("v0.111.0"));
    assert_eq!(new["state"]["event_name"], json!("Neow"));
    let (v_new, seed_new) = live_run(&wb);
    assert_eq!(v_new, GameVersion::V0_111_0);
    assert_eq!(seed_new, GameVersion::V0_111_0.run_seed_hash("TEST"));
    assert_ne!(seed_new, seed_old, "XxHash64 is not the 32-bit hash");
    assert_eq!(wb.session().unwrap().game_version, GameVersion::V0_111_0);

    // Ascension routes through the other factory; still the build's seed.
    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Silent", "seed": "TEST", "ascension": 5,
               "game_version": "v0.111.0"}),
    );
    assert_eq!(
        live_run(&wb),
        (
            GameVersion::V0_111_0,
            GameVersion::V0_111_0.run_seed_hash("TEST")
        )
    );
}

#[test]
fn a_faithful_run_carries_the_build() {
    let mut wb = new_wb();
    let step = call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"character": "Ironclad", "seed": "run_3", "sims": 10, "boss_sims": 10,
               "game_version": "v0.111.0"}),
    );
    assert_eq!(step["state"]["game_version"], json!("v0.111.0"));
    // An integer seed is a `uint` master in both builds.
    assert_eq!(live_run(&wb), (GameVersion::V0_111_0, 3));
    let persist = wb.export_persist();
    assert_eq!(persist["game_version"], json!("v0.111.0"));
    let mut wb2 = new_wb();
    wb2.import_persist(&persist, |_, _| {}).expect("reload");
    assert_eq!(live_run(&wb2), (GameVersion::V0_111_0, 3));
}

#[test]
fn save_persist_and_run_log_keep_a_v0111_run() {
    let mut wb = new_wb();
    let step = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "TEST", "act1": "overgrowth",
               "game_version": "v0.111.0"}),
    );
    let step = advance(&mut wb, step, 3);
    let seed = live_run(&wb).1;

    // Seek to the root and back rebuilds in the same build.
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let leaf = tl["active_leaf"].clone();
    call(&mut wb, "POST", "/api/live/seek", json!({"node_id": 0}));
    assert_eq!(live_run(&wb), (GameVersion::V0_111_0, seed));
    let back = call(&mut wb, "POST", "/api/live/seek", json!({"node_id": leaf}));
    assert_eq!(back["step"]["state"], step["state"]);

    // Save -> restore.
    let save = wb.build_save("v111", "2026-10-20T00:00:00Z").expect("save");
    assert_eq!(save["game_version"], json!("v0.111.0"));
    let mut wb2 = new_wb();
    let restored = wb2
        .restore_save(&save, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .expect("restore");
    assert_eq!(live_run(&wb2), (GameVersion::V0_111_0, seed));
    assert_eq!(restored["state"], step["state"]);

    // Persist -> import.
    let persist = wb.export_persist();
    assert_eq!(persist["game_version"], json!("v0.111.0"));
    let mut wb3 = new_wb();
    let imported = wb3.import_persist(&persist, |_, _| {}).expect("reload");
    assert_eq!(live_run(&wb3), (GameVersion::V0_111_0, seed));
    assert_eq!(imported["state"], step["state"]);

    // Run log meta -> "play from here".
    call(&mut wb, "POST", "/api/live/quit", Value::Null);
    let log = wb.take_run_log().expect("log");
    let meta: Value = serde_json::from_str(log.text.lines().next().unwrap()).unwrap();
    assert_eq!(meta["data"]["game_version"], json!("v0.111.0"));
    let mut wb4 = new_wb();
    wb4.load_run(&log.filename, &log.text).expect("load");
    call(&mut wb4, "POST", "/api/live/from_replay", json!({"idx": 2}));
    assert_eq!(live_run(&wb4), (GameVersion::V0_111_0, seed));

    // An unknown build in a stored payload is refused, not replayed as 0.107.
    let mut bad = save.clone();
    bad["game_version"] = json!("v9");
    assert!(new_wb()
        .restore_save(&bad, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .is_err_and(|e| e.status == 409));
    let mut bad = persist.clone();
    bad["game_version"] = json!("v9");
    assert!(new_wb()
        .import_persist(&bad, |_, _| {})
        .is_err_and(|e| e.status == 409));
}

#[test]
fn legacy_build_payloads_name_it_too() {
    let mut wb = new_wb();
    let step = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "TEST", "act1": "overgrowth", "game_version": "v0.107.1"}),
    );
    advance(&mut wb, step, 2);
    let mut save = wb.build_save("v107", "t").expect("save");
    assert_eq!(save["game_version"], json!("v0.107.1"));
    let persist = wb.export_persist();
    assert_eq!(persist["game_version"], json!("v0.107.1"));
    // A save from before the key was always written names none, and
    // restores as v0.107.1.
    save.as_object_mut()
        .expect("save object")
        .remove("game_version");
    let mut wb2 = new_wb();
    wb2.restore_save(&save, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .expect("restore");
    assert_eq!(live_run(&wb2).0, GameVersion::V0_107_1);
    call(&mut wb, "POST", "/api/live/quit", Value::Null);
    let log = wb.take_run_log().expect("log");
    assert!(log.text.contains(r#""game_version":"v0.107.1""#));
}

#[test]
fn an_imported_run_is_built_in_the_requested_build() {
    let paste = json!({
        "type": "advice_request",
        "advice_for": "shop",
        "seed": "SMOKESEED",
        "act": 1,
        "floor": 4,
        "run_state": {
            "character": "IRONCLAD",
            "ascension": 0,
            "hp": 44,
            "max_hp": 80,
            "gold": 150,
            "potion_slot_count": 3,
            "master_deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
    })
    .to_string();
    let mut wb = new_wb();
    let step = call(
        &mut wb,
        "POST",
        "/api/live/import-state",
        json!({"payload": paste, "game_version": "v0.111.0"}),
    );
    assert_eq!(step["state"]["game_version"], json!("v0.111.0"));
    let want = GameVersion::V0_111_0.run_seed_hash(&sts2_core::canonicalize_seed("SMOKESEED"));
    assert_eq!(live_run(&wb), (GameVersion::V0_111_0, want));
    let persist = wb.export_persist();
    assert_eq!(persist["game_version"], json!("v0.111.0"));
    let mut wb2 = new_wb();
    let restored = wb2.import_persist(&persist, |_, _| {}).expect("reload");
    assert_eq!(restored["state"]["player"]["hp"], json!(44));
    assert_eq!(live_run(&wb2), (GameVersion::V0_111_0, want));
}

/// The combat lane of a v0.111.0 import rebuilds the fight in v0.111.0: the
/// bridge's combat wire never names a build, so without the run's build the
/// wire parser fell back to v0.107.1 and refused the run's 64-bit master
/// seed (or ran the fight under 0.107 rules on a 0.111 run).
#[test]
fn an_imported_fight_is_rebuilt_in_the_requested_build() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/state_import_combat.json"
    );
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("skipping: {path} not present");
        return;
    };
    let doc: Value = serde_json::from_str(&text).expect("golden parses");
    let case = doc["cases"]
        .as_array()
        .and_then(|c| c.iter().find(|c| c["name"] == "single_enemy_ironclad"))
        .expect("the single-enemy capture");
    let mut paste: Value =
        serde_json::from_str(case["paste"].as_str().expect("paste")).expect("paste parses");
    let seed = paste["seed"]
        .as_str()
        .map_or_else(|| paste["seed"].to_string(), str::to_string);
    let master = GameVersion::V0_111_0.run_seed_hash(&sts2_core::canonicalize_seed(&seed));
    assert!(master > u64::from(u32::MAX), "the test needs a 64-bit seed");
    paste["rng"] = json!({"run_seed": master, "counters": {"shuffle": 1, "monster_ai": 1}});

    let mut wb = new_wb();
    let step = call(
        &mut wb,
        "POST",
        "/api/live/import-state",
        json!({"payload": paste.to_string(), "lane": "auto", "game_version": "v0.111.0"}),
    );
    assert_eq!(step["import"]["lane"], json!("combat"));
    assert_eq!(step["decision"], json!("combat_play"), "{step}");
    assert_eq!(step["state"]["game_version"], json!("v0.111.0"));
    assert_eq!(live_run(&wb), (GameVersion::V0_111_0, master));
    let cs = wb
        .session()
        .and_then(|s| s.game())
        .and_then(|g| g.combat_state())
        .expect("a live fight");
    assert_eq!(cs.game_version, GameVersion::V0_111_0);

    // Persist -> import rebuilds the same fight in the same build.
    let persist = wb.export_persist();
    let mut wb2 = new_wb();
    wb2.import_persist(&persist, |_, _| {}).expect("reload");
    let cs2 = wb2
        .session()
        .and_then(|s| s.game())
        .and_then(|g| g.combat_state())
        .expect("a live fight after reload");
    assert_eq!(cs2.game_version, GameVersion::V0_111_0);
}

/// GH #877: the deck analyzer grades with the run's own build's tables —
/// a card new in v0.111.0 is a real attack there and unknown to v0.107.1 —
/// and an unknown build is a 400.
#[cfg(workbench_game_data_v0_111_0)]
#[test]
fn reward_preview_grades_with_the_requested_builds_tables() {
    let mut wb = new_wb();
    let mut deck = vec![json!({"id": "STRIKE_IRONCLAD"}); 5];
    deck.extend(vec![json!({"id": "DEFEND_IRONCLAD"}); 4]);
    let body = json!({"deck": deck, "candidates": [{"index": 0, "id": "OUTRAGE"}]});
    let mut v107_body = body.clone();
    v107_body["game_version"] = json!("v0.107.1");
    let base = call(&mut wb, "POST", "/api/deck/reward-preview", v107_body);
    let mut v111_body = body.clone();
    v111_body["game_version"] = json!("v0.111.0");
    let v111 = call(&mut wb, "POST", "/api/deck/reward-preview", v111_body);
    assert_eq!(
        base["base"], v111["base"],
        "the starter cards did not change"
    );
    assert_ne!(
        base["candidates"][0]["delta"], v111["candidates"][0]["delta"],
        "OUTRAGE is only a card in v0.111.0"
    );
    let mut bad = body;
    bad["game_version"] = json!("v9");
    let err = wb
        .call("POST", "/api/deck/reward-preview", &bad)
        .expect_err("an unknown build");
    assert_eq!(err.status, 400);
}

/// GH #877: SimLab deals a fight in the requested build and names its
/// catalog in that build's text.
#[test]
fn simlab_deals_in_the_requested_build() {
    let mut wb = new_wb();
    let body = json!({"character": "Ironclad", "encounter": "FUZZY_WURM_CRAWLER_WEAK", "seed": 4});
    let mut v107_body = body.clone();
    v107_body["game_version"] = json!("v0.107.1");
    let base = call(&mut wb, "POST", "/api/simlab/position/new", v107_body);
    assert!(base["position"].get("game_version").is_none());
    let mut v111_body = body;
    v111_body["game_version"] = json!("v0.111.0");
    let v111 = call(&mut wb, "POST", "/api/simlab/position/new", v111_body);
    assert_eq!(v111["position"]["game_version"], json!("v0.111.0"));
    let moves = call(
        &mut wb,
        "POST",
        "/api/simlab/position/moves",
        json!({"position": v111["position"]}),
    );
    assert!(moves["moves"].as_array().is_some_and(|m| !m.is_empty()));
    let err = wb
        .call("GET", "/api/simlab/catalog?game_version=v9", &json!({}))
        .expect_err("an unknown build");
    assert_eq!(err.status, 400);
    let cat = call(
        &mut wb,
        "GET",
        "/api/simlab/catalog?game_version=v0.111.0",
        json!({}),
    );
    assert!(cat["cards"].is_array());
    #[cfg(workbench_game_data_v0_111_0)]
    assert!(cat["cards"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["name"] == "Cacophony"));
}
