//! `/api/seedcheck/*` over `Workbench::call`: a bad seed is a 400 naming the
//! character, attempts and logs need the net (424), and with the net an
//! attempt, its summary and its run log agree.
#![cfg(feature = "e2e")]

use serde_json::{json, Value};
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

fn call(wb: &mut Workbench, route: &str, body: Value) -> Value {
    match wb.call("POST", route, &body) {
        Ok(v) => v,
        Err(e) => panic!("{route} -> {} {}", e.status, e.detail),
    }
}

#[test]
fn seedcheck_routes_reject_bad_seed() {
    let mut wb = new_wb();
    let e = wb
        .call("POST", "/api/seedcheck/preview", &json!({"seed": "AB-C"}))
        .expect_err("bad seed");
    assert_eq!(e.status, 400);
    assert!(e.detail.contains("alphabet"), "{}", e.detail);
    let e = wb
        .call(
            "POST",
            "/api/seedcheck/attempt",
            &json!({"seed": "AB-C", "seq": "check", "k": 0}),
        )
        .expect_err("bad seed");
    assert_eq!(e.status, 400);
}

#[test]
fn seedcheck_attempt_needs_net() {
    let mut wb = new_wb();
    let body = json!({"seed": "1SNVTCEXKR0Q", "seq": "check", "k": 0});
    assert_eq!(
        wb.call("POST", "/api/seedcheck/attempt", &body)
            .unwrap_err()
            .status,
        424
    );
    let body = json!({"seed": "1SNVTCEXKR0Q", "classes": [], "seq": "check", "k": 0, "now": ""});
    assert_eq!(
        wb.call("POST", "/api/seedcheck/log", &body)
            .unwrap_err()
            .status,
        424
    );
}

#[test]
fn seedcheck_preview_route() {
    let mut wb = new_wb();
    let p = call(
        &mut wb,
        "/api/seedcheck/preview",
        json!({"seed": "1snvtcexkr0q"}),
    );
    assert_eq!(p["seed"], "1SNVTCEXKR0Q");
    assert_eq!(p["neow"].as_array().unwrap().len(), 3);
}

#[test]
fn greedy_attempt_summary_and_log_agree() {
    let Some(w) = weights() else {
        eprintln!("skipping: no e2e weights");
        return;
    };
    let mut wb = new_wb();
    wb.load_e2e(&w).expect("net loads");
    let seed = "1SNVTCEXKR0Q";
    let a = call(
        &mut wb,
        "/api/seedcheck/attempt",
        json!({"seed": seed, "seq": "check", "k": 0}),
    );
    // Net-agnostic: the shipped net need not win this seed greedily (d1250
    // did, d2500 avg5 died on floor 22, d5250 on floor 46); summary and log
    // must agree with it.
    assert_eq!(a["won"], a["death_room"].is_null(), "{a}");
    assert_eq!(a["temp"], 0.0);
    let s = call(
        &mut wb,
        "/api/seedcheck/summary",
        json!({"seed": seed, "classes": a["classes"]}),
    );
    assert_eq!(s["won"], a["won"], "{s}");
    let log = call(
        &mut wb,
        "/api/seedcheck/log",
        json!({"seed": seed, "classes": a["classes"], "seq": "check", "k": 0, "now": "2026-11-08T12:00:00Z"}),
    );
    let path = log["path"].as_str().unwrap().to_string();
    let text = log["text"].as_str().unwrap().to_string();
    assert_eq!(
        std::path::Path::new(&path).extension(),
        Some("jsonl".as_ref())
    );
    // The replay viewer opens it.
    wb.load_run(&path, &text).expect("the log loads");
}

#[test]
fn seedcheck_routes_check_character_version_and_ascension() {
    let mut wb = new_wb();
    let pv = |wb: &mut Workbench, extra: Value| {
        let mut body = json!({"seed": "1SNVTCEXKR0Q"});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        wb.call("POST", "/api/seedcheck/preview", &body)
    };
    let p = pv(
        &mut wb,
        json!({"ascension": 0, "character": "Ironclad", "game_version": "v0.111.0"}),
    )
    .unwrap();
    assert_eq!(p["ascension"], 0);
    // No double boss below A10.
    assert_eq!(p["bosses"][2].as_array().unwrap().len(), 1);
    assert_eq!(pv(&mut wb, json!({})).unwrap()["ascension"], 10);
    for (extra, needle) in [
        (json!({"character": "Silent"}), "Silent"),
        (json!({"game_version": "v0.107.1"}), "v0.107.1"),
        (json!({"ascension": 11}), "0-10"),
        (json!({"ascension": "ten"}), "0-10"),
    ] {
        let e = pv(&mut wb, extra.clone()).expect_err("refused");
        assert_eq!(e.status, 400, "{extra}");
        assert!(e.detail.contains(needle), "{extra}: {}", e.detail);
    }
}
