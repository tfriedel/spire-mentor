//! The Fool (v0.113.0) on the live advisor: its fights are searched, and its
//! screens, which no run-policy net covers yet, say so instead of the whole
//! board being refused.

use serde_json::{json, Value};
use sts2_advisor::advise_combat::compute_combat_advice;
use sts2_advisor::display_names::Names;
use sts2_advisor::policy::{PolicyCache, NO_FOOL_NET};
use sts2_advisor::screen_advice::{compute_screen_advice, Outcome, ScreenEnv};
use sts2_advisor::search::RecommendConfig;
use sts2_advisor::serve::unsupported_character;
use sts2_core::potions::CharacterId;

fn fool_board() -> Value {
    serde_json::from_str(include_str!("fixtures/fool_v0113_nibbit.json")).expect("fixture parses")
}

#[test]
fn a_fool_fight_is_searched() {
    let board = fool_board();
    assert_eq!(unsupported_character(&board), None);
    let mut log = Vec::new();
    let out = compute_combat_advice(
        &board,
        &RecommendConfig::default(),
        64,
        sts2_advisor::turn_plan::PLAN_BUDGET_S,
        None,
        &Names::default(),
        &mut log,
    )
    .expect("the Fool's fight builds and searches");
    assert_eq!(out.cfg.character.as_deref(), Some("Fool"));
    assert_eq!(out.fight.state.player.character, Some(CharacterId::Fool));
    let best = &out.advice["best"];
    assert!(best.is_object(), "a ranked best action: {}", out.advice);
    assert!(
        out.advice["options"]
            .as_array()
            .is_some_and(|o| o.len() > 1),
        "{}",
        out.advice
    );
}

#[test]
fn a_fool_ally_rides_along_in_the_search() {
    let mut board = fool_board();
    board["run_state"]["character"] = json!("IRONCLAD");
    board["allies"] = json!([{"hp": 70, "max_hp": 77, "block": 0, "character": "FOOL"}]);
    for c in board["hand"].as_array_mut().unwrap() {
        if c["id"] == "FOOLISH_SHOT" {
            c["id"] = json!("BASH");
        }
    }
    assert_eq!(unsupported_character(&board), None);
    let mut log = Vec::new();
    compute_combat_advice(
        &board,
        &RecommendConfig::default(),
        32,
        sts2_advisor::turn_plan::PLAN_BUDGET_S,
        None,
        &Names::default(),
        &mut log,
    )
    .expect("a Fool ally does not stop the search");
}

#[test]
fn a_fool_screen_has_no_run_net_and_says_so() {
    let state = json!({
        "type": "map_select",
        "game_version": "v0.113.0",
        "run_state": {"character": "FOOL"},
        "options": [{"index": 0, "label": "Monster"}, {"index": 1, "label": "Elite"}],
    });
    let cache = PolicyCache::new(std::path::PathBuf::from("/nonexistent"));
    let mut misses = Vec::new();
    let mut policy_for = |character: &str| cache.lookup(character, &mut misses);
    let mut wire_for = sts2_advisor::board::wire_for;
    let names = Names::default();
    let mut env = ScreenEnv {
        session: None,
        mirror: None,
        cfg: None,
        sims: None,
        names: &names,
        wire_for: &mut wire_for,
        policy_for: &mut policy_for,
        log: Vec::new(),
    };
    let Outcome::Advice(advice) = compute_screen_advice(&state, "map_select", &mut env) else {
        panic!("answered here")
    };
    assert_eq!(
        advice["note"],
        json!(format!("no run-policy net: {NO_FOOL_NET}"))
    );
    assert!(advice["hand_back"].is_string(), "{advice}");
}
