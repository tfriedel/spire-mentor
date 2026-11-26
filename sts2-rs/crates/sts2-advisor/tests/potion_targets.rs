//! A targeted potion's advice names its target, as a targeted card's does
//! (GH #1769): `Fire Potion → <enemy>` in the label and `target_index` on
//! `best`, which the mod reads to aim the drink and to put its reticle on the
//! enemy.
//!
//! Before GH #1604 the engine offered every potion as one untargeted move, so
//! neither Instinct nor the search ever had a target to name: the label was
//! the bare potion name and the game threw it at its first living enemy.

use serde_json::{json, Value};
use sts2_advisor::advise_combat::compute_combat_advice;
use sts2_advisor::board_targets::BoardTargets;
use sts2_advisor::combat_advice::instinct_combat_advice;
use sts2_advisor::display_names::Names;
use sts2_advisor::instinct::Pick;
use sts2_advisor::instinct_advisor::engine_game_version;
use sts2_advisor::instinct_combat::{combat_fight, enter_live_combat, room_kind};
use sts2_advisor::search::RecommendConfig;
use sts2_e2e::env::{Driver, Next};

fn board() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/ironclad_fire_potion_three_enemies.json"
    ))
    .expect("fixture parses")
}

fn enemy_names(board: &Value) -> Vec<String> {
    board["enemies"]
        .as_array()
        .expect("enemies")
        .iter()
        .map(|e| e["name"].as_str().expect("name").to_string())
        .collect()
}

/// The Fire Potion rows of an advice: `(target_index, label)`.
fn fire_potion_rows(advice: &Value) -> Vec<(Value, String)> {
    advice["options"]
        .as_array()
        .expect("options")
        .iter()
        .filter(|o| o["action"] == "potion" && o["slot"] == 0)
        .map(|o| {
            (
                o["target_index"].clone(),
                o["label"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// Instinct, minus the net: the decision the student is asked on this board
/// (the grafted run entered into the fight, peeked by a fresh driver), its
/// mass on the Fire Potion thrown at the second enemy.
#[test]
fn instincts_fire_potion_names_the_enemy_it_is_thrown_at() {
    sts2_advisor::serve::install_engine();
    let board = board();
    let (mut rm, _notes) =
        sts2_advisor::graft::graft_run(&board, engine_game_version(&board)).expect("graft");
    let (fight, wire) =
        combat_fight(&board, &RecommendConfig::default(), None).expect("the fight builds");
    let _ = enter_live_combat(&mut rm, fight, &board, room_kind(&board));
    let mut driver = Driver::new(i64::from(rm.seed));
    let Next::Decision(d) = driver.peek(&mut rm) else {
        panic!("a combat turn to decide");
    };
    let decision = d.to_value();
    let classes = decision["classes"].as_array().expect("classes");
    let aimed_at = |target: i64| {
        classes.iter().position(|c| {
            let a = &c["actions"][0];
            a["action"] == "drink_potion" && a["slot"] == 0 && a["target_index"] == target
        })
    };
    let best = aimed_at(1).unwrap_or_else(|| {
        panic!("no class throws the Fire Potion at enemy 1: {decision}");
    });
    let mut probs = vec![0.1 / (classes.len() - 1) as f64; classes.len()];
    probs[best] = 0.9;
    let pick = Pick {
        decision: decision.clone(),
        probs,
        best,
        ms: 0.0,
    };

    let advice = instinct_combat_advice(&pick, &board, &wire, &BoardTargets::default())
        .expect("Instinct advises");
    let names = enemy_names(&board);
    assert_eq!(
        advice["best"],
        json!({
            "action": "potion",
            "slot": 0,
            "target_index": 1,
            "label": format!("Fire Potion → {}", names[1]),
        }),
        "{advice}"
    );
    let mut rows = fire_potion_rows(&advice);
    rows.sort_by_key(|(t, _)| t.as_i64());
    let want: Vec<(Value, String)> = (0..3)
        .map(|i| (json!(i), format!("Fire Potion → {}", names[i])))
        .collect();
    assert_eq!(rows, want, "one Fire Potion row per enemy: {advice}");
}

/// The search's rows name the same targets.
#[test]
fn the_searchs_fire_potion_rows_name_their_enemies() {
    // The potion target types live in content's registry (the advisor
    // installs it at start-up); without it every potion is untargeted.
    sts2_advisor::serve::install_engine();
    let board = board();
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
    .expect("the fight searches");
    let names = enemy_names(&board);
    let mut rows = fire_potion_rows(&out.advice);
    rows.sort_by_key(|(t, _)| t.as_i64());
    let want: Vec<(Value, String)> = (0..3)
        .map(|i| (json!(i), format!("Fire Potion → {}", names[i])))
        .collect();
    assert_eq!(rows, want, "{}", out.advice);
}
