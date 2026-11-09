//! `peek_intents` on the three moves whose numbers C# computes when it
//! telegraphs them (#1181): TestSubject MULTI_CLAW's hit count, TheForgotten
//! DREAD and WaterfallGiant EXPLODE. Each forecast step's intents must be the
//! intents the fight then telegraphs at that step, steps 2 and 3 included;
//! the move table alone gets all three wrong.

mod common;

use common::{boss, boss_mut, end_turn, fight, move_id, play, pressurized_giant};
use serde_json::{json, Value};
use sts2_core::cards::CardId;
use sts2_core::{CombatState, StepResult};
use sts2_wire::emit::combat::{enemy_intent_forecast, enemy_telegraphs};

/// `{kind, amount, times}` of an intent dict (no board-baked keys).
fn bare(intent: &Value) -> Value {
    json!([intent["kind"], intent["amount"], intent["times"]])
}

fn bare_list(intents: &Value) -> Vec<Value> {
    intents
        .as_array()
        .expect("intent list")
        .iter()
        .map(bare)
        .collect()
}

/// Ends `turns` turns. Before each, records every enemy's 3-step forecast;
/// after each, checks the telegraph against every earlier certain forecast
/// that predicted this turn (same move id). Returns how many step-1, -2 and
/// -3 predictions of `move_of_interest` were checked.
fn walk_and_check(state: &mut CombatState, turns: usize, move_of_interest: &str) -> [usize; 3] {
    let mut history: Vec<Value> = Vec::new();
    let mut checked = [0_usize; 3];
    for _ in 0..turns {
        if state.is_over {
            break;
        }
        history.push(enemy_intent_forecast(state, 3).expect("forecast"));
        if end_turn(state) != StepResult::Advanced {
            break;
        }
        let tele = enemy_telegraphs(state).expect("telegraph");
        let now = history.len();
        for (made, forecast) in history.iter().enumerate() {
            let step = now - made; // this turn is step `step` of that forecast
            if step > 3 {
                continue;
            }
            for (e, steps) in forecast.as_array().unwrap().iter().enumerate() {
                let Some(s) = steps.as_array().and_then(|l| l.get(step - 1)) else {
                    continue;
                };
                let uid = state.enemy_uids[e];
                let c = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
                if c.current_hp <= 0 || s["certainty"] != 1.0 {
                    continue;
                }
                let Some(ai) = c.ai.as_ref() else { continue };
                if s["move_id"] != ai.current_state_id() {
                    continue;
                }
                assert_eq!(
                    bare_list(&s["intents"]),
                    bare_list(&tele[e]),
                    "step {step} forecast of {} made {} turn(s) ago",
                    ai.current_state_id(),
                    step
                );
                if ai.current_state_id() == move_of_interest {
                    checked[step - 1] += 1;
                }
            }
        }
    }
    checked
}

#[test]
fn dread_forecasts_the_dexterity_each_miasma_adds() {
    let mut state = fight(
        "Ironclad",
        &["DEFEND_IRONCLAD"; 20],
        900,
        &[],
        &[],
        "THE_LOST_AND_FORGOTTEN_NORMAL",
        10,
    );
    // TheForgotten opens on MIASMA, so its step-1 DREAD comes after one
    // MIASMA: A9+ base 15 plus 2 Dexterity, where the table holds 15.
    let f = enemy_intent_forecast(&state, 3).expect("forecast");
    let forgotten = state
        .enemy_uids
        .iter()
        .position(|u| {
            state
                .creatures
                .iter()
                .find(|c| c.id == u.0)
                .unwrap()
                .species
                == Some(sts2_core::MonsterKind::TheForgotten)
        })
        .expect("TheForgotten");
    assert_eq!(f[forgotten][0]["move_id"], "DREAD");
    assert_eq!(f[forgotten][0]["intents"][0]["amount"], 17);
    assert_eq!(f[forgotten][2]["move_id"], "DREAD");
    assert_eq!(f[forgotten][2]["intents"][0]["amount"], 19);

    let checked = walk_and_check(&mut state, 8, "DREAD");
    assert!(checked.iter().all(|&n| n > 0), "DREAD checked {checked:?}");
}

#[test]
fn multi_claw_forecasts_one_more_hit_per_repeat() {
    let mut state = fight(
        "Ironclad",
        &["STRIKE_IRONCLAD"; 20],
        2000,
        &[],
        &[],
        "TEST_SUBJECT_BOSS",
        10,
    );
    // Kill the first form: the second form is the one with MULTI_CLAW.
    boss_mut(&mut state).current_hp = 1;
    assert_eq!(
        play(&mut state, CardId::StrikeIronclad),
        StepResult::Advanced
    );
    // Dead until its next enemy turn, which performs RESPAWN
    // (`TestSubject.cs:258-284`).
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert!(boss(&state).current_hp > 1, "the TestSubject revived");
    let checked = walk_and_check(&mut state, 14, "MULTI_CLAW");
    assert!(
        checked.iter().all(|&n| n > 0),
        "MULTI_CLAW checked {checked:?} (moves: {})",
        move_id(boss(&state))
    );
}

#[test]
fn explode_forecasts_the_steam_about_to_blow_captures() {
    let mut state = pressurized_giant(&["STRIKE_IRONCLAD"; 20], &[], &[]);
    boss_mut(&mut state).current_hp = 1;
    assert_eq!(
        play(&mut state, CardId::StrikeIronclad),
        StepResult::Advanced
    );
    assert_eq!(move_id(boss(&state)), "ABOUT_TO_BLOW");
    let f = enemy_intent_forecast(&state, 1).expect("forecast");
    assert_eq!(f[0][0]["move_id"], "EXPLODE");
    let steam = common::power_amount(&state, boss(&state), sts2_core::PowerKind::SteamEruption);
    assert!(steam > 0);
    assert_eq!(f[0][0]["intents"][0]["amount"], steam, "the table holds 0");

    let checked = walk_and_check(&mut state, 1, "EXPLODE");
    assert_eq!(checked[0], 1, "EXPLODE checked {checked:?}");
}

/// `state_from_dict` sets each rebuilt AI's owner (#1181), so a forecast
/// read before the AI's first roll resolves owner-aware predicates; and the
/// forecast of a round-tripped state matches the original's.
#[test]
fn a_round_tripped_state_keeps_the_ai_owner_and_the_forecast() {
    let state = fight(
        "Ironclad",
        &["DEFEND_IRONCLAD"; 20],
        900,
        &[],
        &[],
        "THE_LOST_AND_FORGOTTEN_NORMAL",
        10,
    );
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("serializes");
    let back = sts2_wire::state_from_json(&wire).expect("round-trips");
    for uid in &back.enemy_uids {
        let c = back.creatures.iter().find(|c| c.id == uid.0).unwrap();
        assert_eq!(
            c.ai.as_ref().and_then(sts2_core::MonsterAi::owner_uid),
            Some(*uid)
        );
    }
    assert_eq!(
        enemy_intent_forecast(&back, 3).expect("forecast"),
        enemy_intent_forecast(&state, 3).expect("forecast")
    );
}

/// The moves that rewrite their own table for the next use (PRESSURE_GUN,
/// INCREASING_INTENSITY_MOVE, PHEROMONE_SPIT): a step past another use of the
/// same move shows the ramp the fight then telegraphs.
#[test]
fn self_ramping_moves_forecast_their_next_uses() {
    for (encounter, mv, version) in [
        (
            "WATERFALL_GIANT_BOSS",
            "PRESSURE_GUN",
            sts2_core::GameVersion::V0_111_0,
        ),
        (
            "AEONGLASS_BOSS",
            "INCREASING_INTENSITY_MOVE",
            sts2_core::GameVersion::V0_111_0,
        ),
        (
            "ENTOMANCER_ELITE",
            "PHEROMONE_SPIT",
            sts2_core::GameVersion::V0_111_0,
        ),
        (
            "ENTOMANCER_ELITE",
            "PHEROMONE_SPIT",
            sts2_core::GameVersion::V0_107_1,
        ),
    ] {
        let mut state = sts2_wire::scenario::build_scenario_state_for_version(
            version,
            7,
            "Ironclad",
            &vec!["DEFEND_IRONCLAD".to_owned(); 20],
            9999,
            9999,
            &[],
            &[],
            encounter,
            10,
            None,
            None,
        )
        .expect("scenario builds");
        let checked = walk_and_check(&mut state, 16, mv);
        assert!(
            checked[1] + checked[2] > 0,
            "{encounter} {version:?}: {mv} steps 2-3 checked {checked:?}"
        );
    }
}

/// Every encounter on both builds, played passively for eight enemy turns:
/// each certain forecast step (1-3) that names the move the fight then
/// telegraphs carries that telegraph's intents.
#[test]
fn across_every_encounter_forecast_intents_are_the_telegraphs() {
    for version in sts2_core::GameVersion::ALL {
        for idx in 0..sts2_content::encounters::n_encounters() {
            let def = sts2_content::encounters::by_index(idx).expect("in range");
            let mut state = sts2_wire::scenario::build_scenario_state_for_version(
                version,
                7,
                "Ironclad",
                &vec!["DEFEND_IRONCLAD".to_owned(); 20],
                9999,
                9999,
                &[],
                &[],
                def.id,
                10,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{version:?} {}: {e}", def.id));
            let _ = walk_and_check(&mut state, 8, "");
        }
    }
}
