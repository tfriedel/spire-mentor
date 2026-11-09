//! `state_to_dict` → `state_from_dict` keeps each enemy's move machine
//! (#1181): the move log the repeat rules read (`CannotRepeat`,
//! `UseOnlyOnce`, cooldowns), the performed flags, the counter and a queued
//! forced move. Before, a rebuilt AI started with an empty log, so it could
//! repeat a move the game forbids or roll a once-per-fight move again.

use sts2_core::{apply, Action, CombatState, StepResult};
use sts2_wire::emit::combat::combat_state_to_value;

/// log, performed_first_move, performed, forced, counter, forced_next,
/// current, initial_resolved.
type Machine = (
    Vec<&'static str>,
    bool,
    bool,
    bool,
    i32,
    Option<&'static str>,
    &'static str,
    bool,
);

/// Every field of the machine a roll reads, per enemy.
fn machines(state: &CombatState) -> Vec<Option<Machine>> {
    state
        .enemy_uids
        .iter()
        .map(|uid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0)?;
            let ai = c.ai.as_ref()?;
            Some((
                ai.move_history().to_vec(),
                ai.performed_first_move(),
                ai.performed_at_least_once(),
                ai.current_forced(),
                ai.counter(),
                ai.forced_next_state(),
                ai.current_state_id(),
                ai.initial_resolved(),
            ))
        })
        .collect()
}

fn roster(state: &CombatState) -> Vec<(i32, Option<&'static str>)> {
    // (hp, kind) per enemy, in roster order.
    state
        .enemy_uids
        .iter()
        .map(|uid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
            (
                c.current_hp,
                c.ai.as_ref().map(sts2_core::MonsterAi::current_state_id),
            )
        })
        .collect()
}

fn moves(state: &CombatState) -> Vec<Option<&'static str>> {
    state
        .enemy_uids
        .iter()
        .map(|uid| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .and_then(|c| c.ai.as_ref())
                .map(sts2_core::MonsterAi::current_state_id)
        })
        .collect()
}

#[test]
fn across_every_encounter_a_rebuilt_state_keeps_its_move_machines() {
    let mut compared = 0usize;
    let mut with_history = 0usize;
    let mut misses: Vec<String> = Vec::new();
    let mut reordered = 0usize;
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
            for turn in 0..8 {
                if state.is_over {
                    break;
                }
                let wire = combat_state_to_value(&state).expect("serializes");
                let mut back = sts2_wire::state_from_json(&wire).expect("round-trips");
                let ctx = format!("{version:?} {} turn {turn}", def.id);
                // Only enemies the rebuild parked on the same move are
                // comparable; the census of `move_state` itself lives
                // elsewhere.
                let (want, got) = (machines(&state), machines(&back));
                assert_eq!(want.len(), got.len(), "{ctx}");
                for (w, g) in want.iter().zip(&got) {
                    if let (Some(w), Some(g)) = (w, g) {
                        if w.6 == g.6 {
                            if w != g {
                                misses.push(format!("{ctx}: machine {w:?} vs rebuilt {g:?}"));
                            }
                            compared += 1;
                            with_history += usize::from(!w.0.is_empty());
                        }
                    }
                }
                // Both roll the same moves next turn.
                let a = apply(&mut state, Action::EndTurn);
                let b = apply(&mut back, Action::EndTurn);
                assert_eq!(a, b, "{ctx}");
                // A summon's encounter slot is not on the wire (Fabricator
                // bots, Ovicopter eggs, TwoTailedRat's backup), so the rebuild
                // can list the roster in another order, and the monster RNG
                // is drawn in roster order. That is a separate gap; the rolls
                // are compared only while the rosters line up.
                if roster(&state) != roster(&back) {
                    reordered += 1;
                    break;
                }
                if moves(&state) != moves(&back) {
                    misses.push(format!(
                        "{ctx}: next moves {:?} vs rebuilt {:?}",
                        moves(&state),
                        moves(&back)
                    ));
                    break;
                }
                if a != StepResult::Advanced {
                    break;
                }
            }
        }
    }
    eprintln!("{compared} machines compared, {with_history} with a history, {reordered} rosters reordered");
    assert!(misses.is_empty(), "{}", misses.join("\n"));
    assert!(reordered < 10, "{reordered} rosters reordered");
    assert!(compared > 1000, "{compared} machines compared");
    assert!(with_history > 500, "{with_history} machines with a history");
}

/// The C# producers' convention: `log` holds the current move while it is
/// rolled but unperformed. A C#-style log ending on the current telegraph
/// restores the engine log without it.
#[test]
fn a_csharp_log_drops_the_unperformed_current_move() {
    let state = sts2_wire::scenario::build_scenario_state_for_version(
        sts2_core::GameVersion::V0_111_0,
        7,
        "Ironclad",
        &vec!["DEFEND_IRONCLAD".to_owned(); 20],
        9999,
        9999,
        &[],
        &[],
        "TERROR_EEL_ELITE",
        10,
        None,
        None,
    )
    .expect("builds");
    let mut s = state.clone();
    for _ in 0..2 {
        assert_eq!(apply(&mut s, Action::EndTurn), StepResult::Advanced);
    }
    let mut wire = combat_state_to_value(&s).expect("serializes");
    let engine_log = machines(&s)[0].as_ref().unwrap().0.clone();
    let current = machines(&s)[0].as_ref().unwrap().6;
    // What C# sends: the log with the rolled current move, C#-spelled ids,
    // and no engine-only keys.
    let enemy = &mut wire["encounter"][0];
    let mut log: Vec<String> = engine_log.iter().map(ToString::to_string).collect();
    log.push(current.to_owned());
    enemy["ai"] = serde_json::json!({
        "log": log,
        "performed_first_move": true,
        "performed": false,
    });
    let back = sts2_wire::state_from_json(&wire).expect("round-trips");
    assert_eq!(machines(&back)[0].as_ref().unwrap().0, engine_log);
}

/// A monster stunned by the generic stun (C# `Creature.StunInternal`'s
/// synthetic `"STUNNED"`), sent the way the C# producers send it (no
/// engine-only keys, `forced_next` = the stun's `FollowUpStateId`), rebuilds
/// stunned, with its log and the move it resumes. It used to fall back to
/// the AI's initial move.
#[test]
fn a_generic_stun_rebuilds_with_its_resume_from_a_csharp_dict() {
    let mut state = sts2_wire::scenario::build_scenario_state_for_version(
        sts2_core::GameVersion::V0_111_0,
        7,
        "Ironclad",
        &vec!["DEFEND_IRONCLAD".to_owned(); 20],
        9999,
        9999,
        &[],
        &[],
        "OWL_MAGISTRATE_NORMAL",
        10,
        None,
        None,
    )
    .expect("builds");
    for _ in 0..2 {
        assert_eq!(apply(&mut state, Action::EndTurn), StepResult::Advanced);
    }
    let uid = state.enemy_uids[0];
    let owl = state.creatures.iter_mut().find(|c| c.id == uid.0).unwrap();
    let ai = owl.ai.as_mut().unwrap();
    let resume = ai.current_state_id();
    let stun = ai.ensure_generic_stun_state();
    assert!(ai.stun_immediate(stun, None));
    let want = machines(&state)[0].clone().unwrap();
    assert_eq!(want.6, "STUNNED");
    assert_eq!(want.5, Some(resume));

    let mut wire = combat_state_to_value(&state).expect("serializes");
    let enemy = &mut wire["encounter"][0];
    let ai = enemy["ai"].as_object_mut().unwrap();
    for engine_only in ["forced", "counter"] {
        ai.remove(engine_only);
    }
    let back = sts2_wire::state_from_json(&wire).expect("round-trips");
    let got = machines(&back)[0].clone().unwrap();
    assert_eq!(got.6, "STUNNED", "rebuilt on the stun");
    assert_eq!(got.5, Some(resume), "resumes the interrupted move");
    assert_eq!(got.0, want.0, "the log");
    assert!(got.3, "inferred as forced: the log does not end on it");

    // Without `forced_next` (an older C# producer), C#'s default resume:
    // the last logged move.
    let mut wire2 = wire.clone();
    wire2["encounter"][0]["ai"]
        .as_object_mut()
        .unwrap()
        .remove("forced_next");
    let back2 = sts2_wire::state_from_json(&wire2).expect("round-trips");
    assert_eq!(machines(&back2)[0].clone().unwrap().5, Some(resume));
}
