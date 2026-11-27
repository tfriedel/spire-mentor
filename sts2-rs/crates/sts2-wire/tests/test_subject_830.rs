//! Test Subject fidelity (#830 boss audit): BURNING_GROWL's Burns and the
//! later forms' A8 HP. Every expectation cites the v0.107.1 C# it reproduces.

mod common;

use common::{boss, boss_mut, boss_uid, end_turn, fight, play, power_amount};
use sts2_core::{CombatState, PowerKind, StepResult};

/// The Test Subject at A10, its move forced to `state_id`
/// (telegraph included) for the next enemy turn.
fn test_subject_on(state_id: &'static str, deck: &[&str]) -> CombatState {
    let mut state = fight("Ironclad", deck, 90, &[], &[], "TEST_SUBJECT_BOSS", 10);
    let uid = boss_uid(&state);
    sts2_core::hooks::apply_effects(
        &mut state,
        &[sts2_core::hooks::Effect::ForceMonsterMove {
            target: uid,
            state_id,
        }],
    );
    state
}

/// D5. BURNING_GROWL adds its Burns to the DISCARD pile
/// (`TestSubject.cs:335`, `PileType.Discard`; 5 at A9+, `:102`) and then gains
/// Strength (`:336`, 3 at A9+, `:104`). Rust used to put them in the hand.
#[test]
fn burning_growl_burns_go_to_the_discard_pile() {
    use sts2_core::cards::CardId;
    let mut state = test_subject_on("BURNING_GROWL", &["DEFEND_IRONCLAD"; 20]);
    let burns = |pile: &[sts2_core::CardUid], state: &CombatState| {
        pile.iter()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Burn)
            .count()
    };
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    // Player turn 2: the hand was drawn from the Defend-only draw pile.
    assert_eq!(burns(&state.player.discard, &state), 5);
    assert_eq!(burns(&state.player.hand, &state), 0);
    assert_eq!(burns(&state.player.draw, &state), 0);
    assert_eq!(power_amount(&state, boss(&state), PowerKind::Strength), 3);
}

/// D6. Forms 2 and 3 revive to `SecondFormHp` / `ThirdFormHp` =
/// `GetValueIfAscension(ToughEnemies, 212, 200)` / `(…, 313, 300)`
/// (`TestSubject.cs:84-86`, applied by `Revive`, `:339-345`): 212 and 313 at
/// A8+. Rust used the bare 200 / 300.
#[test]
fn test_subject_later_forms_scale_with_tough_enemies() {
    use sts2_core::cards::CardId;
    for (ascension, second, third) in [(10, 212, 313), (7, 200, 300)] {
        let mut state = fight(
            "Ironclad",
            &["STRIKE_IRONCLAD"; 20],
            90,
            &[],
            &[],
            "TEST_SUBJECT_BOSS",
            ascension,
        );
        for expected in [second, third] {
            boss_mut(&mut state).current_hp = 1;
            assert_eq!(
                play(&mut state, CardId::StrikeIronclad),
                StepResult::Advanced
            );
            assert_eq!(boss(&state).current_hp, 0, "dead until its next turn");
            // Its next enemy turn performs RESPAWN (`TestSubject.cs:258-284`).
            assert_eq!(end_turn(&mut state), StepResult::Advanced);
            let ts = boss(&state);
            assert_eq!(
                (ts.current_hp, ts.max_hp),
                (expected, expected),
                "A{ascension} revive HP"
            );
        }
    }
}

/// GH #1933: the wire carries no `Respawns`, so a rebuilt 3rd form reads its
/// revive count off its powers like the 2nd form does: Nemesis is applied on
/// the 2nd revive (`TestSubject.cs:271-274`). A 3rd form rebuilt on RESPAWN
/// (the move it revived through) must route REVIVE_BRANCH to the 3rd-form
/// opening, PHASE3_LACERATE (`Respawns >= 2`), not the 2nd form's MULTI_CLAW.
#[test]
fn a_rebuilt_third_form_counts_both_revives() {
    use sts2_core::cards::CardId;
    let mut state = fight(
        "Ironclad",
        &["STRIKE_IRONCLAD"; 20],
        900,
        &[],
        &[],
        "TEST_SUBJECT_BOSS",
        10,
    );
    for _ in 0..2 {
        boss_mut(&mut state).current_hp = 1;
        assert_eq!(
            play(&mut state, CardId::StrikeIronclad),
            StepResult::Advanced
        );
        assert_eq!(end_turn(&mut state), StepResult::Advanced);
    }
    assert!(power_amount(&state, boss(&state), PowerKind::Nemesis) > 0);
    let mut wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("serializes");
    let enemy = &mut wire["encounter"][0];
    enemy["move_state"] = "RESPAWN".into();
    enemy["ai"].as_object_mut().unwrap().remove("counter");
    let back = sts2_wire::state_from_json(&wire).expect("rebuilds");
    let forecast = sts2_wire::emit::combat::enemy_intent_forecast(&back, 1).expect("forecast");
    let step1 = forecast[0]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["step"] == 1)
        .expect("a step-1 forecast");
    assert_eq!(step1["move_id"], "PHASE3_LACERATE", "{forecast:?}");
}
