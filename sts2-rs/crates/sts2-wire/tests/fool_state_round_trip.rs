//! v0.113.0: `PlayerCombatState::fool` (arrows, impaled lists, melody
//! tallies, the Aim/Stretch/Quietus latches, Encore/Lullaby bookkeeping)
//! survives a `state_to_dict` -> `state_from_dict` round trip.

use smallvec::smallvec;
use sts2_core::fool::{ArrowSlot, ArrowStatus, FoolState, MelodyTurn};
use sts2_wire::emit::combat_state_to_value;
use sts2_wire::state_from_json;

fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    serde_json::from_str(&text).expect("fixture parses")
}

fn arrow(status: ArrowStatus, destroy_after_impaled: bool) -> ArrowSlot {
    ArrowSlot {
        status,
        destroy_after_impaled,
    }
}

#[test]
fn fool_state_round_trips() {
    let mut st = state_from_json(&fixture()).expect("builds");
    let card = st.player.hand[0];
    let enemy = st.enemy_uids[0];
    let round = st.round;
    let f = FoolState {
        arrows: smallvec![
            arrow(ArrowStatus::Retrieved, false),
            arrow(ArrowStatus::Away, true),
            arrow(ArrowStatus::Destroyed, false),
        ],
        impaled: smallvec![(enemy, smallvec![1, 0])],
        melody: MelodyTurn {
            stamp: Some((round, true)),
            finished: [1, 2, 0],
            melody_cards_finished: 2,
            started: [1, 2, 1],
            first_started_seq: [3, 4, 0],
        },
        play_seq: 7,
        open_plays: smallvec![(7, card)],
        aim_latch: Some(5),
        aim_source: Some(card),
        quietus_latched: true,
        arrow_attack_depth: 1,
        stretch_latch: Some((card, 6, true)),
        encore_plays: 2,
        last_finished_seq: 6,
        lullaby_round: Some(round),
        attacked_card: Some(card),
    };
    st.player.fool = Some(Box::new(f.clone()));
    let back = state_from_json(&combat_state_to_value(&st).expect("emits")).expect("re-parses");
    assert_eq!(back.player.fool.as_deref(), Some(&f));
}

#[test]
fn no_fool_state_emits_nothing() {
    let st = state_from_json(&fixture()).expect("builds");
    assert!(st.player.fool.is_none());
    let v = combat_state_to_value(&st).expect("emits");
    assert!(v.get("fool_state").is_none());
    assert!(state_from_json(&v)
        .expect("re-parses")
        .player
        .fool
        .is_none());
}

#[test]
fn a_bad_fool_state_is_a_schema_error() {
    for bad in [
        serde_json::json!("x"),
        serde_json::json!([99]),
        serde_json::json!([1, 5]),
    ] {
        let mut v = fixture();
        v["fool_state"] = bad;
        assert!(state_from_json(&v).is_err(), "{v}");
    }
}
