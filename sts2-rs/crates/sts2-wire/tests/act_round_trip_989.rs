//! GH #989: the act and a boss room survive the wire.
//!
//! `state_from_dict` never set `CombatState::act_index` / `is_boss`, so every
//! rebuilt board was act 1 and not a boss. A co-op Axebot respawned on floor
//! 44 (act 3) with HP scaled by act 1's 1.1 instead of 1.2 (189 vs the game's
//! 206), and the persistent-gain horizon counted the final boss as act 1.

use sts2_core::PlayerCombatState;
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

#[test]
fn the_act_and_a_boss_room_set_the_fight() {
    let mut v = fixture();
    v["act_index"] = serde_json::json!(3);
    v["is_boss"] = serde_json::json!(true);
    let mut st = state_from_json(&v).expect("builds");
    assert_eq!((st.act_index, st.is_boss), (3, true));
    // Two players in act 3's boss fight: × 2 × 1.3.
    st.allies.push(PlayerCombatState::default());
    assert_eq!(sts2_core::coop::scale_for_players(&st, 100), 260);
    st.is_boss = false;
    assert_eq!(sts2_core::coop::scale_for_players(&st, 100), 240);
}

#[test]
fn the_act_and_a_boss_room_round_trip_through_the_emitter() {
    for (act, boss) in [(2u8, false), (3, true), (1, true)] {
        let mut st = state_from_json(&fixture()).expect("builds");
        st.act_index = act;
        st.is_boss = boss;
        let back = state_from_json(&combat_state_to_value(&st).expect("emits")).expect("re-parses");
        assert_eq!((back.act_index, back.is_boss), (act, boss));
    }
}

#[test]
fn absent_keys_keep_act_one_and_no_boss() {
    let st = state_from_json(&fixture()).expect("builds");
    assert_eq!((st.act_index, st.is_boss), (1, false));
    let v = combat_state_to_value(&st).expect("emits");
    assert!(v.get("act_index").is_none() && v.get("is_boss").is_none());
}

#[test]
fn a_bad_act_is_a_schema_error() {
    for bad in [
        serde_json::json!(0),
        serde_json::json!(-1),
        serde_json::json!("3"),
    ] {
        let mut v = fixture();
        v["act_index"] = bad;
        assert!(state_from_json(&v).is_err());
    }
}

/// v0.113.0 TheTower reads `RunState.TotalFloor`: it rides the wire as
/// `total_floor`, absent at 0.
#[test]
fn total_floor_round_trips() {
    let st = state_from_json(&fixture()).expect("builds");
    assert_eq!(st.total_floor, 0);
    assert!(combat_state_to_value(&st)
        .expect("emits")
        .get("total_floor")
        .is_none());
    let mut v = fixture();
    v["total_floor"] = serde_json::json!(23);
    let st = state_from_json(&v).expect("builds");
    assert_eq!(st.total_floor, 23);
    let back = state_from_json(&combat_state_to_value(&st).expect("emits")).expect("re-parses");
    assert_eq!(back.total_floor, 23);
}
