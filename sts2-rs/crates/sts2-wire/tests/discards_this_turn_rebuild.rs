//! The emitter writes `discards_this_turn` (GH #1197) and the host sends it;
//! a rebuild must read it back (Memento Mori's multiplier, the net's
//! S_DISCARDS_TURN). Spire Bench seed 180001050: Whispering Earring's
//! turn-1 auto-plays discarded 3 before the first screen; the rebuild read 0.

use sts2_wire::emit::combat::combat_state_to_value;

#[test]
fn the_turns_discard_count_survives_the_rebuild() {
    let deck: Vec<String> = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"]
        .map(str::to_owned)
        .to_vec();
    let mut s = sts2_wire::scenario::build_scenario_state_for_version(
        sts2_core::GameVersion::V0_111_0,
        7,
        "Ironclad",
        &deck,
        80,
        80,
        &[],
        &[],
        "AXEBOTS_NORMAL",
        10,
        None,
        None,
    )
    .expect("scenario");
    s.combat_log.cards_discarded_this_turn = 3;
    let v = combat_state_to_value(&s).expect("emits");
    assert_eq!(v["player"]["discards_this_turn"], 3);
    let back = sts2_wire::state_from_json(&v).expect("re-parses");
    assert_eq!(back.combat_log.cards_discarded_this_turn, 3);
}
