//! Supermassive's multiplier through a rebuild (corpus seeds 180000111,
//! 180000887: a Radiant Pearl Luminesce generated at combat start, before the
//! advisor's first mirror, was missing and the card read 3 damage short).

use sts2_wire::emit::combat::combat_state_to_value;

#[test]
fn the_combats_generated_card_count_survives_the_rebuild() {
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
    s.player.generated_cards_this_combat = 2;
    let v = combat_state_to_value(&s).expect("emits");
    assert_eq!(v["player"]["cards_generated_this_combat"], 2);
    let back = sts2_wire::state_from_json(&v).expect("re-parses");
    assert_eq!(back.player.generated_cards_this_combat, 2);
}
