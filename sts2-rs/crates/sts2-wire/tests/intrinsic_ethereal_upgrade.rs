//! The board flags a card `ethereal` for its own keyword too (Void Form).
//! The rebuild must not read that as a per-fight grant: C#
//! `VoidForm.OnUpgrade` removes Ethereal, and a grant stamped on top kept
//! the upgraded card Ethereal in the mirror (Spire Bench 180000322,
//! Armaments+ upgraded the hand).

use sts2_wire::emit::combat::combat_state_to_value;

#[test]
fn an_upgraded_void_form_rebuilt_from_its_ethereal_flag_is_not_ethereal() {
    let state = sts2_wire::scenario::build_scenario_state_for_version(
        sts2_core::GameVersion::V0_111_0,
        7,
        "Ironclad",
        &vec!["VOID_FORM".to_owned(); 10],
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
    let wire = combat_state_to_value(&state).expect("serializes");
    let hand = wire["deck"]["hand"].as_array().expect("hand");
    assert!(!hand.is_empty());
    assert!(
        hand[0]["flags"].to_string().contains("ethereal"),
        "the board shows Void Form Ethereal"
    );
    let mut back = sts2_wire::state_from_json(&wire).expect("parses");
    let uids: Vec<_> = back.player.hand.to_vec();
    for uid in uids {
        let card = back.cards.iter_mut().find(|c| c.uid == uid).expect("card");
        assert!(card.upgrade_for(sts2_core::GameVersion::V0_111_0));
    }
    let again = combat_state_to_value(&back).expect("serializes");
    for c in again["deck"]["hand"].as_array().expect("hand") {
        assert!(
            !c.get("flags")
                .map(ToString::to_string)
                .unwrap_or_default()
                .contains("ethereal"),
            "upgraded Void Form still Ethereal: {c}"
        );
    }
}
