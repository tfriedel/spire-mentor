//! v0.113.0 `StretchPower.ModifyBlockAdditive` on a card-face preview.
//! `BlockVar` / `CalculatedBlockVar` preview through
//! `Hook.ModifyBlock(..., card, cardPlay: null)`, and Stretch answers a
//! null `cardPlay` with its `Amount` (unspent) whenever no play is latched —
//! but 0 while another play holds the latch.

use sts2_core::{GameVersion, PowerKind};
use sts2_wire::emit::card_block_preview;
use sts2_wire::state_from_json;

fn state_with_a_defend() -> sts2_core::CombatState {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["deck"]["hand"] = serde_json::json!([
        {"id": "DEFEND_IRONCLAD", "stable_idx": 4, "kind": "card", "upgraded": false}
    ]);
    let mut st = state_from_json(&v).expect("builds");
    st.game_version = GameVersion::V0_113_0;
    st
}

fn defend_preview(st: &sts2_core::CombatState) -> i32 {
    card_block_preview(st)[0].expect("a Defend shows block")
}

#[test]
fn stretch_adds_its_amount_to_an_unlatched_block_preview() {
    let mut st = state_with_a_defend();
    let base = defend_preview(&st);
    let p = st.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut st, p, PowerKind::Stretch, 3);
    assert_eq!(defend_preview(&st), base + 3);
    // A preview spends nothing: the live state keeps its Stretch, unlatched.
    assert_eq!(st.player_power_amount(PowerKind::Stretch), 3);
    assert!(st
        .player
        .fool
        .as_ref()
        .is_none_or(|f| f.stretch_latch.is_none()));
}

#[test]
fn stretch_adds_nothing_to_a_preview_while_a_play_is_latched() {
    let mut st = state_with_a_defend();
    let base = defend_preview(&st);
    let p = st.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut st, p, PowerKind::Stretch, 3);
    let other = st.player.hand[0];
    // Some other play holds the latch (uid of a card that is not previewed).
    let f = sts2_core::fool::FoolState {
        stretch_latch: Some((sts2_core::CardUid::new(999).expect("nz"), 3, false)),
        ..Default::default()
    };
    assert_ne!(f.stretch_latch.map(|l| l.0), Some(other));
    st.player.fool = Some(Box::new(f));
    assert_eq!(defend_preview(&st), base);
}
