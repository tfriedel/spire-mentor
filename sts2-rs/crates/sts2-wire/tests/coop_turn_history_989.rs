//! GH #989: a seat's turn so far, which a rebuilt board cannot recount, rides
//! the wire. `cards_exhausted_this_turn` is what C# `EvilEye`'s
//! `WasCardExhaustedThisTurn` scans History for (a Toasty Mittens exhaust at
//! turn start doubles a later Evil Eye), and `card_block_gains_this_turn` is
//! what `UnmovablePower.ModifyBlockMultiplicative` counts before it doubles a
//! card's block (a second Defend in the turn is not doubled).

use serde_json::json;
use sts2_core::{apply, Action, CombatState};
use sts2_wire::emit::combat_state_to_value;
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> serde_json::Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

/// The fixture's Ironclad (block 0) holding `card`, with an Ironclad teammate
/// holding the same, as a whole co-op fight; each seat's `extra` merged into
/// its `$.player` / `$.allies[0]`.
fn coop_board(card_id: &str, ours: &serde_json::Value, theirs: &serde_json::Value) -> CombatState {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["player"]["block"] = json!(0);
    v["player"]["powers"] = json!([]);
    v["deck"]["hand"] = json!([card(card_id, 4)]);
    v["coop_full"] = json!(true);
    let mut ally = json!({
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [card(card_id, 30)],
        "draw_pile": (40..45).map(|i| card("STRIKE_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [], "relics": [],
    });
    for (seat, extra) in [(&mut v["player"], ours), (&mut ally, theirs)] {
        for (k, val) in extra.as_object().expect("an object") {
            seat[k] = val.clone();
        }
    }
    v["allies"] = json!([ally]);
    state_from_json(&v).expect("builds")
}

fn play(st: &mut CombatState) {
    let _ = apply(
        st,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
}

fn play_as_teammate(st: &mut CombatState) {
    let _ = sts2_core::coop::apply_as(
        st,
        0,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
}

#[test]
fn a_card_exhausted_earlier_this_turn_doubles_evil_eye() {
    let mut st = coop_board("EVIL_EYE", &json!({}), &json!({}));
    play(&mut st);
    assert_eq!(st.player.block, 8, "nothing exhausted yet");

    let mut st = coop_board(
        "EVIL_EYE",
        &json!({"cards_exhausted_this_turn": 1}),
        &json!({}),
    );
    play(&mut st);
    assert_eq!(st.player.block, 16);

    let mut st = coop_board(
        "EVIL_EYE",
        &json!({}),
        &json!({"cards_exhausted_this_turn": 1}),
    );
    play_as_teammate(&mut st);
    assert_eq!(st.allies[0].block, 16, "the teammate's own exhaust");
}

#[test]
fn unmovable_doubles_only_the_turns_first_card_block() {
    let unmovable = |gains: u32| {
        let mut seat = json!({"powers": [{"name": "UNMOVABLE", "amount": 1}]});
        if gains > 0 {
            seat["card_block_gains_this_turn"] = json!(gains);
        }
        seat
    };
    let mut st = coop_board("DEFEND_IRONCLAD", &unmovable(0), &json!({}));
    play(&mut st);
    assert_eq!(st.player.block, 10, "the turn's first card block doubles");

    let mut st = coop_board("DEFEND_IRONCLAD", &unmovable(1), &json!({}));
    play(&mut st);
    assert_eq!(st.player.block, 5, "a Defend earlier this turn spent it");

    let mut st = coop_board("DEFEND_IRONCLAD", &json!({}), &unmovable(1));
    play_as_teammate(&mut st);
    assert_eq!(st.allies[0].block, 5, "the teammate's Unmovable too");
}

#[test]
fn a_seats_turn_so_far_round_trips() {
    let seat = json!({"cards_exhausted_this_turn": 2, "card_block_gains_this_turn": 1});
    let st = coop_board("DEFEND_IRONCLAD", &seat, &seat);
    let back = combat_state_to_value(&st).expect("emits");
    for path in ["/player", "/allies/0"] {
        let s = back.pointer(path).expect("seat");
        assert_eq!(s["cards_exhausted_this_turn"], json!(2), "{path}");
        assert_eq!(s["card_block_gains_this_turn"], json!(1), "{path}");
    }
}

/// A board emitted mid-play sends only the earlier plays' block gains: the
/// resolving play's own gains carry its card's uid as their tag, and seeded
/// back they would read as an earlier play's (PR #1023 review).
#[test]
fn a_resolving_plays_own_block_gains_are_not_emitted() {
    let mut st = coop_board(
        "DEFEND_IRONCLAD",
        &json!({"card_block_gains_this_turn": 1}),
        &json!({}),
    );
    st.combat_log.unmovable_card_block_plays.push(1);
    let back = combat_state_to_value(&st).expect("emits");
    assert_eq!(back["player"]["card_block_gains_this_turn"], json!(1));
    let rebuilt = state_from_json(&back).expect("re-parses");
    assert_eq!(rebuilt.combat_log.unmovable_card_block_plays.len(), 1);
}
