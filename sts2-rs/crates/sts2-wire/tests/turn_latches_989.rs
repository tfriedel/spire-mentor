//! GH #989: per-turn facts C# reads from History survive a rebuilt board.
//!
//! Ripple Basin blocks 4 at turn end unless its owner played an Attack this
//! turn. C# scans History; the engine counts the Attacks on the relic's
//! `counter`, which the wire derives from the seat's
//! `cards_played_this_turn`, ours and a teammate's.
//!
//! Hardened Shell caps the HP its owner loses each turn; C# shows the rest as
//! `DisplayAmount`, from which the wire derives the damage taken so far.

use serde_json::json;
use sts2_core::{apply, Action};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> serde_json::Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

/// The fixture's Ironclad (no block) against a Nibbit attacking for 9, with
/// an Ironclad teammate seat, as a whole co-op fight. Both hold Ripple Basin
/// and have played `ours` / `theirs` Attacks this turn.
fn board(ours: u32, theirs: u32) -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let basin = json!({"id": "RIPPLE_BASIN", "counter": null});
    v["player"]["block"] = json!(0);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([basin]);
    v["player"]["cards_played_this_turn"] = json!([ours, 0, 0, 0, 0]);
    v["deck"]["hand"] = json!([]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [],
        "draw_pile": (40..50).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [basin],
        "cards_played_this_turn": [theirs, 0, 0, 0, 0],
    }]);
    v
}

/// Our HP and the teammate's after the enemy turn.
fn hp_after_the_enemy_turn(ours: u32, theirs: u32) -> (i32, i32) {
    install();
    let mut st = state_from_json(&board(ours, theirs)).expect("builds");
    let _ = apply(&mut st, Action::EndTurn);
    (st.player.current_hp, st.allies[0].current_hp)
}

#[test]
fn ripple_basin_blocks_only_after_a_turn_without_attacks() {
    let (us_idle, them_idle) = hp_after_the_enemy_turn(0, 0);
    let (us_attacked, them_attacked) = hp_after_the_enemy_turn(1, 1);
    assert_eq!(us_idle - us_attacked, 4, "our basin");
    assert_eq!(them_idle - them_attacked, 4, "the teammate's basin");
}

#[test]
fn a_teammates_play_buckets_round_trip() {
    install();
    let st = state_from_json(&board(0, 2)).expect("builds");
    assert_eq!(st.allies[0].cards_played_this_turn, [2, 0, 0, 0, 0, 0]);
    let out = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    assert_eq!(
        out["allies"][0]["cards_played_this_turn"],
        json!([2, 0, 0, 0, 0])
    );
}

#[test]
fn a_teammates_juggling_counts_their_attacks_this_turn() {
    install();
    let mut v = board(0, 2);
    v["allies"][0]["powers"] = json!([{"name": "JUGGLING", "amount": 1}]);
    let st = state_from_json(&v).expect("builds");
    let juggling = st.coop.stash[0]
        .powers
        .iter()
        .filter_map(|pu| st.powers.get_by_id(pu.0))
        .find(|p| p.kind == sts2_core::PowerKind::Juggling)
        .expect("their Juggling");
    assert_eq!(juggling.counter, 2);
}

/// The Nibbit's HP lost to the fixture's Strike, behind a Hardened Shell of
/// 20 that shows `display` HP still allowed this turn.
fn nibbit_hp_lost_behind_a_shell(display: i32) -> i32 {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["deck"]["hand"] = json!([card("STRIKE_IRONCLAD", 4)]);
    v["encounter"][0]["powers"] =
        json!([{"name": "HARDENED_SHELL", "amount": 20, "display": display}]);
    let mut st = state_from_json(&v).expect("builds");
    let nibbit = st.enemy_uids[0];
    let hp = |s: &sts2_core::CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == nibbit.0)
            .expect("nibbit")
            .current_hp
    };
    let before = hp(&st);
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    before - hp(&st)
}

#[test]
fn a_hardened_shell_nearly_spent_this_turn_lets_only_the_rest_through() {
    assert_eq!(nibbit_hp_lost_behind_a_shell(3), 3);
    assert_eq!(
        nibbit_hp_lost_behind_a_shell(20),
        8,
        "Strike 6 + Strength 2"
    );
}
