//! Midnight's `AfterCardEnteredCombat` (a Midnight that enters combat late
//! starts cheaper by every exhaust so far) and how two Guarded instances
//! stack, on co-op fixture boards.

use serde_json::{json, Value};
use sts2_core::cards::CardId;
use sts2_core::hooks::CreatureRef;
use sts2_core::hooks::ValueProps;
use sts2_core::{apply, Action, CombatState};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

fn ally(hand: Value, stable: u32) -> Value {
    json!({
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": false,
        "hand": hand,
        "draw_pile": (stable..stable + 10).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    })
}

/// The v0.111.0 Ironclad fixture as a co-op fight: our hand, and one Ironclad
/// teammate per entry of `allies`.
fn board(hand: Value, allies: Vec<Value>) -> Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["block"] = json!(0);
    v["player"]["energy"] = json!(3);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = hand;
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["encounter"][0]["block"] = json!(0);
    v["coop_full"] = json!(true);
    v["allies"] = Value::Array(allies);
    v
}

fn build(v: &Value) -> CombatState {
    install();
    state_from_json(v).expect("builds")
}

fn spawn_midnight(st: &mut CombatState) -> i8 {
    let c = sts2_content::cards::spec_index::card_pod_from_id(CardId::Midnight, false)
        .expect("Midnight");
    let uid = sts2_core::card_effects::spawn_card_into_hand(st, c);
    i8::try_from(st.cards[uid.0.get() as usize - 1].local_cost()).expect("fits")
}

#[test]
fn a_new_midnight_starts_cheaper_by_everyones_exhausts() {
    // C# counts every `CardExhaustedEntry`, whoever exhausted.
    let mut v = board(
        json!([card("DEFEND_IRONCLAD", 1)]),
        vec![ally(json!([]), 40)],
    );
    v["player"]["cards_exhausted_this_combat"] = json!(2);
    v["allies"][0]["cards_exhausted_this_combat"] = json!(3);
    let mut st = build(&v);
    assert_eq!(spawn_midnight(&mut st), 12 - 5);
}

#[test]
fn an_exhaust_in_the_engine_counts_for_a_later_midnight() {
    let v = board(json!([card("IMPERVIOUS", 1)]), vec![ally(json!([]), 40)]);
    let mut st = build(&v);
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(sts2_core::coop::cards_exhausted_this_combat(&st), 1);
    assert_eq!(spawn_midnight(&mut st), 11);
}

#[test]
fn a_copy_of_a_midnight_keeps_its_cost_and_is_not_shaved_again() {
    // C# `CreateClone` keeps the cost modifiers and Midnight skips the
    // catch-up for a clone (`if (base.IsClone) return`).
    let mut midnight = card("MIDNIGHT", 2);
    midnight["cost_override"] = json!(8);
    let mut v = board(
        json!([card("DUAL_WIELD", 1), midnight]),
        vec![ally(json!([]), 40)],
    );
    v["player"]["cards_exhausted_this_combat"] = json!(4);
    let mut st = build(&v);
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    let costs: Vec<i8> = st
        .player
        .hand
        .iter()
        .map(|u| &st.cards[u.0.get() as usize - 1])
        .filter(|c| c.card_id == CardId::Midnight)
        .map(|c| i8::try_from(c.local_cost()).expect("fits"))
        .collect();
    assert_eq!(costs, vec![8, 8]);
}

#[test]
fn the_exhaust_count_survives_a_round_trip() {
    let mut v = board(
        json!([card("DEFEND_IRONCLAD", 1)]),
        vec![ally(json!([]), 40)],
    );
    v["player"]["cards_exhausted_this_combat"] = json!(2);
    v["allies"][0]["cards_exhausted_this_combat"] = json!(3);
    let out = sts2_wire::emit::combat_state_to_value(&build(&v)).expect("emits");
    assert_eq!(out["player"]["cards_exhausted_this_combat"], json!(2));
    assert_eq!(out["allies"][0]["cards_exhausted_this_combat"], json!(3));
}

#[test]
fn two_tanks_guard_a_third_player_twice() {
    // Guarded is instanced (one per Tank) and each halves a powered hit:
    // C# multiplies every listener's `ModifyDamageMultiplicative`.
    let v = board(
        json!([card("TANK_CARD", 1)]),
        vec![
            ally(json!([card("TANK_CARD", 30)]), 40),
            ally(json!([]), 60),
        ],
    );
    let mut st = build(&v);
    let play = Action::PlayCard {
        hand_index: 0,
        target: None,
    };
    let _ = apply(&mut st, play.clone());
    let _ = sts2_core::coop::apply_as(&mut st, 0, play);
    let enemy = CreatureRef::Enemy(st.enemy_uids[0]);
    let mut third = 0;
    sts2_core::coop::with_seat(&mut st, 2, |s| {
        third =
            sts2_core::combat_loop::preview_damage_to_player(s, enemy, 20, ValueProps::MOVE, None);
    });
    assert_eq!(third, 5, "20 x 0.5 x 0.5");
    // Each Tank is guarded by the other: 20 x 1.5 x 0.5.
    let ours =
        sts2_core::combat_loop::preview_damage_to_player(&st, enemy, 20, ValueProps::MOVE, None);
    assert_eq!(ours, 15);
}
