//! The graft setters (`sts2_wire::graft`, the `RunManager.debug_*` Python
//! surface the Instinct advisor parks a rebuilt run with): each one fed the
//! output of its matching getter on a REAL run standing at that decision
//! must reproduce the decision — the phase, the action list, the offered
//! ids — and the pick must land the same way.
//!
//! The real runs are walked with HP pinned high and a seeded random policy
//! until the wanted phase comes up; the "graft" side is a clone of that run
//! knocked back to the map (`debug_place_on_map` at its own node) before the
//! setter re-opens the decision.

use serde_json::{json, Value};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::GameVersion;
use sts2_wire::emit::run_manager as emit;
use sts2_wire::graft;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
}

/// A v0.111.0 Ironclad run built the way the e2e campaign / `sts2_rs.
/// RunManager(mode="agent", act1="roll", interactive_potion_swap=True)`
/// builds it.
fn new_run(seed: u32, ascension: u8) -> RunManager {
    install();
    let rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        seed as i32,
        CharacterId::Ironclad,
        ascension,
    )
    .with_game_version(GameVersion::V0_111_0)
    .with_run_seed(u64::from(seed))
    .with_neow_start();
    let mut rm = rm;
    rm.run_state.interactive_potion_swap = true;
    rm.run_state.set_agent_decisions();
    let mut rm = rm.with_rolled_act1();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    rm
}

struct Lcg(u64);
impl Lcg {
    fn pick(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % n.max(1)
    }
}

/// Walk seeded random runs (HP pinned out of combat) until `want` holds;
/// returns that run. Combat plays the first legal action, so fights end.
fn walk_until(want: impl Fn(&RunManager) -> bool) -> RunManager {
    for seed in 1..=12_u32 {
        let mut rm = new_run(seed, 0);
        let mut rng = Lcg(u64::from(seed));
        for _ in 0..20_000 {
            if rm.is_over() {
                break;
            }
            if rm.phase() != Phase::Combat {
                rm.debug_set_player(Some(900), Some(900), None);
            }
            rm.bind_pending_event();
            if want(&rm) {
                return rm;
            }
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = if rm.phase() == Phase::Combat {
                0
            } else {
                rng.pick(actions.len())
            };
            let _ = rm.take_action(actions[idx]);
        }
    }
    panic!("no walked run reached the wanted decision");
}

/// `real` knocked back to the map at its own node: the base a graft starts
/// from (same deck, relics, RNG, map position — but no decision open).
fn knocked_back(real: &RunManager) -> RunManager {
    let mut rm = real.clone();
    let room = rm.run_state.current_room.expect("a room");
    let coord = room.map_coord.expect("a map node");
    let visited = rm.run_state.visited_map_coords.clone();
    let (act, af, tf) = (
        rm.run_state.act_index,
        rm.run_state.act_floor,
        rm.run_state.total_floor,
    );
    rm.debug_place_on_map(act, coord, visited, af, tf)
        .expect("placed on its own node");
    assert_eq!(rm.phase(), Phase::MapChoice);
    rm
}

fn card_offer_value(rm: &RunManager) -> Value {
    let ids = emit::card_reward_option_ids(rm);
    let up = emit::card_reward_option_upgraded(rm);
    let en = emit::card_reward_option_enchantments(rm);
    Value::Array(
        ids.iter()
            .enumerate()
            .map(|(i, id)| {
                json!({
                    "id": id,
                    "upgraded": up.get(i).copied().unwrap_or(false),
                    "enchantment": en.get(i).cloned().flatten(),
                })
            })
            .collect(),
    )
}

fn deck_view(rm: &RunManager) -> (Vec<(String, bool)>, Vec<Option<(String, i64)>>) {
    (
        emit::master_deck_detailed(rm),
        emit::master_deck_enchantments(rm),
    )
}

#[test]
fn master_deck_round_trips_with_enchantments() {
    let mut real = new_run(3, 0);
    real.run_state.master_deck[0].set_enchantment(sts2_core::cards::EnchantmentId::Goopy, 2);
    real.run_state.master_deck[1].upgraded = true;
    let deck = deck_view(&real);
    let value = Value::Array(
        deck.0
            .iter()
            .zip(&deck.1)
            .map(|((id, up), en)| json!({"id": id, "upgraded": up, "enchantment": en}))
            .collect(),
    );
    let mut rm = new_run(3, 0);
    rm.run_state.master_deck.clear();
    let missing = graft::set_master_deck(&mut rm, &value).expect("parses");
    assert!(missing.is_empty(), "{missing:?}");
    assert_eq!(deck_view(&rm), deck);
    // The upgraded card is built upgraded (the spec's upgraded stats), not a
    // flag flipped on the base card.
    let built = sts2_core::build_card_for_version(
        rm.run_state.master_deck[1].card_id,
        true,
        GameVersion::V0_111_0,
    )
    .expect("spec");
    assert_eq!(rm.run_state.master_deck[1], built);
    // An id the engine cannot build is reported, not fatal.
    let missing =
        graft::set_master_deck(&mut rm, &json!(["BASH", "NOT_A_CARD", "BASH+1"])).expect("ok");
    assert_eq!(missing, vec!["NOT_A_CARD".to_owned()]);
    assert_eq!(rm.run_state.master_deck.len(), 2);
}

#[test]
fn room_type_round_trips_through_current_room() {
    let mut rm = new_run(3, 0);
    for rt in [
        "monster",
        "elite",
        "boss",
        "event",
        "shop",
        "rest_site",
        "treasure",
    ] {
        graft::set_room_type(&mut rm, rt).expect("known");
        assert_eq!(emit::current_room(&rm)["room_type"], json!(rt));
    }
    assert!(graft::set_room_type(&mut rm, "lounge").is_err());
}

#[test]
fn card_reward_round_trips_and_the_pick_lands_the_same() {
    let real = walk_until(|rm| {
        rm.phase() == Phase::CardReward
            && !rm.card_reward_options().is_empty()
            && rm.card_reward_bundles().is_none()
            && !rm.card_offer_choose_screen()
    });
    let mut rm = knocked_back(&real);
    graft::open_card_offer(
        &mut rm,
        &card_offer_value(&real),
        real.card_offer_can_skip(),
        real.card_offer_choose_screen(),
        None,
    )
    .expect("opens");
    assert_eq!(rm.phase(), Phase::CardReward);
    assert_eq!(rm.get_available_actions(), real.get_available_actions());
    assert_eq!(
        emit::card_reward_option_ids(&rm),
        emit::card_reward_option_ids(&real)
    );
    let (mut a, mut b) = (real.clone(), rm.clone());
    let _ = a.take_action(RunAction::PickCardReward { idx: Some(0) });
    let _ = b.take_action(RunAction::PickCardReward { idx: Some(0) });
    assert_eq!(deck_view(&b), deck_view(&a));
    // Skip and a forced pick.
    graft::open_card_offer(&mut rm, &card_offer_value(&real), false, true, None).expect("opens");
    assert!(rm.card_offer_choose_screen());
    assert!(!rm
        .get_available_actions()
        .contains(&RunAction::PickCardReward { idx: None }));
}

#[test]
fn card_offer_carries_enchantments_and_bundles() {
    let mut rm = new_run(5, 0);
    graft::open_card_offer(
        &mut rm,
        &json!([{"id": "CARD.BASH", "upgraded": true, "enchantment": "GLAM", "enchantment_amount": 1},
                "ANGER"]),
        true,
        false,
        None,
    )
    .expect("opens");
    assert_eq!(
        emit::card_reward_option_enchantments(&rm),
        vec![Some(("GLAM".to_owned(), 1)), None]
    );
    assert_eq!(emit::card_reward_option_upgraded(&rm), vec![true, false]);
    let deck_len = rm.run_state.master_deck.len();
    let _ = rm.take_action(RunAction::PickCardReward { idx: Some(0) });
    let picked = rm.run_state.master_deck.last().expect("added");
    assert_eq!(rm.run_state.master_deck.len(), deck_len + 1);
    assert!(picked.upgraded);
    assert_eq!(picked.enchantment(sts2_core::cards::EnchantmentId::Glam), 1);

    // A bundle pick, in `card_reward_bundles()`'s shape.
    let packs = json!([["CARD.BASH", "CARD.ANGER"], ["CARD.CLASH"]]);
    graft::open_card_offer(&mut rm, &json!([]), true, false, Some(&packs)).expect("opens");
    assert_eq!(
        emit::card_reward_bundles(&rm),
        Some(vec![
            vec!["CARD.BASH".to_owned(), "CARD.ANGER".to_owned()],
            vec!["CARD.CLASH".to_owned()],
        ])
    );
    assert_eq!(
        rm.get_available_actions(),
        vec![
            RunAction::PickCardReward { idx: Some(0) },
            RunAction::PickCardReward { idx: Some(1) }
        ]
    );
    let deck_len = rm.run_state.master_deck.len();
    let _ = rm.take_action(RunAction::PickCardReward { idx: Some(0) });
    assert_eq!(rm.run_state.master_deck.len(), deck_len + 2);
}

#[test]
fn event_round_trips_and_its_pages_replay() {
    let real = walk_until(|rm| {
        rm.phase() == Phase::Event
            && rm
                .run_state
                .current_event
                .is_some_and(|e| e != sts2_core::EventId::Neow)
    });
    let name = real.run_state.current_event.expect("bound").python_name();
    let mut rm = knocked_back(&real);
    // The C# `Id.Entry` spelling the mod sends.
    let mut entry = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            entry.push('_');
        }
        entry.push(ch.to_ascii_uppercase());
    }
    let visited_before = rm.run_state.acts[usize::from(rm.run_state.act_index) - 1].events_visited;
    assert_eq!(graft::enter_event(&mut rm, &entry).expect("known"), name);
    assert_eq!(rm.phase(), Phase::Event);
    assert_eq!(
        rm.run_state.acts[usize::from(rm.run_state.act_index) - 1].events_visited,
        visited_before,
        "a graft is not a pool draw"
    );
    assert_eq!(emit::rng_counters(&rm), emit::rng_counters(&real));
    assert_eq!(
        emit::event_option_infos(&rm),
        emit::event_option_infos(&real)
    );
    assert_eq!(rm.get_available_actions(), real.get_available_actions());
    // The next page (or the exit) is reached by replaying the choice.
    let first = real.get_available_actions()[0];
    let (mut a, mut b) = (real.clone(), rm.clone());
    let _ = a.take_action(first);
    let _ = b.take_action(first);
    assert_eq!(b.phase(), a.phase());
    assert_eq!(emit::event_option_infos(&b), emit::event_option_infos(&a));
}

#[test]
fn a_multi_page_event_replays_from_page_zero() {
    let mut rm = new_run(7, 0);
    graft::enter_event(&mut rm, "TINKER_TIME").expect("known");
    let page0 = emit::event_option_infos(&rm);
    assert!(page0.is_array());
    let _ = rm.take_action(rm.get_available_actions()[0]);
    assert_eq!(rm.phase(), Phase::Event, "Tinker Time has a second page");
    assert_ne!(emit::event_option_infos(&rm), page0);
    assert!(graft::enter_event(&mut rm, "NOT_AN_EVENT").is_err());
}

#[test]
fn rest_site_round_trips_and_smith_opens_the_pick() {
    let real =
        walk_until(|rm| rm.phase() == Phase::RestSite && rm.run_state.pending_choice.is_none());
    let ids: Vec<String> = emit::rest_site_options(&real)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let mut rm = knocked_back(&real);
    let offered = graft::enter_rest_site(&mut rm, &[]).expect("enters");
    assert_eq!(offered, ids);
    assert_eq!(rm.phase(), Phase::RestSite);
    assert_eq!(emit::rest_site_options(&rm), emit::rest_site_options(&real));
    assert_eq!(rm.get_available_actions(), real.get_available_actions());
    let smith = ids.iter().position(|i| i == "SMITH").expect("smith");
    let (mut a, mut b) = (real.clone(), rm.clone());
    let _ = a.take_action(RunAction::RestOption { idx: smith });
    let _ = b.take_action(RunAction::RestOption { idx: smith });
    assert_eq!(emit::smith_options(&b), emit::smith_options(&a));
    assert!(!emit::smith_options(&b).is_empty());
    assert_eq!(b.get_available_actions(), a.get_available_actions());

    // MiniatureTent's second pick: HEAL taken, leaving is legal.
    let left = graft::enter_rest_site(&mut rm, &["HEAL".to_owned()]).expect("enters");
    assert!(!left.contains(&"HEAL".to_owned()));
    assert!(rm.get_available_actions().contains(&RunAction::LeaveShop));
    assert!(graft::enter_rest_site(&mut rm, &["DIG".to_owned()]).is_err());
}

#[test]
fn shop_round_trips_with_discount_relics() {
    let mut real = walk_until(|rm| rm.phase() == Phase::Shop && rm.shop_inventory().is_some());
    // Courier + Membership: the displayed prices are 0.4× the stored ones.
    real.run_state.relics.extend([
        sts2_core::RelicId::TheCourier,
        sts2_core::RelicId::MembershipCard,
    ]);
    real.run_state.gold = 999;
    let inv = emit::shop_inventory(&real);
    let mut rm = knocked_back(&real);
    let unmatched = graft::open_shop(&mut rm, &inv).expect("opens");
    assert!(unmatched.is_empty(), "{unmatched:?}");
    assert_eq!(rm.phase(), Phase::Shop);
    assert_eq!(emit::shop_inventory(&rm), inv);
    assert_eq!(rm.get_available_actions(), real.get_available_actions());
    // The stored prices themselves round-trip (the smallest base reproducing
    // the displayed price; the rolled base can differ only where the floor
    // folds two bases together, which the displayed price cannot tell apart).
    let gold_before = rm.run_state.gold;
    let (mut a, mut b) = (real.clone(), rm.clone());
    let _ = a.take_action(RunAction::ShopBuyRelic { idx: 0 });
    let _ = b.take_action(RunAction::ShopBuyRelic { idx: 0 });
    assert_eq!(
        gold_before - b.run_state.gold,
        gold_before - a.run_state.gold
    );
}

#[test]
fn boss_relic_round_trips_from_a_placed_boss_node() {
    let real = walk_until(|rm| rm.phase() == Phase::BossRelic);
    let offered = emit::boss_relic_options(&real);
    // Placing on the boss node is the graft of a boss fight (it used to be
    // refused: the boss point sits outside the room grid).
    let mut rm = knocked_back(&real);
    assert_eq!(
        rm.run_state.current_room.map(|r| r.room_type),
        Some(sts2_core::runstate::RoomType::Boss)
    );
    graft::open_boss_relic(&mut rm, &offered).expect("opens");
    assert_eq!(rm.phase(), Phase::BossRelic);
    assert_eq!(emit::boss_relic_options(&rm), offered);
    assert_eq!(rm.get_available_actions(), real.get_available_actions());
    let (mut a, mut b) = (real.clone(), rm.clone());
    let _ = a.take_action(RunAction::PickBossRelic { idx: 0 });
    let _ = b.take_action(RunAction::PickBossRelic { idx: 0 });
    assert_eq!(b.run_state.act_index, a.run_state.act_index);
    assert_eq!(b.run_state.act_index, 2);
    assert_eq!(b.run_state.relics, a.run_state.relics);
    assert_eq!(
        b.run_state.current_hp, a.run_state.current_hp,
        "no second heal"
    );
    assert_eq!(b.phase(), a.phase());
}

#[test]
fn deck_pick_queues_preempt_the_phase() {
    let mut rm = new_run(9, 0);
    let deck = rm.run_state.master_deck.len();
    rm.debug_queue_deck_removals(1);
    assert_eq!(
        rm.get_available_actions().len(),
        rm.run_state.removal_offer_order().len()
    );
    let _ = rm.take_action(RunAction::RemoveDeckCard { idx: 0 });
    assert_eq!(rm.run_state.master_deck.len(), deck - 1);

    rm.debug_queue_deck_upgrades(1);
    assert!(matches!(
        rm.get_available_actions()[0],
        RunAction::UpgradeDeckCard { .. }
    ));
    let _ = rm.take_action(RunAction::UpgradeDeckCard { idx: 0 });
    assert_eq!(rm.run_state.pending_deck_upgrades, 0);

    rm.debug_queue_deck_transforms(1, false);
    assert_eq!(
        rm.get_available_actions().len(),
        rm.transform_eligible_deck_indices().len()
    );
    let _ = rm.take_action(RunAction::TransformDeckCard { idx: 0 });
    assert!(rm.run_state.pending_deck_transforms.is_empty());

    graft::queue_deck_enchant(&mut rm, "ENCHANTMENT.SHARP", 1, false, 1, Some("attack"))
        .expect("known");
    let eligible = rm.enchant_eligible_deck_indices();
    assert!(!eligible.is_empty());
    assert!(eligible
        .iter()
        .all(|&i| rm.run_state.master_deck[i].card_type == sts2_core::cards::CardType::Attack));
    let _ = rm.take_action(RunAction::EnchantDeckCard { idx: 0 });
    assert_eq!(
        rm.run_state.master_deck[eligible[0]].enchantment(sts2_core::cards::EnchantmentId::Sharp),
        1
    );

    graft::queue_card_grant(
        &mut rm,
        &json!(["CARD.ANGER", "CARD.CLASH+1", "CARD.BASH"]),
        2,
        2,
    )
    .expect("builds");
    assert_eq!(
        emit::card_grant_options(&rm),
        vec![
            ("Anger".to_owned(), false),
            ("Clash".to_owned(), true),
            ("Bash".to_owned(), false)
        ]
    );
    assert_eq!(rm.get_available_actions().len(), 3);

    let mut rm = new_run(9, 0);
    rm.run_state.potions = vec![sts2_core::potions::PotionKind::FirePotion; 3];
    graft::queue_potion_swap(&mut rm, "POTION.BLOCK_POTION").expect("known");
    assert_eq!(rm.get_available_actions().len(), 4);
    assert_eq!(
        rm.get_available_actions()[0],
        RunAction::PotionSwap { discard_slot: None }
    );
}
