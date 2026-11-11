//! Replays run the card's enchantment `OnPlay` (GH #1021), and Momentum
//! keeps its own stash apart from Rampage's growth.
//!
//! C# `CardModel.OnPlayWrapper` (v0.107.1 `CardModel.cs:1905-1965`) loops
//! `playCount` times; every iteration runs the card's `OnPlay`, then
//! `Enchantment.OnPlay`, then `Hook.AfterCardPlayed`. So an Adroit card
//! replayed by OneTwoPunch / EchoForm / a base replay gains its block on every
//! play, Corrupted deals its 2 self-damage on every play, and Momentum ramps
//! on every play. Rust ran the enchantment `OnPlay` on the first play only.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardTag, CardTags, CardType, EnchantmentId, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid,
};

const ENEMY_HP: i32 = 9999;

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: ENEMY_HP,
        max_hp: ENEMY_HP,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 9;
    state.player.max_energy = 9;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn next_uid(state: &CombatState) -> sts2_core::CardUid {
    sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("u32")).expect("nz")
}

/// A 5-block Defend carrying `ench` (amount `amount`), replayed `replays`
/// extra times, pushed into the hand.
fn push_defend(state: &mut CombatState, ench: EnchantmentId, amount: i32, replays: i32) {
    let uid = next_uid(state);
    let mut card = Card {
        uid,
        card_id: CardId::DefendIronclad,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        base_block: 5,
        tags: CardTags::NONE.with(CardTag::Defend),
        base_replay_count: replays,
        ..Card::pod_defaults()
    };
    card.set_enchantment(ench, amount);
    state.cards.push(card);
    state.player.hand.push(uid);
}

/// A 6-damage Strike carrying `ench`, replayed `replays` extra times.
fn push_strike(state: &mut CombatState, ench: EnchantmentId, amount: i32, replays: i32) {
    let uid = next_uid(state);
    let mut card = Card {
        uid,
        card_id: CardId::StrikeIronclad,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        base_damage: 6,
        tags: CardTags::NONE.with(CardTag::Strike),
        base_replay_count: replays,
        ..Card::pod_defaults()
    };
    card.set_enchantment(ench, amount);
    state.cards.push(card);
    state.player.hand.push(uid);
}

fn add_echo_form(state: &mut CombatState) {
    let pid = NonZeroU32::new(7700).expect("nz");
    state.powers.push(Power {
        id: pid,
        kind: PowerKind::EchoForm,
        amount: 1,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(PowerUid(pid));
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

#[test]
fn adroit_without_a_replay_gains_its_block_once() {
    let mut state = fresh();
    push_defend(&mut state, EnchantmentId::Adroit, 3, 0);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 8, "5 + Adroit 3");
}

#[test]
fn a_replayed_adroit_card_gains_its_block_per_play() {
    let mut state = fresh();
    push_defend(&mut state, EnchantmentId::Adroit, 3, 1);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 16, "(5 + Adroit 3) x 2 plays");
}

#[test]
fn an_echoed_adroit_card_gains_its_block_per_play() {
    let mut state = fresh();
    add_echo_form(&mut state);
    push_defend(&mut state, EnchantmentId::Adroit, 3, 0);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(
        state.player.block, 16,
        "(5 + Adroit 3) x 2 plays under EchoForm"
    );
}

#[test]
fn a_replayed_corrupted_card_hurts_its_owner_per_play() {
    let mut state = fresh();
    push_strike(&mut state, EnchantmentId::Corrupted, 1, 1);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.player.current_hp, 76, "2 self-damage x 2 plays");
    assert_eq!(enemy_hp(&state), ENEMY_HP - 18, "6 x 1.5 = 9, twice");
}

#[test]
fn a_replayed_momentum_card_ramps_per_play() {
    let mut state = fresh();
    push_strike(&mut state, EnchantmentId::Momentum, 2, 2);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    // Plays deal 6, 8, 10: each OnPlay adds 2 after its hit.
    assert_eq!(enemy_hp(&state), ENEMY_HP - 24);
    assert_eq!(
        state.cards[0].enchantment(EnchantmentId::Momentum),
        2,
        "the amount itself does not change"
    );
}

/// An echoed Goopy card grows once per play (GH #952) — Echo never ran the
/// enchantment's `AfterCardPlayed`.
#[test]
fn an_echoed_goopy_card_grows_per_play() {
    let mut state = fresh();
    add_echo_form(&mut state);
    push_defend(&mut state, EnchantmentId::Goopy, 1, 0);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 5 + 6, "play 1: 5 + 0; play 2: 5 + 1");
    assert_eq!(state.cards[0].enchantment(EnchantmentId::Goopy), 3);
}

/// Momentum on Rampage: C# keeps Rampage's growth in `Damage.BaseValue`
/// and Momentum's in `Momentum._extraDamage`, so the second play deals
/// (9 + 5) + 3 = 17. Sharing one slot made it 9 + 8 + 8 = 25.
#[test]
fn momentum_on_rampage_counts_each_growth_once() {
    let mut state = fresh();
    let uid = next_uid(&state);
    let mut card =
        sts2_content::cards::spec_index::card_pod_from_id(CardId::Rampage, false).expect("Rampage");
    card.uid = uid;
    card.cost = sts2_core::CardEnergyCost::new(0);
    card.set_enchantment(EnchantmentId::Momentum, 3);
    state.cards.push(card);
    state.player.hand.push(uid);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(enemy_hp(&state), ENEMY_HP - 9, "play 1: base 9");

    state.player.discard.retain(|u| *u != uid);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(
        enemy_hp(&state),
        ENEMY_HP - 9 - 17,
        "play 2: 9 + Rampage 5 + Momentum 3"
    );
}
