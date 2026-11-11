//! Smoke tests for the `sts2-rs-5om` pile-search + card-spawn handlers.
//!
//! Exercises one card per target pile-kind (hand / discard / exhaust) plus
//! a hand-search handler. Confirms the handler runs without panicking and
//! that the expected spawn / search side-effect lands.

use std::num::NonZeroU32;

use smallvec::SmallVec;
use sts2_core::card_effects::{dispatch, EffectResult};
use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
use sts2_core::{
    CardUid, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
};

use sts2_content::cards::dispatch as content_dispatch;

fn fresh_state_with_enemy() -> (CombatState, CreatureUid) {
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(0_i32),
        ..Default::default()
    };
    let enemy_id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id: enemy_id,
        current_hp: 100,
        max_hp: 100,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    let uid = CreatureUid(enemy_id);
    state.enemy_uids.push(uid);
    (state, uid)
}

/// Push a played card onto state.cards + state.player.play so handlers
/// that read `current_card_damage` / `current_card_block` find their POD.
fn push_play(state: &mut CombatState, card_id: CardId, damage: i32, block: i32) -> CardUid {
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("u32")).expect("nz");
    state.cards.push(Card {
        uid,
        card_id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Skill,
        target_type: TargetType::None,
        upgraded: false,
        base_damage: damage,
        base_block: block,
        combat_vars: sts2_core::cards::CombatVars::new(),
        enchantments: sts2_core::cards::CardEnchantments::new(),
        tags: CardTags::NONE,
        energy_spent: 0,
        base_replay_count: 0,
        keyword_override: None,
        affliction: None,
    });
    state.player.play.push(uid);
    uid
}

/// `HiddenDaggers`: spawns 2 Shivs into hand (Hand pile-kind).
#[test]
fn hidden_daggers_spawns_shivs_into_hand() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = push_play(&mut state, CardId::HiddenDaggers, 0, 0);
    let pre_hand = state.player.hand.len();
    let res = dispatch(&mut state, CardId::HiddenDaggers, None);
    assert_eq!(res, EffectResult::Resolved);
    assert_eq!(state.player.hand.len(), pre_hand + 2);
    // Confirm both new entries are Shivs.
    for uid in state.player.hand.iter().skip(pre_hand) {
        let idx = uid.0.get() as usize - 1;
        assert_eq!(state.cards[idx].card_id, CardId::Shiv);
    }
}

/// `GunkUp`: deals damage + spawns a Slimed into Discard pile.
#[test]
fn gunk_up_spawns_slimed_into_discard() {
    content_dispatch::ensure_installed();
    let (mut state, e) = fresh_state_with_enemy();
    let _ = push_play(&mut state, CardId::GunkUp, 4, 0);
    let pre_discard = state.player.discard.len();
    let res = dispatch(&mut state, CardId::GunkUp, Some(e));
    assert_eq!(res, EffectResult::Resolved);
    assert_eq!(state.player.discard.len(), pre_discard + 1);
    let last = *state.player.discard.last().expect("Slimed spawned");
    let idx = last.0.get() as usize - 1;
    assert_eq!(state.cards[idx].card_id, CardId::Slimed);
}

/// Synthesis (C# `Synthesis.cs`): deals 12 damage + applies 1 FreePower
/// to self. It must NOT touch the exhaust pile (the earlier port
/// fabricated an exhaust-retrieval effect; bead `sts2-rs-m76f8`).
#[test]
fn synthesis_deals_damage_and_applies_free_power() {
    content_dispatch::ensure_installed();
    let (mut state, e) = fresh_state_with_enemy();
    // Seed exhaust pile with a card that must STAY exhausted.
    let exhausted_uid = {
        let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("u32")).expect("nz");
        state.cards.push(Card {
            uid,
            card_id: CardId::Wound,
            cost: sts2_core::CardEnergyCost::new(-1),
            card_type: CardType::Status,
            target_type: TargetType::None,
            upgraded: false,
            base_damage: 0,
            base_block: 0,
            ..Card::pod_defaults()
        });
        state.player.exhaust.push(uid);
        uid
    };
    let _ = push_play(&mut state, CardId::Synthesis, 12, 0);
    let pre_hand = state.player.hand.len();
    let res = dispatch(&mut state, CardId::Synthesis, Some(e));
    assert_eq!(res, EffectResult::Resolved);
    // Exhaust pile untouched; hand unchanged.
    assert!(state.player.exhaust.contains(&exhausted_uid));
    assert_eq!(state.player.hand.len(), pre_hand);
    // FreePower applied to self.
    assert!(state
        .powers
        .iter()
        .any(|p| p.kind == sts2_core::PowerKind::FreePower && p.amount == 1));
}

/// Storm of Steel: discards every hand card, then spawns one Shiv per
/// discarded. Covers `PileKind::Hand` spawn + discard interaction.
#[test]
fn storm_of_steel_replaces_hand_with_shivs() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    // Seed hand with 2 generic cards.
    for _ in 0..2 {
        let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("u32")).expect("nz");
        state.cards.push(Card {
            uid,
            card_id: CardId::StrikeSilent,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            ..Card::pod_defaults()
        });
        state.player.hand.push(uid);
    }
    let _ = push_play(&mut state, CardId::StormOfSteel, 0, 0);
    let res = dispatch(&mut state, CardId::StormOfSteel, None);
    assert_eq!(res, EffectResult::Resolved);
    // 2 cards were discarded → 2 Shivs in hand.
    let shivs = state
        .player
        .hand
        .iter()
        .filter(|u| {
            let idx = u.0.get() as usize - 1;
            state.cards[idx].card_id == CardId::Shiv
        })
        .count();
    assert_eq!(shivs, 2);
}
