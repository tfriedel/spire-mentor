//! Regression: Spiral / Glam enchantments replay the card (GH #806).
//!
//! C# `CardModel.GeneratePlayCount` (v0.107.1 `CardModel.cs:2016-2021`) seeds
//! the play count with `GetEnchantedReplayCount() + 1`, i.e.
//! `Enchantment.EnchantPlayCount(BaseReplayCount)` — `Spiral.cs:30-33` adds
//! `Times` (1) always, `Glam.cs:34-41` adds it until its `AfterCardPlayed`
//! latches `UsedThisCombat` (`Glam.cs:44-56`). Rust never folded the
//! enchantment bonus into the play count, so a Glam card (Silken Tress
//! enchants the first card reward with it) hit once instead of twice.
//!
//! Surfaced by the whole-run sweep, PARITY149 (Necrobinder / Regent /
//! Ironclad all took Silken Tress at Neow): `enemies_hp` cs=[33] rs=[42] after
//! a Glam SCULPTING_STRIKE.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, EnchantmentId, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
};

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
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
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 9;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// Push a 6-damage Strike carrying `enchantment` into the hand.
fn push_strike(state: &mut CombatState, enchantment: Option<EnchantmentId>) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = Card {
        uid,
        card_id: CardId::StrikeIronclad,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 6,
        base_block: 0,
        ..Card::pod_defaults()
    };
    if let Some(e) = enchantment {
        card.set_enchantment(e, 1);
    }
    state.cards.push(card);
    state.player.hand.push(uid);
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

#[test]
fn plain_strike_hits_once() {
    let mut state = fresh();
    push_strike(&mut state, None);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(enemy_hp(&state), 94);
}

/// Glam: the first play this combat is replayed (6 x2); the second play of the
/// same card is not (Glam is spent).
#[test]
fn glam_replays_first_play_only() {
    let mut state = fresh();
    push_strike(&mut state, Some(EnchantmentId::Glam));
    let uid = state.player.hand[0];
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(enemy_hp(&state), 88, "Glam replays the first play: 6 x2");

    // Return the same card to hand and play it again: Glam is used up.
    state.player.discard.retain(|u| *u != uid);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(
        enemy_hp(&state),
        82,
        "spent Glam: the second play hits once"
    );
}

/// Spiral replays every play.
#[test]
fn spiral_replays_every_play() {
    let mut state = fresh();
    push_strike(&mut state, Some(EnchantmentId::Spiral));
    let uid = state.player.hand[0];
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(enemy_hp(&state), 88, "Spiral: 6 x2");
    state.player.discard.retain(|u| *u != uid);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(enemy_hp(&state), 76, "Spiral replays again: 6 x2");
}

/// The PARITY149 shape: a Glam SCULPTING_STRIKE (9 damage, then "add Ethereal
/// to a card in your Hand") whose first play SURFACES its hand select. The
/// replay must survive the surfaced choice: 9 x2 damage in total.
#[test]
fn glam_sculpting_strike_replays_across_surfaced_select() {
    let mut state = fresh();
    state.side = sts2_core::Side::Player;
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = Card {
        uid,
        card_id: CardId::SculptingStrike,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 9,
        base_block: 0,
        ..Card::pod_defaults()
    };
    card.set_enchantment(EnchantmentId::Glam, 1);
    state.cards.push(card);
    state.player.hand.push(uid);
    push_strike(&mut state, None);
    push_strike(&mut state, None);
    push_strike(&mut state, None);

    let mut step = sts2_core::apply(
        &mut state,
        sts2_core::Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let mut selects = 0;
    while step == sts2_core::StepResult::PendingChoice {
        selects += 1;
        assert!(selects < 5, "runaway choice loop");
        let pick = sts2_core::legal_actions(&state)
            .into_iter()
            .find(|a| {
                matches!(a, sts2_core::Action::ResolveChoice { selection } if !selection.is_empty())
            })
            .expect("a selection");
        step = sts2_core::apply(&mut state, pick);
    }
    assert_eq!(enemy_hp(&state), 82, "Glam replays SCULPTING_STRIKE: 9 x2");
}
