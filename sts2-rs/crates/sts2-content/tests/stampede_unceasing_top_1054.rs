//! GH #1054 (Ironclad seed 1004, v0.111.0): a Stampede auto-play that empties
//! the hand must fire UnceasingTop.
//!
//! `StampedePower.AfterAutoPostPlayPhaseEntered` (StampedePower.cs:18-33)
//! auto-plays through `CardCmd.AutoPlay` -> `OnPlayWrapper`, which ends with
//! `CombatManager.CheckForEmptyHand` (CardModel.cs:2006). The phase is
//! AutoPostPlay (CombatManager.cs:1547-1549), which UnceasingTop's
//! `IsValidPhase` admits, so the relic draws 1 — and the end-of-turn flush then
//! discards that card. Rust skipped the check, so the card stayed on top of the
//! draw pile and was drawn at the next turn start instead.

use std::num::NonZeroU32;

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, Action, CardUid, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
    MonsterKind, PlayerCombatState, PowerKind, Relic, RelicKind,
};

/// Hand = [one Strike], draw = 8 Defends (enough that the next turn's draw
/// never reshuffles), Stampede(1). Returns the state and the draw pile's top.
fn setup(with_top: bool) -> (CombatState, CardUid) {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();

    let mut cards: Vec<Card> = Vec::new();
    let push = |cards: &mut Vec<Card>, card_id, card_type, target_type, dmg| -> CardUid {
        let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
        cards.push(Card {
            uid,
            card_id,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type,
            target_type,
            upgraded: false,
            base_damage: dmg,
            base_block: 5,
            ..Card::pod_defaults()
        });
        uid
    };
    let strike = push(
        &mut cards,
        CardId::StrikeIronclad,
        CardType::Attack,
        TargetType::Enemy,
        6,
    );
    let draw_uids: SmallVec<[CardUid; 32]> = (0..8)
        .map(|_| {
            push(
                &mut cards,
                CardId::DefendIronclad,
                CardType::Skill,
                TargetType::Self_,
                0,
            )
        })
        .collect();

    let enemy_uid = CreatureUid::new(1).expect("non-zero");
    let creatures = vec![Creature {
        id: enemy_uid.0,
        current_hp: 400,
        max_hp: 400,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::ShrinkerBeetle),
        monster_id: MonsterId(MonsterKind::ShrinkerBeetle.python_name()),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: Some(build_ai(
            MonsterKind::ShrinkerBeetle,
            MonsterSpawnArgs::None,
            SpawnPosition {
                alone: true,
                front: true,
                back: true,
                index: 0,
            },
        )),
    }];
    let player = PlayerCombatState {
        draw: draw_uids.clone(),
        max_energy: 3,
        current_hp: 400,
        max_hp: 400,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards: cards.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy_uid]),
        ..CombatState::default()
    };
    if with_top {
        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: RelicKind::UnceasingTop,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }
    let _ = start_combat(&mut state);
    let puid = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(&mut state, puid, PowerKind::Stampede, 1);
    let _ = start_player_turn(&mut state);

    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = draw_uids.iter().copied().collect();
    state.player.hand.push(strike);
    state.player.energy = 3;
    let top = state.player.draw[0];
    (state, top)
}

#[test]
#[serial_test::serial(global_registry)]
fn stampede_emptying_the_hand_fires_unceasing_top() {
    let (mut state, top) = setup(true);
    let _ = apply(&mut state, Action::EndTurn);
    assert!(
        state.player.discard.contains(&top),
        "UnceasingTop must draw the top card during AutoPostPlay and the \
         end-of-turn flush must discard it"
    );
    assert!(
        !state.player.hand.contains(&top),
        "the drawn card must not survive into the next turn's hand"
    );
}

#[test]
#[serial_test::serial(global_registry)]
fn stampede_without_unceasing_top_draws_nothing_extra() {
    let (mut state, top) = setup(false);
    let _ = apply(&mut state, Action::EndTurn);
    assert!(
        state.player.hand.contains(&top),
        "without UnceasingTop the top card is drawn at the next turn start"
    );
}

/// Ironclad 2164 act 3 (v0.111.0, Queen): Stampede picks the only attack in
/// hand, a `Bound` card after a Bound card was already played this turn.
/// `Hook.ShouldPlay` vetoes it, so C# `CardCmd.AutoPlay` sends it to its result
/// pile via `MoveToResultPileWithoutPlaying` and returns (v0.111.0 CardCmd.cs
/// AutoPlay; v0.107.1 CardCmd.cs:61-66). It never enters `OnPlayWrapper`, whose
/// tail is the only card-play `CheckForEmptyHand` (CardModel.cs:2006;
/// v0.107.1 :1993), so UnceasingTop does NOT draw. Rust checked the empty hand
/// after the blocked auto-play too and drew the draw-pile top.
#[test]
#[serial_test::serial(global_registry)]
fn stampede_blocked_auto_play_does_not_fire_unceasing_top() {
    let (mut state, top) = setup(true);
    let strike = state.player.hand[0];
    let idx = strike.0.get() as usize - 1;
    assert!(
        state.cards[idx].set_affliction(sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Bound,
            1,
        ))
    );
    // The Queen's PUPPET_STRINGS applies ChainsOfBinding on the legacy player
    // surface (`apply_power_to_player_via_legacy_surface`, act3_glory.rs).
    let next_id = state.powers.iter().map(|p| p.id.get()).max().unwrap_or(0) + 1;
    let id = NonZeroU32::new(next_id).expect("power id non-zero");
    state.powers.push(sts2_core::Power {
        id,
        kind: PowerKind::ChainsOfBinding,
        amount: 3,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(sts2_core::PowerUid(id));
    state.combat_log.chains_bound_card_played_this_turn = true;

    let _ = apply(&mut state, Action::EndTurn);

    assert!(
        state.player.discard.contains(&strike),
        "the vetoed Bound Strike goes to discard unplayed"
    );
    assert!(
        !state.player.discard.contains(&top),
        "a ShouldPlay-vetoed auto-play never reaches CheckForEmptyHand, so \
         UnceasingTop must not draw the top card into the end-of-turn flush"
    );
    assert!(
        state.player.hand.contains(&top),
        "the top card is drawn by the next turn's normal draw instead"
    );
}
