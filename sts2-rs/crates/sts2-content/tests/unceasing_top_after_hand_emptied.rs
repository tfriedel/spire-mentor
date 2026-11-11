//! GH #806: `AfterHandEmptied` was registered but never dispatched, so
//! UnceasingTop never drew. C# `CardModel.OnPlayWrapper` ends with
//! `CombatManager.CheckForEmptyHand` once the played card has reached its
//! result pile (v0.107.1 CardModel.cs:1993), and UnceasingTop draws 1 when the
//! hand is empty in a valid phase (UnceasingTop.cs:16-23).

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{CombatPhase, CombatState, GameVersion, Relic, RelicKind};

fn push_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 5,
        ..Card::pod_defaults()
    });
    uid
}

/// One Defend in hand, one in the draw pile. Returns (hand uid, draw uid).
fn setup(version: GameVersion, with_top: bool) -> (CombatState, sts2_core::CardUid) {
    content_dispatch::ensure_installed();
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;
    if with_top {
        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: RelicKind::UnceasingTop,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }
    let hand = push_card(&mut state, CardId::DefendSilent);
    let draw = push_card(&mut state, CardId::DefendSilent);
    state.player.hand.push(hand);
    state.player.draw.push(draw);
    (state, draw)
}

#[test]
#[serial_test::serial(global_registry)]
fn unceasing_top_draws_when_last_card_is_played() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let (mut state, draw) = setup(version, true);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        assert_eq!(
            state.player.hand.as_slice(),
            &[draw],
            "{version:?}: emptying the hand must draw 1 via UnceasingTop"
        );
        assert!(state.player.draw.is_empty());
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn no_draw_without_unceasing_top() {
    let (mut state, _) = setup(GameVersion::V0_107_1, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.draw.len(), 1);
}
