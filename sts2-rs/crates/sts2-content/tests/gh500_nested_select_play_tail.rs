//! GH #500: a card-select drained INSIDE an in-flight auto-play must not run
//! the parent HAND-play's deferred tail.
//!
//! C# `CardModel.OnPlayWrapper` moves a played card to its result pile only
//! after `OnPlay` returns (`CardModel.cs:1538-1552`), and `Catastrophe.OnPlay`
//! `await`s every `CardCmd.AutoPlay` in its loop — so the auto-played cards
//! ALWAYS reach the discard pile before Catastrophe itself does.
//!
//! Rust defers the same tail on `CombatState::pending_play` while a surfaced
//! select is open, and `resolve_choice_action` runs it once the pick lands. But
//! that resolver is re-entrant: the resume re-enters Catastrophe's loop
//! (`continuation`), and a select opened by a LATER iteration is drained inline
//! through the same `resolve_choice_action` — which used to consume
//! `pending_play` from underneath the still-running loop. Catastrophe then
//! landed in the discard pile in the MIDDLE of its own auto-play batch.
//!
//! The pile CONTENTS are identical either way, so the parity harness's
//! count-based state diff stayed clean; the ORDER only became observable at the
//! next reshuffle, whose `StableShuffle` is order-sensitive for tied sort keys
//! and therefore dealt a different hand from that point on (oracle seed 55 of
//! the auto-play/replay-heavy Ironclad campaign).

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{Action, CombatPhase, CombatState, StepResult};

fn push_card(state: &mut CombatState, id: CardId, card_type: CardType) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 0,
        ..Card::pod_defaults()
    });
    uid
}

/// CATASTROPHE auto-plays two ARMAMENTS off the draw pile. The first one's
/// upgrade select SURFACES (3 upgradable hand cards) and parks both the
/// auto-play tail and CATASTROPHE's own play tail; resolving it resumes the
/// Resuming the loop surfaces the second ARMAMENTS' select too.
/// CATASTROPHE stays on Play until both selects are answered.
#[test]
fn catastrophe_discards_after_the_auto_plays_it_is_still_awaiting() {
    two_select_batch(CardId::Catastrophe, None);
}

#[test]
fn cascade_surfaces_each_collected_sibling_selection() {
    two_select_batch(CardId::Cascade, None);
}

#[test]
fn cascade_preserves_outer_siblings_after_a_nested_havoc_selection() {
    two_select_batch(CardId::Cascade, Some(CardId::Havoc));
}

#[test]
fn cascade_keeps_nested_catastrophes_tail_parked_across_its_next_selection() {
    two_select_batch(CardId::Cascade, Some(CardId::Catastrophe));
}

fn two_select_batch(parent: CardId, inner: Option<CardId>) {
    content_dispatch::ensure_installed();

    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = if inner == Some(CardId::Catastrophe) {
        1
    } else {
        2
    };
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;

    // Hand: CATASTROPHE (index 0) + three upgradable fodder cards, so BOTH
    // auto-played ARMAMENTS see >= 2 upgradable cards and open a real choice
    // (the decomp auto-picks at <= 1 without prompting).
    let catastrophe = push_card(&mut state, parent, CardType::Skill);
    if parent == CardId::Cascade {
        state.cards[catastrophe.get() as usize - 1].cost = sts2_core::CardEnergyCost::new(-1);
    }
    state.player.hand.push(catastrophe);
    for _ in 0..3 {
        let fodder = push_card(&mut state, CardId::StrikeIronclad, CardType::Attack);
        state.player.hand.push(fodder);
    }
    // Draw pile: exactly the two cards CATASTROPHE will auto-play. Identical
    // ids, so the `StableShuffle` pick order cannot change what the test sees.
    let arm_a = push_card(
        &mut state,
        inner.unwrap_or(CardId::Armaments),
        CardType::Skill,
    );
    let arm_b = push_card(&mut state, CardId::Armaments, CardType::Skill);
    state.player.draw.push(arm_a);
    state.player.draw.push(arm_b);
    if inner.is_some() {
        let inner = push_card(&mut state, CardId::Armaments, CardType::Skill);
        state.player.draw.push(inner);
    }

    let step = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        step,
        StepResult::PendingChoice,
        "the first auto-played ARMAMENTS surfaces its upgrade select"
    );
    assert!(
        state.pending_play.is_some(),
        "CATASTROPHE's own play tail is deferred behind the surfaced choice"
    );
    assert!(
        !state.player.discard.contains(&catastrophe),
        "CATASTROPHE must not reach discard while its OnPlay is still suspended"
    );

    let selection: smallvec::SmallVec<[u8; 4]> = [0u8].into_iter().collect();
    let _ = sts2_core::apply(&mut state, Action::ResolveChoice { selection });

    assert!(state.pending_choice.is_some());
    assert!(state.pending_play.is_some());
    assert!(!state.player.discard.contains(&catastrophe));
    let selection: smallvec::SmallVec<[u8; 4]> = [0u8].into_iter().collect();
    let _ = sts2_core::apply(&mut state, Action::ResolveChoice { selection });
    assert!(
        state.pending_choice.is_none(),
        "both ARMAMENTS selects are answered"
    );
    assert!(
        state.pending_play.is_none(),
        "CATASTROPHE's deferred tail ran once the whole batch finished"
    );
    assert!(
        state.player.play.is_empty(),
        "no card is stranded on the Play pile"
    );

    let discard_ids: Vec<CardId> = state
        .player
        .discard
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect();
    assert_eq!(
        discard_ids,
        if inner == Some(CardId::Catastrophe) {
            vec![
                CardId::Armaments,
                CardId::Armaments,
                CardId::Catastrophe,
                parent,
            ]
        } else {
            vec![
                inner.unwrap_or(CardId::Armaments),
                CardId::Armaments,
                parent,
            ]
        },
        "C# awaits both auto-plays inside Catastrophe's OnPlay, so Catastrophe \
         is the LAST of the three to reach the discard pile"
    );

    let upgraded = state
        .player
        .hand
        .iter()
        .filter(|u| state.cards[u.0.get() as usize - 1].upgraded)
        .count();
    assert_eq!(upgraded, 2, "each ARMAMENTS upgraded exactly one hand card");
}
