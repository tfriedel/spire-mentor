//! `ReboundPower` result-pile redirect, driven through the REAL Rebound card
//! (GH #637).
//!
//! This file exists for one assertion that the sts2-core unit tests cannot
//! make: **the Rebound card does not redirect its own play.**
//!
//! C# resolves `(pileType, position)` at the TOP of `CardModel.OnPlayWrapper`
//! (`CardModel.cs:1891`), before `OnPlay` runs. The Rebound card's `OnPlay`
//! grants `Rebound(1)`, so at the moment its own result pile was decided the
//! power did not exist — the card goes to Discard and the stack survives for
//! the NEXT card. Rust resolves the pile late, in `finish_card_play`, so a port
//! reading the LIVE stack there sees the power the card just applied, sends the
//! Rebound card to draw-top and burns the stack immediately.
//!
//! That is not hypothetical: it is what the first cut of GH #637 did, and the
//! C#-oracle coverage sweep caught it as
//! `card:REBOUND -> PILE_COUNT :: discard_count` (C# discard 1 / Rust 0,
//! `draw_ids[0]` REBOUND vs DEFEND_IRONCLAD, `player.powers.len` 2 vs 1). The
//! fix is `CardPlayFinish::rebound_amount_at_play_start`, the same play-start
//! snapshot `echo_form_amount_at_play_start` already uses for the same reason.
//!
//! The card lives in sts2-content, so only a content-side test can play it.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{CardUid, CombatState, MonsterKind, PowerKind};

/// A play-ready Ironclad state with `card_id` as the only card in hand.
fn state_with_card_in_hand(card_id: CardId) -> (CombatState, CardUid) {
    content_dispatch::install_default_registry();
    // `install_default_registry` wires the effect dispatch but not the card
    // POD factory, which lives on its own OnceLock. Idempotent.
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        ..Default::default()
    };
    state.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.phase = sts2_core::CombatPhase::PlayerPlayPhase;
    // `play_rebound` applies the power through `state.player.creature_uid`, so
    // the mirror Creature must exist or the card's own effect is a no-op.
    state.init_player_creature();

    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 200;
        c.current_hp = 200;
    }

    let card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    let mut card = card;
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    (state, uid)
}

#[test]
fn the_rebound_card_does_not_redirect_its_own_play() {
    let (mut state, uid) = state_with_card_in_hand(CardId::Rebound);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert!(
        state.player.discard.contains(&uid),
        "the Rebound card itself goes to DISCARD — its own result pile was \
         decided before its OnPlay granted the power (got draw={:?} discard={:?})",
        state.player.draw,
        state.player.discard
    );
    assert!(
        !state.player.draw.contains(&uid),
        "and specifically NOT to the draw pile"
    );
    assert_eq!(
        state.player_power_amount(PowerKind::Rebound),
        1,
        "the freshly granted stack survives for the NEXT card, unspent"
    );
}

#[test]
fn the_stack_it_granted_redirects_the_following_card() {
    // The other half: the port is not simply inert. The stack the Rebound card
    // left behind must redirect the very next Discard-bound play.
    let (mut state, rebound_uid) = state_with_card_in_hand(CardId::Rebound);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.player_power_amount(PowerKind::Rebound), 1);

    let strike = sts2_core::lookup_card_factory()
        .and_then(|f| f(CardId::StrikeIronclad, false))
        .expect("card factory has Strike");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    let mut strike = strike;
    strike.uid = uid;
    state.cards.push(strike);
    state.player.hand.push(uid);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert_eq!(
        state.player.draw.first().copied(),
        Some(uid),
        "the NEXT card lands on top of the draw pile"
    );
    assert!(
        state.player.discard.contains(&rebound_uid),
        "and the Rebound card is still sitting in discard where it landed"
    );
    assert_eq!(
        state.player_power_amount(PowerKind::Rebound),
        0,
        "that redirect spent the stack, removing the power"
    );
}

/// Push a fresh `card_id` into the hand and return its uid.
fn add_to_hand(state: &mut CombatState, card_id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    uid
}

#[test]
fn a_select_opening_card_spends_the_stack_before_its_select() {
    // C# `OnPlayWrapper` fires `ReboundPower.AfterModifyingCardPlayResultLocation`
    // (flash + decrement) before `OnPlay` (CardModel.cs:1881-1886, v0.111.0),
    // so Burning Pact's exhaust pick already shows the stack spent (Spire Bench
    // seed 180001292, GH #1907). The card still goes to the draw top.
    let (mut state, _rebound) = state_with_card_in_hand(CardId::Rebound);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.player_power_amount(PowerKind::Rebound), 1);
    state.player.energy = 3;

    let pact = add_to_hand(&mut state, CardId::BurningPact);
    // Two cards to pick from: a single option is taken without a select.
    let _fodder = add_to_hand(&mut state, CardId::StrikeIronclad);
    let _fodder2 = add_to_hand(&mut state, CardId::DefendIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);

    assert!(
        state.pending_choice.is_some(),
        "Burning Pact opens its exhaust pick"
    );
    assert_eq!(
        state.player_power_amount(PowerKind::Rebound),
        0,
        "the stack is spent before the select opens"
    );

    let _ = sts2_core::resolve_choice_atomic_apply(&mut state, &[0]);
    assert!(state.pending_choice.is_none());
    assert!(
        state.player.draw.contains(&pact) && !state.player.discard.contains(&pact),
        "Burning Pact still lands on the draw pile (draw={:?} discard={:?})",
        state.player.draw,
        state.player.discard
    );
    assert_eq!(state.player_power_amount(PowerKind::Rebound), 0);
}
