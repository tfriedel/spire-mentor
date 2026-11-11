//! GH #403: FTL must be played while its draw window is still open.
//!
//! FTL reads `Deal 5/6 damage. If you have played fewer than 3/4 cards this
//! turn, draw 1 card.` (C# `Ftl.cs` — `PlayMax` 3, `OnUpgrade` bumps it to
//! 4). It costs 0, so the rider is free upside that expires the moment
//! enough other cards are played; a human plays it early without thinking.
//! The reported defect is runs where it went late and the draw never fired.
//!
//! Root cause is not the card port (verified faithful against `Ftl.cs`) but
//! the search: one extra card moves `sts2_core::evaluate_in_progress` by
//! ~0.002 after its 114.56 normalizer, an order of magnitude under the
//! ~0.02 Q noise at a run's 1000-sim budget, so `argmax(Q)` coin-flips the
//! ordering. Measured over 300 MCTS-driven fights on a hard Act-1 mix
//! (`cargo run --release -p sts2-bench --example ftl_window_probe`), only
//! 67.7% of base-FTL plays landed inside the window;
//! `expert::DRAW_WINDOW_ROOT_Q_BONUS` lifts that to 91.2% with mean
//! end-of-fight HP unchanged.
//!
//! This file pins the contract: the predicate that decides whether a window
//! is live, and a hand where the search must take the rider. The fixture in
//! [`plays_ftl_while_the_window_is_open`] is deliberately a *busy* hand —
//! two-card fixtures are not discriminating (the search already gets those
//! right at 12/12 seeds with the bonus disabled), and a test that passes
//! either way pins nothing.

mod common;

use common::build_handful_ex;
use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
use sts2_core::{
    apply, is_terminal, legal_actions, Action, CardUid, CombatPhase, CombatState, Intent,
};
use sts2_mcts::expert::draw_window_is_live;
use sts2_mcts::{Mcts, MctsConfig};

const SIMS: u32 = 1000;
const SEEDS: &[u64] = &[0, 1, 2, 3, 4];

/// `hand` plus one FTL appended last (so it is never at hand index 0 and a
/// naive left-to-right pick fails), `cards_played` cards already played
/// this turn, and `draw_pile` Strikes left to draw. Player and enemy HP are
/// set so the turn is not lethal in either direction — the position is
/// *only* about the rider.
fn fixture(
    seed: u64,
    upgraded: bool,
    cards_played: u32,
    draw_pile: usize,
    rest_of_hand: &[(CardId, i32, i32, i8, TargetType)],
) -> CombatState {
    let mut hand = rest_of_hand.to_vec();
    hand.push((
        CardId::Ftl,
        if upgraded { 6 } else { 5 },
        0,
        0,
        TargetType::Enemy,
    ));
    let mut state = build_handful_ex(
        seed,
        &hand,
        /*cultist_hp=*/ 45,
        /*player_hp=*/ 70,
        Intent::Attack { damage: 5 },
        &[],
        /*energy=*/ 3,
    );
    // `build_handful` always writes `upgraded: false`; FTL+ needs the flag
    // because `PlayMax` reads it (damage came in through the hand tuple).
    if upgraded {
        let ftl_uid = *state.player.hand.last().expect("hand non-empty");
        state.cards[ftl_uid.0.get() as usize - 1].upgraded = true;
    }
    // Cards already played this turn — the whole point of the fixture.
    state.player.cards_played_this_turn[CardType::Attack.index()] = cards_played;
    for _ in 0..draw_pile {
        let raw = u32::try_from(state.cards.len() + 1).expect("fits u32");
        let uid = CardUid::new(raw).expect("non-zero");
        state.cards.push(Card {
            uid,
            card_id: CardId::StrikeDefect,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        state.player.draw.push(uid);
    }
    state
}

fn card_of(state: &CombatState, hand_index: u8) -> Option<CardId> {
    let uid = *state.player.hand.get(hand_index as usize)?;
    state.cards.iter().find(|c| c.uid == uid).map(|c| c.card_id)
}

/// Drive the turn with MCTS, returning the `CardId`s played in order.
fn played_sequence(state: &CombatState, seed: u64) -> Vec<CardId> {
    let mut state = state.clone();
    let mut played = Vec::new();
    // Runaway guard; the fixture has 2 cards plus at most a drawn Strike.
    for _ in 0..8 {
        if state.phase != CombatPhase::PlayerPlayPhase || is_terminal(&state) {
            break;
        }
        if legal_actions(&state).is_empty() {
            break;
        }
        let cfg = MctsConfig {
            max_simulations: SIMS,
            seed,
            ..MctsConfig::default()
        };
        let action = Mcts::new(state.clone(), cfg).search();
        if let Action::PlayCard { hand_index, .. } = action {
            if let Some(id) = card_of(&state, hand_index) {
                played.push(id);
            }
        }
        let _ = apply(&mut state, action.clone());
        if matches!(action, Action::EndTurn) || is_terminal(&state) {
            break;
        }
    }
    played
}

/// The busy hand the search gets wrong without the fix: a 0-cost attack, a
/// 1-cost attack, a 1-cost block, a 2-cost debuff-attacher and a second
/// 0-cost attack, with 3 energy — enough options that no single ordering
/// dominates on backed-up value, which is exactly when the coin-flip bites.
/// Measured at 0/12 seeds picking FTL with the bonus disabled, 12/12 with it.
const BUSY_HAND: [(CardId, i32, i32, i8, TargetType); 5] = [
    (CardId::Slice, 6, 0, 0, TargetType::Enemy),
    (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
    (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
    (CardId::Bash, 8, 0, 2, TargetType::Enemy),
    (CardId::Slice, 6, 0, 0, TargetType::Enemy),
];

/// The predicate behind the whole fix: a window is live only for a card that
/// has one, only while the play count is strictly under `PlayMax`, and only
/// when there is actually something left to draw.
#[test]
fn draw_window_predicate_matches_the_csharp_gate() {
    let one_card: [(CardId, i32, i32, i8, TargetType); 1] =
        [(CardId::Slice, 6, 0, 0, TargetType::Enemy)];
    let ftl_at = |state: &CombatState| -> Card {
        let uid = *state.player.hand.last().expect("hand non-empty");
        state.cards[uid.0.get() as usize - 1]
    };

    // Base FTL: PlayMax 3 — live at 2 plays, dead at 3.
    let s = fixture(0, false, 2, 3, &one_card);
    assert!(draw_window_is_live(&s, &ftl_at(&s)));
    let s = fixture(0, false, 3, 3, &one_card);
    assert!(!draw_window_is_live(&s, &ftl_at(&s)));

    // Upgraded FTL: `OnUpgrade` bumps PlayMax to 4 — still live at 3.
    let s = fixture(0, true, 3, 3, &one_card);
    assert!(draw_window_is_live(&s, &ftl_at(&s)));
    let s = fixture(0, true, 4, 3, &one_card);
    assert!(!draw_window_is_live(&s, &ftl_at(&s)));

    // Nothing to draw (empty draw AND discard) — the rider is a no-op, so
    // the window is not "live" and the search gets no nudge.
    let s = fixture(0, false, 0, 0, &one_card);
    assert!(s.player.draw.is_empty() && s.player.discard.is_empty());
    assert!(!draw_window_is_live(&s, &ftl_at(&s)));

    // A card with no play-count-gated rider is never live, whatever the
    // play count — Slice sits at hand index 0.
    let s = fixture(0, false, 0, 3, &one_card);
    let slice = s.cards[s.player.hand[0].0.get() as usize - 1];
    assert_eq!(slice.card_id, CardId::Slice);
    assert!(!draw_window_is_live(&s, &slice));
}

/// The #403 contract: out of a busy hand, FTL goes first while its window is
/// open. Every card here gets played this turn either way, so the orderings
/// deal the same damage — the only thing that separates FTL-first is the
/// free card, and FTL-first therefore weakly dominates. The margin is below
/// the search's noise floor, so without
/// `expert::DRAW_WINDOW_ROOT_Q_BONUS` the pick is a coin flip that the
/// static bucket order (FTL at 40, behind attacks at 35 and free attacks at
/// 30) resolves the wrong way on every seed.
#[test]
fn plays_ftl_while_the_window_is_open() {
    let mut wrong = Vec::new();
    for &seed in SEEDS {
        let state = fixture(
            seed, /*upgraded=*/ false, /*cards_played=*/ 0, 3, &BUSY_HAND,
        );
        let seq = played_sequence(&state, seed);
        if seq.first() != Some(&CardId::Ftl) {
            wrong.push((seed, seq));
        }
    }
    assert!(
        wrong.is_empty(),
        "MCTS deferred FTL out of a busy hand. FTL costs 0 and draws a card while fewer \
         than PlayMax cards have been played, so playing it first is free upside — every \
         ordering here plays the same cards for the same damage. \
         Offending (seed, sequence) = {wrong:?}",
    );
}

/// Same hand at the last play that can still draw (2 played, `PlayMax` 3).
/// Separate from the open-window case because this is where getting it wrong
/// actually costs the card rather than just risking it.
#[test]
fn plays_ftl_before_the_window_closes() {
    let mut wrong = Vec::new();
    for &seed in SEEDS {
        let state = fixture(
            seed, /*upgraded=*/ false, /*cards_played=*/ 2, 3, &BUSY_HAND,
        );
        let seq = played_sequence(&state, seed);
        if seq.first() != Some(&CardId::Ftl) {
            wrong.push((seed, seq));
        }
    }
    assert!(
        wrong.is_empty(),
        "MCTS let FTL's draw window close: with 2 cards already played and PlayMax 3, \
         the next play is the last one that can still draw. \
         Offending (seed, sequence) = {wrong:?}",
    );
}

/// The other side of the contract: once the window has closed, the bonus
/// must stop firing and the search is back to ordering on merit. Guards
/// against the tie-break degenerating into a permanent FTL-first bias —
/// that would be a different bug, not a fix. Asserted on the predicate
/// (which is what gates every consumer) plus a smoke check that the search
/// still plays the hand out.
#[test]
fn no_preference_once_the_window_has_closed() {
    let state = fixture(
        0, /*upgraded=*/ false, /*cards_played=*/ 3, 3, &BUSY_HAND,
    );
    let ftl_uid = *state.player.hand.last().expect("hand non-empty");
    let ftl = state.cards[ftl_uid.0.get() as usize - 1];
    assert!(!draw_window_is_live(&state, &ftl));
    assert!(!sts2_mcts::expert::has_live_draw_window(&state));
    let seq = played_sequence(&state, 0);
    assert!(
        seq.contains(&CardId::Ftl),
        "expected FTL to still be played once the window is closed; got {seq:?}",
    );
}
