//! The card `OnPlay` tails that C# awaits behind a `CardPileCmd.Draw`, run by
//! `sts2-core` once a draw suspended on a reshuffle select (Stratagem)
//! finishes. See [`sts2_core::DrawThen`].

use sts2_core::cards::VarKey;
use sts2_core::{CardUid, CombatState, DrawThen, PowerKind};

use super::regent::helpers::forge_from_card;
use super::status::helpers::spawn_burn;

/// The tail each card runs after its draw, also called inline when the draw
/// did not suspend, so both paths share one order.
pub fn run(state: &mut CombatState, tail: DrawThen) {
    match tail {
        DrawThen::None => {}
        DrawThen::BigBang => {
            // BigBang.cs:42-45 (both builds): GainStars, GainEnergy, Forge.
            state.gain_stars(1);
            state.gain_energy(1);
            let _ = forge_from_card(state, 5);
        }
        DrawThen::Restlessness => state.gain_energy(2),
        DrawThen::RestlessnessPlus => state.gain_energy(3),
        DrawThen::BattleTrance => {
            super::ironclad::helpers::apply_power_to_player(state, PowerKind::NoDraw, 1);
        }
        DrawThen::Glow => {
            super::regent::helpers::apply_power_to_player(state, PowerKind::DrawCardsNextTurn, 1);
        }
        DrawThen::Neurosurge => {
            super::necrobinder::helpers::apply_power_to_player(state, PowerKind::Neurosurge, 3);
        }
        DrawThen::Overclock => {
            let _ = spawn_burn(state, sts2_core::card_effects::PileKind::Discard);
        }
        DrawThen::Stoke(_)
        | DrawThen::StokePlus(_)
        | DrawThen::FiendFire(_)
        | DrawThen::FlakCannon(_) => {
            if !exhaust_owed(state, tail) {
                after_exhaust_loop(state, tail);
            }
        }
    }
}

/// The code an exhaust loop's card runs once every exhaust is done.
fn after_exhaust_loop(state: &mut CombatState, tail: DrawThen) {
    match tail {
        DrawThen::Stoke(count) => {
            super::ironclad::stoke::generate_stoke_cards(state, count, false);
        }
        DrawThen::StokePlus(count) => {
            super::ironclad::stoke::generate_stoke_cards(state, count, true);
        }
        DrawThen::FiendFire(packed) => super::ironclad::fiend_fire::fiend_fire_hits(state, packed),
        DrawThen::FlakCannon(hits) => super::defect::flak_cannon::flak_cannon_hits(state, hits),
        _ => {}
    }
}

/// An exhaust loop's `CardCmd.Exhaust` of one snapshotted card, from
/// whichever pile holds it now (a card already exhausted is skipped).
fn exhaust_one(state: &mut CombatState, uid: CardUid, tail: DrawThen) {
    if !matches!(tail, DrawThen::FlakCannon(_)) && state.player.hand.contains(&uid) {
        super::ironclad::helpers::exhaust_hand_card(state, uid);
    } else {
        let _ = sts2_core::card_effects::exhaust_card_uid(state, uid);
    }
}

/// An exhaust loop that C# awaits card by card (Stoke.cs:24-27,
/// FiendFire.cs:32-35, FlakCannon.cs:38-41, both builds), then `tail`, the
/// card's code after it. When an exhaust's `AfterCardExhausted` walk suspends
/// a draw on a reshuffle select (Dark Embrace / Joss Paper drawing into
/// Stratagem), C# finishes that pick and draw before the next exhaust: the
/// cards still owed are marked ([`VarKey::ExhaustOwedAfterDraw`]) and `tail`
/// parks on the draw, which exhausts them when it resumes.
pub fn exhaust_then(state: &mut CombatState, uids: &[CardUid], tail: DrawThen) {
    if !exhaust_each(state, uids, tail) {
        after_exhaust_loop(state, tail);
    }
}

/// Exhaust `uids` in order; `true` when one suspended a draw and `tail` (with
/// the rest marked owed) parked on it.
fn exhaust_each(state: &mut CombatState, uids: &[CardUid], tail: DrawThen) -> bool {
    let draw_was_suspended = state.pending_draw_resume.is_some();
    for (i, &uid) in uids.iter().enumerate() {
        exhaust_one(state, uid, tail);
        if state.park_draw_then(draw_was_suspended, tail) {
            for (order, &rest) in (1..).zip(&uids[i + 1..]) {
                if let Some(card) = sts2_core::card_by_uid_mut(&mut state.cards, rest) {
                    card.set_combat_var(VarKey::ExhaustOwedAfterDraw, order);
                }
            }
            return true;
        }
    }
    false
}

/// Resume the exhausts an earlier [`exhaust_then`] left owed, in snapshot
/// order; `true` when one of them parked `tail` again.
fn exhaust_owed(state: &mut CombatState, tail: DrawThen) -> bool {
    let key = VarKey::ExhaustOwedAfterDraw;
    let mut owed: smallvec::SmallVec<[(i32, CardUid); 8]> = state
        .cards
        .iter()
        .filter(|c| c.combat_var(key) > 0)
        .map(|c| (c.combat_var(key), c.uid))
        .collect();
    if owed.is_empty() {
        return false;
    }
    owed.sort_unstable_by_key(|&(order, _)| order);
    let uids: smallvec::SmallVec<[CardUid; 8]> = owed.iter().map(|&(_, uid)| uid).collect();
    for &uid in &uids {
        if let Some(card) = sts2_core::card_by_uid_mut(&mut state.cards, uid) {
            card.combat_vars.zero(key);
        }
    }
    exhaust_each(state, &uids, tail)
}

/// After a card's draw: park `tail` on the draw if it suspended (the
/// resumed draw runs it), else run it now. `draw_was_suspended` is whether a
/// draw was already suspended before the card drew.
pub fn after_draw(state: &mut CombatState, draw_was_suspended: bool, tail: DrawThen) {
    if !state.park_draw_then(draw_was_suspended, tail) {
        run(state, tail);
    }
}
