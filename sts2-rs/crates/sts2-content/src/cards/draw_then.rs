//! The card `OnPlay` tails that C# awaits behind a `CardPileCmd.Draw`, run by
//! `sts2-core` once a draw suspended on a reshuffle select (Stratagem)
//! finishes. See [`sts2_core::DrawThen`].

use sts2_core::{CombatState, DrawThen, PowerKind};

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
        DrawThen::Stoke(count) => {
            super::ironclad::stoke::generate_stoke_cards(state, count, false);
        }
        DrawThen::StokePlus(count) => {
            super::ironclad::stoke::generate_stoke_cards(state, count, true);
        }
    }
}

/// After a card's draw: park `tail` on the draw if it suspended (the
/// resumed draw runs it), else run it now. `draw_was_suspended` is whether a
/// draw was already suspended before the card drew.
pub fn after_draw(state: &mut CombatState, draw_was_suspended: bool, tail: DrawThen) {
    if !state.park_draw_then(draw_was_suspended, tail) {
        run(state, tail);
    }
}
