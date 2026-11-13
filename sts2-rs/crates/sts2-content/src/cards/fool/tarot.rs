//! Tarot tokens (MythicFormPower's deck) whose effects use pre-v0.113
//! mechanics. Values from the v0.113.0 decomp `Models.Cards/<Name>.cs`.
use sts2_core::card_effects::{self, EffectResult, PileKind};
use sts2_core::{CombatState, CreatureUid, DrawThen};

use super::helpers::var;
use crate::cards::draw_then::after_draw;

/// `WheelOfFortune.cs`: `CardCmd.DiscardAndDraw(hand, 7 (+2))` when the hand
/// has cards, else a plain `Draw(7 (+2))`.
pub(crate) fn play_wheel_of_fortune(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let n = var(state, 7, 9);
    let hand: smallvec::SmallVec<[sts2_core::CardUid; 16]> =
        state.player.hand.iter().copied().collect();
    if hand.is_empty() {
        let was_suspended = state.pending_draw_resume.is_some();
        let _ = sts2_core::cards::draw_via_state(state, n);
        after_draw(state, was_suspended, DrawThen::None);
    } else {
        let _ = card_effects::discard_and_draw_with_hooks(state, &hand, n);
    }
    EffectResult::Resolved
}

/// `TheMoon.cs`: `Shiv.CreateInHand(MaxCardsInHand - hand count)`, each
/// upgraded afterwards when TheMoon is upgraded (`CardCmd.Upgrade`).
pub(crate) fn play_the_moon(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let count = sts2_core::cards::HAND_LIMIT.saturating_sub(state.player.hand.len());
    if count == 0 || sts2_core::combat_loop::combat_is_ending(state) {
        return EffectResult::Resolved;
    }
    let upgraded = super::helpers::upgraded(state);
    let mut shivs: smallvec::SmallVec<[sts2_core::CardUid; 10]> = smallvec::SmallVec::new();
    for _ in 0..count {
        shivs.push(crate::cards::status::helpers::spawn_shiv(
            state,
            PileKind::Hand,
        ));
    }
    if upgraded {
        for uid in shivs {
            let _ = card_effects::upgrade_card_uid(state, uid);
        }
    }
    EffectResult::Resolved
}

/// `TheMagician.cs`: `OrbCmd.AddSlots(3 (+1))`, then channel that many
/// `OrbModel.GetRandomOrb(CombatOrbGeneration)` orbs.
pub(crate) fn play_the_magician(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 3, 4);
    sts2_core::orbs::ensure_character_base_slots(state);
    state.player.orbs.add_slots(usize::try_from(n).unwrap_or(0));
    for _ in 0..n {
        let orb = sts2_core::orbs::random_combat_orb(state);
        sts2_core::orbs::channel_orb(state, orb);
    }
    EffectResult::Resolved
}

/// `TheSun.cs`: `Draw(2 (+1))`, then `GainEnergy(2 (+1))` and one Soot into
/// the discard pile (`AddToCombatAndPreview<Soot>`, a generated card), parked
/// as [`DrawThen::FoolTheSun`] when the draw suspends.
pub(crate) fn play_the_sun(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let up = super::helpers::upgraded(state);
    let n = if up { 3 } else { 2 };
    let was_suspended = state.pending_draw_resume.is_some();
    let _ = sts2_core::cards::draw_via_state(state, n);
    after_draw(state, was_suspended, DrawThen::FoolTheSun(up));
    EffectResult::Resolved
}

/// The tail of [`play_the_sun`].
pub(crate) fn the_sun_tail(state: &mut CombatState, upgraded: bool) {
    state.gain_energy(if upgraded { 3 } else { 2 });
    let soot = crate::cards::status::card_from_spec(&crate::cards::status_specs_generated::soot());
    let _ = card_effects::spawn_card_into_pile(state, soot, PileKind::Discard);
}
