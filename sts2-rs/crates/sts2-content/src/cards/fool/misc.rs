//! Fool cards with a one-off shape: Anguish, TheHangedMan, SelfDeprecate,
//! Bounce. Values from the v0.113.0 decomp `Models.Cards/<Name>.cs`.
use sts2_core::card_effects::{deal_damage_player_to_enemy, EffectResult};
use sts2_core::cards::{CardId, VarKey};
use sts2_core::{CombatState, CreatureUid, PowerKind};

use super::helpers::{apply_to_player, generate_into_hand, var};

/// The Sadness cards the player generated this combat: C#
/// `History.Entries.OfType<CardGeneratedEntry>()` with `Card is Sadness`.
/// Every generation (Ennui, SelfDeprecate, MourningBlow's transform, ...)
/// mints a fresh arena row stamped `GeneratedInCombat`, and a later
/// transform (PunchOfJoy, Ecstasy) leaves that row behind, so the arena
/// keeps the history.
fn sadness_generated_this_combat(state: &CombatState) -> i32 {
    let n = state
        .cards
        .iter()
        .filter(|c| c.card_id == CardId::Sadness && c.combat_var(VarKey::GeneratedInCombat) != 0)
        .count();
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// `Anguish.cs`: `CalculatedDamage` = 6 (+1) + 6 (+1) x Sadness generated
/// this combat.
pub(crate) fn play_anguish(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = t.expect("Anguish: AnyEnemy-targeted");
    let per = var(state, 6, 7);
    let dmg = per.saturating_add(per.saturating_mul(sadness_generated_this_combat(state)));
    let _ = deal_damage_player_to_enemy(state, t, dmg);
    EffectResult::Resolved
}

/// `TheHangedMan.cs`: `CreatureCmd.Damage(self, 6 (-3), Unblockable |
/// Unpowered | Move)`, then `Apply<ExtraTurnPower>(1)`. v0.113.0 renamed
/// v0.111.0's `AmbergrisPower` to `ExtraTurnPower` (same body; the Ambergris
/// potion applies it too), which the engine keeps as [`PowerKind::Ambergris`].
pub(crate) fn play_the_hanged_man(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    crate::cards::ironclad::helpers::self_hp_loss(state, var(state, 6, 3));
    if state.player.current_hp > 0 {
        apply_to_player(state, PowerKind::Ambergris, 1);
    }
    EffectResult::Resolved
}

/// `SelfDeprecate.cs` (multiplayer-only): one Sadness to the owner's hand,
/// then 2 (+1) Joy to each living teammate, which a single-player combat
/// does not have.
pub(crate) fn play_self_deprecate(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    generate_into_hand(state, CardId::Sadness, 1);
    EffectResult::Resolved
}

/// `Bounce.cs`: `Apply<StretchPower>(10 (+3))`. Its `BeforeHandDraw` return
/// to hand is the Bolas sweep (`combat_loop::bolas_hatchet_returning`),
/// keyed on the play's turn stamp.
pub(crate) fn play_bounce(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    apply_to_player(state, PowerKind::Stretch, var(state, 10, 13));
    crate::cards::colorless::helpers::stamp_return_if_played(state);
    EffectResult::Resolved
}

/// `TheTower.cs`: `CalculatedDamage` = 0 + 2 (+1) x `RunState.TotalFloor`
/// (the combat's [`CombatState::total_floor`] snapshot).
pub(crate) fn play_the_tower(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = t.expect("TheTower: AnyEnemy-targeted");
    let dmg = var(state, 2, 3).saturating_mul(i32::from(state.total_floor));
    let _ = deal_damage_player_to_enemy(state, t, dmg);
    EffectResult::Resolved
}
