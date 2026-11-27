//! Fool cards that only use pre-v0.113 mechanics: damage, block, draw,
//! energy, existing powers, and the Sadness / Joy cards. Values from the
//! v0.113.0 decomp `Models.Cards/<Name>.cs`; damage / block upgrades flow
//! through the spec's upgrade delta, secondary vars through [`var`].
use sts2_core::card_effects::{self, deal_damage_player_to_enemy, EffectResult};
use sts2_core::cards::CardId;
use sts2_core::{CombatState, CreatureUid, DrawThen, PowerKind};

use super::helpers::{
    apply_to_enemy, apply_to_player, current_block, current_damage, damage_all_enemies,
    generate_into_hand, v0114, var, var_by,
};
use crate::cards::draw_then::after_draw;

fn target(t: Option<CreatureUid>, card: &str) -> CreatureUid {
    t.unwrap_or_else(|| panic!("{card}: AnyEnemy-targeted"))
}

/// `CardPileCmd.Draw(n)` followed by `tail` (parked if the draw suspends).
fn draw_then(state: &mut CombatState, n: i32, tail: DrawThen) {
    let was_suspended = state.pending_draw_resume.is_some();
    let _ = sts2_core::cards::draw_via_state(state, n);
    after_draw(state, was_suspended, tail);
}

/// `StrikeFool.cs` / `Justice.cs`: one `DamageCmd.Attack` on the target
/// (6 / 30 damage).
pub(crate) fn play_single_attack(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "single attack");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    EffectResult::Resolved
}

/// `DefendFool.cs` / `Temperance.cs`: `GainBlock(Block)` (5 / 30).
pub(crate) fn play_block(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    card_effects::gain_block_player(state, current_block(state));
    EffectResult::Resolved
}

/// `TheChariot.cs` (25, all opponents) / `WahSweep.cs` (8).
pub(crate) fn play_all_attack(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    damage_all_enemies(state, current_damage(state));
    EffectResult::Resolved
}

/// `TheEmperor.cs`: `GainEnergy(Energy)`, 7 (+2); v0.114.0 3 (+1).
pub(crate) fn play_the_emperor(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let e = var_by(state, (7, 9), (3, 4));
    state.gain_energy(e);
    EffectResult::Resolved
}

/// `TheEmpress.cs`: Weak then Vulnerable, `Power` 3 (+1), on the target.
pub(crate) fn play_the_empress(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "TheEmpress");
    let n = var(state, 3, 4);
    apply_to_enemy(state, t, PowerKind::Weak, n);
    apply_to_enemy(state, t, PowerKind::Vulnerable, n);
    EffectResult::Resolved
}

/// `Strength.cs`: `Apply<StrengthPower>(Owner, 4 (+1))`.
pub(crate) fn play_strength(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 4, 5);
    apply_to_player(state, PowerKind::Strength, n);
    EffectResult::Resolved
}

/// `Death.cs`: `Apply<DoomPower>(target, 75 (+24))`; v0.114.0 66 (+15).
pub(crate) fn play_death(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Death");
    let n = var_by(state, (75, 99), (66, 81));
    apply_to_enemy(state, t, PowerKind::Doom, n);
    EffectResult::Resolved
}

/// `Joy.cs`: `GainEnergy(1 (+1))` then `Draw(1)`.
pub(crate) fn play_joy(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let e = var(state, 1, 2);
    state.gain_energy(e);
    draw_then(state, 1, DrawThen::None);
    EffectResult::Resolved
}

/// `TheHighPriestess.cs`: `Draw(4 (+1))`.
pub(crate) fn play_the_high_priestess(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    let n = var(state, 4, 5);
    draw_then(state, n, DrawThen::None);
    EffectResult::Resolved
}

/// `TheStar.cs`: `EnergyNextTurn` 3 (+1), then `DrawCardsNextTurn` 3 (+1).
pub(crate) fn play_the_star(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 3, 4);
    apply_to_player(state, PowerKind::EnergyNextTurn, n);
    apply_to_player(state, PowerKind::DrawCardsNextTurn, n);
    EffectResult::Resolved
}

/// `WahTouch.cs`: attack (6, +2) then Vulnerable 1 (+1) on the target.
pub(crate) fn play_wah_touch(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "WahTouch");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    let n = var(state, 1, 2);
    apply_to_enemy(state, t, PowerKind::Vulnerable, n);
    EffectResult::Resolved
}

/// `WahCho.cs`: `GainBlock` (3, +2) then the attack (3, +2).
pub(crate) fn play_wah_cho(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "WahCho");
    card_effects::gain_block_player(state, current_block(state));
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    EffectResult::Resolved
}

/// `WahChoKi.cs`: `GainBlock` (9, +3), the attack (9, +3), `Draw(1 (+1))`.
/// v0.114.0 attacks FIRST, then gains the block (8 / 8, +2 each):
///
/// [decompiled C# omitted — see the cited path]
pub(crate) fn play_wah_cho_ki(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "WahChoKi");
    if v0114(state) {
        let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
        card_effects::gain_block_player(state, current_block(state));
    } else {
        card_effects::gain_block_player(state, current_block(state));
        let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    }
    let n = var(state, 1, 2);
    draw_then(state, n, DrawThen::None);
    EffectResult::Resolved
}

/// `KiWhip.cs`: attack (5, +1), then `EnergyNextTurn` 2 (+1).
pub(crate) fn play_ki_whip(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "KiWhip");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    let n = var(state, 2, 3);
    apply_to_player(state, PowerKind::EnergyNextTurn, n);
    EffectResult::Resolved
}

/// `WahKi.cs`: a 2-hit attack (2, +2), then `EnergyNextTurn` 1 and
/// `DrawCardsNextTurn` 1 (neither upgrades).
pub(crate) fn play_wah_ki(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "WahKi");
    let dmg = current_damage(state);
    for _ in 0..card_effects::resolving_card_attack_hits(state, 2) {
        let _ = deal_damage_player_to_enemy(state, t, dmg);
    }
    if v0114(state) {
        // v0.114.0 `WahKi.cs`: 2 hits of 3 (+1), then one discard card to the
        // top of the draw pile (`FromCombatPile(Discard, prefs(1))`).
        super::arrows::discard_card_to_draw_top(state);
        return EffectResult::Resolved;
    }
    apply_to_player(state, PowerKind::EnergyNextTurn, 1);
    apply_to_player(state, PowerKind::DrawCardsNextTurn, 1);
    EffectResult::Resolved
}

/// `Timber.cs`: attack (20, +5); if that hit killed the target, a second
/// attack on all opponents with `OnKillDamage` (30, +5).
///
/// Through v0.113.1 those are two `DamageCmd.Attack` commands, so the first
/// one's `AfterAttack` spends Vigor / Gigantification before the sweep.
/// v0.114.0 opens one `AttackContext` around both `CreatureCmd.Damage`
/// calls (Timber.cs:39-58), so the buffs boost both and are spent once,
/// after the sweep, as for Echoing Slash.
///
/// [decompiled C# omitted — see the cited path]
pub(crate) fn play_timber(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Timber");
    let killed = card_effects::deal_damage_player_to_enemy_full(state, t, current_damage(state))
        .was_target_killed;
    if !v0114(state) {
        // v0.113.1 `Timber.cs:32-34`: the first `DamageCmd.Attack` ends here.
        if let Some(card) = card_effects::resolving_card_uid(state) {
            sts2_core::hooks::end_player_attack_command(state, card);
        }
    }
    if killed {
        let n = var(state, 30, 35);
        damage_all_enemies(state, n);
    }
    EffectResult::Resolved
}

/// `SorrowfulStrike.cs`: attack (13, +4), then 2 Sadness to hand (v0.114.0:
/// `CardsVar(1)`, one).
pub(crate) fn play_sorrowful_strike(
    state: &mut CombatState,
    t: Option<CreatureUid>,
) -> EffectResult {
    let t = target(t, "SorrowfulStrike");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    generate_into_hand(state, CardId::Sadness, if v0114(state) { 1 } else { 2 });
    EffectResult::Resolved
}

/// `Jest.cs`: `GainBlock` (14, +4), then 1 Sadness to hand.
pub(crate) fn play_jest(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    card_effects::gain_block_player(state, current_block(state));
    generate_into_hand(state, CardId::Sadness, 1);
    EffectResult::Resolved
}

/// `Vitriol.cs`: Weak 2 (+1) on the target, then 1 Sadness to hand
/// (v0.114.0: `CardsVar(2)`, two in one batch).
pub(crate) fn play_vitriol(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = target(t, "Vitriol");
    let n = var(state, 2, 3);
    apply_to_enemy(state, t, PowerKind::Weak, n);
    generate_into_hand(state, CardId::Sadness, if v0114(state) { 2 } else { 1 });
    EffectResult::Resolved
}

/// v0.114.0 `Optimism.cs`: `CardCmd.Exhaust` every Sadness in hand (hand
/// order), then, when there was any, `Draw(count)`.
///
/// [decompiled C# omitted — see the cited path]
pub(crate) fn play_optimism(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let sadness: smallvec::SmallVec<[sts2_core::CardUid; 10]> = state
        .player
        .hand
        .iter()
        .copied()
        .filter(|uid| state.cards[uid.0.get() as usize - 1].card_id == CardId::Sadness)
        .collect();
    for &uid in &sadness {
        let _ = card_effects::exhaust_card_uid(state, uid);
    }
    if !sadness.is_empty() {
        let n = i32::try_from(sadness.len()).unwrap_or(i32::MAX);
        draw_then(state, n, DrawThen::None);
    }
    EffectResult::Resolved
}

/// `BrightenUp.cs`: 2 Sadness, then 2 (+1) Joy, to hand.
pub(crate) fn play_brighten_up(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    generate_into_hand(state, CardId::Sadness, 2);
    let n = var(state, 2, 3);
    generate_into_hand(state, CardId::Joy, n);
    EffectResult::Resolved
}

/// `Rummage.cs`: `Draw(3 (+1))`, then 1 Sadness to hand
/// ([`DrawThen::FoolSadness`] when the draw suspends).
pub(crate) fn play_rummage(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 3, 4);
    draw_then(state, n, DrawThen::FoolSadness);
    EffectResult::Resolved
}
