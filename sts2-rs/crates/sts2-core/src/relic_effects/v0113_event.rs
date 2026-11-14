//! v0.113.0 event relics with a combat hook: ObsidianVice, TormentousCollar,
//! MightyWings, BathTowel and BathRobes (`Models.Relics/<Name>.cs` in the
//! v0.113.0 decomp). They come from the v0.113.0-only events
//! (`sts2_content::events::v0113`), so owning one is the version gate; the
//! explicit `game_version` check keeps a hand-built scenario honest too.
use super::prelude::*;

/// `ObsidianVice` `DamageVar(1m, ValueProp.Unpowered)`: damage per card in hand.
pub(crate) const OBSIDIAN_VICE_DAMAGE_PER_CARD: i32 = 1;
/// `TormentousCollar` `PowerVar<WeakPower>(2m)`.
pub(crate) const TORMENTOUS_COLLAR_WEAK: i32 = 2;
/// `MightyWings` `DynamicVar("DamageThreshold", 30m)`.
pub(crate) const MIGHTY_WINGS_DAMAGE_THRESHOLD: i32 = 30;
/// `MightyWings` `PowerVar<StrengthPower>(2m)`.
pub(crate) const MIGHTY_WINGS_STRENGTH: i32 = 2;
/// `BathTowel` `HealVar(8m)`.
pub(crate) const BATH_TOWEL_HEAL: i32 = 8;

fn active(state: &CombatState, kind: crate::RelicKind) -> bool {
    state.game_version >= crate::GameVersion::V0_113_0 && player_owns_relic(state, kind)
}

/// `ObsidianVice.BeforeSideTurnEnd` (owner's side): with N cards in hand,
/// deal N unpowered damage to `Rng.CombatTargets.NextItem(HittableEnemies)`.
/// An empty hand draws no target.
pub(crate) fn on_before_turn_end_obsidian_vice(
    state: &mut CombatState,
    side: Side,
) -> EffectResult {
    if side != Side::Player || !active(state, crate::RelicKind::ObsidianVice) {
        return EffectResult::NotImplemented;
    }
    let count = i32::try_from(state.player.hand.len()).unwrap_or(i32::MAX);
    if count == 0 {
        return EffectResult::NotImplemented;
    }
    let Some(target) = state.get_random_enemy() else {
        return EffectResult::Resolved;
    };
    let _ = crate::damage::apply_damage(
        state,
        CreatureRef::Player,
        target,
        OBSIDIAN_VICE_DAMAGE_PER_CARD.saturating_mul(count),
        ValueProps::UNPOWERED,
        None,
    );
    EffectResult::Resolved
}

/// `TormentousCollar.BeforeSideTurnStart`: in an Elite room, on the owner's
/// first turn, 2 Weak to every hittable enemy.
pub(crate) fn on_before_side_turn_start_tormentous_collar(
    state: &mut CombatState,
    side: Side,
) -> EffectResult {
    if side != Side::Player
        || state.turn_number() > 1
        || !state.is_elite()
        || !active(state, crate::RelicKind::TormentousCollar)
    {
        return EffectResult::NotImplemented;
    }
    let targets: smallvec::SmallVec<[crate::CreatureUid; 8]> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .collect();
    for t in targets {
        apply_power_to_enemy(state, t, PowerKind::Weak, TORMENTOUS_COLLAR_WEAK);
    }
    EffectResult::Resolved
}

/// `MightyWings.AfterDamageGiven`: a powered attack by the owner (or their
/// Osty) whose `TotalDamage` is at least 30 gives the owner 2 Strength.
pub(crate) fn on_after_damage_given_mighty_wings(
    state: &mut CombatState,
    ctx: &DamageGivenCtx,
) -> EffectResult {
    if !active(state, crate::RelicKind::MightyWings) {
        return EffectResult::NotImplemented;
    }
    let by_owner =
        state.player.creature_uid == Some(ctx.dealer) || state.player.pet_uid == Some(ctx.dealer);
    if !by_owner || !ctx.props.is_powered() || ctx.total_damage < MIGHTY_WINGS_DAMAGE_THRESHOLD {
        return EffectResult::NotImplemented;
    }
    let Some(uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    apply_power_to_creature(state, uid, PowerKind::Strength, MIGHTY_WINGS_STRENGTH);
    EffectResult::Resolved
}

/// `BathTowel.AfterPlayerTurnStartLate`: heal 8 on the owner's first turn.
/// Run from [`super::blood_vial::dispatch_after_player_turn_start_late`].
pub(crate) fn on_after_player_turn_start_late_bath_towel(state: &mut CombatState) -> EffectResult {
    if state.turn_number() > 1 || !active(state, crate::RelicKind::BathTowel) {
        return EffectResult::NotImplemented;
    }
    let Some(uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    state.creature_heal(uid, BATH_TOWEL_HEAL);
    EffectResult::Resolved
}

/// `BathRobes.BeforeCombatStart`: the first Boss fight heals the owner to
/// full, then the relic is used up (`WasUsed`, a saved property). `counter`
/// is `WasUsed` (0 armed, 1 used), carried run-side by the relic-counter
/// bridge like LizardTail's.
pub(crate) fn on_before_combat_start_bath_robes(state: &mut CombatState) -> EffectResult {
    if !state.is_boss || !active(state, crate::RelicKind::BathRobes) {
        return EffectResult::NotImplemented;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == crate::RelicKind::BathRobes && r.counter == 0)
    else {
        return EffectResult::NotImplemented;
    };
    let Some(uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    let Some(c) = state.creatures.iter().find(|c| c.id == uid.0) else {
        return EffectResult::NotImplemented;
    };
    if c.current_hp <= 0 {
        return EffectResult::NotImplemented;
    }
    let missing = c.max_hp.saturating_sub(c.current_hp);
    state.creature_heal(uid, missing);
    state.relics[idx].counter = 1;
    EffectResult::Resolved
}

pub(crate) fn install() {
    register_before_turn_end(
        RelicId::ObsidianVice,
        on_before_turn_end_obsidian_vice as BeforeTurnEndFn,
    );
    register_before_side_turn_start(
        RelicId::TormentousCollar,
        on_before_side_turn_start_tormentous_collar as BeforeSideTurnStartFn,
    );
    register_after_damage_given(
        RelicId::MightyWings,
        on_after_damage_given_mighty_wings as AfterDamageGivenFn,
    );
    register_before_combat_start(
        RelicId::BathRobes,
        on_before_combat_start_bath_robes as BeforeCombatStartFn,
    );
}
