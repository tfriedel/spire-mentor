use super::prelude::*;

/// `BurningSticks.AfterCombatEnd` (C# `Relics/BurningSticks.cs:56-60`): `WasUsedThisCombat = false`
/// (and `AfterRoomEntered` re-arms it too, :33-42).
///
/// The exhaust listener already exists in `sts2-content`
/// (`shop_event.rs::listener_burning_sticks`, latch in `Relic.counter`: 0 = ready, 1 = used) but nothing ever
/// re-armed the latch, so after the first combat that exhausted a Skill the relic stayed dead for the rest of
/// the run (the real game returns the first exhausted Skill of EVERY combat). v0.111.0 only, so the v0.107.1
/// engine stays byte-identical.
pub(crate) fn on_after_combat_end_burning_sticks(state: &mut CombatState) -> EffectResult {
    if state.game_version < crate::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    let Some(r) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == crate::RelicKind::BurningSticks)
    else {
        return EffectResult::NotImplemented;
    };
    r.counter = 0;
    EffectResult::Resolved
}

pub(crate) fn install() {
    register_owned_after_combat_end(
        RelicId::BurningSticks,
        crate::RelicKind::BurningSticks,
        on_after_combat_end_burning_sticks as AfterCombatEndFn,
    );
}
