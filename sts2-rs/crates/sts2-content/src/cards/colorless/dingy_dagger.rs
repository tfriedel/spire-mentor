//! Dingy Dagger — v0.113.0-only event card (Waterfall of Obvious Secrets).
use super::prelude::*;

/// Dingy Dagger (Event, 0 energy, Attack, `AnyEnemy`) — v0.113.0
/// `Cards/DingyDagger.cs`: `DamageVar(7)`, one hit on the target.
///
/// `OnUpgrade` adds `3 + CurrentUpgradeLevel` and `MaxUpgradeLevel` is
/// unbounded. `UpgradeInternal` bumps the level BEFORE `OnUpgrade`, so the
/// first upgrade is +4 (7 -> 11). The engine's card POD carries a single
/// `upgraded` flag, so only that first step is modelled.
pub(crate) fn play_dingy_dagger(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let t = target.expect("DingyDagger: Enemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::DingyDagger,
        name: "DINGY_DAGGER",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 7,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}
