//! My Shinies — v0.113.0-only Rare colorless Attack.
use super::prelude::*;

/// Gold per hit: C# `GoldVar(20)`; `OnUpgrade` runs
/// `Gold.UpgradeValueBy(-10)`.
pub const MY_SHINIES_GOLD_PER_HIT: i32 = 20;
/// [`MY_SHINIES_GOLD_PER_HIT`] after the upgrade.
pub const MY_SHINIES_GOLD_PER_HIT_UPGRADED: i32 = 10;

/// `CalculatedHits` (v0.113.0 `Cards/MyShinies.cs`): `CalculationBase(0) +
/// CalculationExtra(1) * Owner.Gold / Gold.BaseValue`, cast to `int` (a
/// truncation). Shared with the card face — see `crate::cards::preview`.
pub(crate) fn preview_my_shinies(
    state: &CombatState,
    card: &Card,
    _target: Option<CreatureUid>,
) -> i32 {
    let per_hit = if card.upgraded {
        MY_SHINIES_GOLD_PER_HIT_UPGRADED
    } else {
        MY_SHINIES_GOLD_PER_HIT
    };
    state.gold.max(0) / per_hit
}

/// `DamageCmd.Attack(Damage.BaseValue).WithHitCount(CalculatedHits)` on the
/// target: 1 damage per hit.
pub(crate) fn play_my_shinies(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let t = target.expect("MyShinies: Enemy-targeted");
    let base = current_damage(state);
    let hits = resolving_preview(state, preview_my_shinies, target);
    for _ in 0..hits {
        let _ = deal_damage_player_to_enemy(state, t, base);
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::MyShinies,
        name: "MY_SHINIES",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 1,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}
