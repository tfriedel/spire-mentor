//! God Byrd — v0.113.0-only event card (Byrdpip's evolution into Byrdine).
use super::prelude::*;

/// God Byrd (Event, 0 energy, Skill, `TargetType.None`) — v0.113.0
/// `Cards/GodByrd.cs`: `PowerCmd.Apply<GodByrdPower>(Owner.Creature,
/// Damage.BaseValue)` with `DamageVar(30, Unpowered)`; `OnUpgrade` adds 10.
/// The power deals its amount to every hittable enemy at the owner's next
/// player turn start (`sts2_core::combat_loop`).
pub(crate) fn play_god_byrd(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let amount = current_damage(state);
    sts2_core::fool::apply_to_player(state, sts2_core::powers::PowerKind::GodByrd, amount);
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::GodByrd,
        name: "GOD_BYRD",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::None,
        rarity: CardRarity::Event,
        base_damage: 30,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}
