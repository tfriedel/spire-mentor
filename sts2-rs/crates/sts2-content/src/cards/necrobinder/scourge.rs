//! Necrobinder pool card — hand-authored from the C# decomp.
use super::prelude::*;

/// Scourge (Common, 1 energy, Skill). Apply 13 Doom to the target enemy,
/// then draw 1 card. Upgrade: Doom +3 (->16), Cards +1 (->2).
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Scourge.cs
pub(crate) fn play_scourge(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    // C# OnPlay: PowerCmd.Apply<DoomPower>(target, Doom.BaseValue), then
    // CardPileCmd.Draw(Cards.BaseValue). OnUpgrade bumps Doom by 3 and Cards
    // by 1 (PowerVar/CardsVar), gated on current_card_upgraded.
    let doom = if current_card_upgraded(state) { 16 } else { 13 };

    let t = target.expect("Scourge: AnyEnemy-targeted");
    apply_power_to_enemy(state, t, PowerKind::Doom, doom);
    // `Cards.BaseValue` is read after the awaited Doom apply: Sleight of
    // Flesh's hit on it can kill the Dampen caster, whose removal re-upgrades
    // this card before the draw (loadout Necrobinder v0.111.0 8527).
    let cards = if current_card_upgraded(state) { 2 } else { 1 };
    let _ = sts2_core::cards::draw_via_state(state, cards);
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Scourge,
        name: "SCOURGE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade bumps only PowerVar (Doom) and CardsVar (Cards), both applied
    // in the handler via the current_card_upgraded branch — no spec-stat delta.
    None
}
