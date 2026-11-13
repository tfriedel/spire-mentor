//! Mighty Shield — v0.113.0-only event card (Shrine of the Warlord).
use super::prelude::*;

/// Dazed added per play (`CardsVar(3)`, `MightyShield.cs`).
const MIGHTY_SHIELD_DAZED: usize = 3;

/// Mighty Shield (Event, 1 energy, Skill, Self) — v0.113.0
/// `Cards/MightyShield.cs`: gain 20 block (+5 upgraded), then three Dazed
/// shuffled into the draw pile at random positions.
pub(crate) fn play_mighty_shield(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    gain_block_player(state, current_block(state));
    super::mighty_weapon::add_dazed_to_draw_random(state, MIGHTY_SHIELD_DAZED);
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::MightyShield,
        name: "MIGHTY_SHIELD",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 20,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}
