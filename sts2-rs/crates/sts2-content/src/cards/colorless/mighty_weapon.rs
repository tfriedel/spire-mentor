//! Mighty Weapon — v0.113.0-only event card (Shrine of the Warlord).
use super::prelude::*;

/// Dazed added per play (`CardsVar(2)`, `MightyWeapon.cs`).
const MIGHTY_WEAPON_DAZED: usize = 2;

/// Mighty Weapon (Event, 1 energy, Attack, `AllEnemies`) — v0.113.0
/// `Cards/MightyWeapon.cs`: 25 damage to all opponents (+5 upgraded), then
/// two `CreateCard<Dazed>` each `AddGeneratedCardToCombat(.., Draw, ..,
/// CardPilePosition.Random)` — one `Rng.Shuffle` read per Dazed.
pub(crate) fn play_mighty_weapon(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let dmg = current_damage(state);
    deal_damage_all_enemies(state, dmg);
    add_dazed_to_draw_random(state, MIGHTY_WEAPON_DAZED);
    EffectResult::Resolved
}

/// `count` x `AddGeneratedCardToCombat(Dazed, Draw, Random)`.
pub(crate) fn add_dazed_to_draw_random(state: &mut CombatState, count: usize) {
    for _ in 0..count {
        if let Some(card) = crate::cards::spec_index::card_pod_from_id(CardId::Dazed, false) {
            let _ = sts2_core::card_effects::spawn_card_into_draw_pile_random(state, card);
        }
    }
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::MightyWeapon,
        name: "MIGHTY_WEAPON",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Event,
        base_damage: 25,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}
