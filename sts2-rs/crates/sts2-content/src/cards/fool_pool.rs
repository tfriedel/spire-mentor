//! The Fool's card pool glue (v0.113.0).
//!
//! The Fool's cards are ported by the FOOL CARDS lane; this module is the
//! single seam the character-pool consumers (`spec_index`, `pool_order`, the
//! combat-generatable pools, starter deck, Kaleidoscope / Splash / Colorful
//! Philosophers) read. Every function returns the empty pool until the card
//! ids land, so a Fool run degrades to "no class cards" rather than to
//! another character's pool.
//!
//! Source of truth: `Models.CardPools/FoolCardPool.cs` (pool, decomp order)
//! and `Models.Characters/Fool.cs` (starting deck).

use super::types::CardSpec;
use sts2_core::cards::CardId;

/// `FoolCardPool.GenerateAllCards()` as spec factories, in decomp order:
/// the first 90 entries of the cards lane's `fool::specs::FOOL_SPECS`
/// (which is in `fool::FOOL_CARDS` order = the pool, then the tokens).
/// Pinned by `pool_specs_match_pool_order`.
pub const FOOL_SPECS: &[fn() -> CardSpec] = super::fool::specs::FOOL_SPECS
    .split_at(super::fool::FOOL_POOL_V0113.len())
    .0;

/// `FoolCardPool.GenerateAllCards()` as ids, in decomp order (the
/// `pool_order::generate_all_cards` list for `Pool::Character(Fool)`).
#[must_use]
pub fn generate_all_cards() -> &'static [CardId] {
    &super::fool::FOOL_POOL_V0113
}

/// The Fool's Basic Strike / Defend pair (`StrikeFool`, `DefendFool`), for
/// the transform-a-Strike/Defend relics.
#[must_use]
pub fn basic_strike_defend() -> Option<[CardId; 2]> {
    Some([CardId::StrikeFool, CardId::DefendFool])
}

/// `Fool.StartingDeck` (`Models.Characters/Fool.cs`): 5 StrikeFool,
/// 4 DefendFool, FoolishShot.
#[must_use]
pub fn starter_deck() -> Vec<CardSpec> {
    use super::fool::specs::{defend_fool, foolish_shot, strike_fool};
    let mut deck = Vec::with_capacity(10);
    deck.extend(std::iter::repeat_n(strike_fool(), 5));
    deck.extend(std::iter::repeat_n(defend_fool(), 4));
    deck.push(foolish_shot());
    deck
}

/// The Wah/Cho/Ki cards of `UnlockState.CharacterCardPools` that survive
/// `CardFactory.FilterForCombat` (no Basic/Ancient/Event rarity,
/// `CanBeGeneratedInCombat`), in pool order, optionally narrowed to one
/// melody `tag`. Only the Fool pool carries melody tags, so the other
/// characters' pools contribute nothing. Ocarina and BottledSong
/// (`GetDistinctForCombat` over this list) read it.
#[must_use]
pub fn combat_melody_pool(
    tag: Option<sts2_core::cards::CardTag>,
) -> smallvec::SmallVec<[CardId; 48]> {
    use super::types::CardRarity;
    FOOL_SPECS
        .iter()
        .map(|f| f())
        .filter(|s| {
            let tags = super::status::helpers::tags_for_card_id(s.id);
            tags.has_melody()
                && tag.is_none_or(|t| tags.contains(t))
                && !matches!(
                    s.rarity,
                    CardRarity::Basic | CardRarity::Ancient | CardRarity::Event
                )
                && super::colorless::helpers::card_can_be_generated_in_combat(s.id)
                && !super::spec_index::is_multiplayer_only(s.id)
        })
        .map(|s| s.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_specs_match_pool_order() {
        let ids: Vec<CardId> = FOOL_SPECS.iter().map(|f| f().id).collect();
        assert_eq!(ids, super::super::fool::FOOL_POOL_V0113.to_vec());
    }

    #[test]
    fn starter_deck_is_five_strikes_four_defends_one_shot() {
        let ids: Vec<CardId> = starter_deck().iter().map(|s| s.id).collect();
        assert_eq!(ids.iter().filter(|&&c| c == CardId::StrikeFool).count(), 5);
        assert_eq!(ids.iter().filter(|&&c| c == CardId::DefendFool).count(), 4);
        assert_eq!(ids.last(), Some(&CardId::FoolishShot));
        assert_eq!(ids.len(), 10);
    }
}
