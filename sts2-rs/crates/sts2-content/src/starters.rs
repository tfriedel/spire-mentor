//! Content-side starter-deck dispatcher.
//!
//! Pairs with `sts2_core::characters::character_starter` (the per-character
//! HP/gold/starter-relic metadata registry, bead `sts2-rs-jjcb8`) to give
//! the run-loop driver a single composed accessor for everything it needs
//! to seed a fresh run:
//!
//! ```ignore
//! let meta = sts2_core::characters::character_starter(CharacterId::Ironclad);
//! let deck = sts2_content::starters::starter_deck_for(CharacterId::Ironclad);
//! run_state.player.max_hp     = meta.hp;
//! run_state.player.current_hp = meta.hp;
//! run_state.gold              = meta.gold;
//! run_state.relics.push(meta.starter_relic);
//! run_state.master_deck       = deck.iter().map(|s| ... ).collect();
//! ```
//!
//! Mirrors Python `_get_starter_deck(character_id)` at
//! `sts2-rl-agent@01c094d:sts2_env/run/run_manager.py:111-130`, including
//! the fallback to the Ironclad deck for unknown character ids.
//!
//! Lives in `sts2-content` (not `sts2-core`) because the per-character
//! starter-deck factories live next to their card pools here; the dep
//! direction `content → core` forbids `sts2-core` from importing them
//! directly.
//!
//! Bead `sts2-rs-oce9m` (P9.2.l).

use sts2_core::potions::CharacterId;

use crate::cards::types::CardSpec;
use crate::cards::{defect, ironclad_basic, necrobinder, regent, silent};

/// Return the starter deck for `character`.
///
/// Mirrors the Python dispatch `_get_starter_deck`
/// (`run_manager.py:111-130`). Every character has a real factory —
/// `Defect`'s landed with GH#29 — so there is no fallback arm.
///
/// Returned [`CardSpec`] order matches each character's factory; the
/// run-loop is responsible for `RunState`-side ascension adjustments
/// (e.g. the A5 `AscendersBane` append, handled by
/// `RunManager::with_ascension` at the run-state seam — bead
/// `sts2-rs-9vypz`).
#[must_use]
pub fn starter_deck_for(character: CharacterId) -> Vec<CardSpec> {
    match character {
        CharacterId::Ironclad => ironclad_basic::ironclad_starter_deck(),
        CharacterId::Silent => silent::silent_starter_deck(),
        CharacterId::Regent => regent::regent_starter_deck(),
        // Necrobinder: re-ported starter deck (StrikeNecrobinder ×4,
        // DefendNecrobinder ×4, Bodyguard, Unleash — per C# Necrobinder.StartingDeck).
        CharacterId::Necrobinder => necrobinder::necrobinder_starter_deck(),
        // Defect: StrikeDefect ×4, DefendDefect ×4, Zap, Dualcast (GH#29 —
        // previously fell back to the Ironclad deck, diverging every Defect
        // run at the first event).
        CharacterId::Defect => defect::defect_starter_deck(),
        // The Fool (v0.113.0): `Fool.StartingDeck`.
        CharacterId::Fool => crate::cards::fool_pool::starter_deck(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::types::CardRarity;

    /// Every playable character returns a non-empty deck. Mirrors
    /// Python `_get_starter_deck`'s contract that every branch
    /// returns a populated list.
    #[test]
    fn every_character_returns_nonempty_deck() {
        for c in [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Necrobinder,
            CharacterId::Regent,
        ] {
            assert!(
                !starter_deck_for(c).is_empty(),
                "{c:?} starter deck is empty"
            );
        }
    }

    /// Ironclad starter matches the canonical 10-card pile
    /// (5×Strike, 4×Defend, 1×Bash) — same shape the
    /// `ironclad_basic::ironclad_starter_deck` unit tests already
    /// pin. This is the smoke test for the dispatcher wiring.
    #[test]
    fn ironclad_dispatch_routes_to_ironclad_factory() {
        let via_dispatch = starter_deck_for(CharacterId::Ironclad);
        let direct = ironclad_basic::ironclad_starter_deck();
        assert_eq!(via_dispatch.len(), direct.len());
    }

    /// Silent dispatch goes to the Silent factory, not the
    /// Ironclad fallback. Distinguished by deck size: Silent's
    /// starter is 10 cards (5×Strike, 5×Defend) but uses different
    /// `CardId`s — the dispatch test here just confirms the
    /// factory routing by spec equality.
    #[test]
    fn silent_dispatch_routes_to_silent_factory() {
        let via_dispatch = starter_deck_for(CharacterId::Silent);
        let direct = silent::silent_starter_deck();
        assert_eq!(via_dispatch.len(), direct.len());
        assert_eq!(
            via_dispatch.iter().map(|s| s.id).collect::<Vec<_>>(),
            direct.iter().map(|s| s.id).collect::<Vec<_>>(),
        );
    }

    /// Defect dispatch routes to the Defect factory (GH#29) — the
    /// canonical 10-card pile 4×StrikeDefect, 4×DefendDefect, Zap,
    /// Dualcast — NOT the Ironclad fallback it used to hit.
    #[test]
    fn defect_dispatch_routes_to_defect_factory() {
        use sts2_core::cards::CardId;
        let deck = starter_deck_for(CharacterId::Defect);
        assert_eq!(deck.len(), 10, "Defect starter is 10 cards");
        let mut counts: std::collections::BTreeMap<CardId, usize> =
            std::collections::BTreeMap::new();
        for spec in &deck {
            *counts.entry(spec.id).or_default() += 1;
        }
        assert_eq!(counts.get(&CardId::StrikeDefect).copied(), Some(4));
        assert_eq!(counts.get(&CardId::DefendDefect).copied(), Some(4));
        assert_eq!(counts.get(&CardId::Zap).copied(), Some(1));
        assert_eq!(counts.get(&CardId::Dualcast).copied(), Some(1));
        // Regression guard: must NOT be the Ironclad fallback.
        assert!(
            !deck.iter().any(|s| s.id == CardId::StrikeIronclad),
            "Defect deck must not contain Ironclad cards (GH#29)"
        );
    }

    /// Every starter-deck card is `Basic` rarity — Python's
    /// `_get_starter_deck` doesn't mix in higher-rarity cards.
    /// Acts as a regression check against accidentally pointing a
    /// starter factory at a content-side pool slice that includes
    /// non-Basic cards.
    #[test]
    fn every_starter_card_is_basic_rarity() {
        for c in [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Necrobinder,
            CharacterId::Regent,
        ] {
            for spec in starter_deck_for(c) {
                assert_eq!(
                    spec.rarity,
                    CardRarity::Basic,
                    "{c:?} starter card {:?} is not Basic rarity",
                    spec.id,
                );
            }
        }
    }
}
