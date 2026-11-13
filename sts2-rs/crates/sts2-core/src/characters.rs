//! Per-character starter pack registry.
//!
//! Mirrors `_CHARACTER_CONFIG` in
//! `sts2-rl-agent@01c094d:sts2_env/run/run_manager.py:74-105` plus the
//! `_get_starter_deck` dispatch at `:111-130`. The registry holds the
//! *metadata* portion (starting HP/gold/starter-relic/post-combat heal
//! amount); the starter-deck card data lives next to the card factories
//! in `sts2-content` (e.g.
//! [`sts2_content::cards::ironclad_basic::ironclad_starter_deck`]) since
//! `sts2-core` deliberately doesn't depend on `sts2-content`.
//!
//! Consumers (the in-flight `enter_combat(rs, encounter)` bridge for the
//! P9.0 thin slice — bead `sts2-rs-tjn0v`) compose the two: call
//! [`character_starter`] for HP/gold/starter-relic, then call into
//! `sts2-content`'s per-character deck factory for the master_deck seed.
//!
//! Bead `sts2-rs-jjcb8` (P9.0.a).

use crate::potions::CharacterId;
use crate::RelicId;

/// Per-character starter metadata.
///
/// Ports the value subset of Python's `_CHARACTER_CONFIG[character]` dict
/// that the run-loop reads at run-creation and at post-combat heal time.
/// Per pin `01c094d`, every character starts with **gold = 99** (`gold`
/// is kept on the struct for parity with the Python schema even though it
/// is currently invariant across characters).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CharacterStarter {
    /// Starting max HP. Mirrors `_CHARACTER_CONFIG[char]["hp"]`.
    pub hp: i32,
    /// Starting gold. Mirrors `_CHARACTER_CONFIG[char]["gold"]`.
    pub gold: i32,
    /// Starter relic granted before the run's first room. Mirrors
    /// `_CHARACTER_CONFIG[char]["starter_relic"]`.
    pub starter_relic: RelicId,
    /// HP healed at the end of every non-boss combat. Mirrors
    /// `_CHARACTER_CONFIG[char]["heal_after_combat"]` (only Ironclad's
    /// Burning Blood grants 6; the other characters' starter relics
    /// confer 0 post-combat heal).
    pub heal_after_combat: i32,
}

/// Look up the starter pack for a character.
///
/// Every [`CharacterId`] has a real row, so this is total with no fallback
/// arm. It used to carry one for a sixth variant STS2 does not have.
#[must_use]
pub const fn character_starter(character: CharacterId) -> CharacterStarter {
    match character {
        // Ironclad: HP 80, BurningBlood, +6 heal after each non-boss combat.
        CharacterId::Ironclad => CharacterStarter {
            hp: 80,
            gold: 99,
            starter_relic: RelicId::BurningBlood,
            heal_after_combat: 6,
        },
        // Silent: HP 70, RingOfTheSnake, no post-combat heal.
        CharacterId::Silent => CharacterStarter {
            hp: 70,
            gold: 99,
            starter_relic: RelicId::RingOfTheSnake,
            heal_after_combat: 0,
        },
        // Defect: HP 75, CrackedCore.
        CharacterId::Defect => CharacterStarter {
            hp: 75,
            gold: 99,
            starter_relic: RelicId::CrackedCore,
            heal_after_combat: 0,
        },
        // Necrobinder: HP 66 (C# Necrobinder.StartingHp => 66), BoundPhylactery.
        // The prior value (75) was Python-pin drift; corrected against the C#
        // decomp during the from-scratch re-port.
        CharacterId::Necrobinder => CharacterStarter {
            hp: 66,
            gold: 99,
            starter_relic: RelicId::BoundPhylactery,
            heal_after_combat: 0,
        },
        // Regent: HP 75, DivineRight.
        CharacterId::Regent => CharacterStarter {
            hp: 75,
            gold: 99,
            starter_relic: RelicId::DivineRight,
            heal_after_combat: 0,
        },
        // The Fool (v0.113.0): C# `Fool.cs` StartingHp => 77,
        // StartingGold => 99, StartingRelics => SongBow.
        CharacterId::Fool => CharacterStarter {
            hp: 77,
            gold: 99,
            starter_relic: RelicId::SongBow,
            heal_after_combat: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ironclad_starter_matches_python_pin() {
        let s = character_starter(CharacterId::Ironclad);
        assert_eq!(s.hp, 80);
        assert_eq!(s.gold, 99);
        assert_eq!(s.starter_relic, RelicId::BurningBlood);
        assert_eq!(s.heal_after_combat, 6);
    }

    #[test]
    fn silent_starter_matches_python_pin() {
        let s = character_starter(CharacterId::Silent);
        assert_eq!(s.hp, 70);
        assert_eq!(s.gold, 99);
        assert_eq!(s.starter_relic, RelicId::RingOfTheSnake);
        assert_eq!(s.heal_after_combat, 0);
    }

    #[test]
    fn defect_starter_matches_python_pin() {
        let s = character_starter(CharacterId::Defect);
        assert_eq!(s.hp, 75);
        assert_eq!(s.starter_relic, RelicId::CrackedCore);
        assert_eq!(s.heal_after_combat, 0);
    }

    #[test]
    fn necrobinder_starter_matches_csharp() {
        // C# Necrobinder.cs: StartingHp => 66, StartingRelics => BoundPhylactery.
        let s = character_starter(CharacterId::Necrobinder);
        assert_eq!(s.hp, 66);
        assert_eq!(s.starter_relic, RelicId::BoundPhylactery);
        assert_eq!(s.heal_after_combat, 0);
    }

    #[test]
    fn regent_starter_matches_python_pin() {
        let s = character_starter(CharacterId::Regent);
        assert_eq!(s.hp, 75);
        assert_eq!(s.starter_relic, RelicId::DivineRight);
        assert_eq!(s.heal_after_combat, 0);
    }

    #[test]
    fn fool_starter_matches_csharp() {
        // C# Fool.cs: StartingHp => 77, StartingGold => 99, SongBow.
        let s = character_starter(CharacterId::Fool);
        assert_eq!(s.hp, 77);
        assert_eq!(s.gold, 99);
        assert_eq!(s.starter_relic, RelicId::SongBow);
        assert_eq!(s.heal_after_combat, 0);
    }

    #[test]
    fn every_character_starts_with_99_gold() {
        for c in [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Necrobinder,
            CharacterId::Regent,
        ] {
            assert_eq!(character_starter(c).gold, 99, "{c:?} gold");
        }
    }

    #[test]
    fn const_eval_works() {
        // Compile-time call so downstream code can use the registry in
        // `const fn` contexts (e.g. as a default value).
        const IC: CharacterStarter = character_starter(CharacterId::Ironclad);
        assert_eq!(IC.hp, 80);
    }
}
