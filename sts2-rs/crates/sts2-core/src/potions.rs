//! Potion discriminant + POD — bead `sts2-rs-13q` (P7-6).
//!
//! Ports `sts2-rl-agent@01c094d:sts2_env/potions/{all,base,effects}.py`.
//! 63 potions total (20 Common / 20 Uncommon / 20 Rare / 3 Event/Token).
//!
//! ## Crate boundary
//!
//! [`PotionKind`] is the *runtime discriminant* used in
//! [`crate::PlayerCombatState::potions`] (the live slot inventory) and
//! in the [`crate::Potion`] arena POD (legacy uid-keyed handle). It is
//! distinct from `sts2_content::potions::PotionSpec`, which is the
//! *static template* carrying rarity / usage / target metadata. The
//! split mirrors `CardKind`↔`CardSpec` and `RelicKind`↔`RelicSpec`.

/// Maximum number of potion slots a player can hold simultaneously.
/// Mirrors Python's `MAX_POTIONS = 5` (`sts2_env/core/combat.py`); the
/// `PotionBelt` relic relaxes this at the pickup-rule layer, not in
/// the core data shape.
pub const MAX_POTIONS: usize = 5;

/// Character identity for character-locked potion gating. Mirrors the
/// `character_pool: str | None` field on Python's `PotionModel`
/// (`sts2_env/potions/base.py`). The string variants used in the
/// Python source are `"Ironclad" | "Silent" | "Defect" | "Regent" |
/// "Necrobinder"`.
///
/// There is deliberately no `Watcher` variant. STS2 has no Watcher class —
/// `../sts2-wiki/decompiled/v0.107.1/` contains zero references to one — and
/// the variant that used to sit here existed only "for API completeness"
/// against an old `sts2-rl-agent` pin. It had no starter row, no card pool and
/// no potion locks, so every match arm on it was dead code that still had to be
/// written, and it kept being mistaken for a real character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CharacterId {
    Ironclad,
    Silent,
    Defect,
    Regent,
    Necrobinder,
    /// The Fool — v0.113.0-only (`Models.Characters/Fool.cs`). Every
    /// Fool-specific path is gated on `GameVersion >= V0_113_0`; earlier
    /// builds never construct this variant.
    Fool,
}

/// Potion rarity bucket. Mirrors
/// `sts2-rl-agent@01c094d:sts2_env/core/enums.py::PotionRarity` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionRarity {
    None,
    Common,
    Uncommon,
    Rare,
    /// `RelicRarity.EVENT` in Python — drops from events (Foul, Glowwater).
    Event,
    /// `RelicRarity.TOKEN` in Python — `PotionShapedRock`.
    Token,
}

/// When a potion is drinkable. Mirrors `PotionUsageType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionUsageType {
    /// Only legal inside `PlayerPlayPhase`.
    CombatOnly,
    /// Legal inside combat AND on the map.
    AnyTime,
    /// Auto-fires when triggered (`FairyInABottle` on lethal damage).
    Automatic,
}

/// Required target shape for a potion. Mirrors `PotionTargetType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionTargetType {
    Self_,
    AnyEnemy,
    AllEnemies,
    AnyPlayer,
}

/// Potion discriminant. One variant per Python `PotionModel` registered
/// in `sts2_env/potions/all.py` at pin `01c094d` (63 total).
///
/// Variant order matches the Python file's `_r(...)` declaration order
/// so the (eventual) parity exporter walks the same sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum PotionKind {
    // ── Common (20) ─────────────────────────────────────────────────
    AttackPotion,
    BlockPotion,
    BloodPotion,
    ColorlessPotion,
    DexterityPotion,
    EnergyPotion,
    ExplosiveAmpoule,
    FirePotion,
    FlexPotion,
    FocusPotion,
    PoisonPotion,
    PotionOfDoom,
    PowerPotion,
    SkillPotion,
    SpeedPotion,
    StarPotion,
    StrengthPotion,
    SwiftPotion,
    VulnerablePotion,
    WeakPotion,
    // ── Uncommon (20) ──────────────────────────────────────────────
    Ashwater,
    BlessingOfTheForge,
    BoneBrew,
    Clarity,
    CunningPotion,
    CureAll,
    Duplicator,
    Fortifier,
    FyshOil,
    GamblersBrew,
    HeartOfIron,
    KingsCourage,
    LiquidBronze,
    PotionOfBinding,
    PotionOfCapacity,
    PowderedDemise,
    RadiantTincture,
    RegenPotion,
    StableSerum,
    TouchOfInsanity,
    // ── Rare (20) ──────────────────────────────────────────────────
    BeetleJuice,
    BottledPotential,
    CosmicConcoction,
    DistilledChaos,
    DropletOfPrecognition,
    EntropicBrew,
    EssenceOfDarkness,
    FairyInABottle,
    FruitJuice,
    GhostInAJar,
    GigantificationPotion,
    LiquidMemories,
    LuckyTonic,
    MazalethsGift,
    OrobicAcid,
    PotOfGhouls,
    ShacklingPotion,
    ShipInABottle,
    SneckoOil,
    SoldiersStew,
    // ── Event / Token (3) ──────────────────────────────────────────
    FoulPotion,
    GlowwaterPotion,
    PotionShapedRock,
    // ── v0.111.0-only (GH #792) ────────────────────────────────────
    // Not in [`PotionKind::ALL`], which stays the v0.107.1 universe (PPO
    // vocab, random pools, dispatch-coverage tests read it); see
    // [`PotionKind::V0111_ONLY`].
    /// v0.111.0 Event potion (`Potions/Ambergris.cs`): heal 50% max HP,
    /// plus an extra turn in combat. Procured by NeowsSacrifice.
    Ambergris,
    // ── v0.113.0-only: The Fool's potion pool (`Fool4Epoch.Potions`
    // order). Not in [`PotionKind::ALL`]; see [`PotionKind::V0113_ONLY`].
    /// v0.113.0 Fool Common (`Potions/PrecisionPotion.cs`): gain 21 Aim.
    PrecisionPotion,
    /// v0.113.0 Fool Uncommon (`Potions/BottledSong.cs`): one free-this-turn
    /// Wah, Cho and Ki melody card each.
    BottledSong,
    /// v0.113.0 Fool Rare (`Potions/CrystallineEuphoria.cs`): 3 Joy, then
    /// fill the hand with Sadness.
    CrystallineEuphoria,
}

impl PotionKind {
    /// Python `PotionId.<self>.name` — the string the parity exporter
    /// writes (future Phase-7 work). Upper-snake-case mirroring
    /// Python's `enum.auto()` member naming.
    #[must_use]
    pub const fn python_name(self) -> &'static str {
        match self {
            PotionKind::AttackPotion => "ATTACK_POTION",
            PotionKind::BlockPotion => "BLOCK_POTION",
            PotionKind::BloodPotion => "BLOOD_POTION",
            PotionKind::ColorlessPotion => "COLORLESS_POTION",
            PotionKind::DexterityPotion => "DEXTERITY_POTION",
            PotionKind::EnergyPotion => "ENERGY_POTION",
            PotionKind::ExplosiveAmpoule => "EXPLOSIVE_AMPOULE",
            PotionKind::FirePotion => "FIRE_POTION",
            PotionKind::FlexPotion => "FLEX_POTION",
            PotionKind::FocusPotion => "FOCUS_POTION",
            PotionKind::PoisonPotion => "POISON_POTION",
            PotionKind::PotionOfDoom => "POTION_OF_DOOM",
            PotionKind::PowerPotion => "POWER_POTION",
            PotionKind::SkillPotion => "SKILL_POTION",
            PotionKind::SpeedPotion => "SPEED_POTION",
            PotionKind::StarPotion => "STAR_POTION",
            PotionKind::StrengthPotion => "STRENGTH_POTION",
            PotionKind::SwiftPotion => "SWIFT_POTION",
            PotionKind::VulnerablePotion => "VULNERABLE_POTION",
            PotionKind::WeakPotion => "WEAK_POTION",
            PotionKind::Ashwater => "ASHWATER",
            PotionKind::BlessingOfTheForge => "BLESSING_OF_THE_FORGE",
            PotionKind::BoneBrew => "BONE_BREW",
            PotionKind::Clarity => "CLARITY",
            PotionKind::CunningPotion => "CUNNING_POTION",
            PotionKind::CureAll => "CURE_ALL",
            PotionKind::Duplicator => "DUPLICATOR",
            PotionKind::Fortifier => "FORTIFIER",
            PotionKind::FyshOil => "FYSH_OIL",
            PotionKind::GamblersBrew => "GAMBLERS_BREW",
            PotionKind::HeartOfIron => "HEART_OF_IRON",
            PotionKind::KingsCourage => "KINGS_COURAGE",
            PotionKind::LiquidBronze => "LIQUID_BRONZE",
            PotionKind::PotionOfBinding => "POTION_OF_BINDING",
            PotionKind::PotionOfCapacity => "POTION_OF_CAPACITY",
            PotionKind::PowderedDemise => "POWDERED_DEMISE",
            PotionKind::RadiantTincture => "RADIANT_TINCTURE",
            PotionKind::RegenPotion => "REGEN_POTION",
            PotionKind::StableSerum => "STABLE_SERUM",
            PotionKind::TouchOfInsanity => "TOUCH_OF_INSANITY",
            PotionKind::BeetleJuice => "BEETLE_JUICE",
            PotionKind::BottledPotential => "BOTTLED_POTENTIAL",
            PotionKind::CosmicConcoction => "COSMIC_CONCOCTION",
            PotionKind::DistilledChaos => "DISTILLED_CHAOS",
            PotionKind::DropletOfPrecognition => "DROPLET_OF_PRECOGNITION",
            PotionKind::EntropicBrew => "ENTROPIC_BREW",
            PotionKind::EssenceOfDarkness => "ESSENCE_OF_DARKNESS",
            PotionKind::FairyInABottle => "FAIRY_IN_A_BOTTLE",
            PotionKind::FruitJuice => "FRUIT_JUICE",
            PotionKind::GhostInAJar => "GHOST_IN_A_JAR",
            PotionKind::GigantificationPotion => "GIGANTIFICATION_POTION",
            PotionKind::LiquidMemories => "LIQUID_MEMORIES",
            PotionKind::LuckyTonic => "LUCKY_TONIC",
            PotionKind::MazalethsGift => "MAZALETHS_GIFT",
            PotionKind::OrobicAcid => "OROBIC_ACID",
            PotionKind::PotOfGhouls => "POT_OF_GHOULS",
            PotionKind::ShacklingPotion => "SHACKLING_POTION",
            PotionKind::ShipInABottle => "SHIP_IN_A_BOTTLE",
            PotionKind::SneckoOil => "SNECKO_OIL",
            PotionKind::SoldiersStew => "SOLDIERS_STEW",
            PotionKind::FoulPotion => "FOUL_POTION",
            PotionKind::GlowwaterPotion => "GLOWWATER_POTION",
            PotionKind::PotionShapedRock => "POTION_SHAPED_ROCK",
            PotionKind::Ambergris => "AMBERGRIS",
            PotionKind::PrecisionPotion => "PRECISION_POTION",
            PotionKind::BottledSong => "BOTTLED_SONG",
            PotionKind::CrystallineEuphoria => "CRYSTALLINE_EUPHORIA",
        }
    }

    /// Character lock — `Some(c)` iff this potion may only be drunk
    /// by character `c`. Mirrors the `character_pool: str | None`
    /// field on Python's `PotionModel` (`sts2_env/potions/base.py`);
    /// values are scraped from `sts2_env/potions/all.py` at pin
    /// `01c094d`. Drives the `legal_actions` gate for
    /// [`crate::Action::DrinkPotion`].
    #[must_use]
    pub const fn character_lock(self) -> Option<CharacterId> {
        match self {
            // Ironclad
            PotionKind::BloodPotion | PotionKind::Ashwater | PotionKind::SoldiersStew => {
                Some(CharacterId::Ironclad)
            }
            // Silent
            PotionKind::PoisonPotion | PotionKind::CunningPotion | PotionKind::GhostInAJar => {
                Some(CharacterId::Silent)
            }
            // Defect
            PotionKind::FocusPotion
            | PotionKind::PotionOfCapacity
            | PotionKind::EssenceOfDarkness => Some(CharacterId::Defect),
            // Regent
            PotionKind::StarPotion | PotionKind::KingsCourage | PotionKind::CosmicConcoction => {
                Some(CharacterId::Regent)
            }
            // Necrobinder
            PotionKind::PotionOfDoom | PotionKind::BoneBrew | PotionKind::PotOfGhouls => {
                Some(CharacterId::Necrobinder)
            }
            // The Fool (v0.113.0, `FoolPotionPool`)
            PotionKind::PrecisionPotion
            | PotionKind::BottledSong
            | PotionKind::CrystallineEuphoria => Some(CharacterId::Fool),
            _ => None,
        }
    }

    /// `true` for `PotionUsage.Automatic` potions — potions the player
    /// can never drink manually; they fire themselves from an engine
    /// trigger (Fairy in a Bottle's death-prevention in
    /// `combat_loop::try_fairy_in_a_bottle_prevent_death`). C#
    /// `FairyInABottle.OnUse` is documented as "should never be
    /// executed directly by `UsePotionAction`" and the UI disables the
    /// drink button (`NPotionPopup.cs:273`); Fairy is the ONLY
    /// `Automatic` potion in the v0.107.1 decomp. Kept in lockstep with
    /// the content-side spec table by
    /// `sts2-content::potions::all` (`automatic_spec_agrees_with_core
    /// _predicate`) — extend BOTH if a new Automatic potion ever lands.
    /// A8 audit §7a (2026-08-09): manual Fairy drinks wasted the revive.
    #[must_use]
    pub const fn is_automatic(self) -> bool {
        matches!(self, PotionKind::FairyInABottle)
    }

    /// `true` for the three event-only / token rarities
    /// (`FoulPotion`, `GlowwaterPotion`, `PotionShapedRock`) and for
    /// any potion whose Python source carried
    /// `can_be_generated_in_combat=False` (`RegenPotion`,
    /// `FairyInABottle`, `FruitJuice`). Used by the (eventual)
    /// reward-roller to filter the
    /// common drop pool.
    #[must_use]
    pub const fn is_event_only(self) -> bool {
        matches!(
            self,
            PotionKind::FoulPotion
                | PotionKind::GlowwaterPotion
                | PotionKind::PotionShapedRock
                | PotionKind::Ambergris
                | PotionKind::RegenPotion
                | PotionKind::FairyInABottle
                | PotionKind::FruitJuice
        )
    }

    /// All 63 potion kinds in Python declaration order. Load-bearing
    /// for the per-potion dispatch unit test (macro-generated in
    /// `sts2-content::potions::dispatch`).
    pub const ALL: [PotionKind; 63] = [
        PotionKind::AttackPotion,
        PotionKind::BlockPotion,
        PotionKind::BloodPotion,
        PotionKind::ColorlessPotion,
        PotionKind::DexterityPotion,
        PotionKind::EnergyPotion,
        PotionKind::ExplosiveAmpoule,
        PotionKind::FirePotion,
        PotionKind::FlexPotion,
        PotionKind::FocusPotion,
        PotionKind::PoisonPotion,
        PotionKind::PotionOfDoom,
        PotionKind::PowerPotion,
        PotionKind::SkillPotion,
        PotionKind::SpeedPotion,
        PotionKind::StarPotion,
        PotionKind::StrengthPotion,
        PotionKind::SwiftPotion,
        PotionKind::VulnerablePotion,
        PotionKind::WeakPotion,
        PotionKind::Ashwater,
        PotionKind::BlessingOfTheForge,
        PotionKind::BoneBrew,
        PotionKind::Clarity,
        PotionKind::CunningPotion,
        PotionKind::CureAll,
        PotionKind::Duplicator,
        PotionKind::Fortifier,
        PotionKind::FyshOil,
        PotionKind::GamblersBrew,
        PotionKind::HeartOfIron,
        PotionKind::KingsCourage,
        PotionKind::LiquidBronze,
        PotionKind::PotionOfBinding,
        PotionKind::PotionOfCapacity,
        PotionKind::PowderedDemise,
        PotionKind::RadiantTincture,
        PotionKind::RegenPotion,
        PotionKind::StableSerum,
        PotionKind::TouchOfInsanity,
        PotionKind::BeetleJuice,
        PotionKind::BottledPotential,
        PotionKind::CosmicConcoction,
        PotionKind::DistilledChaos,
        PotionKind::DropletOfPrecognition,
        PotionKind::EntropicBrew,
        PotionKind::EssenceOfDarkness,
        PotionKind::FairyInABottle,
        PotionKind::FruitJuice,
        PotionKind::GhostInAJar,
        PotionKind::GigantificationPotion,
        PotionKind::LiquidMemories,
        PotionKind::LuckyTonic,
        PotionKind::MazalethsGift,
        PotionKind::OrobicAcid,
        PotionKind::PotOfGhouls,
        PotionKind::ShacklingPotion,
        PotionKind::ShipInABottle,
        PotionKind::SneckoOil,
        PotionKind::SoldiersStew,
        PotionKind::FoulPotion,
        PotionKind::GlowwaterPotion,
        PotionKind::PotionShapedRock,
    ];

    /// Potions that exist only in game build v0.111.0 (GH #792).
    pub const V0111_ONLY: [PotionKind; 1] = [PotionKind::Ambergris];

    /// Potions that exist only from game build v0.113.0 on: The Fool's pool,
    /// in `Fool4Epoch.Potions` order (`FoolPotionPool.GenerateAllPotions`).
    pub const V0113_ONLY: [PotionKind; 3] = [
        PotionKind::PrecisionPotion,
        PotionKind::BottledSong,
        PotionKind::CrystallineEuphoria,
    ];
}

/// Player-facing potion-slot index. Mirrors the existing `Option<u8>`
/// target shape used by `Action::PlayCard` / `Action::UsePotion`.
pub type TargetIndex = u8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_has_63_entries() {
        assert_eq!(PotionKind::ALL.len(), 63);
    }

    #[test]
    fn python_names_are_unique_and_upper_snake() {
        let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        for &k in &PotionKind::ALL {
            let name = k.python_name();
            assert!(seen.insert(name), "duplicate name {name}");
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()),
                "non upper-snake-case for {k:?}: {name}"
            );
        }
    }

    #[test]
    fn potion_kind_is_copy_eq_hash() {
        fn assert_bounds<T: Copy + Eq + std::hash::Hash>() {}
        assert_bounds::<PotionKind>();
        assert_bounds::<PotionRarity>();
        assert_bounds::<PotionUsageType>();
        assert_bounds::<PotionTargetType>();
        assert_bounds::<CharacterId>();
    }
}
