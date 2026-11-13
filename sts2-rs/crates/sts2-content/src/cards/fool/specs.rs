//! `CardSpec` POD factories for The Fool's cards (v0.113.0): cost / type /
//! rarity / target / keywords / base damage / base block, read off each card's
//! constructor, `CanonicalKeywords` and `DamageVar` / `BlockVar` in
//! `Models.Cards/<Name>.cs` in the v0.113.0 decompilation.
//! `HasEnergyCostX` cards and the unplayable status carry cost -1. Ally-
//! targeted cards (all `MultiplayerOnly`) map to `Self_` like the other
//! single-player ally cards. Upgrade deltas are in [`upgrade_delta`].

use sts2_core::cards::{CardId, CardType, TargetType};

use crate::cards::types::{CardKeywords, CardRarity, CardSpec, CardUpgradeDelta};

#[must_use]
pub const fn all_is_well() -> CardSpec {
    CardSpec {
        id: CardId::AllIsWellCard,
        name: "ALL_IS_WELL",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn anguish() -> CardSpec {
    CardSpec {
        id: CardId::Anguish,
        name: "ANGUISH",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn archery() -> CardSpec {
    CardSpec {
        id: CardId::ArcheryCard,
        name: "ARCHERY",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn arrow_stab() -> CardSpec {
    CardSpec {
        id: CardId::ArrowStab,
        name: "ARROW_STAB",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 9,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn blossom_shot() -> CardSpec {
    CardSpec {
        id: CardId::BlossomShot,
        name: "BLOSSOM_SHOT",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 11,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn blot_out_the_sky() -> CardSpec {
    CardSpec {
        id: CardId::BlotOutTheSky,
        name: "BLOT_OUT_THE_SKY",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn bounce() -> CardSpec {
    CardSpec {
        id: CardId::Bounce,
        name: "BOUNCE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn bow_smack() -> CardSpec {
    CardSpec {
        id: CardId::BowSmack,
        name: "BOW_SMACK",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 8,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn brace() -> CardSpec {
    CardSpec {
        id: CardId::Brace,
        name: "BRACE",
        cost: -1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn brighten_up() -> CardSpec {
    CardSpec {
        id: CardId::BrightenUp,
        name: "BRIGHTEN_UP",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_barbs() -> CardSpec {
    CardSpec {
        id: CardId::ChoBarbsCard,
        name: "CHO_BARBS",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 7,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_chorus() -> CardSpec {
    CardSpec {
        id: CardId::ChoChorus,
        name: "CHO_CHORUS",
        cost: 3,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 12,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_firm() -> CardSpec {
    CardSpec {
        id: CardId::ChoFirm,
        name: "CHO_FIRM",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 6,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_ki() -> CardSpec {
    CardSpec {
        id: CardId::ChoKi,
        name: "CHO_KI",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 8,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_limbs() -> CardSpec {
    CardSpec {
        id: CardId::ChoLimbs,
        name: "CHO_LIMBS",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 5,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn cho_ready() -> CardSpec {
    CardSpec {
        id: CardId::ChoReady,
        name: "CHO_READY",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 8,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn coda() -> CardSpec {
    CardSpec {
        id: CardId::Coda,
        name: "CODA",
        cost: 6,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 40,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn compost() -> CardSpec {
    CardSpec {
        id: CardId::CompostCard,
        name: "COMPOST",
        cost: 0,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn covering_fire() -> CardSpec {
    CardSpec {
        id: CardId::CoveringFireCard,
        name: "COVERING_FIRE",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn deep_stretch() -> CardSpec {
    CardSpec {
        id: CardId::DeepStretch,
        name: "DEEP_STRETCH",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn defend_fool() -> CardSpec {
    CardSpec {
        id: CardId::DefendFool,
        name: "DEFEND_FOOL",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Basic,
        base_damage: 0,
        base_block: 5,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn dislike_you() -> CardSpec {
    CardSpec {
        id: CardId::DislikeYouCard,
        name: "DISLIKE_YOU",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 8,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn divine_shot() -> CardSpec {
    CardSpec {
        id: CardId::DivineShot,
        name: "DIVINE_SHOT",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Ancient,
        base_damage: 40,
        base_block: 0,
        keywords: CardKeywords::RETAIN,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ecstasy() -> CardSpec {
    CardSpec {
        id: CardId::EcstasyCard,
        name: "ECSTASY",
        cost: 2,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn encore() -> CardSpec {
    CardSpec {
        id: CardId::EncoreCard,
        name: "ENCORE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ennui() -> CardSpec {
    CardSpec {
        id: CardId::EnnuiCard,
        name: "ENNUI",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn enrapture() -> CardSpec {
    CardSpec {
        id: CardId::Enrapture,
        name: "ENRAPTURE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ensemble() -> CardSpec {
    CardSpec {
        id: CardId::EnsembleCard,
        name: "ENSEMBLE",
        cost: 3,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn epic_retrieval() -> CardSpec {
    CardSpec {
        id: CardId::EpicRetrieval,
        name: "EPIC_RETRIEVAL",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 15,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn explosive_arrows() -> CardSpec {
    CardSpec {
        id: CardId::ExplosiveArrowsCard,
        name: "EXPLOSIVE_ARROWS",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn florescence() -> CardSpec {
    CardSpec {
        id: CardId::Florescence,
        name: "FLORESCENCE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn growth() -> CardSpec {
    CardSpec {
        id: CardId::Growth,
        name: "GROWTH",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn harmonic_shield() -> CardSpec {
    CardSpec {
        id: CardId::HarmonicShield,
        name: "HARMONIC_SHIELD",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn heavy_shot() -> CardSpec {
    CardSpec {
        id: CardId::HeavyShot,
        name: "HEAVY_SHOT",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 20,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn hidden_roots() -> CardSpec {
    CardSpec {
        id: CardId::HiddenRootsCard,
        name: "HIDDEN_ROOTS",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn hobbling_shot() -> CardSpec {
    CardSpec {
        id: CardId::HobblingShot,
        name: "HOBBLING_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 13,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn honing() -> CardSpec {
    CardSpec {
        id: CardId::HoningCard,
        name: "HONING",
        cost: 3,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn jest() -> CardSpec {
    CardSpec {
        id: CardId::Jest,
        name: "JEST",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 14,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ki_chant() -> CardSpec {
    CardSpec {
        id: CardId::KiChant,
        name: "KI_CHANT",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ki_hop() -> CardSpec {
    CardSpec {
        id: CardId::KiHop,
        name: "KI_HOP",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 7,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ki_rest() -> CardSpec {
    CardSpec {
        id: CardId::KiRest,
        name: "KI_REST",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ki_verse() -> CardSpec {
    CardSpec {
        id: CardId::KiVerseCard,
        name: "KI_VERSE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ki_whip() -> CardSpec {
    CardSpec {
        id: CardId::KiWhip,
        name: "KI_WHIP",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 5,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn locked_in() -> CardSpec {
    CardSpec {
        id: CardId::LockedIn,
        name: "LOCKED_IN",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn foolish_shot() -> CardSpec {
    CardSpec {
        id: CardId::FoolishShot,
        name: "FOOLISH_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Basic,
        base_damage: 15,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn lost_lute() -> CardSpec {
    CardSpec {
        id: CardId::LostLuteCard,
        name: "LOST_LUTE",
        cost: 3,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn lullaby() -> CardSpec {
    CardSpec {
        id: CardId::LullabyCard,
        name: "LULLABY",
        cost: 0,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn melodic_shot() -> CardSpec {
    CardSpec {
        id: CardId::MelodicShot,
        name: "MELODIC_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 12,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn motif() -> CardSpec {
    CardSpec {
        id: CardId::MotifCard,
        name: "MOTIF",
        cost: 2,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn mourning_blow() -> CardSpec {
    CardSpec {
        id: CardId::MourningBlow,
        name: "MOURNING_BLOW",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn mythic_form() -> CardSpec {
    CardSpec {
        id: CardId::MythicFormCard,
        name: "MYTHIC_FORM",
        cost: 3,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn nail() -> CardSpec {
    CardSpec {
        id: CardId::Nail,
        name: "NAIL",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 20,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn nock() -> CardSpec {
    CardSpec {
        id: CardId::Nock,
        name: "NOCK",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Common,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn punch_of_joy() -> CardSpec {
    CardSpec {
        id: CardId::PunchOfJoy,
        name: "PUNCH_OF_JOY",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 13,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn quick_shot() -> CardSpec {
    CardSpec {
        id: CardId::QuickShot,
        name: "QUICK_SHOT",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn quietus() -> CardSpec {
    CardSpec {
        id: CardId::QuietusCard,
        name: "QUIETUS",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn quiver() -> CardSpec {
    CardSpec {
        id: CardId::Quiver,
        name: "QUIVER",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rain_shot() -> CardSpec {
    CardSpec {
        id: CardId::RainShot,
        name: "RAIN_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 12,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rapid_shot() -> CardSpec {
    CardSpec {
        id: CardId::RapidShot,
        name: "RAPID_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 3,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn recall() -> CardSpec {
    CardSpec {
        id: CardId::Recall,
        name: "RECALL",
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

#[must_use]
pub const fn repeat_shot() -> CardSpec {
    CardSpec {
        id: CardId::RepeatShot,
        name: "REPEAT_SHOT",
        cost: -1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 15,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rhythm_strike() -> CardSpec {
    CardSpec {
        id: CardId::RhythmStrike,
        name: "RHYTHM_STRIKE",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 4,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rummage() -> CardSpec {
    CardSpec {
        id: CardId::Rummage,
        name: "RUMMAGE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn self_deprecate() -> CardSpec {
    CardSpec {
        id: CardId::SelfDeprecate,
        name: "SELF_DEPRECATE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn serrated_arrows() -> CardSpec {
    CardSpec {
        id: CardId::SerratedArrowsCard,
        name: "SERRATED_ARROWS",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn show_off() -> CardSpec {
    CardSpec {
        id: CardId::ShowOffCard,
        name: "SHOW_OFF",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn simpleton() -> CardSpec {
    CardSpec {
        id: CardId::SimpletonCard,
        name: "SIMPLETON",
        cost: 2,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn sky_shot() -> CardSpec {
    CardSpec {
        id: CardId::SkyShot,
        name: "SKY_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 30,
        base_block: 0,
        keywords: CardKeywords::EXHAUST.with(CardKeywords::INNATE),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn song_path() -> CardSpec {
    CardSpec {
        id: CardId::SongPathCard,
        name: "SONG_PATH",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn sorrowful_strike() -> CardSpec {
    CardSpec {
        id: CardId::SorrowfulStrike,
        name: "SORROWFUL_STRIKE",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 13,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn steady_cadence() -> CardSpec {
    CardSpec {
        id: CardId::SteadyCadence,
        name: "STEADY_CADENCE",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn strike_fool() -> CardSpec {
    CardSpec {
        id: CardId::StrikeFool,
        name: "STRIKE_FOOL",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Basic,
        base_damage: 6,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn timber() -> CardSpec {
    CardSpec {
        id: CardId::Timber,
        name: "TIMBER",
        cost: 3,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 20,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn trick_shot() -> CardSpec {
    CardSpec {
        id: CardId::TrickShot,
        name: "TRICK_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 12,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn tune() -> CardSpec {
    CardSpec {
        id: CardId::TuneCard,
        name: "TUNE",
        cost: 2,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn uproot() -> CardSpec {
    CardSpec {
        id: CardId::Uproot,
        name: "UPROOT",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn violent_chord() -> CardSpec {
    CardSpec {
        id: CardId::ViolentChordCard,
        name: "VIOLENT_CHORD",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn vitriol() -> CardSpec {
    CardSpec {
        id: CardId::Vitriol,
        name: "VITRIOL",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_blast() -> CardSpec {
    CardSpec {
        id: CardId::WahBlast,
        name: "WAH_BLAST",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Uncommon,
        base_damage: 14,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_cho() -> CardSpec {
    CardSpec {
        id: CardId::WahCho,
        name: "WAH_CHO",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 3,
        base_block: 3,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_cho_ki() -> CardSpec {
    CardSpec {
        id: CardId::WahChoKi,
        name: "WAH_CHO_KI",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 9,
        base_block: 9,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_hammer() -> CardSpec {
    CardSpec {
        id: CardId::WahHammer,
        name: "WAH_HAMMER",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 9,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_ki() -> CardSpec {
    CardSpec {
        id: CardId::WahKi,
        name: "WAH_KI",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 2,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_symphony() -> CardSpec {
    CardSpec {
        id: CardId::WahSymphony,
        name: "WAH_SYMPHONY",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_sweep() -> CardSpec {
    CardSpec {
        id: CardId::WahSweep,
        name: "WAH_SWEEP",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Common,
        base_damage: 8,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wah_touch() -> CardSpec {
    CardSpec {
        id: CardId::WahTouch,
        name: "WAH_TOUCH",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Common,
        base_damage: 6,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn whistling_arrows() -> CardSpec {
    CardSpec {
        id: CardId::WhistlingArrows,
        name: "WHISTLING_ARROWS",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 3,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wrest() -> CardSpec {
    CardSpec {
        id: CardId::Wrest,
        name: "WREST",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn yah() -> CardSpec {
    CardSpec {
        id: CardId::YahCard,
        name: "YAH",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn yoga() -> CardSpec {
    CardSpec {
        id: CardId::Yoga,
        name: "YOGA",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn death() -> CardSpec {
    CardSpec {
        id: CardId::Death,
        name: "DEATH",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn joy() -> CardSpec {
    CardSpec {
        id: CardId::Joy,
        name: "JOY",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn judgement() -> CardSpec {
    CardSpec {
        id: CardId::Judgement,
        name: "JUDGEMENT",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn justice() -> CardSpec {
    CardSpec {
        id: CardId::Justice,
        name: "JUSTICE",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 30,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn strength() -> CardSpec {
    CardSpec {
        id: CardId::StrengthCard,
        name: "STRENGTH",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn temperance() -> CardSpec {
    CardSpec {
        id: CardId::Temperance,
        name: "TEMPERANCE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 30,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_chariot() -> CardSpec {
    CardSpec {
        id: CardId::TheChariot,
        name: "THE_CHARIOT",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Token,
        base_damage: 25,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_devil() -> CardSpec {
    CardSpec {
        id: CardId::TheDevil,
        name: "THE_DEVIL",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_fool() -> CardSpec {
    CardSpec {
        id: CardId::TheFool,
        name: "THE_FOOL",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_lovers() -> CardSpec {
    CardSpec {
        id: CardId::TheLovers,
        name: "THE_LOVERS",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_emperor() -> CardSpec {
    CardSpec {
        id: CardId::TheEmperor,
        name: "THE_EMPEROR",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_empress() -> CardSpec {
    CardSpec {
        id: CardId::TheEmpress,
        name: "THE_EMPRESS",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_hanged_man() -> CardSpec {
    CardSpec {
        id: CardId::TheHangedMan,
        name: "THE_HANGED_MAN",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_hermit() -> CardSpec {
    CardSpec {
        id: CardId::TheHermit,
        name: "THE_HERMIT",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_hierophant() -> CardSpec {
    CardSpec {
        id: CardId::TheHierophant,
        name: "THE_HIEROPHANT",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_high_priestess() -> CardSpec {
    CardSpec {
        id: CardId::TheHighPriestess,
        name: "THE_HIGH_PRIESTESS",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST.with(CardKeywords::RETAIN),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_magician() -> CardSpec {
    CardSpec {
        id: CardId::TheMagician,
        name: "THE_MAGICIAN",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_moon() -> CardSpec {
    CardSpec {
        id: CardId::TheMoon,
        name: "THE_MOON",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_shot() -> CardSpec {
    CardSpec {
        id: CardId::TheShot,
        name: "THE_SHOT",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_star() -> CardSpec {
    CardSpec {
        id: CardId::TheStar,
        name: "THE_STAR",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_sun() -> CardSpec {
    CardSpec {
        id: CardId::TheSun,
        name: "THE_SUN",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_tower() -> CardSpec {
    CardSpec {
        id: CardId::TheTower,
        name: "THE_TOWER",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn the_world() -> CardSpec {
    CardSpec {
        id: CardId::TheWorld,
        name: "THE_WORLD",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wheel_of_fortune() -> CardSpec {
    CardSpec {
        id: CardId::WheelOfFortune,
        name: "WHEEL_OF_FORTUNE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn sadness() -> CardSpec {
    CardSpec {
        id: CardId::Sadness,
        name: "SADNESS",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::RETAIN.with(CardKeywords::UNPLAYABLE),
        ..CardSpec::base()
    }
}

/// Every Fool card factory, in the order of [`super::FOOL_CARDS`].
pub const FOOL_SPECS: &[fn() -> CardSpec] = &[
    all_is_well,
    anguish,
    archery,
    arrow_stab,
    blossom_shot,
    blot_out_the_sky,
    bounce,
    bow_smack,
    brace,
    brighten_up,
    cho_barbs,
    cho_chorus,
    cho_firm,
    cho_ki,
    cho_limbs,
    cho_ready,
    coda,
    compost,
    covering_fire,
    deep_stretch,
    defend_fool,
    dislike_you,
    divine_shot,
    ecstasy,
    encore,
    ennui,
    enrapture,
    ensemble,
    epic_retrieval,
    explosive_arrows,
    florescence,
    growth,
    harmonic_shield,
    heavy_shot,
    hidden_roots,
    hobbling_shot,
    honing,
    jest,
    ki_chant,
    ki_hop,
    ki_rest,
    ki_verse,
    ki_whip,
    locked_in,
    foolish_shot,
    lost_lute,
    lullaby,
    melodic_shot,
    motif,
    mourning_blow,
    mythic_form,
    nail,
    nock,
    punch_of_joy,
    quick_shot,
    quietus,
    quiver,
    rain_shot,
    rapid_shot,
    recall,
    repeat_shot,
    rhythm_strike,
    rummage,
    self_deprecate,
    serrated_arrows,
    show_off,
    simpleton,
    sky_shot,
    song_path,
    sorrowful_strike,
    steady_cadence,
    strike_fool,
    timber,
    trick_shot,
    tune,
    uproot,
    violent_chord,
    vitriol,
    wah_blast,
    wah_cho,
    wah_cho_ki,
    wah_hammer,
    wah_ki,
    wah_symphony,
    wah_sweep,
    wah_touch,
    whistling_arrows,
    wrest,
    yah,
    yoga,
    death,
    joy,
    judgement,
    justice,
    strength,
    temperance,
    the_chariot,
    the_devil,
    the_fool,
    the_lovers,
    the_emperor,
    the_empress,
    the_hanged_man,
    the_hermit,
    the_hierophant,
    the_high_priestess,
    the_magician,
    the_moon,
    the_shot,
    the_star,
    the_sun,
    the_tower,
    the_world,
    wheel_of_fortune,
    sadness,
];

/// Spec-level upgraded values (absolute) for a Fool card, from its C#
/// `OnUpgrade`: `Damage` / `Block.UpgradeValueBy`, `EnergyCost.UpgradeBy`,
/// `AddKeyword`. `None` for a non-Fool id or an effect-only upgrade.
#[must_use]
pub fn upgrade_delta(card_id: CardId) -> Option<CardUpgradeDelta> {
    match card_id {
        // `ArrowStab.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::ArrowStab => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // `BlossomShot.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::BlossomShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(14),
            base_block: None,
            keywords: None,
        }),
        // `BlotOutTheSky.cs`: EnergyCost.UpgradeBy(-1);
        CardId::BlotOutTheSky => Some(CardUpgradeDelta {
            cost: Some(0),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `BowSmack.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::BowSmack => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(11),
            base_block: None,
            keywords: None,
        }),
        // `ChoBarbs.cs`: DynamicVars.Block.UpgradeValueBy(2m); DynamicVars["ChoBarbsPower"].UpgradeValueBy(1m);
        CardId::ChoBarbsCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(9),
            keywords: None,
        }),
        // `ChoChorus.cs`: DynamicVars.Block.UpgradeValueBy(3m);
        CardId::ChoChorus => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(15),
            keywords: None,
        }),
        // `ChoFirm.cs`: DynamicVars.Block.UpgradeValueBy(2m);
        CardId::ChoFirm => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(8),
            keywords: None,
        }),
        // `ChoKi.cs`: DynamicVars.Block.UpgradeValueBy(3m);
        CardId::ChoKi => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(11),
            keywords: None,
        }),
        // `ChoLimbs.cs`: DynamicVars.Block.UpgradeValueBy(2m);
        CardId::ChoLimbs => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(7),
            keywords: None,
        }),
        // `ChoReady.cs`: DynamicVars.Block.UpgradeValueBy(3m); DynamicVars["AimPower"].UpgradeValueBy(2m);
        CardId::ChoReady => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(11),
            keywords: None,
        }),
        // `Coda.cs`: DynamicVars.Damage.UpgradeValueBy(15m);
        CardId::Coda => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(55),
            base_block: None,
            keywords: None,
        }),
        // `DefendFool.cs`: DynamicVars.Block.UpgradeValueBy(3m);
        CardId::DefendFool => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(8),
            keywords: None,
        }),
        // `DislikeYou.cs`: DynamicVars.Damage.UpgradeValueBy(1m); DynamicVars["DislikeYouPower"].UpgradeValueBy(25m);
        CardId::DislikeYouCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // `DivineShot.cs`: DynamicVars.Damage.UpgradeValueBy(10m);
        CardId::DivineShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(50),
            base_block: None,
            keywords: None,
        }),
        // `Ecstasy.cs`: EnergyCost.UpgradeBy(-1);
        CardId::EcstasyCard => Some(CardUpgradeDelta {
            cost: Some(1),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `EpicRetrieval.cs`: DynamicVars.Damage.UpgradeValueBy(5m); DynamicVars.Energy.UpgradeValueBy(1m);
        CardId::EpicRetrieval => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(20),
            base_block: None,
            keywords: None,
        }),
        // `HeavyShot.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::HeavyShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(25),
            base_block: None,
            keywords: None,
        }),
        // `HobblingShot.cs`: DynamicVars.Damage.UpgradeValueBy(1m); DynamicVars["Power"].UpgradeValueBy(1m);
        CardId::HobblingShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(14),
            base_block: None,
            keywords: None,
        }),
        // `Honing.cs`: EnergyCost.UpgradeBy(-1);
        CardId::HoningCard => Some(CardUpgradeDelta {
            cost: Some(2),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `Jest.cs`: DynamicVars.Block.UpgradeValueBy(4m);
        CardId::Jest => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(18),
            keywords: None,
        }),
        // `KiHop.cs`: DynamicVars.Damage.UpgradeValueBy(1m); DynamicVars.Cards.UpgradeValueBy(1m);
        CardId::KiHop => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(8),
            base_block: None,
            keywords: None,
        }),
        // `KiWhip.cs`: DynamicVars.Damage.UpgradeValueBy(1m); DynamicVars.Energy.UpgradeValueBy(1m);
        CardId::KiWhip => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(6),
            base_block: None,
            keywords: None,
        }),
        // `LockedIn.cs`: EnergyCost.UpgradeBy(-1);
        CardId::LockedIn => Some(CardUpgradeDelta {
            cost: Some(0),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `FoolishShot.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::FoolishShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(20),
            base_block: None,
            keywords: None,
        }),
        // `LostLute.cs`: EnergyCost.UpgradeBy(-1);
        CardId::LostLuteCard => Some(CardUpgradeDelta {
            cost: Some(2),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `Lullaby.cs`: AddKeyword(CardKeyword.Innate);
        CardId::LullabyCard => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: None,
            keywords: Some(CardKeywords::INNATE),
        }),
        // `MelodicShot.cs`: EnergyCost.UpgradeBy(-1); DynamicVars.Damage.UpgradeValueBy(1m);
        CardId::MelodicShot => Some(CardUpgradeDelta {
            cost: Some(0),
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // `Motif.cs`: EnergyCost.UpgradeBy(-1);
        CardId::MotifCard => Some(CardUpgradeDelta {
            cost: Some(1),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `MourningBlow.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::MourningBlow => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(15),
            base_block: None,
            keywords: None,
        }),
        // `Nail.cs`: DynamicVars.Damage.UpgradeValueBy(5m); DynamicVars["Power"].UpgradeValueBy(1m);
        CardId::Nail => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(25),
            base_block: None,
            keywords: None,
        }),
        // `PunchOfJoy.cs`: DynamicVars.Damage.UpgradeValueBy(4m);
        CardId::PunchOfJoy => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(17),
            base_block: None,
            keywords: None,
        }),
        // `QuickShot.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::QuickShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // `RainShot.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::RainShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(15),
            base_block: None,
            keywords: None,
        }),
        // `Recall.cs`: EnergyCost.UpgradeBy(-1);
        CardId::Recall => Some(CardUpgradeDelta {
            cost: Some(0),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `RepeatShot.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::RepeatShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(20),
            base_block: None,
            keywords: None,
        }),
        // `RhythmStrike.cs`: DynamicVars.Damage.UpgradeValueBy(1m);
        CardId::RhythmStrike => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(5),
            base_block: None,
            keywords: None,
        }),
        // `Simpleton.cs`: EnergyCost.UpgradeBy(-1);
        CardId::SimpletonCard => Some(CardUpgradeDelta {
            cost: Some(1),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `SkyShot.cs`: DynamicVars.Damage.UpgradeValueBy(10m);
        CardId::SkyShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(40),
            base_block: None,
            keywords: None,
        }),
        // `SorrowfulStrike.cs`: DynamicVars.Damage.UpgradeValueBy(4m);
        CardId::SorrowfulStrike => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(17),
            base_block: None,
            keywords: None,
        }),
        // `SteadyCadence.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::SteadyCadence => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(13),
            base_block: None,
            keywords: None,
        }),
        // `StrikeFool.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::StrikeFool => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(9),
            base_block: None,
            keywords: None,
        }),
        // `Timber.cs`: DynamicVars.Damage.UpgradeValueBy(5m); DynamicVars["OnKillDamage"].UpgradeValueBy(5m);
        CardId::Timber => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(25),
            base_block: None,
            keywords: None,
        }),
        // `TrickShot.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::TrickShot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(15),
            base_block: None,
            keywords: None,
        }),
        // `Tune.cs`: EnergyCost.UpgradeBy(-1);
        CardId::TuneCard => Some(CardUpgradeDelta {
            cost: Some(1),
            base_damage: None,
            base_block: None,
            keywords: None,
        }),
        // `WahBlast.cs`: DynamicVars.Damage.UpgradeValueBy(4m);
        CardId::WahBlast => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(18),
            base_block: None,
            keywords: None,
        }),
        // `WahCho.cs`: DynamicVars.Damage.UpgradeValueBy(2m); DynamicVars.Block.UpgradeValueBy(2m);
        CardId::WahCho => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(5),
            base_block: Some(5),
            keywords: None,
        }),
        // `WahChoKi.cs`: DynamicVars.Damage.UpgradeValueBy(3m); DynamicVars.Block.UpgradeValueBy(3m); DynamicVars.Cards.UpgradeValueBy(1m);
        CardId::WahChoKi => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: Some(12),
            keywords: None,
        }),
        // `WahHammer.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::WahHammer => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(12),
            base_block: None,
            keywords: None,
        }),
        // `WahKi.cs`: DynamicVars.Damage.UpgradeValueBy(2m);
        CardId::WahKi => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(4),
            base_block: None,
            keywords: None,
        }),
        // `WahSweep.cs`: DynamicVars.Damage.UpgradeValueBy(3m);
        CardId::WahSweep => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(11),
            base_block: None,
            keywords: None,
        }),
        // `WahTouch.cs`: DynamicVars.Damage.UpgradeValueBy(2m); DynamicVars.Vulnerable.UpgradeValueBy(1m);
        CardId::WahTouch => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(8),
            base_block: None,
            keywords: None,
        }),
        // `WhistlingArrows.cs`: DynamicVars.Block.UpgradeValueBy(1m);
        CardId::WhistlingArrows => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(4),
            keywords: None,
        }),
        // `Justice.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::Justice => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(35),
            base_block: None,
            keywords: None,
        }),
        // `Temperance.cs`: DynamicVars.Block.UpgradeValueBy(5m);
        CardId::Temperance => Some(CardUpgradeDelta {
            cost: None,
            base_damage: None,
            base_block: Some(35),
            keywords: None,
        }),
        // `TheChariot.cs`: DynamicVars.Damage.UpgradeValueBy(5m);
        CardId::TheChariot => Some(CardUpgradeDelta {
            cost: None,
            base_damage: Some(30),
            base_block: None,
            keywords: None,
        }),
        _ => None,
    }
}
