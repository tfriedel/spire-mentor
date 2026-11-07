// POD card-spec factories for the Status card pool (cost / type /
// rarity / target / keywords / base_damage / base_block).
//
// HAND-MAINTAINED against the C# decompilation
// (../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/*.cs),
// which is the source of truth for all card stats. Originally seeded
// from the Python pin (sts2-rl-agent@01c094d) but the generator was
// removed once values diverged from the pin by design; edit stat
// values here directly and verify against each card's C# DamageVar /
// BlockVar / constructor.

#![allow(dead_code)]

use sts2_core::cards::{CardId, CardType, TargetType};

use super::types::{CardKeywords, CardRarity, CardSpec};
#[must_use]
pub const fn wound() -> CardSpec {
    CardSpec {
        id: CardId::Wound,
        name: "WOUND",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn dazed() -> CardSpec {
    CardSpec {
        id: CardId::Dazed,
        name: "DAZED",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETHEREAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn slimed() -> CardSpec {
    CardSpec {
        id: CardId::Slimed,
        name: "SLIMED",
        cost: 1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn burn() -> CardSpec {
    CardSpec {
        id: CardId::Burn,
        name: "BURN",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn void() -> CardSpec {
    CardSpec {
        id: CardId::Void,
        name: "VOID",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETHEREAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn beckon() -> CardSpec {
    CardSpec {
        id: CardId::Beckon,
        name: "BECKON",
        cost: 1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn debris() -> CardSpec {
    CardSpec {
        id: CardId::Debris,
        name: "DEBRIS",
        cost: 1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn frantic_escape() -> CardSpec {
    CardSpec {
        id: CardId::FranticEscape,
        name: "FRANTIC_ESCAPE",
        cost: 1,
        card_type: CardType::Status,
        target_type: TargetType::Self_,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn infection() -> CardSpec {
    CardSpec {
        id: CardId::Infection,
        name: "INFECTION",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn soot() -> CardSpec {
    CardSpec {
        id: CardId::Soot,
        name: "SOOT",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn toxic() -> CardSpec {
    CardSpec {
        id: CardId::Toxic,
        name: "TOXIC",
        cost: 1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn ascenders_bane() -> CardSpec {
    CardSpec {
        id: CardId::AscendersBane,
        name: "ASCENDERS_BANE",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# AscendersBane.cs CanonicalKeywords = [Eternal, Unplayable, Ethereal].
        // ETERNAL is the removal gate itself (card_is_removable_instance tests
        // this bit and nothing else); bead sts2-rs-8ixr6.
        keywords: CardKeywords::UNPLAYABLE
            .with(CardKeywords::ETHEREAL)
            .with(CardKeywords::ETERNAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn bad_luck() -> CardSpec {
    CardSpec {
        id: CardId::BadLuck,
        name: "BAD_LUCK",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# `BadLuck.cs` CanonicalKeywords includes `CardKeyword.Eternal`;
        // it was missing here, so `card_is_removable_csharp` called this
        // curse removable and `removal_offer_order` offered it. Live
        // `remove:ENTHRALLED` shop picks named a card no
        // `FromDeckForRemoval` screen can show (2026-08-26).
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETERNAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn clumsy() -> CardSpec {
    CardSpec {
        id: CardId::Clumsy,
        name: "CLUMSY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETHEREAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn curse_of_the_bell() -> CardSpec {
    CardSpec {
        id: CardId::CurseOfTheBell,
        name: "CURSE_OF_THE_BELL",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# `CurseOfTheBell.cs` CanonicalKeywords includes `CardKeyword.Eternal`;
        // it was missing here, so `card_is_removable_csharp` called this
        // curse removable and `removal_offer_order` offered it. Live
        // `remove:ENTHRALLED` shop picks named a card no
        // `FromDeckForRemoval` screen can show (2026-08-26).
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETERNAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn debt() -> CardSpec {
    CardSpec {
        id: CardId::Debt,
        name: "DEBT",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn decay() -> CardSpec {
    CardSpec {
        id: CardId::Decay,
        name: "DECAY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn doubt() -> CardSpec {
    CardSpec {
        id: CardId::Doubt,
        name: "DOUBT",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn enthralled() -> CardSpec {
    CardSpec {
        id: CardId::Enthralled,
        name: "ENTHRALLED",
        cost: 2,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# `Enthralled.cs` CanonicalKeywords includes `CardKeyword.Eternal`;
        // it was missing here, so `card_is_removable_csharp` called this
        // curse removable and `removal_offer_order` offered it. Live
        // `remove:ENTHRALLED` shop picks named a card no
        // `FromDeckForRemoval` screen can show (2026-08-26).
        keywords: CardKeywords::ETERNAL,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn folly() -> CardSpec {
    CardSpec {
        id: CardId::Folly,
        name: "FOLLY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# `Folly.cs` CanonicalKeywords includes `CardKeyword.Eternal`;
        // it was missing here, so `card_is_removable_csharp` called this
        // curse removable and `removal_offer_order` offered it. Live
        // `remove:ENTHRALLED` shop picks named a card no
        // `FromDeckForRemoval` screen can show (2026-08-26). `Ethereal` was
        // missing too, so the end-of-turn expel discarded it (GH #1054).
        keywords: CardKeywords::UNPLAYABLE
            .with(CardKeywords::INNATE)
            .with(CardKeywords::ETERNAL)
            .with(CardKeywords::ETHEREAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn greed() -> CardSpec {
    CardSpec {
        id: CardId::Greed,
        name: "GREED",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        // C# Greed.cs CanonicalKeywords = {Eternal, Unplayable}. ETERNAL makes
        // it non-removable by the generic removal offer (C# IsRemovable =>
        // !Eternal); without it Greed wrongly occupied a removal-offer index
        // and shifted the lockstep pick (sts2-rs-RP22).
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::ETERNAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn guilty() -> CardSpec {
    CardSpec {
        id: CardId::Guilty,
        name: "GUILTY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn injury() -> CardSpec {
    CardSpec {
        id: CardId::Injury,
        name: "INJURY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn normality() -> CardSpec {
    CardSpec {
        id: CardId::Normality,
        name: "NORMALITY",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn pain() -> CardSpec {
    CardSpec {
        id: CardId::Pain,
        name: "PAIN",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn parasite() -> CardSpec {
    CardSpec {
        id: CardId::Parasite,
        name: "PARASITE",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn poor_sleep() -> CardSpec {
    CardSpec {
        id: CardId::PoorSleep,
        name: "POOR_SLEEP",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::RETAIN),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn regret() -> CardSpec {
    CardSpec {
        id: CardId::Regret,
        name: "REGRET",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn shame() -> CardSpec {
    CardSpec {
        id: CardId::Shame,
        name: "SHAME",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn spore_mind() -> CardSpec {
    CardSpec {
        id: CardId::SporeMind,
        name: "SPORE_MIND",
        cost: 1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn writhe() -> CardSpec {
    CardSpec {
        id: CardId::Writhe,
        name: "WRITHE",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::INNATE.with(CardKeywords::UNPLAYABLE),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rip_and_tear() -> CardSpec {
    CardSpec {
        id: CardId::RipAndTear,
        name: "RIP_AND_TEAR",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Random,
        rarity: CardRarity::Event,
        base_damage: 7,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn apotheosis() -> CardSpec {
    CardSpec {
        id: CardId::Apotheosis,
        name: "APOTHEOSIS",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST.with(CardKeywords::INNATE),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn apparition() -> CardSpec {
    CardSpec {
        id: CardId::Apparition,
        name: "APPARITION",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST.with(CardKeywords::ETHEREAL),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn brightest_flame() -> CardSpec {
    CardSpec {
        id: CardId::BrightestFlame,
        name: "BRIGHTEST_FLAME",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn maul() -> CardSpec {
    CardSpec {
        id: CardId::Maul,
        name: "MAUL",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Ancient,
        base_damage: 5,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn neows_fury() -> CardSpec {
    CardSpec {
        id: CardId::NeowsFury,
        name: "NEOWS_FURY",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Ancient,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn relax() -> CardSpec {
    CardSpec {
        id: CardId::Relax,
        name: "RELAX",
        cost: 3,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 15,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn whistle() -> CardSpec {
    CardSpec {
        id: CardId::Whistle,
        name: "WHISTLE",
        cost: 3,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Ancient,
        base_damage: 33,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn wish() -> CardSpec {
    CardSpec {
        id: CardId::Wish,
        name: "WISH",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Ancient,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn byrd_swoop() -> CardSpec {
    CardSpec {
        id: CardId::ByrdSwoop,
        name: "BYRD_SWOOP",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 14,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn caltrops() -> CardSpec {
    CardSpec {
        id: CardId::Caltrops,
        name: "CALTROPS",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn clash() -> CardSpec {
    CardSpec {
        id: CardId::Clash,
        name: "CLASH",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 14,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn distraction() -> CardSpec {
    CardSpec {
        id: CardId::Distraction,
        name: "DISTRACTION",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn dual_wield() -> CardSpec {
    CardSpec {
        id: CardId::DualWield,
        name: "DUAL_WIELD",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn enlightenment() -> CardSpec {
    CardSpec {
        id: CardId::Enlightenment,
        name: "ENLIGHTENMENT",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn entrench() -> CardSpec {
    CardSpec {
        id: CardId::Entrench,
        name: "ENTRENCH",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn exterminate() -> CardSpec {
    CardSpec {
        id: CardId::Exterminate,
        name: "EXTERMINATE",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Event,
        base_damage: 3,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn feeding_frenzy() -> CardSpec {
    CardSpec {
        id: CardId::FeedingFrenzyCard,
        name: "FEEDING_FRENZY_CARD",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn hello_world() -> CardSpec {
    CardSpec {
        id: CardId::HelloWorldCard,
        name: "HELLO_WORLD_CARD",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

/// GH #663. `target_type` is `Self_`, not `Enemy`, because C# reads it off the
/// per-instance variant rather than off the constructor: `MadScience.cs:82-92`
/// returns `TargetType.Self` unless `TinkerTimeType == Attack`, and
/// `TinkerTimeType` is written ONLY by the TinkerTime event
/// (`TinkerTime.cs:90,148,156`). A MadScience that never went through the event
/// holds `default(CardType)`, and C#'s enum starts at `None` — so the
/// unconfigured card is untargeted, and the `MadScience(1, CardType.Attack, …,
/// AnyEnemy)` constructor (`MadScience.cs:167-170`) never describes a real one.
/// The C#-oracle sweep caught this as `card:MAD_SCIENCE` offering three plays
/// (one per enemy) against C#'s one.
///
/// A CONFIGURED card does not read this: `set_mad_science_variant` stamps both
/// `card_type` and `target_type` from the variant, which is what the event and
/// the bridge both go through. So this default is only ever the unconfigured
/// case — the one the two engines disagreed about.
///
/// `card_type` stays `Attack` because Rust has no `CardType::None` to hold C#'s
/// unset value, and the play path's Attack fallback stands in for C# throwing
/// `InvalidOperationException` on an unconfigured play (`MadScience.cs:185`) —
/// a behaviour worth diverging from rather than porting.
#[must_use]
pub const fn mad_science() -> CardSpec {
    CardSpec {
        id: CardId::MadScience,
        name: "MAD_SCIENCE",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 12,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn metamorphosis() -> CardSpec {
    CardSpec {
        id: CardId::Metamorphosis,
        name: "METAMORPHOSIS",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn outmaneuver() -> CardSpec {
    CardSpec {
        id: CardId::Outmaneuver,
        name: "OUTMANEUVER",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn peck() -> CardSpec {
    CardSpec {
        id: CardId::Peck,
        name: "PECK",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 2,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn rebound() -> CardSpec {
    CardSpec {
        id: CardId::Rebound,
        name: "REBOUND",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 9,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn squash() -> CardSpec {
    CardSpec {
        id: CardId::Squash,
        name: "SQUASH",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Event,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn stack() -> CardSpec {
    CardSpec {
        id: CardId::Stack,
        name: "STACK",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn toric_toughness() -> CardSpec {
    CardSpec {
        id: CardId::ToricToughness,
        name: "TORIC_TOUGHNESS",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Event,
        base_damage: 0,
        base_block: 5,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn fuel() -> CardSpec {
    CardSpec {
        id: CardId::Fuel,
        name: "FUEL",
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
pub const fn giant_rock() -> CardSpec {
    CardSpec {
        id: CardId::GiantRock,
        name: "GIANT_ROCK",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 16,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn luminesce() -> CardSpec {
    CardSpec {
        id: CardId::Luminesce,
        name: "LUMINESCE",
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
pub const fn minion_dive_bomb() -> CardSpec {
    CardSpec {
        id: CardId::MinionDiveBomb,
        name: "MINION_DIVE_BOMB",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 13,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn minion_sacrifice() -> CardSpec {
    CardSpec {
        id: CardId::MinionSacrifice,
        name: "MINION_SACRIFICE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Token,
        base_damage: 0,
        // C# `MinionSacrifice.cs` (v0.107.1) `BlockVar(8m)`, OnUpgrade +3 → 11
        // (was 9/12 in v0.103.2).
        base_block: 8,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn minion_strike() -> CardSpec {
    CardSpec {
        id: CardId::MinionStrike,
        name: "MINION_STRIKE",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 6,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn shiv() -> CardSpec {
    CardSpec {
        id: CardId::Shiv,
        name: "SHIV",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 4,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn soul() -> CardSpec {
    CardSpec {
        id: CardId::Soul,
        name: "SOUL",
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
pub const fn sovereign_blade() -> CardSpec {
    CardSpec {
        id: CardId::SovereignBlade,
        name: "SOVEREIGN_BLADE",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Token,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::RETAIN,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn sweeping_gaze() -> CardSpec {
    CardSpec {
        id: CardId::SweepingGaze,
        name: "SWEEPING_GAZE",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Random,
        rarity: CardRarity::Token,
        // OstyDamageVar(10m) — flat base dealt FROM the Osty (15 upgraded via the
        // card_upgrades_generated delta). The prior Python-pin port had 0 here.
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::ETHEREAL.with(CardKeywords::EXHAUST),
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn disintegration() -> CardSpec {
    CardSpec {
        id: CardId::Disintegration,
        name: "DISINTEGRATION",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn mind_rot() -> CardSpec {
    CardSpec {
        id: CardId::MindRot,
        name: "MIND_ROT",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn sloth_status() -> CardSpec {
    CardSpec {
        id: CardId::SlothStatus,
        name: "SLOTH_STATUS",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn waste_away() -> CardSpec {
    CardSpec {
        id: CardId::WasteAway,
        name: "WASTE_AWAY",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn byrdonis_egg() -> CardSpec {
    CardSpec {
        id: CardId::ByrdonisEgg,
        name: "BYRDONIS_EGG",
        cost: -1,
        card_type: CardType::Quest,
        target_type: TargetType::None,
        rarity: CardRarity::Quest,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn lantern_key() -> CardSpec {
    CardSpec {
        id: CardId::LanternKey,
        name: "LANTERN_KEY",
        cost: -1,
        card_type: CardType::Quest,
        target_type: TargetType::Self_,
        rarity: CardRarity::Quest,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

/// Dowsing — **v0.111.0-only** Quest card (`Cards/Dowsing.cs`, GH #792),
/// added to the deck by the DowsingRod Neow relic:
///
/// [decompiled C# omitted — see the cited path]
///
/// Like the other Quest cards (ByrdonisEgg / LanternKey / SpoilsMap) it is
/// an Unplayable `CardType::Quest` card. Its `BeforeRoomEntered` quest (5 `?`
/// rooms -> transform into Abundance) is card content owned by S8a (#793) and not ported here.
#[must_use]
pub const fn dowsing() -> CardSpec {
    CardSpec {
        id: CardId::Dowsing,
        name: "DOWSING",
        cost: -1,
        card_type: CardType::Quest,
        target_type: TargetType::None,
        rarity: CardRarity::Quest,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

#[must_use]
pub const fn spoils_map() -> CardSpec {
    CardSpec {
        id: CardId::SpoilsMap,
        name: "SPOILS_MAP",
        cost: -1,
        card_type: CardType::Quest,
        target_type: TargetType::Self_,
        rarity: CardRarity::Quest,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE,
        ..CardSpec::base()
    }
}

/// Wither — v0.107.1 status card (`Cards/Wither.cs`). Cost -1, Unplayable,
/// `MaxUpgradeLevel == 0` (cannot be upgraded — modeled via the
/// NON_UPGRADEABLE keyword). `HasTurnEndInHandEffect`: deals
/// `DamageVar(3, Unpowered|Move)` to its owner at turn-end-in-hand, scaling
/// `+3` per FakeUpgrade level (see `combat_loop` turn-end handler +
/// `VarKey::WitherFakeUpgradeLevel`). Not retained by Runic Pyramid.
#[must_use]
pub const fn wither() -> CardSpec {
    CardSpec {
        id: CardId::Wither,
        name: "WITHER",
        cost: -1,
        card_type: CardType::Status,
        target_type: TargetType::None,
        rarity: CardRarity::Status,
        base_damage: 3,
        base_block: 0,
        keywords: CardKeywords::UNPLAYABLE.with(CardKeywords::NON_UPGRADEABLE),
        ..CardSpec::base()
    }
}
