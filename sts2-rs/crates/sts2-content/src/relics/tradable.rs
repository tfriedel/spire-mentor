//! Which of your relics the trade events are allowed to take.
//!
//! C# `RelicModel.IsTradable` (`Models/RelicModel.cs:176`) is read by exactly
//! two events — `RelicTrader.GetValidRelics` and `RanwidTheElder` — and gates
//! both their `IsAllowed` and the stock they offer. It is a conjunction of a
//! rarity test and four per-relic flags:
//!
//! ```text
//! !IsUsedUp && !HasUponPickupEffect && !IsMelted && !SpawnsPets
//!   && Rarity != Starter && Rarity is not Event or Ancient
//! ```
//!
//! (the decompiler renders the last clause as `(uint)(rarity - 6) <= 1u`, i.e.
//! the two enum members after `Shop` — `Event` and `Ancient`. `None`, the
//! Circlet's rarity, is deliberately NOT excluded.)
//!
//! Three of those resolve statically off the v0.107.1 decomp and are baked
//! into the table below: rarity, `HasUponPickupEffect` (15 relics that survive
//! the rarity cut — a relic whose whole effect fired when you picked it up
//! would be a free trade-in) and `SpawnsPets` (3, all of which the rarity cut
//! already removes).
//!
//! `IsMelted` is per-run state (Toy Box's wax relics melt one per combat, see
//! `RunState::melted_relics`), so it is checked by [`relic_is_tradable_in`],
//! which every trade event must use — a melted relic is skipped by C#'s
//! `GetValidRelics` and by the `IsAllowed` counts.
//!
//! `IsUsedUp` is per-run state too. Of the relics that override it, only
//! Lizard Tail survives the static cuts, and its `_wasUsed` is the relic's
//! run counter (1 once it has revived you), so [`relic_is_tradable_in`]
//! checks that as well.

use sts2_core::relics::RelicId;
use sts2_core::runstate::RunState;

/// [`relic_is_tradable`] plus the per-run `!IsMelted` clause of C#
/// `RelicModel.IsTradable` (`Models/RelicModel.cs:188`, both builds): a Toy
/// Box wax relic that has melted stays in the inventory but no trade event
/// may take it, nor does it count toward their `IsAllowed` thresholds.
/// Also the per-run `!IsUsedUp` clause: a Lizard Tail that has revived you
/// (`LizardTail.IsUsedUp => _wasUsed`, run counter 1) is not tradable.
#[must_use]
pub fn relic_is_tradable_in(state: &RunState, id: RelicId) -> bool {
    relic_is_tradable(id)
        && !state.melted_relics.contains(&id)
        && !(id == RelicId::LizardTail && state.relic_counters.get(&id).is_some_and(|&c| c != 0))
}

/// True when a trade event may take `id` off you — see the module doc.
///
/// 137 of the 297 v0.107.1 relics qualify. The list is POSITIVE on purpose:
/// a relic added to the enum without being classified here reads as
/// untradable, which costs a trade option rather than offering a relic the
/// game would never put on the table.
#[must_use]
pub const fn relic_is_tradable(id: RelicId) -> bool {
    matches!(
        id,
        // None (2)
        RelicId::Circlet | RelicId::DeprecatedRelic |
        // Common (26)
        RelicId::AmethystAubergine | RelicId::Anchor | RelicId::BagOfMarbles |
        RelicId::BagOfPreparation | RelicId::BloodVial | RelicId::BoneFlute |
        RelicId::BookOfFiveRings | RelicId::BronzeScales | RelicId::CentennialPuzzle |
        RelicId::DataDisk | RelicId::FencingManual | RelicId::FestivePopper | RelicId::Gorget |
        RelicId::HappyFlower | RelicId::JuzuBracelet | RelicId::Lantern | RelicId::MealTicket |
        RelicId::OddlySmoothStone | RelicId::Pendulum | RelicId::RedMask | RelicId::RedSkull |
        RelicId::RegalPillow | RelicId::SneckoSkull | RelicId::StrikeDummy | RelicId::Vajra |
        RelicId::VenerableTeaSet |
        // Uncommon (39)
        RelicId::Akabeko | RelicId::BookRepairKnife | RelicId::BowlerHat |
        RelicId::Candelabra | RelicId::EternalFeather | RelicId::FuneraryMask |
        RelicId::GalacticDust | RelicId::GoldPlatedCables | RelicId::GremlinHorn |
        RelicId::HornCleat | RelicId::JossPaper | RelicId::Kusarigama | RelicId::LastingCandy |
        RelicId::LetterOpener | RelicId::LuckyFysh | RelicId::MercuryHourglass |
        RelicId::MiniatureCannon | RelicId::Nunchaku | RelicId::Orichalcum |
        RelicId::OrnamentalFan | RelicId::Pantograph | RelicId::PaperPhrog |
        RelicId::ParryingShield | RelicId::PenNib | RelicId::Permafrost |
        RelicId::PetrifiedToad | RelicId::Planisphere | RelicId::Regalite |
        RelicId::ReptileTrinket | RelicId::RippleBasin | RelicId::SelfFormingClay |
        RelicId::SparklingRouge | RelicId::StoneCracker | RelicId::SymbioticVirus |
        RelicId::Tingsha | RelicId::TinyMailbox | RelicId::TuningFork |
        RelicId::TwistedFunnel | RelicId::Vambrace |
        // Rare (48)
        RelicId::ArtOfWar | RelicId::BeatingRemnant | RelicId::Bellows | RelicId::BigHat |
        RelicId::Bookmark | RelicId::CaptainsWheel | RelicId::Chandelier |
        RelicId::CharonsAshes | RelicId::CloakClasp | RelicId::DemonTongue |
        RelicId::EmotionChip | RelicId::FrozenEgg | RelicId::GamblingChip |
        RelicId::GamePiece | RelicId::Girya | RelicId::HelicalDart | RelicId::IceCream |
        RelicId::IntimidatingHelmet | RelicId::IvoryTile | RelicId::Kunai |
        RelicId::LizardTail | RelicId::LunarPastry | RelicId::MeatOnTheBone |
        RelicId::Metronome | RelicId::MiniRegent | RelicId::MoltenEgg |
        RelicId::MummifiedHand | RelicId::OrangeDough | RelicId::PaperKrane |
        RelicId::Pocketwatch | RelicId::PowerCell | RelicId::PrayerWheel |
        RelicId::RainbowRing | RelicId::RazorTooth | RelicId::RuinedHelmet | RelicId::Shovel |
        RelicId::Shuriken | RelicId::StoneCalendar | RelicId::SturdyClamp |
        RelicId::TheCourier | RelicId::ToughBandages | RelicId::ToxicEgg |
        RelicId::TungstenRod | RelicId::UnceasingTop | RelicId::UnsettlingLamp |
        RelicId::VexingPuzzlebox | RelicId::WhiteBeastStatue | RelicId::WhiteStar |
        // Shop (22)
        RelicId::BeltBuckle | RelicId::Bread | RelicId::Brimstone | RelicId::BurningSticks |
        RelicId::ChemicalX | RelicId::DingyRug | RelicId::DragonFruit | RelicId::GhostSeed |
        RelicId::LavaLamp | RelicId::MembershipCard | RelicId::MiniatureTent |
        RelicId::MysticLighter | RelicId::NinjaScroll | RelicId::RingingTriangle |
        RelicId::RunicCapacitor | RelicId::ScreamingFlagon | RelicId::SlingOfCourage |
        RelicId::TheAbacus | RelicId::Toolbox | RelicId::UndyingSigil |
        RelicId::VitruvianMinion | RelicId::WingCharm |
        // v0.113.0 The Fool's non-starter relics (`FoolRelicPool`): Common
        // ExerciseMat, Uncommon Ocarina / WhittlingKnife, Rare ConductorsBaton
        // / Fertilizer / Twine, Shop BizarreLamp; none has an upon-pickup
        // effect or spawns pets. They exist in no earlier run.
        RelicId::ExerciseMat | RelicId::Ocarina | RelicId::WhittlingKnife |
        RelicId::ConductorsBaton | RelicId::Fertilizer | RelicId::Twine |
        RelicId::BizarreLamp
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.113.0 `RelicModel.IsTradable` over The Fool's relics: the seven
    /// Common/Uncommon/Rare/Shop ones qualify; the two Starters (`SongBow.cs:15`,
    /// `Storyteller.cs:20`) do not. Missing them shrank `RelicTrader`'s
    /// `GetValidRelics` before its `StableShuffle`, so a Fool holding Ocarina
    /// was offered a different trade (Fool:1000 act3 event@1007).
    #[test]
    fn fool_relics_follow_is_tradable() {
        for id in [
            RelicId::ExerciseMat,
            RelicId::Ocarina,
            RelicId::WhittlingKnife,
            RelicId::ConductorsBaton,
            RelicId::Fertilizer,
            RelicId::Twine,
            RelicId::BizarreLamp,
        ] {
            assert!(relic_is_tradable(id), "{id:?}");
        }
        assert!(!relic_is_tradable(RelicId::SongBow));
        assert!(!relic_is_tradable(RelicId::Storyteller));
    }
}
