//! C#-faithful relic grab-bag composition.
//!
//! Mirrors `MegaCrit.Sts2.Core.Runs.RelicGrabBag.Populate` from the
//! authoritative decomp (`../sts2-wiki/decompiled/v0.103.2`):
//!
//! [decompiled C# omitted — see the cited path]
//!
//! The pre-shuffle order of each rarity deque is therefore: the
//! `SharedRelicPool` relics of that rarity (in `GenerateAllRelics`
//! declaration order) followed by the character pool's relics of that
//! rarity (in declaration order). Because the shuffle is in-place
//! Fisher-Yates, this initial order is part of the RNG-exact result —
//! getting the composition *and* order right is what makes the
//! workbench's `SmallCapsule` after-obtain roll land on `MealTicket`
//! for seed `Y3X2YZSBQJ` Ironclad A0 (bead `sts2-rs-7e04v`, GH #70).
//!
//! The lumped `common::pool_relic_ids()` / `uncommon` / `rare` lists
//! (a Python-pin port) drifted from C# on rarity tags
//! (`Permafrost`/`TinyMailbox` are Uncommon; `BagOfMarbles`/`RedMask`
//! are Common) and leaked foreign-character commons
//! (`BoneFlute`/`DataDisk`/`FencingManual`/`SneckoSkull`) into the
//! Ironclad pool. Those modules stay the canonical home for
//! *effect-handler registration* (rarity-agnostic); this module is the
//! authoritative source for *grab-bag composition*.
//!
//! Rarity tags below are read straight from each relic's
//! `RelicRarity Rarity => …` in `*.Models.Relics/*.cs`; pool membership
//! and order from `*.Models.RelicPools/*.cs`.

use std::collections::BTreeMap;

use sts2_core::potions::CharacterId;
use sts2_core::relics::RelicId;
use sts2_core::RelicRarity;

// --- SharedRelicPool, split by grab-bag rarity, in declaration order. -------

const SHARED_COMMON: &[RelicId] = &[
    RelicId::AmethystAubergine,
    RelicId::Anchor,
    RelicId::BagOfMarbles,
    RelicId::BagOfPreparation,
    RelicId::BloodVial,
    RelicId::BookOfFiveRings,
    RelicId::BronzeScales,
    RelicId::CentennialPuzzle,
    RelicId::FestivePopper,
    RelicId::Gorget,
    RelicId::HappyFlower,
    RelicId::JuzuBracelet,
    RelicId::Lantern,
    RelicId::MealTicket,
    RelicId::OddlySmoothStone,
    RelicId::Pendulum,
    RelicId::PotionBelt,
    RelicId::RedMask,
    RelicId::RegalPillow,
    RelicId::Strawberry,
    RelicId::StrikeDummy,
    RelicId::Vajra,
    RelicId::VenerableTeaSet,
    RelicId::WarPaint,
    RelicId::Whetstone,
];

const SHARED_UNCOMMON: &[RelicId] = &[
    RelicId::Akabeko,
    RelicId::BowlerHat,
    RelicId::Candelabra,
    RelicId::EternalFeather,
    RelicId::GremlinHorn,
    RelicId::HornCleat,
    RelicId::JossPaper,
    RelicId::Kusarigama,
    RelicId::LastingCandy,
    RelicId::LetterOpener,
    RelicId::LuckyFysh,
    RelicId::MercuryHourglass,
    RelicId::MiniatureCannon,
    RelicId::Nunchaku,
    RelicId::Orichalcum,
    RelicId::OrnamentalFan,
    RelicId::Pantograph,
    RelicId::ParryingShield,
    RelicId::Pear,
    RelicId::PenNib,
    RelicId::Permafrost,
    RelicId::PetrifiedToad,
    RelicId::Planisphere,
    RelicId::ReptileTrinket,
    RelicId::RippleBasin,
    RelicId::SparklingRouge,
    RelicId::StoneCracker,
    RelicId::TinyMailbox,
    RelicId::TuningFork,
    RelicId::Vambrace,
];

const SHARED_RARE: &[RelicId] = &[
    RelicId::ArtOfWar,
    RelicId::BeatingRemnant,
    RelicId::Bellows,
    RelicId::CaptainsWheel,
    RelicId::Chandelier,
    RelicId::CloakClasp,
    RelicId::FrozenEgg,
    RelicId::GamblingChip,
    RelicId::GamePiece,
    RelicId::Girya,
    RelicId::IceCream,
    RelicId::IntimidatingHelmet,
    RelicId::Kunai,
    RelicId::LizardTail,
    RelicId::Mango,
    RelicId::MeatOnTheBone,
    RelicId::MoltenEgg,
    RelicId::MummifiedHand,
    RelicId::OldCoin,
    RelicId::Pocketwatch,
    RelicId::PrayerWheel,
    RelicId::RainbowRing,
    RelicId::RazorTooth,
    RelicId::Shovel,
    RelicId::Shuriken,
    RelicId::StoneCalendar,
    RelicId::SturdyClamp,
    RelicId::TheCourier,
    RelicId::ToxicEgg,
    RelicId::TungstenRod,
    RelicId::UnceasingTop,
    RelicId::UnsettlingLamp,
    RelicId::VexingPuzzlebox,
    RelicId::WhiteBeastStatue,
    RelicId::WhiteStar,
];

const SHARED_SHOP: &[RelicId] = &[
    RelicId::BeltBuckle,
    RelicId::Bread,
    RelicId::BurningSticks,
    RelicId::Cauldron,
    RelicId::ChemicalX,
    RelicId::DingyRug,
    RelicId::DollysMirror,
    RelicId::DragonFruit,
    RelicId::GhostSeed,
    RelicId::GnarledHammer,
    RelicId::Kifuda,
    RelicId::LavaLamp,
    RelicId::LeesWaffle,
    RelicId::MembershipCard,
    RelicId::MiniatureTent,
    RelicId::MysticLighter,
    RelicId::Orrery,
    RelicId::PunchDagger,
    RelicId::RingingTriangle,
    RelicId::RoyalStamp,
    RelicId::ScreamingFlagon,
    RelicId::SlingOfCourage,
    RelicId::TheAbacus,
    RelicId::Toolbox,
    RelicId::WingCharm,
];

// --- Per-character pools, split by grab-bag rarity, declaration order. ------
// (The character pool's remaining relic is its Starter/Boss-rarity relic,
// e.g. Ironclad's `BurningBlood`, which is excluded from the grab bag.)

const IRONCLAD_COMMON: &[RelicId] = &[RelicId::RedSkull];
const IRONCLAD_UNCOMMON: &[RelicId] = &[RelicId::PaperPhrog, RelicId::SelfFormingClay];
const IRONCLAD_RARE: &[RelicId] = &[
    RelicId::CharonsAshes,
    RelicId::DemonTongue,
    RelicId::RuinedHelmet,
];
const IRONCLAD_SHOP: &[RelicId] = &[RelicId::Brimstone];

const SILENT_COMMON: &[RelicId] = &[RelicId::SneckoSkull];
const SILENT_UNCOMMON: &[RelicId] = &[RelicId::Tingsha, RelicId::TwistedFunnel];
const SILENT_RARE: &[RelicId] = &[
    RelicId::HelicalDart,
    RelicId::PaperKrane,
    RelicId::ToughBandages,
];
const SILENT_SHOP: &[RelicId] = &[RelicId::NinjaScroll];

const DEFECT_COMMON: &[RelicId] = &[RelicId::DataDisk];
const DEFECT_UNCOMMON: &[RelicId] = &[RelicId::GoldPlatedCables, RelicId::SymbioticVirus];
const DEFECT_RARE: &[RelicId] = &[RelicId::EmotionChip, RelicId::PowerCell, RelicId::Metronome];
const DEFECT_SHOP: &[RelicId] = &[RelicId::RunicCapacitor];

const REGENT_COMMON: &[RelicId] = &[RelicId::FencingManual];
const REGENT_UNCOMMON: &[RelicId] = &[RelicId::GalacticDust, RelicId::Regalite];
const REGENT_RARE: &[RelicId] = &[
    RelicId::LunarPastry,
    RelicId::MiniRegent,
    RelicId::OrangeDough,
];
const REGENT_SHOP: &[RelicId] = &[RelicId::VitruvianMinion];

const NECROBINDER_COMMON: &[RelicId] = &[RelicId::BoneFlute];
const NECROBINDER_UNCOMMON: &[RelicId] = &[RelicId::BookRepairKnife, RelicId::FuneraryMask];
const NECROBINDER_RARE: &[RelicId] = &[RelicId::BigHat, RelicId::Bookmark, RelicId::IvoryTile];
const NECROBINDER_SHOP: &[RelicId] = &[RelicId::UndyingSigil];

// The Fool (v0.113.0, `FoolRelicPool.cs`, alphabetical; SongBow is Starter).
const FOOL_COMMON: &[RelicId] = &[RelicId::ExerciseMat];
const FOOL_UNCOMMON: &[RelicId] = &[RelicId::Ocarina, RelicId::WhittlingKnife];
const FOOL_RARE: &[RelicId] = &[
    RelicId::ConductorsBaton,
    RelicId::Fertilizer,
    RelicId::Twine,
];
const FOOL_SHOP: &[RelicId] = &[RelicId::BizarreLamp];

/// The four `[Common, Uncommon, Rare, Shop]` slices for `character`'s own
/// relic pool. C# maps unimplemented/random characters onto
/// `IroncladRelicPool`, so we do too.
fn character_buckets(character: CharacterId) -> [&'static [RelicId]; 4] {
    match character {
        CharacterId::Ironclad => [
            IRONCLAD_COMMON,
            IRONCLAD_UNCOMMON,
            IRONCLAD_RARE,
            IRONCLAD_SHOP,
        ],
        CharacterId::Silent => [SILENT_COMMON, SILENT_UNCOMMON, SILENT_RARE, SILENT_SHOP],
        CharacterId::Defect => [DEFECT_COMMON, DEFECT_UNCOMMON, DEFECT_RARE, DEFECT_SHOP],
        CharacterId::Regent => [REGENT_COMMON, REGENT_UNCOMMON, REGENT_RARE, REGENT_SHOP],
        CharacterId::Necrobinder => [
            NECROBINDER_COMMON,
            NECROBINDER_UNCOMMON,
            NECROBINDER_RARE,
            NECROBINDER_SHOP,
        ],
        CharacterId::Fool => [FOOL_COMMON, FOOL_UNCOMMON, FOOL_RARE, FOOL_SHOP],
    }
}

/// Shared-pool slices in the same `[Common, Uncommon, Rare, Shop]` order.
const SHARED_BUCKETS: [&[RelicId]; 4] = [SHARED_COMMON, SHARED_UNCOMMON, SHARED_RARE, SHARED_SHOP];

const GRAB_BAG_RARITIES: [RelicRarity; 4] = [
    RelicRarity::Common,
    RelicRarity::Uncommon,
    RelicRarity::Rare,
    RelicRarity::Shop,
];

/// C#-faithful per-rarity grab-bag buckets for `character`.
///
/// Each bucket is `SharedRelicPool` relics of that rarity followed by the
/// character pool's relics of that rarity, both in C# declaration order —
/// the exact pre-shuffle deque order of `RelicGrabBag.Populate`. The
/// caller is responsible for shuffling each bucket with the run RNG (see
/// the PyO3 `RunManager` constructor) in C# Dictionary insertion order
/// (Uncommon → Common → Rare → Shop).
///
/// This is the *player* grab bag (`player.RelicGrabBag`,
/// `RunManager.cs:314`) — the one `RelicFactory.PullNextRelicFromFront`
/// actually draws from. Note that C# *also* populates a separate
/// shared-only bag first ([`shared_grab_bag_buckets`]); see its doc for
/// the RNG-ordering consequence the caller must replicate.
#[must_use]
pub fn grab_bag_buckets(character: CharacterId) -> BTreeMap<RelicRarity, Vec<RelicId>> {
    let char_buckets = character_buckets(character);
    let mut by_rarity = BTreeMap::new();
    for (i, rarity) in GRAB_BAG_RARITIES.into_iter().enumerate() {
        let mut bucket = SHARED_BUCKETS[i].to_vec();
        bucket.extend_from_slice(char_buckets[i]);
        by_rarity.insert(rarity, bucket);
    }
    by_rarity
}

/// The shared-only per-rarity buckets (`SharedRelicPool`, no character
/// pool), in C# declaration order.
///
/// C#'s `RunManager.InitializeNewRun` (`RunManager.cs:309-314`) populates
/// **two** grab bags off the *same* `State.Rng.UpFront` stream, in order:
///
/// 1. `State.SharedRelicGrabBag.Populate(SharedRelicPool.GetUnlockedRelics, UpFront)`
///    — these shared-only buckets, shuffled first.
/// 2. `player.PopulateRelicGrabBagIfNecessary(UpFront)` — the player bag
///    ([`grab_bag_buckets`]), shuffled second.
///
/// Relic rewards (`SmallCapsule`, treasure, …) draw from the *player*
/// bag, but the shared bag's shuffle consumes `UpFront` first. Shuffling
/// only the player bag (as the original port did) starts the player
/// Common shuffle ~111 draws too early, so the post-shuffle front — and
/// thus the seeded `SmallCapsule` roll — diverges from C#. Callers must
/// shuffle these buckets (same Uncommon → Common → Rare → Shop order) to
/// advance `up_front` before shuffling the player bag, even though the
/// shuffled result itself is unused in single-player. Bead
/// `sts2-rs-7e04v` (GH #70).
#[must_use]
pub fn shared_grab_bag_buckets() -> BTreeMap<RelicRarity, Vec<RelicId>> {
    let mut by_rarity = BTreeMap::new();
    for (i, rarity) in GRAB_BAG_RARITIES.into_iter().enumerate() {
        by_rarity.insert(rarity, SHARED_BUCKETS[i].to_vec());
    }
    by_rarity
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Ironclad Common bucket must be the exact 26-entry C# deque:
    /// 25 `SharedRelicPool` Commons (declaration order) then `RedSkull`.
    /// `MealTicket` sits at index 13 — its position is what the seed
    /// `Y3X2YZSBQJ` `SmallCapsule` roll depends on (bead `sts2-rs-7e04v`).
    #[test]
    fn ironclad_common_bucket_matches_csharp_deque() {
        let buckets = grab_bag_buckets(CharacterId::Ironclad);
        let common = &buckets[&RelicRarity::Common];
        assert_eq!(common.len(), 26, "25 shared Commons + RedSkull");
        assert_eq!(common[0], RelicId::AmethystAubergine);
        assert_eq!(common[13], RelicId::MealTicket);
        assert_eq!(common[2], RelicId::BagOfMarbles);
        assert_eq!(common[17], RelicId::RedMask);
        assert_eq!(*common.last().unwrap(), RelicId::RedSkull);
    }

    /// Python-pin rarity drift must be corrected: the two relics the old
    /// `common::pool_relic_ids()` mis-tagged Common are Uncommon here, and
    /// the two `uncommon::` mis-tagged Uncommon are Common.
    #[test]
    fn corrects_python_pin_rarity_drift() {
        let buckets = grab_bag_buckets(CharacterId::Ironclad);
        let common = &buckets[&RelicRarity::Common];
        let uncommon = &buckets[&RelicRarity::Uncommon];
        assert!(uncommon.contains(&RelicId::Permafrost));
        assert!(uncommon.contains(&RelicId::TinyMailbox));
        assert!(!common.contains(&RelicId::Permafrost));
        assert!(!common.contains(&RelicId::TinyMailbox));
        assert!(common.contains(&RelicId::BagOfMarbles));
        assert!(common.contains(&RelicId::RedMask));
    }

    /// Foreign-character commons must NOT leak into the Ironclad pool.
    #[test]
    fn excludes_foreign_character_commons() {
        let buckets = grab_bag_buckets(CharacterId::Ironclad);
        let all: Vec<RelicId> = buckets.values().flatten().copied().collect();
        for leaked in [
            RelicId::BoneFlute,
            RelicId::DataDisk,
            RelicId::FencingManual,
            RelicId::SneckoSkull,
        ] {
            assert!(
                !all.contains(&leaked),
                "{leaked:?} is a foreign-char common"
            );
        }
    }

    /// Bucket sizes match the C# pool extraction (shared + character).
    #[test]
    fn bucket_sizes_match_csharp() {
        let buckets = grab_bag_buckets(CharacterId::Ironclad);
        assert_eq!(buckets[&RelicRarity::Common].len(), 25 + 1);
        assert_eq!(buckets[&RelicRarity::Uncommon].len(), 30 + 2);
        assert_eq!(buckets[&RelicRarity::Rare].len(), 35 + 3);
        assert_eq!(buckets[&RelicRarity::Shop].len(), 25 + 1);
    }

    /// Every character builds non-empty buckets with the character relic
    /// appended after the shared block.
    #[test]
    fn character_relic_appended_after_shared() {
        for (character, common_tail) in [
            (CharacterId::Silent, RelicId::SneckoSkull),
            (CharacterId::Defect, RelicId::DataDisk),
            (CharacterId::Regent, RelicId::FencingManual),
            (CharacterId::Necrobinder, RelicId::BoneFlute),
        ] {
            let buckets = grab_bag_buckets(character);
            let common = &buckets[&RelicRarity::Common];
            assert_eq!(*common.last().unwrap(), common_tail);
            assert_eq!(common[0], RelicId::AmethystAubergine);
        }
    }
}
