//! Relic discriminant.
//!
//! Mirrors `sts2-rl-agent@01c094d:sts2_env/relics/base.py::RelicId`
//! (~190 variants). Bead sts2-rs-u4u (P4-2) ships exactly the
//! `BurningBlood` variant — Ironclad's starter relic. The Phase 7
//! umbrella bead (sts2-rs-h9u) adds the rest of the relic pool plus
//! the after-combat-victory / after-rest / on-card-played
//! interpreter that fires `RelicEffect`s.
//!
//! `#[non_exhaustive]` so adding a variant later does not break
//! downstream `match RelicKind { ... }` arms — they will fail to
//! compile only when the match is exhaustive, which is exactly the
//! correctness invariant we want.
//!
//! ## Crate boundary
//!
//! [`RelicKind`] is the *runtime discriminant* paired with the
//! `Relic` POD in `sts2_core::lib`. It is distinct from
//! `sts2_content::relics::RelicSpec`, which is the *static template*
//! (immutable data shared across all instances of the same relic).
//! Phase 7 will add a `RelicSpec -> Relic` materialization step
//! at run-build / relic-pickup time. This split mirrors the
//! `CardId` / `CardSpec` split established by bead sts2-rs-zcl
//! (P4-1) and the `MonsterKind` / future-monster-DSL split
//! established by bead sts2-rs-pg5 (P4-3).
//!
//! ## `RelicId` vs `RelicKind` (bead sts2-rs-l7l / P7-1)
//!
//! [`RelicId`] is the auto-generated, exhaustive Python-pinned enum
//! (one variant per `RelicId.<NAME>` member at pin `01c094d`, 289
//! variants). The `relic_effects` dispatch registry keys handlers by
//! [`RelicId`].
//!
//! [`RelicKind`] (legacy, shipped by bead `sts2-rs-u4u` / P4-2) is
//! `#[non_exhaustive]` and only carries variants actually backed by a
//! `RelicSpec` factory in `sts2-content`. It exists for backwards
//! compatibility with [`crate::Relic::kind`] until the P7-2..P7-5
//! pool ports migrate to keying on [`RelicId`] directly. New code
//! should prefer [`RelicId`].

#[path = "relic_ids_generated.rs"]
mod relic_ids_generated;
pub use relic_ids_generated::RelicId;

/// Declares [`RelicKind`] together with a mechanically-derived
/// [`RelicKind::ALL`] roster.
///
/// **Why a macro (2026-07-17 soft-drop census fix, wave 3):** the
/// downstream registry mirrors (`sts2-py::state_dict::RELIC_KINDS`,
/// `sts2-cli::sim::scenario::relics::RELIC_KINDS`) were hand-maintained
/// copies of this enum's variant list, and three separate waves of
/// pycli soft-drop bugs (commits `8d81f85f5`, `9d0cae1c4`, plus the
/// wave-3 census) were all the SAME failure mode: a variant existed
/// here — often with a fully-wired combat listener — but was missing
/// from a mirror, so `list_registry_names("relic")` lied about the
/// engine and the MCTS adapter silently dropped the relic from every
/// search state. Generating `ALL` from the declaration itself makes
/// that class of drift impossible: a new variant is in `ALL` (and in
/// every mirror that aliases `ALL`) by construction, with no opt-out.
macro_rules! declare_relic_kinds {
    (
        $(#[$enum_meta:meta])*
        pub enum RelicKind {
            $($(#[$variant_meta:meta])* $variant:ident,)+
        }
    ) => {
        $(#[$enum_meta])*
        pub enum RelicKind {
            $($(#[$variant_meta])* $variant,)+
        }

        impl RelicKind {
            /// Every `RelicKind` variant, in declaration order. Derived
            /// mechanically by [`declare_relic_kinds!`] from the enum
            /// declaration itself, so a variant can never be missing
            /// from this roster. Downstream registry mirrors
            /// (`sts2-py::state_dict::RELIC_KINDS`,
            /// `sts2-cli::sim::scenario::relics::RELIC_KINDS`) alias
            /// this const instead of maintaining their own copies.
            pub const ALL: &'static [RelicKind] = &[$(RelicKind::$variant,)+];
        }
    };
}

declare_relic_kinds! {
/// Discriminant for [`crate::Relic`]. Names mirror Python's
/// `RelicId.<NAME>` exactly so the (future Phase-7) parity exporter's
/// [`python_name`](Self::python_name) returns the same string Python
/// emits via `RelicId.<v>.name`.
///
/// Variant order mirrors Python's `RelicId` declaration order at
/// `sts2_env/relics/base.py:27-` so a walk over [`RelicKind::ALL`]
/// can be paired 1-to-1 with Python's enum iterator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RelicKind {
    // ── Starter relics ──────────────────────────────────────────────────
    /// Burning Blood — Ironclad's starter relic. At end of combat
    /// (after victory), heal the owner for 6 HP. Python class:
    /// `sts2_env/relics/starter.py::BurningBlood`. Hook:
    /// `after_combat_victory`.
    BurningBlood,
    /// Cracked Core — Defect starter (Channel 1 Lightning orb round 1).
    /// `sts2_env/relics/starter.py::CrackedCore`. Ported by P7-2.
    CrackedCore,
    /// Ring of the Snake — Silent starter (+2 cards on round 1).
    /// `sts2_env/relics/starter.py::RingOfTheSnake`. Ported by P7-2.
    RingOfTheSnake,
    /// Divine Right — Regent starter (+3 Stars on round 1). Replaces
    /// vanilla STS's `PureWater` (not present in the fork).
    /// `sts2_env/relics/starter.py::DivineRight`. Ported by P7-2.
    DivineRight,
    /// Bound Phylactery — Necrobinder starter. Summons 1 Osty at
    /// combat start and on every energy-reset after round 1.
    /// `sts2_env/relics/starter.py::BoundPhylactery`. Plumbing
    /// landed by `sts2-rs-qeo` (umbrella `sts2-rs-bay`); the
    /// `summon_osty` runtime + on-trigger handler land with the
    /// Necrobinder character sub-bead.
    BoundPhylactery,

    // ── Common relics (sts2_env/relics/common.py) — ported by P7-2 ──
    AmethystAubergine,
    Anchor,
    BagOfPreparation,
    BloodVial,
    BoneFlute,
    BronzeScales,
    CentennialPuzzle,
    DataDisk,
    FencingManual,
    FestivePopper,
    Gorget,
    HappyFlower,
    JuzuBracelet,
    Lantern,
    MealTicket,
    OddlySmoothStone,
    Pendulum,
    Permafrost,
    PotionBelt,
    RedSkull,
    RegalPillow,
    SneckoSkull,
    Strawberry,
    StrikeDummy,
    TinyMailbox,
    Vajra,
    VenerableTeaSet,
    WarPaint,
    Whetstone,
    BookOfFiveRings,

    // ── Uncommon relics referenced for deep-test parity (P7-2) ──
    // Full uncommon roster lives in `sts2-content::relics::uncommon`
    // (bead `sts2-rs-l22`); only the two relics that the P7-2 deep
    // tests assert against were mirrored here originally — bead
    // `sts2-rs-6ynt.5.5` then extended the A–K alphabetical half of
    // the uncommon pool with `RelicSpec` factories, so the rest of
    // the A–K uncommons live here too.
    Akabeko,
    BagOfMarbles,
    Bellows,
    BookRepairKnife,
    BowlerHat,
    Candelabra,
    EternalFeather,
    FuneraryMask,
    GalacticDust,
    GoldPlatedCables,
    GremlinHorn,
    HornCleat,
    JossPaper,
    Kusarigama,
    // L–Z uncommon roster (bead `sts2-rs-6ynt.5.6`).
    LetterOpener,
    LuckyFysh,
    MercuryHourglass,
    MiniatureCannon,
    Nunchaku,
    Orichalcum,
    OrnamentalFan,
    Pantograph,
    PaperPhrog,
    ParryingShield,
    Pear,
    PenNib,
    PetrifiedToad,
    Planisphere,
    RedMask,
    Regalite,
    ReptileTrinket,
    RippleBasin,
    SelfFormingClay,
    SparklingRouge,
    StoneCracker,
    SymbioticVirus,
    Tingsha,
    TuningFork,
    TwistedFunnel,
    Vambrace,

    // ── Event-reward relics (sts2_env/relics/shop_event.py) — ported by P7-5b ──
    // 35 `RelicRarity.EVENT` entries from `shop_event.py`. Per-relic
    // factories live in `sts2-content::relics::event_reward` (bead
    // `sts2-rs-7ez`). The complementary `pool=EVENT`/`rarity=ANCIENT`
    // bucket plus `RelicRarity.SHOP` entries belong to sibling bead
    // `sts2-rs-yyw`.
    BigMushroom,
    BingBong,
    BoneTea,
    Byrdpip,
    ChosenCheese,
    DarkstonePeriapt,
    DaughterOfTheWind,
    DreamCatcher,
    EmberTea,
    FakeAnchor,
    FakeBloodVial,
    FakeHappyFlower,
    FakeLeesWaffle,
    FakeMango,
    FakeMerchantsRug,
    FakeOrichalcum,
    FakeSneckoEye,
    FakeStrikeDummy,
    FakeVenerableTeaSet,
    ForgottenSoul,
    FragrantMushroom,
    FresnelLens,
    HandDrill,
    HistoryCourse,
    LostWisp,
    MawBank,
    MrStruggles,
    PollinousCore,
    RoyalPoison,
    SwordOfStone,
    SwordOfJade,
    TeaOfDiscourtesy,
    TheBoot,
    WongoCustomerAppreciationBadge,
    WongosMysteryTicket,

    // ── Rare relics (sts2_env/relics/rare.py) — bead sts2-rs-042x ──
    // Variants are gates only — `player_owns_relic(state, RelicKind::X)`
    // listener checks in `relic_effects.rs`. Body wiring lands with
    // `sts2-rs-tohx` (rare), `sts2-rs-o39p` (boss-tier subset), and
    // `sts2-rs-k8vc` (shop-event remainder).
    ArtOfWar,
    BeatingRemnant,
    BigHat,
    Bookmark,
    CaptainsWheel,
    Chandelier,
    CharonsAshes,
    CloakClasp,
    DemonTongue,
    EmotionChip,
    FrozenEgg,
    GamblingChip,
    GamePiece,
    Girya,
    HelicalDart,
    IceCream,
    IntimidatingHelmet,
    IvoryTile,
    Kunai,
    LastingCandy,
    LizardTail,
    LunarPastry,
    Mango,
    MeatOnTheBone,
    Metronome,
    MiniRegent,
    MoltenEgg,
    MummifiedHand,
    OldCoin,
    OrangeDough,
    PaperKrane,
    Pocketwatch,
    PowerCell,
    PrayerWheel,
    RainbowRing,
    RazorTooth,
    RuinedHelmet,
    Shovel,
    Shuriken,
    StoneCalendar,
    SturdyClamp,
    TheCourier,
    ToughBandages,
    ToxicEgg,
    TungstenRod,
    UnceasingTop,
    UnsettlingLamp,
    VexingPuzzlebox,
    WhiteBeastStatue,
    WhiteStar,

    // ── Boss-tier (ANCIENT) wave-0 gates (sts2_env/relics/shop_event.py) ──
    // Variants added incrementally as listener bodies are wired in
    // `relic_effects.rs`. Bead `sts2-rs-o39p` (boss-tier ports).
    /// Brimstone — at start of each player turn, gain `STRENGTH(+2)`
    /// and every alive enemy gains `STRENGTH(+1)`. Hook:
    /// `before_side_turn_start`. Bead `sts2-rs-o39p`.
    Brimstone,
    /// BoomingConch — round 1 in elite combat, draw +2 cards. Hook:
    /// `modify_hand_draw` (round + elite gated). Bead `sts2-rs-o39p`
    /// wave 1.
    BoomingConch,
    /// TheAbacus — on each draw-pile shuffle, gain `THE_ABACUS_BLOCK=6`.
    /// Hook: `after_shuffle_bus`. Bead `sts2-rs-k8vc` wave 0.
    TheAbacus,
    /// SlingOfCourage — at combat start in elite rooms, gain
    /// `SLING_OF_COURAGE_STRENGTH=2`. Hook: `before_combat_start`
    /// (elite-gated). Bead `sts2-rs-k8vc` wave 1.
    SlingOfCourage,

    // ── Event-pool starter upgrades (sts2_env/relics/starter.py) ──
    /// Black Blood — event-pool upgrade of `BurningBlood` (HEAL=12).
    /// Bead `sts2-rs-dp7o`. Hook: `after_combat_victory`.
    BlackBlood,
    /// Ring of the Drake — event-pool upgrade of `RingOfTheSnake`
    /// (EXTRA_CARDS=2, TURNS=3). Bead `sts2-rs-mvrj`. Hook:
    /// `modify_hand_draw`.
    RingOfTheDrake,
    /// Infused Core — event-pool upgrade of `CrackedCore`
    /// (LIGHTNING_COUNT=3, round 1). Bead `sts2-rs-6ynt.5.2`. Hook:
    /// `after_side_turn_start`.
    InfusedCore,
    /// Phylactery Unbound — event-pool upgrade of `BoundPhylactery`
    /// (START_COMBAT_COUNT=5, START_TURN_COUNT=2). Bead
    /// `sts2-rs-6ynt.5.2`. Hooks: `before_combat_start` +
    /// `after_side_turn_start`.
    PhylacteryUnbound,
    /// Divine Destiny — event-pool upgrade of `DivineRight` (STARS=6,
    /// round 1). Bead `sts2-rs-6ynt.5.2`. Hook:
    /// `after_side_turn_start`.
    DivineDestiny,
    // ── sts2log gap-report markers (bead `sts2-rs-pgvl`) ─────────────────
    //
    // Pure runtime-discriminant markers — listener bodies stay deferred
    // to the P7 relic-surface umbrella (`sts2-rs-01o` and children).
    // Adding these variants unblocks `seed_relics` for 40+ fixtures in
    // `docs/reality-checks/2026-05-16-sts2log-content-gap-report.md` §93.
    /// Scroll Boxes (ANCIENT) — pay all gold, choose from 2 bundles of
    /// cards. Marker — body deferred.
    ScrollBoxes,
    /// Lead Paperweight — choice in shops; +50 gold when defeating
    /// Bosses. Marker — body deferred.
    LeadPaperweight,
    /// Arcane Scroll — each time you exhaust a card, gain 1 random
    /// colorless card. Marker — body deferred.
    ArcaneScroll,
    /// New Leaf — at start of each combat, draw 1 additional card on
    /// the first turn. Marker — body deferred.
    NewLeaf,
    /// Precarious Shears — at start of combat, remove a random card
    /// from your hand. Marker — body deferred.
    PrecariousShears,
    /// Leafy Poultice — when you take unblocked damage, heal 4 HP.
    /// Marker — body deferred.
    LeafyPoultice,
    /// Cursed Pearl — at the start of combat, gain a random curse in
    /// your draw pile. Marker — body deferred.
    CursedPearl,
    // ── s5vb validation blockers (bead `sts2-rs-pibg`) ───────────────────
    //
    // Top 4 runtime-RelicKind gaps that sim-error 26/29 deck-perfect
    // fixtures in the rebuilt sts2log corpus (db35cb0). Adding the
    // markers + RELIC_KINDS table entries restores load-ability so MCTS
    // validation gates can measure engine quality on these fixtures.
    // Dispatch handlers already exist in `sts2-content::relics::shop_event`
    // for the three event-pool relics; bodies wired via the existing
    // `on_trigger_*` entries.
    /// Lost Coffer (ANCIENT) — offer 1 extra card reward + 1 potion
    /// reward at end of combat. Marker — `on_trigger_lost_coffer` in
    /// shop_event.rs covers dispatch. Bead `sts2-rs-pibg`.
    LostCoffer,
    /// Pomander (ANCIENT) — upgrade 1 random card from your deck on
    /// pickup. Marker — `on_trigger_pomander` in shop_event.rs covers
    /// dispatch. Bead `sts2-rs-pibg`.
    Pomander,
    /// Silver Crucible (ANCIENT) — first 3 card rewards in combat all
    /// arrive pre-upgraded; skip the first treasure room. Marker —
    /// `on_trigger_silver_crucible` in shop_event.rs covers dispatch.
    /// Bead `sts2-rs-pibg`.
    SilverCrucible,
    /// Fishing Rod — Ancient/Neow relic. Every 3rd normal (Monster)
    /// combat, upgrade a random upgradable deck card (RNG: Niche). Now
    /// paired with `RelicId::FishingRod`; behavior lives in
    /// `sts2-content::run_bridge::exit_combat`. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/FishingRod.cs`. Bead
    /// `sts2-rs-v0107-content-port-wb8lk.6`.
    FishingRod,
    /// Kaleidoscope — Ancient/Neow relic (all-character-pools unlocked).
    /// On pickup, offer 2 card rewards each drawn from 3 random
    /// *other*-character pools. Offered at Neow, but its `AfterObtained`
    /// cross-character-pool draw is DEFERRED pending the
    /// `CardRewardAlternative` reward surface (port doc §0). C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/Kaleidoscope.cs`. Bead
    /// `sts2-rs-v0107-content-port-wb8lk.6`.
    Kaleidoscope,
    /// Silken Tress — Ancient/Neow curse relic. On pickup, lose ALL gold;
    /// the first card reward afterward is enchanted with Glam (one-time,
    /// `IsUsedUp` latch). Behavior lives in
    /// `sts2-content::relics::neow_relics`. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/SilkenTress.cs`. Bead
    /// `sts2-rs-v0107-content-port-wb8lk.6`.
    SilkenTress,
    // ── fwxe.5 runtime-RelicKind blockers (bead `sts2-rs-8uvz`) ──────────
    //
    // Three RelicId variants that already exist in
    // `relic_ids_generated.rs` but lacked corresponding RelicKind
    // entries — 19 fixtures (SmallCapsule x14, ArchaicTooth x3,
    // PaelsEye x2) in the fwxe baseline failed with "no runtime
    // RelicKind". Markers only; bodies deferred to future dispatch
    // follow-ups when they fire on parity fixtures.
    /// Small Capsule — likely a card-reward modifier. Marker —
    /// body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/SmallCapsule.cs`.
    /// Bead `sts2-rs-8uvz`.
    SmallCapsule,
    /// Archaic Tooth — likely curse-related per naming. Marker —
    /// body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/ArchaicTooth.cs`.
    /// Bead `sts2-rs-8uvz`.
    ArchaicTooth,
    /// Pael's Eye — likely an observation/map relic. Marker —
    /// body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/PaelsEye.cs`.
    /// Bead `sts2-rs-8uvz`.
    PaelsEye,
    /// Astrolabe — boss-relic; transform 3 cards in deck on pickup.
    /// Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/Astrolabe.cs`. Bead
    /// `sts2-rs-97t6` (fwxe.10): 6 fixtures in fwxe.7's sim-error
    /// breakdown failed with "no runtime RelicKind" on this id.
    Astrolabe,
    // ── fwxe.8 runtime-RelicKind blockers (bead `sts2-rs-qffh`) ──────────
    //
    // Two more RelicId variants present in `relic_ids_generated.rs` but
    // missing RelicKind entries — 6 fixtures (PaelsGrowth x3,
    // NutritiousSoup x3) in the post-fwxe.7 baseline failed with
    // "no runtime RelicKind". (UnsettlingLamp already had a RelicKind
    // entry above; only its RELIC_KINDS table registration was missing
    // — 2 fixtures.) Markers only; bodies deferred.
    /// Pael's Growth — likely a passive scaling relic per the
    /// Pael's-* family naming. Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/PaelsGrowth.cs`.
    /// Bead `sts2-rs-qffh`.
    PaelsGrowth,
    /// Nutritious Soup — likely a healing / max-HP relic per naming.
    /// Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/NutritiousSoup.cs`.
    /// Bead `sts2-rs-qffh`.
    NutritiousSoup,
    /// Very Hot Cocoa — likely a healing / energy / warmth relic per
    /// naming. Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/VeryHotCocoa.cs`. Bead
    /// `sts2-rs-wc7h` (fwxe.12): 7 fixtures in fwxe.11's sim-error
    /// breakdown failed with "no runtime RelicKind" on this id.
    VeryHotCocoa,
    /// Delicate Frond — likely a fragile / one-shot relic per naming.
    /// Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/DelicateFrond.cs`. Bead
    /// `sts2-rs-i9fl` (fwxe.14): 7 fixtures in fwxe.13's sim-error
    /// breakdown.
    DelicateFrond,
    /// Tri Boomerang — likely a multi-hit attack-modifier relic per
    /// naming. Marker — body deferred. C# decomp at
    /// `MegaCrit.Sts2.Core.Models.Relics/TriBoomerang.cs`. Bead
    /// `sts2-rs-i9fl` (fwxe.14): 6 fixtures in fwxe.13's sim-error
    /// breakdown.
    TriBoomerang,
    // ── cia9 batch-extract (bead `sts2-rs-cia9`) ──────────────────────
    //
    // Batch-add 101 RelicKind variants that exist in `RelicId` (Python
    // pin `sts2-rl-agent@01c094d`) but had no RelicKind counterpart.
    // Closes the systemic gap that `sts2-rs-84ys` / `8uvz` / `97t6` /
    // `qffh` / `wc7h` / `i9fl` / `s1nv` / `asaq` were draining
    // incrementally. Each variant is a pure runtime discriminant
    // (marker only); per-relic effect bodies remain follow-up work
    // filed under the existing relic-surface umbrella when fixtures
    // demand them. Names cover starter/common/uncommon/rare/event/shop
    // pools indiscriminately — the rarity grouping lives in
    // `sts2-content::relics::*` factory installers, not in this enum.
    AlchemicalCoffer,
    BeautifulBracelet,
    BeltBuckle,
    BiiigHug,
    BlackStar,
    BlessedAntler,
    BloodSoakedRose,
    Bread,
    BrilliantScarf,
    BurningSticks,
    CallingBell,
    Cauldron,
    ChemicalX,
    ChoicesParadox,
    Circlet,
    Claws,
    Crossbow,
    DeprecatedRelic,
    DiamondDiadem,
    DingyRug,
    DistinguishedCape,
    DollysMirror,
    DragonFruit,
    Driftwood,
    DustyTome,
    Ectoplasm,
    ElectricShrymp,
    EmptyCage,
    Fiddle,
    FurCoat,
    GhostSeed,
    GlassEye,
    Glitter,
    GnarledHammer,
    GoldenCompass,
    GoldenPearl,
    IronClub,
    JeweledMask,
    JewelryBox,
    Kifuda,
    LargeCapsule,
    LavaLamp,
    LavaRock,
    LeesWaffle,
    LoomingFruit,
    LordsParasol,
    MassiveScroll,
    MeatCleaver,
    MembershipCard,
    MiniatureTent,
    MusicBox,
    MysticLighter,
    NeowsTorment,
    NinjaScroll,
    NutritiousOyster,
    Orrery,
    PaelsBlood,
    PaelsClaw,
    PaelsFlesh,
    PaelsHorn,
    PaelsLegion,
    PaelsTears,
    PaelsTooth,
    PaelsWing,
    PandorasBox,
    PhilosophersStone,
    PreciseScissors,
    PreservedFog,
    PrismaticGem,
    PumpkinCandle,
    PunchDagger,
    RadiantPearl,
    RingingTriangle,
    RoyalStamp,
    RunicCapacitor,
    RunicPyramid,
    Sai,
    SandCastle,
    ScreamingFlagon,
    SeaGlass,
    SealOfGold,
    SereTalon,
    SignetRing,
    SneckoEye,
    Sozu,
    SpikedGauntlets,
    StoneHumidifier,
    Storybook,
    TanxsWhistle,
    ThrowingAxe,
    ToastyMittens,
    Toolbox,
    TouchOfOrobas,
    ToyBox,
    UndyingSigil,
    VelvetChoker,
    VitruvianMinion,
    WarHammer,
    WhisperingEarring,
    WingCharm,
    YummyCookie,
    // ── combat-inert Neow reward markers (soft-drop census 2026-07-17) ──
    //
    // Both relics are pure run/map-time per the v0.107.1 C# decomp and
    // have ZERO combat hooks, but the pycli hint adapter soft-drops any
    // relic whose id is missing from `list_registry_names("relic")`
    // (879 drop events for these two across the seeds-1000-1019 census).
    // Pairing them with a marker `RelicKind` — the same no-op pattern as
    // the cia9 batch above — lets the registry mirror list them so
    // real-game states carrying them reconstruct without drops. Their
    // run-side pickup behaviors stay deferred (W9 / bead `sts2-rs-4rwv4`).
    /// Neow's Talisman — on pickup, upgrade the last Basic Strike and
    /// the last Basic Defend in the deck (C#
    /// `MegaCrit.Sts2.Core.Models.Relics/NeowsTalisman.cs::AfterObtained`).
    /// Combat-inert marker — no combat hooks in the decomp.
    NeowsTalisman,
    /// Winged Boots — free map travel (ignore path connections) 3 times
    /// (C# `MegaCrit.Sts2.Core.Models.Relics/WingedBoots.cs::
    /// ShouldAllowFreeTravel` / `AfterRoomEntered`). Combat-inert
    /// marker — no combat hooks in the decomp.
    WingedBoots,
    // ── remaining W9 Neow reward markers (soft-drop census wave 3, 2026-07-17) ──
    //
    // The wave-3 census surfaced PHIAL_HOLSTER (via its leaked
    // `FISHING_ROD.title`-style loc-key resolution path) still being
    // soft-dropped: it was one of the last three `RelicId`s with no
    // combat-side `RelicKind` (the deliberate W9 `to_relic_kind → None`
    // deferral, bead `sts2-rs-4rwv4`). All three are pure pickup-time
    // per the v0.107.1 C# decomp — ZERO combat hooks — and their
    // run-side `after_obtained` listeners already exist in
    // `sts2-content::relics::neow_relics` keyed by `RelicId`. Pairing
    // them with no-op marker kinds (the NeowsTalisman / WingedBoots
    // pattern) makes `RelicId::to_relic_kind` total and lets the
    // registry mirror (`RelicKind::ALL`) list them.
    /// Phial Holster — on pickup, +1 max potion slot, then procure 2
    /// random out-of-combat potions (C#
    /// `MegaCrit.Sts2.Core.Models.Relics/PhialHolster.cs::AfterObtained`).
    /// Combat-inert marker; run-side pickup behavior:
    /// `neow_relics::listener_phial_holster` (bead `sts2-rs-zgy5m`).
    PhialHolster,
    /// Neow's Bones — on pickup, 2-relic shuffle reward + Niche curse
    /// (C# `MegaCrit.Sts2.Core.Models.Relics/NeowsBones.cs::AfterObtained`).
    /// Combat-inert marker; run-side pickup behavior:
    /// `neow_relics::listener_neows_bones` (bead `sts2-rs-wjuoi`).
    NeowsBones,
    /// Hefty Tablet — on pickup, skippable 1-of-3 Rare card select +
    /// 1 Injury curse (C#
    /// `MegaCrit.Sts2.Core.Models.Relics/HeftyTablet.cs::AfterObtained`).
    /// Combat-inert marker; run-side pickup behavior:
    /// `neow_relics::listener_hefty_tablet` (bead `sts2-rs-w6hdj`).
    HeftyTablet,
    // ── v0.113.0-only: The Fool's relics (`RelicId::V0113_ONLY`). Their
    // combat listeners live in `crate::fool`. They are left out of the
    // `"relic"` registry (`sts2-wire`), whose sorted rows are a combat-ctx
    // embedding; see [`RelicKind::is_v0113_only`].
    /// v0.113.0 Fool relic, see [`RelicId::SongBow`].
    SongBow,
    /// v0.113.0 Fool relic, see [`RelicId::Storyteller`].
    Storyteller,
    /// v0.113.0 Fool relic, see [`RelicId::BizarreLamp`].
    BizarreLamp,
    /// v0.113.0 Fool relic, see [`RelicId::ConductorsBaton`].
    ConductorsBaton,
    /// v0.113.0 Fool relic, see [`RelicId::ExerciseMat`].
    ExerciseMat,
    /// v0.113.0 Fool relic, see [`RelicId::Fertilizer`].
    Fertilizer,
    /// v0.113.0 Fool relic, see [`RelicId::Ocarina`].
    Ocarina,
    /// v0.113.0 Fool relic, see [`RelicId::Twine`].
    Twine,
    /// v0.113.0 Fool relic, see [`RelicId::WhittlingKnife`].
    WhittlingKnife,
    /// v0.113.0 event relic, see [`RelicId::Byrdine`] (Byrdpip's evolution).
    /// Its only combat hook is the `BeforeCombatStart` pet summon.
    Byrdine,
    // ── v0.113.0-only event relics with a combat hook. Listeners in
    // `crate::relic_effects::v0113_event`; also off the `"relic"` registry.
    /// v0.113.0 event relic, see [`RelicId::ObsidianVice`].
    ObsidianVice,
    /// v0.113.0 event relic, see [`RelicId::TormentousCollar`].
    TormentousCollar,
    /// v0.113.0 event relic, see [`RelicId::MightyWings`].
    MightyWings,
    /// v0.113.0 event relic, see [`RelicId::BathTowel`].
    BathTowel,
    /// v0.113.0 event relic, see [`RelicId::BathRobes`].
    BathRobes,
}
}

impl RelicId {
    /// Whether the relic's C# `Rarity` is `RelicRarity.Starter`: the six
    /// characters' starters and their Touch-of-Orobas upgrades
    /// (`BurningBlood.cs`, `BlackBlood.cs`, ..., v0.113.0 `SongBow.cs:15`,
    /// `Storyteller.cs:20`). `TouchOfOrobas.GetStarterRelic` and Orobas'
    /// `OptionPool3` key on it.
    #[must_use]
    pub const fn is_starter_rarity(self) -> bool {
        matches!(
            self,
            Self::BurningBlood
                | Self::BlackBlood
                | Self::RingOfTheSnake
                | Self::RingOfTheDrake
                | Self::CrackedCore
                | Self::InfusedCore
                | Self::BoundPhylactery
                | Self::PhylacteryUnbound
                | Self::DivineRight
                | Self::DivineDestiny
                | Self::SongBow
                | Self::Storyteller
        )
    }

    /// `TouchOfOrobas.RefinementUpgrades` (`TouchOfOrobas.cs:21-54`, v0.113.0
    /// adds SongBow -> Storyteller): the Ancient version of a starter relic,
    /// `None` for a relic with no entry (`GetUpgradedStarterRelic` then
    /// falls back to Circlet).
    #[must_use]
    pub const fn refined_starter(self) -> Option<Self> {
        match self {
            Self::BurningBlood => Some(Self::BlackBlood),
            Self::RingOfTheSnake => Some(Self::RingOfTheDrake),
            Self::CrackedCore => Some(Self::InfusedCore),
            Self::BoundPhylactery => Some(Self::PhylacteryUnbound),
            Self::DivineRight => Some(Self::DivineDestiny),
            Self::SongBow => Some(Self::Storyteller),
            _ => None,
        }
    }

    /// Map this run-side relic identifier to its combat-side
    /// [`RelicKind`] discriminant, or `None` when the relic has no
    /// combat-side representation.
    ///
    /// The two enums share variant names 1-to-1 (both generated to
    /// mirror Python's `RelicId.<NAME>`), so this is a pure name
    /// projection. As of the 2026-07-17 soft-drop census wave 3, the
    /// projection is **total** — the last three W9 Neow reward relics
    /// (`HeftyTablet`, `NeowsBones`, `PhialHolster`, deferred by bead
    /// `sts2-rs-4rwv4`) gained combat-inert marker kinds, following
    /// `NeowsTalisman` / `WingedBoots` from wave 2. The signature keeps
    /// `Option` so a future C#-superset `RelicId` without a kind stays
    /// representable; `to_relic_kind_is_total` pins the current
    /// all-`Some` state.
    ///
    /// This is the bridge that lets [`crate::Relic`] rows be
    /// materialized from a [`crate::runstate::RunState`]'s relic list
    /// at combat entry (`sts2-content::run_bridge::enter_combat`), so
    /// the registered per-relic listeners — which self-guard via
    /// `player_owns_relic(state, kind)` — actually fire in
    /// run-bridged combat.
    #[must_use]
    pub const fn to_relic_kind(self) -> Option<RelicKind> {
        match self {
            Self::BurningBlood => Some(RelicKind::BurningBlood),
            Self::BlackBlood => Some(RelicKind::BlackBlood),
            Self::RingOfTheSnake => Some(RelicKind::RingOfTheSnake),
            Self::RingOfTheDrake => Some(RelicKind::RingOfTheDrake),
            Self::CrackedCore => Some(RelicKind::CrackedCore),
            Self::InfusedCore => Some(RelicKind::InfusedCore),
            Self::BoundPhylactery => Some(RelicKind::BoundPhylactery),
            Self::PhylacteryUnbound => Some(RelicKind::PhylacteryUnbound),
            Self::DivineRight => Some(RelicKind::DivineRight),
            Self::DivineDestiny => Some(RelicKind::DivineDestiny),
            Self::AmethystAubergine => Some(RelicKind::AmethystAubergine),
            Self::Anchor => Some(RelicKind::Anchor),
            Self::BagOfPreparation => Some(RelicKind::BagOfPreparation),
            Self::BloodVial => Some(RelicKind::BloodVial),
            Self::BoneFlute => Some(RelicKind::BoneFlute),
            Self::BronzeScales => Some(RelicKind::BronzeScales),
            Self::CentennialPuzzle => Some(RelicKind::CentennialPuzzle),
            Self::DataDisk => Some(RelicKind::DataDisk),
            Self::FencingManual => Some(RelicKind::FencingManual),
            Self::FestivePopper => Some(RelicKind::FestivePopper),
            Self::Gorget => Some(RelicKind::Gorget),
            Self::HappyFlower => Some(RelicKind::HappyFlower),
            Self::JuzuBracelet => Some(RelicKind::JuzuBracelet),
            Self::Lantern => Some(RelicKind::Lantern),
            Self::MealTicket => Some(RelicKind::MealTicket),
            Self::OddlySmoothStone => Some(RelicKind::OddlySmoothStone),
            Self::Pendulum => Some(RelicKind::Pendulum),
            Self::Permafrost => Some(RelicKind::Permafrost),
            Self::PotionBelt => Some(RelicKind::PotionBelt),
            Self::RedSkull => Some(RelicKind::RedSkull),
            Self::RegalPillow => Some(RelicKind::RegalPillow),
            Self::SneckoSkull => Some(RelicKind::SneckoSkull),
            Self::Strawberry => Some(RelicKind::Strawberry),
            Self::StrikeDummy => Some(RelicKind::StrikeDummy),
            Self::TinyMailbox => Some(RelicKind::TinyMailbox),
            Self::Vajra => Some(RelicKind::Vajra),
            Self::VenerableTeaSet => Some(RelicKind::VenerableTeaSet),
            Self::WarPaint => Some(RelicKind::WarPaint),
            Self::Whetstone => Some(RelicKind::Whetstone),
            Self::BookOfFiveRings => Some(RelicKind::BookOfFiveRings),
            Self::Akabeko => Some(RelicKind::Akabeko),
            Self::BagOfMarbles => Some(RelicKind::BagOfMarbles),
            Self::Bellows => Some(RelicKind::Bellows),
            Self::BookRepairKnife => Some(RelicKind::BookRepairKnife),
            Self::BowlerHat => Some(RelicKind::BowlerHat),
            Self::Candelabra => Some(RelicKind::Candelabra),
            Self::EternalFeather => Some(RelicKind::EternalFeather),
            Self::FuneraryMask => Some(RelicKind::FuneraryMask),
            Self::GalacticDust => Some(RelicKind::GalacticDust),
            Self::GoldPlatedCables => Some(RelicKind::GoldPlatedCables),
            Self::GremlinHorn => Some(RelicKind::GremlinHorn),
            Self::HornCleat => Some(RelicKind::HornCleat),
            Self::JossPaper => Some(RelicKind::JossPaper),
            Self::Kusarigama => Some(RelicKind::Kusarigama),
            Self::LetterOpener => Some(RelicKind::LetterOpener),
            Self::LuckyFysh => Some(RelicKind::LuckyFysh),
            Self::MercuryHourglass => Some(RelicKind::MercuryHourglass),
            Self::MiniatureCannon => Some(RelicKind::MiniatureCannon),
            Self::Nunchaku => Some(RelicKind::Nunchaku),
            Self::Orichalcum => Some(RelicKind::Orichalcum),
            Self::OrnamentalFan => Some(RelicKind::OrnamentalFan),
            Self::Pantograph => Some(RelicKind::Pantograph),
            Self::PaperPhrog => Some(RelicKind::PaperPhrog),
            Self::ParryingShield => Some(RelicKind::ParryingShield),
            Self::Pear => Some(RelicKind::Pear),
            Self::PenNib => Some(RelicKind::PenNib),
            Self::PetrifiedToad => Some(RelicKind::PetrifiedToad),
            Self::Planisphere => Some(RelicKind::Planisphere),
            Self::RedMask => Some(RelicKind::RedMask),
            Self::Regalite => Some(RelicKind::Regalite),
            Self::ReptileTrinket => Some(RelicKind::ReptileTrinket),
            Self::RippleBasin => Some(RelicKind::RippleBasin),
            Self::SelfFormingClay => Some(RelicKind::SelfFormingClay),
            Self::SparklingRouge => Some(RelicKind::SparklingRouge),
            Self::StoneCracker => Some(RelicKind::StoneCracker),
            Self::TuningFork => Some(RelicKind::TuningFork),
            Self::Tingsha => Some(RelicKind::Tingsha),
            Self::Vambrace => Some(RelicKind::Vambrace),
            Self::ArtOfWar => Some(RelicKind::ArtOfWar),
            Self::BeatingRemnant => Some(RelicKind::BeatingRemnant),
            Self::BigHat => Some(RelicKind::BigHat),
            Self::Bookmark => Some(RelicKind::Bookmark),
            Self::CaptainsWheel => Some(RelicKind::CaptainsWheel),
            Self::Chandelier => Some(RelicKind::Chandelier),
            Self::CharonsAshes => Some(RelicKind::CharonsAshes),
            Self::DemonTongue => Some(RelicKind::DemonTongue),
            Self::EmotionChip => Some(RelicKind::EmotionChip),
            Self::FrozenEgg => Some(RelicKind::FrozenEgg),
            Self::GamblingChip => Some(RelicKind::GamblingChip),
            Self::GamePiece => Some(RelicKind::GamePiece),
            Self::Girya => Some(RelicKind::Girya),
            Self::HelicalDart => Some(RelicKind::HelicalDart),
            Self::IceCream => Some(RelicKind::IceCream),
            Self::IntimidatingHelmet => Some(RelicKind::IntimidatingHelmet),
            Self::IvoryTile => Some(RelicKind::IvoryTile),
            Self::Kunai => Some(RelicKind::Kunai),
            Self::LastingCandy => Some(RelicKind::LastingCandy),
            Self::LizardTail => Some(RelicKind::LizardTail),
            Self::LunarPastry => Some(RelicKind::LunarPastry),
            Self::Mango => Some(RelicKind::Mango),
            Self::MeatOnTheBone => Some(RelicKind::MeatOnTheBone),
            Self::Metronome => Some(RelicKind::Metronome),
            Self::MiniRegent => Some(RelicKind::MiniRegent),
            Self::MoltenEgg => Some(RelicKind::MoltenEgg),
            Self::MummifiedHand => Some(RelicKind::MummifiedHand),
            Self::OldCoin => Some(RelicKind::OldCoin),
            Self::OrangeDough => Some(RelicKind::OrangeDough),
            Self::Pocketwatch => Some(RelicKind::Pocketwatch),
            Self::PowerCell => Some(RelicKind::PowerCell),
            Self::PrayerWheel => Some(RelicKind::PrayerWheel),
            Self::RainbowRing => Some(RelicKind::RainbowRing),
            Self::RazorTooth => Some(RelicKind::RazorTooth),
            Self::Shovel => Some(RelicKind::Shovel),
            Self::Shuriken => Some(RelicKind::Shuriken),
            Self::StoneCalendar => Some(RelicKind::StoneCalendar),
            Self::SturdyClamp => Some(RelicKind::SturdyClamp),
            Self::TheCourier => Some(RelicKind::TheCourier),
            Self::ToughBandages => Some(RelicKind::ToughBandages),
            Self::ToxicEgg => Some(RelicKind::ToxicEgg),
            Self::TungstenRod => Some(RelicKind::TungstenRod),
            Self::UnceasingTop => Some(RelicKind::UnceasingTop),
            Self::UnsettlingLamp => Some(RelicKind::UnsettlingLamp),
            Self::RuinedHelmet => Some(RelicKind::RuinedHelmet),
            Self::VexingPuzzlebox => Some(RelicKind::VexingPuzzlebox),
            Self::WhiteBeastStatue => Some(RelicKind::WhiteBeastStatue),
            Self::WhiteStar => Some(RelicKind::WhiteStar),
            Self::PaperKrane => Some(RelicKind::PaperKrane),
            Self::BeltBuckle => Some(RelicKind::BeltBuckle),
            Self::Bread => Some(RelicKind::Bread),
            Self::Brimstone => Some(RelicKind::Brimstone),
            Self::BurningSticks => Some(RelicKind::BurningSticks),
            Self::Cauldron => Some(RelicKind::Cauldron),
            Self::ChemicalX => Some(RelicKind::ChemicalX),
            Self::DingyRug => Some(RelicKind::DingyRug),
            Self::DollysMirror => Some(RelicKind::DollysMirror),
            Self::DragonFruit => Some(RelicKind::DragonFruit),
            Self::GhostSeed => Some(RelicKind::GhostSeed),
            Self::GnarledHammer => Some(RelicKind::GnarledHammer),
            Self::Kifuda => Some(RelicKind::Kifuda),
            Self::LavaLamp => Some(RelicKind::LavaLamp),
            Self::LeesWaffle => Some(RelicKind::LeesWaffle),
            Self::MembershipCard => Some(RelicKind::MembershipCard),
            Self::MiniatureTent => Some(RelicKind::MiniatureTent),
            Self::MysticLighter => Some(RelicKind::MysticLighter),
            Self::NinjaScroll => Some(RelicKind::NinjaScroll),
            Self::Orrery => Some(RelicKind::Orrery),
            Self::PunchDagger => Some(RelicKind::PunchDagger),
            Self::RingingTriangle => Some(RelicKind::RingingTriangle),
            Self::RoyalStamp => Some(RelicKind::RoyalStamp),
            Self::RunicCapacitor => Some(RelicKind::RunicCapacitor),
            Self::ScreamingFlagon => Some(RelicKind::ScreamingFlagon),
            Self::SlingOfCourage => Some(RelicKind::SlingOfCourage),
            Self::TheAbacus => Some(RelicKind::TheAbacus),
            Self::Toolbox => Some(RelicKind::Toolbox),
            Self::UndyingSigil => Some(RelicKind::UndyingSigil),
            Self::VitruvianMinion => Some(RelicKind::VitruvianMinion),
            Self::WingCharm => Some(RelicKind::WingCharm),
            Self::AlchemicalCoffer => Some(RelicKind::AlchemicalCoffer),
            Self::ArcaneScroll => Some(RelicKind::ArcaneScroll),
            Self::ArchaicTooth => Some(RelicKind::ArchaicTooth),
            Self::Astrolabe => Some(RelicKind::Astrolabe),
            Self::BeautifulBracelet => Some(RelicKind::BeautifulBracelet),
            Self::BigMushroom => Some(RelicKind::BigMushroom),
            Self::BiiigHug => Some(RelicKind::BiiigHug),
            Self::BingBong => Some(RelicKind::BingBong),
            Self::BlackStar => Some(RelicKind::BlackStar),
            Self::BlessedAntler => Some(RelicKind::BlessedAntler),
            Self::BloodSoakedRose => Some(RelicKind::BloodSoakedRose),
            Self::BoneTea => Some(RelicKind::BoneTea),
            Self::BoomingConch => Some(RelicKind::BoomingConch),
            Self::BrilliantScarf => Some(RelicKind::BrilliantScarf),
            Self::Byrdpip => Some(RelicKind::Byrdpip),
            Self::CallingBell => Some(RelicKind::CallingBell),
            Self::ChoicesParadox => Some(RelicKind::ChoicesParadox),
            Self::ChosenCheese => Some(RelicKind::ChosenCheese),
            Self::Claws => Some(RelicKind::Claws),
            Self::Crossbow => Some(RelicKind::Crossbow),
            Self::CursedPearl => Some(RelicKind::CursedPearl),
            Self::DarkstonePeriapt => Some(RelicKind::DarkstonePeriapt),
            Self::DaughterOfTheWind => Some(RelicKind::DaughterOfTheWind),
            Self::DelicateFrond => Some(RelicKind::DelicateFrond),
            Self::DiamondDiadem => Some(RelicKind::DiamondDiadem),
            Self::DistinguishedCape => Some(RelicKind::DistinguishedCape),
            Self::DreamCatcher => Some(RelicKind::DreamCatcher),
            Self::Driftwood => Some(RelicKind::Driftwood),
            Self::DustyTome => Some(RelicKind::DustyTome),
            Self::Ectoplasm => Some(RelicKind::Ectoplasm),
            Self::ElectricShrymp => Some(RelicKind::ElectricShrymp),
            Self::EmberTea => Some(RelicKind::EmberTea),
            Self::EmptyCage => Some(RelicKind::EmptyCage),
            Self::FakeAnchor => Some(RelicKind::FakeAnchor),
            Self::FakeBloodVial => Some(RelicKind::FakeBloodVial),
            Self::FakeHappyFlower => Some(RelicKind::FakeHappyFlower),
            Self::FakeLeesWaffle => Some(RelicKind::FakeLeesWaffle),
            Self::FakeMango => Some(RelicKind::FakeMango),
            Self::FakeMerchantsRug => Some(RelicKind::FakeMerchantsRug),
            Self::FakeOrichalcum => Some(RelicKind::FakeOrichalcum),
            Self::FakeSneckoEye => Some(RelicKind::FakeSneckoEye),
            Self::FakeStrikeDummy => Some(RelicKind::FakeStrikeDummy),
            Self::FakeVenerableTeaSet => Some(RelicKind::FakeVenerableTeaSet),
            Self::Fiddle => Some(RelicKind::Fiddle),
            Self::ForgottenSoul => Some(RelicKind::ForgottenSoul),
            Self::FragrantMushroom => Some(RelicKind::FragrantMushroom),
            Self::FresnelLens => Some(RelicKind::FresnelLens),
            Self::FurCoat => Some(RelicKind::FurCoat),
            Self::GlassEye => Some(RelicKind::GlassEye),
            Self::Glitter => Some(RelicKind::Glitter),
            Self::GoldenCompass => Some(RelicKind::GoldenCompass),
            Self::GoldenPearl => Some(RelicKind::GoldenPearl),
            Self::HandDrill => Some(RelicKind::HandDrill),
            Self::HistoryCourse => Some(RelicKind::HistoryCourse),
            Self::IronClub => Some(RelicKind::IronClub),
            Self::JeweledMask => Some(RelicKind::JeweledMask),
            Self::JewelryBox => Some(RelicKind::JewelryBox),
            Self::LargeCapsule => Some(RelicKind::LargeCapsule),
            Self::LavaRock => Some(RelicKind::LavaRock),
            Self::LeadPaperweight => Some(RelicKind::LeadPaperweight),
            Self::LeafyPoultice => Some(RelicKind::LeafyPoultice),
            Self::LoomingFruit => Some(RelicKind::LoomingFruit),
            Self::LordsParasol => Some(RelicKind::LordsParasol),
            Self::LostCoffer => Some(RelicKind::LostCoffer),
            Self::LostWisp => Some(RelicKind::LostWisp),
            Self::MassiveScroll => Some(RelicKind::MassiveScroll),
            Self::MawBank => Some(RelicKind::MawBank),
            Self::MeatCleaver => Some(RelicKind::MeatCleaver),
            Self::MrStruggles => Some(RelicKind::MrStruggles),
            Self::MusicBox => Some(RelicKind::MusicBox),
            Self::NeowsTorment => Some(RelicKind::NeowsTorment),
            Self::NewLeaf => Some(RelicKind::NewLeaf),
            Self::NutritiousOyster => Some(RelicKind::NutritiousOyster),
            Self::NutritiousSoup => Some(RelicKind::NutritiousSoup),
            Self::PaelsBlood => Some(RelicKind::PaelsBlood),
            Self::PaelsClaw => Some(RelicKind::PaelsClaw),
            Self::PaelsEye => Some(RelicKind::PaelsEye),
            Self::PaelsFlesh => Some(RelicKind::PaelsFlesh),
            Self::PaelsGrowth => Some(RelicKind::PaelsGrowth),
            Self::PaelsHorn => Some(RelicKind::PaelsHorn),
            Self::PaelsLegion => Some(RelicKind::PaelsLegion),
            Self::PaelsTears => Some(RelicKind::PaelsTears),
            Self::PaelsTooth => Some(RelicKind::PaelsTooth),
            Self::PaelsWing => Some(RelicKind::PaelsWing),
            Self::PandorasBox => Some(RelicKind::PandorasBox),
            Self::PhilosophersStone => Some(RelicKind::PhilosophersStone),
            Self::PollinousCore => Some(RelicKind::PollinousCore),
            Self::Pomander => Some(RelicKind::Pomander),
            Self::PrecariousShears => Some(RelicKind::PrecariousShears),
            Self::PreciseScissors => Some(RelicKind::PreciseScissors),
            Self::PreservedFog => Some(RelicKind::PreservedFog),
            Self::PrismaticGem => Some(RelicKind::PrismaticGem),
            Self::PumpkinCandle => Some(RelicKind::PumpkinCandle),
            Self::RadiantPearl => Some(RelicKind::RadiantPearl),
            Self::RoyalPoison => Some(RelicKind::RoyalPoison),
            Self::RunicPyramid => Some(RelicKind::RunicPyramid),
            Self::Sai => Some(RelicKind::Sai),
            Self::SandCastle => Some(RelicKind::SandCastle),
            Self::ScrollBoxes => Some(RelicKind::ScrollBoxes),
            Self::SeaGlass => Some(RelicKind::SeaGlass),
            Self::SealOfGold => Some(RelicKind::SealOfGold),
            Self::SereTalon => Some(RelicKind::SereTalon),
            Self::SignetRing => Some(RelicKind::SignetRing),
            Self::SilverCrucible => Some(RelicKind::SilverCrucible),
            Self::SmallCapsule => Some(RelicKind::SmallCapsule),
            Self::SneckoEye => Some(RelicKind::SneckoEye),
            Self::Sozu => Some(RelicKind::Sozu),
            Self::SpikedGauntlets => Some(RelicKind::SpikedGauntlets),
            Self::StoneHumidifier => Some(RelicKind::StoneHumidifier),
            Self::Storybook => Some(RelicKind::Storybook),
            Self::SwordOfStone => Some(RelicKind::SwordOfStone),
            Self::SwordOfJade => Some(RelicKind::SwordOfJade),
            Self::TanxsWhistle => Some(RelicKind::TanxsWhistle),
            Self::TeaOfDiscourtesy => Some(RelicKind::TeaOfDiscourtesy),
            Self::TheBoot => Some(RelicKind::TheBoot),
            Self::ThrowingAxe => Some(RelicKind::ThrowingAxe),
            Self::ToastyMittens => Some(RelicKind::ToastyMittens),
            Self::TouchOfOrobas => Some(RelicKind::TouchOfOrobas),
            Self::ToyBox => Some(RelicKind::ToyBox),
            Self::TriBoomerang => Some(RelicKind::TriBoomerang),
            Self::VelvetChoker => Some(RelicKind::VelvetChoker),
            Self::VeryHotCocoa => Some(RelicKind::VeryHotCocoa),
            Self::WarHammer => Some(RelicKind::WarHammer),
            Self::WhisperingEarring => Some(RelicKind::WhisperingEarring),
            Self::WongoCustomerAppreciationBadge => Some(RelicKind::WongoCustomerAppreciationBadge),
            Self::WongosMysteryTicket => Some(RelicKind::WongosMysteryTicket),
            Self::YummyCookie => Some(RelicKind::YummyCookie),
            Self::SymbioticVirus => Some(RelicKind::SymbioticVirus),
            Self::TwistedFunnel => Some(RelicKind::TwistedFunnel),
            Self::CloakClasp => Some(RelicKind::CloakClasp),
            Self::Circlet => Some(RelicKind::Circlet),
            Self::DeprecatedRelic => Some(RelicKind::DeprecatedRelic),
            // Combat-inert markers (2026-07-17 soft-drop census): paired
            // with no-op `RelicKind`s so the py registry mirror can list
            // them and the pycli adapter stops soft-dropping them from
            // reconstructed real-game states. Zero combat hooks in the
            // v0.107.1 C# decomp; run-side pickup behavior stays deferred.
            Self::NeowsTalisman => Some(RelicKind::NeowsTalisman),
            Self::WingedBoots => Some(RelicKind::WingedBoots),
            // Wave 3 (2026-07-17): the last three W9 deferrals gain
            // marker kinds — see the enum-tail doc block.
            Self::PhialHolster => Some(RelicKind::PhialHolster),
            Self::NeowsBones => Some(RelicKind::NeowsBones),
            Self::HeftyTablet => Some(RelicKind::HeftyTablet),
            Self::FishingRod => Some(RelicKind::FishingRod),
            Self::Kaleidoscope => Some(RelicKind::Kaleidoscope),
            Self::SilkenTress => Some(RelicKind::SilkenTress),
            // v0.111.0-only (GH #792): pure on-pickup relics with no combat
            // hook (`DowsingRod.cs`, `NeowsSacrifice.cs` override only
            // `AfterObtained`), so they need no combat-side kind. Kept off
            // `RelicKind` so its universe (combat encoders, digests) stays
            // the v0.107.1 one.
            Self::DowsingRod | Self::NeowsSacrifice => None,
            // v0.113.0-only event relics. Thumbscrew (a new
            // `ModifyAttackHitCount` hook) and Wuppy (a pet that takes the
            // owner's powered attacks) have no combat side yet; Byrdine's
            // pet summon is `RelicKind::Byrdine`; the rest have listeners in
            // `crate::relic_effects::v0113_event`.
            Self::Thumbscrew | Self::Wuppy => None,
            Self::Byrdine => Some(RelicKind::Byrdine),
            Self::ObsidianVice => Some(RelicKind::ObsidianVice),
            Self::TormentousCollar => Some(RelicKind::TormentousCollar),
            Self::MightyWings => Some(RelicKind::MightyWings),
            Self::BathTowel => Some(RelicKind::BathTowel),
            Self::BathRobes => Some(RelicKind::BathRobes),
            // v0.113.0 Fool relics: combat listeners in `crate::fool`.
            Self::SongBow => Some(RelicKind::SongBow),
            Self::Storyteller => Some(RelicKind::Storyteller),
            Self::BizarreLamp => Some(RelicKind::BizarreLamp),
            Self::ConductorsBaton => Some(RelicKind::ConductorsBaton),
            Self::ExerciseMat => Some(RelicKind::ExerciseMat),
            Self::Fertilizer => Some(RelicKind::Fertilizer),
            Self::Ocarina => Some(RelicKind::Ocarina),
            Self::Twine => Some(RelicKind::Twine),
            Self::WhittlingKnife => Some(RelicKind::WhittlingKnife),
        }
    }
}

impl RelicKind {
    /// The combat-side `Relic.counter` value a freshly-obtained instance
    /// of this relic starts with — the C# relic's `_combatsLeft` /
    /// equivalent `[SavedProperty]` initializer. Most relics either don't
    /// use the counter or use it as a zero-based accumulator (Wongo's
    /// `_combats_finished`, LastingCandy's `_combats_seen`), so the
    /// default is `0`. The exceptions are the **countdown** relics whose
    /// per-combat listeners gate on `counter > 0` and decrement toward
    /// disable — seeding them at `0` silently disables the relic on its
    /// very first combat.
    ///
    /// Callers that build a one-off combat with no run context (the parity
    /// scenario builder, debug CLI) must seed `Relic.counter` from this so
    /// these relics fire exactly as C# does for a fresh combat. Bead
    /// `sts2-rs-8s5wf`.
    #[must_use]
    pub const fn initial_counter(self) -> i32 {
        match self {
            // C# `EmberTea._combatsLeft = 5` (Strength(2) on combat-room
            // entered, 5 combats).
            RelicKind::EmberTea => 5,
            // C# `BoneTea._combatsLeft = 1` (upgrade hand on round-1
            // side-turn-start, 1 combat).
            RelicKind::BoneTea => 1,
            // C# `TeaOfDiscourtesy._combatsLeft = 1` (2 Dazed → draw pile
            // on combat start, 1 combat).
            RelicKind::TeaOfDiscourtesy => 1,
            // C# `SwordOfStone._combatsLeft = 5` (transform after 5 elite
            // victories).
            RelicKind::SwordOfStone => 5,
            _ => 0,
        }
    }

    /// Python `RelicId.<self>.name` — the string the (Phase 7) parity
    /// exporter writes into the `relics[*].id` field of a
    /// `BridgeReplayTrace`. Names are upper-snake-case to match
    /// Python's `enum.auto()` member naming convention
    /// (e.g. `RelicId.BURNING_BLOOD.name == "BURNING_BLOOD"`).
    #[must_use]
    pub const fn python_name(self) -> &'static str {
        match self {
            RelicKind::BurningBlood => "BURNING_BLOOD",
            RelicKind::CrackedCore => "CRACKED_CORE",
            RelicKind::RingOfTheSnake => "RING_OF_THE_SNAKE",
            RelicKind::DivineRight => "DIVINE_RIGHT",
            RelicKind::BoundPhylactery => "BOUND_PHYLACTERY",
            RelicKind::AmethystAubergine => "AMETHYST_AUBERGINE",
            RelicKind::Anchor => "ANCHOR",
            RelicKind::BagOfPreparation => "BAG_OF_PREPARATION",
            RelicKind::BloodVial => "BLOOD_VIAL",
            RelicKind::BoneFlute => "BONE_FLUTE",
            RelicKind::BronzeScales => "BRONZE_SCALES",
            RelicKind::CentennialPuzzle => "CENTENNIAL_PUZZLE",
            RelicKind::DataDisk => "DATA_DISK",
            RelicKind::FencingManual => "FENCING_MANUAL",
            RelicKind::FestivePopper => "FESTIVE_POPPER",
            RelicKind::Gorget => "GORGET",
            RelicKind::HappyFlower => "HAPPY_FLOWER",
            RelicKind::JuzuBracelet => "JUZU_BRACELET",
            RelicKind::Lantern => "LANTERN",
            RelicKind::MealTicket => "MEAL_TICKET",
            RelicKind::OddlySmoothStone => "ODDLY_SMOOTH_STONE",
            RelicKind::Pendulum => "PENDULUM",
            RelicKind::Permafrost => "PERMAFROST",
            RelicKind::PotionBelt => "POTION_BELT",
            RelicKind::RedSkull => "RED_SKULL",
            RelicKind::RegalPillow => "REGAL_PILLOW",
            RelicKind::SneckoSkull => "SNECKO_SKULL",
            RelicKind::Strawberry => "STRAWBERRY",
            RelicKind::StrikeDummy => "STRIKE_DUMMY",
            RelicKind::TinyMailbox => "TINY_MAILBOX",
            RelicKind::Vajra => "VAJRA",
            RelicKind::VenerableTeaSet => "VENERABLE_TEA_SET",
            RelicKind::WarPaint => "WAR_PAINT",
            RelicKind::Whetstone => "WHETSTONE",
            RelicKind::BookOfFiveRings => "BOOK_OF_FIVE_RINGS",
            RelicKind::Akabeko => "AKABEKO",
            RelicKind::BagOfMarbles => "BAG_OF_MARBLES",
            RelicKind::Bellows => "BELLOWS",
            RelicKind::BookRepairKnife => "BOOK_REPAIR_KNIFE",
            RelicKind::BowlerHat => "BOWLER_HAT",
            RelicKind::Candelabra => "CANDELABRA",
            RelicKind::EternalFeather => "ETERNAL_FEATHER",
            RelicKind::FuneraryMask => "FUNERARY_MASK",
            RelicKind::GalacticDust => "GALACTIC_DUST",
            RelicKind::GoldPlatedCables => "GOLD_PLATED_CABLES",
            RelicKind::GremlinHorn => "GREMLIN_HORN",
            RelicKind::HornCleat => "HORN_CLEAT",
            RelicKind::JossPaper => "JOSS_PAPER",
            RelicKind::Kusarigama => "KUSARIGAMA",
            RelicKind::LetterOpener => "LETTER_OPENER",
            RelicKind::LuckyFysh => "LUCKY_FYSH",
            RelicKind::MercuryHourglass => "MERCURY_HOURGLASS",
            RelicKind::MiniatureCannon => "MINIATURE_CANNON",
            RelicKind::Nunchaku => "NUNCHAKU",
            RelicKind::Orichalcum => "ORICHALCUM",
            RelicKind::OrnamentalFan => "ORNAMENTAL_FAN",
            RelicKind::Pantograph => "PANTOGRAPH",
            RelicKind::PaperPhrog => "PAPER_PHROG",
            RelicKind::ParryingShield => "PARRYING_SHIELD",
            RelicKind::Pear => "PEAR",
            RelicKind::PenNib => "PEN_NIB",
            RelicKind::PetrifiedToad => "PETRIFIED_TOAD",
            RelicKind::Planisphere => "PLANISPHERE",
            RelicKind::RedMask => "RED_MASK",
            RelicKind::Regalite => "REGALITE",
            RelicKind::ReptileTrinket => "REPTILE_TRINKET",
            RelicKind::RippleBasin => "RIPPLE_BASIN",
            RelicKind::SelfFormingClay => "SELF_FORMING_CLAY",
            RelicKind::SparklingRouge => "SPARKLING_ROUGE",
            RelicKind::StoneCracker => "STONE_CRACKER",
            RelicKind::SymbioticVirus => "SYMBIOTIC_VIRUS",
            RelicKind::Tingsha => "TINGSHA",
            RelicKind::TuningFork => "TUNING_FORK",
            RelicKind::TwistedFunnel => "TWISTED_FUNNEL",
            RelicKind::Vambrace => "VAMBRACE",
            RelicKind::BigMushroom => "BIG_MUSHROOM",
            RelicKind::BingBong => "BING_BONG",
            RelicKind::BoneTea => "BONE_TEA",
            RelicKind::Byrdpip => "BYRDPIP",
            RelicKind::ChosenCheese => "CHOSEN_CHEESE",
            RelicKind::DarkstonePeriapt => "DARKSTONE_PERIAPT",
            RelicKind::DaughterOfTheWind => "DAUGHTER_OF_THE_WIND",
            RelicKind::DreamCatcher => "DREAM_CATCHER",
            RelicKind::EmberTea => "EMBER_TEA",
            RelicKind::FakeAnchor => "FAKE_ANCHOR",
            RelicKind::FakeBloodVial => "FAKE_BLOOD_VIAL",
            RelicKind::FakeHappyFlower => "FAKE_HAPPY_FLOWER",
            RelicKind::FakeLeesWaffle => "FAKE_LEES_WAFFLE",
            RelicKind::FakeMango => "FAKE_MANGO",
            RelicKind::FakeMerchantsRug => "FAKE_MERCHANTS_RUG",
            RelicKind::FakeOrichalcum => "FAKE_ORICHALCUM",
            RelicKind::FakeSneckoEye => "FAKE_SNECKO_EYE",
            RelicKind::FakeStrikeDummy => "FAKE_STRIKE_DUMMY",
            RelicKind::FakeVenerableTeaSet => "FAKE_VENERABLE_TEA_SET",
            RelicKind::ForgottenSoul => "FORGOTTEN_SOUL",
            RelicKind::FragrantMushroom => "FRAGRANT_MUSHROOM",
            RelicKind::FresnelLens => "FRESNEL_LENS",
            RelicKind::HandDrill => "HAND_DRILL",
            RelicKind::HistoryCourse => "HISTORY_COURSE",
            RelicKind::LostWisp => "LOST_WISP",
            RelicKind::MawBank => "MAW_BANK",
            RelicKind::MrStruggles => "MR_STRUGGLES",
            RelicKind::PollinousCore => "POLLINOUS_CORE",
            RelicKind::RoyalPoison => "ROYAL_POISON",
            RelicKind::SwordOfStone => "SWORD_OF_STONE",
            RelicKind::SwordOfJade => "SWORD_OF_JADE",
            RelicKind::TeaOfDiscourtesy => "TEA_OF_DISCOURTESY",
            RelicKind::TheBoot => "THE_BOOT",
            RelicKind::WongoCustomerAppreciationBadge => "WONGO_CUSTOMER_APPRECIATION_BADGE",
            RelicKind::WongosMysteryTicket => "WONGOS_MYSTERY_TICKET",
            RelicKind::ArtOfWar => "ART_OF_WAR",
            RelicKind::BeatingRemnant => "BEATING_REMNANT",
            RelicKind::BigHat => "BIG_HAT",
            RelicKind::Bookmark => "BOOKMARK",
            RelicKind::CaptainsWheel => "CAPTAINS_WHEEL",
            RelicKind::Chandelier => "CHANDELIER",
            RelicKind::CharonsAshes => "CHARONS_ASHES",
            RelicKind::CloakClasp => "CLOAK_CLASP",
            RelicKind::DemonTongue => "DEMON_TONGUE",
            RelicKind::EmotionChip => "EMOTION_CHIP",
            RelicKind::FrozenEgg => "FROZEN_EGG",
            RelicKind::GamblingChip => "GAMBLING_CHIP",
            RelicKind::GamePiece => "GAME_PIECE",
            RelicKind::Girya => "GIRYA",
            RelicKind::HelicalDart => "HELICAL_DART",
            RelicKind::IceCream => "ICE_CREAM",
            RelicKind::IntimidatingHelmet => "INTIMIDATING_HELMET",
            RelicKind::IvoryTile => "IVORY_TILE",
            RelicKind::Kunai => "KUNAI",
            RelicKind::LastingCandy => "LASTING_CANDY",
            RelicKind::LizardTail => "LIZARD_TAIL",
            RelicKind::LunarPastry => "LUNAR_PASTRY",
            RelicKind::Mango => "MANGO",
            RelicKind::MeatOnTheBone => "MEAT_ON_THE_BONE",
            RelicKind::Metronome => "METRONOME",
            RelicKind::MiniRegent => "MINI_REGENT",
            RelicKind::MoltenEgg => "MOLTEN_EGG",
            RelicKind::MummifiedHand => "MUMMIFIED_HAND",
            RelicKind::OldCoin => "OLD_COIN",
            RelicKind::OrangeDough => "ORANGE_DOUGH",
            RelicKind::PaperKrane => "PAPER_KRANE",
            RelicKind::Pocketwatch => "POCKETWATCH",
            RelicKind::PowerCell => "POWER_CELL",
            RelicKind::PrayerWheel => "PRAYER_WHEEL",
            RelicKind::RainbowRing => "RAINBOW_RING",
            RelicKind::RazorTooth => "RAZOR_TOOTH",
            RelicKind::RuinedHelmet => "RUINED_HELMET",
            RelicKind::Shovel => "SHOVEL",
            RelicKind::Shuriken => "SHURIKEN",
            RelicKind::StoneCalendar => "STONE_CALENDAR",
            RelicKind::SturdyClamp => "STURDY_CLAMP",
            RelicKind::TheCourier => "THE_COURIER",
            RelicKind::ToughBandages => "TOUGH_BANDAGES",
            RelicKind::ToxicEgg => "TOXIC_EGG",
            RelicKind::TungstenRod => "TUNGSTEN_ROD",
            RelicKind::UnceasingTop => "UNCEASING_TOP",
            RelicKind::UnsettlingLamp => "UNSETTLING_LAMP",
            RelicKind::VexingPuzzlebox => "VEXING_PUZZLEBOX",
            RelicKind::WhiteBeastStatue => "WHITE_BEAST_STATUE",
            RelicKind::WhiteStar => "WHITE_STAR",
            RelicKind::Brimstone => "BRIMSTONE",
            RelicKind::BoomingConch => "BOOMING_CONCH",
            RelicKind::TheAbacus => "THE_ABACUS",
            RelicKind::SlingOfCourage => "SLING_OF_COURAGE",
            RelicKind::BlackBlood => "BLACK_BLOOD",
            RelicKind::RingOfTheDrake => "RING_OF_THE_DRAKE",
            RelicKind::InfusedCore => "INFUSED_CORE",
            RelicKind::PhylacteryUnbound => "PHYLACTERY_UNBOUND",
            RelicKind::DivineDestiny => "DIVINE_DESTINY",
            // ── sts2log gap-report markers (bead `sts2-rs-pgvl`) ─────
            RelicKind::ScrollBoxes => "SCROLL_BOXES",
            RelicKind::LeadPaperweight => "LEAD_PAPERWEIGHT",
            RelicKind::ArcaneScroll => "ARCANE_SCROLL",
            RelicKind::NewLeaf => "NEW_LEAF",
            RelicKind::PrecariousShears => "PRECARIOUS_SHEARS",
            RelicKind::LeafyPoultice => "LEAFY_POULTICE",
            RelicKind::CursedPearl => "CURSED_PEARL",
            // ── s5vb validation blockers (bead `sts2-rs-pibg`) ─────
            RelicKind::LostCoffer => "LOST_COFFER",
            RelicKind::Pomander => "POMANDER",
            RelicKind::SilverCrucible => "SILVER_CRUCIBLE",
            RelicKind::FishingRod => "FISHING_ROD",
            RelicKind::Kaleidoscope => "KALEIDOSCOPE",
            RelicKind::SilkenTress => "SILKEN_TRESS",
            // ── fwxe.5 runtime-RelicKind blockers (bead `sts2-rs-8uvz`) ─
            RelicKind::SmallCapsule => "SMALL_CAPSULE",
            RelicKind::ArchaicTooth => "ARCHAIC_TOOTH",
            RelicKind::PaelsEye => "PAELS_EYE",
            // Bead `sts2-rs-97t6` (fwxe.10).
            RelicKind::Astrolabe => "ASTROLABE",
            // ── fwxe.8 runtime-RelicKind blockers (bead `sts2-rs-qffh`) ─
            RelicKind::PaelsGrowth => "PAELS_GROWTH",
            RelicKind::NutritiousSoup => "NUTRITIOUS_SOUP",
            // Bead `sts2-rs-wc7h` (fwxe.12).
            RelicKind::VeryHotCocoa => "VERY_HOT_COCOA",
            // Bead `sts2-rs-i9fl` (fwxe.14).
            RelicKind::DelicateFrond => "DELICATE_FROND",
            RelicKind::TriBoomerang => "TRI_BOOMERANG",
            // ── cia9 batch-extract (bead `sts2-rs-cia9`) ─────────────
            RelicKind::AlchemicalCoffer => "ALCHEMICAL_COFFER",
            RelicKind::BeautifulBracelet => "BEAUTIFUL_BRACELET",
            RelicKind::BeltBuckle => "BELT_BUCKLE",
            RelicKind::BiiigHug => "BIIIG_HUG",
            RelicKind::BlackStar => "BLACK_STAR",
            RelicKind::BlessedAntler => "BLESSED_ANTLER",
            RelicKind::BloodSoakedRose => "BLOOD_SOAKED_ROSE",
            RelicKind::Bread => "BREAD",
            RelicKind::BrilliantScarf => "BRILLIANT_SCARF",
            RelicKind::BurningSticks => "BURNING_STICKS",
            RelicKind::CallingBell => "CALLING_BELL",
            RelicKind::Cauldron => "CAULDRON",
            RelicKind::ChemicalX => "CHEMICAL_X",
            RelicKind::ChoicesParadox => "CHOICES_PARADOX",
            RelicKind::Circlet => "CIRCLET",
            RelicKind::Claws => "CLAWS",
            RelicKind::Crossbow => "CROSSBOW",
            RelicKind::DeprecatedRelic => "DEPRECATED_RELIC",
            RelicKind::DiamondDiadem => "DIAMOND_DIADEM",
            RelicKind::DingyRug => "DINGY_RUG",
            RelicKind::DistinguishedCape => "DISTINGUISHED_CAPE",
            RelicKind::DollysMirror => "DOLLYS_MIRROR",
            RelicKind::DragonFruit => "DRAGON_FRUIT",
            RelicKind::Driftwood => "DRIFTWOOD",
            RelicKind::DustyTome => "DUSTY_TOME",
            RelicKind::Ectoplasm => "ECTOPLASM",
            RelicKind::ElectricShrymp => "ELECTRIC_SHRYMP",
            RelicKind::EmptyCage => "EMPTY_CAGE",
            RelicKind::Fiddle => "FIDDLE",
            RelicKind::FurCoat => "FUR_COAT",
            RelicKind::GhostSeed => "GHOST_SEED",
            RelicKind::GlassEye => "GLASS_EYE",
            RelicKind::Glitter => "GLITTER",
            RelicKind::GnarledHammer => "GNARLED_HAMMER",
            RelicKind::GoldenCompass => "GOLDEN_COMPASS",
            RelicKind::GoldenPearl => "GOLDEN_PEARL",
            RelicKind::IronClub => "IRON_CLUB",
            RelicKind::JeweledMask => "JEWELED_MASK",
            RelicKind::JewelryBox => "JEWELRY_BOX",
            RelicKind::Kifuda => "KIFUDA",
            RelicKind::LargeCapsule => "LARGE_CAPSULE",
            RelicKind::LavaLamp => "LAVA_LAMP",
            RelicKind::LavaRock => "LAVA_ROCK",
            RelicKind::LeesWaffle => "LEES_WAFFLE",
            RelicKind::LoomingFruit => "LOOMING_FRUIT",
            RelicKind::LordsParasol => "LORDS_PARASOL",
            RelicKind::MassiveScroll => "MASSIVE_SCROLL",
            RelicKind::MeatCleaver => "MEAT_CLEAVER",
            RelicKind::MembershipCard => "MEMBERSHIP_CARD",
            RelicKind::MiniatureTent => "MINIATURE_TENT",
            RelicKind::MusicBox => "MUSIC_BOX",
            RelicKind::MysticLighter => "MYSTIC_LIGHTER",
            RelicKind::NeowsTorment => "NEOWS_TORMENT",
            RelicKind::NinjaScroll => "NINJA_SCROLL",
            RelicKind::NutritiousOyster => "NUTRITIOUS_OYSTER",
            RelicKind::Orrery => "ORRERY",
            RelicKind::PaelsBlood => "PAELS_BLOOD",
            RelicKind::PaelsClaw => "PAELS_CLAW",
            RelicKind::PaelsFlesh => "PAELS_FLESH",
            RelicKind::PaelsHorn => "PAELS_HORN",
            RelicKind::PaelsLegion => "PAELS_LEGION",
            RelicKind::PaelsTears => "PAELS_TEARS",
            RelicKind::PaelsTooth => "PAELS_TOOTH",
            RelicKind::PaelsWing => "PAELS_WING",
            RelicKind::PandorasBox => "PANDORAS_BOX",
            RelicKind::PhilosophersStone => "PHILOSOPHERS_STONE",
            RelicKind::PreciseScissors => "PRECISE_SCISSORS",
            RelicKind::PreservedFog => "PRESERVED_FOG",
            RelicKind::PrismaticGem => "PRISMATIC_GEM",
            RelicKind::PumpkinCandle => "PUMPKIN_CANDLE",
            RelicKind::PunchDagger => "PUNCH_DAGGER",
            RelicKind::RadiantPearl => "RADIANT_PEARL",
            RelicKind::RingingTriangle => "RINGING_TRIANGLE",
            RelicKind::RoyalStamp => "ROYAL_STAMP",
            RelicKind::RunicCapacitor => "RUNIC_CAPACITOR",
            RelicKind::RunicPyramid => "RUNIC_PYRAMID",
            RelicKind::Sai => "SAI",
            RelicKind::SandCastle => "SAND_CASTLE",
            RelicKind::ScreamingFlagon => "SCREAMING_FLAGON",
            RelicKind::SeaGlass => "SEA_GLASS",
            RelicKind::SealOfGold => "SEAL_OF_GOLD",
            RelicKind::SereTalon => "SERE_TALON",
            RelicKind::SignetRing => "SIGNET_RING",
            RelicKind::SneckoEye => "SNECKO_EYE",
            RelicKind::Sozu => "SOZU",
            RelicKind::SpikedGauntlets => "SPIKED_GAUNTLETS",
            RelicKind::StoneHumidifier => "STONE_HUMIDIFIER",
            RelicKind::Storybook => "STORYBOOK",
            RelicKind::TanxsWhistle => "TANXS_WHISTLE",
            RelicKind::ThrowingAxe => "THROWING_AXE",
            RelicKind::ToastyMittens => "TOASTY_MITTENS",
            RelicKind::Toolbox => "TOOLBOX",
            RelicKind::TouchOfOrobas => "TOUCH_OF_OROBAS",
            RelicKind::ToyBox => "TOY_BOX",
            RelicKind::UndyingSigil => "UNDYING_SIGIL",
            RelicKind::VelvetChoker => "VELVET_CHOKER",
            RelicKind::VitruvianMinion => "VITRUVIAN_MINION",
            RelicKind::WarHammer => "WAR_HAMMER",
            RelicKind::WhisperingEarring => "WHISPERING_EARRING",
            RelicKind::WingCharm => "WING_CHARM",
            RelicKind::YummyCookie => "YUMMY_COOKIE",
            // ── combat-inert Neow reward markers (2026-07-17) ────────
            RelicKind::NeowsTalisman => "NEOWS_TALISMAN",
            RelicKind::WingedBoots => "WINGED_BOOTS",
            RelicKind::PhialHolster => "PHIAL_HOLSTER",
            RelicKind::NeowsBones => "NEOWS_BONES",
            RelicKind::HeftyTablet => "HEFTY_TABLET",
            RelicKind::SongBow => "SONG_BOW",
            RelicKind::Storyteller => "STORYTELLER",
            RelicKind::BizarreLamp => "BIZARRE_LAMP",
            RelicKind::ConductorsBaton => "CONDUCTORS_BATON",
            RelicKind::ExerciseMat => "EXERCISE_MAT",
            RelicKind::Fertilizer => "FERTILIZER",
            RelicKind::Ocarina => "OCARINA",
            RelicKind::Twine => "TWINE",
            RelicKind::WhittlingKnife => "WHITTLING_KNIFE",
            RelicKind::Byrdine => "BYRDINE",
            RelicKind::ObsidianVice => "OBSIDIAN_VICE",
            RelicKind::TormentousCollar => "TORMENTOUS_COLLAR",
            RelicKind::MightyWings => "MIGHTY_WINGS",
            RelicKind::BathTowel => "BATH_TOWEL",
            RelicKind::BathRobes => "BATH_ROBES",
        }
    }

    /// The v0.113.0-only kinds (The Fool's relics and the v0.113.0 event
    /// relics with a combat hook). Callers that must keep
    /// the v0.111.0 universe (the `"relic"` embedding registry) skip these.
    #[must_use]
    pub const fn is_v0113_only(self) -> bool {
        matches!(
            self,
            RelicKind::SongBow
                | RelicKind::Storyteller
                | RelicKind::BizarreLamp
                | RelicKind::ConductorsBaton
                | RelicKind::ExerciseMat
                | RelicKind::Fertilizer
                | RelicKind::Ocarina
                | RelicKind::Twine
                | RelicKind::WhittlingKnife
                | RelicKind::Byrdine
                | RelicKind::ObsidianVice
                | RelicKind::TormentousCollar
                | RelicKind::MightyWings
                | RelicKind::BathTowel
                | RelicKind::BathRobes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the upper-snake-case Python name emitted (eventually) by
    /// the parity exporter for every variant shipped through bead
    /// sts2-rs-u4u. The string matches `RelicId.<v>.name` at pin
    /// `01c094d` (`sts2_env/relics/base.py:30`).
    #[test]
    fn python_name_matches_python_relic_id() {
        assert_eq!(RelicKind::BurningBlood.python_name(), "BURNING_BLOOD");
    }

    /// `sts2-rs-pfiwm`: the `RelicId → RelicKind` bridge is a pure
    /// name projection. For every `RelicId` that maps to a `RelicKind`,
    /// the two enums must agree on the canonical Python name — this
    /// catches any future variant whose names drift apart (so the
    /// mapping is faithful, not merely present).
    #[test]
    fn to_relic_kind_round_trips_via_python_name() {
        for &id in RelicId::all() {
            if let Some(kind) = id.to_relic_kind() {
                assert_eq!(
                    kind.python_name(),
                    id.to_python_name(),
                    "bridge name mismatch for {id:?}"
                );
            }
        }
    }

    /// `sts2-rs-pfiwm` → 2026-07-17 wave 3: the `RelicId → RelicKind`
    /// projection is now **total**. The last three W9 Neow deferrals
    /// (`PhialHolster` / `NeowsBones` / `HeftyTablet`) gained
    /// combat-inert marker kinds, so a `None` from `to_relic_kind` can
    /// only mean a freshly-added `RelicId` whose author deliberately
    /// chose not to pair it — which must show up here as a deliberate
    /// edit, never a silent regression.
    #[test]
    fn to_relic_kind_is_total() {
        let none_ids: Vec<RelicId> = RelicId::all()
            .iter()
            .copied()
            .filter(|id| id.to_relic_kind().is_none())
            .collect();
        assert_eq!(
            none_ids,
            Vec::<RelicId>::new(),
            "every RelicId must bridge to a RelicKind (add a marker kind, \
             or deliberately amend this pin)"
        );
    }

    /// 2026-07-17 wave 3: the three remaining W9 Neow reward relics are
    /// combat-inert markers — they bridge to a `RelicKind` (so the py
    /// registry mirror lists them and pycli stops soft-dropping them)
    /// and round-trip the canonical Python name. Their run-side pickup
    /// listeners live in `sts2-content::relics::neow_relics`.
    #[test]
    fn w9_neow_relics_bridge_to_marker_kinds() {
        let cases = [
            (
                RelicId::PhialHolster,
                RelicKind::PhialHolster,
                "PHIAL_HOLSTER",
            ),
            (RelicId::NeowsBones, RelicKind::NeowsBones, "NEOWS_BONES"),
            (RelicId::HeftyTablet, RelicKind::HeftyTablet, "HEFTY_TABLET"),
        ];
        for (id, kind, name) in cases {
            assert_eq!(id.to_relic_kind(), Some(kind));
            assert_eq!(kind.python_name(), name);
        }
    }

    /// Structural guard (2026-07-17 wave 3): [`RelicKind::ALL`] is
    /// macro-derived from the enum declaration, so it is exhaustive by
    /// construction. This test locks the two invariants downstream
    /// registry mirrors rely on:
    ///
    /// 1. every entry has a unique `python_name` (no alias collisions
    ///    inside `list_registry_names("relic")`), and
    /// 2. every `RelicKind` is reachable from some `RelicId` via
    ///    [`RelicId::to_relic_kind`] — a new `RelicKind` variant whose
    ///    author forgot the `to_relic_kind` arm fails HERE at test
    ///    time instead of silently never materializing in run-bridged
    ///    combat.
    #[test]
    fn relic_kind_all_is_unique_and_reachable() {
        let mut names = std::collections::BTreeSet::new();
        for &kind in RelicKind::ALL {
            assert!(
                names.insert(kind.python_name()),
                "duplicate python_name {:?} in RelicKind::ALL",
                kind.python_name()
            );
        }
        assert_eq!(names.len(), RelicKind::ALL.len());
        // Every build's universe: the v0.113.0 Fool relics are kinds too.
        let reachable: std::collections::BTreeSet<&'static str> =
            RelicId::all_for(crate::GameVersion::V0_113_0)
                .iter()
                .filter_map(|id| id.to_relic_kind())
                .map(RelicKind::python_name)
                .collect();
        for &kind in RelicKind::ALL {
            assert!(
                reachable.contains(kind.python_name()),
                "RelicKind::{kind:?} is unreachable from every RelicId — \
                 add its RelicId::to_relic_kind arm"
            );
        }
    }

    /// 2026-07-17 soft-drop census fix: `NeowsTalisman` / `WingedBoots`
    /// are combat-inert markers — they bridge to a `RelicKind` (so the
    /// registry mirror can list them) and round-trip the canonical
    /// Python name.
    #[test]
    fn neows_talisman_and_winged_boots_bridge_to_marker_kinds() {
        assert_eq!(
            RelicId::NeowsTalisman.to_relic_kind(),
            Some(RelicKind::NeowsTalisman)
        );
        assert_eq!(
            RelicId::WingedBoots.to_relic_kind(),
            Some(RelicKind::WingedBoots)
        );
        assert_eq!(RelicKind::NeowsTalisman.python_name(), "NEOWS_TALISMAN");
        assert_eq!(RelicKind::WingedBoots.python_name(), "WINGED_BOOTS");
    }

    /// `sts2-rs-8s5wf`: the four countdown relics seed a positive combat
    /// counter from `_combatsLeft`; everything else defaults to 0.
    #[test]
    fn initial_counter_seeds_countdown_relics() {
        assert_eq!(RelicKind::EmberTea.initial_counter(), 5);
        assert_eq!(RelicKind::BoneTea.initial_counter(), 1);
        assert_eq!(RelicKind::TeaOfDiscourtesy.initial_counter(), 1);
        assert_eq!(RelicKind::SwordOfStone.initial_counter(), 5);
        // Accumulator / counterless relics start at 0.
        assert_eq!(RelicKind::BurningBlood.initial_counter(), 0);
        assert_eq!(RelicKind::WongosMysteryTicket.initial_counter(), 0);
    }

    /// Every variant produces a non-empty, upper-snake-case Python
    /// name. Catches future variants that forget to add an arm to
    /// `python_name` (which is also a compile error — the match has no
    /// wildcard) and, since `RelicKind::ALL` is macro-derived, actually
    /// exercises every variant instead of a sample.
    #[test]
    fn every_variant_has_python_name() {
        for &k in RelicKind::ALL {
            let name = k.python_name();
            assert!(!name.is_empty(), "empty Python name for {k:?}");
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()),
                "non upper-snake-case Python name for {k:?}: {name}"
            );
        }
    }

    /// `RelicKind` round-trips through equality / hashing. Load-bearing
    /// because Phase 7 may key relic-effect lookups by `RelicKind`.
    #[test]
    fn relic_kind_is_copy_eq_hash() {
        fn assert_copy<T: Copy + Eq + std::hash::Hash>() {}
        assert_copy::<RelicKind>();
        assert_eq!(RelicKind::BurningBlood, RelicKind::BurningBlood);
    }
}
