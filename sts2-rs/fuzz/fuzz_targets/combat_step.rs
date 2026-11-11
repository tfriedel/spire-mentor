//! Fuzz target: drive the sts2-core combat engine through a bounded number
//! of legal `Action`s and assert a broad set of engine invariants after
//! every step.
//!
//! Bead: sts2-rs-722dx (GH #1).
//!
//! Surface (expanded 2026-05-20 per GH #1 follow-up):
//! - Random character pick (Ironclad / Silent / Defect / Necrobinder /
//!   Regent), each with its own card pool.
//! - Random 8–15-card deck sampled from that character's pool, with
//!   random per-card upgrade bits.
//! - 1–3 enemies sampled from a broad allowlist that includes
//!   respawners / spawners / egg-hatchers; spawn-args-aware monsters
//!   (Inklet, Chomper, Axebot, KinFollower, Exoskeleton, InfestedPrism,
//!   DenseVegetationWriggler) get their per-slot/per-flag args derived
//!   from the fuzz input so the slot-dependent opener arms get
//!   exercised.
//! - 0–5 relics from the full relic pool.
//! - 0–3 potions from the full potion pool (legal_actions surfaces
//!   `DrinkPotion` automatically once they're in `player.potions`).
//! - The fuzzer still picks `decisions: Vec<u16>` of indices into
//!   `legal_actions(&state)` so every byte mutation is legal.
//!
//! Invariants checked after every step (see `check_invariants` for
//! per-step checks, `check_post_step` for delta checks, and
//! `check_turn_boundary` for cross-turn checks):
//!
//! Per-step (after every `apply()`):
//!  1. Player + creature HP ∈ [0, max_hp]; player + creature block ≥ 0;
//!     player energy ≥ 0.
//!  2. `apply()` never returns `InvalidTarget` for a legal action.
//!  3. Deck partition conservation: every CardUid that exists in
//!     state.cards is in exactly one pile (draw / hand / discard /
//!     exhaust / play / pinned_top / pinned_bottom / retain_this_turn).
//!  4. Arena consistency: every uid in any pile resolves to a card in
//!     state.cards.
//!  5. Hand size ≤ 10 (STS hand cap).
//!  6. Enemy-uid ↔ creature mirror: every uid in state.enemy_uids
//!     resolves to a Creature with that id.
//!  7. Combat termination consistency: if state.is_over, then either
//!     player_won && all enemies at 0 HP, or player at 0 HP.
//!  8. Player creature mirror (per CLAUDE.md d03 unification): if
//!     player.creature_uid is Some, the mirrored Creature's HP/block/
//!     max_hp must match the legacy PlayerCombatState fields.
//!  9. No duplicate PowerKind on a single owner (player or creature).
//! 10. Non-negative bound on Vulnerable / Weak / Frail / Poison /
//!     Intangible stacks. (Strength + Dexterity legitimately go negative.)
//! 11. Unplayable status cards (Wound / Dazed / Burn / Void or
//!     anything registered via `register_card_unplayable`) must NOT
//!     appear in legal_actions as PlayCard targets.
//! 12. Intent presence: an enemy with `intent_history.turn_index > 0`
//!     (has acted at least once) must have `Some(intent)` during
//!     `EnemyTurnExec`. Exempts designed-inert spawns (eggs, sleepers)
//!     whose intent legitimately starts as None.
//!
//! Per-step delta (before/after one `apply()`):
//! 13. `is_over` is monotonic: once true, never reverts to false.
//! 14. Energy delta for PlayCard equals the card's `cost` (skipped for
//!     X-cost cards and mid-resolve states with open pending_choice).
//! 15. X-cost cards spend ALL available energy (post-resolve energy == 0).
//!
//! Cross-turn (fires on round-increment back to PlayerPlayPhase):
//! 16. Per-turn counters reset: cards_played_this_turn,
//!     shivs_played_this_turn, powered_hits_this_turn,
//!     skills_played_this_turn, stars_gained_this_turn,
//!     pending_retain_count, request_end_turn_after_play.
//! 17. Player block resets to 0 unless a retention relic (SturdyClamp)
//!     is attached.
//! 18. `play` pile drained at turn boundary.
//! 19. `active_attack` is None at turn boundary.
//! 20. `echo_form_remaining` decrements by exactly 1 per player turn
//!     when previously non-zero.
//! 21. `pending_play` does not persist across a turn boundary.
//!
//! Determinism (`fuzz_target!`): each fuzz input is executed twice;
//! the resulting state-digests must match. Catches hidden non-
//! determinism (HashMap iteration order, uninitialized fields,
//! global state).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use smallvec::SmallVec;

use std::num::NonZeroU32;
use sts2_content::cards::ironclad_basic::ironclad_starter_deck;
use sts2_content::cards::necrobinder::necrobinder_starter_deck;
use sts2_content::cards::regent::regent_starter_deck;
use sts2_content::cards::silent::silent_starter_deck;
use sts2_content::cards::spec_index::{
    COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, SILENT_SPECS,
};
use sts2_content::cards::types::CardSpec;
use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_content::monsters::act1_overgrowth::{InkletSlot, KinFollowerSlot};
use sts2_content::monsters::act2_hive::{ExoskeletonSlot, InfestedPrismSlot};
use sts2_content::monsters::shared::DenseVegetationWrigglerSlot;
use sts2_core::cards::Card;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::invariants::{
    check_combat_invariants, check_post_step, check_turn_boundary, snapshot_pre,
    snapshot_turn_anchor,
};
use sts2_core::{
    apply, legal_actions, CardUid, CharacterId, CombatPhase, CombatState, Creature, CreatureUid,
    DotNetRng, IntentHistory, MonsterId, MonsterKind, PlayerCombatState, PotionKind, Relic,
    RelicKind, RunRngSet, StepResult,
};

const MAX_DECISIONS: usize = 200;
const MAX_DECK_SIZE: usize = 15;
const MIN_DECK_SIZE: usize = 8;
const MAX_ENEMIES: usize = 3;
const MAX_RELICS: usize = 5;
const MAX_POTION_SLOTS: usize = 3;
/// Ascension the fuzzed combat runs at. A0 — the target has never varied it,
/// and `CombatState::ascension` must agree with the `MonsterKind::starting_hp`
/// roll (A8 `ToughEnemies` picks the high HP of each monster's pair), so the
/// two reads share this one constant rather than each hard-coding a zero.
const ASCENSION: u8 = 0;

// ── Character + card pools ──────────────────────────────────────────

#[derive(Debug, Arbitrary, Clone, Copy)]
enum FuzzCharacter {
    Ironclad,
    Silent,
    Defect,
    Necrobinder,
    Regent,
}

impl FuzzCharacter {
    fn to_character_id(self) -> CharacterId {
        match self {
            FuzzCharacter::Ironclad => CharacterId::Ironclad,
            FuzzCharacter::Silent => CharacterId::Silent,
            FuzzCharacter::Defect => CharacterId::Defect,
            FuzzCharacter::Necrobinder => CharacterId::Necrobinder,
            FuzzCharacter::Regent => CharacterId::Regent,
        }
    }

    /// Per-character card pool, with the shared COLORLESS pool folded in.
    /// For Ironclad/Silent/Defect we use the full spec_index slices (~85
    /// cards each). Necrobinder and Regent fall back to their starter-deck
    /// factories since they don't have a full spec_index entry yet at this
    /// pin.
    ///
    /// GH #422: the colorless pool was previously absent, so ~90 cards —
    /// including every `current_damage` / `current_block` reader that the
    /// empty-Play-pile panic fired from (BOLAS, THRUMMING_HATCHET,
    /// SEEKER_STRIKE, NEOWS_FURY, …) — were unreachable from this target.
    /// Colorless cards enter a real run from card rewards, events, Neow and
    /// mid-combat generation, so any character can hold them.
    fn card_pool(self) -> Vec<CardSpec> {
        let mut pool: Vec<CardSpec> = match self {
            FuzzCharacter::Ironclad => IRONCLAD_SPECS.iter().map(|f| f()).collect(),
            FuzzCharacter::Silent => SILENT_SPECS.iter().map(|f| f()).collect(),
            FuzzCharacter::Defect => DEFECT_SPECS.iter().map(|f| f()).collect(),
            FuzzCharacter::Necrobinder => necrobinder_starter_deck(),
            FuzzCharacter::Regent => regent_starter_deck(),
        };
        pool.extend(COLORLESS_SPECS.iter().map(|f| f()));
        pool
    }

    fn starter_deck(self) -> Vec<CardSpec> {
        match self {
            FuzzCharacter::Ironclad => ironclad_starter_deck(),
            FuzzCharacter::Silent => silent_starter_deck(),
            FuzzCharacter::Defect => DEFECT_SPECS.iter().map(|f| f()).collect(),
            FuzzCharacter::Necrobinder => necrobinder_starter_deck(),
            FuzzCharacter::Regent => regent_starter_deck(),
        }
    }
}

// ── Monster pool ─────────────────────────────────────────────────────

const PLAIN_ENEMIES: &[MonsterKind] = &[
    // Act 1 weak / common
    MonsterKind::ShrinkerBeetle,
    MonsterKind::FuzzyWurmCrawler,
    MonsterKind::LeafSlimeS,
    MonsterKind::TwigSlimeS,
    MonsterKind::LeafSlimeM,
    MonsterKind::TwigSlimeM,
    MonsterKind::CubexConstruct,
    MonsterKind::Flyconid,
    MonsterKind::EyeWithTeeth,
    MonsterKind::Fogmog,
    MonsterKind::Mawler,
    MonsterKind::VineShambler,
    MonsterKind::SlitheringStrangler,
    MonsterKind::SnappingJaxfruit,
    // Act 1 elites + boss
    MonsterKind::AssassinRubyRaider,
    MonsterKind::AxeRubyRaider,
    MonsterKind::BruteRubyRaider,
    MonsterKind::CrossbowRubyRaider,
    MonsterKind::TrackerRubyRaider,
    MonsterKind::BygoneEffigy,
    MonsterKind::Byrdonis,
    // Act 2/3/4 — respawners, egg-spawners, summoners
    MonsterKind::PhrogParasite,
    MonsterKind::Parafright,
    MonsterKind::Vantom,
    MonsterKind::CeremonialBeast,
    MonsterKind::KinPriest,
    MonsterKind::Wriggler,
    MonsterKind::ThievingHopper,
    MonsterKind::Tunneler,
    MonsterKind::BowlbugEgg,
    MonsterKind::BowlbugNectar,
    MonsterKind::BowlbugRock,
    MonsterKind::BowlbugSilk,
    MonsterKind::HunterKiller,
    MonsterKind::LouseProgenitor,
    MonsterKind::Myte,
    MonsterKind::ToughEgg,
    MonsterKind::Ovicopter,
    MonsterKind::SlumberingBeetle,
    MonsterKind::SpinyToad,
    MonsterKind::TheObscura,
    MonsterKind::Entomancer,
    MonsterKind::TheInsatiable,
    MonsterKind::KnowledgeDemon,
    MonsterKind::Crusher,
    MonsterKind::Rocket,
    MonsterKind::DevotedSculptor,
    MonsterKind::ScrollOfBiting,
    MonsterKind::TurretOperator,
    MonsterKind::Zapbot,
    MonsterKind::Stabbot,
    MonsterKind::Guardbot,
    MonsterKind::Noisebot,
    // Knights + misc act 4
    MonsterKind::FrogKnight,
    MonsterKind::GlobeHead,
    MonsterKind::OwlMagistrate,
    MonsterKind::SlimedBerserker,
    MonsterKind::FlailKnight,
    MonsterKind::MysteriousKnight,
    MonsterKind::LivingShield,
    MonsterKind::MagiKnight,
    MonsterKind::SpectralKnight,
    MonsterKind::MechaKnight,
];

#[derive(Debug, Arbitrary, Clone, Copy)]
enum ArgsEnemy {
    Inklet,
    KinFollower,
    Exoskeleton,
    Chomper,
    Axebot,
    InfestedPrism,
    DenseVegWriggler,
}

impl ArgsEnemy {
    fn build(self, byte: u8) -> (MonsterKind, MonsterSpawnArgs) {
        match self {
            ArgsEnemy::Inklet => {
                let slot = match byte % 3 {
                    0 => InkletSlot::First,
                    1 => InkletSlot::Second,
                    _ => InkletSlot::Other,
                };
                (MonsterKind::Inklet, MonsterSpawnArgs::Inklet { slot })
            }
            ArgsEnemy::KinFollower => {
                let slot = match byte % 3 {
                    0 => KinFollowerSlot::First,
                    1 => KinFollowerSlot::Second,
                    _ => KinFollowerSlot::Third,
                };
                (
                    MonsterKind::KinFollower,
                    MonsterSpawnArgs::KinFollower { slot },
                )
            }
            ArgsEnemy::Exoskeleton => {
                let slot = match byte % 4 {
                    0 => ExoskeletonSlot::First,
                    1 => ExoskeletonSlot::Second,
                    2 => ExoskeletonSlot::Third,
                    _ => ExoskeletonSlot::Fourth,
                };
                (
                    MonsterKind::Exoskeleton,
                    MonsterSpawnArgs::Exoskeleton { slot },
                )
            }
            ArgsEnemy::Chomper => (
                MonsterKind::Chomper,
                MonsterSpawnArgs::Chomper {
                    scream_first: byte & 1 == 0,
                },
            ),
            ArgsEnemy::Axebot => (
                MonsterKind::Axebot,
                MonsterSpawnArgs::Axebot {
                    start_with_boot_up: byte & 1 == 0,
                },
            ),
            ArgsEnemy::InfestedPrism => {
                let slot = match byte % 3 {
                    0 => InfestedPrismSlot::First,
                    1 => InfestedPrismSlot::Second,
                    _ => InfestedPrismSlot::Other,
                };
                (
                    MonsterKind::InfestedPrism,
                    MonsterSpawnArgs::InfestedPrism { slot },
                )
            }
            ArgsEnemy::DenseVegWriggler => {
                let slot = match byte % 4 {
                    0 => DenseVegetationWrigglerSlot::First,
                    1 => DenseVegetationWrigglerSlot::Second,
                    2 => DenseVegetationWrigglerSlot::Third,
                    _ => DenseVegetationWrigglerSlot::Fourth,
                };
                (
                    MonsterKind::DenseVegetationWriggler,
                    MonsterSpawnArgs::DenseVegetationWriggler { slot },
                )
            }
        }
    }
}

#[derive(Debug, Arbitrary)]
enum EnemyPick {
    Plain(u8),
    Args(ArgsEnemy, u8),
}

impl EnemyPick {
    fn resolve(&self) -> (MonsterKind, MonsterSpawnArgs) {
        match self {
            EnemyPick::Plain(idx) => (
                PLAIN_ENEMIES[(*idx as usize) % PLAIN_ENEMIES.len()],
                MonsterSpawnArgs::None,
            ),
            EnemyPick::Args(kind, byte) => kind.build(*byte),
        }
    }
}

// ── Relic + potion pools ────────────────────────────────────────────

/// Curated RelicKind list (~260) at pin `01c094d`. Hand-listed because
/// the enum doesn't derive EnumIter and we can't add it from inside
/// the fuzz crate. Skips Fake* and the Pael's-set relics (which expect
/// scripted run-state context not present here).
const RELIC_POOL: &[RelicKind] = &[
    RelicKind::BurningBlood,
    RelicKind::CrackedCore,
    RelicKind::RingOfTheSnake,
    RelicKind::DivineRight,
    RelicKind::BoundPhylactery,
    RelicKind::AmethystAubergine,
    RelicKind::Anchor,
    RelicKind::BagOfPreparation,
    RelicKind::BloodVial,
    RelicKind::BoneFlute,
    RelicKind::BronzeScales,
    RelicKind::CentennialPuzzle,
    RelicKind::DataDisk,
    RelicKind::FencingManual,
    RelicKind::FestivePopper,
    RelicKind::Gorget,
    RelicKind::HappyFlower,
    RelicKind::JuzuBracelet,
    RelicKind::Lantern,
    RelicKind::MealTicket,
    RelicKind::OddlySmoothStone,
    RelicKind::Pendulum,
    RelicKind::Permafrost,
    RelicKind::PotionBelt,
    RelicKind::RedSkull,
    RelicKind::RegalPillow,
    RelicKind::SneckoSkull,
    RelicKind::Strawberry,
    RelicKind::StrikeDummy,
    RelicKind::TinyMailbox,
    RelicKind::Vajra,
    RelicKind::VenerableTeaSet,
    RelicKind::WarPaint,
    RelicKind::Whetstone,
    RelicKind::BookOfFiveRings,
    RelicKind::Akabeko,
    RelicKind::BagOfMarbles,
    RelicKind::Bellows,
    RelicKind::BookRepairKnife,
    RelicKind::BowlerHat,
    RelicKind::Candelabra,
    RelicKind::EternalFeather,
    RelicKind::FuneraryMask,
    RelicKind::GalacticDust,
    RelicKind::GoldPlatedCables,
    RelicKind::GremlinHorn,
    RelicKind::HornCleat,
    RelicKind::JossPaper,
    RelicKind::Kusarigama,
    RelicKind::LetterOpener,
    RelicKind::LuckyFysh,
    RelicKind::MercuryHourglass,
    RelicKind::MiniatureCannon,
    RelicKind::Nunchaku,
    RelicKind::Orichalcum,
    RelicKind::OrnamentalFan,
    RelicKind::Pantograph,
    RelicKind::PaperPhrog,
    RelicKind::ParryingShield,
    RelicKind::Pear,
    RelicKind::PenNib,
    RelicKind::PetrifiedToad,
    RelicKind::Planisphere,
    RelicKind::RedMask,
    RelicKind::Regalite,
    RelicKind::ReptileTrinket,
    RelicKind::RippleBasin,
    RelicKind::SelfFormingClay,
    RelicKind::SparklingRouge,
    RelicKind::StoneCracker,
    RelicKind::SymbioticVirus,
    RelicKind::Tingsha,
    RelicKind::TuningFork,
    RelicKind::TwistedFunnel,
    RelicKind::Vambrace,
    RelicKind::BigMushroom,
    RelicKind::BingBong,
    RelicKind::BoneTea,
    RelicKind::Byrdpip,
    RelicKind::ChosenCheese,
    RelicKind::DarkstonePeriapt,
    RelicKind::DaughterOfTheWind,
    RelicKind::DreamCatcher,
    RelicKind::EmberTea,
    RelicKind::ForgottenSoul,
    RelicKind::FragrantMushroom,
    RelicKind::FresnelLens,
    RelicKind::HandDrill,
    RelicKind::HistoryCourse,
    RelicKind::LostWisp,
    RelicKind::MawBank,
    RelicKind::MrStruggles,
    RelicKind::PollinousCore,
    RelicKind::RoyalPoison,
    RelicKind::SwordOfStone,
    RelicKind::SwordOfJade,
    RelicKind::TeaOfDiscourtesy,
    RelicKind::TheBoot,
    RelicKind::ArtOfWar,
    RelicKind::BeatingRemnant,
    RelicKind::BigHat,
    RelicKind::Bookmark,
    RelicKind::CaptainsWheel,
    RelicKind::Chandelier,
    RelicKind::CharonsAshes,
    RelicKind::CloakClasp,
    RelicKind::DemonTongue,
    RelicKind::EmotionChip,
    RelicKind::FrozenEgg,
    RelicKind::GamblingChip,
    RelicKind::GamePiece,
    RelicKind::Girya,
    RelicKind::HelicalDart,
    RelicKind::IceCream,
    RelicKind::IntimidatingHelmet,
    RelicKind::IvoryTile,
    RelicKind::Kunai,
    RelicKind::LastingCandy,
    RelicKind::LizardTail,
    RelicKind::LunarPastry,
    RelicKind::Mango,
    RelicKind::MeatOnTheBone,
    RelicKind::Metronome,
    RelicKind::MiniRegent,
    RelicKind::MoltenEgg,
    RelicKind::MummifiedHand,
    RelicKind::OldCoin,
    RelicKind::OrangeDough,
    RelicKind::PaperKrane,
    RelicKind::Pocketwatch,
    RelicKind::PowerCell,
    RelicKind::PrayerWheel,
    RelicKind::RainbowRing,
    RelicKind::RazorTooth,
    RelicKind::RuinedHelmet,
    RelicKind::Shovel,
    RelicKind::Shuriken,
    RelicKind::StoneCalendar,
    RelicKind::SturdyClamp,
    RelicKind::TheCourier,
    RelicKind::ToughBandages,
    RelicKind::ToxicEgg,
    RelicKind::TungstenRod,
    RelicKind::UnceasingTop,
    RelicKind::UnsettlingLamp,
    RelicKind::VexingPuzzlebox,
    RelicKind::WhiteBeastStatue,
    RelicKind::WhiteStar,
    RelicKind::Brimstone,
    RelicKind::BoomingConch,
    RelicKind::TheAbacus,
    RelicKind::SlingOfCourage,
    RelicKind::BlackBlood,
    RelicKind::RingOfTheDrake,
    RelicKind::InfusedCore,
    RelicKind::PhylacteryUnbound,
    RelicKind::DivineDestiny,
    RelicKind::ScrollBoxes,
    RelicKind::LeadPaperweight,
    RelicKind::ArcaneScroll,
    RelicKind::NewLeaf,
    RelicKind::PrecariousShears,
    RelicKind::LeafyPoultice,
    RelicKind::CursedPearl,
    RelicKind::LostCoffer,
    RelicKind::Pomander,
    RelicKind::SilverCrucible,
    RelicKind::FishingRod,
    RelicKind::SmallCapsule,
    RelicKind::ArchaicTooth,
    RelicKind::Astrolabe,
    RelicKind::NutritiousSoup,
    RelicKind::VeryHotCocoa,
    RelicKind::DelicateFrond,
    RelicKind::TriBoomerang,
    RelicKind::AlchemicalCoffer,
    RelicKind::BeautifulBracelet,
    RelicKind::BeltBuckle,
    RelicKind::BiiigHug,
    RelicKind::BlackStar,
    RelicKind::BlessedAntler,
    RelicKind::BloodSoakedRose,
    RelicKind::Bread,
    RelicKind::BrilliantScarf,
    RelicKind::BurningSticks,
    RelicKind::CallingBell,
    RelicKind::Cauldron,
    RelicKind::ChemicalX,
    RelicKind::ChoicesParadox,
    RelicKind::Circlet,
    RelicKind::Claws,
    RelicKind::Crossbow,
    RelicKind::DiamondDiadem,
    RelicKind::DingyRug,
    RelicKind::DistinguishedCape,
    RelicKind::DollysMirror,
    RelicKind::DragonFruit,
    RelicKind::Driftwood,
    RelicKind::DustyTome,
    RelicKind::Ectoplasm,
    RelicKind::ElectricShrymp,
    RelicKind::EmptyCage,
    RelicKind::Fiddle,
    RelicKind::FurCoat,
    RelicKind::GhostSeed,
    RelicKind::GlassEye,
    RelicKind::Glitter,
    RelicKind::GnarledHammer,
    RelicKind::GoldenCompass,
    RelicKind::GoldenPearl,
    RelicKind::IronClub,
    RelicKind::JeweledMask,
    RelicKind::JewelryBox,
    RelicKind::Kifuda,
    RelicKind::LargeCapsule,
    RelicKind::LavaLamp,
    RelicKind::LavaRock,
    RelicKind::LeesWaffle,
    RelicKind::LoomingFruit,
    RelicKind::LordsParasol,
    RelicKind::MassiveScroll,
    RelicKind::MeatCleaver,
    RelicKind::MembershipCard,
    RelicKind::MiniatureTent,
    RelicKind::MusicBox,
    RelicKind::MysticLighter,
    RelicKind::NinjaScroll,
    RelicKind::NutritiousOyster,
    RelicKind::Orrery,
    RelicKind::PandorasBox,
    RelicKind::PhilosophersStone,
    RelicKind::PreciseScissors,
    RelicKind::PreservedFog,
    RelicKind::PrismaticGem,
    RelicKind::PumpkinCandle,
    RelicKind::PunchDagger,
    RelicKind::RadiantPearl,
    RelicKind::RingingTriangle,
    RelicKind::RoyalStamp,
    RelicKind::RunicCapacitor,
    RelicKind::RunicPyramid,
    RelicKind::Sai,
    RelicKind::SandCastle,
    RelicKind::ScreamingFlagon,
    RelicKind::SeaGlass,
    RelicKind::SealOfGold,
    RelicKind::SereTalon,
    RelicKind::SignetRing,
    RelicKind::SneckoEye,
    RelicKind::Sozu,
    RelicKind::SpikedGauntlets,
    RelicKind::StoneHumidifier,
    RelicKind::Storybook,
    RelicKind::TanxsWhistle,
    RelicKind::ThrowingAxe,
    RelicKind::ToastyMittens,
    RelicKind::Toolbox,
    RelicKind::TouchOfOrobas,
    RelicKind::ToyBox,
    RelicKind::UndyingSigil,
    RelicKind::VelvetChoker,
    RelicKind::WarHammer,
    RelicKind::WhisperingEarring,
    RelicKind::WingCharm,
    RelicKind::YummyCookie,
];

/// All 63 PotionKind variants at pin `01c094d`.
const POTION_POOL: &[PotionKind] = &[
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

// ── FuzzInput ───────────────────────────────────────────────────────

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    seed: u64,
    character: FuzzCharacter,
    deck_picks: Vec<u8>,
    upgrade_bits: Vec<u8>,
    enemies: Vec<EnemyPick>,
    relics: Vec<u16>,
    potions: Vec<u8>,
    /// Per-deck-slot `base_replay_count` (mod 3) — the HiddenGem `Replay`
    /// enchant. GH #422: the empty-Play-pile panic fired from a card's
    /// REPLAY iteration, a path with no other reachable source in this
    /// target (OneTwoPunch/EchoForm have to be drawn AND played first).
    replay_bits: Vec<u8>,
    decisions: Vec<u16>,
}

// ── Builder ────────────────────────────────────────────────────────

fn build_deck(character: FuzzCharacter, picks: &[u8], upgrades: &[u8]) -> Vec<(CardSpec, bool)> {
    let mut pool = character.card_pool();
    if pool.is_empty() {
        pool = character.starter_deck();
    }
    if pool.is_empty() {
        return Vec::new();
    }
    let n = MIN_DECK_SIZE.max(picks.len().min(MAX_DECK_SIZE));
    (0..n)
        .map(|i| {
            let pool_idx = picks
                .get(i)
                .map(|p| (*p as usize) % pool.len())
                .unwrap_or(i % pool.len());
            let upgraded = upgrades.get(i).map(|b| b & 1 == 1).unwrap_or(false);
            (pool[pool_idx], upgraded)
        })
        .collect()
}

fn build_combat(input: &FuzzInput) -> CombatState {
    let deck = build_deck(input.character, &input.deck_picks, &input.upgrade_bits);
    let mut draw: SmallVec<[CardUid; 32]> = SmallVec::new();
    let mut cards_arena: Vec<Card> = Vec::with_capacity(deck.len());
    for (i, (spec, upgraded)) in deck.iter().enumerate() {
        let raw = u32::try_from(i + 1).expect("deck size fits in u32");
        let uid = CardUid::new(raw).expect("non-zero");
        draw.push(uid);
        cards_arena.push(Card {
            uid,
            card_id: spec.id,
            cost: sts2_core::CardEnergyCost::new(spec.cost),
            card_type: spec.card_type,
            target_type: spec.target_type,
            upgraded: *upgraded,
            base_damage: spec.base_damage,
            base_block: spec.base_block,
            // HiddenGem's `Replay` enchant (GH #422 — see `replay_bits`).
            base_replay_count: input.replay_bits.get(i).map_or(0, |b| i32::from(*b % 3)),
            ..Card::pod_defaults()
        });
    }

    let mut enemy_picks: Vec<&EnemyPick> = input.enemies.iter().take(MAX_ENEMIES).collect();
    let fallback = EnemyPick::Plain(0);
    if enemy_picks.is_empty() {
        enemy_picks.push(&fallback);
    }
    let n_enemies = enemy_picks.len();

    let mut rng = DotNetRng::new(input.seed as i32);
    let mut creatures = Vec::with_capacity(n_enemies);
    let mut enemy_uids: SmallVec<[CreatureUid; 8]> = SmallVec::new();
    let alone = n_enemies == 1;
    for (idx, pick) in enemy_picks.iter().enumerate() {
        let (kind, args) = pick.resolve();
        let raw = u32::try_from(idx + 1).expect("enemy idx fits u32");
        let uid = CreatureUid::new(raw).expect("non-zero");
        let hp = kind.starting_hp(&mut rng, ASCENSION);
        let pos = SpawnPosition {
            alone,
            front: idx == 0,
            back: idx + 1 == n_enemies,
            index: u8::try_from(idx).unwrap_or(u8::MAX),
        };
        creatures.push(Creature {
            id: uid.0,
            current_hp: hp,
            max_hp: hp,
            block: 0,
            powers: SmallVec::new(),
            species: Some(kind),
            monster_id: MonsterId(kind.python_name()),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: Some(build_ai(kind, args, pos)),
        });
        enemy_uids.push(uid);
    }

    let mut relics_vec: Vec<Relic> = Vec::new();
    for (i, raw_idx) in input.relics.iter().take(MAX_RELICS).enumerate() {
        let kind = RELIC_POOL[(*raw_idx as usize) % RELIC_POOL.len()];
        if relics_vec.iter().any(|r| r.kind == kind) {
            continue;
        }
        let id = NonZeroU32::new((i + 1) as u32).expect("non-zero");
        relics_vec.push(Relic {
            id,
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }

    let mut potions_sv: SmallVec<[PotionKind; 5]> = SmallVec::new();
    for raw_idx in input.potions.iter().take(MAX_POTION_SLOTS) {
        let p = POTION_POOL[(*raw_idx as usize) % POTION_POOL.len()];
        potions_sv.push(p);
    }

    // Only the fields the fuzzer actually varies are named; everything else
    // comes from `Default`. This crate is OUTSIDE the workspace (cargo-fuzz
    // needs nightly), so `cargo check --workspace` never compiles it and an
    // exhaustive literal here silently rots on every new engine field — it
    // had accumulated 17 missing fields before GH #408 noticed.
    let player = PlayerCombatState {
        draw,
        max_energy: 3,
        current_hp: 80,
        max_hp: 80,
        character: Some(input.character.to_character_id()),
        potions: potions_sv,
        ..PlayerCombatState::default()
    };

    let mut state = CombatState {
        rng: DotNetRng::new(input.seed as i32),
        run_rng: RunRngSet::new(input.seed as i32),
        creatures,
        cards: cards_arena.into(),
        relics: relics_vec.into(),
        player,
        enemy_uids,
        ascension: ASCENSION,
        ..CombatState::default()
    };

    let r0 = start_combat(&mut state);
    assert_eq!(r0, StepResult::Advanced, "start_combat should advance");
    let r1 = start_player_turn(&mut state);
    assert_eq!(r1, StepResult::Advanced, "start_player_turn should advance");
    state
}

// ── Determinism digest ─────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
struct Digest {
    php: i32,
    pblock: i32,
    penergy: i64,
    hand: usize,
    draw: usize,
    discard: usize,
    exhaust: usize,
    play: usize,
    round: u64,
    is_over: bool,
    player_won: bool,
    sum_creature_hp: i64,
    sum_creature_block: i64,
    n_creatures: usize,
}

fn digest(state: &CombatState) -> Digest {
    Digest {
        php: state.player.current_hp,
        pblock: state.player.block,
        penergy: state.player.energy as i64,
        hand: state.player.hand.len(),
        draw: state.player.draw.len(),
        discard: state.player.discard.len(),
        exhaust: state.player.exhaust.len(),
        play: state.player.play.len(),
        round: state.round as u64,
        is_over: state.is_over,
        player_won: state.player_won,
        sum_creature_hp: state.creatures.iter().map(|c| c.current_hp as i64).sum(),
        sum_creature_block: state.creatures.iter().map(|c| c.block as i64).sum(),
        n_creatures: state.creatures.len(),
    }
}

// ── Drive ──────────────────────────────────────────────────────────

fn drive(input: &FuzzInput) -> Digest {
    let mut state = build_combat(input);
    check_combat_invariants(&state, "post-setup").unwrap();
    let mut turn_anchor = snapshot_turn_anchor(&state);

    for (steps, pick) in input.decisions.iter().copied().enumerate() {
        if steps >= MAX_DECISIONS || state.is_over {
            break;
        }
        let actions = legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        let idx = (pick as usize) % actions.len();
        let action = actions[idx].clone();
        let pre = snapshot_pre(&state, &action);
        let result = apply(&mut state, action.clone());
        match result {
            StepResult::Advanced
            | StepResult::CombatWon
            | StepResult::CombatLost
            | StepResult::PendingChoice => {}
            StepResult::InvalidTarget => {
                panic!(
                    "InvalidTarget on a legal_actions-derived action: action={action:?} \
                     character={:?} seed={} step={steps}",
                    input.character, input.seed
                );
            }
            // `legal_actions` never offers a play the engine refuses (GH #1122).
            StepResult::IllegalPlay => {
                panic!(
                    "IllegalPlay on a legal_actions-derived action: action={action:?} \
                     character={:?} seed={} step={steps}",
                    input.character, input.seed
                );
            }
        }
        check_post_step(&pre, &state, "post-step").unwrap();
        check_combat_invariants(&state, "post-step").unwrap();

        // Turn-boundary detection: `round` increments on every full
        // round (player turn + enemy turn). When it changes AND we're
        // back in the player play phase, the new player turn just
        // started — fire the cross-turn invariants.
        if state.round > turn_anchor.round && matches!(state.phase, CombatPhase::PlayerPlayPhase) {
            check_turn_boundary(&turn_anchor, &state, "turn-boundary").unwrap();
            turn_anchor = snapshot_turn_anchor(&state);
        }
    }
    digest(&state)
}

fuzz_target!(|input: FuzzInput| {
    let d1 = drive(&input);
    let d2 = drive(&input);
    assert_eq!(d1, d2, "non-deterministic replay (seed={})", input.seed);
});
