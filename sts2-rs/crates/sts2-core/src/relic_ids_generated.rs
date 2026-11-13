// HAND-MAINTAINED against ../sts2-wiki/decompiled/v0.107.1/ — edit directly.
// The `_generated` filename is historical; there is no generator any more.
//
// Origin: bead sts2-rs-l7l (P7-1) bulk-scraped this enum from
// `sts2-rl-agent@01c094d:sts2_env/relics/base.py::RelicId` via a
// `scripts/scrape_relic_ids.py` that no longer exists. Unlike its card and
// power siblings this one was still byte-identical to its output when
// retired — relics happened not to churn in v0.107.1 — but it read the same
// archived, non-authoritative Python pin, so the first hand-fix against the
// C# would have been silently reverted by the next rerun. Retired with the
// others; the v0.107.1 C# decompilation is the source of truth.
//
// The `to_python_name` arm still pins the upper-snake-case string the parity
// exporter emits; keep it in sync with the enum by hand.
//
// Trailing variants (bead sts2-rs-4rwv4, W9) exist in the C# game but not
// the Python pin, so the enum is a superset of both sources.

/// Stable relic identifier — one variant per distinct Python `RelicId`.
///
/// Mirrors the `CardId` pattern established by bead `sts2-rs-ov4`
/// (P6-1). The runtime POD `sts2_core::Relic` carries an
/// [`RelicId`] field; the dispatch registry in
/// `sts2_core::relic_effects` keys per-relic on-trigger handlers
/// by this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum RelicId {
    /// `RelicId.BURNING_BLOOD` in Python.
    BurningBlood,
    /// `RelicId.BLACK_BLOOD` in Python.
    BlackBlood,
    /// `RelicId.RING_OF_THE_SNAKE` in Python.
    RingOfTheSnake,
    /// `RelicId.RING_OF_THE_DRAKE` in Python.
    RingOfTheDrake,
    /// `RelicId.CRACKED_CORE` in Python.
    CrackedCore,
    /// `RelicId.INFUSED_CORE` in Python.
    InfusedCore,
    /// `RelicId.BOUND_PHYLACTERY` in Python.
    BoundPhylactery,
    /// `RelicId.PHYLACTERY_UNBOUND` in Python.
    PhylacteryUnbound,
    /// `RelicId.DIVINE_RIGHT` in Python.
    DivineRight,
    /// `RelicId.DIVINE_DESTINY` in Python.
    DivineDestiny,
    /// `RelicId.AMETHYST_AUBERGINE` in Python.
    AmethystAubergine,
    /// `RelicId.ANCHOR` in Python.
    Anchor,
    /// `RelicId.BAG_OF_PREPARATION` in Python.
    BagOfPreparation,
    /// `RelicId.BLOOD_VIAL` in Python.
    BloodVial,
    /// `RelicId.BONE_FLUTE` in Python.
    BoneFlute,
    /// `RelicId.BRONZE_SCALES` in Python.
    BronzeScales,
    /// `RelicId.CENTENNIAL_PUZZLE` in Python.
    CentennialPuzzle,
    /// `RelicId.DATA_DISK` in Python.
    DataDisk,
    /// `RelicId.FENCING_MANUAL` in Python.
    FencingManual,
    /// `RelicId.FESTIVE_POPPER` in Python.
    FestivePopper,
    /// `RelicId.GORGET` in Python.
    Gorget,
    /// `RelicId.HAPPY_FLOWER` in Python.
    HappyFlower,
    /// `RelicId.JUZU_BRACELET` in Python.
    JuzuBracelet,
    /// `RelicId.LANTERN` in Python.
    Lantern,
    /// `RelicId.MEAL_TICKET` in Python.
    MealTicket,
    /// `RelicId.ODDLY_SMOOTH_STONE` in Python.
    OddlySmoothStone,
    /// `RelicId.PENDULUM` in Python.
    Pendulum,
    /// `RelicId.PERMAFROST` in Python.
    Permafrost,
    /// `RelicId.POTION_BELT` in Python.
    PotionBelt,
    /// `RelicId.RED_SKULL` in Python.
    RedSkull,
    /// `RelicId.REGAL_PILLOW` in Python.
    RegalPillow,
    /// `RelicId.SNECKO_SKULL` in Python.
    SneckoSkull,
    /// `RelicId.STRAWBERRY` in Python.
    Strawberry,
    /// `RelicId.STRIKE_DUMMY` in Python.
    StrikeDummy,
    /// `RelicId.TINY_MAILBOX` in Python.
    TinyMailbox,
    /// `RelicId.VAJRA` in Python.
    Vajra,
    /// `RelicId.VENERABLE_TEA_SET` in Python.
    VenerableTeaSet,
    /// `RelicId.WAR_PAINT` in Python.
    WarPaint,
    /// `RelicId.WHETSTONE` in Python.
    Whetstone,
    /// `RelicId.BOOK_OF_FIVE_RINGS` in Python.
    BookOfFiveRings,
    /// `RelicId.AKABEKO` in Python.
    Akabeko,
    /// `RelicId.BAG_OF_MARBLES` in Python.
    BagOfMarbles,
    /// `RelicId.BELLOWS` in Python.
    Bellows,
    /// `RelicId.BOOK_REPAIR_KNIFE` in Python.
    BookRepairKnife,
    /// `RelicId.BOWLER_HAT` in Python.
    BowlerHat,
    /// `RelicId.CANDELABRA` in Python.
    Candelabra,
    /// `RelicId.ETERNAL_FEATHER` in Python.
    EternalFeather,
    /// `RelicId.FUNERARY_MASK` in Python.
    FuneraryMask,
    /// `RelicId.GALACTIC_DUST` in Python.
    GalacticDust,
    /// `RelicId.GOLD_PLATED_CABLES` in Python.
    GoldPlatedCables,
    /// `RelicId.GREMLIN_HORN` in Python.
    GremlinHorn,
    /// `RelicId.HORN_CLEAT` in Python.
    HornCleat,
    /// `RelicId.JOSS_PAPER` in Python.
    JossPaper,
    /// `RelicId.KUSARIGAMA` in Python.
    Kusarigama,
    /// `RelicId.LETTER_OPENER` in Python.
    LetterOpener,
    /// `RelicId.LUCKY_FYSH` in Python.
    LuckyFysh,
    /// `RelicId.MERCURY_HOURGLASS` in Python.
    MercuryHourglass,
    /// `RelicId.MINIATURE_CANNON` in Python.
    MiniatureCannon,
    /// `RelicId.NUNCHAKU` in Python.
    Nunchaku,
    /// `RelicId.ORICHALCUM` in Python.
    Orichalcum,
    /// `RelicId.ORNAMENTAL_FAN` in Python.
    OrnamentalFan,
    /// `RelicId.PANTOGRAPH` in Python.
    Pantograph,
    /// `RelicId.PAPER_PHROG` in Python.
    PaperPhrog,
    /// `RelicId.PARRYING_SHIELD` in Python.
    ParryingShield,
    /// `RelicId.PEAR` in Python.
    Pear,
    /// `RelicId.PEN_NIB` in Python.
    PenNib,
    /// `RelicId.PETRIFIED_TOAD` in Python.
    PetrifiedToad,
    /// `RelicId.PLANISPHERE` in Python.
    Planisphere,
    /// `RelicId.RED_MASK` in Python.
    RedMask,
    /// `RelicId.REGALITE` in Python.
    Regalite,
    /// `RelicId.REPTILE_TRINKET` in Python.
    ReptileTrinket,
    /// `RelicId.RIPPLE_BASIN` in Python.
    RippleBasin,
    /// `RelicId.SELF_FORMING_CLAY` in Python.
    SelfFormingClay,
    /// `RelicId.SPARKLING_ROUGE` in Python.
    SparklingRouge,
    /// `RelicId.STONE_CRACKER` in Python.
    StoneCracker,
    /// `RelicId.TUNING_FORK` in Python.
    TuningFork,
    /// `RelicId.TINGSHA` in Python.
    Tingsha,
    /// `RelicId.VAMBRACE` in Python.
    Vambrace,
    /// `RelicId.ART_OF_WAR` in Python.
    ArtOfWar,
    /// `RelicId.BEATING_REMNANT` in Python.
    BeatingRemnant,
    /// `RelicId.BIG_HAT` in Python.
    BigHat,
    /// `RelicId.BOOKMARK` in Python.
    Bookmark,
    /// `RelicId.CAPTAINS_WHEEL` in Python.
    CaptainsWheel,
    /// `RelicId.CHANDELIER` in Python.
    Chandelier,
    /// `RelicId.CHARONS_ASHES` in Python.
    CharonsAshes,
    /// `RelicId.DEMON_TONGUE` in Python.
    DemonTongue,
    /// `RelicId.EMOTION_CHIP` in Python.
    EmotionChip,
    /// `RelicId.FROZEN_EGG` in Python.
    FrozenEgg,
    /// `RelicId.GAMBLING_CHIP` in Python.
    GamblingChip,
    /// `RelicId.GAME_PIECE` in Python.
    GamePiece,
    /// `RelicId.GIRYA` in Python.
    Girya,
    /// `RelicId.HELICAL_DART` in Python.
    HelicalDart,
    /// `RelicId.ICE_CREAM` in Python.
    IceCream,
    /// `RelicId.INTIMIDATING_HELMET` in Python.
    IntimidatingHelmet,
    /// `RelicId.IVORY_TILE` in Python.
    IvoryTile,
    /// `RelicId.KUNAI` in Python.
    Kunai,
    /// `RelicId.LASTING_CANDY` in Python.
    LastingCandy,
    /// `RelicId.LIZARD_TAIL` in Python.
    LizardTail,
    /// `RelicId.LUNAR_PASTRY` in Python.
    LunarPastry,
    /// `RelicId.MANGO` in Python.
    Mango,
    /// `RelicId.MEAT_ON_THE_BONE` in Python.
    MeatOnTheBone,
    /// `RelicId.METRONOME` in Python.
    Metronome,
    /// `RelicId.MINI_REGENT` in Python.
    MiniRegent,
    /// `RelicId.MOLTEN_EGG` in Python.
    MoltenEgg,
    /// `RelicId.MUMMIFIED_HAND` in Python.
    MummifiedHand,
    /// `RelicId.OLD_COIN` in Python.
    OldCoin,
    /// `RelicId.ORANGE_DOUGH` in Python.
    OrangeDough,
    /// `RelicId.POCKETWATCH` in Python.
    Pocketwatch,
    /// `RelicId.POWER_CELL` in Python.
    PowerCell,
    /// `RelicId.PRAYER_WHEEL` in Python.
    PrayerWheel,
    /// `RelicId.RAINBOW_RING` in Python.
    RainbowRing,
    /// `RelicId.RAZOR_TOOTH` in Python.
    RazorTooth,
    /// `RelicId.SHOVEL` in Python.
    Shovel,
    /// `RelicId.SHURIKEN` in Python.
    Shuriken,
    /// `RelicId.STONE_CALENDAR` in Python.
    StoneCalendar,
    /// `RelicId.STURDY_CLAMP` in Python.
    SturdyClamp,
    /// `RelicId.THE_COURIER` in Python.
    TheCourier,
    /// `RelicId.TOUGH_BANDAGES` in Python.
    ToughBandages,
    /// `RelicId.TOXIC_EGG` in Python.
    ToxicEgg,
    /// `RelicId.TUNGSTEN_ROD` in Python.
    TungstenRod,
    /// `RelicId.UNCEASING_TOP` in Python.
    UnceasingTop,
    /// `RelicId.UNSETTLING_LAMP` in Python.
    UnsettlingLamp,
    /// `RelicId.RUINED_HELMET` in Python.
    RuinedHelmet,
    /// `RelicId.VEXING_PUZZLEBOX` in Python.
    VexingPuzzlebox,
    /// `RelicId.WHITE_BEAST_STATUE` in Python.
    WhiteBeastStatue,
    /// `RelicId.WHITE_STAR` in Python.
    WhiteStar,
    /// `RelicId.PAPER_KRANE` in Python.
    PaperKrane,
    /// `RelicId.BELT_BUCKLE` in Python.
    BeltBuckle,
    /// `RelicId.BREAD` in Python.
    Bread,
    /// `RelicId.BRIMSTONE` in Python.
    Brimstone,
    /// `RelicId.BURNING_STICKS` in Python.
    BurningSticks,
    /// `RelicId.CAULDRON` in Python.
    Cauldron,
    /// `RelicId.CHEMICAL_X` in Python.
    ChemicalX,
    /// `RelicId.DINGY_RUG` in Python.
    DingyRug,
    /// `RelicId.DOLLYS_MIRROR` in Python.
    DollysMirror,
    /// `RelicId.DRAGON_FRUIT` in Python.
    DragonFruit,
    /// `RelicId.GHOST_SEED` in Python.
    GhostSeed,
    /// `RelicId.GNARLED_HAMMER` in Python.
    GnarledHammer,
    /// `RelicId.KIFUDA` in Python.
    Kifuda,
    /// `RelicId.LAVA_LAMP` in Python.
    LavaLamp,
    /// `RelicId.LEES_WAFFLE` in Python.
    LeesWaffle,
    /// `RelicId.MEMBERSHIP_CARD` in Python.
    MembershipCard,
    /// `RelicId.MINIATURE_TENT` in Python.
    MiniatureTent,
    /// `RelicId.MYSTIC_LIGHTER` in Python.
    MysticLighter,
    /// `RelicId.NINJA_SCROLL` in Python.
    NinjaScroll,
    /// `RelicId.ORRERY` in Python.
    Orrery,
    /// `RelicId.PUNCH_DAGGER` in Python.
    PunchDagger,
    /// `RelicId.RINGING_TRIANGLE` in Python.
    RingingTriangle,
    /// `RelicId.ROYAL_STAMP` in Python.
    RoyalStamp,
    /// `RelicId.RUNIC_CAPACITOR` in Python.
    RunicCapacitor,
    /// `RelicId.SCREAMING_FLAGON` in Python.
    ScreamingFlagon,
    /// `RelicId.SLING_OF_COURAGE` in Python.
    SlingOfCourage,
    /// `RelicId.THE_ABACUS` in Python.
    TheAbacus,
    /// `RelicId.TOOLBOX` in Python.
    Toolbox,
    /// `RelicId.UNDYING_SIGIL` in Python.
    UndyingSigil,
    /// `RelicId.VITRUVIAN_MINION` in Python.
    VitruvianMinion,
    /// `RelicId.WING_CHARM` in Python.
    WingCharm,
    /// `RelicId.ALCHEMICAL_COFFER` in Python.
    AlchemicalCoffer,
    /// `RelicId.ARCANE_SCROLL` in Python.
    ArcaneScroll,
    /// `RelicId.ARCHAIC_TOOTH` in Python.
    ArchaicTooth,
    /// `RelicId.ASTROLABE` in Python.
    Astrolabe,
    /// `RelicId.BEAUTIFUL_BRACELET` in Python.
    BeautifulBracelet,
    /// `RelicId.BIG_MUSHROOM` in Python.
    BigMushroom,
    /// `RelicId.BIIIG_HUG` in Python.
    BiiigHug,
    /// `RelicId.BING_BONG` in Python.
    BingBong,
    /// `RelicId.BLACK_STAR` in Python.
    BlackStar,
    /// `RelicId.BLESSED_ANTLER` in Python.
    BlessedAntler,
    /// `RelicId.BLOOD_SOAKED_ROSE` in Python.
    BloodSoakedRose,
    /// `RelicId.BONE_TEA` in Python.
    BoneTea,
    /// `RelicId.BOOMING_CONCH` in Python.
    BoomingConch,
    /// `RelicId.BRILLIANT_SCARF` in Python.
    BrilliantScarf,
    /// `RelicId.BYRDPIP` in Python.
    Byrdpip,
    /// `RelicId.CALLING_BELL` in Python.
    CallingBell,
    /// `RelicId.CHOICES_PARADOX` in Python.
    ChoicesParadox,
    /// `RelicId.CHOSEN_CHEESE` in Python.
    ChosenCheese,
    /// `RelicId.CLAWS` in Python.
    Claws,
    /// `RelicId.CROSSBOW` in Python.
    Crossbow,
    /// `RelicId.CURSED_PEARL` in Python.
    CursedPearl,
    /// `RelicId.DARKSTONE_PERIAPT` in Python.
    DarkstonePeriapt,
    /// `RelicId.DAUGHTER_OF_THE_WIND` in Python.
    DaughterOfTheWind,
    /// `RelicId.DELICATE_FROND` in Python.
    DelicateFrond,
    /// `RelicId.DIAMOND_DIADEM` in Python.
    DiamondDiadem,
    /// `RelicId.DISTINGUISHED_CAPE` in Python.
    DistinguishedCape,
    /// `RelicId.DREAM_CATCHER` in Python.
    DreamCatcher,
    /// `RelicId.DRIFTWOOD` in Python.
    Driftwood,
    /// `RelicId.DUSTY_TOME` in Python.
    DustyTome,
    /// `RelicId.ECTOPLASM` in Python.
    Ectoplasm,
    /// `RelicId.ELECTRIC_SHRYMP` in Python.
    ElectricShrymp,
    /// `RelicId.EMBER_TEA` in Python.
    EmberTea,
    /// `RelicId.EMPTY_CAGE` in Python.
    EmptyCage,
    /// `RelicId.FAKE_ANCHOR` in Python.
    FakeAnchor,
    /// `RelicId.FAKE_BLOOD_VIAL` in Python.
    FakeBloodVial,
    /// `RelicId.FAKE_HAPPY_FLOWER` in Python.
    FakeHappyFlower,
    /// `RelicId.FAKE_LEES_WAFFLE` in Python.
    FakeLeesWaffle,
    /// `RelicId.FAKE_MANGO` in Python.
    FakeMango,
    /// `RelicId.FAKE_MERCHANTS_RUG` in Python.
    FakeMerchantsRug,
    /// `RelicId.FAKE_ORICHALCUM` in Python.
    FakeOrichalcum,
    /// `RelicId.FAKE_SNECKO_EYE` in Python.
    FakeSneckoEye,
    /// `RelicId.FAKE_STRIKE_DUMMY` in Python.
    FakeStrikeDummy,
    /// `RelicId.FAKE_VENERABLE_TEA_SET` in Python.
    FakeVenerableTeaSet,
    /// `RelicId.FIDDLE` in Python.
    Fiddle,
    /// `RelicId.FORGOTTEN_SOUL` in Python.
    ForgottenSoul,
    /// `RelicId.FRAGRANT_MUSHROOM` in Python.
    FragrantMushroom,
    /// `RelicId.FRESNEL_LENS` in Python.
    FresnelLens,
    /// `RelicId.FUR_COAT` in Python.
    FurCoat,
    /// `RelicId.GLASS_EYE` in Python.
    GlassEye,
    /// `RelicId.GLITTER` in Python.
    Glitter,
    /// `RelicId.GOLDEN_COMPASS` in Python.
    GoldenCompass,
    /// `RelicId.GOLDEN_PEARL` in Python.
    GoldenPearl,
    /// `RelicId.HAND_DRILL` in Python.
    HandDrill,
    /// `RelicId.HISTORY_COURSE` in Python.
    HistoryCourse,
    /// `RelicId.IRON_CLUB` in Python.
    IronClub,
    /// `RelicId.JEWELED_MASK` in Python.
    JeweledMask,
    /// `RelicId.JEWELRY_BOX` in Python.
    JewelryBox,
    /// `RelicId.LARGE_CAPSULE` in Python.
    LargeCapsule,
    /// `RelicId.LAVA_ROCK` in Python.
    LavaRock,
    /// `RelicId.LEAD_PAPERWEIGHT` in Python.
    LeadPaperweight,
    /// `RelicId.LEAFY_POULTICE` in Python.
    LeafyPoultice,
    /// `RelicId.LOOMING_FRUIT` in Python.
    LoomingFruit,
    /// `RelicId.LORDS_PARASOL` in Python.
    LordsParasol,
    /// `RelicId.LOST_COFFER` in Python.
    LostCoffer,
    /// `RelicId.LOST_WISP` in Python.
    LostWisp,
    /// `RelicId.MASSIVE_SCROLL` in Python.
    MassiveScroll,
    /// `RelicId.MAW_BANK` in Python.
    MawBank,
    /// `RelicId.MEAT_CLEAVER` in Python.
    MeatCleaver,
    /// `RelicId.MR_STRUGGLES` in Python.
    MrStruggles,
    /// `RelicId.MUSIC_BOX` in Python.
    MusicBox,
    /// `RelicId.NEOWS_TORMENT` in Python.
    NeowsTorment,
    /// `RelicId.NEW_LEAF` in Python.
    NewLeaf,
    /// `RelicId.NUTRITIOUS_OYSTER` in Python.
    NutritiousOyster,
    /// `RelicId.NUTRITIOUS_SOUP` in Python.
    NutritiousSoup,
    /// `RelicId.PAELS_BLOOD` in Python.
    PaelsBlood,
    /// `RelicId.PAELS_CLAW` in Python.
    PaelsClaw,
    /// `RelicId.PAELS_EYE` in Python.
    PaelsEye,
    /// `RelicId.PAELS_FLESH` in Python.
    PaelsFlesh,
    /// `RelicId.PAELS_GROWTH` in Python.
    PaelsGrowth,
    /// `RelicId.PAELS_HORN` in Python.
    PaelsHorn,
    /// `RelicId.PAELS_LEGION` in Python.
    PaelsLegion,
    /// `RelicId.PAELS_TEARS` in Python.
    PaelsTears,
    /// `RelicId.PAELS_TOOTH` in Python.
    PaelsTooth,
    /// `RelicId.PAELS_WING` in Python.
    PaelsWing,
    /// `RelicId.PANDORAS_BOX` in Python.
    PandorasBox,
    /// `RelicId.PHILOSOPHERS_STONE` in Python.
    PhilosophersStone,
    /// `RelicId.POLLINOUS_CORE` in Python.
    PollinousCore,
    /// `RelicId.POMANDER` in Python.
    Pomander,
    /// `RelicId.PRECARIOUS_SHEARS` in Python.
    PrecariousShears,
    /// `RelicId.PRECISE_SCISSORS` in Python.
    PreciseScissors,
    /// `RelicId.PRESERVED_FOG` in Python.
    PreservedFog,
    /// `RelicId.PRISMATIC_GEM` in Python.
    PrismaticGem,
    /// `RelicId.PUMPKIN_CANDLE` in Python.
    PumpkinCandle,
    /// `RelicId.RADIANT_PEARL` in Python.
    RadiantPearl,
    /// `RelicId.ROYAL_POISON` in Python.
    RoyalPoison,
    /// `RelicId.RUNIC_PYRAMID` in Python.
    RunicPyramid,
    /// `RelicId.SAI` in Python.
    Sai,
    /// `RelicId.SAND_CASTLE` in Python.
    SandCastle,
    /// `RelicId.SCROLL_BOXES` in Python.
    ScrollBoxes,
    /// `RelicId.SEA_GLASS` in Python.
    SeaGlass,
    /// `RelicId.SEAL_OF_GOLD` in Python.
    SealOfGold,
    /// `RelicId.SERE_TALON` in Python.
    SereTalon,
    /// `RelicId.SIGNET_RING` in Python.
    SignetRing,
    /// `RelicId.SILVER_CRUCIBLE` in Python.
    SilverCrucible,
    /// `RelicId.SMALL_CAPSULE` in Python.
    SmallCapsule,
    /// `RelicId.SNECKO_EYE` in Python.
    SneckoEye,
    /// `RelicId.SOZU` in Python.
    Sozu,
    /// `RelicId.SPIKED_GAUNTLETS` in Python.
    SpikedGauntlets,
    /// `RelicId.STONE_HUMIDIFIER` in Python.
    StoneHumidifier,
    /// `RelicId.STORYBOOK` in Python.
    Storybook,
    /// `RelicId.SWORD_OF_STONE` in Python.
    SwordOfStone,
    /// `RelicId.SWORD_OF_JADE` in Python.
    SwordOfJade,
    /// `RelicId.TANXS_WHISTLE` in Python.
    TanxsWhistle,
    /// `RelicId.TEA_OF_DISCOURTESY` in Python.
    TeaOfDiscourtesy,
    /// `RelicId.THE_BOOT` in Python.
    TheBoot,
    /// `RelicId.THROWING_AXE` in Python.
    ThrowingAxe,
    /// `RelicId.TOASTY_MITTENS` in Python.
    ToastyMittens,
    /// `RelicId.TOUCH_OF_OROBAS` in Python.
    TouchOfOrobas,
    /// `RelicId.TOY_BOX` in Python.
    ToyBox,
    /// `RelicId.TRI_BOOMERANG` in Python.
    TriBoomerang,
    /// `RelicId.VELVET_CHOKER` in Python.
    VelvetChoker,
    /// `RelicId.VERY_HOT_COCOA` in Python.
    VeryHotCocoa,
    /// `RelicId.WAR_HAMMER` in Python.
    WarHammer,
    /// `RelicId.WHISPERING_EARRING` in Python.
    WhisperingEarring,
    /// `RelicId.WONGO_CUSTOMER_APPRECIATION_BADGE` in Python.
    WongoCustomerAppreciationBadge,
    /// `RelicId.WONGOS_MYSTERY_TICKET` in Python.
    WongosMysteryTicket,
    /// `RelicId.YUMMY_COOKIE` in Python.
    YummyCookie,
    /// `RelicId.SYMBIOTIC_VIRUS` in Python.
    SymbioticVirus,
    /// `RelicId.TWISTED_FUNNEL` in Python.
    TwistedFunnel,
    /// `RelicId.CLOAK_CLASP` in Python.
    CloakClasp,
    /// `RelicId.CIRCLET` in Python.
    Circlet,
    /// `RelicId.DEPRECATED_RELIC` in Python.
    DeprecatedRelic,
    /// `RelicId.NEOWS_TALISMAN` in Python.
    NeowsTalisman,
    /// `RelicId.WINGED_BOOTS` in Python.
    WingedBoots,
    /// `RelicId.PHIAL_HOLSTER` in Python.
    PhialHolster,
    /// `RelicId.NEOWS_BONES` in Python.
    NeowsBones,
    /// `RelicId.HEFTY_TABLET` in Python.
    HeftyTablet,
    /// `RelicId.FISHING_ROD` in Python.
    FishingRod,
    /// `RelicId.KALEIDOSCOPE` in Python.
    Kaleidoscope,
    /// `RelicId.SILKEN_TRESS` in Python.
    SilkenTress,
    // ── v0.111.0-only (GH #792). Not in [`RelicId::all`], which stays the
    // v0.107.1 universe (ranker feature tables, PPO vocab, random fixture
    // samplers read it); see [`RelicId::V0111_ONLY`] / [`RelicId::all_for`].
    /// v0.111.0 Neow curse option: adds the Dowsing quest card
    /// (`DowsingRod.cs`).
    DowsingRod,
    /// v0.111.0 Neow curse option: procures Ambergris, adds Guilty
    /// (`NeowsSacrifice.cs`).
    NeowsSacrifice,
    // ── v0.113.0-only. Not in [`RelicId::all`]; see [`RelicId::V0113_ONLY`].
    /// v0.113.0 event relic (`ObsidianVice.cs`) — MuseumOfTorture: turn end, 1 damage per card in hand to a random enemy.
    ObsidianVice,
    /// v0.113.0 event relic (`Thumbscrew.cs`) — MuseumOfTorture: multi-hit attack cards hit once more.
    Thumbscrew,
    /// v0.113.0 event relic (`TormentousCollar.cs`) — MuseumOfTorture: 2 Weak to all enemies at the start of elite fights.
    TormentousCollar,
    /// v0.113.0 event relic (`MightyWings.cs`) — ShrineOfTheWarlord: +2 Strength when a powered attack deals 30+.
    MightyWings,
    /// v0.113.0 event relic (`BathTowel.cs`) — ThermaeNonupeipe: heal 8 at the start of turn 1.
    BathTowel,
    /// v0.113.0 event relic (`BathRobes.cs`) — ThermaeNonupeipe: once, full heal at the start of the boss fight.
    BathRobes,
    /// v0.113.0 event relic (`Wuppy.cs`) — WuppyInTrouble: HP-tracked pet (AddsPet).
    Wuppy,
    /// v0.113.0 event relic (`Byrdine.cs`) — Byrdpip evolution (AddsPet).
    Byrdine,
}

impl RelicId {
    /// Python `RelicId.<v>.name` — the canonical parity-exporter string.
    #[must_use]
    pub const fn to_python_name(self) -> &'static str {
        match self {
            Self::BurningBlood => "BURNING_BLOOD",
            Self::BlackBlood => "BLACK_BLOOD",
            Self::RingOfTheSnake => "RING_OF_THE_SNAKE",
            Self::RingOfTheDrake => "RING_OF_THE_DRAKE",
            Self::CrackedCore => "CRACKED_CORE",
            Self::InfusedCore => "INFUSED_CORE",
            Self::BoundPhylactery => "BOUND_PHYLACTERY",
            Self::PhylacteryUnbound => "PHYLACTERY_UNBOUND",
            Self::DivineRight => "DIVINE_RIGHT",
            Self::DivineDestiny => "DIVINE_DESTINY",
            Self::AmethystAubergine => "AMETHYST_AUBERGINE",
            Self::Anchor => "ANCHOR",
            Self::BagOfPreparation => "BAG_OF_PREPARATION",
            Self::BloodVial => "BLOOD_VIAL",
            Self::BoneFlute => "BONE_FLUTE",
            Self::BronzeScales => "BRONZE_SCALES",
            Self::CentennialPuzzle => "CENTENNIAL_PUZZLE",
            Self::DataDisk => "DATA_DISK",
            Self::FencingManual => "FENCING_MANUAL",
            Self::FestivePopper => "FESTIVE_POPPER",
            Self::Gorget => "GORGET",
            Self::HappyFlower => "HAPPY_FLOWER",
            Self::JuzuBracelet => "JUZU_BRACELET",
            Self::Lantern => "LANTERN",
            Self::MealTicket => "MEAL_TICKET",
            Self::OddlySmoothStone => "ODDLY_SMOOTH_STONE",
            Self::Pendulum => "PENDULUM",
            Self::Permafrost => "PERMAFROST",
            Self::PotionBelt => "POTION_BELT",
            Self::RedSkull => "RED_SKULL",
            Self::RegalPillow => "REGAL_PILLOW",
            Self::SneckoSkull => "SNECKO_SKULL",
            Self::Strawberry => "STRAWBERRY",
            Self::StrikeDummy => "STRIKE_DUMMY",
            Self::TinyMailbox => "TINY_MAILBOX",
            Self::Vajra => "VAJRA",
            Self::VenerableTeaSet => "VENERABLE_TEA_SET",
            Self::WarPaint => "WAR_PAINT",
            Self::Whetstone => "WHETSTONE",
            Self::BookOfFiveRings => "BOOK_OF_FIVE_RINGS",
            Self::Akabeko => "AKABEKO",
            Self::BagOfMarbles => "BAG_OF_MARBLES",
            Self::Bellows => "BELLOWS",
            Self::BookRepairKnife => "BOOK_REPAIR_KNIFE",
            Self::BowlerHat => "BOWLER_HAT",
            Self::Candelabra => "CANDELABRA",
            Self::EternalFeather => "ETERNAL_FEATHER",
            Self::FuneraryMask => "FUNERARY_MASK",
            Self::GalacticDust => "GALACTIC_DUST",
            Self::GoldPlatedCables => "GOLD_PLATED_CABLES",
            Self::GremlinHorn => "GREMLIN_HORN",
            Self::HornCleat => "HORN_CLEAT",
            Self::JossPaper => "JOSS_PAPER",
            Self::Kusarigama => "KUSARIGAMA",
            Self::LetterOpener => "LETTER_OPENER",
            Self::LuckyFysh => "LUCKY_FYSH",
            Self::MercuryHourglass => "MERCURY_HOURGLASS",
            Self::MiniatureCannon => "MINIATURE_CANNON",
            Self::Nunchaku => "NUNCHAKU",
            Self::Orichalcum => "ORICHALCUM",
            Self::OrnamentalFan => "ORNAMENTAL_FAN",
            Self::Pantograph => "PANTOGRAPH",
            Self::PaperPhrog => "PAPER_PHROG",
            Self::ParryingShield => "PARRYING_SHIELD",
            Self::Pear => "PEAR",
            Self::PenNib => "PEN_NIB",
            Self::PetrifiedToad => "PETRIFIED_TOAD",
            Self::Planisphere => "PLANISPHERE",
            Self::RedMask => "RED_MASK",
            Self::Regalite => "REGALITE",
            Self::ReptileTrinket => "REPTILE_TRINKET",
            Self::RippleBasin => "RIPPLE_BASIN",
            Self::SelfFormingClay => "SELF_FORMING_CLAY",
            Self::SparklingRouge => "SPARKLING_ROUGE",
            Self::StoneCracker => "STONE_CRACKER",
            Self::TuningFork => "TUNING_FORK",
            Self::Tingsha => "TINGSHA",
            Self::Vambrace => "VAMBRACE",
            Self::ArtOfWar => "ART_OF_WAR",
            Self::BeatingRemnant => "BEATING_REMNANT",
            Self::BigHat => "BIG_HAT",
            Self::Bookmark => "BOOKMARK",
            Self::CaptainsWheel => "CAPTAINS_WHEEL",
            Self::Chandelier => "CHANDELIER",
            Self::CharonsAshes => "CHARONS_ASHES",
            Self::DemonTongue => "DEMON_TONGUE",
            Self::EmotionChip => "EMOTION_CHIP",
            Self::FrozenEgg => "FROZEN_EGG",
            Self::GamblingChip => "GAMBLING_CHIP",
            Self::GamePiece => "GAME_PIECE",
            Self::Girya => "GIRYA",
            Self::HelicalDart => "HELICAL_DART",
            Self::IceCream => "ICE_CREAM",
            Self::IntimidatingHelmet => "INTIMIDATING_HELMET",
            Self::IvoryTile => "IVORY_TILE",
            Self::Kunai => "KUNAI",
            Self::LastingCandy => "LASTING_CANDY",
            Self::LizardTail => "LIZARD_TAIL",
            Self::LunarPastry => "LUNAR_PASTRY",
            Self::Mango => "MANGO",
            Self::MeatOnTheBone => "MEAT_ON_THE_BONE",
            Self::Metronome => "METRONOME",
            Self::MiniRegent => "MINI_REGENT",
            Self::MoltenEgg => "MOLTEN_EGG",
            Self::MummifiedHand => "MUMMIFIED_HAND",
            Self::OldCoin => "OLD_COIN",
            Self::OrangeDough => "ORANGE_DOUGH",
            Self::Pocketwatch => "POCKETWATCH",
            Self::PowerCell => "POWER_CELL",
            Self::PrayerWheel => "PRAYER_WHEEL",
            Self::RainbowRing => "RAINBOW_RING",
            Self::RazorTooth => "RAZOR_TOOTH",
            Self::Shovel => "SHOVEL",
            Self::Shuriken => "SHURIKEN",
            Self::StoneCalendar => "STONE_CALENDAR",
            Self::SturdyClamp => "STURDY_CLAMP",
            Self::TheCourier => "THE_COURIER",
            Self::ToughBandages => "TOUGH_BANDAGES",
            Self::ToxicEgg => "TOXIC_EGG",
            Self::TungstenRod => "TUNGSTEN_ROD",
            Self::UnceasingTop => "UNCEASING_TOP",
            Self::UnsettlingLamp => "UNSETTLING_LAMP",
            Self::RuinedHelmet => "RUINED_HELMET",
            Self::VexingPuzzlebox => "VEXING_PUZZLEBOX",
            Self::WhiteBeastStatue => "WHITE_BEAST_STATUE",
            Self::WhiteStar => "WHITE_STAR",
            Self::PaperKrane => "PAPER_KRANE",
            Self::BeltBuckle => "BELT_BUCKLE",
            Self::Bread => "BREAD",
            Self::Brimstone => "BRIMSTONE",
            Self::BurningSticks => "BURNING_STICKS",
            Self::Cauldron => "CAULDRON",
            Self::ChemicalX => "CHEMICAL_X",
            Self::DingyRug => "DINGY_RUG",
            Self::DollysMirror => "DOLLYS_MIRROR",
            Self::DragonFruit => "DRAGON_FRUIT",
            Self::GhostSeed => "GHOST_SEED",
            Self::GnarledHammer => "GNARLED_HAMMER",
            Self::Kifuda => "KIFUDA",
            Self::LavaLamp => "LAVA_LAMP",
            Self::LeesWaffle => "LEES_WAFFLE",
            Self::MembershipCard => "MEMBERSHIP_CARD",
            Self::MiniatureTent => "MINIATURE_TENT",
            Self::MysticLighter => "MYSTIC_LIGHTER",
            Self::NinjaScroll => "NINJA_SCROLL",
            Self::Orrery => "ORRERY",
            Self::PunchDagger => "PUNCH_DAGGER",
            Self::RingingTriangle => "RINGING_TRIANGLE",
            Self::RoyalStamp => "ROYAL_STAMP",
            Self::RunicCapacitor => "RUNIC_CAPACITOR",
            Self::ScreamingFlagon => "SCREAMING_FLAGON",
            Self::SlingOfCourage => "SLING_OF_COURAGE",
            Self::TheAbacus => "THE_ABACUS",
            Self::Toolbox => "TOOLBOX",
            Self::UndyingSigil => "UNDYING_SIGIL",
            Self::VitruvianMinion => "VITRUVIAN_MINION",
            Self::WingCharm => "WING_CHARM",
            Self::AlchemicalCoffer => "ALCHEMICAL_COFFER",
            Self::ArcaneScroll => "ARCANE_SCROLL",
            Self::ArchaicTooth => "ARCHAIC_TOOTH",
            Self::Astrolabe => "ASTROLABE",
            Self::BeautifulBracelet => "BEAUTIFUL_BRACELET",
            Self::BigMushroom => "BIG_MUSHROOM",
            Self::BiiigHug => "BIIIG_HUG",
            Self::BingBong => "BING_BONG",
            Self::BlackStar => "BLACK_STAR",
            Self::BlessedAntler => "BLESSED_ANTLER",
            Self::BloodSoakedRose => "BLOOD_SOAKED_ROSE",
            Self::BoneTea => "BONE_TEA",
            Self::BoomingConch => "BOOMING_CONCH",
            Self::BrilliantScarf => "BRILLIANT_SCARF",
            Self::Byrdpip => "BYRDPIP",
            Self::CallingBell => "CALLING_BELL",
            Self::ChoicesParadox => "CHOICES_PARADOX",
            Self::ChosenCheese => "CHOSEN_CHEESE",
            Self::Claws => "CLAWS",
            Self::Crossbow => "CROSSBOW",
            Self::CursedPearl => "CURSED_PEARL",
            Self::DarkstonePeriapt => "DARKSTONE_PERIAPT",
            Self::DaughterOfTheWind => "DAUGHTER_OF_THE_WIND",
            Self::DelicateFrond => "DELICATE_FROND",
            Self::DiamondDiadem => "DIAMOND_DIADEM",
            Self::DistinguishedCape => "DISTINGUISHED_CAPE",
            Self::DreamCatcher => "DREAM_CATCHER",
            Self::Driftwood => "DRIFTWOOD",
            Self::DustyTome => "DUSTY_TOME",
            Self::Ectoplasm => "ECTOPLASM",
            Self::ElectricShrymp => "ELECTRIC_SHRYMP",
            Self::EmberTea => "EMBER_TEA",
            Self::EmptyCage => "EMPTY_CAGE",
            Self::FakeAnchor => "FAKE_ANCHOR",
            Self::FakeBloodVial => "FAKE_BLOOD_VIAL",
            Self::FakeHappyFlower => "FAKE_HAPPY_FLOWER",
            Self::FakeLeesWaffle => "FAKE_LEES_WAFFLE",
            Self::FakeMango => "FAKE_MANGO",
            Self::FakeMerchantsRug => "FAKE_MERCHANTS_RUG",
            Self::FakeOrichalcum => "FAKE_ORICHALCUM",
            Self::FakeSneckoEye => "FAKE_SNECKO_EYE",
            Self::FakeStrikeDummy => "FAKE_STRIKE_DUMMY",
            Self::FakeVenerableTeaSet => "FAKE_VENERABLE_TEA_SET",
            Self::Fiddle => "FIDDLE",
            Self::ForgottenSoul => "FORGOTTEN_SOUL",
            Self::FragrantMushroom => "FRAGRANT_MUSHROOM",
            Self::FresnelLens => "FRESNEL_LENS",
            Self::FurCoat => "FUR_COAT",
            Self::GlassEye => "GLASS_EYE",
            Self::Glitter => "GLITTER",
            Self::GoldenCompass => "GOLDEN_COMPASS",
            Self::GoldenPearl => "GOLDEN_PEARL",
            Self::HandDrill => "HAND_DRILL",
            Self::HistoryCourse => "HISTORY_COURSE",
            Self::IronClub => "IRON_CLUB",
            Self::JeweledMask => "JEWELED_MASK",
            Self::JewelryBox => "JEWELRY_BOX",
            Self::LargeCapsule => "LARGE_CAPSULE",
            Self::LavaRock => "LAVA_ROCK",
            Self::LeadPaperweight => "LEAD_PAPERWEIGHT",
            Self::LeafyPoultice => "LEAFY_POULTICE",
            Self::LoomingFruit => "LOOMING_FRUIT",
            Self::LordsParasol => "LORDS_PARASOL",
            Self::LostCoffer => "LOST_COFFER",
            Self::LostWisp => "LOST_WISP",
            Self::MassiveScroll => "MASSIVE_SCROLL",
            Self::MawBank => "MAW_BANK",
            Self::MeatCleaver => "MEAT_CLEAVER",
            Self::MrStruggles => "MR_STRUGGLES",
            Self::MusicBox => "MUSIC_BOX",
            Self::NeowsTorment => "NEOWS_TORMENT",
            Self::NewLeaf => "NEW_LEAF",
            Self::NutritiousOyster => "NUTRITIOUS_OYSTER",
            Self::NutritiousSoup => "NUTRITIOUS_SOUP",
            Self::PaelsBlood => "PAELS_BLOOD",
            Self::PaelsClaw => "PAELS_CLAW",
            Self::PaelsEye => "PAELS_EYE",
            Self::PaelsFlesh => "PAELS_FLESH",
            Self::PaelsGrowth => "PAELS_GROWTH",
            Self::PaelsHorn => "PAELS_HORN",
            Self::PaelsLegion => "PAELS_LEGION",
            Self::PaelsTears => "PAELS_TEARS",
            Self::PaelsTooth => "PAELS_TOOTH",
            Self::PaelsWing => "PAELS_WING",
            Self::PandorasBox => "PANDORAS_BOX",
            Self::PhilosophersStone => "PHILOSOPHERS_STONE",
            Self::PollinousCore => "POLLINOUS_CORE",
            Self::Pomander => "POMANDER",
            Self::PrecariousShears => "PRECARIOUS_SHEARS",
            Self::PreciseScissors => "PRECISE_SCISSORS",
            Self::PreservedFog => "PRESERVED_FOG",
            Self::PrismaticGem => "PRISMATIC_GEM",
            Self::PumpkinCandle => "PUMPKIN_CANDLE",
            Self::RadiantPearl => "RADIANT_PEARL",
            Self::RoyalPoison => "ROYAL_POISON",
            Self::RunicPyramid => "RUNIC_PYRAMID",
            Self::Sai => "SAI",
            Self::SandCastle => "SAND_CASTLE",
            Self::ScrollBoxes => "SCROLL_BOXES",
            Self::SeaGlass => "SEA_GLASS",
            Self::SealOfGold => "SEAL_OF_GOLD",
            Self::SereTalon => "SERE_TALON",
            Self::SignetRing => "SIGNET_RING",
            Self::SilverCrucible => "SILVER_CRUCIBLE",
            Self::SmallCapsule => "SMALL_CAPSULE",
            Self::SneckoEye => "SNECKO_EYE",
            Self::Sozu => "SOZU",
            Self::SpikedGauntlets => "SPIKED_GAUNTLETS",
            Self::StoneHumidifier => "STONE_HUMIDIFIER",
            Self::Storybook => "STORYBOOK",
            Self::SwordOfStone => "SWORD_OF_STONE",
            Self::SwordOfJade => "SWORD_OF_JADE",
            Self::TanxsWhistle => "TANXS_WHISTLE",
            Self::TeaOfDiscourtesy => "TEA_OF_DISCOURTESY",
            Self::TheBoot => "THE_BOOT",
            Self::ThrowingAxe => "THROWING_AXE",
            Self::ToastyMittens => "TOASTY_MITTENS",
            Self::TouchOfOrobas => "TOUCH_OF_OROBAS",
            Self::ToyBox => "TOY_BOX",
            Self::TriBoomerang => "TRI_BOOMERANG",
            Self::VelvetChoker => "VELVET_CHOKER",
            Self::VeryHotCocoa => "VERY_HOT_COCOA",
            Self::WarHammer => "WAR_HAMMER",
            Self::WhisperingEarring => "WHISPERING_EARRING",
            Self::WongoCustomerAppreciationBadge => "WONGO_CUSTOMER_APPRECIATION_BADGE",
            Self::WongosMysteryTicket => "WONGOS_MYSTERY_TICKET",
            Self::YummyCookie => "YUMMY_COOKIE",
            Self::SymbioticVirus => "SYMBIOTIC_VIRUS",
            Self::TwistedFunnel => "TWISTED_FUNNEL",
            Self::CloakClasp => "CLOAK_CLASP",
            Self::Circlet => "CIRCLET",
            Self::DeprecatedRelic => "DEPRECATED_RELIC",
            Self::NeowsTalisman => "NEOWS_TALISMAN",
            Self::WingedBoots => "WINGED_BOOTS",
            Self::PhialHolster => "PHIAL_HOLSTER",
            Self::NeowsBones => "NEOWS_BONES",
            Self::HeftyTablet => "HEFTY_TABLET",
            Self::FishingRod => "FISHING_ROD",
            Self::Kaleidoscope => "KALEIDOSCOPE",
            Self::SilkenTress => "SILKEN_TRESS",
            Self::DowsingRod => "DOWSING_ROD",
            Self::NeowsSacrifice => "NEOWS_SACRIFICE",
            Self::ObsidianVice => "OBSIDIAN_VICE",
            Self::Thumbscrew => "THUMBSCREW",
            Self::TormentousCollar => "TORMENTOUS_COLLAR",
            Self::MightyWings => "MIGHTY_WINGS",
            Self::BathTowel => "BATH_TOWEL",
            Self::BathRobes => "BATH_ROBES",
            Self::Wuppy => "WUPPY",
            Self::Byrdine => "BYRDINE",
        }
    }

    /// Every variant, in source order. Useful for exhaustive iteration.
    #[must_use]
    pub const fn all() -> &'static [RelicId] {
        &[
            Self::BurningBlood,
            Self::BlackBlood,
            Self::RingOfTheSnake,
            Self::RingOfTheDrake,
            Self::CrackedCore,
            Self::InfusedCore,
            Self::BoundPhylactery,
            Self::PhylacteryUnbound,
            Self::DivineRight,
            Self::DivineDestiny,
            Self::AmethystAubergine,
            Self::Anchor,
            Self::BagOfPreparation,
            Self::BloodVial,
            Self::BoneFlute,
            Self::BronzeScales,
            Self::CentennialPuzzle,
            Self::DataDisk,
            Self::FencingManual,
            Self::FestivePopper,
            Self::Gorget,
            Self::HappyFlower,
            Self::JuzuBracelet,
            Self::Lantern,
            Self::MealTicket,
            Self::OddlySmoothStone,
            Self::Pendulum,
            Self::Permafrost,
            Self::PotionBelt,
            Self::RedSkull,
            Self::RegalPillow,
            Self::SneckoSkull,
            Self::Strawberry,
            Self::StrikeDummy,
            Self::TinyMailbox,
            Self::Vajra,
            Self::VenerableTeaSet,
            Self::WarPaint,
            Self::Whetstone,
            Self::BookOfFiveRings,
            Self::Akabeko,
            Self::BagOfMarbles,
            Self::Bellows,
            Self::BookRepairKnife,
            Self::BowlerHat,
            Self::Candelabra,
            Self::EternalFeather,
            Self::FuneraryMask,
            Self::GalacticDust,
            Self::GoldPlatedCables,
            Self::GremlinHorn,
            Self::HornCleat,
            Self::JossPaper,
            Self::Kusarigama,
            Self::LetterOpener,
            Self::LuckyFysh,
            Self::MercuryHourglass,
            Self::MiniatureCannon,
            Self::Nunchaku,
            Self::Orichalcum,
            Self::OrnamentalFan,
            Self::Pantograph,
            Self::PaperPhrog,
            Self::ParryingShield,
            Self::Pear,
            Self::PenNib,
            Self::PetrifiedToad,
            Self::Planisphere,
            Self::RedMask,
            Self::Regalite,
            Self::ReptileTrinket,
            Self::RippleBasin,
            Self::SelfFormingClay,
            Self::SparklingRouge,
            Self::StoneCracker,
            Self::TuningFork,
            Self::Tingsha,
            Self::Vambrace,
            Self::ArtOfWar,
            Self::BeatingRemnant,
            Self::BigHat,
            Self::Bookmark,
            Self::CaptainsWheel,
            Self::Chandelier,
            Self::CharonsAshes,
            Self::DemonTongue,
            Self::EmotionChip,
            Self::FrozenEgg,
            Self::GamblingChip,
            Self::GamePiece,
            Self::Girya,
            Self::HelicalDart,
            Self::IceCream,
            Self::IntimidatingHelmet,
            Self::IvoryTile,
            Self::Kunai,
            Self::LastingCandy,
            Self::LizardTail,
            Self::LunarPastry,
            Self::Mango,
            Self::MeatOnTheBone,
            Self::Metronome,
            Self::MiniRegent,
            Self::MoltenEgg,
            Self::MummifiedHand,
            Self::OldCoin,
            Self::OrangeDough,
            Self::Pocketwatch,
            Self::PowerCell,
            Self::PrayerWheel,
            Self::RainbowRing,
            Self::RazorTooth,
            Self::Shovel,
            Self::Shuriken,
            Self::StoneCalendar,
            Self::SturdyClamp,
            Self::TheCourier,
            Self::ToughBandages,
            Self::ToxicEgg,
            Self::TungstenRod,
            Self::UnceasingTop,
            Self::UnsettlingLamp,
            Self::RuinedHelmet,
            Self::VexingPuzzlebox,
            Self::WhiteBeastStatue,
            Self::WhiteStar,
            Self::PaperKrane,
            Self::BeltBuckle,
            Self::Bread,
            Self::Brimstone,
            Self::BurningSticks,
            Self::Cauldron,
            Self::ChemicalX,
            Self::DingyRug,
            Self::DollysMirror,
            Self::DragonFruit,
            Self::GhostSeed,
            Self::GnarledHammer,
            Self::Kifuda,
            Self::LavaLamp,
            Self::LeesWaffle,
            Self::MembershipCard,
            Self::MiniatureTent,
            Self::MysticLighter,
            Self::NinjaScroll,
            Self::Orrery,
            Self::PunchDagger,
            Self::RingingTriangle,
            Self::RoyalStamp,
            Self::RunicCapacitor,
            Self::ScreamingFlagon,
            Self::SlingOfCourage,
            Self::TheAbacus,
            Self::Toolbox,
            Self::UndyingSigil,
            Self::VitruvianMinion,
            Self::WingCharm,
            Self::AlchemicalCoffer,
            Self::ArcaneScroll,
            Self::ArchaicTooth,
            Self::Astrolabe,
            Self::BeautifulBracelet,
            Self::BigMushroom,
            Self::BiiigHug,
            Self::BingBong,
            Self::BlackStar,
            Self::BlessedAntler,
            Self::BloodSoakedRose,
            Self::BoneTea,
            Self::BoomingConch,
            Self::BrilliantScarf,
            Self::Byrdpip,
            Self::CallingBell,
            Self::ChoicesParadox,
            Self::ChosenCheese,
            Self::Claws,
            Self::Crossbow,
            Self::CursedPearl,
            Self::DarkstonePeriapt,
            Self::DaughterOfTheWind,
            Self::DelicateFrond,
            Self::DiamondDiadem,
            Self::DistinguishedCape,
            Self::DreamCatcher,
            Self::Driftwood,
            Self::DustyTome,
            Self::Ectoplasm,
            Self::ElectricShrymp,
            Self::EmberTea,
            Self::EmptyCage,
            Self::FakeAnchor,
            Self::FakeBloodVial,
            Self::FakeHappyFlower,
            Self::FakeLeesWaffle,
            Self::FakeMango,
            Self::FakeMerchantsRug,
            Self::FakeOrichalcum,
            Self::FakeSneckoEye,
            Self::FakeStrikeDummy,
            Self::FakeVenerableTeaSet,
            Self::Fiddle,
            Self::ForgottenSoul,
            Self::FragrantMushroom,
            Self::FresnelLens,
            Self::FurCoat,
            Self::GlassEye,
            Self::Glitter,
            Self::GoldenCompass,
            Self::GoldenPearl,
            Self::HandDrill,
            Self::HistoryCourse,
            Self::IronClub,
            Self::JeweledMask,
            Self::JewelryBox,
            Self::LargeCapsule,
            Self::LavaRock,
            Self::LeadPaperweight,
            Self::LeafyPoultice,
            Self::LoomingFruit,
            Self::LordsParasol,
            Self::LostCoffer,
            Self::LostWisp,
            Self::MassiveScroll,
            Self::MawBank,
            Self::MeatCleaver,
            Self::MrStruggles,
            Self::MusicBox,
            Self::NeowsTorment,
            Self::NewLeaf,
            Self::NutritiousOyster,
            Self::NutritiousSoup,
            Self::PaelsBlood,
            Self::PaelsClaw,
            Self::PaelsEye,
            Self::PaelsFlesh,
            Self::PaelsGrowth,
            Self::PaelsHorn,
            Self::PaelsLegion,
            Self::PaelsTears,
            Self::PaelsTooth,
            Self::PaelsWing,
            Self::PandorasBox,
            Self::PhilosophersStone,
            Self::PollinousCore,
            Self::Pomander,
            Self::PrecariousShears,
            Self::PreciseScissors,
            Self::PreservedFog,
            Self::PrismaticGem,
            Self::PumpkinCandle,
            Self::RadiantPearl,
            Self::RoyalPoison,
            Self::RunicPyramid,
            Self::Sai,
            Self::SandCastle,
            Self::ScrollBoxes,
            Self::SeaGlass,
            Self::SealOfGold,
            Self::SereTalon,
            Self::SignetRing,
            Self::SilverCrucible,
            Self::SmallCapsule,
            Self::SneckoEye,
            Self::Sozu,
            Self::SpikedGauntlets,
            Self::StoneHumidifier,
            Self::Storybook,
            Self::SwordOfStone,
            Self::SwordOfJade,
            Self::TanxsWhistle,
            Self::TeaOfDiscourtesy,
            Self::TheBoot,
            Self::ThrowingAxe,
            Self::ToastyMittens,
            Self::TouchOfOrobas,
            Self::ToyBox,
            Self::TriBoomerang,
            Self::VelvetChoker,
            Self::VeryHotCocoa,
            Self::WarHammer,
            Self::WhisperingEarring,
            Self::WongoCustomerAppreciationBadge,
            Self::WongosMysteryTicket,
            Self::YummyCookie,
            Self::SymbioticVirus,
            Self::TwistedFunnel,
            Self::CloakClasp,
            Self::Circlet,
            Self::DeprecatedRelic,
            Self::NeowsTalisman,
            Self::WingedBoots,
            Self::PhialHolster,
            Self::NeowsBones,
            Self::HeftyTablet,
            Self::FishingRod,
            Self::Kaleidoscope,
            Self::SilkenTress,
        ]
    }

    /// Relics that exist only in game build v0.111.0 (GH #792), in the
    /// order the C# added them.
    pub const V0111_ONLY: [RelicId; 2] = [Self::DowsingRod, Self::NeowsSacrifice];

    /// Relics that exist only from game build v0.113.0 on.
    pub const V0113_ONLY: [RelicId; 8] = [
        Self::ObsidianVice,
        Self::Thumbscrew,
        Self::TormentousCollar,
        Self::MightyWings,
        Self::BathTowel,
        Self::BathRobes,
        Self::Wuppy,
        Self::Byrdine,
    ];

    /// The build a relic first exists in.
    #[must_use]
    pub const fn introduced_in(self) -> crate::GameVersion {
        match self {
            Self::DowsingRod | Self::NeowsSacrifice => crate::GameVersion::V0_111_0,
            Self::ObsidianVice
            | Self::Thumbscrew
            | Self::TormentousCollar
            | Self::MightyWings
            | Self::BathTowel
            | Self::BathRobes
            | Self::Wuppy
            | Self::Byrdine => crate::GameVersion::V0_113_0,
            _ => crate::GameVersion::V0_107_1,
        }
    }

    /// Every relic of game build `version`: [`Self::all`] (the v0.107.1
    /// universe, unchanged so every pre-port consumer stays bit-identical)
    /// plus, from v0.111.0 on, [`Self::V0111_ONLY`].
    #[must_use]
    pub fn all_for(version: crate::GameVersion) -> Vec<RelicId> {
        let mut v = Self::all().to_vec();
        if version >= crate::GameVersion::V0_111_0 {
            v.extend_from_slice(&Self::V0111_ONLY);
        }
        if version >= crate::GameVersion::V0_113_0 {
            v.extend_from_slice(&Self::V0113_ONLY);
        }
        v
    }
}
