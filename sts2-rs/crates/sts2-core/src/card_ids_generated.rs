// HAND-MAINTAINED against ../sts2-wiki/decompiled/v0.107.1/ — edit directly.
// The `_generated` filename is historical; there is no generator any more.
//
// Origin: bead sts2-rs-ov4 (P6-1) bulk-scraped this enum from
// `sts2-rl-agent@01c094d:sts2_env/core/enums.py::CardId` via a
// `scripts/scrape_card_ids.py` that no longer exists. The Python pin is
// archived and is NOT the source of truth for game content (see
// `PYTHON_SIM_PIN.md` and the hard rule in CLAUDE.md); the v0.107.1 C#
// decompilation is. The scraper was retired because it could only ever
// *add* variants, so every v0.107.1 rename or removal below would be
// reverted by a rerun — e.g. it re-added the removed `GRAPPLE`, and
// un-renamed `Scare` back to the deleted `FollowThrough`.
//
// The `to_python_name` arm still pins the upper-snake-case string the parity
// exporter emits. For variants that predate v0.107.1 that string equals
// `CardId.<v>.name` from the old Python source; for C#-only content it is
// simply the C# id in upper-snake-case. Keep the two in sync by hand.
//
// C#-only variants (no Python counterpart at the pin) are tagged `// MANUAL`
// at each of their three sites (enum / to_python_name / all):
//   - NotYet — C#-only Ironclad Rare.

/// Stable card identifier — one variant per distinct Python `CardId`.
///
/// Upgrade level is *not* encoded here: each base card has a single
/// `CardId`, and the `upgraded: bool` field on `Card` switches between
/// base and `+` variants. Enchantments live in `Card::enchantments`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum CardId {
    /// `CardId.STRIKE_IRONCLAD` in Python.
    StrikeIronclad,
    /// `CardId.DEFEND_IRONCLAD` in Python.
    DefendIronclad,
    /// `CardId.BASH` in Python.
    Bash,
    /// `CardId.ANGER` in Python.
    Anger,
    /// `CardId.ARMAMENTS` in Python.
    Armaments,
    /// `CardId.BLOOD_WALL` in Python.
    BloodWall,
    /// `CardId.BLOODLETTING` in Python.
    Bloodletting,
    /// `CardId.BODY_SLAM` in Python.
    BodySlam,
    /// `CardId.BREAKTHROUGH` in Python.
    Breakthrough,
    /// `CardId.CINDER` in Python.
    Cinder,
    /// `CardId.HAVOC` in Python.
    Havoc,
    /// `CardId.HEADBUTT` in Python.
    Headbutt,
    /// `CardId.IRON_WAVE` in Python.
    IronWave,
    /// `CardId.MOLTEN_FIST` in Python.
    MoltenFist,
    /// `CardId.PERFECTED_STRIKE` in Python.
    PerfectedStrike,
    /// `CardId.POMMEL_STRIKE` in Python.
    PommelStrike,
    /// `CardId.SETUP_STRIKE_CARD` in Python.
    SetupStrikeCard,
    /// `CardId.SHRUG_IT_OFF` in Python.
    ShrugItOff,
    /// `CardId.SWORD_BOOMERANG` in Python.
    SwordBoomerang,
    /// `CardId.THUNDERCLAP` in Python.
    Thunderclap,
    /// `CardId.TREMBLE` in Python.
    Tremble,
    /// `CardId.TRUE_GRIT` in Python.
    TrueGrit,
    /// `CardId.TWIN_STRIKE` in Python.
    TwinStrike,
    /// `CardId.ASHEN_STRIKE` in Python.
    AshenStrike,
    /// `CardId.BATTLE_TRANCE` in Python.
    BattleTrance,
    /// `CardId.BLUDGEON` in Python.
    Bludgeon,
    /// `CardId.BULLY` in Python.
    Bully,
    /// `CardId.BURNING_PACT` in Python.
    BurningPact,
    /// `CardId.DEMONIC_SHIELD` in Python.
    DemonicShield,
    /// `CardId.DISMANTLE` in Python.
    Dismantle,
    /// `CardId.DOMINATE` in Python.
    Dominate,
    /// `CardId.DRUM_OF_BATTLE_CARD` in Python.
    DrumOfBattleCard,
    /// `CardId.EVIL_EYE` in Python.
    EvilEye,
    /// `CardId.EXPECT_A_FIGHT` in Python.
    ExpectAFight,
    /// `CardId.FEEL_NO_PAIN_CARD` in Python.
    FeelNoPainCard,
    /// `CardId.FIGHT_ME` in Python.
    FightMe,
    /// `CardId.FLAME_BARRIER_CARD` in Python.
    FlameBarrierCard,
    /// `CardId.FORGOTTEN_RITUAL` in Python.
    ForgottenRitual,
    /// `CardId.HEMOKINESIS` in Python.
    Hemokinesis,
    /// `CardId.HOWL_FROM_BEYOND` in Python.
    HowlFromBeyond,
    /// `CardId.INFERNAL_BLADE` in Python.
    InfernalBlade,
    /// `CardId.INFERNO_CARD` in Python.
    InfernoCard,
    /// `CardId.INFLAME` in Python.
    Inflame,
    /// `CardId.JUGGLING_CARD` in Python.
    JugglingCard,
    /// `CardId.PILLAGE` in Python.
    Pillage,
    /// `CardId.RAGE_CARD` in Python.
    RageCard,
    /// `CardId.RAMPAGE` in Python.
    Rampage,
    /// `CardId.RUPTURE_CARD` in Python.
    RuptureCard,
    /// `CardId.SECOND_WIND` in Python.
    SecondWind,
    /// `CardId.SPITE` in Python.
    Spite,
    /// `CardId.STAMPEDE_CARD` in Python.
    StampedeCard,
    /// `CardId.STOMP` in Python.
    Stomp,
    /// `CardId.STONE_ARMOR` in Python.
    StoneArmor,
    /// `CardId.TAUNT` in Python.
    Taunt,
    /// `CardId.UNRELENTING` in Python.
    Unrelenting,
    /// `CardId.UPPERCUT` in Python.
    Uppercut,
    /// `CardId.VICIOUS_CARD` in Python.
    ViciousCard,
    /// `CardId.WHIRLWIND` in Python.
    Whirlwind,
    /// `CardId.AGGRESSION_CARD` in Python.
    AggressionCard,
    /// `CardId.BARRICADE_CARD` in Python.
    BarricadeCard,
    /// `CardId.BRAND` in Python.
    Brand,
    /// `CardId.CASCADE` in Python.
    Cascade,
    /// `CardId.COLOSSUS_CARD` in Python.
    ColossusCard,
    /// `CardId.CONFLAGRATION` in Python.
    Conflagration,
    /// `CardId.CRIMSON_MANTLE` in Python.
    CrimsonMantle,
    /// `CardId.CRUELTY_CARD` in Python.
    CrueltyCard,
    /// `CardId.DARK_EMBRACE_CARD` in Python.
    DarkEmbraceCard,
    /// `CardId.DEMON_FORM_CARD` in Python.
    DemonFormCard,
    /// `CardId.FEED` in Python.
    Feed,
    /// `CardId.FIEND_FIRE` in Python.
    FiendFire,
    /// `CardId.HELLRAISER_CARD` in Python.
    HellraiserCard,
    /// `CardId.IMPERVIOUS` in Python.
    Impervious,
    /// `CardId.JUGGERNAUT_CARD` in Python.
    JuggernautCard,
    /// `CardId.MANGLE` in Python.
    Mangle,
    /// `NOT_YET` — C#-only Ironclad Rare; not in the Python pin. // MANUAL
    NotYet,
    /// `CardId.OFFERING` in Python.
    Offering,
    /// `CardId.ONE_TWO_PUNCH_CARD` in Python.
    OneTwoPunchCard,
    /// `CardId.PACTS_END` in Python.
    PactsEnd,
    /// `CardId.PRIMAL_FORCE` in Python.
    PrimalForce,
    /// `CardId.PYRE` in Python.
    Pyre,
    /// `CardId.STOKE` in Python.
    Stoke,
    /// `CardId.TANK_CARD` in Python.
    TankCard,
    /// `CardId.TEAR_ASUNDER` in Python.
    TearAsunder,
    /// `CardId.THRASH` in Python.
    Thrash,
    /// `CardId.UNMOVABLE` in Python.
    Unmovable,
    /// `CardId.BREAK` in Python.
    Break,
    /// `CardId.CORRUPTION_CARD` in Python.
    CorruptionCard,
    /// `CardId.STRIKE_SILENT` in Python.
    StrikeSilent,
    /// `CardId.DEFEND_SILENT` in Python.
    DefendSilent,
    /// `CardId.NEUTRALIZE` in Python.
    Neutralize,
    /// `CardId.SURVIVOR` in Python.
    Survivor,
    /// `CardId.STRIKE_DEFECT` in Python.
    StrikeDefect,
    /// `CardId.DEFEND_DEFECT` in Python.
    DefendDefect,
    /// `CardId.ZAP` in Python.
    Zap,
    /// `CardId.DUALCAST` in Python.
    Dualcast,
    /// `CardId.STRIKE_REGENT` in Python.
    StrikeRegent,
    /// `CardId.DEFEND_REGENT` in Python.
    DefendRegent,
    /// `CardId.FALLING_STAR` in Python.
    FallingStar,
    /// `CardId.VENERATE` in Python.
    Venerate,
    /// `CardId.STRIKE_NECROBINDER` in Python.
    StrikeNecrobinder,
    /// `CardId.DEFEND_NECROBINDER` in Python.
    DefendNecrobinder,
    /// `CardId.BODYGUARD` in Python.
    Bodyguard,
    /// `CardId.UNLEASH` in Python.
    Unleash,
    /// `CardId.WOUND` in Python.
    Wound,
    /// `CardId.DAZED` in Python.
    Dazed,
    /// `CardId.SLIMED` in Python.
    Slimed,
    /// `CardId.BURN` in Python.
    Burn,
    /// `CardId.VOID` in Python.
    Void,
    /// `CardId.ASCENDERS_BANE` in Python.
    AscendersBane,
    /// `CardId.CLUMSY` in Python.
    Clumsy,
    /// `CardId.DECAY` in Python.
    Decay,
    /// `CardId.DOUBT` in Python.
    Doubt,
    /// `CardId.INJURY` in Python.
    Injury,
    /// `CardId.NORMALITY` in Python.
    Normality,
    /// `CardId.PAIN` in Python.
    Pain,
    /// `CardId.PARASITE` in Python.
    Parasite,
    /// `CardId.REGRET` in Python.
    Regret,
    /// `CardId.SHAME` in Python.
    Shame,
    /// `CardId.WRITHE` in Python.
    Writhe,
    /// `CardId.SHIV` in Python.
    Shiv,
    /// `CardId.INFECTION` in Python.
    Infection,
    /// `CardId.BECKON` in Python.
    Beckon,
    /// `CardId.ACROBATICS` in Python.
    Acrobatics,
    /// `CardId.ANTICIPATE` in Python.
    Anticipate,
    /// `CardId.BACKFLIP` in Python.
    Backflip,
    /// `CardId.BLADE_DANCE` in Python.
    BladeDance,
    /// `CardId.CLOAK_AND_DAGGER` in Python.
    CloakAndDagger,
    /// `CardId.DAGGER_SPRAY` in Python.
    DaggerSpray,
    /// `CardId.DAGGER_THROW` in Python.
    DaggerThrow,
    /// `CardId.DEADLY_POISON` in Python.
    DeadlyPoison,
    /// `CardId.DEFLECT` in Python.
    Deflect,
    /// `CardId.DODGE_AND_ROLL` in Python.
    DodgeAndRoll,
    /// `CardId.FLICK_FLACK` in Python.
    FlickFlack,
    /// `CardId.LEADING_STRIKE` in Python.
    LeadingStrike,
    /// `CardId.PIERCING_WAIL` in Python.
    PiercingWail,
    /// `CardId.POISONED_STAB` in Python.
    PoisonedStab,
    /// `CardId.PREPARED` in Python.
    Prepared,
    /// `CardId.RICOCHET` in Python.
    Ricochet,
    /// `CardId.SLICE` in Python.
    Slice,
    /// `CardId.SNAKEBITE` in Python.
    Snakebite,
    /// `CardId.SUCKER_PUNCH` in Python.
    SuckerPunch,
    /// `CardId.UNTOUCHABLE` in Python.
    Untouchable,
    /// `CardId.ACCURACY_CARD` in Python.
    AccuracyCard,
    /// `CardId.BACKSTAB` in Python.
    Backstab,
    /// `CardId.BLUR_CARD` in Python.
    BlurCard,
    /// `CardId.BOUNCING_FLASK` in Python.
    BouncingFlask,
    /// `CardId.BUBBLE_BUBBLE` in Python.
    BubbleBubble,
    /// `CardId.CALCULATED_GAMBLE` in Python.
    CalculatedGamble,
    /// `CardId.DASH` in Python.
    Dash,
    /// `CardId.ESCAPE_PLAN` in Python.
    EscapePlan,
    /// `CardId.EXPERTISE` in Python.
    Expertise,
    /// `CardId.EXPOSE` in Python.
    Expose,
    /// `CardId.FINISHER` in Python.
    Finisher,
    /// `CardId.FLANKING` in Python.
    Flanking,
    /// `CardId.FLECHETTES` in Python.
    Flechettes,
    /// `CardId.SCARE` in Python. v0.107.1 rename+rework of the removed
    /// `FollowThrough` (its `FollowThrough.cs` decomp was deleted) — Scare
    /// occupies the same Silent pool slot. See `Cards/Scare.cs`.
    Scare,
    /// `CardId.FOOTWORK` in Python.
    Footwork,
    /// `CardId.HAND_TRICK` in Python.
    HandTrick,
    /// `CardId.HAZE` in Python.
    Haze,
    /// `CardId.HIDDEN_DAGGERS` in Python.
    HiddenDaggers,
    /// `CardId.INFINITE_BLADES_CARD` in Python.
    InfiniteBladesCard,
    /// `CardId.LEG_SWEEP` in Python.
    LegSweep,
    /// `CardId.MEMENTO_MORI` in Python.
    MementoMori,
    /// `CardId.MIRAGE` in Python.
    Mirage,
    /// `CardId.NOXIOUS_FUMES_CARD` in Python.
    NoxiousFumesCard,
    /// `CardId.OUTBREAK_CARD` in Python.
    OutbreakCard,
    /// `CardId.PHANTOM_BLADES_CARD` in Python.
    PhantomBladesCard,
    /// `CardId.PINPOINT` in Python.
    Pinpoint,
    /// `CardId.POUNCE` in Python.
    Pounce,
    /// `CardId.PRECISE_CUT` in Python.
    PreciseCut,
    /// `CardId.PREDATOR` in Python.
    Predator,
    /// `CardId.REFLEX` in Python.
    Reflex,
    /// `CardId.SKEWER` in Python.
    Skewer,
    /// `CardId.SPEEDSTER_CARD` in Python.
    SpeedsterCard,
    /// `CardId.STRANGLE` in Python.
    Strangle,
    /// `CardId.TACTICIAN` in Python.
    Tactician,
    /// `CardId.UP_MY_SLEEVE` in Python.
    UpMySleeve,
    /// `CardId.WELL_LAID_PLANS` in Python.
    WellLaidPlans,
    /// `CardId.ABRASIVE` in Python.
    Abrasive,
    /// `CardId.ACCELERANT` in Python.
    Accelerant,
    /// `CardId.ADRENALINE` in Python.
    Adrenaline,
    /// `CardId.AFTERIMAGE_CARD` in Python.
    AfterimageCard,
    /// `CardId.ASSASSINATE` in Python.
    Assassinate,
    /// `CardId.BLADE_OF_INK` in Python.
    BladeOfInk,
    /// `CardId.BULLET_TIME` in Python.
    BulletTime,
    /// `CardId.BURST` in Python.
    Burst,
    /// `CardId.CORROSIVE_WAVE` in Python.
    CorrosiveWave,
    /// `CardId.ECHOING_SLASH` in Python.
    EchoingSlash,
    /// `CardId.ENVENOM_CARD` in Python.
    EnvenomCard,
    /// `CardId.FAN_OF_KNIVES_CARD` in Python.
    FanOfKnivesCard,
    /// `CardId.GRAND_FINALE` in Python.
    GrandFinale,
    /// `CardId.KNIFE_TRAP` in Python.
    KnifeTrap,
    /// `CardId.MALAISE` in Python.
    Malaise,
    /// `CardId.MASTER_PLANNER` in Python.
    MasterPlanner,
    /// `CardId.MURDER` in Python.
    Murder,
    /// `CardId.NIGHTMARE` in Python.
    Nightmare,
    /// `CardId.SERPENT_FORM_CARD` in Python.
    SerpentFormCard,
    /// `CardId.SHADOW_STEP` in Python.
    ShadowStep,
    /// `CardId.SHADOWMELD` in Python.
    Shadowmeld,
    /// `CardId.SNEAKY_CARD` in Python.
    SneakyCard,
    /// `CardId.STORM_OF_STEEL` in Python.
    StormOfSteel,
    /// `CardId.THE_HUNT` in Python.
    TheHunt,
    /// `CardId.TOOLS_OF_THE_TRADE` in Python.
    ToolsOfTheTrade,
    /// `CardId.TRACKING` in Python.
    Tracking,
    /// `CardId.SUPPRESS` in Python.
    Suppress,
    /// `CardId.WRAITH_FORM` in Python.
    WraithForm,
    /// `CardId.BALL_LIGHTNING` in Python.
    BallLightning,
    /// `CardId.BARRAGE` in Python.
    Barrage,
    /// `CardId.BEAM_CELL` in Python.
    BeamCell,
    /// `CardId.BOOST_AWAY` in Python.
    BoostAway,
    /// `CardId.CHARGE_BATTERY` in Python.
    ChargeBattery,
    /// `CardId.CLAW` in Python.
    Claw,
    /// `CardId.COLD_SNAP` in Python.
    ColdSnap,
    /// `CardId.COMPILE_DRIVER` in Python.
    CompileDriver,
    /// `CardId.COOLHEADED` in Python.
    Coolheaded,
    /// `CardId.FOCUSED_STRIKE_CARD` in Python.
    FocusedStrikeCard,
    /// `CardId.GO_FOR_THE_EYES` in Python.
    GoForTheEyes,
    /// `CardId.GUNK_UP` in Python.
    GunkUp,
    /// `CardId.HOLOGRAM` in Python.
    Hologram,
    /// `CardId.HOTFIX` in Python.
    Hotfix,
    /// `CardId.LEAP` in Python.
    Leap,
    /// `CardId.LIGHTNING_ROD` in Python.
    LightningRod,
    /// `CardId.MOMENTUM_STRIKE` in Python.
    MomentumStrike,
    /// `CardId.SWEEPING_BEAM` in Python.
    SweepingBeam,
    /// `CardId.TURBO` in Python.
    Turbo,
    /// `CardId.UPROAR` in Python.
    Uproar,
    /// `CardId.BOOT_SEQUENCE` in Python.
    BootSequence,
    /// `CardId.BULK_UP` in Python.
    BulkUp,
    /// `CardId.CAPACITOR` in Python.
    Capacitor,
    /// `CardId.CHAOS` in Python.
    Chaos,
    /// `CardId.CHILL` in Python.
    Chill,
    /// `CardId.COMPACT` in Python.
    Compact,
    /// `CardId.DARKNESS_CARD` in Python.
    DarknessCard,
    /// `CardId.DOUBLE_ENERGY` in Python.
    DoubleEnergy,
    /// `CardId.ENERGY_SURGE` in Python.
    EnergySurge,
    /// `CardId.FERAL` in Python.
    Feral,
    /// `CardId.FIGHT_THROUGH` in Python.
    FightThrough,
    /// `CardId.FTL` in Python.
    Ftl,
    /// `CardId.FUSION` in Python.
    Fusion,
    /// `CardId.GLACIER` in Python.
    Glacier,
    /// `CardId.GLASSWORK` in Python.
    Glasswork,
    /// `CardId.HAILSTORM` in Python.
    Hailstorm,
    /// `CardId.ITERATION_CARD` in Python.
    IterationCard,
    /// `CardId.LOOP_CARD` in Python.
    LoopCard,
    /// `CardId.NULL_CARD` in Python.
    NullCard,
    /// `CardId.OVERCLOCK` in Python.
    Overclock,
    /// `CardId.REFRACT` in Python.
    Refract,
    /// `CardId.ROCKET_PUNCH` in Python.
    RocketPunch,
    /// `CardId.SCAVENGE` in Python.
    Scavenge,
    /// `CardId.SCRAPE` in Python.
    Scrape,
    /// `CardId.SHADOW_SHIELD` in Python.
    ShadowShield,
    /// `CardId.SKIM` in Python.
    Skim,
    /// `CardId.SMOKESTACK` in Python.
    Smokestack,
    /// `CardId.STORM_CARD` in Python.
    StormCard,
    /// `CardId.SUBROUTINE` in Python.
    Subroutine,
    /// `CardId.SUNDER` in Python.
    Sunder,
    /// `CardId.SYNCHRONIZE` in Python.
    Synchronize,
    /// `CardId.SYNTHESIS` in Python.
    Synthesis,
    /// `CardId.TEMPEST` in Python.
    Tempest,
    /// `CardId.TESLA_COIL` in Python.
    TeslaCoil,
    /// `CardId.THUNDER_CARD` in Python.
    ThunderCard,
    /// `CardId.WHITE_NOISE` in Python.
    WhiteNoise,
    /// `CardId.ADAPTIVE_STRIKE` in Python.
    AdaptiveStrike,
    /// `CardId.ALL_FOR_ONE` in Python.
    AllForOne,
    /// `CardId.BUFFER_CARD` in Python.
    BufferCard,
    /// `CardId.CONSUMING_SHADOW` in Python.
    ConsumingShadow,
    /// `CardId.COOLANT` in Python.
    Coolant,
    /// `CardId.CREATIVE_AI_CARD` in Python.
    CreativeAiCard,
    /// `CardId.DEFRAGMENT` in Python.
    Defragment,
    /// `CardId.ECHO_FORM_CARD` in Python.
    EchoFormCard,
    /// `CardId.FLAK_CANNON` in Python.
    FlakCannon,
    /// `CardId.GENETIC_ALGORITHM` in Python.
    GeneticAlgorithm,
    /// `CardId.HELIX_DRILL` in Python.
    HelixDrill,
    /// `CardId.HYPERBEAM` in Python.
    Hyperbeam,
    /// `CardId.ICE_LANCE` in Python.
    IceLance,
    /// `CardId.IGNITION` in Python.
    Ignition,
    /// `CardId.MACHINE_LEARNING_CARD` in Python.
    MachineLearningCard,
    /// `CardId.METEOR_STRIKE` in Python.
    MeteorStrike,
    /// `CardId.MODDED` in Python.
    Modded,
    /// `CardId.MULTI_CAST` in Python.
    MultiCast,
    /// `CardId.RAINBOW` in Python.
    Rainbow,
    /// `CardId.REBOOT` in Python.
    Reboot,
    /// `CardId.SHATTER` in Python.
    Shatter,
    /// `CardId.SIGNAL_BOOST` in Python.
    SignalBoost,
    /// `CardId.SPINNER_CARD` in Python.
    SpinnerCard,
    /// `CardId.SUPERCRITICAL` in Python.
    Supercritical,
    /// `CardId.TRASH_TO_TREASURE` in Python.
    TrashToTreasure,
    /// `CardId.VOLTAIC` in Python.
    Voltaic,
    /// `CardId.BIASED_COGNITION_CARD` in Python.
    BiasedCognitionCard,
    /// `CardId.QUADCAST` in Python.
    Quadcast,
    /// `CardId.AFTERLIFE` in Python.
    Afterlife,
    /// `CardId.BLIGHT_STRIKE` in Python.
    BlightStrike,
    /// `CardId.DEFILE` in Python.
    Defile,
    /// `CardId.DEFY` in Python.
    Defy,
    /// `CardId.DRAIN_POWER` in Python.
    DrainPower,
    /// `CardId.FEAR` in Python.
    Fear,
    /// `CardId.FLATTEN` in Python.
    Flatten,
    /// `CardId.GRAVE_WARDEN` in Python.
    GraveWarden,
    /// `CardId.GRAVEBLAST` in Python.
    Graveblast,
    /// `CardId.INVOKE` in Python.
    Invoke,
    /// `CardId.NEGATIVE_PULSE` in Python.
    NegativePulse,
    /// `CardId.POKE` in Python.
    Poke,
    /// `CardId.PULL_AGGRO` in Python.
    PullAggro,
    /// `CardId.REAP` in Python.
    Reap,
    /// `CardId.REAVE` in Python.
    Reave,
    /// `CardId.SCOURGE` in Python.
    Scourge,
    /// `CardId.SCULPTING_STRIKE` in Python.
    SculptingStrike,
    /// `CardId.SNAP` in Python.
    Snap,
    /// `CardId.SOW` in Python.
    Sow,
    /// `CardId.WISP` in Python.
    Wisp,
    /// `CardId.BONE_SHARDS` in Python.
    BoneShards,
    /// `CardId.BORROWED_TIME` in Python.
    BorrowedTime,
    /// `CardId.BURY` in Python.
    Bury,
    /// `CardId.CALCIFY_CARD` in Python.
    CalcifyCard,
    /// `CardId.CAPTURE_SPIRIT` in Python.
    CaptureSpirit,
    /// `CardId.CLEANSE` in Python.
    Cleanse,
    /// `CardId.COUNTDOWN_CARD` in Python.
    CountdownCard,
    /// `CardId.DANSE_MACABRE` in Python.
    DanseMacabre,
    /// `CardId.DEATH_MARCH` in Python.
    DeathMarch,
    /// `CardId.DEATHBRINGER` in Python.
    Deathbringer,
    /// `CardId.DEATHS_DOOR` in Python.
    DeathsDoor,
    /// `CardId.DEBILITATE_CARD` in Python.
    DebilitateCard,
    /// `CardId.DELAY` in Python.
    Delay,
    /// `CardId.DIRGE` in Python.
    Dirge,
    /// `CardId.DREDGE` in Python.
    Dredge,
    /// `CardId.ENFEEBLING_TOUCH` in Python.
    EnfeeblingTouch,
    /// `CardId.FETCH` in Python.
    Fetch,
    /// `CardId.FRIENDSHIP` in Python.
    Friendship,
    /// `CardId.HAUNT` in Python.
    Haunt,
    /// `CardId.HIGH_FIVE` in Python.
    HighFive,
    /// `CardId.LEGION_OF_BONE` in Python.
    LegionOfBone,
    /// `CardId.LETHALITY_CARD` in Python.
    LethalityCard,
    /// `CardId.MELANCHOLY` in Python.
    Melancholy,
    /// `CardId.NO_ESCAPE` in Python.
    NoEscape,
    /// `CardId.PAGESTORM` in Python.
    Pagestorm,
    /// `CardId.PARSE` in Python.
    Parse,
    /// `CardId.PULL_FROM_BELOW` in Python.
    PullFromBelow,
    /// `CardId.PUTREFY` in Python.
    Putrefy,
    /// `CardId.RATTLE` in Python.
    Rattle,
    /// `CardId.RIGHT_HAND_HAND` in Python.
    RightHandHand,
    /// `CardId.SEVERANCE` in Python.
    Severance,
    /// `CardId.SHROUD` in Python.
    Shroud,
    /// `CardId.SIC_EM` in Python.
    SicEm,
    /// `CardId.SLEIGHT_OF_FLESH` in Python.
    SleightOfFlesh,
    /// `CardId.SPUR` in Python.
    Spur,
    /// `CardId.VEILPIERCER` in Python.
    Veilpiercer,
    /// `CardId.BANSHEES_CRY` in Python.
    BansheesCry,
    /// `CardId.CALL_OF_THE_VOID` in Python.
    CallOfTheVoid,
    /// `CardId.DEMESNE` in Python.
    Demesne,
    /// `CardId.DEVOUR_LIFE_CARD` in Python.
    DevourLifeCard,
    /// `CardId.EIDOLON` in Python.
    Eidolon,
    /// `CardId.END_OF_DAYS` in Python.
    EndOfDays,
    /// `CardId.ERADICATE` in Python.
    Eradicate,
    /// `CardId.GLIMPSE_BEYOND` in Python.
    GlimpseBeyond,
    /// `CardId.HANG` in Python.
    Hang,
    /// `CardId.MISERY` in Python.
    Misery,
    /// `CardId.NECRO_MASTERY_CARD` in Python.
    NecroMasteryCard,
    /// `CardId.NEUROSURGE` in Python.
    Neurosurge,
    /// `CardId.OBLIVION` in Python.
    Oblivion,
    /// `CardId.REANIMATE` in Python.
    Reanimate,
    /// `CardId.REAPER_FORM` in Python.
    ReaperForm,
    /// `CardId.SACRIFICE` in Python.
    Sacrifice,
    /// `CardId.SEANCE` in Python.
    Seance,
    /// `CardId.SENTRY_MODE` in Python.
    SentryMode,
    /// `CardId.SHARED_FATE` in Python.
    SharedFate,
    /// `CardId.SOUL_STORM` in Python.
    SoulStorm,
    /// `CardId.SPIRIT_OF_ASH` in Python.
    SpiritOfAsh,
    /// `CardId.SQUEEZE` in Python.
    Squeeze,
    /// `CardId.THE_SCYTHE` in Python.
    TheScythe,
    /// `CardId.TIMES_UP` in Python.
    TimesUp,
    /// `CardId.TRANSFIGURE` in Python.
    Transfigure,
    /// `CardId.UNDEATH` in Python.
    Undeath,
    /// `CardId.FORBIDDEN_GRIMOIRE` in Python.
    ForbiddenGrimoire,
    /// `CardId.PROTECTOR` in Python.
    Protector,
    /// `CardId.ASTRAL_PULSE` in Python.
    AstralPulse,
    /// `CardId.BEGONE` in Python.
    Begone,
    /// `CardId.CELESTIAL_MIGHT` in Python.
    CelestialMight,
    /// `CardId.CLOAK_OF_STARS` in Python.
    CloakOfStars,
    /// `CardId.COLLISION_COURSE` in Python.
    CollisionCourse,
    /// `CardId.COSMIC_INDIFFERENCE` in Python.
    CosmicIndifference,
    /// `CardId.CRESCENT_SPEAR` in Python.
    CrescentSpear,
    /// `CardId.CRUSH_UNDER` in Python.
    CrushUnder,
    /// `CardId.GATHER_LIGHT` in Python.
    GatherLight,
    /// `CardId.GLITTERSTREAM` in Python.
    Glitterstream,
    /// `CardId.GLOW` in Python.
    Glow,
    /// `CardId.GUIDING_STAR` in Python.
    GuidingStar,
    /// `CardId.HIDDEN_CACHE` in Python.
    HiddenCache,
    /// `CardId.KNOW_THY_PLACE` in Python.
    KnowThyPlace,
    /// `CardId.PATTER` in Python.
    Patter,
    /// `CardId.PHOTON_CUT` in Python.
    PhotonCut,
    /// `CardId.REFINE_BLADE` in Python.
    RefineBlade,
    /// `CardId.SOLAR_STRIKE` in Python.
    SolarStrike,
    /// `CardId.SPOILS_OF_BATTLE` in Python.
    SpoilsOfBattle,
    /// `CardId.WROUGHT_IN_WAR` in Python.
    WroughtInWar,
    /// `CardId.ALIGNMENT` in Python.
    Alignment,
    /// `CardId.BLACK_HOLE` in Python.
    BlackHole,
    /// `CardId.BULWARK` in Python.
    Bulwark,
    /// `CardId.CHARGE` in Python.
    Charge,
    /// `CardId.CHILD_OF_THE_STARS` in Python.
    ChildOfTheStars,
    /// `CardId.CONQUEROR` in Python.
    Conqueror,
    /// `CardId.CONVERGENCE` in Python.
    Convergence,
    /// `CardId.DEVASTATE` in Python.
    Devastate,
    /// `CardId.FURNACE` in Python.
    Furnace,
    /// `CardId.GAMMA_BLAST` in Python.
    GammaBlast,
    /// `CardId.GLIMMER` in Python.
    Glimmer,
    /// `CardId.HEGEMONY` in Python.
    Hegemony,
    /// `CardId.KINGLY_KICK` in Python.
    KinglyKick,
    /// `CardId.KINGLY_PUNCH` in Python.
    KinglyPunch,
    /// `CardId.KNOCKOUT_BLOW` in Python.
    KnockoutBlow,
    /// `CardId.LARGESSE` in Python.
    Largesse,
    /// `CardId.LUNAR_BLAST` in Python.
    LunarBlast,
    /// `CardId.MANIFEST_AUTHORITY` in Python.
    ManifestAuthority,
    /// `CardId.MONOLOGUE_CARD` in Python.
    MonologueCard,
    /// `CardId.ORBIT` in Python.
    Orbit,
    /// `CardId.PALE_BLUE_DOT` in Python.
    PaleBlueDot,
    /// `CardId.PARRY_CARD` in Python.
    ParryCard,
    /// `CardId.PARTICLE_WALL` in Python.
    ParticleWall,
    /// `CardId.PILLAR_OF_CREATION` in Python.
    PillarOfCreation,
    /// `CardId.PROPHESIZE` in Python.
    Prophesize,
    /// `CardId.QUASAR` in Python.
    Quasar,
    /// `CardId.RADIATE` in Python.
    Radiate,
    /// `CardId.REFLECT_CARD` in Python.
    ReflectCard,
    /// `CardId.RESONANCE` in Python.
    Resonance,
    /// `CardId.ROYAL_GAMBLE` in Python.
    RoyalGamble,
    /// `CardId.SHINING_STRIKE` in Python.
    ShiningStrike,
    /// `CardId.SPECTRUM_SHIFT` in Python.
    SpectrumShift,
    /// `CardId.STARDUST` in Python.
    Stardust,
    /// `CardId.SUMMON_FORTH` in Python.
    SummonForth,
    /// `CardId.SUPERMASSIVE` in Python.
    Supermassive,
    /// `CardId.TERRAFORMING` in Python.
    Terraforming,
    /// `CardId.ARSENAL` in Python.
    Arsenal,
    /// `CardId.BEAT_INTO_SHAPE` in Python.
    BeatIntoShape,
    /// `CardId.BIG_BANG` in Python.
    BigBang,
    /// `CardId.BOMBARDMENT` in Python.
    Bombardment,
    /// `CardId.BUNDLE_OF_JOY` in Python.
    BundleOfJoy,
    /// `CardId.COMET` in Python.
    Comet,
    /// `CardId.CRASH_LANDING` in Python.
    CrashLanding,
    /// `CardId.DECISIONS_DECISIONS` in Python.
    DecisionsDecisions,
    /// `CardId.DYING_STAR` in Python.
    DyingStar,
    /// `CardId.FOREGONE_CONCLUSION` in Python.
    ForegoneConclusion,
    /// `CardId.GENESIS` in Python.
    Genesis,
    /// `CardId.GUARDS` in Python.
    Guards,
    /// `CardId.HAMMER_TIME` in Python.
    HammerTime,
    /// `CardId.HEAVENLY_DRILL` in Python.
    HeavenlyDrill,
    /// `CardId.HEIRLOOM_HAMMER` in Python.
    HeirloomHammer,
    /// `CardId.I_AM_INVINCIBLE` in Python.
    IAmInvincible,
    /// `CardId.MAKE_IT_SO` in Python.
    MakeItSo,
    /// `CardId.MONARCHS_GAZE_CARD` in Python.
    MonarchsGazeCard,
    /// `CardId.NEUTRON_AEGIS` in Python.
    NeutronAegis,
    /// `CardId.ROYALTIES_CARD` in Python.
    RoyaltiesCard,
    /// `CardId.SEEKING_EDGE` in Python.
    SeekingEdge,
    /// `CardId.SEVEN_STARS` in Python.
    SevenStars,
    /// `CardId.SWORD_SAGE` in Python.
    SwordSage,
    /// `CardId.THE_SMITH` in Python.
    TheSmith,
    /// `CardId.TYRANNY_CARD` in Python.
    TyrannyCard,
    /// `CardId.VOID_FORM` in Python.
    VoidForm,
    /// `CardId.METEOR_SHOWER` in Python.
    MeteorShower,
    /// `CardId.THE_SEALED_THRONE` in Python.
    TheSealedThrone,
    /// `CardId.AUTOMATION` in Python.
    Automation,
    /// `CardId.BELIEVE_IN_YOU` in Python.
    BelieveInYou,
    /// `CardId.CATASTROPHE` in Python.
    Catastrophe,
    /// `CardId.COORDINATE_CARD` in Python.
    CoordinateCard,
    /// `CardId.DARK_SHACKLES` in Python.
    DarkShackles,
    /// `CardId.DISCOVERY` in Python.
    Discovery,
    /// `CardId.DRAMATIC_ENTRANCE` in Python.
    DramaticEntrance,
    /// `CardId.EQUILIBRIUM` in Python.
    Equilibrium,
    /// `CardId.FASTEN` in Python.
    Fasten,
    /// `CardId.FINESSE` in Python.
    Finesse,
    /// `CardId.FISTICUFFS` in Python.
    Fisticuffs,
    /// `CardId.FLASH_OF_STEEL` in Python.
    FlashOfSteel,
    /// `CardId.GANG_UP` in Python.
    GangUp,
    /// `CardId.HUDDLE_UP` in Python.
    HuddleUp,
    /// `CardId.IMPATIENCE` in Python.
    Impatience,
    /// `CardId.INTERCEPT_CARD` in Python.
    InterceptCard,
    /// `CardId.JACK_OF_ALL_TRADES` in Python.
    JackOfAllTrades,
    /// `CardId.LIFT` in Python.
    Lift,
    /// `CardId.MIND_BLAST` in Python.
    MindBlast,
    /// `CardId.OMNISLICE` in Python.
    Omnislice,
    /// `CardId.PANACHE_CARD` in Python.
    PanacheCard,
    /// `CardId.PANIC_BUTTON` in Python.
    PanicButton,
    /// `CardId.PREP_TIME` in Python.
    PrepTime,
    /// `CardId.PRODUCTION` in Python.
    Production,
    /// `CardId.PROLONG` in Python.
    Prolong,
    /// `CardId.PROWESS` in Python.
    Prowess,
    /// `CardId.PURITY` in Python.
    Purity,
    /// `CardId.RESTLESSNESS` in Python.
    Restlessness,
    /// `CardId.SEEKER_STRIKE` in Python.
    SeekerStrike,
    /// `CardId.SHOCKWAVE` in Python.
    Shockwave,
    /// `CardId.SPLASH` in Python.
    Splash,
    /// `CardId.STRATAGEM` in Python.
    Stratagem,
    /// `CardId.TAG_TEAM` in Python.
    TagTeam,
    /// `CardId.THE_BOMB_CARD` in Python.
    TheBombCard,
    /// `CardId.THINKING_AHEAD` in Python.
    ThinkingAhead,
    /// `CardId.THRUMMING_HATCHET` in Python.
    ThrummingHatchet,
    /// `CardId.ULTIMATE_DEFEND` in Python.
    UltimateDefend,
    /// `CardId.ULTIMATE_STRIKE` in Python.
    UltimateStrike,
    /// `CardId.VOLLEY` in Python.
    Volley,
    /// `CardId.ALCHEMIZE` in Python.
    Alchemize,
    /// `CardId.ANOINTED` in Python.
    Anointed,
    /// `CardId.BEACON_OF_HOPE` in Python.
    BeaconOfHope,
    /// `CardId.BEAT_DOWN` in Python.
    BeatDown,
    /// `CardId.BOLAS` in Python.
    Bolas,
    /// `CardId.CALAMITY_CARD` in Python.
    CalamityCard,
    /// `CardId.ENTROPY` in Python.
    Entropy,
    /// `CardId.ETERNAL_ARMOR` in Python.
    EternalArmor,
    /// `CardId.GOLD_AXE` in Python.
    GoldAxe,
    /// `CardId.HAND_OF_GREED` in Python.
    HandOfGreed,
    /// `CardId.HIDDEN_GEM` in Python.
    HiddenGem,
    /// `CardId.JACKPOT` in Python.
    Jackpot,
    /// `CardId.KNOCKDOWN` in Python.
    Knockdown,
    /// `CardId.MASTER_OF_STRATEGY` in Python.
    MasterOfStrategy,
    /// `CardId.MAYHEM_CARD` in Python.
    MayhemCard,
    /// `CardId.MIMIC` in Python.
    Mimic,
    /// `CardId.NOSTALGIA_CARD` in Python.
    NostalgiaCard,
    /// `CardId.RALLY` in Python.
    Rally,
    /// `CardId.REND` in Python.
    Rend,
    /// `CardId.ROLLING_BOULDER` in Python.
    RollingBoulder,
    /// `CardId.SALVO` in Python.
    Salvo,
    /// `CardId.SCRAWL` in Python.
    Scrawl,
    /// `CardId.SECRET_TECHNIQUE` in Python.
    SecretTechnique,
    /// `CardId.SECRET_WEAPON` in Python.
    SecretWeapon,
    /// `CardId.THE_GAMBIT` in Python.
    TheGambit,
    /// `CardId.RIP_AND_TEAR` in Python.
    RipAndTear,
    /// `CardId.APOTHEOSIS` in Python.
    Apotheosis,
    /// `CardId.APPARITION` in Python.
    Apparition,
    /// `CardId.BRIGHTEST_FLAME` in Python.
    BrightestFlame,
    /// `CardId.MAUL` in Python.
    Maul,
    /// `CardId.NEOWS_FURY` in Python.
    NeowsFury,
    /// `CardId.RELAX` in Python.
    Relax,
    /// `CardId.WHISTLE` in Python.
    Whistle,
    /// `CardId.WISH` in Python.
    Wish,
    /// `CardId.BYRD_SWOOP` in Python.
    ByrdSwoop,
    /// `CardId.CALTROPS` in Python.
    Caltrops,
    /// `CardId.CLASH` in Python.
    Clash,
    /// `CardId.DISTRACTION` in Python.
    Distraction,
    /// `CardId.DUAL_WIELD` in Python.
    DualWield,
    /// `CardId.ENLIGHTENMENT` in Python.
    Enlightenment,
    /// `CardId.ENTRENCH` in Python.
    Entrench,
    /// `CardId.EXTERMINATE` in Python.
    Exterminate,
    /// `CardId.FEEDING_FRENZY_CARD` in Python.
    FeedingFrenzyCard,
    /// `CardId.HELLO_WORLD_CARD` in Python.
    HelloWorldCard,
    /// `CardId.MAD_SCIENCE` in Python.
    MadScience,
    /// `CardId.METAMORPHOSIS` in Python.
    Metamorphosis,
    /// `CardId.OUTMANEUVER` in Python.
    Outmaneuver,
    /// `CardId.PECK` in Python.
    Peck,
    /// `CardId.REBOUND` in Python.
    Rebound,
    /// `CardId.SQUASH` in Python.
    Squash,
    /// `CardId.STACK` in Python.
    Stack,
    /// `CardId.TORIC_TOUGHNESS` in Python.
    ToricToughness,
    /// `CardId.FUEL` in Python.
    Fuel,
    /// `CardId.GIANT_ROCK` in Python.
    GiantRock,
    /// `CardId.LUMINESCE` in Python.
    Luminesce,
    /// `CardId.MINION_DIVE_BOMB` in Python.
    MinionDiveBomb,
    /// `CardId.MINION_SACRIFICE` in Python.
    MinionSacrifice,
    /// `CardId.MINION_STRIKE` in Python.
    MinionStrike,
    /// `CardId.SOUL` in Python.
    Soul,
    /// `CardId.SOVEREIGN_BLADE` in Python.
    SovereignBlade,
    /// `CardId.SWEEPING_GAZE` in Python.
    SweepingGaze,
    /// `CardId.DISINTEGRATION` in Python.
    Disintegration,
    /// `CardId.MIND_ROT` in Python.
    MindRot,
    /// `CardId.SLOTH_STATUS` in Python.
    SlothStatus,
    /// `CardId.WASTE_AWAY` in Python.
    WasteAway,
    /// `CardId.DEBRIS` in Python.
    Debris,
    /// `CardId.FRANTIC_ESCAPE` in Python.
    FranticEscape,
    /// `CardId.SOOT` in Python.
    Soot,
    /// `CardId.TOXIC` in Python.
    Toxic,
    /// `CardId.BAD_LUCK` in Python.
    BadLuck,
    /// `CardId.CURSE_OF_THE_BELL` in Python.
    CurseOfTheBell,
    /// `CardId.DEBT` in Python.
    Debt,
    /// `CardId.ENTHRALLED` in Python.
    Enthralled,
    /// `CardId.FOLLY` in Python.
    Folly,
    /// `CardId.GREED` in Python.
    Greed,
    /// `CardId.GUILTY` in Python.
    Guilty,
    /// `CardId.POOR_SLEEP` in Python.
    PoorSleep,
    /// `CardId.SPORE_MIND` in Python.
    SporeMind,
    /// `CardId.BYRDONIS_EGG` in Python.
    ByrdonisEgg,
    /// `CardId.LANTERN_KEY` in Python.
    LanternKey,
    /// `CardId.SPOILS_MAP` in Python.
    SpoilsMap,
    /// `CardId.WITHER` in Python. v0.107.1 status card (`Cards/Wither.cs`):
    /// Unplayable, deals 3 self-damage at turn-end-in-hand; scales via a
    /// per-instance FakeUpgrade (Aeonglass syncs it).
    Wither,
    /// `CardId.GENERIC` in Python.
    Generic,
    // ── v0.111.0-only (GH #792). Not in [`CardId::all`], which stays the
    // v0.107.1 universe (ranker card table, PPO vocab, random samplers read
    // it); see [`CardId::V0111_ONLY`] / [`CardId::all_for`].
    /// v0.111.0 Quest card (`Cards/Dowsing.cs`), granted by the DowsingRod
    /// Neow relic. Unplayable; transforms into Abundance after 5 `?` rooms.
    Dowsing,
    /// v0.111.0 Ancient Skill (`Cards/Abundance.cs`): the Dowsing quest reward. GH #793.
    Abundance,
    /// v0.111.0 MP-only Silent Skill (`Cards/BladeSymphony.cs`). GH #793.
    BladeSymphony,
    /// v0.111.0 MP-only Ironclad Skill (`Cards/Blaze.cs`). GH #793.
    Blaze,
    /// v0.111.0 MP-only Necrobinder Power (`Cards/Cacophony.cs`); `_CARD` disambiguates it from `CacophonyPower`. GH #793.
    CacophonyCard,
    /// v0.111.0 MP-only Silent Skill (`Cards/Concoct.cs`); `_CARD` disambiguates it from `ConcoctPower`. GH #793.
    ConcoctCard,
    /// v0.111.0 MP-only Regent Skill (`Cards/Constellation.cs`). GH #793.
    Constellation,
    /// v0.111.0 MP-only Silent Skill (`Cards/Fade.cs`); `_CARD` disambiguates it from `FadePower`. GH #793.
    FadeCard,
    /// v0.111.0 MP-only Defect Skill (`Cards/Hibernate.cs`); `_CARD` disambiguates it from `HibernatePower`. GH #793.
    HibernateCard,
    /// v0.111.0 MP-only Defect Skill (`Cards/ImitationLearning.cs`); `_CARD` disambiguates it from `ImitationLearningPower`. GH #793.
    ImitationLearningCard,
    /// v0.111.0 MP-only Ironclad Attack (`Cards/Midnight.cs`). GH #793.
    Midnight,
    /// v0.111.0 MP-only Defect Power (`Cards/OneForAll.cs`); `_CARD` disambiguates it from `OneForAllPower`. GH #793.
    OneForAllCard,
    /// v0.111.0 MP-only Ironclad Attack (`Cards/Outrage.cs`). GH #793.
    Outrage,
    /// v0.111.0 MP-only Regent Skill (`Cards/Plot.cs`). GH #793.
    Plot,
    /// v0.111.0 Silent Uncommon Skill (`Cards/Sidestep.cs`); takes Scare's Silent pool slot. GH #793.
    Sidestep,
    /// v0.111.0 MP-only Necrobinder Power (`Cards/Soulbound.cs`); `_CARD` disambiguates it from `SoulboundPower`. GH #793.
    SoulboundCard,
    /// v0.111.0 MP-only Colorless Attack (`Cards/TheBall.cs`). GH #793.
    TheBall,
    /// v0.111.0 MP-only Regent Skill (`Cards/Tutor.cs`). GH #793.
    Tutor,
    /// v0.111.0 MP-only Necrobinder Skill (`Cards/Underworld.cs`); `_CARD` disambiguates it from `UnderworldPower`. GH #793.
    UnderworldCard,
    // ── v0.113.0-only. Not in [`CardId::all`]; see [`CardId::V0113_ONLY`].
    /// v0.113.0 event Attack (`Cards/DingyDagger.cs`), from Waterfall of
    /// Obvious Secrets: 0 cost, 7 damage, upgradable without limit.
    DingyDagger,
    /// v0.113.0 event Attack (`Cards/MightyWeapon.cs`), from Shrine of the
    /// Warlord: 1 cost, 25 damage to all, 2 Dazed into the draw pile.
    MightyWeapon,
    /// v0.113.0 event Skill (`Cards/MightyShield.cs`), from Shrine of the
    /// Warlord: 1 cost, 20 block, 3 Dazed into the draw pile.
    MightyShield,
    /// v0.113.0 Rare colorless Attack (`Cards/MyShinies.cs`): 1 damage
    /// `floor(Gold / 20)` times (`/ 10` upgraded). After MindBlast in
    /// `ColorlessCardPool`.
    MyShinies,
}

impl CardId {
    /// Python `CardId.<v>.name` — the canonical parity-exporter string.
    #[must_use]
    pub const fn to_python_name(self) -> &'static str {
        match self {
            Self::StrikeIronclad => "STRIKE_IRONCLAD",
            Self::DefendIronclad => "DEFEND_IRONCLAD",
            Self::Bash => "BASH",
            Self::Anger => "ANGER",
            Self::Armaments => "ARMAMENTS",
            Self::BloodWall => "BLOOD_WALL",
            Self::Bloodletting => "BLOODLETTING",
            Self::BodySlam => "BODY_SLAM",
            Self::Breakthrough => "BREAKTHROUGH",
            Self::Cinder => "CINDER",
            Self::Havoc => "HAVOC",
            Self::Headbutt => "HEADBUTT",
            Self::IronWave => "IRON_WAVE",
            Self::MoltenFist => "MOLTEN_FIST",
            Self::PerfectedStrike => "PERFECTED_STRIKE",
            Self::PommelStrike => "POMMEL_STRIKE",
            Self::SetupStrikeCard => "SETUP_STRIKE_CARD",
            Self::ShrugItOff => "SHRUG_IT_OFF",
            Self::SwordBoomerang => "SWORD_BOOMERANG",
            Self::Thunderclap => "THUNDERCLAP",
            Self::Tremble => "TREMBLE",
            Self::TrueGrit => "TRUE_GRIT",
            Self::TwinStrike => "TWIN_STRIKE",
            Self::AshenStrike => "ASHEN_STRIKE",
            Self::BattleTrance => "BATTLE_TRANCE",
            Self::Bludgeon => "BLUDGEON",
            Self::Bully => "BULLY",
            Self::BurningPact => "BURNING_PACT",
            Self::DemonicShield => "DEMONIC_SHIELD",
            Self::Dismantle => "DISMANTLE",
            Self::Dominate => "DOMINATE",
            Self::DrumOfBattleCard => "DRUM_OF_BATTLE_CARD",
            Self::EvilEye => "EVIL_EYE",
            Self::ExpectAFight => "EXPECT_A_FIGHT",
            Self::FeelNoPainCard => "FEEL_NO_PAIN_CARD",
            Self::FightMe => "FIGHT_ME",
            Self::FlameBarrierCard => "FLAME_BARRIER_CARD",
            Self::ForgottenRitual => "FORGOTTEN_RITUAL",
            Self::Hemokinesis => "HEMOKINESIS",
            Self::HowlFromBeyond => "HOWL_FROM_BEYOND",
            Self::InfernalBlade => "INFERNAL_BLADE",
            Self::InfernoCard => "INFERNO_CARD",
            Self::Inflame => "INFLAME",
            Self::JugglingCard => "JUGGLING_CARD",
            Self::Pillage => "PILLAGE",
            Self::RageCard => "RAGE_CARD",
            Self::Rampage => "RAMPAGE",
            Self::RuptureCard => "RUPTURE_CARD",
            Self::SecondWind => "SECOND_WIND",
            Self::Spite => "SPITE",
            Self::StampedeCard => "STAMPEDE_CARD",
            Self::Stomp => "STOMP",
            Self::StoneArmor => "STONE_ARMOR",
            Self::Taunt => "TAUNT",
            Self::Unrelenting => "UNRELENTING",
            Self::Uppercut => "UPPERCUT",
            Self::ViciousCard => "VICIOUS_CARD",
            Self::Whirlwind => "WHIRLWIND",
            Self::AggressionCard => "AGGRESSION_CARD",
            Self::BarricadeCard => "BARRICADE_CARD",
            Self::Brand => "BRAND",
            Self::Cascade => "CASCADE",
            Self::ColossusCard => "COLOSSUS_CARD",
            Self::Conflagration => "CONFLAGRATION",
            Self::CrimsonMantle => "CRIMSON_MANTLE",
            Self::CrueltyCard => "CRUELTY_CARD",
            Self::DarkEmbraceCard => "DARK_EMBRACE_CARD",
            Self::DemonFormCard => "DEMON_FORM_CARD",
            Self::Feed => "FEED",
            Self::FiendFire => "FIEND_FIRE",
            Self::HellraiserCard => "HELLRAISER_CARD",
            Self::Impervious => "IMPERVIOUS",
            Self::JuggernautCard => "JUGGERNAUT_CARD",
            Self::Mangle => "MANGLE",
            Self::NotYet => "NOT_YET", // MANUAL — C#-only, not in Python pin
            Self::Offering => "OFFERING",
            Self::OneTwoPunchCard => "ONE_TWO_PUNCH_CARD",
            Self::PactsEnd => "PACTS_END",
            Self::PrimalForce => "PRIMAL_FORCE",
            Self::Pyre => "PYRE",
            Self::Stoke => "STOKE",
            Self::TankCard => "TANK_CARD",
            Self::TearAsunder => "TEAR_ASUNDER",
            Self::Thrash => "THRASH",
            Self::Unmovable => "UNMOVABLE",
            Self::Break => "BREAK",
            Self::CorruptionCard => "CORRUPTION_CARD",
            Self::StrikeSilent => "STRIKE_SILENT",
            Self::DefendSilent => "DEFEND_SILENT",
            Self::Neutralize => "NEUTRALIZE",
            Self::Survivor => "SURVIVOR",
            Self::StrikeDefect => "STRIKE_DEFECT",
            Self::DefendDefect => "DEFEND_DEFECT",
            Self::Zap => "ZAP",
            Self::Dualcast => "DUALCAST",
            Self::StrikeRegent => "STRIKE_REGENT",
            Self::DefendRegent => "DEFEND_REGENT",
            Self::FallingStar => "FALLING_STAR",
            Self::Venerate => "VENERATE",
            Self::StrikeNecrobinder => "STRIKE_NECROBINDER",
            Self::DefendNecrobinder => "DEFEND_NECROBINDER",
            Self::Bodyguard => "BODYGUARD",
            Self::Unleash => "UNLEASH",
            Self::Wound => "WOUND",
            Self::Dazed => "DAZED",
            Self::Slimed => "SLIMED",
            Self::Burn => "BURN",
            Self::Void => "VOID",
            Self::AscendersBane => "ASCENDERS_BANE",
            Self::Clumsy => "CLUMSY",
            Self::Decay => "DECAY",
            Self::Doubt => "DOUBT",
            Self::Injury => "INJURY",
            Self::Normality => "NORMALITY",
            Self::Pain => "PAIN",
            Self::Parasite => "PARASITE",
            Self::Regret => "REGRET",
            Self::Shame => "SHAME",
            Self::Writhe => "WRITHE",
            Self::Shiv => "SHIV",
            Self::Infection => "INFECTION",
            Self::Beckon => "BECKON",
            Self::Acrobatics => "ACROBATICS",
            Self::Anticipate => "ANTICIPATE",
            Self::Backflip => "BACKFLIP",
            Self::BladeDance => "BLADE_DANCE",
            Self::CloakAndDagger => "CLOAK_AND_DAGGER",
            Self::DaggerSpray => "DAGGER_SPRAY",
            Self::DaggerThrow => "DAGGER_THROW",
            Self::DeadlyPoison => "DEADLY_POISON",
            Self::Deflect => "DEFLECT",
            Self::DodgeAndRoll => "DODGE_AND_ROLL",
            Self::FlickFlack => "FLICK_FLACK",
            Self::LeadingStrike => "LEADING_STRIKE",
            Self::PiercingWail => "PIERCING_WAIL",
            Self::PoisonedStab => "POISONED_STAB",
            Self::Prepared => "PREPARED",
            Self::Ricochet => "RICOCHET",
            Self::Slice => "SLICE",
            Self::Snakebite => "SNAKEBITE",
            Self::SuckerPunch => "SUCKER_PUNCH",
            Self::Untouchable => "UNTOUCHABLE",
            Self::AccuracyCard => "ACCURACY_CARD",
            Self::Backstab => "BACKSTAB",
            Self::BlurCard => "BLUR_CARD",
            Self::BouncingFlask => "BOUNCING_FLASK",
            Self::BubbleBubble => "BUBBLE_BUBBLE",
            Self::CalculatedGamble => "CALCULATED_GAMBLE",
            Self::Dash => "DASH",
            Self::EscapePlan => "ESCAPE_PLAN",
            Self::Expertise => "EXPERTISE",
            Self::Expose => "EXPOSE",
            Self::Finisher => "FINISHER",
            Self::Flanking => "FLANKING",
            Self::Flechettes => "FLECHETTES",
            Self::Scare => "SCARE",
            Self::Footwork => "FOOTWORK",
            Self::HandTrick => "HAND_TRICK",
            Self::Haze => "HAZE",
            Self::HiddenDaggers => "HIDDEN_DAGGERS",
            Self::InfiniteBladesCard => "INFINITE_BLADES_CARD",
            Self::LegSweep => "LEG_SWEEP",
            Self::MementoMori => "MEMENTO_MORI",
            Self::Mirage => "MIRAGE",
            Self::NoxiousFumesCard => "NOXIOUS_FUMES_CARD",
            Self::OutbreakCard => "OUTBREAK_CARD",
            Self::PhantomBladesCard => "PHANTOM_BLADES_CARD",
            Self::Pinpoint => "PINPOINT",
            Self::Pounce => "POUNCE",
            Self::PreciseCut => "PRECISE_CUT",
            Self::Predator => "PREDATOR",
            Self::Reflex => "REFLEX",
            Self::Skewer => "SKEWER",
            Self::SpeedsterCard => "SPEEDSTER_CARD",
            Self::Strangle => "STRANGLE",
            Self::Tactician => "TACTICIAN",
            Self::UpMySleeve => "UP_MY_SLEEVE",
            Self::WellLaidPlans => "WELL_LAID_PLANS",
            Self::Abrasive => "ABRASIVE",
            Self::Accelerant => "ACCELERANT",
            Self::Adrenaline => "ADRENALINE",
            Self::AfterimageCard => "AFTERIMAGE_CARD",
            Self::Assassinate => "ASSASSINATE",
            Self::BladeOfInk => "BLADE_OF_INK",
            Self::BulletTime => "BULLET_TIME",
            Self::Burst => "BURST",
            Self::CorrosiveWave => "CORROSIVE_WAVE",
            Self::EchoingSlash => "ECHOING_SLASH",
            Self::EnvenomCard => "ENVENOM_CARD",
            Self::FanOfKnivesCard => "FAN_OF_KNIVES_CARD",
            Self::GrandFinale => "GRAND_FINALE",
            Self::KnifeTrap => "KNIFE_TRAP",
            Self::Malaise => "MALAISE",
            Self::MasterPlanner => "MASTER_PLANNER",
            Self::Murder => "MURDER",
            Self::Nightmare => "NIGHTMARE",
            Self::SerpentFormCard => "SERPENT_FORM_CARD",
            Self::ShadowStep => "SHADOW_STEP",
            Self::Shadowmeld => "SHADOWMELD",
            Self::SneakyCard => "SNEAKY_CARD",
            Self::StormOfSteel => "STORM_OF_STEEL",
            Self::TheHunt => "THE_HUNT",
            Self::ToolsOfTheTrade => "TOOLS_OF_THE_TRADE",
            Self::Tracking => "TRACKING",
            Self::Suppress => "SUPPRESS",
            Self::WraithForm => "WRAITH_FORM",
            Self::BallLightning => "BALL_LIGHTNING",
            Self::Barrage => "BARRAGE",
            Self::BeamCell => "BEAM_CELL",
            Self::BoostAway => "BOOST_AWAY",
            Self::ChargeBattery => "CHARGE_BATTERY",
            Self::Claw => "CLAW",
            Self::ColdSnap => "COLD_SNAP",
            Self::CompileDriver => "COMPILE_DRIVER",
            Self::Coolheaded => "COOLHEADED",
            Self::FocusedStrikeCard => "FOCUSED_STRIKE_CARD",
            Self::GoForTheEyes => "GO_FOR_THE_EYES",
            Self::GunkUp => "GUNK_UP",
            Self::Hologram => "HOLOGRAM",
            Self::Hotfix => "HOTFIX",
            Self::Leap => "LEAP",
            Self::LightningRod => "LIGHTNING_ROD",
            Self::MomentumStrike => "MOMENTUM_STRIKE",
            Self::SweepingBeam => "SWEEPING_BEAM",
            Self::Turbo => "TURBO",
            Self::Uproar => "UPROAR",
            Self::BootSequence => "BOOT_SEQUENCE",
            Self::BulkUp => "BULK_UP",
            Self::Capacitor => "CAPACITOR",
            Self::Chaos => "CHAOS",
            Self::Chill => "CHILL",
            Self::Compact => "COMPACT",
            Self::DarknessCard => "DARKNESS_CARD",
            Self::DoubleEnergy => "DOUBLE_ENERGY",
            Self::EnergySurge => "ENERGY_SURGE",
            Self::Feral => "FERAL",
            Self::FightThrough => "FIGHT_THROUGH",
            Self::Ftl => "FTL",
            Self::Fusion => "FUSION",
            Self::Glacier => "GLACIER",
            Self::Glasswork => "GLASSWORK",
            Self::Hailstorm => "HAILSTORM",
            Self::IterationCard => "ITERATION_CARD",
            Self::LoopCard => "LOOP_CARD",
            Self::NullCard => "NULL_CARD",
            Self::Overclock => "OVERCLOCK",
            Self::Refract => "REFRACT",
            Self::RocketPunch => "ROCKET_PUNCH",
            Self::Scavenge => "SCAVENGE",
            Self::Scrape => "SCRAPE",
            Self::ShadowShield => "SHADOW_SHIELD",
            Self::Skim => "SKIM",
            Self::Smokestack => "SMOKESTACK",
            Self::StormCard => "STORM_CARD",
            Self::Subroutine => "SUBROUTINE",
            Self::Sunder => "SUNDER",
            Self::Synchronize => "SYNCHRONIZE",
            Self::Synthesis => "SYNTHESIS",
            Self::Tempest => "TEMPEST",
            Self::TeslaCoil => "TESLA_COIL",
            Self::ThunderCard => "THUNDER_CARD",
            Self::WhiteNoise => "WHITE_NOISE",
            Self::AdaptiveStrike => "ADAPTIVE_STRIKE",
            Self::AllForOne => "ALL_FOR_ONE",
            Self::BufferCard => "BUFFER_CARD",
            Self::ConsumingShadow => "CONSUMING_SHADOW",
            Self::Coolant => "COOLANT",
            Self::CreativeAiCard => "CREATIVE_AI_CARD",
            Self::Defragment => "DEFRAGMENT",
            Self::EchoFormCard => "ECHO_FORM_CARD",
            Self::FlakCannon => "FLAK_CANNON",
            Self::GeneticAlgorithm => "GENETIC_ALGORITHM",
            Self::HelixDrill => "HELIX_DRILL",
            Self::Hyperbeam => "HYPERBEAM",
            Self::IceLance => "ICE_LANCE",
            Self::Ignition => "IGNITION",
            Self::MachineLearningCard => "MACHINE_LEARNING_CARD",
            Self::MeteorStrike => "METEOR_STRIKE",
            Self::Modded => "MODDED",
            Self::MultiCast => "MULTI_CAST",
            Self::Rainbow => "RAINBOW",
            Self::Reboot => "REBOOT",
            Self::Shatter => "SHATTER",
            Self::SignalBoost => "SIGNAL_BOOST",
            Self::SpinnerCard => "SPINNER_CARD",
            Self::Supercritical => "SUPERCRITICAL",
            Self::TrashToTreasure => "TRASH_TO_TREASURE",
            Self::Voltaic => "VOLTAIC",
            Self::BiasedCognitionCard => "BIASED_COGNITION_CARD",
            Self::Quadcast => "QUADCAST",
            Self::Afterlife => "AFTERLIFE",
            Self::BlightStrike => "BLIGHT_STRIKE",
            Self::Defile => "DEFILE",
            Self::Defy => "DEFY",
            Self::DrainPower => "DRAIN_POWER",
            Self::Fear => "FEAR",
            Self::Flatten => "FLATTEN",
            Self::GraveWarden => "GRAVE_WARDEN",
            Self::Graveblast => "GRAVEBLAST",
            Self::Invoke => "INVOKE",
            Self::NegativePulse => "NEGATIVE_PULSE",
            Self::Poke => "POKE",
            Self::PullAggro => "PULL_AGGRO",
            Self::Reap => "REAP",
            Self::Reave => "REAVE",
            Self::Scourge => "SCOURGE",
            Self::SculptingStrike => "SCULPTING_STRIKE",
            Self::Snap => "SNAP",
            Self::Sow => "SOW",
            Self::Wisp => "WISP",
            Self::BoneShards => "BONE_SHARDS",
            Self::BorrowedTime => "BORROWED_TIME",
            Self::Bury => "BURY",
            Self::CalcifyCard => "CALCIFY_CARD",
            Self::CaptureSpirit => "CAPTURE_SPIRIT",
            Self::Cleanse => "CLEANSE",
            Self::CountdownCard => "COUNTDOWN_CARD",
            Self::DanseMacabre => "DANSE_MACABRE",
            Self::DeathMarch => "DEATH_MARCH",
            Self::Deathbringer => "DEATHBRINGER",
            Self::DeathsDoor => "DEATHS_DOOR",
            Self::DebilitateCard => "DEBILITATE_CARD",
            Self::Delay => "DELAY",
            Self::Dirge => "DIRGE",
            Self::Dredge => "DREDGE",
            Self::EnfeeblingTouch => "ENFEEBLING_TOUCH",
            Self::Fetch => "FETCH",
            Self::Friendship => "FRIENDSHIP",
            Self::Haunt => "HAUNT",
            Self::HighFive => "HIGH_FIVE",
            Self::LegionOfBone => "LEGION_OF_BONE",
            Self::LethalityCard => "LETHALITY_CARD",
            Self::Melancholy => "MELANCHOLY",
            Self::NoEscape => "NO_ESCAPE",
            Self::Pagestorm => "PAGESTORM",
            Self::Parse => "PARSE",
            Self::PullFromBelow => "PULL_FROM_BELOW",
            Self::Putrefy => "PUTREFY",
            Self::Rattle => "RATTLE",
            Self::RightHandHand => "RIGHT_HAND_HAND",
            Self::Severance => "SEVERANCE",
            Self::Shroud => "SHROUD",
            Self::SicEm => "SIC_EM",
            Self::SleightOfFlesh => "SLEIGHT_OF_FLESH",
            Self::Spur => "SPUR",
            Self::Veilpiercer => "VEILPIERCER",
            Self::BansheesCry => "BANSHEES_CRY",
            Self::CallOfTheVoid => "CALL_OF_THE_VOID",
            Self::Demesne => "DEMESNE",
            Self::DevourLifeCard => "DEVOUR_LIFE_CARD",
            Self::Eidolon => "EIDOLON",
            Self::EndOfDays => "END_OF_DAYS",
            Self::Eradicate => "ERADICATE",
            Self::GlimpseBeyond => "GLIMPSE_BEYOND",
            Self::Hang => "HANG",
            Self::Misery => "MISERY",
            Self::NecroMasteryCard => "NECRO_MASTERY_CARD",
            Self::Neurosurge => "NEUROSURGE",
            Self::Oblivion => "OBLIVION",
            Self::Reanimate => "REANIMATE",
            Self::ReaperForm => "REAPER_FORM",
            Self::Sacrifice => "SACRIFICE",
            Self::Seance => "SEANCE",
            Self::SentryMode => "SENTRY_MODE",
            Self::SharedFate => "SHARED_FATE",
            Self::SoulStorm => "SOUL_STORM",
            Self::SpiritOfAsh => "SPIRIT_OF_ASH",
            Self::Squeeze => "SQUEEZE",
            Self::TheScythe => "THE_SCYTHE",
            Self::TimesUp => "TIMES_UP",
            Self::Transfigure => "TRANSFIGURE",
            Self::Undeath => "UNDEATH",
            Self::ForbiddenGrimoire => "FORBIDDEN_GRIMOIRE",
            Self::Protector => "PROTECTOR",
            Self::AstralPulse => "ASTRAL_PULSE",
            Self::Begone => "BEGONE",
            Self::CelestialMight => "CELESTIAL_MIGHT",
            Self::CloakOfStars => "CLOAK_OF_STARS",
            Self::CollisionCourse => "COLLISION_COURSE",
            Self::CosmicIndifference => "COSMIC_INDIFFERENCE",
            Self::CrescentSpear => "CRESCENT_SPEAR",
            Self::CrushUnder => "CRUSH_UNDER",
            Self::GatherLight => "GATHER_LIGHT",
            Self::Glitterstream => "GLITTERSTREAM",
            Self::Glow => "GLOW",
            Self::GuidingStar => "GUIDING_STAR",
            Self::HiddenCache => "HIDDEN_CACHE",
            Self::KnowThyPlace => "KNOW_THY_PLACE",
            Self::Patter => "PATTER",
            Self::PhotonCut => "PHOTON_CUT",
            Self::RefineBlade => "REFINE_BLADE",
            Self::SolarStrike => "SOLAR_STRIKE",
            Self::SpoilsOfBattle => "SPOILS_OF_BATTLE",
            Self::WroughtInWar => "WROUGHT_IN_WAR",
            Self::Alignment => "ALIGNMENT",
            Self::BlackHole => "BLACK_HOLE",
            Self::Bulwark => "BULWARK",
            Self::Charge => "CHARGE",
            Self::ChildOfTheStars => "CHILD_OF_THE_STARS",
            Self::Conqueror => "CONQUEROR",
            Self::Convergence => "CONVERGENCE",
            Self::Devastate => "DEVASTATE",
            Self::Furnace => "FURNACE",
            Self::GammaBlast => "GAMMA_BLAST",
            Self::Glimmer => "GLIMMER",
            Self::Hegemony => "HEGEMONY",
            Self::KinglyKick => "KINGLY_KICK",
            Self::KinglyPunch => "KINGLY_PUNCH",
            Self::KnockoutBlow => "KNOCKOUT_BLOW",
            Self::Largesse => "LARGESSE",
            Self::LunarBlast => "LUNAR_BLAST",
            Self::ManifestAuthority => "MANIFEST_AUTHORITY",
            Self::MonologueCard => "MONOLOGUE_CARD",
            Self::Orbit => "ORBIT",
            Self::PaleBlueDot => "PALE_BLUE_DOT",
            Self::ParryCard => "PARRY_CARD",
            Self::ParticleWall => "PARTICLE_WALL",
            Self::PillarOfCreation => "PILLAR_OF_CREATION",
            Self::Prophesize => "PROPHESIZE",
            Self::Quasar => "QUASAR",
            Self::Radiate => "RADIATE",
            Self::ReflectCard => "REFLECT_CARD",
            Self::Resonance => "RESONANCE",
            Self::RoyalGamble => "ROYAL_GAMBLE",
            Self::ShiningStrike => "SHINING_STRIKE",
            Self::SpectrumShift => "SPECTRUM_SHIFT",
            Self::Stardust => "STARDUST",
            Self::SummonForth => "SUMMON_FORTH",
            Self::Supermassive => "SUPERMASSIVE",
            Self::Terraforming => "TERRAFORMING",
            Self::Arsenal => "ARSENAL",
            Self::BeatIntoShape => "BEAT_INTO_SHAPE",
            Self::BigBang => "BIG_BANG",
            Self::Bombardment => "BOMBARDMENT",
            Self::BundleOfJoy => "BUNDLE_OF_JOY",
            Self::Comet => "COMET",
            Self::CrashLanding => "CRASH_LANDING",
            Self::DecisionsDecisions => "DECISIONS_DECISIONS",
            Self::DyingStar => "DYING_STAR",
            Self::ForegoneConclusion => "FOREGONE_CONCLUSION",
            Self::Genesis => "GENESIS",
            Self::Guards => "GUARDS",
            Self::HammerTime => "HAMMER_TIME",
            Self::HeavenlyDrill => "HEAVENLY_DRILL",
            Self::HeirloomHammer => "HEIRLOOM_HAMMER",
            Self::IAmInvincible => "I_AM_INVINCIBLE",
            Self::MakeItSo => "MAKE_IT_SO",
            Self::MonarchsGazeCard => "MONARCHS_GAZE_CARD",
            Self::NeutronAegis => "NEUTRON_AEGIS",
            Self::RoyaltiesCard => "ROYALTIES_CARD",
            Self::SeekingEdge => "SEEKING_EDGE",
            Self::SevenStars => "SEVEN_STARS",
            Self::SwordSage => "SWORD_SAGE",
            Self::TheSmith => "THE_SMITH",
            Self::TyrannyCard => "TYRANNY_CARD",
            Self::VoidForm => "VOID_FORM",
            Self::MeteorShower => "METEOR_SHOWER",
            Self::TheSealedThrone => "THE_SEALED_THRONE",
            Self::Automation => "AUTOMATION",
            Self::BelieveInYou => "BELIEVE_IN_YOU",
            Self::Catastrophe => "CATASTROPHE",
            Self::CoordinateCard => "COORDINATE_CARD",
            Self::DarkShackles => "DARK_SHACKLES",
            Self::Discovery => "DISCOVERY",
            Self::DramaticEntrance => "DRAMATIC_ENTRANCE",
            Self::Equilibrium => "EQUILIBRIUM",
            Self::Fasten => "FASTEN",
            Self::Finesse => "FINESSE",
            Self::Fisticuffs => "FISTICUFFS",
            Self::FlashOfSteel => "FLASH_OF_STEEL",
            Self::GangUp => "GANG_UP",
            Self::HuddleUp => "HUDDLE_UP",
            Self::Impatience => "IMPATIENCE",
            Self::InterceptCard => "INTERCEPT_CARD",
            Self::JackOfAllTrades => "JACK_OF_ALL_TRADES",
            Self::Lift => "LIFT",
            Self::MindBlast => "MIND_BLAST",
            Self::Omnislice => "OMNISLICE",
            Self::PanacheCard => "PANACHE_CARD",
            Self::PanicButton => "PANIC_BUTTON",
            Self::PrepTime => "PREP_TIME",
            Self::Production => "PRODUCTION",
            Self::Prolong => "PROLONG",
            Self::Prowess => "PROWESS",
            Self::Purity => "PURITY",
            Self::Restlessness => "RESTLESSNESS",
            Self::SeekerStrike => "SEEKER_STRIKE",
            Self::Shockwave => "SHOCKWAVE",
            Self::Splash => "SPLASH",
            Self::Stratagem => "STRATAGEM",
            Self::TagTeam => "TAG_TEAM",
            Self::TheBombCard => "THE_BOMB_CARD",
            Self::ThinkingAhead => "THINKING_AHEAD",
            Self::ThrummingHatchet => "THRUMMING_HATCHET",
            Self::UltimateDefend => "ULTIMATE_DEFEND",
            Self::UltimateStrike => "ULTIMATE_STRIKE",
            Self::Volley => "VOLLEY",
            Self::Alchemize => "ALCHEMIZE",
            Self::Anointed => "ANOINTED",
            Self::BeaconOfHope => "BEACON_OF_HOPE",
            Self::BeatDown => "BEAT_DOWN",
            Self::Bolas => "BOLAS",
            Self::CalamityCard => "CALAMITY_CARD",
            Self::Entropy => "ENTROPY",
            Self::EternalArmor => "ETERNAL_ARMOR",
            Self::GoldAxe => "GOLD_AXE",
            Self::HandOfGreed => "HAND_OF_GREED",
            Self::HiddenGem => "HIDDEN_GEM",
            Self::Jackpot => "JACKPOT",
            Self::Knockdown => "KNOCKDOWN",
            Self::MasterOfStrategy => "MASTER_OF_STRATEGY",
            Self::MayhemCard => "MAYHEM_CARD",
            Self::Mimic => "MIMIC",
            Self::NostalgiaCard => "NOSTALGIA_CARD",
            Self::Rally => "RALLY",
            Self::Rend => "REND",
            Self::RollingBoulder => "ROLLING_BOULDER",
            Self::Salvo => "SALVO",
            Self::Scrawl => "SCRAWL",
            Self::SecretTechnique => "SECRET_TECHNIQUE",
            Self::SecretWeapon => "SECRET_WEAPON",
            Self::TheGambit => "THE_GAMBIT",
            Self::RipAndTear => "RIP_AND_TEAR",
            Self::Apotheosis => "APOTHEOSIS",
            Self::Apparition => "APPARITION",
            Self::BrightestFlame => "BRIGHTEST_FLAME",
            Self::Maul => "MAUL",
            Self::NeowsFury => "NEOWS_FURY",
            Self::Relax => "RELAX",
            Self::Whistle => "WHISTLE",
            Self::Wish => "WISH",
            Self::ByrdSwoop => "BYRD_SWOOP",
            Self::Caltrops => "CALTROPS",
            Self::Clash => "CLASH",
            Self::Distraction => "DISTRACTION",
            Self::DualWield => "DUAL_WIELD",
            Self::Enlightenment => "ENLIGHTENMENT",
            Self::Entrench => "ENTRENCH",
            Self::Exterminate => "EXTERMINATE",
            Self::FeedingFrenzyCard => "FEEDING_FRENZY_CARD",
            Self::HelloWorldCard => "HELLO_WORLD_CARD",
            Self::MadScience => "MAD_SCIENCE",
            Self::Metamorphosis => "METAMORPHOSIS",
            Self::Outmaneuver => "OUTMANEUVER",
            Self::Peck => "PECK",
            Self::Rebound => "REBOUND",
            Self::Squash => "SQUASH",
            Self::Stack => "STACK",
            Self::ToricToughness => "TORIC_TOUGHNESS",
            Self::Fuel => "FUEL",
            Self::GiantRock => "GIANT_ROCK",
            Self::Luminesce => "LUMINESCE",
            Self::MinionDiveBomb => "MINION_DIVE_BOMB",
            Self::MinionSacrifice => "MINION_SACRIFICE",
            Self::MinionStrike => "MINION_STRIKE",
            Self::Soul => "SOUL",
            Self::SovereignBlade => "SOVEREIGN_BLADE",
            Self::SweepingGaze => "SWEEPING_GAZE",
            Self::Disintegration => "DISINTEGRATION",
            Self::MindRot => "MIND_ROT",
            Self::SlothStatus => "SLOTH_STATUS",
            Self::WasteAway => "WASTE_AWAY",
            Self::Debris => "DEBRIS",
            Self::FranticEscape => "FRANTIC_ESCAPE",
            Self::Soot => "SOOT",
            Self::Toxic => "TOXIC",
            Self::BadLuck => "BAD_LUCK",
            Self::CurseOfTheBell => "CURSE_OF_THE_BELL",
            Self::Debt => "DEBT",
            Self::Enthralled => "ENTHRALLED",
            Self::Folly => "FOLLY",
            Self::Greed => "GREED",
            Self::Guilty => "GUILTY",
            Self::PoorSleep => "POOR_SLEEP",
            Self::SporeMind => "SPORE_MIND",
            Self::ByrdonisEgg => "BYRDONIS_EGG",
            Self::LanternKey => "LANTERN_KEY",
            Self::SpoilsMap => "SPOILS_MAP",
            Self::Wither => "WITHER",
            Self::Generic => "GENERIC",
            Self::Dowsing => "DOWSING",
            Self::Abundance => "ABUNDANCE",
            Self::BladeSymphony => "BLADE_SYMPHONY",
            Self::Blaze => "BLAZE",
            Self::CacophonyCard => "CACOPHONY_CARD",
            Self::ConcoctCard => "CONCOCT_CARD",
            Self::Constellation => "CONSTELLATION",
            Self::FadeCard => "FADE_CARD",
            Self::HibernateCard => "HIBERNATE_CARD",
            Self::ImitationLearningCard => "IMITATION_LEARNING_CARD",
            Self::Midnight => "MIDNIGHT",
            Self::OneForAllCard => "ONE_FOR_ALL_CARD",
            Self::Outrage => "OUTRAGE",
            Self::Plot => "PLOT",
            Self::Sidestep => "SIDESTEP",
            Self::SoulboundCard => "SOULBOUND_CARD",
            Self::TheBall => "THE_BALL",
            Self::Tutor => "TUTOR",
            Self::UnderworldCard => "UNDERWORLD_CARD",
            Self::DingyDagger => "DINGY_DAGGER",
            Self::MightyWeapon => "MIGHTY_WEAPON",
            Self::MightyShield => "MIGHTY_SHIELD",
            Self::MyShinies => "MY_SHINIES",
        }
    }

    /// Every variant, in source order. Useful for exhaustive iteration.
    #[must_use]
    pub const fn all() -> &'static [CardId] {
        &[
            Self::StrikeIronclad,
            Self::DefendIronclad,
            Self::Bash,
            Self::Anger,
            Self::Armaments,
            Self::BloodWall,
            Self::Bloodletting,
            Self::BodySlam,
            Self::Breakthrough,
            Self::Cinder,
            Self::Havoc,
            Self::Headbutt,
            Self::IronWave,
            Self::MoltenFist,
            Self::PerfectedStrike,
            Self::PommelStrike,
            Self::SetupStrikeCard,
            Self::ShrugItOff,
            Self::SwordBoomerang,
            Self::Thunderclap,
            Self::Tremble,
            Self::TrueGrit,
            Self::TwinStrike,
            Self::AshenStrike,
            Self::BattleTrance,
            Self::Bludgeon,
            Self::Bully,
            Self::BurningPact,
            Self::DemonicShield,
            Self::Dismantle,
            Self::Dominate,
            Self::DrumOfBattleCard,
            Self::EvilEye,
            Self::ExpectAFight,
            Self::FeelNoPainCard,
            Self::FightMe,
            Self::FlameBarrierCard,
            Self::ForgottenRitual,
            Self::Hemokinesis,
            Self::HowlFromBeyond,
            Self::InfernalBlade,
            Self::InfernoCard,
            Self::Inflame,
            Self::JugglingCard,
            Self::Pillage,
            Self::RageCard,
            Self::Rampage,
            Self::RuptureCard,
            Self::SecondWind,
            Self::Spite,
            Self::StampedeCard,
            Self::Stomp,
            Self::StoneArmor,
            Self::Taunt,
            Self::Unrelenting,
            Self::Uppercut,
            Self::ViciousCard,
            Self::Whirlwind,
            Self::AggressionCard,
            Self::BarricadeCard,
            Self::Brand,
            Self::Cascade,
            Self::ColossusCard,
            Self::Conflagration,
            Self::CrimsonMantle,
            Self::CrueltyCard,
            Self::DarkEmbraceCard,
            Self::DemonFormCard,
            Self::Feed,
            Self::FiendFire,
            Self::HellraiserCard,
            Self::Impervious,
            Self::JuggernautCard,
            Self::Mangle,
            Self::NotYet, // MANUAL — C#-only, not in Python pin
            Self::Offering,
            Self::OneTwoPunchCard,
            Self::PactsEnd,
            Self::PrimalForce,
            Self::Pyre,
            Self::Stoke,
            Self::TankCard,
            Self::TearAsunder,
            Self::Thrash,
            Self::Unmovable,
            Self::Break,
            Self::CorruptionCard,
            Self::StrikeSilent,
            Self::DefendSilent,
            Self::Neutralize,
            Self::Survivor,
            Self::StrikeDefect,
            Self::DefendDefect,
            Self::Zap,
            Self::Dualcast,
            Self::StrikeRegent,
            Self::DefendRegent,
            Self::FallingStar,
            Self::Venerate,
            Self::StrikeNecrobinder,
            Self::DefendNecrobinder,
            Self::Bodyguard,
            Self::Unleash,
            Self::Wound,
            Self::Dazed,
            Self::Slimed,
            Self::Burn,
            Self::Void,
            Self::AscendersBane,
            Self::Clumsy,
            Self::Decay,
            Self::Doubt,
            Self::Injury,
            Self::Normality,
            Self::Pain,
            Self::Parasite,
            Self::Regret,
            Self::Shame,
            Self::Writhe,
            Self::Shiv,
            Self::Infection,
            Self::Beckon,
            Self::Acrobatics,
            Self::Anticipate,
            Self::Backflip,
            Self::BladeDance,
            Self::CloakAndDagger,
            Self::DaggerSpray,
            Self::DaggerThrow,
            Self::DeadlyPoison,
            Self::Deflect,
            Self::DodgeAndRoll,
            Self::FlickFlack,
            Self::LeadingStrike,
            Self::PiercingWail,
            Self::PoisonedStab,
            Self::Prepared,
            Self::Ricochet,
            Self::Slice,
            Self::Snakebite,
            Self::SuckerPunch,
            Self::Untouchable,
            Self::AccuracyCard,
            Self::Backstab,
            Self::BlurCard,
            Self::BouncingFlask,
            Self::BubbleBubble,
            Self::CalculatedGamble,
            Self::Dash,
            Self::EscapePlan,
            Self::Expertise,
            Self::Expose,
            Self::Finisher,
            Self::Flanking,
            Self::Flechettes,
            Self::Scare,
            Self::Footwork,
            Self::HandTrick,
            Self::Haze,
            Self::HiddenDaggers,
            Self::InfiniteBladesCard,
            Self::LegSweep,
            Self::MementoMori,
            Self::Mirage,
            Self::NoxiousFumesCard,
            Self::OutbreakCard,
            Self::PhantomBladesCard,
            Self::Pinpoint,
            Self::Pounce,
            Self::PreciseCut,
            Self::Predator,
            Self::Reflex,
            Self::Skewer,
            Self::SpeedsterCard,
            Self::Strangle,
            Self::Tactician,
            Self::UpMySleeve,
            Self::WellLaidPlans,
            Self::Abrasive,
            Self::Accelerant,
            Self::Adrenaline,
            Self::AfterimageCard,
            Self::Assassinate,
            Self::BladeOfInk,
            Self::BulletTime,
            Self::Burst,
            Self::CorrosiveWave,
            Self::EchoingSlash,
            Self::EnvenomCard,
            Self::FanOfKnivesCard,
            Self::GrandFinale,
            Self::KnifeTrap,
            Self::Malaise,
            Self::MasterPlanner,
            Self::Murder,
            Self::Nightmare,
            Self::SerpentFormCard,
            Self::ShadowStep,
            Self::Shadowmeld,
            Self::SneakyCard,
            Self::StormOfSteel,
            Self::TheHunt,
            Self::ToolsOfTheTrade,
            Self::Tracking,
            Self::Suppress,
            Self::WraithForm,
            Self::BallLightning,
            Self::Barrage,
            Self::BeamCell,
            Self::BoostAway,
            Self::ChargeBattery,
            Self::Claw,
            Self::ColdSnap,
            Self::CompileDriver,
            Self::Coolheaded,
            Self::FocusedStrikeCard,
            Self::GoForTheEyes,
            Self::GunkUp,
            Self::Hologram,
            Self::Hotfix,
            Self::Leap,
            Self::LightningRod,
            Self::MomentumStrike,
            Self::SweepingBeam,
            Self::Turbo,
            Self::Uproar,
            Self::BootSequence,
            Self::BulkUp,
            Self::Capacitor,
            Self::Chaos,
            Self::Chill,
            Self::Compact,
            Self::DarknessCard,
            Self::DoubleEnergy,
            Self::EnergySurge,
            Self::Feral,
            Self::FightThrough,
            Self::Ftl,
            Self::Fusion,
            Self::Glacier,
            Self::Glasswork,
            Self::Hailstorm,
            Self::IterationCard,
            Self::LoopCard,
            Self::NullCard,
            Self::Overclock,
            Self::Refract,
            Self::RocketPunch,
            Self::Scavenge,
            Self::Scrape,
            Self::ShadowShield,
            Self::Skim,
            Self::Smokestack,
            Self::StormCard,
            Self::Subroutine,
            Self::Sunder,
            Self::Synchronize,
            Self::Synthesis,
            Self::Tempest,
            Self::TeslaCoil,
            Self::ThunderCard,
            Self::WhiteNoise,
            Self::AdaptiveStrike,
            Self::AllForOne,
            Self::BufferCard,
            Self::ConsumingShadow,
            Self::Coolant,
            Self::CreativeAiCard,
            Self::Defragment,
            Self::EchoFormCard,
            Self::FlakCannon,
            Self::GeneticAlgorithm,
            Self::HelixDrill,
            Self::Hyperbeam,
            Self::IceLance,
            Self::Ignition,
            Self::MachineLearningCard,
            Self::MeteorStrike,
            Self::Modded,
            Self::MultiCast,
            Self::Rainbow,
            Self::Reboot,
            Self::Shatter,
            Self::SignalBoost,
            Self::SpinnerCard,
            Self::Supercritical,
            Self::TrashToTreasure,
            Self::Voltaic,
            Self::BiasedCognitionCard,
            Self::Quadcast,
            Self::Afterlife,
            Self::BlightStrike,
            Self::Defile,
            Self::Defy,
            Self::DrainPower,
            Self::Fear,
            Self::Flatten,
            Self::GraveWarden,
            Self::Graveblast,
            Self::Invoke,
            Self::NegativePulse,
            Self::Poke,
            Self::PullAggro,
            Self::Reap,
            Self::Reave,
            Self::Scourge,
            Self::SculptingStrike,
            Self::Snap,
            Self::Sow,
            Self::Wisp,
            Self::BoneShards,
            Self::BorrowedTime,
            Self::Bury,
            Self::CalcifyCard,
            Self::CaptureSpirit,
            Self::Cleanse,
            Self::CountdownCard,
            Self::DanseMacabre,
            Self::DeathMarch,
            Self::Deathbringer,
            Self::DeathsDoor,
            Self::DebilitateCard,
            Self::Delay,
            Self::Dirge,
            Self::Dredge,
            Self::EnfeeblingTouch,
            Self::Fetch,
            Self::Friendship,
            Self::Haunt,
            Self::HighFive,
            Self::LegionOfBone,
            Self::LethalityCard,
            Self::Melancholy,
            Self::NoEscape,
            Self::Pagestorm,
            Self::Parse,
            Self::PullFromBelow,
            Self::Putrefy,
            Self::Rattle,
            Self::RightHandHand,
            Self::Severance,
            Self::Shroud,
            Self::SicEm,
            Self::SleightOfFlesh,
            Self::Spur,
            Self::Veilpiercer,
            Self::BansheesCry,
            Self::CallOfTheVoid,
            Self::Demesne,
            Self::DevourLifeCard,
            Self::Eidolon,
            Self::EndOfDays,
            Self::Eradicate,
            Self::GlimpseBeyond,
            Self::Hang,
            Self::Misery,
            Self::NecroMasteryCard,
            Self::Neurosurge,
            Self::Oblivion,
            Self::Reanimate,
            Self::ReaperForm,
            Self::Sacrifice,
            Self::Seance,
            Self::SentryMode,
            Self::SharedFate,
            Self::SoulStorm,
            Self::SpiritOfAsh,
            Self::Squeeze,
            Self::TheScythe,
            Self::TimesUp,
            Self::Transfigure,
            Self::Undeath,
            Self::ForbiddenGrimoire,
            Self::Protector,
            Self::AstralPulse,
            Self::Begone,
            Self::CelestialMight,
            Self::CloakOfStars,
            Self::CollisionCourse,
            Self::CosmicIndifference,
            Self::CrescentSpear,
            Self::CrushUnder,
            Self::GatherLight,
            Self::Glitterstream,
            Self::Glow,
            Self::GuidingStar,
            Self::HiddenCache,
            Self::KnowThyPlace,
            Self::Patter,
            Self::PhotonCut,
            Self::RefineBlade,
            Self::SolarStrike,
            Self::SpoilsOfBattle,
            Self::WroughtInWar,
            Self::Alignment,
            Self::BlackHole,
            Self::Bulwark,
            Self::Charge,
            Self::ChildOfTheStars,
            Self::Conqueror,
            Self::Convergence,
            Self::Devastate,
            Self::Furnace,
            Self::GammaBlast,
            Self::Glimmer,
            Self::Hegemony,
            Self::KinglyKick,
            Self::KinglyPunch,
            Self::KnockoutBlow,
            Self::Largesse,
            Self::LunarBlast,
            Self::ManifestAuthority,
            Self::MonologueCard,
            Self::Orbit,
            Self::PaleBlueDot,
            Self::ParryCard,
            Self::ParticleWall,
            Self::PillarOfCreation,
            Self::Prophesize,
            Self::Quasar,
            Self::Radiate,
            Self::ReflectCard,
            Self::Resonance,
            Self::RoyalGamble,
            Self::ShiningStrike,
            Self::SpectrumShift,
            Self::Stardust,
            Self::SummonForth,
            Self::Supermassive,
            Self::Terraforming,
            Self::Arsenal,
            Self::BeatIntoShape,
            Self::BigBang,
            Self::Bombardment,
            Self::BundleOfJoy,
            Self::Comet,
            Self::CrashLanding,
            Self::DecisionsDecisions,
            Self::DyingStar,
            Self::ForegoneConclusion,
            Self::Genesis,
            Self::Guards,
            Self::HammerTime,
            Self::HeavenlyDrill,
            Self::HeirloomHammer,
            Self::IAmInvincible,
            Self::MakeItSo,
            Self::MonarchsGazeCard,
            Self::NeutronAegis,
            Self::RoyaltiesCard,
            Self::SeekingEdge,
            Self::SevenStars,
            Self::SwordSage,
            Self::TheSmith,
            Self::TyrannyCard,
            Self::VoidForm,
            Self::MeteorShower,
            Self::TheSealedThrone,
            Self::Automation,
            Self::BelieveInYou,
            Self::Catastrophe,
            Self::CoordinateCard,
            Self::DarkShackles,
            Self::Discovery,
            Self::DramaticEntrance,
            Self::Equilibrium,
            Self::Fasten,
            Self::Finesse,
            Self::Fisticuffs,
            Self::FlashOfSteel,
            Self::GangUp,
            Self::HuddleUp,
            Self::Impatience,
            Self::InterceptCard,
            Self::JackOfAllTrades,
            Self::Lift,
            Self::MindBlast,
            Self::Omnislice,
            Self::PanacheCard,
            Self::PanicButton,
            Self::PrepTime,
            Self::Production,
            Self::Prolong,
            Self::Prowess,
            Self::Purity,
            Self::Restlessness,
            Self::SeekerStrike,
            Self::Shockwave,
            Self::Splash,
            Self::Stratagem,
            Self::TagTeam,
            Self::TheBombCard,
            Self::ThinkingAhead,
            Self::ThrummingHatchet,
            Self::UltimateDefend,
            Self::UltimateStrike,
            Self::Volley,
            Self::Alchemize,
            Self::Anointed,
            Self::BeaconOfHope,
            Self::BeatDown,
            Self::Bolas,
            Self::CalamityCard,
            Self::Entropy,
            Self::EternalArmor,
            Self::GoldAxe,
            Self::HandOfGreed,
            Self::HiddenGem,
            Self::Jackpot,
            Self::Knockdown,
            Self::MasterOfStrategy,
            Self::MayhemCard,
            Self::Mimic,
            Self::NostalgiaCard,
            Self::Rally,
            Self::Rend,
            Self::RollingBoulder,
            Self::Salvo,
            Self::Scrawl,
            Self::SecretTechnique,
            Self::SecretWeapon,
            Self::TheGambit,
            Self::RipAndTear,
            Self::Apotheosis,
            Self::Apparition,
            Self::BrightestFlame,
            Self::Maul,
            Self::NeowsFury,
            Self::Relax,
            Self::Whistle,
            Self::Wish,
            Self::ByrdSwoop,
            Self::Caltrops,
            Self::Clash,
            Self::Distraction,
            Self::DualWield,
            Self::Enlightenment,
            Self::Entrench,
            Self::Exterminate,
            Self::FeedingFrenzyCard,
            Self::HelloWorldCard,
            Self::MadScience,
            Self::Metamorphosis,
            Self::Outmaneuver,
            Self::Peck,
            Self::Rebound,
            Self::Squash,
            Self::Stack,
            Self::ToricToughness,
            Self::Fuel,
            Self::GiantRock,
            Self::Luminesce,
            Self::MinionDiveBomb,
            Self::MinionSacrifice,
            Self::MinionStrike,
            Self::Soul,
            Self::SovereignBlade,
            Self::SweepingGaze,
            Self::Disintegration,
            Self::MindRot,
            Self::SlothStatus,
            Self::WasteAway,
            Self::Debris,
            Self::FranticEscape,
            Self::Soot,
            Self::Toxic,
            Self::BadLuck,
            Self::CurseOfTheBell,
            Self::Debt,
            Self::Enthralled,
            Self::Folly,
            Self::Greed,
            Self::Guilty,
            Self::PoorSleep,
            Self::SporeMind,
            Self::ByrdonisEgg,
            Self::LanternKey,
            Self::SpoilsMap,
            Self::Wither,
            Self::Generic,
        ]
    }

    /// Cards that exist only in game build v0.111.0 (GH #792).
    pub const V0111_ONLY: [CardId; 19] = [
        Self::Dowsing,
        Self::Abundance,
        Self::BladeSymphony,
        Self::Blaze,
        Self::CacophonyCard,
        Self::ConcoctCard,
        Self::Constellation,
        Self::FadeCard,
        Self::HibernateCard,
        Self::ImitationLearningCard,
        Self::Midnight,
        Self::OneForAllCard,
        Self::Outrage,
        Self::Plot,
        Self::Sidestep,
        Self::SoulboundCard,
        Self::TheBall,
        Self::Tutor,
        Self::UnderworldCard,
    ];

    /// Cards that exist only from game build v0.113.0 on.
    pub const V0113_ONLY: [CardId; 4] = [
        Self::DingyDagger,
        Self::MightyWeapon,
        Self::MightyShield,
        Self::MyShinies,
    ];

    /// The build a card first exists in.
    #[must_use]
    pub const fn introduced_in(self) -> crate::GameVersion {
        match self {
            Self::Dowsing
            | Self::Abundance
            | Self::BladeSymphony
            | Self::Blaze
            | Self::CacophonyCard
            | Self::ConcoctCard
            | Self::Constellation
            | Self::FadeCard
            | Self::HibernateCard
            | Self::ImitationLearningCard
            | Self::Midnight
            | Self::OneForAllCard
            | Self::Outrage
            | Self::Plot
            | Self::Sidestep
            | Self::SoulboundCard
            | Self::TheBall
            | Self::Tutor
            | Self::UnderworldCard => crate::GameVersion::V0_111_0,
            Self::DingyDagger | Self::MightyWeapon | Self::MightyShield | Self::MyShinies => {
                crate::GameVersion::V0_113_0
            }
            _ => crate::GameVersion::V0_107_1,
        }
    }

    /// Every card of game build `version`: [`Self::all`] (the v0.107.1
    /// universe, unchanged) plus, from v0.111.0 on, [`Self::V0111_ONLY`].
    #[must_use]
    pub fn all_for(version: crate::GameVersion) -> Vec<CardId> {
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
