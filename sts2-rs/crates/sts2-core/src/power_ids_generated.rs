// HAND-MAINTAINED against ../sts2-wiki/decompiled/v0.107.1/ — edit directly.
// The `_generated` filename is historical; there is no generator any more.
//
// Origin: bead sts2-rs-8c0 (P6-2) bulk-scraped this enum from
// `sts2-rl-agent@01c094d:sts2_env/core/enums.py::PowerId` via a
// `scripts/scrape_power_ids.py` that no longer exists. The Python pin is
// archived and is NOT the source of truth for game content (see
// `PYTHON_SIM_PIN.md` and the hard rule in CLAUDE.md); the v0.107.1 C#
// decompilation is. The scraper was retired because it could only ever
// *add* variants: it re-added the deliberately removed `DrumOfBattle` and
// dropped every C#-sourced variant added since, so a rerun reverted
// v0.107.1 content work. The old header asked the next person to
// re-apply those deletions by hand after regenerating; deleting the
// scraper makes that impossible to get wrong.
//
// The `python_name` arm still pins the upper-snake-case string the parity
// exporter emits. For variants that predate v0.107.1 that string equals
// `PowerId.<v>.name` from the old Python source; for C#-only powers it is
// the C# id in upper-snake-case. Keep the two in sync by hand.
//
// Listener bodies stay TODO / no-op for the player-side long tail;
// the per-character card-pool beads (P6-3..P6-7) wire concrete
// behavior where needed. Variants not yet wired fall through to
// the passthrough arms in `crate::hooks`.

/// Discriminant for [`crate::Power`]. Names mirror Python's
/// `PowerId.<NAME>` exactly so the parity exporter's
/// [`python_name`](Self::python_name) returns the same string Python
/// emits in `combat_state_to_bridge_state`.
///
/// `#[non_exhaustive]` so adding a variant later does not break
/// downstream `match PowerKind { ... }` arms outside this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum PowerKind {
    /// `PowerId.ACCURACY` in Python.
    Accuracy,
    /// `PowerId.CALCIFY` in Python.
    Calcify,
    /// `PowerId.COLOSSUS` in Python.
    Colossus,
    /// `PowerId.CONQUEROR` in Python.
    Conqueror,
    /// `PowerId.COVERED` in Python.
    Covered,
    /// `PowerId.DIAMOND_DIADEM` in Python.
    DiamondDiadem,
    /// `PowerId.DOUBLE_DAMAGE` in Python.
    DoubleDamage,
    /// `PowerId.FLANKING` in Python.
    Flanking,
    /// `PowerId.FLUTTER` in Python.
    Flutter,
    /// `PowerId.GIGANTIFICATION` in Python.
    Gigantification,
    /// `PowerId.GUARDED` in Python.
    Guarded,
    /// `PowerId.HANG` in Python.
    Hang,
    /// `PowerId.INTERCEPT` in Python.
    Intercept,
    /// `PowerId.KNOCKDOWN` in Python.
    Knockdown,
    /// `PowerId.LEADERSHIP` in Python.
    Leadership,
    /// `PowerId.LETHALITY` in Python.
    Lethality,
    /// `PowerId.PHANTOM_BLADES` in Python.
    PhantomBlades,
    /// `PowerId.SHRINK` in Python.
    Shrink,
    /// `PowerId.SLOW` in Python.
    Slow,
    /// `PowerId.SOAR` in Python.
    Soar,
    /// `PowerId.STRENGTH` in Python.
    Strength,
    /// `PowerId.SURROUNDED` in Python.
    Surrounded,
    /// `PowerId.TANK` in Python.
    Tank,
    /// `PowerId.TRACKING` in Python.
    Tracking,
    /// `PowerId.VIGOR` in Python.
    Vigor,
    /// `PowerId.VULNERABLE` in Python.
    Vulnerable,
    /// `PowerId.WEAK` in Python.
    Weak,
    /// `PowerId.DEXTERITY` in Python.
    Dexterity,
    /// `PowerId.FASTEN` in Python.
    Fasten,
    /// `PowerId.FRAIL` in Python.
    Frail,
    /// `PowerId.NO_BLOCK` in Python.
    NoBlock,
    /// `PowerId.SHADOWMELD` in Python.
    Shadowmeld,
    /// `PowerId.UNMOVABLE` in Python.
    Unmovable,
    /// `PowerId.AGGRESSION` in Python.
    Aggression,
    /// `PowerId.BIASED_COGNITION` in Python.
    BiasedCognition,
    /// `PowerId.BLUR` in Python.
    Blur,
    /// `PowerId.COOLANT` in Python.
    Coolant,
    /// `PowerId.COUNTDOWN` in Python.
    Countdown,
    /// `PowerId.CRIMSON_MANTLE` in Python.
    CrimsonMantle,
    /// `PowerId.DEMON_FORM` in Python.
    DemonForm,
    /// `PowerId.ENTROPY` in Python.
    Entropy,
    /// `PowerId.FERAL` in Python.
    Feral,
    /// `PowerId.FURNACE` in Python.
    Furnace,
    /// `PowerId.LOOP` in Python.
    Loop,
    /// `PowerId.NEUROSURGE` in Python.
    Neurosurge,
    /// `PowerId.NOXIOUS_FUMES` in Python.
    NoxiousFumes,
    /// `PowerId.PLATING` in Python.
    Plating,
    /// `PowerId.POISON` in Python.
    Poison,
    /// `PowerId.PREP_TIME` in Python.
    PrepTime,
    /// `PowerId.RAMPART` in Python.
    Rampart,
    /// `PowerId.ROLLING_BOULDER` in Python.
    RollingBoulder,
    /// `PowerId.SANDPIT` in Python.
    Sandpit,
    /// `PowerId.SHADOW_STEP` in Python.
    ShadowStep,
    /// `PowerId.SUMMON_NEXT_TURN` in Python.
    SummonNextTurn,
    /// `PowerId.WRAITH_FORM` in Python.
    WraithForm,
    /// `PowerId.BATTLEWORN_DUMMY_TIME_LIMIT` in Python.
    BattlewornDummyTimeLimit,
    /// `PowerId.CONSTRICT` in Python.
    Constrict,
    /// `PowerId.CONSUMING_SHADOW` in Python.
    ConsumingShadow,
    /// `PowerId.DEBILITATE` in Python.
    Debilitate,
    /// `PowerId.DEMISE` in Python.
    Demise,
    /// `PowerId.ESCAPE_ARTIST` in Python.
    EscapeArtist,
    /// `PowerId.GRAPPLE` in Python.
    Grapple,
    /// `PowerId.HATCH` in Python.
    Hatch,
    /// `PowerId.HIGH_VOLTAGE` in Python.
    HighVoltage,
    /// `PowerId.INTANGIBLE` in Python.
    Intangible,
    /// `PowerId.MAGIC_BOMB` in Python.
    MagicBomb,
    /// `PowerId.NEMESIS` in Python.
    Nemesis,
    /// `PowerId.NO_DRAW` in Python.
    NoDraw,
    /// `PowerId.REBOUND` in Python.
    Rebound,
    /// `PowerId.REGEN` in Python.
    Regen,
    /// `PowerId.RETAIN_HAND` in Python.
    RetainHand,
    /// `PowerId.RITUAL` in Python.
    Ritual,
    /// `PowerId.SIC_EM` in Python.
    SicEm,
    /// `PowerId.TEMPORARY_DEXTERITY` in Python.
    TemporaryDexterity,
    /// `PowerId.TEMPORARY_FOCUS` in Python.
    TemporaryFocus,
    /// `PowerId.TEMPORARY_STRENGTH` in Python.
    TemporaryStrength,
    /// `PowerId.TERRITORIAL` in Python.
    Territorial,
    /// `PowerId.WELL_LAID_PLANS` in Python.
    WellLaidPlans,
    /// `PowerId.AFTERIMAGE` in Python.
    Afterimage,
    /// `PowerId.ARSENAL` in Python.
    Arsenal,
    /// `PowerId.BLACK_HOLE` in Python.
    BlackHole,
    /// `PowerId.BLADE_OF_INK` in Python.
    BladeOfInk,
    /// `PowerId.BURST` in Python.
    Burst,
    /// `PowerId.CALAMITY` in Python.
    Calamity,
    /// `PowerId.CHAINS_OF_BINDING` in Python.
    ChainsOfBinding,
    /// `PowerId.CURL_UP` in Python.
    CurlUp,
    /// `PowerId.DANSE_MACABRE` in Python.
    DanseMacabre,
    /// `PowerId.DEVOUR_LIFE` in Python.
    DevourLife,
    /// `PowerId.DUPLICATION` in Python.
    Duplication,
    /// `PowerId.ECHO_FORM` in Python.
    EchoForm,
    /// `PowerId.ENRAGE` in Python.
    Enrage,
    /// `PowerId.FREE_ATTACK` in Python.
    FreeAttack,
    /// `PowerId.FREE` in Python.
    Free,
    /// `PowerId.FREE_SKILL` in Python.
    FreeSkill,
    /// `PowerId.GALVANIC` in Python.
    Galvanic,
    /// `PowerId.GRAVITY` in Python.
    Gravity,
    /// `PowerId.HAUNT` in Python.
    Haunt,
    /// `PowerId.HELLRAISER` in Python.
    Hellraiser,
    /// `PowerId.JUGGLING` in Python.
    Juggling,
    /// `PowerId.MASTER_PLANNER` in Python.
    MasterPlanner,
    /// `PowerId.MONOLOGUE` in Python.
    Monologue,
    /// `PowerId.OBLIVION` in Python.
    Oblivion,
    /// `PowerId.ONE_TWO_PUNCH` in Python.
    OneTwoPunch,
    /// `PowerId.PAINFUL_STABS` in Python.
    PainfulStabs,
    /// `PowerId.PANACHE` in Python.
    Panache,
    /// `PowerId.RAGE` in Python.
    Rage,
    /// `PowerId.RUPTURE` in Python.
    Rupture,
    /// `PowerId.SERPENT_FORM` in Python.
    SerpentForm,
    /// `PowerId.SIGNAL_BOOST` in Python.
    SignalBoost,
    /// `PowerId.SKITTISH` in Python.
    Skittish,
    /// `PowerId.SLOTH` in Python.
    Sloth,
    /// `PowerId.SMOGGY` in Python.
    Smoggy,
    /// `PowerId.SNEAKY` in Python.
    Sneaky,
    /// `PowerId.SPIRIT_OF_ASH` in Python.
    SpiritOfAsh,
    /// `PowerId.STORM` in Python.
    Storm,
    /// `PowerId.STRANGLE` in Python.
    Strangle,
    /// `PowerId.SUBROUTINE` in Python.
    Subroutine,
    /// `PowerId.SUCK` in Python.
    Suck,
    /// `PowerId.TAG_TEAM` in Python.
    TagTeam,
    /// `PowerId.TENDER` in Python.
    Tender,
    /// `PowerId.THE_SEALED_THRONE` in Python.
    TheSealedThrone,
    /// `PowerId.VEILPIERCER` in Python.
    Veilpiercer,
    /// `PowerId.VOID_FORM` in Python.
    VoidForm,
    /// `PowerId.ASLEEP` in Python.
    Asleep,
    /// `PowerId.FLAME_BARRIER` in Python.
    FlameBarrier,
    /// `PowerId.HARDENED_SHELL` in Python.
    HardenedShell,
    /// `PowerId.INFERNO` in Python.
    Inferno,
    /// `PowerId.PERSONAL_HIVE` in Python.
    PersonalHive,
    /// `PowerId.PLOW` in Python.
    Plow,
    /// `PowerId.REFLECT` in Python.
    Reflect,
    /// `PowerId.SHRIEK` in Python.
    Shriek,
    /// `PowerId.SLIPPERY` in Python.
    Slippery,
    /// `PowerId.SLUMBER` in Python.
    Slumber,
    /// `PowerId.THE_GAMBIT` in Python.
    TheGambit,
    /// `PowerId.THORNS` in Python.
    Thorns,
    /// `PowerId.VITAL_SPARK` in Python.
    VitalSpark,
    /// `PowerId.AUTOMATION` in Python.
    Automation,
    /// `PowerId.CONFUSED` in Python.
    Confused,
    /// `PowerId.CORROSIVE_WAVE` in Python.
    CorrosiveWave,
    /// `PowerId.DARK_EMBRACE` in Python.
    DarkEmbrace,
    /// `PowerId.FEEL_NO_PAIN` in Python.
    FeelNoPain,
    /// `PowerId.HEX` in Python.
    Hex,
    /// `PowerId.ITERATION` in Python.
    Iteration,
    /// `PowerId.PAGESTORM` in Python.
    Pagestorm,
    /// `PowerId.PILLAR_OF_CREATION` in Python.
    PillarOfCreation,
    /// `PowerId.RINGING` in Python.
    Ringing,
    /// `PowerId.SMOKESTACK` in Python.
    Smokestack,
    /// `PowerId.SPEEDSTER` in Python.
    Speedster,
    /// `PowerId.SWORD_SAGE` in Python.
    SwordSage,
    /// `PowerId.TANGLED` in Python.
    Tangled,
    /// `PowerId.TRASH_TO_TREASURE` in Python.
    TrashToTreasure,
    /// `PowerId.BLOCK_NEXT_TURN` in Python.
    BlockNextTurn,
    /// `PowerId.SELF_FORMING_CLAY` in Python.
    SelfFormingClay,
    /// `PowerId.TORIC_TOUGHNESS` in Python.
    ToricToughness,
    /// `PowerId.CLARITY` in Python.
    Clarity,
    /// `PowerId.CORRUPTION` in Python.
    Corruption,
    /// `PowerId.CURIOUS` in Python.
    Curious,
    /// `PowerId.DEMESNE` in Python.
    Demesne,
    /// `PowerId.DRAW_CARDS_NEXT_TURN` in Python.
    DrawCardsNextTurn,
    /// `PowerId.FRIENDSHIP` in Python.
    Friendship,
    /// `PowerId.MACHINE_LEARNING` in Python.
    MachineLearning,
    /// `PowerId.MIND_ROT` in Python.
    MindRot,
    /// `PowerId.ORBIT` in Python.
    Orbit,
    /// `PowerId.PALE_BLUE_DOT` in Python.
    PaleBlueDot,
    /// `PowerId.PYRE` in Python.
    Pyre,
    /// `PowerId.TOOLS_OF_THE_TRADE` in Python.
    ToolsOfTheTrade,
    /// `PowerId.TYRANNY` in Python.
    Tyranny,
    /// `PowerId.WASTE_AWAY` in Python.
    WasteAway,
    /// `PowerId.ADAPTABLE` in Python.
    Adaptable,
    /// `PowerId.CRAB_RAGE` in Python.
    CrabRage,
    /// `PowerId.DAMPEN` in Python.
    Dampen,
    /// `PowerId.DIE_FOR_YOU` in Python.
    DieForYou,
    /// `PowerId.DOOR_REVIVAL` in Python.
    DoorRevival,
    /// `PowerId.ILLUSION` in Python.
    Illusion,
    /// `PowerId.INFESTED` in Python.
    Infested,
    /// `PowerId.MINION` in Python.
    Minion,
    /// `PowerId.POSSESS_SPEED` in Python.
    PossessSpeed,
    /// `PowerId.POSSESS_STRENGTH` in Python.
    PossessStrength,
    /// `PowerId.RAVENOUS` in Python.
    Ravenous,
    /// `PowerId.REATTACH` in Python.
    Reattach,
    /// `PowerId.STEAM_ERUPTION` in Python.
    SteamEruption,
    /// `PowerId.STOCK` in Python.
    Stock,
    /// `PowerId.SURPRISE` in Python.
    Surprise,
    /// `PowerId.ACCELERANT` in Python.
    Accelerant,
    /// `PowerId.ANTICIPATE` in Python.
    Anticipate,
    /// `PowerId.ARTIFACT` in Python.
    Artifact,
    /// `PowerId.BACK_ATTACK_LEFT` in Python.
    BackAttackLeft,
    /// `PowerId.BACK_ATTACK_RIGHT` in Python.
    BackAttackRight,
    /// `PowerId.BARRICADE` in Python.
    Barricade,
    /// `PowerId.BEACON_OF_HOPE` in Python.
    BeaconOfHope,
    /// `PowerId.BUFFER` in Python.
    Buffer,
    /// `PowerId.BURROWED` in Python.
    Burrowed,
    /// `PowerId.CALL_OF_THE_VOID` in Python.
    CallOfTheVoid,
    /// `PowerId.CHILD_OF_THE_STARS` in Python.
    ChildOfTheStars,
    /// `PowerId.COORDINATE` in Python.
    Coordinate,
    /// `PowerId.CREATIVE_AI` in Python.
    CreativeAi,
    /// `PowerId.CRUELTY` in Python.
    Cruelty,
    /// `PowerId.CRUSH_UNDER` in Python.
    CrushUnder,
    /// `PowerId.DARK_SHACKLES` in Python.
    DarkShackles,
    /// `PowerId.DISINTEGRATION` in Python.
    Disintegration,
    /// `PowerId.DOOM` in Python.
    Doom,
    /// `PowerId.DYING_STAR` in Python.
    DyingStar,
    /// `PowerId.ENERGY_NEXT_TURN` in Python.
    EnergyNextTurn,
    /// `PowerId.ENFEEBLING_TOUCH` in Python.
    EnfeeblingTouch,
    /// `PowerId.ENVENOM` in Python.
    Envenom,
    /// `PowerId.FAN_OF_KNIVES` in Python.
    FanOfKnives,
    /// `PowerId.FEEDING_FRENZY` in Python.
    FeedingFrenzy,
    /// `PowerId.FLEX_POTION` in Python.
    FlexPotion,
    /// `PowerId.FOCUS` in Python.
    Focus,
    /// `PowerId.FOCUSED_STRIKE` in Python.
    FocusedStrike,
    /// `PowerId.FORBIDDEN_GRIMOIRE` in Python.
    ForbiddenGrimoire,
    /// `PowerId.FOREGONE_CONCLUSION` in Python.
    ForegoneConclusion,
    /// `PowerId.GENESIS` in Python.
    Genesis,
    /// `PowerId.HAILSTORM` in Python.
    Hailstorm,
    /// `PowerId.HAMMER_TIME` in Python.
    HammerTime,
    /// `PowerId.HARD_TO_KILL` in Python.
    HardToKill,
    /// `PowerId.HEIST` in Python.
    Heist,
    /// `PowerId.HELICAL_DART` in Python.
    HelicalDart,
    /// `PowerId.HELLO_WORLD` in Python.
    HelloWorld,
    /// `PowerId.HOTFIX` in Python.
    Hotfix,
    /// `PowerId.IMBALANCED` in Python.
    Imbalanced,
    /// `PowerId.IMPROVEMENT` in Python.
    Improvement,
    /// `PowerId.INFINITE_BLADES` in Python.
    InfiniteBlades,
    /// `PowerId.JUGGERNAUT` in Python.
    Juggernaut,
    /// `PowerId.LIGHTNING_ROD` in Python.
    LightningRod,
    /// `PowerId.MANGLE` in Python.
    Mangle,
    /// `PowerId.MAYHEM` in Python.
    Mayhem,
    /// `PowerId.MONARCHS_GAZE` in Python.
    MonarchsGaze,
    /// `PowerId.MONARCHS_GAZE_STRENGTH_DOWN` in Python.
    MonarchsGazeStrengthDown,
    /// `PowerId.NECRO_MASTERY` in Python.
    NecroMastery,
    /// `PowerId.NIGHTMARE` in Python.
    Nightmare,
    /// `PowerId.NOSTALGIA` in Python.
    Nostalgia,
    /// `PowerId.OUTBREAK` in Python.
    Outbreak,
    /// `PowerId.PAPER_CUTS` in Python.
    PaperCuts,
    /// `PowerId.PARRY` in Python.
    Parry,
    /// `PowerId.PIERCING_WAIL` in Python.
    PiercingWail,
    /// `PowerId.RADIANCE` in Python.
    Radiance,
    /// `PowerId.REAPER_FORM` in Python.
    ReaperForm,
    /// `PowerId.REPTILE_TRINKET` in Python.
    ReptileTrinket,
    /// `PowerId.ROYALTIES` in Python.
    Royalties,
    /// `PowerId.SEEKING_EDGE` in Python.
    SeekingEdge,
    /// `PowerId.SENTRY_MODE` in Python.
    SentryMode,
    /// `PowerId.SETUP_STRIKE` in Python.
    SetupStrike,
    /// `PowerId.SHACKLING_POTION` in Python.
    ShacklingPotion,
    /// `PowerId.SHROUD` in Python.
    Shroud,
    /// `PowerId.SLEIGHT_OF_FLESH` in Python.
    SleightOfFlesh,
    /// `PowerId.SPECTRUM_SHIFT` in Python.
    SpectrumShift,
    /// `PowerId.SPEED_POTION` in Python.
    SpeedPotion,
    /// `PowerId.SPINNER` in Python.
    Spinner,
    /// `PowerId.STAMPEDE` in Python.
    Stampede,
    /// `PowerId.STAR_NEXT_TURN` in Python.
    StarNextTurn,
    /// `PowerId.STRATAGEM` in Python.
    Stratagem,
    /// `PowerId.SWIPE` in Python.
    Swipe,
    /// `PowerId.SYNCHRONIZE` in Python.
    Synchronize,
    /// `PowerId.THE_BOMB` in Python.
    TheBomb,
    /// `PowerId.THE_HUNT` in Python.
    TheHunt,
    /// `PowerId.THIEVERY` in Python.
    Thievery,
    /// `PowerId.THUNDER` in Python.
    Thunder,
    /// `PowerId.VICIOUS` in Python.
    Vicious,
    /// `PowerId.ELECTRODYNAMICS` in Python.
    Electrodynamics,
    /// `PowerId.METALLICIZE` in Python.
    Metallicize,
    /// `PowerId.ENTANGLED` in Python.
    Entangled,
    /// `PowerId.GRAPPLE_POWER` in Python.
    GrapplePower,
    /// `PowerId.MANGLE_POWER` in Python.
    ManglePower,
    /// `PowerId.FREE_POWER` in Python.
    FreePower,
    /// `PowerId.GENERIC` in Python.
    Generic,
    /// C# `NoEnergyGainPower` (no Python `PowerId` counterpart at the pin).
    /// Suppresses energy gain while active. Added for ExpectAFight
    /// (bead `sts2-rs-o7d5d`).
    NoEnergyGain,
    // `Hunger` / `Scrutiny` / `Grasp` (Doormaker's phase-marker powers, with
    // the Devoured/Weighted affliction interactions) were removed in v0.107.1
    // alongside the Doormaker boss (bead `…wb8lk.2`).
    /// C# `BorrowedTimePower` — Necrobinder Borrowed Time's self-applied
    /// debuff: `TryModifyEnergyCostInCombat` adds `Amount` to EVERY card the
    /// owner plays this turn, and `AfterTurnEnd` removes it on the owner's
    /// turn end. Wired in `crate::hooks` (`modify_card_cost` + `after_turn_end`
    /// arms). Added manually for the C#-first Necrobinder port; no Python
    /// `PowerId.BORROWED_TIME` counterpart (the Python sim never ported the
    /// card). C#: `.../Models/Powers/BorrowedTimePower.cs`.
    BorrowedTime,
    /// C# `SicEmPower` — Necrobinder Sic Em's enemy debuff (Counter). When the
    /// player's Osty deals damage to the marked enemy, the Osty is summoned/grown
    /// by `Amount` (`OstyCmd.Summon`); removed at the owner's turn end. Wired in
    /// `crate::hooks` (`dispatch_after_damage_given` Osty-dealer arm +
    /// `after_turn_end` removal). Added manually for the C#-first Necrobinder
    /// port; no Python `PowerId.SIC_EM` counterpart. C#:
    /// `.../Models/Powers/SicEmPower.cs`.
    SicEmPower,
    /// C# `DevourLifePower` — Necrobinder Devour Life's self buff (Counter).
    /// Whenever the owner plays a Soul card, the Osty is summoned/grown by
    /// `Amount` (`OstyCmd.Summon`). Persists until combat end (no removal hook).
    /// Wired in `crate::hooks` (`dispatch_after_card_played` arm). Added manually
    /// for the C#-first Necrobinder port; no Python `PowerId.DEVOUR_LIFE`
    /// counterpart. C#: `.../Models/Powers/DevourLifePower.cs`.
    DevourLifePower,
    /// C# `TaintedPower` (v0.107.1) — the in-combat "Tainted" keyword
    /// (Debuff, Counter) on the creature that played a Tainted card. Adds
    /// `Amount` ADDITIVE damage to powered attacks whose TARGET is the owner;
    /// auto-removes at the enemy `AfterSideTurnEnd`. Applied by
    /// `VitalSparkPower` (Infested Prism) when a Tainted Skill card is played.
    /// Added manually for the C#-first v0.107.1 content port; no Python
    /// `PowerId.TAINTED` counterpart. C#:
    /// `.../Models/Powers/TaintedPower.cs`.
    Tainted,
    /// C# `WitheringPresencePower` (v0.107.1) — the Aeonglass Act-3 boss
    /// applies one instance to ITSELF on spawn (Buff, Counter, Instanced;
    /// `Aeonglass.cs:78-83` — owner == target == the monster, only the
    /// power's `Target` field points at the player). `amount` stays a
    /// constant 6 (the serialized stack C# reports); the per-instance
    /// `counter` counts cards played toward the next Wither, and on reaching
    /// `amount` a `Wither` is added to the player's hand (hand-capped →
    /// discard) and the counter resets. The freshly-added `Wither` is
    /// `FakeUpgrade`d to the owning monster's `MonsterAi::counter`
    /// (`WitherUpgradeCount`, bumped by the `IncreasingIntensity` move).
    /// Added manually for the C#-first v0.107.1 content port; no Python
    /// counterpart. C#: `.../Models/Powers/WitheringPresencePower.cs`. Bead
    /// `sts2-rs-v0107-content-port-wb8lk.2`.
    WitheringPresence,
    // ── v0.111.0-only (GH #794). Not in [`PowerKind::all`], which stays the
    // v0.107.1 universe (wire parser, registry names, affliction coverage);
    // see [`PowerKind::V0111_ONLY`] / [`PowerKind::all_for`].
    /// v0.111.0 `AmbergrisPower` (`Powers/AmbergrisPower.cs`): hidden Counter
    /// buff applied by the Ambergris potion. While `Amount > 0` the owner takes
    /// an extra turn; `AfterTakingExtraTurn` decrements it.
    Ambergris,
    /// v0.111.0 `CacophonyPower` (`Powers/CacophonyPower.cs`): Instanced
    /// Counter buff (MP-only Cacophony card). Every 33 cards drawn it deals
    /// `Amount` unpowered damage to a random hittable enemy. `counter` holds the
    /// cards drawn since the last trigger.
    Cacophony,
    /// v0.111.0 `ConcoctPower` (`Powers/ConcoctPower.cs`): the owner's powered
    /// attacks that deal unblocked damage apply `Amount` Poison; removed at the
    /// end of the opposing side's turn (MP-only Concoct card).
    Concoct,
    /// v0.111.0 `FadePower : TemporaryDexterityPower` (`Powers/FadePower.cs`,
    /// MP-only Fade card): +`Amount` Dexterity until the owner's turn end.
    Fade,
    /// v0.111.0 `HibernatePower` (`Powers/HibernatePower.cs`): marker Counter
    /// read by FrostOrb; decrements at the owner's player-turn start.
    Hibernate,
    /// v0.111.0 `HyperbeamFocusDownPower : TemporaryFocusPower` with
    /// `IsPositive => false` (`Powers/HyperbeamFocusDownPower.cs`): the
    /// reworked Hyperbeam's -`Amount` Focus until the owner's turn end. A
    /// Debuff (`TemporaryFocusPower.Type`).
    HyperbeamFocusDown,
    /// v0.111.0 `ImitationLearningPower` (`Powers/ImitationLearningPower.cs`):
    /// Instanced, bound to another player whose Power plays it copies (MP-only).
    ImitationLearning,
    /// v0.111.0 `OneForAllPower` (`Powers/OneForAllPower.cs`): +`Amount`
    /// damage to the owner's powered attacks from cards played for 0 energy
    /// (not X-cost).
    OneForAll,
    /// v0.111.0 `SoulboundPower` (`Powers/SoulboundPower.cs`): when the
    /// applier generates a Soul, add `Amount` Souls to the owner's draw pile.
    Soulbound,
    /// v0.111.0 `UnderworldPower` (`Powers/UnderworldPower.cs`): allies' (not
    /// the owner's, not the owner's pet's) powered attacks apply Doom equal to
    /// the damage times `Amount`; removed at the end of the enemy turn.
    Underworld,
    // ── v0.113.0-only: The Fool (`Models.Powers`). Not in [`PowerKind::all`];
    // see [`PowerKind::V0113_ONLY`] / [`PowerKind::all_for`].
    /// v0.113.0 `AimPower`: Counter buff, +`Amount` to the owner's powered `FromArrow` attacks; consumed by the first such attack command (Vigor-like).
    Aim,
    /// v0.113.0 `AllIsWellPower`: gain `Amount` block whenever the owner generates a Sadness.
    AllIsWell,
    /// v0.113.0 `ArcheryPower`: gain `Amount` Aim at each owner side turn start.
    Archery,
    /// v0.113.0 `ChoBarbsPower : TemporaryThornsPower`: +`Amount` Thorns until the owner's next turn.
    ChoBarbs,
    /// v0.113.0 `CompostPower` (MP-only Compost card): teammates gain energy when an enemy dies.
    Compost,
    /// v0.113.0 `ConductorsBatonPower` (ConductorsBaton relic): take x0.75 powered-attack damage until the owner's next turn start. Single.
    ConductorsBaton,
    /// v0.113.0 `CoveringFirePower` (MP-only): teammates gain block when the owner shoots an arrow.
    CoveringFire,
    /// v0.113.0 `DislikeYouPower`: Debuff, target takes +`Amount`% from powered arrow attacks; removed at player turn end.
    DislikeYou,
    /// v0.113.0 `EcstasyPower`: player turn start, transform up to `Amount` Sadness in hand into Joy.
    Ecstasy,
    /// v0.113.0 `EncorePower`: Debuff, only `Amount` more cards may be played this turn.
    Encore,
    /// v0.113.0 `EnnuiPower`: +`Amount` hand draw; before hand draw add `Cards` Sadness.
    Ennui,
    /// v0.113.0 `EnsemblePower` (MP-only): cards cost 0 while active, decremented when a teammate plays.
    Ensemble,
    /// v0.113.0 `ExplosiveArrowsPower`: when the owner's arrow impales, deal `Amount` to all enemies.
    ExplosiveArrows,
    /// v0.113.0 `FreeCardPower`: the owner's next `Amount` card(s) in hand/play cost 0 (applied by TunePower).
    FreeCard,
    /// v0.113.0 `HiddenRootsPower`: gain `Amount` Stretch at each owner side turn start.
    HiddenRoots,
    /// v0.113.0 `HoningPower`: the owner's powered arrow attacks deal +`Amount`%.
    Honing,
    /// v0.113.0 `ImpaledPower`: InstancedPerApplier arrow tracker on an enemy (hit counter 3; dislodges an arrow every 3 powered hits from the applier).
    Impaled,
    /// v0.113.0 `KiVersePower`: the owner's next melody (Wah/Cho/Ki) card plays an extra time.
    KiVerse,
    /// v0.113.0 `LostLutePower`: first Wah/Cho/Ki of the turn autoplays `Amount` cards from the draw pile per new note.
    LostLute,
    /// v0.113.0 `LullabyPower`: once per turn after Wah, Cho and Ki were all played, gain `Amount` block.
    Lullaby,
    /// v0.113.0 `MotifPower`: first Wah/Cho/Ki of the turn draws `Amount` cards per new note.
    Motif,
    /// v0.113.0 `MythicFormPower`: before hand draw add `Amount` random tarot cards.
    MythicForm,
    /// v0.113.0 `QuietusPower`: the next powered arrow attack deals x`Amount` and its arrow is destroyed after impaling.
    Quietus,
    /// v0.113.0 `SerratedArrowsPower`: +`Amount`% powered damage against Impaled targets.
    SerratedArrows,
    /// v0.113.0 `ShowOffPower`: Instanced; on a fatal enemy kill raise the ShowOff card's Aim.
    ShowOff,
    /// v0.113.0 `SimpletonPower`: post-play phase autoplays the cheapest playable card(s) in hand.
    Simpleton,
    /// v0.113.0 `SongPathPower`: first Cho each turn gains `Amount` block.
    SongPath,
    /// v0.113.0 `StretchPower`: Counter buff, +`Amount` block to the next block-gaining card play; then consumed.
    Stretch,
    /// v0.113.0 `TunePower`: player turn end, if a Ki was played this turn gain `Amount` FreeCard.
    Tune,
    /// v0.113.0 `ViolentChordPower`: first Wah each turn deals `Amount` to all enemies.
    ViolentChord,
    /// v0.113.0 `YahPower`: once per turn when the owner's arrow is dislodged gain `Amount` energy.
    Yah,
    /// v0.113.0 `GodByrdPower` (GodByrd card, Byrdine relic): Counter buff;
    /// at the owner's next player turn start (`AmountOnTurnStart != 0`) deal
    /// `Amount` Unpowered to every hittable enemy, then remove itself. Not a
    /// Fool power: appended after the Fool block so their indices hold.
    GodByrd,
}

impl PowerKind {
    /// Python `PowerId.<self>.name` — the canonical parity-exporter
    /// string. Used by [`crate::parity`].
    #[must_use]
    pub const fn python_name(self) -> &'static str {
        match self {
            Self::Accuracy => "ACCURACY",
            Self::Calcify => "CALCIFY",
            Self::Colossus => "COLOSSUS",
            Self::Conqueror => "CONQUEROR",
            Self::Covered => "COVERED",
            Self::DiamondDiadem => "DIAMOND_DIADEM",
            Self::DoubleDamage => "DOUBLE_DAMAGE",
            Self::Flanking => "FLANKING",
            Self::Flutter => "FLUTTER",
            Self::Gigantification => "GIGANTIFICATION",
            Self::Guarded => "GUARDED",
            Self::Hang => "HANG",
            Self::Intercept => "INTERCEPT",
            Self::Knockdown => "KNOCKDOWN",
            Self::Leadership => "LEADERSHIP",
            Self::Lethality => "LETHALITY",
            Self::PhantomBlades => "PHANTOM_BLADES",
            Self::Shrink => "SHRINK",
            Self::Slow => "SLOW",
            Self::Soar => "SOAR",
            Self::Strength => "STRENGTH",
            Self::Surrounded => "SURROUNDED",
            Self::Tank => "TANK",
            Self::Tracking => "TRACKING",
            Self::Vigor => "VIGOR",
            Self::Vulnerable => "VULNERABLE",
            Self::Weak => "WEAK",
            Self::Dexterity => "DEXTERITY",
            Self::Fasten => "FASTEN",
            Self::Frail => "FRAIL",
            Self::NoBlock => "NO_BLOCK",
            Self::Shadowmeld => "SHADOWMELD",
            Self::Unmovable => "UNMOVABLE",
            Self::Aggression => "AGGRESSION",
            Self::BiasedCognition => "BIASED_COGNITION",
            Self::Blur => "BLUR",
            Self::Coolant => "COOLANT",
            Self::Countdown => "COUNTDOWN",
            Self::CrimsonMantle => "CRIMSON_MANTLE",
            Self::DemonForm => "DEMON_FORM",
            Self::Entropy => "ENTROPY",
            Self::Feral => "FERAL",
            Self::Furnace => "FURNACE",
            Self::Loop => "LOOP",
            Self::Neurosurge => "NEUROSURGE",
            Self::NoxiousFumes => "NOXIOUS_FUMES",
            Self::Plating => "PLATING",
            Self::Poison => "POISON",
            Self::PrepTime => "PREP_TIME",
            Self::Rampart => "RAMPART",
            Self::RollingBoulder => "ROLLING_BOULDER",
            Self::Sandpit => "SANDPIT",
            Self::ShadowStep => "SHADOW_STEP",
            Self::SummonNextTurn => "SUMMON_NEXT_TURN",
            Self::WraithForm => "WRAITH_FORM",
            Self::BattlewornDummyTimeLimit => "BATTLEWORN_DUMMY_TIME_LIMIT",
            Self::Constrict => "CONSTRICT",
            Self::ConsumingShadow => "CONSUMING_SHADOW",
            Self::Debilitate => "DEBILITATE",
            Self::Demise => "DEMISE",
            Self::EscapeArtist => "ESCAPE_ARTIST",
            Self::Grapple => "GRAPPLE",
            Self::Hatch => "HATCH",
            Self::HighVoltage => "HIGH_VOLTAGE",
            Self::Intangible => "INTANGIBLE",
            Self::MagicBomb => "MAGIC_BOMB",
            Self::Nemesis => "NEMESIS",
            Self::NoDraw => "NO_DRAW",
            Self::Rebound => "REBOUND",
            Self::Regen => "REGEN",
            Self::RetainHand => "RETAIN_HAND",
            Self::Ritual => "RITUAL",
            Self::SicEm => "SIC_EM",
            Self::TemporaryDexterity => "TEMPORARY_DEXTERITY",
            Self::TemporaryFocus => "TEMPORARY_FOCUS",
            Self::TemporaryStrength => "TEMPORARY_STRENGTH",
            Self::Territorial => "TERRITORIAL",
            Self::WellLaidPlans => "WELL_LAID_PLANS",
            Self::Afterimage => "AFTERIMAGE",
            Self::Arsenal => "ARSENAL",
            Self::BlackHole => "BLACK_HOLE",
            Self::BladeOfInk => "BLADE_OF_INK",
            Self::Burst => "BURST",
            Self::Calamity => "CALAMITY",
            Self::ChainsOfBinding => "CHAINS_OF_BINDING",
            Self::CurlUp => "CURL_UP",
            Self::DanseMacabre => "DANSE_MACABRE",
            Self::DevourLife => "DEVOUR_LIFE",
            Self::Duplication => "DUPLICATION",
            Self::EchoForm => "ECHO_FORM",
            Self::Enrage => "ENRAGE",
            Self::FreeAttack => "FREE_ATTACK",
            Self::Free => "FREE",
            Self::FreeSkill => "FREE_SKILL",
            Self::Galvanic => "GALVANIC",
            Self::Gravity => "GRAVITY",
            Self::Haunt => "HAUNT",
            Self::Hellraiser => "HELLRAISER",
            Self::Juggling => "JUGGLING",
            Self::MasterPlanner => "MASTER_PLANNER",
            Self::Monologue => "MONOLOGUE",
            Self::Oblivion => "OBLIVION",
            Self::OneTwoPunch => "ONE_TWO_PUNCH",
            Self::PainfulStabs => "PAINFUL_STABS",
            Self::Panache => "PANACHE",
            Self::Rage => "RAGE",
            Self::Rupture => "RUPTURE",
            Self::SerpentForm => "SERPENT_FORM",
            Self::SignalBoost => "SIGNAL_BOOST",
            Self::Skittish => "SKITTISH",
            Self::Sloth => "SLOTH",
            Self::Smoggy => "SMOGGY",
            Self::Sneaky => "SNEAKY",
            Self::SpiritOfAsh => "SPIRIT_OF_ASH",
            Self::Storm => "STORM",
            Self::Strangle => "STRANGLE",
            Self::Subroutine => "SUBROUTINE",
            Self::Suck => "SUCK",
            Self::TagTeam => "TAG_TEAM",
            Self::Tender => "TENDER",
            Self::TheSealedThrone => "THE_SEALED_THRONE",
            Self::Veilpiercer => "VEILPIERCER",
            Self::VoidForm => "VOID_FORM",
            Self::Asleep => "ASLEEP",
            Self::FlameBarrier => "FLAME_BARRIER",
            Self::HardenedShell => "HARDENED_SHELL",
            Self::Inferno => "INFERNO",
            Self::PersonalHive => "PERSONAL_HIVE",
            Self::Plow => "PLOW",
            Self::Reflect => "REFLECT",
            Self::Shriek => "SHRIEK",
            Self::Slippery => "SLIPPERY",
            Self::Slumber => "SLUMBER",
            Self::TheGambit => "THE_GAMBIT",
            Self::Thorns => "THORNS",
            Self::VitalSpark => "VITAL_SPARK",
            Self::Automation => "AUTOMATION",
            Self::Confused => "CONFUSED",
            Self::CorrosiveWave => "CORROSIVE_WAVE",
            Self::DarkEmbrace => "DARK_EMBRACE",
            Self::FeelNoPain => "FEEL_NO_PAIN",
            Self::Hex => "HEX",
            Self::Iteration => "ITERATION",
            Self::Pagestorm => "PAGESTORM",
            Self::PillarOfCreation => "PILLAR_OF_CREATION",
            Self::Ringing => "RINGING",
            Self::Smokestack => "SMOKESTACK",
            Self::Speedster => "SPEEDSTER",
            Self::SwordSage => "SWORD_SAGE",
            Self::Tangled => "TANGLED",
            Self::TrashToTreasure => "TRASH_TO_TREASURE",
            Self::BlockNextTurn => "BLOCK_NEXT_TURN",
            Self::SelfFormingClay => "SELF_FORMING_CLAY",
            Self::ToricToughness => "TORIC_TOUGHNESS",
            Self::Clarity => "CLARITY",
            Self::Corruption => "CORRUPTION",
            Self::Curious => "CURIOUS",
            Self::Demesne => "DEMESNE",
            Self::DrawCardsNextTurn => "DRAW_CARDS_NEXT_TURN",
            Self::Friendship => "FRIENDSHIP",
            Self::MachineLearning => "MACHINE_LEARNING",
            Self::MindRot => "MIND_ROT",
            Self::Orbit => "ORBIT",
            Self::PaleBlueDot => "PALE_BLUE_DOT",
            Self::Pyre => "PYRE",
            Self::ToolsOfTheTrade => "TOOLS_OF_THE_TRADE",
            Self::Tyranny => "TYRANNY",
            Self::WasteAway => "WASTE_AWAY",
            Self::Adaptable => "ADAPTABLE",
            Self::CrabRage => "CRAB_RAGE",
            Self::Dampen => "DAMPEN",
            Self::DieForYou => "DIE_FOR_YOU",
            Self::DoorRevival => "DOOR_REVIVAL",
            Self::Illusion => "ILLUSION",
            Self::Infested => "INFESTED",
            Self::Minion => "MINION",
            Self::PossessSpeed => "POSSESS_SPEED",
            Self::PossessStrength => "POSSESS_STRENGTH",
            Self::Ravenous => "RAVENOUS",
            Self::Reattach => "REATTACH",
            Self::SteamEruption => "STEAM_ERUPTION",
            Self::Stock => "STOCK",
            Self::Surprise => "SURPRISE",
            Self::Accelerant => "ACCELERANT",
            Self::Anticipate => "ANTICIPATE",
            Self::Artifact => "ARTIFACT",
            Self::BackAttackLeft => "BACK_ATTACK_LEFT",
            Self::BackAttackRight => "BACK_ATTACK_RIGHT",
            Self::Barricade => "BARRICADE",
            Self::BeaconOfHope => "BEACON_OF_HOPE",
            Self::Buffer => "BUFFER",
            Self::Burrowed => "BURROWED",
            Self::CallOfTheVoid => "CALL_OF_THE_VOID",
            Self::ChildOfTheStars => "CHILD_OF_THE_STARS",
            Self::Coordinate => "COORDINATE",
            Self::CreativeAi => "CREATIVE_AI",
            Self::Cruelty => "CRUELTY",
            Self::CrushUnder => "CRUSH_UNDER",
            Self::DarkShackles => "DARK_SHACKLES",
            Self::Disintegration => "DISINTEGRATION",
            Self::Doom => "DOOM",
            Self::DyingStar => "DYING_STAR",
            Self::EnergyNextTurn => "ENERGY_NEXT_TURN",
            Self::EnfeeblingTouch => "ENFEEBLING_TOUCH",
            Self::Envenom => "ENVENOM",
            Self::FanOfKnives => "FAN_OF_KNIVES",
            Self::FeedingFrenzy => "FEEDING_FRENZY",
            Self::FlexPotion => "FLEX_POTION",
            Self::Focus => "FOCUS",
            Self::FocusedStrike => "FOCUSED_STRIKE",
            Self::ForbiddenGrimoire => "FORBIDDEN_GRIMOIRE",
            Self::ForegoneConclusion => "FOREGONE_CONCLUSION",
            Self::Genesis => "GENESIS",
            Self::Hailstorm => "HAILSTORM",
            Self::HammerTime => "HAMMER_TIME",
            Self::HardToKill => "HARD_TO_KILL",
            Self::Heist => "HEIST",
            Self::HelicalDart => "HELICAL_DART",
            Self::HelloWorld => "HELLO_WORLD",
            Self::Hotfix => "HOTFIX",
            Self::Imbalanced => "IMBALANCED",
            Self::Improvement => "IMPROVEMENT",
            Self::InfiniteBlades => "INFINITE_BLADES",
            Self::Juggernaut => "JUGGERNAUT",
            Self::LightningRod => "LIGHTNING_ROD",
            Self::Mangle => "MANGLE",
            Self::Mayhem => "MAYHEM",
            Self::MonarchsGaze => "MONARCHS_GAZE",
            Self::MonarchsGazeStrengthDown => "MONARCHS_GAZE_STRENGTH_DOWN",
            Self::NecroMastery => "NECRO_MASTERY",
            Self::Nightmare => "NIGHTMARE",
            Self::Nostalgia => "NOSTALGIA",
            Self::Outbreak => "OUTBREAK",
            Self::PaperCuts => "PAPER_CUTS",
            Self::Parry => "PARRY",
            Self::PiercingWail => "PIERCING_WAIL",
            Self::Radiance => "RADIANCE",
            Self::ReaperForm => "REAPER_FORM",
            Self::ReptileTrinket => "REPTILE_TRINKET",
            Self::Royalties => "ROYALTIES",
            Self::SeekingEdge => "SEEKING_EDGE",
            Self::SentryMode => "SENTRY_MODE",
            Self::SetupStrike => "SETUP_STRIKE",
            Self::ShacklingPotion => "SHACKLING_POTION",
            Self::Shroud => "SHROUD",
            Self::SleightOfFlesh => "SLEIGHT_OF_FLESH",
            Self::SpectrumShift => "SPECTRUM_SHIFT",
            Self::SpeedPotion => "SPEED_POTION",
            Self::Spinner => "SPINNER",
            Self::Stampede => "STAMPEDE",
            Self::StarNextTurn => "STAR_NEXT_TURN",
            Self::Stratagem => "STRATAGEM",
            Self::Swipe => "SWIPE",
            Self::Synchronize => "SYNCHRONIZE",
            Self::TheBomb => "THE_BOMB",
            Self::TheHunt => "THE_HUNT",
            Self::Thievery => "THIEVERY",
            Self::Thunder => "THUNDER",
            Self::Vicious => "VICIOUS",
            Self::Electrodynamics => "ELECTRODYNAMICS",
            Self::Metallicize => "METALLICIZE",
            Self::Entangled => "ENTANGLED",
            Self::GrapplePower => "GRAPPLE_POWER",
            Self::ManglePower => "MANGLE_POWER",
            Self::FreePower => "FREE_POWER",
            Self::Generic => "GENERIC",
            Self::NoEnergyGain => "NO_ENERGY_GAIN",
            Self::BorrowedTime => "BORROWED_TIME",
            Self::SicEmPower => "SIC_EM",
            Self::DevourLifePower => "DEVOUR_LIFE",
            Self::Tainted => "TAINTED",
            Self::WitheringPresence => "WITHERING_PRESENCE",
            Self::Ambergris => "AMBERGRIS",
            Self::Cacophony => "CACOPHONY",
            Self::Concoct => "CONCOCT",
            Self::Fade => "FADE",
            Self::Hibernate => "HIBERNATE",
            Self::HyperbeamFocusDown => "HYPERBEAM_FOCUS_DOWN",
            Self::ImitationLearning => "IMITATION_LEARNING",
            Self::OneForAll => "ONE_FOR_ALL",
            Self::Soulbound => "SOULBOUND",
            Self::Underworld => "UNDERWORLD",
            Self::Aim => "AIM",
            Self::AllIsWell => "ALL_IS_WELL",
            Self::Archery => "ARCHERY",
            Self::ChoBarbs => "CHO_BARBS",
            Self::Compost => "COMPOST",
            Self::ConductorsBaton => "CONDUCTORS_BATON",
            Self::CoveringFire => "COVERING_FIRE",
            Self::DislikeYou => "DISLIKE_YOU",
            Self::Ecstasy => "ECSTASY",
            Self::Encore => "ENCORE",
            Self::Ennui => "ENNUI",
            Self::Ensemble => "ENSEMBLE",
            Self::ExplosiveArrows => "EXPLOSIVE_ARROWS",
            Self::FreeCard => "FREE_CARD",
            Self::HiddenRoots => "HIDDEN_ROOTS",
            Self::Honing => "HONING",
            Self::Impaled => "IMPALED",
            Self::KiVerse => "KI_VERSE",
            Self::LostLute => "LOST_LUTE",
            Self::Lullaby => "LULLABY",
            Self::Motif => "MOTIF",
            Self::MythicForm => "MYTHIC_FORM",
            Self::Quietus => "QUIETUS",
            Self::SerratedArrows => "SERRATED_ARROWS",
            Self::ShowOff => "SHOW_OFF",
            Self::Simpleton => "SIMPLETON",
            Self::SongPath => "SONG_PATH",
            Self::Stretch => "STRETCH",
            Self::Tune => "TUNE",
            Self::ViolentChord => "VIOLENT_CHORD",
            Self::Yah => "YAH",
            Self::GodByrd => "GOD_BYRD",
        }
    }

    /// Every variant, in Python source order. Useful for exhaustive
    /// iteration (parity sweeps, property tests).
    #[must_use]
    pub const fn all() -> &'static [PowerKind] {
        &[
            Self::Accuracy,
            Self::Calcify,
            Self::Colossus,
            Self::Conqueror,
            Self::Covered,
            Self::DiamondDiadem,
            Self::DoubleDamage,
            Self::Flanking,
            Self::Flutter,
            Self::Gigantification,
            Self::Guarded,
            Self::Hang,
            Self::Intercept,
            Self::Knockdown,
            Self::Leadership,
            Self::Lethality,
            Self::PhantomBlades,
            Self::Shrink,
            Self::Slow,
            Self::Soar,
            Self::Strength,
            Self::Surrounded,
            Self::Tank,
            Self::Tracking,
            Self::Vigor,
            Self::Vulnerable,
            Self::Weak,
            Self::Dexterity,
            Self::Fasten,
            Self::Frail,
            Self::NoBlock,
            Self::Shadowmeld,
            Self::Unmovable,
            Self::Aggression,
            Self::BiasedCognition,
            Self::Blur,
            Self::Coolant,
            Self::Countdown,
            Self::CrimsonMantle,
            Self::DemonForm,
            Self::Entropy,
            Self::Feral,
            Self::Furnace,
            Self::Loop,
            Self::Neurosurge,
            Self::NoxiousFumes,
            Self::Plating,
            Self::Poison,
            Self::PrepTime,
            Self::Rampart,
            Self::RollingBoulder,
            Self::Sandpit,
            Self::ShadowStep,
            Self::SummonNextTurn,
            Self::WraithForm,
            Self::BattlewornDummyTimeLimit,
            Self::Constrict,
            Self::ConsumingShadow,
            Self::Debilitate,
            Self::Demise,
            Self::EscapeArtist,
            Self::Grapple,
            Self::Hatch,
            Self::HighVoltage,
            Self::Intangible,
            Self::MagicBomb,
            Self::Nemesis,
            Self::NoDraw,
            Self::Rebound,
            Self::Regen,
            Self::RetainHand,
            Self::Ritual,
            Self::SicEm,
            Self::TemporaryDexterity,
            Self::TemporaryFocus,
            Self::TemporaryStrength,
            Self::Territorial,
            Self::WellLaidPlans,
            Self::Afterimage,
            Self::Arsenal,
            Self::BlackHole,
            Self::BladeOfInk,
            Self::Burst,
            Self::Calamity,
            Self::ChainsOfBinding,
            Self::CurlUp,
            Self::DanseMacabre,
            Self::DevourLife,
            Self::Duplication,
            Self::EchoForm,
            Self::Enrage,
            Self::FreeAttack,
            Self::Free,
            Self::FreeSkill,
            Self::Galvanic,
            Self::Gravity,
            Self::Haunt,
            Self::Hellraiser,
            Self::Juggling,
            Self::MasterPlanner,
            Self::Monologue,
            Self::Oblivion,
            Self::OneTwoPunch,
            Self::PainfulStabs,
            Self::Panache,
            Self::Rage,
            Self::Rupture,
            Self::SerpentForm,
            Self::SignalBoost,
            Self::Skittish,
            Self::Sloth,
            Self::Smoggy,
            Self::Sneaky,
            Self::SpiritOfAsh,
            Self::Storm,
            Self::Strangle,
            Self::Subroutine,
            Self::Suck,
            Self::TagTeam,
            Self::Tender,
            Self::TheSealedThrone,
            Self::Veilpiercer,
            Self::VoidForm,
            Self::Asleep,
            Self::FlameBarrier,
            Self::HardenedShell,
            Self::Inferno,
            Self::PersonalHive,
            Self::Plow,
            Self::Reflect,
            Self::Shriek,
            Self::Slippery,
            Self::Slumber,
            Self::TheGambit,
            Self::Thorns,
            Self::VitalSpark,
            Self::Automation,
            Self::Confused,
            Self::CorrosiveWave,
            Self::DarkEmbrace,
            Self::FeelNoPain,
            Self::Hex,
            Self::Iteration,
            Self::Pagestorm,
            Self::PillarOfCreation,
            Self::Ringing,
            Self::Smokestack,
            Self::Speedster,
            Self::SwordSage,
            Self::Tangled,
            Self::TrashToTreasure,
            Self::BlockNextTurn,
            Self::SelfFormingClay,
            Self::ToricToughness,
            Self::Clarity,
            Self::Corruption,
            Self::Curious,
            Self::Demesne,
            Self::DrawCardsNextTurn,
            Self::Friendship,
            Self::MachineLearning,
            Self::MindRot,
            Self::Orbit,
            Self::PaleBlueDot,
            Self::Pyre,
            Self::ToolsOfTheTrade,
            Self::Tyranny,
            Self::WasteAway,
            Self::Adaptable,
            Self::CrabRage,
            Self::Dampen,
            Self::DieForYou,
            Self::DoorRevival,
            Self::Illusion,
            Self::Infested,
            Self::Minion,
            Self::PossessSpeed,
            Self::PossessStrength,
            Self::Ravenous,
            Self::Reattach,
            Self::SteamEruption,
            Self::Stock,
            Self::Surprise,
            Self::Accelerant,
            Self::Anticipate,
            Self::Artifact,
            Self::BackAttackLeft,
            Self::BackAttackRight,
            Self::Barricade,
            Self::BeaconOfHope,
            Self::Buffer,
            Self::Burrowed,
            Self::CallOfTheVoid,
            Self::ChildOfTheStars,
            Self::Coordinate,
            Self::CreativeAi,
            Self::Cruelty,
            Self::CrushUnder,
            Self::DarkShackles,
            Self::Disintegration,
            Self::Doom,
            Self::DyingStar,
            Self::EnergyNextTurn,
            Self::EnfeeblingTouch,
            Self::Envenom,
            Self::FanOfKnives,
            Self::FeedingFrenzy,
            Self::FlexPotion,
            Self::Focus,
            Self::FocusedStrike,
            Self::ForbiddenGrimoire,
            Self::ForegoneConclusion,
            Self::Genesis,
            Self::Hailstorm,
            Self::HammerTime,
            Self::HardToKill,
            Self::Heist,
            Self::HelicalDart,
            Self::HelloWorld,
            Self::Hotfix,
            Self::Imbalanced,
            Self::Improvement,
            Self::InfiniteBlades,
            Self::Juggernaut,
            Self::LightningRod,
            Self::Mangle,
            Self::Mayhem,
            Self::MonarchsGaze,
            Self::MonarchsGazeStrengthDown,
            Self::NecroMastery,
            Self::Nightmare,
            Self::Nostalgia,
            Self::Outbreak,
            Self::PaperCuts,
            Self::Parry,
            Self::PiercingWail,
            Self::Radiance,
            Self::ReaperForm,
            Self::ReptileTrinket,
            Self::Royalties,
            Self::SeekingEdge,
            Self::SentryMode,
            Self::SetupStrike,
            Self::ShacklingPotion,
            Self::Shroud,
            Self::SleightOfFlesh,
            Self::SpectrumShift,
            Self::SpeedPotion,
            Self::Spinner,
            Self::Stampede,
            Self::StarNextTurn,
            Self::Stratagem,
            Self::Swipe,
            Self::Synchronize,
            Self::TheBomb,
            Self::TheHunt,
            Self::Thievery,
            Self::Thunder,
            Self::Vicious,
            Self::Electrodynamics,
            Self::Metallicize,
            Self::Entangled,
            Self::GrapplePower,
            Self::ManglePower,
            Self::FreePower,
            Self::Generic,
            Self::NoEnergyGain,
            Self::BorrowedTime,
            Self::SicEmPower,
            Self::DevourLifePower,
            Self::Tainted,
            Self::WitheringPresence,
        ]
    }

    /// Powers that exist only in game build v0.111.0 (GH #794).
    pub const V0111_ONLY: [PowerKind; 10] = [
        Self::Ambergris,
        Self::Cacophony,
        Self::Concoct,
        Self::Fade,
        Self::Hibernate,
        Self::HyperbeamFocusDown,
        Self::ImitationLearning,
        Self::OneForAll,
        Self::Soulbound,
        Self::Underworld,
    ];

    /// Powers that exist only from game build v0.113.0 on: The Fool's
    /// (`Models.Powers`, alphabetical like the decomp directory).
    pub const V0113_ONLY: [PowerKind; 32] = [
        Self::Aim,
        Self::AllIsWell,
        Self::Archery,
        Self::ChoBarbs,
        Self::Compost,
        Self::ConductorsBaton,
        Self::CoveringFire,
        Self::DislikeYou,
        Self::Ecstasy,
        Self::Encore,
        Self::Ennui,
        Self::Ensemble,
        Self::ExplosiveArrows,
        Self::FreeCard,
        Self::HiddenRoots,
        Self::Honing,
        Self::Impaled,
        Self::KiVerse,
        Self::LostLute,
        Self::Lullaby,
        Self::Motif,
        Self::MythicForm,
        Self::Quietus,
        Self::SerratedArrows,
        Self::ShowOff,
        Self::Simpleton,
        Self::SongPath,
        Self::Stretch,
        Self::Tune,
        Self::ViolentChord,
        Self::Yah,
        Self::GodByrd,
    ];

    /// Powers that v0.111.0 deleted (GH #794): `DiamondDiademPower` went with
    /// the DiamondDiadem relic rework and `OutbreakPower` with the Outbreak
    /// card rework (decomp `v0.107.1/.../Powers/DiamondDiademPower.cs`,
    /// `OutbreakPower.cs`; neither file exists in v0.111.0).
    pub const REMOVED_IN_V0111: [PowerKind; 2] = [Self::DiamondDiadem, Self::Outbreak];

    /// The build a power first exists in.
    #[must_use]
    pub const fn introduced_in(self) -> crate::GameVersion {
        match self {
            Self::Ambergris
            | Self::Cacophony
            | Self::Concoct
            | Self::Fade
            | Self::Hibernate
            | Self::HyperbeamFocusDown
            | Self::ImitationLearning
            | Self::OneForAll
            | Self::Soulbound
            | Self::Underworld => crate::GameVersion::V0_111_0,
            Self::Aim
            | Self::AllIsWell
            | Self::Archery
            | Self::ChoBarbs
            | Self::Compost
            | Self::ConductorsBaton
            | Self::CoveringFire
            | Self::DislikeYou
            | Self::Ecstasy
            | Self::Encore
            | Self::Ennui
            | Self::Ensemble
            | Self::ExplosiveArrows
            | Self::FreeCard
            | Self::HiddenRoots
            | Self::Honing
            | Self::Impaled
            | Self::KiVerse
            | Self::LostLute
            | Self::Lullaby
            | Self::Motif
            | Self::MythicForm
            | Self::Quietus
            | Self::SerratedArrows
            | Self::ShowOff
            | Self::Simpleton
            | Self::SongPath
            | Self::Stretch
            | Self::Tune
            | Self::ViolentChord
            | Self::Yah
            | Self::GodByrd => crate::GameVersion::V0_113_0,
            _ => crate::GameVersion::V0_107_1,
        }
    }

    /// Whether the power exists in game build `version`: introduced at or
    /// before it, and not one of [`Self::REMOVED_IN_V0111`] from v0.111.0 on.
    #[must_use]
    pub const fn exists_in(self, version: crate::GameVersion) -> bool {
        let introduced = self.introduced_in() as u8 <= version as u8;
        let removed = matches!(self, Self::DiamondDiadem | Self::Outbreak)
            && version as u8 >= crate::GameVersion::V0_111_0 as u8;
        introduced && !removed
    }

    /// Every power of game build `version`: [`Self::all`] (the v0.107.1
    /// universe, unchanged) for v0.107.1; from v0.111.0 on, minus
    /// [`Self::REMOVED_IN_V0111`] plus [`Self::V0111_ONLY`].
    #[must_use]
    pub fn all_for(version: crate::GameVersion) -> Vec<PowerKind> {
        let mut v: Vec<PowerKind> = Self::all()
            .iter()
            .copied()
            .filter(|k| k.exists_in(version))
            .collect();
        if version >= crate::GameVersion::V0_111_0 {
            v.extend_from_slice(&Self::V0111_ONLY);
        }
        if version >= crate::GameVersion::V0_113_0 {
            v.extend_from_slice(&Self::V0113_ONLY);
        }
        v
    }
}
