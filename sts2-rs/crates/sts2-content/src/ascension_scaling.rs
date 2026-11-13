//! Ascension attack-damage scaling — data table + post-build patch.
//!
//! C# scales individual monster attack damages at `AscensionLevel.DeadlyEnemies`
//! (>= ascension 9) via `AscensionHelper.GetValueIfAscension(DeadlyEnemies,
//! deadly, base)` — and a handful of move values at `ToughEnemies` (>= A8),
//! kept in the `ASCENSION_TOUGH_*` tables (GH #858). Rather than thread `ascension` into all ~123 monster AI
//! factories, we keep the factories ascension-blind (they encode the A0/base
//! damage) and patch the built `MonsterAi` in [`crate::monster_dispatch::
//! build_ai_with_ascension`]: at ascension >= 9 each scaled move's attack
//! intent damage is bumped to its DeadlyEnemies value. Ascension is fixed for
//! the whole combat, so patching once at spawn is equivalent to C# resolving
//! the `GetValueIfAscension` property on every telegraph.
//!
//! The table is keyed by `(MonsterKind, MoveState id)` and carries BOTH the
//! non-ascension `base` and the `deadly` value. `base` is redundant for the
//! patch itself but lets [`tests::ascension_attack_table_matches_real_moves`]
//! assert that every row resolves to a real move whose A0 damage equals
//! `base` — a compile-and-test guard that the table stays in sync with the
//! factories and that each row was transcribed from the C# correctly.
//!
//! Values transcribed from `../sts2-wiki/decompiled/v0.103.2/` per-monster
//! `<Class>.cs` `GetValueIfAscension(DeadlyEnemies, …)` attack properties.

use sts2_core::monster_ai::{MonsterAi, MonsterState};
use sts2_core::monsters::{Intent, MonsterKind};
use sts2_core::powers::PowerKind;
use sts2_core::AscensionLevel;

/// `(MonsterKind, MoveState id, base damage, DeadlyEnemies damage)`.
/// Only attacks whose damage actually scales (deadly != base) are listed.
#[rustfmt::skip]
pub(crate) const ASCENSION_ATTACK_SCALING: &[(MonsterKind, &str, i32, i32)] = &[
    // Aeonglass (v0.107.1): EbbDamage GetValueIfAscension(DeadlyEnemies, 32, 26)
    // ⇒ A0 26 / A9 32; EyeLasersDamage (..12, 11) ⇒ A0 11 / A9 12.
    (MonsterKind::Aeonglass, "EBB_MOVE", 26, 32),
    (MonsterKind::Aeonglass, "EYE_LASERS_MOVE", 11, 12),
    // `…wb8lk.11`: v0.107.1 KillshotDamage GetValueIfAscension(DeadlyEnemies,
    // 11, 10) ⇒ A0 10 / A9 11 (was A0 11 / A9 12).
    (MonsterKind::AssassinRubyRaider, "KILLSHOT", 10, 11),
    (MonsterKind::AxeRubyRaider, "BIG_SWING", 12, 13),
    (MonsterKind::AxeRubyRaider, "SWING_1", 5, 6),
    (MonsterKind::AxeRubyRaider, "SWING_2", 5, 6),
    // `…wb8lk.11`: v0.107.1 Axebot rework. HammerUppercutDamage
    // GetValueIfAscension(DeadlyEnemies, 14, 12) ⇒ A0 12 / A9 14;
    // OneTwoDamage (..10, 9) ⇒ A0 9 / A9 10.
    (MonsterKind::Axebot, "HAMMER_UPPERCUT", 12, 14),
    (MonsterKind::Axebot, "ONE_TWO", 9, 10),
    (MonsterKind::BowlbugEgg, "BITE", 7, 8),
    (MonsterKind::BowlbugRock, "HEADBUTT", 15, 16),
    (MonsterKind::BowlbugSilk, "THRASH", 4, 5),
    (MonsterKind::BruteRubyRaider, "BEAT", 7, 8),
    (MonsterKind::BygoneEffigy, "SLASHES", 13, 15),
    (MonsterKind::Byrdonis, "PECK", 3, 4),
    (MonsterKind::Byrdonis, "SWOOP", 17, 19),
    (MonsterKind::CalcifiedCultist, "DARK_STRIKE", 9, 11),
    (MonsterKind::CeremonialBeast, "CRUSH", 17, 19),
    (MonsterKind::CeremonialBeast, "PLOW", 18, 20),
    (MonsterKind::CeremonialBeast, "STOMP", 15, 17),
    (MonsterKind::Chomper, "CLAMP", 8, 9),
    (MonsterKind::CorpseSlug, "GLOMP", 8, 9),
    (MonsterKind::CrossbowRubyRaider, "FIRE", 14, 16),
    (MonsterKind::Crusher, "BUG_STING", 6, 7),
    (MonsterKind::Crusher, "GUARDED_STRIKE", 12, 14),
    (MonsterKind::Crusher, "THRASH", 12, 14),
    (MonsterKind::CubexConstruct, "EXPEL_BLAST", 5, 6),
    (MonsterKind::CubexConstruct, "REPEATER", 7, 8),
    (MonsterKind::CubexConstruct, "REPEATER_2", 7, 8),
    (MonsterKind::DampCultist, "DARK_STRIKE", 1, 3),
    (MonsterKind::DecimillipedeSegment, "BULK", 6, 7),
    (MonsterKind::DecimillipedeSegment, "CONSTRICT", 8, 9),
    (MonsterKind::DecimillipedeSegment, "WRITHE", 5, 6),
    (MonsterKind::DecimillipedeSegmentBack, "BULK", 6, 7),
    (MonsterKind::DecimillipedeSegmentBack, "CONSTRICT", 8, 9),
    (MonsterKind::DecimillipedeSegmentBack, "WRITHE", 5, 6),
    (MonsterKind::DecimillipedeSegmentFront, "BULK", 6, 7),
    (MonsterKind::DecimillipedeSegmentFront, "CONSTRICT", 8, 9),
    (MonsterKind::DecimillipedeSegmentFront, "WRITHE", 5, 6),
    (MonsterKind::DecimillipedeSegmentMiddle, "BULK", 6, 7),
    (MonsterKind::DecimillipedeSegmentMiddle, "CONSTRICT", 8, 9),
    (MonsterKind::DecimillipedeSegmentMiddle, "WRITHE", 5, 6),
    (MonsterKind::DenseVegetationWriggler, "NASTY_BITE", 6, 7),
    (MonsterKind::DevotedSculptor, "SAVAGE", 12, 15),
    (MonsterKind::Entomancer, "SPEAR", 18, 20),
    (MonsterKind::Exoskeleton, "MANDIBLE", 8, 9),
    (MonsterKind::Fabricator, "DISINTEGRATE", 11, 13),
    (MonsterKind::Fabricator, "FABRICATING_STRIKE", 18, 21),
    (MonsterKind::FakeMerchantMonster, "SWIPE", 13, 15),
    (MonsterKind::FakeMerchantMonster, "THROW_RELIC", 9, 10),
    (MonsterKind::FlailKnight, "FLAIL", 9, 10),
    (MonsterKind::FlailKnight, "RAM", 15, 17),
    (MonsterKind::Flyconid, "FRAIL_SPORES", 8, 9),
    (MonsterKind::Flyconid, "SMASH", 11, 12),
    (MonsterKind::Fogmog, "HEADBUTT", 14, 16),
    (MonsterKind::Fogmog, "SWIPE", 8, 9),
    (MonsterKind::Fogmog, "SWIPE_RANDOM", 8, 9),
    (MonsterKind::FossilStalker, "LASH", 3, 4),
    (MonsterKind::FossilStalker, "LATCH", 12, 14),
    (MonsterKind::FossilStalker, "TACKLE", 9, 11),
    (MonsterKind::FrogKnight, "BEETLE_CHARGE", 35, 40),
    (MonsterKind::FrogKnight, "STRIKE_DOWN_EVIL", 21, 23),
    (MonsterKind::FrogKnight, "TONGUE_LASH", 13, 14),
    (MonsterKind::FuzzyWurmCrawler, "ACID_GOOP", 4, 6),
    (MonsterKind::FuzzyWurmCrawler, "FIRST_ACID_GOOP", 4, 6),
    (MonsterKind::GasBomb, "EXPLODE", 8, 9),
    (MonsterKind::GlobeHead, "GALVANIC_BURST", 16, 17),
    (MonsterKind::GlobeHead, "SHOCKING_SLAP", 13, 14),
    (MonsterKind::GlobeHead, "THUNDER_STRIKE", 6, 7),
    // GremlinMerc GIMME / DOUBLE_SMASH / HEHE are ToughEnemies (A8) in C#,
    // so they live in ASCENSION_TOUGH_ATTACK_SCALING (GH #858).
    // `…wb8lk.11`: v0.107.1 HauntedShip attacks scale at A9. SwipeDamage
    // GetValueIfAscension(DeadlyEnemies, 14, 13) ⇒ A0 13 / A9 14; StompDamage
    // (..5, 4) ⇒ A0 4 / A9 5 (per hit, × StompRepeat 3). Decomp
    // `v0.107.1/…/Monsters/HauntedShip.cs:31,33`.
    (MonsterKind::HauntedShip, "SWIPE_MOVE", 13, 14),
    (MonsterKind::HauntedShip, "STOMP_MOVE", 4, 5),
    (MonsterKind::HunterKiller, "BITE_MOVE", 17, 19),
    (MonsterKind::HunterKiller, "PUNCTURE_MOVE", 7, 8),
    // `…wb8lk.11`: v0.107.1 InfestedPrism stat cuts. JabDamage
    // GetValueIfAscension(DeadlyEnemies, 17, 15) ⇒ A0 15 / A9 17;
    // RadiateDamage (..13, 11) ⇒ A0 11 / A9 13; WhirlwindDamage (..6, 5)
    // ⇒ A0 5 / A9 6; PulsateDamage (..10, 8) ⇒ A0 8 / A9 10 (PULSATE
    // gained an attack in v0.107.1, replacing the pure Strength buff).
    (MonsterKind::InfestedPrism, "JAB", 15, 17),
    (MonsterKind::InfestedPrism, "PULSATE", 8, 10),
    (MonsterKind::InfestedPrism, "RADIATE", 11, 13),
    (MonsterKind::InfestedPrism, "WHIRLWIND", 5, 6),
    (MonsterKind::Inklet, "JAB", 3, 4),
    (MonsterKind::Inklet, "PIERCING_GAZE", 10, 11),
    (MonsterKind::Inklet, "WHIRLWIND", 2, 3),
    (MonsterKind::KinPriest, "ORB_OF_FRAILTY_MOVE", 8, 9),
    (MonsterKind::KinPriest, "ORB_OF_WEAKNESS_MOVE", 8, 9),
    (MonsterKind::KnowledgeDemon, "KNOWLEDGE_OVERWHELMING", 8, 9),
    (MonsterKind::KnowledgeDemon, "PONDER", 11, 13),
    (MonsterKind::KnowledgeDemon, "SLAP", 17, 18),
    (MonsterKind::LagavulinMatriarch, "DISEMBOWEL", 9, 10),
    (MonsterKind::LagavulinMatriarch, "SLASH", 19, 21),
    (MonsterKind::LagavulinMatriarch, "SLASH2", 12, 14),
    (MonsterKind::LeafSlimeM, "CLUMP_SHOT", 8, 9),
    (MonsterKind::LeafSlimeS, "BUTT", 3, 4),
    (MonsterKind::LivingFog, "ADVANCED_GAS", 8, 9),
    (MonsterKind::LivingFog, "BLOAT", 5, 6),
    (MonsterKind::LivingFog, "SUPER_GAS_BLAST", 8, 9),
    (MonsterKind::LivingShield, "SMASH", 16, 18),
    (MonsterKind::LouseProgenitor, "POUNCE_MOVE", 14, 16),
    (MonsterKind::LouseProgenitor, "WEB_CANNON_MOVE", 9, 10),
    (MonsterKind::MagiKnight, "BOMB", 35, 40),
    (MonsterKind::MagiKnight, "FIRST_POWER_SHIELD", 6, 7),
    (MonsterKind::MagiKnight, "RAM", 10, 11),
    (MonsterKind::Mawler, "CLAW", 4, 5),
    (MonsterKind::Mawler, "RIP_AND_TEAR", 14, 16),
    (MonsterKind::MechaKnight, "CHARGE", 25, 30),
    (MonsterKind::MechaKnight, "HEAVY_CLEAVE", 35, 40),
    (MonsterKind::MysteriousKnight, "FLAIL", 9, 10),
    (MonsterKind::MysteriousKnight, "RAM", 15, 17),
    (MonsterKind::Myte, "BITE_MOVE", 13, 15),
    (MonsterKind::Myte, "SUCK_MOVE", 4, 6),
    (MonsterKind::Nibbit, "BUTT", 12, 13),
    (MonsterKind::Nibbit, "SLICE", 6, 7),
    (MonsterKind::Ovicopter, "SMASH_MOVE", 16, 17),
    (MonsterKind::Ovicopter, "TENDERIZER_MOVE", 7, 8),
    (MonsterKind::OwlMagistrate, "MAGISTRATE_SCRUTINY", 16, 17),
    (MonsterKind::OwlMagistrate, "VERDICT", 33, 36),
    (MonsterKind::Parafright, "SLAM", 16, 17),
    (MonsterKind::PhrogParasite, "LASH", 4, 5),
    (MonsterKind::PunchConstruct, "FAST_PUNCH", 5, 6),
    (MonsterKind::PunchConstruct, "STRONG_PUNCH", 14, 16),
    (MonsterKind::Queen, "EXECUTION_MOVE", 15, 18),
    (MonsterKind::Queen, "OFF_WITH_YOUR_HEAD_MOVE", 3, 4),
    (MonsterKind::Rocket, "LASER_MOVE", 31, 35),
    (MonsterKind::Rocket, "PRECISION_BEAM_MOVE", 18, 20),
    (MonsterKind::Rocket, "TARGETING_RETICLE_MOVE", 3, 4),
    (MonsterKind::ScrollOfBiting, "CHEW", 5, 6),
    (MonsterKind::ScrollOfBiting, "CHOMP", 14, 16),
    (MonsterKind::Seapunk, "SEA_KICK", 11, 13),
    (MonsterKind::SewerClam, "JET", 10, 11),
    (MonsterKind::ShrinkerBeetle, "CHOMP", 7, 8),
    (MonsterKind::ShrinkerBeetle, "STOMP", 13, 14),
    // `…wb8lk.11`: v0.107.1 SkulkingColony removed SMASH_MOVE and added a
    // second Zoom (ZOOM_MOVE_2, same ZoomDamage). InertiaDamage/Zoom/
    // PiercingStabs values are unchanged from v0.103.2.
    (MonsterKind::SkulkingColony, "INERTIA_MOVE", 9, 11),
    (MonsterKind::SkulkingColony, "PIERCING_STABS_MOVE", 7, 8),
    (MonsterKind::SkulkingColony, "ZOOM_MOVE", 14, 16),
    (MonsterKind::SkulkingColony, "ZOOM_MOVE_2", 14, 16),
    (MonsterKind::SlimedBerserker, "FURIOUS_PUMMELING_MOVE", 4, 5),
    (MonsterKind::SlimedBerserker, "SMOTHER_MOVE", 30, 33),
    (MonsterKind::SlitheringStrangler, "LASH", 12, 13),
    (MonsterKind::SlitheringStrangler, "TWACK", 7, 8),
    (MonsterKind::SludgeSpinner, "OIL_SPRAY", 8, 9),
    (MonsterKind::SludgeSpinner, "RAGE", 6, 7),
    (MonsterKind::SludgeSpinner, "SLAM", 11, 12),
    (MonsterKind::SlumberingBeetle, "ROLL_OUT_MOVE", 16, 18),
    // `wb-010871941d034711` sweep: SneakyGremlin was absent from this table.
    // C# `SneakyGremlin.cs:25` `TackleDamage =>
    // GetValueIfAscension(DeadlyEnemies, 10, 9)`.
    (MonsterKind::SneakyGremlin, "TACKLE", 9, 10),
    (MonsterKind::SnappingJaxfruit, "ENERGY_ORB_MOVE", 3, 4),
    (MonsterKind::SoulFysh, "DE_GAS", 16, 17),
    (MonsterKind::SoulFysh, "GAZE", 7, 8),
    // `…wb8lk.11`: v0.107.1 ScreamDamage GetValueIfAscension(DeadlyEnemies,
    // 15, 13) ⇒ A0 13 / A9 15 (was A0 11 / A9 12).
    (MonsterKind::SoulFysh, "SCREAM", 13, 15),
    (MonsterKind::SoulNexus, "DRAIN_LIFE_MOVE", 18, 19),
    (MonsterKind::SoulNexus, "MAELSTROM_MOVE", 6, 7),
    (MonsterKind::SoulNexus, "SOUL_BURN_MOVE", 29, 31),
    (MonsterKind::SpectralKnight, "SOUL_FLAME", 3, 4),
    (MonsterKind::SpectralKnight, "SOUL_SLASH", 15, 17),
    (MonsterKind::SpinyToad, "EXPLOSION", 23, 25),
    (MonsterKind::SpinyToad, "LASH", 17, 19),
    (MonsterKind::Stabbot, "STAB", 11, 12),
    (MonsterKind::TerrorEel, "CRASH_MOVE", 16, 18),
    (MonsterKind::TerrorEel, "ThrashMove", 3, 4),
    (MonsterKind::TestSubject, "BITE", 20, 22),
    (MonsterKind::TestSubject, "SKULL_BASH", 14, 16),
    // `wb-010871941d034711` sweep: the two phase-3 MultiAttacks were missing.
    // C# `TestSubject.cs:94,98`: MultiClawDamage / Phase3LacerateDamage =>
    // GetValueIfAscension(DeadlyEnemies, 11, 10). MULTI_CLAW's hit COUNT
    // grows via `ExtraMultiClawCount`, not ascension, so only damage scales.
    (MonsterKind::TestSubject, "MULTI_CLAW", 10, 11),
    (MonsterKind::TestSubject, "PHASE3_LACERATE", 10, 11),
    (MonsterKind::TheForgotten, "DREAD", 13, 15),
    (MonsterKind::TheInsatiable, "LUNGING_BITE", 28, 31),
    // `wb-010871941d034711` sweep: both THRASH move states share C#
    // `TheInsatiable.cs:62` `ThrashDamage =>
    // GetValueIfAscension(DeadlyEnemies, 9, 8)` (2 hits each) and were
    // missing while LUNGING_BITE was present.
    (MonsterKind::TheInsatiable, "THRASH_1", 8, 9),
    (MonsterKind::TheInsatiable, "THRASH_2", 8, 9),
    (MonsterKind::TheLost, "EYE_LASERS", 4, 5),
    (MonsterKind::TheObscura, "HARDENING_STRIKE_MOVE", 6, 7),
    (MonsterKind::TheObscura, "PIERCING_GAZE_MOVE", 10, 11),
    (MonsterKind::ThievingHopper, "HAT_TRICK_MOVE", 21, 23),
    (MonsterKind::ThievingHopper, "NAB_MOVE", 14, 16),
    (MonsterKind::ThievingHopper, "THIEVERY_MOVE", 17, 19),
    (MonsterKind::TorchHeadAmalgam, "TACKLE_1", 18, 19),
    (MonsterKind::TorchHeadAmalgam, "TACKLE_2", 18, 19),
    (MonsterKind::TorchHeadAmalgam, "TACKLE_3", 14, 15),
    (MonsterKind::TorchHeadAmalgam, "TACKLE_4", 14, 15),
    // `wb-010871941d034711` (seed DZV0PD70HW, IC A9, Underdocks): Toadpole
    // had NO rows here, so A9 telegraphed/dealt the A0 values. C#
    // `Toadpole.cs:49,53`: SpikeSpitDamage =>
    // GetValueIfAscension(DeadlyEnemies, 4, 3) (x3 hits), WhirlDamage =>
    // (.., 8, 7).
    (MonsterKind::Toadpole, "SPIKE_SPIT_MOVE", 3, 4),
    (MonsterKind::Toadpole, "WHIRL_MOVE", 7, 8),
    (MonsterKind::ToughEgg, "NIBBLE_MOVE", 4, 5),
    // NOTE: TrackerRubyRaider HOUNDS scales its MultiAttack *hits* (8→9 at
    // DeadlyEnemies), not per-hit damage (C# `TrackerRubyRaider.cs:23`
    // HoundsDamage is 1 at both levels), so its row lives in
    // ASCENSION_HITS_SCALING below, not here.
    (MonsterKind::Tunneler, "BELOW_MOVE_1", 23, 26),
    (MonsterKind::Tunneler, "BITE_MOVE", 13, 15),
    (MonsterKind::TurretOperator, "UNLOAD_MOVE_1", 3, 4),
    (MonsterKind::TurretOperator, "UNLOAD_MOVE_2", 3, 4),
    (MonsterKind::TwigSlimeM, "CLUMP_SHOT", 11, 12),
    (MonsterKind::TwigSlimeS, "BUTT", 4, 5),
    (MonsterKind::TwoTailedRat, "DISEASE_BITE", 6, 7),
    (MonsterKind::TwoTailedRat, "SCRATCH", 8, 9),
    // `…wb8lk.11`: v0.107.1 DismemberDamage 27→26 at A0 (decomp
    // `Vantom.cs:72 GetValueIfAscension(DeadlyEnemies, 30, 26)`).
    (MonsterKind::Vantom, "DISMEMBER", 26, 30),
    (MonsterKind::Vantom, "INKY_LANCE", 6, 7),
    (MonsterKind::Vantom, "INK_BLOT", 7, 8),
    (MonsterKind::VineShambler, "CHOMP", 16, 18),
    (MonsterKind::VineShambler, "GRASPING_VINES", 8, 9),
    (MonsterKind::VineShambler, "SWIPE", 6, 7),
    (MonsterKind::WaterfallGiant, "PRESSURE_GUN", 20, 23),
    (MonsterKind::WaterfallGiant, "PRESSURE_UP", 13, 14),
    (MonsterKind::WaterfallGiant, "RAM", 10, 11),
    (MonsterKind::WaterfallGiant, "STOMP", 15, 16),
    (MonsterKind::Wriggler, "NASTY_BITE", 6, 7),
    (MonsterKind::Zapbot, "ZAP", 14, 15),
];

/// Attack damages C# scales at `AscensionLevel.ToughEnemies` (A8), not
/// `DeadlyEnemies` — the A8 sibling of [`ASCENSION_ATTACK_SCALING`], same
/// `(kind, move_id, base, tough)` shape. Identical in v0.107.1 and v0.111.0.
#[rustfmt::skip]
pub(crate) const ASCENSION_TOUGH_ATTACK_SCALING: &[(MonsterKind, &str, i32, i32)] = &[
    // C# `GremlinMerc.cs:36,40,44`: `GimmeDamage => (ToughEnemies, 8, 7)`,
    // `DoubleSmashDamage => (.., 7, 6)`, `HeheDamage => (.., 9, 8)`. These sat
    // in the A9 table, so A8 was one point soft on all three (GH #858).
    (MonsterKind::GremlinMerc, "DOUBLE_SMASH", 6, 7),
    (MonsterKind::GremlinMerc, "GIMME", 7, 8),
    (MonsterKind::GremlinMerc, "HEHE", 8, 9),
];

/// Move BLOCKS C# scales at `AscensionLevel.ToughEnemies` (A8) — the A8
/// sibling of [`ASCENSION_BLOCK_SCALING`], and used the same way: the patch
/// rewrites the move's `Intent::Defend` (a primary Defend is what
/// `perform_intent` grants), and the effect_fns that grant a SECONDARY
/// Defend read the same row through [`tough_block`]. Identical in v0.107.1
/// and v0.111.0. `(kind, move_id, base_block, tough_block)`. GH #858.
#[rustfmt::skip]
pub(crate) const ASCENSION_TOUGH_BLOCK_SCALING: &[(MonsterKind, &str, i32, i32)] = &[
    // C# `InfestedPrism.cs:44,113` `PulsateBlock => (ToughEnemies, 22, 20)`;
    // secondary, granted by `infested_prism_pulsate_effect`.
    (MonsterKind::InfestedPrism, "PULSATE", 20, 22),
    // C# `LagavulinMatriarch.cs:57,219` `Slash2Block => (.., 14, 12)`;
    // secondary, granted by `lagavulin_matriarch_slash2_effect`.
    (MonsterKind::LagavulinMatriarch, "SLASH2", 12, 14),
    // C# `LouseProgenitor.cs:65,106` `CurlBlock => (.., 18, 14)` — a PRIMARY
    // Defend. The same property is the spawn CurlUp amount
    // (`MonsterKind::on_spawn_self_buff_tough`).
    (MonsterKind::LouseProgenitor, "CURL_AND_GROW_MOVE", 14, 18),
    // C# `MagiKnight.cs:49,104,111` `PowerShieldBlock => (.., 9, 5)`, read by
    // POWER_SHIELD (secondary, `magi_knight_power_shield_effect`) and PREP
    // (primary).
    (MonsterKind::MagiKnight, "FIRST_POWER_SHIELD", 5, 9),
    (MonsterKind::MagiKnight, "PREP", 5, 9),
    // C# `Nibbit.cs:32,108` `SliceBlock => (.., 6, 5)`; secondary (head is
    // the SingleAttack), granted by `nibbit_slice_effect`.
    (MonsterKind::Nibbit, "SLICE", 5, 6),
    // C# `Seapunk.cs:38,86` `BubbleBlock => (.., 8, 7)`; secondary (the head
    // intent is the Buff), granted by `seapunk_bubble_burp_effect`. The grant
    // already read the A8 pair inline but the Defend telegraph kept 7 (GH #893).
    (MonsterKind::Seapunk, "BUBBLE_BURP", 7, 8),
    // C# `Tunneler.cs:52,100` `BlockGain => (.., 37, 32)`; secondary,
    // granted by `tunneler_burrow_effect`.
    (MonsterKind::Tunneler, "BURROW_MOVE", 32, 37),
];

/// The `ascension`-resolved block for `(kind, move_id)` from
/// [`ASCENSION_TOUGH_BLOCK_SCALING`] — the value C#'s `…Block` property
/// yields. Returns `base` below `ToughEnemies` and for any move with no row.
/// The ToughEnemies counterpart of [`deadly_block`], for the secondary-defend
/// effect_fns whose grant the intent patch cannot reach.
#[must_use]
pub fn tough_block(ascension: u8, kind: MonsterKind, move_id: &str, base: i32) -> i32 {
    if ascension < AscensionLevel::ToughEnemies as u8 {
        return base;
    }
    ASCENSION_TOUGH_BLOCK_SCALING
        .iter()
        .find(|e| e.0 == kind && e.1 == move_id)
        .map_or(base, |e| e.3)
}

/// Monsters whose `DeadlyEnemies` (A9) scaling bumps a move's BLOCK.
///
/// C# reads a `…Block` property in `CreatureCmd.GainBlock`, and the move's
/// `DefendIntent` telegraphs the same property, so one row drives both: the
/// patch rewrites `Intent::Defend { block }`, and the effect_fns that grant
/// a SECONDARY defend (which `perform_intent` drops) read the same pair via
/// [`deadly_block`] so telegraph and application cannot diverge.
///
/// `(kind, move_id, base_block, deadly_block)`.
pub(crate) const ASCENSION_BLOCK_SCALING: &[(MonsterKind, &str, i32, i32)] = &[
    // C# `AxeRubyRaider.cs:26` `SwingBlock => (.., 6, 5)`; both SWING move
    // states share `SwingMove`.
    (MonsterKind::AxeRubyRaider, "SWING_1", 5, 6),
    (MonsterKind::AxeRubyRaider, "SWING_2", 5, 6),
    // C# `Axebot.cs:36` `BootUpBlock => (.., 15, 10)` — a PRIMARY defend,
    // applied by `perform_intent` straight off the patched intent.
    (MonsterKind::Axebot, "BOOT_UP", 10, 15),
    // C# `BowlbugEgg.cs:25` `ProtectBlock => (.., 8, 7)`.
    (MonsterKind::BowlbugEgg, "BITE", 7, 8),
    // C# `InfestedPrism.cs:48` `RadiateBlock => (.., 13, 11)`.
    (MonsterKind::InfestedPrism, "RADIATE", 11, 13),
    // C# `TheObscura.cs:44` `HardeningStrikeBlock => (.., 7, 6)`.
    (MonsterKind::TheObscura, "HARDENING_STRIKE_MOVE", 6, 7),
];

/// The `ascension`-resolved block for `(kind, move_id)` — the value C#'s
/// `…Block` property yields. Returns `base` below `DeadlyEnemies` and for
/// any move with no row. Used by the secondary-defend effect_fns, whose
/// grant is invisible to the [`ASCENSION_BLOCK_SCALING`] intent patch.
#[must_use]
pub fn deadly_block(ascension: u8, kind: MonsterKind, move_id: &str, base: i32) -> i32 {
    if ascension < AscensionLevel::DeadlyEnemies as u8 {
        return base;
    }
    ASCENSION_BLOCK_SCALING
        .iter()
        .find(|e| e.0 == kind && e.1 == move_id)
        .map_or(base, |e| e.3)
}

/// Monsters whose `DeadlyEnemies` (A9) scaling bumps a MultiAttack's HIT
/// COUNT rather than its per-hit damage. The intent's `hits` drives BOTH
/// the telegraph and the number of `apply_damage_to_player` calls in
/// `perform_intent`, so patching it scales the dealt damage too — the
/// hits-side analogue of [`ASCENSION_ATTACK_SCALING`].
///
/// `(kind, move_id, base_hits, deadly_hits)`.
const ASCENSION_HITS_SCALING: &[(MonsterKind, &str, u8, u8)] = &[
    // C# `Exoskeleton.cs:36`: `SkitterRepeats =>
    // GetValueIfAscension(DeadlyEnemies, 4, 3)` — consumed by both the
    // telegraph (`:55` `MultiAttackIntent(SkitterDamage, SkitterRepeats)`)
    // and the effect (`:79` `.WithHitCount(SkitterRepeats)`). Per-hit
    // `SkitterDamage` (`:34`) is a flat 1 with no ascension pair, so the
    // damage table correctly has no SKITTER row; pre-fix the A9 SKITTER
    // dealt 1×3 instead of 1×4 (25% low) and telegraphed the same.
    // C# `Entomancer.cs:29` `BeesRepeat =>
    // GetValueIfAscension(DeadlyEnemies, 8, 7)`, read by both the telegraph
    // (`:45` `MultiAttackIntent(BeesDamage, BeesRepeat)`) and the effect
    // (`:73` `.WithHitCount(BeesRepeat)`). Per-hit `BeesDamage` is a flat 3.
    (MonsterKind::Entomancer, "BEES", 7, 8),
    (MonsterKind::Exoskeleton, "SKITTER", 3, 4),
    // C# `TrackerRubyRaider.cs:23,25`: `HoundsDamage =>
    // GetValueIfAscension(DeadlyEnemies, 1, 1)` (flat — no damage row) and
    // `HoundsRepeat => (.., 9, 8)`, consumed by both the telegraph (`:33`
    // `MultiAttackIntent(HoundsDamage, HoundsRepeat)`) and the effect
    // (`:49` `.WithHitCount(HoundsRepeat)`).
    (MonsterKind::TrackerRubyRaider, "HOUNDS", 8, 9),
];

/// One move's granted-power amount: `(kind, move_id, power, base, scaled)`.
pub(crate) type BuffRow = (MonsterKind, &'static str, PowerKind, i32, i32);

/// Power amounts a move grants that C# scales at `DeadlyEnemies` (A9), as
/// `(kind, move_id, power, base, deadly)` — the buff sibling of
/// [`ASCENSION_ATTACK_SCALING`] / [`ASCENSION_BLOCK_SCALING`], used the same
/// way: [`apply_ascension_attack_scaling`] rewrites the move's
/// `Intent::Buff { kind: power, amount }`, and the move's effect_fn reads the
/// same row through [`scaled_buff`], so the telegraphed number is the
/// applied one (GH #893: the effect_fns read their own inline pairs, so the
/// intent kept its A0 number at A9+ while the grant scaled). C#'s
/// `BuffIntent` carries no number; ours reaches the wire
/// (`sts2-wire/src/emit/combat.rs`). Same values in both builds; the
/// v0.111.0-only buff is in [`V0111_BUFF_VALUES`].
#[rustfmt::skip]
pub(crate) const ASCENSION_BUFF_SCALING: &[BuffRow] = &[
    // Aeonglass `IncreasingIntensityBaseStrength => (.., 4, 3)` on the FIRST
    // use (AdditionalStrength == 0); the effect adds the upgrade count after
    // that and re-telegraphs the next use.
    (MonsterKind::Aeonglass, "INCREASING_INTENSITY_MOVE", PowerKind::Strength, 3, 4),
    (MonsterKind::BowlbugNectar, "BUFF", PowerKind::Strength, 15, 16),
    (MonsterKind::CeremonialBeast, "CRUSH", PowerKind::Strength, 3, 4),
    (MonsterKind::CeremonialBeast, "STAMP", PowerKind::Plow, 150, 160),
    (MonsterKind::Crusher, "ADAPT", PowerKind::Strength, 2, 3),
    (MonsterKind::DampCultist, "INCANTATION", PowerKind::Ritual, 5, 6),
    // `…wb8lk.11`: v0.107.1 PULSATE re-applies VitalSpark (the Tainted
    // applier), not Strength. `VitalSparkAmount => (.., 3, 2)`.
    (MonsterKind::InfestedPrism, "PULSATE", PowerKind::VitalSpark, 2, 3),
    (MonsterKind::KinFollower, "POWER_DANCE", PowerKind::Strength, 2, 3),
    (MonsterKind::KinPriest, "RITUAL_MOVE", PowerKind::Strength, 2, 3),
    (MonsterKind::KnowledgeDemon, "PONDER", PowerKind::Strength, 2, 3),
    // NOTE: MagiKnight FIRST_POWER_SHIELD's PowerShieldBlock (5→9) is a
    // ToughEnemies BLOCK (ASCENSION_TOUGH_BLOCK_SCALING, GH #858); its
    // attack damage (6→7) is covered by the attack table.
    (MonsterKind::Myte, "SUCK_MOVE", PowerKind::Strength, 2, 3),
    (MonsterKind::Nibbit, "HISS", PowerKind::Strength, 2, 3),
    (MonsterKind::Ovicopter, "NUTRITIONAL_PASTE_MOVE", PowerKind::Strength, 3, 4),
    (MonsterKind::PhantasmalGardener, "ENLARGE", PowerKind::Strength, 2, 3),
    (MonsterKind::Rocket, "CHARGE_UP_MOVE", PowerKind::Strength, 2, 3),
    // C# `Seapunk.cs:40` `BubbleStr => (.., 2, 1)`.
    (MonsterKind::Seapunk, "BUBBLE_BURP", PowerKind::Strength, 1, 2),
    // v0.107.1 `SkulkingColony.cs:49` is (4, 2) — A9 gives 4, not 3.
    (MonsterKind::SkulkingColony, "INERTIA_MOVE", PowerKind::Strength, 2, 4),
    // C# `TestSubject.cs:104` `BurningGrowlStrengthGain => (.., 3, 2)`.
    (MonsterKind::TestSubject, "BURNING_GROWL", PowerKind::Strength, 2, 3),
    (MonsterKind::TheInsatiable, "SALIVATE", PowerKind::Strength, 2, 3),
    (MonsterKind::WaterfallGiant, "PRESSURIZE", PowerKind::SteamEruption, 15, 20),
];

/// v0.111.0 buff amounts that differ from v0.107.1, as the full
/// `(kind, move_id, power, base, deadly)` pair of the 0.111 property. Applied
/// by [`apply_v0111_attack_values`] and [`scaled_buff`] only when the
/// combat's `game_version` is `V0_111_0` or later.
#[rustfmt::skip]
pub(crate) const V0111_BUFF_VALUES: &[BuffRow] = &[
    // v0.111.0: `GrowStrength => GetValueIfAscension(DeadlyEnemies, 7, 5)`;
    // v0.107.1 applied a flat 5 (decomp LouseProgenitor.cs:65,107).
    (MonsterKind::LouseProgenitor, "CURL_AND_GROW_MOVE", PowerKind::Strength, 5, 7),
];

fn buff_row(
    rows: &[BuffRow],
    kind: MonsterKind,
    move_id: &str,
    power: PowerKind,
) -> Option<BuffRow> {
    rows.iter()
        .find(|e| e.0 == kind && e.1 == move_id && e.2 == power)
        .copied()
}

/// The `ascension`-resolved amount of `power` that move `(kind, move_id)`
/// grants on build `version` — what C#'s amount property yields — or `base`
/// when no row scales it. The move's effect_fn applies this, and the
/// spawn-time patch telegraphs the same value on its `Intent::Buff`.
#[must_use]
pub fn scaled_buff(
    ascension: u8,
    version: sts2_core::GameVersion,
    kind: MonsterKind,
    move_id: &str,
    power: PowerKind,
    base: i32,
) -> i32 {
    let deadly = ascension >= AscensionLevel::DeadlyEnemies as u8;
    let pick = |r: BuffRow| if deadly { r.4 } else { r.3 };
    if version >= sts2_core::GameVersion::V0_111_0 {
        if let Some(r) = buff_row(V0111_BUFF_VALUES, kind, move_id, power) {
            return pick(r);
        }
    }
    buff_row(ASCENSION_BUFF_SCALING, kind, move_id, power).map_or(base, pick)
}

/// The `ascension`-resolved attack damage of move `(kind, move_id)` on
/// build `version` — the value the spawn-time patch
/// ([`apply_ascension_attack_scaling`] then [`apply_v0111_attack_values`])
/// telegraphs — or `base` when no row scales it. For an effect_fn that
/// re-derives a stored attack intent mid-combat (TheForgotten's DREAD), so
/// the recompute starts from the scaled value, not the A0 constant.
#[must_use]
pub fn scaled_damage(
    ascension: u8,
    version: sts2_core::GameVersion,
    kind: MonsterKind,
    move_id: &str,
    base: i32,
) -> i32 {
    let row = |t: &[(MonsterKind, &'static str, i32, i32)]| {
        t.iter()
            .find(|e| e.0 == kind && e.1 == move_id)
            .map(|e| (e.2, e.3))
    };
    let deadly = ascension >= AscensionLevel::DeadlyEnemies as u8;
    if version >= sts2_core::GameVersion::V0_111_0 {
        if let Some((v0111_base, v0111_deadly)) = row(V0111_ATTACK_VALUES) {
            return if deadly { v0111_deadly } else { v0111_base };
        }
    }
    let mut value = base;
    if ascension >= AscensionLevel::ToughEnemies as u8 {
        if let Some((_, tough)) = row(ASCENSION_TOUGH_ATTACK_SCALING) {
            value = tough;
        }
    }
    if deadly {
        if let Some((_, deadly_value)) = row(ASCENSION_ATTACK_SCALING) {
            value = deadly_value;
        }
    }
    value
}

/// Patch `ai` in place to the moves C# telegraphs at `ascension`: the
/// `ToughEnemies` (A8) attack / block rows, then the `DeadlyEnemies` (A9)
/// attack / block / hit-count / buff rows. No-op below A8 (the factory-built
/// A0 values stand) and for a kind with no row.
pub fn apply_ascension_attack_scaling(ai: &mut MonsterAi, kind: MonsterKind, ascension: u8) {
    let tough = ascension >= AscensionLevel::ToughEnemies as u8;
    let deadly = ascension >= AscensionLevel::DeadlyEnemies as u8;
    if !tough {
        return;
    }
    // Most kinds carry no row: leave the shared (prototype) table alone
    // rather than copy it on write for nothing — a summon of a row-less kind
    // (EyeWithTeeth, FatGremlin, Guardbot, Noisebot) stays a refcount bump.
    let has_row = |t: &[(MonsterKind, &str, i32, i32)]| t.iter().any(|e| e.0 == kind);
    let tough_rows =
        has_row(ASCENSION_TOUGH_ATTACK_SCALING) || has_row(ASCENSION_TOUGH_BLOCK_SCALING);
    let deadly_rows = has_row(ASCENSION_ATTACK_SCALING)
        || has_row(ASCENSION_BLOCK_SCALING)
        || ASCENSION_HITS_SCALING.iter().any(|e| e.0 == kind)
        || ASCENSION_BUFF_SCALING.iter().any(|e| e.0 == kind);
    if !(tough_rows || (deadly && deadly_rows)) {
        return;
    }
    let row = |t: &[(MonsterKind, &str, i32, i32)], id: &str| {
        t.iter().find(|e| e.0 == kind && e.1 == id).map(|e| e.3)
    };
    let set_damage = |intents: &mut [Intent], value: i32| {
        for intent in intents {
            match intent {
                Intent::Attack { damage }
                | Intent::DeathBlow { damage }
                | Intent::MultiAttack { damage, .. } => *damage = value,
                _ => {}
            }
        }
    };
    let set_block = |intents: &mut [Intent], value: i32| {
        for intent in intents {
            if let Intent::Defend { block } = intent {
                *block = value;
            }
        }
    };
    for state in std::sync::Arc::make_mut(&mut ai.states).values_mut() {
        let MonsterState::Move(m) = state else {
            continue;
        };
        if let Some(v) = row(ASCENSION_TOUGH_ATTACK_SCALING, m.id) {
            set_damage(&mut m.intents[..], v);
        }
        if let Some(v) = row(ASCENSION_TOUGH_BLOCK_SCALING, m.id) {
            set_block(&mut m.intents[..], v);
        }
        if !deadly {
            continue;
        }
        if let Some(v) = row(ASCENSION_ATTACK_SCALING, m.id) {
            set_damage(&mut m.intents[..], v);
        }
        if let Some(v) = row(ASCENSION_BLOCK_SCALING, m.id) {
            set_block(&mut m.intents[..], v);
        }
        if let Some(entry) = ASCENSION_HITS_SCALING
            .iter()
            .find(|e| e.0 == kind && e.1 == m.id)
        {
            let deadly_hits = entry.3;
            for intent in &mut m.intents {
                if let Intent::MultiAttack { hits, .. } = intent {
                    *hits = deadly_hits;
                }
            }
        }
        for intent in &mut m.intents {
            if let Intent::Buff {
                kind: power,
                amount,
            } = intent
            {
                if let Some(r) = buff_row(ASCENSION_BUFF_SCALING, kind, m.id, *power) {
                    *amount = r.4;
                }
            }
        }
    }
}

/// v0.111.0 attack damages that differ from v0.107.1, as `(kind, move_id,
/// base, deadly)` — the full `GetValueIfAscension(DeadlyEnemies, deadly,
/// base)` pair of the 0.111 property, so a row resolves the value at any
/// ascension on its own. Applied by [`apply_v0111_attack_values`] AFTER the
/// v0.107.1 tables (which stay the 0.107 truth), and only when the combat's
/// `game_version` is `V0_111_0`. GH #796.
#[rustfmt::skip]
pub(crate) const V0111_ATTACK_VALUES: &[(MonsterKind, &str, i32, i32)] = &[
    // v0.111.0: EbbDamage (.., 26, 22); was (.., 32, 26) (decomp Aeonglass.cs:40)
    (MonsterKind::Aeonglass, "EBB_MOVE", 22, 26),
    // v0.111.0: HammerUppercutDamage (.., 18, 14); was (.., 14, 12) (decomp Axebot.cs:46)
    (MonsterKind::Axebot, "HAMMER_UPPERCUT", 14, 18),
    // v0.111.0: OneTwoDamage (.., 11, 10); was (.., 10, 9) (decomp Axebot.cs:42)
    (MonsterKind::Axebot, "ONE_TWO", 10, 11),
    // v0.111.0: FlamethrowerDamage (.., 12, 8) — FLAMETHROWER gained a
    // SingleAttackIntent (decomp MechaKnight.cs:54,85)
    (MonsterKind::MechaKnight, "FLAMETHROWER", 8, 12),
    // v0.111.0: DeGasDamage (.., 18, 16); was (.., 17, 16) (decomp SoulFysh.cs:54)
    (MonsterKind::SoulFysh, "DE_GAS", 16, 18),
    // v0.111.0: StrongTackleDamage (.., 32, 26) — the new opener
    // (decomp TorchHeadAmalgam.cs:32,51)
    (MonsterKind::TorchHeadAmalgam, "STRONG_TACKLE", 26, 32),
    // v0.111.0: TackleDamage (.., 22, 18); was (.., 19, 18) (decomp TorchHeadAmalgam.cs:34)
    (MonsterKind::TorchHeadAmalgam, "TACKLE_2", 18, 22),
    // v0.111.0: WeakTackleDamage (.., 16, 14); was (.., 15, 14) (decomp TorchHeadAmalgam.cs:36)
    (MonsterKind::TorchHeadAmalgam, "TACKLE_3", 14, 16),
    (MonsterKind::TorchHeadAmalgam, "TACKLE_4", 14, 16),
];

/// Overwrite every [`V0111_ATTACK_VALUES`] move's attack damage and every
/// [`V0111_BUFF_VALUES`] move's `Intent::Buff` amount with its v0.111.0
/// value at `ascension`. A no-op for any other build, so v0.107.1
/// combats keep exactly what [`apply_ascension_attack_scaling`] produced.
pub fn apply_v0111_attack_values(
    ai: &mut MonsterAi,
    kind: MonsterKind,
    ascension: u8,
    version: sts2_core::GameVersion,
) {
    if version < sts2_core::GameVersion::V0_111_0 {
        return;
    }
    if !V0111_ATTACK_VALUES.iter().any(|e| e.0 == kind)
        && !V0111_BUFF_VALUES.iter().any(|e| e.0 == kind)
    {
        return;
    }
    let deadly = ascension >= AscensionLevel::DeadlyEnemies as u8;
    for state in std::sync::Arc::make_mut(&mut ai.states).values_mut() {
        let MonsterState::Move(m) = state else {
            continue;
        };
        if let Some(&(_, _, base, deadly_value)) = V0111_ATTACK_VALUES
            .iter()
            .find(|e| e.0 == kind && e.1 == m.id)
        {
            let value = if deadly { deadly_value } else { base };
            for intent in &mut m.intents {
                match intent {
                    Intent::Attack { damage }
                    | Intent::DeathBlow { damage }
                    | Intent::MultiAttack { damage, .. } => *damage = value,
                    _ => {}
                }
            }
        }
        // v0.111.0 buff amounts (the whole pair, so A0 too).
        for intent in &mut m.intents {
            if let Intent::Buff {
                kind: power,
                amount,
            } = intent
            {
                if let Some(r) = buff_row(V0111_BUFF_VALUES, kind, m.id, *power) {
                    *amount = if deadly { r.4 } else { r.3 };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use sts2_core::powers::PowerKind;

    use super::*;
    use crate::monster_dispatch::{
        build_ai, build_ai_with_ascension, MonsterSpawnArgs, SpawnPosition,
    };

    fn move_attack_damage(ai: &MonsterAi, move_id: &str) -> Option<i32> {
        ai.states.values().find_map(|s| match s {
            MonsterState::Move(m) if m.id == move_id => m.intents.iter().find_map(|i| match i {
                Intent::Attack { damage } | Intent::DeathBlow { damage } => Some(*damage),
                Intent::MultiAttack { damage, .. } => Some(*damage),
                _ => None,
            }),
            _ => None,
        })
    }

    /// Every table row must resolve to a real attack move whose A0 damage
    /// equals `base`, and applying DeadlyEnemies scaling must bump it to
    /// `deadly`. This validates the C#-transcribed table against the actual
    /// factories — a bad move-id, wrong base, or non-attack move fails here.
    #[test]
    fn ascension_attack_table_matches_real_moves() {
        let mut fails = Vec::new();
        for &(kind, move_id, base, deadly) in ASCENSION_ATTACK_SCALING {
            if deadly <= base {
                fails.push(format!(
                    "{kind:?} {move_id}: deadly {deadly} <= base {base}"
                ));
                continue;
            }
            let ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
            match move_attack_damage(&ai, move_id) {
                None => fails.push(format!("{kind:?} {move_id}: no attack move with that id")),
                Some(dmg) if dmg != base => {
                    fails.push(format!(
                        "{kind:?} {move_id}: A0 damage {dmg} != table base {base}"
                    ));
                }
                Some(_) => {
                    let mut ai9 = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                    apply_ascension_attack_scaling(&mut ai9, kind, 9);
                    let dmg9 = move_attack_damage(&ai9, move_id).unwrap();
                    if dmg9 != deadly {
                        fails.push(format!(
                            "{kind:?} {move_id}: A9 damage {dmg9} != table deadly {deadly}"
                        ));
                    }
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} bad rows:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// Every ToughEnemies (A8) row resolves to a real move whose A0 value is
    /// `base` and whose A8 value is `tough` — and stays `base` at A7 — through
    /// `build_ai_for_version`, the path combat uses; `tough_block` agrees
    /// with the block table on both sides of the threshold. The A8 sibling of
    /// `ascension_attack_table_matches_real_moves` / `_block_table_` (GH #858).
    #[test]
    fn ascension_tough_tables_match_real_moves() {
        fn first_value(ai: &MonsterAi, move_id: &str, block: bool) -> Option<i32> {
            let Some(MonsterState::Move(m)) = ai.states.get(move_id) else {
                return None;
            };
            m.intents.iter().find_map(|i| match (block, i) {
                (true, Intent::Defend { block }) => Some(*block),
                (false, Intent::Attack { damage } | Intent::MultiAttack { damage, .. }) => {
                    Some(*damage)
                }
                _ => None,
            })
        }
        let mut fails = Vec::new();
        for (table, block) in [
            (ASCENSION_TOUGH_ATTACK_SCALING, false),
            (ASCENSION_TOUGH_BLOCK_SCALING, true),
        ] {
            for &(kind, move_id, base, tough) in table {
                if tough <= base {
                    fails.push(format!("{kind:?} {move_id}: tough {tough} <= base {base}"));
                }
                for version in sts2_core::GameVersion::ALL {
                    for (asc, want) in [(0u8, base), (7, base), (8, tough), (10, tough)] {
                        let ai = crate::monster_dispatch::build_ai_for_version(
                            kind,
                            MonsterSpawnArgs::None,
                            SpawnPosition::default(),
                            asc,
                            version,
                        );
                        let got = first_value(&ai, move_id, block);
                        if got != Some(want) {
                            fails.push(format!(
                                "{kind:?} {move_id} A{asc} {version:?}: {got:?} != {want}"
                            ));
                        }
                    }
                }
                if block {
                    for (asc, want) in [(7u8, base), (8u8, tough)] {
                        let got = tough_block(asc, kind, move_id, base);
                        if got != want {
                            fails.push(format!(
                                "{kind:?} {move_id}: tough_block(A{asc}) = {got} != {want}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} bad rows:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// Regression for the missing-`PECK`-row bug: at A9 `DeadlyEnemies`,
    /// Byrdonis `PECK` (a `MultiAttack`) must scale per-hit damage 3→4 while
    /// keeping hits=3 (C# `Byrdonis.cs:23` `PeckDamage` scales 4/3; `:25`
    /// `PeckRepeat` stays 3/3), so total damage goes 9→12. `SWOOP` (the
    /// already-present row) must still scale 17→19. The whole-table self-test
    /// can only validate rows that EXIST, so a dropped row like `PECK` was
    /// invisible there — this exercises the built AI end-to-end via
    /// `build_ai_with_ascension`, the path combat actually uses.
    #[test]
    fn byrdonis_peck_and_swoop_scale_at_a9() {
        fn move_intent<'a>(ai: &'a MonsterAi, id: &str) -> &'a Intent {
            let Some(MonsterState::Move(m)) = ai.states.get(id) else {
                panic!("no move {id}");
            };
            &m.intents[0]
        }

        let a0 = build_ai_with_ascension(
            MonsterKind::Byrdonis,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            0,
        );
        assert!(matches!(
            move_intent(&a0, "PECK"),
            Intent::MultiAttack { damage: 3, hits: 3 }
        ));
        assert!(matches!(
            move_intent(&a0, "SWOOP"),
            Intent::Attack { damage: 17 }
        ));

        let a9 = build_ai_with_ascension(
            MonsterKind::Byrdonis,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            9,
        );
        assert!(
            matches!(
                move_intent(&a9, "PECK"),
                Intent::MultiAttack { damage: 4, hits: 3 }
            ),
            "PECK must scale per-hit 3->4 and keep hits=3 at A9"
        );
        assert!(matches!(
            move_intent(&a9, "SWOOP"),
            Intent::Attack { damage: 19 }
        ));
    }

    /// C# `Exoskeleton.cs:36` `SkitterRepeats =>
    /// GetValueIfAscension(DeadlyEnemies, 4, 3)`: the A9 scaling bumps the
    /// SKITTER hit COUNT (1×3 → 1×4), not the flat per-hit 1 — the first
    /// member of the [`ASCENSION_HITS_SCALING`] lane. MANDIBLE keeps its
    /// damage-lane 8→9 row (`Exoskeleton.cs:38`). End-to-end via
    /// `build_ai_with_ascension`, the path combat actually uses.
    #[test]
    fn exoskeleton_skitter_hits_scale_at_a9() {
        fn move_intent<'a>(ai: &'a MonsterAi, id: &str) -> &'a Intent {
            let Some(MonsterState::Move(m)) = ai.states.get(id) else {
                panic!("no move {id}");
            };
            &m.intents[0]
        }

        let a0 = build_ai_with_ascension(
            MonsterKind::Exoskeleton,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            0,
        );
        assert!(matches!(
            move_intent(&a0, "SKITTER"),
            Intent::MultiAttack { damage: 1, hits: 3 }
        ));
        assert!(matches!(
            move_intent(&a0, "MANDIBLE"),
            Intent::Attack { damage: 8 }
        ));

        let a9 = build_ai_with_ascension(
            MonsterKind::Exoskeleton,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            9,
        );
        assert!(
            matches!(
                move_intent(&a9, "SKITTER"),
                Intent::MultiAttack { damage: 1, hits: 4 }
            ),
            "SKITTER must scale hits 3->4 (per-hit stays 1) at A9"
        );
        assert!(matches!(
            move_intent(&a9, "MANDIBLE"),
            Intent::Attack { damage: 9 }
        ));

        // A8 (ToughEnemies) is below the DeadlyEnemies threshold — no bump.
        let a8 = build_ai_with_ascension(
            MonsterKind::Exoskeleton,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            8,
        );
        assert!(matches!(
            move_intent(&a8, "SKITTER"),
            Intent::MultiAttack { damage: 1, hits: 3 }
        ));
    }

    /// Every hits-scaling row must resolve to a real MultiAttack move whose
    /// A0 hit count equals `base_hits`, bumped to `deadly_hits` at A9 —
    /// the hits-lane analogue of `ascension_attack_table_matches_real_moves`.
    #[test]
    fn ascension_hits_table_matches_real_moves() {
        fn move_hits(ai: &MonsterAi, move_id: &str) -> Option<u8> {
            ai.states.values().find_map(|s| match s {
                MonsterState::Move(m) if m.id == move_id => {
                    m.intents.iter().find_map(|i| match i {
                        Intent::MultiAttack { hits, .. } => Some(*hits),
                        _ => None,
                    })
                }
                _ => None,
            })
        }
        let mut fails = Vec::new();
        for &(kind, move_id, base, deadly) in ASCENSION_HITS_SCALING {
            if deadly <= base {
                fails.push(format!(
                    "{kind:?} {move_id}: deadly {deadly} <= base {base}"
                ));
                continue;
            }
            let ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
            match move_hits(&ai, move_id) {
                None => fails.push(format!(
                    "{kind:?} {move_id}: no MultiAttack move with that id"
                )),
                Some(h) if h != base => {
                    fails.push(format!(
                        "{kind:?} {move_id}: A0 hits {h} != table base {base}"
                    ));
                }
                Some(_) => {
                    let mut ai9 = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                    apply_ascension_attack_scaling(&mut ai9, kind, 9);
                    let h9 = move_hits(&ai9, move_id).unwrap();
                    if h9 != deadly {
                        fails.push(format!(
                            "{kind:?} {move_id}: A9 hits {h9} != table deadly {deadly}"
                        ));
                    }
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} bad rows:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// Workbench report `wb-010871941d034711` (seed DZV0PD70HW, Ironclad
    /// A9, Underdocks `ToadpolesWeak`): the real game telegraphed WHIRL 8
    /// and SPIKE_SPIT 4x3 while we showed 7 and 3x3 — Toadpole had NO rows
    /// in [`ASCENSION_ATTACK_SCALING`]. C# `Toadpole.cs:49,53`:
    /// `SpikeSpitDamage => GetValueIfAscension(DeadlyEnemies, 4, 3)`,
    /// `WhirlDamage => (.., 8, 7)`.
    #[test]
    fn toadpole_attacks_scale_at_a9() {
        fn move_intent<'a>(ai: &'a MonsterAi, id: &str) -> &'a Intent {
            let Some(MonsterState::Move(m)) = ai.states.get(id) else {
                panic!("no move {id}");
            };
            &m.intents[0]
        }

        let a0 = build_ai_with_ascension(
            MonsterKind::Toadpole,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            0,
        );
        assert!(matches!(
            move_intent(&a0, "SPIKE_SPIT_MOVE"),
            Intent::MultiAttack { damage: 3, hits: 3 }
        ));
        assert!(matches!(
            move_intent(&a0, "WHIRL_MOVE"),
            Intent::Attack { damage: 7 }
        ));

        let a9 = build_ai_with_ascension(
            MonsterKind::Toadpole,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            9,
        );
        assert!(
            matches!(
                move_intent(&a9, "SPIKE_SPIT_MOVE"),
                Intent::MultiAttack { damage: 4, hits: 3 }
            ),
            "SPIKE_SPIT must be 4x3 at A9 (DeadlyEnemies), not 3x3"
        );
        assert!(
            matches!(move_intent(&a9, "WHIRL_MOVE"), Intent::Attack { damage: 8 }),
            "WHIRL must be 8 at A9 (DeadlyEnemies), not 7"
        );
    }

    /// Every [`ASCENSION_BLOCK_SCALING`] row must resolve to a real move
    /// whose A0 `Intent::Defend` block equals `base`, bumped to `deadly` at
    /// A9 — the block-lane analogue of
    /// `ascension_attack_table_matches_real_moves`.
    #[test]
    fn ascension_block_table_matches_real_moves() {
        fn move_block(ai: &MonsterAi, move_id: &str) -> Option<i32> {
            ai.states.values().find_map(|s| match s {
                MonsterState::Move(m) if m.id == move_id => {
                    m.intents.iter().find_map(|i| match i {
                        Intent::Defend { block } => Some(*block),
                        _ => None,
                    })
                }
                _ => None,
            })
        }
        let mut fails = Vec::new();
        for &(kind, move_id, base, deadly) in ASCENSION_BLOCK_SCALING {
            if deadly <= base {
                fails.push(format!(
                    "{kind:?} {move_id}: deadly {deadly} <= base {base}"
                ));
                continue;
            }
            let ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
            match move_block(&ai, move_id) {
                None => fails.push(format!(
                    "{kind:?} {move_id}: no Defend intent on a move with that id"
                )),
                Some(b) if b != base => fails.push(format!(
                    "{kind:?} {move_id}: A0 block {b} != table base {base}"
                )),
                Some(_) => {
                    let mut ai9 = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                    apply_ascension_attack_scaling(&mut ai9, kind, 9);
                    let b9 = move_block(&ai9, move_id).unwrap();
                    if b9 != deadly {
                        fails.push(format!(
                            "{kind:?} {move_id}: A9 block {b9} != table deadly {deadly}"
                        ));
                    }
                }
            }
            // `deadly_block` — the seam the secondary-defend effect_fns use —
            // must agree with the table on both sides of the threshold.
            for (asc, want) in [(8u8, base), (9u8, deadly)] {
                let got = deadly_block(asc, kind, move_id, base);
                if got != want {
                    fails.push(format!(
                        "{kind:?} {move_id}: deadly_block(A{asc}) = {got} != {want}"
                    ));
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} bad block rows:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// The four monsters whose Defend is a SECONDARY intent grant their
    /// block from an `effect_fn` (`perform_intent` only applies a move's
    /// PRIMARY intent), so the intent patch alone would leave the telegraph
    /// and the granted block disagreeing at A9. Drive the effect_fn itself.
    #[test]
    fn secondary_defend_block_grants_scale_at_a9() {
        use sts2_core::{CombatState, MonsterKind};

        let cases: &[(MonsterKind, &str, i32, i32)] = &[
            (MonsterKind::AxeRubyRaider, "SWING_1", 5, 6),
            (MonsterKind::AxeRubyRaider, "SWING_2", 5, 6),
            (MonsterKind::BowlbugEgg, "BITE", 7, 8),
            (MonsterKind::InfestedPrism, "RADIATE", 11, 13),
            (MonsterKind::TheObscura, "HARDENING_STRIKE_MOVE", 6, 7),
        ];
        let mut fails = Vec::new();
        for &(kind, move_id, base, deadly) in cases {
            for (asc, want) in [(0u8, base), (9u8, deadly)] {
                let mut state = CombatState {
                    ascension: asc,
                    ..CombatState::default()
                };
                let uid = state.spawn_enemy(kind);
                let ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                let Some(MonsterState::Move(m)) = ai.states.get(move_id) else {
                    fails.push(format!("{kind:?} {move_id}: no such move"));
                    continue;
                };
                (m.effect)(&mut state, uid);
                let got = state
                    .creatures
                    .iter()
                    .find(|c| c.id == uid.0)
                    .map_or(-1, |c| c.block);
                if got != want {
                    fails.push(format!(
                        "{kind:?} {move_id} A{asc}: granted {got} block != {want}"
                    ));
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} secondary-block mismatches:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// C# `Entomancer.cs:29` `BeesRepeat => (.., 8, 7)` — the A9 scaling
    /// bumps the BEES hit COUNT (3x7 → 3x8); per-hit `BeesDamage` is flat 3.
    #[test]
    fn entomancer_bees_hits_scale_at_a9() {
        fn move_intent<'a>(ai: &'a MonsterAi, id: &str) -> &'a Intent {
            let Some(MonsterState::Move(m)) = ai.states.get(id) else {
                panic!("no move {id}");
            };
            &m.intents[0]
        }
        let a0 = build_ai_with_ascension(
            MonsterKind::Entomancer,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            0,
        );
        assert!(matches!(
            move_intent(&a0, "BEES"),
            Intent::MultiAttack { damage: 3, hits: 7 }
        ));
        let a9 = build_ai_with_ascension(
            MonsterKind::Entomancer,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            9,
        );
        assert!(
            matches!(
                move_intent(&a9, "BEES"),
                Intent::MultiAttack { damage: 3, hits: 8 }
            ),
            "BEES must be 3x8 at A9 (per-hit stays 3)"
        );
    }

    /// C# `AfterAddedToRoom` amounts that scale: CorpseSlug's Ravenous
    /// (`CorpseSlug.cs:51`) and TestSubject's Enrage (`TestSubject.cs:88`).
    /// `on_spawn_self_buffs` carries the A0 amount; `spawn_enemy` must swap
    /// in the DeadlyEnemies value at A9.
    #[test]
    fn on_spawn_self_buffs_scale_at_a9() {
        use sts2_core::powers::PowerKind;
        use sts2_core::{CombatState, MonsterKind};

        fn spawned_amount(kind: MonsterKind, asc: u8, power: PowerKind) -> i32 {
            let mut state = CombatState {
                ascension: asc,
                ..CombatState::default()
            };
            let uid = state.spawn_enemy(kind);
            let creature = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
            let owned: Vec<_> = creature.powers.to_vec();
            state
                .powers
                .iter()
                .filter(|p| p.kind == power && owned.iter().any(|u| u.0 == p.id))
                .map(|p| p.amount)
                .sum()
        }

        for (kind, power, base, deadly) in [
            (MonsterKind::CorpseSlug, PowerKind::Ravenous, 4, 5),
            (MonsterKind::TestSubject, PowerKind::Enrage, 2, 3),
        ] {
            assert_eq!(
                spawned_amount(kind, 8, power),
                base,
                "{kind:?} {power:?} at A8 (below DeadlyEnemies)"
            );
            assert_eq!(
                spawned_amount(kind, 9, power),
                deadly,
                "{kind:?} {power:?} at A9 (DeadlyEnemies)"
            );
        }
    }

    /// The summed `power` stacks a fresh `kind` carries straight after
    /// `CombatState::spawn_enemy` — the real spawn path, which runs
    /// `apply_creature_spawn_buffs` — at `asc` on `version`.
    fn spawned_power(
        kind: MonsterKind,
        asc: u8,
        version: sts2_core::GameVersion,
        power: PowerKind,
    ) -> i32 {
        let mut state = sts2_core::CombatState {
            ascension: asc,
            game_version: version,
            ..sts2_core::CombatState::default()
        };
        let uid = state.spawn_enemy(kind);
        let creature = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
        creature
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == power)
            .map(|p| p.amount)
            .sum()
    }

    /// The decomp dir for `version` (`None` when the `sts2-wiki` sibling is
    /// not checked out, e.g. the public snapshot).
    fn decomp_monsters_dir(version: sts2_core::GameVersion) -> Option<std::path::PathBuf> {
        let tag = match version {
            sts2_core::GameVersion::V0_107_1 => "v0.107.1",
            sts2_core::GameVersion::V0_111_0 => "v0.111.0",
            sts2_core::GameVersion::V0_113_0 => "v0.113.0",
        };
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../../sts2-wiki/decompiled/{tag}/MegaCrit.Sts2.Core.Models.Monsters"
        ));
        dir.is_dir().then_some(dir)
    }

    /// Every ported monster class in `dir`, as `(kind, C# source)`.
    fn ported_monster_sources(dir: &std::path::Path) -> Vec<(MonsterKind, String)> {
        let mut out = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .expect("decomp dir readable")
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for path in entries {
            if path.extension().is_none_or(|e| e != "cs") {
                continue;
            }
            let Some(class) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(&kind) = MonsterKind::ALL_KNOWN
                .iter()
                .find(|k| k.python_name() == class)
            else {
                continue;
            };
            out.push((
                kind,
                std::fs::read_to_string(&path).expect("decomp readable"),
            ));
        }
        out
    }

    /// `(level, scaled, base)` of a `… GetValueIfAscension(AscensionLevel.<L>,
    /// scaled, base)` on `line`, if it has one.
    fn ascension_pair(line: &str) -> Option<(&str, i32, i32)> {
        let (_, rest) = line.split_once("AscensionHelper.GetValueIfAscension(AscensionLevel.")?;
        let (level, rest) = rest.split_once(", ")?;
        let args = rest.split(')').next()?;
        let (scaled, base) = args.split_once(", ")?;
        Some((
            level,
            scaled.trim().parse().ok()?,
            base.trim().parse().ok()?,
        ))
    }

    /// GH #858 decomp guard: every `PowerCmd.Apply<XPower>(…, base.Creature,
    /// <amount>, …)` in a ported monster's `AfterAddedToRoom` whose amount is
    /// a `GetValueIfAscension` property or local must spawn — through the
    /// real `CombatState::spawn_enemy` path — at its base amount one level
    /// below the threshold and at its scaled amount at it, on both builds.
    ///
    /// The DeadlyEnemies table guard counts PROPERTIES, so InfestedPrism's
    /// `VitalSparkAmount` passed it through its PULSATE use while its spawn
    /// use stayed A0; this checks every spawn-time USE. Floors pin today's
    /// scan (10 rows on v0.107.1, 11 on v0.111.0 with GlobeHead's Galvanic),
    /// so a parse that silently skips rows fails. Skips without the decomp.
    #[test]
    fn on_spawn_powers_match_every_scaling_after_added_to_room_amount() {
        use sts2_core::GameVersion;

        for (version, floor) in [(GameVersion::V0_107_1, 10), (GameVersion::V0_111_0, 11)] {
            let Some(dir) = decomp_monsters_dir(version) else {
                eprintln!("skipping: decomp for {version:?} not present");
                return;
            };
            let mut rows = 0usize;
            let mut fails = Vec::new();
            for (kind, src) in ported_monster_sources(&dir) {
                // Class-level properties: name -> (level, scaled, base).
                let mut pairs: Vec<(String, (String, i32, i32))> = Vec::new();
                for line in src.lines() {
                    let Some((level, scaled, base)) = ascension_pair(line) else {
                        continue;
                    };
                    let head = line
                        .split("AscensionHelper.")
                        .next()
                        .unwrap_or("")
                        .trim_end()
                        .trim_end_matches(['=', '>', ' ']);
                    let name = head
                        .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    pairs.push((name, (level.to_string(), scaled, base)));
                }
                // The `AfterAddedToRoom` body, by brace matching.
                let Some(start) = src.find("Task AfterAddedToRoom()") else {
                    continue;
                };
                let open = start + src[start..].find('{').expect("body brace");
                let mut depth = 0i32;
                let mut end = open;
                for (i, c) in src[open..].char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = open + i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                for line in src[open..end].lines() {
                    let Some((_, after)) = line.split_once("PowerCmd.Apply<") else {
                        continue;
                    };
                    let Some((power_class, args)) = after.split_once(">(") else {
                        continue;
                    };
                    // The amount follows the TARGET `base.Creature, `.
                    let Some((_, amount_tail)) = args.split_once("base.Creature, ") else {
                        continue;
                    };
                    let amount = amount_tail.split(',').next().unwrap_or("").trim();
                    let Some((_, (level, scaled, base))) = pairs.iter().find(|(n, _)| n == amount)
                    else {
                        continue; // a literal amount
                    };
                    if scaled == base {
                        continue;
                    }
                    rows += 1;
                    let stem = power_class
                        .strip_suffix("Power")
                        .unwrap_or(power_class)
                        .to_ascii_uppercase();
                    let Some(power) = PowerKind::all_for(version)
                        .into_iter()
                        .find(|p| p.python_name().replace('_', "") == stem)
                    else {
                        fails.push(format!("{kind:?}: no PowerKind for {power_class}"));
                        continue;
                    };
                    let threshold = match level.as_str() {
                        "ToughEnemies" => sts2_core::AscensionLevel::ToughEnemies as u8,
                        "DeadlyEnemies" => sts2_core::AscensionLevel::DeadlyEnemies as u8,
                        other => {
                            fails.push(format!("{kind:?} {power:?}: unexpected level {other}"));
                            continue;
                        }
                    };
                    if !kind.on_spawn_self_buffs().contains(&(power, *base)) {
                        fails.push(format!(
                            "{kind:?}: on_spawn_self_buffs lacks ({power:?}, {base})"
                        ));
                    }
                    for (asc, want) in [(threshold - 1, *base), (threshold, *scaled), (10, *scaled)]
                    {
                        let got = spawned_power(kind, asc, version, power);
                        if got != want {
                            fails.push(format!(
                                "{kind:?} {power:?} A{asc} {version:?}: spawned {got} != C# {want}"
                            ));
                        }
                    }
                }
            }
            assert!(
                rows >= floor,
                "{version:?}: only {rows} scaled AfterAddedToRoom applies parsed (floor {floor})"
            );
            assert!(
                fails.is_empty(),
                "{version:?}: {} on-spawn amount(s) differ from the decomp:\n{}",
                fails.len(),
                fails.join("\n")
            );
        }
    }

    /// The first intent of `move_id` matching `pick`, and its index.
    fn move_intent_where(
        ai: &MonsterAi,
        move_id: &str,
        pick: impl Fn(&Intent) -> bool,
    ) -> Option<(usize, Intent)> {
        let Some(MonsterState::Move(m)) = ai.states.get(move_id) else {
            return None;
        };
        m.intents
            .iter()
            .enumerate()
            .find(|(_, i)| pick(i))
            .map(|(n, i)| (n, *i))
    }

    /// Run `kind`'s `move_id` effect_fn on a fresh `kind` (spawned at `asc` on
    /// `version`, then cut to 1 HP) and return the block + HP it gained.
    fn effect_gain(
        kind: MonsterKind,
        move_id: &str,
        asc: u8,
        version: sts2_core::GameVersion,
    ) -> Option<i32> {
        let mut state = sts2_core::CombatState {
            ascension: asc,
            game_version: version,
            ..sts2_core::CombatState::default()
        };
        let uid = state.spawn_enemy(kind);
        let ai = crate::monster_dispatch::build_ai_for_version(
            kind,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            asc,
            version,
        );
        let Some(MonsterState::Move(m)) = ai.states.get(move_id) else {
            return None;
        };
        let effect = m.effect;
        let snapshot = |st: &sts2_core::CombatState| {
            st.creatures
                .iter()
                .find(|c| c.id == uid.0)
                .map(|c| (c.block, c.current_hp))
                .unwrap()
        };
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
            c.current_hp = 1;
        }
        let (b0, h0) = snapshot(&state);
        effect(&mut state, uid);
        let (b1, h1) = snapshot(&state);
        Some((b1 - b0) + (h1 - h0))
    }

    /// The block a move grants at `asc`: its `Intent::Defend` telegraph, and —
    /// when the Defend is not the primary intent, so `perform_intent` drops
    /// it — what its effect_fn actually grants (the two must agree).
    fn move_block(
        kind: MonsterKind,
        move_id: &str,
        asc: u8,
        version: sts2_core::GameVersion,
    ) -> Result<i32, String> {
        let ai = crate::monster_dispatch::build_ai_for_version(
            kind,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            asc,
            version,
        );
        let Some((at, Intent::Defend { block })) =
            move_intent_where(&ai, move_id, |i| matches!(i, Intent::Defend { .. }))
        else {
            return Err(format!("{kind:?} {move_id}: no Defend intent"));
        };
        if at == 0 {
            return Ok(block);
        }
        let granted = effect_gain(kind, move_id, asc, version)
            .ok_or_else(|| format!("{kind:?} {move_id}: no such move"))?;
        if granted == block {
            Ok(block)
        } else {
            Err(format!(
                "{kind:?} {move_id} A{asc} {version:?}: telegraphs {block} but grants {granted}"
            ))
        }
    }

    /// GH #858 siblings: ToughEnemies (A8) BLOCK members the engine kept at
    /// their A0 value. C#: `LouseProgenitor.cs:65,106` `CurlBlock => (.., 18,
    /// 14)` (CURL_AND_GROW, primary Defend); `InfestedPrism.cs:44,113`
    /// `PulsateBlock => (.., 22, 20)`; `Tunneler.cs:52,100` `BlockGain =>
    /// (.., 37, 32)`; `MagiKnight.cs:49,104,111` `PowerShieldBlock => (.., 9,
    /// 5)` (POWER_SHIELD and PREP); `LagavulinMatriarch.cs:57,219`
    /// `Slash2Block => (.., 14, 12)`. Identical in v0.111.0.
    #[test]
    fn tough_enemies_move_blocks_scale_at_a8_858() {
        use sts2_core::GameVersion;
        let cases: &[(MonsterKind, &str, i32, i32)] = &[
            (MonsterKind::LouseProgenitor, "CURL_AND_GROW_MOVE", 14, 18),
            (MonsterKind::InfestedPrism, "PULSATE", 20, 22),
            (MonsterKind::Tunneler, "BURROW_MOVE", 32, 37),
            (MonsterKind::MagiKnight, "FIRST_POWER_SHIELD", 5, 9),
            (MonsterKind::MagiKnight, "PREP", 5, 9),
            (MonsterKind::LagavulinMatriarch, "SLASH2", 12, 14),
        ];
        let mut fails = Vec::new();
        for &(kind, move_id, base, tough) in cases {
            for version in GameVersion::ALL {
                for (asc, want) in [(0u8, base), (7, base), (8, tough), (10, tough)] {
                    match move_block(kind, move_id, asc, version) {
                        Ok(got) if got == want => {}
                        Ok(got) => fails.push(format!(
                            "{kind:?} {move_id} A{asc} {version:?}: block {got} != {want}"
                        )),
                        Err(e) => fails.push(e),
                    }
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} wrong:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// GH #858 sibling: GremlinMerc's three attacks are ToughEnemies in C#
    /// (`GremlinMerc.cs:36,40,44`: `GimmeDamage => (ToughEnemies, 8, 7)`,
    /// `DoubleSmashDamage => (.., 7, 6)`, `HeheDamage => (.., 9, 8)`; same in
    /// v0.111.0), but sat in the A9 table — wrong at exactly A8.
    #[test]
    fn gremlin_merc_attacks_scale_at_a8_858() {
        use sts2_core::GameVersion;
        let mut fails = Vec::new();
        for (move_id, base, tough) in [("GIMME", 7, 8), ("DOUBLE_SMASH", 6, 7), ("HEHE", 8, 9)] {
            for version in GameVersion::ALL {
                for (asc, want) in [(0u8, base), (7, base), (8, tough), (9, tough)] {
                    let ai = crate::monster_dispatch::build_ai_for_version(
                        MonsterKind::GremlinMerc,
                        MonsterSpawnArgs::None,
                        SpawnPosition::default(),
                        asc,
                        version,
                    );
                    let got = move_attack_damage(&ai, move_id);
                    if got != Some(want) {
                        fails.push(format!("{move_id} A{asc} {version:?}: {got:?} != {want}"));
                    }
                }
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    /// How the engine honours one C# `ToughEnemies` member (GH #858 guard).
    #[derive(Debug)]
    enum Tough {
        /// An `AfterAddedToRoom` self-power, read after `spawn_enemy`.
        SpawnPower(PowerKind),
        /// A move's block — see [`move_block`].
        Block(&'static str),
        /// A move's attack damage.
        Damage(&'static str),
        /// Block + HP a move's effect_fn gives its owner (read inline).
        Effect(&'static str),
        /// `MinInitialHp` through another member
        /// (`TestSubject.MinInitialHp => FirstFormHp`).
        MinHp,
        /// ToughEgg's hatchling HP band ends.
        HatchlingMin,
        HatchlingMax,
        /// TestSubject's HP after its `n`-th Adaptable revive
        /// (`Revive(SecondFormHp)` / `Revive(ThirdFormHp)`).
        ReviveHp(u8),
    }

    /// Every non-HP C# `GetValueIfAscension(AscensionLevel.ToughEnemies, …)`
    /// member of a ported monster, `(class, member, base, tough, uses)` — one
    /// counterpart per C# use site (a member read twice needs both reads
    /// honoured: `CurlBlock` is the spawn CurlUp AND CURL_AND_GROW's block).
    /// `MinInitialHp` / `MaxInitialHp` are checked straight off
    /// `min_initial_hp_for` / `max_initial_hp_for` instead.
    const TOUGH_ENEMIES_MEMBERS: &[(&str, &str, i32, i32, &[Tough])] = &[
        (
            "FrogKnight",
            "PlatingAmount",
            15,
            19,
            &[Tough::SpawnPower(PowerKind::Plating)],
        ),
        (
            "GremlinMerc",
            "GimmeDamage",
            7,
            8,
            &[Tough::Damage("GIMME")],
        ),
        (
            "GremlinMerc",
            "DoubleSmashDamage",
            6,
            7,
            &[Tough::Damage("DOUBLE_SMASH")],
        ),
        ("GremlinMerc", "HeheDamage", 8, 9, &[Tough::Damage("HEHE")]),
        (
            "InfestedPrism",
            "PulsateBlock",
            20,
            22,
            &[Tough::Block("PULSATE")],
        ),
        (
            "LagavulinMatriarch",
            "Slash2Block",
            12,
            14,
            &[Tough::Block("SLASH2")],
        ),
        (
            "LouseProgenitor",
            "CurlBlock",
            14,
            18,
            &[
                Tough::SpawnPower(PowerKind::CurlUp),
                Tough::Block("CURL_AND_GROW_MOVE"),
            ],
        ),
        (
            "MagiKnight",
            "PowerShieldBlock",
            5,
            9,
            &[Tough::Block("FIRST_POWER_SHIELD"), Tough::Block("PREP")],
        ),
        ("Nibbit", "SliceBlock", 5, 6, &[Tough::Effect("SLICE")]),
        (
            "PhantasmalGardener",
            "SkittishAmount",
            6,
            7,
            &[Tough::SpawnPower(PowerKind::Skittish)],
        ),
        (
            "Seapunk",
            "BubbleBlock",
            7,
            8,
            &[Tough::Block("BUBBLE_BURP")],
        ),
        // `SewerClam.cs:32` is a local inside `AfterAddedToRoom`.
        (
            "SewerClam",
            "valueIfAscension",
            8,
            9,
            &[Tough::SpawnPower(PowerKind::Plating)],
        ),
        (
            "SlumberingBeetle",
            "PlatingAmount",
            15,
            18,
            &[Tough::SpawnPower(PowerKind::Plating)],
        ),
        (
            "TerrorEel",
            "ShriekAmount",
            70,
            75,
            &[Tough::SpawnPower(PowerKind::Shriek)],
        ),
        ("TestSubject", "FirstFormHp", 100, 111, &[Tough::MinHp]),
        (
            "TestSubject",
            "SecondFormHp",
            200,
            212,
            &[Tough::ReviveHp(1)],
        ),
        (
            "TestSubject",
            "ThirdFormHp",
            300,
            313,
            &[Tough::ReviveHp(2)],
        ),
        ("ToughEgg", "HatchlingMinHp", 19, 20, &[Tough::HatchlingMin]),
        ("ToughEgg", "HatchlingMaxHp", 22, 23, &[Tough::HatchlingMax]),
        (
            "Tunneler",
            "BlockGain",
            32,
            37,
            &[Tough::Block("BURROW_MOVE")],
        ),
        (
            "Vantom",
            "SlipperyAmt",
            8,
            9,
            &[Tough::SpawnPower(PowerKind::Slippery)],
        ),
        (
            "WaterfallGiant",
            "SiphonHeal",
            10,
            15,
            &[Tough::Effect("SIPHON")],
        ),
    ];

    /// The hatchling HP band ToughEgg's HATCH_MOVE rolls at `asc` on
    /// `version`, as C#'s `(HatchlingMinHp, HatchlingMaxHp)`: v0.107.1 rolls
    /// `NextInt(min, max)` (max-exclusive, `ToughEgg.cs:172`), v0.111.0
    /// `NextInt(min, max + 1)`.
    fn hatchling_band(asc: u8, version: sts2_core::GameVersion) -> (i32, i32) {
        let ai = crate::monster_dispatch::build_ai_for_version(
            MonsterKind::ToughEgg,
            MonsterSpawnArgs::None,
            SpawnPosition::default(),
            asc,
            version,
        );
        let Some(MonsterState::Move(hatch)) = ai.states.get("HATCH_MOVE") else {
            panic!("ToughEgg has no HATCH_MOVE");
        };
        let hatch = hatch.effect;
        let rolls: Vec<i32> = (0..400)
            .map(|seed| {
                let mut state = sts2_core::CombatState {
                    ascension: asc,
                    game_version: version,
                    run_rng: sts2_core::run_rng_set::RunRngSet::new(seed),
                    ..sts2_core::CombatState::default()
                };
                let egg = state.spawn_enemy(MonsterKind::ToughEgg);
                hatch(&mut state, egg);
                state
                    .creatures
                    .iter()
                    .find(|c| c.id == egg.0)
                    .unwrap()
                    .max_hp
            })
            .collect();
        let lo = *rolls.iter().min().unwrap();
        let hi = *rolls.iter().max().unwrap();
        let exclusive = version < sts2_core::GameVersion::V0_111_0;
        (lo, if exclusive { hi + 1 } else { hi })
    }

    /// What the engine yields for one counterpart at `asc` on `version`.
    fn tough_observed(
        kind: MonsterKind,
        use_: &Tough,
        asc: u8,
        version: sts2_core::GameVersion,
    ) -> Result<i32, String> {
        Ok(match use_ {
            Tough::SpawnPower(p) => spawned_power(kind, asc, version, *p),
            Tough::Block(m) => move_block(kind, m, asc, version)?,
            Tough::Damage(m) => {
                let ai = crate::monster_dispatch::build_ai_for_version(
                    kind,
                    MonsterSpawnArgs::None,
                    SpawnPosition::default(),
                    asc,
                    version,
                );
                move_attack_damage(&ai, m).ok_or_else(|| format!("{kind:?} {m}: no attack"))?
            }
            Tough::Effect(m) => effect_gain(kind, m, asc, version)
                .ok_or_else(|| format!("{kind:?} {m}: no such move"))?,
            Tough::MinHp => kind.min_initial_hp_for(asc, version),
            Tough::HatchlingMin => hatchling_band(asc, version).0,
            Tough::HatchlingMax => hatchling_band(asc, version).1,
            Tough::ReviveHp(n) => {
                let mut state = sts2_core::CombatState {
                    ascension: asc,
                    game_version: version,
                    ..sts2_core::CombatState::default()
                };
                let uid = state.spawn_enemy(kind);
                for _ in 0..*n {
                    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
                        c.current_hp = 0;
                    }
                    crate::monsters::act3_glory::adaptable_after_death(&mut state, uid);
                }
                let c = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
                if c.current_hp != c.max_hp {
                    return Err(format!(
                        "{kind:?} revive {n}: hp {}/{}",
                        c.current_hp, c.max_hp
                    ));
                }
                c.max_hp
            }
        })
    }

    /// GH #858 completeness guard for `AscensionLevel.ToughEnemies` (A8),
    /// the sibling of the DeadlyEnemies table guard above.
    ///
    /// Scans every ported monster's decomp, both builds, for
    /// `GetValueIfAscension(AscensionLevel.ToughEnemies, tough, base)` with
    /// `tough != base`:
    /// - `MinInitialHp` / `MaxInitialHp` must equal `min_initial_hp_for` /
    ///   `max_initial_hp_for` at A7 (base) and A8 (tough);
    /// - every other member must have a [`TOUGH_ENEMIES_MEMBERS`] row with the
    ///   same values and ONE counterpart per C# use site, and each counterpart
    ///   must yield `base` at A7 and `tough` at A8 and A10 — checked through
    ///   the real spawn / AI-build / effect paths, never against a table.
    ///
    /// Positive control: the scan must find at least the 149 HP and 22 other
    /// members it finds today on each build, and every table row must be
    /// found. Skips (with a note) when the decomp is not checked out.
    #[test]
    fn tough_enemies_members_all_have_an_engine_counterpart_858() {
        use sts2_core::GameVersion;

        for version in GameVersion::ALL {
            // v0.113.0 (beta) changed monster content the engine has not
            // ported yet (Axebot HP, Infested Prism). Tracked in GH #1303;
            // drop this skip when the monster wave lands.
            if version == GameVersion::V0_113_0 {
                continue;
            }
            let Some(dir) = decomp_monsters_dir(version) else {
                eprintln!("skipping: decomp for {version:?} not present");
                return;
            };
            let mut hp_members = 0usize;
            let mut other_members = 0usize;
            let mut seen_rows = vec![false; TOUGH_ENEMIES_MEMBERS.len()];
            let mut fails = Vec::new();
            for (kind, src) in ported_monster_sources(&dir) {
                let class = kind.python_name();
                let mut enclosing = String::new();
                for line in src.lines() {
                    // The member a lowercase local belongs to is the last
                    // declared property / method name.
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("public ") || trimmed.starts_with("private ") {
                        if let Some(name) = trimmed
                            .split('(')
                            .next()
                            .and_then(|h| h.split(" =>").next())
                            .and_then(|h| h.split_whitespace().last())
                        {
                            enclosing = name.to_string();
                        }
                    }
                    let Some(("ToughEnemies", tough, base)) = ascension_pair(line) else {
                        continue;
                    };
                    if tough == base {
                        continue;
                    }
                    let head = line
                        .split("AscensionHelper.")
                        .next()
                        .unwrap_or("")
                        .trim_end()
                        .trim_end_matches(['=', '>', ' ']);
                    let member = head
                        .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("");
                    if member == "MinInitialHp" || member == "MaxInitialHp" {
                        hp_members += 1;
                        for (asc, want) in [(7u8, base), (8, tough), (10, tough)] {
                            let got = if member == "MinInitialHp" {
                                kind.min_initial_hp_for(asc, version)
                            } else {
                                kind.max_initial_hp_for(asc, version)
                            };
                            if got != want {
                                fails.push(format!(
                                    "{class}.{member} A{asc} {version:?}: engine {got} != C# {want}"
                                ));
                            }
                        }
                        continue;
                    }
                    other_members += 1;
                    let Some(row) = TOUGH_ENEMIES_MEMBERS
                        .iter()
                        .position(|r| r.0 == class && r.1 == member)
                    else {
                        fails.push(format!(
                            "{class}.{member} (ToughEnemies {tough}/{base}, in `{enclosing}`) \
                             has no TOUGH_ENEMIES_MEMBERS row — is it modelled?"
                        ));
                        continue;
                    };
                    seen_rows[row] = true;
                    let (_, _, row_base, row_tough, uses) = TOUGH_ENEMIES_MEMBERS[row];
                    if (row_base, row_tough) != (base, tough) {
                        fails.push(format!(
                            "{class}.{member}: row says {row_base}/{row_tough}, C# {base}/{tough}"
                        ));
                    }
                    // Every read of the member, minus its declaration and the
                    // intent constructors (a telegraph, not an application).
                    let reads = src
                        .lines()
                        .filter(|l| {
                            !l.contains("GetValueIfAscension") && !l.contains("new MoveState(")
                        })
                        .filter(|l| {
                            l.split(|c: char| !c.is_alphanumeric() && c != '_')
                                .any(|w| w == member)
                        })
                        .count();
                    if reads != uses.len() {
                        fails.push(format!(
                            "{class}.{member}: C# reads it {reads}x, the row lists {} use(s)",
                            uses.len()
                        ));
                    }
                    for use_ in uses {
                        for (asc, want) in [(7u8, base), (8, tough), (10, tough)] {
                            match tough_observed(kind, use_, asc, version) {
                                Ok(got) if got == want => {}
                                Ok(got) => fails.push(format!(
                                    "{class}.{member} {use_:?} A{asc} {version:?}: engine {got} != C# {want}"
                                )),
                                Err(e) => fails.push(e),
                            }
                        }
                    }
                }
            }
            for (row, seen) in TOUGH_ENEMIES_MEMBERS.iter().zip(&seen_rows) {
                if !seen {
                    fails.push(format!(
                        "{}.{}: row not found in the {version:?} decomp",
                        row.0, row.1
                    ));
                }
            }
            assert!(
                hp_members >= 149 && other_members >= 22,
                "{version:?}: scan found only {hp_members} HP / {other_members} other \
                 ToughEnemies members — is the parse broken?"
            );
            assert!(
                fails.is_empty(),
                "{version:?}: {} ToughEnemies member(s) not honoured:\n{}",
                fails.len(),
                fails.join("\n")
            );
        }
    }

    /// Properties whose A9 value is honoured somewhere other than the four
    /// tables, listed so the completeness guard below stays exhaustive
    /// instead of silently ignoring a lane. `(class, property, where)`.
    const ACCOUNTED_ELSEWHERE: &[(&str, &str, &str)] = &[
        (
            "Axebot",
            "BootUpStrGain",
            "axebot_boot_up_strength reads it directly; the amount is multiplied \
             by (2 - StockAmount), which the flat buff-case harness cannot express, \
             so axebot_stock_after_death re-telegraphs BOOT_UP once the stock is set",
        ),
        (
            "CorpseSlug",
            "RavenousStr",
            "AfterAddedToRoom power — MonsterKind::on_spawn_self_buff_deadly",
        ),
        (
            "TestSubject",
            "EnrageAmount",
            "AfterAddedToRoom power — MonsterKind::on_spawn_self_buff_deadly",
        ),
        (
            "TestSubject",
            "BurningGrowlBurnCount",
            "a Burn STATUS-CARD count, not a power — \
             test_subject::BURNING_GROWL_BURN_COUNT_DEADLY",
        ),
        (
            "Aeonglass",
            "WitherAmount",
            "a Wither STATUS-CARD count, not a power — read inline by \
             aeonglass_increasing_intensity_effect",
        ),
    ];

    /// Completeness guard against the v0.107.1 decomp, across every lane.
    ///
    /// Each `…  => AscensionHelper.GetValueIfAscension(AscensionLevel.
    /// DeadlyEnemies, deadly, base)` member with `deadly != base` must be
    /// represented on our side: `…Damage` by an [`ASCENSION_ATTACK_SCALING`]
    /// row, `…Block` by [`ASCENSION_BLOCK_SCALING`], `…Repeat(s)` by
    /// [`ASCENSION_HITS_SCALING`], anything else by an
    /// [`ASCENSION_BUFF_SCALING`] case or an [`ACCOUNTED_ELSEWHERE`] entry.
    ///
    /// The sibling `ascension_attack_table_matches_real_moves` only
    /// validates rows that ARE present, so a monster with NO rows passed it.
    /// `wb-010871941d034711` was exactly that: Toadpole dealt its A0 3x3 / 7
    /// at A9. This guard also caught SneakyGremlin, TestSubject and
    /// TheInsatiable in the damage lane, the whole block lane, Entomancer's
    /// BEES hit count and two spawn-time powers.
    ///
    /// Move ids differ between C# (`TACKLE_MOVE`) and our factories
    /// (`TACKLE`), so this counts members per monster rather than matching
    /// ids; a row pointing at the wrong move still trips
    /// `ascension_attack_table_matches_real_moves`. Skips (with a printed
    /// note) when the decomp sibling is not checked out.
    #[test]
    fn ascension_attack_table_covers_every_scaling_decomp_property() {
        use std::collections::BTreeMap;

        let Some(dir) = decomp_monsters_dir(sts2_core::GameVersion::V0_107_1) else {
            eprintln!("skipping: v0.107.1 decomp dir not present");
            return;
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!("skipping: decomp dir not readable at {}", dir.display());
            return;
        };

        // lane name -> class name -> how many values we carry for it.
        let mut have: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
        for &(kind, ..) in ASCENSION_ATTACK_SCALING {
            *have
                .entry("damage")
                .or_default()
                .entry(kind.python_name())
                .or_default() += 1;
        }
        for &(kind, ..) in ASCENSION_BLOCK_SCALING {
            *have
                .entry("block")
                .or_default()
                .entry(kind.python_name())
                .or_default() += 1;
        }
        for &(kind, ..) in ASCENSION_HITS_SCALING {
            *have
                .entry("hits")
                .or_default()
                .entry(kind.python_name())
                .or_default() += 1;
        }
        for &(kind, ..) in ASCENSION_BUFF_SCALING {
            *have
                .entry("other")
                .or_default()
                .entry(kind.python_name())
                .or_default() += 1;
        }
        for &(class, ..) in ACCOUNTED_ELSEWHERE {
            *have.entry("other").or_default().entry(class).or_default() += 1;
        }

        // Same, from the decomp.
        let mut want: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "cs") {
                continue;
            }
            let Some(class) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // Only classes we have actually ported.
            if !MonsterKind::ALL_KNOWN
                .iter()
                .any(|k| k.python_name() == class)
            {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            // C# sometimes assigns the scaled value to a local inside the
            // property body (`TheForgotten.DreadDamage`), so remember the
            // enclosing member and fall back to it for a lowercase token.
            let mut member = "";
            for line in src.lines() {
                if let Some(rest) = line.trim().strip_prefix("private int ") {
                    member = rest
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .next()
                        .unwrap_or("");
                }
                let Some((head, args)) = line.split_once(
                    "AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, ",
                ) else {
                    continue;
                };
                let Some(args) = args.split(')').next() else {
                    continue;
                };
                let Some((deadly, base)) = args.split_once(", ") else {
                    continue;
                };
                if deadly.trim() == base.trim() {
                    continue;
                }
                // Member name: the last identifier before `=>` / `=`.
                let token = head
                    .trim_end()
                    .trim_end_matches(['=', '>', ' '])
                    .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                // A lowercase token is a local (`int valueIfAscension = …`);
                // the lane is decided by the property that wraps it.
                let name = if token.starts_with(|c: char| c.is_lowercase()) {
                    member
                } else {
                    token
                };
                let lane = if name.ends_with("Damage") {
                    "damage"
                } else if name.ends_with("Block") {
                    "block"
                } else if name.ends_with("Repeat") || name.ends_with("Repeats") {
                    "hits"
                } else {
                    "other"
                };
                *want
                    .entry(lane)
                    .or_default()
                    .entry(class.to_string())
                    .or_default() += 1;
            }
        }

        let mut fails = Vec::new();
        for (lane, classes) in &want {
            for (class, n) in classes {
                let got = have
                    .get(lane)
                    .and_then(|m| m.get(class.as_str()))
                    .copied()
                    .unwrap_or(0);
                if got < *n {
                    fails.push(format!(
                        "{class}: {n} scaling `{lane}` member(s) in the decomp, \
                         but we carry only {got}"
                    ));
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} monster/lane pair(s) under-covered by the ascension tables:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// GH #796: every [`V0111_ATTACK_VALUES`] row resolves to a real attack
    /// move of the v0.111.0 build whose A0 damage is `base` and whose A9
    /// damage is `deadly`.
    #[test]
    fn v0111_attack_table_matches_real_moves_v0111() {
        use sts2_core::GameVersion;
        let mut fails = Vec::new();
        for &(kind, move_id, base, deadly) in V0111_ATTACK_VALUES {
            for (asc, want) in [(0_u8, base), (9_u8, deadly)] {
                let ai = crate::monster_dispatch::build_ai_for_version(
                    kind,
                    MonsterSpawnArgs::None,
                    SpawnPosition::default(),
                    asc,
                    GameVersion::V0_111_0,
                );
                match move_attack_damage(&ai, move_id) {
                    Some(d) if d == want => {}
                    got => fails.push(format!("{kind:?} {move_id} A{asc}: {got:?} != {want}")),
                }
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    /// The v0.111.0 table never touches a v0.107.1 build: for every kind it
    /// names, the v0.107.1 AI is exactly what `build_ai_with_ascension` built
    /// before the table existed.
    #[test]
    fn v0111_attack_table_leaves_v0107_untouched() {
        use sts2_core::GameVersion;
        for &(kind, ..) in V0111_ATTACK_VALUES {
            for asc in [0_u8, 9] {
                let mut legacy = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                apply_ascension_attack_scaling(&mut legacy, kind, asc);
                let built = crate::monster_dispatch::build_ai_for_version(
                    kind,
                    MonsterSpawnArgs::None,
                    SpawnPosition::default(),
                    asc,
                    GameVersion::V0_107_1,
                );
                let ids: Vec<_> = legacy.states.keys().copied().collect();
                assert_eq!(ids, built.states.keys().copied().collect::<Vec<_>>());
                for id in ids {
                    assert_eq!(
                        move_attack_damage(&legacy, id),
                        move_attack_damage(&built, id),
                        "{kind:?} {id} A{asc}"
                    );
                }
            }
        }
    }

    /// Below the DeadlyEnemies threshold the patch is a no-op.
    #[test]
    fn ascension_attack_scaling_noop_below_a9() {
        if let Some(&(kind, move_id, base, _)) = ASCENSION_ATTACK_SCALING.first() {
            let mut ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
            apply_ascension_attack_scaling(&mut ai, kind, 8);
            assert_eq!(move_attack_damage(&ai, move_id), Some(base));
        }
    }
    /// DeadlyEnemies (>= A9) buff scaling: triggering each scaled move's
    /// `effect_fn` must apply `base` Strength/Ritual/etc at A0 and `deadly`
    /// at A9. Location-agnostic — asserts the delta in the total amount of
    /// the target `PowerKind` across the whole combat, so it works whether
    /// the buff lands on self or the player. Transcribed from the C#
    /// `GetValueIfAscension(DeadlyEnemies, …)` power amounts.
    #[test]
    fn ascension_buff_scaling_matches_csharp() {
        use sts2_core::powers::PowerKind;
        use sts2_core::CombatState;

        fn total_power(state: &CombatState, kind: PowerKind) -> i32 {
            state
                .powers
                .iter()
                .filter(|p| p.kind == kind)
                .map(|p| p.amount)
                .sum()
        }

        let cases = ASCENSION_BUFF_SCALING;
        let mut fails = Vec::new();
        for &(kind, move_id, power, base, deadly) in cases {
            for (asc, expected) in [(0u8, base), (9u8, deadly)] {
                let mut state = CombatState {
                    ascension: asc,
                    ..CombatState::default()
                };
                let uid = state.spawn_enemy(kind);
                let ai = build_ai(kind, MonsterSpawnArgs::None, SpawnPosition::default());
                let Some(MonsterState::Move(m)) = ai.states.get(move_id) else {
                    fails.push(format!("{kind:?} {move_id}: no such move"));
                    continue;
                };
                let effect = m.effect;
                let before = total_power(&state, power);
                effect(&mut state, uid);
                let got = total_power(&state, power) - before;
                if got != expected {
                    fails.push(format!(
                        "{kind:?} {move_id} {power:?} A{asc}: applied {got} != {expected}"
                    ));
                }
                // The row also drives the telegraph (GH #893).
                for version in sts2_core::GameVersion::ALL {
                    let built = crate::monster_dispatch::build_ai_for_version(
                        kind,
                        MonsterSpawnArgs::None,
                        SpawnPosition::default(),
                        asc,
                        version,
                    );
                    let shown = move_intent_where(
                        &built,
                        move_id,
                        |i| matches!(i, Intent::Buff { kind: p, .. } if *p == power),
                    );
                    if !matches!(shown, Some((_, Intent::Buff { amount, .. })) if amount == expected)
                    {
                        fails.push(format!(
                            "{kind:?} {move_id} {power:?} A{asc} {version:?}: telegraph \
                             {shown:?} != {expected}"
                        ));
                    }
                }
            }
        }
        // v0.111.0-only rows: telegraph and grant on that build, and the
        // v0.107.1 build keeps its own (unscaled) value.
        for &(kind, move_id, power, base, deadly) in V0111_BUFF_VALUES {
            for (asc, expected) in [(0u8, base), (9u8, deadly)] {
                assert_eq!(
                    scaled_buff(
                        asc,
                        sts2_core::GameVersion::V0_111_0,
                        kind,
                        move_id,
                        power,
                        -1
                    ),
                    expected,
                    "{kind:?} {move_id} A{asc}"
                );
                assert_eq!(
                    scaled_buff(
                        asc,
                        sts2_core::GameVersion::V0_107_1,
                        kind,
                        move_id,
                        power,
                        -1
                    ),
                    -1,
                    "{kind:?} {move_id} A{asc} must not scale on v0.107.1"
                );
            }
        }
        assert!(
            fails.is_empty(),
            "{} buff mismatches:\n{}",
            fails.len(),
            fails.join("\n")
        );
    }
}
