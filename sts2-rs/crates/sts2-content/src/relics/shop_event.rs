//! Shop + Ancient (Boss) relic dispatch — bead `sts2-rs-yyw` (P7-5).
//!
//! Covers the `RelicRarity.SHOP` (30 entries) and
//! `RelicRarity.ANCIENT` (92 entries) classes — 122 relics total. The
//! `RelicRarity.EVENT` subset lives in `event_reward.rs`.
//!
//! ## Terminology note
//!
//! The bead description and the game UI both refer to these as
//! "Boss relics". In the Python source they live in the
//! `RelicRarity.ANCIENT` bucket (see
//! `sts2-rl-agent@01c094d:sts2_env/core/enums.py`). Throughout this
//! file `ANCIENT` is the Python rarity and "Boss" is the gameplay
//! label — they refer to the same set.
//!
//! ## Dead Branch caveat
//!
//! The bead spec called for deep pin-tests on `SneckoEye`,
//! `DeadBranch`, and `PandorasBox`. `DeadBranch` does **not** exist in
//! the `01c094d` fork of `sts2-rl-agent` (no `class DeadBranch` in
//! `shop_event.py`, no `RelicId::DEAD_BRANCH` in `base.py`). The deep
//! pin-tests below cover `SneckoEye`, `PandorasBox`, and a third
//! high-impact boss relic (`Brimstone`, which mutates per-turn
//! Strength on both sides — the closest combat-math analogue to Dead
//! Branch's "auto-generate cards" hook in this fork).
//!
//! ## Handler convention
//!
//! Each handler is a marker that returns `EffectResult::Resolved`
//! (matches the canonical pattern landed in `common.rs` /
//! `uncommon.rs` / `rare.rs` by sibling beads `evs`, `l22`, `ge1`).
//! Gameplay plumbing — potion-slot growth, energy modifications,
//! card-reward filters, rest-site additions, gold-gain hooks —
//! deferred to downstream integration beads. Each handler's doc
//! comment records the Python class name + rarity so the wiring
//! bead can locate the source listener in one grep.

use sts2_core::card_effects::{
    apply_power_to_creature, generate_free_attack_in_hand, set_card_var,
    spawn_card_into_hand_with_origin, EffectResult,
};
use sts2_core::cards::{CardType, VarKey};
use sts2_core::hooks::CardPlayCountCtx;
use sts2_core::hooks::{self, BlockCtx, BlockGainedCtx, CreatureRef, ValueProps};
use sts2_core::powers::PowerKind;
use sts2_core::relic_effects::{
    register_after_block_gained, register_after_card_played, register_after_player_turn_start,
    register_after_side_turn_start, register_before_combat_start, register_before_side_turn_start,
    register_before_turn_end, register_before_turn_end_early, register_modify_block_multiplicative,
    register_modify_damage_additive, register_owned_after_card_exhausted,
    register_owned_after_potion_used, register_owned_after_shuffle_bus,
    register_owned_before_hand_draw, register_owned_modify_hand_draw,
    register_owned_modify_x_value, register_owned_relic_modify_card_play_count, AfterBlockGainedFn,
    AfterCardExhaustedFn, AfterCardPlayedFn, AfterPlayerTurnStartFn, AfterPotionUsedFn,
    AfterShuffleFn, AfterSideTurnStartFn, BeforeCombatStartFn, BeforeHandDrawFn,
    BeforeSideTurnStartFn, BeforeTurnEndEarlyFn, BeforeTurnEndFn, CardEventCtx,
    ModifyBlockMultiplicativeFn, ModifyDamageAdditiveFn, ModifyDamageCtx, ModifyHandDrawFn,
    ModifyXValueFn, RelicModifyCardPlayCountFn, ShuffleCtx,
};
use sts2_core::relics::RelicId;
use sts2_core::run_effects::{
    register_after_obtained, register_after_rest_site_heal, register_after_room_entered,
    register_allow_card_reward_reroll, register_modify_card_reward_creation_options,
    register_modify_card_reward_options_late, register_modify_generated_map,
    register_modify_generated_map_late, register_modify_rest_site_options, register_modify_rewards,
    register_modify_unknown_map_point_room_types, register_on_gold_gained,
    register_should_disable_remaining_rest_site_options, register_should_gain_gold,
    register_should_generate_treasure, register_should_procure_potion, AfterObtainedRunFn,
    AfterRestSiteHealRunFn, AfterRoomEnteredRunFn, AllowCardRewardRerollRunFn,
    ModifyCardRewardCreationOptionsRunFn, ModifyCardRewardOptionsLateRunFn,
    ModifyGeneratedMapLateRunFn, ModifyGeneratedMapRunFn, ModifyRestSiteOptionsRunFn,
    ModifyRewardsRunFn, ModifyUnknownMapPointRoomTypesRunFn, OnGoldGainedRunFn,
    ShouldDisableRemainingRestSiteOptionsRunFn, ShouldGainGoldRunFn, ShouldGenerateTreasureRunFn,
    ShouldProcurePotionRunFn,
};
use sts2_core::runstate::{
    CardReward, CardRewardGenerationOptions, PotionReward, RelicRarity, RelicReward,
    RestSiteOption, Reward, RoomType, RunState,
};
use sts2_core::{CombatState, RelicKind, Side};

// ---------------------------------------------------------------------------
// Pool installer
// ---------------------------------------------------------------------------

/// All 122 shop + boss + ancient relic ids owned by this pool, in
/// Python `@register_relic` declaration order. Membership/identity data
/// only — live behaviour flows through the typed hook-channel listeners
/// registered by [`install_pool`], not through any dispatch marker.
/// Backs [`pool_relic_ids`] (the run-loop boss-relic source).
const SHOP_BOSS_RELIC_IDS: &[RelicId] = &[
    RelicId::BeltBuckle,
    RelicId::Bread,
    RelicId::Brimstone,
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
    RelicId::NinjaScroll,
    RelicId::Orrery,
    RelicId::PunchDagger,
    RelicId::RingingTriangle,
    RelicId::RoyalStamp,
    RelicId::RunicCapacitor,
    RelicId::ScreamingFlagon,
    RelicId::SlingOfCourage,
    RelicId::TheAbacus,
    RelicId::Toolbox,
    RelicId::UndyingSigil,
    RelicId::VitruvianMinion,
    RelicId::WingCharm,
    RelicId::AlchemicalCoffer,
    RelicId::ArcaneScroll,
    RelicId::ArchaicTooth,
    RelicId::Astrolabe,
    RelicId::BeautifulBracelet,
    RelicId::BiiigHug,
    RelicId::BlackStar,
    RelicId::BlessedAntler,
    RelicId::BloodSoakedRose,
    RelicId::BoomingConch,
    RelicId::BrilliantScarf,
    RelicId::CallingBell,
    RelicId::ChoicesParadox,
    RelicId::Claws,
    RelicId::Crossbow,
    RelicId::CursedPearl,
    RelicId::DelicateFrond,
    RelicId::DiamondDiadem,
    RelicId::DistinguishedCape,
    RelicId::Driftwood,
    RelicId::DustyTome,
    RelicId::Ectoplasm,
    RelicId::ElectricShrymp,
    RelicId::EmptyCage,
    RelicId::Fiddle,
    RelicId::FurCoat,
    RelicId::GlassEye,
    RelicId::Glitter,
    RelicId::GoldenCompass,
    RelicId::GoldenPearl,
    RelicId::IronClub,
    RelicId::JeweledMask,
    RelicId::JewelryBox,
    RelicId::LargeCapsule,
    RelicId::LavaRock,
    RelicId::LeadPaperweight,
    RelicId::LeafyPoultice,
    RelicId::LoomingFruit,
    RelicId::LordsParasol,
    RelicId::LostCoffer,
    RelicId::MassiveScroll,
    RelicId::MeatCleaver,
    RelicId::MusicBox,
    RelicId::NeowsTorment,
    RelicId::NewLeaf,
    RelicId::NutritiousOyster,
    RelicId::NutritiousSoup,
    RelicId::PaelsBlood,
    RelicId::PaelsClaw,
    RelicId::PaelsEye,
    RelicId::PaelsFlesh,
    RelicId::PaelsGrowth,
    RelicId::PaelsHorn,
    RelicId::PaelsLegion,
    RelicId::PaelsTears,
    RelicId::PaelsTooth,
    RelicId::PaelsWing,
    RelicId::PandorasBox,
    RelicId::PhilosophersStone,
    RelicId::Pomander,
    RelicId::PrecariousShears,
    RelicId::PreciseScissors,
    RelicId::PreservedFog,
    RelicId::PrismaticGem,
    RelicId::PumpkinCandle,
    RelicId::RadiantPearl,
    RelicId::RunicPyramid,
    RelicId::Sai,
    RelicId::SandCastle,
    RelicId::ScrollBoxes,
    RelicId::SeaGlass,
    RelicId::SealOfGold,
    RelicId::SereTalon,
    RelicId::SignetRing,
    RelicId::SilverCrucible,
    RelicId::SmallCapsule,
    RelicId::SneckoEye,
    RelicId::Sozu,
    RelicId::SpikedGauntlets,
    RelicId::StoneHumidifier,
    RelicId::Storybook,
    RelicId::TanxsWhistle,
    RelicId::ThrowingAxe,
    RelicId::ToastyMittens,
    RelicId::TouchOfOrobas,
    RelicId::ToyBox,
    RelicId::TriBoomerang,
    RelicId::VelvetChoker,
    RelicId::VeryHotCocoa,
    RelicId::WarHammer,
    RelicId::WhisperingEarring,
    RelicId::YummyCookie,
];

/// Install every shop/boss-relic handler. Called from
/// [`crate::relics::dispatch::install_default_registry`]. Idempotent
/// (re-registering overwrites the same function pointer).
/// Every [`RelicId`] registered in this module's shop+boss pool — the
/// droppable relics of this tier. Used to seed the run's relic grab
/// bag so combat/treasure/shop relic rewards are stocked (the run
/// loop's `populate_relic_grab_bag_by_rarity` source). Bead `sts2-rs-64mq2`.
#[must_use]
pub fn pool_relic_ids() -> Vec<RelicId> {
    SHOP_BOSS_RELIC_IDS.to_vec()
}

pub fn install_pool() {
    // Bead `sts2-rs-bmw3`: real `after_obtained` listener for
    // `LeesWaffle::MAX_HP = 7` + heal-to-full (Python
    // `shop_event.py::LeesWaffle.after_obtained`). Combat-side
    // `on_trigger_lees_waffle` stays a `resolved_marker`; this is the
    // run-state mutator that actually bumps max HP and tops the player up.
    register_after_obtained(
        RelicId::LeesWaffle,
        listener_lees_waffle as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-r4pr`: real `after_obtained` listeners for
    // `LoomingFruit::MAX_HP = 31` and `NutritiousOyster::MAX_HP = 11`
    // (Python `shop_event.py::LoomingFruit` / `::NutritiousOyster`).
    // Both ANCIENT-pool, pure `gain_max_hp` bodies — same gap shape as
    // Strawberry/Pear/Mango/OldCoin/LeesWaffle.
    register_after_obtained(
        RelicId::LoomingFruit,
        listener_looming_fruit as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::NutritiousOyster,
        listener_nutritious_oyster as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-yfi0`: real `after_obtained` listener for
    // `GoldenPearl::GOLD = 150` (Python `shop_event.py::GoldenPearl`).
    // Same gap shape as OldCoin (tyqo) — combat-side
    // `on_trigger_golden_pearl` stays a `resolved_marker`; this is the
    // run-state mutator that actually grants the gold.
    register_after_obtained(
        RelicId::GoldenPearl,
        listener_golden_pearl as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-2ciy`: real `after_obtained` listener for
    // `SignetRing::GOLD = 999` (Python `shop_event.py::SignetRing`).
    // Same gap shape as GoldenPearl (yfi0) — combat-side
    // `on_trigger_signet_ring` stays a `resolved_marker`; this is the
    // run-state mutator that actually grants the gold.
    register_after_obtained(
        RelicId::SignetRing,
        listener_signet_ring as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-run-rest-option`: PumpkinCandle ANCIENT (v0.107.1) —
    // `AfterObtained → Rekindle` sets `KindleCount` (stored in
    // `rs.relic_counters[PumpkinCandle]`) by adding `kindleAmount = 5`.
    // While > 0 the combat-side `modify_max_energy` arm grants +1 max
    // energy; the count decays 1/combat and the rest-site Kindle option
    // tops it back up.
    register_after_obtained(
        RelicId::PumpkinCandle,
        listener_pumpkin_candle as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-run-rest-option`: PumpkinCandle's
    // `TryModifyRestSiteOptions` appends the KINDLE option to every rest
    // site while owned (C# `PumpkinCandle.cs:77-85`). Mirrors the Shovel
    // `modify_rest_site_options` precedent in `rare.rs`.
    register_modify_rest_site_options(
        RelicId::PumpkinCandle,
        listener_pumpkin_candle_rest_site_options as ModifyRestSiteOptionsRunFn,
    );
    // Bead `sts2-rs-u5emw`: DingyRug SHOP — replace the implicit
    // character pool with an explicit `custom_card_ids` slate
    // (character + colorless) on the
    // `modify_card_reward_creation_options` channel (tu7g5
    // precedent — first consumer of the channel). The combat-side
    // dispatch in `on_trigger_dingy_rug` stays a `Resolved` marker.
    register_modify_card_reward_creation_options(
        RelicId::DingyRug,
        listener_dingy_rug as ModifyCardRewardCreationOptionsRunFn,
    );
    register_modify_card_reward_creation_options(
        RelicId::PrismaticGem,
        listener_prismatic_gem as ModifyCardRewardCreationOptionsRunFn,
    );
    // Bead `sts2-rs-psf9`: real `after_obtained` listener for Orrery.
    // Python `shop_event.py::Orrery.after_obtained` loops `CARDS = 5`
    // calls to `owner.offer_card_reward()`. Rust's `RelicReward` /
    // `CardReward` POD layer is the sentinel surface for umbrella
    // sts2-rs-01o (random reward rolling); same convention as
    // BlackStar (83sa) / PrayerWheel (4ihz) / WhiteStar (b4tb) —
    // push `Reward::Card(CardReward::empty())` placeholders that the
    // 01o roller will materialize.
    register_after_obtained(RelicId::Orrery, listener_orrery as AfterObtainedRunFn);
    // Bead `sts2-rs-beda`: real `after_obtained` listener for
    // `LeafyPoultice::MAX_HP_LOSS = 10` (Python
    // `shop_event.py::LeafyPoultice.after_obtained`). Python also
    // transforms 1 Strike + 1 Defend from the basic deck; that clause
    // is deferred to umbrella sts2-rs-01o / run-side deck transform
    // surface — same gating as StoneCracker / PunchDagger / GnarledHammer.
    register_after_obtained(
        RelicId::LeafyPoultice,
        listener_leafy_poultice as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-u4wv`: real `after_obtained` listener for
    // CallingBell. Python `shop_event.py::CallingBell.after_obtained`
    // adds CurseOfTheBell to the deck and calls
    // `owner.offer_relic_rewards(3, rarities=...)`. The relic-rewards
    // portion uses `rs.offer_relic_rewards(RelicId::CallingBell, 3)`
    // sentinel — umbrella sts2-rs-01o resolves the rarity tiers at
    // flush time. The CurseOfTheBell card-add clause is deferred to
    // umbrella sts2-rs-01o pending run-side deck mutation surface
    // (same gating as NeowsTorment / PaelsHorn).
    register_after_obtained(
        RelicId::CallingBell,
        listener_calling_bell as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-gf5s`: real `after_obtained` listener for
    // SmallCapsule. Python `shop_event.py::SmallCapsule.after_obtained`:
    // `owner.offer_relic_rewards(1)`. Same shape as CallingBell (u4wv)
    // but 1 reward instead of 3.
    register_after_obtained(
        RelicId::SmallCapsule,
        listener_small_capsule as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-2nrk`: real `after_obtained` listener for Cauldron.
    // Python `shop_event.py::Cauldron.after_obtained` does
    // `owner.offer_potions(5)`. Push 5 `Reward::Potion(PotionReward::empty())`
    // sentinels — the dispatcher rolls concrete PotionKinds at flush
    // time (ib04 convention). Same shape as Orrery's CardReward push.
    register_after_obtained(RelicId::Cauldron, listener_cauldron as AfterObtainedRunFn);
    // Bead `sts2-rs-64vb`: real `after_obtained` listener for
    // MassiveScroll. Python `shop_event.py::MassiveScroll.after_obtained`
    // calls `owner.offer_multiplayer_cards(3)`. Coop/multiplayer is
    // out of scope per CLAUDE.md — collapses to 3 regular CardReward
    // offers, same convention as Orrery (psf9).
    register_after_obtained(
        RelicId::MassiveScroll,
        listener_massive_scroll as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.4` (enchant-on-obtain wave): real
    // `after_obtained` listeners for the five static-enchantment
    // shop/event relics (Python `shop_event.py::{GnarledHammer,
    // Kifuda, PunchDagger, BeautifulBracelet, RoyalStamp}`). All
    // share the same `enchant_selected_cards` direct-apply shape,
    // routed through the [`enchant_eligible_deck_cards`] helper.
    // The `defer_followup_rewards` reward-queue arm is gated on
    // umbrella `sts2-rs-01o` (future `Reward::Enchant` variant).
    register_after_obtained(
        RelicId::GnarledHammer,
        listener_gnarled_hammer as AfterObtainedRunFn,
    );
    register_after_obtained(RelicId::Kifuda, listener_kifuda as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::PunchDagger,
        listener_punch_dagger as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::BeautifulBracelet,
        listener_beautiful_bracelet as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::RoyalStamp,
        listener_royal_stamp as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.4` — TriBoomerang: same enchant-on-obtain
    // pattern, Instinct(1) on up to 3 Attack cards. Uses the same
    // `enchant_eligible_deck_cards` helper as the previous wave; cost
    // reduction on non-X-cost attacks fires via
    // `apply_static_enchantment`.
    register_after_obtained(
        RelicId::TriBoomerang,
        listener_tri_boomerang as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.4` — Storybook: push one `BrightestFlame`
    // card to `rs.master_deck` on obtain. Same shape as NeowsTorment
    // (a49q3) / PaelsHorn (a49q3) — pure deck-add via `push_deck_card`.
    register_after_obtained(RelicId::Storybook, listener_storybook as AfterObtainedRunFn);
    // Bead `sts2-rs-70anj.4` — BiiigHug: two-leg listener. Run-side
    // `after_obtained` removes 4 cards from master_deck; combat-side
    // `after_shuffle_bus` pushes one Soot card to the draw pile on
    // every shuffle event while owned.
    register_after_obtained(
        RelicId::BiiigHug,
        listener_biiig_hug_remove as AfterObtainedRunFn,
    );
    register_owned_after_shuffle_bus(
        RelicId::BiiigHug,
        RelicKind::BiiigHug,
        listener_biiig_hug_shuffle as AfterShuffleFn,
    );
    // Bead `sts2-rs-70anj.4` — NinjaScroll: turn 1, before the hand draw,
    // spawns 3 Shivs into the player's hand. C# `NinjaScroll.BeforeHandDraw`
    // (NinjaScroll.cs:23, both builds): it rides the equip-ordered relic
    // before-hand-draw walk with Toolbox (loadout Silent 5189 v0.111.0).
    register_owned_before_hand_draw(
        RelicId::NinjaScroll,
        RelicKind::NinjaScroll,
        listener_ninja_scroll as BeforeHandDrawFn,
    );
    // Bead `sts2-rs-70anj.4` — ElectricShrymp: enchant 1 Skill card
    // with Imbued(1) on obtain. Same helper as the prior wave.
    register_after_obtained(
        RelicId::ElectricShrymp,
        listener_electric_shrymp as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.4` — RadiantPearl: turn 1, before the hand draw,
    // spawns 1 Luminesce into hand. C# `RadiantPearl.BeforeHandDraw`
    // (RadiantPearl.cs:32, both builds); same channel as NinjaScroll.
    register_owned_before_hand_draw(
        RelicId::RadiantPearl,
        RelicKind::RadiantPearl,
        listener_radiant_pearl as BeforeHandDrawFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 5):
    // ToyBox / PaelsClaw / WingCharm.
    register_after_obtained(RelicId::ToyBox, listener_toy_box as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::PaelsClaw,
        listener_paels_claw as AfterObtainedRunFn,
    );
    register_modify_card_reward_options_late(
        RelicId::WingCharm,
        listener_wing_charm_card_reward as ModifyCardRewardOptionsLateRunFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 6):
    // PaelsGrowth — enchant 1 eligible deck card with Clone(4).
    register_after_obtained(
        RelicId::PaelsGrowth,
        listener_paels_growth as AfterObtainedRunFn,
    );
    // ...and offer the CLONE rest-site option while owned
    // (`PaelsGrowth.TryModifyRestSiteOptions`).
    register_modify_rest_site_options(
        RelicId::PaelsGrowth,
        listener_paels_growth_rest_site_options as ModifyRestSiteOptionsRunFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 7):
    // Ectoplasm — should_gain_gold returns False, vetoing every
    // positive gold gain. The +1 max_energy leg is already wired
    // via hooks::modify_max_energy (bead `sts2-rs-70anj.2`).
    register_should_gain_gold(
        RelicId::Ectoplasm,
        listener_ectoplasm_should_gain_gold as ShouldGainGoldRunFn,
    );
    // Bead `sts2-rs-70anj.3.2`: Sozu vetoes every potion procurement
    // via the `should_procure_potion` channel. The +1 max_energy leg
    // is already wired via hooks::modify_max_energy.
    register_should_procure_potion(
        RelicId::Sozu,
        listener_sozu_should_procure_potion as ShouldProcurePotionRunFn,
    );
    // Bead `sts2-rs-70anj.3.5`: Driftwood — unconditionally allow
    // card-reward reroll. The reroll UI consumer is downstream of this
    // channel; the primitive is the bool the consumer reads.
    register_allow_card_reward_reroll(
        RelicId::Driftwood,
        listener_driftwood_allow_reroll as AllowCardRewardRerollRunFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 8):
    // VelvetChoker — per-turn play cap of `VELVET_CHOKER_MAX_CARDS = 6`.
    // `after_card_played` bumps `relic.counter`; `before_side_turn_start`
    // resets it on the PLAYER side. The play gate is
    // hooks::relic_should_play_blocked (reads relic.counter), called by
    // combat_loop::player_should_play_blocked and legal_actions; the +1
    // max_energy leg is the VelvetChoker arm of hooks::modify_max_energy
    // (GH #742 wired both).
    register_after_card_played(
        RelicId::VelvetChoker,
        listener_velvet_choker_card_played as AfterCardPlayedFn,
    );
    register_before_side_turn_start(
        RelicId::VelvetChoker,
        listener_velvet_choker_turn_reset as BeforeSideTurnStartFn,
    );
    // Bead `sts2-rs-70anj.3.21`: DiamondDiadem — at PLAYER turn-end, if
    // <=2 cards were played that turn, apply +1 DiamondDiadem power to
    // the player. The power (PowerKind::DiamondDiadem) is single-stack,
    // halves powered-attack damage targeting owner, and self-removes at
    // ENEMY turn-end via the after_turn_end arm in hooks.rs.
    register_before_turn_end(
        RelicId::DiamondDiadem,
        listener_diamond_diadem_before_turn_end as BeforeTurnEndFn,
    );
    // v0.111.0 rework: 20 Block + Blur 1 at the start of turn 1 (decomp
    // DiamondDiadem.cs:24). Inert in v0.107.1, as the turn-end arm above is
    // in v0.111.0 (which removed DiamondDiademPower).
    register_after_side_turn_start(
        RelicId::DiamondDiadem,
        listener_diamond_diadem_v0111 as AfterSideTurnStartFn,
    );
    // DollysMirror — on obtain, clone the first non-Quest master_deck card
    // (the headless pick of C#'s deck select) via
    // RunState::duplicate_card_from_deck.
    register_after_obtained(
        RelicId::DollysMirror,
        listener_dollys_mirror as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.3.15`: ArcaneScroll — on obtain, add one
    // random Rare card to the master deck. Uses the pool helper
    // `cards::spec_index::random_card_from_rarity`. Character filter
    // is `None` at this pin (RunState has no character field yet;
    // documented pin-divergence vs Python's owner.character_id filter).
    register_after_obtained(
        RelicId::ArcaneScroll,
        listener_arcane_scroll as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.3.15` follow-on: DustyTome — on obtain, add
    // 1 random Ancient-rarity upgraded card. Pool subsets the
    // Transcendence cards present in our slices (Break/Quadcast/Suppress;
    // Protector + MeteorShower live in Necrobinder/Regent slices not
    // yet generated).
    register_after_obtained(
        RelicId::DustyTome,
        listener_dusty_tome as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.3.15` follow-on: SereTalon — on obtain, add
    // 2 random Curses to deck. The "+3 Wish cards" leg is deferred
    // (Wish card not ported).
    register_after_obtained(
        RelicId::SereTalon,
        listener_sere_talon as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.3.17`: StoneHumidifier — gain +5 max HP each
    // time a rest-site REST actually heals. Python
    // `shop_event.py::StoneHumidifier.after_rest_site_heal` at pin
    // `01c094d:sts2_env/relics/shop_event.py:3431`.
    register_after_rest_site_heal(
        RelicId::StoneHumidifier,
        listener_stone_humidifier as AfterRestSiteHealRunFn,
    );
    // Bead `sts2-rs-ekf4j`: MeatCleaver — append Cook option to the
    // rest-site option list if not already present. Python
    // `shop_event.py::MeatCleaver.modify_rest_site_options`.
    register_modify_rest_site_options(
        RelicId::MeatCleaver,
        listener_meat_cleaver_rest_site_options as ModifyRestSiteOptionsRunFn,
    );
    // Bead `sts2-rs-70anj.3.4`: MiniatureTent — vote `Some(false)` on
    // the disable-remaining channel so picking one rest-site option
    // doesn't grey out the rest. Python
    // `shop_event.py::MiniatureTent.should_disable_remaining_rest_site_options`.
    register_should_disable_remaining_rest_site_options(
        RelicId::MiniatureTent,
        listener_miniature_tent_disable_remaining as ShouldDisableRemainingRestSiteOptionsRunFn,
    );
    // Bead `sts2-rs-70anj.3.9`: Astrolabe — on its own pickup, replace
    // 3 random non-curse master-deck cards with random Rare upgrades.
    // Python `shop_event.py::Astrolabe.after_obtained` transforms then
    // upgrades 3 cards. The Rust port samples replacements from the
    // Rare bucket (vs Python's same-rarity-as-original) because there
    // is no `CardId -> CardSpec` lookup on the run side yet; documented
    // divergence — matches the ArcaneScroll/DustyTome pattern of
    // `random_card_from_rarity` + fresh `Card::pod_defaults()`.
    register_after_obtained(RelicId::Astrolabe, listener_astrolabe as AfterObtainedRunFn);
    // Bead `sts2-rs-p6dj`: real `after_side_turn_start` listener for
    // `PaelsFlesh::ENERGY = 1` (Python `shop_event.py::PaelsFlesh`).
    // From player round 3 onward, +1 energy each player turn.
    // Combat-side `on_trigger_paels_flesh` stays a `resolved_marker`;
    // this is the side-turn-start listener that fires the energy gain.
    register_after_side_turn_start(
        RelicId::PaelsFlesh,
        listener_paels_flesh as AfterSideTurnStartFn,
    );
    // Bead `sts2-rs-83sa`: real `modify_rewards` listener for BlackStar.
    // Python `shop_event.py::BlackStar.modify_rewards` appends a
    // `RelicReward` when `room.room_type == ELITE`. Same shape as
    // PrayerWheel (4ihz) / WhiteStar (b4tb) / AmethystAubergine (0463)
    // but appends `Reward::Relic` instead of CardReward/Gold. The
    // "random relic" semantics in Python are deferred — current
    // `RelicReward` POD carries a concrete `relic_id` field with no
    // "pick from N" variant; per the WongosMysteryTicket placeholder
    // convention we push the relic's own id as a sentinel.
    register_modify_rewards(
        RelicId::BlackStar,
        listener_black_star as ModifyRewardsRunFn,
    );
    // Bead `sts2-rs-70anj.1` (Wave 1 first batch): real
    // `after_side_turn_start` listener for `Sai::BLOCK = 7` (Python
    // `shop_event.py::Sai`). Every player turn, gain 7 block unpowered.
    // Combat-side `on_trigger_sai` stays a `resolved_marker`.
    register_after_side_turn_start(RelicId::Sai, listener_sai as AfterSideTurnStartFn);
    // Bead `sts2-rs-70anj.1`: real `after_side_turn_start` listener for
    // `VeryHotCocoa::ENERGY = 4` (Python `shop_event.py::VeryHotCocoa`).
    // Round 1 player turn only.
    register_after_side_turn_start(
        RelicId::VeryHotCocoa,
        listener_very_hot_cocoa as AfterSideTurnStartFn,
    );
    // Bead `sts2-rs-70anj.1` (Wave 1 body-fill batch 2): real
    // `after_card_played` listener for `IronClub::CARD_THRESHOLD = 4`
    // (Python `shop_event.py::IronClub`). Every 4 cards played
    // (persistent counter across turns within a combat), draw 1 card.
    // Per-instance counter tracked on `Relic.counter`.
    register_after_card_played(RelicId::IronClub, listener_iron_club as AfterCardPlayedFn);
    // Bead `sts2-rs-70anj.1`: `before_hand_draw` listener for
    // `ToastyMittens::STRENGTH = 1` (C# `ToastyMittens.BeforeHandDraw`). Each
    // player turn BEFORE the hand is drawn: exhaust the first non-Innate draw
    // card (round 1) / top card (later), then gain 1 Strength. Registered on
    // `before_hand_draw` — NOT `after_side_turn_start`, which fired after the
    // opening draw and exhausted the wrong card (see `listener_toasty_mittens`).
    register_owned_before_hand_draw(
        RelicId::ToastyMittens,
        RelicKind::ToastyMittens,
        listener_toasty_mittens as BeforeHandDrawFn,
    );
    // v0.111.0: ToastyMittens is an `AfterPlayerTurnStart` choose-1-from-
    // hand exhaust (decomp ToastyMittens.cs:31). Inert in v0.107.1, as the
    // `before_hand_draw` arm above is in v0.111.0.
    register_after_player_turn_start(
        RelicId::ToastyMittens,
        listener_toasty_mittens_v0111 as AfterPlayerTurnStartFn,
    );
    // Bead `sts2-rs-70anj.1` (hook corrected 2026-07-17): `before_hand_draw`
    // listener for `JeweledMask` (C# `JeweledMask.BeforeHandDraw`). Turn-1
    // hand draw only, BEFORE the opening draw: pick a random Power card in
    // the draw pile (CombatCardSelection stream), set `FREE_THIS_TURN` on it,
    // and move it to hand. Registered on `before_hand_draw` — NOT
    // `after_side_turn_start`, which fired after the opening draw (see
    // `listener_jeweled_mask` and the ToastyMittens note above).
    register_owned_before_hand_draw(
        RelicId::JeweledMask,
        RelicKind::JeweledMask,
        listener_jeweled_mask as BeforeHandDrawFn,
    );
    // Bead `sts2-rs-h450h` (70anj.5.runic_capacitor): real
    // `after_side_turn_start` listener for `RunicCapacitor` (SHOP,
    // DEFECT pool, `ORB_SLOTS = 3`). Round 1 player-side turn: add 3
    // orb slots via `state.player.orbs.add_slots(3)`. Mirrors the
    // VeryHotCocoa round-1 player-gate pattern.
    register_after_side_turn_start(
        RelicId::RunicCapacitor,
        listener_runic_capacitor as AfterSideTurnStartFn,
    );
    // Bead `sts2-rs-70anj.1`: real `after_card_exhausted` listener for
    // `BurningSticks` (Python `shop_event.py::BurningSticks`). First
    // Skill exhausted per combat: clone it into hand. Per-combat latch
    // via `Relic.counter` (`0` = available, `1` = consumed).
    register_owned_after_card_exhausted(
        RelicId::BurningSticks,
        RelicKind::BurningSticks,
        listener_burning_sticks as AfterCardExhaustedFn,
    );
    register_before_combat_start(
        RelicId::BurningSticks,
        listener_burning_sticks_reset as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-70anj.1`: real `modify_damage_additive` listener
    // for `MysticLighter::EXTRA_DAMAGE = 9` (Python
    // `shop_event.py::MysticLighter`). Enchanted Attack cards dealt by
    // the player get +9 damage on `MOVE` damage events.
    register_modify_damage_additive(
        RelicId::MysticLighter,
        listener_mystic_lighter as ModifyDamageAdditiveFn,
    );
    // Bead `sts2-rs-70anj.1` (Wave 1 body-fill batch 3): real
    // `before_combat_start` + `modify_hand_draw` legs for `SneckoEye`
    // (Python `shop_event.py::SneckoEye`). At combat start apply
    // `PowerKind::Confused` (stack 1) to the player; every turn add
    // `SNECKO_EYE_EXTRA_DRAW = 2` to the player's hand-draw count.
    // Mirrors `before_combat_start` (Confused application) and
    // `modify_hand_draw` (+EXTRA_DRAW = 2).
    register_before_combat_start(
        RelicId::SneckoEye,
        listener_snecko_eye_combat_start as BeforeCombatStartFn,
    );
    register_owned_modify_hand_draw(
        RelicId::SneckoEye,
        RelicKind::SneckoEye,
        listener_snecko_eye_hand_draw as ModifyHandDrawFn,
    );
    // Bead `sts2-rs-70anj.3.7`: ChemicalX adds +2 to every X-cost play's
    // effective `x` value. Fires from combat_loop::play_card just before
    // `card.energy_spent` is recorded.
    register_owned_modify_x_value(
        RelicId::ChemicalX,
        RelicKind::ChemicalX,
        listener_chemical_x_modify_x_value as ModifyXValueFn,
    );
    // Bead `sts2-rs-70anj.3.3`: ThrowingAxe — first card each combat is
    // played twice. Per-combat latch lives on `Relic.counter` (0 = unused,
    // 1 = fired). The companion `before_combat_start` listener resets the
    // latch so the next combat re-arms (Python's `before_combat_start` +
    // `after_combat_end` both clear `self._used_this_combat`).
    register_owned_relic_modify_card_play_count(
        RelicId::ThrowingAxe,
        RelicKind::ThrowingAxe,
        listener_throwing_axe_play_count as RelicModifyCardPlayCountFn,
    );
    register_before_combat_start(
        RelicId::ThrowingAxe,
        listener_throwing_axe_reset as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-70anj.3.12`: DelicateFrond — fills empty potion
    // slots with random non-event, character-locked potions at combat
    // start. Mirrors `relics/shop_event.py::DelicateFrond.before_combat_start`
    // (`sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py:1476`).
    register_before_combat_start(
        RelicId::DelicateFrond,
        listener_delicate_frond_combat_start as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-sei89`: GhostSeed — stamp `EtherealThisCombat` on
    // every basic Strike/Defend in any of the player's piles at combat
    // start. Mirrors Python `shop_event.py::GhostSeed.before_combat_
    // start`. The companion `after_card_entered_combat` arm (covers
    // mid-combat additions) is a follow-up — the bead scope is
    // explicitly the combat-start sweep only.
    register_before_combat_start(
        RelicId::GhostSeed,
        listener_ghost_seed_combat_start as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-70anj.3.13.tb`: Toolbox — round-1 only, offer 3
    // colorless cards via AddGeneratedToHand choice. Per-combat latch
    // on Relic.counter resets at combat start. Mirrors Python
    // `shop_event.py::Toolbox.before_hand_draw` + `before_combat_start`.
    register_owned_before_hand_draw(
        RelicId::Toolbox,
        RelicKind::Toolbox,
        listener_toolbox_before_hand_draw as BeforeHandDrawFn,
    );
    register_before_combat_start(
        RelicId::Toolbox,
        listener_toolbox_reset as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-70anj.3.13.cp`: ChoicesParadox — round-1 only,
    // offer 5 retain-tagged character cards via
    // AddGeneratedRetainedToHand choice. C# overrides `AfterPlayerTurnStart`
    // (ChoicesParadox.cs:27), so it rides that equip-ordered relic walk, which
    // suspends on its pick: a later `AfterPlayerTurnStart` relic (RoyalPoison's
    // 4 damage) lands only after the card is chosen. It was on
    // `after_side_turn_start`, so RoyalPoison fired before the pick (GH #1054,
    // Ironclad/Defect-1000 act-3 `player.hp` splits).
    // Per-combat latch on Relic.counter resets at combat start.
    register_after_player_turn_start(
        RelicId::ChoicesParadox,
        listener_choices_paradox_turn_start as AfterPlayerTurnStartFn,
    );
    register_before_combat_start(
        RelicId::ChoicesParadox,
        listener_choices_paradox_reset as BeforeCombatStartFn,
    );
    // `BrilliantScarf`: the relic's `counter` is C#'s `CardsPlayedThisTurn`,
    // reset here at the player's side-turn start
    // (`BrilliantScarf.BeforeSideTurnStart`) and counted by its own
    // `AfterCardPlayed` listener, in its equip slot of the relic walk. The
    // "fifth card costs 0" hooks read it in sts2-core
    // `hooks::brilliant_scarf_makes_free`.
    register_before_side_turn_start(
        RelicId::BrilliantScarf,
        listener_brilliant_scarf_turn_reset as BeforeSideTurnStartFn,
    );
    register_after_card_played(
        RelicId::BrilliantScarf,
        listener_brilliant_scarf_card_played as AfterCardPlayedFn,
    );
    // `BlessedAntler`. C# v0.107.1 `BlessedAntler.BeforeHandDraw`
    // (`TurnNumber == 1`): add `BLESSED_ANTLER_DAZED_COUNT = 3` Dazed to the
    // owner's draw pile at CardPilePosition.Random BEFORE the turn-1 hand is
    // drawn, so a Dazed can land in the opening hand. Registered on
    // `before_hand_draw`, NOT `before_side_turn_start` (the prior port, citing
    // the Python pin): that channel fires earlier than C#'s, which reorders
    // this relic's three Shuffle-stream reads against the other
    // `before_hand_draw` generators (FuneraryMask's Souls). Same move as
    // FuneraryMask / ToastyMittens (`sts2-rs-70anj.1`). The companion
    // `modify_max_energy: +1` leg is deferred — no `modify_max_energy`
    // register channel exists yet (the noop_hook in `hooks.rs` would need a
    // RelicKind match-arm). Beads `sts2-rs-70anj.1`, `sts2-rs-lfdaa`.
    register_owned_before_hand_draw(
        RelicId::BlessedAntler,
        RelicKind::BlessedAntler,
        listener_blessed_antler as BeforeHandDrawFn,
    );
    // Bead `sts2-rs-vfzgs`: `PaelsEye` listeners. Python
    // `shop_event.py::PaelsEye` uses two per-instance flags
    // (`_used_this_combat`, `_any_cards_played`) packed into bit 0 and
    // bit 1 of `Relic.counter` respectively. The
    // `should_take_extra_turn` predicate + `_used_this_combat` latch
    // live in the engine surface (`hooks.rs` + `combat_loop.rs`); the
    // content-side legs wired here are:
    //   - `before_side_turn_start` (PLAYER) → clear bit 1.
    //   - `before_turn_end_early` (PLAYER) → exhaust the entire hand when
    //     extra-turn conditions hold.
    // Bit 1 ("a non-auto card play finished this turn") is set by the
    // engine's hand-play path (`combat_loop::bump_paels_eye_played`), not
    // the relic `after_card_played` bus: that bus also fires for auto-plays,
    // which C# `AnyCardsPlayedThisTurn` skips (`!CardPlay.IsAutoPlay`).
    register_before_side_turn_start(
        RelicId::PaelsEye,
        listener_paels_eye_side_turn_start as BeforeSideTurnStartFn,
    );
    // C# `PaelsEye.BeforeSideTurnEndEarly` (PaelsEye.cs:118, both builds):
    // the Early walk, before Screaming Flagon's normal `BeforeSideTurnEnd`.
    register_before_turn_end_early(
        RelicId::PaelsEye,
        listener_paels_eye_before_turn_end as BeforeTurnEndEarlyFn,
    );
    register_before_combat_start(
        RelicId::PaelsEye,
        listener_paels_eye_reset as BeforeCombatStartFn,
    );
    // Bead `sts2-rs-ad53n`: `MusicBox` listeners. Python
    // `shop_event.py::MusicBox` — first Attack played per player turn,
    // spawn an Ethereal clone into hand. Per-turn latch is bit 0 of
    // `Relic.counter`. The Ethereal-on-clone bit rides on the per-card
    // `VarKey::EtherealThisCombat` slot (honored by the end-of-turn
    // ethereal sweep in `combat_loop::end_player_turn_phase_two`).
    // GH #1110: C# latches the first Attack to START (`BeforeCardPlayed`)
    // and clones only that card when it finishes, not the first to finish.
    sts2_core::relic_effects::register_before_card_played(
        RelicId::MusicBox,
        listener_music_box_before_card_played,
    );
    register_after_card_played(
        RelicId::MusicBox,
        listener_music_box_card_played as AfterCardPlayedFn,
    );
    register_before_side_turn_start(
        RelicId::MusicBox,
        listener_music_box_side_turn_start as BeforeSideTurnStartFn,
    );
    // Bead `sts2-rs-zh82z`: `Crossbow.after_side_turn_start` — every
    // player turn, generate one random 0-cost Attack from the player's
    // character pool into hand. Engine surface
    // (`card_effects::generate_free_attack_in_hand`) handles the
    // sampler + free-this-turn flag.
    register_after_side_turn_start(RelicId::Crossbow, listener_crossbow as AfterSideTurnStartFn);
    // Bead `sts2-rs-1e0ae`: `SealOfGold.after_side_turn_start` — every
    // player turn, if gold >= 5, spend 5 and gain 1 energy. Routes
    // through the closure-registry gold accessors
    // (`card_effects::read_player_gold` + `try_spend_player_gold`).
    // No-ops gracefully in core-only test builds where no accessor is
    // installed.
    register_after_side_turn_start(
        RelicId::SealOfGold,
        listener_seal_of_gold as AfterSideTurnStartFn,
    );
    // 2026-07-17 soft-drop census: `Bread.AfterSideTurnStart` — on the
    // FIRST player turn, lose `BREAD_LOSE_ENERGY = 2` energy (C#
    // `MegaCrit.Sts2.Core.Models.Relics/Bread.cs:AfterSideTurnStart`).
    // The +1 max-energy leg (turns 2+) already lives in the inline
    // `modify_max_energy` arm (`hooks.rs`, bead `sts2-rs-70anj.2`);
    // this listener closes the missing turn-1 penalty leg.
    register_after_side_turn_start(RelicId::Bread, listener_bread as AfterSideTurnStartFn);
    // Bead `sts2-rs-70anj.3` (Wave 3 first primitive — starter_relic_swap):
    // `TouchOfOrobas.setup_for_player`-style behavior wired as an
    // `after_obtained` listener. Python pin
    // `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::TouchOfOrobas`
    // maps the 5 starter relics to their upgraded variants:
    //   BURNING_BLOOD → BLACK_BLOOD (Ironclad)
    //   RING_OF_THE_SNAKE → RING_OF_THE_DRAKE (Silent)
    //   CRACKED_CORE → INFUSED_CORE (Defect)
    //   BOUND_PHYLACTERY → PHYLACTERY_UNBOUND (Necrobinder)
    //   DIVINE_RIGHT → DIVINE_DESTINY (Regent)
    // We mutate `rs.relics` in place at obtain time.
    register_after_obtained(
        RelicId::TouchOfOrobas,
        listener_touch_of_orobas as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.3`: `LargeCapsule::after_obtained` offers
    // 2 random relic rewards (mirror of `SmallCapsule`'s 1-reward
    // landing under `sts2-rs-gf5s`). Python pin
    // `shop_event.py::LargeCapsule.RELICS = 2` plus a deferred
    // Strike+Defend deck-add clause (requires shop_event-side
    // `add_card_to_master_deck` plumbing, separate primitive).
    register_after_obtained(
        RelicId::LargeCapsule,
        listener_large_capsule as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 shop_event body-fill, batch 1):
    // JewelryBox + TanxsWhistle both port to single `master_deck.push`
    // calls at obtain time. Python pin
    // `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py`:
    // - JewelryBox.after_obtained: `owner.add_card_to_deck(
    //   CardInstance(CardId.APOTHEOSIS))` — adds 1 unupgraded
    //   Apotheosis (Colorless rare scaling card) to the run deck.
    // - TanxsWhistle.after_obtained: `owner.add_card_to_deck(
    //   CardInstance(CardId.WHISTLE))` — adds 1 unupgraded Whistle
    //   (Colorless common scaling card) to the run deck.
    // Both fall through the existing `RunState.master_deck`
    // Vec<Card> surface via a `Card::pod_defaults()` + `card_id` /
    // `upgraded` override (the same shape `mk_deck_card` uses
    // privately in `runstate.rs` tests). No `add_card_to_master_deck`
    // helper exists yet on RunState; that's deferred to a follow-up
    // refactor bead — these listeners push directly per the
    // CLAUDE.md "don't add features beyond what the task requires"
    // rule. The deck-add hook `on_card_added_to_deck_run` channel
    // would fire here in production but isn't drained from this
    // path at this pin (out of scope for the body-fill bead).
    register_after_obtained(
        RelicId::JewelryBox,
        listener_jewelry_box as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::TanxsWhistle,
        listener_tanxs_whistle as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-r2y4x` (70anj.4.5): two more run-side
    // `after_obtained` listeners.
    //
    // DistinguishedCape: lose 9 max HP via C#-faithful `lose_max_hp`
    // (partial — Apparition card-add deferred pending `add_card_to_deck`).
    // Same shape as LeafyPoultice (`sts2-rs-beda`).
    //
    // SandCastle: upgrade 6 random deck cards via `upgrade_random_cards`
    // (full implementation — the `defer_followup_rewards` reward-offer
    // path follows the LeesWaffle / SignetRing convention of taking
    // the non-defer Python fallback). Wires through the
    // `upgrade_random_cards` primitive landed by `sts2-rs-7krx0`.
    register_after_obtained(
        RelicId::DistinguishedCape,
        listener_distinguished_cape as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::SandCastle,
        listener_sand_castle as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-66l9f` (70anj.4.6): three more reward-sentinel
    // pushers, all following the Orrery / MassiveScroll / LostCoffer
    // convention of pushing empty `Reward::Card` / `Reward::Potion`
    // placeholders deferred to umbrella `sts2-rs-01o` for materialization.
    //
    // GlassEye: 5 Reward::Card sentinels (rarity-forcing deferred).
    // LostCoffer: 1 Reward::Card + 1 Reward::Potion sentinel.
    // ScrollBoxes (v0.107.1): generate 2 card bundles (no gold loss).
    register_after_obtained(RelicId::GlassEye, listener_glass_eye as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::LostCoffer,
        listener_lost_coffer as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::ScrollBoxes,
        listener_scroll_boxes as AfterObtainedRunFn,
    );
    // v0.107.1 Ancient/Neow relic (bead `sts2-rs-utgn4`): 2 cross-character
    // card rewards generated on pickup (Niche-shuffled non-owner pools +
    // base-odds card creation on Rewards). See `listener_kaleidoscope`.
    register_after_obtained(
        RelicId::Kaleidoscope,
        listener_kaleidoscope as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-a49q3` (70anj.4.7): four more after_obtained
    // listeners using the direct-deck-push pattern established by
    // JewelryBox (k3y) / TanxsWhistle (k3y) — confirms add_card_to_deck
    // doesn't need a dedicated RunState primitive when `Card::pod_defaults()`
    // suffices for adding curses / status cards.
    //
    // NeowsTorment: +1 NeowsFury card.
    // PaelsHorn: +2 Relax cards.
    // CursedPearl: +1 Greed curse + 333 gold (single relic, both legs).
    // BloodSoakedRose: +1 Enthralled curse (modify_max_energy combat
    //   leg deferred to that channel).
    register_after_obtained(
        RelicId::NeowsTorment,
        listener_neows_torment as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::PaelsHorn,
        listener_paels_horn as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::CursedPearl,
        listener_cursed_pearl as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::BloodSoakedRose,
        listener_blood_soaked_rose as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-qw9oe` (70anj.4.8): three more after_obtained
    // listeners.
    //
    // LeadPaperweight: 2 Reward::Card sentinels (colorless filter
    //   deferred to 01o roller).
    // SeaGlass: optional 0..=15 grid of another character's cards.
    // YummyCookie: upgrade_random_cards(4) — same SandCastle (r2y4x)
    //   convention of using random-upgrade for select-N Python paths.
    register_after_obtained(
        RelicId::LeadPaperweight,
        listener_lead_paperweight as AfterObtainedRunFn,
    );
    register_after_obtained(RelicId::SeaGlass, listener_sea_glass as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::YummyCookie,
        listener_yummy_cookie as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-cvbxs` (70anj.4.9): three more after_obtained
    // listeners.
    //
    // Pomander: upgrade_random_cards(1) — SandCastle / YummyCookie
    //   convention.
    // PrecariousShears: remove 2 cards, then lose_hp(16). Remove-2-cards
    //   leg deferred pending remove_cards_from_deck primitive.
    // AlchemicalCoffer: gain_potion_slots(4), then fill the 4 new slots
    //   with 4 random out-of-combat potions (bead `sts2-rs-cvbxs` closed).
    register_after_obtained(RelicId::Pomander, listener_pomander as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::PrecariousShears,
        listener_precarious_shears as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::AlchemicalCoffer,
        listener_alchemical_coffer as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-t762n`: deck-remove listeners enabled by the
    // newly-landed `RunState::remove_cards_from_deck` primitive.
    //
    // PreciseScissors: remove 1 card.
    // EmptyCage: remove 2 cards.
    // PreservedFog: remove 5 cards + add 1 Folly curse.
    // PrecariousShears: already registered above with the partial
    //   (lose_hp) leg; the listener body was upgraded in this bead
    //   to also call remove_cards_from_deck(2).
    register_after_obtained(
        RelicId::PreciseScissors,
        listener_precise_scissors as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::EmptyCage,
        listener_empty_cage as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::PreservedFog,
        listener_preserved_fog as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-rvwas`: ArchaicTooth — starter-card transform via
    // direct master_deck mutation (preserves upgrade + enchantment
    // state automatically since the card POD's other fields are
    // untouched).
    register_after_obtained(
        RelicId::ArchaicTooth,
        listener_archaic_tooth as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-w9vf8`: Claws + NutritiousSoup deck mutators.
    //
    // Claws: transform up to 6 deck cards into Maul via in-place
    //   card_id rewrites (ArchaicTooth pattern; preserves upgrade /
    //   enchantments).
    // NutritiousSoup: set TezcatarasEmber on every basic-Strike card
    //   (idempotent static enchantment).
    register_after_obtained(RelicId::Claws, listener_claws as AfterObtainedRunFn);
    register_after_obtained(
        RelicId::NutritiousSoup,
        listener_nutritious_soup as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-5xg1w`: Glitter — apply Glam(1) to every card-
    // reward option as it's generated. Same channel as FresnelLens.
    register_modify_card_reward_options_late(
        RelicId::Glitter,
        listener_glitter_card_reward as ModifyCardRewardOptionsLateRunFn,
    );
    // Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 2):
    // DragonFruit ports to a single `on_gold_gained` listener that
    // tops `rs.max_hp` by 1 per positive gold gain. Python pin
    // `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::DragonFruit`:
    // ```
    // def on_gold_gained(self, owner, amount):
    //     if amount > 0:
    //         owner.gain_max_hp(self.MAX_HP)  # MAX_HP = 1
    // ```
    // The `on_gold_gained` run-side channel landed under bead
    // `sts2-rs-wag75`; this is the second consumer after BowlerHat.
    // ownership-gate via `rs.relics.contains` matches the canonical
    // run-side listener pattern (LeesWaffle, FragrantMushroom, …).
    register_on_gold_gained(
        RelicId::DragonFruit,
        listener_dragon_fruit_on_gold_gained as OnGoldGainedRunFn,
    );
    // Bead `sts2-rs-ytiok` (70anj.4.10): NewLeaf — open a 1-card
    // transform prompt on pickup. Prompt-opener branch only; the
    // in-place transform branch is blocked on a character-pool
    // sampler that requires `RunState.character_id` to land (same
    // gap as jya0y's `roll_card_reward_cards`).
    register_after_obtained(RelicId::NewLeaf, listener_new_leaf as AfterObtainedRunFn);
    // Bead `sts2-rs-70anj.4` (PandorasBox sub-pick): real
    // `after_obtained` listener for `PandorasBox` (Python
    // `relics/shop_event.py::PandorasBox.after_obtained` —
    // `owner.transform_specific_cards(owner.basic_strike_defend_cards())`).
    // Unblocked by ab8uo (`transform_specific_cards`), xbkhd
    // (`RunState.character_id`), and the existing
    // `basic_strike_defend_cards` lookup in spec_index.rs. Combat-side
    // `on_trigger_pandoras_box` stays a `resolved_marker`.
    register_after_obtained(
        RelicId::PandorasBox,
        listener_pandoras_box as AfterObtainedRunFn,
    );
    // Bead `sts2-rs-535i7` (70anj.4 handoff): SilverCrucible —
    // multi-channel relic that (a) counts treasure-room entries via
    // `after_room_entered`, (b) vetoes the first treasure room via
    // `should_generate_treasure` (channel elcvh), and (c) pre-upgrades
    // the first 3 card rewards via `modify_card_reward_options_late`
    // (channel qotxh). The two per-relic state fields Python keeps
    // (`_times_used`, `_treasure_rooms_entered`) are packed into a
    // single `rs.relic_counters[SilverCrucible]` i32 slot — low 8 bits
    // for times_used, next 8 bits for treasure_rooms_entered (both
    // bounded < 256 over any real run). Helpers
    // `silver_crucible_{pack,times_used,treasure_rooms}` keep the
    // encoding cohesive. Combat-side `on_trigger_silver_crucible`
    // stays a `resolved_marker`.
    register_after_room_entered(
        RelicId::SilverCrucible,
        listener_silver_crucible_after_room_entered as AfterRoomEnteredRunFn,
    );
    register_should_generate_treasure(
        RelicId::SilverCrucible,
        listener_silver_crucible_should_generate_treasure as ShouldGenerateTreasureRunFn,
    );
    register_modify_card_reward_options_late(
        RelicId::SilverCrucible,
        listener_silver_crucible_modify_card_reward_options_late
            as ModifyCardRewardOptionsLateRunFn,
    );
    // FurCoat — two-channel listener. after_obtained latches the obtain
    // act and marks the current act map; modify_generated_map_late marks a
    // map generated later in that act. The combat-side 1-HP clamp on a
    // marked room lives in `run_bridge::enter_combat_from_roster`.
    register_after_obtained(
        RelicId::FurCoat,
        listener_fur_coat_after_obtained as AfterObtainedRunFn,
    );
    register_modify_generated_map_late(
        RelicId::FurCoat,
        listener_fur_coat_modify_generated_map_late as ModifyGeneratedMapLateRunFn,
    );
    // Bead `sts2-rs-k2nuu` (70anj.4 handoff): GoldenCompass — three
    // listeners. `after_obtained` captures `rs.act_index` into
    // `rs.relic_counters[GoldenCompass]`; `modify_generated_map`
    // overwrites the rolled act_map with `generate_golden_path_map(1)`
    // when the captured act matches; `modify_unknown_map_point_room_types`
    // narrows the `?`-node candidate set to {Event}. Python pin
    // `relics/shop_event.py::GoldenCompass` also calls
    // `run_state.generate_map()` from `after_obtained` to force
    // immediate regen; the Rust port lacks a `RunState::generate_map()`
    // analogue (RunState doesn't own an `ActMap` field at this Rust
    // pin), so the swap only activates on the next driver-side
    // regeneration. Documented divergence; combat-side
    // `on_trigger_golden_compass` stays a `resolved_marker`.
    register_after_obtained(
        RelicId::GoldenCompass,
        listener_golden_compass_after_obtained as AfterObtainedRunFn,
    );
    register_modify_generated_map(
        RelicId::GoldenCompass,
        listener_golden_compass_modify_generated_map as ModifyGeneratedMapRunFn,
    );
    register_modify_unknown_map_point_room_types(
        RelicId::GoldenCompass,
        listener_golden_compass_modify_unknown_map_point_room_types
            as ModifyUnknownMapPointRoomTypesRunFn,
    );
    // BeltBuckle — "while you have no potions, gain +BELT_BUCKLE_DEXTERITY
    // Dexterity." C# `BeltBuckle.cs` latches `_dexterityApplied` and
    // applies/removes `DexterityPower(2)` as the player's potion count
    // crosses 0. The combat-side achievable subset at this pin:
    //   * `before_combat_start` resets the latch and applies +2 Dex when
    //     the player starts combat with no potions;
    //   * `after_potion_used` applies +2 Dex when using the last potion
    //     leaves the player potionless (latch-guarded).
    // `AfterPotionProcured` (-2 Dex when a potion is gained mid-combat) is
    // `CombatState::try_procure_potion` in sts2-core, the one combat-side
    // procure. `AfterPotionDiscarded` has no combat-side discard to hook. The latch
    // lives in `relic.counter` (0 = not applied, 1 = applied). Bead
    // `sts2-rs-xu8d6`.
    register_before_combat_start(
        RelicId::BeltBuckle,
        listener_belt_buckle_before_combat_start as BeforeCombatStartFn,
    );
    register_owned_after_potion_used(
        RelicId::BeltBuckle,
        RelicKind::BeltBuckle,
        listener_belt_buckle_after_potion_used as AfterPotionUsedFn,
    );
    // PaelsLegion (sts2-rs-ppabi) — boss/pet relic: "the first time you
    // gain block from a card each [cooldown] turns, double it." C#
    // `PaelsLegion.cs` summons a (behaviourless) pet and runs a
    // block-doubling cooldown machine:
    //   * BeforeCombatStart    → summon pet + reset cooldown (armed);
    //   * ModifyBlockMultiplicative → 2× owner block from a card while
    //     off cooldown;
    //   * AfterModifyingBlockAmount + AfterCardPlayed → start the
    //     `Turns`-turn cooldown once the triggering card-play resolves;
    //   * AfterSideTurnStart    → decrement the cooldown each owner turn.
    // The cooldown lives in `relic.counter` with the sentinel scheme
    //   0  = armed (ready to double),
    //   <0 = doubled during the play of card uid `-counter` (C#'s
    //        `AffectedCardPlay`), cooldown pending that play's end,
    //   >0 = turns of cooldown remaining.
    // The pet is cosmetic (animation triggers only); the block-doubling is
    // the mechanic. See the listener doc-comments.
    register_before_combat_start(
        RelicId::PaelsLegion,
        listener_paels_legion_before_combat_start as BeforeCombatStartFn,
    );
    register_modify_block_multiplicative(
        RelicId::PaelsLegion,
        listener_paels_legion_modify_block as ModifyBlockMultiplicativeFn,
    );
    register_after_block_gained(
        RelicId::PaelsLegion,
        listener_paels_legion_after_block_gained as AfterBlockGainedFn,
    );
    register_after_card_played(
        RelicId::PaelsLegion,
        listener_paels_legion_after_card_played as AfterCardPlayedFn,
    );
    register_after_side_turn_start(
        RelicId::PaelsLegion,
        listener_paels_legion_after_side_turn_start as AfterSideTurnStartFn,
    );
}

/// `PaelsLegion::before_combat_start` — summon the behaviourless
/// PaelsLegion pet and reset the cooldown to armed (`relic.counter = 0`).
/// Mirrors C# `PaelsLegion.cs::BeforeCombatStart` (`SummonPet`) plus the
/// `AfterCombatEnd` reset (`Cooldown = 0`). Bead `sts2-rs-ppabi`.
fn listener_paels_legion_before_combat_start(state: &mut CombatState) -> EffectResult {
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
    else {
        return EffectResult::NotImplemented;
    };
    state.relics[idx].counter = 0;
    let _ = state.summon_pet(sts2_core::MonsterKind::PaelsLegion, PAELS_LEGION_PET_HP);
    EffectResult::Resolved
}

/// `PaelsLegion::modify_block_multiplicative` — double (`2×`) block the
/// player gains from a card while off cooldown. Mirrors C#
/// `PaelsLegion.cs::ModifyBlockMultiplicative`: gates on
/// `props.IsCardOrMonsterMove()` + `cardSource != null` (here
/// `ctx.card_source.is_some()`), `target == Owner` (the player), and
/// `Cooldown <= 0` (sentinel `counter <= 0`, covering both armed `0` and
/// the pending `-1`). Returns identity otherwise. Bead `sts2-rs-ppabi`.
fn listener_paels_legion_modify_block(state: &CombatState, ctx: &BlockCtx) -> hooks::DamageMul {
    if ctx.card_source.is_none() {
        return hooks::DamageMul::ONE;
    }
    if !matches!(ctx.target, CreatureRef::Player) {
        return hooks::DamageMul::ONE;
    }
    let counter = state
        .relics
        .iter()
        .find(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
        .map(|r| r.counter);
    let on_cooldown = counter.is_some_and(|c| c > 0);
    // v0.113.0 `PaelsLegion.cs:152-156`: while a doubling is pending on one
    // `CardPlay` (`AffectedCardPlay`), a different card's block is no longer
    // doubled.
    let latched_elsewhere = state.game_version >= sts2_core::GameVersion::V0_113_0
        && counter.is_some_and(|c| c < 0)
        && ctx
            .card_source
            .is_some_and(|src| paels_legion_pending(src) != counter.unwrap_or(0));
    let owned = state.relics.owns(RelicKind::PaelsLegion);
    if owned && !on_cooldown && !latched_elsewhere {
        hooks::DamageMul::new(2, 1)
    } else {
        hooks::DamageMul::ONE
    }
}

/// `PaelsLegion::after_block_gained` — once a doubled card-block lands
/// (armed, card-sourced, owner-targeted), mark the cooldown pending with
/// the card's negated uid. Subsequent block gains in the SAME card-play stay
/// `<= 0`, so they are still doubled by
/// [`listener_paels_legion_modify_block`] — matching C#'s
/// `AfterModifyingBlockAmount` latching the whole `CardPlay` before
/// `AfterCardPlayed` starts the cooldown. A pending latch is kept
/// (`AffectedCardPlay != null && AffectedCardPlay != cardPlay` returns).
/// Bead `sts2-rs-ppabi`.
fn listener_paels_legion_after_block_gained(
    state: &mut CombatState,
    ctx: &BlockGainedCtx,
) -> EffectResult {
    let Some(source) = ctx.card_source else {
        return EffectResult::NotImplemented;
    };
    if !matches!(ctx.target, CreatureRef::Player) {
        return EffectResult::NotImplemented;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
    else {
        return EffectResult::NotImplemented;
    };
    if state.relics[idx].counter == 0 {
        state.relics[idx].counter = paels_legion_pending(source);
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// The pending sentinel for a doubling latched by the play of `card`.
fn paels_legion_pending(card: sts2_core::CardUid) -> i32 {
    i32::try_from(card.0.get()).map_or(i32::MIN, |uid| -uid)
}

/// `PaelsLegion::after_card_played` — when the card-play that triggered
/// the doubling resolves (`counter` pending on this card's uid), start the
/// `PAELS_LEGION_TURNS`-turn cooldown. Mirrors C#
/// `PaelsLegion.cs::AfterCardPlayed` (`AffectedCardPlay == cardPlay` →
/// `Cooldown = DynamicVars["Turns"]`): a different card played inside the
/// latched play (a Hellraiser auto-play) finishes first and leaves it
/// pending. Bead `sts2-rs-ppabi`.
fn listener_paels_legion_after_card_played(
    state: &mut CombatState,
    ctx: &CardEventCtx,
) -> EffectResult {
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
    else {
        return EffectResult::NotImplemented;
    };
    let counter = state.relics[idx].counter;
    let latched = counter < 0
        && ctx
            .card_uid
            .is_none_or(|uid| paels_legion_pending(uid) == counter);
    if latched {
        state.relics[idx].counter = PAELS_LEGION_TURNS;
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `PaelsLegion::after_side_turn_start` — decrement the cooldown each
/// owner (player) turn start; at `0` the relic is armed again. Mirrors
/// C# `PaelsLegion.cs::AfterSideTurnStart` (`if side == Owner.Side:
/// Cooldown--`). A pending (negative) latch is left untouched (the cooldown has not
/// started yet). Bead `sts2-rs-ppabi`.
fn listener_paels_legion_after_side_turn_start(
    state: &mut CombatState,
    side: Side,
) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
    else {
        return EffectResult::NotImplemented;
    };
    if state.relics[idx].counter > 0 {
        state.relics[idx].counter -= 1;
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// Shared BeltBuckle apply helper: when the player owns BeltBuckle, holds
/// no potions, and the latch (`relic.counter`) is unset, fold
/// +`BELT_BUCKLE_DEXTERITY` Dexterity into the player and set the latch.
/// Mirrors C# `BeltBuckle.ApplyDexterity` (`if (!DexterityApplied) {
/// DexterityApplied = true; PowerCmd.Apply<DexterityPower>(+2); }`).
/// Returns `true` iff Dexterity was applied this call.
fn belt_buckle_apply_dexterity(state: &mut CombatState) -> bool {
    if !state.player.potions.is_empty() {
        return false;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::BeltBuckle)
    else {
        return false;
    };
    if state.relics[idx].counter != 0 {
        return false;
    }
    state.relics[idx].counter = 1;
    if let Some(uid) = state.player.creature_uid {
        apply_power_to_creature(state, uid, PowerKind::Dexterity, BELT_BUCKLE_DEXTERITY);
    }
    true
}

/// `BeltBuckle::before_combat_start` — reset the `_dexterityApplied`
/// latch (`relic.counter` → 0) and, if the player holds no potions,
/// apply +`BELT_BUCKLE_DEXTERITY` Dexterity. Mirrors C#
/// `BeltBuckle.BeforeCombatStart` (`DexterityApplied = false; if
/// (!Owner.Potions.Any()) ApplyDexterity();`). Bead `sts2-rs-xu8d6`.
fn listener_belt_buckle_before_combat_start(state: &mut CombatState) -> EffectResult {
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::BeltBuckle)
    else {
        return EffectResult::NotImplemented;
    };
    state.relics[idx].counter = 0;
    if belt_buckle_apply_dexterity(state) {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `BeltBuckle::after_potion_used` — when using a potion leaves the
/// player with no potions, apply +`BELT_BUCKLE_DEXTERITY` Dexterity
/// (latch-guarded). Mirrors C# `BeltBuckle.AfterPotionUsed` (`if
/// (IsInProgress && !Owner.Potions.Any()) ApplyDexterity();`).
///
/// The `RemoveDexterity` (-2 Dex) path of C# `AfterPotionProcured` lives in
/// sts2-core's `CombatState::try_procure_potion`, which every combat-side
/// procure (Alchemize, Entropic Brew, Petrified Toad, ...) goes through.
fn listener_belt_buckle_after_potion_used(
    state: &mut CombatState,
    _potion: sts2_core::potions::PotionKind,
    _target: Option<u8>,
) -> EffectResult {
    if belt_buckle_apply_dexterity(state) {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `TouchOfOrobas::after_obtained` — starter-relic swap (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::TouchOfOrobas`).
/// At obtain time, iterate `rs.relics` and replace the first
/// occurrence of any starter-relic id with its character-specific
/// upgraded variant:
///   `BurningBlood` → `BlackBlood` (Ironclad)
///   `RingOfTheSnake` → `RingOfTheDrake` (Silent)
///   `CrackedCore` → `InfusedCore` (Defect)
///   `BoundPhylactery` → `PhylacteryUnbound` (Necrobinder)
///   `DivineRight` → `DivineDestiny` (Regent)
///   `SongBow` → `Storyteller` (The Fool, v0.113.0)
/// An already-refined starter (no `RefinementUpgrades` entry) becomes
/// Circlet. Bead `sts2-rs-70anj.3`.
fn listener_touch_of_orobas(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::TouchOfOrobas) {
        return EffectResult::NotImplemented;
    }
    // C# `TouchOfOrobas.AfterObtained` (`TouchOfOrobas.cs:158-165`): the
    // FIRST relic with `Rarity == Starter` (`GetStarterRelic`) is
    // `RelicCmd.Replace`d, in its slot, by its `RefinementUpgrades` entry —
    // or by Circlet when it has none (an already-refined starter).
    let Some(slot) = rs.relics.iter().position(|r| r.is_starter_rarity()) else {
        return EffectResult::NotImplemented;
    };
    let old = rs.relics[slot];
    rs.relics[slot] = old.refined_starter().unwrap_or(RelicId::Circlet);
    if !rs.relics.contains(&old) {
        rs.relic_counters.remove(&old);
    }
    EffectResult::Resolved
}

/// `LargeCapsule::after_obtained` — offers 2 random relic rewards
/// (Python pin `shop_event.py::LargeCapsule.RELICS = 2`, plus a
/// Strike+Defend deck-add clause deferred — that requires the
/// shop_event-side `add_card_to_master_deck` plumbing). Mirror of
/// [`listener_small_capsule`] using `RelicId::LargeCapsule` as the
/// random-sampling TBD sentinel. Bead `sts2-rs-70anj.3`.
fn listener_large_capsule(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LargeCapsule) {
        return EffectResult::NotImplemented;
    }
    // C# `LargeCapsule.AfterObtained` (LargeCapsule.cs:23-35): OBTAIN 2 relics
    // directly (`await RelicCmd.Obtain` — no pick), THEN add the character's
    // basic Strike and Defend to the deck (Strike first). The obtain-BEFORE-add
    // order is load-bearing: an egg relic just obtained here (e.g. MoltenEgg)
    // upgrades the Strike as it is added via the `modify_card_being_added_to_deck`
    // hook. Offering the relics as pickable rewards (the old shape) left them
    // unowned at add-time, so MoltenEgg never saw the Strike (RP1 PARITY36:
    // C# adds an upgraded Strike, Rust an unupgraded one). The grab-bag pop
    // order is unchanged (same `roll_reward_relic_rarity` + `pop_relic_from_grab_bag`).
    for _ in 0..LARGE_CAPSULE_RELICS {
        let rarity = rs.roll_reward_relic_rarity();
        if let Some(id) = rs.pop_relic_from_grab_bag(rarity) {
            rs.obtain_relic(id);
        }
    }
    // Restore the pickup marker the inner obtains overwrote, so the rest of
    // this after_obtained walk (and the NeowsBones grant chain, when LargeCapsule
    // was granted by it) still sees LargeCapsule as the owner.
    rs.last_obtained_relic = Some(RelicId::LargeCapsule);
    let character_id = rs.character_id;
    if let Some([strike_id, defend_id]) =
        crate::cards::spec_index::basic_strike_defend_cards(character_id)
    {
        for id in [strike_id, defend_id] {
            if let Some(mut card) = crate::cards::spec_index::card_pod_from_id(id, false) {
                // C# `CardPileCmd.Add` runs every relic's
                // `TryModifyCardBeingAddedToDeck` (MoltenEgg upgrades Attacks,
                // FrozenEgg Powers, ToxicEgg Skills).
                rs.modify_card_being_added_to_deck(&mut card);
                // C# `CardPileCmd.Add(.., PileType.Deck)` fires
                // `AbstractRelic.AfterCardChangedPiles` for every relic, which
                // is where `LuckyFysh.cs` grants its `GoldVar(15)` per card that
                // lands in the deck (LuckyFysh.cs:AfterCardChangedPiles). Capture
                // the card identity/type BEFORE the moving push, then dispatch the
                // run-side `on_card_added_to_deck` channel so LuckyFysh (and any
                // other deck-add listener) fires for each of the Strike and Defend
                // — without this the two adds were raw-pushed silently and
                // LuckyFysh never granted its 30 gold (parity map_gold_minus30).
                let ctx = sts2_core::run_effects::CardAddedRunCtx {
                    card_id: card.card_id,
                    card_type: card.card_type,
                    source: sts2_core::relic_effects::AddSource::RelicGain,
                };
                rs.master_deck.push(card);
                let _ = sts2_core::run_effects::dispatch_on_card_added_to_deck(rs, &ctx);
            }
        }
    }
    EffectResult::Resolved
}

/// `LargeCapsule.RELICS = 2` (Python pin `shop_event.py:LargeCapsule.RELICS`).
pub const LARGE_CAPSULE_RELICS: usize = 2;

/// `JewelryBox::after_obtained` — Python pin
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::JewelryBox`:
/// adds one unupgraded `Apotheosis` (Colorless rare scaling card,
/// `CardId::Apotheosis`) to the run-side master deck. Bead
/// `sts2-rs-70anj.4` (Wave 4 body-fill, batch 1). Helper-free direct
/// push to `rs.master_deck` — same shape as the `mk_deck_card`
/// pattern used in `runstate.rs` tests, kept inline per CLAUDE.md
/// "don't add features beyond what the task requires" until a
/// second consumer justifies hoisting a `RunState::add_card_to_deck`
/// helper.
fn listener_jewelry_box(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::JewelryBox) {
        return EffectResult::NotImplemented;
    }
    // `CardPileCmd.Add(CreateCard<Apotheosis>, Deck)` (`JewelryBox.cs:21-22`).
    push_deck_card(rs, sts2_core::cards::CardId::Apotheosis);
    EffectResult::Resolved
}

/// `TanxsWhistle::after_obtained` — Python pin
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::TanxsWhistle`:
/// adds one unupgraded `Whistle` (Colorless common scaling card,
/// `CardId::Whistle`) to the run-side master deck. Bead
/// `sts2-rs-70anj.4` (Wave 4 body-fill, batch 1). Same direct-push
/// shape as [`listener_jewelry_box`].
fn listener_tanxs_whistle(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::TanxsWhistle) {
        return EffectResult::NotImplemented;
    }
    // `CardPileCmd.Add(CreateCard<Whistle>, Deck)` (`TanxsWhistle.cs:19-20`).
    push_deck_card(rs, sts2_core::cards::CardId::Whistle);
    EffectResult::Resolved
}

/// `DragonFruit::on_gold_gained` — Python pin
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::DragonFruit`:
/// `if amount > 0: owner.gain_max_hp(self.MAX_HP)` with `MAX_HP = 1`.
/// Fires on every positive gold gain at run-state level. Bead
/// `sts2-rs-70anj.4` (Wave 4 body-fill, batch 2). Ownership-gated to
/// avoid mutating the run for non-owners (sibling channel listeners
/// dispatch on all registered relics, not just owned ones).
fn listener_dragon_fruit_on_gold_gained(rs: &mut RunState, amount: i32) -> EffectResult {
    if !rs.owns_active_relic(RelicId::DragonFruit) {
        return EffectResult::NotImplemented;
    }
    if amount <= 0 {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(DRAGON_FRUIT_MAX_HP);
    EffectResult::Resolved
}

/// `Sai::after_side_turn_start` — `if side == PLAYER:
/// owner.gain_block(7, unpowered=True)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::Sai`,
/// `BLOCK = 7`). Fires every player turn for the entire combat.
/// Bead `sts2-rs-70anj.1`.
fn listener_sai(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::Sai) {
        return EffectResult::NotImplemented;
    }
    // C# `Sai.cs` declares `BlockVar(7m, ValueProp.Unpowered)` —
    // bypass Frail/Dexterity (bead `sts2-rs-dko39`).
    gain_block_player_unpowered(state, SAI_BLOCK);
    EffectResult::Resolved
}

/// Bead `sts2-rs-dko39`: file-local block grant that bypasses
/// Frail/Dexterity — mirrors C# `BlockVar(N, ValueProp.Unpowered)`.
fn gain_block_player_unpowered(state: &mut CombatState, base: i32) {
    let ctx = BlockCtx {
        target: CreatureRef::Player,
        props: ValueProps::MOVE.union(ValueProps::UNPOWERED),
        card_source: None,
    };
    let final_block = hooks::modify_block(state, base, &ctx);
    state.player.block = state.player.block.saturating_add(final_block);
    state.sync_creature_block_from_player();
    if final_block > 0 {
        let gained_ctx = BlockGainedCtx {
            target: CreatureRef::Player,
            amount: final_block,
            card_source: None,
        };
        hooks::dispatch_after_block_gained(state, &gained_ctx);
        let _ = sts2_core::relic_effects::dispatch_after_block_gained(state, &gained_ctx);
    }
}

/// `VeryHotCocoa::after_side_turn_start` — `if side == PLAYER and
/// combat.round_number == 1: combat.gain_energy(owner, 4)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::VeryHotCocoa`,
/// `ENERGY = 4`). Round 1 only. Bead `sts2-rs-70anj.1`.
fn listener_very_hot_cocoa(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::VeryHotCocoa) {
        return EffectResult::NotImplemented;
    }
    state.gain_energy(VERY_HOT_COCOA_ENERGY);
    EffectResult::Resolved
}

/// `PaelsFlesh::after_side_turn_start` — `if side == PLAYER and
/// combat.round_number >= 3: combat.gain_energy(owner, 1)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::PaelsFlesh`,
/// `ENERGY = 1`). Single-channel, no per-instance latch (the
/// `round_no` gate alone is sufficient — fires every player turn
/// from round 3 to combat end). Bead `sts2-rs-p6dj`.
fn listener_paels_flesh(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::PaelsFlesh) {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() < PAELS_FLESH_ROUND_MIN {
        return EffectResult::NotImplemented;
    }
    state.gain_energy(PAELS_FLESH_ENERGY);
    EffectResult::Resolved
}

/// `IronClub::after_card_played` — increments a per-instance counter
/// on every player card play (persistent across turns within a combat)
/// and triggers `combat.draw_cards(owner, 1)` every
/// `IRON_CLUB_CARD_THRESHOLD = 4` plays (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::IronClub`).
/// Counter lives on `Relic.counter`. Bead `sts2-rs-70anj.1`.
fn listener_iron_club(state: &mut CombatState, _ctx: &CardEventCtx) -> EffectResult {
    let mut should_draw = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::IronClub {
            relic.counter = relic.counter.saturating_add(1);
            if relic.counter % IRON_CLUB_CARD_THRESHOLD == 0 {
                should_draw = true;
            }
            break;
        }
    }
    if should_draw {
        // `CardPileCmd.Draw(choiceContext, 1, Owner)` (IronClub.cs:92) is a
        // card-effect draw, so `Hook.ShouldDraw` vetoes it under NoDraw
        // (CardPileCmd.cs:826-830): the 4th card of a Battle Trance turn
        // draws nothing (act23 Ironclad:1012).
        let _ = state.draw_into_hand_card_effect(1);
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `ToastyMittens::before_side_turn_start` — `if side == PLAYER:
/// combat.exhaust_top_of_draw_pile(owner); owner.apply_power(STRENGTH,
/// 1)` (Python `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::
/// ToastyMittens`, `STRENGTH = 1`). Fires every player turn — exhausts
/// the top of the draw pile (one card; Python's
/// `exhaust_top_of_draw_pile` pops `draw[0]`, matching Rust's
/// `exhaust_from_draw_pile(1)`) and applies +1 Strength to the player.
/// Bead `sts2-rs-70anj.1`.
fn listener_toasty_mittens(state: &mut CombatState) -> EffectResult {
    if !state.relics.owns(RelicKind::ToastyMittens) {
        return EffectResult::NotImplemented;
    }
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    // C# `ToastyMittens.BeforeHandDraw` (`ToastyMittens.cs`): every player
    // turn, BEFORE the hand is drawn — `ShuffleIfNecessary`, then exhaust the
    // first NON-Innate draw card on round 1 (else the top card), then gain +1
    // Strength. This MUST fire on `before_hand_draw`, not `after_side_turn_start`
    // (CombatManager.cs:539 fires `Hook.BeforeHandDraw` before the :560
    // `CardPileCmd.Draw`). The old `after_side_turn_start` registration fired
    // AFTER the opening draw, so the post-shuffle top card had already moved to
    // hand → the relic exhausted the wrong card and the intended top card stayed
    // in hand (act-2 IC:1004: C# exhausts BLUDGEON pre-draw; Rust kept it in the
    // opening hand and exhausted the next card down). ToastyMittens is an Ancient
    // relic gained entering act 2, so the desync first surfaced at the first
    // act-2 combat. Bead `sts2-rs-70anj.1`.
    let skip_innate = state.round_no() == 1;
    // `await CardPileCmd.ShuffleIfNecessary` (ToastyMittens.cs:35): a
    // Stratagem pick its `AfterShuffle` opens resolves before the exhaust and
    // the Strength, so park both on that pick. The `before_hand_draw` walk
    // suspends behind it and resumes at the next relic.
    let had_choice = state.pending_choice.is_some();
    state.shuffle_before_draw_exhaust();
    if !had_choice {
        if let Some(pc) = state.pending_choice.as_mut() {
            if pc.kind == sts2_core::ChoiceKind::AddFromDrawToHand && pc.post_effect.is_none() {
                pc.post_effect = Some(sts2_core::PostChoiceEffect::ExhaustDrawCardGainStrength {
                    skip_innate,
                    strength: TOASTY_MITTENS_STRENGTH,
                });
                return EffectResult::Resolved;
            }
        }
    }
    let draw_was_suspended = state.pending_draw_resume.is_some();
    state.exhaust_first_draw_card(skip_innate);
    // `await CardCmd.Exhaust` (ToastyMittens.cs:48) includes a Dark Embrace
    // draw that reshuffled into a Stratagem pick: the Strength waits for it.
    if !draw_was_suspended && state.pending_choice.is_some() {
        if let Some(pdr) = state.pending_draw_resume.as_mut() {
            pdr.then_gain_strength = pdr
                .then_gain_strength
                .saturating_add(i16::try_from(TOASTY_MITTENS_STRENGTH).unwrap_or(i16::MAX));
            return EffectResult::Resolved;
        }
    }
    if let Some(player_uid) = state.player.creature_uid {
        apply_power_to_creature(
            state,
            player_uid,
            PowerKind::Strength,
            TOASTY_MITTENS_STRENGTH,
        );
    }
    EffectResult::Resolved
}

/// v0.111.0 `ToastyMittens.AfterPlayerTurnStart` (decomp
/// ToastyMittens.cs:31-43), every player turn, after the hand is drawn:
///
/// [decompiled C# omitted — see the cited path]
///
/// A forced pick of exactly 1 hand card (`FromHand` auto-takes a 1-card
/// hand), then +1 Strength. The Strength rides the choice as its
/// `PostChoiceEffect`, so it lands after the exhaust, like Brand's; with an
/// empty hand there is nothing to pick and it applies at once. The open
/// screen suspends the `AfterPlayerTurnStart` walk like GamblingChip's.
fn listener_toasty_mittens_v0111(state: &mut CombatState) -> EffectResult {
    if state.game_version < sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::ToastyMittens) {
        return EffectResult::NotImplemented;
    }
    // Never clobber a screen an earlier surface left open (GamblingChip's
    // guard); C# would queue this select behind it.
    if state.pending_choice.is_some() {
        return EffectResult::NotImplemented;
    }
    if state.player.hand.is_empty() {
        if let Some(player_uid) = state.player.creature_uid {
            apply_power_to_creature(
                state,
                player_uid,
                PowerKind::Strength,
                TOASTY_MITTENS_STRENGTH,
            );
        }
        return EffectResult::Resolved;
    }
    let options: Vec<sts2_core::ChoiceOption> = state
        .player
        .hand
        .iter()
        .filter_map(|&uid| {
            let card = sts2_core::card_by_uid(&state.cards, uid)?;
            Some(sts2_core::ChoiceOption {
                card_id: card.card_id,
                upgraded: card.upgraded,
                card_uid: Some(uid),
                source_pile: Some(sts2_core::cards::PileKind::Hand),
            })
        })
        .collect();
    state.open_hand_choice(sts2_core::PendingChoice {
        kind: sts2_core::ChoiceKind::ExhaustFromHand,
        options,
        min_select: 1,
        max_select: 1,
        can_skip: false,
        post_effect: Some(sts2_core::PostChoiceEffect::GainStrength(
            TOASTY_MITTENS_STRENGTH,
        )),
        picked: smallvec::SmallVec::new(),
    });
    EffectResult::Resolved
}

/// `JeweledMask::before_hand_draw` — C# v0.107.1 `JeweledMask.cs`
/// `BeforeHandDraw`: `if (TurnNumber <= 1) { list = Draw.Cards.Where(c =>
/// c.Type == Power); if non-empty: card = Rng.CombatCardSelection
/// .NextItem(list); card.SetToFreeThisTurn(); CardPileCmd.Add(card, Hand) }`.
/// Rust port uses `LocalCostMod::FREE_THIS_TURN` as the "free for this
/// turn" mechanism (same pattern as `InfernalBlade` in
/// `cards/ironclad.rs`). Bead `sts2-rs-70anj.1`.
///
/// Timing (2026-07-17 fix, same shape as the ToastyMittens/FuneraryMask
/// corrections above): the C# hook is `BeforeHandDraw`, which fires
/// BEFORE the opening 5-card draw (`CombatManager.cs:539` vs the `:560`
/// `CardPileCmd.Draw`). The prior port registered on
/// `after_side_turn_start` (post-draw), so a Power that happened to be
/// drawn naturally was excluded from the selection pool — and when the
/// deck's ONLY Power got drawn, the relic silently did nothing.
///
/// RNG (same fix): C# reads `Rng.CombatCardSelection.NextItem(list)` =
/// exactly one `NextInt(0, count)` tick on the CombatCardSelection
/// stream (`Rng.cs:256/264`); the old port read the generic
/// `state.rng`, desyncing the `combat_card_selection` counter.
fn listener_jeweled_mask(state: &mut CombatState) -> EffectResult {
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::JeweledMask) {
        return EffectResult::NotImplemented;
    }
    // Pool = Power cards in the (pre-draw) draw pile, in pile order —
    // mirrors the C# `Draw.Cards.Where(...)` enumeration order.
    let pool: smallvec::SmallVec<[sts2_core::CardUid; 8]> = state
        .player
        .draw
        .iter()
        .copied()
        .filter(|uid| {
            let idx = uid.0.get() as usize - 1;
            state
                .cards
                .get(idx)
                .is_some_and(|c| c.card_type == CardType::Power)
        })
        .collect();
    if pool.is_empty() {
        // C# guards `list.Count != 0` BEFORE the Rng read — an empty
        // pool must not consume a CombatCardSelection tick.
        return EffectResult::NotImplemented;
    }
    // v0.111.0: prefer the non-Innate Powers, falling back to all of them
    // when every Power is Innate (decomp JeweledMask.cs:27-31).
    let pool = if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        let non_innate: smallvec::SmallVec<[sts2_core::CardUid; 8]> = pool
            .iter()
            .copied()
            .filter(|uid| {
                let idx = uid.0.get() as usize - 1;
                state
                    .cards
                    .get(idx)
                    .is_some_and(|c| !sts2_core::card_effects::card_innate_inst(c))
            })
            .collect();
        if non_innate.is_empty() {
            pool
        } else {
            non_innate
        }
    } else {
        pool
    };
    let idx = state
        .run_rng
        .combat_card_selection
        .next_int_exclusive(0, pool.len() as i32) as usize;
    let uid = pool[idx];
    if state.game_version >= sts2_core::GameVersion::V0_113_0 {
        // v0.113.0 `JeweledMask.cs:34`: `SetToFreeThisCombat`.
        if let Some(card) = state.cards.get_mut(uid.0.get() as usize - 1) {
            card.set_to_free_this_combat();
        }
    } else {
        sts2_core::card_effects::set_card_free_this_turn(state, uid);
    }
    state.move_card_to_hand(uid);
    EffectResult::Resolved
}

/// `BurningSticks::after_card_exhausted` — `if not used_this_combat and
/// card.card_type == SKILL: used_this_combat = True;
/// combat.clone_card_to_hand(owner, card)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::BurningSticks`).
/// Per-combat latch is stored on `Relic.counter` (`0` = available,
/// `1` = already used this combat — Python resets via
/// `before_combat_start`; the Rust counter is zeroed at the same
/// point by the combat-state reset path). Clone is built by cloning
/// the exhausted card's POD from the arena and routing it through
/// `spawn_card_into_hand`. Bead `sts2-rs-70anj.1`.
fn listener_burning_sticks(state: &mut CombatState, ctx: &CardEventCtx) -> EffectResult {
    let Some(card_uid) = ctx.card_uid else {
        return EffectResult::NotImplemented;
    };
    let idx = card_uid.0.get() as usize - 1;
    let Some(card) = state.cards.get(idx) else {
        return EffectResult::NotImplemented;
    };
    if card.card_type != CardType::Skill {
        return EffectResult::NotImplemented;
    }
    let mut consumed = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::BurningSticks && relic.counter == 0 {
            relic.counter = 1;
            consumed = true;
            break;
        }
    }
    if !consumed {
        return EffectResult::NotImplemented;
    }
    // Re-read the exhausted card POD (Relic mutation above may have
    // invalidated any prior borrow) and clone it into hand.
    // C# `card.CreateClone()` (BurningSticks.cs).
    let cloned = state.cards[idx].create_clone();
    // C# `AddGeneratedCardToCombat(card2, Hand, base.Owner)` (BurningSticks.cs:50):
    // player-created even when the exhaust happens outside a card play (Play
    // pile empty), so pass `added_by_player = true` explicitly.
    let copy = spawn_card_into_hand_with_origin(state, cloned, true);
    // The clone keeps Master Planner's Sly and the single-turn Retain / Sly
    // (loadouts Silent 5229 / 5235 v0.111.0).
    state.inherit_clone_flags(card_uid, copy);
    EffectResult::Resolved
}

/// Re-arm BurningSticks for each combat: C# clears `WasUsedThisCombat` on
/// entering a combat room and at combat end (`BurningSticks.cs:33-42`,
/// `:56-61`). The latch lives on `Relic.counter`, which the run carries
/// between combats (`run_bridge::enter_combat_from_roster` seeds it from
/// `rs.relic_counters`), so without this reset the relic fired in its first
/// combat only (act23 Ironclad:1007).
fn listener_burning_sticks_reset(state: &mut CombatState) -> EffectResult {
    let mut any = false;
    for relic in &mut state.relics {
        if relic.kind == RelicKind::BurningSticks {
            relic.counter = 0;
            any = true;
        }
    }
    if any {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `MysticLighter::modify_damage_additive` — `if dealer is owner and
/// the played card instance is enchanted and props is powered (MOVE and
/// not UNPOWERED): return EXTRA_DAMAGE = 9` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::MysticLighter`).
/// "Enchanted" means `!card.enchantments.is_empty()` on the Rust
/// `Card` POD (mirrors Python `CardInstance.enchantments: dict[str,
/// int]` non-empty). Bead `sts2-rs-70anj.1`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_mystic_lighter(state: &CombatState, ctx: &ModifyDamageCtx) -> i32 {
    if !state.relics.owns(RelicKind::MysticLighter) {
        return 0;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return 0;
    };
    // C# gates only on `cardSource.Owner != base.Owner` (MysticLighter.cs:
    // 22-29), never on the dealer, so the player's enchanted card attacking
    // through the Osty (`FromOsty`) still gets +9 — a Vigorous Rattle hit for
    // 7 + 8 + 9 = 24 in C# (loadout Necrobinder 1048). A card source on a
    // player- or Osty-dealt hit is the player's card (MiniatureCannon's gate).
    if ctx.dealer != Some(player_uid) && ctx.dealer.is_none_or(|d| state.player.pet_uid != Some(d))
    {
        return 0;
    }
    // C# `MysticLighter.ModifyDamageAdditive` (v0.111.0
    // `MysticLighter.cs:16-31`) gates on
    // `props.IsPoweredAttack()` (MOVE && !UNPOWERED,
    // ValuePropExtensions.cs:5) and `cardSource?.Enchantment != null`,
    // with no card-type check.
    if !ctx.props.is_powered() {
        return 0;
    }
    // `cardSource` is the SPECIFIC played instance, so key on the instance
    // uid, not `ctx.card`'s id. The old `find(card_id == …)` read the FIRST
    // arena copy with that id: with a Pael's Growth Clone on master_deck[0]
    // (the LAST arena copy, since enter_combat feeds
    // `master_deck.iter().rev()`) the enchanted Strike lost its +9 (act23
    // parity Ironclad 1019: C# 22 vs Rust 9); in the reverse order every
    // copy got it. Same shape as MiniatureCannon bead `sts2-rs-nwz1n`.
    let Some(card_uid) = ctx.card_uid else {
        return 0;
    };
    let is_enchanted = state
        .cards
        .get(card_uid.0.get() as usize - 1)
        .is_some_and(|c| !c.enchantments.is_empty());
    if is_enchanted {
        MYSTIC_LIGHTER_EXTRA_DAMAGE
    } else {
        0
    }
}

/// `SneckoEye::before_combat_start` — `owner.apply_power(CONFUSED, 1)`
/// (Python `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::
/// SneckoEye`). Applies a single stack of `PowerKind::Confused` to
/// the player at combat start. Bead `sts2-rs-70anj.1`.
fn listener_snecko_eye_combat_start(state: &mut CombatState) -> EffectResult {
    if !state.relics.owns(RelicKind::SneckoEye) {
        return EffectResult::NotImplemented;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    apply_power_to_creature(state, player_uid, PowerKind::Confused, 1);
    EffectResult::Resolved
}

/// `GhostSeed::before_combat_start` — stamp
/// [`VarKey::EtherealThisCombat`] = 1 on every basic Strike / Defend
/// in any of the player's piles (`draw`, `hand`, `discard`,
/// `exhaust`). Mirrors Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::GhostSeed.
/// before_combat_start` — Python iterates `state.all_piles` and gates
/// on `CardRarity.BASIC` + `"STRIKE" in card_id.name or "DEFEND" in
/// card_id.name` + `not card.is_ethereal`. Rust port matches on the
/// concrete basic Strike / Defend [`CardId`] variants (one Strike +
/// one Defend per character), which is the closed set of cards
/// satisfying Python's rarity-and-name filter. The companion
/// `after_card_entered_combat` arm (mid-combat additions) is out of
/// scope per the bead body — file a follow-up if the residual
/// becomes load-bearing for parity. The existing end-of-turn
/// ethereal sweep in [`crate::sts2_core::combat_loop`] honors the
/// flag (bead `sts2-rs-ad53n`). Bead `sts2-rs-sei89`.
fn listener_ghost_seed_combat_start(state: &mut CombatState) -> EffectResult {
    if !state.relics.owns(RelicKind::GhostSeed) {
        return EffectResult::NotImplemented;
    }
    // C# `GhostSeed.cs:51`: `Rarity == Basic && (Strike || Defend)` tag.
    // Derived Strike-tag cards (`SetupStrikeCard`, ...) are non-basic and
    // must NOT match, which the closed id lists preserve.
    use super::neow_relics::{BASIC_DEFEND_IDS, BASIC_STRIKE_IDS};
    let mut uids: Vec<sts2_core::CardUid> = Vec::new();
    uids.extend(state.player.draw.iter().copied());
    uids.extend(state.player.hand.iter().copied());
    uids.extend(state.player.discard.iter().copied());
    uids.extend(state.player.exhaust.iter().copied());
    for uid in uids {
        let card_id = match state.cards.get(uid.0.get() as usize - 1) {
            Some(c) => c.card_id,
            None => continue,
        };
        if BASIC_STRIKE_IDS.contains(&card_id) || BASIC_DEFEND_IDS.contains(&card_id) {
            set_card_var(state, uid, VarKey::EtherealThisCombat, 1);
        }
    }
    EffectResult::Resolved
}

/// `SneckoEye::modify_hand_draw` — `return draw + EXTRA_DRAW = 2`
/// (Python `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::
/// SneckoEye`). No round gate — fires every turn while owned. Bead
/// `sts2-rs-70anj.1`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_snecko_eye_hand_draw(state: &CombatState, current_draw: i32) -> i32 {
    if !state.relics.owns(RelicKind::SneckoEye) {
        return current_draw;
    }
    current_draw + SNECKO_EYE_EXTRA_DRAW
}

/// `ThrowingAxe::modify_card_play_count` — first card each combat is
/// played twice. Mirrors Python `sts2-rl-agent@01c094d:
/// sts2_env/relics/shop_event.py::ThrowingAxe`:
///
/// ```py
/// def modify_card_play_count(self, owner, count, card):
///     if not self._used_this_combat:
///         self._used_this_combat = True
///         return count + 1
///     return count
/// ```
///
/// Per-combat latch lives on the relic's `counter` field
/// (`0` = unused, `1` = fired) — same 3nis pattern used by
/// `Permafrost`/`EmberTea`. We only fire on the player's own play
/// (`ctx.owner == CreatureRef::Player`). Returns the *delta* (Rust
/// convention) not the new total. Bead `sts2-rs-70anj.3.3`.
fn listener_throwing_axe_play_count(state: &mut CombatState, ctx: &CardPlayCountCtx) -> i32 {
    if ctx.owner != sts2_core::hooks::CreatureRef::Player {
        return 0;
    }
    let Some(relic) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == RelicKind::ThrowingAxe)
    else {
        return 0;
    };
    if relic.counter != 0 {
        return 0;
    }
    relic.counter = 1;
    1
}

/// `ThrowingAxe::before_combat_start` — clear the once-per-combat latch.
/// Mirrors Python's `self._used_this_combat = False` reset (Python hooks
/// both `before_combat_start` and `after_combat_end`; the Rust port hooks
/// the symmetric pre-combat channel so the reset is observable on the
/// very first dispatch of the next combat). Bead `sts2-rs-70anj.3.3`.
fn listener_throwing_axe_reset(state: &mut CombatState) -> EffectResult {
    let Some(relic) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == RelicKind::ThrowingAxe)
    else {
        return EffectResult::NotImplemented;
    };
    relic.counter = 0;
    EffectResult::Resolved
}

/// `DelicateFrond::before_combat_start` — fills every open potion slot
/// with random *out-of-combat* potions rolled off the run-level
/// `CombatPotionGeneration` stream. Mirrors C# `DelicateFrond.BeforeCombatStart`:
/// `while (Owner.HasOpenPotionSlots) TryToProcure(
///   CreateRandomPotionOutOfCombat(Owner, RunState.Rng.CombatPotionGeneration))`
/// (`DelicateFrond.cs:12-22`) — the exact loop `EntropicBrew.OnUse` runs, so
/// it reuses [`crate::potions::all::entropic_brew_fill_belt`] (NextFloat
/// rarity + NextItem = 2 draws per potion, out-of-combat pool, first-free
/// slot). GH #1054: this used to call the core
/// `CombatState::fill_empty_potion_slots`, which drew one uniform pick off
/// the combat-local `rng` over the in-combat pool — wrong stream, wrong
/// algorithm, narrower pool (Defect seed 1015). Bead `sts2-rs-70anj.3.12`.
fn listener_delicate_frond_combat_start(state: &mut CombatState) -> EffectResult {
    // Ownership gate: the `before_combat_start` channel fires every
    // registered listener unconditionally, so without this guard the belt
    // would be filled in *every* combat (even for players who don't carry
    // DelicateFrond) — surfacing phantom drinkable potions. Mirrors the
    // implicit ownership scoping every other per-relic listener applies.
    if !state.relics.owns(RelicKind::DelicateFrond) {
        return EffectResult::NotImplemented;
    }
    crate::potions::all::entropic_brew_fill_belt(state);
    EffectResult::Resolved
}

/// `ChemicalX::modify_x_value` — `return x + INCREASE = 2`
/// (Python `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::
/// ChemicalX`). Adds +2 to every X-cost play's effective `x` value
/// while owned. Bead `sts2-rs-70anj.3.7`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_chemical_x_modify_x_value(state: &CombatState, current_x: i32) -> i32 {
    if !state.relics.owns(RelicKind::ChemicalX) {
        return current_x;
    }
    current_x + CHEMICAL_X_INCREASE
}

/// `BrilliantScarf::before_side_turn_start` — `if side == PLAYER:
/// self._cards_this_turn = 0` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::
/// BrilliantScarf`). Resets the per-turn play counter on every
/// player turn start. Bead `sts2-rs-70anj.1`.
fn listener_brilliant_scarf_turn_reset(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::BrilliantScarf {
            relic.counter = 0;
            return EffectResult::Resolved;
        }
    }
    EffectResult::NotImplemented
}

/// C# `BrilliantScarf.AfterCardPlayed` (BrilliantScarf.cs:89-105, both
/// builds): `CardsPlayedThisTurn++` for each play iteration that is not an
/// auto-play (`cardPlay.IsAutoPlay` returns). It is a relic listener in the
/// player's relic walk, so a relic equipped after it (Mummified Hand) already
/// sees the count: with the fourth play a Power, every hand card is globally
/// free when Mummified Hand rolls (Defect random loadout v0.111.0 seed 5416).
/// Hand-play replays (One-Two Punch, Echo Form) are not auto-plays and count.
fn listener_brilliant_scarf_card_played(
    state: &mut CombatState,
    _ctx: &sts2_core::relic_effects::CardEventCtx,
) -> EffectResult {
    if state.combat_log.after_played_is_auto_play {
        return EffectResult::NotImplemented;
    }
    let Some(relic) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == RelicKind::BrilliantScarf)
    else {
        return EffectResult::NotImplemented;
    };
    relic.counter = relic.counter.saturating_add(1);
    EffectResult::Resolved
}

/// `BlessedAntler::before_hand_draw` — on the turn-1 hand draw, add
/// `BLESSED_ANTLER_DAZED_COUNT = 3` Dazed to the owner's draw pile at RANDOM
/// depths. C# `BlessedAntler.BeforeHandDraw` (`BlessedAntler.cs:46-58`), gated
/// on `Owner.PlayerCombatState.TurnNumber == 1`:
/// [decompiled C# omitted — see the cited path]
/// `AddGeneratedCardsToCombat` loops the singular add per card
/// (`CardPileCmd.cs:1043`), each resolving `CardPilePosition.Random` to
/// `AddInternal(card, Rng.Shuffle.NextInt(Cards.Count + 1))`
/// (`CardPileCmd.cs:511-515`) — so three fresh random depths against a pile
/// that grows by one each iteration, and three Shuffle-stream reads. The prior
/// port followed the Python pin's top-insert
/// (`combat.add_card_to_draw_pile`), which diverged on DRAW_ORDER and consumed
/// ZERO Shuffle draws, desyncing every later draw/reshuffle in the combat.
/// Same shape and same fix as `FuneraryMask` / `TeaOfDiscourtesy` (bead
/// `sts2-rs-n4sa4`).
///
/// Channel: registered on `before_hand_draw`, matching C#'s `BeforeHandDraw`
/// — the Dazed must be in the pile before the opening draw so one can land in
/// the turn-1 hand (the same correction `FuneraryMask` took). The `+1`
/// max-energy `ModifyMaxEnergy` leg is still deferred — no `modify_max_energy`
/// register channel exists at this pin. Beads `sts2-rs-70anj.1`,
/// `sts2-rs-lfdaa`.
fn listener_blessed_antler(state: &mut CombatState) -> EffectResult {
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::BlessedAntler) {
        return EffectResult::NotImplemented;
    }
    #[allow(clippy::cast_sign_loss)]
    for _ in 0..(BLESSED_ANTLER_DAZED_COUNT as usize) {
        let dazed =
            crate::cards::status::card_from_spec(&crate::cards::status_specs_generated::dazed());
        let _ = sts2_core::card_effects::spawn_card_into_draw_pile_random_position(state, dazed);
    }
    EffectResult::Resolved
}

/// Re-arm PaelsEye for each combat: C# `AfterCombatEnd` clears
/// `UsedThisCombat` (`PaelsEye.cs:142-147`). Both flags live on
/// `Relic.counter`, which the run carries between combats, so without this
/// reset the extra turn fired in one combat per run only.
fn listener_paels_eye_reset(state: &mut CombatState) -> EffectResult {
    let mut any = false;
    for relic in &mut state.relics {
        if relic.kind == RelicKind::PaelsEye {
            relic.counter = 0;
            any = true;
        }
    }
    if any {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `PaelsEye::before_side_turn_start` — `if side == PLAYER:
/// self._any_cards_played = False` (Python pin same class). Clears
/// bit 1 of `Relic.counter` on every player side-turn-start. Bead
/// `sts2-rs-vfzgs`.
fn listener_paels_eye_side_turn_start(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::PaelsEye {
            relic.counter &= !0b10;
            return EffectResult::Resolved;
        }
    }
    EffectResult::NotImplemented
}

/// `PaelsEye::before_turn_end` — `if not _used_this_combat and not
/// _any_cards_played and side == PLAYER: exhaust entire hand` (Python
/// pin same class). The extra-turn re-entry + `_used_this_combat`
/// latch live in the engine surface (`combat_loop::end_player_turn`
/// short-circuit); the relic listener only handles the exhaust-hand
/// body. Gated on bit 0 (used) == 0 AND C# `AnyCardsPlayedThisTurn` false
/// ([`sts2_core::hooks::paels_eye_any_cards_played_this_turn`]: bit 1, plus
/// the turn-1 Whispering Earring rule) so the exhaust matches the would-be
/// `should_take_extra_turn` outcome.
/// Bead `sts2-rs-vfzgs`.
fn listener_paels_eye_before_turn_end(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    let should_exhaust = state.relics.iter().any(|r| {
        !r.is_melted
            && r.kind == RelicKind::PaelsEye
            && (r.counter & 0b01) == 0
            && !sts2_core::hooks::paels_eye_any_cards_played_this_turn(state, r)
    });
    if !should_exhaust {
        return EffectResult::NotImplemented;
    }
    state.exhaust_hand();
    EffectResult::Resolved
}

/// `MusicBox::before_card_played` — C# `MusicBox.BeforeCardPlayed`
/// (`MusicBox.cs`): while nothing is
/// latched and the relic is unused this turn (bit 0 of `Relic.counter`),
/// the first Attack to start playing becomes `CardBeingPlayed`
/// (`combat_log.music_box_card_being_played`). GH #1110.
fn listener_music_box_before_card_played(
    state: &mut CombatState,
    ctx: &CardEventCtx,
) -> EffectResult {
    let Some(uid) = ctx.card_uid else {
        return EffectResult::NotImplemented;
    };
    if state.combat_log.music_box_card_being_played.is_some() {
        return EffectResult::NotImplemented;
    }
    let is_attack =
        sts2_core::card_by_uid(&state.cards, uid).is_some_and(|c| c.card_type == CardType::Attack);
    if !is_attack {
        return EffectResult::NotImplemented;
    }
    let armed = state
        .relics
        .iter()
        .any(|r| !r.is_melted && r.kind == RelicKind::MusicBox && (r.counter & 0b01) == 0);
    if !armed {
        return EffectResult::NotImplemented;
    }
    state.combat_log.music_box_card_being_played = Some(uid);
    EffectResult::Resolved
}

/// `MusicBox::after_card_played` — C# `MusicBox.AfterCardPlayed`: only the
/// latched card (see [`listener_music_box_before_card_played`]) is cloned,
/// as an Ethereal copy into hand; then the relic is used for the turn
/// (bit 0 of `Relic.counter`) and the latch clears. An Attack auto-played
/// inside the latched one finishes first but is not cloned. The clone is
/// built by copying the played card's POD, setting
/// `VarKey::EtherealThisCombat = 1` on the new instance (honored by the
/// end-of-turn ethereal sweep), and routing through `spawn_card_into_hand`.
fn listener_music_box_card_played(state: &mut CombatState, ctx: &CardEventCtx) -> EffectResult {
    let Some(played_uid) = ctx.card_uid else {
        return EffectResult::NotImplemented;
    };
    if state.combat_log.music_box_card_being_played != Some(played_uid) {
        return EffectResult::NotImplemented;
    }
    let idx = played_uid.0.get() as usize - 1;
    if idx >= state.cards.len() {
        return EffectResult::NotImplemented;
    }
    let Some(relic) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == RelicKind::MusicBox)
    else {
        return EffectResult::NotImplemented;
    };
    relic.counter |= 0b01;
    state.combat_log.music_box_card_being_played = None;
    // C# `cardPlay.Card.CreateClone()` (MusicBox.cs:77): a new instance, so
    // no inherited play-history stamp (Ironclad-6161-a0-act2#b0).
    let mut cloned = state.cards[idx].create_clone();
    if state.game_version >= sts2_core::GameVersion::V0_113_0 {
        // v0.113.0: Swift / Sown disable after Music Box (relics precede card
        // enchantments in `AfterCardPlayed`), so the clone keeps them Normal.
        sts2_core::enchantment_hooks::clear_pending_first_play_disable(&mut cloned);
    }
    // C# `AddGeneratedCardToCombat(card, Hand, base.Owner)` (MusicBox.cs:80):
    // player-created regardless of Play-pile occupancy.
    let new_uid = spawn_card_into_hand_with_origin(state, cloned, true);
    // `CreateClone` keeps the played card's Retain / Sly flags.
    state.inherit_clone_flags(state.cards[idx].uid, new_uid);
    set_card_var(state, new_uid, VarKey::EtherealThisCombat, 1);
    EffectResult::Resolved
}

/// `MusicBox::before_side_turn_start` — `if side == PLAYER:
/// self._used_this_turn = False` (Python pin same class). Clears bit 0
/// of `Relic.counter` so the next player turn re-arms the
/// first-Attack-clone trigger. Bead `sts2-rs-ad53n`.
fn listener_music_box_side_turn_start(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::MusicBox {
            relic.counter &= !0b01;
            state.combat_log.music_box_card_being_played = None;
            return EffectResult::Resolved;
        }
    }
    EffectResult::NotImplemented
}

/// `Crossbow.AfterSideTurnStart` (`Crossbow.cs:17-37`): on the owner's
/// side, one random Attack from the character pool, free this turn,
/// generated into hand. The engine side,
/// `card_effects::generate_free_attack_in_hand`, samples the card, sets
/// `LocalCostMod::FREE_THIS_TURN`, and adds it the way C#
/// `AddGeneratedCardsToCombat` does: to the discard pile when the hand
/// is full, with the generation listeners fired.
fn listener_crossbow(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::Crossbow) {
        return EffectResult::NotImplemented;
    }
    generate_free_attack_in_hand(state);
    EffectResult::Resolved
}

const SEAL_OF_GOLD_COST: i32 = 5;
const SEAL_OF_GOLD_ENERGY: i32 = 1;

/// `SealOfGold`'s `GoldVar` for the build: 5 in v0.107.1.
///
/// v0.111.0: `GoldVar(5)` -> `GoldVar(3)` (decomp SealOfGold.cs:20).
const fn seal_of_gold_cost(version: sts2_core::GameVersion) -> i32 {
    match version {
        sts2_core::GameVersion::V0_107_1 => SEAL_OF_GOLD_COST,
        sts2_core::GameVersion::V0_111_0 | sts2_core::GameVersion::V0_113_0 => 3,
    }
}

/// `SealOfGold::after_side_turn_start` — `if side == PLAYER and
/// owner.gold >= GOLD_COST = 5: owner.lose_gold(5);
/// combat.gain_energy(owner, ENERGY = 1)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::SealOfGold`).
/// Reads gold via `card_effects::read_player_gold` (closure registry
/// installed by the run-loop driver at combat start) and spends via
/// `try_spend_player_gold`. Returns `NotImplemented` when:
/// - No gold accessor is installed (core-only test build).
/// - `gold < 5` (insufficient funds).
/// - The atomic spend fails (the run-loop's gold mutator returns
///   false — e.g. another listener already drained gold mid-walk).
/// Bead `sts2-rs-1e0ae`.
fn listener_seal_of_gold(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::SealOfGold) {
        return EffectResult::NotImplemented;
    }
    // C# `SealOfGold.AfterSideTurnStart`: while gold ≥ 5, spend 5 for +1 energy.
    // Reads/spends the combat-time gold mirror (`enter_combat` snapshots run
    // gold in, `exit_combat` writes it back). The old `read_player_gold` /
    // `try_spend_player_gold` global-accessor path was never installed by any
    // driver, so this relic silently no-opped (IC:1002 @377). Bead
    // `sts2-rs-RP17`.
    let cost = seal_of_gold_cost(state.game_version);
    if state.gold < cost {
        return EffectResult::NotImplemented;
    }
    state.gold -= cost;
    state.gain_energy(SEAL_OF_GOLD_ENERGY);
    EffectResult::Resolved
}

/// `Bread::after_side_turn_start` — on the FIRST player turn only,
/// `PlayerCmd.LoseEnergy(LoseEnergy = 2)` (C# v0.107.1
/// `MegaCrit.Sts2.Core.Models.Relics/Bread.cs:AfterSideTurnStart`:
/// `if participants.Contains(Owner.Creature) && TurnNumber == 1:
/// LoseEnergy(2)`). C# `PlayerCombatState.LoseEnergy` clamps at 0
/// (`PlayerCombatState.cs:178-185`), so the subtraction saturates.
/// The complementary `ModifyMaxEnergy` leg (+1 from turn 2 onward,
/// nothing on turn 1) is the inline `RelicKind::Bread` arm in
/// `sts2_core::hooks::modify_max_energy`. 2026-07-17 soft-drop
/// census fix.
fn listener_bread(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::Bread) {
        return EffectResult::NotImplemented;
    }
    state.player.energy = (state.player.energy - BREAD_LOSE_ENERGY).max(0);
    EffectResult::Resolved
}

/// `LeesWaffle::after_obtained` — `owner.gain_max_hp(7)` + `owner.heal(max_hp)`
/// (Python `shop_event.py::LeesWaffle`). Bead `sts2-rs-bmw3`.
fn listener_lees_waffle(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LeesWaffle) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(LEES_WAFFLE_MAX_HP);
    rs.heal(rs.max_hp);
    EffectResult::Resolved
}

/// `LoomingFruit::after_obtained` — `owner.gain_max_hp(31)` (Python
/// `shop_event.py::LoomingFruit`, `MAX_HP=31`). Bead `sts2-rs-r4pr`.
fn listener_looming_fruit(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LoomingFruit) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(LOOMING_FRUIT_MAX_HP);
    EffectResult::Resolved
}

/// `NutritiousOyster::after_obtained` — `owner.gain_max_hp(11)` (Python
/// `shop_event.py::NutritiousOyster`, `MAX_HP=11`). Bead `sts2-rs-r4pr`.
fn listener_nutritious_oyster(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::NutritiousOyster) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(NUTRITIOUS_OYSTER_MAX_HP);
    EffectResult::Resolved
}

/// `GoldenPearl::after_obtained` — `owner.gain_gold(150)` (Python
/// `shop_event.py::GoldenPearl`, `GOLD=150`). Bead `sts2-rs-yfi0`.
fn listener_golden_pearl(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::GoldenPearl) {
        return EffectResult::NotImplemented;
    }
    rs.gain_gold(GOLDEN_PEARL_GOLD);
    EffectResult::Resolved
}

/// `SignetRing::after_obtained` — `owner.gain_gold(999)` (Python
/// `shop_event.py::SignetRing`, `GOLD=999`). Bead `sts2-rs-2ciy`.
fn listener_signet_ring(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::SignetRing) {
        return EffectResult::NotImplemented;
    }
    // v0.111.0: `GoldVar(999)` -> `GoldVar(888)` (decomp SignetRing.cs:13).
    let gold = match rs.game_version {
        sts2_core::GameVersion::V0_107_1 => SIGNET_RING_GOLD,
        sts2_core::GameVersion::V0_111_0 | sts2_core::GameVersion::V0_113_0 => 888,
    };
    rs.gain_gold(gold);
    EffectResult::Resolved
}

/// `PumpkinCandle::AfterObtained → Rekindle` (v0.107.1) — add
/// `kindleAmount = 5` to the relic's `KindleCount`, stored in
/// `rs.relic_counters[PumpkinCandle]`. Mirrors C#
/// `MegaCrit.Sts2.Core.Models.Relics/PumpkinCandle.cs:52-56`
/// (`AfterObtained → Rekindle`) + `:87-91` (`Rekindle → KindleCount += 5`).
///
/// While `KindleCount > 0` the combat-side `modify_max_energy` arm in
/// `core/hooks.rs` grants +1 max energy; the count decays by 1 per
/// combat end (`exit_combat` bridge) and the rest-site `Kindle` option
/// tops it back up by another 5. The v0.107.1 redesign replaced the
/// old v0.103.2 act-snapshot capture this listener used to perform.
///
/// `rs.relic_counters[PumpkinCandle]` → combat-side `Relic.counter`
/// thread-through lives on the `enter_combat(rs, encounter)` bridge
/// (`sts2-rs-tjn0v`). Bead `sts2-rs-run-rest-option`.
fn listener_pumpkin_candle(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PumpkinCandle) {
        return EffectResult::NotImplemented;
    }
    *rs.relic_counters.entry(RelicId::PumpkinCandle).or_insert(0) +=
        sts2_core::runstate::PUMPKIN_CANDLE_KINDLE_AMOUNT;
    EffectResult::Resolved
}

/// `PumpkinCandle::TryModifyRestSiteOptions` — append
/// [`RestSiteOption::Kindle`] to the rest-site option list while the
/// relic is owned. Mirrors C#
/// `MegaCrit.Sts2.Core.Models.Relics/PumpkinCandle.cs:77-85`
/// (`options.Add(new KindleRestSiteOption(player))`, owner-gated) and
/// the Shovel `modify_rest_site_options` precedent
/// (`rare.rs::listener_shovel_rest_site_options`). Appended after the
/// base `[Heal, Smith]` list, matching C# `RestSiteOption.Generate`'s
/// `Hook.ModifyRestSiteOptions` walk. Bead `sts2-rs-run-rest-option`.
fn listener_pumpkin_candle_rest_site_options(
    rs: &RunState,
    options: &mut Vec<RestSiteOption>,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::PumpkinCandle) {
        return EffectResult::NotImplemented;
    }
    if options.iter().any(|opt| opt.option_id() == "KINDLE") {
        return EffectResult::NotImplemented;
    }
    options.push(RestSiteOption::Kindle);
    EffectResult::Resolved
}

/// `PrismaticGem::ModifyCardRewardCreationOptions` — a card reward rolls from
/// every character's pool: `player.UnlockState.CharacterCardPools
/// .Union(options.CardPools)` (`PrismaticGem.cs:32-52`; v0.111.0 the same
/// minus the `CustomCardPool` clause, which that build no longer has). The
/// union keeps `ModelDb.AllCharacters` order (Ironclad, Silent, Regent,
/// Necrobinder, Defect — `ModelDb.cs:123-130`) ahead of the reward's own
/// pools, and that order is the `NextItem` index space. Skipped for
/// `NoCardPoolModifications`, a custom pool, or an all-colorless pool. Every
/// caller of this channel rolls a `CardReward`, so the `IsCardReward` gate
/// always passes. The relic was registered nowhere, so its rewards rolled from
/// the owner's pool only (act23 probe: Defect / Necrobinder seed 1002 offered
/// different cards for the same Rewards reads).
fn listener_prismatic_gem(
    rs: &mut RunState,
    opts: &mut CardRewardGenerationOptions,
) -> EffectResult {
    use sts2_core::potions::CharacterId;
    /// `ModelDb.AllCharacters` (all unlocked in the headless game).
    const ALL_CHARACTERS: [CharacterId; 5] = [
        CharacterId::Ironclad,
        CharacterId::Silent,
        CharacterId::Regent,
        CharacterId::Necrobinder,
        CharacterId::Defect,
    ];
    if !rs.owns_active_relic(RelicId::PrismaticGem) {
        return EffectResult::NotImplemented;
    }
    if !opts.allow_card_pool_modifications || opts.has_custom_card_pool {
        return EffectResult::NotImplemented;
    }
    let mut own: Vec<CharacterId> = if !opts.character_ids.is_empty() {
        opts.character_ids.clone()
    } else if opts.use_default_character_pool {
        vec![rs.character_id]
    } else {
        Vec::new()
    };
    if own.is_empty() {
        // Only the colorless pool (or nothing) — C# leaves it alone.
        return EffectResult::NotImplemented;
    }
    let mut pools: Vec<CharacterId> = ALL_CHARACTERS.to_vec();
    // v0.113.0: The Fool joins `ModelDb.AllCharacters`, last.
    if rs.game_version >= sts2_core::GameVersion::V0_113_0 {
        pools.push(CharacterId::Fool);
    }
    own.retain(|c| !pools.contains(c));
    pools.extend(own);
    opts.character_ids = pools;
    opts.use_default_character_pool = false;
    EffectResult::Resolved
}

/// `DingyRug::modify_card_reward_creation_options` — a card reward also
/// rolls from the colorless pool (C# `DingyRug.ModifyCardRewardCreationOptions`).
/// Skipped for `NoCardPoolModifications`, a custom pool, or a reward that
/// already includes colorless. Bead `sts2-rs-u5emw`.
fn listener_dingy_rug(rs: &mut RunState, opts: &mut CardRewardGenerationOptions) -> EffectResult {
    if !rs.owns_active_relic(RelicId::DingyRug) {
        return EffectResult::NotImplemented;
    }
    if !opts.allow_card_pool_modifications || opts.has_custom_card_pool || opts.include_colorless {
        return EffectResult::NotImplemented;
    }
    // C# `DingyRug.ModifyCardRewardCreationOptions` (`DingyRug.cs:13-36`;
    // v0.111.0 the same as a `Union`) appends the colorless pool to the
    // reward's own pools, keeping its filter — `include_colorless`, which
    // every roller already orders character-then-colorless. The old port
    // built an explicit custom pool: it kept MultiplayerOnly cards (the
    // custom path skips `FilterForPlayerCount`, shifting the `NextItem`
    // index), and it cleared `character_ids`, so Lasting Candy found no
    // Powers to add (act23 probe: Silent 1018, a 3-card reward where C#
    // offered 4 with a different third card).
    opts.include_colorless = true;
    EffectResult::Resolved
}

/// `MassiveScroll::after_obtained` — push `MASSIVE_SCROLL_CARDS = 3`
/// `Reward::Card(CardReward::empty())` sentinels (Python
/// `shop_event.py::MassiveScroll`: `owner.offer_multiplayer_cards(3)`).
/// Coop/multiplayer is out of scope per CLAUDE.md, so this collapses
/// to regular card-reward offers — same convention as Orrery (psf9).
/// Bead `sts2-rs-64vb`.
fn listener_massive_scroll(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::MassiveScroll) {
        return EffectResult::NotImplemented;
    }
    for _ in 0..MASSIVE_SCROLL_CARDS {
        rs.pending_rewards.push(Reward::Card(CardReward::empty()));
    }
    EffectResult::Resolved
}

/// `Cauldron::after_obtained` — push `CAULDRON_POTIONS = 5`
/// `Reward::Potion(PotionReward::empty())` sentinels to
/// `rs.pending_rewards` (Python `shop_event.py::Cauldron`:
/// `owner.offer_potions(5)`). The empty PotionReward is the
/// random-potion placeholder pending dispatcher-side roll at flush
/// time (ib04 convention). Same shape as Orrery's CardReward push.
/// Bead `sts2-rs-2nrk`.
fn listener_cauldron(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Cauldron) {
        return EffectResult::NotImplemented;
    }
    for _ in 0..CAULDRON_POTIONS {
        rs.pending_rewards
            .push(Reward::Potion(PotionReward::empty()));
    }
    EffectResult::Resolved
}

/// `SmallCapsule::after_obtained` — push `SMALL_CAPSULE_RELICS = 1`
/// RelicReward sentinel via `rs.offer_relic_rewards` (Python
/// `shop_event.py::SmallCapsule.after_obtained`:
/// `owner.offer_relic_rewards(1)`). Same shape as CallingBell (u4wv)
/// but 1 reward instead of 3. Bead `sts2-rs-gf5s`.
fn listener_small_capsule(rs: &mut RunState) -> EffectResult {
    // One-shot on own pickup: gate on `last_obtained_relic`, not on
    // `relics.contains` — the dispatcher walks every relic acquisition,
    // so an `is-owned` check would re-fire on every later pickup and
    // flood the relic-reward queue.
    if rs.last_obtained_relic != Some(RelicId::SmallCapsule) {
        return EffectResult::NotImplemented;
    }
    // Python `offer_relic_rewards(1)` — rolls one random rarity per slot.
    rs.offer_random_relic_rewards(SMALL_CAPSULE_RELICS, &[]);
    EffectResult::Resolved
}

/// `CallingBell::after_obtained` — push `CALLING_BELL_RELICS = 3`
/// RelicReward sentinels via `rs.offer_relic_rewards` (Python
/// `shop_event.py::CallingBell.after_obtained`: `owner.offer_relic_rewards(
/// 3, rarities=(COMMON, UNCOMMON, RARE))`). The RelicId acts as the
/// sentinel placeholder; umbrella sts2-rs-01o resolves rarity tiers
/// at flush time. The CurseOfTheBell card-add clause runs in both builds
/// (GH #806). Bead `sts2-rs-u4wv`.
fn listener_calling_bell(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::CallingBell) {
        return EffectResult::NotImplemented;
    }
    // The curse goes into the deck before the relics are offered
    // (`CardPileCmd.AddCurseToDeck<CurseOfTheBell>`, decomp
    // `Models.Relics/CallingBell.cs:29`, identical in v0.107.1 and v0.111.0,
    // so both builds: GH #806).
    if let Some(curse) =
        crate::cards::spec_index::card_pod_from_id(sts2_core::cards::CardId::CurseOfTheBell, false)
    {
        rs.add_card_to_deck_with_hooks(curse, sts2_core::relic_effects::AddSource::RelicGain);
    }
    // Python pin: `offer_relic_rewards(3, rarities=(COMMON, UNCOMMON, RARE))`.
    rs.offer_random_relic_rewards(
        CALLING_BELL_RELICS,
        &[
            Some(RelicRarity::Common),
            Some(RelicRarity::Uncommon),
            Some(RelicRarity::Rare),
        ],
    );
    EffectResult::Resolved
}

/// `LeafyPoultice::after_obtained` — C# `LeafyPoultice.cs`:
/// [decompiled C# omitted — see the cited path]
/// Loses `LEAFY_POULTICE_MAX_HP_LOSS = 12` max HP (C# `MaxHpVar(12m)`),
/// then transforms the **first** Basic Strike and **first** Basic Defend
/// in the deck into two random `{Common, Uncommon, Rare}` cards from the
/// character pool, each rolled off the `transformations` RNG stream.
///
/// **RNG order** matches C# `CardCmd.Transform`, which calls
/// `GetReplacement(rng)` for each transformation in the *input list*
/// order — Strike first, Defend second — **before** the pile-index sort
/// (the sort only reorders insertion, not RNG reads). So we roll the
/// Strike replacement, then the Defend replacement, in that order. The
/// per-card pool / pick is `CardFactory.CreateRandomCardForTransform`
/// out of combat — see [`crate::cards::spec_index::roll_transform_replacement_basic`]
/// for the pool derivation. Slots are deterministic (first Strike, first
/// Defend), so they consume no RNG; the in-place swap goes through
/// [`sts2_core::runstate::RunState::transform_specific_cards`] (same
/// position-preserving primitive PandorasBox uses).
///
/// A character returning `None` from `basic_strike_defend_cards` would no-op
/// the transform leg while the HP loss still applies; every character now has
/// a pair, so that branch is unreachable. Bead `sts2-rs-beda` (transform leg under
/// umbrella `sts2-rs-01o`).
fn listener_leafy_poultice(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LeafyPoultice) {
        return EffectResult::NotImplemented;
    }
    rs.lose_max_hp(LEAFY_POULTICE_MAX_HP_LOSS);
    let character_id = rs.character_id;
    let Some([strike_id, defend_id]) =
        crate::cards::spec_index::basic_strike_defend_cards(character_id)
    else {
        return EffectResult::Resolved;
    };
    // C# collects the first Basic Strike, then the first Basic Defend,
    // and rolls each replacement in that order off `Transformations`.
    let mut indices: Vec<usize> = Vec::with_capacity(2);
    let mut replacements: Vec<sts2_core::Card> = Vec::with_capacity(2);
    for original_id in [strike_id, defend_id] {
        let Some(deck_idx) = rs
            .master_deck
            .iter()
            .position(|card| card.card_id == original_id)
        else {
            continue;
        };
        let Some(new_id) = crate::cards::spec_index::roll_transform_replacement_basic(
            rs,
            character_id,
            original_id,
        ) else {
            continue;
        };
        // Build the replacement from its spec, not a blank `pod_defaults()`
        // with only `card_id` set: a transformed card carries its full
        // stats into combat (the master-deck POD is the combat card's
        // source), so a stat-less POD plays for zero damage / block. C#
        // `CardCmd.Transform` inserts a fully-constructed `CardModel`
        // (`GetReplacement` → `CardFactory.CreateRandomCardForTransform`).
        // Transform replacements are never upgraded out of combat. GH#31:
        // this is why a Leafy-Poultice-rolled Collision Course dealt 0 in
        // Rust vs 11 in C#.
        let Some(card) = crate::cards::spec_index::card_pod_from_id(new_id, false) else {
            continue;
        };
        indices.push(deck_idx);
        replacements.push(card);
    }
    if !indices.is_empty() {
        let _ = rs.transform_specific_cards(&indices, replacements);
    }
    EffectResult::Resolved
}

/// `DistinguishedCape::after_obtained` — `owner.lose_max_hp(9)` plus
/// push 3 Apparition cards onto `rs.master_deck` (Python
/// `shop_event.py::DistinguishedCape.after_obtained`, `HP_LOSS = 9`,
/// `CARDS = 3`). HP-loss leg landed in bead `sts2-rs-r2y4x`; the
/// Apparition deck-add leg landed in bead `sts2-rs-a49q3` once the
/// direct-deck-push pattern (JewelryBox / TanxsWhistle) was confirmed
/// sufficient. Uses the C#-faithful `lose_max_hp` (only clamps current HP
/// down when the new max falls below it) per the LeafyPoultice precedent.
fn listener_distinguished_cape(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::DistinguishedCape) {
        return EffectResult::NotImplemented;
    }
    if rs.game_version >= sts2_core::GameVersion::V0_111_0 {
        // v0.111.0: the Max-HP loss moved to SereTalon; the Cape adds 2
        // random generatable curses instead, then the Apparitions (decomp
        // DistinguishedCape.cs:32-53).
        add_random_generatable_curses(rs, DISTINGUISHED_CAPE_CURSES_V0111, true);
        for _ in 0..DISTINGUISHED_CAPE_CARDS {
            push_deck_card(rs, sts2_core::cards::CardId::Apparition);
        }
        return EffectResult::Resolved;
    }
    rs.lose_max_hp(DISTINGUISHED_CAPE_HP_LOSS);
    for _ in 0..DISTINGUISHED_CAPE_CARDS {
        push_deck_card(rs, sts2_core::cards::CardId::Apparition);
    }
    EffectResult::Resolved
}

/// `SandCastle::after_obtained` — C# `SandCastle.cs:22-32`:
/// `Deck.Cards.Where(IsUpgradable).StableShuffle(Rng.Niche).Take(6)` →
/// `CardCmd.Upgrade`. This upgrades 6 RANDOM cards with no player choice
/// (the grid is a preview animation, not a chooser), so it routes through
/// [`RunState::upgrade_random_cards_stable`], NOT the interactive-aware
/// `upgrade_random_cards`: the latter, under the workbench / lockstep
/// `interactive_card_upgrade` flag, would wrongly present a "pick 6 cards"
/// selection screen. Bead `sts2-rs-r2y4x`.
fn listener_sand_castle(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::SandCastle) {
        return EffectResult::NotImplemented;
    }
    rs.upgrade_random_cards_stable(SAND_CASTLE_CARDS);
    EffectResult::Resolved
}

/// `GlassEye::after_obtained` — queue its 5 pickup card rewards (Common,
/// Common, Uncommon, Uncommon, Rare; pick 1 of 3 each), rolled NOW as C#
/// `GlassEye.AfterObtained` does (`GlassEye.cs:15-33` → `RewardsCmd.
/// OfferCustom`): uniform odds within the forced rarity and NO upgrade roll
/// ([`crate::cards::spec_index::roll_glass_eye_card_rewards`]). The old
/// `CardReward::empty()` sentinels rolled at drain time with standard reward
/// odds — wrong rarities, upgrade floats and Rewards-stream reads (act23
/// probe: Ironclad 1000 / Regent 1007 Orobas → Glass Eye). Each slate runs
/// the card-reward slate hooks (`CardReward.cs:89`), like Colorful
/// Philosophers. Bead `sts2-rs-66l9f`.
fn listener_glass_eye(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::GlassEye) {
        return EffectResult::NotImplemented;
    }
    for cards in crate::cards::spec_index::roll_glass_eye_card_rewards(rs) {
        let cards = rs.offer_card_slate(cards, true);
        rs.pending_rewards
            .push(Reward::Card(CardReward::from_cards(&cards)));
    }
    EffectResult::Resolved
}

/// `LostCoffer::after_obtained` — push one `Reward::Card` + one
/// `Reward::Potion` sentinel (Python
/// `shop_event.py::LostCoffer.after_obtained`: `owner.offer_card_reward()`
/// then `owner.offer_potion_reward()`). The Rust port uses
/// `CardReward::empty()` / `PotionReward::empty()` placeholders
/// pending umbrella `sts2-rs-01o` random rolling — same convention
/// as Orrery / MassiveScroll. Counts hard-coded to 1 in Python (no
/// class-level constant). Bead `sts2-rs-66l9f`.
fn listener_lost_coffer(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LostCoffer) {
        return EffectResult::NotImplemented;
    }
    rs.pending_rewards.push(Reward::Card(CardReward::empty()));
    rs.pending_rewards
        .push(Reward::Potion(PotionReward::empty()));
    EffectResult::Resolved
}

/// `Kaleidoscope::after_obtained` — push 2 pre-resolved `Reward::Card`
/// sentinels, one per card reward Kaleidoscope offers on pickup
/// (`Kaleidoscope.cs:33-50`, `DynamicVars.Cards == 2`). Each reward's 3
/// cross-character cards are rolled NOW (at obtain time, as C# does in
/// `AfterObtained`) via [`crate::cards::spec_index::roll_kaleidoscope_card_rewards`]
/// — Niche-shuffled non-owner pools + base-odds card creation on the Rewards
/// stream — so the offered slate is materialized verbatim (the non-empty
/// `Reward::Card` surfaces directly through `drain_pending_rewards` instead
/// of re-rolling the owner pool). The two rewards surface sequentially as
/// `card_reward` picks (C# `RewardsCmd.OfferCustom([reward1, reward2])`).
/// Fires once on Kaleidoscope's own pickup (`last_obtained_relic`), like
/// HeftyTablet / ScrollBoxes — gating on `relics.contains` would re-roll the
/// streams on every later relic obtain. Bead `sts2-rs-utgn4`.
fn listener_kaleidoscope(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Kaleidoscope) {
        return EffectResult::NotImplemented;
    }
    let owner = rs.character_id;
    let rewards = crate::cards::spec_index::roll_kaleidoscope_card_rewards(rs, owner);
    // `RewardsSet.Offer` populates each reward; a manually-set CardReward's
    // `Populate` runs `TryModifyCardRewardOptions` over its slate with
    // `IsCardReward` set (`CardReward.cs:136,148-156`), so SilkenTress /
    // SilverCrucible / the eggs reach Kaleidoscope's cards too.
    for reward in rewards {
        let cards = rs.offer_card_slate(reward, true);
        rs.pending_rewards
            .push(Reward::Card(CardReward::from_cards(&cards)));
    }
    EffectResult::Resolved
}

/// `ScrollBoxes::after_obtained` — generate 2 card bundles and queue the
/// pick-1-of-N-packs choice. v0.107.1 delta
/// (`sts2-rs-v0107-content-port-wb8lk.9` §7): the relic **no longer loses
/// all gold** on pickup (`ScrollBoxes.cs:30-40` — the prior v0.103.2
/// `PlayerCmd.LoseGold(Owner.Gold)` line was removed; the AfterObtained
/// body now only generates + offers the bundles). The Neow-pool
/// `IsAllowedAtNeow` gate (≥4 unlocked commons / ≥2 unlocked uncommons via
/// `CanGenerateBundles`) is a no-op for the shipped character pools (all
/// satisfy it) and there is no Neow-offer filtering surface in the Rust
/// run loop to hook it into, so it is not modeled.
///
/// The card-bundle mechanic (each bundle is a multi-card "all-or-nothing"
/// pick) surfaces via `rs.pending_bundles` /
/// `RunManager::surface_pending_bundle`. Bead `sts2-rs-66l9f` (initial) /
/// `…wb8lk.9` (v0.107.1: drop the gold loss).
fn listener_scroll_boxes(rs: &mut RunState) -> EffectResult {
    // Fire exactly once, on ScrollBoxes's own pickup (mirrors HeftyTablet).
    // Gating on `relics.contains` would re-fire on every subsequent relic
    // obtain — RE-ROLLING the Rewards RNG and clobbering `pending_bundles`,
    // which would both desync downstream RNG and resurface a phantom bundle
    // pick. `last_obtained_relic` fires once.
    if rs.last_obtained_relic != Some(RelicId::ScrollBoxes) {
        return EffectResult::NotImplemented;
    }
    // v0.107.1: NO gold loss (the v0.103.2 `LoseGold(Owner.Gold)` was removed).
    // Generate the 2 packs (2 common + 1 uncommon each) and queue the
    // pick-1-of-N-packs choice (`ScrollBoxes.cs:25-34`
    // `GenerateRandomBundles` → `FromChooseABundleScreen` → add the whole
    // chosen pack to the Deck). The run loop surfaces `pending_bundles` as a
    // no-skip CardReward fan-out (`RunManager::surface_pending_bundle`).
    let character = rs.character_id;
    let bundles = crate::cards::spec_index::generate_scroll_box_bundles(rs, character);
    debug_assert_eq!(
        bundles.len(),
        SCROLL_BOXES_BUNDLES,
        "ScrollBoxes makes 2 packs"
    );
    if !bundles.is_empty() {
        rs.pending_bundles = bundles;
    }
    EffectResult::Resolved
}

/// Helper: push a single fresh `CardId` POD onto `rs.master_deck`.
/// Mirrors the direct-deck-push pattern used by JewelryBox /
/// TanxsWhistle, with a [`CardSpec`] lookup so `card_type`, `cost`,
/// `base_damage`, `base_block`, and `target_type` come from the per-
/// card factory rather than the generic `Card::pod_defaults` defaults
/// (which would tag every card as a 1-cost Attack regardless of
/// `card_id`). Falls back to `pod_defaults` when the spec isn't
/// registered. Bead `sts2-rs-a49q3` (initial form) / `sts2-rs-jzu61`
/// (spec lookup added so curses correctly receive `CardType::Curse`
/// and downstream gates like RoyalStamp's `card_type in {Attack,
/// Skill}` filter work).
fn push_deck_card(rs: &mut RunState, card_id: sts2_core::cards::CardId) {
    let mut card = sts2_core::cards::Card::pod_defaults();
    card.card_id = card_id;
    if let Some(spec) =
        crate::names::find_deck_card_spec(&crate::names::deck_card_spec_pool(), card_id)
    {
        // The deck card is copied into combat as it stands, so it must carry
        // the run's game build's numbers (Whistle costs 2 and Relax blocks 16
        // on v0.111.0).
        let spec = crate::cards::v0111::spec_for(spec, rs.game_version);
        card.card_type = spec.card_type;
        card.cost.set_base(spec.cost);
        card.base_damage = spec.base_damage;
        card.base_block = spec.base_block;
        card.target_type = spec.target_type;
    }
    rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
}

/// Enchant up to `max_count` eligible cards in `rs.master_deck` with
/// `ench` at the given `amount`. Mirrors the direct-apply arm of
/// `owner.enchant_selected_cards(...)` from
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py` —
/// i.e. the path taken when `run_state.defer_followup_rewards` is
/// `False`. Eligibility uses [`crate::enchantments::can_enchant_card`]
/// against the matching [`crate::cards::CardSpec`]; cards lacking a
/// spec entry are skipped. Picks the first `max_count` eligible
/// cards in deck order (deterministic) — Python's random selection
/// is gated on the deferred-reward arm, which lands later via
/// umbrella `sts2-rs-01o` (`Reward::Enchant` variant). Returns the
/// number actually enchanted (≤ `max_count`).
/// Collect the `CardId`s of every master-deck card eligible to receive
/// `ench`. Mirrors Python's `[card for card in owner.deck if
/// can_enchant_card(card, ench)]` list comprehension used by the
/// deferred-reward arm of every shop_event enchant-on-obtain relic.
///
/// Returns one entry per deck-position, so duplicates are preserved
/// (matches Python's list semantics; the reward UI dedupes display-
/// side, not data-side). Bead `sts2-rs-2v8y2`.
fn enchantable_candidate_card_ids(
    rs: &RunState,
    ench: sts2_core::cards::EnchantmentId,
) -> Vec<sts2_core::cards::CardId> {
    let pool = crate::names::deck_card_spec_pool();
    rs.master_deck
        .iter()
        .filter_map(|card| {
            let spec = crate::names::find_deck_card_spec(&pool, card.card_id)?;
            crate::enchantments::can_enchant_card(&spec, card, ench, rs.game_version)
                .then_some(card.card_id)
        })
        .collect()
}

fn enchant_eligible_deck_cards(
    rs: &mut RunState,
    ench: sts2_core::cards::EnchantmentId,
    amount: i32,
    max_count: usize,
) -> usize {
    if max_count == 0 || rs.master_deck.is_empty() {
        return 0;
    }
    let pool = crate::names::deck_card_spec_pool();
    let version = rs.game_version;
    let mut applied = 0;
    for card in &mut rs.master_deck {
        if applied >= max_count {
            break;
        }
        let Some(spec) = crate::names::find_deck_card_spec(&pool, card.card_id) else {
            continue;
        };
        if !crate::enchantments::can_enchant_card(&spec, card, ench, version) {
            continue;
        }
        let _ = crate::enchantments::apply_static_enchantment(card, ench, amount);
        applied += 1;
    }
    applied
}

/// The `CardType` restriction(s) for an enchant-on-obtain relic's
/// interactive deck card-select, mirroring the per-enchantment
/// `card_type` arm of [`crate::enchantments::can_enchant_card`]. Returns
/// `(primary, secondary)`: the player may pick any deck card whose type
/// matches either. `(None, None)` = unrestricted (any enchantable card).
///
/// Kept in sync with `can_enchant_card` by hand (core's
/// `enchant_eligible_deck_indices` can't reach the content-layer gate);
/// covers only the enchantments the shop enchant-on-obtain relics use.
fn enchant_card_type_restriction(
    ench: sts2_core::cards::EnchantmentId,
) -> (
    Option<sts2_core::cards::CardType>,
    Option<sts2_core::cards::CardType>,
) {
    use sts2_core::cards::{CardType, EnchantmentId};
    match ench {
        // Attack-only (Sharp/Momentum/… group in `can_enchant_card`).
        EnchantmentId::Sharp | EnchantmentId::Momentum | EnchantmentId::Instinct => {
            (Some(CardType::Attack), None)
        }
        // Skill-only.
        EnchantmentId::Imbued => (Some(CardType::Skill), None),
        // Attack OR Skill — RoyalStamp's RoyallyApproved.
        EnchantmentId::RoyallyApproved => (Some(CardType::Attack), Some(CardType::Skill)),
        // Unrestricted (Adroit, Swift, …).
        _ => (None, None),
    }
}

/// Interactive arm shared by every shop enchant-on-obtain relic: when the
/// run surfaces deck card-selects (`interactive_card_transform`) AND the
/// enchant offers a real choice (more eligible candidates than picks),
/// queue `count` interactive [`PendingDeckEnchant`] picks and return
/// `true`. Mirrors C# `CardSelectCmd.FromDeckForEnchantment(count)`.
///
/// Returns `false` (caller falls through to its defer/auto arms) when not
/// interactive, or when `candidates <= count` so every eligible card is
/// enchanted regardless of the pick — matching the SelfHelpBook/Wood
/// Carvings `eligible > 1` convention generalized to `count`.
fn try_queue_interactive_enchant(
    rs: &mut RunState,
    ench: sts2_core::cards::EnchantmentId,
    amount: i32,
    count: usize,
    optional: bool,
) -> bool {
    if !rs.interactive_card_transform {
        return false;
    }
    let candidates = enchantable_candidate_card_ids(rs, ench);
    // `FromDeckForEnchantment` auto-takes only when `cards.Count <=
    // prefs.MinSelect` (`CardSelectCmd.cs:691`): for a mandatory pick that is
    // `candidates <= count`, for an optional `0..=count` select (Kifuda,
    // Gnarled Hammer) only an empty pool, so any card opens the screen and
    // the player confirms 0..=min(count, candidates) of them.
    let picks = if optional {
        if candidates.is_empty() {
            return false;
        }
        count.min(candidates.len())
    } else {
        if candidates.len() <= count {
            return false;
        }
        count
    };
    let (type_restriction, secondary_type) = enchant_card_type_restriction(ench);
    for _ in 0..picks {
        rs.pending_deck_enchants
            .push(sts2_core::PendingDeckEnchant {
                enchantment: ench,
                amount,
                type_restriction,
                secondary_type,
                optional,
            });
    }
    true
}

/// `GnarledHammer::after_obtained` — enchant up to `GNARLED_HAMMER_CARDS`
/// eligible deck cards with `Sharp(GNARLED_HAMMER_SHARP)` (Python
/// `shop_event.py::GnarledHammer.after_obtained`,
/// `enchant_selected_cards("Sharp", 3, 3, cards=candidates)`).
/// Interactive arm (`interactive_card_transform`) surfaces a deck
/// card-select (C# `CardSelectCmd.FromDeckForEnchantment`); the defer
/// reward-queue and direct-apply arms cover the other cases.
/// Bead `sts2-rs-70anj.4`.
fn listener_gnarled_hammer(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::GnarledHammer) {
        return EffectResult::NotImplemented;
    }
    // C# `GnarledHammer.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` — surface the deck
    // card-select when the run is interactive.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::Sharp,
        GNARLED_HAMMER_SHARP,
        GNARLED_HAMMER_CARDS,
        // `CardSelectorPrefs(EnchantSelectionPrompt, 0, Cards)`: up to 3
        // (`GnarledHammer.cs:30`).
        true,
    ) {
        return EffectResult::Resolved;
    }
    // Bead `sts2-rs-2v8y2`: deferred-reward arm. Mirrors Python
    // `relics/shop_event.py:410` (GnarledHammer.after_obtained):
    // when redeemed as part of a reward cascade, queue a player-pick
    // EnchantCards reward (`min_count=0` makes it skippable) instead
    // of auto-enchanting the first N eligible deck cards.
    if rs.defer_followup_rewards {
        let candidates = enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::Sharp);
        rs.offer_enchant_cards_reward(
            sts2_core::cards::EnchantmentId::Sharp,
            GNARLED_HAMMER_SHARP,
            GNARLED_HAMMER_CARDS,
            candidates,
            Some(0),
        );
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(
        rs,
        sts2_core::cards::EnchantmentId::Sharp,
        GNARLED_HAMMER_SHARP,
        GNARLED_HAMMER_CARDS,
    );
    EffectResult::Resolved
}

/// `Kifuda::after_obtained` — enchant up to `KIFUDA_CARDS` eligible
/// deck cards with `Adroit(KIFUDA_ADROIT)` (Python
/// `shop_event.py::Kifuda.after_obtained`,
/// `enchant_selected_cards("Adroit", 3, 3, cards=candidates)`).
/// Interactive arm (`interactive_card_transform`) surfaces a deck
/// card-select (C# `CardSelectCmd.FromDeckForEnchantment`); the defer
/// and direct-apply arms cover the reward-queue / auto cases.
/// Bead `sts2-rs-70anj.4`.
fn listener_kifuda(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Kifuda) {
        return EffectResult::NotImplemented;
    }
    // C# `Kifuda.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` — surface the deck
    // card-select when the run is interactive.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::Adroit,
        KIFUDA_ADROIT,
        KIFUDA_CARDS,
        // `CardSelectorPrefs(EnchantSelectionPrompt, 0, Cards)`: up to 3
        // (`Kifuda.cs:26`).
        true,
    ) {
        return EffectResult::Resolved;
    }
    // Bead `sts2-rs-g5ksv`: Python `relics/shop_event.py:428`
    // (Kifuda.after_obtained) — `min_count=0` (skippable).
    if rs.defer_followup_rewards {
        let candidates =
            enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::Adroit);
        rs.offer_enchant_cards_reward(
            sts2_core::cards::EnchantmentId::Adroit,
            KIFUDA_ADROIT,
            KIFUDA_CARDS,
            candidates,
            Some(0),
        );
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(
        rs,
        sts2_core::cards::EnchantmentId::Adroit,
        KIFUDA_ADROIT,
        KIFUDA_CARDS,
    );
    EffectResult::Resolved
}

/// `PunchDagger::after_obtained` — enchant 1 eligible deck card with
/// `Momentum(PUNCH_DAGGER_MOMENTUM)` (Python
/// `shop_event.py::PunchDagger.after_obtained`,
/// `enchant_selected_cards("Momentum", 5, 1, cards=candidates)`).
/// Count hard-coded to 1 in Python (no separate const).
/// Interactive arm (`interactive_card_transform`) surfaces a deck
/// card-select (C# `CardSelectCmd.FromDeckForEnchantment`); the defer
/// and direct-apply arms cover the reward-queue / auto cases.
/// Bead `sts2-rs-70anj.4`.
fn listener_punch_dagger(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PunchDagger) {
        return EffectResult::NotImplemented;
    }
    // C# `PunchDagger.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` — surface the deck
    // card-select when the run is interactive.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::Momentum,
        PUNCH_DAGGER_MOMENTUM,
        1,
        false,
    ) {
        return EffectResult::Resolved;
    }
    // Bead `sts2-rs-g5ksv`: Python `relics/shop_event.py:587`
    // (PunchDagger.after_obtained) — no `min_count` (exact-count
    // semantics per `offer_enchant_cards_reward` default).
    if rs.defer_followup_rewards {
        let candidates =
            enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::Momentum);
        rs.offer_enchant_cards_reward(
            sts2_core::cards::EnchantmentId::Momentum,
            PUNCH_DAGGER_MOMENTUM,
            1,
            candidates,
            None,
        );
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(
        rs,
        sts2_core::cards::EnchantmentId::Momentum,
        PUNCH_DAGGER_MOMENTUM,
        1,
    );
    EffectResult::Resolved
}

/// `BeautifulBracelet::after_obtained` — enchant up to
/// `BEAUTIFUL_BRACELET_CARDS` eligible deck cards with
/// `Swift(BEAUTIFUL_BRACELET_SWIFT)` (Python
/// `shop_event.py::BeautifulBracelet.after_obtained`,
/// `enchant_selected_cards("Swift", 3, 3, cards=candidates)`).
/// Interactive arm (`interactive_card_transform`) surfaces a deck
/// card-select (C# `CardSelectCmd.FromDeckForEnchantment`); the defer
/// and direct-apply arms cover the reward-queue / auto cases.
/// Bead `sts2-rs-70anj.4`.
fn listener_beautiful_bracelet(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::BeautifulBracelet) {
        return EffectResult::NotImplemented;
    }
    if rs.game_version >= sts2_core::GameVersion::V0_111_0 {
        beautiful_bracelet_v0111(rs);
        return EffectResult::Resolved;
    }
    // C# `BeautifulBracelet.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` — surface the deck
    // card-select when the run is interactive.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::Swift,
        BEAUTIFUL_BRACELET_SWIFT,
        BEAUTIFUL_BRACELET_CARDS,
        false,
    ) {
        return EffectResult::Resolved;
    }
    // Bead `sts2-rs-g5ksv`: Python `relics/shop_event.py:1030`
    // (BeautifulBracelet.after_obtained) — no `min_count` (exact).
    if rs.defer_followup_rewards {
        let candidates = enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::Swift);
        rs.offer_enchant_cards_reward(
            sts2_core::cards::EnchantmentId::Swift,
            BEAUTIFUL_BRACELET_SWIFT,
            BEAUTIFUL_BRACELET_CARDS,
            candidates,
            None,
        );
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(
        rs,
        sts2_core::cards::EnchantmentId::Swift,
        BEAUTIFUL_BRACELET_SWIFT,
        BEAUTIFUL_BRACELET_CARDS,
    );
    EffectResult::Resolved
}

/// v0.111.0 `BeautifulBracelet.AfterObtained` (decomp
/// BeautifulBracelet.cs:31-44): no player choice any more. Four random
/// enchantable deck cards get Swift 2:
///
/// [decompiled C# omitted — see the cited path]
///
/// `TakeRandom` is `ToList().UnstableShuffle(rng).Take(count)`
/// (IEnumerableExtensions.cs:17), i.e. a Fisher-Yates over the WHOLE
/// eligible list on `Rng.Niche`, then the first four.
fn beautiful_bracelet_v0111(rs: &mut RunState) {
    let pool = crate::names::deck_card_spec_pool();
    let ench = sts2_core::cards::EnchantmentId::Swift;
    let mut eligible: Vec<usize> = rs
        .master_deck
        .iter()
        .enumerate()
        .filter_map(|(i, card)| {
            let spec = crate::names::find_deck_card_spec(&pool, card.card_id)?;
            crate::enchantments::can_enchant_card(&spec, card, ench, rs.game_version).then_some(i)
        })
        .collect();
    let version = rs.game_version;
    rs.rng
        .non_combat_or_niche()
        .shuffle_for(version, &mut eligible);
    for &i in eligible.iter().take(BEAUTIFUL_BRACELET_CARDS_V0111) {
        let _ = crate::enchantments::apply_static_enchantment(
            &mut rs.master_deck[i],
            ench,
            BEAUTIFUL_BRACELET_SWIFT_V0111,
        );
    }
}

/// `RoyalStamp::after_obtained` — enchant 1 eligible deck card
/// (Attack or Skill, per `can_enchant_card` gate) with
/// `RoyallyApproved(1)` (Python `shop_event.py::RoyalStamp.after_obtained`,
/// `enchant_selected_cards("RoyallyApproved", 1, 1, cards=candidates)`).
/// Count hard-coded to 1 in Python (no separate const). Adds INNATE +
/// RETAIN to the card's effective keywords via
/// [`crate::enchantments::effective_keywords`]. Interactive arm
/// (`interactive_card_transform`) surfaces a deck card-select (C#
/// `CardSelectCmd.FromDeckForEnchantment`); defer + direct-apply
/// arms cover the reward-queue / auto cases. Bead `sts2-rs-70anj.4`.
fn listener_royal_stamp(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::RoyalStamp) {
        return EffectResult::NotImplemented;
    }
    // C# hands the grid `list.UnstableShuffle(Rng.Niche)` of the enchantable
    // cards (RoyalStamp.cs AfterObtained); `FromDeckForEnchantment` re-sorts
    // them by deck index (CardSelectCmd.cs:576-577), so only the n - 1 Niche
    // reads survive — and they skew every later Niche roll (monster HP) when
    // skipped (act23 Necrobinder 1003/1018: Niche 18-19 reads behind).
    let n =
        enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::RoyallyApproved).len();
    let mut shuffled: Vec<usize> = (0..n).collect();
    let version = rs.game_version;
    rs.rng
        .non_combat_or_niche()
        .shuffle_for(version, &mut shuffled);
    // C# `RoyalStamp.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` (pick 1 of the enchantable
    // cards) — surface that deck card-select when the run is interactive.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::RoyallyApproved,
        1,
        1,
        false,
    ) {
        return EffectResult::Resolved;
    }
    // Bead `sts2-rs-g5ksv`: Python `relics/shop_event.py:619`
    // (RoyalStamp.after_obtained) — no `min_count` (exact).
    if rs.defer_followup_rewards {
        let candidates =
            enchantable_candidate_card_ids(rs, sts2_core::cards::EnchantmentId::RoyallyApproved);
        rs.offer_enchant_cards_reward(
            sts2_core::cards::EnchantmentId::RoyallyApproved,
            1,
            1,
            candidates,
            None,
        );
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(rs, sts2_core::cards::EnchantmentId::RoyallyApproved, 1, 1);
    EffectResult::Resolved
}

/// `ElectricShrymp::after_obtained` — enchant 1 eligible Skill card
/// in `rs.master_deck` with `Imbued(1)` (Python
/// `shop_event.py::ElectricShrymp.after_obtained`,
/// `enchant_selected_cards("Imbued", 1, 1, cards=candidates)`).
/// Imbued is Skill-only per
/// [`crate::enchantments::can_enchant_card`]. Count + amount both
/// hard-coded to 1 in the Python call (no class const).
/// Interactive arm (`interactive_card_transform`) surfaces a deck
/// card-select (C# `CardSelectCmd.FromDeckForEnchantment`); the defer
/// and direct-apply arms cover the reward-queue / auto cases.
/// Bead `sts2-rs-70anj.4`.
fn listener_electric_shrymp(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::ElectricShrymp) {
        return EffectResult::NotImplemented;
    }
    // C# `ElectricShrymp.AfterObtained` awaits an interactive
    // `CardSelectCmd.FromDeckForEnchantment` — surface the deck
    // card-select when the run is interactive.
    if try_queue_interactive_enchant(rs, sts2_core::cards::EnchantmentId::Imbued, 1, 1, false) {
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(rs, sts2_core::cards::EnchantmentId::Imbued, 1, 1);
    EffectResult::Resolved
}

/// `RadiantPearl::before_side_turn_start` — on the round-1 player
/// side-turn start, generate 1 Luminesce card into the player's hand
/// (Python `shop_event.py::RadiantPearl.before_side_turn_start`:
/// `combat.create_card_in_hand(owner, "Luminesce")`). Luminesce is
/// `CardType::Skill`, cost 0, target Self_, EXHAUST+RETAIN — pinned
/// from `status_specs_generated::luminesce`. Bead `sts2-rs-70anj.4`.
fn listener_radiant_pearl(state: &mut CombatState) -> EffectResult {
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::RadiantPearl) {
        return EffectResult::NotImplemented;
    }
    let mut card = sts2_core::cards::Card::pod_defaults();
    card.card_id = sts2_core::cards::CardId::Luminesce;
    card.card_type = sts2_core::cards::CardType::Skill;
    card.cost = sts2_core::CardEnergyCost::new(0);
    card.base_damage = 0;
    card.base_block = 0;
    card.target_type = sts2_core::cards::TargetType::Self_;
    // C# `AddGeneratedCardsToCombat(list, Hand, base.Owner)` (RadiantPearl.cs:41)
    // at turn start, Play pile empty: the play-pile heuristic would read
    // `false` and skip Arsenal / PillarOfCreation / Smokestack.
    let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(state, card, true);
    EffectResult::Resolved
}

/// `ToyBox::AfterObtained` — offer `TOY_BOX_RELICS = 4` relic rewards, each
/// stamped WAX. C# `MegaCrit.Sts2.Core.Models.Relics/ToyBox.cs:87-97`:
///
/// [decompiled C# omitted — see the cited path]
///
/// The wax flag rides the OFFER, not the pool: a skipped reward never
/// becomes wax. `RunState::accept_relic_reward` records the taken ones into
/// `rs.wax_relics`, and `run_bridge::exit_combat` melts one every
/// [`TOY_BOX_COMBATS`] combats. Bead `sts2-rs-70anj.4`.
///
/// Pool nuance: C# `PullNextRelicFromFront` takes the next relic off the
/// front of the shared bag; the Rust port rolls a rarity then pops from the
/// matching grab bag (the same surface every other multi-relic offer uses).
/// That is a pre-existing offer-composition difference, not a melt one.
fn listener_toy_box(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::ToyBox) {
        return EffectResult::NotImplemented;
    }
    rs.offer_random_relic_rewards_wax(toy_box_relics(rs.game_version), &[], true);
    EffectResult::Resolved
}

/// `PaelsClaw::after_obtained` — enchant every deck card that passes
/// `Goopy.CanEnchant` with `Goopy(1)` (C# `Models.Relics/PaelsClaw.cs:33-44`:
/// `if (ModelDb.Enchantment<Goopy>().CanEnchant(item)) CardCmd.Enchant<Goopy>`).
/// GH #1054: the earlier port mirrored Python's blind `enchant_all_cards`.
fn listener_paels_claw(rs: &mut RunState) -> EffectResult {
    use sts2_core::cards::EnchantmentId;
    if rs.last_obtained_relic != Some(RelicId::PaelsClaw) {
        return EffectResult::NotImplemented;
    }
    for card in &mut rs.master_deck {
        if goopy_can_enchant(card) {
            card.add_enchantment(EnchantmentId::Goopy, 1);
        }
    }
    EffectResult::Resolved
}

/// C# `Goopy.CanEnchant` (`Goopy.cs:16-23`) for a master-deck card:
/// `base.CanEnchant && Tags.Contains(CardTag.Defend)`. The base gate
/// (`EnchantmentModel.cs:273-294`) rejects Status/Curse/Quest, Unplayable
/// deck cards, and any card already enchanted (e.g. a Sapphire Seed `Sown`
/// Defend): Goopy is not `IsStackable` (`EnchantmentModel.cs:173`), so even
/// an existing Goopy blocks a second one.
fn goopy_can_enchant(card: &sts2_core::cards::Card) -> bool {
    use sts2_core::cards::{CardTag, CardType};
    if matches!(card.card_type, CardType::Status | CardType::Curse) {
        return false;
    }
    if sts2_core::card_effects::card_unplayable_inst(card) {
        return false;
    }
    if !card.enchantments.is_empty() {
        return false;
    }
    card.has_tag(CardTag::Defend)
        || crate::cards::status::tags_for_card_id(card.card_id).contains(CardTag::Defend)
}

/// `VelvetChoker::after_card_played` — bump `relic.counter` once per
/// card played (Python `shop_event.py::VelvetChoker.after_card_played`:
/// `self._cards_this_turn += 1`). The per-turn cap is enforced by
/// `hooks::relic_should_play_blocked` (reads the counter and rejects play
/// when >= `VELVET_CHOKER_MAX_CARDS`). Bead `sts2-rs-70anj.4`.
fn listener_velvet_choker_card_played(
    state: &mut CombatState,
    _ctx: &CardEventCtx,
) -> EffectResult {
    let mut bumped = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::VelvetChoker {
            relic.counter = relic.counter.saturating_add(1);
            bumped = true;
            break;
        }
    }
    if bumped {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `DiamondDiadem::before_turn_end` — at PLAYER turn end, if the player
/// played <=2 cards this turn, apply +1 DiamondDiadem power to the
/// player. Mirrors Python `shop_event.py::DiamondDiadem.before_turn_end`
/// (sts2-rl-agent@01c094d, line ~1165): `if side == PLAYER and
/// self._cards_this_turn <= CARD_THRESHOLD: owner.apply_power(
/// DIAMOND_DIADEM, 1)`. We use the engine's global
/// `count_player_cards_played_this_turn(None)` instead of a per-relic
/// counter; both have identical semantics under the
/// before_turn_end fire schedule (counter is reset on PLAYER
/// before_side_turn_start before the next-turn count begins).
/// Bead `sts2-rs-70anj.3.21`.
fn listener_diamond_diadem_before_turn_end(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::DiamondDiadem) {
        return EffectResult::NotImplemented;
    }
    const CARD_THRESHOLD: u32 = 2;
    if state.count_player_cards_played_this_turn(None) > CARD_THRESHOLD {
        return EffectResult::NotImplemented;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    sts2_core::card_effects::apply_power_to_creature(
        state,
        player_uid,
        sts2_core::PowerKind::DiamondDiadem,
        1,
    );
    EffectResult::Resolved
}

/// v0.111.0 `DiamondDiadem.AfterSideTurnStart` (decomp
/// DiamondDiadem.cs:24-31):
///
/// [decompiled C# omitted — see the cited path]
///
/// `BlockVar(20m, ValueProp.Unpowered)` (:20): Dexterity / Frail skip it.
fn listener_diamond_diadem_v0111(state: &mut CombatState, side: Side) -> EffectResult {
    if state.game_version < sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    if side != Side::Player || state.turn_number() > 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::DiamondDiadem) {
        return EffectResult::NotImplemented;
    }
    let _ = sts2_core::card_effects::gain_block_player_unpowered(state, DIAMOND_DIADEM_BLOCK_V0111);
    if let Some(player_uid) = state.player.creature_uid {
        apply_power_to_creature(state, player_uid, PowerKind::Blur, 1);
    }
    EffectResult::Resolved
}

/// v0.111.0 `DiamondDiadem` `BlockVar(20m)` (decomp DiamondDiadem.cs:20).
const DIAMOND_DIADEM_BLOCK_V0111: i32 = 20;

/// `VelvetChoker::before_side_turn_start` — reset `relic.counter` on
/// PLAYER side (Python `shop_event.py::VelvetChoker.before_side_turn_start`:
/// `if side == PLAYER: self._cards_this_turn = 0`). Bead `sts2-rs-70anj.4`.
fn listener_velvet_choker_turn_reset(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    let mut reset = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::VelvetChoker {
            relic.counter = 0;
            reset = true;
            break;
        }
    }
    if reset {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `Ectoplasm::should_gain_gold` — veto every positive gold gain
/// while the relic is owned (Python
/// `shop_event.py::Ectoplasm.should_gain_gold` returns `False`
/// unconditionally). Sets the `cancel` out-parameter to short-circuit
/// the dispatch loop. The +1 max-energy leg is already wired via
/// `hooks::modify_max_energy`. Bead `sts2-rs-70anj.4`.
fn listener_ectoplasm_should_gain_gold(
    rs: &mut RunState,
    _amount: i32,
    cancel: &mut bool,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::Ectoplasm) {
        return EffectResult::NotImplemented;
    }
    *cancel = true;
    EffectResult::Resolved
}

/// `Sozu::should_procure_potion` — veto every potion procurement
/// while the relic is owned (Python
/// `shop_event.py::Sozu.should_procure_potion` returns `False`
/// unconditionally). Bead `sts2-rs-70anj.3.2`.
fn listener_sozu_should_procure_potion(rs: &mut RunState, cancel: &mut bool) -> EffectResult {
    if !rs.owns_active_relic(RelicId::Sozu) {
        return EffectResult::NotImplemented;
    }
    *cancel = true;
    EffectResult::Resolved
}

/// `Driftwood::allow_card_reward_reroll` — return `True` unconditionally
/// (Python `shop_event.py::Driftwood.allow_card_reward_reroll`). Setting
/// `*allow = true` short-circuits the dispatcher; the reroll UI consumer
/// reads the bool to expose the action. Bead `sts2-rs-70anj.3.5`.
fn listener_driftwood_allow_reroll(rs: &mut RunState, allow: &mut bool) -> EffectResult {
    if !rs.owns_active_relic(RelicId::Driftwood) {
        return EffectResult::NotImplemented;
    }
    *allow = true;
    EffectResult::Resolved
}

/// `DollysMirror::after_obtained` — clone one deck card
/// (`DollysMirror.cs:17-24`). C# lets the player pick from a deck select;
/// headless, `RunState::duplicate_card_from_deck` takes the first non-Quest
/// card, with no RNG draw.
fn listener_dollys_mirror(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::DollysMirror) {
        return EffectResult::NotImplemented;
    }
    // Interactive host: C# `DollysMirror.AfterObtained` (`DollysMirror.cs:16-24`)
    // awaits `CardSelectCmd.FromDeckGeneric(prefs 1, Type != Quest)`: the player
    // picks the card, nothing is rolled, and `CloneCard(card)` joins the deck.
    // Surface it as a pick-1 grant over copies of the eligible deck cards; a
    // single eligible card is taken directly (`Count <= MinSelect`).
    if rs.interactive_card_transform {
        let options: Vec<sts2_core::Card> = rs
            .master_deck
            .iter()
            .filter(|c| !c.is_quest())
            .copied()
            .collect();
        match options.len() {
            0 => {}
            1 => rs.add_card_to_deck_with_hooks(
                options[0],
                sts2_core::relic_effects::AddSource::RelicGain,
            ),
            _ => rs
                .pending_card_grants
                .push(sts2_core::runstate::PendingCardGrant {
                    options,
                    pick: 1,
                    min_pick: 1,
                }),
        }
        return EffectResult::Resolved;
    }
    let _ = rs.duplicate_card_from_deck();
    EffectResult::Resolved
}

/// `ArcaneScroll::after_obtained` — add one random Rare card to the
/// master deck. C# `ArcaneScroll.AfterObtained` builds
/// `CardCreationOptions` over `Owner.Character.CardPool` (the **current
/// character's** pool, uniform odds, `c.Rarity == Rare`) — so the roll
/// must be filtered to `rs.character_id`, not the all-character union.
/// Passing `None` previously let a Silent run roll a Defect Rare (GH#30:
/// RP6 added `SUPERCRITICAL` where C# added `MALAISE`).
/// Bead `sts2-rs-70anj.3.15`; character-pool fix for GH#30.
fn listener_arcane_scroll(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::ArcaneScroll) {
        return EffectResult::NotImplemented;
    }
    let character_id = rs.character_id;
    let Some(card_id) = crate::cards::spec_index::random_card_from_rarity(
        rs,
        Some(character_id),
        crate::cards::CardRarity::Rare,
    ) else {
        return EffectResult::NotImplemented;
    };
    // Build through the registered card factory so spec stats are
    // populated (a bare `pod_defaults()` leaves base_damage/cost at the
    // POD defaults — see the transform-leg fix in this file, GH#30).
    let card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .unwrap_or_else(|| {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = card_id;
            c
        });
    rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
    EffectResult::Resolved
}

/// `StoneHumidifier::AfterRestSiteHeal` — `owner.gain_max_hp(5)` every time
/// a rest-site REST fires the heal hook, REGARDLESS of how much (if any) HP
/// was restored. C# `HealRestSiteOption.ExecuteRestSiteHeal`
/// (`HealRestSiteOption.cs:102-109`) heals then UNCONDITIONALLY awaits
/// `Hook.AfterRestSiteHeal`, whose signature (`Hook.cs:772-776`) carries NO
/// heal amount; `StoneHumidifier.AfterRestSiteHeal` (`StoneHumidifier.cs:18-25`)
/// then grants +5 MaxHp (which itself heals +5). So a campfire rest at full
/// HP still grants the MaxHp. The old `healed <= 0` guard came from the Python
/// pin (`shop_event.py:3431` passes `healed=amount`), which drifts from the
/// amount-agnostic C# hook — the C# decomp wins. Bead `sts2-rs-RP21`
/// (was `sts2-rs-70anj.3.17`). The dispatch site `rest_site_heal`
/// (`runstate.rs:2415-2417`) fires the channel even at amount==0, so the
/// listener is reached at full HP.
fn listener_stone_humidifier(rs: &mut RunState, _healed: i32) -> EffectResult {
    if !rs.owns_active_relic(RelicId::StoneHumidifier) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(STONE_HUMIDIFIER_MAX_HP);
    EffectResult::Resolved
}

/// `MeatCleaver::modify_rest_site_options` — append a `Cook` option if
/// the player owns MeatCleaver and Cook isn't already in the list.
/// Python `shop_event.py::MeatCleaver.modify_rest_site_options` at pin
/// `01c094d:sts2_env/relics/shop_event.py:2442`.
///
/// `enabled` is `true` iff the deck has at least
/// [`COOK_OPTION_CARD_COUNT`] removable cards (matches Python's
/// `has_enough_removable=removable_count >= CARD_COUNT`). Bead
/// `sts2-rs-ekf4j`.
fn listener_meat_cleaver_rest_site_options(
    rs: &RunState,
    options: &mut Vec<RestSiteOption>,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::MeatCleaver) {
        return EffectResult::NotImplemented;
    }
    if options.iter().any(|opt| opt.option_id() == "COOK") {
        return EffectResult::NotImplemented;
    }
    let enabled = rs.removable_deck_card_count() >= COOK_OPTION_CARD_COUNT;
    options.push(RestSiteOption::Cook { enabled });
    EffectResult::Resolved
}

/// `MiniatureTent::should_disable_remaining_rest_site_options` —
/// returns `Some(false)` to override the default disable-after-pick UX
/// so the player can take multiple rest-site actions per visit.
/// Python `shop_event.py::MiniatureTent.should_disable_remaining_rest_site_options`
/// at pin `01c094d:sts2_env/relics/shop_event.py:513`. Bead
/// `sts2-rs-70anj.3.4`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_miniature_tent_disable_remaining(
    rs: &RunState,
    _chosen: RestSiteOption,
) -> Option<bool> {
    if !rs.owns_active_relic(RelicId::MiniatureTent) {
        return None;
    }
    Some(false)
}

/// `Astrolabe::after_obtained` — transform 3 random non-curse cards
/// from the master deck into random Rare cards, then upgrade them.
/// Python `shop_event.py::Astrolabe.after_obtained` at pin
/// `01c094d:sts2_env/relics/shop_event.py:1002` calls
/// `owner.transform_and_upgrade_cards(self.CARDS, cards=candidates)`
/// (CARDS = 3); without the headless flag the path delegates to
/// `_transform_and_upgrade_selected` over the candidate list head.
///
/// Self-gated on [`RunState::last_obtained_relic`] so the listener
/// fires exactly once — on Astrolabe's own pickup — rather than on
/// every subsequent relic acquisition (the `after_obtained` channel
/// dispatcher walks ALL listeners on every pickup; see
/// [`crate::run_effects::dispatch_after_obtained`] doc-comment).
///
/// Divergence from Python: the replacement pool collapses to a
/// uniform Rare-rarity sample (`character = None`) for every chosen
/// slot, because Rust has no `CardId -> CardSpec` lookup on the run
/// side yet — Python preserves the original rarity bucket. Same
/// shortcut used by `listener_arcane_scroll` / `listener_dusty_tome`.
/// Bead `sts2-rs-70anj.3.9`.
///
/// Interactive hosts ([`RunState::interactive_card_transform`]) get the C#
/// shape instead (`Astrolabe.cs:31-39`, same in v0.111.0): the player picks
/// 3 deck cards (`CardSelectCmd.FromDeckForTransformation`), and each is
/// replaced by `CardFactory.CreateRandomCardForTransform(item, false,
/// Rng.Niche)`, upgraded, then transformed in. That queues 3 upgrading
/// [`sts2_core::PendingDeckTransform`] picks and draws nothing now — the
/// legacy path below read the Rewards stream, which C# never touches here
/// (act23 probe: Defect seed 1004, Darv → Astrolabe).
fn listener_astrolabe(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Astrolabe) {
        return EffectResult::NotImplemented;
    }
    if rs.master_deck.is_empty() {
        return EffectResult::Resolved;
    }
    if rs.interactive_card_transform {
        // With 3 or fewer transformable cards the select opens no screen and
        // takes them all (`transform_from_deck`).
        let pick = sts2_core::PendingDeckTransform {
            upgrade: true,
            ..Default::default()
        };
        crate::run_callbacks::transform_from_deck(rs, &vec![pick; ASTROLABE_CARDS]);
        return EffectResult::Resolved;
    }
    // Candidate set: deck indices whose cards are transformable.
    // Python skips curses/statuses; the Rust master_deck does not
    // currently hold curses, so the filter degrades to "all indices"
    // — but keep the structure so adding a real filter later is local.
    let mut candidates: Vec<usize> = (0..rs.master_deck.len()).collect();
    let take = ASTROLABE_CARDS.min(candidates.len());
    // Partial Fisher-Yates draws `take` distinct deck indices using
    // the rewards-RNG stream (same convention as `upgrade_random_cards`).
    {
        // Bead `sts2-rs-hm2eb` (943w7 Stage 3): dotnet `RunRngSet` reward
        // stream, not the legacy Xoshiro `rewards_rng()` shim.
        let rng = &mut rs.rng.rewards;
        for i in 0..take {
            let len_remaining = (candidates.len() - i) as i32;
            let j = i + rng.next_int_exclusive(0, len_remaining) as usize;
            candidates.swap(i, j);
        }
    }
    let chosen: Vec<usize> = candidates.iter().take(take).copied().collect();
    for deck_idx in chosen {
        let Some(new_id) = crate::cards::spec_index::random_card_from_rarity(
            rs,
            None,
            crate::cards::CardRarity::Rare,
        ) else {
            continue;
        };
        let mut fresh = sts2_core::Card::pod_defaults();
        fresh.card_id = new_id;
        fresh.upgrade_for(rs.game_version);
        rs.master_deck[deck_idx] = fresh;
    }
    EffectResult::Resolved
}

/// `Toolbox::before_hand_draw` — once per combat on round 1, open a
/// 3-option colorless `AddGeneratedToHand` choice.
///
/// C# `Toolbox.BeforeHandDraw` (v0.111.0 `Toolbox.cs:22-33`) offers
/// `CardFactory.GetDistinctForCombat(Owner, ColorlessCardPool unlocked
/// cards, 3, Rng.CombatCardGeneration)`: the player-count + combat filters
/// (no MultiplayerOnly cards solo, no `CanBeGeneratedInCombat=false`), then
/// `UnstableShuffle(CombatCardGeneration).Take(3)` — exactly
/// [`crate::cards::colorless::distinct_colorless_modifier_ids`].
///
/// Per-combat latch lives on the Toolbox `Relic.counter` (0 = unused,
/// 1 = fired). Reset to 0 by [`listener_toolbox_reset`] on
/// `before_combat_start`. Bead `sts2-rs-70anj.3.13.tb`.
fn listener_toolbox_before_hand_draw(state: &mut CombatState) -> EffectResult {
    use sts2_core::{ChoiceKind, ChoiceOption};

    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    // Locate the Toolbox relic + check the once-per-combat latch.
    let Some(relic_idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::Toolbox)
    else {
        return EffectResult::NotImplemented;
    };
    if state.relics[relic_idx].counter != 0 {
        return EffectResult::NotImplemented;
    }
    let ids = crate::cards::colorless::distinct_colorless_modifier_ids(state, TOOLBOX_CARDS);
    if ids.is_empty() {
        return EffectResult::NotImplemented;
    }
    let options: Vec<ChoiceOption> = ids
        .into_iter()
        .map(|card_id| ChoiceOption {
            card_id,
            upgraded: false,
            card_uid: None,
            source_pile: None,
        })
        .collect();
    state.request_card_choice(ChoiceKind::AddGeneratedToHand, options, 1, 1, false);
    state.relics[relic_idx].counter = 1;
    EffectResult::Resolved
}

/// `Toolbox::before_combat_start` / `after_combat_end` — reset the
/// per-combat latch on `Relic.counter` so the round-1 offer fires
/// once per combat.
fn listener_toolbox_reset(state: &mut CombatState) -> EffectResult {
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::Toolbox {
            relic.counter = 0;
            return EffectResult::Resolved;
        }
    }
    EffectResult::NotImplemented
}

/// `ChoicesParadox::after_player_turn_start` — once per combat on
/// round 1, open a 5-option retain-tagged character `AddGeneratedRetainedToHand`
/// choice. Mirrors Python `shop_event.py::ChoicesParadox.after_player_turn_start`
/// at `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py:1340`.
///
/// Routed through the Rust `after_side_turn_start(Side::Player)` channel
/// — Python's `after_player_turn_start` is the player-side specialization
/// of the general after-turn-start fire. Per-combat latch on Relic.counter.
///
/// The `AddGeneratedRetainedToHand` resolver applies the persistent Retain
/// keyword to the pick, matching C# `CardCmd.ApplyKeyword(item,
/// CardKeyword.Retain)` (`ChoicesParadox.cs`). Bead `sts2-rs-70anj.3.13.cp`.
/// `AfterPlayerTurnStart` entry for [`listener_choices_paradox`].
fn listener_choices_paradox_turn_start(state: &mut CombatState) -> EffectResult {
    listener_choices_paradox(state, Side::Player)
}

fn listener_choices_paradox(state: &mut CombatState, side: Side) -> EffectResult {
    use sts2_core::cards::CardId;
    use sts2_core::{ChoiceKind, ChoiceOption};

    if side != Side::Player || state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    let Some(relic_idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::ChoicesParadox)
    else {
        return EffectResult::NotImplemented;
    };
    if state.relics[relic_idx].counter != 0 {
        return EffectResult::NotImplemented;
    }
    let Some(character) = state.player.character else {
        return EffectResult::NotImplemented;
    };
    // C# `CardFactory.GetDistinctForCombat(Owner, CardPool.GetUnlockedCards(..),
    // Cards(5), Rng.CombatCardGeneration)` (ChoicesParadox.cs AfterPlayerTurnStart):
    // an UnstableShuffle of the whole fight pool on CombatCardGeneration, take 5.
    // The Python-pin partial Fisher-Yates over the search RNG drew different
    // cards and left CombatCardGeneration ~80 reads behind (act23 Necrobinder
    // 1014).
    let mut candidates: Vec<CardId> =
        crate::cards::colorless::helpers::fight_pool(state, character)
            .iter()
            .map(|s| s.id)
            .collect();
    if candidates.is_empty() {
        return EffectResult::NotImplemented;
    }
    state
        .run_rng
        .combat_card_generation
        .shuffle_for(state.game_version, candidates.as_mut_slice());
    let take = CHOICES_PARADOX_CARDS.min(candidates.len());
    let options: Vec<ChoiceOption> = candidates
        .into_iter()
        .take(take)
        .map(|card_id| ChoiceOption {
            card_id,
            upgraded: false,
            card_uid: None,
            source_pile: None,
        })
        .collect();
    state.request_card_choice(ChoiceKind::AddGeneratedRetainedToHand, options, 1, 1, false);
    state.relics[relic_idx].counter = 1;
    EffectResult::Resolved
}

/// `ChoicesParadox::before_combat_start` — reset per-combat latch.
fn listener_choices_paradox_reset(state: &mut CombatState) -> EffectResult {
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::ChoicesParadox {
            relic.counter = 0;
            return EffectResult::Resolved;
        }
    }
    EffectResult::NotImplemented
}

/// `DustyTome::after_obtained` — add 1 random Ancient-rarity upgraded
/// card to the master deck (Python `shop_event.py::DustyTome` at line
/// 1230: "Add 1 random Ancient-rarity upgraded card to deck"). The
/// Python pool restricts to a Transcendence-cards subset
/// (Break/Suppress/Protector/MeteorShower/Quadcast); our pool helper
/// returns the Ancient-rarity intersection of the Ironclad/Silent/
/// Defect slices (3 of 5: Break, Suppress, Quadcast — Protector and
/// MeteorShower live in the Necrobinder/Regent slices not yet
/// generated). Bead `sts2-rs-70anj.3.15`.
fn listener_dusty_tome(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::DustyTome) {
        return EffectResult::NotImplemented;
    }
    // C# `DustyTome.AfterObtained` (`DustyTome.cs:58`) adds the card chosen by
    // `SetupForPlayer` at OPTION-GENERATION time — it does NOT roll again here.
    // When DustyTome was offered through the normal Darv path, the
    // `setup_boss_relic_ancient` callback already consumed the `Rewards` read
    // and stashed the card; consume it (no second read). Fall back to a fresh
    // roll only when no setup card is present (e.g. a direct grant path that
    // never ran option setup, or a unit test exercising the listener alone).
    let card_id = if let Some(id) = rs.dusty_tome_setup_card.take() {
        id
    } else {
        let Some(id) = crate::cards::spec_index::random_card_from_rarity(
            rs,
            None,
            crate::cards::CardRarity::Ancient,
        ) else {
            return EffectResult::NotImplemented;
        };
        id
    };
    // `CreateCard` + `Upgrade`, then `CardPileCmd.Add(card, PileType.Deck)`
    // (decomp `Models.Relics/DustyTome.cs:58-63`, both builds), whose
    // `AfterCardChangedPiles` pays LuckyFysh and friends (GH #806).
    let card =
        crate::cards::v0111::card_pod_for(card_id, true, rs.game_version).unwrap_or_else(|| {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = card_id;
            c.upgraded = true;
            c
        });
    rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
    EffectResult::Resolved
}

const SERE_TALON_CURSES: usize = 2;

/// `SereTalon` `DynamicVar("Wishes", 3m)` (decomp SereTalon.cs:26).
const SERE_TALON_WISHES: usize = 3;

/// v0.111.0 `SereTalon` `HpLossVar(9m)` (decomp SereTalon.cs:25).
const SERE_TALON_HP_LOSS_V0111: i32 = 9;

/// Add `count` curses to the deck the way v0.111.0 DistinguishedCape (and
/// v0.107.1 SereTalon) pick them: the `CurseCardPool` cards with
/// `CanBeGeneratedByModifiers`, `orderby c.Id`, drawn without replacement by
/// `Rng.Niche.NextItem` (decomp DistinguishedCape.cs:32-43). The pool is the
/// one NeowsBones samples (`NEOWS_BONES_CURSE_POOL`, already in id order).
fn add_random_generatable_curses(rs: &mut RunState, count: usize, non_combat_from_v0113: bool) {
    let mut available: Vec<sts2_core::cards::CardId> =
        crate::relics::neow_relics::NEOWS_BONES_CURSE_POOL.to_vec();
    for _ in 0..count {
        if available.is_empty() {
            return;
        }
        let len = i32::try_from(available.len()).unwrap_or(i32::MAX);
        // Distinguished Cape draws on `PlayerRng.NonCombatGeneration` from
        // v0.113.0 on; Sere Talon stays on `Rng.Niche`.
        let rng = if non_combat_from_v0113 {
            rs.rng.non_combat_or_niche()
        } else {
            &mut rs.rng.niche
        };
        let idx = rng.next_int_exclusive(0, len) as usize;
        let card_id = available.remove(idx);
        if let Some(card) = crate::cards::spec_index::card_pod_from_id(card_id, false) {
            rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
        }
    }
}

/// `SereTalon::after_obtained`. v0.111.0 loses 9 Max HP, then adds the 3
/// Wishes (decomp SereTalon.cs:31-43). v0.107.1 adds 2 curses, then the 3
/// Wishes (v0.107.1 `SereTalon.cs:32-55`):
///
/// [decompiled C# omitted — see the cited path]
///
/// Every card is a separate hooked deck add. The v0.107.1 arm used to draw
/// 2 curses off `Rewards` from every curse spec and skip the Wishes (GH #806).
fn listener_sere_talon(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::SereTalon) {
        return EffectResult::NotImplemented;
    }
    if rs.game_version >= sts2_core::GameVersion::V0_111_0 {
        rs.lose_max_hp(SERE_TALON_HP_LOSS_V0111);
    } else {
        add_random_generatable_curses(rs, SERE_TALON_CURSES, false);
    }
    for _ in 0..SERE_TALON_WISHES {
        if let Some(card) =
            crate::cards::spec_index::card_pod_from_id(sts2_core::cards::CardId::Wish, false)
        {
            rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
        }
    }
    EffectResult::Resolved
}

/// `PaelsGrowth::after_obtained` — enchant one eligible deck card
/// with `Clone(4)` (Python `shop_event.py::PaelsGrowth.after_obtained`:
/// `owner.enchant_selected_cards("Clone", 4, 1)`). The rest-site
/// `CloneOption` leg is deferred pending a rest-site options channel.
/// Bead `sts2-rs-70anj.4`.
fn listener_paels_growth(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PaelsGrowth) {
        return EffectResult::NotImplemented;
    }
    // `PaelsGrowth.AfterObtained` (`PaelsGrowth.cs:24`) awaits
    // `CardSelectCmd.FromDeckForEnchantment(prefs: 1)`: the player picks the
    // card. Found by the pep299 fork. One or zero candidates need no pick.
    if try_queue_interactive_enchant(rs, sts2_core::cards::EnchantmentId::Clone, 4, 1, false) {
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(rs, sts2_core::cards::EnchantmentId::Clone, 4, 1);
    EffectResult::Resolved
}

/// C# `PaelsGrowth.TryModifyRestSiteOptions` (`PaelsGrowth.cs:31-39`, same
/// in v0.107.1 and v0.111.0): append a `CloneRestSiteOption` for the owner,
/// unconditionally (no Clone-card check — the option is offered, and
/// succeeds, even when no deck card carries Clone). The pick itself is
/// `RunState::do_rest_site_action(RestSiteOption::Clone)`.
fn listener_paels_growth_rest_site_options(
    rs: &RunState,
    options: &mut Vec<RestSiteOption>,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::PaelsGrowth) {
        return EffectResult::NotImplemented;
    }
    options.push(RestSiteOption::Clone);
    EffectResult::Resolved
}

/// `WingCharm::modify_card_reward_options_late` — stamp `Swift(1)` on one
/// random Swift-eligible option. C# `WingCharm.TryModifyCardRewardOptionsLate`
/// (`WingCharm.cs:24-43`, identical in v0.111.0) filters the slate to
/// `ModelDb.Enchantment<Swift>().CanEnchant(card)` (the base
/// `EnchantmentModel.CanEnchant` rules, `EnchantmentModel.cs:273-294`) and
/// draws the pick with `RunState.Rng.Niche.NextItem`. The port drew it off
/// `Rewards` over the whole slate, adding a Rewards read to every card
/// reward (act23 probe: Regent 1009 Room Full of Cheese, Rewards 238 vs
/// C# 237). Same channel as Glitter / FresnelLens. Bead `sts2-rs-70anj.4`.
fn listener_wing_charm_card_reward(
    rs: &mut RunState,
    cards: &mut Vec<sts2_core::cards::Card>,
) -> EffectResult {
    use sts2_core::cards::EnchantmentId;
    if !rs.owns_active_relic(RelicId::WingCharm) {
        return EffectResult::NotImplemented;
    }
    let eligible: Vec<usize> = cards
        .iter()
        .enumerate()
        .filter(|(_, c)| reward_option_can_enchant(c))
        .map(|(i, _)| i)
        .collect();
    if eligible.is_empty() {
        return EffectResult::NotImplemented;
    }
    let pick = rs
        .rng
        .rewards_or_niche()
        .next_int_exclusive(0, eligible.len() as i32) as usize;
    cards[eligible[pick]].add_enchantment(EnchantmentId::Swift, WING_CHARM_SWIFT);
    EffectResult::Resolved
}

/// C# `ModelDb.Enchantment<Swift>().CanEnchant(card)` (and, identically,
/// `Glam`'s) on a reward slate card: neither Swift nor Glam adds a rule of
/// its own (`Swift.cs`, `Glam.cs`), so this is the base
/// `EnchantmentModel.CanEnchant` (`EnchantmentModel.cs:273-294`) — no
/// Status / Curse / Quest type, and, no enchantment being stackable, no card
/// that already carries one. Its Unplayable clause only bites for a card
/// already in the Deck pile, which a reward option is not.
fn reward_option_can_enchant(card: &sts2_core::cards::Card) -> bool {
    use sts2_core::cards::CardType;
    card.enchantments.is_empty()
        && !matches!(card.card_type, CardType::Status | CardType::Curse)
        && !card.is_quest()
}

/// `BiiigHug::after_obtained` — remove `BIIIG_HUG_CARDS = 4` cards
/// from `rs.master_deck` (Python `shop_event.py::BiiigHug.after_obtained`:
/// `owner.remove_cards_from_deck(4)`). Routes through
/// [`RunState::remove_cards_from_deck`], which mirrors Python's
/// head-N drain (no per-card removable filter at this pin — no Quest
/// type exists yet, so the candidate set is the deck verbatim).
/// Direct-apply arm only — the `enable_deck_choice_requests`
/// interactive arm lands later with umbrella `sts2-rs-01o`.
/// Bead `sts2-rs-70anj.4`.
fn listener_biiig_hug_remove(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::BiiigHug) {
        return EffectResult::NotImplemented;
    }
    let _ = rs.remove_cards_from_deck(BIIIG_HUG_CARDS);
    EffectResult::Resolved
}

/// `BiiigHug::after_shuffle` — every draw-pile shuffle while the
/// relic is owned adds one `Soot` card to the draw pile at a RANDOM
/// depth. C# `BiiigHug.AfterShuffle` (`BiiigHug.cs:30-40`):
/// [decompiled C# omitted — see the cited path]
/// `CardPilePosition.Random` resolves to
/// `AddInternal(card, Rng.Shuffle.NextInt(Cards.Count + 1))`
/// (`CardPileCmd.cs:511-515`), so the insert lands at a uniform depth AND
/// consumes exactly one Shuffle-stream draw. The prior top-insert
/// (`add_card_to_draw_pile`) diverged on DRAW_ORDER, consumed zero Shuffle
/// draws (desyncing every later draw/reshuffle in the combat), skipped the
/// `AfterCardGeneratedForCombat` cascade that `AddGeneratedCardToCombat`
/// fires, and hand-rolled a Soot POD that dropped the spec's tags and the
/// `GeneratedInCombat` marker. Same shape and same fix as
/// `TeaOfDiscourtesy` (bead `sts2-rs-n4sa4`). `Soot` is pinned from
/// `sts2-content::cards::status_specs_generated::soot`. Beads
/// `sts2-rs-70anj.4`, `sts2-rs-lfdaa`.
fn listener_biiig_hug_shuffle(state: &mut CombatState, _ctx: &ShuffleCtx) -> EffectResult {
    if !state.relics.owns(RelicKind::BiiigHug) {
        return EffectResult::NotImplemented;
    }
    let soot = crate::cards::status::card_from_spec(&crate::cards::status_specs_generated::soot());
    let _ = sts2_core::card_effects::spawn_card_into_draw_pile_random_position(state, soot);
    EffectResult::Resolved
}

/// `NinjaScroll::before_side_turn_start` — on the round-1 player
/// side-turn start, generate `NINJA_SCROLL_SHIVS = 3` Shivs into the
/// player's hand (Python `shop_event.py::NinjaScroll.before_side_turn_start`:
/// `combat.create_shivs_in_hand(owner, SHIVS = 3)` /
/// `add_shivs_to_hand` fallback). Each Shiv is `Attack`, cost 0,
/// damage 4, Enemy target, EXHAUST — pinned from
/// `sts2-content::cards::status_specs_generated::shiv`. Routes through
/// [`crate::card_effects::spawn_card_into_hand`] so hand-cap overflow
/// (>10 cards) sends Shivs to discard instead of crashing. Bead
/// `sts2-rs-70anj.4`.
fn listener_ninja_scroll(state: &mut CombatState) -> EffectResult {
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::NinjaScroll) {
        return EffectResult::NotImplemented;
    }
    for _ in 0..NINJA_SCROLL_SHIVS {
        // C# `Shiv.CreateInHand(base.Owner, ..)` → `AddGeneratedCardsToCombat(..,
        // creator ?? owner)` (NinjaScroll.cs:28, Shiv.cs:103) at turn start,
        // Play pile empty — player-created, so pass `true` explicitly. The
        // real Shiv spec: a hand-built POD lacked the `Shiv` tag (and the
        // Exhaust keyword), so Accuracy skipped these Shivs (loadout Silent
        // 1157 v0.107.1: Shiv cs=8 rs=4).
        let _ = crate::cards::status::spawn_shiv_with_origin(
            state,
            sts2_core::card_effects::PileKind::Hand,
            true,
        );
    }
    EffectResult::Resolved
}

/// `TriBoomerang::after_obtained` — enchant up to `TRI_BOOMERANG_CARDS`
/// eligible Attack cards with `Instinct(1)` (Python
/// `shop_event.py::TriBoomerang.after_obtained`,
/// `enchant_selected_cards("Instinct", 1, 3, cards=candidates)`).
/// Instinct (C# `Instinct.cs`) is a 2× powered-attack damage multiplier with
/// no cost behavior. Direct-apply arm
/// only — the `defer_followup_rewards` reward-queue arm lands later
/// with umbrella `sts2-rs-01o`. Bead `sts2-rs-70anj.4`.
fn listener_tri_boomerang(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::TriBoomerang) {
        return EffectResult::NotImplemented;
    }
    // `TriBoomerang.AfterObtained` (`TriBoomerang.cs:26`) awaits
    // `CardSelectCmd.FromDeckForEnchantment(prefs: Cards)`: the player picks
    // the three Attacks. Found by the pep299 fork. With three or fewer
    // candidates every one is enchanted and no screen opens.
    if try_queue_interactive_enchant(
        rs,
        sts2_core::cards::EnchantmentId::Instinct,
        1,
        TRI_BOOMERANG_CARDS,
        false,
    ) {
        return EffectResult::Resolved;
    }
    let _ = enchant_eligible_deck_cards(
        rs,
        sts2_core::cards::EnchantmentId::Instinct,
        1,
        TRI_BOOMERANG_CARDS,
    );
    EffectResult::Resolved
}

/// `Storybook::after_obtained` — push one `BrightestFlame` card onto
/// `rs.master_deck` (Python `shop_event.py::Storybook.after_obtained`:
/// `owner.add_card_to_deck("BrightestFlame")`). Count hard-coded to 1
/// in the Python call (no class const). Same shape as NeowsTorment
/// (a49q3). Direct-apply arm only — the `defer_followup_rewards`
/// reward-queue arm lands later with umbrella `sts2-rs-01o`.
/// Bead `sts2-rs-70anj.4`.
fn listener_storybook(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Storybook) {
        return EffectResult::NotImplemented;
    }
    push_deck_card(rs, sts2_core::cards::CardId::BrightestFlame);
    EffectResult::Resolved
}

/// `NeowsTorment::after_obtained` — push one `NeowsFury` card onto
/// `rs.master_deck` (Python `shop_event.py::NeowsTorment.after_obtained`:
/// `owner.add_card_to_deck("NeowsFury")`). Count hard-coded to 1 in
/// the Python call (no class const). Bead `sts2-rs-a49q3`.
fn listener_neows_torment(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::NeowsTorment) {
        return EffectResult::NotImplemented;
    }
    push_deck_card(rs, sts2_core::cards::CardId::NeowsFury);
    EffectResult::Resolved
}

/// `PaelsHorn::after_obtained` — push `PAELS_HORN_CARDS = 2` Relax
/// cards onto `rs.master_deck` (Python `shop_event.py::PaelsHorn.after_obtained`:
/// `for _ in range(2): owner.add_card_to_deck("Relax")`). Bead
/// `sts2-rs-a49q3`.
fn listener_paels_horn(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PaelsHorn) {
        return EffectResult::NotImplemented;
    }
    for _ in 0..PAELS_HORN_CARDS {
        push_deck_card(rs, sts2_core::cards::CardId::Relax);
    }
    EffectResult::Resolved
}

/// `CursedPearl::after_obtained` — push one Greed curse onto
/// `rs.master_deck` AND grant `CURSED_PEARL_GOLD = 333` gold (Python
/// `shop_event.py::CursedPearl.after_obtained`: `owner.add_card_to_deck("Greed")`
/// then `owner.gold += self.GOLD`). The Python `should_gain_gold`
/// veto-chain is omitted here — at obtain time `rs.relics` contains
/// only CursedPearl, and CursedPearl itself does not implement
/// `should_gain_gold`, so the chain unconditionally accepts. Bead
/// `sts2-rs-a49q3`.
fn listener_cursed_pearl(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::CursedPearl) {
        return EffectResult::NotImplemented;
    }
    push_deck_card(rs, sts2_core::cards::CardId::Greed);
    rs.gain_gold(CURSED_PEARL_GOLD);
    EffectResult::Resolved
}

/// `BloodSoakedRose::after_obtained` — push one Enthralled curse
/// onto `rs.master_deck` (Python `shop_event.py::BloodSoakedRose.after_obtained`:
/// `owner.add_card_to_deck("Enthralled")`). The Python class also
/// implements `modify_max_energy` (+1 max energy) — that combat-side
/// modifier is deferred pending the `modify_max_energy` hook channel
/// (same gap as Bread / Ectoplasm / VelvetChoker / PhilosophersStone
/// energy legs). Bead `sts2-rs-a49q3`.
fn listener_blood_soaked_rose(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::BloodSoakedRose) {
        return EffectResult::NotImplemented;
    }
    push_deck_card(rs, sts2_core::cards::CardId::Enthralled);
    EffectResult::Resolved
}

/// `LeadPaperweight::after_obtained` — roll `LEAD_PAPERWEIGHT_CARDS = 2`
/// concrete Colorless options off `PlayerRng.Rewards` (per card:
/// base-odds rarity → `NextItem` → upgrade float; see
/// [`crate::cards::spec_index::roll_lead_paperweight_colorless_options`])
/// and queue a skippable choose-1 screen, mirroring C#
/// `LeadPaperweight.AfterObtained` (`LeadPaperweight.cs:19-37`):
/// `CardFactory.CreateForReward(owner, 2, ColorlessCardPool/Other/
/// RegularEncounter)` → `CardSelectCmd.FromChooseACardScreen(canSkip:
/// true)` → chosen card appended to the Deck tail. The run loop surfaces
/// the queue as an interactive `card_select` (min 0 / max 1).
///
/// Gate: `last_obtained_relic` (NOT `relics.contains`) — the
/// after-obtained channel dispatches EVERY registered listener on EVERY
/// relic obtain, so a contains-gate would re-roll the Rewards stream and
/// re-queue a phantom screen on each later pickup while owned (the old
/// sentinel-pusher had exactly that bug). Bead 6ynt.4.deck-add / GH #34.
fn listener_lead_paperweight(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::LeadPaperweight) {
        return EffectResult::NotImplemented;
    }
    let options = crate::cards::spec_index::roll_lead_paperweight_colorless_options(
        rs,
        LEAD_PAPERWEIGHT_CARDS,
    );
    if !options.is_empty() {
        rs.pending_card_selects
            .push(sts2_core::runstate::PendingCardSelect::from_options(
                options,
            ));
    }
    EffectResult::Resolved
}

/// `SeaGlass::after_obtained` — C# `SeaGlass.AfterObtained`
/// (`SeaGlass.cs:74-92`, same in v0.107.1 and v0.111.0): three
/// `CardFactory.CreateForReward` calls of `15 / 3` uniform-odds cards
/// (Common, then Uncommon, then Rare) from the pool of the character the relic
/// was stamped with by Orobas ([`RunState::sea_glass_character`]), on
/// `PlayerRng.Rewards` with no rarity and no upgrade roll; then ONE
/// `FromSimpleGridForRewards` with `CardSelectorPrefs(prompt, 0, 15)`: the
/// player takes any 0..=15 of them, each added to the deck. Queued as an
/// optional [`sts2_core::runstate::PendingCardGrant`] (`min_pick == 0`).
///
/// Each `CreateForReward` runs `Hook.TryModifyCardRewardOptions` on its own
/// slate (no `NoModifyHooks`, no `IsCardReward`), like Gorge, hence
/// `offer_card_slate(.., false)` per rarity.
fn listener_sea_glass(rs: &mut RunState) -> EffectResult {
    use crate::cards::CardRarity;
    if rs.last_obtained_relic != Some(RelicId::SeaGlass) {
        return EffectResult::NotImplemented;
    }
    let character = rs.sea_glass_character();
    let mut options = Vec::with_capacity(SEA_GLASS_CARDS);
    for rarity in [CardRarity::Common, CardRarity::Uncommon, CardRarity::Rare] {
        let cards = crate::cards::spec_index::roll_uniform_pool_cards(
            rs,
            character,
            rarity,
            SEA_GLASS_CARDS / 3,
        );
        options.extend(rs.offer_card_slate(cards, false));
    }
    if !options.is_empty() {
        let pick = options.len();
        rs.pending_card_grants
            .push(sts2_core::runstate::PendingCardGrant {
                options,
                pick,
                min_pick: 0,
            });
    }
    EffectResult::Resolved
}

/// `YummyCookie::after_obtained` — C# `YummyCookie.cs`:
/// `CardSelectCmd.FromDeckForUpgrade(count 4)` → `CardCmd.Upgrade`. This is
/// a genuine PLAYER PICK (choose 4 cards to upgrade), so it correctly routes
/// through the interactive-aware [`RunState::upgrade_random_cards`], which
/// under `interactive_card_upgrade` queues a `pending_deck_upgrades`
/// card_select and otherwise auto-upgrades. (Contrast SandCastle, which is a
/// no-pick random `StableShuffle` upgrade.) Bead `sts2-rs-qw9oe`.
fn listener_yummy_cookie(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::YummyCookie) {
        return EffectResult::NotImplemented;
    }
    rs.upgrade_random_cards(YUMMY_COOKIE_CARDS);
    EffectResult::Resolved
}

/// `Pomander::after_obtained` — C# `Pomander.cs`:
/// `CardSelectCmd.FromDeckForUpgrade(count 1)` → `CardCmd.Upgrade`. A genuine
/// PLAYER PICK (choose 1 card to upgrade), so it correctly routes through the
/// interactive-aware [`RunState::upgrade_random_cards`] (queues a
/// `pending_deck_upgrades` card_select under `interactive_card_upgrade`).
/// Bead `sts2-rs-cvbxs`.
fn listener_pomander(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Pomander) {
        return EffectResult::NotImplemented;
    }
    rs.upgrade_random_cards(1);
    EffectResult::Resolved
}

/// `PrecariousShears::after_obtained` — remove `PRECARIOUS_SHEARS_CARDS
/// = 2` cards from the deck, THEN take `PRECARIOUS_SHEARS_DAMAGE = 16`
/// HP damage. HP-loss leg landed in bead `sts2-rs-cvbxs`; the remove-
/// cards leg landed in bead `sts2-rs-t762n` once
/// `RunState::remove_cards_from_deck` shipped.
///
/// C# `PrecariousShears.AfterObtained` (`PrecariousShears.cs:24-31`)
/// orders the legs `await FromDeckForRemoval(2)` → `RemoveFromDeck` →
/// `Damage(16)` — the damage applies only AFTER the removal pick
/// resolves. In interactive mode the removal surfaces a `card_select`,
/// so the damage must be deferred to after the pick drains (else HP
/// drops a step early vs C#); the non-interactive auto-removal path keeps
/// the eager order since it removes synchronously. Bead `sts2-rs-1hiht`.
fn listener_precarious_shears(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PrecariousShears) {
        return EffectResult::NotImplemented;
    }
    rs.remove_cards_from_deck(PRECARIOUS_SHEARS_CARDS);
    if rs.interactive_card_removal && rs.pending_deck_removals > 0 {
        // Removal is now a pending interactive `card_select`; defer the damage
        // until the batch resolves (drained in `do_remove_deck_card_at`).
        rs.pending_post_removal_hp_loss += PRECARIOUS_SHEARS_DAMAGE;
    } else {
        let _ = rs.lose_hp(PRECARIOUS_SHEARS_DAMAGE);
    }
    EffectResult::Resolved
}

/// `PreciseScissors::after_obtained` — remove 1 card from the deck
/// (Python `shop_event.py::PreciseScissors.after_obtained`: count
/// hard-coded to 1). Bead `sts2-rs-t762n`.
fn listener_precise_scissors(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PreciseScissors) {
        return EffectResult::NotImplemented;
    }
    rs.remove_cards_from_deck(1);
    EffectResult::Resolved
}

/// `EmptyCage::after_obtained` — remove `EMPTY_CAGE_CARDS = 2` cards
/// from the deck (Python `shop_event.py::EmptyCage.after_obtained`).
/// Bead `sts2-rs-t762n`.
fn listener_empty_cage(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::EmptyCage) {
        return EffectResult::NotImplemented;
    }
    rs.remove_cards_from_deck(EMPTY_CAGE_CARDS);
    EffectResult::Resolved
}

/// `PreservedFog::after_obtained` — remove `PRESERVED_FOG_CARDS = 3`
/// cards from the deck AND push one `Folly` curse onto the deck
/// (C# `PreservedFog.cs:20-27` AfterObtained = CardsVar(3) deck
/// removals + AddCurseToDeck<Folly>). Bead `sts2-rs-t762n`
/// (initial port at 5 from Python pin); bead `sts2-rs-rn76k`
/// rebased to 3 per C#.
fn listener_preserved_fog(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PreservedFog) {
        return EffectResult::NotImplemented;
    }
    // C# `PreservedFog.AfterObtained` (`PreservedFog.cs:20-27`, same in
    // v0.111.0) removes the picked cards, THEN `AddCurseToDeck<Folly>`. For an
    // interactive host the removal is a deferred pick, so Folly must wait for
    // the batch to drain or it sits in the removal `card_select` pool and the
    // deck is one card larger at that decision (act23 probe: Regent 1011 /
    // Silent 1011, Vakuu → Preserved Fog). Same shape as Field of Man-Sized
    // Holes' Resist.
    if rs.interactive_card_removal {
        rs.remove_cards_from_deck(PRESERVED_FOG_CARDS);
        if let Some(folly) =
            crate::cards::spec_index::card_pod_from_id(sts2_core::cards::CardId::Folly, false)
        {
            rs.pending_post_removal_cards.push(folly);
        }
        return EffectResult::Resolved;
    }
    rs.remove_cards_from_deck(PRESERVED_FOG_CARDS);
    push_deck_card(rs, sts2_core::cards::CardId::Folly);
    EffectResult::Resolved
}

/// `ArchaicTooth::after_obtained` — find the first card in
/// `rs.master_deck` whose `card_id` matches a starter→ancient mapping
/// entry and rewrite its `card_id` to the ancient version, preserving
/// upgrade + enchantment state (Python
/// `shop_event.py::ArchaicTooth.after_obtained`, lines 682-746 at pin
/// `01c094d`).
///
/// Mapping:
/// * `Bash` → `Break`
/// * `Neutralize` → `Suppress`
/// * `Unleash` → `Protector`
/// * `FallingStar` → `MeteorShower`
/// * `Dualcast` → `Quadcast`
///
/// Python's two-phase design (`setup_for_player` picks the starter
/// type at character creation; `after_obtained` consults that
/// recorded type first) is collapsed to a single pass here. The
/// recorded-starter-id path only differs from the fallback if the
/// deck has been mutated to hold multiple starter types between
/// character creation and relic obtain — that's not a path the
/// engine takes at the current pin. The fallback's "find first
/// matching card in deck order" gives the same result for normal
/// runs.
///
/// `preserve_upgrades=True` + `preserve_enchantments=True` in Python
/// are honored automatically by mutating `card.card_id` in place
/// without touching `card.upgraded` or `card.enchantments`.
/// Bead `sts2-rs-rvwas`.
fn listener_archaic_tooth(rs: &mut RunState) -> EffectResult {
    use sts2_core::cards::CardId;
    if rs.last_obtained_relic != Some(RelicId::ArchaicTooth) {
        return EffectResult::NotImplemented;
    }
    let mapping = |id: CardId| -> Option<CardId> {
        match id {
            CardId::Bash => Some(CardId::Break),
            CardId::Neutralize => Some(CardId::Suppress),
            CardId::Unleash => Some(CardId::Protector),
            CardId::FallingStar => Some(CardId::MeteorShower),
            CardId::Dualcast => Some(CardId::Quadcast),
            // v0.113.0 (`ArchaicTooth.cs:48-51`); FoolishShot is v0.113-only.
            CardId::FoolishShot => Some(CardId::DivineShot),
            _ => None,
        }
    };
    // `GetTranscendenceStarterCard` = first matching deck card; the
    // replacement is a fresh card, upgraded and re-enchanted to match
    // (ArchaicTooth.cs:144-166). `CardCmd.Transform` removes the original and
    // APPENDS the replacement to the deck through the add-to-deck hooks
    // (CardCmd.cs:402,427-436) — the old in-place `card_id` rewrite kept the
    // slot, so the next combat's shuffle differed (act23 Defect 1013,
    // Necrobinder 1010/1013, Regent 1011), and left the starter's stats.
    let Some((idx, target)) = rs
        .master_deck
        .iter()
        .enumerate()
        .find_map(|(i, c)| mapping(c.card_id).map(|t| (i, t)))
    else {
        return EffectResult::Resolved;
    };
    let original = rs.master_deck[idx];
    let Some(mut replacement) =
        crate::cards::spec_index::card_pod_from_id(target, original.upgraded)
    else {
        return EffectResult::Resolved;
    };
    replacement.uid = original.uid;
    replacement.enchantments = original.enchantments;
    let _ = rs.transform_specific_cards(&[idx], vec![replacement]);
    EffectResult::Resolved
}

/// `Claws::after_obtained` — transform up to `CLAWS_CARDS = 6` deck cards
/// into Maul, preserving upgrade state and any enchantment Maul can take.
///
/// C# (`Claws.cs:22-29`, v0.107.1 and v0.111.0 alike) opens a min-0/max-6
/// `CardSelectCmd.FromDeckForTransformation` select whose candidates are
/// `Type != Quest && IsTransformable` (`CardSelectCmd.cs:592`); a deck card
/// is transformable iff it is not Eternal (`CardModel.cs:738-751`). A Maul
/// already in the deck is a legal (if pointless) pick. With
/// [`RunState::interactive_card_transform`] set the select opens as
/// [`RunState::pending_claws_selection`] (pick 0..=6, then confirm; GH #1353);
/// otherwise Rust takes the first 6 candidates in deck order. The transform then
/// follows `CardCmd.Transform` (`CardCmd.cs:383-436`): originals leave the
/// deck and each Maul is appended through the add-to-deck hooks in original
/// pile-index order, which [`RunState::transform_specific_cards`] does.
fn listener_claws(rs: &mut RunState) -> EffectResult {
    use sts2_core::cards::CardId;
    if rs.last_obtained_relic != Some(RelicId::Claws) {
        return EffectResult::NotImplemented;
    }
    let interactive = rs.interactive_card_transform;
    let mut candidates = Vec::new();
    let version = rs.game_version;
    let maul_spec = crate::cards::spec_index::spec_for_card_id(CardId::Maul);
    for (i, card) in rs.master_deck.iter().enumerate() {
        if !interactive && candidates.len() >= CLAWS_CARDS {
            break;
        }
        if card.is_quest() || !crate::cards::spec_index::card_is_removable_instance(card) {
            continue;
        }
        // Rebuild the instance from Maul's spec instead of only flipping
        // `card_id`: a `Card` stores its combat fields per-instance, so a
        // bare `card_id = Maul` left a Defend-sourced Maul untargeted and
        // `play_maul`'s `target.expect` panicked (bead sts2-rs-scds8).
        if let Some(mut maul) =
            crate::cards::spec_index::card_pod_from_id(CardId::Maul, card.upgraded)
        {
            maul.uid = card.uid;
            // C# `CreateMaulFromOriginal` (`Claws.cs:46-61`) re-applies the
            // cloned enchantment through `CardCmd.Enchant` only when
            // `enchantmentModel.CanEnchant(maul)` — a Defend's Nimble/Goopy/
            // Imbued or a Strike's Spiral does not survive onto the Maul.
            for (ench, amount) in &card.enchantments {
                if maul_spec.as_ref().is_some_and(|spec| {
                    crate::enchantments::can_enchant_card(spec, &maul, ench, version)
                }) {
                    let _ = crate::enchantments::apply_static_enchantment(&mut maul, ench, amount);
                }
            }
            candidates.push((i, maul));
        }
    }
    if interactive {
        // C# opens the screen even with no candidate; the empty confirm is
        // then the only move, so skip the no-op decision.
        if !candidates.is_empty() {
            rs.pending_claws_selection = Some(sts2_core::runstate::PendingClawsSelection {
                candidates,
                selected: Vec::new(),
                max_count: CLAWS_CARDS,
            });
        }
    } else {
        let (indices, mauls): (Vec<_>, Vec<_>) = candidates.into_iter().unzip();
        let _ = rs.transform_specific_cards(&indices, mauls);
    }
    EffectResult::Resolved
}

/// `NutritiousSoup::after_obtained` — apply the `TezcatarasEmber`
/// static enchantment to every basic-Strike card in the deck (Python
/// `shop_event.py::NutritiousSoup.after_obtained`:
/// `owner.enchant_basic_strikes("TezcatarasEmber")`). Static
/// enchantments use `set_enchantment` (not `add`) so the call is
/// idempotent. Basic Strikes come from the shared
/// [`super::neow_relics::BASIC_STRIKE_IDS`] list (every character's). Bead `sts2-rs-w9vf8`.
fn listener_nutritious_soup(rs: &mut RunState) -> EffectResult {
    use sts2_core::cards::{CardId, EnchantmentId};
    if rs.last_obtained_relic != Some(RelicId::NutritiousSoup) {
        return EffectResult::NotImplemented;
    }
    let is_basic_strike = |id: CardId| super::neow_relics::BASIC_STRIKE_IDS.contains(&id);
    for card in &mut rs.master_deck {
        // `CanEnchant` refuses a card that already carries an enchantment
        // (EnchantmentModel.cs:289, Ember is not stackable).
        if is_basic_strike(card.card_id) && card.enchantments.is_empty() {
            // `CardCmd.Enchant` runs `TezcatarasEmber.OnEnchant`, which zeroes
            // the energy cost (TezcatarasEmber.cs:17) as well as adding
            // Eternal. A bare `set_enchantment` left the Strikes at cost 1
            // (Regent act23 seeds 1012/1016).
            let _ = crate::enchantments::apply_static_enchantment(
                card,
                EnchantmentId::TezcatarasEmber,
                1,
            );
        }
    }
    EffectResult::Resolved
}

/// `Glitter::modify_card_reward_options_late` — add `Glam = 1` to
/// every option in the card-reward list as it's generated (Python
/// `shop_event.py::Glitter.modify_card_reward_options_late`, lines
/// 1605-1621 at pin `01c094d`). Same channel + idiom as FresnelLens
/// (`Nimble`) and the rare-tier eggs (rarity-specific upgrades).
/// Listener iterates `cards: &mut Vec<Card>` and calls
/// `add_enchantment(Glam, 1)` per option `glam.CanEnchant` accepts
/// (`Glitter.cs:18-35`). Bead `sts2-rs-5xg1w`.
fn listener_glitter_card_reward(
    rs: &mut RunState,
    cards: &mut Vec<sts2_core::cards::Card>,
) -> EffectResult {
    use sts2_core::cards::EnchantmentId;
    if !rs.owns_active_relic(RelicId::Glitter) {
        return EffectResult::NotImplemented;
    }
    if cards.is_empty() {
        return EffectResult::NotImplemented;
    }
    // `if (glam.CanEnchant(card))` (Glitter.cs:28): an option WingCharm (or
    // any earlier listener) already enchanted, or a Status / Curse / Quest
    // card, is skipped.
    for card in cards.iter_mut().filter(|c| reward_option_can_enchant(c)) {
        card.add_enchantment(EnchantmentId::Glam, 1);
    }
    EffectResult::Resolved
}

/// `AlchemicalCoffer::after_obtained` — gain 4 potion slots, then fill
/// them with 4 random out-of-combat potions. Mirrors C#
/// `MegaCrit.Sts2.Core.Models.Relics/AlchemicalCoffer.cs:20-29`
/// (`AfterObtained`): `GainMaxPotionCount(PotionSlots=4)` THEN
/// `PotionFactory.CreateRandomPotionsOutOfCombat(owner, 4,
/// RunState.Rng.CombatPotionGeneration)` + a `PotionCmd.TryToProcure`
/// loop into the new slots. The roll uses the `combat_potion_generation`
/// stream to match C#'s `Rng.CombatPotionGeneration` so the cursor stays
/// faithful. Each potion is procured into slot `originalSlotCount + i`
/// (`add_potion_to_belt_at_slot`), so pre-existing empty slots stay
/// empty.
fn listener_alchemical_coffer(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::AlchemicalCoffer) {
        return EffectResult::NotImplemented;
    }
    let original_slot_count = rs.potion_belt_cap();
    rs.gain_potion_slots(ALCHEMICAL_COFFER_POTION_SLOTS);
    // v0.113.0 rolls on `PlayerRng.NonCombatGeneration` (`AlchemicalCoffer.cs`).
    let kinds = crate::potions::all::create_random_potions_out_of_combat(
        rs.rng.non_combat_or_combat_potion(),
        rs.character_id,
        ALCHEMICAL_COFFER_POTION_SLOTS as usize,
    );
    for (i, kind) in kinds.into_iter().enumerate() {
        // Direct `PotionCmd.TryToProcure(potion, owner, originalSlotCount
        // + i)` loop, NOT a reward screen: each potion goes into its OWN
        // new slot (earlier empty slots stay empty), and C# drops on
        // TooFull with no discard-to-claim offer, so count the overflow
        // but never queue a swap (GH #252).
        let slot = u8::try_from(original_slot_count + i).unwrap_or(u8::MAX);
        if !rs.add_potion_to_belt_at_slot(kind, slot) {
            rs.potions_overflow_dropped += 1;
        }
    }
    EffectResult::Resolved
}

/// `NewLeaf::after_obtained` — open a transform-1-card prompt on
/// the run-side master deck. Python pin
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::NewLeaf`:
/// ```text
/// candidates = owner.transformable_deck_cards()
/// if owner.run_state.enable_deck_choice_requests:
///     owner.offer_transform_cards_reward(1, cards=candidates)
///     return
/// owner.transform_cards(1, cards=candidates)
/// ```
///
/// **Port stance (bead `sts2-rs-ytiok`, 70anj.4.10; interactive surface
/// 6ynt.4.deck-add / GH #34):** calls
/// [`RunState::offer_transform_cards_reward`] with `count = 1`. Under
/// `interactive_card_transform` (workbench / run_parity lockstep) that
/// queues [`RunState::pending_deck_transforms`] picks the run loop
/// surfaces as a deck `card_select`, with the replacement rolled at
/// resolve time off `Rng.Niche` (C# `NewLeaf.AfterObtained` →
/// `CardSelectCmd.FromDeckForTransformation` →
/// `CardCmd.TransformToRandom(item, RunState.Rng.Niche)`) via the
/// `roll_transform_replacement` content callback. The legacy default
/// opens the run-level `pending_choice` prompt (bead `sts2-rs-aw73x`).
///
/// **Parity drift:** the Rust `offer_transform_cards_reward` opens a
/// prompt over the **entire** `master_deck`, whereas Python filters
/// to `transformable_deck_cards()` (excludes Quest cards + non-
/// removable cards). At this pin Quest/curse handling isn't deeply
/// modelled in the POD Card, so the over-offer is a temporary
/// divergence pinned by [`tests::new_leaf_opens_transform_prompt`].
/// Ownership-gated via `rs.relics.contains` per the canonical
/// after_obtained pattern (LeesWaffle, GoldenPearl, …).
fn listener_new_leaf(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::NewLeaf) {
        return EffectResult::NotImplemented;
    }
    rs.offer_transform_cards_reward(1);
    EffectResult::Resolved
}

/// `FurCoat.AfterObtained` (FurCoat.cs:66-71): latch the obtain act into
/// [`RunState::fur_coat_act_index`] and mark the act map that already exists.
/// An Ancient hands the relic over after the act's map is generated, so this
/// is where the marks are normally made; the generation-time
/// [`listener_fur_coat_modify_generated_map_late`] only covers a map built
/// later in the same act. The latch keeps the first act it saw (C# overwrites
/// it on every obtain; a FurCoat is obtained at most once per run).
fn listener_fur_coat_after_obtained(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::FurCoat) {
        return EffectResult::NotImplemented;
    }
    if rs.fur_coat_act_index.is_none() {
        rs.fur_coat_act_index = Some(rs.act_index);
    }
    if rs.fur_coat_act_index == Some(rs.act_index) {
        if let Some(mut map) = rs.current_act_map.take() {
            fur_coat_mark_rooms(rs, &mut map);
            rs.current_act_map = Some(map);
        }
    }
    EffectResult::Resolved
}

/// `FurCoat.ModifyGeneratedMapLate` (FurCoat.cs:73-76): mark a map generated
/// in the obtain act. A no-op in every other act (`CurrentActIndex !=
/// FurCoatActIndex`, FurCoat.cs:81-84).
fn listener_fur_coat_modify_generated_map_late(
    rs: &mut RunState,
    act_map: &mut sts2_core::map::ActMap,
    act_index: u8,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::FurCoat) {
        return EffectResult::NotImplemented;
    }
    if rs.fur_coat_act_index != Some(act_index) {
        return EffectResult::NotImplemented;
    }
    fur_coat_mark_rooms(rs, act_map)
}

/// `FurCoat.AddMarkedRooms` (FurCoat.cs:78-128). A map that already carries
/// the marks keeps them (the C# saved-coords branch re-applies the same coords
/// and draws nothing). Otherwise the candidates are the live Monster/Elite
/// points in `GetAllMapPoints` order — column-major over the grid, which is
/// `ActMap::grid`'s `(col, row)` key order and skips pruned ghost nodes —
/// `UnstableShuffle`d by `new Rng(Owner, Id, 0)` (the `FUR_COAT` named
/// stream, slot 0) and the first `Combats` of them marked.
fn fur_coat_mark_rooms(rs: &RunState, act_map: &mut sts2_core::map::ActMap) -> EffectResult {
    use sts2_core::map::{MapCoord, MapPointType};

    if !act_map.marked_coords_for(RelicId::FurCoat).is_empty() {
        return EffectResult::Resolved;
    }
    let mut candidates: Vec<MapCoord> = act_map
        .grid
        .iter()
        .filter(|(_, &idx)| {
            matches!(
                act_map.points[idx as usize].point_type,
                MapPointType::Monster | MapPointType::Elite
            )
        })
        .map(|(&coord, _)| coord)
        .collect();
    let mut rng = rs.rng.named("FUR_COAT");
    rng.shuffle_for(rs.game_version, &mut candidates);
    candidates.truncate(fur_coat_combats(rs.game_version));
    act_map.mark_rooms(candidates, RelicId::FurCoat);
    EffectResult::Resolved
}

/// `PandorasBox::after_obtained` — Python pin
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::PandorasBox`:
/// ```python
/// def after_obtained(self, owner):
///     cards = owner.basic_strike_defend_cards()
///     ...
///     owner.transform_specific_cards(cards)
/// ```
/// Walks `rs.master_deck`, collects every index whose card matches
/// the character's basic Strike or Defend (via
/// [`crate::cards::spec_index::basic_strike_defend_cards`]), rolls one
/// replacement card per match from the character pool, and applies
/// the in-place swap through
/// [`sts2_core::runstate::RunState::transform_specific_cards`] (bead
/// `sts2-rs-ab8uo`).
///
/// **Parity approximation:** Python's `create_transform_card` samples
/// from an "original-card-specific" eligibility pool that varies per
/// transform source. The Rust port doesn't have that depth yet; we
/// approximate by sampling Common-rarity cards from the character
/// pool via [`crate::cards::spec_index::random_card_from_rarity`].
/// This matches the established sibling-relic convention (Astrolabe
/// at line 2498 uses the same `random_card_from_rarity(rs, _, Rare)`
/// shape). A full rarity-rolled / original-keyed sampler is a
/// follow-up beyond this body-fill bead.
///
/// The listener no-ops for any character returning `None` from
/// `basic_strike_defend_cards`; every character now has a pair, so that
/// branch is unreachable.
/// Pickup behaviour stays observable for Ironclad / Silent / Defect.
///
/// Owner-gated via `rs.relics.contains` per the canonical
/// after-obtained pattern (TouchOfOrobas / LeesWaffle / NewLeaf).
fn listener_pandoras_box(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PandorasBox) {
        return EffectResult::NotImplemented;
    }
    let character_id = rs.character_id;
    let Some([strike_id, defend_id]) =
        crate::cards::spec_index::basic_strike_defend_cards(character_id)
    else {
        return EffectResult::NotImplemented;
    };
    let indices: Vec<usize> = rs
        .master_deck
        .iter()
        .enumerate()
        .filter(|(_, card)| card.card_id == strike_id || card.card_id == defend_id)
        .map(|(idx, _)| idx)
        .collect();
    if indices.is_empty() {
        return EffectResult::Resolved;
    }
    pandoras_box_transform(rs, &indices)
}

/// PandorasBox: each basic Strike / Defend becomes a random card of
/// its own pool, rolled off the shared `Niche` stream in deck order:
///
/// [decompiled C# omitted — see the cited path]
///
/// (decomp `Models.Relics/PandorasBox.cs:22`; `CardCmd.Transform` materializes
/// the transformations once, in order, `CardCmd.cs:377`). The pool is the
/// default transform pool (Common/Uncommon/Rare, not MultiplayerOnly, not the
/// original, `CardFactory.cs:171-213`), the one NewLeaf already rolls through
/// [`crate::cards::spec_index::roll_transform_replacement_basic_niche`].
/// Identical in v0.107.1 and v0.111.0, so both builds roll it; the old
/// Python-pin Common-only `rewards` sampler is gone (GH #806).
fn pandoras_box_transform(rs: &mut RunState, indices: &[usize]) -> EffectResult {
    let character_id = rs.character_id;
    let mut replacements: Vec<sts2_core::Card> = Vec::with_capacity(indices.len());
    for &i in indices {
        let original = rs.master_deck[i].card_id;
        let Some(new_id) = crate::cards::spec_index::roll_transform_replacement_basic_niche(
            rs,
            character_id,
            original,
        ) else {
            return EffectResult::NotImplemented;
        };
        let Some(card) = crate::cards::v0111::card_pod_for(new_id, false, rs.game_version) else {
            return EffectResult::NotImplemented;
        };
        replacements.push(card);
    }
    let _ = rs.transform_specific_cards(indices, replacements);
    EffectResult::Resolved
}

// ── SilverCrucible — bead `sts2-rs-535i7` (70anj.4 handoff) ────────────

/// `SilverCrucible::CARD_REWARDS` from Python pin
/// `shop_event.py::SilverCrucible`.
const SILVER_CRUCIBLE_CARD_REWARDS: i32 = 3;

/// Pack `(times_used, treasure_rooms_entered)` into a single i32 for
/// storage in `rs.relic_counters[SilverCrucible]`. Both counters are
/// bounded (CARD_REWARDS=3, treasure rooms per run ≤ a small
/// constant), so 8 bits per field is more than enough. Negative
/// inputs clamp to 0; overflow saturates at 0xFF (defensive).
const fn silver_crucible_pack(times_used: i32, treasure_rooms_entered: i32) -> i32 {
    let t = if times_used < 0 {
        0
    } else if times_used > 0xFF {
        0xFF
    } else {
        times_used
    };
    let r = if treasure_rooms_entered < 0 {
        0
    } else if treasure_rooms_entered > 0xFF {
        0xFF
    } else {
        treasure_rooms_entered
    };
    (r << 8) | t
}

const fn silver_crucible_times_used(packed: i32) -> i32 {
    packed & 0xFF
}

const fn silver_crucible_treasure_rooms_entered(packed: i32) -> i32 {
    (packed >> 8) & 0xFF
}

/// `SilverCrucible::after_room_entered` — Python pin: increment the
/// `_treasure_rooms_entered` counter on every `RoomType::Treasure`.
/// Owner-gated via `rs.relics.contains`.
fn listener_silver_crucible_after_room_entered(
    rs: &mut RunState,
    room_type: RoomType,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::SilverCrucible) {
        return EffectResult::NotImplemented;
    }
    if room_type != RoomType::Treasure {
        return EffectResult::NotImplemented;
    }
    let packed = rs
        .relic_counters
        .get(&RelicId::SilverCrucible)
        .copied()
        .unwrap_or(0);
    let new_packed = silver_crucible_pack(
        silver_crucible_times_used(packed),
        silver_crucible_treasure_rooms_entered(packed) + 1,
    );
    rs.relic_counters
        .insert(RelicId::SilverCrucible, new_packed);
    EffectResult::Resolved
}

/// `SilverCrucible::should_generate_treasure` — Python pin returns
/// `self._treasure_rooms_entered > 1`. Skips the first treasure
/// room. The `after_room_entered` listener increments the counter
/// BEFORE this check fires (Python's `_fire_modifiers_after_room_entered`
/// runs before the should-generate-treasure poll in `_enter_room`),
/// so:
/// - First treasure: counter becomes 1, `1 > 1` is false → SKIP.
/// - Second+: counter ≥ 2, `2 > 1` is true → don't skip.
///
/// Rust channel encoding (bead `sts2-rs-elcvh`): `Resolved` = SKIP,
/// `NotImplemented` = abstain.
fn listener_silver_crucible_should_generate_treasure(rs: &mut RunState) -> EffectResult {
    if !rs.owns_active_relic(RelicId::SilverCrucible) {
        return EffectResult::NotImplemented;
    }
    let packed = rs
        .relic_counters
        .get(&RelicId::SilverCrucible)
        .copied()
        .unwrap_or(0);
    if silver_crucible_treasure_rooms_entered(packed) > 1 {
        EffectResult::NotImplemented
    } else {
        EffectResult::Resolved
    }
}

/// `SilverCrucible::modify_card_reward_options_late` — pre-upgrade
/// every card option until `_times_used >= CARD_REWARDS`. Python
/// uses a per-reward `_silver_crucible_upgraded` latch to avoid
/// double-counting across multiple listener fires for the same
/// reward; the Rust channel has no per-reward identifier so the
/// listener increments `_times_used` once per dispatch (the call
/// convention is one dispatch per reward generation, so semantics
/// line up under that invariant).
fn listener_silver_crucible_modify_card_reward_options_late(
    rs: &mut RunState,
    cards: &mut Vec<sts2_core::Card>,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::SilverCrucible) {
        return EffectResult::NotImplemented;
    }
    // Only a slate with `IsCardReward` (decomp
    // `Models.Relics/SilverCrucible.cs:104`, both builds); see
    // `RunState::offer_card_slate`.
    if rs.card_slate_not_card_reward {
        return EffectResult::NotImplemented;
    }
    let packed = rs
        .relic_counters
        .get(&RelicId::SilverCrucible)
        .copied()
        .unwrap_or(0);
    let times_used = silver_crucible_times_used(packed);
    if times_used >= SILVER_CRUCIBLE_CARD_REWARDS {
        return EffectResult::NotImplemented;
    }
    for card in cards.iter_mut() {
        let _ = card.upgrade_for(rs.game_version);
    }
    let new_packed = silver_crucible_pack(
        times_used + 1,
        silver_crucible_treasure_rooms_entered(packed),
    );
    rs.relic_counters
        .insert(RelicId::SilverCrucible, new_packed);
    EffectResult::Resolved
}

// ── GoldenCompass — bead `sts2-rs-k2nuu` (70anj.4 handoff) ─────────────
//
// Captured-act sentinel in `rs.relic_counters[GoldenCompass]`:
// - 0 = not picked up yet (default for unset entries)
// - 1/2/3 = the `rs.act_index` at obtain time
//
// All three listeners gate on `rs.relics.contains(&GoldenCompass)`
// AND the captured-act match.

/// `GoldenCompass::after_obtained` — capture the current
/// `rs.act_index` into `rs.relic_counters[GoldenCompass]`. Python
/// pin also calls `run_state.generate_map()` to force immediate
/// re-roll of the current act's map; the Rust port doesn't have
/// that primitive yet (no `RunState::generate_map` — RunState
/// doesn't own an `ActMap` field at this pin) so the golden-path
/// swap activates on the next driver-side regeneration. Documented
/// divergence.
fn listener_golden_compass_after_obtained(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::GoldenCompass) {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters
        .insert(RelicId::GoldenCompass, i32::from(rs.act_index));
    // Bead `sts2-rs-4aunm` closes the in-act regen gap that k2nuu
    // documented: pickup-time, force the current act's map to be
    // re-rolled so the `modify_generated_map` listener below can
    // swap in the golden-path layout immediately rather than
    // waiting for the next driver-side regen at act-advance.
    rs.generate_map();
    EffectResult::Resolved
}

/// `GoldenCompass::modify_generated_map` — when the freshly-rolled
/// map's act matches the captured act, overwrite the contents with
/// `generate_golden_path_map(1)`. Python pin returns a fresh map
/// object; the Rust channel signature is `&mut ActMap` so we
/// overwrite in place via `*act_map = ...`. Mirrors:
///
/// ```python
/// def modify_generated_map(self, owner, run_state, act_map, act_index):
///     if act_index != self._golden_path_act:
///         return act_map
///     from sts2_env.map.generator import generate_golden_path_map
///     return generate_golden_path_map(player_count=len(getattr(run_state, "players", (owner,))))
/// ```
///
/// `player_count` always passes 1 in the single-player Rust port.
fn listener_golden_compass_modify_generated_map(
    rs: &mut RunState,
    act_map: &mut sts2_core::map::ActMap,
    act_index: u8,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::GoldenCompass) {
        return EffectResult::NotImplemented;
    }
    let captured = rs
        .relic_counters
        .get(&RelicId::GoldenCompass)
        .copied()
        .unwrap_or(0);
    if captured == 0 || i32::from(act_index) != captured {
        return EffectResult::NotImplemented;
    }
    *act_map = sts2_core::map_generator::generate_golden_path_map(1);
    EffectResult::Resolved
}

/// `GoldenCompass::modify_unknown_map_point_room_types` — on the
/// captured golden-path act, force every `?` resolution to
/// `RoomType::Event`. Python pin:
///
/// ```python
/// def modify_unknown_map_point_room_types(self, owner, room_types):
///     run_state = getattr(owner, "run_state", None)
///     if run_state is None or run_state.current_act_index != self._golden_path_act:
///         return room_types
///     return {RoomType.EVENT}
/// ```
///
/// Rust channel mutates the set in place — clear it and insert the
/// single `Event` variant.
fn listener_golden_compass_modify_unknown_map_point_room_types(
    rs: &mut RunState,
    room_types: &mut std::collections::BTreeSet<RoomType>,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::GoldenCompass) {
        return EffectResult::NotImplemented;
    }
    let captured = rs
        .relic_counters
        .get(&RelicId::GoldenCompass)
        .copied()
        .unwrap_or(0);
    if captured == 0 || i32::from(rs.act_index) != captured {
        return EffectResult::NotImplemented;
    }
    room_types.clear();
    room_types.insert(RoomType::Event);
    EffectResult::Resolved
}

/// `Orrery::after_obtained` — push `CARDS = 5` `Reward::Card`
/// sentinels to `rs.pending_rewards` (Python
/// `shop_event.py::Orrery.after_obtained`: `for _ in range(self.CARDS):
/// owner.offer_card_reward()`). Each entry is an empty `CardReward`
/// placeholder pending umbrella sts2-rs-01o random-card rolling;
/// same convention as BlackStar / PrayerWheel / WhiteStar. Bead
/// `sts2-rs-psf9`.
fn listener_orrery(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Orrery) {
        return EffectResult::NotImplemented;
    }
    for _ in 0..ORRERY_CARDS {
        rs.pending_rewards.push(Reward::Card(CardReward::empty()));
    }
    EffectResult::Resolved
}

/// `BlackStar::modify_rewards` — append a `Reward::Relic` to the
/// rewards vec when the player just cleared an Elite room (Python
/// `shop_event.py::BlackStar.modify_rewards`, room-type gate
/// `RoomType.ELITE`). Python's `RelicReward(player_id)` with no
/// concrete `relic_id` argument rolls a random relic at flush time;
/// the current Rust `RelicReward` POD only carries a concrete
/// `relic_id`, so this listener follows the WongosMysteryTicket
/// placeholder convention and pushes the relic's own id as a sentinel.
/// The random-roll will land once umbrella `sts2-rs-01o` extends
/// `RelicReward` with a "pick from N" variant. Ownership-gated.
/// Bead `sts2-rs-83sa`.
fn listener_black_star(
    rs: &mut RunState,
    rewards: &mut Vec<Reward>,
    room_type: RoomType,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::BlackStar) {
        return EffectResult::NotImplemented;
    }
    if room_type != RoomType::Elite {
        return EffectResult::NotImplemented;
    }
    rewards.push(Reward::Relic(RelicReward::new(RelicId::BlackStar)));
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Public per-relic constants (pinned from Python)
// ---------------------------------------------------------------------------
//
// Exposed as `pub const` so the downstream integration bead can wire
// real hook bodies without re-typing the numbers from Python.

/// `VeryHotCocoa::ENERGY` — energy gained on player round 1
/// (`shop_event.py::VeryHotCocoa.ENERGY`). Bead `sts2-rs-70anj.1`.
pub const VERY_HOT_COCOA_ENERGY: i32 = 4;

/// `PaelsFlesh::ENERGY` — energy gained each player turn from round
/// 3 onward (`shop_event.py::PaelsFlesh.ENERGY`). Bead `sts2-rs-p6dj`.
pub const PAELS_FLESH_ENERGY: i32 = 1;
/// `PaelsFlesh` — first player round in which the energy gain fires
/// (`if combat.round_number >= 3`,
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::PaelsFlesh`).
/// `u16` matches the engine's `round_no()` type.
pub const PAELS_FLESH_ROUND_MIN: u16 = 3;

/// `BeltBuckle::DEXTERITY` — +2 / -2 Dex on potion-procurement state
/// (C# `BeltBuckle.cs` `PowerVar<DexterityPower>(2m)`).
pub const BELT_BUCKLE_DEXTERITY: i32 = sts2_core::BELT_BUCKLE_DEXTERITY;

/// `PaelsLegion::Turns` — cooldown length (turns) after the block-double
/// fires, before it re-arms (C# `PaelsLegion.cs` `DynamicVar("Turns", 2)`).
/// Bead `sts2-rs-ppabi`.
pub const PAELS_LEGION_TURNS: i32 = 2;
/// `PaelsLegion` pet HP — behaviourless 9999-HP companion (C#
/// `Monsters/PaelsLegion.cs` `MinInitialHp == MaxInitialHp == 9999`).
/// Bead `sts2-rs-ppabi`.
const PAELS_LEGION_PET_HP: i32 = 9999;

/// `Bread::GAIN_ENERGY` / `LOSE_ENERGY` — +1 max energy, -2 round-1
/// (`shop_event.py::Bread`).
pub const BREAD_GAIN_ENERGY: i32 = 1;
pub const BREAD_LOSE_ENERGY: i32 = 2;

/// `Brimstone::SELF_STRENGTH` / `ENEMY_STRENGTH` — per-turn +2 Str /
/// +1 Str to all enemies (`shop_event.py::Brimstone`).
pub const BRIMSTONE_SELF_STRENGTH: i32 = 2;
pub const BRIMSTONE_ENEMY_STRENGTH: i32 = 1;

/// `PhilosophersStone::ENEMY_STRENGTH` / `ENERGY` — at combat start
/// every alive enemy gains STRENGTH+1; +1 max energy
/// (`shop_event.py::PhilosophersStone`). Only the `ENEMY_STRENGTH`
/// leg is wired by bead `sts2-rs-w0wl`; the `modify_max_energy: +1`
/// leg ships in a follow-up bead.
pub const PHILOSOPHERS_STONE_ENEMY_STRENGTH: i32 = 1;
pub const PHILOSOPHERS_STONE_ENERGY: i32 = 1;

/// `BoomingConch::EXTRA_DRAW` — round-1 elite-only +2 hand draw
/// (`shop_event.py::BoomingConch`). The listener body in
/// `sts2_core::relic_effects::on_modify_hand_draw_booming_conch`
/// inlines the literal `2`; a parity test in this file keeps the
/// constant and the inlined literal in lockstep.
pub const BOOMING_CONCH_EXTRA_DRAW: i32 = 2;

/// `MassiveScroll::CARDS` — multiplayer-card offers (coop is OOS,
/// so this is the count of CardReward sentinels pushed on obtain;
/// `shop_event.py::MassiveScroll` hard-codes 3).
pub const MASSIVE_SCROLL_CARDS: usize = 3;

/// `Cauldron::POTIONS` — random potions offered on obtain
/// (`shop_event.py::Cauldron.POTIONS`).
pub const CAULDRON_POTIONS: i32 = 5;

/// `ChemicalX::INCREASE` — bonus added to X-cost card X-value
/// (`shop_event.py::ChemicalX.INCREASE`).
pub const CHEMICAL_X_INCREASE: i32 = 2;

/// `DragonFruit::MAX_HP` — max-HP gained per gold-gain event
/// (`shop_event.py::DragonFruit.MAX_HP`).
pub const DRAGON_FRUIT_MAX_HP: i32 = 1;

/// `LeesWaffle::MAX_HP` — max-HP gained (and full heal) on obtain
/// (`shop_event.py::LeesWaffle.MAX_HP`).
pub const LEES_WAFFLE_MAX_HP: i32 = 7;

/// `LoomingFruit::MAX_HP` — max-HP gained on obtain
/// (`shop_event.py::LoomingFruit.MAX_HP`).
pub const LOOMING_FRUIT_MAX_HP: i32 = 31;

/// `GoldenPearl::GOLD` — gold granted on obtain
/// (`shop_event.py::GoldenPearl.GOLD`).
pub const GOLDEN_PEARL_GOLD: i32 = 150;

/// `SignetRing::GOLD` — gold granted on obtain
/// (`shop_event.py::SignetRing.GOLD`).
pub const SIGNET_RING_GOLD: i32 = 999;

/// `Orrery::CARDS` — number of card-reward sentinels pushed on obtain
/// (`shop_event.py::Orrery.CARDS`).
pub const ORRERY_CARDS: usize = 5;

/// `GnarledHammer::CARDS` — count of deck cards enchanted on obtain
/// (`shop_event.py::GnarledHammer.CARDS`). Bead `sts2-rs-qsxiy`.
pub const GNARLED_HAMMER_CARDS: usize = 3;

/// `GnarledHammer::SHARP` — Sharp keyword amount applied to each
/// enchanted card (`shop_event.py::GnarledHammer.SHARP`). Bead
/// `sts2-rs-qsxiy`.
pub const GNARLED_HAMMER_SHARP: i32 = 3;

/// `Kifuda::CARDS` — count of deck cards enchanted with Adroit on
/// obtain (`shop_event.py::Kifuda.CARDS`). Bead `sts2-rs-4nyt7`.
pub const KIFUDA_CARDS: usize = 3;

/// `Kifuda::ADROIT` — Adroit keyword amount applied to each
/// enchanted card (`shop_event.py::Kifuda` hard-codes `3` in the
/// `enchant_selected_cards` call). Bead `sts2-rs-4nyt7`.
pub const KIFUDA_ADROIT: i32 = 3;

/// `PunchDagger::MOMENTUM` — Momentum keyword amount applied to the
/// one card enchanted on obtain (`shop_event.py::PunchDagger.MOMENTUM`).
/// Count is hard-coded to 1 in the Python call; no separate constant.
/// Bead `sts2-rs-4nyt7`.
pub const PUNCH_DAGGER_MOMENTUM: i32 = 5;

/// `BeautifulBracelet::CARDS` — count of deck cards enchanted with
/// Swift on obtain (`shop_event.py::BeautifulBracelet.CARDS`). Bead
/// `sts2-rs-4nyt7`.
pub const BEAUTIFUL_BRACELET_CARDS: usize = 3;

/// `BeautifulBracelet::SWIFT` — Swift keyword amount applied to each
/// enchanted card (`shop_event.py::BeautifulBracelet.SWIFT`). Bead
/// `sts2-rs-4nyt7`.
pub const BEAUTIFUL_BRACELET_SWIFT: i32 = 3;

/// v0.111.0 `BeautifulBracelet` `CardsVar(4)` (decomp BeautifulBracelet.cs:25).
pub const BEAUTIFUL_BRACELET_CARDS_V0111: usize = 4;

/// v0.111.0 `BeautifulBracelet` `DynamicVar("Swift", 2m)` (decomp
/// BeautifulBracelet.cs:26).
pub const BEAUTIFUL_BRACELET_SWIFT_V0111: i32 = 2;

/// `TriBoomerang::CARDS` — count of deck cards enchanted with
/// Instinct on obtain (`shop_event.py::TriBoomerang.CARDS`).
/// Instinct amount is hard-coded to 1 in the Python call (no separate
/// class const). Bead `sts2-rs-70anj.4`.
pub const TRI_BOOMERANG_CARDS: usize = 3;

/// `BiiigHug::CARDS` — count of cards removed from master deck on
/// obtain (`shop_event.py::BiiigHug.CARDS`). Bead `sts2-rs-70anj.4`.
pub const BIIIG_HUG_CARDS: usize = 4;

/// `ToyBox::RELICS` — count of wax-relic RelicReward sentinels pushed
/// on obtain (`shop_event.py::ToyBox.RELICS`). Bead `sts2-rs-70anj.4`.
pub const TOY_BOX_RELICS: usize = 4;

/// `ToyBox::DynamicVars["Relics"]` for the build: [`TOY_BOX_RELICS`] in
/// v0.107.1.
///
/// v0.111.0: `DynamicVar("Relics", 4m)` -> `5m` (decomp ToyBox.cs:83).
#[must_use]
pub const fn toy_box_relics(version: sts2_core::GameVersion) -> usize {
    match version {
        sts2_core::GameVersion::V0_107_1 => TOY_BOX_RELICS,
        sts2_core::GameVersion::V0_111_0 | sts2_core::GameVersion::V0_113_0 => 5,
    }
}

/// `ToyBox::DynamicVars["Combats"]` — combats between melts
/// (`MegaCrit.Sts2.Core.Models.Relics/ToyBox.cs:84`, `new DynamicVar("Combats", 3m)`).
/// `AfterCombatEnd` increments `CombatsSeen` FIRST, then melts when
/// `CombatsSeen % Combats == 0`, so the first melt lands on combat 3.
pub const TOY_BOX_COMBATS: i32 = 3;

/// `WingCharm::SWIFT` — Swift enchant amount applied to one random
/// card-reward option (`shop_event.py::WingCharm.SWIFT`). Bead
/// `sts2-rs-70anj.4`.
pub const WING_CHARM_SWIFT: i32 = 1;

/// `VelvetChoker::MAX_CARDS` — per-turn play cap (Python
/// `shop_event.py::VelvetChoker.MAX_CARDS`). Bead `sts2-rs-70anj.4`.
pub const VELVET_CHOKER_MAX_CARDS: i32 = 6;

/// `Astrolabe::CARDS` — count of deck cards selected for transform +
/// upgrade on obtain (`shop_event.py::Astrolabe.CARDS`). Bead
/// `sts2-rs-kluld`.
pub const ASTROLABE_CARDS: usize = 3;

/// `StoneHumidifier::MAX_HP` — max-HP gained per rest-site heal
/// (`shop_event.py::StoneHumidifier.MAX_HP`). Bead `sts2-rs-70anj.3.17`.
pub const STONE_HUMIDIFIER_MAX_HP: i32 = 5;

/// `CookOption::CARD_COUNT` — number of deck cards removed by Cook
/// (`run/rest_site.py::CookOption.CARD_COUNT`). Bead `sts2-rs-ekf4j`.
pub const COOK_OPTION_CARD_COUNT: usize = 2;

/// `CookOption::MAX_HP` — max HP gained on Cook
/// (`run/rest_site.py::CookOption.MAX_HP`). Bead `sts2-rs-ekf4j`.
pub const COOK_OPTION_MAX_HP: i32 = 9;

/// `Toolbox::CARDS` — number of colorless choices offered on round 1
/// (`shop_event.py::Toolbox.CARDS`).
pub const TOOLBOX_CARDS: usize = 3;

/// `ChoicesParadox::CARDS` — number of retain-tagged character choices
/// offered on round 1 (`shop_event.py::ChoicesParadox.CARDS`).
pub const CHOICES_PARADOX_CARDS: usize = 5;

/// `EmptyCage::CARDS` — count of deck cards removed on obtain
/// (`shop_event.py::EmptyCage.CARDS`). Bead `sts2-rs-kluld`.
pub const EMPTY_CAGE_CARDS: usize = 2;

/// `DistinguishedCape::HP_LOSS` — max-HP lost on obtain
/// (`shop_event.py::DistinguishedCape.HP_LOSS`). Bead `sts2-rs-r2y4x`.
pub const DISTINGUISHED_CAPE_HP_LOSS: i32 = 9;

/// `DistinguishedCape::CARDS` — count of Apparition cards added on
/// obtain (`shop_event.py::DistinguishedCape.CARDS`). Bead
/// `sts2-rs-r2y4x`. Currently documentation-only — the Apparition
/// deck-add clause is deferred pending the run-side
/// `add_card_to_deck` primitive (umbrella `sts2-rs-01o`).
pub const DISTINGUISHED_CAPE_CARDS: usize = 3;

/// v0.111.0 `DistinguishedCape` `DynamicVar("Curses", 2m)` (decomp
/// DistinguishedCape.cs:26).
pub const DISTINGUISHED_CAPE_CURSES_V0111: usize = 2;

/// `SandCastle::CARDS` — count of random deck cards upgraded on
/// obtain (`shop_event.py::SandCastle.CARDS`). Bead `sts2-rs-r2y4x`.
pub const SAND_CASTLE_CARDS: usize = 6;

/// `GlassEye` — count of card-reward sentinels pushed on obtain
/// (Python `shop_event.py::GlassEye.after_obtained` hard-codes a
/// 5-iteration loop over `(COMMON, COMMON, UNCOMMON, UNCOMMON, RARE)`;
/// no class-level constant). Bead `sts2-rs-66l9f`.
pub const GLASS_EYE_CARDS: usize = 5;

/// `ScrollBoxes` — count of card-bundle sentinels pushed on obtain
/// (Python `shop_event.py::ScrollBoxes.offer_card_bundles` produces
/// 2 bundles; no class-level constant). Bead `sts2-rs-66l9f`.
pub const SCROLL_BOXES_BUNDLES: usize = 2;

/// `PaelsHorn` — count of Relax cards added on obtain. Python hard-
/// codes a `range(2)` loop (`shop_event.py::PaelsHorn`); no class-
/// level constant. Bead `sts2-rs-a49q3`.
pub const PAELS_HORN_CARDS: usize = 2;

/// `LeadPaperweight` — count of colorless-card-reward sentinels
/// pushed on obtain (Python `shop_event.py::LeadPaperweight.after_obtained`:
/// `owner.offer_colorless_cards(2)`; no class const). Bead
/// `sts2-rs-qw9oe`.
pub const LEAD_PAPERWEIGHT_CARDS: usize = 2;

/// `SeaGlass` `CardsVar(15)` (`SeaGlass.cs:70`): cards offered on obtain,
/// a third per rarity.
pub const SEA_GLASS_CARDS: usize = 15;

/// `YummyCookie::CARDS` — count of deck cards upgraded on obtain
/// (`shop_event.py::YummyCookie.CARDS`). Bead `sts2-rs-qw9oe`.
pub const YUMMY_COOKIE_CARDS: usize = 4;

/// `PrecariousShears::CARDS` — count of deck cards removed on obtain
/// (`shop_event.py::PrecariousShears.CARDS`). Bead `sts2-rs-cvbxs`.
/// Pinned for parity; the remove-cards leg is deferred pending a
/// run-side `remove_cards_from_deck` primitive.
pub const PRECARIOUS_SHEARS_CARDS: usize = 2;

/// `PrecariousShears::DAMAGE` — HP damage taken on obtain. C#
/// `PrecariousShears.cs` `CanonicalVars` is `[CardsVar(2),
/// DamageVar(16m, Unpowered)]`; the Python pin's
/// `shop_event.py::PrecariousShears.DAMAGE = 13` drifted from the
/// shipped game. Bead `sts2-rs-3bigm`.
pub const PRECARIOUS_SHEARS_DAMAGE: i32 = 16;

/// `PreservedFog::CardsVar = 3` per C# v0.103.2
/// `MegaCrit.Sts2.Core.Models.Relics/PreservedFog.cs:18` — count of
/// deck cards removed on obtain (then 1 Folly curse is added).
/// Bead `sts2-rs-t762n` introduced the value as 5 (Python pin
/// `shop_event.py::PreservedFog.CARDS = 5`); bead `sts2-rs-rn76k`
/// rebases to 3 per the C# decomp (CLAUDE.md hard rule).
pub const PRESERVED_FOG_CARDS: usize = 3;

/// `Claws::CARDS` — count of deck cards transformed into Maul on
/// obtain (`shop_event.py::Claws.CARDS`). Bead `sts2-rs-w9vf8`.
pub const CLAWS_CARDS: usize = 6;

// `AlchemicalCoffer::POTION_SLOTS` constant is already defined
// elsewhere in this file (`ALCHEMICAL_COFFER_POTION_SLOTS`); the
// `sts2-rs-cvbxs` listener wires the `gain_potion_slots(N)` leg
// using that existing constant. Random procure-potion clauses are
// deferred pending a run-side `procure_potion` primitive.

// `CursedPearl::GOLD` — gold granted on obtain — pinned upstream by
// `sts2-rs-n06dm` as [`CURSED_PEARL_GOLD`]; `sts2-rs-a49q3` reuses
// it inside `listener_cursed_pearl` alongside the Greed card-add.

/// `LeafyPoultice::MAX_HP_LOSS` — max-HP lost on obtain. C#
/// `LeafyPoultice.cs` `CanonicalVars` is `MaxHpVar(12m)`; the
/// Python pin's `shop_event.py::LeafyPoultice.MAX_HP_LOSS = 10`
/// drifted from the shipped game. Bead `sts2-rs-3bigm`.
pub const LEAFY_POULTICE_MAX_HP_LOSS: i32 = 12;

/// `CallingBell::RELICS` — count of RelicReward sentinels pushed on
/// obtain (`shop_event.py::CallingBell.RELICS`).
pub const CALLING_BELL_RELICS: usize = 3;

/// `SmallCapsule::RELICS` — RelicReward sentinels offered on obtain
/// (`shop_event.py::SmallCapsule.after_obtained`: hard-coded `1`).
pub const SMALL_CAPSULE_RELICS: usize = 1;

/// `NutritiousOyster::MAX_HP` — max-HP gained on obtain
/// (`shop_event.py::NutritiousOyster.MAX_HP`).
pub const NUTRITIOUS_OYSTER_MAX_HP: i32 = 11;

/// `MembershipCard::DISCOUNT` — merchant discount %
/// (`shop_event.py::MembershipCard.DISCOUNT`).
pub const MEMBERSHIP_CARD_DISCOUNT: i32 = 50;

/// `MysticLighter::EXTRA_DAMAGE` — bonus damage on enchanted attacks
/// (`shop_event.py::MysticLighter.EXTRA_DAMAGE`).
pub const MYSTIC_LIGHTER_EXTRA_DAMAGE: i32 = 9;

/// `NinjaScroll::SHIVS` — round-1 Shivs generated
/// (`shop_event.py::NinjaScroll.SHIVS`).
pub const NINJA_SCROLL_SHIVS: i32 = 3;

/// `ScreamingFlagon::DAMAGE` — round-end self-damage
/// (`shop_event.py::ScreamingFlagon.DAMAGE`).
pub const SCREAMING_FLAGON_DAMAGE: i32 = 20;

/// `SlingOfCourage::STRENGTH` — elite-only round-1 Strength
/// (`shop_event.py::SlingOfCourage.STRENGTH`).
pub const SLING_OF_COURAGE_STRENGTH: i32 = 2;

/// `TheAbacus::BLOCK` — block gained on shuffle
/// (`shop_event.py::TheAbacus.BLOCK`).
pub const THE_ABACUS_BLOCK: i32 = 6;

/// `RunicCapacitor::ORB_SLOTS` — extra Defect orb slots
/// (`shop_event.py::RunicCapacitor.ORB_SLOTS`).
pub const RUNIC_CAPACITOR_ORB_SLOTS: i32 = 3;

/// `AlchemicalCoffer::POTION_SLOTS` — bonus potion slots on obtain
/// (`shop_event.py::AlchemicalCoffer.POTION_SLOTS`).
pub const ALCHEMICAL_COFFER_POTION_SLOTS: i32 = 4;

/// `FurCoat::COMBATS` — number of combat rooms FurCoat marks on the
/// obtain act's map (`shop_event.py::FurCoat.COMBATS`). Bead
/// `sts2-rs-eboqq`.
pub const FUR_COAT_COMBATS: usize = 7;

/// `FurCoat::DynamicVars["Combats"]` for the build: [`FUR_COAT_COMBATS`] in
/// v0.107.1.
///
/// v0.111.0: `DynamicVar("Combats", 7m)` -> `8m` (decomp FurCoat.cs:64). The
/// marking RNG is `new Rng(Owner, Id, 0uL)` (FurCoat.cs:93), which is
/// `RunRngSet::named` in the run's build; the new `IsAffectedByFurCoat`
/// history flag (FurCoat.cs:140) is run-history bookkeeping only.
#[must_use]
pub const fn fur_coat_combats(version: sts2_core::GameVersion) -> usize {
    match version {
        sts2_core::GameVersion::V0_107_1 => FUR_COAT_COMBATS,
        sts2_core::GameVersion::V0_111_0 | sts2_core::GameVersion::V0_113_0 => 8,
    }
}

/// `BlessedAntler::ENERGY` / `DAZED_COUNT` — +1 max energy with 3 Dazed
/// shuffled into draw round 1 (`shop_event.py::BlessedAntler`).
pub const BLESSED_ANTLER_ENERGY: i32 = 1;
pub const BLESSED_ANTLER_DAZED_COUNT: i32 = 3;

/// `BloodSoakedRose::ENERGY` — +1 max energy
/// (`shop_event.py::BloodSoakedRose.ENERGY`).
pub const BLOOD_SOAKED_ROSE_ENERGY: i32 = 1;

/// `CursedPearl::GOLD` — gold gained when the Greed curse is added
/// (`shop_event.py::CursedPearl.GOLD`).
pub const CURSED_PEARL_GOLD: i32 = 333;

/// `Ectoplasm::ENERGY` — +1 max energy, prevents all gold gain
/// (`shop_event.py::Ectoplasm.ENERGY`).
pub const ECTOPLASM_ENERGY: i32 = 1;

/// `Sai::BLOCK` — per-turn block gained
/// (`shop_event.py::Sai.BLOCK`).
pub const SAI_BLOCK: i32 = 7;

/// `SneckoEye::EXTRA_DRAW` — extra cards drawn per turn (Confused on
/// combat start) (`shop_event.py::SneckoEye.EXTRA_DRAW`).
pub const SNECKO_EYE_EXTRA_DRAW: i32 = 2;

/// `Sozu::ENERGY` — +1 max energy, blocks potion procurement
/// (`shop_event.py::Sozu.ENERGY`).
pub const SOZU_ENERGY: i32 = 1;

/// `SpikedGauntlets::ENERGY` — +1 max energy, Power cards cost +1
/// (`shop_event.py::SpikedGauntlets.ENERGY`).
pub const SPIKED_GAUNTLETS_ENERGY: i32 = 1;

/// `IronClub::CARD_THRESHOLD` — every Nth player card play triggers a
/// `draw_cards(1)` (`shop_event.py::IronClub.CARD_THRESHOLD`). Bead
/// `sts2-rs-70anj.1`.
pub const IRON_CLUB_CARD_THRESHOLD: i32 = 4;

/// `ToastyMittens::STRENGTH` — per-player-turn Strength gain
/// (`shop_event.py::ToastyMittens.STRENGTH`). Bead `sts2-rs-70anj.1`.
pub const TOASTY_MITTENS_STRENGTH: i32 = 1;

// ---------------------------------------------------------------------------
// Typed-channel listener bodies
// ---------------------------------------------------------------------------

/// `RunicCapacitor::after_side_turn_start` — `if side == PLAYER and
/// combat.round_number == 1: combat.add_orb_slots(owner, 3)` (Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::RunicCapacitor`,
/// `ORB_SLOTS = 3`, pool=DEFECT). Round 1 only. Mirrors the
/// VeryHotCocoa / SymbioticVirus round-1 player-gate pattern; the
/// underlying engine call routes to
/// [`sts2_core::orbs::OrbSlots::add_slots`].
fn listener_runic_capacitor(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    if !state.relics.owns(RelicKind::RunicCapacitor) {
        return EffectResult::NotImplemented;
    }
    let slots = usize::try_from(RUNIC_CAPACITOR_ORB_SLOTS).unwrap_or(0);
    state.player.orbs.add_slots(slots);
    EffectResult::Resolved
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// The id list covers exactly 122 entries — 30 SHOP + 92 ANCIENT in
    /// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py`.
    #[test]
    fn pool_has_one_hundred_twenty_two_entries() {
        assert_eq!(SHOP_BOSS_RELIC_IDS.len(), 122);
    }

    /// Every `RelicId` in the list is distinct. Catches accidental
    /// duplicate paste-edits inside [`SHOP_BOSS_RELIC_IDS`].
    #[test]
    fn pool_entries_are_unique_by_relic_id() {
        let mut ids: Vec<RelicId> = SHOP_BOSS_RELIC_IDS.to_vec();
        ids.sort();
        let before = ids.len();
        ids.dedup();
        assert_eq!(
            before,
            ids.len(),
            "duplicate RelicId in SHOP_BOSS_RELIC_IDS"
        );
    }

    /// `LargeCapsule::after_obtained` obtains `LARGE_CAPSULE_RELICS = 2`
    /// relics DIRECTLY (C# `RelicCmd.Obtain` — no pick) and then adds the
    /// character's basic Strike + Defend to the deck. Obtain-before-add is
    /// load-bearing for the egg-relic upgrade-on-add hook (PARITY36).
    #[test]
    fn listener_large_capsule_obtains_two_relics_and_adds_strike_defend() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
        rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
        rs.relics.push(RelicId::LargeCapsule);
        rs.last_obtained_relic = Some(RelicId::LargeCapsule);
        let relics_before = rs.relics.len();
        let deck_before = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // 2 relics obtained directly (not offered as pickable rewards).
        assert_eq!(rs.relic_rewards.len(), 0, "obtained directly, not offered");
        assert_eq!(rs.relics.len(), relics_before + LARGE_CAPSULE_RELICS);
        // Strike + Defend added to the deck.
        assert_eq!(rs.master_deck.len(), deck_before + 2);
    }

    /// Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 1):
    /// `JewelryBox::after_obtained` adds 1 unupgraded `Apotheosis` to
    /// the run-side master deck. Pins the deck-push path AND the
    /// ownership-guard no-op shape.
    #[test]
    fn listener_jewelry_box_adds_apotheosis_to_master_deck_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        // Without ownership: dispatcher fires the registered listener
        // which ownership-gates to no-op; master_deck stays untouched.
        let mut rs_noop = RunState::new();
        let pre_len = rs_noop.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs_noop);
        assert_eq!(rs_noop.master_deck.len(), pre_len);

        // With ownership: master_deck grows by 1 with the Apotheosis
        // POD (unupgraded). Default RunState has an empty master_deck.
        let mut rs = RunState::new();
        rs.relics.push(RelicId::JewelryBox);
        rs.last_obtained_relic = Some(RelicId::JewelryBox);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), 1);
        let added = &rs.master_deck[0];
        assert_eq!(added.card_id, sts2_core::cards::CardId::Apotheosis);
        assert!(!added.upgraded, "Python pin adds the un-upgraded card");
    }

    /// Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 1):
    /// `TanxsWhistle::after_obtained` adds 1 unupgraded `Whistle` to
    /// the run-side master deck.
    #[test]
    fn listener_tanxs_whistle_adds_whistle_to_master_deck_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::TanxsWhistle);
        rs.last_obtained_relic = Some(RelicId::TanxsWhistle);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), 1);
        let added = &rs.master_deck[0];
        assert_eq!(added.card_id, sts2_core::cards::CardId::Whistle);
        assert!(!added.upgraded);
    }

    /// A relic's deck card carries the card's numbers in the run's game
    /// build. The deck card is copied into combat as it stands
    /// (`run_bridge::enter_combat`), so the v0.107.1 numbers would be played
    /// on v0.111.0: Whistle costs 3 -> 2 (`Whistle.cs:23`), Relax blocks
    /// 15 -> 16 (`Relax.cs` `BlockVar`). Found in the cynilyn fork.
    #[test]
    fn relic_deck_cards_carry_the_game_builds_numbers() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        use sts2_core::GameVersion;
        crate::relics::dispatch::ensure_installed();

        for (version, whistle_cost, relax_block) in [
            (GameVersion::V0_107_1, 3, 15),
            (GameVersion::V0_111_0, 2, 16),
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            rs.relics.push(RelicId::TanxsWhistle);
            rs.last_obtained_relic = Some(RelicId::TanxsWhistle);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(rs.master_deck.len(), 1, "{version:?}");
            assert_eq!(rs.master_deck[0].card_id, CardId::Whistle, "{version:?}");
            assert_eq!(rs.master_deck[0].local_cost(), whistle_cost, "{version:?}");
            assert_eq!(rs.master_deck[0].base_damage, 33, "{version:?}");

            let mut rs = RunState::new();
            rs.game_version = version;
            rs.relics.push(RelicId::PaelsHorn);
            rs.last_obtained_relic = Some(RelicId::PaelsHorn);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(rs.master_deck.len(), PAELS_HORN_CARDS, "{version:?}");
            for card in &rs.master_deck {
                assert_eq!(card.card_id, CardId::Relax, "{version:?}");
                assert_eq!(card.base_block, relax_block, "{version:?}");
            }
        }
    }

    /// Bead `sts2-rs-70anj.4` (Wave 4 body-fill, batch 2):
    /// `DragonFruit::on_gold_gained` bumps `rs.max_hp` by 1 per
    /// positive gold gain when owned. Negative / zero gold amounts
    /// are no-ops (Python `if amount > 0` guard). Non-ownership is
    /// also no-op (the listener fires on the dispatch walk over all
    /// registered relics — only the owned one mutates run state).
    #[test]
    fn listener_dragon_fruit_max_hp_grows_per_positive_gold_gain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        // Owned + positive gain → +1 max_hp per gain.
        let mut rs = RunState::new();
        rs.relics.push(RelicId::DragonFruit);
        let pre_max_hp = rs.max_hp;
        let _ = sts2_core::run_effects::dispatch_on_gold_gained(&mut rs, 5);
        assert_eq!(rs.max_hp, pre_max_hp + DRAGON_FRUIT_MAX_HP);
        // Second gain bumps again.
        let _ = sts2_core::run_effects::dispatch_on_gold_gained(&mut rs, 12);
        assert_eq!(rs.max_hp, pre_max_hp + 2 * DRAGON_FRUIT_MAX_HP);

        // Owned + zero or negative gain → no-op (Python pin guards).
        let mut rs0 = RunState::new();
        rs0.relics.push(RelicId::DragonFruit);
        let pre = rs0.max_hp;
        let _ = sts2_core::run_effects::dispatch_on_gold_gained(&mut rs0, 0);
        let _ = sts2_core::run_effects::dispatch_on_gold_gained(&mut rs0, -3);
        assert_eq!(rs0.max_hp, pre);

        // Not owned → no-op.
        let mut rs_noown = RunState::new();
        let pre_noown = rs_noown.max_hp;
        let _ = sts2_core::run_effects::dispatch_on_gold_gained(&mut rs_noown, 100);
        assert_eq!(rs_noown.max_hp, pre_noown);
    }

    /// Bead `sts2-rs-70anj.3` (Wave 3 starter_relic_swap primitive):
    /// `TouchOfOrobas::after_obtained` swaps each of the 5 starter
    /// relics in `rs.relics` to its upgraded variant when obtained.
    /// Mirrors Python pin
    /// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::TouchOfOrobas`.
    #[test]
    fn listener_touch_of_orobas_upgrades_each_starter_relic() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let cases = [
            (RelicId::BurningBlood, RelicId::BlackBlood),
            (RelicId::RingOfTheSnake, RelicId::RingOfTheDrake),
            (RelicId::CrackedCore, RelicId::InfusedCore),
            (RelicId::BoundPhylactery, RelicId::PhylacteryUnbound),
            (RelicId::DivineRight, RelicId::DivineDestiny),
            // v0.113.0 `TouchOfOrobas.cs:52-53`.
            (RelicId::SongBow, RelicId::Storyteller),
            // No `RefinementUpgrades` entry: `GetUpgradedStarterRelic`
            // falls back to Circlet (`TouchOfOrobas.cs:124-131`).
            (RelicId::BlackBlood, RelicId::Circlet),
        ];

        for (starter, upgraded) in cases {
            let mut rs = RunState::new();
            rs.relics.push(starter);
            rs.relics.push(RelicId::TouchOfOrobas);
            rs.last_obtained_relic = Some(RelicId::TouchOfOrobas);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert!(
                rs.relics.contains(&upgraded),
                "TouchOfOrobas must upgrade {starter:?} → {upgraded:?}"
            );
            assert!(
                !rs.relics.contains(&starter),
                "Original starter {starter:?} must be replaced"
            );
        }
    }

    /// No-op when TouchOfOrobas is owned but no starter relic is
    /// present (degenerate case — Python skips the swap silently).
    #[test]
    fn listener_touch_of_orobas_is_noop_without_starter() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::TouchOfOrobas);
        let before = rs.relics.clone();
        rs.last_obtained_relic = Some(RelicId::TouchOfOrobas);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relics, before);
    }

    /// Every shop/boss `RelicId` carries an upper-snake-case Python
    /// name. Locks the parity-output naming for the future exporter.
    #[test]
    fn every_shop_boss_relic_has_python_name() {
        for &id in SHOP_BOSS_RELIC_IDS {
            let name = id.to_python_name();
            assert!(!name.is_empty(), "empty python_name for {id:?}");
            assert!(
                name.bytes()
                    .all(|b| b == b'_' || b.is_ascii_uppercase() || b.is_ascii_digit()),
                "non upper-snake-case python_name for {id:?}: {name:?}"
            );
        }
    }

    // --- Deep pin-tests for the headline boss/shop relics ----------------
    //
    // The bead spec asked for deep tests on Snecko Eye / Dead Branch /
    // Pandora's Box. `DeadBranch` is absent from the `01c094d` fork
    // (see module-level note), so the third deep test covers
    // `Brimstone` — the closest combat-math analogue (per-turn
    // Strength on both sides). Additional pin-tests cover Sozu, Sai,
    // BlackStar, MembershipCard, MysticLighter, RunicPyramid,
    // Ectoplasm, Bread, AlchemicalCoffer, CursedPearl, NinjaScroll,
    // SpikedGauntlets, BloodSoakedRose, ChemicalX, LeesWaffle,
    // DragonFruit, ScreamingFlagon — pinning the load-bearing Python
    // constants so the integration bead can drop in real logic
    // without retyping numbers.

    /// `SneckoEye` — pins `EXTRA_DRAW = 2` and locks the Confused
    /// pre-combat hook. This is the headline boss relic from the bead
    /// spec (`shop_event.py::SneckoEye`).
    #[test]
    fn snecko_eye_deep_pin() {
        assert_eq!(SNECKO_EYE_EXTRA_DRAW, 2);
        assert_eq!(RelicId::SneckoEye.to_python_name(), "SNECKO_EYE");
    }

    /// `PandorasBox` — pins the transform-all-basic-Strikes/Defends
    /// hook on `after_obtained`. The other headline boss relic from
    /// the bead spec (`shop_event.py::PandorasBox`).
    #[test]
    fn pandoras_box_deep_pin() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::PandorasBox.to_python_name(), "PANDORAS_BOX");
    }

    /// `Brimstone` — pins `SELF_STRENGTH = 2` and `ENEMY_STRENGTH = 1`
    /// (per-turn Strength on both sides). The closest combat-math
    /// analogue to Dead Branch in this fork — directly mutates damage
    /// pipeline every turn (`shop_event.py::Brimstone`).
    #[test]
    fn brimstone_deep_pin() {
        assert_eq!(BRIMSTONE_SELF_STRENGTH, 2);
        assert_eq!(BRIMSTONE_ENEMY_STRENGTH, 1);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Brimstone.to_python_name(), "BRIMSTONE");
    }

    /// `Sozu` — pins `ENERGY = 1` and the `should_procure_potion`
    /// override semantics (one-shot block at the registry layer).
    #[test]
    fn sozu_deep_pin() {
        assert_eq!(SOZU_ENERGY, 1);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Sozu.to_python_name(), "SOZU");
    }

    /// `Sai` — pins `BLOCK = 7` (per-turn block) for the
    /// `after_side_turn_start` hook.
    #[test]
    fn sai_deep_pin() {
        assert_eq!(SAI_BLOCK, 7);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Sai.to_python_name(), "SAI");
    }

    /// `RunicPyramid` — locks the `should_flush` hook (retain all
    /// cards every turn — Pyramid's signature effect).
    #[test]
    fn runic_pyramid_deep_pin() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::RunicPyramid.to_python_name(), "RUNIC_PYRAMID");
    }

    /// `Ectoplasm` — pins `ENERGY = 1` and the gold-gain override
    /// (always returns `False` to block all gold gain).
    #[test]
    fn ectoplasm_deep_pin() {
        assert_eq!(ECTOPLASM_ENERGY, 1);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Ectoplasm.to_python_name(), "ECTOPLASM");
    }

    /// `Bread` — pins +1 max energy / -2 round-1 energy
    /// (`shop_event.py::Bread`).
    #[test]
    fn bread_deep_pin() {
        assert_eq!(BREAD_GAIN_ENERGY, 1);
        assert_eq!(BREAD_LOSE_ENERGY, 2);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Bread.to_python_name(), "BREAD");
    }

    /// `MembershipCard` — pins `DISCOUNT = 50` for the
    /// `modify_merchant_price` hook.
    #[test]
    fn membership_card_deep_pin() {
        assert_eq!(MEMBERSHIP_CARD_DISCOUNT, 50);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::MembershipCard.to_python_name(), "MEMBERSHIP_CARD");
    }

    /// `MysticLighter` — pins `EXTRA_DAMAGE = 9` for the
    /// `modify_damage_additive` hook.
    #[test]
    fn mystic_lighter_deep_pin() {
        assert_eq!(MYSTIC_LIGHTER_EXTRA_DAMAGE, 9);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::MysticLighter.to_python_name(), "MYSTIC_LIGHTER");
    }

    /// `NinjaScroll` — pins `SHIVS = 3` for the round-1
    /// `before_side_turn_start` hook (Silent pool).
    #[test]
    fn ninja_scroll_deep_pin() {
        assert_eq!(NINJA_SCROLL_SHIVS, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::NinjaScroll.to_python_name(), "NINJA_SCROLL");
    }

    /// `AlchemicalCoffer` — pins `POTION_SLOTS = 4` for the
    /// `after_obtained` hook.
    #[test]
    fn alchemical_coffer_deep_pin() {
        assert_eq!(ALCHEMICAL_COFFER_POTION_SLOTS, 4);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(
            RelicId::AlchemicalCoffer.to_python_name(),
            "ALCHEMICAL_COFFER"
        );
    }

    /// C# `AlchemicalCoffer.AfterObtained` procures into slot
    /// `originalSlotCount + i`, so an empty old slot stays empty (fuzz
    /// v0.107.1 seed 1123).
    #[test]
    fn alchemical_coffer_fills_only_the_new_slots() {
        let mut rs = RunState::new();
        assert!(rs.add_potion_to_belt(sts2_core::potions::PotionKind::FirePotion));
        rs.last_obtained_relic = Some(RelicId::AlchemicalCoffer);
        assert_eq!(listener_alchemical_coffer(&mut rs), EffectResult::Resolved);
        assert_eq!(rs.potions.len(), 5);
        assert_eq!(rs.potion_slot_indices, vec![0, 3, 4, 5, 6]);
    }

    /// `CursedPearl` — pins `GOLD = 333` (and the Greed-curse
    /// `after_obtained` semantics).
    #[test]
    fn cursed_pearl_deep_pin() {
        assert_eq!(CURSED_PEARL_GOLD, 333);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::CursedPearl.to_python_name(), "CURSED_PEARL");
    }

    /// `LeesWaffle` — pins `MAX_HP = 7` for the obtain-time max-HP +
    /// heal-to-full hook.
    #[test]
    fn lees_waffle_deep_pin() {
        assert_eq!(LEES_WAFFLE_MAX_HP, 7);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::LeesWaffle.to_python_name(), "LEES_WAFFLE");
    }

    /// `ChemicalX` — pins `INCREASE = 2` for the X-cost card hook.
    #[test]
    fn chemical_x_deep_pin() {
        assert_eq!(CHEMICAL_X_INCREASE, 2);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::ChemicalX.to_python_name(), "CHEMICAL_X");
    }

    /// `ScreamingFlagon` — pins `DAMAGE = 20` for the
    /// `before_turn_end` self-damage hook.
    #[test]
    fn screaming_flagon_deep_pin() {
        assert_eq!(SCREAMING_FLAGON_DAMAGE, 20);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(
            RelicId::ScreamingFlagon.to_python_name(),
            "SCREAMING_FLAGON"
        );
    }

    /// `BlackStar` — locks the extra-relic-after-elite hook
    /// (`modify_rewards`). A high-impact reward-stream relic.
    #[test]
    fn black_star_deep_pin() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::BlackStar.to_python_name(), "BLACK_STAR");
    }

    /// `BeltBuckle` — pins `DEXTERITY = 2` (the +2/-2 dex on
    /// potion-procurement state — load-bearing combat-math toggle).
    #[test]
    fn belt_buckle_deep_pin() {
        assert_eq!(BELT_BUCKLE_DEXTERITY, 2);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::BeltBuckle.to_python_name(), "BELT_BUCKLE");
    }

    // -----------------------------------------------------------------
    // BeltBuckle listener tests — bead `sts2-rs-xu8d6`
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn belt_buckle_setup(
        state: &mut CombatState,
        equip: bool,
        potions: &[sts2_core::potions::PotionKind],
    ) {
        use std::num::NonZeroU32;
        use sts2_core::Relic;
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.player.potions.clear();
        for &p in potions {
            state.player.potions.push(p);
        }
        if equip {
            state.relics.push(Relic {
                id: NonZeroU32::new(1).expect("non-zero"),
                kind: RelicKind::BeltBuckle,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
    }

    #[cfg(test)]
    fn belt_buckle_player_dexterity(state: &CombatState) -> i32 {
        let Some(uid) = state.player.creature_uid else {
            return 0;
        };
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .map_or(0, |c| {
                c.powers
                    .iter()
                    .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                    .filter(|p| p.kind == PowerKind::Dexterity)
                    .map(|p| p.amount)
                    .sum()
            })
    }

    /// **BeltBuckle #1** — combat start with no potions grants
    /// +`BELT_BUCKLE_DEXTERITY` Dexterity and latches.
    #[test]
    fn belt_buckle_grants_dexterity_when_potionless_at_combat_start() {
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, true, &[]);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(
            belt_buckle_player_dexterity(&state),
            BELT_BUCKLE_DEXTERITY,
            "potionless combat start grants +BELT_BUCKLE_DEXTERITY Dex",
        );
        // Latch set — a second apply attempt must not double the bonus.
        let _ = belt_buckle_apply_dexterity(&mut state);
        assert_eq!(
            belt_buckle_player_dexterity(&state),
            BELT_BUCKLE_DEXTERITY,
            "latch prevents double-apply",
        );
    }

    /// **BeltBuckle #2** — combat start while holding a potion is inert.
    #[test]
    fn belt_buckle_inert_when_holding_potion_at_combat_start() {
        use sts2_core::potions::PotionKind;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, true, &[PotionKind::BlockPotion]);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(
            belt_buckle_player_dexterity(&state),
            0,
            "holding a potion suppresses the Dexterity grant",
        );
    }

    /// **BeltBuckle #3** — using the last potion (leaving the player
    /// potionless) grants the Dexterity mid-combat.
    #[test]
    fn belt_buckle_grants_dexterity_after_using_last_potion() {
        use sts2_core::potions::PotionKind;
        use sts2_core::relic_effects::{dispatch_after_potion_used, dispatch_before_combat_start};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, true, &[PotionKind::BlockPotion]);
        // Start with a potion: combat-start grant is suppressed.
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(belt_buckle_player_dexterity(&state), 0);
        // Simulate using the potion (inventory now empty), then fire the hook.
        state.player.potions.clear();
        let _ = dispatch_after_potion_used(&mut state, PotionKind::BlockPotion, None);
        assert_eq!(
            belt_buckle_player_dexterity(&state),
            BELT_BUCKLE_DEXTERITY,
            "using the last potion grants the Dexterity",
        );
    }

    /// **BeltBuckle #5** — procuring a potion mid-combat retracts the
    /// Dexterity. C# `BeltBuckle.AfterPotionProcured` (v0.111.0
    /// `BeltBuckle.cs`, same in v0.107.1): `if (IsInProgress &&
    /// Owner.Potions.Any()) RemoveDexterity()` applies `-2` Dexterity, which
    /// takes the power to 0 and removes it (Ironclad seed 2148 act 3,
    /// Alchemize). `PotionCmd.TryToProcure` fires the hook after a successful
    /// add (`PotionCmd.cs:39-46`).
    #[test]
    fn belt_buckle_retracts_dexterity_when_a_potion_is_procured() {
        use sts2_core::potions::PotionKind;
        use sts2_core::relic_effects::{dispatch_after_potion_used, dispatch_before_combat_start};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, true, &[]);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(belt_buckle_player_dexterity(&state), BELT_BUCKLE_DEXTERITY);
        assert!(state.try_procure_potion(PotionKind::BlockPotion));
        assert_eq!(belt_buckle_player_dexterity(&state), 0, "procure retracts");
        assert!(
            !state.powers.iter().any(|p| p.kind == PowerKind::Dexterity),
            "Dexterity 2 - 2 = 0 removes the power"
        );
        // Latch cleared: drinking the potion re-applies (`ApplyDexterity`).
        state.player.potions.clear();
        let _ = dispatch_after_potion_used(&mut state, PotionKind::BlockPotion, None);
        assert_eq!(belt_buckle_player_dexterity(&state), BELT_BUCKLE_DEXTERITY);
        // A second procure retracts again; one more is a latch no-op.
        assert!(state.try_procure_potion(PotionKind::BlockPotion));
        assert!(state.try_procure_potion(PotionKind::BlockPotion));
        assert_eq!(belt_buckle_player_dexterity(&state), 0);
    }

    /// **BeltBuckle #4** — no-op without ownership.
    #[test]
    fn belt_buckle_inert_without_ownership() {
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, false, &[]);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(
            belt_buckle_player_dexterity(&state),
            0,
            "no relic, no Dexterity",
        );
    }

    /// Ironclad seed 4170 act 3 (v0.111.0): Petrified Toad obtained BEFORE
    /// Delicate Frond. C# `Hook.BeforeCombatStart` (Hook.cs:455-467, both
    /// builds) walks every `BeforeCombatStart` (Frond fills the empty belt,
    /// `DelicateFrond.cs:12-22`), then every `BeforeCombatStartLate`
    /// (`PetrifiedToad.cs:16`), where the Rock no longer fits. Rust ran the
    /// Toad in the first walk, so the Rock took a slot and Frond rolled one
    /// potion fewer (`CombatPotionGeneration` 4 draws instead of 6).
    #[test]
    fn petrified_toad_procures_after_delicate_frond_whatever_the_obtain_order() {
        use std::num::NonZeroU32;
        use sts2_core::potions::{CharacterId, PotionKind};
        use sts2_core::Relic;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        belt_buckle_setup(&mut state, false, &[]);
        state.player.character = Some(CharacterId::Ironclad);
        state.player.potion_slots = 3;
        for (i, kind) in [RelicKind::PetrifiedToad, RelicKind::DelicateFrond]
            .into_iter()
            .enumerate()
        {
            state.relics.push(Relic {
                id: NonZeroU32::new(u32::try_from(i).unwrap() + 1).expect("non-zero"),
                kind,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
        let cpg_before = state.run_rng.combat_potion_generation.counter();
        let _ = sts2_core::combat_loop::start_combat(&mut state);
        assert_eq!(state.player.potions.len(), 3, "Frond fills all 3 slots");
        assert!(
            !state.player.potions.contains(&PotionKind::PotionShapedRock),
            "the Late Toad finds the belt full: no Rock"
        );
        assert_eq!(
            state.run_rng.combat_potion_generation.counter() - cpg_before,
            6,
            "3 Frond potions x 2 draws"
        );
    }

    /// Toad's Late Rock and Belt Buckle: C# `BeltBuckle.BeforeCombatStart`
    /// (normal walk) sees an empty belt and applies +2 Dexterity, then the
    /// Late Rock fires `AfterPotionProcured`, which removes it again
    /// (`BeltBuckle.cs`, both builds) — Dexterity 0, latch cleared, whichever
    /// relic was obtained first.
    #[test]
    fn petrified_toad_late_rock_retracts_belt_buckle_dexterity() {
        use std::num::NonZeroU32;
        use sts2_core::potions::PotionKind;
        use sts2_core::Relic;
        crate::relics::dispatch::ensure_installed();
        for order in [
            [RelicKind::PetrifiedToad, RelicKind::BeltBuckle],
            [RelicKind::BeltBuckle, RelicKind::PetrifiedToad],
        ] {
            let mut state = CombatState::default();
            belt_buckle_setup(&mut state, false, &[]);
            state.player.potion_slots = 3;
            for (i, kind) in order.into_iter().enumerate() {
                state.relics.push(Relic {
                    id: NonZeroU32::new(u32::try_from(i).unwrap() + 1).expect("non-zero"),
                    kind,
                    counter: 0,
                    is_wax: false,
                    is_melted: false,
                });
            }
            let _ = sts2_core::combat_loop::start_combat(&mut state);
            assert_eq!(
                state.player.potions.as_slice(),
                [PotionKind::PotionShapedRock].as_slice(),
                "{order:?}"
            );
            assert_eq!(belt_buckle_player_dexterity(&state), 0, "{order:?}");
            let buckle = state
                .relics
                .iter()
                .find(|r| r.kind == RelicKind::BeltBuckle)
                .expect("buckle");
            assert_eq!(buckle.counter, 0, "latch cleared by the procure: {order:?}");
        }
    }

    // -----------------------------------------------------------------
    // PaelsLegion listener tests — bead `sts2-rs-ppabi`
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn paels_legion_counter(state: &CombatState) -> i32 {
        state
            .relics
            .iter()
            .find(|r| !r.is_melted && r.kind == RelicKind::PaelsLegion)
            .expect("PaelsLegion present")
            .counter
    }

    #[cfg(test)]
    fn paels_card_event_ctx() -> CardEventCtx {
        CardEventCtx {
            card_id: sts2_core::cards::CardId::DefendIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        }
    }

    /// `PaelsLegion` doubles the first card-granted block, then enters a
    /// `PAELS_LEGION_TURNS`-turn cooldown, then re-arms — full cycle.
    #[test]
    fn paels_legion_doubles_card_block_then_cooldowns_then_rearms() {
        use std::num::NonZeroU32;
        use sts2_core::card_effects::gain_block_player;
        use sts2_core::relic_effects::{
            dispatch_after_card_played, dispatch_after_side_turn_start,
        };
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PaelsLegion,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        // A card "in play" so block grants are card-sourced.
        state
            .player
            .play
            .push(sts2_core::CardUid(NonZeroU32::new(1).expect("non-zero")));

        // 1) Armed: 5 base block doubled to 10; cooldown pending (-1).
        gain_block_player(&mut state, 5);
        assert_eq!(state.player.block, 10, "card block doubled while armed");
        assert_eq!(paels_legion_counter(&state), -1, "cooldown pending");

        // 2) Card-play resolves → cooldown starts.
        let _ = dispatch_after_card_played(&mut state, &paels_card_event_ctx());
        assert_eq!(
            paels_legion_counter(&state),
            PAELS_LEGION_TURNS,
            "cooldown armed to Turns after card resolves"
        );

        // 3) On cooldown: next card block is NOT doubled (10 + 5 = 15).
        gain_block_player(&mut state, 5);
        assert_eq!(state.player.block, 15, "no doubling while on cooldown");
        assert_eq!(paels_legion_counter(&state), PAELS_LEGION_TURNS);

        // 4) Two player turn-starts decrement the cooldown to 0 (re-armed).
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(paels_legion_counter(&state), 0, "re-armed after 2 turns");

        // 5) Armed again: doubled (15 + 10 = 25).
        gain_block_player(&mut state, 5);
        assert_eq!(state.player.block, 25, "doubled again after re-arm");
    }

    /// C# `PaelsLegion.AfterCardPlayed` starts the cooldown only for the
    /// `CardPlay` it latched in `AfterModifyingBlockAmount`. A different card
    /// played inside that play (Hellraiser auto-playing a Strike that Dark
    /// Embrace drew mid-Second Wind) finishes first and must not start it,
    /// so Second Wind's later gains stay doubled (loadout Ironclad 7070
    /// v0.107.1: cs 5 x 16 = 80, rs 56).
    #[test]
    fn paels_legion_nested_play_does_not_end_the_latched_play() {
        use std::num::NonZeroU32;
        use sts2_core::card_effects::gain_block_player;
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PaelsLegion,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let outer = sts2_core::CardUid(NonZeroU32::new(1).expect("non-zero"));
        let nested = sts2_core::CardUid(NonZeroU32::new(2).expect("non-zero"));
        state.player.play.push(outer);

        gain_block_player(&mut state, 8);
        assert_eq!(state.player.block, 16, "first gain doubled");

        // The nested auto-play resolves and fires its own AfterCardPlayed.
        let nested_ctx = CardEventCtx {
            card_uid: Some(nested),
            ..paels_card_event_ctx()
        };
        let _ = dispatch_after_card_played(&mut state, &nested_ctx);

        gain_block_player(&mut state, 8);
        assert_eq!(
            state.player.block, 32,
            "the outer play's next gain is still doubled"
        );

        let outer_ctx = CardEventCtx {
            card_uid: Some(outer),
            ..paels_card_event_ctx()
        };
        let _ = dispatch_after_card_played(&mut state, &outer_ctx);
        assert_eq!(paels_legion_counter(&state), PAELS_LEGION_TURNS);
    }

    /// `PaelsLegion` does not double non-card (system) block — `card_source`
    /// is `None` for turn-start / relic block grants.
    #[test]
    fn paels_legion_ignores_non_card_block() {
        use std::num::NonZeroU32;
        use sts2_core::card_effects::gain_block_player;
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PaelsLegion,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        // Empty play pile → block grant is non-card (card_source = None).
        gain_block_player(&mut state, 5);
        assert_eq!(state.player.block, 5, "non-card block is not doubled");
        assert_eq!(
            paels_legion_counter(&state),
            0,
            "non-card block does not arm the cooldown"
        );
    }

    /// `PaelsLegion::before_combat_start` summons the pet and resets the
    /// cooldown to armed.
    #[test]
    fn paels_legion_summons_pet_and_resets_cooldown() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PaelsLegion,
            counter: 9, // stale cooldown from a prior combat
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(paels_legion_counter(&state), 0, "cooldown reset to armed");
        let pets = state
            .creatures
            .iter()
            .filter(|c| c.species == Some(sts2_core::MonsterKind::PaelsLegion) && c.current_hp > 0)
            .count();
        assert_eq!(pets, 1, "PaelsLegion pet summoned");
    }

    /// `DragonFruit` — pins `MAX_HP = 1` per gold-gain (slow-roll
    /// run-scoped HP growth — pins the `on_gold_gained` hook name).
    #[test]
    fn dragon_fruit_deep_pin() {
        assert_eq!(DRAGON_FRUIT_MAX_HP, 1);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::DragonFruit.to_python_name(), "DRAGON_FRUIT");
    }

    /// Spot-check the load-bearing public constants exist with the
    /// values Python's `shop_event.py` pins. Catches accidental edits
    /// to the `pub const` block above.
    #[test]
    fn public_constants_match_python_pins() {
        assert_eq!(BELT_BUCKLE_DEXTERITY, 2);
        assert_eq!(BREAD_GAIN_ENERGY, 1);
        assert_eq!(BREAD_LOSE_ENERGY, 2);
        assert_eq!(BRIMSTONE_SELF_STRENGTH, 2);
        assert_eq!(BRIMSTONE_ENEMY_STRENGTH, 1);
        assert_eq!(CAULDRON_POTIONS, 5);
        assert_eq!(CHEMICAL_X_INCREASE, 2);
        assert_eq!(DRAGON_FRUIT_MAX_HP, 1);
        assert_eq!(LEES_WAFFLE_MAX_HP, 7);
        assert_eq!(MEMBERSHIP_CARD_DISCOUNT, 50);
        assert_eq!(MYSTIC_LIGHTER_EXTRA_DAMAGE, 9);
        assert_eq!(NINJA_SCROLL_SHIVS, 3);
        assert_eq!(SCREAMING_FLAGON_DAMAGE, 20);
        assert_eq!(SLING_OF_COURAGE_STRENGTH, 2);
        assert_eq!(THE_ABACUS_BLOCK, 6);
        assert_eq!(RUNIC_CAPACITOR_ORB_SLOTS, 3);
        assert_eq!(ALCHEMICAL_COFFER_POTION_SLOTS, 4);
        assert_eq!(BLESSED_ANTLER_ENERGY, 1);
        assert_eq!(BLESSED_ANTLER_DAZED_COUNT, 3);
        assert_eq!(BLOOD_SOAKED_ROSE_ENERGY, 1);
        assert_eq!(CURSED_PEARL_GOLD, 333);
        assert_eq!(ECTOPLASM_ENERGY, 1);
        assert_eq!(SAI_BLOCK, 7);
        assert_eq!(SNECKO_EYE_EXTRA_DRAW, 2);
        assert_eq!(SOZU_ENERGY, 1);
        assert_eq!(SPIKED_GAUNTLETS_ENERGY, 1);
    }

    /// `Brimstone` behavioural pin (bead `sts2-rs-o39p` wave 0):
    /// the `after_side_turn_start` listener applies STRENGTH(+2) to
    /// the player and STRENGTH(+1) to every alive enemy on each
    /// player turn — no round gate. Mirrors Python
    /// `shop_event.py::Brimstone.before_side_turn_start`.
    #[test]
    fn brimstone_applies_strength_to_player_and_every_alive_enemy() {
        use smallvec::SmallVec;
        use std::num::NonZeroU32;
        use sts2_core::card_effects::EffectResult;
        use sts2_core::relic_effects::{
            dispatch_after_side_turn_start, dispatch_before_side_turn_start,
        };
        use sts2_core::{
            Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, PowerKind, Relic,
            RelicKind, Side,
        };

        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();

        // Player creature row + uid (Brimstone listener needs
        // `state.player.creature_uid` to be Some).
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        state.sync_player_creature();

        // Two alive enemies + one dead one (the listener must skip
        // the corpse so STR is only applied to the alive set).
        let mut push_enemy = |raw: u32, hp: i32| -> CreatureUid {
            let uid = CreatureUid::new(raw).expect("non-zero");
            state.creatures.push(Creature {
                id: uid.0,
                current_hp: hp,
                max_hp: hp.max(1),
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                monster_id: MonsterId(""),
                intent: None,
                intent_history: IntentHistory::default(),
                ai: None,
            });
            state.enemy_uids.push(uid);
            uid
        };
        let enemy_a = push_enemy(10, 30);
        let enemy_b = push_enemy(11, 30);
        let _corpse = push_enemy(12, 0);

        // Equip Brimstone.
        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Brimstone,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        // C# `Brimstone` overrides only `AfterSideTurnStart` (Brimstone.cs:31,
        // both builds): the BEFORE walk, which runs ahead of block clear,
        // draw and `AfterPlayerTurnStart`, must not touch Strength (Ironclad
        // 4068: a Parafright killed by Inferno at AfterPlayerTurnStart still
        // got +1 in Rust).
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        assert!(
            !state.powers.iter().any(|p| p.kind == PowerKind::Strength),
            "Brimstone must not fire in BeforeSideTurnStart"
        );
        // Fire the listener on the player side-turn-start.
        assert_eq!(
            dispatch_after_side_turn_start(&mut state, Side::Player),
            EffectResult::Resolved
        );

        let power_amount_on = |state: &CombatState, uid: CreatureUid, kind: PowerKind| -> i32 {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .map_or(0, |c| {
                    c.powers
                        .iter()
                        .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                        .find(|p| p.kind == kind)
                        .map_or(0, |p| p.amount)
                })
        };

        assert_eq!(
            power_amount_on(&state, player_uid, PowerKind::Strength),
            BRIMSTONE_SELF_STRENGTH,
            "Brimstone must apply SELF_STRENGTH(+2) to the player"
        );
        assert_eq!(
            power_amount_on(&state, enemy_a, PowerKind::Strength),
            BRIMSTONE_ENEMY_STRENGTH,
            "Brimstone must apply ENEMY_STRENGTH(+1) to alive enemy A"
        );
        assert_eq!(
            power_amount_on(&state, enemy_b, PowerKind::Strength),
            BRIMSTONE_ENEMY_STRENGTH,
            "Brimstone must apply ENEMY_STRENGTH(+1) to alive enemy B"
        );

        // Enemy side-turn-start must be inert (mirrors the Python
        // `if side == PLAYER:` guard).
        let mut fresh = CombatState::default();
        fresh.player.current_hp = 80;
        fresh.player.max_hp = 80;
        let _ = fresh.init_player_creature();
        fresh.sync_player_creature();
        fresh.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Brimstone,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_after_side_turn_start(&mut fresh, Side::Enemy);
        assert_eq!(
            power_amount_on(
                &fresh,
                fresh.player.creature_uid.unwrap(),
                PowerKind::Strength
            ),
            0,
            "Brimstone must NOT fire on enemy side-turn-start"
        );
    }

    /// `PhilosophersStone` behavioural pin (bead `sts2-rs-w0wl`):
    /// the `before_combat_start` listener applies STRENGTH(+1) to every
    /// alive enemy. The `modify_max_energy: +1` leg is deferred to a
    /// separate bead and is NOT exercised here. Mirrors Python
    /// `shop_event.py::PhilosophersStone.before_combat_start`.
    #[test]
    fn philosophers_stone_applies_strength_to_every_alive_enemy_at_combat_start() {
        use smallvec::SmallVec;
        use std::num::NonZeroU32;
        use sts2_core::card_effects::EffectResult;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{
            Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, PowerKind, Relic,
            RelicKind,
        };

        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        state.sync_player_creature();

        let mut push_enemy = |raw: u32, hp: i32| -> CreatureUid {
            let uid = CreatureUid::new(raw).expect("non-zero");
            state.creatures.push(Creature {
                id: uid.0,
                current_hp: hp,
                max_hp: hp.max(1),
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                monster_id: MonsterId(""),
                intent: None,
                intent_history: IntentHistory::default(),
                ai: None,
            });
            state.enemy_uids.push(uid);
            uid
        };
        let enemy_a = push_enemy(10, 30);
        let enemy_b = push_enemy(11, 30);
        let _corpse = push_enemy(12, 0);

        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PhilosophersStone,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        assert_eq!(
            dispatch_before_combat_start(&mut state),
            EffectResult::Resolved
        );

        let power_amount_on = |state: &CombatState, uid: CreatureUid, kind: PowerKind| -> i32 {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .map_or(0, |c| {
                    c.powers
                        .iter()
                        .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                        .find(|p| p.kind == kind)
                        .map_or(0, |p| p.amount)
                })
        };

        assert_eq!(
            power_amount_on(&state, enemy_a, PowerKind::Strength),
            PHILOSOPHERS_STONE_ENEMY_STRENGTH,
            "PhilosophersStone must apply ENEMY_STRENGTH(+1) to alive enemy A"
        );
        assert_eq!(
            power_amount_on(&state, enemy_b, PowerKind::Strength),
            PHILOSOPHERS_STONE_ENEMY_STRENGTH,
            "PhilosophersStone must apply ENEMY_STRENGTH(+1) to alive enemy B"
        );
        // Dead enemy must not receive Strength.
        assert_eq!(
            power_amount_on(&state, CreatureUid::new(12).unwrap(), PowerKind::Strength),
            0,
            "PhilosophersStone must skip corpses (current_hp == 0)"
        );
        // Player must NOT receive Strength (PhilosophersStone is enemy-only).
        assert_eq!(
            power_amount_on(&state, player_uid, PowerKind::Strength),
            0,
            "PhilosophersStone must NOT apply Strength to the player"
        );
    }

    /// `Brimstone` is inert when the player does not own the relic
    /// (the listener's `player_owns_relic` gate must short-circuit
    /// before applying any Strength).
    #[test]
    fn brimstone_inert_when_not_owned() {
        use sts2_core::card_effects::EffectResult;
        use sts2_core::relic_effects::dispatch_before_side_turn_start;
        use sts2_core::{PowerKind, Side};

        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        state.sync_player_creature();

        // No relic equipped.
        let res = dispatch_before_side_turn_start(&mut state, Side::Player);
        // No listeners registered for this kind → channel returns
        // Resolved (no-op) regardless.
        assert!(matches!(
            res,
            EffectResult::Resolved | EffectResult::NotImplemented
        ));

        let strength = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .map_or(0, |c| {
                c.powers
                    .iter()
                    .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                    .find(|p| p.kind == PowerKind::Strength)
                    .map_or(0, |p| p.amount)
            });
        assert_eq!(strength, 0, "Brimstone listener must be owner-gated");
    }

    /// `BoomingConch` behavioural pin (bead `sts2-rs-o39p` wave 1):
    /// the `modify_hand_draw` listener returns `base + EXTRA_DRAW=2`
    /// when the player owns the relic, the round is 1, and the
    /// combat is an elite. All three gates are tested independently
    /// (round != 1, !is_elite, not owned) — each must short-circuit
    /// to `base` (no bonus draw). Mirrors Python
    /// `shop_event.py::BoomingConch.modify_hand_draw`.
    #[test]
    fn booming_conch_modify_hand_draw_round_1_elite_only() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_modify_hand_draw;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();

        let push_conch = |state: &mut CombatState| {
            state.relics.push(Relic {
                id: NonZeroU32::new(1).expect("non-zero"),
                kind: RelicKind::BoomingConch,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        };

        // (a) Round 1 + is_elite + owned → +2 draw.
        let mut s = CombatState {
            round: 1,
            is_elite: true,
            is_boss: true,
            act_index: 1,
            ..CombatState::default()
        };
        push_conch(&mut s);
        assert_eq!(
            dispatch_modify_hand_draw(&s, 5),
            5 + BOOMING_CONCH_EXTRA_DRAW,
            "round 1 + elite + owned must add EXTRA_DRAW"
        );

        // (b) Round 2 + is_elite + owned → no bonus (round gate).
        let mut s = CombatState {
            round: 2,
            is_elite: true,
            is_boss: true,
            act_index: 1,
            ..CombatState::default()
        };
        push_conch(&mut s);
        assert_eq!(
            dispatch_modify_hand_draw(&s, 5),
            5,
            "round > 1 must short-circuit to base"
        );

        // (c) Round 1 + !is_elite + owned → no bonus (elite gate).
        let mut s = CombatState {
            round: 1,
            is_elite: false,
            is_boss: false,
            act_index: 1,
            ..CombatState::default()
        };
        push_conch(&mut s);
        assert_eq!(
            dispatch_modify_hand_draw(&s, 5),
            5,
            "non-elite combat must short-circuit to base"
        );

        // (d) Round 1 + is_elite + NOT owned → no bonus (owner gate).
        let s = CombatState {
            round: 1,
            is_elite: true,
            is_boss: true,
            act_index: 1,
            ..CombatState::default()
        };
        // No push_conch call — relic not owned.
        assert_eq!(
            dispatch_modify_hand_draw(&s, 5),
            5,
            "BoomingConch listener must be owner-gated"
        );
    }

    /// `TheAbacus` behavioural pin (bead `sts2-rs-k8vc` shop-tier
    /// wave 0): the `after_shuffle_bus` listener grants
    /// `THE_ABACUS_BLOCK=6` to the player on every shuffle event.
    /// Routed through `gain_block_player` so Frail/Dexterity folds
    /// in (this fixture has neither, so the raw +6 is observable).
    /// Mirrors Python `shop_event.py::TheAbacus.after_shuffle`.
    #[test]
    fn the_abacus_gains_six_block_on_shuffle() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::{dispatch_after_shuffle_bus, ShuffleCtx};
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();

        // Player creature row + uid so `gain_block_player` lands
        // on the mirrored creature record.
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();

        // Equip TheAbacus.
        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::TheAbacus,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        // Fire the listener (auto-reshuffle path).
        let _ = dispatch_after_shuffle_bus(
            &mut state,
            &ShuffleCtx {
                from_reshuffle: true,
            },
        );
        assert_eq!(
            state.player.block, THE_ABACUS_BLOCK,
            "TheAbacus must grant BLOCK=6 on shuffle"
        );

        // Second shuffle stacks (no per-combat cap in the pin).
        let _ = dispatch_after_shuffle_bus(
            &mut state,
            &ShuffleCtx {
                from_reshuffle: false,
            },
        );
        assert_eq!(
            state.player.block,
            THE_ABACUS_BLOCK * 2,
            "TheAbacus stacks across multiple shuffle events"
        );
    }

    /// `SlingOfCourage` behavioural pin (bead `sts2-rs-k8vc` shop-tier
    /// wave 1): in elite combat, the `before_combat_start` listener
    /// grants the player `STRENGTH(+SLING_OF_COURAGE_STRENGTH=2)`.
    /// Mirrors Python `shop_event.py::SlingOfCourage.before_combat_start`.
    #[test]
    fn sling_of_courage_grants_strength_in_elite() {
        use std::num::NonZeroU32;
        use sts2_core::powers::PowerKind;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState {
            is_elite: true,
            is_boss: true,
            act_index: 1,
            ..CombatState::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();

        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::SlingOfCourage,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        let _ = dispatch_before_combat_start(&mut state);

        let player_uid = state.player.creature_uid.expect("player uid");
        let strength = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .map_or(0, |c| {
                c.powers
                    .iter()
                    .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                    .find(|p| p.kind == PowerKind::Strength)
                    .map_or(0, |p| p.amount)
            });
        assert_eq!(
            strength, SLING_OF_COURAGE_STRENGTH,
            "SlingOfCourage must grant STRENGTH=2 at elite combat start"
        );
    }

    /// `SlingOfCourage` is inert in non-elite combats.
    #[test]
    fn sling_of_courage_inert_in_non_elite() {
        use std::num::NonZeroU32;
        use sts2_core::powers::PowerKind;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();

        state.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::SlingOfCourage,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        let _ = dispatch_before_combat_start(&mut state);

        let player_uid = state.player.creature_uid.expect("player uid");
        let has_strength = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .is_some_and(|c| {
                c.powers
                    .iter()
                    .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                    .any(|p| p.kind == PowerKind::Strength)
            });
        assert!(
            !has_strength,
            "SlingOfCourage must not fire outside elite rooms"
        );
    }

    /// `SlingOfCourage` is inert when the player does not own the relic.
    #[test]
    fn sling_of_courage_inert_when_not_owned() {
        use sts2_core::powers::PowerKind;
        use sts2_core::relic_effects::dispatch_before_combat_start;

        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            is_elite: true,
            is_boss: true,
            act_index: 1,
            ..CombatState::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();

        let _ = dispatch_before_combat_start(&mut state);

        let player_uid = state.player.creature_uid.expect("player uid");
        let has_strength = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .is_some_and(|c| {
                c.powers
                    .iter()
                    .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                    .any(|p| p.kind == PowerKind::Strength)
            });
        assert!(!has_strength, "SlingOfCourage listener must be owner-gated");
    }

    /// Bead `sts2-rs-bmw3`: LeesWaffle's `after_obtained` listener must
    /// bump max HP by `LEES_WAFFLE_MAX_HP = 7` and then heal the player to
    /// full. Pins the Python `gain_max_hp(7)` + `heal(max_hp)` sequence.
    #[test]
    fn listener_lees_waffle_gains_seven_max_hp_and_heals_to_full() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Set up an injured player so the heal leg is observable.
        rs.current_hp = 50;
        let before_max = rs.max_hp;
        rs.relics.push(RelicId::LeesWaffle);
        rs.last_obtained_relic = Some(RelicId::LeesWaffle);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.max_hp,
            before_max + LEES_WAFFLE_MAX_HP,
            "LeesWaffle must bump max HP by 7"
        );
        assert_eq!(
            rs.current_hp, rs.max_hp,
            "LeesWaffle must heal the player to full after the max-HP bump"
        );
    }

    /// Bead `sts2-rs-run-rest-option`: PumpkinCandle's (v0.107.1)
    /// `AfterObtained → Rekindle` listener adds `kindleAmount = 5` to
    /// `KindleCount` (`rs.relic_counters[PumpkinCandle]`), independent
    /// of act. The combat-side `modify_max_energy` arm then grants +1
    /// max energy while that count is `> 0`.
    #[test]
    fn listener_pumpkin_candle_rekindles_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        for act in [1_u8, 2, 3] {
            let mut rs = RunState::new();
            rs.act_index = act;
            rs.relics.push(RelicId::PumpkinCandle);
            rs.last_obtained_relic = Some(RelicId::PumpkinCandle);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(
                rs.relic_counters.get(&RelicId::PumpkinCandle).copied(),
                Some(5),
                "PumpkinCandle Rekindle must set KindleCount=5 on obtain (act={act})",
            );
        }
    }

    /// PumpkinCandle's rest-site listener appends the KINDLE option
    /// while owned, and `do_rest_site_action(Kindle)` Rekindles the
    /// count by another 5 (C# `KindleRestSiteOption.OnSelect →
    /// Rekindle`). Together with `after_obtained` this gives the full
    /// KindleCount lifecycle.
    #[test]
    fn listener_pumpkin_candle_rest_site_kindle_option_and_rekindle() {
        use sts2_core::runstate::{RestSiteActionOutcome, RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::PumpkinCandle);
        // Simulate pickup Rekindle (KindleCount = 5).
        rs.last_obtained_relic = Some(RelicId::PumpkinCandle);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);

        // The rest-site option list now offers KINDLE alongside the base
        // Heal/Smith pair.
        let opts = rs.rest_site_options();
        assert!(
            opts.iter().any(|o| matches!(o, RestSiteOption::Kindle)),
            "PumpkinCandle owner must be offered the Kindle rest-site option",
        );

        // Selecting Kindle Rekindles: 5 -> 10.
        let outcome = rs.do_rest_site_action(RestSiteOption::Kindle);
        assert_eq!(outcome, RestSiteActionOutcome::Kindled);
        assert_eq!(
            rs.relic_counters.get(&RelicId::PumpkinCandle).copied(),
            Some(10),
            "Kindle rest-site pick must add 5 to KindleCount",
        );
    }

    /// PumpkinCandle's listener is a no-op when the relic isn't
    /// owned — guards against a stray write into `relic_counters`
    /// from a dispatch fired for an unrelated relic.
    #[test]
    fn listener_pumpkin_candle_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.act_index = 2;
        // Do NOT push PumpkinCandle into rs.relics.
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            !rs.relic_counters.contains_key(&RelicId::PumpkinCandle),
            "PumpkinCandle counter must not be written when the relic is unowned",
        );
    }

    // ── DingyRug — bead sts2-rs-u5emw ─────────────────────────────

    /// DingyRug's `modify_card_reward_creation_options` listener
    /// must flip `has_custom_card_pool = true`, clear
    /// `character_ids`, clear `use_default_character_pool`, and
    /// populate `custom_card_ids` with the union of the active
    /// character's pool + the colorless pool. Mirrors Python
    /// `DingyRug.modify_card_reward_creation_options`'s return
    /// shape.
    /// Prismatic Gem widens a card reward to every character pool in
    /// `ModelDb.AllCharacters` order (`PrismaticGem.cs:32-52`), and stays out
    /// of `NoCardPoolModifications` rolls.
    #[test]
    fn listener_prismatic_gem_widens_to_all_character_pools() {
        use sts2_core::potions::CharacterId;
        use sts2_core::runstate::{CardRewardGenerationOptions, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = CharacterId::Defect;
        rs.relics.push(RelicId::PrismaticGem);
        let opts = rs.modify_card_reward_creation_options(CardRewardGenerationOptions::default());
        assert_eq!(
            opts.character_ids,
            vec![
                CharacterId::Ironclad,
                CharacterId::Silent,
                CharacterId::Regent,
                CharacterId::Necrobinder,
                CharacterId::Defect,
            ]
        );
        let locked = rs.modify_card_reward_creation_options(CardRewardGenerationOptions {
            allow_card_pool_modifications: false,
            ..CardRewardGenerationOptions::default()
        });
        assert!(locked.character_ids.is_empty());
    }

    #[test]
    fn listener_dingy_rug_unions_the_colorless_pool() {
        use sts2_core::runstate::{CardRewardGenerationOptions, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::potions::CharacterId::Ironclad;
        rs.relics.push(RelicId::DingyRug);
        let mut opts = CardRewardGenerationOptions::default();

        let res = sts2_core::run_effects::dispatch_modify_card_reward_creation_options(
            &mut rs, &mut opts,
        );
        assert_eq!(res, sts2_core::card_effects::EffectResult::Resolved);
        // `options.CardPools ++ [Colorless]` (`DingyRug.cs:13-36`): the
        // character pool stays, colorless joins it, no custom pool.
        assert!(opts.include_colorless);
        assert!(opts.use_default_character_pool);
        assert!(!opts.has_custom_card_pool);
    }

    /// DingyRug's listener is a no-op when the relic isn't owned —
    /// the channel walks every registered listener, so an unowned
    /// relic must self-gate.
    #[test]
    fn listener_dingy_rug_is_noop_without_ownership() {
        use sts2_core::runstate::{CardRewardGenerationOptions, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut opts = CardRewardGenerationOptions::default();
        // Do NOT push DingyRug into rs.relics.
        let _ = sts2_core::run_effects::dispatch_modify_card_reward_creation_options(
            &mut rs, &mut opts,
        );
        assert!(
            !opts.has_custom_card_pool,
            "has_custom_card_pool must stay false when DingyRug is unowned",
        );
        assert!(
            opts.custom_card_ids.is_empty(),
            "custom_card_ids must stay empty when DingyRug is unowned",
        );
    }

    /// DingyRug honors `allow_card_pool_modifications=false` (the
    /// per-relic skip handshake): when the flag is clear, the
    /// listener returns NotImplemented and leaves `opts` untouched.
    #[test]
    fn listener_dingy_rug_skips_when_pool_modifications_disabled() {
        use sts2_core::runstate::{CardRewardGenerationOptions, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::potions::CharacterId::Ironclad;
        rs.relics.push(RelicId::DingyRug);
        let mut opts = CardRewardGenerationOptions {
            allow_card_pool_modifications: false,
            ..CardRewardGenerationOptions::default()
        };

        let _ = sts2_core::run_effects::dispatch_modify_card_reward_creation_options(
            &mut rs, &mut opts,
        );

        assert!(
            !opts.has_custom_card_pool,
            "has_custom_card_pool must stay false with allow_card_pool_modifications=false",
        );
        assert!(
            opts.custom_card_ids.is_empty(),
            "custom_card_ids must stay empty with allow_card_pool_modifications=false",
        );
    }

    // ── FurCoat — bead eboqq ──────────────────────────────────────

    /// Bead `sts2-rs-eboqq`: FurCoat's `after_obtained` listener
    /// must latch `rs.act_index` into `rs.fur_coat_act_index` on
    /// pickup so the late map-mark listener can act-gate. Latch is
    /// idempotent — once `Some`, never reassigned.
    #[test]
    fn listener_fur_coat_latches_obtain_act_on_pickup() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.act_index = 2;
        rs.relics.push(RelicId::FurCoat);
        assert!(rs.fur_coat_act_index.is_none(), "fresh run has no latch");
        rs.last_obtained_relic = Some(RelicId::FurCoat);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.fur_coat_act_index,
            Some(2),
            "after_obtained must latch act_index into fur_coat_act_index"
        );
        rs.act_index = 3;
        rs.last_obtained_relic = Some(RelicId::FurCoat);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.fur_coat_act_index,
            Some(2),
            "latch is sticky — re-dispatch in act 3 must not overwrite"
        );
    }

    /// Bead `sts2-rs-eboqq`: FurCoat's `after_obtained` listener is
    /// a no-op when the relic is not owned.
    #[test]
    fn listener_fur_coat_after_obtained_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.act_index = 2;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.fur_coat_act_index.is_none(),
            "no FurCoat in rs.relics → no latch"
        );
    }

    /// Bead `sts2-rs-eboqq`: FurCoat's late map-mark listener picks
    /// exactly `FUR_COAT_COMBATS = 7` Monster/Elite rooms and marks
    /// them via the map-layer registry from mxbai. Idempotent on
    /// re-fire.
    #[test]
    fn listener_fur_coat_marks_seven_monster_or_elite_rooms() {
        use sts2_core::map::{ActMap, MapCoord, MapPointType};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.act_index = 1;
        rs.fur_coat_act_index = Some(1);
        rs.relics.push(RelicId::FurCoat);
        let mut map = ActMap::new(15);
        for row in 1..=4 {
            for col in 0..3 {
                let idx = map.get_or_create(col, row);
                map.point_mut(idx).point_type = if (col + row) % 2 == 0 {
                    MapPointType::Monster
                } else {
                    MapPointType::Elite
                };
            }
        }
        let result =
            sts2_core::run_effects::dispatch_modify_generated_map_late(&mut rs, &mut map, 1);
        assert_eq!(result, sts2_core::card_effects::EffectResult::Resolved);
        let marked: Vec<MapCoord> = map.marked_coords_for(RelicId::FurCoat);
        assert_eq!(
            marked.len(),
            FUR_COAT_COMBATS,
            "FurCoat must mark exactly 7 rooms"
        );
        let snapshot: std::collections::BTreeSet<MapCoord> = marked.iter().copied().collect();
        let _ = sts2_core::run_effects::dispatch_modify_generated_map_late(&mut rs, &mut map, 1);
        let after: std::collections::BTreeSet<MapCoord> = map
            .marked_coords_for(RelicId::FurCoat)
            .into_iter()
            .collect();
        assert_eq!(snapshot, after, "re-dispatch must be idempotent");
    }

    /// `FurCoat.AfterObtained` marks the map that already exists
    /// (FurCoat.cs:66-71): the relic is handed over by an Ancient after the
    /// act's map is generated, so the generation-time hook never sees it.
    /// The picks are the live Monster/Elite points in `GetAllMapPoints`
    /// (column-major) order, `UnstableShuffle`d by `new Rng(Owner, Id, 0)` —
    /// the `FUR_COAT` named stream — then the first `Combats` (FurCoat.cs:93-101).
    /// Same code in both builds; only the quota differs (7 vs 8).
    #[test]
    fn fur_coat_after_obtained_marks_the_current_map_in_both_builds() {
        use sts2_core::map::{ActMap, MapCoord, MapPointType};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            rs.act_index = 2;
            let mut map = ActMap::new(15);
            // Insert out of grid order so a points-vector walk differs from
            // the column-major one.
            for row in (1..=5).rev() {
                for col in (0..4).rev() {
                    let idx = map.get_or_create(col, row);
                    map.point_mut(idx).point_type = match (col + row) % 3 {
                        0 => MapPointType::Monster,
                        1 => MapPointType::Elite,
                        _ => MapPointType::RestSite,
                    };
                }
            }
            let mut expected: Vec<MapCoord> = map
                .grid
                .keys()
                .copied()
                .filter(|c| (c.col + c.row) % 3 != 2)
                .collect();
            rs.rng.named("FUR_COAT").shuffle(&mut expected);
            expected.truncate(fur_coat_combats(version));
            rs.current_act_map = Some(map);
            rs.relics.push(RelicId::FurCoat);
            rs.last_obtained_relic = Some(RelicId::FurCoat);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            let map = rs.current_act_map.as_ref().expect("map kept");
            let mut marked = map.marked_coords_for(RelicId::FurCoat);
            marked.sort();
            expected.sort();
            assert_eq!(marked, expected, "{version:?}");
        }
    }

    /// v0.111.0: FurCoat marks 8 rooms (decomp FurCoat.cs:64).
    #[test]
    fn listener_fur_coat_marks_eight_monster_or_elite_rooms_v0111() {
        use sts2_core::map::{ActMap, MapPointType};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        rs.act_index = 1;
        rs.fur_coat_act_index = Some(1);
        rs.relics.push(RelicId::FurCoat);
        let mut map = ActMap::new(15);
        for row in 1..=4 {
            for col in 0..3 {
                let idx = map.get_or_create(col, row);
                map.point_mut(idx).point_type = MapPointType::Monster;
            }
        }
        let _ = sts2_core::run_effects::dispatch_modify_generated_map_late(&mut rs, &mut map, 1);
        assert_eq!(map.marked_coords_for(RelicId::FurCoat).len(), 8);
    }

    /// Bead `sts2-rs-eboqq`: FurCoat's late map-mark listener
    /// no-ops in any act other than the obtain act, matching
    /// Python's `current_act_index != _fur_coat_act_index` guard.
    #[test]
    fn listener_fur_coat_late_noops_on_wrong_act() {
        use sts2_core::map::{ActMap, MapPointType};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.act_index = 2;
        rs.fur_coat_act_index = Some(1);
        rs.relics.push(RelicId::FurCoat);
        let mut map = ActMap::new(15);
        for col in 0..5 {
            let idx = map.get_or_create(col, 1);
            map.point_mut(idx).point_type = MapPointType::Monster;
        }
        let _ = sts2_core::run_effects::dispatch_modify_generated_map_late(&mut rs, &mut map, 2);
        assert!(
            map.marked_coords_for(RelicId::FurCoat).is_empty(),
            "FurCoat must not mark rooms in a non-obtain act"
        );
    }

    /// Bead `sts2-rs-r4pr`: LoomingFruit's `after_obtained` listener must
    /// bump max HP by `LOOMING_FRUIT_MAX_HP = 31` when the run owns the relic.
    #[test]
    fn listener_looming_fruit_gains_thirty_one_max_hp_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        rs.relics.push(RelicId::LoomingFruit);
        rs.last_obtained_relic = Some(RelicId::LoomingFruit);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max + LOOMING_FRUIT_MAX_HP);
        assert_eq!(rs.current_hp, before_hp + LOOMING_FRUIT_MAX_HP);
    }

    /// Bead `sts2-rs-r4pr`: NutritiousOyster's `after_obtained` listener
    /// must bump max HP by `NUTRITIOUS_OYSTER_MAX_HP = 11` when owned.
    #[test]
    fn listener_nutritious_oyster_gains_eleven_max_hp_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        rs.relics.push(RelicId::NutritiousOyster);
        rs.last_obtained_relic = Some(RelicId::NutritiousOyster);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max + NUTRITIOUS_OYSTER_MAX_HP);
        assert_eq!(rs.current_hp, before_hp + NUTRITIOUS_OYSTER_MAX_HP);
    }

    /// Bead `sts2-rs-ytiok` (70anj.4.10): NewLeaf's `after_obtained`
    /// listener must open a 1-card transform prompt
    /// ([`sts2_core::ChoiceKind::CardSelect`]) on the master deck
    /// when the run owns the relic. Pins the prompt-opener branch
    /// (the in-place transform branch is deferred — see
    /// [`listener_new_leaf`]).
    #[test]
    fn new_leaf_opens_transform_prompt() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Master deck needs at least 1 card or
        // `offer_transform_cards_reward` no-ops.
        rs.master_deck.push(Card {
            card_id: CardId::StrikeIronclad,
            upgraded: false,
            ..Card::pod_defaults()
        });
        rs.relics.push(RelicId::NewLeaf);
        assert!(
            rs.pending_choice.is_none(),
            "precondition: no prompt before fire"
        );
        rs.last_obtained_relic = Some(RelicId::NewLeaf);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.pending_choice.is_some(),
            "NewLeaf must open a transform-1 prompt on pickup"
        );
    }

    /// Bead `sts2-rs-ytiok` (70anj.4.10): NewLeaf listener is a no-op
    /// when the run does not own the relic. Mirrors the canonical
    /// ownership-gate pattern (GoldenPearl, LeesWaffle, …).
    #[test]
    fn new_leaf_listener_is_noop_without_ownership() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck.push(Card {
            card_id: CardId::StrikeIronclad,
            upgraded: false,
            ..Card::pod_defaults()
        });
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.pending_choice.is_none(),
            "no NewLeaf in rs.relics → no prompt opens"
        );
    }

    /// Bead `sts2-rs-yfi0`: GoldenPearl's `after_obtained` listener must
    /// grant `GOLDEN_PEARL_GOLD = 150` to `rs.gold` when owned.
    #[test]
    fn listener_golden_pearl_gains_one_hundred_fifty_gold_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        rs.relics.push(RelicId::GoldenPearl);
        rs.last_obtained_relic = Some(RelicId::GoldenPearl);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold + GOLDEN_PEARL_GOLD);
    }

    /// Bead `sts2-rs-yfi0`: GoldenPearl listener is noop without ownership.
    #[test]
    fn listener_golden_pearl_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold);
    }

    /// Bead `sts2-rs-2ciy`: SignetRing's `after_obtained` listener must
    /// grant `SIGNET_RING_GOLD = 999` to `rs.gold` when owned.
    #[test]
    fn listener_signet_ring_gains_nine_hundred_ninety_nine_gold_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        rs.relics.push(RelicId::SignetRing);
        rs.last_obtained_relic = Some(RelicId::SignetRing);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold + SIGNET_RING_GOLD);
    }

    /// v0.111.0: SignetRing grants 888 gold (decomp SignetRing.cs:13).
    #[test]
    fn listener_signet_ring_gains_eight_hundred_eighty_eight_gold_on_obtain_v0111() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        let before_gold = rs.gold;
        rs.relics.push(RelicId::SignetRing);
        rs.last_obtained_relic = Some(RelicId::SignetRing);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold + 888);
    }

    /// Bead `sts2-rs-2ciy`: SignetRing listener is noop without ownership.
    #[test]
    fn listener_signet_ring_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold);
    }

    /// Bead `sts2-rs-psf9`: Orrery's `after_obtained` listener must push
    /// exactly `ORRERY_CARDS = 5` `Reward::Card` sentinels to
    /// `rs.pending_rewards` when owned.
    #[test]
    fn listener_orrery_pushes_five_card_reward_sentinels_on_obtain() {
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        rs.relics.push(RelicId::Orrery);
        rs.last_obtained_relic = Some(RelicId::Orrery);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len + ORRERY_CARDS);
        let pushed = &rs.pending_rewards[before_len..];
        assert_eq!(pushed.len(), 5);
        for reward in pushed {
            assert!(
                matches!(reward, Reward::Card(_)),
                "Orrery must push Reward::Card sentinels, got {reward:?}"
            );
        }
    }

    /// Bead `sts2-rs-beda` (corrected by `sts2-rs-3bigm`):
    /// LeafyPoultice's `after_obtained` listener must reduce
    /// `rs.max_hp` and `rs.current_hp` by
    /// `LEAFY_POULTICE_MAX_HP_LOSS = 12` (C# `MaxHpVar(12m)`).
    #[test]
    fn listener_leafy_poultice_loses_twelve_max_hp_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_cur = rs.current_hp;
        rs.relics.push(RelicId::LeafyPoultice);
        rs.last_obtained_relic = Some(RelicId::LeafyPoultice);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max - LEAFY_POULTICE_MAX_HP_LOSS);
        assert_eq!(rs.current_hp, before_cur - LEAFY_POULTICE_MAX_HP_LOSS);
    }

    /// Bead `sts2-rs-beda` (transform leg): LeafyPoultice transforms the
    /// **first** Basic Strike and **first** Basic Defend in the deck —
    /// NOT every basic (that is PandorasBox). With 5 Strikes + 4 Defends
    /// seeded, exactly one Strike (index 0) and one Defend (first Defend
    /// slot) are swapped for non-basic character-pool cards, leaving
    /// 4 Strikes + 3 Defends; the deck length is preserved (in-place
    /// transform).
    #[test]
    fn listener_leafy_poultice_transforms_first_strike_and_first_defend() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.character_id = CharacterId::Ironclad;
        rs.relics.push(RelicId::LeafyPoultice);
        let deck_len = pandoras_box_seed_ironclad_deck(&mut rs);

        rs.last_obtained_relic = Some(RelicId::LeafyPoultice);
        let _ = dispatch_after_obtained(&mut rs);

        assert_eq!(
            rs.master_deck.len(),
            deck_len,
            "transform_specific_cards removes+appends — deck length preserved"
        );
        let strikes = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == InnerCardId::StrikeIronclad)
            .count();
        let defends = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == InnerCardId::DefendIronclad)
            .count();
        assert_eq!(strikes, 4, "exactly the FIRST basic Strike is transformed");
        assert_eq!(defends, 3, "exactly the FIRST basic Defend is transformed");
        // C# deck transform removes the original and appends the replacement
        // to the tail, in ascending original-index order. The first Strike
        // (idx 0) and first Defend (idx 5) are removed; their replacements
        // append last, Strike's before Defend's.
        assert_ne!(
            rs.master_deck[deck_len - 2].card_id,
            InnerCardId::StrikeIronclad,
            "first Strike's replacement appended second-to-last"
        );
        assert_ne!(
            rs.master_deck[deck_len - 1].card_id,
            InnerCardId::DefendIronclad,
            "first Defend's replacement appended last"
        );
        // The anchor non-basic shifts up two slots (two cards removed before
        // it, two appended after it).
        assert_eq!(rs.master_deck[deck_len - 3].card_id, InnerCardId::Anger);
        // The surviving basics keep their relative head order.
        assert_eq!(rs.master_deck[0].card_id, InnerCardId::StrikeIronclad);
        assert_eq!(rs.master_deck[4].card_id, InnerCardId::DefendIronclad);
        // GH#31: each appended replacement must be spec-populated (not a
        // blank `pod_defaults()`), so its full stats reach combat. Compare
        // against the canonical POD the spec index builds for that id.
        for slot in [deck_len - 2, deck_len - 1] {
            let got = &rs.master_deck[slot];
            let want = crate::cards::spec_index::card_pod_from_id(got.card_id, false)
                .expect("rolled replacement id resolves to a spec");
            assert_eq!(
                got.base_damage, want.base_damage,
                "replacement carries its spec base_damage, not pod_defaults"
            );
            assert_eq!(
                got.base_block, want.base_block,
                "replacement carries its spec base_block, not pod_defaults"
            );
            assert_eq!(got.cost, want.cost, "replacement carries its spec cost");
        }
    }

    /// LeafyPoultice finds the Strike / Defend to transform by deck POSITION:
    /// `master_deck.position(|card| card.card_id == original_id)` (line 2526).
    /// The `==` → `!=` mutant matches the first card whose id DIFFERS from the
    /// target, transforming the wrong slot.
    ///
    /// Fixture: deck `[Anger, Strike, Defend]` — the basics are NOT at the
    /// head. Correct logic finds Strike at index 1 and Defend at index 2 and
    /// transforms both, leaving ZERO basic Strike/Defend. The `!=` mutant finds
    /// index 0 (Anger ≠ Strike, and Anger ≠ Defend) for BOTH, so it never
    /// touches the real Strike/Defend — they survive.
    #[test]
    fn listener_leafy_poultice_finds_basics_by_position() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.character_id = CharacterId::Ironclad;
        rs.relics.push(RelicId::LeafyPoultice);
        rs.master_deck.clear();
        for id in [
            InnerCardId::Anger,
            InnerCardId::StrikeIronclad,
            InnerCardId::DefendIronclad,
        ] {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = id;
            rs.master_deck.push(c);
        }

        rs.last_obtained_relic = Some(RelicId::LeafyPoultice);
        let _ = dispatch_after_obtained(&mut rs);

        let strikes = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == InnerCardId::StrikeIronclad)
            .count();
        let defends = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == InnerCardId::DefendIronclad)
            .count();
        assert_eq!(
            strikes, 0,
            "the Strike at index 1 must be transformed (a `!=` match would skip it)"
        );
        assert_eq!(
            defends, 0,
            "the Defend at index 2 must be transformed (a `!=` match would skip it)"
        );
    }

    /// Bead `sts2-rs-beda`: the transform leg is deterministic under a
    /// fixed seed — two identical runs produce the same replacement
    /// cards (the `transformations` stream is seeded from the run seed).
    #[test]
    fn listener_leafy_poultice_transform_is_deterministic() {
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let run = || {
            let mut rs = RunState::new();
            rs.character_id = CharacterId::Ironclad;
            rs.relics.push(RelicId::LeafyPoultice);
            pandoras_box_seed_ironclad_deck(&mut rs);
            rs.last_obtained_relic = Some(RelicId::LeafyPoultice);
            let _ = dispatch_after_obtained(&mut rs);
            rs.master_deck.iter().map(|c| c.card_id).collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → same transformed deck");
    }

    /// Bead `sts2-rs-64vb`: MassiveScroll's `after_obtained` listener
    /// must push `MASSIVE_SCROLL_CARDS = 3` `Reward::Card` sentinels
    /// to `rs.pending_rewards` when owned.
    #[test]
    fn listener_massive_scroll_pushes_three_card_reward_sentinels_on_obtain() {
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        rs.relics.push(RelicId::MassiveScroll);
        rs.last_obtained_relic = Some(RelicId::MassiveScroll);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len + MASSIVE_SCROLL_CARDS);
        let pushed = &rs.pending_rewards[before_len..];
        for reward in pushed {
            assert!(
                matches!(reward, Reward::Card(_)),
                "MassiveScroll must push Reward::Card sentinels, got {reward:?}"
            );
        }
    }

    /// Bead `sts2-rs-64vb`: MassiveScroll listener is noop without ownership.
    #[test]
    fn listener_massive_scroll_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-2nrk`: Cauldron's `after_obtained` listener must
    /// push `CAULDRON_POTIONS = 5` `Reward::Potion` sentinels to
    /// `rs.pending_rewards` when owned.
    #[test]
    fn listener_cauldron_pushes_five_potion_reward_sentinels_on_obtain() {
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        rs.relics.push(RelicId::Cauldron);
        rs.last_obtained_relic = Some(RelicId::Cauldron);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.pending_rewards.len(),
            before_len + CAULDRON_POTIONS as usize
        );
        let pushed = &rs.pending_rewards[before_len..];
        assert_eq!(pushed.len(), CAULDRON_POTIONS as usize);
        for reward in pushed {
            assert!(
                matches!(reward, Reward::Potion(_)),
                "Cauldron must push Reward::Potion sentinels, got {reward:?}"
            );
        }
    }

    /// Bead `sts2-rs-2nrk`: Cauldron listener is noop without ownership.
    #[test]
    fn listener_cauldron_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-gf5s`: SmallCapsule's `after_obtained` listener
    /// must push `SMALL_CAPSULE_RELICS = 1` RelicReward sentinel.
    #[test]
    fn listener_small_capsule_pushes_one_relic_reward_sentinel_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Post-#69: listener gates on `last_obtained_relic` and rolls a
        // real id from the grab bag — set both and drop the SmallCapsule-
        // identity assertion (the rolled id is random, by design).
        let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
        rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
        let before_len = rs.relic_rewards.len();
        rs.relics.push(RelicId::SmallCapsule);
        rs.last_obtained_relic = Some(RelicId::SmallCapsule);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relic_rewards.len(), before_len + SMALL_CAPSULE_RELICS);
    }

    /// Bead `sts2-rs-gf5s`: SmallCapsule listener is noop without ownership.
    #[test]
    fn listener_small_capsule_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.relic_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relic_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-u4wv`: CallingBell's `after_obtained` listener
    /// must push `CALLING_BELL_RELICS = 3` RelicReward sentinels to
    /// `rs.relic_rewards` when owned.
    #[test]
    fn listener_calling_bell_pushes_three_relic_reward_sentinels_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Post-#69: gate on `last_obtained_relic`, roll real (C/U/R) ids
        // from per-rarity buckets. Identity assertion dropped — pushed
        // ids are the random rolls, not CallingBell itself.
        let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
        rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
        let before_len = rs.relic_rewards.len();
        rs.relics.push(RelicId::CallingBell);
        rs.last_obtained_relic = Some(RelicId::CallingBell);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relic_rewards.len(), before_len + CALLING_BELL_RELICS);
    }

    /// `CallingBell.AfterObtained` (`CallingBell.cs:29`, same in both builds):
    /// one CurseOfTheBell goes into the deck, then the three relics are
    /// offered (GH #806).
    #[test]
    fn listener_calling_bell_adds_curse_of_the_bell_both_versions() {
        use sts2_core::runstate::RunState;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
            rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
            let deck_before = rs.master_deck.len();
            rs.relics.push(RelicId::CallingBell);
            rs.last_obtained_relic = Some(RelicId::CallingBell);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(rs.relic_rewards.len(), CALLING_BELL_RELICS);
            assert_eq!(rs.master_deck.len(), deck_before + 1, "{version:?}");
            assert_eq!(
                rs.master_deck.last().map(|c| c.card_id),
                Some(sts2_core::cards::CardId::CurseOfTheBell)
            );
        }
    }

    /// Bead `sts2-rs-u4wv`: CallingBell listener is noop without ownership.
    #[test]
    fn listener_calling_bell_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.relic_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relic_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-beda`: LeafyPoultice listener is noop without ownership.
    #[test]
    fn listener_leafy_poultice_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_cur = rs.current_hp;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max);
        assert_eq!(rs.current_hp, before_cur);
    }

    /// Bead `sts2-rs-r2y4x` + `sts2-rs-a49q3`: DistinguishedCape's
    /// `after_obtained` listener must reduce `rs.max_hp` and
    /// `rs.current_hp` by `DISTINGUISHED_CAPE_HP_LOSS = 9` AND push
    /// `DISTINGUISHED_CAPE_CARDS = 3` Apparition cards to
    /// `rs.master_deck` when owned. HP-loss leg landed in r2y4x; the
    /// Apparition deck-add leg landed in a49q3.
    #[test]
    fn listener_distinguished_cape_loses_nine_max_hp_and_adds_three_apparitions_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_cur = rs.current_hp;
        let before_deck_len = rs.master_deck.len();
        rs.relics.push(RelicId::DistinguishedCape);
        rs.last_obtained_relic = Some(RelicId::DistinguishedCape);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max - DISTINGUISHED_CAPE_HP_LOSS);
        assert_eq!(rs.current_hp, before_cur - DISTINGUISHED_CAPE_HP_LOSS);
        assert_eq!(
            rs.master_deck.len(),
            before_deck_len + DISTINGUISHED_CAPE_CARDS
        );
        let added = &rs.master_deck[before_deck_len..];
        assert!(added.iter().all(|c| c.card_id == CardId::Apparition));
    }

    /// v0.111.0: DistinguishedCape keeps Max HP and adds 2 distinct
    /// generatable curses (two `Rng.Niche` draws) before the 3 Apparitions
    /// (decomp DistinguishedCape.cs:32-53).
    #[test]
    fn listener_distinguished_cape_adds_two_curses_and_three_apparitions_on_obtain_v0111() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        let before_max = rs.max_hp;
        let before_deck_len = rs.master_deck.len();
        let niche_before = rs.rng.niche.counter();
        rs.relics.push(RelicId::DistinguishedCape);
        rs.last_obtained_relic = Some(RelicId::DistinguishedCape);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max, "no Max HP loss in v0.111.0");
        assert_eq!(rs.rng.niche.counter(), niche_before + 2);
        let added = &rs.master_deck[before_deck_len..];
        assert_eq!(added.len(), 2 + DISTINGUISHED_CAPE_CARDS);
        let pool = crate::relics::neow_relics::NEOWS_BONES_CURSE_POOL;
        assert!(pool.contains(&added[0].card_id) && pool.contains(&added[1].card_id));
        assert_ne!(
            added[0].card_id, added[1].card_id,
            "drawn without replacement"
        );
        assert!(added[2..].iter().all(|c| c.card_id == CardId::Apparition));
    }

    /// Bead `sts2-rs-r2y4x`: DistinguishedCape listener is noop without
    /// ownership.
    #[test]
    fn listener_distinguished_cape_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_cur = rs.current_hp;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max);
        assert_eq!(rs.current_hp, before_cur);
    }

    /// Bead `sts2-rs-r2y4x`: SandCastle's `after_obtained` listener
    /// must upgrade up to `SAND_CASTLE_CARDS = 6` non-upgraded master-
    /// deck cards when owned. We seed the deck with 10 non-upgraded
    /// cards and assert at most `SAND_CASTLE_CARDS` get upgraded.
    #[test]
    fn listener_sand_castle_upgrades_six_random_cards_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::SandCastle);
        rs.last_obtained_relic = Some(RelicId::SandCastle);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, SAND_CASTLE_CARDS);
    }

    /// SandCastle is a RANDOM upgrade (C# `StableShuffle(Rng.Niche).Take(6)`),
    /// NOT a player pick — so it must upgrade immediately even when the host
    /// sets `interactive_card_upgrade` (the workbench / lockstep flag), and
    /// must NEVER queue a `pending_deck_upgrades` card_select. Regression for
    /// the workbench bug where SandCastle wrongly offered "select 6 cards".
    #[test]
    fn listener_sand_castle_ignores_interactive_card_upgrade_flag() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        // Workbench / run_parity lockstep host opts into interactive upgrades
        // (for the genuine FromDeckForUpgrade relics like Pomander) — SandCastle
        // must not be caught by it.
        rs.interactive_card_upgrade = true;
        rs.relics.push(RelicId::SandCastle);
        rs.last_obtained_relic = Some(RelicId::SandCastle);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.pending_deck_upgrades, 0,
            "SandCastle must not queue an interactive card_select"
        );
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(
            n_upgraded, SAND_CASTLE_CARDS,
            "SandCastle must upgrade 6 random cards immediately"
        );
    }

    /// Bead `sts2-rs-r2y4x`: SandCastle listener is noop without
    /// ownership — no deck card should be upgraded.
    #[test]
    fn listener_sand_castle_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, 0);
    }

    /// GlassEye's `after_obtained` queues 5 pre-rolled 3-card rewards of
    /// rarities Common, Common, Uncommon, Uncommon, Rare, all base, for one
    /// `Rewards` read per card (`GlassEye.cs:15-33`: uniform odds,
    /// `NoUpgradeRoll`).
    #[test]
    fn listener_glass_eye_queues_five_forced_rarity_rewards_on_obtain() {
        use crate::cards::CardRarity;
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let rewards_before = rs.rng.rewards.counter();
        rs.relics.push(RelicId::GlassEye);
        rs.last_obtained_relic = Some(RelicId::GlassEye);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len + GLASS_EYE_CARDS);
        assert_eq!(rs.rng.rewards.counter() - rewards_before, 15);
        let want = [
            CardRarity::Common,
            CardRarity::Common,
            CardRarity::Uncommon,
            CardRarity::Uncommon,
            CardRarity::Rare,
        ];
        for (reward, rarity) in rs.pending_rewards[before_len..].iter().zip(want) {
            let Reward::Card(c) = reward else {
                panic!("GlassEye queues card rewards only");
            };
            assert_eq!(c.options.len(), 3);
            for &id in &c.options {
                assert_eq!(
                    crate::cards::spec_index::rarity_for_card_id_in(id, rs.game_version),
                    Some(rarity)
                );
            }
            assert!(c.upgraded.iter().all(|u| !u), "NoUpgradeRoll: base cards");
        }
    }

    /// Bead `sts2-rs-66l9f`: GlassEye listener is noop without ownership.
    #[test]
    fn listener_glass_eye_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-66l9f`: LostCoffer's `after_obtained` listener
    /// must push 1 `Reward::Card` + 1 `Reward::Potion` sentinel when
    /// owned (counts hard-coded to 1 in Python).
    #[test]
    fn listener_lost_coffer_pushes_one_card_and_one_potion_sentinel_on_obtain() {
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        rs.relics.push(RelicId::LostCoffer);
        rs.last_obtained_relic = Some(RelicId::LostCoffer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len + 2);
        let last_two = &rs.pending_rewards[rs.pending_rewards.len() - 2..];
        assert!(matches!(last_two[0], Reward::Card(_)));
        assert!(matches!(last_two[1], Reward::Potion(_)));
    }

    /// Bead `sts2-rs-66l9f`: LostCoffer listener is noop without ownership.
    #[test]
    fn listener_lost_coffer_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// ScrollBoxes's `after_obtained` listener (v0.107.1) **preserves** gold
    /// and queues the real two-pack bundle choice on `pending_bundles` (each
    /// pack = 2 common + 1 uncommon, no card id repeated across packs),
    /// mirroring C# `ScrollBoxes.GenerateRandomBundles`. The v0.103.2 gold
    /// loss was removed (bead `…wb8lk.9` §7). Supersedes the deferred
    /// 2-card-sentinel stub (bead `sts2-rs-66l9f`).
    #[test]
    fn listener_scroll_boxes_preserves_gold_and_queues_two_card_bundles_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new(); // Ironclad: ample commons/uncommons
        rs.gain_gold(250);
        let gold_before = rs.gold;
        rs.relics.push(RelicId::ScrollBoxes);
        rs.last_obtained_relic = Some(RelicId::ScrollBoxes);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, gold_before, "v0.107.1: ScrollBoxes keeps gold");
        assert_eq!(rs.pending_bundles.len(), SCROLL_BOXES_BUNDLES);
        for pack in &rs.pending_bundles {
            assert_eq!(pack.len(), 3, "each pack is 2 common + 1 uncommon");
        }
        // No duplicate card id across the two packs (shared `usedCardIds`).
        let mut all: Vec<_> = rs.pending_bundles.iter().flatten().copied().collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), total, "packs must not share a card id");
    }

    /// ScrollBoxes listener is a noop unless it was the just-obtained relic
    /// (gating on `last_obtained_relic`, not mere ownership) — gold preserved,
    /// no bundle queued. Guards against re-firing / re-rolling the Rewards RNG
    /// on later relic obtains.
    #[test]
    fn listener_scroll_boxes_is_noop_without_being_just_obtained() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.gain_gold(250);
        rs.relics.push(RelicId::ScrollBoxes); // owned, but not just-obtained
        let before_gold = rs.gold;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold);
        assert!(rs.pending_bundles.is_empty());
    }

    /// Bead `sts2-rs-utgn4`: Kaleidoscope's `after_obtained` queues 2
    /// pre-resolved `Reward::Card` sentinels (C# `OfferCustom([r1, r2])`),
    /// each carrying 3 cross-character cards from OTHER-character pools (never
    /// the owner's). Fires only on its own pickup.
    #[test]
    fn listener_kaleidoscope_queues_two_three_card_rewards_from_non_owner_pools_on_obtain() {
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new(); // Ironclad owner
        let owner = rs.character_id;
        rs.relics.push(RelicId::Kaleidoscope);
        rs.last_obtained_relic = Some(RelicId::Kaleidoscope);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let cards: Vec<_> = rs
            .pending_rewards
            .iter()
            .filter_map(|r| match r {
                Reward::Card(c) => Some(c.options.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(cards.len(), 2, "Kaleidoscope offers Cards==2 rewards");
        for reward in &cards {
            assert_eq!(reward.len(), 3, "each reward draws 3 cross-pool cards");
            for &id in reward {
                // Every offered card belongs to a DIFFERENT character (C#
                // `CharacterCardPools.Where(p != owner.CardPool)`).
                assert_ne!(
                    crate::cards::spec_index::character_for_card_id(id),
                    Some(owner),
                    "Kaleidoscope never offers the owner's own pool"
                );
            }
        }
    }

    /// Kaleidoscope's rewards are manually-set `CardReward`s whose `Populate`
    /// runs `TryModifyCardRewardOptions` with `IsCardReward`
    /// (`CardReward.cs:136,148-156`): an unused SilkenTress Glams the first
    /// slate and latches, so the second slate stays plain. Defect / Ironclad
    /// 5626 v0.107.1 (Neow SilkenTress, then Kaleidoscope).
    #[test]
    fn kaleidoscope_slates_run_the_card_reward_hooks() {
        use sts2_core::cards::EnchantmentId;
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::SilkenTress);
        rs.relics.push(RelicId::Kaleidoscope);
        rs.last_obtained_relic = Some(RelicId::Kaleidoscope);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let slates: Vec<_> = rs
            .pending_rewards
            .iter()
            .filter_map(|r| match r {
                Reward::Card(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(slates.len(), 2);
        assert!(
            slates[0]
                .enchantments
                .iter()
                .any(|e| e.iter().any(|x| x.0 == EnchantmentId::Glam)),
            "the first Kaleidoscope slate is Glam-enchanted: {:?}",
            slates[0]
        );
        assert_eq!(
            rs.relic_counters.get(&RelicId::SilkenTress).copied(),
            Some(1)
        );
    }

    /// Kaleidoscope listener is a noop unless it was the just-obtained relic
    /// (gating on `last_obtained_relic`, not mere ownership) — re-firing would
    /// re-roll the Niche + Rewards streams on every later relic obtain.
    #[test]
    fn listener_kaleidoscope_is_noop_without_being_just_obtained() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Kaleidoscope); // owned, but not just-obtained
        let before = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before);
    }

    /// Bead `sts2-rs-a49q3`: NeowsTorment's `after_obtained` listener
    /// must push one `NeowsFury` card to `rs.master_deck` when owned.
    #[test]
    fn listener_neows_torment_adds_one_neows_fury_to_master_deck_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::NeowsTorment);
        rs.last_obtained_relic = Some(RelicId::NeowsTorment);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
        assert_eq!(rs.master_deck.last().unwrap().card_id, CardId::NeowsFury);
    }

    /// Bead `sts2-rs-a49q3`: NeowsTorment listener is noop without
    /// ownership.
    #[test]
    fn listener_neows_torment_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-a49q3`: PaelsHorn's `after_obtained` listener
    /// must push `PAELS_HORN_CARDS = 2` Relax cards to `rs.master_deck`
    /// when owned.
    #[test]
    fn listener_paels_horn_adds_two_relax_cards_to_master_deck_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::PaelsHorn);
        rs.last_obtained_relic = Some(RelicId::PaelsHorn);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + PAELS_HORN_CARDS);
        let added = &rs.master_deck[before_len..];
        assert!(added.iter().all(|c| c.card_id == CardId::Relax));
    }

    /// Bead `sts2-rs-a49q3`: PaelsHorn listener is noop without
    /// ownership.
    #[test]
    fn listener_paels_horn_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-a49q3`: CursedPearl's `after_obtained` listener
    /// must push one `Greed` curse to `rs.master_deck` AND grant
    /// `CURSED_PEARL_GOLD = 333` gold when owned.
    #[test]
    fn listener_cursed_pearl_adds_greed_and_grants_gold_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::CursedPearl);
        rs.last_obtained_relic = Some(RelicId::CursedPearl);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold + CURSED_PEARL_GOLD);
        assert_eq!(rs.master_deck.len(), before_len + 1);
        assert_eq!(rs.master_deck.last().unwrap().card_id, CardId::Greed);
    }

    /// Bead `sts2-rs-a49q3`: CursedPearl listener is noop without
    /// ownership.
    #[test]
    fn listener_cursed_pearl_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_gold = rs.gold;
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.gold, before_gold);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-a49q3`: BloodSoakedRose's `after_obtained`
    /// listener must push one `Enthralled` curse to `rs.master_deck`
    /// when owned. modify_max_energy combat leg deferred.
    #[test]
    fn listener_blood_soaked_rose_adds_enthralled_to_master_deck_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::BloodSoakedRose);
        rs.last_obtained_relic = Some(RelicId::BloodSoakedRose);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
        assert_eq!(rs.master_deck.last().unwrap().card_id, CardId::Enthralled);
    }

    /// Bead `sts2-rs-a49q3`: BloodSoakedRose listener is noop without
    /// ownership.
    #[test]
    fn listener_blood_soaked_rose_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-qw9oe`: LeadPaperweight's `after_obtained`
    /// listener must push `LEAD_PAPERWEIGHT_CARDS = 2` `Reward::Card`
    /// sentinels when owned. Colorless filter deferred to 01o.
    #[test]
    fn listener_lead_paperweight_queues_two_distinct_colorless_choose_screen() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LeadPaperweight);
        rs.last_obtained_relic = Some(RelicId::LeadPaperweight);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // No reward sentinels — a concrete skippable choose-1 screen
        // (C# FromChooseACardScreen) is queued instead.
        assert!(rs.pending_rewards.is_empty());
        assert_eq!(rs.pending_card_selects.len(), 1);
        let opts = &rs.pending_card_selects[0].options;
        assert_eq!(opts.len(), LEAD_PAPERWEIGHT_CARDS, "got {opts:?}");
        assert_ne!(opts[0], opts[1], "options distinct, got {opts:?}");
        for id in opts {
            assert!(
                crate::cards::spec_index::COLORLESS_SPECS
                    .iter()
                    .any(|f| f().id == *id),
                "option {id:?} must come from the colorless pool"
            );
        }
        assert!(rs.pending_card_selects[0].post_add.is_empty());
    }

    /// LeadPaperweight gates on `last_obtained_relic` — owning it is not
    /// enough (the after-obtained channel fires every listener on every
    /// relic obtain; a contains-gate re-rolled the Rewards stream and
    /// queued a phantom screen on each later pickup).
    #[test]
    fn listener_lead_paperweight_is_noop_when_not_the_obtained_relic() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LeadPaperweight);
        rs.last_obtained_relic = Some(RelicId::BurningBlood);
        let rewards_cursor_before = rs.rng.rewards.counter();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(rs.pending_card_selects.is_empty());
        assert_eq!(
            rs.rng.rewards.counter(),
            rewards_cursor_before,
            "no Rewards reads without a fresh LeadPaperweight obtain"
        );
    }

    /// C# `SeaGlass.AfterObtained` (`SeaGlass.cs:74-92`, both builds): 5
    /// Common + 5 Uncommon + 5 Rare uniform-odds cards from ANOTHER
    /// character's pool (one Rewards read each, no rarity/upgrade roll) in an
    /// optional 0..=15 grid, not 15 empty reward sentinels.
    #[test]
    fn listener_sea_glass_offers_optional_grid_of_other_character_cards() {
        use crate::cards::{spec_index, CardRarity};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            let character = rs.sea_glass_character();
            assert_ne!(character, rs.character_id, "another character's pool");
            let rewards_before = rs.rng.rewards.counter();
            let rewards_len = rs.pending_rewards.len();
            rs.relics.push(RelicId::SeaGlass);
            rs.last_obtained_relic = Some(RelicId::SeaGlass);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(rs.pending_rewards.len(), rewards_len, "{version:?}");
            assert_eq!(
                rs.rng.rewards.counter(),
                rewards_before + SEA_GLASS_CARDS as u32,
                "{version:?}: one Rewards read per card"
            );
            assert_eq!(rs.pending_card_grants.len(), 1);
            let grant = &rs.pending_card_grants[0];
            assert_eq!((grant.pick, grant.min_pick), (SEA_GLASS_CARDS, 0));
            assert_eq!(grant.options.len(), SEA_GLASS_CARDS);
            let rarity_of = |id| {
                spec_index::specs_for_character(character)
                    .iter()
                    .map(|f| f())
                    .find(|s| s.id == id)
                    .map(|s| s.rarity)
            };
            for (i, card) in grant.options.iter().enumerate() {
                let expected = [CardRarity::Common, CardRarity::Uncommon, CardRarity::Rare][i / 5];
                assert_eq!(
                    rarity_of(card.card_id),
                    Some(expected),
                    "{version:?} option {i}"
                );
                assert!(!card.upgraded, "no upgrade roll");
            }
            for block in grant.options.chunks(5) {
                let mut ids: Vec<_> = block.iter().map(|c| c.card_id).collect();
                ids.sort_by_key(|id| format!("{id:?}"));
                ids.dedup();
                assert_eq!(ids.len(), 5, "{version:?}: distinct within a rarity");
            }
        }
    }

    /// Bead `sts2-rs-qw9oe`: SeaGlass listener is noop without
    /// ownership.
    #[test]
    fn listener_sea_glass_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-qw9oe`: YummyCookie's `after_obtained` listener
    /// must upgrade up to `YUMMY_COOKIE_CARDS = 4` non-upgraded
    /// master-deck cards when owned. Same convention as SandCastle.
    #[test]
    fn listener_yummy_cookie_upgrades_four_random_cards_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::YummyCookie);
        rs.last_obtained_relic = Some(RelicId::YummyCookie);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, YUMMY_COOKIE_CARDS);
    }

    /// Bead `sts2-rs-qw9oe`: YummyCookie listener is noop without
    /// ownership — no deck card should be upgraded.
    #[test]
    fn listener_yummy_cookie_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, 0);
    }

    /// Bead `sts2-rs-cvbxs`: Pomander's `after_obtained` listener
    /// must upgrade one non-upgraded master-deck card when owned.
    #[test]
    fn listener_pomander_upgrades_one_random_card_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::Pomander);
        rs.last_obtained_relic = Some(RelicId::Pomander);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, 1);
    }

    /// Bead `sts2-rs-cvbxs`: Pomander listener is noop without
    /// ownership.
    #[test]
    fn listener_pomander_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(n_upgraded, 0);
    }

    /// Bead `sts2-rs-cvbxs` + `sts2-rs-t762n`: PrecariousShears's
    /// `after_obtained` listener must reduce `rs.current_hp` by
    /// `PRECARIOUS_SHEARS_DAMAGE = 16` AND remove
    /// `PRECARIOUS_SHEARS_CARDS = 2` cards from the deck when owned
    /// (non-interactive auto-removal path — damage applies eagerly).
    /// HP-loss leg landed in cvbxs; remove-cards leg landed in t762n.
    #[test]
    fn listener_precarious_shears_loses_hp_and_removes_two_cards_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_cur = rs.current_hp;
        let before_deck_len = rs.master_deck.len();
        rs.relics.push(RelicId::PrecariousShears);
        rs.last_obtained_relic = Some(RelicId::PrecariousShears);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.current_hp, before_cur - PRECARIOUS_SHEARS_DAMAGE);
        assert_eq!(
            rs.master_deck.len(),
            before_deck_len - PRECARIOUS_SHEARS_CARDS
        );
    }

    /// Bead `sts2-rs-1hiht`: in interactive-removal mode PrecariousShears
    /// must NOT lower HP at obtain — it queues the removal `card_select`
    /// (`pending_deck_removals`) and DEFERS the damage
    /// (`pending_post_removal_hp_loss`) until the pick resolves, matching
    /// C#'s `FromDeckForRemoval` → `RemoveFromDeck` → `Damage` order.
    #[test]
    fn listener_precarious_shears_defers_hp_loss_in_interactive_mode() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.interactive_card_removal = true;
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_cur = rs.current_hp;
        let before_deck_len = rs.master_deck.len();
        rs.relics.push(RelicId::PrecariousShears);
        rs.last_obtained_relic = Some(RelicId::PrecariousShears);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // HP unchanged; deck unchanged (removal is now a pending pick);
        // damage parked for after the pick.
        assert_eq!(rs.current_hp, before_cur);
        assert_eq!(rs.master_deck.len(), before_deck_len);
        assert_eq!(rs.pending_deck_removals, PRECARIOUS_SHEARS_CARDS as u32);
        assert_eq!(rs.pending_post_removal_hp_loss, PRECARIOUS_SHEARS_DAMAGE);
    }

    /// Bead `sts2-rs-cvbxs`: PrecariousShears listener is noop without
    /// ownership.
    #[test]
    fn listener_precarious_shears_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_cur = rs.current_hp;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.current_hp, before_cur);
    }

    /// Bead `sts2-rs-cvbxs`: AlchemicalCoffer's `after_obtained`
    /// listener must increase `rs.max_potion_slots` by
    /// `ALCHEMICAL_COFFER_POTION_SLOTS = 4` AND fill the new slots with
    /// 4 random out-of-combat potions, mirroring C#
    /// `AlchemicalCoffer.cs:20-29` (`GainMaxPotionCount` +
    /// `CreateRandomPotionsOutOfCombat` + `TryToProcure` loop).
    #[test]
    fn listener_alchemical_coffer_gains_four_potion_slots_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_slots = rs.max_potion_slots;
        let before_potions = rs.potions.len();
        rs.relics.push(RelicId::AlchemicalCoffer);
        rs.last_obtained_relic = Some(RelicId::AlchemicalCoffer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.max_potion_slots,
            before_slots + ALCHEMICAL_COFFER_POTION_SLOTS
        );
        // The 4 new slots are filled with 4 random potions (belt started
        // empty in a fresh RunState; cap rose to >=4 above).
        assert_eq!(
            rs.potions.len(),
            before_potions + ALCHEMICAL_COFFER_POTION_SLOTS as usize
        );
    }

    /// Bead `sts2-rs-cvbxs`: AlchemicalCoffer listener is noop without
    /// ownership.
    #[test]
    fn listener_alchemical_coffer_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_slots = rs.max_potion_slots;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_potion_slots, before_slots);
    }

    /// Bead `sts2-rs-t762n`: PreciseScissors's `after_obtained`
    /// listener must remove one card from `rs.master_deck` when owned.
    #[test]
    fn listener_precise_scissors_removes_one_card_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::PreciseScissors);
        rs.last_obtained_relic = Some(RelicId::PreciseScissors);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len - 1);
    }

    /// Bead `sts2-rs-t762n`: PreciseScissors listener is noop without
    /// ownership.
    #[test]
    fn listener_precise_scissors_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-t762n`: EmptyCage's `after_obtained` listener
    /// must remove `EMPTY_CAGE_CARDS = 2` cards from `rs.master_deck`
    /// when owned.
    #[test]
    fn listener_empty_cage_removes_two_cards_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::EmptyCage);
        rs.last_obtained_relic = Some(RelicId::EmptyCage);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len - EMPTY_CAGE_CARDS);
    }

    /// Bead `sts2-rs-t762n`: EmptyCage listener is noop without
    /// ownership.
    #[test]
    fn listener_empty_cage_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-t762n` / `sts2-rs-rn76k`: PreservedFog's
    /// `after_obtained` listener must remove `PRESERVED_FOG_CARDS = 3`
    /// cards (per C# `PreservedFog.cs:18` `CardsVar(3)`) then push 1
    /// Folly curse, net deck delta = -2 when owned.
    /// Interactive removal: Folly lands only after the removal batch
    /// (`PreservedFog.cs:20-27`), so it is not in the removal pool.
    #[test]
    fn listener_preserved_fog_interactive_defers_folly_past_the_removals() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.interactive_card_removal = true;
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::PreservedFog);
        rs.last_obtained_relic = Some(RelicId::PreservedFog);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), 5, "nothing added before the picks");
        assert_eq!(rs.pending_deck_removals, 3);
        assert_eq!(rs.pending_post_removal_cards.len(), 1);
        assert_eq!(rs.pending_post_removal_cards[0].card_id, CardId::Folly);
    }

    #[test]
    fn listener_preserved_fog_removes_three_cards_and_adds_one_folly_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::PreservedFog);
        rs.last_obtained_relic = Some(RelicId::PreservedFog);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // -3 strikes removed, +1 Folly = net -2
        assert_eq!(rs.master_deck.len(), before_len - PRESERVED_FOG_CARDS + 1);
        assert_eq!(rs.master_deck.last().unwrap().card_id, CardId::Folly);
    }

    /// Bead `sts2-rs-t762n`: PreservedFog listener is noop without
    /// ownership.
    #[test]
    fn listener_preserved_fog_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-rvwas`: ArchaicTooth's `after_obtained` listener
    /// must rewrite the first eligible starter card's `card_id` to its
    /// ancient counterpart when owned, preserving upgrade flag.
    #[test]
    fn listener_archaic_tooth_transforms_bash_to_break_preserving_upgrade() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Three Strikes (not in mapping), one upgraded Bash, one Defend.
        for _ in 0..3 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let mut bash = sts2_core::cards::Card::pod_defaults();
        bash.card_id = CardId::Bash;
        bash.upgraded = true;
        rs.master_deck.push(bash);
        let mut defend = sts2_core::cards::Card::pod_defaults();
        defend.card_id = CardId::DefendIronclad;
        rs.master_deck.push(defend);
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::ArchaicTooth);
        rs.last_obtained_relic = Some(RelicId::ArchaicTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
        // Bash should be transformed to Break, upgrade preserved.
        let break_idx = rs
            .master_deck
            .iter()
            .position(|c| c.card_id == CardId::Break)
            .expect("Bash transformed to Break");
        assert!(rs.master_deck[break_idx].upgraded);
        assert!(!rs.master_deck.iter().any(|c| c.card_id == CardId::Bash));
        // CardCmd.Transform appends a deck replacement (CardCmd.cs:427-436).
        assert_eq!(break_idx, before_len - 1, "Break is appended last");
        assert_eq!(
            rs.master_deck[break_idx].card_type,
            sts2_core::cards::CardType::Attack,
            "a fresh Break, not Bash's stats under a new id"
        );
    }

    /// v0.113.0 `ArchaicTooth.TranscendenceUpgrades` adds FoolishShot ->
    /// DivineShot (`ArchaicTooth.cs:48-51`).
    #[test]
    fn listener_archaic_tooth_transforms_foolish_shot_to_divine_shot() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_113_0;
        for id in [CardId::StrikeFool, CardId::FoolishShot] {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = id;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::ArchaicTooth);
        rs.last_obtained_relic = Some(RelicId::ArchaicTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let ids: Vec<_> = rs.master_deck.iter().map(|c| c.card_id).collect();
        assert_eq!(ids, vec![CardId::StrikeFool, CardId::DivineShot]);
    }

    /// Bead `sts2-rs-rvwas`: ArchaicTooth transforms only the first
    /// matching starter and leaves the rest of the deck alone.
    #[test]
    fn listener_archaic_tooth_transforms_only_first_match() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Two Bash cards.
        for _ in 0..2 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::Bash;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::ArchaicTooth);
        rs.last_obtained_relic = Some(RelicId::ArchaicTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_break = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::Break)
            .count();
        let n_bash = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::Bash)
            .count();
        assert_eq!(n_break, 1);
        assert_eq!(n_bash, 1);
    }

    /// Bead `sts2-rs-rvwas`: ArchaicTooth listener is noop without
    /// ownership — no deck card transformed.
    #[test]
    fn listener_archaic_tooth_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut bash = sts2_core::cards::Card::pod_defaults();
        bash.card_id = CardId::Bash;
        rs.master_deck.push(bash);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck[0].card_id, CardId::Bash);
    }

    /// Bead `sts2-rs-w9vf8`: Claws's `after_obtained` listener must
    /// rewrite up to `CLAWS_CARDS = 6` deck cards to `CardId::Maul`,
    /// preserving upgrade state.
    #[test]
    fn listener_claws_transforms_six_cards_to_maul_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::Claws);
        rs.last_obtained_relic = Some(RelicId::Claws);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let n_maul = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::Maul)
            .count();
        assert_eq!(n_maul, CLAWS_CARDS);
    }

    /// Bead `sts2-rs-scds8`: a Claws transform must REBUILD the instance from
    /// Maul's spec, not just flip `card_id` — otherwise a non-Enemy source
    /// (target_type None, base_damage 0) becomes a malformed Maul that
    /// `play_card` dispatches untargeted, panicking `play_maul`'s
    /// `target.expect`. Assert the transformed card carries Maul's combat
    /// fields, while uid + upgrade + enchantment state survive.
    #[test]
    fn listener_claws_rebuilds_instance_from_maul_spec() {
        use sts2_core::cards::{CardId, EnchantmentId, TargetType};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // A Defend-like source: non-Enemy target, zero damage, upgraded, with
        // an enchantment — every field that a bare `card_id =` would leave stale.
        let mut src = sts2_core::cards::Card::pod_defaults();
        src.card_id = CardId::DefendIronclad;
        src.target_type = TargetType::None;
        src.card_type = sts2_core::cards::CardType::Skill;
        src.base_damage = 0;
        src.upgraded = true;
        src.enchantments.push((EnchantmentId::Glam, 1));
        let src_uid = src.uid;
        rs.master_deck.push(src);
        rs.relics.push(RelicId::Claws);
        rs.last_obtained_relic = Some(RelicId::Claws);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);

        let maul_ref = crate::cards::spec_index::card_pod_from_id(CardId::Maul, true)
            .expect("Maul spec registered");
        let c = &rs.master_deck[0];
        assert_eq!(c.card_id, CardId::Maul);
        assert_eq!(c.target_type, TargetType::Enemy, "must take Maul's target");
        assert_eq!(c.card_type, sts2_core::cards::CardType::Attack);
        assert_eq!(c.base_damage, maul_ref.base_damage, "upgraded Maul damage");
        assert_eq!(c.cost, maul_ref.cost);
        assert_eq!(c.uid, src_uid, "uid preserved");
        assert!(c.upgraded, "upgrade state preserved");
        assert_eq!(
            c.enchantments.to_vec(),
            &[(EnchantmentId::Glam, 1)],
            "enchantments preserved"
        );
    }

    /// C# `Claws.CreateMaulFromOriginal` (v0.111.0 `Claws.cs:46-61`,
    /// v0.107.1 identical) clones the original's enchantment and keeps it only
    /// `if (enchantmentModel.CanEnchant(cardModel))` — i.e. against the MAUL.
    /// Maul is an untagged non-Exhaust Attack, so a Defend's Nimble / Goopy /
    /// Imbued or a basic Strike's Spiral is dropped, while Sharp (Attack) and
    /// Glam (base rule) carry over.
    #[test]
    fn listener_claws_keeps_only_enchantments_maul_can_take() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::default(),
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            let cases = [
                (CardId::DefendIronclad, EnchantmentId::Nimble, false),
                (CardId::DefendIronclad, EnchantmentId::Goopy, false),
                (CardId::DefendIronclad, EnchantmentId::Imbued, false),
                (CardId::StrikeIronclad, EnchantmentId::Spiral, false),
                (CardId::StrikeIronclad, EnchantmentId::Sharp, true),
                (CardId::DefendIronclad, EnchantmentId::Glam, true),
            ];
            for (id, ench, _) in cases {
                let mut src = crate::cards::spec_index::card_pod_from_id(id, false).expect("spec");
                src.enchantments.push((ench, 2));
                rs.master_deck.push(src);
            }
            rs.relics.push(RelicId::Claws);
            rs.last_obtained_relic = Some(RelicId::Claws);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            for (i, (id, ench, kept)) in cases.into_iter().enumerate() {
                let c = &rs.master_deck[i];
                assert_eq!(c.card_id, CardId::Maul);
                let want: &[(EnchantmentId, i32)] = if kept { &[(ench, 2)] } else { &[] };
                assert_eq!(
                    c.enchantments.to_vec(),
                    want,
                    "{version:?} {id:?}+{ench:?} -> Maul"
                );
            }
        }
    }

    /// Claws builds its candidate list the way C# `CardSelectCmd.
    /// FromDeckForTransformation` does (v0.111.0 `CardSelectCmd.cs:592`):
    /// `Type != Quest && IsTransformable`, where a deck card is transformable
    /// iff it is removable, i.e. not Eternal (`CardModel.cs:738-751`). So an
    /// Eternal curse (AscendersBane) and a Quest card (LanternKey) are never
    /// picked, while a Maul already in the deck IS a legal pick (C# does not
    /// exclude it). The automatic pick stays the first 6 candidates in deck
    /// order.
    #[test]
    fn listener_claws_candidates_skip_eternal_and_quest_but_not_maul() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let ids = [
            CardId::AscendersBane,
            CardId::LanternKey,
            CardId::Maul,
            CardId::StrikeIronclad,
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::DefendIronclad,
            CardId::Bash,
            CardId::StrikeIronclad,
        ];
        for id in ids {
            let c = crate::cards::spec_index::card_pod_from_id(id, false).expect("spec");
            rs.master_deck.push(c);
        }
        let uids: Vec<_> = rs.master_deck.iter().map(|c| c.uid).collect();
        rs.relics.push(RelicId::Claws);
        rs.last_obtained_relic = Some(RelicId::Claws);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let got: Vec<_> = rs.master_deck.iter().map(|c| c.card_id).collect();
        // Candidates 2..=7 (Maul, 2 Strikes, 2 Defends, Bash) become Mauls;
        // the Eternal curse, the Quest card and the 7th candidate stay put.
        assert_eq!(
            got,
            vec![
                CardId::AscendersBane,
                CardId::LanternKey,
                CardId::StrikeIronclad,
                CardId::Maul,
                CardId::Maul,
                CardId::Maul,
                CardId::Maul,
                CardId::Maul,
                CardId::Maul,
            ]
        );
        // `CardCmd.Transform` (CardCmd.cs:383-436) removes each original
        // and appends its replacement to the deck in original pile-index
        // order, so the six Mauls sit at the tail in source order.
        let tail: Vec<_> = rs.master_deck[3..].iter().map(|c| c.uid).collect();
        assert_eq!(tail, uids[2..8].to_vec());
    }

    /// Bead `sts2-rs-w9vf8`: Claws clamps to deck size when the deck
    /// has fewer than `CLAWS_CARDS` cards.
    #[test]
    fn listener_claws_clamps_to_deck_size() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::Claws);
        rs.last_obtained_relic = Some(RelicId::Claws);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(rs.master_deck.iter().all(|c| c.card_id == CardId::Maul));
        assert_eq!(rs.master_deck.len(), 3);
    }

    /// Bead `sts2-rs-w9vf8`: Claws listener is noop without
    /// ownership — no card transformed.
    #[test]
    fn listener_claws_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(rs
            .master_deck
            .iter()
            .all(|c| c.card_id == CardId::StrikeIronclad));
    }

    /// Bead `sts2-rs-w9vf8`: NutritiousSoup's `after_obtained` listener
    /// must set `TezcatarasEmber = 1` on every basic-Strike card in
    /// the deck, leaving non-Strike cards untouched.
    #[test]
    fn listener_nutritious_soup_enchants_all_basic_strikes_with_tezcataras_ember() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for id in [
            CardId::StrikeIronclad,
            CardId::StrikeSilent,
            CardId::DefendIronclad,
        ] {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = id;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::NutritiousSoup);
        rs.last_obtained_relic = Some(RelicId::NutritiousSoup);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // Strikes got the enchantment.
        for card in &rs.master_deck {
            let val = card.enchantment(EnchantmentId::TezcatarasEmber);
            match card.card_id {
                CardId::StrikeIronclad | CardId::StrikeSilent => {
                    assert_eq!(val, 1, "{:?} should be enchanted", card.card_id);
                    // TezcatarasEmber.OnEnchant zeroes the energy cost.
                    assert_eq!(card.local_cost(), 0, "{:?} should cost 0", card.card_id);
                }
                CardId::DefendIronclad => {
                    assert_eq!(val, 0, "Defend should NOT be enchanted");
                    assert_eq!(card.local_cost(), 1, "Defend keeps its cost");
                }
                _ => {}
            }
        }
    }

    /// Bead `sts2-rs-w9vf8`: NutritiousSoup listener is noop without
    /// ownership.
    #[test]
    fn listener_nutritious_soup_is_noop_without_ownership() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut c = sts2_core::cards::Card::pod_defaults();
        c.card_id = CardId::StrikeIronclad;
        rs.master_deck.push(c);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::TezcatarasEmber),
            0
        );
    }

    /// Bead `sts2-rs-70anj.4`: GnarledHammer's `after_obtained`
    /// listener must enchant up to `GNARLED_HAMMER_CARDS = 3` eligible
    /// Attack cards with `Sharp(GNARLED_HAMMER_SHARP = 3)`.
    #[test]
    fn listener_gnarled_hammer_enchants_up_to_three_attacks_with_sharp_three() {
        use sts2_core::cards::EnchantmentId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Five attacks: only the first three should be enchanted.
        for _ in 0..5 {
            push_deck_card(&mut rs, sts2_core::cards::CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::GnarledHammer);
        rs.last_obtained_relic = Some(RelicId::GnarledHammer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let enchanted = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Sharp) == GNARLED_HAMMER_SHARP)
            .count();
        assert_eq!(
            enchanted, GNARLED_HAMMER_CARDS,
            "first 3 attacks must be Sharp"
        );
        assert_eq!(rs.master_deck.len(), 5, "no cards added/removed");
    }

    /// Bead `sts2-rs-70anj.4`: GnarledHammer listener is no-op without
    /// ownership; deck enchantments untouched.
    #[test]
    fn listener_gnarled_hammer_is_noop_without_ownership() {
        use sts2_core::cards::EnchantmentId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, sts2_core::cards::CardId::StrikeIronclad);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck[0].enchantment(EnchantmentId::Sharp), 0);
    }

    /// Bead `sts2-rs-70anj.4`: Sharp gate must skip non-Attack cards
    /// (per `can_enchant_card` rules). With 1 attack + 5 skills,
    /// only the attack is enchanted.
    #[test]
    fn listener_gnarled_hammer_skips_non_attack_cards() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::DefendIronclad);
        }
        rs.relics.push(RelicId::GnarledHammer);
        rs.last_obtained_relic = Some(RelicId::GnarledHammer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::Sharp),
            GNARLED_HAMMER_SHARP,
            "the lone Attack must be enchanted"
        );
        let defends_enchanted = rs.master_deck[1..]
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Sharp) != 0)
            .count();
        assert_eq!(defends_enchanted, 0, "Skills must not receive Sharp");
    }

    /// Build a single-card deck whose card carries `AscendersBane`'s id (whose
    /// spec is `UNPLAYABLE`) but an Attack `card_type`, so the Sharp gate's
    /// `card.card_type == Attack` check passes and only the SPEC lookup decides
    /// eligibility. Used to pin the `pool.iter().find(|s| s.id == card.card_id)`
    /// spec match in `enchantable_candidate_card_ids` / `enchant_eligible_deck_cards`.
    fn rs_with_unplayable_attack_card() -> sts2_core::runstate::RunState {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::runstate::RunState;
        let mut rs = RunState::new();
        rs.master_deck.clear();
        let mut card = Card::pod_defaults();
        // AscendersBane's real spec is UNPLAYABLE (pool index 3); a `!=` spec
        // match instead latches onto the FIRST non-matching spec (StrikeIronclad,
        // playable), flipping eligibility.
        card.card_id = CardId::AscendersBane;
        card.card_type = CardType::Attack;
        rs.master_deck.push(card);
        rs
    }

    /// `enchantable_candidate_card_ids` looks up each deck card's spec via
    /// `pool.iter().find(|s| s.id == card.card_id)` (line 2724). The card's
    /// REAL spec (`AscendersBane`, UNPLAYABLE) makes `can_enchant_card` return
    /// false, so the card is NOT a Sharp candidate. The `==` → `!=` mutant
    /// matches the first OTHER spec (StrikeIronclad, playable) instead, which
    /// WOULD pass the UNPLAYABLE check and wrongly admit the card.
    #[test]
    fn enchantable_candidate_uses_matching_spec() {
        let rs = rs_with_unplayable_attack_card();
        let candidates =
            enchantable_candidate_card_ids(&rs, sts2_core::cards::EnchantmentId::Sharp);
        assert!(
            candidates.is_empty(),
            "the card's own spec is UNPLAYABLE → not a Sharp candidate; a \
             mismatched spec would wrongly admit it, got {candidates:?}"
        );
    }

    /// `enchant_eligible_deck_cards` looks up each deck card's spec via
    /// `pool.iter().find(|s| s.id == card.card_id)` (line 2745) before the
    /// `can_enchant_card` gate. With the card's REAL spec (`AscendersBane`,
    /// UNPLAYABLE) the gate rejects it → nothing enchanted. The `==` → `!=`
    /// mutant matches StrikeIronclad's (playable) spec → the card is wrongly
    /// enchanted with Sharp.
    #[test]
    fn enchant_eligible_uses_matching_spec() {
        use sts2_core::cards::EnchantmentId;
        let mut rs = rs_with_unplayable_attack_card();
        let applied = enchant_eligible_deck_cards(&mut rs, EnchantmentId::Sharp, 3, 3);
        assert_eq!(
            applied, 0,
            "the card's own spec is UNPLAYABLE → not enchantable; a mismatched \
             spec would enchant it"
        );
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::Sharp),
            0,
            "no Sharp should land on the UNPLAYABLE-spec card"
        );
    }

    /// `gain_block_player_unpowered` dispatches the `after_block_gained` buses
    /// ONLY when `final_block > 0` (line 1594). `JuggernautPower` reacts to that
    /// bus by dealing its amount to a random alive enemy, so a positive block
    /// grant with Juggernaut installed MUST damage the enemy — proving the
    /// guard fired the dispatch. Kills the `> → ==` and `> → <` mutants (both
    /// suppress the dispatch for a positive grant). The `> → >=` mutant is
    /// EQUIVALENT: `base = SAI_BLOCK = 7 > 0`, so `final_block` is always > 0
    /// here and `>= 0` never diverges from `> 0`.
    #[test]
    fn gain_block_player_unpowered_fires_after_block_gained() {
        use smallvec::SmallVec;
        use sts2_core::{
            Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, PowerKind, Side,
        };
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.side = Side::Player;
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        state.sync_player_creature();

        let enemy_uid = CreatureUid::new(10).expect("nz");
        state.creatures.push(Creature {
            id: enemy_uid.0,
            current_hp: 40,
            max_hp: 40,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Nibbit),
            monster_id: MonsterId(""),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: None,
        });
        state.enemy_uids.push(enemy_uid);

        // Juggernaut(5) on the player: reacts to after_block_gained by dealing
        // 5 unpowered damage to a random alive enemy.
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player_uid,
            PowerKind::Juggernaut,
            5,
        );

        let enemy_hp = |state: &CombatState| {
            state
                .creatures
                .iter()
                .find(|c| c.id == enemy_uid.0)
                .map_or(0, |c| c.current_hp)
        };
        let before = enemy_hp(&state);

        gain_block_player_unpowered(&mut state, SAI_BLOCK);

        assert_eq!(
            state.player.block, SAI_BLOCK,
            "7 block granted unconditionally"
        );
        assert_eq!(
            before - enemy_hp(&state),
            5,
            "final_block (7) > 0 → after_block_gained dispatched → Juggernaut dealt 5"
        );
    }

    /// Bead `sts2-rs-2v8y2`: when `defer_followup_rewards` is set,
    /// GnarledHammer queues a player-pick `Reward::EnchantCards`
    /// instead of auto-enchanting the deck. Pool carries only the
    /// Sharp-eligible Attacks; `min_count = Some(0)` matches Python
    /// `relics/shop_event.py:410` (skippable).
    #[test]
    fn listener_gnarled_hammer_deferred_path_queues_enchant_reward() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        for _ in 0..2 {
            push_deck_card(&mut rs, CardId::DefendIronclad);
        }
        rs.relics.push(RelicId::GnarledHammer);
        rs.defer_followup_rewards = true;
        rs.last_obtained_relic = Some(RelicId::GnarledHammer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // No card enchanted in the master deck.
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::Sharp) == 0),
            "deferred path must not enchant the deck directly"
        );
        // Exactly one EnchantCards reward queued.
        let queued = rs
            .pending_rewards
            .iter()
            .filter(|r| matches!(r, Reward::EnchantCards(_)))
            .count();
        assert_eq!(queued, 1, "exactly one EnchantCards reward queued");
        let Reward::EnchantCards(r) = rs
            .pending_rewards
            .iter()
            .find(|r| matches!(r, Reward::EnchantCards(_)))
            .expect("reward queued")
        else {
            unreachable!()
        };
        assert_eq!(r.enchantment, EnchantmentId::Sharp);
        assert_eq!(r.amount, GNARLED_HAMMER_SHARP);
        assert_eq!(r.count, GNARLED_HAMMER_CARDS);
        assert_eq!(r.min_count, Some(0));
        // Pool: 3 Sharp-eligible Strikes; 2 Defends excluded by
        // `can_enchant_card(Sharp)`.
        assert_eq!(
            r.options.len(),
            3,
            "pool carries the 3 Sharp-eligible Attacks only"
        );
        assert!(r.options.iter().all(|cid| *cid == CardId::StrikeIronclad));
    }

    /// Bead `sts2-rs-g5ksv`: Kifuda deferred path queues an
    /// EnchantCards reward (Adroit, min_count=Some(0)) instead of
    /// auto-enchanting. Python pin `relics/shop_event.py:428`.
    #[test]
    fn listener_kifuda_deferred_path_queues_enchant_reward() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::Kifuda);
        rs.defer_followup_rewards = true;
        rs.last_obtained_relic = Some(RelicId::Kifuda);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::Adroit) == 0),
            "deferred path leaves the deck untouched"
        );
        let Reward::EnchantCards(r) = rs
            .pending_rewards
            .iter()
            .find(|r| matches!(r, Reward::EnchantCards(_)))
            .expect("EnchantCards reward queued")
        else {
            unreachable!()
        };
        assert_eq!(r.enchantment, EnchantmentId::Adroit);
        assert_eq!(r.amount, KIFUDA_ADROIT);
        assert_eq!(r.count, KIFUDA_CARDS);
        assert_eq!(r.min_count, Some(0));
    }

    /// Bead `sts2-rs-g5ksv`: PunchDagger deferred path queues an
    /// EnchantCards reward (Momentum, no min_count). Python pin
    /// `relics/shop_event.py:587`.
    #[test]
    fn listener_punch_dagger_deferred_path_queues_enchant_reward() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::PunchDagger);
        rs.defer_followup_rewards = true;
        rs.last_obtained_relic = Some(RelicId::PunchDagger);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::Momentum) == 0),
            "deferred path leaves the deck untouched"
        );
        let Reward::EnchantCards(r) = rs
            .pending_rewards
            .iter()
            .find(|r| matches!(r, Reward::EnchantCards(_)))
            .expect("EnchantCards reward queued")
        else {
            unreachable!()
        };
        assert_eq!(r.enchantment, EnchantmentId::Momentum);
        assert_eq!(r.amount, PUNCH_DAGGER_MOMENTUM);
        assert_eq!(r.count, 1, "count hard-coded to 1 per Python pin");
        assert_eq!(r.min_count, None);
    }

    /// Bead `sts2-rs-g5ksv`: BeautifulBracelet deferred path queues
    /// an EnchantCards reward (Swift, no min_count). Python pin
    /// `relics/shop_event.py:1030`.
    #[test]
    fn listener_beautiful_bracelet_deferred_path_queues_enchant_reward() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::BeautifulBracelet);
        rs.defer_followup_rewards = true;
        rs.last_obtained_relic = Some(RelicId::BeautifulBracelet);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::Swift) == 0),
            "deferred path leaves the deck untouched"
        );
        let Reward::EnchantCards(r) = rs
            .pending_rewards
            .iter()
            .find(|r| matches!(r, Reward::EnchantCards(_)))
            .expect("EnchantCards reward queued")
        else {
            unreachable!()
        };
        assert_eq!(r.enchantment, EnchantmentId::Swift);
        assert_eq!(r.amount, BEAUTIFUL_BRACELET_SWIFT);
        assert_eq!(r.count, BEAUTIFUL_BRACELET_CARDS);
        assert_eq!(r.min_count, None);
    }

    /// Bead `sts2-rs-g5ksv`: RoyalStamp deferred path queues an
    /// EnchantCards reward (RoyallyApproved, no min_count). Python
    /// pin `relics/shop_event.py:619`.
    /// RoyalStamp's `UnstableShuffle(Rng.Niche)` of the enchantable cards
    /// costs n - 1 Niche reads whatever is picked.
    #[test]
    fn listener_royal_stamp_shuffles_the_candidates_on_niche() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
            push_deck_card(&mut rs, id);
        }
        rs.relics.push(RelicId::RoyalStamp);
        rs.last_obtained_relic = Some(RelicId::RoyalStamp);
        let before = rs.rng.niche.counter();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.rng.niche.counter() - before, 2);
    }

    #[test]
    fn listener_royal_stamp_deferred_path_queues_enchant_reward() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::{Reward, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        rs.relics.push(RelicId::RoyalStamp);
        rs.defer_followup_rewards = true;
        rs.last_obtained_relic = Some(RelicId::RoyalStamp);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::RoyallyApproved) == 0),
            "deferred path leaves the deck untouched"
        );
        let Reward::EnchantCards(r) = rs
            .pending_rewards
            .iter()
            .find(|r| matches!(r, Reward::EnchantCards(_)))
            .expect("EnchantCards reward queued")
        else {
            unreachable!()
        };
        assert_eq!(r.enchantment, EnchantmentId::RoyallyApproved);
        assert_eq!(r.amount, 1);
        assert_eq!(r.count, 1);
        assert_eq!(r.min_count, None);
    }

    /// Bead `sts2-rs-70anj.4`: Kifuda's `after_obtained` listener must
    /// enchant up to `KIFUDA_CARDS = 3` eligible cards with
    /// `Adroit(KIFUDA_ADROIT = 3)`. Adroit is unrestricted (any card),
    /// so 3 of 5 cards get the enchantment.
    #[test]
    fn listener_kifuda_enchants_up_to_three_cards_with_adroit_three() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::Kifuda);
        rs.last_obtained_relic = Some(RelicId::Kifuda);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let enchanted = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Adroit) == KIFUDA_ADROIT)
            .count();
        assert_eq!(enchanted, KIFUDA_CARDS);
    }

    /// Bead `sts2-rs-70anj.4`: PunchDagger enchants exactly 1 Attack
    /// card with `Momentum(PUNCH_DAGGER_MOMENTUM = 5)`.
    #[test]
    fn listener_punch_dagger_enchants_one_attack_with_momentum_five() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::PunchDagger);
        rs.last_obtained_relic = Some(RelicId::PunchDagger);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::Momentum),
            PUNCH_DAGGER_MOMENTUM
        );
        assert_eq!(rs.master_deck[1].enchantment(EnchantmentId::Momentum), 0);
        assert_eq!(rs.master_deck[2].enchantment(EnchantmentId::Momentum), 0);
    }

    /// Bead `sts2-rs-70anj.4`: BeautifulBracelet enchants up to
    /// `BEAUTIFUL_BRACELET_CARDS = 3` cards with
    /// `Swift(BEAUTIFUL_BRACELET_SWIFT = 3)`.
    #[test]
    fn listener_beautiful_bracelet_enchants_up_to_three_with_swift_three() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..4 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::BeautifulBracelet);
        rs.last_obtained_relic = Some(RelicId::BeautifulBracelet);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let enchanted = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Swift) == BEAUTIFUL_BRACELET_SWIFT)
            .count();
        assert_eq!(enchanted, BEAUTIFUL_BRACELET_CARDS);
    }

    /// v0.111.0: BeautifulBracelet enchants 4 RANDOM enchantable cards with
    /// Swift 2 and never opens a deck select, even in an interactive run: one
    /// `Rng.Niche` Fisher-Yates over the eligible list (5 draws for 6 cards),
    /// then the first four (decomp BeautifulBracelet.cs:31-44).
    #[test]
    fn listener_beautiful_bracelet_enchants_four_random_cards_with_swift_two_v0111() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        rs.interactive_card_transform = true;
        for _ in 0..6 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        let niche_before = rs.rng.niche.counter();
        rs.relics.push(RelicId::BeautifulBracelet);
        rs.last_obtained_relic = Some(RelicId::BeautifulBracelet);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(rs.pending_deck_enchants.is_empty(), "no deck select");
        let enchanted = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Swift) == 2)
            .count();
        assert_eq!(enchanted, 4);
        assert_eq!(rs.rng.niche.counter(), niche_before + 5);
    }

    /// v0.111.0 `BeautifulBracelet.AfterObtained` (BeautifulBracelet.cs:31-44)
    /// shuffles EVERY `swift.CanEnchant` deck card on `Rng.Niche`. Abundance
    /// (v0.111-only, Skill, from the Dowsing quest) passes the base
    /// `EnchantmentModel.CanEnchant` (EnchantmentModel.cs:273-294), so 6
    /// Strikes + Abundance is 7 eligible cards = 6 Niche draws. The deck-pool
    /// lookup used to miss Abundance (V0111_ONLY_SPECS sit outside it) and
    /// drew 5 (act23 Ironclad 8199/8135, Niche one draw behind).
    #[test]
    fn beautiful_bracelet_v0111_counts_v0111_only_deck_cards() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        rs.interactive_card_transform = true;
        for _ in 0..6 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        push_deck_card(&mut rs, CardId::Abundance);
        assert_eq!(
            rs.master_deck[6].card_type,
            sts2_core::cards::CardType::Skill,
            "push_deck_card must resolve the v0.111-only spec"
        );
        let niche_before = rs.rng.niche.counter();
        rs.relics.push(RelicId::BeautifulBracelet);
        rs.last_obtained_relic = Some(RelicId::BeautifulBracelet);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.rng.niche.counter(), niche_before + 6);
    }

    fn three_strike_slate() -> Vec<sts2_core::cards::Card> {
        (0..3)
            .map(|_| {
                crate::cards::spec_index::card_pod_from_id(
                    sts2_core::cards::CardId::StrikeIronclad,
                    false,
                )
                .expect("strike pod")
            })
            .collect()
    }

    /// C# `Hook.TryModifyCardRewardOptions` (Hook.cs:1598-1620) walks
    /// `RunState.IterateHookListeners` — the player's relics in OBTAIN order
    /// (RunState.cs:148), not `RelicId` order. Glitter obtained before
    /// WingCharm Glams every option first (Glitter.cs:18-35), so WingCharm's
    /// `swift.CanEnchant` filter is empty and it returns before
    /// `Rng.Niche.NextItem` (WingCharm.cs:24-43): no Niche draw, no Swift
    /// (act23 Ironclad 8180 act3#b3 step 729).
    #[test]
    fn card_reward_hooks_run_in_obtain_order_glitter_then_wing_charm() {
        use sts2_core::cards::EnchantmentId;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Glitter);
        rs.relics.push(RelicId::WingCharm);
        let niche_before = rs.rng.niche.counter();
        let cards = rs.offer_card_reward(three_strike_slate());
        assert_eq!(
            rs.rng.niche.counter(),
            niche_before,
            "WingCharm found nothing"
        );
        for c in &cards {
            assert_eq!(c.enchantment(EnchantmentId::Glam), 1);
            assert_eq!(c.enchantment(EnchantmentId::Swift), 0);
        }
    }

    /// The other order: WingCharm Swifts one option (one Niche draw), and
    /// Glitter then skips it — `glam.CanEnchant` refuses a card that already
    /// carries a (non-stackable) enchantment (EnchantmentModel.cs:289-292).
    #[test]
    fn card_reward_hooks_run_in_obtain_order_wing_charm_then_glitter() {
        use sts2_core::cards::EnchantmentId;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WingCharm);
        rs.relics.push(RelicId::Glitter);
        let niche_before = rs.rng.niche.counter();
        let cards = rs.offer_card_reward(three_strike_slate());
        assert_eq!(rs.rng.niche.counter(), niche_before + 1);
        let swift: Vec<_> = cards
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Swift) > 0)
            .collect();
        assert_eq!(swift.len(), 1);
        assert_eq!(
            swift[0].enchantment(EnchantmentId::Glam),
            0,
            "Glitter skips it"
        );
        let glam = cards
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Glam) == 1)
            .count();
        assert_eq!(glam, 2);
    }

    /// Glitter's `glam.CanEnchant` gate (Glitter.cs:28): no Glam on a Curse
    /// or Status option.
    #[test]
    fn glitter_skips_cards_glam_cannot_enchant() {
        use sts2_core::cards::{CardId, EnchantmentId};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Glitter);
        let slate = vec![
            crate::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false).unwrap(),
            crate::cards::spec_index::card_pod_from_id(CardId::Wound, false).unwrap(),
        ];
        let cards = rs.offer_card_reward(slate);
        assert_eq!(cards[0].enchantment(EnchantmentId::Glam), 1);
        assert_eq!(cards[1].enchantment(EnchantmentId::Glam), 0);
    }

    /// Bead `sts2-rs-70anj.4`: RoyalStamp enchants exactly 1 card
    /// (Attack or Skill, per `can_enchant_card`) with
    /// `RoyallyApproved(1)`.
    #[test]
    fn listener_royal_stamp_enchants_one_card_with_royally_approved_one() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        push_deck_card(&mut rs, CardId::DefendIronclad);
        rs.relics.push(RelicId::RoyalStamp);
        rs.last_obtained_relic = Some(RelicId::RoyalStamp);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::RoyallyApproved),
            1
        );
        assert_eq!(
            rs.master_deck[1].enchantment(EnchantmentId::RoyallyApproved),
            0
        );
    }

    /// RoyalStamp in INTERACTIVE mode surfaces a deck card-select
    /// (`CardSelectCmd.FromDeckForEnchantment`, `RoyalStamp.cs:37`)
    /// instead of auto-enchanting the first eligible card: it queues a
    /// single `PendingDeckEnchant` (RoyallyApproved) and leaves the deck
    /// untouched until the player picks.
    #[test]
    fn listener_royal_stamp_interactive_queues_deck_enchant() {
        use sts2_core::cards::{CardId, CardType, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad); // Attack
        push_deck_card(&mut rs, CardId::DefendIronclad); // Skill
        rs.interactive_card_transform = true;
        rs.relics.push(RelicId::RoyalStamp);
        rs.last_obtained_relic = Some(RelicId::RoyalStamp);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // Nothing auto-applied — the pick is deferred to the player.
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::RoyallyApproved) == 0),
            "interactive path must not auto-enchant any card"
        );
        assert_eq!(rs.pending_deck_enchants.len(), 1, "one interactive pick");
        let entry = &rs.pending_deck_enchants[0];
        assert_eq!(entry.enchantment, EnchantmentId::RoyallyApproved);
        assert_eq!(entry.amount, 1);
        // RoyallyApproved is Attack|Skill (`can_enchant_card`).
        assert_eq!(entry.type_restriction, Some(CardType::Attack));
        assert_eq!(entry.secondary_type, Some(CardType::Skill));
    }

    /// A multi-card enchant relic (GnarledHammer, `GNARLED_HAMMER_CARDS`
    /// picks) queues one interactive pick per card when interactive and
    /// the candidate pool exceeds the pick count.
    #[test]
    fn listener_gnarled_hammer_interactive_queues_count_picks() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..(GNARLED_HAMMER_CARDS + 2) {
            push_deck_card(&mut rs, CardId::StrikeIronclad); // Attack — Sharp-eligible
        }
        rs.interactive_card_transform = true;
        rs.relics.push(RelicId::GnarledHammer);
        rs.last_obtained_relic = Some(RelicId::GnarledHammer);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.master_deck
                .iter()
                .all(|c| c.enchantment(EnchantmentId::Sharp) == 0),
            "interactive path must not auto-enchant"
        );
        assert_eq!(rs.pending_deck_enchants.len(), GNARLED_HAMMER_CARDS);
        assert!(rs
            .pending_deck_enchants
            .iter()
            .all(|e| e.enchantment == EnchantmentId::Sharp));
    }

    /// Degenerate interactive case: when the candidate pool is no larger
    /// than the pick count there is no real choice, so the relic
    /// auto-applies (matching the SelfHelpBook `eligible > 1` convention).
    #[test]
    fn listener_royal_stamp_interactive_single_candidate_auto_applies() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad); // sole eligible card
        rs.interactive_card_transform = true;
        rs.relics.push(RelicId::RoyalStamp);
        rs.last_obtained_relic = Some(RelicId::RoyalStamp);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.pending_deck_enchants.is_empty(),
            "no interactive pick for a single candidate"
        );
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::RoyallyApproved),
            1,
            "auto-applied to the sole eligible card"
        );
    }

    /// Bead `sts2-rs-70anj.4`: BiiigHug's `after_obtained` listener
    /// must remove `BIIIG_HUG_CARDS = 4` cards from `rs.master_deck`
    /// (head-N drain) when the relic is owned.
    #[test]
    fn listener_biiig_hug_removes_four_cards_from_master_deck_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::BiiigHug);
        let before_len = rs.master_deck.len();
        rs.last_obtained_relic = Some(RelicId::BiiigHug);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len - BIIIG_HUG_CARDS);
    }

    /// Bead `sts2-rs-70anj.4`: BiiigHug's `after_obtained` listener is
    /// no-op without ownership; deck length unchanged.
    #[test]
    fn listener_biiig_hug_remove_is_noop_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Beads `sts2-rs-70anj.4`, `sts2-rs-lfdaa`: BiiigHug's `after_shuffle`
    /// listener adds one Soot to the draw pile per shuffle event while owned,
    /// at a `CardPilePosition.Random` depth — which costs exactly one
    /// Shuffle-stream read (C# `AddInternal(card, Rng.Shuffle.NextInt(Count +
    /// 1))`, `CardPileCmd.cs:511-515`). The earlier top-insert consumed zero
    /// reads and desynced every later draw/reshuffle in the combat.
    #[test]
    fn listener_biiig_hug_shuffle_adds_soot_at_random_draw_depth() {
        use sts2_core::cards::{CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_shuffle_bus, ShuffleCtx};
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        equip_relic(&mut state, RelicKind::BiiigHug);
        // Seed a draw pile so a random index has more than one slot to land in.
        for i in 1..=4u32 {
            state
                .player
                .draw
                .push(CardUid::new(i + 100).expect("non-zero"));
        }
        let before_draw_len = state.player.draw.len();
        let before_shuffle_reads = state.run_rng.shuffle.counter();

        let _ = dispatch_after_shuffle_bus(&mut state, &ShuffleCtx::default());

        assert_eq!(state.player.draw.len(), before_draw_len + 1);
        assert_eq!(
            state.run_rng.shuffle.counter(),
            before_shuffle_reads + 1,
            "CardPilePosition.Random costs exactly one Shuffle-stream read",
        );
        let soot_uid = *state
            .player
            .draw
            .iter()
            .find(|u| state.cards.iter().any(|c| c.uid == **u))
            .expect("the spawned Soot is the only uid backed by the card arena");
        let card = state
            .cards
            .iter()
            .find(|c| c.uid == soot_uid)
            .expect("soot card in arena");
        assert_eq!(card.card_id, CardId::Soot);
        assert_eq!(card.card_type, CardType::Status);
        assert_eq!(card.local_cost(), -1);
    }

    /// A card-effect draw that reshuffles fires Biiig Hug mid-draw (C#
    /// `ShuffleIfNecessary` → `Hook.AfterShuffle` inside `CardPileCmd.Draw`),
    /// so the Soot joins the reshuffled pile before the rest of the draw.
    /// act23 Necrobinder 1019: Soul's draw-2 reshuffle added no Soot.
    #[test]
    fn biiig_hug_fires_on_a_card_effect_draw_reshuffle() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        equip_relic(&mut state, RelicKind::BiiigHug);
        for i in 1..=4u32 {
            let uid = CardUid::new(i).expect("non-zero");
            state.cards.push(Card {
                uid,
                card_id: CardId::StrikeIronclad,
                ..Card::pod_defaults()
            });
            state.player.discard.push(uid);
        }
        let _ = sts2_core::cards::draw_via_state(&mut state, 2);
        let soot_count = state
            .player
            .draw
            .iter()
            .chain(state.player.hand.iter())
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Soot)
            .count();
        assert_eq!(soot_count, 1, "the reshuffle added one Soot");
        assert_eq!(state.player.hand.len(), 2);
    }

    /// C# `Hook.AfterShuffle` walks `IterateCombatHookListeners`: the
    /// player's powers before their relics. So `StratagemPower.AfterShuffle`
    /// (StratagemPower.cs:17-28) moves its pick to hand, awaiting the select,
    /// BEFORE `BiiigHug.AfterShuffle` (BiiigHug.cs:30-39) inserts Soot at a
    /// random depth of the draw pile, now one card shorter. Rust ran Biiig Hug
    /// first, so the Soot's random slot was rolled over a pile still holding
    /// the pick (act23 Ironclad 6116 act 2, step 985).
    #[test]
    fn stratagem_select_resolves_before_biiig_hug_soot() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::{CardUid, Power, PowerKind, PowerUid};
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        equip_relic(&mut state, RelicKind::BiiigHug);
        let pid = std::num::NonZeroU32::new(50).expect("nz");
        state.powers.push(Power::new(pid, PowerKind::Stratagem, 1));
        state.player.powers.push(PowerUid(pid));
        for i in 1..=4u32 {
            let uid = CardUid::new(i).expect("non-zero");
            state.cards.push(Card {
                uid,
                card_id: CardId::StrikeIronclad,
                ..Card::pod_defaults()
            });
            state.player.discard.push(uid);
        }
        let soot_in = |s: &sts2_core::CombatState, pile: &[CardUid]| {
            pile.iter()
                .filter(|u| s.cards[u.0.get() as usize - 1].card_id == CardId::Soot)
                .count()
        };
        let _ = sts2_core::cards::draw_via_state(&mut state, 2);
        assert!(state.pending_choice.is_some(), "Stratagem grid opens");
        assert_eq!(
            soot_in(&state, &state.player.draw),
            0,
            "Biiig Hug waits for the Stratagem pick"
        );
        assert_eq!(state.player.draw.len(), 4);

        let counter_before = state.run_rng.shuffle.counter();
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::ResolveChoice {
                selection: smallvec::smallvec![0],
            },
        );
        assert!(state.pending_choice.is_none());
        assert!(state.run_rng.shuffle.counter() > counter_before);
        let all: Vec<CardUid> = state
            .player
            .draw
            .iter()
            .chain(state.player.hand.iter())
            .copied()
            .collect();
        assert_eq!(soot_in(&state, &all), 1, "one Soot, added after the pick");
        assert_eq!(state.player.hand.len(), 3, "the pick + the 2 drawn");
    }

    /// The Soot POD comes from the pinned status spec, so it carries the
    /// `GeneratedInCombat` marker (C# `CardModel.DeckVersion == null`) that
    /// the old hand-rolled `Card::pod_defaults()` build dropped. Bead
    /// `sts2-rs-lfdaa`.
    #[test]
    fn listener_biiig_hug_soot_is_marked_generated_in_combat() {
        use sts2_core::cards::VarKey;
        use sts2_core::relic_effects::{dispatch_after_shuffle_bus, ShuffleCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        equip_relic(&mut state, RelicKind::BiiigHug);
        let _ = dispatch_after_shuffle_bus(&mut state, &ShuffleCtx::default());
        let soot = state.cards.last().expect("soot card in arena");
        assert_eq!(soot.combat_var(VarKey::GeneratedInCombat), 1);
    }

    /// Bead `sts2-rs-70anj.4`: NinjaScroll spawns `NINJA_SCROLL_SHIVS = 3`
    /// Shivs into the player's hand before the turn-1 hand draw.
    #[test]
    fn listener_ninja_scroll_spawns_three_shivs_on_round_one_player_turn() {
        use std::num::NonZeroU32;
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).unwrap()).unwrap();
        state.relics.push(sts2_core::Relic {
            id,
            kind: RelicKind::NinjaScroll,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        state.round = 1;
        let before_hand = state.player.hand.len();
        let res = listener_ninja_scroll(&mut state);
        assert!(matches!(res, EffectResult::Resolved));
        assert_eq!(
            state.player.hand.len(),
            before_hand + usize::try_from(NINJA_SCROLL_SHIVS).unwrap()
        );
        // Every newly-added card is a Shiv.
        let new_uids = &state.player.hand[before_hand..];
        for uid in new_uids {
            let card = state.cards.iter().find(|c| c.uid == *uid).unwrap();
            assert_eq!(card.card_id, CardId::Shiv);
            assert_eq!(card.local_cost(), 0);
            assert_eq!(card.base_damage, 4);
        }
    }

    /// Bead `sts2-rs-70anj.4`: NinjaScroll listener is a no-op on rounds
    /// >= 2 (C# `TurnNumber <= 1`, NinjaScroll.cs:25).
    #[test]
    fn listener_ninja_scroll_is_noop_off_round_one() {
        use std::num::NonZeroU32;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).unwrap()).unwrap();
        state.relics.push(sts2_core::Relic {
            id,
            kind: RelicKind::NinjaScroll,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let before_hand = state.player.hand.len();

        // Round 2: noop.
        state.round = 2;
        let _ = listener_ninja_scroll(&mut state);
        assert_eq!(state.player.hand.len(), before_hand);
    }

    /// Bead `sts2-rs-70anj.4`: ElectricShrymp enchants exactly 1
    /// Skill card with `Imbued(1)`. With a deck of [Strike, Defend,
    /// Strike], only the Defend (Skill) gets enchanted — the first
    /// Strike (Attack) is skipped by the Skill-only Imbued gate.
    #[test]
    fn listener_electric_shrymp_enchants_one_skill_card_with_imbued_one() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        push_deck_card(&mut rs, CardId::DefendIronclad);
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        rs.relics.push(RelicId::ElectricShrymp);
        rs.last_obtained_relic = Some(RelicId::ElectricShrymp);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let enchanted: Vec<_> = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Imbued) == 1)
            .collect();
        assert_eq!(enchanted.len(), 1, "exactly 1 card enchanted");
        assert_eq!(
            enchanted[0].card_id,
            CardId::DefendIronclad,
            "the Defend (Skill) must be the enchanted card, not the Strikes"
        );
    }

    /// Bead `sts2-rs-70anj.4`: RadiantPearl spawns 1 Luminesce into
    /// the player's hand before the turn-1 hand draw.
    #[test]
    fn listener_radiant_pearl_spawns_one_luminesce_on_round_one_player_turn() {
        use std::num::NonZeroU32;
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).unwrap()).unwrap();
        state.relics.push(sts2_core::Relic {
            id,
            kind: RelicKind::RadiantPearl,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        state.round = 1;
        let before_hand = state.player.hand.len();
        let res = listener_radiant_pearl(&mut state);
        assert!(matches!(res, EffectResult::Resolved));
        assert_eq!(state.player.hand.len(), before_hand + 1);
        let new_uid = *state.player.hand.last().unwrap();
        let card = state.cards.iter().find(|c| c.uid == new_uid).unwrap();
        assert_eq!(card.card_id, CardId::Luminesce);
    }

    /// Bead `sts2-rs-70anj.4`: RadiantPearl listener is a no-op on rounds
    /// >= 2 (C# `TurnNumber == 1`, RadiantPearl.cs:34).
    #[test]
    fn listener_radiant_pearl_is_noop_off_round_one() {
        use std::num::NonZeroU32;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).unwrap()).unwrap();
        state.relics.push(sts2_core::Relic {
            id,
            kind: RelicKind::RadiantPearl,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        state.round = 1;
        let before_hand = state.player.hand.len();
        // Round 2: noop (the hand-draw channel is the player's only).
        state.round = 2;
        let _ = listener_radiant_pearl(&mut state);
        assert_eq!(state.player.hand.len(), before_hand);
    }

    /// Bead `sts2-rs-70anj.4`: ToyBox pushes `TOY_BOX_RELICS = 4`
    /// RelicReward sentinels onto `rs.pending_rewards`.
    /// v0.111.0: ToyBox offers 5 wax relics (decomp ToyBox.cs:83).
    #[test]
    fn listener_toy_box_pushes_five_relic_reward_sentinels_on_obtain_v0111() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
        rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
        rs.relics.push(RelicId::ToyBox);
        rs.last_obtained_relic = Some(RelicId::ToyBox);
        let before = rs.relic_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.relic_rewards.len() - before, 5);
        assert!(rs.relic_rewards[before..].iter().all(|r| r.is_wax));
    }

    #[test]
    fn listener_toy_box_pushes_four_relic_reward_sentinels_on_obtain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Post-#69: gate on `last_obtained_relic` + roll real ids; the
        // per-entry ToyBox-identity check no longer holds.
        let (by_rarity, fallback, boss) = crate::run_callbacks::default_relic_grab_bag();
        rs.populate_relic_grab_bag_by_rarity(by_rarity, fallback, boss);
        rs.relics.push(RelicId::ToyBox);
        rs.last_obtained_relic = Some(RelicId::ToyBox);
        let before = rs.relic_rewards.len();
        rs.last_obtained_relic = Some(RelicId::ToyBox);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let pushed = rs.relic_rewards.len() - before;
        assert_eq!(pushed, TOY_BOX_RELICS);
        // C# `ToyBox.cs:93` stamps `IsWax = true` on every offered model.
        assert!(
            rs.relic_rewards[before..].iter().all(|r| r.is_wax),
            "every ToyBox offer is wax"
        );
        // The flag rides the OFFER: only a TAKEN reward becomes wax.
        let taken = rs.relic_rewards[before].relic_id;
        let skipped = rs.relic_rewards[before + 1].relic_id;
        assert!(rs.wax_relics.is_empty(), "nothing wax before a pick");
        let _ = rs.accept_relic_reward(before);
        assert!(rs.wax_relics.contains(&taken), "taken offer is wax");
        assert!(
            !rs.wax_relics.contains(&skipped),
            "an unpicked offer never becomes wax"
        );
    }

    /// Bead `sts2-rs-70anj.4`: ToyBox listener is a no-op without
    /// ownership.
    #[test]
    fn listener_toy_box_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before = rs.relic_rewards.len();
        let res = listener_toy_box(&mut rs);
        assert!(matches!(res, EffectResult::NotImplemented));
        assert_eq!(rs.relic_rewards.len(), before);
    }

    /// GH #1054: PaelsClaw enchants only `Goopy.CanEnchant` cards (C#
    /// `PaelsClaw.cs:33-44`) — un-enchanted Defend-tagged cards. A Strike,
    /// a non-Defend card and a Defend already carrying `Sown` are skipped;
    /// a Goopy Defend exhausts on play (`Goopy.OnEnchant` adds Exhaust).
    #[test]
    fn listener_paels_claw_enchants_only_unenchanted_defends() {
        use sts2_core::card_effects::card_exhausts_inst;
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeDefect);
        push_deck_card(&mut rs, CardId::DefendDefect);
        push_deck_card(&mut rs, CardId::DefendDefect);
        push_deck_card(&mut rs, CardId::DefendDefect);
        push_deck_card(&mut rs, CardId::DefendDefect);
        push_deck_card(&mut rs, CardId::Zap);
        rs.master_deck[1].set_enchantment(EnchantmentId::Sown, 1);
        rs.relics.push(RelicId::PaelsClaw);
        rs.last_obtained_relic = Some(RelicId::PaelsClaw);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let goopy: Vec<bool> = rs
            .master_deck
            .iter()
            .map(|c| c.has_enchantment(EnchantmentId::Goopy))
            .collect();
        assert_eq!(goopy, vec![false, false, true, true, true, false]);
        assert!(
            card_exhausts_inst(&rs.master_deck[2]),
            "Goopy Defend exhausts"
        );
        assert!(
            !card_exhausts_inst(&rs.master_deck[1]),
            "Sown Defend does not"
        );
        assert!(!card_exhausts_inst(&rs.master_deck[0]));
    }

    /// Bead `sts2-rs-70anj.4`: PaelsClaw listener is a no-op without
    /// ownership.
    #[test]
    fn listener_paels_claw_is_noop_without_ownership() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        let res = listener_paels_claw(&mut rs);
        assert!(matches!(res, EffectResult::NotImplemented));
        assert_eq!(rs.master_deck[0].enchantment(EnchantmentId::Goopy), 0);
    }

    /// Bead `sts2-rs-70anj.4`: VelvetChoker's `after_card_played`
    /// listener bumps `relic.counter` once; the player-side
    /// `before_side_turn_start` resets it back to 0.
    #[test]
    fn listener_velvet_choker_counter_bumps_and_resets() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::CardEventCtx;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).unwrap()).unwrap();
        state.relics.push(sts2_core::Relic {
            id,
            kind: RelicKind::VelvetChoker,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let ctx = CardEventCtx {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let _ = listener_velvet_choker_card_played(&mut state, &ctx);
        let ctx = CardEventCtx {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let _ = listener_velvet_choker_card_played(&mut state, &ctx);
        assert_eq!(state.relics[0].counter, 2, "counter bumped twice");
        // Enemy side does not reset.
        let _ = listener_velvet_choker_turn_reset(&mut state, Side::Enemy);
        assert_eq!(state.relics[0].counter, 2, "enemy side: no reset");
        // Player side resets.
        let _ = listener_velvet_choker_turn_reset(&mut state, Side::Player);
        assert_eq!(state.relics[0].counter, 0, "player side: reset to 0");
    }

    /// Bead `sts2-rs-70anj.4`: VelvetChoker counter listener is a no-op
    /// without ownership.
    #[test]
    fn listener_velvet_choker_card_played_is_noop_without_ownership() {
        use sts2_core::relic_effects::CardEventCtx;
        crate::relics::dispatch::ensure_installed();
        let mut state = sts2_core::CombatState::default();
        let ctx = CardEventCtx {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let res = listener_velvet_choker_card_played(&mut state, &ctx);
        assert!(matches!(res, EffectResult::NotImplemented));
    }

    /// Bead `sts2-rs-70anj.4`: Ectoplasm vetoes every positive gold
    /// gain via the should_gain_gold dispatcher.
    #[test]
    fn listener_ectoplasm_vetoes_positive_gold_gain() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Ectoplasm);
        let gold_before = rs.gold;
        rs.gain_gold(100);
        assert_eq!(rs.gold, gold_before, "Ectoplasm vetoes positive gain");
    }

    /// Bead `sts2-rs-70anj.4`: Ectoplasm listener directly is a no-op
    /// without ownership and does not set the cancel flag.
    #[test]
    fn listener_ectoplasm_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut cancel = false;
        let res = listener_ectoplasm_should_gain_gold(&mut rs, 50, &mut cancel);
        assert!(matches!(res, EffectResult::NotImplemented));
        assert!(!cancel, "no veto without ownership");
    }

    /// Bead `sts2-rs-70anj.4`: PaelsGrowth enchants exactly 1 eligible
    /// deck card with `Clone(4)`. Rest-site CloneOption leg is deferred.
    #[test]
    fn listener_paels_growth_enchants_one_card_with_clone_four() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        rs.relics.push(RelicId::PaelsGrowth);
        rs.last_obtained_relic = Some(RelicId::PaelsGrowth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let cloned = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Clone) == 4)
            .count();
        assert_eq!(cloned, 1, "exactly 1 card enchanted with Clone(4)");
    }

    /// Bead `sts2-rs-70anj.4`: PaelsGrowth listener is a no-op without
    /// ownership.
    #[test]
    fn listener_paels_growth_is_noop_without_ownership() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        let res = listener_paels_growth(&mut rs);
        assert!(matches!(res, EffectResult::NotImplemented));
        assert_eq!(rs.master_deck[0].enchantment(EnchantmentId::Clone), 0);
    }

    /// Bead `sts2-rs-70anj.4`: WingCharm tags exactly one random card-
    /// reward option with `Swift(1)`.
    #[test]
    fn listener_wing_charm_enchants_exactly_one_option_with_swift_one() {
        use sts2_core::cards::{Card, CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WingCharm);
        let mut options: Vec<Card> = (0..3)
            .map(|_| {
                let mut c = Card::pod_defaults();
                c.card_id = CardId::StrikeIronclad;
                c
            })
            .collect();
        let res = listener_wing_charm_card_reward(&mut rs, &mut options);
        assert!(matches!(res, EffectResult::Resolved));
        let swift_count = options
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Swift) == WING_CHARM_SWIFT)
            .count();
        assert_eq!(swift_count, 1, "exactly one option enchanted");
    }

    /// C# draws WingCharm's pick off `Rng.Niche` over the Swift-eligible
    /// options only (`WingCharm.cs:24-43`): an already-enchanted option is
    /// skipped and the Rewards stream is untouched.
    #[test]
    fn listener_wing_charm_draws_niche_over_eligible_options() {
        use sts2_core::cards::{Card, CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WingCharm);
        let mut options: Vec<Card> = (0..2)
            .map(|_| {
                let mut c = Card::pod_defaults();
                c.card_id = CardId::StrikeIronclad;
                c
            })
            .collect();
        options[0].add_enchantment(EnchantmentId::Nimble, 2);
        let rewards = rs.rng.rewards.counter();
        let niche = rs.rng.niche.counter();
        let res = listener_wing_charm_card_reward(&mut rs, &mut options);
        assert!(matches!(res, EffectResult::Resolved));
        assert_eq!(
            options[1].enchantment(EnchantmentId::Swift),
            WING_CHARM_SWIFT
        );
        assert_eq!(options[0].enchantment(EnchantmentId::Swift), 0);
        assert_eq!(rs.rng.rewards.counter(), rewards);
        assert_eq!(rs.rng.niche.counter(), niche + 1);
    }

    /// Bead `sts2-rs-70anj.4`: WingCharm is a no-op when the option
    /// list is empty.
    #[test]
    fn listener_wing_charm_is_noop_on_empty_options() {
        use sts2_core::cards::Card;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WingCharm);
        let mut options: Vec<Card> = vec![];
        let res = listener_wing_charm_card_reward(&mut rs, &mut options);
        assert!(matches!(res, EffectResult::NotImplemented));
    }

    /// Bead `sts2-rs-70anj.4`: TriBoomerang enchants up to
    /// `TRI_BOOMERANG_CARDS = 3` Attack cards with `Instinct(1)`
    /// (a 2× powered-attack damage multiplier; no cost change).
    #[test]
    fn listener_tri_boomerang_enchants_three_attacks_with_instinct_one() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            push_deck_card(&mut rs, CardId::StrikeIronclad);
        }
        let cost_before = rs.master_deck[0].cost;
        rs.relics.push(RelicId::TriBoomerang);
        rs.last_obtained_relic = Some(RelicId::TriBoomerang);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let enchanted = rs
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Instinct) == 1)
            .count();
        assert_eq!(enchanted, TRI_BOOMERANG_CARDS);
        // C# `Instinct.cs` is a 2× powered-attack damage multiplier with no
        // cost behavior — enchanting must NOT change the card's cost.
        assert_eq!(rs.master_deck[0].cost, cost_before);
        assert_eq!(
            rs.master_deck[4].cost, cost_before,
            "unenchanted card cost unchanged"
        );
    }

    /// Bead `sts2-rs-70anj.4`: Storybook's `after_obtained` listener
    /// must push exactly one `BrightestFlame` card to `rs.master_deck`.
    #[test]
    fn listener_storybook_adds_one_brightest_flame_to_master_deck_on_obtain() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        rs.relics.push(RelicId::Storybook);
        rs.last_obtained_relic = Some(RelicId::Storybook);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
        assert_eq!(
            rs.master_deck.last().unwrap().card_id,
            CardId::BrightestFlame
        );
    }

    /// Bead `sts2-rs-70anj.4`: Storybook listener is no-op without
    /// ownership.
    #[test]
    fn listener_storybook_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    /// Bead `sts2-rs-5xg1w`: Glitter's
    /// `modify_card_reward_options_late` listener must add Glam=1 to
    /// every card in the reward options list when owned.
    #[test]
    fn listener_glitter_applies_glam_to_all_card_reward_options() {
        use sts2_core::cards::{Card, CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Glitter);
        let mut cards: Vec<Card> = [CardId::StrikeIronclad, CardId::DefendIronclad]
            .iter()
            .map(|id| {
                let mut c = Card::pod_defaults();
                c.card_id = *id;
                c
            })
            .collect();
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(cards
            .iter()
            .all(|c| c.enchantment(EnchantmentId::Glam) == 1));
    }

    /// Bead `sts2-rs-5xg1w`: Glitter listener is noop without
    /// ownership.
    #[test]
    fn listener_glitter_is_noop_without_ownership() {
        use sts2_core::cards::{Card, CardId, EnchantmentId};
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut cards: Vec<Card> = [CardId::StrikeIronclad]
            .iter()
            .map(|id| {
                let mut c = Card::pod_defaults();
                c.card_id = *id;
                c
            })
            .collect();
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert_eq!(cards[0].enchantment(EnchantmentId::Glam), 0);
    }

    /// Bead `sts2-rs-psf9`: Orrery listener is noop without ownership.
    #[test]
    fn listener_orrery_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.pending_rewards.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    /// Bead `sts2-rs-r4pr`: both r4pr listeners must be noop when their
    /// relic is not in `rs.relics`. Combined into one test to keep the
    /// fixture count tight.
    #[test]
    fn listener_r4pr_relics_are_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max);
        assert_eq!(rs.current_hp, before_hp);
    }

    /// No-op when the relic is not yet in `rs.relics`.
    #[test]
    fn listener_lees_waffle_is_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max);
        assert_eq!(rs.current_hp, before_hp);
    }

    // -----------------------------------------------------------------
    // PaelsFlesh listener tests — bead `sts2-rs-p6dj`
    // -----------------------------------------------------------------
    //
    // Single-channel `after_side_turn_start` listener: from player
    // round 3 onward, the player gains +1 energy each player turn.
    // Tests stand up a minimal CombatState with a player creature and
    // exercise the listener end-to-end via the dispatch bus.

    #[cfg(test)]
    fn paels_flesh_setup(state: &mut CombatState, round: u16, equip: bool) {
        use std::num::NonZeroU32;
        use sts2_core::Relic;
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        state.player.energy = 3;
        state.round = round;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        if equip {
            state.relics.push(Relic {
                id: NonZeroU32::new(1).expect("non-zero"),
                kind: RelicKind::PaelsFlesh,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
    }

    /// **PaelsFlesh #1** — round 3 player turn-start grants +1 energy.
    #[test]
    fn listener_paels_flesh_grants_energy_on_round_three_player_turn() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        paels_flesh_setup(&mut state, 3, true);
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, before + PAELS_FLESH_ENERGY);
    }

    /// **PaelsFlesh #2** — no-op on player rounds 1 / 2 (below the
    /// `round_min` gate).
    #[test]
    fn listener_paels_flesh_is_noop_before_round_three() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        for round in [1u16, 2] {
            let mut state = CombatState::default();
            paels_flesh_setup(&mut state, round, true);
            let before = state.player.energy;
            let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
            assert_eq!(state.player.energy, before, "round {round} must not fire");
        }
    }

    /// **PaelsFlesh #3** — enemy-side turn-start is a no-op even on
    /// round 3+. The listener gates on `side == Player`.
    #[test]
    fn listener_paels_flesh_is_noop_on_enemy_side() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        paels_flesh_setup(&mut state, 5, true);
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.energy, before);
    }

    /// **PaelsFlesh #4** — no-op when the relic is not owned, even
    /// on round 3+. Required because the dispatch bus iterates every
    /// registered listener regardless of ownership.
    #[test]
    fn listener_paels_flesh_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        paels_flesh_setup(&mut state, 7, false);
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, before);
    }

    /// SealOfGold #1 — each player turn, while combat gold ≥ 5, spend 5 for +1
    /// energy (C# `SealOfGold.AfterSideTurnStart`). Bead `sts2-rs-RP17`.
    #[test]
    fn listener_seal_of_gold_spends_gold_for_energy() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::SealOfGold);
        state.gold = 12;
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.gold, 7, "spends 5 combat gold");
        assert_eq!(
            state.player.energy,
            before + SEAL_OF_GOLD_ENERGY,
            "gains 1 energy"
        );
    }

    /// v0.111.0: SealOfGold spends 3 (`GoldVar(3)`, decomp SealOfGold.cs:20),
    /// and 3 gold is now enough.
    #[test]
    fn listener_seal_of_gold_spends_gold_for_energy_v0111() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        equip_relic(&mut state, RelicKind::SealOfGold);
        state.gold = 3;
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.gold, 0, "spends 3 combat gold");
        assert_eq!(state.player.energy, before + 1, "gains 1 energy");
    }

    /// SealOfGold #2 — no spend / no energy when gold < 5 (C# `Gold >= 5` gate).
    #[test]
    fn listener_seal_of_gold_noop_when_below_cost() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::SealOfGold);
        state.gold = SEAL_OF_GOLD_COST - 1;
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.gold, SEAL_OF_GOLD_COST - 1, "no spend below the gate");
        assert_eq!(state.player.energy, before, "no energy below the gate");
    }

    /// Bread #1 — turn 1 loses `BREAD_LOSE_ENERGY = 2` energy (C#
    /// `Bread.cs:AfterSideTurnStart`, gate `TurnNumber == 1`).
    /// 2026-07-17 soft-drop census fix.
    #[test]
    fn listener_bread_turn_one_loses_two_energy() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Bread);
        state.round = 1;
        state.player.energy = 3;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, 1, "turn 1: 3 energy - 2 = 1");
    }

    /// Bread #2 — no energy loss from turn 2 onward, and none on the
    /// enemy side (`participants.Contains(Owner.Creature)` gate).
    #[test]
    fn listener_bread_noop_after_turn_one_and_on_enemy_side() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Bread);
        state.round = 2;
        state.player.energy = 4;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, 4, "turn 2+: no penalty");
        state.round = 1;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.energy, 4, "enemy side: no penalty");
    }

    /// Bread #3 — the loss clamps at 0 (C# `PlayerCombatState.LoseEnergy`
    /// clamps: `Math.Clamp(Energy - amount, 0, …)`).
    #[test]
    fn listener_bread_energy_loss_clamps_at_zero() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Bread);
        state.round = 1;
        state.player.energy = 1;
        assert_eq!(
            super::listener_bread(&mut state, Side::Player),
            EffectResult::Resolved
        );
        assert_eq!(state.player.energy, 0, "1 - 2 clamps to 0, not -1");
    }

    // -----------------------------------------------------------------
    // Sai + VeryHotCocoa listener tests — bead `sts2-rs-70anj.1`
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn equip_relic(state: &mut CombatState, kind: RelicKind) {
        use std::num::NonZeroU32;
        use sts2_core::Relic;
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        let id = NonZeroU32::new(state.relics.len() as u32 + 1).expect("non-zero");
        state.relics.push(Relic {
            id,
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }

    #[test]
    fn listener_sai_grants_block_on_player_turn() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Sai);
        let before = state.player.block;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.block, before + SAI_BLOCK);
    }

    #[test]
    fn listener_sai_is_noop_on_enemy_side() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Sai);
        let before = state.player.block;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.block, before);
    }

    #[test]
    fn listener_sai_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsFlesh); // wrong relic
        let before = state.player.block;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.block, before);
    }

    #[test]
    fn listener_very_hot_cocoa_grants_energy_on_round_one() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::VeryHotCocoa);
        state.round = 1;
        state.player.energy = 3;
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, before + VERY_HOT_COCOA_ENERGY);
    }

    #[test]
    fn listener_very_hot_cocoa_is_noop_after_round_one() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        for round in [2u16, 3, 5, 10] {
            let mut state = CombatState::default();
            equip_relic(&mut state, RelicKind::VeryHotCocoa);
            state.round = round;
            state.player.energy = 3;
            let before = state.player.energy;
            let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
            assert_eq!(state.player.energy, before, "round {round} must not fire");
        }
    }

    #[test]
    fn listener_very_hot_cocoa_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsFlesh);
        state.round = 1;
        state.player.energy = 3;
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, before);
    }

    // -----------------------------------------------------------------
    // Wave 1 body-fill batch 2 listener tests — bead `sts2-rs-70anj.1`
    // -----------------------------------------------------------------

    #[test]
    fn listener_iron_club_draws_every_fourth_card() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_card_played, CardEventCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::IronClub);
        // Seed the draw pile with at least 4 cards so the eventual
        // draw_into_hand call has something to move.
        for _ in 0..6 {
            let card = Card {
                card_id: CardId::StrikeIronclad,
                card_type: CardType::Attack,
                ..Card::pod_defaults()
            };
            let _ = state.add_card_to_draw_pile(card);
        }
        let ctx = CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let hand_before = state.player.hand.len();
        // 3 plays — nothing drawn.
        for _ in 0..3 {
            let _ = dispatch_after_card_played(&mut state, &ctx);
        }
        assert_eq!(state.player.hand.len(), hand_before);
        // 4th play triggers draw 1.
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "IronClub must draw 1 card on the 4th play"
        );
        // 5th-7th: silent again; 8th draws again.
        for _ in 0..3 {
            let _ = dispatch_after_card_played(&mut state, &ctx);
        }
        assert_eq!(state.player.hand.len(), hand_before + 1);
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 2,
            "IronClub must draw 1 card on the 8th play"
        );
    }

    /// Relic `AfterCardPlayed` listeners fire in EQUIP order: C#
    /// `Hook.AfterCardPlayed` walks `CombatState.IterateHookListeners`, which
    /// lists each player's relics in `player.Relics` (obtain) order
    /// (CombatState.cs:118-140, both builds). Ironclad 4029 act 3: Iron Club
    /// obtained before Mummified Hand, a Power played as the 4th card — Iron
    /// Club draws first (`IronClub.cs:83-93`), so Mummified Hand
    /// (`MummifiedHand.cs:16-46`) picks from a hand holding the drawn card.
    /// Rust walked `RelicId` order (Mummified Hand first) and picked from the
    /// pre-draw hand. Obtained the other way round, the draw comes after.
    #[test]
    fn after_card_played_relics_fire_in_equip_order_iron_club_mummified_hand() {
        use std::num::NonZeroU32;
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_card_played, CardEventCtx};
        use sts2_core::Relic;
        crate::relics::dispatch::ensure_installed();
        for iron_club_first in [true, false] {
            let mut state = CombatState::default();
            state.player.current_hp = 80;
            state.player.max_hp = 80;
            let _ = state.init_player_creature();
            state.sync_player_creature();
            let order = if iron_club_first {
                [RelicKind::IronClub, RelicKind::MummifiedHand]
            } else {
                [RelicKind::MummifiedHand, RelicKind::IronClub]
            };
            for (i, kind) in order.into_iter().enumerate() {
                state.relics.push(Relic {
                    id: NonZeroU32::new(u32::try_from(i).unwrap() + 1).expect("non-zero"),
                    kind,
                    // Iron Club: three plays already, this one is the 4th.
                    counter: if kind == RelicKind::IronClub { 3 } else { 0 },
                    is_wax: false,
                    is_melted: false,
                });
            }
            let power = state.add_card_to_draw_pile(Card {
                card_id: CardId::Inflame,
                card_type: CardType::Power,
                cost: sts2_core::CardEnergyCost::new(1),
                ..Card::pod_defaults()
            });
            state.player.draw.retain(|u| *u != power);
            state.player.play.push(power);
            let strike = state.add_card_to_draw_pile(Card {
                card_id: CardId::StrikeIronclad,
                card_type: CardType::Attack,
                cost: sts2_core::CardEnergyCost::new(1),
                ..Card::pod_defaults()
            });
            let ctx = CardEventCtx {
                card_id: CardId::Inflame,
                card_uid: Some(power),
                target: None,
                card_tags: sts2_core::cards::CardTags::NONE,
                energy_spent: 1,
            };
            let _ = dispatch_after_card_played(&mut state, &ctx);
            assert_eq!(state.player.hand.as_slice(), [strike].as_slice());
            let free = state
                .cards
                .iter()
                .find(|c| c.uid == strike)
                .expect("strike")
                .cost
                .has(sts2_core::LocalCostMod::FREE_THIS_TURN);
            assert_eq!(
                free, iron_club_first,
                "Mummified Hand frees the drawn card iff Iron Club fired first \
                 (iron_club_first={iron_club_first})"
            );
        }
    }

    /// C# `BrilliantScarf.AfterCardPlayed` counts the play in its own slot of
    /// the relic `AfterCardPlayed` walk (BrilliantScarf.cs:89-105, both
    /// builds). Obtained before Mummified Hand, with three plays counted and
    /// the 4th a Power, the Scarf makes every hand card globally free before
    /// Mummified Hand rolls (MummifiedHand.cs:18-42): pools 1 and 2 are empty
    /// and the roll lands on pool 3, which still offers the card already
    /// free. Obtained after, the count is 3 when Mummified Hand rolls and
    /// pool 1 (cards that still cost) is used. Defect random loadout v0.111.0
    /// seed 5416.
    #[test]
    fn brilliant_scarf_counts_in_its_relic_slot_before_mummified_hand() {
        use std::num::NonZeroU32;
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::CardEventCtx;
        use sts2_core::Relic;
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            for scarf_first in [true, false] {
                let mut state = CombatState {
                    game_version: version,
                    ..CombatState::default()
                };
                state.player.current_hp = 80;
                state.player.max_hp = 80;
                let _ = state.init_player_creature();
                state.sync_player_creature();
                let order = if scarf_first {
                    [RelicKind::BrilliantScarf, RelicKind::MummifiedHand]
                } else {
                    [RelicKind::MummifiedHand, RelicKind::BrilliantScarf]
                };
                for (i, kind) in order.into_iter().enumerate() {
                    state.relics.push(Relic {
                        id: NonZeroU32::new(u32::try_from(i).unwrap() + 1).expect("nz"),
                        kind,
                        counter: if kind == RelicKind::BrilliantScarf {
                            3
                        } else {
                            0
                        },
                        is_wax: false,
                        is_melted: false,
                    });
                }
                let power = state.add_card_to_draw_pile(Card {
                    card_id: CardId::Inflame,
                    card_type: CardType::Power,
                    cost: sts2_core::CardEnergyCost::new(1),
                    ..Card::pod_defaults()
                });
                state.player.draw.retain(|u| *u != power);
                state.player.play.push(power);
                let mut hand = Vec::new();
                for id in [CardId::StrikeIronclad, CardId::DefendIronclad] {
                    let uid = state.add_card_to_draw_pile(Card {
                        card_id: id,
                        card_type: CardType::Attack,
                        cost: sts2_core::CardEnergyCost::new(1),
                        ..Card::pod_defaults()
                    });
                    state.player.draw.retain(|u| *u != uid);
                    hand.push(uid);
                }
                // The already-free card sits where a two-card roll lands.
                let k = usize::try_from(
                    state
                        .run_rng
                        .combat_card_selection
                        .clone()
                        .next_int_exclusive(0, 2),
                )
                .expect("index");
                let (free, other) = (hand[k], hand[1 - k]);
                state.player.hand.extend(hand.iter().copied());
                state.cards[free.0.get() as usize - 1].set_to_free_this_turn();
                let ctx = CardEventCtx {
                    card_id: CardId::Inflame,
                    card_uid: Some(power),
                    target: None,
                    card_tags: sts2_core::cards::CardTags::NONE,
                    energy_spent: 1,
                };
                // A hand play's relic walk (`IsAutoPlay` clear).
                let _ = sts2_core::relic_effects::dispatch_after_card_played(&mut state, &ctx);
                let scarf = state
                    .relics
                    .iter()
                    .find(|r| r.kind == RelicKind::BrilliantScarf)
                    .expect("scarf");
                assert_eq!(scarf.counter, 4, "{version:?} scarf_first={scarf_first}");
                let other_freed = state.cards[other.0.get() as usize - 1]
                    .cost
                    .has(sts2_core::LocalCostMod::FREE_THIS_TURN);
                assert_eq!(
                    other_freed, !scarf_first,
                    "{version:?}: pool 3 (already-free card picked) iff the Scarf counted \
                     first (scarf_first={scarf_first})"
                );
            }
        }
    }

    /// The Scarf skips an auto-play's `AfterCardPlayed` (`cardPlay.IsAutoPlay`).
    #[test]
    fn brilliant_scarf_skips_auto_plays() {
        use sts2_core::cards::{CardId, CardTags};
        use sts2_core::relic_effects::CardEventCtx;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BrilliantScarf);
        let ctx = CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: CardTags::NONE,
            energy_spent: 0,
        };
        let hook_ctx = sts2_core::hooks::CardCtx {
            card_id: Some(CardId::StrikeIronclad),
            card_uid: None,
            card_tags: CardTags::NONE,
            card_type: None,
        };
        sts2_core::combat_loop::dispatch_after_card_played_buses(&mut state, &ctx, &hook_ctx);
        let count = |s: &CombatState| {
            s.relics
                .iter()
                .find(|r| r.kind == RelicKind::BrilliantScarf)
                .expect("scarf")
                .counter
        };
        assert_eq!(count(&state), 0, "an auto-play does not count");
        let _ = sts2_core::relic_effects::dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(count(&state), 1, "a hand play counts");
    }

    /// IronClub's draw is a card-effect `CardPileCmd.Draw`, so NoDraw (Battle
    /// Trance) vetoes it (`Hook.ShouldDraw`, CardPileCmd.cs:826).
    #[test]
    fn listener_iron_club_draw_respects_no_draw() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_card_played, CardEventCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::IronClub);
        if state.player.creature_uid.is_none() {
            let _ = state.init_player_creature();
        }
        for _ in 0..6 {
            let card = Card {
                card_id: CardId::StrikeIronclad,
                card_type: CardType::Attack,
                ..Card::pod_defaults()
            };
            let _ = state.add_card_to_draw_pile(card);
        }
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            sts2_core::powers::PowerKind::NoDraw,
            1,
        );
        let ctx = CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let hand_before = state.player.hand.len();
        for _ in 0..4 {
            let _ = dispatch_after_card_played(&mut state, &ctx);
        }
        assert_eq!(
            state.player.hand.len(),
            hand_before,
            "NoDraw blocks the 4th-play draw"
        );
    }

    #[test]
    fn listener_toasty_mittens_exhausts_and_grants_strength() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::powers::PowerKind;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::ToastyMittens);
        let card = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        let uid = state.add_card_to_draw_pile(card);
        let draw_before = state.player.draw.len();
        // C# fires this on `BeforeHandDraw` (before the natural draw), not on
        // the post-draw turn-start walk. Bead `sts2-rs-70anj.1`.
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(
            state.player.draw.len(),
            draw_before - 1,
            "ToastyMittens must exhaust the top card of the draw pile"
        );
        assert!(state.player.exhaust.contains(&uid));
        // Player gains +1 Strength.
        let player_uid = state.player.creature_uid.expect("player uid");
        let creature = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .expect("player creature");
        let strength = creature
            .powers
            .iter()
            .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
            .find(|p| p.kind == PowerKind::Strength)
            .map_or(0, |p| p.amount);
        assert_eq!(strength, TOASTY_MITTENS_STRENGTH);
    }

    /// v0.107.1 `ToastyMittens.BeforeHandDraw` awaits `ShuffleIfNecessary`
    /// (ToastyMittens.cs:35), and with it a Stratagem pick the reshuffle
    /// opens, BEFORE it exhausts the top draw card and gains Strength. Rust
    /// exhausted and granted the Strength under the open pick, so the
    /// exhausted card was not the one C# exhausts (act23 random-loadout
    /// Ironclad v0.107.1 seed 1131).
    #[test]
    fn listener_toasty_mittens_waits_for_its_shuffle_stratagem_pick() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::relic_effects::dispatch_before_hand_draw_from;
        use sts2_core::{CardUid, Power, PowerKind, PowerUid};
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.round = 2;
        equip_relic(&mut state, RelicKind::ToastyMittens);
        if state.player.creature_uid.is_none() {
            let _ = state.init_player_creature();
        }
        let pid = std::num::NonZeroU32::new(50).expect("nz");
        state.powers.push(Power::new(pid, PowerKind::Stratagem, 1));
        state.player.powers.push(PowerUid(pid));
        for i in 1..=4u32 {
            let uid = CardUid::new(i).expect("non-zero");
            state.cards.push(Card {
                uid,
                card_id: CardId::StrikeIronclad,
                ..Card::pod_defaults()
            });
            state.player.discard.push(uid);
        }

        let next = dispatch_before_hand_draw_from(&mut state, 0);
        assert!(next.is_some(), "the walk suspends on the Stratagem pick");
        assert!(state.pending_choice.is_some());
        assert!(state.player.exhaust.is_empty(), "no exhaust under the pick");
        assert_eq!(player_strength(&state), 0, "no Strength under the pick");
        assert_eq!(state.player.draw.len(), 4);

        let pick = state.pending_choice.as_ref().expect("pick").options[0]
            .card_uid
            .expect("uid");
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::ResolveChoice {
                selection: smallvec::smallvec![0],
            },
        );
        assert_eq!(state.player.hand.as_slice(), &[pick], "the pick in hand");
        assert_eq!(state.player.exhaust.len(), 1, "then the exhaust");
        assert_eq!(state.player.draw.len(), 2);
        assert_eq!(player_strength(&state), TOASTY_MITTENS_STRENGTH);
    }

    /// Toasty Mittens exhausts the last draw card; Dark Embrace's draw then
    /// reshuffles and opens the Stratagem pick. C# awaits the pick inside that
    /// draw, the draw inside the exhaust hook walk (powers, then relics), and
    /// the exhaust inside Mittens: the pick lands first, then Dark Embrace's
    /// card, then Joss Paper's, and only then the Strength. Rust ran Joss
    /// Paper's draw and the Strength under the open pick (random-loadout
    /// v0.107.1 seed 1016: hand [DEFEND, pick, ...] vs C# [pick, DEFEND, ...]).
    #[test]
    fn mittens_dark_embrace_reshuffle_pick_lands_before_the_draw() {
        use sts2_core::cards::{Card, CardId};
        use sts2_core::relic_effects::dispatch_before_hand_draw_from;
        use sts2_core::{CardUid, Power, PowerKind, PowerUid};
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.round = 2;
        equip_relic(&mut state, RelicKind::ToastyMittens);
        state.relics.push(sts2_core::Relic {
            id: std::num::NonZeroU32::new(2).expect("nz"),
            kind: RelicKind::JossPaper,
            counter: 4,
            is_wax: false,
            is_melted: false,
        });
        if state.player.creature_uid.is_none() {
            let _ = state.init_player_creature();
        }
        let pid = std::num::NonZeroU32::new(50).expect("nz");
        state.powers.push(Power::new(pid, PowerKind::Stratagem, 1));
        state.player.powers.push(PowerUid(pid));
        crate::cards::ironclad::helpers::apply_power_to_player(
            &mut state,
            PowerKind::DarkEmbrace,
            1,
        );
        let ids = [
            CardId::TrueGrit,
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Bash,
        ];
        for (i, id) in ids.iter().enumerate() {
            let uid = CardUid::new(u32::try_from(i + 1).expect("u32")).expect("non-zero");
            state.cards.push(Card {
                uid,
                card_id: *id,
                ..Card::pod_defaults()
            });
            if i == 0 {
                state.player.draw.push(uid);
            } else {
                state.player.discard.push(uid);
            }
        }

        let next = dispatch_before_hand_draw_from(&mut state, 0);
        assert!(next.is_some(), "the walk suspends on the Stratagem pick");
        assert!(state.pending_choice.is_some());
        assert!(state.player.hand.is_empty(), "nothing drawn under the pick");
        assert_eq!(player_strength(&state), 0, "no Strength under the pick");
        assert_eq!(
            state
                .relics
                .iter()
                .find(|r| r.kind == RelicKind::JossPaper)
                .map(|r| r.counter),
            Some(4),
            "Joss Paper waits for Dark Embrace's draw"
        );
        let pc = state.pending_choice.as_ref().expect("pick");
        let pick = pc.options[2].card_uid.expect("uid");
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::ResolveChoice {
                selection: smallvec::smallvec![2],
            },
        );
        // The pick, then Dark Embrace's draw, then Joss Paper's.
        assert_eq!(state.player.hand.len(), 3, "hand {:?}", state.player.hand);
        assert_eq!(state.player.hand[0], pick, "the pick lands first");
        assert_eq!(player_strength(&state), TOASTY_MITTENS_STRENGTH);
    }

    fn player_strength(state: &CombatState) -> i32 {
        use sts2_core::powers::PowerKind;
        let player_uid = state.player.creature_uid.expect("player uid");
        let creature = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .expect("player creature");
        creature
            .powers
            .iter()
            .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
            .find(|p| p.kind == PowerKind::Strength)
            .map_or(0, |p| p.amount)
    }

    fn push_hand_strike(state: &mut CombatState) -> sts2_core::CardUid {
        use sts2_core::cards::{Card, CardId, CardType};
        let uid = state.add_card_to_draw_pile(Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        });
        state.move_card_to_hand(uid);
        uid
    }

    /// v0.111.0: ToastyMittens no longer touches the draw pile before the
    /// draw; after turn start it opens a forced pick-1 exhaust over the
    /// hand, and the +1 Strength lands after the exhaust
    /// (decomp ToastyMittens.cs:31-43).
    #[test]
    fn listener_toasty_mittens_exhausts_and_grants_strength_v0111() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{
            dispatch_after_player_turn_start, dispatch_before_hand_draw,
        };
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        equip_relic(&mut state, RelicKind::ToastyMittens);
        let in_draw = state.add_card_to_draw_pile(Card {
            card_id: CardId::DefendIronclad,
            card_type: CardType::Skill,
            ..Card::pod_defaults()
        });
        let _ = dispatch_before_hand_draw(&mut state);
        assert!(state.player.draw.contains(&in_draw), "draw pile untouched");
        assert_eq!(player_strength(&state), 0);

        let a = push_hand_strike(&mut state);
        let b = push_hand_strike(&mut state);
        let _ = dispatch_after_player_turn_start(&mut state);
        let pc = state
            .pending_choice
            .as_ref()
            .expect("exhaust select opened");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::ExhaustFromHand);
        assert_eq!((pc.min_select, pc.max_select), (1, 1));
        assert_eq!(player_strength(&state), 0, "Strength waits for the pick");
        let pick = u8::try_from(
            pc.options
                .iter()
                .position(|o| o.card_uid == Some(b))
                .expect("b"),
        )
        .expect("u8");
        let _ = sts2_core::resolve_choice_atomic_apply(&mut state, &[pick]);
        assert!(state.player.exhaust.contains(&b));
        assert!(state.player.hand.contains(&a));
        assert_eq!(player_strength(&state), TOASTY_MITTENS_STRENGTH);
    }

    /// v0.111.0: a one-card hand is auto-taken (`FromHand` fast path) and an
    /// empty hand still grants the Strength.
    #[test]
    fn listener_toasty_mittens_auto_takes_small_hands_v0111() {
        use sts2_core::relic_effects::dispatch_after_player_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        equip_relic(&mut state, RelicKind::ToastyMittens);
        let _ = dispatch_after_player_turn_start(&mut state);
        assert!(state.pending_choice.is_none());
        assert_eq!(player_strength(&state), TOASTY_MITTENS_STRENGTH);

        let only = push_hand_strike(&mut state);
        let _ = dispatch_after_player_turn_start(&mut state);
        assert!(state.pending_choice.is_none(), "single card auto-taken");
        assert!(state.player.exhaust.contains(&only));
        assert_eq!(player_strength(&state), 2 * TOASTY_MITTENS_STRENGTH);
    }

    #[test]
    fn listener_toasty_mittens_skips_innate_on_round_one() {
        // C# `ToastyMittens.BeforeHandDraw`: on round 1 it exhausts the first
        // NON-Innate draw card (`cards.FirstOrDefault(!Innate)`), not the literal
        // top card. Bead `sts2-rs-70anj.1`.
        use sts2_core::cards::{Card, CardId, CardType, KeywordOverride};
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.round = 1;
        equip_relic(&mut state, RelicKind::ToastyMittens);
        // Top of draw is an Innate card; the next card is a normal Strike.
        let innate = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            keyword_override: Some(KeywordOverride::INNATE),
            ..Card::pod_defaults()
        };
        let normal = Card {
            card_id: CardId::DefendIronclad,
            card_type: CardType::Skill,
            ..Card::pod_defaults()
        };
        // `add_card_to_draw_pile` front-inserts, so add the normal card first
        // (lands below) then the innate card (lands on top).
        let normal_uid = state.add_card_to_draw_pile(normal);
        let innate_uid = state.add_card_to_draw_pile(innate);
        let _ = dispatch_before_hand_draw(&mut state);
        assert!(
            state.player.exhaust.contains(&normal_uid),
            "round-1 ToastyMittens must skip the Innate top card and exhaust the first non-Innate card"
        );
        assert!(
            state.player.draw.contains(&innate_uid),
            "the Innate top card must remain in the draw pile"
        );
    }

    #[test]
    fn listener_jeweled_mask_moves_random_power_to_hand_free() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::JeweledMask);
        state.round = 1;
        // Mix of Attack + Power cards in draw pile.
        for _ in 0..3 {
            let attack = Card {
                card_id: CardId::StrikeIronclad,
                card_type: CardType::Attack,
                ..Card::pod_defaults()
            };
            let _ = state.add_card_to_draw_pile(attack);
        }
        let power = Card {
            card_id: CardId::Inflame,
            card_type: CardType::Power,
            ..Card::pod_defaults()
        };
        let power_uid = state.add_card_to_draw_pile(power);
        let hand_before = state.player.hand.len();
        let ccs_before = state.run_rng.combat_card_selection.counter();
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "JeweledMask must move 1 Power card to hand"
        );
        assert!(state.player.hand.contains(&power_uid));
        let card = &state.cards[power_uid.0.get() as usize - 1];
        assert!(
            card.cost.has(sts2_core::LocalCostMod::FREE_THIS_TURN),
            "Selected Power must have FREE_THIS_TURN"
        );
        // C# `Rng.CombatCardSelection.NextItem(list)` = exactly one
        // `NextInt` tick on the CombatCardSelection stream (Rng.cs:264).
        assert_eq!(
            state.run_rng.combat_card_selection.counter(),
            ccs_before + 1,
            "selection must consume exactly one combat_card_selection read"
        );
    }

    /// v0.111.0: JeweledMask skips Innate Powers while a non-Innate one is
    /// in the draw pile (decomp JeweledMask.cs:27-31).
    #[test]
    fn listener_jeweled_mask_moves_random_power_to_hand_free_v0111() {
        use sts2_core::cards::{Card, CardId, CardType, KeywordOverride};
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        for _ in 0..8 {
            let mut state = CombatState {
                game_version: sts2_core::GameVersion::V0_111_0,
                ..CombatState::default()
            };
            equip_relic(&mut state, RelicKind::JeweledMask);
            state.round = 1;
            let innate = Card {
                card_id: CardId::Inflame,
                card_type: CardType::Power,
                keyword_override: Some(KeywordOverride::INNATE),
                ..Card::pod_defaults()
            };
            let innate_uid = state.add_card_to_draw_pile(innate);
            let plain = Card {
                card_id: CardId::Inflame,
                card_type: CardType::Power,
                ..Card::pod_defaults()
            };
            let plain_uid = state.add_card_to_draw_pile(plain);
            let _ = dispatch_before_hand_draw(&mut state);
            assert!(
                state.player.hand.contains(&plain_uid),
                "non-Innate Power chosen"
            );
            assert!(
                state.player.draw.contains(&innate_uid),
                "Innate Power stays"
            );
        }
    }

    #[test]
    fn listener_jeweled_mask_is_noop_after_round_one() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::JeweledMask);
        state.round = 2;
        let power = Card {
            card_id: CardId::Inflame,
            card_type: CardType::Power,
            ..Card::pod_defaults()
        };
        let power_uid = state.add_card_to_draw_pile(power);
        let _ = dispatch_before_hand_draw(&mut state);
        assert!(
            state.player.draw.contains(&power_uid),
            "JeweledMask must not fire after round 1"
        );
    }

    #[test]
    fn listener_jeweled_mask_empty_pool_skips_rng_read() {
        // C# guards `list.Count != 0` BEFORE the `NextItem` call — a
        // Power-less draw pile must neither add a card to hand nor
        // consume a CombatCardSelection tick.
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::JeweledMask);
        state.round = 1;
        let attack = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        let _ = state.add_card_to_draw_pile(attack);
        let hand_before = state.player.hand.len();
        let ccs_before = state.run_rng.combat_card_selection.counter();
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(
            state.player.hand.len(),
            hand_before,
            "no Power in draw pile: nothing moves to hand"
        );
        assert_eq!(
            state.run_rng.combat_card_selection.counter(),
            ccs_before,
            "no Power in draw pile: no combat_card_selection read"
        );
    }

    #[test]
    fn listener_burning_sticks_clones_first_skill_exhausted() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_card_exhausted, CardEventCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BurningSticks);
        // Stage a Skill card in the arena (no need to actually exhaust
        // through the engine — the listener only inspects card POD).
        let skill = Card {
            card_id: CardId::DefendIronclad,
            card_type: CardType::Skill,
            ..Card::pod_defaults()
        };
        let skill_uid = state.add_card_to_draw_pile(skill);
        let ctx = CardEventCtx {
            card_id: CardId::DefendIronclad,
            card_uid: Some(skill_uid),
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let hand_before = state.player.hand.len();
        let _ = dispatch_after_card_exhausted(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "BurningSticks must spawn a clone of the Skill into hand"
        );
        // Second Skill exhaust does NOT clone (per-combat latch).
        let _ = dispatch_after_card_exhausted(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "BurningSticks must only fire once per combat"
        );
    }

    /// Stage the Dark Embrace + Burning Sticks exhaust fixture: one Skill in
    /// hand, one Strike on the draw pile, DarkEmbrace 1 and BurningSticks.
    /// Returns `(skill_uid, strike_uid)`.
    fn dark_embrace_burning_sticks_fixture(
        state: &mut CombatState,
    ) -> (sts2_core::CardUid, sts2_core::CardUid) {
        use sts2_core::cards::{move_card, Card, CardId, PileKind};
        crate::relics::dispatch::ensure_installed();
        equip_relic(state, RelicKind::BurningSticks);
        crate::cards::ironclad::helpers::apply_power_to_player(state, PowerKind::DarkEmbrace, 1);
        let skill = state.add_card_to_draw_pile(Card {
            card_id: CardId::DefendIronclad,
            card_type: CardType::Skill,
            ..Card::pod_defaults()
        });
        let _ = move_card(&mut state.player, skill, PileKind::Draw, PileKind::Hand);
        let strike = state.add_card_to_draw_pile(Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        });
        (skill, strike)
    }

    /// C# `Hook.AfterCardExhausted` (v0.111.0 `Hook.cs:381-390`) walks
    /// `IterateHookListeners` (`CombatState.cs:116-131`): the player's
    /// powers, then that player's relics. So DarkEmbracePower draws BEFORE
    /// BurningSticks clones the exhausted Skill (act23
    /// Ironclad-2146/2158: hand order POMMEL_STRIKE, then the clone).
    /// Hand-card exhaust (Second Wind / Fiend Fire / Thrash) path.
    #[test]
    fn exhaust_hand_card_fires_player_powers_before_relics() {
        let mut state = CombatState::default();
        let (skill, strike) = dark_embrace_burning_sticks_fixture(&mut state);
        crate::cards::ironclad::helpers::exhaust_hand_card(&mut state, skill);
        let hand_ids: Vec<_> = state
            .player
            .hand
            .iter()
            .map(|u| state.cards[u.0.get() as usize - 1].card_id)
            .collect();
        assert_eq!(state.player.hand[0], strike, "DarkEmbrace draw lands first");
        assert_eq!(
            hand_ids,
            vec![
                sts2_core::cards::CardId::StrikeIronclad,
                sts2_core::cards::CardId::DefendIronclad
            ],
            "BurningSticks clone lands after the DarkEmbrace draw"
        );
    }

    /// Same C# order for an exhaust out of the Play pile (a play-time
    /// Exhaust card, Corruption): `combat_loop::fire_after_card_exhausted`.
    #[test]
    fn fire_after_card_exhausted_fires_player_powers_before_relics() {
        use sts2_core::cards::{move_card, CardId, PileKind};
        let mut state = CombatState::default();
        let (skill, strike) = dark_embrace_burning_sticks_fixture(&mut state);
        let _ = move_card(&mut state.player, skill, PileKind::Hand, PileKind::Exhaust);
        sts2_core::combat_loop::fire_after_card_exhausted(
            &mut state,
            CardId::DefendIronclad,
            skill,
            None,
            sts2_core::cards::CardTags::NONE,
            CardType::Skill,
        );
        assert_eq!(state.player.hand.len(), 2);
        assert_eq!(state.player.hand[0], strike, "DarkEmbrace draw lands first");
    }

    #[test]
    fn listener_burning_sticks_ignores_attacks() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::{dispatch_after_card_exhausted, CardEventCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BurningSticks);
        let attack = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        let attack_uid = state.add_card_to_draw_pile(attack);
        let ctx = CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: Some(attack_uid),
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let hand_before = state.player.hand.len();
        let _ = dispatch_after_card_exhausted(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before,
            "BurningSticks must ignore non-Skill exhausts"
        );
    }

    #[test]
    fn listener_mystic_lighter_adds_damage_to_enchanted_attacks() {
        use sts2_core::cards::{Card, CardId, CardType, EnchantmentId};
        use sts2_core::hooks::ValueProps;
        use sts2_core::relic_effects::{dispatch_modify_damage_additive, ModifyDamageCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MysticLighter);
        let mut card = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        card.set_enchantment(EnchantmentId::Sharp, 3);
        let card_uid = state.add_card_to_draw_pile(card);
        let player_uid = state.player.creature_uid.expect("player uid");
        let target_uid = player_uid; // any uid — listener only inspects dealer + card
        let ctx = ModifyDamageCtx {
            dealer: Some(player_uid),
            target: target_uid,
            base_damage: 5,
            props: ValueProps::MOVE,
            card: Some(CardId::StrikeIronclad),
            card_uid: Some(card_uid),
        };
        let delta = dispatch_modify_damage_additive(&state, &ctx);
        assert_eq!(delta, MYSTIC_LIGHTER_EXTRA_DAMAGE);
        // C# `IsPoweredAttack` excludes Unpowered damage.
        let unpowered = ModifyDamageCtx {
            props: ValueProps(ValueProps::MOVE.0 | ValueProps::UNPOWERED.0),
            ..ctx
        };
        assert_eq!(dispatch_modify_damage_additive(&state, &unpowered), 0);
    }

    /// Act23 parity Ironclad 1019 (#1054): Pael's Growth put Clone on
    /// master_deck[0], which `enter_combat` places LAST among the Strike
    /// copies in the arena. Mystic Lighter must read the PLAYED instance
    /// (C# `cardSource.Enchantment`, MysticLighter.cs:22), not the first
    /// arena copy sharing its `CardId`: the enchanted copy gets +9 and the
    /// plain copy +0, regardless of arena order.
    #[test]
    fn listener_mystic_lighter_reads_played_instance_with_duplicate_ids() {
        use sts2_core::cards::{Card, CardId, CardType, EnchantmentId};
        use sts2_core::hooks::ValueProps;
        use sts2_core::relic_effects::{dispatch_modify_damage_additive, ModifyDamageCtx};
        crate::relics::dispatch::ensure_installed();
        let strike = || Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        for enchanted_first in [false, true] {
            let mut state = CombatState::default();
            equip_relic(&mut state, RelicKind::MysticLighter);
            let mut enchanted = strike();
            enchanted.set_enchantment(EnchantmentId::Clone, 4);
            let (plain_uid, enchanted_uid) = if enchanted_first {
                let e = state.add_card_to_draw_pile(enchanted);
                (state.add_card_to_draw_pile(strike()), e)
            } else {
                let p = state.add_card_to_draw_pile(strike());
                (p, state.add_card_to_draw_pile(enchanted))
            };
            let player_uid = state.player.creature_uid.expect("player uid");
            let ctx_for = |uid| ModifyDamageCtx {
                dealer: Some(player_uid),
                target: player_uid,
                base_damage: 6,
                props: ValueProps::MOVE,
                card: Some(CardId::StrikeIronclad),
                card_uid: Some(uid),
            };
            assert_eq!(
                dispatch_modify_damage_additive(&state, &ctx_for(enchanted_uid)),
                MYSTIC_LIGHTER_EXTRA_DAMAGE,
                "enchanted copy must get +9 (enchanted_first={enchanted_first})"
            );
            assert_eq!(
                dispatch_modify_damage_additive(&state, &ctx_for(plain_uid)),
                0,
                "plain copy must get +0 (enchanted_first={enchanted_first})"
            );
        }
    }

    /// Loadout Necrobinder 1048: an enchanted card's Osty attack (`FromOsty`,
    /// dealer = the Osty) still gets +9 — C# keys on `cardSource.Owner`,
    /// not the dealer (MysticLighter.cs:22-29). A foreign dealer gets none.
    #[test]
    fn listener_mystic_lighter_applies_to_osty_dealt_hits() {
        use sts2_core::cards::{Card, CardId, CardType, EnchantmentId};
        use sts2_core::hooks::ValueProps;
        use sts2_core::relic_effects::{dispatch_modify_damage_additive, ModifyDamageCtx};
        use sts2_core::CreatureUid;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MysticLighter);
        let mut card = Card {
            card_id: CardId::Rattle,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        card.set_enchantment(EnchantmentId::Vigorous, 8);
        let card_uid = state.add_card_to_draw_pile(card);
        let player_uid = state.player.creature_uid.expect("player uid");
        let osty_uid = CreatureUid::new(77).expect("non-zero");
        state.player.pet_uid = Some(osty_uid);
        let ctx = ModifyDamageCtx {
            dealer: Some(osty_uid),
            target: player_uid,
            base_damage: 7,
            props: ValueProps::MOVE,
            card: Some(CardId::Rattle),
            card_uid: Some(card_uid),
        };
        assert_eq!(
            dispatch_modify_damage_additive(&state, &ctx),
            MYSTIC_LIGHTER_EXTRA_DAMAGE
        );
        let foreign = ModifyDamageCtx {
            dealer: Some(CreatureUid::new(99).expect("non-zero")),
            ..ctx
        };
        assert_eq!(dispatch_modify_damage_additive(&state, &foreign), 0);
    }

    #[test]
    fn listener_mystic_lighter_ignores_non_enchanted_attacks() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::hooks::ValueProps;
        use sts2_core::relic_effects::{dispatch_modify_damage_additive, ModifyDamageCtx};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MysticLighter);
        let card = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        let card_uid = state.add_card_to_draw_pile(card);
        let player_uid = state.player.creature_uid.expect("player uid");
        let ctx = ModifyDamageCtx {
            dealer: Some(player_uid),
            target: player_uid,
            base_damage: 5,
            props: ValueProps::MOVE,
            card: Some(CardId::StrikeIronclad),
            card_uid: Some(card_uid),
        };
        let delta = dispatch_modify_damage_additive(&state, &ctx);
        assert_eq!(delta, 0);
    }

    // -----------------------------------------------------------------
    // SneckoEye / BrilliantScarf / BlessedAntler listener tests —
    // bead `sts2-rs-70anj.1` (Wave 1 batch 3)
    // -----------------------------------------------------------------

    #[test]
    fn listener_snecko_eye_applies_confused_on_combat_start() {
        use sts2_core::powers::PowerKind;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::SneckoEye);
        let _ = dispatch_before_combat_start(&mut state);
        let player_uid = state.player.creature_uid.expect("player uid");
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature");
        let player_power_ids: Vec<_> = state.creatures[player_idx]
            .powers
            .iter()
            .map(|p| p.0)
            .collect();
        let has_confused = state.powers.iter().any(|p| {
            player_power_ids.contains(&p.id) && p.kind == PowerKind::Confused && p.amount >= 1
        });
        assert!(
            has_confused,
            "SneckoEye must apply 1 stack of Confused to the player at combat start"
        );
    }

    #[test]
    fn listener_snecko_eye_grants_extra_hand_draw() {
        use sts2_core::relic_effects::dispatch_modify_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::SneckoEye);
        let base = 5;
        let modified = dispatch_modify_hand_draw(&state, base);
        assert_eq!(modified, base + SNECKO_EYE_EXTRA_DRAW);
    }

    #[test]
    fn listener_snecko_eye_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_modify_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let state = CombatState::default();
        assert_eq!(dispatch_modify_hand_draw(&state, 5), 5);
    }

    /// A hand play's relic walk counts the play; only the player's side-turn
    /// start resets the count.
    #[test]
    fn listener_brilliant_scarf_resets_on_player_turn_only() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::{
            dispatch_after_card_played, dispatch_before_side_turn_start, CardEventCtx,
        };
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BrilliantScarf);
        let counter = |state: &CombatState| {
            state
                .relics
                .iter()
                .find(|r| !r.is_melted && r.kind == RelicKind::BrilliantScarf)
                .map_or(99, |r| r.counter)
        };
        let ctx = CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(counter(&state), 1, "a hand play counts");
        for r in &mut state.relics {
            r.counter = 4;
        }
        let _ = dispatch_before_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(counter(&state), 4, "enemy-turn start must not reset");
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        assert_eq!(counter(&state), 0);
    }

    #[test]
    fn listener_blessed_antler_inserts_three_dazed_on_round_one_hand_draw() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BlessedAntler);
        state.round = 1;
        let draw_before = state.player.draw.len();
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(
            state.player.draw.len(),
            draw_before + BLESSED_ANTLER_DAZED_COUNT as usize,
        );
        // All newly-added cards should be Dazed.
        let dazed_count = state
            .player
            .draw
            .iter()
            .filter(|&&uid| {
                let idx = uid.0.get() as usize - 1;
                state.cards[idx].card_id == CardId::Dazed
            })
            .count();
        assert_eq!(dazed_count, BLESSED_ANTLER_DAZED_COUNT as usize);
    }

    /// Bead `sts2-rs-lfdaa`: C# adds each Dazed at `CardPilePosition.Random`,
    /// which costs one Shuffle-stream read apiece
    /// (`AddInternal(card, Rng.Shuffle.NextInt(Cards.Count + 1))`). The prior
    /// top-insert consumed none, desyncing every later draw/reshuffle in the
    /// combat. Also pins that a Dazed can land BELOW an existing card — a
    /// top-insert would put all three above the seeded pile.
    #[test]
    fn listener_blessed_antler_dazed_land_at_random_depths() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BlessedAntler);
        state.round = 1;
        // Seed a draw pile so the random index has more than one slot.
        for i in 1..=6u32 {
            state
                .player
                .draw
                .push(CardUid::new(i + 500).expect("non-zero"));
        }
        let reads_before = state.run_rng.shuffle.counter();

        let _ = dispatch_before_hand_draw(&mut state);

        assert_eq!(
            state.run_rng.shuffle.counter(),
            reads_before + BLESSED_ANTLER_DAZED_COUNT.unsigned_abs(),
            "one Shuffle read per CardPilePosition.Random insert",
        );
        let is_dazed = |uid: &sts2_core::CardUid| {
            let idx = uid.0.get() as usize - 1;
            state
                .cards
                .get(idx)
                .is_some_and(|c| c.card_id == CardId::Dazed)
        };
        let last_dazed_at = state.player.draw.iter().rposition(is_dazed);
        assert!(
            last_dazed_at.is_some_and(|i| i >= BLESSED_ANTLER_DAZED_COUNT as usize),
            "a random-depth insert must be able to land below the seeded pile, \
             not stack all three on top",
        );
    }

    #[test]
    fn listener_blessed_antler_is_noop_after_round_one() {
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::BlessedAntler);
        state.round = 2;
        let draw_before = state.player.draw.len();
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(state.player.draw.len(), draw_before);
    }

    #[test]
    fn listener_blessed_antler_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::VeryHotCocoa);
        state.round = 1;
        let draw_before = state.player.draw.len();
        let _ = dispatch_before_hand_draw(&mut state);
        assert_eq!(state.player.draw.len(), draw_before);
    }

    /// `TheAbacus` is inert when the player does not own the relic.
    #[test]
    fn the_abacus_inert_when_not_owned() {
        use sts2_core::relic_effects::{dispatch_after_shuffle_bus, ShuffleCtx};

        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        // No relic equipped.
        let _ = dispatch_after_shuffle_bus(&mut state, &ShuffleCtx::default());
        assert_eq!(
            state.player.block, 0,
            "TheAbacus listener must be owner-gated"
        );
    }

    /// Bead `sts2-rs-83sa`: BlackStar appends a `Reward::Relic` on
    /// Elite-room clears. The concrete `RelicId` carried by the
    /// `RelicReward` POD is BlackStar itself (placeholder sentinel for
    /// the deferred "random relic" semantics — see listener doc).
    #[test]
    fn listener_black_star_appends_relic_reward_on_elite_rooms() {
        use sts2_core::run_effects::dispatch_modify_rewards;
        use sts2_core::runstate::{RelicReward, Reward, RoomType, RunState};

        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::BlackStar);
        let mut rewards: Vec<Reward> = Vec::new();
        let _ = dispatch_modify_rewards(&mut rs, &mut rewards, RoomType::Elite);
        assert_eq!(rewards.len(), 1);
        assert_eq!(
            rewards[0],
            Reward::Relic(RelicReward::new(RelicId::BlackStar))
        );
    }

    /// Bead `sts2-rs-83sa`: BlackStar is silent on every non-Elite
    /// room — Monster rooms in particular are PrayerWheel's territory.
    #[test]
    fn listener_black_star_is_noop_on_non_elite_rooms() {
        use sts2_core::run_effects::dispatch_modify_rewards;
        use sts2_core::runstate::{Reward, RoomType, RunState};

        crate::relics::dispatch::ensure_installed();
        for rt in [
            RoomType::Monster,
            RoomType::Boss,
            RoomType::Shop,
            RoomType::Treasure,
            RoomType::Event,
            RoomType::RestSite,
            RoomType::Unknown,
        ] {
            let mut rs = RunState::new();
            rs.relics.push(RelicId::BlackStar);
            let mut rewards: Vec<Reward> = Vec::new();
            let _ = dispatch_modify_rewards(&mut rs, &mut rewards, rt);
            assert!(
                rewards.is_empty(),
                "BlackStar must be silent on {rt:?} rooms"
            );
        }
    }

    /// Bead `sts2-rs-83sa`: BlackStar listener is a no-op when the
    /// player does not own the relic.
    #[test]
    fn listener_black_star_is_noop_without_ownership() {
        use sts2_core::run_effects::dispatch_modify_rewards;
        use sts2_core::runstate::{Reward, RoomType, RunState};

        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let mut rewards: Vec<Reward> = Vec::new();
        let _ = dispatch_modify_rewards(&mut rs, &mut rewards, RoomType::Elite);
        assert!(rewards.is_empty());
    }

    // ─── 70anj.4.1: run-state-only no-op pins (bead `sts2-rs-qsxiy`) ─
    //
    // Five `shop_event.rs` relics whose only Python-side method is a
    // run-state hook (`after_obtained` for at-pickup pickups,
    // `modify_card_reward_creation_options` for reward-pool mutators).
    // Their `on_trigger_*` bodies are intentionally trivial — wiring
    // real combat-side logic would be a parity divergence from the
    // Python pin. These tests pin that contract so a future
    // body-fill pass doesn't accidentally introduce combat-side
    // behavior under the same RelicId.

    /// `Cauldron` — combat dispatch is a no-op by design. Future
    /// body-fill targets the run-state at-obtain channel
    /// (`POTIONS = 5` already pinned by [`CAULDRON_POTIONS`]).
    #[test]
    fn cauldron_combat_dispatch_is_noop() {
        assert_eq!(CAULDRON_POTIONS, 5);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Cauldron.to_python_name(), "CAULDRON");
    }

    /// `DingyRug` — combat dispatch is a no-op by design. The
    /// effect lives on the run-state card-reward channel
    /// (`modify_card_reward_creation_options` in Python).
    #[test]
    fn dingy_rug_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::DingyRug.to_python_name(), "DINGY_RUG");
    }

    /// `DollysMirror` — combat dispatch is a no-op by design.
    /// Future body-fill mutates the run-state master deck at obtain.
    #[test]
    fn dollys_mirror_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::DollysMirror.to_python_name(), "DOLLYS_MIRROR");
    }

    /// `Orrery` — combat dispatch is a no-op by design. Future
    /// body-fill queues five card rewards on the run-state at-obtain
    /// channel (count pinned by [`ORRERY_CARDS`]).
    #[test]
    fn orrery_combat_dispatch_is_noop() {
        assert_eq!(ORRERY_CARDS, 5);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Orrery.to_python_name(), "ORRERY");
    }

    /// `GnarledHammer` — combat dispatch is a no-op by design.
    /// Future body-fill enchants up to 3 deck cards with Sharp(3)
    /// at obtain (counts pinned by [`GNARLED_HAMMER_CARDS`] +
    /// [`GNARLED_HAMMER_SHARP`]).
    #[test]
    fn gnarled_hammer_combat_dispatch_is_noop() {
        assert_eq!(GNARLED_HAMMER_CARDS, 3);
        assert_eq!(GNARLED_HAMMER_SHARP, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::GnarledHammer.to_python_name(), "GNARLED_HAMMER");
    }

    // ─── 70anj.4.2: second no-op pin batch (bead `sts2-rs-4nyt7`) ──

    /// `Kifuda` — combat dispatch is a no-op by design. Future body-
    /// fill enchants up to 3 deck cards with Adroit(3) at obtain
    /// (counts pinned by [`KIFUDA_CARDS`] + [`KIFUDA_ADROIT`]).
    #[test]
    fn kifuda_combat_dispatch_is_noop() {
        assert_eq!(KIFUDA_CARDS, 3);
        assert_eq!(KIFUDA_ADROIT, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Kifuda.to_python_name(), "KIFUDA");
    }

    /// `MiniatureTent` — combat dispatch is a no-op by design. The
    /// effect (`should_disable_remaining_rest_site_options`) is a
    /// run-state map-screen predicate and never fires in combat.
    #[test]
    fn miniature_tent_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::MiniatureTent.to_python_name(), "MINIATURE_TENT");
    }

    /// `PunchDagger` — combat dispatch is a no-op by design. Future
    /// body-fill enchants 1 deck card with Momentum(5) at obtain
    /// (amount pinned by [`PUNCH_DAGGER_MOMENTUM`]).
    #[test]
    fn punch_dagger_combat_dispatch_is_noop() {
        assert_eq!(PUNCH_DAGGER_MOMENTUM, 5);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::PunchDagger.to_python_name(), "PUNCH_DAGGER");
    }

    /// `BeautifulBracelet` — combat dispatch is a no-op by design.
    /// Future body-fill enchants 3 deck cards with Swift(3) at obtain
    /// (counts pinned by [`BEAUTIFUL_BRACELET_CARDS`] +
    /// [`BEAUTIFUL_BRACELET_SWIFT`]).
    #[test]
    fn beautiful_bracelet_combat_dispatch_is_noop() {
        assert_eq!(BEAUTIFUL_BRACELET_CARDS, 3);
        assert_eq!(BEAUTIFUL_BRACELET_SWIFT, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(
            RelicId::BeautifulBracelet.to_python_name(),
            "BEAUTIFUL_BRACELET"
        );
    }

    /// `CallingBell` — combat dispatch is a no-op by design. Future
    /// body-fill adds CurseOfTheBell to deck and queues 3 relic
    /// rewards at obtain (count pinned by [`CALLING_BELL_RELICS`]).
    #[test]
    fn calling_bell_combat_dispatch_is_noop() {
        assert_eq!(CALLING_BELL_RELICS, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::CallingBell.to_python_name(), "CALLING_BELL");
    }

    // -----------------------------------------------------------------
    // 70anj.4.3 no-op pin tests — bead `sts2-rs-n06dm`
    // -----------------------------------------------------------------

    /// `MembershipCard` — combat dispatch is a no-op by design. The
    /// effect (`modify_merchant_price`) is a run-state shop-screen
    /// hook and never fires in combat.
    #[test]
    fn membership_card_combat_dispatch_is_noop() {
        assert_eq!(MEMBERSHIP_CARD_DISCOUNT, 50);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::MembershipCard.to_python_name(), "MEMBERSHIP_CARD");
    }

    /// `RoyalStamp` — combat dispatch is a no-op by design. Future
    /// body-fill enchants one deck card with `RoyallyApproved` at
    /// obtain (count hard-coded to 1 in Python; no separate const).
    #[test]
    fn royal_stamp_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::RoyalStamp.to_python_name(), "ROYAL_STAMP");
    }

    /// `CursedPearl` — combat dispatch is a no-op by design. Future
    /// body-fill adds the Greed curse to the deck and grants
    /// `CURSED_PEARL_GOLD` gold at obtain.
    #[test]
    fn cursed_pearl_combat_dispatch_is_noop() {
        assert_eq!(CURSED_PEARL_GOLD, 333);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::CursedPearl.to_python_name(), "CURSED_PEARL");
    }

    /// `ArcaneScroll` — combat dispatch is a no-op by design. Future
    /// body-fill adds one random Rare card to the deck at obtain
    /// (count hard-coded to 1 in Python; no separate const).
    #[test]
    fn arcane_scroll_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::ArcaneScroll.to_python_name(), "ARCANE_SCROLL");
    }

    /// `AlchemicalCoffer` — combat dispatch is a no-op by design.
    /// Future body-fill gains `ALCHEMICAL_COFFER_POTION_SLOTS` potion
    /// slots and procures that many random potions at obtain.
    #[test]
    fn alchemical_coffer_combat_dispatch_is_noop() {
        assert_eq!(ALCHEMICAL_COFFER_POTION_SLOTS, 4);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(
            RelicId::AlchemicalCoffer.to_python_name(),
            "ALCHEMICAL_COFFER"
        );
    }

    // -----------------------------------------------------------------
    // 70anj.4.4 no-op pin tests — bead `sts2-rs-kluld`
    // -----------------------------------------------------------------

    /// `ArchaicTooth` — combat dispatch is a no-op by design. The
    /// effect (`setup_for_player` + `after_obtained` starter-card
    /// transform) is pure run-state.
    #[test]
    fn archaic_tooth_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::ArchaicTooth.to_python_name(), "ARCHAIC_TOOTH");
    }

    /// `Astrolabe` — combat dispatch is a no-op by design. Future
    /// body-fill transforms+upgrades [`ASTROLABE_CARDS`] deck cards
    /// at obtain.
    #[test]
    fn astrolabe_combat_dispatch_is_noop() {
        assert_eq!(ASTROLABE_CARDS, 3);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Astrolabe.to_python_name(), "ASTROLABE");
    }

    /// `EmptyCage` — combat dispatch is a no-op by design. Future
    /// body-fill removes [`EMPTY_CAGE_CARDS`] deck cards at obtain.
    #[test]
    fn empty_cage_combat_dispatch_is_noop() {
        assert_eq!(EMPTY_CAGE_CARDS, 2);
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::EmptyCage.to_python_name(), "EMPTY_CAGE");
    }

    /// `Pomander` — combat dispatch is a no-op by design. Future
    /// body-fill upgrades 1 deck card at obtain (count hard-coded to
    /// 1 in Python; no separate const).
    #[test]
    fn pomander_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(RelicId::Pomander.to_python_name(), "POMANDER");
    }

    /// `PreciseScissors` — combat dispatch is a no-op by design. Future
    /// body-fill removes 1 deck card at obtain (count hard-coded to
    /// 1 in Python; no separate const).
    #[test]
    fn precise_scissors_combat_dispatch_is_noop() {
        crate::relics::dispatch::ensure_installed();
        assert_eq!(
            RelicId::PreciseScissors.to_python_name(),
            "PRECISE_SCISSORS"
        );
    }

    // -----------------------------------------------------------------
    // RunicCapacitor listener tests — bead `sts2-rs-h450h`
    // -----------------------------------------------------------------

    #[test]
    fn listener_runic_capacitor_adds_three_slots_on_round_one() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::RunicCapacitor);
        state.round = 1;
        // Start from the default 3 Defect slots so the assertion
        // catches the +3 delta, not just `>= 3`.
        state.player.orbs = sts2_core::orbs::OrbSlots::with_default_capacity();
        let before = state.player.orbs.capacity();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            state.player.orbs.capacity(),
            before + usize::try_from(RUNIC_CAPACITOR_ORB_SLOTS).unwrap(),
            "round-1 player adds {RUNIC_CAPACITOR_ORB_SLOTS} slots"
        );
    }

    #[test]
    fn listener_runic_capacitor_is_noop_after_round_one() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        for round in [2u16, 3, 5, 10] {
            let mut state = CombatState::default();
            equip_relic(&mut state, RelicKind::RunicCapacitor);
            state.round = round;
            state.player.orbs = sts2_core::orbs::OrbSlots::with_default_capacity();
            let before = state.player.orbs.capacity();
            let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
            assert_eq!(
                state.player.orbs.capacity(),
                before,
                "round {round} must not fire"
            );
        }
    }

    #[test]
    fn listener_runic_capacitor_is_noop_on_enemy_side() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::RunicCapacitor);
        state.round = 1;
        state.player.orbs = sts2_core::orbs::OrbSlots::with_default_capacity();
        let before = state.player.orbs.capacity();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.orbs.capacity(), before);
    }

    #[test]
    fn listener_runic_capacitor_is_noop_without_ownership() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsFlesh); // wrong relic
        state.round = 1;
        state.player.orbs = sts2_core::orbs::OrbSlots::with_default_capacity();
        let before = state.player.orbs.capacity();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.orbs.capacity(), before);
    }

    // -----------------------------------------------------------------
    /// v0.111.0 DiamondDiadem: turn-1 side turn start grants 20 unpowered
    /// Block and Blur 1; the v0.107.1 turn-end power arm is inert
    /// (decomp DiamondDiadem.cs:20-31).
    #[test]
    fn listener_diamond_diadem_v0111_grants_block_and_blur_on_turn_one() {
        use sts2_core::relic_effects::{dispatch_after_side_turn_start, dispatch_before_turn_end};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        equip_relic(&mut state, RelicKind::DiamondDiadem);
        state.round = 1;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.block, 0, "enemy side inert");
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.block, 20);
        assert_eq!(state.player_power_amount(sts2_core::PowerKind::Blur), 1);
        state.round = 2;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.block, 20, "turn 2 inert");
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert_eq!(
            state.player_power_amount(sts2_core::PowerKind::DiamondDiadem),
            0,
            "no DiamondDiademPower in v0.111.0"
        );
    }

    // DiamondDiadem listener tests — bead `sts2-rs-70anj.3.21`.
    // Player turn-end: if <=2 cards played, apply +1 DiamondDiadem power
    // to the player. The power-side dispatch arms (damage 0.5x + auto-
    // remove on enemy turn end) live in hooks.rs.
    // -----------------------------------------------------------------

    fn player_has_diamond_diadem_power(state: &CombatState) -> bool {
        state
            .powers
            .iter()
            .any(|p| p.kind == sts2_core::PowerKind::DiamondDiadem)
    }

    #[test]
    fn listener_diamond_diadem_applies_power_when_two_cards_played() {
        use sts2_core::relic_effects::dispatch_before_turn_end;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::DiamondDiadem);
        // 2 cards played (total) is <= threshold — power must apply.
        state.player.cards_played_this_turn = [2, 0, 0, 0, 0, 0];
        assert!(!player_has_diamond_diadem_power(&state));
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert!(player_has_diamond_diadem_power(&state));
    }

    #[test]
    fn listener_diamond_diadem_applies_power_when_zero_cards_played() {
        use sts2_core::relic_effects::dispatch_before_turn_end;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::DiamondDiadem);
        // 0 cards played also satisfies the <=2 condition.
        state.player.cards_played_this_turn = [0; 6];
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert!(player_has_diamond_diadem_power(&state));
    }

    #[test]
    fn listener_diamond_diadem_skips_when_three_cards_played() {
        use sts2_core::relic_effects::dispatch_before_turn_end;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::DiamondDiadem);
        // 3 total > threshold — no application.
        state.player.cards_played_this_turn = [1, 1, 1, 0, 0, 0];
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert!(!player_has_diamond_diadem_power(&state));
    }

    #[test]
    fn listener_diamond_diadem_skips_on_enemy_side() {
        use sts2_core::relic_effects::dispatch_before_turn_end;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::DiamondDiadem);
        state.player.cards_played_this_turn = [0; 6];
        let _ = dispatch_before_turn_end(&mut state, Side::Enemy);
        assert!(!player_has_diamond_diadem_power(&state));
    }

    #[test]
    fn listener_diamond_diadem_skips_without_relic() {
        use sts2_core::relic_effects::dispatch_before_turn_end;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Sai); // wrong relic
        state.player.cards_played_this_turn = [0; 6];
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert!(!player_has_diamond_diadem_power(&state));
    }

    // -----------------------------------------------------------------
    // Sozu / should_procure_potion channel tests — bead `sts2-rs-70anj.3.2`.
    // -----------------------------------------------------------------

    #[test]
    fn procure_potion_pushes_reward_without_sozu() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        assert!(rs.procure_potion(PotionReward::empty()));
        assert!(matches!(rs.pending_rewards.last(), Some(Reward::Potion(_))));
    }

    #[test]
    fn procure_potion_vetoed_by_sozu() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Sozu);
        let before_len = rs.pending_rewards.len();
        assert!(!rs.procure_potion(PotionReward::empty()));
        assert_eq!(rs.pending_rewards.len(), before_len);
    }

    #[test]
    fn claim_potion_reward_vetoed_by_sozu() {
        // C# `PotionReward.OnSelect` -> `PotionCmd.TryToProcure` returns
        // `NotAllowed` under Sozu: the reward is consumed, nothing lands
        // on the belt, and (unlike TooFull) no swap prompt / overflow.
        // Necrobinder seed 1002 floor-22 drop (GH #1054).
        use sts2_core::potions::PotionKind;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Sozu);
        rs.interactive_potion_swap = true;
        let belt_before = rs.potions.clone();
        assert!(!rs.claim_potion_reward(PotionKind::VulnerablePotion));
        assert_eq!(rs.potions, belt_before);
        assert!(rs.pending_potion_swaps.is_empty());
        rs.interactive_potion_swap = false;
        assert!(!rs.claim_potion_reward(PotionKind::VulnerablePotion));
        assert_eq!(rs.potions, belt_before);
        assert_eq!(rs.potions_overflow_dropped, 0);
    }

    #[test]
    fn claim_potion_reward_lands_without_sozu() {
        use sts2_core::potions::PotionKind;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::PaelsFlesh);
        assert!(rs.claim_potion_reward(PotionKind::VulnerablePotion));
        assert!(rs.potions.contains(&PotionKind::VulnerablePotion));
    }

    #[test]
    fn sozu_trips_the_surface_veto_probe() {
        // GH #270: the shop action surface consults
        // `would_veto_procure_potion` before surfacing the potion row —
        // verify the probe fires for the REAL registered Sozu listener
        // (the sts2-core surface tests use a stand-in).
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        assert!(!sts2_core::run_effects::would_veto_procure_potion(&rs));
        rs.relics.push(RelicId::Sozu);
        assert!(sts2_core::run_effects::would_veto_procure_potion(&rs));
    }

    #[test]
    fn procure_potion_no_veto_without_sozu_relic_in_inventory() {
        // Sozu's listener is process-globally registered (via
        // ensure_installed); the veto fires only when the relic is in
        // rs.relics. Without it, the channel must not block.
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Some unrelated relic in inventory.
        rs.relics.push(RelicId::PaelsFlesh);
        assert!(rs.procure_potion(PotionReward::empty()));
    }

    // -----------------------------------------------------------------
    // DollysMirror / duplicate_card_from_deck tests — bead `sts2-rs-70anj.3.10`.
    // -----------------------------------------------------------------

    fn make_card(id: sts2_core::cards::CardId) -> sts2_core::cards::Card {
        let mut c = sts2_core::Card::pod_defaults();
        c.card_id = id;
        c
    }

    #[test]
    fn duplicate_card_from_deck_appends_clone() {
        let mut rs = RunState::new();
        rs.master_deck
            .push(make_card(sts2_core::cards::CardId::StrikeIronclad));
        rs.master_deck
            .push(make_card(sts2_core::cards::CardId::DefendIronclad));
        let before_len = rs.master_deck.len();
        assert!(rs.duplicate_card_from_deck());
        assert_eq!(rs.master_deck.len(), before_len + 1);
    }

    /// C# `DollysMirror.AfterObtained` (v0.111.0 / v0.107.1
    /// `Relics/DollysMirror.cs:17-24`) opens `CardSelectCmd.FromDeckGeneric`
    /// with a `Type != Quest` filter and no sort (`CardSelectCmd.cs:759-770`
    /// lists the deck in order). There is no Rng in the class, so the clone
    /// must not tick `rng.rewards`; the headless pick is the first eligible
    /// deck card (the select's index 0).
    #[test]
    fn duplicate_card_from_deck_clones_first_non_quest_card_without_rng() {
        use sts2_core::cards::CardId;
        let mut rs = RunState::new();
        rs.master_deck.clear();
        rs.master_deck.push(make_card(CardId::SpoilsMap));
        rs.master_deck.push(make_card(CardId::Bash));
        rs.master_deck.push(make_card(CardId::StrikeIronclad));
        rs.master_deck.push(make_card(CardId::DefendIronclad));
        let counter_before = rs.rng.rewards.counter();
        assert!(rs.duplicate_card_from_deck());
        assert_eq!(rs.rng.rewards.counter(), counter_before);
        assert_eq!(rs.master_deck.len(), 5);
        assert_eq!(rs.master_deck[4].card_id, CardId::Bash);
    }

    #[test]
    fn duplicate_card_from_deck_skips_all_quest_deck() {
        let mut rs = RunState::new();
        rs.master_deck.clear();
        rs.master_deck
            .push(make_card(sts2_core::cards::CardId::SpoilsMap));
        assert!(!rs.duplicate_card_from_deck());
        assert_eq!(rs.master_deck.len(), 1);
    }

    #[test]
    fn duplicate_card_from_deck_returns_false_on_empty_deck() {
        let mut rs = RunState::new();
        rs.master_deck.clear();
        assert!(!rs.duplicate_card_from_deck());
        assert!(rs.master_deck.is_empty());
    }

    #[test]
    fn listener_dollys_mirror_grows_deck_when_relic_obtained() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck
            .push(make_card(sts2_core::cards::CardId::StrikeIronclad));
        rs.relics.push(RelicId::DollysMirror);
        let before_len = rs.master_deck.len();
        rs.last_obtained_relic = Some(RelicId::DollysMirror);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
    }

    #[test]
    fn listener_dollys_mirror_noop_without_relic() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck
            .push(make_card(sts2_core::cards::CardId::StrikeIronclad));
        // No DollysMirror in inventory.
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    // -----------------------------------------------------------------
    // Pool helpers + ArcaneScroll tests — bead `sts2-rs-70anj.3.15`.
    // -----------------------------------------------------------------

    #[test]
    fn random_card_from_rarity_returns_rare_card() {
        use crate::cards::spec_index::random_card_from_rarity;
        use crate::cards::CardRarity;
        let mut rs = RunState::new();
        let pick = random_card_from_rarity(&mut rs, None, CardRarity::Rare);
        assert!(pick.is_some(), "pool must contain at least one Rare card");
    }

    #[test]
    fn random_power_card_returns_power_typed() {
        use crate::cards::spec_index::random_power_card;
        let mut rs = RunState::new();
        let pick = random_power_card(&mut rs, None);
        assert!(pick.is_some(), "pool must contain at least one Power card");
    }

    #[test]
    fn random_curse_card_returns_curse_typed() {
        use crate::cards::spec_index::random_curse_card;
        let mut rs = RunState::new();
        let pick = random_curse_card(&mut rs);
        assert!(
            pick.is_some(),
            "STATUS_SPECS must contain at least one Curse"
        );
    }

    #[test]
    fn random_card_from_rarity_per_character_ironclad_only() {
        use crate::cards::spec_index::random_card_from_rarity;
        use crate::cards::CardRarity;
        use sts2_core::potions::CharacterId;
        let mut rs = RunState::new();
        for _ in 0..8 {
            assert!(random_card_from_rarity(
                &mut rs,
                Some(CharacterId::Ironclad),
                CardRarity::Rare
            )
            .is_some());
        }
    }

    #[test]
    fn listener_arcane_scroll_adds_card_to_deck() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::ArcaneScroll);
        let before_len = rs.master_deck.len();
        rs.last_obtained_relic = Some(RelicId::ArcaneScroll);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
    }

    #[test]
    fn listener_arcane_scroll_noop_without_relic() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_len = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len);
    }

    #[test]
    fn listener_dusty_tome_adds_upgraded_ancient_card_to_deck() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::DustyTome);
        let before_len = rs.master_deck.len();
        rs.last_obtained_relic = Some(RelicId::DustyTome);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.master_deck.len(), before_len + 1);
        // The added card must be upgraded.
        let added = rs.master_deck.last().expect("card added");
        assert!(added.upgraded, "DustyTome must add an upgraded card");
    }

    /// `DustyTome.AfterObtained` (`DustyTome.cs:58-63`, both builds) adds the
    /// card through `CardPileCmd.Add`, so LuckyFysh's `AfterCardChangedPiles`
    /// pays 15 gold (GH #806: the v0.107.1 arm used to push bare).
    #[test]
    fn listener_dusty_tome_pays_lucky_fysh_both_builds() {
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        for (version, gain) in [
            (sts2_core::GameVersion::V0_107_1, 15),
            (sts2_core::GameVersion::V0_111_0, 15),
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            rs.relics.push(RelicId::LuckyFysh);
            rs.relics.push(RelicId::DustyTome);
            let gold = rs.gold;
            rs.last_obtained_relic = Some(RelicId::DustyTome);
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
            assert_eq!(rs.gold - gold, gain, "{version}");
            assert!(rs.master_deck.last().expect("card added").upgraded);
        }
    }

    /// v0.107.1 SereTalon: 2 distinct generatable curses off `Niche`, then 3
    /// Wishes (v0.107.1 `SereTalon.cs:32-55`).
    #[test]
    fn listener_sere_talon_adds_two_curse_cards_to_deck() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::SereTalon);
        let before_len = rs.master_deck.len();
        rs.last_obtained_relic = Some(RelicId::SereTalon);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let added = &rs.master_deck[before_len..];
        assert_eq!(added.len(), 5);
        assert!(added[..2]
            .iter()
            .all(|c| crate::relics::neow_relics::NEOWS_BONES_CURSE_POOL.contains(&c.card_id)));
        assert_ne!(added[0].card_id, added[1].card_id);
        assert!(added[2..].iter().all(|c| c.card_id == CardId::Wish));
    }

    /// v0.111.0: SereTalon loses 9 Max HP and adds 3 Wishes, no curses
    /// (decomp SereTalon.cs:31-43).
    #[test]
    fn listener_sere_talon_loses_nine_max_hp_and_adds_three_wishes_v0111() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.game_version = sts2_core::GameVersion::V0_111_0;
        rs.relics.push(RelicId::SereTalon);
        let before_len = rs.master_deck.len();
        let before_max = rs.max_hp;
        rs.last_obtained_relic = Some(RelicId::SereTalon);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max - 9);
        let added = &rs.master_deck[before_len..];
        assert_eq!(added.len(), 3);
        assert!(added.iter().all(|c| c.card_id == CardId::Wish));
    }

    /// Bead `sts2-rs-70anj.3.5`: Driftwood — `allow_card_reward_reroll`
    /// returns `true` when owned, `false` (empty-bus default) otherwise.
    #[test]
    fn listener_driftwood_allows_card_reward_reroll_when_owned() {
        use sts2_core::run_effects::dispatch_allow_card_reward_reroll;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        assert!(
            !dispatch_allow_card_reward_reroll(&mut rs),
            "without Driftwood the channel must reject the reroll"
        );
        rs.relics.push(RelicId::Driftwood);
        assert!(
            dispatch_allow_card_reward_reroll(&mut rs),
            "with Driftwood the channel must allow the reroll"
        );
    }

    /// Bead `sts2-rs-70anj.3.7`: ChemicalX adds `+INCREASE = 2` to every
    /// X-cost play's effective `x` value when owned. With no relic owner
    /// the dispatcher returns the base x unchanged.
    #[test]
    fn listener_chemical_x_adds_two_to_x_when_owned() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_modify_x_value;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // No ChemicalX owned → fold returns base.
        assert_eq!(dispatch_modify_x_value(&s, 3), 3);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::ChemicalX,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        assert_eq!(
            dispatch_modify_x_value(&s, 3),
            3 + CHEMICAL_X_INCREASE,
            "ChemicalX must add INCREASE to running x"
        );
    }

    /// Bead `sts2-rs-70anj.3.3`: ThrowingAxe returns `+1` on the first
    /// card played each combat, then `0` thereafter. The latch lives on
    /// `Relic.counter`; a fresh combat (`before_combat_start` reset)
    /// re-arms.
    #[test]
    fn listener_throwing_axe_fires_once_per_combat() {
        use std::num::NonZeroU32;
        use sts2_core::hooks::{CardPlayCountCtx, CreatureRef};
        use sts2_core::relic_effects::{
            dispatch_before_combat_start, dispatch_relic_modify_card_play_count,
        };
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        let ctx = CardPlayCountCtx {
            owner: CreatureRef::Player,
            ..CardPlayCountCtx::default()
        };
        // No ThrowingAxe owned → 0 delta.
        assert_eq!(dispatch_relic_modify_card_play_count(&mut s, &ctx), 0);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::ThrowingAxe,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        // First play: +1; latch trips.
        assert_eq!(dispatch_relic_modify_card_play_count(&mut s, &ctx), 1);
        assert_eq!(s.relics[0].counter, 1, "latch must be set");
        // Subsequent plays: 0.
        assert_eq!(dispatch_relic_modify_card_play_count(&mut s, &ctx), 0);
        assert_eq!(dispatch_relic_modify_card_play_count(&mut s, &ctx), 0);
        // before_combat_start clears the latch.
        let _ = dispatch_before_combat_start(&mut s);
        assert_eq!(s.relics[0].counter, 0, "reset must clear latch");
        assert_eq!(
            dispatch_relic_modify_card_play_count(&mut s, &ctx),
            1,
            "next combat must re-arm"
        );
    }

    /// Bead `sts2-rs-70anj.3.3`: ThrowingAxe gates on `ctx.owner ==
    /// Player`. An enemy-side play (if such a play count dispatch ever
    /// fires for an enemy) must not consume the latch.
    #[test]
    fn listener_throwing_axe_player_only() {
        use std::num::NonZeroU32;
        use sts2_core::hooks::{CardPlayCountCtx, CreatureRef};
        use sts2_core::relic_effects::dispatch_relic_modify_card_play_count;
        use sts2_core::{CreatureUid, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::ThrowingAxe,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let enemy_ctx = CardPlayCountCtx {
            owner: CreatureRef::Enemy(CreatureUid(NonZeroU32::new(2).expect("non-zero"))),
            ..CardPlayCountCtx::default()
        };
        assert_eq!(dispatch_relic_modify_card_play_count(&mut s, &enemy_ctx), 0);
        assert_eq!(s.relics[0].counter, 0, "enemy play must not trip latch");
    }

    /// Bead `sts2-rs-70anj.3.12`: DelicateFrond's `before_combat_start`
    /// fills empty potion slots up to `MAX_POTIONS` with random
    /// out-of-combat, character-compatible potions. Asserts the fill brings the belt up
    /// to the cap; the exact picks are RNG-dependent so we only check
    /// counts.
    #[test]
    fn listener_delicate_frond_fills_potion_belt() {
        use std::num::NonZeroU32;
        use sts2_core::potions::MAX_POTIONS;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::DelicateFrond,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        assert!(s.player.potions.is_empty(), "fresh state has empty belt");
        let _ = dispatch_before_combat_start(&mut s);
        assert_eq!(
            s.player.potions.len(),
            MAX_POTIONS,
            "DelicateFrond fills the belt to MAX_POTIONS"
        );
    }

    /// DelicateFrond's `before_combat_start` listener is ownership-gated:
    /// when the player does NOT carry it, the belt stays empty (no phantom
    /// potions). Regression for the workbench autoplay showing 5 bogus
    /// drinkable potions on a run that owns no potion-filling relic.
    #[test]
    fn listener_delicate_frond_noop_when_not_owned() {
        use sts2_core::relic_effects::dispatch_before_combat_start;

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
        // No relics owned.
        assert!(s.player.potions.is_empty(), "fresh state has empty belt");
        let _ = dispatch_before_combat_start(&mut s);
        assert!(
            s.player.potions.is_empty(),
            "belt must stay empty when DelicateFrond is not owned",
        );
    }

    /// GH #1054 (Defect seed 1015): DelicateFrond rolls each fill off the
    /// run-level `CombatPotionGeneration` stream via
    /// `CreateRandomPotionOutOfCombat` (2 draws per potion: NextFloat rarity
    /// then NextItem) — `DelicateFrond.cs:12-22`. With 1 of 3 slots filled it
    /// fills the 2 open slots with exactly the potions the out-of-combat roll
    /// yields for the same stream, advancing it by 4, and leaves the
    /// combat-local `rng` untouched.
    #[test]
    fn listener_delicate_frond_rolls_off_combat_potion_generation() {
        use std::num::NonZeroU32;
        use sts2_core::potions::{CharacterId, PotionKind};
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.character = Some(CharacterId::Defect);
        s.player.potion_slots = 3;
        s.player.potions.push(PotionKind::BlockPotion);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::DelicateFrond,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });

        let mut rng = s.run_rng.combat_potion_generation.clone();
        let r1 = crate::potions::all::create_random_potions_out_of_combat(
            &mut rng,
            CharacterId::Defect,
            1,
        )[0];
        let r2 = crate::potions::all::create_random_potions_out_of_combat(
            &mut rng,
            CharacterId::Defect,
            1,
        )[0];
        let cpg_before = s.run_rng.combat_potion_generation.counter();
        let combat_rng_before = s.rng.counter();

        let _ = dispatch_before_combat_start(&mut s);

        assert_eq!(
            s.player.potions.as_slice(),
            [PotionKind::BlockPotion, r1, r2].as_slice(),
            "the 2 open slots get the out-of-combat rolls, in order"
        );
        assert_eq!(
            s.run_rng.combat_potion_generation.counter() - cpg_before,
            4,
            "2 draws per filled slot on CombatPotionGeneration"
        );
        assert_eq!(
            s.rng.counter(),
            combat_rng_before,
            "the combat-local rng must not be drawn"
        );
    }

    /// GH #1054 (Ironclad seed 2170, v0.111.0): Sozu + DelicateFrond on an
    /// empty belt. C# `DelicateFrond.BeforeCombatStart` rolls one potion
    /// (2 `CombatPotionGeneration` draws), `PotionCmd.TryToProcure` fails
    /// `NotAllowed` (`Sozu.ShouldProcurePotion`, `PotionCmd.cs:31`), and the
    /// loop breaks (`DelicateFrond.cs:15-21`): empty belt, stream +2.
    #[test]
    fn listener_delicate_frond_vetoed_by_sozu() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
        s.player.potion_slots = 3;
        for (id, kind) in [(1, RelicKind::Sozu), (2, RelicKind::DelicateFrond)] {
            s.relics.push(Relic {
                id: NonZeroU32::new(id).expect("non-zero"),
                kind,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
        let cpg_before = s.run_rng.combat_potion_generation.counter();
        let _ = dispatch_before_combat_start(&mut s);
        assert!(s.player.potions.is_empty(), "Sozu blocks the fill");
        assert_eq!(s.run_rng.combat_potion_generation.counter() - cpg_before, 2);
    }

    /// Bead `sts2-rs-70anj.3.12`: DelicateFrond no-ops when the belt is
    /// already full — the `HasOpenPotionSlots` loop never enters.
    #[test]
    fn listener_delicate_frond_noop_when_full() {
        use std::num::NonZeroU32;
        use sts2_core::potions::{PotionKind, MAX_POTIONS};
        use sts2_core::relic_effects::dispatch_before_combat_start;
        use sts2_core::{Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
        for _ in 0..MAX_POTIONS {
            s.player.potions.push(PotionKind::BlockPotion);
        }
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::DelicateFrond,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let belt_before = s.player.potions.clone();
        let _ = dispatch_before_combat_start(&mut s);
        assert_eq!(s.player.potions, belt_before, "full belt must be untouched");
    }

    /// Bead `sts2-rs-70anj.3.9`: Astrolabe transforms ASTROLABE_CARDS (=3)
    /// random non-curse deck cards and upgrades them. Asserts the deck
    /// length is preserved (in-place replacement) and that at least the
    /// chosen replacements come out upgraded.
    /// Interactive hosts get C#'s Astrolabe: 3 upgrading transform picks
    /// queued, no RNG drawn at obtain (`Astrolabe.cs:31-39`).
    #[test]
    fn listener_astrolabe_interactive_queues_three_upgrading_transforms() {
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.interactive_card_transform = true;
        for _ in 0..5 {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = sts2_core::cards::CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        let rewards = rs.rng.rewards.counter();
        let deck_before = rs.master_deck.clone();
        rs.relics.push(RelicId::Astrolabe);
        rs.last_obtained_relic = Some(RelicId::Astrolabe);
        let _ = dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_deck_transforms.len(), ASTROLABE_CARDS);
        assert!(rs.pending_deck_transforms.iter().all(|t| t.upgrade));
        assert_eq!(rs.rng.rewards.counter(), rewards);
        assert_eq!(rs.master_deck, deck_before);
    }

    /// Astrolabe's 3-card select opens no screen when the deck holds 3 or
    /// fewer transformable cards (`list.Count <= MinSelect`,
    /// CardSelectCmd.cs): both cards transform, off `Rng.Niche`, upgraded.
    #[test]
    fn listener_astrolabe_takes_every_card_when_three_or_fewer_are_transformable() {
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::CharacterId::Ironclad;
        rs.interactive_card_transform = true;
        for _ in 0..2 {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(
                    sts2_core::cards::CardId::StrikeIronclad,
                    false,
                )
                .unwrap(),
            );
        }
        let niche = rs.rng.niche.counter();
        rs.relics.push(RelicId::Astrolabe);
        rs.last_obtained_relic = Some(RelicId::Astrolabe);
        let _ = dispatch_after_obtained(&mut rs);
        assert!(rs.pending_deck_transforms.is_empty(), "no screen opens");
        assert_eq!(rs.rng.niche.counter(), niche + 2);
        assert_eq!(rs.master_deck.len(), 2);
        assert!(rs
            .master_deck
            .iter()
            .all(|c| c.upgraded && c.card_id != sts2_core::cards::CardId::StrikeIronclad));
    }

    #[test]
    fn listener_astrolabe_transforms_and_upgrades_three_cards() {
        use sts2_core::cards::CardId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new().with_rewards_seed(0xa57_201ab);
        // Seed the master deck with 10 unupgraded Strikes (rarity
        // `Basic`, NOT a curse/status — eligible for transform).
        for _ in 0..10 {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::Astrolabe);
        rs.last_obtained_relic = Some(RelicId::Astrolabe);
        let _ = dispatch_after_obtained(&mut rs);
        rs.last_obtained_relic = None;
        assert_eq!(
            rs.master_deck.len(),
            10,
            "deck length preserved by in-place transform"
        );
        let upgraded_count = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert!(
            upgraded_count >= 1,
            "astrolabe upgrades at least one replacement (got {upgraded_count})"
        );
        let non_strike = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id != CardId::StrikeIronclad)
            .count();
        assert!(
            non_strike >= 1,
            "astrolabe transforms at least one Strike (got {non_strike} non-Strike)"
        );
    }

    /// Bead `sts2-rs-70anj.3.9`: Astrolabe must NOT fire when an
    /// unrelated relic is obtained after pickup — `last_obtained_relic`
    /// gates the listener to its own pickup event.
    #[test]
    fn listener_astrolabe_is_noop_on_other_relic_pickup() {
        use sts2_core::cards::CardId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new().with_rewards_seed(0xa57_201ab);
        for _ in 0..10 {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        rs.relics.push(RelicId::Astrolabe);
        // Some OTHER relic was just obtained.
        rs.last_obtained_relic = Some(RelicId::Strawberry);
        let before: Vec<_> = rs
            .master_deck
            .iter()
            .map(|c| (c.card_id, c.upgraded))
            .collect();
        let _ = dispatch_after_obtained(&mut rs);
        let after: Vec<_> = rs
            .master_deck
            .iter()
            .map(|c| (c.card_id, c.upgraded))
            .collect();
        assert_eq!(
            before, after,
            "astrolabe must not fire on unrelated relic pickup"
        );
    }

    /// Bead `sts2-rs-70anj.3.17`: StoneHumidifier gains MAX_HP=5 each
    /// time `RunState::rest_site_heal` actually heals (>0).
    #[test]
    fn listener_stone_humidifier_gains_max_hp_on_rest_heal() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.max_hp = 80;
        rs.current_hp = 40;
        rs.relics.push(RelicId::StoneHumidifier);
        let before_max = rs.max_hp;
        let healed = rs.rest_site_heal(10);
        assert!(healed > 0, "rest-site heal must apply >0");
        assert_eq!(
            rs.max_hp,
            before_max + STONE_HUMIDIFIER_MAX_HP,
            "StoneHumidifier adds {STONE_HUMIDIFIER_MAX_HP} max HP on heal",
        );
    }

    /// Bead `sts2-rs-RP21`: StoneHumidifier grants +5 MaxHp on EVERY rest-site
    /// heal hook, even when the player is already at max HP (no HP restored).
    /// C# `Hook.AfterRestSiteHeal` carries no heal amount and fires
    /// unconditionally, so a full-HP campfire rest still bumps MaxHp. (Inverts
    /// the prior `noop_when_no_heal` assertion, which encoded the Python-pin
    /// `healed<=0` guard that drifted from the C# decomp.)
    #[test]
    fn listener_stone_humidifier_grants_max_hp_even_at_full_hp() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.max_hp = 80;
        rs.current_hp = 80;
        rs.relics.push(RelicId::StoneHumidifier);
        let before_max = rs.max_hp;
        let healed = rs.rest_site_heal(10);
        assert_eq!(healed, 0, "no rest heal lands when already at max HP");
        assert_eq!(
            rs.max_hp,
            before_max + STONE_HUMIDIFIER_MAX_HP,
            "MaxHp still bumped at full HP (amount-agnostic C# hook)"
        );
    }

    /// Bead `sts2-rs-70anj.3.17`: ownership gate — `rest_site_heal`
    /// fires the channel but with StoneHumidifier absent the listener
    /// does nothing.
    #[test]
    fn listener_stone_humidifier_noop_without_ownership() {
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.max_hp = 80;
        rs.current_hp = 40;
        let before_max = rs.max_hp;
        let healed = rs.rest_site_heal(10);
        assert!(healed > 0);
        assert_eq!(rs.max_hp, before_max, "no listener fires without relic");
    }

    // ------------------------------------------------------------------
    // Toolbox / ChoicesParadox (bead `sts2-rs-70anj.3.13.tb` / `.cp`)
    // ------------------------------------------------------------------

    /// Toolbox opens a 3-option AddGeneratedToHand choice on round 1
    /// and trips its latch.
    #[test]
    fn listener_toolbox_opens_3_choice_round_1() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        use sts2_core::{ChoiceKind, CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // Alive player -- a dead one trips the C# IsOverOrEnding select guard.
        s.player.current_hp = 80;
        s.round = 1;
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Toolbox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_before_hand_draw(&mut s);
        let pc = s
            .pending_choice
            .as_ref()
            .expect("Toolbox must open a choice");
        assert_eq!(pc.kind, ChoiceKind::AddGeneratedToHand);
        assert_eq!(pc.options.len(), TOOLBOX_CARDS);
        assert_eq!(s.relics[0].counter, 1, "latch trips");
    }

    /// Toolbox samples C# `GetDistinctForCombat` (v0.111.0 `Toolbox.cs:27`):
    /// the filtered solo colorless pool (never MultiplayerOnly HuddleUp, never
    /// `CanBeGeneratedInCombat=false` Alchemize), shuffled with
    /// `run_rng.combat_card_generation`, leaving `state.rng` untouched.
    #[test]
    fn listener_toolbox_samples_combat_card_generation_filtered_pool() {
        use std::num::NonZeroU32;
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        use sts2_core::{CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        for seed in 0..200_i32 {
            let mut s = CombatState::default();
            s.player.current_hp = 80;
            s.round = 1;
            s.run_rng.combat_card_generation = sts2_core::dotnet_rng::Rng::new(seed);
            s.relics.push(Relic {
                id: NonZeroU32::new(1).expect("non-zero"),
                kind: RelicKind::Toolbox,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
            let rng_before = s.rng.clone();
            let mut expected = s.clone();
            let want = crate::cards::colorless::distinct_colorless_modifier_ids(
                &mut expected,
                TOOLBOX_CARDS,
            );
            let _ = dispatch_before_hand_draw(&mut s);
            let got: Vec<CardId> = s
                .pending_choice
                .as_ref()
                .expect("Toolbox must open a choice")
                .options
                .iter()
                .map(|o| o.card_id)
                .collect();
            assert_eq!(
                got,
                want.to_vec(),
                "seed {seed}: GetDistinctForCombat order"
            );
            assert!(
                !got.contains(&CardId::HuddleUp) && !got.contains(&CardId::Alchemize),
                "seed {seed}: filtered card offered: {got:?}",
            );
            assert_eq!(s.rng, rng_before, "seed {seed}: combat rng untouched");
            assert_eq!(
                s.run_rng.combat_card_generation, expected.run_rng.combat_card_generation,
                "seed {seed}: CombatCardGeneration advanced by the shuffle",
            );
            assert_ne!(
                s.run_rng.combat_card_generation,
                sts2_core::dotnet_rng::Rng::new(seed),
                "seed {seed}: CombatCardGeneration must be consumed",
            );
        }
    }

    /// Toolbox is a no-op on round != 1.
    #[test]
    fn listener_toolbox_noop_post_round_1() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        use sts2_core::{CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.round = 2;
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Toolbox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_before_hand_draw(&mut s);
        assert!(s.pending_choice.is_none());
        assert_eq!(s.relics[0].counter, 0);
    }

    /// Toolbox fires exactly once per combat; latch resets at
    /// before_combat_start.
    #[test]
    fn listener_toolbox_latch_resets_per_combat() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::{dispatch_before_combat_start, dispatch_before_hand_draw};
        use sts2_core::{CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // Alive player -- a dead one trips the C# IsOverOrEnding select guard.
        s.player.current_hp = 80;
        s.round = 1;
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Toolbox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_before_hand_draw(&mut s);
        assert!(s.pending_choice.is_some());
        s.pending_choice = None;
        let _ = dispatch_before_hand_draw(&mut s);
        assert!(s.pending_choice.is_none(), "latched; no re-fire");

        let _ = dispatch_before_combat_start(&mut s);
        assert_eq!(s.relics[0].counter, 0, "reset must clear latch");
        let _ = dispatch_before_hand_draw(&mut s);
        assert!(s.pending_choice.is_some(), "re-arms after reset");
    }

    /// Resolving the Toolbox choice adds the chosen card to hand.
    #[test]
    fn resolve_toolbox_choice_adds_card_to_hand() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_before_hand_draw;
        use sts2_core::{apply, Action, CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // Alive player -- a dead one trips the C# IsOverOrEnding select guard.
        s.player.current_hp = 80;
        s.round = 1;
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::Toolbox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let hand_before = s.player.hand.len();
        let _ = dispatch_before_hand_draw(&mut s);
        assert!(s.pending_choice.is_some());
        let _ = apply(
            &mut s,
            Action::ResolveChoice {
                selection: smallvec![0u8],
            },
        );
        assert!(s.pending_choice.is_none(), "choice cleared");
        assert_eq!(
            s.player.hand.len(),
            hand_before + 1,
            "chosen card lands in hand",
        );
    }

    /// GH #1054 (act-3 `player.hp` splits): ChoicesParadox is a C#
    /// `AfterPlayerTurnStart` relic, so the equip-ordered walk suspends on its
    /// pick and a later RoyalPoison deals its 4 damage only once the pick
    /// resolves. On `after_side_turn_start` it opened after RoyalPoison had
    /// already fired.
    #[test]
    fn choices_paradox_pick_suspends_a_later_royal_poison() {
        use std::num::NonZeroU32;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::{
            dispatch_after_player_turn_start, dispatch_after_player_turn_start_from,
        };
        use sts2_core::{CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        s.player.current_hp = 80;
        s.player.max_hp = 80;
        s.round = 1;
        s.player.character = Some(CharacterId::Ironclad);
        let _ = s.init_player_creature();
        for (raw, kind) in [(1, RelicKind::ChoicesParadox), (2, RelicKind::RoyalPoison)] {
            s.relics.push(Relic {
                id: NonZeroU32::new(raw).expect("non-zero"),
                kind,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
        let resume = dispatch_after_player_turn_start(&mut s).expect("the pick suspends the walk");
        assert!(s.pending_choice.is_some());
        assert_eq!(s.player.current_hp, 80, "RoyalPoison waits for the pick");
        s.pending_choice = None;
        let _ = dispatch_after_player_turn_start_from(&mut s, resume);
        assert_eq!(s.player.current_hp, 76, "RoyalPoison fires after the pick");
    }

    /// ChoicesParadox opens a 5-option AddGeneratedRetainedToHand
    /// choice on round 1 player-turn-start.
    #[test]
    fn listener_choices_paradox_opens_5_retained_choice() {
        use std::num::NonZeroU32;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_player_turn_start;
        use sts2_core::{ChoiceKind, CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // Alive player -- a dead one trips the C# IsOverOrEnding select guard.
        s.player.current_hp = 80;
        s.round = 1;
        s.player.character = Some(CharacterId::Ironclad);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::ChoicesParadox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_after_player_turn_start(&mut s);
        let pc = s
            .pending_choice
            .as_ref()
            .expect("ChoicesParadox must open a choice");
        assert_eq!(pc.kind, ChoiceKind::AddGeneratedRetainedToHand);
        assert_eq!(pc.options.len(), CHOICES_PARADOX_CARDS);
        assert_eq!(s.relics[0].counter, 1, "latch trips");
        // `GetDistinctForCombat` shuffles the whole fight pool on
        // CombatCardGeneration: n - 1 reads, and the options are its first 5.
        let pool = crate::cards::colorless::helpers::fight_pool(&s, CharacterId::Ironclad);
        assert_eq!(
            s.run_rng.combat_card_generation.counter() as usize,
            pool.len() - 1,
            "one UnstableShuffle of the pool"
        );
    }

    // ------------------------------------------------------------------
    // MeatCleaver / MiniatureTent (bead `sts2-rs-ekf4j` + `.4`)
    // ------------------------------------------------------------------

    /// MeatCleaver appends a Cook option to the rest-site list when
    /// owned, with `enabled` reflecting removable-card count.
    #[test]
    fn listener_meat_cleaver_appends_cook_option() {
        use sts2_core::runstate::{RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MeatCleaver);
        // Any three removable cards clear Cook's `>= 2` bar.
        let mut c1 = sts2_core::Card::pod_defaults();
        c1.card_id = sts2_core::cards::CardId::Anger;
        let mut c2 = sts2_core::Card::pod_defaults();
        c2.card_id = sts2_core::cards::CardId::Bash;
        let mut c3 = sts2_core::Card::pod_defaults();
        c3.card_id = sts2_core::cards::CardId::Cinder;
        rs.master_deck.extend([c1, c2, c3]);
        let opts = rs.rest_site_options();
        assert!(opts.iter().any(|o| matches!(o, RestSiteOption::Rest)));
        assert!(opts
            .iter()
            .any(|o| matches!(o, RestSiteOption::Cook { enabled: true })));
    }

    /// C# `CookRestSiteOption.IsEnabled => GetRemovableCardCount(Owner) >= 2`
    /// counting `c.IsRemovable` (`CookRestSiteOption.cs:36,64`), so a
    /// starting deck of basic Strikes and Defends ENABLES Cook. Until
    /// 2026-08-26 `removable_deck_card_count` filtered on the legacy
    /// basics-excluding predicate, so this deck counted 0 and Cook was
    /// disabled on the exact deck the relic is picked up with.
    #[test]
    fn cook_is_enabled_on_a_deck_of_basics_and_disabled_by_eternal_curses() {
        use sts2_core::runstate::{RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        // Cook counts through the CARD registry, not the relic one — without
        // this the removable predicate is absent and every card counts, which
        // would pass the basics half of this test for the wrong reason.
        crate::cards::dispatch::install_default_registry();
        let cook_enabled = |rs: &RunState| {
            rs.rest_site_options()
                .iter()
                .any(|o| matches!(o, RestSiteOption::Cook { enabled: true }))
        };
        let card = |id| sts2_core::Card {
            card_id: id,
            ..sts2_core::Card::pod_defaults()
        };

        let mut rs = RunState::new();
        rs.relics.push(RelicId::MeatCleaver);
        rs.master_deck.extend([
            card(sts2_core::cards::CardId::StrikeIronclad),
            card(sts2_core::cards::CardId::DefendIronclad),
        ]);
        assert!(cook_enabled(&rs), "two basics are two removable cards");

        // Eternal curses do NOT count toward the bar.
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MeatCleaver);
        rs.master_deck.extend([
            card(sts2_core::cards::CardId::Greed),
            card(sts2_core::cards::CardId::AscendersBane),
            card(sts2_core::cards::CardId::StrikeIronclad),
        ]);
        assert!(
            !cook_enabled(&rs),
            "only the Strike is removable, so the count is 1"
        );
    }

    /// MeatCleaver appends Cook with `enabled = false` when the deck
    /// has fewer than COOK_OPTION_CARD_COUNT removable cards.
    #[test]
    fn listener_meat_cleaver_cook_disabled_when_deck_too_small() {
        use sts2_core::runstate::{RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MeatCleaver);
        let mut c1 = sts2_core::Card::pod_defaults();
        c1.card_id = sts2_core::cards::CardId::Anger;
        rs.master_deck.push(c1);
        let opts = rs.rest_site_options();
        assert!(opts
            .iter()
            .any(|o| matches!(o, RestSiteOption::Cook { enabled: false })));
    }

    /// MeatCleaver does NOT duplicate Cook if some other source has
    /// already pushed one.
    #[test]
    fn listener_meat_cleaver_does_not_duplicate_cook() {
        use sts2_core::runstate::RestSiteOption;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MeatCleaver);
        let mut options = vec![RestSiteOption::Rest, RestSiteOption::Cook { enabled: true }];
        let _ = sts2_core::run_effects::dispatch_modify_rest_site_options(&rs, &mut options);
        let cook_count = options
            .iter()
            .filter(|o| matches!(o, RestSiteOption::Cook { .. }))
            .count();
        assert_eq!(cook_count, 1, "Cook must not duplicate");
    }

    /// MiniatureTent votes `false` on the disable-remaining channel.
    #[test]
    fn listener_miniature_tent_keeps_remaining_options_enabled() {
        use sts2_core::runstate::{RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MiniatureTent);
        let disable = rs.should_disable_remaining_rest_site_options(RestSiteOption::Rest);
        assert!(
            !disable,
            "MiniatureTent must keep remaining options enabled"
        );
    }

    /// Default (no MiniatureTent) is `true` — picking one rest-site
    /// option disables the rest.
    #[test]
    fn rest_site_disable_default_when_no_listener_or_opinion() {
        use sts2_core::runstate::{RestSiteOption, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        assert!(rs.should_disable_remaining_rest_site_options(RestSiteOption::Rest));
    }

    /// Resolving a ChoicesParadox choice tags the new card for retention.
    #[test]
    fn resolve_choices_paradox_choice_retains_card() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_player_turn_start;
        use sts2_core::{apply, Action, CombatState, Relic, RelicKind};

        crate::relics::dispatch::ensure_installed();
        let mut s = CombatState::default();
        // Alive player -- a dead one trips the C# IsOverOrEnding select guard.
        s.player.current_hp = 80;
        s.round = 1;
        s.player.character = Some(CharacterId::Ironclad);
        s.relics.push(Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::ChoicesParadox,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_after_player_turn_start(&mut s);
        let hand_before = s.player.hand.len();
        let _ = apply(
            &mut s,
            Action::ResolveChoice {
                selection: smallvec![0u8],
            },
        );
        assert_eq!(s.player.hand.len(), hand_before + 1);
        let added_uid = *s.player.hand.last().expect("card was added");
        // C# `ChoicesParadox.AfterPlayerTurnStart` applies the persistent
        // Retain KEYWORD (`CardCmd.ApplyKeyword(item, CardKeyword.Retain)`),
        // not a one-turn retain: an unplayed pick must stay retained on turn 2+.
        let added = sts2_core::card_by_uid(&s.cards, added_uid).expect("card exists");
        assert!(
            sts2_core::card_effects::card_retain_inst(added),
            "added card must carry the persistent Retain keyword",
        );
    }

    // -----------------------------------------------------------------
    // GhostSeed listener tests — bead `sts2-rs-sei89`
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn ghost_seed_push_card(
        state: &mut CombatState,
        pile: GhostSeedPile,
        card_id: sts2_core::cards::CardId,
    ) -> sts2_core::CardUid {
        use std::num::NonZeroU32;
        use sts2_core::cards::{Card, CardTags, CardType, TargetType};
        let next_raw = state
            .cards
            .iter()
            .map(|c| c.uid.0.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let uid = sts2_core::CardUid(NonZeroU32::new(next_raw).expect("non-zero"));
        state.cards.push(Card {
            uid,
            card_id,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        match pile {
            GhostSeedPile::Draw => state.player.draw.push(uid),
            GhostSeedPile::Hand => state.player.hand.push(uid),
            GhostSeedPile::Discard => state.player.discard.push(uid),
            GhostSeedPile::Exhaust => state.player.exhaust.push(uid),
        }
        uid
    }

    #[cfg(test)]
    #[derive(Copy, Clone)]
    enum GhostSeedPile {
        Draw,
        Hand,
        Discard,
        Exhaust,
    }

    #[cfg(test)]
    fn card_has_ethereal_this_combat(state: &CombatState, uid: sts2_core::CardUid) -> bool {
        let idx = uid.0.get() as usize - 1;
        state.cards.get(idx).is_some_and(|c| {
            c.combat_vars
                .iter()
                .any(|(k, v)| k == VarKey::EtherealThisCombat && v == 1)
        })
    }

    /// **GhostSeed #1** — happy path: every basic Strike/Defend across
    /// every pile gets the `EtherealThisCombat` stamp.
    #[test]
    fn listener_ghost_seed_stamps_basic_strike_defend_across_all_piles() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::GhostSeed);
        let in_draw = ghost_seed_push_card(&mut state, GhostSeedPile::Draw, CardId::StrikeIronclad);
        let in_hand = ghost_seed_push_card(&mut state, GhostSeedPile::Hand, CardId::DefendIronclad);
        let in_disc =
            ghost_seed_push_card(&mut state, GhostSeedPile::Discard, CardId::StrikeSilent);
        let in_exh = ghost_seed_push_card(&mut state, GhostSeedPile::Exhaust, CardId::DefendDefect);

        let _ = dispatch_before_combat_start(&mut state);

        assert!(card_has_ethereal_this_combat(&state, in_draw));
        assert!(card_has_ethereal_this_combat(&state, in_hand));
        assert!(card_has_ethereal_this_combat(&state, in_disc));
        assert!(card_has_ethereal_this_combat(&state, in_exh));
    }

    /// **GhostSeed #2** — non-Strike/Defend cards are left alone.
    #[test]
    fn listener_ghost_seed_skips_non_strike_defend_cards() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::GhostSeed);
        // Bash and a generic non-basic card; neither is a basic
        // Strike/Defend so neither should be stamped.
        let bash = ghost_seed_push_card(&mut state, GhostSeedPile::Hand, CardId::Bash);

        let _ = dispatch_before_combat_start(&mut state);

        assert!(!card_has_ethereal_this_combat(&state, bash));
    }

    /// **GhostSeed #3** — derived-name cards like `SetupStrikeCard`
    /// (Python `CardRarity != BASIC`) are NOT stamped, even though
    /// their name contains "STRIKE".
    #[test]
    fn listener_ghost_seed_skips_setup_strike_card() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::GhostSeed);
        let setup = ghost_seed_push_card(&mut state, GhostSeedPile::Hand, CardId::SetupStrikeCard);

        let _ = dispatch_before_combat_start(&mut state);

        assert!(
            !card_has_ethereal_this_combat(&state, setup),
            "SetupStrikeCard is not basic — must be skipped"
        );
    }

    /// **GhostSeed #4** — no-op without relic ownership.
    #[test]
    fn listener_ghost_seed_no_stamp_without_ownership() {
        use sts2_core::cards::CardId;
        use sts2_core::relic_effects::dispatch_before_combat_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        // Set up player creature without equipping GhostSeed.
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        let strike = ghost_seed_push_card(&mut state, GhostSeedPile::Hand, CardId::StrikeIronclad);

        let _ = dispatch_before_combat_start(&mut state);

        assert!(!card_has_ethereal_this_combat(&state, strike));
    }

    // -----------------------------------------------------------------
    // PandorasBox listener tests — bead `sts2-rs-70anj.4` (PandorasBox)
    // -----------------------------------------------------------------

    /// Seeds an Ironclad master_deck with 5 basic Strikes + 4 basic
    /// Defends + 1 non-basic (uses CardId::Anger as a sentinel
    /// non-basic Ironclad card). Returns the original deck length.
    fn pandoras_box_seed_ironclad_deck(rs: &mut sts2_core::runstate::RunState) -> usize {
        use sts2_core::cards::CardId as InnerCardId;
        rs.master_deck.clear();
        for _ in 0..5 {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = InnerCardId::StrikeIronclad;
            rs.master_deck.push(c);
        }
        for _ in 0..4 {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = InnerCardId::DefendIronclad;
            rs.master_deck.push(c);
        }
        let mut anchor = sts2_core::Card::pod_defaults();
        anchor.card_id = InnerCardId::Anger;
        rs.master_deck.push(anchor);
        rs.master_deck.len()
    }

    /// Pickup with ownership replaces every basic Strike + Defend in
    /// the master deck with a non-basic character-pool card; the deck
    /// length is unchanged and the non-basic anchor card survives
    /// (shifted to the head, since the C# deck transform removes each
    /// basic and appends its replacement to the tail).
    #[test]
    fn listener_pandoras_box_transforms_all_basic_strikes_and_defends() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.character_id = CharacterId::Ironclad;
        rs.relics.push(RelicId::PandorasBox);
        let deck_len = pandoras_box_seed_ironclad_deck(&mut rs);

        rs.last_obtained_relic = Some(RelicId::PandorasBox);
        let _ = dispatch_after_obtained(&mut rs);

        assert_eq!(
            rs.master_deck.len(),
            deck_len,
            "transform_specific_cards removes+appends — deck length preserved"
        );
        // Every basic Strike + Defend slot should be replaced; the
        // anchor non-basic (Anger at the tail) survives.
        let basics_remaining = rs
            .master_deck
            .iter()
            .filter(|c| {
                c.card_id == InnerCardId::StrikeIronclad || c.card_id == InnerCardId::DefendIronclad
            })
            .count();
        assert_eq!(
            basics_remaining, 0,
            "every basic Strike + Defend must be transformed"
        );
        // All 9 basics (indices 0..8) are removed and their replacements
        // appended, so the non-basic anchor (Anger, originally at the tail)
        // shifts to the head; the 9 replacements follow it.
        assert_eq!(
            rs.master_deck[0].card_id,
            InnerCardId::Anger,
            "non-basic anchor shifts to the head after all basics are removed",
        );
    }

    /// `PandorasBox.AfterObtained` (`PandorasBox.cs:22`, same in both builds):
    /// one default-transform roll per basic, in deck order, off the `Niche`
    /// stream (not the old Common-only `rewards` sampler; GH #806).
    #[test]
    fn listener_pandoras_box_rolls_niche_transform_pool_both_versions() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        for version in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut rs = RunState::new();
            rs.game_version = version;
            rs.character_id = CharacterId::Ironclad;
            rs.relics.push(RelicId::PandorasBox);
            let deck_len = pandoras_box_seed_ironclad_deck(&mut rs);

            // Expected: the same rolls NewLeaf makes, one per basic in deck order.
            let mut expect_rs = rs.clone();
            let originals: Vec<InnerCardId> =
                rs.master_deck[..9].iter().map(|c| c.card_id).collect();
            let expected: Vec<InnerCardId> = originals
                .iter()
                .map(|&o| {
                    crate::cards::spec_index::roll_transform_replacement_basic_niche(
                        &mut expect_rs,
                        CharacterId::Ironclad,
                        o,
                    )
                    .expect("non-empty pool")
                })
                .collect();

            rs.last_obtained_relic = Some(RelicId::PandorasBox);
            let _ = dispatch_after_obtained(&mut rs);

            assert_eq!(rs.master_deck.len(), deck_len);
            assert_eq!(rs.master_deck[0].card_id, InnerCardId::Anger);
            let got: Vec<InnerCardId> = rs.master_deck[1..].iter().map(|c| c.card_id).collect();
            assert_eq!(got, expected, "{version:?}");
            assert_eq!(rs.rng.niche.counter(), expect_rs.rng.niche.counter());
        }
    }

    /// No-op when PandorasBox is not in `rs.relics`.
    #[test]
    fn listener_pandoras_box_is_noop_without_ownership() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::potions::CharacterId;
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.character_id = CharacterId::Ironclad;
        let deck_len = pandoras_box_seed_ironclad_deck(&mut rs);

        let _ = dispatch_after_obtained(&mut rs);

        assert_eq!(rs.master_deck.len(), deck_len);
        let basics_remaining = rs
            .master_deck
            .iter()
            .filter(|c| {
                c.card_id == InnerCardId::StrikeIronclad || c.card_id == InnerCardId::DefendIronclad
            })
            .count();
        assert_eq!(
            basics_remaining, 9,
            "deck must be untouched without ownership"
        );
    }

    // REMOVED with the sixth `CharacterId` variant (2026-08-08):
    // `listener_pandoras_box_is_noop_for_unmapped_character` pinned the no-op
    // path taken when `basic_strike_defend_cards` returns `None`. Its only
    // unmapped character was one STS2 does not have (Necrobinder and Regent
    // have since been mapped), so that function is now total: every arm returns
    // `Some`. Its `Option` return and this listener's `None` branch are both
    // dead; collapsing them is a separate change.

    // -----------------------------------------------------------------
    // SilverCrucible listener tests — bead `sts2-rs-535i7`
    // -----------------------------------------------------------------

    /// Pack/unpack roundtrip pins the encoding contract: the same i32
    /// stores both counters independently.
    #[test]
    fn silver_crucible_pack_unpack_roundtrip() {
        let packed = super::silver_crucible_pack(2, 5);
        assert_eq!(super::silver_crucible_times_used(packed), 2);
        assert_eq!(super::silver_crucible_treasure_rooms_entered(packed), 5);
        // Edge: max bounds.
        let packed = super::silver_crucible_pack(0xFF, 0xFF);
        assert_eq!(super::silver_crucible_times_used(packed), 0xFF);
        assert_eq!(super::silver_crucible_treasure_rooms_entered(packed), 0xFF);
        // Edge: negative clamps to 0.
        let packed = super::silver_crucible_pack(-1, -42);
        assert_eq!(super::silver_crucible_times_used(packed), 0);
        assert_eq!(super::silver_crucible_treasure_rooms_entered(packed), 0);
    }

    /// `after_room_entered` increments the treasure-rooms counter
    /// on `RoomType::Treasure` and ignores other room types.
    #[test]
    fn silver_crucible_after_room_entered_increments_only_on_treasure() {
        use sts2_core::run_effects::dispatch_after_room_entered;
        use sts2_core::runstate::{RoomType, RunState};
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::SilverCrucible);

        let _ = dispatch_after_room_entered(&mut rs, RoomType::Monster);
        let packed = rs
            .relic_counters
            .get(&RelicId::SilverCrucible)
            .copied()
            .unwrap_or(0);
        assert_eq!(
            super::silver_crucible_treasure_rooms_entered(packed),
            0,
            "Monster room must not bump the treasure counter"
        );

        let _ = dispatch_after_room_entered(&mut rs, RoomType::Treasure);
        let packed = rs.relic_counters[&RelicId::SilverCrucible];
        assert_eq!(super::silver_crucible_treasure_rooms_entered(packed), 1);

        let _ = dispatch_after_room_entered(&mut rs, RoomType::Treasure);
        let packed = rs.relic_counters[&RelicId::SilverCrucible];
        assert_eq!(super::silver_crucible_treasure_rooms_entered(packed), 2);
    }

    /// `should_generate_treasure` vetoes the first treasure
    /// (counter == 1 after `after_room_entered`) and allows
    /// subsequent treasures (counter >= 2).
    #[test]
    fn silver_crucible_should_generate_treasure_skips_first_only() {
        use sts2_core::run_effects::{
            dispatch_after_room_entered, dispatch_should_generate_treasure,
        };
        use sts2_core::runstate::{RoomType, RunState};
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::SilverCrucible);

        // Simulate the room-driver flow: after_room_entered fires first,
        // then should_generate_treasure is consulted.
        let _ = dispatch_after_room_entered(&mut rs, RoomType::Treasure);
        assert!(
            !dispatch_should_generate_treasure(&mut rs),
            "first treasure must be skipped (counter == 1)"
        );

        let _ = dispatch_after_room_entered(&mut rs, RoomType::Treasure);
        assert!(
            dispatch_should_generate_treasure(&mut rs),
            "second treasure must NOT be skipped (counter == 2)"
        );
    }

    /// `modify_card_reward_options_late` pre-upgrades up to 3 reward
    /// generations, then stops. Counter ticks up by 1 per dispatch.
    #[test]
    fn silver_crucible_pre_upgrades_first_three_card_rewards_then_stops() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::run_effects::dispatch_modify_card_reward_options_late;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::SilverCrucible);

        fn fresh_card() -> sts2_core::Card {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = InnerCardId::StrikeIronclad;
            c.upgraded = false;
            c
        }

        for i in 1..=3 {
            let mut cards = vec![fresh_card(), fresh_card(), fresh_card()];
            let _ = dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
            for card in &cards {
                assert!(card.upgraded, "reward #{i} must pre-upgrade every option");
            }
            let packed = rs.relic_counters[&RelicId::SilverCrucible];
            assert_eq!(
                super::silver_crucible_times_used(packed),
                i,
                "_times_used must tick by 1 per dispatch"
            );
        }

        // 4th dispatch: no more pre-upgrades.
        let mut cards = vec![fresh_card()];
        let _ = dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(
            !cards[0].upgraded,
            "4th reward must NOT pre-upgrade (saturated at CARD_REWARDS=3)"
        );
        let packed = rs.relic_counters[&RelicId::SilverCrucible];
        assert_eq!(
            super::silver_crucible_times_used(packed),
            3,
            "_times_used stays clamped at CARD_REWARDS once saturated"
        );
    }

    /// Ownership gate: all three listeners no-op when SilverCrucible
    /// isn't in `rs.relics`. Compact pin since each branch has the
    /// same `if !contains { return NotImplemented }` head — one fail
    /// reading from each channel is enough.
    #[test]
    fn silver_crucible_listeners_noop_without_ownership() {
        use sts2_core::cards::CardId as InnerCardId;
        use sts2_core::run_effects::{
            dispatch_after_room_entered, dispatch_modify_card_reward_options_late,
        };
        use sts2_core::runstate::{RoomType, RunState};
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();

        let _ = dispatch_after_room_entered(&mut rs, RoomType::Treasure);
        assert!(
            !rs.relic_counters.contains_key(&RelicId::SilverCrucible),
            "after_room_entered must not allocate a counter slot without ownership"
        );

        let mut cards = {
            let mut c = sts2_core::Card::pod_defaults();
            c.card_id = InnerCardId::StrikeIronclad;
            vec![c]
        };
        let _ = dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(
            !cards[0].upgraded,
            "modify_card_reward_options_late must not upgrade without ownership"
        );
    }

    // -----------------------------------------------------------------
    // GoldenCompass listener tests — bead `sts2-rs-k2nuu`
    // -----------------------------------------------------------------

    #[test]
    fn golden_compass_after_obtained_captures_current_act_index() {
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::GoldenCompass);
        rs.act_index = 2;
        rs.last_obtained_relic = Some(RelicId::GoldenCompass);
        let _ = dispatch_after_obtained(&mut rs);
        let captured = rs
            .relic_counters
            .get(&RelicId::GoldenCompass)
            .copied()
            .unwrap_or(0);
        assert_eq!(captured, 2, "after_obtained must capture rs.act_index");
    }

    #[test]
    fn golden_compass_after_obtained_is_noop_without_ownership() {
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.act_index = 2;
        let _ = dispatch_after_obtained(&mut rs);
        assert!(
            !rs.relic_counters.contains_key(&RelicId::GoldenCompass),
            "no counter slot allocated without ownership"
        );
    }

    /// `modify_generated_map` overwrites the act_map with the
    /// golden-path layout on the captured act. The point-count
    /// matches `generate_golden_path_map(1)` exactly.
    #[test]
    fn golden_compass_modify_generated_map_swaps_on_captured_act() {
        use sts2_core::map_generator::generate_golden_path_map;
        use sts2_core::run_effects::dispatch_modify_generated_map;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::GoldenCompass);
        rs.relic_counters.insert(RelicId::GoldenCompass, 1);

        let mut act_map = sts2_core::map::ActMap::new(0);
        let _ = dispatch_modify_generated_map(&mut rs, &mut act_map, 1);

        let expected = generate_golden_path_map(1);
        assert_eq!(
            act_map.points.len(),
            expected.points.len(),
            "post-listener map must match golden-path point count"
        );
    }

    /// `modify_generated_map` is a no-op when the dispatched
    /// act_index doesn't match the captured act.
    #[test]
    fn golden_compass_modify_generated_map_skips_other_acts() {
        use sts2_core::run_effects::dispatch_modify_generated_map;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::GoldenCompass);
        rs.relic_counters.insert(RelicId::GoldenCompass, 1);

        let mut act_map = sts2_core::map::ActMap::new(42);
        let original_len = act_map.points.len();
        let _ = dispatch_modify_generated_map(&mut rs, &mut act_map, 2);
        assert_eq!(
            act_map.points.len(),
            original_len,
            "non-captured-act dispatch must leave act_map untouched"
        );
    }

    /// `modify_unknown_map_point_room_types` narrows the candidate
    /// set to {Event} on the captured act.
    #[test]
    fn golden_compass_modify_unknown_map_point_room_types_narrows_to_event() {
        use sts2_core::run_effects::dispatch_modify_unknown_map_point_room_types;
        use sts2_core::runstate::{RoomType, RunState};
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::GoldenCompass);
        rs.act_index = 1;
        rs.relic_counters.insert(RelicId::GoldenCompass, 1);

        let mut room_types: std::collections::BTreeSet<RoomType> = [
            RoomType::Monster,
            RoomType::Shop,
            RoomType::RestSite,
            RoomType::Event,
            RoomType::Treasure,
        ]
        .into_iter()
        .collect();
        let _ = dispatch_modify_unknown_map_point_room_types(&mut rs, &mut room_types);
        assert_eq!(room_types.len(), 1);
        assert!(room_types.contains(&RoomType::Event));
    }

    /// On a non-captured act, the candidate set is untouched.
    #[test]
    fn golden_compass_modify_unknown_map_point_room_types_skips_other_acts() {
        use sts2_core::run_effects::dispatch_modify_unknown_map_point_room_types;
        use sts2_core::runstate::{RoomType, RunState};
        crate::relics::dispatch::ensure_installed();

        let mut rs = RunState::new();
        rs.relics.push(RelicId::GoldenCompass);
        rs.act_index = 2;
        rs.relic_counters.insert(RelicId::GoldenCompass, 1);

        let original: std::collections::BTreeSet<RoomType> =
            [RoomType::Monster, RoomType::Shop].into_iter().collect();
        let mut room_types = original.clone();
        let _ = dispatch_modify_unknown_map_point_room_types(&mut rs, &mut room_types);
        assert_eq!(
            room_types, original,
            "non-captured-act dispatch must leave candidates untouched"
        );
    }

    // ── cargo-mutants coverage (issue #65) ────────────────────────────
    // Kills surviving mutants in the shop_event listeners + enchant
    // predicates that the behavioural pins above did not discriminate.

    /// `PaelsLegion::after_side_turn_start` must NOT decrement an
    /// already-armed (counter == 0) cooldown. Kills `> 0` → `>= 0` at
    /// the decrement guard: with `>= 0` a re-armed counter would drift
    /// to -1 (a phantom "pending" state).
    #[test]
    fn paels_legion_turn_start_leaves_armed_counter_at_zero() {
        use std::num::NonZeroU32;
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();

        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind: RelicKind::PaelsLegion,
            counter: 0, // already armed
            is_wax: false,
            is_melted: false,
        });
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            paels_legion_counter(&state),
            0,
            "armed (0) counter must stay 0 — not decrement to -1"
        );
    }

    // PaelsEye: bit 0 = used-this-combat, bit 1 = any-cards-played.
    // before_turn_end exhausts the hand only when BOTH bits are clear.

    #[cfg(test)]
    fn paels_eye_counter(state: &CombatState) -> i32 {
        state
            .relics
            .iter()
            .find(|r| !r.is_melted && r.kind == RelicKind::PaelsEye)
            .expect("PaelsEye present")
            .counter
    }

    #[cfg(test)]
    fn paels_eye_event_ctx() -> sts2_core::relic_effects::CardEventCtx {
        sts2_core::relic_effects::CardEventCtx {
            card_id: sts2_core::cards::CardId::DefendIronclad,
            card_uid: None,
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        }
    }

    /// The relic `after_card_played` bus also fires for auto-plays
    /// (`finish_auto_played_card_tail`), which C# `AnyCardsPlayedThisTurn`
    /// skips, so it must NOT set bit 1; the hand-play path sets it instead
    /// (`sts2-wire/tests/paels_eye.rs`).
    #[test]
    fn paels_eye_after_card_played_bus_leaves_any_cards_bit() {
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsEye);
        let _ = dispatch_after_card_played(&mut state, &paels_eye_event_ctx());
        assert_eq!(paels_eye_counter(&state), 0, "bus does not set the bit");
    }

    /// `PaelsEye::before_side_turn_start` clears bit 1 on the player
    /// turn. Kills `&= !0b10` → `|= !0b10`, the `delete !`, and the
    /// side/ownership guards.
    #[test]
    fn paels_eye_side_turn_start_clears_any_cards_bit() {
        use sts2_core::relic_effects::dispatch_before_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsEye);
        // Both bits set: clearing must leave only bit 0 (used).
        state
            .relics
            .iter_mut()
            .find(|r| !r.is_melted && r.kind == RelicKind::PaelsEye)
            .unwrap()
            .counter = 0b11;
        // Enemy side: no clear.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(
            paels_eye_counter(&state),
            0b11,
            "enemy side-turn-start must not clear the bit"
        );
        // Player side: clears bit 1 only.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            paels_eye_counter(&state),
            0b01,
            "player side-turn-start clears bit 1, preserves bit 0"
        );
    }

    /// `PaelsEye::before_turn_end` exhausts the hand iff both bits are
    /// clear (`trailing_zeros() >= 2`). Kills the `delete !`, the `>=`
    /// boundary, the side guard, and the `&&`/`==` guards.
    #[test]
    fn paels_eye_before_turn_end_exhausts_only_when_both_bits_clear() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::relic_effects::dispatch_before_turn_end_early;
        crate::relics::dispatch::ensure_installed();

        let stage_hand = |state: &mut CombatState| {
            let c = Card {
                card_id: CardId::StrikeIronclad,
                card_type: CardType::Attack,
                ..Card::pod_defaults()
            };
            let uid = state.add_card_to_draw_pile(c);
            state.player.draw.retain(|u| *u != uid);
            state.player.hand.push(uid);
        };

        // Case A: counter == 0 (both clear) → exhaust the whole hand.
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsEye);
        stage_hand(&mut state);
        assert_eq!(state.player.hand.len(), 1);
        let _ = dispatch_before_turn_end_early(&mut state, Side::Player);
        assert!(
            state.player.hand.is_empty(),
            "both bits clear → hand exhausted"
        );

        // Case B: any-cards bit set (counter == 0b10) → no exhaust.
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsEye);
        state
            .relics
            .iter_mut()
            .find(|r| !r.is_melted && r.kind == RelicKind::PaelsEye)
            .unwrap()
            .counter = 0b10;
        stage_hand(&mut state);
        let _ = dispatch_before_turn_end_early(&mut state, Side::Player);
        assert_eq!(
            state.player.hand.len(),
            1,
            "any-cards bit set → hand NOT exhausted"
        );

        // Case C: both clear but ENEMY side → no exhaust (side guard).
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsEye);
        stage_hand(&mut state);
        let _ = dispatch_before_turn_end_early(&mut state, Side::Enemy);
        assert_eq!(
            state.player.hand.len(),
            1,
            "enemy turn-end must not exhaust the player hand"
        );
    }

    /// Pael's Eye exhausts the hand in C#'s `BeforeSideTurnEndEarly` pass
    /// (`PaelsEye.cs:118-128`, both builds), a full listener walk that
    /// `Hook.BeforeSideTurnEnd` (Hook.cs:1388-1417; v0.107.1 `BeforeTurnEnd`)
    /// finishes before the normal `BeforeSideTurnEnd` pass, so Screaming
    /// Flagon (`ScreamingFlagon.cs:21`, normal) sees the emptied hand and
    /// deals its 20 to every enemy. Rust ran both on one relic walk in
    /// `RelicId` order (Flagon first), checking the still-full hand
    /// (Ironclad 4147 / 4119 act 2: every monster 20 HP higher).
    #[test]
    fn paels_eye_exhausts_hand_before_screaming_flagon_checks_it() {
        use sts2_core::cards::{Card, CardId, CardType};
        use sts2_core::{CombatPhase, MonsterKind};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::ScreamingFlagon);
        state.relics.push(sts2_core::Relic {
            id: std::num::NonZeroU32::new(2).expect("non-zero"),
            kind: RelicKind::PaelsEye,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        let hp_of = |state: &CombatState| {
            state
                .creatures
                .iter()
                .find(|c| c.id == enemy.0)
                .expect("enemy")
                .current_hp
        };
        let hp_before = hp_of(&state);
        let c = Card {
            card_id: CardId::StrikeIronclad,
            card_type: CardType::Attack,
            ..Card::pod_defaults()
        };
        let uid = state.add_card_to_draw_pile(c);
        state.player.draw.retain(|u| *u != uid);
        state.player.hand.push(uid);
        state.phase = CombatPhase::PlayerPlayPhase;

        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);

        assert!(
            state.player.exhaust.contains(&uid),
            "Pael's Eye exhausted the hand"
        );
        assert_eq!(
            hp_of(&state),
            hp_before - SCREAMING_FLAGON_DAMAGE,
            "Flagon sees the empty hand and fires"
        );
    }

    // MusicBox: bit 0 = used-this-turn. First Attack played clones into
    // hand (ethereal), then the latch blocks further clones until the
    // next player side-turn-start re-arms it.

    #[cfg(test)]
    fn music_box_counter(state: &CombatState) -> i32 {
        state
            .relics
            .iter()
            .find(|r| !r.is_melted && r.kind == RelicKind::MusicBox)
            .expect("MusicBox present")
            .counter
    }

    #[cfg(test)]
    fn music_box_play_card(
        state: &mut CombatState,
        card_type: sts2_core::cards::CardType,
    ) -> sts2_core::relic_effects::CardEventCtx {
        use sts2_core::cards::{Card, CardId};
        let c = Card {
            card_id: CardId::StrikeIronclad,
            card_type,
            ..Card::pod_defaults()
        };
        let uid = state.add_card_to_draw_pile(c);
        let ctx = sts2_core::relic_effects::CardEventCtx {
            card_id: CardId::StrikeIronclad,
            card_uid: Some(uid),
            target: None,
            card_tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
        };
        // Every play starts with `BeforeCardPlayed`, where MusicBox latches.
        let _ = sts2_core::relic_effects::dispatch_before_card_played(state, &ctx);
        ctx
    }

    /// GH #1110: an Attack auto-played inside another Attack finishes first,
    /// but C# clones the OUTER card, the first Attack to start
    /// (`MusicBox.BeforeCardPlayed` latches it). The inner card's
    /// `AfterCardPlayed` must not clone or use up the relic.
    #[test]
    fn music_box_clones_the_first_attack_to_start_not_to_finish() {
        use sts2_core::cards::{CardId, CardType};
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MusicBox);

        let outer = music_box_play_card(&mut state, CardType::Attack);
        let mut inner = music_box_play_card(&mut state, CardType::Attack);
        let inner_uid = inner.card_uid.unwrap();
        state.cards[inner_uid.0.get() as usize - 1].card_id = CardId::Bash;
        inner.card_id = CardId::Bash;

        let hand_before = state.player.hand.len();
        let _ = dispatch_after_card_played(&mut state, &inner);
        assert_eq!(
            state.player.hand.len(),
            hand_before,
            "the inner Attack finished first but was not the one latched"
        );
        assert_eq!(music_box_counter(&state) & 0b01, 0, "relic still unused");

        let _ = dispatch_after_card_played(&mut state, &outer);
        assert_eq!(state.player.hand.len(), hand_before + 1);
        let clone_uid = *state.player.hand.last().unwrap();
        assert_eq!(
            state.cards[clone_uid.0.get() as usize - 1].card_id,
            CardId::StrikeIronclad,
            "the clone is the outer (first-started) Attack"
        );
        assert_eq!(music_box_counter(&state) & 0b01, 0b01);
        assert_eq!(state.combat_log.music_box_card_being_played, None);
    }

    /// `MusicBox::after_card_played` clones the first Attack into hand
    /// and latches (bit 0). Kills the `card_type == Attack` guard
    /// (`== / !=`), the latch `&` mask mutations, `|=` → `&=`, the
    /// `delete !` on the consumed guard, and the `&& / ==` in the latch
    /// check.
    #[test]
    fn music_box_clones_first_attack_then_latches() {
        use sts2_core::cards::{CardType, VarKey};
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MusicBox);
        assert_eq!(music_box_counter(&state), 0, "latch starts clear");

        // First Attack → clone into hand, latch set.
        let ctx = music_box_play_card(&mut state, CardType::Attack);
        let hand_before = state.player.hand.len();
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "first Attack must clone a card into hand"
        );
        assert_eq!(
            music_box_counter(&state) & 0b01,
            0b01,
            "latch must set bit 0"
        );
        let clone_uid = *state.player.hand.last().unwrap();
        assert_eq!(
            state.cards[clone_uid.0.get() as usize - 1].combat_var(VarKey::EtherealThisCombat),
            1,
            "clone must be flagged EtherealThisCombat"
        );

        // Second Attack → latched, no further clone.
        let ctx2 = music_box_play_card(&mut state, CardType::Attack);
        let hand_now = state.player.hand.len();
        let _ = dispatch_after_card_played(&mut state, &ctx2);
        assert_eq!(
            state.player.hand.len(),
            hand_now,
            "latched MusicBox must not clone a second Attack"
        );
    }

    /// C# `MusicBox.AfterCardPlayed` (MusicBox.cs:73-84) adds a
    /// `CreateClone()` — a NEW instance, absent from
    /// `History.CardPlaysFinished`. ThrummingHatchet / Bolas return only a card
    /// whose OWN play is in that history (ThrummingHatchet.cs:35-40,
    /// Bolas.cs:41-46), so the clone must not inherit the played card's
    /// `ReturnIfPlayedRound` stamp (Ironclad-6161-a0-act2#b0 @591: the
    /// exhausted Ethereal clone came back to hand next turn).
    #[test]
    fn music_box_clone_does_not_inherit_return_if_played_stamp() {
        use sts2_core::cards::{CardType, VarKey};
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MusicBox);

        let ctx = music_box_play_card(&mut state, CardType::Attack);
        let played = ctx.card_uid.unwrap();
        state.cards[played.0.get() as usize - 1].set_combat_var(VarKey::ReturnIfPlayedRound, 2);
        let _ = dispatch_after_card_played(&mut state, &ctx);
        let clone_uid = *state.player.hand.last().unwrap();
        assert_ne!(clone_uid, played);
        assert_eq!(
            state.cards[clone_uid.0.get() as usize - 1].combat_var(VarKey::ReturnIfPlayedRound),
            0,
            "the clone has never been played, so it carries no return stamp"
        );
        assert_eq!(
            state.cards[played.0.get() as usize - 1].combat_var(VarKey::ReturnIfPlayedRound),
            2,
            "the played original keeps its stamp"
        );
    }

    /// `MusicBox::after_card_played` ignores non-Attack cards — kills
    /// the `card_type == Attack` guard flipped to `!=`.
    #[test]
    fn music_box_ignores_non_attack_cards() {
        use sts2_core::cards::CardType;
        use sts2_core::relic_effects::dispatch_after_card_played;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MusicBox);
        let ctx = music_box_play_card(&mut state, CardType::Skill);
        let hand_before = state.player.hand.len();
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(
            state.player.hand.len(),
            hand_before,
            "Skill play must not clone"
        );
        assert_eq!(
            music_box_counter(&state),
            0,
            "non-Attack must not consume the latch"
        );
    }

    /// `MusicBox::before_side_turn_start` re-arms (clears bit 0) on the
    /// player turn. Kills `&= !0b01` → `|= !0b01`, the `delete !`, and
    /// the side/ownership guards.
    #[test]
    fn music_box_side_turn_start_rearms_latch() {
        use sts2_core::relic_effects::dispatch_before_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::MusicBox);
        // Latch set, plus a stray high bit that must survive the clear.
        state
            .relics
            .iter_mut()
            .find(|r| !r.is_melted && r.kind == RelicKind::MusicBox)
            .unwrap()
            .counter = 0b11;
        // Enemy side: no re-arm.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(
            music_box_counter(&state),
            0b11,
            "enemy side-turn-start must not re-arm"
        );
        // Player side: clears bit 0 only.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            music_box_counter(&state),
            0b10,
            "player side-turn-start clears bit 0, preserves bit 1"
        );
    }

    /// `Crossbow::after_side_turn_start` generates a free Attack in
    /// hand on the player turn. Kills the ownership `== / !=` guards,
    /// the side `delete !`, and the wrong-relic `==` guard.
    #[test]
    fn crossbow_generates_free_attack_on_player_turn() {
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Crossbow);
        state.player.character = Some(CharacterId::Ironclad);

        let hand_before = state.player.hand.len();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            state.player.hand.len(),
            hand_before + 1,
            "Crossbow must add one free Attack to hand on the player turn"
        );
        let uid = *state.player.hand.last().unwrap();
        assert!(
            state.cards[uid.0.get() as usize - 1]
                .cost
                .has(sts2_core::LocalCostMod::FREE_THIS_TURN),
            "generated Attack must be FREE_THIS_TURN"
        );
    }

    /// Crossbow with 10 cards already in hand. C# `CardPileCmd.Add` puts
    /// the generated Attack on the discard pile (`CardPileCmd.cs:485-489`),
    /// and `AddGeneratedCardsToCombat` still records the generation and
    /// fires its hook (`CardPileCmd.cs:308-311`). The card stays free this
    /// turn. This is fidelity gate F9's "hand size 11 exceeds cap 10".
    #[test]
    fn crossbow_with_full_hand_puts_free_attack_on_discard() {
        use sts2_core::cards::HAND_LIMIT;
        use sts2_core::event_log::EventLogEntry;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Crossbow);
        state.player.character = Some(CharacterId::Ironclad);
        let hand: Vec<_> = (0..HAND_LIMIT)
            .map(|_| push_hand_strike(&mut state))
            .collect();
        let discard_before = state.player.discard.len();
        state.combat_log.event_log.set_recording(true);

        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);

        assert_eq!(
            state.player.hand.as_slice(),
            hand.as_slice(),
            "a full hand takes no card"
        );
        assert_eq!(state.player.discard.len(), discard_before + 1);
        let uid = *state.player.discard.last().unwrap();
        let card = &state.cards[uid.0.get() as usize - 1];
        assert_eq!(card.card_type, CardType::Attack);
        assert!(card.cost.has(sts2_core::LocalCostMod::FREE_THIS_TURN));
        assert_eq!(
            state.player.generated_cards_this_combat, 1,
            "the generation counts for Supermassive"
        );
        let generated: Vec<_> = state
            .combat_log
            .event_log
            .rows()
            .iter()
            .filter_map(|r| match r.entry {
                EventLogEntry::CardGenerated {
                    card,
                    pile,
                    by_player,
                    ..
                } => Some((card, pile, by_player)),
                _ => None,
            })
            .collect();
        assert_eq!(generated, vec![(card.card_id, "discard", true)]);
    }

    /// Crossbow's card goes through the generation listeners, as C#
    /// `AddGeneratedCardsToCombat` does: Regalite grants its block for a
    /// card the player generated.
    #[test]
    fn crossbow_card_triggers_regalite() {
        use std::num::NonZeroU32;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        use sts2_core::Relic;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Crossbow);
        state.relics.push(Relic {
            id: NonZeroU32::new(state.relics.len() as u32 + 1).expect("non-zero"),
            kind: RelicKind::Regalite,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        state.player.character = Some(CharacterId::Ironclad);
        let block_before = state.player.block;

        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);

        assert_eq!(state.player.hand.len(), 1);
        assert_eq!(
            state.player.block - block_before,
            super::super::uncommon::REGALITE_BLOCK,
            "Regalite must see the Crossbow card"
        );
    }

    /// v0.111.0: Regalite grants Block for the first created card each
    /// turn only (decomp Regalite.cs:22,39). The Crossbow card is that first
    /// card, so it takes the 4 Block and sets the latch, and a second
    /// generation in the same turn grants nothing (#1105).
    #[test]
    fn crossbow_card_consumes_regalite_latch_v0111() {
        use std::num::NonZeroU32;
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::{
            dispatch_after_side_turn_start, dispatch_before_side_turn_start,
        };
        use sts2_core::Relic;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        equip_relic(&mut state, RelicKind::Crossbow);
        state.relics.push(Relic {
            id: NonZeroU32::new(state.relics.len() as u32 + 1).expect("non-zero"),
            kind: RelicKind::Regalite,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        state.player.character = Some(CharacterId::Ironclad);
        let block_before = state.player.block;
        let regalite_counter = |s: &CombatState| {
            s.relics
                .iter()
                .find(|r| r.kind == RelicKind::Regalite)
                .expect("regalite equipped")
                .counter
        };

        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);

        assert_eq!(
            state.player.block - block_before,
            super::super::uncommon::REGALITE_BLOCK_V0111,
            "the Crossbow card is the turn's first created card"
        );
        assert_eq!(
            regalite_counter(&state),
            1,
            "the Crossbow card set the latch"
        );

        // A second generation the same turn: the latch holds.
        sts2_core::card_effects::generate_free_attack_in_hand(&mut state);
        assert_eq!(state.player.hand.len(), 2);
        assert_eq!(
            state.player.block - block_before,
            super::super::uncommon::REGALITE_BLOCK_V0111,
            "Regalite grants once per turn in v0.111.0"
        );
    }

    /// `Crossbow` is inert on the enemy side and when unowned — kills
    /// the side `delete !` and the ownership guard mutations from the
    /// positive direction.
    #[test]
    fn crossbow_is_noop_on_enemy_side_and_unowned() {
        use sts2_core::potions::CharacterId;
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();

        // Owned, enemy side → no card.
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::Crossbow);
        state.player.character = Some(CharacterId::Ironclad);
        let before = state.player.hand.len();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(state.player.hand.len(), before, "enemy side must not fire");

        // Unowned (wrong relic), player side → no card.
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::PaelsFlesh);
        state.player.character = Some(CharacterId::Ironclad);
        let before = state.player.hand.len();
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            state.player.hand.len(),
            before,
            "unowned Crossbow must not fire"
        );
    }

    /// `SealOfGold` fires at exactly the cost boundary (gold == 5).
    /// Kills `gold < COST` → `gold <= COST`: with `<=`, gold == 5 would
    /// (wrongly) be treated as insufficient and no energy gained.
    #[test]
    fn seal_of_gold_fires_at_exact_cost_boundary() {
        use sts2_core::relic_effects::dispatch_after_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        equip_relic(&mut state, RelicKind::SealOfGold);
        state.gold = SEAL_OF_GOLD_COST; // exactly 5
        let before = state.player.energy;
        let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(
            state.gold, 0,
            "spends exactly the 5-gold cost at the boundary"
        );
        assert_eq!(
            state.player.energy,
            before + SEAL_OF_GOLD_ENERGY,
            "gains energy at the exact boundary (gold == COST)"
        );
    }

    /// `DingyRug` adds colorless whatever the rarity flags
    /// (`DingyRug.cs:13-36` has no rarity gate).
    #[test]
    fn dingy_rug_adds_colorless_with_rarity_modifications_off() {
        use sts2_core::runstate::{CardRewardGenerationOptions, RunState};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.character_id = sts2_core::potions::CharacterId::Ironclad;
        rs.relics.push(RelicId::DingyRug);
        let mut opts = CardRewardGenerationOptions {
            allow_rarity_modifications: false,
            ..CardRewardGenerationOptions::default()
        };
        let _ = sts2_core::run_effects::dispatch_modify_card_reward_creation_options(
            &mut rs, &mut opts,
        );
        assert!(opts.include_colorless);
    }

    /// `LeafyPoultice` ownership/last-obtained guard: the listener must
    /// be inert when a DIFFERENT relic was just obtained. Kills the
    /// `last_obtained_relic == Some(LeafyPoultice)` guard flipped to
    /// `!=` (which would fire on every other relic's obtain).
    #[test]
    fn leafy_poultice_noop_when_other_relic_obtained() {
        use sts2_core::run_effects::dispatch_after_obtained;
        use sts2_core::runstate::RunState;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LeafyPoultice);
        // A different relic was the one just obtained.
        rs.last_obtained_relic = Some(RelicId::DingyRug);
        let before_max = rs.max_hp;
        let _ = dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.max_hp, before_max,
            "LeafyPoultice must not lose max HP when a different relic was obtained"
        );
    }

    /// `enchant_eligible_deck_cards` guard `max_count == 0 || deck empty`.
    /// Kills `|| → &&` (with `&&`, a zero max_count over a non-empty
    /// deck would wrongly proceed) and the empty-deck leg.
    #[test]
    fn enchant_eligible_deck_cards_guard_branches() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;

        // max_count == 0, non-empty deck → nothing enchanted.
        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        let applied = enchant_eligible_deck_cards(&mut rs, EnchantmentId::Sharp, 3, 0);
        assert_eq!(applied, 0, "max_count == 0 must enchant nothing");
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::Sharp),
            0,
            "no enchantment written when max_count == 0"
        );

        // Empty deck, max_count > 0 → nothing enchanted, no panic.
        let mut rs_empty = RunState::new();
        let applied = enchant_eligible_deck_cards(&mut rs_empty, EnchantmentId::Sharp, 3, 3);
        assert_eq!(applied, 0, "empty deck must enchant nothing");
    }

    /// `enchant_eligible_deck_cards` looks up each card's OWN spec via
    /// `s.id == card.card_id`. Kills the `== → !=` in the `find`: a
    /// mismatched-spec lookup makes the Attack-only `Sharp` gate read
    /// the wrong card type and skip the eligible Strike.
    #[test]
    fn enchant_eligible_deck_cards_matches_own_spec() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;

        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        let applied = enchant_eligible_deck_cards(&mut rs, EnchantmentId::Sharp, 3, 3);
        assert_eq!(
            applied, 1,
            "the lone Strike is Sharp-eligible and enchanted"
        );
        assert_eq!(
            rs.master_deck[0].enchantment(EnchantmentId::Sharp),
            3,
            "Strike must receive Sharp(3) via its OWN spec lookup"
        );
    }

    /// `enchantable_candidate_card_ids` collects ids whose OWN spec is
    /// enchant-eligible (`s.id == card.card_id`). Kills the `== → !=`
    /// in the `find`: the mismatched spec would reject the eligible
    /// Strike (Sharp is Attack-only).
    #[test]
    fn enchantable_candidate_card_ids_matches_own_spec() {
        use sts2_core::cards::{CardId, EnchantmentId};
        use sts2_core::runstate::RunState;

        let mut rs = RunState::new();
        push_deck_card(&mut rs, CardId::StrikeIronclad);
        let ids = enchantable_candidate_card_ids(&rs, EnchantmentId::Sharp);
        assert_eq!(
            ids,
            vec![CardId::StrikeIronclad],
            "the Sharp-eligible Strike must be collected via its own spec"
        );
    }
}

/// Found by the pep299 fork: these two relics opened no card select.
#[cfg(test)]
mod interactive_enchant_on_obtain_tests {
    use super::*;
    use sts2_core::cards::{CardId, EnchantmentId};
    use sts2_core::potions::CharacterId;
    use sts2_core::run_manager::{ActionResult, RunAction, RunManager};

    fn manager_for(relic: RelicId, cards: &[CardId]) -> RunManager {
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut rm = RunManager::new(0, CharacterId::Ironclad);
        rm.run_state.set_agent_decisions();
        rm.run_state.master_deck.clear();
        for &id in cards {
            push_deck_card(&mut rm.run_state, id);
        }
        rm.run_state.obtain_relic(relic);
        rm
    }

    fn stacks(rm: &RunManager, ench: EnchantmentId) -> Vec<i32> {
        let deck = rm.run_state.master_deck.iter();
        deck.map(|c| c.enchantment(ench)).collect()
    }

    #[test]
    fn tri_boomerang_enchants_the_three_attacks_the_player_picks() {
        use CardId::{DefendIronclad as Defend, StrikeIronclad as Strike};
        let mut rm = manager_for(
            RelicId::TriBoomerang,
            &[Strike, Strike, Defend, Strike, Strike],
        );
        assert_eq!(rm.run_state.pending_deck_enchants.len(), 3);
        // Attacks only (`Instinct.CanEnchantCardType`).
        assert_eq!(rm.enchant_eligible_deck_indices(), vec![0, 1, 3, 4]);
        // Each pick leaves the offer, so idx 0 twice names two cards.
        for idx in [3, 0, 0] {
            assert_eq!(
                rm.take_action(RunAction::EnchantDeckCard { idx }),
                ActionResult::Advanced
            );
        }
        assert!(rm.run_state.pending_deck_enchants.is_empty());
        assert_eq!(stacks(&rm, EnchantmentId::Instinct), [1, 1, 0, 0, 1]);
    }

    #[test]
    fn tri_boomerang_opens_no_screen_for_three_or_fewer_attacks() {
        use CardId::{DefendIronclad as Defend, StrikeIronclad as Strike};
        let rm = manager_for(RelicId::TriBoomerang, &[Strike, Defend, Strike]);
        assert!(rm.run_state.pending_deck_enchants.is_empty());
        assert_eq!(stacks(&rm, EnchantmentId::Instinct), [1, 0, 1]);
    }

    #[test]
    fn paels_growth_clones_the_card_the_player_picks() {
        use CardId::StrikeIronclad as Strike;
        let mut rm = manager_for(RelicId::PaelsGrowth, &[Strike, Strike]);
        assert_eq!(rm.run_state.pending_deck_enchants.len(), 1);
        assert_eq!(
            rm.take_action(RunAction::EnchantDeckCard { idx: 1 }),
            ActionResult::Advanced
        );
        assert_eq!(stacks(&rm, EnchantmentId::Clone), [0, 4]);

        let single = manager_for(RelicId::PaelsGrowth, &[Strike]);
        assert!(single.run_state.pending_deck_enchants.is_empty());
        assert_eq!(stacks(&single, EnchantmentId::Clone), [4]);
    }
}

/// GH #1353: Claws' "up to 6 into Maul" select is a player choice.
#[cfg(test)]
mod claws_selection_tests {
    use super::*;
    use sts2_core::cards::CardId;
    use sts2_core::potions::CharacterId;
    use sts2_core::run_manager::{ActionResult, RunAction, RunManager};

    /// An agent-mode run whose deck is exactly `cards`, after obtaining Claws.
    fn claws_run(cards: &[CardId]) -> RunManager {
        crate::cards::dispatch::ensure_installed();
        crate::relics::dispatch::ensure_installed();
        let mut rm = RunManager::new(0, CharacterId::Ironclad);
        rm.run_state.set_agent_decisions();
        rm.run_state.master_deck.clear();
        for &id in cards {
            push_deck_card(&mut rm.run_state, id);
        }
        rm.run_state.obtain_relic(RelicId::Claws);
        rm
    }

    fn mauls(rm: &RunManager) -> usize {
        rm.run_state
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::Maul)
            .count()
    }

    const EIGHT: [CardId; 8] = [
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
    ];

    #[test]
    fn the_select_offers_every_candidate_and_the_confirm() {
        let rm = claws_run(&EIGHT);
        let actions = rm.get_available_actions();
        assert_eq!(actions.len(), 9);
        assert_eq!(actions.last(), Some(&RunAction::ConfirmDeckSelection));
        assert!(actions[..8]
            .iter()
            .all(|a| matches!(a, RunAction::TransformDeckCard { .. })));
    }

    #[test]
    fn zero_one_or_six_picks_transform_exactly_those_on_confirm() {
        for picks in [0, 1, 6] {
            let mut rm = claws_run(&EIGHT);
            let original = rm.run_state.master_deck.clone();
            for _ in 0..picks {
                assert_eq!(
                    rm.take_action(RunAction::TransformDeckCard { idx: 0 }),
                    ActionResult::Advanced
                );
                // The deck changes only on the confirm (one C# Transform).
                assert_eq!(rm.run_state.master_deck, original);
            }
            assert_eq!(
                rm.take_action(RunAction::ConfirmDeckSelection),
                ActionResult::Advanced
            );
            assert!(rm.run_state.pending_claws_selection.is_none());
            assert_eq!(mauls(&rm), picks);
            assert_eq!(rm.run_state.master_deck.len(), original.len());
        }
    }

    #[test]
    fn the_cap_leaves_only_the_confirm() {
        let mut rm = claws_run(&EIGHT);
        for _ in 0..CLAWS_CARDS {
            assert_eq!(
                rm.take_action(RunAction::TransformDeckCard { idx: 0 }),
                ActionResult::Advanced
            );
        }
        assert_eq!(
            rm.get_available_actions(),
            vec![RunAction::ConfirmDeckSelection]
        );
        assert_eq!(
            rm.take_action(RunAction::TransformDeckCard { idx: 0 }),
            ActionResult::InvalidForPhase
        );
    }

    #[test]
    fn a_later_card_can_be_picked_and_mauls_join_in_deck_order() {
        let mut rm = claws_run(&[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash]);
        // Pick the Bash (offer idx 2), then the Defend (offer idx 1).
        for idx in [2, 1] {
            assert_eq!(
                rm.take_action(RunAction::TransformDeckCard { idx }),
                ActionResult::Advanced
            );
        }
        assert_eq!(rm.transform_eligible_deck_indices(), vec![0]);
        assert_eq!(
            rm.take_action(RunAction::ConfirmDeckSelection),
            ActionResult::Advanced
        );
        let deck = &rm.run_state.master_deck;
        assert_eq!(deck[0].card_id, CardId::StrikeIronclad);
        assert_eq!(deck[1].card_id, CardId::Maul);
        assert_eq!(deck[2].card_id, CardId::Maul);
        assert_eq!(deck[1].target_type, sts2_core::cards::TargetType::Enemy);
    }

    #[test]
    fn the_select_preempts_the_phase_until_confirmed() {
        let mut rm = claws_run(&EIGHT);
        assert!(rm.run_state.has_pending_grant_choice());
        rm.take_action(RunAction::ConfirmDeckSelection);
        assert!(!rm
            .get_available_actions()
            .iter()
            .any(|a| matches!(a, RunAction::ConfirmDeckSelection)));
    }

    #[test]
    fn non_interactive_runs_keep_the_first_six() {
        crate::relics::dispatch::ensure_installed();
        let mut rm = RunManager::new(0, CharacterId::Ironclad);
        rm.run_state.master_deck.clear();
        for &id in &EIGHT {
            push_deck_card(&mut rm.run_state, id);
        }
        rm.run_state.obtain_relic(RelicId::Claws);
        assert!(rm.run_state.pending_claws_selection.is_none());
        assert_eq!(mauls(&rm), CLAWS_CARDS);
    }
}
