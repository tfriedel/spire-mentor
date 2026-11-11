//! Event-reward relic dispatch — bead `sts2-rs-7ez` (P7-5b).
//!
//! Ports the 35 `RelicRarity.EVENT` entries from
//! `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py`. The sibling
//! bead `sts2-rs-yyw` (P7-5) owns the complementary shop+ancient
//! subset of the same Python file.
//!
//! Scope decision (recorded in `docs/known_divergences.md`): this bead
//! handles `RelicRarity.EVENT` only. The ~92 `RelicRarity.ANCIENT`
//! entries — including `CallingBell` / `BlackStar` / `PandorasBox`
//! and the named examples in the bead description — carry
//! `pool = RelicPool.EVENT` in Python but live under the `ANCIENT`
//! rarity bucket and are owned by `sts2-rs-yyw`. Splitting on
//! `rarity` (not `pool`) avoids any overlap with the sibling bead and
//! keeps every relic file rooted in a single rarity bucket.
//!
//! Handler bodies are intentionally minimal markers
//! ([`EffectResult::Resolved`]) per the bead description: most
//! event-reward relics depend on event-spawn / `after_obtained` /
//! `modify_rewards` / pet-system hooks. The dispatch acceptance
//! criterion is "every variant resolves through the registry instead
//! of falling through to `NotImplemented`".
//!
//! Surface beads `sts2-rs-2ax` / `tcf` / `anx` / `62t` / `txd` / `3f8`
//! (the six n3w sub-beads) landed the listener-bus infrastructure on
//! [`sts2_core::relic_effects`]. Beads `sts2-rs-j87` + `zq8` + `ouv`
//! + `6jk` (the j87 trio) wired every relevant dispatch channel into
//! the in-combat call sites. The umbrella retrofit `sts2-rs-01o`
//! registers per-relic listener bodies via [`install_listeners`].
//!
//! ### Wave 1 (`sts2-rs-01o` Easy-tier subset, this commit)
//!
//! Four listeners landed against in-combat-only channels:
//!
//! - **`FakeAnchor`** — `before_combat_start` → +4 block.
//! - **`FakeOrichalcum`** — `before_turn_end` → +3 block if block==0.
//! - **`DaughterOfTheWind`** — `after_card_played` → +1 block per Attack.
//! - **`TheBoot`** — `modify_hp_lost` floor to 5 on powered attacks 1–4.
//!
//! Every listener is gated by `state.relics.owns(…)` (a non-melted relic of
//! that kind) so the registered handler is a no-op for runs that don't carry the
//! relic. The follow-up bead `sts2-rs-46ms` (wave 2) lands six more
//! listeners against the P7-surf prereqs.
//!
//! ### Wave 2 (`sts2-rs-46ms` Easy/Medium-tier subset, 2026-05-13)
//!
//! Six additional listeners landed against the surfaces lifted by
//! P7-surf prereqs `vt7n` (`apply_power_to_creature`) + `xqvn`
//! (`RunState` HP/gold + `after_max_hp_changed`) + `4nh5`
//! (`ModifyDamageCtx.card`):
//!
//! - **`FakeMango`** / **`FakeLeesWaffle`** / **`BigMushroom`** —
//!   `run_effects::after_obtained` → `rs.gain_max_hp` / `rs.heal`.
//! - **`SwordOfJade`** / **`FakeSneckoEye`** —
//!   `before_combat_start` → `apply_power_to_creature(player, …)`.
//! - **`FakeStrikeDummy`** — `modify_damage_additive` → +1 when
//!   `ctx.card` has `CardTag::Strike`.
//!
//! ### Wave 4 (`sts2-rs-v7mp`, 2026-05-13)
//!
//! Five additional listeners landed against the combat-side helper
//! surfaces lifted by this bead (`CombatState::creature_heal` /
//! `get_random_enemy` / `round_no` + `card_effects::deal_damage_to_all_alive_enemies`):
//!
//! - **`FakeBloodVial`** — `after_side_turn_start` (round-1, player) →
//!   `creature_heal(player_uid, 1)`.
//! - **`RoyalPoison`** — `after_player_turn_start` (round-1) →
//!   `apply_damage_to_player(4, UNBLOCKABLE | UNPOWERED)`.
//! - **`MrStruggles`** — `after_side_turn_start` (player) →
//!   `deal_damage_to_all_alive_enemies(round_no, UNPOWERED)`.
//! - **`LostWisp`** — `after_card_played` (`card_type == Power`) →
//!   `deal_damage_to_all_alive_enemies(8, UNPOWERED)`.
//! - **`ForgottenSoul`** — `after_card_exhausted` →
//!   `apply_damage(get_random_enemy(), 1, UNPOWERED)`.
//!
//! ### Wave 5 (`sts2-rs-jvk3`, 2026-05-14)
//!
//! Two additional listeners landed against the P7-surf prereqs
//! lifted by this bead (`CombatState::is_elite` + `RunState::{
//! lose_hp, take_damage, transform_relic, upgrade_random_cards}`):
//!
//! - **`SwordOfStone`** — `after_combat_victory` → when `is_elite`,
//!   decrement `Relic.counter` (init 5). The `transform_relic` half
//!   lives on the run-side `after_combat_victory_run` channel, fired by
//!   `RunManager::exit_combat_and_transition`.
//! - **`FragrantMushroom`** — `after_obtained` (`RunState`) →
//!   `rs.take_damage(15) + rs.upgrade_random_cards(3)`. Both
//!   clauses are real (bead `sts2-rs-7krx0` wired
//!   `upgrade_random_cards` to `master_deck` + `rewards_rng`);
//!   the upgrade saturates at the eligible-card count when
//!   `master_deck` has fewer than 3 non-upgraded cards.
//!
//! The remaining 18 relics still resolve through the dispatch-table
//! marker handler; each carries an explicit `// no-op: <reason>`
//! comment naming the blocker. See
//! `docs/reality-checks/2026-05-12-event-reward-audit.md` footer for
//! the per-relic plan.
//!
//! The per-relic doc-comments below mirror the **Python pin**
//! behaviour at `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py`
//! and the per-relic classification in
//! `docs/reality-checks/2026-05-12-event-reward-audit.md`. Stale
//! "Fake-X means inverted clause" framing pulled from the
//! sts2-rl-agent docstrings was wrong for ~18 relics — Python's
//! `Fake*` classes are mostly plain (smaller) versions of their real
//! counterparts, not punitive inversions. Audit landed 2026-05-13.
//!
//! ### Per-relic state notes
//!
//! Several event-reward relics carry per-instance counters
//! (`BingBong._skip_next`, `BoneTea._combats_left`,
//! `HandDrill._dmg_buff_used`,
//! `HistoryCourse._uses_left`, `MawBank._spent`,
//! `MrStruggles._purchases_left`, `WongosMysteryTicket._used`).
//! Their per-instance counters will live on the runtime `Relic` POD
//! (`sts2_core::Relic.counter`, not yet wired) — modelled in this
//! bead as comments on each handler so the wiring bead has a
//! one-shot reference.

use sts2_core::card_effects::{
    apply_power_to_creature, deal_damage_to_all_alive_enemies, upgrade_card_uid, EffectResult,
};
use sts2_core::cards::{Card, EnchantmentId};
use sts2_core::hooks::{self, BlockCtx, BlockGainedCtx, CreatureRef, ValueProps};
use sts2_core::powers::PowerKind;
use sts2_core::relic_effects::{
    register_after_card_played, register_after_damage_given, register_after_player_turn_start,
    register_after_side_turn_start, register_before_combat_start, register_before_side_turn_start,
    register_before_turn_end, register_modify_damage_additive, register_modify_hp_lost,
    register_on_card_added_to_deck, register_owned_after_card_exhausted,
    register_owned_after_combat_victory, register_owned_after_energy_reset,
    register_owned_before_hand_draw, register_owned_before_play_phase_start,
    register_owned_modify_hand_draw, AfterCardExhaustedFn, AfterCardPlayedFn, AfterCombatVictoryFn,
    AfterDamageGivenFn, AfterEnergyResetFn, AfterPlayerTurnStartFn, AfterSideTurnStartFn,
    BeforeCombatStartFn, BeforeHandDrawFn, BeforePlayPhaseStartFn, BeforeSideTurnStartFn,
    BeforeTurnEndFn, CardAddedCtx, CardEventCtx, DamageGivenCtx, ModifyDamageAdditiveFn,
    ModifyDamageCtx, ModifyHandDrawFn, ModifyHpLostCtx, ModifyHpLostFn, OnCardAddedToDeckFn,
};
use sts2_core::relics::RelicId;
use sts2_core::run_effects::{
    register_after_combat_end, register_after_combat_victory_run, register_after_obtained,
    register_after_room_entered, register_modify_card_being_added_to_deck,
    register_modify_card_reward_options_late, register_modify_merchant_card_creation_results,
    register_modify_rest_site_heal_rewards, register_modify_rewards,
    register_on_card_added_to_deck as register_on_card_added_to_deck_run,
    register_on_item_purchased, AfterCombatEndRunFn, AfterCombatVictoryRunFn, AfterObtainedRunFn,
    AfterRoomEnteredRunFn, CardAddedRunCtx, CombatVictoryRunCtx, ModifyCardBeingAddedToDeckRunFn,
    ModifyCardRewardOptionsLateRunFn, ModifyMerchantCardCreationResultsRunFn,
    ModifyRestSiteHealRewardsRunFn, ModifyRewardsRunFn, OnCardAddedToDeckRunFn,
    OnItemPurchasedRunFn,
};
use sts2_core::runstate::{CardReward, RelicReward, Reward, RoomType, RunState};
use sts2_core::{
    cards::{CardTag, CardType},
    CombatState, RelicKind, Side,
};

use crate::relics::{RelicEffect, RelicRarity, RelicSpec};

// ---------------------------------------------------------------------------
// Per-relic factories — 35 event-reward relics
// ---------------------------------------------------------------------------

macro_rules! relic_factory {
    ($vis:vis $fn_name:ident, $kind:ident, $name:literal) => {
        #[must_use]
        $vis const fn $fn_name() -> RelicSpec {
            RelicSpec {
                id: RelicKind::$kind,
                name: $name,
                rarity: RelicRarity::Event,
                effect: RelicEffect::Noop,
            }
        }
    };
}

relic_factory!(pub big_mushroom, BigMushroom, "BIG_MUSHROOM");
relic_factory!(pub bing_bong, BingBong, "BING_BONG");
relic_factory!(pub bone_tea, BoneTea, "BONE_TEA");
relic_factory!(pub byrdpip, Byrdpip, "BYRDPIP");
relic_factory!(pub chosen_cheese, ChosenCheese, "CHOSEN_CHEESE");
relic_factory!(pub darkstone_periapt, DarkstonePeriapt, "DARKSTONE_PERIAPT");
relic_factory!(pub daughter_of_the_wind, DaughterOfTheWind, "DAUGHTER_OF_THE_WIND");
relic_factory!(pub dream_catcher, DreamCatcher, "DREAM_CATCHER");
relic_factory!(pub ember_tea, EmberTea, "EMBER_TEA");
relic_factory!(pub fake_anchor, FakeAnchor, "FAKE_ANCHOR");
relic_factory!(pub fake_blood_vial, FakeBloodVial, "FAKE_BLOOD_VIAL");
relic_factory!(pub fake_happy_flower, FakeHappyFlower, "FAKE_HAPPY_FLOWER");
relic_factory!(pub fake_lees_waffle, FakeLeesWaffle, "FAKE_LEES_WAFFLE");
relic_factory!(pub fake_mango, FakeMango, "FAKE_MANGO");
relic_factory!(pub fake_merchants_rug, FakeMerchantsRug, "FAKE_MERCHANTS_RUG");
relic_factory!(pub fake_orichalcum, FakeOrichalcum, "FAKE_ORICHALCUM");
relic_factory!(pub fake_snecko_eye, FakeSneckoEye, "FAKE_SNECKO_EYE");
relic_factory!(pub fake_strike_dummy, FakeStrikeDummy, "FAKE_STRIKE_DUMMY");
relic_factory!(pub fake_venerable_tea_set, FakeVenerableTeaSet, "FAKE_VENERABLE_TEA_SET");
relic_factory!(pub forgotten_soul, ForgottenSoul, "FORGOTTEN_SOUL");
relic_factory!(pub fragrant_mushroom, FragrantMushroom, "FRAGRANT_MUSHROOM");
relic_factory!(pub fresnel_lens, FresnelLens, "FRESNEL_LENS");
relic_factory!(pub hand_drill, HandDrill, "HAND_DRILL");
relic_factory!(pub history_course, HistoryCourse, "HISTORY_COURSE");
relic_factory!(pub lost_wisp, LostWisp, "LOST_WISP");
relic_factory!(pub maw_bank, MawBank, "MAW_BANK");
relic_factory!(pub mr_struggles, MrStruggles, "MR_STRUGGLES");
relic_factory!(pub pollinous_core, PollinousCore, "POLLINOUS_CORE");
relic_factory!(pub royal_poison, RoyalPoison, "ROYAL_POISON");
relic_factory!(pub sword_of_stone, SwordOfStone, "SWORD_OF_STONE");
relic_factory!(pub sword_of_jade, SwordOfJade, "SWORD_OF_JADE");
relic_factory!(pub tea_of_discourtesy, TeaOfDiscourtesy, "TEA_OF_DISCOURTESY");
relic_factory!(pub the_boot, TheBoot, "THE_BOOT");
relic_factory!(pub wongo_customer_appreciation_badge, WongoCustomerAppreciationBadge, "WONGO_CUSTOMER_APPRECIATION_BADGE");
relic_factory!(pub wongos_mystery_ticket, WongosMysteryTicket, "WONGOS_MYSTERY_TICKET");

// ---------------------------------------------------------------------------
// Pool installer
// ---------------------------------------------------------------------------

/// All 35 `RelicRarity.EVENT` relic ids from `shop_event.py`. The order
/// mirrors the Python `@register_relic class …` declaration order. This
/// is membership/identity data only — live behaviour flows through the
/// typed hook-channel listeners registered by [`install_listeners`], not
/// through any per-relic dispatch marker.
#[cfg(test)]
const EVENT_REWARD_RELIC_IDS: &[RelicId] = &[
    RelicId::BigMushroom,
    RelicId::BingBong,
    RelicId::BoneTea,
    RelicId::Byrdpip,
    RelicId::ChosenCheese,
    RelicId::DarkstonePeriapt,
    RelicId::DaughterOfTheWind,
    RelicId::DreamCatcher,
    RelicId::EmberTea,
    RelicId::FakeAnchor,
    RelicId::FakeBloodVial,
    RelicId::FakeHappyFlower,
    RelicId::FakeLeesWaffle,
    RelicId::FakeMango,
    RelicId::FakeMerchantsRug,
    RelicId::FakeOrichalcum,
    RelicId::FakeSneckoEye,
    RelicId::FakeStrikeDummy,
    RelicId::FakeVenerableTeaSet,
    RelicId::ForgottenSoul,
    RelicId::FragrantMushroom,
    RelicId::FresnelLens,
    RelicId::HandDrill,
    RelicId::HistoryCourse,
    RelicId::LostWisp,
    RelicId::MawBank,
    RelicId::MrStruggles,
    RelicId::PollinousCore,
    RelicId::RoyalPoison,
    RelicId::SwordOfStone,
    RelicId::SwordOfJade,
    RelicId::TeaOfDiscourtesy,
    RelicId::TheBoot,
    RelicId::WongoCustomerAppreciationBadge,
    RelicId::WongosMysteryTicket,
];

/// Install every event-reward handler. Called from
/// [`crate::relics::dispatch::install_default_registry`]. Idempotent
/// (re-registering overwrites the same function pointer).
pub fn install_pool() {
    install_listeners();
}

/// Register the per-relic hook-channel listeners for the subset of
/// event-reward relics whose bodies are landed today (`sts2-rs-01o`
/// Easy-tier wave 1). The remaining 31 relics still resolve through
/// the dispatch-table marker handler and are deferred to follow-up
/// beads — see the audit doc for the per-relic blocker.
fn install_listeners() {
    // Wave 1 (sts2-rs-01o) — landed 2026-05-13.
    register_before_combat_start(
        RelicId::FakeAnchor,
        listener_fake_anchor as BeforeCombatStartFn,
    );
    register_before_turn_end(
        RelicId::FakeOrichalcum,
        listener_fake_orichalcum as BeforeTurnEndFn,
    );
    register_after_card_played(
        RelicId::DaughterOfTheWind,
        listener_daughter_of_the_wind as AfterCardPlayedFn,
    );
    register_modify_hp_lost(RelicId::TheBoot, listener_the_boot as ModifyHpLostFn);

    // Wave 2 (sts2-rs-46ms) — landed 2026-05-13.
    //
    // RunState-side after_obtained leg (uses xqvn's RunState HP surface).
    register_after_obtained(
        RelicId::FakeMango,
        listener_fake_mango as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::FakeLeesWaffle,
        listener_fake_lees_waffle as AfterObtainedRunFn,
    );
    register_after_obtained(
        RelicId::BigMushroom,
        listener_big_mushroom as AfterObtainedRunFn,
    );

    // Combat-side before_combat_start power-apply leg (uses vt7n's
    // creature-keyed apply_power_to_creature).
    register_before_combat_start(
        RelicId::SwordOfJade,
        listener_sword_of_jade as BeforeCombatStartFn,
    );
    register_before_combat_start(
        RelicId::FakeSneckoEye,
        listener_fake_snecko_eye as BeforeCombatStartFn,
    );

    // Wave 3 (sts2-rs-3nis) — per-instance counter POD now lives on
    // `Relic.counter`. EmberTea uses it as `_combats_left` (init 5,
    // decrements each combat start until 0).
    register_before_combat_start(RelicId::EmberTea, listener_ember_tea as BeforeCombatStartFn);

    // BoneTea (sts2-rs-eejzj) — on the owner's round-1 side-turn-start,
    // upgrade every card in hand, then decrement the per-instance
    // `_combats_left` counter (init 1). C# `BoneTea.cs::AfterSideTurnStart`
    // gates on `!IsUsedUp && side == Owner.Side && RoundNumber <= 1`.
    register_after_side_turn_start(RelicId::BoneTea, listener_bone_tea as AfterSideTurnStartFn);
    // v0.111.0: BoneTea moved to `AfterPlayerTurnStart` (decomp
    // BoneTea.cs:50), which fires inside `SetupPlayerTurn` before the
    // outer `AfterSideTurnStart` walk (so e.g. FencingManual's forged blade
    // is never upgraded). Each listener is inert in the other build.
    register_after_player_turn_start(
        RelicId::BoneTea,
        listener_bone_tea_player_turn_start as AfterPlayerTurnStartFn,
    );

    // FakeVenerableTeaSet (sts2-rs-aqals) — rest-site latch
    // (`after_room_entered`, RunState) + first-energy-reset consume
    // (`after_energy_reset`, combat). C# `FakeVenerableTeaSet.cs`:
    // `AfterRoomEntered(RestSiteRoom)` sets `GainEnergyInNextCombat`;
    // `AfterEnergyReset` gains `Energy = 1` once and clears the flag.
    register_after_room_entered(
        RelicId::FakeVenerableTeaSet,
        listener_fake_venerable_tea_set_rest_latch as AfterRoomEnteredRunFn,
    );
    register_owned_after_energy_reset(
        RelicId::FakeVenerableTeaSet,
        RelicKind::FakeVenerableTeaSet,
        listener_fake_venerable_tea_set_consume as AfterEnergyResetFn,
    );

    // Byrdpip (sts2-rs-4x616) — `AddsPet`/`SpawnsPets` event relic. C#
    // `Byrdpip.cs`: `BeforeCombatStart` summons the (behaviourless,
    // NOTHING_MOVE) Byrdpip pet; `AfterObtained` transforms every
    // `ByrdonisEgg` in the deck into `ByrdSwoop` (and re-summons the pet
    // if obtained mid-combat). The pet itself has no combat behaviour, so
    // the player-facing mechanic is the egg→swoop transform.
    register_before_combat_start(
        RelicId::Byrdpip,
        listener_byrdpip_summon_pet as BeforeCombatStartFn,
    );
    register_after_obtained(
        RelicId::Byrdpip,
        listener_byrdpip_transform_eggs as AfterObtainedRunFn,
    );

    // HistoryCourse (sts2-rs-ummci) — at the start of each play phase
    // after round 1, auto-play a free dupe of the last Attack/Skill the
    // player played the previous round. C# `HistoryCourse.cs::
    // BeforePlayPhaseStart` scans `History.CardPlaysFinished` for the last
    // non-dupe Attack/Skill with `RoundNumber == round - 1` and
    // `CardCmd.AutoPlay`s a dupe.
    register_owned_before_play_phase_start(
        RelicId::HistoryCourse,
        RelicKind::HistoryCourse,
        listener_history_course as BeforePlayPhaseStartFn,
    );

    // modify_damage_additive leg (uses 4nh5's ModifyDamageCtx.card field).
    register_modify_damage_additive(
        RelicId::FakeStrikeDummy,
        listener_fake_strike_dummy as ModifyDamageAdditiveFn,
    );

    // Wave 4 (sts2-rs-v7mp) — combat-side helpers landed alongside.
    //
    // MrStruggles fires on `after_side_turn_start` (player side);
    // LostWisp fires on `after_card_played` (Power gate); ForgottenSoul
    // fires on `after_card_exhausted`. FakeBloodVial is C#
    // `AfterPlayerTurnStartLate` (`FakeBloodVial.cs:19`): no registration
    // here, the turn-start tail calls
    // `sts2_core::relic_effects::blood_vial::dispatch_after_player_turn_start_late`.
    // C# `RoyalPoison.AfterPlayerTurnStart` (RoyalPoison.cs:18, both builds):
    // inside `SetupPlayerTurn`, before every `AfterPlayerTurnStartLate`
    // (BloodVial) and before the side-hook walk.
    register_after_player_turn_start(
        RelicId::RoyalPoison,
        listener_royal_poison as AfterPlayerTurnStartFn,
    );
    // C# `MrStruggles.AfterPlayerTurnStart` (MrStruggles.cs:15-22) — fires
    // BEFORE the side-hook walk where an enemy LivingShield's RampartPower
    // arms its teammate's block, so the round-number damage lands on bare
    // HP (sweep seed 5321; same relocation class as MercuryHourglass /
    // FestivePopper, commit 67b211cc).
    register_after_player_turn_start(
        RelicId::MrStruggles,
        listener_mr_struggles as AfterPlayerTurnStartFn,
    );
    register_after_card_played(RelicId::LostWisp, listener_lost_wisp as AfterCardPlayedFn);
    register_owned_after_card_exhausted(
        RelicId::ForgottenSoul,
        RelicKind::ForgottenSoul,
        listener_forgotten_soul as AfterCardExhaustedFn,
    );

    // Wave 5 (sts2-rs-acyi) — RunState-arg back-edge channels landed
    // alongside `run_effects::{after_combat_end, on_card_added_to_deck}`.
    // No production firing site yet (a future combat→run back-edge
    // bead will wire these), but the listeners are registered eagerly
    // so the flip activates them without a re-registration pass.
    register_after_combat_end(
        RelicId::ChosenCheese,
        listener_chosen_cheese as AfterCombatEndRunFn,
    );
    register_on_card_added_to_deck_run(
        RelicId::DarkstonePeriapt,
        listener_darkstone_periapt as OnCardAddedToDeckRunFn,
    );

    // Bead `sts2-rs-hyv0p` — BingBong: clone-and-re-add the card just
    // appended to the master deck (combat-side firing site live via
    // CombatState::add_to_master_deck; re-entry guarded by
    // AddSource::BingBongDuplicate on the dispatch ctx).
    register_on_card_added_to_deck(RelicId::BingBong, listener_bing_bong as OnCardAddedToDeckFn);
    // The run-side half: a deck add between fights (event grant, relic
    // pickup, card reward) clones the card too (GH #806).
    register_on_card_added_to_deck_run(
        RelicId::BingBong,
        listener_bing_bong_run as OnCardAddedToDeckRunFn,
    );

    // Wave 6 (sts2-rs-9wx7) — reward-flow hooks for DreamCatcher /
    // FresnelLens / WongosMysteryTicket. All five channels live in
    // `run_effects` (RunState-arg); no production firing site lands
    // with this bead, but the listeners are registered eagerly so the
    // future port of the run-loop reward fan-out activates them
    // without a re-registration pass.
    register_modify_rest_site_heal_rewards(
        RelicId::DreamCatcher,
        listener_dream_catcher as ModifyRestSiteHealRewardsRunFn,
    );
    register_modify_card_reward_options_late(
        RelicId::FresnelLens,
        listener_fresnel_lens_card_reward as ModifyCardRewardOptionsLateRunFn,
    );
    register_modify_card_being_added_to_deck(
        RelicId::FresnelLens,
        listener_fresnel_lens_deck_add as ModifyCardBeingAddedToDeckRunFn,
    );
    register_modify_merchant_card_creation_results(
        RelicId::FresnelLens,
        listener_fresnel_lens_merchant as ModifyMerchantCardCreationResultsRunFn,
    );
    register_modify_rewards(
        RelicId::WongosMysteryTicket,
        listener_wongos_mystery_ticket_rewards as ModifyRewardsRunFn,
    );
    register_after_combat_end(
        RelicId::WongosMysteryTicket,
        listener_wongos_mystery_ticket_combat_end as AfterCombatEndRunFn,
    );

    // sts2-rs-a6j4v — LavaRock: after Act 1 boss, push 2 RelicReward
    // sentinels onto the rewards list (once per run).
    register_modify_rewards(
        RelicId::LavaRock,
        listener_lava_rock_rewards as ModifyRewardsRunFn,
    );

    // sts2-rs-yb6rm — MawBank: +12g on every room enter until the
    // first non-zero-gold shop purchase latches the relic. Latch lives
    // on `relic_counters[MawBank]`; firing site for the latch is the
    // `on_item_purchased` channel that fires from `shop::buy_*`
    // (landed in the same bead).
    register_after_room_entered(
        RelicId::MawBank,
        listener_maw_bank_after_room_entered as AfterRoomEnteredRunFn,
    );
    register_on_item_purchased(
        RelicId::MawBank,
        listener_maw_bank_on_item_purchased as OnItemPurchasedRunFn,
    );

    // sts2-rs-67co3 — LavaLamp: when no damage was taken in the
    // just-ended combat, upgrade every card on the reward screen.
    // Firing-site for the per-combat flag is bead `sts2-rs-o62e7`
    // (combat→run flushback); listener registered eagerly so when
    // the firing site lands the upgrade fires without
    // re-registration.
    register_modify_card_reward_options_late(
        RelicId::LavaLamp,
        listener_lava_lamp_card_reward as ModifyCardRewardOptionsLateRunFn,
    );
    // C# `LavaLamp.AfterRoomEntered` clears `TookDamageThisCombat` on
    // every room entry (both builds), so the flag is per-combat.
    register_after_room_entered(
        RelicId::LavaLamp,
        listener_lava_lamp_after_room_entered as AfterRoomEnteredRunFn,
    );

    // sts2-rs-v9zer — PaelsTooth: on pickup, stash up to 5
    // upgradable+removable master-deck cards into
    // `rs.paels_tooth_stored`; each combat-end, draw one via the
    // rewards stream, upgrade it, and push it back onto the master
    // deck. The interactive deck-choice variant is deferred to a
    // follow-up sub-bead — this listener implements Python's auto-
    // pick fallback branch (`shop_event.py:2810`).
    register_after_obtained(
        RelicId::PaelsTooth,
        listener_paels_tooth_after_obtained as AfterObtainedRunFn,
    );
    register_after_combat_end(
        RelicId::PaelsTooth,
        listener_paels_tooth_after_combat_end as AfterCombatEndRunFn,
    );

    // Wave 5 (sts2-rs-jvk3) — RunState helpers + combat.is_elite.
    //
    // SwordOfStone fires on `after_combat_victory` (CombatState-arg);
    // when `is_elite`, decrement its `Relic.counter` (init 5). The
    // transform-into-SwordOfJade arm at counter==0 lives on the
    // parallel run-side channel `after_combat_victory_run`
    // (bead `sts2-rs-y8zi4`) — both listeners read the same elite-
    // victory signal but only the run-side path can mutate
    // `RunState.relics` to actually perform the transform_relic
    // swap. The run channel fires from
    // `RunManager::exit_combat_and_transition`, after the exit
    // callback has flushed the combat-side count.
    //
    // FragrantMushroom fires on `after_obtained` (RunState-arg);
    // `rs.take_damage(15)` lands fully, and the deck random-upgrade
    // clause routes through `rs.upgrade_random_cards` (real impl
    // since bead `sts2-rs-7krx0` — partial Fisher-Yates on
    // master_deck, gated only by deck size).
    register_owned_after_combat_victory(
        RelicId::SwordOfStone,
        RelicKind::SwordOfStone,
        listener_sword_of_stone as AfterCombatVictoryFn,
    );
    register_after_combat_victory_run(
        RelicId::SwordOfStone,
        listener_sword_of_stone_run as AfterCombatVictoryRunFn,
    );

    // Bead `sts2-rs-mg5ur` — WarHammer: on elite victory, upgrade
    // 4 random upgradable deck cards. Shares the run-side firing site
    // with SwordOfStone.
    register_after_combat_victory_run(
        RelicId::WarHammer,
        listener_war_hammer_run as AfterCombatVictoryRunFn,
    );

    register_after_obtained(
        RelicId::FragrantMushroom,
        listener_fragrant_mushroom as AfterObtainedRunFn,
    );

    // Wave 5 (sts2-rs-c7ak) — combat-side accessors lifted by this
    // bead (`gain_energy`, `add_card_to_draw_pile`, `active_attack`).
    //
    // FakeHappyFlower fires on `after_side_turn_start` (player side,
    // every 5 player turns, gates on `Relic.counter % 5 == 0`).
    // TeaOfDiscourtesy fires on `before_combat_start` (gates on
    // `Relic.counter > 0`, decrements once per combat). HandDrill
    // fires on `after_damage_given` (gates on
    // `state.active_attack().is_some() && ctx.blocked > 0 && target
    // post-block == 0`).
    register_after_side_turn_start(
        RelicId::FakeHappyFlower,
        listener_fake_happy_flower as AfterSideTurnStartFn,
    );
    register_before_combat_start(
        RelicId::TeaOfDiscourtesy,
        listener_tea_of_discourtesy as BeforeCombatStartFn,
    );
    register_after_damage_given(
        RelicId::HandDrill,
        listener_hand_drill as AfterDamageGivenFn,
    );
    // PollinousCore — +2 hand-draw every `POLLINOUS_CORE_TURNS = 4`
    // player turns. `TurnsSeen` lives on the relic's `counter` (i32). In
    // v0.107.1 the `before_side_turn_start` listener ticks it on player-side
    // starts, the `modify_hand_draw` listener returns `+EXTRA` when
    // `counter == TURNS`, and the turn setup resets it after the hand-draw
    // count (`AfterModifyingHandDraw`).
    register_before_side_turn_start(
        RelicId::PollinousCore,
        listener_pollinous_core_tick as BeforeSideTurnStartFn,
    );
    register_owned_modify_hand_draw(
        RelicId::PollinousCore,
        RelicKind::PollinousCore,
        listener_pollinous_core_modify_hand_draw as ModifyHandDrawFn,
    );
    // v0.111.0: the count moved to `BeforeHandDraw` (decomp
    // PollinousCore.cs:88). Inert in v0.107.1, as the side-turn tick above
    // is in v0.111.0.
    register_owned_before_hand_draw(
        RelicId::PollinousCore,
        RelicKind::PollinousCore,
        listener_pollinous_core_before_hand_draw as BeforeHandDrawFn,
    );
}

// ---------------------------------------------------------------------------
// Listener bodies — Easy-tier wave 1 (sts2-rs-01o)
// ---------------------------------------------------------------------------

fn player_owns(state: &CombatState, kind: RelicKind) -> bool {
    state.relics.owns(kind)
}

/// Local analogue of `card_effects::gain_block_player` but with
/// `ValueProps::MOVE | ValueProps::UNPOWERED`, so the Frail /
/// Dexterity multiplicative-block fold skips. C# `CreatureCmd.GainBlock`
/// pipes the `BlockVar`'s embedded `ValueProp.Unpowered` through when
/// the relic declares it (e.g. `FakeAnchor`, `FakeOrichalcum`,
/// `DaughterOfTheWind`, `Permafrost` — all `BlockVar(*, Unpowered)`).
/// Bead `sts2-rs-jo6sc`.
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

/// `FakeAnchor::before_combat_start` — C# `FakeAnchor.cs` v0.103.2:
/// `BlockVar(4m, ValueProp.Unpowered)`; `GainBlock(Owner.Creature,
/// DynamicVars.Block, null)` pipes the Unpowered prop. Frail/Dexterity
/// skip the grant. Bead `sts2-rs-jo6sc` (replaces the prior
/// `gain_block_player(4)` which folded Frail).
fn listener_fake_anchor(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::FakeAnchor) {
        return EffectResult::NotImplemented;
    }
    gain_block_player_unpowered(state, 4);
    EffectResult::Resolved
}

/// `FakeOrichalcum::before_turn_end` — C# `FakeOrichalcum.cs` v0.103.2:
/// `BlockVar(3m, ValueProp.Unpowered)`; on player end-of-turn if
/// `owner.block == 0`, GainBlock(3) Unpowered. Bead `sts2-rs-jo6sc`.
///
/// C# `FakeOrichalcum.BeforeTurnEndVeryEarly` latches its `Block == 0`
/// decision in the FIRST of three ordered turn-end passes, then grants in
/// the Normal pass — so an earlier-ordered Normal-pass grant (RippleBasin's
/// relic block, Plating's power block) does NOT cancel it. Rust runs every
/// `before_turn_end` listener in a single pass, so this must read the
/// `block_zero_at_turn_end` snapshot taken in `end_player_turn_phase_one`,
/// not live `player.block` (which a sibling grant in the same pass would
/// have already raised above 0). Matches the sts2-core `Orichalcum` /
/// `FakeOrichalcum` handlers (bead `sts2-rs-xk9yo`); reading live block here
/// dropped 3 turn-end block whenever FakeOrichalcum shared a combat with any
/// earlier-dispatched block-granting relic (coverage seed 2353: RippleBasin
/// granted 4 first, FakeOrichalcum then saw block=4 and skipped its 3).
fn listener_fake_orichalcum(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player || !player_owns(state, RelicKind::FakeOrichalcum) {
        return EffectResult::NotImplemented;
    }
    if !state.combat_log.block_zero_at_turn_end {
        return EffectResult::NotImplemented;
    }
    gain_block_player_unpowered(state, 3);
    EffectResult::Resolved
}

/// `DaughterOfTheWind::after_card_played` — `if card.card_type == ATTACK:
/// owner.gain_block(1, unpowered=True)` (Python `shop_event.py:1134-1136`).
/// `card_type` is read from the live `Card` via `card_uid`; the listener is
/// a no-op when the card is off-store (synthesized) or the relic is not owned.
fn listener_daughter_of_the_wind(state: &mut CombatState, ctx: &CardEventCtx) -> EffectResult {
    if !player_owns(state, RelicKind::DaughterOfTheWind) {
        return EffectResult::NotImplemented;
    }
    let Some(card_uid) = ctx.card_uid else {
        return EffectResult::NotImplemented;
    };
    let idx = card_uid.0.get() as usize - 1;
    let Some(card) = state.cards.get(idx) else {
        return EffectResult::NotImplemented;
    };
    if card.card_type != sts2_core::cards::CardType::Attack {
        return EffectResult::NotImplemented;
    }
    // C# DaughterOfTheWind.cs v0.103.2: BlockVar(1m, ValueProp.Unpowered).
    // Bead `sts2-rs-jo6sc`.
    gain_block_player_unpowered(state, 1);
    EffectResult::Resolved
}

/// `TheBoot::modify_hp_lost` — when `dealer is owner`, `target is not owner`,
/// `props & MOVE`, `not (props & UNPOWERED)`, and `0 < amount < 5`, floor the
/// amount up to 5 (Python `shop_event.py:2819-2829`). Ownership-gated.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_the_boot(state: &CombatState, ctx: &ModifyHpLostCtx, amount: i32) -> i32 {
    if !player_owns(state, RelicKind::TheBoot) {
        return amount;
    }
    let Some(dealer) = ctx.dealer else {
        return amount;
    };
    let Some(player_uid) = state.player.creature_uid else {
        return amount;
    };
    // C# `TheBoot.ModifyHpLostAfterOstyLate` (TheBoot.cs:27-31): the dealer
    // may be the owner OR the owner's Osty (GH #806).
    let dealer_is_owner_or_osty = dealer == player_uid || state.player.pet_uid == Some(dealer);
    if !dealer_is_owner_or_osty || ctx.target == player_uid {
        return amount;
    }
    if !ctx.props.contains(ValueProps::MOVE) || ctx.props.contains(ValueProps::UNPOWERED) {
        return amount;
    }
    if amount > 0 && amount < 5 {
        5
    } else {
        amount
    }
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 2 (sts2-rs-46ms)
// ---------------------------------------------------------------------------

fn run_owns(rs: &RunState, id: RelicId) -> bool {
    rs.owns_active_relic(id)
}

/// `FakeMango::after_obtained` — `owner.gain_max_hp(3)` (Python
/// `shop_event.py::FakeMango`, `MAX_HP=3`). Ownership-gated so the
/// dispatch is a no-op when the player doesn't carry the relic.
fn listener_fake_mango(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::FakeMango) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(3);
    EffectResult::Resolved
}

/// `FakeLeesWaffle::after_obtained` — heal `max_hp * 10 / 100` (Python
/// `shop_event.py::FakeLeesWaffle`, `HEAL_PCT=10`). Computes the heal
/// amount against the post-obtain `max_hp`, matching Python's
/// `owner.heal(owner.max_hp * 10 // 100)` ordering.
fn listener_fake_lees_waffle(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::FakeLeesWaffle) {
        return EffectResult::NotImplemented;
    }
    let amount = rs.max_hp * 10 / 100;
    rs.heal(amount);
    EffectResult::Resolved
}

/// `BigMushroom::after_obtained` — `owner.gain_max_hp(20)` (Python
/// `shop_event.py::BigMushroom`, `MAX_HP=20`). The companion round-1
/// `modify_hand_draw(-2)` leg is deferred (`modify_hand_draw` is a
/// combat-side channel and the round-1 gate needs a per-combat counter
/// POD).
fn listener_big_mushroom(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::BigMushroom) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(20);
    EffectResult::Resolved
}

/// `SwordOfJade::before_combat_start` — `apply_power(STRENGTH, 3)` to
/// owner (Python `shop_event.py::SwordOfJade`, `STRENGTH=3`).
fn listener_sword_of_jade(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::SwordOfJade) {
        return EffectResult::NotImplemented;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    apply_power_to_creature(state, player_uid, PowerKind::Strength, 3);
    EffectResult::Resolved
}

/// `FakeSneckoEye::before_combat_start` — `apply_power(CONFUSED, 1)`
/// to owner (Python `shop_event.py::FakeSneckoEye`, `CONFUSED=1`).
fn listener_fake_snecko_eye(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::FakeSneckoEye) {
        return EffectResult::NotImplemented;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    apply_power_to_creature(state, player_uid, PowerKind::Confused, 1);
    EffectResult::Resolved
}

/// `EmberTea::before_combat_start` — when the per-instance counter
/// (`_combats_left` on the Python class, `Relic.counter` here) is
/// positive, decrement it and `apply_power(STRENGTH, 2)` to the owner
/// (Python `shop_event.py::EmberTea`, `STRENGTH=2`, `COMBATS=5`).
///
/// Counter semantics: callers seed `Relic { counter: 5, .. }` at
/// obtain time; the listener decrements per combat start and stops
/// firing once the counter hits 0. No-op on uninitialised counters
/// (`counter <= 0`) — the listener treats those as "no charges left",
/// matching Python's "init to COMBATS" behaviour from the relic's
/// `__init__`. Bead `sts2-rs-3nis`.
fn listener_ember_tea(state: &mut CombatState) -> EffectResult {
    // Find the EmberTea relic instance and decrement its counter.
    let mut fired = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::EmberTea && relic.counter > 0 {
            relic.counter -= 1;
            fired = true;
            break;
        }
    }
    if !fired {
        return EffectResult::NotImplemented;
    }
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    apply_power_to_creature(state, player_uid, PowerKind::Strength, 2);
    EffectResult::Resolved
}

/// `BoneTea::after_side_turn_start` — on the owner's round-1
/// side-turn-start, upgrade every card in the player's hand, then
/// decrement the per-instance `_combats_left` counter (`Relic.counter`,
/// init 1). Mirrors C# `BoneTea.cs::AfterSideTurnStart`:
/// [decompiled C# omitted — see the cited path]
///
/// Counter semantics mirror [`listener_ember_tea`]: callers seed
/// `Relic { counter: 1, .. }` at obtain time; the listener fires once
/// (round-1 of the first combat after pickup) and then self-disables
/// (`counter == 0`). No-op on uninitialised counters. The
/// `enter_combat` mirror that seeds `relic.counter` from
/// `rs.relic_counters` is the shared deferred-mirror bridge — until it
/// lands the listener is inert in real runs but correctly shaped (tests
/// push the relic + counter directly into `state.relics`). Bead
/// `sts2-rs-eejzj`.
fn listener_bone_tea(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    // v0.111.0 runs on `AfterPlayerTurnStart` instead (see below).
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    bone_tea_upgrade_hand(state)
}

/// v0.111.0 `BoneTea.AfterPlayerTurnStart` (decomp BoneTea.cs:50-66): the
/// same round-1 hand upgrade, on the player-turn-start hook.
fn listener_bone_tea_player_turn_start(state: &mut CombatState) -> EffectResult {
    if state.game_version < sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    bone_tea_upgrade_hand(state)
}

/// BoneTea's body: on round 1, while `CombatsLeft > 0`, upgrade the hand and
/// spend one combat.
fn bone_tea_upgrade_hand(state: &mut CombatState) -> EffectResult {
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::BoneTea && r.counter > 0)
    else {
        return EffectResult::NotImplemented;
    };
    let hand: smallvec::SmallVec<[sts2_core::CardUid; 16]> =
        state.player.hand.iter().copied().collect();
    for uid in hand {
        upgrade_card_uid(state, uid);
    }
    state.relics[idx].counter -= 1;
    EffectResult::Resolved
}

/// `FakeVenerableTeaSet::after_room_entered` — on rest-site entry, latch
/// `rs.relic_counters[FakeVenerableTeaSet] = 1` so the next combat's
/// first energy reset grants `ENERGY = 1`. Mirrors C#
/// `FakeVenerableTeaSet.cs::AfterRoomEntered` (`if (room is
/// RestSiteRoom) GainEnergyInNextCombat = true;`). Idempotent on
/// repeated rest-site entries (the boolean flag has the same shape).
/// Sibling of the common-pool [`VenerableTeaSet`] latch
/// (`relics::common::listener_venerable_tea_set_rest_latch`). Bead
/// `sts2-rs-aqals`.
fn listener_fake_venerable_tea_set_rest_latch(
    rs: &mut RunState,
    room_type: RoomType,
) -> EffectResult {
    if !rs.owns_active_relic(RelicId::FakeVenerableTeaSet) {
        return EffectResult::NotImplemented;
    }
    if !matches!(room_type, RoomType::RestSite) {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters.insert(RelicId::FakeVenerableTeaSet, 1);
    EffectResult::Resolved
}

/// `FakeVenerableTeaSet::after_energy_reset` — if the rest-site latch is
/// set (`Relic.counter == 1`), grant `ENERGY = 1` to the player and
/// clear the latch. Mirrors C# `FakeVenerableTeaSet.cs::AfterEnergyReset`
/// (`if (Owner == player && GainEnergyInNextCombat) { GainEnergy(1);
/// GainEnergyInNextCombat = false; }`). Fires on the first energy reset
/// of the combat (latch consumed → subsequent resets no-op).
///
/// The latch storage lives on the combat-side `Relic.counter` slot; the
/// `enter_combat` bridge that mirrors `rs.relic_counters` into this
/// counter is the shared deferred-mirror follow-up. Until it lands the
/// consume is inert in real runs but correctly shaped (tests push the
/// relic + counter directly into `state.relics`). Bead `sts2-rs-aqals`.
fn listener_fake_venerable_tea_set_consume(state: &mut CombatState) -> EffectResult {
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::FakeVenerableTeaSet)
    else {
        return EffectResult::NotImplemented;
    };
    if state.relics[idx].counter != 1 {
        return EffectResult::NotImplemented;
    }
    state.relics[idx].counter = 0;
    state.gain_energy(FAKE_VENERABLE_TEA_SET_ENERGY);
    EffectResult::Resolved
}

/// `Byrdpip::before_combat_start` — summon the behaviourless Byrdpip pet
/// (C# `Byrdpip.cs::BeforeCombatStart` → `PlayerCmd.AddPet<Byrdpip>`).
/// The pet is a `NOTHING_MOVE`, no-health-bar companion with no combat
/// behaviour of its own, spawned via [`CombatState::summon_pet`]
/// (idempotent per combat). Bead `sts2-rs-4x616`.
fn listener_byrdpip_summon_pet(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::Byrdpip) {
        return EffectResult::NotImplemented;
    }
    let _ = state.summon_pet(sts2_core::MonsterKind::Byrdpip, BYRDPIP_PET_HP);
    EffectResult::Resolved
}

/// `Byrdpip::after_obtained` — transform every `ByrdonisEgg` in the
/// master deck into a fresh (un-upgraded) `ByrdSwoop`. Mirrors C#
/// `Byrdpip.cs::AfterObtained`, which iterates the deck and runs
/// `CardCmd.TransformTo<ByrdSwoop>` on each `ByrdonisEgg`. The C# method
/// also transforms the combat-side `PlayerCombatState.AllCards` and
/// re-summons the pet when obtained mid-combat; at this pin `after_obtained`
/// is a run-side (RunState-only) hook, so the combat-side leg is deferred
/// (relics are obtained on the reward screen out of combat in the common
/// flow). Bead `sts2-rs-4x616`.
fn listener_byrdpip_transform_eggs(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::Byrdpip) {
        return EffectResult::NotImplemented;
    }
    // `CardCmd.TransformTo` removes each egg and APPENDS its Byrd Swoop to
    // the deck through the add-to-deck hooks (CardCmd.cs:402,427-436); an
    // in-place rewrite kept the egg's slot, so the next combat's shuffle
    // dealt a different opening hand (Regent act23 seed 1011, rest-site
    // Hatch).
    let mut indices: Vec<usize> = Vec::new();
    let mut replacements: Vec<sts2_core::cards::Card> = Vec::new();
    for (i, card) in rs.master_deck.iter().enumerate() {
        if card.card_id == sts2_core::cards::CardId::ByrdonisEgg {
            if let Some(mut swoop) = crate::cards::spec_index::card_pod_from_id(
                sts2_core::cards::CardId::ByrdSwoop,
                false,
            ) {
                swoop.uid = card.uid;
                indices.push(i);
                replacements.push(swoop);
            }
        }
    }
    if rs.transform_specific_cards(&indices, replacements) > 0 {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

/// `HistoryCourse::before_play_phase_start` — after round 1, auto-play a
/// free dupe of the last Attack/Skill the player played the previous
/// round. Mirrors C# `HistoryCourse.cs::BeforePlayPhaseStart`:
/// [decompiled C# omitted — see the cited path]
///
/// "Last Attack/Skill of the previous round" is tracked by the single
/// `combat_log.last_attack_skill_played` field (read-then-reset at each
/// play-phase boundary — see its doc-comment). The auto-played dupe runs
/// the card's effect directly via [`sts2_core::card_effects::dispatch`]
/// (the same primitive EchoForm / OneTwoPunch replays use), so it spends
/// no energy, is not added to any pile, and — by NOT routing through
/// `play_card` — never updates `last_attack_skill_played`, matching C#'s
/// `IsDupe` exclusion (the dupe can't trigger itself next round).
///
/// Targeting mirrors C# `CardCmd.AutoPlay(…, target: null)`: an `AnyEnemy`
/// card with a null target resolves it via
/// `Rng.CombatTargets.NextItem(HittableEnemies)` — a RANDOM alive enemy that
/// consumes the `combat_targets` stream (even on a single-enemy board;
/// `NextItem` on a 1-element list still reads). Self / non-targeted dupes
/// pass no target and consume no RNG. An `AnyEnemy` dupe with no resolvable
/// target is moved to its result pile without playing (no dispatch).
/// Bead `sts2-rs-ummci`; coverage-sweep `relic:HISTORY_COURSE`.
fn listener_history_course(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::HistoryCourse) {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() <= 1 {
        return EffectResult::NotImplemented;
    }
    let Some((card_id, upgraded, src_uid)) = state.combat_log.prev_round_last_attack_skill_played
    else {
        return EffectResult::NotImplemented;
    };
    // C# `CardCmd.AutoPlay(card.CreateDupe())`: synthesize a dupe card
    // instance and play it for free. Many handlers read the originating
    // card's stats off the top of the play pile (`current_card_damage` /
    // `current_card_block` / `current_card_upgraded`), so the dupe must be
    // pushed onto `state.cards` + the play pile before dispatch, then
    // popped off after (it spends no energy and joins no pile — a
    // transient replay, exactly like EchoForm/OneTwoPunch re-dispatch but
    // for an off-deck dupe).
    //
    // `CreateDupe()` clones the live card MODEL, so per-instance state
    // accumulated during combat (Rampage's +5/play `RampageDamage` var,
    // Thrash's exhaust fold into `base_damage`) carries into the dupe —
    // clone the source instance when it still resolves, falling back to
    // the spec POD only for a stale uid (sweep seeds 2963/3087/4902).
    let Some(mut dupe) = src_uid
        .and_then(|uid| state.cards.get(uid.0.get() as usize - 1).copied())
        .filter(|c| c.card_id == card_id)
        .or_else(|| crate::cards::spec_index::card_pod_from_id(card_id, upgraded))
    else {
        return EffectResult::NotImplemented;
    };
    // Resolve the dupe's target the way C# `AutoPlay(…, null)` does: a random
    // alive enemy (consuming `combat_targets`) for `AnyEnemy` cards, no
    // target otherwise. The prior port picked the first alive enemy
    // deterministically, which both mis-targeted multi-enemy boards and
    // skipped the RNG read — desyncing every later combat roll.
    // `AutoPlay` reads the card's live `TargetType` (decomp CardCmd.cs:74),
    // and a Shiv under FanOfKnives is `AllEnemies` (Shiv.cs:23-32), so its
    // dupe rolls no `CombatTargets` target. Reading the POD's `Enemy` spent a
    // combat_targets draw C# never makes (Silent mid seed 1472, SEAPUNK_WEAK).
    // Same C# in v0.107.1 and v0.111.0, so both builds (GH #806).
    let dupe_target_type = sts2_core::cards::effective_target_type(state, &dupe);
    // C# `CardCmd.AutoPlay` checks `Hook.ShouldPlay` BEFORE resolving the
    // target / playing (`CardCmd.cs:64-73`): a Ringing / Sloth / Bound-blocked
    // auto-play is rerouted via `MoveToResultPileWithoutPlaying` — the card is
    // added to the Play pile (firing `AfterCardEnteredCombat`) then immediately
    // moved to its result pile (None for a dupe → removed). No CombatTargets
    // RNG roll, no `OnPlay`, no after-played buses, no play counters. The
    // RingingPower limits the player to ONE card play per turn, so when Mayhem
    // (a Power, dispatched before relics) has already auto-played a card this
    // turn, the HistoryCourse dupe is blocked (Ironclad seed 2135: at turn-5
    // start Mayhem auto-plays STOMP, then the HistoryCourse BULLY dupe is
    // Ringing-blocked in C# — beast hp cs=118; Rust replayed the dupe for +4 →
    // rs=114). The minimal dupe path previously skipped this gate. The Ringing
    // gate uses STARTED plays, so it sees Mayhem's just-resolved Stomp. The
    // CombatTargets roll happens only AFTER ShouldPlay in C#, so a blocked dupe
    // must NOT consume the `combat_targets` stream.
    let blocked = sts2_core::combat_loop::player_should_play_blocked(state, &dupe);
    let target = match dupe_target_type {
        sts2_core::cards::TargetType::Enemy if !blocked => state.get_random_enemy(),
        _ => None,
    };
    // An `AnyEnemy` dupe with no hittable enemy (`NextItem` over an empty
    // `HittableEnemies` returns null: every enemy dead, e.g. TestSubject
    // between revives) takes the same `MoveToResultPileWithoutPlaying` route
    // (CardCmd.cs:74-85, both builds): no BeforeCardPlayed (Stomp shave,
    // FreeAttack), no OnPlay, no play counters (Ironclad-2176 act 3: a phantom
    // Bash dupe banked a Stomp shave, Drum-drawn STOMP C# 3, Rust 2).
    let no_target =
        matches!(dupe_target_type, sts2_core::cards::TargetType::Enemy) && target.is_none();
    if blocked || no_target {
        let new_uid = sts2_core::CardUid(
            std::num::NonZeroU32::new(u32::try_from(state.cards.len() + 1).unwrap_or(u32::MAX))
                .expect("non-zero"),
        );
        dupe.uid = new_uid;
        dupe.set_combat_var(sts2_core::cards::VarKey::IsDupe, 1);
        state.cards.push(dupe);
        state.player.play.push(new_uid);
        // C# `MoveToResultPileWithoutPlaying` adds the card to Play, which fires
        // the two-arg `AfterCardEnteredCombat` (monster powers), then removes the
        // IsDupe card from combat (result pile None).
        let _ =
            sts2_core::monster_power_effects::dispatch_after_card_entered_combat_on_monster_powers(
                state, new_uid,
            );
        let popped = state.player.play.pop();
        debug_assert_eq!(
            popped,
            Some(new_uid),
            "HistoryCourse blocked-dupe play-pile imbalance"
        );
        return EffectResult::Resolved;
    }
    let new_uid = sts2_core::CardUid(
        std::num::NonZeroU32::new(u32::try_from(state.cards.len() + 1).unwrap_or(u32::MAX))
            .expect("non-zero"),
    );
    dupe.uid = new_uid;
    // C# `CreateDupe()` stamps `IsDupe` — `GetResultPileType` returns None,
    // the HC scan excludes it, and any card ITS OnPlay clones inherits the
    // flag (MemberwiseClone; sweep seed 5192). The instance-clone above may
    // carry stale vars; force the flag regardless of source.
    dupe.set_combat_var(sts2_core::cards::VarKey::IsDupe, 1);
    state.cards.push(dupe);
    // C# `CardCmd.AutoPlay` adds the freshly-`CreateDupe()`'d card (`Pile==null`)
    // to the Play pile via plain `CardPileCmd.Add` (`CardCmd.cs:115-117`), which
    // fires `Hook.AfterCardEnteredCombat(combatState, card)` (`CardPileCmd.cs:520`)
    // — the TWO-arg "entered combat" hook (no `creator`). It is NOT routed
    // through `AddGeneratedCardsToCombat`, so the THREE-arg
    // `AfterCardGeneratedForCombat(card, creator)` hook does NOT fire: Regalite
    // and the other generation-only listeners (Arsenal / Pillar / Smokestack /
    // RocketPunch) grant nothing for the dupe (Ironclad seed 3238). The monster
    // powers DO override the two-arg hook, so dispatch theirs.
    state.player.play.push(new_uid);
    let _ = sts2_core::monster_power_effects::dispatch_after_card_entered_combat_on_monster_powers(
        state, new_uid,
    );
    // The rest is the ordinary auto-play: C# runs the dupe through the same
    // `OnPlayWrapper` as any `CardCmd.AutoPlay` — `BeforeCardPlayed` (Stomp,
    // FreeAttack, Surrounded, DanseMacabre, …), the X capture from current
    // energy, EchoForm and `GeneratePlayCount` replays, the after-played buses
    // and tallies, and the result pile (`None` for a dupe, Exhaust for a Skill
    // under Corruption). A select it opens prompts the player and the dupe
    // waits for it (GH #1131); `finish_turn_start_tail` waits too.
    let _ = sts2_core::combat_loop::auto_play_card_on_play(state, new_uid, target);
    EffectResult::Resolved
}

/// `FakeStrikeDummy::modify_damage_additive` — +1 when the originating
/// card carries `CardTag::Strike` (Python `shop_event.py::FakeStrikeDummy`,
/// `BONUS=1`). Reads `ctx.card` (4nh5 surface) and locates the first
/// `CardInstance` in `state.cards` matching that `CardId` to inspect
/// its `tags`. Returns `0` (no contribution) when ungated.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_fake_strike_dummy(state: &CombatState, ctx: &ModifyDamageCtx) -> i32 {
    if !player_owns(state, RelicKind::FakeStrikeDummy) {
        return 0;
    }
    let Some(card_id) = ctx.card else {
        return 0;
    };
    let is_strike = state
        .cards
        .iter()
        .find(|c| c.card_id == card_id)
        .is_some_and(|c| c.has_tag(CardTag::Strike));
    i32::from(is_strike)
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 4 (sts2-rs-v7mp)
// ---------------------------------------------------------------------------

/// `RoyalPoison::after_player_turn_start` (RoyalPoison.cs:18) — deal 4 unblockable
/// (`UNBLOCKABLE | UNPOWERED`) damage to the owner on the player's
/// round-1 turn-start (Python `shop_event.py::RoyalPoison`,
/// `DAMAGE=4`). Routes through
/// [`sts2_core::combat_loop::apply_damage_to_player`] with the
/// canonical `props` bitset so block absorption is skipped and the
/// `before_damage_received` / `after_damage_received` event hooks fire
/// just like a normal player-target attack.
///
/// The Python pin passes `combat.deal_damage(None, owner, …)` — i.e.
/// the dealer ref is null. The Rust pipeline expects a `CreatureRef`,
/// so we synthesize `CreatureRef::Player` (the relic-owned damage
/// source) to keep the hook dispatchers happy. This is observationally
/// identical for the relevant fixtures because no `modify_damage`
/// listener at this pin gates on `dealer == None`.
fn listener_royal_poison(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::RoyalPoison) {
        return EffectResult::NotImplemented;
    }
    if state.turn_number() != 1 {
        return EffectResult::NotImplemented;
    }
    let _ = sts2_core::combat_loop::apply_damage_to_player(
        state,
        CreatureRef::Player,
        4,
        ValueProps::UNBLOCKABLE.union(ValueProps::UNPOWERED),
        None,
    );
    EffectResult::Resolved
}

/// `MrStruggles::after_side_turn_start` — deal damage to all alive enemies
/// on the player's turn-start. C# `MrStruggles.cs:21`
/// `CreatureCmd.Damage(HittableEnemies, <amount>, Unpowered, Owner)`.
///
/// AMOUNT — v0.107.1 changed it from `combatState.RoundNumber` (v0.103.2 /
/// Python `combat.round_number`) to `Owner.PlayerCombatState.TurnNumber`.
/// These differ ONLY when the player takes an EXTRA turn (PaelsEye et al.),
/// where TurnNumber increments but RoundNumber does not. This engine does
/// not model extra turns yet (PaelsEye extra-turn handling is deferred —
/// see [`sts2-rs-v0107-content-port-wb8lk.9`] follow-ups), so in every
/// currently-reachable state `round_no()` == `TurnNumber`. This is the same
/// `RoundNumber -> PlayerTurnNumber` swap §1a.2 already classified as a solo
/// no-op; `round_no()` stays the source until extra-turn support lands, at
/// which point this (and the §1a.2 relics) switch to the player turn count.
///
/// No counter — the class is plain, no per-instance state.
fn listener_mr_struggles(state: &mut CombatState) -> EffectResult {
    if !player_owns(state, RelicKind::MrStruggles) {
        return EffectResult::NotImplemented;
    }
    // v0.107.1 amount = PlayerCombatState.TurnNumber; == round_no() until
    // extra-turn support lands (see doc above).
    let damage = i32::from(state.turn_number());
    if damage <= 0 {
        return EffectResult::NotImplemented;
    }
    let _ = deal_damage_to_all_alive_enemies(state, damage, ValueProps::UNPOWERED, None);
    EffectResult::Resolved
}

/// `LostWisp::after_card_played` — deal 8 damage to all alive enemies
/// when a Power-type card is played (Python `shop_event.py::LostWisp`,
/// `DAMAGE=8`, gate `card.card_type == CardType.POWER`). The
/// `CardEventCtx` carries `card_id`; we re-derive `card_type` from the
/// matching `Card` row in `state.cards` (the same lookup pattern used
/// by `listener_daughter_of_the_wind`).
fn listener_lost_wisp(state: &mut CombatState, ctx: &CardEventCtx) -> EffectResult {
    if !player_owns(state, RelicKind::LostWisp) {
        return EffectResult::NotImplemented;
    }
    let Some(card_uid) = ctx.card_uid else {
        return EffectResult::NotImplemented;
    };
    let idx = card_uid.0.get() as usize - 1;
    let Some(card) = state.cards.get(idx) else {
        return EffectResult::NotImplemented;
    };
    if card.card_type != sts2_core::cards::CardType::Power {
        return EffectResult::NotImplemented;
    }
    let _ = deal_damage_to_all_alive_enemies(state, 8, ValueProps::UNPOWERED, None);
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 5 (sts2-rs-c7ak)
// ---------------------------------------------------------------------------

/// `FakeHappyFlower::after_side_turn_start` — every 5 player turns,
/// `combat.gain_energy(owner, 1)` (Python
/// `shop_event.py::FakeHappyFlower`, `ENERGY=1`, `TURNS=5`, counter
/// `_turns_seen` cycled modulo 5).
///
/// Counter semantics: the `_turns_seen` field on the Python class
/// (`__init__` initialises to 0, post-increment in
/// `after_side_turn_start`, fires when `_turns_seen % 5 == 0`)
/// lives on the per-instance [`sts2_core::Relic::counter`] POD
/// (bead `sts2-rs-3nis`). The listener post-increments
/// `Relic.counter` and applies energy when the new value is a
/// non-zero multiple of 5 — observably identical to Python.
fn listener_fake_happy_flower(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    // Find FakeHappyFlower in `state.relics` and post-increment its
    // counter (mirroring `self._turns_seen += 1` in Python).
    let mut new_count = 0_i32;
    let mut found = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::FakeHappyFlower {
            relic.counter = relic.counter.saturating_add(1);
            new_count = relic.counter;
            found = true;
            break;
        }
    }
    if !found {
        return EffectResult::NotImplemented;
    }
    if new_count <= 0 || new_count % 5 != 0 {
        return EffectResult::NotImplemented;
    }
    state.gain_energy(1);
    EffectResult::Resolved
}

/// `TeaOfDiscourtesy::before_combat_start` — add 2 Dazed to the draw
/// pile when the per-instance `_combats_left` counter is positive,
/// then decrement (Python `shop_event.py::TeaOfDiscourtesy`,
/// `DAZED_COUNT=2`, `COMBATS=1`).
///
/// Counter semantics: callers seed `Relic { counter: 1, .. }` at
/// obtain time; the listener fires once (`counter > 0`) and clears.
/// Builds each `Dazed` POD from the status-pool spec
/// (`super::super::cards::status_specs_generated::dazed`) and inserts
/// both at a random draw-pile depth. C#
/// `TeaOfDiscourtesy.BeforeCombatStart` (`TeaOfDiscourtesy.cs`) calls
/// `CardPileCmd.AddToCombatAndPreview<Dazed>(owner, Draw, 2, .., CardPilePosition.Random)`,
/// which loops `AddGeneratedCardToCombat` per card →
/// `AddInternal(card, Rng.Shuffle.NextInt(Cards.Count + 1))`
/// (`CardPileCmd.cs:433`). Each Dazed lands at a fresh random index and
/// consumes one Shuffle-stream draw. The prior top-insert
/// (`add_card_to_draw_pile`) diverged on PILE_COUNT/DRAW_ORDER and
/// consumed zero Shuffle draws. Bead `sts2-rs-n4sa4`.
fn listener_tea_of_discourtesy(state: &mut CombatState) -> EffectResult {
    // Find TeaOfDiscourtesy and decrement its counter.
    let mut fired = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::TeaOfDiscourtesy && relic.counter > 0 {
            relic.counter -= 1;
            fired = true;
            break;
        }
    }
    if !fired {
        return EffectResult::NotImplemented;
    }
    for _ in 0..2 {
        let dazed =
            crate::cards::status::card_from_spec(&crate::cards::status_specs_generated::dazed());
        let _ = sts2_core::card_effects::spawn_card_into_draw_pile_random_position(state, dazed);
    }
    EffectResult::Resolved
}

/// `HandDrill::after_damage_given` — when the dealer is the owner
/// (player) and the most-recent attack hit broke the target's block
/// (`results[-1].blocked > 0 && target.block == 0`), apply
/// `Vulnerable(2)` to the target (Python
/// `shop_event.py::HandDrill`, `VULNERABLE=2`).
///
/// Gates:
///   1. `state.active_attack().is_some()` — Python's
///      `attack = combat.active_attack; if attack is None: return`.
///   2. `dealer == player` — Python's `if dealer is not owner: return`.
///   3. Last `AttackResult.blocked > 0` AND target's post-hit
///      `creature.block == 0` — Python's `result.blocked > 0 and
///      target.block == 0`.
///
/// Wiring note: the `active_attack` field defaults to `None` and is
/// not auto-pushed by `combat_loop::play_card` at this pin
/// (deferred to a follow-up bead — see
/// [`CombatState::active_attack`](sts2_core::CombatState::active_attack)
/// doc-comment). Until then the gate at (1) is `false` in production
/// combat and this listener is observationally inert outside of
/// tests that drive the field manually.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_hand_drill(state: &mut CombatState, ctx: &DamageGivenCtx) -> EffectResult {
    if !player_owns(state, RelicKind::HandDrill) {
        return EffectResult::NotImplemented;
    }
    // v0.111.0 HandDrill triggers from `AfterBlockBroken` instead (decomp
    // HandDrill.cs:21), the relic arm of `hooks::dispatch_after_block_broken`.
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    // Gate on `dealer == player` (C# `dealer == Owner.Creature ||
    // dealer.PetOwner == Owner` — no pets at this pin).
    let Some(player_uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    if ctx.dealer != player_uid {
        return EffectResult::NotImplemented;
    }
    // C# guards `!target.IsPlayer`. This listener also runs on the
    // player-received bus, so guard explicitly even though that path
    // already passes `was_block_broken: false`.
    if state.player.creature_uid == Some(ctx.target) {
        return EffectResult::NotImplemented;
    }
    // C# `result.WasBlockBroken` — threaded onto the ctx by the damage
    // pipeline (mirrors `CreatureCmd.cs:153`).
    if !ctx.was_block_broken {
        return EffectResult::NotImplemented;
    }
    // C# HandDrill.cs:27 — `PowerCmd.Apply<VulnerablePower>(target,
    // DynamicVars.Vulnerable.BaseValue=2, ...)`. VulnerablePower is
    // StackType=Counter so this stacks additively with any existing
    // Vulnerable.
    // Its `cardSource` is null, so Unsettling Lamp ignores it.
    sts2_core::card_effects::apply_power_to_creature_cardless(
        state,
        ctx.target,
        PowerKind::Vulnerable,
        2,
    );
    EffectResult::Resolved
}

/// `ForgottenSoul::after_card_exhausted` — deal 1 damage to a random
/// alive enemy when any card is exhausted (Python
/// `shop_event.py::ForgottenSoul`, `DAMAGE=1`,
/// `target = combat.get_random_enemy()`). Uses the v7mp
/// `CombatState::get_random_enemy` surface (RNG-keyed alive-enemy
/// pick). No-op when no enemy is alive.
fn listener_forgotten_soul(state: &mut CombatState, _ctx: &CardEventCtx) -> EffectResult {
    if !player_owns(state, RelicKind::ForgottenSoul) {
        return EffectResult::NotImplemented;
    }
    let Some(target_uid) = state.get_random_enemy() else {
        return EffectResult::NotImplemented;
    };
    let _ = sts2_core::damage::apply_damage(
        state,
        CreatureRef::Player,
        target_uid,
        1,
        ValueProps::UNPOWERED,
        None,
    );
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 5 (sts2-rs-acyi)
// ---------------------------------------------------------------------------

/// `ChosenCheese::after_combat_end` — `owner.gain_max_hp(1)` (C#
/// `ChosenCheese.AfterCombatEnd`, `MaxHpVar(1)`). Registered onto the
/// run-side `after_combat_end` channel
/// ([`sts2_core::run_effects::dispatch_after_combat_end`]) by
/// [`install_listeners`]; fired from
/// `RunManager::exit_combat_and_transition` (the C#
/// `Hook.AfterCombatEnd` back-edge, `CombatManager.cs:784`).
/// Ownership-gated so the dispatch is a no-op when the player doesn't
/// carry the relic. Bead `sts2-rs-acyi`; firing site wired for the
/// RP16 run-parity hp/max_hp +1 family.
fn listener_chosen_cheese(rs: &mut RunState) -> EffectResult {
    if !run_owns(rs, RelicId::ChosenCheese) {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(1);
    EffectResult::Resolved
}

/// `DarkstonePeriapt::on_card_added_to_deck` — if the added card is a
/// Curse, `owner.gain_max_hp(6)` (Python `shop_event.py:1121-1123` +
/// `run_state.py:303-306` where the `CardType::Curse` gate lives in the
/// firing site). Registered onto the run-side `on_card_added_to_deck`
/// channel ([`sts2_core::run_effects::dispatch_on_card_added_to_deck`])
/// by [`install_listeners`]. The Rust port pushes the Curse-gate down
/// into the listener (rather than the firing site) so a single
/// dispatch path covers all listeners; the Python code keys on the
/// relic class identity instead. Ownership-gated. Bead `sts2-rs-acyi`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_darkstone_periapt(rs: &mut RunState, ctx: &CardAddedRunCtx) -> EffectResult {
    if !run_owns(rs, RelicId::DarkstonePeriapt) {
        return EffectResult::NotImplemented;
    }
    if ctx.card_type != CardType::Curse {
        return EffectResult::NotImplemented;
    }
    rs.gain_max_hp(6);
    EffectResult::Resolved
}

/// Run-side `BingBong.AfterCardChangedPiles` (`BingBong.cs:28-38`, both builds):
///
/// [decompiled C# omitted — see the cited path]
///
/// The clone is a full hooked deck add at the bottom (eggs, LuckyFysh, …), whose
/// own `AfterCardChangedPiles` carries `clonedBy = this`, so BingBong skips
/// it: here [`AddSource::BingBongDuplicate`]. The added card is the newest deck
/// card with the context's id (a batch add appends every card before any
/// listener runs, `CardPileCmd.cs:497-636`).
///
/// [`AddSource::BingBongDuplicate`]: sts2_core::relic_effects::AddSource::BingBongDuplicate
fn listener_bing_bong_run(rs: &mut RunState, ctx: &CardAddedRunCtx) -> EffectResult {
    if ctx.source == sts2_core::relic_effects::AddSource::BingBongDuplicate {
        return EffectResult::NotImplemented;
    }
    if !run_owns(rs, RelicId::BingBong) {
        return EffectResult::NotImplemented;
    }
    let Some(clone) = rs
        .master_deck
        .iter()
        .rev()
        .find(|c| c.card_id == ctx.card_id)
        .copied()
    else {
        return EffectResult::NotImplemented;
    };
    rs.add_card_to_deck_with_hooks(
        clone,
        sts2_core::relic_effects::AddSource::BingBongDuplicate,
    );
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 6 (sts2-rs-9wx7) — reward-flow retrofits
// ---------------------------------------------------------------------------

// Constants ported from Python `shop_event.py`.
/// `FakeVenerableTeaSet` energy granted on the first energy reset of the
/// next combat after a rest (C# `EnergyVar(1)`). Bead `sts2-rs-aqals`.
const FAKE_VENERABLE_TEA_SET_ENERGY: i32 = 1;
/// `Byrdpip` pet HP — the pet is a behaviourless 9999-HP companion
/// (C# `Monsters/Byrdpip.cs` `MinInitialHp == MaxInitialHp == 9999`).
/// Bead `sts2-rs-4x616`.
const BYRDPIP_PET_HP: i32 = 9999;
const FRESNEL_LENS_NIMBLE: i32 = 2;
const WONGOS_COMBATS_NEEDED: i32 = 5;
const WONGOS_RELICS: usize = 3;

// `WongosMysteryTicket` carries two Python counters
// (`_combats_finished` 0..=5 and `_gave_relic` bool). Encoded into a
// single `i32` in `RunState::relic_counters`:
//   `value < 0`  → `_gave_relic = true` (the relic has already paid out).
//   `value >= 0` → `_combats_finished = value`.
// The audit doc footer flagged the single-slot conflict; this
// encoding is the canonical resolution.
const WONGOS_GAVE_SENTINEL: i32 = -1;

/// `DreamCatcher::modify_rest_site_heal_rewards` — push one `CardReward`
/// on EVERY rest-site heal (real or mimicked, e.g. Dense Vegetation's
/// Rest). C# `DreamCatcher.TryModifyRestSiteHealRewards` (identical in
/// v0.107.1 and v0.111.0) has no counter and no cap. The old Python port's
/// 3-proc cap kept its count in `relic_counters`, where the end-of-combat
/// counter flush-back wrote the combat relic's 0 over it, switching the
/// relic off after its first combat. The card is
/// `CardReward(CardCreationOptions.ForRoom(player, RoomType.Monster), 3)`:
/// an `Encounter`-source roll at RegularEncounter odds (pity, upgrade roll),
/// not a base-odds non-combat one. Ownership-gated. Bead `sts2-rs-9wx7`.
fn listener_dream_catcher(rs: &mut RunState, rewards: &mut Vec<Reward>) -> EffectResult {
    if !run_owns(rs, RelicId::DreamCatcher) {
        return EffectResult::NotImplemented;
    }
    rewards.push(Reward::Card(CardReward::for_room(RoomType::Monster)));
    EffectResult::Resolved
}

/// `FresnelLens::modify_card_reward_options_late` — every card in the
/// reward option list gets `Nimble(2)` (`shop_event.py:1539-1551`,
/// `NIMBLE=2`). Ownership-gated. Bead `sts2-rs-9wx7`.
fn listener_fresnel_lens_card_reward(rs: &mut RunState, cards: &mut Vec<Card>) -> EffectResult {
    if !run_owns(rs, RelicId::FresnelLens) {
        return EffectResult::NotImplemented;
    }
    for card in cards {
        if nimble_can_enchant(card, rs.game_version) {
            card.add_enchantment(EnchantmentId::Nimble, FRESNEL_LENS_NIMBLE);
        }
    }
    EffectResult::Resolved
}

/// C# `ModelDb.Enchantment<Nimble>().CanEnchant(card)`, the gate on every
/// FresnelLens hook (`FresnelLens.cs:47,58`): the base `EnchantmentModel`
/// rules (no Status / Curse / Quest, no Unplayable, `EnchantmentModel.cs:273-294`)
/// plus `card.GainsBlock` (`Nimble.cs:7-14`), via
/// [`crate::enchantments::can_enchant_card`]. No enchantment is stackable
/// (`EnchantmentModel.IsStackable => false`, `:173`), so a card that already
/// carries one, including the Nimble the reward slate stamped, is refused:
/// the picked reward card keeps Nimble 2 instead of the deck-add hook stacking
/// it to 4. Every run-side deck add runs the deck-add hook now (GH #806), so
/// without this gate FresnelLens also stamped Nimble on added curses.
fn nimble_can_enchant(card: &Card, version: sts2_core::GameVersion) -> bool {
    card.enchantments.is_empty()
        && crate::names::find_deck_card_spec(&crate::names::deck_card_spec_pool(), card.card_id)
            .is_some_and(|spec| {
                crate::enchantments::can_enchant_card(&spec, card, EnchantmentId::Nimble, version)
            })
}

/// `FresnelLens::modify_card_being_added_to_deck` — the added card
/// gets `Nimble(2)` (`shop_event.py:1553-1555`). Ownership-gated.
/// Bead `sts2-rs-9wx7`.
fn listener_fresnel_lens_deck_add(rs: &mut RunState, card: &mut Card) -> EffectResult {
    if !run_owns(rs, RelicId::FresnelLens) {
        return EffectResult::NotImplemented;
    }
    if !nimble_can_enchant(card, rs.game_version) {
        return EffectResult::NotImplemented;
    }
    card.add_enchantment(EnchantmentId::Nimble, FRESNEL_LENS_NIMBLE);
    EffectResult::Resolved
}

/// `FresnelLens::modify_merchant_card_creation_results` — every
/// merchant-spawned card gets `Nimble(2)`
/// (`shop_event.py:1557-1565`). The `is_colorless` kwarg is unused
/// (Python's body doesn't gate on it). Ownership-gated. Bead
/// `sts2-rs-9wx7`.
fn listener_fresnel_lens_merchant(
    rs: &mut RunState,
    card: &mut Card,
    _is_colorless: bool,
) -> EffectResult {
    if !run_owns(rs, RelicId::FresnelLens) {
        return EffectResult::NotImplemented;
    }
    // `EnchantValidCards` (`FresnelLens.cs:56-66`): Nimble-eligible only.
    if !nimble_can_enchant(card, rs.game_version) {
        return EffectResult::NotImplemented;
    }
    card.add_enchantment(EnchantmentId::Nimble, FRESNEL_LENS_NIMBLE);
    EffectResult::Resolved
}

/// `WongosMysteryTicket::modify_rewards` — once 5 combats have
/// finished, push 3 `RelicReward` entries on the next Monster /
/// Elite / Boss room reward and latch `_gave_relic = true`
/// (`shop_event.py:3057-3072`). Ownership-gated. The counter packing
/// is documented on [`WONGOS_GAVE_SENTINEL`]. Bead `sts2-rs-9wx7`.
fn listener_wongos_mystery_ticket_rewards(
    rs: &mut RunState,
    rewards: &mut Vec<Reward>,
    room_type: RoomType,
) -> EffectResult {
    if !run_owns(rs, RelicId::WongosMysteryTicket) {
        return EffectResult::NotImplemented;
    }
    let counter = rs
        .relic_counters
        .get(&RelicId::WongosMysteryTicket)
        .copied()
        .unwrap_or(0);
    if counter < 0 || counter < WONGOS_COMBATS_NEEDED {
        return EffectResult::NotImplemented;
    }
    if !matches!(
        room_type,
        RoomType::Monster | RoomType::Elite | RoomType::Boss
    ) {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters
        .insert(RelicId::WongosMysteryTicket, WONGOS_GAVE_SENTINEL);
    for _ in 0..WONGOS_RELICS {
        rewards.push(Reward::Relic(RelicReward::new(
            RelicId::WongosMysteryTicket,
        )));
    }
    EffectResult::Resolved
}

// ── LavaRock (bead `sts2-rs-a6j4v`) ──────────────────────────────────

/// Number of `RelicReward` sentinels LavaRock pushes onto the post-Act-1
/// boss reward set. Python `LavaRock.RELICS = 2`
/// (`shop_event.py::LavaRock`).
const LAVA_ROCK_RELICS: usize = 2;

/// `relic_counters` value LavaRock writes once it has fired — gates the
/// listener so subsequent boss rooms (re-entry, scripted fixtures) can't
/// re-trigger. Python `_triggered: bool = True`. Stored as `i32` because
/// `relic_counters` is `BTreeMap<RelicId, i32>` and we already use this
/// pattern for other once-per-run latches.
const LAVA_ROCK_TRIGGERED: i32 = 1;

/// `LavaRock::modify_rewards` — after the Act 1 boss is cleared, push
/// 2 `RelicReward` sentinels onto the reward set and latch the
/// `relic_counters` entry so the bonus only fires once.
///
/// Mirrors Python `LavaRock.modify_rewards`
/// (`shop_event.py:2277-2295`): gated on ownership + room being a Boss
/// + Python's `current_act_index == 0` (0-based) ↔ Rust's `act_index ==
/// 1` (1-based per
/// `sts2-rl-agent@01c094d:sts2_env/run/run_state.py:1340`). Python
/// sets `self._triggered = True; self.enabled = False`; the Rust port
/// uses `relic_counters[LavaRock] = LAVA_ROCK_TRIGGERED` as the
/// once-per-run latch (no separate `enabled` field — the counter
/// guard is sufficient).
///
/// Bead `sts2-rs-a6j4v` (Wave 7).
fn listener_lava_rock_rewards(
    rs: &mut RunState,
    rewards: &mut Vec<Reward>,
    room_type: RoomType,
) -> EffectResult {
    if !run_owns(rs, RelicId::LavaRock) {
        return EffectResult::NotImplemented;
    }
    if room_type != RoomType::Boss {
        return EffectResult::NotImplemented;
    }
    if rs.act_index != 1 {
        return EffectResult::NotImplemented;
    }
    let already_fired = rs
        .relic_counters
        .get(&RelicId::LavaRock)
        .copied()
        .unwrap_or(0)
        != 0;
    if already_fired {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters
        .insert(RelicId::LavaRock, LAVA_ROCK_TRIGGERED);
    for _ in 0..LAVA_ROCK_RELICS {
        rewards.push(Reward::Relic(RelicReward::new(RelicId::LavaRock)));
    }
    EffectResult::Resolved
}

// ── MawBank (bead `sts2-rs-yb6rm`) ───────────────────────────────────

/// `MawBank.GOLD` — gold granted per room enter until the latch fires.
/// Python `MawBank.GOLD = 12` (`shop_event.py:2408`).
const MAW_BANK_GOLD: i32 = 12;

/// `relic_counters[MawBank]` value written by
/// [`listener_maw_bank_on_item_purchased`] once the player completes
/// the first non-zero-gold shop purchase. Mirrors Python
/// `MawBank._purchased = True`. The "still paying out" state is the
/// absent / 0 case; the room-enter listener stops granting gold once
/// the counter is non-zero.
const MAW_BANK_PURCHASED: i32 = 1;

/// `MawBank::after_room_entered` — grant
/// [`MAW_BANK_GOLD`] gold on every room enter while the `_purchased`
/// latch is open. Mirrors Python
/// `MawBank.after_room_entered` (`shop_event.py:2418-2420`): no
/// room-type filter — Python fires for every room, including Boss /
/// Treasure / Shop / RestSite. Gating happens purely through the
/// `relic_counters[MawBank]` latch written by
/// [`listener_maw_bank_on_item_purchased`].
fn listener_maw_bank_after_room_entered(rs: &mut RunState, _rt: RoomType) -> EffectResult {
    if !run_owns(rs, RelicId::MawBank) {
        return EffectResult::NotImplemented;
    }
    let latched = rs
        .relic_counters
        .get(&RelicId::MawBank)
        .copied()
        .unwrap_or(0)
        != 0;
    if latched {
        return EffectResult::NotImplemented;
    }
    rs.gain_gold(MAW_BANK_GOLD);
    EffectResult::Resolved
}

/// `MawBank::on_item_purchased` — close the latch once the player has
/// spent gold on any shop item. Mirrors Python
/// `MawBank.on_item_purchased` (`shop_event.py:2422-2432`): only the
/// `gold_spent > 0` branch latches, so the free auto-purchase path
/// (LordsParasol) leaves the relic active. Writes
/// `relic_counters[MawBank] = MAW_BANK_PURCHASED` exactly once;
/// subsequent purchases are idempotent.
fn listener_maw_bank_on_item_purchased(rs: &mut RunState, gold_spent: i32) -> EffectResult {
    if !run_owns(rs, RelicId::MawBank) {
        return EffectResult::NotImplemented;
    }
    if gold_spent <= 0 {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters
        .insert(RelicId::MawBank, MAW_BANK_PURCHASED);
    EffectResult::Resolved
}

// ── LavaLamp (bead `sts2-rs-67co3`) ──────────────────────────────────

/// `LavaLamp::after_room_entered` — clear the took-damage flag. C#
/// `LavaLamp.AfterRoomEntered` sets `TookDamageThisCombat = false` on
/// every room entry, no room-type filter (`LavaLamp.cs`, v0.107.1 and
/// v0.111.0). Without it the flag latched by one damaged fight rode the
/// `enter_combat` seed / `exit_combat` write-back forever, denying the
/// upgrade after every later clean fight (fuzz v0.111.0 seed 1159).
fn listener_lava_lamp_after_room_entered(rs: &mut RunState, _rt: RoomType) -> EffectResult {
    if !run_owns(rs, RelicId::LavaLamp) {
        return EffectResult::NotImplemented;
    }
    rs.relic_counters.insert(RelicId::LavaLamp, 0);
    EffectResult::Resolved
}

/// `LavaLamp::modify_card_reward_options_late` — upgrade every card
/// on the post-combat reward screen iff the player took no damage
/// during the just-ended combat.
///
/// Mirrors Python `LavaLamp.modify_card_reward_options_late`
/// (`shop_event.py:2458-2469`): on a combat-room reward (Python gate
/// `room.room_type in {MONSTER, ELITE, BOSS}`), if `_took_damage ==
/// False`, walk the offered cards and call
/// `owner.upgrade_card_instance(card)` on each one.
///
/// Rust port semantics:
/// - Ownership-gated on `RelicId::LavaLamp`.
/// - The Python `room.room_type` combat-room check collapses into
///   the flag semantics — the flag only sets during a combat, so
///   if it's clean after combat the reward is from a combat room
///   by construction.
/// - The flag itself lives at `rs.relic_counters[LavaLamp]`.
///   Absent / `0` ↔ "no damage taken"; `LAVA_LAMP_TOOK_DAMAGE`
///   (= 1) ↔ "damage taken". The combat→run flushback (bead
///   `sts2-rs-o62e7`) is fully wired: `run_bridge::enter_combat` seeds
///   the combat-side mirror (`Relic.counter = 0`),
///   `relic_effects::lava_lamp::on_after_damage_received_lava_lamp`
///   latches it to `1` on unblocked player damage, and
///   `run_bridge::exit_combat` (the `RunManager` `exit_combat` callback)
///   mirrors it back to this run-side slot. See the
///   `lava_lamp_full_chain_*` integration tests in `run_bridge`.
///
/// Bead `sts2-rs-67co3` (Wave 7); firing site `sts2-rs-o62e7`.
fn listener_lava_lamp_card_reward(rs: &mut RunState, cards: &mut Vec<Card>) -> EffectResult {
    if !run_owns(rs, RelicId::LavaLamp) {
        return EffectResult::NotImplemented;
    }
    // C# `if (!(base.Owner.RunState.CurrentRoom is CombatRoom)) return false;`
    // (`LavaLamp.cs:66`, both builds). The flag alone does not imply a combat
    // room: a no-damage fight leaves it clear for the shop / event / Neow
    // slates that follow (Orrery, Lost Coffer, event grants, the Gorge), which
    // C# never upgrades. An event FIGHT is a real C# `CombatRoom`: Rust keeps
    // the `?` room's `Event` type through it, but `finish_event_combat` tears
    // the event down (`current_event = None`) before rolling the room
    // rewards, so only a live event counts as a non-combat room. A room-less
    // RunState (unit fixtures) still passes. The act-opening Ancient is an
    // `EventRoom` too, though Rust keeps the boss room through it (Silent
    // v0.111.0 seed 5451: Sea Glass's slate after a clean boss fight).
    if rs.in_non_combat_room() {
        return EffectResult::NotImplemented;
    }
    let took_damage = rs
        .relic_counters
        .get(&RelicId::LavaLamp)
        .copied()
        .unwrap_or(0)
        != 0;
    if took_damage {
        return EffectResult::NotImplemented;
    }
    let mut upgraded_any = false;
    // GH #793: the run build's upgrade (v0.111.0 Hyperbeam +6, not +8).
    let version = rs.game_version;
    for card in cards.iter_mut() {
        if card.upgrade_for(version) {
            upgraded_any = true;
        }
    }
    if upgraded_any {
        EffectResult::Resolved
    } else {
        EffectResult::NotImplemented
    }
}

// ── PaelsTooth (bead `sts2-rs-v9zer`) ────────────────────────────────

/// Maximum number of cards PaelsTooth stows on pickup. Python
/// `PaelsTooth.CARDS = 5` (`shop_event.py:2781`).
const PAELS_TOOTH_CARDS: usize = 5;

/// C# `PaelsTooth.AfterObtained` selects via
/// `CardSelectCmd.FromDeckForRemoval(filter: c => c.IsUpgradable)`
/// (`PaelsTooth.cs:81`), so the pool is `IsRemovable && IsUpgradable` — the
/// SAME removability rule as every other consumer, no basic-card carve-out.
/// Until 2026-08-26 this hand-rolled a basics-excluding list mirroring the
/// legacy Python `Card.is_removable`, which withheld exactly the cards the
/// relic most wants (an unupgraded basic Strike is both removable and
/// upgradable, so C# offers it first).
fn is_paels_tooth_removable(card: &sts2_core::Card) -> bool {
    crate::cards::spec_index::card_is_removable_instance(card)
}

/// `PaelsTooth::after_obtained` (auto-pick fallback path) —
/// stow up to 5 removable+upgradable master-deck cards into
/// `rs.paels_tooth_stored` and drop them from the deck. Self-gated
/// on `last_obtained_relic == Some(PaelsTooth)` so the listener
/// fires exactly once on pickup.
///
/// Python pin: `shop_event.py:2797-2810`. Python's preferred path
/// is an interactive `request_deck_choice` (player picks which
/// cards to feed the tooth); the auto-pick branch fires on the
/// no-interactive-handler / fall-through case. This Rust port
/// implements the auto-pick path — the interactive variant is
/// deferred to a follow-up sub-bead since it needs a new
/// `ChoiceKind::DeckRemoveAndStore` flavor or
/// `request_deck_choice`-style primitive.
///
/// Sort order matches Python: by `card_id` (the Python pin sorts
/// by `reference_card_entry(card.card_id)`, i.e. a stable
/// registry-order key; the Rust enum's `Ord` impl is a stable
/// proxy that matches Python's intent of "stable across runs").
///
/// Bead `sts2-rs-v9zer`.
fn listener_paels_tooth_after_obtained(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::PaelsTooth) {
        return EffectResult::NotImplemented;
    }
    // Candidate set: master-deck (idx, card_id, upgraded) tuples for
    // upgradable AND removable cards. `IsUpgradable`, not `!upgraded`: a
    // curse is never upgradable, so a Guilty must not count toward the
    // removals queued below (the select's own filter already skips it).
    let mut candidates: Vec<(usize, sts2_core::cards::CardId, bool)> = rs
        .master_deck
        .iter()
        .enumerate()
        .filter_map(|(idx, c)| {
            (c.is_upgradable() && is_paels_tooth_removable(c))
                .then_some((idx, c.card_id, c.upgraded))
        })
        .collect();
    // Sort by card_id for parity with Python's `reference_card_entry`
    // stable-order key.
    candidates.sort_by_key(|&(_idx, card_id, _upg)| card_id);
    let take = candidates.len().min(PAELS_TOOTH_CARDS);
    if take == 0 {
        return EffectResult::Resolved;
    }
    let max_count = u8::try_from(take).unwrap_or(u8::MAX);

    // Interactive hosts: C# `PaelsTooth.AfterObtained` (`PaelsTooth.cs:68-78`,
    // same in v0.111.0) is a mandatory `FromDeckForRemoval(prefs(5), filter:
    // IsUpgradable)` card select, then stashes the picks in `Id.Entry` order.
    // Route it through the interactive removal queue (a `card_select` the run
    // loop surfaces) instead of the run-level `pending_choice`, which nothing
    // but the rest-site Smith ever surfaces — the stash prompt was stranded
    // and the act23 lockstep saw no select at all (v0.111.0 Defect:1018,
    // Pael → Pael's Tooth).
    if rs.interactive_card_removal {
        rs.pending_removal_paels_tooth = true;
        rs.pending_deck_removals += u32::from(max_count);
        return EffectResult::Resolved;
    }

    // Bead `sts2-rs-n7tha`: try interactive deck-choice first. When
    // `candidates.len() > max_count` the agent has real discretion
    // (pick N of M, where M > N); open a `PaelsToothStash` prompt and
    // bail without mutating master_deck. The agent resolves via
    // `RunState::resolve_paels_tooth_stash`.
    //
    // When `candidates.len() <= max_count` the "choice" is forced
    // (must pick them all), so `open_paels_tooth_stash_choice`
    // returns false and we fall through to the auto-pick path.
    if rs.open_paels_tooth_stash_choice(&candidates, max_count) {
        return EffectResult::Resolved;
    }

    // Auto-pick fallback (Python `shop_event.py:2810`): take the
    // first N candidates and stash them directly.
    let chosen_indices: Vec<usize> = candidates
        .into_iter()
        .take(take)
        .map(|(idx, _, _)| idx)
        .collect();
    for &idx in &chosen_indices {
        rs.paels_tooth_stored.push(rs.master_deck[idx]);
    }
    let mut to_remove = chosen_indices;
    to_remove.sort_unstable();
    for idx in to_remove.into_iter().rev() {
        rs.master_deck.remove(idx);
    }
    EffectResult::Resolved
}

/// `PaelsTooth::after_combat_end` — draw one card from
/// `paels_tooth_stored` via the `rng.rewards` stream, upgrade it,
/// and push it back onto the master deck. Ownership-gated.
///
/// C# `PaelsTooth.AfterCombatEnd` (`PaelsTooth.cs:93-114`): one
/// `PlayerRng.Rewards.NextItem` draw, then `List.Remove` of the first
/// equal stash entry.
fn listener_paels_tooth_after_combat_end(rs: &mut RunState) -> EffectResult {
    if !run_owns(rs, RelicId::PaelsTooth) {
        return EffectResult::NotImplemented;
    }
    if rs.paels_tooth_stored.is_empty() {
        return EffectResult::NotImplemented;
    }
    let len = rs.paels_tooth_stored.len();
    // `next_int_exclusive(0, len)` returns `[0, len)`. Storage capped
    // at PAELS_TOOTH_CARDS so the i32 cast is safe.
    let idx = rs.rng.rewards.next_int_exclusive(0, len as i32) as usize;
    let mut card = rs.paels_tooth_stored[idx];
    // C# `SerializableCards.Remove(serializableCard)` (`PaelsTooth.cs:111`)
    // is `List.Remove` under `SerializableCard.Equals` (Id, upgrade level,
    // enchantment Id + Amount: `SerializableCard.cs:73-89`), so it drops the
    // FIRST equal entry, which need not be the drawn one.
    let first_equal = rs
        .paels_tooth_stored
        .iter()
        .position(|c| {
            c.card_id == card.card_id
                && c.upgraded == card.upgraded
                && c.enchantments == card.enchantments
        })
        .unwrap_or(idx);
    rs.paels_tooth_stored.remove(first_equal);
    // GH #793: the run build's upgrade delta.
    let _ = card.upgrade_for(rs.game_version);
    rs.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::RelicGain);
    EffectResult::Resolved
}

/// `WongosMysteryTicket::after_combat_end` — `_combats_finished += 1`
/// while `_gave_relic == False` (`shop_event.py:3074-3076`). Once the
/// payout has happened the counter latches at
/// [`WONGOS_GAVE_SENTINEL`] and stops incrementing. Ownership-gated.
/// Note: the firing site (combat→run back-edge) is not yet wired;
/// this listener is currently exercised only by tests + the future
/// firing-site flip in a follow-up bead. Bead `sts2-rs-9wx7`.
fn listener_wongos_mystery_ticket_combat_end(rs: &mut RunState) -> EffectResult {
    if !run_owns(rs, RelicId::WongosMysteryTicket) {
        return EffectResult::NotImplemented;
    }
    let counter = rs
        .relic_counters
        .entry(RelicId::WongosMysteryTicket)
        .or_insert(0);
    if *counter < 0 {
        // already paid out; latch.
        return EffectResult::NotImplemented;
    }
    *counter += 1;
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Listener bodies — Wave 5 (sts2-rs-jvk3)
// ---------------------------------------------------------------------------

/// `SwordOfStone::after_combat_victory` — when the combat was an
/// elite encounter, decrement the per-instance elite-defeat counter
/// (Python `shop_event.py::SwordOfStone`, `ELITES_NEEDED=5`,
/// counter `_elites_defeated` here stored as the down-counting
/// `Relic.counter`). When the counter reaches `0`, Python invokes
/// `owner.transform_relic(self, RelicId.SWORD_OF_JADE)`.
///
/// The counter half lands here against the combat-side
/// `after_combat_victory` channel (it takes `&mut CombatState` only);
/// the exit callback flushes the count into `RunState.relic_counters`,
/// and [`listener_sword_of_stone_run`] — fired by
/// `RunManager::exit_combat_and_transition` — replaces the relic once
/// that count reaches 0.
///
/// Counter seed convention (matches `EmberTea`): callers init
/// `Relic { counter: 5, .. }` at obtain time. Ownership-gated.
fn listener_sword_of_stone(state: &mut CombatState) -> EffectResult {
    if !state.is_elite() {
        return EffectResult::NotImplemented;
    }
    let mut fired = false;
    for relic in &mut state.relics {
        if relic.is_melted {
            continue;
        }
        if relic.kind == RelicKind::SwordOfStone && relic.counter > 0 {
            relic.counter -= 1;
            fired = true;
            break;
        }
    }
    if !fired {
        return EffectResult::NotImplemented;
    }
    // Counter-decrement half lands here on the CombatState mirror; the
    // transform-on-zero arm fires on the parallel run-side channel via
    // [`listener_sword_of_stone_run`] (bead `sts2-rs-y8zi4`), which
    // owns the `RunState.transform_relic(SwordOfStone -> SwordOfJade)`
    // call; the run manager fires it at the combat's exit, after the
    // exit callback flushed this count.
    EffectResult::Resolved
}

/// `SwordOfStone::after_combat_victory` — run-side transform arm. C#
/// `SwordOfStone.AfterCombatVictory` (`SwordOfStone.cs:40-51`) counts an
/// elite and, at the fifth, `RelicCmd.Replace`s itself with Sword of Jade.
///
/// The count is the combat side's: [`listener_sword_of_stone`] decrements the
/// combat relic's down-counter (seeded 5) on each elite victory, and the exit
/// callback flushes it into `RunState.relic_counters`. This run-side arm
/// fires after that flush, from `RunManager::exit_combat_and_transition`,
/// and only reads the count: at 0 it replaces the relic
/// (decrementing here too would count every elite twice). Idempotent: once
/// Sword of Stone is gone the ownership check fails fast.
fn listener_sword_of_stone_run(rs: &mut RunState, ctx: &CombatVictoryRunCtx) -> EffectResult {
    if !ctx.is_elite {
        return EffectResult::NotImplemented;
    }
    if !run_owns(rs, RelicId::SwordOfStone) {
        return EffectResult::NotImplemented;
    }
    let counter = rs
        .relic_counters
        .get(&RelicId::SwordOfStone)
        .copied()
        .unwrap_or(5);
    if counter > 0 {
        return EffectResult::NotImplemented;
    }
    let _ = rs.transform_relic(RelicId::SwordOfStone, RelicId::SwordOfJade);
    EffectResult::Resolved
}

/// `WarHammer::after_combat_victory` — on elite victory, upgrade 4
/// random non-upgraded deck cards. Mirrors Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::WarHammer`
/// (`CARDS=4`):
///
/// ```python
/// def after_combat_victory(self, owner, combat):
///     if not combat.is_elite:
///         return
///     owner.upgrade_random_cards(None, self.CARDS, rng=...niche)
/// ```
///
/// Runs on the run-side `after_combat_victory` channel because the
/// upgrade target is `RunState.master_deck`. The channel fires from
/// `RunManager::exit_combat_and_transition` after a won combat
/// (C# `WarHammer.cs:19-31`).
///
/// WarHammer has no combat-side listener — the real logic lives here
/// on the run-side channel.
///
/// Bead `sts2-rs-mg5ur`.
fn listener_war_hammer_run(rs: &mut RunState, ctx: &CombatVictoryRunCtx) -> EffectResult {
    if !ctx.is_elite {
        return EffectResult::NotImplemented;
    }
    if !run_owns(rs, RelicId::WarHammer) {
        return EffectResult::NotImplemented;
    }
    const CARDS: usize = 4;
    // C# `WarHammer.cs`: `Where(IsUpgradable).StableShuffle(Rng.Niche).Take(4)`
    // → `CardCmd.Upgrade` — random, no player pick, so use the always-immediate
    // niche-shuffle primitive (NOT the interactive-aware `upgrade_random_cards`,
    // which would queue a card_select under `interactive_card_upgrade`).
    rs.upgrade_random_cards_stable(CARDS);
    EffectResult::Resolved
}

/// `FragrantMushroom::after_obtained` — Python
/// `shop_event.py::FragrantMushroom` (`DAMAGE=15`, `CARDS=3`):
/// `owner.take_damage(15)` plus `owner.upgrade_random_cards(None, 3)`
/// (when the run's deferred-followup-rewards path is off; the
/// alternative `offer_upgrade_cards_reward` path is queued via the
/// reward system and is part of the deferred-rewards-queue plumbing
/// not landed at this pin).
///
/// Both legs land at this pin: the take-damage leg routes through
/// [`RunState::take_damage`], and the upgrade leg routes through the
/// real-body [`RunState::upgrade_random_cards`] (bead
/// `sts2-rs-7krx0`). The reward-queue alternative path
/// (Python `defer_followup_rewards`) is intentionally skipped — no
/// such flag exists on Rust [`RunState`] yet.
///
/// Bead `sts2-rs-jvk3` (initial surface) + `sts2-rs-7krx0` (body) +
/// `sts2-rs-ombcw` (integration test).
fn listener_fragrant_mushroom(rs: &mut RunState) -> EffectResult {
    if rs.last_obtained_relic != Some(RelicId::FragrantMushroom) {
        return EffectResult::NotImplemented;
    }
    rs.take_damage(15);
    // C# `FragrantMushroom.cs:29-36`:
    // `Where(IsUpgradable).StableShuffle(Rng.Niche).Take(Cards)` →
    // `CardCmd.Upgrade` — random, no player pick (use the always-immediate
    // niche primitive). `Cards` is `CardsVar(2)` (`FragrantMushroom.cs:26`,
    // same in v0.107.1 and v0.111.0); the Python pin's `CARDS = 3` upgraded
    // one card too many.
    let _ = rs.upgrade_random_cards_stable(2);
    EffectResult::Resolved
}

// ---------------------------------------------------------------------------
// Interleaved typed-channel listener bodies
// ---------------------------------------------------------------------------

/// `BingBong::on_card_added_to_deck` — clone-and-re-add the card
/// just appended to the master deck. Mirrors Python
/// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py::BingBong`:
///
/// ```python
/// def on_card_added_to_deck(self, owner, card, source=None):
///     if source is None:
///         owner.add_card_instance_to_deck(owner.clone_card_for_deck(card), source=self)
/// ```
///
/// The re-entry guard: Python passes `source=self` on the clone-add,
/// which keeps the next dispatch from re-cloning. The Rust port
/// encodes the same idea via [`AddSource::BingBongDuplicate`] — the
/// duplicate path that goes through
/// [`CombatState::duplicate_last_added_card`] (bead `sts2-rs-anx`)
/// stamps that source on the re-dispatch, and this listener bails
/// when it sees it.
///
/// Combat-side firing site is live via
/// [`CombatState::add_to_master_deck`]; RunState-side
/// master_deck propagation is deferred (same back-edge story as
/// SwordOfStone/WarHammer). Bead `sts2-rs-hyv0p`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn listener_bing_bong(state: &mut CombatState, ctx: &CardAddedCtx) -> EffectResult {
    if ctx.source == sts2_core::relic_effects::AddSource::BingBongDuplicate {
        return EffectResult::NotImplemented;
    }
    if !player_owns(state, RelicKind::BingBong) {
        return EffectResult::NotImplemented;
    }
    if !state.duplicate_last_added_card() {
        return EffectResult::NotImplemented;
    }
    EffectResult::Resolved
}

// --- Fake-relic family ----------------------------------------------------
//
// Twelve "Fake X" relics from `sts2_env/relics/shop_event.py` — granted
// by the Mind Bloom / Cursed Tome / Counterfeit events. **2026-05-13
// audit correction:** the Python bodies are plain (smaller) versions of
// the real relics they mimic, not punitive inversions of the useful
// clause. Each `Fake*` class subclasses the corresponding real relic's
// `RelicInstance` and overrides a single hook (`after_…`) with a
// reduced-magnitude effect (e.g. `FakeBloodVial` heals 1 HP on round 1
// where `BloodVial` heals more on obtain). None carry per-instance
// state; the marker handler below is the entire wiring surface this
// bead exposes until the umbrella retrofit `sts2-rs-01o` lands real
// bodies. See `docs/reality-checks/2026-05-12-event-reward-audit.md`
// for the per-relic Python-pin classification.

// --- Remaining event-reward relics ---------------------------------------

/// `PollinousCore.EXTRA_CARDS` — extra hand-draw on fire
/// (`shop_event.py::PollinousCore`).
pub const POLLINOUS_CORE_EXTRA_CARDS: i32 = 2;
/// `PollinousCore.TURNS` — fire period in player-turns
/// (`shop_event.py::PollinousCore`).
pub const POLLINOUS_CORE_TURNS: i32 = 4;

/// v0.107.1 `PollinousCore.BeforeSideTurnStart` (PollinousCore.cs:90-97):
/// `TurnsSeen++` on every turn start the owner takes part in, with no wrap.
/// Only `AfterModifyingHandDraw` resets it to 0, and only on a turn whose
/// `ModifyHandDraw` it raised (`TurnsSeen == 4`); Rust runs that reset after
/// the turn-start `BeforeHandDraw` walk
/// (`sts2_core::relic_effects::pendulum::pollinous_core_after_modifying_hand_draw_v0107`).
/// A combat that ends after the tick to 4 but before the hand draw leaves
/// 4, and the relic never fires again (loadout Defect 5189 v0.107.1). The
/// count persists across combats on `Relic.counter`.
fn listener_pollinous_core_tick(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    // v0.111.0 counts in `BeforeHandDraw` instead (see below).
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    let Some(idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == RelicKind::PollinousCore)
    else {
        return EffectResult::NotImplemented;
    };
    state.relics[idx].counter = state.relics[idx].counter.saturating_add(1);
    EffectResult::Resolved
}

/// v0.111.0 `PollinousCore.BeforeHandDraw` + `AfterModifyingHandDraw`
/// (decomp PollinousCore.cs:88-125): `TurnsSeen++`, and once the draw was
/// raised (`TurnsSeen >= Turns`, see the `modify_hand_draw` arm) back to 0.
fn listener_pollinous_core_before_hand_draw(state: &mut CombatState) -> EffectResult {
    if state.game_version < sts2_core::GameVersion::V0_111_0 {
        return EffectResult::NotImplemented;
    }
    let Some(relic) = state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == RelicKind::PollinousCore)
    else {
        return EffectResult::NotImplemented;
    };
    relic.counter = relic.counter.saturating_add(1);
    if relic.counter >= POLLINOUS_CORE_TURNS {
        relic.counter = 0;
    }
    EffectResult::Resolved
}

/// `PollinousCore::modify_hand_draw` — when the per-turn counter
/// equals `TURNS`, return `current_draw + EXTRA_CARDS`. Python pin:
///
/// ```python
/// def modify_hand_draw(self, owner, draw, combat):
///     if self._turns_seen == self.TURNS:
///         return draw + self.EXTRA_CARDS
///     return draw
/// ```
fn listener_pollinous_core_modify_hand_draw(state: &CombatState, current_draw: i32) -> i32 {
    let Some(relic) = state
        .relics
        .iter()
        .find(|r| !r.is_melted && r.kind == RelicKind::PollinousCore)
    else {
        return current_draw;
    };
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        // v0.111.0 (decomp PollinousCore.cs:88-118): `BeforeHandDraw` bumps
        // `TurnsSeen`, `ModifyHandDraw` adds 2 once `TurnsSeen >= Turns` (was
        // `!=`), `AfterModifyingHandDraw` resets it to 0. C# computes the
        // draw AFTER `BeforeHandDraw`; Rust folds `modify_hand_draw` just
        // before the relic `before_hand_draw` walk, so read the count this
        // turn's tick is about to make (`counter + 1`). Nothing between the
        // two touches the counter; the tick applies the bump and the reset.
        return if relic.counter.saturating_add(1) >= POLLINOUS_CORE_TURNS {
            current_draw + POLLINOUS_CORE_EXTRA_CARDS
        } else {
            current_draw
        };
    }
    if relic.counter == POLLINOUS_CORE_TURNS {
        current_draw + POLLINOUS_CORE_EXTRA_CARDS
    } else {
        current_draw
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The id list covers exactly 35 entries — the `RelicRarity.EVENT`
    /// count Python pin at
    /// `sts2-rl-agent@01c094d:sts2_env/relics/shop_event.py`
    /// (`grep -c 'rarity = RelicRarity.EVENT' shop_event.py == 35`).
    #[test]
    fn pool_has_thirty_five_entries() {
        assert_eq!(EVENT_REWARD_RELIC_IDS.len(), 35);
    }

    /// Every `RelicId` in the list is distinct. Catches accidental
    /// duplicate paste-edits inside [`EVENT_REWARD_RELIC_IDS`].
    #[test]
    fn pool_entries_are_unique_by_relic_id() {
        let mut ids: Vec<RelicId> = EVENT_REWARD_RELIC_IDS.to_vec();
        ids.sort();
        let before = ids.len();
        ids.dedup();
        assert_eq!(
            before,
            ids.len(),
            "duplicate RelicId in EVENT_REWARD_RELIC_IDS"
        );
    }

    /// Every factory produces a `RelicSpec` whose `name` round-trips
    /// through `RelicId::to_python_name`.
    #[test]
    fn every_factory_name_round_trips() {
        let pairs: &[(RelicSpec, RelicId)] = &[
            (big_mushroom(), RelicId::BigMushroom),
            (bing_bong(), RelicId::BingBong),
            (bone_tea(), RelicId::BoneTea),
            (byrdpip(), RelicId::Byrdpip),
            (chosen_cheese(), RelicId::ChosenCheese),
            (darkstone_periapt(), RelicId::DarkstonePeriapt),
            (daughter_of_the_wind(), RelicId::DaughterOfTheWind),
            (dream_catcher(), RelicId::DreamCatcher),
            (ember_tea(), RelicId::EmberTea),
            (fake_anchor(), RelicId::FakeAnchor),
            (fake_blood_vial(), RelicId::FakeBloodVial),
            (fake_happy_flower(), RelicId::FakeHappyFlower),
            (fake_lees_waffle(), RelicId::FakeLeesWaffle),
            (fake_mango(), RelicId::FakeMango),
            (fake_merchants_rug(), RelicId::FakeMerchantsRug),
            (fake_orichalcum(), RelicId::FakeOrichalcum),
            (fake_snecko_eye(), RelicId::FakeSneckoEye),
            (fake_strike_dummy(), RelicId::FakeStrikeDummy),
            (fake_venerable_tea_set(), RelicId::FakeVenerableTeaSet),
            (forgotten_soul(), RelicId::ForgottenSoul),
            (fragrant_mushroom(), RelicId::FragrantMushroom),
            (fresnel_lens(), RelicId::FresnelLens),
            (hand_drill(), RelicId::HandDrill),
            (history_course(), RelicId::HistoryCourse),
            (lost_wisp(), RelicId::LostWisp),
            (maw_bank(), RelicId::MawBank),
            (mr_struggles(), RelicId::MrStruggles),
            (pollinous_core(), RelicId::PollinousCore),
            (royal_poison(), RelicId::RoyalPoison),
            (sword_of_stone(), RelicId::SwordOfStone),
            (sword_of_jade(), RelicId::SwordOfJade),
            (tea_of_discourtesy(), RelicId::TeaOfDiscourtesy),
            (the_boot(), RelicId::TheBoot),
            (
                wongo_customer_appreciation_badge(),
                RelicId::WongoCustomerAppreciationBadge,
            ),
            (wongos_mystery_ticket(), RelicId::WongosMysteryTicket),
        ];
        assert_eq!(pairs.len(), 35);
        for (spec, id) in pairs {
            assert_eq!(spec.name, id.to_python_name(), "spec {spec:?} vs id {id:?}");
            assert_eq!(spec.rarity, RelicRarity::Event, "wrong rarity for {spec:?}");
            assert_eq!(spec.effect, RelicEffect::Noop);
        }
    }

    /// Every event `RelicId` carries an upper-snake-case Python name.
    /// Locks the parity-output naming for the future exporter.
    #[test]
    fn every_event_relic_has_python_name() {
        for &id in EVENT_REWARD_RELIC_IDS {
            let name = id.to_python_name();
            assert!(!name.is_empty(), "empty python_name for {id:?}");
            assert!(
                name.bytes()
                    .all(|b| b == b'_' || b.is_ascii_uppercase() || b.is_ascii_digit()),
                "non upper-snake-case python_name for {id:?}: {name:?}"
            );
        }
    }

    // --- Listener bodies (sts2-rs-01o wave 1) ----------------------------
    //
    // Each test installs the pool (which registers the listener),
    // attaches the relic to `state.relics` so the ownership gate fires,
    // drives the matching dispatch channel, and asserts the observable
    // mutation. The ownership gate negative-case is covered by a single
    // companion test per relic that omits the `state.relics.push(...)`.

    use std::num::NonZeroU32;
    use sts2_core::hooks::ValueProps;
    use sts2_core::relic_effects::{
        dispatch_after_card_played, dispatch_before_combat_start, dispatch_before_turn_end,
        dispatch_modify_hp_lost, CardEventCtx, ModifyHpLostCtx,
    };
    use sts2_core::{CardUid, Creature, CreatureUid, MonsterKind, Relic, Side};

    fn push_relic(state: &mut CombatState, kind: RelicKind) {
        let next = state
            .relics
            .iter()
            .map(|r| r.id.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        state.relics.push(Relic {
            id: NonZeroU32::new(next).expect("non-zero"),
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }

    fn init_player_uid(state: &mut CombatState) -> CreatureUid {
        // apply_power_to_creature gates on current_hp > 0, so wave 2
        // tests need the player to be alive before init.
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let uid = state.init_player_creature();
        state.sync_player_creature();
        uid
    }

    fn push_attack_card(state: &mut CombatState) -> CardUid {
        let next_raw = state
            .cards
            .iter()
            .map(|c| c.uid.0.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let uid = CardUid(NonZeroU32::new(next_raw).expect("non-zero"));
        state.cards.push(sts2_core::Card {
            uid,
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: sts2_core::cards::CardType::Attack,
            target_type: sts2_core::cards::TargetType::Enemy,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: sts2_core::cards::CardTags::one(sts2_core::cards::CardTag::Strike),
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        uid
    }

    fn push_skill_card(state: &mut CombatState) -> CardUid {
        let next_raw = state
            .cards
            .iter()
            .map(|c| c.uid.0.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let uid = CardUid(NonZeroU32::new(next_raw).expect("non-zero"));
        state.cards.push(sts2_core::Card {
            uid,
            card_id: sts2_core::cards::CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: sts2_core::cards::CardType::Skill,
            target_type: sts2_core::cards::TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 5,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        uid
    }

    fn ctx_for(card_uid: CardUid, state: &CombatState) -> CardEventCtx {
        let idx = card_uid.0.get() as usize - 1;
        let card = &state.cards[idx];
        CardEventCtx {
            card_id: card.card_id,
            card_uid: Some(card_uid),
            target: None,
            card_tags: card.tags,
            energy_spent: 0,
        }
    }

    fn push_cultist(state: &mut CombatState) -> CreatureUid {
        let next_raw = state
            .creatures
            .iter()
            .map(|c| c.id.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let id = NonZeroU32::new(next_raw).expect("non-zero");
        state.creatures.push(Creature {
            id,
            current_hp: 48,
            max_hp: 48,
            block: 0,
            powers: smallvec::smallvec![],
            species: Some(MonsterKind::Nibbit),
            intent: None,
            intent_history: sts2_core::IntentHistory::default(),
            monster_id: sts2_core::MonsterId("Nibbit"),
            ai: None,
        });
        CreatureUid(id)
    }

    #[test]
    fn listener_fake_anchor_grants_four_block_at_combat_start() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeAnchor);
        let _player_uid = init_player_uid(&mut state);
        assert_eq!(state.player.block, 0);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(state.player.block, 4);
    }

    #[test]
    fn listener_fake_anchor_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let _player_uid = init_player_uid(&mut state);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(state.player.block, 0);
    }

    /// Bead `sts2-rs-jo6sc`: C# FakeAnchor.cs is `BlockVar(4m,
    /// Unpowered)` — Frail must NOT fold the 4-block grant. Canonical
    /// of the three event-reward block relics (FakeOrichalcum +
    /// DaughterOfTheWind share the fix path via
    /// `gain_block_player_unpowered`).
    #[test]
    fn listener_fake_anchor_ignores_frail_per_csharp_unpowered_prop() {
        use sts2_core::{Power, PowerKind, PowerUid};
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeAnchor);
        let player_uid = init_player_uid(&mut state);
        // Install Frail(2) on the player creature mirror.
        let puid = PowerUid::new(1).expect("nz");
        state.powers.push(Power {
            id: puid.0,
            kind: PowerKind::Frail,
            amount: 2,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state
            .creatures
            .iter_mut()
            .find(|c| c.id == player_uid.0)
            .expect("player creature row")
            .powers
            .push(puid);
        let _ = dispatch_before_combat_start(&mut state);
        assert_eq!(
            state.player.block, 4,
            "Frail does not fold the 4-block grant (would be 3 if it folded)"
        );
    }

    #[test]
    fn listener_fake_orichalcum_grants_three_block_when_zero() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = init_player_uid(&mut state);
        // C# arms on `Block == 0` in the VeryEarly pass; model that snapshot.
        state.combat_log.block_zero_at_turn_end = true;
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert_eq!(state.player.block, 3);
    }

    #[test]
    fn listener_fake_orichalcum_is_noop_with_existing_block() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = init_player_uid(&mut state);
        // Player held block at turn-end start ⇒ VeryEarly snapshot is false.
        state.combat_log.block_zero_at_turn_end = false;
        state.player.block = 7;
        state.sync_player_creature();
        let _ = dispatch_before_turn_end(&mut state, Side::Player);
        assert_eq!(state.player.block, 7);
    }

    /// Regression (coverage seed 2353): a sibling `before_turn_end` relic
    /// (e.g. RippleBasin) that grants block earlier in the SAME pass raises
    /// live `player.block` above 0, but C# `FakeOrichalcum` latched its
    /// `Block == 0` decision in the VeryEarly pass — so it must STILL grant
    /// its 3. Reading the snapshot (not live block) preserves this.
    #[test]
    fn listener_fake_orichalcum_grants_despite_sibling_block_this_turn() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = init_player_uid(&mut state);
        // VeryEarly snapshot armed (block was 0 at turn-end start), but a
        // sibling grant already put 4 block on the player this pass.
        state.combat_log.block_zero_at_turn_end = true;
        state.player.block = 4;
        state.sync_player_creature();
        let _ = listener_fake_orichalcum(&mut state, Side::Player);
        assert_eq!(
            state.player.block, 7,
            "snapshot-armed grant must add 3 on top of the sibling's 4"
        );
    }

    #[test]
    fn listener_fake_orichalcum_is_noop_on_enemy_side() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = init_player_uid(&mut state);
        let _ = dispatch_before_turn_end(&mut state, Side::Enemy);
        assert_eq!(state.player.block, 0);
    }

    #[test]
    fn listener_daughter_of_the_wind_grants_block_on_attack() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::DaughterOfTheWind);
        let _ = init_player_uid(&mut state);
        let uid = push_attack_card(&mut state);
        let ctx = ctx_for(uid, &state);
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(state.player.block, 1);
    }

    #[test]
    fn listener_daughter_of_the_wind_is_noop_on_skill() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::DaughterOfTheWind);
        let _ = init_player_uid(&mut state);
        let uid = push_skill_card(&mut state);
        let ctx = ctx_for(uid, &state);
        let _ = dispatch_after_card_played(&mut state, &ctx);
        assert_eq!(state.player.block, 0);
    }

    #[test]
    fn listener_the_boot_floors_low_powered_damage() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::TheBoot);
        let player_uid = init_player_uid(&mut state);
        let target_uid = push_cultist(&mut state);
        let ctx = ModifyHpLostCtx {
            target: target_uid,
            dealer: Some(player_uid),
            props: ValueProps::MOVE,
        };
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 3), 5);
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 1), 5);
        // ≥ 5 unchanged; 0 unchanged (Python `0 < amount < 5`).
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 5), 5);
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 7), 7);
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 0), 0);
    }

    #[test]
    fn listener_the_boot_skips_unpowered_damage() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::TheBoot);
        let player_uid = init_player_uid(&mut state);
        let target_uid = push_cultist(&mut state);
        let ctx = ModifyHpLostCtx {
            target: target_uid,
            dealer: Some(player_uid),
            props: ValueProps::MOVE.union(ValueProps::UNPOWERED),
        };
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 3), 3);
    }

    #[test]
    fn listener_the_boot_skips_when_dealer_is_not_player() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::TheBoot);
        let _ = init_player_uid(&mut state);
        let target_uid = push_cultist(&mut state);
        let ctx = ModifyHpLostCtx {
            target: target_uid,
            dealer: Some(target_uid),
            props: ValueProps::MOVE,
        };
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 3), 3);
    }

    // --- Listener bodies (sts2-rs-46ms wave 2) ---------------------------

    use sts2_core::powers::PowerKind;
    use sts2_core::run_effects::dispatch_after_obtained;
    use sts2_core::runstate::RunState;

    #[test]
    fn listener_fake_mango_gains_three_max_hp_on_obtain() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        rs.relics.push(sts2_core::relics::RelicId::FakeMango);
        rs.last_obtained_relic = Some(RelicId::FakeMango);
        let _ = dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max + 3);
        assert_eq!(rs.current_hp, before_hp + 3);
    }

    #[test]
    fn listener_fake_mango_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let _ = dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max);
    }

    #[test]
    fn listener_fake_lees_waffle_heals_ten_percent_max_hp() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.current_hp = 50; // open up healing room (default max_hp == 80)
        rs.relics.push(sts2_core::relics::RelicId::FakeLeesWaffle);
        rs.last_obtained_relic = Some(RelicId::FakeLeesWaffle);
        let _ = dispatch_after_obtained(&mut rs);
        // 80 * 10 / 100 == 8 → current_hp 50 + 8 == 58.
        assert_eq!(rs.current_hp, 58);
        assert_eq!(rs.max_hp, 80);
    }

    #[test]
    fn listener_big_mushroom_gains_twenty_max_hp_on_obtain() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before_max = rs.max_hp;
        let before_hp = rs.current_hp;
        rs.relics.push(sts2_core::relics::RelicId::BigMushroom);
        rs.last_obtained_relic = Some(RelicId::BigMushroom);
        let _ = dispatch_after_obtained(&mut rs);
        assert_eq!(rs.max_hp, before_max + 20);
        assert_eq!(rs.current_hp, before_hp + 20);
    }

    #[test]
    fn listener_sword_of_jade_applies_strength_three_at_combat_start() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::SwordOfJade);
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        // Locate the Strength power on the player creature.
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        let strength: i32 = state.creatures[player_idx]
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == PowerKind::Strength)
            .map(|p| p.amount)
            .sum();
        assert_eq!(strength, 3);
    }

    #[test]
    fn listener_sword_of_jade_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        assert!(state.creatures[player_idx].powers.is_empty());
    }

    #[test]
    fn listener_fake_snecko_eye_applies_confused_one_at_combat_start() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeSneckoEye);
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        let confused: i32 = state.creatures[player_idx]
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == PowerKind::Confused)
            .map(|p| p.amount)
            .sum();
        assert_eq!(confused, 1);
    }

    #[test]
    fn listener_fake_strike_dummy_adds_one_on_strike_tagged_card() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeStrikeDummy);
        let _ = init_player_uid(&mut state);
        let strike_uid = push_attack_card(&mut state); // tags = Strike (helper above)
        let strike_id = state.cards[strike_uid.0.get() as usize - 1].card_id;
        let target_uid = push_cultist(&mut state);
        let ctx = ModifyDamageCtx {
            dealer: state.player.creature_uid,
            target: target_uid,
            base_damage: 6,
            props: ValueProps::MOVE,
            card: Some(strike_id),
            card_uid: None,
        };
        let delta = sts2_core::relic_effects::dispatch_modify_damage_additive(&state, &ctx);
        assert!(
            delta >= 1,
            "expected FakeStrikeDummy contribution >= 1, got {delta}"
        );
    }

    #[test]
    fn listener_fake_strike_dummy_is_zero_on_non_strike_card() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeStrikeDummy);
        let _ = init_player_uid(&mut state);
        let skill_uid = push_skill_card(&mut state); // tags = NONE
        let skill_id = state.cards[skill_uid.0.get() as usize - 1].card_id;
        let target_uid = push_cultist(&mut state);
        // Probe a freshly-allocated relic id that won't match other handlers
        // adding their own deltas — we want to assert FakeStrikeDummy itself
        // contributes zero. The dispatcher accumulates across all listeners,
        // so we just assert the total is unaffected by the non-Strike gate
        // by snapshotting twice (Strike vs Skill) and asserting the delta.
        let ctx_skill = ModifyDamageCtx {
            dealer: state.player.creature_uid,
            target: target_uid,
            base_damage: 6,
            props: ValueProps::MOVE,
            card: Some(skill_id),
            card_uid: None,
        };
        let skill_total =
            sts2_core::relic_effects::dispatch_modify_damage_additive(&state, &ctx_skill);
        // Push a Strike card to compare.
        let strike_uid = push_attack_card(&mut state);
        let strike_id = state.cards[strike_uid.0.get() as usize - 1].card_id;
        let ctx_strike = ModifyDamageCtx {
            dealer: state.player.creature_uid,
            target: target_uid,
            base_damage: 6,
            props: ValueProps::MOVE,
            card: Some(strike_id),
            card_uid: None,
        };
        let strike_total =
            sts2_core::relic_effects::dispatch_modify_damage_additive(&state, &ctx_strike);
        assert!(
            strike_total > skill_total,
            "FakeStrikeDummy should add +1 on Strike vs Skill; skill_total={skill_total}, strike_total={strike_total}",
        );
    }

    // --- Listener bodies (sts2-rs-3nis wave 3 — per-instance counter) ----

    /// Push a relic with an explicit per-instance `counter` value.
    /// Mirrors [`push_relic`] but for counter-bearing relics.
    fn push_relic_with_counter(state: &mut CombatState, kind: RelicKind, counter: i32) {
        let next = state
            .relics
            .iter()
            .map(|r| r.id.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        state.relics.push(Relic {
            id: NonZeroU32::new(next).expect("non-zero"),
            kind,
            counter,
            is_wax: false,
            is_melted: false,
        });
    }

    /// Per-instance `Relic.counter` POD lands as a `i32` field, defaults
    /// to `0`, and is mutable. Bead `sts2-rs-3nis` sentinel.
    #[test]
    fn relic_counter_pod_round_trips() {
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::EmberTea);
        let idx = state
            .relics
            .iter()
            .position(|r| r.kind == RelicKind::EmberTea)
            .expect("EmberTea relic present");
        assert_eq!(state.relics[idx].counter, 0, "default counter is 0");
        state.relics[idx].counter = 5;
        assert_eq!(state.relics[idx].counter, 5);
        state.relics[idx].counter -= 1;
        assert_eq!(state.relics[idx].counter, 4);
    }

    /// `EmberTea` applies Strength(2) on combat start and decrements the
    /// counter from 5 → 4. Mirrors Python `shop_event.py::EmberTea`.
    #[test]
    fn listener_ember_tea_applies_strength_two_and_decrements_counter() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::EmberTea, 5);
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        // Strength on player.
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        let strength: i32 = state.creatures[player_idx]
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == PowerKind::Strength)
            .map(|p| p.amount)
            .sum();
        assert_eq!(strength, 2);
        // Counter decremented.
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::EmberTea)
            .expect("EmberTea row");
        assert_eq!(relic.counter, 4);
    }

    /// `EmberTea` stops firing once the counter hits 0 — no Strength
    /// applied, counter unchanged.
    #[test]
    fn listener_ember_tea_is_noop_when_counter_zero() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::EmberTea, 0);
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        let strength: i32 = state.creatures[player_idx]
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == PowerKind::Strength)
            .map(|p| p.amount)
            .sum();
        assert_eq!(strength, 0);
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::EmberTea)
            .expect("EmberTea row");
        assert_eq!(relic.counter, 0);
    }

    // -----------------------------------------------------------------
    // BoneTea listener tests — bead `sts2-rs-eejzj`
    // -----------------------------------------------------------------

    /// `BoneTea` upgrades every card in hand on the player's round-1
    /// turn-start, then decrements the `_combats_left` counter 1 → 0.
    #[test]
    fn listener_bone_tea_upgrades_hand_and_decrements_counter() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 1);
        let _ = init_player_uid(&mut state);
        state.round = 1;
        let a = push_attack_card(&mut state);
        let b = push_skill_card(&mut state);
        state.player.hand.push(a);
        state.player.hand.push(b);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        for uid in [a, b] {
            let idx = uid.0.get() as usize - 1;
            assert!(
                state.cards[idx].upgraded,
                "every hand card must be upgraded"
            );
        }
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::BoneTea)
            .expect("BoneTea row");
        assert_eq!(relic.counter, 0, "counter decremented 1 → 0");
    }

    /// v0.111.0: BoneTea fires on `AfterPlayerTurnStart`, not on the side
    /// turn start (decomp BoneTea.cs:50).
    #[test]
    fn listener_bone_tea_upgrades_hand_and_decrements_counter_v0111() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState {
            game_version: sts2_core::GameVersion::V0_111_0,
            ..CombatState::default()
        };
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 1);
        let _ = init_player_uid(&mut state);
        state.round = 1;
        let a = push_attack_card(&mut state);
        state.player.hand.push(a);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        assert!(
            !state.cards[a.0.get() as usize - 1].upgraded,
            "side turn start no longer upgrades in v0.111.0"
        );
        let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
        assert!(state.cards[a.0.get() as usize - 1].upgraded);
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::BoneTea)
            .expect("BoneTea row");
        assert_eq!(relic.counter, 0, "counter decremented 1 -> 0");
    }

    /// v0.107.1 pin: the new `AfterPlayerTurnStart` arm stays inert.
    #[test]
    fn listener_bone_tea_player_turn_start_is_inert_v0107() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 1);
        let _ = init_player_uid(&mut state);
        state.round = 1;
        let a = push_attack_card(&mut state);
        state.player.hand.push(a);
        let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
        assert!(!state.cards[a.0.get() as usize - 1].upgraded);
    }

    /// `BoneTea` only fires on round 1 — a round-2 turn-start is inert.
    #[test]
    fn listener_bone_tea_is_noop_outside_round_one() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 1);
        let _ = init_player_uid(&mut state);
        state.round = 2;
        let a = push_attack_card(&mut state);
        state.player.hand.push(a);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        let idx = a.0.get() as usize - 1;
        assert!(!state.cards[idx].upgraded, "round != 1 must not upgrade");
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::BoneTea)
            .expect("BoneTea row");
        assert_eq!(relic.counter, 1, "counter unchanged outside round 1");
    }

    /// `BoneTea` stops firing once the counter hits 0 (already used up).
    #[test]
    fn listener_bone_tea_is_noop_when_used_up() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 0);
        let _ = init_player_uid(&mut state);
        state.round = 1;
        let a = push_attack_card(&mut state);
        state.player.hand.push(a);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        let idx = a.0.get() as usize - 1;
        assert!(!state.cards[idx].upgraded, "used-up BoneTea must not fire");
    }

    /// `BoneTea` is inert on the enemy side-turn-start.
    #[test]
    fn listener_bone_tea_is_noop_on_enemy_side() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::BoneTea, 1);
        let _ = init_player_uid(&mut state);
        state.round = 1;
        let a = push_attack_card(&mut state);
        state.player.hand.push(a);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Enemy);
        let idx = a.0.get() as usize - 1;
        assert!(!state.cards[idx].upgraded, "enemy side must not fire");
    }

    // -----------------------------------------------------------------
    // FakeVenerableTeaSet listener tests — bead `sts2-rs-aqals`
    // -----------------------------------------------------------------

    /// Run-side: rest-site entry latches the counter to 1.
    #[test]
    fn listener_fake_venerable_tea_set_latches_on_rest_site_entry() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::FakeVenerableTeaSet);
        let _ = sts2_core::run_effects::dispatch_after_room_entered(&mut rs, RoomType::RestSite);
        assert_eq!(
            rs.relic_counters.get(&RelicId::FakeVenerableTeaSet),
            Some(&1),
            "rest-site entry latches counter to 1"
        );
    }

    /// Run-side: non-rest-site rooms never latch; ownership-gated.
    #[test]
    fn listener_fake_venerable_tea_set_rest_latch_gates() {
        crate::relics::dispatch::ensure_installed();
        // Non-rest-site room with ownership: no latch.
        let mut rs = RunState::new();
        rs.relics.push(RelicId::FakeVenerableTeaSet);
        let _ = sts2_core::run_effects::dispatch_after_room_entered(&mut rs, RoomType::Monster);
        assert!(!rs
            .relic_counters
            .contains_key(&RelicId::FakeVenerableTeaSet));
        // Rest site without ownership: no latch.
        let mut rs2 = RunState::new();
        let _ = sts2_core::run_effects::dispatch_after_room_entered(&mut rs2, RoomType::RestSite);
        assert!(!rs2
            .relic_counters
            .contains_key(&RelicId::FakeVenerableTeaSet));
    }

    /// Combat-side: a latched counter (1) grants `ENERGY = 1` on the
    /// first energy reset and clears the latch.
    #[test]
    fn listener_fake_venerable_tea_set_consumes_latch_and_grants_energy() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::FakeVenerableTeaSet, 1);
        let _ = init_player_uid(&mut state);
        let before = state.player.energy;
        let _ = sts2_core::relic_effects::dispatch_after_energy_reset(&mut state);
        assert_eq!(
            state.player.energy,
            before + FAKE_VENERABLE_TEA_SET_ENERGY,
            "+1 energy on latched consume"
        );
        let idx = state
            .relics
            .iter()
            .position(|r| r.kind == RelicKind::FakeVenerableTeaSet)
            .expect("relic present");
        assert_eq!(state.relics[idx].counter, 0, "latch clears after consume");
    }

    /// Combat-side: an unlatched counter (0) grants no energy.
    #[test]
    fn listener_fake_venerable_tea_set_is_noop_when_unlatched() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::FakeVenerableTeaSet, 0);
        let _ = init_player_uid(&mut state);
        let before = state.player.energy;
        let _ = sts2_core::relic_effects::dispatch_after_energy_reset(&mut state);
        assert_eq!(
            state.player.energy, before,
            "unlatched combat grants no energy"
        );
    }

    // -----------------------------------------------------------------
    // Byrdpip listener tests — bead `sts2-rs-4x616`
    // -----------------------------------------------------------------

    /// `Byrdpip` summons a behaviourless Byrdpip pet at combat start
    /// (idempotent — a second fire reuses the live pet).
    #[test]
    fn listener_byrdpip_summons_pet_idempotently() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::Byrdpip);
        let _ = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        let pet_count = |s: &CombatState| {
            s.creatures
                .iter()
                .filter(|c| c.species == Some(sts2_core::MonsterKind::Byrdpip) && c.current_hp > 0)
                .count()
        };
        assert_eq!(pet_count(&state), 1, "one Byrdpip pet spawned");
        // Pet is NOT an enemy — it never takes a turn / is never targeted.
        let pet = state
            .creatures
            .iter()
            .find(|c| c.species == Some(sts2_core::MonsterKind::Byrdpip))
            .expect("pet present");
        assert!(
            !state.enemy_uids.iter().any(|u| u.0 == pet.id),
            "pet must not be registered as an enemy"
        );
        // Idempotent: a second combat-start does not stack a duplicate.
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        assert_eq!(pet_count(&state), 1, "no duplicate pet on re-summon");
    }

    /// `Byrdpip::after_obtained` transforms every `ByrdonisEgg` in the
    /// master deck into a `ByrdSwoop`.
    #[test]
    fn listener_byrdpip_transforms_eggs_to_swoops() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::Byrdpip);
        rs.last_obtained_relic = Some(RelicId::Byrdpip);
        // Two eggs + an unrelated card.
        for id in [
            CardId::ByrdonisEgg,
            CardId::StrikeIronclad,
            CardId::ByrdonisEgg,
        ] {
            rs.master_deck
                .push(crate::cards::spec_index::card_pod_from_id(id, false).expect("pod exists"));
        }
        rs.last_obtained_relic = Some(RelicId::Byrdpip);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        let eggs = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::ByrdonisEgg)
            .count();
        let swoops = rs
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::ByrdSwoop)
            .count();
        assert_eq!(eggs, 0, "all ByrdonisEgg transformed away");
        assert_eq!(swoops, 2, "two ByrdSwoop created");
        assert_eq!(rs.master_deck.len(), 3, "deck size preserved");
        // CardCmd.Transform appends deck replacements (CardCmd.cs:427-436).
        assert_eq!(rs.master_deck[0].card_id, CardId::StrikeIronclad);
    }

    /// Bead `sts2-rs-RP24`: selecting the `Hatch` rest-site option drives
    /// the full egg→`ByrdSwoop` transform end-to-end — `do_rest_site_action`
    /// obtains Byrdpip, whose `after_obtained` listener (registered via
    /// `ensure_installed`) transforms the deck's eggs.
    #[test]
    fn hatch_rest_site_option_transforms_eggs_via_byrdpip() {
        use sts2_core::cards::CardId;
        use sts2_core::runstate::{RestSiteActionOutcome, RestSiteOption};
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck
            .push(crate::cards::spec_index::card_pod_from_id(CardId::ByrdonisEgg, false).unwrap());
        // The egg surfaces Hatch in the option list.
        assert!(rs
            .rest_site_options()
            .iter()
            .any(|o| matches!(o, RestSiteOption::Hatch)));
        let outcome = rs.do_rest_site_action(RestSiteOption::Hatch);
        assert_eq!(outcome, RestSiteActionOutcome::Hatched);
        assert!(rs.relics.contains(&RelicId::Byrdpip), "Byrdpip obtained");
        assert_eq!(
            rs.master_deck
                .iter()
                .filter(|c| c.card_id == CardId::ByrdonisEgg)
                .count(),
            0,
            "egg transformed away"
        );
        assert_eq!(
            rs.master_deck
                .iter()
                .filter(|c| c.card_id == CardId::ByrdSwoop)
                .count(),
            1,
            "one ByrdSwoop created"
        );
    }

    // -----------------------------------------------------------------
    // HistoryCourse listener tests — bead `sts2-rs-ummci`
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn history_course_setup(state: &mut CombatState, round: u16) -> CreatureUid {
        push_relic(state, RelicKind::HistoryCourse);
        let _ = init_player_uid(state);
        let enemy = push_cultist(state);
        state.enemy_uids.push(enemy);
        state.round = round;
        enemy
    }

    #[cfg(test)]
    fn enemy_hp(state: &CombatState, uid: CreatureUid) -> i32 {
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .map_or(0, |c| c.current_hp)
    }

    /// After round 1, HistoryCourse auto-plays a free dupe of last round's
    /// last Attack/Skill (Strike → 6 damage to the enemy).
    #[test]
    fn listener_history_course_replays_last_attack_after_round_one() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 2);
        let before = enemy_hp(&state, enemy);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            before - enemy_hp(&state, enemy),
            6,
            "Strike dupe dealt 6 damage to the enemy"
        );
    }

    /// A HistoryCourse dupe auto-played under Hex is Ethereal when its play
    /// finishes (`HexPower.AfterCardEnteredCombat` afflicts it Hexed, and
    /// `CardPlayFinishedEntry.WasEthereal` reads the live keywords), so
    /// Banshee's Cry in the draw pile drops by 2 (`BansheesCry.AfterCardPlayed`).
    /// The auto-play tally read only the static keyword (loadout Necrobinder
    /// 1067: a Hexed Sovereign Blade dupe, Banshee's Cry cs=7 rs=9).
    #[test]
    fn listener_history_course_hexed_dupe_discounts_banshees_cry() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            sts2_core::PowerKind::Hex,
            2,
        );
        let mut banshee = crate::cards::spec_index::card_pod_from_id(CardId::BansheesCry, false)
            .expect("banshee pod");
        let buid = sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
            .expect("nz");
        banshee.uid = buid;
        state.cards.push(banshee);
        state.player.draw.push(buid);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(state.combat_log.ethereal_cards_played_this_combat, 1);
        assert_eq!(sts2_core::hooks::effective_card_cost(&state, buid), 7);
    }

    /// The HistoryCourse dupe records a `CardPlayFinished` in C#
    /// (`CardModel.cs:1523`), so it must advance the combat-wide
    /// `player_card_plays_this_combat` counter that `GoldAxe.CalculatedDamage`
    /// scales on. The minimal dupe path previously skipped this bump, so a
    /// GoldAxe deck carrying HistoryCourse under-scaled by one per dupe (sweep
    /// seed 2761). One dupe ⇒ counter advances by exactly 1.
    #[test]
    fn listener_history_course_dupe_counts_as_a_card_play() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let before = state.combat_log.player_card_plays_this_combat;
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            state.combat_log.player_card_plays_this_combat,
            before + 1,
            "HistoryCourse dupe counts as one finished card play"
        );
    }

    /// A HiddenGem Replay-3 Scavenge dupe plays 4 times, and each play's
    /// exhaust pick prompts before the next replay opens its own (C# awaits
    /// every `OnPlay`, CardModel.cs:1896-1965). Before, each replay overwrote
    /// the open pick and only one card was exhausted (Defect full parity seed
    /// 2076); later the first three were drained inline (GH #1131).
    #[test]
    fn listener_history_course_replayed_select_dupe_resolves_each_pick() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let push = |state: &mut CombatState, id: CardId| {
            let mut c = crate::cards::spec_index::card_pod_from_id(id, false).unwrap();
            let uid =
                sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).unwrap()).unwrap();
            c.uid = uid;
            state.cards.push(c);
            uid
        };
        let scavenge = push(&mut state, CardId::Scavenge);
        state.cards[scavenge.0.get() as usize - 1].base_replay_count = 3;
        state.player.discard.push(scavenge);
        for _ in 0..5 {
            let s = push(&mut state, CardId::StrikeIronclad);
            state.player.hand.push(s);
        }
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::Scavenge, false, Some(scavenge)));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        for picked in 0..4 {
            assert!(
                state.pending_choice.is_some(),
                "pick {} prompts",
                picked + 1
            );
            assert_eq!(state.player.exhaust.len(), picked);
            let _ = sts2_core::apply(
                &mut state,
                sts2_core::Action::ResolveChoice {
                    selection: smallvec::smallvec![0],
                },
            );
        }
        assert!(state.pending_choice.is_none());
        assert!(state.pending_autoplay_resume.is_none());
        assert_eq!(state.player.exhaust.len(), 4);
        assert_eq!(state.player.hand.len(), 1);
        assert!(state.player.play.is_empty(), "the dupe is gone");
    }

    /// The dupe's finished play on THIS turn, in the build's per-turn tally.
    #[cfg(test)]
    fn history_course_dupe_turn_tally(version: sts2_core::GameVersion) -> u32 {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState {
            game_version: version,
            ..CombatState::default()
        };
        let _enemy = history_course_setup(&mut state, 2);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        state.combat_log.card_plays_finished_this_turn
    }

    /// v0.107.1 counts the dupe too: `OnPlayWrapper` records its
    /// `CardPlayFinishedEntry` with no `IsDupe` filter, and PaleBlueDot reads
    /// last turn's entries (decomp v0.107.1 `PaleBlueDotPower.cs:31`).
    #[test]
    fn listener_history_course_dupe_turn_tally_v0107() {
        assert_eq!(
            history_course_dupe_turn_tally(sts2_core::GameVersion::V0_107_1),
            1
        );
    }

    /// v0.111.0: the dupe is a `CardPlayFinished` this turn, which
    /// PaleBlueDotPower counts (decomp PaleBlueDotPower.cs:27). Parity fuzz
    /// Regent seed 537: the dupe + 4 hand plays = the 5th play, so C# applied
    /// DrawCardsNextTurn on PaleBlueDot's own play (GH #798).
    #[test]
    fn listener_history_course_dupe_turn_tally_v0111() {
        assert_eq!(
            history_course_dupe_turn_tally(sts2_core::GameVersion::V0_111_0),
            1
        );
    }

    /// Bead `sts2-rs-uh2mt`: a HistoryCourse dupe of an X-cost card (Whirlwind)
    /// captures the player's CURRENT energy as X — Whirlwind deals 5 × X to all
    /// enemies. The base-stat dupe had `energy_spent = 0` and dealt 0 (coverage
    /// seed 4708: LivingShield/Turret took 0 instead of 15). With 3 energy the
    /// dupe must deal 5 × 3 = 15.
    #[test]
    fn listener_history_course_xcost_dupe_captures_current_energy() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 2);
        state.player.energy = 3;
        let before = enemy_hp(&state, enemy);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::Whirlwind, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            before - enemy_hp(&state, enemy),
            15,
            "Whirlwind dupe deals 5 x 3 energy = 15 (X captured from current player energy)"
        );
    }

    /// Bead `sts2-rs-ay8cl`: a HistoryCourse dupe of an ATTACK fires
    /// `Stomp.BeforeCardPlayed` (C# `OnPlayWrapper` runs the full hook chain),
    /// reducing every Stomp's cost by 1. The minimal dupe path skipped this, so
    /// a Stomp in hand kept full cost (coverage seed 3460: STOMP cost rs=3 vs
    /// cs=2). With a Strike dupe and a base-cost-3 Stomp in hand, Stomp → 2.
    #[test]
    fn listener_history_course_attack_dupe_reduces_stomp_cost() {
        use sts2_core::cards::{Card, CardId};
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let suid = sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
            .expect("nz");
        let mut stomp = Card::pod_defaults();
        stomp.uid = suid;
        stomp.card_id = CardId::Stomp;
        stomp.cost = sts2_core::CardEnergyCost::new(3);
        state.cards.push(stomp);
        state.player.hand.push(suid);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        let cost = state
            .cards
            .iter()
            .find(|c| c.uid == suid)
            .expect("stomp")
            .local_cost();
        assert_eq!(
            cost, 2,
            "Stomp cost reduced by 1 — the attack dupe fired BeforeCardPlayed"
        );
    }

    /// A HistoryCourse dupe of a HiddenGem Replay-2 Attack plays 3 times, and
    /// C# `OnPlayWrapper` fires `Hook.BeforeCardPlayed` on EVERY play iteration
    /// (CardModel.cs ~1926), so `Stomp.BeforeCardPlayed` (Stomp.cs:62-73)
    /// shaves 1 per iteration: a cost-3 Stomp drops to 0. The dupe replay loop
    /// shaved only on the first play (act23 Ironclad-8066 act 3 @1117: a
    /// Replay-2 Whirlwind dupe left STOMP at cost cs=0 vs rs=2).
    #[test]
    fn listener_history_course_replayed_attack_dupe_shaves_stomp_per_play() {
        use sts2_core::cards::{Card, CardId};
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let push = |state: &mut CombatState, id: CardId| {
            let mut c = crate::cards::spec_index::card_pod_from_id(id, false).unwrap();
            let uid =
                sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).unwrap()).unwrap();
            c.uid = uid;
            state.cards.push(c);
            uid
        };
        let strike = push(&mut state, CardId::StrikeIronclad);
        state.cards[strike.0.get() as usize - 1].base_replay_count = 2;
        state.player.discard.push(strike);
        let suid = sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
            .expect("nz");
        let mut stomp = Card::pod_defaults();
        stomp.uid = suid;
        stomp.card_id = CardId::Stomp;
        stomp.cost = sts2_core::CardEnergyCost::new(3);
        state.cards.push(stomp);
        state.player.hand.push(suid);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, Some(strike)));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        let cost = state
            .cards
            .iter()
            .find(|c| c.uid == suid)
            .expect("stomp")
            .local_cost();
        assert_eq!(
            cost, 0,
            "one Stomp shave per play iteration (1 play + 2 replays)"
        );
    }

    /// An `AnyEnemy` dupe with no hittable enemy (TestSubject at 0 HP between
    /// revives) is moved to its result pile without playing (C#
    /// `CardCmd.AutoPlay`, CardCmd.cs:74-85): no `BeforeCardPlayed` Stomp
    /// shave, no finished-play count, nothing left on Play (Ironclad-2176 act
    /// 3: a phantom Bash dupe shaved a Stomp, C# 3 vs Rust 2).
    #[test]
    fn listener_history_course_no_hittable_target_dupe_is_not_played() {
        use sts2_core::cards::{Card, CardId};
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 2);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.current_hp = 0;
        }
        let suid = sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits"))
            .expect("nz");
        let mut stomp = Card::pod_defaults();
        stomp.uid = suid;
        stomp.card_id = CardId::Stomp;
        stomp.cost = sts2_core::CardEnergyCost::new(3);
        state.cards.push(stomp);
        state.player.draw.push(suid);
        let plays_before = state.combat_log.player_card_plays_this_combat;
        state.combat_log.prev_round_last_attack_skill_played = Some((CardId::Bash, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        let stomp = state.cards.iter().find(|c| c.uid == suid).expect("stomp");
        assert_eq!(stomp.local_cost(), 3, "no BeforeCardPlayed Stomp shave");
        assert!(!stomp.cost.has_local_modifiers());
        assert_eq!(
            state.combat_log.player_card_plays_this_combat, plays_before,
            "the unplayed dupe is not a finished play"
        );
        assert!(state.player.play.is_empty(), "the dupe left combat");
    }

    /// Round 1 never replays (C# `RoundNumber == 1` guard).
    #[test]
    fn listener_history_course_inert_on_round_one() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 1);
        let before = enemy_hp(&state, enemy);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(enemy_hp(&state, enemy), before, "round 1 must not replay");
    }

    /// No replay when no Attack/Skill was played last round.
    #[test]
    fn listener_history_course_inert_without_history() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 2);
        let before = enemy_hp(&state, enemy);
        state.combat_log.last_attack_skill_played = None;
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(enemy_hp(&state, enemy), before, "no history → no replay");
    }

    /// No replay without ownership.
    #[test]
    fn listener_history_course_inert_without_ownership() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let _ = init_player_uid(&mut state);
        let enemy = push_cultist(&mut state);
        state.enemy_uids.push(enemy);
        state.round = 2;
        let before = enemy_hp(&state, enemy);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(enemy_hp(&state, enemy), before, "no relic → no replay");
    }

    // --- Listener bodies (sts2-rs-v7mp wave 4) ---------------------------

    /// `FakeBloodVial` heals 1 HP on the player's round-1 turn-start, in the
    /// `AfterPlayerTurnStartLate` pass only. The side-turn-start walk must not
    /// heal a second time (a content-side registration left on that bus
    /// double-healed: quick-tier Ironclad seeds 361 / 365, cs=53 rs=54).
    #[test]
    fn listener_fake_blood_vial_heals_one_on_round_one() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeBloodVial);
        state.player.current_hp = 50;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        state.sync_player_creature();
        state.round = 1;
        sts2_core::relic_effects::blood_vial::dispatch_after_player_turn_start_late(&mut state);
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.current_hp, 51);
        let row = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .expect("player creature row");
        assert_eq!(row.current_hp, 51);
    }

    /// `FakeBloodVial` does NOT heal outside round 1.
    #[test]
    fn listener_fake_blood_vial_noop_outside_round_one() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeBloodVial);
        state.player.current_hp = 50;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        state.sync_player_creature();
        state.round = 2;
        sts2_core::relic_effects::blood_vial::dispatch_after_player_turn_start_late(&mut state);
        assert_eq!(state.player.current_hp, 50);
    }

    /// `RoyalPoison` deals 4 unblockable damage to the player on round-1
    /// turn-start (bypasses block).
    #[test]
    fn listener_royal_poison_deals_four_unblockable_self_damage_round_one() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::RoyalPoison);
        state.player.current_hp = 50;
        state.player.max_hp = 80;
        state.player.block = 10; // Block should NOT absorb (UNBLOCKABLE).
        let _player_uid = state.init_player_creature();
        state.sync_player_creature();
        state.round = 1;
        // `AfterPlayerTurnStart` (RoyalPoison.cs:18).
        let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
        assert_eq!(state.player.current_hp, 46, "4 HP loss bypasses block");
        assert_eq!(state.player.block, 10, "block untouched");
    }

    /// `MrStruggles` deals damage = `round_no` to every alive enemy on the
    /// player's turn-start.
    #[test]
    fn listener_mr_struggles_deals_round_damage_to_all_enemies() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::MrStruggles);
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        state.sync_player_creature();
        let e1 = push_cultist(&mut state);
        let e2 = push_cultist(&mut state);
        state.enemy_uids.push(e1);
        state.enemy_uids.push(e2);
        state.round = 3;
        // MrStruggles now fires on C#'s `AfterPlayerTurnStart` (sweep seed
        // 5321 relocation).
        let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
        for uid in [e1, e2] {
            let row = state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .expect("enemy row");
            assert_eq!(row.current_hp, 48 - 3, "3 dmg to each enemy");
        }
    }

    /// `LostWisp` deals 8 damage to all alive enemies when a Power card
    /// is played. Skill cards do NOT trigger.
    #[test]
    fn listener_lost_wisp_fires_on_power_card_played_only() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::LostWisp);
        let _player_uid = init_player_uid(&mut state);
        let enemy = push_cultist(&mut state);
        state.enemy_uids.push(enemy);
        // Push a Power card.
        let next_raw = state
            .cards
            .iter()
            .map(|c| c.uid.0.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let power_uid = CardUid(NonZeroU32::new(next_raw).expect("non-zero"));
        state.cards.push(sts2_core::Card {
            uid: power_uid,
            card_id: sts2_core::cards::CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: sts2_core::cards::CardType::Power,
            target_type: sts2_core::cards::TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: sts2_core::cards::CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        let ctx_power = ctx_for(power_uid, &state);
        let _ = sts2_core::relic_effects::dispatch_after_card_played(&mut state, &ctx_power);
        let row = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .expect("enemy row");
        assert_eq!(row.current_hp, 48 - 8);
        // Skill does not retrigger.
        let skill_uid = push_skill_card(&mut state);
        let ctx_skill = ctx_for(skill_uid, &state);
        let _ = sts2_core::relic_effects::dispatch_after_card_played(&mut state, &ctx_skill);
        let row = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .expect("enemy row");
        assert_eq!(row.current_hp, 48 - 8, "Skill is not a Power trigger");
    }

    /// `ForgottenSoul` deals 1 damage to a random alive enemy on
    /// `after_card_exhausted`. With a single enemy the pick is
    /// deterministic.
    #[test]
    fn listener_forgotten_soul_hits_random_enemy_on_exhaust() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::ForgottenSoul);
        let _player_uid = init_player_uid(&mut state);
        let enemy = push_cultist(&mut state);
        state.enemy_uids.push(enemy);
        let card_uid = push_skill_card(&mut state);
        let ctx = ctx_for(card_uid, &state);
        let _ = sts2_core::relic_effects::dispatch_after_card_exhausted(&mut state, &ctx);
        let row = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .expect("enemy row");
        assert_eq!(row.current_hp, 48 - 1);
    }

    /// **Live-mirror residual class "Queen / TorchHeadAmalgam ±1 / ±3
    /// enemy-HP", root-caused 2026-08-03 — NOT a rounding-order bug.**
    ///
    /// The ±1 and ±3 enemy-HP records in `sts2-pycli/logs/diag/mirror_live3
    /// .jsonl` are the two `AfterCardExhausted` relic damage hooks landing one
    /// decision apart from the live snapshot, not a percent/rounding mismatch:
    ///
    /// - C# `ForgottenSoul.cs:16` — `DamageVar(1m, ValueProp.Unpowered)`
    /// - C# `CharonsAshes.cs:16` — `DamageVar(3m, ValueProp.Unpowered)`
    ///
    /// and `VulnerablePower.ModifyDamageMultiplicative` early-returns `1m`
    /// when `!props.IsPoweredAttack()` (`VulnerablePower.cs:29-32`). So both
    /// ticks are FLAT — there is no `× 1.5` to floor-vs-round in the first
    /// place, which is exactly what falsifies the "percentage rounding-order"
    /// reading. Pinned against live record 336 (seed 1011): TorchHeadAmalgam
    /// sitting at `VULNERABLE 3` still moved by exactly 3, and the untouched
    /// Queen went 397 → 394.
    ///
    /// The `ValueProps::MOVE` control arm keeps the test honest: on the same
    /// board a *powered* 3 does become `floor(3 × 1.5) = 4`, so a regression
    /// that dropped the Vulnerable multiplier entirely would still fail here.
    #[test]
    fn exhaust_damage_relics_are_flat_unpowered_and_ignore_vulnerable() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::CharonsAshes);
        let _player_uid = init_player_uid(&mut state);
        // Two enemies: the Vulnerable one stands in for TorchHeadAmalgam, the
        // clean one for the Queen (CharonsAshes hits HittableEnemies).
        let vulnerable_enemy = push_cultist(&mut state);
        let clean_enemy = push_cultist(&mut state);
        state.enemy_uids.push(vulnerable_enemy);
        state.enemy_uids.push(clean_enemy);
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            vulnerable_enemy,
            PowerKind::Vulnerable,
            3,
        );

        let card_uid = push_skill_card(&mut state);
        let ctx = ctx_for(card_uid, &state);
        let _ = sts2_core::relic_effects::dispatch_after_card_exhausted(&mut state, &ctx);

        let hp = |state: &CombatState, uid: CreatureUid| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .expect("enemy row")
                .current_hp
        };
        assert_eq!(
            hp(&state, vulnerable_enemy),
            48 - 3,
            "CharonsAshes is Unpowered — Vulnerable must not scale the tick"
        );
        assert_eq!(hp(&state, clean_enemy), 48 - 3);

        // Control: the SAME 3 dealt as a powered move does pick the multiplier
        // up, with a single floor — floor(3 × 1.5) = 4.
        let _ = sts2_core::damage::apply_damage(
            &mut state,
            sts2_core::hooks::CreatureRef::Player,
            vulnerable_enemy,
            3,
            ValueProps::MOVE,
            None,
        );
        assert_eq!(
            hp(&state, vulnerable_enemy),
            48 - 3 - 4,
            "powered 3 under Vulnerable is floor(4.5) = 4"
        );
    }

    /// Companion of `exhaust_damage_relics_are_flat_unpowered_and_ignore_vulnerable`
    /// for `ForgottenSoul`'s 1-damage tick: live records 22/23 and 26/27
    /// (seed 1000) moved TorchHeadAmalgam by exactly 1 while it carried
    /// `VULNERABLE 4` / `VULNERABLE 8`. `DamageVar(1m, Unpowered)` is flat.
    #[test]
    fn forgotten_soul_exhaust_tick_ignores_vulnerable() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::ForgottenSoul);
        let _player_uid = init_player_uid(&mut state);
        let enemy = push_cultist(&mut state);
        state.enemy_uids.push(enemy);
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            enemy,
            PowerKind::Vulnerable,
            4,
        );
        let card_uid = push_skill_card(&mut state);
        let ctx = ctx_for(card_uid, &state);
        let _ = sts2_core::relic_effects::dispatch_after_card_exhausted(&mut state, &ctx);
        let row = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .expect("enemy row");
        assert_eq!(
            row.current_hp,
            48 - 1,
            "ForgottenSoul is Unpowered — Vulnerable must not scale the tick"
        );
    }

    /// `EmberTea` is a no-op when the player doesn't carry the relic.
    #[test]
    fn listener_ember_tea_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        let player_uid = init_player_uid(&mut state);
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        let player_idx = state
            .creatures
            .iter()
            .position(|c| c.id == player_uid.0)
            .expect("player creature row");
        assert!(state.creatures[player_idx].powers.is_empty());
    }

    // --- Wave 6 (sts2-rs-9wx7) reward-flow retrofits -----------------------

    /// `DreamCatcher` pushes one `CardReward` on every rest heal, uncapped
    /// (C# `DreamCatcher.TryModifyRestSiteHealRewards`), and keeps no
    /// counter: a combat's counter flush-back (which writes the combat
    /// relic's 0 into `relic_counters`) must not switch it off.
    #[test]
    fn listener_dream_catcher_rewards_every_rest_heal() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::DreamCatcher);
        for _ in 0..5 {
            let rewards = rs.offer_rest_site_rewards();
            assert_eq!(rewards.len(), 1);
            assert!(matches!(rewards[0], Reward::Card(_)));
        }
        // What `exit_combat`'s counter flush-back leaves behind.
        rs.relic_counters.insert(RelicId::DreamCatcher, 0);
        let rewards = rs.offer_rest_site_rewards();
        assert_eq!(rewards.len(), 1);
        assert!(matches!(rewards[0], Reward::Card(_)));
    }

    /// `DreamCatcher` is a no-op when the player doesn't carry it.
    #[test]
    fn listener_dream_catcher_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let rewards = rs.offer_rest_site_rewards();
        assert!(rewards.is_empty());
        assert!(!rs.relic_counters.contains_key(&RelicId::DreamCatcher));
    }

    /// `FresnelLens::modify_card_being_added_to_deck` enchants the
    /// added card with `Nimble(2)`.
    #[test]
    fn listener_fresnel_lens_enchants_deck_add() {
        use sts2_core::cards::{CardType as CT, EnchantmentId, TargetType};
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::FresnelLens);
        let mut card = Card {
            uid: CardUid(std::num::NonZeroU32::new(42).unwrap()),
            card_id: sts2_core::cards::CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CT::Skill,
            target_type: TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 5,
            ..Card::pod_defaults()
        };
        rs.modify_card_being_added_to_deck(&mut card);
        assert_eq!(card.enchantment(EnchantmentId::Nimble), 2);
        // Already enchanted: no enchantment stacks (`EnchantmentModel.cs:173`),
        // so a second add does not stack it to 4.
        rs.modify_card_being_added_to_deck(&mut card);
        assert_eq!(card.enchantment(EnchantmentId::Nimble), 2);
        // A Strike gains no block, so Nimble cannot enchant it (`Nimble.cs:11`).
        let mut strike = crate::cards::spec_index::card_pod_from_id(
            sts2_core::cards::CardId::StrikeIronclad,
            false,
        )
        .expect("strike");
        rs.modify_card_being_added_to_deck(&mut strike);
        assert!(strike.enchantments.is_empty());
    }

    /// `FresnelLens::modify_card_reward_options_late` enchants every
    /// option in the reward list.
    #[test]
    fn listener_fresnel_lens_enchants_card_reward_options() {
        use sts2_core::cards::{CardType as CT, EnchantmentId, TargetType};
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::FresnelLens);
        let mk = |uid: u32| Card {
            uid: CardUid(std::num::NonZeroU32::new(uid).unwrap()),
            card_id: sts2_core::cards::CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CT::Skill,
            target_type: TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 5,
            ..Card::pod_defaults()
        };
        let options = vec![mk(1), mk(2), mk(3)];
        let enchanted = rs.offer_card_reward(options);
        assert_eq!(enchanted.len(), 3);
        for card in &enchanted {
            assert_eq!(card.enchantment(EnchantmentId::Nimble), 2);
        }
    }

    /// `FresnelLens::modify_merchant_card_creation_results` enchants
    /// the merchant card regardless of the `is_colorless` flag.
    #[test]
    fn listener_fresnel_lens_enchants_merchant_card() {
        use sts2_core::cards::{CardType as CT, EnchantmentId, TargetType};
        use sts2_core::CardUid;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::FresnelLens);
        let mut card = Card {
            uid: CardUid(std::num::NonZeroU32::new(42).unwrap()),
            card_id: sts2_core::cards::CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CT::Skill,
            target_type: TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 5,
            ..Card::pod_defaults()
        };
        rs.modify_merchant_card_creation_results(&mut card, true);
        assert_eq!(card.enchantment(EnchantmentId::Nimble), 2);
    }

    /// `WongosMysteryTicket::after_combat_end` increments the
    /// `_combats_finished` counter; once 5 lapse, the next
    /// `modify_rewards` on a Monster/Elite/Boss room pushes 3
    /// `RelicReward` entries and latches `_gave_relic`. Subsequent
    /// calls are no-ops.
    #[test]
    fn listener_wongos_mystery_ticket_grants_three_relics_after_five_combats() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WongosMysteryTicket);

        // 4 combats — not yet at threshold.
        for _ in 0..4 {
            let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        }
        let rewards = rs.offer_rewards(RoomType::Monster, Vec::new());
        assert!(rewards.is_empty(), "below threshold should not pay out");

        // 5th combat — now at threshold.
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);

        // Wrong room type — must still be a no-op.
        let rewards = rs.offer_rewards(RoomType::RestSite, Vec::new());
        assert!(rewards.is_empty(), "RestSite must not pay out");

        // Right room type — pay out.
        let rewards = rs.offer_rewards(RoomType::Monster, Vec::new());
        assert_eq!(rewards.len(), 3);
        for r in &rewards {
            assert!(matches!(r, Reward::Relic(_)));
        }

        // After the payout, the relic latches and no further rewards
        // are produced.
        let rewards = rs.offer_rewards(RoomType::Monster, Vec::new());
        assert!(rewards.is_empty());
        // Counter increments stop once latched.
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert!(rs.relic_counters[&RelicId::WongosMysteryTicket] < 0);
    }

    /// `WongosMysteryTicket` is a no-op when the player doesn't carry
    /// it.
    #[test]
    fn listener_wongos_mystery_ticket_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..10 {
            let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        }
        let rewards = rs.offer_rewards(RoomType::Monster, Vec::new());
        assert!(rewards.is_empty());
        assert!(!rs
            .relic_counters
            .contains_key(&RelicId::WongosMysteryTicket));
    }

    // -----------------------------------------------------------------
    // sts2-rs-a6j4v — LavaRock modify_rewards listener
    // -----------------------------------------------------------------

    /// `LavaRock::modify_rewards` appends 2 `RelicReward` sentinels
    /// when the player owns it and the post-combat reward fan-out
    /// fires for the Act-1 Boss room. Latches via `relic_counters` so
    /// subsequent Boss-room reward fan-outs are no-ops.
    #[test]
    fn listener_lava_rock_grants_two_relics_after_act_1_boss() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaRock);
        // Default RunState::new() sets act_index=1; sanity-confirm.
        assert_eq!(rs.act_index, 1, "fresh RunState defaults to act 1");

        // Wrong room type — must NOT pay out.
        let rewards = rs.offer_rewards(RoomType::Monster, Vec::new());
        assert!(rewards.is_empty(), "Monster room must not trigger LavaRock");
        let rewards = rs.offer_rewards(RoomType::Elite, Vec::new());
        assert!(rewards.is_empty(), "Elite room must not trigger LavaRock");

        // Right room type at act 1 — pay out.
        let rewards = rs.offer_rewards(RoomType::Boss, Vec::new());
        assert_eq!(rewards.len(), 2, "Boss room at act 1 grants 2 relics");
        for r in &rewards {
            assert!(matches!(r, Reward::Relic(rr) if rr.relic_id == RelicId::LavaRock));
        }
        assert_eq!(
            rs.relic_counters[&RelicId::LavaRock],
            1,
            "counter latched at 1 after firing"
        );

        // After the payout the listener latches: a second Boss-room
        // fan-out produces no extra rewards.
        let rewards = rs.offer_rewards(RoomType::Boss, Vec::new());
        assert!(rewards.is_empty(), "second Boss-room fan-out is a no-op");
    }

    /// LavaRock is a no-op past Act 1 (acts 2 / 3 boss rooms don't
    /// trigger it).
    #[test]
    fn listener_lava_rock_noop_at_act_2_and_act_3_boss_rooms() {
        crate::relics::dispatch::ensure_installed();
        for act in [2u8, 3u8] {
            let mut rs = RunState::new();
            rs.relics.push(RelicId::LavaRock);
            rs.act_index = act;
            let rewards = rs.offer_rewards(RoomType::Boss, Vec::new());
            assert!(
                rewards.is_empty(),
                "act {act} boss room must not trigger LavaRock"
            );
            assert!(
                !rs.relic_counters.contains_key(&RelicId::LavaRock),
                "act {act} boss room must not latch the counter"
            );
        }
    }

    /// LavaRock is a no-op when the player doesn't own it (even at
    /// act 1 boss room).
    #[test]
    fn listener_lava_rock_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        assert_eq!(rs.act_index, 1);
        let rewards = rs.offer_rewards(RoomType::Boss, Vec::new());
        assert!(rewards.is_empty(), "no relic, no reward");
        assert!(!rs.relic_counters.contains_key(&RelicId::LavaRock));
    }

    /// LavaRock preserves caller-supplied rewards — its 2 relic
    /// sentinels are *appended* to the input vector, not a replacement.
    #[test]
    fn listener_lava_rock_appends_to_existing_rewards() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaRock);
        let prior = vec![Reward::Gold { amount: 50 }];
        let rewards = rs.offer_rewards(RoomType::Boss, prior);
        assert_eq!(rewards.len(), 3, "prior gold + 2 LavaRock relics");
        assert!(matches!(rewards[0], Reward::Gold { amount: 50 }));
        for r in &rewards[1..] {
            assert!(matches!(r, Reward::Relic(rr) if rr.relic_id == RelicId::LavaRock));
        }
    }

    // -----------------------------------------------------------------
    // sts2-rs-yb6rm — MawBank after_room_entered + on_item_purchased
    // -----------------------------------------------------------------

    /// MawBank grants +12g on every room enter until the latch fires.
    /// Mirrors Python `MawBank.after_room_entered` (no room-type
    /// filter): Monster / Elite / Boss / Shop / RestSite / Treasure /
    /// Event all trigger the payout while `_purchased == False`.
    #[test]
    fn listener_maw_bank_grants_gold_every_room_enter() {
        use sts2_core::runstate::Room;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MawBank);
        let start = rs.gold;

        for rt in [
            RoomType::Monster,
            RoomType::Elite,
            RoomType::Boss,
            RoomType::Shop,
            RoomType::RestSite,
            RoomType::Treasure,
            RoomType::Event,
        ] {
            let before = rs.gold;
            rs.enter_room(Room::new(rt));
            assert_eq!(
                rs.gold,
                before + MAW_BANK_GOLD,
                "MawBank must grant {MAW_BANK_GOLD}g on entering {rt:?}"
            );
        }
        assert_eq!(rs.gold, start + 7 * MAW_BANK_GOLD);
        assert!(
            !rs.relic_counters.contains_key(&RelicId::MawBank),
            "latch stays open until a non-zero purchase fires"
        );
    }

    /// A non-zero-gold shop purchase latches MawBank — subsequent room
    /// enters become no-ops.
    #[test]
    fn listener_maw_bank_latches_on_paid_purchase() {
        use sts2_core::run_effects::dispatch_on_item_purchased;
        use sts2_core::runstate::Room;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MawBank);

        rs.enter_room(Room::new(RoomType::Monster));
        assert_eq!(rs.gold - 99, MAW_BANK_GOLD, "first room pays out");

        let _ = dispatch_on_item_purchased(&mut rs, 50);
        assert_eq!(
            rs.relic_counters.get(&RelicId::MawBank).copied(),
            Some(MAW_BANK_PURCHASED),
            "non-zero purchase latches the relic"
        );

        let before = rs.gold;
        rs.enter_room(Room::new(RoomType::Shop));
        rs.enter_room(Room::new(RoomType::Boss));
        assert_eq!(rs.gold, before, "post-latch room enters are no-ops");
    }

    /// A `gold_spent == 0` event (LordsParasol auto-purchase, free
    /// pickups) does NOT latch the relic — matches Python's
    /// `if gold_spent > 0` gate.
    #[test]
    fn listener_maw_bank_free_purchase_does_not_latch() {
        use sts2_core::run_effects::dispatch_on_item_purchased;
        use sts2_core::runstate::Room;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::MawBank);

        let _ = dispatch_on_item_purchased(&mut rs, 0);
        assert!(
            !rs.relic_counters.contains_key(&RelicId::MawBank),
            "free auto-purchase must not latch the relic"
        );

        let before = rs.gold;
        rs.enter_room(Room::new(RoomType::Monster));
        assert_eq!(
            rs.gold,
            before + MAW_BANK_GOLD,
            "room-enter payout still fires after a free pickup"
        );
    }

    /// Without the relic the listener is fully inert — no gold payout,
    /// no counter write — even after `on_item_purchased` fires.
    #[test]
    fn listener_maw_bank_noop_without_ownership() {
        use sts2_core::run_effects::dispatch_on_item_purchased;
        use sts2_core::runstate::Room;
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let start = rs.gold;

        rs.enter_room(Room::new(RoomType::Monster));
        let _ = dispatch_on_item_purchased(&mut rs, 50);
        assert_eq!(rs.gold, start, "no relic, no gold tick");
        assert!(!rs.relic_counters.contains_key(&RelicId::MawBank));
    }

    // -----------------------------------------------------------------
    // sts2-rs-67co3 — LavaLamp modify_card_reward_options_late listener
    // -----------------------------------------------------------------

    fn unupgraded_card(card_id: sts2_core::cards::CardId) -> sts2_core::cards::Card {
        sts2_core::cards::Card {
            card_id,
            upgraded: false,
            ..sts2_core::cards::Card::pod_defaults()
        }
    }

    /// C# `LavaLamp.AfterRoomEntered` clears the took-damage flag on every
    /// room entry, so a damaged fight does not deny the upgrade after the
    /// next clean one (fuzz v0.111.0 seed 1159).
    #[test]
    fn lava_lamp_flag_resets_on_room_entry() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaLamp);
        rs.relic_counters.insert(RelicId::LavaLamp, 1);
        rs.enter_room(sts2_core::runstate::Room::new(RoomType::Monster));
        assert_eq!(rs.relic_counters[&RelicId::LavaLamp], 0);
    }

    /// LavaLamp upgrades every offered card when `relic_counters[LavaLamp]`
    /// is absent (the "no damage taken" state).
    #[test]
    fn listener_lava_lamp_upgrades_all_when_no_damage_taken() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaLamp);
        // Counter absent ↔ no damage taken.
        assert!(!rs.relic_counters.contains_key(&RelicId::LavaLamp));
        let mut cards = vec![
            unupgraded_card(sts2_core::cards::CardId::StrikeIronclad),
            unupgraded_card(sts2_core::cards::CardId::DefendIronclad),
            unupgraded_card(sts2_core::cards::CardId::Bash),
        ];
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        for card in &cards {
            assert!(card.upgraded, "{:?} should be upgraded", card.card_id);
        }
    }

    /// LavaLamp upgrades the reward cards with the run build's delta: a
    /// v0.111.0 Hyperbeam goes 24 -> 30 (`Hyperbeam.cs:61`
    /// `UpgradeValueBy(6)`), not the v0.107.1 +8.
    #[test]
    fn listener_lava_lamp_upgrades_with_build_delta_v0111() {
        crate::relics::dispatch::ensure_installed();
        let v = sts2_core::GameVersion::V0_111_0;
        let mut rs = RunState::new();
        rs.game_version = v;
        rs.relics.push(RelicId::LavaLamp);
        let mut cards =
            vec![
                crate::cards::v0111::card_pod_for(sts2_core::cards::CardId::Hyperbeam, false, v)
                    .expect("spec"),
            ];
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(cards[0].upgraded);
        assert_eq!(cards[0].base_damage, 30);
    }

    /// C# `PaelsTooth.AfterCombatEnd` (`PaelsTooth.cs:99,111`, same in
    /// v0.107.1) draws `Rewards.NextItem(SerializableCards)` and then calls
    /// `SerializableCards.Remove(item)`: `List.Remove` with the overridden
    /// `SerializableCard.Equals` (Id + CurrentUpgradeLevel + Enchantment Id and
    /// Amount, `SerializableCard.cs:73-89`), so it drops the FIRST equal entry,
    /// not the drawn index. With an enchanted Strike between plain ones, drawing
    /// a later plain Strike removes the earlier plain one.
    #[test]
    fn listener_paels_tooth_return_removes_first_equal_stash_entry() {
        use sts2_core::cards::{CardId, EnchantmentId};
        crate::relics::dispatch::ensure_installed();
        let v = sts2_core::GameVersion::V0_111_0;
        let card = |id| crate::cards::v0111::card_pod_for(id, false, v).unwrap();
        let mut sown = card(CardId::StrikeIronclad);
        sown.enchantments.set(EnchantmentId::Steady, 1);
        let mut rs = RunState::new();
        rs.game_version = v;
        rs.relics.push(RelicId::PaelsTooth);
        rs.paels_tooth_stored = vec![
            card(CardId::Bash),
            card(CardId::StrikeIronclad),
            sown,
            card(CardId::StrikeIronclad),
            card(CardId::StrikeIronclad),
        ];
        // Burn rewards draws until the next `[0, 5)` draw lands on index 3.
        loop {
            let mut probe = rs.rng.rewards.clone();
            if probe.next_int_exclusive(0, 5) == 3 {
                break;
            }
            let _ = rs.rng.rewards.next_int_exclusive(0, 5);
        }
        let _ = listener_paels_tooth_after_combat_end(&mut rs);
        let left: Vec<(CardId, bool)> = rs
            .paels_tooth_stored
            .iter()
            .map(|c| (c.card_id, c.enchantments.get(EnchantmentId::Steady) > 0))
            .collect();
        assert_eq!(
            left,
            vec![
                (CardId::Bash, false),
                (CardId::StrikeIronclad, true),
                (CardId::StrikeIronclad, false),
                (CardId::StrikeIronclad, false),
            ]
        );
        let back = rs.master_deck.last().expect("returned to the deck");
        assert_eq!(back.card_id, CardId::StrikeIronclad);
        assert_eq!(back.enchantments.get(EnchantmentId::Steady), 0);
    }

    /// PaelsTooth hands the stored card back upgraded with the run build's
    /// delta (v0.111.0 Hyperbeam 30, Expect a Fight keeps cost 3).
    #[test]
    fn listener_paels_tooth_upgrades_with_build_delta_v0111() {
        crate::relics::dispatch::ensure_installed();
        let v = sts2_core::GameVersion::V0_111_0;
        for (id, check) in [
            (
                sts2_core::cards::CardId::Hyperbeam,
                (|c: &Card| c.base_damage == 30) as fn(&Card) -> bool,
            ),
            (sts2_core::cards::CardId::ExpectAFight, |c: &Card| {
                c.local_cost() == 3
            }),
        ] {
            let mut rs = RunState::new();
            rs.game_version = v;
            rs.relics.push(RelicId::PaelsTooth);
            rs.paels_tooth_stored = vec![crate::cards::v0111::card_pod_for(id, false, v).unwrap()];
            let _ = listener_paels_tooth_after_combat_end(&mut rs);
            let card = rs.master_deck.last().expect("returned to the deck");
            assert!(card.upgraded && check(card), "{id:?}: {card:?}");
        }
    }

    /// LavaLamp is a no-op when `relic_counters[LavaLamp] != 0`
    /// (the "damage taken" state set by the future firing-site bead).
    #[test]
    fn listener_lava_lamp_is_noop_when_damage_taken_flag_set() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaLamp);
        rs.relic_counters.insert(RelicId::LavaLamp, 1);
        let mut cards = vec![
            unupgraded_card(sts2_core::cards::CardId::StrikeIronclad),
            unupgraded_card(sts2_core::cards::CardId::DefendIronclad),
        ];
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        for card in &cards {
            assert!(!card.upgraded, "damage-taken flag must suppress upgrades");
        }
    }

    /// LavaLamp is a no-op when the player doesn't own it.
    #[test]
    fn listener_lava_lamp_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // No LavaLamp in relics.
        let mut cards = vec![unupgraded_card(sts2_core::cards::CardId::StrikeIronclad)];
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(!cards[0].upgraded);
    }

    /// LavaLamp doesn't re-upgrade an already-upgraded card (idempotent
    /// via `Card::upgrade` semantics — flag flips only on the first
    /// call). The listener still resolves as long as at least one
    /// card was actually flipped.
    #[test]
    fn listener_lava_lamp_idempotent_on_mixed_cards() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::LavaLamp);
        let mut pre_upgraded = unupgraded_card(sts2_core::cards::CardId::StrikeIronclad);
        pre_upgraded.upgraded = true;
        let mut cards = vec![
            pre_upgraded,
            unupgraded_card(sts2_core::cards::CardId::DefendIronclad),
        ];
        let _ =
            sts2_core::run_effects::dispatch_modify_card_reward_options_late(&mut rs, &mut cards);
        assert!(cards[0].upgraded, "pre-upgraded stays upgraded");
        assert!(cards[1].upgraded, "non-upgraded gets upgraded");
    }

    // -----------------------------------------------------------------
    // Wave 5 (sts2-rs-jvk3) tests — SwordOfStone + FragrantMushroom
    // -----------------------------------------------------------------

    /// `SwordOfStone::after_combat_victory` decrements the elite
    /// counter exactly once per elite combat. Bead `sts2-rs-jvk3`.
    #[test]
    fn listener_sword_of_stone_decrements_counter_on_elite_victory() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::SwordOfStone, 5);
        state.is_elite = true;
        let _ = sts2_core::relic_effects::dispatch_after_combat_victory(&mut state);
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::SwordOfStone)
            .expect("relic present");
        assert_eq!(relic.counter, 4);
    }

    /// The listener is a no-op when the combat was not an elite
    /// encounter — `is_elite == false` skips the decrement.
    #[test]
    fn listener_sword_of_stone_noop_on_non_elite_victory() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::SwordOfStone, 5);
        assert!(!state.is_elite);
        let _ = sts2_core::relic_effects::dispatch_after_combat_victory(&mut state);
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::SwordOfStone)
            .expect("relic present");
        assert_eq!(relic.counter, 5);
    }

    /// The listener stops firing once the counter hits zero — pins the
    /// "no underflow" contract for the post-threshold combats.
    #[test]
    fn listener_sword_of_stone_noop_at_zero_counter() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::SwordOfStone, 0);
        state.is_elite = true;
        let _ = sts2_core::relic_effects::dispatch_after_combat_victory(&mut state);
        let relic = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::SwordOfStone)
            .expect("relic present");
        assert_eq!(relic.counter, 0);
    }

    /// `FragrantMushroom::after_obtained` deals 15 damage to the
    /// player via `RunState::take_damage` and routes the
    /// upgrade-deck clause through `RunState::upgrade_random_cards`
    /// (real impl since bead `sts2-rs-7krx0`; this test exercises
    /// the HP-loss leg, while the deck-upgrade leg is covered by
    /// `runstate::tests::upgrade_random_cards_*`). Bead
    /// `sts2-rs-jvk3`.
    #[test]
    fn listener_fragrant_mushroom_takes_fifteen_on_obtain() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Pre-obtain HP at default 80; offer + accept the relic so
        // `after_obtained` fires through the canonical channel and the
        // listener observes ownership.
        rs.offer_relic_rewards(RelicId::FragrantMushroom, 1);
        let acquired = rs.accept_relic_reward(0);
        assert_eq!(acquired, Some(RelicId::FragrantMushroom));
        // Default HP is 80; expect 80 - 15 = 65.
        assert_eq!(rs.current_hp, 65);
    }

    /// FragrantMushroom's upgrade-leg integration test (bead
    /// `sts2-rs-ombcw`, the 7krx0.3 follow-up). With a populated
    /// `master_deck` of 5 non-upgraded Cards, the canonical
    /// `after_obtained` firing path bumps two of them to
    /// `upgraded = true` (C# `CardsVar(2)`, `FragrantMushroom.cs:26`).
    #[test]
    fn listener_fragrant_mushroom_upgrades_two_cards_on_obtain() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new().with_rewards_seed(42);
        for i in 0..5_u32 {
            rs.master_deck.push(Card {
                uid: sts2_core::CardUid::new(100 + i).expect("non-zero"),
                ..Card::pod_defaults()
            });
        }
        assert!(rs.master_deck.iter().all(|c| !c.upgraded));

        rs.offer_relic_rewards(RelicId::FragrantMushroom, 1);
        let acquired = rs.accept_relic_reward(0);
        assert_eq!(acquired, Some(RelicId::FragrantMushroom));

        let upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(
            upgraded, 2,
            "FragrantMushroom should upgrade exactly 2 of the 5 candidates",
        );
        assert_eq!(
            rs.current_hp, 65,
            "HP-leg still fires alongside the upgrade-leg"
        );
    }

    // -----------------------------------------------------------------
    // Wave 5 follow-up (sts2-rs-y8zi4) tests — SwordOfStone transform
    // end-to-end via the run-side after_combat_victory channel.
    // -----------------------------------------------------------------

    /// The run-side arm reads the count the combat side flushed: an elite
    /// victory with the count at 0 replaces Sword of Stone with Sword of
    /// Jade; above 0 it leaves both the relic and its count alone.
    #[test]
    fn listener_sword_of_stone_run_transforms_at_zero_counter() {
        crate::relics::dispatch::ensure_installed();
        let ctx = CombatVictoryRunCtx { is_elite: true };
        let mut rs = RunState::new();
        rs.relics.push(RelicId::SwordOfStone);
        rs.relic_counters.insert(RelicId::SwordOfStone, 1);
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert_eq!(
            rs.relic_counters.get(&RelicId::SwordOfStone).copied(),
            Some(1)
        );
        assert!(rs.relics.contains(&RelicId::SwordOfStone));
        rs.relic_counters.insert(RelicId::SwordOfStone, 0);
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert!(!rs.relics.contains(&RelicId::SwordOfStone));
        assert!(rs.relics.contains(&RelicId::SwordOfJade));
    }

    /// Non-elite victories leave the run-side counter and the relic alone —
    /// mirrors the combat-side `is_elite` guard.
    #[test]
    fn listener_sword_of_stone_run_noop_on_non_elite() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::SwordOfStone);
        rs.relic_counters.insert(RelicId::SwordOfStone, 5);
        let ctx = CombatVictoryRunCtx { is_elite: false };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert_eq!(
            rs.relic_counters.get(&RelicId::SwordOfStone).copied(),
            Some(5),
        );
        assert!(rs.relics.contains(&RelicId::SwordOfStone));
    }

    /// The listener is a no-op when the player doesn't own
    /// `SwordOfStone` (e.g. after the transform has already fired or
    /// when the relic was never picked up).
    #[test]
    fn listener_sword_of_stone_run_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        // Already-transformed run: only SwordOfJade is present.
        rs.relics.push(RelicId::SwordOfJade);
        let ctx = CombatVictoryRunCtx { is_elite: true };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        // No counter is written, no extra Jade entry appears.
        assert!(!rs.relic_counters.contains_key(&RelicId::SwordOfStone));
        assert_eq!(
            rs.relics
                .iter()
                .filter(|r| **r == RelicId::SwordOfJade)
                .count(),
            1,
        );
    }

    /// End-to-end with both buses as the run manager drives them: five
    /// elite victories, each the combat side's decrement, the exit flush of
    /// its count into `relic_counters`, then the run-side arm. The fifth
    /// replaces the relic; none before it does.
    #[test]
    fn sword_of_stone_combat_count_then_run_transform_end_to_end() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::SwordOfStone, 5);
        state.is_elite = true;
        let mut rs = RunState::new();
        rs.relics.push(RelicId::SwordOfStone);
        let ctx = CombatVictoryRunCtx { is_elite: true };
        for elite in 1..=5 {
            let _ = sts2_core::relic_effects::dispatch_after_combat_victory(&mut state);
            let count = state
                .relics
                .iter()
                .find(|r| r.kind == RelicKind::SwordOfStone)
                .expect("combat-side SwordOfStone")
                .counter;
            rs.relic_counters.insert(RelicId::SwordOfStone, count);
            let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
            assert_eq!(
                rs.relics.contains(&RelicId::SwordOfJade),
                elite == 5,
                "elite {elite}"
            );
        }
        assert!(!rs.relics.contains(&RelicId::SwordOfStone));
    }

    /// The listener is a no-op when the player doesn't own the relic
    /// — guards against the dispatcher firing for unrelated obtains.
    #[test]
    fn listener_fragrant_mushroom_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.current_hp = 80;
        let before = rs.current_hp;
        // Drive the dispatcher directly without pushing FragrantMushroom
        // onto rs.relics — the listener should ownership-gate to no-op.
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.current_hp, before);
    }

    // -----------------------------------------------------------------
    // WarHammer (bead sts2-rs-mg5ur)
    // -----------------------------------------------------------------

    fn push_unupgraded_anger(rs: &mut RunState, n: usize) {
        for _ in 0..n {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = sts2_core::cards::CardId::Anger;
            rs.master_deck.push(c);
        }
    }

    /// On elite victory with WarHammer owned, 4 non-upgraded deck cards
    /// flip to upgraded. Mirrors Python `WarHammer.CARDS = 4`.
    #[test]
    fn listener_war_hammer_run_upgrades_four_on_elite_victory() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WarHammer);
        push_unupgraded_anger(&mut rs, 6);
        let ctx = CombatVictoryRunCtx { is_elite: true };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        let upgraded = rs.master_deck.iter().filter(|c| c.upgraded).count();
        assert_eq!(upgraded, 4);
    }

    /// Non-elite victories must NOT touch the deck — mirrors Python
    /// `if not combat.is_elite: return`.
    #[test]
    fn listener_war_hammer_run_noop_on_non_elite() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WarHammer);
        push_unupgraded_anger(&mut rs, 6);
        let ctx = CombatVictoryRunCtx { is_elite: false };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert_eq!(rs.master_deck.iter().filter(|c| c.upgraded).count(), 0);
    }

    /// Listener no-ops when the player doesn't own WarHammer — the
    /// dispatcher walks all listeners, so the gate must live in the
    /// body.
    #[test]
    fn listener_war_hammer_run_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        push_unupgraded_anger(&mut rs, 6);
        let ctx = CombatVictoryRunCtx { is_elite: true };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert_eq!(rs.master_deck.iter().filter(|c| c.upgraded).count(), 0);
    }

    /// Saturates at the available non-upgraded count — when the deck
    /// has fewer than 4 unupgraded cards, every available card is
    /// upgraded (matches `RunState::upgrade_random_cards` semantics).
    #[test]
    fn listener_war_hammer_run_saturates_on_small_deck() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::WarHammer);
        push_unupgraded_anger(&mut rs, 2);
        let ctx = CombatVictoryRunCtx { is_elite: true };
        let _ = sts2_core::run_effects::dispatch_after_combat_victory_run(&mut rs, &ctx);
        assert_eq!(rs.master_deck.iter().filter(|c| c.upgraded).count(), 2);
    }

    // -----------------------------------------------------------------
    // BingBong (bead sts2-rs-hyv0p)
    // -----------------------------------------------------------------

    /// BingBong owned: a Reward-sourced add fires the listener,
    /// which calls `duplicate_last_added_card` once. The duplicate's
    /// re-dispatch carries `AddSource::BingBongDuplicate` and the
    /// re-entry guard bails — net deck delta is +2 (the original
    /// add + one duplicate).
    #[test]
    fn listener_bing_bong_duplicates_once_on_reward_add() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::BingBong);
        assert_eq!(state.master_deck.len(), 0);
        state.add_to_master_deck(
            sts2_core::cards::CardId::StrikeIronclad,
            sts2_core::relic_effects::AddSource::Reward,
        );
        assert_eq!(state.master_deck.len(), 2, "original + 1 BingBong dup");
        assert_eq!(
            state.master_deck[0],
            sts2_core::cards::CardId::StrikeIronclad
        );
        assert_eq!(
            state.master_deck[1],
            sts2_core::cards::CardId::StrikeIronclad
        );
    }

    /// BingBongDuplicate-sourced add is the re-entry path; the
    /// listener must bail or the duplicate would recurse infinitely.
    #[test]
    fn listener_bing_bong_noop_on_bing_bong_duplicate_source() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::BingBong);
        state.add_to_master_deck(
            sts2_core::cards::CardId::Bash,
            sts2_core::relic_effects::AddSource::BingBongDuplicate,
        );
        assert_eq!(state.master_deck.len(), 1, "no re-entry duplicate");
    }

    /// Listener is ownership-gated — without BingBong, adds pass
    /// through unduplicated.
    #[test]
    fn listener_bing_bong_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        state.add_to_master_deck(
            sts2_core::cards::CardId::Anger,
            sts2_core::relic_effects::AddSource::Reward,
        );
        assert_eq!(state.master_deck.len(), 1);
    }

    // -----------------------------------------------------------------
    // PollinousCore listener tests
    // -----------------------------------------------------------------

    #[cfg(test)]
    fn pollinous_core_setup(state: &mut CombatState, equip: bool) {
        use std::num::NonZeroU32;
        use sts2_core::Relic;
        if equip {
            state.relics.push(Relic {
                id: NonZeroU32::new(1).expect("non-zero"),
                kind: RelicKind::PollinousCore,
                counter: 0,
                is_wax: false,
                is_melted: false,
            });
        }
    }

    /// **PollinousCore #1** — v0.107.1 `BeforeSideTurnStart` is a plain
    /// `TurnsSeen++` (PollinousCore.cs:90-97); only `AfterModifyingHandDraw`
    /// resets it. A counter left at 4 by a combat that ended before its hand
    /// draw climbs past 4 and the +2 never fires again (loadout Defect 5189
    /// v0.107.1).
    #[test]
    fn listener_pollinous_core_tick_never_wraps() {
        use sts2_core::relic_effects::{
            dispatch_before_side_turn_start, dispatch_modify_hand_draw,
        };
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        pollinous_core_setup(&mut state, true);
        let read_counter = |s: &CombatState| {
            s.relics
                .iter()
                .find(|r| r.kind == RelicKind::PollinousCore)
                .map(|r| r.counter)
                .expect("pollinous owned")
        };
        for expected in 1..=4 {
            let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
            assert_eq!(read_counter(&state), expected);
        }
        // No hand draw consumed the 4 (the combat ended): it keeps climbing.
        for expected in 5..=12 {
            let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
            assert_eq!(read_counter(&state), expected);
            assert_eq!(dispatch_modify_hand_draw(&state, 5), 5, "turn {expected}");
        }
    }

    /// **PollinousCore #2** — modify_hand_draw fires +EXTRA on the
    /// turn the counter equals TURNS; no-op otherwise.
    #[test]
    fn listener_pollinous_core_modify_hand_draw_fires_on_period() {
        use sts2_core::relic_effects::{
            dispatch_before_side_turn_start, dispatch_modify_hand_draw,
        };
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        pollinous_core_setup(&mut state, true);
        // Turns 1-3: no extra.
        for expected_counter in 1..=3 {
            let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
            let draw = dispatch_modify_hand_draw(&state, 5);
            assert_eq!(draw, 5, "turn {expected_counter}: counter ≠ 4, no extra");
        }
        // Turn 4: fire.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        let draw = dispatch_modify_hand_draw(&state, 5);
        assert_eq!(
            draw,
            5 + POLLINOUS_CORE_EXTRA_CARDS,
            "turn 4: +{POLLINOUS_CORE_EXTRA_CARDS}"
        );
        // Turn 5 without the hand draw's reset: 5, no extra.
        let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
        let draw = dispatch_modify_hand_draw(&state, 5);
        assert_eq!(draw, 5, "turn 5: counter 5, no extra");
    }

    /// **PollinousCore #3** — Monster-side ticks do not advance the
    /// player-turn counter.
    #[test]
    fn listener_pollinous_core_tick_monster_side_skipped() {
        use sts2_core::relic_effects::dispatch_before_side_turn_start;
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        pollinous_core_setup(&mut state, true);
        let _ = dispatch_before_side_turn_start(&mut state, Side::Enemy);
        let counter = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::PollinousCore)
            .expect("owned")
            .counter;
        assert_eq!(counter, 0, "monster side does not tick player-turn counter");
    }

    /// **PollinousCore #4** — not owned: modify_hand_draw is a no-op
    /// even if a stale counter would otherwise fire.
    #[test]
    fn listener_pollinous_core_modify_hand_draw_no_relic_no_op() {
        use sts2_core::relic_effects::dispatch_modify_hand_draw;
        crate::relics::dispatch::ensure_installed();
        let state = CombatState::default();
        // No relic equipped — listener returns current_draw unchanged.
        let draw = dispatch_modify_hand_draw(&state, 5);
        assert_eq!(draw, 5);
    }

    // -----------------------------------------------------------------
    // sts2-rs-v9zer — PaelsTooth after_obtained / after_combat_end
    // -----------------------------------------------------------------

    fn deck_card(card_id: sts2_core::cards::CardId) -> sts2_core::cards::Card {
        sts2_core::cards::Card {
            card_id,
            upgraded: false,
            ..sts2_core::cards::Card::pod_defaults()
        }
    }

    /// On pickup, PaelsTooth stows up to 5 removable+upgradable cards from
    /// the master deck into `paels_tooth_stored` and removes them.
    ///
    /// C# selects via `FromDeckForRemoval(filter: c => c.IsUpgradable)`
    /// (`PaelsTooth.cs:81`), so the pool is `IsRemovable && IsUpgradable`:
    /// a basic Strike qualifies on both counts and IS stowed. Until
    /// 2026-08-26 this test asserted the opposite ("Basic Strike was not
    /// stowed") against a hand-rolled basics-excluding list.
    #[test]
    fn listener_paels_tooth_after_obtained_stows_basics_but_not_eternal_cards() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::Bash));
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::Anger));
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::Reave));
        // Removable AND upgradable in C# — a prime tooth candidate.
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::StrikeIronclad));
        // Eternal: `IsRemovable` is false, so it is never offered.
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::Greed));
        let deck_len_before = rs.master_deck.len();
        // Simulate the obtain flow: push relic + set last_obtained.
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(
            rs.paels_tooth_stored.len(),
            4,
            "stowed every removable+upgradable card, basic Strike included"
        );
        assert_eq!(
            rs.master_deck.len(),
            deck_len_before - 4,
            "master_deck shrunk by 4"
        );
        assert!(
            rs.paels_tooth_stored
                .iter()
                .any(|c| c.card_id == sts2_core::cards::CardId::StrikeIronclad),
            "a basic Strike is removable and upgradable, so the tooth takes it"
        );
        assert_eq!(
            rs.master_deck
                .iter()
                .filter(|c| c.card_id == sts2_core::cards::CardId::Greed)
                .count(),
            1,
            "Greed is Eternal, so FromDeckForRemoval never offers it"
        );
        for stored in &rs.paels_tooth_stored {
            assert!(!stored.upgraded, "stowed cards are not pre-upgraded");
        }
    }

    /// The tooth's pool is `IsUpgradable`, not "not yet upgraded": a curse
    /// (`MaxUpgradeLevel => 0`) is never offered. With 4 upgradable cards
    /// and a Guilty curse, C# auto-takes the 4 (`FromDeckGeneric`, count <=
    /// MinSelect) and the run moves on. Rust counted the curse as a 5th
    /// candidate and queued 5 removals, so after the 4 a `card_select` with
    /// no offer was left open (random-loadout Necrobinder 7602 v0.107.1).
    #[test]
    fn listener_paels_tooth_after_obtained_skips_a_curse() {
        crate::relics::dispatch::ensure_installed();
        let v = sts2_core::GameVersion::V0_107_1;
        let mut rs = RunState::new();
        for _ in 0..4 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.master_deck.push(
            sts2_core::build_card_for_version(sts2_core::cards::CardId::Guilty, false, v)
                .expect("Guilty spec"),
        );
        rs.interactive_card_removal = true;
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.pending_deck_removals, 4, "the curse is not upgradable");
        assert!(rs.pending_removal_paels_tooth);
    }

    /// PaelsTooth caps stowage at 5 cards even if more candidates exist.
    /// When candidates > 5, the interactive path (sts2-rs-n7tha)
    /// opens a `PaelsToothStash` pending_choice; the auto-pick path
    /// does NOT fire. The agent must resolve via
    /// `RunState::resolve_paels_tooth_stash`.
    #[test]
    fn listener_paels_tooth_after_obtained_opens_choice_when_candidates_exceed_cap() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..7 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // Auto-pick did NOT fire: storage empty, deck unchanged.
        assert!(
            rs.paels_tooth_stored.is_empty(),
            "interactive path defers stowing until resolve"
        );
        assert_eq!(rs.master_deck.len(), 7, "deck unchanged pre-resolve");
        // pending_choice opened.
        let pc = rs.pending_choice.as_ref().expect("choice opened");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::PaelsToothStash);
        assert_eq!(pc.options.len(), 7);
        assert_eq!(pc.min_select, 5);
        assert_eq!(pc.max_select, 5);
        assert!(!pc.can_skip);
    }

    /// Resolver path: agent picks 5 indices, stash fills from
    /// master_deck, prompt closes.
    #[test]
    fn resolve_paels_tooth_stash_stows_chosen_cards() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..7 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // Resolve picking 5 of the 7 options.
        let ok = rs.resolve_paels_tooth_stash(&[0, 1, 2, 3, 4]);
        assert!(ok);
        assert_eq!(rs.paels_tooth_stored.len(), 5);
        assert_eq!(rs.master_deck.len(), 2, "5 cards removed from deck");
        assert!(rs.pending_choice.is_none(), "prompt closed");
    }

    /// Resolver rejects invalid selections (wrong count) and leaves
    /// the prompt open + state untouched.
    #[test]
    fn resolve_paels_tooth_stash_rejects_wrong_count() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..7 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        // Need exactly 5 (min=max=5); 4 is invalid.
        assert!(!rs.resolve_paels_tooth_stash(&[0, 1, 2, 3]));
        assert!(rs.paels_tooth_stored.is_empty());
        assert_eq!(rs.master_deck.len(), 7);
        assert!(rs.pending_choice.is_some(), "prompt stays open on reject");
    }

    /// Resolver guards on the open prompt kind — won't consume an
    /// unrelated PendingChoice (defensive routing check).
    #[test]
    fn resolve_paels_tooth_stash_rejects_when_prompt_is_different_kind() {
        let mut rs = RunState::new();
        // Open a CardSelect prompt (unrelated to PaelsTooth).
        let opts = vec![sts2_core::ChoiceOption {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            upgraded: false,
            card_uid: None,
            source_pile: None,
        }];
        assert!(rs.request_card_choice(sts2_core::ChoiceKind::CardSelect, opts, 1, 1, false));
        // PaelsTooth resolver must not consume it.
        assert!(!rs.resolve_paels_tooth_stash(&[0]));
        assert!(rs.pending_choice.is_some(), "unrelated prompt left intact");
    }

    /// When candidates <= 5 (forced choice), the auto-pick path
    /// fires unchanged. Regression guard for the v9zer behavior.
    #[test]
    fn listener_paels_tooth_after_obtained_auto_picks_when_candidates_at_or_below_cap() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..3 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.paels_tooth_stored.len(), 3, "auto-picked all 3");
        assert_eq!(rs.master_deck.len(), 0);
        assert!(rs.pending_choice.is_none(), "no prompt — forced choice");
    }

    /// Listener is a no-op when `last_obtained_relic` is something else
    /// — the after_obtained channel walks ALL listeners on every pickup.
    #[test]
    fn listener_paels_tooth_after_obtained_self_gates_on_last_obtained() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.master_deck
            .push(deck_card(sts2_core::cards::CardId::Bash));
        rs.relics.push(RelicId::PaelsTooth);
        // last_obtained points elsewhere — listener should NOT fire.
        rs.last_obtained_relic = Some(RelicId::Anchor);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert!(
            rs.paels_tooth_stored.is_empty(),
            "no stow when last_obtained != PaelsTooth"
        );
    }

    /// after_combat_end draws a card from storage, upgrades it, and
    /// pushes it back onto the master deck. Stored count decrements by
    /// 1 per combat.
    #[test]
    fn listener_paels_tooth_after_combat_end_returns_one_upgraded_card() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::PaelsTooth);
        rs.paels_tooth_stored
            .push(deck_card(sts2_core::cards::CardId::Bash));
        rs.paels_tooth_stored
            .push(deck_card(sts2_core::cards::CardId::Anger));
        let deck_len_before = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.paels_tooth_stored.len(), 1, "one card drawn");
        assert_eq!(rs.master_deck.len(), deck_len_before + 1, "card returned");
        // Returned card is upgraded.
        let returned = rs.master_deck.last().expect("card pushed");
        assert!(returned.upgraded, "returned card upgraded");
    }

    /// after_combat_end is a no-op when storage is empty.
    #[test]
    fn listener_paels_tooth_after_combat_end_is_noop_when_storage_empty() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::PaelsTooth);
        assert!(rs.paels_tooth_stored.is_empty());
        let deck_len_before = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.master_deck.len(), deck_len_before);
    }

    /// after_combat_end is a no-op without ownership.
    #[test]
    fn listener_paels_tooth_after_combat_end_is_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.paels_tooth_stored
            .push(deck_card(sts2_core::cards::CardId::Bash));
        // No relic equipped.
        let deck_len_before = rs.master_deck.len();
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.master_deck.len(), deck_len_before);
        assert_eq!(rs.paels_tooth_stored.len(), 1, "storage untouched");
    }

    /// End-to-end: 5 cards stowed → 5 combats drain them one by one,
    /// returning upgraded cards to the deck. Empties storage cleanly.
    #[test]
    fn paels_tooth_pipeline_drains_storage_over_n_combats() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        for _ in 0..5 {
            rs.master_deck
                .push(deck_card(sts2_core::cards::CardId::Anger));
        }
        rs.relics.push(RelicId::PaelsTooth);
        rs.last_obtained_relic = Some(RelicId::PaelsTooth);
        let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
        assert_eq!(rs.paels_tooth_stored.len(), 5);
        assert_eq!(rs.master_deck.len(), 0);
        for combat in 1..=5 {
            let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
            assert_eq!(
                rs.paels_tooth_stored.len(),
                5 - combat,
                "after combat {combat}, {} cards remain",
                5 - combat
            );
            assert_eq!(rs.master_deck.len(), combat);
        }
        // 6th combat is a no-op.
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.master_deck.len(), 5);
        for c in &rs.master_deck {
            assert!(c.upgraded, "every returned card upgraded");
        }
    }

    // ----------------------------------------------------------------
    // HandDrill (after_damage_given block-break → Vulnerable(2))
    // ----------------------------------------------------------------

    /// Sum the Vulnerable amount on a creature row.
    fn creature_vulnerable(state: &CombatState, uid: CreatureUid) -> i32 {
        let idx = state
            .creatures
            .iter()
            .position(|c| c.id == uid.0)
            .expect("creature row");
        state.creatures[idx]
            .powers
            .iter()
            .filter_map(|pu| state.powers.get_by_id(pu.0))
            .filter(|p| p.kind == PowerKind::Vulnerable)
            .map(|p| p.amount)
            .sum()
    }

    #[test]
    fn hand_drill_applies_vulnerable_on_enemy_block_break() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::HandDrill);
        let player_uid = init_player_uid(&mut state);
        let enemy_uid = push_cultist(&mut state);
        // The player's Strike just broke the enemy's block.
        let ctx = sts2_core::relic_effects::DamageGivenCtx {
            dealer: player_uid,
            target: enemy_uid,
            damage: 7,
            props: ValueProps::NONE,
            was_block_broken: true,
        };
        let result = sts2_core::relic_effects::dispatch_after_damage_given(&mut state, &ctx);
        assert_eq!(result, EffectResult::Resolved);
        assert_eq!(creature_vulnerable(&state, enemy_uid), 2);
    }

    #[test]
    fn hand_drill_is_noop_without_block_break() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::HandDrill);
        let player_uid = init_player_uid(&mut state);
        let enemy_uid = push_cultist(&mut state);
        // Same hit but no block was broken — must not apply Vulnerable.
        let ctx = sts2_core::relic_effects::DamageGivenCtx {
            dealer: player_uid,
            target: enemy_uid,
            damage: 7,
            props: ValueProps::NONE,
            was_block_broken: false,
        };
        let _ = sts2_core::relic_effects::dispatch_after_damage_given(&mut state, &ctx);
        assert_eq!(creature_vulnerable(&state, enemy_uid), 0);
    }

    #[test]
    fn hand_drill_does_not_vulnerable_player_on_player_block_break() {
        // An enemy breaking the PLAYER's block must apply NO Vulnerable to
        // the player (C# `!target.IsPlayer`). The player-received bus also
        // passes `was_block_broken: false`, so this is doubly guarded; set
        // it `true` here to prove the explicit player guard holds.
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::HandDrill);
        let player_uid = init_player_uid(&mut state);
        let enemy_uid = push_cultist(&mut state);
        let ctx = sts2_core::relic_effects::DamageGivenCtx {
            dealer: enemy_uid,
            target: player_uid,
            damage: 7,
            props: ValueProps::NONE,
            was_block_broken: true,
        };
        let _ = sts2_core::relic_effects::dispatch_after_damage_given(&mut state, &ctx);
        assert_eq!(creature_vulnerable(&state, player_uid), 0);
    }

    // -----------------------------------------------------------------
    // Mutant-coverage tests (issue #65) — close surviving cargo-mutants
    // gaps in event_reward.rs.
    // -----------------------------------------------------------------

    /// `gain_block_player_unpowered` only dispatches the
    /// `after_block_gained` buses when `final_block > 0`. Juggernaut
    /// reacts to that bus by hitting a random enemy, so a block grant
    /// with Juggernaut installed must damage the enemy — this proves the
    /// guard fired the dispatch (kills the `> → ==` / `> → <` mutants at
    /// the `if final_block > 0` line).
    #[test]
    fn gain_block_player_unpowered_fires_after_block_gained_for_juggernaut() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeAnchor);
        let player_uid = init_player_uid(&mut state);
        let enemy = push_cultist(&mut state);
        state.enemy_uids.push(enemy);
        // Juggernaut(5) on the player reacts to after_block_gained by
        // dealing 5 unpowered damage to a random alive enemy.
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player_uid,
            PowerKind::Juggernaut,
            5,
        );
        let before = enemy_hp(&state, enemy);
        // FakeAnchor grants 4 block at combat start → final_block = 4 > 0
        // → after_block_gained dispatches → Juggernaut hits the enemy.
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        assert_eq!(state.player.block, 4, "block granted");
        assert_eq!(
            before - enemy_hp(&state, enemy),
            5,
            "after_block_gained fired → Juggernaut dealt 5 to the enemy"
        );
    }

    /// `FakeOrichalcum` gate is `side != Player || !owns`. On the enemy
    /// side it must NOT grant block even when the relic is owned and the
    /// VeryEarly snapshot is armed (kills the `|| → &&` mutant, which
    /// would fire on the enemy side). The pre-existing enemy-side test
    /// leaves the snapshot unset, so it does not pin this branch.
    #[test]
    fn listener_fake_orichalcum_noop_on_enemy_side_even_when_armed() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = init_player_uid(&mut state);
        state.combat_log.block_zero_at_turn_end = true;
        let _ = dispatch_before_turn_end(&mut state, Side::Enemy);
        assert_eq!(
            state.player.block, 0,
            "enemy-side turn-end must not grant FakeOrichalcum block"
        );
    }

    /// `TheBoot` only floors when `0 < amount < 5`. The `< 5` bound is
    /// strict: `amount == 5` stays 5 (already at the floor), but
    /// `amount == 4` is floored to 5 — distinguishing `< 5` from `<= 5`
    /// requires checking that an amount BELOW the floor is raised while
    /// the boundary value 5 is unchanged. The pre-existing test checks 5
    /// and 7 but the `< → <=` mutant at `amount < 5` survives because 5
    /// maps to 5 under both. Pin amount 4 → 5 explicitly here.
    #[test]
    fn listener_the_boot_floors_four_but_not_five() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::TheBoot);
        let player_uid = init_player_uid(&mut state);
        let target_uid = push_cultist(&mut state);
        let ctx = ModifyHpLostCtx {
            target: target_uid,
            dealer: Some(player_uid),
            props: ValueProps::MOVE,
        };
        // 4 < 5 → floored to 5.
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 4), 5);
        // 6 is above the floor and below nothing → unchanged. The
        // `< → <=`/`< → ==`/`< → >` mutants all change which amounts are
        // floored; 4→5 and 6→6 jointly constrain the predicate to `< 5`.
        assert_eq!(dispatch_modify_hp_lost(&state, &ctx, 6), 6);
    }

    /// A `HistoryCourse` dupe sourced from a live card instance
    /// (`src_uid = Some(uid)`) clones THAT instance — including a
    /// modified `base_damage` — rather than falling back to the spec
    /// POD. This pins the `uid.0.get() as usize - 1` index arithmetic
    /// (`- → +` / `- → /`) and the `card_id == card_id` instance filter
    /// (`== → !=`): any of those mutants mis-indexes or rejects the
    /// instance and falls back to the 6-damage spec Strike.
    #[test]
    fn listener_history_course_clones_live_instance_damage() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let enemy = history_course_setup(&mut state, 2);
        // A Strike whose instance base_damage was buffed to 20 (vs the
        // spec POD's 6). The dupe must clone this instance.
        let strike_uid = push_attack_card(&mut state);
        let idx = strike_uid.0.get() as usize - 1;
        state.cards[idx].base_damage = 20;
        let before = enemy_hp(&state, enemy);
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::StrikeIronclad, false, Some(strike_uid)));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            before - enemy_hp(&state, enemy),
            20,
            "dupe cloned the buffed live instance (20), not the spec POD (6)"
        );
    }

    /// A `HistoryCourse` Skill dupe counts as a Skill played this turn (C#
    /// records the dupe as a finished play, which MakeItSo and Pinpoint read).
    #[test]
    fn listener_history_course_skill_dupe_bumps_skill_tally() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let skill = sts2_core::cards::CardType::Skill.index();
        let before = state.player.cards_played_this_turn[skill];
        // Defend is a Skill; its dupe must bump the skill tally by 1.
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::DefendIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            state.player.cards_played_this_turn[skill],
            before + 1,
            "Skill dupe counts toward skills_played_this_turn"
        );
    }

    /// Under `Corruption`, a `HistoryCourse` Skill dupe is redirected to
    /// the Exhaust pile (Corruption exhausts every owner Skill). This
    /// pins the `corruption_exhausts_dupe` predicate:
    /// `dupe_card_type == Skill && player_power_amount(Corruption) > 0`.
    /// The `== → !=` (1175) and the `> → ==` / `> → <` (1176) mutants
    /// all suppress the exhaust, leaving the pile unchanged.
    #[test]
    fn listener_history_course_skill_dupe_exhausts_under_corruption() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let player_uid = state.player.creature_uid.expect("player uid");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player_uid,
            PowerKind::Corruption,
            1,
        );
        let before_exhaust = state.player.exhaust.len();
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::DefendIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            state.player.exhaust.len(),
            before_exhaust + 1,
            "Corruption sends the Skill dupe to the Exhaust pile"
        );
    }

    /// Without `Corruption`, a `HistoryCourse` Skill dupe is NOT
    /// exhausted (it is popped off the Play pile and discarded). Pins
    /// the `&&` join (`&& → ||` at 1176:9 would exhaust on the Skill
    /// gate alone) and the `> → >=` mutant (1176:72, which would treat a
    /// zero Corruption stack as "present").
    #[test]
    fn listener_history_course_skill_dupe_not_exhausted_without_corruption() {
        use sts2_core::cards::CardId;
        crate::relics::dispatch::ensure_installed();
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        let _enemy = history_course_setup(&mut state, 2);
        let before_exhaust = state.player.exhaust.len();
        state.combat_log.prev_round_last_attack_skill_played =
            Some((CardId::DefendIronclad, false, None));
        let _ = sts2_core::relic_effects::dispatch_before_play_phase_start(&mut state);
        assert_eq!(
            state.player.exhaust.len(),
            before_exhaust,
            "no Corruption → the Skill dupe is not exhausted"
        );
    }

    /// `FakeBloodVial` gate is `side != Player || !owns`. On the enemy
    /// side it must NOT heal even when owned and on round 1 (kills the
    /// `|| → &&` mutant, which would heal on the enemy side).
    #[test]
    fn listener_fake_blood_vial_noop_on_enemy_side() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::FakeBloodVial);
        state.player.current_hp = 50;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        state.sync_player_creature();
        state.round = 1;
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(
            state.player.current_hp, 50,
            "enemy-side turn-start must not heal FakeBloodVial"
        );
    }

    /// `RoyalPoison.cs:18` is `AfterPlayerTurnStart` (both builds), which
    /// runs inside `SetupPlayerTurn`, before every `AfterPlayerTurnStartLate`
    /// (BloodVial.cs:17) and before `AfterSideTurnStart`. So a Blood Vial
    /// obtained FIRST still heals after the poison, instead of no-opping at
    /// full HP: 80 -> 76 -> 78. The side-turn-start bus (either side) no
    /// longer fires the poison, and Blood Vial runs in the Late pass.
    #[test]
    fn royal_poison_fires_on_player_turn_start_before_blood_vial() {
        crate::relics::dispatch::ensure_installed();
        for v in [
            sts2_core::GameVersion::V0_107_1,
            sts2_core::GameVersion::V0_111_0,
        ] {
            let mut state = CombatState {
                game_version: v,
                ..CombatState::default()
            };
            push_relic(&mut state, RelicKind::BloodVial);
            push_relic(&mut state, RelicKind::RoyalPoison);
            state.player.current_hp = 80;
            state.player.max_hp = 80;
            let _ = state.init_player_creature();
            state.sync_player_creature();
            state.round = 1;
            let _ =
                sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Enemy);
            assert_eq!(state.player.current_hp, 80, "{v:?}: not on side turn start");
            let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
            assert_eq!(state.player.current_hp, 76, "{v:?}");
            sts2_core::relic_effects::blood_vial::dispatch_after_player_turn_start_late(&mut state);
            assert_eq!(state.player.current_hp, 78, "{v:?}");
        }
    }

    /// `FakeHappyFlower` grants 1 energy on every 5th player turn. With a
    /// pre-seeded counter of 4 the post-increment lands on 5 → fire.
    /// Pins the ownership-found branch (`!found` delete, `relic.kind ==`
    /// at 1360), the `side != Player` gate (1352), and the firing
    /// predicate `new_count <= 0 || new_count % 5 != 0` (the `<=`/`||`/
    /// `%`/`!=` mutants at 1370 all suppress this fire).
    #[test]
    fn listener_fake_happy_flower_grants_energy_on_fifth_turn() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::FakeHappyFlower, 4);
        let _ = init_player_uid(&mut state);
        state.player.energy = 0;
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, 1, "counter 4 → 5 fires +1 energy");
    }

    /// `FakeHappyFlower` does NOT grant energy on a non-multiple-of-5
    /// turn. With a counter of 0 the post-increment lands on 1 → no
    /// fire. This is the negative half that pins the `<= → >` (1370:18)
    /// and `|| → &&` (1370:23) mutants, which would fire on turn 1.
    #[test]
    fn listener_fake_happy_flower_no_energy_off_period() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::FakeHappyFlower, 0);
        let _ = init_player_uid(&mut state);
        state.player.energy = 0;
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Player);
        assert_eq!(state.player.energy, 0, "counter 0 → 1 does not fire");
    }

    /// `FakeHappyFlower` is a no-op on the enemy side (pins the
    /// `side != Player` gate at 1352, `!= → ==`, which would skip the
    /// player side and fire on the enemy side). Counter pre-seeded to 4
    /// so a mis-gated fire would be observable as +1 energy.
    #[test]
    fn listener_fake_happy_flower_noop_on_enemy_side() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::FakeHappyFlower, 4);
        let _ = init_player_uid(&mut state);
        state.player.energy = 0;
        let _ = sts2_core::relic_effects::dispatch_after_side_turn_start(&mut state, Side::Enemy);
        assert_eq!(
            state.player.energy, 0,
            "enemy-side turn-start must not tick or fire FakeHappyFlower"
        );
    }

    /// `TeaOfDiscourtesy` spawns 2 Dazed into the draw pile when its
    /// counter is positive, then decrements it to 0. Pins the
    /// `counter > 0` gate (1399), the `-= ` decrement (1400, where
    /// `+= ` would leave the counter non-zero and `/= 0` would panic),
    /// and the `!fired` early-return (1405).
    #[test]
    fn listener_tea_of_discourtesy_spawns_two_dazed_and_decrements() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::TeaOfDiscourtesy, 1);
        let _ = init_player_uid(&mut state);
        let before = state.player.draw.len();
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        assert_eq!(
            state.player.draw.len(),
            before + 2,
            "two Dazed spawned into the draw pile"
        );
        let counter = state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::TeaOfDiscourtesy)
            .expect("relic present")
            .counter;
        assert_eq!(counter, 0, "counter decremented 1 → 0");
    }

    /// `TeaOfDiscourtesy` is a no-op once its counter is 0 (pins the
    /// `counter > 0` gate's `> → >=` mutant, which would fire at 0).
    #[test]
    fn listener_tea_of_discourtesy_noop_at_zero_counter() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::TeaOfDiscourtesy, 0);
        let _ = init_player_uid(&mut state);
        let before = state.player.draw.len();
        let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
        assert_eq!(
            state.player.draw.len(),
            before,
            "drained counter must not spawn Dazed"
        );
    }

    /// `ChosenCheese::after_combat_end` grants +1 max HP. Pins the
    /// `!run_owns` ownership gate (1508 `delete !`, which would invert
    /// the gate and skip the grant when owned).
    #[test]
    fn listener_chosen_cheese_gains_one_max_hp_on_combat_end() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::ChosenCheese);
        let before = rs.max_hp;
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.max_hp, before + 1, "ChosenCheese grants +1 max HP");
    }

    /// `ChosenCheese` is a no-op without ownership (the other half of
    /// the `delete !` gate at 1508).
    #[test]
    fn listener_chosen_cheese_noop_without_ownership() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        let before = rs.max_hp;
        let _ = sts2_core::run_effects::dispatch_after_combat_end(&mut rs);
        assert_eq!(rs.max_hp, before, "no ChosenCheese → no max HP gain");
    }

    /// `DarkstonePeriapt::on_card_added_to_deck` grants +6 max HP when
    /// the added card is a Curse. Pins the `!run_owns` ownership gate
    /// (1526 `delete !`) and the `ctx.card_type != Curse` gate (1529
    /// `!= → ==`, which would only fire on non-Curse adds).
    #[test]
    fn listener_darkstone_periapt_gains_six_max_hp_on_curse_add() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::DarkstonePeriapt);
        let before = rs.max_hp;
        let ctx = CardAddedRunCtx {
            // The listener gates on `card_type`, not `card_id`.
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            card_type: CardType::Curse,
            source: sts2_core::relic_effects::AddSource::Reward,
        };
        let _ = sts2_core::run_effects::dispatch_on_card_added_to_deck(&mut rs, &ctx);
        assert_eq!(rs.max_hp, before + 6, "Curse add grants +6 max HP");
    }

    /// `DarkstonePeriapt` is a no-op when the added card is NOT a Curse
    /// (the other half of the `!= → ==` gate at 1529).
    #[test]
    fn listener_darkstone_periapt_noop_on_non_curse_add() {
        crate::relics::dispatch::ensure_installed();
        let mut rs = RunState::new();
        rs.relics.push(RelicId::DarkstonePeriapt);
        let before = rs.max_hp;
        let ctx = CardAddedRunCtx {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            card_type: CardType::Attack,
            source: sts2_core::relic_effects::AddSource::Reward,
        };
        let _ = sts2_core::run_effects::dispatch_on_card_added_to_deck(&mut rs, &ctx);
        assert_eq!(rs.max_hp, before, "non-Curse add must not grant max HP");
    }

    /// `SwordOfStone::after_combat_victory` resolves (returns
    /// `EffectResult::Resolved`) when it fires on an elite victory, and
    /// returns `NotImplemented` otherwise. Pins the `!fired` branch at
    /// 2027 (`delete !`, which inverts the resolved/not-implemented
    /// return value). The counter-decrement is already covered; this
    /// asserts the return discriminant the `delete !` mutant flips.
    #[test]
    fn listener_sword_of_stone_returns_resolved_when_firing() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic_with_counter(&mut state, RelicKind::SwordOfStone, 5);
        state.is_elite = true;
        assert_eq!(
            listener_sword_of_stone(&mut state),
            EffectResult::Resolved,
            "elite victory with a live counter resolves"
        );
        // Non-firing (counter drained) → NotImplemented.
        assert_eq!(
            listener_sword_of_stone(&mut state),
            EffectResult::Resolved,
            "still resolves while counter > 0"
        );
        // Drain to 0, then the next elite victory does not fire.
        for relic in &mut state.relics {
            if relic.kind == RelicKind::SwordOfStone {
                relic.counter = 0;
            }
        }
        assert_eq!(
            listener_sword_of_stone(&mut state),
            EffectResult::NotImplemented,
            "drained counter does not fire → NotImplemented"
        );
    }

    /// `BingBong::on_card_added_to_deck` resolves when it duplicates a
    /// freshly-added card. The pre-existing tests assert the deck delta;
    /// this pins the `!player_owns` return discriminant via the
    /// `dispatch_on_card_added_to_deck` aggregate result (the `delete !`
    /// mutant at 2189 inverts owned/unowned resolution). Driven through
    /// the combat-side typed channel with a non-duplicate source.
    #[test]
    fn listener_bing_bong_resolves_on_owned_reward_add() {
        crate::relics::dispatch::ensure_installed();
        let mut state = CombatState::default();
        push_relic(&mut state, RelicKind::BingBong);
        // Seed `last_added_card` so `duplicate_last_added_card` has a
        // target to clone.
        state
            .master_deck
            .push(sts2_core::cards::CardId::StrikeIronclad);
        state.last_added_card = Some(sts2_core::cards::CardId::StrikeIronclad);
        let ctx = CardAddedCtx {
            card_id: sts2_core::cards::CardId::StrikeIronclad,
            source: sts2_core::relic_effects::AddSource::Reward,
        };
        let result = sts2_core::relic_effects::dispatch_on_card_added_to_deck(&mut state, &ctx);
        assert_eq!(
            result,
            EffectResult::Resolved,
            "owned BingBong duplicates and resolves"
        );
    }
}
