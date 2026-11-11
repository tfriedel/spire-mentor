//! Listener-fire smoke tests for the rich-ctx relic-bus hook channels
//! whose payload types do not implement `Default`. Issue #53
//! (`sts2-rs-ri9um`-extension) extends the earlier 16 smoke tests
//! (`sts2-rs-y4ltf` + `sts2-rs-ri9um`) to cover ~24 additional channels
//! that need realistic ctx fixtures (`CreatureUid`, `CardId`,
//! `ValueProps`, …).
//!
//! Each test:
//! 1. Builds a minimal `CombatState` with one player creature, one
//!    monster creature, and one card in the player's hand.
//! 2. Registers a counting listener via the channel's public
//!    `register_*` API under a distinct `RelicId` so it does not
//!    overwrite any other test's listener on the same channel.
//! 3. Dispatches the channel once with a real ctx built from the
//!    fixture and asserts the counter incremented exactly once.
//!
//! These tests act as a regression guard for the empty-bus early-out
//! optimization (bead `sts2-rs-4lk9`): if a future refactor accidentally
//! short-circuits `dispatch_*` before invoking a registered listener
//! the corresponding smoke test fails.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::SmallVec;

use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
use sts2_core::hooks::{
    BlockClearedCtx, BlockCtx, CardCtx, CreatureRef, DamageCtx, DamageMul, PowerAmountCtx,
    StarsSpentCtx, ValueProps,
};
use sts2_core::potions::PotionKind;
use sts2_core::relic_effects::{
    dispatch_after_block_cleared, dispatch_after_block_gained, dispatch_after_card_discarded,
    dispatch_after_card_entered_combat, dispatch_after_card_exhausted, dispatch_after_card_played,
    dispatch_after_combat_end, dispatch_after_combat_victory, dispatch_after_damage_given,
    dispatch_after_damage_received, dispatch_after_energy_reset, dispatch_after_hand_emptied,
    dispatch_after_max_hp_changed, dispatch_after_max_hp_gained,
    dispatch_after_modifying_power_amount_given, dispatch_after_modifying_power_amount_received,
    dispatch_after_obtained, dispatch_after_potion_used, dispatch_after_room_entered,
    dispatch_after_shuffle_bus, dispatch_after_side_turn_start, dispatch_after_turn_end,
    dispatch_before_card_played, dispatch_before_combat_start,
    dispatch_before_power_amount_changed, dispatch_before_side_turn_start,
    dispatch_before_turn_end, dispatch_count_pets, dispatch_modify_block_multiplicative,
    dispatch_modify_card_reward_options_late, dispatch_modify_damage_additive,
    dispatch_modify_damage_multiplicative, dispatch_modify_hand_draw, dispatch_modify_hp_lost,
    dispatch_modify_power_amount_given, dispatch_modify_power_amount_received,
    dispatch_modify_rest_site_heal_rewards, dispatch_modify_rewards, dispatch_modify_shop_price,
    dispatch_on_card_added_to_deck, dispatch_on_orb_channeled, dispatch_on_pet_summoned,
    dispatch_on_stars_spent, dispatch_should_clear_block, register_after_block_cleared,
    register_after_block_gained, register_after_card_discarded, register_after_card_entered_combat,
    register_after_card_exhausted, register_after_card_played, register_after_combat_end,
    register_after_combat_victory, register_after_damage_given, register_after_damage_received,
    register_after_energy_reset, register_after_hand_emptied, register_after_max_hp_changed,
    register_after_max_hp_gained, register_after_modifying_power_amount_given,
    register_after_modifying_power_amount_received, register_after_obtained,
    register_after_potion_used, register_after_room_entered, register_after_shuffle_bus,
    register_after_side_turn_start, register_after_turn_end, register_before_card_played,
    register_before_combat_start, register_before_power_amount_changed,
    register_before_side_turn_start, register_before_turn_end, register_count_pets,
    register_modify_block_multiplicative, register_modify_card_reward_options_late,
    register_modify_damage_additive, register_modify_damage_multiplicative,
    register_modify_hand_draw, register_modify_hp_lost, register_modify_power_amount_given,
    register_modify_power_amount_received, register_modify_rest_site_heal_rewards,
    register_modify_rewards, register_modify_shop_price, register_on_card_added_to_deck,
    register_on_orb_channeled, register_on_pet_summoned, register_on_stars_spent,
    register_should_clear_block, AddSource, AfterBlockClearedFn, AfterBlockGainedFn,
    AfterCardDiscardedFn, AfterCardEnteredCombatFn, AfterCardExhaustedFn, AfterCardPlayedFn,
    AfterCombatEndFn, AfterCombatVictoryFn, AfterDamageGivenFn, AfterDamageReceivedFn,
    AfterEnergyResetFn, AfterHandEmptiedFn, AfterMaxHpChangedFn, AfterMaxHpGainedFn,
    AfterModifyingPowerAmountGivenFn, AfterModifyingPowerAmountReceivedFn, AfterObtainedFn,
    AfterPotionUsedFn, AfterRoomEnteredFn, AfterShuffleFn, AfterSideTurnStartFn, AfterTurnEndFn,
    BeforeCardPlayedFn, BeforeCombatStartFn, BeforePowerAmountChangedFn, BeforeSideTurnStartFn,
    BeforeTurnEndFn, CardAddedCtx, CardEventCtx, CountPetsFn, DamageGivenCtx, MaxHpChangeCtx,
    ModifyBlockMultiplicativeFn, ModifyCardRewardOptionsLateFn, ModifyDamageAdditiveFn,
    ModifyDamageCtx, ModifyDamageMultiplicativeFn, ModifyHandDrawFn, ModifyHpLostCtx,
    ModifyHpLostFn, ModifyPowerAmountGivenFn, ModifyPowerAmountReceivedFn,
    ModifyRestSiteHealRewardsFn, ModifyRewardsFn, ModifyShopPriceFn, OnCardAddedToDeckFn,
    OnOrbChanneledFn, OnPetSummonedFn, OnStarsSpentFn, PetKind, PetSummonCtx, Reward, ShopItemKind,
    ShopPriceCtx, ShouldClearBlockFn, ShuffleCtx,
};
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
    MonsterKind, PowerKind, RelicId, Side,
};

// ---------------------------------------------------------------------------
// Shared fixture: one player Creature, one Nibbit monster, one Strike card.
// ---------------------------------------------------------------------------

struct HookSmokeFixture {
    state: CombatState,
    player_uid: CreatureUid,
    enemy_uid: CreatureUid,
    a_card_uid: CardUid,
    a_card_id: CardId,
}

fn build_hook_smoke_fixture() -> HookSmokeFixture {
    let mut state = CombatState {
        phase: CombatPhase::PlayerPlayPhase,
        round: 1,
        ..CombatState::default()
    };
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 3;
    state.player.max_energy = 3;

    let player_uid = state.init_player_creature();

    let enemy_id = NonZeroU32::new(2).expect("non-zero");
    let enemy_uid = CreatureUid(enemy_id);
    state.creatures.push(Creature {
        id: enemy_id,
        current_hp: 48,
        max_hp: 48,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(enemy_uid);

    let a_card_uid = CardUid::new(1).expect("non-zero");
    let a_card_id = CardId::StrikeIronclad;
    state.cards.push(Card {
        uid: a_card_uid,
        card_id: a_card_id,
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
    state.player.hand.push(a_card_uid);

    // Bead `sts2-rs-b8z17`: the relic dispatchers now fire only OWNED relics'
    // listeners (handlers self-check `player_owns_relic`). Every `*_listener_fires`
    // smoke test registers its sentinel handler on `RelicId::DeprecatedRelic`, so
    // equip it here once. `DeprecatedRelic` has no real content handler, so owning
    // it is inert for the power-side / non-relic smoke tests that share this fixture.
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::DeprecatedRelic,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });

    HookSmokeFixture {
        state,
        player_uid,
        enemy_uid,
        a_card_uid,
        a_card_id,
    }
}

// ---------------------------------------------------------------------------
// Per-ctx builders.
// ---------------------------------------------------------------------------

fn make_card_event_ctx(f: &HookSmokeFixture) -> CardEventCtx {
    CardEventCtx {
        card_id: f.a_card_id,
        card_uid: Some(f.a_card_uid),
        target: Some(f.enemy_uid),
        card_tags: CardTags::NONE,
        energy_spent: 0,
    }
}

fn make_card_ctx(f: &HookSmokeFixture) -> CardCtx {
    CardCtx {
        card_id: Some(f.a_card_id),
        card_uid: Some(f.a_card_uid),
        card_tags: CardTags::NONE,
        card_type: Some(CardType::Attack),
    }
}

fn make_damage_given_ctx(f: &HookSmokeFixture) -> DamageGivenCtx {
    DamageGivenCtx {
        dealer: f.player_uid,
        target: f.enemy_uid,
        damage: 6,
        props: ValueProps::NONE,
        was_block_broken: false,
    }
}

fn make_damage_ctx(f: &HookSmokeFixture) -> DamageCtx {
    DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(f.enemy_uid),
        props: ValueProps::NONE,
    }
}

fn make_modify_damage_ctx(f: &HookSmokeFixture) -> ModifyDamageCtx {
    ModifyDamageCtx {
        dealer: Some(f.player_uid),
        target: f.enemy_uid,
        base_damage: 6,
        props: ValueProps::NONE,
        card: Some(f.a_card_id),
        card_uid: None,
    }
}

fn make_modify_hp_lost_ctx(f: &HookSmokeFixture) -> ModifyHpLostCtx {
    ModifyHpLostCtx {
        target: f.player_uid,
        dealer: Some(f.enemy_uid),
        props: ValueProps::NONE,
    }
}

fn make_max_hp_change_ctx(f: &HookSmokeFixture) -> MaxHpChangeCtx {
    MaxHpChangeCtx {
        owner: f.player_uid,
        delta: 4,
    }
}

fn make_power_amount_ctx() -> PowerAmountCtx {
    PowerAmountCtx::new(
        Some(CreatureRef::Player),
        CreatureRef::Player,
        PowerKind::Strength,
        2,
    )
}

fn make_card_added_ctx(f: &HookSmokeFixture) -> CardAddedCtx {
    CardAddedCtx {
        card_id: f.a_card_id,
        source: AddSource::Reward,
    }
}

// ---------------------------------------------------------------------------
// Tests — one per channel. Each uses a distinct `RelicId` so the
// process-wide registries don't clobber across tests.
// ---------------------------------------------------------------------------

static C_AFTER_CARD_PLAYED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_card_played(
    _state: &mut CombatState,
    _ctx: &CardEventCtx,
) -> sts2_core::EffectResult {
    C_AFTER_CARD_PLAYED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_card_played_listener_fires() {
    C_AFTER_CARD_PLAYED.store(0, Ordering::SeqCst);
    register_after_card_played(
        RelicId::DeprecatedRelic,
        listener_after_card_played as AfterCardPlayedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_card_event_ctx(&f);
    let _ = dispatch_after_card_played(&mut f.state, &ctx);
    assert_eq!(C_AFTER_CARD_PLAYED.load(Ordering::SeqCst), 1);
}

static C_AFTER_CARD_EXHAUSTED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_card_exhausted(
    _state: &mut CombatState,
    _ctx: &CardEventCtx,
) -> sts2_core::EffectResult {
    C_AFTER_CARD_EXHAUSTED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_card_exhausted_listener_fires() {
    C_AFTER_CARD_EXHAUSTED.store(0, Ordering::SeqCst);
    register_after_card_exhausted(
        RelicId::DeprecatedRelic,
        listener_after_card_exhausted as AfterCardExhaustedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_card_event_ctx(&f);
    let _ = dispatch_after_card_exhausted(&mut f.state, &ctx);
    assert_eq!(C_AFTER_CARD_EXHAUSTED.load(Ordering::SeqCst), 1);
}

static C_AFTER_CARD_DISCARDED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_card_discarded(
    _state: &mut CombatState,
    _ctx: &CardCtx,
) -> sts2_core::EffectResult {
    C_AFTER_CARD_DISCARDED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_card_discarded_listener_fires() {
    C_AFTER_CARD_DISCARDED.store(0, Ordering::SeqCst);
    register_after_card_discarded(
        RelicId::DeprecatedRelic,
        listener_after_card_discarded as AfterCardDiscardedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_card_ctx(&f);
    let _ = dispatch_after_card_discarded(&mut f.state, &ctx);
    assert_eq!(C_AFTER_CARD_DISCARDED.load(Ordering::SeqCst), 1);
}

static C_AFTER_DAMAGE_GIVEN: AtomicUsize = AtomicUsize::new(0);
fn listener_after_damage_given(
    _state: &mut CombatState,
    _ctx: &DamageGivenCtx,
) -> sts2_core::EffectResult {
    C_AFTER_DAMAGE_GIVEN.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_damage_given_listener_fires() {
    C_AFTER_DAMAGE_GIVEN.store(0, Ordering::SeqCst);
    register_after_damage_given(
        RelicId::DeprecatedRelic,
        listener_after_damage_given as AfterDamageGivenFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_damage_given_ctx(&f);
    let _ = dispatch_after_damage_given(&mut f.state, &ctx);
    assert_eq!(C_AFTER_DAMAGE_GIVEN.load(Ordering::SeqCst), 1);
}

static C_AFTER_DAMAGE_RECEIVED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_damage_received(
    _state: &mut CombatState,
    _ctx: &DamageCtx,
    _damage: i32,
) -> sts2_core::EffectResult {
    C_AFTER_DAMAGE_RECEIVED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_damage_received_listener_fires() {
    C_AFTER_DAMAGE_RECEIVED.store(0, Ordering::SeqCst);
    register_after_damage_received(
        RelicId::DeprecatedRelic,
        listener_after_damage_received as AfterDamageReceivedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_damage_ctx(&f);
    let _ = dispatch_after_damage_received(&mut f.state, &ctx, 5);
    assert_eq!(C_AFTER_DAMAGE_RECEIVED.load(Ordering::SeqCst), 1);
}

static C_MODIFY_DAMAGE_ADDITIVE: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_damage_additive(_state: &CombatState, _ctx: &ModifyDamageCtx) -> i32 {
    C_MODIFY_DAMAGE_ADDITIVE.fetch_add(1, Ordering::SeqCst);
    0
}
#[test]
fn modify_damage_additive_listener_fires() {
    C_MODIFY_DAMAGE_ADDITIVE.store(0, Ordering::SeqCst);
    register_modify_damage_additive(
        RelicId::DeprecatedRelic,
        listener_modify_damage_additive as ModifyDamageAdditiveFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = make_modify_damage_ctx(&f);
    let _ = dispatch_modify_damage_additive(&f.state, &ctx);
    assert_eq!(C_MODIFY_DAMAGE_ADDITIVE.load(Ordering::SeqCst), 1);
}

static C_AFTER_SIDE_TURN_START: AtomicUsize = AtomicUsize::new(0);
fn listener_after_side_turn_start(
    _state: &mut CombatState,
    _side: Side,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_SIDE_TURN_START.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-cptkn`: smoke-cover dispatch_after_side_turn_start.
/// Same Side-enum signature as before_turn_end (2n8ho).
#[test]
fn after_side_turn_start_listener_fires() {
    C_AFTER_SIDE_TURN_START.store(0, Ordering::SeqCst);
    register_after_side_turn_start(
        RelicId::DeprecatedRelic,
        listener_after_side_turn_start as AfterSideTurnStartFn,
    );
    // `build_hook_smoke_fixture` equips `DeprecatedRelic` (bead `sts2-rs-b8z17`),
    // which the relic dispatchers require to fire an owned relic's listener.
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_side_turn_start(&mut f.state, Side::Player);
    assert_eq!(C_AFTER_SIDE_TURN_START.load(Ordering::SeqCst), 1);
}

static C_BEFORE_SIDE_TURN_START: AtomicUsize = AtomicUsize::new(0);
fn listener_before_side_turn_start(
    _state: &mut CombatState,
    _side: Side,
) -> sts2_core::card_effects::EffectResult {
    C_BEFORE_SIDE_TURN_START.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-cvugd`: smoke-cover dispatch_before_side_turn_start.
/// Same Side-enum signature as before_turn_end (2n8ho) and
/// after_side_turn_start (cptkn).
#[test]
fn before_side_turn_start_listener_fires() {
    C_BEFORE_SIDE_TURN_START.store(0, Ordering::SeqCst);
    register_before_side_turn_start(
        RelicId::DeprecatedRelic,
        listener_before_side_turn_start as BeforeSideTurnStartFn,
    );
    // `build_hook_smoke_fixture` equips `DeprecatedRelic` (bead `sts2-rs-b8z17`,
    // formerly pushed inline per m4jaj), which the relic dispatcher requires to
    // fire an owned relic's listener.
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_before_side_turn_start(&mut f.state, Side::Player);
    assert_eq!(C_BEFORE_SIDE_TURN_START.load(Ordering::SeqCst), 1);
}

static C_BEFORE_TURN_END: AtomicUsize = AtomicUsize::new(0);
fn listener_before_turn_end(
    _state: &mut CombatState,
    _side: Side,
) -> sts2_core::card_effects::EffectResult {
    C_BEFORE_TURN_END.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-2n8ho`: smoke-cover dispatch_before_turn_end.
/// Signature: fn(&mut CombatState, Side) -> EffectResult.
#[test]
fn before_turn_end_listener_fires() {
    C_BEFORE_TURN_END.store(0, Ordering::SeqCst);
    register_before_turn_end(
        RelicId::DeprecatedRelic,
        listener_before_turn_end as BeforeTurnEndFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_before_turn_end(&mut f.state, Side::Player);
    assert_eq!(C_BEFORE_TURN_END.load(Ordering::SeqCst), 1);
}

static C_AFTER_ENERGY_RESET: AtomicUsize = AtomicUsize::new(0);
fn listener_after_energy_reset(_state: &mut CombatState) -> sts2_core::card_effects::EffectResult {
    C_AFTER_ENERGY_RESET.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-mc80v`: smoke-cover dispatch_after_energy_reset.
/// No-payload signature.
#[test]
fn after_energy_reset_listener_fires() {
    C_AFTER_ENERGY_RESET.store(0, Ordering::SeqCst);
    register_after_energy_reset(
        RelicId::DeprecatedRelic,
        listener_after_energy_reset as AfterEnergyResetFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_energy_reset(&mut f.state);
    assert_eq!(C_AFTER_ENERGY_RESET.load(Ordering::SeqCst), 1);
}

static C_AFTER_HAND_EMPTIED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_hand_emptied(
    _state: &mut CombatState,
    _ctx: &sts2_core::hooks::CombatLifecycleCtx,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_HAND_EMPTIED.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-j8buj`: smoke-cover dispatch_after_hand_emptied.
/// Signature: fn(&mut CombatState, &CombatLifecycleCtx) -> EffectResult.
#[test]
fn after_hand_emptied_listener_fires() {
    C_AFTER_HAND_EMPTIED.store(0, Ordering::SeqCst);
    register_after_hand_emptied(
        RelicId::DeprecatedRelic,
        listener_after_hand_emptied as AfterHandEmptiedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = sts2_core::hooks::CombatLifecycleCtx::default();
    let _ = dispatch_after_hand_emptied(&mut f.state, &ctx);
    assert_eq!(C_AFTER_HAND_EMPTIED.load(Ordering::SeqCst), 1);
}

static C_AFTER_COMBAT_END: AtomicUsize = AtomicUsize::new(0);
fn listener_after_combat_end(_state: &mut CombatState) -> sts2_core::card_effects::EffectResult {
    C_AFTER_COMBAT_END.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-quhr2`: smoke-cover dispatch_after_combat_end.
/// Same no-payload signature as before_combat_start (4ek41).
#[test]
fn after_combat_end_listener_fires() {
    C_AFTER_COMBAT_END.store(0, Ordering::SeqCst);
    register_after_combat_end(
        RelicId::DeprecatedRelic,
        listener_after_combat_end as AfterCombatEndFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_combat_end(&mut f.state);
    assert_eq!(C_AFTER_COMBAT_END.load(Ordering::SeqCst), 1);
}

static C_AFTER_COMBAT_VICTORY: AtomicUsize = AtomicUsize::new(0);
fn listener_after_combat_victory(
    _state: &mut CombatState,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_COMBAT_VICTORY.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-5d27m`: smoke-cover dispatch_after_combat_victory.
/// Same no-payload signature as after_combat_end (quhr2) and
/// before_combat_start (4ek41). Production listener: BurningBlood.
#[test]
fn after_combat_victory_listener_fires() {
    C_AFTER_COMBAT_VICTORY.store(0, Ordering::SeqCst);
    register_after_combat_victory(
        RelicId::DeprecatedRelic,
        listener_after_combat_victory as AfterCombatVictoryFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_combat_victory(&mut f.state);
    assert_eq!(C_AFTER_COMBAT_VICTORY.load(Ordering::SeqCst), 1);
}

static C_BEFORE_COMBAT_START: AtomicUsize = AtomicUsize::new(0);
fn listener_before_combat_start(_state: &mut CombatState) -> sts2_core::card_effects::EffectResult {
    C_BEFORE_COMBAT_START.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-4ek41`: smoke-cover dispatch_before_combat_start.
/// Simplest signature in the hook family — fn(&mut CombatState).
#[test]
fn before_combat_start_listener_fires() {
    C_BEFORE_COMBAT_START.store(0, Ordering::SeqCst);
    register_before_combat_start(
        RelicId::DeprecatedRelic,
        listener_before_combat_start as BeforeCombatStartFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_before_combat_start(&mut f.state);
    assert_eq!(C_BEFORE_COMBAT_START.load(Ordering::SeqCst), 1);
}

static C_AFTER_CARD_ENTERED_COMBAT: AtomicUsize = AtomicUsize::new(0);
fn listener_after_card_entered_combat(
    _state: &mut CombatState,
    _card_uid: CardUid,
    _added_by_player: bool,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_CARD_ENTERED_COMBAT.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-35yrr`: smoke-cover dispatch_after_card_entered_combat.
/// Sibling of after_potion_used (x9hy2). Fires from spawn-card paths
/// after the card has been arena-allocated.
#[test]
fn after_card_entered_combat_listener_fires() {
    C_AFTER_CARD_ENTERED_COMBAT.store(0, Ordering::SeqCst);
    register_after_card_entered_combat(
        RelicId::DeprecatedRelic,
        listener_after_card_entered_combat as AfterCardEnteredCombatFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_card_entered_combat(&mut f.state, f.a_card_uid, true);
    assert_eq!(C_AFTER_CARD_ENTERED_COMBAT.load(Ordering::SeqCst), 1);
}

static C_AFTER_POTION_USED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_potion_used(
    _state: &mut CombatState,
    _kind: PotionKind,
    _target: Option<u8>,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_POTION_USED.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-x9hy2`: smoke-cover dispatch_after_potion_used.
/// Production listener fires from Action::DrinkPotion after the
/// potion body resolves. Was uncovered in the smoke suite.
#[test]
fn after_potion_used_listener_fires() {
    C_AFTER_POTION_USED.store(0, Ordering::SeqCst);
    register_after_potion_used(
        RelicId::DeprecatedRelic,
        listener_after_potion_used as AfterPotionUsedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_potion_used(&mut f.state, PotionKind::BlockPotion, None);
    assert_eq!(C_AFTER_POTION_USED.load(Ordering::SeqCst), 1);
}

static C_MODIFY_DAMAGE_MULT: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_damage_multiplicative(_state: &CombatState, _ctx: &DamageCtx) -> DamageMul {
    C_MODIFY_DAMAGE_MULT.fetch_add(1, Ordering::SeqCst);
    DamageMul::ONE
}
/// Bead `sts2-rs-h3agt`: smoke-cover dispatch_modify_damage_multiplicative.
/// Sibling of modify_damage_additive (line 334) — both feed into the
/// damage pipeline but the multiplicative leg was missing.
#[test]
fn modify_damage_multiplicative_listener_fires() {
    C_MODIFY_DAMAGE_MULT.store(0, Ordering::SeqCst);
    register_modify_damage_multiplicative(
        RelicId::DeprecatedRelic,
        listener_modify_damage_multiplicative as ModifyDamageMultiplicativeFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(f.enemy_uid),
        props: ValueProps::MOVE,
    };
    let _ = dispatch_modify_damage_multiplicative(&f.state, &ctx);
    assert_eq!(C_MODIFY_DAMAGE_MULT.load(Ordering::SeqCst), 1);
}

static C_MODIFY_HP_LOST: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_hp_lost(_state: &CombatState, _ctx: &ModifyHpLostCtx, current: i32) -> i32 {
    C_MODIFY_HP_LOST.fetch_add(1, Ordering::SeqCst);
    current
}
#[test]
fn modify_hp_lost_listener_fires() {
    C_MODIFY_HP_LOST.store(0, Ordering::SeqCst);
    register_modify_hp_lost(
        RelicId::DeprecatedRelic,
        listener_modify_hp_lost as ModifyHpLostFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = make_modify_hp_lost_ctx(&f);
    let _ = dispatch_modify_hp_lost(&f.state, &ctx, 4);
    assert_eq!(C_MODIFY_HP_LOST.load(Ordering::SeqCst), 1);
}

static C_MODIFY_HAND_DRAW: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_hand_draw(_state: &CombatState, current: i32) -> i32 {
    C_MODIFY_HAND_DRAW.fetch_add(1, Ordering::SeqCst);
    current
}
#[test]
fn modify_hand_draw_listener_fires() {
    C_MODIFY_HAND_DRAW.store(0, Ordering::SeqCst);
    register_modify_hand_draw(
        RelicId::DeprecatedRelic,
        listener_modify_hand_draw as ModifyHandDrawFn,
    );
    let f = build_hook_smoke_fixture();
    let _ = dispatch_modify_hand_draw(&f.state, 5);
    assert_eq!(C_MODIFY_HAND_DRAW.load(Ordering::SeqCst), 1);
}

static C_MODIFY_BLOCK_MULT: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_block_mult(_state: &CombatState, _ctx: &BlockCtx) -> DamageMul {
    C_MODIFY_BLOCK_MULT.fetch_add(1, Ordering::SeqCst);
    DamageMul::ONE
}
#[test]
fn modify_block_multiplicative_listener_fires() {
    C_MODIFY_BLOCK_MULT.store(0, Ordering::SeqCst);
    register_modify_block_multiplicative(
        RelicId::DeprecatedRelic,
        listener_modify_block_mult as ModifyBlockMultiplicativeFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = BlockCtx::default();
    let _ = dispatch_modify_block_multiplicative(&f.state, &ctx);
    assert_eq!(C_MODIFY_BLOCK_MULT.load(Ordering::SeqCst), 1);
}

static C_SHOULD_CLEAR_BLOCK: AtomicUsize = AtomicUsize::new(0);
fn listener_should_clear_block(_state: &CombatState, _ctx: &BlockClearedCtx) -> bool {
    C_SHOULD_CLEAR_BLOCK.fetch_add(1, Ordering::SeqCst);
    true
}
#[test]
fn should_clear_block_listener_fires() {
    C_SHOULD_CLEAR_BLOCK.store(0, Ordering::SeqCst);
    register_should_clear_block(
        RelicId::DeprecatedRelic,
        listener_should_clear_block as ShouldClearBlockFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = BlockClearedCtx::default();
    let _ = dispatch_should_clear_block(&f.state, &ctx);
    assert_eq!(C_SHOULD_CLEAR_BLOCK.load(Ordering::SeqCst), 1);
}

static C_AFTER_MAX_HP_CHANGED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_max_hp_changed(
    _state: &mut CombatState,
    _ctx: &MaxHpChangeCtx,
) -> sts2_core::EffectResult {
    C_AFTER_MAX_HP_CHANGED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_max_hp_changed_listener_fires() {
    C_AFTER_MAX_HP_CHANGED.store(0, Ordering::SeqCst);
    register_after_max_hp_changed(
        RelicId::DeprecatedRelic,
        listener_after_max_hp_changed as AfterMaxHpChangedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_max_hp_change_ctx(&f);
    let _ = dispatch_after_max_hp_changed(&mut f.state, &ctx);
    assert_eq!(C_AFTER_MAX_HP_CHANGED.load(Ordering::SeqCst), 1);
}

static C_AFTER_MAX_HP_GAINED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_max_hp_gained(
    _state: &mut CombatState,
    _ctx: &MaxHpChangeCtx,
) -> sts2_core::EffectResult {
    C_AFTER_MAX_HP_GAINED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_max_hp_gained_listener_fires() {
    C_AFTER_MAX_HP_GAINED.store(0, Ordering::SeqCst);
    register_after_max_hp_gained(
        RelicId::DeprecatedRelic,
        listener_after_max_hp_gained as AfterMaxHpGainedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_max_hp_change_ctx(&f);
    let _ = dispatch_after_max_hp_gained(&mut f.state, &ctx);
    assert_eq!(C_AFTER_MAX_HP_GAINED.load(Ordering::SeqCst), 1);
}

static C_MODIFY_POWER_GIVEN: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_power_given(_state: &CombatState, ctx: &PowerAmountCtx) -> i32 {
    C_MODIFY_POWER_GIVEN.fetch_add(1, Ordering::SeqCst);
    ctx.amount
}
#[test]
fn modify_power_amount_given_listener_fires() {
    C_MODIFY_POWER_GIVEN.store(0, Ordering::SeqCst);
    register_modify_power_amount_given(
        RelicId::DeprecatedRelic,
        listener_modify_power_given as ModifyPowerAmountGivenFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = make_power_amount_ctx();
    let _ = dispatch_modify_power_amount_given(&f.state, &ctx);
    assert_eq!(C_MODIFY_POWER_GIVEN.load(Ordering::SeqCst), 1);
}

static C_MODIFY_POWER_RECEIVED: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_power_received(_state: &CombatState, ctx: &PowerAmountCtx) -> i32 {
    C_MODIFY_POWER_RECEIVED.fetch_add(1, Ordering::SeqCst);
    ctx.amount
}
#[test]
fn modify_power_amount_received_listener_fires() {
    C_MODIFY_POWER_RECEIVED.store(0, Ordering::SeqCst);
    register_modify_power_amount_received(
        RelicId::DeprecatedRelic,
        listener_modify_power_received as ModifyPowerAmountReceivedFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = make_power_amount_ctx();
    let _ = dispatch_modify_power_amount_received(&f.state, &ctx);
    assert_eq!(C_MODIFY_POWER_RECEIVED.load(Ordering::SeqCst), 1);
}

static C_BEFORE_POWER_AMOUNT_CHANGED: AtomicUsize = AtomicUsize::new(0);
fn listener_before_power_amount_changed(
    _state: &mut CombatState,
    _ctx: &PowerAmountCtx,
) -> sts2_core::EffectResult {
    C_BEFORE_POWER_AMOUNT_CHANGED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn before_power_amount_changed_listener_fires() {
    C_BEFORE_POWER_AMOUNT_CHANGED.store(0, Ordering::SeqCst);
    register_before_power_amount_changed(
        RelicId::DeprecatedRelic,
        listener_before_power_amount_changed as BeforePowerAmountChangedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_power_amount_ctx();
    let _ = dispatch_before_power_amount_changed(&mut f.state, &ctx);
    assert_eq!(C_BEFORE_POWER_AMOUNT_CHANGED.load(Ordering::SeqCst), 1);
}

static C_AFTER_MOD_POWER_GIVEN: AtomicUsize = AtomicUsize::new(0);
fn listener_after_mod_power_given(
    _state: &mut CombatState,
    _ctx: &PowerAmountCtx,
) -> sts2_core::EffectResult {
    C_AFTER_MOD_POWER_GIVEN.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_modifying_power_amount_given_listener_fires() {
    C_AFTER_MOD_POWER_GIVEN.store(0, Ordering::SeqCst);
    register_after_modifying_power_amount_given(
        RelicId::DeprecatedRelic,
        listener_after_mod_power_given as AfterModifyingPowerAmountGivenFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_power_amount_ctx();
    let _ = dispatch_after_modifying_power_amount_given(&mut f.state, &ctx);
    assert_eq!(C_AFTER_MOD_POWER_GIVEN.load(Ordering::SeqCst), 1);
}

static C_AFTER_MOD_POWER_RECEIVED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_mod_power_received(
    _state: &mut CombatState,
    _ctx: &PowerAmountCtx,
) -> sts2_core::EffectResult {
    C_AFTER_MOD_POWER_RECEIVED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_modifying_power_amount_received_listener_fires() {
    C_AFTER_MOD_POWER_RECEIVED.store(0, Ordering::SeqCst);
    register_after_modifying_power_amount_received(
        RelicId::DeprecatedRelic,
        listener_after_mod_power_received as AfterModifyingPowerAmountReceivedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_power_amount_ctx();
    let _ = dispatch_after_modifying_power_amount_received(&mut f.state, &ctx);
    assert_eq!(C_AFTER_MOD_POWER_RECEIVED.load(Ordering::SeqCst), 1);
}

static C_MODIFY_REWARDS: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_rewards(_state: &CombatState, _rewards: &mut Vec<Reward>) {
    C_MODIFY_REWARDS.fetch_add(1, Ordering::SeqCst);
}
#[test]
fn modify_rewards_listener_fires() {
    C_MODIFY_REWARDS.store(0, Ordering::SeqCst);
    register_modify_rewards(
        RelicId::DeprecatedRelic,
        listener_modify_rewards as ModifyRewardsFn,
    );
    let f = build_hook_smoke_fixture();
    let mut rewards: Vec<Reward> = Vec::new();
    dispatch_modify_rewards(&f.state, &mut rewards);
    assert_eq!(C_MODIFY_REWARDS.load(Ordering::SeqCst), 1);
}

static C_MODIFY_REST_HEAL: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_rest_heal(_state: &CombatState, current: i32) -> i32 {
    C_MODIFY_REST_HEAL.fetch_add(1, Ordering::SeqCst);
    current
}
#[test]
fn modify_rest_site_heal_rewards_listener_fires() {
    C_MODIFY_REST_HEAL.store(0, Ordering::SeqCst);
    register_modify_rest_site_heal_rewards(
        RelicId::DeprecatedRelic,
        listener_modify_rest_heal as ModifyRestSiteHealRewardsFn,
    );
    let f = build_hook_smoke_fixture();
    let _ = dispatch_modify_rest_site_heal_rewards(&f.state, 30);
    assert_eq!(C_MODIFY_REST_HEAL.load(Ordering::SeqCst), 1);
}

static C_MODIFY_CARD_REWARDS_LATE: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_card_rewards_late(_state: &CombatState, _options: &mut Vec<CardId>) {
    C_MODIFY_CARD_REWARDS_LATE.fetch_add(1, Ordering::SeqCst);
}
#[test]
fn modify_card_reward_options_late_listener_fires() {
    C_MODIFY_CARD_REWARDS_LATE.store(0, Ordering::SeqCst);
    register_modify_card_reward_options_late(
        RelicId::DeprecatedRelic,
        listener_modify_card_rewards_late as ModifyCardRewardOptionsLateFn,
    );
    let f = build_hook_smoke_fixture();
    let mut options: Vec<CardId> = vec![CardId::StrikeIronclad];
    dispatch_modify_card_reward_options_late(&f.state, &mut options);
    assert_eq!(C_MODIFY_CARD_REWARDS_LATE.load(Ordering::SeqCst), 1);
}

static C_MODIFY_SHOP_PRICE: AtomicUsize = AtomicUsize::new(0);
fn listener_modify_shop_price(_state: &CombatState, _ctx: &ShopPriceCtx, current: i32) -> i32 {
    C_MODIFY_SHOP_PRICE.fetch_add(1, Ordering::SeqCst);
    current
}
#[test]
fn modify_shop_price_listener_fires() {
    C_MODIFY_SHOP_PRICE.store(0, Ordering::SeqCst);
    register_modify_shop_price(
        RelicId::DeprecatedRelic,
        listener_modify_shop_price as ModifyShopPriceFn,
    );
    let f = build_hook_smoke_fixture();
    let ctx = ShopPriceCtx {
        item_kind: ShopItemKind::Card,
    };
    let _ = dispatch_modify_shop_price(&f.state, &ctx, 100);
    assert_eq!(C_MODIFY_SHOP_PRICE.load(Ordering::SeqCst), 1);
}

static C_ON_PET_SUMMONED: AtomicUsize = AtomicUsize::new(0);
fn listener_on_pet_summoned(
    _state: &mut CombatState,
    _ctx: &PetSummonCtx,
) -> sts2_core::EffectResult {
    C_ON_PET_SUMMONED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn on_pet_summoned_listener_fires() {
    C_ON_PET_SUMMONED.store(0, Ordering::SeqCst);
    register_on_pet_summoned(
        RelicId::DeprecatedRelic,
        listener_on_pet_summoned as OnPetSummonedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = PetSummonCtx {
        pet_kind: PetKind::Byrdpip,
    };
    let _ = dispatch_on_pet_summoned(&mut f.state, &ctx);
    assert_eq!(C_ON_PET_SUMMONED.load(Ordering::SeqCst), 1);
}

static C_COUNT_PETS: AtomicUsize = AtomicUsize::new(0);
fn listener_count_pets(_state: &CombatState) -> i32 {
    C_COUNT_PETS.fetch_add(1, Ordering::SeqCst);
    0
}
#[test]
fn count_pets_listener_fires() {
    C_COUNT_PETS.store(0, Ordering::SeqCst);
    // `RelicId::PaelsLegion` is used by an in-module unit test in
    // `relic_effects.rs` for this same channel, and `RelicId::Byrdpip`
    // is used there too. Pick a distinct id to keep this smoke test's
    // counter unambiguous.
    register_count_pets(RelicId::DeprecatedRelic, listener_count_pets as CountPetsFn);
    let f = build_hook_smoke_fixture();
    let _ = dispatch_count_pets(&f.state);
    assert_eq!(C_COUNT_PETS.load(Ordering::SeqCst), 1);
}

static C_ON_CARD_ADDED: AtomicUsize = AtomicUsize::new(0);
fn listener_on_card_added(
    _state: &mut CombatState,
    _ctx: &CardAddedCtx,
) -> sts2_core::EffectResult {
    C_ON_CARD_ADDED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn on_card_added_to_deck_listener_fires() {
    C_ON_CARD_ADDED.store(0, Ordering::SeqCst);
    register_on_card_added_to_deck(
        RelicId::DeprecatedRelic,
        listener_on_card_added as OnCardAddedToDeckFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_card_added_ctx(&f);
    let _ = dispatch_on_card_added_to_deck(&mut f.state, &ctx);
    assert_eq!(C_ON_CARD_ADDED.load(Ordering::SeqCst), 1);
}

static C_AFTER_SHUFFLE_BUS: AtomicUsize = AtomicUsize::new(0);
fn listener_after_shuffle_bus(
    _state: &mut CombatState,
    _ctx: &ShuffleCtx,
) -> sts2_core::EffectResult {
    C_AFTER_SHUFFLE_BUS.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn after_shuffle_bus_listener_fires() {
    C_AFTER_SHUFFLE_BUS.store(0, Ordering::SeqCst);
    register_after_shuffle_bus(
        RelicId::DeprecatedRelic,
        listener_after_shuffle_bus as AfterShuffleFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = ShuffleCtx::default();
    let _ = dispatch_after_shuffle_bus(&mut f.state, &ctx);
    assert_eq!(C_AFTER_SHUFFLE_BUS.load(Ordering::SeqCst), 1);
}

static C_ON_STARS_SPENT: AtomicUsize = AtomicUsize::new(0);
fn listener_on_stars_spent(
    _state: &mut CombatState,
    _ctx: &StarsSpentCtx,
) -> sts2_core::EffectResult {
    C_ON_STARS_SPENT.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
#[test]
fn on_stars_spent_listener_fires() {
    C_ON_STARS_SPENT.store(0, Ordering::SeqCst);
    register_on_stars_spent(
        RelicId::DeprecatedRelic,
        listener_on_stars_spent as OnStarsSpentFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = StarsSpentCtx { amount: 1 };
    let _ = dispatch_on_stars_spent(&mut f.state, &ctx);
    assert_eq!(C_ON_STARS_SPENT.load(Ordering::SeqCst), 1);
}

static C_AFTER_TURN_END: AtomicUsize = AtomicUsize::new(0);
fn listener_after_turn_end(
    _state: &mut CombatState,
    _side: Side,
) -> sts2_core::card_effects::EffectResult {
    C_AFTER_TURN_END.fetch_add(1, Ordering::SeqCst);
    sts2_core::card_effects::EffectResult::Resolved
}
/// Bead `sts2-rs-hcd6n`: smoke-cover dispatch_after_turn_end.
/// Same Side-enum signature as before_turn_end (2n8ho),
/// after_side_turn_start (cptkn), before_side_turn_start (cvugd).
/// Production listener: LunarPastry.
#[test]
fn after_turn_end_listener_fires() {
    C_AFTER_TURN_END.store(0, Ordering::SeqCst);
    register_after_turn_end(
        RelicId::DeprecatedRelic,
        listener_after_turn_end as AfterTurnEndFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_turn_end(&mut f.state, Side::Player);
    assert_eq!(C_AFTER_TURN_END.load(Ordering::SeqCst), 1);
}

static C_BEFORE_CARD_PLAYED: AtomicUsize = AtomicUsize::new(0);
fn listener_before_card_played(
    _state: &mut CombatState,
    _ctx: &CardEventCtx,
) -> sts2_core::EffectResult {
    C_BEFORE_CARD_PLAYED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-60izs`: smoke-cover dispatch_before_card_played.
/// Same CardEventCtx signature as after_card_played (existing
/// smoke test). Production listener: IntimidatingHelmet.
#[test]
fn before_card_played_listener_fires() {
    C_BEFORE_CARD_PLAYED.store(0, Ordering::SeqCst);
    register_before_card_played(
        RelicId::DeprecatedRelic,
        listener_before_card_played as BeforeCardPlayedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = make_card_event_ctx(&f);
    let _ = dispatch_before_card_played(&mut f.state, &ctx);
    assert_eq!(C_BEFORE_CARD_PLAYED.load(Ordering::SeqCst), 1);
}

static C_ON_ORB_CHANNELED: AtomicUsize = AtomicUsize::new(0);
fn listener_on_orb_channeled(
    _state: &mut CombatState,
    _ctx: &sts2_core::hooks::OrbChanneledCtx,
) -> sts2_core::EffectResult {
    C_ON_ORB_CHANNELED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-grb6d`: smoke-cover dispatch_on_orb_channeled.
/// OrbChanneledCtx is a default-constructible placeholder ctx
/// (hooks.rs:1528). Production listener: Metronome (s5ia).
#[test]
fn on_orb_channeled_listener_fires() {
    C_ON_ORB_CHANNELED.store(0, Ordering::SeqCst);
    register_on_orb_channeled(
        RelicId::DeprecatedRelic,
        listener_on_orb_channeled as OnOrbChanneledFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = sts2_core::hooks::OrbChanneledCtx::default();
    let _ = dispatch_on_orb_channeled(&mut f.state, &ctx);
    assert_eq!(C_ON_ORB_CHANNELED.load(Ordering::SeqCst), 1);
}

static C_AFTER_BLOCK_GAINED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_block_gained(
    _state: &mut CombatState,
    _ctx: &sts2_core::hooks::BlockGainedCtx,
) -> sts2_core::EffectResult {
    C_AFTER_BLOCK_GAINED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-asw7u`: smoke-cover dispatch_after_block_gained.
/// BlockGainedCtx is default-constructible (hooks.rs:1562).
#[test]
fn after_block_gained_listener_fires() {
    C_AFTER_BLOCK_GAINED.store(0, Ordering::SeqCst);
    register_after_block_gained(
        RelicId::DeprecatedRelic,
        listener_after_block_gained as AfterBlockGainedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = sts2_core::hooks::BlockGainedCtx::default();
    let _ = dispatch_after_block_gained(&mut f.state, &ctx);
    assert_eq!(C_AFTER_BLOCK_GAINED.load(Ordering::SeqCst), 1);
}

static C_AFTER_BLOCK_CLEARED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_block_cleared(
    _state: &mut CombatState,
    _ctx: &sts2_core::hooks::BlockClearedCtx,
) -> sts2_core::EffectResult {
    C_AFTER_BLOCK_CLEARED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-11tqu`: smoke-cover dispatch_after_block_cleared.
/// BlockClearedCtx is default-constructible (hooks.rs:1517).
/// Production listeners: CaptainsWheel, SturdyClamp, HornCleat,
/// SparklingRouge.
#[test]
fn after_block_cleared_listener_fires() {
    C_AFTER_BLOCK_CLEARED.store(0, Ordering::SeqCst);
    register_after_block_cleared(
        RelicId::DeprecatedRelic,
        listener_after_block_cleared as AfterBlockClearedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let ctx = sts2_core::hooks::BlockClearedCtx::default();
    let _ = dispatch_after_block_cleared(&mut f.state, &ctx);
    assert_eq!(C_AFTER_BLOCK_CLEARED.load(Ordering::SeqCst), 1);
}

static C_AFTER_OBTAINED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_obtained(_state: &mut CombatState) -> sts2_core::EffectResult {
    C_AFTER_OBTAINED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-lgktc`: smoke-cover dispatch_after_obtained.
/// No-payload signature (4ek41 template).
#[test]
fn after_obtained_listener_fires() {
    C_AFTER_OBTAINED.store(0, Ordering::SeqCst);
    register_after_obtained(
        RelicId::DeprecatedRelic,
        listener_after_obtained as AfterObtainedFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_obtained(&mut f.state);
    assert_eq!(C_AFTER_OBTAINED.load(Ordering::SeqCst), 1);
}

static C_AFTER_ROOM_ENTERED: AtomicUsize = AtomicUsize::new(0);
fn listener_after_room_entered(
    _state: &mut CombatState,
    _room: sts2_core::runstate::RoomType,
) -> sts2_core::EffectResult {
    C_AFTER_ROOM_ENTERED.fetch_add(1, Ordering::SeqCst);
    sts2_core::EffectResult::Resolved
}
/// Bead `sts2-rs-1vnjd`: smoke-cover dispatch_after_room_entered.
/// Enum-arg signature (2n8ho-style template, RoomType in place of
/// Side). Closes the final identified relic-side smoke gap.
#[test]
fn after_room_entered_listener_fires() {
    C_AFTER_ROOM_ENTERED.store(0, Ordering::SeqCst);
    register_after_room_entered(
        RelicId::DeprecatedRelic,
        listener_after_room_entered as AfterRoomEnteredFn,
    );
    let mut f = build_hook_smoke_fixture();
    let _ = dispatch_after_room_entered(&mut f.state, sts2_core::runstate::RoomType::Monster);
    assert_eq!(C_AFTER_ROOM_ENTERED.load(Ordering::SeqCst), 1);
}
