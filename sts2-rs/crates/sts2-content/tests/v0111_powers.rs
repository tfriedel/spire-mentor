//! GH #794 (S8b): the v0.111.0 power changes, each next to its v0.107.1 twin.
//!
//! Source of truth: `sts2-wiki/decompiled/v0.111.0/MegaCrit.Sts2.Core.Models.Powers/`
//! against `.../v0.107.1/...`. Every v0.111.0 behaviour is gated on
//! `CombatState::game_version`; the `*_v0107` tests pin that the default build
//! is unchanged.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::hooks::{self, CreatureRef, DamageCtx, TurnSideCtx, ValueProps};
use sts2_core::{Action, CardUid, CombatPhase, CombatState, CreatureUid, GameVersion};
use sts2_core::{MonsterKind, PowerKind, Side};

/// A play-ready Ironclad state with one 200-HP Nibbit and an empty hand.
fn fresh(version: GameVersion) -> (CombatState, CreatureUid) {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: version,
        ..Default::default()
    };
    state.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.phase = CombatPhase::PlayerPlayPhase;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 200;
        c.current_hp = 200;
    }
    (state, enemy)
}

fn player_uid(state: &CombatState) -> CreatureUid {
    state.player.creature_uid.expect("player mirror creature")
}

fn hp(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn give(state: &mut CombatState, uid: CreatureUid, kind: PowerKind, amount: i32) {
    sts2_core::card_effects::apply_power_to_creature(state, uid, kind, amount);
}

fn add_to_hand(state: &mut CombatState, card_id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    uid
}

/// The multiplicative damage fold applied to 100 base damage.
fn mul_of_100(state: &CombatState, dealer: CreatureRef, target: CreatureRef) -> i32 {
    let ctx = DamageCtx {
        dealer,
        target,
        props: ValueProps::MOVE,
    };
    hooks::dispatch_modify_damage_multiplicative(state, &ctx).apply(100)
}

// ── TankPower (value) ─────────────────────────────────────────────────────

#[test]
fn tank_doubles_damage_taken_v0107() {
    let (mut state, enemy) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Tank, 1);
    assert_eq!(
        mul_of_100(&state, CreatureRef::Enemy(enemy), CreatureRef::Player),
        200
    );
}

/// `TankPower.cs:54`: the `DamageIncrease` var is 1.5.
#[test]
fn tank_damage_taken_is_one_and_a_half_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Tank, 1);
    assert_eq!(
        mul_of_100(&state, CreatureRef::Enemy(enemy), CreatureRef::Player),
        150
    );
}

// ── TrackingPower ─────────────────────────────────────────────────────────

#[test]
fn tracking_multiplier_is_amount_v0107() {
    let (mut state, enemy) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    give(&mut state, enemy, PowerKind::Weak, 2);
    give(&mut state, p, PowerKind::Tracking, 2);
    assert_eq!(
        mul_of_100(&state, CreatureRef::Player, CreatureRef::Enemy(enemy)),
        200
    );
}

/// `TrackingPower.cs:37`: `1 + Amount / 100` — Tracking(50) is x1.5 and a
/// second stack (100) x2.
#[test]
fn tracking_multiplier_is_one_plus_amount_percent_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, enemy, PowerKind::Weak, 2);
    give(&mut state, p, PowerKind::Tracking, 50);
    assert_eq!(
        mul_of_100(&state, CreatureRef::Player, CreatureRef::Enemy(enemy)),
        150
    );
    give(&mut state, p, PowerKind::Tracking, 50);
    assert_eq!(
        mul_of_100(&state, CreatureRef::Player, CreatureRef::Enemy(enemy)),
        200
    );
}

// ── VulnerablePower: Cruelty via the pet owner ────────────────────────────

fn osty_vs_vulnerable(version: GameVersion) -> i32 {
    let (mut state, enemy) = fresh(version);
    let p = player_uid(&state);
    let osty = state.summon_osty(5).expect("osty summoned");
    give(&mut state, enemy, PowerKind::Vulnerable, 2);
    give(&mut state, p, PowerKind::Cruelty, 25);
    mul_of_100(&state, CreatureRef::Enemy(osty), CreatureRef::Enemy(enemy))
}

#[test]
fn osty_attack_ignores_owner_cruelty_v0107() {
    assert_eq!(osty_vs_vulnerable(GameVersion::V0_107_1), 150);
}

/// `VulnerablePower.cs:46`: `dealer.PetOwner?.Creature.GetPower<CrueltyPower>()`.
#[test]
fn osty_attack_reads_owner_cruelty_v0111() {
    assert_eq!(osty_vs_vulnerable(GameVersion::V0_111_0), 175);
}

// ── DoomPower timing ──────────────────────────────────────────────────────

#[test]
fn player_doom_kills_before_turn_end_v0107() {
    let (mut state, _) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Doom, 100);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert!(
        hp(&state, p) <= 0,
        "v0.107.1 doom-kills in BeforeSideTurnEnd"
    );
}

/// `DoomPower.cs:62-73`: player-side Doom waits for `AfterSideTurnEnd`.
#[test]
fn player_doom_kills_after_turn_end_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Doom, 100);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(
        hp(&state, p),
        80,
        "no player-side kill in BeforeSideTurnEnd"
    );
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert!(hp(&state, p) <= 0, "killed in AfterSideTurnEnd");
}

/// Player at 80 HP with Constrict 9 applied BEFORE Doom 75: Constrict's
/// turn-end damage (ConstrictPower.cs:15, `AfterSideTurnEnd`) takes the
/// player to 71, inside the Doom threshold. Returns the HP after the
/// player's full turn end (before + after hooks).
fn constrict_then_doom_turn_end(version: GameVersion) -> i32 {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Constrict, 9);
    give(&mut state, p, PowerKind::Doom, 75);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    hp(&state, p)
}

/// v0.107.1: Doom checks in `BeforeSideTurnEnd` (80 > 75, survives); the
/// Constrict damage lands after it.
#[test]
fn player_doom_checks_before_constrict_v0107() {
    assert_eq!(constrict_then_doom_turn_end(GameVersion::V0_107_1), 71);
}

/// v0.111.0: both ride `AfterSideTurnEnd`; C# awaits the listeners in order,
/// so the Doom check (DoomPower.cs:68) sees Constrict's damage and kills
/// (Necrobinder v2 fuzz `card+:ENTROPY` path, Neurosurge Doom + Constrict).
#[test]
fn player_doom_sees_earlier_constrict_damage_v0111() {
    assert!(constrict_then_doom_turn_end(GameVersion::V0_111_0) <= 0);
}

/// The same ordering through a real `EndTurn` (the combat loop's turn-end
/// sequence, not the bare hook dispatch).
fn constrict_then_doom_end_turn(version: GameVersion) -> (i32, bool) {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Constrict, 9);
    give(&mut state, p, PowerKind::Doom, 75);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    (hp(&state, p), state.is_over)
}

#[test]
fn player_doom_checks_before_constrict_end_turn_v0107() {
    let (hp_after, over) = constrict_then_doom_end_turn(GameVersion::V0_107_1);
    assert!(hp_after > 0 && !over, "hp {hp_after}, over {over}");
}

#[test]
fn player_doom_sees_earlier_constrict_damage_end_turn_v0111() {
    let (hp_after, over) = constrict_then_doom_end_turn(GameVersion::V0_111_0);
    assert!(hp_after <= 0 && over, "hp {hp_after}, over {over}");
}

/// Monster debuffs reach the player through the legacy `player.powers`
/// surface (`apply_power_to_player_via_legacy_surface`), which the listener
/// walk visits after the mirror creature's powers. Constrict applied that way
/// FIRST, then Doom on the mirror: C# still walks Constrict first.
fn legacy_constrict_then_doom_end_turn(version: GameVersion) -> (i32, bool) {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    let next = state.powers.iter().map(|pw| pw.id.get()).max().unwrap_or(0) + 1;
    let id = std::num::NonZeroU32::new(next).expect("non-zero");
    state.powers.push(sts2_core::Power {
        id,
        kind: PowerKind::Constrict,
        amount: 9,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(sts2_core::PowerUid(id));
    give(&mut state, p, PowerKind::Doom, 75);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    (hp(&state, p), state.is_over)
}

#[test]
fn player_doom_before_legacy_constrict_v0107() {
    let (hp_after, over) = legacy_constrict_then_doom_end_turn(GameVersion::V0_107_1);
    assert!(hp_after > 0 && !over, "hp {hp_after}, over {over}");
}

/// The Necrobinder v2 fuzz path (`card+:ENTROPY`, quick, 8 paths, seed 832):
/// Strangler's Constrict 9 (turn 0) then Neurosurge's Doom 6 (turn 4); C#
/// Constrict takes 13 HP to 4 and Doom kills at the player's turn end.
#[test]
fn player_doom_after_legacy_constrict_v0111() {
    let (hp_after, over) = legacy_constrict_then_doom_end_turn(GameVersion::V0_111_0);
    assert!(hp_after <= 0 && over, "hp {hp_after}, over {over}");
}

/// v0.107.1: Regen ticks in `AfterSideTurnEnd` (RegenPower.cs:17-24) beside
/// Constrict (ConstrictPower.cs:15), and C# walks them in application order
/// (Hook.cs:1411-1426). Constrict 9 lands on the legacy surface FIRST, then
/// Regen 5 on the mirror: 80 -> 71 -> 76. Walking the mirror first heals
/// at full HP (no-op), then hits: 71 (random-loadout Necrobinder 5777 @59).
#[test]
fn legacy_constrict_before_mirror_regen_v0107() {
    let (mut state, _) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    let next = state.powers.iter().map(|pw| pw.id.get()).max().unwrap_or(0) + 1;
    let id = std::num::NonZeroU32::new(next).expect("non-zero");
    state.powers.push(sts2_core::Power {
        id,
        kind: PowerKind::Constrict,
        amount: 9,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(sts2_core::PowerUid(id));
    give(&mut state, p, PowerKind::Regen, 5);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(hp(&state, p), 76);
}

/// Disintegration applied BEFORE Doom still damages AFTER the Doom check:
/// it rides `AfterSideTurnEndLate` (DisintegrationPower.cs:15), which C# runs
/// after every `AfterSideTurnEnd` listener. 80 HP, Disintegration 9, Doom 75:
/// the check sees 80 and the player ends the turn at 71.
fn disintegration_then_doom_end_turn(version: GameVersion) -> (i32, bool) {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Disintegration, 9);
    give(&mut state, p, PowerKind::Doom, 75);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    (hp(&state, p), state.is_over)
}

#[test]
fn player_doom_checks_before_late_disintegration_v0107() {
    let (hp_after, over) = disintegration_then_doom_end_turn(GameVersion::V0_107_1);
    assert!(hp_after > 0 && !over, "hp {hp_after}, over {over}");
}

/// Necrobinder v2 fuzz (full, seed 5151): Knowledge Demon's Disintegration 6
/// and Neurosurge's Doom 6 at 12 HP; C# survives the turn end at 6 HP.
#[test]
fn player_doom_checks_before_late_disintegration_v0111() {
    let (hp_after, over) = disintegration_then_doom_end_turn(GameVersion::V0_111_0);
    assert!(hp_after > 0 && !over, "hp {hp_after}, over {over}");
}

/// Enemy-side Doom keeps its `BeforeSideTurnEnd` timing in v0.111.0.
#[test]
fn enemy_doom_still_kills_before_turn_end_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    give(&mut state, enemy, PowerKind::Doom, 500);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert!(hp(&state, enemy) <= 0);
}

// ── RegenPower timing ─────────────────────────────────────────────────────

fn regen_after(version: GameVersion, before_only: bool) -> i32 {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == p.0) {
        c.current_hp = 50;
    }
    give(&mut state, p, PowerKind::Regen, 5);
    hooks::dispatch_before_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    if !before_only {
        hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    }
    hp(&state, p)
}

#[test]
fn regen_heals_after_turn_end_v0107() {
    assert_eq!(regen_after(GameVersion::V0_107_1, true), 50);
    assert_eq!(regen_after(GameVersion::V0_107_1, false), 55);
}

/// `RegenPower.cs:20`: the heal moved to `BeforeSideTurnEndEarly`, and it
/// does not heal again after the turn end.
#[test]
fn regen_heals_before_turn_end_v0111() {
    assert_eq!(regen_after(GameVersion::V0_111_0, true), 55);
    assert_eq!(regen_after(GameVersion::V0_111_0, false), 55);
}

// ── JugglingPower: BeforeCardPlayed ───────────────────────────────────────

/// Fiend Fire as the turn's 3rd Attack under Juggling(1), with two Defends in
/// hand. Returns (hand size, exhaust size) after the play.
fn juggled_fiend_fire(version: GameVersion) -> (usize, usize) {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Juggling, 1);
    for pid in state.player_power_uids() {
        if let Some(pw) = state.powers.get_mut_by_id(pid.0) {
            if pw.kind == PowerKind::Juggling {
                pw.counter = 2;
            }
        }
    }
    add_to_hand(&mut state, CardId::FiendFire);
    add_to_hand(&mut state, CardId::DefendIronclad);
    add_to_hand(&mut state, CardId::DefendIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    (state.player.hand.len(), state.player.exhaust.len())
}

/// v0.107.1 clones after the play: the copy survives Fiend Fire's exhaust.
#[test]
fn juggling_clones_after_play_v0107() {
    assert_eq!(juggled_fiend_fire(GameVersion::V0_107_1), (1, 3));
}

/// `JugglingPower.cs:36`: the copy reaches the hand before Fiend Fire
/// resolves, so Fiend Fire exhausts it with the rest of the hand.
#[test]
fn juggling_clones_before_play_v0111() {
    assert_eq!(juggled_fiend_fire(GameVersion::V0_111_0), (0, 4));
}

/// Strike as the turn's 3rd Attack under Juggling(1) and Calamity(1), the two
/// powers applied in `juggling_first` order. Returns the hand's card ids.
fn juggled_with_calamity(juggling_first: bool) -> Vec<CardId> {
    let (mut state, _) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    if juggling_first {
        give(&mut state, p, PowerKind::Juggling, 1);
        give(&mut state, p, PowerKind::Calamity, 1);
    } else {
        give(&mut state, p, PowerKind::Calamity, 1);
        give(&mut state, p, PowerKind::Juggling, 1);
    }
    for pid in state.player_power_uids() {
        if let Some(pw) = state.powers.get_mut_by_id(pid.0) {
            if pw.kind == PowerKind::Juggling {
                pw.counter = 2;
            }
        }
    }
    add_to_hand(&mut state, CardId::StrikeIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    state
        .player
        .hand
        .iter()
        .filter_map(|u| sts2_core::card_by_uid(&state.cards, *u).map(|c| c.card_id))
        .collect()
}

/// C# `Hook.AfterCardPlayed` walks the owner's powers in application order
/// (`CombatState.IterateHookListeners`), so in v0.107.1 Juggling's clone and
/// Calamity's conjured Attack land in the order the powers were applied.
#[test]
fn juggling_and_calamity_add_to_hand_in_application_order_v0107() {
    let hand = juggled_with_calamity(true);
    assert_eq!(hand.len(), 2);
    assert_eq!(hand[0], CardId::StrikeIronclad, "{hand:?}");
    let hand = juggled_with_calamity(false);
    assert_eq!(hand.len(), 2);
    assert_eq!(hand[1], CardId::StrikeIronclad, "{hand:?}");
}

// ── PaleBlueDotPower rework ───────────────────────────────────────────────

/// Play `n` Defends under PaleBlueDot(1); returns the DrawCardsNextTurn stack.
fn pale_blue_dot_after_plays(version: GameVersion, n: usize) -> (CombatState, i32) {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::PaleBlueDot, 1);
    state.player.energy = 10;
    for _ in 0..n {
        add_to_hand(&mut state, CardId::DefendIronclad);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    }
    let dcnt = state.player_power_amount(PowerKind::DrawCardsNextTurn);
    (state, dcnt)
}

#[test]
fn pale_blue_dot_has_no_play_trigger_v0107() {
    let (_, dcnt) = pale_blue_dot_after_plays(GameVersion::V0_107_1, 6);
    assert_eq!(dcnt, 0);
}

/// `PaleBlueDotPower.cs:42`: the 5th finished play this turn applies
/// DrawCardsNextTurn(Amount), once per turn.
#[test]
fn pale_blue_dot_fifth_play_grants_draw_next_turn_once_v0111() {
    let (_, four) = pale_blue_dot_after_plays(GameVersion::V0_111_0, 4);
    assert_eq!(four, 0);
    let (mut state, six) = pale_blue_dot_after_plays(GameVersion::V0_111_0, 6);
    assert_eq!(six, 1, "fires on the 5th play only");
    // `AfterSideTurnEnd` re-arms the latch.
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    let latched = state
        .player_power_uids()
        .into_iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .any(|pw| pw.kind == PowerKind::PaleBlueDot && pw.counter != 0);
    assert!(!latched);
}

/// v0.111.0 PaleBlueDot no longer adds to the hand draw.
#[test]
fn pale_blue_dot_no_hand_draw_bonus_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::PaleBlueDot, 1);
    state.combat_log.card_plays_finished_last_player_turn = 6;
    let ctx = hooks::HandDrawCtx::default();
    assert_eq!(hooks::dispatch_modify_hand_draw(&state, &ctx), 0);
    state.game_version = GameVersion::V0_107_1;
    assert_eq!(hooks::dispatch_modify_hand_draw(&state, &ctx), 1);
}

// ── WellLaidPlansPower rework ─────────────────────────────────────────────

/// End the turn holding three Defends under WellLaidPlans(1).
fn end_turn_with_wlp(version: GameVersion) -> CombatState {
    let (mut state, _) = fresh(version);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::WellLaidPlans, 1);
    for _ in 0..3 {
        add_to_hand(&mut state, CardId::DefendIronclad);
    }
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    state
}

/// v0.107.1 opens the "retain up to Amount" pick.
#[test]
fn well_laid_plans_opens_retain_pick_v0107() {
    let state = end_turn_with_wlp(GameVersion::V0_107_1);
    assert!(state.pending_choice.is_some(), "retain select surfaced");
}

/// `WellLaidPlansPower.cs:12`: `ShouldFlush` is false — the whole hand stays,
/// with no pick.
#[test]
fn well_laid_plans_keeps_whole_hand_v0111() {
    let state = end_turn_with_wlp(GameVersion::V0_111_0);
    assert!(state.pending_choice.is_none(), "no retain select");
    assert_eq!(state.player.discard.len(), 0, "nothing flushed");
    assert!(state.player.hand.len() >= 3, "kept the three Defends");
}

// ── FeralPower ────────────────────────────────────────────────────────────
// The IsDupe guard (`FeralPower.cs:51`) is new in v0.111.0, on every play
// path: v0.107.1's `ModifyCardPlayResultPileTypeAndPosition` redirects a dupe.

fn add_zero_cost_to_hand(state: &mut CombatState, card_id: CardId) -> CardUid {
    let uid = add_to_hand(state, card_id);
    state.cards[uid.get() as usize - 1].cost = sts2_core::CardEnergyCost::new(0);
    uid
}

/// C# `CardModel.OnPlayWrapper` (CardModel.cs:1881-1886) fixes the result
/// location, and `FeralPower.AfterModifyingCardPlayResultLocation` spends the
/// budget, BEFORE `OnPlay`. A 0-cost Uproar under Feral(1) therefore claims
/// the one slot before its `OnPlay` auto-plays FTL: Uproar returns to the top
/// of the hand and FTL goes to discard. Rust used to decide at disposal, so
/// the nested FTL (disposed first) took the slot (Defect act23 seed 1014).
#[test]
fn feral_slot_goes_to_the_outer_card_of_a_nested_auto_play() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Feral, 1);
    let uproar = add_zero_cost_to_hand(&mut state, CardId::Uproar);
    let ftl = add_to_hand(&mut state, CardId::Ftl);
    let defend = add_to_hand(&mut state, CardId::DefendIronclad);
    // Move FTL and a Defend (FTL's draw) from the hand to the draw pile.
    state.player.hand.retain(|u| *u == uproar);
    state.player.draw.push(ftl);
    state.player.draw.push(defend);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert_eq!(
        state.player.hand.first(),
        Some(&uproar),
        "Uproar at hand[0]"
    );
    assert!(state.player.discard.contains(&ftl), "FTL discarded");
    assert!(!state.player.hand.contains(&ftl), "FTL not returned");
}

/// v0.107.1 `FeralPower.ModifyCardPlayResultPileTypeAndPosition` has no
/// `IsDupe` check, so a hand-played 0-cost dupe Attack returns to the hand top
/// instead of leaving combat. v0.111.0 added `!card.IsDupe` (FeralPower.cs:51)
/// and the dupe is removed (Defect random loadout v0.107.1 seed 5354).
#[test]
fn feral_dupe_check_is_new_in_v0111() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let (mut state, _) = fresh(version);
        let p = player_uid(&state);
        give(&mut state, p, PowerKind::Feral, 1);
        let dupe = add_zero_cost_to_hand(&mut state, CardId::StrikeIronclad);
        state.cards[dupe.get() as usize - 1].set_combat_var(sts2_core::cards::VarKey::IsDupe, 1);
        let _ = add_to_hand(&mut state, CardId::DefendIronclad);

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        let in_hand = state.player.hand.first() == Some(&dupe);
        let anywhere = state.player.hand.contains(&dupe)
            || state.player.discard.contains(&dupe)
            || state.player.play.contains(&dupe)
            || state.player.exhaust.contains(&dupe);
        if version == GameVersion::V0_107_1 {
            assert!(in_hand, "{version:?}: the dupe returns to hand[0]");
        } else {
            assert!(!anywhere, "{version:?}: the dupe leaves combat");
        }
    }
}

/// Feral returns the card to (Hand, Top); with a full hand `CardPileCmd.Add`
/// sends it to Discard at that same Top position (index 0,
/// CardPileCmd.cs:420-424/511-516), not the bottom (loadout Defect v0.107.1
/// 2008). Flash of Steel draws the tenth card first.
#[test]
fn feral_return_to_a_full_hand_lands_on_discard_top() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let (mut state, _) = fresh(version);
        let p = player_uid(&state);
        give(&mut state, p, PowerKind::Feral, 1);
        let flash = add_zero_cost_to_hand(&mut state, CardId::FlashOfSteel);
        for _ in 0..9 {
            let _ = add_to_hand(&mut state, CardId::DefendIronclad);
        }
        let drawn = add_to_hand(&mut state, CardId::DefendIronclad);
        let old = add_to_hand(&mut state, CardId::DefendIronclad);
        state.player.hand.retain(|u| *u != drawn && *u != old);
        state.player.draw.push(drawn);
        state.player.discard.push(old);
        assert_eq!(state.player.hand.len(), 10);

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        assert_eq!(state.player.hand.len(), 10, "{version:?}: hand full");
        assert_eq!(
            state.player.discard.as_slice(),
            &[flash, old],
            "{version:?}: Feral's Top overflow goes to discard[0]"
        );
    }
}

/// `FeralPower.cs:60-66`: once Feral is on, `zeroCostAttacksPlayed` only moves
/// when Feral actually redirects a play. A 0-cost Attack played over budget
/// does not spend anything, so a second Feral stack (Amount 2) leaves one more
/// slot open for the next 0-cost Attack.
#[test]
fn feral_over_budget_attack_does_not_spend_the_tally() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    state.player.energy = 10;
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Feral, 1);
    let first = add_zero_cost_to_hand(&mut state, CardId::StrikeIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.player.hand.first(), Some(&first), "first returns");
    // Over budget: the same card plays again and is discarded.
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert!(
        state.player.discard.contains(&first),
        "over budget: discard"
    );

    give(&mut state, p, PowerKind::Feral, 1);
    let next = add_zero_cost_to_hand(&mut state, CardId::StrikeIronclad);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(state.player.hand.first(), Some(&next), "the second slot");
}

// ── SurprisePower: creature creation order ────────────────────────────────

fn merc_dies(version: GameVersion) -> CombatState {
    content_dispatch::install_default_registry();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(777_i32),
        game_version: version,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.init_player_creature();
    let merc = state.spawn_enemy(MonsterKind::GremlinMerc);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == merc.0) {
        c.current_hp = 0;
    }
    sts2_content::monsters::act1_underdocks::surprise_after_death_spawn(&mut state, merc);
    state
}

fn species_order(state: &CombatState) -> Vec<MonsterKind> {
    state
        .enemy_uids
        .iter()
        .filter_map(|u| state.creatures.iter().find(|c| c.id == u.0))
        .filter_map(|c| c.species)
        .collect()
}

fn max_hp_of(state: &CombatState, kind: MonsterKind) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.species == Some(kind))
        .map_or(0, |c| c.max_hp)
}

/// Both builds keep the slot order `[merc, sneaky, fat]`.
#[test]
fn surprise_keeps_enemy_order_both_builds() {
    for v in GameVersion::ALL {
        assert_eq!(
            species_order(&merc_dies(v)),
            [
                MonsterKind::GremlinMerc,
                MonsterKind::SneakyGremlin,
                MonsterKind::FatGremlin
            ],
            "{v}"
        );
    }
}

/// `SurprisePower.cs:24`: v0.111.0 creates (and HP-rolls) the FatGremlin
/// first; v0.107.1 rolls the SneakyGremlin first. The Niche draws are the same
/// two reads in the opposite order.
#[test]
fn surprise_rolls_fat_gremlin_first_v0111() {
    let old = merc_dies(GameVersion::V0_107_1);
    let new = merc_dies(GameVersion::V0_111_0);
    let fat_uid = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.species == Some(MonsterKind::FatGremlin))
            .map(|c| c.id)
    };
    let sneaky_uid = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.species == Some(MonsterKind::SneakyGremlin))
            .map(|c| c.id)
    };
    assert!(
        sneaky_uid(&old) < fat_uid(&old),
        "v0.107.1 creates the sneaky first"
    );
    assert!(
        fat_uid(&new) < sneaky_uid(&new),
        "v0.111.0 creates the fat first"
    );
    // Re-roll the expected v0.111.0 values off the same Niche stream.
    let mut niche = sts2_core::run_rng_set::RunRngSet::new(777_i32).niche;
    let none = std::collections::HashSet::new();
    // The merc's own spawn roll comes first; it is dead (out of `taken`) by
    // the time the gremlins roll.
    let _merc = MonsterKind::GremlinMerc.starting_hp_unique(&mut niche, &none, 0);
    let fat = MonsterKind::FatGremlin.starting_hp_unique(&mut niche, &none, 0);
    let sneaky = MonsterKind::SneakyGremlin.starting_hp_unique(&mut niche, &none, 0);
    assert_eq!(max_hp_of(&new, MonsterKind::FatGremlin), fat);
    assert_eq!(max_hp_of(&new, MonsterKind::SneakyGremlin), sneaky);
}

// ── AmbergrisPower (new) ──────────────────────────────────────────────────

/// `Ambergris.cs`: in combat the potion also applies AmbergrisPower(1).
#[test]
fn ambergris_potion_applies_power_v0111() {
    sts2_content::potions::dispatch::ensure_installed();
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let _ = sts2_core::potion_effects::dispatch(
        &mut state,
        sts2_core::potions::PotionKind::Ambergris,
        None,
    );
    assert_eq!(state.player_power_amount(PowerKind::Ambergris), 1);
}

/// `AmbergrisPower.cs`: one stack buys one extra player turn (no enemy turn),
/// spent as the extra turn begins.
#[test]
fn ambergris_grants_one_extra_turn_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Ambergris, 1);
    let enemy_hp = hp(&state, enemy);
    let round = state.round;
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.round, round, "an extra turn keeps the round");
    assert_eq!(
        state.player_power_amount(PowerKind::Ambergris),
        0,
        "stack spent"
    );
    assert_eq!(hp(&state, enemy), enemy_hp);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert!(state.round > round, "the next end turn runs the enemy turn");
}

/// An extra turn taken in round 1 is the player's turn 2: C#
/// `SwitchSides` runs `PlayerCombatState.IncrementTurnNumber()` without
/// `RoundNumber++` (CombatManager.cs:1866-1888), and BagOfPreparation reads
/// `TurnNumber > 1` (BagOfPreparation.cs:20), so the extra turn draws the
/// plain 5, not 5 + 2.
#[test]
fn ambergris_extra_turn_is_turn_two_for_bag_of_preparation_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    state.round = 1;
    state.relics.push(sts2_core::Relic {
        id: std::num::NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::BagOfPreparation,
        counter: sts2_core::RelicKind::BagOfPreparation.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..15 {
        let mut card = sts2_core::lookup_card_factory()
            .and_then(|f| f(CardId::StrikeIronclad, false))
            .expect("card factory has the card");
        let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.draw.push(uid);
    }
    assert_eq!(state.turn_number(), 1);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Ambergris, 1);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.round, 1, "the extra turn keeps the round");
    assert_eq!(state.turn_number(), 2, "but is the player's turn 2");
    assert_eq!(
        state.player.hand.len(),
        5,
        "BagOfPreparation's turn-1 +2 does not fire again on the extra turn"
    );
}

/// PaelsEye's extra turn (no card played this turn) skips the enemy side and
/// is the player's turn 2 while the round stays 1: C# `SwitchSides` keeps
/// `CurrentSide` on the player and runs `IncrementTurnNumber()` without
/// `RoundNumber++` (v0.111.0 CombatManager.cs:1866-1888, v0.107.1
/// :1390-1424, same body). GH #806: v0.107.1 used to run the enemy turn
/// anyway and keep the turn number equal to the round.
#[test]
fn paels_eye_extra_turn_skips_the_enemy_side_both_versions() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let (mut state, _) = fresh(version);
        state.round = 1;
        state.relics.push(sts2_core::Relic {
            id: std::num::NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::PaelsEye,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let hp_before = state.player.current_hp;
        let _ = sts2_core::apply(&mut state, Action::EndTurn);
        assert_eq!(state.phase, CombatPhase::PlayerPlayPhase, "{version:?}");
        assert_eq!(
            state.round, 1,
            "the extra turn keeps the round ({version:?})"
        );
        assert_eq!(state.turn_number(), 2, "but is turn 2 ({version:?})");
        assert_eq!(
            state.player.current_hp, hp_before,
            "no enemy turn ran ({version:?})"
        );
        let _ = sts2_core::apply(&mut state, Action::EndTurn);
        assert!(
            state.round > 1,
            "the next end turn runs the enemy ({version:?})"
        );
    }
}

/// An enemy's Plating re-grants its round-1 block on a PaelsEye extra turn.
/// C# `PlatingPower.BeforeSideTurnStart` (v0.111.0 and v0.107.1
/// PlatingPower.cs:41-56) gates only on `side == Player`, a non-player owner
/// and `RoundNumber <= 1`, with no once-only flag; the extra turn keeps the
/// round at 1, and no enemy turn ran to clear the first grant, so the block
/// stacks. Deep act23 probe Ironclad:1018 act3: `monsters[0].block` cs=30
/// rs=15 (Rust's `counter` latch stopped the second grant).
#[test]
fn enemy_plating_regrants_round_one_block_on_paels_eye_extra_turn_both_versions() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let (mut state, enemy) = fresh(version);
        state.round = 1;
        state.relics.push(sts2_core::Relic {
            id: std::num::NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::PaelsEye,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        give(&mut state, enemy, PowerKind::Plating, 15);
        let block = |s: &CombatState| {
            s.creatures
                .iter()
                .find(|c| c.id == enemy.0)
                .map_or(0, |c| c.block)
        };
        // Combat start's player-side turn start: the round-1 grant.
        hooks::dispatch_before_side_turn_start(&mut state, &TurnSideCtx { side: Side::Player });
        assert_eq!(block(&state), 15, "round-1 grant ({version:?})");
        let _ = sts2_core::apply(&mut state, Action::EndTurn);
        assert_eq!(state.phase, CombatPhase::PlayerPlayPhase, "{version:?}");
        assert_eq!(state.round, 1, "extra turn keeps the round ({version:?})");
        assert_eq!(
            block(&state),
            30,
            "the extra turn re-grants the round-1 block ({version:?})"
        );
    }
}

/// The extra-turn predicate ignores AmbergrisPower under v0.107.1 (it cannot
/// exist there; the gate is belt-and-braces).
#[test]
fn ambergris_no_extra_turn_v0107() {
    let (mut state, _) = fresh(GameVersion::V0_107_1);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Ambergris, 1);
    assert!(!hooks::ambergris_should_take_extra_turn_v0111(&state));
}

// ── OneForAllPower (new) ──────────────────────────────────────────────────

fn one_for_all_bonus(cost_paid: i32) -> i32 {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::OneForAll, 3);
    let uid = add_to_hand(&mut state, CardId::StrikeIronclad);
    // Stand the Strike on the play pile as the resolving card.
    state.player.hand.retain(|u| *u != uid);
    state.player.play.push(uid);
    state.cards[uid.0.get() as usize - 1].energy_spent = cost_paid;
    let ctx = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    hooks::dispatch_modify_damage_additive(&state, &ctx)
}

/// `OneForAllPower.cs:14`: +Amount only when the play spent 0 energy.
#[test]
fn one_for_all_adds_to_free_plays_v0111() {
    assert_eq!(one_for_all_bonus(0), 3);
    assert_eq!(one_for_all_bonus(1), 0);
}

/// `CardCmd.AutoPlay` plays with `EnergySpent = 0` (v0.111.0
/// Commands/CardCmd.cs:124), so OneForAll boosts an auto-played 1-cost
/// attack: Havoc auto-plays the 6-damage Strike for 6 + 3. Parity fuzz
/// Defect seed 545 (Mayhem → Catastrophe → RocketPunch, GH #798).
#[test]
fn one_for_all_adds_to_auto_played_card_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::OneForAll, 3);
    let havoc = add_to_hand(&mut state, CardId::Havoc);
    let strike = add_to_hand(&mut state, CardId::StrikeIronclad);
    state.player.hand.retain(|u| *u != strike);
    state.player.draw.insert(0, strike);
    let idx = state
        .player
        .hand
        .iter()
        .position(|u| *u == havoc)
        .expect("havoc in hand");
    let before = hp(&state, enemy);
    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: u8::try_from(idx).expect("fits"),
            target: None,
        },
    );
    assert!(
        state.player.exhaust.contains(&strike),
        "Havoc auto-played the Strike"
    );
    assert_eq!(before - hp(&state, enemy), 9, "auto-played Strike gets +3");
    assert!(
        !state.combat_log.resolving_auto_play,
        "the flag is dispatch-scoped"
    );
}

// ── Temporary Dexterity / Focus subclasses (new) ──────────────────────────

/// `FadePower : TemporaryDexterityPower`: the owner's turn end removes it and
/// takes the Dexterity back.
#[test]
fn fade_reverts_dexterity_at_turn_end_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Dexterity, 2);
    give(&mut state, p, PowerKind::Fade, 2);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(state.player_power_amount(PowerKind::Fade), 0);
    assert_eq!(state.player_power_amount(PowerKind::Dexterity), 0);
}

/// `HyperbeamFocusDownPower` (`IsPositive => false`): the turn end gives the
/// Focus back.
#[test]
fn hyperbeam_focus_down_restores_focus_at_turn_end_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Focus, 1);
    sts2_core::card_effects::apply_signed_power_to_creature_allow_dead(
        &mut state,
        p,
        PowerKind::Focus,
        -3,
    );
    give(&mut state, p, PowerKind::HyperbeamFocusDown, 3);
    assert_eq!(state.player_power_amount(PowerKind::Focus), -2);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(state.player_power_amount(PowerKind::HyperbeamFocusDown), 0);
    assert_eq!(state.player_power_amount(PowerKind::Focus), 1);
    assert!(sts2_core::afflictions::classify(PowerKind::HyperbeamFocusDown).is_debuff());
}

// ── Other new powers ──────────────────────────────────────────────────────

/// `HibernatePower.AfterPlayerTurnStart` decrements once per player turn.
#[test]
fn hibernate_decrements_at_player_turn_start_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Hibernate, 2);
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.player_power_amount(PowerKind::Hibernate), 1);
}

/// `ConcoctPower`: a powered player hit with unblocked damage poisons the
/// target; the power leaves at the end of the opposing (enemy) turn.
#[test]
fn concoct_poisons_on_unblocked_hit_and_expires_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Concoct, 4);
    let _ = sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        6,
        ValueProps::MOVE,
        None,
    );
    let poison = state
        .creatures
        .iter()
        .find(|c| c.id == enemy.0)
        .into_iter()
        .flat_map(|c| c.powers.iter())
        .filter_map(|u| state.powers.get_by_id(u.0))
        .find(|pw| pw.kind == PowerKind::Poison)
        .map_or(0, |pw| pw.amount);
    assert_eq!(poison, 4);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(
        state.player_power_amount(PowerKind::Concoct),
        4,
        "own turn end keeps it"
    );
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert_eq!(state.player_power_amount(PowerKind::Concoct), 0);
}

/// `UnderworldPower` leaves at the enemy turn end.
#[test]
fn underworld_expires_at_enemy_turn_end_v0111() {
    let (mut state, _) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Underworld, 1);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(state.player_power_amount(PowerKind::Underworld), 1);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert_eq!(state.player_power_amount(PowerKind::Underworld), 0);
}

/// `CacophonyPower`: the 33rd card drawn deals `Amount` unpowered damage to a
/// random enemy and resets the count.
#[test]
fn cacophony_hits_on_the_33rd_draw_v0111() {
    let (mut state, enemy) = fresh(GameVersion::V0_111_0);
    let p = player_uid(&state);
    give(&mut state, p, PowerKind::Cacophony, 9);
    for _ in 0..40 {
        let uid = add_to_hand(&mut state, CardId::DefendIronclad);
        state.player.hand.retain(|u| *u != uid);
        state.player.draw.push(uid);
    }
    let before = hp(&state, enemy);
    let _ = state.draw_into_hand_card_effect(5);
    for _ in 0..27 {
        state.player.hand.clear();
        let _ = state.draw_into_hand_card_effect(1);
    }
    assert_eq!(hp(&state, enemy), before, "32 draws: no hit yet");
    state.player.hand.clear();
    let _ = state.draw_into_hand_card_effect(1);
    assert_eq!(hp(&state, enemy), before - 9, "the 33rd draw hits");
}

// ── Version-scoped power universe ─────────────────────────────────────────

#[test]
fn power_universe_is_version_scoped() {
    let old = PowerKind::all_for(GameVersion::V0_107_1);
    assert_eq!(old.as_slice(), PowerKind::all());
    let new = PowerKind::all_for(GameVersion::V0_111_0);
    for k in PowerKind::V0111_ONLY {
        assert!(!old.contains(&k), "{k:?} leaked into v0.107.1");
        assert!(new.contains(&k), "{k:?} missing from v0.111.0");
    }
    for k in PowerKind::REMOVED_IN_V0111 {
        assert!(old.contains(&k));
        assert!(!new.contains(&k), "{k:?} must not exist in v0.111.0");
    }
}

/// DiamondDiademPower no longer halves damage under v0.111.0 (the power was
/// deleted; only a stale applier could create it).
#[test]
fn diamond_diadem_power_is_inert_v0111() {
    for (v, want) in [(GameVersion::V0_107_1, 50), (GameVersion::V0_111_0, 100)] {
        let (mut state, enemy) = fresh(v);
        let p = player_uid(&state);
        give(&mut state, p, PowerKind::DiamondDiadem, 1);
        assert_eq!(
            mul_of_100(&state, CreatureRef::Enemy(enemy), CreatureRef::Player),
            want,
            "{v}"
        );
    }
}
