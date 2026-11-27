//! `PowerFlashed` emission for `Hook.AfterModifyingDamageAmount` — the three
//! sites GH #635 landed: `HardToKillPower.cs:23`, `IntangiblePower.cs:61` and
//! `SlowPower.cs:48`.
//!
//! All three C# arms are a bare `Flash()` with no guard of their own. Every
//! condition lives in WHICH listeners `Hook.ModifyDamage` reports as having
//! moved the number (`Hook.cs:2655-2702`), which is why the negative halves
//! below are the substance of this file rather than its padding: a flash wired
//! to fire whenever the power exists passes every positive assertion here.
//!
//! Separate from `power_flash.rs` because these are driven through the real
//! damage pipeline rather than through a single dispatcher call — the selection
//! rule is a property of the pipeline, not of an arm.

use sts2_core::event_log::EventLogEntry;
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{CombatState, CreatureUid, MonsterKind, Power, PowerKind, PowerUid};

// ── fixtures ────────────────────────────────────────────────────────────

/// A powered attack. `ValueProps::is_powered` is `MOVE && !UNPOWERED`
/// (`hooks.rs:306`), so `MOVE` alone is the powered case and the bare `NONE`
/// default is NOT — which is the difference `SlowPower.cs:39` reads.
const POWERED: ValueProps = ValueProps::MOVE;
/// The same hit with the powered flag off — `ValueProp.Unpowered`.
const UNPOWERED: ValueProps = ValueProps::MOVE.union(ValueProps::UNPOWERED);

fn power_uid(n: u32) -> PowerUid {
    PowerUid::new(n).expect("non-zero power uid")
}

fn recording_state() -> CombatState {
    let mut state = CombatState::default();
    state.combat_log.event_log.set_recording(true);
    state
}

fn add_enemy(state: &mut CombatState, hp: i32) -> CreatureUid {
    let uid = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
        c.max_hp = hp;
        c.current_hp = hp;
    }
    uid
}

/// Give `enemy` a power of its own, with `counter` for the Counter-style ones.
fn push_enemy_power(
    state: &mut CombatState,
    id: u32,
    enemy: CreatureUid,
    kind: PowerKind,
    amount: i32,
    counter: i32,
) -> PowerUid {
    let uid = power_uid(id);
    state.powers.push(Power {
        id: uid.0,
        kind,
        amount,
        counter,
        applier: None,
        skip_next_tick: false,
    });
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.powers.push(uid);
    }
    uid
}

fn flash_rows(state: &CombatState, power: PowerKind) -> usize {
    state
        .combat_log
        .event_log
        .rows()
        .iter()
        .filter(|r| matches!(&r.entry, EventLogEntry::PowerFlashed { power: p, .. } if *p == power))
        .count()
}

/// Land one player→enemy hit through the real pipeline.
fn hit(state: &mut CombatState, target: CreatureUid, raw: i32) {
    sts2_core::damage::apply_damage_full(state, CreatureRef::Player, target, raw, POWERED, None);
}

// ── the workhorse is HardToKill, and that is deliberate ────────────────
//
// `IntangiblePower` has TWO flash sites and the other one was already ported:
// `:37`'s `AfterModifyingHpLostAfterOsty` fires from `damage.rs`'s
// `flash_intangible_on_target` whenever a hit got through at all. So every
// Intangible assertion here would be reading a SUM of two arms, and a
// regression in the new one could hide behind the old one.
//
// `HardToKillPower` has exactly one (`:23`), and rides the same
// `modify_damage_cap` fold, so it isolates the stage under test. The two-site
// interaction gets its own test at the bottom rather than contaminating these.

#[test]
fn hard_to_kill_flashes_when_its_cap_actually_binds() {
    // `HardToKillPower.ModifyDamageCap` returns `base.Amount`, so the cap is
    // the stack count rather than a constant — 9 here.
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::HardToKill, 9, 0);

    hit(&mut state, enemy, 15);
    assert_eq!(
        flash_rows(&state, PowerKind::HardToKill),
        1,
        "15 against a cap of 9 is the case `num > num5` admits"
    );
}

#[test]
fn hard_to_kill_stays_dark_when_the_hit_was_already_under_its_cap() {
    // `Hook.cs:2692`'s `if (num > num5)`. The listener returned a cap, but it
    // did not MOVE the number, so it is not in `out modifiers` — the whole
    // reason the flash cannot simply follow "does this creature have the
    // power?". Reachable constantly: most hits on a HardToKill(9) Exoskeleton
    // are under 9.
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::HardToKill, 9, 0);

    hit(&mut state, enemy, 4);
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 0);
}

#[test]
fn hard_to_kill_stays_dark_when_someone_else_is_the_target() {
    // `HardToKillPower.cs:17`: `if (target != base.Owner) return
    // decimal.MaxValue;` — an uncapped listener never enters `modifiers`, so a
    // board of four Exoskeletons flashes only the one being hit.
    let mut state = recording_state();
    let bystander = add_enemy(&mut state, 90);
    let victim = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, bystander, PowerKind::HardToKill, 9, 0);

    hit(&mut state, victim, 15);
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 0);
}

#[test]
fn a_second_larger_cap_is_not_credited_once_a_smaller_one_has_bound() {
    // `Hook.cs:2689`'s `num4` is a RUNNING MINIMUM and the test is strict
    // (`num5 < num4`), so of two caps only one can be credited once the
    // smaller has been seen. Intangible's 1 is below HardToKill's 9, and the
    // listener walk reaches HardToKill first here — so HardToKill lowers the
    // minimum from MAX to 9 and is credited, then Intangible lowers it again
    // to 1 and is too. Reverse the amounts and the larger one drops out; that
    // is the half a `cap.min(...)` fold cannot express, and the reason this
    // walk transcribes the loop rather than reusing the fold's result.
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::HardToKill, 1, 0);
    push_enemy_power(&mut state, 2, enemy, PowerKind::Intangible, 1, 0);

    hit(&mut state, enemy, 20);
    assert_eq!(
        flash_rows(&state, PowerKind::HardToKill),
        1,
        "HardToKill(1) is seen first and lowers MAX to 1"
    );
    // Intangible's own cap is also 1, so `num5 < num4` is `1 < 1` — false. It
    // is NOT credited for the damage-amount stage. Its `:37` hp-loss arm still
    // fires (see the two-site test below), which is why this asserts on
    // HardToKill.
}

// ── Slow (SlowPower.cs:48) ──────────────────────────────────────────────
//
// NOTE the `the_slow_power_` prefix: `.config/nextest.toml`'s `default-filter`
// excludes tests whose name starts with `slow_`, which is the repo's marker for
// long-running sweeps. A `slow_flashes_…` test here compiled, passed locally
// under an explicit filter, and was SKIPPED by every default run.

#[test]
fn the_slow_power_flashes_only_once_its_counter_has_moved_off_zero() {
    // `SlowPower.ModifyDamageMultiplicative` is `1 + 0.1 * SlowAmount`, which
    // is the identity at `SlowAmount == 0` — and `Hook.cs:2677`'s
    // `if (num3 != 1m)` is what turns that into "no flash". Reachable in
    // ordinary play: the counter resets at every side turn start.
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    let slow = push_enemy_power(&mut state, 1, enemy, PowerKind::Slow, 1, 0);

    hit(&mut state, enemy, 10);
    assert_eq!(
        flash_rows(&state, PowerKind::Slow),
        0,
        "SlowAmount 0 multiplies by exactly 1"
    );

    if let Some(p) = state.powers.get_mut_by_id(slow.0) {
        p.counter = 3;
    }
    hit(&mut state, enemy, 10);
    assert_eq!(flash_rows(&state, PowerKind::Slow), 1);
}

#[test]
fn the_slow_power_stays_dark_on_an_unpowered_hit() {
    // `SlowPower.cs:39`: `if (!props.IsPoweredAttack()) return 1m;`
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::Slow, 1, 3);

    sts2_core::damage::apply_damage_full(
        &mut state,
        CreatureRef::Player,
        enemy,
        10,
        UNPOWERED,
        None,
    );
    assert_eq!(flash_rows(&state, PowerKind::Slow), 0);
}

// ── cardinality and the preview boundary ────────────────────────────────

#[test]
fn an_aoe_flashes_once_per_target_it_moved() {
    // `CreatureCmd.cs:262` dispatches inside the per-target loop, so four
    // HardToKill targets are four flashes. This is the case that would be
    // wrong by OMISSION if the emission had been put on the single-hit path
    // alone: the AoE route goes through `compute_modified_damage` +
    // `apply_damage_with_final`, never through `apply_damage_full`.
    let mut state = recording_state();
    for i in 0..4 {
        let uid = add_enemy(&mut state, 90);
        push_enemy_power(&mut state, i + 1, uid, PowerKind::HardToKill, 9, 0);
    }

    sts2_core::card_effects::deal_damage_to_all_alive_enemies(&mut state, 15, POWERED, None);
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 4);
}

#[test]
fn a_preview_flashes_nothing() {
    // `AttackIntent.GetSingleDamage` runs the same `Hook.ModifyDamage` with
    // `CardPreviewMode.None` and dispatches NOTHING after it. The workbench
    // reads a telegraph on every state it renders, so a flash here would fire
    // for hits that never land.
    let mut state = recording_state();
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::HardToKill, 9, 0);

    let previewed = sts2_core::damage::compute_modified_damage(
        &state,
        CreatureRef::Player,
        enemy,
        15,
        POWERED,
        None,
    );
    assert_eq!(
        previewed, 9,
        "the preview still caps, it just does not flash"
    );
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 0);
}

#[test]
fn the_player_side_telegraph_flashes_nothing_but_the_hit_does() {
    // `preview_damage_to_player` is shared verbatim between the intent
    // telegraph and `apply_damage_to_player`, so this pair is what proves the
    // emission sits on the resolution and not in the shared helper. Nothing
    // else in this file exercises the player-side route at all.
    let mut state = recording_state();
    state.player.max_hp = 80;
    state.player.current_hp = 80;
    let enemy = add_enemy(&mut state, 50);
    let hard = power_uid(1);
    state.powers.push(Power {
        id: hard.0,
        kind: PowerKind::HardToKill,
        amount: 9,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(hard);

    let previewed = sts2_core::combat_loop::preview_damage_to_player(
        &state,
        CreatureRef::Enemy(enemy),
        15,
        POWERED,
        None,
    );
    assert_eq!(previewed, 9);
    assert_eq!(
        flash_rows(&state, PowerKind::HardToKill),
        0,
        "the telegraph must stay silent"
    );

    sts2_core::combat_loop::apply_damage_to_player(
        &mut state,
        CreatureRef::Enemy(enemy),
        15,
        POWERED,
        None,
    );
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 1);
}

#[test]
fn intangible_fires_both_of_its_sites_on_a_hit_the_cap_binds() {
    // The interaction the rest of this file routes around, pinned rather than
    // left implicit. `IntangiblePower` overrides the After-hook TWICE: `:37`
    // (`AfterModifyingHpLostAfterOsty`, ported before GH #635, fires whenever
    // a hit got through) and `:61` (`AfterModifyingDamageAmount`, this change,
    // fires only when the cap moved the number). The game really does flash
    // twice for one big hit; a small one flashes once.
    let mut big = recording_state();
    let enemy = add_enemy(&mut big, 50);
    push_enemy_power(&mut big, 1, enemy, PowerKind::Intangible, 1, 0);
    hit(&mut big, enemy, 9);
    assert_eq!(flash_rows(&big, PowerKind::Intangible), 2);

    let mut small = recording_state();
    let enemy = add_enemy(&mut small, 50);
    push_enemy_power(&mut small, 1, enemy, PowerKind::Intangible, 1, 0);
    hit(&mut small, enemy, 1);
    assert_eq!(
        flash_rows(&small, PowerKind::Intangible),
        1,
        "the hp-loss arm fires, the damage-amount arm does not — a 1-damage \
         hit against a cap of 1 moved nothing"
    );
}

#[test]
fn a_disarmed_run_records_nothing_and_lands_the_same_damage() {
    // The MCTS-facing contract. `flash_damage_amount_modifiers` early-returns
    // before the selection walk, so search pays one branch — and, more
    // importantly, the damage is bit-identical either way.
    let mut armed = recording_state();
    let mut silent = CombatState::default();
    for state in [&mut armed, &mut silent] {
        let enemy = add_enemy(state, 90);
        push_enemy_power(state, 1, enemy, PowerKind::HardToKill, 9, 0);
        hit(state, enemy, 15);
    }
    assert_eq!(flash_rows(&armed, PowerKind::HardToKill), 1);
    assert_eq!(flash_rows(&silent, PowerKind::HardToKill), 0);

    let hp_of = |s: &CombatState| s.creatures.first().map(|c| c.current_hp);
    assert_eq!(hp_of(&armed), hp_of(&silent), "the flash must not move HP");
}

// ── v0.114.0: HardToKill caps HP lost, after Block (GH #1924) ──────────────

fn hard_to_kill_with_block(version: sts2_core::GameVersion) -> (CombatState, CreatureUid) {
    let mut state = recording_state();
    state.game_version = version;
    let enemy = add_enemy(&mut state, 90);
    push_enemy_power(&mut state, 1, enemy, PowerKind::HardToKill, 9, 0);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.block = 10;
    }
    (state, enemy)
}

fn hp_and_block(state: &CombatState, enemy: CreatureUid) -> (i32, i32) {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == enemy.0)
        .expect("enemy");
    (c.current_hp, c.block)
}

#[test]
fn hard_to_kill_caps_the_damage_before_block_through_v0113() {
    // v0.113.1 `HardToKillPower.ModifyDamageCap`: 25 is capped to 9, which
    // the 10 Block absorbs whole.
    let (mut state, enemy) = hard_to_kill_with_block(sts2_core::GameVersion::V0_113_0);
    hit(&mut state, enemy, 25);
    assert_eq!(hp_and_block(&state, enemy), (90, 1));
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 1);
}

#[test]
fn hard_to_kill_caps_the_hp_lost_after_block_on_v0114() {
    // v0.114.0 `HardToKillPower.ModifyHpLostBeforeOstyLate`: 25 takes all 10
    // Block, and the 15 that gets through is capped to 9.
    let (mut state, enemy) = hard_to_kill_with_block(sts2_core::GameVersion::V0_114_0);
    hit(&mut state, enemy, 25);
    assert_eq!(hp_and_block(&state, enemy), (81, 0));
    assert_eq!(
        flash_rows(&state, PowerKind::HardToKill),
        1,
        "the cap bound"
    );
    // 14 into no Block: 14 > 9, capped again.
    hit(&mut state, enemy, 14);
    assert_eq!(hp_and_block(&state, enemy), (72, 0));
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 2);
    // Under the cap: lands in full, no flash.
    hit(&mut state, enemy, 5);
    assert_eq!(hp_and_block(&state, enemy), (67, 0));
    assert_eq!(flash_rows(&state, PowerKind::HardToKill), 2);
}
