//! Full-path (`combat_loop::play_card` + the real content registry)
//! regression tests for GoldAxe (bead `sts2-rs-wvpjv`).
//!
//! GoldAxe's C# `CalculatedDamageVar` (GoldAxe.cs:18-27) =
//! `0 + 1 * History.CardPlaysFinished.Count(e => e.CardPlay.Card.Owner ==
//! owner)` — i.e. the number of player card plays FINISHED this combat,
//! excluding the in-flight GoldAxe. The Rust analogue is
//! `state.combat_log.player_card_plays_this_combat` (bumped once per play
//! AND once per OneTwoPunch / EchoForm replay, matching C#'s per-iteration
//! `History.CardPlayFinished` in `CardModel.OnPlayWrapper`).
//!
//! Unlike the handler-level tests in `cards/colorless/tests.rs` (which
//! pre-seed the counter and call the handler directly), these drive the
//! real `play_card` increment so they catch (1) per-combat scope across a
//! turn boundary and (2) the replay off-by-one that handler-level tests
//! cannot see.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid,
};

fn fresh() -> (CombatState, CreatureUid) {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 9999,
        max_hp: 9999,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    let uid = CreatureUid(id);
    state.enemy_uids.push(uid);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 99;
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    (state, uid)
}

/// Push an Attack into hand; returns its hand index.
fn push(state: &mut CombatState, id: CardId, dmg: i32) -> u8 {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: dmg,
        base_block: 0,
        ..Card::pod_defaults()
    });
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("hand index fits u8")
}

fn play(state: &mut CombatState, hand_idx: u8) {
    let _ = sts2_core::combat_loop::play_card(state, hand_idx, Some(0));
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

/// First card of combat: GoldAxe deals 0 (no prior finished plays).
#[test]
fn gold_axe_first_card_deals_zero() {
    let (mut state, _e) = fresh();
    let before = enemy_hp(&state);
    let hi = push(&mut state, CardId::GoldAxe, 0);
    play(&mut state, hi);
    assert_eq!(
        before - enemy_hp(&state),
        0,
        "GoldAxe as the first play of combat deals 0",
    );
}

/// After two prior FINISHED plays this combat, GoldAxe deals 2.
#[test]
fn gold_axe_after_two_prior_plays_deals_two() {
    let (mut state, _e) = fresh();
    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);
    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);

    let before = enemy_hp(&state);
    let hi = push(&mut state, CardId::GoldAxe, 0);
    play(&mut state, hi);
    assert_eq!(
        before - enemy_hp(&state),
        2,
        "GoldAxe deals 2 after two finished plays this combat",
    );
}

/// Per-combat scope: plays from a PREVIOUS turn still count. The
/// combat-wide counter must survive a turn boundary (it is NOT cleared in
/// `reset_per_turn`).
#[test]
fn gold_axe_counts_plays_from_previous_turn() {
    let (mut state, _e) = fresh();
    // Two plays on "turn 1".
    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);
    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);

    // Cross a turn boundary: the per-turn reset must NOT clear the
    // combat-wide play counter.
    state.combat_log.reset_per_turn();
    assert_eq!(
        state.combat_log.player_card_plays_this_combat, 2,
        "combat-wide play counter must survive the per-turn reset",
    );

    // Now play GoldAxe on "turn 2" — it still sees the two turn-1 plays.
    let before = enemy_hp(&state);
    let hi = push(&mut state, CardId::GoldAxe, 0);
    play(&mut state, hi);
    assert_eq!(
        before - enemy_hp(&state),
        2,
        "GoldAxe counts plays from a previous turn (per-combat scope)",
    );
}

/// Root-cause regression: a OneTwoPunch'd Attack records TWO finished
/// plays in C# (initial + replay), so a following GoldAxe deals 2, not 1.
/// Before the fix the Rust counter bumped only once per top-level play and
/// GoldAxe was off by one (dealt 1).
#[test]
fn gold_axe_counts_one_two_punch_replays() {
    let (mut state, _e) = fresh();
    // OneTwoPunch(1): the next Attack plays twice.
    let pid = NonZeroU32::new(7601).expect("nz");
    state.powers.push(Power {
        id: pid,
        kind: PowerKind::OneTwoPunch,
        amount: 1,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(PowerUid(pid));

    // One Strike, replayed once → 2 finished plays.
    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);
    assert_eq!(
        state.combat_log.player_card_plays_this_combat, 2,
        "OneTwoPunch'd Strike must count as 2 finished plays (initial + replay)",
    );

    let before = enemy_hp(&state);
    let hi = push(&mut state, CardId::GoldAxe, 0);
    play(&mut state, hi);
    assert_eq!(
        before - enemy_hp(&state),
        2,
        "GoldAxe counts each OneTwoPunch replay as a finished play",
    );
}

/// EchoForm replays likewise each record a finished play.
#[test]
fn gold_axe_counts_echo_form_replays() {
    let (mut state, _e) = fresh();
    // EchoForm(1): the next card plays one extra time.
    let pid = NonZeroU32::new(7602).expect("nz");
    state.powers.push(Power {
        id: pid,
        kind: PowerKind::EchoForm,
        amount: 1,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(PowerUid(pid));

    let hi = push(&mut state, CardId::StrikeIronclad, 6);
    play(&mut state, hi);
    assert_eq!(
        state.combat_log.player_card_plays_this_combat, 2,
        "EchoForm'd Strike must count as 2 finished plays (initial + replay)",
    );

    let before = enemy_hp(&state);
    let hi = push(&mut state, CardId::GoldAxe, 0);
    play(&mut state, hi);
    assert_eq!(
        before - enemy_hp(&state),
        2,
        "GoldAxe counts each EchoForm replay as a finished play",
    );
}
