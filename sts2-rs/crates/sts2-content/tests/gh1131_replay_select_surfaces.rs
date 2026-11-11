//! GH #1131: a replayed card's select prompts the player on every play.
//!
//! C# `CardModel.OnPlayWrapper` runs one `for (i = 0; i < playCount; i++)`
//! loop and `await`s `OnPlay` on each iteration, so the replay's
//! `CardSelectCmd.FromHand` prompts exactly like the first play's does
//! (`CardModel.cs:1896-1970`). Burst / Double Tap / One-Two Punch add plays
//! through `ModifyCardPlayCount`, Echo Form adds the last one.
//!
//! Rust used to drain every replay's choice inline with option 0, because the
//! hand-play tail had no point to wait at. The tail now parks in `pending_play`
//! and resumes right after the replay's `OnPlay`.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    Action, ChoiceKind, CombatPhase, CombatState, Power, PowerKind, PowerUid, StepResult,
};

fn push_hand(state: &mut CombatState, id: CardId, card_type: CardType) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 6,
        base_block: 5,
        ..Card::pod_defaults()
    });
    state.player.hand.push(uid);
    uid
}

fn add_player_power(state: &mut CombatState, kind: PowerKind, amount: i32) {
    let pid = std::num::NonZeroU32::new(9100).expect("nz");
    state.powers.push(Power {
        id: pid,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(PowerUid(pid));
}

fn pick_first(state: &mut CombatState) -> StepResult {
    let selection = [0u8].into_iter().collect();
    sts2_core::apply(state, Action::ResolveChoice { selection })
}

/// Armaments (upgrade one card in hand) with three upgradable cards beside it,
/// under `power`. Returns the state right after the hand play.
fn play_armaments_under(power: PowerKind) -> (CombatState, sts2_core::CardUid) {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;
    let armaments = push_hand(&mut state, CardId::Armaments, CardType::Skill);
    for _ in 0..3 {
        let _ = push_hand(&mut state, CardId::DefendIronclad, CardType::Skill);
    }
    add_player_power(&mut state, power, 1);
    let r = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(r, StepResult::PendingChoice, "the first play prompts");
    (state, armaments)
}

fn assert_prompts_once_per_play(power: PowerKind) {
    let (mut state, armaments) = play_armaments_under(power);
    let upgraded = |s: &CombatState| s.cards.iter().filter(|c| c.upgraded).count();

    let kind = state.pending_choice.as_ref().map(|pc| pc.kind);
    assert_eq!(kind, Some(ChoiceKind::UpgradeInHand));
    let _ = pick_first(&mut state);
    assert_eq!(upgraded(&state), 1);

    // The replay prompts too, with the card still in flight.
    let pc = state.pending_choice.as_ref().expect("the replay prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert!(state.pending_play.is_some(), "the tail waits for the pick");
    assert!(state.player.play.contains(&armaments));

    let _ = pick_first(&mut state);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_play.is_none());
    assert_eq!(upgraded(&state), 2);
    assert!(state.player.discard.contains(&armaments));
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}

#[test]
fn burst_replay_select_prompts() {
    assert_prompts_once_per_play(PowerKind::Burst);
}

#[test]
fn echo_form_replay_select_prompts() {
    assert_prompts_once_per_play(PowerKind::EchoForm);
}

/// An auto-played card's replay prompts too. Havoc plays Armaments off the
/// draw pile under Burst: the first play's select parks the auto-play, and once
/// it is answered the replay's select parks it again instead of taking
/// option 0. Havoc's own tail waits behind both.
#[test]
fn auto_played_card_replay_select_prompts() {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;
    // Typed as an Attack so Burst (Skills only) is left for Armaments.
    let havoc = push_hand(&mut state, CardId::Havoc, CardType::Attack);
    for _ in 0..3 {
        let _ = push_hand(&mut state, CardId::DefendIronclad, CardType::Skill);
    }
    let armaments = push_hand(&mut state, CardId::Armaments, CardType::Skill);
    state.player.hand.retain(|u| *u != armaments);
    state.player.draw.push(armaments);
    add_player_power(&mut state, PowerKind::Burst, 1);
    let upgraded = |s: &CombatState| s.cards.iter().filter(|c| c.upgraded).count();

    let r = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        r,
        StepResult::PendingChoice,
        "Armaments' first play prompts"
    );
    let _ = pick_first(&mut state);
    assert_eq!(upgraded(&state), 1);

    let pc = state.pending_choice.as_ref().expect("the replay prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert!(state.pending_autoplay_resume.is_some());
    assert!(state.player.play.contains(&armaments));
    assert!(state.player.play.contains(&havoc));

    let _ = pick_first(&mut state);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert!(state.pending_play.is_none());
    assert_eq!(upgraded(&state), 2);
    assert!(
        state.player.exhaust.contains(&armaments),
        "Havoc exhausts it"
    );
    assert!(!state.player.play.contains(&havoc));
}
