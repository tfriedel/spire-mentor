//! GH #1131: a discarded Sly card's select prompts the player.
//!
//! C# `CardCmd.Discard` / `DiscardAndDraw` `await`s `CardCmd.AutoPlay` for
//! each discarded Sly card in turn (`CardCmd.cs:201-204`), and the discarding
//! card's `OnPlay` resumes only after the last one. A Sly card whose `OnPlay`
//! opens `CardSelectCmd.FromHand` prompts the player there.
//!
//! Rust used to drain those selects inline with option 0 because the Sly loop
//! had no point to wait at. It now parks a loop continuation that carries the
//! not-yet-played Sly cards and the discarding card's follow-up (Hidden
//! Daggers' Shivs).

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{Action, ChoiceKind, CombatPhase, CombatState, MonsterKind, StepResult};

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

fn resolve(state: &mut CombatState, picks: &[u8]) -> StepResult {
    let selection = picks.iter().copied().collect();
    sts2_core::apply(state, Action::ResolveChoice { selection })
}

fn fresh_state() -> CombatState {
    content_dispatch::ensure_installed();
    // Real runs register the card factory; the Shivs need it.
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;
    state
}

fn shivs_in_hand(state: &CombatState) -> usize {
    state
        .player
        .hand
        .iter()
        .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Shiv)
        .count()
}

/// Hidden Daggers discards two Sly Armaments. Each one's upgrade select
/// prompts in turn, and the Shivs only arrive after both, so they are never
/// upgrade options.
#[test]
fn discarded_sly_selects_prompt_before_the_discarder_finishes() {
    let mut state = fresh_state();
    let daggers = push_hand(&mut state, CardId::HiddenDaggers, CardType::Skill);
    let first = push_hand(&mut state, CardId::Armaments, CardType::Skill);
    let second = push_hand(&mut state, CardId::Armaments, CardType::Skill);
    for _ in 0..3 {
        let _ = push_hand(&mut state, CardId::DefendIronclad, CardType::Skill);
    }
    state.player.sly_this_combat.push(first);
    state.player.sly_this_combat.push(second);
    let upgraded = |s: &CombatState| s.cards.iter().filter(|c| c.upgraded).count();

    let r = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(r, StepResult::PendingChoice);
    let kind = state.pending_choice.as_ref().map(|pc| pc.kind);
    assert_eq!(kind, Some(ChoiceKind::DiscardFromHand));

    // Discard both Armaments (hand order: Armaments, Armaments, 3 Defends).
    // A resolve that opens the next select reports `Advanced`, as every
    // chained select does; the open choice is what counts.
    let _ = resolve(&mut state, &[0, 1]);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the first Sly select prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert_eq!(pc.options.len(), 3, "three Defends, no Shivs yet");
    assert_eq!(shivs_in_hand(&state), 0);
    assert!(state.player.play.contains(&daggers), "the discarder waits");

    let _ = resolve(&mut state, &[0]);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the second Sly select prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert_eq!(pc.options.len(), 2, "two Defends left, still no Shivs");
    assert_eq!(shivs_in_hand(&state), 0);
    assert_eq!(upgraded(&state), 1);

    let _ = resolve(&mut state, &[0]);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert!(state.pending_play.is_none());
    assert_eq!(upgraded(&state), 2);
    assert_eq!(shivs_in_hand(&state), 2);
    assert!(state.player.discard.contains(&daggers));
    assert!(state.player.discard.contains(&first));
    assert!(state.player.discard.contains(&second));
}

/// Storm of Steel discards the whole hand, and its Shivs come after every Sly
/// card. A Sly Armaments then has nothing to upgrade, so nothing prompts, and
/// the Shivs still arrive.
#[test]
fn whole_hand_discard_still_spawns_after_sly() {
    let mut state = fresh_state();
    let storm = push_hand(&mut state, CardId::StormOfSteel, CardType::Skill);
    let arm = push_hand(&mut state, CardId::Armaments, CardType::Skill);
    let _ = push_hand(&mut state, CardId::DefendIronclad, CardType::Skill);
    state.player.sly_this_combat.push(arm);

    let r = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_ne!(r, StepResult::PendingChoice);
    assert_eq!(shivs_in_hand(&state), 2);
    assert!(state.player.discard.contains(&storm));
    assert!(state.player.discard.contains(&arm));
}

fn push_real(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Havoc auto-plays a Survivor, whose discard prompts and parks Havoc's
/// auto-play. Discarding a Sly Armaments then auto-plays it under that parked
/// auto-play, and its upgrade select prompts too. After the pick, Armaments
/// finishes, then Survivor (force-exhausted by Havoc), then Havoc.
#[test]
fn sly_select_under_a_parked_auto_play_prompts() {
    let mut state = fresh_state();
    let survivor = push_real(&mut state, CardId::Survivor);
    state.player.draw.push(survivor);
    let havoc = push_real(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);
    let arm = push_real(&mut state, CardId::Armaments);
    state.player.hand.push(arm);
    state.player.sly_this_combat.push(arm);
    for _ in 0..2 {
        let d = push_real(&mut state, CardId::DefendIronclad);
        state.player.hand.push(d);
    }

    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    let pc = state.pending_choice.as_ref().expect("Survivor's discard");
    assert_eq!(pc.kind, ChoiceKind::DiscardFromHand);
    assert!(state.pending_autoplay_resume.is_some());

    // Hand order: Armaments, Defend, Defend.
    let _ = resolve(&mut state, &[0]);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the Sly Armaments select prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert_eq!(pc.options.len(), 2, "the two Defends");
    assert!(state.player.play.contains(&survivor), "Survivor waits");

    let _ = resolve(&mut state, &[0]);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert!(
        state.player.play.is_empty(),
        "play: {:?}",
        state.player.play
    );
    assert_eq!(state.cards.iter().filter(|c| c.upgraded).count(), 1);
    assert!(state.player.exhaust.contains(&survivor));
    assert!(state.player.discard.contains(&arm));
    assert!(state.player.discard.contains(&havoc));
}
