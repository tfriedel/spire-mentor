//! Bead `sts2-rs-mcts-surface-onplay-select`: a top-level player card whose
//! `OnPlay` opens an interactive select (Survivor's "discard 1", DaggerThrow,
//! Acrobatics, …) now SURFACES that choice through `apply`/`legal_actions`
//! instead of draining it eagerly to first-legal. This is what lets an MCTS
//! agent actually CHOOSE which card to discard, rather than always taking
//! option 0.
//!
//! The ordering-fidelity guard for the deferred play tail lives in
//! `curl_up_tingsha_discard_order.rs`; this file guards the agent-facing
//! capability: the choice is enumerated with one option per discardable card,
//! and picking different options yields different resulting states.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{Action, CombatPhase, CombatState, StepResult};

fn push_card(state: &mut CombatState, id: CardId, card_type: CardType) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 0,
        ..Card::pod_defaults()
    });
    uid
}

/// SURVIVOR (gain Block, then "discard 1 card") played into a 2-card residual
/// hand surfaces an interactive discard select: `play_card` returns
/// `PendingChoice`, `legal_actions` offers one `ResolveChoice` per discardable
/// card, and answering with two DIFFERENT selections discards two DIFFERENT
/// cards — proving the agent has a real, consequential choice (not a forced
/// first-legal pick).
#[test]
fn survivor_surfaces_a_consequential_discard_choice() {
    content_dispatch::ensure_installed();

    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 99;
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;

    // Hand: SURVIVOR (hand index 0) + two DISTINGUISHABLE residual cards.
    let survivor = push_card(&mut state, CardId::Survivor, CardType::Skill);
    let strike = push_card(&mut state, CardId::StrikeSilent, CardType::Attack);
    let defend = push_card(&mut state, CardId::DefendSilent, CardType::Skill);
    state.player.hand.push(survivor);
    state.player.hand.push(strike);
    state.player.hand.push(defend);

    // Play SURVIVOR — its OnPlay discard select must SURFACE (not drain).
    let step = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        step,
        StepResult::PendingChoice,
        "SURVIVOR's discard select must surface for the agent to pick"
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("a discard choice is open");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::DiscardFromHand);
    assert_eq!(pc.options.len(), 2, "one option per discardable hand card");
    assert!(
        state.pending_play.is_some(),
        "the play tail is deferred behind the open choice"
    );

    // Enumerate the agent-visible choices: a distinct ResolveChoice per card.
    let selections: Vec<_> = sts2_core::legal_actions(&state)
        .into_iter()
        .filter_map(|a| match a {
            Action::ResolveChoice { selection } if !selection.is_empty() => Some(selection),
            _ => None,
        })
        .collect();
    assert_eq!(
        selections.len(),
        2,
        "MCTS must see both discard options, not a single forced pick"
    );

    // Picking different options must discard different cards.
    let discarded_id = |sel: &sts2_core::Action| -> CardId {
        let mut s = state.clone();
        let _ = sts2_core::apply(&mut s, sel.clone());
        assert!(
            s.pending_choice.is_none() && s.pending_play.is_none(),
            "resolving the choice runs the deferred tail to completion"
        );
        // Exactly one residual card was discarded; SURVIVOR itself routed to
        // discard by the finish tail too, so identify the discarded *residual*.
        let in_hand: Vec<CardId> = s
            .player
            .hand
            .iter()
            .map(|u| s.cards[u.0.get() as usize - 1].card_id)
            .collect();
        // The residual card NOT still in hand is the one discarded.
        if in_hand.contains(&CardId::StrikeSilent) {
            CardId::DefendSilent
        } else {
            CardId::StrikeSilent
        }
    };

    let a0 = Action::ResolveChoice {
        selection: selections[0].clone(),
    };
    let a1 = Action::ResolveChoice {
        selection: selections[1].clone(),
    };
    assert_ne!(
        discarded_id(&a0),
        discarded_id(&a1),
        "different selections must discard different cards — the choice is consequential"
    );

    // Sanity: at least one residual card remains in hand after a single discard.
    let mut s = state.clone();
    let _ = sts2_core::apply(&mut s, a0);
    assert_eq!(
        s.player.hand.len(),
        1,
        "SURVIVOR discards exactly one residual card; the other stays in hand"
    );
}

/// Build a minimal player-turn combat with DREDGE in hand (index 0) and a
/// discard pile of `n` distinct cards.
fn dredge_state(discard_n: usize) -> (CombatState, sts2_core::CardUid) {
    content_dispatch::ensure_installed();

    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    state.player.energy = 99;
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;

    let dredge = push_card(&mut state, CardId::Dredge, CardType::Skill);
    state.player.hand.push(dredge);

    // Distinct discard cards (alternating ids so they're distinguishable).
    for i in 0..discard_n {
        let id = if i % 2 == 0 {
            CardId::StrikeSilent
        } else {
            CardId::DefendSilent
        };
        let uid = push_card(&mut state, id, CardType::Skill);
        state.player.discard.push(uid);
    }
    (state, dredge)
}

/// DREDGE with a discard pile LARGER than `num` (CardsVar 3) surfaces an
/// interactive discard→hand multi-select: `AddFromDiscardToHand` with
/// `min == max == num`, one option per discard-pile card, NOT auto-resolved.
#[test]
fn dredge_surplus_surfaces_min_max_num_discard_select() {
    let (mut state, _dredge) = dredge_state(5); // 5 > num(3)

    let step = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        step,
        StepResult::PendingChoice,
        "DREDGE's discard multi-select must surface when discard.len() > num"
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("a discard→hand choice is open");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDiscardToHand);
    assert_eq!(
        pc.min_select, 3,
        "min == num (CardsVar 3, capped by hand space)"
    );
    assert_eq!(pc.max_select, 3, "max == num");
    assert_eq!(
        pc.options.len(),
        5,
        "one option per discard-pile card (whole pile is selectable)"
    );

    // Bead `sts2-rs-sopgf`: the multi-pick now decomposes into SEQUENTIAL
    // single-index picks. Each legal `ResolveChoice` is a single incremental
    // pick (len 1) until `min_select` is reached; the resolver runs ONCE at
    // FINALIZE. Drive the lowest-index unpicked option each step until the
    // pending choice clears.
    for _ in 0..10 {
        if state.pending_choice.is_none() {
            break;
        }
        let sel = sts2_core::legal_actions(&state)
            .into_iter()
            .find_map(|a| match a {
                // Take the first non-empty (lowest-index) pick while picks
                // remain; once only FINALIZE (empty) is offered, take it.
                Action::ResolveChoice { selection } if !selection.is_empty() => Some(selection),
                _ => None,
            });
        let selection = sel.unwrap_or_default(); // empty = FINALIZE
        assert!(selection.len() <= 1, "sequential picks are single-index");
        let _ = sts2_core::apply(&mut state, Action::ResolveChoice { selection });
    }
    assert!(state.pending_choice.is_none(), "choice resolved");
    assert_eq!(
        state.player.hand.len(),
        3,
        "exactly num cards moved to hand"
    );
    assert_eq!(
        state.player.discard.len(),
        2,
        "the unpicked 2 stay in discard"
    );
}

/// DREDGE with a discard pile NO LARGER than `num` auto-takes the whole pile
/// (the FromSimpleGrid `cards.Count <= MinSelect` fast path) — no prompt, every
/// discard card moves to hand.
#[test]
fn dredge_takes_all_when_discard_le_num_no_prompt() {
    let (mut state, _dredge) = dredge_state(2); // 2 <= num(3)

    let step = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_ne!(
        step,
        StepResult::PendingChoice,
        "DREDGE must NOT surface a choice when discard.len() <= num"
    );
    assert!(state.pending_choice.is_none(), "no prompt opened");
    assert_eq!(
        state.player.hand.len(),
        2,
        "both discard cards moved to hand (DREDGE itself exhausted)"
    );
    assert!(
        state.player.discard.is_empty(),
        "discard pile emptied into hand"
    );
}

/// A potion that opens a choose-a-card screen must SAY so.
///
/// The four card-generating potions (Attack / Skill / Power / Colorless) all
/// route through `open_generated_free_choice` and park a `pending_choice`, but
/// the `DrinkPotion` arm used to return a bare `Advanced` — so the screen was
/// invisible to any driver that steps this engine off the `StepResult`. The
/// DLL combat mirror saw `advanced`, kept no parked choice, and then answered
/// the real game's card grant on a state that had never opened one: 102 of the
/// 107 unparked injections in a 4-seed probe were exactly this, one per
/// card-generating potion drunk. In-process rollouts never noticed, because
/// they read `pending_choice` directly instead of the result.
#[test]
fn a_card_generating_potion_reports_its_pending_choice() {
    content_dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;
    state.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
    state
        .player
        .potions
        .push(sts2_core::potions::PotionKind::AttackPotion);
    // A live enemy, so the drink cannot resolve as a vacuous victory.
    state.spawn_enemy(sts2_core::monsters::MonsterKind::Nibbit);

    let step = sts2_core::apply(
        &mut state,
        Action::DrinkPotion {
            potion_slot: 0,
            target: None,
        },
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("AttackPotion must open a choose-a-card screen");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddGeneratedFreeToHand);
    assert_eq!(
        step,
        StepResult::PendingChoice,
        "and the step result must surface it, not report a plain advance"
    );
}
