//! Bead `sts2-rs-eirvo`: `StampedePower`'s end-of-turn auto-play batch must
//! AWAIT each play, exactly as C# does.
//!
//! `StampedePower.AfterAutoPostPlayPhaseEntered` (`StampedePower.cs:18-34`) is
//!
//! [decompiled C# omitted — see the cited path]
//!
//! The `await` is load-bearing. A Stampede-picked `SEEKER_STRIKE` awaits an
//! interactive `CardSelectCmd.FromCombatPile` inside its own `OnPlay`, and C#
//! blocks the whole turn transition there: iteration `i+1` does not pick, the
//! `HowlFromBeyond` replay does not run, `BeforeTurnEnd` does not fire, and the
//! enemy turn does not start until the player answers.
//!
//! The Rust port used to run straight through: the batch played its next card
//! while the first was still mid-resolve (two cards left sitting on the Play
//! pile), and the end-turn chain only suspended AFTER `BeforeTurnEnd`. That is
//! the interleaving GH #422's `dispatch: empty Play pile` panic fell out of —
//! the stranded card survived into the next turn's setup, where the Bolas /
//! ThrummingHatchet `BeforeHandDraw` sweep (it scans every non-Hand pile, Play
//! included) pulled it back into hand with its plays still owed.

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, legal_actions, Action, CardUid, CombatPhase, CombatState, Creature, CreatureUid,
    EndTurnResume, IntentHistory, MonsterId, MonsterKind, PlayerCombatState, PowerKind, StepResult,
};

/// Two SEEKER_STRIKEs in hand (the only Attacks), four Defends in the draw
/// pile (so each SeekerStrike's draw-select has >1 option and therefore
/// SURFACES rather than auto-resolving), and StampedePower(2).
///
/// Both hand Attacks are the same card, so which one the batch's
/// `Rng.Shuffle.NextItem` picks first does not matter — the assertions below
/// hold for either order.
fn stampede_state() -> CombatState {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();

    let mut cards: Vec<Card> = Vec::new();
    let push = |cards: &mut Vec<Card>, card_id, card_type, target_type, dmg| -> CardUid {
        let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
        cards.push(Card {
            uid,
            card_id,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type,
            target_type,
            upgraded: false,
            base_damage: dmg,
            base_block: 5,
            ..Card::pod_defaults()
        });
        uid
    };

    let ss1 = push(
        &mut cards,
        CardId::SeekerStrike,
        CardType::Attack,
        TargetType::Enemy,
        6,
    );
    let ss2 = push(
        &mut cards,
        CardId::SeekerStrike,
        CardType::Attack,
        TargetType::Enemy,
        6,
    );
    let draw_uids: SmallVec<[CardUid; 32]> = (0..4)
        .map(|_| {
            push(
                &mut cards,
                CardId::DefendIronclad,
                CardType::Skill,
                TargetType::Self_,
                0,
            )
        })
        .collect();

    let enemy_uid = CreatureUid::new(1).expect("non-zero");
    let creatures = vec![Creature {
        id: enemy_uid.0,
        // Fat enough that neither Stampede attack can end the combat — an
        // ending combat suppresses the select entirely (C# `CardSelectCmd`
        // returns empty when `IsEnding`).
        current_hp: 400,
        max_hp: 400,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::ShrinkerBeetle),
        monster_id: MonsterId(MonsterKind::ShrinkerBeetle.python_name()),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: Some(build_ai(
            MonsterKind::ShrinkerBeetle,
            MonsterSpawnArgs::None,
            SpawnPosition {
                alone: true,
                front: true,
                back: true,
                index: 0,
            },
        )),
    }];

    let player = PlayerCombatState {
        draw: draw_uids.clone(),
        max_energy: 3,
        current_hp: 400,
        max_hp: 400,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards: cards.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy_uid]),
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    let puid = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(&mut state, puid, PowerKind::Stampede, 2);
    let _ = start_player_turn(&mut state);

    // Override whatever the opening draw did: the two Attacks in hand, every
    // Defend back in the draw pile (SeekerStrike's select reads the DRAW pile
    // and no-ops on an empty one).
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = draw_uids.iter().copied().collect();
    state.player.hand.push(ss1);
    state.player.hand.push(ss2);
    state.player.energy = 3;
    state.energy_wasted = 0;
    state
}

/// Drive the first legal non-empty `ResolveChoice`.
fn answer_choice(state: &mut CombatState) -> StepResult {
    let pick = legal_actions(state)
        .into_iter()
        .find_map(|a| match a {
            Action::ResolveChoice { selection } if !selection.is_empty() => Some(selection),
            _ => None,
        })
        .expect("a non-empty ResolveChoice is legal while a select is surfaced");
    apply(state, Action::ResolveChoice { selection: pick })
}

#[test]
fn stampede_batch_awaits_each_auto_play() {
    let mut state = stampede_state();
    let round_at_end_turn = state.round;

    // ── Iteration 1 surfaces its select ────────────────────────────────────
    assert_eq!(
        apply(&mut state, Action::EndTurn),
        StepResult::PendingChoice,
        "the surfaced draw-select must suspend the end-turn chain"
    );

    assert_eq!(
        state.pending_end_turn_resume,
        Some(EndTurnResume::AutoPostPlay {
            stampede_remaining: 1
        }),
        "the chain must suspend INSIDE AutoPostPlay with the batch's remaining \
         iteration recorded — not later, after BeforeTurnEnd"
    );
    assert_eq!(
        state.player.play.len(),
        1,
        "C# is still inside its first `await`, so exactly ONE card is mid-play; \
         running the batch ahead used to leave both SEEKER_STRIKEs on the pile"
    );
    assert!(
        state.pending_autoplay_resume.is_some(),
        "the awaited card's tail is parked"
    );
    assert_eq!(
        state.energy_wasted, 0,
        "the unspent-energy snapshot lives after the batch, so it must NOT have \
         been taken yet — it is the cheapest observable proof that nothing past \
         the `await` ran"
    );
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(
        state.round, round_at_end_turn,
        "the turn has not rolled over"
    );

    // ── Answering it resumes the batch, which surfaces iteration 2 ─────────
    assert_eq!(
        answer_choice(&mut state),
        StepResult::PendingChoice,
        "the second Stampede attack is played on resume and surfaces its own \
         select — C# blocks once per awaited selection"
    );
    assert_eq!(
        state.pending_end_turn_resume,
        Some(EndTurnResume::AutoPostPlay {
            stampede_remaining: 0
        }),
        "batch exhausted; the rest of phase one is still owed"
    );
    assert_eq!(
        state.player.play.len(),
        1,
        "again exactly one card mid-play"
    );
    assert_eq!(state.energy_wasted, 0, "still short of the snapshot");

    // ── Answering that one runs the rest of the chain ──────────────────────
    let done = answer_choice(&mut state);
    assert!(
        matches!(
            done,
            StepResult::Advanced | StepResult::CombatWon | StepResult::CombatLost
        ),
        "the end-turn chain completes once the last awaited select is answered: {done:?}"
    );
    assert!(
        state.pending_end_turn_resume.is_none(),
        "the resume marker must not survive the turn"
    );
    assert!(
        state.pending_autoplay_resume.is_none(),
        "no auto-play may stay parked past the turn"
    );
    assert!(
        state.player.play.is_empty(),
        "both Stampede cards reached their result pile: {:?}",
        state.player.play
    );
    assert_eq!(
        state.energy_wasted, 3,
        "the post-batch snapshot ran exactly once, with the turn's 3 unspent energy"
    );
    assert!(
        state.round > round_at_end_turn,
        "the next player turn started"
    );
}
