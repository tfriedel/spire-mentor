//! Bead `sts2-rs-2itwm`: two auto-play drivers that let a SURFACED card-select
//! run past them, stranding cards on the Play pile for the rest of the combat
//! (turn-boundary invariant #18, ~390 per 3M randomized combats).
//!
//! Both are the same C# shape as `sts2-rs-eirvo`: `await CardCmd.AutoPlay(card)`
//! blocks until the awaited `CardSelectCmd` is answered, so nothing after it in
//! the enclosing loop — or after the loop, in the enclosing turn-start /
//! `OnPlayWrapper` tail — may run first.
//!
//! * `MayhemPower.AfterAutoPrePlayPhaseEntered` →
//!   `CardPileCmd.AutoPlayFromDrawPile(Amount, Top)`. The Rust phase-2 loop
//!   played the next collected sibling on top of the suspended one and let the
//!   `AfterPlayerTurnStart` cluster finish its turn-start tail, so the turn
//!   rolled over with the select still open and the siblings still on Play.
//!
//! * `HistoryCourse`'s `CardCmd.AutoPlay(card.CreateDupe())`. The dupe's
//!   `OnPlayWrapper` runs `GeneratePlayCount` replays of the whole play body
//!   before the ONE disposal at the end, but only the FIRST dispatch was held
//!   under `autoplay_inline_resolve` — so a replayed dupe whose `OnPlay`
//!   auto-played a surfacing card suspended, and the disposal's blind
//!   `play.pop()` then removed the SURFACING card instead of the dupe.

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, legal_actions, Action, CardUid, CombatState, Creature, CreatureUid, IntentHistory,
    MonsterId, MonsterKind, PlayerCombatState, PowerKind, Relic, RelicKind, StepResult,
};

/// A fat lone enemy: no auto-played attack may end the combat (an ending combat
/// suppresses the select entirely — C# `CardSelectCmd` returns empty when
/// `IsEnding` — which would defeat the point of these tests).
fn lone_fat_enemy() -> (Vec<Creature>, CreatureUid) {
    let enemy_uid = CreatureUid::new(1).expect("non-zero");
    let creatures = vec![Creature {
        id: enemy_uid.0,
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
    (creatures, enemy_uid)
}

fn push_card(cards: &mut sts2_core::CardStore, card_id: CardId, card_type: CardType) -> CardUid {
    let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
    cards.push(Card {
        uid,
        card_id,
        cost: 1,
        card_type,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 5,
        ..Card::pod_defaults()
    });
    uid
}

/// Five DefendIronclads (the turn-start draw fills the hand with them, so the
/// UPGRADE_IN_HAND select an ARMAMENTS opens has five options and therefore
/// SURFACES rather than auto-resolving) followed by `tail` — the cards the
/// auto-play driver under test picks up.
fn state_with_draw_tail(tail: &[CardId], relics: Vec<Relic>) -> (CombatState, Vec<CardUid>) {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();

    let mut cards = sts2_core::CardStore::new();
    let filler: Vec<CardUid> = (0..5)
        .map(|_| push_card(&mut cards, CardId::DefendIronclad, CardType::Skill))
        .collect();
    let tail_uids: Vec<CardUid> = tail
        .iter()
        .map(|&id| push_card(&mut cards, id, CardType::Skill))
        .collect();

    let (creatures, enemy_uid) = lone_fat_enemy();
    let player = PlayerCombatState {
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
        cards,
        relics: relics.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy_uid]),
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    // Set the pile AFTER `start_combat` (which shuffles the starting deck) and
    // BEFORE the turn setup draws from it, so the order below is exactly what
    // the turn-start draw and then the auto-play driver see.
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = filler.iter().chain(tail_uids.iter()).copied().collect();
    (state, tail_uids)
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
fn mayhem_batch_awaits_each_auto_play() {
    // MayhemPower(2) collects ARMAMENTS then DEFEND off the draw top. The
    // ARMAMENTS surfaces its upgrade-select, so C# is still inside its first
    // `await`: the DEFEND has not been played and the turn-start tail has not
    // run.
    let (mut state, tail) =
        state_with_draw_tail(&[CardId::Armaments, CardId::DefendIronclad], Vec::new());
    let (armaments, sibling) = (tail[0], tail[1]);
    let puid = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(&mut state, puid, PowerKind::Mayhem, 2);

    let res = start_player_turn(&mut state);
    assert_eq!(
        res,
        StepResult::PendingChoice,
        "the surfaced upgrade-select must suspend the turn start"
    );
    assert!(
        state.pending_choice.is_some(),
        "the select is the open decision"
    );
    assert!(
        state.pending_autoplay_resume.is_some(),
        "the awaited ARMAMENTS' tail is parked"
    );
    assert_eq!(
        state
            .pending_autoplay_resume
            .as_ref()
            .map(|r| r.remaining.as_slice()),
        Some([sibling].as_slice()),
        "the not-yet-played batch sibling is owed to the resume, NOT played \
         ahead of the awaited card"
    );
    assert!(
        state.pending_cluster_resume.is_some(),
        "the `AfterPlayerTurnStart` cluster must stay suspended — its turn-start \
         tail runs only after the awaited select is answered"
    );
    assert!(
        state.player.play.contains(&armaments) && state.player.play.contains(&sibling),
        "both collected cards are still on Play mid-batch: {:?}",
        state.player.play
    );

    // Answering it finishes ARMAMENTS, plays the sibling, and runs the tail.
    let done = answer_choice(&mut state);
    assert!(
        matches!(done, StepResult::Advanced | StepResult::PendingChoice),
        "the batch resumes on the answer: {done:?}"
    );
    assert!(
        state.player.play.is_empty(),
        "every batch card reached its result pile: {:?}",
        state.player.play
    );
    assert!(
        state.pending_autoplay_resume.is_none(),
        "no auto-play may stay parked"
    );
    assert!(
        state.pending_cluster_resume.is_none(),
        "the cluster's turn-start tail ran"
    );
    assert!(
        state
            .player
            .hand
            .iter()
            .any(|u| { state.cards[u.0.get() as usize - 1].upgraded }),
        "the ARMAMENTS upgrade actually landed on a hand card"
    );
}

#[test]
fn history_course_dupe_replay_awaits_its_nested_auto_play() {
    // HistoryCourse replays last round's last Attack/Skill as a free dupe. The
    // dupe here is a HAVOC carrying `BaseReplayCount = 1`, so C#
    // `OnPlayWrapper` runs its `OnPlay` TWICE inside the one awaited
    // `CardCmd.AutoPlay(dupe)` before the single disposal at the end:
    //   play 1 → auto-plays the draw top (a DEFEND — no select)
    //   play 2 → auto-plays the next draw top (an ARMAMENTS — surfacing select)
    // The second dispatch used to run with `autoplay_inline_resolve` already
    // restored, so the ARMAMENTS suspended and sat on Play ABOVE the dupe —
    // and the disposal's blind `play.pop()` dropped the ARMAMENTS and left the
    // HAVOC dupe stranded. Now the ARMAMENTS select prompts the player (GH
    // #1131), and the turn-start tail waits for the answer.
    let relics = vec![Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::HistoryCourse,
        counter: 0,
        is_wax: false,
        is_melted: false,
    }];
    let (mut state, tail) =
        state_with_draw_tail(&[CardId::DefendIronclad, CardId::Armaments], relics);
    let armaments = tail[1];

    // The card HistoryCourse replays: a HAVOC that is NOT in any pile (C#
    // `CreateDupe()` clones the model, wherever the original ended up).
    let havoc = push_card(&mut state.cards, CardId::Havoc, CardType::Skill);
    state.cards[havoc.0.get() as usize - 1].base_replay_count = 1;
    // `start_player_turn` rotates `last_attack_skill_played` into
    // `prev_round_last_attack_skill_played` before the relic tail reads it, so
    // stamp the CURRENT slot and let the rotation do its job.
    state.combat_log.last_attack_skill_played = Some((CardId::Havoc, false, Some(havoc)));
    // HistoryCourse is inert on round 1.
    state.round = 2;

    let res = start_player_turn(&mut state);
    assert_eq!(
        res,
        StepResult::PendingChoice,
        "the ARMAMENTS select prompts"
    );
    assert!(
        state.pending_play_phase_tail.is_some(),
        "the turn-start tail waits"
    );
    assert!(state.player.play.contains(&armaments));
    let _ = answer_choice(&mut state);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_play_phase_tail.is_none(), "the tail ran");
    assert!(
        state.player.play.is_empty(),
        "neither the dupe nor its nested auto-plays may be left on the Play \
         pile — that is the invariant-#18 leak: {:?}",
        state.player.play
    );
    assert!(
        state.pending_autoplay_resume.is_none(),
        "the answer finishes the ARMAMENTS and the dupe, so nothing stays \
         parked"
    );
    assert!(
        !state.player.draw.contains(&armaments),
        "the replayed HAVOC auto-played the ARMAMENTS off the draw top"
    );
    assert!(
        state.player.exhaust.contains(&armaments),
        "HAVOC force-exhausts what it auto-plays, so the ARMAMENTS is in the \
         exhaust pile — it used to be dropped out of every pile by the dupe \
         disposal's blind `play.pop()`: exhaust={:?} discard={:?}",
        state.player.exhaust,
        state.player.discard
    );
}
