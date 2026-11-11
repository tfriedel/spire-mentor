//! GH #500 residual: two ORDER defects in the turn-start auto-play cluster.
//!
//! Both have the same C# reference shape — `await` means the awaited work is
//! finished before ANYTHING that follows it in the enclosing scope runs — and
//! both are invisible to a count-based state diff, because the set of cards
//! played and the piles they end in are identical either way. Only the ORDER
//! moves, and it cashes out later (a `StableShuffle` dealing a different hand,
//! or a play-counting power reading one card too few).
//!
//! **A. The turn-start tail must not run from a NESTED choice resolve.**
//! `resolve_choice_action` is re-entrant. A card played out of the auto-play
//! resume's `remaining` drains its own `OnPlay` select INLINE, re-entering the
//! resolver with `flush_played = false` while `pending_autoplay_resume` is
//! already taken — and the `pending_cluster_resume` arm then ran the whole
//! turn-start tail from inside that card's `OnPlay`. C# cannot: the tail is
//! everything after `Hook.AfterAutoPrePlayPhaseEntered`'s `await`, and the
//! sibling's `OnPlayWrapper` has not returned yet — so its `AfterCardPlayed`
//! has not fired. `HistoryCourse`'s dupe therefore replayed against a
//! card-play counter that was one short (Ironclad oracle seed 29: a Slow'd
//! BygoneEffigy took 13 instead of 14 from the duped SeekerStrike).
//!
//! **B. An auto-play LOOP must finish before its own card's tail.**
//! `Catastrophe.OnPlay` awaits every `CardCmd.AutoPlay` in its `for` loop, so
//! the loop drains completely before `OnPlay` returns — before
//! `OnPlayWrapper` fires `AfterCardPlayed`, and before the enclosing
//! `AutoPlayFromDrawPile` batch moves on to its next collected sibling. The
//! resume ran the whole `parent_chain` (each enclosing card's tail AND its
//! siblings) before the `continuation`, so a Mayhem sibling jumped ahead of
//! Catastrophe's own next iteration — and from there the two engines were
//! taking different cards off the draw top (oracle seed 189).

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
/// `IsEnding`), and its HP is the readout for test A.
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
        cost: sts2_core::CardEnergyCost::new(1),
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
/// UPGRADE_IN_HAND select an ARMAMENTS opens has several options and therefore
/// SURFACES rather than auto-resolving) followed by `tail` — the cards the
/// auto-play drivers under test pick up.
fn state_with_draw_tail(
    tail: &[CardId],
    relics: Vec<Relic>,
) -> (CombatState, Vec<CardUid>, CreatureUid) {
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
    // the turn-start draw and then the auto-play drivers see.
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = filler.iter().chain(tail_uids.iter()).copied().collect();
    (state, tail_uids, enemy_uid)
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

fn discard_ids(state: &CombatState) -> Vec<CardId> {
    state
        .player
        .discard
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect()
}

/// **A.** MayhemPower(2) collects two ARMAMENTS off the draw top. The first
/// surfaces its upgrade-select; the second is played out of the resume's
/// `remaining` and drains ITS select inline. HistoryCourse's dupe is part of
/// the turn-start tail, so it must land after BOTH have finished — i.e. with
/// the Slow'd enemy's per-turn card-play counter at 2, not 1.
#[test]
fn history_course_dupe_lands_after_the_whole_mayhem_batch() {
    let relics = vec![Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::HistoryCourse,
        counter: 0,
        is_wax: false,
        is_melted: false,
    }];
    let (mut state, _tail, enemy_uid) =
        state_with_draw_tail(&[CardId::Armaments, CardId::Armaments], relics);

    // Slow (BygoneEffigy's): every card play bumps the owner's per-turn
    // counter, and a powered attack landing on the owner is scaled by
    // `1 + 0.1 * counter`. That makes the DUPE's damage a direct readout of how
    // many plays had FINISHED when it resolved.
    sts2_core::card_effects::apply_power_to_creature(&mut state, enemy_uid, PowerKind::Slow, 1);

    // The card HistoryCourse replays: a 6-damage STRIKE that is in no pile (C#
    // `CreateDupe()` clones the model wherever the original ended up).
    let strike = push_card(&mut state.cards, CardId::StrikeIronclad, CardType::Attack);
    {
        let c = &mut state.cards[strike.0.get() as usize - 1];
        c.card_type = CardType::Attack;
        c.target_type = TargetType::Enemy;
        c.base_damage = 6;
    }
    // `start_player_turn` rotates `last_attack_skill_played` into
    // `prev_round_last_attack_skill_played` before the relic tail reads it.
    state.combat_log.last_attack_skill_played = Some((CardId::StrikeIronclad, false, Some(strike)));
    // HistoryCourse is inert on round 1.
    state.round = 2;

    let mayhem_owner = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        mayhem_owner,
        PowerKind::Mayhem,
        2,
    );

    let res = start_player_turn(&mut state);
    assert_eq!(
        res,
        StepResult::PendingChoice,
        "the first ARMAMENTS' upgrade-select suspends the turn start"
    );
    assert!(
        state.pending_cluster_resume.is_some(),
        "the turn-start tail (HistoryCourse included) is parked behind it"
    );

    let _ = answer_choice(&mut state);
    assert!(
        state.pending_choice.is_some(),
        "the second ARMAMENTS awaits its selection"
    );
    assert!(state.pending_cluster_resume.is_some());
    let _ = answer_choice(&mut state);
    assert!(
        state.pending_choice.is_none(),
        "both ARMAMENTS selections have completed"
    );
    assert!(
        state.pending_cluster_resume.is_none(),
        "the turn-start tail ran once the batch was done"
    );
    assert!(
        state.player.play.is_empty(),
        "no card is stranded on the Play pile: {:?}",
        state.player.play
    );

    // Both ARMAMENTS finished before the dupe, so SlowAmount is 2 and the
    // 6-damage dupe is scaled to `floor(6 * 1.2) = 7`. With the tail running
    // from inside the second ARMAMENTS' inline resolve it saw SlowAmount 1 and
    // dealt `floor(6 * 1.1) = 6`.
    let enemy_hp = state
        .creatures
        .iter()
        .find(|c| c.id == enemy_uid.0)
        .expect("enemy")
        .current_hp;
    assert_eq!(
        enemy_hp, 393,
        "the HistoryCourse dupe must resolve AFTER both Mayhem auto-plays have \
         fired their AfterCardPlayed (SlowAmount 2 → 7 damage, not 1 → 6)"
    );
}

/// **B.** MayhemPower(2) collects CATASTROPHE and a DEFEND sibling.
/// CATASTROPHE auto-plays two ARMAMENTS out of the draw pile; the first
/// surfaces its select. On resume the loop's SECOND iteration must run before
/// CATASTROPHE's own tail, and both before the Mayhem sibling.
#[test]
fn catastrophes_loop_finishes_before_its_tail_and_the_batch_sibling() {
    let (mut state, tail, _enemy_uid) = state_with_draw_tail(
        &[
            CardId::Catastrophe,
            CardId::DefendIronclad,
            CardId::Armaments,
            CardId::Armaments,
        ],
        Vec::new(),
    );
    let sibling = tail[1];

    let puid = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(&mut state, puid, PowerKind::Mayhem, 2);

    let res = start_player_turn(&mut state);
    assert_eq!(
        res,
        StepResult::PendingChoice,
        "the first auto-played ARMAMENTS surfaces its upgrade-select"
    );
    let resume = state
        .pending_autoplay_resume
        .as_ref()
        .expect("the awaited ARMAMENTS' tail is parked");
    assert_eq!(
        resume.parent_chain.len(),
        1,
        "CATASTROPHE unwound to the same gate and owes its tail"
    );
    assert!(
        resume.parent_chain[0].continuation.is_some(),
        "CATASTROPHE's remaining loop iteration rides on ITS OWN frame — left in \
         the top-level slot it would run after the whole chain"
    );
    assert_eq!(
        resume.parent_chain[0].remaining.as_slice(),
        [sibling].as_slice(),
        "the Mayhem sibling is owed to CATASTROPHE's frame, after its tail"
    );
    assert!(
        resume.continuation.is_none(),
        "the top-level slot is for a HAND-played loop card only"
    );

    let _ = answer_choice(&mut state);
    assert!(
        state.pending_choice.is_some(),
        "the nested second ARMAMENTS awaits its selection"
    );
    assert!(state.pending_autoplay_resume.is_some());
    let _ = answer_choice(&mut state);
    assert!(
        state.pending_choice.is_none(),
        "both nested ARMAMENTS selections have completed"
    );
    assert!(
        state.pending_autoplay_resume.is_none(),
        "no auto-play may stay parked"
    );
    assert!(
        state.player.play.is_empty(),
        "no card is stranded on the Play pile: {:?}",
        state.player.play
    );
    assert_eq!(
        discard_ids(&state),
        vec![
            CardId::Armaments,
            CardId::Armaments,
            CardId::Catastrophe,
            CardId::DefendIronclad,
        ],
        "C# order: both awaited auto-plays, then CATASTROPHE's own tail, then \
         the Mayhem batch sibling"
    );
}

/// A Mayhem-auto-played VoidForm calls `PlayerCmd.EndTurn` (VoidForm.cs:26)
/// during the AutoPrePlay phase, so C# ends the turn as soon as the play phase
/// opens: the enemy acts and the NEXT player turn starts. Rust only honoured
/// that for the Whispering Earring and left the player in a turn C# had
/// already ended (Regent quick seed 244).
#[test]
fn mayhem_auto_played_void_form_ends_the_turn() {
    let (mut state, tail, _enemy) = state_with_draw_tail(&[CardId::VoidForm], Vec::new());
    state.cards[tail[0].0.get() as usize - 1].card_type = CardType::Power;
    let owner = state.player.creature_uid.expect("player creature mirror");
    sts2_core::card_effects::apply_power_to_creature(&mut state, owner, PowerKind::Mayhem, 1);
    let round0 = state.round;
    let _ = start_player_turn(&mut state);
    assert!(
        state.round > round0,
        "VoidForm's EndTurn ran the enemy turn and the next player turn"
    );
    assert!(!state.player.request_end_turn_after_play);
}
