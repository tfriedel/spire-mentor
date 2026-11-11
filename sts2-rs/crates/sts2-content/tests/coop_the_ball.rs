//! The Ball in a co-op fight (GH #1017): after its play the card goes to a
//! random living teammate, carrying its grown damage.
//!
//! C# v0.111.0 `TheBall.GetResultLocationForCardPlay` (`TheBall.cs:59-77`):
//! the living players other than the owner, in `CombatState.GetTeammatesOf`
//! order, one picked with `Rng.CombatTargets.NextItem`; a Discard result
//! becomes their draw pile at `CardPilePosition.Random`
//! (`Rng.Shuffle.NextInt(Count + 1)`, `CardPileCmd.cs:511`), any other result
//! (Exhaust) is kept but lands in their pile. `OnPlayWrapper` hands it over
//! with `CardPileCmd.GiveToAnotherPlayer` after the play (`CardModel.cs:1981`).

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{CardId, KeywordOverride};
use sts2_core::{Action, CardUid, CombatPhase, CombatState, MonsterKind, PlayerCombatState};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, false).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Us and `mates` teammate seats in a v0.111.0 fight with a Nibbit, our play
/// phase open, 5 energy each; every teammate's draw pile holds four Strikes.
fn fight(mates: usize, seed: i32) -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(seed),
        game_version: sts2_core::GameVersion::V0_111_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.init_player_creature();
    for _ in 0..mates {
        state.allies.push(PlayerCombatState {
            current_hp: 70,
            max_hp: 70,
            energy: 5,
            ..PlayerCombatState::default()
        });
    }
    for i in 0..mates {
        for _ in 0..4 {
            let s = mint(&mut state, CardId::StrikeIronclad);
            state.allies[i].draw.push(s);
        }
    }
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn enemy_hp(state: &CombatState) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.species.is_some())
        .expect("enemy")
        .current_hp
}

fn damage_of(state: &CombatState, uid: CardUid) -> i32 {
    state.cards[uid.0.get() as usize - 1].base_damage
}

/// Put a fresh Ball in our hand and play it at the enemy.
fn play_ball(state: &mut CombatState, keyword: Option<KeywordOverride>) -> CardUid {
    let uid = mint(state, CardId::TheBall);
    state.cards[uid.0.get() as usize - 1].keyword_override = keyword;
    state.player.hand.push(uid);
    let hand_index = u8::try_from(state.player.hand.len() - 1).expect("fits");
    let _ = sts2_core::apply(
        state,
        Action::PlayCard {
            hand_index,
            target: Some(0),
        },
    );
    assert!(!state.player.hand.contains(&uid), "The Ball was played");
    uid
}

fn in_our_piles(state: &CombatState, uid: CardUid) -> bool {
    let p = &state.player;
    [
        &p.hand[..],
        &p.draw[..],
        &p.discard[..],
        &p.exhaust[..],
        &p.play[..],
    ]
    .iter()
    .any(|pile| pile.contains(&uid))
}

#[test]
fn two_seats_the_ball_goes_to_the_teammates_draw_pile_at_a_random_place() {
    let mut state = fight(1, 4242);
    let mut targets = state.run_rng.combat_targets.clone();
    let mut shuffle = state.run_rng.shuffle.clone();
    let hp = enemy_hp(&state);
    let uid = play_ball(&mut state, None);

    assert_eq!(enemy_hp(&state), hp - 10);
    assert!(!in_our_piles(&state, uid), "the card left us");
    // One `CombatTargets.NextItem` over the one teammate, then one
    // `Shuffle.NextInt(4 + 1)` for the place in their draw pile.
    let _ = targets.next_int_exclusive(0, 1);
    let at = usize::try_from(shuffle.next_int(0, 4)).expect("index");
    assert_eq!(state.run_rng.combat_targets, targets);
    assert_eq!(state.run_rng.shuffle, shuffle);
    assert_eq!(state.allies[0].draw.len(), 5);
    assert_eq!(state.allies[0].draw[at], uid);
    assert_eq!(damage_of(&state, uid), 20, "the growth stays on the card");
}

#[test]
fn three_seats_the_ball_goes_to_the_teammate_the_combat_targets_stream_picks() {
    let mut picked = [0_u32; 2];
    for seed in 1..=24 {
        let mut state = fight(2, seed);
        let mut targets = state.run_rng.combat_targets.clone();
        let want = usize::try_from(targets.next_int_exclusive(0, 2)).expect("index");
        let uid = play_ball(&mut state, None);
        assert!(
            state.allies[want].draw.contains(&uid),
            "seed {seed}: teammate {want} gets it"
        );
        assert!(!state.allies[1 - want].draw.contains(&uid));
        assert!(!in_our_piles(&state, uid));
        picked[want] += 1;
    }
    assert!(
        picked[0] > 0 && picked[1] > 0,
        "both teammates get picked: {picked:?}"
    );
}

#[test]
fn a_dead_teammate_is_never_picked_but_the_roll_is_still_made() {
    for seed in 1..=8 {
        let mut state = fight(2, seed);
        state.allies[0].current_hp = 0;
        let mut targets = state.run_rng.combat_targets.clone();
        let uid = play_ball(&mut state, None);
        assert!(state.allies[1].draw.contains(&uid), "seed {seed}");
        let _ = targets.next_int_exclusive(0, 1);
        assert_eq!(state.run_rng.combat_targets, targets);
    }
}

#[test]
fn the_ball_stays_with_us_when_no_teammate_lives() {
    let mut state = fight(1, 4242);
    state.allies[0].current_hp = 0;
    let targets = state.run_rng.combat_targets.clone();
    let uid = play_ball(&mut state, None);
    assert_eq!(state.player.discard.as_slice(), [uid]);
    assert_eq!(
        state.run_rng.combat_targets, targets,
        "no teammate, no roll"
    );

    // Solo: the default result location.
    let mut solo = fight(0, 4242);
    let uid = play_ball(&mut solo, None);
    assert_eq!(solo.player.discard.as_slice(), [uid]);
    assert_eq!(damage_of(&solo, uid), 20);
}

#[test]
fn an_exhausting_ball_lands_in_the_teammates_exhaust_pile() {
    let mut state = fight(1, 4242);
    let shuffle = state.run_rng.shuffle.clone();
    let uid = play_ball(&mut state, Some(KeywordOverride::EXHAUST));
    // Only a Discard result becomes Draw/Random: Exhaust stays Exhaust, at
    // the teammate's (`GiveToAnotherPlayer(card, player, Exhaust, Bottom)`).
    assert_eq!(state.allies[0].exhaust.as_slice(), [uid]);
    assert!(state.player.exhaust.is_empty());
    assert!(!in_our_piles(&state, uid));
    assert_eq!(state.run_rng.shuffle, shuffle, "no random place to roll");
}

#[test]
fn a_ball_auto_played_by_havoc_goes_to_the_teammates_exhaust_pile() {
    let mut state = fight(1, 4242);
    let ball = mint(&mut state, CardId::TheBall);
    state.player.draw.insert(0, ball);
    let havoc = mint(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);
    let hp = enemy_hp(&state);
    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(enemy_hp(&state), hp - 10, "Havoc played The Ball");
    assert_eq!(state.allies[0].exhaust.as_slice(), [ball]);
    assert!(!in_our_piles(&state, ball));
    assert_eq!(damage_of(&state, ball), 20);
}

#[test]
fn the_teammate_plays_it_at_its_grown_damage_and_passes_it_back() {
    let mut state = fight(1, 4242);
    let uid = play_ball(&mut state, None);
    // The teammate draws it.
    state.allies[0].draw.retain(|u| *u != uid);
    state.allies[0].hand.push(uid);
    let hand_index = u8::try_from(state.allies[0].hand.len() - 1).expect("fits");
    let hp = enemy_hp(&state);
    let _ = sts2_core::coop::apply_as(
        &mut state,
        0,
        Action::PlayCard {
            hand_index,
            target: Some(0),
        },
    );
    assert_eq!(enemy_hp(&state), hp - 20, "it hits at its grown damage");
    assert_eq!(damage_of(&state, uid), 30);
    assert_eq!(state.allies[0].energy, 4, "the teammate paid for it");
    assert!(
        state.player.draw.contains(&uid),
        "back to us, the only other player"
    );
    let a = &state.allies[0];
    assert!(![
        &a.hand[..],
        &a.draw[..],
        &a.discard[..],
        &a.exhaust[..],
        &a.play[..]
    ]
    .iter()
    .any(|pile| pile.contains(&uid)));
}

/// A teammate's Ball picks among the others in the RUN's player order
/// (`GetTeammatesOf` walks `CombatState.Players`): with us third
/// (`seat_index` 2) the first seat's candidates are [seat 2, us], not
/// [us, seat 2].
#[test]
fn a_teammates_ball_picks_in_the_runs_player_order() {
    let mut picked = [0_u32; 2];
    for seed in 1..=24 {
        let mut state = fight(2, seed);
        state.coop.seat_index = 2;
        let uid = mint(&mut state, CardId::TheBall);
        state.allies[0].hand.push(uid);
        let hand_index = u8::try_from(state.allies[0].hand.len() - 1).expect("fits");
        let mut targets = state.run_rng.combat_targets.clone();
        let k = targets.next_int_exclusive(0, 2);
        let _ = sts2_core::coop::apply_as(
            &mut state,
            0,
            Action::PlayCard {
                hand_index,
                target: Some(0),
            },
        );
        if k == 0 {
            assert!(
                state.allies[1].draw.contains(&uid),
                "seed {seed}: seat 2 first"
            );
        } else {
            assert!(state.player.draw.contains(&uid), "seed {seed}: then us");
        }
        picked[usize::try_from(k).expect("index")] += 1;
    }
    assert!(picked[0] > 0 && picked[1] > 0, "{picked:?}");
}

/// The killing blow: C# `GiveToAnotherPlayer` returns early once combat is
/// over or ending (`CardPileCmd.cs:226`), and so does the owner's fallback
/// `CardPileCmd.Add(Draw, Random)` (`CardPileCmd.cs:379`, `:467`), before the
/// `Shuffle` read at `:511`. The teammate roll was already made at play start.
#[test]
fn a_killing_ball_reads_no_shuffle_and_stays_with_us() {
    let mut state = fight(1, 4242);
    let enemy = state.enemy_uids[0];
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.current_hp = 5;
    }
    let mut targets = state.run_rng.combat_targets.clone();
    let shuffle = state.run_rng.shuffle.clone();
    let uid = play_ball(&mut state, None);
    let _ = targets.next_int_exclusive(0, 1);
    assert_eq!(state.run_rng.combat_targets, targets, "the play-start roll");
    assert_eq!(state.run_rng.shuffle, shuffle, "no random place is read");
    assert!(!state.allies[0].draw.contains(&uid));
    assert!(state.coop.result_player.is_empty());
}

/// A Ball still in Play when a choice closes (its play left one open) goes to
/// the teammate rolled at its play start.
#[test]
fn a_ball_flushed_after_a_choice_goes_to_its_teammate() {
    let mut state = fight(1, 4242);
    let uid = mint(&mut state, CardId::TheBall);
    state.player.play.push(uid);
    sts2_core::coop::roll_result_player(&mut state, uid, CardId::TheBall);
    sts2_core::combat_loop::flush_played_cards_after_choice(&mut state);
    assert!(state.allies[0].draw.contains(&uid));
    assert!(!in_our_piles(&state, uid));
    assert!(state.coop.result_player.is_empty());
}

/// C# `EnergyCost.AfterCardPlayedCleanup()` (`CardModel.cs:2007`) runs for
/// every play: a free-this-turn Ball arrives at its teammate's full cost, by
/// hand or auto-played.
#[test]
fn a_handed_over_ball_drops_its_when_played_cost() {
    let mut state = fight(1, 4242);
    let uid = mint(&mut state, CardId::TheBall);
    state.cards[uid.0.get() as usize - 1].set_to_free_this_turn();
    state.player.hand.push(uid);
    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    assert!(state.allies[0].draw.contains(&uid));
    assert!(!state.cards[uid.0.get() as usize - 1]
        .cost
        .has(sts2_core::LocalCostMod::FREE_THIS_TURN));

    let mut state = fight(1, 4242);
    let ball = mint(&mut state, CardId::TheBall);
    state.cards[ball.0.get() as usize - 1].set_to_free_this_turn();
    state.player.draw.insert(0, ball);
    let havoc = mint(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);
    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(state.allies[0].exhaust.as_slice(), [ball]);
    assert!(!state.cards[ball.0.get() as usize - 1]
        .cost
        .has(sts2_core::LocalCostMod::FREE_THIS_TURN));
}
