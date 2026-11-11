//! GH #806 (group G3): three pre-existing bugs whose C# is identical in
//! v0.107.1 and v0.111.0, so every test runs at both builds.
//!
//! * `IAmInvincible.AfterAutoPostPlayPhaseEntered` (IAmInvincible.cs:29-39)
//!   fires once per card instance as the turn enters AutoPostPlay, i.e. before
//!   `BeforeTurnEnd` and the hand flush
//!   (CombatManager.EndPlayerTurnPhaseOneInternal). Rust used to run it after
//!   the flush, as a `while draw-top is IAmInvincible` loop that replayed a
//!   copy Nostalgia had put back on top.
//! * `SuckPower.AfterAttack` (SuckPower.cs:22-47) counts hits whose
//!   `DamageResult.UnblockedDamage > 0`, filled before death prevention, so a
//!   lethal hit that FairyInABottle revives still grants Strength.

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::potions::PotionKind;
use sts2_core::{
    CardUid, CombatPhase, CombatState, CreatureUid, GameVersion, MonsterKind, PowerKind, Relic,
    RelicKind, Side,
};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn push_card(state: &mut CombatState, id: CardId, block: i32) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: block,
        ..Card::pod_defaults()
    });
    uid
}

fn add_relic(state: &mut CombatState, kind: RelicKind) {
    let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("u32")).expect("nz");
    state.relics.push(Relic {
        id,
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn player_power(state: &CombatState, kind: PowerKind) -> i32 {
    state.player_power_amount(kind)
}

/// Player in the play phase with IAmInvincible on top of the draw pile and a
/// filler card beneath it. Returns (state, IAmInvincible uid, filler uid).
fn turn_end_state(version: GameVersion) -> (CombatState, CardUid, CardUid) {
    content_dispatch::ensure_installed();
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = Side::Player;
    let iai = push_card(&mut state, CardId::IAmInvincible, 10);
    let filler = push_card(&mut state, CardId::DefendSilent, 5);
    state.player.draw.push(iai);
    state.player.draw.push(filler);
    (state, iai, filler)
}

#[test]
#[serial_test::serial(global_registry)]
fn i_am_invincible_auto_plays_before_the_flush() {
    for version in VERSIONS {
        let (mut state, iai, _) = turn_end_state(version);
        let held = push_card(&mut state, CardId::DefendSilent, 5);
        state.player.hand.push(held);
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(
            state.player.block, 10,
            "{version:?}: one IAmInvincible play"
        );
        assert!(
            state.player.discard.contains(&iai),
            "{version:?}: the auto-played card reaches discard in AutoPostPlay"
        );
        assert_eq!(
            state.player.hand.as_slice(),
            &[held],
            "{version:?}: the hand flush has not run yet"
        );
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn fake_orichalcum_sees_i_am_invincible_block() {
    for version in VERSIONS {
        let (mut state, _, _) = turn_end_state(version);
        add_relic(&mut state, RelicKind::FakeOrichalcum);
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(
            state.player.block, 10,
            "{version:?}: FakeOrichalcum's Block == 0 check runs after the auto-play"
        );
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn nostalgia_returned_copy_is_not_replayed() {
    for version in VERSIONS {
        let (mut state, iai, _) = turn_end_state(version);
        let player = state.player.creature_uid.expect("player creature");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::Nostalgia,
            1,
        );
        assert_eq!(player_power(&state, PowerKind::Nostalgia), 1);
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(
            state.player.block, 10,
            "{version:?}: each IAmInvincible instance answers the hook once"
        );
        assert_eq!(
            state.player.draw.first(),
            Some(&iai),
            "{version:?}: Nostalgia put the played copy back on top"
        );
    }
}

/// NostalgiaPower counts the plays STARTED before this one when C# resolves
/// the result pile, ahead of the replay loop (CardModel.cs:1891), so a card's
/// own Replay iterations never cost it the redirect. Rust read the live tally
/// after the replays (Ironclad fuzz seed 5917: a Replay-3 Ultimate Strike
/// auto-played by Hellraiser went to discard instead of the draw top).
#[test]
#[serial_test::serial(global_registry)]
fn nostalgia_redirects_an_auto_played_card_with_replay() {
    for version in VERSIONS {
        let (mut state, iai, _) = turn_end_state(version);
        state.cards[iai.0.get() as usize - 1].base_replay_count = 1;
        let player = state.player.creature_uid.expect("player creature");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::Nostalgia,
            1,
        );
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(
            state.player.block, 20,
            "{version:?}: played twice (Replay 1)"
        );
        assert_eq!(
            state.player.draw.first(),
            Some(&iai),
            "{version:?}: Nostalgia still sends it to the draw top"
        );
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn stacked_copies_each_play_once() {
    for version in VERSIONS {
        let (mut state, iai, filler) = turn_end_state(version);
        let second = push_card(&mut state, CardId::IAmInvincible, 10);
        state.player.draw = [iai, second, filler].into_iter().collect();
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(state.player.block, 20, "{version:?}: both copies play");
        assert_eq!(state.player.draw.as_slice(), &[filler]);
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn unceasing_top_draws_after_the_auto_play() {
    // The hand is already empty at AutoPostPlay entry; the auto-play's
    // trailing CheckForEmptyHand (CardModel.cs:1993) fires AfterHandEmptied,
    // and AutoPostPlay is a valid UnceasingTop phase (UnceasingTop.cs:25-32).
    for version in VERSIONS {
        let (mut state, _, filler) = turn_end_state(version);
        add_relic(&mut state, RelicKind::UnceasingTop);
        let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
        assert_eq!(
            state.player.hand.as_slice(),
            &[filler],
            "{version:?}: UnceasingTop draws 1 after the auto-play"
        );
    }
}

fn suck_state(version: GameVersion, player_hp: i32, fairy: bool) -> (CombatState, CreatureUid) {
    content_dispatch::ensure_installed();
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = player_hp;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    if fairy {
        state.player.potions.push(PotionKind::FairyInABottle);
    }
    state.phase = CombatPhase::EnemyTurnExec;
    state.side = Side::Enemy;
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 40;
        c.current_hp = 40;
        c.intent = Some(sts2_core::monsters::Intent::Attack { damage: 10 });
        c.ai = None;
    }
    sts2_core::card_effects::apply_power_to_creature(&mut state, enemy, PowerKind::Suck, 2);
    (state, enemy)
}

fn enemy_strength(state: &CombatState, enemy: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == enemy.0)
        .map_or(0, |c| {
            c.powers
                .iter()
                .filter_map(|uid| state.powers.get_by_id(uid.0))
                .filter(|p| p.kind == PowerKind::Strength)
                .map(|p| p.amount)
                .sum()
        })
}

#[test]
#[serial_test::serial(global_registry)]
fn suck_counts_a_lethal_hit_the_fairy_revives() {
    for version in VERSIONS {
        let (mut state, enemy) = suck_state(version, 5, true);
        let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
        assert_eq!(
            state.player.current_hp, 15,
            "{version:?}: Fairy revives to 30%"
        );
        assert!(
            state.player.potions.is_empty(),
            "{version:?}: the Fairy is spent"
        );
        assert_eq!(
            enemy_strength(&state, enemy),
            2,
            "{version:?}: UnblockedDamage was 5 > 0 before the revive"
        );
    }
}

#[test]
#[serial_test::serial(global_registry)]
fn suck_still_counts_an_ordinary_unblocked_hit_and_skips_a_blocked_one() {
    for version in VERSIONS {
        let (mut state, enemy) = suck_state(version, 50, false);
        let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
        assert_eq!(state.player.current_hp, 40);
        assert_eq!(enemy_strength(&state, enemy), 2, "{version:?}");

        let (mut blocked, enemy) = suck_state(version, 50, false);
        blocked.player.block = 20;
        blocked.sync_creature_block_from_player();
        let _ = sts2_core::combat_loop::enemy_turn_exec(&mut blocked);
        assert_eq!(blocked.player.current_hp, 50);
        assert_eq!(enemy_strength(&blocked, enemy), 0, "{version:?}");
    }
}
