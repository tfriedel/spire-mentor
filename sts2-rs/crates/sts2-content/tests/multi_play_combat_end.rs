//! GH #791 (S6): a multi-played card stops replaying once combat is ending.
//!
//! v0.111.0 `CardModel.OnPlayWrapper` re-checks `IsOverOrEnding` at the top of
//! every play iteration (`MegaCrit.Sts2.Core.Models/CardModel.cs:1896-1901`):
//!
//! [decompiled C# omitted — see the cited path]
//!
//! v0.107.1 has no such check (`CardModel.cs:1905`), so a OneTwoPunch / Echo
//! Form copy still runs its `BeforeCardPlayed` / `OnPlay` / affliction /
//! `AfterCardPlayed` after the first copy killed the last enemy. The Weighted
//! affliction's per-play energy drain (`Weighted.OnPlay`, inside the loop) and
//! the `CardPlayFinished` tally make the skipped copy observable.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::card_afflictions::{CardAffliction, CardAfflictionKind};
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

/// One Nibbit at `enemy_hp`, player on 3 energy in the play phase.
fn fresh(version: GameVersion, enemy_hp: i32) -> CombatState {
    content_dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: enemy_hp,
        max_hp: 40,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn add_player_power(state: &mut CombatState, id: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(id).expect("nz");
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

/// A cost-1 Strike carrying Weighted(1): every resolved play drains 1 more
/// energy, so `energy` counts the plays that actually ran.
fn push_weighted_strike(state: &mut CombatState) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = Card {
        uid,
        card_id: CardId::StrikeIronclad,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 6,
        base_block: 0,
        ..Card::pod_defaults()
    };
    card.affliction = Some(CardAffliction {
        kind: CardAfflictionKind::Weighted,
        stacks: 1,
        applied_exhaust: false,
    });
    state.cards.push(card);
    state.player.hand.push(uid);
}

fn play_lethal(state: &mut CombatState) {
    push_weighted_strike(state);
    let _ = sts2_core::combat_loop::play_card(state, 0, Some(0));
}

/// v0.107.1: the OneTwoPunch copy still resolves after the first hit killed the
/// lone enemy — cost 1 + Weighted twice = 3 energy, two finished plays.
#[test]
fn one_two_punch_replay_runs_after_lethal_v0107() {
    let mut state = fresh(GameVersion::V0_107_1, 1);
    add_player_power(&mut state, 7601, PowerKind::OneTwoPunch, 1);
    play_lethal(&mut state);
    assert_eq!(state.player.energy, 0, "both copies drained Weighted");
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}

/// v0.111.0: the copy is skipped — only the first play's cost + drain (2).
#[test]
fn one_two_punch_replay_stops_after_lethal_v0111() {
    let mut state = fresh(GameVersion::V0_111_0, 1);
    add_player_power(&mut state, 7601, PowerKind::OneTwoPunch, 1);
    play_lethal(&mut state);
    assert_eq!(state.player.energy, 1, "the replay never ran");
    assert_eq!(state.combat_log.player_card_plays_this_combat, 1);
}

/// v0.111.0: a non-lethal first hit leaves the replay alone.
#[test]
fn one_two_punch_replay_runs_when_enemy_survives_v0111() {
    let mut state = fresh(GameVersion::V0_111_0, 30);
    add_player_power(&mut state, 7601, PowerKind::OneTwoPunch, 1);
    play_lethal(&mut state);
    assert_eq!(state.player.energy, 0);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}

/// v0.107.1: the Echo Form copy runs after a lethal first copy.
#[test]
fn echo_form_replay_runs_after_lethal_v0107() {
    let mut state = fresh(GameVersion::V0_107_1, 1);
    add_player_power(&mut state, 7602, PowerKind::EchoForm, 1);
    play_lethal(&mut state);
    assert_eq!(state.player.energy, 0);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}

/// v0.111.0: the Echo Form copy is the loop's second iteration — skipped.
#[test]
fn echo_form_replay_stops_after_lethal_v0111() {
    let mut state = fresh(GameVersion::V0_111_0, 1);
    add_player_power(&mut state, 7602, PowerKind::EchoForm, 1);
    play_lethal(&mut state);
    assert_eq!(state.player.energy, 1);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 1);
}

/// v0.111.0: killing an Infested enemy is not "ending" — its Wrigglers spawn
/// inside the killing hit (C# `InfestedPower.AfterDeath`; Rust's death
/// chokepoint), so they are alive when the loop re-checks and the copy plays on.
#[test]
fn replay_continues_while_infested_spawn_pending_v0111() {
    let mut state = fresh(GameVersion::V0_111_0, 1);
    let pid = NonZeroU32::new(7700).expect("nz");
    state.powers.push(Power::new(pid, PowerKind::Infested, 4));
    state.creatures[0].powers.push(PowerUid(pid));
    add_player_power(&mut state, 7601, PowerKind::OneTwoPunch, 1);
    play_lethal(&mut state);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}
