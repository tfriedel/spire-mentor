//! Regression: a Weighted card's energy drain fires once PER play iteration,
//! including under EchoForm / OneTwoPunch replays.
//!
//! C# `Weighted.OnPlay` (Afflictions/Weighted.cs:8-11) calls
//! `PlayerCmd.LoseEnergy(Amount)`, and `CardModel.OnPlayWrapper`
//! (CardModel.cs:1513-1522) runs the affliction `OnPlay` INSIDE the
//! `for i < playCount2` loop — so a replayed Weighted card drains its `Amount`
//! once per play, not just on the first. Before the fix Rust applied the drain
//! exactly once, so a Weighted COMPACT replayed by EchoForm left the player
//! with 1 extra energy vs C#.
//!
//! Surfaced by Defect parity sweep seed 423 (DOORMAKER_BOSS GraspPower applies
//! Weighted; ECHO_FORM doubled the play): `player.energy` cs=0 rs=1.
//! Bead `sts2-rs-7ert6`.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::card_afflictions::{CardAffliction, CardAfflictionKind};
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid,
};

fn fresh() -> CombatState {
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
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// Push a Weighted(1) COMPACT (cost 1, Skill, Self) into hand at index 0.
fn push_weighted_compact(state: &mut CombatState) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = Card {
        uid,
        card_id: CardId::Compact,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 6,
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

fn add_power(state: &mut CombatState, id: u32, kind: PowerKind, amount: i32) {
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

/// Baseline (no replay): cost 1 + Weighted 1 = 2 drained, energy 3 -> 1.
#[test]
fn weighted_compact_no_replay_drains_once() {
    let mut state = fresh();
    push_weighted_compact(&mut state);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(
        state.player.energy, 1,
        "cost 1 + one Weighted drain = 2; 3 - 2 = 1",
    );
}

/// EchoForm(1) doubles the play, so Weighted drains TWICE: cost 1 + 1 + 1 = 3,
/// energy 3 -> 0. (Matches C# seed 423.)
#[test]
fn weighted_compact_echo_form_drains_per_play() {
    let mut state = fresh();
    add_power(&mut state, 7602, PowerKind::EchoForm, 1);
    push_weighted_compact(&mut state);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(
        state.player.energy, 0,
        "EchoForm replays the Weighted card: cost 1 + 2x Weighted = 3; 3 - 3 = 0",
    );
    // Block confirms the replay actually happened (6 x2 = 12).
    assert_eq!(state.player.block, 12, "EchoForm doubled COMPACT's 6 block");
}

/// OneTwoPunch(1) replays the next Attack once — the Weighted drain (handled in
/// the `extra_plays` replay loop) must fire on the replay too. COMPACT is a
/// Skill, so use a Weighted Strike (cost 1) here.
#[test]
fn weighted_attack_one_two_punch_drains_per_play() {
    let mut state = fresh();
    add_power(&mut state, 7601, PowerKind::OneTwoPunch, 1);
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

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(
        state.player.energy, 0,
        "OneTwoPunch replays the Weighted Attack: cost 1 + 2x Weighted = 3; 3 - 3 = 0",
    );
}
