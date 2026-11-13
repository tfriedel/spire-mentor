//! v0.113.0 combat card reworks (port lane COMBAT, wave 5).
//!
//! Same harness as `cards_v0111.rs`: the real play path on a state whose
//! `game_version` is v0.113.0, cards built by `v0111::card_pod_for`.
//! Source of truth: the v0.113.0 decomp's `Models.Cards/` classes.
#![allow(dead_code)]

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for as card_pod_from_id_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V111: GameVersion = GameVersion::V0_111_0;
const V113: GameVersion = GameVersion::V0_113_0;

fn add_enemy(state: &mut CombatState, raw: u32, hp: i32) -> CreatureUid {
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: hp,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    CreatureUid(id)
}

/// One 100-HP Nibbit; player with 5 energy in the play phase.
fn fresh(version: GameVersion, character: CharacterId) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let _ = add_enemy(&mut state, 1, 100);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_from_id_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn push_draw(state: &mut CombatState, id: CardId) {
    let uid = new_card(state, id, false);
    state.player.draw.push(uid);
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn enemy_power(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("creature");
    c.powers
        .iter()
        .filter_map(|p| state.powers.get_by_id(p.0))
        .find(|p| p.kind == kind)
        .map_or(0, |p| p.amount)
}

fn add_enemy_power(
    state: &mut CombatState,
    uid: CreatureUid,
    raw: u32,
    kind: PowerKind,
    amount: i32,
) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state
        .creatures
        .iter_mut()
        .find(|c| c.id == uid.0)
        .expect("creature")
        .powers
        .push(PowerUid(pid));
}

fn first_enemy(state: &CombatState) -> CreatureUid {
    state.enemy_uids[0]
}

fn hp_of(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

#[test]
fn expect_a_fight_v0113() {
    let mut state = fresh(V113, CharacterId::Ironclad);
    add_player_power(&mut state, 900, PowerKind::Strength, 2);
    let atk = push_hand(&mut state, CardId::StrikeIronclad, false);
    let atk_uid = state.player.hand[usize::from(atk)];
    let i = push_hand(&mut state, CardId::ExpectAFight, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.block, 15, "flat BlockVar 15, no Strength term");
    let energy = state.player.energy;
    let j = state
        .player
        .hand
        .iter()
        .position(|u| *u == atk_uid)
        .expect("in hand");
    let _ = sts2_core::combat_loop::play_card(&mut state, u8::try_from(j).expect("u8"), Some(0));
    assert_eq!(
        state.player.energy, energy,
        "Attacks in hand cost 1 less until played"
    );
}

#[test]
fn expect_a_fight_upgrade_v0113() {
    let mut state = fresh(V113, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ExpectAFight, true);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.block, 20);
}

#[test]
fn forgotten_ritual_v0113_needs_an_exhaust() {
    let mut state = fresh(V113, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ForgottenRitual, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 4, "condition is back in v0.113");
}

#[test]
fn forgotten_ritual_v0111_unconditional() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ForgottenRitual, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 7);
}

#[test]
fn expertise_v0113() {
    let mut state = fresh(V113, CharacterId::Silent);
    for _ in 0..4 {
        push_draw(&mut state, CardId::StrikeSilent);
    }
    let i = push_hand(&mut state, CardId::Expertise, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.hand.len(), 2);
    assert_eq!(state.player_power_amount(PowerKind::RetainHand), 1);
}

#[test]
fn memento_mori_v0113_hits_lowest_hp_enemy() {
    let mut state = fresh(V113, CharacterId::Silent);
    let low = add_enemy(&mut state, 50, 40);
    let i = push_hand(&mut state, CardId::MementoMori, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 2, "costs 3");
    assert_eq!(hp_of(&state, low), 21, "19 damage to the lowest-HP enemy");
    assert_eq!(enemy_hp(&state), 100);
}

#[test]
fn salvo_and_dive_bomb_values_v0113() {
    for (id, base, up) in [(CardId::Salvo, 10, 14), (CardId::MinionDiveBomb, 11, 13)] {
        let c = card_pod_from_id_for(id, false, V113).expect("spec");
        assert_eq!(c.base_damage, base, "{id:?}");
        let c = card_pod_from_id_for(id, true, V113).expect("spec");
        assert_eq!(c.base_damage, up, "{id:?}+");
    }
}

/// v0.113 Memento Mori gains Sly (MementoMori.cs:15): a discard auto-plays it.
#[test]
fn memento_mori_is_sly_v0113_only() {
    use sts2_core::card_effects::card_sly_in;
    assert!(card_sly_in(CardId::MementoMori, GameVersion::V0_113_0));
    assert!(!card_sly_in(CardId::MementoMori, GameVersion::V0_111_0));
}
