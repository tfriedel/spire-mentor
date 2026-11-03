//! `CreatureCmd.LoseMaxHp` below the current HP deals damage first (v0.111.0).

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, PowerKind,
};

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

/// An Ironclad with 80/80 HP and 5 energy in the play phase.
fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId) -> u8 {
    let uid = new_card(state, id);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

/// A full-HP player with Centennial Puzzle plays Brightest Flame (draw 2, lose max HP).
/// Returns the hand size.
fn brightest_flame_puzzle_hand(version: GameVersion) -> usize {
    let mut state = fresh(version);
    let _ = add_enemy(&mut state, 2, 100);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::CentennialPuzzle,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..10 {
        let uid = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(uid);
    }
    let i = push_hand(&mut state, CardId::BrightestFlame);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    state.player.hand.len()
}

/// `CreatureCmd.LoseMaxHp` (`CreatureCmd.cs:643-660`): when the new max HP is below the current
/// HP it first deals `Unblockable | Unpowered | Move` damage, so Centennial Puzzle's draw 3 on the
/// first HP loss of the combat fires: 2 (Brightest Flame) + 3 (Puzzle).
#[test]
fn lose_max_hp_below_current_triggers_after_damage_listeners_v0111() {
    assert_eq!(brightest_flame_puzzle_hand(GameVersion::V0_111_0), 5);
}

fn player_uid(state: &CombatState) -> CreatureUid {
    state.player.creature_uid.expect("player creature")
}

/// Full HP with 10 Block: the loss deals 5 unblockable damage, so HP and max HP both drop to 75
/// and the Block is untouched.
#[test]
fn lose_max_hp_below_current_hp_damages_through_block_v0111() {
    let mut state = fresh(GameVersion::V0_111_0);
    let _ = add_enemy(&mut state, 2, 100);
    state.player.block = 10;
    let uid = player_uid(&state);
    state.creature_lose_max_hp_from(uid, 5, true);
    assert_eq!(
        (
            state.player.current_hp,
            state.player.max_hp,
            state.player.block
        ),
        (75, 75, 10)
    );
}

/// A wounded player (60/80) losing 5 max HP stays above the new cap: no damage, HP unchanged.
#[test]
fn lose_max_hp_above_current_hp_deals_no_damage_v0111() {
    let mut state = fresh(GameVersion::V0_111_0);
    let _ = add_enemy(&mut state, 2, 100);
    let uid = player_uid(&state);
    // The player's HP lives on both the player and its mirror creature.
    state.player.current_hp = 60;
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
        c.current_hp = 60;
    }
    state.creature_lose_max_hp_from(uid, 5, true);
    assert_eq!((state.player.current_hp, state.player.max_hp), (60, 75));
}

/// `RupturePower.AfterDamageReceived` (`RupturePower.cs:49-51`) grants Strength at once for
/// damage with no card source, as LoseMaxHp's is (`CreatureCmd.cs:657`), even while a card is
/// still being played (before its `AfterCardPlayed`).
#[test]
fn lose_max_hp_damage_grants_rupture_strength_immediately_v0111() {
    let mut state = fresh(GameVersion::V0_111_0);
    let _ = add_enemy(&mut state, 2, 100);
    let uid = player_uid(&state);
    sts2_core::card_effects::apply_power_to_creature(&mut state, uid, PowerKind::Rupture, 1);
    let in_play = new_card(&mut state, CardId::BrightestFlame);
    state.player.play.push(in_play);
    state.creature_lose_max_hp_from(uid, 5, true);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 1);
}

/// The non-card entry point (`isFromCard = false`), which Paper Cuts uses, changes too: full HP
/// losing 1 max HP takes 1 damage first, so Centennial Puzzle draws.
#[test]
fn lose_max_hp_not_from_card_also_damages_first_v0111() {
    let mut state = fresh(GameVersion::V0_111_0);
    let _ = add_enemy(&mut state, 2, 100);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::CentennialPuzzle,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..10 {
        let uid = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(uid);
    }
    let uid = player_uid(&state);
    state.creature_lose_max_hp(uid, 1);
    assert_eq!((state.player.current_hp, state.player.max_hp), (79, 79));
    assert_eq!(state.player.hand.len(), 3);
}

/// v0.107.1 keeps the engine's old behaviour (the cap shrinks, no damage).
#[test]
fn lose_max_hp_plain_shrink_v0107_unchanged() {
    assert_eq!(brightest_flame_puzzle_hand(GameVersion::V0_107_1), 2);
}
