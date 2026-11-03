//! `BurningSticks.AfterCombatEnd` re-arms the once-per-combat latch (v0.111.0).

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind,
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

/// Exhausts one Forgotten Ritual (the content listener spends the latch and returns a clone to
/// hand), ends the combat, and returns `(latch counter, hand size)`.
fn burning_sticks_latch_after_combat(version: GameVersion) -> (i32, usize) {
    let mut state = fresh(version);
    let _ = add_enemy(&mut state, 2, 100);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::BurningSticks,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let a = push_hand(&mut state, CardId::ForgottenRitual);
    let _ = sts2_core::combat_loop::play_card(&mut state, a, None);
    let clone_in_hand = state.player.hand.len();
    let _ = sts2_core::relic_effects::dispatch_after_combat_end(&mut state);
    let counter = state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::BurningSticks)
        .map(|r| r.counter)
        .expect("relic");
    (counter, clone_in_hand)
}

/// `BurningSticks.AfterCombatEnd` (`BurningSticks.cs:56-60`) clears `WasUsedThisCombat`, so the
/// next combat's first exhausted Skill comes back again.
#[test]
fn burning_sticks_latch_is_rearmed_at_combat_end_v0111() {
    assert_eq!(
        burning_sticks_latch_after_combat(GameVersion::V0_111_0),
        (0, 1)
    );
}

/// v0.107.1 keeps the engine's old behaviour (the latch stays spent).
#[test]
fn burning_sticks_latch_stays_spent_v0107_unchanged() {
    assert_eq!(
        burning_sticks_latch_after_combat(GameVersion::V0_107_1),
        (1, 1)
    );
}
