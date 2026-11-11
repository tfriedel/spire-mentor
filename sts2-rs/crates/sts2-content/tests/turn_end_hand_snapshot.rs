//! C# `CombatManager.DoTurnEnd` sorts the hand into `turnEndCards` and the
//! Ethereal list in ONE pass before any exhaust (v0.107.1
//! `CombatManager.cs:1228-1248`, v0.111.0 `:1614-1631`). A turn-end-in-hand
//! card drawn by a listener of an Ethereal exhaust is therefore not in
//! `turnEndCards` and its effect never fires that turn.

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

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerEndTurnPhaseOne;
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

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn add_enemy(state: &mut CombatState, id: u32, hp: i32) {
    let nz = NonZeroU32::new(id).expect("nz");
    state.creatures.push(Creature {
        id: nz,
        current_hp: hp,
        max_hp: 100,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Guardbot),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Guardbot"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(nz));
}

/// Clumsy's Ethereal exhaust fires Forgotten Soul, whose 1 damage kills a
/// 1-HP Guardbot; Gremlin Horn draws a Debt. C# never runs that Debt's
/// `OnTurnEndInHand` (no gold lost) and discards it with the hand, after
/// Thinking Ahead. Random-loadout Ironclad v0.111.0 seed 6369.
#[test]
fn debt_drawn_during_the_ethereal_sweep_skips_its_turn_end_effect() {
    for v in BOTH {
        let mut state = fresh(v);
        state.gold = 50;
        // Both enemies at 1 HP: whichever Forgotten Soul hits dies, and the
        // other keeps the combat going.
        add_enemy(&mut state, 2, 1);
        add_enemy(&mut state, 3, 1);
        add_relic(&mut state, sts2_core::RelicKind::ForgottenSoul);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let debt = new_card(&mut state, CardId::Debt);
        state.player.draw.push(debt);
        let thinking = new_card(&mut state, CardId::ThinkingAhead);
        state.player.hand.push(thinking);
        let clumsy = new_card(&mut state, CardId::Clumsy);
        state.player.hand.push(clumsy);

        let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

        assert!(
            state.player.exhaust.contains(&clumsy),
            "{v:?}: Clumsy exhausted"
        );
        assert_eq!(state.gold, 50, "{v:?}: the drawn Debt took no gold");
        assert_eq!(
            state.player.discard.as_slice(),
            &[thinking, debt],
            "{v:?}: Debt is discarded with the hand, after Thinking Ahead"
        );
    }
}
