//! C# `CardCmd.Transform` returns nothing as its first statement once
//! `CombatManager.Instance.IsEnding` (v0.107.1 `CardCmd.cs:371-374`, v0.111.0
//! `:373-376`), before `GetReplacement` rolls `CombatCardSelection`.
//! `TransformToRandom`, `TransformTo<T>` and `Transform(original,
//! replacement)` all funnel into it.

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

/// A Defect combat whose only enemy is dead: the combat is ending.
fn ending(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 0,
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
    state.player.character = Some(CharacterId::Defect);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    assert!(sts2_core::combat_loop::combat_is_ending(&state));
    state
}

fn hand_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    uid
}

/// Entropy's second transform after the first one's Trash to Treasure
/// channel evoked a Dark orb that killed the last enemy: no replacement and
/// no `CombatCardSelection` draw. Random-loadout Defect v0.107.1 seed 6416.
#[test]
fn transform_to_random_is_refused_once_the_combat_is_ending() {
    for v in BOTH {
        let mut state = ending(v);
        let wound = hand_card(&mut state, CardId::Wound);
        let before = state.run_rng.combat_card_selection.counter();

        let out = sts2_core::card_effects::transform_card_to_random(&mut state, wound);

        assert_eq!(out, None, "{v:?}");
        assert_eq!(
            state.run_rng.combat_card_selection.counter(),
            before,
            "{v:?}"
        );
        assert_eq!(state.player.hand.as_slice(), &[wound], "{v:?}");
    }
}

/// The same guard on a transform into a given card (`TransformTo<T>`).
#[test]
fn transform_to_a_given_card_is_refused_once_the_combat_is_ending() {
    for v in BOTH {
        let mut state = ending(v);
        let strike = hand_card(&mut state, CardId::StrikeDefect);
        let fuel = card_pod_for(CardId::Fuel, false, v).expect("spec");

        assert_eq!(state.transform_card(strike, fuel), None, "{v:?}");
        assert_eq!(
            sts2_core::card_effects::transform_card_in_place(&mut state, strike, fuel),
            None,
            "{v:?}"
        );
        assert_eq!(state.player.hand.as_slice(), &[strike], "{v:?}");
    }
}
