//! GH #1623: a card select requested after the combat already ended.
//!
//! C# `CardSelectCmd.FromHandForUpgrade` / `FromHand` return nothing once
//! `CombatManager.IsOverOrEnding`, and `FromCombatPile` once `IsEnding`
//! (v0.111.0 `CardSelectCmd.cs:915` / `:824` / `:452`; v0.113.1 `:753` /
//! `:660` / `:274`). The guard comes before the lone-candidate shortcut, so
//! nothing is picked at all. Each test lets Juggernaut turn the card's own
//! block into the killing blow on a 1-HP enemy, then checks that no select
//! opened and the hand / piles are untouched.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const VERSIONS: [GameVersion; 3] = [
    GameVersion::V0_111_0,
    GameVersion::V0_113_0,
    GameVersion::V0_114_0,
];

/// One 1-HP Nibbit, a player with Juggernaut 5 and 5 energy.
fn fresh(version: GameVersion, character: CharacterId) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 1,
        max_hp: 1,
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
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    let pid = NonZeroU32::new(50).expect("nz");
    state.powers.push(Power::new(pid, PowerKind::Juggernaut, 5));
    state.player.powers.push(PowerUid(pid));
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn play(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::combat_loop::play_card(state, idx, None);
    assert!(state.creatures[0].current_hp <= 0, "Juggernaut kills");
    assert!(
        state.pending_choice.is_none(),
        "no select once the combat is ending: {:?}",
        state.pending_choice
    );
}

fn upgraded(state: &CombatState, uid: sts2_core::CardUid) -> bool {
    state.cards[uid.0.get() as usize - 1].upgraded
}

/// Agent-parity seed AP1008: base Armaments with two upgradable hand cards.
#[test]
fn armaments_opens_no_upgrade_select_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Ironclad);
        let a = new_card(&mut state, CardId::DefendIronclad, false);
        let b = new_card(&mut state, CardId::TwinStrike, false);
        state.player.hand.extend([a, b]);
        let i = push_hand(&mut state, CardId::Armaments, false);
        play(&mut state, i);
        assert!(!upgraded(&state, a) && !upgraded(&state, b), "{version:?}");
    }
}

/// `FromHandForUpgrade`'s `Count <= 1` auto-pick sits behind the same guard.
#[test]
fn armaments_auto_upgrades_nothing_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Ironclad);
        let a = new_card(&mut state, CardId::DefendIronclad, false);
        state.player.hand.push(a);
        let i = push_hand(&mut state, CardId::Armaments, false);
        play(&mut state, i);
        assert!(!upgraded(&state, a), "{version:?}");
    }
}

/// True Grit+ asks `FromHand` for its exhaust after the block.
#[test]
fn true_grit_plus_opens_no_exhaust_select_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Ironclad);
        let a = new_card(&mut state, CardId::DefendIronclad, false);
        let b = new_card(&mut state, CardId::StrikeIronclad, false);
        state.player.hand.extend([a, b]);
        let i = push_hand(&mut state, CardId::TrueGrit, true);
        play(&mut state, i);
        assert!(state.player.hand.contains(&a) && state.player.hand.contains(&b));
        assert!(state.player.exhaust.is_empty(), "{version:?}");
    }
}

/// Hand Trick's lone Skill is not made Sly.
#[test]
fn hand_trick_makes_nothing_sly_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Silent);
        let a = new_card(&mut state, CardId::DefendSilent, false);
        state.player.hand.push(a);
        let i = push_hand(&mut state, CardId::HandTrick, false);
        play(&mut state, i);
        assert!(state.player.sly_this_turn.is_empty(), "{version:?}");
    }
}

/// Hologram's lone discard card stays in the discard pile.
#[test]
fn hologram_returns_nothing_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Defect);
        let a = new_card(&mut state, CardId::StrikeDefect, false);
        state.player.discard.push(a);
        let i = push_hand(&mut state, CardId::Hologram, false);
        play(&mut state, i);
        assert!(state.player.discard.contains(&a), "{version:?}");
        assert!(!state.player.hand.contains(&a), "{version:?}");
    }
}

/// Cosmic Indifference's lone discard card is not put on the draw pile.
#[test]
fn cosmic_indifference_moves_nothing_after_the_kill() {
    for version in VERSIONS {
        let mut state = fresh(version, CharacterId::Regent);
        let a = new_card(&mut state, CardId::StrikeRegent, false);
        state.player.discard.push(a);
        let i = push_hand(&mut state, CardId::CosmicIndifference, false);
        play(&mut state, i);
        assert!(state.player.discard.contains(&a), "{version:?}");
        assert!(!state.player.draw.contains(&a), "{version:?}");
    }
}

/// `open_hand_choice` (Brand, Burning Pact, True Grit+, Scavenge, Toasty
/// Mittens) opens nothing once the combat is ending.
#[test]
fn open_hand_choice_opens_nothing_once_the_combat_is_ending() {
    let mut state = fresh(GameVersion::V0_111_0, CharacterId::Ironclad);
    let a = new_card(&mut state, CardId::DefendIronclad, false);
    let b = new_card(&mut state, CardId::StrikeIronclad, false);
    state.player.hand.extend([a, b]);
    state.creatures[0].current_hp = 0;
    let option = |uid| sts2_core::ChoiceOption {
        card_id: CardId::DefendIronclad,
        upgraded: false,
        card_uid: Some(uid),
        source_pile: Some(sts2_core::cards::PileKind::Hand),
    };
    state.open_hand_choice(sts2_core::PendingChoice {
        kind: sts2_core::ChoiceKind::ExhaustFromHand,
        options: vec![option(a), option(b)],
        min_select: 1,
        max_select: 1,
        can_skip: false,
        post_effect: Some(sts2_core::PostChoiceEffect::DrawCards(2)),
        picked: SmallVec::new(),
    });
    assert!(state.pending_choice.is_none());
    assert_eq!(state.player.hand.len(), 2);
}
