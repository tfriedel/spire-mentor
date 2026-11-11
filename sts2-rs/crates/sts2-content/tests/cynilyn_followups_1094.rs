//! Follow-ups from the cynilyn v0.111 fork port (GH #1094). The C# for each is
//! the same in v0.107.1 and v0.111.0 unless a test says otherwise.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::{CardId, CardType, EnchantmentId, VarKey};
use sts2_core::potions::CharacterId;
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, PowerKind,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;
const BOTH: [GameVersion; 2] = [V107, V111];

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

/// Ironclad with 5 energy in the play phase and one 100-HP enemy.
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
    let _ = add_enemy(&mut state, 2, 100);
    state
}

fn new_card(state: &mut CombatState, id: CardId) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn card(state: &CombatState, uid: CardUid) -> &sts2_core::cards::Card {
    &state.cards[uid.0.get() as usize - 1]
}

// ── Imbued plays from every pile ──

/// `Imbued.AfterAutoPrePlayPhaseEntered` fires for each Imbued card the
/// `IterateHookListeners` snapshot holds: Hand, Draw, Discard, Exhaust, Play,
/// in that order (CombatState.cs). An Imbued card drawn into the opening hand
/// (a small deck) auto-plays too, before the one still in the draw pile.
#[test]
fn imbued_auto_plays_from_hand_and_discard_before_the_draw_pile() {
    for v in BOTH {
        let mut state = fresh(v);
        let in_draw = new_card(&mut state, CardId::DefendIronclad);
        let plain = new_card(&mut state, CardId::StrikeIronclad);
        let in_hand = new_card(&mut state, CardId::DefendIronclad);
        let in_discard = new_card(&mut state, CardId::DefendIronclad);
        for uid in [in_draw, in_hand, in_discard] {
            state.cards[uid.0.get() as usize - 1].set_enchantment(EnchantmentId::Imbued, 1);
        }
        state.player.draw.push(plain);
        state.player.draw.push(in_draw);
        state.player.hand.push(in_hand);
        state.player.discard.push(in_discard);
        state.round = 1;
        assert_eq!(state.turn_number(), 1);

        sts2_core::enchantment_hooks::dispatch_after_player_turn_start(&mut state);

        assert!(state.player.hand.is_empty(), "{v:?}: the hand copy played");
        assert_eq!(
            state.player.draw.as_slice(),
            &[plain],
            "{v:?}: only the Strike stays in draw"
        );
        // Each lands on the discard pile as it resolves: hand, draw, then
        // the copy that already sat in discard.
        assert_eq!(
            state.player.discard.as_slice(),
            &[in_hand, in_draw, in_discard],
            "{v:?}"
        );
        assert_eq!(state.player.block, 15, "{v:?}: three Defends");
    }
}

/// Past the first turn (`TurnNumber <= 1`) nothing plays, from any pile.
#[test]
fn imbued_in_hand_does_not_play_after_turn_one() {
    for v in BOTH {
        let mut state = fresh(v);
        let in_hand = new_card(&mut state, CardId::DefendIronclad);
        state.cards[0].set_enchantment(EnchantmentId::Imbued, 1);
        state.player.hand.push(in_hand);
        state.round = 2;
        sts2_core::enchantment_hooks::dispatch_after_player_turn_start(&mut state);
        assert_eq!(state.player.hand.as_slice(), &[in_hand], "{v:?}");
        assert_eq!(state.player.block, 0, "{v:?}");
    }
}

// ── An auto-played Attack shaves Stomp before its body ──

/// A Stomp auto-played as the turn's 3rd Attack under Juggling(1); returns
/// the cost of the Juggling copy in hand.
fn auto_played_juggled_stomp_copy_cost(v: GameVersion) -> i32 {
    let mut state = fresh(v);
    let p = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Juggling, 1);
    for pid in state.player_power_uids() {
        if let Some(pw) = state.powers.get_mut_by_id(pid.0) {
            if pw.kind == PowerKind::Juggling {
                pw.counter = 2;
            }
        }
    }
    state.player.cards_played_this_turn[CardType::Attack.index()] = 2;
    let stomp = new_card(&mut state, CardId::Stomp);
    // Two Attacks already shaved it 3 -> 1.
    state.cards[stomp.0.get() as usize - 1]
        .cost
        .add_this_turn(-2);
    state.player.draw.push(stomp);
    assert!(sts2_core::combat_loop::auto_play_card_from_draw_top(
        &mut state, false
    ));
    assert_eq!(state.player.hand.len(), 1, "{v:?}: the Juggling copy");
    card(&state, state.player.hand[0]).local_cost()
}

/// `Stomp.BeforeCardPlayed` shaves; the hook's listener list is snapshotted
/// before v0.111 `JugglingPower.cs:36` clones in the same hook, so the copy
/// keeps its cloned cost (1). The auto-play path used to shave in its
/// post-play tally, which also reached the copy (0). v0.107.1 clones after
/// the play, from the already-shaved card (0).
#[test]
fn auto_played_stomp_juggling_copy_keeps_its_cloned_cost() {
    assert_eq!(auto_played_juggled_stomp_copy_cost(V111), 1);
    assert_eq!(auto_played_juggled_stomp_copy_cost(V107), 0);
}

/// The auto-played Attack still shaves every Stomp once (not twice).
#[test]
fn auto_played_attack_shaves_stomp_once() {
    for v in BOTH {
        let mut state = fresh(v);
        let stomp = new_card(&mut state, CardId::Stomp);
        state.player.hand.push(stomp);
        let base = card(&state, stomp).local_cost();
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(strike);
        assert!(sts2_core::combat_loop::auto_play_card_from_draw_top(
            &mut state, false
        ));
        assert_eq!(card(&state, stomp).local_cost(), base - 1, "{v:?}");
        assert_eq!(
            card(&state, stomp).cost.mods(),
            &[sts2_core::LocalCostMod::relative(
                -1,
                sts2_core::CostExpiry::Turn
            )],
            "{v:?}"
        );
    }
}

// ── Clones are marked, so Stomp/Pinpoint skip their entry catch-up exactly ──

/// `Stomp.cs:53` / `Pinpoint.cs:43` skip the entered-combat catch-up for a
/// clone (`IsClone`). The engine used to guess "clone" from a non-zero
/// per-turn bank, so a clone of a card with an empty bank was shaved. Now
/// `create_clone` marks the copy and the check is exact.
#[test]
fn a_clone_skips_the_entered_combat_catch_up_and_a_fresh_card_does_not() {
    for v in BOTH {
        for (id, ty) in [
            (CardId::Stomp, CardType::Attack),
            (CardId::Pinpoint, CardType::Skill),
        ] {
            let mut state = fresh(v);
            state.player.cards_played_this_turn[ty.index()] = 2;
            let base = card_pod_for(id, false, v).expect("spec").local_cost();

            let source = card_pod_for(id, false, v).expect("spec");
            assert!(!source.cost.has_local_modifiers());
            let clone = source.create_clone();
            assert!(clone.is_clone());
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, clone);
            assert!(
                card(&state, uid).is_clone(),
                "{v:?} {id:?}: spawn keeps the mark"
            );
            assert_eq!(card(&state, uid).local_cost(), base, "{v:?} {id:?} clone");

            let fresh_card = card_pod_for(id, false, v).expect("spec");
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, fresh_card);
            assert!(!card(&state, uid).is_clone());
            assert_ne!(
                card(&state, uid).combat_var(VarKey::GeneratedInCombat),
                0,
                "{v:?} {id:?}: still marked generated"
            );
            assert_eq!(
                card(&state, uid).local_cost(),
                base - 2,
                "{v:?} {id:?} fresh"
            );
        }
    }
}

/// Dual Wield's copy is a C# `CreateClone()`, so it is marked as a clone.
#[test]
fn dual_wield_copy_is_a_clone() {
    for v in BOTH {
        let mut state = fresh(v);
        let stomp = new_card(&mut state, CardId::Stomp);
        state.player.hand.push(stomp);
        let dw = new_card(&mut state, CardId::DualWield);
        state.player.hand.push(dw);
        let _ = sts2_core::combat_loop::play_card(&mut state, 1, None);
        let copies: Vec<CardUid> = state
            .player
            .hand
            .iter()
            .copied()
            .filter(|&u| u != stomp && card(&state, u).card_id == CardId::Stomp)
            .collect();
        assert_eq!(copies.len(), 1, "{v:?}: one copy");
        assert!(card(&state, copies[0]).is_clone(), "{v:?}");
        assert!(!card(&state, stomp).is_clone(), "{v:?}: the source is not");
    }
}
