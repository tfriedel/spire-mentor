//! Order of the `Hook.AfterCardDrawn` listener walk.
//!
//! C# `CardPileCmd.Draw` records the draw in the combat history and then
//! awaits `Hook.AfterCardDrawn` for each card (CardPileCmd.cs). The hook walks
//! `CombatState.IterateHookListeners` (CombatState.cs): the player's powers in
//! the order they were applied, then the relics, potions and orbs, then every
//! card in the player's piles followed by its affliction and enchantment. Each
//! listener is awaited before the next starts, so a nested draw (Pagestorm,
//! Iteration) or a kill (Speedster, Cacophony) runs in full, Gremlin Horn's
//! energy and draw included, before a later listener sees the card. The card's
//! own hook (Void's energy loss) comes after every power.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::{CardId, VarKey};
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, PowerKind,
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
    state.player.energy = 0;
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

fn add_enemy(state: &mut CombatState, id: u32, hp: i32) -> CreatureUid {
    let nz = NonZeroU32::new(id).expect("nz");
    state.creatures.push(Creature {
        id: nz,
        current_hp: hp,
        max_hp: 100,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    let uid = CreatureUid(nz);
    state.enemy_uids.push(uid);
    uid
}

/// Apply player powers in the given order (application order = arena id).
fn apply_player_powers(state: &mut CombatState, powers: &[(PowerKind, i32)]) {
    let p = state.player.creature_uid.expect("player");
    for &(kind, amount) in powers {
        sts2_core::card_effects::apply_power_to_creature(state, p, kind, amount);
    }
}

fn set_counter(state: &mut CombatState, kind: PowerKind, counter: i32) {
    for pw in &mut state.powers {
        if pw.kind == kind {
            pw.counter = counter;
        }
    }
}

fn counter(state: &CombatState, kind: PowerKind) -> i32 {
    state
        .powers
        .iter()
        .find(|p| p.kind == kind)
        .map(|p| p.counter)
        .expect("power")
}

fn hp(state: &CombatState, id: u32) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id.get() == id)
        .expect("creature")
        .current_hp
}

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

/// A Void drawn at 0 energy with a nested-draw power (Iteration on the turn's
/// first Status, Pagestorm on an Ethereal card) and Automation one card short
/// of its energy. Either power draws the next card, which ticks Automation
/// before the Void's own Automation tick, so the Void's tick grants the energy
/// and only then does `Void.AfterCardDrawn` take it back (a card listener, it
/// follows every power). Rust lost the Void's energy first (clamped at 0) and
/// granted Automation's on the nested card afterwards, ending on 1.
#[test]
fn void_loses_its_energy_after_a_nested_draws_automation_gain() {
    for v in BOTH {
        for drawer in [PowerKind::Iteration, PowerKind::Pagestorm] {
            for drawer_first in [true, false] {
                let mut state = fresh(v);
                add_enemy(&mut state, 2, 50);
                let mut order = [(drawer, 1), (PowerKind::Automation, 1)];
                if !drawer_first {
                    order.reverse();
                }
                apply_player_powers(&mut state, &order);
                set_counter(&mut state, PowerKind::Automation, 8);
                let void = new_card(&mut state, CardId::Void);
                let defend = new_card(&mut state, CardId::DefendIronclad);
                state.player.draw.push(void);
                state.player.draw.push(defend);

                let _ = state.draw_into_hand_card_effect(1);

                let ctx = format!("{v:?} {drawer:?} first={drawer_first}");
                assert_eq!(state.player.hand.as_slice(), &[void, defend], "{ctx}");
                assert_eq!(counter(&state, PowerKind::Automation), 0, "{ctx}");
                assert_eq!(state.player.energy, 0, "{ctx}: Void took the gain back");
            }
        }
    }
}

/// Speedster's hit on a drawn Void kills a Nibbit; Gremlin Horn's
/// `AfterDeath` grants 1 energy and draws (GremlinHorn.cs), all awaited
/// inside Speedster's listener, before the Void's own `LoseEnergy(1)`. Rust
/// took the Void's energy first, at 0, and kept the Horn's.
#[test]
fn void_loses_its_energy_after_a_speedster_kills_gremlin_horn_energy() {
    for v in BOTH {
        let mut state = fresh(v);
        add_enemy(&mut state, 2, 2);
        add_enemy(&mut state, 3, 50);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        apply_player_powers(&mut state, &[(PowerKind::Speedster, 2)]);
        let void = new_card(&mut state, CardId::Void);
        let defend = new_card(&mut state, CardId::DefendIronclad);
        state.player.draw.push(void);
        state.player.draw.push(defend);

        let _ = state.draw_into_hand_card_effect(1);

        assert_eq!(hp(&state, 2), 0, "{v:?}: Speedster killed");
        assert_eq!(state.player.hand.as_slice(), &[void, defend], "{v:?}");
        assert_eq!(state.player.energy, 0, "{v:?}: Void took the Horn's energy");
    }
}

/// Speedster applied before the Queen's Chains of Binding: the card whose
/// Speedster hit kills a Nibbit sets off Gremlin Horn's nested draw, and that
/// card takes the turn's only `Bound` before Chains checks the card that
/// drew the kill. Rust checked Chains first, in its per-card pass, and dealt
/// the Speedster hits after it.
#[test]
fn speedster_kill_draw_takes_the_bound_before_its_trigger() {
    for v in BOTH {
        let mut state = fresh(v);
        add_enemy(&mut state, 2, 2);
        add_enemy(&mut state, 3, 50);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        apply_player_powers(&mut state, &[(PowerKind::Speedster, 2)]);
        // The Queen's Chains of Binding lands on the legacy player list,
        // applied later.
        let chains = NonZeroU32::new(50).expect("nz");
        state
            .powers
            .push(sts2_core::Power::new(chains, PowerKind::ChainsOfBinding, 1));
        state.player.powers.push(sts2_core::PowerUid(chains));
        let outer = new_card(&mut state, CardId::StrikeIronclad);
        let nested = new_card(&mut state, CardId::DefendIronclad);
        state.player.draw.push(outer);
        state.player.draw.push(nested);

        let _ = state.draw_into_hand_card_effect(1);

        assert_eq!(state.player.hand.as_slice(), &[outer, nested], "{v:?}");
        let bound =
            |uid: sts2_core::CardUid| state.cards[uid.get() as usize - 1].affliction.is_some();
        assert!(bound(nested), "{v:?}: the Gremlin Horn draw took the Bound");
        assert!(
            !bound(outer),
            "{v:?}: Chains reached the outer card at its cap"
        );
    }
}

/// Confused applied after Pagestorm: an Ethereal card first runs Pagestorm's
/// nested draw, whose card rolls its Confused cost, and only then rolls its
/// own. Rust rolled the Ethereal card first. Applied the other way round, the
/// Ethereal card rolls first.
#[test]
fn confused_after_pagestorm_rolls_the_nested_card_first() {
    for v in BOTH {
        for pagestorm_first in [true, false] {
            let mut state = fresh(v);
            add_enemy(&mut state, 2, 50);
            let mut order = [(PowerKind::Pagestorm, 1), (PowerKind::Confused, 1)];
            if !pagestorm_first {
                order.reverse();
            }
            apply_player_powers(&mut state, &order);
            // Find a stream position whose next two rolls differ.
            let (r1, r2) = loop {
                let mut rng = state.run_rng.combat_energy_costs.clone();
                let r1 = rng.next_int(0, 3);
                let r2 = rng.next_int(0, 3);
                if r1 != r2 {
                    break (r1, r2);
                }
                let _ = state.run_rng.combat_energy_costs.next_int(0, 3);
            };
            let ethereal = new_card(&mut state, CardId::DefendIronclad);
            state.cards[ethereal.get() as usize - 1].set_combat_var(VarKey::EtherealThisCombat, 1);
            let nested = new_card(&mut state, CardId::DefendIronclad);
            state.player.draw.push(ethereal);
            state.player.draw.push(nested);

            let _ = state.draw_into_hand_card_effect(1);

            let ctx = format!("{v:?} pagestorm_first={pagestorm_first}");
            assert_eq!(state.player.hand.as_slice(), &[ethereal, nested], "{ctx}");
            let cost = |uid: sts2_core::CardUid| state.cards[uid.get() as usize - 1].cost.local();
            let (outer_roll, nested_roll) = if pagestorm_first { (r2, r1) } else { (r1, r2) };
            assert_eq!(cost(ethereal), outer_roll, "{ctx}");
            assert_eq!(cost(nested), nested_roll, "{ctx}");
        }
    }
}

/// Iteration fires on a Status only while the turn's history holds at most
/// one drawn Status (IterationPower.cs: `num <= 1`, counted when Iteration's
/// listener runs). With Pagestorm applied first, the Ethereal Dazed's nested
/// draw brings a Slimed into the history before Iteration looks at the
/// Dazed, so Iteration draws nothing. Rust tested the Dazed before the
/// nested draw and drew an extra card.
#[test]
fn pagestorm_before_iteration_counts_the_nested_status() {
    for v in BOTH {
        for pagestorm_first in [true, false] {
            let mut state = fresh(v);
            add_enemy(&mut state, 2, 50);
            let mut order = [(PowerKind::Pagestorm, 1), (PowerKind::Iteration, 1)];
            if !pagestorm_first {
                order.reverse();
            }
            apply_player_powers(&mut state, &order);
            let dazed = new_card(&mut state, CardId::Dazed);
            let slimed = new_card(&mut state, CardId::Slimed);
            let d1 = new_card(&mut state, CardId::DefendIronclad);
            let d2 = new_card(&mut state, CardId::DefendIronclad);
            for uid in [dazed, slimed, d1, d2] {
                state.player.draw.push(uid);
            }

            let _ = state.draw_into_hand_card_effect(1);

            let ctx = format!("{v:?} pagestorm_first={pagestorm_first}");
            if pagestorm_first {
                assert_eq!(state.player.hand.as_slice(), &[dazed, slimed], "{ctx}");
            } else {
                // Iteration fires on the Dazed (one Status so far), then
                // Pagestorm draws for it.
                assert_eq!(state.player.hand.as_slice(), &[dazed, slimed, d1], "{ctx}");
            }
        }
    }
}

/// v0.111.0 Cacophony applied before Speedster picks its random target among
/// both Nibbits before Speedster's hit kills the 1-HP one. Rust dealt every
/// Speedster hit first and rolled Cacophony over the survivor only.
#[test]
fn cacophony_before_speedster_rolls_over_both_enemies() {
    for cacophony_first in [true, false] {
        let mut state = fresh(GameVersion::V0_111_0);
        add_enemy(&mut state, 2, 1);
        add_enemy(&mut state, 3, 100);
        let mut order = [(PowerKind::Cacophony, 10), (PowerKind::Speedster, 1)];
        if !cacophony_first {
            order.reverse();
        }
        apply_player_powers(&mut state, &order);
        set_counter(&mut state, PowerKind::Cacophony, 32);
        // A stream position whose two-enemy pick is the 1-HP Nibbit.
        while state
            .run_rng
            .combat_targets
            .clone()
            .next_int_exclusive(0, 2)
            != 0
        {
            let _ = state.run_rng.combat_targets.next_int_exclusive(0, 2);
        }
        let card = new_card(&mut state, CardId::DefendIronclad);
        state.player.draw.push(card);

        let _ = state.draw_into_hand_card_effect(1);

        assert_eq!(hp(&state, 2), 0, "cacophony_first={cacophony_first}");
        let expected = if cacophony_first { 99 } else { 89 };
        assert_eq!(hp(&state, 3), expected, "cacophony_first={cacophony_first}");
    }
}

/// Pagestorm applied before Automation: a Void's Pagestorm draw reshuffles
/// into Stratagem's pick. C# is still inside Pagestorm's listener, so the
/// Void's Automation tick and its energy loss wait for the pick and the rest
/// of that draw. Rust ran them before the pick.
#[test]
fn after_drawn_walk_resumes_after_a_nested_draws_stratagem_pick() {
    for v in BOTH {
        let mut state = fresh(v);
        add_enemy(&mut state, 2, 50);
        apply_player_powers(
            &mut state,
            &[
                (PowerKind::Stratagem, 1),
                (PowerKind::Pagestorm, 1),
                (PowerKind::Automation, 1),
            ],
        );
        set_counter(&mut state, PowerKind::Automation, 8);
        let void = new_card(&mut state, CardId::Void);
        state.player.draw.push(void);
        for id in [CardId::DefendIronclad, CardId::Bash, CardId::Anger] {
            let uid = new_card(&mut state, id);
            state.player.discard.push(uid);
        }

        let _ = state.draw_into_hand_card_effect(1);
        assert_eq!(state.player.hand.as_slice(), &[void], "{v:?}");
        let pc = state.pending_choice.as_ref().expect("the grid surfaced");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand, "{v:?}");
        assert_eq!(
            counter(&state, PowerKind::Automation),
            8,
            "{v:?}: the Void's walk waits inside Pagestorm"
        );

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{v:?}");
        assert!(state.pending_draw_resume.is_none(), "{v:?}");
        // Pick + Pagestorm's one card; the nested card ticks 8 -> 9, the Void
        // 9 -> 10 (+1 energy), and the Void then takes it back.
        assert_eq!(state.player.hand.len(), 3, "{v:?}");
        assert_eq!(counter(&state, PowerKind::Automation), 0, "{v:?}");
        assert_eq!(state.player.energy, 0, "{v:?}");
    }
}
