//! The card-arena presence gates (J02), held OPEN.
//!
//! The per-turn and per-play card sweeps skip their arena walk when a summary
//! says it has nothing to do: the arena's gate-id bits (`CardStore::holds_id`
//! for Stomp, Pinpoint, Melancholy, Bolas, ThrummingHatchet) and each card's
//! "may be nonzero" mask over its `combat_vars`. A combat that never holds those cards
//! only ever takes the closed side, so these tests put the cards in the arena,
//! outside the hand where it matters, and check that each sweep still acts
//! through the engine:
//!
//! * Stomp / Pinpoint discounts and their end-of-turn expiry;
//! * Melancholy's on-death discount (combat piles only);
//! * the Bolas / ThrummingHatchet return to hand from any non-hand pile;
//! * Snecko Oil's randomised costs reverting at the next turn start;
//! * a free-this-turn modifier dropped at end of turn.
//!
//! The last test plays random combats whose decks hold all of them. In debug
//! builds every skipped sweep also runs its ungated walk on a clone and
//! asserts it changed nothing, the arena bits are re-derived at every `apply`
//! entry, and every mask test is checked against a scan, so the random
//! combats exercise both sides of every gate.

use sts2_core::cards::{CardId, VarKey};
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::{apply, legal_actions, Action, CardUid, CombatPhase, CombatState, MonsterKind};
use sts2_core::{CostExpiry, LocalCostMod};

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
}

/// A player with 70 HP facing one Nibbit per entry of `enemy_hp`.
fn bare_state(character: CharacterId, seed: i32, enemy_hp: &[i32]) -> CombatState {
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(seed),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(seed),
        ..Default::default()
    };
    state.player.character = Some(character);
    state.player.current_hp = 70;
    state.player.max_hp = 70;
    state.player.max_energy = 3;
    state.init_player_creature();
    for &hp in enemy_hp {
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.max_hp = hp;
            c.current_hp = hp;
        }
    }
    state
}

/// Add a fresh `card_id` to the arena (not to any pile) and return its uid.
fn add_card(state: &mut CombatState, card_id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

#[derive(Clone, Copy)]
enum Pile {
    Draw,
    Discard,
    Exhaust,
    Nowhere,
}

/// A combat started on a draw pile of `filler`, then rearranged mid-turn 1:
/// `hand` is the whole hand, each `placed` card sits in its pile, energy 3.
/// Returns the placed cards' uids.
fn ready(
    character: CharacterId,
    enemy_hp: &[i32],
    filler: CardId,
    hand: &[CardId],
    placed: &[(CardId, Pile)],
) -> (CombatState, Vec<CardUid>) {
    let mut state = bare_state(character, 4242, enemy_hp);
    for _ in 0..10 {
        let uid = add_card(&mut state, filler);
        state.player.draw.push(uid);
    }
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    let drawn: Vec<CardUid> = state.player.hand.drain(..).collect();
    state.player.draw.extend(drawn);
    for &id in hand {
        let uid = add_card(&mut state, id);
        state.player.hand.push(uid);
    }
    let placed = placed
        .iter()
        .map(|&(id, pile)| {
            let uid = add_card(&mut state, id);
            match pile {
                Pile::Draw => state.player.draw.push(uid),
                Pile::Discard => state.player.discard.push(uid),
                Pile::Exhaust => state.player.exhaust.push(uid),
                Pile::Nowhere => {}
            }
            uid
        })
        .collect();
    state.player.energy = 3;
    (state, placed)
}

fn card(state: &CombatState, uid: CardUid) -> &sts2_core::Card {
    &state.cards[uid.get() as usize - 1]
}

fn play_first(state: &mut CombatState, target: Option<u8>) {
    let _ = apply(
        state,
        Action::PlayCard {
            hand_index: 0,
            target,
        },
    );
}

/// End turn 1 and come back to the player's turn 2.
fn end_turn(state: &mut CombatState) {
    let _ = apply(state, Action::EndTurn);
    assert_eq!(state.round, 2, "the player survives to turn 2");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
}

/// Stomp in the discard pile: each Attack shaves 1 off it, and
/// the turn end drops the shave.
#[test]
fn stomp_outside_the_hand_is_discounted_per_attack_and_restored() {
    install();
    let (mut state, placed) = ready(
        CharacterId::Ironclad,
        &[300],
        CardId::DefendIronclad,
        &[CardId::StrikeIronclad, CardId::StrikeIronclad],
        &[(CardId::Stomp, Pile::Discard)],
    );
    let stomp = placed[0];
    let base = card(&state, stomp).local_cost();
    assert!(base >= 2);
    for shaved in 1..=2 {
        play_first(&mut state, Some(0));
        assert_eq!(card(&state, stomp).local_cost(), base - shaved);
    }
    assert_eq!(
        card(&state, stomp).cost.mods(),
        &[LocalCostMod::relative(-2, CostExpiry::Turn)]
    );
    end_turn(&mut state);
    assert_eq!(card(&state, stomp).local_cost(), base);
    assert!(!card(&state, stomp).cost.has_local_modifiers());
}

/// Pinpoint in the draw pile: each Skill shaves 1 off it, dropped at the
/// turn end.
#[test]
fn pinpoint_outside_the_hand_is_discounted_per_skill_and_restored() {
    install();
    let (mut state, placed) = ready(
        CharacterId::Silent,
        &[300],
        CardId::StrikeSilent,
        &[CardId::DefendSilent, CardId::DefendSilent],
        &[(CardId::Pinpoint, Pile::Draw)],
    );
    let pinpoint = placed[0];
    let base = card(&state, pinpoint).local_cost();
    assert!(base >= 2);
    for shaved in 1..=2 {
        play_first(&mut state, None);
        assert_eq!(card(&state, pinpoint).local_cost(), base - shaved);
    }
    end_turn(&mut state);
    assert_eq!(card(&state, pinpoint).local_cost(), base);
    assert!(!card(&state, pinpoint).cost.has_local_modifiers());
}

/// A creature death shaves 1 off every Melancholy in a combat pile (here the
/// exhaust pile), but not off one that is in the arena and in no pile.
#[test]
fn melancholy_in_a_combat_pile_is_discounted_on_death() {
    install();
    let (mut state, placed) = ready(
        CharacterId::Necrobinder,
        &[1, 300],
        CardId::DefendNecrobinder,
        &[CardId::StrikeNecrobinder],
        &[
            (CardId::Melancholy, Pile::Exhaust),
            (CardId::Melancholy, Pile::Nowhere),
        ],
    );
    let (in_pile, in_no_pile) = (placed[0], placed[1]);
    let base = card(&state, in_pile).local_cost();
    play_first(&mut state, Some(0));
    assert!(!state.is_over, "the 300-HP Nibbit is still up");
    assert_eq!(
        card(&state, in_pile).local_cost(),
        base - 1,
        "the 1-HP Nibbit died"
    );
    assert_eq!(card(&state, in_no_pile).local_cost(), base);
}

/// Bolas and ThrummingHatchet played on turn 1 come back to hand at the start
/// of turn 2, from the discard pile and from the exhaust pile alike.
#[test]
fn bolas_and_thrumming_hatchet_return_from_any_non_hand_pile() {
    install();
    let (mut state, _) = ready(
        CharacterId::Ironclad,
        &[300],
        CardId::DefendIronclad,
        &[CardId::Bolas, CardId::ThrummingHatchet],
        &[],
    );
    let bolas = state.player.hand[0];
    let hatchet = state.player.hand[1];
    play_first(&mut state, Some(0));
    play_first(&mut state, Some(0));
    assert!(state.player.discard.contains(&bolas));
    assert!(state.player.discard.contains(&hatchet));
    // Move Bolas on to the exhaust pile: it must still come back.
    state.player.discard.retain(|u| *u != bolas);
    state.player.exhaust.push(bolas);
    end_turn(&mut state);
    assert!(state.player.hand.contains(&bolas), "Bolas returned");
    assert!(state.player.hand.contains(&hatchet), "Hatchet returned");
}

/// Snecko Oil randomises the hand's costs with `SetThisTurnOrUntilPlayed`;
/// the turn end drops every roll. Seeds are scanned so that some rolls
/// actually changed a cost and the revert has something to undo.
#[test]
fn snecko_randomised_costs_revert_at_the_next_turn_start() {
    install();
    let mut rolled = 0;
    for seed in 1..=20 {
        let mut state = bare_state(CharacterId::Ironclad, seed, &[300]);
        for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
            for _ in 0..4 {
                let uid = add_card(&mut state, id);
                state.player.draw.push(uid);
            }
        }
        state.player.potions.push(PotionKind::SneckoOil);
        state.player.potion_slots = 3;
        let _ = start_combat(&mut state);
        let _ = start_player_turn(&mut state);
        let base: Vec<i32> = state
            .cards
            .iter()
            .map(sts2_core::Card::local_cost)
            .collect();
        let _ = apply(
            &mut state,
            Action::DrinkPotion {
                potion_slot: 0,
                target: None,
            },
        );
        rolled += state
            .cards
            .iter()
            .zip(&base)
            .filter(|(c, &b)| c.local_cost() != b)
            .count();
        end_turn(&mut state);
        for (c, &cost) in state.cards.iter().zip(&base) {
            assert_eq!(
                c.local_cost(),
                cost,
                "seed {seed}: {:?} kept its rolled cost",
                c.uid
            );
            assert!(!c.cost.has_local_modifiers(), "seed {seed}: {:?}", c.uid);
        }
    }
    assert!(rolled > 0, "no Snecko roll changed a cost");
}

/// Infernal Blade's spawn is free this turn; end of turn clears the flag.
#[test]
fn free_this_turn_is_cleared_at_end_of_turn() {
    install();
    let (mut state, _) = ready(
        CharacterId::Ironclad,
        &[300],
        CardId::DefendIronclad,
        &[CardId::InfernalBlade],
        &[],
    );
    play_first(&mut state, None);
    let spawned = *state
        .player
        .hand
        .last()
        .expect("Infernal Blade spawned a card");
    assert!(card(&state, spawned).cost.has(LocalCostMod::FREE_THIS_TURN));
    assert_eq!(card(&state, spawned).local_cost(), 0);
    end_turn(&mut state);
    assert!(!card(&state, spawned).cost.has(LocalCostMod::FREE_THIS_TURN));
}

const IRONCLAD_DECK: &[CardId] = &[
    CardId::Stomp,
    CardId::Stomp,
    CardId::Bolas,
    CardId::ThrummingHatchet,
    CardId::InfernalBlade,
    CardId::Melancholy,
    CardId::StrikeIronclad,
    CardId::StrikeIronclad,
    CardId::StrikeIronclad,
    CardId::DefendIronclad,
    CardId::DefendIronclad,
    CardId::Bash,
];

const SILENT_DECK: &[CardId] = &[
    CardId::Pinpoint,
    CardId::Pinpoint,
    CardId::Neutralize,
    CardId::StrikeSilent,
    CardId::StrikeSilent,
    CardId::StrikeSilent,
    CardId::DefendSilent,
    CardId::DefendSilent,
    CardId::DefendSilent,
    CardId::DefendSilent,
];

const NECROBINDER_DECK: &[CardId] = &[
    CardId::Melancholy,
    CardId::Melancholy,
    CardId::StrikeNecrobinder,
    CardId::StrikeNecrobinder,
    CardId::StrikeNecrobinder,
    CardId::StrikeNecrobinder,
    CardId::DefendNecrobinder,
    CardId::DefendNecrobinder,
    CardId::DefendNecrobinder,
    CardId::DefendNecrobinder,
];

/// Count, over one random combat, the states in which each gate was open:
/// a Stomp shave, a Pinpoint shave, a live Snecko roll, a free-this-turn
/// card, a Bolas/Hatchet return stamp, and a Melancholy below its starting
/// cost.
fn random_combat(character: CharacterId, deck: &[CardId], seed: u64, open: &mut [u32; 6]) {
    let mut state = bare_state(character, i32::try_from(seed).expect("small"), &[30, 30]);
    for &id in deck {
        let uid = add_card(&mut state, id);
        state.player.draw.push(uid);
    }
    state.player.potions.push(PotionKind::SneckoOil);
    state.player.potions.push(PotionKind::SneckoOil);
    state.player.potion_slots = 3;
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    let melancholy_base: Vec<(usize, i32)> = state
        .cards
        .iter()
        .enumerate()
        .filter(|(_, c)| c.card_id == CardId::Melancholy)
        .map(|(i, c)| (i, c.local_cost()))
        .collect();
    let mut agent = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for _ in 0..300 {
        if state.is_over {
            break;
        }
        let actions = legal_actions(&state);
        if actions.is_empty() {
            break;
        }
        agent ^= agent << 13;
        agent ^= agent >> 7;
        agent ^= agent << 17;
        let action = actions[(agent % actions.len() as u64) as usize].clone();
        let _ = apply(&mut state, action);
        let shaved = |id: CardId| {
            state.cards.iter().any(|c| {
                c.card_id == id
                    && c.cost
                        .mods()
                        .iter()
                        .any(|m| !m.is_absolute() && m.expiry() == CostExpiry::Turn)
            })
        };
        let gates = [
            shaved(CardId::Stomp),
            shaved(CardId::Pinpoint),
            state.cards.iter().any(|c| {
                c.cost.mods().iter().any(|m| {
                    m.is_absolute() && m.amount() != 0 && m.expiry() == CostExpiry::TurnOrPlayed
                })
            }),
            state
                .cards
                .iter()
                .any(|c| c.cost.has(LocalCostMod::FREE_THIS_TURN)),
            state
                .cards
                .iter()
                .any(|c| c.combat_var(VarKey::ReturnIfPlayedRound) != 0),
        ];
        for (i, open_now) in gates.into_iter().enumerate() {
            if open_now {
                open[i] += 1;
            }
        }
        if melancholy_base
            .iter()
            .any(|&(i, cost)| state.cards[i].local_cost() < cost)
        {
            open[5] += 1;
        }
    }
}

/// Random combats whose decks hold every gate card (and two Snecko Oils) run
/// clean under the debug shadows, and every gate really opens in them.
#[test]
fn random_combats_with_every_gate_card_run_clean_under_the_shadows() {
    install();
    let mut open = [0u32; 6];
    for seed in 1..=4 {
        random_combat(CharacterId::Ironclad, IRONCLAD_DECK, seed, &mut open);
        random_combat(CharacterId::Silent, SILENT_DECK, seed, &mut open);
        random_combat(CharacterId::Necrobinder, NECROBINDER_DECK, seed, &mut open);
    }
    let names = [
        "Stomp shave",
        "Pinpoint shave",
        "Snecko roll",
        "free this turn",
        "Bolas/Hatchet stamp",
        "Melancholy discount",
    ];
    for (name, n) in names.iter().zip(open) {
        assert!(n > 0, "the {name} gate never opened: {open:?}");
    }
}
