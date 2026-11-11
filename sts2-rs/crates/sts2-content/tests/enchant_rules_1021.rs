//! The smaller GH #1021 enchantment fixes, through the real engine paths.
//!
//! - C# `PerfectFit.ModifyShuffleOrder` moves each PerfectFit card to the top
//!   one listener at a time (draw pile, then discard pile, in pile order), so
//!   with two of them the one visited LAST is drawn first.
//! - C# `Entrench` grants `Unpowered | Move` block from its `cardPlay`:
//!   Dexterity does not apply, a Nimble enchantment does.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{CardId, EnchantmentId};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid,
};

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
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
    state.player.energy = 9;
    state.player.max_energy = 9;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn push_card(
    state: &mut CombatState,
    card_id: CardId,
    ench: Option<EnchantmentId>,
) -> sts2_core::CardUid {
    let uid =
        sts2_core::CardUid::new(u32::try_from(state.cards.len() + 1).expect("u32")).expect("nz");
    let mut card = sts2_content::cards::spec_index::card_pod_from_id(card_id, false).expect("card");
    card.uid = uid;
    card.cost = sts2_core::CardEnergyCost::new(0);
    if let Some(e) = ench {
        card.set_enchantment(e, 1);
    }
    state.cards.push(card);
    uid
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

/// Two PerfectFit cards (a Strike and a Bash) in the discard pile around four
/// Defends; returns (Strike uid, Bash uid, hand after drawing 2). `bash_first`
/// puts the Bash ahead of the Strike in the discard pile.
fn draw_two_perfect_fits(
    bash_first: bool,
) -> (
    sts2_core::CardUid,
    sts2_core::CardUid,
    Vec<sts2_core::CardUid>,
) {
    let mut state = fresh();
    let strike = push_card(
        &mut state,
        CardId::StrikeIronclad,
        Some(EnchantmentId::PerfectFit),
    );
    let bash = push_card(&mut state, CardId::Bash, Some(EnchantmentId::PerfectFit));
    let filler: Vec<_> = (0..4)
        .map(|_| push_card(&mut state, CardId::DefendIronclad, None))
        .collect();
    let (a, b) = if bash_first {
        (bash, strike)
    } else {
        (strike, bash)
    };
    state.player.discard.push(a);
    state.player.discard.extend(filler.iter().copied());
    state.player.discard.push(b);
    assert_eq!(state.draw_into_hand(2), 2);
    (strike, bash, state.player.hand.to_vec())
}

/// The reshuffle's `StableShuffle` sorts before shuffling, so the shuffled
/// order does not depend on the discard order — but C#'s PerfectFit order
/// does: the card visited last (later in the discard pile) is drawn first.
#[test]
fn the_last_perfect_fit_card_visited_is_drawn_first() {
    let (strike, bash, hand) = draw_two_perfect_fits(false);
    assert_eq!(hand, vec![bash, strike], "discard [Strike, .., Bash]");
    let (strike, bash, hand) = draw_two_perfect_fits(true);
    assert_eq!(hand, vec![strike, bash], "discard [Bash, .., Strike]");
}

#[test]
fn entrench_ignores_dexterity() {
    let mut state = fresh();
    add_power(&mut state, 7801, PowerKind::Dexterity, 3);
    state.player.block = 10;
    let uid = push_card(&mut state, CardId::Entrench, None);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(
        state.player.block, 20,
        "10 doubled; Unpowered block skips Dexterity"
    );
}

#[test]
fn a_nimble_entrench_adds_its_bonus() {
    let mut state = fresh();
    state.player.block = 10;
    let uid = push_card(&mut state, CardId::Entrench, None);
    state.cards[uid.0.get() as usize - 1].set_enchantment(EnchantmentId::Nimble, 2);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 22, "10 + (10 + Nimble 2)");
}

fn equip(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("u32")).expect("nz");
    state.relics.push(sts2_core::Relic {
        id,
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

/// C# `CardPileCmd.Shuffle` runs `Hook.ModifyShuffleOrder` before
/// `Hook.AfterShuffle`, so BiiigHug's Soot goes to a random depth of the
/// already-reordered pile and sometimes lands above the PerfectFit card.
/// With the hooks the other way round the PerfectFit card was always on top.
#[test]
fn a_biiig_hug_soot_can_land_above_a_perfect_fit_card() {
    let mut soot_on_top = 0;
    let mut pf_on_top = 0;
    for seed in 0..64 {
        let mut state = fresh();
        state.run_rng = sts2_core::run_rng_set::RunRngSet::new(seed);
        equip(&mut state, sts2_core::RelicKind::BiiigHug);
        let pf = push_card(&mut state, CardId::Bash, Some(EnchantmentId::PerfectFit));
        state.player.discard.push(pf);
        for _ in 0..3 {
            let d = push_card(&mut state, CardId::DefendIronclad, None);
            state.player.discard.push(d);
        }
        state.shuffle_draw_pile();
        assert_eq!(state.player.draw.len(), 5, "4 cards + 1 Soot");
        let top = state.player.draw[0];
        if top == pf {
            pf_on_top += 1;
        } else {
            let card = &state.cards[top.0.get() as usize - 1];
            assert_eq!(
                card.card_id,
                CardId::Soot,
                "seed {seed}: only Soot may outrank PerfectFit"
            );
            assert_eq!(
                state.player.draw[1], pf,
                "seed {seed}: PerfectFit right under the Soot"
            );
            soot_on_top += 1;
        }
    }
    assert!(soot_on_top > 0, "Soot never landed on top");
    assert!(pf_on_top > 0, "PerfectFit never on top");
}

/// Exhausting from an empty draw pile reshuffles through C#
/// `CardPileCmd.Shuffle`, which runs `ModifyShuffleOrder`: the PerfectFit
/// card is on top and is the one exhausted.
#[test]
fn exhaust_from_draw_reshuffle_puts_perfect_fit_on_top() {
    for seed in 0..16 {
        let mut state = fresh();
        state.run_rng = sts2_core::run_rng_set::RunRngSet::new(seed);
        let pf = push_card(&mut state, CardId::Bash, Some(EnchantmentId::PerfectFit));
        for _ in 0..4 {
            let d = push_card(&mut state, CardId::DefendIronclad, None);
            state.player.discard.push(d);
        }
        state.player.discard.push(pf);
        state.exhaust_from_draw_pile(1);
        assert_eq!(state.player.exhaust.as_slice(), &[pf], "seed {seed}");
    }
}
