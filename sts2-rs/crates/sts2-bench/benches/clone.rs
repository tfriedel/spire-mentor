//! Clone-cost benches for the MCTS fork path.
//!
//! `combatstate_clone_default` is the historical fixture — but note its card
//! arena is EMPTY, so it cannot see the cost that Card-as-Copy targets. The
//! `grown_arena` fixtures model a late-combat state: the arena is append-only
//! and never compacted (generated/transformed cards leave orphan slots), so a
//! real mid/late fight clones a 40–60 card `Vec<Card>` 2–3× per simulation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sts2_core::cards::{EnchantmentId, VarKey};
use sts2_core::{Card, CardUid, CombatState};

/// Late-combat card arena: `n` cards, every third carrying combat vars
/// (Rampage-style scaling + a free-this-turn flag) and every fifth an
/// enchantment stack — matching the shape the old SmallVec sized for.
fn grown_arena(n: u32) -> Vec<Card> {
    (1..=n)
        .map(|i| {
            let mut card = Card {
                uid: CardUid::new(i).expect("non-zero"),
                ..Card::pod_defaults()
            };
            if i % 3 == 0 {
                card.set_combat_var(VarKey::RampageDamage, i as i32);
                card.set_to_free_this_turn();
            }
            if i % 5 == 0 {
                card.set_enchantment(EnchantmentId::Steady, 1);
            }
            card
        })
        .collect()
}

fn bench_clone(c: &mut Criterion) {
    c.bench_function("combatstate_clone_default", |b| {
        let state = CombatState::default();
        b.iter(|| black_box(state.clone()));
    });

    c.bench_function("combatstate_clone_grown_arena_48", |b| {
        let state = CombatState {
            cards: grown_arena(48).into(),
            ..Default::default()
        };
        b.iter(|| black_box(state.clone()));
    });

    c.bench_function("card_arena_clone_48", |b| {
        let arena = grown_arena(48);
        b.iter(|| black_box(arena.clone()));
    });
}

criterion_group!(benches, bench_clone);
criterion_main!(benches);
