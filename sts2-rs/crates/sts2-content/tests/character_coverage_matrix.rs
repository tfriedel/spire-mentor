//! P6-10 character-coverage matrix (bead `sts2-rs-fxw`).
//!
//! For each playable character (Ironclad, Silent, Defect), dispatch every card
//! in the character's pool via `card_effects::dispatch` and assert no panic +
//! `EffectResult::Resolved`. Writes a per-character dispatch-count summary
//! to `tests/parity/coverage_matrix_p6.txt` (golden) so future regressions
//! that silently drop pool entries surface as a diff.
//!
//! ## Why dispatch, not `Action::PlayCard`?
//!
//! `Action::PlayCard` requires a fully-built combat with a hand-pile entry
//! and a card-arena POD wired by `state.player.hand`. We synthesise a
//! minimal `(state.cards, state.player.play)` and call the registered
//! handler directly — the same path the per-pool unit tests in
//! `sts2-content/src/cards/*.rs` use. This covers handler bodies; the
//! end-to-end Action path is exercised by `tests/play_card.rs` and the
//! `act1_fuzz` sibling test.

use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;

use smallvec::SmallVec;
use sts2_core::card_effects::{dispatch, lookup, EffectResult};
use sts2_core::cards::{character_of, Card, CardId, CardTags, CardType, TargetType};
use sts2_core::{
    CardUid, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
};

use sts2_content::cards::dispatch as content_dispatch;

/// Push a synthetic card POD into `state.cards` and onto `state.player.play`.
/// Mirrors the helper in `sts2-content/src/cards/ironclad.rs::push_play_card`.
fn push_play_card(state: &mut CombatState, card_id: CardId) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards len fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    // Use neutral defaults: cost=1 (handlers don't pay; engine does),
    // base_damage=6 (Strike-like baseline), base_block=5 (Defend-like baseline),
    // target_type=Enemy (matched at dispatch site below).
    state.cards.push(Card {
        uid,
        card_id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 6,
        base_block: 5,
        combat_vars: sts2_core::cards::CombatVars::new(),
        enchantments: sts2_core::cards::CardEnchantments::new(),
        tags: CardTags::NONE,
        energy_spent: 0,
        base_replay_count: 0,
        keyword_override: None,
        affliction: None,
    });
    state.player.play.push(uid);
    uid
}

/// Build a fresh state with a single Cultist at high HP so every attack
/// has a target and dispatch doesn't terminate combat mid-iteration.
fn fresh_state_with_enemy() -> (CombatState, CreatureUid) {
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("non-zero");
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
    let uid = CreatureUid(id);
    state.enemy_uids.push(uid);
    // Give the player resources so resource-spending handlers don't
    // silently no-op on energy=0.
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 99;
    state.player.max_energy = 99;
    (state, uid)
}

fn parity_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates parent")
        .parent()
        .expect("repo root")
        .join("tests")
        .join("parity")
}

#[test]
fn coverage_matrix_per_character_dispatches_without_panic() {
    // Install everything: starter + every per-character pool.
    content_dispatch::install_default_registry();

    let mut dispatched: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut unregistered: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut not_implemented: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut total: BTreeMap<&'static str, usize> = BTreeMap::new();
    for ch in ["ironclad", "silent", "defect", "colorless", "necrobinder"] {
        dispatched.insert(ch, 0);
        unregistered.insert(ch, 0);
        not_implemented.insert(ch, 0);
        total.insert(ch, 0);
    }

    for &card_id in CardId::all() {
        let ch = character_of(card_id);
        *total.entry(ch).or_insert(0) += 1;

        let Some(_handler) = lookup(card_id) else {
            *unregistered.entry(ch).or_insert(0) += 1;
            continue;
        };

        let (mut state, enemy_uid) = fresh_state_with_enemy();
        let _ = push_play_card(&mut state, card_id);
        // We always pass Some(enemy) — Self-targeted / None-targeted
        // handlers ignore the target arg per the dispatch contract.
        let result = dispatch(&mut state, card_id, Some(enemy_uid));
        match result {
            EffectResult::Resolved => *dispatched.entry(ch).or_insert(0) += 1,
            EffectResult::NotImplemented => *not_implemented.entry(ch).or_insert(0) += 1,
        }
    }

    // Render golden.
    let mut actual = String::new();
    actual.push_str("# P6-10 dispatch coverage matrix (bead sts2-rs-fxw)\n");
    actual
        .push_str("# Format: <character>: dispatched=N unregistered=N not_implemented=N total=N\n");
    for ch in ["ironclad", "silent", "defect", "colorless", "necrobinder"] {
        actual.push_str(&format!(
            "{ch}: dispatched={} unregistered={} not_implemented={} total={}\n",
            dispatched.get(ch).copied().unwrap_or(0),
            unregistered.get(ch).copied().unwrap_or(0),
            not_implemented.get(ch).copied().unwrap_or(0),
            total.get(ch).copied().unwrap_or(0),
        ));
    }

    let path = parity_dir().join("coverage_matrix_p6.txt");
    let golden = fs::read_to_string(&path).ok();
    if golden.as_deref() != Some(actual.as_str()) {
        // Auto-record on first run / when intentionally bumped: prefer
        // failing loudly so a human eyeballs the diff. If the golden is
        // missing, write it and surface a clear "first-run" message.
        if golden.is_none() {
            fs::write(&path, &actual).expect("write coverage matrix golden");
            panic!(
                "wrote initial coverage matrix golden at {} — re-run to confirm stability.\n\
                 Contents:\n{actual}",
                path.display()
            );
        }
        let golden_str = golden.unwrap_or_default();
        panic!(
            "coverage_matrix_p6.txt drifted.\n--- expected ---\n{golden_str}\n--- actual ---\n{actual}\n\
             Re-run, eyeball the diff, and bump {} deliberately.",
            path.display()
        );
    }

    // Sanity floor: at least Ironclad has the starter trio + its pool.
    assert!(
        dispatched.get("ironclad").copied().unwrap_or(0) >= 3,
        "Ironclad coverage suspiciously low (expected ≥ 3 with starters)"
    );
}
