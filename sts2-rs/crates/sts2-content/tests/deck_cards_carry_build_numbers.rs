//! Invariant: on a non-default game build, every master-deck card already
//! carries that build's numbers (cost, damage, block, type, target,
//! keywords). Run-side code reads deck cards directly (deck view, enchant /
//! transform / removal eligibility), so a card left at the base build's
//! numbers is wrong there even though combat entry re-normalizes it
//! (`CombatState::add_card_to_draw_pile`). A new beta changes more cards, so
//! any deck-write path that skips the build's numbers turns into a bug for
//! them; this test catches such a path by playing random runs.

use sts2_content::run_callbacks::new_run_manager_for_character_from_string_seed_for;
use sts2_core::full_run::{drive_full_run, FullRunConfig, RandomPolicy, RunPolicy};
use sts2_core::{CharacterId, GameVersion, RunAction, RunManager};

struct DeckCheck {
    inner: RandomPolicy,
    bad: Vec<String>,
    changed: std::collections::BTreeSet<sts2_core::cards::CardId>,
}

impl RunPolicy for DeckCheck {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let v = rm.run_state.game_version;
        for c in &rm.run_state.master_deck {
            if sts2_content::cards::v0111::pod_changes_in(c.card_id, v) {
                self.changed.insert(c.card_id);
            }
            let mut n = *c;
            n.normalize_for_version(v);
            if n != *c && self.bad.len() < 20 {
                self.bad.push(format!(
                    "{:?} up={} cost {}→{} dmg {}→{} blk {}→{} type {:?}→{:?} (phase {:?})",
                    c.card_id,
                    c.upgraded,
                    c.local_cost(),
                    n.local_cost(),
                    c.base_damage,
                    n.base_damage,
                    c.base_block,
                    n.base_block,
                    c.card_type,
                    n.card_type,
                    rm.phase(),
                ));
            }
        }
        self.inner.choose(rm, actions)
    }

    fn prepare(&mut self, rm: &mut RunManager) {
        self.inner.prepare(rm);
    }
}

#[test]
fn v0111_deck_cards_carry_the_builds_numbers_through_random_runs() {
    let cfg = FullRunConfig {
        invincible: true,
        max_steps: 1_500,
        ..FullRunConfig::default()
    };
    let mut bad = Vec::new();
    let mut seen_changed = std::collections::BTreeSet::default();
    for character in [
        CharacterId::Ironclad,
        CharacterId::Silent,
        CharacterId::Defect,
        CharacterId::Regent,
        CharacterId::Necrobinder,
    ] {
        for seed in ["DECKVER001", "DECKVER002", "DECKVER003", "DECKVER004"] {
            let mut rm = new_run_manager_for_character_from_string_seed_for(
                GameVersion::V0_111_0,
                seed,
                character,
            );
            let mut policy = DeckCheck {
                inner: RandomPolicy::from_run_seed(rm.seed),
                bad: Vec::new(),
                changed: std::collections::BTreeSet::default(),
            };
            let _ = drive_full_run(&mut rm, &mut policy, &cfg);
            seen_changed.extend(policy.changed);
            bad.extend(
                policy
                    .bad
                    .into_iter()
                    .map(|b| format!("{character:?} {seed}: {b}")),
            );
        }
    }
    // Guard against the vacuous pass: the runs must put cards whose numbers
    // changed in v0.111.0 into decks (10 distinct ones at the time of writing).
    assert!(
        seen_changed.len() >= 5,
        "only {seen_changed:?} v0.111.0-changed cards entered a deck"
    );
    assert!(
        bad.is_empty(),
        "deck cards at base-build numbers:\n{}",
        bad.join("\n")
    );
}

/// Random runs rarely reach the relic and event grants that build a card
/// themselves, so obtain every relic on a fresh v0.111.0 run for each
/// character and check the deck right after.
#[test]
fn v0111_every_relic_grant_leaves_the_deck_at_the_builds_numbers() {
    // The relic listeners install lazily; without this every grant is a no-op
    // and the test passes vacuously.
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut bad = Vec::new();
    let mut granted = 0usize;
    for character in [
        CharacterId::Ironclad,
        CharacterId::Silent,
        CharacterId::Defect,
        CharacterId::Regent,
        CharacterId::Necrobinder,
    ] {
        let base = new_run_manager_for_character_from_string_seed_for(
            GameVersion::V0_111_0,
            "DECKRELIC",
            character,
        );
        for &relic in sts2_core::RelicId::all() {
            let mut rm = base.clone();
            rm.run_state.relics.push(relic);
            rm.run_state.last_obtained_relic = Some(relic);
            let before = rm.run_state.master_deck.len();
            let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rm.run_state);
            if rm.run_state.master_deck.len() > before {
                granted += 1;
            }
            let v = rm.run_state.game_version;
            for c in &rm.run_state.master_deck {
                let mut n = *c;
                n.normalize_for_version(v);
                if n != *c {
                    bad.push(format!(
                        "{character:?} {relic:?}: {:?} cost {}→{} dmg {}→{} blk {}→{}",
                        c.card_id,
                        c.local_cost(),
                        n.local_cost(),
                        c.base_damage,
                        n.base_damage,
                        c.base_block,
                        n.base_block
                    ));
                }
            }
        }
    }
    // Guard against the vacuous pass: several relics grant deck cards
    // (Tanx's Whistle, Pael's Horn, Storybook, Large Capsule, …).
    assert!(
        granted >= 20,
        "only {granted} relic grants added a card; listeners not installed?"
    );
    bad.sort();
    bad.dedup();
    assert!(
        bad.is_empty(),
        "relic grants left base-build deck cards:\n{}",
        bad.join("\n")
    );
}
