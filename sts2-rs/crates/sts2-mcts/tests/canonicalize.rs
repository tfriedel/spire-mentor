//! Action-canonicalization regression gates for bead `sts2-rs-nkf9`.
//!
//! `canonicalize_actions` is private to the `sts2-mcts` crate, so we
//! observe its effects through the public `Mcts::root_outcomes` /
//! untried-action-popping surface:
//!
//!   * **Duplicate collapse** — two hand copies of the exact same card
//!     (same `card_id`, `cost`, `upgraded`, `target`) produce the same
//!     simulator transition. Canonicalization should leave only one
//!     PlayCard root child for the duplicate-pair, not two.
//!   * **Play-priority first pop** — when the hand contains a clearly
//!     high-priority card (e.g. Apotheosis, priority -20) alongside
//!     lower-priority basics, the very first untried action expanded
//!     at the root must be the high-priority one. We exercise this by
//!     running a single simulation from the root and checking which
//!     PlayCard child was created first.

mod common;

use common::build_handful;
use sts2_core::cards::{CardId, TargetType};
use sts2_core::{legal_actions, Action};
use sts2_mcts::{Mcts, MctsConfig, RolloutPolicyKind};

/// Two adjacent identical Strikes in hand collapse to a single root
/// child after canonicalization. Without dedup MCTS would expand both
/// hand_index 0 and hand_index 1 as distinct PlayCards even though the
/// resulting state is bit-identical.
#[test]
fn duplicate_adjacent_cards_collapse_to_one_root_child() {
    // Two identical Strikes + a Defend. Cultist@30 keeps the fight
    // open so the search expands real children rather than ending
    // immediately on a kill.
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
    ];
    let state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);

    // Sanity: legal_actions reports both Strike copies as separate
    // entries (hand_index 0 and hand_index 1) — canonicalization is
    // an MCTS-side transform, not a core-side one.
    let raw = legal_actions(&state);
    let strike_legal: Vec<_> = raw
        .iter()
        .filter(|a| {
            matches!(
                a,
                Action::PlayCard {
                    hand_index: 0 | 1,
                    target: Some(0)
                }
            )
        })
        .collect();
    assert_eq!(
        strike_legal.len(),
        2,
        "core legal_actions should still expose both Strike copies pre-dedup; got {strike_legal:?}"
    );

    // Run enough sims to pop every untried action at the root at
    // least once, then count distinct Strike root-child entries.
    let cfg = MctsConfig {
        max_simulations: 200,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    let strike_children: Vec<_> = outs
        .iter()
        .filter(|o| {
            matches!(
                o.action,
                Action::PlayCard {
                    hand_index: 0 | 1,
                    target: Some(0),
                }
            )
        })
        .collect();
    assert_eq!(
        strike_children.len(),
        1,
        "canonicalize_actions must collapse adjacent duplicate Strikes; got {} root children: {strike_children:?}",
        strike_children.len()
    );
}

/// A8 audit §7b: two hand copies that differ ONLY in `combat_vars`
/// (here a Rampage-style scaling var on one Strike) map to the same
/// `EdgeAction` identity — `EdgeAction::resolve` ignores combat vars
/// and always picks the first matching hand card — so canonicalization
/// must collapse them. The old key carried `combat_vars.len()`, which
/// kept both alive as two root children resolving to the same concrete
/// action: byte-identical duplicate root actions, and child visits
/// summing past the simulation budget via the shared transposition
/// child.
#[test]
fn combat_vars_difference_does_not_duplicate_root_children() {
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
    ];
    let mut state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);
    // Second Strike (arena slot 1) gets a scaling var; identity fields
    // (card_id, upgraded, cost, free, target) stay equal.
    state.cards[1]
        .combat_vars
        .push((sts2_core::cards::VarKey::RampageDamage, 5));

    let sims = 200;
    let cfg = MctsConfig {
        max_simulations: sims,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    let strike_children = outs
        .iter()
        .filter(|o| {
            matches!(o.action, Action::PlayCard { .. }
            if !matches!(o.action, Action::PlayCard { hand_index: 2, .. }))
        })
        .count();
    assert_eq!(
        strike_children, 1,
        "copies differing only in combat_vars must collapse to one root child; got {outs:?}"
    );
    // No two root children may resolve to the same concrete action.
    for (i, a) in outs.iter().enumerate() {
        for b in &outs[..i] {
            assert_ne!(
                a.action, b.action,
                "byte-identical duplicate root actions: {outs:?}"
            );
        }
    }
    // Fresh tree, no reuse: each simulation backs up through at most
    // one root child, so the children's visits sum to <= sims run.
    let total: u32 = outs.iter().map(|o| o.visits).sum();
    assert!(
        total <= sims,
        "root child visits ({total}) exceed the simulation budget ({sims}): {outs:?}"
    );
}

/// The counter-direction to the dedup fix: a free-this-turn copy IS a
/// distinct play (costs no energy) and carries a distinct `EdgeAction`
/// (`free` field, commit `86a4c0f5c`), so canonicalization must KEEP
/// both copies as separate root children.
#[test]
fn free_copy_stays_a_distinct_root_child() {
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
    ];
    let mut state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);
    // Second Strike is a PowerPotion-style free copy: base cost untouched,
    // freeness carried by a free-this-turn cost modifier.
    state.cards[1].set_to_free_this_turn();

    let cfg = MctsConfig {
        max_simulations: 200,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    let strike_children = outs
        .iter()
        .filter(|o| matches!(o.action, Action::PlayCard { .. }))
        .count();
    assert_eq!(
        strike_children, 2,
        "free and non-free copies are distinct plays and must both survive; got {outs:?}"
    );
}

/// The first untried action expanded at the root is the highest-priority
/// PlayCard. We exercise this by running exactly one MCTS simulation
/// (one expansion happens during the simulation's expansion phase) on
/// a hand that mixes a high-priority card (Apotheosis, -20) with
/// low-priority basics (Strike, Defend at 55). The single root child
/// after one sim must be the Apotheosis play.
#[test]
fn play_priority_sort_pops_high_priority_first() {
    // Apotheosis (priority -20) is targetless / Self_; Strike is
    // Enemy-targeted; Defend is Self_. After canonicalization
    // Apotheosis comes first because -20 < 55.
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
        (CardId::Apotheosis, 0, 0, 2, TargetType::Self_),
    ];
    let state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);

    let cfg = MctsConfig {
        max_simulations: 1,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    // Exactly one root child should exist after a single simulation.
    assert_eq!(
        outs.len(),
        1,
        "expected exactly one root child after one simulation; got {} children: {outs:?}",
        outs.len(),
    );
    let first = &outs[0];
    assert!(
        matches!(first.action, Action::PlayCard { hand_index: 2, .. },),
        "first popped action must be the high-priority Apotheosis (hand_index 2); got {:?}",
        first.action,
    );
}

/// EndTurn is sorted to the back of the canonicalized list (priority
/// 10 000) so it is never the first popped child when a real card
/// play is legal — guards the rollout policy from prematurely ending
/// the turn during expansion-driven exploration.
#[test]
fn end_turn_pops_after_card_plays() {
    let hand = [(CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy)];
    let state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);

    let cfg = MctsConfig {
        max_simulations: 1,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    assert_eq!(
        outs.len(),
        1,
        "expected exactly one expansion after one sim"
    );
    assert!(
        matches!(outs[0].action, Action::PlayCard { hand_index: 0, .. }),
        "Strike must pop before EndTurn; got {:?}",
        outs[0].action
    );
}

/// Twin of `free_copy_stays_a_distinct_root_child`, for the `affliction`
/// field added to the `EdgeAction::PlayCard` identity tuple on 2026-08-16.
///
/// An afflicted and a clean copy of one card are NOT the same play:
/// `legal_actions` gates on the affliction kind (ChainsOfBinding blocks every
/// further `Bound` card once one has been played this turn; SmoggyPower does
/// the same for `Smog`). With them collapsed into one edge, `resolve` returned
/// the FIRST matching hand slot — which could be the copy `legal_actions` had
/// gated out, producing a root action that is illegal against the root state.
///
/// Observed before the fix (`sts2-runai::ppo_policy_run` at 24 sims, c_puct
/// 0.8): hand `[Strike(Bound), ShrugItOff(Bound), ShrugItOff, Strike]`, legal
/// only slots 2 and 3, edge resolved to slot 1. `Mcts::illegal_root_actions`
/// counts that class if it ever returns.
#[test]
fn afflicted_copy_stays_a_distinct_root_child() {
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
    ];
    let mut state = build_handful(0, &hand, /*cultist_hp=*/ 30, /*player_hp=*/ 50);
    // Second Strike carries a Bound affliction; the first is clean. Same
    // card_id / upgraded / cost / free — the tuple's other four fields cannot
    // tell them apart.
    state.cards[1].affliction = Some(sts2_core::card_afflictions::CardAffliction::new(
        sts2_core::card_afflictions::CardAfflictionKind::Bound,
        1,
    ));

    let cfg = MctsConfig {
        max_simulations: 200,
        seed: 0,
        rollout_policy: RolloutPolicyKind::Uniform,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    let outs = mcts.root_outcomes();

    let strike_children = outs
        .iter()
        .filter(|o| matches!(o.action, Action::PlayCard { .. }))
        .count();
    assert_eq!(
        strike_children, 2,
        "afflicted and clean copies are distinct plays and must both survive; got {outs:?}"
    );
    assert_eq!(
        mcts.illegal_root_actions(),
        0,
        "the root pick must be legal against the root state"
    );
}

/// GH #1017 follow-up: the co-op teammate play is its own `EdgeAction`
/// variant, so a single-player `PlayCard` edge keeps its exact `Debug` text.
/// The pinned tree fingerprints (`eval_levers`, `net_expand_order`) hash that
/// text through `children` and `untried_actions`; an `ally: None` field on
/// every edge broke all four pins without changing a single-player search.
#[test]
fn a_single_player_play_edge_prints_as_before() {
    let edge = sts2_mcts::EdgeAction::PlayCard {
        card_id: CardId::StrikeIronclad,
        upgraded: false,
        cost: sts2_core::CardEnergyCost::new(1),
        free: false,
        affliction: None,
        target: Some(0),
    };
    assert_eq!(
        format!("{edge:?}"),
        "PlayCard { card_id: StrikeIronclad, upgraded: false, cost: 1, free: false, \
         affliction: None, target: Some(0) }"
    );
}
