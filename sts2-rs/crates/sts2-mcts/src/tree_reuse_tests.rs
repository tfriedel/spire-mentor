//! GH #830, deep bets build 2: [`Mcts::reroot_observed`] (tree reuse
//! between the pilot's own decisions) and the [`MctsConfig::tt_hash_v2`]
//! transposition key, checked against the search's own arena.
//!
//! Lives inside the crate so it can walk the arena and probe the
//! transposition table.

use super::*;
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::{PotionKind, PowerKind};

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::relic_effects::ensure_installed();
}

/// The opening position of the first combat of a fresh Ironclad run
/// (seed `seed`), reached by first-option picks.
fn first_combat(seed: i32) -> CombatState {
    install();
    let mut rm: RunManager =
        sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed,
            sts2_core::CharacterId::Ironclad,
            0,
        )
        .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    for _ in 0..64 {
        if rm.phase() == Phase::Combat {
            return rm.combat.clone().expect("in combat");
        }
        let actions = rm.get_available_actions();
        assert!(!actions.is_empty(), "run stalled before its first combat");
        let _ = rm.take_action(actions[0]);
    }
    panic!("no combat within 64 run actions");
}

/// The deployed honest search's config (`BeliefDraw` and its dialect,
/// depth 200) at `sims` simulations and seed `seed`.
fn deployed(sims: u32, seed: u64) -> MctsConfig {
    let mut cfg = MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: 200,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    cfg.apply_mode_dialect();
    cfg
}

const SIMS: u32 = 200;

fn searched(state: &CombatState, cfg: MctsConfig) -> Mcts<UniformPolicy> {
    let mut m = Mcts::new(state.clone(), cfg).with_leaf_eval(LeafEval::TexelWide);
    m.try_search_n(cfg.max_simulations)
        .expect("non-terminal root");
    m
}

/// The most-visited root child reached by a deterministic edge (a decision
/// node, not a chance node), as the concrete action and the child.
fn deterministic_root_child(m: &Mcts<UniformPolicy>) -> (Action, NodeId) {
    let root = m.tree.node(m.tree.root);
    root.children
        .iter()
        .filter(|(_, c)| m.tree.node(*c).kind == NodeKind::Decision)
        .max_by_key(|(_, c)| m.tree.node(*c).visits)
        .map(|(e, c)| (e.resolve(&m.tree.root_state).expect("root edge"), *c))
        .expect("a deterministic root child")
}

/// The root's `EndTurn` chance child.
fn end_turn_child(m: &Mcts<UniformPolicy>) -> NodeId {
    let root = m.tree.node(m.tree.root);
    root.children
        .iter()
        .find(|(e, _)| *e == EdgeAction::Other(Action::EndTurn))
        .map(|(_, c)| *c)
        .filter(|&c| m.tree.node(c).kind == NodeKind::Chance)
        .expect("EndTurn expanded as a chance node")
}

/// Nodes reachable from `from`, each with its shallowest depth.
fn reachable(m: &Mcts<UniformPolicy>, from: NodeId) -> Vec<(NodeId, usize)> {
    let mut seen = vec![false; m.tree.arena.len()];
    let mut out = vec![(from, 0)];
    seen[from.0 as usize] = true;
    let mut i = 0;
    while i < out.len() {
        let (id, d) = out[i];
        i += 1;
        let n = m.tree.node(id);
        for s in n
            .children
            .iter()
            .map(|(_, c)| *c)
            .chain(n.chance_outcomes.iter().map(|o| o.child))
        {
            if !seen[s.0 as usize] {
                seen[s.0 as usize] = true;
                out.push((s, d + 1));
            }
        }
    }
    out
}

/// The arena is self-consistent: every edge points inside it, every chance
/// node's parent lists it as a child, the root has no parent and holds the
/// root state, and every node is reachable from the root.
fn assert_tree_consistent(m: &Mcts<UniformPolicy>) {
    let len = m.tree.arena.len();
    assert_eq!(m.tree.root, NodeId(0));
    let root = m.tree.node(m.tree.root);
    assert!(root.parent.is_none());
    assert_eq!(
        state_hash(root.realized.as_deref().expect("root state")),
        state_hash(&m.tree.root_state)
    );
    for (i, n) in m.tree.arena.iter().enumerate() {
        for (_, c) in &n.children {
            assert!((c.0 as usize) < len, "node {i}: child out of the arena");
        }
        for o in &n.chance_outcomes {
            assert!(
                (o.child.0 as usize) < len,
                "node {i}: outcome out of the arena"
            );
        }
        if n.kind == NodeKind::Chance {
            let p = n.parent.expect("a chance node keeps its parent");
            assert!(
                m.tree
                    .node(p)
                    .children
                    .iter()
                    .any(|(_, c)| c.0 as usize == i),
                "chance node {i} is not its parent's child"
            );
        } else {
            assert!(n.realized.is_some(), "decision node {i} has no state");
        }
    }
    assert_eq!(
        reachable(m, m.tree.root).len(),
        len,
        "unreachable nodes kept"
    );
}

/// Every non-root decision node can be found in the transposition table
/// under its key (or a node with the same key is there: the first one seen
/// keeps a shared key). The depth bucket is the length of the path the node
/// was FIRST reached by, which is never shorter than its shallowest depth but
/// can be longer: a transposed parent reached by a longer path first files its
/// new children one bucket down. So the node sits in its shallowest depth's
/// bucket or a deeper one.
fn assert_tt_covers_the_tree(m: &Mcts<UniformPolicy>) {
    let tt = m.transposition.as_ref().expect("table allocated");
    for (id, d) in reachable(m, m.tree.root).into_iter().skip(1) {
        let n = m.tree.node(id);
        if n.kind != NodeKind::Decision {
            continue;
        }
        // `search_hash` already carries a pruned node's dominance salt
        // (GH #846); the v2 key folds the same salt in.
        let key_of = |n: &MctsNode| {
            if m.config.tt_hash_v2 {
                sts2_core::tt_hash_v2(n.realized.as_deref().expect("state"))
                    ^ dominance::salt(&n.dominance_pruned)
            } else {
                n.search_hash
            }
        };
        #[allow(clippy::cast_possible_truncation)]
        let shallowest = (d / 4) as u8;
        let key = key_of(n);
        let hit = (shallowest..=u8::MAX)
            .find_map(|bucket| tt.peek((key, bucket)))
            .expect("kept node missing from the table");
        assert!((hit.0 as usize) < m.tree.arena.len());
        assert_eq!(key_of(m.tree.node(hit)), key, "table entry for another key");
    }
}

/// Every statistic of the root's children, floats as bits.
fn outcome_bits(m: &Mcts<UniformPolicy>) -> Vec<(String, u32, u32, u32, u32, u32, u32)> {
    m.root_outcomes()
        .iter()
        .map(|o| {
            (
                format!("{:?}", o.action),
                o.visits,
                o.wins,
                o.losses,
                o.ties,
                o.value_sum.to_bits(),
                o.value_sq_sum.to_bits(),
            )
        })
        .collect()
}

/// `s` with its hidden information replaced: the unknown draw pile reversed
/// and every combat RNG stream (monster AI included) reseeded. Nothing a
/// player can see changes.
fn rehide(s: &CombatState, salt: u64) -> CombatState {
    assert!(
        s.player.pinned_top.is_empty() && s.player.pinned_bottom.is_empty(),
        "reversing a pinned pile would move what the player knows"
    );
    let mut t = s.clone();
    t.player.draw.reverse();
    t.run_rng.reseed_combat_streams(salt, 7);
    t.rng = sts2_core::dotnet_rng::Rng::from_search_seed(salt ^ 0xDEAD_BEEF);
    assert_eq!(
        sts2_core::chance_outcome_hash(&t),
        sts2_core::chance_outcome_hash(s),
        "rehide changed what the player sees"
    );
    t
}

#[test]
fn reroot_keeps_the_subtree_of_a_deterministic_play() {
    let s = first_combat(4);
    let mut m = searched(&s, deployed(SIMS, 11));
    let (a, child) = deterministic_root_child(&m);
    let child_visits = m.tree.node(child).visits;
    let subtree = reachable(&m, child).len();
    assert!(
        child_visits > 0 && subtree > 1,
        "{child_visits} visits, {subtree} nodes"
    );

    let mut t = s.clone();
    let _ = core_apply(&mut t, a.clone());
    let kept = m
        .reroot_observed(&a, &t, deployed(SIMS, 12))
        .expect("the true play matches its own child");
    assert_eq!(kept.retained_visits, child_visits);
    assert_eq!(kept.retained_nodes, subtree);
    assert!(!kept.via_chance);

    // The root is exactly what a fresh search on `t` roots at.
    let fresh = Mcts::new(t.clone(), deployed(SIMS, 12));
    assert_eq!(
        state_hash(&m.tree.root_state),
        state_hash(&fresh.tree.root_state)
    );
    assert_eq!(m.config.seed, 12);
    assert_eq!(m.config.max_rollout_depth, fresh.config.max_rollout_depth);
    assert_tree_consistent(&m);
    assert_tt_covers_the_tree(&m);

    // A full budget of NEW simulations on top of the kept ones.
    m.try_search_n(SIMS).expect("searchable");
    assert_eq!(m.tree.node(m.tree.root).visits, child_visits + SIMS);
    assert_tree_consistent(&m);
}

#[test]
fn reroot_falls_back_and_leaves_the_tree_alone_on_a_mismatch() {
    let start = first_combat(4);
    let mut m = searched(&start, deployed(SIMS, 11));
    let (play, _) = deterministic_root_child(&m);
    let before = (
        m.tree.arena.len(),
        m.tree.node(m.tree.root).visits,
        state_hash(&m.tree.root_state),
    );
    let unchanged = |m: &Mcts<UniformPolicy>| {
        (
            m.tree.arena.len(),
            m.tree.node(m.tree.root).visits,
            state_hash(&m.tree.root_state),
        )
    };

    // Deterministic edge, but the game shows something else (an enemy one
    // HP lower than the play left it).
    let mut next = start.clone();
    let _ = core_apply(&mut next, play.clone());
    let enemy = next.enemy_uids[0].0;
    let foe = next
        .creatures
        .iter_mut()
        .find(|c| c.id == enemy)
        .expect("enemy");
    foe.current_hp -= 1;
    assert_eq!(
        m.reroot_observed(&play, &next, deployed(SIMS, 12)),
        Err(RerootMiss::Mismatch)
    );
    assert_eq!(unchanged(&m), before);

    // A potion the play could not have produced is a mismatch too (the
    // observation hash does not read the belt).
    let mut next = start.clone();
    let _ = core_apply(&mut next, play.clone());
    next.player.potions.push(PotionKind::StrengthPotion);
    assert_eq!(
        m.reroot_observed(&play, &next, deployed(SIMS, 12)),
        Err(RerootMiss::Mismatch)
    );

    // Chance edge whose realised outcome no stored outcome matches.
    let mut next = start.clone();
    let _ = core_apply(&mut next, Action::EndTurn);
    let enemy = next.enemy_uids[0].0;
    let foe = next
        .creatures
        .iter_mut()
        .find(|c| c.id == enemy)
        .expect("enemy");
    foe.current_hp = 1;
    assert_eq!(
        m.reroot_observed(&Action::EndTurn, &next, deployed(SIMS, 12)),
        Err(RerootMiss::OutcomeUnseen)
    );
    assert_eq!(unchanged(&m), before);

    // An action the search never expanded.
    let mut thin = Mcts::new(start.clone(), deployed(2, 11)).with_leaf_eval(LeafEval::TexelWide);
    thin.try_search_n(2).expect("searchable");
    let unexpanded = legal_actions(&start)
        .into_iter()
        .find(|x| thin.root_child_for(x).is_none())
        .expect("2 sims leave a root action unexpanded");
    let mut next = start.clone();
    let _ = core_apply(&mut next, unexpanded.clone());
    assert_eq!(
        thin.reroot_observed(&unexpanded, &next, deployed(SIMS, 12)),
        Err(RerootMiss::NotExpanded)
    );

    // Another hidden-information mode than the tree was built under.
    let mut cfg = deployed(SIMS, 12);
    cfg.hidden_info = HiddenInfoMode::DrawChance;
    let mut next = start.clone();
    let _ = core_apply(&mut next, play.clone());
    assert_eq!(
        m.reroot_observed(&play, &next, cfg),
        Err(RerootMiss::UnsupportedMode)
    );
    assert_eq!(unchanged(&m), before);
}

#[test]
fn reroot_through_end_turn_keeps_the_matching_outcome() {
    let s = first_combat(4);
    let mut m = searched(&s, deployed(SIMS, 11));
    let chance = end_turn_child(&m);
    let outcome = m
        .tree
        .node(chance)
        .chance_outcomes
        .iter()
        .map(|o| o.child)
        .max_by_key(|&c| m.tree.node(c).visits)
        .expect("EndTurn has an outcome");
    let want_visits = m.tree.node(outcome).visits;
    let want_nodes = reachable(&m, outcome).len();
    // The game realised exactly that outcome — but in a world with its
    // own hidden draw order and RNG.
    let realised = rehide(
        m.tree.node(outcome).realized.as_deref().expect("state"),
        0x5EED,
    );
    let kept = m
        .reroot_observed(&Action::EndTurn, &realised, deployed(SIMS, 12))
        .expect("a stored outcome matches");
    assert!(kept.via_chance);
    assert_eq!(kept.retained_visits, want_visits);
    assert_eq!(kept.retained_nodes, want_nodes);
    assert_tree_consistent(&m);
    assert_tt_covers_the_tree(&m);
    m.try_search_n(SIMS).expect("searchable");
    assert_eq!(m.tree.node(m.tree.root).visits, want_visits + SIMS);
    assert_tree_consistent(&m);
}

/// The honesty pin: two games that differ only in hidden information (draw
/// order, every RNG stream) give a reused search the same tree and the same
/// statistics, bit for bit. A reroot that let the caller's state into the
/// tree would read the reversed pile and the reseeded streams.
#[test]
fn reroot_reads_no_hidden_information_from_the_new_state() {
    let s = first_combat(4);
    let mut m1 = searched(&s, deployed(SIMS, 11));
    let mut m2 = searched(&s, deployed(SIMS, 11));
    let (a, _) = deterministic_root_child(&m1);
    let mut t1 = s.clone();
    let _ = core_apply(&mut t1, a.clone());
    let t2 = rehide(&t1, 0xC0FFEE);
    assert_ne!(
        state_hash(&t1),
        state_hash(&t2),
        "rehide moved nothing hidden"
    );
    let k1 = m1
        .reroot_observed(&a, &t1, deployed(SIMS, 12))
        .expect("match");
    let k2 = m2
        .reroot_observed(&a, &t2, deployed(SIMS, 12))
        .expect("match");
    assert_eq!(k1, k2);
    assert_eq!(
        state_hash(&m1.tree.root_state),
        state_hash(&m2.tree.root_state)
    );
    let p1 = m1.try_search_n(SIMS).expect("searchable");
    let p2 = m2.try_search_n(SIMS).expect("searchable");
    assert_eq!(p1, p2);
    assert_eq!(outcome_bits(&m1), outcome_bits(&m2));

    // Same through a chance edge.
    let chance1 = end_turn_child(&m1);
    let o = m1.tree.node(chance1).chance_outcomes[0].child;
    let seen = m1.tree.node(o).realized.as_deref().expect("state").clone();
    let (r1, r2) = (rehide(&seen, 1), rehide(&seen, 2));
    let k1 = m1
        .reroot_observed(&Action::EndTurn, &r1, deployed(SIMS, 13))
        .expect("match");
    let k2 = m2
        .reroot_observed(&Action::EndTurn, &r2, deployed(SIMS, 13))
        .expect("match");
    assert_eq!(k1, k2);
    m1.try_search_n(SIMS).expect("searchable");
    m2.try_search_n(SIMS).expect("searchable");
    assert_eq!(outcome_bits(&m1), outcome_bits(&m2));
}

#[test]
fn reroot_is_deterministic() {
    let run = || {
        let start = first_combat(9);
        let mut m = searched(&start, deployed(SIMS, 21));
        let mut log = vec![outcome_bits(&m)];
        let (play, _) = deterministic_root_child(&m);
        let mut next = start.clone();
        let _ = core_apply(&mut next, play.clone());
        let k = m
            .reroot_observed(&play, &next, deployed(SIMS, 22))
            .expect("match");
        m.try_search_n(SIMS).expect("searchable");
        log.push(outcome_bits(&m));
        (log, k, m.tt_hits(), m.tree.arena.len())
    };
    assert_eq!(run(), run());
}

/// Default config: the v1 key, so the deployed search is unchanged.
#[test]
fn tt_hash_v2_is_off_by_default() {
    assert!(!MctsConfig::default().tt_hash_v2);
    assert!(!deployed(SIMS, 1).tt_hash_v2);
}

/// `s` holding `kind` in belt slot 0 of a 3-slot belt.
fn with_potion(s: &CombatState, kind: PotionKind) -> CombatState {
    let mut t = s.clone();
    t.player.potions.clear();
    t.player.potion_slot_indices.clear();
    t.player.potion_slots = 3;
    assert!(t.player.add_potion_first_free_slot(kind));
    t
}

/// `s` after drinking slot 0, or `None` when no drink of slot 0 is legal.
fn drink(s: &CombatState) -> Option<CombatState> {
    let a = legal_actions(s)
        .into_iter()
        .find(|a| matches!(a, Action::DrinkPotion { potion_slot: 0, .. }))?;
    let mut t = s.clone();
    let _ = core_apply(&mut t, a);
    Some(t)
}

/// The failing-first pin for v2's potion fix: drinking any of these
/// potions at the opening of a fight leaves the v1 key (`search_hash`) —
/// and the chance-outcome dedup key — unchanged, so the table merges "drank
/// it" into "still holds it". Two blind spots line up: the belt (v1 hashes
/// the legacy potion arena) and the power the potion grants (it lands on
/// the player's mirror creature, which v1 does not visit). The v2 key sees
/// every drink.
#[test]
fn tt_hash_v1_misses_these_drinks_and_v2_sees_them() {
    let s = first_combat(4);
    for kind in [
        PotionKind::StrengthPotion,
        PotionKind::DexterityPotion,
        PotionKind::FlexPotion,
        PotionKind::LuckyTonic,
        PotionKind::StableSerum,
        PotionKind::FyshOil,
    ] {
        let before = with_potion(&s, kind);
        let after = drink(&before).unwrap_or_else(|| panic!("{kind:?} not drinkable"));
        assert!(
            after.player.potions.is_empty(),
            "{kind:?}: belt not emptied"
        );
        assert!(
            after.player.powers.is_empty(),
            "{kind:?}: legacy power list used"
        );
        assert_eq!(
            sts2_core::search_hash(&before),
            sts2_core::search_hash(&after),
            "{kind:?}: v1 key moved"
        );
        assert_eq!(
            sts2_core::chance_outcome_hash(&before),
            sts2_core::chance_outcome_hash(&after),
            "{kind:?}: observation hash moved"
        );
        assert_ne!(
            sts2_core::tt_hash_v2(&before),
            sts2_core::tt_hash_v2(&after),
            "{kind:?}: v2 key did not move"
        );
    }
}

/// Over all 63 potions: every drink moves the v2 key; report how many the
/// v1 key misses (the census behind the pin above).
#[test]
fn tt_hash_v2_moves_on_every_drink() {
    let s = first_combat(4);
    let mut v1_blind = Vec::new();
    let mut drinkable = 0;
    for kind in PotionKind::ALL {
        let before = with_potion(&s, kind);
        let Some(after) = drink(&before) else {
            continue;
        };
        drinkable += 1;
        assert_ne!(
            sts2_core::tt_hash_v2(&before),
            sts2_core::tt_hash_v2(&after),
            "{kind:?}: v2 key did not move"
        );
        if sts2_core::search_hash(&before) == sts2_core::search_hash(&after) {
            v1_blind.push(kind);
        }
    }
    eprintln!(
        "v1 key unchanged by {} of {drinkable} drinkable potions: {v1_blind:?}",
        v1_blind.len()
    );
    assert!(drinkable >= 50, "only {drinkable} potions drinkable");
    assert!(v1_blind.len() >= 6);
}

/// The failing-first pin for v2's power-order fix: the same powers applied
/// in the other order are one position. v1 keys them apart (its power rows
/// carry the arena uid, i.e. the application order); v2 does not.
#[test]
fn tt_hash_v2_merges_power_application_orders() {
    let s = first_combat(4);
    let player = s.player.creature_uid.expect("player creature");
    let enemy = s.enemy_uids[0];
    let apply_in = |order: &[(sts2_core::CreatureUid, PowerKind, i32)]| {
        let mut t = s.clone();
        for &(who, kind, amount) in order {
            sts2_core::card_effects::apply_power_to_creature(&mut t, who, kind, amount);
        }
        t
    };
    let vuln_then_weak = apply_in(&[
        (enemy, PowerKind::Vulnerable, 2),
        (enemy, PowerKind::Weak, 1),
    ]);
    let weak_then_vuln = apply_in(&[
        (enemy, PowerKind::Weak, 1),
        (enemy, PowerKind::Vulnerable, 2),
    ]);
    assert_ne!(
        sts2_core::search_hash(&vuln_then_weak),
        sts2_core::search_hash(&weak_then_vuln),
        "v1 already merges the orders"
    );
    assert_eq!(
        sts2_core::tt_hash_v2(&vuln_then_weak),
        sts2_core::tt_hash_v2(&weak_then_vuln)
    );
    // Across creatures: a player power applied first shifts the enemy's
    // power uid.
    let a = apply_in(&[
        (enemy, PowerKind::Vulnerable, 2),
        (player, PowerKind::Strength, 2),
    ]);
    let b = apply_in(&[
        (player, PowerKind::Strength, 2),
        (enemy, PowerKind::Vulnerable, 2),
    ]);
    assert_ne!(sts2_core::search_hash(&a), sts2_core::search_hash(&b));
    assert_eq!(sts2_core::tt_hash_v2(&a), sts2_core::tt_hash_v2(&b));
    // And v2 still tells different powers apart.
    assert_ne!(
        sts2_core::tt_hash_v2(&vuln_then_weak),
        sts2_core::tt_hash_v2(&apply_in(&[(enemy, PowerKind::Vulnerable, 2)]))
    );
}

/// The failing-first pin for v2's player-power fix. Player powers live on
/// the player's mirror creature, which no section of v1 (or of the
/// observation hash) visits: the player with Strength 2, with Strength 3 and
/// with none is one key under v1. v2 tells all three apart.
#[test]
fn tt_hash_v1_cannot_see_the_players_powers_and_v2_can() {
    let s = first_combat(4);
    let player = s.player.creature_uid.expect("player creature");
    let with = |amount: i32| {
        let mut t = s.clone();
        sts2_core::card_effects::apply_power_to_creature(
            &mut t,
            player,
            PowerKind::Strength,
            amount,
        );
        t
    };
    let (none, two, three) = (s.clone(), with(2), with(3));
    assert!(
        two.player.powers.is_empty(),
        "the legacy list got the power"
    );
    assert_eq!(
        two.player_power_uids().len(),
        1,
        "the mirror creature has it"
    );
    for h in [sts2_core::search_hash, sts2_core::chance_outcome_hash] {
        assert_eq!(
            h(&none),
            h(&two),
            "v1-family hash saw the player's Strength"
        );
        assert_eq!(h(&two), h(&three));
    }
    let v2 = sts2_core::tt_hash_v2;
    assert_ne!(v2(&none), v2(&two));
    assert_ne!(v2(&two), v2(&three));
}

/// The knob reaches the table: on a belt the v1 key cannot see, the two
/// keys merge different nodes, so the searches differ.
#[test]
fn tt_hash_v2_changes_the_search_that_reads_it() {
    let s = with_potion(&first_combat(4), PotionKind::StrengthPotion);
    let v1 = searched(&s, deployed(SIMS, 31));
    let mut cfg = deployed(SIMS, 31);
    cfg.tt_hash_v2 = true;
    let v2 = searched(&s, cfg);
    assert_ne!(
        (outcome_bits(&v1), v1.tt_hits()),
        (outcome_bits(&v2), v2.tt_hits())
    );
}

/// A first combat where dominance pruning fires: a lone FuzzyWurmCrawler
/// against Bash and Strike in the opening hand (Bash, then Strike, beats
/// the other order). Seed 4's Nibbit reads its ally, which turns pruning
/// off ([`dominance::enemy_hp_not_monotone`]).
const PRUNING_SEED: i32 = 3;

/// A deployed search of `s` with dominance pruning (GH #846) on, and the
/// number of its nodes that carry a pruned set.
fn pruned_search(s: &CombatState, mut cfg: MctsConfig) -> (Mcts<UniformPolicy>, usize) {
    cfg.dominance_pruning = true;
    let m = searched(s, cfg);
    let pruned = m
        .tree
        .arena
        .iter()
        .filter(|n| !n.dominance_pruned.is_empty())
        .count();
    (m, pruned)
}

/// GH #846 salts a pruned child's transposition key with its pruned set, so
/// a node reached by another edge never inherits a set computed for another
/// parent. The v2 key replaces `search_hash`, so it has to carry the same
/// salt: without it a pruned node is filed under the bare v2 hash and
/// merges with the unpruned node of the same position.
#[test]
fn tt_hash_v2_keeps_the_dominance_salt() {
    let mut cfg = deployed(SIMS, 41);
    cfg.tt_hash_v2 = true;
    let (m, pruned) = pruned_search(&first_combat(PRUNING_SEED), cfg);
    assert!(pruned > 0, "nothing was pruned: the pin is vacuous");
    assert_tt_covers_the_tree(&m);
}

/// Dominance pruning is path-conditional: a node's pruned set rests on a
/// reorder of the card plays that led to it, and below the new root those
/// can include plays above it, which the game has already made. Reuse
/// refuses a tree built with pruning rather than keep those sets (a kept
/// root could even lose legal moves); a tree built without it carries no
/// sets, so turning pruning on for the next search is fine.
#[test]
fn reroot_refuses_a_dominance_pruned_tree() {
    let start = first_combat(PRUNING_SEED);
    let (mut m, pruned) = pruned_search(&start, deployed(SIMS, 42));
    assert!(pruned > 0, "nothing was pruned: the pin is vacuous");
    let (play, _) = deterministic_root_child(&m);
    let mut next = start.clone();
    let _ = core_apply(&mut next, play.clone());
    let mut on = deployed(SIMS, 43);
    on.dominance_pruning = true;
    assert_eq!(
        m.reroot_observed(&play, &next, deployed(SIMS, 43)),
        Err(RerootMiss::UnsupportedMode)
    );
    assert_eq!(
        m.reroot_observed(&play, &next, on),
        Err(RerootMiss::UnsupportedMode)
    );

    let mut plain = searched(&start, deployed(SIMS, 42));
    let (play, _) = deterministic_root_child(&plain);
    let mut next = start.clone();
    let _ = core_apply(&mut next, play.clone());
    plain
        .reroot_observed(&play, &next, on)
        .expect("an unpruned tree is kept");
    plain.try_search_n(SIMS).expect("searchable");
    assert_tree_consistent(&plain);
}
