//! `MctsConfig::net_expand_order` (GH #830 rung 4): order the untried
//! actions of NON-ROOT decision nodes by the playout net's own preference.
//!
//! Off (the default) must be byte-identical to the engine before the knob
//! existed: the fingerprints below were captured on `5638d0252`, before
//! the knob was written, and pin every node's visits, value bits, child
//! edges, chance outcomes and remaining untried queue.

mod common;

use common::build_vertical_slice_hp;
use sts2_core::{legal_actions, CharacterId, CombatState};
use sts2_mcts::rollout_net::playout_logits;
use sts2_mcts::{
    EdgeAction, ExpandOrderStats, HiddenInfoMode, Mcts, MctsConfig, NodeKind, RolloutPolicyKind,
};

/// `STS2_ROLLOUT_NET` replaces the built-in net, so a pin taken with the
/// built-in net cannot hold under it; say so instead of failing obscurely.
fn no_experiment_net() {
    assert!(
        std::env::var_os("STS2_ROLLOUT_NET").is_none(),
        "unset STS2_ROLLOUT_NET to run this test: it replaces every built-in net"
    );
}

/// FNV-1a over a byte slice: stable across runs and builds.
fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    let mut h = hash;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// A hash of the whole search tree: per node (arena order) its kind,
/// visits, value-sum bits, child edges, chance outcomes and untried queue,
/// then the chosen action.
fn tree_fingerprint<P: sts2_mcts::Policy>(m: &Mcts<P>, chosen: &sts2_core::Action) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325_u64;
    for n in &m.tree.arena {
        let children: Vec<(String, u32)> = n
            .children
            .iter()
            .map(|(e, c)| (format!("{e:?}"), c.0))
            .collect();
        let outcomes: Vec<(u32, u32)> =
            n.chance_outcomes.iter().map(|o| (o.n, o.child.0)).collect();
        let line = format!(
            "{:?}|{}|{:08x}|{children:?}|{outcomes:?}|{:?}\n",
            n.kind,
            n.visits,
            n.value_sum.to_bits(),
            n.untried_actions,
        );
        h = fnv1a(h, line.as_bytes());
    }
    fnv1a(h, format!("{chosen:?}").as_bytes())
}

/// The deployed honest dialect (BeliefDraw, the playout net) at `sims`.
fn deployed_cfg(seed: u64, sims: u32) -> MctsConfig {
    let mut cfg = MctsConfig {
        max_simulations: sims,
        max_rollout_depth: 200,
        seed,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    cfg.apply_mode_dialect();
    cfg
}

/// Ironclad starter deck vs a Nibbit, `seed` deals the opening hand.
fn ironclad(seed: u64) -> CombatState {
    let mut s = build_vertical_slice_hp(seed, 80, 40);
    s.player.character = Some(CharacterId::Ironclad);
    s
}

/// `(state, config)` cases the off-pin covers: the deployed dialect on
/// three deals, and the legacy clairvoyant heuristic search on one.
fn pin_cases() -> Vec<(CombatState, MctsConfig)> {
    vec![
        (ironclad(3), deployed_cfg(11, 300)),
        (ironclad(5), deployed_cfg(12, 300)),
        (ironclad(7), deployed_cfg(13, 300)),
        (
            ironclad(5),
            MctsConfig {
                max_simulations: 200,
                max_rollout_depth: 200,
                seed: 14,
                ..MctsConfig::default()
            },
        ),
    ]
}

/// Captured on `5638d0252` (before `net_expand_order` existed); re-captured
/// when the card cost model (`CardEnergyCost`) stopped leaving zero-valued
/// cost keys in `combat_vars`, which the search hash (and the streams seeded
/// from it) folds in.
const PRE_CHANGE_FINGERPRINTS: [u64; 4] = [
    0xb2b7_6fd7_8bda_b76f,
    0x1fcb_f661_aac3_eeb0,
    0x85b8_ecec_60e3_05b4,
    0xefbf_179c_ce3b_1533,
];

#[cfg(rollout_net_ironclad)]
#[test]
fn knob_off_tree_is_byte_identical_to_pre_change() {
    no_experiment_net();
    let got: Vec<u64> = pin_cases()
        .into_iter()
        .map(|(state, cfg)| {
            let mut m = Mcts::new(state, cfg);
            let chosen = m.search();
            tree_fingerprint(&m, &chosen)
        })
        .collect();
    eprintln!("fingerprints: {got:#018x?}");
    assert_eq!(got, PRE_CHANGE_FINGERPRINTS.to_vec());
}

/// `cfg` with the knob on.
fn on(mut cfg: MctsConfig) -> MctsConfig {
    cfg.net_expand_order = true;
    cfg
}

/// A finished search of `state` and its pick.
fn search(state: CombatState, cfg: MctsConfig) -> (Mcts, sts2_core::Action) {
    let mut m = Mcts::new(state, cfg);
    let chosen = m.search();
    (m, chosen)
}

/// The root's child edges in expansion order, then its remaining queue.
fn root_queue(m: &Mcts) -> String {
    let root = &m.tree.arena[m.tree.root.0 as usize];
    let edges: Vec<&EdgeAction> = root.children.iter().map(|(e, _)| e).collect();
    format!("{edges:?} / {:?}", root.untried_actions)
}

/// At every searched non-root decision node the first expanded child —
/// the first move of the continuation the tree stores — is the playout
/// net's argmax among the node's legal actions; the root keeps its
/// canonical order.
#[cfg(rollout_net_ironclad)]
#[test]
fn knob_on_first_child_is_the_nets_argmax_and_root_order_is_unchanged() {
    no_experiment_net();
    for (state, cfg) in pin_cases().into_iter().take(3) {
        let (off, _) = search(state.clone(), cfg);
        let (m, _) = search(state, on(cfg));
        assert_eq!(root_queue(&m), root_queue(&off), "root order unchanged");
        let mut checked = 0;
        for (i, n) in m.tree.arena.iter().enumerate() {
            if i == m.tree.root.0 as usize || n.kind != NodeKind::Decision || n.children.is_empty()
            {
                continue;
            }
            let st = n.realized.as_deref().expect("decision node has a state");
            let legal = legal_actions(st);
            if legal.len() < 2 {
                continue;
            }
            let logits = playout_logits(st, &legal, cfg.rollout_end_turn_priority).expect("a net");
            let first = n.children[0].0.resolve(st).expect("first edge resolves");
            let at = legal
                .iter()
                .position(|a| *a == first)
                .expect("first edge is legal");
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                logits[at].to_bits(),
                max.to_bits(),
                "node {i}: first child {first:?} has logit {} but the net's best is {max} ({legal:?} {logits:?})",
                logits[at]
            );
            checked += 1;
        }
        let stats = m.expand_order_stats();
        assert!(checked >= 10, "only {checked} non-root nodes checked");
        assert!(
            stats.scored > 0 && stats.scored == stats.considered,
            "{stats:?}"
        );
        assert!(
            stats.first_changed > 0,
            "the net never disagreed with the ladder: {stats:?}"
        );
        assert!(stats.first_changed <= stats.reordered && stats.reordered <= stats.scored);
        assert_eq!(off.expand_order_stats(), ExpandOrderStats::default());
    }
}

/// Knob on, same seed twice: the same tree, pick and counters. And the
/// knob does change the deployed search (the pin above is not vacuous).
#[cfg(rollout_net_ironclad)]
#[test]
fn knob_on_is_deterministic_for_a_fixed_seed() {
    no_experiment_net();
    let mut changed = 0;
    for (k, (state, cfg)) in pin_cases().into_iter().take(3).enumerate() {
        let (a, pick_a) = search(state.clone(), on(cfg));
        let (b, pick_b) = search(state, on(cfg));
        assert_eq!(pick_a, pick_b);
        let fp = tree_fingerprint(&a, &pick_a);
        assert_eq!(fp, tree_fingerprint(&b, &pick_b), "case {k}");
        assert_eq!(a.expand_order_stats(), b.expand_order_stats());
        changed += usize::from(fp != PRE_CHANGE_FINGERPRINTS[k]);
    }
    assert!(changed > 0, "the knob changed no deployed tree");
}

/// The knob is inert without the net playout (the clairvoyant heuristic
/// case: byte-identical to the pin, all counters zero) and for a fight
/// without a net (no character, since every character has one: nodes are
/// considered, none scored, tree unchanged).
#[cfg(rollout_net_ironclad)]
#[test]
fn knob_on_is_inert_without_a_playout_net() {
    no_experiment_net();
    let (state, cfg) = pin_cases().pop().expect("the heuristic case");
    assert_eq!(cfg.rollout_policy, RolloutPolicyKind::Heuristic);
    let (m, pick) = search(state, on(cfg));
    assert_eq!(tree_fingerprint(&m, &pick), PRE_CHANGE_FINGERPRINTS[3]);
    assert_eq!(m.expand_order_stats(), ExpandOrderStats::default());

    let mut no_net = ironclad(5);
    no_net.player.character = None;
    let cfg = deployed_cfg(12, 300);
    let (off, pick_off) = search(no_net.clone(), cfg);
    let (m, pick) = search(no_net, on(cfg));
    assert_eq!(
        tree_fingerprint(&m, &pick),
        tree_fingerprint(&off, &pick_off)
    );
    let stats = m.expand_order_stats();
    assert!(stats.considered > 0, "{stats:?}");
    assert_eq!(stats.scored, 0, "{stats:?}");
}
