//! `MctsConfig::effective_incoming`, `MctsConfig::removed_counts_forms` and
//! the playout net's feature version 3 (GH #830 boss audit): the off
//! positions and the v1 net must be the engine before any of them existed.
//!
//! The fingerprints are `tests/net_expand_order.rs`'s, captured on
//! `5638d0252` before either lever (or `net_expand_order`) was written; they
//! pin every node's visits, value bits, child edges, chance outcomes and
//! untried queue. The encounter-level tests (effective damage against the
//! wire's displayed number, the form rules on Test Subject / Waterfall
//! Giant / …) live in `sts2-wire/tests/eval_levers.rs`, which has the
//! scenario builder.
//!
//! Every test here reads the built-in Ironclad net, so the file compiles
//! only where `build.rs` found it.
#![cfg(rollout_net_ironclad)]

mod common;

use common::build_vertical_slice_hp;
use rand::{Rng, SeedableRng};
use sts2_core::{apply, is_terminal, legal_actions, CharacterId, CombatState};
use sts2_mcts::enemy_eval::{enemy_hp_remaining, IncomingDamage};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::{FeatureVersion, RolloutNet, StepCtx, FEATURES_V1, FEATURES_V3};
use sts2_mcts::{HiddenInfoMode, Mcts, MctsConfig};

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

/// A hash of the whole search tree (same definition as
/// `tests/net_expand_order.rs`).
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

/// `tests/net_expand_order.rs`'s cases: the deployed dialect on three deals,
/// and the legacy clairvoyant heuristic search on one.
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

/// Captured on `5638d0252`, before either lever existed; re-captured when
/// the card cost model (`CardEnergyCost`) stopped leaving zero-valued cost
/// keys in `combat_vars`, which the search hash (and the streams seeded from
/// it) folds in. The bit-identity pin is unaffected: both knobs are off here.
const PRE_CHANGE_FINGERPRINTS: [u64; 4] = [
    0xb2b7_6fd7_8bda_b76f,
    0x1fcb_f661_aac3_eeb0,
    0x85b8_ecec_60e3_05b4,
    0xefbf_179c_ce3b_1533,
];

fn fingerprints(edit: impl Fn(&mut MctsConfig)) -> Vec<u64> {
    pin_cases()
        .into_iter()
        .map(|(state, mut cfg)| {
            edit(&mut cfg);
            let mut m = Mcts::new(state, cfg);
            let chosen = m.search();
            tree_fingerprint(&m, &chosen)
        })
        .collect()
}

#[test]
fn knobs_off_tree_is_byte_identical_to_pre_change() {
    no_experiment_net();
    let base = MctsConfig::default();
    assert!(!base.effective_incoming && !base.removed_counts_forms);
    let got = fingerprints(|cfg| {
        cfg.effective_incoming = false;
        cfg.removed_counts_forms = false;
    });
    eprintln!("fingerprints: {got:#018x?}");
    assert_eq!(got, PRE_CHANGE_FINGERPRINTS.to_vec());
}

/// No Nibbit carries a form power, so the form-aware reading equals the
/// base one on every state these searches visit: the knob on must not move
/// a bit of them.
#[test]
fn removed_counts_forms_is_inert_without_a_form_enemy() {
    no_experiment_net();
    let got = fingerprints(|cfg| cfg.removed_counts_forms = true);
    assert_eq!(got, PRE_CHANGE_FINGERPRINTS.to_vec());
}

/// Decision states of random walks from starter-deck fights (the v1 golden
/// test's walk, shorter).
fn walk_states() -> Vec<CombatState> {
    let mut out = Vec::new();
    for seed in 1..=12u64 {
        let hp = 20 + 4 * i32::try_from(seed).expect("small");
        let mut s = build_vertical_slice_hp(seed, hp, 30 + 3 * hp);
        if seed % 3 == 0 {
            s.player.potions.push(sts2_core::PotionKind::FirePotion);
        }
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed * 31);
        for _ in 0..40 {
            if is_terminal(&s) {
                break;
            }
            let acts = legal_actions(&s);
            if acts.is_empty() {
                break;
            }
            out.push(s.clone());
            let i = rng.gen_range(0..acts.len());
            let _ = apply(&mut s, acts[i].clone());
        }
    }
    out
}

/// Whether every enemy's telegraph reads the same both ways.
fn intents_unmodified(s: &CombatState) -> bool {
    s.creatures
        .iter()
        .all(|c| IncomingDamage::Base.total(s, c) == IncomingDamage::Effective.total(s, c))
}

/// The built-in net file relabeled `feature_version: 3`.
fn builtin_as(version: u32) -> RolloutNet {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/rollout_nets/ironclad.json"
    ))
    .expect("built-in net file");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("json");
    v["feature_version"] = serde_json::json!(version);
    RolloutNet::from_json(&v.to_string()).expect("loads")
}

/// v3 is v1's layout: the rl2 weights load as v3 unchanged, and where no
/// modifier touches an intent and no enemy has forms, every v3 feature and
/// logit is v1's to the bit. Where a modifier does touch one, v3's
/// `incoming` is the displayed sum and v1's the base one.
#[test]
fn v3_is_v1_where_nothing_modifies_an_intent() {
    let (v1, v3) = (builtin_as(1), builtin_as(3));
    assert_eq!(v3.feature_spec().version, FeatureVersion::V3);
    assert_eq!((v3.state_dim(), v3.action_dim()), (32, 24));
    let (mut same, mut differ) = (0, 0);
    for s in walk_states() {
        let acts = legal_actions(&s);
        let (c1, c3) = (
            StepCtx::with_spec(&s, &acts, &FEATURES_V1),
            StepCtx::with_spec(&s, &acts, &FEATURES_V3),
        );
        let incoming = |read: IncomingDamage| -> i32 {
            s.enemy_uids
                .iter()
                .filter_map(|u| sts2_core::creature_by_id(&s.creatures, u.0))
                .filter(|c| c.current_hp > 0 || c.is_beaten_sentinel())
                .map(|c| read.total(&s, c))
                .sum()
        };
        #[allow(clippy::cast_precision_loss)]
        let mhp = s.player.max_hp.max(1) as f32;
        #[allow(clippy::cast_precision_loss)]
        let expect = |v: i32| (v as f32 / mhp).clamp(0.0, 3.0);
        assert_eq!(
            c1.state[10].to_bits(),
            expect(incoming(IncomingDamage::Base)).to_bits()
        );
        assert_eq!(
            c3.state[10].to_bits(),
            expect(incoming(IncomingDamage::Effective)).to_bits()
        );
        let l1: Vec<u32> = v1
            .logits(&s, &acts, END_TURN_PRIORITY)
            .iter()
            .map(|l| l.to_bits())
            .collect();
        let l3: Vec<u32> = v3
            .logits(&s, &acts, END_TURN_PRIORITY)
            .iter()
            .map(|l| l.to_bits())
            .collect();
        let (h, m) = enemy_hp_remaining(&s, true);
        if intents_unmodified(&s) && (h, m) == enemy_hp_remaining(&s, false) {
            let bits = |c: &StepCtx| c.state.map(f32::to_bits);
            assert_eq!(bits(&c1), bits(&c3));
            assert_eq!(l1, l3);
            same += 1;
        } else {
            differ += 1;
        }
    }
    assert!(same > 100, "{same} unmodified states");
    eprintln!("{same} states read alike, {differ} differ");
}
