//! Feature version 5 of the playout net (`sts2_mcts::rollout_net`): v4 plus
//! the card-mechanics block (`sts2_core::card_mechanics`), so a net can read
//! what another character's orb or Osty card does instead of keying it to an
//! untrained row.
//!
//! Every fight here is an Ironclad's: the point is a character holding cards
//! that are not its own (a generated Zap, a Poke, a Bodyguard).

mod common;

use common::{build_vertical_slice_hp, fixture};
use sts2_core::cards::CardId;
use sts2_core::{apply, legal_actions, Action, CharacterId, CombatState};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::{
    export_features_spec, ExportedFeatures, RolloutNet, ACTION_DIM_V4, ACTION_DIM_V5, FEATURES_V4,
    FEATURES_V5, STATE_DIM_V4, STATE_DIM_V5, V5_FOCUS, V5_LIGHTNING_SO_FAR, V5_ORBS, V5_ORB_SLOTS,
    V5_OSTY, V5_X_CHANNELS, V5_X_CHANNEL_KINDS, V5_X_ORB_PAYOFF, V5_X_OSTY_ATTACK,
    V5_X_OSTY_LETHAL, V5_X_OSTY_MISSING, V5_X_OVERFLOW, V5_X_SUMMON,
};

const EPS: f32 = 1e-6;

/// An Ironclad on turn 1 holding `hand`, the enemy at `enemy_hp`.
fn ironclad(hand: &[CardId], enemy_hp: i32, energy: i8) -> CombatState {
    let mut s = fixture(hand, enemy_hp, &[], energy);
    s.player.character = Some(CharacterId::Ironclad);
    s
}

fn v5(s: &CombatState) -> ExportedFeatures {
    export_features_spec(s, &FEATURES_V5)
}

/// `x(a)` of the first legal play of `id` (aimed at enemy 0 when targeted).
fn x_of(f: &ExportedFeatures, id: CardId) -> Vec<f32> {
    let name = format!("card:{}", id.to_python_name());
    f.actions
        .iter()
        .find(|(_, k, _)| k.name() == name)
        .map_or_else(|| panic!("no legal play of {id:?}"), |(_, _, x)| x.clone())
}

/// Play the first legal play of `id`.
fn play(s: &mut CombatState, id: CardId) {
    let name = format!("card:{}", id.to_python_name());
    let a: Action = v5(s)
        .actions
        .into_iter()
        .find(|(_, k, _)| k.name() == name)
        .map_or_else(|| panic!("no legal play of {id:?}"), |(a, _, _)| a);
    let _ = apply(s, a);
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

/// v5's first 44 / 29 features are v4's, value for value.
#[test]
fn v5_extends_v4_exactly() {
    let states = vec![
        ironclad(
            &[
                CardId::Zap,
                CardId::Poke,
                CardId::Bodyguard,
                CardId::StrikeIronclad,
            ],
            40,
            3,
        ),
        ironclad(
            &[
                CardId::Tempest,
                CardId::Chill,
                CardId::Dualcast,
                CardId::Bash,
            ],
            12,
            2,
        ),
        build_vertical_slice_hp(3, 70, 50),
    ];
    for s in &states {
        let (a, b) = (export_features_spec(s, &FEATURES_V4), v5(s));
        assert_eq!((a.state.len(), b.state.len()), (STATE_DIM_V4, STATE_DIM_V5));
        assert_eq!(a.state[..], b.state[..STATE_DIM_V4]);
        assert_eq!(a.actions.len(), b.actions.len());
        for ((aa, ak, ax), (ba, bk, bx)) in a.actions.iter().zip(&b.actions) {
            assert_eq!((aa, ak), (ba, bk));
            assert_eq!((ax.len(), bx.len()), (ACTION_DIM_V4, ACTION_DIM_V5));
            assert_eq!(ax[..], bx[..ACTION_DIM_V4]);
        }
    }
}

/// Strike, Defend and Bash touch neither system: their v5 block is zero, and
/// so is the state block of a fight with no belt and no Osty.
#[test]
fn cards_without_mechanics_leave_the_block_zero() {
    let s = ironclad(
        &[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash],
        40,
        3,
    );
    let f = v5(&s);
    assert!(
        f.state[STATE_DIM_V4..].iter().all(|v| *v == 0.0),
        "{:?}",
        f.state
    );
    for (_, _, x) in &f.actions {
        assert!(x[ACTION_DIM_V4..].iter().all(|v| *v == 0.0), "{x:?}");
    }
}

/// A generated Zap on an Ironclad: one Lightning into a belt that does not
/// exist yet. C# `OrbCmd.Channel` grows one slot for a character with no
/// base slots, so the first Zap fits and the second evokes the first.
#[test]
fn an_ironclad_reads_a_zap() {
    let mut s = ironclad(&[CardId::Zap, CardId::Zap, CardId::StrikeIronclad], 40, 3);
    let f = v5(&s);
    assert!(close(f.state[V5_ORB_SLOTS], 0.0));
    let x = x_of(&f, CardId::Zap);
    assert!(close(x[V5_X_CHANNELS], 1.0 / 3.0));
    assert!(close(x[V5_X_CHANNEL_KINDS], 1.0 / 3.0), "Lightning");
    assert!(x[V5_X_CHANNEL_KINDS + 1..V5_X_CHANNEL_KINDS + 6]
        .iter()
        .all(|v| *v == 0.0));
    assert!(close(x[V5_X_OVERFLOW], 0.0));
    // v1 sees a 1-cost Skill with no damage and no block.
    assert!(x[12] == 0.0 && x[13] == 0.0);

    play(&mut s, CardId::Zap);
    assert_eq!(s.player.orbs.capacity, 1, "C# grants one slot");
    let f = v5(&s);
    assert!(close(f.state[V5_ORB_SLOTS], 1.0 / 3.0));
    assert!(close(f.state[V5_ORB_SLOTS + 1], 1.0 / 3.0), "one orb held");
    assert!(close(f.state[V5_ORBS], 1.0 / 3.0), "a Lightning");
    assert!(close(f.state[V5_LIGHTNING_SO_FAR], 0.1));
    assert!(close(f.state[V5_FOCUS], 0.0));
    let x = x_of(&f, CardId::Zap);
    assert!(
        close(x[V5_X_OVERFLOW], 1.0 / 3.0),
        "the second Zap evokes the first"
    );
}

/// A Poke with no Osty does nothing (`Osty.CheckMissingWithAnim`), yet v1
/// reads it as a 6-damage Attack that kills a 5-HP enemy. v5 flags it, and
/// after a Bodyguard summons Osty the same Poke is a real, lethal hit.
#[test]
fn an_osty_attack_without_osty_is_flagged_and_a_summon_fixes_it() {
    let mut s = ironclad(&[CardId::Poke, CardId::Bodyguard], 5, 3);
    let f = v5(&s);
    assert!(close(f.state[V5_OSTY], 0.0));
    let poke = x_of(&f, CardId::Poke);
    assert!(close(poke[19], 1.0), "v1's lethal flag believes the Poke");
    assert!(close(poke[V5_X_OSTY_ATTACK], 1.0));
    assert!(close(poke[V5_X_OSTY_MISSING], 1.0));
    assert!(close(poke[V5_X_OSTY_LETHAL], 0.0));
    let guard = x_of(&f, CardId::Bodyguard);
    assert!(close(guard[V5_X_SUMMON], 0.5), "Bodyguard summons 5");
    assert!(close(guard[V5_X_OSTY_MISSING], 0.0));

    play(&mut s, CardId::Bodyguard);
    let f = v5(&s);
    assert!(close(f.state[V5_OSTY], 1.0), "Osty is up");
    assert!(close(f.state[V5_OSTY + 1], 5.0 / 20.0));
    let poke = x_of(&f, CardId::Poke);
    assert!(close(poke[V5_X_OSTY_MISSING], 0.0));
    assert!(
        close(poke[V5_X_OSTY_LETHAL], 1.0),
        "6 Osty damage kills 5 HP"
    );
}

/// Tempest channels X Lightning and Chill one Frost per enemy; both are read
/// at the board's X and enemy count. Dualcast is a payoff with nothing to
/// evoke until something is channeled.
#[test]
fn variable_channels_and_payoffs_read_the_board() {
    let mut state = ironclad(
        &[
            CardId::Tempest,
            CardId::Chill,
            CardId::Dualcast,
            CardId::Zap,
        ],
        40,
        3,
    );
    let feats = v5(&state);
    let tempest = x_of(&feats, CardId::Tempest);
    assert!(close(tempest[V5_X_CHANNELS], 1.0), "X = 3 energy");
    assert!(close(tempest[V5_X_CHANNEL_KINDS], 1.0));
    assert!(
        close(tempest[V5_X_OVERFLOW], 2.0 / 3.0),
        "one slot, three Lightning"
    );
    let chill = x_of(&feats, CardId::Chill);
    assert!(
        close(chill[V5_X_CHANNEL_KINDS + 1], 1.0 / 3.0),
        "one enemy, one Frost"
    );
    let dual = x_of(&feats, CardId::Dualcast);
    assert!(close(dual[V5_X_ORB_PAYOFF], 1.0));
    assert!(
        close(dual[V5_X_ORB_PAYOFF + 1], 0.0),
        "nothing to evoke yet"
    );
    assert!(close(dual[V5_X_CHANNELS], 0.0));

    play(&mut state, CardId::Zap);
    let dual = x_of(&v5(&state), CardId::Dualcast);
    assert!(
        close(dual[V5_X_ORB_PAYOFF + 1], 1.0 / 3.0),
        "one orb to evoke"
    );
}

/// A v1 net, and a v4 one, with zero weights on every v5 input play the
/// same policy: what `rollout_learn rl --feature-version 5` relies on to start
/// from a shipped net. Checked on the shipped Ironclad net, on a hand of
/// other characters' cards.
#[test]
fn zero_padded_nets_keep_their_logits_under_v5() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/rollout_nets/ironclad.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: no {}", path.display());
        return;
    };
    let v1: serde_json::Value = serde_json::from_str(&text).expect("json");
    let pad = |net: &serde_json::Value, version: u32, sd: usize, ad: usize| {
        let mut out = net.clone();
        out["feature_version"] = version.into();
        out["state_dim"] = sd.into();
        out["action_dim"] = ad.into();
        let rows = |key: &str, to: usize| -> serde_json::Value {
            net[key]
                .as_array()
                .expect("rows")
                .iter()
                .map(|r| {
                    let mut r: Vec<f64> = serde_json::from_value(r.clone()).expect("row");
                    r.resize(to, 0.0);
                    r
                })
                .collect::<Vec<_>>()
                .into()
        };
        out["w1"] = rows("w1", sd);
        out["proj"] = rows("proj", ad);
        let mut lin: Vec<f64> = serde_json::from_value(net["lin"].clone()).expect("lin");
        lin.resize(ad, 0.0);
        out["lin"] = lin.into();
        out
    };
    let v4 = pad(&v1, 4, STATE_DIM_V4, ACTION_DIM_V4);
    let v5_from_v1 = pad(&v1, 5, STATE_DIM_V5, ACTION_DIM_V5);
    let v5_from_v4 = pad(&v4, 5, STATE_DIM_V5, ACTION_DIM_V5);
    let load = |v: &serde_json::Value| RolloutNet::from_json(&v.to_string()).expect("loads");
    let nets = [load(&v1), load(&v4), load(&v5_from_v1), load(&v5_from_v4)];
    let mut s = ironclad(
        &[
            CardId::Zap,
            CardId::Poke,
            CardId::Bodyguard,
            CardId::Tempest,
            CardId::StrikeIronclad,
        ],
        30,
        3,
    );
    for step in 0..3 {
        let acts = legal_actions(&s);
        let base = nets[0].logits(&s, &acts, END_TURN_PRIORITY);
        for net in &nets[1..] {
            let l = net.logits(&s, &acts, END_TURN_PRIORITY);
            for (a, b) in base.iter().zip(&l) {
                assert!((a - b).abs() < 1e-5, "step {step}: {base:?} vs {l:?}");
            }
        }
        // Grow a belt and an Osty so the v5 inputs are non-zero.
        let next = [CardId::Zap, CardId::Bodyguard, CardId::Poke][step];
        play(&mut s, next);
    }
    assert!(s.player.orbs.filled_len() > 0 && s.living_own_osty());
}
