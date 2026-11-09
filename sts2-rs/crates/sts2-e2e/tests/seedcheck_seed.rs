//! Runs from a game seed string (the workbench seed checker): the seed hashes
//! the way the v0.111.0 game does, so a pasted seed is the real run.
//!
//! The Neow offers and bosses below were read from the Python engine
//! (`sts2_rs.RunManager(string_seed=...)`) and the C# oracle on 2026-11-08.

use serde_json::Value;
use sts2_e2e::env::{canonical_game_seed, new_run_from_game_seed, E2eEnv, Step};
use sts2_wire::emit::run_manager as emit_rm;

fn neow(seed: &str) -> (String, String, String) {
    let rm = new_run_from_game_seed(seed, 10).expect("valid seed");
    emit_rm::neow_offers(&rm).expect("a Neow start")
}

fn bosses(seed: &str) -> Vec<(String, Option<String>)> {
    let rm = new_run_from_game_seed(seed, 10).expect("valid seed");
    emit_rm::generated_rooms(&rm)
        .as_array()
        .expect("a list")
        .iter()
        .map(|r| {
            (
                r["boss"].as_str().unwrap_or_default().to_string(),
                r["second_boss"].as_str().map(str::to_string),
            )
        })
        .collect()
}

fn owned(a: &str, b: &str, c: &str) -> (String, String, String) {
    (a.into(), b.into(), c.into())
}

#[test]
fn game_seed_neow_and_bosses() {
    assert_eq!(
        neow("1SNVTCEXKR0Q"),
        owned("ARCANE_SCROLL", "LAVA_ROCK", "NEOWS_BONES")
    );
    assert_eq!(
        bosses("1SNVTCEXKR0Q"),
        vec![
            ("VANTOM_BOSS".to_string(), None),
            ("THE_INSATIABLE_BOSS".to_string(), None),
            ("AEONGLASS_BOSS".to_string(), Some("QUEEN_BOSS".to_string())),
        ]
    );
    assert_eq!(
        neow("1SNVTCEXKR0G"),
        owned("SCROLL_BOXES", "BOOMING_CONCH", "LEAFY_POULTICE")
    );
    assert_eq!(bosses("1SNVTCEXKR0G")[0].0, "LAGAVULIN_MATRIARCH_BOSS");
}

#[test]
fn game_seed_is_canonicalised() {
    let want = emit_rm::generated_rooms(&new_run_from_game_seed("1SNVTCEXKR0Q", 10).unwrap());
    for typed in ["1snvtcexkr0q ", "1SNVTCEXKRoQ", "ISNVTCEXKR0Q"] {
        assert_eq!(canonical_game_seed(typed).unwrap(), "1SNVTCEXKR0Q");
        let got = emit_rm::generated_rooms(&new_run_from_game_seed(typed, 10).unwrap());
        assert_eq!(got, want, "{typed:?}");
    }
}

#[test]
fn game_seed_rejects_bad_chars() {
    assert!(canonical_game_seed("").is_err());
    assert!(canonical_game_seed("   ").is_err());
    let e = canonical_game_seed("AB-C").unwrap_err();
    assert!(e.contains('-') && e.contains("alphabet"), "{e}");
    assert!(new_run_from_game_seed("AB-C", 10).is_err());
}

#[test]
fn python_proof_replays_to_a_win() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/seedcheck_1SNVTCEXKR0Q_proof.json"
    );
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let seed = doc["seed"].as_str().expect("seed");
    let log: Vec<usize> = doc["log_idx"]
        .as_array()
        .expect("log_idx")
        .iter()
        .map(|v| usize::try_from(v.as_u64().expect("index")).expect("fits"))
        .collect();
    let mut env = E2eEnv::over_run(new_run_from_game_seed(seed, 10).unwrap(), 0);
    let mut step = env.reset();
    for (n, &ai) in log.iter().enumerate() {
        match step {
            Step::Decision(_) => step = env.step_action(ai),
            Step::Terminal(t) => panic!("run ended at step {n} of {}: {}", log.len(), t.outcome),
        }
    }
    match step {
        Step::Terminal(t) => assert!(t.won, "outcome {}", t.outcome),
        Step::Decision(_) => panic!("the log ended before the run did"),
    }
}
