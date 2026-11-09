//! `event_option_infos_with(rm, masked)` (#1181): the opt-in masked view of
//! the event-option previews. The raw view must not move (the deployed PPO
//! champions read it); the masked view hides the ROLLED relic's pickup that
//! the raw preview folds into an option's gains.

use serde_json::Value;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::{EventId, GameVersion};
use sts2_wire::emit::run_manager as emit;
use sts2_wire::graft;

const MASKED_GAINS: [&str; 5] = [
    "gold_delta",
    "hp_delta",
    "max_hp_delta",
    "potions_gained",
    "cards_gained",
];

fn new_run(seed: u32) -> RunManager {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        seed as i32,
        CharacterId::Ironclad,
        0,
    )
    .with_game_version(GameVersion::V0_111_0)
    .with_run_seed(u64::from(seed))
    .with_neow_start();
    rm.run_state.set_agent_decisions();
    let mut rm = rm.with_rolled_act1();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    rm
}

fn entry(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

/// Checks one event screen; returns how many relic options the mask changed.
fn check(rm: &RunManager, raw_before: &Value) -> usize {
    let masked = emit::event_option_infos_with(rm, true);
    let raw = emit::event_option_infos(rm);
    assert_eq!(
        &raw, raw_before,
        "reading the masked view moved the raw one"
    );
    assert_eq!(emit::event_option_infos_with(rm, false), raw);
    let (Some(raw), Some(masked)) = (raw.as_array(), masked.as_array()) else {
        assert_eq!(raw, masked);
        return 0;
    };
    assert_eq!(raw.len(), masked.len());
    let mut changed = 0;
    for (r, m) in raw.iter().zip(masked) {
        let gains_relic = r.get("relics_gained").and_then(Value::as_i64).unwrap_or(0) > 0;
        if !gains_relic {
            assert_eq!(r, m, "an option without a relic gain is unchanged");
            continue;
        }
        let (r, m) = (r.as_object().unwrap(), m.as_object().unwrap());
        assert_eq!(r.len(), m.len(), "same keys");
        for (k, v) in r {
            let mv = &m[k];
            if MASKED_GAINS.contains(&k.as_str()) {
                assert_eq!(mv, 0, "{k} is masked");
            } else if k == "relics_gained" {
                assert_eq!(mv, 1);
            } else {
                assert_eq!(mv, v, "{k} is kept");
            }
        }
        if r != m {
            changed += 1;
        }
    }
    changed
}

#[test]
fn masked_view_hides_relic_pickups_and_leaves_raw_alone() {
    let mut relic_options_masked = 0;
    let mut screens = 0;
    for seed in 1..=6_u32 {
        // Neow: the run opens on it.
        let mut rm = new_run(seed);
        rm.bind_pending_event();
        if rm.phase() == Phase::Event {
            let raw = emit::event_option_infos(&rm);
            relic_options_masked += check(&rm, &raw);
            screens += 1;
        }
        for &ev in EventId::all() {
            if ev == EventId::Neow {
                continue;
            }
            let mut rm = new_run(seed);
            if graft::enter_event(&mut rm, &entry(ev.python_name())).is_err() {
                continue;
            }
            let raw = emit::event_option_infos(&rm);
            relic_options_masked += check(&rm, &raw);
            screens += 1;
        }
    }
    assert!(screens > 50, "only {screens} event screens checked");
    // The leak the mask exists for: some relic option's raw preview carries
    // its rolled relic's pickup (Neow's gold / HP relics, at least).
    assert!(
        relic_options_masked > 0,
        "no relic option's preview carried a pickup gain; the test no longer exercises the mask"
    );
}
