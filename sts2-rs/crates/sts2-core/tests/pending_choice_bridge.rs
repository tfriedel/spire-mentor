//! Pending-choice bridge fixtures (bead sts2-rs-ck2).
//!
//! Loads JSON test vectors under `tests/parity/scripted/pending_choice/`
//! and asserts that `sts2_core::parity::pending_choice_to_bridge_value`
//! produces the documented dict shape per `bridge_replay.py:303-316`,
//! with the Rust-port `kind` discriminant added.
//!
//! These fixtures stand in for true Python↔Rust parity fixtures: at
//! pin `01c094d` no production code constructs a [`PendingChoice`]
//! instance (sister bead `sts2-rs-cqm` was supposed to wire Headbutt /
//! Discovery / Exhume but was auto-closed without an implementation —
//! see follow-up bead filed under `.pi-flywheel/`). Once the wiring
//! lands the fixture format can be migrated into the main
//! `parity_harness.rs` loop.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sts2_core::cards::{Card, CardStore, CardTags, CardType, PileKind, TargetType};
use sts2_core::{cards::CardId, parity, ChoiceKind, ChoiceOption, PendingChoice};

fn fixtures_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(|root| {
            root.join("tests")
                .join("parity")
                .join("scripted")
                .join("pending_choice")
        })
        .expect("workspace root must be ancestor of CARGO_MANIFEST_DIR")
}

fn parse_kind(s: &str) -> ChoiceKind {
    match s {
        "CardSelect" => ChoiceKind::CardSelect,
        "PutBack" => ChoiceKind::PutBack,
        "ExhaustPile" => ChoiceKind::ExhaustPile,
        "OrbSlot" => ChoiceKind::OrbSlot,
        "EnemyTarget" => ChoiceKind::EnemyTarget,
        other => panic!("unknown ChoiceKind in fixture: {other:?}"),
    }
}

fn parse_card_id(s: &str) -> CardId {
    // The fixtures use Rust enum-variant names (e.g. "Bash") — match a
    // small set sufficient for the 3 fixtures here. Extend as new
    // fixtures land.
    match s {
        "Bash" => CardId::Bash,
        "StrikeIronclad" => CardId::StrikeIronclad,
        "DefendIronclad" => CardId::DefendIronclad,
        "Inflame" => CardId::Inflame,
        "Headbutt" => CardId::Headbutt,
        "PerfectedStrike" => CardId::PerfectedStrike,
        other => panic!("unknown CardId in fixture: {other:?}"),
    }
}

/// Build a minimal `Card` arena that resolves each option's `card_uid`
/// to a concrete runtime card with the right `card_type` / `upgraded`
/// flags. Maps the option index → uid `idx+1` so the bridge helper
/// finds them via the `card_uid` lookup path.
fn build_arena(options_json: &[Value]) -> (CardStore, Vec<ChoiceOption>) {
    let mut arena: CardStore = Vec::with_capacity(options_json.len()).into();
    let mut opts: Vec<ChoiceOption> = Vec::with_capacity(options_json.len());
    for (idx, opt) in options_json.iter().enumerate() {
        let card_id = parse_card_id(opt["card_id"].as_str().expect("card_id is string"));
        let upgraded = opt["upgraded"].as_bool().unwrap_or(false);
        let raw = u32::try_from(idx + 1).expect("small idx");
        let uid = sts2_core::CardUid::new(raw).expect("non-zero");
        let card_type = match card_id {
            CardId::Bash | CardId::StrikeIronclad | CardId::Headbutt | CardId::PerfectedStrike => {
                CardType::Attack
            }
            CardId::DefendIronclad => CardType::Skill,
            CardId::Inflame => CardType::Power,
            _ => CardType::Skill,
        };
        arena.push(Card {
            uid,
            card_id,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type,
            target_type: TargetType::None,
            upgraded,
            base_damage: 0,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: CardTags::default(),
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });
        opts.push(ChoiceOption {
            card_id,
            upgraded,
            card_uid: Some(uid),
            source_pile: Some(PileKind::Discard),
        });
    }
    (arena, opts)
}

/// Variant of [`build_arena`] that returns synthetic options (no
/// `card_uid`), which is the shape used by discovery / orb-slot flavors
/// where the choice is over pre-materialization candidates.
fn build_synthetic_options(options_json: &[Value]) -> Vec<ChoiceOption> {
    options_json
        .iter()
        .map(|opt| {
            let card_id = parse_card_id(opt["card_id"].as_str().expect("card_id is string"));
            ChoiceOption {
                card_id,
                upgraded: opt["upgraded"].as_bool().unwrap_or(false),
                card_uid: None,
                source_pile: None,
            }
        })
        .collect()
}

fn run_fixture(path: &Path, synthetic: bool) {
    let raw = fs::read_to_string(path).expect("fixture readable");
    let fixture: Value = serde_json::from_str(&raw).expect("fixture parses as JSON");
    let pc_json = &fixture["pending_choice"];
    let options_json = pc_json["options"].as_array().expect("options is array");

    let (cards_arena, options) = if synthetic {
        (CardStore::new(), build_synthetic_options(options_json))
    } else {
        build_arena(options_json)
    };

    let pc = PendingChoice {
        kind: parse_kind(pc_json["kind"].as_str().expect("kind is string")),
        options,
        min_select: pc_json["min_select"].as_u64().unwrap_or(1) as u8,
        max_select: pc_json["max_select"].as_u64().unwrap_or(1) as u8,
        can_skip: pc_json["can_skip"].as_bool().unwrap_or(false),
        post_effect: None,
        picked: smallvec::SmallVec::new(),
    };

    let actual = parity::pending_choice_to_bridge_value(&pc, &cards_arena);
    let expected = &fixture["expected_bridge_value"];
    assert_eq!(
        &actual,
        expected,
        "fixture {} bridge value mismatch:\n  actual: {}\n  expected: {}",
        path.display(),
        serde_json::to_string_pretty(&actual).unwrap(),
        serde_json::to_string_pretty(expected).unwrap()
    );
}

#[test]
fn headbutt_putback_bridge_shape() {
    // Headbutt: real-card options pulled from the discard pile, so
    // the arena resolution path is exercised.
    run_fixture(&fixtures_dir().join("headbutt_putback.json"), false);
}

#[test]
fn discovery_pick_bridge_shape() {
    // Discovery: synthetic options (no card_uid) — exercises the
    // `type`-omission fallback in `choice_option_to_bridge_value`.
    run_fixture(&fixtures_dir().join("discovery_pick.json"), true);
}

#[test]
fn exhume_pick_bridge_shape() {
    // ExhaustPile flavor with `can_skip: true` — verifies the
    // can_skip pass-through.
    run_fixture(&fixtures_dir().join("exhume_pick.json"), false);
}
