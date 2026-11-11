//! A captured wire dict goes from JSON text to a live `CombatState` with no
//! Python interpreter anywhere in the process.
//!
//! This is the property the crate exists for. Before the split, reaching
//! `CombatState` from a wire dict meant linking `libpython` and holding the
//! GIL, so nothing outside the `PyO3` extension could replay a capture. The
//! test crate here depends only on `sts2-wire` — if a `pyo3` symbol ever
//! creeps back into the parse path, this stops linking.

use sts2_wire::{parse_json, state_from_json_str, WireError};

fn fixture() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture readable")
}

#[test]
fn a_captured_dict_parses_without_python() {
    let text = fixture();
    let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    let parsed = parse_json(&value).expect("fixture must parse");

    assert_eq!(parsed.wire_version, 1);
    assert_eq!(parsed.player.class, "ironclad");
    assert_eq!(parsed.player.hp, 54);
    assert_eq!(parsed.player.relics.len(), 2);
    assert_eq!(parsed.player.gold, Some(137));
    assert_eq!(parsed.deck.hand.len(), 2);
    assert_eq!(parsed.deck.draw_pile.len(), 3);
    assert_eq!(parsed.deck.pinned_top, vec![1]);
    assert_eq!(parsed.encounter.len(), 2);
    assert_eq!(parsed.ascension, 7);

    // A `u64` seed above `i64::MAX` — the case a naive `as_i64` read would
    // have silently rejected.
    assert_eq!(parsed.rng_seed, Some(18_446_744_073_709_551_557));

    // Key-sorted, per the `WireValue::entries` contract.
    // Sorted, though the fixture lists them shuffle / monster_ai /
    // combat_targets.
    let counter_names: Vec<&str> = parsed
        .rng_counters
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(counter_names, ["combat_targets", "monster_ai", "shuffle"]);
}

#[test]
fn a_captured_dict_rebuilds_a_combat_state() {
    let state = state_from_json_str(&fixture()).expect("fixture must rebuild");

    assert_eq!(state.player.current_hp, 54);
    assert_eq!(state.player.max_hp, 80);
    assert_eq!(state.player.block, 5);
    // Two enemies on the wire, one of them dead.
    assert_eq!(state.enemy_uids.len(), 2);
    // The piles came across, not just the scalars.
    assert_eq!(state.player.hand.len(), 2);
    assert_eq!(state.player.draw.len(), 3);
    assert_eq!(state.player.discard.len(), 1);
}

/// A wire `cost_mods` block rebuilds the card's exact cost model: the
/// fixture's hand Strike was made free this turn (`SetToFreeThisTurn`, an
/// Absolute 0 that expires at turn end or on play), so it reads 0 now and
/// its printed 1 once the modifier expires.
#[test]
fn wire_cost_mods_rebuild_the_cost_model() {
    use sts2_core::cards::CardId;

    let state = state_from_json_str(&fixture()).expect("fixture must rebuild");
    let strike = state
        .player
        .hand
        .iter()
        .map(|uid| &state.cards[uid.get() as usize - 1])
        .find(|c| c.card_id == CardId::StrikeIronclad)
        .expect("the hand Strike came across");
    assert!(strike.cost.has(sts2_core::LocalCostMod::FREE_THIS_TURN));
    assert_eq!(strike.base_cost(), 1);
    assert_eq!(strike.local_cost(), 0);
    let mut later = *strike;
    later.cost.end_of_turn_cleanup();
    assert_eq!(later.local_cost(), 1);
}

/// The synthetic `status:` / `curse:` ids resolve to their real cards rather
/// than degrading to the AscendersBane placeholder — the single largest
/// reconstruction gap the DLL path ever had, and worth pinning on the
/// interpreter-free route too.
#[test]
fn synthetic_status_and_curse_ids_resolve() {
    let value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    let parsed = parse_json(&value).unwrap();

    let slimed = parsed
        .deck
        .draw_pile
        .iter()
        .find(|c| c.raw_id == "status:Slimed")
        .expect("slimed in draw pile");
    assert!(
        slimed.card_id.is_some(),
        "`status:Slimed` must resolve to a real CardId, not fall through to the placeholder"
    );

    let clumsy = parsed
        .deck
        .hand
        .iter()
        .find(|c| c.raw_id == "curse:Clumsy")
        .expect("clumsy in hand");
    assert!(clumsy.card_id.is_some(), "`curse:Clumsy` must resolve");
}

#[test]
fn failures_carry_their_path_and_class() {
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();

    // Unmodelled content → Unsupported.
    value["player"]["relics"][1]["id"] = serde_json::json!("DEFINITELY_NOT_A_RELIC");
    let err = parse_json(&value).unwrap_err();
    assert!(matches!(err, WireError::Unsupported { .. }), "got {err:?}");
    assert_eq!(err.path(), "$.player.relics[1]");
    assert!(err.to_string().contains("DEFINITELY_NOT_A_RELIC"));

    // Invariant violation → Schema.
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    value["player"]["hp"] = serde_json::json!(999);
    let err = parse_json(&value).unwrap_err();
    assert!(matches!(err, WireError::Schema { .. }), "got {err:?}");
    assert_eq!(err.path(), "$.player");

    // Missing required key → Schema, at the parent path.
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    value.as_object_mut().unwrap().remove("phase");
    let err = parse_json(&value).unwrap_err();
    assert_eq!(err.path(), "$");
    assert!(err.to_string().contains("missing required key `phase`"));
}

#[test]
fn malformed_json_text_fails_at_the_root() {
    let err = state_from_json_str("{not json").unwrap_err();
    assert_eq!(err.path(), "$");
    assert!(err.to_string().contains("invalid JSON"));
}

/// GH #787: `$.game_version` is optional — absent means the default build —
/// and an explicit one reaches `CombatState::game_version`.
#[test]
fn game_version_key_is_optional_and_threaded() {
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
    assert!(
        value.get("game_version").is_none(),
        "fixture predates the key"
    );
    let default = sts2_wire::state_from_json(&value).expect("default rebuild");
    assert_eq!(default.game_version, sts2_core::GameVersion::V0_107_1);

    value["game_version"] = serde_json::json!("0.111.0");
    let v111 = sts2_wire::state_from_json(&value).expect("v0.111.0 rebuild");
    assert_eq!(v111.game_version, sts2_core::GameVersion::V0_111_0);

    value["game_version"] = serde_json::json!("v9.9.9");
    assert!(sts2_wire::state_from_json(&value).is_err(), "unknown build");
}

/// The emitter writes `game_version` only for a non-default build, so every
/// v0.107.1 wire keeps its historical key set, and the key round-trips.
#[test]
fn game_version_is_emitted_only_when_non_default() {
    let mut state = state_from_json_str(&fixture()).expect("fixture must rebuild");
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert!(wire.get("game_version").is_none());

    state.game_version = sts2_core::GameVersion::V0_111_0;
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert_eq!(wire["game_version"], serde_json::json!("v0.111.0"));
    let back = sts2_wire::state_from_json(&wire).expect("round trip");
    assert_eq!(back.game_version, sts2_core::GameVersion::V0_111_0);
}

/// GH #788: `$.rng.run_seed` is a `uint` for v0.107.1 and a `ulong` for
/// v0.111.0; the reconstruction seeds its streams in the wire's build.
#[test]
fn run_seed_width_follows_the_game_version() {
    let wide: u64 = 0xCCD0_8772_EAB0_03D2;
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
    value["rng"]["run_seed"] = serde_json::json!(wide);
    value["rng"]["counters"] = serde_json::json!({});
    let err = sts2_wire::state_from_json(&value).unwrap_err();
    assert_eq!(err.path(), "$.rng.run_seed", "v0.107.1 seeds are 32-bit");

    value["game_version"] = serde_json::json!("v0.111.0");
    let state = sts2_wire::state_from_json(&value).expect("v0.111.0 accepts a u64 seed");
    assert_eq!(state.run_rng.seed, wide);
    assert_eq!(state.run_rng.game_version, sts2_core::GameVersion::V0_111_0);
    assert_eq!(
        state.run_rng,
        sts2_core::RunRngSet::new_for(sts2_core::GameVersion::V0_111_0, wide)
    );
    // ...and the emitter writes the full width back out.
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert_eq!(wire["rng"]["run_seed"], serde_json::json!(wide));
}

/// The scenario builder threads the version like ascension, and the legacy
/// entry point is exactly the default build.
#[test]
fn scenario_builder_threads_game_version() {
    let deck = vec!["STRIKE_IRONCLAD".to_string()];
    let build = |v| {
        sts2_wire::scenario::build_scenario_state_for_version(
            v,
            3,
            "Ironclad",
            &deck,
            50,
            80,
            &[],
            &[],
            "SHRINKER_BEETLE_WEAK",
            0,
            None,
            None,
        )
        .expect("scenario builds")
    };
    assert_eq!(
        build(sts2_core::GameVersion::V0_111_0).game_version,
        sts2_core::GameVersion::V0_111_0
    );
    let legacy = sts2_wire::scenario::build_scenario_state(
        3,
        "Ironclad",
        &deck,
        50,
        80,
        &[],
        &[],
        "SHRINKER_BEETLE_WEAK",
        0,
        None,
        None,
    )
    .expect("scenario builds");
    assert_eq!(legacy, build(sts2_core::GameVersion::V0_107_1));
}

/// GH #929: a v0.111.0 wire whose hand holds a v0.111.0-only card (the
/// DowsingRod's `DOWSING` quest card, or any of the build's new cards) must
/// rebuild. The id lookup used to walk only the v0.107.1 universe, so every
/// live advice request with such a card in a pile failed `UnsupportedStateError`.
#[test]
fn v0111_only_cards_parse_from_a_v0111_wire() {
    use sts2_core::cards::CardId;
    for id in CardId::V0111_ONLY {
        let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
        value["game_version"] = serde_json::json!("v0.111.0");
        value["deck"]["hand"][0] = serde_json::json!({
            "id": id.to_python_name(),
            "stable_idx": 4,
            "kind": "card",
            "upgraded": false,
        });
        let state = sts2_wire::state_from_json(&value)
            .unwrap_or_else(|e| panic!("{}: {e:?}", id.to_python_name()));
        let first = state.player.hand[0];
        assert_eq!(state.cards[first.0.get() as usize - 1].card_id, id);
    }
}

/// v0.111.0 Haze lost its intrinsic Sly (`Haze.cs:19`, no
/// `CanonicalKeywords`), so whether a wire `sly` flag on a Haze is the
/// card's own keyword or MasterPlanner's combat-long grant depends on the
/// build. The decoder and the emitter must both ask
/// `card_sly_in(id, game_version)`: a version-blind decoder dropped the grant
/// at v0.111.0, and a version-blind emitter wrote `sly` on every v0.111.0
/// Haze, which the fixed decoder would then read back as a grant.
#[test]
fn haze_sly_round_trips_in_both_builds() {
    for (version, intrinsic) in [("0.107.1", true), ("0.111.0", false)] {
        for granted in [false, true] {
            let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
            value["game_version"] = serde_json::json!(version);
            let flags = if granted || intrinsic {
                serde_json::json!(["sly"])
            } else {
                serde_json::json!([])
            };
            value["deck"]["hand"] = serde_json::json!([{
                "id": "HAZE", "stable_idx": 0, "kind": "card", "upgraded": false, "flags": flags
            }]);
            let state = state_from_json_str(&value.to_string()).expect("rebuilds");
            let uid = state.player.hand[0];
            let label = format!("{version} granted={granted}");
            assert_eq!(
                sts2_core::card_effects::card_is_sly(&state, uid),
                intrinsic || granted,
                "{label}"
            );
            assert_eq!(
                state.player.sly_this_combat.contains(&uid),
                granted && !intrinsic,
                "{label}: only a grant is combat-scoped"
            );
            let wire = sts2_wire::emit::combat_state_to_value(&state).expect("emits");
            let has_sly = wire["deck"]["hand"][0]["flags"]
                .as_array()
                .is_some_and(|f| f.iter().any(|x| x == "sly"));
            assert_eq!(has_sly, intrinsic || granted, "{label}: emitted flag");
            let back = state_from_json_str(&wire.to_string()).expect("rebuilds again");
            assert_eq!(
                sts2_core::card_effects::card_is_sly(&back, back.player.hand[0]),
                intrinsic || granted,
                "{label}: round trip"
            );
        }
    }
}

/// MasterPlanner's `AddKeyword(Sly)` lasts the combat and stays on the card
/// in every pile (`MasterPlannerPower.cs`), so a `sly` flag on a pile card is
/// read too, not only in hand: a rebuilt granted-Sly Defend in the draw pile
/// is still Sly once drawn. Ported from manoj fork 841f558f3 (decoder half).
#[test]
fn granted_sly_is_read_from_every_pile() {
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    value["deck"]["draw_pile"][1]["flags"] = serde_json::json!(["sly"]);
    let state = state_from_json_str(&value.to_string()).expect("rebuilds");
    let card = state.player.draw[1];
    assert!(state.player.sly_this_combat.contains(&card));
    assert!(!state.player.sly_this_combat.contains(&state.player.draw[0]));
}

/// A small fight: a 100-hp Nibbit, `hand` (card ids) in hand, three Strikes to
/// draw, 3 energy.
fn small_fight(hand: &[&str]) -> serde_json::Value {
    let card = |id: &str, i: usize| serde_json::json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false});
    serde_json::json!({
        "wire_version": 1, "phase": "PlayerPlayPhase", "round": 1, "side": "Player",
        "is_elite": false, "turn": 1, "combat_id": 0, "rng": {"seed": 1, "ascension": 0},
        "player": {"class": "ironclad", "hp": 80, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3, "energy_regen_per_turn": 3, "gold": 0, "relics": [],
                   "powers": [], "potions": []},
        "deck": {"hand": hand.iter().enumerate().map(|(i, id)| card(id, i + 1)).collect::<Vec<_>>(),
                 "draw_pile": (0..3).map(|i| card("STRIKE_IRONCLAD", 10 + i)).collect::<Vec<_>>(),
                 "discard_pile": [], "exhaust_pile": []},
        "encounter": [{"kind": "Nibbit", "uid": 1, "hp": 100, "max_hp": 100, "block": 0,
                       "powers": [], "intents": [{"kind": "Attack", "amount": 6, "times": 1}],
                       "intent_history": [], "is_dead": false, "is_escaped": false}],
    })
}

/// C# `SoulsPower.OnEnchant` removes the card's Exhaust keyword
/// (`Models.Enchantments/SoulsPower.cs`; `CanEnchant` requires a local
/// Exhaust): a Soul's Power Molten Fist goes to the discard pile, a plain one
/// is exhausted. Ported from manoj fork ef47c32ec.
#[test]
fn a_souls_power_card_does_not_exhaust() {
    sts2_content::cards::dispatch::ensure_installed();
    for (enchanted, discarded, exhausted) in [(true, 1, 0), (false, 0, 1)] {
        let mut value = small_fight(&["MOLTEN_FIST"]);
        if enchanted {
            value["deck"]["hand"][0]["enchantment"] = serde_json::json!("SOULS_POWER");
            value["deck"]["hand"][0]["enchantment_amount"] = serde_json::json!(1);
        }
        let mut state = state_from_json_str(&value.to_string()).expect("rebuilds");
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::PlayCard {
                hand_index: 0,
                target: Some(0),
            },
        );
        assert_eq!(
            (state.player.discard.len(), state.player.exhaust.len()),
            (discarded, exhausted),
            "Soul's Power: {enchanted}"
        );
    }
}

/// `OvicopterNormal.Slots = [egg1..egg5, ovicopter]`: a rebuilt Ovicopter
/// holds slot 5 and the listed living eggs the top egg slots, so its next
/// lay puts the new eggs before it, as the game's slot-sorted enemy list
/// does. Ported from manoj fork ef47c32ec.
#[test]
fn an_ovicopter_fight_rebuilds_with_its_egg_slots() {
    sts2_content::cards::dispatch::ensure_installed();
    let mut value = small_fight(&["STRIKE_IRONCLAD"]);
    let enemy = |kind: &str, uid: i32, mv: &str| {
        serde_json::json!({"kind": kind, "uid": uid, "hp": 20, "max_hp": 20, "block": 0,
            "powers": [], "intents": [], "intent_history": [], "move_state": mv,
            "is_dead": false, "is_escaped": false})
    };
    value["encounter"] = serde_json::json!([
        enemy("ToughEgg", 1, "HATCH_MOVE"),
        enemy("ToughEgg", 2, "HATCH_MOVE"),
        enemy("Ovicopter", 3, "SMASH_MOVE")
    ]);
    let state = state_from_json_str(&value.to_string()).expect("rebuilds");
    let slots: Vec<Option<u8>> = state
        .enemy_uids
        .iter()
        .map(|uid| state.enemy_slot(*uid))
        .collect();
    assert_eq!(slots, [Some(3), Some(4), Some(5)]);
}
