//! Parity-trace serialization. Output shape mirrors
//! `sts2_env.parity.bridge_replay.combat_state_to_bridge_state` from
//! sts2-rl-agent@01c094d. Field order matches Python dict insertion order
//! per `normalize_bridge_state` (line 243).
//!
//! Body fully populated through P1–P8 (per-enemy hp/intent/powers,
//! per-card metadata, player powers with the `_normalize_powers`
//! zero-amount filter). Determinism rule (v2.1 implementation rule
//! 11): never iterate `HashMap` — use `BTreeMap` or sorted keys.
//!
//! Note on key ordering: Python's `save_replay_trace` writes traces with
//! `json.dumps(..., sort_keys=True)`, so the on-disk JSON is alphabetical.
//! `serde_json::Map` defaults to `BTreeMap`, which also serializes keys
//! alphabetically — meaning the bytes-on-disk output of this exporter is
//! byte-identical to Python's, even though the in-memory dict insertion
//! order is the canonical sequence shown in `normalize_bridge_state`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::cards::{Card, CardType, TargetType};
use crate::{ChoiceOption, CombatState, Intent, PendingChoice, PowerKind};

/// Top-level replay envelope. Mirrors Python's `BridgeReplayTrace`
/// (`bridge_replay.py:71-104`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Schema version. Always `1` at this pin.
    pub version: u32,
    /// Replay mode. `"combat"` is the only supported mode in Phase 0.
    pub mode: String,
    /// Free-form metadata. Use `BTreeMap` for deterministic key order.
    pub metadata: BTreeMap<String, Value>,
    /// Initial bridge state captured at recorder start.
    pub initial_state: Value,
    /// Per-action steps (`action`, `resulting_state`).
    pub steps: Vec<TraceStep>,
}

impl Default for Trace {
    fn default() -> Self {
        Self {
            version: 1,
            mode: "combat".to_string(),
            metadata: BTreeMap::new(),
            initial_state: Value::Object(serde_json::Map::new()),
            steps: Vec::new(),
        }
    }
}

/// One recorded `(action, resulting_state)` pair. Mirrors
/// `BridgeReplayStep` (`bridge_replay.py:63-68`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// The action sent to the bridge (or `null` if not recorded).
    pub action: Value,
    /// The bridge state observed after the action was applied.
    pub resulting_state: Value,
}

/// Serialize a [`CombatState`] into the bridge replay shape. Matches
/// `combat_state_to_bridge_state` (`bridge_replay.py:301-374`).
///
/// Emits the full bridge state: player scalars (hp/block/energy/round),
/// piles, card metadata, and per-enemy `id`/`hp`/`max_hp`/`block`/
/// `is_alive`/`powers`/optional intent fields (conditional on
/// `creature.intent.is_some()` per Python's `if ai is not None` gate).
/// Player + enemy powers apply the `_normalize_powers` zero-amount
/// filter and sort by `(id, amount)` for byte-equality with Python.
///
/// Bead history: `sts2-rs-cy8` (VTHIN-1) wired player scalars + round;
/// `sts2-rs-5if` (VTHIN-2) enriched `Creature` with hp/block/powers;
/// `sts2-rs-pg5` (P4-3) replaced the hard-coded `"Cultist"` id with
/// `creature.kind.python_name()` and added conditional intent emission;
/// `sts2-rs-o7r` (P3-stretch) deliberately omits `intent_history`
/// (Python keeps it in the live `MonsterAI`, not the serialized
/// `Creature`).
#[must_use]
pub fn combat_state_to_bridge_state(state: &CombatState) -> Value {
    // Branch on pending_choice first to mirror Python lines 303-316.
    // Bead sts2-rs-ck2 fills in the real options/min/max emission;
    // Python always emits `'type': 'card_select'` (single class), Rust
    // additionally emits `'kind'` for the [`ChoiceKind`] discriminant
    // so callers can branch without prompt parsing.
    if let Some(pc) = state.pending_choice.as_ref() {
        return pending_choice_to_bridge_value(pc, &state.cards);
    }

    // Per-enemy shape mirrors Python `_normalize_enemies`
    // (`bridge_replay.py:222-240`).
    //
    // Field provenance:
    //   * Bead sts2-rs-5if (VTHIN-2): enriched `Creature` with
    //     `current_hp` / `max_hp` / `block` / `powers`, so those are
    //     real values rather than VTHIN-1 placeholders.
    //   * Bead sts2-rs-pg5 (P4-3): replaced the hard-coded
    //     `"Cultist"` id with `creature.kind.python_name()`, and added
    //     conditional `intent` / `intent_damage` / `intent_hits` emission
    //     gated on `creature.intent.is_some()`. This matches Python
    //     `combat_state_to_bridge_state` lines 332-339 which only emit
    //     intent fields when `ai is not None`. Critically: VTHIN-2's
    //     `Cultist with intent=None` fixture continues to pass byte-
    //     equality because the new fields are omitted when intent is
    //     absent.
    //   * Bead sts2-rs-o7r (P3-stretch): `Creature::intent_history` is
    //     INTENTIONALLY NOT EMITTED. Python pin 01c094d keeps the
    //     equivalent state inside the live `MonsterAI` object (not a
    //     serialized field of `Creature`); emitting it on the Rust side
    //     would diverge byte-for-byte from `combat_state_to_bridge_state`
    //     and break every existing fixture. The field is engine-internal
    //     plumbing for the next-intent roll; the parity surface stays
    //     just `intent` / `intent_damage` / `intent_hits`.
    //
    // **Power normalization** mirrors Python `_normalize_powers`
    // (`bridge_replay.py:188-196`): emit `[{"id": "<NAME>", "amount": N}]`
    // sorted by `(id, amount)`, **omitting any power with `amount == 0`**
    // (per `combat_state_to_bridge_state` line 326-329:
    // `if power.amount != 0`). The sort step is implicit — Python sorts
    // by string key, but Rust BTreeMap-backed Map already sorts
    // alphabetically when serialized. The `amount` filter is explicit
    // here.
    let enemies: Vec<Value> = state
        .enemy_uids
        .iter()
        .map(|uid| {
            let creature = state.creatures.iter().find(|c| c.id == uid.0);
            if let Some(c) = creature {
                let mut powers: Vec<Value> = c
                    .powers
                    .iter()
                    .filter_map(|puid| {
                        let p = state.powers.get_by_id(puid.0)?;
                        if p.amount == 0 {
                            return None;
                        }
                        Some(json!({
                            "id": power_kind_to_python_name(p.kind),
                            "amount": p.amount,
                        }))
                    })
                    .collect();
                sort_powers_in_place(&mut powers);

                // Build the per-enemy map. We use `serde_json::Map` directly
                // (rather than `json!({...})`) so we can conditionally
                // insert `intent` / `intent_damage` / `intent_hits` only
                // when the creature has an active intent — matching
                // Python's `if ai is not None` gate.
                let mut obj = serde_json::Map::new();
                obj.insert("id".into(), Value::String(c.monster_id.0.into()));
                obj.insert("hp".into(), json!(c.current_hp));
                obj.insert("max_hp".into(), json!(c.max_hp));
                obj.insert("block".into(), json!(c.block));
                obj.insert("is_alive".into(), json!(c.current_hp > 0));
                obj.insert("powers".into(), Value::Array(powers));
                if let Some(intent) = c.intent {
                    insert_intent_fields(&mut obj, intent);
                }
                Value::Object(obj)
            } else {
                // Defensive fallback — `enemy_uids` should always
                // resolve into `creatures`, but a missing entry should
                // not panic the exporter.
                json!({
                    "id": "UNKNOWN",
                    "hp": 0,
                    "max_hp": 0,
                    "block": 0,
                    "is_alive": false,
                    "powers": [],
                })
            }
        })
        .collect();

    // Player-side power emission mirrors the enemy block above and
    // Python `_normalize_powers` (`bridge_replay.py:188-196`): filter
    // any power whose `amount == 0`, emit `[{"id", "amount"}]`. The
    // array is then explicitly sorted by `(id, amount)` to match
    // Python's tuple sort — this is the real ordering source for
    // byte-equality (insertion order is NOT stable in CombatState).
    let mut player_powers: Vec<Value> = state
        .player
        .powers
        .iter()
        .filter_map(|puid| {
            let p = state.powers.get_by_id(puid.0)?;
            if p.amount == 0 {
                return None;
            }
            Some(json!({
                "id": power_kind_to_python_name(p.kind),
                "amount": p.amount,
            }))
        })
        .collect();
    sort_powers_in_place(&mut player_powers);

    // Hand serialization — mirrors Python `combat_state_to_bridge_state`
    // (`bridge_replay.py:356-368`) → `_normalize_cards` (lines 199-219).
    // Each card emits `{id, type, cost, target, playable, [upgraded],
    // base_damage, base_block}`. `upgraded` is emitted ONLY when truthy
    // (Python's `if card.get("upgraded"): item["upgraded"] = True`).
    // Insertion order in the emitted map does not matter — outer
    // serializer is BTreeMap-backed and sorts keys alphabetically.
    //
    // Bead sts2-rs-hb0: replaces the prior `"hand": []` stub. Resolves the
    // upstream blocker for card-play parity fixtures (06w deferral).
    let hand: Vec<Value> = state
        .player
        .hand
        .iter()
        .filter_map(|uid| state.cards.iter().find(|c| c.uid == *uid))
        .map(|card| card_to_bridge_value(card, state))
        .collect();

    // Combat shape — keys laid out in Python `normalize_bridge_state`
    // (lines 248-264) insertion order. `serde_json::Map` is backed by a
    // `BTreeMap` (default), so on-disk serialization is alphabetical
    // (matches Python's `json.dumps(..., sort_keys=True)`).
    //
    // Type tag matches sts2-rl-agent@01c094d:sts2_env/parity/bridge_replay.py:32
    // STATE_TYPE_COMBAT = "combat_action".
    json!({
        "type": "combat_action",
        "player": {
            "hp": state.player.current_hp,
            "max_hp": state.player.max_hp,
            "block": state.player.block,
            "energy": state.player.energy,
            "max_energy": state.player.max_energy,
            "powers": player_powers,
        },
        "hand": hand,
        "enemies": enemies,
        "draw_pile_count": state.player.draw.len(),
        "discard_pile_count": state.player.discard.len(),
        "exhaust_pile_count": state.player.exhaust.len(),
        "round": state.round,
    })
}

/// Serialize a single [`ChoiceOption`] entry per the per-card shape
/// emitted at `bridge_replay.py:307-311` — `{"id", "type", "upgraded"}`.
/// `type` is omitted when the option has no resolvable `card_uid`
/// (synthetic options for orb-slot / enemy-target flavors).
#[must_use]
fn choice_option_to_bridge_value(option: ChoiceOption, cards: &[Card]) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "id".into(),
        Value::String(option.card_id.to_python_name().into()),
    );
    let resolved = option
        .card_uid
        .and_then(|uid| cards.iter().find(|c| c.uid == uid));
    let (ctype_opt, upgraded) = match resolved {
        Some(card) => (Some(card.card_type), card.upgraded),
        None => (None, option.upgraded),
    };
    if let Some(ct) = ctype_opt {
        obj.insert(
            "type".into(),
            Value::String(card_type_python_name(ct).into()),
        );
    }
    obj.insert("upgraded".into(), Value::Bool(upgraded));
    Value::Object(obj)
}

/// Render a [`PendingChoice`] as the bridge-state payload. Mirrors
/// Python `combat_state_to_bridge_state` lines 303-316 with the
/// Rust-port extension of an explicit `kind` discriminant.
#[must_use]
pub fn pending_choice_to_bridge_value(pc: &PendingChoice, cards: &[Card]) -> Value {
    let cards_arr: Vec<Value> = pc
        .options
        .iter()
        .map(|opt| choice_option_to_bridge_value(*opt, cards))
        .collect();
    let kind_str: &'static str = match pc.kind {
        crate::ChoiceKind::CardSelect => "card_select",
        crate::ChoiceKind::PutBack => "put_back",
        crate::ChoiceKind::ExhaustPile => "exhaust_pile",
        crate::ChoiceKind::OrbSlot => "orb_slot",
        crate::ChoiceKind::EnemyTarget => "enemy_target",
        crate::ChoiceKind::AddGeneratedToHand => "add_generated_to_hand",
        crate::ChoiceKind::AddGeneratedRetainedToHand => "add_generated_retained_to_hand",
        crate::ChoiceKind::AddGeneratedFreeToHand => "add_generated_free_to_hand",
        crate::ChoiceKind::PaelsToothStash => "paels_tooth_stash",
        crate::ChoiceKind::UpgradeInHand => "upgrade_in_hand",
        crate::ChoiceKind::AddFromDrawByType => "add_from_draw_by_type",
        crate::ChoiceKind::AddFromDiscardFree => "add_from_discard_free",
        crate::ChoiceKind::DiscardFromHand => "discard_from_hand",
        crate::ChoiceKind::ExhaustFromHand => "exhaust_from_hand",
        crate::ChoiceKind::DiscardFromHandAndDraw => "discard_from_hand_and_draw",
        crate::ChoiceKind::MakeFreeInHand => "make_free_in_hand",
        crate::ChoiceKind::MonsterCurseSelect => "monster_curse_select",
        crate::ChoiceKind::AddFromDrawToHand => "add_from_draw_to_hand",
        crate::ChoiceKind::ForegoneConclusionFromDraw => "foregone_conclusion_from_draw",
        crate::ChoiceKind::AddFromDiscardToHand => "add_from_discard_to_hand",
        crate::ChoiceKind::ApplyRetainInHand => "apply_retain_in_hand",
        crate::ChoiceKind::ApplySlyInHand => "apply_sly_in_hand",
        crate::ChoiceKind::ApplyEtherealInHand => "apply_ethereal_in_hand",
        crate::ChoiceKind::TransformInHandToMinionStrike => "transform_in_hand_to_minion_strike",
        crate::ChoiceKind::TransformInHandToMinionSacrifice => {
            "transform_in_hand_to_minion_sacrifice"
        }
        crate::ChoiceKind::TransfigureCardInHand => "transfigure_card_in_hand",
        crate::ChoiceKind::TransformDrawCardToSoul => "transform_draw_card_to_soul",
        crate::ChoiceKind::TransformDrawCardsToMinionDiveBomb => {
            "transform_draw_cards_to_minion_dive_bomb"
        }
        crate::ChoiceKind::CloneChosenCardNextTurn => "clone_chosen_card_next_turn",
        crate::ChoiceKind::RetainAtTurnEnd => "retain_at_turn_end",
        crate::ChoiceKind::AutoPlaySkillFromHand => "auto_play_skill_from_hand",
    };
    json!({
        "type": "card_select",
        "kind": kind_str,
        "cards": cards_arr,
        "min_select": pc.min_select,
        "max_select": pc.max_select,
        "can_skip": pc.can_skip,
    })
}

/// Map a [`PowerKind`] to the Python `PowerId.<NAME>.name` string emitted
/// by `combat_state_to_bridge_state` at `bridge_replay.py:327`. Thin shim
/// over `PowerKind::python_name` exposed at module scope so the exporter
/// reads cleanly.
///
/// **Bead sts2-rs-aot (P2-3) note:** Python's `PowerId` enum at pin
/// `01c094d` has zero `ENEMY_`-prefixed variants — Strength on a player
/// and Strength on a cultist both serialize as `"STRENGTH"`. The
/// player/enemy-side distinction lives in *which creature's `powers`
/// list* contains the power, not in the id string. Therefore this map
/// remains 1-to-1 with `PowerKind`'s 10 variants and grows only when new
/// non-side-tagged Python `PowerId` members are ported.
#[must_use]
fn power_kind_to_python_name(kind: PowerKind) -> &'static str {
    kind.python_name()
}

/// Map a [`CardType`] to the title-cased string emitted by Python's
/// `_CARD_TYPE_NAMES` table (`bridge_replay.py:43-50`). Note the *title*
/// casing — not screaming-snake — because `combat_state_to_bridge_state`
/// looks up via the table rather than `.name`.
#[must_use]
const fn card_type_python_name(t: CardType) -> &'static str {
    match t {
        CardType::Attack => "Attack",
        CardType::Skill => "Skill",
        CardType::Power => "Power",
        CardType::Status => "Status",
        CardType::Curse => "Curse",
        CardType::Quest => "Quest",
    }
}

/// Map a [`TargetType`] to the title-cased string emitted by Python's
/// `_TARGET_TYPE_NAMES` table (`bridge_replay.py:52-60`). The Rust enum
/// collapses several Python `TargetType` variants (e.g. `ANY_ENEMY` and a
/// hypothetical `ENEMY` alias) into `Enemy`; the canonical Python label
/// for that bucket is `AnyEnemy`, which is what fixtures observe.
#[must_use]
const fn target_type_python_name(t: TargetType) -> &'static str {
    match t {
        TargetType::Enemy => "AnyEnemy",
        TargetType::AllEnemies => "AllEnemies",
        TargetType::Self_ => "Self",
        TargetType::None => "None",
        // `Random` ↔ Python `RANDOM_ENEMY`.
        TargetType::Random => "RandomEnemy",
    }
}

/// Cost-affordability check matching the cost arm of Python's
/// `combat.can_play_card(card)` (`combat.py:can_play_card`). Other
/// playability gates (Entangled, unplayable-curse flags, etc.) are not
/// yet modelled in the Rust port and default to "playable" — the hand
/// fixtures we record for bead sts2-rs-hb0 use only starter Strike /
/// Defend / Bash so this is sufficient for byte-equality on those
/// fixtures. When richer playability lands, extend this predicate (and
/// add a TODO if it diverges).
#[must_use]
fn card_is_playable(card: &Card, state: &CombatState) -> bool {
    state.player.energy >= card.local_cost()
}

/// Serialize one [`Card`] from the player's hand into the bridge-state
/// shape. Output matches Python's `_normalize_cards` output entry-for-
/// entry: `id`, `type`, `cost`, `target`, `playable`, optional
/// `upgraded` (true-only), `base_damage`, `base_block`. The outer
/// serializer is `BTreeMap`-backed so on-disk key order is alphabetical.
#[must_use]
fn card_to_bridge_value(card: &Card, state: &CombatState) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "id".into(),
        Value::String(card.card_id.to_python_name().into()),
    );
    obj.insert(
        "type".into(),
        Value::String(card_type_python_name(card.card_type).into()),
    );
    obj.insert("cost".into(), json!(card.local_cost()));
    obj.insert(
        "target".into(),
        Value::String(target_type_python_name(card.target_type).into()),
    );
    obj.insert("playable".into(), json!(card_is_playable(card, state)));
    if card.upgraded {
        obj.insert("upgraded".into(), Value::Bool(true));
    }
    obj.insert("base_damage".into(), json!(card.base_damage));
    obj.insert("base_block".into(), json!(card.base_block));
    Value::Object(obj)
}

/// Sort an emitted powers array by `(id, amount)` to match Python's
/// `_normalize_powers` (`bridge_replay.py:188-196`), which calls
/// `sorted(powers, key=lambda p: (p.id, p.amount))`. Without this,
/// Rust's emit order tracks `Power.id` insertion which is NOT stable
/// across `CombatState` mutations and diverges byte-for-byte from
/// Python whenever a creature carries 2+ powers added in a non-
/// alphabetical order.
///
/// Surfaced by sts2-rs-m2r (P2-6 1k-seed fuzz) — bead sts2-rs-yyf
/// is the fix.
fn sort_powers_in_place(powers: &mut [Value]) {
    powers.sort_by(|a, b| {
        let a_id = a.get("id").and_then(Value::as_str).unwrap_or("");
        let b_id = b.get("id").and_then(Value::as_str).unwrap_or("");
        let a_amount = a.get("amount").and_then(Value::as_i64).unwrap_or(0);
        let b_amount = b.get("amount").and_then(Value::as_i64).unwrap_or(0);
        (a_id, a_amount).cmp(&(b_id, b_amount))
    });
}

/// Insert the three `intent` / `intent_damage` / `intent_hits` fields into
/// `obj` in the order Python emits them. Mirrors
/// `combat_state_to_bridge_state` lines 337-339:
/// ```python
/// enemy_data["intent"] = first_intent.intent_type.name
/// enemy_data["intent_damage"] = first_intent.damage
/// enemy_data["intent_hits"] = first_intent.hits
/// ```
/// The fields are unconditionally emitted whenever the creature has an
/// active intent (Python only gates on `ai is not None`, then on
/// `if intents:`). On-disk byte ordering is alphabetical (`BTreeMap`
/// backing, matching Python's `sort_keys=True`), so insertion order
/// here is purely cosmetic — but kept Python-canonical for readability.
fn insert_intent_fields(obj: &mut serde_json::Map<String, Value>, intent: Intent) {
    obj.insert("intent".into(), Value::String(intent.python_name().into()));
    obj.insert("intent_damage".into(), json!(intent.damage()));
    obj.insert("intent_hits".into(), json!(intent.hits()));
}

/// Build a [`TraceStep`] capturing `action` and the post-state derived from
/// `state`. `action == None` is recorded as JSON `null`.
#[must_use]
pub fn trace_step(state: &CombatState, action: Option<Value>) -> TraceStep {
    TraceStep {
        action: action.unwrap_or(Value::Null),
        resulting_state: combat_state_to_bridge_state(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bead `sts2-rs-9ntvs`: when `state.pending_choice = Some(_)`,
    /// `combat_state_to_bridge_state` must short-circuit to the
    /// `pending_choice_to_bridge_value` shape (parity.rs:84-86)
    /// instead of emitting the combat shape. The inner helper has
    /// direct coverage in tests/pending_choice_bridge.rs but the
    /// integrated path was untested — a regression that removed the
    /// short-circuit would silently break every fixture capturing a
    /// mid-choice state.
    #[test]
    fn combat_state_to_bridge_state_short_circuits_on_pending_choice() {
        let pc = PendingChoice {
            kind: crate::ChoiceKind::CardSelect,
            options: vec![ChoiceOption {
                card_id: crate::cards::CardId::StrikeIronclad,
                upgraded: false,
                card_uid: None,
                source_pile: None,
            }],
            min_select: 1,
            max_select: 1,
            can_skip: false,
            post_effect: None,
            picked: smallvec::SmallVec::new(),
        };
        let state = CombatState {
            pending_choice: Some(pc),
            ..CombatState::default()
        };
        let value = combat_state_to_bridge_state(&state);
        // Choice shape: type="card_select", carries `cards` not `enemies`.
        assert_eq!(value["type"], json!("card_select"));
        assert_eq!(value["kind"], json!("card_select"));
        assert!(value.get("cards").is_some(), "must carry cards key");
        assert!(
            value.get("enemies").is_none(),
            "must not carry combat-shape enemies"
        );
        assert!(
            value.get("player").is_none(),
            "must not carry combat-shape player"
        );
    }

    /// Regression for sts2-rs-yyf — `sort_powers_in_place` must order
    /// emitted powers by `(id, amount)`, matching Python's
    /// `_normalize_powers` tuple sort. Pre-fix Rust emitted insertion
    /// order, which diverged byte-for-byte from Python whenever a
    /// creature carried 2+ powers added in non-alphabetical order.
    #[test]
    fn sort_powers_in_place_orders_by_id_then_amount() {
        let mut powers = vec![
            json!({"id": "VULNERABLE", "amount": 2}),
            json!({"id": "STRENGTH", "amount": 5}),
            json!({"id": "STRENGTH", "amount": 1}),
            json!({"id": "WEAK", "amount": 1}),
        ];
        sort_powers_in_place(&mut powers);
        assert_eq!(
            powers,
            vec![
                json!({"id": "STRENGTH", "amount": 1}),
                json!({"id": "STRENGTH", "amount": 5}),
                json!({"id": "VULNERABLE", "amount": 2}),
                json!({"id": "WEAK", "amount": 1}),
            ]
        );
    }

    /// All top-level keys from `normalize_bridge_state` (lines 248-264) must
    /// be present in the canonical order. `serde_json` (BTreeMap-backed)
    /// emits keys alphabetically — same as Python's `json.dumps(...,
    /// sort_keys=True)` — so we assert the alphabetical order which is
    /// what hits disk.
    #[test]
    fn empty_combat_state_serializes_with_expected_keys() {
        let state = CombatState::default();
        let value = combat_state_to_bridge_state(&state);
        let json = serde_json::to_string(&value).expect("serialize");

        // Canonical key set per `normalize_bridge_state`. All must appear.
        for key in [
            "\"type\"",
            "\"player\"",
            "\"hand\"",
            "\"enemies\"",
            "\"draw_pile_count\"",
            "\"discard_pile_count\"",
            "\"exhaust_pile_count\"",
            "\"round\"",
        ] {
            assert!(
                json.contains(key),
                "missing key {key} in serialized output: {json}"
            );
        }

        // Player sub-object keys per Python lines 250-257.
        for key in [
            "\"hp\"",
            "\"max_hp\"",
            "\"block\"",
            "\"energy\"",
            "\"max_energy\"",
            "\"powers\"",
        ] {
            assert!(json.contains(key), "missing player key {key}");
        }

        // BTreeMap-backed Map sorts alphabetically. This is the same order
        // Python emits with `sort_keys=True`. Spot-check a few key
        // positions to lock in determinism.
        let pos = |needle: &str| json.find(needle).expect(needle);
        assert!(pos("\"discard_pile_count\"") < pos("\"draw_pile_count\""));
        assert!(pos("\"draw_pile_count\"") < pos("\"enemies\""));
        assert!(pos("\"enemies\"") < pos("\"exhaust_pile_count\""));
        assert!(pos("\"exhaust_pile_count\"") < pos("\"hand\""));
        assert!(pos("\"hand\"") < pos("\"player\""));
        assert!(pos("\"player\"") < pos("\"round\""));
        assert!(pos("\"round\"") < pos("\"type\""));
    }

    #[test]
    fn empty_combat_state_emits_combat_type() {
        // Matches sts2-rl-agent@01c094d:sts2_env/parity/bridge_replay.py:32
        // STATE_TYPE_COMBAT = "combat_action"
        let state = CombatState::default();
        let value = combat_state_to_bridge_state(&state);
        assert_eq!(value["type"], json!("combat_action"));
        assert_eq!(value["round"], json!(0));
        assert_eq!(value["draw_pile_count"], json!(0));
        assert_eq!(value["discard_pile_count"], json!(0));
        assert_eq!(value["exhaust_pile_count"], json!(0));
    }

    /// Pins the combat-branch type tag against the Python source-of-truth
    /// constant. Any drift in either side will trip this test.
    // Matches sts2-rl-agent@01c094d:sts2_env/parity/bridge_replay.py:32
    // STATE_TYPE_COMBAT = "combat_action"
    #[test]
    fn combat_branch_type_tag_matches_python_state_type_combat() {
        const PYTHON_STATE_TYPE_COMBAT: &str = "combat_action";
        let state = CombatState::default();
        let value = combat_state_to_bridge_state(&state);
        assert_eq!(value["type"], json!(PYTHON_STATE_TYPE_COMBAT));
    }

    #[test]
    fn trace_envelope_round_trips() {
        let state = CombatState::default();
        let step = trace_step(&state, Some(json!({ "action": "end_turn" })));

        let trace = Trace {
            version: 1,
            mode: "combat".to_string(),
            metadata: BTreeMap::new(),
            initial_state: combat_state_to_bridge_state(&state),
            steps: vec![step],
        };

        let serialized = serde_json::to_string(&trace).expect("serialize");
        let round_tripped: Trace = serde_json::from_str(&serialized).expect("deserialize");

        assert_eq!(round_tripped.version, trace.version);
        assert_eq!(round_tripped.mode, trace.mode);
        assert_eq!(round_tripped.metadata, trace.metadata);
        assert_eq!(round_tripped.initial_state, trace.initial_state);
        assert_eq!(round_tripped.steps.len(), 1);
        assert_eq!(round_tripped.steps[0].action, trace.steps[0].action);
        assert_eq!(
            round_tripped.steps[0].resulting_state,
            trace.steps[0].resulting_state
        );
    }

    /// Default state (no powers populated): both `player.powers` and
    /// `enemies` serialize as empty arrays. The companion
    /// `_normalize_powers` zero-amount filter has shipped (see the
    /// inline `if p.amount == 0 { return None; }` guard in
    /// `combat_state_to_bridge_state`) and is exercised by the
    /// non-default fixtures in `tests/parity/`.
    #[test]
    fn power_amount_zero_omitted() {
        let state = CombatState::default();
        let value = combat_state_to_bridge_state(&state);
        assert_eq!(value["player"]["powers"], json!([]));
        assert_eq!(value["enemies"], json!([]));
    }

    #[test]
    fn trace_step_with_no_action_is_null() {
        let state = CombatState::default();
        let step = trace_step(&state, None);
        assert_eq!(step.action, Value::Null);
        assert_eq!(step.resulting_state["type"], json!("combat_action"));
    }

    #[test]
    fn default_trace_matches_python_defaults() {
        // Python `BridgeReplayTrace` defaults: version=1, mode="combat",
        // empty metadata, empty initial_state dict, no steps.
        let trace = Trace::default();
        assert_eq!(trace.version, 1);
        assert_eq!(trace.mode, "combat");
        assert!(trace.metadata.is_empty());
        assert_eq!(trace.initial_state, json!({}));
        assert!(trace.steps.is_empty());
    }

    // ── Intent emission (bead sts2-rs-pg5 P4-3) ──────────────────────

    /// Helper: build a state with a single enemy carrying optional
    /// `kind` + `intent`. Mirrors the recorder shape used by the parity
    /// fixtures.
    fn state_with_one_enemy(
        kind: crate::MonsterKind,
        intent: Option<crate::Intent>,
    ) -> crate::CombatState {
        use crate::{Creature, CreatureUid};
        use smallvec::SmallVec;
        use std::num::NonZeroU32;

        let mut state = crate::CombatState::default();
        let cultist_uid = CreatureUid::new(1).expect("non-zero");
        state.creatures.push(Creature {
            id: NonZeroU32::new(1).unwrap(),
            current_hp: 48,
            max_hp: 48,
            block: 0,
            powers: SmallVec::new(),
            species: Some(kind),
            intent,
            intent_history: crate::IntentHistory::default(),
            monster_id: crate::MonsterId(kind.python_name()),
            ai: None,
        });
        state.enemy_uids.push(cultist_uid);
        state
    }

    /// VTHIN-2 baseline: an enemy with `intent: None` must NOT emit
    /// `intent` / `intent_damage` / `intent_hits`. This is the byte-
    /// equality property that lets the existing `vthin2_vulnerable_damage`
    /// fixture continue to pass after the P4-3 enrichment.
    #[test]
    fn parity_omits_intent_fields_when_intent_is_none() {
        let state = state_with_one_enemy(crate::MonsterKind::ShrinkerBeetle, None);
        let value = combat_state_to_bridge_state(&state);
        let enemy = &value["enemies"][0];
        assert_eq!(enemy["id"], json!("ShrinkerBeetle"));
        assert_eq!(enemy["hp"], json!(48));
        assert_eq!(enemy["max_hp"], json!(48));
        assert_eq!(enemy["block"], json!(0));
        assert_eq!(enemy["is_alive"], json!(true));
        assert_eq!(enemy["powers"], json!([]));
        // None of the intent fields appear.
        assert!(
            enemy.get("intent").is_none(),
            "intent should be absent: {enemy}"
        );
        assert!(
            enemy.get("intent_damage").is_none(),
            "intent_damage should be absent: {enemy}"
        );
        assert!(
            enemy.get("intent_hits").is_none(),
            "intent_hits should be absent: {enemy}"
        );
    }

    /// With `intent: Some(Attack { damage: 6 })`, emit all three
    /// fields per Python's `combat_state_to_bridge_state` lines
    /// 337-339. The string `"ATTACK"` matches `IntentType.ATTACK.name`
    /// at the Python pin.
    #[test]
    fn parity_emits_intent_fields_for_attack() {
        let state = state_with_one_enemy(
            crate::MonsterKind::ShrinkerBeetle,
            Some(crate::Intent::Attack { damage: 6 }),
        );
        let value = combat_state_to_bridge_state(&state);
        let enemy = &value["enemies"][0];
        assert_eq!(enemy["intent"], json!("ATTACK"));
        assert_eq!(enemy["intent_damage"], json!(6));
        assert_eq!(enemy["intent_hits"], json!(1));
    }

    /// `MultiAttack { damage: 4, hits: 3 }` emits hits=3, mirroring
    /// Python's `multi_attack_intent` shape (`intents.py:32`).
    #[test]
    fn parity_emits_intent_fields_for_multi_attack() {
        let state = state_with_one_enemy(
            crate::MonsterKind::Mawler,
            Some(crate::Intent::MultiAttack { damage: 4, hits: 3 }),
        );
        let value = combat_state_to_bridge_state(&state);
        let enemy = &value["enemies"][0];
        assert_eq!(enemy["id"], json!("Mawler"));
        assert_eq!(enemy["intent"], json!("MULTI_ATTACK"));
        assert_eq!(enemy["intent_damage"], json!(4));
        assert_eq!(enemy["intent_hits"], json!(3));
    }

    /// Non-attack intents must emit `intent_damage = 0` and
    /// `intent_hits = 1`, matching Python's `Intent.damage = 0` /
    /// `hits = 1` defaults at `intents.py:14-15`.
    #[test]
    fn parity_emits_zero_damage_for_non_attack_intents() {
        for (intent, name) in [
            (crate::Intent::Defend { block: 0 }, "DEFEND"),
            (
                crate::Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 0,
                },
                "BUFF",
            ),
            (
                crate::Intent::Debuff {
                    kind: PowerKind::Weak,
                    amount: 0,
                },
                "DEBUFF",
            ),
            (crate::Intent::DebuffStrong, "DEBUFF_STRONG"),
            (crate::Intent::Sleep, "SLEEP"),
            (crate::Intent::Stun, "STUN"),
            (crate::Intent::StatusCard, "STATUS_CARD"),
            (crate::Intent::Unknown, "UNKNOWN"),
        ] {
            let state = state_with_one_enemy(crate::MonsterKind::FuzzyWurmCrawler, Some(intent));
            let value = combat_state_to_bridge_state(&state);
            let enemy = &value["enemies"][0];
            assert_eq!(
                enemy["intent"],
                json!(name),
                "wrong intent name for {intent:?}"
            );
            assert_eq!(
                enemy["intent_damage"],
                json!(0),
                "expected damage=0 for {intent:?}"
            );
            assert_eq!(
                enemy["intent_hits"],
                json!(1),
                "expected hits=1 for {intent:?}"
            );
        }
    }

    /// **Bead sts2-rs-aot (P2-3)** locks the design decision that no
    /// `PowerKind` variant emits an `ENEMY_`-prefixed Python name. The
    /// player vs. enemy-side distinction is encoded by *which creature
    /// owns the power*, not by the id string — verified against
    /// Python pin `01c094d`'s `PowerId` enum at
    /// `sts2_env/core/enums.py:181-461`. This test catches accidental
    /// future drift if someone adds an `EnemyStrength` variant without
    /// rechecking the Python pin.
    #[test]
    fn no_power_kind_emits_enemy_prefixed_python_name() {
        let kinds = [
            PowerKind::Strength,
            PowerKind::Vulnerable,
            PowerKind::Weak,
            PowerKind::Dexterity,
            PowerKind::Frail,
            PowerKind::Intangible,
            PowerKind::Thorns,
            PowerKind::FlameBarrier,
            PowerKind::Poison,
            PowerKind::Artifact,
        ];
        for k in kinds {
            let name = power_kind_to_python_name(k);
            assert!(
                !name.starts_with("ENEMY_"),
                "PowerKind::{k:?} maps to {name:?} which starts with \
                 ENEMY_ — Python pin 01c094d has no ENEMY_ variants in \
                 PowerId; the player/enemy distinction is owner-based, \
                 not id-based (see crates/sts2-core/src/powers.rs \
                 module-doc, sts2-rs-aot)."
            );
        }
    }

    /// Each [`crate::MonsterKind`] variant must emit its canonical
    /// Python-name as the enemy's `id` field. Locks the parity
    /// exporter's `id` field against the STS2 PascalCase enemy names.
    #[test]
    fn parity_emits_monster_kind_python_name_as_id() {
        for kind in [
            crate::MonsterKind::ShrinkerBeetle,
            crate::MonsterKind::FuzzyWurmCrawler,
            crate::MonsterKind::LeafSlimeS,
            crate::MonsterKind::TwigSlimeS,
            crate::MonsterKind::Mawler,
        ] {
            let state = state_with_one_enemy(kind, None);
            let value = combat_state_to_bridge_state(&state);
            let enemy = &value["enemies"][0];
            assert_eq!(
                enemy["id"],
                json!(kind.python_name()),
                "wrong id for {kind:?}",
            );
        }
    }

    /// Bead sts2-rs-hb0: a 3-card hand must serialize to the Python
    /// `_normalize_cards` shape — `{id, type, cost, target, playable,
    /// [upgraded], base_damage, base_block}`. `upgraded` is emitted only
    /// when truthy. The outer state's `hand` array preserves the order
    /// the cards appear in `player.hand`.
    #[test]
    fn parity_hand_export_matches_python_shape() {
        use crate::cards::CardId;
        use crate::cards::{Card, CardType, TargetType};
        use crate::CardUid;

        use std::num::NonZeroU32;

        let mut state = crate::CombatState::default();
        // Player has enough energy to play Strike (cost 1) but maybe not
        // Bash (cost 2) and certainly not Strike+ if cost were higher;
        // the playable bit is per-card.
        state.player.energy = 1;

        let mk = |raw_uid: u32,
                  card_id: CardId,
                  cost: i8,
                  card_type: CardType,
                  target_type: TargetType,
                  upgraded: bool,
                  base_damage: i32,
                  base_block: i32| {
            Card {
                uid: CardUid(NonZeroU32::new(raw_uid).unwrap()),
                card_id,
                cost: crate::CardEnergyCost::new(cost),
                card_type,
                target_type,
                upgraded,
                base_damage,
                base_block,
                ..Card::pod_defaults()
            }
        };

        let strike = mk(
            1,
            CardId::StrikeIronclad,
            1,
            CardType::Attack,
            TargetType::Enemy,
            false,
            6,
            0,
        );
        let defend = mk(
            2,
            CardId::DefendIronclad,
            1,
            CardType::Skill,
            TargetType::Self_,
            false,
            0,
            5,
        );
        let bash = mk(
            3,
            CardId::Bash,
            2,
            CardType::Attack,
            TargetType::Enemy,
            false,
            8,
            0,
        );

        state.player.hand.push(strike.uid);
        state.player.hand.push(defend.uid);
        state.player.hand.push(bash.uid);
        state.cards.push(strike);
        state.cards.push(defend);
        state.cards.push(bash);

        let value = combat_state_to_bridge_state(&state);
        let hand = value["hand"].as_array().expect("hand must be an array");
        assert_eq!(hand.len(), 3, "hand length");

        // Card 0: Strike — affordable @ energy=1.
        assert_eq!(hand[0]["id"], json!("STRIKE_IRONCLAD"));
        assert_eq!(hand[0]["type"], json!("Attack"));
        assert_eq!(hand[0]["cost"], json!(1));
        assert_eq!(hand[0]["target"], json!("AnyEnemy"));
        assert_eq!(hand[0]["playable"], json!(true));
        assert_eq!(hand[0]["base_damage"], json!(6));
        assert_eq!(hand[0]["base_block"], json!(0));
        assert!(
            hand[0].get("upgraded").is_none(),
            "non-upgraded card must omit `upgraded` key"
        );

        // Card 1: Defend.
        assert_eq!(hand[1]["id"], json!("DEFEND_IRONCLAD"));
        assert_eq!(hand[1]["type"], json!("Skill"));
        assert_eq!(hand[1]["target"], json!("Self"));
        assert_eq!(hand[1]["playable"], json!(true));
        assert_eq!(hand[1]["base_block"], json!(5));

        // Card 2: Bash — cost 2 > energy 1 → playable=false.
        assert_eq!(hand[2]["id"], json!("BASH"));
        assert_eq!(hand[2]["cost"], json!(2));
        assert_eq!(hand[2]["playable"], json!(false));
        assert_eq!(hand[2]["base_damage"], json!(8));
    }

    /// Upgraded cards must surface `upgraded: true` (truthy-only emit,
    /// per Python `_normalize_cards`).
    #[test]
    fn parity_hand_upgraded_card_emits_truthy_key() {
        use crate::cards::CardId;
        use crate::cards::{Card, CardType, TargetType};
        use crate::CardUid;

        use std::num::NonZeroU32;

        let mut state = crate::CombatState::default();
        state.player.energy = 3;
        let upgraded_strike = Card {
            uid: CardUid(NonZeroU32::new(1).unwrap()),
            card_id: CardId::StrikeIronclad,
            cost: crate::CardEnergyCost::new(1),
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: true,
            base_damage: 9,
            base_block: 0,
            ..Card::pod_defaults()
        };
        state.player.hand.push(upgraded_strike.uid);
        state.cards.push(upgraded_strike);

        let value = combat_state_to_bridge_state(&state);
        assert_eq!(value["hand"][0]["upgraded"], json!(true));
        assert_eq!(value["hand"][0]["base_damage"], json!(9));
    }
}
