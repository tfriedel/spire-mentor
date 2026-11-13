//! `sts2_wire::emit` — the wire *serializers*, with no Python in them.
//!
//! # Parity by construction
//!
//! Every dict the `sts2_rs` Python module hands out — `state_to_dict`, the
//! `RunManager` accessors (`full_map`, `shop_inventory`, `available_actions`,
//! …), `legal_actions`, `pending_choice`, the card previews — used to be
//! built by hand inside the `PyO3` crate, one `PyDict::set_item` at a time.
//! That pinned every consumer of those shapes to having `libpython` in the
//! process, and it meant a second, non-Python producer of the same shape (a
//! wasm build of the workbench, a Rust test, a fixture replayer) could only
//! ever be a *copy* that drifts.
//!
//! This module is the single definition. Each emitter here builds a
//! [`serde_json::Value`] (or, where the Python-visible shape is a tuple or an
//! int-keyed dict that JSON cannot spell, a plain serde-serializable Rust
//! type — see below), and the `PyO3` crate produces its Python object by
//! running that value through `pythonize`. There is no second copy: whatever
//! the browser reads as JSON is byte-for-byte what Python reads as a dict,
//! because both come from the same function.
//!
//! # Conventions
//!
//! * `Value::Null` is emitted exactly where the `PyO3` code set `py.None()`,
//!   and a key is OMITTED exactly where the `PyO3` code omitted it. The two
//!   are distinguishable on both sides (`d.get("k")` is `None` vs
//!   `Some(Null)`), and consumers rely on the distinction (a missing
//!   `calculated_value` means "this card has no such number").
//! * Integers are emitted as JSON numbers via `serde_json::to_value` from
//!   the same Rust integer type the `PyO3` code passed to `set_item`; floats
//!   are widened `f32 → f64`.
//! * **Typed returns.** JSON has no tuples and no int-keyed maps. Where the
//!   Python surface returned one — `master_deck_detailed() -> [(id, upgraded)]`,
//!   `neow_offers() -> (a, b, c) | None`, `card_damage_preview() ->
//!   [{enemy_idx: dmg}]` — the emitter returns the matching Rust type
//!   (`Vec<(String, bool)>`, `Option<(String, String, String)>`,
//!   `Vec<BTreeMap<usize, i32>>`). `PyO3` converts those natively (tuples stay
//!   tuples, `usize` keys stay `int`), and `serde_json::to_value` serializes
//!   them as arrays / string-keyed objects for a JSON consumer. Two places
//!   nest a tuple INSIDE a dict (`act_map()[i]["children"]` and
//!   `current_room()["map_coord"]`); those are emitted as JSON arrays and the
//!   `PyO3` crate restores the tuple after `pythonize` — a dialect concern,
//!   not a content one.
//! * Key ORDER inside an object is `serde_json::Map`'s (sorted), not the
//!   `set_item` insertion order the hand-built dicts had. Dict equality and
//!   every consumer are order-blind; only a textual JSON diff would notice.
//!
//! # Layout
//!
//! * [`combat`] — `CombatState` → the hint/bridge wire dict
//!   ([`combat::combat_state_to_value`]), plus the bridge-shape action /
//!   pending-choice dicts and the hand previews.
//! * [`mcts`] — the `mcts_recommend` / `mcts_search_tree` result dicts and
//!   the one parser of their loose `config` dict.
//! * [`run_action`] — `RunAction` ⇄ `{"action": "...", …}` dict.
//! * [`run_manager`] — the `RunManager` accessor twins, one free function per
//!   Python method, named after it.
//! * [`names`] — the string tables (`phase_str`, `character_str`, …) both the
//!   emitters and the `PyO3` / workbench hosts share.

use std::fmt;

pub mod combat;
pub mod mcts;
pub mod names;
pub mod run_action;
pub mod run_manager;

pub use combat::{
    card_block_preview, card_damage_preview, combat_action_to_value, combat_state_to_value,
    legal_actions, locate_card, pending_choice,
};
pub use run_action::{run_action_from_value, run_action_to_value};

/// An emitter's refusal — a state that violates its own invariants (a pile
/// uid with no card in the arena), or an action dict that names no action
/// the engine has.
///
/// Rendered as `"{path}: {detail}"` when a path is known, matching
/// [`crate::WireError`]'s form; the `PyO3` crate raises it as `ValueError`,
/// which is what the hand-built `PyDict` code raised for the same faults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitError {
    message: String,
}

impl EmitError {
    /// An error with a free-form message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// A schema-style error at a JSON path, rendered `"{path}: {detail}"`.
    #[must_use]
    pub fn at(path: &str, detail: &str) -> Self {
        Self {
            message: format!("{path}: {detail}"),
        }
    }

    /// The rendered message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EmitError {}

impl From<crate::WireError> for EmitError {
    fn from(e: crate::WireError) -> Self {
        Self::new(e.to_string())
    }
}

/// `serde_json::Map` builder helpers shared by the emitters.
///
/// `set` is `serde_json::to_value` with the one failure mode that cannot
/// happen here (a non-string map key or a non-finite float — nothing the
/// emitters pass) turned into a `Null` rather than a panic path.
pub(crate) mod map_ext {
    use serde::Serialize;
    use serde_json::{Map, Value};

    /// A fresh object.
    #[must_use]
    pub fn obj() -> Map<String, Value> {
        Map::new()
    }

    /// `serde_json::to_value`, `Null` on the impossible failure.
    pub fn val<T: Serialize>(v: T) -> Value {
        serde_json::to_value(v).unwrap_or(Value::Null)
    }

    /// Insert `key = v`.
    pub fn set<T: Serialize>(m: &mut Map<String, Value>, key: &str, v: T) {
        m.insert(key.to_owned(), val(v));
    }
}

#[cfg(test)]
mod tests {
    use super::EmitError;

    #[test]
    fn display_is_path_colon_detail_when_pathed() {
        let e = EmitError::at("$.deck.hand", "uid 7 not in arena");
        assert_eq!(e.to_string(), "$.deck.hand: uid 7 not in arena");
        let e = EmitError::new("unknown action kind \"x\"");
        assert_eq!(e.message(), "unknown action kind \"x\"");
    }
}
