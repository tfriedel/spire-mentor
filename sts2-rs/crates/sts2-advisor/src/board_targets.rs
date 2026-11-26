//! Engine enemy slots vs the mod's.
//!
//! An engine action names its enemy by **stable** slot, an index into
//! `CombatState::enemy_uids` (`AliveEnemies` walks that list). The mod names
//! it by index into C# `CombatState.Enemies`, which is what it serializes as
//! the board's `enemies[]` (`SimCombatHandler.ResolveTarget`), and C# takes a
//! dead enemy out of that list (`RemoveCreature`) unless a power keeps its
//! corpse. The engine keeps every corpse in its slot (a reviver must find
//! it); [`CombatState::is_removed_corpse`] says which ones C# removed, and
//! [`CombatState::active_ordinal_of_enemy`] bridges the two.
//!
//! A fight rebuilt from the board has no removed corpse, so the slots
//! coincide. The advisor's mirror carries the fight across boards and keeps
//! the corpses of the enemies that died in it, and from the first such death
//! on every enemy after it sits one slot further in the engine than on the
//! board. Read verbatim, Instinct's `target_index` then named a slot the
//! board does not have ("recommended target no longer hittable"), and after
//! three refusals the mod ended the player's turn, again on every turn left
//! in the fight (autobattle, 2026-11-23).

use serde_json::Value;
use sts2_core::CombatState;

use crate::py;

/// Engine enemy slot -> the mod's `enemies[]` index, for one position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoardTargets {
    /// Per engine slot, its board index (`None`: a corpse C# removed).
    /// `None` when every slot is its own board index.
    map: Option<Vec<Option<i64>>>,
}

impl BoardTargets {
    /// The slots of `fight` (the engine fight the advice was computed on).
    #[must_use]
    pub fn of(fight: &CombatState) -> Self {
        let map: Vec<Option<i64>> = (0..fight.enemy_uids.len())
            .map(|i| {
                fight
                    .active_ordinal_of_enemy(i)
                    .and_then(|o| i64::try_from(o).ok())
            })
            .collect();
        let identity = map
            .iter()
            .enumerate()
            .all(|(i, o)| *o == i64::try_from(i).ok());
        Self {
            map: (!identity).then_some(map),
        }
    }

    /// Whether engine and board slots coincide.
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.map.is_none()
    }

    /// The board index of engine slot `slot`; `None` for a removed corpse
    /// (no board index) or a slot the fight does not have.
    #[must_use]
    pub fn to_board(&self, slot: i64) -> Option<i64> {
        match &self.map {
            None => Some(slot),
            Some(m) => usize::try_from(slot)
                .ok()
                .and_then(|i| m.get(i).copied().flatten()),
        }
    }

    /// The engine slot of board index `index` (the inverse of
    /// [`Self::to_board`]).
    #[must_use]
    pub fn to_engine(&self, index: i64) -> Option<i64> {
        match &self.map {
            None => Some(index),
            Some(m) => m
                .iter()
                .position(|o| *o == Some(index))
                .and_then(|i| i64::try_from(i).ok()),
        }
    }

    /// `action` (engine or board combat form, its enemy in `target_index`)
    /// with that enemy moved through `f`; `None` when `f` has no slot for
    /// it. An action without an enemy target comes back unchanged.
    fn moved(action: &Value, f: impl Fn(i64) -> Option<i64>) -> Option<Value> {
        let t = py::get(action, "target_index");
        if t.is_null() {
            return Some(action.clone());
        }
        let to = f(py::int_of(t))?;
        let mut out = action.clone();
        out["target_index"] = to.into();
        Some(out)
    }

    /// An engine action with its target as the board's index.
    #[must_use]
    pub fn board_action(&self, action: &Value) -> Option<Value> {
        if self.is_identity() {
            return Some(action.clone());
        }
        Self::moved(action, |t| self.to_board(t))
    }

    /// A board action (the mod's report of a play) with its target as the
    /// engine's slot.
    #[must_use]
    pub fn engine_action(&self, action: &Value) -> Option<Value> {
        if self.is_identity() {
            return Some(action.clone());
        }
        Self::moved(action, |t| self.to_engine(t))
    }

    /// A search `result` (`{"actions": [{"action": ...}, ...], ...}`) with
    /// each ranked row's action targeting the board's index; a row whose
    /// enemy has no board index is left out. The `pv` stays in engine
    /// slots: the turn plan replays it on the engine fight.
    #[must_use]
    pub fn board_result(&self, result: &Value) -> Value {
        if self.is_identity() {
            return result.clone();
        }
        let mut out = result.clone();
        if let Some(rows) = out.get_mut("actions").and_then(Value::as_array_mut) {
            rows.retain_mut(|row| {
                let Some(action) = row.get("action").filter(|a| a.is_object()) else {
                    return true;
                };
                match self.board_action(action) {
                    Some(a) => {
                        row["action"] = a;
                        true
                    }
                    None => false,
                }
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn targets(map: Vec<Option<i64>>) -> BoardTargets {
        BoardTargets { map: Some(map) }
    }

    #[test]
    fn identity_moves_nothing() {
        let t = BoardTargets::default();
        let a = json!({"action": "play", "card_index": 2, "target_index": 1});
        assert_eq!(t.board_action(&a), Some(a.clone()));
        assert_eq!(t.engine_action(&a), Some(a));
    }

    #[test]
    fn a_removed_corpse_shifts_every_later_enemy_down_a_slot() {
        // [Living Shield (dead, removed), Turret Operator]: the board lists
        // the Turret alone, at 0.
        let t = targets(vec![None, Some(0)]);
        let play = |ti: i64| json!({"action": "play", "card_index": 3, "target_index": ti});
        assert_eq!(t.board_action(&play(1)), Some(play(0)));
        assert_eq!(t.board_action(&play(0)), None);
        assert_eq!(t.engine_action(&play(0)), Some(play(1)));
        let untargeted = json!({"action": "play", "card_index": 1, "target_index": null});
        assert_eq!(t.board_action(&untargeted), Some(untargeted));
    }

    #[test]
    fn a_ranked_row_on_a_removed_corpse_leaves_the_result() {
        let t = targets(vec![Some(0), None, None, Some(1)]);
        let result = json!({
            "actions": [
                {"action": {"action": "play", "card_index": 0, "target_index": 3}, "p_win": 0.5},
                {"action": {"action": "play", "card_index": 0, "target_index": 1}, "p_win": 0.4},
                {"action": {"action": "end_turn"}, "p_win": 0.1},
            ],
            "pv": [{"action": "play", "card_index": 0, "target_index": 3}],
        });
        let out = t.board_result(&result);
        let rows = out["actions"].as_array().expect("rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["action"]["target_index"], json!(1));
        assert_eq!(rows[1]["action"]["action"], json!("end_turn"));
        assert_eq!(out["pv"][0]["target_index"], json!(3));
    }
}
