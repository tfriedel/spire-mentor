//! Relic content: static data definitions for every relic in the pool.
//!
//! Each character / pool module ports the matching file from
//! `sts2-rl-agent@01c094d:sts2_env/relics/`. Bead P4-2 (sts2-rs-u4u)
//! ships the Ironclad starter relic (`BurningBlood`). Phase 7
//! (sts2-rs-h9u) fills the remaining ~190 variants and wires a
//! relic-effect interpreter that consumes [`RelicEffect`].
//!
//! ## Crate boundary
//!
//! [`RelicSpec`] is the *static template* (immutable data shared
//! across all instances of the same relic). It is distinct from
//! `sts2_core::Relic`, which is the *runtime instance* (per-run /
//! per-combat, with its own `RelicUid`). Phase 7 will add a
//! `RelicSpec -> Relic` materialization step at relic-pickup time.
//! This split mirrors the Phase-4 `CardSpec` / `Card` split from
//! bead sts2-rs-zcl.
//!
//! [`RelicEffect`] enumerates the on-trigger effects encoded so far.
//! Bodies are pure data; the interpreter lives in `sts2-core`'s
//! Phase-7 hooks dispatch + relic-effect executor (umbrella bead
//! sts2-rs-h9u). Adding a relic whose effect doesn't fit the current
//! variants requires extending [`RelicEffect`] rather than reaching
//! for a closure — keeping the engine deterministic, copy-able, and
//! serializable.

use sts2_core::{PowerKind, RelicKind};

pub mod common;
pub mod dispatch;
pub mod event_reward;
pub mod fool;
pub mod grab_bag;
pub mod ironclad;
pub mod neow_relics;
pub mod rare;
pub mod shop_event;
pub mod starter;
pub mod tradable;
pub mod uncommon;

// ---------------------------------------------------------------------------
// RelicEffect — on-trigger effect AST
// ---------------------------------------------------------------------------

/// Pure-data on-trigger effect for a relic. Phase 7 (umbrella bead
/// `sts2-rs-h9u`) will implement an interpreter that consumes this
/// enum from the matching dispatcher hook.
///
/// Variants intentionally cover only the shapes shipped by bead
/// sts2-rs-u4u (Burning Blood) plus a `Noop` for relics whose
/// behaviour is bookkeeping-only (or not yet ported). Phase 7 adds
/// more variants as additional relics land — `DrawExtraCards`,
/// `GainGoldOnVictory`, `RemoveCurseOnPickup`, ... Closures are
/// deliberately avoided so [`RelicSpec`] stays `Copy`-eligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelicEffect {
    /// No on-trigger effect (placeholder for relics not yet ported,
    /// or for relics whose behaviour is implemented entirely via
    /// other hooks).
    Noop,
    /// Heal the owner for `amount` HP at end of combat (after
    /// victory). Mirrors Python's `BurningBlood.after_combat_victory`
    /// at `sts2_env/relics/starter.py:31-33`. The Python check
    /// `if owner.current_hp > 0` is implicit at the interpreter
    /// level — heal is only applied to a creature that survived
    /// combat (i.e. `after_combat_victory` only fires for the
    /// player on a win).
    HealAtCombatEnd { amount: i32 },
    /// Channel `count` orb(s) of `orb_kind` on the player's
    /// side-turn-start while `round_number <= max_round`. `phase`
    /// selects `before_` (CrackedCore, `starter.py:90-93`) vs
    /// `after_` (InfusedCore, `starter.py:104-107`) dispatch.
    ChannelOrbAtSideTurnStart {
        orb_kind: ChannelOrbKind,
        count: i32,
        phase: SideTurnPhase,
        max_round: u16,
    },
    /// Add `extra_cards` to the player's hand-draw count while
    /// `round_number <= max_round`. Mirrors Python
    /// `RingOfTheSnake.modify_hand_draw` (`starter.py:59-62`,
    /// `max_round = 1`) and `RingOfTheDrake.modify_hand_draw`
    /// (`starter.py:74-77`, `max_round = 3`). Implemented as a
    /// `modify_hand_draw` fold transform.
    ExtraHandDrawForRounds { extra_cards: i32, max_round: u16 },
    /// Grant `amount` Stars to the player on `after_side_turn_start`
    /// while `round_number <= max_round`. Mirrors Python
    /// `DivineRight.after_side_turn_start` (`starter.py:155-157`,
    /// `STARS = 3`, round 1) and `DivineDestiny.after_side_turn_start`
    /// (`starter.py:168-170`, `STARS = 6`, round 1).
    GainStarsAtSideTurnStart { amount: i32, max_round: u16 },
    /// Summon `count` Osty at combat start. Mirrors Python
    /// `BoundPhylactery.before_combat_start` (`starter.py:120-121`,
    /// `SUMMON_COUNT = 1`) and `PhylacteryUnbound.before_combat_start`
    /// (`starter.py:137-138`, `START_COMBAT_COUNT = 5`). The
    /// `summon_osty` runtime lands with the Necrobinder character
    /// bead; the listener is a marker until then.
    SummonAtCombatStart { count: i32 },
    /// Summon `count` Osty on the player's `after_energy_reset` hook
    /// when `round_number > 1`. Mirrors Python
    /// `BoundPhylactery.after_energy_reset` (`starter.py:123-125`).
    /// The `summon_osty` runtime lands with the Necrobinder
    /// character bead; the listener is a marker until then.
    SummonAtEnergyResetAfterRound1 { count: i32 },
    /// Append a `GoldReward(amount, amount)` to the rewards list when
    /// the player opens a chest/reward. Mirrors Python
    /// `AmethystAubergine.modify_rewards` (`common.py:29-40`,
    /// `GOLD = 10`). Bead `sts2-rs-6ynt.5.3`.
    AppendRewardGold { amount: i32 },
    /// Gain `amount` Block (unpowered) at `before_combat_start`.
    /// Mirrors Python `Anchor.before_combat_start` (`common.py:52-53`,
    /// `BLOCK = 10`). Bead `sts2-rs-6ynt.5.3`.
    GainBlockAtCombatStart { amount: i32 },
    /// Heal `amount` HP on the player's own `after_side_turn_start`
    /// hook. Mirrors Python `BloodVial.after_side_turn_start`
    /// (`common.py:78-80`, `HEAL = 2`). Bead `sts2-rs-6ynt.5.3`.
    HealAtPlayerSideTurnStart { amount: i32 },
    /// Apply `amount` stacks of [`PowerKind`] `kind` to the player at
    /// `before_combat_start`. Mirrors Python relics whose body is
    /// `owner.apply_power(PowerId.<X>, AMOUNT)` inside
    /// `before_combat_start` — `BronzeScales` (`common.py:105-106`,
    /// `THORNS = 3`), `Gorget` (`common.py:180-181`, `PLATING = 4`),
    /// `OddlySmoothStone`, `DataDisk`, `Vajra`. Bead `sts2-rs-6ynt.5.3`
    /// covers BronzeScales + Gorget; later A–H sub-beads reuse the
    /// shape.
    ApplyPowerAtCombatStart { kind: PowerKind, amount: i32 },
    /// On the *first* `after_damage_received` event of each combat,
    /// draw `amount` cards. The `_active` boolean in Python
    /// (`CentennialPuzzle.__init__`) resets to `True` in
    /// `after_combat_end`, so the trigger budget is per-combat.
    /// Mirrors Python `CentennialPuzzle.after_damage_received`
    /// (`common.py:121-127`, `CARDS = 3`). Bead `sts2-rs-6ynt.5.3`.
    DrawCardsOnFirstDamagePerCombat { amount: i32 },
    /// On the player's `after_side_turn_start` while `round_number <=
    /// max_round`, deal `amount` unpowered damage to every alive
    /// enemy. Mirrors Python `FestivePopper.after_side_turn_start`
    /// (`common.py:166-170`, `DAMAGE = 9`, gated on round 1). Bead
    /// `sts2-rs-6ynt.5.3`.
    DealDamageToAllEnemiesOnSideTurnStart { amount: i32, max_round: u16 },
    /// On every `period`-th player `after_side_turn_start`, gain
    /// `energy` energy. The counter persists across the combat
    /// (Python `HappyFlower._turns_seen`). Mirrors Python
    /// `HappyFlower.after_side_turn_start` (`common.py:197-201`,
    /// `ENERGY = 1`, `TURNS = 3`). Bead `sts2-rs-6ynt.5.3`.
    GainEnergyEveryNPlayerSideTurns { energy: i32, period: u16 },
    /// On `after_obtained`, gain `amount` max HP (and heal by the same
    /// amount per Python `Creature.gain_max_hp`). Run-state hook.
    /// Mirrors Python `Mango.after_obtained` (`rare.py:443-444`,
    /// `MAX_HP = 14`). Bead `sts2-rs-6ynt.5.8`.
    GainMaxHpOnObtained { amount: i32 },
    /// On `after_obtained`, gain `amount` gold. Run-state hook.
    /// Mirrors Python `OldCoin.after_obtained` (`rare.py:576-577`,
    /// `GOLD = 300`). Bead `sts2-rs-6ynt.5.8`.
    GainGoldOnObtained { amount: i32 },
    /// At `after_combat_victory`, if the player's current HP is at
    /// most `pct`% of max HP (Python integer-divide:
    /// `current_hp <= max_hp * pct // 100`), heal `heal`. Mirrors
    /// Python `MeatOnTheBone.after_combat_victory`
    /// (`rare.py:456-458`, `HP_THRESHOLD_PCT = 50`, `HEAL = 12`).
    /// Bead `sts2-rs-6ynt.5.8`.
    HealAtCombatVictoryBelowHpPct { heal: i32, pct: i32 },
    /// On the player's `before_turn_end` while `round_number ==
    /// round`, deal `amount` unpowered damage to every alive enemy.
    /// Mirrors Python `StoneCalendar.before_turn_end`
    /// (`rare.py:773-776`, `DAMAGE = 52`, `DAMAGE_TURN = 7`). Bead
    /// `sts2-rs-6ynt.5.8`.
    DealDamageToAllEnemiesOnPlayerTurnEndRound { amount: i32, round: u16 },
    /// On `modify_hp_lost` for the player, reduce the incoming HP-loss
    /// `amount` by `reduction` (clamped at 0). Mirrors Python
    /// `TungstenRod.modify_hp_lost` (`rare.py:884-889`,
    /// `REDUCTION = 1`). Bead `sts2-rs-6ynt.5.8`.
    ReducePlayerHpLossAdditive { reduction: i32 },
}

/// Orb-kind selector for [`RelicEffect::ChannelOrbAtSideTurnStart`].
/// Mirrors the string literal Python passes into `combat.channel_orb`
/// (`"LIGHTNING"` only at pin `01c094d` for starter-tier relics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelOrbKind {
    Lightning,
}

/// Discriminates `before_` vs `after_` side-turn-start dispatch.
/// Python `CrackedCore` uses `before_side_turn_start`; `InfusedCore`
/// uses `after_side_turn_start`. Lives on the [`RelicEffect`] AST so
/// the interpreter can route to the correct dispatch channel at
/// registration time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SideTurnPhase {
    Before,
    After,
}

// ---------------------------------------------------------------------------
// RelicRarity — bucket enum mirroring Python's RelicRarity
// ---------------------------------------------------------------------------

/// Relic rarity bucket. Mirrors
/// `sts2-rl-agent@01c094d:sts2_env/core/enums.py::RelicRarity` exactly.
/// All 8 Python variants are listed so Phase 7 doesn't need to
/// re-edit this enum when adding non-starter relics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelicRarity {
    Starter,
    Common,
    Uncommon,
    Rare,
    Shop,
    Event,
    Boss,
    Ancient,
}

// ---------------------------------------------------------------------------
// RelicSpec — static template for one relic
// ---------------------------------------------------------------------------

/// Static relic template. One [`RelicSpec`] exists per distinct
/// [`RelicKind`]. Concrete in-run relics (`sts2_core::Relic`) are
/// instantiated from a `RelicSpec` at relic-pickup / run-build time.
///
/// Field selection mirrors the load-bearing fields of
/// `sts2_env/relics/base.py::RelicInstance` minus runtime-only state
/// (`counter`, `pulse_value`, ...). The `name` is the parity-output
/// identifier matching Python's `RelicId.<NAME>.name`
/// (e.g. `"BURNING_BLOOD"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelicSpec {
    /// Stable identifier (matches `sts2_core::RelicKind`).
    pub id: RelicKind,
    /// Python `RelicId.<v>.name`. Used directly by the (future
    /// Phase-7) parity exporter (e.g. `"BURNING_BLOOD"`).
    pub name: &'static str,
    pub rarity: RelicRarity,
    /// On-trigger effect AST. Consumed by the Phase-7 interpreter.
    pub effect: RelicEffect,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RelicSpec` is `Copy`-able when no allocation is involved. This
    /// is load-bearing for cheap instance materialisation in Phase 7.
    #[test]
    fn relic_spec_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<RelicSpec>();
        assert_copy::<RelicEffect>();
        assert_copy::<RelicRarity>();
    }

    /// Every [`RelicRarity`] variant is constructible (catches
    /// accidental field-style enum drift).
    #[test]
    fn relic_rarity_variants_are_distinct() {
        let all = [
            RelicRarity::Starter,
            RelicRarity::Common,
            RelicRarity::Uncommon,
            RelicRarity::Rare,
            RelicRarity::Shop,
            RelicRarity::Event,
            RelicRarity::Boss,
            RelicRarity::Ancient,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
