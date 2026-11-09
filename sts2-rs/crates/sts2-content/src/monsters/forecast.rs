//! Forecast intents for moves whose numbers the game computes when it
//! telegraphs them (#1181).
//!
//! `MonsterAi::forecast_after_current` copies each predicted move's intents
//! from the move table. For most moves that is the telegraph. Three moves
//! carry a C# lambda instead, and the Rust port keeps their live number
//! elsewhere (a power counter, a value MIASMA writes back into the table):
//!
//! - TestSubject `MULTI_CLAW`: `MultiAttackIntent(MultiClawDamage, () =>
//!   MultiClawTotalCount)`, `BaseMultiClawCount + ExtraMultiClawCount`, and
//!   every perform adds one (`TestSubject.cs:162,198,303`). The engine keeps
//!   the extra count in the owner's PainfulStabs counter and adds it when it
//!   telegraphs (`combat_loop::roll_next_enemy_intents`).
//! - TheForgotten `DREAD`: `SingleAttackIntent(() => DreadDamage)`, the
//!   ascension base plus the owner's Dexterity (`TheForgotten.cs:24-29,46`),
//!   and every MIASMA gives it 2 Dexterity (`:57-59`).
//! - WaterfallGiant `EXPLODE`: `DeathBlowIntent(() => SteamEruptionDamage)`,
//!   which ABOUT_TO_BLOW sets from the owner's SteamEruption stacks
//!   (`WaterfallGiant.cs:283-289`); the table holds a placeholder 0 until then.
//!
//! Three more moves rewrite their own table entry for the next use when
//! they are performed, so a forecast past another use of the same move is one
//! ramp behind: WaterfallGiant `PRESSURE_GUN` (+5 damage), Aeonglass
//! `INCREASING_INTENSITY_MOVE` (+1 Strength) and Entomancer `PHEROMONE_SPIT`
//! (the Strength a hive the spit grows decides).
//!
//! [`forecast_step_intents`] gives the number the game will show at a
//! forecast step, assuming every move before it (the current telegraph, then
//! the earlier steps) is performed. Nothing else on the board is projected.

use sts2_core::Intent;
use sts2_core::{CombatState, CreatureUid, MonsterKind, PowerKind};

use smallvec::SmallVec;

/// C# `TheForgotten.DebilitatingSmogDexStealAmount` (2 at every ascension).
const MIASMA_DEX_GAIN: i32 = 2;

fn owner_power(
    state: &CombatState,
    owner: CreatureUid,
    kind: PowerKind,
) -> Option<&sts2_core::Power> {
    let c = state.creatures.iter().find(|c| c.id == owner.0)?;
    c.powers
        .iter()
        .find_map(|pu| state.powers.iter().find(|p| p.id == pu.0 && p.kind == kind))
}

/// The intents a forecast step of `owner` shows for `move_id`, given the
/// move table's `intents` and the moves performed before it (`prior`: the
/// current telegraph first, then the earlier forecast steps). Returns the
/// table's intents unchanged for every move but the six this module
/// covers.
#[must_use]
pub fn forecast_step_intents(
    state: &CombatState,
    owner: CreatureUid,
    prior: &[&str],
    move_id: &str,
    intents: &[Intent],
) -> SmallVec<[Intent; 4]> {
    let mut out: SmallVec<[Intent; 4]> = intents.iter().copied().collect();
    let Some(kind) = state
        .creatures
        .iter()
        .find(|c| c.id == owner.0)
        .and_then(|c| c.species)
    else {
        return out;
    };
    let performed = |id: &str| prior.iter().filter(|m| **m == id).count();
    match (kind, move_id) {
        (MonsterKind::TestSubject, "MULTI_CLAW") => {
            // No PainfulStabs: the engine adds nothing either (the power
            // brackets the second form, the only one MULTI_CLAW is in).
            let Some(stabs) = owner_power(state, owner, PowerKind::PainfulStabs) else {
                return out;
            };
            let extra = i64::from(stabs.counter) + performed("MULTI_CLAW") as i64;
            for intent in &mut out {
                if let Intent::MultiAttack { hits, .. } = intent {
                    let total = i64::from(*hits) + extra.max(0);
                    *hits = u8::try_from(total).unwrap_or(u8::MAX);
                }
            }
        }
        (MonsterKind::TheForgotten, "DREAD") => {
            let base = crate::ascension_scaling::scaled_damage(
                state.ascension,
                state.game_version,
                MonsterKind::TheForgotten,
                "DREAD",
                13,
            );
            let dex = owner_power(state, owner, PowerKind::Dexterity).map_or(0, |p| p.amount);
            let gain = MIASMA_DEX_GAIN * i32::try_from(performed("MIASMA")).unwrap_or(i32::MAX / 4);
            for intent in &mut out {
                if let Intent::Attack { damage } = intent {
                    *damage = base + dex + gain;
                }
            }
        }
        (MonsterKind::WaterfallGiant, "EXPLODE") if performed("ABOUT_TO_BLOW") > 0 => {
            // ABOUT_TO_BLOW captures the stacks the giant holds when it
            // performs; once it has, the table already holds that number.
            let captured = super::act1_underdocks::waterfall_giant_explode_damage(state, owner);
            for intent in &mut out {
                if let Intent::DeathBlow { damage } = intent {
                    *damage = captured;
                }
            }
        }
        (MonsterKind::WaterfallGiant, "PRESSURE_GUN") => {
            // `CurrentPressureGunDamage += PressureGunIncrease (5)` after each
            // hit (`waterfall_giant_pressure_gun_effect`); the table holds
            // the next one.
            let ramp = 5 * i32::try_from(performed("PRESSURE_GUN")).unwrap_or(0);
            for intent in &mut out {
                if let Intent::Attack { damage } = intent {
                    *damage = damage.saturating_add(ramp);
                }
            }
        }
        (MonsterKind::Aeonglass, "INCREASING_INTENSITY_MOVE") => {
            // `IncreasingIntensityTotalStrength = base + AdditionalStrength`,
            // `AdditionalStrength++` per use (`Aeonglass.cs:78,154-155`); the
            // table holds the next use's.
            let ramp = i32::try_from(performed("INCREASING_INTENSITY_MOVE")).unwrap_or(0);
            for intent in &mut out {
                if let Intent::Buff {
                    kind: PowerKind::Strength,
                    amount,
                } = intent
                {
                    *amount = amount.saturating_add(ramp);
                }
            }
        }
        (MonsterKind::Entomancer, "PHEROMONE_SPIT") => {
            // The Strength depends on the hive, which a 1-Strength spit grows
            // (`entomancer_pheromone_spit_effect`).
            use super::act2_hive::{
                entomancer_personal_hive_amount, entomancer_spit_strength_for_hive,
            };
            let mut hive = entomancer_personal_hive_amount(state, owner);
            for _ in 0..performed("PHEROMONE_SPIT") {
                if entomancer_spit_strength_for_hive(state.game_version, hive) == 1 {
                    hive += 1;
                }
            }
            let strength = entomancer_spit_strength_for_hive(state.game_version, hive);
            for intent in &mut out {
                if let Intent::Buff {
                    kind: PowerKind::Strength,
                    amount,
                } = intent
                {
                    *amount = strength;
                }
            }
        }
        _ => {}
    }
    out
}
