//! Misery — Necrobinder rare attack.
//! C# source:
//! ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Misery.cs
use super::prelude::*;

/// Misery (Rare, 0 energy, Attack). Snapshot the target's debuffs, deal 7
/// damage to it, then copy each of those debuffs (by amount) onto every other
/// hittable enemy.
///
/// C# `Misery.cs::OnPlay`:
/// 1. `originalDebuffs = target.Powers.Where(p.TypeForCurrentAmount ==
///    PowerType.Debuff).Select(ClonePreservingMutability)` — snapshot taken
///    **before** the damage hit.
/// 2. `DamageCmd.Attack(7).Targeting(target)`.
/// 3. For each `HittableEnemies` except the target, for each snapshotted
///    debuff: if the enemy already has that power (and it's not instanced),
///    `PowerCmd.ModifyAmount(+item.Amount)`; otherwise `PowerCmd.Apply(copy,
///    enemy, item.Amount)`.
///
/// The Rust power model sums by `PowerKind`, so `apply_power_to_enemy(kind,
/// amount)` covers both the modify-existing and apply-fresh C# branches in
/// one call — but only for positive amounts; a snapshotted negative-attribute
/// debuff (see below) is copied via `apply_power_signed_to_creature` since the
/// positive-only helper early-outs on `amount <= 0`.
/// `DoHackyThingsForSpecificPowers`/`ITemporaryPower.IgnoreNextInstance` has no
/// Rust analogue — per `colorless/rend.rs`, `ITemporaryPower` is implemented
/// only by Temporary{Strength,Focus,Dexterity}, none of which are debuffs. The
/// HitFx is cosmetic and dropped. OnUpgrade (Damage 7->9, +Retain) lives in the
/// central `card_upgrades_generated.rs` table; the damage is read via
/// `current_damage`.
///
/// Bead `sts2-rs-rend-misery-signed-debuff-llzzt`: the debuff snapshot uses C#
/// `TypeForCurrentAmount` (`Misery.cs:28`), which is sign-sensitive — a
/// drained Strength/Dexterity/Focus (negative `AllowNegative` Counter attr)
/// reads as a Debuff. Filter via `is_debuff_for_amount`, not the sign-agnostic
/// `classify`, matching `colorless/rend.rs`.
pub(crate) fn play_misery(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Misery: Enemy-targeted");

    // 1. Snapshot the target's debuffs (kind + amount + C# `Applier`) before
    // the hit. A corpse has none: C# `CreatureCmd.Kill` strips it with
    // `RemoveAllPowersAfterDeath`, keeping only a few non-debuff survivors
    // (Minion, Adaptable, ...). Rust keeps corpse powers, so a replay of a
    // Misery that killed its target copied them again (random-loadout
    // Necrobinder 7600 v0.111.0: the Rock's Imbalanced, 2 instead of 1).
    let original_debuffs: Vec<Debuff> = state
        .creatures
        .iter()
        .find(|c| c.id == t.0 && c.current_hp > 0)
        .map(|c| {
            c.powers
                .iter()
                .filter_map(|p_uid| state.powers.get_by_id(p_uid.0))
                .filter(|p| sts2_core::afflictions::is_debuff_for_amount(p.kind, p.amount))
                .map(|p| (p.kind, p.amount, foreign_applier(state, c, p)))
                .collect()
        })
        .unwrap_or_default();
    // C# `TemporaryStrengthPower.BeforeApplied` applies the Strength BEFORE the
    // temporary power itself lands (PowerCmd.cs:131-135), so in C#'s
    // `Powers` list the Strength precedes e.g. ShacklingPotion; the Rust
    // ports apply the temporary power first. Restore C#'s copy order — it
    // decides which copy an Artifact on the receiving enemy negates
    // (Necrobinder full seed 2643).
    let mut original_debuffs = original_debuffs;
    original_debuffs
        .sort_by_key(|(k, _, _)| u8::from(sts2_core::afflictions::is_temporary_power_debuff(*k)));

    // 2. Deal the attack damage to the target.
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));

    // v0.111.0 (decomp Misery.cs:27-66, GH #793): temporary debuffs fold
    // their amount into the power they apply internally, and zero amounts
    // are skipped — see `copy_debuffs_v0111`.
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        copy_debuffs_v0111(state, t, &original_debuffs);
        return EffectResult::Resolved;
    }
    // 3. Copy each snapshotted debuff onto every other hittable enemy.
    if !original_debuffs.is_empty() {
        let others: Vec<CreatureUid> = state
            .active_enemy_uids()
            .into_iter()
            .filter(|uid| *uid != t)
            .collect();
        for enemy in others {
            let alive = state
                .creatures
                .iter()
                .any(|c| c.id == enemy.0 && c.current_hp > 0);
            if !alive {
                continue;
            }
            for &(kind, amount, applier) in &original_debuffs {
                copy_debuff(state, enemy, kind, amount, applier);
            }
        }
    }

    EffectResult::Resolved
}

/// One snapshotted target debuff: kind, amount, and the C# `item.Applier`
/// when that is NOT the player (see [`foreign_applier`]).
type Debuff = (PowerKind, i32, Option<CreatureUid>);

/// The creature that applied power `p` on `owner`, when it is not the player
/// — C# Misery copies with `item.Applier` (`Misery.cs`: `PowerCmd.Apply(...,
/// enemy, item.Amount, item.Applier, this)`), and Unsettling Lamp captures and
/// doubles only an apply whose applier is its owner (`UnsettlingLamp.cs`
/// `BeforePowerAmountChanged`: `applier != base.Owner.Creature` → return).
///
/// Rust records `Power.applier` only on the applier-threaded surface; the
/// spawn self-powers (BowlbugRock's `ImbalancedPower`, `BowlbugRock.cs:66`
/// `PowerCmd.Apply<ImbalancedPower>(..., base.Creature, 1m, base.Creature,
/// null)`) carry `None` there, so a debuff in the owner's
/// `on_spawn_self_buffs` list is the owner's own. Anything else unrecorded is
/// a player card / potion debuff.
fn foreign_applier(
    state: &CombatState,
    owner: &sts2_core::Creature,
    p: &sts2_core::Power,
) -> Option<CreatureUid> {
    let player = state.player.creature_uid;
    match p.applier {
        Some(uid) if Some(uid) != player => Some(uid),
        Some(_) => None,
        None => owner
            .species
            .is_some_and(|s| s.on_spawn_self_buffs().iter().any(|(k, _)| *k == p.kind))
            .then_some(CreatureUid(owner.id)),
    }
}

/// Copy one snapshotted debuff onto `enemy` — C#'s `ModifyAmount` / `Apply`
/// branch with the original applier.
///
/// A copy of a monster's own debuff (`applier` = that monster) cannot capture
/// Unsettling Lamp's trigger, and it is not a player-applied debuff for
/// Sleight of Flesh. It IS doubled once an earlier player-applied copy of the
/// same play captured the Lamp: the doubling arm checks only the card source
/// (GH #1150, `relic_effects::helpers::unsettling_lamp_doubles`). Rust treated every copy as
/// player-applied: Misery copying BowlbugRock's Imbalanced 1 onto its Nectar
/// sibling under the Lamp landed Imbalanced 2 and burnt the Lamp's trigger
/// (loadout Necrobinder 1096 v0.107.1 act2; C# Imbalanced 1).
///
/// A snapshotted debuff can be a negative-amount attribute power (drained
/// Strength reads as a Debuff via `is_debuff_for_amount`), which
/// `apply_power_to_enemy` rejects; the signed helper applies it, after the
/// Lamp's Before / Given passes when the player applied the original.
fn copy_debuff(
    state: &mut CombatState,
    enemy: CreatureUid,
    kind: PowerKind,
    amount: i32,
    applier: Option<CreatureUid>,
) {
    match (amount < 0, applier) {
        // A monster-applied copy is not the player's apply: Sleight of Flesh
        // (`applier == base.Owner`, SleightOfFleshPower.cs:22) stays quiet, so
        // skip the player-drain reactors (loadout Necrobinder 1210-v0.107.1:
        // Sleight of Flesh hit Crusher for Rocket's own drained Strength -7).
        (true, Some(_)) => {
            super::helpers::apply_power_signed_to_creature_raw(state, enemy, kind, amount);
        }
        (true, None) => {
            let amount =
                sts2_core::card_effects::fold_player_card_signed_apply(state, enemy, kind, amount);
            apply_power_signed_to_creature(state, enemy, kind, amount);
        }
        (false, Some(_)) => sts2_core::card_effects::apply_power_to_creature_with_applier(
            state, enemy, kind, amount, applier,
        ),
        (false, None) => apply_power_to_enemy(state, enemy, kind, amount),
    }
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Misery,
        name: "MISERY",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 7,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade (Damage 7->9, AddKeyword(Retain)) lives in the central
    // `card_upgrades_generated.rs` table (CardId::Misery row); per-card
    // delta is None.
    None
}

/// v0.111.0 Misery copy step (decomp Misery.cs:27-66, GH #793):
///
/// [decompiled C# omitted — see the cited path]
///
/// Every temporary DEBUFF power in this build is a `TemporaryStrengthPower`
/// subclass (`afflictions::is_temporary_power_debuff`), so its internally
/// applied power is Strength. A drained Strength `-N` that came from such a
/// power reads `-N + N = 0` after the fold and is skipped; the temporary
/// power is then applied normally and its `BeforeApplied` takes the `N`
/// Strength off again (applied explicitly here, the PiercingWail pattern).
/// v0.107.1 instead copied both and suppressed the second Strength loss
/// (`IgnoreNextInstance`).
fn copy_debuffs_v0111(state: &mut CombatState, t: CreatureUid, original: &[Debuff]) {
    use sts2_core::afflictions::is_temporary_power_debuff;
    let mut amounts: Vec<Debuff> = original.to_vec();
    for i in 0..amounts.len() {
        let (kind, amount, _) = amounts[i];
        if is_temporary_power_debuff(kind) {
            if let Some(slot) = amounts
                .iter_mut()
                .find(|(k, _, _)| *k == PowerKind::Strength)
            {
                slot.1 += amount;
            }
        }
    }
    if amounts.is_empty() {
        return;
    }
    let others: Vec<CreatureUid> = state
        .active_enemy_uids()
        .into_iter()
        .filter(|uid| *uid != t)
        .collect();
    for enemy in others {
        let alive = state
            .creatures
            .iter()
            .any(|c| c.id == enemy.0 && c.current_hp > 0);
        if !alive {
            continue;
        }
        for &(kind, amount, applier) in &amounts {
            if amount == 0 {
                continue;
            }
            if is_temporary_power_debuff(kind) {
                // `TemporaryStrengthPower.BeforeApplied` (IsPositive => false):
                // `Strength -amount` (the modified amount) before the
                // temporary power itself lands. Only cards and potions apply
                // these, so the applier is the player.
                sts2_core::card_effects::apply_temporary_strength_debuff(
                    state, enemy, kind, amount,
                );
            } else {
                copy_debuff(state, enemy, kind, amount, applier);
            }
        }
    }
}
