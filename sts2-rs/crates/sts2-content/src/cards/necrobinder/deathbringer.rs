//! Deathbringer (Necrobinder pool). Hand-authored C#-first port.
use super::prelude::*;

/// Deathbringer (Uncommon, 2 energy, Skill, AllEnemies). Apply 21 Doom and
/// 1 Weak to ALL enemies. C# `Deathbringer.cs` OnPlay:
/// `PowerCmd.Apply<DoomPower>(HittableEnemies, 21)` +
/// `PowerCmd.Apply<WeakPower>(HittableEnemies, 1)`.
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Deathbringer.cs
pub(crate) fn play_deathbringer(
    state: &mut CombatState,
    _target: Option<CreatureUid>,
) -> EffectResult {
    // C# DoomVar base 21; OnUpgrade Doom.UpgradeValueBy(5) → 26. PowerVar bump,
    // so gate on the upgraded check (no central-table / spec stat involvement).
    let doom = if current_card_upgraded(state) { 26 } else { 21 };
    // C# WeakVar base 1, no OnUpgrade bump.
    let weak = 1;
    // C# applies Doom to every `CombatState.HittableEnemies`, then Weak to
    // every one: two `PowerCmd.Apply` calls, each over a fresh list
    // (Deathbringer.cs). A per-enemy Doom+Weak pair reorders the debuff
    // reactions: Sleight of Flesh's Weak hit killing one Kaiser Crab arm
    // enrages the other before its Doom hit lands. Snapshot uids per apply
    // and recheck alive per target, mirroring deal_damage_all_enemies.
    for (kind, amount) in [(PowerKind::Doom, doom), (PowerKind::Weak, weak)] {
        let uids: Vec<CreatureUid> = state.enemy_uids.iter().copied().collect();
        for uid in uids {
            if state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
            {
                apply_power_to_enemy(state, uid, kind, amount);
            }
        }
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Deathbringer,
        name: "DEATHBRINGER",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade only bumps the DoomPower amount (21 → 26), which is a PowerVar
    // bump handled in play_deathbringer via current_card_upgraded — no stat delta.
    None
}
