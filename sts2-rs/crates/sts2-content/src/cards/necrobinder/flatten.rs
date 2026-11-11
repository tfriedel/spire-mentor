//! Flatten (Necrobinder). Hand-authored C#-first port via the character-port skill.
//! C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Flatten.cs
use super::prelude::*;

/// Flatten (Common, 2 energy, Attack, AnyEnemy). The Osty attacks the target
/// for 12 (16 upgraded). OstyAttack card (cosmetic glow tag).
//
// C# OnPlay:
//   ArgumentNullException.ThrowIfNull(cardPlay.Target);
//   if (!Osty.CheckMissingWithAnim(Owner))
//       DamageCmd.Attack(OstyDamage.BaseValue)   // 12, +4 on upgrade → 16
//           .FromOsty(Owner.Osty, this)          // attacker is the OSTY
//           .Targeting(cardPlay.Target)
//           .WithHitFx(...)                       // cosmetic, ignored
//           .Execute(...);
//
// CanonicalVars:
//   OstyDamageVar(12m, Move)        → base_damage 12 (cosmetic metadata; the hit
//                                     is dealt by the Osty, not the player).
//                                     OnUpgrade +4 → 16.
// CanonicalTags: OstyAttack (cosmetic glow tag; not carried in CardSpec here).
// ShouldGlowGoldInternal => HasOstyAttackedThisTurn, ShouldGlowRedInternal =>
//   Owner.IsOstyMissing — pure UI, ignored.
//
// COST-REDUCTION MECHANIC (PORTED — bead `sts2-rs-flatten-osty-cost`):
//   Flatten overrides the card-side reactive hooks `AfterCardEnteredCombat` and
//   `AfterAttack`. When this card enters combat AND the Osty has already attacked
//   this turn, OR whenever the Owner's Osty performs any attack, it calls
//   `ReduceCost()` → `EnergyCost.SetThisTurn(0)` (cost drops to 0 for the rest of
//   the turn). `SetThisTurn(0)` has an end-of-turn lifetime and
//   `CombatLog::osty_attacks_this_turn` is monotonic within a turn (reset only at
//   turn start), so the two reactive hooks reduce to a single predicate —
//   "Osty attacked this turn" — evaluated at cost-read time. Implemented as a
//   base-0 rider in `hooks::effective_card_cost` (CardId::Flatten +
//   osty_attacks_this_turn > 0), mirroring the `FREE_THIS_TURN` rider. (Cost is
//   not affected by upgrade in C#.)
//
// `deal_damage_from_osty_to_enemy` is a no-op when the Osty is absent
// (`osty_dealer_ref` → None), reproducing `CheckMissingWithAnim` (minus the
// cosmetic summon-prompt animation we don't model).
pub(crate) fn play_flatten(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Flatten: enemy target required");

    // OstyDamage: 12 base, +4 on upgrade. Computed here (the OstyDamage is not a
    // plain player-attack stat), gated on current_card_upgraded like the sibling
    // OstyAttack cards (sic_em/poke/unleash/bone_shards). No-op if Osty missing.
    let dmg = if current_card_upgraded(state) { 16 } else { 12 };
    deal_damage_from_osty_to_enemy(state, t, dmg);
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Flatten,
        name: "FLATTEN",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy, // C# TargetType.AnyEnemy → single-player Enemy
        rarity: CardRarity::Common,
        base_damage: 12, // OstyDamageVar(12m) — cosmetic; dealt via the Osty
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: OstyDamage.UpgradeValueBy(4) (12 → 16). The OstyDamage is not
    // a plain attack stat (the hit is dealt by the Osty), so the bump is computed
    // in the handler via current_card_upgraded rather than a static spec delta.
    // Returns None to keep one source of truth, matching the sibling OstyAttack
    // cards (sic_em).
    None
}
