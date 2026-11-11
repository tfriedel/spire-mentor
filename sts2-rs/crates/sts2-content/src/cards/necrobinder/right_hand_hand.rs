//! Right Hand Hand (Necrobinder). Hand-authored C#-first port via the
//! character-port skill.
//! C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/RightHandHand.cs
use super::prelude::*;

/// Right Hand Hand (Uncommon, 0 energy, Attack, AnyEnemy). The Osty attacks
/// the target for 4 (6 upgraded). Then, whenever you play a card that spends
/// 2 or more energy while Right Hand Hand is in your discard pile, it returns
/// to your hand.
//
// C# CanonicalVars:
//   OstyDamageVar(4m, Move)  → base_damage 4 (cosmetic metadata; the hit is
//                              dealt by the Osty, not the player). OnUpgrade +2 → 6.
//   EnergyVar(2)             → the energy threshold for the return-to-hand
//                              trigger (`base.DynamicVars.Energy.IntValue` == 2).
//                              NOT touched by OnUpgrade.
// CanonicalTags: OstyAttack (cosmetic glow tag; not carried in CardSpec).
// ShouldGlowRedInternal => Owner.IsOstyMissing — pure UI, ignored.
//
// C# OnPlay:
//   if (!Osty.CheckMissingWithAnim(Owner))
//       DamageCmd.Attack(OstyDamage.BaseValue)   // 4, +2 on upgrade → 6
//           .FromOsty(Owner.Osty, this)          // attacker is the OSTY
//           .Targeting(cardPlay.Target).Execute(...);
//
// `deal_damage_from_osty_to_enemy` is itself a no-op when the Osty is absent
// (`osty_dealer_ref` → None), so calling it unconditionally reproduces
// `CheckMissingWithAnim` (minus the cosmetic summon-prompt animation).
pub(crate) fn play_right_hand_hand(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let t = target.expect("Right Hand Hand: enemy target required");

    // OstyDamage: 4 base, +2 on upgrade. Computed here (the OstyDamage is not a
    // plain player-attack stat), gated on current_card_upgraded like the sibling
    // OstyAttack cards (sic_em/poke/unleash/bone_shards).
    let dmg = if current_card_upgraded(state) { 6 } else { 4 };
    deal_damage_from_osty_to_enemy(state, t, dmg);
    EffectResult::Resolved
}

/// Energy threshold (`EnergyVar(2)`) for the return-to-hand trigger.
const RIGHT_HAND_HAND_ENERGY_THRESHOLD: i32 = 2;

/// Late-effect for `RightHandHand` — fired by
/// [`card_effects::fire_late_effects`] after every card play.
///
/// C# `AfterCardPlayedLate`:
/// [decompiled C# omitted — see the cited path]
///
/// Single-player port: the owner check is elided (one player). `EnergyValue`
/// equals the energy spent on the triggering play (`PlayCardAction` sets
/// `EnergyValue = EnergySpent = item`), which the engine stamps into the
/// played card's `energy_spent` field. The watched Right Hand Hand returns to
/// hand only while it sits in the **discard** pile.
pub(crate) fn right_hand_hand_late_effect(
    state: &mut CombatState,
    watched_uid: sts2_core::CardUid,
    played_ctx: &sts2_core::hooks::CardCtx,
) {
    // The triggering play must spend >= 2 energy. Read the played card's
    // stamped energy_spent (== C# ResourceInfo.EnergyValue).
    let Some(played_uid) = played_ctx.card_uid else {
        return;
    };
    let played_idx = played_uid.0.get() as usize - 1;
    let Some(played) = state.cards.get(played_idx) else {
        return;
    };
    if sts2_core::cards::play_energy_value(played) < RIGHT_HAND_HAND_ENERGY_THRESHOLD {
        return;
    }

    // Only return to hand while in the discard pile (C# Pile.Type == Discard).
    if !state.player.discard.contains(&watched_uid) {
        return;
    }
    // `CardPileCmd.Add(this, Hand)` (RightHandHand.cs:51) refuses a combat
    // pile once the combat is ending (CardPileCmd.cs:312).
    if sts2_core::combat_loop::combat_is_ending(state) {
        return;
    }
    let _moved = card_effects::scan_piles_and_move_to_hand(
        state,
        state
            .player
            .creature_uid
            .unwrap_or_else(|| CreatureUid::new(1).expect("non-zero")),
        &[PileKind::Discard],
        |c| c.uid == watched_uid,
    );
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::RightHandHand,
        name: "RIGHT_HAND_HAND",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy, // C# TargetType.AnyEnemy → single-player Enemy
        rarity: CardRarity::Uncommon,
        base_damage: 4, // OstyDamageVar(4m) — cosmetic; dealt via the Osty
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: OstyDamage.UpgradeValueBy(2) (4 → 6). The OstyDamage is not
    // a plain player-attack stat (it is dealt by the Osty), so the bump is
    // applied in the handler via current_card_upgraded rather than a static spec
    // delta. Returns None to keep one source of truth. (EnergyVar is unchanged.)
    None
}
