//! Hand-authored C#-first port. Keep the prelude import.
use super::prelude::*;

/// Neurosurge (Rare, 0 energy, Power, Self). Gain 3 energy now (upgraded: 4),
/// draw 2 cards, and apply 3 stacks of Neurosurge to self.
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Neurosurge.cs
//
// ctor: base(0, CardType.Power, CardRarity.Rare, TargetType.Self)
// CanonicalVars:
//   PowerVar<NeurosurgePower>(3) -> Neurosurge stacks applied to self (3)
//   EnergyVar(3)                 -> energy gained now           (OnUpgrade: +1 -> 4)
//   CardsVar(2)                  -> cards drawn (CardPileCmd.Draw)
// OnPlay:
//   PlayerCmd.GainEnergy(Energy.BaseValue, Owner)              -> gain 3 (up 4)
//   CardPileCmd.Draw(ctx, Cards.BaseValue, Owner)              -> draw 2
//   PowerCmd.Apply<NeurosurgePower>(Owner.Creature, NeurosurgePower=3, ...)
// OnUpgrade: DynamicVars.Energy.UpgradeValueBy(1) -> the gain becomes 4 (the
//   NeurosurgePower stack count and the draw count are NOT upgraded).
//
// NeurosurgePower (../sts2-wiki/.../Powers/NeurosurgePower.cs):
//   Type=Debuff, StackType=Counter. AfterSideTurnStart: when the owner's side
//   turn starts, applies Amount stacks of DoomPower to the owner. `PowerKind::
//   Neurosurge` has its AfterSideTurnStart listener wired in crate::hooks (apply
//   Doom == Amount on each of the owner's turn starts, bead `sts2-rs-4wppb`), so
//   the stacking-Doom behavior is live at runtime.
pub(crate) fn play_neurosurge(
    state: &mut CombatState,
    _target: Option<CreatureUid>,
) -> EffectResult {
    // PlayerCmd.GainEnergy(Energy.BaseValue, Owner). OnUpgrade bumps the gain
    // from 3 to 4 (Energy.UpgradeValueBy(1)).
    let gain = if current_card_upgraded(state) { 4 } else { 3 };
    state.gain_energy(gain);

    // CardPileCmd.Draw(ctx, Cards.BaseValue=2, Owner). Not upgraded.
    let draw_was_suspended = state.pending_draw_resume.is_some();
    let _ = sts2_core::cards::draw_via_state(state, 2);

    // PowerCmd.Apply<NeurosurgePower>(Owner.Creature, 3, ...): self-applied
    // Neurosurge (a Debuff/Counter) that, on each of the owner's turn starts,
    // applies its Amount in Doom to the owner. The NeurosurgePower stack count
    // is NOT upgraded (only the energy gain is), so it is always 3.
    crate::cards::draw_then::after_draw(state, draw_was_suspended, sts2_core::DrawThen::Neurosurge);

    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Neurosurge,
        name: "NEUROSURGE",
        cost: 0,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade: DynamicVars.Energy.UpgradeValueBy(1). The energy-gain bump is
    // an EnergyVar magnitude (not a damage/block/cost stat), so it is applied
    // in the handler via the `current_card_upgraded` branch and this returns
    // None (no spec-stat delta to record).
    None
}
