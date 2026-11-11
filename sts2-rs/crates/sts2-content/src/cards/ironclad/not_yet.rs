//! Hand-authored (C#-first port). Keep the prelude import.
use super::prelude::*;

/// NotYet (Rare, 2 energy, Skill, Self, Exhaust). Heal the player by 10
/// (upgrade +3 → 13).
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/NotYet.cs
//
// C# `OnPlay`:
//   `CreatureCmd.TriggerAnim(Owner.Creature, "Cast", …)` — cosmetic, ignored.
//   `CreatureCmd.Heal(Owner.Creature, Heal.BaseValue)` — heal the player by 10.
// `CanonicalVars` = `{ HealVar(10m) }`; `CanonicalKeywords` = `{ Exhaust }`;
// `CanBeGeneratedInCombat => false` (reward/shop pool only — mirrored by the
// reward-pool eligibility filter, not a combat-generation gate here).
// `OnUpgrade`: `Heal.UpgradeValueBy(3m)` (10 → 13) — a magnitude bump, so it
// is gated on `current_upgraded` and `upgrade()` returns `None`.
pub(crate) fn play_not_yet(state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    let heal = if current_upgraded(state) { 13 } else { 10 };
    if let Some(player) = state.player.creature_uid {
        let _ = state.creature_heal(player, heal);
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::NotYet,
        name: "NOT_YET",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade bumps the HealVar (10 → 13); applied in the handler via the
    // `current_upgraded` branch, so no stat-field delta here.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a state whose player mirror is at `cur`/`max` HP with a single
    /// NotYet card on the Play pile (`current_upgraded` reads the top of Play
    /// and panics on an empty pile). The heal mirrors back to
    /// `player.current_hp` via `player.creature_uid`.
    fn state_with_player(cur: i32, max: i32, upgraded: bool) -> CombatState {
        let mut state = CombatState::default();
        state.player.current_hp = cur;
        state.player.max_hp = max;
        let _ = state.init_player_creature();
        state.sync_player_creature();
        let uid = sts2_core::CardUid::new(1).expect("nz");
        state.cards.push(sts2_core::cards::Card {
            uid,
            card_id: CardId::NotYet,
            cost: sts2_core::CardEnergyCost::new(2),
            card_type: CardType::Skill,
            target_type: TargetType::Self_,
            upgraded,
            ..sts2_core::cards::Card::pod_defaults()
        });
        state.player.play.push(uid);
        state
    }

    #[test]
    fn base_heals_ten_and_clamps_to_max() {
        let mut state = state_with_player(20, 80, false);
        assert_eq!(play_not_yet(&mut state, None), EffectResult::Resolved);
        assert_eq!(state.player.current_hp, 30, "base NotYet heals 10");

        // Clamp: healing past max HP stops at max.
        let mut full = state_with_player(75, 80, false);
        let _ = play_not_yet(&mut full, None);
        assert_eq!(full.player.current_hp, 80, "heal clamps to max HP");
    }

    #[test]
    fn upgraded_heals_thirteen() {
        let mut state = state_with_player(20, 80, true);
        let _ = play_not_yet(&mut state, None);
        assert_eq!(state.player.current_hp, 33, "upgraded NotYet heals 13");
    }
}
