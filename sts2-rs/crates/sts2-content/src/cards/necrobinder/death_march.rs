//! Hand-authored (C#-first port). Necrobinder DeathMarch.
use super::prelude::*;

/// `DeathMarch` (Uncommon, 1 energy, Attack). Deals `CalculationBase +
/// ExtraDamage × N` where N is the number of cards drawn this turn by the
/// owner that were NOT drawn from the regular hand-draw (i.e. card-effect
/// draws). Base: `8 + 4×N`; upgraded: `9 + 6×N`.
//
// C# source (v0.107.1): ../sts2-wiki/decompiled/v0.107.1/MegaCrit.Sts2.Core.Models.Cards/DeathMarch.cs
//   CanonicalVars = [ CalculationBaseVar(8), ExtraDamageVar(4),
//     CalculatedDamageVar(Move).WithMultiplier(_ =>
//       History.Entries.OfType<CardDrawnEntry>().Count(e =>
//         e.HappenedThisTurn(cs) && e.Actor == owner.Creature && !e.FromHandDraw)) ]
//   OnPlay  -> DamageCmd.Attack(CalculatedDamage).Targeting(target)
//   OnUpgrade -> CalculationBase 8->9 (the spec base_damage upgrade delta),
//                ExtraDamage      4->6 (gated below on current_card_upgraded).
//
// CalculatedVar semantics (CalculatedVar.cs): value = Base + Extra × multiplier.
// Here Base = CalculationBase (spec `base_damage`, upgraded via CardUpgradeDelta),
// Extra = ExtraDamage, multiplier = non-hand-draw count this turn.
pub(crate) fn play_death_march(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let t = target.expect("DeathMarch: Enemy-targeted");

    let dmg = resolving_preview(state, preview_death_march, target);
    let _ = deal_damage_player_to_enemy(state, t, dmg);
    EffectResult::Resolved
}

/// `CalculatedDamage`: base + ExtraDamage per card-effect draw this turn.
/// Shared with the card face — see `crate::cards::preview`.
///
/// CalculationBase: `card.base_damage` already reflects the OnUpgrade 8->9 bump
/// via the spec's CardUpgradeDelta. ExtraDamage per non-hand-draw: 4 base, 6
/// upgraded (OnUpgrade var bump).
///
/// Multiplier N: player card-effect (non-hand) draws THIS turn — C#
/// `History.Entries.OfType<CardDrawnEntry>().Count(e => e.HappenedThisTurn &&
/// e.Actor == owner && !e.FromHandDraw)`. The engine tallies these on the single
/// `draw_via_state` chokepoint into `cards_drawn_by_effect_this_turn` (the
/// turn-start hand draw routes through `draw_into_hand` and is excluded).
/// DeathMarch itself is an Attack (no draw), so it never self-counts.
pub(crate) fn preview_death_march(
    state: &CombatState,
    card: &Card,
    _target: Option<CreatureUid>,
) -> i32 {
    let extra = if card.upgraded { 6 } else { 4 };
    let non_hand_draws_this_turn = i32::from(state.combat_log.cards_drawn_by_effect_this_turn);
    card.base_damage + extra * non_hand_draws_this_turn
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::DeathMarch,
        name: "DEATH_MARCH",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 8, // CalculationBaseVar(8)
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade: CalculationBase 8->9 (here, the spec base). ExtraDamage 3->4
    // is gated in the handler on `current_card_upgraded`, not a stat field.
    Some(CardUpgradeDelta {
        cost: None,
        base_damage: Some(9),
        base_block: None,
        keywords: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mint(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
        let mut card = crate::cards::spec_index::card_pod_from_id(id, false).expect("spec exists");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        uid
    }

    /// Bead `sts2-rs-4wppb` follow-up: Pagestorm's extra draw is a C#
    /// `fromHandDraw = false` `CardPileCmd.Draw`, so it counts toward
    /// `cards_drawn_by_effect_this_turn` (the multiplier DeathMarch's
    /// `CalculatedDamageVar` consumes). The rider previously routed through
    /// `draw_into_hand` (the turn-start hand-draw path) and was excluded,
    /// undercounting DeathMarch (sweep seeds 53/412).
    #[test]
    fn pagestorm_extra_draw_counts_toward_death_march_multiplier() {
        let mut state = CombatState::default();
        // A live player: a dead one makes the combat ending, so draws move no card.
        state.player.current_hp = 80;
        // Player needs a mirror creature so the draw/power accessors resolve.
        state.player.energy = 3;

        // Ethereal keyword is a runtime-populated registry; install the
        // default registry (walks every spec slice, registering ETHEREAL)
        // so `card_ethereal(Defy)` resolves in this isolated unit test.
        crate::cards::dispatch::install_default_registry();

        // Pagestorm(1): each Ethereal card drawn pulls 1 more card.
        apply_power_to_player(&mut state, PowerKind::Pagestorm, 1);

        // Draw pile (FIFO from front): an Ethereal Defy that triggers the
        // Pagestorm rider, plus a plain card for the rider to pull.
        let defy = mint(&mut state, CardId::Defy);
        let filler = mint(&mut state, CardId::StrikeNecrobinder);
        state.player.draw.push(defy);
        state.player.draw.push(filler);

        // A single card-effect draw: pulls Defy (counted) → Pagestorm pulls
        // the filler (also a `fromHandDraw = false` draw, so also counted).
        sts2_core::cards::draw_via_state(&mut state, 1);

        assert_eq!(
            state.combat_log.cards_drawn_by_effect_this_turn, 2,
            "Defy + the Pagestorm-triggered filler are both non-hand draws"
        );
    }

    /// C# `CardPileCmd.Draw` awaits `Hook.AfterCardDrawn` per card, so
    /// Pagestorm's nested draw (`fromHandDraw = false`) runs INSIDE the
    /// turn-start hand draw, right after the Ethereal card lands, and takes a
    /// hand slot before the hand draw's next card. With the 10-card cap
    /// cutting the hand draw short, C# records one hand draw fewer and one
    /// card-effect draw more than a batch draw that takes all its hand draws
    /// first (loadout Necrobinder 1190: DeathMarch hit for 28 instead of 32).
    #[test]
    fn pagestorm_draw_inside_capped_hand_draw_counts_toward_death_march() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        crate::cards::dispatch::install_default_registry();
        apply_power_to_player(&mut state, PowerKind::Pagestorm, 1);
        // 7 cards already in hand: room for 3 more of the 5-card hand draw.
        for _ in 0..7 {
            let uid = mint(&mut state, CardId::StrikeNecrobinder);
            state.player.hand.push(uid);
        }
        let defy = mint(&mut state, CardId::Defy);
        let pulled = mint(&mut state, CardId::DefendNecrobinder);
        let next = mint(&mut state, CardId::StrikeNecrobinder);
        state.player.draw.push(defy);
        state.player.draw.push(pulled);
        state.player.draw.push(next);
        for _ in 0..2 {
            let uid = mint(&mut state, CardId::StrikeNecrobinder);
            state.player.draw.push(uid);
        }

        // Hand draw: Defy (hand draw) -> Pagestorm pulls `pulled` (card
        // effect) -> `next` (hand draw) fills the hand to 10.
        state.draw_into_hand(5);

        assert_eq!(state.player.hand.len(), 10);
        assert_eq!(&state.player.hand[7..], &[defy, pulled, next]);
        assert_eq!(
            state.combat_log.cards_drawn_by_effect_this_turn, 1,
            "the Pagestorm draw is the turn's one card-effect draw"
        );
    }

    fn mint_into_play(state: &mut CombatState, upgraded: bool) -> sts2_core::CardUid {
        let mut card =
            crate::cards::spec_index::card_pod_from_id(CardId::DeathMarch, upgraded).expect("spec");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.play.push(uid);
        uid
    }

    fn enemy_hp(state: &CombatState, uid: CreatureUid) -> i32 {
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .expect("enemy present")
            .current_hp
    }

    /// v0.107.1 ExtraDamage 4 (was 3) / upgraded 6 (was 4). With 2 non-hand
    /// draws this turn: base 8 + 4×2 = 16, upgraded 9 + 6×2 = 21.
    #[test]
    fn death_march_extra_damage_scales_per_v107() {
        let mut state = CombatState::default();
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        state.combat_log.cards_drawn_by_effect_this_turn = 2;
        mint_into_play(&mut state, false);
        let hp0 = enemy_hp(&state, enemy);
        play_death_march(&mut state, Some(enemy));
        assert_eq!(hp0 - enemy_hp(&state, enemy), 16, "8 + 4×2 = 16");

        let mut up_state = CombatState::default();
        let up_enemy = up_state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        up_state.combat_log.cards_drawn_by_effect_this_turn = 2;
        mint_into_play(&mut up_state, true);
        let up_hp0 = enemy_hp(&up_state, up_enemy);
        play_death_march(&mut up_state, Some(up_enemy));
        assert_eq!(up_hp0 - enemy_hp(&up_state, up_enemy), 21, "9 + 6×2 = 21");
    }
}
