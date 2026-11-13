//! Hand-authored (C#-first port). Keep the prelude import.
use super::prelude::*;

/// Seance (Rare, 1 energy, Skill, Self). Ethereal. Choose `CardsVar`(1) card in
/// your draw pile and transform it into a Soul.
// C# source: ../sts2-wiki/decompiled/v0.107.1/MegaCrit.Sts2.Core.Models.Cards/Seance.cs
//
// C# `OnPlay` (v0.107.1):
//   1. CardSelectCmd.FromCombatPile(ctx, PileType.Draw.GetPile(Owner), Owner,
//          prefs(TransformSelectionPrompt, Cards.IntValue=1))      // select 1
//   2. CreatureCmd.TriggerAnim(Owner.Creature, "Cast", ...)        // cosmetic, ignored
//   3. foreach chosen: CardCmd.TransformTo<Soul>(item)
//
// CanonicalVars is a single `CardsVar(1)` (the number of cards to transform);
// `OnUpgrade` only does `EnergyCost.UpgradeBy(-1)` (1 → 0), so the transform
// count stays 1 at every upgrade level. The cost delta is a stat-field change,
// so it lives in `upgrade()` (cost Some(0)).
//
// The C# `CardSelectCmd.FromCombatPile(ctx, Draw.GetPile(Owner), owner,
// prefs(prompt, 1))` is a forced single-pick (min=max=1) over the draw pile.
// Like Begone / Guards / Transfigure, the select-cmd auto-applies (NO prompt)
// only when `cards.Count <= MinSelect` (i.e. ≤1 draw card); with 2+ draw cards
// it SURFACES a `card_select`. This surfaces the select as a
// [`sts2_core::ChoiceKind::TransformDrawCardToSoul`] single-pick choice over
// the draw pile; the resolver mints a fresh (un-upgraded) Soul and transforms
// the chosen card. The ≤1-card path is kept inline as a faithful auto-apply.
//
// `CardCmd.TransformTo<Soul>` materializes a fresh, un-upgraded Soul (the
// transform does not carry the source card's upgrade flag), so the resolver
// mints `CardId::Soul` with `upgraded = false`.
//
// v0.107.1 bead 1a.12: `FromSimpleGrid(orderby c.Rarity, c.Id …)` →
// `FromCombatPile(PileType.Draw.GetPile(Owner) …)`. The `(rarity, id)` sort is
// NOT gone for the SURFACED screen: `CardSelectCmd.FromCombatPile`
// (`CardSelectCmd.cs` @ :401-409, the `Selector` branch) re-applies
// `orderby c.Rarity, c.Id` for any `PileType.Draw` pile. Only the auto-resolve
// fast path (`!RequireManualConfirmation && count <= MinSelect` @ :397) returns
// the pile in RAW order. So the ≤1-draw auto-take keeps raw pile order, but the
// 2+-draw surfaced screen is sorted `(Rarity, then Id == card Entry name)`,
// exactly like the sibling Charge / Stratagem / ForegoneConclusion draw-select
// screens. A prior port misread bead 1a.12's swap as dropping the sort for BOTH
// paths (it only dropped it for the local-UI else branch), so an index-replay
// harness picked the wrong draw slot off the screen (DRAW_ORDER|SEANCE: Seance
// transformed raw slot 0 where C# transformed the (rarity, id)-sorted slot).
pub(crate) fn play_seance(state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    // Raw draw-pile order for the ≤1 auto-take fast path; the 2+ interactive
    // branch below re-sorts by (Rarity, Id) per C# FromCombatPile.
    let ordered: Vec<sts2_core::CardUid> = state.player.draw.iter().copied().collect();

    match ordered.as_slice() {
        // Empty draw pile → FirstOrDefault() == null → no transform.
        [] => {}
        // FromSimpleGrid auto-applies to the lone draw card (count <= MinSelect):
        // transform it directly into a fresh (un-upgraded) Soul.
        &[uid] => {
            let soul = card_from_spec(&status_specs::soul());
            let _ = state.transform_card(uid, soul);
        }
        // 2+ draw cards: open a single-pick choice. C# `Seance.OnPlay` uses
        // `CardSelectCmd.FromCombatPile(PileType.Draw.GetPile(Owner), prefs(1))`.
        // The auto-resolve fast path (`count <= MinSelect`, i.e. the ≤1 branch
        // above) returns the pile in RAW order, but the INTERACTIVE (`Selector`)
        // branch — taken here with 2+ cards — re-applies `orderby c.Rarity, c.Id`
        // for any `PileType.Draw` pile (`CardSelectCmd.cs` @ :401-409). So the
        // surfaced screen is sorted by `(Rarity, Id)`, exactly like the sibling
        // Charge / Stratagem / ForegoneConclusion draw-select screens. The
        // resolver mints a fresh Soul and transforms the chosen card.
        _ => {
            let mut keyed: Vec<(u8, &'static str, sts2_core::CardUid)> = ordered
                .iter()
                .filter_map(|&uid| {
                    let card = state.cards.get(uid.0.get() as usize - 1)?;
                    Some((
                        crate::cards::spec_index::card_rarity_rank_csharp_in(
                            card.card_id,
                            state.game_version,
                        ),
                        sts2_core::cards::card_model_entry(card.card_id),
                        uid,
                    ))
                })
                .collect();
            keyed.sort_by(|a, b| {
                a.0.cmp(&b.0).then_with(|| {
                    sts2_core::cards::draw_select_tiebreak(
                        &state.cards,
                        a.2,
                        b.2,
                        state.game_version,
                    )
                })
            });
            let options: Vec<sts2_core::ChoiceOption> = keyed
                .iter()
                .filter_map(|&(_, _, uid)| {
                    let card = state.cards.get(uid.0.get() as usize - 1)?;
                    Some(sts2_core::ChoiceOption {
                        card_id: card.card_id,
                        upgraded: card.upgraded,
                        card_uid: Some(uid),
                        source_pile: Some(sts2_core::cards::PileKind::Draw),
                    })
                })
                .collect();
            state.request_card_choice(
                sts2_core::ChoiceKind::TransformDrawCardToSoul,
                options,
                1,
                1,
                false,
            );
        }
    }

    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Seance,
        name: "SEANCE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::ETHEREAL, // CanonicalKeywords { Ethereal }
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: EnergyCost.UpgradeBy(-1) → 1 → 0. The CardsVar transform
    // count is untouched by the upgrade.
    Some(CardUpgradeDelta {
        cost: Some(0),
        base_damage: None,
        base_block: None,
        keywords: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    fn mint(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
        let mut card = crate::cards::spec_index::card_pod_from_id(id, false).expect("spec exists");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        uid
    }

    /// 2+ draw-pile cards: v0.107.1 `Seance.OnPlay` uses
    /// `CardSelectCmd.FromCombatPile(PileType.Draw.GetPile(Owner) …)` —
    /// `prefs(prompt, 1)` surfaces a forced single-pick `card_select`
    /// (count > MinSelect == 1). The interactive `Selector` branch of
    /// `FromCombatPile` re-applies `orderby c.Rarity, c.Id` for `PileType.Draw`,
    /// so the options are **(Rarity, Id)-sorted**, NOT raw pile order.
    /// Resolving a 1-index selection transforms exactly that draw slot.
    #[test]
    fn seance_two_draw_cards_opens_transform_choice_and_transforms_chosen() {
        crate::cards::dispatch::install_default_registry();
        // The core-side resolver mints Soul through the registered card factory.
        sts2_core::register_card_factory(crate::cards::spec_index::card_pod_from_id);
        let mut state = CombatState::default();
        // Alive player -- dead trips the request_card_choice ending guard.
        state.player.current_hp = 80;
        // Two observable draw-pile cards. The raw pile order [Strike, Defend]
        // DIFFERS from the (rarity, id) sort the interactive screen applies:
        // both are Basic rarity so the sort keys on the entry-id string
        // ("DEFEND_NECROBINDER" < "STRIKE_NECROBINDER"), reordering the SCREEN to
        // [Defend, Strike] — so option[0] must be the Defend, proving the sort
        // is applied (raw order would have put the Strike first).
        let strike = mint(&mut state, CardId::StrikeNecrobinder);
        let defend = mint(&mut state, CardId::DefendNecrobinder);
        state.player.draw.extend([strike, defend]);
        // The resolving Seance sits on top of the Play pile.
        let seance = mint(&mut state, CardId::Seance);
        state.player.play.push(seance);

        let r = play_seance(&mut state, None);
        assert_eq!(r, EffectResult::Resolved);
        // 2+ draw cards → a single-pick choice opens (no auto-apply).
        let pc = state
            .pending_choice
            .as_ref()
            .expect("Seance opens a choice with 2+ draw cards");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::TransformDrawCardToSoul);
        assert_eq!(pc.min_select, 1);
        assert_eq!(pc.max_select, 1);
        assert_eq!(pc.options.len(), 2, "both draw cards are options");
        // Options follow the (Rarity, Id) sort the interactive FromCombatPile
        // Selector branch applies to Draw piles: the Defend (id
        // "DEFEND_NECROBINDER") sorts before the Strike (id
        // "STRIKE_NECROBINDER").
        assert_eq!(
            pc.options[0].card_uid,
            Some(defend),
            "options[0] is the (rarity, id)-sorted Defend, NOT the pile-first Strike"
        );
        assert_eq!(
            pc.options[0].card_id,
            CardId::DefendNecrobinder,
            "option[0] is the (rarity, id)-sorted Defend; raw pile order would have put the Strike first"
        );
        assert_eq!(pc.options[1].card_uid, Some(strike));
        assert_eq!(
            pc.options[0].source_pile,
            Some(sts2_core::cards::PileKind::Draw)
        );

        // Pick option 1 (the sorted-second card == the Strike) — must transform
        // `strike`, leaving `defend` untouched.
        let selection: SmallVec<[u8; 4]> = [1u8].into_iter().collect();
        let res = sts2_core::apply(&mut state, sts2_core::Action::ResolveChoice { selection });
        assert_eq!(res, sts2_core::StepResult::Advanced);
        assert!(state.pending_choice.is_none());
        // `transform_card` rebinds the chosen draw-pile SLOT to a fresh card
        // (new uid), leaving `defend`'s uid untouched and removing `strike`'s.
        assert!(
            state.player.draw.contains(&defend),
            "unchosen draw card untouched"
        );
        assert_ne!(
            state.cards[defend.0.get() as usize - 1].card_id,
            CardId::Soul,
            "unchosen draw card not transformed"
        );
        assert!(
            !state.player.draw.contains(&strike),
            "chosen draw card slot rebound to the new Soul"
        );
        let soul_uid = *state
            .player
            .draw
            .iter()
            .find(|&&u| u != defend)
            .expect("transformed draw slot present");
        assert_eq!(
            state.cards[soul_uid.0.get() as usize - 1].card_id,
            CardId::Soul,
            "chosen draw card transformed into Soul"
        );
        assert!(
            !state.cards[soul_uid.0.get() as usize - 1].upgraded,
            "TransformTo<Soul> mints a fresh un-upgraded Soul"
        );
    }

    /// Exactly one draw card: C# `FromSimpleGrid` auto-applies (count <=
    /// MinSelect) with NO prompt; Rust transforms the lone draw card into a Soul
    /// directly (preserving the prior auto-apply behavior).
    #[test]
    fn seance_one_draw_card_auto_applies_no_choice() {
        crate::cards::dispatch::install_default_registry();
        let mut state = CombatState::default();
        // A live player: a transform is refused once the combat is ending.
        state.player.current_hp = 1;
        let a = mint(&mut state, CardId::StrikeNecrobinder);
        state.player.draw.push(a);
        let seance = mint(&mut state, CardId::Seance);
        state.player.play.push(seance);

        let r = play_seance(&mut state, None);
        assert_eq!(r, EffectResult::Resolved);
        assert!(
            state.pending_choice.is_none(),
            "no choice surfaced for the lone draw card"
        );
        // The lone draw SLOT is rebound to a fresh Soul (new uid); `a`'s old
        // uid is gone from the draw pile.
        assert!(!state.player.draw.contains(&a), "draw slot rebound");
        assert_eq!(state.player.draw.len(), 1, "still one draw card");
        let soul_uid = state.player.draw[0];
        assert_eq!(
            state.cards[soul_uid.0.get() as usize - 1].card_id,
            CardId::Soul,
            "lone draw card transformed into Soul"
        );
    }

    /// Empty draw pile: FirstOrDefault() == null → no-op, no choice.
    #[test]
    fn seance_empty_draw_is_a_noop() {
        let mut state = CombatState::default();
        let seance = mint(&mut state, CardId::Seance);
        state.player.play.push(seance);
        let r = play_seance(&mut state, None);
        assert_eq!(r, EffectResult::Resolved);
        assert!(state.pending_choice.is_none());
    }
}
