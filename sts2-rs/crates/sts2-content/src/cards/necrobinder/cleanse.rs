//! Hand-authored (C#-first port). Keep the prelude import.
use super::prelude::*;

/// Cleanse (Uncommon, 1 energy, Skill, Self). Summon 3 (Osty), then choose
/// one card in the draw pile to Exhaust.
// C# source: ../sts2-wiki/decompiled/v0.107.1/MegaCrit.Sts2.Core.Models.Cards/Cleanse.cs
//
// C# `OnPlay` (v0.107.1):
//   1. CreatureCmd.TriggerAnim(Owner.Creature, "Cast", ...)   // cosmetic, ignored
//   2. OstyCmd.Summon(ctx, Owner, Summon.BaseValue, this)     // grow/spawn the Osty
//   3. CardSelectCmd.FromCombatPile(ctx, PileType.Draw.GetPile(Owner), Owner,
//          prefs(ExhaustSelectionPrompt, 1))                  // (Rarity, Id)-sorted screen
//   4. if chosen != null { CardCmd.Exhaust(ctx, chosen) }
//
// CanonicalVars is a single `SummonVar(3m)`, so the base summon amount is 3;
// `OnUpgrade` does `Summon.UpgradeValueBy(2m)` → 5 upgraded. The summon amount
// is a magnitude (not damage/block/cost), so the upgrade bump is gated here on
// `current_card_upgraded` and `upgrade()` returns `None`. `OstyCmd.Summon` maps
// to `CombatState::summon_osty` (grows the live Osty's max HP by `amount`, or
// spawns/revives it at `amount` max HP when dead/absent) — same primitive
// Bodyguard uses.
//
// The card-select-to-exhaust is modeled with the engine's pending-choice
// surface. C# selects from the DRAW pile (not hand) in RAW pile order and
// exhausts the single chosen card. We reuse `ChoiceKind::ExhaustFromHand` whose
// resolver (`resolve_exhaust_from_hand`) calls `exhaust_card_uid` per selected
// `card_uid` — and `exhaust_card_uid` is pile-agnostic (it removes the card from
// whichever pile holds it, including Draw, and fires `after_card_exhausted`). So
// despite the variant name, sourcing options from the draw pile exhausts the
// right card and fires the right listeners. `prefs(..., 1)` with the
// `FirstOrDefault()` null-check ⇒ select exactly 1 if any draw-pile card exists,
// otherwise a clean no-op (empty draw pile ⇒ no choice opened).
//
// v0.107.1 bead 1a.12 (`Cleanse.cs:33`): `FromSimpleGrid(… orderby c.Rarity,
// c.Id …)` → `FromCombatPile(PileType.Draw.GetPile(Owner) …)`. The explicit
// `(rarity, id)` grid sort is NOT gone — it MOVED into `FromCombatPile`'s
// interactive `Selector` branch. `CardSelectCmd.FromCombatPile`
// (`CardSelectCmd.cs:396-410`) has two paths: the fast path (`num <= MinSelect`,
// i.e. a 0/1-card draw pile) returns the pile in raw order, but the interactive
// `Selector` branch — taken whenever the draw pile holds ≥2 cards — re-applies
// `orderby c.Rarity, c.Id` for any `PileType.Draw` pile before presenting the
// screen (`CardSelectCmd.cs:404-409`). So with ≥2 draw cards C# exhausts the
// `(Rarity, Id)`-MINIMUM draw card, NOT the raw-first one — exactly like the
// Stratagem / ForegoneConclusion / Charge draw-select screens. We therefore sort
// the surfaced OPTIONS by `(card_rarity_rank_csharp, id-string)`; sorting a
// 0/1-card pile is a no-op so this also matches the C# fast path. The sort
// reorders only the selection OPTIONS, not the draw pile, so `exhaust_card_uid`
// still removes the chosen card. (The earlier raw-order port read bead 1a.12's
// `FromSimpleGrid → FromCombatPile` swap as dropping the sort engine-wide — that
// was wrong; it surfaced the raw-first slot and exhausted the wrong card, the
// Necrobinder DRAW_ORDER|CLEANSE sweep divergence.)
pub(crate) fn play_cleanse(state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    // 1. Summon the Osty (3 base / 5 upgraded).
    let amount = if current_card_upgraded(state) { 5 } else { 3 };
    let _ = state.summon_osty(amount);

    // 2. Build the draw-pile selection grid, sorted by `(Rarity, Id)` to mirror
    //    `FromCombatPile`'s interactive Selector screen (`CardSelectCmd.cs:404-409`,
    //    `orderby c.Rarity, c.Id` for PileType.Draw). Same key as Charge /
    //    Stratagem. Sorting a 0/1-card pile is a no-op, matching the C# fast path.
    let mut keyed: Vec<(u8, &'static str, sts2_core::CardUid)> = state
        .player
        .draw
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
            sts2_core::cards::draw_select_tiebreak(&state.cards, a.2, b.2, state.game_version)
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

    // 3. Pick exactly one to exhaust (no-op when the draw pile is empty —
    //    `request_card_choice` returns without opening a choice on empty
    //    options, mirroring C# `FirstOrDefault()` == null).
    state.request_card_choice(sts2_core::ChoiceKind::ExhaustFromHand, options, 1, 1, false);

    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Cleanse,
        name: "CLEANSE",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        // No `CanonicalKeywords` override in C# — the `FromKeyword(Exhaust)`
        // ExtraHoverTip just explains the keyword applied to the *selected*
        // draw-pile card, not to Cleanse itself.
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade bumps the SummonVar (3 → 5); applied in the handler via the
    // `current_card_upgraded` branch, so no stat-field delta here.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mint(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
        let mut card =
            crate::cards::spec_index::card_pod_from_id(id, upgraded).expect("spec exists");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        uid
    }

    /// ≥2 draw-pile cards: v0.107.1 `Cleanse.OnPlay` uses
    /// `CardSelectCmd.FromCombatPile(PileType.Draw.GetPile(Owner) …)` with
    /// `prefs(prompt, 1)`. With ≥2 draw cards the interactive `Selector` branch
    /// re-applies `orderby c.Rarity, c.Id` for `PileType.Draw`
    /// (`CardSelectCmd.cs:404-409`), so the surfaced options are
    /// **(Rarity, Id)-sorted**, NOT raw pile order — the headless Selector then
    /// exhausts option[0] = the (Rarity, Id)-minimum draw card.
    #[test]
    fn cleanse_sorts_draw_exhaust_options_by_rarity_id_not_raw_order() {
        crate::cards::dispatch::install_default_registry();
        sts2_core::register_card_factory(crate::cards::spec_index::card_pod_from_id);
        let mut state = CombatState::default();
        // Alive player -- dead trips the request_card_choice ending guard.
        state.player.current_hp = 80;
        // Raw draw order [Strike, Defend, Defend] DIFFERS from the (rarity, id)
        // sort: both are Basic rarity so the sort keys on the entry-id string
        // ("DEFEND_NECROBINDER" < "STRIKE_NECROBINDER"), reordering the SCREEN to
        // [Defend, Defend, Strike] — option[0] must be a Defend, proving the sort.
        let s1 = mint(&mut state, CardId::StrikeNecrobinder, false);
        let d1 = mint(&mut state, CardId::DefendNecrobinder, false);
        let d2 = mint(&mut state, CardId::DefendNecrobinder, false);
        state.player.draw.extend([s1, d1, d2]);
        // The resolving (un-upgraded) CLEANSE sits on top of the Play pile so
        // `current_card_upgraded` resolves.
        let cleanse = mint(&mut state, CardId::Cleanse, false);
        state.player.play.push(cleanse);

        let r = play_cleanse(&mut state, None);
        assert_eq!(r, EffectResult::Resolved);
        let pc = state
            .pending_choice
            .as_ref()
            .expect("Cleanse opens a choice with draw-pile cards");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::ExhaustFromHand);
        assert_eq!(pc.min_select, 1);
        assert_eq!(pc.max_select, 1);
        assert_eq!(pc.options.len(), 3, "all three draw cards are options");
        // Options follow the (Rarity, Id) sort the interactive FromCombatPile
        // Selector branch applies to Draw piles: both Defends (id
        // "DEFEND_NECROBINDER") sort before the Strike (id "STRIKE_NECROBINDER");
        // the two Defends tie so the stable sort keeps their pile order [d1, d2].
        let option_uids: Vec<_> = pc.options.iter().map(|o| o.card_uid).collect();
        assert_eq!(
            option_uids,
            vec![Some(d1), Some(d2), Some(s1)],
            "options in (rarity, id) order [Defend, Defend, Strike], NOT raw pile order"
        );
        assert_eq!(
            pc.options[0].card_id,
            CardId::DefendNecrobinder,
            "option[0] is the (rarity, id)-min Defend; raw pile order would have surfaced the Strike first"
        );
        for o in &pc.options {
            assert_eq!(o.source_pile, Some(sts2_core::cards::PileKind::Draw));
        }
    }
}
