//! Necrobinder Eidolon. Hand-authored C#-first port.
use super::prelude::*;

/// Eidolon (Rare, 2 energy, Skill, Self). Faithful to C#
/// `Eidolon.OnPlay`: exhaust EVERY card in hand (firing exhaust synergies via
/// `CardCmd.Exhaust`), counting how many were exhausted; if at least 9 were
/// exhausted, gain 1 Intangible. The Intangible threshold (9) is a fixed
/// constant and is NOT affected by upgrade — `OnUpgrade` only lowers the energy
/// cost by 1.
///
/// NOTE: Eidolon exhausts your HAND, but the card itself is NOT Exhaust — the
/// C# `Eidolon` ctor passes no `CardKeyword.Exhaust` (the `ExtraHoverTips`
/// Exhaust tip just documents the hand-exhaust effect), so after resolving the
/// card routes to DISCARD, not Exhaust (Necrobinder sweep `PILE_COUNT|EIDOLON`:
/// C# discard `[…, EIDOLON]` vs an earlier Rust port that self-exhausted it).
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Eidolon.cs
///
/// v0.111.0 (`Eidolon.cs:25-32`, GH #793) reworked the card; it is now
/// Exhaust and `CanBeGeneratedInCombat => false`:
///
/// [decompiled C# omitted — see the cited path]
pub(crate) fn play_eidolon(state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    if state.game_version >= sts2_core::GameVersion::V0_111_0 {
        return play_eidolon_v0111(state);
    }
    // Snapshot the hand, then exhaust each card. The Eidolon card itself is on
    // the Play pile (not in hand), so the snapshot is the exact set the C#
    // iterates. `exhaust_card_uid` fires `after_card_exhausted` on the power +
    // relic buses, matching C# `CardCmd.Exhaust`. C# awaits each exhaust and
    // counts every iteration (`exhaustedCount++` follows the awaited
    // `CardCmd.Exhaust` unconditionally, Eidolon.cs:46-50), so the count is
    // the snapshot's size; an exhaust whose Dark Embrace draw reshuffles into
    // a Stratagem pick holds the later exhausts and the Intangible until the
    // pick and the draw finish ([`crate::cards::draw_then::exhaust_then`]).
    let hand: Vec<sts2_core::CardUid> = state.player.hand.to_vec();
    let count = u8::try_from(hand.len()).unwrap_or(u8::MAX);
    crate::cards::draw_then::exhaust_then(state, &hand, sts2_core::DrawThen::Eidolon(count));
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Eidolon,
        name: "EIDOLON",
        cost: 2,
        card_type: CardType::Skill,
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
    // C# OnUpgrade: EnergyCost.UpgradeBy(-1). Threshold (9) is constant.
    Some(CardUpgradeDelta {
        cost: Some(1),
        base_damage: None,
        base_block: None,
        keywords: None,
    })
}

/// v0.111.0 body: auto-play every Ethereal, playable card in the exhaust pile
/// (decomp Eidolon.cs:27-31). The list is snapshotted first; each card is then
/// played by reference wherever it sits (C# `CardCmd.AutoPlay` with a `null`
/// target rolls a random target for an enemy-targeted card).
fn play_eidolon_v0111(state: &mut CombatState) -> EffectResult {
    let picks: Vec<sts2_core::CardUid> = state
        .player
        .exhaust
        .iter()
        .copied()
        .filter(|&u| {
            let card = &state.cards[u.0.get() as usize - 1];
            sts2_core::combat_loop::card_is_ethereal_at_play(state, card)
                && !sts2_core::card_effects::card_unplayable_inst(card)
        })
        .collect();
    eidolon_run(state, &picks);
    EffectResult::Resolved
}

/// Auto-play Eidolon's captured exhaust cards in order. C# `await`s each
/// `CardCmd.AutoPlay`, so a card that opens a surfacing select (Graveblast's
/// discard pick) fully resolves before the next captured card plays (Seance's
/// draw-pile pick). On a surfaced select, suspend and hand the rest of the
/// list to the resolver, like BeatDown. Without this, Seance's select
/// replaced Graveblast's and the Graveblast pick never happened
/// (Necrobinder-1001 act-3 `hand_count` split, GH #1054).
fn eidolon_run(state: &mut CombatState, cards: &[sts2_core::CardUid]) {
    for (i, &uid) in cards.iter().enumerate() {
        // C# `CardCmd.AutoPlay` returns at once when combat is over or ending
        // (decomp CardCmd.cs:54).
        if sts2_core::combat_loop::combat_is_ending(state) || state.player.current_hp <= 0 {
            break;
        }
        let _ = sts2_core::combat_loop::auto_play_specific_card_from_discard(state, uid, false);
        if let Some(resume) = state.pending_autoplay_resume.as_mut() {
            resume.continuation = Some(sts2_core::AutoPlayLoopCont {
                resume_fn: eidolon_resume,
                iters: 0,
                uids: cards[i + 1..].iter().copied().collect(),
                force_exhaust: false,
                then: None,
            });
            return;
        }
    }
}

/// Resume entry point stored on [`sts2_core::AutoPlayLoopCont`]: play the
/// remaining captured Eidolon cards.
fn eidolon_resume(state: &mut CombatState, cont: &sts2_core::AutoPlayLoopCont) {
    eidolon_run(state, &cont.uids);
}
