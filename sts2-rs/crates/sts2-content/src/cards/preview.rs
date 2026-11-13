// HAND-MAINTAINED — edit directly.
//! What a card's face shows for its `{Calculated*}` placeholder right now.
//!
//! Some cards state a value they only know at play time: Ashen Strike is
//! `CalculationBase(6) + ExtraDamage(3) x <cards in your Exhaust Pile>`. The
//! game re-renders that number on the card as the board changes, so a player
//! reads the real total off the face before committing.
//!
//! C# computes it in [`CalculatedVar.Calculate`], which every such card owns a
//! subclass of:
//!
//! [decompiled C# omitted — see the cited path]
//!
//! Outside combat the multiplier is zero, which is why a card at rest reads its
//! plain base — the wiki manifest's `description_plain` already says that, and
//! this module deliberately says nothing there ([`calculated_value`] is only
//! ever called with a live [`CombatState`]).
//!
//! # Why the play handlers delegate here
//!
//! The multiplier is the card's *behaviour*. A preview that recomputed it
//! separately would be free to drift from what the card actually does, and a
//! confidently wrong number on the face is worse than no number at all. So each
//! card's formula lives exactly once, in a `preview_*` beside its `play_*`, and
//! the play handler calls it. The handler keeps the in-flight card implicit
//! (`resolving_card_inst`); the preview takes the card explicitly, because the
//! face being previewed is usually a card sitting in hand, not the one
//! resolving.
//!
//! # Targeting
//!
//! Five cards read the *target*: Bully's Vulnerable, Rend's debuffs, Time's Up
//! and No Escape's Doom, and Beat Into Shape's prior hits on that enemy. The
//! game re-renders those on hover and shows the untargeted value otherwise.
//! Display callers pass `None` and get exactly what the game shows before you
//! hover; the play handlers pass the real target.

use sts2_core::cards::{Card, CardId};
use sts2_core::hooks::CreatureRef;
use sts2_core::{CombatState, CreatureUid};

/// What kind of number a card's `{Calculated*}` placeholder is, which decides
/// which display hooks run over it.
///
/// C# gives `CalculatedVar` two hook-running subclasses and leaves the rest
/// bare: `CalculatedDamageVar.UpdateCardPreview` folds the value through
/// `Hook.ModifyDamage(... ModifyDamageHookType.All ...)` and then floors at
/// zero, `CalculatedBlockVar` folds through `Hook.ModifyBlock`, and the plain
/// `CalculatedVar` used for every count (`CalculatedHits`, `CalculatedCards`,
/// `CalculatedDoom`, …) just reports `Calculate(target)`.
///
/// This is why the face on a card in hand already shows your Strength and the
/// target's Vulnerable folded in — and why the shared `preview_*` formulas must
/// NOT apply them: the play path folds the same hooks itself inside
/// `deal_damage_*` / `gain_block_player`, so doing it twice would double-count.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    /// `CalculatedDamage` dealt by the player.
    Damage,
    /// `CalculatedDamage` dealt by the Osty (C# `CalculatedDamageVar.FromOsty`).
    OstyDamage,
    /// `CalculatedBlock`.
    Block,
    /// A plain count — hits, cards, Doom, Energy, Focus, Forge, Shivs, Channels.
    Count,
}

/// A card's "what would this be right now" calculation.
///
/// Takes the card explicitly rather than reading the resolving card, so it can
/// be asked about any instance in any pile.
pub(crate) type PreviewFn = fn(&CombatState, &Card, Option<CreatureUid>) -> i32;

/// Run a `preview_*` against the card currently being played.
///
/// The bridge that lets a play handler reuse its own preview: it supplies the
/// in-flight card the handler's old `current_damage()` / `current_upgraded()`
/// reads were implicitly about. Returns 0 when nothing is resolving, matching
/// those helpers' own behaviour.
pub(crate) fn resolving_preview(
    state: &CombatState,
    f: PreviewFn,
    target: Option<CreatureUid>,
) -> i32 {
    sts2_core::card_effects::resolving_card_inst(state).map_or(0, |card| f(state, card, target))
}

/// The number `card`'s `{Calculated*}` placeholder resolves to on this board,
/// or `None` for the ~530 cards that state no runtime-calculated value.
///
/// `target` is the creature the player is currently aiming at, or `None` for an
/// untargeted render (see the module docs).
///
/// Display consumers must treat `None` as "this card has no such number",
/// **not** as zero: a card whose face says `(Hits 0 times)` reads as a fact,
/// and for these cards zero is never the truth on a live board.
#[must_use]
pub fn calculated_value(
    state: &CombatState,
    card: &Card,
    target: Option<CreatureUid>,
) -> Option<i32> {
    // Cards whose placeholder is `CalculatedDamage` (see `ValueKind`).
    let (f, kind): (PreviewFn, ValueKind) = match card.card_id {
        CardId::AshenStrike => (
            crate::cards::ironclad::ashen_strike::preview_ashen_strike,
            ValueKind::Damage,
        ),
        CardId::BodySlam => (
            crate::cards::ironclad::body_slam::preview_body_slam,
            ValueKind::Damage,
        ),
        CardId::Bully => (
            crate::cards::ironclad::bully::preview_bully,
            ValueKind::Damage,
        ),
        CardId::CrescentSpear => (
            crate::cards::regent::crescent_spear::preview_crescent_spear,
            ValueKind::Damage,
        ),
        CardId::DeathMarch => (
            crate::cards::necrobinder::death_march::preview_death_march,
            ValueKind::Damage,
        ),
        CardId::GangUp => (
            crate::cards::colorless::gang_up::preview_gang_up,
            ValueKind::Damage,
        ),
        CardId::GoldAxe => (
            crate::cards::colorless::gold_axe::preview_gold_axe,
            ValueKind::Damage,
        ),
        CardId::MementoMori => (
            crate::cards::silent::memento_mori::preview_memento_mori,
            ValueKind::Damage,
        ),
        CardId::MindBlast => (
            crate::cards::colorless::mind_blast::preview_mind_blast,
            ValueKind::Damage,
        ),
        CardId::Murder => (
            crate::cards::silent::murder::preview_murder,
            ValueKind::Damage,
        ),
        CardId::PerfectedStrike => (
            crate::cards::ironclad::perfected_strike::preview_perfected_strike,
            ValueKind::Damage,
        ),
        CardId::PreciseCut => (
            crate::cards::silent::precise_cut::preview_precise_cut,
            ValueKind::Damage,
        ),
        CardId::Protector => (
            crate::cards::necrobinder::protector::preview_protector,
            ValueKind::OstyDamage,
        ),
        CardId::Rend => (
            crate::cards::colorless::rend::preview_rend,
            ValueKind::Damage,
        ),
        CardId::SoulStorm => (
            crate::cards::necrobinder::soul_storm::preview_soul_storm,
            ValueKind::Damage,
        ),
        CardId::Squeeze => (
            crate::cards::necrobinder::squeeze::preview_squeeze,
            ValueKind::OstyDamage,
        ),
        CardId::Supermassive => (
            crate::cards::regent::supermassive::preview_supermassive,
            ValueKind::Damage,
        ),
        CardId::TimesUp => (
            crate::cards::necrobinder::times_up::preview_times_up,
            ValueKind::Damage,
        ),
        CardId::Unleash => (
            crate::cards::necrobinder::unleash::preview_unleash,
            ValueKind::OstyDamage,
        ),
        // `CalculatedBlock` and the plain counts.
        CardId::Barrage => (
            crate::cards::defect::barrage::preview_barrage,
            ValueKind::Count,
        ),
        CardId::BeatIntoShape => (
            crate::cards::regent::beat_into_shape::preview_beat_into_shape,
            ValueKind::Count,
        ),
        CardId::CompileDriver => (
            crate::cards::defect::compile_driver::preview_compile_driver,
            ValueKind::Count,
        ),
        CardId::DemonicShield => (
            crate::cards::ironclad::demonic_shield::preview_demonic_shield,
            ValueKind::Block,
        ),
        CardId::ExpectAFight => (
            crate::cards::ironclad::expect_a_fight::preview_expect_a_fight,
            // v0.111.0: the placeholder became a `CalculatedBlockVar`
            // (decomp ExpectAFight.cs:23); v0.107.1 counted Energy.
            if state.game_version >= sts2_core::GameVersion::V0_111_0 {
                ValueKind::Block
            } else {
                ValueKind::Count
            },
        ),
        CardId::Finisher => (
            crate::cards::silent::finisher::preview_finisher,
            ValueKind::Count,
        ),
        CardId::FlakCannon => (
            crate::cards::defect::flak_cannon::preview_flak_cannon,
            ValueKind::Count,
        ),
        CardId::Flechettes => (
            crate::cards::silent::flechettes::preview_flechettes,
            ValueKind::Count,
        ),
        CardId::HelixDrill => (
            crate::cards::defect::helix_drill::preview_helix_drill,
            ValueKind::Count,
        ),
        CardId::MyShinies => (
            crate::cards::colorless::my_shinies::preview_my_shinies,
            ValueKind::Count,
        ),
        CardId::KnifeTrap => (
            crate::cards::silent::knife_trap::preview_knife_trap,
            ValueKind::Count,
        ),
        CardId::LunarBlast => (
            crate::cards::regent::lunar_blast::preview_lunar_blast,
            ValueKind::Count,
        ),
        CardId::Mirage => (
            crate::cards::silent::mirage::preview_mirage,
            ValueKind::Block,
        ),
        CardId::NoEscape => (
            crate::cards::necrobinder::no_escape::preview_no_escape,
            ValueKind::Count,
        ),
        CardId::PullFromBelow => (
            crate::cards::necrobinder::pull_from_below::preview_pull_from_below,
            ValueKind::Count,
        ),
        CardId::Radiate => (
            crate::cards::regent::radiate::preview_radiate,
            ValueKind::Count,
        ),
        CardId::Rattle => (
            crate::cards::necrobinder::rattle::preview_rattle,
            ValueKind::Count,
        ),
        CardId::Sacrifice => (
            crate::cards::necrobinder::sacrifice::preview_sacrifice,
            ValueKind::Block,
        ),
        CardId::Stack => (
            crate::cards::colorless::stack::preview_stack,
            ValueKind::Block,
        ),
        CardId::Synchronize => (
            crate::cards::defect::synchronize::preview_synchronize,
            ValueKind::Count,
        ),
        CardId::TearAsunder => (
            crate::cards::ironclad::tear_asunder::preview_tear_asunder,
            ValueKind::Count,
        ),
        CardId::Voltaic => (
            crate::cards::defect::voltaic::preview_voltaic,
            ValueKind::Count,
        ),
        // `Normality.cs:27-31`: `CalculatedCards` = CalculationBase(3) +
        // CalculationExtra(-1) x min(3, card plays started this turn), so the
        // plays still allowed. Same on both builds.
        CardId::Normality => (preview_normality, ValueKind::Count),
        _ => return None,
    };
    let raw = f(state, card, target);
    Some(match kind {
        ValueKind::Damage => modified_damage(state, raw, CreatureRef::Player, target),
        // No living Osty means no hit at all, so the face shows nothing to fold.
        ValueKind::OstyDamage => match sts2_core::card_effects::osty_dealer_ref(state) {
            Some(dealer) => modified_damage(state, raw, dealer, target),
            None => raw.max(0),
        },
        ValueKind::Block => sts2_core::hooks::modify_block(
            state,
            raw,
            &sts2_core::hooks::BlockCtx {
                target: CreatureRef::Player,
                props: sts2_core::hooks::ValueProps::MOVE,
                card_source: Some(card.uid),
            },
        ),
        ValueKind::Count => raw,
    })
}
/// Fold a damage number through the outgoing/incoming modifier hooks the game
/// runs before painting it on the card — Strength and Vigor on your side, the
/// target's Vulnerable on theirs — then floor at zero.
///
/// C# `CalculatedDamageVar.UpdateCardPreview`:
///
/// [decompiled C# omitted — see the cited path]
///
/// C#'s `target` is `Creature?` and is null for an unhovered card, so the
/// target-side half of the fold has nothing to read and only the dealer's own
/// modifiers apply. `DamageCtx.target` is not optional here, so an untargeted
/// preview aims at [`absent_creature`] — a ref that matches no listener owner
/// and no power row, which is the same thing.
///
/// Naming the *dealer* instead (the obvious shortcut) silently reads the
/// player's own debuffs as the target's: `Tainted`, `Debilitate`, `HardToKill`
/// and the Vulnerable multiplier all gate on `owner == Some(ctx.target)`, and
/// the Intangible arm of `modify_damage_cap` caps at 1. A player carrying
/// Intangible would have read "Deal 1 damage." on a card about to deal 24.
fn modified_damage(
    state: &CombatState,
    raw: i32,
    dealer: CreatureRef,
    target: Option<CreatureUid>,
) -> i32 {
    let ctx = sts2_core::hooks::DamageCtx {
        dealer,
        target: target.map_or_else(|| absent_creature(state), CreatureRef::Enemy),
        props: sts2_core::hooks::ValueProps::MOVE,
    };
    sts2_core::hooks::modify_damage(state, raw, &ctx).max(0)
}

/// A [`CreatureRef`] that provably resolves to nobody in this arena — one past
/// the highest live creature id, so every `owner == Some(ctx.target)` compare
/// is false and every power lookup misses. The hook fold only ever *compares*
/// or *looks up* the target ref (never indexes with it), so an absent one is
/// safe; see [`modified_damage`] for why it is also the correct one.
fn absent_creature(state: &CombatState) -> CreatureRef {
    let free = state
        .creatures
        .iter()
        .map(|c| c.id.get())
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    CreatureRef::Enemy(CreatureUid::new(free).expect("max id + 1 is non-zero"))
}

/// `Normality`'s `{CalculatedCards}`: plays still allowed this turn. Reads the
/// same counter as the `ShouldPlay` veto, `History.CardPlaysStarted`.
fn preview_normality(state: &CombatState, _card: &Card, _target: Option<CreatureUid>) -> i32 {
    let started = i32::try_from(state.player.card_plays_started_this_turn).unwrap_or(i32::MAX);
    3 - started.min(3)
}
