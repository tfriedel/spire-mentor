//! Fool cards whose own effect is applying a (mostly new) power. The power's
//! behaviour is the core lane's; these handlers apply it with the card's
//! amount, in the C# order. Values from the v0.113.0 decomp.
use sts2_core::card_effects::{self, deal_damage_player_to_enemy, EffectResult};
use sts2_core::{CombatState, CreatureUid, DrawThen, PowerKind};

use super::helpers::{
    apply_to_enemy, apply_to_player, current_block, current_damage, v0114, var, var_by,
};
use crate::cards::draw_then::after_draw;

/// One self-applied power per card: `(power, base amount, upgraded amount)`.
/// A card whose upgrade only changes its cost repeats the amount.
pub(crate) fn self_power(id: sts2_core::cards::CardId) -> Option<(PowerKind, i32, i32)> {
    use sts2_core::cards::CardId as C;
    use PowerKind as P;
    Some(match id {
        C::AllIsWellCard => (P::AllIsWell, 4, 5),
        // `Archery.cs`: applies ArcheryPower with its `AimPower` var.
        C::ArcheryCard => (P::Archery, 4, 5),
        C::CompostCard => (P::Compost, 1, 2),
        C::CoveringFireCard => (P::CoveringFire, 8, 11),
        // `Ecstasy.cs`: `CardsVar(1)`; the upgrade is cost 2 -> 1.
        C::EcstasyCard => (P::Ecstasy, 1, 1),
        C::EnsembleCard => (P::Ensemble, 4, 5),
        C::ExplosiveArrowsCard => (P::ExplosiveArrows, 6, 8),
        C::HiddenRootsCard => (P::HiddenRoots, 3, 4),
        C::HoningCard => (P::Honing, 50, 50),
        C::KiVerseCard => (P::KiVerse, 1, 2),
        C::LostLuteCard => (P::LostLute, 1, 1),
        // `Lullaby.cs`: the upgrade adds Innate.
        C::LullabyCard => (P::Lullaby, 20, 20),
        C::MotifCard => (P::Motif, 1, 1),
        C::QuietusCard => (P::Quietus, 2, 3),
        C::SerratedArrowsCard => (P::SerratedArrows, 25, 50),
        C::SimpletonCard => (P::Simpleton, 1, 1),
        C::SongPathCard => (P::SongPath, 5, 7),
        // `Tune.cs`: `EnergyVar(1)`; the upgrade is cost 2 -> 1.
        C::TuneCard => (P::Tune, 1, 1),
        C::ViolentChordCard => (P::ViolentChord, 9, 12),
        C::YahCard => (P::Yah, 1, 2),
        C::Nock => (P::Aim, 5, 7),
        _ => return None,
    })
}

/// v0.114.0's `(base, upgraded)` power amount for a [`self_power`] card
/// whose `PowerVar` or its upgrade moved (decomp
/// `v0.114.0/.../Models.Cards/<Name>.cs`), `None` when v0.113.0's holds.
fn self_power_v0114(id: sts2_core::cards::CardId) -> Option<(i32, i32)> {
    use sts2_core::cards::CardId as C;
    Some(match id {
        // AllIsWell.cs: `UpgradeValueBy(2m)`.
        C::AllIsWellCard => (4, 6),
        // Archery.cs: `PowerVar<AimPower>(6m)`, +2.
        C::ArcheryCard => (6, 8),
        // ExplosiveArrows.cs: `PowerVar(8m)`, +3.
        C::ExplosiveArrowsCard => (8, 11),
        // Honing.cs: `PowerVar(100m)` (the upgrade is still cost -1).
        C::HoningCard => (100, 100),
        // Lullaby.cs: `PowerVar(12m)`, +4 (no Innate).
        C::LullabyCard => (12, 16),
        // SerratedArrows.cs: `PowerVar(50m)`, +25.
        C::SerratedArrowsCard => (50, 75),
        // Simpleton.cs: the upgrade is `SimpletonPower` +1, not cost -1.
        C::SimpletonCard => (1, 2),
        // SongPath.cs: `PowerVar(4m)`, +2.
        C::SongPathCard => (4, 6),
        // Tune.cs: `CardsVar(1)`, +1 (cards TunePower upgrades).
        C::TuneCard => (1, 2),
        // ViolentChord.cs: `PowerVar(7m)`, +2.
        C::ViolentChordCard => (7, 9),
        _ => return None,
    })
}

/// `PowerCmd.Apply<XPower>(Owner.Creature, amount)` for [`self_power`] cards.
pub(crate) fn play_self_power(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let id = card_effects::originating_card_id(state).expect("resolving card");
    let (kind, base, up) = self_power(id).expect("self-power card");
    let (base, up) = if v0114(state) {
        self_power_v0114(id).unwrap_or((base, up))
    } else {
        (base, up)
    };
    let n = var(state, base, up);
    apply_to_player(state, kind, n);
    EffectResult::Resolved
}

/// `ChoBarbs.cs`: `GainBlock` (7, +2), then ChoBarbsPower 3 (+1); v0.114.0
/// block 8 (+2), power 2 (+1).
pub(crate) fn play_cho_barbs(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    card_effects::gain_block_player(state, current_block(state));
    let n = var_by(state, (3, 4), (2, 3));
    apply_to_player(state, PowerKind::ChoBarbs, n);
    EffectResult::Resolved
}

/// `ChoReady.cs`: `GainBlock` (8, +3), then Aim 3 (+2); v0.114.0 block 7
/// (+2), Aim 3 (+1).
pub(crate) fn play_cho_ready(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    card_effects::gain_block_player(state, current_block(state));
    let n = var_by(state, (3, 5), (3, 4));
    apply_to_player(state, PowerKind::Aim, n);
    EffectResult::Resolved
}

/// `DislikeYou.cs`: attack (8, +1), then DislikeYouPower 50 (+25) on the
/// target.
pub(crate) fn play_dislike_you(state: &mut CombatState, t: Option<CreatureUid>) -> EffectResult {
    let t = t.expect("DislikeYou: AnyEnemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    let n = var(state, 50, 75);
    apply_to_enemy(state, t, PowerKind::DislikeYou, n);
    EffectResult::Resolved
}

/// `Brace.cs` (X cost): X separate `Apply<AimPower>` of 8 (+3).
pub(crate) fn play_brace(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let x = card_effects::resolving_card_x_value(state);
    let n = var(state, 8, 11);
    for _ in 0..x {
        apply_to_player(state, PowerKind::Aim, n);
    }
    EffectResult::Resolved
}

/// `DeepStretch.cs`: Stretch 7 (+3), then `Draw(1)`.
pub(crate) fn play_deep_stretch(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 7, 10);
    apply_to_player(state, PowerKind::Stretch, n);
    let was_suspended = state.pending_draw_resume.is_some();
    let _ = sts2_core::cards::draw_via_state(state, 1);
    after_draw(state, was_suspended, DrawThen::None);
    EffectResult::Resolved
}

/// `Encore.cs`: `GainEnergy(1)`, `Draw(2 (+1))`, then EncorePower 1
/// ([`DrawThen::FoolEncore`] when the draw suspends).
pub(crate) fn play_encore(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    state.gain_energy(1);
    let n = var(state, 2, 3);
    let was_suspended = state.pending_draw_resume.is_some();
    let _ = sts2_core::cards::draw_via_state(state, n);
    after_draw(state, was_suspended, DrawThen::FoolEncore);
    EffectResult::Resolved
}

/// `Enrapture.cs` (MP-only, AnyAlly): Vigor 6 (+3) then Stretch 6 (+3) on
/// the target ally. Unplayable solo (no living ally to target).
pub(crate) fn play_enrapture(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // `AnyAlly` (`card_effects::card_is_any_ally`): "Give another player ..."
    // — both powers land on the card's target teammate, not the owner. Read
    // the amount before the target is seated (the card leaves with our seat).
    let n = var(state, 6, 9);
    sts2_core::coop::apply_power_to_card_target(state, PowerKind::Vigor, n);
    sts2_core::coop::apply_power_to_card_target(state, PowerKind::Stretch, n);
    EffectResult::Resolved
}

/// `Yoga.cs` (MP-only, AllAllies): Stretch 6 (+2) on every living player,
/// the owner included ("ALL players gain Stretch").
pub(crate) fn play_yoga(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // Read before the seats move: the played card stays with its owner.
    let n = var(state, 6, 8);
    sts2_core::coop::every_player(state, |state| {
        apply_to_player(state, PowerKind::Stretch, n);
    });
    EffectResult::Resolved
}

/// The player's most recent `kind` power instance (the one an `Apply` just
/// created or stacked into), on the creature mirror or the legacy list.
fn last_player_power_mut(
    state: &mut CombatState,
    kind: PowerKind,
) -> Option<&mut sts2_core::Power> {
    let mut ids: smallvec::SmallVec<[sts2_core::PowerUid; 8]> =
        state.player.powers.iter().copied().collect();
    if let Some(cuid) = state.player.creature_uid {
        if let Some(c) = state.creatures.iter().find(|c| c.id == cuid.0) {
            ids.extend(c.powers.iter().copied());
        }
    }
    let id = ids.into_iter().rev().find(|p| {
        state
            .powers
            .get_by_id(p.0)
            .is_some_and(|pw| pw.kind == kind)
    })?;
    state.powers.get_mut_by_id(id.0)
}

/// `Ennui.cs`: `Apply<EnnuiPower>(2 (+1))?.IncrementSadnessAmount()`. The
/// power's `Cards` var (Sadness added before each hand draw) is its
/// `counter`, +1 per Ennui played.
pub(crate) fn play_ennui(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let n = var(state, 2, 3);
    apply_to_player(state, PowerKind::Ennui, n);
    if let Some(p) = last_player_power_mut(state, PowerKind::Ennui) {
        p.counter = p.counter.saturating_add(1);
    }
    EffectResult::Resolved
}

/// `MythicForm.cs`: `Apply<MythicFormPower>(1)`, then `IncrementUpgraded()`
/// when upgraded. The power's `Upgraded` var (how many of its tarot cards
/// come upgraded) is its `counter`.
pub(crate) fn play_mythic_form(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    apply_to_player(state, PowerKind::MythicForm, 1);
    // v0.114.0 `MythicForm.cs`: the upgrade only drops Ethereal; the power
    // has no `Upgraded` counter (`IncrementUpgraded` is gone).
    if super::helpers::upgraded(state) && !v0114(state) {
        if let Some(p) = last_player_power_mut(state, PowerKind::MythicForm) {
            p.counter = p.counter.saturating_add(1);
        }
    }
    EffectResult::Resolved
}

/// `ShowOff.cs`: `Apply<AimPower>(CurrentAim)` where `CurrentAim = 5 +
/// IncreasedAim` (the card's `Generic` var), then the Instanced
/// `ShowOffPower` 2 (+1) with `ShowOffInstance = this`: the new instance's
/// `counter` holds the card's uid, for the power's fatal-kill `IncreaseAim`.
pub(crate) fn play_show_off(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let uid = card_effects::resolving_card_uid(state);
    let increased = card_effects::resolving_card_inst(state)
        .map_or(0, |c| c.combat_var(sts2_core::cards::VarKey::Generic));
    apply_to_player(state, PowerKind::Aim, 5 + increased);
    let n = var(state, 2, 3);
    apply_to_player(state, PowerKind::ShowOff, n);
    if let (Some(uid), Some(p)) = (uid, last_player_power_mut(state, PowerKind::ShowOff)) {
        p.counter = i32::try_from(uid.0.get()).unwrap_or(0);
    }
    EffectResult::Resolved
}
