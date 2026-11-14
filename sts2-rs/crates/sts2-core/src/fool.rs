//! The Fool's combat mechanics (v0.113.0): arrows, `ImpaledPower`, the
//! `AimPower` / `QuietusPower` attack latch, and melody (`Wah` / `Cho` /
//! `Ki`) tracking.
//!
//! Source of truth (decomp `v0.113.0`): `Commands/ArrowCmd.cs`,
//! `Commands/MelodyCmd.cs`, `Entities.FoolArrows/*`,
//! `Models.Powers/{Impaled,Aim,Quietus,Yah,ExplosiveArrows}Power.cs`,
//! `Models.Relics/Twine.cs`, `Models.Cards/{WhistlingArrows,EpicRetrieval}.cs`.
//!
//! Everything lives in [`crate::PlayerCombatState::fool`], an
//! `Option<Box<FoolState>>` that stays `None` until a Fool mechanic first
//! touches it — so builds before v0.113.0, and v0.113.0 combats that never
//! see a Fool card, keep their exact state, size and hash.
//!
//! ## Arrows
//! A player owns `ArrowSlot`s (`PlayerCombatState._arrows`): the Fool starts
//! with `BaseArrowSlotCount` = 1, and any character gains one when a
//! `Shoot`-tagged card enters a combat pile while it has none
//! (`CardPile.AddInternal`). The next arrow to shoot is the LAST available
//! (`Retrieved`) one. Shooting marks it `Away`; impaling an enemy pushes it
//! onto that enemy's FIFO arrow list (the `ImpaledPower` instance whose
//! applier is the player — one per enemy in single player); dislodging pops
//! the FIRST arrow and retrieves it (or, for a Quietus arrow marked
//! `DestroyAfterImpaled`, destroys it).
//!
//! ## Melody
//! `MelodyCmd` scans `History.CardPlaysStarted` / `CardPlaysFinished` for
//! entries that `HappenedThisTurn` (same `RoundNumber` and `CurrentSide`).
//! [`note_play_started`] / [`note_play_finished`] mirror those two history
//! records at the engine's play chokepoints; the per-turn tallies reset
//! lazily whenever the `(round, side)` stamp moves.

use smallvec::SmallVec;

use crate::cards::{CardId, CardTag, CardTags};
use crate::hooks::{CreatureRef, ValueProps};
use crate::{CardUid, CombatState, CreatureUid, PowerKind, Side};

/// `ImpaledPower.HitCounter` start value (`DynamicVar("HitCounter", 3m)`).
pub const IMPALED_HITS: i32 = 3;

/// C# `ArrowStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArrowStatus {
    /// In the quiver, ready to shoot (`IsAvailable`).
    Retrieved,
    /// Shot, sitting in an enemy (or in flight).
    Away,
    /// Gone for the combat (a Quietus arrow after impaling).
    Destroyed,
}

/// C# `ArrowSlot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArrowSlot {
    pub status: ArrowStatus,
    pub destroy_after_impaled: bool,
}

impl ArrowSlot {
    const NEW: Self = Self {
        status: ArrowStatus::Retrieved,
        destroy_after_impaled: false,
    };
}

/// Per-turn melody tallies, valid while `stamp` equals the live
/// `(round, side)`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MelodyTurn {
    /// `(round, side == Player)` the tallies belong to; `None` = empty.
    pub stamp: Option<(u16, bool)>,
    /// Finished plays this turn per note, index [`note_index`].
    pub finished: [u8; 3],
    /// Finished plays this turn carrying any note.
    pub melody_cards_finished: u8,
    /// Started plays this turn per note.
    pub started: [u8; 3],
    /// Play sequence number of the FIRST started play this turn per note
    /// (0 = none) — `MelodyCmd.IsFirstPlayThisTurn`'s `FirstOrDefault`.
    pub first_started_seq: [u32; 3],
}

/// The Fool mechanics' combat state (see the module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FoolState {
    /// `PlayerCombatState.Arrows`, in gain order.
    pub arrows: SmallVec<[ArrowSlot; 4]>,
    /// Impaled enemies: `(enemy, FIFO of arrow-slot indices)`, the
    /// `ImpaledPower.arrowSources` list of the player's instance.
    pub impaled: SmallVec<[(CreatureUid, SmallVec<[u8; 4]>); 2]>,
    /// Melody tallies for the current turn.
    pub melody: MelodyTurn,
    /// Monotone card-play sequence (one per `CardPlayStarted`).
    pub play_seq: u32,
    /// Open plays, innermost last: `(seq, card)`.
    pub open_plays: SmallVec<[(u32, CardUid); 4]>,
    /// `AimPower` latch: the Aim amount when the latched arrow attack began
    /// (`amountWhenAttackStarted`); `None` = no attack latched.
    pub aim_latch: Option<i32>,
    /// The latched attack command's `ModelSource`: the card resolving when
    /// [`Self::aim_latch`] was taken (`None` = a cardless command). Arrow
    /// damage from any other card gets no Aim while the latch stands.
    pub aim_source: Option<CardUid>,
    /// `QuietusPower` latch: an arrow attack is boosted and will spend it.
    pub quietus_latched: bool,
    /// Nesting depth of open arrow attack commands.
    pub arrow_attack_depth: u8,
    /// `StretchPower` latch: `(card, amountWhenCardStarted, gaveBonusBlock)`
    /// for the block card whose play it boosts.
    pub stretch_latch: Option<(CardUid, i32, bool)>,
    /// `EncorePower` plays counted since it was applied
    /// (`numberOfCardsPlayedSinceApplied + cardsBeingPlayed.Count`).
    pub encore_plays: u32,
    /// Sequence number of the play that finished last: the `cardPlay` the
    /// `AfterCardPlayed` listeners see.
    pub last_finished_seq: u32,
    /// `LullabyPower.ProccdThisTurn`, as the round it fired in.
    pub lullaby_round: Option<u16>,
    /// The card whose current play has begun an arrow attack command (see
    /// [`attack_skipped`]); cleared at every play start.
    pub attacked_card: Option<CardUid>,
}

/// Index of a melody note in the per-note arrays.
#[must_use]
pub const fn note_index(tag: CardTag) -> Option<usize> {
    match tag {
        CardTag::Wah => Some(0),
        CardTag::Cho => Some(1),
        CardTag::Ki => Some(2),
        _ => None,
    }
}

/// Whether the v0.113.0 Fool mechanics run in this combat.
#[inline]
#[must_use]
pub fn enabled(state: &CombatState) -> bool {
    state.game_version >= crate::GameVersion::V0_113_0
}

/// The Fool state, if any mechanic has created it.
#[inline]
#[must_use]
pub fn fool(state: &CombatState) -> Option<&FoolState> {
    state.player.fool.as_deref()
}

/// The Fool state, created on first use.
pub fn fool_mut(state: &mut CombatState) -> &mut FoolState {
    state.player.fool.get_or_insert_with(Box::default)
}

// ───────────────────────────── arrows ─────────────────────────────

/// `ArrowCmd.GainArrowSlot` / `PlayerCombatState.GainArrow`.
pub fn gain_arrow_slot(state: &mut CombatState) {
    fool_mut(state).arrows.push(ArrowSlot::NEW);
}

fn arrow_slot_count_raw(state: &CombatState) -> usize {
    fool(state).map_or(0, |f| f.arrows.len())
}

/// `PlayerCombatState.Arrows.Count` (every status).
#[must_use]
pub fn arrow_slot_count(state: &CombatState) -> usize {
    arrow_slot_count_raw(state) + usize::from(shoot_slot_pending(state))
}

/// `ArrowCmd.GetAvailableArrowCount`.
#[must_use]
pub fn available_arrow_count(state: &CombatState) -> i32 {
    let pending = i32::from(shoot_slot_pending(state));
    pending
        + fool(state).map_or(0, |f| {
            f.arrows
                .iter()
                .filter(|a| a.status == ArrowStatus::Retrieved)
                .count() as i32
        })
}

/// `ArrowCmd.PlayerHasAnyArrows`.
#[must_use]
pub fn player_has_any_arrows(state: &CombatState) -> bool {
    shoot_slot_pending(state) || next_arrow(state).is_some()
}

/// `ArrowCmd.TryGetNextArrowToShoot`: the LAST available arrow.
#[must_use]
pub fn next_arrow(state: &CombatState) -> Option<usize> {
    fool(state)?
        .arrows
        .iter()
        .rposition(|a| a.status == ArrowStatus::Retrieved)
}

/// The player's `ImpaledPower` uid on `creature`, if it is impaled.
#[must_use]
pub fn impaled_power_uid(state: &CombatState, creature: CreatureUid) -> Option<crate::PowerUid> {
    let c = crate::creature_by_id(&state.creatures, creature.0)?;
    c.powers.iter().copied().find(|u| {
        state
            .powers
            .get_by_id(u.0)
            .is_some_and(|p| p.kind == PowerKind::Impaled)
    })
}

/// `ArrowCmd.CreatureIsImpaledByArrow` (and, in single player,
/// `CreatureIsImpaledByPlayersArrow`).
#[must_use]
pub fn creature_is_impaled(state: &CombatState, creature: CreatureUid) -> bool {
    impaled_power_uid(state, creature).is_some()
}

/// `ImpaledPower.HitCounter` of the player's instance on `creature`.
#[must_use]
pub fn impaled_hit_counter(state: &CombatState, creature: CreatureUid) -> Option<i32> {
    let uid = impaled_power_uid(state, creature)?;
    state.powers.get_by_id(uid.0).map(|p| p.counter)
}

/// Arrows currently stuck in `creature` (`ImpaledPower.Arrows.Count`).
#[must_use]
pub fn arrows_in(state: &CombatState, creature: CreatureUid) -> usize {
    fool(state)
        .and_then(|f| f.impaled.iter().find(|(c, _)| *c == creature))
        .map_or(0, |(_, a)| a.len())
}

fn is_alive(state: &CombatState, creature: CreatureUid) -> bool {
    crate::creature_by_id(&state.creatures, creature.0).is_some_and(|c| c.current_hp > 0)
}

/// `ArrowCmd.TryShootArrow`: mark the next arrow `Away` and run
/// `Hook.AfterArrowShot` (which fires even when there is no arrow).
/// `from_card` is whether the shot has a card source. Returns the slot.
pub fn try_shoot_arrow(state: &mut CombatState, from_card: bool) -> Option<usize> {
    sync_shoot_slot(state);
    let slot = next_arrow(state);
    if let Some(i) = slot {
        fool_mut(state).arrows[i].status = ArrowStatus::Away;
    }
    after_arrow_shot(state, slot, from_card);
    slot
}

/// One arrow attack command's damage: `hits` hits of `base` from the player
/// to `target` with `Move | FromArrow`, wrapped in the Aim / Quietus
/// `BeforeAttack` latch and `AfterAttack` spend. Returns the total HP loss.
pub fn arrow_attack(state: &mut CombatState, target: CreatureUid, base: i32, hits: u32) -> i32 {
    begin_arrow_attack(state);
    let card = crate::card_effects::originating_card_id(state);
    let mut total = 0;
    for _ in 0..hits {
        if !is_alive(state, target) {
            break;
        }
        total += crate::damage::apply_damage(
            state,
            CreatureRef::Player,
            target,
            base,
            ValueProps::MOVE.union(ValueProps::FROM_ARROW),
            card,
        );
    }
    end_arrow_attack(state);
    total
}

/// `AimPower.BeforeAttack` / `QuietusPower.BeforeAttack` for a player
/// `Move | FromArrow` attack command. Call [`end_arrow_attack`] after it.
/// Card-lane attacks that carry `FromArrow` without going through
/// [`shoot_arrow_at_target`] (BlossomShot, RainShot, ...) bracket their
/// damage with this pair too.
pub fn begin_arrow_attack(state: &mut CombatState) {
    let aim = player_power_amount(state, PowerKind::Aim);
    let quietus = player_power_amount(state, PowerKind::Quietus) > 0;
    let card = crate::card_effects::resolving_card_uid(state);
    let f = fool_mut(state);
    if card.is_some() {
        f.attacked_card = card;
    }
    f.arrow_attack_depth = f.arrow_attack_depth.saturating_add(1);
    if f.aim_latch.is_none() && aim > 0 {
        f.aim_latch = Some(aim);
        f.aim_source = card;
    }
    if quietus {
        f.quietus_latched = true;
    }
}

/// The resolving Attack card's play issues no `AttackCommand` after all
/// (Nail on a target without an arrow in it; `ShootArrowAtTarget` on a dead
/// target or with an empty quiver). Rust latches Vigor and Gigantification at
/// play start (`hooks::latch_vigor_at_play_start`), but C# latches them in
/// `BeforeAttack`, which never runs here, so the latches are released
/// unspent — unless this play already began an arrow attack, whose latch
/// stands.
pub fn attack_skipped(state: &mut CombatState) {
    let Some(card) = crate::card_effects::resolving_card_uid(state) else {
        return;
    };
    if fool(state).is_some_and(|f| f.attacked_card == Some(card)) {
        return;
    }
    if state.player.vigor_latch_card == Some(card) {
        state.player.vigor_latch_card = None;
        let vigor: SmallVec<[crate::PowerUid; 2]> = state
            .player_power_uids()
            .into_iter()
            .filter(|u| {
                state
                    .powers
                    .get_by_id(u.0)
                    .is_some_and(|p| p.kind == PowerKind::Vigor)
            })
            .collect();
        for u in vigor {
            if let Some(p) = state.powers.get_mut_by_id(u.0) {
                p.counter = 0;
            }
        }
    }
    let card_i32 = i32::try_from(card.0.get()).unwrap_or(i32::MAX);
    let giants: SmallVec<[crate::PowerUid; 2]> = state.player_power_uids().into_iter().collect();
    for u in giants {
        if let Some(p) = state.powers.get_mut_by_id(u.0) {
            if p.kind == PowerKind::Gigantification && p.counter == card_i32 {
                p.counter = 0;
            }
        }
    }
}

/// `AimPower.ModifyDamageAdditive`'s source guard: while a command is
/// latched, damage whose card source (the resolving card) is not the latched
/// command's `ModelSource` gets no Aim (`commandToModify != null &&
/// cardSource != null && cardSource != commandToModify.ModelSource`).
#[must_use]
pub fn aim_blocked_for_source(state: &CombatState) -> bool {
    let Some(f) = fool(state) else {
        return false;
    };
    if f.aim_latch.is_none() {
        return false;
    }
    match crate::card_effects::resolving_card_uid(state) {
        Some(source) => f.aim_source != Some(source),
        None => false,
    }
}

/// `AimPower.AfterAttack` (spend the latched start amount) and
/// `QuietusPower.AfterAttack` (remove itself) for the latched command.
pub fn end_arrow_attack(state: &mut CombatState) {
    let f = fool_mut(state);
    f.arrow_attack_depth = f.arrow_attack_depth.saturating_sub(1);
    if f.arrow_attack_depth > 0 {
        return;
    }
    let aim = f.aim_latch.take();
    f.aim_source = None;
    let quietus = std::mem::take(&mut f.quietus_latched);
    if let Some(spent) = aim {
        reduce_player_power(state, PowerKind::Aim, spent);
    }
    if quietus {
        remove_player_power(state, PowerKind::Quietus);
    }
}

/// `ArrowCmd.ShootArrowAtTarget(target, player, attackCommand, ctx)`: shoot
/// the next arrow and, if there was one, run the attack and impale.
/// Returns whether an arrow was shot (`false`: dead target or empty quiver,
/// and then NO attack happens).
pub fn shoot_arrow_at_target(
    state: &mut CombatState,
    target: CreatureUid,
    base: i32,
    hits: u32,
) -> bool {
    shoot_arrow_at_target_with(state, target, |state| {
        arrow_attack(state, target, base, hits);
    })
}

/// `ArrowCmd.ShootArrowAtTarget(target, player, ctx, card, actionToPerform)`:
/// like [`shoot_arrow_at_target`] with an arbitrary action in place of the
/// attack (run between the shot and the impale).
pub fn shoot_arrow_at_target_with(
    state: &mut CombatState,
    target: CreatureUid,
    action: impl FnOnce(&mut CombatState),
) -> bool {
    if !enabled(state) || !is_alive(state, target) {
        attack_skipped(state);
        return false;
    }
    let from_card = crate::card_effects::resolving_card_uid(state).is_some();
    let Some(slot) = try_shoot_arrow(state, from_card) else {
        attack_skipped(state);
        return false;
    };
    action(state);
    impale(state, target, slot);
    true
}

/// The tail of `ArrowCmd.ImpaleTarget` after the attack: ensure the
/// player's `ImpaledPower` (amount 1, `HitCounter` 3), add the arrow, run
/// `Hook.AfterArrowImpaled`, then destroy a Quietus arrow or — if the attack
/// killed the target — dislodge it right back.
pub fn impale(state: &mut CombatState, target: CreatureUid, slot: usize) {
    let alive = is_alive(state, target);
    if alive && impaled_power_uid(state, target).is_none() {
        crate::card_effects::apply_power_to_creature_with_applier(
            state,
            target,
            PowerKind::Impaled,
            1,
            state.player.creature_uid,
        );
        if let Some(uid) = impaled_power_uid(state, target) {
            if let Some(p) = state.powers.get_mut_by_id(uid.0) {
                p.counter = IMPALED_HITS;
            }
        }
    }
    {
        let f = fool_mut(state);
        let slot_u8 = slot as u8;
        if let Some((_, list)) = f.impaled.iter_mut().find(|(c, _)| *c == target) {
            list.push(slot_u8);
        } else {
            let mut list = SmallVec::new();
            list.push(slot_u8);
            f.impaled.push((target, list));
        }
    }
    after_arrow_impaled(state);
    let destroy = fool(state).is_some_and(|f| f.arrows[slot].destroy_after_impaled);
    if destroy {
        // `DestroyArrow`: a live target drops it (`ForceRemoveArrow`, which
        // removes the power once empty); the slot is destroyed either way.
        // On a dead target C# skips `ForceRemoveArrow`, but there the arrow
        // only ever sat in an `ImpaledPower` that `PowerCmd.Apply` could not
        // attach (`CanReceivePowers`), so nothing is left in the corpse.
        if fool(state).is_some_and(|f| f.arrows[slot].status != ArrowStatus::Destroyed) {
            remove_arrow_from(state, target, slot);
            if is_alive(state, target) {
                remove_impaled_if_no_arrows(state, target);
            }
            fool_mut(state).arrows[slot].status = ArrowStatus::Destroyed;
        }
    } else if !is_alive(state, target)
        && fool(state).is_some_and(|f| {
            f.impaled
                .iter()
                .any(|(c, l)| *c == target && l.contains(&(slot as u8)))
        })
    {
        dislodge_first(state, target);
    }
}

fn remove_arrow_from(state: &mut CombatState, target: CreatureUid, slot: usize) {
    let f = fool_mut(state);
    if let Some(i) = f.impaled.iter().position(|(c, _)| *c == target) {
        f.impaled[i].1.retain(|s| usize::from(*s) != slot);
        if f.impaled[i].1.is_empty() {
            f.impaled.remove(i);
        }
    }
}

/// `ImpaledPower.RemoveIfNoArrows`.
fn remove_impaled_if_no_arrows(state: &mut CombatState, target: CreatureUid) {
    if arrows_in(state, target) == 0 {
        if let Some(uid) = impaled_power_uid(state, target) {
            state.drop_power(uid.0);
        }
    }
}

/// `ArrowCmd.DislodgeArrow(combatState, impaled, ...)`: pop the FIRST arrow
/// stuck in `target`. A `DestroyAfterImpaled` arrow is destroyed silently;
/// otherwise `Hook.AfterArrowDislodged` runs and the arrow is retrieved.
/// Does NOT remove an emptied `ImpaledPower` (callers do, as in C#).
pub fn dislodge_first(state: &mut CombatState, target: CreatureUid) {
    let slot = {
        let f = fool_mut(state);
        let Some(i) = f.impaled.iter().position(|(c, _)| *c == target) else {
            return;
        };
        let slot = usize::from(f.impaled[i].1.remove(0));
        if f.impaled[i].1.is_empty() {
            f.impaled.remove(i);
        }
        slot
    };
    if fool(state).is_some_and(|f| f.arrows[slot].destroy_after_impaled) {
        fool_mut(state).arrows[slot].status = ArrowStatus::Destroyed;
        return;
    }
    after_arrow_dislodged(state);
    let a = &mut fool_mut(state).arrows[slot];
    a.status = ArrowStatus::Retrieved;
    a.destroy_after_impaled = false;
}

/// `ArrowCmd.DislodgeArrow(target, player, ...)` (the card-facing form):
/// dislodge one arrow, then drop the emptied `ImpaledPower`.
pub fn dislodge_arrow(state: &mut CombatState, target: CreatureUid) {
    if arrows_in(state, target) == 0 {
        return;
    }
    dislodge_first(state, target);
    remove_impaled_if_no_arrows(state, target);
}

/// `ImpaledPower.DislodgeAllArrows` + `PowerCmd.Remove`.
pub fn dislodge_all(state: &mut CombatState, target: CreatureUid) {
    while arrows_in(state, target) > 0 {
        dislodge_first(state, target);
    }
    if let Some(uid) = impaled_power_uid(state, target) {
        state.drop_power(uid.0);
    }
}

/// `ImpaledPower.AfterDamageReceived`: a powered attack hit from the
/// applier (the player) on the impaled creature ticks `HitCounter`; at 0 it
/// resets to 3, dislodges one arrow and drops an emptied power.
pub fn on_damage_received(
    state: &mut CombatState,
    target: CreatureUid,
    dealer: CreatureRef,
    props: ValueProps,
) {
    if dealer != CreatureRef::Player || !props.is_powered() {
        return;
    }
    let Some(uid) = impaled_power_uid(state, target) else {
        return;
    };
    let fire = match state.powers.get_mut_by_id(uid.0) {
        Some(p) => {
            p.counter -= 1;
            if p.counter <= 0 {
                p.counter = IMPALED_HITS;
                true
            } else {
                false
            }
        }
        None => false,
    };
    if fire {
        dislodge_first(state, target);
        remove_impaled_if_no_arrows(state, target);
    }
}

/// `ImpaledPower.AfterDeath`: the impaled creature died while the player
/// lives → every arrow comes back and the power goes.
pub fn on_creature_death(state: &mut CombatState, dead: CreatureUid) {
    if fool(state).is_none() || state.player.current_hp <= 0 {
        return;
    }
    if arrows_in(state, dead) > 0 || impaled_power_uid(state, dead).is_some() {
        dislodge_all(state, dead);
    }
}

/// `CreatureCmd.Escape` dislodges every arrow from the fleeing creature.
pub fn on_creature_escape(state: &mut CombatState, fled: CreatureUid) {
    if fool(state).is_some() && arrows_in(state, fled) > 0 {
        dislodge_all(state, fled);
    }
}

/// `CardPile.AddInternal`: a `Shoot` card entering a combat pile gives a
/// character with no base arrow slots and no arrows its first slot. Every
/// combat card lives in the card arena from the moment it enters a pile, so
/// "a Shoot card was ever added" is "the arena holds a Shoot card"; the slot
/// is granted lazily here (and read virtually by the getters) instead of at
/// each of the engine's many card-creation sites.
#[must_use]
///
/// The Fool's own `BaseArrowSlotCount` (1) slot, which `PlayerCombatState`
/// builds when combat starts, is likewise granted on first use.
pub fn shoot_slot_pending(state: &CombatState) -> bool {
    enabled(state)
        && arrow_slot_count_raw(state) == 0
        && (state.player.character == Some(crate::potions::CharacterId::Fool)
            || state.cards.holds_shoot_tag())
}

/// Materialise a pending [`shoot_slot_pending`] slot.
pub fn sync_shoot_slot(state: &mut CombatState) {
    if shoot_slot_pending(state) {
        gain_arrow_slot(state);
    }
}

// ───────────────────────────── arrow hooks ─────────────────────────────

/// `Hook.AfterArrowShot`: player powers (Quietus), then cards
/// (WhistlingArrows in the discard pile returns to hand).
fn after_arrow_shot(state: &mut CombatState, slot: Option<usize>, from_card: bool) {
    let Some(slot) = slot else {
        return;
    };
    if from_card && player_power_amount(state, PowerKind::Quietus) > 0 {
        fool_mut(state).arrows[slot].destroy_after_impaled = true;
    }
    let whistling: SmallVec<[CardUid; 2]> = state
        .player
        .discard
        .iter()
        .copied()
        .filter(|u| card_id_of(state, *u) == Some(CardId::WhistlingArrows))
        .collect();
    for uid in whistling {
        state.move_card_to_hand(uid);
    }
}

/// `Hook.AfterArrowImpaled`: ExplosiveArrows deals its Amount, Unpowered, to
/// every hittable enemy.
fn after_arrow_impaled(state: &mut CombatState) {
    let amount = player_power_amount(state, PowerKind::ExplosiveArrows);
    if amount > 0 {
        crate::card_effects::deal_damage_to_all_alive_enemies(
            state,
            amount,
            ValueProps::UNPOWERED,
            None,
        );
    }
}

/// `EpicRetrieval` `EnergyVar(3)` (v0.113.0 `Cards/EpicRetrieval.cs`): the
/// energy its `AfterArrowDislodged` gains when it is the dislodging source
/// (4 upgraded).
pub const EPIC_RETRIEVAL_ENERGY: i32 = 3;

/// `Hook.AfterArrowDislodged` for a player arrow: Yah (once per turn,
/// energy), Twine (block), EpicRetrieval (energy when it is the source).
fn after_arrow_dislodged(state: &mut CombatState) {
    // YahPower: `triggeredThisTurn` lives in the instance's counter as the
    // round it fired (reset by `BeforeSideTurnStart` = a new player round).
    let round = i32::from(state.round);
    let mut yah_energy = 0;
    for uid in state.player_power_uids() {
        if let Some(p) = state.powers.get_mut_by_id(uid.0) {
            if p.kind == PowerKind::Yah && p.amount > 0 && p.counter != round {
                p.counter = round;
                yah_energy += p.amount;
            }
        }
    }
    if yah_energy > 0 {
        state.gain_energy(yah_energy);
    }
    if crate::relic_effects::helpers::player_owns_relic(state, crate::RelicKind::Twine) {
        crate::card_effects::gain_block_player_unpowered(state, 4);
    }
    if let Some(uid) = crate::card_effects::resolving_card_uid(state) {
        if let Some(c) = state.cards.get(uid.0.get() as usize - 1) {
            if c.card_id == CardId::EpicRetrieval {
                // `EpicRetrieval.cs`: `EnergyVar(3)`, `OnUpgrade` +1.
                let energy = if c.upgraded {
                    EPIC_RETRIEVAL_ENERGY + 1
                } else {
                    EPIC_RETRIEVAL_ENERGY
                };
                state.gain_energy(energy);
            }
        }
    }
}

// ───────────────────────────── melody ─────────────────────────────

fn turn_stamp(state: &CombatState) -> (u16, bool) {
    (state.round, state.side == Side::Player)
}

fn melody_now(state: &CombatState) -> Option<&MelodyTurn> {
    let m = &fool(state)?.melody;
    (m.stamp == Some(turn_stamp(state))).then_some(m)
}

fn melody_now_mut(state: &mut CombatState) -> &mut MelodyTurn {
    let stamp = turn_stamp(state);
    let m = &mut fool_mut(state).melody;
    if m.stamp != Some(stamp) {
        *m = MelodyTurn {
            stamp: Some(stamp),
            ..MelodyTurn::default()
        };
    }
    m
}

fn card_tags_of(state: &CombatState, uid: CardUid) -> CardTags {
    state
        .cards
        .get(uid.0.get() as usize - 1)
        .map_or(CardTags::NONE, |c| c.tags)
}

pub(crate) fn card_id_of(state: &CombatState, uid: CardUid) -> Option<CardId> {
    state.cards.get(uid.0.get() as usize - 1).map(|c| c.card_id)
}

/// `History.CardPlayStarted` for one play iteration of `card`.
pub fn note_play_started(state: &mut CombatState, card: CardUid) {
    if !enabled(state) {
        return;
    }
    if let Some(f) = state.player.fool.as_deref_mut() {
        f.attacked_card = None;
    }
    let tags = card_tags_of(state, card);
    sync_shoot_slot(state);
    // `EncorePower.BeforeCardPlayed` counts the plays that start while it is
    // on the owner; with no Encore the count restarts.
    let encore = state.player_power_amount(PowerKind::Encore) > 0;
    if !tags.has_melody() && state.player.fool.is_none() && !encore {
        return;
    }
    let f = fool_mut(state);
    f.encore_plays = if encore {
        f.encore_plays.saturating_add(1)
    } else {
        0
    };
    f.play_seq = f.play_seq.wrapping_add(1).max(1);
    let seq = f.play_seq;
    f.open_plays.push((seq, card));
    if tags.has_melody() {
        let m = melody_now_mut(state);
        for tag in CardTag::MELODY {
            if tags.contains(tag) {
                let i = note_index(tag).unwrap_or(0);
                m.started[i] = m.started[i].saturating_add(1);
                if m.first_started_seq[i] == 0 {
                    m.first_started_seq[i] = seq;
                }
            }
        }
    }
}

/// `History.CardPlayFinished` for one play iteration of `card`.
pub fn note_play_finished(state: &mut CombatState, card: CardUid) {
    if !enabled(state) || state.player.fool.is_none() {
        return;
    }
    let tags = card_tags_of(state, card);
    {
        let f = fool_mut(state);
        if let Some(i) = f.open_plays.iter().rposition(|(_, c)| *c == card) {
            f.last_finished_seq = f.open_plays.remove(i).0;
        }
    }
    if tags.has_melody() {
        let m = melody_now_mut(state);
        m.melody_cards_finished = m.melody_cards_finished.saturating_add(1);
        for tag in CardTag::MELODY {
            if tags.contains(tag) {
                let i = note_index(tag).unwrap_or(0);
                m.finished[i] = m.finished[i].saturating_add(1);
            }
        }
    }
}

/// `MelodyCmd.PlayedThisTurn(player, tag)` (finished plays).
#[must_use]
pub fn played_this_turn(state: &CombatState, tag: CardTag) -> bool {
    let Some(i) = note_index(tag) else {
        return false;
    };
    melody_now(state).is_some_and(|m| m.finished[i] > 0)
}

/// `MelodyCmd.MelodiesThisTurn`: distinct notes finished this turn.
#[must_use]
pub fn melodies_this_turn(state: &CombatState) -> i32 {
    CardTag::MELODY
        .iter()
        .filter(|t| played_this_turn(state, **t))
        .count() as i32
}

/// `MelodyCmd.MelodyCardsPlayedThisTurn`.
#[must_use]
pub fn melody_cards_played_this_turn(state: &CombatState) -> i32 {
    melody_now(state).map_or(0, |m| i32::from(m.melody_cards_finished))
}

/// Started plays this turn carrying `tag` (`History.CardPlaysStarted`
/// count, as LostLute / Motif read it).
#[must_use]
pub fn started_this_turn(state: &CombatState, tag: CardTag) -> i32 {
    let Some(i) = note_index(tag) else {
        return 0;
    };
    melody_now(state).map_or(0, |m| i32::from(m.started[i]))
}

/// `MelodyCmd.IsFirstPlayThisTurn(cardPlay, tag)` for the innermost open
/// play (the card currently resolving).
#[must_use]
pub fn is_first_play_this_turn(state: &CombatState, tag: CardTag) -> bool {
    let Some(i) = note_index(tag) else {
        return false;
    };
    let Some(f) = fool(state) else {
        return false;
    };
    let Some(&(seq, _)) = f.open_plays.last() else {
        return false;
    };
    melody_now(state).is_some_and(|m| m.first_started_seq[i] == seq)
}

/// Sequence number of the innermost open play, if any.
#[must_use]
pub fn current_play_seq(state: &CombatState) -> Option<u32> {
    fool(state)?.open_plays.last().map(|(s, _)| *s)
}

// ───────────────────────────── power helpers ─────────────────────────────

/// The player's total amount of `kind` (0 when absent).
#[must_use]
pub fn player_power_amount(state: &CombatState, kind: PowerKind) -> i32 {
    state.player_power_amount(kind)
}

/// `PowerCmd.ModifyAmount(power, -by)` on the player's `kind`, removing it
/// at 0.
pub fn reduce_player_power(state: &mut CombatState, kind: PowerKind, by: i32) {
    let mut drop = None;
    for uid in state.player_power_uids() {
        if let Some(p) = state.powers.get_mut_by_id(uid.0) {
            if p.kind == kind {
                p.amount = p.amount.saturating_sub(by);
                if p.amount <= 0 {
                    drop = Some(uid.0);
                }
                break;
            }
        }
    }
    if let Some(pid) = drop {
        state.drop_power(pid);
    }
}

/// `PowerCmd.Remove` on the player's `kind`.
pub fn remove_player_power(state: &mut CombatState, kind: PowerKind) {
    let pids: SmallVec<[core::num::NonZeroU32; 2]> = state
        .player_power_uids()
        .into_iter()
        .filter(|u| state.powers.get_by_id(u.0).is_some_and(|p| p.kind == kind))
        .map(|u| u.0)
        .collect();
    for pid in pids {
        state.drop_power(pid);
    }
}

/// Apply `amount` of `kind` to the player (stacking).
pub fn apply_to_player(state: &mut CombatState, kind: PowerKind, amount: i32) {
    if amount == 0 {
        return;
    }
    if let Some(uid) = state.player.creature_uid {
        crate::card_effects::apply_power_to_creature(state, uid, kind, amount);
    }
}

/// Fold the Fool state into a hasher (MCTS transposition table). Called
/// only when the state exists, so non-Fool hashes are unchanged. The play
/// sequence numbers are bookkeeping, not game state, and are left out
/// (only whether a note's first play happened is folded).
pub fn hash_into<H: core::hash::Hasher>(f: &FoolState, h: &mut H) {
    use core::hash::Hash;
    f.arrows.hash(h);
    f.impaled.hash(h);
    let m = &f.melody;
    m.stamp.hash(h);
    m.finished.hash(h);
    m.melody_cards_finished.hash(h);
    m.started.hash(h);
    m.first_started_seq.map(|s| s != 0).hash(h);
    f.aim_latch.hash(h);
    f.aim_source.hash(h);
    f.quietus_latched.hash(h);
    f.stretch_latch.hash(h);
    f.encore_plays.hash(h);
    f.lullaby_round.hash(h);
    f.attacked_card.hash(h);
}

fn generate_into_hand(state: &mut CombatState, id: CardId, upgraded: bool) {
    if let Some(card) = crate::lookup_card_factory().and_then(|f| f(id, upgraded)) {
        let _ = crate::card_effects::spawn_card_into_hand_with_origin(state, card, true);
    }
}

/// v0.113.0 `GodByrdPower.AfterPlayerTurnStart` (`GodByrdPower.cs`): for
/// the owning player, when `AmountOnTurnStart != 0`, deal `Amount` Unpowered
/// to every hittable enemy, then `PowerCmd.Remove(this)`. The turn-start
/// walk iterates the snapshot taken before the cluster ran, so a GodByrd
/// played this turn (`AmountOnTurnStart == 0`) waits for the next turn.
/// Not a Fool power — any character can evolve Byrdpip — but it lives with
/// the other v0.113.0 turn-start powers.
pub fn god_byrd_after_player_turn_start(
    state: &mut CombatState,
    uid: crate::PowerUid,
    amount: i32,
) {
    if amount == 0 {
        return;
    }
    crate::card_effects::deal_damage_to_all_alive_enemies(
        state,
        amount,
        ValueProps::UNPOWERED,
        None,
    );
    state.drop_power(uid.0);
}

/// `EcstasyPower.AfterPlayerTurnStart`: up to `amount` of the Sadness in
/// hand (`StableShuffle` on `CombatCardSelection`) transform into Joy.
pub fn ecstasy_after_player_turn_start(state: &mut CombatState, amount: i32) {
    if !enabled(state) || amount <= 0 {
        return;
    }
    let mut sad: SmallVec<[CardUid; 10]> = state
        .player
        .hand
        .iter()
        .copied()
        .filter(|u| {
            state
                .cards
                .get(u.0.get() as usize - 1)
                .is_some_and(|c| c.card_id == CardId::Sadness)
        })
        .collect();
    if sad.is_empty() {
        return;
    }
    // Every Sadness sorts equal, and the .NET sort of <= 16 ties keeps them
    // in hand order before the shuffle.
    let version = state.game_version;
    state
        .run_rng
        .combat_card_selection
        .shuffle_for(version, sad.as_mut_slice());
    let take = usize::try_from(amount).unwrap_or(0).min(sad.len());
    for old in sad.into_iter().take(take) {
        if let Some(joy) = crate::lookup_card_factory().and_then(|f| f(CardId::Joy, false)) {
            let _ = crate::card_effects::transform_card_in_place(state, old, joy);
        }
    }
}

/// `TarotCards.All`, in decomp order. All 22 are allowed in a single-player
/// standard run.
pub const TAROT_CARDS: [CardId; 22] = [
    CardId::Judgement,
    CardId::TheFool,
    CardId::TheHighPriestess,
    CardId::TheEmperor,
    CardId::TheEmpress,
    CardId::TheHangedMan,
    CardId::TheHierophant,
    CardId::TheChariot,
    CardId::Justice,
    CardId::TheMagician,
    CardId::TheMoon,
    CardId::TheTower,
    CardId::StrengthCard,
    CardId::Death,
    CardId::Temperance,
    CardId::TheDevil,
    CardId::TheHermit,
    CardId::TheLovers,
    CardId::TheStar,
    CardId::TheSun,
    CardId::TheWorld,
    CardId::WheelOfFortune,
];

/// `EnnuiPower.BeforeHandDraw`: its `Cards` var (the power's `counter`, +1
/// per Ennui card played) of Sadness into hand.
pub fn ennui_before_hand_draw(state: &mut CombatState, sadness: i32) {
    for _ in 0..sadness.max(0) {
        generate_into_hand(state, CardId::Sadness, false);
    }
}

/// `MythicFormPower.BeforeHandDraw`: `amount` random tarots
/// (`CombatCardSelection.NextItem`), the first `upgraded` (its `Upgraded`
/// var, the power's `counter`) of them upgraded.
pub fn mythic_form_before_hand_draw(state: &mut CombatState, amount: i32, upgraded: i32) {
    for i in 0..amount.max(0) {
        let idx = state
            .run_rng
            .combat_card_selection
            .next_int_exclusive(0, TAROT_CARDS.len() as i32) as usize;
        generate_into_hand(state, TAROT_CARDS[idx], i < upgraded);
    }
}

/// `MelodyCmd.IsFirstPlayThisTurn(cardPlay, tag)` for the play that just
/// finished (the `cardPlay` of `AfterCardPlayed`).
#[must_use]
pub fn finished_play_was_first(state: &CombatState, tag: CardTag) -> bool {
    let (Some(i), Some(f)) = (note_index(tag), fool(state)) else {
        return false;
    };
    f.last_finished_seq != 0
        && melody_now(state).is_some_and(|m| m.first_started_seq[i] == f.last_finished_seq)
}

/// LostLute / Motif: how many of the card's melody notes have exactly one
/// started play this turn (`History.CardPlaysStarted.Count(..) == 1`).
fn first_note_count(state: &CombatState, tags: crate::cards::CardTags) -> i32 {
    CardTag::MELODY
        .iter()
        .filter(|t| tags.contains(**t) && started_this_turn(state, **t) == 1)
        .count() as i32
}

/// The Fool power kinds [`after_card_played_power`] has an arm for.
pub const AFTER_CARD_PLAYED_KINDS: [PowerKind; 5] = [
    PowerKind::SongPath,
    PowerKind::ViolentChord,
    PowerKind::Lullaby,
    PowerKind::Motif,
    PowerKind::LostLute,
];

/// One Fool power's `AfterCardPlayed`, for the owner's play of a card with
/// `tags`, in the player-power walk order.
pub fn after_card_played_power(
    state: &mut CombatState,
    kind: PowerKind,
    amount: i32,
    tags: crate::cards::CardTags,
) {
    if amount <= 0 || !enabled(state) {
        return;
    }
    match kind {
        // `SongPathPower`: the turn's first Cho play → Amount block.
        PowerKind::SongPath if tags.contains(CardTag::Cho) => {
            if finished_play_was_first(state, CardTag::Cho) {
                crate::card_effects::gain_block_player_unpowered(state, amount);
            }
        }
        // `ViolentChordPower`: the turn's first Wah play → Amount unpowered
        // damage to every hittable enemy (one multi-target command).
        PowerKind::ViolentChord if tags.contains(CardTag::Wah) => {
            if finished_play_was_first(state, CardTag::Wah) {
                let targets: SmallVec<[CreatureUid; 4]> = state
                    .enemy_uids
                    .iter()
                    .copied()
                    .filter(|u| {
                        state
                            .creatures
                            .iter()
                            .any(|c| c.id == u.0 && c.current_hp > 0)
                    })
                    .collect();
                crate::damage::with_deferred_enemy_deaths(state, |state| {
                    for t in targets {
                        crate::damage::apply_damage(
                            state,
                            crate::hooks::CreatureRef::Player,
                            t,
                            amount,
                            crate::hooks::ValueProps::UNPOWERED,
                            None,
                        );
                    }
                });
            }
        }
        // `LullabyPower`: once per turn, after all three notes were played →
        // Amount unpowered block.
        PowerKind::Lullaby => {
            let round = state.round;
            if !state.is_over
                && fool(state).is_some_and(|f| f.lullaby_round != Some(round))
                && melodies_this_turn(state) == 3
            {
                fool_mut(state).lullaby_round = Some(round);
                crate::card_effects::gain_block_player_unpowered(state, amount);
            }
        }
        // `MotifPower`: draw Amount per note the card is the turn's only
        // started play of.
        PowerKind::Motif => {
            let n = first_note_count(state, tags);
            if n > 0 {
                state.draw_into_hand_card_effect(n * amount);
            }
        }
        // `LostLutePower`: auto-play Amount from the draw pile per such note.
        PowerKind::LostLute => {
            let n = first_note_count(state, tags);
            if n > 0 {
                crate::combat_loop::auto_play_n_from_draw_top(state, n * amount, false);
            }
        }
        _ => {}
    }
}

/// `LullabyPower.AfterApplied`: applied after all three notes were already
/// played this turn, it counts as having fired.
pub fn lullaby_after_applied(state: &mut CombatState) {
    if enabled(state) && melodies_this_turn(state) == 3 {
        let round = state.round;
        fool_mut(state).lullaby_round = Some(round);
    }
}

/// `EncorePower.ShouldPlay`: no further owned plays once `Amount` have
/// started since it was applied.
#[must_use]
pub fn encore_blocks(state: &CombatState) -> bool {
    let amount = state.player_power_amount(PowerKind::Encore);
    amount > 0
        && state.player.fool.as_ref().map_or(0, |f| f.encore_plays)
            >= u32::try_from(amount).unwrap_or(0)
}

// ── StretchPower ───────────────────────────────────────────────────────

/// `StretchPower.BeforeCardPlayed`: an owned card that `GainsBlock` latches
/// the play and the Stretch amount when it started (first latch wins).
pub fn latch_stretch_at_play_start(state: &mut CombatState, card: CardUid) {
    if !enabled(state) || fool(state).is_some_and(|f| f.stretch_latch.is_some()) {
        return;
    }
    let amount = state.player_power_amount(PowerKind::Stretch);
    if amount <= 0 {
        return;
    }
    let version = state.game_version;
    let gains_block = state
        .cards
        .get(card.0.get() as usize - 1)
        .is_some_and(|c| c.gains_block_for(version));
    if gains_block {
        fool_mut(state).stretch_latch = Some((card, amount, false));
    }
}

/// `StretchPower.ModifyBlockAdditive`: the Stretch amount for powered
/// card block (`IsPoweredCardOrMonsterMoveBlock`) of the latched play.
#[must_use]
pub fn stretch_block_bonus(
    state: &CombatState,
    amount: i32,
    props: crate::hooks::ValueProps,
    card_source: Option<CardUid>,
) -> i32 {
    use crate::hooks::ValueProps;
    let Some(card) = card_source else {
        return 0;
    };
    if !props.contains(ValueProps::MOVE) || props.contains(ValueProps::UNPOWERED) {
        return 0;
    }
    match fool(state).and_then(|f| f.stretch_latch) {
        Some((latched, _, _)) if latched == card => amount,
        _ => 0,
    }
}

/// `StretchPower.ModifyBlockAdditive` with `cardPlay == null`: a card-face
/// preview (`BlockVar` / `CalculatedBlockVar` run `Hook.ModifyBlock(...,
/// card, null)`) gets the Stretch `Amount`, unspent, whenever no play is
/// latched, and 0 while one is. Gameplay block always carries its play
/// (`GainBlock` passes `cardPlay?.Card` as the source), so this arm is
/// preview-only and [`stretch_block_bonus`] keeps the play semantics.
///
/// Returns the state to fold `card`'s preview against: `None` means `state`
/// itself; `Some` is a clone where `card` stands in as the latched play, so
/// the shared bonus yields `Amount` — cloned only when Stretch is held and
/// nothing is latched.
#[must_use]
pub fn stretch_preview_state(state: &CombatState, card: CardUid) -> Option<CombatState> {
    let amount = state.player_power_amount(PowerKind::Stretch);
    if amount <= 0 || fool(state).is_some_and(|f| f.stretch_latch.is_some()) {
        return None;
    }
    let mut staged = state.clone();
    fool_mut(&mut staged).stretch_latch = Some((card, amount, false));
    Some(staged)
}

/// Record that the latched play's block took the Stretch bonus
/// (`gaveBonusBlock = true`).
pub fn note_stretch_block(
    state: &mut CombatState,
    props: crate::hooks::ValueProps,
    card_source: Option<CardUid>,
) {
    let amount = state.player_power_amount(PowerKind::Stretch);
    if amount > 0 && stretch_block_bonus(state, amount, props, card_source) != 0 {
        if let Some(latch) = state
            .player
            .fool
            .as_mut()
            .and_then(|f| f.stretch_latch.as_mut())
        {
            latch.2 = true;
        }
    }
}

/// `StretchPower.AfterCardPlayed` for the latched play: release the latch
/// and, if the play took the bonus, spend the start amount.
pub fn release_stretch(state: &mut CombatState, card: Option<CardUid>) {
    let Some(f) = state.player.fool.as_mut() else {
        return;
    };
    match f.stretch_latch {
        Some((latched, start, gave)) if Some(latched) == card => {
            f.stretch_latch = None;
            if gave {
                reduce_player_power(state, PowerKind::Stretch, start);
            }
        }
        _ => {}
    }
}

/// The pick step of `KiChant.cs`: `FromCombatPile(Draw, 1, top count)` to
/// hand, then `CardCmd.Discard` the other top cards still in the draw pile
/// (the select's [`crate::PostChoiceEffect::DiscardUnpickedFromDraw`]). A
/// single top card is taken without a screen (`cards.Count <= MinSelect`);
/// the grid sorts by `(Rarity, Id)` like every Draw-pile select. Runs at the
/// end of the card's `OnPlay`, or after the Stratagem pick its shuffle
/// opened ([`crate::PostChoiceEffect::KiChantPick`]).
pub fn ki_chant_pick(state: &mut CombatState, count: usize) {
    let mut top: SmallVec<[crate::CardUid; 4]> =
        state.player.draw.iter().take(count).copied().collect();
    match top.len() {
        0 => {}
        1 => {
            let uid = top[0];
            if let Some(pos) = state.player.draw.iter().position(|&u| u == uid) {
                state.player.draw.remove(pos);
                crate::cards::unpin(&mut state.player, uid);
                let _ = crate::cards::try_add_to_hand(&mut state.player, uid);
            }
        }
        _ => {
            let rank = crate::lookup_card_rarity_rank();
            let version = state.game_version;
            let rarity = |state: &CombatState, uid: crate::CardUid| {
                let id = state.cards[uid.0.get() as usize - 1].card_id;
                rank.map_or(0, |f| f(id, version))
            };
            top.sort_by(|a, b| {
                rarity(state, *a)
                    .cmp(&rarity(state, *b))
                    .then_with(|| crate::cards::draw_select_tiebreak(&state.cards, *a, *b, version))
            });
            let options: Vec<crate::ChoiceOption> = top
                .iter()
                .map(|&uid| {
                    let c = &state.cards[uid.0.get() as usize - 1];
                    crate::ChoiceOption {
                        card_id: c.card_id,
                        upgraded: c.upgraded,
                        card_uid: Some(uid),
                        source_pile: Some(crate::cards::PileKind::Draw),
                    }
                })
                .collect();
            state.open_hand_choice(crate::PendingChoice {
                kind: crate::ChoiceKind::AddFromDrawByType,
                options,
                min_select: 1,
                max_select: 1,
                can_skip: false,
                post_effect: Some(crate::PostChoiceEffect::DiscardUnpickedFromDraw),
                picked: SmallVec::new(),
            });
        }
    }
}

/// `ShowOffPower.AfterDeath` for an enemy death: each instance raises its
/// ShowOff card's `IncreasedAim` (the card's `VarKey::Generic`; the card
/// uid sits in the power's `counter`) by its Amount and removes itself. The
/// C# also raises the deck version's: the run bridge folds the var back onto
/// the master-deck card when the combat ends.
pub fn show_off_after_death(state: &mut CombatState) {
    let jobs: SmallVec<[(core::num::NonZeroU32, i32, i32); 2]> = state
        .player_power_uids()
        .into_iter()
        .filter_map(|u| {
            let p = state.powers.get_by_id(u.0)?;
            (p.kind == PowerKind::ShowOff).then_some((u.0, p.amount, p.counter))
        })
        .collect();
    for (pid, amount, card) in jobs {
        let idx = usize::try_from(card).ok().filter(|i| *i > 0).map(|i| i - 1);
        if let Some(c) = idx.and_then(|i| state.cards.get_mut(i)) {
            let key = crate::cards::VarKey::Generic;
            let v = c.combat_var(key).saturating_add(amount);
            c.set_combat_var(key, v);
        }
        state.drop_power(pid);
    }
}
