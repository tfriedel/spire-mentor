//! v0.113.0 Fool relics: SongBow, Storyteller, ExerciseMat,
//! WhittlingKnife, ConductorsBaton, BizarreLamp and Ocarina.
//!
//! Fertilizer (an `AfterCardGeneratedForCombat` listener) lives in
//! `sts2_core::card_effects`, and Twine (an `AfterArrowDislodged` listener)
//! lives in `sts2_core::fool`, because neither hook is a relic channel.
//!
//! Every listener is gated on the relic being owned. These relics only
//! exist from v0.113.0, so ownership is the version gate; the
//! explicit `game_version` checks below make that independent of how a
//! scenario was assembled.

use sts2_core::card_effects::EffectResult;
use sts2_core::cards::CardId;
use sts2_core::powers::PowerKind;
use sts2_core::relic_effects::{
    register_after_card_played, register_after_side_turn_start, register_before_combat_start,
    register_before_side_turn_start, register_before_turn_end, register_owned_after_shuffle_bus,
    register_owned_before_hand_draw, AfterCardPlayedFn, AfterShuffleFn, AfterSideTurnStartFn,
    BeforeCombatStartFn, BeforeHandDrawFn, BeforeSideTurnStartFn, BeforeTurnEndFn, CardEventCtx,
    ShuffleCtx,
};
use sts2_core::relics::RelicId;
use sts2_core::{CombatState, GameVersion, RelicKind, Side};

/// `SongBow` `PowerVar<AimPower>(6m)`.
pub const SONG_BOW_AIM: i32 = 6;
/// `Storyteller` `PowerVar<AimPower>(12m)`.
pub const STORYTELLER_AIM: i32 = 12;
/// `Storyteller` `PowerVar<AimPower>("EndOfTurnAim", 3m)`.
pub const STORYTELLER_END_OF_TURN_AIM: i32 = 3;
/// `ExerciseMat` `PowerVar<StretchPower>(6m)`.
pub const EXERCISE_MAT_STRETCH: i32 = 6;
/// `WhittlingKnife` `PowerVar<AimPower>(5m)`.
pub const WHITTLING_KNIFE_AIM: i32 = 5;
/// `ConductorsBaton`: melodies played this turn before it fires.
pub const CONDUCTORS_BATON_MELODIES: i32 = 3;

fn active(state: &CombatState, kind: RelicKind) -> bool {
    state.game_version >= GameVersion::V0_113_0 && state.relics.owns(kind)
}

/// The first non-melted relic of `kind`, mutably.
fn relic_mut(state: &mut CombatState, kind: RelicKind) -> Option<&mut sts2_core::Relic> {
    state
        .relics
        .iter_mut()
        .find(|r| !r.is_melted && r.kind == kind)
}

pub fn install_pool() {
    register_before_combat_start(RelicId::SongBow, listener_song_bow as BeforeCombatStartFn);
    register_before_combat_start(
        RelicId::Storyteller,
        listener_storyteller_combat_start as BeforeCombatStartFn,
    );
    register_before_turn_end(
        RelicId::Storyteller,
        listener_storyteller_turn_end as BeforeTurnEndFn,
    );
    register_after_side_turn_start(
        RelicId::ExerciseMat,
        listener_exercise_mat as AfterSideTurnStartFn,
    );
    register_owned_after_shuffle_bus(
        RelicId::WhittlingKnife,
        RelicKind::WhittlingKnife,
        listener_whittling_knife as AfterShuffleFn,
    );
    register_after_card_played(
        RelicId::ConductorsBaton,
        listener_conductors_baton_played as AfterCardPlayedFn,
    );
    register_before_side_turn_start(
        RelicId::ConductorsBaton,
        listener_conductors_baton_reset as BeforeSideTurnStartFn,
    );
    register_owned_before_hand_draw(
        RelicId::BizarreLamp,
        RelicKind::BizarreLamp,
        listener_bizarre_lamp as BeforeHandDrawFn,
    );
    register_owned_before_hand_draw(
        RelicId::Ocarina,
        RelicKind::Ocarina,
        listener_ocarina as BeforeHandDrawFn,
    );
}

/// `SongBow.AfterRoomEntered` (combat room): 6 Aim.
fn listener_song_bow(state: &mut CombatState) -> EffectResult {
    if !active(state, RelicKind::SongBow) {
        return EffectResult::NotImplemented;
    }
    sts2_core::fool::apply_to_player(state, PowerKind::Aim, SONG_BOW_AIM);
    EffectResult::Resolved
}

/// `Storyteller.AfterRoomEntered` (combat room): 12 Aim.
fn listener_storyteller_combat_start(state: &mut CombatState) -> EffectResult {
    if !active(state, RelicKind::Storyteller) {
        return EffectResult::NotImplemented;
    }
    sts2_core::fool::apply_to_player(state, PowerKind::Aim, STORYTELLER_AIM);
    EffectResult::Resolved
}

/// `Storyteller.BeforeSideTurnEnd` (owner's side): 3 Aim.
fn listener_storyteller_turn_end(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player || !active(state, RelicKind::Storyteller) {
        return EffectResult::NotImplemented;
    }
    sts2_core::fool::apply_to_player(state, PowerKind::Aim, STORYTELLER_END_OF_TURN_AIM);
    EffectResult::Resolved
}

/// `ExerciseMat.AfterSideTurnStart`: on the owner's first turn, 6 Stretch.
fn listener_exercise_mat(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player || state.turn_number() > 1 || !active(state, RelicKind::ExerciseMat) {
        return EffectResult::NotImplemented;
    }
    sts2_core::fool::apply_to_player(state, PowerKind::Stretch, EXERCISE_MAT_STRETCH);
    EffectResult::Resolved
}

/// `WhittlingKnife.AfterShuffle` (owner shuffled): 5 Aim.
fn listener_whittling_knife(state: &mut CombatState, _ctx: &ShuffleCtx) -> EffectResult {
    if !active(state, RelicKind::WhittlingKnife) {
        return EffectResult::NotImplemented;
    }
    sts2_core::fool::apply_to_player(state, PowerKind::Aim, WHITTLING_KNIFE_AIM);
    EffectResult::Resolved
}

/// `ConductorsBaton.AfterCardPlayed`: the first time in a turn that
/// `MelodyCmd.MelodiesThisTurn >= 3`, apply ConductorsBatonPower(1).
/// `relic.counter` holds C#'s `_triggeredThisTurn`.
fn listener_conductors_baton_played(state: &mut CombatState, _ctx: &CardEventCtx) -> EffectResult {
    if !active(state, RelicKind::ConductorsBaton) || state.is_over {
        return EffectResult::NotImplemented;
    }
    if sts2_core::fool::melodies_this_turn(state) < CONDUCTORS_BATON_MELODIES {
        return EffectResult::NotImplemented;
    }
    let Some(relic) = relic_mut(state, RelicKind::ConductorsBaton) else {
        return EffectResult::NotImplemented;
    };
    if relic.counter != 0 {
        return EffectResult::NotImplemented;
    }
    relic.counter = 1;
    sts2_core::fool::apply_to_player(state, PowerKind::ConductorsBaton, 1);
    EffectResult::Resolved
}

/// `ConductorsBaton.BeforeSideTurnStart` (owner's side): re-arm.
fn listener_conductors_baton_reset(state: &mut CombatState, side: Side) -> EffectResult {
    if side != Side::Player {
        return EffectResult::NotImplemented;
    }
    match relic_mut(state, RelicKind::ConductorsBaton) {
        Some(relic) => {
            relic.counter = 0;
            EffectResult::Resolved
        }
        None => EffectResult::NotImplemented,
    }
}

fn add_generated_to_hand(state: &mut CombatState, id: CardId) {
    if let Some(card) = crate::cards::spec_index::card_pod_from_id(id, false) {
        let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(state, card, true);
    }
}

/// `BizarreLamp.BeforeHandDraw` (turn 1): Joy and Sadness into hand.
fn listener_bizarre_lamp(state: &mut CombatState) -> EffectResult {
    if state.turn_number() > 1 || !active(state, RelicKind::BizarreLamp) {
        return EffectResult::NotImplemented;
    }
    add_generated_to_hand(state, CardId::Joy);
    add_generated_to_hand(state, CardId::Sadness);
    EffectResult::Resolved
}

/// `Ocarina.BeforeHandDraw` (turn 1): one random melody card into hand
/// (`GetDistinctForCombat(.., 1, CombatCardGeneration)`).
fn listener_ocarina(state: &mut CombatState) -> EffectResult {
    if state.turn_number() > 1 || !active(state, RelicKind::Ocarina) {
        return EffectResult::NotImplemented;
    }
    let mut pool = crate::cards::fool_pool::combat_melody_pool(None);
    if pool.is_empty() {
        return EffectResult::NotImplemented;
    }
    let version = state.game_version;
    state
        .run_rng
        .combat_card_generation
        .shuffle_for(version, pool.as_mut_slice());
    add_generated_to_hand(state, pool[0]);
    EffectResult::Resolved
}
