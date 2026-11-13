//! v0.113.0 Fool potions: PrecisionPotion, BottledSong, CrystallineEuphoria.
use super::prelude::*;
use sts2_core::cards::{CardId, CardTag, HAND_LIMIT};

/// `PrecisionPotion` `PowerVar<AimPower>(21m)`.
pub(crate) const PRECISION_POTION_AIM: i32 = 21;
/// `CrystallineEuphoria` `CardsVar(3)`: Joys added before the Sadness fill.
pub(crate) const CRYSTALLINE_EUPHORIA_JOYS: usize = 3;

fn add_generated(state: &mut CombatState, id: CardId, free_this_turn: bool) {
    if let Some(mut card) = crate::cards::spec_index::card_pod_from_id(id, false) {
        if free_this_turn {
            card.set_to_free_this_turn();
        }
        let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(state, card, true);
    }
}

/// `PrecisionPotion.OnUse`: 21 Aim on the (only) player.
pub(crate) fn drink_precision_potion(state: &mut CombatState, _target: Option<u8>) -> EffectResult {
    if let Some(uid) = state.player.creature_uid {
        apply_power_to_creature(state, uid, PowerKind::Aim, PRECISION_POTION_AIM);
    }
    EffectResult::Resolved
}

/// `BottledSong.OnUse`: for Wah, Cho and Ki in turn, one
/// `GetDistinctForCombat(.., 1, CombatCardGeneration)` pick among the
/// melody cards carrying that tag, set free this turn; then all of them go
/// to hand together.
pub(crate) fn drink_bottled_song(state: &mut CombatState, _target: Option<u8>) -> EffectResult {
    let mut picks: smallvec::SmallVec<[CardId; 3]> = smallvec::SmallVec::new();
    for tag in CardTag::MELODY {
        let mut pool = crate::cards::fool_pool::combat_melody_pool(Some(tag));
        if pool.is_empty() {
            continue;
        }
        let version = state.game_version;
        state
            .run_rng
            .combat_card_generation
            .shuffle_for(version, pool.as_mut_slice());
        picks.push(pool[0]);
    }
    for id in picks {
        add_generated(state, id, true);
    }
    EffectResult::Resolved
}

/// `CrystallineEuphoria.OnUse`: 3 Joy into hand, then fill the hand to
/// `MaxCardsInHand` with Sadness (the count is taken after the Joys land).
pub(crate) fn drink_crystalline_euphoria(
    state: &mut CombatState,
    _target: Option<u8>,
) -> EffectResult {
    for _ in 0..CRYSTALLINE_EUPHORIA_JOYS {
        add_generated(state, CardId::Joy, false);
    }
    let room = HAND_LIMIT.saturating_sub(state.player.hand.len());
    for _ in 0..room {
        add_generated(state, CardId::Sadness, false);
    }
    EffectResult::Resolved
}
