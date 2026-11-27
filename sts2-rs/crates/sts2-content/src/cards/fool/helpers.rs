//! Shared helpers for the Fool card handlers.
use sts2_core::card_effects;
use sts2_core::cards::{Card, CardId};
use sts2_core::{CombatState, CreatureUid, PowerKind};

use super::specs;
use crate::cards::status::card_from_spec;

/// `base_damage` of the resolving card (upgrade-aware).
pub(crate) fn current_damage(state: &CombatState) -> i32 {
    card_effects::resolving_card_base_damage(state)
}

/// `base_block` of the resolving card (upgrade-aware).
pub(crate) fn current_block(state: &CombatState) -> i32 {
    card_effects::resolving_card_base_block(state)
}

/// Whether the resolving card is upgraded (for secondary vars).
pub(crate) fn upgraded(state: &CombatState) -> bool {
    card_effects::resolving_card_upgraded(state)
}

/// `up` when the resolving card is upgraded, else `base`.
pub(crate) fn var(state: &CombatState, base: i32, up: i32) -> i32 {
    if upgraded(state) {
        up
    } else {
        base
    }
}

/// Whether the fight is on game build v0.114.0 or later (the Fool rework,
/// GH #1924).
pub(crate) fn v0114(state: &CombatState) -> bool {
    state.game_version >= sts2_core::GameVersion::V0_114_0
}

/// [`var`] with the build's `(base, upgraded)` pair: `v0113` through
/// v0.113.0, `v0114` from v0.114.0 on.
pub(crate) fn var_by(state: &CombatState, v0113: (i32, i32), v0114: (i32, i32)) -> i32 {
    let (base, up) = if self::v0114(state) { v0114 } else { v0113 };
    var(state, base, up)
}

/// `PowerCmd.Apply<kind>(Owner.Creature, amount)`.
pub(crate) fn apply_to_player(state: &mut CombatState, kind: PowerKind, amount: i32) {
    if amount <= 0 {
        return;
    }
    if let Some(uid) = state.player.creature_uid {
        card_effects::apply_power_to_creature(state, uid, kind, amount);
    }
}

/// `PowerCmd.Apply<kind>(target, amount)` on an enemy.
pub(crate) fn apply_to_enemy(
    state: &mut CombatState,
    t: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    card_effects::apply_power_to_enemy(state, t, kind, amount);
}

/// `DamageCmd.Attack(dmg).TargetingAllOpponents(..)`: one batched attack.
pub(crate) fn damage_all_enemies(state: &mut CombatState, dmg: i32) {
    let card = card_effects::originating_card_id(state);
    let _ = card_effects::deal_damage_to_all_alive_enemies(
        state,
        dmg,
        sts2_core::hooks::ValueProps::MOVE,
        card,
    );
}

/// A fresh Fool card instance (no upgrade).
pub(crate) fn fool_card(id: CardId) -> Card {
    let spec = specs::FOOL_SPECS
        .iter()
        .map(|f| f())
        .find(|s| s.id == id)
        .expect("Fool card spec");
    card_from_spec(&spec)
}

/// `CardPileCmd.AddGeneratedCardsToCombat(n x CreateCard<id>, Hand, Owner)`.
pub(crate) fn generate_into_hand(state: &mut CombatState, id: CardId, n: i32) {
    for _ in 0..n {
        let _ = card_effects::spawn_card_into_hand(state, fool_card(id));
    }
}
