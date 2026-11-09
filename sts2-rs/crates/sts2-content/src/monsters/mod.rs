pub mod dsl;

// Act-1 has two zone variants in STS2: Overgrowth (default) and
// Underdocks (alternate). The Python pin labels Underdocks as `act4.py`
// — a historical misnomer; the C# game has only 3 acts. We use C#
// canonical zone names (see `MegaCrit.Sts2.Core.Models.Acts.*`).
pub mod act1_overgrowth;
pub mod act1_underdocks;
pub mod act1_weak;
pub mod act2_hive;
pub mod act3_glory;
pub mod forecast;
pub mod shared;

mod legacy_power_surface;

pub(crate) use legacy_power_surface::{
    apply_power_to_player_via_legacy_surface, apply_signed_power_to_player,
};

/// Helpers shared by the per-act monster test modules.
#[cfg(test)]
pub(crate) mod test_util {
    use sts2_core::monster_ai::{MonsterAi, MonsterState};
    use sts2_core::monsters::Intent;
    use sts2_core::{CombatState, CreatureUid, GameVersion, MonsterKind};

    use crate::monster_dispatch::{build_ai_for_version, MonsterSpawnArgs, SpawnPosition};

    /// Every `(game build, ascension)` cell a mid-combat-spawn ascension test
    /// runs: both builds, each side of the `DeadlyEnemies` (A9) threshold, and
    /// A8 (`ToughEnemies`) to pin that the attack tables do not fire a level
    /// early.
    pub(crate) const SPAWN_ASC_MATRIX: [(GameVersion, u8); 8] = [
        (GameVersion::V0_107_1, 0),
        (GameVersion::V0_107_1, 8),
        (GameVersion::V0_107_1, 9),
        (GameVersion::V0_107_1, 10),
        (GameVersion::V0_111_0, 0),
        (GameVersion::V0_111_0, 8),
        (GameVersion::V0_111_0, 9),
        (GameVersion::V0_111_0, 10),
    ];

    /// Every enemy of `kind`, in `enemy_uids` order.
    pub(crate) fn enemies_of_kind(state: &CombatState, kind: MonsterKind) -> Vec<CreatureUid> {
        state
            .enemy_uids
            .iter()
            .copied()
            .filter(|u| {
                state
                    .creatures
                    .iter()
                    .any(|c| c.id == u.0 && c.species == Some(kind))
            })
            .collect()
    }

    /// `uid`'s AI. Panics when the creature or its AI is missing.
    pub(crate) fn creature_ai(state: &CombatState, uid: CreatureUid) -> &MonsterAi {
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .unwrap_or_else(|| panic!("no creature {uid:?}"))
            .ai
            .as_ref()
            .unwrap_or_else(|| panic!("creature {uid:?} has no AI"))
    }

    /// The first intent of move `move_id` in `uid`'s AI. Panics when the
    /// creature, its AI or the move is missing, so a wrong id fails loudly
    /// instead of passing vacuously.
    pub(crate) fn ai_move_intent(state: &CombatState, uid: CreatureUid, move_id: &str) -> Intent {
        match creature_ai(state, uid).states.get(move_id) {
            Some(MonsterState::Move(m)) => *m
                .intents
                .first()
                .unwrap_or_else(|| panic!("{uid:?} move {move_id} has no intents")),
            _ => panic!("{uid:?} has no move {move_id}"),
        }
    }

    /// Every move of `ai` with its intents, in state-table order.
    fn move_table(ai: &MonsterAi) -> Vec<(&'static str, Vec<Intent>)> {
        ai.states
            .iter()
            .filter_map(|(id, s)| match s {
                MonsterState::Move(m) => Some((*id, m.intents.to_vec())),
                _ => None,
            })
            .collect()
    }

    /// Assert that the AI `uid` carries is, move for move, the AI the
    /// dispatcher builds for an encounter member of `kind` at this combat's
    /// ascension and game build (GH #847: a summon is never softer than its
    /// summoner).
    pub(crate) fn assert_ai_is_dispatcher_build(
        state: &CombatState,
        uid: CreatureUid,
        kind: MonsterKind,
        args: MonsterSpawnArgs,
    ) {
        assert_ai_is_dispatcher_build_with(state, uid, kind, args, |_| {});
    }

    /// [`assert_ai_is_dispatcher_build`], with `adjust` applied to the
    /// dispatcher build first — for a spawn whose death effect re-telegraphs
    /// a move from state the dispatcher cannot see (Axebot's BOOT_UP
    /// Strength depends on the respawn's stock, GH #893).
    pub(crate) fn assert_ai_is_dispatcher_build_with(
        state: &CombatState,
        uid: CreatureUid,
        kind: MonsterKind,
        args: MonsterSpawnArgs,
        adjust: impl FnOnce(&mut sts2_core::monster_ai::MonsterAi),
    ) {
        let mut want = build_ai_for_version(
            kind,
            args,
            SpawnPosition::default(),
            state.ascension,
            state.game_version,
        );
        adjust(&mut want);
        let have = move_table(creature_ai(state, uid));
        assert!(!have.is_empty(), "{kind:?} {uid:?} has an empty move table");
        assert_eq!(
            have,
            move_table(&want),
            "{kind:?} {uid:?} at A{} {:?}: spawned AI differs from the dispatcher build",
            state.ascension,
            state.game_version
        );
    }
}
