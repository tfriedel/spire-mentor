//! Lava Lamp leaves the act-opening Ancient's card slates alone.
//!
//! C# `LavaLamp.TryModifyCardRewardOptionsLate` returns early unless
//! `RunState.CurrentRoom is CombatRoom` (`LavaLamp.cs:64-69`, both builds).
//! After an act-1/2 boss, the next act's Ancient is entered as an `EventRoom`
//! (`EventRoom.cs:81`), so a slate rolled there (Sea Glass, taken from the
//! Ancient) is not upgraded even after a clean boss fight. Rust keeps the
//! boss room as `current_room` until the next map move, and the Ancient's
//! room entry has already cleared Lava Lamp's took-damage flag, so the slate
//! was upgraded (Silent v0.111.0 seed 5451: `HYPERBEAM|1` vs C# `|0`).

use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::runstate::RoomType;
use sts2_core::{CombatState, GameVersion, RelicId};

/// A run that just won a clean act-1 boss fight with Lava Lamp, parked on the
/// boss card reward.
fn after_clean_boss_fight(version: GameVersion) -> RunManager {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = RunManager::new(5451, CharacterId::Silent)
        .with_content_callbacks(sts2_content::run_callbacks::ironclad_content_callbacks())
        .with_game_version(version);
    rm.run_state.relics.push(RelicId::LavaLamp);
    rm.run_state.act_index = 1;
    rm.debug_set_room_type(RoomType::Boss);
    rm.debug_enter_combat(
        CombatState {
            is_over: true,
            player_won: true,
            ..CombatState::default()
        },
        RoomType::Boss,
    );
    let _ = rm.take_action(RunAction::EndTurn);
    assert_eq!(
        rm.phase(),
        Phase::CardReward,
        "{version:?}: boss card reward"
    );
    rm
}

/// Obtain Sea Glass and return its 15-card slate's upgrade flags.
fn sea_glass_slate_upgrades(rm: &mut RunManager) -> Vec<bool> {
    let rs = &mut rm.run_state;
    let grants_before = rs.pending_card_grants.len();
    rs.relics.push(RelicId::SeaGlass);
    rs.last_obtained_relic = Some(RelicId::SeaGlass);
    let _ = sts2_core::run_effects::dispatch_after_obtained(rs);
    assert_eq!(rs.pending_card_grants.len(), grants_before + 1);
    let grant = rs.pending_card_grants.last().expect("Sea Glass grant");
    assert_eq!(grant.options.len(), 15);
    grant.options.iter().map(|c| c.upgraded).collect()
}

#[test]
fn lava_lamp_does_not_upgrade_sea_glass_in_the_ancient_room() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        // Control: still in the boss's combat room, the clean fight upgrades.
        let mut control = after_clean_boss_fight(version);
        let upgrades = sea_glass_slate_upgrades(&mut control);
        assert!(
            upgrades.iter().any(|&u| u),
            "{version:?}: a clean combat room upgrades the slate: {upgrades:?}"
        );

        // The boss reward drains into the next act's Ancient (an EventRoom).
        let mut rm = after_clean_boss_fight(version);
        while rm.phase() == Phase::CardReward {
            let _ = rm.take_action(RunAction::PickCardReward { idx: None });
        }
        // Whether the Ancient opens a screen or not, the player stands in its
        // room until the next map move (C# opens the map from there).
        assert!(
            matches!(rm.phase(), Phase::BossRelic | Phase::MapChoice),
            "{version:?}: in the Ancient, got {:?}",
            rm.phase()
        );
        assert!(rm.run_state.in_non_combat_room(), "{version:?}");
        let upgrades = sea_glass_slate_upgrades(&mut rm);
        assert!(
            upgrades.iter().all(|&u| !u),
            "{version:?}: the Ancient is not a CombatRoom: {upgrades:?}"
        );

        // The next map move leaves the Ancient for a real room.
        if rm.phase() == Phase::MapChoice {
            rm.debug_clear_pending_choices();
            let mv = rm
                .get_available_actions()
                .into_iter()
                .find(|a| matches!(a, RunAction::Move { .. }))
                .expect("a map move");
            let _ = rm.take_action(mv);
            assert!(
                !rm.run_state.in_ancient_room,
                "{version:?}: left the Ancient"
            );
        }
    }
}
