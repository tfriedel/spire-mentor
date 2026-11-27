//! GH #1907: agent mode withholds the event options the game flags
//! `WillKillPlayer` (`EventOption.ThatDoesDamage` / `ThatDecreasesMaxHp` /
//! `ThatWillKillPlayerIf`) while another enabled option survives, as every
//! live chooser does. Spire Bench seed 180001036: at low HP the game arm
//! never saw Room Full of Cheese's Search, the native env still offered it.

use sts2_content::events::dispatch::{option_will_kill_player, present, resolve_by_index};
use sts2_core::events::EventId;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::RunManager;
use sts2_core::runstate::RunState;
use sts2_core::GameVersion;

fn run(agent: bool, version: GameVersion) -> RunManager {
    let mut rm =
        sts2_content::run_callbacks::new_run_manager_for_character(7, CharacterId::Ironclad);
    rm.run_state.game_version = version;
    if agent {
        rm.run_state.set_agent_decisions();
    }
    rm
}

fn option_ids(rs: &RunState) -> Vec<&'static str> {
    present(rs)
        .expect("in an event")
        .iter()
        .map(|o| o.option_id)
        .collect()
}

/// The ids of the presented options the game flags lethal right now.
fn lethal_ids(rs: &RunState) -> Vec<&'static str> {
    present(rs)
        .expect("in an event")
        .iter()
        .filter(|o| option_will_kill_player(rs, o))
        .map(|o| o.option_id)
        .collect()
}

#[test]
fn room_full_of_cheese_search_is_withheld_at_its_damage_or_less() {
    // `ThatDoesDamage(14)`: lethal at `CurrentHp <= 14`.
    for (hp, offered) in [
        (14, vec!["gorge"]),
        (1, vec!["gorge"]),
        (15, vec!["gorge", "search"]),
    ] {
        let mut rm = run(true, GameVersion::V0_111_0);
        rm.run_state.current_hp = hp;
        rm.run_state.enter_event_room(EventId::RoomFullOfCheese);
        assert_eq!(option_ids(&rm.run_state), offered, "hp {hp}");
    }
}

#[test]
fn resolving_by_index_follows_the_withheld_list() {
    let mut rm = run(true, GameVersion::V0_111_0);
    rm.run_state.current_hp = 10;
    rm.run_state.enter_event_room(EventId::RoomFullOfCheese);
    let relics_before = rm.run_state.relics.clone();
    // Index 0 is Gorge, the only option left; it must not take the damage.
    assert!(resolve_by_index(&mut rm.run_state, 0).is_some());
    assert_eq!(rm.run_state.current_hp, 10, "Gorge, not Search");
    assert_eq!(rm.run_state.relics, relics_before);
}

#[test]
fn parity_mode_still_offers_lethal_options() {
    let mut rm = run(false, GameVersion::V0_111_0);
    rm.run_state.current_hp = 5;
    rm.run_state.enter_event_room(EventId::RoomFullOfCheese);
    assert_eq!(option_ids(&rm.run_state), ["gorge", "search"]);
}

/// Every option C# flags resolves its threshold from what the port presents:
/// at 1 HP and 1 max HP each listed option reads lethal (parity mode, so
/// nothing is withheld before the check).
#[test]
fn every_flagged_option_resolves_its_threshold() {
    let cases: &[(EventId, &[&str])] = &[
        (EventId::AbyssalBaths, &["immerse"]),
        (EventId::BrainLeech, &["rip"]),
        (EventId::ColossalFlower, &["reach_deeper"]),
        (EventId::DenseVegetation, &["trudge_on"]),
        (EventId::DollRoom, &["take_time", "examine"]),
        (EventId::DrowningBeacon, &["climb"]),
        (EventId::HungryForMushrooms, &["fragrant"]),
        (EventId::JungleMazeAdventure, &["solo"]),
        (EventId::RoomFullOfCheese, &["search"]),
        (EventId::RoundTeaParty, &["pick_fight"]),
        (EventId::SlipperyBridge, &["hold_on"]),
        (EventId::SpiritGrafter, &["rejection"]),
        (EventId::StoneOfAllTime, &["push"]),
        (EventId::SunkenStatue, &["dive"]),
        (EventId::TabletOfTruth, &["decipher"]),
        (EventId::TheLegendsWereTrue, &["find_exit"]),
        (EventId::ThisOrThat, &["plain"]),
        (EventId::TrashHeap, &["dive_in"]),
        (EventId::UnrestSite, &["kill"]),
        (EventId::WhisperingHollow, &["hug"]),
    ];
    for version in [
        GameVersion::V0_107_1,
        GameVersion::V0_111_0,
        GameVersion::V0_113_0,
    ] {
        for &(event, flagged) in cases {
            let mut rm = run(false, version);
            rm.run_state.enter_event_room(event);
            rm.run_state.current_hp = 1;
            rm.run_state.max_hp = 1;
            assert_eq!(lethal_ids(&rm.run_state), flagged, "{version:?} {event:?}");
            // And at full health with a large max HP none is.
            rm.run_state.max_hp = 999;
            rm.run_state.current_hp = 999;
            assert!(
                lethal_ids(&rm.run_state).is_empty(),
                "{version:?} {event:?}"
            );
        }
    }
    for &(event, flagged) in &[
        (EventId::ThermaeNonupeipe, &["drink_waters"][..]),
        // Be Cautious is flagged, but its HpLoss is 5% of the current HP, so
        // it never reaches it (C# `IsAllowed`: `CurrentHp > CurrentHp * 0.05`).
        (EventId::WuppyInTrouble, &[][..]),
    ] {
        let mut rm = run(false, GameVersion::V0_113_0);
        rm.run_state.enter_event_room(event);
        rm.run_state.current_hp = 1;
        assert_eq!(lethal_ids(&rm.run_state), flagged, "{event:?}");
    }
}

/// The later pages keep their flag: Abyssal Baths' Linger, Colossal Flower's
/// deeper digs, Slippery Bridge's next Hold On, Tablet of Truth's next
/// Decipher.
#[test]
fn later_pages_keep_the_flag() {
    for (event, first, next) in [
        (EventId::AbyssalBaths, "immerse", "linger"),
        (EventId::ColossalFlower, "reach_deeper", "reach_deeper"),
        (EventId::SlipperyBridge, "hold_on", "hold_on"),
        (EventId::TabletOfTruth, "decipher", "decipher"),
    ] {
        let mut rm = run(false, GameVersion::V0_111_0);
        rm.run_state.current_hp = 80;
        rm.run_state.max_hp = 80;
        rm.run_state.enter_event_room(event);
        let idx = option_ids(&rm.run_state)
            .iter()
            .position(|o| *o == first)
            .expect("first option");
        assert!(resolve_by_index(&mut rm.run_state, idx).is_some());
        assert!(
            option_ids(&rm.run_state).contains(&next),
            "{event:?}: {:?}",
            option_ids(&rm.run_state)
        );
        rm.run_state.current_hp = 1;
        rm.run_state.max_hp = 1;
        assert!(
            lethal_ids(&rm.run_state).contains(&next),
            "{event:?} page 2: {:?}",
            present(&rm.run_state)
                .unwrap()
                .iter()
                .map(|o| (o.option_id, o.loc_key, o.vars.clone()))
                .collect::<Vec<_>>()
        );
    }
}

/// The workbench's human play (`offer_lethal_event_options`) is shown the
/// lethal options, as the game shows them; a policy acting on the run is
/// told which ones agent mode withholds and drops them.
#[test]
fn a_human_run_offers_the_lethal_options_and_names_them_for_the_policies() {
    use sts2_content::events::dispatch::{
        agent_withheld_event_options, drop_agent_withheld_event_choices,
    };
    use sts2_core::run_manager::RunAction;
    let mut rm = run(true, GameVersion::V0_111_0);
    rm.run_state.offer_lethal_event_options = true;
    rm.run_state.current_hp = 10;
    rm.run_state.enter_event_room(EventId::RoomFullOfCheese);
    assert_eq!(option_ids(&rm.run_state), ["gorge", "search"]);
    assert_eq!(agent_withheld_event_options(&rm.run_state), [1]);
    let actions = [
        RunAction::EventChoice { idx: 0 },
        RunAction::EventChoice { idx: 1 },
    ];
    assert_eq!(
        drop_agent_withheld_event_choices(&rm.run_state, &actions),
        [RunAction::EventChoice { idx: 0 }]
    );
    // The human may take it: Search's 14 damage resolves at index 1.
    assert!(resolve_by_index(&mut rm.run_state, 1).is_some());
    assert!(rm.run_state.current_hp <= 0, "{}", rm.run_state.current_hp);
}

/// Nothing is named where nothing is withheld: above the threshold, on an
/// agent-mode run that already withholds, and in parity mode.
#[test]
fn only_a_human_run_names_withheld_options() {
    use sts2_content::events::dispatch::agent_withheld_event_options;
    for (agent, offer, hp) in [(true, true, 80), (true, false, 10), (false, true, 10)] {
        let mut rm = run(agent, GameVersion::V0_111_0);
        rm.run_state.offer_lethal_event_options = offer;
        rm.run_state.current_hp = hp;
        rm.run_state.enter_event_room(EventId::RoomFullOfCheese);
        assert!(
            agent_withheld_event_options(&rm.run_state).is_empty(),
            "agent {agent} offer {offer} hp {hp}"
        );
    }
}

/// Trial's Double Down stays withheld on a human run: the engine cannot
/// abandon the run, so the option would be a free exit from the Trial.
#[test]
fn a_human_run_still_withholds_trial_double_down() {
    let mut rm = run(true, GameVersion::V0_111_0);
    rm.run_state.offer_lethal_event_options = true;
    rm.run_state.enter_event_room(EventId::Trial);
    let idx = option_ids(&rm.run_state)
        .iter()
        .position(|o| *o == "reject")
        .expect("reject");
    assert!(resolve_by_index(&mut rm.run_state, idx).is_some());
    assert_eq!(option_ids(&rm.run_state), ["reject_accept"]);
}
