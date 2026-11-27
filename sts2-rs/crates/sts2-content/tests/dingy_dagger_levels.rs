//! v0.113.0 Dingy Dagger upgrades any number of times (GH #1391).
//!
//! `DingyDagger.cs`: `MaxUpgradeLevel => 999999999`; `OnUpgrade` adds
//! `3 + CurrentUpgradeLevel` damage after `UpgradeInternal` bumped the level,
//! so 7 → 11 → 16 → 22. `DowngradeInternal` resets the level to 0.

use sts2_content::cards::v0111::card_pod_for;
use sts2_content::run_callbacks::new_run_manager_for_character;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::GameVersion;

const V113: GameVersion = GameVersion::V0_113_0;

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
}

#[test]
fn dingy_dagger_upgrades_again_and_again() {
    install();
    let mut card = card_pod_for(CardId::DingyDagger, false, V113).expect("spec");
    assert_eq!((card.base_damage, card.upgrade_level()), (7, 0));
    for (level, damage) in [(1, 11), (2, 16), (3, 22), (4, 29)] {
        assert!(card.is_upgradable(), "level {}", card.upgrade_level());
        assert!(card.upgrade_for(V113));
        assert_eq!((card.upgrade_level(), card.base_damage), (level, damage));
        assert!(card.upgraded);
    }
    // A downgrade goes all the way back, as C# `DowngradeInternal`.
    assert!(card.downgrade_for(V113));
    assert_eq!((card.upgrade_level(), card.base_damage), (0, 7));
    assert!(!card.upgraded);
}

#[test]
fn every_other_card_still_upgrades_once() {
    install();
    let mut card = card_pod_for(CardId::StrikeIronclad, false, V113).expect("spec");
    assert!(card.upgrade_for(V113));
    assert!(!card.is_upgradable());
    assert!(!card.upgrade_for(V113));
    assert_eq!(card.upgrade_level(), 1);
}

#[test]
fn smith_offers_an_upgraded_dingy_dagger_and_upgrades_it_again() {
    install();
    let mut rm = new_run_manager_for_character(1, CharacterId::Ironclad).with_game_version(V113);
    let mut dagger = card_pod_for(CardId::DingyDagger, true, V113).expect("spec");
    assert_eq!(dagger.base_damage, 11);
    dagger.uid = sts2_core::CardUid::new(9_000).expect("nz");
    rm.run_state.master_deck.push(dagger);
    let offered = rm.run_state.upgradable_deck_cards();
    assert!(offered.contains(&CardId::DingyDagger), "{offered:?}");
    let at = rm
        .run_state
        .master_deck
        .iter()
        .position(|c| c.card_id == CardId::DingyDagger)
        .expect("in deck");
    assert!(rm.run_state.master_deck[at].upgrade_for(V113));
    let up = rm.run_state.master_deck[at];
    assert_eq!((up.upgrade_level(), up.base_damage), (2, 16));
}

/// v0.114.0 `CardCmd.Downgrade` takes one level off (`DowngradeInternal(1)`
/// replays `level - 1` upgrades); `RemoveAllUpgrades` (Dampen) still resets.
#[test]
fn a_v0114_downgrade_takes_one_level_off() {
    install();
    let v114 = GameVersion::V0_114_0;
    let mut card = card_pod_for(CardId::DingyDagger, false, v114).expect("spec");
    for _ in 0..3 {
        assert!(card.upgrade_for(v114));
    }
    assert_eq!((card.upgrade_level(), card.base_damage), (3, 22));
    assert!(card.downgrade_for(v114));
    assert_eq!((card.upgrade_level(), card.base_damage), (2, 16));
    assert!(card.remove_all_upgrades_for(v114));
    assert_eq!((card.upgrade_level(), card.base_damage), (0, 7));
    assert!(!card.downgrade_for(v114), "nothing left to take off");
}
