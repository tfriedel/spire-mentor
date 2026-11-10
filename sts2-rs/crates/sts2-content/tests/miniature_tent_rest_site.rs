//! MiniatureTent lets the owner take more than one rest-site option.
//!
//! C# `MiniatureTent.ShouldDisableRemainingRestSiteOptions` returns `false`
//! for its owner, and `RestSiteSynchronizer.ChooseOption` (v0.111.0
//! `:202-215`) then removes only the chosen option, finishing the rest site
//! when none remain. The player can also leave with options still open
//! (`BeforeLocalRestSiteExited`).

use sts2_core::cards::{Card, CardId};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::runstate::RestSiteOption;
use sts2_core::RelicId;

fn rest_site(with_tent: bool) -> RunManager {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = RunManager::new(0, CharacterId::Ironclad)
        .with_content_callbacks(sts2_content::run_callbacks::ironclad_content_callbacks());
    if with_tent {
        rm.run_state.relics.push(RelicId::MiniatureTent);
    }
    rm.run_state.master_deck.clear();
    for card_id in [CardId::Bash, CardId::PommelStrike, CardId::Anger] {
        rm.run_state.master_deck.push(Card {
            card_id,
            upgraded: false,
            ..Card::pod_defaults()
        });
    }
    rm.run_state.max_hp = 80;
    rm.run_state.current_hp = 40;
    rm.enter_rest_site();
    rm
}

#[test]
fn without_tent_one_option_ends_the_rest_site() {
    let mut rm = rest_site(false);
    rm.take_action(RunAction::RestOption { idx: 0 });
    assert_eq!(rm.phase(), Phase::MapChoice);
}

#[test]
fn tent_keeps_the_other_options_after_rest() {
    let mut rm = rest_site(true);
    let hp_before = rm.run_state.current_hp;
    rm.take_action(RunAction::RestOption { idx: 0 });
    assert!(rm.run_state.current_hp > hp_before, "rested");
    assert_eq!(rm.phase(), Phase::RestSite, "tent keeps the rest site open");
    assert_eq!(
        rm.run_state.rest_site_options(),
        vec![RestSiteOption::Smith { enabled: true }],
        "only the chosen option is removed"
    );
    assert_eq!(
        rm.get_available_actions(),
        vec![RunAction::RestOption { idx: 0 }, RunAction::LeaveShop],
    );

    // Smith opens the upgrade pick; resolving it uses up the last option.
    rm.take_action(RunAction::RestOption { idx: 0 });
    assert!(matches!(
        rm.get_available_actions().first(),
        Some(RunAction::SmithCard { .. })
    ));
    rm.take_action(RunAction::SmithCard { idx: 0 });
    assert!(rm.run_state.master_deck.iter().any(|c| c.upgraded));
    assert_eq!(rm.phase(), Phase::MapChoice, "no options left");
    assert!(rm.run_state.rest_site_taken.is_empty());
}

#[test]
fn tent_owner_can_leave_with_options_open() {
    let mut rm = rest_site(true);
    assert!(
        !rm.get_available_actions().contains(&RunAction::LeaveShop),
        "leaving is offered only after an option was taken"
    );
    rm.take_action(RunAction::RestOption { idx: 0 });
    rm.take_action(RunAction::LeaveShop);
    assert_eq!(rm.phase(), Phase::MapChoice);
    assert!(rm.run_state.rest_site_taken.is_empty());
    assert_eq!(
        rm.run_state.rest_site_options().len(),
        2,
        "next rest site is fresh"
    );
}

/// A rest site whose deck is fully upgraded: C# `SmithRestSiteOption
/// .IsEnabled => Deck.UpgradableCardCount != 0` is false.
fn rest_site_fully_upgraded(with_tent: bool) -> RunManager {
    let mut rm = rest_site(with_tent);
    for c in &mut rm.run_state.master_deck {
        c.upgraded = true;
    }
    rm.enter_rest_site();
    rm
}

/// C# `NRestSiteButton.Create` marks a disabled option `_isUnclickable`
/// (`NRestSiteButton.cs:229`) and `OnRelease` ignores the click, so a
/// disabled Smith is not a move: it must not be offered, and picking it
/// anyway must not consume the rest site.
#[test]
fn disabled_smith_is_not_offered() {
    let mut rm = rest_site_fully_upgraded(false);
    assert_eq!(
        rm.run_state.rest_site_options(),
        vec![
            RestSiteOption::Rest,
            RestSiteOption::Smith { enabled: false }
        ],
        "the disabled Smith is still listed (greyed) at its slot"
    );
    assert_eq!(
        rm.get_available_actions(),
        vec![RunAction::RestOption { idx: 0 }]
    );
    let hp = rm.run_state.current_hp;
    assert_eq!(
        rm.take_action(RunAction::RestOption { idx: 1 }),
        sts2_core::run_manager::ActionResult::InvalidForPhase,
        "a disabled option is rejected, not a free skip"
    );
    assert_eq!(rm.phase(), Phase::RestSite);
    assert_eq!(rm.run_state.current_hp, hp);
}

/// MiniatureTent + nothing upgradable: after resting only the disabled Smith
/// remains. C# keeps it (`options.Count == 1`, the rest site is not
/// completed) but `AfterSelectingOptionAsync` enables Proceed
/// (`NRestSiteRoom.cs:591`), so the only legal move is to leave.
#[test]
fn tent_with_only_disabled_options_left_offers_leave() {
    let mut rm = rest_site_fully_upgraded(true);
    rm.take_action(RunAction::RestOption { idx: 0 });
    assert_eq!(rm.phase(), Phase::RestSite);
    assert_eq!(
        rm.run_state.rest_site_options(),
        vec![RestSiteOption::Smith { enabled: false }]
    );
    assert_eq!(rm.get_available_actions(), vec![RunAction::LeaveShop]);
    rm.take_action(RunAction::LeaveShop);
    assert_eq!(rm.phase(), Phase::MapChoice);
}

/// MeatCleaver's Cook needs 2 removable cards (`CookRestSiteOption.IsEnabled
/// => GetRemovableCardCount >= 2`); with one card it is listed but not offered.
#[test]
fn disabled_cook_is_not_offered() {
    let mut rm = rest_site(false);
    rm.run_state.relics.push(RelicId::MeatCleaver);
    rm.run_state.master_deck.truncate(1);
    rm.enter_rest_site();
    let opts = rm.run_state.rest_site_options();
    let cook = opts
        .iter()
        .position(|o| matches!(o, RestSiteOption::Cook { .. }))
        .expect("MeatCleaver adds Cook");
    assert!(!opts[cook].enabled());
    let actions = rm.get_available_actions();
    assert!(!actions.contains(&RunAction::RestOption { idx: cook }));
    assert!(actions.contains(&RunAction::RestOption { idx: 0 }));
}
