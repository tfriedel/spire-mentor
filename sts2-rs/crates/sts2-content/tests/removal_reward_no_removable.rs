//! A card-removal reward with no removable card left removes nothing.
//!
//! C# `CardRemovalReward.OnSelect` → `DoUnsyncedCardRemoval` →
//! `CardSelectCmd.FromDeckForRemoval`, whose `FromDeckGeneric` filters
//! `IsRemovable` (`!Eternal`) and returns an empty selection, with no
//! select screen, when nothing is left (`CardSelectCmd.cs:771-774`, both
//! builds). Each removal reward is its own reward, so N rewards over a deck
//! with fewer than N removable cards ask only as many times as there are
//! removable cards. Parity seed Necrobinder 6125 v0.111.0: three Forbidden
//! Grimoires (Eternal) and one Debilitate, two Grimoire removals; C# removed
//! Debilitate and went to the map, Rust surfaced an empty second removal.

use sts2_core::cards::{Card, CardId};
use sts2_core::RunState;

fn card(card_id: CardId) -> Card {
    Card {
        card_id,
        ..Card::pod_defaults()
    }
}

#[test]
fn removal_rewards_queue_only_as_many_picks_as_removable_cards() {
    sts2_content::cards::dispatch::install_default_registry();
    let mut rs = RunState::new();
    rs.interactive_card_removal = true;
    for id in [
        CardId::ForbiddenGrimoire,
        CardId::ForbiddenGrimoire,
        CardId::ForbiddenGrimoire,
        CardId::DebilitateCard,
    ] {
        rs.master_deck.push(card(id));
    }
    assert_eq!(
        rs.removal_offer_order().len(),
        1,
        "only Debilitate is removable"
    );

    rs.remove_cards_from_deck(2);
    assert_eq!(rs.pending_deck_removals, 1);

    // A later removal reward while that pick is still queued finds nothing
    // left over once the queued pick has taken Debilitate.
    rs.remove_cards_from_deck(1);
    assert_eq!(rs.pending_deck_removals, 1);
}
