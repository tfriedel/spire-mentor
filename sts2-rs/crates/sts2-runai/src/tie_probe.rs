//! Move identity for the tie-rule probes (GH #1022 phase 3):
//! `tie_rule_probe` (sts2-bench) and `sts2_rs.tie_rule_picks` (the human
//! agreement probe) key every root move with [`move_key`], so the two count
//! the same thing.
//!
//! An [`Action`] names a hand *slot*, and two copies of a card in different
//! slots are different actions but the same move. A probe that compared
//! actions would call a search that swaps two Strikes between seeds unstable,
//! and a rule that picks the other Strike a flip. The key is the card (id,
//! upgrade, cost) and the target instead, so neither counts.

use sts2_core::{Action, CardUid, CombatState};

/// `{card id}{+ when upgraded}({cost})`, or `?` for a uid not in the state.
fn card_label(state: &CombatState, uid: Option<CardUid>) -> String {
    uid.and_then(|uid| state.cards.iter().find(|c| c.uid == uid))
        .map_or_else(
            || "?".to_owned(),
            |card| {
                let plus = if card.upgraded { "+" } else { "" };
                format!("{:?}{plus}({})", card.card_id, card.local_cost())
            },
        )
}

fn hand_card(state: &CombatState, hand_index: u8) -> String {
    card_label(
        state,
        state.player.hand.get(usize::from(hand_index)).copied(),
    )
}

fn with_target(head: String, target: Option<u8>) -> String {
    match target {
        Some(t) => format!("{head}>{t}"),
        None => head,
    }
}

fn potion(state: &CombatState, slot: u8, target: Option<u8>) -> String {
    let name = state
        .player
        .potions
        .get(usize::from(slot))
        .map_or("?", |p| p.python_name());
    with_target(format!("potion {name}"), target)
}

/// The move `action` makes at `state`, with copies of a card merged.
///
/// A choice selection is keyed by its options' cards, sorted, so picking
/// either of two identical curses to discard is one move too.
#[must_use]
pub fn move_key(state: &CombatState, action: &Action) -> String {
    match action {
        Action::PlayCard { hand_index, target } => {
            with_target(hand_card(state, *hand_index), *target)
        }
        Action::PlayCardAtAlly { hand_index, ally } => {
            format!("{}>ally{ally}", hand_card(state, *hand_index))
        }
        Action::EndTurn => "EndTurn".to_owned(),
        Action::UsePotion { slot, target } => potion(state, *slot, *target),
        Action::DrinkPotion {
            potion_slot,
            target,
        } => potion(state, *potion_slot, *target),
        Action::ResolveChoice { selection } => {
            let options = state.pending_choice.as_ref().map(|c| &c.options);
            let mut cards: Vec<String> = selection
                .iter()
                .map(|&i| {
                    options.and_then(|o| o.get(usize::from(i))).map_or_else(
                        || format!("#{i}"),
                        |o| {
                            if o.card_uid.is_some() {
                                card_label(state, o.card_uid)
                            } else {
                                let plus = if o.upgraded { "+" } else { "" };
                                format!("{:?}{plus}", o.card_id)
                            }
                        },
                    )
                })
                .collect();
            cards.sort();
            format!("choose[{}]", cards.join(","))
        }
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_of_a_card_are_one_move() {
        let state = crate::policy::mcts_fixtures::build_vertical_slice(7);
        let mut keys: Vec<(String, u8)> = Vec::new();
        for i in 0..u8::try_from(state.player.hand.len()).expect("small hand") {
            keys.push((
                move_key(
                    &state,
                    &Action::PlayCard {
                        hand_index: i,
                        target: Some(0),
                    },
                ),
                i,
            ));
        }
        // Two slots holding the same card at the same cost share a key;
        // slots holding different cards do not.
        let mut by_card = std::collections::BTreeMap::<String, Vec<u8>>::new();
        for (k, i) in &keys {
            by_card.entry(k.clone()).or_default().push(*i);
        }
        assert!(
            by_card.values().any(|slots| slots.len() > 1),
            "fixture holds no duplicate card: {keys:?}"
        );
        assert!(by_card.len() > 1, "fixture holds one card only: {keys:?}");
        for (k, slots) in &by_card {
            let label = card_label(
                &state,
                state.player.hand.get(usize::from(slots[0])).copied(),
            );
            assert_eq!(k, &format!("{label}>0"));
        }
        assert_eq!(move_key(&state, &Action::EndTurn), "EndTurn");
        assert_ne!(
            move_key(
                &state,
                &Action::PlayCard {
                    hand_index: 0,
                    target: Some(0)
                }
            ),
            move_key(
                &state,
                &Action::PlayCard {
                    hand_index: 0,
                    target: Some(1)
                }
            )
        );
    }
}
