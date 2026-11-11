//! Plain-text pretty-printers for `CombatState` slices. Mirrors the shape
//! of `sts_lightspeed/src/sim/PrintHelpers.cpp` but stays minimal — no
//! ANSI, no boxes; one line per entity.

use std::io::{self, Write};

use sts2_core::cards::Card;
use sts2_core::{Action, CombatState, Creature, PlayerCombatState};

pub fn print_combat_state<W: Write>(out: &mut W, state: &CombatState) -> io::Result<()> {
    writeln!(out, "── combat state ─────────────────────────")?;
    writeln!(out, "  phase   : {:?}", state.phase)?;
    writeln!(out, "  round   : {}", state.round)?;
    print_player(out, &state.player)?;
    writeln!(
        out,
        "  draw    : {}  discard: {}  exhaust: {}",
        state.player.draw.len(),
        state.player.discard.len(),
        state.player.exhaust.len()
    )?;
    writeln!(out, "  monsters:")?;
    for (slot, uid) in state.enemy_uids.iter().enumerate() {
        let creature_idx = state
            .creatures
            .iter()
            .position(|c| c.id == uid.0)
            .expect("enemy_uids point to live creatures");
        let c = &state.creatures[creature_idx];
        print_monster(out, slot, c)?;
    }
    print_hand(out, state)?;
    Ok(())
}

pub fn print_player<W: Write>(out: &mut W, p: &PlayerCombatState) -> io::Result<()> {
    writeln!(
        out,
        "  player  : HP {hp}/{max_hp}  block {block}  energy {energy}/{max_energy}",
        hp = p.current_hp,
        max_hp = p.max_hp,
        block = p.block,
        energy = p.energy,
        max_energy = p.max_energy,
    )
}

pub fn print_monster<W: Write>(out: &mut W, slot: usize, c: &Creature) -> io::Result<()> {
    let intent = c
        .intent
        .map_or_else(|| "—".to_string(), |i| format!("{i:?}"));
    // `slot N` (no brackets) — monsters are referenced by slot inside
    // action descriptions ("enemy 0"), but the slot itself is not a
    // dispatchable keystroke. Reserve `[N]` for the action menu.
    writeln!(
        out,
        "    slot {slot}: {name:<12} HP {hp}/{max} block {block}  intent {intent}  alive: {alive}",
        name = c.monster_id.0,
        hp = c.current_hp,
        max = c.max_hp,
        block = c.block,
        alive = c.current_hp > 0,
    )
}

pub fn print_hand<W: Write>(out: &mut W, state: &CombatState) -> io::Result<()> {
    writeln!(out, "  hand    : {} card(s)", state.player.hand.len())?;
    for (i, uid) in state.player.hand.iter().enumerate() {
        let card = card_by_uid(state, *uid);
        // `slot N` (no brackets) — hand indices appear inside action
        // descriptions ("play 3 0") but typing the hand index alone
        // doesn't dispatch. Reserve `[N]` for the action menu.
        writeln!(
            out,
            "    slot {i}: {name:?}  cost {cost}  ({ctype:?}/{ttype:?}{upg})",
            name = card.card_id,
            cost = card.local_cost(),
            ctype = card.card_type,
            ttype = card.target_type,
            upg = if card.upgraded { " +1" } else { "" },
        )?;
    }
    Ok(())
}

pub fn print_card_info<W: Write>(
    out: &mut W,
    state: &CombatState,
    hand_index: u8,
) -> io::Result<()> {
    let idx = usize::from(hand_index);
    if idx >= state.player.hand.len() {
        return writeln!(
            out,
            "  (no card at hand index {hand_index} — hand size {})",
            state.player.hand.len()
        );
    }
    let card = card_by_uid(state, state.player.hand[idx]);
    writeln!(out, "  card {hand_index}: {:?}", card.card_id)?;
    writeln!(out, "    cost          : {}", card.local_cost())?;
    writeln!(
        out,
        "    type / target : {:?} / {:?}",
        card.card_type, card.target_type
    )?;
    writeln!(out, "    base damage   : {}", card.base_damage)?;
    writeln!(out, "    base block    : {}", card.base_block)?;
    writeln!(out, "    upgraded      : {}", card.upgraded)?;
    Ok(())
}

pub fn print_legal_actions<W: Write>(out: &mut W, state: &CombatState) -> io::Result<()> {
    // Delegate to the engine's `legal_actions` so the `actions` verb
    // shows the exact same dispatchable set as the auto-prompt menu
    // (including potions). Format mirrors `print_combat_prompt` so the
    // `[N]` index is a typeable keystroke in either rendering.
    let actions = sts2_core::legal_actions(state);
    writeln!(out, "  actions (type the bracketed index to dispatch):")?;
    for (i, action) in actions.iter().enumerate() {
        writeln!(out, "    [{i}] {}", format_action_for_menu(state, action))?;
    }
    Ok(())
}

pub fn print_help<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  commands:")?;
    writeln!(
        out,
        "    <n>             dispatch action-menu entry <n> (bare-int)"
    )?;
    writeln!(
        out,
        "    play <i> [t]    play card at hand index <i>, optional target enemy slot <t>"
    )?;
    writeln!(
        out,
        "    potion <s> [t]  drink potion in slot <s>, optional target enemy slot <t>"
    )?;
    writeln!(out, "    end | e         end the player turn")?;
    writeln!(out, "    hand            print hand")?;
    writeln!(out, "    state           print full combat state")?;
    writeln!(out, "    actions         enumerate legal actions")?;
    writeln!(
        out,
        "    info <i>        print card details for hand index <i>"
    )?;
    writeln!(out, "    card <i>        alias for `info <i>`")?;
    writeln!(
        out,
        "    assert hp <op> <n> | assert phase <name> | assert monsters_alive <n>"
    )?;
    writeln!(
        out,
        "    # comment       comment line (echoed in scripted mode)"
    )?;
    writeln!(out, "    hint [sims] [--fair[=N] | --fair-all[=N]]")?;
    writeln!(
        out,
        "                    MCTS move suggestion ranked by visits (default 1000 sims)."
    )?;
    writeln!(
        out,
        "                    Add --fair to also run an independent MCTS for the top-3"
    )?;
    writeln!(
        out,
        "                    actions at N sims each (default 3000) — own-search win%,"
    )?;
    writeln!(
        out,
        "                    free of the main-tree UCB selection bias. Use --fair-all"
    )?;
    writeln!(
        out,
        "                    to evaluate every legal action instead of just the top-3."
    )?;
    writeln!(
        out,
        "    eval <i> [sims] independent MCTS from the state AFTER action-menu entry <i>;"
    )?;
    writeln!(
        out,
        "                    reports win% (default 3000 sims). Same idea as `hint --fair`"
    )?;
    writeln!(
        out,
        "                    but for one action you pick by menu index."
    )?;
    writeln!(out, "    help | ?        show this help")?;
    writeln!(out, "    quit            exit the REPL")?;
    Ok(())
}

/// Render the lightspeed-parity auto-prompt block: full combat state
/// followed by a numbered action menu the caller can dispatch into
/// via bare-int (`Command::ActionIndex`). Called once per agent
/// decision; in-REPL `state`/`hand`/`actions` re-renders use the
/// per-pile helpers above and do NOT re-cache `last_actions`.
pub fn print_combat_prompt<W: Write>(
    out: &mut W,
    state: &CombatState,
    actions: &[Action],
) -> io::Result<()> {
    print_combat_state(out, state)?;
    writeln!(out, "  actions (type the bracketed index to dispatch):")?;
    for (i, action) in actions.iter().enumerate() {
        writeln!(out, "    [{i}] {}", format_action_for_menu(state, action))?;
    }
    Ok(())
}

pub(crate) fn format_action_for_menu(state: &CombatState, action: &Action) -> String {
    match action {
        Action::PlayCardAtAlly { hand_index, ally } => {
            let card_name = state.player.hand.get(usize::from(*hand_index)).map_or_else(
                || "<oob>".to_string(),
                |uid| format!("{:?}", card_by_uid(state, *uid).card_id),
            );
            format!("play {hand_index} ally {ally}  ({card_name} → ally {ally})")
        }
        Action::PlayCard { hand_index, target } => {
            let card_name = state.player.hand.get(usize::from(*hand_index)).map_or_else(
                || "<oob>".to_string(),
                |uid| {
                    let card = card_by_uid(state, *uid);
                    format!("{:?}", card.card_id)
                },
            );
            target.as_ref().map_or_else(
                || format!("play {hand_index}    ({card_name})"),
                |slot| format!("play {hand_index} {slot}  ({card_name} → enemy {slot})"),
            )
        }
        Action::DrinkPotion {
            potion_slot,
            target,
        } => {
            let potion_name = state
                .player
                .potions
                .get(usize::from(*potion_slot))
                .map_or_else(|| "<oob>".to_string(), |kind| format!("{kind:?}"));
            target.as_ref().map_or_else(
                || format!("potion {potion_slot}  ({potion_name})"),
                |slot| format!("potion {potion_slot} {slot}  ({potion_name} → enemy {slot})"),
            )
        }
        Action::EndTurn => "end           (end player turn)".to_string(),
        other => format!("{other:?}"),
    }
}

fn card_by_uid(state: &CombatState, uid: sts2_core::CardUid) -> &Card {
    state
        .cards
        .iter()
        .find(|c| c.uid == uid)
        .expect("hand uids point to live cards")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scenario::{MonsterSpec, Scenario};
    use crate::sim::PlayerClass;
    use sts2_core::MonsterKind;

    /// A Nibbit combat advanced into `PlayerPlayPhase` (hand dealt).
    fn started() -> CombatState {
        let scenario = Scenario::from_monsters(vec![MonsterSpec::new(MonsterKind::Nibbit)]);
        let mut state = crate::sim::scenario::starter::build_state(
            &scenario,
            1,
            PlayerClass::Ironclad,
            None,
            None,
        );
        let _ = sts2_core::combat_loop::start_combat(&mut state);
        // start_combat leaves us in PlayerTurnStart; start_player_turn
        // deals the opening hand + refills energy (PlayerPlayPhase).
        let _ = sts2_core::combat_loop::start_player_turn(&mut state);
        state
    }

    fn render<F>(f: F) -> String
    where
        F: FnOnce(&mut Vec<u8>) -> io::Result<()>,
    {
        let mut buf = Vec::new();
        f(&mut buf).expect("write");
        String::from_utf8(buf).expect("utf8")
    }

    #[test]
    fn print_combat_state_has_all_sections() {
        let s = started();
        let out = render(|w| print_combat_state(w, &s));
        assert!(out.contains("combat state"));
        assert!(out.contains("phase"));
        assert!(out.contains("player  : HP"));
        assert!(out.contains("monsters:"));
        assert!(out.contains("hand    :"));
        // The single enemy is rendered at slot 0.
        assert!(out.contains("slot 0:"));
    }

    #[test]
    fn print_player_and_hand_and_monster() {
        let s = started();
        let player = render(|w| print_player(w, &s.player));
        assert!(player.contains("HP") && player.contains("energy"));

        let hand = render(|w| print_hand(w, &s));
        assert!(hand.contains("card(s)"));

        let c = &s.creatures[0];
        let mon = render(|w| print_monster(w, 0, c));
        assert!(mon.contains("slot 0:") && mon.contains("intent") && mon.contains("alive:"));
    }

    #[test]
    fn print_card_info_valid_and_out_of_bounds() {
        let s = started();
        // Hand is non-empty after start_combat; index 0 is valid.
        let ok = render(|w| print_card_info(w, &s, 0));
        assert!(ok.contains("cost") && ok.contains("base damage"));

        let oob = render(|w| print_card_info(w, &s, 200));
        assert!(oob.contains("no card at hand index 200"));
    }

    #[test]
    fn print_legal_actions_and_prompt_and_help() {
        let s = started();
        let actions = render(|w| print_legal_actions(w, &s));
        assert!(actions.contains("actions") && actions.contains("[0]"));

        let legal = sts2_core::legal_actions(&s).into_vec();
        let prompt = render(|w| print_combat_prompt(w, &s, &legal));
        assert!(prompt.contains("combat state") && prompt.contains("[0]"));

        let help = render(print_help);
        assert!(help.contains("commands:") && help.contains("quit"));
    }

    #[test]
    fn format_action_for_menu_covers_branches() {
        let s = started();
        // PlayCard with a valid hand index renders the card name.
        let play = format_action_for_menu(
            &s,
            &Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        );
        assert!(play.starts_with("play 0"));
        // Targeted form.
        let play_t = format_action_for_menu(
            &s,
            &Action::PlayCard {
                hand_index: 0,
                target: Some(0),
            },
        );
        assert!(play_t.contains("→ enemy 0"));
        // Out-of-bounds hand index -> <oob>.
        let oob = format_action_for_menu(
            &s,
            &Action::PlayCard {
                hand_index: 250,
                target: None,
            },
        );
        assert!(oob.contains("<oob>"));
        // No potions held -> DrinkPotion renders <oob>.
        let pot = format_action_for_menu(
            &s,
            &Action::DrinkPotion {
                potion_slot: 0,
                target: None,
            },
        );
        assert!(pot.contains("<oob>"));
        // EndTurn + catch-all arm.
        assert!(format_action_for_menu(&s, &Action::EndTurn).contains("end player turn"));
        assert!(format_action_for_menu(&s, &Action::Evoke { times: 1 }).contains("Evoke"));
    }
}
