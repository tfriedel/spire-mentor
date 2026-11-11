//! The number on a card's face must be the number the card deals.
//!
//! ~40 cards state a value they only know at play time (Ashen Strike is
//! `CalculationBase(6) + ExtraDamage(3) x <cards in your Exhaust Pile>`), and
//! `sts2_content::cards::preview::calculated_value` is what the workbench puts
//! on the face. A face that disagrees with the hit is worse than a face with no
//! number on it, so each card's formula lives once — in a `preview_*` its
//! `play_*` handler calls — and this test pins that they cannot come apart.
//!
//! It is a *behavioural* check, not a "does the code call the shared function"
//! check: it previews, then dispatches, then asserts the enemy actually lost
//! exactly the previewed HP. Re-inlining a formula into a handler would pass a
//! structural check and fail this one.

use std::num::NonZeroU32;

use smallvec::SmallVec;
use sts2_core::card_effects::{dispatch, EffectResult};
use sts2_core::cards::{
    Card, CardEnchantments, CardId, CardTags, CardType, CombatVars, TargetType,
};
use sts2_core::{
    CardUid, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid,
};

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::preview::calculated_value;

/// Single-target attacks whose damage is a `{CalculatedDamage}` the player
/// deals directly, so "what the face said" and "what the enemy lost" are the
/// same number. The Osty attacks (Protector / Squeeze / Unleash) and the
/// multi-hit / non-damage cards are deliberately out of scope here — their
/// observable is not a single HP delta.
const DIRECT_DAMAGE: &[(CardId, i32)] = &[
    (CardId::AshenStrike, 6),
    (CardId::BodySlam, 0),
    (CardId::Bully, 4),
    (CardId::CrescentSpear, 8),
    (CardId::DeathMarch, 8),
    (CardId::GangUp, 5),
    (CardId::GoldAxe, 0),
    (CardId::MementoMori, 9),
    (CardId::MindBlast, 0),
    (CardId::Murder, 1),
    (CardId::PerfectedStrike, 6),
    (CardId::PreciseCut, 13),
    (CardId::Rend, 15),
    (CardId::SoulStorm, 9),
    (CardId::Supermassive, 5),
    (CardId::TimesUp, 0),
];

fn push_card(state: &mut CombatState, card_id: CardId, base_damage: i32) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards len fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    state.cards.push(Card {
        uid,
        card_id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage,
        base_block: 0,
        combat_vars: CombatVars::new(),
        enchantments: CardEnchantments::new(),
        tags: CardTags::NONE,
        energy_spent: 0,
        base_replay_count: 0,
        keyword_override: None,
        affliction: None,
    });
    uid
}

/// A board with something in every pile and a debuffed enemy, so the
/// multipliers these cards read are non-zero. A preview that dropped its
/// multiplier would still match the hit on an empty board — this is what makes
/// the assertion bite.
fn busy_board() -> (CombatState, CreatureUid) {
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("non-zero");
    state.creatures.push(Creature {
        id,
        current_hp: 9999,
        max_hp: 9999,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    let uid = CreatureUid(id);
    state.enemy_uids.push(uid);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 99;
    state.player.max_energy = 99;
    state.player.block = 7;

    // Fill the piles the multipliers count. Souls in Exhaust feed Soul Storm;
    // the Strike tag feeds Perfected Strike; Stardust carries a Star cost for
    // Crescent Spear.
    for (pile, ids) in [
        (0usize, vec![CardId::StrikeIronclad, CardId::DefendIronclad]),
        (1, vec![CardId::Soul, CardId::Soul, CardId::StrikeIronclad]),
        (2, vec![CardId::DefendIronclad, CardId::Stardust]),
        (3, vec![CardId::StrikeIronclad]),
    ] {
        for cid in ids {
            let uid = push_card(&mut state, cid, 6);
            let idx = uid.0.get() as usize - 1;
            if cid == CardId::StrikeIronclad {
                state.cards[idx].tags = CardTags::NONE.with(sts2_core::cards::CardTag::Strike);
            }
            match pile {
                0 => state.player.hand.push(uid),
                1 => state.player.exhaust.push(uid),
                2 => state.player.discard.push(uid),
                _ => state.player.draw.push(uid),
            }
        }
    }

    // Debuffs on the enemy: Vulnerable feeds Bully, and Rend counts uniques.
    for (n, kind, amount) in [(1u32, PowerKind::Vulnerable, 3), (2, PowerKind::Weak, 2)] {
        let pid = PowerUid::new(n).expect("non-zero power uid");
        state.powers.push(Power {
            id: pid.0,
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.creatures[0].powers.push(pid);
    }

    // Counters the turn/combat-scoped multipliers read.
    state.cards_drawn = 4;
    state.combat_log.cards_discarded_this_turn = 2;
    state.combat_log.cards_drawn_by_effect_this_turn = 3;
    state.combat_log.player_card_plays_this_combat = 5;

    (state, uid)
}

#[test]
fn face_value_equals_damage_dealt() {
    content_dispatch::install_default_registry();

    let mut scaled = 0;
    for &(card_id, base_damage) in DIRECT_DAMAGE {
        let (mut state, enemy) = busy_board();
        let uid = push_card(&mut state, card_id, base_damage);
        // The handler reads the card being played off the play pile.
        state.player.play.push(uid);
        let idx = uid.0.get() as usize - 1;

        let face = calculated_value(&state, &state.cards[idx], Some(enemy))
            .unwrap_or_else(|| panic!("{card_id:?} has no calculated value"));
        assert!(
            face >= 0,
            "{card_id:?}: the face must never show negative damage"
        );
        if face > base_damage {
            scaled += 1;
        }

        let before = state.creatures[0].current_hp;
        let res = dispatch(&mut state, card_id, Some(enemy));
        assert_eq!(res, EffectResult::Resolved, "{card_id:?} did not resolve");
        let dealt = before - state.creatures[0].current_hp;

        assert_eq!(
            dealt, face,
            "{card_id:?}: face said {face}, enemy lost {dealt}"
        );
    }

    // Guard the guard: if the board stopped making multipliers fire, every
    // assertion above would still pass while testing nothing.
    assert!(
        scaled >= 8,
        "only {scaled} cards scaled above their base — the test board went stale"
    );
}

#[test]
fn untargeted_face_matches_the_games_pre_hover_number() {
    // C# `CalculatedVar.Calculate` takes the target as an argument and the game
    // re-renders on hover; with no target the multiplier contributes nothing.
    // Display callers pass `None`, so a target-reading card must fall back to
    // its plain base rather than silently reading some other creature.
    content_dispatch::install_default_registry();
    for (card_id, base, expect_untargeted) in [
        (CardId::Bully, 4, 4),
        (CardId::Rend, 0, 15),
        (CardId::TimesUp, 0, 0),
    ] {
        let (mut state, enemy) = busy_board();
        let uid = push_card(&mut state, card_id, base);
        state.player.play.push(uid);
        let idx = uid.0.get() as usize - 1;

        let untargeted = calculated_value(&state, &state.cards[idx], None);
        assert_eq!(
            untargeted,
            Some(expect_untargeted),
            "{card_id:?} untargeted"
        );

        let targeted =
            calculated_value(&state, &state.cards[idx], Some(enemy)).expect("targeted value");
        assert!(
            targeted >= expect_untargeted,
            "{card_id:?}: targeting must only add ({targeted} < {expect_untargeted})"
        );
    }
}

#[test]
fn cards_without_a_runtime_value_report_none() {
    // The wire omits the field entirely for these, and the workbench must read
    // "absent" as "this card has no such number" rather than as zero.
    content_dispatch::install_default_registry();
    let (mut state, _) = busy_board();
    for card_id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
        let uid = push_card(&mut state, card_id, 6);
        let idx = uid.0.get() as usize - 1;
        assert_eq!(
            calculated_value(&state, &state.cards[idx], None),
            None,
            "{card_id:?} should not claim a calculated value"
        );
    }
}

#[test]
fn untargeted_face_ignores_the_players_own_debuffs() {
    // C# previews an unhovered card with `target = null`, so the target half of
    // `Hook.ModifyDamage` contributes nothing. Naming the DEALER as the target
    // instead (the obvious shortcut when the ctx field is not optional) reads
    // the player's own debuffs as the victim's: Vulnerable would inflate every
    // scaling attack's face by 50%, and Intangible's cap would pin a 24-damage
    // Ashen Strike to "Deal 1 damage."
    content_dispatch::install_default_registry();
    let (mut state, _) = busy_board();
    let uid = push_card(&mut state, CardId::AshenStrike, 6);
    state.player.play.push(uid);
    let idx = uid.0.get() as usize - 1;

    let clean = calculated_value(&state, &state.cards[idx], None).expect("value");
    assert!(clean > 6, "the board stopped scaling Ashen Strike");

    for (n, kind, amount) in [
        (90u32, PowerKind::Vulnerable, 3),
        (91, PowerKind::Intangible, 2),
    ] {
        let pid = PowerUid::new(n).expect("non-zero power uid");
        state.powers.push(Power {
            id: pid.0,
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(pid);
    }

    assert_eq!(
        calculated_value(&state, &state.cards[idx], None),
        Some(clean),
        "the player's own debuffs must not reach the target side of the face"
    );
}

#[test]
fn precise_cut_face_does_not_count_itself() {
    // `PreciseCut.cs` decrements the hand count when the previewed card is
    // itself in Hand — which is the only pile a card face is ever drawn in.
    // The play path sits on the other side of that guard (the card has already
    // moved to Play), so both callers must land on the SAME number.
    content_dispatch::install_default_registry();
    let (mut state, _) = busy_board();
    let uid = push_card(&mut state, CardId::PreciseCut, 13);
    let idx = uid.0.get() as usize - 1;

    // Untargeted, as the game draws an unhovered card: the enemy's Vulnerable
    // would otherwise scale the face back over its own base and hide the shrink.
    state.player.hand.push(uid);
    let face = calculated_value(&state, &state.cards[idx], None).expect("value");
    assert!(face < 13, "the board stopped shrinking Precise Cut");

    // Moving the card out of Hand changes the pile the multiplier counts AND
    // flips C#'s guard, and the two cancel exactly. If the preview counted
    // itself, the in-hand face would come out one ExtraDamage step low — which
    // is the only state a player ever reads it in.
    state.player.hand.retain(|c| *c != uid);
    state.player.play.push(uid);
    assert_eq!(
        calculated_value(&state, &state.cards[idx], None),
        Some(face),
        "the number must not change as the card leaves your hand to resolve"
    );
}

#[test]
fn normality_face_counts_down_with_the_plays_started_this_turn() {
    let (mut state, _) = busy_board();
    let uid = push_card(&mut state, CardId::Normality, 0);
    let idx = uid.0.get() as usize - 1;
    let face = |s: &CombatState| calculated_value(s, &s.cards[idx], None);
    assert_eq!(face(&state), Some(3));
    state.player.card_plays_started_this_turn = 2;
    assert_eq!(face(&state), Some(1));
    state.player.card_plays_started_this_turn = 5;
    assert_eq!(face(&state), Some(0));
}
