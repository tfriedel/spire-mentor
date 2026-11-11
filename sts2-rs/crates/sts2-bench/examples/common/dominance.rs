//! The engine-side dominance proof, shared by every consumer.
//!
//! Two distinct card plays that are both affordable together and both
//! still playable after each other are two **orderings of the same pair**.
//! Apply both, zero the dominance axes, and compare the whole
//! `CombatState` — derived `Eq`, so both RNG streams, every pile, power
//! and intent are in the comparison. If the remainder is equal and one
//! ordering is weakly better on every axis and strictly better on one,
//! that ordering is proven at least as good. No search is consulted, so
//! nothing here inherits a search's bias.
//!
//! This lives in `common/` rather than in either caller because there are
//! now two: `dominance_pairs` (which enumerates a suite from it) and
//! `combat_position_book` (which uses it to mark the positions where its
//! own 50 000-sim oracle is provably wrong). A second copy of a theorem
//! is a copy that drifts.
//!
//! Deliberately **not** an axis: enemy debuff stacks. "More Vulnerable is
//! better" is a judgement, not an engine theorem.

use sts2_core::{apply, legal_actions, Action, CombatPhase, CombatState, StepResult};

/// Player HP / block read through the unified `Creature` mirror when the
/// combat has one (bead `sts2-rs-d03`), falling back to the legacy
/// `PlayerCombatState` scalars. Reading only the legacy field would make
/// the axis silently stale in any combat that routes through the mirror.
pub fn player_scalars(s: &CombatState) -> (i32, i32) {
    s.player
        .creature_uid
        .and_then(|u| s.creatures.iter().find(|c| c.id == u.0))
        .map_or((s.player.current_hp, s.player.block), |c| {
            (c.current_hp, c.block)
        })
}

pub fn enemy_hp_total(s: &CombatState) -> i32 {
    s.creatures
        .iter()
        .filter(|c| s.enemy_uids.iter().any(|u| u.0 == c.id))
        .map(|c| c.current_hp.max(0))
        .sum()
}

/// Zero every dominance-axis field so `==` compares *everything else*:
/// both RNG streams, every pile, every power, every intent and intent
/// history, energy, round, phase. `CombatState: Eq` is derived, so this
/// is a whole-state comparison and not a hand-maintained field list that
/// can rot.
pub fn neutralize_axes(s: &mut CombatState) {
    let player_id = s.player.creature_uid.map(|u| u.0);
    s.player.current_hp = 0;
    s.player.block = 0;
    for c in &mut s.creatures {
        c.current_hp = 0;
        if Some(c.id) == player_id {
            c.block = 0;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Axes {
    pub enemy_hp: i32,
    pub player_hp: i32,
    pub player_block: i32,
}

pub fn axes_of(s: &CombatState) -> Axes {
    let (hp, block) = player_scalars(s);
    Axes {
        enemy_hp: enemy_hp_total(s),
        player_hp: hp,
        player_block: block,
    }
}

/// `Some(gains)` when `x` weakly dominates `y` and strictly beats it on at
/// least one axis; `None` when neither dominates (including the equal
/// case, which carries no decision to be right about).
pub fn dominance(x: &Axes, y: &Axes) -> Option<(Vec<String>, i32, i32, i32)> {
    let g_enemy = y.enemy_hp - x.enemy_hp;
    let g_hp = x.player_hp - y.player_hp;
    let g_block = x.player_block - y.player_block;
    if g_enemy < 0 || g_hp < 0 || g_block < 0 {
        return None;
    }
    let mut axes = Vec::new();
    if g_enemy > 0 {
        axes.push("enemy_hp".to_string());
    }
    if g_hp > 0 {
        axes.push("player_hp".to_string());
    }
    if g_block > 0 {
        axes.push("player_block".to_string());
    }
    (!axes.is_empty()).then_some((axes, g_enemy, g_hp, g_block))
}

/// Card-identity key for re-finding "the same card" in a hand whose
/// indices have shifted because another card was played. Mirrors the
/// tuple `sts2_mcts::EdgeAction::PlayCard` keys on, minus the pieces that
/// cannot change between two plies of the same turn.
pub type CardKey = (sts2_core::cards::CardId, bool, i8, Option<u8>);

pub fn card_key(s: &CombatState, a: &Action) -> Option<CardKey> {
    let Action::PlayCard { hand_index, target } = a else {
        return None;
    };
    let uid = s.player.hand.get(*hand_index as usize)?;
    let card = s.cards.get(uid.0.get() as usize - 1)?;
    let cost = i8::try_from(card.local_cost()).unwrap_or(i8::MIN);
    Some((card.card_id, card.upgraded, cost, *target))
}

/// Resolve a `CardKey` back to a concrete legal `PlayCard` in `s`.
pub fn resolve(s: &CombatState, key: &CardKey) -> Option<Action> {
    let legal = legal_actions(s);
    legal
        .into_iter()
        .find(|a| card_key(s, a).as_ref() == Some(key))
}

/// Apply `first` then `second` (re-resolved against the intermediate
/// state). `None` if either play is rejected, the combat ends, or the
/// player does not come back to a play phase — a pair is only meaningful
/// when both orderings leave the agent standing in the same situation.
pub fn apply_order(base: &CombatState, first: &Action, second: &CardKey) -> Option<CombatState> {
    let mut s = base.clone();
    if !matches!(apply(&mut s, first.clone()), StepResult::Advanced) {
        return None;
    }
    if s.is_over || s.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    let next = resolve(&s, second)?;
    if !matches!(apply(&mut s, next), StepResult::Advanced) {
        return None;
    }
    if s.is_over || s.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    Some(s)
}

/// Every proven ordering pair at one decision point.
pub type FoundPair = (Action, Action, String, String, Vec<String>, i32, i32, i32);

/// Debug label for the card a `PlayCard` action refers to.
pub fn card_label(state: &CombatState, a: &Action) -> String {
    card_key(state, a).map_or_else(
        || format!("{a:?}"),
        |(id, up, cost, _)| {
            let plus = if up { "+" } else { "" };
            format!("{id:?}{plus}({cost})")
        },
    )
}

pub fn pairs_at(state: &CombatState) -> (usize, Vec<FoundPair>) {
    let legal = legal_actions(state);
    let plays: Vec<Action> = legal
        .into_iter()
        .filter(|a| matches!(a, Action::PlayCard { .. }))
        .collect();
    let mut tried = 0_usize;
    let mut out = Vec::new();
    for i in 0..plays.len() {
        for j in (i + 1)..plays.len() {
            let (Some(ki), Some(kj)) = (card_key(state, &plays[i]), card_key(state, &plays[j]))
            else {
                continue;
            };
            // Two hand slots holding the same card identity are the same
            // move; their orderings are trivially equal and would only
            // pad the suite with pairs nothing can get wrong.
            if ki == kj {
                continue;
            }
            tried += 1;
            let (Some(s_ij), Some(s_ji)) = (
                apply_order(state, &plays[i], &kj),
                apply_order(state, &plays[j], &ki),
            ) else {
                continue;
            };
            // Structural equality modulo the axes, then the axis compare.
            let (mut n_ij, mut n_ji) = (s_ij.clone(), s_ji.clone());
            neutralize_axes(&mut n_ij);
            neutralize_axes(&mut n_ji);
            if n_ij != n_ji {
                continue;
            }
            let (a_ij, a_ji) = (axes_of(&s_ij), axes_of(&s_ji));
            let (li, lj) = (card_label(state, &plays[i]), card_label(state, &plays[j]));
            if let Some((axes, e, h, b)) = dominance(&a_ij, &a_ji) {
                out.push((plays[i].clone(), plays[j].clone(), li, lj, axes, e, h, b));
            } else if let Some((axes, e, h, b)) = dominance(&a_ji, &a_ij) {
                out.push((plays[j].clone(), plays[i].clone(), lj, li, axes, e, h, b));
            }
        }
    }
    (tried, out)
}
