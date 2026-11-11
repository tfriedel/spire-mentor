//! Conservative lethal-this-turn detector (bead `sts2-rs-5yzrh` +
//! follow-up GH issue #51).
//!
//! Returns `true` only when a small set of provable conditions hold and
//! a per-monster damage knapsack confirms the player can deal at least
//! `monster.current_hp + monster.block` to **every** living monster
//! within the current energy budget. The implementation is intentionally
//! conservative — false positives would corrupt MCTS proven-win
//! propagation, so when in doubt the detector bails (returns `false`).
//!
//! ## Scope at this revision
//!
//! Gates (each one a bail point):
//!
//! 1. **Phase**: must be `CombatPhase::PlayerPlayPhase` and the combat
//!    must not already be over.
//! 2. **Living monsters**: at least one. Multi-monster cases go through
//!    the AOE-only branch (see gate 5).
//! 3. **Player powers**: only `Strength`, `Weak`, and `Frail` are
//!    tolerated (and Frail is irrelevant to damage dealt — it folds out
//!    silently). Any other power kind bails.
//! 4. **Monster powers**: only `Vulnerable` is tolerated. Any other
//!    monster power kind bails. (Strength on a monster is irrelevant
//!    to damage *taken*, so it folds out; conservatively we bail
//!    because monster-side Strength is rare in early combats and the
//!    audit overhead isn't worth the false-positive risk.)
//! 5. **Cards**: only an explicit allow-list of single-target /
//!    AOE attack cards is recognised — see [`card_damage`]. All
//!    cards in the player's hand must be in the allow-list OR be a
//!    no-cost skill that contributes nothing (we conservatively bail
//!    on any unrecognised card so the energy budget isn't accidentally
//!    spent on a damage-less card the detector can't reason about).
//! 6. **Relics**: only a small allow-list of damage-neutral relics is
//!    tolerated — see [`is_damage_neutral_relic`]. Any other relic
//!    bails.
//!
//! ## Cross-reference
//!
//! Damage pipeline order canonical: `crates/sts2-core/src/hooks.rs`
//! (`modify_damage_*`). Weak (×0.75 floor) → Strength (+N additive) →
//! Vulnerable (×1.5).

use sts2_core::cards::{CardId, TargetType};
use sts2_core::powers::PowerKind;
use sts2_core::{CombatPhase, CombatState, Power, RelicKind};

/// Recognised attack-card shape — base damage + AOE flag.
///
/// The canonical card cost is intentionally NOT stored here: cards
/// in the live `state.cards` arena can carry per-instance cost
/// mutations (Mayhem, SeeingRed-style), so the knapsack reads
/// `card.cost` directly from the arena rather than the spec value.
#[derive(Clone, Copy, Debug)]
struct CardShape {
    base_damage: i32,
    is_aoe: bool,
}

/// Hand-card view used by the knapsack — captures the only fields the
/// damage projection needs.
#[derive(Clone, Copy, Debug)]
struct PlayableCard {
    cost: i8,
    damage_per_target: i32,
    is_aoe: bool,
}

/// Return the recognised shape of a card, or `None` if the detector
/// does not know how to bound its damage. Conservative: anything
/// outside this allow-list bails the whole detection.
///
/// Base damages mirror the Ironclad canonical pin
/// (`sts2-rl-agent@01c094d:sts2_env/cards/ironclad.py`).
fn card_shape(card_id: CardId) -> Option<CardShape> {
    match card_id {
        CardId::StrikeIronclad => Some(CardShape {
            base_damage: 6,
            is_aoe: false,
        }),
        // Cleave doesn't exist in the canonical `01c094d` pin's STS2
        // card pool; Breakthrough is the equivalent 1-cost AOE on
        // Ironclad (9 base damage per target).
        CardId::Breakthrough => Some(CardShape {
            base_damage: 9,
            is_aoe: true,
        }),
        CardId::Thunderclap => Some(CardShape {
            base_damage: 4,
            is_aoe: true,
        }),
        CardId::IronWave => Some(CardShape {
            base_damage: 5,
            is_aoe: false,
        }),
        _ => None,
    }
}

/// Relics audited as damage-pipeline-neutral. Each entry was confirmed
/// against the relic-handler tables in
/// `crates/sts2-core/src/relic_effects.rs` (search the variant name)
/// to ensure it does not fold into `modify_damage_*`.
///
/// Damage-affecting relics (Akabeko first-attack rider, Vajra
/// combat-start Strength, StrikeDummy Strike+3, PenNib 10th-attack 2×,
/// Ginger Weak-related, BronzeScales thorns, FrozenEye expert-play
/// signal) are intentionally NOT in this list — see the issue body for
/// the audit narrative.
fn is_damage_neutral_relic(kind: RelicKind) -> bool {
    matches!(
        kind,
        RelicKind::RingOfTheSnake     // combat-start draw, no damage
            | RelicKind::JuzuBracelet // event filter, no combat interaction
            | RelicKind::OddlySmoothStone // combat-start Dex+1 — block only
            | RelicKind::Pendulum     // shop price, no combat interaction
            | RelicKind::Lantern      // combat-start +1 energy — irrelevant once we are mid-turn
            | RelicKind::HappyFlower  // energy every 3 turns — no damage interaction
            | RelicKind::Anchor       // combat-start +10 block — block only
            | RelicKind::HornCleat // turn-3 +14 block — block only
    )
}

/// Look up a `Power` by uid in the state's power arena. Returns `None`
/// for dangling uids (which shouldn't occur in well-formed states; the
/// detector treats this as a bail).
fn find_power(state: &CombatState, puid: sts2_core::PowerUid) -> Option<Power> {
    state.powers.get_by_id(puid.0).copied()
}

/// Fold the player's powers into `(strength, weak_active)`. Returns
/// `None` (i.e. bail) if any unsupported power kind is present.
fn fold_player_powers(state: &CombatState) -> Option<(i32, bool)> {
    let mut strength = 0i32;
    let mut weak_active = false;
    for puid in &state.player.powers {
        let power = find_power(state, *puid)?;
        match power.kind {
            PowerKind::Strength => {
                strength = power.amount;
            }
            PowerKind::Weak => {
                if power.amount > 0 {
                    weak_active = true;
                }
            }
            // Frail is on the player but affects block-gained, not
            // damage-dealt. Safe to ignore.
            PowerKind::Frail => {}
            _ => return None,
        }
    }
    Some((strength, weak_active))
}

/// Fold a monster's powers into `vulnerable_active`. Returns `None`
/// (i.e. bail) if any unsupported power kind is present on the monster.
fn fold_monster_powers(state: &CombatState, powers: &[sts2_core::PowerUid]) -> Option<bool> {
    let mut vulnerable_active = false;
    for puid in powers {
        let power = find_power(state, *puid)?;
        match power.kind {
            PowerKind::Vulnerable => {
                if power.amount > 0 {
                    vulnerable_active = true;
                }
            }
            _ => return None,
        }
    }
    Some(vulnerable_active)
}

/// Compute the damage one play of `card` deals to a target with
/// `vulnerable_active`, given the player's `(strength, weak_active)`
/// modifier folding. Pipeline order matches the engine:
/// `Weak (×0.75 floor) → Strength (+N) → Vulnerable (×1.5)`.
fn projected_damage(
    base_damage: i32,
    strength: i32,
    weak_active: bool,
    vulnerable_active: bool,
) -> i32 {
    let mut dmg = base_damage + strength;
    if dmg <= 0 {
        return 0;
    }
    if weak_active {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        {
            dmg = (dmg as f32 * 0.75).floor() as i32;
        }
    }
    if vulnerable_active {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        {
            dmg = (dmg as f32 * 1.5).floor() as i32;
        }
    }
    dmg.max(0)
}

/// Conservative lethal-this-turn detector. Returns `true` only when a
/// chain of plays from the current hand is guaranteed to leave every
/// living monster at `current_hp <= 0`, within `state.player.energy`.
///
/// Soundness: when the detector returns `true`, applying the implied
/// card chain on this turn produces a winning state (modulo unrelated
/// terminal conditions like player death from monster-side
/// `at_end_of_turn` damage, which would only fire after `EndTurn` —
/// the detector is about *this turn's* damage knapsack).
#[must_use]
pub fn detect_lethal(state: &CombatState) -> bool {
    // Gate 1: phase / combat-over.
    if state.is_over {
        return false;
    }
    if state.phase != CombatPhase::PlayerPlayPhase {
        return false;
    }
    if state.pending_choice.is_some() || state.pending_play.is_some() {
        return false;
    }

    // Gate 6: relic allow-list. Any unaudited relic bails.
    for relic in &state.relics {
        if relic.is_melted {
            continue;
        }
        if !is_damage_neutral_relic(relic.kind) {
            return false;
        }
    }

    // Gate 3: player-side power folding.
    let Some((strength, weak_active)) = fold_player_powers(state) else {
        return false;
    };

    // Gate 2: collect living monsters + per-monster modifiers.
    struct MonsterView {
        hp_plus_block: i32,
        vulnerable: bool,
    }
    let mut monsters: Vec<MonsterView> = Vec::new();
    for uid in &state.enemy_uids {
        let Some(c) = state.creatures.iter().find(|c| c.id == uid.0) else {
            continue;
        };
        if c.current_hp <= 0 {
            continue;
        }
        let Some(vulnerable) = fold_monster_powers(state, &c.powers) else {
            return false;
        };
        monsters.push(MonsterView {
            hp_plus_block: c.current_hp + c.block,
            vulnerable,
        });
    }
    if monsters.is_empty() {
        return false;
    }

    // Gate 5: collect playable cards from the hand. Every card MUST
    // resolve to a known shape — an unknown card bails the detection
    // (we cannot reason about whether playing it would consume energy
    // that the knapsack assumed was available for damage).
    let mut playable: Vec<PlayableCard> = Vec::with_capacity(state.player.hand.len());
    for card_uid in &state.player.hand {
        let idx = card_uid.0.get() as usize - 1;
        let Some(card) = state.cards.get(idx) else {
            return false;
        };
        let Some(shape) = card_shape(card.card_id) else {
            return false;
        };
        // Sanity: target_type must match the AOE flag.
        if shape.is_aoe && card.target_type != TargetType::AllEnemies {
            return false;
        }
        if !shape.is_aoe && card.target_type != TargetType::Enemy {
            return false;
        }
        // Cost may be modified per-instance (Mayhem etc.); use the
        // live local cost.
        playable.push(PlayableCard {
            cost: i8::try_from(card.local_cost()).unwrap_or(i8::MAX),
            damage_per_target: shape.base_damage,
            is_aoe: shape.is_aoe,
        });
    }

    let energy = state.player.energy;
    if energy <= 0 {
        return false;
    }

    if monsters.len() == 1 {
        // Single-monster: any mix of single-target and AOE cards works
        // (AOE collapses to single-target when there is one monster).
        // Build the knapsack as `(cost, damage)` and check whether the
        // max-damage subset within energy reaches `hp + block` after
        // folding Weak / Strength / Vulnerable.
        let m = &monsters[0];
        let items: Vec<(i32, i32)> = playable
            .iter()
            .map(|c| {
                (
                    i32::from(c.cost),
                    projected_damage(c.damage_per_target, strength, weak_active, m.vulnerable),
                )
            })
            .collect();
        let best = knapsack_max_damage(&items, energy);
        return best >= m.hp_plus_block;
    }

    // Multi-monster: AOE-only branch. Restrict to AOE cards; each AOE
    // play deals its (Weak/Strength-folded) damage to every monster,
    // then Vulnerable scales per-target. The knapsack maximises raw
    // damage; the per-monster check then folds Vulnerable independently
    // since the same set of AOE plays applies to every monster.
    let aoe_items: Vec<(i32, i32)> = playable
        .iter()
        .filter(|c| c.is_aoe)
        .map(|c| {
            // Base (no Vulnerable) per-target damage from one play.
            let dmg = projected_damage(c.damage_per_target, strength, weak_active, false);
            (i32::from(c.cost), dmg)
        })
        .collect();
    if aoe_items.is_empty() {
        return false;
    }
    let raw = knapsack_max_damage(&aoe_items, energy);
    monsters.iter().all(|m| {
        let after_vuln = if m.vulnerable {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            {
                (raw as f32 * 1.5).floor() as i32
            }
        } else {
            raw
        };
        after_vuln >= m.hp_plus_block
    })
}

/// 0/1 knapsack: max sum of `damage` for items chosen with total
/// `cost <= capacity`. Items with `cost <= 0` are always taken (free
/// damage). Caller guarantees the bag is small (typical hand: <= 10
/// cards, capacity <= 5 energy) so the O(n × capacity) DP is trivial.
fn knapsack_max_damage(items: &[(i32, i32)], capacity: i32) -> i32 {
    if capacity <= 0 {
        // Free items (cost <= 0) still contribute.
        return items.iter().filter(|(c, _)| *c <= 0).map(|(_, d)| *d).sum();
    }
    let cap = capacity as usize;
    let mut dp = vec![0i32; cap + 1];
    let mut free = 0i32;
    for &(cost, dmg) in items {
        if cost <= 0 {
            free += dmg;
            continue;
        }
        let c = cost as usize;
        if c > cap {
            continue;
        }
        for w in (c..=cap).rev() {
            let cand = dp[w - c] + dmg;
            if cand > dp[w] {
                dp[w] = cand;
            }
        }
    }
    free + dp[cap]
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::SmallVec;
    use std::num::NonZeroU32;
    use sts2_core::cards::{Card, CardTags, CardType};
    use sts2_core::{
        CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
        MonsterKind, PlayerCombatState, Power, PowerUid, Relic, RelicKind, Side,
    };

    // Build a `CombatState` with the given hand and one monster. Minimal
    // surface — only the fields the detector reads are populated.
    fn build_state(
        hand: &[(CardId, i8, i32, TargetType, bool)], // (id, cost, base_damage, target, _upgraded ignored)
        monster_hp: i32,
        monster_block: i32,
        energy: i8,
    ) -> CombatState {
        build_state_multi(hand, &[(monster_hp, monster_block)], energy)
    }

    fn build_state_multi(
        hand: &[(CardId, i8, i32, TargetType, bool)],
        monsters: &[(i32, i32)],
        energy: i8,
    ) -> CombatState {
        let mut cards: Vec<Card> = Vec::with_capacity(hand.len());
        let mut hand_uids: SmallVec<[CardUid; 16]> = SmallVec::new();
        for (i, &(id, cost, dmg, target, upgraded)) in hand.iter().enumerate() {
            let uid = CardUid::new((i + 1) as u32).unwrap();
            let card_type = match target {
                TargetType::Enemy | TargetType::AllEnemies => CardType::Attack,
                _ => CardType::Skill,
            };
            cards.push(Card {
                uid,
                card_id: id,
                cost: sts2_core::CardEnergyCost::new(cost),
                card_type,
                target_type: target,
                upgraded,
                base_damage: dmg,
                base_block: 0,
                combat_vars: sts2_core::cards::CombatVars::new(),
                enchantments: sts2_core::cards::CardEnchantments::new(),
                tags: CardTags::NONE,
                energy_spent: 0,
                base_replay_count: 0,
                keyword_override: None,
                affliction: None,
            });
            hand_uids.push(uid);
        }

        let mut creatures: Vec<Creature> = Vec::new();
        let mut enemy_uids: SmallVec<[CreatureUid; 8]> = SmallVec::new();
        for (i, &(hp, block)) in monsters.iter().enumerate() {
            let uid = CreatureUid::new((i + 1) as u32).unwrap();
            creatures.push(Creature {
                id: uid.0,
                current_hp: hp,
                max_hp: hp.max(1),
                block,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                monster_id: MonsterId("Nibbit"),
                intent: None,
                intent_history: IntentHistory::default(),
                ai: None,
            });
            enemy_uids.push(uid);
        }

        let player = PlayerCombatState {
            hand: hand_uids,
            energy: i32::from(energy),
            max_energy: energy.max(3),
            current_hp: 50,
            max_hp: 80,
            ..PlayerCombatState::default()
        };

        CombatState {
            phase: CombatPhase::PlayerPlayPhase,
            round: 1,
            side: Side::Player,
            rng: sts2_core::dotnet_rng::Rng::new(0_i32),
            creatures,
            cards: cards.into(),
            enemy_uids,
            player,
            ..CombatState::default()
        }
    }

    fn push_power(state: &mut CombatState, kind: PowerKind, amount: i32, on_player: bool) {
        let id = NonZeroU32::new((state.powers.len() as u32) + 1).unwrap();
        state.powers.push(Power::new(id, kind, amount));
        if on_player {
            state.player.powers.push(PowerUid(id));
        } else {
            let monster_id = state.enemy_uids[0].0;
            if let Some(c) = state.creatures.iter_mut().find(|c| c.id == monster_id) {
                c.powers.push(PowerUid(id));
            }
        }
        // Mirror via `Power.id` lookup: ensure the just-pushed Power's
        // arena `id` matches the uid we just bound.
        let last = state.powers.len() - 1;
        state.powers[last].id = id;
    }

    fn push_monster_power(
        state: &mut CombatState,
        monster_idx: usize,
        kind: PowerKind,
        amount: i32,
    ) {
        let id = NonZeroU32::new((state.powers.len() as u32) + 1).unwrap();
        state.powers.push(Power::new(id, kind, amount));
        let uid = state.enemy_uids[monster_idx].0;
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid) {
            c.powers.push(PowerUid(id));
        }
        let last = state.powers.len() - 1;
        state.powers[last].id = id;
    }

    fn push_relic(state: &mut CombatState, kind: RelicKind) {
        let id = NonZeroU32::new((state.relics.len() as u32) + 1).unwrap();
        state.relics.push(Relic {
            id,
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    }

    // ── Bootstrap: single-monster single-target Strike-only ────────────────

    #[test]
    fn single_strike_lethal_at_six() {
        let state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        assert!(detect_lethal(&state));
    }

    #[test]
    fn single_strike_bails_at_seven() {
        let state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            7,
            0,
            1,
        );
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn two_strikes_lethal_at_twelve() {
        let state = build_state(
            &[
                (CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false),
                (CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false),
            ],
            12,
            0,
            2,
        );
        assert!(detect_lethal(&state));
    }

    #[test]
    fn strike_bails_when_block_pushes_over() {
        let state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            5,
            2,
            1,
        );
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn out_of_phase_bails() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        state.phase = CombatPhase::EnemyTurnExec;
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn terminal_state_bails() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        state.is_over = true;
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn unknown_card_bails() {
        // `BodySlam` not in the allow-list — bail even when the rest of
        // the hand is fine.
        let state = build_state(
            &[
                (CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false),
                (CardId::BodySlam, 1, 0, TargetType::Enemy, false),
            ],
            6,
            0,
            1,
        );
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn no_living_monsters_bails() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        state.creatures[0].current_hp = 0;
        assert!(!detect_lethal(&state));
    }

    // ── Task 1: Strength / Weak / Vulnerable folding ──────────────────────

    #[test]
    fn strength_pushes_lethal_over_hp() {
        // Strike base 6 + Strength 2 = 8. Monster HP 8 → fires.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            8,
            0,
            1,
        );
        push_power(&mut state, PowerKind::Strength, 2, true);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn weak_floors_damage_below_hp() {
        // Strike 6 × 0.75 = 4 (floor). Monster HP 6 → bails.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_power(&mut state, PowerKind::Weak, 1, true);
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn weak_still_lethal_when_knapsack_covers_floor() {
        // Strike 6 × 0.75 = 4. Monster HP 4 → fires.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            4,
            0,
            1,
        );
        push_power(&mut state, PowerKind::Weak, 1, true);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn vulnerable_makes_lethal() {
        // Strike 6 × 1.5 = 9. Monster HP 9 → fires.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            9,
            0,
            1,
        );
        push_monster_power(&mut state, 0, PowerKind::Vulnerable, 1);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn zero_amount_strength_is_no_op() {
        // Strength sentinel with amount=0 must NOT bail — the bootstrap
        // bailed on any power presence; task-1 should fold and proceed.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_power(&mut state, PowerKind::Strength, 0, true);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn unknown_player_power_bails() {
        // PowerKind::DemonForm is not in the supported folding set.
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_power(&mut state, PowerKind::DemonForm, 1, true);
        assert!(!detect_lethal(&state));
    }

    // ── Task 2: Multi-monster AOE detection ───────────────────────────────

    #[test]
    fn breakthrough_kills_two_equal_monsters() {
        // 2 monsters HP=9, 1× Breakthrough (9 AOE), energy=1 → fires.
        let state = build_state_multi(
            &[(CardId::Breakthrough, 1, 9, TargetType::AllEnemies, false)],
            &[(9, 0), (9, 0)],
            1,
        );
        assert!(detect_lethal(&state));
    }

    #[test]
    fn breakthrough_bails_when_one_monster_too_tall() {
        // 2 monsters HP=9 + HP=10, 1× Breakthrough (9 AOE) → bails.
        let state = build_state_multi(
            &[(CardId::Breakthrough, 1, 9, TargetType::AllEnemies, false)],
            &[(9, 0), (10, 0)],
            1,
        );
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn breakthrough_with_single_target_in_hand_bails_multi() {
        // 2 monsters HP=9 + HP=10; hand has Breakthrough + Strike. AOE-only
        // branch ignores Strike, so the AOE total is just 9; second
        // monster HP 10 → bails (mixed AOE+ST is out of scope).
        let state = build_state_multi(
            &[
                (CardId::Breakthrough, 1, 9, TargetType::AllEnemies, false),
                (CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false),
            ],
            &[(9, 0), (10, 0)],
            2,
        );
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn two_breakthroughs_kill_two_tougher_monsters() {
        // 2× Breakthrough = 18 AOE; HP 18 each → fires.
        let state = build_state_multi(
            &[
                (CardId::Breakthrough, 1, 9, TargetType::AllEnemies, false),
                (CardId::Breakthrough, 1, 9, TargetType::AllEnemies, false),
            ],
            &[(18, 0), (18, 0)],
            2,
        );
        assert!(detect_lethal(&state));
    }

    #[test]
    fn thunderclap_plus_vulnerable_on_one_monster() {
        // Thunderclap is 4 AOE. 2 monsters HP=4 and HP=6 (vulnerable).
        // Raw 4 covers monster A (HP 4 + 0 = 4).
        // Monster B is vulnerable: 4 × 1.5 = 6 → covers HP 6.
        let mut state = build_state_multi(
            &[(CardId::Thunderclap, 1, 4, TargetType::AllEnemies, false)],
            &[(4, 0), (6, 0)],
            1,
        );
        push_monster_power(&mut state, 1, PowerKind::Vulnerable, 1);
        assert!(detect_lethal(&state));
    }

    // ── Task 3: Relic allow-list ──────────────────────────────────────────

    #[test]
    fn allowed_relic_fires() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_relic(&mut state, RelicKind::RingOfTheSnake);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn anchor_relic_allowed() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_relic(&mut state, RelicKind::Anchor);
        assert!(detect_lethal(&state));
    }

    #[test]
    fn excluded_relic_bails() {
        // StrikeDummy is NOT damage-neutral — even though the detector
        // could "afford" to be conservative-and-correct, the gate is
        // "no damage-modifying relics" not "compute with relic".
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_relic(&mut state, RelicKind::StrikeDummy);
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn akabeko_bails() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_relic(&mut state, RelicKind::Akabeko);
        assert!(!detect_lethal(&state));
    }

    #[test]
    fn multiple_allowed_relics_fire() {
        let mut state = build_state(
            &[(CardId::StrikeIronclad, 1, 6, TargetType::Enemy, false)],
            6,
            0,
            1,
        );
        push_relic(&mut state, RelicKind::RingOfTheSnake);
        push_relic(&mut state, RelicKind::JuzuBracelet);
        push_relic(&mut state, RelicKind::OddlySmoothStone);
        assert!(detect_lethal(&state));
    }

    // ── Knapsack edge cases ───────────────────────────────────────────────

    #[test]
    fn knapsack_picks_best_subset() {
        // 3 energy; cards (cost, dmg): (2, 10), (1, 4), (1, 4). Best
        // = 10 + 4 = 14 (the second 4 doesn't fit).
        let items = [(2, 10), (1, 4), (1, 4)];
        assert_eq!(knapsack_max_damage(&items, 3), 14);
    }

    #[test]
    fn knapsack_zero_capacity_includes_free_items() {
        let items = [(0, 5), (1, 4)];
        assert_eq!(knapsack_max_damage(&items, 0), 5);
    }
}
