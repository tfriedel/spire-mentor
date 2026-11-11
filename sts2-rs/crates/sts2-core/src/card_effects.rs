//! Card-play effect primitives + the `EffectResult` ADT consumed by the
//! `PlayCard` dispatch registry (bead `sts2-rs-ov4` / P6-1).
//!
//! Ports the player-side primitives Python keeps inline inside the
//! `sts2-rl-agent@01c094d:sts2_env/cards/*` factory bodies — `apply_power`,
//! `gain_block`, `deal_damage`. The dispatch registry lives in
//! `sts2-content::cards::dispatch`; the actual game-state mutation lives
//! here so that downstream beads can call these primitives directly
//! without a registry round-trip.

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::{OnceLock, RwLock};

use crate::cards::{Card, CardId, KeywordOverride, VarKey};
use crate::damage;
use crate::hooks::{self, BlockCtx, BlockGainedCtx, CreatureRef, ValueProps};
use crate::relic_effects::channels::{
    dispatch_if_live, fold_if_live, relic_channel_live, relic_guarded_channel_live, GuardedChannel,
    PlannedChannel,
};
use crate::{CardUid, CombatState, CreatureUid, Power, PowerKind, PowerUid};

// ---------------------------------------------------------------------------
// EffectResult
// ---------------------------------------------------------------------------

/// Outcome of a single dispatched `PlayCard` effect.
///
/// The registry handler returns one of these; `combat_loop::play_card`
/// merges the result into its terminal-or-advanced answer. `NotImplemented`
/// is the explicit "this `CardId` has no registered handler" sentinel — the
/// fallback path falls through to the same energy/hand/discard plumbing as
/// a no-op effect, matching Python's behaviour for unimplemented cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectResult {
    /// Effect ran successfully (or was a deliberate no-op like `AscendersBane`).
    Resolved,
    /// No registered handler for this `CardId`. Caller treats this the same as
    /// `Resolved` for the purposes of the play sequence but downstream tests
    /// can assert which cards still need wiring.
    NotImplemented,
}

// ---------------------------------------------------------------------------
// Damage / block / power primitives shared by every dispatch handler
// ---------------------------------------------------------------------------

/// Deal `base` damage from the player to `target_uid` through the canonical
/// `apply_damage` pipeline. Returns the resulting HP loss.
///
/// **Bead sts2-rs-4nh5**: the originating [`CardId`] is auto-derived from
/// `state.player.play.last()` — i.e. the card currently mid-resolution
/// (`combat_loop::play_card` moves the played card into `Play` BEFORE
/// dispatching the effect, and back out to `Discard` AFTER). Turn-end orb
/// passive ticks and other non-card damage paths that route through this
/// helper will see an empty play pile and thread `None`. Orb evoke effects
/// fired from inside a card's dispatch (Catalyst/Discharge equivalents)
/// ARE card-sourced and correctly carry the played card's id — that
/// matches "what is the proximate damage source?" semantics relied on by
/// `FakeStrikeDummy`'s card-tag gate (46ms).
pub fn deal_damage_player_to_enemy(
    state: &mut CombatState,
    target_uid: CreatureUid,
    base: i32,
) -> i32 {
    let card = originating_card_id(state);
    damage::apply_damage(
        state,
        CreatureRef::Player,
        target_uid,
        base,
        ValueProps::MOVE,
        card,
    )
}

/// Fractional-base twin of [`deal_damage_player_to_enemy`]: deal a RATIONAL
/// `base_num/base_den` damage from the player to `target_uid` through
/// [`damage::apply_damage_full_frac`] (single floor at the end). Used by
/// Thrash, whose `Damage.BaseValue` accumulates an un-floored decimal fold.
/// Returns the resulting HP loss. Bead `sts2-rs-thrash-decimal`.
pub fn deal_damage_player_to_enemy_frac(
    state: &mut CombatState,
    target_uid: CreatureUid,
    base_num: i64,
    base_den: i64,
) -> i32 {
    let card = originating_card_id(state);
    damage::apply_damage_full_frac(
        state,
        CreatureRef::Player,
        target_uid,
        base_num,
        base_den,
        ValueProps::MOVE,
        card,
    )
    .hp_loss
}

/// Compute the player's outgoing **modified** attack damage for a card-sourced
/// `base` WITHOUT dealing it — mirrors C# `Hook.ModifyDamage(runState,
/// combatState, target: null, owner.Creature, base, ValueProp.Move, card,
/// ModifyDamageHookType.All, …)`.
///
/// Runs the read-only powers/relic-arm `modify_damage` pipeline as the player
/// (dealer), so dealer-side modifiers apply: Strength (additive) and Weak
/// (multiplicative), with the same single-floor as a real hit. C# passes a
/// `null` target, so no target-side modifier (Vulnerable) applies; Rust spells
/// that as [`CreatureRef::Nobody`], which owns nothing and so fails every
/// `owner == ctx.target` gate by construction. (Before GH #558 this passed the
/// PLAYER as a "harmless" stand-in target, which silently applied the player's
/// own Vulnerable to the fold.) The `relic_effects`-family additive
/// (MiniatureCannon &c.) is intentionally omitted — those are target/card
/// specific and don't participate in a generic damage *fold*.
///
/// Used by Thrash's permanent damage-fold (`Thrash.OnPlay`): C# folds the
/// exhausted card's `ModifyDamage`'d value into Thrash's `Damage.BaseValue`,
/// not its raw printed damage, so the fold compounds the player's Strength.
#[must_use]
pub fn player_modified_card_damage(state: &CombatState, base: i32) -> i32 {
    let ctx = hooks::DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Nobody,
        props: ValueProps::MOVE,
    };
    hooks::modify_damage(state, base, &ctx)
}

/// Un-floored RATIONAL twin of [`player_modified_card_damage`]: returns the
/// player's outgoing-modified `base` as `(num, den)` without flooring, so a
/// caller (Thrash's decimal fold) can accumulate the true fraction. Delegates
/// to [`hooks::player_modified_card_damage_frac`]. Bead `sts2-rs-thrash-decimal`.
#[must_use]
pub fn player_modified_card_damage_frac(state: &CombatState, base: i32) -> (i64, i64) {
    hooks::player_modified_card_damage_frac(state, base)
}

/// Rational-base twin of [`player_modified_card_damage_frac`], delegating to
/// [`hooks::player_modified_card_damage_frac_of`].
#[must_use]
pub fn player_modified_card_damage_frac_of(
    state: &CombatState,
    base_num: i64,
    base_den: i64,
) -> (i64, i64) {
    hooks::player_modified_card_damage_frac_of(state, base_num, base_den)
}

/// Like [`deal_damage_player_to_enemy`] but returns the full
/// [`damage::DamageOutcome`] so callers that need the unclamped post-block
/// damage (incl. overkill) — e.g. Fisticuffs' block-on-damage rider
/// (bead `sts2-rs-67yqf`) — can read `total_damage` rather than the clamped
/// `hp_loss` scalar.
pub fn deal_damage_player_to_enemy_full(
    state: &mut CombatState,
    target_uid: CreatureUid,
    base: i32,
) -> damage::DamageOutcome {
    let card = originating_card_id(state);
    damage::apply_damage_full(
        state,
        CreatureRef::Player,
        target_uid,
        base,
        ValueProps::MOVE,
        card,
    )
}

/// Deal `base` damage from the player to **every alive enemy** through
/// the canonical [`damage::apply_damage`] pipeline. Iterates
/// [`CombatState::enemy_uids`] in canonical order; each per-enemy hit
/// runs the full additive → multiplicative → cap → floor → clamp
/// pipeline independently.
///
/// `props` is the [`ValueProps`] bitset the caller wants to pass to
/// each per-enemy `apply_damage` call. Relics that mirror Python's
/// `combat.deal_damage(owner, enemy, dmg, ValueProp.UNPOWERED)` (i.e.
/// `LostWisp`, `MrStruggles`) pass `ValueProps::UNPOWERED`; future
/// callers that mirror Python's `MOVE` default pass `ValueProps::MOVE`.
///
/// Returns the **sum** of per-enemy HP-loss returned by `apply_damage`,
/// so a caller can short-circuit on `0` (no enemies hit / fully
/// absorbed). Skips enemies whose `current_hp` is already `<= 0` at the
/// per-iteration check (the kill-on-previous-iteration case — e.g. a
/// listener that died to the same fan-out tick).
///
/// Used by the event-reward listeners landed in bead `sts2-rs-v7mp`
/// (`LostWisp` on Power-card-played, `MrStruggles` on player-turn-start)
/// and is the canonical iteration shape for any future "deal damage to
/// all alive enemies" surface (e.g. Cleave-style cards).
pub fn deal_damage_to_all_alive_enemies(
    state: &mut CombatState,
    base: i32,
    props: ValueProps,
    card: Option<CardId>,
) -> i32 {
    deal_damage_to_all_alive_enemies_counting_kills(state, base, props, card).0
}

/// [`deal_damage_to_all_alive_enemies`] that also returns how many of its hits
/// killed their target: C# `results.Count(r => r.WasTargetKilled)`
/// (`EchoingSlash.cs:34`), read off each hit's
/// [`damage::DamageOutcome::was_target_killed`] rather than from who is alive
/// afterwards, so a target revived inside the hit (the pressurized Waterfall
/// Giant) still counts. Returns `(total hp loss, kills)`.
pub fn deal_damage_to_all_alive_enemies_counting_kills(
    state: &mut CombatState,
    base: i32,
    props: ValueProps,
    card: Option<CardId>,
) -> (i32, usize) {
    // One `CreatureCmd.Damage` over every target: its kills die after the
    // last hit (CreatureCmd.cs:351-409).
    damage::with_deferred_enemy_deaths(state, |state| {
        deal_damage_to_all_alive_enemies_counting_kills_inner(state, base, props, card)
    })
}

fn deal_damage_to_all_alive_enemies_counting_kills_inner(
    state: &mut CombatState,
    base: i32,
    props: ValueProps,
    card: Option<CardId>,
) -> (i32, usize) {
    // Snapshot the iteration set up-front. Subsequent listener fan-out
    // (e.g. on_damage_received → Thorns reflection → Combust) can mutate
    // `state.enemy_uids` between iterations; the Python pin reads
    // `combat.get_alive_enemies()` once and iterates that list, so the
    // snapshot mirrors the Python semantics exactly.
    let targets: smallvec::SmallVec<[CreatureUid; 8]> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .collect();
    // Phase 1 — C# `CreatureCmd.Damage` first loop (`CreatureCmd.cs:134-151`):
    // compute `Hook.ModifyDamage` for EVERY target against the constant
    // pre-death dealer state, before any `Kill` (`CreatureCmd.cs:285`) runs.
    // Locking the per-target damage here means a mid-AoE death — Shrink
    // removal (dealer ×0.70 powered-attack debuff drops when the applier
    // dies) or a Possess give-back (dealer Strength restored on the thief's
    // death) — cannot retroactively change a surviving target's already-
    // resolved hit. `compute_modified_damage` is read-only, so staging all
    // targets up-front is per-target identical to the old interleaved call.
    // Bead `sts2-rs-8gzvn`.
    // A `for` loop rather than a `map`, because the flashing variant needs
    // `&mut state` for the event log. Still one pass over the targets before
    // any damage is applied, which is the property the pre-compute exists for —
    // and it puts every target's `AfterModifyingDamageAmount` inside the first
    // loop, exactly where `CreatureCmd.cs:262` dispatches it (before the single
    // trailing `Kill`), rather than interleaved with the applies below.
    let mut precomputed: smallvec::SmallVec<[(CreatureUid, i32); 8]> = smallvec::SmallVec::new();
    for uid in targets {
        let dmg = damage::compute_modified_damage_flashing(
            state,
            CreatureRef::Player,
            uid,
            base,
            props,
            card,
        );
        precomputed.push((uid, dmg));
    }
    let mut total = 0_i32;
    let mut kills = 0_usize;
    for (target_uid, final_damage) in precomputed {
        // Per-iteration alive recheck: a previous iteration may have
        // killed a chained-listener target (or this very target through
        // reflection). Skip dead targets so the pipeline doesn't write
        // a no-op HP-loss event with the dead-creature row.
        let alive = state
            .creatures
            .iter()
            .any(|c| c.id == target_uid.0 && c.current_hp > 0);
        if !alive {
            continue;
        }
        let outcome = damage::apply_damage_with_final(
            state,
            CreatureRef::Player,
            target_uid,
            final_damage,
            props,
        );
        total = total.saturating_add(outcome.hp_loss);
        kills += usize::from(outcome.was_target_killed);
    }
    (total, kills)
}

/// Resolve the player's Osty pet to the [`CreatureRef`] used as the **dealer**
/// for Osty-attributed attacks (C# `DamageCmd.Attack(...).FromOsty(Owner.Osty,
/// card)`). Returns `None` when no Osty has been summoned, or when the recorded
/// pet creature is missing/dead.
///
/// The Osty lives in `state.creatures` keyed by [`CreatureUid`] but is NOT a
/// member of `enemy_uids`; the engine's `owner_of_listener` already classifies
/// any non-player creature row as [`CreatureRef::Enemy`] (the `Ally(u8)` variant
/// is reserved for true multiplayer allies in `state.allies`), so an
/// Osty-as-dealer is expressed as `Enemy(pet_uid)`. Routing the dealer this way
/// means dealer-side modifiers read the Osty's powers (Strength, Calcify's
/// `dealer is Osty` gate, …) and player-dealer-gated effects correctly do NOT
/// fire — matching C# `FromOsty`, where the proximate dealer is the pet.
#[must_use]
pub fn osty_dealer_ref(state: &CombatState) -> Option<CreatureRef> {
    let pet_uid = state.player.pet_uid?;
    let alive = state
        .creatures
        .iter()
        .any(|c| c.id == pet_uid.0 && c.current_hp > 0);
    alive.then_some(CreatureRef::Enemy(pet_uid))
}

/// Deal `base` damage to a single enemy with the player's **Osty** as the
/// dealer (C# `DamageCmd.Attack(base).FromOsty(Owner.Osty, card).Targeting(t)`).
/// No-op returning `0` when the Osty is missing/dead (callers normally gate on
/// the Osty-missing check first, but this is defensive). The originating card
/// id is threaded as the damage source, matching C# `cardSource = this`.
pub fn deal_damage_from_osty_to_enemy(
    state: &mut CombatState,
    target_uid: CreatureUid,
    base: i32,
) -> i32 {
    let Some(dealer) = osty_dealer_ref(state) else {
        return 0;
    };
    let card = originating_card_id(state);
    // One `AttackCommand` even when the target is already dead: C# still runs
    // it (no hit) and its `AfterAttack` (BoneFlute) — a replayed Unleash /
    // Snap whose first play killed the target (loadout Necrobinder
    // 1038-v0.111.0, 1192-v0.107.1).
    osty_attack_command(state, |state| {
        damage::apply_damage(state, dealer, target_uid, base, ValueProps::MOVE, card)
    })
}

/// Run `f` as ONE Osty `AttackCommand`: its hits leave
/// `osty_attacks_this_turn` alone and the command counts once when it ends.
/// C# `AttackCommand.Execute` records a single `CreatureAttackedEntry` after
/// all hits and targets (AttackCommand.cs:672), and Rattle's hit count reads
/// those entries. Call only when the Osty is present (`osty_dealer_ref`):
/// a missing Osty issues no command.
pub fn osty_attack_command<R>(state: &mut CombatState, f: impl FnOnce(&mut CombatState) -> R) -> R {
    let outer = state.combat_log.osty_attack_command_open;
    state.combat_log.osty_attack_command_open = true;
    let r = f(state);
    state.combat_log.osty_attack_command_open = outer;
    if !outer {
        state.combat_log.osty_attacks_this_turn =
            state.combat_log.osty_attacks_this_turn.saturating_add(1);
        flatten_after_osty_attack(state);
        // `Hook.AfterAttack` runs after the hit loop even when no hit landed
        // (`AttackCommand.cs:551-673`: no live target breaks the loop), so the
        // command owes `BoneFlute.AfterAttack`'s block either way. Bone Shards
        // into a lone reviving TestSubject (Adaptable) hits nothing and still
        // grants +2 (loadout Necrobinder 1192-v0.111.0). Hits latch it too
        // (`bone_flute::on_after_damage_given_bone_flute`); it is a flag, so
        // the command grants once.
        if !state.is_over {
            state.combat_log.bone_flute_latched = true;
        }
        // The command's `Hook.AfterAttack` runs now, before the rest of the
        // card (Bone Shards: BoneFlute's +2, then the card's own block), and
        // with it Skittish's grant for a hit this command armed.
        crate::hooks::grant_latched_skittish(state);
    }
    r
}

/// C# `Flatten.AfterAttack` (`Flatten.cs`): when the owner's Osty finishes an
/// attack, every Flatten in combat gets `EnergyCost.SetThisTurn(0)`.
pub fn flatten_after_osty_attack(state: &mut CombatState) {
    for c in &mut state.cards {
        if c.card_id == crate::cards::CardId::Flatten {
            c.cost.set_this_turn(0, false);
        }
    }
}

/// Osty-dealer analogue of [`deal_damage_to_all_alive_enemies`] (C#
/// `.FromOsty(...).TargetingAllOpponents(...)`, e.g. Bone Shards). Snapshots the
/// alive-enemy set up front (matching the player AoE helper's
/// listener-fan-out-safe iteration), then hits each with the Osty as dealer.
/// No-op returning `0` when the Osty is missing/dead.
pub fn deal_damage_from_osty_to_all_alive_enemies(state: &mut CombatState, base: i32) -> i32 {
    let Some(dealer) = osty_dealer_ref(state) else {
        return 0;
    };
    let card = originating_card_id(state);
    let targets: smallvec::SmallVec<[CreatureUid; 8]> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .collect();
    // One command across every target (`TargetingAllOpponents`), whose hit is
    // one multi-target `CreatureCmd.Damage(validTargets)` (AttackCommand.cs:654):
    // every target's HP loss lands before any after-hit hook or kill.
    osty_attack_command(state, |state| {
        damage::with_deferred_enemy_deaths(state, |state| {
            let mut total = 0_i32;
            for target_uid in targets {
                let alive = state
                    .creatures
                    .iter()
                    .any(|c| c.id == target_uid.0 && c.current_hp > 0);
                if !alive {
                    continue;
                }
                total = total.saturating_add(damage::apply_damage(
                    state,
                    dealer,
                    target_uid,
                    base,
                    ValueProps::MOVE,
                    card,
                ));
            }
            total
        })
    })
}

/// Sample a uniform-random alive enemy from `state.enemy_uids`, filtered
/// to `current_hp > 0`, using `state.rng`. Returns `None` when no enemy
/// is alive.
///
/// Mirrors Python's `combat.rng.choice(combat.alive_enemies)` shape at
/// `sts2-rl-agent@01c094d:sts2_env/core/combat.py` (the `alive_enemies`
/// view + `rng.choice` call used by handlers like
/// `cards/regent.py:L549-555` Stardust). Takes `&mut CombatState` rather
/// than a separate `&mut RngCore` so callers don't have to manually
/// split-borrow against `state.rng` mid-loop.
///
/// Companion to [`CombatState::get_random_enemy`] (bead `sts2-rs-v7mp`,
/// same iteration shape) — this is the `card_effects`-level
/// re-export so card handlers don't have to reach into combat-state
/// methods for the same primitive.
///
/// Bead `sts2-rs-c46l`: the canonical random-target picker for
/// `card_effects`-level fan-out (Stardust today; future random-target
/// cards follow the same shape).
#[must_use]
pub fn pick_random_alive_enemy(state: &mut CombatState) -> Option<CreatureUid> {
    let alive: smallvec::SmallVec<[CreatureUid; 8]> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .collect();
    if alive.is_empty() {
        return None;
    }
    let idx = state
        .run_rng
        .combat_targets
        .next_int_exclusive(0, alive.len() as i32) as usize;
    Some(alive[idx])
}

/// Derive the originating [`CardId`] from the play pile — the card on top
/// of `state.player.play` is the one whose dispatch is currently executing
/// (see `combat_loop::play_card`: `Hand → Play` happens before dispatch,
/// `Play → Discard` after). Public so the per-character `deal_damage_all_enemies`
/// content helpers can thread the same card source into
/// [`deal_damage_to_all_alive_enemies`] that the single-target
/// [`deal_damage_player_to_enemy`] derives internally.
#[must_use]
pub fn originating_card_id(state: &CombatState) -> Option<CardId> {
    resolving_card_inst(state).map(|c| c.card_id)
}

/// Add `base` block to the player after running the canonical `modify_block`
/// listener chain (Frail / Dexterity / Inflame-equivalents fold in there).
pub fn gain_block_player(state: &mut CombatState, base: i32) -> i32 {
    gain_block_player_with_props(state, base, ValueProps::MOVE)
}

/// Compute the block the player WOULD gain from `base` through the canonical
/// powered (`Move`) [`hooks::modify_block`] pipeline (Dexterity / Frail / …)
/// **without** actually gaining it.
///
/// Mirrors C# `Hook.ModifyBlock(state, creature, base, ValueProp.Move, card,
/// cardPlay, out _)` for player block: same `Move` ctx + card source (top of
/// the play pile) as [`gain_block_player`]. Used by cards that bake a
/// *modified* block amount into a `BlockNextTurn`-style power whose printed
/// value differs from the now-block — e.g. Glitterstream applies `Block` now
/// **and** `BlockNextTurnPower` equal to `ModifyBlock(BlockNextTurn)`, so the
/// next-turn block must carry Dexterity/Frail too (`Glitterstream.cs:33-39`).
#[must_use]
pub fn modify_player_block_move(state: &CombatState, base: i32) -> i32 {
    let ctx = BlockCtx {
        target: CreatureRef::Player,
        props: ValueProps::MOVE,
        card_source: resolving_card_uid(state),
    };
    hooks::modify_block(state, base, &ctx)
}

/// [`hooks::modify_block_is_positive`] for the player's powered card block,
/// with the same ctx as [`modify_player_block_move`].
#[must_use]
pub fn player_block_move_is_positive(state: &CombatState, base: i32) -> bool {
    let ctx = BlockCtx {
        target: CreatureRef::Player,
        props: ValueProps::MOVE,
        card_source: resolving_card_uid(state),
    };
    hooks::modify_block_is_positive(state, base, &ctx)
}

/// C# `PowerCmd.Apply` with a decimal amount in (0, 1): non-zero, so it
/// applies, but `PowerModel.ApplyInternal` stores `(int)amount`
/// (PowerModel.cs:564-573), leaving a 0-amount instance. Onto an existing
/// instance the truncated 0 adds nothing. No-op when the player already has
/// one or has no creature row.
pub fn apply_truncated_zero_power_to_player(state: &mut CombatState, kind: PowerKind) {
    let Some(player_uid) = state.player.creature_uid else {
        return;
    };
    let Some(creature_idx) = state.creatures.iter().position(|c| c.id == player_uid.0) else {
        return;
    };
    let has = state.creatures[creature_idx].powers.iter().any(|puid| {
        state
            .powers
            .get_by_id(puid.0)
            .is_some_and(|p| p.kind == kind)
    });
    if has || crate::combat_loop::combat_is_ending(state) {
        return;
    }
    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    state.powers.push(Power {
        id: new_id,
        kind,
        amount: 0,
        counter: 0,
        applier: Some(player_uid),
        skip_next_tick: false,
    });
    state.creatures[creature_idx].powers.push(PowerUid(new_id));
    crate::coop::stamp_power_seat(state, new_id, kind);
}

/// Block a player card gives co-op teammate `ally` (`state.allies[ally]`):
/// C# `CreatureCmd.GainBlock(cardPlay.Target, block, Move, cardPlay)` from an
/// `AnyAlly` card. Runs the powered block pipeline with the teammate as the
/// target, so the player's Dexterity applies (the card is theirs) and the
/// player's own Frail does not. The teammate's powers are not simulated, so
/// their Frail is not either. Returns the block gained.
pub fn gain_block_ally(state: &mut CombatState, ally: u8, base: i32) -> i32 {
    let ctx = BlockCtx {
        target: CreatureRef::Ally(ally),
        props: ValueProps::MOVE,
        card_source: resolving_card_uid(state),
    };
    let gained = hooks::modify_block(state, base, &ctx).max(0);
    if let Some(a) = state.allies.get_mut(usize::from(ally)) {
        a.block += gained;
    }
    gained
}

/// Player block gain that is `Unpowered` (Frail-exempt). C# powers that
/// grant block via `CreatureCmd.GainBlock(..., ValueProp.Unpowered, ...)`
/// — e.g. `CrimsonMantlePower.AfterPlayerTurnStart` (CrimsonMantlePower.cs:32)
/// — bypass `IsPoweredCardOrMonsterMoveBlock`, so Frail's 0.75× must NOT
/// apply. The default [`gain_block_player`] path is `Move` (powered) for
/// card/relic block; this sibling carries `Unpowered` for those specific
/// power-triggered grants.
pub fn gain_block_player_unpowered(state: &mut CombatState, base: i32) -> i32 {
    gain_block_player_with_props(state, base, ValueProps::UNPOWERED)
}

/// Shared body for player block gain. `props` selects whether the block is
/// `Move` (powered → Frail's 0.75× applies) or `Unpowered` (Frail-exempt).
pub fn gain_block_player_with_props(state: &mut CombatState, base: i32, props: ValueProps) -> i32 {
    // q6bix F-N: thread the originating card (top of the play pile during
    // a card's dispatch) into the block ctx so tag-predicated additive
    // listeners — `FastenPower` (+block on Defend-tagged cards) and the
    // VitruvianMinion multiplicative arm — can read the source card's
    // tags.
    //
    // ONLY powered (`MOVE`) block carries the card: that is C#'s
    // card/monster-move block, where `CreatureCmd.GainBlock` receives the
    // `cardPlay`. `Unpowered` block (relics, power hooks like FeelNoPain,
    // CrimsonMantle) is granted with `cardSource = null` in C#. Crucially,
    // such a power hook can fire *during* a card-play — e.g.
    // `FeelNoPainPower.AfterCardExhausted` while `SecondWind` exhausts the
    // hand — so the play pile is NOT empty; inferring `card_source` from it
    // would falsely attribute the grant to the in-flight card and let
    // card-predicated listeners (Vambrace's `2×`, Fasten, …) fire on block
    // C# never card-sources (q6bix PLAYER_STATE cluster `1db105`:
    // Vambrace doubling FeelNoPain on top of SecondWind).
    let card_source = if props.contains(ValueProps::MOVE) {
        resolving_card_uid(state)
    } else {
        None
    };
    let ctx = BlockCtx {
        target: CreatureRef::Player,
        props,
        card_source,
    };
    let final_block = hooks::modify_block(state, base, &ctx);
    // C# `FastenPower.AfterModifyingBlockAmount` (`FastenPower.cs:39`) flashes
    // unconditionally — it carries no target/props of its own, so (mirroring
    // the established `AfterModifying*` precedent: `Effect::FlashPower`'s doc,
    // `damage::flash_intangible_on_target`) it fires whenever the
    // `ModifyBlockAdditive` pipeline runs for the owner, independent of
    // whether Fasten's own additive guard (`is_powered` + Defend tag) applied.
    // `gain_block_player_with_props` is the single canonical chokepoint for
    // every player block gain (Move and Unpowered alike), so it is also the
    // one Rust site that matches C#'s call frequency for this hook.
    if crate::event_log::armed(state) && state.player_power_amount(PowerKind::Fasten) > 0 {
        crate::event_log::flash_player_power(state, PowerKind::Fasten);
    }
    state.player.block = state.player.block.saturating_add(final_block);
    // Event log (C# `BlockGainedEntry`), post-modify-chain like C#. Zero /
    // negative grants record nothing, matching the `final_block > 0` gate the
    // `after_block_gained` dispatch below uses.
    if final_block > 0 && crate::event_log::armed(state) {
        let receiver = crate::event_log::LogCreature::Player(state.player.character);
        crate::event_log::record(
            state,
            crate::event_log::EventLogEntry::BlockGained {
                receiver,
                amount: final_block,
            },
        );
    }
    // Bead `sts2-rs-73up8` (d03 Phase-2 prereq): back-sync the legacy
    // player block into the mirror Creature row so the Phase-2
    // inversion (Creature as read source of truth) stays coherent.
    state.sync_creature_block_from_player();
    // Bead `sts2-rs-6ynt.6.7.1`: fire `after_block_gained` so listeners
    // (e.g. `GrapplePower` per bead `sts2-rs-6ynt.6.7`) can react. Only
    // dispatched when block actually landed (>0) — matches Python's
    // `creature.gain_block` short-circuit when the post-modifier amount
    // clamps to zero.
    if final_block > 0 {
        let gained_ctx = BlockGainedCtx {
            target: CreatureRef::Player,
            amount: final_block,
            card_source,
        };
        hooks::dispatch_after_block_gained(state, &gained_ctx);
        // Bead `sts2-rs-fopk` (6z05.3) — relic-side mirror of
        // `after_block_gained`. Empty-bus early-out keeps the no-
        // listener fast path lock-free.
        if relic_channel_live(state, PlannedChannel::AfterBlockGained) {
            let _ = crate::relic_effects::dispatch_after_block_gained(state, &gained_ctx);
        }
    }
    // Return the actual post-modifier block gained (C#
    // `CreatureCmd.GainBlock` returns the granted amount). DodgeAndRoll
    // feeds this into its BlockNextTurn apply so the carried-over block
    // reflects Dexterity/Frail, matching C#.
    final_block
}

/// Apply `amount` stacks of `kind` to the [`Creature`] at `target_uid`,
/// regardless of whether that creature is the player-slot mirror or an
/// enemy. Existing instances stack additively (matches Python's
/// `StackingPower.stack_power` semantics for the load-bearing debuffs).
/// No-op if the creature is dead, missing, or `amount <= 0`.
///
/// This is the creature-keyed primitive that downstream relic listeners
/// (event-reward Wave 2: `SwordOfJade`, `EmberTea`, `FakeSneckoEye`, …
/// per bead `sts2-rs-46ms`) call to install player-slot powers
/// (`Strength`, `Confused`, …) the same way enemy debuffs are routed.
///
/// Bead `sts2-rs-vt7n`: lifted from the prior enemy-only
/// [`apply_power_to_enemy`] surface. The enemy variant is retained as a
/// thin wrapper to avoid churning the ~25 P6 card-effect call sites.
///
/// Artifact debuff-negation gate, scoped to the **target** creature.
///
/// When `kind` is a debuff (per [`crate::afflictions::is_affliction`]) and
/// the creature at `creature_idx` holds an `Artifact` power with `amount > 0`,
/// consume one stack (dropping the power at 0) and return `true` to signal the
/// debuff is negated. Returns `false` otherwise (debuff applies normally).
///
/// This mirrors C# `ArtifactPower` (`ArtifactPower.cs`): its
/// `TryModifyPowerAmountReceived` zeroes a visible debuff destined for its own
/// owner, and `AfterModifyingPowerAmountReceived` runs `PowerCmd.Decrement`.
/// `PowerCmd.Apply` invokes those through the `ModifyPowerAmountReceived`
/// channel. Because Rust's `modify_power_amount_received` dispatch is
/// relic-only (Artifact is a power, not a relic), the equivalent lives here as
/// an early target-scoped gate — covering both player- and monster-held
/// Artifact uniformly. The negation is deterministic (no RNG draw), matching
/// C#. The earlier Python-pin `try_block_debuff` hook is left unused: its ctx
/// is a placeholder and it walks listeners globally (no target scoping), so it
/// cannot express "Artifact guards only its own owner" — bead `sts2-rs-zdjjx`.
/// Remove EVERY instance of `kind` from `creature` (all stacks), mirroring C#
/// `PowerCmd.Remove<TPower>`. No-op when the creature holds none. Strips the
/// power from the global `state.powers` arena and from the creature's `powers`
/// uid list (plus the legacy `state.player.powers` list when the target is the
/// player's mirror). Used by Expose (`Remove<ArtifactPower>` before applying
/// Vulnerable, so the debuff is not eaten by Artifact). Reuses the same
/// triple-retain teardown [`try_consume_artifact_for_debuff`] uses at 0.
pub fn remove_creature_power(state: &mut CombatState, creature: CreatureUid, kind: PowerKind) {
    let Some(idx) = state.creatures.iter().position(|c| c.id == creature.0) else {
        return;
    };
    let pids: smallvec::SmallVec<[core::num::NonZeroU32; 4]> = state.creatures[idx]
        .powers
        .iter()
        .filter_map(|uid| {
            state
                .powers
                .iter()
                .find(|p| p.id == uid.0 && p.kind == kind)
                .map(|p| p.id)
        })
        .collect();
    for pid in pids {
        state.powers.retain(|p| p.id != pid);
        state.player.powers.retain(|u| u.0 != pid);
        for c in &mut state.creatures {
            c.powers.retain(|u| u.0 != pid);
        }
    }
}

/// Consume one Artifact stack on the creature at `creature_idx`, returning
/// `true` when a stack was present (and decremented / removed at 0). Shared
/// by the affliction- and signed-attribute Artifact gates.
fn consume_one_artifact_stack(state: &mut CombatState, creature_idx: usize) -> bool {
    // Perf 2026-07-04: O(1) absence gate — no live Artifact anywhere in the
    // arena (the `present_kinds` superset proof) means the per-uid scan
    // below cannot find one; return "not negated" without walking.
    if !state.powers.present_kinds().contains(PowerKind::Artifact) {
        return false;
    }
    let Some(art_pid) = state.creatures[creature_idx].powers.iter().find_map(|uid| {
        state
            .powers
            .get_by_id(uid.0)
            .filter(|p| p.kind == PowerKind::Artifact && p.amount > 0)
            .map(|p| p.id)
    }) else {
        return false;
    };
    let mut remove = false;
    if let Some(p) = state.powers.get_mut_by_id(art_pid) {
        p.amount -= 1;
        remove = p.amount == 0;
    }
    if remove {
        state.powers.retain(|p| p.id != art_pid);
        state.player.powers.retain(|u| u.0 != art_pid);
        for c in &mut state.creatures {
            c.powers.retain(|u| u.0 != art_pid);
        }
    }
    true
}

fn try_consume_artifact_for_debuff(
    state: &mut CombatState,
    creature_idx: usize,
    kind: PowerKind,
) -> bool {
    if !crate::afflictions::is_affliction(kind) {
        return false;
    }
    consume_one_artifact_stack(state, creature_idx)
}

/// C# `ArtifactPower.TryModifyPowerAmountReceived` negates ANY application
/// whose `canonicalPower.GetTypeForAmount(amount) == PowerType.Debuff`
/// (ArtifactPower.cs) — which, for an `AllowNegative` attribute power
/// (Strength / Dexterity / Focus), is the case whenever `amount < 0`. The
/// affliction-keyed [`try_consume_artifact_for_debuff`] does not cover these
/// signed attribute DRAINS (Malaise's `-Strength`, Mangle, EnfeeblingTouch),
/// so the content `apply_power_signed_to_creature` helpers — which bypass the
/// canonical apply path — must consult this for the negative direction.
/// Returns `true` (consuming one Artifact stack) when the drain is negated;
/// a positive (buff-direction) restore is never gated, matching C#.
#[must_use]
pub fn try_consume_artifact_for_signed_debuff(
    state: &mut CombatState,
    target_uid: CreatureUid,
    amount: i32,
) -> bool {
    if amount >= 0 {
        return false;
    }
    let Some(creature_idx) = state
        .creatures
        .iter()
        .position(|c| c.id == target_uid.0 && c.current_hp > 0)
    else {
        return false;
    };
    consume_one_artifact_stack(state, creature_idx)
}

/// C# `PowerModel.IsInstanced` — kinds where every `PowerCmd.Apply` creates
/// a FRESH power instance with its own amount/counter instead of stacking
/// into an existing one. v0.103.2 marks 18 kinds instanced (MagicBomb,
/// Nightmare, Heist, Monologue, TheBomb, Guarded, Covered, Flanking,
/// ToricToughness, TagTeam, RollingBoulder, Swipe, Thievery, Sandpit,
/// Knockdown, Panache, Orbit, Automation). Only the kinds whose Rust
/// listener bodies are verified per-instance-safe are flipped here —
/// migrating a kind whose handler still assumes a single merged stack
/// would silently change its behavior, so the rest move over one at a
/// time as their content beads land. TheBomb: each play is an
/// independent 3-turn countdown carrying its own damage in `counter`
/// (coverage-sweep `card:THE_BOMB_CARD` powers.len cs=3 rs=2 — two
/// overlapping Bombs must NOT collapse into one amount-4 stack). Sandpit:
/// single application per combat (TheInsatiable LIQUIFY), instancing is
/// a no-op today but matches C#. Panache: each play is an independent
/// instance carrying its own damage `amount` and its own per-turn card
/// counter in `counter` (two overlapping Panaches must NOT merge into
/// one 24-damage stack — see bead `sts2-rs-5jgpx`). Bead `sts2-rs-01ygn`.
///
/// TagTeam (bead `sts2-rs-2z2sn`): C# `TagTeamPower.cs:19-21`
/// (StackType=Counter, IsInstanced=true). Each `TagTeam.OnPlay`
/// (`TagTeam.cs:30`) does `PowerCmd.Apply<TagTeamPower>(target, 1)`, so N
/// plays leave N separate amount-1 instances, NOT one amount-N stack
/// (coverage-sweep seed 1702 turn 3: `powers.len` cs=2 rs=1,
/// `powers[1].amount` cs=1 rs=2). Per-instance-safe in Rust: TagTeam is
/// classified `debuff_perm` (no per-turn tick/expiry) and has no
/// `modify_card_play_count` listener body wired yet, so its only
/// observable surface is the instance's existence + amount — two amount-1
/// instances behave identically to (and now match C#'s) two stacks. C#
/// `AfterApplied` only stamps a display-only `Applier` string with no Rust
/// analog, so no post-apply data stamping is required.
///
/// Knockdown (bead `sts2-rs-t2jqf`): C# `KnockdownPower.cs:21-22`
/// (StackType=Counter, IsInstanced=true). `Knockdown.OnPlay`
/// (`Knockdown.cs:35`) does `PowerCmd.Apply<KnockdownPower>(target, 2)`, so N
/// plays leave N separate amount-2 instances, NOT one amount-2N stack
/// (DISTILLED_CHAOS sweep-v2 full seed 1224 turn 1: two auto-played KNOCKDOWN
/// cards gave `monsters[0].powers.len` cs=2 rs=1 + `powers[0].amount` cs=2
/// rs=4). Per-instance-safe in Rust: KnockdownPower's only ported observable
/// surface is the instance's existence + amount + its `AfterTurnEnd` removal
/// at the owner's (enemy) turn end (the `PowerKind::Knockdown` arm in
/// `hooks`) — N amount-2 instances all get removed together at that turn
/// end, identical to one merged stack. The C# `ModifyDamageMultiplicative`
/// surface (which WOULD fold N instances multiplicatively) is unported
/// because Knockdown is a `MultiplayerOnly` card whose multiplier only fires
/// for a non-applier ally attacker — irrelevant in the single-player sim. C#
/// `AfterApplied` stamps a display-only `Applier` string with no Rust analog.
#[must_use]
pub const fn power_kind_is_instanced(kind: PowerKind) -> bool {
    matches!(
        kind,
        PowerKind::TheBomb
            | PowerKind::Sandpit
            | PowerKind::Panache
            | PowerKind::TagTeam
            | PowerKind::Knockdown
            // C# `FlankingPower.IsInstanced => true` — each Flanking play is an
            // independent damage-multiplier debuff instance (the same
            // MultiplayerOnly pattern as Knockdown: inert in single-player, but
            // OBSERVABLE in the enemy's power list). Without this two plays
            // merged into one stacked `FLANKING 4` instead of C#'s two separate
            // `FLANKING 2` instances. The per-instance owner-side turn-end
            // removal already lives in `hooks::after_turn_end`.
            | PowerKind::Flanking
            // C# `GuardedPower.IsInstanced => true`: one per Tank that gave
            // it, each halving a powered hit, so two Tanks quarter it. A
            // merged stack halved once. Each instance leaves with its own
            // Tank (`coop::drop_guarded_of`).
            | PowerKind::Guarded
            | PowerKind::Automation
            // v0.111.0 `CacophonyPower.InstanceType => Instanced` (decomp
            // CacophonyPower.cs:16) and `ImitationLearningPower` (:24).
            | PowerKind::Cacophony
            | PowerKind::ImitationLearning
            // C# `RollingBoulderPower.IsInstanced => true` — each card play
            // is an independently-ramping boulder (its own amount + per-turn
            // +5). Bead `sts2-rs-boulder-inst`. The paired per-instance
            // turn-start fire lives in `combat_loop::after_player_turn_start`.
            | PowerKind::RollingBoulder
            // C# `ToricToughnessPower.IsInstanced => true` — each ToricToughness
            // play is an independent block-retention timer (its own stored block
            // + Turns counter). The per-instance `after_block_cleared` re-grant
            // lives in `hooks::after_block_cleared`. Bead `sts2-rs-toric-power`.
            | PowerKind::ToricToughness
            // C# `OrbitPower.InstanceType => Instanced` (OrbitPower.cs:28) — each
            // Orbit card play is an independent energy-spent tracker (its own
            // `Data.energySpent`/`triggerCount`). N plays leave N separate
            // amount-1 instances, NOT one amount-N stack (Regent sweep
            // PLAYER_STATE|ORBIT: two Orbit plays gave `player.powers.len` cs=2
            // rs=1 + `powers[1].amount` cs=1 rs=2). Per-instance-safe in Rust:
            // `hooks::dispatch_after_energy_spent` already iterates EACH Orbit
            // instance's own `counter` (cumulative energy spent) and grants
            // `amount(1) * triggers` per instance — N amount-1 instances grant
            // the same energy as the old amount-N merge, so only the surfaced
            // power list changes. Bead `sts2-rs-orbit-inst`.
            | PowerKind::Orbit
            // C# `MonologuePower.InstanceType => Instanced` (MonologuePower.cs) —
            // each Monologue card play is an independent per-card Strength
            // engine (its own `amountsForPlayedCards`/`StrengthApplied`). N plays
            // leave N separate amount-1 instances, NOT one amount-N stack (Regent
            // sweep PLAYER_STATE|MONOLOGUE: two Monologue plays gave
            // `player.powers.len` cs=2 rs=1). The OLD merge ALSO over-granted
            // Strength: `dispatch_after_card_played` reads the per-card grant off
            // the instance `amount`, so a merged Monologue@2 applied +2 Strength
            // per card where C# applies the constant Strength var (1) PER armed
            // instance — i.e. +1 from the single pre-existing instance (the
            // just-played Monologue is unarmed). Instancing fixes both: each
            // instance carries amount=1, and the per-instance arm/grant/undo in
            // `hooks` (`skip_next_tick` arm, `counter` accumulator,
            // `after_turn_end` undo) is already per-uid. Bead
            // `sts2-rs-monologue-inst`.
            | PowerKind::Monologue
            // C# `NightmarePower.InstanceType => Instanced` (NightmarePower.cs:30)
            // + `StackType.Counter`. Each Nightmare play is an independent
            // clone engine: its `counter` stashes THIS play's selected card
            // uid and its `amount` is that play's clone-count. Two plays must
            // leave TWO instances (card A ×3 + card B ×3), NOT one merged
            // `NIGHTMARE 6` whose single counter would overwrite the first
            // play's card. The turn-start consumer
            // (`hooks::dispatch_before_hand_draw_powers`) and the card-uid
            // stash (`resolve_clone_chosen_card_next_turn`, reverse walk to
            // the freshly-applied instance) are ALREADY per-instance; this
            // aligns the apply path with them. Bead `sts2-rs-nightmare-inst`.
            | PowerKind::Nightmare
    )
}

/// Find the player/creature's existing non-instanced power instance of `kind`
/// for a stack-merge, walking the **mirror + legacy union**.
///
/// `apply_power_to_creature` registers a player power on the mirror Creature
/// (`state.creatures[..].powers`), while monster debuffs applied through the
/// `apply_*_via_legacy_surface` helpers register ONLY on the legacy
/// `state.player.powers` list. A finder that consulted just the mirror would
/// miss a legacy-registered Strength/Dexterity debuff and spuriously create a
/// SECOND instance instead of merging (coverage seed 1593: monster
/// Strength -2 + FyshOil +1 → C# `STRENGTH -1` single, Rust `STRENGTH 1` +
/// `STRENGTH -2`). Reads already use the union (`player_power_uids`); this is
/// the write-side twin. For a non-player `target_uid` only the mirror is
/// consulted (enemies have no legacy surface). Bead `sts2-rs-strmerge`.
fn existing_power_uid_unioned(
    state: &CombatState,
    creature_idx: usize,
    target_uid: CreatureUid,
    kind: PowerKind,
) -> Option<PowerUid> {
    // Perf 2026-07-04: O(1) absence gate. `present_kinds` is a conservative
    // superset of every live power's kind, so a miss proves neither the
    // mirror nor the legacy list can hold an instance of `kind` — the merge
    // scan below would find nothing, and the caller falls through to the
    // fresh-push path exactly as before.
    if !state.powers.present_kinds().contains(kind) {
        return None;
    }
    let has_kind = |uid: PowerUid| {
        state
            .powers
            .get_by_id(uid.0)
            .is_some_and(|p| p.kind == kind)
    };
    if let Some(uid) = state.creatures[creature_idx]
        .powers
        .iter()
        .copied()
        .find(|&uid| has_kind(uid))
    {
        return Some(uid);
    }
    if state.player.creature_uid == Some(target_uid) {
        return state
            .player
            .powers
            .iter()
            .copied()
            .find(|&uid| has_kind(uid));
    }
    None
}

/// [`Creature`]: crate::Creature
///
/// **Artifact debuff-negation (bead `sts2-rs-zdjjx`):** before applying a
/// debuff, [`try_consume_artifact_for_debuff`] checks the *target* creature
/// for an `Artifact` stack; if present it consumes one stack and the debuff
/// is negated entirely (the apply is a no-op). Mirrors C#
/// `ArtifactPower.TryModifyPowerAmountReceived` (zeroes a visible debuff on
/// its owner) + `AfterModifyingPowerAmountReceived` (`PowerCmd.Decrement`),
/// which `PowerCmd.Apply` folds into the `ModifyPowerAmountReceived` channel.
pub fn apply_power_to_creature(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    apply_power_to_creature_inner(
        state, target_uid, kind, amount, /*allow_dead=*/ false, false, None,
    );
}

/// [`apply_power_to_creature`] for a relic's own `PowerCmd.Apply(...,
/// cardSource: null)` that fires while a card is resolving (Hand Drill's
/// Vulnerable on a block break, `HandDrill.cs`): the apply carries no card
/// source, so Unsettling Lamp (`cardSource == null` -> return,
/// `UnsettlingLamp.cs`) neither captures nor doubles it.
pub fn apply_power_to_creature_cardless(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    apply_power_to_creature_inner(
        state, target_uid, kind, amount, /*allow_dead=*/ false, true, None,
    );
}

/// Like [`apply_power_to_creature`] but permits a **dead-but-present** target.
///
/// C# `Creature.CanReceivePowers` is explicitly NOT an alive check — "dead
/// creatures can still have powers applied to them" (Creature.cs:305-316). Use
/// this ONLY for the synchronous killing-blow reaction (MonarchsGaze
/// `AfterDamageGiven` → StrengthDown on the just-killed target), where the
/// corpse is still present in BOTH engines. Do NOT relax the general gate: Rust
/// retains corpses past C#'s post-damage-batch cleanup, so a blanket relaxation
/// over-applies later single-target debuffs (measured: +14 divergent on the
/// Regent full sweep vs a scoped fix's zero). Bead `sts2-rs-monarchs-corpse`.
pub fn apply_power_to_creature_allow_dead(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    apply_power_to_creature_inner(
        state, target_uid, kind, amount, /*allow_dead=*/ true, false, None,
    );
}

/// [`apply_power_to_creature_allow_dead`] for a power hook's own
/// `PowerCmd.Apply(..., cardSource: null)` on the target of a hit (Envenom's
/// Poison, `EnvenomPower.cs:24`): no card source, so Unsettling Lamp neither
/// captures nor doubles it (see [`apply_power_to_creature_cardless`]).
pub fn apply_power_to_creature_cardless_allow_dead(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    apply_power_to_creature_inner(
        state, target_uid, kind, amount, /*allow_dead=*/ true, true, None,
    );
}

/// Whether a dead creature may still receive powers. C#
/// `Creature.CanReceivePowers` asks every combat hook listener
/// `ShouldAllowHitting`, and `DieForYouPower.ShouldAllowHitting` returns
/// `creature.IsAlive` for ANY creature, not just its Osty
/// (`DieForYouPower.cs:35`). So while an Osty carries it, alive or dead, no
/// corpse receives powers: MonarchsGaze's killing-blow StrengthDown never
/// lands (Regent full tier seed 5371, Bound Phylactery; GH #1054 item 5).
pub(crate) fn corpses_can_receive_powers(state: &CombatState) -> bool {
    !state.powers.iter().any(|p| p.kind == PowerKind::DieForYou)
}

/// Whether a corpse was killed and removed from the combat by a nested damage
/// call while an outer one is still resolving: C# `RemoveCreature` cleared its
/// `CombatState`, so `CanReceivePowers` is false (Creature.cs:308-321). See
/// `CombatLog::removed_in_damage_scope`.
fn removed_mid_damage(state: &CombatState, uid: CreatureUid) -> bool {
    let bit = uid.0.get() - 1;
    bit < u16::BITS && state.combat_log.removed_in_damage_scope & (1 << bit) != 0
}

/// Signed power fold onto a **dead-but-present** creature — the `allow_dead`
/// twin of the `Effect::ApplyPower` negative-amount arm, used by MonarchsGaze's
/// `-Strength` leg so it lands on a corpse killed by the same powered attack
/// (see [`apply_power_to_creature_allow_dead`]). Mirrors that arm's mirror+legacy
/// union find: stacks into an existing instance (saturating), drops it at
/// exactly 0, or creates a fresh instance when absent. Scoped to the corpse
/// case; the alive path stays on `Effect::ApplyPower`.
pub fn apply_signed_power_to_creature_allow_dead(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    if amount == 0 || state.is_over {
        return;
    }
    let Some(creature_idx) = state.creatures.iter().position(|c| c.id == target_uid.0) else {
        return;
    };
    if state.creatures[creature_idx].current_hp <= 0
        && (!corpses_can_receive_powers(state) || removed_mid_damage(state, target_uid))
    {
        return;
    }
    let has_kind = |state: &CombatState, u: PowerUid| {
        state.powers.iter().any(|p| p.id == u.0 && p.kind == kind)
    };
    let existing_uid = state.creatures[creature_idx]
        .powers
        .iter()
        .copied()
        .find(|&u| has_kind(state, u))
        .or_else(|| {
            if state.player.creature_uid == Some(target_uid) {
                state
                    .player
                    .powers
                    .iter()
                    .copied()
                    .find(|&u| has_kind(state, u))
            } else {
                None
            }
        });
    if let Some(power_uid) = existing_uid {
        let mut remove_pid = None;
        if let Some(pow) = state.powers.get_mut_by_id(power_uid.0) {
            pow.amount = pow.amount.saturating_add(amount);
            if pow.amount == 0 {
                remove_pid = Some(power_uid.0);
            }
        }
        if let Some(pid) = remove_pid {
            state.powers.retain(|p| p.id != pid);
            state.creatures[creature_idx].powers.retain(|u| u.0 != pid);
            if state.player.creature_uid == Some(target_uid) {
                state.player.powers.retain(|u| u.0 != pid);
            }
        }
    } else {
        let next_id_raw = state
            .powers
            .iter()
            .map(|p| p.id.get())
            .max()
            .map_or(1, |m| m.saturating_add(1));
        let new_id = std::num::NonZeroU32::new(next_id_raw).expect("power id non-zero");
        state.powers.push(crate::Power {
            id: new_id,
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.creatures[creature_idx]
            .powers
            .push(crate::PowerUid(new_id));
    }
}

/// A temporary-Strength debuff's Strength leg, run by
/// [`apply_power_to_creature_inner`] with the post-fold amount — see
/// [`apply_temporary_strength_debuff`].
type TemporaryStrengthLeg = fn(&mut CombatState, CreatureUid, i32);

fn apply_power_to_creature_inner(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    allow_dead: bool,
    cardless: bool,
    strength_leg: Option<TemporaryStrengthLeg>,
) {
    if amount < 0 {
        return;
    }
    // C# `PowerCmd.Apply` no-ops once the combat `IsEnding`
    // (PowerCmd.cs:69-72 / :103-106 v0.107.1, :73 / :107 v0.111.0) while
    // `PowerCmd.Remove` is unguarded — a mid-move power swap on the killing
    // blow drops the old power but never applies the new one (sweep seed
    // 3094: Doormaker's SwapPhasePower removed Scrutiny, Grasp not applied).
    // `IsEnding` holds as soon as the last primary enemy is dead, before the
    // win check sets `is_over` (loadout Regent 5212 v0.107.1: a Royalties+
    // replay after the kill still stacked).
    if state.is_over || crate::combat_loop::enemy_side_is_ending(state) {
        return;
    }
    let can_receive = |state: &CombatState| {
        crate::creature_pos(&state.creatures, target_uid.0).filter(|&i| {
            state.creatures[i].current_hp > 0
                || (allow_dead
                    && corpses_can_receive_powers(state)
                    && !removed_mid_damage(state, target_uid))
        })
    };
    let Some(mut creature_idx) = can_receive(state) else {
        return;
    };
    // `PowerCmd.Apply<T>(0)`: onto no instance it is a no-op (`Apply` returns
    // on `amount == 0`, PowerCmd.cs:103 v0.107.1); onto an existing one it
    // goes through `ModifyAmount(0)` (PowerCmd.cs:84, :228), which runs
    // `Hook.BeforePowerAmountChanged` and the received-modify chain (Artifact
    // eats a 0 debuff too) but no `AfterPowerAmountChanged` (offset 0).
    // Unsettling Lamp captures a 0 Weak from Malaise X=0 there
    // (UnsettlingLamp.cs:71-101; loadout Silent 6886 v0.107.1).
    if amount == 0 {
        if existing_power_uid_unioned(state, creature_idx, target_uid, kind).is_none() {
            return;
        }
        fire_power_amount_observer_channels(
            state,
            None,
            target_uid,
            kind,
            0,
            PowerAmountObserverPhase::Before,
            cardless,
        );
        let _ = try_consume_artifact_for_debuff(state, creature_idx, kind);
        return;
    }

    // Bead `sts2-rs-47jv` + `sts2-rs-7uo4`: C#-canon channel order is
    // `before_power_amount_changed` → modify chain → mutation →
    // `after_modifying_power_amount_*`. The before-channel fires on the
    // PRE-fold amount so listeners (e.g. UnsettlingLamp) can mutate
    // state (set a "captured" flag) that the modify chain then reads.
    // The base `apply_power_to_creature` doesn't carry an applier;
    // listeners that gate on the giver should treat `None` as
    // "primitive caller" and pass through.
    fire_power_amount_observer_channels(
        state,
        None,
        target_uid,
        kind,
        amount,
        PowerAmountObserverPhase::Before,
        cardless,
    );
    // Bead `sts2-rs-zdjjx`: Artifact negates a debuff destined for its owner
    // (and consumes a stack). C# folds ArtifactPower into
    // `ModifyPowerAmountReceived`, which runs AFTER `BeforePowerAmountChanged`
    // (`PowerCmd.cs` Apply/ModifyAmount), so a Before listener (Unsettling
    // Lamp, v0.107.1) still sees — and captures — an Artifact-eaten debuff.
    if try_consume_artifact_for_debuff(state, creature_idx, kind) {
        return;
    }
    let amount =
        pipe_power_amount_through_modify_chain(state, None, target_uid, kind, amount, cardless);
    // C# `PowerCmd.Apply` scales some enemy powers by the player count.
    let amount = crate::coop::scale_enemy_power(state, target_uid, kind, amount);
    if amount <= 0 {
        return;
    }

    // A fresh temporary-Strength debuff runs `BeforeApplied` (its Strength
    // leg) BEFORE it lands, and lands only `if (target.CanReceivePowers)`
    // (PowerCmd.cs:132-135 v0.111.0) — the leg's reactors can kill the target.
    // A stack instead goes through `ModifyAmount`, and the leg follows the
    // change (`TemporaryStrengthPower.AfterPowerAmountChanged`), below.
    if let Some(leg) = strength_leg {
        if existing_power_uid_unioned(state, creature_idx, target_uid, kind).is_none() {
            leg(state, target_uid, amount);
            match can_receive(state) {
                Some(i) => creature_idx = i,
                None => return,
            }
        }
    }

    // Event log (C# `PowerReceivedEntry`). This entry point carries no
    // applier, so the row renders C#'s applier-less phrasing ("X received N
    // POWER") — see the applier-threaded twin in
    // `apply_power_to_creature_with_applier`.
    if crate::event_log::armed(state) {
        let receiver = crate::event_log::LogCreature::of_uid(state, target_uid);
        crate::event_log::record(
            state,
            crate::event_log::EventLogEntry::PowerReceived {
                applier: None,
                receiver,
                power: kind,
                amount,
            },
        );
    }

    // 1. If the creature already has an instance of this kind, stack into it.
    //
    // **Per-instance field stack policy** (bead `sts2-rs-6ynt.6.11`):
    //   * `amount`: saturating-summed (existing behaviour).
    //   * `counter`: **preserved** on the existing instance. Per-instance
    //     scratch (Unmovable's block-gains-this-turn, CrimsonMantle's
    //     self_damage accumulator) is a runtime running total — a new
    //     stack of the same power adds duration/intensity (`amount`)
    //     but does NOT reset the in-flight counter. Powers that need
    //     fresh-stack-resets-counter semantics should override in their
    //     own listener body (none do at this pin).
    //   * `applier`: this surface does NOT carry an applier, so re-application
    //     through here leaves the field unchanged. Grapple-style powers
    //     (bead `sts2-rs-6ynt.6.7`) will route through a dedicated
    //     `apply_power_with_applier` variant that overwrites `applier` on
    //     re-apply (latest-wins, mirroring Python `PowerInstance.applier`
    //     re-binding when an existing stacked power receives a fresh
    //     `apply_power` call).
    //
    // C# `PowerModel.IsInstanced` kinds skip the merge entirely — each
    // apply is a FRESH instance with its own amount/counter (see
    // [`power_kind_is_instanced`]).
    // C# `InstancedPerApplier` kinds merge only into the instance the same
    // player applied (`crate::coop::tracks_applier_seat`).
    let existing_uid = if power_kind_is_instanced(kind) {
        None
    } else if crate::coop::tracks_applier_seat(kind) && crate::coop::has_seats(state) {
        state.creatures[creature_idx]
            .powers
            .iter()
            .copied()
            .find(|&u| {
                state.powers.get_by_id(u.0).is_some_and(|p| p.kind == kind)
                    && crate::coop::applied_by_seated(state, u.0)
            })
    } else {
        existing_power_uid_unioned(state, creature_idx, target_uid, kind)
    };

    if let Some(uid) = existing_uid {
        let mut remove_pid = None;
        if let Some(pow) = state.powers.get_mut_by_id(uid.0) {
            pow.amount = pow.amount.saturating_add(amount);
            // q6bix F-D: C# `PowerCmd.ModifyAmount` runs
            // `ShouldRemoveDueToAmount()` after the change, dropping the
            // power at EXACTLY 0 even for `AllowNegative` kinds like
            // Strength (`PowerModel.cs:413-424`). A positive stack that
            // reverts a temporary debuff (DarkShackles / ShacklingPotion /
            // FlexPotion teardowns apply `+N` Strength onto an existing
            // `-N`) lands at 0 — drop it rather than leave an inert
            // 0-amount instance. Mirrors the negative-stack arm in
            // `apply_effect`'s `ApplyPower` branch.
            if pow.amount == 0 {
                remove_pid = Some(uid.0);
            }
        }
        fire_power_amount_observer_channels(
            state,
            None,
            target_uid,
            kind,
            amount,
            PowerAmountObserverPhase::After,
            cardless,
        );
        fire_after_power_applied(state, target_uid, kind, amount);
        if let Some(pid) = remove_pid {
            state.powers.retain(|p| p.id != pid);
            state.player.powers.retain(|u| u.0 != pid);
            for c in &mut state.creatures {
                c.powers.retain(|u| u.0 != pid);
            }
        }
        if let Some(leg) = strength_leg {
            leg(state, target_uid, amount);
        }
        return;
    }

    // 2. Otherwise allocate a new PowerUid and push.
    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    let applier = enemy_self_attribute_applier(state, target_uid, kind);
    state.powers.push(Power {
        id: new_id,
        kind,
        amount,
        counter: 0,
        applier,
        // Bead `sts2-rs-st9tl`: defer the first decrement tick for
        // Vulnerable / Weak / Frail, but ONLY when the target is the
        // player. Mirrors C# `PowerCmd.Apply` (PowerCmd.cs:104-106):
        // `if (target.Side == CombatSide.Player && power.Type ==
        // PowerType.Debuff) power.SkipNextDurationTick = true;`. The
        // skip exists so a debuff the enemy lands on the player during
        // the enemy turn isn't immediately ticked at that same
        // enemy-turn-end. A player-applied debuff on a MONSTER must NOT
        // skip — it decrements normally at the next enemy-turn-end.
        // (The original `sts2-rs-j0ek` port set this UNCONDITIONALLY,
        // diverging from BOTH references: the Python pin also gates on
        // `target.side == PLAYER` at `combat.py:2038-2043` — j0ek
        // mis-cited the `common.py` *consume* site, not this *set*
        // site — leaving monster V/W/F stuck one tick high vs both.)
        skip_next_tick: arm_skip_next_tick(state, target_uid, kind),
    });
    state.creatures[creature_idx].powers.push(PowerUid(new_id));
    crate::coop::stamp_power_seat(state, new_id, kind);
    fire_power_amount_observer_channels(
        state,
        None,
        target_uid,
        kind,
        amount,
        PowerAmountObserverPhase::After,
        cardless,
    );
    fire_after_power_applied(state, target_uid, kind, amount);
    // Bead `sts2-rs-ig66w`: monster-power AfterApplied dispatch fires
    // only on the FRESH-install path (C# `HungerPower`/`GraspPower` are
    // `PowerStackType.Single`, so this matches the C# semantics — the
    // on-applied "walk every player card" body runs once at install).
    let _ = crate::monster_power_effects::dispatch_after_applied_for_power(
        state, target_uid, kind, None,
    );
}

/// The C# `Applier` a fresh Strength / Dexterity / Focus instance gets when
/// it lands on an ENEMY through the applier-less positive surface
/// ([`apply_power_to_creature`]): the enemy itself.
///
/// A positive attribute on an enemy is a monster buff — Rocket's
/// `PowerCmd.Apply<StrengthPower>(base.Creature, 2, base.Creature, null)`,
/// Crusher's Adapt, CrabRage (`base.Owner, ..., base.Owner`) — and C# keeps
/// the creating applier for the instance's lifetime (`PowerCmd.Apply` sets
/// `Applier` only on a fresh instance, PowerCmd.cs:113-119; a stack goes
/// through `ModifyAmount`). The instance can later read as a Debuff once a
/// player drain takes it negative, and Misery then copies it with that
/// monster applier, so Sleight of Flesh (`applier == Owner`) does not react.
/// Rust left it `None`, which Misery reads as player-applied: loadout
/// Necrobinder 1210-v0.107.1, Misery on Rocket (Strength 2 -> -7 under Dark
/// Shackles) fired Sleight of Flesh on the copied Strength too (C# 37 -> 28
/// -> 22, Rust 37 -> 13). A player card that grants an enemy Strength (Fight
/// Me) restamps its fresh instance.
fn enemy_self_attribute_applier(
    state: &CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
) -> Option<CreatureUid> {
    (matches!(
        kind,
        PowerKind::Strength | PowerKind::Dexterity | PowerKind::Focus
    ) && state.enemy_uids.contains(&target_uid))
    .then_some(target_uid)
}

/// C# `PowerCmd.Apply` of a temporary-Strength DEBUFF (a
/// `TemporaryStrengthPower` with `IsPositive => false`: CrushUnder,
/// DarkShackles, DyingStar, EnfeeblingTouch, ManglePower, PiercingWail,
/// ShacklingPotion, ...) to `target_uid`, Strength leg included.
///
/// The leg is `Strength -amount` with the MODIFIED amount (after Unsettling
/// Lamp's x2 / Artifact's 0; the Lamp's `HasDoubledTemporaryPowerSource`
/// keeps the leg itself from doubling again). Order follows C#:
/// * a fresh power: `BeforeApplied` drains the Strength first, then the power
///   lands `if (target.CanReceivePowers)` (PowerCmd.cs:132-135 v0.111.0,
///   TemporaryStrengthPower.cs:134-136). The drain fires Sleight of Flesh,
///   which can kill the target first: a dying Parafright (reviving
///   IllusionPower, `ShouldAllowHitting` false) ends with the Strength but
///   without the power (loadout Necrobinder 1189 v0.111.0: Shackling Potion,
///   then Dark Shackles);
/// * a stack: `ModifyAmount`, then `AfterPowerAmountChanged` drains
///   (TemporaryStrengthPower.cs:139-145).
///
/// The per-card ports applied the power and then drained what landed, which
/// gets the amount right but not the order.
pub fn apply_temporary_strength_debuff(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    debug_assert!(
        crate::afflictions::is_temporary_power_debuff(kind),
        "apply_temporary_strength_debuff is for TemporaryStrengthPower debuffs"
    );
    apply_power_to_creature_inner(
        state,
        target_uid,
        kind,
        amount,
        false,
        false,
        Some(temporary_strength_leg),
    );
}

/// [`apply_temporary_strength_debuff`] for a power hook's own
/// `PowerCmd.Apply(..., cardSource: null)`: Monarch's Gaze's
/// `Apply<MonarchsGazeStrengthDownPower>(target, Amount, Owner, null)`
/// (MonarchsGazePower.cs:17). No card source, so Unsettling Lamp neither
/// captures nor doubles it, though a card's attack is resolving.
pub fn apply_temporary_strength_debuff_cardless(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    debug_assert!(
        crate::afflictions::is_temporary_power_debuff(kind),
        "apply_temporary_strength_debuff_cardless is for TemporaryStrengthPower debuffs"
    );
    apply_power_to_creature_inner(
        state,
        target_uid,
        kind,
        amount,
        false,
        true,
        Some(temporary_strength_leg),
    );
}

/// `TemporaryStrengthPower`'s `PowerCmd.Apply<StrengthPower>(target,
/// -amount, applier, cardSource, silent: true)`: a silent signed drain whose
/// `AfterPowerAmountChanged` reactors (Sleight of Flesh) still fire.
fn temporary_strength_leg(state: &mut CombatState, target_uid: CreatureUid, amount: i32) {
    signed_power_apply_with_reactors(state, target_uid, PowerKind::Strength, -amount, |s| {
        apply_signed_attr_power(s, target_uid, PowerKind::Strength, -amount);
    });
}

/// Apply a **signed** delta to one of the `AllowNegative` attribute powers
/// (Strength / Dexterity / Focus) on `target_uid`, creating a fresh
/// (possibly negative) instance when none exists.
///
/// This fills a gap the two canonical apply paths leave open:
///   * [`apply_power_to_creature`] early-returns on `amount <= 0`.
///   * [`crate::hooks::Effect::ApplyPower`]'s negative arm only stacks into
///     an **existing** instance (its doc-comment notes the negative-only
///     create path is "unreachable in normal play").
///
/// C# `PowerCmd.Apply<StrengthPower>(owner, -1m)` lands even from a 0/absent
/// base because Strength/Dexterity/Focus are `AllowNegative` — the apply
/// creates a `-1` instance. `TenderPower` (HunterKiller's TENDERIZING_GOOP
/// debuff) needs exactly that: every player card played this turn drops
/// Strength **and** Dexterity by 1, regardless of the (usually 0) base.
///
/// Consistent with the negative-`ApplyPower` convention, negative deltas
/// skip the `modify_power_amount_*` chain and the observer/`AfterApplied`
/// channels (C# applies these with `silent: true`); the positive restore at
/// turn end routes through [`crate::hooks::Effect::ApplyPower`] as usual. An
/// instance landing at exactly 0 is removed (C# `ShouldRemoveDueToAmount`).
/// Bead `sts2-rs-ybtmu` (Group C — HunterKiller Tender).
pub fn apply_signed_attr_power(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    delta: i32,
) {
    debug_assert!(
        matches!(
            kind,
            PowerKind::Strength | PowerKind::Dexterity | PowerKind::Focus
        ),
        "apply_signed_attr_power is only for AllowNegative attribute powers"
    );
    if delta == 0 {
        return;
    }
    let Some(creature_idx) = state
        .creatures
        .iter()
        .position(|c| c.id == target_uid.0 && c.current_hp > 0)
    else {
        return;
    };

    // Stack into an existing instance when present (matches the existing /
    // negative-`ApplyPower` stack policy: saturating-add, drop at exactly 0).
    let existing_uid = existing_power_uid_unioned(state, creature_idx, target_uid, kind);
    if let Some(uid) = existing_uid {
        let mut remove_pid = None;
        if let Some(pow) = state.powers.get_mut_by_id(uid.0) {
            pow.amount = pow.amount.saturating_add(delta);
            if pow.amount == 0 {
                remove_pid = Some(uid.0);
            }
        }
        if let Some(pid) = remove_pid {
            state.powers.retain(|p| p.id != pid);
            state.player.powers.retain(|u| u.0 != pid);
            for c in &mut state.creatures {
                c.powers.retain(|u| u.0 != pid);
            }
        }
        return;
    }

    // No instance: allocate a fresh one carrying the (possibly negative)
    // delta. Mirrors the new-power branch of `apply_power_to_creature` minus
    // the modify chain / observer channels (silent, per above).
    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    state.powers.push(Power {
        id: new_id,
        kind,
        amount: delta,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.creatures[creature_idx].powers.push(PowerUid(new_id));
}

/// The amount a player CARD's signed drain on `target_uid` actually applies —
/// C#'s `PowerCmd.Apply` front half (`BeforePowerAmountChanged`, then
/// `ModifyPowerAmountGiven`, PowerCmd.cs:120-125) for an apply the signed
/// helpers would otherwise skip. Call it before the signed apply and pass the
/// result on.
///
/// Shared Fate's enemy `Strength -2`, Malaise's `Strength -X` and Resonance's
/// `Strength -1` are ordinary `PowerCmd.Apply<StrengthPower>(target, -N,
/// owner, this)` calls, and a negative amount of an `AllowNegative` power
/// reads as a Debuff (`PowerModel.GetTypeForAmount`). So Unsettling Lamp
/// captures and doubles them like any card debuff (loadout Necrobinder 1101
/// v0.111.0: Shared Fate took a Mecha Knight's Strength 10 -> 6 in C#, 10 -> 8
/// in Rust, which never showed the drain to the Lamp).
///
/// Only for card-level drains: the signed helpers also carry
/// `TemporaryStrengthPower.BeforeApplied`'s inner `Strength -N`, which must
/// stay unfolded, because it already drains the landed (doubled) amount and
/// the Lamp's `HasDoubledTemporaryPowerSource` skips it. Artifact
/// (`ModifyPowerAmountReceived`) stays with the signed helper.
pub fn fold_player_card_signed_apply(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) -> i32 {
    if amount == 0 || state.is_over {
        return amount;
    }
    // C# `PowerCmd.Apply` returns before any hook when the target cannot
    // receive powers; the signed helpers skip dead targets the same way.
    if !state
        .creatures
        .iter()
        .any(|c| c.id == target_uid.0 && c.current_hp > 0)
    {
        return amount;
    }
    let applier = state.player.creature_uid;
    fire_power_amount_observer_channels(
        state,
        applier,
        target_uid,
        kind,
        amount,
        PowerAmountObserverPhase::Before,
        false,
    );
    if !relic_guarded_channel_live(state, GuardedChannel::ModifyPowerAmountGiven) {
        return amount;
    }
    let ctx = hooks::PowerAmountCtx {
        source_card: source_card(state, false),
        ..hooks::PowerAmountCtx::new(
            applier.map(|uid| creature_ref_for(state, uid)),
            creature_ref_for(state, target_uid),
            kind,
            amount,
        )
    };
    crate::relic_effects::dispatch_modify_power_amount_given(state, &ctx)
}

/// Run a content-side SIGNED power apply (`raw`, a player-applied drain such
/// as Shared Fate's enemy `Strength -2`) and, when it actually changed the
/// target's `kind` total (not negated by Artifact, target alive), fire the
/// player-applied reactors the positive-only pipeline skips — see
/// [`hooks::dispatch_after_signed_power_applied_by_player`]. C# fires
/// `Hook.AfterPowerAmountChanged` for every non-zero apply, silent or not
/// (PowerCmd.cs:152-156, 247-250).
pub fn signed_power_apply_with_reactors(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    delta: i32,
    raw: impl FnOnce(&mut CombatState),
) {
    let total = |s: &CombatState| -> i32 {
        s.creatures
            .iter()
            .find(|c| c.id == target_uid.0)
            .map_or(0, |c| {
                c.powers
                    .iter()
                    .filter_map(|u| s.powers.get_by_id(u.0))
                    .filter(|p| p.kind == kind)
                    .map(|p| p.amount)
                    .fold(0_i32, i32::saturating_add)
            })
    };
    let before = total(state);
    raw(state);
    if total(state) != before {
        hooks::dispatch_after_signed_power_applied_by_player(state, target_uid, kind, delta);
    }
}

/// Resolve `target_uid` to a [`CreatureRef`] for the hooks family —
/// `Player` if the uid matches the player's mirror Creature, else
/// `Enemy(uid)`. Helper for the apply-power pipeline's hook ctx
/// construction.
fn creature_ref_for(state: &CombatState, target_uid: CreatureUid) -> hooks::CreatureRef {
    if state.player.creature_uid.map(|u| u.0) == Some(target_uid.0) {
        hooks::CreatureRef::Player
    } else {
        hooks::CreatureRef::Enemy(target_uid)
    }
}

/// True when `target_uid` resolves to the player's mirror Creature.
/// Gates the `skip_next_tick` arm in the apply-power pipeline to match
/// C# `PowerCmd.Apply`'s `target.Side == CombatSide.Player` guard
/// (PowerCmd.cs:104). Bead `sts2-rs-st9tl`.
fn target_is_player(state: &CombatState, target_uid: CreatureUid) -> bool {
    state.player.creature_uid.map(|u| u.0) == Some(target_uid.0)
}

/// Whether a freshly-applied power should arm [`Power::skip_next_tick`]
/// (defer its first owner-turn-end tick). One field folds two distinct
/// C# skip mechanisms:
///   * **V/W/F on the PLAYER** — C# `PowerCmd.Apply` sets
///     `SkipNextDurationTick` when `target.Side == Player && Type ==
///     Debuff` (PowerCmd.cs:104) so a debuff the enemy lands during its
///     turn isn't ticked at that same enemy-turn-end. Bead `sts2-rs-st9tl`.
///   * **Ritual on an ENEMY** — C# `RitualPower.AfterApplied` sets
///     `WasJustAppliedByEnemy = true` when `Owner.IsEnemy`
///     (RitualPower.cs:35-41); `AfterTurnEnd` then skips the first
///     owner-turn-end Strength grant (RitualPower.cs:44-56). Bead
///     `sts2-rs-ybtmu` — DevotedSculptor applies Ritual(9) to itself and
///     Rust granted Strength(9) on the same turn (the prior port assumed
///     "no enemy-applied-Ritual path"). Player-applied Ritual
///     (MazalethsGift potion) is NOT armed — it grants Strength at the
///     first player-turn-end, matching C#'s non-enemy owner.
fn arm_skip_next_tick(state: &CombatState, target_uid: CreatureUid, kind: PowerKind) -> bool {
    let is_player = target_is_player(state, target_uid);
    (is_player
        && matches!(
            kind,
            PowerKind::Vulnerable | PowerKind::Weak | PowerKind::Frail
        ))
        || (!is_player && kind == PowerKind::Ritual)
}

/// Bead `sts2-rs-47jv`: walk the C#-canon `modify_power_amount_given`
/// + `_received` chain on the relic-side dispatch. Returns the final
/// amount after both folds. Called by both `apply_power_to_creature`
/// flavors before the actual stack/push.
fn pipe_power_amount_through_modify_chain(
    state: &CombatState,
    applier: Option<CreatureUid>,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    cardless: bool,
) -> i32 {
    // J03: with both relic folds dead (no owned handler) both return
    // `amount`; skip building their context.
    let live = relic_guarded_channel_live(state, GuardedChannel::ModifyPowerAmountGiven)
        || relic_guarded_channel_live(state, GuardedChannel::ModifyPowerAmountReceived);
    fold_if_live(state, live, amount, |state| {
        let target_ref = creature_ref_for(state, target_uid);
        let applier_ref = applier.map(|uid| creature_ref_for(state, uid));
        let ctx = hooks::PowerAmountCtx {
            source_card: source_card(state, cardless),
            ..hooks::PowerAmountCtx::new(applier_ref, target_ref, kind, amount)
        };
        let amt = crate::relic_effects::dispatch_modify_power_amount_given(state, &ctx);
        let ctx = hooks::PowerAmountCtx { amount: amt, ..ctx };
        crate::relic_effects::dispatch_modify_power_amount_received(state, &ctx)
    })
}

/// The card this power application is coming from, if any — C#'s
/// `CardModel? cardSource` argument on `BeforePowerAmountChanged` /
/// `ModifyPowerAmountGivenMultiplicative`.
///
/// The play pile holds the card currently resolving (`player.play`; nested
/// plays push, so the last entry is the innermost, which is the one whose
/// effect is running). `None` when the pile is empty, which is exactly the
/// non-card surfaces C# passes `null` for: potions, power ticks, relic
/// effects, enemy moves.
///
/// GH #365: `PowerAmountCtx.source_card` existed and its doc comment even
/// named UnsettlingLamp, but both production sites built via
/// `PowerAmountCtx::new`, which hardcodes `None` — so the field was dead and
/// every gate reading it silently passed.
fn resolving_card(state: &CombatState) -> Option<CardUid> {
    resolving_card_uid(state)
}

/// The `cardSource` of one power application: [`resolving_card`], or `None`
/// for a `cardless` apply — a power hook's own `PowerCmd.Apply(..., null)`
/// that runs while a card is resolving (Reaper Form's Doom,
/// `ReaperFormPower.cs:57-64`), which C#'s `cardSource == null` gates
/// (Unsettling Lamp) must not see as card-sourced.
fn source_card(state: &CombatState, cardless: bool) -> Option<CardUid> {
    if cardless {
        None
    } else {
        resolving_card(state)
    }
}

/// Bead `sts2-rs-47jv`: fire the observer channels (before-changed +
/// after-modifying-given + after-modifying-received) around the
/// power-amount mutation. `before_power_amount_changed` runs before
/// the caller mutates the power; the two `after_modifying_*` channels
/// run after. The amount passed in is the final post-fold amount.
fn fire_power_amount_observer_channels(
    state: &mut CombatState,
    applier: Option<CreatureUid>,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    when: PowerAmountObserverPhase,
    cardless: bool,
) {
    // J03: the observers' answers are discarded, so channels with no owned
    // handler need neither the context nor the dispatch.
    let live = match when {
        PowerAmountObserverPhase::Before => {
            relic_guarded_channel_live(state, GuardedChannel::BeforePowerAmountChanged)
        }
        PowerAmountObserverPhase::After => {
            relic_guarded_channel_live(state, GuardedChannel::AfterModifyingPowerAmountGiven)
                || relic_guarded_channel_live(
                    state,
                    GuardedChannel::AfterModifyingPowerAmountReceived,
                )
        }
    };
    dispatch_if_live(state, live, (), |state| {
        let target_ref = creature_ref_for(state, target_uid);
        let applier_ref = applier.map(|uid| creature_ref_for(state, uid));
        let ctx = hooks::PowerAmountCtx {
            source_card: source_card(state, cardless),
            ..hooks::PowerAmountCtx::new(applier_ref, target_ref, kind, amount)
        };
        match when {
            PowerAmountObserverPhase::Before => {
                let _ = crate::relic_effects::dispatch_before_power_amount_changed(state, &ctx);
            }
            PowerAmountObserverPhase::After => {
                let _ =
                    crate::relic_effects::dispatch_after_modifying_power_amount_given(state, &ctx);
                let _ = crate::relic_effects::dispatch_after_modifying_power_amount_received(
                    state, &ctx,
                );
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
enum PowerAmountObserverPhase {
    Before,
    After,
}

fn fire_after_power_applied(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    delta: i32,
) {
    let target_ref = if state.player.creature_uid.map(|u| u.0) == Some(target_uid.0) {
        hooks::CreatureRef::Player
    } else {
        hooks::CreatureRef::Enemy(target_uid)
    };
    let ctx = hooks::PowerAppliedCtx {
        target: target_ref,
        power_kind: kind,
        delta,
        applier: Some(hooks::CreatureRef::Player),
    };
    hooks::dispatch_after_power_applied(state, &ctx);
}

/// Same as [`fire_after_power_applied`] but threads the REAL applier through
/// (resolved to a [`hooks::CreatureRef`]) instead of hardcoding the player.
/// Used by [`apply_power_to_creature_with_applier`] so the player-applied
/// reactors in [`hooks::dispatch_after_power_applied`] (Shroud's block-on-Doom,
/// Vicious's draw-on-Vulnerable, SleightOfFlesh's damage-on-debuff) fire for
/// powers routed through the applier surface — e.g. `Effect::ApplyPower`
/// carrying an applier, which NEUROSURGE uses to self-Doom the player at turn
/// start. The base helper fired this for `apply_power_to_creature`, but the
/// applier surface previously only fired the relic observer channels, so the
/// player reactors were silently skipped (Necrobinder fuzz seeds 192/340/410/
/// 495: SHROUD failed to grant block off NEUROSURGE's self-Doom — C# block 2,
/// Rust 0). The dispatch gates on `applier == Player`, so an enemy-applied
/// power correctly fires nothing.
fn fire_after_power_applied_with(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    delta: i32,
    applier: Option<CreatureUid>,
) {
    let target_ref = if state.player.creature_uid.map(|u| u.0) == Some(target_uid.0) {
        hooks::CreatureRef::Player
    } else {
        hooks::CreatureRef::Enemy(target_uid)
    };
    let ctx = hooks::PowerAppliedCtx {
        target: target_ref,
        power_kind: kind,
        delta,
        applier: applier.map(|uid| creature_ref_for(state, uid)),
    };
    hooks::dispatch_after_power_applied(state, &ctx);
}

/// Apply a power to `target_uid` while recording the `applier` on the
/// resulting [`Power`] instance. Mirrors [`apply_power_to_creature`]
/// exactly except that the `applier` field is set on the new instance
/// (or **overwritten latest-wins** when stacking into an existing
/// instance of the same `kind`). No-op when `amount <= 0` or the
/// target is dead/missing — matches the base helper's contract.
///
/// Bead `sts2-rs-6ynt.6.7`: dedicated apply-with-applier surface so
/// `GrapplePower`'s `after_block_gained` listener can gate on
/// `creature == self.applier` (Python pin
/// `sts2-rl-agent@01c094d:sts2_env/powers/remaining_a.py:1450-1485`).
///
/// [`Power`]: crate::Power
pub fn apply_power_to_creature_with_applier(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    applier: Option<CreatureUid>,
) {
    apply_power_to_creature_with_applier_inner(
        state, target_uid, kind, amount, applier, false, false,
    );
}

/// [`apply_power_to_creature_with_applier`] that also lands on a
/// **dead-but-present** target. Same scope rule as
/// [`apply_power_to_creature_allow_dead`]: only for a synchronous
/// `AfterDamageGiven` reaction on the creature the same hit just killed
/// (C# fires `AfterDamageGiven` before `Kill`, CreatureCmd.cs:390-409, and
/// `CanReceivePowers` is not an alive check, Creature.cs:308-321).
pub fn apply_power_to_creature_with_applier_allow_dead(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    applier: Option<CreatureUid>,
) {
    apply_power_to_creature_with_applier_inner(
        state, target_uid, kind, amount, applier, true, false,
    );
}

/// [`apply_power_to_creature_with_applier_allow_dead`] for a power hook's
/// own `PowerCmd.Apply(..., cardSource: null)` that fires while a card is
/// resolving (Reaper Form's Doom, `ReaperFormPower.cs:57-64`): the apply's
/// `cardSource` is `None`, not the resolving card, so Unsettling Lamp
/// (`cardSource == null` → return) neither captures nor doubles it.
pub fn apply_power_to_creature_cardless_with_applier_allow_dead(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    applier: Option<CreatureUid>,
) {
    apply_power_to_creature_with_applier_inner(
        state, target_uid, kind, amount, applier, true, true,
    );
}

fn apply_power_to_creature_with_applier_inner(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
    applier: Option<CreatureUid>,
    allow_dead: bool,
    cardless: bool,
) {
    if amount <= 0 {
        return;
    }
    // C# `PowerCmd.Apply` no-ops once the combat `IsEnding`; see
    // `apply_power_to_creature_inner`.
    if state.is_over || crate::combat_loop::enemy_side_is_ending(state) {
        return;
    }
    // A corpse receives the power only while no `DieForYou` is in the fight
    // ([`corpses_can_receive_powers`]): with an Osty summoned, Reaper Form's
    // killing-blow Doom never lands, so Shroud grants no block for it
    // (loadout Necrobinder 1073/1078/1092/1114/1122/1035/1109).
    let allow_dead =
        allow_dead && corpses_can_receive_powers(state) && !removed_mid_damage(state, target_uid);
    let Some(creature_idx) = state
        .creatures
        .iter()
        .position(|c| c.id == target_uid.0 && (allow_dead || c.current_hp > 0))
    else {
        return;
    };

    // Bead `sts2-rs-47jv` + `sts2-rs-7uo4`: C#-canon order is
    // Before → Modify → mutation → After, with the real applier
    // threaded through ctx for listeners that gate on it.
    fire_power_amount_observer_channels(
        state,
        applier,
        target_uid,
        kind,
        amount,
        PowerAmountObserverPhase::Before,
        cardless,
    );
    // Bead `sts2-rs-zdjjx`: Artifact debuff-negation gate, AFTER the Before
    // channel as in C# (see `apply_power_to_creature_inner`). Scoped to the
    // target creature, so a player-applied debuff onto an Artifact-bearing
    // monster is consumed here.
    if try_consume_artifact_for_debuff(state, creature_idx, kind) {
        return;
    }
    let amount =
        pipe_power_amount_through_modify_chain(state, applier, target_uid, kind, amount, cardless);
    // C# `PowerCmd.Apply` scales some enemy powers by the player count.
    let amount = crate::coop::scale_enemy_power(state, target_uid, kind, amount);
    if amount <= 0 {
        return;
    }

    // Event log (C# `PowerReceivedEntry`) — recorded exactly where C# records
    // it, past every gate the comment below enumerates, on the post-modify
    // amount.
    if crate::event_log::armed(state) {
        let receiver = crate::event_log::LogCreature::of_uid(state, target_uid);
        let by = applier.map(|a| crate::event_log::LogCreature::of_uid(state, a));
        crate::event_log::record(
            state,
            crate::event_log::EventLogEntry::PowerReceived {
                applier: by,
                receiver,
                power: kind,
                amount,
            },
        );
    }

    // Necrobinder `DeathsDoor.cs` `WasDoomAppliedThisTurn` scans for a
    // `PowerReceivedEntry` with `Power is DoomPower && Applier == Owner.Creature`
    // (the player). By this point every C# "entry recorded" gate has passed:
    // combat not over (`:1219`), target alive (`:1222`), Artifact not consumed
    // (`:1233`), post-modify amount > 0 (`:1249`). Arm the 3x-block flag only
    // for a PLAYER-applied Doom — enemy-applied Doom must NOT arm it. This
    // covers the applier-threaded grants (Neurosurge/Countdown, and now
    // Oblivion/ReaperForm/PotionOfDoom); the card-play enemy path keeps its
    // own flag logic in `apply_power_to_enemy` (idempotent — flag is a bool).
    if kind == PowerKind::Doom && applier.is_some() && applier == state.player.creature_uid {
        state.combat_log.doom_applied_by_player_this_turn = true;
    }

    // Stack policy: same as `apply_power_to_creature` for `amount` /
    // `counter`; `applier` is **latest-wins** — re-application overwrites
    // the stored applier reference (mirrors Python's
    // `PowerInstance.applier` re-binding when an already-stacked power
    // receives a fresh `apply_power` call).
    // Perf 2026-07-04: same O(1) absence gate as
    // `existing_power_uid_unioned` — a `present_kinds` miss proves the
    // merge scan finds nothing.
    let existing_uid = if state.powers.present_kinds().contains(kind) {
        state.creatures[creature_idx]
            .powers
            .iter()
            .copied()
            .find(|uid| {
                state
                    .powers
                    .get_by_id(uid.0)
                    .is_some_and(|p| p.kind == kind)
            })
    } else {
        None
    };
    if let Some(uid) = existing_uid {
        let mut remove_pid = None;
        if let Some(pow) = state.powers.get_mut_by_id(uid.0) {
            pow.amount = pow.amount.saturating_add(amount);
            pow.applier = applier;
            // q6bix F-D: remove-at-0 mirror (see `apply_power_to_creature`).
            if pow.amount == 0 {
                remove_pid = Some(uid.0);
            }
        }
        fire_power_amount_observer_channels(
            state,
            applier,
            target_uid,
            kind,
            amount,
            PowerAmountObserverPhase::After,
            cardless,
        );
        fire_after_power_applied_with(state, target_uid, kind, amount, applier);
        if let Some(pid) = remove_pid {
            state.powers.retain(|p| p.id != pid);
            state.player.powers.retain(|u| u.0 != pid);
            for c in &mut state.creatures {
                c.powers.retain(|u| u.0 != pid);
            }
        }
        return;
    }

    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    state.powers.push(Power {
        id: new_id,
        kind,
        amount,
        counter: 0,
        applier,
        // Beads `sts2-rs-st9tl` (player V/W/F) + `sts2-rs-ybtmu` (enemy
        // Ritual): same skip-first-tick semantics as
        // [`apply_power_to_creature`]. See [`arm_skip_next_tick`].
        skip_next_tick: arm_skip_next_tick(state, target_uid, kind),
    });
    state.creatures[creature_idx].powers.push(PowerUid(new_id));
    fire_power_amount_observer_channels(
        state,
        applier,
        target_uid,
        kind,
        amount,
        PowerAmountObserverPhase::After,
        cardless,
    );
    fire_after_power_applied_with(state, target_uid, kind, amount, applier);
    // Bead `sts2-rs-ig66w`: parallel AfterApplied dispatch for the
    // `_with_applier` surface. Threads the applier through to the
    // listener so future Hunger-shaped powers can gate on the giver.
    let _ = crate::monster_power_effects::dispatch_after_applied_for_power(
        state, target_uid, kind, applier,
    );
}

/// C# `PowerModel.ShouldOwnerDeathTriggerFatal()` folded over every power
/// on `target` — `cardPlay.Target.Powers.All(p => p.ShouldOwnerDeathTriggerFatal())`,
/// the pre-damage snapshot Feed / HandOfGreed / TheHunt each take before
/// their `DamageCmd.Attack(...).Execute(...)` (`Feed.cs:38`,
/// `HandOfGreed.cs:49`, `TheHunt.cs:41`).
///
/// The base predicate is `true` (`PowerModel.cs:648`) and exactly two powers
/// override it in v0.107.1:
///
/// - `MinionPower.cs:20` → **always** `false`. Killing a summoned minion is
///   never a "fatal" for on-kill card riders.
/// - `ReattachPower.cs:106` → `AreAllOtherSegmentsDead()`
///   (`ReattachPower.cs:156-159`): `GetTeammatesOf(Owner).Except([Owner])
///   .Where(c => c.HasPower<ReattachPower>()).All(s => s.IsDead)`. A
///   Decimillipede segment killed while a sibling still stands is NOT fatal
///   (it will REATTACH); the killing blow on the LAST standing segment IS.
///
/// The Rust ports previously treated `Reattach` as unconditionally
/// death-blocking ("the safe direction"), which silently swallowed the
/// Feed max-HP / HandOfGreed gold / TheHunt reward on the one hit that
/// actually ends the Decimillipede elite. This models the real condition.
///
/// Note `IllusionPower` / `AdaptablePower` do NOT override this predicate —
/// they gate corpse retention via `ShouldCreatureBeRemovedFromCombatAfterDeath`,
/// a distinct method — so a reviving illusion still triggers fatal.
#[must_use]
pub fn should_owner_death_trigger_fatal(state: &CombatState, target: CreatureUid) -> bool {
    fatal_with(state, target, |_| false)
}

/// [`should_owner_death_trigger_fatal`] read the way C# `Hook.AfterDiedToDoom`
/// sees it: once the whole `DoomPower.DoomKill` batch has died
/// (`DoomPower.cs:31-43`). Rust kills a batch one victim at a time and fires
/// the bus per victim, so a sibling still standing but doomed (`Doom >= HP`)
/// dies in this same batch and counts as dead here.
#[must_use]
pub fn should_owner_death_trigger_fatal_after_doom_batch(
    state: &CombatState,
    target: CreatureUid,
) -> bool {
    fatal_with(state, target, |c| {
        c.powers.iter().any(|pu| {
            state
                .powers
                .get_by_id(pu.0)
                .is_some_and(|p| p.kind == PowerKind::Doom && p.amount >= c.current_hp)
        })
    })
}

fn fatal_with(
    state: &CombatState,
    target: CreatureUid,
    dies_with_it: impl Fn(&crate::Creature) -> bool,
) -> bool {
    let Some(creature) = state.creatures.iter().find(|c| c.id == target.0) else {
        return true;
    };
    creature.powers.iter().all(|pu| {
        state.powers.get_by_id(pu.0).is_none_or(|p| match p.kind {
            PowerKind::Minion => false,
            PowerKind::Reattach => all_other_segments_dead(state, target, &dies_with_it),
            _ => true,
        })
    })
}

/// C# `ReattachPower.AreAllOtherSegmentsDead` (`ReattachPower.cs:149-159`):
/// every OTHER teammate carrying a `ReattachPower` is dead. A corpse keeps
/// its Reattach (`ShouldPowerBeRemovedAfterOwnerDeath` → `false`), so the
/// scan is over live-and-dead segments alike and only `IsDead` decides.
fn all_other_segments_dead(
    state: &CombatState,
    owner: CreatureUid,
    dies_with_it: &impl Fn(&crate::Creature) -> bool,
) -> bool {
    !state.enemy_uids.iter().any(|uid| {
        *uid != owner
            && state.creatures.iter().any(|c| {
                c.id == uid.0
                    && c.current_hp > 0
                    && !dies_with_it(c)
                    && c.powers.iter().any(|pu| {
                        state
                            .powers
                            .get_by_id(pu.0)
                            .is_some_and(|p| p.kind == PowerKind::Reattach)
                    })
            })
    })
}

/// Enemy-targeted alias for [`apply_power_to_creature`]. Existing
/// Phase-6 card handlers route through this surface; new code should
/// prefer the creature-keyed primitive directly. The enemy alive-check
/// and amount-guard are folded into [`apply_power_to_creature`].
///
/// Phase-6 scope: `Vulnerable` / `Weak` / `Poison` are the load-bearing
/// debuffs exercised by ported cards; the helper is generic over
/// `PowerKind` so downstream beads can route through the same path
/// without growing a dedicated apply-per-power surface.
pub fn apply_power_to_enemy(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    // Necrobinder `DeathsDoor.cs` `WasDoomAppliedThisTurn` scans the combat
    // history for an actually-RECEIVED Doom `PowerReceivedEntry`. Only flip the
    // flag when the apply truly lands — combat not over AND the target still
    // alive (the same gates `apply_power_to_creature` enforces below). A Doom
    // applied to an enemy the attack just KILLED (e.g. BlightStrike's
    // damage-equal Doom on a lethal hit) records no C# entry, so it must NOT
    // arm DeathsDoor's 3x block (parity fuzz: player.block cs=11 rs=23 — Rust
    // wrongly gained 3x off a kill-shot Doom). `amount > 0` mirrors a real
    // entry (a 0-stack apply records nothing). Bead `sts2-rs-deaths-door-doom`.
    //
    // Also exclude an Artifact-negated apply: Doom is a debuff
    // (`afflictions::is_affliction`), so `apply_power_to_creature` below
    // consumes one Artifact stack and drops it — no `PowerReceivedEntry` in C#
    // (`ArtifactPower.TryModifyPowerAmountReceived`), so `WasDoomAppliedThisTurn`
    // stays false. Peek (do NOT consume here — the apply consumes it) for an
    // Artifact stack on the live target (parity fuzz seed 492: a Doom onto a
    // CUBEX_CONSTRUCT's Artifact 1 wrongly armed the 3x block, cs=11 rs=23).
    // Pure read — evaluated lazily inside the `doom_landed` conjunction below
    // (short-circuits on the cheap `kind == Doom` gate first) so the common
    // non-Doom apply skips the creature/power walk entirely.
    let target_has_artifact = |state: &CombatState| {
        state.creatures.iter().any(|c| {
            c.id == target_uid.0
                && c.current_hp > 0
                && c.powers.iter().any(|uid| {
                    state
                        .powers
                        .get_by_id(uid.0)
                        .is_some_and(|p| p.kind == PowerKind::Artifact && p.amount > 0)
                })
        })
    };
    let doom_landed = kind == PowerKind::Doom
        && amount > 0
        && !state.is_over
        && !target_has_artifact(state)
        && state
            .creatures
            .iter()
            .any(|c| c.id == target_uid.0 && c.current_hp > 0);
    if doom_landed {
        state.combat_log.doom_applied_by_player_this_turn = true;
    }
    apply_power_to_creature(state, target_uid, kind, amount);
}

/// Set / overwrite a `combat_vars` slot on the card identified by `card_uid`.
///
/// Bead sts2-rs-pmf (P6-1b): pool-bead handlers (Rampage, Claw, Genetic
/// Algorithm, ...) reach for `state.cards[idx].set_combat_var(...)` enough
/// times that hand-inlining the UID -> index translation is a paper-cut
/// source of off-by-one bugs. This thin wrapper centralises the lookup
/// (UID's `NonZeroU32` is 1-based against `CombatState::cards`) so pool
/// authors never write the index math themselves.
///
/// No-op if `card_uid` is out of bounds.
pub fn set_card_var(state: &mut CombatState, card_uid: CardUid, key: VarKey, value: i32) {
    let idx = card_uid.0.get() as usize - 1;
    if let Some(card) = state.cards.get_mut(idx) {
        card.set_combat_var(key, value);
    }
}

/// C# `CardModel.SetToFreeThisTurn()` on card `card_uid` ([`Card::set_to_free_this_turn`]).
/// No-op if `card_uid` is out of bounds.
pub fn set_card_free_this_turn(state: &mut CombatState, card_uid: CardUid) {
    if let Some(card) = state.cards.get_mut(card_uid.0.get() as usize - 1) {
        card.set_to_free_this_turn();
    }
}

// ---------------------------------------------------------------------------
// Hand-search primitives (bead sts2-rs-oac / G3 plumbing)
// ---------------------------------------------------------------------------
//
// Several Python card effects request a player-driven card choice from
// hand via `combat.request_card_choice(...)` / `request_multi_card_choice`
// (e.g. Survivor "discard a hand card", Armaments "upgrade a hand card",
// Acrobatics "discard 1 of N drawn"). The full pending-choice surface
// (action enumeration over `cards-in-hand`, choice deferral across
// engine ticks) is deferred to a separate bead; meanwhile the parity-
// approximation we ship is **auto-pick the first match**.
//
// Why "first match" and not "random pick":
// - Python's `request_card_choice` is *agent-driven* — the choice is
//   exposed as an Action and consumed at the policy layer. Auto-picking
//   randomly would make the engine's behaviour non-deterministic for a
//   given seed in a way that diverges from Python (which never picks at
//   all without an agent prompt). Auto-picking the first match is a
//   stable, deterministic stand-in that keeps the rest of the effect
//   (block, draw, power-grant) reachable from card play.
// - Once `sts2-rs-oac`'s sibling pending-choice bead lands, these call
//   sites swap their auto-pick for a pending-choice push without
//   changing the search predicate.
//
// Mirrors the predicate-based hand iteration shape Python uses in line
// (`[c for c in combat.hand if c.is_skill]` / `if not c.upgraded`).

/// Iterate hand cards matching `predicate`, returning their `CardUid`s in
/// hand order. The predicate receives the resolved [`crate::cards::Card`]
/// POD (looked up via `state.cards[uid - 1]`).
///
/// Bead `sts2-rs-oac` (G3 plumbing). Mirrors the inline
/// `[c for c in combat.hand if <predicate>]` shape that Python factories
/// use across `sts2-rl-agent@01c094d:sts2_env/cards/{silent,ironclad}.py`
/// (Armaments / Flechettes / Prepared / etc.).
///
/// Hand-order is the canonical iteration order — Python preserves
/// insertion order in `combat.hand: list[CardInstance]`, and the Rust
/// `PlayerCombatState.hand: SmallVec<[CardUid; 16]>` matches.
#[must_use]
pub fn search_hand<F>(state: &CombatState, mut predicate: F) -> smallvec::SmallVec<[CardUid; 8]>
where
    F: FnMut(&crate::cards::Card) -> bool,
{
    let mut out = smallvec::SmallVec::<[CardUid; 8]>::new();
    for &uid in &state.player.hand {
        let idx = uid.0.get() as usize - 1;
        if let Some(card) = state.cards.get(idx) {
            if predicate(card) {
                out.push(uid);
            }
        }
    }
    out
}

/// Pick the first hand card matching `predicate` (the auto-pick parity
/// approximation — see module-level note above).
///
/// Bead `sts2-rs-oac`. Returns `None` when no hand card matches (which
/// includes the empty-hand case). Per Python's
/// `if not combat.hand: return` early-out shape — callers should treat
/// `None` as a no-op tail (the rest of the effect already ran).
#[must_use]
pub fn pick_first_in_hand<F>(state: &CombatState, mut predicate: F) -> Option<CardUid>
where
    F: FnMut(&crate::cards::Card) -> bool,
{
    state.player.hand.iter().copied().find(|uid| {
        let idx = uid.0.get() as usize - 1;
        match state.cards.get(idx) {
            Some(card) => predicate(card),
            None => false,
        }
    })
}

/// Open an in-combat **"discard `count` cards from hand"** selection,
/// mirroring C# `CardSelectCmd.FromHandForDiscard(count)` (the surface
/// behind Survivor / Acrobatics / Prepared / DaggerThrow / …). Replaces
/// the [`pick_first_in_hand`] auto-pick stand-in (bead `sts2-rs-oac`) for
/// the cards that the C# game presents as a genuine player choice — the
/// whole-run lockstep needs Rust to PAUSE at the same `card_select`
/// decision point C# does, not silently auto-discard (else the per-step
/// decision counters desync mid-combat: run-parity `cs=card_select
/// rs=combat`, e.g. Silent seed RP2 @step35).
///
/// Threshold matches the decomp (same `FromSimpleGrid`
/// `count <= MinSelect` short-circuit Headbutt documents): when the hand
/// holds `<= count` cards, C# auto-resolves and discards them all without
/// a prompt, so we discard the whole hand inline; only `hand.len() >
/// count` opens a `ChoiceKind::DiscardFromHand` pending choice (resolved
/// by `resolve_discard_from_hand`, which performs the pile move + fires
/// `AfterCardDiscarded`). Options are the current hand **in hand order** —
/// the card being played has already moved to the Play pile by the time
/// its OnPlay effect runs (`combat_loop::play_card` moves Hand→Play before
/// dispatch), so it is correctly excluded. No-op when `count == 0` or the
/// hand is empty, and when a pending choice is already open (defensive).
pub fn request_discard_from_hand(state: &mut CombatState, count: usize) {
    // The card's own draw suspended on a Stratagem reshuffle select: C#
    // awaits the whole `CardPileCmd.Draw` before `FromHandForDiscard`
    // (Prepared.cs:23-24, Acrobatics.cs:23-24, DaggerThrow.cs:33-34), so the discard
    // must open once the resumed draw finishes, over the post-draw hand.
    // Dropping it here (the no-op below) lost Prepared's discard entirely
    // (Silent/Defect v0.111.0 full seed 1475).
    if count > 0 && state.pending_choice.is_some() {
        if let Some(pdr) = state.pending_draw_resume.as_mut() {
            pdr.then_discard_from_hand = u8::try_from(count).unwrap_or(u8::MAX);
            return;
        }
    }
    if count == 0 || state.player.hand.is_empty() || state.pending_choice.is_some() {
        return;
    }
    // C# `CardSelectCmd.FromHand`/`FromSimpleGrid` early-return `Array.Empty`
    // when `CombatManager.IsOverOrEnding`, so a post-damage discard prompt
    // (DaggerThrow, Acrobatics, Survivor, …) never opens when the card's own
    // attack was lethal — combat proceeds straight to the reward. Without this,
    // Rust strands in a phantom `card_select` while C# is already at
    // `card_reward`. Covers the whole FromHandForDiscard family in one place.
    if crate::combat_loop::combat_is_ending(state) {
        return;
    }
    if state.player.hand.len() <= count {
        // `FromHandForDiscard` auto-resolve: discard the whole hand, no prompt.
        // Route through the shared C# `Discard` path so each fires
        // `AfterCardDiscarded` and any discarded `Sly` card auto-plays.
        let hand: smallvec::SmallVec<[CardUid; 16]> = state.player.hand.iter().copied().collect();
        let _ = discard_hand_cards_with_hooks(state, &hand);
        return;
    }
    let options: Vec<crate::ChoiceOption> = state
        .player
        .hand
        .iter()
        .copied()
        .map(|uid| {
            let idx = uid.0.get() as usize - 1;
            crate::ChoiceOption {
                card_id: state.cards[idx].card_id,
                upgraded: state.cards[idx].upgraded,
                card_uid: Some(uid),
                source_pile: Some(crate::cards::PileKind::Hand),
            }
        })
        .collect();
    let n = u8::try_from(count).unwrap_or(u8::MAX);
    state.request_card_choice(crate::ChoiceKind::DiscardFromHand, options, n, n, false);
}

/// Discard a concrete list of hand cards with full C# `CardCmd.DiscardAndDraw`
/// semantics (`MegaCrit.Sts2.Core.Commands/CardCmd.cs:128-161`): for each card,
/// move Hand→Discard and fire `Hook.AfterCardDiscarded`; then, **after** all
/// discards (and any caller-side draw), auto-play every discarded `Sly` card
/// (`AutoPlay(item, SlyDiscard)`).
///
/// The Sly auto-play is the load-bearing piece the raw `cards::discard` move
/// silently dropped: discarding a Sly card (Ricochet, Sneaky, Reflex, …) makes
/// it auto-play wherever it now sits. Every active-discard surface
/// (FromHandForDiscard: Survivor / Acrobatics / Prepared / DaggerThrow /
/// ToolsOfTheTrade, plus the GamblersBrew discard-and-draw) routes through here
/// so the Sly trigger fires identically to CalculatedGamble. The end-of-turn
/// hand flush is a *raw* pile move that fires no discard hook (see
/// `feedback_eot_flush_fires_no_discard_hook`) and must NOT use this path.
///
/// `uids` are discarded in the given order (caller supplies hand order). Cards
/// no longer in hand are skipped defensively. Returns the number actually
/// discarded so callers that draw "that many" (GamblersBrew) can size the draw.
///
/// Thin alias for [`discard_and_draw_with_hooks`] with no interleaved draw —
/// the FromHandForDiscard family (Survivor / Acrobatics / Prepared /
/// DaggerThrow / ToolsOfTheTrade / StormOfSteel / ShadowStep / HiddenDaggers)
/// discards without a paired draw, so the Sly auto-play runs immediately after
/// the discards.
pub fn discard_hand_cards_with_hooks(state: &mut CombatState, uids: &[CardUid]) -> i32 {
    discard_and_draw_with_hooks(state, uids, 0)
}

/// Scrape's tail (Scrape.cs OnPlay): discard the cards its `CardPileCmd.Draw`
/// returned that are not free. v0.107.1 reads
/// `EnergyCost.GetWithModifiers(Local) != 0 || CostsX` (Scrape.cs:30), so a
/// RocketPunch made free until played by a status draw is 0 and stays in hand
/// (Defect full seed 3042). v0.111.0 reads `GetWithModifiers(All)`
/// (Scrape.cs:32): the global modifiers count too, and a negative base (an
/// unplayable Infection / curse, cost -1) returns -1 before any clamp
/// (CardEnergyCost.cs:101-104), so it is discarded (Defect QX197).
/// `CardCmd.Discard` fires `AfterCardDiscarded` per card and auto-plays a
/// discarded Sly card, hence the hooked discard.
pub fn scrape_discard_drawn(state: &mut CombatState, drawn: &[CardUid]) {
    let v0111 = state.game_version >= crate::GameVersion::V0_111_0;
    let to_discard: smallvec::SmallVec<[CardUid; 10]> = drawn
        .iter()
        .copied()
        .filter(|&uid| {
            let Some(card) = state.cards.get(uid.0.get() as usize - 1) else {
                return false;
            };
            if v0111 {
                card.is_x_cost()
                    || card.base_cost() < 0
                    || hooks::effective_card_cost_inst(state, card, uid) != 0
            } else {
                // `GetWithModifiers(Local)` ends in `Math.Max(0, num)`: a Stomp
                // shaved past 0 this turn reads 0 and stays (loadout Ironclad
                // 3138 v0.107.1: Rust discarded it on its -1 debt).
                card.is_x_cost() || card.base_cost() < 0 || card.local_cost().max(0) != 0
            }
        })
        .collect();
    let _ = discard_and_draw_with_hooks(state, &to_discard, 0);
}

/// Full C# `CardCmd.DiscardAndDraw` (`MegaCrit.Sts2.Core.Commands/CardCmd.cs:172-205`):
/// for each card, move Hand→Discard and fire `Hook.AfterCardDiscarded`; then,
/// **after** an optional `draw_count`-card draw, auto-play every discarded
/// `Sly` card (`AutoPlay(item, SlyDiscard)`).
///
/// The draw lands BETWEEN the discards and the Sly auto-play, exactly as C#
/// orders it (CalculatedGamble's `DiscardAndDraw(hand, hand.Count)` reshuffles
/// the just-discarded cards back into draw before re-playing a Sly card, so the
/// auto-play picks it up wherever the draw left it). `draw_count == 0` is the
/// no-draw discard surface (see [`discard_hand_cards_with_hooks`]).
///
/// The draw goes through [`crate::cards::draw_via_state`] — the C# `ShouldDraw`
/// gate (so `NO_DRAW`/Scrutiny vetoes it, matching `CardPileCmd.Draw`) plus the
/// combat-ending reshuffle cap and the Stratagem/Hellraiser per-card loop.
/// (The `draw_into_hand` batch path reshuffles on a different RNG cursor and
/// desynced two CalculatedGamble seeds — 1035/1754 — vs the oracle.) Returns
/// the number of cards actually discarded.
pub fn discard_and_draw_with_hooks(
    state: &mut CombatState,
    uids: &[CardUid],
    draw_count: i32,
) -> i32 {
    discard_and_draw_then(state, uids, draw_count, None)
}

/// [`discard_hand_cards_with_hooks`] for a card whose `OnPlay` goes on after
/// the discard: `then` runs once every discarded Sly card has auto-played,
/// which may be after a Sly card's select is answered (GH #1131).
pub fn discard_hand_cards_then(
    state: &mut CombatState,
    uids: &[CardUid],
    then: Option<crate::LoopThen>,
) -> i32 {
    discard_and_draw_then(state, uids, 0, then)
}

/// [`discard_and_draw_with_hooks`] with a follow-up `then`, run after the
/// last Sly auto-play. See [`play_discarded_sly`].
pub fn discard_and_draw_then(
    state: &mut CombatState,
    uids: &[CardUid],
    draw_count: i32,
    then: Option<crate::LoopThen>,
) -> i32 {
    // Snapshot the Sly subset (in supplied order) BEFORE the discards, mirroring
    // C#'s `slyCards` collection inside the per-card loop. The keyword check is
    // stable across the pile move, so reading it up front is equivalent.
    let mut sly: smallvec::SmallVec<[CardUid; 8]> = smallvec::SmallVec::new();
    let mut discarded: i32 = 0;
    for &uid in uids {
        if !state.player.hand.contains(&uid) {
            continue;
        }
        let idx = uid.0.get() as usize - 1;
        let (card_id, card_tags, card_type) = state.cards.get(idx).map_or_else(
            || (None, crate::cards::CardTags::NONE, None),
            |c| (Some(c.card_id), c.tags, Some(c.card_type)),
        );
        // C# `card.Keywords.Contains(Sly)` resolves to `IsSlyThisTurn` — the
        // permanent `Sly` keyword OR a single-turn Sly (HandTrick) OR a
        // combat-long instance Sly (MasterPlanner). See [`card_is_sly`].
        if card_is_sly(state, uid) {
            sly.push(uid);
        }
        let _ = crate::cards::discard(&mut state.player, uid);
        discarded += 1;
        // C# fires `Hook.AfterCardDiscarded` per discarded card (CardCmd.cs:194).
        let hook_ctx = crate::hooks::CardCtx {
            card_id,
            card_uid: Some(uid),
            card_tags,
            card_type,
        };
        crate::hooks::dispatch_after_card_discarded(state, &hook_ctx);
    }
    // C# draws AFTER every discard fires its hook (CardCmd.cs:197-200), BEFORE
    // the Sly auto-play.
    let was_suspended = state.pending_draw_resume.is_some();
    if draw_count > 0 {
        let _ = crate::cards::draw_via_state(state, draw_count);
    }
    // Auto-play each Sly card BY REFERENCE wherever it now sits (the draw may
    // have reshuffled it from discard into draw/hand) — CardCmd.cs:201-204.
    if then.is_none() {
        play_discarded_sly_after_draw(state, &sly, was_suspended);
    } else {
        // Only the no-draw discard surface carries a follow-up.
        debug_assert!(draw_count == 0, "a DiscardAndDraw draw with a follow-up");
        play_discarded_sly(state, &sly, then);
    }
    discarded
}

/// [`play_discarded_sly`] for a `CardCmd.DiscardAndDraw` whose draw just ran.
///
/// C# awaits the whole `CardPileCmd.Draw` before the Sly auto-plays
/// (CardCmd.cs:197-204, both builds). When that draw suspended on a
/// reshuffle's Stratagem pick (`state.pending_draw_resume` set now but not
/// before, `was_suspended`), the cards still to draw come first: the
/// auto-plays are owed to the suspended draw and run once it finishes
/// ([`crate::CombatState::resume_suspended_draw`]). Playing them at once took
/// a Sly card the reshuffle had put on top straight from the draw pile, so
/// the resumed draw took the next card instead (loadout Silent 1053: a
/// Calculated Gamble whose Ricochet C# drew and then played).
pub fn play_discarded_sly_after_draw(
    state: &mut CombatState,
    sly: &[CardUid],
    was_suspended: bool,
) {
    if !was_suspended && state.pending_choice.is_some() {
        if let Some(pdr) = state.pending_draw_resume.as_mut() {
            pdr.then_play_sly = true;
            let mut next = state
                .cards
                .iter()
                .map(|c| c.combat_var(crate::cards::VarKey::SlyOwedAfterDraw))
                .max()
                .unwrap_or(0);
            for &uid in sly {
                if let Some(card) = crate::card_by_uid_mut(&mut state.cards, uid) {
                    next += 1;
                    card.set_combat_var(crate::cards::VarKey::SlyOwedAfterDraw, next);
                }
            }
            return;
        }
    }
    play_discarded_sly(state, sly, None);
}

/// The Sly cards [`play_discarded_sly_after_draw`] marked as owed to a
/// suspended draw, in the order they were discarded; clears the marks.
pub fn take_sly_owed_after_draw(state: &mut CombatState) -> smallvec::SmallVec<[CardUid; 4]> {
    let key = crate::cards::VarKey::SlyOwedAfterDraw;
    let mut owed: smallvec::SmallVec<[(i32, CardUid); 4]> = state
        .cards
        .iter()
        .filter(|c| c.combat_var(key) > 0)
        .map(|c| (c.combat_var(key), c.uid))
        .collect();
    owed.sort_unstable_by_key(|&(order, _)| order);
    for &(_, uid) in &owed {
        if let Some(card) = crate::card_by_uid_mut(&mut state.cards, uid) {
            card.combat_vars.zero(key);
        }
    }
    owed.into_iter().map(|(_, uid)| uid).collect()
}

/// Auto-play the discarded Sly `cards` in order, then apply `then`.
///
/// C# `await`s each `CardCmd.AutoPlay(item, SlyDiscard)` before the next
/// (`CardCmd.cs:201-204`), and the discarding card's `OnPlay` goes on only
/// after the last. So a Sly card whose `OnPlay` opens a select (an Armaments
/// made Sly by Master Planner) prompts the player, and the rest waits for the
/// pick. When the auto-play parks its resume, this records the cards not yet
/// played and `then` as the resume's loop continuation, the BeatDown way, and
/// returns. The resolver runs [`resume_discarded_sly`] once the pick lands.
///
/// Under an already parked auto-play (the discarding card was itself
/// auto-played and its discard select prompted), that resume is set aside. If
/// a Sly card parks, the set-aside finish becomes the frame after it, with the
/// rest of this loop as the frame's continuation, the DecisionsDecisions way.
/// GH #1131.
pub fn play_discarded_sly(
    state: &mut CombatState,
    cards: &[CardUid],
    then: Option<crate::LoopThen>,
) {
    let outer = state.pending_autoplay_resume.take();
    // A resume parked mid-draw keeps its draw on the top slot only, so it
    // can't become a frame. Drain inline there, as before.
    let nestable = outer.as_ref().is_none_or(|o| o.inner_draw.is_none());
    debug_assert!(nestable, "a Sly discard under a parked draw");
    let prev_inline = state.autoplay_inline_resolve;
    state.autoplay_inline_resolve = prev_inline || !nestable;
    for (i, &uid) in cards.iter().enumerate() {
        let _ = crate::combat_loop::auto_play_specific_card_from_discard(state, uid, false);
        if let Some(resume) = state.pending_autoplay_resume.as_mut() {
            let cont = crate::AutoPlayLoopCont {
                resume_fn: resume_discarded_sly,
                iters: 0,
                uids: cards[i + 1..].iter().copied().collect(),
                force_exhaust: false,
                then,
            };
            debug_assert!(resume.continuation.is_none());
            match outer {
                Some(outer) => {
                    resume
                        .parent_chain
                        .push(crate::combat_loop::AutoPlayParentFrame {
                            finish: outer.finish,
                            remaining: outer.remaining,
                            force_exhaust: outer.force_exhaust,
                            continuation: Some(cont),
                        });
                    resume.parent_chain.extend(outer.parent_chain);
                    resume.continuation = outer.continuation;
                }
                None => resume.continuation = Some(cont),
            }
            state.autoplay_inline_resolve = prev_inline;
            return;
        }
    }
    state.autoplay_inline_resolve = prev_inline;
    state.pending_autoplay_resume = outer;
    if let Some(then) = then {
        then.run(state);
    }
}

/// Loop continuation for [`play_discarded_sly`]: the Sly cards not yet played
/// and the discarding card's follow-up.
fn resume_discarded_sly(state: &mut CombatState, cont: &crate::AutoPlayLoopCont) {
    play_discarded_sly(state, &cont.uids, cont.then);
}

// ---------------------------------------------------------------------------
// Pile-search + card-spawn primitives (bead sts2-rs-5om / G3 plumbing)
// ---------------------------------------------------------------------------
//
// Companion to the hand-search primitives in `sts2-rs-oac` above. Several
// Python card effects either iterate a non-hand pile (Headbutt picks from
// discard; Liquid Memories returns a card from exhaust; Apparition asks
// the agent for a specific Shiv in draw/discard) or spawn a brand-new
// card into a target pile (Hidden Daggers → 3 Shivs into discard;
// Gunk Up → Slimed into discard; Hotfix → a Hotfix into hand; Turbo →
// Void into discard; Cinder → Burn into discard; Anger → copy of itself
// into discard).
//
// As with `search_hand`, the full agent-driven `request_card_choice`
// surface is deferred — the parity-approximation here is **auto-pick
// the first match** (deterministic given a seed, matches the rest of
// the dispatch shape). `spawn_card_into_pile` is the engine-side
// arena-allocation surface; `sts2-content` builds the [`Card`] POD
// from its per-character spec factory and hands it in (sts2-core
// can't take the reverse dependency).

/// Which pile a search / spawn targets. Mirrors Python's
/// `_zones_for_creature(owner)["draw"|"discard"|"exhaust"]` indirection
/// at `sts2-rl-agent@01c094d:sts2_env/core/combat.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PileKind {
    /// `state.player.draw` — top of pile is index 0 (next-drawn).
    Draw,
    /// `state.player.discard` — append-order (back).
    Discard,
    /// `state.player.exhaust` — append-order (back).
    Exhaust,
    /// `state.player.hand` — append-order. Provided for symmetry with the
    /// `search_hand` surface; new callers should prefer `search_hand`.
    Hand,
}

fn pile_uids(state: &CombatState, pile: PileKind) -> &[CardUid] {
    match pile {
        PileKind::Draw => &state.player.draw,
        PileKind::Discard => &state.player.discard,
        PileKind::Exhaust => &state.player.exhaust,
        PileKind::Hand => &state.player.hand,
    }
}

/// Iterate `pile` for cards matching `predicate`. Returns their `CardUid`s
/// in canonical pile order. Auto-pick parity rationale — see the module-
/// level note on `search_hand` above.
///
/// Bead `sts2-rs-5om`. Mirrors the inline
/// `[c for c in combat.<pile> if <predicate>]` shape used across
/// `sts2-rl-agent@01c094d:sts2_env/cards/*.py` for non-hand pile lookups
/// (Headbutt / Liquid Memories / Apparition / Reboot).
#[must_use]
pub fn search_pile<F>(
    state: &CombatState,
    pile: PileKind,
    mut predicate: F,
) -> smallvec::SmallVec<[CardUid; 8]>
where
    F: FnMut(&crate::cards::Card) -> bool,
{
    let mut out = smallvec::SmallVec::<[CardUid; 8]>::new();
    for &uid in pile_uids(state, pile) {
        let idx = uid.0.get() as usize - 1;
        if let Some(card) = state.cards.get(idx) {
            if predicate(card) {
                out.push(uid);
            }
        }
    }
    out
}

/// Pick the first card in `pile` matching `predicate` (auto-pick
/// approximation). Returns `None` on empty pile / no match.
/// Bead `sts2-rs-5om`.
#[must_use]
pub fn pick_first_in_pile<F>(
    state: &CombatState,
    pile: PileKind,
    mut predicate: F,
) -> Option<CardUid>
where
    F: FnMut(&crate::cards::Card) -> bool,
{
    for &uid in pile_uids(state, pile) {
        let idx = uid.0.get() as usize - 1;
        if let Some(card) = state.cards.get(idx) {
            if predicate(card) {
                return Some(uid);
            }
        }
    }
    None
}

/// Allocate a fresh [`CardUid`] in `state.cards` for the supplied `card`
/// POD and enqueue it into `pile`. Returns the assigned `CardUid` so the
/// caller can bookkeep (tests, attached-power references).
///
/// The caller supplies a [`crate::cards::Card`] whose `uid` field is
/// ignored — `state.cards.len() + 1` is the next 1-based slot. Insertion
/// order mirrors Python:
///   - `PileKind::Draw`: `insert(0, …)` (top — same as
///     [`CombatState::add_card_to_draw_pile`]).
///   - `PileKind::Discard` / `Exhaust` / `Hand`: `push` (back).
///
/// Mirrors `sts2-rl-agent@01c094d:sts2_env/core/combat.py::
/// CombatState.create_card` + `_zones_for_creature(owner)[<pile>].append/insert`.
/// Bead `sts2-rs-5om`.
pub fn spawn_card_into_pile(
    state: &mut CombatState,
    card: crate::cards::Card,
    pile: PileKind,
) -> CardUid {
    // `added_by_player` ≈ a player card is mid-play (Play pile non-empty) for
    // the spawn-based path. The explicit-true path is `route_generated_into_hand`
    // (turn-start power / potion / relic generations, all `addedByPlayer: true`
    // in C# but with an empty Play pile). Bead `sts2-rs-gen-added-by-player`.
    //
    // This heuristic is WRONG for an enemy POWER that generates a card WHILE a
    // player card is being played (Play pile non-empty) — chiefly Aeonglass's
    // `WitheringPresence`, which adds a Wither to the player's hand on every 6th
    // card the player plays. Such generators must route through
    // [`spawn_card_into_pile_with_origin`] with `added_by_player = false` so the
    // player-only `AfterCardGeneratedForCombat` listeners (Regalite +2 Block,
    // Arsenal/Pillar/Smokestack/RocketPunch) do not fire on an enemy-spawned
    // card (C# `Regalite.AfterCardGeneratedForCombat` gates on
    // `creator == Owner`). Bead `sts2-rs-regalite-creator`.
    let added_by_player = !state.player.play.is_empty();
    spawn_card_into_pile_with_origin(state, card, pile, added_by_player)
}

/// Stamp a card instance freshly generated into combat: C#
/// `CardModel.DeckVersion == null` ([`crate::cards::VarKey::GeneratedInCombat`]),
/// and no Dampen restore marker. `DampenPower.cs:29` records the exact card
/// instances it downgraded and `AfterRemoved` re-upgrades only those; a
/// clone of a downgraded card (Anger's copy, Juggling, Dual Wield) inherits
/// the downgraded level but is not in that list, so it stays downgraded.
pub fn stamp_generated_in_combat(card: &mut crate::cards::Card) {
    // A clone keeps its `GENERATED_AS_CLONE` mark (C# `IsClone`).
    if !card.is_clone() {
        card.set_combat_var(
            crate::cards::VarKey::GeneratedInCombat,
            crate::cards::GENERATED_FRESH,
        );
    }
    // Only touch the var when set: writing a 0 would still occupy a slot in
    // the card's inline var store and change its hash.
    if card.combat_var(crate::cards::VarKey::DampenDowngradedMarker) != 0 {
        card.set_combat_var(crate::cards::VarKey::DampenDowngradedMarker, 0);
    }
}

/// As [`spawn_card_into_pile`], but with an EXPLICIT `added_by_player` flag
/// (the C# `Player? creator` of `AfterCardGeneratedForCombat`). Enemy-power
/// card generators (WitheringPresence Withers, enemy-inflicted status cards)
/// pass `false`; player generations pass the play-pile heuristic via the
/// wrapper above.
pub fn spawn_card_into_pile_with_origin(
    state: &mut CombatState,
    card: crate::cards::Card,
    pile: PileKind,
    added_by_player: bool,
) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    let mut card = card;
    card.uid = uid;
    // C# `CardModel.DeckVersion == null` for combat-generated cards — the
    // marker deck-scoped pickers (ThievingHopper steal) filter on.
    stamp_generated_in_combat(&mut card);
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    card.normalize_for_version(state.game_version);
    state.cards.push(card);
    // C# `CardPileCmd.Add` hand-cap redirect (`CardPileCmd.cs:420-424`):
    // `isFullHandAdd = targetPile.Type == Hand && Cards.Count >= MaxCardsInHand`
    // (`CardPile.cs:21`, 10) → `targetPile = CardPile.Get(PileType.Discard, …)`.
    // Every spawn routed through this helper corresponds to a C#
    // `CardPileCmd.Add`, so the redirect lives here at the shared choke point.
    // First live consumer: Aeonglass WitheringPresence's every-6th-card Wither
    // (`WitheringPresencePower.cs:55` `AddToCombatAndPreview<Wither>(…,
    // PileType.Hand, 1, null)`) — the pre-fix unconditional push grew an 11th
    // hand card whose `OnTurnEndInHand` burned the player for 3+3·L where C#
    // parked the overflow Wither in discard (0 damage that turn).
    let pile = if pile == PileKind::Hand && state.player.hand.len() >= crate::cards::HAND_LIMIT {
        PileKind::Discard
    } else {
        pile
    };
    // C# `CardPileCmd.Add` fails before touching a combat pile once
    // `CombatManager.IsEnding` ([`combat_pile_add_blocked`]); the generation
    // record and hook still run (their listeners gate on ending themselves).
    if !combat_pile_add_blocked(state) {
        match pile {
            PileKind::Draw => state.player.draw.insert(0, uid),
            PileKind::Discard => state.player.discard.push(uid),
            PileKind::Exhaust => state.player.exhaust.push(uid),
            PileKind::Hand => state.player.hand.push(uid),
        }
    }
    fire_card_generation_listeners(state, uid, pile, added_by_player);
    uid
}

/// C# `CardPileCmd.Add` returns `success = false` before adding anything when
/// `newPile.IsCombatPile && CombatManager.Instance.IsEnding`
/// (CardPileCmd.cs:312 v0.107.1, :379 v0.111.0). A card generated after the
/// last enemy died (Calamity's Attack after a lethal Fiend Fire) never reaches
/// the hand, draw or discard pile. It stays in the arena so its uid still
/// resolves, like C#'s `cardAdded` with `success = false`.
///
/// This reads the enemy half of `IsEnding` (plus a finished combat), not the
/// player's HP. C#'s player half is `_pendingLoss`, which is only set once a
/// death has gone through the death-prevention relics. A Rust player at 0 HP
/// is not that yet, and the 0-HP default player of unit fixtures is not
/// "ending" either.
#[must_use]
pub fn combat_pile_add_blocked(state: &CombatState) -> bool {
    state.is_over || crate::combat_loop::enemy_side_is_ending(state)
}

/// Spawn a combat-generated card into the DRAW pile at a uniform-random index,
/// consuming the run-level `shuffle` stream — the engine-side mirror of C#
/// `CardPileCmd.AddGeneratedCardToCombat(card, PileType.Draw, .., CardPilePosition.Random)`,
/// whose `targetPile.AddInternal(card, Rng.Shuffle.NextInt(Count + 1))`
/// (`CardPileCmd.cs:511-515`) draws ONE `Shuffle` read every time — even when
/// the draw pile is empty (`NextInt(1)` still bumps `Counter`,
/// `Rng.cs:90-94`). `added_by_player` is the play-pile heuristic shared with
/// [`spawn_card_into_pile`]; enemy status generators (Noisebot) spawn with an
/// empty Play pile, so it resolves to `false`. Bead `sts2-rs-noisebot-shuffle`.
pub fn spawn_card_into_draw_pile_random(
    state: &mut CombatState,
    card: crate::cards::Card,
) -> CardUid {
    let added_by_player = !state.player.play.is_empty();
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    let mut card = card;
    card.uid = uid;
    stamp_generated_in_combat(&mut card);
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    card.normalize_for_version(state.game_version);
    state.cards.push(card);
    add_generated_card_to_draw_random(state, uid, added_by_player);
    uid
}

/// Place a card that is ALREADY in the arena (minted by a content-side
/// sampler such as Metamorphosis's `GetForCombat`) at a uniform-random index
/// of the draw pile and fire the generation listeners — the
/// `CardPilePosition.Random` half of C# `CardPileCmd.AddGeneratedCardToCombat`
/// (one `Shuffle` read, then `Hook.AfterCardGeneratedForCombat`). See
/// [`spawn_card_into_draw_pile_random`].
pub fn add_generated_card_to_draw_random(
    state: &mut CombatState,
    uid: CardUid,
    added_by_player: bool,
) {
    // C# `CardPileCmd.Add` returns `success = false` before the slot read
    // when `newPile.IsCombatPile && CombatManager.IsEnding` (CardPileCmd.cs:312
    // v0.107.1, :379 v0.111.0): a card generated after the last enemy died
    // never reaches the draw pile and draws no `Shuffle` read. The caller's
    // `History.CardGenerated` and the generation hook still run (the hook's
    // listeners are themselves gated on ending), so the tally path below is
    // kept. Metamorphosis whose Regalite+Juggernaut kills mid-loop: Rust read
    // one extra `Shuffle` and desynced the next combat's opening shuffle
    // (random-loadout Ironclad v0.107.1 seed 1001).
    if crate::combat_loop::combat_is_ending(state) {
        fire_card_generation_listeners(state, uid, PileKind::Draw, added_by_player);
        return;
    }
    // Bead `sts2-rs-09x5n` (F2 `HiddenInfoMode::BeliefDraw`): under a
    // deferred-draw-order pile there is no physical position for a random
    // index to land in — fold the new uid into the canonical (uid-sorted)
    // unknown segment instead, consuming no RNG.
    if state.player.deferred_draw_order {
        crate::cards::insert_into_unknown_sorted(&mut state.player, uid);
    } else {
        // C# `Rng.Shuffle.NextInt(Count + 1)` — inclusive `[0, len]` (len+1
        // slots, a true append at `idx == len`). The read happens BEFORE the
        // card is in the pile, so `len` is the pre-insert count.
        let len = state.player.draw.len();
        let idx = state.run_rng.shuffle.next_int(0, len as i32) as usize;
        state.player.draw.insert(idx, uid);
    }
    fire_card_generation_listeners(state, uid, PileKind::Draw, added_by_player);
}

/// Fire the player-only `AfterCardGeneratedForCombat` listener cascade that
/// follows any combat card-generation (`AddInternal` → `Hook.AfterCardEnteredCombat`
/// → the relic/power generation listeners). Shared by
/// [`spawn_card_into_pile_with_origin`] and [`spawn_card_into_draw_pile_random`].
///
/// Exposed `pub` so content-crate generators that build + place a card by hand
/// (e.g. `WhiteNoise`, which pushes a generated Power straight into Hand) can
/// fire the same cascade C# `CardPileCmd.AddGeneratedCardToCombat` does — else
/// Regalite/Arsenal/etc. never see the generation. Pass `added_by_player=true`
/// for player-play-time generations (C# `creator = base.Owner`).
pub fn fire_card_generation_listeners(
    state: &mut CombatState,
    uid: CardUid,
    pile: PileKind,
    added_by_player: bool,
) {
    // Event log (C# `CardGeneratedEntry`). This is the shared funnel every
    // combat-generated card passes through, so it is also the one place that
    // sees a monster's inflicted Slimed / Wound — the move row above it says
    // "used Sticky Shot" and this says what that did.
    if crate::event_log::armed(state) {
        if let Some(card) = state.cards.get(uid.0.get() as usize - 1) {
            let (card_id, upgraded) = (card.card_id, card.upgraded);
            crate::event_log::record(
                state,
                crate::event_log::EventLogEntry::CardGenerated {
                    card: card_id,
                    upgraded,
                    pile: match pile {
                        PileKind::Draw => "draw",
                        PileKind::Discard => "discard",
                        PileKind::Hand => "hand",
                        PileKind::Exhaust => "exhaust",
                    },
                    by_player: added_by_player,
                },
            );
        }
    }
    // C# `AddGeneratedCardsToCombat` records `History.CardGenerated(.., creator)`
    // (`CardPileCmd.cs:243`); Supermassive counts the entries whose
    // `Creator == Owner` (`Supermassive.cs:22-27`), so only player-created
    // cards bump the tally (monster-inflicted statuses pass `creator: null`).
    if added_by_player {
        record_player_generated_card(state);
    }
    fire_after_card_entered_combat(state, uid, pile, added_by_player);
    fire_card_generated_for_combat_powers(state, uid, added_by_player);
}

/// Bump the per-combat count of player-created `CardGeneratedEntry`s that
/// Supermassive's damage reads (`Supermassive.cs:22-27`).
pub fn record_player_generated_card(state: &mut CombatState) {
    state.player.generated_cards_this_combat =
        state.player.generated_cards_this_combat.saturating_add(1);
}

/// The `AfterCardGeneratedForCombat` listener bundle. C#
/// `CardPileCmd.AddGeneratedCardToCombat` raises that hook on EVERY listener,
/// so every Rust site that generates a card into combat fires all of these —
/// the ported listeners are hand-rolled per power rather than dispatched off
/// `PowerKind`, and keeping the list in one place is what stops a site from
/// silently carrying only some of them. Powers run before the card listener,
/// matching C# `IterateHookListeners` (Powers -> Relics -> cards).
///
/// `added_by_player` is the C# `creator != null && creator.Creature == Owner`
/// test; each caller supplies it (an explicit `true` where C# passes
/// `addedByPlayer: true` unconditionally, else the spawn path's Play-pile
/// heuristic).
pub(crate) fn fire_card_generated_for_combat_powers(
    state: &mut CombatState,
    uid: CardUid,
    added_by_player: bool,
) {
    // C# `Hook.AfterCardGeneratedForCombat` dispatches through
    // `IterateCombatHookListeners`, which yields NO listeners if combat is
    // over or ending (`CombatManager.IsOverOrEnding`) when the dispatch
    // begins (`Hook.cs:177-207`, `:395-402`). `AddGeneratedCardsToCombat`
    // still records `History.CardGenerated` (its only guard is
    // `!IsInProgress`, `CardPileCmd.cs:281-313`), so callers keep the
    // Supermassive tally outside this gate. (The card itself is NOT added:
    // `CardPileCmd.Add` bails on `IsEnding`, CardPileCmd.cs:312.) A Gunk Up that kills the
    // last primary enemy and then generates Slimed must NOT roll a
    // TrashToTreasure orb (Defect seed 1016: the extra `combat_orbs` read
    // carried into the next combat).
    if crate::combat_loop::combat_is_ending(state) {
        return;
    }
    // The player's power listeners run in the order the powers sit on the
    // player (C# `IterateHookListeners` walks `Creature.Powers`, i.e.
    // application order), not in a fixed order. With TrashToTreasure applied
    // before Smokestack, a Burn's orb channel evokes a Dark orb into the
    // 3-HP Eye before Smokestack's 5 kills it; the other way round the evoke
    // lands on Fogmog (loadout Defect v0.107.1 1378: Fogmog C# 27, Rust 21).
    for kind in generation_power_listener_order(state) {
        match kind {
            crate::PowerKind::Smokestack => {
                fire_smokestack_on_status_generation(state, uid, added_by_player);
            }
            crate::PowerKind::Arsenal => {
                fire_arsenal_strength_on_card_generation(state, uid, added_by_player);
            }
            crate::PowerKind::PillarOfCreation => {
                fire_pillar_of_creation_block_on_card_generation(state, uid, added_by_player);
            }
            crate::PowerKind::TrashToTreasure => {
                fire_trash_to_treasure_orbs_on_status_generation(state, uid, added_by_player);
            }
            _ => {}
        }
    }
    fire_rocketpunch_free_on_status_generation(state, uid, added_by_player);
    fire_soulbound_on_soul_generation(state, uid, added_by_player);
    match_generated_wither_to_aeonglass(state, uid);
}

/// The player-power `AfterCardGeneratedForCombat` listeners (Smokestack,
/// Arsenal, PillarOfCreation, TrashToTreasure) the player holds, each once,
/// in the order the powers sit on the player.
fn generation_power_listener_order(
    state: &CombatState,
) -> smallvec::SmallVec<[crate::PowerKind; 4]> {
    use crate::PowerKind;
    const KINDS: [PowerKind; 4] = [
        PowerKind::Smokestack,
        PowerKind::Arsenal,
        PowerKind::PillarOfCreation,
        PowerKind::TrashToTreasure,
    ];
    let mut out: smallvec::SmallVec<[PowerKind; 4]> = smallvec::SmallVec::new();
    // Perf: no live listener kind anywhere ⇒ nothing to order.
    let present = state.powers.present_kinds();
    if !KINDS.iter().any(|k| present.contains(*k)) {
        return out;
    }
    for puid in state.player_power_uids() {
        if let Some(p) = state.powers.get_by_id(puid.0) {
            if KINDS.contains(&p.kind) && !out.contains(&p.kind) {
                out.push(p.kind);
            }
        }
    }
    out
}

/// C# `Aeonglass.AfterCardGeneratedForCombat`: every Wither generated for
/// combat, whoever makes it, is `FakeUpgrade`d once per `WitherUpgradeCount`
/// (the boss's AI counter). That covers the boss's own Withers (Increasing
/// Intensity, Withering Presence) and a card transformed into a Wither
/// (`CardCmd.Transform` raises the hook too; loadout Defect 1241 v0.107.1:
/// Entropy turned a Dazed into a level-0 Wither while the count was 4).
fn match_generated_wither_to_aeonglass(state: &mut CombatState, uid: CardUid) {
    let idx = uid.0.get() as usize - 1;
    if state.cards.get(idx).map(|c| c.card_id) != Some(crate::cards::CardId::Wither) {
        return;
    }
    let levels: i32 = state
        .creatures
        .iter()
        .filter(|c| c.species == Some(crate::MonsterKind::Aeonglass) && c.current_hp > 0)
        .filter_map(|c| c.ai.as_ref())
        .map(|ai| ai.counter().max(0))
        .sum();
    if levels > 0 {
        if let Some(card) = state.cards.get_mut(idx) {
            let next = card.combat_var(crate::cards::VarKey::WitherFakeUpgradeLevel) + levels;
            card.set_combat_var(crate::cards::VarKey::WitherFakeUpgradeLevel, next);
        }
    }
}

/// v0.111.0 `SoulboundPower.AfterCardGeneratedForCombat`: when the player who
/// put Soulbound on a teammate (`base.Applier`, [`crate::coop::CoopState::power_seat`])
/// makes a Soul, the teammate gets `Amount` Souls of their own, each at a
/// random spot in their draw pile (`CardPilePosition.Random`, one `Shuffle`
/// read apiece). Holders in player order. Those Souls are the holder's own,
/// spawned outside a card play, so they set off no Soulbound in turn (C#'s
/// `IsAddingSoul` guard).
fn fire_soulbound_on_soul_generation(state: &mut CombatState, uid: CardUid, added_by_player: bool) {
    if !added_by_player
        || !crate::coop::has_seats(state)
        || state.cards.get(uid.0.get() as usize - 1).map(|c| c.card_id)
            != Some(crate::cards::CardId::Soul)
        || crate::coop::players_power_amount(state, crate::PowerKind::Soulbound) <= 0
    {
        return;
    }
    let creator = state.coop.seated;
    crate::coop::each_other_player(state, |state| {
        // A Soulbound read off the wire has no recorded applier: any other
        // player's Soul counts (with two players, the only one it can be).
        let applied_by_creator = |id| {
            state
                .coop
                .power_seat
                .iter()
                .find(|(p, _)| *p == id)
                .is_none_or(|(_, seat)| *seat == creator)
        };
        let amount: i32 = state
            .player_power_uids()
            .into_iter()
            .filter_map(|p| state.powers.get_by_id(p.0))
            .filter(|p| p.kind == crate::PowerKind::Soulbound && applied_by_creator(p.id))
            .map(|p| p.amount)
            .sum();
        if amount <= 0 {
            return;
        }
        crate::event_log::flash_player_power(state, crate::PowerKind::Soulbound);
        let Some(factory) = crate::lookup_card_factory() else {
            return;
        };
        for _ in 0..amount {
            if let Some(soul) = factory(crate::cards::CardId::Soul, false) {
                spawn_card_into_draw_pile_random(state, soul);
            }
        }
    });
}

/// C# `ArsenalPower.AfterCardGeneratedForCombat`
/// (`MegaCrit.Sts2.Core.Models.Powers/ArsenalPower.cs`): whenever a card is
/// generated and added to combat BY THE PLAYER (`addedByPlayer && card.Owner ==
/// player`), the Regent's Arsenal(N) buff grants the player `N` Strength. Unlike
/// Smokestack it has NO card-type filter — ANY player-generated card (Status,
/// Attack, Skill, …) triggers it.
///
/// `addedByPlayer` is approximated by "the player is mid-card-play"
/// (`!player.play.is_empty()`), exactly as `fire_smokestack_on_status_generation`
/// does: a player card's `OnPlay` sits in the Play pile while it generates the
/// card, whereas monster-added cards (Wound/Slimed on the enemy turn) spawn with
/// an empty Play pile. Gated on the player actually holding Arsenal — a
/// Regent-pool power — so it is a no-op for every other character / combat.
/// Regent parity fuzz seeds 104/400/427/438/495 (player.powers.len off by one:
/// the missing Strength from a generated card).
pub(crate) fn fire_arsenal_strength_on_card_generation(
    state: &mut CombatState,
    _uid: CardUid,
    added_by_player: bool,
) {
    if !added_by_player {
        return;
    }
    let amount = state.player_power_amount(crate::PowerKind::Arsenal);
    if amount <= 0 {
        return;
    }
    // C# `ArsenalPower.cs:23` calls `Flash()` inside the same
    // `creator != null && creator.Creature == Owner` guard, immediately before
    // `PowerCmd.Apply<StrengthPower>(Owner, Amount, ..)`. The `amount <= 0`
    // return above is the WraithForm under-fire deviation (see
    // `hooks::after_side_turn_start`'s WraithForm arm): C# has no amount test,
    // but a Counter power at 0 is popped.
    crate::event_log::flash_player_power(state, crate::PowerKind::Arsenal);
    // Grant the player `amount` Strength. Route through the canonical creature
    // surface when the mirror is set (so it stacks with relic/combat-start
    // grants), else the legacy player-power list — mirroring the content-side
    // `apply_power_to_player` helper (which core cannot call: it lives in
    // sts2-content).
    if let Some(uid) = state.player.creature_uid {
        apply_power_to_creature(state, uid, crate::PowerKind::Strength, amount);
    } else {
        crate::apply_player_power_legacy(state, crate::PowerKind::Strength, amount);
    }
}

/// C# `PillarOfCreationPower.AfterCardGeneratedForCombat`
/// (`MegaCrit.Sts2.Core.Models.Powers/PillarOfCreationPower.cs`): whenever a card
/// is generated and added to combat BY THE PLAYER (`addedByPlayer && card.Owner
/// == player`), gain `Amount` block (`CreatureCmd.GainBlock(owner, Amount,
/// ValueProp.Unpowered, null)`). Like Arsenal it has NO card-type filter — ANY
/// player-generated card triggers it.
///
/// `addedByPlayer` is approximated by "the player is mid-card-play"
/// (`!player.play.is_empty()`), exactly as `fire_arsenal_strength_on_card_generation`.
/// Gated on the player holding PillarOfCreation — a Regent-pool power — so it is a
/// no-op for every other character / combat. The block-grant effect was defined
/// (`powers::regent::pillar_of_creation_block_per_generation`) but never wired
/// into any generation site, so generated cards granted 0 block (Regent parity
/// fuzz seeds 22/142/309/458: player.block low — e.g. BEGONE's MinionStrike
/// transform granted no block).
pub(crate) fn fire_pillar_of_creation_block_on_card_generation(
    state: &mut CombatState,
    _uid: CardUid,
    added_by_player: bool,
) {
    if !added_by_player {
        return;
    }
    let amount = state.player_power_amount(crate::PowerKind::PillarOfCreation);
    let block = crate::powers::regent::pillar_of_creation_block_per_generation(amount);
    if block <= 0 {
        return;
    }
    // C# `PillarOfCreationPower.cs:23` calls `Flash()` inside the same
    // `creator != null && creator.Creature == Owner` guard, immediately before
    // `CreatureCmd.GainBlock(Owner, Amount, Unpowered, null)`. The `block <= 0`
    // return above is the WraithForm under-fire deviation.
    crate::event_log::flash_player_power(state, crate::PowerKind::PillarOfCreation);
    let _ = gain_block_player_unpowered(state, block);
}

/// C# `SmokestackPower.AfterCardGeneratedForCombat`
/// (`MegaCrit.Sts2.Core.Models.Powers/SmokestackPower.cs`): when a Status card
/// is generated and added to combat BY THE PLAYER (`addedByPlayer &&
/// card.Type == Status && card.Owner == player`), deal `Amount` Unpowered
/// damage to every hittable enemy. Defect's Turbo (generates a Void) and
/// Overclock (Burn) are the common triggers; the prior Rust port modelled
/// Smokestack as an inert "gain block per card" stub, so it did nothing.
///
/// `addedByPlayer` is approximated by "the player is mid-card-play"
/// (`!player.play.is_empty()`): a player card's `OnPlay` sits in the Play pile
/// while it generates the status, whereas monster-added statuses (Wound from
/// PainfulStabs, Slimed) are spawned during the enemy turn with an empty Play
/// pile. The whole check is gated on the player actually holding Smokestack —
/// a Defect-pool power — so it is a no-op for every other character.
pub(crate) fn fire_smokestack_on_status_generation(
    state: &mut CombatState,
    uid: CardUid,
    added_by_player: bool,
) {
    let Some(card) = state.cards.get(uid.0.get() as usize - 1) else {
        return;
    };
    if card.card_type != crate::cards::CardType::Status {
        return;
    }
    if !added_by_player {
        return;
    }
    let amount = state.player_power_amount(crate::PowerKind::Smokestack);
    if amount <= 0 {
        return;
    }
    // C# `SmokestackPower.cs:21` calls `Flash()` inside the same
    // `card.Type == Status && creator != null && creator.Creature == Owner`
    // guard, ABOVE `CreatureCmd.Damage(HittableEnemies, ..)` — so a status
    // generated with the arena already dead still flashes, and the empty
    // `targets` collection below does not suppress it. The `amount <= 0` return
    // is the WraithForm under-fire deviation.
    crate::event_log::flash_player_power(state, crate::PowerKind::Smokestack);
    let targets: smallvec::SmallVec<[CreatureUid; 4]> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|e| {
            state
                .creatures
                .iter()
                .any(|c| c.id == e.0 && c.current_hp > 0)
        })
        .collect();
    // One multi-target `CreatureCmd.Damage` (SmokestackPower.cs): its kills
    // die after the last hit.
    damage::with_deferred_enemy_deaths(state, |state| {
        for target in targets {
            let _ = crate::damage::apply_damage(
                state,
                crate::hooks::CreatureRef::Player,
                target,
                amount,
                crate::hooks::ValueProps::UNPOWERED,
                None,
            );
        }
    });
}

/// C# `TrashToTreasurePower.AfterCardGeneratedForCombat`
/// (`MegaCrit.Sts2.Core.Models.Powers/TrashToTreasurePower.cs:16-27`): whenever
/// the PLAYER generates a Status card (`card.Type == Status && creator != null
/// && creator.Creature == Owner`), channel `Amount` random orbs —
/// `OrbModel.GetRandomOrb(Rng.CombatOrbGeneration)` per orb, then
/// `OrbCmd.Channel`. Defect's Turbo (Void) / Overclock (Burn) are the common
/// triggers, exactly as for Smokestack, and the card itself only grants the
/// power (`TrashToTreasure.cs:20-24`).
///
/// `added_by_player` carries the C# `creator` test, approximated at the spawn
/// sites the same way [`fire_smokestack_on_status_generation`] does. The roll
/// goes through [`crate::orbs::random_combat_orb`] so the 5-orb table stays
/// shared with Chaos, and the channel through [`crate::orbs::channel_orb`] so
/// a full belt overflow-evokes instead of growing. GH #639.
pub(crate) fn fire_trash_to_treasure_orbs_on_status_generation(
    state: &mut CombatState,
    uid: CardUid,
    added_by_player: bool,
) {
    if !added_by_player {
        return;
    }
    let Some(card) = state.cards.get(uid.0.get() as usize - 1) else {
        return;
    };
    if card.card_type != crate::cards::CardType::Status {
        return;
    }
    let amount = state.player_power_amount(crate::PowerKind::TrashToTreasure);
    if amount <= 0 {
        return;
    }
    // C# `Flash()` sits inside the same guard, ABOVE the channel loop, so it
    // fires once per generation and not once per orb. The `amount <= 0` return
    // is the WraithForm under-fire deviation shared with Smokestack.
    crate::event_log::flash_player_power(state, crate::PowerKind::TrashToTreasure);
    for _ in 0..amount {
        let orb = crate::orbs::random_combat_orb(state);
        crate::orbs::channel_orb(state, orb);
    }
}

/// C# `RocketPunch.AfterCardGeneratedForCombat`
/// (`MegaCrit.Sts2.Core.Models.Cards/RocketPunch.cs`): RocketPunch is a CARD
/// listener (not a power) — whenever the PLAYER generates a Status card
/// (`addedByPlayer && card.Type == Status && card.Owner == this.Owner`), every
/// RocketPunch the player holds calls `EnergyCost.SetUntilPlayed(0)` on itself,
/// becoming free until it is next played. The classic trigger is
/// TeaOfDiscourtesy seeding 2 Dazed into the draw pile at combat start
/// (`addedByPlayer: true`, empty Play pile) — parity fuzz Defect seed 111
/// (`hand_costs` C#=0 rust=2 for the ROCKET_PUNCH in the opening hand).
///
/// Unlike the power reactions (Arsenal/Pillar/Smokestack) this scans the
/// player's piles for the RocketPunch instances and stamps each one's
/// per-instance `SetUntilPlayed(0)` modifier marker. Owner-equality
/// is implicit: only player-owned cards live in the player piles, and the
/// `added_by_player` gate already excludes monster-generated statuses (Wound,
/// Slimed) whose generation is not "by the player". No-op for any combat
/// without a RocketPunch in a player pile.
pub(crate) fn fire_rocketpunch_free_on_status_generation(
    state: &mut CombatState,
    uid: CardUid,
    added_by_player: bool,
) {
    if !added_by_player {
        return;
    }
    let Some(card) = state.cards.get(uid.0.get() as usize - 1) else {
        return;
    };
    if card.card_type != crate::cards::CardType::Status {
        return;
    }
    // Scan every player pile for RocketPunch instances and make each free
    // until played. Collect uids first to avoid borrowing `state` mutably
    // while iterating its pile vecs.
    let mut rockets: smallvec::SmallVec<[CardUid; 4]> = smallvec::SmallVec::new();
    let pile_uids = state
        .player
        .draw
        .iter()
        .chain(state.player.hand.iter())
        .chain(state.player.discard.iter())
        .chain(state.player.exhaust.iter())
        .copied();
    for cu in pile_uids {
        if let Some(c) = state.cards.get(cu.0.get() as usize - 1) {
            if c.card_id == crate::cards::CardId::RocketPunch {
                rockets.push(cu);
            }
        }
    }
    let v0111 = state.game_version >= crate::GameVersion::V0_111_0;
    for cu in rockets {
        if v0111 {
            // v0.111.0: `EnergyCost.AddUntilPlayed(-1)` — one more energy
            // off per generated Status, until played (decomp
            // RocketPunch.cs:49).
            state.add_card_cost_reduction_until_played(cu, 1);
        } else if let Some(c) = state.cards.get_mut(cu.0.get() as usize - 1) {
            // v0.107.1: `EnergyCost.SetUntilPlayed(0)` (RocketPunch.cs:49).
            c.cost.set_until_played(0, false);
        }
    }
}

/// Build a fresh [`CardId::Wound`] POD with the canonical Wound stats
/// (cost `-1`, `CardType::Status`, `TargetType::None`, no tags,
/// `UNPLAYABLE`). Mirrors the content-side `specs::wound()` factory —
/// duplicated in core because `sts2-content` is only a dev-dependency of
/// `sts2-core`, so a production core path (e.g. an enemy power's
/// `after_attack` reaction) cannot reach `cards::status::spawn_wound`.
/// Keep the field values in sync with
/// `crates/sts2-content/src/cards/status_specs_generated.rs::wound()`,
/// which is hand-maintained against C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Wound.cs`.
#[must_use]
fn wound_card_pod() -> crate::cards::Card {
    crate::cards::Card {
        card_id: crate::cards::CardId::Wound,
        cost: crate::CardEnergyCost::new(-1),
        card_type: crate::cards::CardType::Status,
        target_type: crate::cards::TargetType::None,
        ..crate::cards::Card::pod_defaults()
    }
}

/// Spawn one [`CardId::Wound`] into `pile`, allocating a fresh arena slot
/// and firing the `after_card_entered_combat` buses (same path as the
/// content-side `cards::status::spawn_wound`). Provided in core so engine
/// powers that add Wounds in the live damage path (`PainfulStabsPower`)
/// don't need the content crate. Bead `sts2-rs-rb25f`.
pub fn spawn_wound_into_pile(state: &mut CombatState, pile: PileKind) -> CardUid {
    spawn_card_into_pile_with_origin(state, wound_card_pod(), pile, false)
}

/// Build a fresh [`CardId::Wither`] POD (v0.107.1 Aeonglass status card:
/// cost `-1`, `CardType::Status`, `TargetType::None`, UNPLAYABLE). Duplicated
/// in core (cf. [`wound_card_pod`]) so the engine-side `WitheringPresence`
/// power reaction — which lives in `crate::hooks` and cannot reach
/// `sts2-content` — can add a Wither to the player's hand. Keep in sync with
/// `crates/sts2-content/src/cards/status_specs_generated.rs::wither()`. Bead
/// `sts2-rs-v0107-content-port-wb8lk.2`.
#[must_use]
fn wither_card_pod() -> crate::cards::Card {
    crate::cards::Card {
        card_id: crate::cards::CardId::Wither,
        cost: crate::CardEnergyCost::new(-1),
        card_type: crate::cards::CardType::Status,
        target_type: crate::cards::TargetType::None,
        ..crate::cards::Card::pod_defaults()
    }
}

/// Spawn one [`CardId::Wither`] into `pile`, firing the
/// `after_card_entered_combat` buses (same path as the content-side
/// `cards::status::spawn_wither`); the generated-for-combat dispatch matches
/// it to the Aeonglass `WitherUpgradeCount`.
/// Bead `sts2-rs-v0107-content-port-wb8lk.2`.
pub fn spawn_wither_into_pile(state: &mut CombatState, pile: PileKind) -> CardUid {
    spawn_card_into_pile_with_origin(state, wither_card_pod(), pile, false)
}

/// Fire the `after_card_entered_combat` relic + monster-power buses for a
/// freshly-spawned card, matching C# `CardPileCmd.Add`'s
/// `if (oldPile == null && targetPile.IsCombatPile)` gate
/// (`CardPileCmd.cs:436-439`): the hook fires once when a *newly created*
/// card lands in any combat pile, regardless of which pile. The
/// `spawn_card_into_*` helpers only ever materialize brand-new cards
/// (`oldPile == null`), so the gate reduces to "is this a combat pile".
/// Exhaust is not a combat pile for this purpose in C# (cards routed
/// straight to Exhaust on creation are rare and not card-entered events),
/// so it is excluded — matching the prior `spawn_card_into_hand`-only
/// coverage plus Draw/Discard. Bead `sts2-rs-11ssi`: Regalite (+2 Block
/// per owner card entering combat) and HungerPower were missing the
/// Strangler-spawned status card because pile spawns bypassed this bus.
fn fire_after_card_entered_combat(
    state: &mut CombatState,
    uid: CardUid,
    pile: PileKind,
    added_by_player: bool,
) {
    if matches!(pile, PileKind::Exhaust) {
        return;
    }
    // C# `Hook.AfterCardEnteredCombat` dispatches through
    // `IterateCombatHookListeners`, which yields NO listeners once combat is
    // over or ending (`Hook.cs:177-207`, `:367-374`).
    if crate::combat_loop::combat_is_ending(state) {
        return;
    }
    let _ = crate::relic_effects::dispatch_after_card_entered_combat(state, uid, added_by_player);
    let _ = crate::monster_power_effects::dispatch_after_card_entered_combat_on_monster_powers(
        state, uid,
    );
    // C# `PhantomBladesPower.AfterCardEnteredCombat`: a Shiv-tagged owned card
    // entering combat while PhantomBlades is active gains Retain. Covers Shivs
    // GENERATED after the power was applied (Blade Dance / Cloak & Dagger /
    // Storm of Steel); the AfterApplied grant on the PhantomBlades card handles
    // Shivs already in combat. reshuffle-pilecount PhantomBlades sub-pop.
    if state.player_power_amount(crate::PowerKind::PhantomBlades) > 0 {
        let idx = uid.0.get() as usize - 1;
        if state
            .cards
            .get(idx)
            .is_some_and(|c| c.tags.contains(crate::cards::CardTag::Shiv))
        {
            if let Some(card) = state.cards.get_mut(idx) {
                apply_retain_keyword(card);
            }
        }
    }
    stamp_entered_combat_cost_backfill(state, uid);
}

/// Card-level `AfterCardEnteredCombat` back-fill: C# `Stomp.cs:45-58`
/// reduces a Stomp that ENTERS combat mid-turn (CalamityPower / Stoke /
/// Discovery generations) by the Attack plays already FINISHED this turn,
/// so a generated Stomp costs the same as one that sat in hand through
/// those plays. `Pinpoint.AfterCardEnteredCombat` is the Skill-keyed
/// twin; the shave is an `AddThisTurn` like the play-time discounts (sweep
/// seeds 2725/5546: a Calamity/Stoke-generated STOMP stayed cost 3 in
/// Rust while C# offered it at 3 minus attacks-this-turn). Called from
/// [`fire_after_card_entered_combat`]; content sites that dispatch the
/// entered-combat buses directly (Stoke / Discovery) must call this too.
/// Bead `sts2-rs-1hlal`.
pub fn stamp_entered_combat_cost_backfill(state: &mut CombatState, uid: CardUid) {
    let idx = uid.0.get() as usize - 1;
    let Some(card) = state.cards.get(idx) else {
        return;
    };
    let plays = match card.card_id {
        crate::cards::CardId::Stomp => {
            state.player.cards_played_this_turn[crate::cards::CardType::Attack.index()]
        }
        crate::cards::CardId::Pinpoint => {
            state.player.cards_played_this_turn[crate::cards::CardType::Skill.index()]
        }
        // A combat-long shave, not a per-turn one.
        crate::cards::CardId::Midnight => {
            crate::hooks::midnight_entered_combat_v0111(state, uid);
            return;
        }
        // Also combat-long.
        crate::cards::CardId::BansheesCry => {
            banshees_cry_entered_combat(state, uid);
            return;
        }
        // `Flatten.AfterCardEnteredCombat`: `SetThisTurn(0)` when the Osty
        // has already attacked this turn (no clone check).
        crate::cards::CardId::Flatten => {
            if state.combat_log.osty_attacks_this_turn > 0 {
                state.cards[idx].cost.set_this_turn(0, false);
            }
            return;
        }
        _ => return,
    };
    // `Stomp.cs:53` / `Pinpoint.cs:43`: `if (base.IsClone) return`. A clone
    // keeps the cost it was cloned with. A dupe is a clone too
    // (`CreateDupe` calls `CreateClone`). Both builds (GH #1094: this used
    // to guess "clone" from a non-zero per-turn bank).
    if card.is_clone() || card.combat_var(crate::cards::VarKey::IsDupe) != 0 {
        return;
    }
    // `ReduceCostBy(plays)` is `EnergyCost.AddThisTurn(-plays)`, appended
    // after whatever the card already carries (Infernal Blade's
    // `SetToFreeThisTurn` runs BEFORE `AddGeneratedCardToCombat`,
    // `InfernalBlade.cs:26-27`, so a free Stomp goes below 0).
    let plays = i32::try_from(plays).unwrap_or(i32::MAX);
    state.cards[idx].cost.add_this_turn(-plays);
}

/// Banshee's Cry's `AfterCardEnteredCombat` (`BansheesCry.cs:42-55`): a
/// Banshee's Cry that enters combat after Ethereal plays (CallOfTheVoid,
/// Discovery-style generations, a transform) starts 2 cheaper per Ethereal
/// card its owner already played this combat.
///
/// [decompiled C# omitted — see the cited path]
///
/// The engine has no clone flag, so the shave covers only the plays the card
/// has not yet counted ([`crate::cards::VarKey::BansheesCryEtherealCounted`]):
/// a fresh card counts none, and a copy of one already in combat carries the
/// original's count. Unclamped like every `AddThisCombat`
/// ([`crate::cards::Card::add_cost_this_combat`]).
fn banshees_cry_entered_combat(state: &mut CombatState, uid: CardUid) {
    let total =
        i32::try_from(state.combat_log.ethereal_cards_played_this_combat).unwrap_or(i32::MAX);
    let Some(c) = state.cards.get_mut(uid.0.get() as usize - 1) else {
        return;
    };
    let key = crate::cards::VarKey::BansheesCryEtherealCounted;
    let missed = total - c.combat_var(key);
    if missed > 0 {
        c.set_combat_var(key, total);
        // Crossbow's `SetToFreeThisTurn` runs BEFORE
        // `AddGeneratedCardsToCombat` (Crossbow.cs:17-37), so this
        // `AddThisCombat` folds after that local zero: `0 - 2n`, and
        // Borrowed Time's +1 skips the negative cost (Necrobinder 6580
        // v0.107.1: C# 0, Rust 1).
        state.add_card_cost_this_combat(uid, -missed.saturating_mul(2));
    }
}

/// Allocate a fresh [`CardUid`] for `card` and insert it into the
/// player's draw pile at a uniformly-random index, consuming exactly one
/// draw from the **Shuffle** RNG stream.
///
/// Mirrors C# `CardPileCmd.Add(.., CardPilePosition.Random)`
/// (`CardPileCmd.cs:433`):
/// [decompiled C# omitted — see the cited path]
/// `next_int(0, len)` is inclusive on both ends, so it yields `len + 1`
/// slots (incl. a true append) — matching `NextInt(Count + 1)`. The index
/// is sampled against `draw.len()` *before* this card is added, so callers
/// that spawn several cards in a loop see the pile grow by one each
/// iteration, exactly like C# (`AddGeneratedCardToCombat` per-card adds).
///
/// Unlike [`move_card_to_pile`]'s `random_position` branch (which reads the
/// `combat_card_selection` stream and skips the draw on an empty pile — a
/// Python-pin shape), this uses the C#-authoritative Shuffle stream and
/// always consumes the read, even when the draw pile is empty
/// (`NextInt(1)` still advances the cursor). See
/// [`CombatState::insert_card_into_draw_pile`] for the move-an-existing-card
/// analogue. Returns the assigned `CardUid`.
///
/// Concrete consumers: `FuneraryMask` (3 Souls) and `TeaOfDiscourtesy`
/// (2 Dazed) — relic listeners that generate cards directly into the draw
/// pile at a random depth. Bead `sts2-rs-n4sa4`.
pub fn spawn_card_into_draw_pile_random_position(
    state: &mut CombatState,
    card: crate::cards::Card,
) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    let mut card = card;
    card.uid = uid;
    // C# `CardModel.DeckVersion == null` marker (see `spawn_card_into_pile`).
    stamp_generated_in_combat(&mut card);
    // C# `CardPileCmd.Add(.., CardPile, ..)` short-circuits with `success=false`
    // BEFORE the `CardPilePosition.Random` shuffle read
    // (`card.Owner.RunState.Rng.Shuffle.NextInt(targetPile.Cards.Count + 1)`,
    // `CardPileCmd.cs:433`) when `newPile.IsCombatPile && CombatManager.IsEnding`
    // (`CardPileCmd.cs:240`). This helper is the ONLY spawn primitive that
    // consumes the run-level `Rng.Shuffle` stream, so without this gate a lethal
    // soul-spawn OnPlay (Severance's killing blow) over-reads the shuffle by one
    // and desyncs every later combat's opening StableShuffle -- the same failure
    // mode `combat_is_ending` already guards for the mid-draw reshuffle
    // (PARITY128 Necrobinder, sibling of PARITY86 Ironclad). Allocate the uid +
    // push to the arena (mirroring C#'s `cardAdded = c` with `success = false`,
    // so the returned uid stays dereferenceable) but DO NOT read the shuffle
    // stream, insert into the draw pile, or fire any post-generation hooks.
    if crate::combat_loop::combat_is_ending(state) {
        state.cards.push(card);
        return uid;
    }
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    card.normalize_for_version(state.game_version);
    state.cards.push(card);
    // Bead `sts2-rs-09x5n` (F2 `HiddenInfoMode::BeliefDraw`): see the
    // matching gate in `spawn_card_into_draw_pile_random` above.
    if state.player.deferred_draw_order {
        crate::cards::insert_into_unknown_sorted(&mut state.player, uid);
    } else {
        let len = state.player.draw.len();
        let idx = state.run_rng.shuffle.next_int(0, len as i32) as usize;
        state.player.draw.insert(idx, uid);
    }
    // Every caller passes `creator: Owner` (see below), so the generation counts
    // toward Supermassive — TeaOfDiscourtesy's Dazed and FuneraryMask's Souls
    // were missing from the tally (Regent full seeds 1401/2143/3222/3790/...).
    record_player_generated_card(state);
    fire_after_card_entered_combat(state, uid, PileKind::Draw, true);
    // C# `AddGeneratedCardToCombat` fires `AfterCardGeneratedForCombat` on every
    // listener. Every current caller of this random-position primitive passes
    // `addedByPlayer: true` — Necrobinder soul OnPlays (GraveWarden, GlimpseBeyond,
    // CaptureSpirit, Dirge, Severance), and the combat-start relics
    // TeaOfDiscourtesy (2 Dazed) / FuneraryMask (souls). TeaOfDiscourtesy fires
    // with an EMPTY Play pile (combat start), so the spawn-path play-pile
    // heuristic would wrongly read `false`; pass `true` explicitly. Without this,
    // a RocketPunch in the opening hand stayed at full cost despite Tea's Dazed
    // making it free (parity fuzz Defect seed 111). Bead
    // `sts2-rs-rocketpunch-statusgen`.
    fire_card_generated_for_combat_powers(state, uid, true);
    uid
}

/// Allocate a fresh [`CardUid`] for `card` and append it to the BOTTOM of
/// the draw pile (drawn last). This is C# `AddGeneratedCard(s)ToCombat`'s
/// DEFAULT position — `CardPilePosition.Bottom` (`CardPileCmd.cs:172/177`),
/// which `AddInternal` maps to index `-1` = append (`CardPileCmd.cs:431`).
/// No RNG read.
///
/// Distinct from [`spawn_card_into_pile`]`(.., PileKind::Draw)`, whose
/// top-insert mirrors the Python pin — a generated card spawned through
/// that helper is drawn FIRST, flipping the next hands' draw order vs C#
/// (run-parity Necrobinder RP5: Reave's Soul drawn turn-3 slot 2 in Rust
/// vs last in C#). Ports of C# sites that omit the `position` argument
/// must use this bottom variant.
pub fn spawn_card_into_draw_pile_bottom(
    state: &mut CombatState,
    card: crate::cards::Card,
) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    let mut card = card;
    card.uid = uid;
    // C# `CardModel.DeckVersion == null` marker (see `spawn_card_into_pile`).
    stamp_generated_in_combat(&mut card);
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    card.normalize_for_version(state.game_version);
    state.cards.push(card);
    if !combat_pile_add_blocked(state) {
        state.player.draw.push(uid);
    }
    // Player-created (Reave / soul OnPlays, `creator: Owner`): counts toward
    // Supermassive (`CardPileCmd.cs:243`, `Supermassive.cs:22-27`).
    record_player_generated_card(state);
    fire_after_card_entered_combat(state, uid, PileKind::Draw, true);
    uid
}

/// Allocate a fresh [`CardUid`] for `card` and append it to the player's
/// hand. If the hand is already at [`crate::cards::HAND_LIMIT`] (Python's
/// `MAX_HAND_SIZE = 10`), the new card is routed to the discard pile
/// instead — matching Python's `move_card_to_creature_hand` overflow.
///
/// This is the "materialize a freshly-generated card into hand" primitive
/// — colorless modifiers, sovereign-blade scans, `CollisionCourse` Debris,
/// `Hotfix`, `Largesse`, etc. Distinct from [`spawn_card_into_pile`] with
/// `PileKind::Hand` (which unconditionally pushes into hand and would
/// violate the cap); callers that need overflow-into-discard semantics
/// should prefer this helper.
///
/// Mirrors `sts2-rl-agent@01c094d:sts2_env/core/combat.py::
/// CombatState.move_card_to_creature_hand` (lines 1682–1691):
/// ```text
///     if len(zones["hand"]) < MAX_HAND_SIZE:
///         zones["hand"].append(card)
///     else:
///         zones["discard"].append(card)
/// ```
/// Bead `sts2-rs-7up3`.
pub fn spawn_card_into_hand(state: &mut CombatState, card: crate::cards::Card) -> CardUid {
    // C# `addedByPlayer` is supplied per call site; the common direct-into-hand
    // generations here are card OnPlay bodies (ManifestAuthority / DualWield /
    // JackOfAllTrades / …), so the play-pile heuristic ("a card is mid-play")
    // matches C#'s explicit `true` for them. Callers whose C# `addedByPlayer`
    // is unconditionally `true` even with an EMPTY play pile (potion-generated
    // cards, whose use puts nothing on the play pile) must instead call
    // [`spawn_card_into_hand_with_origin`] with `true` — see
    // `resolve_add_generated_choice`. Bead `sts2-rs-potion-gen-arsenal`.
    let added_by_player = !state.player.play.is_empty();
    spawn_card_into_hand_with_origin(state, card, added_by_player)
}

/// [`spawn_card_into_hand`] with an EXPLICIT C# `addedByPlayer` flag, instead of
/// the play-pile heuristic. Use this when the generating effect is a player
/// action that nonetheless leaves the play pile empty — chiefly a potion whose
/// `OnUse` calls `CardPileCmd.AddGeneratedCardToCombat(.., addedByPlayer: true)`
/// (SkillPotion / AttackPotion / ColorlessPotion). The flag gates the
/// `AfterCardGeneratedForCombat` reactions (`ArsenalPower` Strength,
/// `PillarOfCreationPower` Block), so a wrong `false` silently drops them.
/// Bead `sts2-rs-potion-gen-arsenal`.
pub fn spawn_card_into_hand_with_origin(
    state: &mut CombatState,
    card: crate::cards::Card,
    added_by_player: bool,
) -> CardUid {
    spawn_card_into_hand_inner(state, card, Some(added_by_player))
}

/// A new card put into hand with a plain C# `CardPileCmd.Add(card, Hand)`
/// rather than `AddGeneratedCardToCombat`: it enters combat (the
/// entered-combat hooks fire) but is not "generated", so no
/// `History.CardGenerated` tally and no `AfterCardGeneratedForCombat`
/// (Regalite, Arsenal, ...). Mad Science's Chaos rider (MadScience.cs:286-288;
/// loadout Defect 1221 v0.107.1: Rust's Regalite gave 2 block).
pub fn add_new_card_to_hand(state: &mut CombatState, card: crate::cards::Card) -> CardUid {
    spawn_card_into_hand_inner(state, card, None)
}

/// `generated`: `Some(added_by_player)` for `AddGeneratedCardToCombat`,
/// `None` for a plain `CardPileCmd.Add`.
fn spawn_card_into_hand_inner(
    state: &mut CombatState,
    card: crate::cards::Card,
    generated: Option<bool>,
) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let uid = CardUid::new(raw).expect("non-zero uid");
    let mut card = card;
    card.uid = uid;
    // C# `CardModel.DeckVersion == null` marker (see `spawn_card_into_pile`).
    stamp_generated_in_combat(&mut card);
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    card.normalize_for_version(state.game_version);
    state.cards.push(card);
    // Bead `sts2-rs-8h2q`: log the generation *before* the hand-cap
    // check, matching Python's `_add_generated_cards_to_hand` at
    // `sts2-rl-agent@01c094d:sts2_env/core/combat.py:2260` (the
    // `_generated_cards_combat.append(target)` runs unconditionally —
    // overflow-to-discard still counts). Read by Supermassive via
    // `CombatState::count_player_generated_cards_this_combat`.
    if generated.is_some() {
        state.player.generated_cards_this_combat =
            state.player.generated_cards_this_combat.saturating_add(1);
    }
    if combat_pile_add_blocked(state) {
        // `CardPileCmd.Add` fails once the combat is ending.
    } else if state.player.hand.len() < crate::cards::HAND_LIMIT {
        state.player.hand.push(uid);
    } else {
        state.player.discard.push(uid);
    }
    // C# `IterateCombatHookListeners` gate (see
    // `fire_after_card_entered_combat`): no entered-combat listeners once
    // combat is over or ending.
    if !crate::combat_loop::combat_is_ending(state) {
        // Bead `sts2-rs-x2rql`: relic-side `after_card_entered_combat`
        // fires AFTER the spawn is committed (matches C# `Hook.AfterCard
        // EnteredCombat` post-`CardPileCmd` ordering — listeners observe a
        // state where the new uid is already in its destination zone).
        // Its one listener, Regalite, is C#'s `AfterCardGeneratedForCombat`,
        // so a plain `CardPileCmd.Add` (`generated == None`) skips it.
        if generated.is_some() {
            let _ = crate::relic_effects::dispatch_after_card_entered_combat(state, uid, true);
        }
        // Bead `sts2-rs-ig66w`: sibling monster-power dispatch lane (9l5c2
        // sub-2). Walks every alive creature's `powers` vector and routes
        // through the per-`PowerKind` registry — fires HungerPower's
        // `AfterCardEnteredCombat` hook when Doormaker is mid-phase-Hunger.
        // C# `HungerPower.cs:42-52` runs the same per-card afflict check
        // after each new card materializes.
        let _ = crate::monster_power_effects::dispatch_after_card_entered_combat_on_monster_powers(
            state, uid,
        );
        // Card-level `AfterCardEnteredCombat` back-fill (bead `sts2-rs-1hlal`): a
        // generated Stomp/Pinpoint materialized straight into hand (Discovery's
        // choose-a-card resolver, potion/effect generations) back-fills this
        // turn's play discounts. No-op for every other card. Mirrors the bundle
        // in `fire_after_card_entered_combat`, which the inline dispatch above
        // otherwise duplicates minus this call.
        stamp_entered_combat_cost_backfill(state, uid);
    }
    // C# `AfterCardGeneratedForCombat` fires for EVERY player-generated card
    // regardless of destination pile (`CardPileCmd.AddGeneratedCardToCombat`).
    // ManifestAuthority / Largesse etc. generate straight into hand, so the
    // whole listener bundle must fire here too (not only in
    // `spawn_card_into_pile`). Regent parity fuzz seeds 104/400/427/438/495.
    // `added_by_player` is supplied by the caller (the play-pile heuristic for
    // the public `spawn_card_into_hand`, or an explicit flag for potion-
    // generated cards — see `spawn_card_into_hand_with_origin`).
    // Until GH #639 this site called only the Arsenal / PillarOfCreation pair,
    // so a Status generated straight into hand silently reached neither
    // Smokestack nor RocketPunch.
    if let Some(added_by_player) = generated {
        fire_card_generated_for_combat_powers(state, uid, added_by_player);
    }
    uid
}

// ---------------------------------------------------------------------------
// Cross-pile scan + move-to-hand (bead sts2-rs-9dj5)
// ---------------------------------------------------------------------------
//
// Companion to [`search_pile`] / [`spawn_card_into_hand`]. The Python pin
// at `sts2-rl-agent@01c094d:sts2_env/cards/regent.py::summon_forth`
// (L558-L568) iterates `(combat.draw_pile, combat.discard_pile,
// combat.exhaust_pile)`, snapshots the matches per pile, then calls
// `combat.move_card_to_creature_hand(owner, blade)` on each match —
// which **removes the card from its source pile** before re-inserting it
// into hand (or routing to discard on overflow).
//
// We mirror those two steps for every pile in `piles` order:
//   1. Snapshot the matching `CardUid`s in canonical pile order.
//   2. For each match: remove the uid from its source pile, then push
//      onto `hand` (or `discard` if hand is at `HAND_LIMIT`).
//
// The `owner` parameter is reserved for future multi-owner support —
// the current Rust port models a single-player perspective so every
// card on the player's piles is owned by the player. Mirroring the
// Python signature now means the caller doesn't churn when ownership
// lands later.

/// Scan `piles` for cards matching `predicate` and move every match into
/// the player's hand, removing each from its source pile. Returns the
/// total number of cards moved.
///
/// Each per-pile snapshot is taken **before** mutation, so the helper
/// is robust to in-place removals (the iteration index is recomputed
/// against the live pile on each removal). Cards that overflow the
/// hand cap ([`crate::cards::HAND_LIMIT`]) are routed to `discard`
/// instead — same overflow semantics as
/// [`spawn_card_into_hand`] and Python's
/// `move_card_to_creature_hand` (`sts2_env/core/combat.py:1682-1691`).
///
/// `owner` is currently advisory — the Rust port models the player's
/// piles directly. The parameter is kept for parity with the Python
/// signature so callers don't churn when multi-owner ownership lands.
///
/// Bead `sts2-rs-9dj5`. Unblocks `SummonForth` (Regent rare) which
/// pulls every `SovereignBlade` across draw / discard / exhaust into
/// hand on play.
pub fn scan_piles_and_move_to_hand<F>(
    state: &mut CombatState,
    _owner: CreatureUid,
    piles: &[PileKind],
    mut predicate: F,
) -> usize
where
    F: FnMut(&crate::cards::Card) -> bool,
{
    let mut moved = 0_usize;
    for &pile in piles {
        // Snapshot matches per pile (Python iterates a `list(pile)`
        // copy then calls `move_card_to_creature_hand` per match;
        // the snapshot is what makes the removal-during-iteration safe).
        let mut matches: smallvec::SmallVec<[CardUid; 8]> = smallvec::SmallVec::new();
        for &uid in pile_uids(state, pile) {
            let idx = uid.0.get() as usize - 1;
            if let Some(card) = state.cards.get(idx) {
                if predicate(card) {
                    matches.push(uid);
                }
            }
        }
        for uid in matches {
            // Remove from source pile (skip if a prior iteration on
            // some weird re-entrant predicate already evicted it).
            let removed = remove_uid_from_pile(state, pile, uid);
            if !removed {
                continue;
            }
            // Re-insert into hand with overflow-to-discard cap.
            if state.player.hand.len() < crate::cards::HAND_LIMIT {
                state.player.hand.push(uid);
            } else {
                state.player.discard.push(uid);
            }
            moved += 1;
        }
    }
    moved
}

fn remove_uid_from_pile(state: &mut CombatState, pile: PileKind, uid: CardUid) -> bool {
    match pile {
        PileKind::Draw => {
            if let Some(pos) = state.player.draw.iter().position(|&u| u == uid) {
                state.player.draw.remove(pos);
                return true;
            }
        }
        PileKind::Discard => {
            if let Some(pos) = state.player.discard.iter().position(|&u| u == uid) {
                state.player.discard.remove(pos);
                return true;
            }
        }
        PileKind::Exhaust => {
            if let Some(pos) = state.player.exhaust.iter().position(|&u| u == uid) {
                state.player.exhaust.remove(pos);
                return true;
            }
        }
        PileKind::Hand => {
            if let Some(pos) = state.player.hand.iter().position(|&u| u == uid) {
                state.player.hand.remove(pos);
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Hand-mutation + cross-pile move primitives (bead sts2-rs-6ynt.2)
// ---------------------------------------------------------------------------
//
// Surface for cards/events/relics that need to mutate the *identity* of an
// existing card in some pile (transform / upgrade) or move a card across
// piles without going through draw (Headbutt → top-of-draw, Exhume →
// exhaust-to-hand, PrimalForce → transform-to-GiantRock, Stoke → exhaust
// each card in hand and draw one back).
//
// Design choices:
//   * No new "command" enum is introduced — sts2-core does not have a
//     deferred command queue. All mutators are direct `&mut CombatState`
//     functions, exactly like the existing `spawn_card_into_*` family
//     above. The two existing dispatch surfaces (the on-play registry
//     below and the typed hook buses in `crate::hooks` /
//     `crate::relic_effects`) cover deferred / fan-out work.
//   * `find_card_pile` / `remove_card_from_any_pile` mirror Python's
//     `_remove_card_from_piles` (`combat.py:1755-1760`) — they walk the
//     four player piles in canonical order and report / strip the first
//     match. `Play` is intentionally NOT in the search set: a mid-resolve
//     card on the play pile must finish dispatch before it can be
//     transformed (the engine moves Play→Discard at the end of
//     `play_card`).
//   * `move_card_to_pile` removes the card from any source pile and
//     re-inserts into `dest`, with the same overflow-into-discard
//     semantics for `PileKind::Hand` that `spawn_card_into_hand` and
//     `scan_piles_and_move_to_hand` already use. For `PileKind::Draw`
//     a `random_position` flag picks the insert index uniformly via
//     `state.rng` (mirrors `insert_card_into_creature_draw_pile` at
//     `combat.py:1737-1753`); the default is top-of-draw (`insert(0,…)`).
//   * `exhaust_card_uid` is the surgical "exhaust this specific card
//     wherever it lives" primitive. It removes from any source pile,
//     pushes onto exhaust, and dual-fires `after_card_exhausted` on
//     both the hooks bus (powers — FeelNoPain, DarkEmbrace) and the
//     relic-effects bus (CharonsAshes), preserving the canonical
//     hooks-then-relics order locked in by `CombatState::exhaust_from_draw_pile`
//     (lib.rs:1773-1824). Used by Stoke / future event-spawned exhausts.
//   * `transform_card_in_place` swaps the card POD at the source pile's
//     index for a freshly-built `Card` POD allocated in `state.cards`.
//     The pile slot's `CardUid` is updated to point at the new arena
//     row; the old arena row is left in place (no compaction). Mirrors
//     Python `transform_card` (`combat.py:1801-1820`) with one
//     deliberate divergence: Python re-uses the *list slot* by
//     replacing `target_pile[target_index] = new_card`; we keep the
//     list slot but allocate a new arena row, because every other
//     non-trivial Card field already lives in the arena. Returns the
//     new `CardUid` so the caller can bookkeep (transform-into-Shiv
//     parity tests, attached-power references on Genetic-Algorithm-
//     style enchantments).
//   * `upgrade_card_uid` / `downgrade_card_uid` are thin wrappers around
//     `Card::upgrade` (idempotent flag flip per bead `sts2-rs-01o`'s
//     surface) and the matching downgrade (clears the flag). The full
//     spec-lookup re-fetch (Python `upgrade_card`, `combat.py:1768`)
//     lives in sts2-content; consumer beads will pair `upgrade_card_uid`
//     with `transform_card_in_place(uid, card_from_spec(&upgraded_spec))`
//     when they need the field-level rewrite.
//
// The "RNG hook for random spawns" mentioned in bead 6ynt.2 lands on
// `move_card_to_pile` via the `random_position` flag (uses `state.rng`
// directly — same shape as `pick_random_alive_enemy` above). A future
// `spawn_card_into_pile_random_position` wrapper can be added when a
// concrete consumer (`InsertIntoCreatureDrawPile(random_position=True)`
// users) needs it; the random-position semantics are already covered
// by `move_card_to_pile`'s post-spawn relocation.

/// Locate the player pile that currently contains `uid`, scanning in
/// the canonical order Hand → Draw → Discard → Exhaust. Returns `None`
/// if the card is mid-resolve (in `Play`) or otherwise absent. Mirrors
/// the read-side of Python `_remove_card_from_piles` at
/// `sts2-rl-agent@01c094d:sts2_env/core/combat.py:1755-1760`.
///
/// Bead `sts2-rs-6ynt.2`. Used by every helper below that needs to
/// look up a card's home pile before mutating.
#[must_use]
pub fn find_card_pile(state: &CombatState, uid: CardUid) -> Option<PileKind> {
    if state.player.hand.contains(&uid) {
        return Some(PileKind::Hand);
    }
    if state.player.draw.contains(&uid) {
        return Some(PileKind::Draw);
    }
    if state.player.discard.contains(&uid) {
        return Some(PileKind::Discard);
    }
    if state.player.exhaust.contains(&uid) {
        return Some(PileKind::Exhaust);
    }
    None
}

/// Remove `uid` from whichever player pile contains it (Hand / Draw /
/// Discard / Exhaust). Returns the source `PileKind` on success, or
/// `None` if the card was not found in any non-`Play` pile. The arena
/// entry in `state.cards` is left intact (uids remain stable for any
/// listener still holding a reference).
///
/// Mirrors Python `_remove_card_from_piles` at `combat.py:1755-1760`.
/// Bead `sts2-rs-6ynt.2`.
pub fn remove_card_from_any_pile(state: &mut CombatState, uid: CardUid) -> Option<PileKind> {
    let src = find_card_pile(state, uid)?;
    if remove_uid_from_pile(state, src, uid) {
        Some(src)
    } else {
        None
    }
}

/// Move `uid` from whichever pile currently holds it to `dest`. Returns
/// the source pile on success, or `None` if `uid` was not found in any
/// player pile.
///
/// Destination semantics:
///   * `Hand` — appended; if hand is at [`crate::cards::HAND_LIMIT`],
///     routed to discard instead (matches Python's
///     `move_card_to_creature_hand` overflow at `combat.py:1682-1691`).
///   * `Draw` — `insert(0, uid)` (top of pile, next-drawn) when
///     `random_position == false`; uniform-random index in
///     `[0, draw.len()]` (inclusive — `insert` accepts `==len`) when
///     `random_position == true`, sampled from `state.rng`. Mirrors
///     `insert_card_into_creature_draw_pile(card, random_position=…)`
///     at `combat.py:1737-1753`.
///   * `Discard` / `Exhaust` — `push` (back). Note: `Exhaust` here is
///     a *bookkeeping* move — it does NOT fire `after_card_exhausted`.
///     Use [`exhaust_card_uid`] for the listener-firing path.
///
/// Bead `sts2-rs-6ynt.2`.
pub fn move_card_to_pile(
    state: &mut CombatState,
    uid: CardUid,
    dest: PileKind,
    random_position: bool,
) -> Option<PileKind> {
    let src = remove_card_from_any_pile(state, uid)?;
    match dest {
        PileKind::Hand => {
            if state.player.hand.len() < crate::cards::HAND_LIMIT {
                state.player.hand.push(uid);
            } else {
                state.player.discard.push(uid);
            }
        }
        PileKind::Draw => {
            if random_position && !state.player.draw.is_empty() {
                // Bead `sts2-rs-09x5n` (F2 `HiddenInfoMode::BeliefDraw`):
                // under a deferred-draw-order pile a raw physical-index
                // insert would break the pins + sorted-unknown-middle layout
                // `cards::pop_next_card`'s index arithmetic assumes — fold
                // the card into the canonical (uid-sorted) unknown segment
                // instead, consuming no RNG (a random position inside an
                // order-unobserved region is unobservable). Note this branch
                // reads `combat_card_selection` (a Python-pin shape), NOT
                // the gh#181 `Shuffle`-stream family — the non-deferred
                // behavior below stays byte-identical.
                if state.player.deferred_draw_order {
                    crate::cards::insert_into_unknown_sorted(&mut state.player, uid);
                } else {
                    let len = state.player.draw.len();
                    // Inclusive [0, len] — len+1 possible slots (true append).
                    let idx = state.run_rng.combat_card_selection.next_int(0, len as i32) as usize;
                    state.player.draw.insert(idx, uid);
                }
            } else {
                state.player.draw.insert(0, uid);
            }
        }
        PileKind::Discard => state.player.discard.push(uid),
        PileKind::Exhaust => state.player.exhaust.push(uid),
    }
    Some(src)
}

/// Surgical exhaust of a specific card. Removes `uid` from whichever
/// pile currently holds it (Hand / Draw / Discard — Exhaust is a no-op
/// since the card is already there), pushes onto the exhaust pile, and
/// dual-fires `after_card_exhausted` on both the hooks bus (powers) and
/// the relic-effects bus (relics) in canonical order.
///
/// Returns `true` iff the card was found and exhausted. A card already
/// in the exhaust pile is treated as a no-op (returns `false`) — the
/// listeners have already fired for the original exhaust.
///
/// Mirrors Python `exhaust_card` at
/// `sts2-rl-agent@01c094d:sts2_env/core/combat.py:1717-1726`. Bead
/// `sts2-rs-6ynt.2`. Required by Stoke (Ironclad rare — exhausts every
/// hand card) and any future event/relic that needs to surgically
/// exhaust a non-top-of-draw card.
pub fn exhaust_card_uid(state: &mut CombatState, uid: CardUid) -> bool {
    // `CardCmd.Exhaust` is refused once the combat is ending (`CardCmd.cs:239`).
    if !crate::combat_loop::exhaust_may_begin(state) {
        return false;
    }
    let Some(src) = find_card_pile(state, uid) else {
        return false;
    };
    if matches!(src, PileKind::Exhaust) {
        return false;
    }
    if !remove_uid_from_pile(state, src, uid) {
        return false;
    }
    state.player.exhaust.push(uid);

    // Resolve typed dispatch contexts from the arena row.
    let idx = uid.0.get() as usize - 1;
    let card = &state.cards[idx];
    let card_id = card.card_id;
    let card_tags = card.tags;
    let card_type = card.card_type;

    // Hooks bus (powers) BEFORE relic-effects bus (relics) — locks the
    // ordering established in `CombatState::exhaust_from_draw_pile` and
    // `combat_loop`'s play-time exhaust path.
    let hook_ctx = crate::hooks::CardCtx {
        card_id: Some(card_id),
        card_uid: Some(uid),
        card_tags,
        card_type: Some(card_type),
    };
    // Surgical card-effect-driven exhaust (Stoke, etc.) — not the
    // end-of-turn ethereal sweep, so `caused_by_ethereal = false`. Bead
    // `sts2-rs-umouk`.
    let draw_was_suspended = state.pending_draw_resume.is_some();
    crate::hooks::dispatch_after_card_exhausted(state, &hook_ctx, false);

    let relic_ctx = crate::relic_effects::CardEventCtx {
        card_id,
        card_uid: Some(uid),
        target: None,
        card_tags,
        energy_spent: 0,
    };
    state.relic_after_card_exhausted_behind_powers(draw_was_suspended, &relic_ctx);
    true
}

/// Replace the arena entry referenced by `old_uid` with the supplied
/// `new_card` POD, reusing the *pile slot* (`old_uid`'s position in
/// whichever pile contains it). The new card is allocated in
/// `state.cards`, its `uid` field is overwritten with the assigned
/// 1-based slot, and the source pile entry at the original index is
/// rewritten to that new uid. Returns the new `CardUid`, or `None` if
/// `old_uid` is not found in any pile (e.g. mid-resolve).
///
/// Mirrors Python `transform_card` at
/// `sts2-rl-agent@01c094d:sts2_env/core/combat.py:1801-1820` — Python
/// rewrites the list slot in place; the Rust port keeps the pile slot
/// and allocates a fresh arena row so the original `Card` POD remains
/// reachable for any in-flight listener that captured the old uid.
/// Bead `sts2-rs-6ynt.2`. Required by PrimalForce (transform every
/// hand Attack into GiantRock), CollisionCourse / Apparition variants,
/// and event-card transformations.
pub fn transform_card_in_place(
    state: &mut CombatState,
    old_uid: CardUid,
    mut new_card: crate::cards::Card,
) -> Option<CardUid> {
    // C# `CardCmd.Transform` returns nothing as its first statement once
    // `CombatManager.IsEnding` (v0.107.1 CardCmd.cs:371-374, v0.111.0
    // :373-376), before anything moves or rolls.
    if crate::combat_loop::combat_is_ending(state) {
        return None;
    }
    let pile = find_card_pile(state, old_uid)?;
    // Allocate a fresh arena row.
    let raw = u32::try_from(state.cards.len() + 1).expect("cards arena fits in u32");
    let new_uid = CardUid::new(raw).expect("non-zero uid");
    new_card.uid = new_uid;
    // A mid-combat transform's product is a fresh combat-scoped CardModel
    // in C# (the run deck is untouched) — `DeckVersion == null` marker.
    stamp_generated_in_combat(&mut new_card);
    // v0.111.0 values for a card minted on a v0.107.1-only path (GH #793).
    new_card.normalize_for_version(state.game_version);
    state.cards.push(new_card);
    // C# `CardCmd.Transform` records a `CardGeneratedEntry` for every
    // non-Deck destination pile (`CardCmd.cs:444`,
    // `History.CardGenerated(combatState, replacement, replacement.Owner)`),
    // creator = the player. This drives Supermassive's damage
    // (`Supermassive.cs:22-27` counts player-owned `CardGeneratedEntry`).
    // Unconditional (outside the `!Exhaust` listener gate below) — the
    // record fires for Exhaust/Play too; only `PileType.Deck` is excluded,
    // and this function never transforms into the run deck.
    state.player.generated_cards_this_combat =
        state.player.generated_cards_this_combat.saturating_add(1);
    // Rewrite the pile slot at the original position so iteration
    // order is preserved (Python `target_pile[target_index] = new_card`
    // semantics). The per-pile match is necessary because each
    // `state.player.<pile>` is a `SmallVec` with a different inline
    // capacity, so `&mut`s to different piles have different types.
    match pile {
        PileKind::Hand => {
            let pos = state
                .player
                .hand
                .iter()
                .position(|&u| u == old_uid)
                .expect("uid present per find_card_pile above");
            state.player.hand[pos] = new_uid;
        }
        PileKind::Draw => {
            let pos = state
                .player
                .draw
                .iter()
                .position(|&u| u == old_uid)
                .expect("uid present per find_card_pile above");
            state.player.draw[pos] = new_uid;
        }
        PileKind::Discard => {
            let pos = state
                .player
                .discard
                .iter()
                .position(|&u| u == old_uid)
                .expect("uid present per find_card_pile above");
            state.player.discard[pos] = new_uid;
        }
        PileKind::Exhaust => {
            let pos = state
                .player
                .exhaust
                .iter()
                .position(|&u| u == old_uid)
                .expect("uid present per find_card_pile above");
            state.player.exhaust[pos] = new_uid;
        }
    }
    // Bead `sts2-rs-xk9yo`: a transform CREATES a new card that ENTERS
    // COMBAT (C# `CardFactory.CreateRandomCardForTransform` → the new card is
    // added to combat), so it must fire the `after_card_entered_combat` bus —
    // Regalite (+2 Block per owner card entering combat) and HungerPower were
    // silently skipped for transformed cards (e.g. Entropy-transformed cards
    // under Regalite). Mirrors `spawn_card_into_*`. Exhaust-pile transforms
    // are excluded inside `fire_after_card_entered_combat` (not a card-entered
    // event), matching the spawn gate.
    fire_after_card_entered_combat(state, new_uid, pile, true);
    // C# `CardCmd.TransformTo` fires `AfterCardGeneratedForCombat(addedByPlayer:
    // true)` UNCONDITIONALLY for the transformed-in card when it lands in a
    // combat pile (CardCmd.cs:395-401), so a transform (Charge → MinionDiveBomb,
    // …) raises the whole listener bundle just like a fresh generation —
    // including a TURN-START transform (Entropy's `AfterPlayerTurnStart`
    // TransformToRandom) where the Play pile is empty (Regent parity fuzz seed
    // 19: a turn-start Entropy transform granted no Arsenal Strength). Pass
    // `added_by_player = true` to match C#, not the Play-pile heuristic. Regent
    // parity fuzz seeds 427/438. Until GH #639 this site too called only the
    // Arsenal / PillarOfCreation pair.
    fire_card_generated_for_combat_powers(state, new_uid, true);
    Some(new_uid)
}

/// Idempotent upgrade of the card at `uid` — calls
/// [`crate::cards::Card::upgrade`] on the arena row. Returns `true`
/// iff the upgrade flag flipped (i.e. the card was previously
/// un-upgraded). Returns `false` for unknown uids.
///
/// This is the surface-level "flag flip" upgrade — full spec-rewrite
/// upgrades (cost / base_damage / base_block / tags from the upgraded
/// `CardSpec`) require an `sts2-content` round-trip and should pair
/// this with [`transform_card_in_place`]. See the `Card::upgrade`
/// doc-comment for the rationale (port-pin
/// `sts2-rl-agent@01c094d:sts2_env/run/run_state.py:259-261`).
///
/// Bead `sts2-rs-6ynt.2`.
///
/// Like C# `CardCmd.Upgrade` (CardCmd.cs:265-270, both builds), it does
/// nothing once the combat is ending:
///
/// [decompiled C# omitted — see the cited path]
///
/// So a card that kills the last enemy is not upgraded by Razor Tooth, and an
/// upgraded Armaments whose Juggernaut hit ends the fight upgrades no hand
/// card. Every in-combat `CardCmd.Upgrade` port goes through here.
pub fn upgrade_card_uid(state: &mut CombatState, uid: CardUid) -> bool {
    if crate::combat_loop::combat_is_ending(state) {
        return false;
    }
    let version = state.game_version;
    crate::card_by_uid_mut(&mut state.cards, uid).is_some_and(|c| c.upgrade_for(version))
}

/// `CardCmd.Downgrade` on an in-combat card: [`Card::downgrade_for`] plus the
/// keyword storage that lives off the card. C# `CardModel.DowngradeInternal`
/// resets `_keywords` to the canonical Local set (CardModel.cs:2157 v0.111.0,
/// :2144 v0.107.1), so a Sly that Master Planner applied with
/// `CardCmd.ApplyKeyword` (MasterPlannerPower.cs:26, kept here in
/// `player.sly_this_combat`) is gone; the single-turn Sly / Retain flags are
/// separate fields and survive (random-loadout Silent 5772 v0.111.0). The
/// on-card keywords (`keyword_override`'s Retain, `EtherealThisCombat`) are
/// reset by `Card::downgrade_for` itself. Returns whether it downgraded.
///
/// [`Card::downgrade_for`]: crate::cards::Card::downgrade_for
pub fn downgrade_card_in_combat(state: &mut CombatState, uid: CardUid) -> bool {
    let version = state.game_version;
    let Some(card) = crate::card_by_uid_mut(&mut state.cards, uid) else {
        return false;
    };
    if !card.downgrade_for(version) {
        return false;
    }
    state.player.sly_this_combat.retain(|u| *u != uid);
    true
}

/// Clear the `upgraded` flag on the card at `uid`. Returns `true` iff
/// the flag was previously set. The Python pin has no first-class
/// `downgrade_card` — Phase-7 events that "downgrade an upgraded card
/// in your deck" replicate this by re-fetching the un-upgraded spec
/// and calling `transform_card`; this surface lets future Rust ports
/// pick whichever path matches the consumer's parity shape.
///
/// Bead `sts2-rs-6ynt.2`.
pub fn downgrade_card_uid(state: &mut CombatState, uid: CardUid) -> bool {
    let idx = uid.0.get() as usize - 1;
    let Some(card) = state.cards.get_mut(idx) else {
        return false;
    };
    if !card.upgraded {
        return false;
    }
    card.upgraded = false;
    true
}

// ---------------------------------------------------------------------------
// PlayCard dispatch registry (bead sts2-rs-ov4 / P6-1)
// ---------------------------------------------------------------------------
//
// Replaces the per-CardId `match` arm in `combat_loop::play_card` with a
// function-pointer table. Registry storage lives in this crate because the
// engine (`combat_loop`) needs to look up entries; concrete handlers are
// installed by `sts2-content::cards::dispatch::install_default_registry`
// at engine-boot. Tests that exercise `play_card` must call
// `sts2_content::cards::dispatch::ensure_installed()` before dispatching
// (or use the auto-install path in `Action::PlayCard`).
//
// Design notes:
// - `OnceLock<RwLock<BTreeMap>>` lets downstream beads extend the table
//   after first init without us pre-allocating ~600 slots. Downstream
//   beads call `register_many` once per character pool.
// - Lookup is O(1) on the lock-free read path: we acquire a read lock,
//   clone the function pointer, and release. Function-pointer copy is
//   one machine word.
// - `EffectResult::NotImplemented` is the fall-through for cards without
//   a registered handler — combat_loop treats it identically to `Resolved`
//   for state-machine purposes.

/// Function-pointer signature for a card on-play handler.
///
/// `target` is `Some(uid)` for `TargetType::Enemy` cards (the engine
/// resolves and alive-checks before dispatch); `None` for `Self_` / `None`.
pub type PlayCardFn = fn(&mut CombatState, target: Option<CreatureUid>) -> EffectResult;

static REGISTRY: OnceLock<RwLock<BTreeMap<CardId, PlayCardFn>>> = OnceLock::new();

/// Dense frozen copy of [`REGISTRY`], read by [`lookup`] once
/// [`freeze_card_tables`] has published it.
static PLAY_TABLE: FrozenCardTable<PlayCardFn> = FrozenCardTable::new();

fn registry() -> &'static RwLock<BTreeMap<CardId, PlayCardFn>> {
    REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register a single handler. Overwrites any previous entry for `card_id`.
pub fn register(card_id: CardId, handler: PlayCardFn) {
    let mut guard = registry().write().expect("dispatch registry poisoned");
    guard.insert(card_id, handler);
    PLAY_TABLE.invalidate();
}

/// Register many handlers in one go. Downstream beads use this per-pool.
pub fn register_many(entries: &[(CardId, PlayCardFn)]) {
    let mut guard = registry().write().expect("dispatch registry poisoned");
    for &(id, f) in entries {
        guard.insert(id, f);
    }
    PLAY_TABLE.invalidate();
}

/// Look up the registered handler for `card_id`, or `None` if absent.
///
/// Runs on every card play. After [`freeze_card_tables`] it is one indexed
/// load from the dense frozen table; before the freeze, or after a
/// registration the freeze has not re-checked, it reads the locked registry.
#[must_use]
pub fn lookup(card_id: CardId) -> Option<PlayCardFn> {
    if let Some(table) = PLAY_TABLE.get() {
        return FrozenCardTable::entry(table, card_id);
    }
    registry()
        .read()
        .expect("dispatch registry poisoned")
        .get(&card_id)
        .copied()
}

/// Dispatch the on-play effect for `card_id`. Returns
/// `EffectResult::NotImplemented` when no handler is registered.
pub fn dispatch(
    state: &mut CombatState,
    card_id: CardId,
    target: Option<CreatureUid>,
) -> EffectResult {
    match lookup(card_id) {
        Some(f) => f(state, target),
        None => EffectResult::NotImplemented,
    }
}

/// [`dispatch`], with the resolving card's identity STATED rather than left
/// to be inferred from the top of the Play pile.
///
/// GH #422. C# `CardModel.OnPlayWrapper` runs every play iteration on `this`
/// and reads `this.Damage` / `this.Block` / `this.CurrentUpgradeLevel`
/// directly; the Rust port infers the same card from `play.last()`, which is
/// only equivalent while the card is still ON the Play pile. It need not be:
/// C# guards its own result-pile disposal on `Pile?.Type == PileType.Play`
/// (`CardModel.cs:1975`) exactly because an effect can move the card out
/// mid-resolve. Stamping [`CombatState::resolving_card`] here — and restoring
/// the previous value on the way out, so a nested auto-play inside this
/// handler restores US when it returns — makes every
/// [`resolving_card_uid`] read exact.
///
/// Every engine play path (hand play + its OneTwoPunch / EchoForm replays,
/// the auto-play resolver + its replay loop, the HistoryCourse dupe) goes
/// through this; the bare [`dispatch`] stays for tests and for callers that
/// have no card instance to name.
pub fn dispatch_for_card(
    state: &mut CombatState,
    card_uid: CardUid,
    card_id: CardId,
    target: Option<CreatureUid>,
) -> EffectResult {
    let prev = state.resolving_card.replace(card_uid);
    let out = dispatch(state, card_id, target);
    state.resolving_card = prev;
    out
}

/// [`dispatch_for_card`] for an AUTO-PLAY (C# `CardCmd.AutoPlay`): raises
/// [`crate::TurnEvents::resolving_auto_play`] for the duration of the
/// dispatch and restores the previous value, so a manual play's nested
/// auto-plays (Catastrophe, Havoc) and an auto-play's nested manual-free
/// replays stack correctly.
pub fn dispatch_for_auto_played_card(
    state: &mut CombatState,
    card_uid: CardUid,
    card_id: CardId,
    target: Option<CreatureUid>,
) -> EffectResult {
    let prev = std::mem::replace(&mut state.combat_log.resolving_auto_play, true);
    let out = dispatch_for_card(state, card_uid, card_id, target);
    state.combat_log.resolving_auto_play = prev;
    out
}

/// The [`CardUid`] whose `OnPlay` body is currently executing.
///
/// Prefers the identity the dispatch site stated
/// ([`CombatState::resolving_card`]) and falls back to the historical
/// `play.last()` inference, which still answers correctly for the listener
/// surfaces that run OUTSIDE a handler (`after_card_played`, monster moves,
/// turn-end ticks) and for the tests that call [`dispatch`] directly. The two
/// agree whenever the card is still on the Play pile — i.e. always, except in
/// the suspended-auto-play interleaving of GH #422 — so routing a read
/// through this is byte-neutral in normal play.
///
/// `None` on the non-card surfaces C# passes a `null` card for: potions,
/// power ticks, relic effects, enemy moves.
#[must_use]
pub fn resolving_card_uid(state: &CombatState) -> Option<CardUid> {
    state
        .resolving_card
        .or_else(|| state.player.play.last().copied())
}

/// The [`crate::cards::Card`] instance currently being played — the arena
/// entry [`resolving_card_uid`] names. `None` when no card is resolving (or
/// when the uid is not in the arena, which cannot happen in a well-formed
/// state).
#[must_use]
pub fn resolving_card_inst(state: &CombatState) -> Option<&crate::cards::Card> {
    let uid = resolving_card_uid(state)?;
    state.cards.get(uid.0.get() as usize - 1)
}

/// The X value of the X-cost card currently being played: its captured
/// `energy_spent` (C# `EnergyCost.CapturedXValue`) folded through the relic
/// `modify_x_value` chain (ChemicalX +2) — C# `CardModel.ResolveEnergyXValue`,
/// which every v0.107.1 X-cost card reads. 0 when nothing is resolving.
#[must_use]
pub fn resolving_card_x_value(state: &CombatState) -> i32 {
    let captured = resolving_card_inst(state).map_or(0, |c| c.energy_spent);
    crate::relic_effects::dispatch_modify_x_value(state, captured)
}

/// `base_damage` of the card currently being played, or 0 when nothing is
/// resolving. C# `DamageVar.BaseValue` off the `CardPlay`'s own card.
#[must_use]
pub fn resolving_card_base_damage(state: &CombatState) -> i32 {
    resolving_card_inst(state).map_or(0, |c| c.base_damage)
}

/// `base_block` of the card currently being played, or 0 when nothing is
/// resolving. C# `BlockVar.BaseValue` off the `CardPlay`'s own card.
#[must_use]
pub fn resolving_card_base_block(state: &CombatState) -> i32 {
    resolving_card_inst(state).map_or(0, |c| c.base_block)
}

/// Whether the card currently being played is upgraded (`false` when nothing
/// is resolving). C# `CurrentUpgradeLevel > 0` off the `CardPlay`'s own card.
#[must_use]
pub fn resolving_card_upgraded(state: &CombatState) -> bool {
    resolving_card_inst(state).is_some_and(|c| c.upgraded)
}

/// Number of currently-registered handlers (mostly useful for tests).
#[must_use]
pub fn registered_count() -> usize {
    registry().read().expect("dispatch registry poisoned").len()
}

// ---------------------------------------------------------------------------
// Frozen dense card tables (M06)
// ---------------------------------------------------------------------------
//
// The `CardId`-keyed fn-pointer registries (on-play handlers above, late
// effects below) are filled at content install and then only read, once per
// card play. Reading them through the `RwLock<BTreeMap>` cost two atomic RMWs
// plus a B-tree walk per lookup, and the late-effect sweep walked every pile
// on every play even in a deck that holds no late-effect card. The content
// installer therefore publishes a dense `CardId`-indexed copy of each map as
// its last step ([`freeze_card_tables`]), the same freeze-after-install
// ordering the relic channels use (M02,
// `docs/perf/2026-10-13-engine-speed-exact-design.md`).
//
// Any registration clears the copy's `valid` flag, so readers fall back to the
// locked map until the next freeze re-checks it. A re-install that registers
// the same handlers again (production installs twice) is re-validated by the
// installer's own closing freeze; a different handler keeps the copy invalid
// for as long as the map differs from it. The copy can therefore never answer
// differently from the map.

/// A `CardId`-indexed copy of one fn-pointer registry. See the section note.
struct FrozenCardTable<F: Copy + 'static> {
    table: OnceLock<Box<[Option<F>]>>,
    /// `true` iff `table` matches the registry. Cleared by every
    /// registration; set by [`Self::freeze`] after it compares.
    valid: std::sync::atomic::AtomicBool,
}

impl<F: Copy + 'static> FrozenCardTable<F> {
    const fn new() -> Self {
        Self {
            table: OnceLock::new(),
            valid: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The frozen table while it is valid; `None` sends the caller to the
    /// locked registry.
    #[inline]
    fn get(&self) -> Option<&[Option<F>]> {
        if self.valid.load(std::sync::atomic::Ordering::Acquire) {
            self.table.get().map(|t| &**t)
        } else {
            None
        }
    }

    /// `card_id`'s entry in a table returned by [`Self::get`]. Ids past the
    /// end have no registered handler (the table spans the largest key).
    #[inline]
    fn entry(table: &[Option<F>], card_id: CardId) -> Option<F> {
        table.get(card_id as usize).copied().flatten()
    }

    /// Called by every registration, under the registry's write lock.
    fn invalidate(&self) {
        self.valid
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Publish `map` as the frozen table, or, when a table is already
    /// published, mark it valid only if `map` still matches it entry for
    /// entry (`addr` gives a handler's address). Call under the registry's
    /// read lock so no registration interleaves.
    fn freeze(&self, map: &BTreeMap<CardId, F>, addr: fn(F) -> usize) {
        let len = map.keys().next_back().map_or(0, |&id| id as usize + 1);
        let mut dense = vec![None; len].into_boxed_slice();
        for (&id, &f) in map {
            dense[id as usize] = Some(f);
        }
        let matches = match self.table.set(dense) {
            Ok(()) => true,
            Err(dense) => self.table.get().is_some_and(|frozen| {
                frozen.len() == dense.len()
                    && frozen.iter().zip(dense.iter()).all(|(a, b)| match (a, b) {
                        (None, None) => true,
                        (Some(a), Some(b)) => addr(*a) == addr(*b),
                        _ => false,
                    })
            }),
        };
        self.valid
            .store(matches, std::sync::atomic::Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Card-keyword metadata registry — bead `sts2-rs-6ynt.3.4`
// ---------------------------------------------------------------------------
//
// Mirrors the `EXHAUST` bit of `sts2-content::cards::types::CardKeywords` for
// engine-side lookups inside `combat_loop::play_card`. The handler dispatch
// registry lives one section above — this is a sibling boolean-set so the
// engine can decide whether to route a played card to Discard or Exhaust
// without depending on `sts2-content` (which would create a workspace cycle).
//
// Population: `sts2-content::cards::dispatch::install_default_registry`
// iterates every per-character spec slice (`spec_index::*_SPECS`) and calls
// `register_card_exhausts(id)` for any spec whose `keywords` contains
// `EXHAUST`. Cards never marked are left out of the set; lookups default to
// `false`.
//
// Source-of-truth for the EXHAUST flag: per-card `CardSpec.keywords` in
// `crates/sts2-content/src/cards/*_specs_generated.rs` (scraped from
// `sts2-rl-agent@01c094d:sts2_env/cards/*.py`).

/// Lock-free, install-once membership set keyed by [`CardId`]'s
/// discriminant.
///
/// The per-keyword card registries (EXHAUST / ETHEREAL / UNPLAYABLE /
/// INNATE / RETAIN / SLY) are populated once at content-install and then
/// queried per-card on hot end-of-turn / draw / playability loops. The
/// previous `OnceLock<RwLock<BTreeSet<CardId>>>` paid a global `RwLock`
/// read **plus** a B-tree walk on every query — together the largest
/// remaining dispatch-overhead bucket after the relic-channel and
/// late-effect locks went lock-free (2026-06-13 profile). This replaces
/// both with a single relaxed atomic load + bit test. Bits are only ever
/// set (the sets are insert-only and never cleared), so `Relaxed` is
/// sound: a query never needs to observe ordering relative to other
/// state, only the eventually-published install writes (which complete
/// before any simulation reads).
struct CardIdSet {
    words: [std::sync::atomic::AtomicU64; CARD_ID_SET_WORDS],
}

/// 1024 bits — comfortable headroom over the 579 `CardId` variants at this
/// pin. [`CardIdSet::insert`] debug-asserts the discriminant fits.
const CARD_ID_SET_WORDS: usize = 16;

impl CardIdSet {
    const fn new() -> Self {
        // The `const` operand is the idiomatic way to build a `static`
        // array of atomics; the interior-mutability lint is a false
        // positive here (each array slot is a distinct, independent cell).
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            words: [ZERO; CARD_ID_SET_WORDS],
        }
    }

    #[inline]
    fn insert(&self, card_id: CardId) {
        let i = card_id as usize;
        debug_assert!(
            i < CARD_ID_SET_WORDS * 64,
            "CardId discriminant {i} exceeds CardIdSet capacity; bump CARD_ID_SET_WORDS"
        );
        if let Some(word) = self.words.get(i / 64) {
            word.fetch_or(1u64 << (i % 64), std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[inline]
    fn contains(&self, card_id: CardId) -> bool {
        let i = card_id as usize;
        match self.words.get(i / 64) {
            Some(word) => (word.load(std::sync::atomic::Ordering::Relaxed) >> (i % 64)) & 1 != 0,
            None => false,
        }
    }
}

static EXHAUST_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `EXHAUST` keyword. Idempotent.
pub fn register_card_exhausts(card_id: CardId) {
    EXHAUST_SET.insert(card_id);
}

/// Mark every entry in `ids` as having the `EXHAUST` keyword.
pub fn register_many_exhausts(ids: &[CardId]) {
    for &id in ids {
        EXHAUST_SET.insert(id);
    }
}

/// `true` iff `card_id` was registered as EXHAUST. `false` for unregistered
/// cards (the default — most cards go to Discard on play).
#[must_use]
pub fn card_exhausts(card_id: CardId) -> bool {
    EXHAUST_SET.contains(card_id)
}

// ---------------------------------------------------------------------------
// ETHEREAL keyword registry — bead `sts2-rs-6ynt.3.3`
// ---------------------------------------------------------------------------
//
// Sibling to [`EXHAUST_REGISTRY`] above. Mirrors the `ETHEREAL` bit of
// `sts2-content::cards::types::CardKeywords` for engine-side lookups inside
// `combat_loop::end_player_turn_phase_two` (the ETHEREAL expel step). At
// `sts2-rl-agent@01c094d` the canonical ETHEREAL cards are Dazed,
// AscendersBane, Clumsy, Void (plus a handful of monster status cards
// like SweepingGaze).
//
// Population: `sts2-content::cards::dispatch::install_default_registry`
// iterates every per-character spec slice (`spec_index::*_SPECS`) and calls
// `register_card_ethereal(id)` for any spec whose `keywords` contains
// `ETHEREAL`. Cards never marked are left out of the set; lookups default
// to `false`.
//
// Source-of-truth for the ETHEREAL flag: per-card `CardSpec.keywords` in
// `crates/sts2-content/src/cards/*_specs_generated.rs` (scraped from
// `sts2-rl-agent@01c094d:sts2_env/cards/*.py`).

static ETHEREAL_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `ETHEREAL` keyword. Idempotent.
pub fn register_card_ethereal(card_id: CardId) {
    ETHEREAL_SET.insert(card_id);
}

/// Mark every entry in `ids` as having the `ETHEREAL` keyword.
pub fn register_many_ethereal(ids: &[CardId]) {
    for &id in ids {
        ETHEREAL_SET.insert(id);
    }
}

/// `true` iff `card_id` was registered as ETHEREAL. `false` for unregistered
/// cards (the default — most cards stay in hand or flush to discard at
/// end-of-turn).
#[must_use]
pub fn card_ethereal(card_id: CardId) -> bool {
    ETHEREAL_SET.contains(card_id)
}

// ---------------------------------------------------------------------------
// UNPLAYABLE keyword registry — bead `sts2-rs-6ynt.8-unplayable`
// ---------------------------------------------------------------------------
//
// Sibling to [`EXHAUST_REGISTRY`] / [`ETHEREAL_REGISTRY`]. Mirrors the
// `UNPLAYABLE` bit of `sts2-content::cards::types::CardKeywords`
// (`CardSpec::is_unplayable`) for engine-side lookups when card bodies
// need to filter pile contents by playability (e.g. Catastrophe's
// `is_unplayable` filter at `sts2_env/cards/colorless.py:73-81`).
// Population follows the same pattern: walk every per-character spec
// slice and `register_card_unplayable(id)` for cards whose keywords
// contain `UNPLAYABLE`.

static UNPLAYABLE_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `UNPLAYABLE` keyword. Idempotent.
pub fn register_card_unplayable(card_id: CardId) {
    UNPLAYABLE_SET.insert(card_id);
}

/// `true` iff `card_id` was registered as UNPLAYABLE (Wound, Dazed,
/// Burn, Void, AscendersBane, every curse, etc.). `false` for
/// unregistered cards.
#[must_use]
pub fn card_unplayable(card_id: CardId) -> bool {
    UNPLAYABLE_SET.contains(card_id)
}

// INNATE registry — sibling to UNPLAYABLE/EXHAUST/ETHEREAL. Mirrors the
// `INNATE` bit of `sts2-content::cards::types::CardKeywords`. Consumed by
// the round-1 opening-draw sort in `combat_loop::start_player_turn`
// (C# `CombatManager` moves Innate draw-pile cards to the top before the
// first draw). Population walks every per-character spec slice via
// `install_innate_keywords` and `register_card_innate(id)`. Bead
// `sts2-rs-2un`.

static INNATE_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `INNATE` keyword. Idempotent.
pub fn register_card_innate(card_id: CardId) {
    INNATE_SET.insert(card_id);
}

/// `true` iff `card_id` was registered as INNATE (MindBlast,
/// DramaticEntrance, Brutality, …). `false` for unregistered cards.
#[must_use]
pub fn card_innate(card_id: CardId) -> bool {
    INNATE_SET.contains(card_id)
}

// RETAIN registry — sibling to UNPLAYABLE/EXHAUST/ETHEREAL/INNATE. Mirrors
// the `RETAIN` bit of `sts2-content::cards::types::CardKeywords`. Consumed
// by the end-of-turn flush in `combat_loop::end_player_turn_phase_two`:
// C# `CombatManager.EndPlayerTurnPhaseTwoInternal` partitions the hand by
// `CardModel.ShouldRetainThisTurn` (true whenever `Keywords.Contains(Retain)`)
// and keeps the retained cards IN hand while the rest flush to discard, so
// the flat turn-start draw lands on top of them. Population walks every
// per-character spec slice via `install_retain_keywords` and
// `register_card_retain(id)`. Bead `sts2-rs-o8ixl` (q6bix RESTLESSNESS).

static RETAIN_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `RETAIN` keyword. Idempotent.
pub fn register_card_retain(card_id: CardId) {
    RETAIN_SET.insert(card_id);
}

/// `true` iff `card_id` was registered as RETAIN (Restlessness, and any
/// card whose base keyword set keeps it in hand at end of turn). `false`
/// for unregistered cards (the default — most cards flush to discard).
#[must_use]
pub fn card_retain(card_id: CardId) -> bool {
    RETAIN_SET.contains(card_id)
}

// SLY keyword registry (C# `CardKeyword.Sly`). A Sly card auto-plays when it
// is discarded by `CardCmd.DiscardAndDraw` (CalculatedGamble) or
// `CardSelectCmd` (manual discard) — `IsSlyThisTurn` gate. Populated from the
// per-character spec slices by `install_sly_keywords` (mirrors RETAIN). All
// Sly cards are Silent (Abrasive, Sneaky, Reflex, Ricochet, Untouchable,
// Tactician, FlickFlack, Haze), so the registry is inert for other characters.
static SLY_SET: CardIdSet = CardIdSet::new();

/// Mark `card_id` as having the `SLY` keyword. Idempotent.
pub fn register_card_sly(card_id: CardId) {
    SLY_SET.insert(card_id);
}

/// `true` iff `card_id` was registered as SLY. `false` for unregistered cards
/// (the default).
#[must_use]
pub fn card_sly(card_id: CardId) -> bool {
    SLY_SET.contains(card_id)
}

/// [`card_sly`] in game build `version` (GH #793): v0.111.0 Haze lost its
/// Sly keyword (decomp `Models.Cards/Haze.cs:19` — no `CanonicalKeywords`).
#[must_use]
pub fn card_sly_in(card_id: CardId, version: crate::GameVersion) -> bool {
    card_sly(card_id) && !(version >= crate::GameVersion::V0_111_0 && card_id == CardId::Haze)
}

/// `true` iff the card instance `uid` is `IsSlyThisTurn` (C#
/// `CardModel.IsSlyThisTurn` / `Keywords.Contains(Sly)`): the permanent `Sly`
/// keyword by [`card_sly`], OR a single-turn Sly (`player.sly_this_turn`,
/// HandTrick), OR a combat-long instance Sly (`player.sly_this_combat`,
/// MasterPlanner). The single Sly-collection predicate every active-discard
/// surface shares so the three Sly sources are honored identically.
#[must_use]
pub fn card_is_sly(state: &CombatState, uid: CardUid) -> bool {
    state
        .cards
        .get(uid.0.get() as usize - 1)
        .is_some_and(|c| card_sly_in(c.card_id, state.game_version))
        || state.player.sly_this_turn.contains(&uid)
        || state.player.sly_this_combat.contains(&uid)
}

// ---------------------------------------------------------------------------
// Instance-aware keyword resolution — bead `sts2-rs-59atd`
// ---------------------------------------------------------------------------
//
// The `card_{exhausts,ethereal,unplayable}` registries above are keyed by
// `CardId`, which is *shared* between a card's upgraded and un-upgraded
// forms. That makes them un-upgrade-aware: a card whose keyword set changes
// on upgrade (BurningPact drops EXHAUST, Aggression/Afterimage/Entropy gain
// INNATE, some cards gain RETAIN — see `sts2-content`'s
// `card_upgrades_generated::upgrade_delta`) cannot be distinguished from its
// base form by `CardId` alone.
//
// `Card::keyword_override` (set at upgrade time from `CardUpgradeDelta.
// keywords`) carries the *absolute* upgraded keyword set for exactly those
// cards. The `*_inst` helpers below consult that per-instance override when
// present and fall back to the `CardId` registry otherwise — so every
// non-overridden card (the overwhelming majority) is byte-for-byte
// unchanged, and only the upgraded keyword-delta cards diverge.
//
// Behaviour-driving call sites (the ETHEREAL expel in
// `combat_loop::end_player_turn_phase_two`, the EXHAUST-on-play branch in
// `combat_loop::resolve_post_play`, the UNPLAYABLE skip in the playable-card
// enumerator in `lib.rs`) consult the `*_inst` variants. Code paths that
// only have a bare `CardId` (e.g. spec-pool scans) keep using the registry
// functions directly.

/// `true` iff `card` exhausts on play, honoring an upgrade-aware
/// [`Card::keyword_override`] when set and otherwise deferring to the
/// `CardId`-keyed EXHAUST registry. Bead `sts2-rs-59atd`.
///
/// C# `SoulsPower.OnEnchant` removes the card's Exhaust keyword (it can only
/// enchant a card that has one, `SoulsPower.CanEnchant`), so a Soul's Power
/// card never exhausts on play. `Goopy.OnEnchant` adds Exhaust (GH #1021).
#[must_use]
pub fn card_exhausts_inst(card: &Card) -> bool {
    if card.has_enchantment(crate::cards::EnchantmentId::SoulsPower) {
        return false;
    }
    if card.has_enchantment(crate::cards::EnchantmentId::Goopy) {
        return true;
    }
    match card.keyword_override {
        Some(k) => k.contains(KeywordOverride::EXHAUST),
        None => card_exhausts(card.card_id),
    }
}

/// `true` iff `card` is ETHEREAL, honoring an upgrade-aware
/// [`Card::keyword_override`] when set and otherwise deferring to the
/// `CardId`-keyed ETHEREAL registry. Bead `sts2-rs-59atd`.
#[must_use]
pub fn card_ethereal_inst(card: &Card) -> bool {
    match card.keyword_override {
        Some(k) => k.contains(KeywordOverride::ETHEREAL),
        None => card_ethereal(card.card_id),
    }
}

/// `true` iff `card` is UNPLAYABLE, honoring an upgrade-aware
/// [`Card::keyword_override`] when set and otherwise deferring to the
/// `CardId`-keyed UNPLAYABLE registry. Bead `sts2-rs-59atd`.
#[must_use]
pub fn card_unplayable_inst(card: &Card) -> bool {
    match card.keyword_override {
        Some(k) => k.contains(KeywordOverride::UNPLAYABLE),
        None => card_unplayable(card.card_id),
    }
}

/// `true` iff `card` is INNATE, honoring an upgrade-aware
/// [`Card::keyword_override`] when set (Aggression/Afterimage/Entropy gain
/// INNATE on upgrade) and otherwise deferring to the `CardId`-keyed INNATE
/// registry. Bead `sts2-rs-2un`.
///
/// `RoyallyApproved.OnEnchant` adds Innate (GH #1021).
#[must_use]
pub fn card_innate_inst(card: &Card) -> bool {
    if card.has_enchantment(crate::cards::EnchantmentId::RoyallyApproved) {
        return true;
    }
    match card.keyword_override {
        Some(k) => k.contains(KeywordOverride::INNATE),
        None => card_innate(card.card_id),
    }
}

/// `true` iff `card` is RETAIN, honoring an upgrade-aware
/// [`Card::keyword_override`] when set (several cards gain RETAIN on
/// upgrade — see `card_upgrades_generated::upgrade_delta`) and otherwise
/// deferring to the `CardId`-keyed RETAIN registry. Drives the
/// keep-in-hand partition of the end-of-turn flush in
/// `combat_loop::end_player_turn_phase_two`. Bead `sts2-rs-o8ixl`.
///
/// `Steady.OnEnchant` and `RoyallyApproved.OnEnchant` add Retain (GH #1021).
/// The keyword lives on the card, so it is read off the enchantment here
/// rather than from the `apply_static_enchantment` keyword diff.
#[must_use]
pub fn card_retain_inst(card: &Card) -> bool {
    use crate::cards::EnchantmentId;
    if card.has_enchantment(EnchantmentId::Steady)
        || card.has_enchantment(EnchantmentId::RoyallyApproved)
    {
        return true;
    }
    match card.keyword_override {
        Some(k) => k.contains(KeywordOverride::RETAIN),
        None => card_retain(card.card_id),
    }
}

/// Build the engine-tracked [`KeywordOverride`] for `card_id` from the static
/// keyword registries (UNPLAYABLE / ETHEREAL / RETAIN / EXHAUST / INNATE). The
/// core-side counterpart of content's `keyword_override_from(spec.keywords)`,
/// keyed by `CardId` so core can compose a per-instance override without a
/// content dependency. ETERNAL is omitted: removability does not come through
/// this override at all, it comes through the registered
/// [`crate::CardRemovableFn`], which reads content's own effective-keyword
/// set (enchantments included) rather than a per-`CardId` registry.
#[must_use]
pub fn registry_keyword_override(card_id: CardId) -> KeywordOverride {
    let mut kw = KeywordOverride::NONE;
    if card_unplayable(card_id) {
        kw = kw.with(KeywordOverride::UNPLAYABLE);
    }
    if card_ethereal(card_id) {
        kw = kw.with(KeywordOverride::ETHEREAL);
    }
    if card_retain(card_id) {
        kw = kw.with(KeywordOverride::RETAIN);
    }
    if card_exhausts(card_id) {
        kw = kw.with(KeywordOverride::EXHAUST);
    }
    if card_innate(card_id) {
        kw = kw.with(KeywordOverride::INNATE);
    }
    kw
}

/// Add the ETHEREAL keyword to a freshly-minted card instance, preserving its
/// other engine-tracked keywords. Mirrors C# `CardCmd.ApplyKeyword(card,
/// CardKeyword.Ethereal)` (an additive keyword grant, used by
/// `CallOfTheVoidPower.BeforeHandDraw`). Seeds the per-instance override from
/// the card's existing `keyword_override` when set, else from the static
/// registry, then unions in ETHEREAL — so a generated Exhaust/Retain card keeps
/// those keywords. Idempotent.
pub fn apply_ethereal_keyword(card: &mut Card) {
    let base = card
        .keyword_override
        .unwrap_or_else(|| registry_keyword_override(card.card_id));
    card.keyword_override = Some(base.with(KeywordOverride::ETHEREAL));
}

/// Add the RETAIN keyword to a card instance, preserving its other
/// engine-tracked keywords. Mirrors C# `CardCmd.ApplyKeyword(card,
/// CardKeyword.Retain)` (an ADDITIVE grant — Snap.cs:42). Seeds the
/// per-instance override from the card's existing `keyword_override` when set,
/// else from the static registry, then unions in RETAIN — so a card that
/// already EXHAUSTs (e.g. base Prolong) keeps EXHAUST instead of having its
/// keyword set replaced by a bare RETAIN. The latter made `card_exhausts_inst`
/// (which reads the override first) return false, so a Snap-Retained Prolong
/// discarded instead of exhausting. Idempotent.
pub fn apply_retain_keyword(card: &mut Card) {
    let base = card
        .keyword_override
        .unwrap_or_else(|| registry_keyword_override(card.card_id));
    card.keyword_override = Some(base.with(KeywordOverride::RETAIN));
}

/// q6bix F-E: `true` for cards whose C# `TargetType` is `AnyAlly`.
///
/// C# `CardModel.CanPlay` (CardModel.cs:1342-1344) marks an `AnyAlly`
/// card unplayable when `PlayerCreatures.Count(c => c.IsAlive) <= 1`
/// (i.e. the player has no living ally to target). The Rust card specs
/// collapse `AnyAlly` → `Self_` (there is no `AnyAlly` `TargetType`
/// variant at this pin), so `legal_actions` re-derives the set by
/// `CardId` here and applies the same gate. List is every v0.103.2 card
/// with `TargetType.AnyAlly`: BelieveInYou, Coordinate, DemonicShield,
/// Ignition, Intercept, Largesse, Lift, Mimic.
#[must_use]
pub fn card_is_any_ally(id: crate::cards::CardId) -> bool {
    use crate::cards::CardId;
    matches!(
        id,
        CardId::BelieveInYou
            | CardId::CoordinateCard
            | CardId::DemonicShield
            | CardId::Ignition
            | CardId::InterceptCard
            | CardId::Largesse
            | CardId::Lift
            | CardId::Mimic
            // v0.111.0-only MP cards with `TargetType.AnyAlly` (GH #793):
            // Blaze, Concoct, Constellation, Fade, ImitationLearning,
            // Soulbound, Tutor. The ids exist in no v0.107.1 state.
            | CardId::Blaze
            | CardId::ConcoctCard
            | CardId::Constellation
            | CardId::FadeCard
            | CardId::ImitationLearningCard
            | CardId::SoulboundCard
            | CardId::Tutor
    )
}

// ---------------------------------------------------------------------------
// Late-effect bus (bead `sts2-rs-j9x8`)
// ---------------------------------------------------------------------------
//
// Mirrors Python's `register_late_effect(card_id)` registry +
// `_fire_after_card_played_late(played_card)` dispatch site at
// `sts2-rl-agent@01c094d:sts2_env/cards/registry.py:30-62` +
// `sts2_env/core/combat.py:2266-2275`.
//
// The bus fires AFTER the on-play effect dispatch and after both
// `after_card_played` buses (relic-effects + hooks). It walks every
// card in every pile (Hand / Draw / Discard / Exhaust / Play) and
// invokes the registered late-effect for that card's `CardId`. The
// signature receives the *watched* card uid (the card being asked
// "did you want to react to this play?") plus the *played* card's
// [`hooks::CardCtx`].
//
// **State-machine over closures rationale**: the engine state is
// serializable / hashable and used as MCTS-rollout primary state.
// Storing a closure per watched card would push non-`PartialEq` /
// non-`Hash` boxed `Fn` objects into `CombatState`. Instead, we key
// the late-effect by `CardId` (a static fact) and store any
// per-instance scratch on the watched `Card`'s existing
// `combat_vars` slot — same pattern as Rampage / ClawBuff.
//
// **Per-pile iteration**: matches Python's `for state in
// combat_player_states: for pile in state.all_piles`. The Rust port
// has a single player, so we iterate `state.player`'s five piles.
// Cards already in `Play` (the just-played card) ARE visited — this
// is intentional and matches Python (the `watched in combat.hand`
// guard in MakeItSo's late-effect gates the actual move).
//
// **Snapshot semantics**: late-effects can mutate piles (e.g.
// MakeItSo moves the watched card from Exhaust to Hand). We snapshot
// the per-pile uid lists into a single SmallVec before invoking any
// listener so a mid-iteration move does not double-fire or skip a
// neighbouring card. Per-instance dedupe (Python's `seen_ids`) is
// enforced by the snapshot — a single uid appears once.

/// Function-pointer signature for a card's late-effect handler.
///
/// * `state` — mutable engine state (the listener may move cards
///   between piles, deal damage, etc).
/// * `watched_uid` — uid of the card whose late-effect is firing.
/// * `played_ctx` — context of the card whose play *triggered* the
///   late-effect sweep; mirrors the `played_card` argument in
///   Python's `register_late_effect` signature. Carries `card_id`,
///   `card_uid`, `card_type`, `card_tags`.
pub type LateEffectFn =
    fn(state: &mut CombatState, watched_uid: CardUid, played_ctx: &hooks::CardCtx);

static LATE_REGISTRY: OnceLock<RwLock<BTreeMap<CardId, LateEffectFn>>> = OnceLock::new();

/// Lock-free "the late-effect registry has ≥1 handler" flag. Set on every
/// registration (monotonic — handlers are only ever inserted, never
/// removed), so [`late_effects_empty`] can answer without taking the
/// global `RwLock`. `fire_late_effects` runs on every card play in the
/// MCTS rollout, and its empty-bus early-out was ~4% of fight wall purely
/// in `RwLock::read` futex traffic (2026-06-13 dispatch profile).
static LATE_EFFECTS_NONEMPTY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Dense frozen copy of [`LATE_REGISTRY`], read by [`fire_late_effects`] once
/// [`freeze_card_tables`] has published it.
static LATE_TABLE: FrozenCardTable<LateEffectFn> = FrozenCardTable::new();

fn late_registry() -> &'static RwLock<BTreeMap<CardId, LateEffectFn>> {
    LATE_REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register a single late-effect handler. Overwrites any previous
/// entry for `card_id`.
pub fn register_late_effect(card_id: CardId, handler: LateEffectFn) {
    let mut guard = late_registry()
        .write()
        .expect("late-effect registry poisoned");
    guard.insert(card_id, handler);
    LATE_TABLE.invalidate();
    drop(guard);
    LATE_EFFECTS_NONEMPTY.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Register many late-effect handlers in one go.
pub fn register_many_late_effects(entries: &[(CardId, LateEffectFn)]) {
    let mut guard = late_registry()
        .write()
        .expect("late-effect registry poisoned");
    for &(id, f) in entries {
        guard.insert(id, f);
    }
    LATE_TABLE.invalidate();
    drop(guard);
    if !entries.is_empty() {
        LATE_EFFECTS_NONEMPTY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Look up the registered late-effect for `card_id`, or `None`.
#[must_use]
pub fn lookup_late_effect(card_id: CardId) -> Option<LateEffectFn> {
    late_registry()
        .read()
        .expect("late-effect registry poisoned")
        .get(&card_id)
        .copied()
}

/// `true` iff no late-effect handlers are registered. Fast empty-bus
/// early-out for [`fire_late_effects`] — a lock-free atomic load (the
/// registry is insert-only, so the flag never goes stale in the
/// non-empty direction).
#[must_use]
pub fn late_effects_empty() -> bool {
    !LATE_EFFECTS_NONEMPTY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Walk every card in every player pile (Hand / Draw / Discard /
/// Exhaust / Play) and fire the registered late-effect for that
/// card's `CardId`, passing `played_ctx` as the triggering-play
/// context. No-op when the registry is empty.
///
/// Per-pile snapshots are taken before any listener runs, so a
/// late-effect that moves the watched card across piles does NOT
/// cause the same uid to fire twice (the snapshot was taken pre-move,
/// and we already deduplicate on uid).
pub fn fire_late_effects(state: &mut CombatState, played_ctx: &hooks::CardCtx) {
    if late_effects_empty() {
        return;
    }
    if let Some(table) = LATE_TABLE.get() {
        fire_late_effects_frozen(state, played_ctx, table);
        return;
    }

    // Snapshot the registered late-effect CardIds ONCE (the registry is
    // a handful of entries) so the per-card filter below is a cheap local
    // membership test, not a locked registry lookup per card.
    let registered: smallvec::SmallVec<[crate::cards::CardId; 8]> = late_registry()
        .read()
        .expect("late-effect registry poisoned")
        .keys()
        .copied()
        .collect();

    // Snapshot the uids whose CardId actually has a registered late-effect.
    // Order matches Python: hand first, then draw, discard, exhaust, play.
    // The dedup mirrors Python's `seen_ids: set[int]` guard.
    //
    // Bug `sts2-rs-65mbx` (GH #44): the prior body snapshotted EVERY pile
    // uid and deduplicated via an O(n) `seen.contains` scan, making the
    // whole sweep O(n²) in deck size. A combat kept open indefinitely
    // (e.g. the invincible coverage harness, where the player can't die)
    // lets a self-duplicating card such as Anger bloat the discard pile
    // into the thousands; the O(n²) dedup then turns a single `play_card`
    // into an effective infinite loop that no step/livelock guard can see.
    // Pre-filtering to handler-bearing cards keeps the snapshot — and its
    // dedup — proportional to the tiny set of cards that can actually fire,
    // so the sweep is O(n) over the deck regardless of how bloated it is.
    // Cards with no registered late-effect were always no-ops here, so this
    // is behaviourally identical (the second loop already skipped them).
    let mut visit: smallvec::SmallVec<[CardUid; 32]> = smallvec::SmallVec::new();
    let mut seen: smallvec::SmallVec<[CardUid; 32]> = smallvec::SmallVec::new();
    for pile in [
        crate::cards::PileKind::Hand,
        crate::cards::PileKind::Draw,
        crate::cards::PileKind::Discard,
        crate::cards::PileKind::Exhaust,
        crate::cards::PileKind::Play,
    ] {
        for uid in crate::cards::pile_iter(&state.player, pile) {
            let idx = uid.0.get() as usize - 1;
            let Some(card) = state.cards.get(idx) else {
                continue;
            };
            if !registered.contains(&card.card_id) {
                continue;
            }
            if !seen.contains(&uid) {
                seen.push(uid);
                visit.push(uid);
            }
        }
    }

    // Resolve every watched card's handler under a SINGLE registry read
    // lock. The prior code called `lookup_late_effect` per card, re-taking
    // the global `RwLock` for every card in every pile — that per-card
    // relock was ~2.6% of fight wall in the MCTS rollout profile
    // (2026-06-13 flamegraph). The registry is install-time-immutable, so
    // one snapshot covers the whole sweep. Collect (uid, handler) pairs and
    // release the guard BEFORE running any handler: handlers take
    // `&mut CombatState` and a re-entrant `register_late_effect` while we
    // held the read lock would deadlock.
    let mut to_fire: smallvec::SmallVec<[(CardUid, LateEffectFn); 8]> = smallvec::SmallVec::new();
    {
        let guard = late_registry()
            .read()
            .expect("late-effect registry poisoned");
        for watched_uid in visit {
            let idx = watched_uid.0.get() as usize - 1;
            let Some(card) = state.cards.get(idx) else {
                continue;
            };
            if let Some(&handler) = guard.get(&card.card_id) {
                to_fire.push((watched_uid, handler));
            }
        }
    }
    for (watched_uid, handler) in to_fire {
        handler(state, watched_uid, played_ctx);
    }
}

/// [`fire_late_effects`] over the frozen late-effect table: the same sweep
/// (pile order, uid dedup, handlers collected before any runs), with no lock
/// and one indexed load per card.
///
/// It first checks the combat's card arena. Every pile uid indexes that
/// arena, so if no arena card has a late effect the sweep would collect
/// nothing; that check costs one pass over the arena and skips the pile walk
/// on every play in a deck without MakeItSo / RightHandHand.
fn fire_late_effects_frozen(
    state: &mut CombatState,
    played_ctx: &hooks::CardCtx,
    table: &[Option<LateEffectFn>],
) {
    if !state
        .cards
        .iter()
        .any(|card| FrozenCardTable::entry(table, card.card_id).is_some())
    {
        return;
    }
    let mut to_fire: smallvec::SmallVec<[(CardUid, LateEffectFn); 8]> = smallvec::SmallVec::new();
    for pile in [
        crate::cards::PileKind::Hand,
        crate::cards::PileKind::Draw,
        crate::cards::PileKind::Discard,
        crate::cards::PileKind::Exhaust,
        crate::cards::PileKind::Play,
    ] {
        for uid in crate::cards::pile_iter(&state.player, pile) {
            let idx = uid.0.get() as usize - 1;
            let Some(card) = state.cards.get(idx) else {
                continue;
            };
            let Some(handler) = FrozenCardTable::entry(table, card.card_id) else {
                continue;
            };
            if !to_fire.iter().any(|&(seen, _)| seen == uid) {
                to_fire.push((uid, handler));
            }
        }
    }
    for (watched_uid, handler) in to_fire {
        handler(state, watched_uid, played_ctx);
    }
}

/// Publish the dense frozen copies of the on-play and late-effect
/// registries, or re-validate them against the registries after a
/// re-install. `sts2-content`'s card installer calls this as its last step;
/// until then (and after any later registration) both tables read the
/// locked registries. See the "Frozen dense card tables" section note.
pub fn freeze_card_tables() {
    {
        let guard = registry().read().expect("dispatch registry poisoned");
        PLAY_TABLE.freeze(&guard, |f| f as usize);
    }
    let guard = late_registry()
        .read()
        .expect("late-effect registry poisoned");
    LATE_TABLE.freeze(&guard, |f| f as usize);
}

/// `true` iff [`lookup`] and [`fire_late_effects`] are both reading their
/// frozen tables (test probe for the install ordering).
#[must_use]
pub fn card_tables_frozen() -> bool {
    PLAY_TABLE.get().is_some() && LATE_TABLE.get().is_some()
}

/// Number of currently-registered late-effect handlers.
#[must_use]
pub fn late_effects_registered_count() -> usize {
    late_registry()
        .read()
        .expect("late-effect registry poisoned")
        .len()
}

// ---------------------------------------------------------------------------
// Character-pool card sampler — bead `sts2-rs-ut1v` Part A
// ---------------------------------------------------------------------------
//
// Function-pointer registry that lets `sts2-core` hooks spawn random
// character-pool cards into hand from inside an `Effect` resolver
// (e.g. `CreativeAi.after_side_turn_start`) without taking a workspace
// dependency on `sts2-content` (the actual sampler lives in
// `sts2-content::cards::colorless::create_character_cards_into_arena`).
//
// `sts2-content::cards::dispatch::install_default_registry` installs the
// content-side trampoline once at boot. Unregistered → returns an empty
// `SmallVec`, so the engine no-ops gracefully in core-only test builds.

/// Sampler signature for "draw `count` random cards from the
/// `character`'s pool filtered to `card_type`, push them into
/// `state.cards`, return the new uids." Implementation lives in
/// `sts2-content::cards::colorless`.
pub type CharacterCardSamplerFn = fn(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
    card_type: Option<crate::cards::CardType>,
) -> smallvec::SmallVec<[CardUid; 4]>;

static CHARACTER_CARD_SAMPLER: OnceLock<CharacterCardSamplerFn> = OnceLock::new();

/// Register the content-side sampler trampoline. Idempotent — second
/// and subsequent calls are silent no-ops (matches `OnceLock`
/// semantics; the trampoline is content-static and never changes
/// across a process).
pub fn register_character_card_sampler(f: CharacterCardSamplerFn) {
    let _ = CHARACTER_CARD_SAMPLER.set(f);
}

/// Sample `count` cards from `character`'s pool (filtered to
/// `card_type` when `Some`), push them into the arena, and return
/// their uids. Returns an empty `SmallVec` when no sampler has been
/// registered (core-only test build).
#[must_use]
pub fn sample_character_cards(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
    card_type: Option<crate::cards::CardType>,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match CHARACTER_CARD_SAMPLER.get() {
        Some(f) => f(state, character, count, card_type),
        None => smallvec::SmallVec::new(),
    }
}

/// `true` iff a content-side sampler has been installed.
#[must_use]
pub fn character_card_sampler_installed() -> bool {
    CHARACTER_CARD_SAMPLER.get().is_some()
}

/// Sampler signature for "generate `count` cards from `character`'s pool
/// filtered to `card_type` **WITH replacement**" — i.e. C#
/// `CardFactory.GetForCombat` (uniform `NextItem` per slot, 1
/// `CombatCardGeneration` read each), as opposed to the distinct
/// shuffle-the-whole-pool [`CharacterCardSamplerFn`]
/// (`GetDistinctForCombat`). Same shape as [`CharacterCardSamplerFn`]
/// otherwise. Implementation lives in `sts2-content::cards::colorless`.
/// Used by the `CalamityPower.after_card_played` body (bead
/// `sts2-rs-q6bix`), which conjures `Amount` random Attacks per Attack
/// the owner plays.
pub type CharacterCardReplacementSamplerFn = fn(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
    card_type: Option<crate::cards::CardType>,
) -> smallvec::SmallVec<[CardUid; 4]>;

static CHARACTER_CARD_REPLACEMENT_SAMPLER: OnceLock<CharacterCardReplacementSamplerFn> =
    OnceLock::new();

/// Register the content-side with-replacement sampler trampoline.
/// Idempotent — matches [`register_character_card_sampler`] semantics.
pub fn register_character_card_replacement_sampler(f: CharacterCardReplacementSamplerFn) {
    let _ = CHARACTER_CARD_REPLACEMENT_SAMPLER.set(f);
}

/// Generate `count` cards from `character`'s pool (filtered to `card_type`
/// when `Some`) WITH replacement, push them into the arena, and return
/// their uids. Returns an empty `SmallVec` when no sampler has been
/// registered (core-only test build). Mirrors C# `GetForCombat`.
#[must_use]
pub fn sample_character_cards_with_replacement(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
    card_type: Option<crate::cards::CardType>,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match CHARACTER_CARD_REPLACEMENT_SAMPLER.get() {
        Some(f) => f(state, character, count, card_type),
        None => smallvec::SmallVec::new(),
    }
}

/// Trampoline signature for "transform the card at `card_uid` into a
/// random card from its transform-options pool, in place." Mirrors C#
/// `CardCmd.TransformToRandom(card, Rng.CombatCardSelection)` →
/// `CardFactory.CreateRandomCardForTransform` (pool =
/// `GetFilteredTransformationOptions`, pick = `rng.NextItem` =
/// `pool[NextInt(0,len)]`). Returns the new `CardUid` (the freshly minted
/// arena row reusing the original's pile slot), or `None` when the card
/// isn't found / the pool is empty / no trampoline installed.
/// Implementation lives in `sts2-content::cards::colorless::helpers`.
/// Bead `sts2-rs-isooo` (q6bix ENTROPY).
pub type TransformToRandomFn = fn(state: &mut CombatState, card_uid: CardUid) -> Option<CardUid>;

static TRANSFORM_TO_RANDOM: OnceLock<TransformToRandomFn> = OnceLock::new();

/// Register the content-side transform-to-random trampoline. Idempotent.
pub fn register_transform_to_random(f: TransformToRandomFn) {
    let _ = TRANSFORM_TO_RANDOM.set(f);
}

/// Transform the card at `card_uid` to a random card from its
/// transform-options pool, in place (RNG off `CombatCardSelection`).
/// Returns the new uid, or `None` when no trampoline is installed
/// (core-only test build) / the card is missing / the pool is empty.
pub fn transform_card_to_random(state: &mut CombatState, card_uid: CardUid) -> Option<CardUid> {
    // C# `CardCmd.Transform` returns nothing as its first statement once
    // `CombatManager.IsEnding` (v0.107.1 CardCmd.cs:371-374, v0.111.0
    // :373-376), before `GetReplacement` rolls
    // `CombatCardSelection` (Entropy's second transform after the first
    // one's Trash to Treasure Dark evoke killed the last enemy: Defect
    // v0.107.1 seed 6416).
    if crate::combat_loop::combat_is_ending(state) {
        return None;
    }
    match TRANSFORM_TO_RANDOM.get() {
        Some(f) => f(state, card_uid),
        None => None,
    }
}

/// Trampoline signature for the Regent **forge** primitive (C#
/// `ForgeCmd.Forge(amount, player, source)`). Ensures the player has at
/// least one active `SovereignBlade`, then adds `amount` damage to every
/// blade across all piles (incl. exhaust) and fires the `after_forge`
/// listener bus. Returns the number of blades that were buffed.
///
/// The implementation lives in
/// `sts2-content::cards::regent::helpers::combat_forge` (it must create a
/// `SovereignBlade` from a content-side `CardSpec`, which core cannot
/// reference without a workspace cycle). Power-triggered forges
/// (`FurnacePower.after_side_turn_start`) reach it through this trampoline
/// via `Effect::Forge`; card-triggered forges call `combat_forge`
/// directly. Bead `sts2-rs-jxz` (Furnace forge wiring).
pub type ForgeFn =
    fn(state: &mut CombatState, forger: CreatureUid, amount: i32, source: Option<CardUid>) -> usize;

static FORGE: OnceLock<ForgeFn> = OnceLock::new();

/// Register the content-side forge trampoline. Idempotent.
pub fn register_forge(f: ForgeFn) {
    let _ = FORGE.set(f);
}

/// Forge `amount` for `forger` (the Regent forge mechanic), routing
/// through the content-side trampoline. No-op returning 0 when no
/// trampoline is installed (core-only test build) — matching the
/// other content-delegating primitives. `source` is the originating
/// card uid (`None` for power-triggered forges).
pub fn forge(
    state: &mut CombatState,
    forger: CreatureUid,
    amount: i32,
    source: Option<CardUid>,
) -> usize {
    match FORGE.get() {
        Some(f) => f(state, forger, amount, source),
        None => 0,
    }
}

/// Bead `sts2-rs-1e0ae`: Combat→RunState gold accessor + spend
/// trampoline for `SealOfGold` (and any future combat-side listener
/// that needs to read/write `RunState.gold`). `CombatState` has no
/// `&mut RunState` back-edge at this pin; we route through a
/// closure registry installed at combat start by the run-loop
/// driver. Core-only test builds (no accessor installed) fall back
/// to read=0, spend=no-op, so listeners gate on `try_spend_player_gold`
/// returning `true` before applying their effect.
///
/// The closure shape is intentionally minimal: `Fn(amount: i32) ->
/// bool` returns whether the spend succeeded (false ⇒ insufficient
/// gold or no accessor). The read-side companion returns the
/// current gold balance. Idempotent registration via `OnceLock`.
pub type PlayerGoldReadFn = fn() -> i32;
pub type PlayerGoldSpendFn = fn(amount: i32) -> bool;

static PLAYER_GOLD_READ: OnceLock<PlayerGoldReadFn> = OnceLock::new();
static PLAYER_GOLD_SPEND: OnceLock<PlayerGoldSpendFn> = OnceLock::new();

/// Register the gold-read accessor. Idempotent.
pub fn register_player_gold_read(f: PlayerGoldReadFn) {
    let _ = PLAYER_GOLD_READ.set(f);
}

/// Register the gold-spend accessor. Idempotent.
pub fn register_player_gold_spend(f: PlayerGoldSpendFn) {
    let _ = PLAYER_GOLD_SPEND.set(f);
}

/// Read the player's current gold via the registered accessor; returns
/// `0` when no accessor is installed (core-only test build).
#[must_use]
pub fn read_player_gold() -> i32 {
    PLAYER_GOLD_READ.get().map_or(0, |f| f())
}

/// Attempt to spend `amount` gold via the registered accessor. Returns
/// `true` on success, `false` when the spend fails (insufficient gold
/// or no accessor installed). The accessor is expected to mutate the
/// run-state's gold atomically and return `false` if the balance
/// would go negative — listeners can use the return value as the
/// "gate-and-apply" handshake.
pub fn try_spend_player_gold(amount: i32) -> bool {
    PLAYER_GOLD_SPEND.get().is_some_and(|f| f(amount))
}

/// C# `PlayerCmd.GainGold` (`PlayerCmd.cs`) for an in-combat gain, on the
/// combat-time gold mirror [`CombatState::gold`] that `exit_combat` writes
/// back to the run (the RP17 path Seal of Gold spends from). Folds
/// `Hook.ModifyGoldGained` — Ectoplasm zeroes the gain, Bowler Hat
/// multiplies it by 1.25 (`Ectoplasm.cs:18`, `BowlerHat.cs:25`) — truncates
/// like `player.Gold += (int)amount`, and runs `Hook.AfterGoldGained`
/// (Dragon Fruit: +1 max HP, `DragonFruit.cs:22`). Returns the gold credited.
pub fn gain_gold_in_combat(state: &mut CombatState, amount: i32) -> i32 {
    use crate::RelicKind;
    if amount <= 0 {
        return 0;
    }
    let mut gained = i64::from(amount);
    if state.relics.owns(RelicKind::BowlerHat) {
        // `amount * 1.25m`, exact in decimal; `(int)` truncates.
        gained = gained * 5 / 4;
    }
    if state.relics.owns(RelicKind::Ectoplasm) {
        gained = 0;
    }
    let gained = i32::try_from(gained).unwrap_or(i32::MAX);
    if gained <= 0 {
        return 0;
    }
    state.gold = state.gold.saturating_add(gained);
    if state.relics.owns(RelicKind::DragonFruit) {
        if let Some(player_uid) = state.player.creature_uid {
            let _ = state.creature_gain_max_hp(player_uid, 1);
        }
    }
    gained
}

/// Bead `sts2-rs-39ofc`: the gain-gold companion to the read/spend
/// trampoline above. `HandOfGreed`'s on-fatal-kill rider (C#
/// `HandOfGreed.OnPlay` → `PlayerCmd.GainGold`) is a run-economy effect
/// that adds gold to `RunState`; the combat engine has no `&mut RunState`
/// back-edge at this pin, so it routes through this closure registry
/// installed by the run-loop driver (symmetric with `PLAYER_GOLD_SPEND`).
/// Core-only test builds (no accessor installed) fall back to a no-op,
/// matching `read_player_gold`/`try_spend_player_gold`.
pub type PlayerGoldGainFn = fn(amount: i32);

static PLAYER_GOLD_GAIN: OnceLock<PlayerGoldGainFn> = OnceLock::new();

/// Register the gold-gain accessor. Idempotent.
pub fn register_player_gold_gain(f: PlayerGoldGainFn) {
    let _ = PLAYER_GOLD_GAIN.set(f);
}

/// Add `amount` gold to the run-state via the registered accessor.
/// No-op when `amount <= 0` or no accessor is installed (core-only test
/// build / no run-loop driver). Mirrors C# `PlayerCmd.GainGold`, which
/// ignores non-positive amounts.
pub fn gain_player_gold(amount: i32) {
    if amount <= 0 {
        return;
    }
    if let Some(f) = PLAYER_GOLD_GAIN.get() {
        f(amount);
    }
}

/// Crossbow's turn-start card: one random Attack from the player's
/// character pool, free this turn, added to hand. C# `Crossbow.cs:17-37`
/// (the same in v0.107.1 and v0.111.0): `GetDistinctForCombat(Owner,
/// attacks, 1, Rng.CombatCardGeneration)`, `SetToFreeThisTurn()` on the
/// card, then `CardPileCmd.AddGeneratedCardsToCombat(list, Hand, Owner)`.
///
/// `AddGeneratedCardsToCombat` (`CardPileCmd.cs:281-314` in v0.111.0,
/// `:216-249` in v0.107.1) does three things per card, in this order:
/// `History.CardGenerated(cs, card, creator)`, `Add(card, Hand)`, and
/// `Hook.AfterCardGeneratedForCombat(cs, card, creator)`. `Add` puts the
/// card on the discard pile instead when the hand already holds
/// `MaxCardsInHand` (10) cards (`CardPileCmd.cs:485-489` in v0.111.0,
/// `:420-424` in v0.107.1), and it fires `AfterCardEnteredCombat` because
/// the card had no pile before. Here [`crate::cards::try_add_to_hand`]
/// does the placement and [`fire_card_generation_listeners`] does the
/// rest: the `CardGenerated` event-log row, the Supermassive tally, the
/// entered-combat listeners (Regalite) and the generated-for-combat
/// powers. The creator is the player, so `added_by_player` is `true`.
///
/// The previous body pushed the card into hand with no check, so a turn
/// that started with 10 cards in hand ended with 11, and none of the
/// listeners ran.
///
/// `LocalCostMod::FREE_THIS_TURN` stands in for both halves of
/// `SetToFreeThisTurn()` (energy and stars). It is set before the card is
/// placed, as in C#.
///
/// No-op when the player has no `character`, when no content-side
/// `CharacterCardSamplerFn` is installed (core-only test build), or when
/// the character pool has no Attack.
pub fn generate_free_attack_in_hand(state: &mut CombatState) {
    let Some(character) = state.player.character else {
        return;
    };
    let sampled = sample_character_cards(state, character, 1, Some(crate::cards::CardType::Attack));
    for &uid in &sampled {
        set_card_free_this_turn(state, uid);
        let pile = if crate::cards::try_add_to_hand(&mut state.player, uid) {
            PileKind::Hand
        } else {
            PileKind::Discard
        };
        fire_card_generation_listeners(state, uid, pile, true);
    }
}

/// Content-side sampler signature for character-pool ethereal-only
/// card generation. Mirrors Python `combat.generate_ethereal_cards`
/// at `sts2-rl-agent@01c094d:sts2_env/core/combat.py:2163` —
/// `create_distinct_character_cards(character_id, …, require_keyword="ethereal")`.
/// The implementation lives in `sts2-content` because the per-character
/// `*_SPECS` slices are content-static; this trampoline lets sts2-core
/// hooks (BigHat — bead `sts2-rs-fa2o`) sample ethereal cards without
/// taking a workspace dep on sts2-content. Bead `sts2-rs-fa2o`.
pub type CharacterEtherealCardSamplerFn = fn(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]>;

static CHARACTER_ETHEREAL_CARD_SAMPLER: OnceLock<CharacterEtherealCardSamplerFn> = OnceLock::new();

/// Register the content-side ethereal-card sampler. Idempotent —
/// matches `CharacterCardSamplerFn` semantics.
pub fn register_character_ethereal_card_sampler(f: CharacterEtherealCardSamplerFn) {
    let _ = CHARACTER_ETHEREAL_CARD_SAMPLER.set(f);
}

/// Sample up to `count` distinct ethereal cards from `character`'s
/// pool, push them into the arena, and return their uids. Returns
/// an empty `SmallVec` when no sampler has been registered (core-only
/// test build) or when the character's pool contains no ethereal
/// cards. Bead `sts2-rs-fa2o`.
#[must_use]
pub fn sample_character_ethereal_cards(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match CHARACTER_ETHEREAL_CARD_SAMPLER.get() {
        Some(f) => f(state, character, count),
        None => smallvec::SmallVec::new(),
    }
}

/// Sampler signature for "draw `count` distinct **Common**-rarity cards from
/// `character`'s pool" — C# `HelloWorldPower.BeforeHandDraw`
/// (`HelloWorldPower.cs:21-29`, GH #636), whose LINQ narrows the pool with
/// `where c.Rarity == CardRarity.Common` before `GetDistinctForCombat`.
///
/// Its own trampoline rather than a rarity parameter on
/// [`CharacterCardSamplerFn`] because `CardRarity` is a `sts2-content` type
/// that sts2-core cannot name — the rarity predicate stays entirely
/// content-side, the same split every sibling sampler here uses.
pub type CharacterCommonCardSamplerFn = fn(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]>;

static CHARACTER_COMMON_CARD_SAMPLER: OnceLock<CharacterCommonCardSamplerFn> = OnceLock::new();

/// Register the content-side Common-rarity sampler. Idempotent — matches
/// [`register_character_card_sampler`] semantics.
pub fn register_character_common_card_sampler(f: CharacterCommonCardSamplerFn) {
    let _ = CHARACTER_COMMON_CARD_SAMPLER.set(f);
}

/// Sample `count` distinct Common cards from `character`'s pool, push them
/// into the arena, and return their uids. Returns an empty `SmallVec` when no
/// sampler has been registered (core-only test build) or when the character's
/// pool holds no Common cards. Used by the `HelloWorldPower.BeforeHandDraw`
/// body in [`crate::hooks::dispatch_before_hand_draw_powers`].
#[must_use]
pub fn sample_character_common_cards(
    state: &mut CombatState,
    character: crate::potions::CharacterId,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match CHARACTER_COMMON_CARD_SAMPLER.get() {
        Some(f) => f(state, character, count),
        None => smallvec::SmallVec::new(),
    }
}

/// Content-side sampler signature for character-independent colorless
/// card generation. Mirrors Python `combat.generate_colorless_cards`
/// at `sts2-rl-agent@01c094d:sts2_env/core/combat.py:2182` — samples
/// WITH replacement from the colorless registry
/// (`[create_card(rng.choice(colorless_ids)) for _ in range(count)]`).
/// Independent of [`CharacterEtherealCardSamplerFn`] because the
/// colorless pool is not character-keyed. The implementation lives in
/// sts2-content; this trampoline lets sts2-core hooks (OrangeDough —
/// bead `sts2-rs-7xih`, VexingPuzzlebox — bead `sts2-rs-t7vn`) sample
/// colorless cards without taking a workspace dep on sts2-content.
/// Bead `sts2-rs-7xih`.
pub type ColorlessCardSamplerFn =
    fn(state: &mut CombatState, count: usize) -> smallvec::SmallVec<[CardUid; 4]>;

static COLORLESS_CARD_SAMPLER: OnceLock<ColorlessCardSamplerFn> = OnceLock::new();

/// Register the content-side colorless-card sampler. Idempotent.
pub fn register_colorless_card_sampler(f: ColorlessCardSamplerFn) {
    let _ = COLORLESS_CARD_SAMPLER.set(f);
}

/// Sample `count` colorless cards WITH replacement (Python parity),
/// push them into the arena, and return their uids. Returns an empty
/// `SmallVec` when no sampler has been registered (core-only test
/// build) or when the colorless pool is empty. Bead `sts2-rs-7xih`.
#[must_use]
pub fn sample_colorless_cards(
    state: &mut CombatState,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match COLORLESS_CARD_SAMPLER.get() {
        Some(f) => f(state, count),
        None => smallvec::SmallVec::new(),
    }
}

static DISTINCT_COLORLESS_CARD_SAMPLER: OnceLock<ColorlessCardSamplerFn> = OnceLock::new();

/// Register the content-side **distinct** colorless-card sampler. Idempotent.
/// Mirrors [`register_colorless_card_sampler`] but the implementation draws
/// `count` DISTINCT cards from the filtered `ColorlessCardPool` combat pool via
/// the `CombatCardGeneration` rng (C# `CardFactory.GetDistinctForCombat` — a
/// whole-pool `UnstableShuffle` then take `count`), as opposed to the
/// with-replacement `GetForCombat` sampler above.
pub fn register_distinct_colorless_card_sampler(f: ColorlessCardSamplerFn) {
    let _ = DISTINCT_COLORLESS_CARD_SAMPLER.set(f);
}

/// Sample `count` **distinct** colorless cards (C# `GetDistinctForCombat`),
/// push them into the arena, and return their uids. Returns an empty
/// `SmallVec` when no sampler has been registered (core-only test build) or
/// when the colorless pool is empty. Used by the SpectrumShiftPower
/// `BeforeHandDraw` body.
#[must_use]
pub fn sample_distinct_colorless_cards(
    state: &mut CombatState,
    count: usize,
) -> smallvec::SmallVec<[CardUid; 4]> {
    match DISTINCT_COLORLESS_CARD_SAMPLER.get() {
        Some(f) => f(state, count),
        None => smallvec::SmallVec::new(),
    }
}

/// Route `uids` into the player's hand, spilling overflow to discard
/// at [`crate::cards::HAND_LIMIT`]. Mirrors
/// `sts2-content::potions::dispatch::route_generated_into_hand`.
/// Increments `generated_cards_this_combat`. Bead `sts2-rs-ut1v`.
///
/// Bead `sts2-rs-regalite-gen`: fires `after_card_entered_combat` per card
/// after it lands, mirroring C# `CardPileCmd.AddGeneratedCardsToCombat`,
/// which calls `Hook.AfterCardEnteredCombat` for every freshly-generated
/// card whose target is a combat pile (CardPileCmd.cs:436-439). Without
/// this, REGALITE's `AfterCardEnteredCombat` (+2 block per card entering
/// combat) — and any other on-entry listener — never fired for cards made
/// by OrobicAcid / CosmicConcoction / ManifestAuthority / BundleOfJoy
/// (seed 1652: REGALITE owed 3×2 = 6 block off OrobicAcid's 3 generated
/// cards, RS granted 0). The Splash path already fires it via
/// `spawn_card_into_hand`; this brings the batch-route path in line.
pub fn route_generated_into_hand(state: &mut CombatState, uids: &[CardUid]) {
    for &uid in uids {
        // A generated card has no `DeckVersion`: Thieving Hopper never steals
        // it (loadout Regent 1255 v0.107.1: Rust stole an Orange Dough Mind
        // Blast instead of a deck card).
        if let Some(card) = state.cards.get_mut(uid.0.get() as usize - 1) {
            stamp_generated_in_combat(card);
        }
        let pile = if combat_pile_add_blocked(state) {
            // `CardPileCmd.Add` fails once the combat is ending.
            PileKind::Hand
        } else if state.player.hand.len() < crate::cards::HAND_LIMIT {
            state.player.hand.push(uid);
            PileKind::Hand
        } else {
            state.player.discard.push(uid);
            PileKind::Discard
        };
        // The shared funnel: the `CardGenerated` event-log row (this path used
        // to write none, #1105), the Supermassive tally, the entered-combat bus
        // and the `AfterCardGeneratedForCombat` listeners. Every caller of this "player generates cards into hand" path passes
        // C# `addedByPlayer: true` (SpectrumShift / CallOfTheVoid / SentryMode
        // turn-start powers, BigHat / VexingPuzzlebox / OrangeDough relics,
        // potions, GenerateCards effect, DevourLife) — verified against the C#
        // `AddGeneratedCardsToCombat(..., addedByPlayer: true)` call sites. The
        // prior port fired NONE of the AfterCardGeneratedForCombat listeners
        // here, so a turn-start SpectrumShift card granted no Regent
        // PillarOfCreation block / Arsenal Strength and no Smokestack damage,
        // even though the Play pile is empty at turn start (parity fuzz Regent
        // seeds 422/494). Fire them with `added_by_player = true` (not the
        // Play-pile heuristic the spawn path uses). Bead
        // `sts2-rs-gen-added-by-player`.
        fire_card_generation_listeners(state, uid, pile, true);
    }
}

// ---------------------------------------------------------------------------
// Default handlers (Strike / Defend / Bash / AscendersBane noop)
// ---------------------------------------------------------------------------
//
// The three shipped Ironclad starter cards live here so that sts2-core's
// own integration tests (`crates/sts2-core/tests/play_card.rs`) can install
// the registry without taking a dev-dependency on `sts2-content` (which
// would create a workspace cycle).
//
// `sts2-content::cards::dispatch::install_default_registry` calls this
// function; downstream beads (P6-3 .. P6-7) register their per-character
// pools from `sts2-content::cards::<character>.rs` via `register_many`.
//
// Source: `sts2-rl-agent@01c094d:sts2_env/cards/ironclad_basic.py`.

use std::sync::Once;

static INSTALL_ONCE: Once = Once::new();

/// Install the core/starter dispatch entries (Strike / Defend / Bash /
/// `AscendersBane`). Idempotent — safe to call from multiple boot paths.
/// Downstream character beads must register their pools separately.
pub fn install_default_registry() {
    INSTALL_ONCE.call_once(|| {
        register_many(&[
            (CardId::StrikeIronclad, play_strike_ironclad as PlayCardFn),
            (CardId::DefendIronclad, play_defend_ironclad as PlayCardFn),
            (CardId::Bash, play_bash as PlayCardFn),
            (CardId::AscendersBane, play_noop as PlayCardFn),
        ]);
    });
}

/// Idempotent alias for [`install_default_registry`]. Convenient for tests.
pub fn ensure_installed() {
    install_default_registry();
}

/// Strike (Ironclad). Deal `card.base_damage` to a single enemy.
/// Ports `make_strike_ironclad` in
/// `sts2-rl-agent@01c094d:sts2_env/cards/ironclad_basic.py:48-57`.
fn play_strike_ironclad(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Strike: Enemy-targeted, engine validated");
    let base = current_card_damage(state);
    let _ = deal_damage_player_to_enemy(state, t, base);
    EffectResult::Resolved
}

/// Defend (Ironclad). Gain `card.base_block` block on the player.
/// Ports `make_defend_ironclad` in `ironclad_basic.py:60-69`.
fn play_defend_ironclad(state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    let base = current_card_block(state);
    gain_block_player(state, base);
    EffectResult::Resolved
}

/// Bash. Deal `card.base_damage` then apply Vulnerable(2) to the target.
/// Vulnerable is applied AFTER damage to match the `BashAttack.use` order
/// at `sts2_env/cards/ironclad/bash.py`; the dead-target guard is folded
/// into `apply_power_to_enemy`.
fn play_bash(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Bash: Enemy-targeted, engine validated");
    let base = current_card_damage(state);
    let _ = deal_damage_player_to_enemy(state, t, base);
    // Bead `sts2-rs-orim9`: C# `Bash.cs` `PowerVar<VulnerablePower>(2)` with
    // `OnUpgrade UpgradeValueBy(1m)` → 2 base / 3 upgraded.
    let amt = if current_card_upgraded(state) { 3 } else { 2 };
    apply_power_to_enemy(state, t, PowerKind::Vulnerable, amt);
    EffectResult::Resolved
}

fn current_card_upgraded(state: &CombatState) -> bool {
    resolving_card_upgraded(state)
}

/// Noop handler — used for unplayable curses like `AscendersBane` that the
/// engine still routes through dispatch for uniformity.
fn play_noop(_state: &mut CombatState, _target: Option<CreatureUid>) -> EffectResult {
    EffectResult::Resolved
}

fn current_card_damage(state: &CombatState) -> i32 {
    resolving_card_base_damage(state)
}

fn current_card_block(state: &CombatState) -> i32 {
    resolving_card_base_block(state)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind};

    use smallvec::SmallVec;

    /// `PlayerCmd.GainGold` in combat: credits the gold mirror, Bowler Hat
    /// ×1.25 truncated, Ectoplasm zeroes it (`BowlerHat.cs:25`,
    /// `Ectoplasm.cs:18`).
    #[test]
    fn gain_gold_in_combat_credits_the_mirror_through_gold_hooks() {
        let relic = |kind| crate::Relic {
            id: NonZeroU32::new(1).expect("non-zero"),
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        };
        let (mut state, _) = fresh_state_with_one_enemy();
        state.gold = 9;
        assert_eq!(gain_gold_in_combat(&mut state, 20), 20);
        assert_eq!(state.gold, 29);
        state.relics.push(relic(crate::RelicKind::BowlerHat));
        assert_eq!(gain_gold_in_combat(&mut state, 25), 31);
        assert_eq!(state.gold, 60);
        state.relics.push(relic(crate::RelicKind::Ectoplasm));
        assert_eq!(gain_gold_in_combat(&mut state, 20), 0);
        assert_eq!(state.gold, 60);
    }

    /// Crossbow sets its generated card free this turn BEFORE
    /// `AddGeneratedCardsToCombat` (Crossbow.cs:17-37), so a Banshee's Cry it
    /// makes takes its `AfterCardEnteredCombat` `AddThisCombat(-2n)`
    /// (BansheesCry.cs:42-55) on top of that local zero: local cost `-2`, and
    /// `Hook.ModifyEnergyCostInCombat` skips Borrowed Time's +1 on a negative
    /// cost — 0, not 1 (Necrobinder 6580, v0.107.1). Once the zero expires the
    /// card reads `9 - 2 = 7`.
    #[test]
    fn banshees_cry_generated_free_shows_its_shave_on_the_zero() {
        let (mut state, _) = fresh_state_with_one_enemy();
        state.combat_log.ethereal_cards_played_this_combat = 1;
        let pid = NonZeroU32::new(1).expect("non-zero");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::BorrowedTime,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        let mut card = crate::cards::Card::pod_defaults();
        card.card_id = crate::cards::CardId::BansheesCry;
        card.cost = crate::CardEnergyCost::new(9);
        card.card_type = crate::cards::CardType::Attack;
        let uid = state.add_card_to_draw_pile(card);
        state.player.draw.retain(|u| *u != uid);
        state.cards[uid.0.get() as usize - 1].set_to_free_this_turn();
        assert!(crate::cards::try_add_to_hand(&mut state.player, uid));
        fire_card_generation_listeners(&mut state, uid, PileKind::Hand, true);
        assert_eq!(crate::hooks::effective_card_cost(&state, uid), 0);
        state.cards[uid.0.get() as usize - 1]
            .cost
            .end_of_turn_cleanup();
        let card = &state.cards[uid.0.get() as usize - 1];
        assert_eq!(card.local_cost(), 7);
    }

    /// The same fold for every other `AddThisCombat` on a card that is not
    /// the one being played: Melancholy's on-death `AddThisCombat(-1)` on a
    /// free-this-turn copy reads `0 - 1`, so Borrowed Time's +1 is skipped —
    /// 0, not 1. Once the zero expires it reads `3 - 1 = 2`.
    #[test]
    fn melancholy_shave_on_a_free_copy_shows_on_the_zero() {
        let (mut state, _) = fresh_state_with_one_enemy();
        let pid = NonZeroU32::new(1).expect("non-zero");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::BorrowedTime,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        let mut card = crate::cards::Card::pod_defaults();
        card.card_id = crate::cards::CardId::Melancholy;
        card.cost = crate::CardEnergyCost::new(3);
        card.card_type = crate::cards::CardType::Skill;
        let uid = state.add_card_to_draw_pile(card);
        state.player.draw.retain(|u| *u != uid);
        state.cards[uid.0.get() as usize - 1].set_to_free_this_turn();
        assert!(crate::cards::try_add_to_hand(&mut state.player, uid));
        assert_eq!(crate::hooks::effective_card_cost(&state, uid), 1);
        state.reduce_melancholy_cost_walk();
        assert_eq!(crate::hooks::effective_card_cost(&state, uid), 0);
        state.cards[uid.0.get() as usize - 1]
            .cost
            .end_of_turn_cleanup();
        let card = &state.cards[uid.0.get() as usize - 1];
        assert_eq!(card.local_cost(), 2);
    }

    fn fresh_state_with_one_enemy() -> (CombatState, CreatureUid) {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Alive player: a dead one means combat is ending, and C#
        // `IterateCombatHookListeners` then runs no generation / entered-combat
        // listener (`Hook.cs:177-207`).
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let id = NonZeroU32::new(1).expect("non-zero");
        state.creatures.push(Creature {
            id,
            current_hp: 30,
            max_hp: 30,
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
        (state, uid)
    }

    /// gh #181: a draw-pile `CardPilePosition.Random` insert (C#
    /// `CardPileCmd.AddInternal(card, Rng.Shuffle.NextInt(Count+1))`,
    /// `CardPileCmd.cs:515`) draws EXACTLY ONE read off the run-level
    /// `shuffle` stream and lands the card at an in-bounds index. The earlier
    /// Noisebot top-of-draw simplification skipped this read, desyncing every
    /// later reshuffle (DRAW_ORDER/RNG_DESYNC/PILE_COUNT|end_turn family).
    #[test]
    fn spawn_card_into_draw_pile_random_consumes_one_shuffle_read() {
        let (mut state, _uid) = fresh_state_with_one_enemy();
        // Seed a 3-card draw pile so the random index has > 1 slot.
        for i in 1..=3u32 {
            state
                .player
                .draw
                .push(CardUid::new(i + 100).expect("non-zero"));
        }
        let before = state.run_rng.shuffle.counter();
        let len_before = state.player.draw.len();
        let uid = spawn_card_into_draw_pile_random(&mut state, wound_card_pod());
        assert_eq!(
            state.run_rng.shuffle.counter(),
            before + 1,
            "exactly one shuffle read per random-position draw insert"
        );
        assert_eq!(state.player.draw.len(), len_before + 1, "card entered draw");
        let idx = state
            .player
            .draw
            .iter()
            .position(|&u| u == uid)
            .expect("inserted card present in draw pile");
        assert!(idx <= len_before, "inserted within inclusive [0, len]");
    }

    /// gh #181: C# `Rng.NextInt(maxExclusive)` bumps `Counter` unconditionally
    /// (`Rng.cs:90-94`), so even an empty-draw `NextInt(1)` consumes one read —
    /// the random insert must NOT short-circuit when the draw pile is empty.
    #[test]
    fn spawn_card_into_draw_pile_random_reads_shuffle_even_when_empty() {
        let (mut state, _uid) = fresh_state_with_one_enemy();
        assert!(state.player.draw.is_empty());
        let before = state.run_rng.shuffle.counter();
        let uid = spawn_card_into_draw_pile_random(&mut state, wound_card_pod());
        assert_eq!(
            state.run_rng.shuffle.counter(),
            before + 1,
            "empty-draw random insert still draws one shuffle read (C# NextInt(1) Counter++)"
        );
        assert_eq!(state.player.draw.as_slice(), &[uid]);
    }

    #[test]
    fn apply_power_to_enemy_zero_amount_is_noop() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 0);
        assert!(state.powers.is_empty());
        assert!(state.creatures[0].powers.is_empty());
    }

    #[test]
    fn apply_power_to_enemy_creates_then_stacks() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 2);
        assert_eq!(state.powers.len(), 1);
        assert_eq!(state.powers[0].amount, 2);
        // Second application stacks into the existing instance.
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 3);
        assert_eq!(state.powers.len(), 1, "stacking, not duplicating");
        assert_eq!(state.powers[0].amount, 5);
    }

    /// Bead `sts2-rs-2z2sn`: `TagTeamPower` is `IsInstanced=true` in C#
    /// (`TagTeamPower.cs:21`), so two `TagTeam.OnPlay`
    /// `PowerCmd.Apply<TagTeamPower>(target, 1)` calls leave TWO separate
    /// amount-1 instances — they must NOT stack-merge into one amount-2
    /// stack (coverage-sweep seed 1702 turn 3: `powers.len` cs=2 rs=1,
    /// `powers[1].amount` cs=1 rs=2). Contrast with the stacking
    /// (non-instanced) Vulnerable in `apply_power_to_enemy_creates_then_stacks`.
    #[test]
    fn apply_instanced_tag_team_does_not_merge() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::TagTeam, 1);
        assert_eq!(state.powers.len(), 1);
        assert_eq!(state.powers[0].amount, 1);

        // Second application of an instanced power creates a FRESH instance
        // instead of stacking into the first.
        apply_power_to_enemy(&mut state, uid, PowerKind::TagTeam, 1);
        assert_eq!(
            state.powers.len(),
            2,
            "instanced TagTeam: two separate instances, not a merged stack"
        );
        assert!(
            state
                .powers
                .iter()
                .all(|p| p.kind == PowerKind::TagTeam && p.amount == 1),
            "each TagTeam instance carries amount 1"
        );
        assert_eq!(
            state.creatures[0].powers.len(),
            2,
            "the creature links both TagTeam instances"
        );
    }

    /// Bead `sts2-rs-t2jqf`: `KnockdownPower` is `IsInstanced=true` in C#
    /// (`KnockdownPower.cs:21-22`), so two `Knockdown.OnPlay`
    /// `PowerCmd.Apply<KnockdownPower>(target, 2)` calls (e.g. two KNOCKDOWN
    /// cards auto-played by DISTILLED_CHAOS) leave TWO separate amount-2
    /// instances — they must NOT stack-merge into one amount-4 stack
    /// (sweep-v2 full seed 1224 turn 1: `monsters[0].powers.len` cs=2 rs=1,
    /// `powers[0].amount` cs=2 rs=4). Mirrors the TagTeam test above.
    #[test]
    fn apply_instanced_knockdown_does_not_merge() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Knockdown, 2);
        assert_eq!(state.powers.len(), 1);
        assert_eq!(state.powers[0].amount, 2);

        // Second application of an instanced power creates a FRESH instance
        // instead of stacking the amount-2 into one amount-4 stack.
        apply_power_to_enemy(&mut state, uid, PowerKind::Knockdown, 2);
        assert_eq!(
            state.powers.len(),
            2,
            "instanced Knockdown: two separate instances, not a merged stack"
        );
        assert!(
            state
                .powers
                .iter()
                .all(|p| p.kind == PowerKind::Knockdown && p.amount == 2),
            "each Knockdown instance carries amount 2"
        );
        assert_eq!(
            state.creatures[0].powers.len(),
            2,
            "the creature links both Knockdown instances"
        );
    }

    /// Bead `sts2-rs-orbit-inst` / `sts2-rs-monologue-inst`: `OrbitPower` and
    /// `MonologuePower` are `InstanceType => Instanced` in v0.107.1, so two
    /// applications must leave TWO separate amount-1 instances, not one
    /// amount-2 stack (Regent sweep PLAYER_STATE|ORBIT + PLAYER_STATE|MONOLOGUE:
    /// `player.powers.len` cs=2 rs=1). The instancing decision lives in
    /// `apply_power_to_creature` and is creature-agnostic, so it is pinned here
    /// on an enemy creature like the TagTeam/Knockdown tests above.
    #[test]
    fn apply_instanced_orbit_and_monologue_do_not_merge() {
        assert!(power_kind_is_instanced(PowerKind::Orbit));
        assert!(power_kind_is_instanced(PowerKind::Monologue));
        for kind in [PowerKind::Orbit, PowerKind::Monologue] {
            let (mut state, uid) = fresh_state_with_one_enemy();
            apply_power_to_enemy(&mut state, uid, kind, 1);
            apply_power_to_enemy(&mut state, uid, kind, 1);
            assert_eq!(
                state.powers.len(),
                2,
                "instanced {kind:?}: two separate instances, not a merged stack"
            );
            assert!(
                state.powers.iter().all(|p| p.kind == kind && p.amount == 1),
                "each {kind:?} instance carries amount 1"
            );
            assert_eq!(
                state.creatures[0].powers.len(),
                2,
                "the creature links both {kind:?} instances"
            );
        }
    }

    /// Bead `sts2-rs-nightmare-inst`: `NightmarePower.InstanceType => Instanced`
    /// (v0.107.1 NightmarePower.cs:30). Two Nightmare plays must leave TWO
    /// instances — a merge would collapse them into one amount-6 stack whose
    /// single `counter` (the selected-card uid) would overwrite the first
    /// play's card, so next turn only ONE card clones (six times) instead of
    /// two cards (three each). The turn-start consumer already reads per
    /// instance; this pins the apply-side instancing that feeds it.
    #[test]
    fn apply_instanced_nightmare_does_not_merge() {
        assert!(power_kind_is_instanced(PowerKind::Nightmare));
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Nightmare, 3);
        apply_power_to_enemy(&mut state, uid, PowerKind::Nightmare, 3);
        assert_eq!(
            state.powers.len(),
            2,
            "instanced Nightmare: two separate instances, not a merged amount-6 stack"
        );
        assert!(
            state
                .powers
                .iter()
                .all(|p| p.kind == PowerKind::Nightmare && p.amount == 3),
            "each Nightmare instance keeps its own clone-count 3"
        );
    }

    /// The general `apply_power_to_creature` path stays alive-gated: Rust
    /// retains corpses past C#'s cleanup, so a blanket relaxation over-applies
    /// (Regent full sweep: +14 divergent). Only the synchronous killing-blow
    /// reaction (`apply_power_to_creature_allow_dead`, used by MonarchsGaze
    /// `AfterDamageGiven`) may target a corpse.
    #[test]
    fn apply_power_to_dead_enemy_is_noop() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        state.creatures[0].current_hp = 0;
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 2);
        assert!(state.powers.is_empty());
    }

    /// `apply_power_to_creature_allow_dead` is the narrow exception: a
    /// dead-but-present corpse DOES receive the power (C# `CanReceivePowers`),
    /// so MonarchsGaze's on-kill StrengthDown lands (Regent 2094/4871).
    #[test]
    fn apply_power_allow_dead_targets_a_corpse() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        // A living second enemy: with the corpse the last one the combat
        // `IsEnding`, and `PowerCmd.Apply` lands nothing (PowerCmd.cs:69).
        let other = NonZeroU32::new(2).expect("non-zero");
        let mut second = state.creatures[0].clone();
        second.id = other;
        state.creatures.push(second);
        state.enemy_uids.push(CreatureUid(other));
        state.creatures[0].current_hp = 0;
        apply_power_to_creature_allow_dead(&mut state, uid, PowerKind::MonarchsGazeStrengthDown, 2);
        assert_eq!(state.powers.len(), 1, "corpse receives the power");
        assert_eq!(state.powers[0].kind, PowerKind::MonarchsGazeStrengthDown);
        assert_eq!(state.creatures[0].powers.len(), 1, "linked on the corpse");
    }

    /// GH #1054 item 5: any `DieForYou` in the fight (Osty's, here on the
    /// enemy only to keep the fixture small) makes C#'s `ShouldAllowHitting`
    /// refuse every corpse, so neither allow_dead path lands anything.
    #[test]
    fn allow_dead_blocked_while_any_die_for_you_is_present() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::DieForYou, 1);
        state.creatures[0].current_hp = 0;
        apply_power_to_creature_allow_dead(&mut state, uid, PowerKind::MonarchsGazeStrengthDown, 2);
        apply_signed_power_to_creature_allow_dead(&mut state, uid, PowerKind::Strength, -2);
        assert_eq!(state.powers.len(), 1, "only DieForYou itself");
        assert_eq!(state.powers[0].kind, PowerKind::DieForYou);
    }

    /// `allow_dead` still respects the combat-over guard (C# `IsOverOrEnding`).
    #[test]
    fn apply_power_allow_dead_still_blocked_when_combat_over() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        state.creatures[0].current_hp = 0;
        state.is_over = true;
        apply_power_to_creature_allow_dead(&mut state, uid, PowerKind::MonarchsGazeStrengthDown, 2);
        assert!(
            state.powers.is_empty(),
            "combat-over blocks even allow_dead"
        );
    }

    /// Bead `sts2-rs-jyd15`: the `amount <= 0` entry clamp must reject a
    /// NEGATIVE amount the same way it rejects zero — neither create a
    /// power on an empty creature nor subtract from an existing stack.
    /// `apply_power_to_*` is a one-directional "gain" surface; removal /
    /// decrement flows through the duration-tick (`Effect::TickPowerOrSkip`)
    /// and `Effect::DecrementPowerAmount` paths, not through a negative gain.
    #[test]
    fn apply_power_negative_amount_is_noop_on_empty_and_existing() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        // Negative on an empty creature: no power created.
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, -5);
        assert!(
            state.powers.is_empty(),
            "negative gain must not create a power"
        );
        assert!(state.creatures[0].powers.is_empty());

        // Negative on an existing stack: amount unchanged (not subtracted).
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 4);
        assert_eq!(state.powers[0].amount, 4);
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, -3);
        assert_eq!(
            state.powers[0].amount, 4,
            "negative gain is a noop — it does NOT decrement the existing stack"
        );
        assert_eq!(state.powers.len(), 1, "no duplicate instance created");
    }

    /// Bead `sts2-rs-zdjjx`: a debuff applied to an Artifact-bearing creature
    /// is negated entirely and consumes exactly one Artifact stack (C#
    /// `ArtifactPower`). Once the stacks run out, debuffs land normally.
    #[test]
    fn artifact_negates_debuff_and_consumes_one_stack() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        // Artifact is a buff, applied normally.
        apply_power_to_enemy(&mut state, uid, PowerKind::Artifact, 2);
        assert_eq!(state.powers.len(), 1);
        assert_eq!(state.powers[0].kind, PowerKind::Artifact);
        assert_eq!(state.powers[0].amount, 2);

        // First debuff negated: Artifact 2 -> 1, no Vulnerable instance.
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 3);
        assert_eq!(
            state.powers.len(),
            1,
            "Vulnerable negated — only Artifact remains"
        );
        assert_eq!(state.powers[0].kind, PowerKind::Artifact);
        assert_eq!(state.powers[0].amount, 1, "one Artifact stack consumed");

        // Second debuff consumes the last stack: Artifact removed.
        apply_power_to_enemy(&mut state, uid, PowerKind::Weak, 2);
        assert!(
            state.powers.is_empty(),
            "Artifact fully consumed and Weak negated"
        );
        assert!(state.creatures[0].powers.is_empty());

        // With no Artifact left the next debuff lands normally.
        apply_power_to_enemy(&mut state, uid, PowerKind::Vulnerable, 2);
        assert_eq!(state.powers.len(), 1);
        assert_eq!(state.powers[0].kind, PowerKind::Vulnerable);
        assert_eq!(state.powers[0].amount, 2);
    }

    /// Bead `sts2-rs-deaths-door-doom`: a Doom negated by the target's
    /// Artifact records no C# `PowerReceivedEntry`, so it must NOT arm
    /// `DeathsDoor`'s 3x block. A Doom landing on an Artifact-free enemy
    /// still arms it. (parity fuzz seed 492: CUBEX_CONSTRUCT Artifact 1.)
    #[test]
    fn artifact_negated_doom_does_not_arm_deaths_door() {
        let (mut state, uid) = fresh_state_with_one_enemy();

        // Doom onto an Artifact-bearing enemy: negated, flag stays false.
        apply_power_to_enemy(&mut state, uid, PowerKind::Artifact, 1);
        apply_power_to_enemy(&mut state, uid, PowerKind::Doom, 10);
        assert!(
            !state.combat_log.doom_applied_by_player_this_turn,
            "Artifact-negated Doom records no entry → DeathsDoor not armed"
        );
        assert!(
            !state.powers.iter().any(|p| p.kind == PowerKind::Doom),
            "Doom was negated (no instance), Artifact consumed"
        );

        // Now the Artifact is spent; a real Doom lands and arms the flag.
        apply_power_to_enemy(&mut state, uid, PowerKind::Doom, 10);
        assert!(
            state.combat_log.doom_applied_by_player_this_turn,
            "a landed Doom arms DeathsDoor's 3x block"
        );
    }

    /// Bead `sts2-rs-zdjjx`: Artifact only guards against debuffs — a buff
    /// (Strength) applied to an Artifact-bearing creature must NOT consume a
    /// stack and must land normally.
    #[test]
    fn artifact_does_not_block_buffs() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Artifact, 1);
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, 2);
        let artifact = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Artifact)
            .expect("Artifact intact");
        assert_eq!(artifact.amount, 1, "a buff must not consume Artifact");
        let strength = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("Strength applied");
        assert_eq!(strength.amount, 2);
    }

    /// Bead `sts2-rs-jyd15`: stacking uses `saturating_add`, so a power at
    /// the `i32` ceiling that receives another stack clamps at `i32::MAX`
    /// instead of overflow-panicking (debug) or wrapping negative (release).
    /// Guards the long-stall / pathological-buff case for permanent
    /// accumulators like Strength.
    /// q6bix F-N: `FastenPower.ModifyBlockAdditive` adds `+amount` block
    /// when the owner gains block from a **Defend-tagged** card
    /// (`FastenPower.cs`). A Defend (base 5) under Fasten(5) lands 10 block.
    #[test]
    fn q6bix_fasten_adds_block_on_defend_tagged_card() {
        use crate::cards::CardTag;
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        apply_power_to_creature(&mut state, player_uid, PowerKind::Fasten, 5);
        // Originating Defend-tagged card sits on top of the play pile.
        let defend = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::DefendIronclad),
            PileKind::Hand,
        );
        state.player.hand.retain(|u| *u != defend);
        state.player.play.push(defend);
        let idx = defend.0.get() as usize - 1;
        state.cards[idx].tags = state.cards[idx].tags.with(CardTag::Defend);
        gain_block_player(&mut state, 5);
        assert_eq!(
            state.player.block, 10,
            "Fasten(5) folds +5 onto a Defend card's 5 block"
        );
    }

    /// q6bix F-N counter-case: Fasten only fires for Defend-tagged sources.
    /// Block gained while a non-Defend card is the originating play is
    /// unmodified.
    #[test]
    fn q6bix_fasten_skips_non_defend_card() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        apply_power_to_creature(&mut state, player_uid, PowerKind::Fasten, 5);
        // A Strike (no Defend tag) is the originating play.
        let strike = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::StrikeIronclad),
            PileKind::Hand,
        );
        state.player.hand.retain(|u| *u != strike);
        state.player.play.push(strike);
        gain_block_player(&mut state, 5);
        assert_eq!(
            state.player.block, 5,
            "Fasten must not fold block for a non-Defend originating card"
        );
    }

    #[test]
    fn apply_power_stacking_saturates_at_i32_max() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, i32::MAX);
        assert_eq!(state.powers[0].amount, i32::MAX);
        // Another large stack must saturate, not overflow.
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, 100);
        assert_eq!(
            state.powers[0].amount,
            i32::MAX,
            "saturating_add clamps at i32::MAX rather than overflowing"
        );
        assert_eq!(state.powers.len(), 1);
    }

    /// Bead `sts2-rs-vt7n`: the creature-keyed primitive installs powers
    /// on the player-slot just as it does for enemy slots. This unblocks
    /// the `SwordOfJade` / `EmberTea` / `FakeSneckoEye` listener bodies
    /// in bead `sts2-rs-46ms` which need to push `Strength` / `Confused`
    /// onto the player's `Creature` mirror.
    #[test]
    fn apply_power_to_creature_targets_player_slot() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 50;
        state.player.max_hp = 50;
        let player_uid = state.init_player_creature();

        // First apply: creates the Strength instance on the player's
        // Creature mirror.
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 2);
        assert_eq!(state.powers.len(), 1, "Strength instance pushed");
        assert_eq!(state.powers[0].kind, PowerKind::Strength);
        assert_eq!(state.powers[0].amount, 2);

        // Read back through the standard accessor path: locate the
        // player's Creature by uid and inspect its `powers` SmallVec.
        let player_creature = state
            .creatures
            .iter()
            .find(|c| c.id == player_uid.0)
            .expect("player creature present after init_player_creature");
        assert_eq!(player_creature.powers.len(), 1);
        let attached_power_id = player_creature.powers[0].0;
        let power = state
            .powers
            .iter()
            .find(|p| p.id == attached_power_id)
            .expect("PowerUid resolves to a Power record");
        assert_eq!(power.kind, PowerKind::Strength);
        assert_eq!(power.amount, 2);

        // Second apply: stacks additively rather than duplicating.
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 3);
        assert_eq!(state.powers.len(), 1, "stacking, not duplicating");
        assert_eq!(state.powers[0].amount, 5);
    }

    /// q6bix F-D: the POSITIVE-stack path must drop a power whose amount
    /// nets to exactly 0, mirroring C# `PowerCmd.ModifyAmount` →
    /// `ShouldRemoveDueToAmount()` (removes at == 0 even for `AllowNegative`
    /// kinds like Strength). The temporary-Strength-debuff teardowns
    /// (DarkShackles / ShacklingPotion / FlexPotion) apply `-N` Strength on
    /// play, then revert with `+N` Strength at owner-side turn end. That
    /// `+N` re-application reaches this positive path and lands the stack at
    /// 0 — without the remove-at-0 guard a lingering `Strength(0)` instance
    /// stays (parity sweep 2026-05-30: PLAYER_STATE|MONSTER_STATE C#=0 Rust=1).
    /// The NEGATIVE path already removes at 0 (apply_effect ApplyPower arm);
    /// this brings the positive path to parity.
    #[test]
    fn f_d_positive_stack_removes_power_at_zero() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        // Seed a Strength(+3), then hand-set it to -10 to model the
        // mid-combat DarkShackles debuff state on a monster.
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, 3);
        state.powers[0].amount = -10;
        // Revert: +10 Strength lands the stack at exactly 0.
        apply_power_to_creature(&mut state, uid, PowerKind::Strength, 10);
        assert!(
            state.powers.is_empty(),
            "Strength(0) must be removed, not linger"
        );
        assert!(
            state.creatures[0].powers.is_empty(),
            "creature's power-uid list must drop the removed instance"
        );
    }

    /// q6bix F-D (applier variant): same remove-at-0 invariant for the
    /// applier-carrying positive-stack entry point.
    #[test]
    fn f_d_positive_stack_with_applier_removes_power_at_zero() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, 3);
        state.powers[0].amount = -7;
        apply_power_to_creature_with_applier(&mut state, uid, PowerKind::Strength, 7, None);
        assert!(
            state.powers.is_empty(),
            "Strength(0) must be removed via the applier path too"
        );
        assert!(state.creatures[0].powers.is_empty());
    }

    /// q6bix F-D guard: a positive stack that nets to a NON-zero result
    /// (e.g. -10 + 3 = -7) must NOT be removed — Strength can legitimately
    /// stay negative.
    #[test]
    fn f_d_positive_stack_keeps_nonzero_negative_result() {
        let (mut state, uid) = fresh_state_with_one_enemy();
        apply_power_to_enemy(&mut state, uid, PowerKind::Strength, 3);
        state.powers[0].amount = -10;
        apply_power_to_creature(&mut state, uid, PowerKind::Strength, 3);
        assert_eq!(state.powers.len(), 1, "non-zero result persists");
        assert_eq!(state.powers[0].amount, -7);
    }

    // -----------------------------------------------------------------
    // sts2-rs-5om: pile-search + card-spawn primitives
    // -----------------------------------------------------------------

    fn placeholder_card(card_id: crate::cards::CardId) -> crate::cards::Card {
        crate::cards::Card {
            uid: CardUid::new(u32::MAX).expect("placeholder, reassigned by spawn"),
            card_id,
            cost: crate::CardEnergyCost::new(0),
            card_type: crate::cards::CardType::Status,
            target_type: crate::cards::TargetType::None,
            upgraded: false,
            base_damage: 0,
            base_block: 0,
            combat_vars: crate::cards::CombatVars::new(),
            enchantments: crate::cards::CardEnchantments::new(),
            tags: crate::cards::CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        }
    }

    /// C# `SmokestackPower.AfterCardGeneratedForCombat`: generating a Status
    /// card while the player holds Smokestack(N) and is mid-card-play deals N
    /// Unpowered damage to every hittable enemy. (Turbo→Void, Overclock→Burn.)
    #[test]
    fn smokestack_damages_enemies_when_player_generates_status() {
        let (mut state, enemy) = fresh_state_with_one_enemy();
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::Smokestack,
            amount: 5,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Player is mid-card-play (a card sits in the Play pile).
        state.player.play.push(CardUid::new(999).expect("nz"));

        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Void),
            PileKind::Discard,
        );

        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .map(|c| c.current_hp)
            .expect("enemy");
        assert_eq!(hp, 25, "Smokestack(5) deals 5 to the enemy on status gen");
    }

    /// Bead `sts2-rs-arsenal-cardgen`: C# `ArsenalPower.AfterCardGeneratedForCombat`
    /// grants the player `Amount` Strength whenever the player generates ANY
    /// card mid-play (no card-type filter). The grant must fire from every
    /// generation chokepoint — `spawn_card_into_hand` (ManifestAuthority),
    /// `spawn_card_into_pile`, and `transform_card_in_place` (Charge). Regent
    /// parity fuzz seeds 104/400/427/438/495.
    #[test]
    fn arsenal_grants_strength_on_player_card_generation() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::Arsenal,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Player is mid-card-play (a card sits in the Play pile).
        state.player.play.push(CardUid::new(999).expect("nz"));

        // Generate a card into hand (ManifestAuthority path).
        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Void));
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            1,
            "Arsenal(1) grants 1 Strength on the first generated card"
        );

        // A second generated card (into a pile) grants another Strength.
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Void),
            PileKind::Discard,
        );
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            2,
            "each player-generated card grants Arsenal Strength"
        );
    }

    /// Arsenal does NOT fire when the play pile is empty (enemy-turn or
    /// non-player generation has `addedByPlayer == false`).
    #[test]
    fn arsenal_inert_when_not_mid_card_play() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::Arsenal,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Play pile empty → not a player-generated card.
        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Void));
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            0,
            "no Arsenal Strength outside a player card play"
        );
    }

    /// Smokestack does NOT fire when the play pile is empty (enemy-turn
    /// monster-added statuses like Wound/Slimed have `addedByPlayer == false`).
    #[test]
    fn smokestack_inert_when_not_mid_card_play() {
        let (mut state, enemy) = fresh_state_with_one_enemy();
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::Smokestack,
            amount: 5,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Play pile empty → not a player-generated status.
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Wound),
            PileKind::Discard,
        );
        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .map(|c| c.current_hp)
            .expect("enemy");
        assert_eq!(hp, 30, "no Smokestack damage outside a player card play");
    }

    /// Give the player `kind`(`amount`) — the status-generation reaction tests
    /// all need a player-side power record and the surrounding tests hand-roll
    /// the same six lines.
    fn grant_player_power(state: &mut CombatState, kind: PowerKind, amount: i32) {
        let raw = u32::try_from(state.powers.len() + 1).expect("power arena fits in u32");
        let pid = NonZeroU32::new(raw).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
    }

    /// C# `TrashToTreasurePower.AfterCardGeneratedForCombat`
    /// (`TrashToTreasurePower.cs:16-27`): generating a Status card while the
    /// player holds TrashToTreasure(N) and is mid-card-play channels N random
    /// orbs. GH #639.
    #[test]
    fn trash_to_treasure_channels_one_orb_per_stack_on_status_generation() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.orbs = crate::orbs::OrbSlots::with_capacity(3);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 2);
        // Player is mid-card-play (a card sits in the Play pile).
        state.player.play.push(CardUid::new(999).expect("nz"));

        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Void),
            PileKind::Discard,
        );

        assert_eq!(
            state.player.orbs.filled_len(),
            2,
            "TrashToTreasure(2) channels 2 orbs on a player-generated Status"
        );
    }

    /// C# `IterateHookListeners` runs the player's powers in the order they
    /// sit on the player, so the `AfterCardGeneratedForCombat` listeners fire
    /// in application order. Loadout Defect v0.107.1 1378: TrashToTreasure
    /// before Smokestack evokes a Dark orb into the low-HP enemy before
    /// Smokestack's damage kills it.
    #[test]
    fn generated_card_power_listeners_fire_in_application_order() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 1);
        grant_player_power(&mut state, PowerKind::Smokestack, 5);
        assert_eq!(
            generation_power_listener_order(&state).as_slice(),
            [PowerKind::TrashToTreasure, PowerKind::Smokestack]
        );

        let (mut state, _enemy) = fresh_state_with_one_enemy();
        grant_player_power(&mut state, PowerKind::Smokestack, 5);
        grant_player_power(&mut state, PowerKind::Arsenal, 1);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 1);
        assert_eq!(
            generation_power_listener_order(&state).as_slice(),
            [
                PowerKind::Smokestack,
                PowerKind::Arsenal,
                PowerKind::TrashToTreasure
            ]
        );
    }

    /// One read off the `combat_orbs` stream per stack — C# rolls
    /// `OrbModel.GetRandomOrb(Rng.CombatOrbGeneration)` once per loop
    /// iteration, so a wrong draw count desyncs every later orb roll.
    #[test]
    fn trash_to_treasure_consumes_one_combat_orbs_read_per_stack() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.orbs = crate::orbs::OrbSlots::with_capacity(3);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 3);
        state.player.play.push(CardUid::new(999).expect("nz"));

        let before = state.run_rng.combat_orbs.counter();
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Void),
            PileKind::Discard,
        );
        assert_eq!(
            state.run_rng.combat_orbs.counter() - before,
            3,
            "one combat_orbs roll per TrashToTreasure stack"
        );
    }

    /// The C# guard is `card.Type == Status`: a generated Attack/Skill/Power
    /// channels nothing.
    #[test]
    fn trash_to_treasure_inert_for_non_status_card() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.orbs = crate::orbs::OrbSlots::with_capacity(3);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 2);
        state.player.play.push(CardUid::new(999).expect("nz"));

        let _ = spawn_card_into_pile(&mut state, rocket_punch_card(), PileKind::Discard);
        assert_eq!(
            state.player.orbs.filled_len(),
            0,
            "no orbs for a generated non-Status card"
        );
        assert_eq!(
            state.run_rng.combat_orbs.counter(),
            0,
            "and no combat_orbs read"
        );
    }

    /// The C# guard is `creator != null && creator.Creature == Owner`: a
    /// monster-inflicted Wound (empty Play pile) channels nothing.
    #[test]
    fn trash_to_treasure_inert_when_not_mid_card_play() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.orbs = crate::orbs::OrbSlots::with_capacity(3);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 2);
        // Play pile empty → not a player-generated status.
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Wound),
            PileKind::Discard,
        );
        assert_eq!(
            state.player.orbs.filled_len(),
            0,
            "no orbs outside a player card play"
        );
    }

    /// The channel goes through [`crate::orbs::channel_orb`], so a belt that is
    /// already full evokes its front orb instead of growing past capacity.
    #[test]
    fn trash_to_treasure_overflows_a_full_belt_instead_of_growing_it() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.orbs = crate::orbs::OrbSlots::with_capacity(1);
        grant_player_power(&mut state, PowerKind::TrashToTreasure, 3);
        state.player.play.push(CardUid::new(999).expect("nz"));

        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Void),
            PileKind::Discard,
        );
        assert_eq!(
            state.player.orbs.filled_len(),
            1,
            "a 1-slot belt holds one orb; the other two overflow-evoke"
        );
    }

    /// Every orb in C# `OrbModel._validOrbs` is reachable — a 4-way roll would
    /// never produce Glass. Mirrors the Chaos pin `sts2-rs-dh30e`.
    #[test]
    fn trash_to_treasure_can_channel_every_orb_kind() {
        use crate::orbs::OrbKind;
        let mut seen = std::collections::HashSet::new();
        for seed in 0..200_i32 {
            let (mut state, _enemy) = fresh_state_with_one_enemy();
            state.player.orbs = crate::orbs::OrbSlots::with_capacity(3);
            state.run_rng.combat_orbs = crate::dotnet_rng::Rng::with_name(seed, "combat_orbs");
            grant_player_power(&mut state, PowerKind::TrashToTreasure, 1);
            state.player.play.push(CardUid::new(999).expect("nz"));
            let _ = spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Void),
                PileKind::Discard,
            );
            for orb in &state.player.orbs.slots {
                if !orb.is_empty() {
                    seen.insert(OrbKind::of(orb));
                }
            }
        }
        assert_eq!(
            seen.len(),
            5,
            "all five valid orbs must be reachable; saw {seen:?}"
        );
    }

    /// GH #639 consolidation: C# `AddGeneratedCardToCombat` fires
    /// `AfterCardGeneratedForCombat` on EVERY listener, so a Status generated
    /// straight into hand must reach Smokestack too — the hand path previously
    /// fired only the Arsenal/PillarOfCreation pair.
    #[test]
    fn smokestack_fires_for_a_status_generated_into_hand() {
        let (mut state, enemy) = fresh_state_with_one_enemy();
        grant_player_power(&mut state, PowerKind::Smokestack, 5);
        state.player.play.push(CardUid::new(999).expect("nz"));

        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Void));

        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .map(|c| c.current_hp)
            .expect("enemy");
        assert_eq!(hp, 25, "Smokestack(5) fires on a hand-generated status");
    }

    /// A 2-cost RocketPunch (Attack) placeholder for the status-generation
    /// reaction tests — overrides the default 0-cost/Status placeholder.
    fn rocket_punch_card() -> crate::cards::Card {
        crate::cards::Card {
            cost: crate::CardEnergyCost::new(2),
            card_type: crate::cards::CardType::Attack,
            target_type: crate::cards::TargetType::Enemy,
            ..placeholder_card(crate::cards::CardId::RocketPunch)
        }
    }

    /// Bead `sts2-rs-rocketpunch-statusgen`: C#
    /// `RocketPunch.AfterCardGeneratedForCombat` — when the PLAYER generates a
    /// Status card, every RocketPunch the player holds becomes free until
    /// played (`EnergyCost.SetUntilPlayed(0)`). Drives the
    /// `spawn_card_into_pile` mid-play path (Smokestack-style).
    #[test]
    fn rocketpunch_freed_when_player_generates_status() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        let rp = spawn_card_into_pile(&mut state, rocket_punch_card(), PileKind::Hand);
        assert_eq!(
            crate::hooks::effective_card_cost(&state, rp),
            2,
            "RocketPunch starts at base cost 2"
        );
        // Player is mid-card-play → generated card is `addedByPlayer`.
        state.player.play.push(CardUid::new(999).expect("nz"));
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Dazed),
            PileKind::Discard,
        );
        assert_eq!(
            crate::hooks::effective_card_cost(&state, rp),
            0,
            "player-generated Status makes RocketPunch free until played"
        );
    }

    /// The TeaOfDiscourtesy path: a Status spawned via the random-position
    /// draw-pile primitive fires the reaction with `addedByPlayer: true`
    /// EVEN at combat start (empty Play pile) — the exact Defect seed-111 case.
    #[test]
    fn rocketpunch_freed_by_combat_start_random_position_status() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        // Live combat: the random-position spawn only runs (and fires its
        // reactions) while combat is in progress; a `Default` player has 0 HP,
        // which `combat_is_ending` treats as a pending loss.
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let rp = spawn_card_into_pile(&mut state, rocket_punch_card(), PileKind::Hand);
        // Empty Play pile (combat start). Tea seeds Dazed into the draw pile.
        assert!(state.player.play.is_empty());
        let _ = spawn_card_into_draw_pile_random_position(
            &mut state,
            placeholder_card(crate::cards::CardId::Dazed),
        );
        assert_eq!(
            crate::hooks::effective_card_cost(&state, rp),
            0,
            "TeaOfDiscourtesy's combat-start Dazed frees RocketPunch (seed 111)"
        );
    }

    /// The reaction is gated on BOTH the card being a Status and the
    /// generation being `addedByPlayer`: a non-Status player generation and a
    /// non-player (empty Play pile) Status generation both leave RocketPunch at
    /// its base cost.
    #[test]
    fn rocketpunch_not_freed_by_nonstatus_or_monster_generation() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        let rp = spawn_card_into_pile(&mut state, rocket_punch_card(), PileKind::Hand);

        // (a) Player generates a NON-Status card → no trigger.
        state.player.play.push(CardUid::new(999).expect("nz"));
        let mut attack = placeholder_card(crate::cards::CardId::BeamCell);
        attack.card_type = crate::cards::CardType::Attack;
        let _ = spawn_card_into_pile(&mut state, attack, PileKind::Discard);
        assert_eq!(
            crate::hooks::effective_card_cost(&state, rp),
            2,
            "a generated non-Status does not free RocketPunch"
        );

        // (b) Monster-added Status (empty Play pile → addedByPlayer false) → no trigger.
        state.player.play.clear();
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Wound),
            PileKind::Discard,
        );
        assert_eq!(
            crate::hooks::effective_card_cost(&state, rp),
            2,
            "a monster-generated Status does not free RocketPunch"
        );
    }

    #[test]
    fn spawn_card_into_each_pile_assigns_uid_and_enqueues() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let u_draw = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Draw,
        );
        let u_discard = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
        );
        let u_exhaust = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Exhaust,
        );
        let u_hand = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Hotfix),
            PileKind::Hand,
        );
        // Sequential uids 1..=4
        assert_eq!(u_draw.0.get(), 1);
        assert_eq!(u_discard.0.get(), 2);
        assert_eq!(u_exhaust.0.get(), 3);
        assert_eq!(u_hand.0.get(), 4);
        // Pile membership.
        assert_eq!(state.player.draw.as_slice(), &[u_draw]);
        assert_eq!(state.player.discard.as_slice(), &[u_discard]);
        assert_eq!(state.player.exhaust.as_slice(), &[u_exhaust]);
        assert_eq!(state.player.hand.as_slice(), &[u_hand]);
        // Spawning a second Draw card lands at index 0 (top), pushing the
        // first to index 1 — mirrors Python `insert(0, …)`.
        let u_draw2 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Draw,
        );
        assert_eq!(state.player.draw.as_slice(), &[u_draw2, u_draw]);
    }

    #[test]
    fn search_pile_filters_by_predicate_and_preserves_order() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let u1 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
        );
        let u2 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let _u3 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
        );
        let burns = search_pile(&state, PileKind::Discard, |c| {
            c.card_id == crate::cards::CardId::Burn
        });
        assert_eq!(burns.as_slice(), &[u2]);
        let first = pick_first_in_pile(&state, PileKind::Discard, |c| {
            c.card_id == crate::cards::CardId::Slimed
        });
        assert_eq!(first, Some(u1));
        // Empty pile / no match.
        let none = pick_first_in_pile(&state, PileKind::Exhaust, |_| true);
        assert!(none.is_none());
    }

    /// Bead `sts2-rs-7up3`: under the hand cap, `spawn_card_into_hand`
    /// appends to `state.player.hand` and leaves `discard` untouched.
    #[test]
    fn spawn_card_into_hand_appends_when_hand_under_cap() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Pre-seed two cards in hand to verify append-order, not overwrite.
        let u0 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Hand,
        );
        let u1 = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Hand,
        );
        let u_new =
            spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Hotfix));
        assert_eq!(state.player.hand.as_slice(), &[u0, u1, u_new]);
        assert!(state.player.discard.is_empty());
        // Arena uid is the next 1-based slot.
        assert_eq!(u_new.0.get(), 3);
    }

    /// Bead `sts2-rs-7up3`: with hand already at `HAND_LIMIT`, the
    /// freshly-generated card is routed to discard instead — mirrors
    /// Python `move_card_to_creature_hand` overflow.
    #[test]
    fn spawn_card_into_hand_routes_to_discard_when_at_cap() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Fill hand to HAND_LIMIT (10).
        let mut hand_uids: Vec<CardUid> = Vec::with_capacity(crate::cards::HAND_LIMIT);
        for _ in 0..crate::cards::HAND_LIMIT {
            hand_uids.push(spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Hand,
            ));
        }
        assert_eq!(state.player.hand.len(), crate::cards::HAND_LIMIT);
        let u_overflow =
            spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Burn));
        // Hand unchanged; overflow lives in discard.
        assert_eq!(state.player.hand.as_slice(), hand_uids.as_slice());
        assert_eq!(state.player.discard.as_slice(), &[u_overflow]);
    }

    /// Bead `sts2-rs-8h2q`: every `spawn_card_into_hand` call increments
    /// `player.generated_cards_this_combat` — unconditionally, before
    /// the hand-cap check, so a card routed to discard via overflow
    /// still counts. Mirrors Python's `_add_generated_cards_to_hand`
    /// at `sts2-rl-agent@01c094d:sts2_env/core/combat.py:2260`.
    #[test]
    fn spawn_card_into_hand_increments_generated_counter() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        assert_eq!(state.count_player_generated_cards_this_combat(), 0);
        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Hotfix));
        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Burn));
        assert_eq!(state.count_player_generated_cards_this_combat(), 2);
        // Fill hand to cap so the next spawn overflows to discard, then
        // confirm it still bumps the counter.
        while state.player.hand.len() < crate::cards::HAND_LIMIT {
            let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Shiv));
        }
        let before_overflow = state.count_player_generated_cards_this_combat();
        let _ = spawn_card_into_hand(&mut state, placeholder_card(crate::cards::CardId::Burn));
        assert_eq!(
            state.count_player_generated_cards_this_combat(),
            before_overflow + 1,
        );
    }

    /// Supermassive counts every `CardGeneratedEntry` whose creator is the
    /// player (`Supermassive.cs:22-27`, `CardPileCmd.cs:243`): the combat-start
    /// random-position spawn (TeaOfDiscourtesy / FuneraryMask) and a
    /// player-created pile spawn count; a monster-inflicted status does not.
    /// Regent full-tier SUPERMASSIVE residuals (seeds 1401/2143/3222/...).
    #[test]
    fn player_created_pile_spawns_count_toward_generated_tally() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        assert_eq!(state.count_player_generated_cards_this_combat(), 0);
        let _ = spawn_card_into_draw_pile_random_position(
            &mut state,
            placeholder_card(crate::cards::CardId::Dazed),
        );
        let _ = spawn_card_into_draw_pile_bottom(
            &mut state,
            placeholder_card(crate::cards::CardId::Soul),
        );
        assert_eq!(state.count_player_generated_cards_this_combat(), 2);
        // Monster-inflicted (creator null): no entry for Supermassive.
        let _ = spawn_card_into_pile_with_origin(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
            false,
        );
        assert_eq!(state.count_player_generated_cards_this_combat(), 2);
        let _ = spawn_card_into_pile_with_origin(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Discard,
            true,
        );
        assert_eq!(state.count_player_generated_cards_this_combat(), 3);
    }

    /// Bead `sts2-rs-v7mp`: `deal_damage_to_all_alive_enemies` hits
    /// every alive enemy through `damage::apply_damage` and skips dead
    /// rows. Locks the per-enemy iteration shape relied on by
    /// `LostWisp` / `MrStruggles` listeners.
    #[test]
    fn deal_damage_to_all_alive_enemies_hits_each_alive_and_skips_dead() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Player Creature is required by the damage pipeline (dealer
        // ref resolves through state.player.creature_uid).
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        // Two alive enemies + one dead enemy.
        for (slot, hp) in [(1_u32, 30_i32), (2_u32, 20_i32), (3_u32, 0_i32)] {
            let id = NonZeroU32::new(slot + 10).expect("non-zero");
            state.creatures.push(Creature {
                id,
                current_hp: hp,
                max_hp: 30,
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                intent: None,
                intent_history: IntentHistory::default(),
                monster_id: MonsterId("Nibbit"),
                ai: None,
            });
            state.enemy_uids.push(CreatureUid(id));
        }
        let total = deal_damage_to_all_alive_enemies(&mut state, 5, ValueProps::UNPOWERED, None);
        // 5 dmg landed on each of the two alive enemies (HP 30, 20 ->
        // both >0 after; total HP-loss = 10).
        assert_eq!(total, 10);
        assert_eq!(state.creatures[1].current_hp, 25);
        assert_eq!(state.creatures[2].current_hp, 15);
        // Dead enemy unchanged.
        assert_eq!(state.creatures[3].current_hp, 0);
    }

    /// C# `CreatureCmd.Damage` over several targets applies every target's HP
    /// loss before any after-hit hook (CreatureCmd.cs:394-430; v0.107.1
    /// :359-408). So the Outbreak that the first target's Envenom Poison sets
    /// off lands after the second target already took the AoE: the second
    /// target dies to Outbreak, not to the AoE, which kills nothing (Echoing
    /// Slash does not repeat). Silent loadout v0.107.1 seeds 1000, 1037.
    #[test]
    fn aoe_after_hit_hooks_wait_for_every_targets_hp_loss() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        for (slot, hp) in [(1_u32, 40_i32), (2_u32, 15_i32)] {
            let id = NonZeroU32::new(slot + 10).expect("non-zero");
            state.creatures.push(Creature {
                id,
                current_hp: hp,
                max_hp: 40,
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                intent: None,
                intent_history: IntentHistory::default(),
                monster_id: MonsterId("Nibbit"),
                ai: None,
            });
            state.enemy_uids.push(CreatureUid(id));
        }
        // Envenom(1), and Outbreak(11) one Poison short of firing.
        for (pid, kind, amount, counter) in [
            (1_u32, PowerKind::Envenom, 1, 0),
            (2_u32, PowerKind::Outbreak, 11, 2),
        ] {
            let pid = NonZeroU32::new(pid).expect("nz");
            state.powers.push(crate::Power {
                id: pid,
                kind,
                amount,
                counter,
                applier: None,
                skip_next_tick: false,
            });
            state.player.powers.push(crate::PowerUid(pid));
        }
        let (total, kills) =
            deal_damage_to_all_alive_enemies_counting_kills(&mut state, 5, ValueProps::MOVE, None);
        assert_eq!(total, 10, "both targets take the full AoE before Outbreak");
        assert_eq!(kills, 0, "the second target dies to Outbreak, not the AoE");
        assert_eq!(state.creatures[1].current_hp, 24, "40 - 5 - 11");
        assert_eq!(state.creatures[2].current_hp, 0, "15 - 5 - 11");
    }

    /// Osty-as-dealer attack path (`FromOsty`): a summoned Osty deals
    /// card damage to an enemy; `kill_osty` zeroes the pet but retains the
    /// corpse (DieForYou) + `pet_uid` so a later summon revives it.
    #[test]
    fn osty_dealer_attacks_then_kill_retains_corpse_for_revive() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _player_uid = state.init_player_creature();
        let enemy_id = NonZeroU32::new(11).expect("non-zero");
        state.creatures.push(Creature {
            id: enemy_id,
            current_hp: 30,
            max_hp: 30,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Nibbit),
            intent: None,
            intent_history: IntentHistory::default(),
            monster_id: MonsterId("Nibbit"),
            ai: None,
        });
        state.enemy_uids.push(CreatureUid(enemy_id));

        // No Osty yet → dealer ref is None, attack is a no-op.
        assert!(osty_dealer_ref(&state).is_none());
        assert_eq!(
            deal_damage_from_osty_to_enemy(&mut state, CreatureUid(enemy_id), 6),
            0
        );
        assert_eq!(state.creatures[1].current_hp, 30);

        // Summon Osty (10 HP) → dealer ref resolves to the pet, NOT an
        // enemy in `enemy_uids`.
        let osty_uid = state.summon_osty(10).expect("osty summoned");
        assert_eq!(osty_dealer_ref(&state), Some(CreatureRef::Enemy(osty_uid)));
        assert!(!state.enemy_uids.contains(&osty_uid));

        // Osty pokes the enemy for 6 (powered MOVE, no modifiers).
        assert_eq!(
            deal_damage_from_osty_to_enemy(&mut state, CreatureUid(enemy_id), 6),
            6
        );
        assert_eq!(state.creatures[1].current_hp, 24);

        // Kill the Osty (Bone Shards' sacrifice): HP 0, corpse retained,
        // pet_uid intact, DieForYou survives the teardown.
        state.kill_osty();
        let osty = state
            .creatures
            .iter()
            .find(|c| c.id == osty_uid.0)
            .expect("corpse retained");
        assert_eq!(osty.current_hp, 0, "Osty killed");
        assert_eq!(state.player.pet_uid, Some(osty_uid), "pet_uid retained");
        assert!(
            osty.powers.iter().any(|u| state
                .powers
                .iter()
                .any(|p| p.id == u.0 && p.kind == PowerKind::DieForYou)),
            "DieForYou survives RemoveAllPowersAfterDeath"
        );
        // Dead Osty → dealer ref None, attack no-ops.
        assert!(osty_dealer_ref(&state).is_none());
        assert_eq!(
            deal_damage_from_osty_to_enemy(&mut state, CreatureUid(enemy_id), 6),
            0
        );

        // Re-summon revives the SAME pet in place (reuses pet_uid).
        let revived = state.summon_osty(8).expect("osty revived");
        assert_eq!(revived, osty_uid, "revive reuses the pet slot");
        let osty = state.creatures.iter().find(|c| c.id == osty_uid.0).unwrap();
        assert_eq!(osty.current_hp, 8, "revived to summon amount");
        assert_eq!(osty_dealer_ref(&state), Some(CreatureRef::Enemy(osty_uid)));
    }

    /// `CalcifyPower` (on the player) adds `Amount` to every Osty attack
    /// (C# `CalcifyPower.ModifyDamageAdditive` gated on `dealer is my Osty`).
    /// A player attack is unaffected; an Osty attack lands `base + amount`.
    #[test]
    fn calcify_boosts_osty_attack_not_player_attack() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        let enemy_id = NonZeroU32::new(11).expect("non-zero");
        state.creatures.push(Creature {
            id: enemy_id,
            current_hp: 60,
            max_hp: 60,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Nibbit),
            intent: None,
            intent_history: IntentHistory::default(),
            monster_id: MonsterId("Nibbit"),
            ai: None,
        });
        state.enemy_uids.push(CreatureUid(enemy_id));
        state.summon_osty(10).expect("osty");
        // Calcify(4) on the player.
        apply_power_to_creature(&mut state, player_uid, PowerKind::Calcify, 4);

        // Player attack (dealer = player) → Calcify does NOT fire: 6 dmg.
        assert_eq!(
            deal_damage_player_to_enemy(&mut state, CreatureUid(enemy_id), 6),
            6
        );
        assert_eq!(state.creatures[1].current_hp, 54);

        // Osty attack (dealer = pet) → Calcify adds +4: 6 + 4 = 10 dmg.
        assert_eq!(
            deal_damage_from_osty_to_enemy(&mut state, CreatureUid(enemy_id), 6),
            10
        );
        assert_eq!(state.creatures[1].current_hp, 44);
    }

    /// Bead `sts2-rs-c46l`: `pick_random_alive_enemy` samples uniformly
    /// over alive enemies, never returns a dead-enemy uid, and returns
    /// `None` when no enemy is alive.
    #[test]
    fn pick_random_alive_enemy_filters_dead_and_samples_alive() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(42_i32),
            ..Default::default()
        };
        // Slots: alive / dead / alive.
        let mut alive_uids: SmallVec<[CreatureUid; 4]> = SmallVec::new();
        for (slot, hp) in [(1_u32, 30_i32), (2_u32, 0_i32), (3_u32, 25_i32)] {
            let id = NonZeroU32::new(slot).expect("non-zero");
            state.creatures.push(Creature {
                id,
                current_hp: hp,
                max_hp: 30,
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
            if hp > 0 {
                alive_uids.push(uid);
            }
        }

        let mut seen_first = false;
        let mut seen_third = false;
        for _ in 0..100 {
            let picked = pick_random_alive_enemy(&mut state).expect("alive enemy available");
            assert!(
                alive_uids.contains(&picked),
                "picked uid must be in alive set"
            );
            if picked == alive_uids[0] {
                seen_first = true;
            }
            if picked == alive_uids[1] {
                seen_third = true;
            }
        }
        assert!(
            seen_first && seen_third,
            "uniform sampling over 100 draws should touch both alive uids"
        );

        // Kill every enemy → returns None, no panic.
        for c in &mut state.creatures {
            c.current_hp = 0;
        }
        assert!(pick_random_alive_enemy(&mut state).is_none());
    }

    // -----------------------------------------------------------------
    // sts2-rs-9dj5: scan_piles_and_move_to_hand
    // -----------------------------------------------------------------

    /// Three copies of `SovereignBlade` spread across draw / discard /
    /// exhaust all land in hand; source piles are empty of matches.
    #[test]
    fn scan_piles_and_move_to_hand_pulls_from_all_three_piles() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let player_uid = state.init_player_creature();
        let in_draw = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::SovereignBlade),
            PileKind::Draw,
        );
        let in_discard = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::SovereignBlade),
            PileKind::Discard,
        );
        let in_exhaust = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::SovereignBlade),
            PileKind::Exhaust,
        );
        // Non-matching decoys remain in place.
        let decoy_draw = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Draw,
        );
        let decoy_discard = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
        );

        let moved = scan_piles_and_move_to_hand(
            &mut state,
            player_uid,
            &[PileKind::Draw, PileKind::Discard, PileKind::Exhaust],
            |c| c.card_id == crate::cards::CardId::SovereignBlade,
        );
        assert_eq!(moved, 3);
        // All three matches now in hand in pile-iteration order
        // (draw -> discard -> exhaust).
        assert_eq!(
            state.player.hand.as_slice(),
            &[in_draw, in_discard, in_exhaust]
        );
        // Decoys untouched.
        assert_eq!(state.player.draw.as_slice(), &[decoy_draw]);
        assert_eq!(state.player.discard.as_slice(), &[decoy_discard]);
        assert!(state.player.exhaust.is_empty());
    }

    /// Empty match-set: zero moves, hand untouched.
    #[test]
    fn scan_piles_and_move_to_hand_no_match_returns_zero() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let player_uid = state.init_player_creature();
        let u_d = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Discard,
        );
        let hand_before: smallvec::SmallVec<[CardUid; 16]> = state.player.hand.clone();

        let moved = scan_piles_and_move_to_hand(
            &mut state,
            player_uid,
            &[PileKind::Draw, PileKind::Discard, PileKind::Exhaust],
            |c| c.card_id == crate::cards::CardId::SovereignBlade,
        );
        assert_eq!(moved, 0);
        assert_eq!(state.player.hand, hand_before);
        assert_eq!(state.player.discard.as_slice(), &[u_d]);
    }

    /// Predicate-matching card in the hand pile is excluded from the
    /// scan (only draw/discard/exhaust are passed) — it stays in hand
    /// at its original position and is not "moved".
    #[test]
    fn scan_piles_and_move_to_hand_skips_hand_pile_when_excluded() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let player_uid = state.init_player_creature();
        let in_hand = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::SovereignBlade),
            PileKind::Hand,
        );
        let moved = scan_piles_and_move_to_hand(
            &mut state,
            player_uid,
            &[PileKind::Draw, PileKind::Discard, PileKind::Exhaust],
            |c| c.card_id == crate::cards::CardId::SovereignBlade,
        );
        assert_eq!(moved, 0);
        assert_eq!(state.player.hand.as_slice(), &[in_hand]);
    }

    // -----------------------------------------------------------------
    // sts2-rs-6ynt.2: hand-mutation + cross-pile move primitives
    // -----------------------------------------------------------------

    #[test]
    fn find_card_pile_locates_in_each_pile_and_misses_unknown() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let h = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Hand,
        );
        let d = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Draw,
        );
        let dis = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Discard,
        );
        let ex = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Exhaust,
        );
        assert_eq!(find_card_pile(&state, h), Some(PileKind::Hand));
        assert_eq!(find_card_pile(&state, d), Some(PileKind::Draw));
        assert_eq!(find_card_pile(&state, dis), Some(PileKind::Discard));
        assert_eq!(find_card_pile(&state, ex), Some(PileKind::Exhaust));
        // Unknown uid (one past the arena) → None.
        let bogus = CardUid::new(99).expect("non-zero");
        assert!(find_card_pile(&state, bogus).is_none());
    }

    #[test]
    fn move_card_to_pile_relocates_and_reports_source() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Discard,
        );
        // Discard → top of Draw (default, random_position=false).
        let src = move_card_to_pile(&mut state, u, PileKind::Draw, false);
        assert_eq!(src, Some(PileKind::Discard));
        assert!(state.player.discard.is_empty());
        assert_eq!(state.player.draw.as_slice(), &[u]);
        // Draw → Hand.
        let src = move_card_to_pile(&mut state, u, PileKind::Hand, false);
        assert_eq!(src, Some(PileKind::Draw));
        assert!(state.player.draw.is_empty());
        assert_eq!(state.player.hand.as_slice(), &[u]);
        // Missing uid is a None return, no panic.
        let bogus = CardUid::new(99).expect("non-zero");
        assert!(move_card_to_pile(&mut state, bogus, PileKind::Discard, false).is_none());
    }

    #[test]
    fn move_card_to_hand_overflows_to_discard_at_cap() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Fill hand to HAND_LIMIT.
        for _ in 0..crate::cards::HAND_LIMIT {
            spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Hand,
            );
        }
        // Spawn one more in discard, then move to hand → overflow.
        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let src = move_card_to_pile(&mut state, u, PileKind::Hand, false);
        assert_eq!(src, Some(PileKind::Discard));
        assert_eq!(state.player.hand.len(), crate::cards::HAND_LIMIT);
        assert_eq!(state.player.discard.as_slice(), &[u]);
    }

    /// `spawn_card_into_draw_pile_random_position` must consume exactly
    /// one **Shuffle**-stream draw per card (C# `CardPileCmd.cs:433`
    /// `Rng.Shuffle.NextInt(Count + 1)`) and land each card somewhere in
    /// the draw pile. Bead `sts2-rs-n4sa4` (FuneraryMask / TeaOfDiscourtesy).
    #[test]
    fn spawn_card_into_draw_random_position_consumes_shuffle_stream() {
        // Live combat (alive enemy + alive player): the random-position spawn
        // only reads the shuffle stream while combat is in progress -- once it
        // is ending the C# `CardPileCmd.Add` short-circuit (mirrored in the
        // helper) skips the read. See
        // `random_position_spawn_skips_shuffle_read_when_combat_ending`.
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        // Pre-seed draw with 4 anchors so insert indices have room to vary.
        for _ in 0..4 {
            spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Draw,
            );
        }
        let before = state.run_rng.shuffle.counter();
        let mut spawned = Vec::new();
        for _ in 0..3 {
            spawned.push(spawn_card_into_draw_pile_random_position(
                &mut state,
                placeholder_card(crate::cards::CardId::Burn),
            ));
        }
        // One Shuffle read per spawned card — the parity-critical property.
        assert_eq!(
            state.run_rng.shuffle.counter() - before,
            3,
            "3 cards => 3 Shuffle-stream draws"
        );
        assert_eq!(state.player.draw.len(), 7, "4 anchors + 3 spawned");
        for u in &spawned {
            assert!(
                state.player.draw.contains(u),
                "spawned card present in draw"
            );
        }
    }

    #[test]
    fn move_card_to_draw_random_position_uses_rng() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Pre-seed draw with 5 anchor cards.
        let mut anchors = Vec::with_capacity(5);
        for _ in 0..5 {
            anchors.push(spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Draw,
            ));
        }
        // Spawn the moving card into discard, then re-insert into draw
        // at a random position.
        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let src = move_card_to_pile(&mut state, u, PileKind::Draw, true);
        assert_eq!(src, Some(PileKind::Discard));
        // The moved card must be present somewhere in the draw pile.
        assert_eq!(state.player.draw.len(), 6);
        assert!(state.player.draw.contains(&u));
        // Determinism check: same seed reproduces the same insert
        // index on a fresh state.
        let mut state2 = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        for _ in 0..5 {
            spawn_card_into_pile(
                &mut state2,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Draw,
            );
        }
        let u2 = spawn_card_into_pile(
            &mut state2,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let _ = move_card_to_pile(&mut state2, u2, PileKind::Draw, true);
        let pos1 = state.player.draw.iter().position(|&x| x == u).unwrap();
        let pos2 = state2.player.draw.iter().position(|&x| x == u2).unwrap();
        assert_eq!(pos1, pos2, "RNG must be deterministic across runs");
    }

    /// Bead `sts2-rs-09x5n` (F2 `HiddenInfoMode::BeliefDraw`): the
    /// Draw+random_position branch must respect the deferred-draw-order
    /// layout invariant (pins at their declared slots, unknown middle
    /// uid-sorted) instead of inserting at a raw physical index that
    /// could land INSIDE the pinned_top block or break the sorted
    /// middle — and it must consume NO RNG (a random position inside an
    /// order-unobserved region is unobservable).
    #[test]
    fn move_card_to_draw_random_position_respects_deferred_order_layout() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // 4 anchor cards in draw, then pin first-to-top / last-to-bottom.
        let mut anchors = Vec::with_capacity(4);
        for _ in 0..4 {
            anchors.push(spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Shiv),
                PileKind::Draw,
            ));
        }
        let top = anchors[0];
        let bottom = anchors[3];
        crate::cards::pin_top(&mut state.player, top);
        crate::cards::pin_bottom(&mut state.player, bottom);
        // Emulate a BeliefDraw search state: flag set + canonical layout
        // (what belief_root / a flag-gated shuffle would have produced).
        state.player.deferred_draw_order = true;
        let mut setup_rng = crate::dotnet_rng::Rng::new(1);
        crate::cards::shuffle(&mut state.player, &mut setup_rng);

        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let ccs_before = state.run_rng.combat_card_selection.counter();
        let shuf_before = state.run_rng.shuffle.counter();
        let src = move_card_to_pile(&mut state, u, PileKind::Draw, true);
        assert_eq!(src, Some(PileKind::Discard));
        assert_eq!(
            state.run_rng.combat_card_selection.counter(),
            ccs_before,
            "deferred insert must NOT read combat_card_selection"
        );
        assert_eq!(
            state.run_rng.shuffle.counter(),
            shuf_before,
            "deferred insert must NOT read the shuffle stream either"
        );

        // Layout invariant intact: pins at their slots, middle sorted.
        assert_eq!(state.player.draw.first().copied(), Some(top));
        assert_eq!(state.player.draw.last().copied(), Some(bottom));
        let middle = &state.player.draw[1..state.player.draw.len() - 1];
        assert!(middle.contains(&u), "moved card landed in the middle");
        assert!(
            middle.windows(2).all(|w| w[0] < w[1]),
            "unknown middle must stay uid-sorted after the insert: {middle:?}"
        );

        // pop_next_card drains the whole pile without panicking, pins in
        // the right order (top first, bottom last).
        let mut rng = crate::dotnet_rng::Rng::new(7);
        let n = state.player.draw.len();
        let mut drawn = Vec::with_capacity(n);
        for _ in 0..n {
            let uid = crate::cards::pop_next_card(&mut state.player, &mut rng);
            crate::cards::unpin(&mut state.player, uid);
            drawn.push(uid);
        }
        assert!(state.player.draw.is_empty());
        assert_eq!(drawn.first().copied(), Some(top));
        assert_eq!(drawn.last().copied(), Some(bottom));
    }

    #[test]
    fn exhaust_card_uid_relocates_from_hand_and_dual_fires() {
        use crate::relic_effects::{
            register_after_card_exhausted, AfterCardExhaustedFn, CardEventCtx,
        };
        use crate::{Power, PowerKind, PowerUid, RelicId};
        use std::sync::atomic::{AtomicU32, Ordering};

        // Relic-bus listener witness — accumulates across the two
        // dispatches in this test.
        static RELIC_HITS: AtomicU32 = AtomicU32::new(0);
        fn h_relic(_s: &mut CombatState, _ctx: &CardEventCtx) -> EffectResult {
            RELIC_HITS.fetch_add(1, Ordering::SeqCst);
            EffectResult::Resolved
        }
        // Idempotent re-register (overwrites same RelicId slot).
        register_after_card_exhausted(RelicId::BlackBlood, h_relic as AfterCardExhaustedFn);

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // A living player: a dead one makes the combat "ending", where
        // `CardCmd.Exhaust` is refused (`exhaust_may_begin`).
        state.player.current_hp = 80;
        state.init_player_creature();
        // Give player FeelNoPain(4) so the hooks-bus post-walk reaction
        // (block gain) is observable; this proves the hooks bus fired
        // without needing a synthetic hook-listener registration API.
        let power_id = NonZeroU32::new(1).expect("non-zero");
        state
            .powers
            .push(Power::new(power_id, PowerKind::FeelNoPain, 4));
        state.player.powers.push(PowerUid(power_id));

        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Hand,
        );
        let block_before = state.player.block;
        let relic_hits_before = RELIC_HITS.load(Ordering::SeqCst);

        let ok = exhaust_card_uid(&mut state, u);
        assert!(ok);
        assert!(state.player.hand.is_empty());
        assert_eq!(state.player.exhaust.as_slice(), &[u]);
        // Hooks bus: FeelNoPain gained 4 block.
        assert_eq!(state.player.block, block_before + 4);
        // Relic bus: listener fired once.
        assert_eq!(RELIC_HITS.load(Ordering::SeqCst), relic_hits_before + 1);

        // Already in exhaust → no-op (returns false, no double-fire).
        let dup = exhaust_card_uid(&mut state, u);
        assert!(!dup);
        assert_eq!(state.player.block, block_before + 4);
        assert_eq!(RELIC_HITS.load(Ordering::SeqCst), relic_hits_before + 1);
    }

    #[test]
    fn transform_card_in_place_reuses_pile_slot() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // A live player: a transform is refused once the combat is ending.
        state.player.current_hp = 1;
        // Hand: [Shiv, Burn, Slimed]. Transform Burn → Hotfix; the
        // pile slot at index 1 must point at the new card.
        let u_shiv = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Hand,
        );
        let u_burn = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Hand,
        );
        let u_slimed = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Hand,
        );
        let new_uid = transform_card_in_place(
            &mut state,
            u_burn,
            placeholder_card(crate::cards::CardId::Hotfix),
        )
        .expect("found in hand");
        assert_ne!(new_uid, u_burn);
        assert_eq!(state.player.hand.as_slice(), &[u_shiv, new_uid, u_slimed]);
        // The new arena row carries the new card_id.
        let idx = new_uid.0.get() as usize - 1;
        assert_eq!(state.cards[idx].card_id, crate::cards::CardId::Hotfix);
        // Old arena row is preserved (no compaction); old uid no longer
        // appears in any pile.
        assert!(find_card_pile(&state, u_burn).is_none());
        // Unknown uid: returns None.
        let bogus = CardUid::new(999).expect("non-zero");
        assert!(transform_card_in_place(
            &mut state,
            bogus,
            placeholder_card(crate::cards::CardId::Wound)
        )
        .is_none());
    }

    #[test]
    fn upgrade_and_downgrade_card_uid_flip_flag_idempotently() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // A live player: at 0 HP combat is ending and nothing upgrades.
        state.player.current_hp = 50;
        let mut shiv = placeholder_card(crate::cards::CardId::Shiv);
        shiv.card_type = crate::cards::CardType::Attack;
        let u = spawn_card_into_pile(&mut state, shiv, PileKind::Hand);
        // Initial state: not upgraded.
        let idx = u.0.get() as usize - 1;
        assert!(!state.cards[idx].upgraded);
        // First upgrade flips → true returned.
        assert!(upgrade_card_uid(&mut state, u));
        assert!(state.cards[idx].upgraded);
        // Idempotent: second upgrade returns false.
        assert!(!upgrade_card_uid(&mut state, u));
        assert!(state.cards[idx].upgraded);
        // Downgrade flips back → true returned.
        assert!(downgrade_card_uid(&mut state, u));
        assert!(!state.cards[idx].upgraded);
        // Idempotent downgrade → false.
        assert!(!downgrade_card_uid(&mut state, u));
        // Unknown uid: both return false.
        let bogus = CardUid::new(999).expect("non-zero");
        assert!(!upgrade_card_uid(&mut state, bogus));
        assert!(!downgrade_card_uid(&mut state, bogus));
    }

    /// C# `CardCmd.Upgrade` (`CardCmd.cs:273`) skips a card that isn't
    /// `IsUpgradable` (a curse, a status, a Quest card), so the combat-side
    /// upgrade by uid leaves it as it is (Bellows used to upgrade Ascender's
    /// Bane).
    #[test]
    fn upgrade_card_uid_is_a_no_op_while_combat_is_ending() {
        // C# `CardCmd.Upgrade` bails on `CombatManager.IsEnding`
        // (CardCmd.cs:267-270): nothing upgrades once the last primary
        // enemy is dead (GH #1150).
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 50;
        let enemy = state.spawn_enemy(crate::MonsterKind::Nibbit);
        let mut strike = placeholder_card(crate::cards::CardId::StrikeIronclad);
        strike.card_type = crate::cards::CardType::Attack;
        let u = spawn_card_into_pile(&mut state, strike, PileKind::Hand);
        crate::creature_by_id_mut(&mut state.creatures, enemy.0)
            .expect("enemy")
            .current_hp = 0;
        assert!(crate::combat_loop::combat_is_ending(&state));
        assert!(!upgrade_card_uid(&mut state, u));
        assert!(!state.cards[u.0.get() as usize - 1].upgraded);

        crate::creature_by_id_mut(&mut state.creatures, enemy.0)
            .expect("enemy")
            .current_hp = 5;
        assert!(upgrade_card_uid(&mut state, u));
    }

    #[test]
    fn upgrade_card_uid_skips_a_card_that_cannot_be_upgraded() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 50;
        let mut bane = placeholder_card(crate::cards::CardId::AscendersBane);
        bane.card_type = crate::cards::CardType::Curse;
        let u = spawn_card_into_pile(&mut state, bane, PileKind::Hand);
        assert!(!upgrade_card_uid(&mut state, u));
        assert!(!state.cards[u.0.get() as usize - 1].upgraded);
    }

    #[test]
    fn remove_card_from_any_pile_walks_canonical_order() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let u = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Shiv),
            PileKind::Exhaust,
        );
        let src = remove_card_from_any_pile(&mut state, u);
        assert_eq!(src, Some(PileKind::Exhaust));
        assert!(state.player.exhaust.is_empty());
        // Re-removal of the same uid → None.
        assert!(remove_card_from_any_pile(&mut state, u).is_none());
    }

    // ------------------------------------------------------------------
    // Late-effect bus (bead `sts2-rs-j9x8`)
    // ------------------------------------------------------------------

    /// Empty-bus early-out: `fire_late_effects` is a no-op when no
    /// handlers are registered for the cards present.
    #[test]
    fn late_effects_empty_bus_is_noop() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let _ = state.init_player_creature();
        // `Slimed` is a status card with no handlers — guaranteed
        // empty bus regardless of which other tests in the process
        // have registered late-effects.
        let _ = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Slimed),
            PileKind::Hand,
        );
        let snapshot_hand = state.player.hand.clone();
        let snapshot_discard = state.player.discard.clone();
        fire_late_effects(&mut state, &hooks::CardCtx::default());
        assert_eq!(state.player.hand, snapshot_hand);
        assert_eq!(state.player.discard, snapshot_discard);
    }

    /// Per-uid dedup: a card with a registered late-effect is visited
    /// exactly once per uid across the five player piles. Uses an
    /// unused-elsewhere `CardId` so the global registry doesn't
    /// collide with parallel tests.
    #[test]
    fn late_effects_dedup_each_uid_visited_once() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static PROBE_COUNT: AtomicU32 = AtomicU32::new(0);
        fn probe(_s: &mut CombatState, _w: CardUid, _c: &hooks::CardCtx) {
            PROBE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        // `Burn` is a status card; no production code registers a
        // late-effect for it, so this test owns the slot in the
        // global registry for its duration.
        register_late_effect(crate::cards::CardId::Burn, probe as LateEffectFn);
        PROBE_COUNT.store(0, Ordering::SeqCst);

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let _ = state.init_player_creature();
        let _h = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Hand,
        );
        let _d = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Discard,
        );
        let _x = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Exhaust,
        );

        fire_late_effects(&mut state, &hooks::CardCtx::default());
        assert_eq!(
            PROBE_COUNT.load(Ordering::SeqCst),
            3,
            "expected one fire per uid (Hand + Discard + Exhaust)"
        );

        // Restore: replace probe with a no-op so any later tests in
        // the same process don't observe spurious fires.
        fn noop(_s: &mut CombatState, _w: CardUid, _c: &hooks::CardCtx) {}
        register_late_effect(crate::cards::CardId::Burn, noop as LateEffectFn);
    }

    /// Bead `sts2-rs-65mbx` (GH #44): the late-effect sweep must fire ONLY
    /// for cards with a registered handler, regardless of how many
    /// handler-less cards bloat the piles. The pre-filter that makes this
    /// O(n) instead of O(n²) (the dedup spans only handler-bearing uids) is
    /// what stops a deck bloated with a self-duplicating card (Anger) from
    /// turning a single `play_card` into an effective infinite loop.
    #[test]
    fn late_effects_only_fire_handler_cards_amid_bloat() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static PROBE_COUNT: AtomicU32 = AtomicU32::new(0);
        fn probe(_s: &mut CombatState, _w: CardUid, _c: &hooks::CardCtx) {
            PROBE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        register_late_effect(crate::cards::CardId::Burn, probe as LateEffectFn);
        PROBE_COUNT.store(0, Ordering::SeqCst);

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let _ = state.init_player_creature();

        // One handler-bearing Burn in hand …
        let _burn = spawn_card_into_pile(
            &mut state,
            placeholder_card(crate::cards::CardId::Burn),
            PileKind::Hand,
        );
        // … buried under thousands of handler-less Anger copies in discard
        // (the #44 deck-bloat shape). With the old O(n²) dedup this scan was
        // ~5M `seen.contains` ops; with the pre-filter the bloat is skipped.
        for _ in 0..5_000 {
            let _ = spawn_card_into_pile(
                &mut state,
                placeholder_card(crate::cards::CardId::Anger),
                PileKind::Discard,
            );
        }

        fire_late_effects(&mut state, &hooks::CardCtx::default());
        assert_eq!(
            PROBE_COUNT.load(Ordering::SeqCst),
            1,
            "exactly the one handler-bearing Burn fires; the 5000 Anger copies are skipped"
        );

        fn noop(_s: &mut CombatState, _w: CardUid, _c: &hooks::CardCtx) {}
        register_late_effect(crate::cards::CardId::Burn, noop as LateEffectFn);
    }

    /// M06: the frozen-table sweep fires the same watched cards, in the same
    /// order and with the same uid dedup, as the locked-registry sweep it
    /// replaces. That includes a uid listed in two piles, and an arena with
    /// no late-effect card, which the frozen gate skips.
    #[test]
    #[serial_test::serial(global_registry)]
    fn frozen_late_sweep_matches_locked_sweep() {
        use std::sync::Mutex;

        static FIRED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        fn probe(_s: &mut CombatState, w: CardUid, _c: &hooks::CardCtx) {
            FIRED.lock().expect("probe log").push(w.0.get());
        }
        fn sweep(state: &CombatState) -> Vec<u32> {
            FIRED.lock().expect("probe log").clear();
            fire_late_effects(&mut state.clone(), &hooks::CardCtx::default());
            FIRED.lock().expect("probe log").clone()
        }
        register_late_effect(crate::cards::CardId::Burn, probe as LateEffectFn);

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let _ = state.init_player_creature();
        let mut burns = Vec::new();
        for pile in [
            crate::cards::PileKind::Play,
            crate::cards::PileKind::Exhaust,
            crate::cards::PileKind::Discard,
            crate::cards::PileKind::Draw,
            crate::cards::PileKind::Hand,
        ] {
            for card_id in [crate::cards::CardId::Anger, crate::cards::CardId::Burn] {
                let uid =
                    spawn_card_into_pile(&mut state, placeholder_card(card_id), PileKind::Hand);
                crate::cards::move_card(&mut state.player, uid, crate::cards::PileKind::Hand, pile)
                    .expect("spawned into hand");
                if card_id == crate::cards::CardId::Burn {
                    burns.push(uid);
                }
            }
        }
        // The Play-pile Burn also listed in discard: one fire, at its first
        // (discard) position.
        state.player.discard.push(burns[0]);
        let mut no_burns = state.clone();
        for card in &mut no_burns.cards {
            card.card_id = crate::cards::CardId::Anger;
        }

        assert!(!card_tables_frozen());
        let locked = sweep(&state);
        let locked_none = sweep(&no_burns);
        freeze_card_tables();
        assert!(card_tables_frozen());
        let frozen = sweep(&state);
        let frozen_none = sweep(&no_burns);

        // burns = [Play, Exhaust, Discard, Draw, Hand]; sweep order is Hand,
        // Draw, Discard (its own Burn, then the Play Burn), Exhaust.
        let expected: Vec<u32> = [4, 3, 2, 0, 1].map(|i| burns[i].0.get()).to_vec();
        assert_eq!(locked, expected);
        assert_eq!(frozen, expected);
        assert!(locked_none.is_empty());
        assert!(frozen_none.is_empty());

        fn noop(_s: &mut CombatState, _w: CardUid, _c: &hooks::CardCtx) {}
        register_late_effect(crate::cards::CardId::Burn, noop as LateEffectFn);
    }

    /// M06: the frozen tables answer exactly as the registries do. Any
    /// registration suspends them; a freeze re-validates a registry that
    /// still matches (the production re-install) and keeps a changed one on
    /// the locked path.
    #[test]
    #[serial_test::serial(global_registry)]
    fn frozen_card_tables_track_the_registries() {
        fn other_strike(_s: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
            EffectResult::NotImplemented
        }
        let addr = |f: Option<PlayCardFn>| f.map(|f| f as usize);
        let strike = crate::cards::CardId::StrikeIronclad;

        install_default_registry();
        let installed = lookup(strike);
        assert!(installed.is_some());
        freeze_card_tables();
        assert!(card_tables_frozen());
        assert_eq!(addr(lookup(strike)), addr(installed));
        assert_eq!(addr(lookup(crate::cards::CardId::Whirlwind)), None);

        // Same handler again (a re-install): suspended until the next freeze,
        // which finds the registry unchanged.
        register(strike, installed.expect("installed"));
        assert!(!card_tables_frozen());
        assert_eq!(addr(lookup(strike)), addr(installed));
        freeze_card_tables();
        assert!(card_tables_frozen());

        // A different handler: the locked registry serves it, and a freeze
        // cannot re-validate the table while the registry differs from it.
        register(strike, other_strike as PlayCardFn);
        freeze_card_tables();
        assert!(!card_tables_frozen());
        assert_eq!(addr(lookup(strike)), addr(Some(other_strike as PlayCardFn)));

        register(strike, installed.expect("installed"));
    }

    /// Bead `sts2-rs-st9tl`: applying Vulnerable / Weak / Frail to the
    /// PLAYER that didn't already carry the power must set
    /// `skip_next_tick = true` on the freshly-pushed instance. Mirrors
    /// C# `PowerCmd.Apply` (PowerCmd.cs:104) `target.Side ==
    /// CombatSide.Player && power.Type == PowerType.Debuff`. (The
    /// monster-target case is the negative in
    /// [`monster_vwf_apply_does_not_arm_skip_next_tick`].)
    #[test]
    fn apply_power_sets_skip_next_tick_for_vuln_weak_frail_on_fresh_apply() {
        for kind in [PowerKind::Vulnerable, PowerKind::Weak, PowerKind::Frail] {
            let mut state = CombatState {
                rng: crate::dotnet_rng::Rng::new(0_i32),
                ..Default::default()
            };
            state.player.current_hp = 50;
            state.player.max_hp = 50;
            let target = state.init_player_creature();
            apply_power_to_creature(&mut state, target, kind, 1);
            let power = state
                .powers
                .iter()
                .find(|p| p.kind == kind)
                .expect("power pushed");
            assert!(
                power.skip_next_tick,
                "{kind:?}: fresh apply on the PLAYER must set skip_next_tick"
            );
            assert_eq!(power.amount, 1);
        }
    }

    /// Bead `sts2-rs-st9tl`: applying Vulnerable / Weak / Frail to a
    /// MONSTER must NOT set `skip_next_tick` — C# `PowerCmd.Apply`
    /// gates the skip on `target.Side == CombatSide.Player`
    /// (PowerCmd.cs:104), so a player-applied monster debuff
    /// decrements normally at the next enemy-turn-end. Regression
    /// guard for the monster-Vulnerable-stuck-one-tick-high bug
    /// (Rust 2 vs C# 1) the original j0ek port introduced by setting
    /// the flag unconditionally (both C# `PowerCmd.cs:104` and the
    /// Python pin `combat.py:2038` gate the skip on a PLAYER target).
    #[test]
    fn monster_vwf_apply_does_not_arm_skip_next_tick() {
        for kind in [PowerKind::Vulnerable, PowerKind::Weak, PowerKind::Frail] {
            let mut state = CombatState::default();
            let target = state.spawn_enemy(MonsterKind::Nibbit);
            apply_power_to_creature(&mut state, target, kind, 2);
            let power = state
                .powers
                .iter()
                .find(|p| p.kind == kind)
                .expect("power pushed");
            assert!(
                !power.skip_next_tick,
                "{kind:?}: fresh apply on a MONSTER must NOT set skip_next_tick"
            );
            assert_eq!(power.amount, 2);
        }
    }

    /// Bead `sts2-rs-ybtmu`: Ritual applied to an ENEMY arms
    /// `skip_next_tick` (C# `RitualPower.AfterApplied` sets
    /// `WasJustAppliedByEnemy` when `Owner.IsEnemy`), so the first
    /// owner-turn-end skips the Strength grant. Ritual applied to the
    /// PLAYER (MazalethsGift potion) does NOT arm it — the player gains
    /// Strength at the first player-turn-end. Regression guard for the
    /// DevotedSculptor "Strength granted one turn early" drift.
    #[test]
    fn ritual_arms_skip_next_tick_for_enemy_not_player() {
        // enemy: armed
        let mut es = CombatState::default();
        let enemy = es.spawn_enemy(MonsterKind::Nibbit);
        apply_power_to_creature(&mut es, enemy, PowerKind::Ritual, 9);
        let ep = es
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Ritual)
            .expect("enemy Ritual pushed");
        assert!(
            ep.skip_next_tick,
            "enemy-applied Ritual must arm skip_next_tick (WasJustAppliedByEnemy)"
        );

        // player: not armed
        let mut ps = CombatState::default();
        ps.player.current_hp = 50;
        ps.player.max_hp = 50;
        let player = ps.init_player_creature();
        apply_power_to_creature(&mut ps, player, PowerKind::Ritual, 1);
        let pp = ps
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Ritual)
            .expect("player Ritual pushed");
        assert!(
            !pp.skip_next_tick,
            "player-applied Ritual must NOT arm skip_next_tick"
        );
    }

    /// Bead `sts2-rs-j0ek`: re-applying an existing V/W/F stack must
    /// NOT touch the skip flag (stacking semantics match Python's
    /// "the flag belongs to the original push and is consumed on
    /// the next tick regardless of re-applications").
    #[test]
    fn apply_power_stacking_does_not_reset_skip_next_tick() {
        use std::num::NonZeroU32;

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let uid = NonZeroU32::new(1).expect("nz");
        state.creatures.push(Creature {
            id: uid,
            current_hp: 50,
            max_hp: 50,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Nibbit),
            monster_id: crate::MonsterId("Nibbit"),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: None,
        });
        let target = CreatureUid(uid);
        apply_power_to_creature(&mut state, target, PowerKind::Weak, 2);
        // Manually clear the skip flag — simulates "the original tick
        // already fired and consumed the flag, then someone stacks on
        // a new turn."
        for p in &mut state.powers {
            if p.kind == PowerKind::Weak {
                p.skip_next_tick = false;
            }
        }
        apply_power_to_creature(&mut state, target, PowerKind::Weak, 3);
        let weak = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Weak)
            .expect("Weak stacked");
        assert_eq!(weak.amount, 5, "stacking sums amounts");
        assert!(
            !weak.skip_next_tick,
            "stacking must not re-set the consumed skip flag"
        );
    }

    /// Bead `sts2-rs-j0ek`: applying a non-V/W/F power must NOT set
    /// the skip flag. The flag is gated to the three duration debuffs
    /// where Python uses the short-circuit; other powers tick
    /// normally on the same turn-end where they were applied.
    #[test]
    fn apply_power_does_not_set_skip_next_tick_for_other_kinds() {
        use std::num::NonZeroU32;

        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        let uid = NonZeroU32::new(1).expect("nz");
        state.creatures.push(Creature {
            id: uid,
            current_hp: 50,
            max_hp: 50,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Nibbit),
            monster_id: crate::MonsterId("Nibbit"),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: None,
        });
        let target = CreatureUid(uid);
        apply_power_to_creature(&mut state, target, PowerKind::Strength, 3);
        let strength = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("Strength pushed");
        assert!(
            !strength.skip_next_tick,
            "Strength is not a tickable-duration power; flag must stay false"
        );
    }

    // -----------------------------------------------------------------
    // Bead `sts2-rs-kand1`: apply_power_to_creature stack semantics.
    // Pins the 4 contract points: amount sums, counter preserved,
    // applier latest-wins (via the _with_applier surface), and
    // skip_next_tick set on FRESH application only.
    // -----------------------------------------------------------------

    /// Stacking a second instance of the same `kind` sums the amount
    /// via saturating_add. Mirrors Python `PowerInstance.add_stacks`.
    #[test]
    fn kand1_apply_power_stacks_amount_saturating() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 2);
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 3);
        let strength = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("Strength pushed");
        assert_eq!(strength.amount, 5, "stacked amount = 2 + 3");
        // Only one instance — the second call stacked into the first.
        assert_eq!(
            state
                .powers
                .iter()
                .filter(|p| p.kind == PowerKind::Strength)
                .count(),
            1,
        );
    }

    /// Per-instance `counter` is preserved across re-application
    /// (bead `sts2-rs-6ynt.6.11` policy: per-instance scratch is a
    /// running total — adding stacks doesn't reset it).
    #[test]
    fn kand1_apply_power_preserves_counter_on_stack() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 1);
        // Simulate a listener that bumped the counter.
        state
            .powers
            .iter_mut()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("present")
            .counter = 7;
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 4);
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("present");
        assert_eq!(p.amount, 5, "amount stacked normally");
        assert_eq!(p.counter, 7, "per-instance counter preserved across stack");
    }

    /// `apply_power_to_creature_with_applier` overwrites the
    /// `applier` field on re-application (latest-wins). The base
    /// surface (no applier) leaves the field unchanged.
    #[test]
    fn kand1_apply_power_with_applier_rebinds_latest_wins() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        let applier_a = state.spawn_enemy(MonsterKind::Nibbit);
        let applier_b = state.spawn_enemy(MonsterKind::Nibbit);

        apply_power_to_creature_with_applier(
            &mut state,
            player_uid,
            PowerKind::Strength,
            1,
            Some(applier_a),
        );
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("present");
        assert_eq!(p.applier, Some(applier_a), "first apply binds applier_a");

        apply_power_to_creature_with_applier(
            &mut state,
            player_uid,
            PowerKind::Strength,
            1,
            Some(applier_b),
        );
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("present");
        assert_eq!(
            p.applier,
            Some(applier_b),
            "stack rebinds to applier_b (latest-wins)"
        );
    }

    /// Bead `sts2-rs-x3ql3`: short-circuit paths in
    /// `apply_power_to_creature`. Each early-return must leave
    /// `state.powers` unchanged.
    #[test]
    fn x3ql3_apply_power_short_circuits_on_non_positive_amount() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        let powers_before = state.powers.len();

        // amount <= 0 returns early before the creature lookup.
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, 0);
        apply_power_to_creature(&mut state, player_uid, PowerKind::Strength, -3);
        assert_eq!(
            state.powers.len(),
            powers_before,
            "amount <= 0 must not add a power",
        );
    }

    /// `apply_power_to_creature` silently no-ops when `target_uid`
    /// is not present in the arena. No panic.
    #[test]
    fn x3ql3_apply_power_short_circuits_on_missing_target() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        let powers_before = state.powers.len();

        let phantom = CreatureUid(NonZeroU32::new(9999).expect("non-zero"));
        apply_power_to_creature(&mut state, phantom, PowerKind::Strength, 5);
        assert_eq!(
            state.powers.len(),
            powers_before,
            "missing target uid must not add a power",
        );
    }

    /// Bead `sts2-rs-1tulq`: companion short-circuit coverage for
    /// `apply_power_to_creature_with_applier`. Same three guards
    /// (amount<=0, missing target, dead target) as the base
    /// surface — pin them all so a regression that drifts one
    /// branch but not the other surfaces immediately.
    #[test]
    fn t1ulq_apply_power_with_applier_short_circuits_on_non_positive_amount() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        let applier = state.spawn_enemy(MonsterKind::Nibbit);
        let powers_before = state.powers.len();
        apply_power_to_creature_with_applier(
            &mut state,
            player_uid,
            PowerKind::Strength,
            0,
            Some(applier),
        );
        apply_power_to_creature_with_applier(
            &mut state,
            player_uid,
            PowerKind::Strength,
            -5,
            Some(applier),
        );
        assert_eq!(state.powers.len(), powers_before);
    }

    #[test]
    fn t1ulq_apply_power_with_applier_short_circuits_on_missing_target() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        let applier = state.spawn_enemy(MonsterKind::Nibbit);
        let powers_before = state.powers.len();
        let phantom = CreatureUid(NonZeroU32::new(9999).expect("non-zero"));
        apply_power_to_creature_with_applier(
            &mut state,
            phantom,
            PowerKind::Strength,
            3,
            Some(applier),
        );
        assert_eq!(state.powers.len(), powers_before);
    }

    #[test]
    fn t1ulq_apply_power_with_applier_short_circuits_on_dead_target() {
        let mut state = CombatState::default();
        let enemy_uid = state.spawn_enemy(MonsterKind::Nibbit);
        let applier_uid = state.spawn_enemy(MonsterKind::Nibbit);
        for c in &mut state.creatures {
            if c.id == enemy_uid.0 {
                c.current_hp = 0;
                break;
            }
        }
        let powers_before = state.powers.len();
        apply_power_to_creature_with_applier(
            &mut state,
            enemy_uid,
            PowerKind::Strength,
            5,
            Some(applier_uid),
        );
        assert_eq!(state.powers.len(), powers_before);
    }

    /// Fresh application via `_with_applier` persists the applier
    /// on the newly-created Power record (kand1 covered the stack
    /// path; this pins the fresh-spawn path).
    #[test]
    fn t1ulq_apply_power_with_applier_persists_applier_on_fresh_spawn() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let player_uid = state.init_player_creature();
        let applier = state.spawn_enemy(MonsterKind::Nibbit);
        apply_power_to_creature_with_applier(
            &mut state,
            player_uid,
            PowerKind::Strength,
            2,
            Some(applier),
        );
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Strength)
            .expect("Strength pushed");
        assert_eq!(p.applier, Some(applier), "fresh apply persists applier");
        assert_eq!(p.amount, 2);
    }

    /// A dead creature (`current_hp == 0`) is treated the same as a
    /// missing target. Mirrors Python `apply_power` filter that
    /// skips dead creatures.
    #[test]
    fn x3ql3_apply_power_short_circuits_on_dead_target() {
        let mut state = CombatState::default();
        let enemy_uid = state.spawn_enemy(MonsterKind::Nibbit);
        // Kill the spawned enemy.
        for c in &mut state.creatures {
            if c.id == enemy_uid.0 {
                c.current_hp = 0;
                break;
            }
        }
        let powers_before = state.powers.len();
        apply_power_to_creature(&mut state, enemy_uid, PowerKind::Strength, 5);
        assert_eq!(
            state.powers.len(),
            powers_before,
            "dead target (current_hp=0) must not receive new powers",
        );
    }

    /// PillarOfCreation grants `Amount` block per PLAYER-generated card
    /// (C# `AfterCardGeneratedForCombat`, gated on `addedByPlayer` ≈ the player
    /// being mid-card-play). The effect was defined but never wired into any
    /// generation site, so generated cards granted 0 block (Regent fuzz seeds
    /// 142/309/458). Mirrors the Arsenal-Strength generation grant.
    #[test]
    fn pillar_of_creation_grants_block_on_player_card_generation() {
        use std::num::NonZeroU32;
        let mut state = CombatState::default();
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::PillarOfCreation,
            amount: 3,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));

        // added_by_player = true (a player-generated card) → block granted.
        fire_pillar_of_creation_block_on_card_generation(
            &mut state,
            crate::CardUid::new(1).expect("nz"),
            true,
        );
        assert_eq!(
            state.player.block, 3,
            "3 block granted per player-generated card"
        );

        // added_by_player = false (a monster-added card on the enemy turn) → no block.
        fire_pillar_of_creation_block_on_card_generation(
            &mut state,
            crate::CardUid::new(1).expect("nz"),
            false,
        );
        assert_eq!(
            state.player.block, 3,
            "no block when not added by the player (monster-generated card)"
        );
    }

    /// `route_generated_into_hand` (the player generates-cards-into-hand path:
    /// SpectrumShift / CallOfTheVoid turn-start powers, BigHat/VexingPuzzlebox
    /// relics, potions) must fire the `AfterCardGeneratedForCombat` listeners
    /// with `addedByPlayer = true` — even at TURN START, where the Play pile is
    /// empty. The prior port fired none of them, so a turn-start SpectrumShift
    /// granted no PillarOfCreation block (parity fuzz Regent seeds 422/494).
    /// Bead `sts2-rs-gen-added-by-player`.
    #[test]
    fn route_generated_into_hand_fires_pillar_at_turn_start() {
        use std::num::NonZeroU32;
        let mut state = CombatState::default();
        // Alive player: a dead one means combat is ending (no listeners run).
        state.player.current_hp = 80;
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::PillarOfCreation,
            amount: 4,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Empty Play pile = turn-start generation (the old heuristic would have
        // blocked it).
        assert!(state.player.play.is_empty());
        // A generated card already materialized in the arena (as the SpectrumShift
        // sampler would leave it), routed into hand.
        let card_uid = CardUid::new(1).expect("nz");
        state.cards.push(crate::cards::Card {
            uid: card_uid,
            card_id: crate::cards::CardId::Shiv,
            cost: crate::CardEnergyCost::new(0),
            card_type: crate::cards::CardType::Skill,
            target_type: crate::cards::TargetType::Enemy,
            upgraded: false,
            base_damage: 0,
            base_block: 0,
            combat_vars: crate::cards::CombatVars::new(),
            enchantments: crate::cards::CardEnchantments::new(),
            tags: crate::cards::CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        });

        route_generated_into_hand(&mut state, &[card_uid]);

        assert_eq!(
            state.player.block, 4,
            "turn-start (empty Play pile) generation must still grant PillarOfCreation block"
        );
    }

    /// A card generated into hand has no `DeckVersion` in C#, so Thieving
    /// Hopper's steal skips it (loadout Regent 1255 v0.107.1: an Orange Dough
    /// Mind Blast was stolen in Rust).
    #[test]
    fn route_generated_into_hand_marks_the_card_generated() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        let card_uid = CardUid::new(1).expect("nz");
        state.cards.push(crate::cards::Card {
            uid: card_uid,
            card_id: crate::cards::CardId::Shiv,
            card_type: crate::cards::CardType::Attack,
            ..crate::cards::Card::pod_defaults()
        });

        route_generated_into_hand(&mut state, &[card_uid]);

        assert_eq!(
            state.cards[0].combat_var(crate::cards::VarKey::GeneratedInCombat),
            1
        );
    }

    /// `route_generated_into_hand` writes the `CardGenerated` event-log row
    /// every other generation path writes (#1105), and still bumps the
    /// Supermassive tally once per card.
    #[test]
    fn route_generated_into_hand_logs_card_generated() {
        let mut state = CombatState::default();
        state.player.current_hp = 80;
        state.combat_log.event_log.set_recording(true);
        let card_uid = CardUid::new(1).expect("nz");
        state.cards.push(crate::cards::Card {
            uid: card_uid,
            card_id: crate::cards::CardId::Shiv,
            ..crate::cards::Card::pod_defaults()
        });

        route_generated_into_hand(&mut state, &[card_uid]);

        assert_eq!(state.player.generated_cards_this_combat, 1);
        let generated = state
            .combat_log
            .event_log
            .rows()
            .iter()
            .filter(|r| {
                matches!(
                    r.entry,
                    crate::event_log::EventLogEntry::CardGenerated {
                        card: crate::cards::CardId::Shiv,
                        pile: "hand",
                        by_player: true,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(generated, 1);
    }

    /// `spawn_card_into_hand_with_origin(.., true)` must fire `ArsenalPower`'s
    /// Strength grant even with an EMPTY Play pile — the path taken by
    /// potion-generated cards (SkillPotion / AttackPotion / ColorlessPotion,
    /// whose `OnUse` puts nothing on the Play pile) and the turn-start
    /// `FurnacePower` forge. C# `CardPileCmd.AddGeneratedCardToCombat` passes
    /// `addedByPlayer: true` for all of these, so Arsenal must trigger. The
    /// plain heuristic-based [`spawn_card_into_hand`] reads `false` on an empty
    /// Play pile and would drop the grant (Regent fuzz seed 42: a
    /// SkillPotion-generated Skill + turn-start forge each missed Arsenal(1)).
    /// Bead `sts2-rs-potion-gen-arsenal`.
    #[test]
    fn spawn_with_origin_true_fires_arsenal_on_empty_play_pile() {
        use std::num::NonZeroU32;
        let mut state = CombatState::default();
        // Alive player: a dead one means combat is ending (no listeners run).
        state.player.current_hp = 80;
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::Arsenal,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));
        // Empty Play pile = a potion use / turn-start forge (the heuristic in
        // plain `spawn_card_into_hand` would read `added_by_player = false`).
        assert!(state.player.play.is_empty());

        let make_card = || crate::cards::Card {
            uid: CardUid::new(1).expect("nz"),
            card_id: crate::cards::CardId::Shiv,
            cost: crate::CardEnergyCost::new(0),
            card_type: crate::cards::CardType::Skill,
            target_type: crate::cards::TargetType::Enemy,
            upgraded: false,
            base_damage: 0,
            base_block: 0,
            combat_vars: crate::cards::CombatVars::new(),
            enchantments: crate::cards::CardEnchantments::new(),
            tags: crate::cards::CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        };

        // Plain heuristic path on an empty Play pile → added_by_player=false → no grant.
        spawn_card_into_hand(&mut state, make_card());
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            0,
            "heuristic path on empty Play pile must NOT fire Arsenal",
        );

        // Explicit-origin path with true → Arsenal fires regardless of Play pile.
        spawn_card_into_hand_with_origin(&mut state, make_card(), true);
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            1,
            "explicit added_by_player=true must fire Arsenal(1) Strength on empty Play pile",
        );
    }

    /// ChildOfTheStars grants `Amount * stars_spent` block when the player
    /// SPENDS stars (C# `ChildOfTheStarsPower.AfterStarsSpent`). The helper was
    /// defined and unit-tested but never wired into `spend_stars`, so star-cost
    /// cards (e.g. CrescentSpear) granted no block (Regent fuzz seeds 22/241).
    #[test]
    fn child_of_the_stars_grants_block_on_stars_spent() {
        use std::num::NonZeroU32;
        let mut state = CombatState::default();
        state.player.stars = 3;
        let pid = NonZeroU32::new(1).expect("nz");
        state.powers.push(crate::Power {
            id: pid,
            kind: PowerKind::ChildOfTheStars,
            amount: 2,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(crate::PowerUid(pid));

        let spent = state.spend_stars(2);
        assert_eq!(spent, 2, "spent 2 of 3 stars");
        assert_eq!(
            state.player.block, 4,
            "ChildOfTheStars(2) * 2 stars spent = 4 block"
        );

        // Spending 0 stars grants no block.
        let s0 = state.spend_stars(0);
        assert_eq!(s0, 0);
        assert_eq!(state.player.block, 4, "no spend -> no extra block");
    }

    /// `skip_next_tick` is set on FRESH application of
    /// Vulnerable/Weak/Frail only. Stacking a second instance does
    /// not re-arm the flag (the flag is a one-shot "defer first
    /// decrement"; a re-application after the flag was already
    /// consumed must not reset it).
    #[test]
    fn kand1_apply_power_skip_next_tick_armed_on_fresh_only() {
        // st9tl: skip_next_tick only arms for PLAYER-targeted debuffs
        // (C# PowerCmd.cs:104), so target the player here.
        let mut state = CombatState::default();
        state.player.current_hp = 50;
        state.player.max_hp = 50;
        let player_uid = state.init_player_creature();
        apply_power_to_creature(&mut state, player_uid, PowerKind::Vulnerable, 2);
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Vulnerable)
            .expect("Vulnerable pushed");
        assert!(p.skip_next_tick, "fresh Vulnerable arms skip_next_tick");

        // Simulate a tick consuming the flag.
        state
            .powers
            .iter_mut()
            .find(|p| p.kind == PowerKind::Vulnerable)
            .expect("present")
            .skip_next_tick = false;

        // Re-apply: stack into the existing power. Flag must NOT
        // re-arm — stacking doesn't reset the one-shot defer.
        apply_power_to_creature(&mut state, player_uid, PowerKind::Vulnerable, 1);
        let p = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Vulnerable)
            .expect("present");
        assert_eq!(p.amount, 3, "amount stacked to 3");
        assert!(
            !p.skip_next_tick,
            "stacked re-application must not re-arm skip_next_tick",
        );
    }

    /// PARITY128 (Necrobinder, seed PARITY128, A0). When a card's OnPlay
    /// generates cards AFTER the killing blow (Severance: damage, then 3 souls),
    /// combat is already ending. C# `CardPileCmd.Add(.., CardPile, ..)`
    /// short-circuits with `success=false` BEFORE the `CardPilePosition.Random`
    /// shuffle read (`Rng.Shuffle.NextInt(Count + 1)`, `CardPileCmd.cs:240/433`),
    /// so no soul enters a pile and -- crucially -- the run-level `Rng.Shuffle`
    /// stream is NOT advanced. The random-position spawn helper is the only one
    /// that reads `run_rng.shuffle`, so over-reading it here would desync every
    /// later combat's opening StableShuffle (sibling of the PARITY86 mid-draw
    /// reshuffle gate).
    #[test]
    fn random_position_spawn_skips_shuffle_read_when_combat_ending() {
        let (mut state, enemy) = fresh_state_with_one_enemy();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();

        // Killing blow already landed: the lone primary enemy is at 0 HP, so
        // `combat_is_ending` is true even though `is_over` has not flipped yet.
        state
            .creatures
            .iter_mut()
            .find(|c| c.id == enemy.0)
            .expect("enemy present")
            .current_hp = 0;
        assert!(
            crate::combat_loop::combat_is_ending(&state),
            "lethal hit: combat is ending",
        );

        let rng_before = state.run_rng.shuffle.clone();
        let arena_before = state.cards.len();
        let draw_before = state.player.draw.len();

        let uid = spawn_card_into_draw_pile_random_position(
            &mut state,
            placeholder_card(crate::cards::CardId::Soul),
        );

        // The run-level shuffle stream must NOT advance (no Random read).
        assert_eq!(
            state.run_rng.shuffle, rng_before,
            "combat ending: must not consume an Rng.Shuffle read",
        );
        // The card is still allocated (uid stays dereferenceable, mirroring
        // C# returning `cardAdded = c` with `success = false`)...
        assert_eq!(
            state.cards.len(),
            arena_before + 1,
            "uid is allocated in the arena even though the spawn is dropped",
        );
        assert!(
            state.cards.iter().any(|c| c.uid == uid),
            "returned uid resolves to the pushed card",
        );
        // ...but it never enters the draw pile.
        assert_eq!(
            state.player.draw.len(),
            draw_before,
            "combat ending: generated soul does not enter the draw pile",
        );
        assert!(
            !state.player.draw.contains(&uid),
            "generated soul not in draw pile",
        );
    }

    /// Sanity counterpart: while combat is IN PROGRESS (enemy still alive), the
    /// random-position spawn DOES consume a shuffle read and inserts the card --
    /// so the gate above is the only behavioral change.
    #[test]
    fn random_position_spawn_reads_shuffle_when_combat_in_progress() {
        let (mut state, _enemy) = fresh_state_with_one_enemy();
        state.player.current_hp = 80;
        state.player.max_hp = 80;
        let _ = state.init_player_creature();
        assert!(
            !crate::combat_loop::combat_is_ending(&state),
            "enemy alive + player alive: combat in progress",
        );

        let rng_before = state.run_rng.shuffle.clone();
        let uid = spawn_card_into_draw_pile_random_position(
            &mut state,
            placeholder_card(crate::cards::CardId::Soul),
        );

        assert_ne!(
            state.run_rng.shuffle, rng_before,
            "in progress: the Random position consumes one Rng.Shuffle read",
        );
        assert!(
            state.player.draw.contains(&uid),
            "in progress: generated soul enters the draw pile",
        );
    }
}

#[cfg(test)]
mod should_owner_death_trigger_fatal_tests {
    use super::*;
    use crate::MonsterKind;

    fn fresh() -> CombatState {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        state.player.current_hp = 70;
        state.player.max_hp = 70;
        let _ = state.init_player_creature();
        state
    }

    fn kill(state: &mut CombatState, uid: CreatureUid) {
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
            c.current_hp = 0;
        }
    }

    /// Baseline: a plain monster carries no `ShouldOwnerDeathTriggerFatal`
    /// override, so the fold is `true` (`PowerModel.cs:648`).
    #[test]
    fn plain_monster_triggers_fatal() {
        let mut state = fresh();
        let nibbit = state.spawn_enemy(MonsterKind::Nibbit);
        assert!(should_owner_death_trigger_fatal(&state, nibbit));
    }

    /// `MinionPower.cs:20` returns `false` unconditionally.
    #[test]
    fn minion_never_triggers_fatal() {
        let mut state = fresh();
        let e = state.spawn_enemy(MonsterKind::Nibbit);
        apply_power_to_creature(&mut state, e, PowerKind::Minion, 1);
        assert!(!should_owner_death_trigger_fatal(&state, e));
    }

    /// `ReattachPower.cs:106` → `AreAllOtherSegmentsDead()`. With a sibling
    /// segment still standing the killed segment will REATTACH, so the death
    /// is NOT fatal; once every other segment is a corpse the same power
    /// returns `true` and the killing blow on the last segment IS fatal.
    ///
    /// This is the case the old "any Reattach blocks" gate got wrong — it
    /// swallowed Feed's max HP / HandOfGreed's gold / TheHunt's reward on
    /// exactly the hit that ends the Decimillipede elite.
    #[test]
    fn reattach_is_fatal_only_once_every_other_segment_is_dead() {
        let mut state = fresh();
        // `spawn_enemy` applies the species' on-spawn buffs, so each segment
        // arrives carrying Reattach(25) — C# `AfterAddedToRoom`.
        let front = state.spawn_enemy(MonsterKind::DecimillipedeSegmentFront);
        let middle = state.spawn_enemy(MonsterKind::DecimillipedeSegmentMiddle);
        let back = state.spawn_enemy(MonsterKind::DecimillipedeSegmentBack);
        for uid in [front, middle, back] {
            assert!(
                state
                    .creatures
                    .iter()
                    .find(|c| c.id == uid.0)
                    .expect("spawned")
                    .powers
                    .iter()
                    .any(|pu| state
                        .powers
                        .get_by_id(pu.0)
                        .is_some_and(|p| p.kind == PowerKind::Reattach)),
                "segment spawns with Reattach"
            );
        }

        // All three alive: killing any one is not fatal.
        for uid in [front, middle, back] {
            assert!(
                !should_owner_death_trigger_fatal(&state, uid),
                "two siblings still standing → the segment reattaches"
            );
        }

        // One sibling down; the other two still block each other.
        kill(&mut state, back);
        assert!(!should_owner_death_trigger_fatal(&state, front));
        assert!(!should_owner_death_trigger_fatal(&state, middle));
        // …and the corpse itself keeps its Reattach, so it too is still
        // non-fatal while a live sibling remains
        // (`ShouldPowerBeRemovedAfterOwnerDeath` → `false`).
        assert!(!should_owner_death_trigger_fatal(&state, back));

        // Last standing segment: every OTHER Reattach carrier is dead.
        kill(&mut state, middle);
        assert!(
            should_owner_death_trigger_fatal(&state, front),
            "killing the LAST segment is a genuine fatal"
        );
    }

    /// The scan keys on the Reattach power, not on the species: a
    /// non-Decimillipede enemy standing beside the segments does not keep
    /// the last segment's death non-fatal.
    #[test]
    fn non_segment_bystander_does_not_block_the_final_fatal() {
        let mut state = fresh();
        let front = state.spawn_enemy(MonsterKind::DecimillipedeSegmentFront);
        let back = state.spawn_enemy(MonsterKind::DecimillipedeSegmentBack);
        let bystander = state.spawn_enemy(MonsterKind::Nibbit);
        kill(&mut state, back);
        assert!(
            state
                .creatures
                .iter()
                .any(|c| c.id == bystander.0 && c.current_hp > 0),
            "bystander is alive"
        );
        assert!(
            should_owner_death_trigger_fatal(&state, front),
            "a live non-Reattach enemy is not an 'other segment'"
        );
    }
}

#[cfg(test)]
mod banshees_cry_entered_combat_tests {
    use super::*;

    /// `BansheesCry.AfterCardEnteredCombat` (`BansheesCry.cs:42-55`): a
    /// Banshee's Cry generated after Ethereal plays starts 2 cheaper per play
    /// (CallOfTheVoid, Necrobinder 1008 split). A copy of one already in combat
    /// (C# `CreateClone`, which skips the catch-up) keeps its cost.
    #[test]
    fn banshees_cry_entering_combat_catches_up_on_ethereal_plays() {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0_i32),
            ..Default::default()
        };
        // Alive player: a dead one means combat is ending (no listeners run).
        state.player.current_hp = 80;
        let banshee = |cost| crate::cards::Card {
            card_id: crate::cards::CardId::BansheesCry,
            cost: crate::CardEnergyCost::new(cost),
            ..crate::cards::Card::pod_defaults()
        };
        let cost_of = |state: &CombatState, uid: CardUid| {
            state.cards[uid.0.get() as usize - 1].local_cost().max(0)
        };
        // A deck copy present before any Ethereal play.
        let deck = spawn_card_into_pile(&mut state, banshee(9), PileKind::Draw);
        assert_eq!(cost_of(&state, deck), 9);

        // One Ethereal play: the AfterCardPlayed half shaves the deck copy.
        state.combat_log.ethereal_cards_played_this_combat = 1;
        state.discount_banshees_cry_on_ethereal_play();
        assert_eq!(cost_of(&state, deck), 7);

        // A fresh generation catches up: 9 - 2*1, upgraded 7 - 2*1.
        let fresh = spawn_card_into_hand(&mut state, banshee(9));
        assert_eq!(cost_of(&state, fresh), 7);
        let upgraded = spawn_card_into_hand(&mut state, banshee(7));
        assert_eq!(cost_of(&state, upgraded), 5);

        // A clone carries the count and is not shaved again.
        let clone = state.cards[deck.0.get() as usize - 1];
        let copy = spawn_card_into_hand(&mut state, clone);
        assert_eq!(cost_of(&state, copy), 7);

        // Floors at 0.
        state.combat_log.ethereal_cards_played_this_combat = 6;
        let floored = spawn_card_into_hand(&mut state, banshee(9));
        assert_eq!(cost_of(&state, floored), 0);
    }
}
