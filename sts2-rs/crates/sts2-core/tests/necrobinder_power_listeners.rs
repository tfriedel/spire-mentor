//! Necrobinder power-listener integration tests (bead `sts2-rs-4wppb`).
//!
//! Each of these Necrobinder powers was previously *inert*: the card played and
//! stamped the `PowerKind`, but no hook arm read it at runtime. These tests pin
//! the now-wired listener bodies end to end through the public `dispatch_*`
//! aggregators (which apply the effects each per-listener body returns), every
//! one mirroring its C# `PowerModel` in
//! `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Powers/`.
//!
//! These exercise pure-core surfaces (no content registry needed): powers are
//! pushed directly onto the player / a spawned enemy and the hook dispatched.

use sts2_core::card_effects;
use sts2_core::cards::{Card, CardId, CardTags, CardType, KeywordOverride};
use sts2_core::hooks::{
    self, CardCostCtx, CardCtx, CombatLifecycleCtx, CreatureRef, DamageCtx, DamageMul, ListenerRef,
    MaxEnergyCtx, PowerAppliedCtx, TurnSideCtx, ValueProps,
};
use sts2_core::{CardUid, CombatState, CreatureUid, MonsterKind, Power, PowerKind, PowerUid, Side};

/// Push a fresh `kind(amount)` power onto the player's legacy power list. The
/// uid is derived from the current `powers` length so it stays unique + nonzero.
/// Drive the folded `AfterPlayerTurnStart` powers (SummonNextTurn et al.) the
/// way `combat_loop` does — over a power snapshot, applying each instance's
/// effects. These kinds are skipped by the `after_side_turn_start` walk (bead
/// `sts2-rs-r2ixj`), so a test of their turn-start effect must drive this path.
fn dispatch_folded_after_player_turn_start(state: &mut CombatState, ctx: &TurnSideCtx) {
    let uids = state.player_power_uids();
    for uid in uids {
        if let Some(eff) = hooks::collect_folded_after_player_turn_start_one(state, uid, ctx) {
            hooks::apply_effects(state, &eff);
        }
    }
}

fn push_player_power(state: &mut CombatState, kind: PowerKind, amount: i32) -> PowerUid {
    let pid = PowerUid::new(state.powers.len() as u32 + 1).expect("nonzero power uid");
    state.powers.push(Power {
        id: pid.0,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(pid);
    pid
}

/// Attach a fresh `kind(amount)` power to an already-spawned enemy creature.
fn attach_enemy_power(
    state: &mut CombatState,
    enemy: CreatureUid,
    kind: PowerKind,
    amount: i32,
) -> PowerUid {
    let pid = PowerUid::new(state.powers.len() as u32 + 1).expect("nonzero power uid");
    state.powers.push(Power {
        id: pid.0,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    let c = state
        .creatures
        .iter_mut()
        .find(|c| c.id == enemy.0)
        .expect("spawned enemy creature present");
    c.powers.push(pid);
    pid
}

/// Sum the `Doom` stacks currently attached to the creature `uid`.
fn doom_on_creature(state: &CombatState, uid: CreatureUid) -> i32 {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("creature present");
    c.powers
        .iter()
        .filter_map(|pu| {
            state
                .powers
                .iter()
                .find(|p| p.id == pu.0 && p.kind == PowerKind::Doom)
        })
        .map(|p| p.amount)
        .sum()
}

/// C# `FriendshipPower.ModifyMaxEnergy`: `amount + base.Amount` for the owning
/// player — a permanent +Amount max-energy buff while the Counter Buff exists.
#[test]
fn friendship_folds_plus_amount_into_max_energy() {
    let mut state = CombatState::default();
    push_player_power(&mut state, PowerKind::Friendship, 2);
    let delta = hooks::dispatch_modify_max_energy(&state, &MaxEnergyCtx::default());
    assert_eq!(
        delta, 2,
        "Friendship(2) folds +2 max energy for the owning player"
    );
}

/// C# `EnergyNextTurnPower.AfterEnergyReset`: for the owning player,
/// `GainEnergy(Amount)` then `PowerCmd.Remove(this)` — a one-shot gain that
/// fully removes itself (NOT a per-turn decrement).
#[test]
fn energy_next_turn_grants_energy_then_self_removes() {
    let mut state = CombatState::default();
    state.gain_energy(3);
    let pid = push_player_power(&mut state, PowerKind::EnergyNextTurn, 2);

    hooks::dispatch_after_energy_reset(&mut state, &CombatLifecycleCtx::default());

    assert_eq!(state.player.energy, 5, "3 base + EnergyNextTurn(2) = 5");
    assert!(
        !state.powers.iter().any(|p| p.id == pid.0),
        "EnergyNextTurn removes itself entirely after firing"
    );
    assert!(
        !state.player.powers.contains(&pid),
        "EnergyNextTurn uid unlinked from the player power list"
    );
}

/// C# `StarNextTurnPower.AfterEnergyReset`: for the owning player,
/// `GainStars(Amount)` then `PowerCmd.Remove(this)` — the stars-side analogue
/// of `EnergyNextTurn`. A one-shot gain that fully removes itself, so the
/// power never lingers as a stale stack (the parity divergence this wiring
/// fixes — Regent's Convergence / HiddenCache apply it and C# converts it at
/// the next energy reset).
#[test]
fn star_next_turn_grants_stars_then_self_removes() {
    let mut state = CombatState::default();
    let pid = push_player_power(&mut state, PowerKind::StarNextTurn, 2);

    hooks::dispatch_after_energy_reset(&mut state, &CombatLifecycleCtx::default());

    assert_eq!(state.player.stars, 2, "StarNextTurn(2) grants 2 stars");
    assert!(
        !state.powers.iter().any(|p| p.id == pid.0),
        "StarNextTurn removes itself entirely after firing"
    );
    assert!(
        !state.player.powers.contains(&pid),
        "StarNextTurn uid unlinked from the player power list"
    );
}

/// C# `NeurosurgePower.AfterSideTurnStart`: on the owner's side turn start,
/// `PowerCmd.Apply<DoomPower>(Owner, Amount, Owner, null)` — the owner
/// doom-marks itself by Amount each of its own turns (no tick-down).
#[test]
fn neurosurge_dooms_self_on_owner_turn_start() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let player_uid = state.init_player_creature();
    push_player_power(&mut state, PowerKind::Neurosurge, 3);

    hooks::dispatch_after_side_turn_start(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(
        doom_on_creature(&state, player_uid),
        3,
        "Neurosurge(3) self-applies Doom(3) on the player's turn start"
    );

    // It does not tick down — a second owner turn start stacks Doom to 6.
    hooks::dispatch_after_side_turn_start(&mut state, &TurnSideCtx { side: Side::Player });
    assert_eq!(
        doom_on_creature(&state, player_uid),
        6,
        "Neurosurge keeps stacking Doom each owner turn (no decrement)"
    );

    // Wrong side (enemy turn start) is a no-op for a player-owned Neurosurge.
    hooks::dispatch_after_side_turn_start(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert_eq!(
        doom_on_creature(&state, player_uid),
        6,
        "enemy-side turn start does not fire a player-owned Neurosurge"
    );
}

/// C# `OblivionPower.Before/AfterCardPlayed` + `AfterTurnEnd`: a player-applied
/// Debuff living on an enemy. For every card the player plays *after* Oblivion
/// exists, apply `Amount` Doom to the owner enemy; the power removes itself at
/// the player's turn end. The `BeforeCardPlayed` arm
/// (`dispatch_before_card_played_powers`) records the card so `AfterCardPlayed`
/// fires — the card that APPLIED Oblivion is never armed (its before-play arm
/// ran before the power existed), so it does NOT self-Doom.
#[test]
fn oblivion_dooms_owner_each_card_and_self_removes_at_player_turn_end() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let pid = attach_enemy_power(&mut state, enemy, PowerKind::Oblivion, 3);

    // The applying card: AfterCardPlayed WITHOUT a preceding arm must NOT Doom
    // (C# `amountsForPlayedCards` never recorded the Oblivion card itself).
    hooks::dispatch_after_card_played(&mut state, &CardCtx::default());
    assert_eq!(
        doom_on_creature(&state, enemy),
        0,
        "the card that applied Oblivion does not self-Doom the owner"
    );

    // A real subsequent card play: arm in BeforeCardPlayed, then fire.
    hooks::dispatch_before_card_played_powers(&mut state, /*resolved_cost=*/ 1, false);
    hooks::dispatch_after_card_played(&mut state, &CardCtx::default());
    assert_eq!(
        doom_on_creature(&state, enemy),
        3,
        "first armed player card applies Oblivion's Amount (3) Doom to the owner enemy"
    );

    hooks::dispatch_before_card_played_powers(&mut state, 1, false);
    hooks::dispatch_after_card_played(&mut state, &CardCtx::default());
    assert_eq!(
        doom_on_creature(&state, enemy),
        6,
        "each subsequent armed player card stacks another 3 Doom"
    );

    // End of the PLAYER's turn removes Oblivion (regardless of owner side).
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    assert!(
        !state.powers.iter().any(|p| p.id == pid.0),
        "Oblivion removes itself at the player's turn end"
    );
}

/// GlobeHead's spawn-applied Galvanic sits before a later Oblivion in its power
/// list, and `Hook.AfterCardPlayed` walks a creature's powers in list order
/// (`CombatState.IterateHookListeners`): playing a Power card takes Galvanic's
/// 6 self-damage FIRST, then Oblivion's Doom fires Shroud. Rust ran Oblivion
/// first, so Shroud's fresh block soaked the hit (loadout Necrobinder
/// 1122-v0.107.1: C# block 3 -> 0 / HP -3 -> Shroud 3; Rust block 0, HP kept).
#[test]
fn galvanic_self_damage_lands_before_oblivion_shroud_block() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    // Spawning applies GlobeHead's Galvanic(6); Oblivion lands after it.
    let globe = state.spawn_enemy(MonsterKind::GlobeHead);
    attach_enemy_power(&mut state, globe, PowerKind::Oblivion, 4);
    push_player_power(&mut state, PowerKind::Shroud, 3);
    state.player.block = 3;

    // The played Power card is Galvanized (GalvanicPower.BeforeCombatStart).
    let card_uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: card_uid,
        card_type: CardType::Power,
        affliction: Some(sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Galvanized,
            6,
        )),
        ..Card::pod_defaults()
    });

    hooks::dispatch_before_card_played_powers(&mut state, 0, false);
    let ctx = CardCtx {
        card_uid: Some(card_uid),
        card_type: Some(CardType::Power),
        ..CardCtx::default()
    };
    hooks::dispatch_after_card_played(&mut state, &ctx);
    assert_eq!(doom_on_creature(&state, globe), 4, "Oblivion fired");
    assert_eq!(state.player.current_hp, 47, "3 of Galvanic's 6 got through");
    assert_eq!(state.player.block, 3, "Shroud's block lands after the hit");
}

/// Regression (`sts2-rs-oblivion-snapshot`): a 2nd Oblivion played while the
/// first is armed stacks the live `Amount` during its OnPlay (3 → 6), but the
/// Doom granted for THAT play must be the pre-OnPlay SNAPSHOT (3), not the
/// since-stacked live amount (6). C# `OblivionPower.BeforeCardPlayed` records
/// `base.Amount` BEFORE OnPlay; `AfterCardPlayed` grants that recorded value.
/// The bug granted Doom off the live amount (parity fuzz: a monster's Doom
/// cs=42 rs=45 — Rust over-granted by the +3 stack from the same play).
#[test]
fn oblivion_grants_pre_onplay_amount_snapshot_not_live_stacked_amount() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let pid = attach_enemy_power(&mut state, enemy, PowerKind::Oblivion, 3);

    // Before-play arm: snapshots the current Amount (3) into the counter.
    hooks::dispatch_before_card_played_powers(&mut state, /*resolved_cost=*/ 1, false);

    // A 2nd Oblivion's OnPlay stacks the live amount 3 → 6 mid-resolution
    // (before AfterCardPlayed fires for this same card play).
    state
        .powers
        .iter_mut()
        .find(|p| p.id == pid.0)
        .expect("Oblivion power present")
        .amount = 6;

    hooks::dispatch_after_card_played(&mut state, &CardCtx::default());
    assert_eq!(
        doom_on_creature(&state, enemy),
        3,
        "Oblivion grants the pre-OnPlay snapshot (3), not the since-stacked live amount (6)"
    );
}

/// The Oblivion debuff survives the *enemy's* turn end — it only clears on the
/// player's turn end (C# `side == CombatSide.Player`).
#[test]
fn oblivion_survives_enemy_turn_end() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let pid = attach_enemy_power(&mut state, enemy, PowerKind::Oblivion, 3);

    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert!(
        state.powers.iter().any(|p| p.id == pid.0),
        "Oblivion is NOT removed at the enemy's turn end"
    );
}

/// C# `ShroudPower.AfterPowerAmountChanged`: `if applier == Owner && power
/// is DoomPower { GainBlock(Owner, Amount, Unpowered) }`. When the owning
/// player applies Doom to any creature, the player gains `Amount` block.
#[test]
fn shroud_gains_block_when_player_applies_doom() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::Shroud, 5);

    // Simulate the player applying Doom(3) to the enemy.
    let ctx = PowerAppliedCtx {
        target: CreatureRef::Enemy(enemy),
        power_kind: PowerKind::Doom,
        delta: 3,
        applier: Some(CreatureRef::Player),
    };
    hooks::dispatch_after_power_applied(&mut state, &ctx);
    assert_eq!(
        state.player.block, 5,
        "Shroud(5) grants 5 block when the player applies Doom"
    );

    // A second Doom application grants another 5.
    hooks::dispatch_after_power_applied(&mut state, &ctx);
    assert_eq!(
        state.player.block, 10,
        "each player Doom application re-fires Shroud's block grant"
    );
}

/// Shroud does not fire when the applier is not the owner, nor on a
/// non-Doom power application.
#[test]
fn shroud_ignores_non_player_applier_and_non_doom() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::Shroud, 5);

    // Enemy-applied Doom: no Shroud block.
    hooks::dispatch_after_power_applied(
        &mut state,
        &PowerAppliedCtx {
            target: CreatureRef::Player,
            power_kind: PowerKind::Doom,
            delta: 3,
            applier: Some(CreatureRef::Enemy(enemy)),
        },
    );
    assert_eq!(
        state.player.block, 0,
        "enemy-applied Doom does not arm Shroud"
    );

    // Player-applied non-Doom (Vulnerable) without Vicious: no Shroud block.
    hooks::dispatch_after_power_applied(
        &mut state,
        &PowerAppliedCtx {
            target: CreatureRef::Enemy(enemy),
            power_kind: PowerKind::Vulnerable,
            delta: 2,
            applier: Some(CreatureRef::Player),
        },
    );
    assert_eq!(
        state.player.block, 0,
        "a non-Doom player application does not arm Shroud"
    );
}

/// Regression (Necrobinder fuzz seeds 192/340/495): Shroud must fire through
/// the APPLIER apply surface, not only the direct `dispatch_after_power_applied`
/// call. NEUROSURGE self-Dooms the player at turn start via
/// `Effect::ApplyPower { applier: player }` →
/// `apply_power_to_creature_with_applier`, which previously fired only the relic
/// observer channels and skipped the player reactors — so Shroud never granted
/// its block off NEUROSURGE's self-Doom (C# block 2, Rust 0). The applier path
/// now fires `dispatch_after_power_applied` with the resolved applier.
#[test]
fn shroud_fires_through_the_applier_apply_surface() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let player_uid = state.init_player_creature();
    push_player_power(&mut state, PowerKind::Shroud, 2);

    // NEUROSURGE-style self-Doom: the player applies Doom to its own mirror
    // creature, recording itself as the applier.
    card_effects::apply_power_to_creature_with_applier(
        &mut state,
        player_uid,
        PowerKind::Doom,
        3,
        Some(player_uid),
    );
    assert_eq!(
        state.player.block, 2,
        "Shroud(2) grants 2 block when the player self-Dooms via the applier path"
    );
}

/// The applier surface must NOT arm Shroud when an ENEMY is the applier
/// (e.g. a monster dooming the player), matching the `applier == Owner` gate.
#[test]
fn shroud_ignores_enemy_applier_through_the_applier_surface() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let player_uid = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::Shroud, 2);

    // Enemy applies Doom to the player via the applier surface: no Shroud block.
    card_effects::apply_power_to_creature_with_applier(
        &mut state,
        player_uid,
        PowerKind::Doom,
        3,
        Some(enemy),
    );
    assert_eq!(
        state.player.block, 0,
        "an enemy-applied Doom through the applier surface does not arm Shroud"
    );
}

/// C# `SummonNextTurnPower.AfterPlayerTurnStart`: on the owning player's
/// turn start, `OstyCmd.Summon(player, Amount, this)` then
/// `PowerCmd.Remove(this)` — summon the Osty by `Amount` and self-remove.
#[test]
fn summon_next_turn_summons_osty_then_self_removes() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let pid = push_player_power(&mut state, PowerKind::SummonNextTurn, 8);
    assert!(state.player.pet_uid.is_none(), "no Osty before turn start");

    dispatch_folded_after_player_turn_start(&mut state, &TurnSideCtx { side: Side::Player });

    let pet = state
        .player
        .pet_uid
        .expect("Osty summoned at player turn start");
    let osty = state
        .creatures
        .iter()
        .find(|c| c.id == pet.0)
        .expect("Osty creature present");
    assert_eq!(osty.max_hp, 8, "SummonNextTurn(8) summons an 8-HP Osty");
    assert!(
        !state.powers.iter().any(|p| p.id == pid.0),
        "SummonNextTurn removes itself after firing"
    );
}

/// SummonNextTurn fires only at the *player's* side turn start — the
/// enemy's turn start is a no-op (no Osty, power persists).
#[test]
fn summon_next_turn_does_not_fire_on_enemy_turn() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let pid = push_player_power(&mut state, PowerKind::SummonNextTurn, 8);

    hooks::dispatch_after_side_turn_start(&mut state, &TurnSideCtx { side: Side::Enemy });
    assert!(
        state.player.pet_uid.is_none(),
        "enemy-side turn start does not summon the player's Osty"
    );
    assert!(
        state.powers.iter().any(|p| p.id == pid.0),
        "SummonNextTurn persists through the enemy's turn start"
    );
}

/// C# `SpiritOfAshPower.BeforeCardPlayed`: `if Card.Owner == Owner.Player &&
/// Card.Keywords.Contains(Ethereal) { GainBlock(Owner, Amount, Unpowered) }`.
/// The block fires when the played card is Ethereal, independent of its cost
/// (so it fires even on a sub-2-cost card, unlike DanseMacabre).
#[test]
fn spirit_of_ash_gains_block_when_an_ethereal_card_is_played() {
    let mut state = CombatState::default();
    push_player_power(&mut state, PowerKind::SpiritOfAsh, 6);

    // Non-Ethereal card (cost 1): no block.
    hooks::dispatch_before_card_played_powers(&mut state, 1, false);
    assert_eq!(
        state.player.block, 0,
        "a non-Ethereal play does not arm SpiritOfAsh"
    );

    // Ethereal card at cost 1 (< DanseMacabre's threshold): still +6 block.
    hooks::dispatch_before_card_played_powers(&mut state, 1, true);
    assert_eq!(
        state.player.block, 6,
        "SpiritOfAsh(6) grants 6 block on an Ethereal play regardless of cost"
    );

    // A second Ethereal play stacks another 6.
    hooks::dispatch_before_card_played_powers(&mut state, 1, true);
    assert_eq!(
        state.player.block, 12,
        "SpiritOfAsh re-fires per Ethereal play"
    );
}

/// Build an Ethereal card POD (uid 1) and push it into the arena.
fn push_ethereal_card(state: &mut CombatState) -> CardUid {
    let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero card uid");
    let card = Card {
        uid,
        keyword_override: Some(KeywordOverride::ETHEREAL),
        ..Card::pod_defaults()
    };
    state.cards.push(card);
    uid
}

/// C# `VeilpiercerPower.TryModifyEnergyCostInCombat`: an owned Ethereal card
/// costs 0 while Veilpiercer has charges; a non-Ethereal card is unaffected.
#[test]
fn veilpiercer_makes_owned_ethereal_card_cost_zero() {
    let mut state = CombatState::default();
    let eth = push_ethereal_card(&mut state);
    // Veilpiercer frees only a Hand/Play card (VeilpiercerPower.cs).
    state.player.hand.push(eth);
    push_player_power(&mut state, PowerKind::Veilpiercer, 2);

    let eth_ctx = CardCostCtx {
        owner: CreatureRef::Player,
        card_id: Some(CardId::StrikeIronclad),
        card_uid: Some(eth),
        card_type: Some(CardType::Attack),
        card_tags: CardTags::NONE,
        base_cost: 3,
    };
    assert_eq!(
        hooks::dispatch_modify_card_cost(&state, &eth_ctx),
        0,
        "Veilpiercer drops an owned Ethereal card to cost 0"
    );

    // A non-Ethereal card (no keyword override → not Ethereal by id) keeps its
    // base cost.
    let plain_uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: plain_uid,
        ..Card::pod_defaults()
    });
    let plain_ctx = CardCostCtx {
        card_uid: Some(plain_uid),
        ..eth_ctx
    };
    assert_eq!(
        hooks::dispatch_modify_card_cost(&state, &plain_ctx),
        3,
        "Veilpiercer does not discount a non-Ethereal card"
    );
}

/// C# `VeilpiercerPower.TryModifyEnergyCostInCombatLate`
/// (v0.111.0 `VeilpiercerPower.cs:18-44`) gates on `card.Keywords`, and
/// `HexPower.TryModifyKeywordsInCombat` (`HexPower.cs`) adds Ethereal to every
/// Hexed card. Under Hex + Veilpiercer a non-Ethereal card is therefore free.
/// The same holds for a card made Ethereal for the combat (`EtherealThisCombat`).
#[test]
fn veilpiercer_discounts_cards_made_ethereal_by_hex_or_this_combat() {
    let mut state = CombatState::default();
    push_player_power(&mut state, PowerKind::Veilpiercer, 1);
    let plain_uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: plain_uid,
        ..Card::pod_defaults()
    });
    state.player.hand.push(plain_uid);
    let ctx = CardCostCtx {
        owner: CreatureRef::Player,
        card_id: Some(CardId::StrikeIronclad),
        card_uid: Some(plain_uid),
        card_type: Some(CardType::Attack),
        card_tags: CardTags::NONE,
        base_cost: 2,
    };
    assert_eq!(hooks::dispatch_modify_card_cost(&state, &ctx), 2);

    // EtherealThisCombat counts (it shows up in C# `card.Keywords`).
    state
        .cards
        .iter_mut()
        .find(|c| c.uid == plain_uid)
        .unwrap()
        .set_combat_var(sts2_core::cards::VarKey::EtherealThisCombat, 1);
    assert_eq!(
        hooks::dispatch_modify_card_cost(&state, &ctx),
        0,
        "EtherealThisCombat card is free under Veilpiercer"
    );
    state
        .cards
        .iter_mut()
        .find(|c| c.uid == plain_uid)
        .unwrap()
        .set_combat_var(sts2_core::cards::VarKey::EtherealThisCombat, 0);

    // Hex makes every Hexed card Ethereal → free.
    push_player_power(&mut state, PowerKind::Hex, 2);
    state.cards[plain_uid.get() as usize - 1].affliction =
        Some(sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Hexed,
            2,
        ));
    assert_eq!(
        hooks::dispatch_modify_card_cost(&state, &ctx),
        0,
        "Hexed card is Ethereal, so Veilpiercer makes it cost 0"
    );
}

/// C# `VeilpiercerPower.BeforeCardPlayed`: each Ethereal play consumes one
/// charge; the power is removed when it reaches 0.
#[test]
fn veilpiercer_decrements_on_ethereal_play_and_removes_at_zero() {
    let mut state = CombatState::default();
    let pid = push_player_power(&mut state, PowerKind::Veilpiercer, 2);

    // First Ethereal play: 2 → 1 (still present).
    hooks::dispatch_before_card_played_powers(&mut state, 1, true);
    let amt = state
        .powers
        .iter()
        .find(|p| p.id == pid.0)
        .map(|p| p.amount);
    assert_eq!(amt, Some(1), "first Ethereal play consumes one charge");

    // A non-Ethereal play does not consume a charge.
    hooks::dispatch_before_card_played_powers(&mut state, 1, false);
    assert_eq!(
        state
            .powers
            .iter()
            .find(|p| p.id == pid.0)
            .map(|p| p.amount),
        Some(1),
        "a non-Ethereal play leaves Veilpiercer untouched"
    );

    // Second Ethereal play: 1 → 0 → removed.
    hooks::dispatch_before_card_played_powers(&mut state, 1, true);
    assert!(
        !state.powers.iter().any(|p| p.id == pid.0),
        "Veilpiercer is removed when its last charge is consumed"
    );
}

/// C# `DanseMacabrePower.BeforeCardPlayed` reads `EnergyCost.GetResolved()`
/// live. A Veilpiercer listed BEFORE it that this Ethereal play strips of its
/// last charge is already gone, so DanseMacabre sees the card's real cost 3
/// (>= 2) and grants block, though the play itself was free (loadout
/// Necrobinder 1309-v0.107.1: REAP, C# block 6, Rust 0).
#[test]
fn danse_macabre_reads_live_cost_after_veilpiercer_runs_out() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let eth = push_ethereal_card(&mut state);
    // Veilpiercer frees only a Hand/Play card (VeilpiercerPower.cs).
    state.player.hand.push(eth);
    state.cards[0].cost = 3;
    push_player_power(&mut state, PowerKind::Veilpiercer, 1);
    push_player_power(&mut state, PowerKind::DanseMacabre, 6);
    assert_eq!(
        hooks::effective_card_cost(&state, eth),
        0,
        "free under Veilpiercer"
    );

    hooks::dispatch_before_card_played_powers_for_card(&mut state, 0, true, Some(eth));
    assert!(
        !state
            .powers
            .iter()
            .any(|p| p.kind == PowerKind::Veilpiercer),
        "the last charge is consumed"
    );
    assert_eq!(
        state.player.block, 6,
        "DanseMacabre sees the restored cost 3"
    );
}

/// The mirror order: a DanseMacabre listed BEFORE the Veilpiercer runs while
/// the card still costs 0, so it grants nothing.
#[test]
fn danse_macabre_before_veilpiercer_sees_the_free_cost() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let eth = push_ethereal_card(&mut state);
    // Veilpiercer frees only a Hand/Play card (VeilpiercerPower.cs).
    state.player.hand.push(eth);
    state.cards[0].cost = 3;
    push_player_power(&mut state, PowerKind::DanseMacabre, 6);
    push_player_power(&mut state, PowerKind::Veilpiercer, 1);

    hooks::dispatch_before_card_played_powers_for_card(&mut state, 0, true, Some(eth));
    assert!(!state
        .powers
        .iter()
        .any(|p| p.kind == PowerKind::Veilpiercer));
    assert_eq!(
        state.player.block, 0,
        "DanseMacabre ran while the card was free"
    );
}

/// GH #1150: a REPLAY of the play (extra plays / Echo) re-reads the cost live.
/// The first iteration spent the last Veilpiercer charge with DanseMacabre
/// listed first (so it saw 0); the replay sees the restored cost 3.
#[test]
fn danse_macabre_on_a_replay_reads_the_cost_after_veilpiercer_ran_out() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let eth = push_ethereal_card(&mut state);
    state.cards[0].cost = 3;
    push_player_power(&mut state, PowerKind::DanseMacabre, 6);
    push_player_power(&mut state, PowerKind::Veilpiercer, 1);

    hooks::dispatch_before_card_played_powers_for_card(&mut state, 0, true, Some(eth));
    assert_eq!(state.player.block, 0, "first play: still free");
    hooks::dispatch_before_card_played_powers_on_replay(&mut state, 0, true, eth);
    assert_eq!(state.player.block, 6, "replay: live cost 3 >= 2");
}

/// A replay while Veilpiercer still holds the card at 0 grants nothing.
#[test]
fn danse_macabre_on_a_replay_under_live_veilpiercer_sees_the_free_cost() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let eth = push_ethereal_card(&mut state);
    state.cards[0].cost = 3;
    push_player_power(&mut state, PowerKind::DanseMacabre, 6);
    push_player_power(&mut state, PowerKind::Veilpiercer, 3);

    hooks::dispatch_before_card_played_powers_for_card(&mut state, 0, true, Some(eth));
    hooks::dispatch_before_card_played_powers_on_replay(&mut state, 0, true, eth);
    assert_eq!(state.player.block, 0);
}

/// Push an Ethereal card POD and append its uid to the draw pile (front-drawn).
fn push_ethereal_into_draw(state: &mut CombatState) -> CardUid {
    let uid = push_ethereal_card(state);
    state.player.draw.push(uid);
    uid
}

/// C# `PagestormPower.AfterCardDrawn`: drawing an owned Ethereal card draws
/// `Amount` more. The extra draws recurse through the on-draw surface, so a
/// pile of Ethereal cards cascades: drawing one pulls the whole chain.
#[test]
fn pagestorm_cascades_ethereal_draws() {
    let mut state = CombatState::default();
    // A live player: a dead one makes the combat ending, so draws move no card.
    state.player.current_hp = 80;
    push_player_power(&mut state, PowerKind::Pagestorm, 1);
    // Three Ethereal cards in the draw pile (uids 1..=3, front-drawn).
    for _ in 0..3 {
        push_ethereal_into_draw(&mut state);
    }

    let moved = state.draw_into_hand(1);
    // Drew 1 directly; Pagestorm(1) fires per Ethereal, recursively pulling the
    // remaining two until the pile is empty.
    assert_eq!(
        state.player.hand.len(),
        3,
        "Pagestorm cascades all 3 Ethereal"
    );
    assert_eq!(moved, 1, "draw_into_hand reports only its own direct draw");
}

/// C# `CardPileCmd.Draw` runs `Hook.AfterCardDrawn` per card, so
/// `PagestormPower`'s nested `Draw` (fromHandDraw = false) lands BEFORE the
/// outer hand draw's next card. With 8 cards in hand, a 2-card hand draw that
/// opens on an Ethereal card fills the hand with the Pagestorm draw — the
/// outer second draw then fails on the 10-card cap. One card was drawn by
/// effect (DeathMarch's count); the batch path drew both hand cards first and
/// the Pagestorm draw failed instead (loadout Necrobinder 1001/1012).
#[test]
fn pagestorm_draw_interleaves_before_the_next_hand_draw() {
    let mut state = CombatState::default();
    state.player.current_hp = 80;
    push_player_power(&mut state, PowerKind::Pagestorm, 1);
    for _ in 0..8 {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.hand.push(uid);
    }
    let eth = push_ethereal_into_draw(&mut state);
    let mut plain = Vec::new();
    for _ in 0..2 {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
        plain.push(uid);
    }

    let moved = state.draw_into_hand(2);
    assert_eq!(moved, 1, "the outer second draw hits the hand cap");
    assert_eq!(state.player.hand.len(), 10);
    assert_eq!(&state.player.hand[8..], &[eth, plain[0]]);
    assert_eq!(
        state.combat_log.cards_drawn_by_effect_this_turn, 1,
        "the Pagestorm draw (fromHandDraw = false) succeeded"
    );
}

/// Without Pagestorm, drawing one Ethereal card draws exactly one — no rider.
#[test]
fn no_pagestorm_means_no_extra_draw() {
    let mut state = CombatState::default();
    // A live player: a dead one makes the combat ending, so draws move no card.
    state.player.current_hp = 80;
    for _ in 0..3 {
        push_ethereal_into_draw(&mut state);
    }
    state.draw_into_hand(1);
    assert_eq!(
        state.player.hand.len(),
        1,
        "no Pagestorm power → drawing an Ethereal draws nothing extra"
    );
}

/// Pagestorm only reacts to Ethereal draws — a non-Ethereal card draws nothing
/// extra even while the power is active.
#[test]
fn pagestorm_ignores_non_ethereal_draws() {
    let mut state = CombatState::default();
    // A live player: a dead one makes the combat ending, so draws move no card.
    state.player.current_hp = 80;
    push_player_power(&mut state, PowerKind::Pagestorm, 1);
    // Three plain (non-Ethereal) cards in the draw pile.
    for _ in 0..3 {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
    }
    state.draw_into_hand(1);
    assert_eq!(
        state.player.hand.len(),
        1,
        "Pagestorm does not fire on a non-Ethereal draw"
    );
}

/// C# `HangPower.ModifyDamageMultiplicative`: damage the Hang card deals to a
/// Hang-afflicted target is multiplied by the existing Hang stacks; non-Hang
/// sources (and absent Hang) leave the damage unscaled.
#[test]
fn hang_card_multiplies_its_damage_on_a_hang_afflicted_target() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    // Target already carries Hang(3) from a prior play.
    let hang_pid = attach_enemy_power(&mut state, enemy, PowerKind::Hang, 3);

    // Put a Hang card on top of the play pile — the mid-resolution source the
    // listener reads (mirrors C# `cardSource`).
    let cuid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: cuid,
        card_id: CardId::Hang,
        ..Card::pod_defaults()
    });
    state.player.play.push(cuid);

    let ctx = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    assert_eq!(
        hooks::modify_damage_multiplicative(&state, ListenerRef::Power(hang_pid), &ctx),
        DamageMul::new(3, 1),
        "Hang card damage on a Hang(3) target is ×3"
    );

    // An Unpowered reaction while Hang resolves (SleightOfFlesh on the Hang
    // apply, Panache's AoE) has no Hang `cardSource`: unscaled (seeds
    // 2203/3154).
    let reaction = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::UNPOWERED,
    };
    assert_eq!(
        hooks::modify_damage_multiplicative(&state, ListenerRef::Power(hang_pid), &reaction),
        DamageMul::ONE,
        "an Unpowered reaction mid-Hang is not scaled"
    );

    // With a non-Hang card on top of play (e.g. another attack), no multiplier.
    state.player.play.clear();
    let other = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: other,
        card_id: CardId::StrikeIronclad,
        ..Card::pod_defaults()
    });
    state.player.play.push(other);
    assert_eq!(
        hooks::modify_damage_multiplicative(&state, ListenerRef::Power(hang_pid), &ctx),
        DamageMul::ONE,
        "non-Hang source does not scale on a Hang target"
    );
}

/// C# `ReaperFormPower.AfterDamageGiven`: a powered attack by the player (or
/// the player's Osty) applies Doom = `TotalDamage * Amount` to the target,
/// gating on TotalDamage (blocked + unblocked), not the unblocked residual.
#[test]
fn reaper_form_dooms_target_by_total_damage_times_amount() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::ReaperForm, 2);

    let ctx = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    // total_damage = 10, of which 6 got through block.
    hooks::dispatch_after_damage_given(&mut state, &ctx, 6, 10, false);
    assert_eq!(
        doom_on_creature(&state, enemy),
        20,
        "ReaperForm(2) applies Doom = TotalDamage(10) * 2"
    );
}

/// A ReaperForm hit that KILLS its target still records the player's Doom:
/// C# `AfterDamageGiven` runs before `Kill` (CreatureCmd.cs:390-409) and
/// `CanReceivePowers` is not an alive check, so the `PowerReceivedEntry`
/// that arms Death's Door is written (Necrobinder full seed 2430).
#[test]
fn reaper_form_killing_blow_still_arms_deaths_door() {
    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let _keep_alive = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::ReaperForm, 1);
    state.combat_log.doom_applied_by_player_this_turn = false;
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.current_hp = 0;
    }

    let ctx = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    hooks::dispatch_after_damage_given(&mut state, &ctx, 11, 11, false);
    assert!(state.combat_log.doom_applied_by_player_this_turn);
    assert_eq!(
        doom_on_creature(&state, enemy),
        0,
        "the death strip removes the Doom again (seed 959: Illusion corpse)"
    );
}

/// With an Osty summoned, its `DieForYou` answers `ShouldAllowHitting` with
/// `creature.IsAlive` for EVERY creature (`DieForYouPower.cs:35`), so a corpse
/// cannot receive powers (`Creature.CanReceivePowers`, `PowerCmd.Apply`):
/// Reaper Form's killing-blow Doom never lands, Shroud grants no block and
/// Death's Door is not armed (loadout Necrobinder 1073/1078/1092/1114/1122/
/// 1035/1109: Rust gained Shroud's block on the kill, C# did not).
#[test]
fn reaper_form_killing_blow_is_dropped_while_die_for_you_is_live() {
    let run = |die_for_you: bool| {
        let mut state = CombatState::default();
        state.player.current_hp = 50;
        state.player.max_hp = 50;
        let _ = state.init_player_creature();
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        let osty = state.spawn_enemy(MonsterKind::Nibbit);
        if die_for_you {
            attach_enemy_power(&mut state, osty, PowerKind::DieForYou, 1);
        }
        push_player_power(&mut state, PowerKind::ReaperForm, 1);
        push_player_power(&mut state, PowerKind::Shroud, 3);
        state.combat_log.doom_applied_by_player_this_turn = false;
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.current_hp = 0;
        }
        let ctx = DamageCtx {
            dealer: CreatureRef::Player,
            target: CreatureRef::Enemy(enemy),
            props: ValueProps::MOVE,
        };
        hooks::dispatch_after_damage_given(&mut state, &ctx, 11, 11, false);
        (
            state.player.block,
            state.combat_log.doom_applied_by_player_this_turn,
        )
    };
    assert_eq!(run(false), (3, true), "no Osty: the corpse takes the Doom");
    assert_eq!(
        run(true),
        (0, false),
        "DieForYou live: no Doom on the corpse, so no Shroud block"
    );
}

/// ReaperForm gates on TotalDamage, so a fully-blocked powered hit (unblocked
/// = 0) still doom-marks; an unpowered hit does not fire at all.
#[test]
fn reaper_form_fires_on_fully_blocked_but_not_unpowered() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    push_player_power(&mut state, PowerKind::ReaperForm, 3);

    let powered = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    // Fully blocked: unblocked = 0 but TotalDamage = 4 → still dooms (4*3).
    hooks::dispatch_after_damage_given(&mut state, &powered, 0, 4, true);
    assert_eq!(
        doom_on_creature(&state, enemy),
        12,
        "a fully-blocked powered hit still triggers ReaperForm (gates on TotalDamage)"
    );

    // Unpowered hit (e.g. reflected/HP-loss style): no ReaperForm.
    let unpowered = DamageCtx {
        dealer: CreatureRef::Player,
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::UNPOWERED,
    };
    hooks::dispatch_after_damage_given(&mut state, &unpowered, 5, 5, false);
    assert_eq!(
        doom_on_creature(&state, enemy),
        12,
        "an unpowered hit does not trigger ReaperForm"
    );
}

/// A minimal card factory for the SentryMode test: returns a POD card carrying
/// the requested id (SweepingGaze) so the core lane can materialize the token
/// without the content crate's real factory.
// `Option<Card>` is mandated by the `CardFactoryFn` pointer type this is
// registered as — the wrap is structural, not removable.
#[allow(clippy::unnecessary_wraps)]
fn stub_card_factory(id: CardId, _upgraded: bool) -> Option<Card> {
    Some(Card {
        card_id: id,
        ..Card::pod_defaults()
    })
}

/// C# `SentryModePower.BeforeHandDraw`: for the owning player, `Amount` times
/// create a SweepingGaze and add it to hand.
#[test]
fn sentry_mode_adds_sweeping_gaze_cards_to_hand() {
    sts2_core::register_card_factory(stub_card_factory);
    let mut state = CombatState::default();
    push_player_power(&mut state, PowerKind::SentryMode, 2);
    assert_eq!(state.player.hand.len(), 0, "hand empty before SentryMode");

    hooks::dispatch_before_hand_draw_powers(&mut state);

    assert_eq!(
        state.player.hand.len(),
        2,
        "SentryMode(2) adds 2 SweepingGaze cards to hand"
    );
    for &uid in &state.player.hand {
        let card = state
            .cards
            .iter()
            .find(|c| c.uid == uid)
            .expect("card present");
        assert_eq!(
            card.card_id,
            CardId::SweepingGaze,
            "SentryMode generates SweepingGaze tokens"
        );
    }
}

/// Current HP of the enemy creature `uid`.
fn enemy_hp(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("enemy present")
        .current_hp
}

/// Regression (`sts2-rs-deaths-door-doom`): `apply_power_to_enemy` arms
/// `doom_applied_by_player_this_turn` (DeathsDoor's `WasDoomAppliedThisTurn`)
/// ONLY when the Doom actually lands. C# scans for a real `PowerReceivedEntry`,
/// which a Doom applied to an already-dead target (e.g. BlightStrike's
/// damage-equal Doom on the killing hit) never produces — so the flag must stay
/// false (parity fuzz: player.block cs=11 rs=23, Rust wrongly armed the 3x off a
/// kill-shot Doom).
#[test]
fn deaths_door_flag_only_arms_on_a_landed_player_doom() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);

    // Doom applied to a DEAD enemy records no PowerReceivedEntry → no arm.
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.current_hp = 0;
    }
    card_effects::apply_power_to_enemy(&mut state, enemy, PowerKind::Doom, 3);
    assert!(
        !state.combat_log.doom_applied_by_player_this_turn,
        "Doom on an already-dead enemy does not arm DeathsDoor"
    );

    // A zero-stack Doom apply records nothing either.
    let live = state.spawn_enemy(MonsterKind::Nibbit);
    card_effects::apply_power_to_enemy(&mut state, live, PowerKind::Doom, 0);
    assert!(
        !state.combat_log.doom_applied_by_player_this_turn,
        "a 0-stack Doom apply does not arm DeathsDoor"
    );

    // A real landed Doom on a live enemy arms the flag.
    card_effects::apply_power_to_enemy(&mut state, live, PowerKind::Doom, 3);
    assert!(
        state.combat_log.doom_applied_by_player_this_turn,
        "a Doom that lands on a live enemy arms DeathsDoor"
    );
}

/// C# `SleightOfFleshPower.AfterPowerAmountChanged`: when the player applies a
/// non-temporary debuff to an enemy, deal `Amount` Unpowered damage to it.
#[test]
fn sleight_of_flesh_damages_enemy_when_player_applies_a_debuff() {
    let mut state = CombatState::default();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let hp0 = enemy_hp(&state, enemy);
    push_player_power(&mut state, PowerKind::SleightOfFlesh, 4);

    // Player applies Doom (a debuff) to the enemy.
    hooks::dispatch_after_power_applied(
        &mut state,
        &PowerAppliedCtx {
            target: CreatureRef::Enemy(enemy),
            power_kind: PowerKind::Doom,
            delta: 3,
            applier: Some(CreatureRef::Player),
        },
    );
    assert_eq!(
        enemy_hp(&state, enemy),
        hp0 - 4,
        "SleightOfFlesh(4) deals 4 Unpowered damage to the debuffed enemy"
    );
}

/// SleightOfFlesh does not fire on a buff applied to an enemy, nor on a debuff
/// applied to the player.
#[test]
fn sleight_of_flesh_ignores_buffs_and_self_debuffs() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let hp0 = enemy_hp(&state, enemy);
    push_player_power(&mut state, PowerKind::SleightOfFlesh, 4);

    // Applying a buff (Strength) to the enemy: not a debuff → no damage.
    hooks::dispatch_after_power_applied(
        &mut state,
        &PowerAppliedCtx {
            target: CreatureRef::Enemy(enemy),
            power_kind: PowerKind::Strength,
            delta: 2,
            applier: Some(CreatureRef::Player),
        },
    );
    assert_eq!(
        enemy_hp(&state, enemy),
        hp0,
        "a buff application does not trigger SleightOfFlesh"
    );

    // Applying a debuff to the PLAYER (not an enemy): no SleightOfFlesh damage.
    let php0 = state.player.current_hp;
    hooks::dispatch_after_power_applied(
        &mut state,
        &PowerAppliedCtx {
            target: CreatureRef::Player,
            power_kind: PowerKind::Vulnerable,
            delta: 2,
            applier: Some(CreatureRef::Player),
        },
    );
    assert_eq!(
        enemy_hp(&state, enemy),
        hp0,
        "enemy untouched by a self-debuff"
    );
    assert_eq!(
        state.player.current_hp, php0,
        "SleightOfFlesh deals no damage when the debuff target is the player"
    );
}

/// C# `DoubleDamagePower.ModifyDamageMultiplicative` gates the dealer on
/// `dealer == Owner || Owner.Pets.Contains(dealer)`: a card-sourced powered
/// Osty hit doubles too, not just the player's own (loadout Necrobinder 1019).
#[test]
fn double_damage_doubles_the_owners_osty_hits() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let pet = state.summon_osty(8).expect("Osty summoned");
    let dd = push_player_power(&mut state, PowerKind::DoubleDamage, 1);
    // A card mid-resolution stands in for C#'s non-null `cardSource`.
    let cuid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
    state.cards.push(Card {
        uid: cuid,
        ..Card::pod_defaults()
    });
    state.player.play.push(cuid);

    let osty_hit = DamageCtx {
        dealer: CreatureRef::Enemy(pet),
        target: CreatureRef::Enemy(enemy),
        props: ValueProps::MOVE,
    };
    assert_eq!(
        hooks::modify_damage_multiplicative(&state, ListenerRef::Power(dd), &osty_hit),
        DamageMul::DOUBLE,
        "the owner's pet's powered card hit is doubled"
    );
    let enemy_hit = DamageCtx {
        dealer: CreatureRef::Enemy(enemy),
        target: CreatureRef::Player,
        props: ValueProps::MOVE,
    };
    assert_eq!(
        hooks::modify_damage_multiplicative(&state, ListenerRef::Power(dd), &enemy_hit),
        DamageMul::ONE,
        "a non-pet enemy's hit is not doubled"
    );
}

/// Regression (Necrobinder fuzz seed 443): a DIRECT Osty kill via
/// [`CombatState::kill_osty`] (SACRIFICE / BONE_SHARDS) must fire
/// `Hook.AfterCurrentHpChanged(Osty, -currentHp)` just like the damage-redirect
/// path, so `NecroMasteryPower` reflects `lostHp * Amount` Unblockable|Unpowered
/// damage to every hittable enemy. C# `CreatureCmd.KillWithoutCheckingWinCondition`
/// (CreatureCmd.cs:349-353) fires this hook after `LoseHpInternal`; the prior
/// Rust `kill_osty` zeroed HP silently, so SACRIFICE dealt 0 reflected damage
/// while C# dealt 8 (monsters[0].hp +8 in Rust).
#[test]
fn killing_osty_directly_reflects_its_hp_via_necro_mastery() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let hp0 = enemy_hp(&state, enemy);

    // Summon an 8-HP Osty and put NecroMastery(1) on the player.
    let pet = state.summon_osty(8).expect("Osty summoned");
    assert_eq!(state.player.pet_uid, Some(pet), "pet uid tracks the Osty");
    push_player_power(&mut state, PowerKind::NecroMastery, 1);

    // SACRIFICE-style direct kill.
    state.kill_osty();

    assert_eq!(
        enemy_hp(&state, enemy),
        hp0 - 8,
        "killing the 8-HP Osty reflects 8 (=lostHp*1) Unblockable damage to the enemy"
    );
    assert!(
        state
            .creatures
            .iter()
            .find(|c| c.id == pet.0)
            .is_some_and(|c| c.current_hp == 0),
        "the Osty is left as a 0-HP corpse"
    );
}

/// NecroMastery scales the reflected damage by its stack amount.
#[test]
fn killing_osty_reflection_scales_with_necro_mastery_amount() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let hp0 = enemy_hp(&state, enemy);

    let _pet = state.summon_osty(5).expect("Osty summoned");
    push_player_power(&mut state, PowerKind::NecroMastery, 3);

    state.kill_osty();

    assert_eq!(
        enemy_hp(&state, enemy),
        hp0 - 15,
        "NecroMastery(3) reflects 5*3 = 15 damage when the 5-HP Osty is killed"
    );
}

/// Without NecroMastery the direct Osty kill reflects nothing.
#[test]
fn killing_osty_without_necro_mastery_reflects_nothing() {
    let mut state = CombatState::default();
    state.player.current_hp = 60;
    state.player.max_hp = 60;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let hp0 = enemy_hp(&state, enemy);

    let _pet = state.summon_osty(8).expect("Osty summoned");
    state.kill_osty();

    assert_eq!(
        enemy_hp(&state, enemy),
        hp0,
        "no NecroMastery means a direct Osty kill deals no reflected damage"
    );
}

/// Draw pile `[A, B, Ethereal, V, C]` drawn 4 by the hand draw under
/// Pagestorm(1) + ChainsOfBinding(3), the powers applied in `order`. Returns
/// which of `[A, B, Ethereal, V, C]` ended up `Bound`.
fn bound_after_pagestorm_chains_draw(order: [PowerKind; 2]) -> [bool; 5] {
    let mut state = CombatState::default();
    state.player.current_hp = 80;
    push_player_power(
        &mut state,
        order[0],
        if order[0] == PowerKind::Pagestorm {
            1
        } else {
            3
        },
    );
    push_player_power(
        &mut state,
        order[1],
        if order[1] == PowerKind::Pagestorm {
            1
        } else {
            3
        },
    );
    let mut plain = || {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
        uid
    };
    let (a, b) = (plain(), plain());
    let eth = push_ethereal_into_draw(&mut state);
    let mut plain = || {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
        uid
    };
    let (v, c) = (plain(), plain());

    state.draw_into_hand(4);
    assert_eq!(state.player.hand.as_slice(), &[a, b, eth, v, c]);
    [a, b, eth, v, c].map(|uid| {
        state.cards[uid.get() as usize - 1]
            .affliction
            .is_some_and(|af| af.kind == sts2_core::card_afflictions::CardAfflictionKind::Bound)
    })
}

/// C# `Hook.AfterCardDrawn` awaits each listener in power-list (application)
/// order. With Pagestorm applied BEFORE Chains of Binding, Pagestorm's nested
/// draw (PagestormPower.cs:19-26) runs to completion — the drawn V takes the
/// third `Bound` — before `ChainsOfBindingPower.AfterCardDrawn` sees the
/// Ethereal card, whose history count is then at the cap
/// (ChainsOfBindingPower.cs:37-47). Rust bound the Ethereal card and left V
/// free, so a later Bound play was vetoed in Rust only (loadout Necrobinder
/// v0.107.1 1244: Defile vs Volley).
#[test]
fn pagestorm_applied_before_chains_binds_its_nested_draw_first() {
    assert_eq!(
        bound_after_pagestorm_chains_draw([PowerKind::Pagestorm, PowerKind::ChainsOfBinding]),
        [true, true, false, true, false],
    );
}

/// GH #1150: Pagestorm (applied first) draws on the Ethereal card, the draw
/// reshuffles into a Stratagem pick and suspends. C# `Hook.AfterCardDrawn`
/// awaits Pagestorm's whole draw, pick included, before Chains of Binding sees
/// the Ethereal card: the resumed draw's Y takes the one `Bound`, and the
/// Ethereal card finds the cap reached. Rust checked the Ethereal card at the
/// suspension.
#[test]
fn chains_waits_for_a_pagestorm_draw_suspended_on_a_stratagem_pick() {
    let mut state = CombatState::default();
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    push_player_power(&mut state, PowerKind::Pagestorm, 1);
    push_player_power(&mut state, PowerKind::ChainsOfBinding, 1);
    push_player_power(&mut state, PowerKind::Stratagem, 1);
    let eth = push_ethereal_into_draw(&mut state);
    let mut plain = || {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        uid
    };
    let (x, y) = (plain(), plain());
    state.player.discard.extend([x, y]);
    let bound = |state: &CombatState, uid: CardUid| {
        state.cards[uid.get() as usize - 1]
            .affliction
            .is_some_and(|af| af.kind == sts2_core::card_afflictions::CardAfflictionKind::Bound)
    };

    state.draw_into_hand(1);
    assert!(state.pending_choice.is_some(), "Stratagem pick open");
    assert!(!bound(&state, eth), "Chains waits for Pagestorm's draw");

    let mut sel = smallvec::SmallVec::<[u8; 4]>::new();
    sel.push(0);
    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice { selection: sel },
    );
    assert!(state.pending_choice.is_none());
    assert_eq!(state.player.hand.len(), 3, "Ethereal + pick + resumed draw");
    let drawn = *state.player.hand.last().expect("resumed draw");
    assert!(bound(&state, drawn), "the resumed draw takes the Bound");
    assert!(
        !bound(&state, eth),
        "the Ethereal card finds the cap reached"
    );
}

/// The other application order: Chains listens first, so the Ethereal card is
/// Bound before Pagestorm draws, and the nested draw finds the cap reached.
#[test]
fn chains_applied_before_pagestorm_binds_the_ethereal_card() {
    assert_eq!(
        bound_after_pagestorm_chains_draw([PowerKind::ChainsOfBinding, PowerKind::Pagestorm]),
        [true, true, true, false, false],
    );
}

/// Application order is the power's id, not where its uid sits in the
/// player's lists: a monster-applied Chains of Binding (the Queen's) can sit
/// after a Pagestorm the player applied later (loadout Necrobinder v0.107.1
/// 1019). Chains (id 1) listed after Pagestorm (id 2) still goes first.
#[test]
fn chains_applied_first_goes_first_wherever_its_uid_is_listed() {
    let mut state = CombatState::default();
    state.player.current_hp = 80;
    for (id, kind, amount) in [
        (1, PowerKind::ChainsOfBinding, 3),
        (2, PowerKind::Pagestorm, 1),
    ] {
        state.powers.push(Power {
            id: std::num::NonZeroU32::new(id).expect("nz"),
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
    }
    // Listed Pagestorm first.
    state.player.powers.push(PowerUid::new(2).expect("nz"));
    state.player.powers.push(PowerUid::new(1).expect("nz"));
    let plain = |state: &mut CombatState| {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
        uid
    };
    let (a, b) = (plain(&mut state), plain(&mut state));
    let eth = push_ethereal_into_draw(&mut state);
    let (v, c) = (plain(&mut state), plain(&mut state));

    state.draw_into_hand(4);
    assert_eq!(state.player.hand.as_slice(), &[a, b, eth, v, c]);
    let bound = [a, b, eth, v, c].map(|uid| {
        state.cards[uid.get() as usize - 1]
            .affliction
            .is_some_and(|af| af.kind == sts2_core::card_afflictions::CardAfflictionKind::Bound)
    });
    assert_eq!(bound, [true, true, true, false, false]);
}

/// Draw pile `[A, Status, X, Y]` drawn 2 by the hand draw under Iteration(2) +
/// ChainsOfBinding(2), the powers applied in `order`. Returns which of
/// `[A, Status, X, Y]` ended up `Bound`.
fn bound_after_iteration_chains_draw(order: [PowerKind; 2]) -> [bool; 4] {
    let mut state = CombatState::default();
    state.player.current_hp = 80;
    for kind in order {
        push_player_power(&mut state, kind, 2);
    }
    let mut card = |card_type: CardType| {
        let uid = CardUid::new(state.cards.len() as u32 + 1).expect("nonzero");
        state.cards.push(Card {
            uid,
            card_type,
            ..Card::pod_defaults()
        });
        state.player.draw.push(uid);
        uid
    };
    let a = card(CardType::Attack);
    let status = card(CardType::Status);
    let (x, y) = (card(CardType::Attack), card(CardType::Attack));

    state.draw_into_hand(2);
    assert_eq!(state.player.hand.as_slice(), &[a, status, x, y]);
    [a, status, x, y].map(|uid| {
        state.cards[uid.get() as usize - 1]
            .affliction
            .is_some_and(|af| af.kind == sts2_core::card_afflictions::CardAfflictionKind::Bound)
    })
}

/// Iteration applied before Chains of Binding: its nested draw on the turn's
/// first Status card (IterationPower.cs:18-29) runs to completion before
/// `ChainsOfBindingPower.AfterCardDrawn` checks the Status card, so X takes
/// the second `Bound` and the Status card finds the cap reached (Defect
/// v0.107.1 5533 @595: C# bound the Void Iteration drew, Rust the Dazed).
#[test]
fn iteration_applied_before_chains_binds_its_nested_draw_first() {
    assert_eq!(
        bound_after_iteration_chains_draw([PowerKind::Iteration, PowerKind::ChainsOfBinding]),
        [true, false, true, false],
    );
}

/// The other order: Chains binds the Status card before Iteration draws.
#[test]
fn chains_applied_before_iteration_binds_the_status_card() {
    assert_eq!(
        bound_after_iteration_chains_draw([PowerKind::ChainsOfBinding, PowerKind::Iteration]),
        [true, true, false, false],
    );
}
