//! Curse + status combat_loop dispatch verification — bead
//! `sts2-rs-6ynt.3`.
//!
//! This file pins the *current* silent-no-op behaviour for every
//! end-of-turn / on-draw / on-play trigger documented in the audit at
//! `docs/audits/2026-05-15-ironclad-completeness/02-colorless-status-curse.md`.
//!
//! **Why "pin the broken behaviour" rather than parity fixtures?**
//! Bead `sts2-rs-6ynt.3` verified that none of these triggers actually
//! fire in `combat_loop`. The follow-up beads `sts2-rs-6ynt.3.{1..4}`
//! track the fixes. Until those land, these tests document the gap so
//! a future implementer cannot quietly land a partial fix without
//! flipping these expectations — every test references the bead that
//! will retire its negative assertion.
//!
//! Citations:
//! * Python source-of-truth: `sts2-rl-agent@01c094d`,
//!   `sts2_env/core/combat.py:2322-2443` (on-draw + turn-end-in-hand).
//! * Rust dispatch sites verified:
//!   - `crates/sts2-core/src/combat_loop.rs:608-685` — Step 2 ethereal
//!     expel + Step 3 turn-end-in-hand, both explicit stubs.
//!   - `crates/sts2-core/src/cards.rs:752-785` — `draw()` does not
//!     invoke any per-card on-draw hook (so Void energy drain
//!     silently no-ops).
//!   - `crates/sts2-core/src/combat_loop.rs:2165-2174` — `play_card`
//!     unconditionally routes Play → Discard, so the EXHAUST keyword
//!     on Slimed / SporeMind / Toxic is silently ignored on the
//!     normal play path.

use std::num::NonZeroU32;

use smallvec::SmallVec;
use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::card_effects::{spawn_card_into_pile, PileKind};
use sts2_core::cards::{Card, CardId, CardTags, CardType, TargetType};
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
};

/// Set the combat phase to `PlayerEndTurnPhaseOne` so the assert at the
/// top of `end_player_turn_phase_two` is satisfied. The real engine
/// reaches this phase via the `end_player_turn_phase_one` step; we
/// skip that step here because its body is orthogonal to the
/// Step 2 / Step 3 stubs the bead verifies.
fn arm_end_turn_phase_two(state: &mut CombatState) {
    state.phase = CombatPhase::PlayerEndTurnPhaseOne;
}

fn fresh_state_with_enemy() -> (CombatState, CreatureUid) {
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
    state.player.energy = 3;
    state.player.max_energy = 3;
    (state, uid)
}

/// Build a minimal Card POD for one of the status/curse CardIds and
/// spawn it directly into the given pile.
fn push_card(state: &mut CombatState, card_id: CardId, cost: i8, pile: PileKind) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let placeholder = CardUid::new(raw).expect("nz");
    let card = Card {
        uid: placeholder,
        card_id,
        cost: sts2_core::CardEnergyCost::new(cost),
        card_type: CardType::Status,
        target_type: TargetType::None,
        upgraded: false,
        base_damage: 0,
        base_block: 0,
        ..Card::pod_defaults()
    };
    spawn_card_into_pile(state, card, pile)
}

// ---------------------------------------------------------------------------
// End-of-turn turn-end-in-hand triggers (sub-bead sts2-rs-6ynt.3.1)
//
// Python `_execute_turn_end_in_hand_effect` (combat.py:2352-2443) fires
// for: Beckon, Burn, BadLuck, Debt, Decay, Doubt, Regret, Shame. The
// Rust port has Step 3 of `end_player_turn_phase_two` (combat_loop.rs:
// 674-685) stubbed; these cards sit in hand, contribute nothing, and
// the player walks away unscathed.
//
// Each test below confirms the silent no-op — the player's HP / weak /
// frail / etc. does NOT change after the card sits in hand through
// an end-of-turn boundary.
// ---------------------------------------------------------------------------

/// Burn (UNPLAYABLE, 2 dmg blockable EoT in Python). Bead 6ynt.3.1
/// landed: Burn fires `deal_damage(owner, owner, 2, UNPOWERED|MOVE)`.
#[test]
fn burn_end_of_turn_damage_fires() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    let _ = push_card(&mut state, CardId::Burn, -1, PileKind::Hand);

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(
        state.player.current_hp,
        hp_before - 2,
        "Burn EoT damage should drop the player by 2 HP (sts2-rs-6ynt.3.1)."
    );
}

/// Wither (v0.107.1 status, UNPLAYABLE). Deals 3 self-damage at turn-end
/// in hand (C# `Wither.OnTurnEndInHand` → `DamageVar(3, Unpowered|Move)`).
/// Bead `sts2-rs-v0107-content-port-wb8lk.5`.
#[test]
fn wither_end_of_turn_damage_fires() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    let _ = push_card(&mut state, CardId::Wither, -1, PileKind::Hand);

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(
        state.player.current_hp,
        hp_before - 3,
        "Wither EoT damage should drop the player by 3 HP."
    );
}

/// Wither's FakeUpgrade path (C# `Wither.FakeUpgrade()`): each level adds
/// `+3` turn-end self-damage. The Aeonglass boss drives `fake_upgrade` to
/// keep its Wither cards synced; here we bump the level to 2 and confirm the
/// turn-end damage is `3 + 3 * 2 = 9`.
#[test]
fn wither_fake_upgrade_scales_turn_end_damage() {
    use sts2_content::cards::status::wither;
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    let uid = push_card(&mut state, CardId::Wither, -1, PileKind::Hand);

    // FakeUpgrade twice → level 2 → damage 9.
    let idx = uid.0.get() as usize - 1;
    wither::fake_upgrade(&mut state.cards[idx]);
    wither::fake_upgrade(&mut state.cards[idx]);
    assert_eq!(wither::turn_end_damage(2), 9, "3 + 3*2 = 9");

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(
        state.player.current_hp,
        hp_before - 9,
        "FakeUpgrade level 2 → 9 self-damage at turn-end.",
    );
}

/// Decay (UNPLAYABLE, 2 dmg EoT). Bead 6ynt.3.1 landed.
#[test]
fn decay_end_of_turn_damage_fires() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    let _ = push_card(&mut state, CardId::Decay, -1, PileKind::Hand);

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(
        state.player.current_hp,
        hp_before - 2,
        "Decay EoT damage should drop the player by 2 HP (sts2-rs-6ynt.3.1)."
    );
}

/// Regret (UNPLAYABLE, lose UNBLOCKABLE HP = cards-in-hand-at-turn-end).
/// Bead 6ynt.3.1 landed: with 4 padding cards + 1 Regret in hand, the
/// snapshot is 5 → Regret deals 5 unblockable HP loss.
#[test]
fn regret_end_of_turn_hp_loss_fires_with_snapshot() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    for _ in 0..4 {
        let _ = push_card(&mut state, CardId::Wound, -1, PileKind::Hand);
    }
    let _ = push_card(&mut state, CardId::Regret, -1, PileKind::Hand);

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    // Snapshot at turn-end = 5 (4 Wounds + 1 Regret); Regret deals 5
    // UNBLOCKABLE HP loss to the player. Wound has no end-of-turn
    // trigger so it contributes nothing else.
    assert_eq!(
        state.player.current_hp,
        hp_before - 5,
        "Regret EoT damage should equal the cards-in-hand-at-turn-end snapshot \
         (sts2-rs-6ynt.3.1)."
    );
}

/// Beckon (cost-1, 6 UNBLOCKABLE HP loss EoT in Python). Bead 6ynt.3.1
/// landed.
#[test]
fn beckon_end_of_turn_hp_loss_fires() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let _ = state.init_player_creature();
    let _ = push_card(&mut state, CardId::Beckon, 1, PileKind::Hand);

    let hp_before = state.player.current_hp;
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(
        state.player.current_hp,
        hp_before - 6,
        "Beckon EoT damage should drop the player by 6 HP (sts2-rs-6ynt.3.1)."
    );
}

// ---------------------------------------------------------------------------
// On-draw triggers (sub-bead sts2-rs-6ynt.3.2)
//
// Python `_invoke_card_drawn` (combat.py:2322-2330) drains 1 energy
// when CardId.VOID is drawn. The Rust `cards::draw()` (cards.rs:752-785)
// does not invoke any per-card on-draw hook.
// ---------------------------------------------------------------------------

/// Void on-draw energy drain. Bead 6ynt.3.2.
#[test]
fn void_on_draw_energy_drain_currently_silent_no_op() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    // Seed one Void at the top of the draw pile.
    let _void_uid = push_card(&mut state, CardId::Void, -1, PileKind::Draw);
    state.player.energy = 3;
    let energy_before = state.player.energy;

    // Drive one draw.
    let drawn = sts2_core::cards::draw(&mut state.player, 1, &state.cards, &mut state.rng);
    assert_eq!(
        drawn, 1,
        "Void should land in the hand on a single-card draw"
    );

    // Python would have dropped energy to energy_before - 1.
    assert_eq!(
        state.player.energy, energy_before,
        "Void on-draw energy drain fires unexpectedly — flip once \
         sts2-rs-6ynt.3.2 lands."
    );
}

// ---------------------------------------------------------------------------
// End-of-turn ETHEREAL expel (sub-bead sts2-rs-6ynt.3.3)
//
// Python `combat.py:992-999` moves every ETHEREAL hand card to exhaust
// and fires `after_card_exhausted`. Bead `sts2-rs-6ynt.3.3` activates
// Step 2 of `end_player_turn_phase_two` by sourcing the predicate from
// the engine-side ETHEREAL keyword registry populated by
// `install_default_registry`.
// ---------------------------------------------------------------------------

/// Dazed (UNPLAYABLE + ETHEREAL). Expels to exhaust at end-of-turn.
/// Bead 6ynt.3.3.
#[test]
fn dazed_ethereal_expel_lands_in_exhaust() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let dazed_uid = push_card(&mut state, CardId::Dazed, -1, PileKind::Hand);

    let exhaust_before = state.player.exhaust.len();
    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert!(
        state.player.exhaust.contains(&dazed_uid),
        "Dazed (ETHEREAL) must land in exhaust at end-of-turn (sts2-rs-6ynt.3.3)",
    );
    assert!(
        !state.player.discard.contains(&dazed_uid),
        "Dazed (ETHEREAL) must NOT land in discard",
    );
    assert!(
        !state.player.hand.contains(&dazed_uid),
        "Dazed (ETHEREAL) must NOT remain in hand",
    );
    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before + 1,
        "exhaust pile grew by exactly one ETHEREAL card",
    );
}

/// Folly (UNPLAYABLE + ETERNAL + INNATE + ETHEREAL, `Folly.cs` in both
/// v0.107.1 and v0.111.0). Its spec was missing ETHEREAL, so Rust discarded
/// it at end of turn where C# exhausts it (Regent 1006 / Necrobinder 1017,
/// GH #1054 round 5). Eternal only blocks deck removal, not the expel.
#[test]
fn folly_ethereal_expel_lands_in_exhaust() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let folly_uid = push_card(&mut state, CardId::Folly, -1, PileKind::Hand);

    arm_end_turn_phase_two(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert!(
        state.player.exhaust.contains(&folly_uid),
        "Folly (ETHEREAL) must land in exhaust at end-of-turn",
    );
    assert!(
        !state.player.discard.contains(&folly_uid),
        "Folly (ETHEREAL) must NOT land in discard",
    );
}

// ---------------------------------------------------------------------------
// Exhaust-on-play (sub-bead sts2-rs-6ynt.3.4)
//
// Python `_execute_card_play` honours the EXHAUST keyword (and the
// `force_exhaust` arg) at combat.py:902-904. The Rust port's
// `combat_loop::play_card` unconditionally routes Play → Discard
// (combat_loop.rs:2165-2174). The TODO comment there acknowledges the
// branch is missing.
// ---------------------------------------------------------------------------

/// Slimed (cost 1, EXHAUST) — playing it should exhaust the card.
/// We verify the silent no-op by playing Slimed via the registered
/// handler in isolation — it returns Resolved without moving the
/// card to exhaust. The full `play_card` end-to-end path lives in
/// the engine crate; here we lock the handler-level no-op shape.
/// Bead 6ynt.3.4.
#[test]
fn slimed_handler_is_a_resolved_no_op() {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    // Park Slimed directly on the Play pile (the dispatch contract
    // reads `state.player.play.last()` for any helper that needs to
    // resolve the in-flight card). `spawn_card_into_pile` only knows
    // Draw / Discard / Exhaust / Hand, so we hand-roll the push.
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: CardId::Slimed,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Status,
        target_type: TargetType::None,
        upgraded: false,
        base_damage: 0,
        base_block: 0,
        ..Card::pod_defaults()
    });
    state.player.play.push(uid);

    let exhaust_before = state.player.exhaust.len();
    let res = sts2_core::card_effects::dispatch(&mut state, CardId::Slimed, None);
    assert_eq!(res, sts2_core::card_effects::EffectResult::Resolved);

    // The handler itself does nothing; the EXHAUST routing must happen
    // in play_card (currently broken — bead 6ynt.3.4). After the fix
    // lands the play_card-level integration test will move the card to
    // exhaust; this handler-level test stays valid as-is.
    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before,
        "Slimed handler unexpectedly mutated exhaust pile — \
         the handler is a marker by design; routing belongs in play_card."
    );
    // Card stays where we put it (Play pile).
    assert!(
        state.player.play.contains(&uid),
        "Slimed handler displaced the card from Play unexpectedly."
    );
}

// ---------------------------------------------------------------------------
// Positive coverage — what DOES work correctly today
// ---------------------------------------------------------------------------

/// Every curse + status handler is registered and returns Resolved
/// when dispatched directly. This is the only invariant the original
/// bead `sts2-rs-ltq` (P6-7) locked, and it remains correct.
#[test]
fn every_curse_and_status_handler_is_registered() {
    content_dispatch::ensure_installed();
    // Curses
    for id in [
        CardId::AscendersBane,
        CardId::BadLuck,
        CardId::Clumsy,
        CardId::CurseOfTheBell,
        CardId::Debt,
        CardId::Decay,
        CardId::Doubt,
        CardId::Enthralled,
        CardId::Folly,
        CardId::Greed,
        CardId::Guilty,
        CardId::Injury,
        CardId::Normality,
        CardId::Pain,
        CardId::Parasite,
        CardId::PoorSleep,
        CardId::Regret,
        CardId::Shame,
        CardId::SporeMind,
        CardId::Writhe,
    ] {
        assert!(
            sts2_core::card_effects::lookup(id).is_some(),
            "curse {id:?} has no registered handler"
        );
    }
    // Statuses
    for id in [
        CardId::Wound,
        CardId::Slimed,
        CardId::Dazed,
        CardId::Burn,
        CardId::Void,
    ] {
        assert!(
            sts2_core::card_effects::lookup(id).is_some(),
            "status {id:?} has no registered handler"
        );
    }
}

// ---------------------------------------------------------------------------
// EXHAUST routing on play (sub-bead sts2-rs-6ynt.3.4)
//
// Verifies that the engine-side EXHAUST keyword registry, populated by
// `content_dispatch::install_default_registry` from each `CardSpec.keywords`,
// makes `combat_loop::play_card` route EXHAUST cards to the Exhaust pile
// (instead of the prior unconditional Discard). Mirrors the Python pin's
// `_execute_card_play` honour of the EXHAUST keyword at
// `sts2-rl-agent@01c094d:sts2_env/core/combat.py:902-904`.
// ---------------------------------------------------------------------------

fn play_card_state_with_one(card_id: CardId, cost: i8, ctype: CardType) -> CombatState {
    content_dispatch::ensure_installed();
    let (mut state, _e) = fresh_state_with_enemy();
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id,
        cost: sts2_core::CardEnergyCost::new(cost),
        card_type: ctype,
        target_type: TargetType::None,
        upgraded: false,
        base_damage: 0,
        base_block: 0,
        combat_vars: sts2_core::cards::CombatVars::new(),
        enchantments: sts2_core::cards::CardEnchantments::new(),
        tags: CardTags::NONE,
        energy_spent: 0,
        base_replay_count: 0,
        keyword_override: None,
        affliction: None,
    });
    state.player.hand.push(uid);
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// Slimed (cost 1, EXHAUST) — `play_card` must route Play → Exhaust, not
/// Play → Discard. Bead `sts2-rs-6ynt.3.4`.
#[test]
fn exhaust_routing_slimed_lands_in_exhaust_pile_after_play() {
    let mut state = play_card_state_with_one(CardId::Slimed, 1, CardType::Status);
    let exhaust_before = state.player.exhaust.len();
    let discard_before = state.player.discard.len();

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);

    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before + 1,
        "Slimed should be in the exhaust pile after play (EXHAUST keyword registry miss)"
    );
    assert_eq!(
        state.player.discard.len(),
        discard_before,
        "Slimed must NOT land in discard (would be redrawable — STS2 deviation)"
    );
    assert!(
        state.player.play.is_empty(),
        "Slimed should leave the Play pile cleanly"
    );
}

/// Feed (Ironclad, cost 1, EXHAUST attack) — non-status EXHAUST card must
/// also route Play → Exhaust on the normal play path. Bead 6ynt.3.4.
#[test]
fn exhaust_routing_feed_lands_in_exhaust_pile_after_play() {
    content_dispatch::ensure_installed();
    let (mut state, _enemy) = fresh_state_with_enemy();
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: CardId::Feed,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 10,
        base_block: 0,
        ..Card::pod_defaults()
    });
    state.player.hand.push(uid);
    state.phase = CombatPhase::PlayerPlayPhase;

    let exhaust_before = state.player.exhaust.len();
    let discard_before = state.player.discard.len();

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before + 1,
        "Feed should be in the exhaust pile after play (EXHAUST keyword)"
    );
    assert_eq!(
        state.player.discard.len(),
        discard_before,
        "Feed must NOT land in discard"
    );
}

/// Apotheosis (status-rarity EXHAUST card from the colorless pool) — same
/// shape: Play → Exhaust, not Play → Discard. Bead 6ynt.3.4.
#[test]
fn exhaust_routing_apotheosis_lands_in_exhaust_pile_after_play() {
    // Apotheosis is registered under STATUS_SPECS (status_specs_generated.rs).
    let mut state = play_card_state_with_one(CardId::Apotheosis, 1, CardType::Status);
    let exhaust_before = state.player.exhaust.len();
    let discard_before = state.player.discard.len();

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);

    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before + 1,
        "Apotheosis should land in exhaust pile after play"
    );
    assert_eq!(
        state.player.discard.len(),
        discard_before,
        "Apotheosis must NOT land in discard"
    );
}

/// Negative control: Strike (Ironclad, cost 1, NO EXHAUST) — still routes
/// Play → Discard (no spurious exhaust regression for the common case).
#[test]
fn exhaust_routing_strike_still_lands_in_discard() {
    content_dispatch::ensure_installed();
    let (mut state, enemy_uid) = fresh_state_with_enemy();
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: CardId::StrikeIronclad,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        upgraded: false,
        base_damage: 6,
        base_block: 0,
        ..Card::pod_defaults()
    });
    state.player.hand.push(uid);
    state.phase = CombatPhase::PlayerPlayPhase;
    let _ = enemy_uid;

    let exhaust_before = state.player.exhaust.len();
    let discard_before = state.player.discard.len();
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

    assert_eq!(
        state.player.exhaust.len(),
        exhaust_before,
        "Strike must NOT exhaust on play"
    );
    assert_eq!(
        state.player.discard.len(),
        discard_before + 1,
        "Strike should land in discard pile (default route)"
    );
}

/// Sanity: the EXHAUST registry has been populated for the canonical
/// status-card EXHAUST set after `install_default_registry` runs.
#[test]
fn exhaust_registry_populated_after_install() {
    content_dispatch::ensure_installed();
    for id in [
        CardId::Slimed,
        CardId::Toxic,
        CardId::Apotheosis,
        CardId::Feed,
    ] {
        assert!(
            sts2_core::card_effects::card_exhausts(id),
            "{id:?} should be marked EXHAUST in the engine-side registry"
        );
    }
    // Negative: Strike / Defend / Wound (UNPLAYABLE, no EXHAUST) are not
    // EXHAUST.
    for id in [
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::Wound,
    ] {
        assert!(
            !sts2_core::card_effects::card_exhausts(id),
            "{id:?} should NOT be marked EXHAUST"
        );
    }
}
