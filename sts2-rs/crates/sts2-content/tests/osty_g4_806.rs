//! GH #806 (S9c, group G4): Necrobinder / Osty fixes whose C# is identical in
//! v0.107.1 and v0.111.0, so every test runs at both builds.

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{CardId, PileKind};
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{
    CardUid, CombatPhase, CombatState, CreatureUid, GameVersion, MonsterKind, PowerKind, Relic,
    RelicKind,
};

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

/// A play-ready Necrobinder state: one 200-HP Nibbit, an empty hand, and (when
/// `osty`) a live 10-HP Osty.
fn fresh(version: GameVersion, osty: bool) -> (CombatState, CreatureUid) {
    content_dispatch::install_default_registry();
    sts2_content::relics::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: version,
        ..Default::default()
    };
    state.player.character = Some(sts2_core::potions::CharacterId::Necrobinder);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 5;
    state.phase = CombatPhase::PlayerPlayPhase;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 200;
        c.current_hp = 200;
        c.block = 0;
    }
    if osty {
        state.summon_osty(10).expect("osty summoned");
    }
    (state, enemy)
}

fn hp(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn block(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.block)
}

fn give(state: &mut CombatState, uid: CreatureUid, kind: PowerKind, amount: i32) {
    sts2_core::card_effects::apply_power_to_creature(state, uid, kind, amount);
}

fn power_amount(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| {
            c.powers
                .iter()
                .filter_map(|u| state.powers.get_by_id(u.0))
                .filter(|p| p.kind == kind)
                .map(|p| p.amount)
                .sum()
        })
}

fn add_to(state: &mut CombatState, card_id: CardId, pile: PileKind) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    match pile {
        PileKind::Hand => state.player.hand.push(uid),
        PileKind::Draw => state.player.draw.push(uid),
        PileKind::Discard => state.player.discard.push(uid),
        _ => panic!("unsupported pile"),
    }
    uid
}

fn hand_index(state: &CombatState, uid: CardUid) -> u8 {
    u8::try_from(
        state
            .player
            .hand
            .iter()
            .position(|u| *u == uid)
            .expect("card in hand"),
    )
    .expect("fits")
}

fn play(state: &mut CombatState, uid: CardUid, target: Option<u8>) {
    let idx = hand_index(state, uid);
    let _ = sts2_core::combat_loop::play_card(state, idx, target);
}

fn own_relic(state: &mut CombatState, kind: RelicKind) {
    let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("fits")).expect("nz");
    state.relics.push(Relic {
        id,
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

// ── GangUp counts the Osty's hits ─────────────────────────────────────────

/// `GangUp.cs:24-29`: `Base(5) + Extra(5) x` the target's powered-attack
/// `DamageReceivedEntry`s this turn whose dealer is an ally on the owner's
/// side but not the owner. One Poke from the Osty makes it 10.
#[test]
#[serial_test::serial(global_registry)]
fn gang_up_counts_osty_hits_on_target() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        let poke = add_to(&mut state, CardId::Poke, PileKind::Hand);
        play(&mut state, poke, Some(0));
        assert_eq!(hp(&state, enemy), 194, "{v:?}: Poke hits for 6");
        let gang = add_to(&mut state, CardId::GangUp, PileKind::Hand);
        play(&mut state, gang, Some(0));
        assert_eq!(hp(&state, enemy), 184, "{v:?}: GangUp = 5 + 5 x 1 Osty hit");
    }
}

/// The player's own hits do NOT count (`e.Dealer != card.Owner.Creature`).
#[test]
#[serial_test::serial(global_registry)]
fn gang_up_ignores_player_hits() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        let strike = add_to(&mut state, CardId::StrikeNecrobinder, PileKind::Hand);
        play(&mut state, strike, Some(0));
        let after_strike = hp(&state, enemy);
        let gang = add_to(&mut state, CardId::GangUp, PileKind::Hand);
        play(&mut state, gang, Some(0));
        assert_eq!(hp(&state, enemy), after_strike - 5, "{v:?}: flat 5");
    }
}

// ── CurlUp arms on an Osty attack (it carries the card) ───────────────────

/// `CurlUpPower.AfterDamageReceived` needs `cardSource != null`, and every
/// `FromOsty(osty, card)` attack carries its card (AttackCommand.cs:227-238),
/// so Poke latches CurlUp and the block lands once Poke finishes.
#[test]
#[serial_test::serial(global_registry)]
fn curl_up_arms_on_osty_attack() {
    sts2_content::monsters::act2_hive::install_curl_up_handlers();
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        give(&mut state, enemy, PowerKind::CurlUp, 8);
        let poke = add_to(&mut state, CardId::Poke, PileKind::Hand);
        play(&mut state, poke, Some(0));
        assert_eq!(block(&state, enemy), 8, "{v:?}: CurlUp granted its block");
        assert_eq!(
            power_amount(&state, enemy, PowerKind::CurlUp),
            0,
            "{v:?}: removed"
        );
    }
}

// ── Gigantification triples and is spent by an Osty attack card ──────────

/// `GigantificationPower.cs:27-75`: `BeforeAttack` latches any powered attack
/// whose source card the owner owns (the Osty's included) and
/// `ModifyDamageMultiplicative` keys on `cardSource.Owner`, not the dealer.
#[test]
#[serial_test::serial(global_registry)]
fn gigantification_triples_osty_attack_and_is_spent() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        let p = state.player.creature_uid.expect("player");
        give(&mut state, p, PowerKind::Gigantification, 1);
        let poke = add_to(&mut state, CardId::Poke, PileKind::Hand);
        play(&mut state, poke, Some(0));
        assert_eq!(hp(&state, enemy), 200 - 18, "{v:?}: 6 x 3");
        assert_eq!(
            power_amount(&state, p, PowerKind::Gigantification),
            0,
            "{v:?}"
        );
    }
}

/// With no Osty the card issues no `AttackCommand` (`CheckMissingWithAnim`),
/// so nothing latches and Gigantification survives.
#[test]
#[serial_test::serial(global_registry)]
fn gigantification_survives_osty_card_without_osty() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, false);
        let p = state.player.creature_uid.expect("player");
        give(&mut state, p, PowerKind::Gigantification, 1);
        let poke = add_to(&mut state, CardId::Poke, PileKind::Hand);
        play(&mut state, poke, Some(0));
        assert_eq!(
            power_amount(&state, p, PowerKind::Gigantification),
            1,
            "{v:?}"
        );
    }
}

// ── TheBoot raises the Osty's small hits ──────────────────────────────────

/// `TheBoot.ModifyHpLostAfterOstyLate` (TheBoot.cs:27-49): the dealer may be
/// the owner OR the owner's Osty; a powered 1..4 HP loss becomes 5.
#[test]
#[serial_test::serial(global_registry)]
fn the_boot_raises_osty_hits() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        own_relic(&mut state, RelicKind::TheBoot);
        let osty = state.player.pet_uid.expect("osty");
        let _ = sts2_core::damage::apply_damage(
            &mut state,
            CreatureRef::Enemy(osty),
            enemy,
            3,
            ValueProps::MOVE,
            None,
        );
        assert_eq!(hp(&state, enemy), 195, "{v:?}: 3 raised to 5");
    }
}

// ── RightHandHand: AfterCardPlayedLate runs while the card is in Play ─────

/// C# `Hook.AfterCardPlayed` runs every `AfterCardPlayedLate` inside the play
/// loop (Hook.cs:422-437, CardModel.cs:1957-1965), BEFORE the card moves to
/// its result pile (CardModel.cs:1977-1990). A RightHandHand that itself cost
/// 2 is still in Play then, so it does not bounce back to hand.
#[test]
#[serial_test::serial(global_registry)]
fn right_hand_hand_played_for_two_stays_in_discard() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, true);
        let rhh = add_to(&mut state, CardId::RightHandHand, PileKind::Hand);
        state.cards[rhh.0.get() as usize - 1].cost = sts2_core::CardEnergyCost::new(2);
        play(&mut state, rhh, Some(0));
        assert!(state.player.discard.contains(&rhh), "{v:?}: RHH in discard");
        assert!(!state.player.hand.contains(&rhh), "{v:?}: RHH not returned");
    }
}

/// A RightHandHand already in discard still returns after a 2-energy play.
#[test]
#[serial_test::serial(global_registry)]
fn right_hand_hand_in_discard_returns_after_two_energy_play() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, true);
        let rhh = add_to(&mut state, CardId::RightHandHand, PileKind::Discard);
        let strike = add_to(&mut state, CardId::StrikeNecrobinder, PileKind::Hand);
        state.cards[strike.0.get() as usize - 1].cost = sts2_core::CardEnergyCost::new(2);
        play(&mut state, strike, Some(0));
        assert!(state.player.hand.contains(&rhh), "{v:?}: RHH returned");
    }
}

// ── Turn end: a turn-end-in-hand card is never ethereal-swept ─────────────

/// `CombatManager.DoTurnEnd` (v0.107.1 CombatManager.cs:1230-1248, v0.111.0
/// 1616-1631) sorts with `if (HasTurnEndInHandEffect) ... else if (Ethereal)`.
/// A Hexed (so Ethereal) Burn still burns, then `OnTurnEndInHandWrapper`
/// exhausts it (CardModel.cs:1683-1699).
#[test]
#[serial_test::serial(global_registry)]
fn hexed_burn_burns_then_exhausts() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, false);
        let p = state.player.creature_uid.expect("player");
        give(&mut state, p, PowerKind::Hex, 1);
        let burn = add_to(&mut state, CardId::Burn, PileKind::Hand);
        sts2_core::monster_power_effects::hex_after_applied(&mut state, 1);
        let hp_before = state.player.current_hp;
        state.phase = CombatPhase::PlayerEndTurnPhaseOne;
        let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);
        assert_eq!(
            state.player.current_hp,
            hp_before - 2,
            "{v:?}: Burn dealt 2"
        );
        assert!(state.player.exhaust.contains(&burn), "{v:?}: exhausted");
        assert!(
            !state.player.discard.contains(&burn),
            "{v:?}: not discarded"
        );
    }
}

// ── The Osty dies with the player ─────────────────────────────────────────

/// `CreatureCmd.KillWithoutCheckingWinCondition` (CreatureCmd.cs:545-552):
/// a dying player kills its live Osty.
#[test]
#[serial_test::serial(global_registry)]
fn osty_dies_with_player() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, true);
        let osty = state.player.pet_uid.expect("osty");
        state.player.current_hp = 0;
        let r = sts2_core::combat_loop::check_combat_end(&mut state);
        assert_eq!(r, sts2_core::StepResult::CombatLost, "{v:?}");
        assert_eq!(hp(&state, osty), 0, "{v:?}: Osty killed");
    }
}

// ── Fetch draws only on its first finished play this turn ─────────────────

/// `Fetch.cs:29,46-49`: the draw is skipped once THIS card has a
/// `CardPlayFinishedEntry` this turn.
#[test]
#[serial_test::serial(global_registry)]
fn fetch_draws_once_per_turn() {
    for v in BOTH {
        let (mut state, _enemy) = fresh(v, true);
        for _ in 0..3 {
            let _ = add_to(&mut state, CardId::StrikeNecrobinder, PileKind::Draw);
        }
        let fetch = add_to(&mut state, CardId::Fetch, PileKind::Hand);
        play(&mut state, fetch, Some(0));
        assert_eq!(state.player.hand.len(), 1, "{v:?}: first play draws");
        // Return the same Fetch to hand and replay it this turn.
        let pos = state
            .player
            .discard
            .iter()
            .position(|u| *u == fetch)
            .expect("in discard");
        state.player.discard.remove(pos);
        state.player.hand.push(fetch);
        play(&mut state, fetch, Some(0));
        assert_eq!(
            state.player.hand.len(),
            1,
            "{v:?}: second play does not draw"
        );
        // A fresh Fetch instance still draws.
        let other = add_to(&mut state, CardId::Fetch, PileKind::Hand);
        play(&mut state, other, Some(0));
        assert_eq!(state.player.hand.len(), 2, "{v:?}: other instance draws");
    }
}

/// GH #1054: `Fetch.cs:40` checks `CheckMissingWithAnim` once, BEFORE the
/// attack. An Osty killed by the target's Thorns during the hit still lets
/// the draw at `Fetch.cs:45-48` happen.
#[test]
#[serial_test::serial(global_registry)]
fn fetch_draws_when_osty_dies_to_thorns() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, true);
        let osty = state.player.pet_uid.expect("osty");
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == osty.0) {
            c.current_hp = 2;
            c.max_hp = 2;
        }
        give(&mut state, enemy, PowerKind::Thorns, 5);
        for _ in 0..3 {
            let _ = add_to(&mut state, CardId::StrikeNecrobinder, PileKind::Draw);
        }
        let fetch = add_to(&mut state, CardId::Fetch, PileKind::Hand);
        play(&mut state, fetch, Some(0));
        assert_eq!(hp(&state, enemy), 197, "{v:?}: Osty hit the enemy");
        assert_eq!(hp(&state, osty), 0, "{v:?}: Thorns killed the Osty");
        assert_eq!(
            state.player.hand.len(),
            1,
            "{v:?}: Fetch still draws after the Osty died mid-attack"
        );
    }
}

// ── HardenedShell: AfterDamageGiven runs before AfterDamageReceived ───────

/// `CreatureCmd.Damage` fires `AfterDamageGiven` (ReaperForm's Doom, whose
/// application makes SleightOfFlesh hit) BEFORE `AfterDamageReceived`
/// (HardenedShell's tally) for the same hit (CreatureCmd.cs:388-393). The
/// nested 9 damage is capped against the tally from before the outer 6.
#[test]
#[serial_test::serial(global_registry)]
fn hardened_shell_tally_lands_after_reaper_form_nested_hit() {
    for v in BOTH {
        let (mut state, enemy) = fresh(v, false);
        let p = state.player.creature_uid.expect("player");
        give(&mut state, enemy, PowerKind::HardenedShell, 10);
        give(&mut state, p, PowerKind::ReaperForm, 1);
        give(&mut state, p, PowerKind::SleightOfFlesh, 9);
        let _ = sts2_core::damage::apply_damage(
            &mut state,
            CreatureRef::Player,
            enemy,
            6,
            ValueProps::MOVE,
            None,
        );
        assert_eq!(
            hp(&state, enemy),
            200 - 6 - 9,
            "{v:?}: nested hit capped at 10"
        );
    }
}
