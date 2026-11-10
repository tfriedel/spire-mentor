//! Combat-end and turn-start timing fixes ported from the cynilyn fork. The C#
//! of every item is identical in v0.107.1 and v0.111.0, so each test runs at
//! both builds.

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{
    CardUid, CharacterId, CombatPhase, CombatState, GameVersion, MonsterKind, PowerKind, Relic,
    RelicKind, Side,
};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

/// Ironclad in the play phase against one Nibbit at `enemy_hp`.
fn fresh(version: GameVersion, enemy_hp: i32) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        game_version: version,
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        ..CombatState::default()
    };
    state.player.character = Some(CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = enemy_hp.max(1);
        c.current_hp = enemy_hp;
    }
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = Side::Player;
    state
}

/// Mint `id` from the content card factory without placing it in a pile.
fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn add_relic(state: &mut CombatState, kind: RelicKind) {
    let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("u32")).expect("nz");
    state.relics.push(Relic {
        id,
        kind,
        counter: kind.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
}

/// `fresh`, rewound to the top of round 2's player turn with `draw` as the
/// draw pile (top first).
fn at_turn_start(version: GameVersion, enemy_hp: i32, draw: &[CardId]) -> CombatState {
    let mut state = fresh(version, enemy_hp);
    for &id in draw {
        let uid = mint(&mut state, id);
        state.player.draw.push(uid);
    }
    state.round = 2;
    state.phase = CombatPhase::PlayerTurnStart;
    state
}

fn give_player_power(state: &mut CombatState, kind: PowerKind, amount: i32) {
    let player = state.player.creature_uid.expect("player creature");
    sts2_core::card_effects::apply_power_to_creature(state, player, kind, amount);
}

// ── Vigor applied after an attack began is not consumed by it ─────────────

/// `VigorPower` latches `commandToModify` + `amountWhenAttackStarted` in
/// `BeforeAttack` and consumes only that command in `AfterAttack`
/// (VigorPower.cs:30-56, :80-87, both builds). Hellraiser auto-plays Seeker
/// Strike off the turn-start draw: its `AttackCommand` runs (no Vigor yet),
/// then it parks on the `FromCombatPile` select (SeekerStrike.cs:33-41). The
/// setup pause lets `AfterSideTurnStart` run, so Prep Time applies Vigor 4
/// (CombatManager StartTurn). That Vigor post-dates Seeker's attack: C# never
/// latches it, and it survives the pick (combat_full seeds 2091 / 4878).
#[test]
#[serial_test::serial(global_registry)]
fn vigor_applied_after_seeker_attack_started_survives_its_finish() {
    use sts2_core::{apply, legal_actions, Action};
    for version in VERSIONS {
        let mut draw = vec![CardId::SeekerStrike];
        draw.extend([CardId::DefendIronclad; 6]);
        let mut state = at_turn_start(version, 50, &draw);
        give_player_power(&mut state, PowerKind::Hellraiser, 1);
        give_player_power(&mut state, PowerKind::PrepTime, 4);
        let _ = sts2_core::combat_loop::start_player_turn(&mut state);
        let mut guard = 0;
        while state.pending_choice.is_some() && guard < 8 {
            guard += 1;
            let pick = legal_actions(&state)
                .into_iter()
                .find_map(|a| match a {
                    Action::ResolveChoice { selection } if !selection.is_empty() => Some(selection),
                    _ => None,
                })
                .expect("a pick is legal");
            let _ = apply(&mut state, Action::ResolveChoice { selection: pick });
        }
        assert!(guard > 0, "{version:?}: Seeker's select surfaced");
        assert!(
            state.pending_choice.is_none(),
            "{version:?}: select answered"
        );
        assert_eq!(
            state.player_power_amount(PowerKind::Vigor),
            4,
            "{version:?}: Prep Time's Vigor must survive Seeker Strike's finish"
        );
    }
}

/// A blocked auto-play issues no `AttackCommand`, so `VigorPower.BeforeAttack`
/// never latches it (CardCmd.cs `MoveToResultPileWithoutPlaying`, both builds).
/// Hellraiser auto-plays a drawn Strike while Normality's three-play cap is
/// spent: the Strike goes to the discard unplayed, and the Vigor must stay
/// unlatched for the next real attack.
#[test]
#[serial_test::serial(global_registry)]
fn blocked_auto_play_does_not_latch_vigor() {
    for version in VERSIONS {
        let mut state = fresh(version, 50);
        let normality = mint(&mut state, CardId::Normality);
        state.player.hand.push(normality);
        let strike = mint(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(strike);
        give_player_power(&mut state, PowerKind::Hellraiser, 1);
        give_player_power(&mut state, PowerKind::Vigor, 5);
        state.player.card_plays_started_this_turn = 3;
        let _ = state.draw_into_hand_card_effect(1);
        assert!(
            state.player.discard.contains(&strike),
            "{version:?}: Normality sent the auto-played Strike to the discard"
        );
        assert_eq!(
            state.player.vigor_latch_card, None,
            "{version:?}: nothing latched"
        );
        assert_eq!(
            state.player_power_amount(PowerKind::Vigor),
            5,
            "{version:?}"
        );
    }
}

// ── Item 1: draws issued once combat is ending move no card ───────────────

/// `CardPileCmd.Draw` (CardPileCmd.cs:822 v0.107.1, :1012 v0.111.0) returns an
/// empty draw when `CombatManager.IsOverOrEnding`: Pommel Strike's post-attack
/// draw after its lethal hit moves no card.
#[test]
#[serial_test::serial(global_registry)]
fn draw_after_lethal_hit_moves_no_card() {
    for version in VERSIONS {
        let mut state = fresh(version, 1);
        for _ in 0..5 {
            let uid = mint(&mut state, CardId::StrikeIronclad);
            state.player.draw.push(uid);
        }
        let pommel = mint(&mut state, CardId::PommelStrike);
        state.player.hand.push(pommel);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        assert!(
            state.player.hand.is_empty(),
            "{version:?}: the post-lethal draw moved {:?}",
            state.player.hand
        );
        assert_eq!(state.player.draw.len(), 5, "{version:?}");
        assert_eq!(state.cards_drawn, 0, "{version:?}: no card counted drawn");
    }
}

/// The per-card loop re-checks `IsOverOrEnding` each iteration
/// (CardPileCmd.cs:852 v0.107.1, :1042 v0.111.0): a Hellraiser auto-play that
/// kills the last enemy mid-draw stops the draw there.
#[test]
#[serial_test::serial(global_registry)]
fn hellraiser_kill_mid_draw_stops_the_draw() {
    for version in VERSIONS {
        let mut state = fresh(version, 1);
        let strike = mint(&mut state, CardId::StrikeIronclad);
        let d1 = mint(&mut state, CardId::DefendIronclad);
        let d2 = mint(&mut state, CardId::DefendIronclad);
        state.player.draw = [strike, d1, d2].into_iter().collect();
        give_player_power(&mut state, PowerKind::Hellraiser, 1);
        let moved = state.draw_into_hand_card_effect(3);
        assert_eq!(moved, 1, "{version:?}: only the lethal Strike was drawn");
        assert!(state.player.hand.is_empty(), "{version:?}");
        assert_eq!(state.player.draw.as_slice(), &[d1, d2], "{version:?}");
    }
}

// ── Item 2: a turn-start kill ends combat before AfterSideTurnStart ───────

/// `Hook.AfterSideTurnStart` walks `IterateCombatHookListeners`, which yields
/// nothing once `CombatManager.IsOverOrEnding` (Hook.cs:101 v0.107.1; same in
/// v0.111.0). A Mercury Hourglass kill in `AfterPlayerTurnStart` therefore
/// leaves DemonForm (an `AfterSideTurnStart` power) silent.
#[test]
#[serial_test::serial(global_registry)]
fn mercury_hourglass_kill_skips_after_side_turn_start() {
    for version in VERSIONS {
        let mut state = at_turn_start(version, 3, &[CardId::DefendIronclad]);
        add_relic(&mut state, RelicKind::MercuryHourglass);
        give_player_power(&mut state, PowerKind::DemonForm, 2);
        let _ = sts2_core::combat_loop::start_player_turn(&mut state);
        assert!(state.is_over, "{version:?}: the hourglass killed the enemy");
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            0,
            "{version:?}: DemonForm must not fire on an ending combat"
        );
    }
}

// ── Item 6: an on-draw victory ends combat before AfterPlayerTurnStart ────

/// `SetupPlayerTurn` draws, then fires `Hook.AfterPlayerTurnStart`
/// (CombatManager.cs:675-677 v0.107.1), whose listener walk is empty once the
/// combat is ending (Hook.cs:101). A Hellraiser auto-play off the turn-start
/// draw that kills the last enemy leaves Crimson Mantle (self-damage + block in
/// `AfterPlayerTurnStart`) silent.
#[test]
#[serial_test::serial(global_registry)]
fn on_draw_kill_skips_after_player_turn_start() {
    for version in VERSIONS {
        let mut state = at_turn_start(version, 1, &[CardId::StrikeIronclad]);
        give_player_power(&mut state, PowerKind::Hellraiser, 1);
        give_player_power(&mut state, PowerKind::CrimsonMantle, 4);
        let _ = sts2_core::combat_loop::start_player_turn(&mut state);
        assert!(state.is_over, "{version:?}: the auto-played Strike killed");
        assert_eq!(state.player.current_hp, 80, "{version:?}: no self-damage");
        assert_eq!(
            state.player.block, 0,
            "{version:?}: no Crimson Mantle block"
        );
    }
}

// ── Item 3: Imbued auto-plays in the AutoPrePlay phase ─────────────────────

/// `Imbued.AfterAutoPrePlayPhaseEntered` (Imbued.cs:20, both builds) runs in
/// `RunAutoPrePlayPhase`, after `SetupPlayerTurn`'s `AfterPlayerTurnStart` and
/// the side's `AfterSideTurnStart` (CombatManager.cs:516-570 v0.107.1). An
/// Imbued Entrench therefore doubles the block Crimson Mantle granted in
/// `AfterPlayerTurnStart` (4 → 8), instead of doubling nothing first.
#[test]
#[serial_test::serial(global_registry)]
fn imbued_auto_plays_after_turn_start_hooks() {
    for version in VERSIONS {
        let mut state = at_turn_start(version, 50, &[CardId::DefendIronclad; 5]);
        state.round = 1;
        let entrench = mint(&mut state, CardId::Entrench);
        state.cards[entrench.0.get() as usize - 1]
            .add_enchantment(sts2_core::cards::EnchantmentId::Imbued, 1);
        state.player.draw.push(entrench);
        give_player_power(&mut state, PowerKind::CrimsonMantle, 4);
        let _ = sts2_core::combat_loop::start_player_turn(&mut state);
        assert!(
            state.player.discard.contains(&entrench) || state.player.exhaust.contains(&entrench),
            "{version:?}: the Imbued card auto-played"
        );
        assert_eq!(
            state.player.block, 8,
            "{version:?}: Entrench doubled the Mantle block"
        );
    }
}

/// C# awaits each Imbued card's `CardCmd.AutoPlay` (Imbued.cs:20-27), so an
/// Imbued Hidden Daggers' discard select resolves before the next Imbued card
/// plays, and the play phase opens only after the last one. Rust played the
/// second Hidden Daggers while the first one's select was still open, so its
/// select read a hand the first discard had not touched yet (Silent random
/// loadout v0.111.0 seed 5447).
#[test]
#[serial_test::serial(global_registry)]
fn imbued_selects_resolve_one_card_at_a_time() {
    for version in VERSIONS {
        let mut state = at_turn_start(version, 200, &[CardId::DefendIronclad; 5]);
        state.round = 1;
        let mut daggers = Vec::new();
        for _ in 0..2 {
            let uid = mint(&mut state, CardId::HiddenDaggers);
            state.cards[uid.0.get() as usize - 1]
                .add_enchantment(sts2_core::cards::EnchantmentId::Imbued, 1);
            state.player.draw.push(uid);
            daggers.push(uid);
        }
        let r = sts2_core::combat_loop::start_player_turn(&mut state);
        assert_eq!(r, sts2_core::StepResult::PendingChoice, "{version:?}");
        assert_eq!(state.player.hand.len(), 5, "{version:?}: the opening hand");
        assert!(
            state.player.draw.contains(&daggers[1]),
            "{version:?}: the second Hidden Daggers waits for the first's select"
        );
        assert_ne!(
            state.phase,
            CombatPhase::PlayerPlayPhase,
            "{version:?}: the play phase waits for the Imbued plays"
        );

        let _ = resolve(&mut state, &[0, 1]);
        let pc = state
            .pending_choice
            .as_ref()
            .unwrap_or_else(|| panic!("{version:?}: the second select prompts"));
        assert_eq!(pc.kind, sts2_core::ChoiceKind::DiscardFromHand);
        assert_eq!(
            pc.options.len(),
            5,
            "{version:?}: three Defends and the first card's two Shivs"
        );
        assert!(state.player.discard.contains(&daggers[0]), "{version:?}");
        assert_ne!(state.phase, CombatPhase::PlayerPlayPhase, "{version:?}");

        let _ = resolve(&mut state, &[0, 1]);
        assert!(state.pending_choice.is_none(), "{version:?}");
        assert!(state.pending_autoplay_resume.is_none(), "{version:?}");
        assert_eq!(state.phase, CombatPhase::PlayerPlayPhase, "{version:?}");
        let shivs = state
            .player
            .hand
            .iter()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Shiv)
            .count();
        assert_eq!(shivs, 4, "{version:?}");
        assert_eq!(
            state.player.discard.len(),
            6,
            "{version:?}: 4 discards + 2 daggers"
        );
        for uid in &daggers {
            assert!(state.player.discard.contains(uid), "{version:?}");
        }
    }
}

fn resolve(state: &mut CombatState, picks: &[u8]) -> sts2_core::StepResult {
    let selection = picks.iter().copied().collect();
    sts2_core::apply(state, sts2_core::Action::ResolveChoice { selection })
}

// ── Item 4: a targetless AnyEnemy auto-play is not played ─────────────────

/// `CardCmd.AutoPlay` (CardCmd.cs:74-85, both builds): an `AnyEnemy` card
/// whose target roll finds no hittable enemy goes to its result pile through
/// `MoveToResultPileWithoutPlaying` — no `CardPlayStarted`, no play hooks, no
/// play counters. The lone enemy here is a dead Adaptable corpse, so the
/// combat is not ending and Havoc does reach the auto-play.
#[test]
#[serial_test::serial(global_registry)]
fn targetless_enemy_auto_play_is_not_played() {
    for version in VERSIONS {
        let mut state = fresh(version, 1);
        let enemy = state.enemy_uids[0];
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            enemy,
            PowerKind::Adaptable,
            1,
        );
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.current_hp = 0;
        }
        let strike = mint(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(strike);
        let havoc = mint(&mut state, CardId::Havoc);
        state.player.hand.push(havoc);
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        assert!(
            state.player.exhaust.contains(&strike),
            "{version:?}: the Strike went to its result pile (Havoc force-exhausts)"
        );
        assert_eq!(
            state.player.cards_played_this_turn[sts2_core::cards::CardType::Attack.index()],
            0,
            "{version:?}: the unplayed Strike is not counted"
        );
        assert_eq!(
            state.combat_log.player_card_plays_this_combat, 1,
            "{version:?}: only Havoc was played"
        );
    }
}

// ── Item 5: Disintegration runs after the relic turn-end hooks ────────────

/// `DisintegrationPower.AfterSideTurnEndLate` (DisintegrationPower.cs:19, both
/// builds) runs in `Hook.AfterTurnEnd`'s second walk, after every listener's
/// `AfterSideTurnEnd` — relics included (Hook.cs:1411-1435 v0.107.1). That
/// walk is empty once the combat is ending (Hook.cs:101), so a Parrying Shield
/// hit that kills the last enemy wins before Disintegration damages the player.
#[test]
#[serial_test::serial(global_registry)]
fn parrying_shield_win_skips_disintegration() {
    for version in VERSIONS {
        let mut state = fresh(version, 1);
        state.round = 2;
        state.phase = CombatPhase::PlayerEndTurnPhaseOne;
        state.player.block = 20;
        add_relic(&mut state, RelicKind::ParryingShield);
        give_player_power(&mut state, PowerKind::Disintegration, 21);
        let result = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);
        assert!(
            state.is_over,
            "{version:?}: Parrying Shield won ({result:?})"
        );
        assert_eq!(
            state.player.current_hp, 80,
            "{version:?}: no Disintegration"
        );
    }
}

/// Without a win the late walk runs, after the relic: Parrying Shield still
/// sees the full Block (20 ≥ 10) and Disintegration then takes 20 of its 21
/// from that Block.
#[test]
#[serial_test::serial(global_registry)]
fn disintegration_follows_parrying_shield() {
    for version in VERSIONS {
        let mut state = fresh(version, 50);
        state.round = 2;
        state.phase = CombatPhase::PlayerEndTurnPhaseOne;
        state.player.block = 20;
        add_relic(&mut state, RelicKind::ParryingShield);
        give_player_power(&mut state, PowerKind::Disintegration, 21);
        let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);
        let enemy = state.enemy_uids[0];
        let enemy_hp = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .map(|c| c.current_hp);
        assert_eq!(enemy_hp, Some(44), "{version:?}: Parrying Shield hit for 6");
        assert_eq!(
            state.player.current_hp, 79,
            "{version:?}: Disintegration 21 - 20 block"
        );
    }
}

/// `ConsumingShadowPower.AfterSideTurnEnd` (ConsumingShadowPower.cs:18-28,
/// both builds) is a player power, so it runs before the player's relics in
/// `IterateHookListeners` (CombatState.cs:115-131): the Frost evoke's 5 Block
/// lifts 8 to 13 BEFORE Parrying Shield tests `Block >= 10`. Rust evoked after
/// the relic bus (Defect random-loadout v0.111.0 seeds 1001/1021).
#[test]
#[serial_test::serial(global_registry)]
fn consuming_shadow_evokes_before_parrying_shield() {
    for version in VERSIONS {
        let mut state = fresh(version, 50);
        state.player.character = Some(CharacterId::Defect);
        state.player.orbs = sts2_core::orbs::OrbSlots::with_capacity(3);
        state.player.orbs.channel(sts2_core::orbs::Orb::frost());
        state.round = 2;
        state.phase = CombatPhase::PlayerEndTurnPhaseOne;
        state.player.block = 8;
        add_relic(&mut state, RelicKind::ParryingShield);
        give_player_power(&mut state, PowerKind::ConsumingShadow, 1);
        let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);
        assert_eq!(state.player.block, 13, "{version:?}: Frost evoke 5");
        let enemy = state.enemy_uids[0];
        let enemy_hp = state
            .creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .map(|c| c.current_hp);
        assert_eq!(
            enemy_hp,
            Some(44),
            "{version:?}: Parrying Shield saw the evoked Block and hit for 6"
        );
    }
}

// ── Item 7: combat-start relic hooks run in equip order ───────────────────

/// `Hook.BeforeCombatStart(runState, combatState)` (CombatManager.cs:405)
/// walks `RunState.IterateHookListeners`, whose combat half lists each
/// player's non-melted relics in `player.Relics` (equip) order
/// (CombatState.cs:130, both builds). Vajra (Strength) and
/// Oddly Smooth Stone (Dexterity) therefore apply their powers in equip order,
/// whichever way round they were obtained.
#[test]
#[serial_test::serial(global_registry)]
fn before_combat_start_relics_fire_in_equip_order() {
    for version in VERSIONS {
        for (first, second, kinds) in [
            (
                RelicKind::Vajra,
                RelicKind::OddlySmoothStone,
                [PowerKind::Strength, PowerKind::Dexterity],
            ),
            (
                RelicKind::OddlySmoothStone,
                RelicKind::Vajra,
                [PowerKind::Dexterity, PowerKind::Strength],
            ),
        ] {
            let mut state = fresh(version, 40);
            add_relic(&mut state, first);
            add_relic(&mut state, second);
            let _ = sts2_core::relic_effects::dispatch_before_combat_start(&mut state);
            let order: Vec<PowerKind> = state
                .player_power_uids()
                .iter()
                .filter_map(|u| state.powers.get_by_id(u.0).map(|p| p.kind))
                .collect();
            assert_eq!(order, kinds, "{version:?}: relics {first:?}, {second:?}");
        }
    }
}
