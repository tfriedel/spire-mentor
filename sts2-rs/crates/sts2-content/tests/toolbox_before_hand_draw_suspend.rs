//! GH #369: a relic whose `before_hand_draw` opens a card-select suspends the
//! natural hand draw until the select resolves.
//!
//! C# `Toolbox.BeforeHandDraw` (`Toolbox.cs:22-33`) `await`s
//! `CardSelectCmd.FromChooseACardScreen` and only then calls
//! `AddGeneratedCardToCombat(.., PileType.Hand)`. The whole listener runs
//! inside the awaited `Hook.BeforeHandDraw` at `CombatManager.cs:654`, which
//! precedes the `Draw` at `:675` — so the chosen card is in hand BEFORE the
//! natural draw of 5, and the draw sees one fewer free hand slot.
//!
//! Rust used to fire the listener, leave the choice sitting in
//! `pending_choice`, and run the draw plus the rest of turn setup straight
//! past it — the player answered during `PlayerPlayPhase` against an
//! already-drawn hand. Two consequences: the grant landed after the draw
//! instead of before (so a full hand dropped a card C# had room for), and the
//! mirror never parked at the moment the world did, which is why every live
//! Toolbox screen arrived `unparked`.

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType, HAND_LIMIT};
use sts2_core::potions::CharacterId;
use sts2_core::{combat_loop, Action, ChoiceKind, CombatState, Relic, RelicKind, StepResult};

/// A round-1 Ironclad state holding Toolbox, with `draw_pile_size` distinct
/// cards in the draw pile and an empty hand.
fn state_with_toolbox(draw_pile_size: usize) -> CombatState {
    // BOTH registries: the relic bus is what raises the Toolbox listener, the
    // card bus is what the colorless-pool sampler behind its options needs.
    sts2_content::relics::dispatch::ensure_installed();
    content_dispatch::install_default_registry();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(12345_i32),
        ..Default::default()
    };
    state.player.character = Some(CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    let _ = state.init_player_creature();
    state.round = 1;
    state.phase = sts2_core::CombatPhase::PlayerTurnStart;
    state.side = sts2_core::Side::Player;
    state.relics.push(Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: RelicKind::Toolbox,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..draw_pile_size {
        let raw = u32::try_from(state.cards.len() + 1).expect("u32");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
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
        state.player.draw.push(uid);
    }
    state
}

/// The whole point: turn setup STOPS at the Toolbox screen with the hand still
/// empty, and the draw only runs once the pick is answered.
#[test]
fn toolbox_suspends_the_natural_draw_until_its_pick_resolves() {
    let mut state = state_with_toolbox(10);

    let result = combat_loop::start_player_turn(&mut state);

    assert_eq!(
        result,
        StepResult::PendingChoice,
        "Toolbox's select must suspend turn setup"
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Toolbox parks a pending choice");
    assert_eq!(pc.kind, ChoiceKind::AddGeneratedToHand);
    assert!(
        state.player.hand.is_empty(),
        "the natural draw must NOT have run yet — C# awaits the select first, \
         hand was {:?}",
        state.player.hand.len()
    );
    assert!(
        state.pending_relic_hand_draw_resume.is_some(),
        "the stashed draw count is what lets the tail resume"
    );

    // Answering the screen runs the deferred tail: the grant lands, THEN the
    // natural draw of 5 on top of it.
    let after = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );

    assert_ne!(
        after,
        StepResult::PendingChoice,
        "the tail must run to completion"
    );
    assert!(
        state.pending_relic_hand_draw_resume.is_none(),
        "the resume marker is consumed"
    );
    assert_eq!(
        state.player.hand.len(),
        6,
        "1 Toolbox grant + the natural draw of 5"
    );
    assert_eq!(
        state.relics[0].counter, 1,
        "the once-per-combat latch tripped exactly once"
    );
}

/// Ordering has an observable consequence at the hand cap: because the grant
/// precedes the draw, it is the DRAW that gets squeezed, not the grant. Before
/// the fix the draw ran first and the grant was the card that fell off.
#[test]
fn the_grant_precedes_the_draw_so_the_draw_is_what_gets_capped() {
    let mut state = state_with_toolbox(10);
    // Pre-fill the hand to one slot below the cap so exactly one more card
    // fits. C# order (grant, then draw) puts the Toolbox card in that slot.
    for _ in 0..(HAND_LIMIT - 1) {
        let raw = u32::try_from(state.cards.len() + 1).expect("u32");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        state.cards.push(Card {
            uid,
            card_id: CardId::DefendIronclad,
            cost: sts2_core::CardEnergyCost::new(1),
            card_type: CardType::Skill,
            target_type: TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block: 5,
            ..Card::pod_defaults()
        });
        state.player.hand.push(uid);
    }
    let filler = state.player.hand.len();

    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    let granted_id = state
        .pending_choice
        .as_ref()
        .expect("pending choice")
        .options[0]
        .card_id;
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );

    assert_eq!(
        state.player.hand.len(),
        HAND_LIMIT,
        "the hand fills to the cap"
    );
    assert_eq!(
        state.player.hand.len() - filler,
        1,
        "exactly one card was added"
    );
    let added = state.player.hand[filler];
    assert_eq!(
        state
            .cards
            .iter()
            .find(|c| c.uid == added)
            .map(|c| c.card_id),
        Some(granted_id),
        "the card that made it in is the Toolbox grant, not a drawn card — \
         the grant runs BEFORE the draw"
    );
}

/// Round 2+ must not suspend: Toolbox is round-1 only, so turn setup runs
/// straight through and the draw happens as normal.
#[test]
fn a_later_round_does_not_suspend() {
    let mut state = state_with_toolbox(10);
    state.round = 2;

    let result = combat_loop::start_player_turn(&mut state);

    assert_ne!(result, StepResult::PendingChoice);
    assert!(state.pending_relic_hand_draw_resume.is_none());
    assert_eq!(state.player.hand.len(), 5, "the natural draw ran");
    assert_eq!(state.relics[0].counter, 0, "the latch never tripped");
}

/// C# `Hook.BeforeHandDraw` awaits each relic listener in EQUIP order, so a
/// `BeforeHandDraw` relic equipped AFTER Toolbox acts only once the Toolbox
/// pick has landed in hand. Here JeweledMask (round 1: move a Power from the
/// draw pile to hand) must not have moved anything while the Toolbox screen is
/// open, and its card lands after the Toolbox grant. The port used to run the
/// whole relic channel before parking, so the Power was already in hand.
#[test]
fn a_relic_equipped_after_toolbox_waits_for_the_pick() {
    let mut state = state_with_toolbox(10);
    state.relics.push(Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::JeweledMask,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let power = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid: power,
        card_id: CardId::Inflame,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        ..Card::pod_defaults()
    });
    state.player.draw.push(power);

    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    assert!(
        state.player.hand.is_empty(),
        "JeweledMask (equipped after Toolbox) must wait for the Toolbox pick"
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert_eq!(
        state.player.hand.get(1).copied(),
        Some(power),
        "Toolbox grant first, then JeweledMask's Power, then the draw"
    );
    assert_eq!(state.relics[0].counter, 1, "Toolbox fired exactly once");
}

/// C# `StartTurn` awaits `SetupPlayerTurn` only until its first player-choice
/// pause (`HookPlayerChoiceContext.WaitForPauseOrCompletionWithoutAssigningTask`,
/// v0.111.0 `CombatManager.cs:779-785`, v0.107.1 `:516-524`), then runs
/// `Hook.AfterSideTurnStart` while the Toolbox screen is open. The screen
/// itself only proceeds once `RunAutoPrePlayPhase` assigns its task
/// (`HookPlayerChoiceContext.SignalPlayerChoiceBegun` awaits
/// `_taskAssignedCompletionSource`), so Lantern's +1 energy is already granted
/// when the pick is made, and it is not granted again afterwards.
#[test]
fn after_side_turn_start_runs_while_the_toolbox_screen_is_open() {
    let mut state = state_with_toolbox(10);
    state.relics.push(Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::Lantern,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });

    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    assert_eq!(
        state.player.energy, 4,
        "Lantern's AfterSideTurnStart +1 lands during the Toolbox pause"
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert_eq!(state.player.energy, 4, "Lantern fires exactly once");
    assert_eq!(state.player.hand.len(), 6, "1 Toolbox grant + 5 drawn");
}

/// The early `AfterSideTurnStart` also runs before `Hook.ModifyHandDraw`: C#
/// `SetupPlayerTurn` computes the draw AFTER `BeforeHandDraw` returns
/// (v0.111.0 `CombatManager.cs:905-907`), i.e. after the pause. So
/// `ClarityPower.AfterSideTurnStart`'s decrement removes a 1-stack Clarity
/// before its `ModifyHandDraw` +1 is read, and the natural draw is 5, not 6.
#[test]
fn a_toolbox_pause_lets_clarity_expire_before_the_draw_count_is_read() {
    let mut state = state_with_toolbox(10);
    let player = state.player.creature_uid.expect("player creature");
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        player,
        sts2_core::PowerKind::Clarity,
        1,
    );

    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert_eq!(
        state.player.hand.len(),
        6,
        "1 Toolbox grant + 5 drawn: Clarity expired during the pause"
    );
}

/// v0.111.0 Pendulum through a Toolbox pause. C# `Pendulum.BeforeHandDraw`
/// ticks `TurnsSeen = (TurnsSeen + 1) % 3` (decomp v0.111.0 Pendulum.cs:76-89)
/// and `ModifyHandDraw` adds `+1` iff `TurnsSeen == 0` (:91-102), read once
/// after `Hook.BeforeHandDraw` returns (`CombatManager.cs:905-907`). The pause
/// makes Rust read the count again after its relic walk, when Pendulum has
/// already ticked, so that read must see the ticked counter instead of looking
/// ahead a second time (Ironclad 2054 act 3, step 888: entry counter 1 drew one
/// card too many).
fn toolbox_pendulum_hand(pendulum_counter: i32, pendulum_first: bool) -> usize {
    let mut state = state_with_toolbox(10);
    state.game_version = sts2_core::GameVersion::V0_111_0;
    let pendulum = Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::Pendulum,
        counter: pendulum_counter,
        is_wax: false,
        is_melted: false,
    };
    if pendulum_first {
        state.relics.insert(0, pendulum);
    } else {
        state.relics.push(pendulum);
    }
    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pendulum = state
        .relics
        .iter()
        .find(|r| r.kind == RelicKind::Pendulum)
        .expect("pendulum");
    assert_eq!(
        pendulum.counter,
        (pendulum_counter + 1) % 3,
        "Pendulum ticks exactly once"
    );
    state.player.hand.len()
}

#[test]
fn a_toolbox_pause_does_not_double_tick_pendulum_s_hand_draw() {
    for pendulum_first in [false, true] {
        assert_eq!(
            toolbox_pendulum_hand(1, pendulum_first),
            6,
            "counter 1 -> TurnsSeen 2: 1 Toolbox grant + 5 drawn, no Pendulum card \
             (pendulum_first={pendulum_first})"
        );
        assert_eq!(
            toolbox_pendulum_hand(2, pendulum_first),
            7,
            "counter 2 -> TurnsSeen 0: 1 Toolbox grant + 5 + 1 Pendulum \
             (pendulum_first={pendulum_first})"
        );
        assert_eq!(
            toolbox_pendulum_hand(0, pendulum_first),
            6,
            "counter 0 -> TurnsSeen 1: no Pendulum card (pendulum_first={pendulum_first})"
        );
    }
}

/// v0.111.0 Pollinous Core through a Toolbox pause. C# `BeforeHandDraw` bumps
/// `TurnsSeen`, `ModifyHandDraw` adds 2 once `TurnsSeen >= 4` and
/// `AfterModifyingHandDraw` resets it (decomp v0.111.0 PollinousCore.cs:88-125).
/// The paused read comes after Rust's tick, which already reset the counter to
/// 0 on the firing turn, so the look-ahead lost the +2 (loadout Regent 5049,
/// v0.111.0: Foregone Conclusion paused setup, C# drew one card more).
fn toolbox_pollinous_hand(counter: i32) -> (usize, i32) {
    toolbox_pollinous_hand_at(counter, sts2_core::GameVersion::V0_111_0)
}

fn toolbox_pollinous_hand_at(counter: i32, version: sts2_core::GameVersion) -> (usize, i32) {
    let mut state = state_with_toolbox(12);
    state.game_version = version;
    state.relics.push(Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::PollinousCore,
        counter,
        is_wax: false,
        is_melted: false,
    });
    assert_eq!(
        combat_loop::start_player_turn(&mut state),
        StepResult::PendingChoice
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let core = state
        .relics
        .iter()
        .find(|r| r.kind == RelicKind::PollinousCore)
        .expect("pollinous core");
    (state.player.hand.len(), core.counter)
}

#[test]
fn a_toolbox_pause_keeps_pollinous_core_s_extra_draw() {
    assert_eq!(
        toolbox_pollinous_hand(3),
        (8, 0),
        "4th turn: 1 Toolbox grant + 5 + 2 Pollinous, counter resets"
    );
    assert_eq!(
        toolbox_pollinous_hand(2),
        (6, 3),
        "3rd turn: 1 Toolbox grant + 5, no bonus"
    );
    assert_eq!(toolbox_pollinous_hand(0), (6, 1), "1st turn: no bonus");
}

/// v0.107.1 Pollinous Core: `BeforeSideTurnStart` does `TurnsSeen++`,
/// `ModifyHandDraw` adds 2 only when `TurnsSeen == 4`, and
/// `AfterModifyingHandDraw` (run only for a relic that changed the draw,
/// CombatManager.cs:656-657) resets it to 0. The count never wraps on its
/// own: a combat that ended after the tick to 4 but before the hand draw
/// leaves 4, the next combat ticks 5, 6, ... and the relic never fires again
/// (loadout Defect 5189 v0.107.1). Rust cycled `(counter % 4) + 1`.
#[test]
fn pollinous_core_v0107_resets_only_after_raising_the_draw() {
    let v = sts2_core::GameVersion::V0_107_1;
    assert_eq!(
        toolbox_pollinous_hand_at(3, v),
        (8, 0),
        "4th turn: 1 Toolbox grant + 5 + 2 Pollinous, counter resets"
    );
    assert_eq!(
        toolbox_pollinous_hand_at(2, v),
        (6, 3),
        "3rd turn: no bonus"
    );
    assert_eq!(
        toolbox_pollinous_hand_at(4, v),
        (6, 5),
        "stuck past 4: no bonus, no wrap"
    );
}

/// Ninja Scroll is a `BeforeHandDraw` relic too (NinjaScroll.cs:23, both
/// builds), so `Hook.BeforeHandDraw` runs it in obtain order with Toolbox:
/// obtained after Toolbox, its 3 Shivs land after the Toolbox pick; obtained
/// before, ahead of the screen. Rust ran it on `BeforeSideTurnStart`, so the
/// Shivs always preceded the pick (loadout Silent 5189 v0.111.0). Returns
/// the hand's card ids after the pick.
fn toolbox_with_ninja_scroll(scroll_first: bool) -> Vec<CardId> {
    let mut state = state_with_toolbox(10);
    let scroll = Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::NinjaScroll,
        counter: 0,
        is_wax: false,
        is_melted: false,
    };
    if scroll_first {
        state.relics.insert(0, scroll);
    } else {
        state.relics.push(scroll);
    }
    let result = combat_loop::start_player_turn(&mut state);
    assert_eq!(result, StepResult::PendingChoice, "Toolbox's screen");
    let shivs_at_screen = state
        .player
        .hand
        .iter()
        .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Shiv)
        .count();
    assert_eq!(
        shivs_at_screen,
        if scroll_first { 3 } else { 0 },
        "Shivs before the pick"
    );
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    state
        .player
        .hand
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect()
}

#[test]
fn ninja_scroll_after_toolbox_adds_its_shivs_after_the_pick() {
    let hand = toolbox_with_ninja_scroll(false);
    assert_eq!(hand.len(), 9, "{hand:?}");
    assert_ne!(hand[0], CardId::Shiv, "the Toolbox card first: {hand:?}");
    assert!(hand[1..4].iter().all(|c| *c == CardId::Shiv), "{hand:?}");
    assert!(
        hand[4..].iter().all(|c| *c == CardId::StrikeIronclad),
        "{hand:?}"
    );
}

#[test]
fn ninja_scroll_before_toolbox_adds_its_shivs_first() {
    let hand = toolbox_with_ninja_scroll(true);
    assert_eq!(hand.len(), 9, "{hand:?}");
    assert!(hand[0..3].iter().all(|c| *c == CardId::Shiv), "{hand:?}");
    assert_ne!(hand[3], CardId::Shiv, "then the Toolbox card: {hand:?}");
}
