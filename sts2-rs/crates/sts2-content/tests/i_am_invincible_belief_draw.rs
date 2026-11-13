//! I Am Invincible under BeliefDraw (GH #785, PR #860) on the AutoPostPlay
//! timing of GH #806 (PR #894).
//!
//! C# runs `IAmInvincible.AfterAutoPostPlayPhaseEntered` (IAmInvincible.cs:29-39)
//! as the first step of `EndPlayerTurnPhaseOneInternal` (CombatManager.cs:1168-1170),
//! ahead of `Hook.BeforeTurnEnd` (:1182), `DoTurnEnd` (:1194) and
//! `Hook.BeforeFlush` (:1207) — the turn-end steps that can ask the player to
//! choose. Under BeliefDraw "is the top an I Am Invincible?" is a chance event
//! that the engine resolves by sampling the hidden top and pinning it there
//! (`cards::peek_top_card_if`). The player only learns whether a copy played,
//! so a decision node that could read that pin would know the hidden top card.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::combat_loop::{end_player_turn_phase_one, end_player_turn_phase_two};
use sts2_core::{
    CardUid, ChoiceKind, CombatPhase, CombatState, GameVersion, MonsterKind, PowerKind, Side,
    StepResult,
};

fn push_card(state: &mut CombatState, id: CardId, block: i32) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(1),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: block,
        ..Card::pod_defaults()
    });
    uid
}

/// A v0.107.1 end of turn whose phase two asks a question: the player holds a
/// Strike and has Well-Laid Plans 1, so `BeforeFlushLate` opens its retain
/// select (v0.107.1 WellLaidPlansPower.cs:23-38; in v0.111.0 the power keeps
/// the whole hand instead, `ShouldFlush`, :12). The draw pile is a BeliefDraw belief over one
/// I Am Invincible (uid 1) and three different cards (uids 2-4): canonical
/// (uid-sorted), nothing pinned. `seed` drives the shuffle stream the peek
/// samples the hidden top from.
fn wlp_belief_state(seed: i32) -> (CombatState, CardUid) {
    content_dispatch::ensure_installed();
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        game_version: GameVersion::V0_107_1,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let player = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = Side::Player;
    let _enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let iai = push_card(&mut state, CardId::IAmInvincible, 10);
    for id in [
        CardId::StrikeSilent,
        CardId::DefendSilent,
        CardId::Neutralize,
    ] {
        let uid = push_card(&mut state, id, 5);
        state.player.draw.push(uid);
    }
    state.player.draw.insert(0, iai);
    let held = push_card(&mut state, CardId::StrikeSilent, 0);
    state.player.hand.push(held);
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        player,
        PowerKind::WellLaidPlans,
        1,
    );
    state.player.deferred_draw_order = true;
    state.run_rng.shuffle = sts2_core::dotnet_rng::Rng::new(seed);
    (state, iai)
}

/// The rider runs, then Well-Laid Plans asks which cards to retain. At that
/// decision the state may depend only on what the player saw — whether the
/// copy played — never on which hidden card the peek sampled onto the top:
/// over 64 shuffle streams (64 realizations of the hidden order), every
/// decision state of one outcome is the same state once the RNG cursor the
/// peek advanced is set aside. Before the merge dropped the sampled pin at
/// the rider's call site, each miss carried its sampled card pinned on top
/// into this choice (and into the search's decision node there).
#[test]
#[serial_test::serial(global_registry)]
fn a_turn_end_choice_after_the_rider_does_not_depend_on_the_hidden_top() {
    let mut at_choice: Vec<Option<CombatState>> = vec![None, None];
    for seed in 0..64i32 {
        let (mut state, iai) = wlp_belief_state(seed);
        assert_eq!(
            end_player_turn_phase_one(&mut state),
            StepResult::Advanced,
            "seed {seed}"
        );
        assert_eq!(
            end_player_turn_phase_two(&mut state),
            StepResult::PendingChoice,
            "seed {seed}: the retain select surfaces"
        );
        assert_eq!(
            state.pending_choice.as_ref().map(|c| c.kind),
            Some(ChoiceKind::RetainAtTurnEnd),
            "seed {seed}"
        );
        let fired = !state.player.draw.contains(&iai);
        assert_eq!(
            state.player.block,
            if fired { 10 } else { 0 },
            "seed {seed}: the copy's block iff it played"
        );
        assert!(
            state.player.pinned_top.is_empty(),
            "seed {seed}: no hidden card is pinned on top at the choice"
        );
        assert!(
            state.player.draw.windows(2).all(|w| w[0] < w[1]),
            "seed {seed}: the unknown middle is canonical: {:?}",
            state.player.draw
        );
        state.run_rng.shuffle = sts2_core::dotnet_rng::Rng::new(0);
        let i = usize::from(fired);
        if let Some(first) = &at_choice[i] {
            assert_eq!(
                &state, first,
                "seed {seed}: fired={fired}: the decision state must not depend on the hidden top"
            );
        } else {
            at_choice[i] = Some(state);
        }
    }
    assert!(
        at_choice.iter().all(Option::is_some),
        "both outcomes (the copy on top, a non-copy on top) were reached"
    );
}
