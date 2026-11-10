//! Card effects that keep going after a draw that suspended on a Stratagem
//! reshuffle grid-select, or whose drawn-card report is read after nested
//! on-draw riders ran.
//!
//! C# awaits `CardPileCmd.Draw` (and the `StratagemPower.AfterShuffle` select
//! inside it) before the rest of the card's `OnPlay` runs. Rust suspends the
//! draw on `pending_draw_resume`; whatever the card does after the draw must
//! wait for the resume.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;

fn fresh(version: GameVersion) -> (CombatState, CreatureUid) {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 500,
        max_hp: 500,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    (state, CreatureUid(id))
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId) -> u8 {
    let uid = new_card(state, id);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

// ── Burning Pact after a suspended Dark Embrace draw ─────────────────────

/// Burning Pact with Havoc as the only other hand card: the forced exhaust
/// resolves inline, Dark Embrace draws 1 into an empty draw pile, and the
/// reshuffle opens Stratagem's grid. C# `BurningPact.OnPlay` (v0.111.0
/// BurningPact.cs:26-32) awaits the exhaust, and with it
/// `DarkEmbracePower.AfterCardExhausted`'s `Draw` (DarkEmbracePower.cs:47)
/// and the select inside it, before its own `Draw(2)`. So nothing is drawn
/// before the pick, and the pick, Dark Embrace's card and Burning Pact's two
/// cards all land: four cards (act23 Ironclad v0.111.0 seed 2076, step 649).
fn burning_pact_after_suspended_dark_embrace(version: GameVersion) {
    let (mut state, _) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    add_player_power(&mut state, 51, PowerKind::DarkEmbrace, 1);
    for id in [
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::Bash,
        CardId::Anger,
        CardId::IronWave,
        CardId::TwinStrike,
    ] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let _ = push_hand(&mut state, CardId::Havoc);
    let bp = push_hand(&mut state, CardId::BurningPact);
    let _ = sts2_core::combat_loop::play_card(&mut state, bp, None);

    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.hand.is_empty(),
        "nothing is drawn before the Stratagem pick: {:?}",
        state.player.hand
    );

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(
        state.player.hand.len(),
        4,
        "Stratagem pick + Dark Embrace 1 + Burning Pact 2"
    );
    assert_eq!(state.player.draw.len(), 2, "6 reshuffled - pick - 3 drawn");
}

#[test]
fn burning_pact_draws_after_suspended_dark_embrace_draw_v0111() {
    burning_pact_after_suspended_dark_embrace(V111);
}

#[test]
fn burning_pact_draws_after_suspended_dark_embrace_draw_v0107() {
    burning_pact_after_suspended_dark_embrace(V107);
}

// ── Pillage reads the card ITS draw returned ─────────────────────────────

fn push_draw(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let uid = new_card(state, id);
    state.player.draw.push(uid);
    uid
}

/// Pillage under Hellraiser draws Pommel Strike, which Hellraiser auto-plays;
/// Pommel's own draw takes Battle Trance (a Skill). C# `Pillage.OnPlay` tests
/// the card ITS `CardPileCmd.Draw` returned (Pillage.cs:31-33) — Pommel, an
/// Attack — so it keeps drawing: Twin Strike (auto-played), then Defend.
/// Rust read `last_drawn_card` after the nested draw had overwritten it with
/// Battle Trance and stopped (act23 Ironclad v0.111.0 seed 2166, Mecha
/// Knight left at 1 hp).
fn pillage_reads_its_own_draw_through_nested_draw(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    let _pommel = push_draw(&mut state, CardId::PommelStrike);
    let trance = push_draw(&mut state, CardId::BattleTrance);
    let _twin = push_draw(&mut state, CardId::TwinStrike);
    let defend = push_draw(&mut state, CardId::DefendIronclad);
    let p = push_hand(&mut state, CardId::Pillage);
    let _ = sts2_core::combat_loop::play_card(&mut state, p, Some(0));

    assert!(state.pending_choice.is_none());
    assert!(
        state.player.hand.contains(&trance),
        "Pommel drew Battle Trance"
    );
    assert!(
        state.player.hand.contains(&defend),
        "Pillage kept drawing past the nested draw to Defend: hand {:?}, draw {:?}",
        state.player.hand,
        state.player.draw
    );
    assert!(state.player.draw.is_empty());
}

#[test]
fn pillage_reads_its_own_draw_through_nested_draw_v0111() {
    pillage_reads_its_own_draw_through_nested_draw(V111);
}

#[test]
fn pillage_reads_its_own_draw_through_nested_draw_v0107() {
    pillage_reads_its_own_draw_through_nested_draw(V107);
}

// ── Pillage stays in its loop across a suspended draw ────────────────────

/// Pillage draws Iron Wave, then its next draw reshuffles an all-Attack
/// discard into a Stratagem select. C# is still inside the awaited `Draw`
/// in Pillage's do/while (Pillage.cs:28-33): after the pick that draw takes
/// an Attack and the loop goes on until the piles run dry. Rust broke out
/// of the loop on the suspend (no card reported yet) and drew only the one
/// resumed card (act23 Ironclad v0.111.0 seed 2114, step 1003).
fn pillage_loops_across_stratagem_suspend(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let iron_wave = push_draw(&mut state, CardId::IronWave);
    for id in [CardId::Anger, CardId::TwinStrike, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let p = push_hand(&mut state, CardId::Pillage);
    let _ = sts2_core::combat_loop::play_card(&mut state, p, Some(0));
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_eq!(state.player.hand.as_slice(), &[iron_wave]);

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(
        state.player.hand.len(),
        4,
        "Iron Wave + the pick + both remaining Attacks: hand {:?}",
        state.player.hand
    );
    assert!(state.player.draw.is_empty());
}

#[test]
fn pillage_loops_across_stratagem_suspend_v0111() {
    pillage_loops_across_stratagem_suspend(V111);
}

#[test]
fn pillage_loops_across_stratagem_suspend_v0107() {
    pillage_loops_across_stratagem_suspend(V107);
}

// ── Brightest Flame loses max HP after its suspended draw ────────────────

fn player_max_hp(state: &CombatState) -> i32 {
    let uid = state.player.creature_uid.expect("player mirror");
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("player creature")
        .max_hp
}

/// Brightest Flame into an empty draw pile: its `Draw(2)` reshuffles and
/// opens Stratagem's grid. C# `BrightestFlame.OnPlay` (BrightestFlame.cs:27-32,
/// both builds) gains energy, then awaits the draw (and the select inside it),
/// and only then loses max HP. So at the select the energy is in and the max
/// HP is untouched; after the pick the draw finishes and the max HP drops
/// (act23 Ironclad v0.111.0 seed 8109, step 1047).
fn brightest_flame_after_suspended_draw(version: GameVersion, loss: i32) {
    let (mut state, _) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let bf = push_hand(&mut state, CardId::BrightestFlame);
    let max_before = player_max_hp(&state);
    let energy_before = state.player.energy;
    let _ = sts2_core::combat_loop::play_card(&mut state, bf, None);

    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_eq!(
        state.player.energy,
        energy_before + 2,
        "energy precedes the draw"
    );
    assert_eq!(
        player_max_hp(&state),
        max_before,
        "max HP is lost only after the awaited draw"
    );

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(state.player.hand.len(), 3, "the pick + Brightest Flame's 2");
    assert_eq!(player_max_hp(&state), max_before - loss);
}

#[test]
fn brightest_flame_loses_max_hp_after_suspended_draw_v0111() {
    brightest_flame_after_suspended_draw(V111, 2);
}

#[test]
fn brightest_flame_loses_max_hp_after_suspended_draw_v0107() {
    brightest_flame_after_suspended_draw(V107, 1);
}

// ── Dark Embrace's turn-end draw blocks the end of turn ──────────────────

/// Dark Embrace with one ethereal exhaust banked draws at the player's turn
/// end into an empty draw pile; the reshuffle opens Stratagem's grid. C#
/// `DarkEmbracePower.AfterSideTurnEnd` awaits `CardPileCmd.Draw` (and the
/// select inside it) within `Hook.AfterSideTurnEnd`, before the side
/// switches, so the enemy turn waits for the pick (act23 Ironclad v0.111.0
/// seed 6103 act 3, step 1055). Rust ran the enemy turn and the next player
/// turn-start with the select still open.
fn dark_embrace_turn_end_select_blocks_end_turn(version: GameVersion) {
    let (mut state, _) = fresh(version);
    state.round = 1;
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    add_player_power(&mut state, 51, PowerKind::DarkEmbrace, 1);
    let de = NonZeroU32::new(51).expect("nz");
    state
        .powers
        .get_mut_by_id(de)
        .expect("dark embrace")
        .counter = 1;
    for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(r, sts2_core::StepResult::PendingChoice);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_eq!(state.round, 1, "the enemy turn waits for the pick");
    assert!(state.pending_end_turn_resume.is_some());

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_end_turn_resume.is_none());
    assert_eq!(state.round, 2, "the turn ended once the pick resolved");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
}

#[test]
fn dark_embrace_turn_end_select_blocks_end_turn_v0111() {
    dark_embrace_turn_end_select_blocks_end_turn(V111);
}

#[test]
fn dark_embrace_turn_end_select_blocks_end_turn_v0107() {
    dark_embrace_turn_end_select_blocks_end_turn(V107);
}

// ── Reboot's draw waits for the Stratagem pick of its own shuffle ────────

/// Reboot shuffles the hand into the draw pile, and the shuffle's
/// `StratagemPower.AfterShuffle` opens a pick. C# `Reboot.OnPlay` awaits
/// `CardPileCmd.Shuffle` (and the select in it) before `CardPileCmd.Draw`
/// (v0.107.1 Reboot.cs:29-30), so nothing is drawn before the pick and the
/// picked card leads the hand. Rust drew one card first (Defect
/// random-loadout v0.107.1 seeds 1016/1038, v0.111.0 seed 1026).
fn reboot_draw_waits_for_stratagem_pick(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    state.player.character = Some(CharacterId::Defect);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [
        CardId::StrikeDefect,
        CardId::DefendDefect,
        CardId::Zap,
        CardId::Dualcast,
        CardId::BallLightning,
        CardId::ColdSnap,
    ] {
        let _ = push_draw(&mut state, id);
    }
    let _ = push_hand(&mut state, CardId::StrikeDefect);
    let r = push_hand(&mut state, CardId::Reboot);
    let _ = sts2_core::combat_loop::play_card(&mut state, r, None);

    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.hand.is_empty(),
        "nothing is drawn before the Stratagem pick: {:?}",
        state.player.hand
    );
    let picked = pc.options[0].card_uid.expect("card option");

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(state.player.hand.len(), 5, "the pick + Reboot's 4 draws");
    assert_eq!(
        state.player.hand[0], picked,
        "the pick reaches the hand first"
    );
    assert_eq!(state.player.draw.len(), 2, "7 shuffled - pick - 4 drawn");
}

#[test]
fn reboot_draw_waits_for_stratagem_pick_v0111() {
    reboot_draw_waits_for_stratagem_pick(V111);
}

#[test]
fn reboot_draw_waits_for_stratagem_pick_v0107() {
    reboot_draw_waits_for_stratagem_pick(V107);
}

// ── Calculated Gamble plays its Sly cards after its suspended draw ───────

/// Calculated Gamble discards a Sly Ricochet and draws 1 from an empty draw
/// pile, so the reshuffle (Ricochet included) opens Stratagem's grid. C#
/// `CardCmd.DiscardAndDraw` awaits the whole `CardPileCmd.Draw`, the pick
/// and the card after it included, before it auto-plays the Sly cards
/// (CardCmd.cs:197-204, both builds). Rust auto-played Ricochet straight
/// from the draw pile while the pick was still open, so the resumed draw
/// took the card under it (loadout Silent 1053 v0.107.1: hand cs=6 rs=7).
fn calculated_gamble_plays_sly_after_suspended_draw(version: GameVersion) {
    let (mut state, enemy) = fresh(version);
    state.player.character = Some(CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [CardId::StrikeSilent, CardId::DefendSilent] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let _ = push_hand(&mut state, CardId::Ricochet);
    let ricochet = state.player.hand[0];
    let cg = push_hand(&mut state, CardId::CalculatedGamble);
    let _ = sts2_core::combat_loop::play_card(&mut state, cg, None);

    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.draw.contains(&ricochet),
        "Ricochet waits in the reshuffled draw pile: draw {:?}",
        state.player.draw
    );
    let hp = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == enemy.0)
            .expect("enemy")
            .current_hp
    };
    assert_eq!(hp(&state), 500, "no Sly auto-play before the pick");

    // Pick a card other than Ricochet.
    let pick = pc
        .options
        .iter()
        .position(|o| o.card_uid != Some(ricochet))
        .expect("a non-Sly option");
    resolve(&mut state, u8::try_from(pick).expect("u8"));
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert!(hp(&state) < 500, "Ricochet auto-played after the draw");
    assert!(!state.player.draw.contains(&ricochet));
    assert!(!state.player.hand.contains(&ricochet));
}

#[test]
fn calculated_gamble_plays_sly_after_suspended_draw_v0111() {
    calculated_gamble_plays_sly_after_suspended_draw(V111);
}

#[test]
fn calculated_gamble_plays_sly_after_suspended_draw_v0107() {
    calculated_gamble_plays_sly_after_suspended_draw(V107);
}

// ── An auto-played card reaches its pile after its AfterCardPlayed draw ──

/// A draw-top auto-play (Mayhem) of a Strike is Iron Club's 4th card: its
/// `AfterCardPlayed` draw finds the draw pile empty, and the reshuffle opens
/// Stratagem's grid. C# awaits `Hook.AfterCardPlayed`, the pick and the
/// drawn card included, before the result-pile move (CardModel.cs:1960 ->
/// :1980-1989), so Nostalgia puts the Strike on the draw top only after Iron
/// Club drew. Rust moved the Strike first and the resumed draw took it back
/// (loadout Defect 1038 v0.107.1: hand Barrage vs C# Overclock).
fn auto_play_reaches_its_pile_after_its_after_played_draw(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    add_player_power(&mut state, 51, PowerKind::Nostalgia, 1);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::IronClub,
        counter: 3,
        is_wax: false,
        is_melted: false,
    });
    for id in [CardId::DefendIronclad, CardId::Bash, CardId::Anger] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let strike = push_draw(&mut state, CardId::StrikeIronclad);
    assert!(sts2_core::combat_loop::auto_play_card_from_draw_top(
        &mut state, false
    ));
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.play.contains(&strike),
        "the Strike waits on Play for the pick"
    );

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert_eq!(
        state.player.hand.len(),
        2,
        "the pick + Iron Club's card: hand {:?}",
        state.player.hand
    );
    assert!(!state.player.hand.contains(&strike));
    assert_eq!(
        state.player.draw.first(),
        Some(&strike),
        "Nostalgia put the Strike on top after the draw"
    );
}

#[test]
fn auto_play_reaches_its_pile_after_its_after_played_draw_v0111() {
    auto_play_reaches_its_pile_after_its_after_played_draw(V111);
}

#[test]
fn auto_play_reaches_its_pile_after_its_after_played_draw_v0107() {
    auto_play_reaches_its_pile_after_its_after_played_draw(V107);
}

// ── v0.111.0 Expertise retains every card its suspended draw returns ─────

/// Expertise draws 2: Strike off the top, then an empty draw pile reshuffles
/// into Stratagem's grid. C# `foreach (item in await CardPileCmd.Draw(..))
/// ApplySingleTurnRetain(item)` (Expertise.cs:21-24) retains both drawn
/// cards, the second drawn after the pick, but not the pick. Rust read the
/// drawn cards as soon as the draw suspended and so retained only the
/// Strike (loadout Silent 1060 v0.111.0).
#[test]
fn expertise_retains_cards_drawn_after_a_stratagem_pick_v0111() {
    let (mut state, _enemy) = fresh(V111);
    state.player.character = Some(CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let strike = push_draw(&mut state, CardId::StrikeSilent);
    for id in [CardId::DefendSilent, CardId::Neutralize, CardId::Survivor] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let e = push_hand(&mut state, CardId::Expertise);
    let _ = sts2_core::combat_loop::play_card(&mut state, e, None);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    let pick = pc.options[0].card_uid.expect("uid");

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(state.player.hand.len(), 3, "Strike, the pick, one more");
    let second = *state
        .player
        .hand
        .iter()
        .find(|&&u| u != strike && u != pick)
        .expect("the card drawn after the pick");
    assert!(state.player.retain_this_turn.contains(&strike));
    assert!(
        state.player.retain_this_turn.contains(&second),
        "the card drawn after the pick retains: {:?}",
        state.player.retain_this_turn
    );
    assert!(
        !state.player.retain_this_turn.contains(&pick),
        "the Stratagem pick was not drawn"
    );
}

/// Expertise retains only what its own `CardPileCmd.Draw` returned
/// (Expertise.cs:21-24). Its first card is the turn's first Status, so
/// Iteration's nested draw (IterationPower.cs:18-29) lands between its two
/// cards; that card is not in Expertise's result and does not retain
/// (Silent 5540 v0.111.0).
#[test]
fn expertise_retains_only_its_own_draw_not_a_nested_one_v0111() {
    let (mut state, _enemy) = fresh(V111);
    state.player.character = Some(CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Iteration, 1);
    let dazed = push_draw(&mut state, CardId::Dazed);
    let nested = push_draw(&mut state, CardId::Neutralize);
    let second = push_draw(&mut state, CardId::Survivor);
    let e = push_hand(&mut state, CardId::Expertise);
    let _ = sts2_core::combat_loop::play_card(&mut state, e, None);
    assert!(state.pending_choice.is_none());
    for uid in [dazed, nested, second] {
        assert!(state.player.hand.contains(&uid));
    }
    assert!(state.player.retain_this_turn.contains(&dazed));
    assert!(state.player.retain_this_turn.contains(&second));
    assert!(
        !state.player.retain_this_turn.contains(&nested),
        "Iteration's card is not Expertise's: {:?}",
        state.player.retain_this_turn
    );
}

// ── Scrape discards only the cards its own draw returned ──────────────────

/// Scrape draws a Slimed off the top, then the draw pile runs out and the
/// reshuffle offers Stratagem a single card, which `FromCombatPile` picks
/// without a prompt. The pick is not part of `CardPileCmd.Draw`'s result, so
/// Scrape's filter (Scrape.cs OnPlay) discards the Slimed and keeps the
/// 2-cost Synthesis in hand (loadout Defect 1167 v0.107.1: Rust discarded
/// both).
fn scrape_keeps_an_auto_picked_stratagem_card(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    state.player.character = Some(CharacterId::Defect);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let slimed = push_draw(&mut state, CardId::Slimed);
    let synthesis = new_card(&mut state, CardId::Synthesis);
    state.player.discard.push(synthesis);
    let s = push_hand(&mut state, CardId::Scrape);
    let _ = sts2_core::combat_loop::play_card(&mut state, s, Some(0));
    assert!(
        state.pending_choice.is_none(),
        "{version:?}: one option, no prompt"
    );
    assert!(
        state.player.hand.contains(&synthesis),
        "{version:?}: the Stratagem pick stays in hand: hand {:?} discard {:?}",
        state.player.hand,
        state.player.discard
    );
    assert!(state.player.discard.contains(&slimed), "{version:?}");
}

#[test]
fn scrape_keeps_an_auto_picked_stratagem_card_v0111() {
    scrape_keeps_an_auto_picked_stratagem_card(V111);
}

#[test]
fn scrape_keeps_an_auto_picked_stratagem_card_v0107() {
    scrape_keeps_an_auto_picked_stratagem_card(V107);
}

// ── Escape Plan blocks on the card its suspended draw returns ─────────────

/// Escape Plan into an empty draw pile with Defend and Strike in the
/// discard: the reshuffle opens Stratagem's grid. C# `EscapePlan.OnPlay`
/// (EscapePlan.cs:24-29, both builds) awaits `CardPileCmd.Draw(1)`, select
/// included, and gains block iff the card the draw RETURNED is a Skill. The
/// pick is not that card; the one drawn after it is. Rust read the drawn card
/// as soon as the draw suspended (none yet) and dropped the block (loadout
/// Silent 5079 v0.107.1). Returns the block after picking `pick`.
fn escape_plan_after_suspended_draw(version: GameVersion, pick: CardId) -> i32 {
    let (mut state, _enemy) = fresh(version);
    state.player.character = Some(CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [CardId::DefendSilent, CardId::StrikeSilent] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let e = push_hand(&mut state, CardId::EscapePlan);
    let _ = sts2_core::combat_loop::play_card(&mut state, e, None);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_eq!(state.player.block, 0, "no block before the awaited draw");
    let idx = pc
        .options
        .iter()
        .position(|o| o.card_id == pick)
        .expect("pick offered");
    resolve(&mut state, u8::try_from(idx).expect("u8"));
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(state.player.hand.len(), 2, "the pick + the drawn card");
    state.player.block
}

#[test]
fn escape_plan_blocks_on_a_skill_drawn_after_a_stratagem_pick() {
    for version in [V107, V111] {
        assert_eq!(
            escape_plan_after_suspended_draw(version, CardId::StrikeSilent),
            3,
            "{version:?}: Defend drawn after the pick"
        );
        assert_eq!(
            escape_plan_after_suspended_draw(version, CardId::DefendSilent),
            0,
            "{version:?}: Strike drawn after the pick"
        );
    }
}

// ── Big Bang's stars, energy and Forge wait for its draw ──────────────────

/// Big Bang into an empty draw pile with two cards in the discard: the
/// reshuffle opens Stratagem's grid. C# `BigBang.OnPlay` awaits
/// `CardPileCmd.Draw(1)`, select included, before `GainStars(1)`,
/// `GainEnergy(1)` and `ForgeCmd.Forge(5)` (BigBang.cs:41-45, both builds).
/// Rust ran the three at once, so the Sovereign Blade reached the hand before
/// the pick (Regent random loadout v0.107.1 seed 5479).
#[test]
fn big_bang_tail_waits_for_its_suspended_draw() {
    for version in [V107, V111] {
        let (mut state, _enemy) = fresh(version);
        state.player.character = Some(CharacterId::Regent);
        add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
        for id in [CardId::StrikeRegent, CardId::DefendRegent] {
            let uid = new_card(&mut state, id);
            state.player.discard.push(uid);
        }
        let b = push_hand(&mut state, CardId::BigBang);
        let stars = state.player.stars;
        let energy = state.player.energy;
        let _ = sts2_core::combat_loop::play_card(&mut state, b, None);
        let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
        assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
        let pick = pc.options[0].card_uid.expect("uid");
        assert!(state.player.hand.is_empty(), "{version:?}: no blade yet");
        assert_eq!(state.player.stars, stars, "{version:?}: stars wait");
        assert_eq!(state.player.energy, energy, "{version:?}: energy waits");

        resolve(&mut state, 0);
        assert!(state.pending_choice.is_none(), "{version:?}");
        assert!(state.pending_draw_resume.is_none(), "{version:?}");
        let ids: Vec<CardId> = state
            .player
            .hand
            .iter()
            .map(|u| state.cards[u.0.get() as usize - 1].card_id)
            .collect();
        assert_eq!(state.player.hand.len(), 3, "{version:?}: {ids:?}");
        assert_eq!(state.player.hand[0], pick, "{version:?}: the pick first");
        assert_eq!(
            ids[2],
            CardId::SovereignBlade,
            "{version:?}: the blade arrives last"
        );
        assert_eq!(state.player.stars, stars + 1, "{version:?}");
        assert_eq!(state.player.energy, energy + 1, "{version:?}");
    }
}

// ── A draw parked behind a shuffle's pick still passes the draw gates ─────

/// Reboot under a draw veto: the shuffle's Stratagem pick still opens, but
/// the `CardPileCmd.Draw` C# awaits after it (Reboot.cs:29-30) starts with
/// `Hook.ShouldDraw` (CardPileCmd.cs:822-828), which Fiddle (Fiddle.cs:24-38,
/// a non-hand draw on the owner's turn) or NoDraw veto. Rust resumed the
/// parked draw straight into `draw_suspending`, past the gates, and drew 4
/// (loadout Defect 5061 v0.107.1: Reboot+ drew 6 under Fiddle).
fn reboot_parked_draw_honours_the_veto(version: GameVersion, fiddle: bool) {
    let (mut state, _enemy) = fresh(version);
    state.player.character = Some(CharacterId::Defect);
    state.side = sts2_core::Side::Player;
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    if fiddle {
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::Fiddle,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
    } else {
        add_player_power(&mut state, 51, PowerKind::NoDraw, 1);
    }
    for id in [
        CardId::StrikeDefect,
        CardId::DefendDefect,
        CardId::Zap,
        CardId::Dualcast,
    ] {
        let _ = push_draw(&mut state, id);
    }
    let r = push_hand(&mut state, CardId::Reboot);
    let _ = sts2_core::combat_loop::play_card(&mut state, r, None);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    let picked = pc.options[0].card_uid.expect("card option");
    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(
        state.player.hand.as_slice(),
        &[picked],
        "{version:?} fiddle={fiddle}: only the pick"
    );
}

#[test]
fn reboot_parked_draw_honours_fiddle_and_no_draw() {
    for version in [V107, V111] {
        reboot_parked_draw_honours_the_veto(version, true);
        reboot_parked_draw_honours_the_veto(version, false);
    }
}

// ── Iteration's first-Status count is per side-turn ───────────────────────

/// C# `IterationPower.AfterCardDrawn` (IterationPower.cs:22-23) counts the
/// Status `CardDrawnEntry`s that `HappenedThisTurn`: same `RoundNumber` AND
/// same `CurrentSide` (CombatHistoryEntry.cs:62-75). A Status drawn on the
/// enemy turn (Centennial Puzzle's draw off an enemy hit) is the first of
/// THAT turn even after one on the player turn, so Iteration draws. Rust
/// reset the count only at the player turn start (loadout Defect 5101
/// v0.111.0).
#[test]
fn iteration_counts_statuses_per_side_turn() {
    for version in [V107, V111] {
        let (mut state, _enemy) = fresh(version);
        state.player.character = Some(CharacterId::Defect);
        add_player_power(&mut state, 50, PowerKind::Iteration, 2);
        // A Status already drawn this player turn.
        state.combat_log.status_cards_drawn_this_turn = 1;
        state.phase = CombatPhase::EnemyTurnStart;
        let _ = sts2_core::combat_loop::enemy_turn_start(&mut state);
        assert_eq!(state.side, sts2_core::Side::Enemy, "{version:?}");
        for id in [CardId::Slimed, CardId::Zap, CardId::Zap] {
            let _ = push_draw(&mut state, id);
        }
        let hand_before = state.player.hand.len();
        let _ = sts2_core::cards::draw_via_state(&mut state, 1);
        assert_eq!(
            state.player.hand.len() - hand_before,
            3,
            "{version:?}: the Status, then Iteration's 2"
        );
    }
}

// ── Thunderclap's Vulnerable waits behind a select one target opened ─────

/// Thunderclap applies Vulnerable one target at a time and awaits each
/// (`PowerCmd.Apply(targets)` loops `await Apply(target)`, PowerCmd.cs:36-48,
/// both builds). The first target's Vulnerable fires Vicious, whose draw takes
/// Seeker Strike; Hellraiser auto-plays it and its draw-pile select opens.
/// C# is still inside the first apply, so the second enemy has no Vulnerable
/// yet and nothing else is drawn until the pick. Rust went on to the second
/// target and Vicious drew over the open select (loadout Ironclad 5130
/// v0.107.1).
fn thunderclap_waits_for_a_select_its_first_target_opened(version: GameVersion) {
    let (mut state, first) = fresh(version);
    let second_id = NonZeroU32::new(3).expect("nz");
    let mut second = state.creatures[0].clone();
    second.id = second_id;
    state.creatures.push(second);
    let second = CreatureUid(second_id);
    state.enemy_uids.push(second);
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    add_player_power(&mut state, 51, PowerKind::Vicious, 1);
    let _seeker = push_draw(&mut state, CardId::SeekerStrike);
    let inflame = push_draw(&mut state, CardId::Inflame);
    let defend = push_draw(&mut state, CardId::DefendIronclad);
    let bash = push_draw(&mut state, CardId::Bash);
    let tc = push_hand(&mut state, CardId::Thunderclap);
    let _ = sts2_core::combat_loop::play_card(&mut state, tc, None);

    let vulnerable = |state: &CombatState, uid: CreatureUid| {
        let creature = state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .expect("enemy");
        creature
            .powers
            .iter()
            .filter_map(|puid| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == puid.0 && p.kind == PowerKind::Vulnerable)
            })
            .map(|p| p.amount)
            .sum::<i32>()
    };
    assert!(
        state.pending_choice.is_some(),
        "{version:?}: Seeker Strike's select"
    );
    assert_eq!(vulnerable(&state, first), 1, "{version:?}");
    assert_eq!(
        vulnerable(&state, second),
        0,
        "{version:?}: the second target waits for the pick"
    );
    assert_eq!(
        state.player.draw.as_slice(),
        &[inflame, defend, bash],
        "{version:?}: nothing drawn over the select"
    );

    // Seeker Strike takes Bash; Vicious's draw for the second target then
    // takes the top card, Inflame.
    let pick = state
        .pending_choice
        .as_ref()
        .and_then(|pc| pc.options.iter().position(|o| o.card_id == CardId::Bash))
        .expect("Bash on offer");
    resolve(&mut state, u8::try_from(pick).expect("u8"));
    assert!(state.pending_choice.is_none(), "{version:?}");
    assert_eq!(vulnerable(&state, second), 1, "{version:?}");
    assert!(state.player.hand.contains(&bash), "{version:?}: the pick");
    assert!(
        state.player.hand.contains(&inflame),
        "{version:?}: Vicious's draw for the second target"
    );
    assert_eq!(state.player.draw.as_slice(), &[defend], "{version:?}");
}

#[test]
fn thunderclap_waits_for_a_select_its_first_target_opened_v0107() {
    thunderclap_waits_for_a_select_its_first_target_opened(V107);
}

#[test]
fn thunderclap_waits_for_a_select_its_first_target_opened_v0111() {
    thunderclap_waits_for_a_select_its_first_target_opened(V111);
}

/// A draw started while an earlier draw is suspended on an open select takes
/// nothing over it: C# is still awaiting that select. Its cards follow the
/// suspended draw's own remainder once the pick resolves (loadout Ironclad
/// 5130 v0.107.1).
#[test]
fn a_draw_over_an_open_select_waits_for_the_pick() {
    for version in [V107, V111] {
        let (mut state, _enemy) = fresh(version);
        add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
        let _seeker = push_draw(&mut state, CardId::SeekerStrike);
        let defend = push_draw(&mut state, CardId::DefendIronclad);
        let bash = push_draw(&mut state, CardId::Bash);
        let inflame = push_draw(&mut state, CardId::Inflame);
        // Draw 2: Seeker Strike is auto-played and its select opens.
        let _ = sts2_core::cards::draw_via_state(&mut state, 2);
        assert!(state.pending_choice.is_some(), "{version:?}");
        let _ = sts2_core::cards::draw_via_state(&mut state, 1);
        assert_eq!(
            state.player.draw.as_slice(),
            &[defend, bash, inflame],
            "{version:?}: nothing drawn over the select"
        );
        let pick = state
            .pending_choice
            .as_ref()
            .and_then(|pc| pc.options.iter().position(|o| o.card_id == CardId::Inflame))
            .expect("Inflame on offer");
        resolve(&mut state, u8::try_from(pick).expect("u8"));
        assert!(state.pending_choice.is_none(), "{version:?}");
        assert_eq!(
            state.player.hand.as_slice(),
            &[inflame, defend, bash],
            "{version:?}: the pick, the first draw's rest, then the second draw"
        );
    }
}

// ── An enchantment's OnPlay select holds the play's finish ───────────────

/// A hand-played Swift Molten Fist: after the card's own `OnPlay`, Swift's
/// draw takes Seeker Strike, Hellraiser auto-plays it and its draw-pile
/// select opens. C# `OnPlayWrapper` awaits `Enchantment.OnPlay` (select
/// included) before `CardPlayFinished`, `AfterCardPlayed` and the result-pile
/// move (CardModel.cs:1938-1970, both builds), so Molten Fist sits in the
/// Play pile until the pick and is then exhausted. Rust ran the rest of the
/// tail over the open select and left Molten Fist in the Play pile
/// (random-loadout Ironclad 5436 v0.111.0).
fn swift_draw_select_holds_the_play_finish(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    let _seeker = push_draw(&mut state, CardId::SeekerStrike);
    for id in [
        CardId::Inflame,
        CardId::DefendIronclad,
        CardId::Bash,
        CardId::Anger,
    ] {
        let _ = push_draw(&mut state, id);
    }
    let fist = push_hand(&mut state, CardId::MoltenFist);
    let fist_uid = state.player.hand[usize::from(fist)];
    let slot = fist_uid.get() as usize - 1;
    state.cards[slot].set_enchantment(sts2_core::cards::EnchantmentId::Swift, 1);
    let plays_before = state.combat_log.card_plays_finished_this_turn;
    let _ = sts2_core::combat_loop::play_card(&mut state, fist, Some(0));

    assert!(
        state.pending_choice.is_some(),
        "{version:?}: Seeker Strike's select"
    );
    assert_eq!(
        state.combat_log.card_plays_finished_this_turn, plays_before,
        "{version:?}: neither Seeker Strike nor Molten Fist has finished its play"
    );
    assert!(
        state.player.play.contains(&fist_uid),
        "{version:?}: Molten Fist waits in the Play pile"
    );

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none(), "{version:?}");
    assert!(state.player.play.is_empty(), "{version:?}: play pile empty");
    assert_eq!(
        state.combat_log.card_plays_finished_this_turn,
        plays_before + 2,
        "{version:?}: both plays finished after the pick"
    );
    assert!(
        state.player.exhaust.contains(&fist_uid),
        "{version:?}: Molten Fist exhausted after the pick"
    );
}

#[test]
fn swift_draw_select_holds_the_play_finish_v0107() {
    swift_draw_select_holds_the_play_finish(V107);
}

#[test]
fn swift_draw_select_holds_the_play_finish_v0111() {
    swift_draw_select_holds_the_play_finish(V111);
}

// ── Energy before the draw: a drawn Void takes it ─────────────────────────

/// Play `card` at 0 energy over a draw pile topped by Void; returns the
/// energy after and whether the Void is in hand.
fn energy_after_drawing_void(version: GameVersion, card: CardId) -> (i32, bool) {
    let (mut state, _enemy) = fresh(version);
    state.player.energy = 0;
    let void = new_card(&mut state, CardId::Void);
    state.player.draw.push(void);
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(uid);
    }
    let i = push_hand(&mut state, card);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    (state.player.energy, state.player.hand.contains(&void))
}

/// v0.107.1 Fuel.OnPlay gains its energy, THEN draws (Fuel.cs): the drawn
/// Void's `LoseEnergy(1)` takes it back to 0. Rust drew first, the loss
/// clamped at 0, and the gain left 1 (Defect random loadout v0.107.1 seed
/// 5496).
#[test]
fn fuel_gains_energy_before_its_draw() {
    assert_eq!(energy_after_drawing_void(V107, CardId::Fuel), (0, true));
}

/// Adrenaline.cs (both builds): GainEnergy, then Draw(2).
#[test]
fn adrenaline_gains_energy_before_its_draw() {
    for version in [V107, V111] {
        assert_eq!(
            energy_after_drawing_void(version, CardId::Adrenaline),
            (0, true),
            "{version:?}"
        );
    }
}

// ── A turn-end-in-hand select blocks the end of turn ─────────────────────

/// Toxic's turn-end self-hit is the combat's first HP loss, so Centennial
/// Puzzle draws into an empty draw pile and the reshuffle opens Stratagem's
/// grid. C# `DoTurnEnd` awaits each `OnTurnEndInHandWrapper`, which awaits
/// `OnTurnEndInHand` before disposing the card: the Toxic stays on Play, the
/// hand is not flushed and the enemy turn waits for the pick (Regent
/// v0.111.0 5574). Rust flushed the hand under the open pick.
fn turn_end_in_hand_select_blocks_end_turn(version: GameVersion) {
    let (mut state, _) = fresh(version);
    state.round = 1;
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::CentennialPuzzle,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let toxic = new_card(&mut state, CardId::Toxic);
    state.player.hand.push(toxic);
    let kept = new_card(&mut state, CardId::DefendIronclad);
    state.player.hand.push(kept);
    for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(r, sts2_core::StepResult::PendingChoice, "{version:?}");
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.play.contains(&toxic),
        "{version:?}: Toxic awaits disposal"
    );
    assert!(
        state.player.hand.contains(&kept),
        "{version:?}: the hand is not flushed under the pick"
    );
    assert_eq!(state.round, 1, "{version:?}: the enemy turn waits");

    resolve(&mut state, 0);
    // The next turn's hand draw may reshuffle into another pick; the end of
    // turn itself is done.
    assert!(state.pending_end_turn_resume.is_none(), "{version:?}");
    assert!(
        !state.player.play.contains(&toxic),
        "{version:?}: Toxic disposed"
    );
    assert_eq!(state.round, 2, "{version:?}: the turn ended after the pick");
}

#[test]
fn turn_end_in_hand_select_blocks_end_turn_v0107() {
    turn_end_in_hand_select_blocks_end_turn(V107);
}

#[test]
fn turn_end_in_hand_select_blocks_end_turn_v0111() {
    turn_end_in_hand_select_blocks_end_turn(V111);
}
