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
