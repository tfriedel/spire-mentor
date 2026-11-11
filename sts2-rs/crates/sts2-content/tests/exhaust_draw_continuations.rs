//! Code that C# runs AFTER an awaited `CardCmd.Exhaust` / `CardPileCmd.Draw`
//! whose nested draw suspended on a reshuffle select (Stratagem).
//!
//! An exhaust fires `Hook.AfterCardExhausted`; Dark Embrace draws there, and a
//! draw into an empty draw pile reshuffles and opens Stratagem's grid. C#
//! awaits that draw, pick included, before the code after the exhaust runs.
//! Each test plays such a card with Dark Embrace + Stratagem up and checks the
//! state at the pick and after it.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;

fn fresh(version: GameVersion) -> (CombatState, CreatureUid) {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
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
    state.player.potion_slots = 3;
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

fn push_hand(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let uid = new_card(state, id);
    state.player.hand.push(uid);
    uid
}

fn hand_index(state: &CombatState, uid: sts2_core::CardUid) -> u8 {
    let i = state
        .player
        .hand
        .iter()
        .position(|&u| u == uid)
        .expect("in hand");
    u8::try_from(i).expect("u8")
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn resolve(state: &mut CombatState, sel: &[u8]) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: sel.iter().copied().collect(),
        },
    );
}

/// Option index of `uid` in the open choice.
fn option_of(state: &CombatState, uid: sts2_core::CardUid) -> u8 {
    let pc = state.pending_choice.as_ref().expect("a choice is open");
    let i = pc
        .options
        .iter()
        .position(|o| o.card_uid == Some(uid))
        .expect("option");
    u8::try_from(i).expect("u8")
}

/// Stratagem + Dark Embrace, an empty draw pile and three Defends in the
/// discard: the first exhaust's Dark Embrace draw reshuffles into the grid.
fn stratagem_dark_embrace(version: GameVersion) -> (CombatState, CreatureUid) {
    let (mut state, enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    add_player_power(&mut state, 51, PowerKind::DarkEmbrace, 1);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.discard.push(uid);
    }
    (state, enemy)
}

fn assert_stratagem_open(state: &CombatState) {
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
}

fn assert_at_rest(state: &CombatState) {
    assert!(state.pending_choice.is_none(), "{:?}", state.pending_choice);
    assert!(state.pending_draw_resume.is_none());
}

// ── Second Wind: block after each exhaust's draw ─────────────────────────

/// Second Wind awaits `CardCmd.Exhaust` and then `GainBlock` per non-Attack
/// (SecondWind.cs:29-33, both builds). The first Skill's Dark Embrace draw
/// opens Stratagem's grid: at the pick no block is gained yet and the
/// second Skill is still in the hand.
fn second_wind_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let sw = push_hand(&mut state, CardId::SecondWind);
    let s1 = push_hand(&mut state, CardId::ShrugItOff);
    let s2 = push_hand(&mut state, CardId::ShrugItOff);
    let i = hand_index(&state, sw);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[s1]);
    assert!(
        state.player.hand.contains(&s2),
        "{version:?}: the second Skill waits"
    );
    assert_eq!(
        state.player.block, 0,
        "{version:?}: no block before the pick"
    );

    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[s1, s2]);
    assert_eq!(state.player.block, 2 * 5, "{version:?}");
}

#[test]
fn second_wind_awaits_each_exhausts_draw_v0111() {
    second_wind_awaits_each_exhausts_draw(V111);
}

#[test]
fn second_wind_awaits_each_exhausts_draw_v0107() {
    second_wind_awaits_each_exhausts_draw(V107);
}
// ── Potions ──────────────────────────────────────────────────────────────

fn drink(state: &mut CombatState, kind: PotionKind) {
    state.player.potions.push(kind);
    let slot = u8::try_from(state.player.potions.len() - 1).expect("u8");
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::DrinkPotion {
            potion_slot: slot,
            target: None,
        },
    );
}

/// Glowwater exhausts the hand card by card, then draws (GlowwaterPotion.cs
/// OnUse, both builds): the second card is still in the hand at the pick the
/// first exhaust's draw opened.
fn glowwater_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let a = push_hand(&mut state, CardId::StrikeIronclad);
    let b = push_hand(&mut state, CardId::Bash);
    drink(&mut state, PotionKind::GlowwaterPotion);

    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[a]);
    assert!(state.player.hand.contains(&b), "{version:?}: Bash waits");
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert!(state.player.exhaust.starts_with(&[a, b]), "{version:?}");
    assert_eq!(
        state.player.hand.len(),
        3,
        "{version:?}: the pick and both draws"
    );
    assert!(state.player.draw.is_empty());
}

#[test]
fn glowwater_awaits_each_exhausts_draw_v0111() {
    glowwater_awaits_each_exhausts_draw(V111);
}

#[test]
fn glowwater_awaits_each_exhausts_draw_v0107() {
    glowwater_awaits_each_exhausts_draw(V107);
}
/// Clarity draws, then applies Clarity (Clarity.cs OnUse, both builds): the
/// draw's reshuffle opens Stratagem's grid and the power waits for it.
fn clarity_power_waits_for_its_draw(version: GameVersion) {
    let (mut state, _) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.discard.push(uid);
    }
    drink(&mut state, PotionKind::Clarity);

    assert_stratagem_open(&state);
    assert_eq!(
        state.player_power_amount(PowerKind::Clarity),
        0,
        "{version:?}: no Clarity before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert!(state.player_power_amount(PowerKind::Clarity) > 0);
}

#[test]
fn clarity_power_waits_for_its_draw_v0111() {
    clarity_power_waits_for_its_draw(V111);
}

#[test]
fn clarity_power_waits_for_its_draw_v0107() {
    clarity_power_waits_for_its_draw(V107);
}
// ── Brand / Scavenge: the follow-up after the select's exhaust ───────────

/// Brand exhausts the chosen card, then applies Strength (Brand.cs:42-49,
/// both builds). The exhaust's Dark Embrace draw opens Stratagem's grid;
/// the Strength waits for it.
fn brand_strength_waits_for_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let brand = push_hand(&mut state, CardId::Brand);
    let strike = push_hand(&mut state, CardId::StrikeIronclad);
    let _ = push_hand(&mut state, CardId::StrikeIronclad);
    let i = hand_index(&state, brand);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state.pending_choice.as_ref().expect("Brand's select");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::ExhaustFromHand);
    let o = option_of(&state, strike);
    resolve(&mut state, &[o]);

    assert_stratagem_open(&state);
    assert_eq!(
        state.player_power_amount(PowerKind::Strength),
        0,
        "{version:?}: no Strength before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 1);
}

#[test]
fn brand_strength_waits_for_exhausts_draw_v0111() {
    brand_strength_waits_for_exhausts_draw(V111);
}

#[test]
fn brand_strength_waits_for_exhausts_draw_v0107() {
    brand_strength_waits_for_exhausts_draw(V107);
}

/// Scavenge exhausts the chosen card, then applies Energy Next Turn
/// (Scavenge.cs:31-36, both builds); the power waits for the draw.
fn scavenge_power_waits_for_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let scav = push_hand(&mut state, CardId::Scavenge);
    let strike = push_hand(&mut state, CardId::StrikeIronclad);
    let _ = push_hand(&mut state, CardId::StrikeIronclad);
    let i = hand_index(&state, scav);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let o = option_of(&state, strike);
    resolve(&mut state, &[o]);

    assert_stratagem_open(&state);
    assert_eq!(
        state.player_power_amount(PowerKind::EnergyNextTurn),
        0,
        "{version:?}: no Energy Next Turn before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(state.player_power_amount(PowerKind::EnergyNextTurn), 2);
}

#[test]
fn scavenge_power_waits_for_exhausts_draw_v0111() {
    scavenge_power_waits_for_exhausts_draw(V111);
}

#[test]
fn scavenge_power_waits_for_exhausts_draw_v0107() {
    scavenge_power_waits_for_exhausts_draw(V107);
}

// ── Purity: each selected exhaust awaits its draw ────────────────────────

/// Purity exhausts the selected cards one by one (Purity.cs:28-31, both
/// builds): the second chosen card is still in the hand at the pick the
/// first exhaust's draw opened.
fn purity_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let purity = push_hand(&mut state, CardId::Purity);
    let a = push_hand(&mut state, CardId::StrikeIronclad);
    let b = push_hand(&mut state, CardId::Bash);
    let i = hand_index(&state, purity);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let sel = [option_of(&state, a), option_of(&state, b)];
    resolve(&mut state, &sel);

    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[a]);
    assert!(state.player.hand.contains(&b), "{version:?}: Bash waits");
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert!(state.player.exhaust.starts_with(&[a, b]), "{version:?}");
}

#[test]
fn purity_awaits_each_exhausts_draw_v0111() {
    purity_awaits_each_exhausts_draw(V111);
}

#[test]
fn purity_awaits_each_exhausts_draw_v0107() {
    purity_awaits_each_exhausts_draw(V107);
}

/// Ashwater exhausts the selected cards one by one (Ashwater.cs OnUse).
fn ashwater_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    let a = push_hand(&mut state, CardId::StrikeIronclad);
    let b = push_hand(&mut state, CardId::Bash);
    drink(&mut state, PotionKind::Ashwater);
    let sel = [option_of(&state, a), option_of(&state, b)];
    resolve(&mut state, &sel);

    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[a]);
    assert!(state.player.hand.contains(&b), "{version:?}: Bash waits");
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert!(state.player.exhaust.starts_with(&[a, b]), "{version:?}");
}

#[test]
fn ashwater_awaits_each_exhausts_draw_v0111() {
    ashwater_awaits_each_exhausts_draw(V111);
}

#[test]
fn ashwater_awaits_each_exhausts_draw_v0107() {
    ashwater_awaits_each_exhausts_draw(V107);
}

// ── Discard-hand cards: the follow-up waits for a Sly auto-play's draw ───

/// Storm of Steel / Shadow Step discard the hand; `CardCmd.DiscardAndDraw`
/// auto-plays the discarded Sly cards after the discards (CardCmd.cs:201-204
/// v0.111.0), and the card's own code waits for them: the Shivs
/// (StormOfSteel.cs OnPlay) and Shadow Step's power. Reflex (Sly) draws 2
/// into an empty draw pile: the reshuffle opens Stratagem's grid, and
/// nothing after the discard has run yet.
fn discard_hand_follow_up_waits_for_sly_draw(version: GameVersion, card: CardId) {
    let (mut state, _) = fresh(version);
    state.player.character = Some(CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.discard.push(uid);
    }
    let played = push_hand(&mut state, card);
    let _ = push_hand(&mut state, CardId::Reflex);
    let i = hand_index(&state, played);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    let shivs = |s: &CombatState| {
        s.player
            .hand
            .iter()
            .filter(|&&u| {
                sts2_core::card_by_uid(&s.cards, u).is_some_and(|c| c.card_id == CardId::Shiv)
            })
            .count()
    };
    assert_eq!(
        shivs(&state),
        0,
        "{version:?} {card:?}: no Shiv before the pick"
    );
    assert_eq!(
        state.player_power_amount(PowerKind::ShadowStep),
        0,
        "{version:?} {card:?}: no Shadow Step before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    if card == CardId::StormOfSteel {
        assert_eq!(shivs(&state), 1, "{version:?}");
    } else {
        assert_eq!(state.player_power_amount(PowerKind::ShadowStep), 1);
    }
}

#[test]
fn storm_of_steel_shivs_wait_for_sly_draw() {
    for v in [V107, V111] {
        discard_hand_follow_up_waits_for_sly_draw(v, CardId::StormOfSteel);
    }
}

#[test]
fn shadow_step_power_waits_for_sly_draw() {
    for v in [V107, V111] {
        discard_hand_follow_up_waits_for_sly_draw(v, CardId::ShadowStep);
    }
}

// ── Joss Paper takes its counter modulo only after its draw ──────────────

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind, counter: i32) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter,
        is_wax: false,
        is_melted: false,
    });
}

fn joss_counter(state: &CombatState) -> i32 {
    state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::JossPaper)
        .expect("joss")
        .counter
}

/// Joss Paper's `DrawIfThresholdMet` awaits `CardPileCmd.Draw` BEFORE
/// `CardsExhausted %= 5` (JossPaper.cs:126-133 v0.111.0; same in v0.107.1).
/// Second Wind exhausts the 5th card: Joss draws Minion Strike, which
/// Hellraiser auto-plays (it draws its own Defend) and which exhausts itself.
/// That nested exhaust sees `CardsExhausted` = 6, meets the threshold again
/// and draws the second Defend; only then does the outer draw's `%=` run.
/// Rust took the modulo first, so the nested exhaust counted from 0 and drew
/// nothing.
fn joss_paper_nested_exhaust_during_its_draw(version: GameVersion) {
    let (mut state, _) = fresh(version);
    add_relic(&mut state, sts2_core::RelicKind::JossPaper, 4);
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    let minion = new_card(&mut state, CardId::MinionStrike);
    state.player.draw.push(minion);
    let own = new_card(&mut state, CardId::DefendIronclad);
    state.player.draw.push(own);
    let defend = new_card(&mut state, CardId::DefendIronclad);
    state.player.draw.push(defend);
    let sw = push_hand(&mut state, CardId::SecondWind);
    let _ = push_hand(&mut state, CardId::ShrugItOff);
    let i = hand_index(&state, sw);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_at_rest(&state);
    assert!(
        state.player.exhaust.contains(&minion),
        "{version:?}: Minion Strike auto-played and exhausted"
    );
    assert!(
        state.player.hand.contains(&own),
        "{version:?}: Minion Strike's draw"
    );
    assert!(
        state.player.hand.contains(&defend),
        "{version:?}: the nested exhaust met the threshold and drew: hand {:?} draw {:?}",
        state.player.hand,
        state.player.draw
    );
    assert_eq!(joss_counter(&state), 1, "{version:?}");
}

#[test]
fn joss_paper_nested_exhaust_during_its_draw_v0111() {
    joss_paper_nested_exhaust_during_its_draw(V111);
}

#[test]
fn joss_paper_nested_exhaust_during_its_draw_v0107() {
    joss_paper_nested_exhaust_during_its_draw(V107);
}

/// The same modulo when Joss Paper's draw reshuffles into Stratagem's grid:
/// C# is still inside the awaited `Draw` at the pick, so `CardsExhausted`
/// reads 5 there and drops to 0 once the pick and the draw finish.
fn joss_paper_counter_waits_for_its_suspended_draw(version: GameVersion) {
    let (mut state, _) = fresh(version);
    add_relic(&mut state, sts2_core::RelicKind::JossPaper, 4);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.discard.push(uid);
    }
    let sw = push_hand(&mut state, CardId::SecondWind);
    let _ = push_hand(&mut state, CardId::ShrugItOff);
    let i = hand_index(&state, sw);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    assert_eq!(
        joss_counter(&state),
        5,
        "{version:?}: unreduced at the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(joss_counter(&state), 0, "{version:?}");
    assert_eq!(
        state.player.hand.len(),
        2,
        "{version:?}: the pick and the draw"
    );
}

#[test]
fn joss_paper_counter_waits_for_its_suspended_draw_v0111() {
    joss_paper_counter_waits_for_its_suspended_draw(V111);
}

#[test]
fn joss_paper_counter_waits_for_its_suspended_draw_v0107() {
    joss_paper_counter_waits_for_its_suspended_draw(V107);
}

// ── Speedster's AfterCardDrawn waits for Hellraiser's auto-play select ───

/// Pommel Strike draws Seeker Strike; Hellraiser's `AfterCardDrawnEarly`
/// auto-plays it and Seeker Strike opens its draw-pile select. C#
/// `Hook.AfterCardDrawn` (Speedster's damage, SpeedsterPower.cs:15-22) runs
/// only after the Early walk, auto-play and select included, so no
/// Speedster damage lands before the pick.
fn speedster_waits_for_hellraiser_select(version: GameVersion) {
    let (mut state, _) = fresh(version);
    let hp = |s: &CombatState| s.creatures[0].current_hp;
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    add_player_power(&mut state, 51, PowerKind::Speedster, 2);
    let seeker = new_card(&mut state, CardId::SeekerStrike);
    state.player.draw.push(seeker);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.draw.push(uid);
    }
    let pommel = push_hand(&mut state, CardId::PommelStrike);
    let i = hand_index(&state, pommel);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));

    let pc = state
        .pending_choice
        .as_ref()
        .expect("Seeker Strike's select");
    let _ = pc;
    assert_eq!(
        hp(&state),
        500 - 9 - 9,
        "{version:?}: no Speedster hit before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(
        hp(&state),
        500 - 9 - 9 - 2,
        "{version:?}: one Speedster hit"
    );
}

#[test]
fn speedster_waits_for_hellraiser_select_v0111() {
    speedster_waits_for_hellraiser_select(V111);
}

#[test]
fn speedster_waits_for_hellraiser_select_v0107() {
    speedster_waits_for_hellraiser_select(V107);
}

// ── Unceasing Top checks the hand only after the whole play ──────────────

/// `CombatManager.CheckForEmptyHand` runs once the card's whole effect is
/// done (CardModel.cs:2006 v0.111.0), so Second Wind exhausting the only
/// Skill does not trigger Unceasing Top while the exhaust's Dark Embrace
/// draw waits on Stratagem's pick: the pick and the draw land first and the
/// hand is not empty.
fn unceasing_top_waits_for_the_suspended_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    add_relic(&mut state, sts2_core::RelicKind::UnceasingTop, 0);
    let sw = push_hand(&mut state, CardId::SecondWind);
    let _ = push_hand(&mut state, CardId::ShrugItOff);
    let i = hand_index(&state, sw);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    assert!(state.player.hand.is_empty());
    assert_eq!(
        state.player.draw.len(),
        3,
        "{version:?}: Unceasing Top has not drawn"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(
        state.player.hand.len(),
        2,
        "{version:?}: the pick and Dark Embrace's card"
    );
    assert_eq!(state.player.draw.len(), 1, "{version:?}");
}

#[test]
fn unceasing_top_waits_for_the_suspended_draw_v0111() {
    unceasing_top_waits_for_the_suspended_draw(V111);
}

#[test]
fn unceasing_top_waits_for_the_suspended_draw_v0107() {
    unceasing_top_waits_for_the_suspended_draw(V107);
}

// ── v0.107.1 Eidolon: each exhaust awaits its draw ───────────────────────

/// v0.107.1 Eidolon exhausts the hand card by card, then gains Intangible at
/// 9+ (Eidolon.cs:44-54): the second card is still in the hand at the pick
/// the first exhaust's Dark Embrace draw opened.
#[test]
fn eidolon_awaits_each_exhausts_draw_v0107() {
    let (mut state, _) = stratagem_dark_embrace(V107);
    state.player.character = Some(CharacterId::Necrobinder);
    let eidolon = push_hand(&mut state, CardId::Eidolon);
    let a = push_hand(&mut state, CardId::StrikeIronclad);
    let b = push_hand(&mut state, CardId::Bash);
    let i = hand_index(&state, eidolon);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[a]);
    assert!(state.player.hand.contains(&b), "Bash waits");
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert!(state.player.exhaust.starts_with(&[a, b]));
    assert_eq!(state.player_power_amount(PowerKind::Intangible), 0);
}

/// Nine hand cards: Intangible once the last exhaust is done.
#[test]
fn eidolon_nine_cards_grants_intangible_v0107() {
    let (mut state, _) = fresh(V107);
    state.player.character = Some(CharacterId::Necrobinder);
    let eidolon = push_hand(&mut state, CardId::Eidolon);
    for _ in 0..9 {
        let _ = push_hand(&mut state, CardId::StrikeIronclad);
    }
    let i = hand_index(&state, eidolon);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_at_rest(&state);
    assert_eq!(state.player.exhaust.len(), 9);
    assert_eq!(state.player_power_amount(PowerKind::Intangible), 1);
}

// ── Modded raises its cost after its draw ────────────────────────────────

/// Modded draws, then `EnergyCost.AddThisCombat(1)` (Modded.cs:27-28, both
/// builds): its draw's reshuffle opens Stratagem's grid and the cost is
/// still 0 at the pick.
fn modded_cost_waits_for_its_draw(version: GameVersion) {
    let (mut state, _) = fresh(version);
    state.player.character = Some(CharacterId::Defect);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for _ in 0..3 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.discard.push(uid);
    }
    let modded = push_hand(&mut state, CardId::Modded);
    let i = hand_index(&state, modded);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let cost = |s: &CombatState| sts2_core::card_by_uid(&s.cards, modded).expect("card").cost;

    assert_stratagem_open(&state);
    assert_eq!(cost(&state), 0, "{version:?}: no cost bump before the pick");
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(cost(&state), 1, "{version:?}");
}

#[test]
fn modded_cost_waits_for_its_draw_v0111() {
    modded_cost_waits_for_its_draw(V111);
}

#[test]
fn modded_cost_waits_for_its_draw_v0107() {
    modded_cost_waits_for_its_draw(V107);
}

// ── v0.111.0 Toasty Mittens: Strength after the exhaust's draw ───────────

/// v0.111.0 Toasty Mittens exhausts the picked hand card, then applies
/// Strength (ToastyMittens.cs:31-43). The exhaust's Dark Embrace draw opens
/// Stratagem's grid; the Strength waits for it.
#[test]
fn toasty_mittens_strength_waits_for_exhausts_draw_v0111() {
    let (mut state, _) = stratagem_dark_embrace(V111);
    add_relic(&mut state, sts2_core::RelicKind::ToastyMittens, 0);
    let a = push_hand(&mut state, CardId::StrikeIronclad);
    let _ = push_hand(&mut state, CardId::StrikeIronclad);
    let _ = sts2_core::relic_effects::dispatch_after_player_turn_start(&mut state);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Toasty Mittens' select");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::ExhaustFromHand);
    let o = option_of(&state, a);
    resolve(&mut state, &[o]);

    assert_stratagem_open(&state);
    assert_eq!(
        state.player_power_amount(PowerKind::Strength),
        0,
        "no Strength before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 1);
}

// ── Beat Down: each auto-play awaits its own suspended draw ──────────────

/// Beat Down auto-plays two Pommel Strikes from the discard, each awaited
/// (BeatDown.cs:28-44, both builds). The first one's draw finds an empty
/// draw pile, reshuffles the Defends into Stratagem's grid, and C# holds the
/// second Pommel Strike until the pick and that draw finish.
fn beat_down_awaits_each_auto_plays_draw(version: GameVersion) {
    let (mut state, _) = fresh(version);
    let hp = |s: &CombatState| s.creatures[0].current_hp;
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [
        CardId::PommelStrike,
        CardId::PommelStrike,
        CardId::DefendIronclad,
        CardId::DefendIronclad,
    ] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let bd = push_hand(&mut state, CardId::BeatDown);
    let i = hand_index(&state, bd);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

    assert_stratagem_open(&state);
    assert_eq!(
        hp(&state),
        500 - 9,
        "{version:?}: one Pommel Strike before the pick"
    );
    resolve(&mut state, &[0]);
    assert_at_rest(&state);
    assert_eq!(hp(&state), 500 - 2 * 9, "{version:?}: both Pommel Strikes");
}

#[test]
fn beat_down_awaits_each_auto_plays_draw_v0111() {
    beat_down_awaits_each_auto_plays_draw(V111);
}

#[test]
fn beat_down_awaits_each_auto_plays_draw_v0107() {
    beat_down_awaits_each_auto_plays_draw(V107);
}

// ── Tyranny's auto-taken hand exhaust awaits each exhaust's draw ─────────

/// Tyranny 10 at turn start: the hand draw fills the hand to 10 from a
/// 10-card draw pile (no reshuffle), so `FromHand(prefs(10))` auto-takes the
/// whole hand and `TyrannyPower.AfterPlayerTurnStart` exhausts it card by
/// card (TyrannyPower.cs:36-39, both builds). The first exhaust's Dark
/// Embrace draw reshuffles the discard into Stratagem's grid; C# holds the
/// other nine exhausts until the pick and that draw finish.
fn tyranny_auto_take_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    add_player_power(&mut state, 52, PowerKind::Tyranny, 10);
    let hand: Vec<_> = (0..10)
        .map(|_| {
            let uid = new_card(&mut state, CardId::StrikeIronclad);
            state.player.draw.push(uid);
            uid
        })
        .collect();
    state.phase = CombatPhase::PlayerTurnStart;
    state.round = 2;
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    assert_stratagem_open(&state);
    assert_eq!(
        state.player.exhaust.len(),
        1,
        "{version:?}: one exhaust before the pick"
    );
    assert_eq!(state.player.hand.len(), 9, "{version:?}: nine Strikes wait");
    resolve(&mut state, &[0]);
    assert!(state.pending_choice.is_none(), "{:?}", state.pending_choice);
    assert!(state.pending_draw_resume.is_none());
    for uid in &hand {
        assert!(
            state.player.exhaust.contains(uid),
            "{version:?}: every Strike exhausted"
        );
    }
    assert_eq!(state.player.exhaust.first(), Some(&hand[0]));
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
}

#[test]
fn tyranny_auto_take_awaits_each_exhausts_draw_v0111() {
    tyranny_auto_take_awaits_each_exhausts_draw(V111);
}

#[test]
fn tyranny_auto_take_awaits_each_exhausts_draw_v0107() {
    tyranny_auto_take_awaits_each_exhausts_draw(V107);
}

// ── Pael's Eye's turn-end hand exhaust awaits each exhaust's draw ────────

/// Pael's Eye (no card played this turn) exhausts the hand card by card in
/// `BeforeSideTurnEndEarly`, awaiting each `CardCmd.Exhaust`
/// (PaelsEye.cs:124-127, both builds). The first exhaust's Dark Embrace draw
/// reshuffles into Stratagem's grid; its `HookPlayerChoiceContext` pauses
/// Pael's Eye there (Hook.cs:1403-1417 v0.111.0) until the select is
/// answered at the walk's `Task.WhenAll`, so at the pick Bash is still in
/// the hand, and it is exhausted only after.
fn paels_eye_awaits_each_exhausts_draw(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    state.round = 2;
    add_relic(&mut state, sts2_core::RelicKind::PaelsEye, 0);
    let strike = push_hand(&mut state, CardId::StrikeIronclad);
    let bash = push_hand(&mut state, CardId::Bash);
    let r = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);

    assert_eq!(r, sts2_core::StepResult::PendingChoice, "{version:?}");
    assert_stratagem_open(&state);
    assert_eq!(state.player.exhaust.as_slice(), &[strike], "{version:?}");
    assert!(state.player.hand.contains(&bash), "{version:?}: Bash waits");
    resolve(&mut state, &[0]);
    // The turn end finished; what is open now is the extra turn's own hand
    // draw reshuffling into a fresh Stratagem pick.
    assert!(state.pending_end_turn_resume.is_none());
    assert_eq!(
        state.phase,
        sts2_core::CombatPhase::PlayerTurnStart,
        "{version:?}"
    );
    assert!(state
        .pending_draw_resume
        .as_ref()
        .is_some_and(|r| r.from_hand_draw));
    assert!(
        state.player.exhaust.starts_with(&[strike, bash]),
        "{version:?}"
    );
}

#[test]
fn paels_eye_awaits_each_exhausts_draw_v0111() {
    paels_eye_awaits_each_exhausts_draw(V111);
}

#[test]
fn paels_eye_awaits_each_exhausts_draw_v0107() {
    paels_eye_awaits_each_exhausts_draw(V107);
}

/// The same with Screaming Flagon: its `BeforeSideTurnEnd` runs while Pael's
/// Eye is paused on the pick, with Bash still in the hand, so it deals
/// nothing (ScreamingFlagon.cs:21-27).
fn paels_eye_pause_keeps_the_hand_for_screaming_flagon(version: GameVersion) {
    let (mut state, _) = stratagem_dark_embrace(version);
    state.round = 2;
    add_relic(&mut state, sts2_core::RelicKind::PaelsEye, 0);
    add_relic(&mut state, sts2_core::RelicKind::ScreamingFlagon, 0);
    let _ = push_hand(&mut state, CardId::StrikeIronclad);
    let _ = push_hand(&mut state, CardId::Bash);
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_stratagem_open(&state);
    assert_eq!(
        state.creatures[0].current_hp, 500,
        "{version:?}: no Flagon hit"
    );
    resolve(&mut state, &[0]);
    assert_eq!(
        state.phase,
        sts2_core::CombatPhase::PlayerTurnStart,
        "{version:?}"
    );
    assert_eq!(
        state.creatures[0].current_hp, 500,
        "{version:?}: no Flagon hit"
    );
}

#[test]
fn paels_eye_pause_keeps_the_hand_for_screaming_flagon_v0111() {
    paels_eye_pause_keeps_the_hand_for_screaming_flagon(V111);
}

#[test]
fn paels_eye_pause_keeps_the_hand_for_screaming_flagon_v0107() {
    paels_eye_pause_keeps_the_hand_for_screaming_flagon(V107);
}
