//! GH #793 (S8a, part 2): v0.111.0 card behaviour reworks and new cards.
//!
//! Each `*_v0111` test drives the real play path (`combat_loop::play_card`)
//! on a `CombatState` whose `game_version` is v0.111.0, with the cards built
//! through the build-aware factory (`v0111::card_pod_for`).
//! Where the v0.107.1 behaviour is the contrast, a `*_v0107` twin pins it.
//! Source of truth: `sts2-wiki/decompiled/v0.111.0/MegaCrit.Sts2.Core.Models.Cards/`.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for as card_pod_from_id_for;
use sts2_core::cards::{CardId, KeywordOverride};
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;

fn add_enemy(state: &mut CombatState, raw: u32, hp: i32) -> CreatureUid {
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: hp,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    CreatureUid(id)
}

/// One 100-HP Nibbit; player with 5 energy in the play phase.
fn fresh(version: GameVersion, character: CharacterId) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let _ = add_enemy(&mut state, 1, 100);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_from_id_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn push_draw(state: &mut CombatState, id: CardId) {
    let uid = new_card(state, id, false);
    state.player.draw.push(uid);
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn enemy_power(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("creature");
    c.powers
        .iter()
        .filter_map(|p| state.powers.get_by_id(p.0))
        .find(|p| p.kind == kind)
        .map_or(0, |p| p.amount)
}

fn add_enemy_power(
    state: &mut CombatState,
    uid: CreatureUid,
    raw: u32,
    kind: PowerKind,
    amount: i32,
) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state
        .creatures
        .iter_mut()
        .find(|c| c.id == uid.0)
        .expect("creature")
        .powers
        .push(PowerUid(pid));
}

fn first_enemy(state: &CombatState) -> CreatureUid {
    state.enemy_uids[0]
}

fn hp_of(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

// ── Hyperbeam ──────────────────────────────────────────────────────────────

#[test]
fn hyperbeam_v0107() {
    let mut state = fresh(V107, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::Hyperbeam, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(enemy_hp(&state), 72, "28 damage");
    assert_eq!(state.player_power_amount(PowerKind::Focus), -3);
    assert_eq!(state.player_power_amount(PowerKind::HyperbeamFocusDown), 0);
}

#[test]
fn hyperbeam_v0111() {
    let mut state = fresh(V111, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::Hyperbeam, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(enemy_hp(&state), 76, "24 damage (Hyperbeam.cs:22)");
    assert_eq!(state.player_power_amount(PowerKind::Focus), -3);
    assert_eq!(
        state.player_power_amount(PowerKind::HyperbeamFocusDown),
        3,
        "the Focus loss is a temporary HyperbeamFocusDownPower (Hyperbeam.cs:56)"
    );
}

#[test]
fn hyperbeam_upgrade_v0111() {
    let mut state = fresh(V111, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::Hyperbeam, true);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(enemy_hp(&state), 70, "24 + 6 (Hyperbeam.cs:61)");
}

/// An in-run / in-combat upgrade of a v0.111.0 card applies that build's
/// delta (+6), not v0.107.1's (+8).
#[test]
fn hyperbeam_upgrade_for_v0111() {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut card = card_pod_from_id_for(CardId::Hyperbeam, false, V111).expect("pod");
    assert!(card.upgrade_for(V111));
    assert_eq!(card.base_damage, 30);
    let mut old = card_pod_from_id_for(CardId::Hyperbeam, false, V107).expect("pod");
    assert!(old.upgrade_for(V107));
    assert_eq!(old.base_damage, 36);
}

/// An upgrade applies only its own keyword change on top of the live set:
/// Rainbow's `OnUpgrade` removes Exhaust, so a Retain gained in combat
/// (ChoicesParadox) survives a Bellows upgrade (loadout Defect v0.111.0 2015).
#[test]
fn upgrade_keeps_a_keyword_gained_in_combat() {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    for version in [V107, V111] {
        let mut card = card_pod_from_id_for(CardId::Rainbow, false, version).expect("pod");
        sts2_core::card_effects::apply_retain_keyword(&mut card);
        assert!(card.upgrade_for(version));
        let kw = card.keyword_override.expect("override");
        assert!(
            kw.contains(KeywordOverride::RETAIN),
            "{version:?}: Retain kept"
        );
        assert!(
            !kw.contains(KeywordOverride::EXHAUST),
            "{version:?}: Exhaust removed"
        );
    }
}

/// A Confused roll held in `cost` (`CostAbsolutelySet`) that lands on Rend's
/// v0.107.1 base (2) must survive an in-combat upgrade (Bone Tea); the
/// upgrade's build normalization must not read it as an unmoved base.
fn confused_rend_upgraded_cost(version: GameVersion) -> i8 {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut card = card_pod_from_id_for(CardId::Rend, false, version).expect("pod");
    card.cost = 2;
    card.set_combat_var(sts2_core::cards::VarKey::CostAbsolutelySet, 1);
    assert!(card.upgrade_for(version));
    card.cost
}

#[test]
fn confused_rend_keeps_rolled_cost_on_upgrade_v0107() {
    assert_eq!(confused_rend_upgraded_cost(V107), 2);
}

/// Necrobinder v2 fuzz (full, seed 2823): Fake Snecko Eye + Bone Tea, Rend
/// rolled 2; C# keeps 2 (Rend.cs:25 base 1, ConfusedPower `SetThisCombat`).
#[test]
fn confused_rend_keeps_rolled_cost_on_upgrade_v0111() {
    assert_eq!(confused_rend_upgraded_cost(V111), 2);
}

// ── Expect a Fight ─────────────────────────────────────────────────────────

#[test]
fn expect_a_fight_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    add_player_power(&mut state, 900, PowerKind::Strength, 2);
    let i = push_hand(&mut state, CardId::ExpectAFight, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 2, "costs 3 (ExpectAFight.cs:29)");
    assert_eq!(state.player.block, 25, "15 + 5 x 2 Strength");
    assert_eq!(state.player_power_amount(PowerKind::NoEnergyGain), 0);
}

#[test]
fn expect_a_fight_upgrade_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    add_player_power(&mut state, 900, PowerKind::Strength, 2);
    let i = push_hand(&mut state, CardId::ExpectAFight, true);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(
        state.player.energy, 2,
        "the upgrade no longer lowers the cost"
    );
    assert_eq!(state.player.block, 32, "16 + 8 x 2 Strength");
}

/// v0.111.0 `ExpectAFight.cs:17` `GainsBlock => true` (the block is a
/// `CalculatedBlockVar`, so the card's `base_block` stays 0), which
/// `Nimble.CanEnchant` gates on (`Nimble.cs:9-14`). v0.107.1's energy card
/// has no override (`CardModel.cs:671` `=> false`).
#[test]
fn expect_a_fight_gains_block_only_in_v0111() {
    let v111 = card_pod_from_id_for(CardId::ExpectAFight, false, V111).expect("spec");
    assert!(v111.gains_block_for(V111), "ExpectAFight.cs:17 GainsBlock");
    let v107 = card_pod_from_id_for(CardId::ExpectAFight, false, V107).expect("spec");
    assert!(!v107.gains_block_for(V107), "v0.107.1 has no override");
    let spec =
        sts2_content::cards::spec_index::spec_for_card_id(CardId::ExpectAFight).expect("spec");
    assert!(sts2_content::enchantments::can_enchant_card(
        &spec,
        &v111,
        sts2_core::cards::EnchantmentId::Nimble,
        V111,
    ));
    assert!(!sts2_content::enchantments::can_enchant_card(
        &spec,
        &v107,
        sts2_core::cards::EnchantmentId::Nimble,
        V107,
    ));
}

/// `Nimble.EnchantBlockAdditive => Amount` (`Nimble.cs:17-20`) applies to the
/// `CreatureCmd.GainBlock(CalculatedBlock.Calculate(..), .., cardPlay)` grant
/// (`ExpectAFight.cs:36`) through `Hook.ModifyBlock`'s card-source chain.
#[test]
fn expect_a_fight_nimble_adds_to_calculated_block_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    add_player_power(&mut state, 900, PowerKind::Strength, 2);
    let i = push_hand(&mut state, CardId::ExpectAFight, false);
    let uid = state.player.hand[usize::from(i)];
    state.cards[uid.0.get() as usize - 1]
        .add_enchantment(sts2_core::cards::EnchantmentId::Nimble, 3);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.block, 28, "15 + 5 x 2 Strength + 3 Nimble");
}

#[test]
fn expect_a_fight_v0107() {
    let mut state = fresh(V107, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ExpectAFight, false);
    let _ = push_hand(&mut state, CardId::StrikeIronclad, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 4, "cost 2, +1 per Attack in hand");
    assert_eq!(state.player.block, 0);
}

// ── Outbreak ───────────────────────────────────────────────────────────────

#[test]
fn outbreak_v0111() {
    let mut state = fresh(V111, CharacterId::Silent);
    let second = add_enemy(&mut state, 50, 50);
    let first = first_enemy(&state);
    let i = push_hand(&mut state, CardId::OutbreakCard, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 2, "cost 3 (Outbreak.cs:23)");
    for uid in [first, second] {
        assert_eq!(
            enemy_power(&state, uid, PowerKind::Poison),
            8,
            "9 Poison, then one tick"
        );
    }
    assert_eq!(hp_of(&state, first), 91);
    assert_eq!(hp_of(&state, second), 41);
    assert_eq!(state.player_power_amount(PowerKind::Outbreak), 0);
    assert!(
        state.player.discard.len() == 1,
        "a Skill now, discarded after play"
    );
}

#[test]
fn outbreak_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    let i = push_hand(&mut state, CardId::OutbreakCard, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player_power_amount(PowerKind::Outbreak), 11);
    assert_eq!(
        enemy_power(&state, first_enemy(&state), PowerKind::Poison),
        0
    );
}

// ── Tracking ───────────────────────────────────────────────────────────────

#[test]
fn tracking_v0111() {
    let mut state = fresh(V111, CharacterId::Silent);
    let i = push_hand(&mut state, CardId::Tracking, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let j = push_hand(&mut state, CardId::Tracking, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, j, None);
    assert_eq!(
        state.player_power_amount(PowerKind::Tracking),
        100,
        "50 per play"
    );
}

#[test]
fn tracking_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    let i = push_hand(&mut state, CardId::Tracking, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let j = push_hand(&mut state, CardId::Tracking, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, j, None);
    assert_eq!(
        state.player_power_amount(PowerKind::Tracking),
        3,
        "2 then 1"
    );
}

// ── Forgotten Ritual ───────────────────────────────────────────────────────

#[test]
fn forgotten_ritual_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ForgottenRitual, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 7, "cost 1, +3 with nothing exhausted");
}

#[test]
fn forgotten_ritual_v0107() {
    let mut state = fresh(V107, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::ForgottenRitual, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 4, "no exhaust this turn → no energy");
}

// ── Guiding Star ───────────────────────────────────────────────────────────

#[test]
fn guiding_star_v0111() {
    let mut state = fresh(V111, CharacterId::Regent);
    state.player.stars = 1;
    push_draw(&mut state, CardId::StrikeRegent);
    push_draw(&mut state, CardId::StrikeRegent);
    let i = push_hand(&mut state, CardId::GuidingStar, false);
    let hand_before = state.player.hand.len();
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(state.player.stars, 0, "star cost 1 (GuidingStar.cs:22)");
    assert_eq!(state.player.hand.len(), hand_before - 1, "no draw now");
    assert_eq!(
        state.player_power_amount(PowerKind::DrawCardsNextTurn),
        2,
        "the draw moved to next turn (GuidingStar.cs:50)"
    );
}

#[test]
fn guiding_star_star_cost_v0107() {
    let mut state = fresh(V107, CharacterId::Regent);
    let i = push_hand(&mut state, CardId::GuidingStar, false);
    let uid = state.player.hand[usize::from(i)];
    let card = state.cards[uid.0.get() as usize - 1];
    assert_eq!(sts2_core::effective_star_cost_in_combat(&state, &card), 2);
}

// ── Expertise ──────────────────────────────────────────────────────────────

#[test]
fn expertise_v0111() {
    let mut state = fresh(V111, CharacterId::Silent);
    for _ in 0..4 {
        push_draw(&mut state, CardId::StrikeSilent);
    }
    let i = push_hand(&mut state, CardId::Expertise, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(
        state.player.hand.len(),
        2,
        "draws exactly 2 (Expertise.cs:12)"
    );
    for uid in state.player.hand.clone() {
        assert!(
            state.player.retain_this_turn.contains(&uid),
            "drawn cards Retain this turn"
        );
    }
}

#[test]
fn expertise_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    for _ in 0..8 {
        push_draw(&mut state, CardId::StrikeSilent);
    }
    let i = push_hand(&mut state, CardId::Expertise, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.hand.len(), 6, "draws up to 6 in hand");
}

// ── Eidolon ────────────────────────────────────────────────────────────────

#[test]
fn eidolon_v0111() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    // An Ethereal attack in the exhaust pile, and a plain one beside it.
    let ethereal = new_card(&mut state, CardId::StrikeNecrobinder, false);
    state.cards[ethereal.0.get() as usize - 1].keyword_override =
        Some(KeywordOverride::NONE.with(KeywordOverride::ETHEREAL));
    state.player.exhaust.push(ethereal);
    let plain = new_card(&mut state, CardId::StrikeNecrobinder, false);
    state.player.exhaust.push(plain);
    let i = push_hand(&mut state, CardId::Eidolon, false);
    let eidolon = state.player.hand[usize::from(i)];
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(enemy_hp(&state), 94, "the Ethereal Strike was auto-played");
    assert!(state.player.exhaust.contains(&plain), "non-Ethereal stays");
    assert!(
        state.player.exhaust.contains(&eidolon),
        "Eidolon is Exhaust now (Eidolon.cs:15)"
    );
}

/// GH #1054 (Necrobinder-1001 act-3 `hand_count` split): C# `Eidolon.OnPlay`
/// `await`s each `CardCmd.AutoPlay`, so the first Ethereal Graveblast's
/// discard pick resolves before the second Graveblast plays. Rust ran the
/// whole loop at once: the second select replaced the first, and only one
/// card came back to hand.
#[test]
fn eidolon_waits_for_each_auto_played_select_v0111() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    for _ in 0..2 {
        let g = new_card(&mut state, CardId::Graveblast, false);
        state.cards[g.0.get() as usize - 1].keyword_override =
            Some(KeywordOverride::NONE.with(KeywordOverride::ETHEREAL));
        state.player.exhaust.push(g);
    }
    for _ in 0..3 {
        let d = new_card(&mut state, CardId::DefendNecrobinder, false);
        state.player.discard.push(d);
    }
    let i = push_hand(&mut state, CardId::Eidolon, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    // The next Graveblast's select waits until this one is answered.
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the first Graveblast select is open");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDiscardToHand);
    assert_eq!(pc.options.len(), 3, "all three discard cards are offered");
    let selection: SmallVec<[u8; 4]> = [0u8].into_iter().collect();
    let _ = sts2_core::apply(&mut state, sts2_core::Action::ResolveChoice { selection });
    let second = state
        .pending_choice
        .as_ref()
        .expect("the second Graveblast's select surfaces too");
    assert_eq!(second.kind, sts2_core::ChoiceKind::AddFromDiscardToHand);
    assert_eq!(
        second.options.len(),
        3,
        "the first Graveblast has reached its result pile"
    );
    let selection: SmallVec<[u8; 4]> = [0u8].into_iter().collect();
    let _ = sts2_core::apply(&mut state, sts2_core::Action::ResolveChoice { selection });
    let returned = state
        .player
        .hand
        .iter()
        .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::DefendNecrobinder)
        .count();
    assert_eq!(returned, 2, "each Graveblast returned one discard card");
    assert!(state.pending_choice.is_none());
}

/// GH #1054 (Defect-1000 act-3 `hand_costs` split): Snecko Oil rolled a
/// MomentumStrike to 0 (`SetThisTurnOrUntilPlayed`), then it was played.
/// C# appends MomentumStrike's `SetThisCombat(0)` AFTER the roll, so when the
/// roll expires on play the cost stays 0. Rust's roll expiry subtracted the
/// roll's delta from the new 0, so the card came back at 1.
#[test]
fn momentum_strike_keeps_zero_after_a_snecko_roll_expires() {
    let mut state = fresh(V111, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::MomentumStrike, false);
    let uid = state.player.hand[usize::from(i)];
    // The same banked modifier Snecko Oil's 0 roll leaves on a 1-cost card.
    state.cards[uid.0.get() as usize - 1].set_this_turn_or_until_played_reduce_only(0);
    assert_eq!(state.cards[uid.0.get() as usize - 1].cost, 0);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(
        state.cards[uid.0.get() as usize - 1].cost,
        0,
        "SetThisCombat(0) outlives the expired roll"
    );
}

// ── Sidestep ───────────────────────────────────────────────────────────────

#[test]
fn sidestep_v0111() {
    let mut state = fresh(V111, CharacterId::Silent);
    let i = push_hand(&mut state, CardId::Sidestep, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.energy, 5, "cost 0");
    assert_eq!(state.player_power_amount(PowerKind::EnergyNextTurn), 1);
    let j = push_hand(&mut state, CardId::Sidestep, true);
    let _ = sts2_core::combat_loop::play_card(&mut state, j, None);
    assert_eq!(
        state.player_power_amount(PowerKind::EnergyNextTurn),
        3,
        "upgraded +2"
    );
}

// ── Rocket Punch ───────────────────────────────────────────────────────────

#[test]
fn rocket_punch_cost_reduction_stacks_v0111() {
    let mut state = fresh(V111, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::RocketPunch, false);
    let uid = state.player.hand[usize::from(i)];
    let dazed = card_pod_from_id_for(CardId::Dazed, false, V111).expect("dazed");
    let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(&mut state, dazed, true);
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, uid),
        1,
        "2 - 1 until played (RocketPunch.cs:49 AddUntilPlayed(-1))"
    );
    let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(&mut state, dazed, true);
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, uid),
        0,
        "stacks"
    );
    let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(&mut state, dazed, true);
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, uid),
        0,
        "floored at 0"
    );
}

#[test]
fn rocket_punch_free_until_played_v0107() {
    let mut state = fresh(V107, CharacterId::Defect);
    let i = push_hand(&mut state, CardId::RocketPunch, false);
    let uid = state.player.hand[usize::from(i)];
    let dazed = card_pod_from_id_for(CardId::Dazed, false, V107).expect("dazed");
    let _ = sts2_core::card_effects::spawn_card_into_hand_with_origin(&mut state, dazed, true);
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, uid),
        0,
        "SetUntilPlayed(0)"
    );
    let card = state.cards[uid.0.get() as usize - 1];
    assert_eq!(
        card.combat_var(sts2_core::cards::VarKey::CostReductionUntilPlayed),
        0,
        "the v0.111.0 var is never set at v0.107.1"
    );
}

// ── Midnight / The Ball (MP-only, sweepable) ───────────────────────────────

#[test]
fn midnight_costs_one_less_per_exhaust_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::Midnight, false);
    let uid = state.player.hand[usize::from(i)];
    let fodder = new_card(&mut state, CardId::StrikeIronclad, false);
    state.player.hand.push(fodder);
    let _ = sts2_core::card_effects::exhaust_card_uid(&mut state, fodder);
    assert_eq!(state.cards[uid.0.get() as usize - 1].cost, 11);
}

#[test]
fn the_ball_grows_per_play_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::TheBall, false);
    let uid = state.player.hand[usize::from(i)];
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_hp(&state), 90);
    assert_eq!(state.cards[uid.0.get() as usize - 1].base_damage, 20);
}

// ── Abundance ──────────────────────────────────────────────────────────────

#[test]
fn abundance_offers_three_upgraded_powers_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let i = push_hand(&mut state, CardId::Abundance, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("a choose-a-card screen");
    assert_eq!(pc.options.len(), 3);
    assert!(
        !pc.can_skip,
        "FromChooseACardScreen defaults canSkip to false"
    );
    for opt in &pc.options {
        assert!(opt.upgraded);
        let spec = sts2_content::cards::spec_index::spec_for_card_id(opt.card_id).expect("spec");
        assert_eq!(spec.card_type, sts2_core::cards::CardType::Power);
    }
}

// ── Misery ─────────────────────────────────────────────────────────────────

#[test]
fn misery_folds_temporary_strength_v0111() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    let other = add_enemy(&mut state, 50, 100);
    let target = first_enemy(&state);
    // Target: drained by a Piercing Wail (Strength -6 + PiercingWail 6).
    add_enemy_power(&mut state, target, 800, PowerKind::Strength, -6);
    add_enemy_power(&mut state, target, 801, PowerKind::PiercingWail, 6);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_power(&state, other, PowerKind::PiercingWail), 6);
    assert_eq!(
        enemy_power(&state, other, PowerKind::Strength),
        -6,
        "Strength -6 + 6 folds to 0 and is skipped; the temporary power takes its 6 itself"
    );
}

/// A Strength-UP target with a temporary Strength-down: Strength reads as a
/// buff, so it is not snapshotted and nothing absorbs the temporary power's
/// amount — at v0.111.0 the copied PiercingWail takes its 6 Strength itself.
#[test]
fn misery_temporary_debuff_without_strength_entry_v0111() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    let other = add_enemy(&mut state, 50, 100);
    let target = first_enemy(&state);
    add_enemy_power(&mut state, target, 800, PowerKind::Strength, 3);
    add_enemy_power(&mut state, target, 801, PowerKind::PiercingWail, 6);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_power(&state, other, PowerKind::PiercingWail), 6);
    assert_eq!(enemy_power(&state, other, PowerKind::Strength), -6);
}

/// The same board at v0.107.1: `IgnoreNextInstance` swallows the copy's
/// Strength loss, so the other enemy keeps its Strength.
#[test]
fn misery_temporary_debuff_without_strength_entry_v0107() {
    let mut state = fresh(V107, CharacterId::Necrobinder);
    let other = add_enemy(&mut state, 50, 100);
    let target = first_enemy(&state);
    add_enemy_power(&mut state, target, 800, PowerKind::Strength, 3);
    add_enemy_power(&mut state, target, 801, PowerKind::PiercingWail, 6);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_power(&state, other, PowerKind::PiercingWail), 6);
    assert_eq!(enemy_power(&state, other, PowerKind::Strength), 0);
}

/// v0.107.1 Misery copies in C# `Powers` order, where a temporary debuff's
/// Strength precedes it (`BeforeApplied` runs before the apply). With one
/// Artifact on the other enemy, the Strength copy is the one negated and the
/// ShacklingPotion copy lands (Necrobinder full seed 2643).
#[test]
fn misery_artifact_negates_the_strength_copy_first_v0107() {
    let mut state = fresh(V107, CharacterId::Necrobinder);
    let other = add_enemy(&mut state, 50, 100);
    let target = first_enemy(&state);
    add_enemy_power(&mut state, target, 801, PowerKind::ShacklingPotion, 7);
    add_enemy_power(&mut state, target, 800, PowerKind::Strength, -7);
    add_enemy_power(&mut state, other, 802, PowerKind::Artifact, 1);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_power(&state, other, PowerKind::ShacklingPotion), 7);
    assert_eq!(enemy_power(&state, other, PowerKind::Strength), 0);
    assert_eq!(enemy_power(&state, other, PowerKind::Artifact), 0);
}

/// A monster's own Strength keeps the monster as its C# `Applier` even after
/// a player drain takes it negative (`PowerCmd.Apply` sets `Applier` only on
/// a fresh instance). Misery copies it with that applier, so Sleight of Flesh
/// (`applier == Owner`) reacts only to the player's Debilitate copy (loadout
/// Necrobinder 1210-v0.107.1: Rocket Strength 2 -> -7 under Dark Shackles;
/// C# Crusher 37 -> 28, Rust took a second Sleight of Flesh hit).
#[test]
fn misery_monster_strength_copy_skips_sleight_of_flesh() {
    for v in [V107, V111] {
        let mut state = fresh(v, CharacterId::Necrobinder);
        let other = add_enemy(&mut state, 50, 100);
        let target = first_enemy(&state);
        // The monster's own buff (Rocket CHARGE_UP), then a player drain.
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            target,
            PowerKind::Strength,
            2,
        );
        sts2_core::card_effects::apply_signed_attr_power(
            &mut state,
            target,
            PowerKind::Strength,
            -9,
        );
        assert_eq!(enemy_power(&state, target, PowerKind::Strength), -7);
        add_enemy_power(&mut state, target, 800, PowerKind::Debilitate, 2);
        add_player_power(&mut state, 801, PowerKind::SleightOfFlesh, 9);
        let i = push_hand(&mut state, CardId::Misery, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(enemy_power(&state, other, PowerKind::Strength), -7, "{v:?}");
        assert_eq!(
            enemy_power(&state, other, PowerKind::Debilitate),
            2,
            "{v:?}"
        );
        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == other.0)
            .expect("other")
            .current_hp;
        assert_eq!(hp, 91, "{v:?}: one Sleight of Flesh hit (Debilitate)");
    }
}

/// The mirror case: a Strength the PLAYER's drain created (no prior buff)
/// is player-applied, so its Misery copy fires Sleight of Flesh as well.
#[test]
fn misery_player_drained_strength_copy_fires_sleight_of_flesh() {
    let mut state = fresh(V107, CharacterId::Necrobinder);
    let other = add_enemy(&mut state, 50, 100);
    let target = first_enemy(&state);
    sts2_core::card_effects::apply_signed_attr_power(&mut state, target, PowerKind::Strength, -9);
    add_enemy_power(&mut state, target, 800, PowerKind::Debilitate, 2);
    add_player_power(&mut state, 801, PowerKind::SleightOfFlesh, 9);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    let hp = state
        .creatures
        .iter()
        .find(|c| c.id == other.0)
        .expect("other")
        .current_hp;
    assert_eq!(hp, 82, "Strength and Debilitate copies each hit");
}

// ── Dowsing quest (run layer) ──────────────────────────────────────────────

fn dowsing_run(version: GameVersion) -> sts2_core::runstate::RunState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut rs = sts2_core::runstate::RunState {
        game_version: version,
        ..sts2_core::runstate::RunState::default()
    };
    rs.master_deck
        .push(card_pod_from_id_for(CardId::StrikeSilent, false, version).expect("strike"));
    rs.master_deck
        .push(card_pod_from_id_for(CardId::Dowsing, false, version).expect("dowsing"));
    rs.master_deck
        .push(card_pod_from_id_for(CardId::DefendSilent, false, version).expect("defend"));
    rs
}

#[test]
fn dowsing_transforms_into_abundance_after_five_unknown_rooms_v0111() {
    let mut rs = dowsing_run(V111);
    for n in 1..5 {
        rs.tick_dowsing_quests_v0111();
        let d = rs
            .master_deck
            .iter()
            .find(|c| c.card_id == CardId::Dowsing)
            .expect("still a quest card");
        assert_eq!(d.combat_var(sts2_core::cards::VarKey::Generic), n);
    }
    rs.tick_dowsing_quests_v0111();
    let ids: Vec<CardId> = rs.master_deck.iter().map(|c| c.card_id).collect();
    assert_eq!(
        ids,
        vec![
            CardId::StrikeSilent,
            CardId::DefendSilent,
            CardId::Abundance
        ],
        "Dowsing leaves its slot; Abundance is appended at the deck tail (CardCmd.cs:416-447)"
    );
}

// ── ThinkingAhead waits for a Stratagem-suspended draw ─────────────────────

/// ThinkingAhead draws 2 into an empty draw pile: the reshuffle opens
/// Stratagem's grid INSIDE the draw. C# finishes the whole draw before its
/// hand put-back select (ThinkingAhead.cs OnPlay), so the grid surfaces first
/// and the put-back offers the full post-draw hand (Necrobinder full seed
/// 4009: Rust's put-back overwrote the grid and offered the pre-draw hand).
#[test]
fn thinking_ahead_put_back_waits_for_a_suspended_draw_v0107() {
    let mut state = fresh(V107, CharacterId::Necrobinder);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let _ = push_hand(&mut state, CardId::StrikeNecrobinder, false);
    for id in [CardId::Footwork, CardId::Accelerant, CardId::Backflip] {
        let uid = new_card(&mut state, id, false);
        state.player.discard.push(uid);
    }
    let i = push_hand(&mut state, CardId::ThinkingAhead, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state.pending_choice.as_ref().expect("a choice is open");
    assert_eq!(
        pc.kind,
        sts2_core::ChoiceKind::AddFromDrawToHand,
        "Stratagem first"
    );

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pc = state.pending_choice.as_ref().expect("the put-back follows");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::PutBack);
    assert_eq!(
        pc.options.len(),
        4,
        "Strike + the Stratagem pick + 2 drawn cards are offered"
    );
}

/// Glimmer has the same shape as ThinkingAhead: C# Glimmer.OnPlay awaits the
/// whole `CardPileCmd.Draw` (incl. Stratagem's reshuffle grid) before its
/// `CardSelectCmd.FromHand` put-back (Regent-3002 act2 parity: Rust's
/// put-back overwrote the grid and offered the partial hand).
#[test]
fn glimmer_put_back_waits_for_a_suspended_draw_v0111() {
    let mut state = fresh(V111, CharacterId::Regent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let _ = push_hand(&mut state, CardId::StrikeRegent, false);
    for id in [
        CardId::Footwork,
        CardId::Accelerant,
        CardId::Backflip,
        CardId::DefendRegent,
    ] {
        let uid = new_card(&mut state, id, false);
        state.player.discard.push(uid);
    }
    let i = push_hand(&mut state, CardId::Glimmer, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state.pending_choice.as_ref().expect("a choice is open");
    assert_eq!(
        pc.kind,
        sts2_core::ChoiceKind::AddFromDrawToHand,
        "Stratagem first"
    );

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pc = state.pending_choice.as_ref().expect("the put-back follows");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::PutBack);
    assert_eq!(
        pc.options.len(),
        5,
        "Strike + the Stratagem pick + 3 drawn cards are offered"
    );
}

/// Photon Cut has the Glimmer shape: C# PhotonCut.OnPlay awaits the whole
/// `CardPileCmd.Draw` (incl. Stratagem's reshuffle grid) before its
/// `CardSelectCmd.FromHand` put-back (Regent loadout v0.107.1 1016: Rust's
/// put-back overwrote the pending Stratagem select).
#[test]
fn photon_cut_put_back_waits_for_a_suspended_draw_v0107() {
    let mut state = fresh(V107, CharacterId::Regent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let _ = push_hand(&mut state, CardId::StrikeRegent, false);
    for id in [CardId::Footwork, CardId::Accelerant, CardId::Backflip] {
        let uid = new_card(&mut state, id, false);
        state.player.discard.push(uid);
    }
    let i = push_hand(&mut state, CardId::PhotonCut, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    let pc = state.pending_choice.as_ref().expect("a choice is open");
    assert_eq!(
        pc.kind,
        sts2_core::ChoiceKind::AddFromDrawToHand,
        "Stratagem first"
    );

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pc = state.pending_choice.as_ref().expect("the put-back follows");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::PutBack);
    assert_eq!(
        pc.options.len(),
        3,
        "Strike + the Stratagem pick + 1 drawn card are offered"
    );
}

/// `ParticleWall.GetResultPileTypeForCardPlay` (ParticleWall.cs:31-39) turns
/// the Discard result into Hand for AUTO-plays too (`OnPlayWrapper` asks the
/// card either way); a full hand overflows to Discard (Regent loadout v0.107.1
/// 1012, v0.111.0 1020: an auto-played Particle Wall went to discard).
#[test]
fn auto_played_particle_wall_returns_to_hand() {
    for version in [V107, V111] {
        for full_hand in [false, true] {
            let mut state = fresh(version, CharacterId::Regent);
            if full_hand {
                for _ in 0..sts2_core::cards::HAND_LIMIT {
                    let _ = push_hand(&mut state, CardId::StrikeRegent, false);
                }
            }
            let wall = new_card(&mut state, CardId::ParticleWall, false);
            state.player.draw.push(wall);
            sts2_core::combat_loop::auto_play_specific_card(&mut state, wall, false);
            assert_eq!(
                state.player.hand.contains(&wall),
                !full_hand,
                "{version:?} full_hand={full_hand}"
            );
            assert_eq!(
                state.player.discard.contains(&wall),
                full_hand,
                "{version:?} full_hand={full_hand}"
            );
        }
    }
}

/// Prepared draws 1 into an empty draw pile: the reshuffle opens Stratagem's
/// grid INSIDE the draw. C# awaits the whole draw before
/// `FromHandForDiscard` (Prepared.cs:23-24), so the discard select must
/// still open after the grid resolves (Silent/Defect v0.111.0 full seed 1475:
/// Rust dropped the discard because a choice was already open).
#[test]
fn prepared_discard_waits_for_a_suspended_draw_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let _ = push_hand(&mut state, CardId::StrikeSilent, false);
    let _ = push_hand(&mut state, CardId::DefendSilent, false);
    for id in [CardId::Footwork, CardId::Accelerant, CardId::Backflip] {
        let uid = new_card(&mut state, id, false);
        state.player.discard.push(uid);
    }
    let i = push_hand(&mut state, CardId::Prepared, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state.pending_choice.as_ref().expect("a choice is open");
    assert_eq!(
        pc.kind,
        sts2_core::ChoiceKind::AddFromDrawToHand,
        "Stratagem first"
    );

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    let pc = state.pending_choice.as_ref().expect("the discard follows");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::DiscardFromHand);
    assert_eq!(
        pc.options.len(),
        4,
        "Strike + Defend + the Stratagem pick + the drawn card are offered"
    );
    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert!(state.pending_choice.is_none());
    assert_eq!(state.player.hand.len(), 3, "one card discarded");
}

/// Catastrophe auto-plays a lone Slimed whose own draw reshuffles a 2-card
/// discard and opens Stratagem's grid. C# awaits Slimed's whole draw inside
/// its `OnPlay` (Slimed.cs:33) before Catastrophe's next pick, so the draw
/// takes the last card and the second pick finds an empty pile. Rust used to
/// run the pick first and resume the draw afterwards (Defect v0.111.0 full
/// seed 1475).
#[test]
fn catastrophe_autoplayed_draw_finishes_before_next_pick_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let slimed = new_card(&mut state, CardId::Slimed, false);
    state.player.draw.push(slimed);
    let a = new_card(&mut state, CardId::DefendSilent, false);
    let b = new_card(&mut state, CardId::Footwork, false);
    state.player.discard.push(a);
    state.player.discard.push(b);
    let i = push_hand(&mut state, CardId::Catastrophe, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert!(state.pending_choice.is_none());
    assert!(
        state.player.hand.contains(&a) && state.player.hand.contains(&b),
        "the Stratagem pick and Slimed's draw both land in hand"
    );
    assert!(state.player.draw.is_empty());
}

// ── Draw-pile grids sort by the build's rarity (review finding a) ─────────

/// Stratagem's reshuffle grid with `[Footwork, Accelerant]` in the discard:
/// the option-0 card. `CardSelectCmd.FromCombatPile` sorts the Draw pile
/// `orderby c.Rarity, c.Id` (v0.111.0 Commands/CardSelectCmd.cs:490-505) on
/// the LIVE rarity, and v0.111.0 made Accelerant Uncommon (Accelerant.cs:21),
/// so it ties with Footwork and wins on the Id.
fn stratagem_grid_first_option(version: GameVersion) -> CardId {
    let mut state = fresh(version, CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    for id in [CardId::Footwork, CardId::Accelerant] {
        let uid = new_card(&mut state, id, false);
        state.player.discard.push(uid);
    }
    let _ = state.draw_into_hand(1);
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    pc.options[0].card_id
}

#[test]
fn stratagem_grid_sorts_by_build_rarity_v0111() {
    assert_eq!(stratagem_grid_first_option(V111), CardId::Accelerant);
}

#[test]
fn stratagem_grid_sorts_by_build_rarity_v0107() {
    // v0.107.1: Accelerant is Rare (4), Footwork Uncommon (3).
    assert_eq!(stratagem_grid_first_option(V107), CardId::Footwork);
}

/// The core rarity-rank hook ranks by the build's rarity: Outbreak is Rare
/// in v0.111.0 (Outbreak.cs:23), Uncommon in v0.107.1.
#[test]
fn rarity_rank_hook_is_build_aware_v0111() {
    content_dispatch::ensure_installed();
    let rank = sts2_core::lookup_card_rarity_rank().expect("registered");
    assert_eq!(rank(CardId::OutbreakCard, V111), 4);
    assert_eq!(rank(CardId::OutbreakCard, V107), 3);
    assert_eq!(rank(CardId::Accelerant, V111), 3);
    assert_eq!(rank(CardId::Accelerant, V107), 4);
}

// ── Upgrade sites use the build's delta (review finding b) ───────────────

/// Play `id` (un-upgraded) holding RazorTooth, which upgrades the played
/// Attack/Skill (`RazorTooth.cs`); return the card afterwards.
fn razor_tooth_upgraded(version: GameVersion, id: CardId) -> sts2_core::cards::Card {
    let mut state = fresh(version, CharacterId::Defect);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::RazorTooth,
        counter: sts2_core::RelicKind::RazorTooth.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
    let i = push_hand(&mut state, id, false);
    let uid = state.player.hand[usize::from(i)];
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    state.cards[uid.0.get() as usize - 1]
}

#[test]
fn razor_tooth_upgrades_with_build_delta_v0111() {
    // Hyperbeam: 24 + 6 (Hyperbeam.cs:61), not v0.107.1's +8.
    let hb = razor_tooth_upgraded(V111, CardId::Hyperbeam);
    assert!(hb.upgraded);
    assert_eq!(hb.base_damage, 30);
    // Expect a Fight: the v0.111.0 upgrade no longer lowers the cost
    // (ExpectAFight.cs:39-43).
    let eaf = razor_tooth_upgraded(V111, CardId::ExpectAFight);
    assert!(eaf.upgraded);
    assert_eq!(eaf.cost, 3);
}

#[test]
fn razor_tooth_upgrades_with_build_delta_v0107() {
    let hb = razor_tooth_upgraded(V107, CardId::Hyperbeam);
    assert_eq!(hb.base_damage, 36);
    let eaf = razor_tooth_upgraded(V107, CardId::ExpectAFight);
    assert_eq!(eaf.cost, 1);
}

/// C# `CardCmd.Upgrade` returns at once while `CombatManager.IsEnding`
/// (CardCmd.cs:265-270, both builds), so Razor Tooth leaves the card that
/// killed the last enemy un-upgraded.
#[test]
fn razor_tooth_skips_the_card_that_ends_the_combat() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect);
        state.creatures[0].current_hp = 1;
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::RazorTooth,
            counter: sts2_core::RelicKind::RazorTooth.initial_counter(),
            is_wax: false,
            is_melted: false,
        });
        let i = push_hand(&mut state, CardId::Hyperbeam, false);
        let uid = state.player.hand[usize::from(i)];
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert!(enemy_hp(&state) <= 0, "{version:?}: the Hyperbeam kills");
        assert!(
            !state.cards[uid.0.get() as usize - 1].upgraded,
            "{version:?}: no upgrade while the combat is ending"
        );
    }
}

/// Adaptive Strike's copy is `CreateClone()` + `EnergyCost.SetThisCombat(0)`
/// (AdaptiveStrike.cs:24-27): the Absolute 0 folds after every modifier the
/// clone copied, so it still costs 0 once a copied this-turn cap expires.
/// Rust wrote a raw `cost = 0` under Enlightenment's cap (2 -> 1 this turn),
/// whose turn-end expiry then added the 1 back (loadout Defect 1035).
#[test]
fn adaptive_strike_copy_stays_free_after_a_copied_turn_cap_expires() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect);
        let i = push_hand(&mut state, CardId::AdaptiveStrike, false);
        let original = state.player.hand[usize::from(i)];
        state.cards[original.0.get() as usize - 1].set_this_turn_or_until_played_reduce_only(1);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        let copy = *state
            .player
            .discard
            .iter()
            .find(|&&u| {
                u != original
                    && state.cards[u.0.get() as usize - 1].card_id == CardId::AdaptiveStrike
            })
            .expect("the copy lands in the discard pile");
        let card = &mut state.cards[copy.0.get() as usize - 1];
        assert_eq!(card.cost, 0, "{version:?}");
        // The turn-end cleanup expires the copied cap.
        card.expire_snecko_roll();
        assert_eq!(
            card.cost, 0,
            "{version:?}: SetThisCombat(0) still owns the cost"
        );
    }
}

/// Up My Sleeve's `EnergyCost.AddThisCombat(-1)` (UpMySleeve.cs:46) is an
/// unclamped Relative: three plays take the 2-cost card to -1, shown as 0, so
/// a later +1 leaves it at 0. Rust clamped at 0 on every play.
#[test]
fn up_my_sleeve_cost_reduction_stays_unclamped() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        state.player.energy = 9;
        let i = push_hand(&mut state, CardId::UpMySleeve, false);
        let ums = state.player.hand[usize::from(i)];
        for _ in 0..3 {
            state.player.hand.clear();
            state.player.discard.retain(|u| *u != ums);
            state.player.hand.push(ums);
            let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        }
        let card = &mut state.cards[ums.0.get() as usize - 1];
        assert_eq!(card.cost, 0, "{version:?}: shown as 0");
        card.add_cost_this_combat(1);
        assert_eq!(card.cost, 0, "{version:?}: -1 + 1 = 0, not 0 + 1");
    }
}

/// Pinpoint's `AddThisTurn(-1)` per Skill (Pinpoint.cs:46-58) is an
/// unclamped Relative local modifier, so the global +1 of Tangled lands on
/// the negative local cost: after four Skills an Entangled Pinpoint (3)
/// reads `max(0, 3 - 4) = 0`, not `max(0, 3 - 3) + 1`. Rust stopped the
/// discount at 0 (loadout Silent 1056 v0.111.0: cs=0 rs=1). The rest clears
/// at turn start with the bank.
#[test]
fn entangled_pinpoint_keeps_its_reduction_below_zero() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        state.player.energy = 9;
        add_player_power(&mut state, 60, PowerKind::Tangled, 1);
        let p = push_hand(&mut state, CardId::Pinpoint, false);
        let pinpoint = state.player.hand[usize::from(p)];
        // Tangled's `AfterApplied` Entangles the player's Attacks.
        state.cards[pinpoint.0.get() as usize - 1].affliction =
            Some(sts2_core::card_afflictions::CardAffliction {
                kind: sts2_core::card_afflictions::CardAfflictionKind::Entangled,
                stacks: 1,
                applied_exhaust: false,
            });
        for _ in 0..4 {
            let i = push_hand(&mut state, CardId::Deflect, false);
            let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        }
        assert_eq!(
            sts2_core::hooks::effective_card_cost(&state, pinpoint),
            0,
            "{version:?}: 3 - 4 skills, then Tangled +1, clamped at the end"
        );
        state.restore_pinpoint_costs_for_turn_start();
        let card = &state.cards[pinpoint.0.get() as usize - 1];
        assert_eq!(card.cost, 3, "{version:?}: next turn the base is back");
        assert_eq!(
            card.combat_var(sts2_core::cards::VarKey::CostDebtThisTurn),
            0,
            "{version:?}"
        );
    }
}

/// Upgraded Armaments upgrades the hand through `CardCmd.Upgrade` after its
/// block (Armaments.cs:22-31). When Juggernaut's reaction to that block kills
/// the last enemy, the combat is ending and nothing is upgraded.
#[test]
fn upgraded_armaments_upgrades_nothing_once_juggernaut_ends_the_combat() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Ironclad);
        state.creatures[0].current_hp = 1;
        add_player_power(&mut state, 50, PowerKind::Juggernaut, 5);
        let strike = new_card(&mut state, CardId::StrikeIronclad, false);
        state.player.hand.push(strike);
        let i = push_hand(&mut state, CardId::Armaments, true);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert!(enemy_hp(&state) <= 0, "{version:?}: Juggernaut kills");
        assert!(
            !state.cards[strike.0.get() as usize - 1].upgraded,
            "{version:?}: no upgrade while the combat is ending"
        );
    }
}

/// C# `OnPlayWrapper` fixes the result pile before `OnPlay`, so an
/// auto-played Know Thy Place that Razor Tooth upgrades in `AfterCardPlayed`
/// (the upgrade drops Exhaust) still exhausts (loadout Regent 1036, Whispering
/// Earring auto-play).
#[test]
fn razor_tooth_upgrade_keeps_an_auto_played_exhaust_card_exhausting() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::RazorTooth,
            counter: sts2_core::RelicKind::RazorTooth.initial_counter(),
            is_wax: false,
            is_melted: false,
        });
        let uid = new_card(&mut state, CardId::KnowThyPlace, false);
        state.player.draw.push(uid);
        assert!(sts2_core::combat_loop::auto_play_card_from_draw_top(
            &mut state, false
        ));
        assert!(
            state.cards[uid.0.get() as usize - 1].upgraded,
            "{version:?}"
        );
        assert!(
            state.player.exhaust.contains(&uid),
            "{version:?}: exhausted, not discarded"
        );
    }
}

// ── Normalizing a combat-set cost ──────────────────────────────────────────

/// A Confused roll (`EnergyCost.SetThisCombat`, ConfusedPower.cs:42) that
/// lands on BeaconOfHope's v0.107.1 cost (1) survives an in-combat upgrade
/// (Bellows → `upgrade_for`) under v0.111.0; only a still-v0.107.1-minted
/// cost is re-based to the build's 2 (Ironclad mid seed 1809).
#[test]
fn normalize_keeps_a_confused_cost_v0111() {
    content_dispatch::ensure_installed();
    let mut rolled = card_pod_from_id_for(CardId::BeaconOfHope, false, V111).expect("pod");
    assert_eq!(rolled.cost, 2, "BeaconOfHope.cs:18");
    rolled.cost = 1;
    rolled.set_combat_var(sts2_core::cards::VarKey::CostAbsolutelySet, 1);
    assert!(rolled.upgrade_for(V111));
    assert_eq!(rolled.cost, 1, "the absolute roll is kept");

    let mut minted = card_pod_from_id_for(CardId::BeaconOfHope, false, V107).expect("pod");
    assert_eq!(minted.cost, 1);
    minted.normalize_for_version(V111);
    assert_eq!(minted.cost, 2, "a v0.107.1-minted base cost is re-based");
}

// ── BeatDown ───────────────────────────────────────────────────────────────

/// BeatDown auto-plays a Smog-afflicted Strike out of discard while the
/// player holds Smoggy, so `ShouldPlay` vetoes the play. Returns the
/// `combat_targets` reads and the enemies' HP lost.
fn beat_down_vetoed_targets_read(version: GameVersion) -> (u32, i32) {
    let mut state = fresh(version, CharacterId::Ironclad);
    let _ = add_enemy(&mut state, 9, 100);
    add_player_power(&mut state, 50, PowerKind::Smoggy, 1);
    let strike = new_card(&mut state, CardId::StrikeIronclad, false);
    state.cards[strike.0.get() as usize - 1].affliction =
        Some(sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Smog,
            1,
        ));
    state.player.discard.push(strike);
    let i = push_hand(&mut state, CardId::BeatDown, false);
    let before = state.run_rng.combat_targets.counter();
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    let enemies = state.enemy_uids.clone();
    let lost = 200 - enemies.iter().map(|u| hp_of(&state, *u)).sum::<i32>();
    (state.run_rng.combat_targets.counter() - before, lost)
}

/// BeatDown rolls the `AnyEnemy` target before `CardCmd.AutoPlay` (decomp
/// Cards/BeatDown.cs:32-35, the same in both builds), so the read happens
/// though the play is vetoed. Parity fuzz Regent seed 272 (ChainsOfBinding's
/// Bound, GH #798); random-loadout Ironclad v0.107.1 seeds 1027 / 1120.
#[test]
fn beat_down_vetoed_play_still_rolls_target() {
    assert_eq!(beat_down_vetoed_targets_read(V107), (1, 0));
    assert_eq!(beat_down_vetoed_targets_read(V111), (1, 0));
}

/// Catastrophe auto-plays BeatDown, whose first Dagger Throw opens a discard
/// select that surfaces; the second Dagger Throw, played once it is answered,
/// kills the enemy, and BeatDown's loop then sees the combat over. BeatDown's `OnPlayWrapper` still fires `AfterCardPlayed`
/// and routes it to Discard: that is gated on `CombatManager.IsInProgress`
/// only, which stays true while the combat is ending (CardModel.cs, both
/// builds). So Tuning Fork counts BeatDown and Catastrophe, two Skills
/// (Silent loadout v0.111.0 seed 1030).
#[test]
fn beat_down_resumed_after_a_lethal_select_still_fires_its_tail() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        state.creatures[0].current_hp = 18;
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::TuningFork,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let i = push_hand(&mut state, CardId::Catastrophe, false);
        for _ in 0..4 {
            let _ = push_hand(&mut state, CardId::DefendSilent, false);
        }
        // The only playable draw card is BeatDown, so Catastrophe's pick is fixed.
        let beat_down = new_card(&mut state, CardId::BeatDown, false);
        state.player.draw.push(beat_down);
        push_draw(&mut state, CardId::Wound);
        // BeatDown takes all three; the second kills, and the third's
        // `IsOverOrEnding` check ends the loop with the combat over.
        for _ in 0..3 {
            let dt = new_card(&mut state, CardId::DaggerThrow, false);
            state.player.discard.push(dt);
        }

        let step = sts2_core::apply(
            &mut state,
            sts2_core::Action::PlayCard {
                hand_index: i,
                target: None,
            },
        );
        assert_eq!(
            step,
            sts2_core::StepResult::PendingChoice,
            "{version:?}: the first Dagger Throw's discard select surfaces"
        );
        let pick = sts2_core::legal_actions(&state)
            .into_iter()
            .find_map(|a| match a {
                sts2_core::Action::ResolveChoice { selection } if !selection.is_empty() => {
                    Some(selection)
                }
                _ => None,
            })
            .expect("a discard pick");
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::ResolveChoice { selection: pick },
        );
        assert!(state.is_over, "{version:?}: the second Dagger Throw kills");
        assert_eq!(
            state.relics[0].counter, 2,
            "{version:?}: BeatDown and Catastrophe both fire AfterCardPlayed"
        );
        // Its move to Discard is refused while the combat is ending
        // (`CardPileCmd.Add`, CardPileCmd.cs:312), so it stays in Play.
        assert!(
            state.player.play.contains(&beat_down),
            "{version:?}: BeatDown stays in Play"
        );
    }
}

/// The Strike BeatDown auto-plays into the last enemy's death stays in Play:
/// `CardPileCmd.Add` refuses a combat pile once the combat is ending
/// (CardPileCmd.cs:312, both builds). Under Echo Form the v0.107.1 replay of
/// BeatDown then shuffles the two Strikes left in the discard (one `Shuffle`
/// read), not three (loadout Defect 1195 v0.107.1: the extra read desynced the
/// next combat's opening shuffle).
#[test]
fn beat_down_leaves_its_lethal_auto_play_in_play() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Ironclad);
        state.creatures[0].current_hp = 6;
        add_player_power(&mut state, 50, PowerKind::EchoForm, 1);
        let strikes: Vec<_> = (0..3)
            .map(|_| {
                let s = new_card(&mut state, CardId::StrikeIronclad, false);
                state.player.discard.push(s);
                s
            })
            .collect();
        let i = push_hand(&mut state, CardId::BeatDown, false);
        let before = state.run_rng.shuffle.counter();
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);

        let in_play: Vec<_> = strikes
            .iter()
            .filter(|s| state.player.play.contains(s))
            .collect();
        assert_eq!(
            in_play.len(),
            1,
            "{version:?}: the lethal Strike stays in Play"
        );
        assert_eq!(
            strikes
                .iter()
                .filter(|s| state.player.discard.contains(s))
                .count(),
            2,
            "{version:?}: the two Strikes never played stay in the discard"
        );
        if version == V107 {
            assert_eq!(
                state.run_rng.shuffle.counter() - before,
                3,
                "three Strikes shuffled (2 reads), then the Echo Form replay's two (1 read)"
            );
        }
    }
}

/// A Replay 2 BeatDown (Hidden Gem) whose first auto-play kills the last
/// enemy. v0.107.1's play loop only stops when the player dies, so both
/// replays still shuffle the discard: three Strikes (2 reads), then the two
/// left once the lethal one stays in Play (1 read each). v0.111.0 breaks at
/// the top of the loop once the combat is ending (loadout Regent 1176
/// v0.107.1: Rust skipped the third play, C# `Shuffle` 169 vs Rust 167).
#[test]
fn replayed_beat_down_keeps_replaying_after_a_lethal_auto_play() {
    for (version, reads) in [(V107, 4), (V111, 2)] {
        let mut state = fresh(version, CharacterId::Ironclad);
        state.creatures[0].current_hp = 6;
        for _ in 0..3 {
            let s = new_card(&mut state, CardId::StrikeIronclad, false);
            state.player.discard.push(s);
        }
        let i = push_hand(&mut state, CardId::BeatDown, false);
        let uid = state.player.hand[usize::from(i)];
        state.cards[uid.0.get() as usize - 1].base_replay_count = 2;
        let before = state.run_rng.shuffle.counter();
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert!(state.is_over, "{version:?}: the first Strike kills");
        assert_eq!(
            state.run_rng.shuffle.counter() - before,
            reads,
            "{version:?}: Shuffle reads across BeatDown's plays"
        );
    }
}

// ── Havoc + Howl from Beyond: a blocked forced-exhaust play ───────────────

/// Havoc auto-plays a Smog-afflicted Howl from Beyond under Smoggy, so
/// `ShouldPlay` vetoes it: `MoveToResultPileWithoutPlaying` exhausts it and
/// leaves `ExhaustOnNextPlay` set (CardModel.cs:2104-2120). At turn end Howl
/// auto-plays out of the exhaust pile, is vetoed again, and so exhausts again
/// instead of landing in discard. Random-loadout Ironclad v0.111.0 seed 1148.
fn havoc_blocked_howl_pile_after_turn_end(version: GameVersion) {
    let mut state = fresh(version, CharacterId::Ironclad);
    add_player_power(&mut state, 50, PowerKind::Smoggy, 1);
    let howl = new_card(&mut state, CardId::HowlFromBeyond, false);
    state.cards[howl.0.get() as usize - 1].affliction =
        Some(sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Smog,
            1,
        ));
    state.player.draw.push(howl);
    let i = push_hand(&mut state, CardId::Havoc, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert!(
        state.player.exhaust.contains(&howl),
        "Havoc exhausts the vetoed Howl"
    );
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert!(
        state.player.exhaust.contains(&howl),
        "the vetoed end-of-turn Howl auto-play exhausts again ({version:?})"
    );
    assert!(!state.player.discard.contains(&howl));
    assert_eq!(
        state.cards[howl.0.get() as usize - 1]
            .combat_var(sts2_core::cards::VarKey::ExhaustOnNextPlay),
        0,
        "EndOfTurnCleanup clears the flag"
    );
}

#[test]
fn havoc_blocked_howl_keeps_exhaust_on_next_play() {
    havoc_blocked_howl_pile_after_turn_end(V107);
    havoc_blocked_howl_pile_after_turn_end(V111);
}

// ── Generated / folded card values follow the build ───────────────────────

/// Stoke+ upgrades the cards it generates: each must carry THIS build's
/// upgraded values (v0.111.0 Taunt+ blocks 7, not v0.107.1's 8 — Taunt.cs:20,
/// 41). Ironclad v0.111.0 full seed 1728.
#[test]
fn stoke_plus_generated_cards_take_build_upgraded_values_v0111() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    for _ in 0..6 {
        let _ = push_hand(&mut state, CardId::StrikeIronclad, false);
    }
    let i = push_hand(&mut state, CardId::Stoke, true);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.hand.len(), 6);
    for uid in state.player.hand.clone() {
        let c = &state.cards[uid.0.get() as usize - 1];
        assert!(c.upgraded);
        let want = card_pod_from_id_for(c.card_id, true, V111).expect("spec");
        assert_eq!(
            (c.base_damage, c.base_block),
            (want.base_damage, want.base_block),
            "{:?}",
            c.card_id
        );
    }
}

/// Thrash folds an exhausted Rend's `CalculatedDamage.Calculate(null)` =
/// its CalculationBase: 15 in v0.107.1, 10 in v0.111.0 (Rend.cs:19). Ironclad
/// v0.111.0 full seed 4645.
fn thrash_rend_fold(version: GameVersion) -> i32 {
    let mut state = fresh(version, CharacterId::Ironclad);
    let _ = push_hand(&mut state, CardId::Rend, false);
    let i = push_hand(&mut state, CardId::Thrash, false);
    let thrash = state.player.hand[usize::from(i)];
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    state.cards[thrash.0.get() as usize - 1].base_damage - 4
}

#[test]
fn thrash_folds_rend_calculation_base_v0111() {
    assert_eq!(thrash_rend_fold(V111), 10);
}

#[test]
fn thrash_folds_rend_calculation_base_v0107() {
    assert_eq!(thrash_rend_fold(V107), 15);
}

/// Escape Plan into an empty draw pile with one Skill in the discard: the
/// reshuffle's Stratagem auto-takes that Skill into hand, then the draw
/// finds nothing. C# tests the card `Draw` RETURNED (EscapePlan.cs:25-26), so
/// no block. Silent v0.111.0 full seed 2548.
#[test]
fn escape_plan_ignores_a_stratagem_pick_v0107() {
    let mut state = fresh(V107, CharacterId::Silent);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let skill = new_card(&mut state, CardId::DefendSilent, false);
    state.player.discard.push(skill);
    let i = push_hand(&mut state, CardId::EscapePlan, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert!(state.player.hand.contains(&skill), "Stratagem took it");
    assert_eq!(state.player.block, 0);
}

/// Sweeping Gaze with no hittable enemy (an Adaptable corpse keeps the combat
/// open): C# still executes the Osty's `AttackCommand`, so `BoneFlute.AfterAttack`
/// grants its 2 block (act23 Necrobinder:1006).
#[test]
fn sweeping_gaze_with_no_target_still_owes_bone_flute() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    state.creatures[0].current_hp = 0;
    let enemy = state.enemy_uids[0];
    sts2_core::card_effects::apply_power_to_creature(&mut state, enemy, PowerKind::Adaptable, 1);
    let _ = state.summon_osty(5);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::BoneFlute,
        counter: sts2_core::RelicKind::BoneFlute.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
    let i = push_hand(&mut state, CardId::SweepingGaze, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.block, 2);
}

/// The same no-hit Osty `AttackCommand` still runs `Hook.AfterAttack`, so
/// `Flatten.AfterAttack` zeroes every Flatten for the turn. Random-loadout
/// Necrobinder v0.111.0 seed 3233: a reviving Test Subject, hand Flatten
/// C# 0, Rust 2.
#[test]
fn sweeping_gaze_with_no_target_still_zeroes_flatten() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    state.creatures[0].current_hp = 0;
    let enemy = state.enemy_uids[0];
    sts2_core::card_effects::apply_power_to_creature(&mut state, enemy, PowerKind::Adaptable, 1);
    let _ = state.summon_osty(5);
    let _ = push_hand(&mut state, CardId::Flatten, false);
    let flatten = *state.player.hand.last().expect("flatten");
    assert!(sts2_core::hooks::effective_card_cost(&state, flatten) > 0);
    let i = push_hand(&mut state, CardId::SweepingGaze, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, flatten), 0);
}

/// C# grants Escape Plan's block only for a drawn `CardType.Skill`; the Quest
/// cards (modelled as Skills in Rust) are `CardType.Quest`, so drawing
/// Byrdonis Egg grants nothing (act23 Silent:1017).
#[test]
fn escape_plan_drawing_a_quest_card_grants_no_block() {
    let mut state = fresh(V111, CharacterId::Silent);
    let egg = new_card(&mut state, CardId::ByrdonisEgg, false);
    state.player.draw.push(egg);
    let i = push_hand(&mut state, CardId::EscapePlan, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert!(state.player.hand.contains(&egg), "the egg was drawn");
    assert_eq!(state.player.block, 0);
}

// ── Pen Nib + a lethal end-of-turn Howl from Beyond ───────────────────────

/// A replayed Howl from Beyond auto-plays out of the exhaust pile at turn end
/// (`HowlFromBeyond.AfterAutoPostPlayPhaseEntered`) and its first copy kills
/// the last enemy. Pen Nib counts that copy (`PenNib.BeforeCardPlayed`):
/// 8 -> 9. v0.107.1 still runs the replay, but its `Hook.BeforeCardPlayed`
/// reaches no listener once combat is ending (Hook.cs:200); v0.111.0 breaks
/// out of the play loop. Random-loadout Ironclad v0.107.1 seed 1066.
fn pen_nib_after_lethal_howl(version: GameVersion) -> i32 {
    let mut state = fresh(version, CharacterId::Ironclad);
    state.creatures[0].current_hp = 5;
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::PenNib,
        counter: 8,
        is_wax: false,
        is_melted: false,
    });
    let howl = new_card(&mut state, CardId::HowlFromBeyond, false);
    state.cards[howl.0.get() as usize - 1].base_replay_count = 1;
    state.player.exhaust.push(howl);
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert!(state.is_over, "the Howl is lethal ({version:?})");
    state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::PenNib)
        .expect("pen nib")
        .counter
}

#[test]
fn lethal_howl_counts_once_for_pen_nib() {
    assert_eq!(pen_nib_after_lethal_howl(V107), 9);
    assert_eq!(pen_nib_after_lethal_howl(V111), 9);
}

// ── Unsettling Lamp ignores Hand Drill's cardless Vulnerable ──────────────

/// A Strike breaks the enemy's block, so Hand Drill applies Vulnerable 2 with
/// `cardSource: null` (HandDrill.cs): Unsettling Lamp bails on a null source
/// (UnsettlingLamp.cs `BeforePowerAmountChanged`), so it neither doubles nor
/// spends its trigger. The next card's debuff (Bash's Vulnerable 2) is the
/// one doubled. Random-loadout Ironclad v0.111.0 seed 1117.
fn lamp_vulnerable_after_hand_drill(version: GameVersion) -> (i32, i32) {
    let mut state = fresh(version, CharacterId::Ironclad);
    state.creatures[0].block = 3;
    for (raw, kind) in [
        (1, sts2_core::RelicKind::HandDrill),
        (2, sts2_core::RelicKind::UnsettlingLamp),
    ] {
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(raw).expect("nz"),
            kind,
            counter: kind.initial_counter(),
            is_wax: false,
            is_melted: false,
        });
    }
    let enemy = state.enemy_uids[0];
    let i = push_hand(&mut state, CardId::StrikeIronclad, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    let after_drill = enemy_power(&state, enemy, PowerKind::Vulnerable);
    let i = push_hand(&mut state, CardId::Bash, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    (
        after_drill,
        enemy_power(&state, enemy, PowerKind::Vulnerable),
    )
}

#[test]
fn unsettling_lamp_ignores_hand_drill_vulnerable() {
    assert_eq!(lamp_vulnerable_after_hand_drill(V107), (2, 6));
    assert_eq!(lamp_vulnerable_after_hand_drill(V111), (2, 6));
}

// ── Two exhausted Howls from Beyond, the first lethal ─────────────────────

/// Each Howl from Beyond in the exhaust pile auto-plays at turn end through
/// `CardCmd.AutoPlay`, which returns at once when `IsOverOrEnding`
/// (CardCmd.cs:54, both builds). The first Howl kills the last enemy, so the
/// second never plays and Pen Nib counts once (8 -> 9). Rust's loop broke
/// only on `is_over`, which the kill does not set until `check_combat_end`, so
/// the second Howl played and Pen Nib wrapped. Random-loadout Ironclad
/// v0.111.0 seed 1048.
fn pen_nib_after_two_exhausted_howls(version: GameVersion) -> i32 {
    let mut state = fresh(version, CharacterId::Ironclad);
    state.creatures[0].current_hp = 5;
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::PenNib,
        counter: 8,
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..2 {
        let howl = new_card(&mut state, CardId::HowlFromBeyond, false);
        state.player.exhaust.push(howl);
    }
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert!(state.is_over, "the first Howl is lethal ({version:?})");
    state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::PenNib)
        .expect("pen nib")
        .counter
}

#[test]
fn second_exhausted_howl_does_not_play_after_a_lethal_first() {
    assert_eq!(pen_nib_after_two_exhausted_howls(V107), 9);
    assert_eq!(pen_nib_after_two_exhausted_howls(V111), 9);
}

// ── Metamorphosis whose own generations end the combat ────────────────────

/// Metamorphosis adds its attacks one at a time through
/// `AddGeneratedCardToCombat(.., Draw, .., Random)`. Each add draws one
/// `Shuffle` read for the slot, then Regalite's block fires Juggernaut. Once
/// that kills the last enemy, `CardPileCmd.Add` returns at once on
/// `CombatManager.IsEnding` (CardPileCmd.cs:312 v0.107.1, :379 v0.111.0), so
/// the remaining cards draw no slot and never reach the draw pile. Rust kept
/// inserting, one extra `Shuffle` read that desynced the next combat's
/// opening shuffle. Random-loadout Ironclad v0.107.1 seed 1001.
fn metamorphosis_lethal_mid_generation(version: GameVersion) -> (u32, usize) {
    let mut state = fresh(version, CharacterId::Ironclad);
    // The first Juggernaut hit (6) kills it; the other two generations are
    // post-lethal. (v0.111.0 Regalite fires once per turn, so one hit is all
    // both builds share.)
    state.creatures[0].current_hp = 6;
    add_player_power(&mut state, 50, PowerKind::Juggernaut, 6);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::Regalite,
        counter: sts2_core::RelicKind::Regalite.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
    for _ in 0..5 {
        push_draw(&mut state, CardId::DefendIronclad);
    }
    let i = push_hand(&mut state, CardId::Metamorphosis, false);
    let before = state.run_rng.shuffle.counter();
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert!(
        sts2_core::combat_loop::combat_is_ending(&state),
        "the first Juggernaut hit is lethal ({version:?})"
    );
    (
        state.run_rng.shuffle.counter() - before,
        state.player.draw.len(),
    )
}

#[test]
fn metamorphosis_stops_adding_once_its_generations_end_the_combat() {
    assert_eq!(metamorphosis_lethal_mid_generation(V107), (1, 6));
    assert_eq!(metamorphosis_lethal_mid_generation(V111), (1, 6));
}

// ── Potion-generated cards are player-created ─────────────────────────────

/// CunningPotion / PotOfGhouls generate through `Shiv.CreateInHand` /
/// `Soul.CreateInHand` → `AddGeneratedCardsToCombat(.., creator ?? owner)`
/// (`Shiv.cs:103`, `Soul.cs`, both builds), which raises
/// `AfterCardGeneratedForCombat` once per card with the player as creator.
/// A drink leaves the Play pile empty, and the spawn path's play-pile
/// heuristic used to read that as a non-player generation: Regalite gave no
/// Block and Supermassive's tally stayed at 0. Regalite is +2 per card in
/// v0.107.1 and +4 once per turn in v0.111.0 (`Regalite.cs`). Random-loadout
/// Silent v0.111.0 seed 1025 (block cs=21 rs=17).
fn regalite_block_after_potion(
    version: GameVersion,
    character: CharacterId,
    potion: sts2_core::potions::PotionKind,
) -> (i32, u32) {
    let mut state = fresh(version, character);
    sts2_content::potions::dispatch::ensure_installed();
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::Regalite,
        counter: sts2_core::RelicKind::Regalite.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
    let _ = sts2_core::potion_effects::dispatch(&mut state, potion, None);
    (state.player.block, state.player.generated_cards_this_combat)
}

#[test]
fn potion_generated_cards_fire_player_generation_listeners() {
    use sts2_core::potions::PotionKind;
    let cunning =
        |v| regalite_block_after_potion(v, CharacterId::Silent, PotionKind::CunningPotion);
    assert_eq!(cunning(V107), (6, 3), "3 Shivs x Regalite 2");
    assert_eq!(cunning(V111), (4, 3), "Regalite 4, once per turn");
    let ghouls =
        |v| regalite_block_after_potion(v, CharacterId::Necrobinder, PotionKind::PotOfGhouls);
    assert_eq!(ghouls(V107), (4, 2), "2 Souls x Regalite 2");
    assert_eq!(ghouls(V111), (4, 2), "Regalite 4, once per turn");
}

// ── Unsettling Lamp: signed drains and foreign appliers ────────────────────

fn add_unsettling_lamp(state: &mut CombatState) {
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::UnsettlingLamp,
        counter: sts2_core::RelicKind::UnsettlingLamp.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
}

fn lamp_counter(state: &CombatState) -> i32 {
    state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::UnsettlingLamp)
        .expect("lamp")
        .counter
}

/// Loadout Necrobinder 1101 v0.111.0 act3: Shared Fate's enemy leg is
/// `PowerCmd.Apply<StrengthPower>(target, -2, owner, this)`; a negative
/// Strength reads as a Debuff (`GetTypeForAmount`, AllowNegative), so
/// Unsettling Lamp captures it and doubles it to -4 (Mecha Knight 10 -> 6).
/// The player's own -2 targets the owner's side and is left alone.
#[test]
fn shared_fate_enemy_drain_is_doubled_by_unsettling_lamp() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Necrobinder);
        add_unsettling_lamp(&mut state);
        let target = first_enemy(&state);
        add_enemy_power(&mut state, target, 800, PowerKind::Strength, 10);
        let i = push_hand(&mut state, CardId::SharedFate, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(enemy_power(&state, target, PowerKind::Strength), 6);
        let player_str: i32 = state
            .player_power_uids()
            .into_iter()
            .filter_map(|u| state.powers.get_by_id(u.0))
            .filter(|p| p.kind == PowerKind::Strength)
            .map(|p| p.amount)
            .sum();
        assert_eq!(player_str, -2);
        assert_ne!(
            lamp_counter(&state),
            sts2_core::RelicKind::UnsettlingLamp.initial_counter(),
            "the drain spent the Lamp's one trigger"
        );
    }
}

/// Loadout Necrobinder 1096 v0.107.1 act2: Misery copies each debuff with its
/// original `item.Applier`. BowlbugRock applies ImbalancedPower to itself
/// (`BowlbugRock.cs:66`), so the copy onto its sibling is not the Lamp
/// owner's apply: `UnsettlingLamp.BeforePowerAmountChanged` returns on
/// `applier != Owner.Creature`, nothing doubles, and the Lamp stays armed.
/// Rust treated the copy as player-applied: Imbalanced 2 and a spent Lamp.
#[test]
fn misery_copy_of_a_monster_own_debuff_is_not_doubled_by_unsettling_lamp() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Necrobinder);
        add_unsettling_lamp(&mut state);
        let rock = first_enemy(&state);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == rock.0) {
            c.species = Some(MonsterKind::BowlbugRock);
            c.monster_id = MonsterId("BowlbugRock");
        }
        add_enemy_power(&mut state, rock, 800, PowerKind::Imbalanced, 1);
        let other = add_enemy(&mut state, 50, 100);
        let i = push_hand(&mut state, CardId::Misery, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(enemy_power(&state, other, PowerKind::Imbalanced), 1);
        assert_eq!(
            lamp_counter(&state),
            sts2_core::RelicKind::UnsettlingLamp.initial_counter(),
            "a monster-applied copy never captures the Lamp"
        );
    }
}

/// The contrast: a player-applied debuff copied by Misery IS the Lamp owner's
/// apply, so it captures and doubles (Weak 2 -> 4 on the sibling).
#[test]
fn misery_copy_of_a_player_debuff_is_doubled_by_unsettling_lamp() {
    let mut state = fresh(V107, CharacterId::Necrobinder);
    add_unsettling_lamp(&mut state);
    let target = first_enemy(&state);
    add_enemy_power(&mut state, target, 800, PowerKind::Weak, 2);
    let other = add_enemy(&mut state, 50, 100);
    let i = push_hand(&mut state, CardId::Misery, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert_eq!(enemy_power(&state, other, PowerKind::Weak), 4);
}

// ── Temporary-Strength debuffs: the Strength leg runs before the power ─────

/// Loadout Necrobinder 1189 v0.111.0 act2 (step 565): Dark Shackles 9 on a
/// 5-HP Parafright with Strength 2, the player holding Sleight of Flesh 9.
/// `TemporaryStrengthPower.BeforeApplied` drains Strength 2 -> -7 BEFORE
/// DarkShacklesPower lands (PowerCmd.cs:132-135); Sleight of Flesh reacts to
/// that drain and kills the Parafright, whose reviving IllusionPower then
/// refuses powers (`ShouldAllowHitting`). C#: the corpse has Illusion,
/// Minion... and Strength -7, no DarkShackles. Rust landed DarkShackles 9
/// first.
#[test]
fn dark_shackles_target_killed_by_its_strength_drain_gets_no_power() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Necrobinder);
        let target = first_enemy(&state);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == target.0) {
            c.current_hp = 5;
        }
        add_enemy_power(&mut state, target, 800, PowerKind::Illusion, 1);
        add_enemy_power(&mut state, target, 801, PowerKind::Strength, 2);
        let _other = add_enemy(&mut state, 50, 100);
        add_player_power(&mut state, 900, PowerKind::SleightOfFlesh, 9);
        let i = push_hand(&mut state, CardId::DarkShackles, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(hp_of(&state, target), 0, "Sleight of Flesh kills it");
        assert_eq!(enemy_power(&state, target, PowerKind::Strength), -7);
        assert_eq!(
            enemy_power(&state, target, PowerKind::DarkShackles),
            0,
            "the drain killed it before DarkShackles could land"
        );
    }
}

/// The survivor case keeps the full package, and a SECOND Dark Shackles on
/// the same target stacks (`ModifyAmount`), its Strength leg following via
/// `AfterPowerAmountChanged`: DarkShackles 18, Strength -18.
#[test]
fn dark_shackles_stacks_and_drains_on_a_live_target() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    let target = first_enemy(&state);
    for _ in 0..2 {
        let i = push_hand(&mut state, CardId::DarkShackles, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    }
    assert_eq!(enemy_power(&state, target, PowerKind::DarkShackles), 18);
    assert_eq!(enemy_power(&state, target, PowerKind::Strength), -18);
}

/// Envenom's Poison is `PowerCmd.Apply<PoisonPower>(target, Amount, Owner,
/// null)` (EnvenomPower.cs:24): no card source, so Unsettling Lamp neither
/// captures nor doubles it, and the attacking card's own debuff is the one it
/// doubles. Suppress hits (1 Poison), then applies Weak 3, doubled to 6
/// (loadout Silent 1062 v0.111.0: Poison cs=1 rs=2).
#[test]
fn unsettling_lamp_ignores_envenom_poison() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        add_unsettling_lamp(&mut state);
        add_player_power(&mut state, 60, PowerKind::Envenom, 1);
        let enemy = CreatureUid(state.creatures[0].id);
        let i = push_hand(&mut state, CardId::Suppress, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(
            enemy_power(&state, enemy, PowerKind::Poison),
            1,
            "{version:?}"
        );
        assert_eq!(
            enemy_power(&state, enemy, PowerKind::Weak),
            6,
            "{version:?}"
        );
    }
}

/// `VoidFormPower.ShouldSkip` leaves a card outside the Hand and Play piles
/// at its cost (VoidFormPower.cs:84-100, both builds): Decisions, Decisions
/// re-plays an Equilibrium from the discard pile at 2 energy, so Intimidating
/// Helmet grants on those plays (loadout Regent 1002 v0.111.0: block cs=68
/// rs=60).
#[test]
fn void_form_frees_only_cards_in_hand_or_play() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        add_player_power(&mut state, 60, PowerKind::VoidForm, 2);
        let uid = new_card(&mut state, CardId::Equilibrium, false);
        state.player.discard.push(uid);
        let base = i32::from(state.cards[uid.0.get() as usize - 1].cost);
        assert!(base > 0, "{version:?}");
        assert_eq!(
            sts2_core::hooks::effective_card_cost(&state, uid),
            base,
            "{version:?}: in the discard pile Void Form leaves it alone"
        );
        state.player.discard.clear();
        state.player.hand.push(uid);
        assert_eq!(
            sts2_core::hooks::effective_card_cost(&state, uid),
            0,
            "{version:?}: in hand it is free"
        );
    }
}

/// Monarch's Gaze applies its StrengthDown with
/// `PowerCmd.Apply<MonarchsGazeStrengthDownPower>(target, Amount, Owner,
/// null)` (MonarchsGazePower.cs:17): no card source, so Unsettling Lamp
/// leaves it alone although the attacking card is resolving. One stack, one
/// Strength (loadout Regent 1062 v0.111.0: StrengthDown cs=1 rs=2).
#[test]
fn unsettling_lamp_ignores_monarchs_gaze() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        add_unsettling_lamp(&mut state);
        add_player_power(&mut state, 60, PowerKind::MonarchsGaze, 1);
        let enemy = CreatureUid(state.creatures[0].id);
        let i = push_hand(&mut state, CardId::StrikeRegent, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(
            enemy_power(&state, enemy, PowerKind::MonarchsGazeStrengthDown),
            1,
            "{version:?}"
        );
        assert_eq!(
            enemy_power(&state, enemy, PowerKind::Strength),
            -1,
            "{version:?}"
        );
    }
}

/// An `AnyEnemy` potion has nothing to aim at while the only enemy is a
/// corpse (a mid-revive Adaptable Test Subject): the game offers no target,
/// and C#'s `UsePotionAction` with a null target is cancelled. So it is not
/// a legal action; a self-targeted potion still is (loadout Regent 1078
/// v0.111.0: Rust drank Powdered Demise and Reptile Trinket fired).
#[test]
fn enemy_targeted_potion_needs_a_living_enemy() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        sts2_content::potions::dispatch::ensure_installed();
        state
            .player
            .potions
            .push(sts2_core::potions::PotionKind::FirePotion);
        state
            .player
            .potions
            .push(sts2_core::potions::PotionKind::BlockPotion);
        let drinkable = |s: &CombatState| -> Vec<u8> {
            sts2_core::legal_actions(s)
                .into_iter()
                .filter_map(|a| match a {
                    sts2_core::Action::DrinkPotion { potion_slot, .. } => Some(potion_slot),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(drinkable(&state), vec![0, 1], "{version:?}");
        state.creatures[0].current_hp = 0;
        assert_eq!(
            drinkable(&state),
            vec![1],
            "{version:?}: no target for Fire"
        );
    }
}

/// v0.111.0 Shining Strike returns to the draw top through
/// `GetResultLocationForCardPlay` (ShiningStrike.cs:26-35), i.e. at the
/// result-pile move after `AfterCardPlayed`, not inside `OnPlay` as in
/// v0.107.1. As Iron Club's 4th card its draw takes the card under it, and
/// the Strike lands on top afterwards (loadout Regent 1089 v0.111.0).
#[test]
fn shining_strike_reaches_the_draw_top_after_after_card_played_v0111() {
    let mut state = fresh(V111, CharacterId::Regent);
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::IronClub,
        counter: 3,
        is_wax: false,
        is_melted: false,
    });
    let below = new_card(&mut state, CardId::DefendRegent, false);
    state.player.draw.push(below);
    let i = push_hand(&mut state, CardId::ShiningStrike, false);
    let ss = state.player.hand[usize::from(i)];
    let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
    assert!(
        state.player.hand.contains(&below),
        "Iron Club drew the Defend"
    );
    assert_eq!(
        state.player.draw.first(),
        Some(&ss),
        "then the Strike went on top"
    );
}

/// C# `CardCmd.AutoPlay` takes `EnergyValue` while the card is still in its
/// pile, before `OnPlayWrapper` moves it to Play (CardCmd.cs:114-126). Out of
/// the discard pile Void Form leaves the cost alone, so an auto-played
/// 2-cost Equilibrium carries 2 and Intimidating Helmet grants its 4 block
/// (loadout Regent 1002 v0.111.0: Decisions, Decisions replays, cs=68
/// rs=60).
#[test]
fn void_form_does_not_zero_an_auto_play_from_the_discard_pile() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        add_player_power(&mut state, 60, PowerKind::VoidForm, 2);
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::IntimidatingHelmet,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let uid = new_card(&mut state, CardId::Equilibrium, false);
        state.player.discard.push(uid);
        let before = state.player.block;
        assert!(
            sts2_core::combat_loop::auto_play_specific_card_from_discard(&mut state, uid, false)
        );
        let gained = state.player.block - before;
        let eq_block =
            sts2_content::cards::v0111::card_pod_for(CardId::Equilibrium, false, version)
                .expect("spec")
                .base_block;
        assert_eq!(gained, eq_block + 4, "{version:?}: Helmet's 4 on top");
    }
}

/// The same out-of-hand `EnergyValue` for a card auto-played straight out of
/// the draw pile (Catastrophe, Uproar): Void Form leaves its cost alone there
/// (loadout Regent 1169 v0.107.1: a Catastrophe'd Rend, block cs=4 rs=0).
#[test]
fn void_form_does_not_zero_an_auto_play_from_the_draw_pile() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        add_player_power(&mut state, 60, PowerKind::VoidForm, 2);
        state.relics.push(sts2_core::Relic {
            id: NonZeroU32::new(1).expect("nz"),
            kind: sts2_core::RelicKind::IntimidatingHelmet,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let uid = new_card(&mut state, CardId::Equilibrium, false);
        state.player.draw.push(uid);
        let before = state.player.block;
        assert!(sts2_core::combat_loop::auto_play_specific_card(
            &mut state, uid, false
        ));
        let gained = state.player.block - before;
        let eq_block =
            sts2_content::cards::v0111::card_pod_for(CardId::Equilibrium, false, version)
                .expect("spec")
                .base_block;
        assert_eq!(gained, eq_block + 4, "{version:?}: Helmet's 4 on top");
    }
}

/// Decisions, Decisions with a lone Skill in hand (no prompt) still awaits
/// each repeat: Charge's draw-pile select prompts three times, and each pick
/// becomes two Minion Dive Bombs (DecisionsDecisions.cs, Charge.cs; loadout
/// Regent 1108 v0.111.0).
#[test]
fn decisions_decisions_repeats_a_lone_charge_through_its_selects() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Regent);
        state.player.energy = 9;
        let _ = push_hand(&mut state, CardId::Charge, false);
        let i = push_hand(&mut state, CardId::DecisionsDecisions, false);
        for _ in 0..9 {
            push_draw(&mut state, CardId::StrikeRegent);
        }
        let _ = sts2_core::apply(
            &mut state,
            sts2_core::Action::PlayCard {
                hand_index: i,
                target: None,
            },
        );
        // A two-card select takes one card per `ResolveChoice`.
        let mut picks = 0;
        while state.pending_choice.is_some() {
            picks += 1;
            assert!(picks <= 6, "{version:?}: three two-card selects");
            let pick = sts2_core::legal_actions(&state)
                .into_iter()
                .find_map(|a| match a {
                    sts2_core::Action::ResolveChoice { selection } => Some(selection),
                    _ => None,
                })
                .expect("a pick");
            let _ = sts2_core::apply(
                &mut state,
                sts2_core::Action::ResolveChoice { selection: pick },
            );
        }
        assert_eq!(picks, 6, "{version:?}: one two-card select per Charge play");
        let bombs = state
            .player
            .draw
            .iter()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::MinionDiveBomb)
            .count();
        assert_eq!(bombs, 6, "{version:?}: two transforms per play");
    }
}

/// Dodge and Roll passes `GainBlock`'s unfloored decimal to
/// `BlockNextTurnPower` (DodgeAndRoll.cs): at Dexterity -3 under Frail that is
/// (4 - 3) * 0.75 = 0.75, non-zero, so the power lands at `(int)0.75 == 0`
/// though no block was gained (loadout Silent 1291 v0.111.0).
#[test]
fn dodge_and_roll_leaves_a_zero_block_next_turn_from_a_fractional_block() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        add_player_power(&mut state, 60, PowerKind::Dexterity, -3);
        add_player_power(&mut state, 61, PowerKind::Frail, 2);
        let i = push_hand(&mut state, CardId::DodgeAndRoll, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(
            state.player.block, 0,
            "{version:?}: 0.75 floors to no block"
        );
        let next = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::BlockNextTurn)
            .map(|p| p.amount);
        assert_eq!(next, Some(0), "{version:?}: a 0-amount BlockNextTurn");
    }
}

/// Nightmare stores `card.CreateClone()` at select time
/// (NightmarePower.SetSelectedCard), which copies the picked card's
/// single-turn Retain. The stored clone is in no pile, so the flush's
/// `EndOfTurnCleanup` never clears it, and next turn's copies come in
/// retained (loadout Silent 1264 v0.111.0: an Expertise-drawn Untouchable).
#[test]
fn nightmare_copies_keep_the_picks_single_turn_retain() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        let pick = new_card(&mut state, CardId::Untouchable, false);
        state.player.hand.push(pick);
        state.player.retain_this_turn.push(pick);
        let i = push_hand(&mut state, CardId::Nightmare, false);
        let step = sts2_core::apply(
            &mut state,
            sts2_core::Action::PlayCard {
                hand_index: i,
                target: None,
            },
        );
        // A lone candidate is picked without a prompt.
        if step == sts2_core::StepResult::PendingChoice {
            let _ = sts2_core::apply(
                &mut state,
                sts2_core::Action::ResolveChoice {
                    selection: smallvec::smallvec![0],
                },
            );
        }
        assert!(
            state.powers.iter().any(|p| p.kind == PowerKind::Nightmare),
            "{version:?}: Nightmare applied"
        );
        // The flush clears this turn's flags; the snapshot keeps its own.
        state.player.retain_this_turn.clear();
        let hand_before = state.player.hand.len();
        let _ = sts2_core::hooks::dispatch_before_hand_draw_powers(&mut state);
        let copies: Vec<_> = state.player.hand[hand_before..].to_vec();
        assert_eq!(copies.len(), 3, "{version:?}: three copies");
        for c in copies {
            assert!(
                state.player.retain_this_turn.contains(&c),
                "{version:?}: each copy is retained this turn"
            );
        }
    }
}

/// Unsettling Lamp's capture needs the player as applier, but its doubling
/// (`ModifyPowerAmountGivenMultiplicative`, UnsettlingLamp.cs:105-122) only
/// compares the card: once Misery's copy of a player-applied Weak captured the
/// Lamp, its copy of the Rock's own Imbalanced (applier: the Rock) from the
/// same play is doubled too (#1150).
#[test]
fn misery_doubles_a_monster_debuff_copy_after_a_player_one_captured_the_lamp() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Necrobinder);
        add_unsettling_lamp(&mut state);
        let rock = first_enemy(&state);
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == rock.0) {
            c.species = Some(MonsterKind::BowlbugRock);
            c.monster_id = MonsterId("BowlbugRock");
        }
        add_enemy_power(&mut state, rock, 800, PowerKind::Weak, 2);
        add_enemy_power(&mut state, rock, 801, PowerKind::Imbalanced, 1);
        let other = add_enemy(&mut state, 50, 100);
        let i = push_hand(&mut state, CardId::Misery, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(
            enemy_power(&state, other, PowerKind::Weak),
            4,
            "{version:?}"
        );
        assert_eq!(
            enemy_power(&state, other, PowerKind::Imbalanced),
            2,
            "{version:?}: the Lamp is captured, the applier does not matter"
        );
    }
}

/// Danse Macabre reads `EnergyCost.GetResolved()` live on every play
/// iteration (DanseMacabrePower.cs:26-33). A Modded at cost 1 replayed once:
/// the first play reads 1 (no block), its `OnPlay` adds 1 (Modded.cs:28), and
/// the replay reads 2 and grants the block. Rust handed the replay the cost
/// from play start (#1150).
#[test]
fn danse_macabre_reads_a_replays_live_cost() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect);
        add_player_power(&mut state, 60, PowerKind::DanseMacabre, 4);
        let i = push_hand(&mut state, CardId::Modded, false);
        let uid = state.player.hand[usize::from(i)];
        let card = &mut state.cards[uid.0.get() as usize - 1];
        card.cost = 1;
        card.base_replay_count = 1;
        let before = state.player.block;
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(
            state.player.block - before,
            4,
            "{version:?}: the replay's block only"
        );
    }
}

/// Hand Trick's select skips Skills that are already Sly this turn
/// (`!card.IsSlyThisTurn`, HandTrick.cs), and a Skill Master Planner made Sly
/// (`CardCmd.ApplyKeyword(card, Sly)`) is one. With only one other Skill
/// left, `FromHand` picks it without a prompt (loadout Silent 1150 v0.111.0:
/// Rust prompted over the Master Planner Skills too).
#[test]
fn hand_trick_skips_a_skill_master_planner_made_sly() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent);
        let relax = new_card(&mut state, CardId::Relax, false);
        let haze = new_card(&mut state, CardId::Haze, false);
        state.player.hand.push(relax);
        state.player.hand.push(haze);
        state.player.sly_this_combat.push(haze);
        let i = push_hand(&mut state, CardId::HandTrick, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert!(
            state.pending_choice.is_none(),
            "{version:?}: one candidate, no prompt"
        );
        assert!(state.player.sly_this_turn.contains(&relax), "{version:?}");
    }
}

/// Darkness triggers each Dark orb's passive through `OrbCmd.Passive`, and
/// `DarkOrb.PassiveVal = ModifyOrbValue(6)` floors at 0, so Focus -12 adds
/// nothing and the evoke stays at its starting 6 (loadout Defect 1110
/// v0.107.1: Rust added a raw 6 - 12 twice and showed -6).
#[test]
fn darkness_under_negative_focus_adds_nothing() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect);
        state.player.orbs.ensure_defect_base();
        add_player_power(&mut state, 60, PowerKind::Focus, -12);
        let i = push_hand(&mut state, CardId::DarknessCard, true);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        let dark = state
            .player
            .orbs
            .slots
            .iter()
            .find(|o| matches!(o, sts2_core::orbs::Orb::Dark { .. }))
            .copied()
            .expect("a Dark orb");
        assert_eq!(
            dark,
            sts2_core::orbs::Orb::Dark { stored: 0 },
            "{version:?}"
        );
    }
}

/// Each drawn card's `AfterCardDrawn` walk finishes before the next card:
/// Automation's `GainEnergy` (a player power) runs before the drawn Void's
/// own `LoseEnergy(1)`. A Void that is the tenth draw for Automation drains
/// the energy Automation just granted (loadout Defect 1152 v0.111.0: energy
/// cs=1 rs=2; Rust granted Automation's energy after the whole draw).
#[test]
fn void_drains_the_energy_automation_grants_on_the_same_draw() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect);
        state.player.energy = 0;
        add_player_power(&mut state, 60, PowerKind::Automation, 1);
        // Nine draws already counted: the next card is the tenth.
        state
            .powers
            .get_mut_by_id(NonZeroU32::new(60).expect("nz"))
            .expect("p")
            .counter = 9;
        push_draw(&mut state, CardId::Void);
        let _ = sts2_core::cards::draw_via_state(&mut state, 1);
        assert_eq!(
            state.player.energy, 0,
            "{version:?}: +1 from Automation, -1 from Void"
        );
    }
}
