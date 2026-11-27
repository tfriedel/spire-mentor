//! v0.114.0 The Fool rework (GH #1924): the cards, powers and arrow /
//! melody commands that changed against v0.113.1, on the real play path.
//! Each test pins the v0.114.0 behaviour and, where the change is a branch,
//! that `V0_113_0` still plays the v0.113.1 decomp. Expected values are read
//! off the v0.114.0 decomp (`Models.Cards/<Name>.cs`,
//! `Models.Powers/<Name>Power.cs`, `Commands/{ArrowCmd,MelodyCmd}.cs`), not
//! off the engine.
#![allow(dead_code)]

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for as card_pod_from_id_for;
use sts2_core::cards::{CardId, CardTag};
use sts2_core::fool;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V113: GameVersion = GameVersion::V0_113_0;
const V114: GameVersion = GameVersion::V0_114_0;

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

/// One 100-HP Nibbit; a Fool with 10 energy in the play phase.
fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: version,
        ..CombatState::default()
    };
    let _ = add_enemy(&mut state, 1, 100);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 10;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Fool);
    state.round = 1;
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

fn play(state: &mut CombatState, id: CardId, upgraded: bool, target: Option<u8>) {
    let i = push_hand(state, id, upgraded);
    let _ = sts2_core::combat_loop::play_card(state, i, target);
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn player_power(state: &CombatState, kind: PowerKind) -> i32 {
    state.player_power_amount(kind)
}

fn ids(state: &CombatState, pile: &[sts2_core::CardUid]) -> Vec<CardId> {
    pile.iter()
        .map(|u| state.cards[u.get() as usize - 1].card_id)
        .collect()
}

fn hand_ids(state: &CombatState) -> Vec<CardId> {
    ids(state, &state.player.hand)
}

// ── card metadata ──────────────────────────────────────────────────────

/// Spec-level values of the reworked cards, per build (`<Name>.cs`
/// constructor `base(cost, ..)`, `DamageVar` / `BlockVar`, keywords,
/// `OnUpgrade`).
#[test]
fn reworked_card_pods_follow_each_build() {
    // The keyword registries back `card_exhausts_inst` for unchanged pods.
    content_dispatch::ensure_installed();
    // (card, upgraded, build, cost, damage, block)
    for (id, up, v, cost, dmg, block) in [
        (CardId::TheShot, false, V113, 1, 0, 0),
        (CardId::TheShot, false, V114, 0, 20, 0),
        (CardId::TheShot, true, V114, 0, 25, 0),
        (CardId::ArrowStab, false, V113, 1, 9, 0),
        (CardId::ArrowStab, false, V114, 2, 13, 0),
        (CardId::ArrowStab, true, V114, 2, 15, 0),
        (CardId::HeavyShot, true, V114, 2, 17, 0),
        (CardId::WahChoKi, false, V114, 2, 8, 8),
        (CardId::WahChoKi, true, V114, 2, 10, 10),
        (CardId::Growth, true, V114, 1, 0, 11),
        (CardId::LullabyCard, false, V113, 0, 0, 0),
        (CardId::LullabyCard, false, V114, 1, 0, 0),
        (CardId::MelodicShot, true, V113, 0, 13, 0),
        (CardId::MelodicShot, true, V114, 1, 14, 0),
        (CardId::TuneCard, true, V114, 1, 0, 0),
        (CardId::SimpletonCard, true, V114, 2, 0, 0),
        (CardId::Recall, false, V114, 0, 0, 0),
        (CardId::ChoFirm, true, V114, 1, 0, 7),
        (CardId::Temperance, false, V114, 0, 0, 20),
        (CardId::TwinShot, true, V114, 1, 10, 0),
        (CardId::Optimism, false, V114, 1, 0, 0),
        (CardId::MyShinies, true, V114, 1, 3, 0),
    ] {
        let c = card_pod_from_id_for(id, up, v).expect("spec");
        assert_eq!(
            (c.cost.base(), c.base_damage, c.base_block),
            (cost, dmg, block),
            "{id:?} up={up} {v:?}"
        );
    }
    let exhausts = |id, up, v| {
        sts2_core::card_effects::card_exhausts_inst(&card_pod_from_id_for(id, up, v).unwrap())
    };
    assert!(!exhausts(CardId::TheShot, false, V113));
    assert!(exhausts(CardId::TheShot, false, V114));
    assert!(exhausts(CardId::Florescence, false, V113));
    assert!(!exhausts(CardId::Florescence, false, V114));
    assert!(exhausts(CardId::Recall, false, V114));
    assert!(!exhausts(CardId::Recall, true, V114));
    assert!(exhausts(CardId::Optimism, false, V114));
    assert!(!exhausts(CardId::Optimism, true, V114));
    let ethereal = |up, v| {
        sts2_core::card_effects::card_ethereal_inst(
            &card_pod_from_id_for(CardId::MythicFormCard, up, v).unwrap(),
        )
    };
    assert!(!ethereal(false, V113));
    assert!(ethereal(false, V114));
    assert!(!ethereal(true, V114));
}

/// The v0.114.0 rarities that moved (`<Name>.cs` constructors).
#[test]
fn reworked_rarities_follow_each_build() {
    use sts2_content::cards::spec_index::rarity_for_card_id_in;
    use sts2_content::cards::types::CardRarity as R;
    for (id, r113, r114) in [
        (CardId::ArcheryCard, R::Uncommon, R::Rare),
        (CardId::ExplosiveArrowsCard, R::Rare, R::Uncommon),
        (CardId::HeavyShot, R::Common, R::Uncommon),
        (CardId::MotifCard, R::Uncommon, R::Rare),
        (CardId::Quiver, R::Rare, R::Uncommon),
    ] {
        assert_eq!(rarity_for_card_id_in(id, V113), Some(r113), "{id:?}");
        assert_eq!(rarity_for_card_id_in(id, V114), Some(r114), "{id:?}");
    }
}

// ── pool ───────────────────────────────────────────────────────────────

#[test]
fn fool_pool_gains_optimism_and_twin_shot_loses_rapid_shot() {
    use sts2_content::cards::pool_order::{generate_all_cards, Pool};
    let p113 = generate_all_cards(Pool::Character(CharacterId::Fool), V113);
    let p114 = generate_all_cards(Pool::Character(CharacterId::Fool), V114);
    assert_eq!((p113.len(), p114.len()), (90, 91));
    assert!(p113.contains(&CardId::RapidShot) && !p114.contains(&CardId::RapidShot));
    let at = |id| p114.iter().position(|&c| c == id).unwrap();
    assert_eq!(at(CardId::Optimism), at(CardId::Nock) + 1);
    let rapid = p113.iter().position(|&c| c == CardId::RapidShot).unwrap();
    assert_eq!(
        at(CardId::TwinShot),
        rapid + 1,
        "RapidShot's slot, after Optimism"
    );
}

/// `MythicForm.cs` drops `CanBeGeneratedInCombat => false` in v0.114.0.
#[test]
fn mythic_form_is_generatable_in_combat_from_v0114() {
    use sts2_content::cards::pool_order::{combat_pool, Pool};
    let has = |v| {
        combat_pool(Pool::Character(CharacterId::Fool), v, false)
            .iter()
            .any(|s| s.id == CardId::MythicFormCard)
    };
    assert!(!has(V113));
    assert!(has(V114));
    let p114 = combat_pool(Pool::Character(CharacterId::Fool), V114, false);
    assert!(p114.iter().any(|s| s.id == CardId::TwinShot));
    assert!(!p114.iter().any(|s| s.id == CardId::RapidShot));
}

// ── new cards ──────────────────────────────────────────────────────────

/// `TwinShot.cs`: one arrow, 2 hits of 7 (`FromArrow`), impales.
#[test]
fn twin_shot_shoots_one_arrow_for_two_hits() {
    let mut s = fresh(V114);
    play(&mut s, CardId::TwinShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 14);
    assert!(fool::creature_is_impaled(&s, s.enemy_uids[0]));
    assert_eq!(fool::available_arrow_count(&s), 0);
    let mut s = fresh(V114);
    play(&mut s, CardId::TwinShot, true, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 20);
}

/// `Optimism.cs`: exhaust every Sadness in hand, draw that many; nothing
/// else is exhausted, and no Sadness means no draw.
#[test]
fn optimism_exhausts_sadness_and_draws_that_many() {
    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::Sadness, false);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    let _ = push_hand(&mut s, CardId::Sadness, false);
    for _ in 0..3 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::Optimism, false, None);
    let exhausted = ids(&s, &s.player.exhaust);
    assert_eq!(
        exhausted.iter().filter(|&&c| c == CardId::Sadness).count(),
        2
    );
    assert!(exhausted.contains(&CardId::Optimism), "Exhaust keyword");
    let hand = hand_ids(&s);
    assert_eq!(hand.len(), 3, "Strike + 2 drawn: {hand:?}");
    assert_eq!(s.player.draw.len(), 1);

    let mut s = fresh(V114);
    push_draw(&mut s, CardId::DefendFool);
    play(&mut s, CardId::Optimism, true, None);
    assert!(s.player.hand.is_empty() && s.player.draw.len() == 1);
    assert_eq!(ids(&s, &s.player.discard), vec![CardId::Optimism]);
}

// ── reworked cards ─────────────────────────────────────────────────────

/// v0.114.0 `TrickShot.cs`: a shot that happened draws 2 (+1); the card
/// goes to the discard pile. v0.113.0 sent it back to hand instead.
#[test]
fn trick_shot_draws_instead_of_returning() {
    let mut s = fresh(V114);
    for _ in 0..3 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::TrickShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 12);
    assert_eq!(hand_ids(&s), vec![CardId::DefendFool; 2]);
    assert_eq!(ids(&s, &s.player.discard), vec![CardId::TrickShot]);

    let mut s = fresh(V113);
    for _ in 0..3 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::TrickShot, false, Some(0));
    assert_eq!(hand_ids(&s), vec![CardId::TrickShot]);
}

/// v0.114.0 `TrickShot.cs` lost its `GetResultLocationForCardPlay` override,
/// so `ReboundPower.ModifyCardPlayResultLocation` claims it like any card: to
/// the top of the draw pile, one stack spent. On v0.113.0 a ready arrow sends
/// it back to hand, which Rebound leaves alone.
#[test]
fn rebound_claims_trick_shot_on_v0114() {
    let mut s = fresh(V114);
    for _ in 0..3 {
        push_draw(&mut s, CardId::DefendFool);
    }
    add_player_power(&mut s, 9301, PowerKind::Rebound, 1);
    assert!(fool::player_has_any_arrows(&s));
    play(&mut s, CardId::TrickShot, false, Some(0));
    assert_eq!(hand_ids(&s), vec![CardId::DefendFool; 2]);
    assert_eq!(ids(&s, &s.player.draw[..1]), vec![CardId::TrickShot]);
    assert!(s.player.discard.is_empty());
    assert_eq!(player_power(&s, PowerKind::Rebound), 0);

    let mut s = fresh(V113);
    for _ in 0..3 {
        push_draw(&mut s, CardId::DefendFool);
    }
    add_player_power(&mut s, 9301, PowerKind::Rebound, 1);
    play(&mut s, CardId::TrickShot, false, Some(0));
    assert_eq!(hand_ids(&s), vec![CardId::TrickShot]);
    assert_eq!(player_power(&s, PowerKind::Rebound), 1);
}

/// v0.114.0 `CreatureCmd.Damage` works the hit out (`GetDamageResults`,
/// CreatureCmd.cs:312) before `BeforeDamageReceived` (:334). The player has
/// Thorns 5 and Twine; a 3-HP impaled enemy hits for 10. The reflect kills
/// it, the arrow comes loose and Twine grants 4 block, but the hit's block
/// was fixed beforehand (0), so all 10 land and the block stays. v0.113.1
/// fired Thorns first, and the fresh 4 block absorbed part of the hit.
#[test]
fn thorns_reflect_block_does_not_absorb_the_hit_on_v0114() {
    use sts2_core::hooks::{CreatureRef, ValueProps};
    for (v, hp_lost, block_left) in [(V114, 10, 4), (V113, 6, 0)] {
        let mut s = fresh(v);
        play(&mut s, CardId::FoolishShot, false, Some(0));
        let enemy = s.enemy_uids[0];
        assert_eq!(fool::arrows_in(&s, enemy), 1, "{v:?}: impaled");
        s.creatures[0].current_hp = 3;
        let id = NonZeroU32::new(u32::try_from(s.relics.len() + 1).expect("u32")).expect("nz");
        s.relics.push(sts2_core::Relic {
            id,
            kind: sts2_core::RelicKind::Twine,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        add_player_power(&mut s, 9302, PowerKind::Thorns, 5);
        s.player.block = 0;
        let hp = s.player.current_hp;
        let _ = sts2_core::combat_loop::apply_damage_to_player(
            &mut s,
            CreatureRef::Enemy(enemy),
            10,
            ValueProps::MOVE,
            None,
        );
        assert_eq!(s.creatures[0].current_hp, 0, "{v:?}: the reflect kills");
        assert_eq!(hp - s.player.current_hp, hp_lost, "{v:?}");
        assert_eq!(s.player.block, block_left, "{v:?}");
    }
}

/// v0.114.0 `ArrowStab.cs`: attack 13, then Aim 6 on the player.
#[test]
fn arrow_stab_attacks_then_gains_aim() {
    let mut s = fresh(V114);
    play(&mut s, CardId::ArrowStab, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 13);
    assert_eq!(player_power(&s, PowerKind::Aim), 6);
    let mut s = fresh(V114);
    play(&mut s, CardId::ArrowStab, true, Some(0));
    assert_eq!(player_power(&s, PowerKind::Aim), 8);
}

/// v0.114.0 `TheShot.cs`: a flat 20 (+5) arrow shot, not x Florescence.
#[test]
fn the_shot_is_a_flat_shot() {
    let mut s = fresh(V114);
    play(&mut s, CardId::TheShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 80);
    assert_eq!(ids(&s, &s.player.exhaust), vec![CardId::TheShot]);
    let mut s = fresh(V113);
    play(&mut s, CardId::TheShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100, "0 x Florescence count");
}

/// v0.114.0 `Florescence.cs`: a lone hand card is transformed into a
/// TheShot (upgraded with Florescence) without a prompt; more open the
/// single-pick transform, whose resolver mints TheShot.
#[test]
fn florescence_transforms_a_hand_card_into_the_shot() {
    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    play(&mut s, CardId::Florescence, true, None);
    assert_eq!(hand_ids(&s), vec![CardId::TheShot]);
    let shot = &s.cards[s.player.hand[0].get() as usize - 1];
    assert!(shot.upgraded);
    assert_eq!((shot.cost.base(), shot.base_damage), (0, 25));
    assert_eq!(ids(&s, &s.player.discard), vec![CardId::Florescence]);

    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    let _ = push_hand(&mut s, CardId::DefendFool, false);
    play(&mut s, CardId::Florescence, false, None);
    let pc = s.pending_choice.clone().expect("a pick");
    assert_eq!(
        pc.kind,
        sts2_core::ChoiceKind::TransformInHandToMinionStrike
    );
    assert_eq!(pc.options.len(), 2);
    let _ = sts2_core::resolve_choice_atomic_apply(&mut s, &[1]);
    assert_eq!(hand_ids(&s), vec![CardId::StrikeFool, CardId::TheShot]);
}

/// v0.114.0 `Recall.cs`: dislodge every arrow, no discard-to-draw pick.
#[test]
fn recall_only_dislodges() {
    let mut s = fresh(V114);
    play(&mut s, CardId::FoolishShot, false, Some(0));
    assert!(fool::creature_is_impaled(&s, s.enemy_uids[0]));
    play(&mut s, CardId::Recall, false, Some(0));
    assert!(!fool::creature_is_impaled(&s, s.enemy_uids[0]));
    assert_eq!(fool::available_arrow_count(&s), 1);
    assert!(s.pending_choice.is_none());
    assert_eq!(s.player.draw.len(), 0, "FoolishShot stays in the discard");
}

/// v0.114.0 `WahKi.cs`: 2 hits of 3, then the one discard card to the top
/// of the draw pile; no EnergyNextTurn / DrawCardsNextTurn.
#[test]
fn wah_ki_puts_a_discard_card_on_top() {
    let mut s = fresh(V114);
    let strike = new_card(&mut s, CardId::StrikeFool, false);
    s.player.discard.push(strike);
    play(&mut s, CardId::WahKi, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 6);
    assert_eq!(s.player.draw.first(), Some(&strike));
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 0);
    let mut s = fresh(V113);
    play(&mut s, CardId::WahKi, false, Some(0));
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 1);
}

/// v0.114.0 `Growth.cs`: block 8, then a Sadness in hand becomes a clone
/// of the Growth.
#[test]
fn growth_blocks_and_turns_sadness_into_growth() {
    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::Sadness, false);
    play(&mut s, CardId::Growth, true, None);
    assert_eq!(s.player.block, 11);
    assert_eq!(hand_ids(&s), vec![CardId::Growth]);
    let clone = &s.cards[s.player.hand[0].get() as usize - 1];
    assert!(clone.upgraded && clone.is_clone());
    assert_eq!(player_power(&s, PowerKind::Stretch), 0);
}

/// v0.114.0 `PunchOfJoy.cs`: 2 (upgraded) Sadness become Joy.
#[test]
fn punch_of_joy_transforms_cards_var_sadness() {
    let mut s = fresh(V114);
    for _ in 0..3 {
        let _ = push_hand(&mut s, CardId::Sadness, false);
    }
    play(&mut s, CardId::PunchOfJoy, true, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 16);
    let hand = hand_ids(&s);
    assert_eq!(hand.iter().filter(|&&c| c == CardId::Joy).count(), 2);
    assert_eq!(hand.iter().filter(|&&c| c == CardId::Sadness).count(), 1);
}

/// v0.114.0 `TheSun.cs`: 2 Soot shuffled into the draw pile, none in the
/// discard pile.
#[test]
fn the_sun_shuffles_two_soot_into_the_draw_pile() {
    let mut s = fresh(V114);
    for _ in 0..4 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::TheSun, false, None);
    let draw = ids(&s, &s.player.draw);
    assert_eq!(draw.iter().filter(|&&c| c == CardId::Soot).count(), 2);
    assert!(!ids(&s, &s.player.discard).contains(&CardId::Soot));
    let mut s = fresh(V113);
    for _ in 0..4 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::TheSun, false, None);
    assert!(ids(&s, &s.player.discard).contains(&CardId::Soot));
}

/// v0.114.0 `HarmonicShield.cs`: `4 + 4 x MelodiesThisTurn`, so 4 with no
/// melody played (v0.113.0: 0).
#[test]
fn harmonic_shield_has_a_base() {
    let mut s = fresh(V114);
    play(&mut s, CardId::HarmonicShield, false, None);
    assert_eq!(s.player.block, 4);
    let mut s = fresh(V113);
    play(&mut s, CardId::HarmonicShield, false, None);
    assert_eq!(s.player.block, 0);
}

/// v0.114.0 `HeavyShot.ModifyAimAmount`: Aim x 3 (+1).
#[test]
fn heavy_shot_triples_aim() {
    for (v, up, want) in [
        (V113, false, 20 + 2 * 2),
        (V114, false, 16 + 3 * 2),
        (V114, true, 17 + 4 * 2),
    ] {
        let mut s = fresh(v);
        add_player_power(&mut s, 90, PowerKind::Aim, 2);
        play(&mut s, CardId::HeavyShot, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - want, "{v:?} up={up}");
    }
}

/// v0.114.0 power-card amounts (`PowerVar` and its upgrade).
#[test]
fn power_card_amounts_follow_v0114() {
    for (id, kind, base, up) in [
        (CardId::AllIsWellCard, PowerKind::AllIsWell, 4, 6),
        (CardId::ArcheryCard, PowerKind::Archery, 6, 8),
        (
            CardId::ExplosiveArrowsCard,
            PowerKind::ExplosiveArrows,
            8,
            11,
        ),
        (CardId::HoningCard, PowerKind::Honing, 100, 100),
        (CardId::LullabyCard, PowerKind::Lullaby, 12, 16),
        (
            CardId::SerratedArrowsCard,
            PowerKind::SerratedArrows,
            50,
            75,
        ),
        (CardId::SimpletonCard, PowerKind::Simpleton, 1, 2),
        (CardId::SongPathCard, PowerKind::SongPath, 4, 6),
        (CardId::TuneCard, PowerKind::Tune, 1, 2),
        (CardId::ViolentChordCard, PowerKind::ViolentChord, 7, 9),
        (CardId::ChoBarbsCard, PowerKind::ChoBarbs, 2, 3),
        (CardId::ChoReady, PowerKind::Aim, 3, 4),
    ] {
        for (up_flag, n) in [(false, base), (true, up)] {
            let mut s = fresh(V114);
            play(&mut s, id, up_flag, None);
            assert_eq!(player_power(&s, kind), n, "{id:?} up={up_flag}");
        }
    }
}

/// v0.114.0 `MythicForm`: no upgraded-tarot counter on the power.
#[test]
fn upgraded_mythic_form_has_no_upgraded_counter() {
    for (v, want) in [(V113, 1), (V114, 0)] {
        let mut s = fresh(v);
        play(&mut s, CardId::MythicFormCard, true, None);
        let counter = s
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::MythicForm)
            .map(|p| p.counter);
        assert_eq!(counter, Some(want), "{v:?}");
    }
}

// ── powers and commands ────────────────────────────────────────────────

/// v0.114.0 `MelodyCmd.PlayedThisTurn` reads `CardPlaysStarted`: a melody
/// card counts as played once it STARTS.
#[test]
fn played_this_turn_reads_started_plays() {
    for (v, want) in [(V113, false), (V114, true)] {
        let mut s = fresh(v);
        let wah = new_card(&mut s, CardId::WahBlast, false);
        s.player.hand.push(wah);
        fool::note_play_started(&mut s, wah);
        assert_eq!(fool::played_this_turn(&s, CardTag::Wah), want, "{v:?}");
        assert_eq!(fool::melodies_this_turn(&s), i32::from(want), "{v:?}");
    }
}

/// v0.114.0 `KiVersePower`: only Wah or Cho cards play again.
#[test]
fn ki_verse_skips_ki_only_cards() {
    // KiHop is Ki-only.
    for (v, left) in [(V113, 0), (V114, 1)] {
        let mut s = fresh(v);
        add_player_power(&mut s, 90, PowerKind::KiVerse, 1);
        play(&mut s, CardId::KiHop, false, Some(0));
        assert_eq!(player_power(&s, PowerKind::KiVerse), left, "{v:?}");
    }
    let mut s = fresh(V114);
    add_player_power(&mut s, 90, PowerKind::KiVerse, 1);
    play(&mut s, CardId::WahTouch, false, Some(0));
    // 6, then 6 x 1.5 through the Vulnerable the first play applied.
    assert_eq!(enemy_hp(&s), 100 - 6 - 9, "Wah card played twice");
    assert_eq!(player_power(&s, PowerKind::KiVerse), 0);
}

/// v0.114.0 `TunePower.AfterCardPlayed`: the turn's first Ki play upgrades
/// `Amount` random upgradable hand cards; a second Ki play does not, and the
/// turn end grants no FreeCard.
#[test]
fn tune_upgrades_hand_cards_on_the_first_ki_play() {
    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    let _ = push_hand(&mut s, CardId::DefendFool, false);
    add_player_power(&mut s, 90, PowerKind::Tune, 1);
    play(&mut s, CardId::KiHop, false, Some(0));
    let upgraded = |s: &CombatState| {
        s.player
            .hand
            .iter()
            .filter(|u| s.cards[u.get() as usize - 1].upgraded)
            .count()
    };
    assert_eq!(upgraded(&s), 1);
    play(&mut s, CardId::KiHop, false, Some(0));
    assert_eq!(upgraded(&s), 1, "not the first Ki play");
    let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut s);
    assert_eq!(player_power(&s, PowerKind::FreeCard), 0);

    // v0.113.0: no upgrade on play.
    let mut s = fresh(V113);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    add_player_power(&mut s, 90, PowerKind::Tune, 1);
    play(&mut s, CardId::KiHop, false, Some(0));
    assert_eq!(upgraded(&s), 0);
}

/// Lost Lute applied before Tune auto-plays a card on the turn's first Ki
/// play; Tune still sees the Ki play as the `cardPlay` (MelodyCmd.cs:49-57
/// compares the play itself), so it still upgrades.
#[test]
fn tune_fires_after_lost_lute_auto_plays() {
    let mut s = fresh(V114);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    push_draw(&mut s, CardId::DefendFool);
    add_player_power(&mut s, 89, PowerKind::LostLute, 1);
    add_player_power(&mut s, 90, PowerKind::Tune, 1);
    play(&mut s, CardId::KiHop, false, Some(0));
    assert!(s.player.draw.is_empty(), "Lost Lute auto-played the Defend");
    let strike = s.player.hand[0];
    assert!(s.cards[strike.get() as usize - 1].upgraded);
}

/// v0.114.0 `QuietusPower` no longer destroys the arrow after impaling.
#[test]
fn quietus_keeps_the_arrow() {
    let mut s = fresh(V114);
    add_player_power(&mut s, 90, PowerKind::Quietus, 2);
    play(&mut s, CardId::FoolishShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 15 * 2);
    assert_eq!(fool::arrows_in(&s, s.enemy_uids[0]), 1);
    assert_ne!(
        fool::fool(&s).unwrap().arrows[0].status,
        fool::ArrowStatus::Destroyed
    );
}

/// v0.114.0 `StretchPower.ModifyBlockAdditive` returns the amount latched
/// when the card started, not the live amount.
#[test]
fn stretch_bonus_is_the_start_amount() {
    use sts2_core::hooks::ValueProps;
    for (v, want) in [(V113, 9), (V114, 6)] {
        let mut s = fresh(v);
        let card = new_card(&mut s, CardId::DefendFool, false);
        s.player.hand.push(card);
        add_player_power(&mut s, 90, PowerKind::Stretch, 6);
        fool::latch_stretch_at_play_start(&mut s, card);
        assert_eq!(
            fool::stretch_block_bonus(&s, 9, ValueProps::MOVE, Some(card)),
            want,
            "{v:?}"
        );
    }
}

/// v0.114.0 `WahSymphony.cs` counts the player's Cho or Ki cards, not Wah.
#[test]
fn wah_symphony_counts_cho_and_ki_cards() {
    for (v, want) in [(V113, 6 + 4 * 3), (V114, 6 + 4 * 2)] {
        let mut s = fresh(v);
        push_draw(&mut s, CardId::WahTouch);
        push_draw(&mut s, CardId::KiHop);
        push_draw(&mut s, CardId::ChoLimbs);
        play(&mut s, CardId::WahSymphony, false, None);
        // The played WahSymphony is a Wah card on the Play pile too.
        let wah_self = i32::from(v == V113) * 4;
        assert_eq!(enemy_hp(&s), 100 - want - wah_self, "{v:?}");
    }
}

/// v0.114.0 `MyShinies.cs` (colorless): `1 + Gold / 25` hits of 2.
#[test]
fn my_shinies_hits_follow_v0114() {
    let mut s = fresh(V114);
    s.gold = 60;
    play(&mut s, CardId::MyShinies, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 3 * 2);
}

/// v0.114.0 `ImpaledPower.AfterDamageReceived` needs a card source: a
/// powered player hit with no card behind it no longer ticks `HitCounter`.
#[test]
fn impaled_ticks_only_on_card_hits_from_v0114() {
    for (v, want) in [(V113, 2), (V114, 3)] {
        let mut s = fresh(v);
        play(&mut s, CardId::FoolishShot, false, Some(0));
        let enemy = s.enemy_uids[0];
        assert_eq!(fool::impaled_hit_counter(&s, enemy), Some(3), "{v:?}");
        sts2_core::damage::apply_damage(
            &mut s,
            sts2_core::hooks::CreatureRef::Player,
            enemy,
            1,
            sts2_core::hooks::ValueProps::MOVE,
            None,
        );
        assert_eq!(fool::impaled_hit_counter(&s, enemy), Some(want), "{v:?}");
        play(&mut s, CardId::StrikeFool, false, Some(0));
        assert_eq!(
            fool::impaled_hit_counter(&s, enemy),
            Some(want - 1),
            "{v:?} card hit"
        );
    }
}

/// v0.114.0 `Timber.cs` runs both damage calls in one `AttackContext`, so
/// Vigor boosts the kill sweep too and is spent once; through v0.113.1 the
/// first `DamageCmd.Attack` spends it before the sweep.
#[test]
fn timber_sweep_shares_the_attack_command_from_v0114() {
    for (v, survivor_hp) in [(V113, 200 - 30), (V114, 200 - 30 - 5)] {
        let mut s = fresh(v);
        s.creatures[0].current_hp = 10;
        let survivor = add_enemy(&mut s, 50, 200);
        add_player_power(&mut s, 90, PowerKind::Vigor, 5);
        play(&mut s, CardId::Timber, false, Some(0));
        assert_eq!(enemy_hp(&s), 0, "{v:?}");
        let hp = s
            .creatures
            .iter()
            .find(|c| c.id == survivor.0)
            .map_or(0, |c| c.current_hp);
        assert_eq!(hp, survivor_hp, "{v:?}");
        assert_eq!(player_power(&s, PowerKind::Vigor), 0, "{v:?}");
    }
}

/// v0.114.0 `ImpaledPower` no longer asks for `dealer == Applier`: the
/// player's Osty hitting with the player's card (Poke) ticks `HitCounter`.
#[test]
fn impaled_ticks_on_osty_card_hits_from_v0114() {
    for (v, want) in [(V113, 3), (V114, 2)] {
        let mut s = fresh(v);
        let _ = s.summon_osty(10);
        play(&mut s, CardId::FoolishShot, false, Some(0));
        let enemy = s.enemy_uids[0];
        assert_eq!(fool::impaled_hit_counter(&s, enemy), Some(3), "{v:?}");
        let hp = enemy_hp(&s);
        play(&mut s, CardId::Poke, false, Some(0));
        assert!(enemy_hp(&s) < hp, "{v:?}: Osty hit");
        assert_eq!(fool::impaled_hit_counter(&s, enemy), Some(want), "{v:?}");
    }
}
