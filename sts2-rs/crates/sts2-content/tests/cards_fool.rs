//! The Fool's cards (v0.113.0, lane FOOL CARDS): per-card tests on the real
//! play path, values from the v0.113.0 decomp `Models.Cards/<Name>.cs`.
//! Same harness as `cards_v0113.rs`. The Fool character is the core lane's,
//! so these play the cards as Ironclad: none of them reads the character.
#![allow(dead_code)]

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for as card_pod_from_id_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V111: GameVersion = GameVersion::V0_111_0;
const V113: GameVersion = GameVersion::V0_113_0;

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

fn play(state: &mut CombatState, id: CardId, upgraded: bool, target: Option<u8>) {
    let i = push_hand(state, id, upgraded);
    let _ = sts2_core::combat_loop::play_card(state, i, target);
}

fn player_power(state: &CombatState, kind: PowerKind) -> i32 {
    state.player_power_amount(kind)
}

fn hand_ids(state: &CombatState) -> Vec<CardId> {
    state
        .player
        .hand
        .iter()
        .map(|u| state.cards[u.get() as usize - 1].card_id)
        .collect()
}

#[test]
fn strike_fool_and_justice() {
    for (id, up, dmg) in [
        (CardId::StrikeFool, false, 6),
        (CardId::StrikeFool, true, 9),
        (CardId::Justice, false, 30),
        (CardId::Justice, true, 35),
    ] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, id, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg, "{id:?} up={up}");
    }
}

#[test]
fn defend_fool_and_temperance() {
    for (id, up, blk) in [
        (CardId::DefendFool, false, 5),
        (CardId::DefendFool, true, 8),
        (CardId::Temperance, false, 30),
        (CardId::Temperance, true, 35),
    ] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, id, up, None);
        assert_eq!(s.player.block, blk, "{id:?} up={up}");
    }
}

#[test]
fn the_chariot_and_wah_sweep_hit_all() {
    for (id, up, dmg) in [
        (CardId::TheChariot, false, 25),
        (CardId::TheChariot, true, 30),
        (CardId::WahSweep, false, 8),
        (CardId::WahSweep, true, 11),
    ] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        let b = add_enemy(&mut s, 50, 100);
        play(&mut s, id, up, None);
        assert_eq!(enemy_hp(&s), 100 - dmg, "{id:?}");
        assert_eq!(hp_of(&s, b), 100 - dmg, "{id:?}");
    }
}

#[test]
fn the_emperor_energy() {
    for (up, e) in [(false, 7), (true, 9)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::TheEmperor, up, None);
        assert_eq!(s.player.energy, 5 + e);
    }
}

#[test]
fn the_empress_weak_and_vulnerable() {
    for (up, n) in [(false, 3), (true, 4)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::TheEmpress, up, Some(0));
        let e = first_enemy(&s);
        assert_eq!(enemy_power(&s, e, PowerKind::Weak), n);
        assert_eq!(enemy_power(&s, e, PowerKind::Vulnerable), n);
    }
}

#[test]
fn strength_tarot_and_death() {
    for (up, n) in [(false, 4), (true, 5)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::StrengthCard, up, None);
        assert_eq!(player_power(&s, PowerKind::Strength), n);
    }
    for (up, n) in [(false, 75), (true, 99)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::Death, up, Some(0));
        assert_eq!(enemy_power(&s, first_enemy(&s), PowerKind::Doom), n);
    }
}

#[test]
fn joy_energy_then_draw() {
    for (up, e) in [(false, 1), (true, 2)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        push_draw(&mut s, CardId::StrikeFool);
        push_draw(&mut s, CardId::StrikeFool);
        play(&mut s, CardId::Joy, up, None);
        assert_eq!(s.player.energy, 5 + e);
        assert_eq!(s.player.hand.len(), 1);
        assert_eq!(s.player.exhaust.len(), 1, "Joy exhausts");
    }
}

#[test]
fn the_high_priestess_draws() {
    for (up, n) in [(false, 4), (true, 5)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        for _ in 0..8 {
            push_draw(&mut s, CardId::StrikeFool);
        }
        play(&mut s, CardId::TheHighPriestess, up, None);
        assert_eq!(s.player.hand.len(), n);
    }
}

#[test]
fn the_star_next_turn_powers() {
    for (up, n) in [(false, 3), (true, 4)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::TheStar, up, None);
        assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), n);
        assert_eq!(player_power(&s, PowerKind::DrawCardsNextTurn), n);
    }
}

#[test]
fn wah_touch_cho_cho_ki() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::WahTouch, true, Some(0));
    assert_eq!(enemy_hp(&s), 92);
    assert_eq!(enemy_power(&s, first_enemy(&s), PowerKind::Vulnerable), 2);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::WahCho, false, Some(0));
    assert_eq!((s.player.block, enemy_hp(&s)), (3, 97));

    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..3 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::WahChoKi, true, Some(0));
    assert_eq!((s.player.block, enemy_hp(&s)), (12, 88));
    assert_eq!(s.player.hand.len(), 2);
}

#[test]
fn ki_whip_and_wah_ki() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::KiWhip, false, Some(0));
    assert_eq!(enemy_hp(&s), 95);
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 2);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::WahKi, true, Some(0));
    assert_eq!(enemy_hp(&s), 92, "2 hits of 4");
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 1);
    assert_eq!(player_power(&s, PowerKind::DrawCardsNextTurn), 1);
}

#[test]
fn timber_kill_splashes_all() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    s.creatures[0].current_hp = 15;
    let b = add_enemy(&mut s, 50, 100);
    play(&mut s, CardId::Timber, false, Some(0));
    assert_eq!(hp_of(&s, b), 70, "30 on-kill damage to the rest");

    let mut s = fresh(V113, CharacterId::Ironclad);
    let b = add_enemy(&mut s, 50, 100);
    play(&mut s, CardId::Timber, false, Some(0));
    assert_eq!(enemy_hp(&s), 80);
    assert_eq!(hp_of(&s, b), 100, "no kill, no splash");
}

#[test]
fn sadness_generators() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::SorrowfulStrike, false, Some(0));
    assert_eq!(enemy_hp(&s), 87);
    assert_eq!(hand_ids(&s), vec![CardId::Sadness; 2]);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::Jest, true, None);
    assert_eq!(s.player.block, 18);
    assert_eq!(hand_ids(&s), vec![CardId::Sadness]);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::Vitriol, false, Some(0));
    assert_eq!(enemy_power(&s, first_enemy(&s), PowerKind::Weak), 2);
    assert_eq!(hand_ids(&s), vec![CardId::Sadness]);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::BrightenUp, true, None);
    assert_eq!(
        hand_ids(&s),
        vec![
            CardId::Sadness,
            CardId::Sadness,
            CardId::Joy,
            CardId::Joy,
            CardId::Joy
        ]
    );

    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..5 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::Rummage, false, None);
    let h = hand_ids(&s);
    assert_eq!(h.len(), 4);
    assert_eq!(h[3], CardId::Sadness, "Sadness after the draw");
}

#[test]
fn sadness_is_unplayable_and_retained() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    let _ = push_hand(&mut s, CardId::Sadness, false);
    let legal = sts2_core::legal_actions(&s);
    assert!(
        !legal
            .iter()
            .any(|a| matches!(a, sts2_core::Action::PlayCard { hand_index: 0, .. })),
        "Sadness is Unplayable"
    );
    assert!(sts2_core::card_effects::card_retain(CardId::Sadness));
}

#[test]
fn self_power_cards_apply_their_power() {
    for (id, kind, base, up) in [
        (CardId::AllIsWellCard, PowerKind::AllIsWell, 4, 5),
        (CardId::ArcheryCard, PowerKind::Archery, 4, 5),
        (CardId::EcstasyCard, PowerKind::Ecstasy, 1, 1),
        (
            CardId::ExplosiveArrowsCard,
            PowerKind::ExplosiveArrows,
            6,
            8,
        ),
        (CardId::HiddenRootsCard, PowerKind::HiddenRoots, 3, 4),
        (CardId::HoningCard, PowerKind::Honing, 50, 50),
        (CardId::KiVerseCard, PowerKind::KiVerse, 1, 2),
        (CardId::LostLuteCard, PowerKind::LostLute, 1, 1),
        (CardId::LullabyCard, PowerKind::Lullaby, 20, 20),
        (CardId::MotifCard, PowerKind::Motif, 1, 1),
        (CardId::QuietusCard, PowerKind::Quietus, 2, 3),
        (
            CardId::SerratedArrowsCard,
            PowerKind::SerratedArrows,
            25,
            50,
        ),
        (CardId::SimpletonCard, PowerKind::Simpleton, 1, 1),
        (CardId::SongPathCard, PowerKind::SongPath, 5, 7),
        (CardId::TuneCard, PowerKind::Tune, 1, 1),
        (CardId::ViolentChordCard, PowerKind::ViolentChord, 9, 12),
        (CardId::YahCard, PowerKind::Yah, 1, 2),
        (CardId::Nock, PowerKind::Aim, 5, 7),
    ] {
        for (up_flag, n) in [(false, base), (true, up)] {
            let mut s = fresh(V113, CharacterId::Ironclad);
            s.player.energy = 9;
            play(&mut s, id, up_flag, None);
            assert_eq!(player_power(&s, kind), n, "{id:?} up={up_flag}");
        }
    }
}

#[test]
fn power_cost_upgrades() {
    for (id, base, up) in [
        (CardId::EcstasyCard, 2, 1),
        (CardId::HoningCard, 3, 2),
        (CardId::LostLuteCard, 3, 2),
        (CardId::MotifCard, 2, 1),
        (CardId::SimpletonCard, 2, 1),
        (CardId::TuneCard, 2, 1),
    ] {
        let c = card_pod_from_id_for(id, false, V113).expect("spec");
        assert_eq!(c.cost.base(), base, "{id:?}");
        let c = card_pod_from_id_for(id, true, V113).expect("spec");
        assert_eq!(c.cost.base(), up, "{id:?}+");
    }
}

#[test]
fn cho_barbs_cho_ready_dislike_you() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::ChoBarbsCard, true, None);
    assert_eq!(s.player.block, 9);
    assert_eq!(player_power(&s, PowerKind::ChoBarbs), 4);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::ChoReady, false, None);
    assert_eq!(s.player.block, 8);
    assert_eq!(player_power(&s, PowerKind::Aim), 3);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::DislikeYouCard, true, Some(0));
    assert_eq!(enemy_hp(&s), 91);
    assert_eq!(enemy_power(&s, first_enemy(&s), PowerKind::DislikeYou), 75);
}

#[test]
fn brace_applies_aim_x_times() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    s.player.energy = 3;
    play(&mut s, CardId::Brace, false, None);
    assert_eq!(s.player.energy, 0);
    assert_eq!(player_power(&s, PowerKind::Aim), 24);
}

#[test]
fn deep_stretch_and_encore() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..4 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::DeepStretch, true, None);
    assert_eq!(player_power(&s, PowerKind::Stretch), 10);
    assert_eq!(s.player.hand.len(), 1);

    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..4 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::EncoreCard, true, None);
    assert_eq!(s.player.energy, 6);
    assert_eq!(s.player.hand.len(), 3);
    assert_eq!(player_power(&s, PowerKind::Encore), 1);
}

#[test]
fn wheel_of_fortune_discards_and_draws() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..12 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    let _ = push_hand(&mut s, CardId::DefendFool, false);
    let _ = push_hand(&mut s, CardId::DefendFool, false);
    play(&mut s, CardId::WheelOfFortune, false, None);
    assert_eq!(s.player.hand.len(), 7);
    assert_eq!(s.player.discard.len(), 2);
    assert!(!hand_ids(&s).contains(&CardId::DefendFool));

    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..12 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::WheelOfFortune, true, None);
    assert_eq!(s.player.hand.len(), 9, "empty hand: plain Draw(9)");
}

#[test]
fn the_moon_fills_the_hand_with_shivs() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    let _ = push_hand(&mut s, CardId::DefendFool, false);
    play(&mut s, CardId::TheMoon, true, None);
    let h = hand_ids(&s);
    assert_eq!(h.len(), 10);
    assert_eq!(h.iter().filter(|c| **c == CardId::Shiv).count(), 9);
    assert!(s
        .player
        .hand
        .iter()
        .skip(1)
        .all(|u| s.cards[u.get() as usize - 1].upgraded));
}

#[test]
fn the_magician_slots_and_orbs() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::TheMagician, false, None);
    assert_eq!(s.player.orbs.capacity(), 3, "Ironclad 0 + 3 slots");
    assert_eq!(s.player.orbs.filled_len(), 3, "3 random orbs channelled");
}

#[test]
fn the_sun_draw_energy_soot() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..5 {
        push_draw(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::TheSun, true, None);
    assert_eq!(s.player.hand.len(), 3);
    assert_eq!(s.player.energy, 8);
    let disc: Vec<CardId> = s
        .player
        .discard
        .iter()
        .map(|u| s.cards[u.get() as usize - 1].card_id)
        .collect();
    assert_eq!(disc, vec![CardId::Soot]);
}

fn push_discard(state: &mut CombatState, id: CardId) {
    let uid = new_card(state, id, false);
    state.player.discard.push(uid);
}

fn count_in(state: &CombatState, pile: &[sts2_core::CardUid], id: CardId) -> usize {
    pile.iter()
        .filter(|u| state.cards[u.get() as usize - 1].card_id == id)
        .count()
}

#[test]
fn the_world_returns_discard_cards_up_to_the_hand_limit() {
    // TheWorld.cs: min(5 (+2), MaxCardsInHand - hand) from the discard pile.
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..8 {
        push_discard(&mut s, CardId::StrikeFool);
    }
    for _ in 0..7 {
        let _ = push_hand(&mut s, CardId::DefendFool, false);
    }
    play(&mut s, CardId::TheWorld, false, None);
    assert_eq!(s.player.hand.len(), 10, "7 + min(5, 3)");
    assert_eq!(s.player.discard.len(), 5);

    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..8 {
        push_discard(&mut s, CardId::StrikeFool);
    }
    play(&mut s, CardId::TheWorld, true, None);
    assert_eq!(s.player.hand.len(), 7, "upgraded: 7");
    assert_eq!(s.player.discard.len(), 1);
    assert_eq!(count_in(&s, &s.player.exhaust.clone(), CardId::TheWorld), 1);
}

#[test]
fn the_hierophant_three_distinct_colorless_one_cheaper() {
    for (up, n) in [(false, 3), (true, 4)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        play(&mut s, CardId::TheHierophant, up, None);
        assert_eq!(s.player.hand.len(), n, "up={up}");
        let ids = hand_ids(&s);
        let mut dedup = ids.clone();
        dedup.sort_by_key(|c| *c as u32);
        dedup.dedup();
        assert_eq!(dedup.len(), n, "distinct");
        for u in &s.player.hand {
            let c = &s.cards[u.get() as usize - 1];
            let base = i32::from(c.cost.base());
            if base > 0 {
                assert_eq!(c.cost.local(), base - 1, "{:?}", c.card_id);
            }
        }
    }
}

#[test]
fn the_lovers_fills_the_hand_from_other_pools() {
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..7 {
        let _ = push_hand(&mut s, CardId::DefendFool, false);
    }
    play(&mut s, CardId::TheLovers, false, None);
    assert_eq!(s.player.hand.len(), 10);

    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::TheLovers, true, None);
    assert_eq!(s.player.hand.len(), 10);
    let ironclad: Vec<CardId> = sts2_content::cards::spec_index::IRONCLAD_SPECS
        .iter()
        .map(|f| f().id)
        .collect();
    for u in &s.player.hand {
        let c = &s.cards[u.get() as usize - 1];
        assert!(c.upgraded, "{:?} upgraded", c.card_id);
        assert!(
            !ironclad.contains(&c.card_id),
            "{:?} not Ironclad",
            c.card_id
        );
    }
}

#[test]
fn judgement_kills_at_or_below_threshold() {
    for (up, hp, dies) in [
        (false, 30, true),
        (false, 31, false),
        (true, 40, true),
        (true, 41, false),
    ] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        let _ = add_enemy(&mut s, 50, 100);
        s.creatures[0].current_hp = hp;
        play(&mut s, CardId::Judgement, up, Some(0));
        assert_eq!(enemy_hp(&s) <= 0, dies, "up={up} hp={hp}");
        if !dies {
            assert_eq!(enemy_hp(&s), hp);
        }
    }
}

#[test]
fn growth_exhausts_sadness_for_stretch() {
    for (up, per) in [(false, 4), (true, 6)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        let _ = push_hand(&mut s, CardId::Sadness, false);
        let _ = push_hand(&mut s, CardId::StrikeFool, false);
        let _ = push_hand(&mut s, CardId::Sadness, false);
        play(&mut s, CardId::Growth, up, None);
        assert_eq!(hand_ids(&s), vec![CardId::StrikeFool]);
        assert_eq!(count_in(&s, &s.player.exhaust.clone(), CardId::Sadness), 2);
        assert_eq!(player_power(&s, PowerKind::Stretch), 2 * per);
    }
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::Growth, false, None);
    assert_eq!(player_power(&s, PowerKind::Stretch), 0);
}

#[test]
fn punch_of_joy_turns_one_sadness_into_joy() {
    for (up, dmg) in [(false, 13), (true, 17)] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        let _ = push_hand(&mut s, CardId::Sadness, false);
        let _ = push_hand(&mut s, CardId::Sadness, false);
        play(&mut s, CardId::PunchOfJoy, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg);
        let h = hand_ids(&s);
        assert_eq!(h.len(), 2);
        assert_eq!(h.iter().filter(|c| **c == CardId::Joy).count(), 1);
        assert_eq!(h.iter().filter(|c| **c == CardId::Sadness).count(), 1);
    }
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::PunchOfJoy, false, Some(0));
    assert!(hand_ids(&s).is_empty());
}

#[test]
fn locked_in_fetches_a_shoot_card_from_draw() {
    // One Shoot candidate: FirstOrDefault, no prompt.
    let mut s = fresh(V113, CharacterId::Ironclad);
    push_draw(&mut s, CardId::StrikeFool);
    push_draw(&mut s, CardId::FoolishShot);
    push_draw(&mut s, CardId::DefendFool);
    play(&mut s, CardId::LockedIn, false, None);
    assert_eq!(hand_ids(&s), vec![CardId::FoolishShot]);
    assert_eq!(s.player.draw.len(), 2);

    // Two Shoot candidates: a 1-of-n select over the draw pile.
    let mut s = fresh(V113, CharacterId::Ironclad);
    push_draw(&mut s, CardId::FoolishShot);
    push_draw(&mut s, CardId::QuickShot);
    push_draw(&mut s, CardId::StrikeFool);
    play(&mut s, CardId::LockedIn, false, None);
    let pc = s.pending_choice.as_ref().expect("select");
    assert_eq!(pc.options.len(), 2);

    // No Shoot card: nothing.
    let mut s = fresh(V113, CharacterId::Ironclad);
    push_draw(&mut s, CardId::StrikeFool);
    play(&mut s, CardId::LockedIn, false, None);
    assert!(hand_ids(&s).is_empty());
    assert!(s.pending_choice.is_none());
}

#[test]
fn the_hermit_discards_or_exhausts_a_curse_then_draws() {
    // Lone hand card: no prompt. A Curse is exhausted.
    for (id, up, exhausted) in [
        (CardId::Doubt, false, true),
        (CardId::StrikeFool, true, false),
    ] {
        let mut s = fresh(V113, CharacterId::Ironclad);
        for _ in 0..5 {
            push_draw(&mut s, CardId::DefendFool);
        }
        let _ = push_hand(&mut s, id, false);
        play(&mut s, CardId::TheHermit, up, None);
        let draw = if up { 3 } else { 2 };
        assert_eq!(s.player.hand.len(), draw, "{id:?}");
        assert_eq!(
            count_in(&s, &s.player.exhaust.clone(), id),
            usize::from(exhausted)
        );
        assert_eq!(
            count_in(&s, &s.player.discard.clone(), id),
            usize::from(!exhausted)
        );
    }

    // Two hand cards: a 1-card DiscardFromHand select; picking the Curse
    // exhausts it, then draws 2.
    let mut s = fresh(V113, CharacterId::Ironclad);
    for _ in 0..5 {
        push_draw(&mut s, CardId::DefendFool);
    }
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    let _ = push_hand(&mut s, CardId::Doubt, false);
    play(&mut s, CardId::TheHermit, false, None);
    let pc = s.pending_choice.clone().expect("select");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::DiscardFromHand);
    assert_eq!(pc.options.len(), 2);
    let pick = u8::try_from(
        pc.options
            .iter()
            .position(|o| o.card_id == CardId::Doubt)
            .expect("Doubt offered"),
    )
    .expect("u8");
    let _ = sts2_core::resolve_choice_atomic_apply(&mut s, &[pick]);
    assert!(s.pending_choice.is_none());
    assert_eq!(count_in(&s, &s.player.exhaust.clone(), CardId::Doubt), 1);
    assert_eq!(s.player.hand.len(), 3, "Strike + 2 drawn");
}

// ───────────── arrows (core `sts2_core::fool`) ─────────────

use sts2_core::fool;

fn fool_state() -> CombatState {
    let mut s = fresh(V113, CharacterId::Fool);
    s.player.energy = 10;
    s
}

/// Shoot the Fool's arrow into the first enemy (FoolishShot, 15).
fn impale_first(s: &mut CombatState) {
    play(s, CardId::FoolishShot, false, Some(0));
    assert!(fool::creature_is_impaled(s, first_enemy(s)));
}

#[test]
fn shots_hit_impale_and_need_an_arrow() {
    for (id, up, dmg) in [
        (CardId::FoolishShot, false, 15),
        (CardId::FoolishShot, true, 20),
        (CardId::QuickShot, false, 10),
        (CardId::SkyShot, true, 40),
        (CardId::DivineShot, false, 40),
        (CardId::HeavyShot, false, 20),
    ] {
        let mut s = fool_state();
        play(&mut s, id, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg, "{id:?} up={up}");
        assert!(fool::creature_is_impaled(&s, first_enemy(&s)));
        assert_eq!(fool::available_arrow_count(&s), 0);
        // Quiver empty: the next shot does nothing.
        play(&mut s, CardId::QuickShot, false, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg, "{id:?}: no arrow, no attack");
    }
    // HeavyShot doubles Aim (core AimPower arm): 20 + 2 x 3.
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Aim, 3);
    play(&mut s, CardId::HeavyShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 26);
}

#[test]
fn rapid_repeat_hobbling_melodic_shots() {
    for (up, hits) in [(false, 3), (true, 4)] {
        let mut s = fool_state();
        play(&mut s, CardId::RapidShot, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - 3 * hits);
    }
    // RepeatShot: X = 3 energy -> 3 hits of 15.
    let mut s = fool_state();
    s.player.energy = 3;
    play(&mut s, CardId::RepeatShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 45);
    assert_eq!(s.player.energy, 0);
    // HobblingShot: debuffs only when an arrow flew.
    let mut s = fool_state();
    play(&mut s, CardId::HobblingShot, true, Some(0));
    let e = first_enemy(&s);
    assert_eq!(enemy_hp(&s), 100 - 14);
    assert_eq!(enemy_power(&s, e, PowerKind::Weak), 2);
    assert_eq!(enemy_power(&s, e, PowerKind::Vulnerable), 2);
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::HobblingShot, false, Some(0));
    assert_eq!(
        enemy_power(&s, first_enemy(&s), PowerKind::Weak),
        1,
        "Shoot card gives Ironclad a slot"
    );
    // MelodicShot: 1 + melodies hits; after a Wah card, 2 hits of 12.
    let mut s = fool_state();
    play(&mut s, CardId::WahSweep, false, None);
    let hp0 = enemy_hp(&s);
    play(&mut s, CardId::MelodicShot, false, Some(0));
    assert_eq!(hp0 - enemy_hp(&s), 24);
}

#[test]
fn impale_readers_bow_smack_ki_hop_nail_wah_hammer() {
    // BowSmack: +1 energy only when the target was impaled.
    let mut s = fool_state();
    play(&mut s, CardId::BowSmack, false, Some(0));
    assert_eq!(s.player.energy, 9);
    impale_first(&mut s);
    let e = s.player.energy;
    play(&mut s, CardId::BowSmack, false, Some(0));
    assert_eq!(s.player.energy, e);

    // KiHop: Draw 2 (+1) when impaled.
    let mut s = fool_state();
    for _ in 0..5 {
        push_draw(&mut s, CardId::DefendFool);
    }
    impale_first(&mut s);
    play(&mut s, CardId::KiHop, true, Some(0));
    assert_eq!(s.player.hand.len(), 3);

    // Nail: nothing unless impaled; then 20 + Weak/Vulnerable 2.
    let mut s = fool_state();
    play(&mut s, CardId::Nail, false, Some(0));
    assert_eq!(enemy_hp(&s), 100);
    impale_first(&mut s);
    play(&mut s, CardId::Nail, false, Some(0));
    let en = first_enemy(&s);
    assert_eq!(enemy_power(&s, en, PowerKind::Weak), 2);
    assert_eq!(enemy_power(&s, en, PowerKind::Vulnerable), 2);

    // WahHammer: x2 against an impaled target.
    let mut s = fool_state();
    play(&mut s, CardId::WahHammer, false, Some(0));
    assert_eq!(enemy_hp(&s), 91);
    impale_first(&mut s);
    let hp0 = enemy_hp(&s);
    play(&mut s, CardId::WahHammer, false, Some(0));
    assert_eq!(hp0 - enemy_hp(&s), 18);
}

#[test]
fn dislodgers_uproot_wrest_recall() {
    let mut s = fool_state();
    impale_first(&mut s);
    play(&mut s, CardId::Uproot, false, Some(0));
    let e = first_enemy(&s);
    assert!(!fool::creature_is_impaled(&s, e));
    assert_eq!(
        fool::available_arrow_count(&s),
        1,
        "arrow back in the quiver"
    );
    assert_eq!(enemy_power(&s, e, PowerKind::Vulnerable), 3);
    play(&mut s, CardId::Uproot, true, Some(0));
    assert_eq!(
        enemy_power(&s, e, PowerKind::Vulnerable),
        3,
        "not impaled: nothing"
    );

    let mut s = fool_state();
    for _ in 0..5 {
        push_draw(&mut s, CardId::DefendFool);
    }
    impale_first(&mut s);
    play(&mut s, CardId::Wrest, false, Some(0));
    assert_eq!(s.player.hand.len(), 2);

    // Recall: the lone discard card (the FoolishShot) goes on top of draw.
    let mut s = fool_state();
    impale_first(&mut s);
    assert_eq!(s.player.discard.len(), 1);
    play(&mut s, CardId::Recall, false, Some(0));
    let top = s.player.draw[0];
    assert_eq!(s.cards[top.get() as usize - 1].card_id, CardId::FoolishShot);
    assert_eq!(fool::available_arrow_count(&s), 1);
}

#[test]
fn arrow_stab_cho_firm_quiver_blossom() {
    // ArrowStab: with an arrow ready, a Shoot card from discard to hand.
    let mut s = fool_state();
    push_discard(&mut s, CardId::StrikeFool);
    push_discard(&mut s, CardId::QuickShot);
    play(&mut s, CardId::ArrowStab, false, Some(0));
    assert_eq!(enemy_hp(&s), 91);
    assert_eq!(hand_ids(&s), vec![CardId::QuickShot]);

    // ChoFirm: block twice with an arrow, once without.
    let mut s = fool_state();
    play(&mut s, CardId::ChoFirm, false, None);
    assert_eq!(s.player.block, 12);
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::ChoFirm, true, None);
    assert_eq!(s.player.block, 8);

    // Quiver: 1 (+1) slots.
    let mut s = fool_state();
    play(&mut s, CardId::Quiver, true, None);
    assert_eq!(fool::arrow_slot_count(&s), 3);

    // BlossomShot: Stretch = damage dealt (11).
    let mut s = fool_state();
    play(&mut s, CardId::BlossomShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 89);
    assert_eq!(player_power(&s, PowerKind::Stretch), 11);
}

#[test]
fn florescence_exhausts_shoot_cards_into_the_shot() {
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::FoolishShot, false);
    push_draw(&mut s, CardId::QuickShot);
    push_discard(&mut s, CardId::RapidShot);
    push_discard(&mut s, CardId::StrikeFool);
    play(&mut s, CardId::Florescence, true, None);
    assert_eq!(
        count_in(&s, &s.player.exhaust.clone(), CardId::QuickShot),
        1
    );
    assert_eq!(s.player.exhaust.len(), 4, "3 Shoot cards + Florescence");
    let h = s.player.hand.clone();
    assert_eq!(h.len(), 1);
    let shot = &s.cards[h[0].get() as usize - 1];
    assert_eq!(shot.card_id, CardId::TheShot);
    assert!(shot.upgraded);
    let _ = sts2_core::combat_loop::play_card(&mut s, 0, Some(0));
    assert_eq!(enemy_hp(&s), 100 - 3 * 13);
}

// ───────────── melody ─────────────

fn cost_in_hand(s: &CombatState, i: usize) -> i32 {
    sts2_core::hooks::effective_card_cost(s, s.player.hand[i])
}

#[test]
fn coda_and_cho_limbs_cost_less_after_melodies() {
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::Coda, false);
    let _ = push_hand(&mut s, CardId::ChoLimbs, false);
    assert_eq!(cost_in_hand(&s, 0), 6);
    assert_eq!(cost_in_hand(&s, 1), 2);
    play(&mut s, CardId::ChoReady, false, None); // Cho: Coda -2, ChoLimbs unchanged
    assert_eq!(cost_in_hand(&s, 0), 4);
    assert_eq!(cost_in_hand(&s, 1), 2);
    play(&mut s, CardId::WahTouch, false, Some(0)); // Wah: Coda -4, ChoLimbs -2
    assert_eq!(cost_in_hand(&s, 0), 2);
    assert_eq!(cost_in_hand(&s, 1), 0);
}

#[test]
fn melody_scaled_cards() {
    // HarmonicShield: 5 (+2) x melodies.
    // WahSweep (Wah) and ChoFirm (Cho): no debuff, no Aim.
    let mut s = fool_state();
    play(&mut s, CardId::WahSweep, false, None);
    play(&mut s, CardId::ChoFirm, false, None);
    let b0 = s.player.block;
    play(&mut s, CardId::HarmonicShield, true, None);
    assert_eq!(s.player.block - b0, 14);
    // RhythmStrike: 1 + 2 melodies = 3 hits of 4.
    let hp0 = enemy_hp(&s);
    play(&mut s, CardId::RhythmStrike, false, Some(0));
    assert_eq!(hp0 - enemy_hp(&s), 12);
    // SteadyCadence: draws melodies (2).
    for _ in 0..4 {
        push_draw(&mut s, CardId::DefendFool);
    }
    let h0 = s.player.hand.len();
    play(&mut s, CardId::SteadyCadence, false, Some(0));
    assert_eq!(s.player.hand.len() - h0, 2);
    // WahBlast hits twice after Cho.
    let hp0 = enemy_hp(&s);
    play(&mut s, CardId::WahBlast, false, None);
    assert_eq!(hp0 - enemy_hp(&s), 28);
    // KiRest: Retain once Wah or Cho was played.
    play(&mut s, CardId::KiRest, false, None);
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 5);
    assert_eq!(player_power(&s, PowerKind::RetainHand), 1);

    let mut s = fool_state();
    play(&mut s, CardId::WahBlast, false, None);
    assert_eq!(enemy_hp(&s), 86, "no Cho/Ki: one hit");
    play(&mut s, CardId::KiRest, true, None);
    assert_eq!(player_power(&s, PowerKind::EnergyNextTurn), 6);
    assert_eq!(
        player_power(&s, PowerKind::RetainHand),
        1,
        "WahBlast is Wah"
    );
}

#[test]
fn wah_symphony_counts_melody_cards() {
    let mut s = fool_state();
    push_draw(&mut s, CardId::WahTouch);
    push_discard(&mut s, CardId::ChoKi);
    push_discard(&mut s, CardId::StrikeFool);
    // WahTouch + ChoKi + WahSymphony itself (on Play) = 3.
    play(&mut s, CardId::WahSymphony, false, None);
    assert_eq!(enemy_hp(&s), 100 - (6 + 4 * 3));
}

#[test]
fn cho_chorus_autoplays_wah_and_ki_hand_cards() {
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::WahTouch, false);
    let _ = push_hand(&mut s, CardId::StrikeFool, false);
    let _ = push_hand(&mut s, CardId::KiWhip, false);
    let e0 = s.player.energy;
    play(&mut s, CardId::ChoChorus, false, None);
    assert_eq!(s.player.block, 12);
    assert_eq!(hand_ids(&s), vec![CardId::StrikeFool]);
    assert_eq!(e0 - s.player.energy, 3, "only ChoChorus is paid for");
    assert!(enemy_hp(&s) < 100);
}

// ───────────── auto-play tarot ─────────────

#[test]
fn the_fool_autoplays_random_hand_cards() {
    for (up, n) in [(false, 2), (true, 3)] {
        let mut s = fool_state();
        for _ in 0..4 {
            let _ = push_hand(&mut s, CardId::StrikeFool, false);
        }
        play(&mut s, CardId::TheFool, up, None);
        assert_eq!(s.player.hand.len(), 4 - n, "up={up}");
        assert_eq!(enemy_hp(&s), 100 - 6 * n as i32);
    }
    // Only unplayable cards: the fallback still auto-plays one (Sadness goes
    // through the unplayable auto-play path, no damage).
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::Sadness, false);
    play(&mut s, CardId::TheFool, false, None);
    assert!(hand_ids(&s).is_empty());
}

#[test]
fn the_devil_exhausts_the_hand_and_plays_from_draw() {
    for (up, dmg) in [(false, 12), (true, 18)] {
        let mut s = fool_state();
        let _ = push_hand(&mut s, CardId::DefendFool, false);
        let _ = push_hand(&mut s, CardId::DefendFool, false);
        for _ in 0..3 {
            push_draw(&mut s, CardId::StrikeFool);
        }
        play(&mut s, CardId::TheDevil, up, None);
        assert_eq!(
            count_in(&s, &s.player.exhaust.clone(), CardId::DefendFool),
            2
        );
        assert_eq!(s.player.draw.len(), 1);
        assert_eq!(enemy_hp(&s), 100 - dmg, "up={up}");
    }
}

#[test]
fn cho_ki_whistling_arrows_blot_out_the_sky() {
    // ChoKi: block 8 (+3) plus StretchPower's +4 bonus; Stretch doubles
    // (4 -> 8), then `StretchPower.AfterCardPlayed` spends the 4 it had when
    // the play started.
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Stretch, 4);
    play(&mut s, CardId::ChoKi, true, None);
    assert_eq!(s.player.block, 15);
    assert_eq!(player_power(&s, PowerKind::Stretch), 4);
    let mut s = fool_state();
    play(&mut s, CardId::ChoKi, false, None);
    assert_eq!(player_power(&s, PowerKind::Stretch), 0);

    // WhistlingArrows: block 3 (+1).
    let mut s = fool_state();
    play(&mut s, CardId::WhistlingArrows, true, None);
    assert_eq!(s.player.block, 4);

    // BlotOutTheSky: one free Shoot card from the Fool pool.
    let mut s = fool_state();
    play(&mut s, CardId::BlotOutTheSky, false, None);
    let h = s.player.hand.clone();
    assert_eq!(h.len(), 1);
    let c = &s.cards[h[0].get() as usize - 1];
    assert!(
        c.has_tag(sts2_core::cards::CardTag::Shoot),
        "{:?}",
        c.card_id
    );
    assert_eq!(sts2_core::hooks::effective_card_cost(&s, h[0]), 0);
    // Ironclad's pool has no Shoot card: nothing.
    let mut s = fresh(V113, CharacterId::Ironclad);
    play(&mut s, CardId::BlotOutTheSky, false, None);
    assert!(s.player.hand.is_empty());
}

#[test]
fn trick_shot_returns_to_hand_only_with_an_arrow_at_play_start() {
    // TrickShot.cs: 12 damage shot; result Discard -> Hand when
    // `PlayerHasAnyArrows` (read before OnPlay).
    let mut s = fool_state();
    play(&mut s, CardId::TrickShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 88);
    assert_eq!(hand_ids(&s), vec![CardId::TrickShot]);
    assert_eq!(count_in(&s, &s.player.discard, CardId::TrickShot), 0);
    // The one arrow is now in the enemy: the next play shoots nothing and
    // goes to the discard pile.
    let _ = sts2_core::combat_loop::play_card(&mut s, 0, Some(0));
    assert_eq!(enemy_hp(&s), 88);
    assert!(s.player.hand.is_empty());
    assert_eq!(count_in(&s, &s.player.discard, CardId::TrickShot), 1);
    assert_eq!(
        s.cards[s.player.discard[0].get() as usize - 1]
            .combat_var(sts2_core::cards::VarKey::Generic),
        0
    );
}

#[test]
fn mourning_blow_returns_to_hand_as_sadness() {
    // MourningBlow.cs: 10 (+5) damage; TransformIntoAfterPlay Sadness;
    // result Discard -> Hand.
    for (up, dmg) in [(false, 10), (true, 15)] {
        let mut s = fool_state();
        play(&mut s, CardId::MourningBlow, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg);
        assert_eq!(hand_ids(&s), vec![CardId::Sadness]);
        assert!(s.player.discard.is_empty());
        assert!(s.player.play.is_empty());
    }
}

#[test]
fn rain_shot_splashes_total_plus_overkill_to_the_others() {
    // RainShot.cs: 12 FromArrow to the target, then TotalDamage +
    // OverkillDamage (Cap hooks only: no Vulnerable) to the other enemies.
    let mut s = fool_state();
    let e2 = add_enemy(&mut s, 20, 50);
    let e3 = add_enemy(&mut s, 21, 50);
    add_enemy_power(&mut s, e3, 40, PowerKind::Vulnerable, 2);
    s.creatures[0].current_hp = 5;
    play(&mut s, CardId::RainShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 0);
    assert_eq!(hp_of(&s, e2), 38);
    assert_eq!(hp_of(&s, e3), 38);
    // No arrow: nothing happens.
    let mut s = fool_state();
    let e2 = add_enemy(&mut s, 20, 50);
    impale_first(&mut s);
    play(&mut s, CardId::RainShot, true, Some(0));
    assert_eq!(enemy_hp(&s), 85);
    assert_eq!(hp_of(&s, e2), 50);
}

#[test]
fn ki_chant_picks_one_of_the_top_cards_and_discards_the_rest() {
    // KiChant.cs: FromCombatPile(Draw, 1, top 3 (+1)); others discarded.
    let mut s = fool_state();
    for id in [
        CardId::StrikeFool,
        CardId::DefendFool,
        CardId::Jest,
        CardId::Vitriol,
    ] {
        push_draw(&mut s, id);
    }
    play(&mut s, CardId::KiChant, false, None);
    let pc = s.pending_choice.as_ref().expect("top-3 select");
    assert_eq!(pc.options.len(), 3);
    let pick = pc
        .options
        .iter()
        .position(|o| o.card_id == CardId::Jest)
        .expect("Jest offered");
    let _ = sts2_core::resolve_choice_atomic_apply(&mut s, &[u8::try_from(pick).unwrap()]);
    assert_eq!(hand_ids(&s), vec![CardId::Jest]);
    assert_eq!(count_in(&s, &s.player.discard, CardId::StrikeFool), 1);
    assert_eq!(count_in(&s, &s.player.discard, CardId::DefendFool), 1);
    assert_eq!(count_in(&s, &s.player.discard, CardId::KiChant), 1);
    assert_eq!(s.player.draw.len(), 1);
}

#[test]
fn ki_chant_discards_the_rest_in_draw_order() {
    // `topCards.Except(picked)` keeps the draw order (Vitriol, Jest on top;
    // the grid sorts them by rarity instead).
    let mut s = fool_state();
    for id in [CardId::Vitriol, CardId::Jest, CardId::StrikeFool] {
        push_draw(&mut s, id);
    }
    play(&mut s, CardId::KiChant, false, None);
    let pc = s.pending_choice.as_ref().expect("select");
    let pick = pc
        .options
        .iter()
        .position(|o| o.card_id == CardId::StrikeFool)
        .expect("offered");
    let _ = sts2_core::resolve_choice_atomic_apply(&mut s, &[u8::try_from(pick).unwrap()]);
    let ids: Vec<CardId> = s
        .player
        .discard
        .iter()
        .map(|u| s.cards[u.get() as usize - 1].card_id)
        .collect();
    assert_eq!(ids, vec![CardId::Vitriol, CardId::Jest, CardId::KiChant]);
}

#[test]
fn ki_chant_short_draw_shuffles_under_the_kept_cards() {
    // Fewer than 3 in draw, discard not empty: Shuffle, then the old draw
    // cards go back on top in order; a single top card is taken unasked.
    let mut s = fool_state();
    push_draw(&mut s, CardId::Jest);
    push_discard(&mut s, CardId::StrikeFool);
    push_discard(&mut s, CardId::DefendFool);
    play(&mut s, CardId::KiChant, false, None);
    // Top 3 = Jest + the two shuffled cards -> a select.
    let pc = s.pending_choice.as_ref().expect("select");
    assert_eq!(pc.options.len(), 3);
    let mut s = fool_state();
    push_discard(&mut s, CardId::Jest);
    play(&mut s, CardId::KiChant, false, None);
    assert!(s.pending_choice.is_none());
    assert_eq!(hand_ids(&s), vec![CardId::Jest]);
}

#[test]
fn anguish_counts_sadness_generated_this_combat() {
    // Anguish.cs: 6 (+1) + 6 (+1) x CardGeneratedEntry(Sadness).
    let mut s = fool_state();
    play(&mut s, CardId::Anguish, false, Some(0));
    assert_eq!(enemy_hp(&s), 94);
    // MourningBlow generates one Sadness (its transform), SelfDeprecate one.
    play(&mut s, CardId::MourningBlow, false, Some(0)); // 84
    play(&mut s, CardId::SelfDeprecate, false, None);
    assert_eq!(count_in(&s, &s.player.hand, CardId::Sadness), 2);
    play(&mut s, CardId::Anguish, true, Some(0));
    assert_eq!(enemy_hp(&s), 84 - (7 + 7 * 2));
    // A Sadness from the starting deck does not count.
    let mut s = fool_state();
    push_draw(&mut s, CardId::Sadness);
    play(&mut s, CardId::Anguish, false, Some(0));
    assert_eq!(enemy_hp(&s), 94);
}

#[test]
fn the_hanged_man_loses_hp_and_takes_an_extra_turn() {
    // TheHangedMan.cs: 6 (-3) unblockable HP loss, ExtraTurnPower 1.
    for (up, loss) in [(false, 6), (true, 3)] {
        let mut s = fool_state();
        s.player.block = 10;
        play(&mut s, CardId::TheHangedMan, up, None);
        assert_eq!(s.player.current_hp, 80 - loss);
        assert_eq!(s.player.block, 10);
        assert_eq!(player_power(&s, PowerKind::Ambergris), 1);
        assert_eq!(count_in(&s, &s.player.exhaust, CardId::TheHangedMan), 1);
    }
}

#[test]
fn bounce_applies_stretch_and_returns_next_turn() {
    // Bounce.cs: Stretch 10 (+3); BeforeHandDraw returns it to hand when it
    // was played last player turn.
    let mut s = fool_state();
    play(&mut s, CardId::Bounce, true, None);
    assert_eq!(player_power(&s, PowerKind::Stretch), 13);
    assert_eq!(count_in(&s, &s.player.discard, CardId::Bounce), 1);
    let played_turn = s.cards[s.player.discard[0].get() as usize - 1]
        .combat_var(sts2_core::cards::VarKey::ReturnIfPlayedRound);
    assert_eq!(played_turn, i32::from(s.turn_number()));
}

fn player_power_counter(s: &CombatState, kind: PowerKind) -> i32 {
    let cuid = s.player.creature_uid.expect("mirror");
    s.creatures
        .iter()
        .find(|c| c.id == cuid.0)
        .expect("player creature")
        .powers
        .iter()
        .chain(s.player.powers.iter())
        .filter_map(|p| s.powers.get_by_id(p.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.counter)
        .sum()
}

#[test]
fn ennui_mythic_form_and_show_off_set_their_power_state() {
    // Ennui.cs: EnnuiPower 2 (+1), IncrementSadnessAmount per play.
    let mut s = fool_state();
    play(&mut s, CardId::EnnuiCard, false, None);
    play(&mut s, CardId::EnnuiCard, true, None);
    assert_eq!(player_power(&s, PowerKind::Ennui), 5);
    assert_eq!(player_power_counter(&s, PowerKind::Ennui), 2);
    // MythicForm.cs: MythicFormPower 1; IncrementUpgraded only when upgraded.
    play(&mut s, CardId::MythicFormCard, false, None);
    play(&mut s, CardId::MythicFormCard, true, None);
    assert_eq!(player_power(&s, PowerKind::MythicForm), 2);
    assert_eq!(player_power_counter(&s, PowerKind::MythicForm), 1);
    // ShowOff.cs: Aim 5 + IncreasedAim, ShowOffPower 2 (+1) linked to the card.
    let mut s = fool_state();
    play(&mut s, CardId::ShowOffCard, true, None);
    assert_eq!(player_power(&s, PowerKind::Aim), 5);
    assert_eq!(player_power(&s, PowerKind::ShowOff), 3);
    let card = s.player.exhaust[0];
    assert_eq!(
        player_power_counter(&s, PowerKind::ShowOff),
        i32::try_from(card.get()).unwrap()
    );
}

#[test]
fn jackpot_generates_zero_cost_fool_cards_for_the_fool() {
    // Jackpot.cs: 3 cards from `Owner.Character.CardPool` with cost 0.
    let mut s = fool_state();
    play(&mut s, CardId::Jackpot, false, Some(0));
    assert_eq!(s.player.hand.len(), 3);
    for &u in &s.player.hand {
        let c = &s.cards[u.get() as usize - 1];
        assert!(
            sts2_content::cards::fool::is_fool_card(c.card_id),
            "{:?}",
            c.card_id
        );
        assert_eq!(c.base_cost(), 0, "{:?}", c.card_id);
    }
}

#[test]
fn the_tower_scales_with_total_floor() {
    // TheTower.cs: 2 (+1) x TotalFloor.
    for (up, dmg) in [(false, 2 * 17), (true, 3 * 17)] {
        let mut s = fool_state();
        s.total_floor = 17;
        play(&mut s, CardId::TheTower, up, Some(0));
        assert_eq!(enemy_hp(&s), 100 - dmg);
        assert_eq!(count_in(&s, &s.player.exhaust, CardId::TheTower), 1);
    }
}
