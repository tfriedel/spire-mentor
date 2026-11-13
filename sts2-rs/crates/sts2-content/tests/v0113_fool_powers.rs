//! v0.113.0 Fool powers owned by the core lane (`sts2_core::fool`,
//! `hooks`), on the real play path. Values from `Models.Powers/<Name>Power.cs`.
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

use sts2_core::fool;
use sts2_core::hooks::{self, TurnSideCtx};
use sts2_core::Side;

fn fool_state() -> CombatState {
    let mut s = fresh(V113, CharacterId::Fool);
    s.player.energy = 10;
    s
}

#[test]
fn free_card_zeroes_the_cost_and_spends_a_stack() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::FreeCard, 1);
    play(&mut s, CardId::DefendFool, false, None);
    assert_eq!(s.player.energy, 10);
    assert_eq!(player_power(&s, PowerKind::FreeCard), 0);
    play(&mut s, CardId::DefendFool, false, None);
    assert_eq!(s.player.energy, 9);
}

#[test]
fn ki_verse_replays_one_melody_card() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::KiVerse, 1);
    play(&mut s, CardId::WahTouch, false, Some(0));
    assert_eq!(fool::melody_cards_played_this_turn(&s), 2);
    assert_eq!(player_power(&s, PowerKind::KiVerse), 0);
    // A non-melody card is not replayed.
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::KiVerse, 1);
    play(&mut s, CardId::DefendFool, false, None);
    assert_eq!(s.player.block, 5);
    assert_eq!(player_power(&s, PowerKind::KiVerse), 1);
}

#[test]
fn encore_caps_plays_since_applied() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Encore, 2);
    play(&mut s, CardId::DefendFool, false, None);
    play(&mut s, CardId::DefendFool, false, None);
    let i = push_hand(&mut s, CardId::DefendFool, false);
    assert_eq!(
        sts2_core::combat_loop::play_card(&mut s, i, None),
        sts2_core::StepResult::IllegalPlay
    );
    assert_eq!(s.player.block, 10);
}

/// SongPath (Cho) and ViolentChord (Wah) fire on the turn's first play of
/// their note only: WahCho carries both.
#[test]
fn song_path_and_violent_chord_fire_on_the_first_note() {
    let mut base = fool_state();
    play(&mut base, CardId::WahCho, false, Some(0));
    let (b_block, b_hp) = (base.player.block, enemy_hp(&base));

    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::SongPath, 4);
    add_player_power(&mut s, 91, PowerKind::ViolentChord, 3);
    play(&mut s, CardId::WahCho, false, Some(0));
    assert_eq!(s.player.block, b_block + 4);
    assert_eq!(enemy_hp(&s), b_hp - 3);
    let (block, hp) = (s.player.block, enemy_hp(&s));
    play(&mut s, CardId::WahCho, false, Some(0));
    assert_eq!(s.player.block - block, b_block);
    assert_eq!(hp - enemy_hp(&s), 100 - b_hp);
}

#[test]
fn lullaby_blocks_once_after_all_three_notes() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Lullaby, 5);
    play(&mut s, CardId::WahCho, false, Some(0));
    let before = s.player.block;
    play(&mut s, CardId::WahKi, false, Some(0));
    let mut base = fool_state();
    play(&mut base, CardId::WahCho, false, Some(0));
    let b0 = base.player.block;
    play(&mut base, CardId::WahKi, false, Some(0));
    let wah_ki_block = base.player.block - b0;
    assert_eq!(s.player.block - before, wah_ki_block + 5);
    let before = s.player.block;
    play(&mut s, CardId::WahKi, false, Some(0));
    assert_eq!(s.player.block - before, wah_ki_block);
}

#[test]
fn motif_draws_on_the_turns_only_play_of_a_note() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Motif, 1);
    for _ in 0..4 {
        push_draw(&mut s, CardId::DefendFool);
    }
    play(&mut s, CardId::KiWhip, false, Some(0));
    assert_eq!(hand_ids(&s), vec![CardId::DefendFool]);
    play(&mut s, CardId::KiWhip, false, Some(0));
    assert_eq!(hand_ids(&s), vec![CardId::DefendFool]);
}

#[test]
fn ennui_adds_sadness_before_the_draw() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::Ennui, 1);
    // The Ennui card's `IncrementSadnessAmount`, twice: the power's counter.
    s.powers.iter_mut().for_each(|p| {
        if p.kind == PowerKind::Ennui {
            p.counter = 2;
        }
    });
    let _ = hooks::dispatch_before_hand_draw_powers(&mut s);
    assert_eq!(hand_ids(&s), vec![CardId::Sadness, CardId::Sadness]);
}

#[test]
fn ecstasy_turns_sadness_into_joy() {
    let mut s = fool_state();
    push_hand(&mut s, CardId::Sadness, false);
    push_hand(&mut s, CardId::DefendFool, false);
    push_hand(&mut s, CardId::Sadness, false);
    fool::ecstasy_after_player_turn_start(&mut s, 1);
    let h = hand_ids(&s);
    assert_eq!(h.iter().filter(|c| **c == CardId::Joy).count(), 1);
    assert_eq!(h.iter().filter(|c| **c == CardId::Sadness).count(), 1);
    assert_eq!(h[1], CardId::DefendFool);
}

#[test]
fn mythic_form_adds_tarots() {
    let mut s = fool_state();
    add_player_power(&mut s, 90, PowerKind::MythicForm, 2);
    s.powers.iter_mut().for_each(|p| {
        if p.kind == PowerKind::MythicForm {
            p.counter = 1;
        }
    });
    let _ = hooks::dispatch_before_hand_draw_powers(&mut s);
    assert_eq!(s.player.hand.len(), 2);
    let cards: Vec<_> = s
        .player
        .hand
        .iter()
        .map(|u| &s.cards[u.get() as usize - 1])
        .collect();
    assert!(cards.iter().all(|c| fool::TAROT_CARDS.contains(&c.card_id)));
    assert!(cards[0].upgraded && !cards[1].upgraded);
}

/// `TemporaryThornsPower`: Thorns with it, both gone after the enemy turn.
#[test]
fn cho_barbs_is_temporary_thorns() {
    let mut s = fool_state();
    let p = s.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut s, p, PowerKind::ChoBarbs, 3);
    assert_eq!(player_power(&s, PowerKind::Thorns), 3);
    hooks::dispatch_after_turn_end(&mut s, &TurnSideCtx { side: Side::Player });
    assert_eq!(player_power(&s, PowerKind::ChoBarbs), 3);
    hooks::dispatch_after_turn_end(&mut s, &TurnSideCtx { side: Side::Enemy });
    assert_eq!(player_power(&s, PowerKind::ChoBarbs), 0);
    assert_eq!(player_power(&s, PowerKind::Thorns), 0);
}

/// `TemporaryThornsPower.BeforeApplied` runs inside `PowerCmd.Apply` BEFORE
/// `ApplyInternal` installs ChoBarbs, so a fresh Thorns lands first in the
/// owner's power list; a stack (`ModifyAmount`) then adds its delta to
/// Thorns AFTER the change (`AfterPowerAmountChanged`, amount != Amount).
#[test]
fn cho_barbs_lands_its_thorns_first_and_stacks_follow() {
    let mut s = fool_state();
    let p = s.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut s, p, PowerKind::ChoBarbs, 3);
    let kinds: Vec<PowerKind> = s
        .creatures
        .iter()
        .find(|c| c.id == p.0)
        .expect("player creature")
        .powers
        .iter()
        .filter_map(|u| s.powers.get_by_id(u.0))
        .map(|pw| pw.kind)
        .filter(|k| matches!(k, PowerKind::ChoBarbs | PowerKind::Thorns))
        .collect();
    assert_eq!(kinds, [PowerKind::Thorns, PowerKind::ChoBarbs]);
    sts2_core::card_effects::apply_power_to_creature(&mut s, p, PowerKind::ChoBarbs, 2);
    assert_eq!(player_power(&s, PowerKind::ChoBarbs), 5);
    assert_eq!(player_power(&s, PowerKind::Thorns), 5);
}

/// `TemporaryThornsPower.AfterPowerAmountChanged` mirrors EVERY change of its
/// own amount (`amount != Amount`), negative too: `PowerCmd.ModifyAmount`
/// fires it before `ShouldRemoveDueToAmount`, so taking 2 off a 3 ChoBarbs
/// takes 2 off Thorns.
#[test]
fn cho_barbs_mirrors_a_negative_change_onto_thorns() {
    let mut s = fool_state();
    let p = s.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut s, p, PowerKind::ChoBarbs, 3);
    hooks::apply_effect(
        &mut s,
        hooks::Effect::ApplyPower {
            target: hooks::CreatureRef::Player,
            kind: PowerKind::ChoBarbs,
            amount: -2,
            applier: None,
        },
    );
    assert_eq!(player_power(&s, PowerKind::ChoBarbs), 1);
    assert_eq!(player_power(&s, PowerKind::Thorns), 1);
}

/// `ShowOffPower.AfterDeath`: a fatal enemy death adds Amount to the ShowOff
/// card's `IncreasedAim` (its `Generic` var) and removes the power.
#[test]
fn show_off_raises_the_cards_aim_on_a_kill() {
    let mut s = fool_state();
    let card = new_card(&mut s, CardId::ShowOffCard, false);
    add_player_power(&mut s, 90, PowerKind::ShowOff, 2);
    for p in s.powers.iter_mut().filter(|p| p.kind == PowerKind::ShowOff) {
        p.counter = i32::try_from(card.0.get()).expect("fits");
    }
    let e = first_enemy(&s);
    sts2_core::damage::apply_damage(
        &mut s,
        hooks::CreatureRef::Player,
        e,
        500,
        hooks::ValueProps::MOVE,
        None,
    );
    assert_eq!(
        s.cards[card.0.get() as usize - 1].combat_var(sts2_core::cards::VarKey::Generic),
        2
    );
    assert_eq!(player_power(&s, PowerKind::ShowOff), 0);
}

fn discard_ids(state: &CombatState) -> Vec<CardId> {
    state
        .player
        .discard
        .iter()
        .map(|u| state.cards[u.get() as usize - 1].card_id)
        .collect()
}

/// `SimpletonPower.AfterAutoPostPlayPhaseEntered`: auto-play one card from the
/// cheapest playable cost group in hand, picked off `CombatCardSelection`.
#[test]
fn simpleton_auto_plays_from_the_cheapest_group() {
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::Bash, false);
    let _ = push_hand(&mut s, CardId::StrikeIronclad, false);
    let _ = push_hand(&mut s, CardId::DefendIronclad, false);
    add_player_power(&mut s, 91, PowerKind::Simpleton, 1);
    let sel = s.run_rng.combat_card_selection.counter();
    let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut s);
    assert_eq!(s.run_rng.combat_card_selection.counter(), sel + 1);
    let played = discard_ids(&s);
    assert_eq!(played.len(), 1, "exactly one card auto-played: {played:?}");
    assert!(matches!(
        played[0],
        CardId::StrikeIronclad | CardId::DefendIronclad
    ));
    assert!(hand_ids(&s).contains(&CardId::Bash));
}

/// Unplayable cards never form the cheapest group; Amount loops re-read the hand.
#[test]
fn simpleton_skips_unplayable_and_loops_amount_times() {
    let mut s = fool_state();
    let _ = push_hand(&mut s, CardId::Wound, false);
    let _ = push_hand(&mut s, CardId::Bash, false);
    let _ = push_hand(&mut s, CardId::StrikeIronclad, false);
    add_player_power(&mut s, 91, PowerKind::Simpleton, 2);
    let hp = enemy_hp(&s);
    let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut s);
    let played = discard_ids(&s);
    assert_eq!(played, vec![CardId::StrikeIronclad, CardId::Bash]);
    assert_eq!(enemy_hp(&s), hp - 6 - 8);
}

/// Simpleton exists only from v0.113.0.
#[test]
fn simpleton_is_inert_before_v0113() {
    let mut s = fresh(V111, CharacterId::Ironclad);
    let _ = push_hand(&mut s, CardId::StrikeIronclad, false);
    add_player_power(&mut s, 91, PowerKind::Simpleton, 1);
    let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut s);
    assert!(discard_ids(&s).is_empty());
    assert_eq!(enemy_hp(&s), 100);
}

/// `Enrapture` targets `AnyAlly`: with no living teammate C# `CanPlay` fails
/// (`NoLivingAllies`), so solo it is never a legal play.
#[test]
fn enrapture_is_unplayable_solo() {
    let mut s = fool_state();
    let i = push_hand(&mut s, CardId::Enrapture, false);
    let offered = sts2_core::legal_actions(&s)
        .iter()
        .any(|a| a.card_play().is_some_and(|(h, _)| h == i));
    assert!(!offered, "Enrapture offered with no ally");
}

/// MythicForm's tarot pick is `CombatCardSelection.NextItem(TarotCards.All)`:
/// one half-open `NextInt(0, 22)` per tarot, indexing the decomp order.
#[test]
fn mythic_form_picks_tarots_half_open() {
    for seed in 0..64_i32 {
        let mut s = fool_state();
        s.run_rng.combat_card_selection = sts2_core::dotnet_rng::Rng::new(seed);
        let mut probe = s.run_rng.combat_card_selection.clone();
        let want: Vec<CardId> = (0..3)
            .map(|_| fool::TAROT_CARDS[probe.next_int_exclusive(0, 22) as usize])
            .collect();
        fool::mythic_form_before_hand_draw(&mut s, 3, 0);
        assert_eq!(hand_ids(&s), want, "seed {seed}");
    }
}

/// `DislikeYouPower` and `EncorePower` are `PowerType.Debuff`: an Artifact
/// stack negates them (and is consumed).
#[test]
fn dislike_you_is_a_debuff_artifact_negates() {
    let mut s = fool_state();
    let e = first_enemy(&s);
    add_enemy_power(&mut s, e, 92, PowerKind::Artifact, 1);
    play(&mut s, CardId::DislikeYouCard, false, Some(0));
    assert_eq!(enemy_power(&s, e, PowerKind::Artifact), 0);
    assert_eq!(enemy_power(&s, e, PowerKind::DislikeYou), 0);
    assert!(sts2_core::afflictions::classify(PowerKind::Encore).is_debuff());
}

/// `CompostPower.AfterDeath`: an enemy death gives the player Amount energy.
#[test]
fn compost_gains_energy_on_an_enemy_death() {
    let mut s = fool_state();
    let _ = add_enemy(&mut s, 7, 50);
    add_player_power(&mut s, 93, PowerKind::Compost, 2);
    let e = first_enemy(&s);
    let before = s.player.energy;
    sts2_core::damage::apply_damage(
        &mut s,
        hooks::CreatureRef::Player,
        e,
        500,
        hooks::ValueProps::MOVE,
        None,
    );
    assert_eq!(s.player.energy, before + 2);
}

/// A Fool Attack that issues no `AttackCommand` (Nail on a target with no
/// arrow in it; a shot with an empty quiver) never reaches
/// `VigorPower.BeforeAttack`, so Vigor is not spent.
#[test]
fn vigor_survives_an_attack_card_that_never_attacks() {
    let mut s = fool_state();
    add_player_power(&mut s, 94, PowerKind::Vigor, 5);
    play(&mut s, CardId::Nail, false, Some(0));
    assert_eq!(
        player_power(&s, PowerKind::Vigor),
        5,
        "Nail without Impaled"
    );
    for a in &mut fool::fool_mut(&mut s).arrows {
        a.status = fool::ArrowStatus::Away;
    }
    assert!(!fool::player_has_any_arrows(&s));
    play(&mut s, CardId::HeavyShot, false, Some(0));
    assert_eq!(
        player_power(&s, PowerKind::Vigor),
        5,
        "HeavyShot, empty quiver"
    );
    assert_eq!(enemy_hp(&s), 100);
    play(&mut s, CardId::StrikeIronclad, false, Some(0));
    assert_eq!(
        player_power(&s, PowerKind::Vigor),
        0,
        "a real attack spends it"
    );
    assert_eq!(enemy_hp(&s), 100 - 6 - 5);
}

/// `RepeatShot.OnPlay` builds its `AttackCommand` only `if (num > 0)`, so at
/// X == 0 there is no attack and Vigor stays.
#[test]
fn vigor_survives_repeat_shot_at_x_zero() {
    let mut s = fool_state();
    s.player.energy = 0;
    add_player_power(&mut s, 94, PowerKind::Vigor, 5);
    play(&mut s, CardId::RepeatShot, false, Some(0));
    assert_eq!(enemy_hp(&s), 100);
    assert_eq!(player_power(&s, PowerKind::Vigor), 5);
}

/// `HarmonicShield.GainsBlock => true` with a `CalculatedBlockVar`: Stretch
/// latches on it even when its block is calculated, not a base value.
#[test]
fn stretch_latches_on_harmonic_shield() {
    let mut s = fool_state();
    add_player_power(&mut s, 95, PowerKind::Stretch, 6);
    play(&mut s, CardId::HarmonicShield, false, None);
    assert_eq!(s.player.block, 6);
    assert_eq!(player_power(&s, PowerKind::Stretch), 0);
}

/// `WahHammer.ModifyDamageMultiplicative` is an ordinary multiplicative
/// listener: it doubles `base + Strength`, not the base alone.
#[test]
fn wah_hammer_doubles_strength_too() {
    let mut s = fool_state();
    let e = first_enemy(&s);
    let slot = fool::try_shoot_arrow(&mut s, false).expect("an arrow");
    fool::impale(&mut s, e, slot);
    assert!(fool::creature_is_impaled(&s, e));
    add_player_power(&mut s, 96, PowerKind::Strength, 3);
    let hp = enemy_hp(&s);
    play(&mut s, CardId::WahHammer, false, Some(0));
    assert_eq!(hp - enemy_hp(&s), (9 + 3) * 2);
}
