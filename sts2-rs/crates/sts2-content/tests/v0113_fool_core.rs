//! v0.113.0 The Fool core mechanics (`sts2_core::fool`): arrows,
//! `ImpaledPower`, the Aim / Quietus attack latch, Yah / Twine /
//! ExplosiveArrows arrow hooks and melody tracking. Expected values are read
//! off the decomp (`Commands/ArrowCmd.cs`, `Models.Powers/*Power.cs`,
//! `Commands/MelodyCmd.cs`), not off the engine.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{CardId, CardTag};
use sts2_core::fool;
use sts2_core::hooks::{self, CreatureRef, TurnSideCtx, ValueProps};
use sts2_core::potions::CharacterId;
use sts2_core::{CardUid, CombatPhase, CombatState, CreatureUid, GameVersion};
use sts2_core::{MonsterKind, PowerKind, Side};

fn fresh_as(version: GameVersion, character: CharacterId) -> (CombatState, CreatureUid) {
    content_dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: version,
        ..Default::default()
    };
    state.player.character = Some(character);
    state.player.current_hp = 77;
    state.player.max_hp = 77;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.round = 1;
    state.side = Side::Player;
    state.phase = CombatPhase::PlayerPlayPhase;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 200;
        c.current_hp = 200;
        c.block = 0;
        c.powers.clear();
    }
    (state, enemy)
}

fn fresh() -> (CombatState, CreatureUid) {
    fresh_as(GameVersion::V0_113_0, CharacterId::Fool)
}

fn hp(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn set_hp(state: &mut CombatState, uid: CreatureUid, v: i32) {
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
        c.current_hp = v;
    }
}

fn give_player(state: &mut CombatState, kind: PowerKind, amount: i32) {
    let p = state.player.creature_uid.expect("player mirror");
    sts2_core::card_effects::apply_power_to_creature(state, p, kind, amount);
}

fn add_card(state: &mut CombatState, card_id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(card_id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    uid
}

/// A plain powered player hit (a Strike), not from an arrow.
fn strike(state: &mut CombatState, target: CreatureUid, dmg: i32) {
    sts2_core::damage::apply_damage(
        state,
        CreatureRef::Player,
        target,
        dmg,
        ValueProps::MOVE,
        None,
    );
}

#[test]
fn fool_starts_with_one_arrow_slot() {
    let (state, _) = fresh();
    assert_eq!(fool::arrow_slot_count(&state), 1);
    assert_eq!(fool::available_arrow_count(&state), 1);
    assert!(fool::player_has_any_arrows(&state));
}

#[test]
fn no_arrows_before_v0113_or_for_other_characters() {
    let (state, _) = fresh_as(GameVersion::V0_111_0, CharacterId::Fool);
    assert_eq!(fool::arrow_slot_count(&state), 0);
    let (state, _) = fresh_as(GameVersion::V0_113_0, CharacterId::Ironclad);
    assert_eq!(fool::arrow_slot_count(&state), 0);
    assert!(state.player.fool.is_none());
}

/// `CardPile.AddInternal`: a Shoot card in a combat pile gives a
/// non-Fool character its first (and only) arrow slot.
#[test]
fn a_shoot_card_gives_other_characters_one_slot() {
    let (mut state, _) = fresh_as(GameVersion::V0_113_0, CharacterId::Ironclad);
    add_card(&mut state, CardId::FoolishShot);
    add_card(&mut state, CardId::QuickShot);
    assert_eq!(fool::arrow_slot_count(&state), 1);
}

/// FoolishShot-style shot: 15 + Aim 6 (SongBow) = 21; Aim is spent by the
/// latched start amount; the target is Impaled (HitCounter 3) holding the
/// arrow; the quiver is empty.
#[test]
fn shot_adds_aim_spends_it_and_impales() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Aim, 6);
    assert!(fool::shoot_arrow_at_target(&mut state, enemy, 15, 1));
    assert_eq!(hp(&state, enemy), 200 - 21);
    assert_eq!(state.player_power_amount(PowerKind::Aim), 0);
    assert!(fool::creature_is_impaled(&state, enemy));
    assert_eq!(fool::impaled_hit_counter(&state, enemy), Some(3));
    assert_eq!(fool::arrows_in(&state, enemy), 1);
    assert_eq!(fool::available_arrow_count(&state), 0);
}

/// `AimPower.ModifyDamageAdditive`: while an attack command is latched,
/// `FromArrow` damage from a DIFFERENT card source gets no Aim
/// (`commandToModify != null && cardSource != null && cardSource !=
/// commandToModify.ModelSource` -> 0); the latched card's own damage does.
/// HeavyShot's doubling (`cardSource is HeavyShot`) follows the damage's own
/// card, not the latched one.
#[test]
fn aim_skips_arrow_damage_from_another_card_while_latched() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Aim, 5);
    let a = add_card(&mut state, CardId::FoolishShot);
    let b = add_card(&mut state, CardId::HeavyShot);
    let arrow = ValueProps::MOVE.union(ValueProps::FROM_ARROW);
    state.resolving_card = Some(a);
    fool::begin_arrow_attack(&mut state);
    // The latched card's own arrow hit: +5.
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        10,
        arrow,
        Some(CardId::FoolishShot),
    );
    assert_eq!(hp(&state, enemy), 200 - 15);
    // Another card's arrow hit inside the latched command: no Aim (and so
    // no HeavyShot doubling either).
    state.resolving_card = Some(b);
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        10,
        arrow,
        Some(CardId::HeavyShot),
    );
    assert_eq!(hp(&state, enemy), 200 - 25);
    state.resolving_card = Some(a);
    fool::end_arrow_attack(&mut state);
    state.resolving_card = None;
    assert_eq!(state.player_power_amount(PowerKind::Aim), 0);

    // HeavyShot latching its own command doubles: 10 + 2 x 4.
    give_player(&mut state, PowerKind::Aim, 4);
    state.resolving_card = Some(b);
    fool::begin_arrow_attack(&mut state);
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        10,
        arrow,
        Some(CardId::HeavyShot),
    );
    fool::end_arrow_attack(&mut state);
    state.resolving_card = None;
    assert_eq!(hp(&state, enemy), 200 - 25 - 18);
}

/// With no arrow left, `ShootArrowAtTarget` does not attack at all.
#[test]
fn no_arrow_no_attack() {
    let (mut state, enemy) = fresh();
    assert!(fool::shoot_arrow_at_target(&mut state, enemy, 15, 1));
    let after_first = hp(&state, enemy);
    assert!(!fool::shoot_arrow_at_target(&mut state, enemy, 15, 1));
    assert_eq!(hp(&state, enemy), after_first);
}

/// Aim only boosts FromArrow damage.
#[test]
fn aim_ignores_plain_attacks() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Aim, 6);
    strike(&mut state, enemy, 6);
    assert_eq!(hp(&state, enemy), 194);
    assert_eq!(state.player_power_amount(PowerKind::Aim), 6);
}

/// `ImpaledPower.AfterDamageReceived`: three powered player hits dislodge
/// the arrow (retrieved) and remove the emptied power.
#[test]
fn three_powered_hits_dislodge_the_arrow() {
    let (mut state, enemy) = fresh();
    fool::shoot_arrow_at_target(&mut state, enemy, 1, 1);
    // The shot itself was a powered player hit on a not-yet-impaled target.
    assert_eq!(fool::impaled_hit_counter(&state, enemy), Some(3));
    strike(&mut state, enemy, 1);
    strike(&mut state, enemy, 1);
    assert_eq!(fool::impaled_hit_counter(&state, enemy), Some(1));
    assert_eq!(fool::available_arrow_count(&state), 0);
    strike(&mut state, enemy, 1);
    assert!(!fool::creature_is_impaled(&state, enemy));
    assert_eq!(fool::available_arrow_count(&state), 1);
    // Unpowered damage never ticks it.
    fool::shoot_arrow_at_target(&mut state, enemy, 1, 1);
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        1,
        ValueProps::UNPOWERED,
        None,
    );
    assert_eq!(fool::impaled_hit_counter(&state, enemy), Some(3));
}

/// A shot that kills its target: the arrow is dislodged straight back.
#[test]
fn killing_shot_returns_the_arrow() {
    let (mut state, enemy) = fresh();
    set_hp(&mut state, enemy, 5);
    assert!(fool::shoot_arrow_at_target(&mut state, enemy, 15, 1));
    assert_eq!(hp(&state, enemy), 0);
    assert_eq!(fool::available_arrow_count(&state), 1);
    assert_eq!(fool::arrows_in(&state, enemy), 0);
}

/// `ImpaledPower.AfterDeath`: an impaled enemy killed by a plain hit gives
/// its arrows back.
#[test]
fn impaled_enemy_death_returns_arrows() {
    let (mut state, enemy) = fresh();
    fool::shoot_arrow_at_target(&mut state, enemy, 1, 1);
    set_hp(&mut state, enemy, 3);
    strike(&mut state, enemy, 10);
    assert_eq!(hp(&state, enemy), 0);
    assert_eq!(fool::available_arrow_count(&state), 1);
}

/// QuietusPower: xAmount on the arrow attack, the arrow is destroyed after
/// impaling, and Quietus removes itself after the attack.
#[test]
fn quietus_multiplies_and_destroys_the_arrow() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Quietus, 3);
    // A card source is required for the destroy-after-impaled mark.
    let card = add_card(&mut state, CardId::FoolishShot);
    state.resolving_card = Some(card);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    state.resolving_card = None;
    assert_eq!(hp(&state, enemy), 170);
    assert_eq!(state.player_power_amount(PowerKind::Quietus), 0);
    assert_eq!(fool::available_arrow_count(&state), 0);
    assert_eq!(fool::arrows_in(&state, enemy), 0);
    assert!(!fool::creature_is_impaled(&state, enemy));
    assert_eq!(
        fool::fool(&state).unwrap().arrows[0].status,
        fool::ArrowStatus::Destroyed
    );
}

/// `ArrowCmd.ImpaleTarget` + `DestroyArrow`: a Quietus arrow that KILLS its
/// target is destroyed without ever sitting in it — C# adds it to an
/// `ImpaledPower` that `PowerCmd.Apply` cannot attach to the dead creature
/// (`CanReceivePowers`), and `DestroyArrow` skips `ForceRemoveArrow` on a
/// dead target. No arrow may stay recorded on the corpse (a reviver would
/// otherwise come back "impaled").
#[test]
fn quietus_arrow_that_kills_leaves_nothing_in_the_corpse() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Quietus, 3);
    set_hp(&mut state, enemy, 5);
    let card = add_card(&mut state, CardId::FoolishShot);
    state.resolving_card = Some(card);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    state.resolving_card = None;
    assert_eq!(hp(&state, enemy), 0);
    assert_eq!(fool::arrows_in(&state, enemy), 0);
    assert!(fool::fool(&state).unwrap().impaled.is_empty());
    assert_eq!(
        fool::fool(&state).unwrap().arrows[0].status,
        fool::ArrowStatus::Destroyed
    );
}

/// Honing: x(1 + A/100) on arrow attacks only.
#[test]
fn honing_boosts_arrow_damage() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Honing, 50);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    assert_eq!(hp(&state, enemy), 185);
}

/// SerratedArrows: x(1 + A/100) on powered hits against an impaled target.
#[test]
fn serrated_arrows_boost_hits_on_impaled_targets() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::SerratedArrows, 50);
    strike(&mut state, enemy, 10);
    assert_eq!(hp(&state, enemy), 190);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    assert_eq!(hp(&state, enemy), 180); // not impaled while the shot lands
    strike(&mut state, enemy, 10);
    assert_eq!(hp(&state, enemy), 165);
}

/// Yah: energy on the player's arrow dislodge, once per turn (a new player
/// round re-arms it).
#[test]
fn yah_grants_energy_once_per_turn() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::Yah, 2);
    let e0 = state.player.energy;
    for _ in 0..2 {
        fool::shoot_arrow_at_target(&mut state, enemy, 1, 1);
        fool::dislodge_arrow(&mut state, enemy);
    }
    assert_eq!(state.player.energy, e0 + 2);
    state.round = 2;
    fool::shoot_arrow_at_target(&mut state, enemy, 1, 1);
    fool::dislodge_arrow(&mut state, enemy);
    assert_eq!(state.player.energy, e0 + 4);
}

/// ExplosiveArrows: Amount Unpowered damage to every enemy per impale.
#[test]
fn explosive_arrows_hit_every_enemy_on_impale() {
    let (mut state, enemy) = fresh();
    give_player(&mut state, PowerKind::ExplosiveArrows, 4);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    assert_eq!(hp(&state, enemy), 200 - 10 - 4);
}

/// DislikeYou on the target: x(1 + A/100) on arrow damage; removed at the
/// player's side turn end.
#[test]
fn dislike_you_boosts_arrow_damage_and_expires() {
    let (mut state, enemy) = fresh();
    sts2_core::card_effects::apply_power_to_creature(&mut state, enemy, PowerKind::DislikeYou, 100);
    fool::shoot_arrow_at_target(&mut state, enemy, 10, 1);
    assert_eq!(hp(&state, enemy), 180);
    hooks::dispatch_after_turn_end(&mut state, &TurnSideCtx { side: Side::Player });
    let has = state
        .creatures
        .iter()
        .find(|c| c.id == enemy.0)
        .unwrap()
        .powers
        .iter()
        .any(|u| {
            state
                .powers
                .get_by_id(u.0)
                .is_some_and(|p| p.kind == PowerKind::DislikeYou)
        });
    assert!(!has);
}

/// Melody: finished plays per note this turn, the first-started-play test,
/// and the reset on a new round.
#[test]
fn melody_tracking_follows_the_history() {
    let (mut state, _) = fresh();
    let wah = add_card(&mut state, CardId::WahBlast);
    let ki = add_card(&mut state, CardId::KiHop);
    assert!(state.cards[wah.get() as usize - 1]
        .tags
        .contains(CardTag::Wah));
    fool::note_play_started(&mut state, wah);
    assert!(fool::is_first_play_this_turn(&state, CardTag::Wah));
    assert!(!fool::played_this_turn(&state, CardTag::Wah));
    fool::note_play_finished(&mut state, wah);
    assert!(fool::played_this_turn(&state, CardTag::Wah));
    // A second Wah play is not the first.
    fool::note_play_started(&mut state, wah);
    assert!(!fool::is_first_play_this_turn(&state, CardTag::Wah));
    fool::note_play_finished(&mut state, wah);
    fool::note_play_started(&mut state, ki);
    fool::note_play_finished(&mut state, ki);
    assert_eq!(fool::melodies_this_turn(&state), 2);
    assert_eq!(fool::melody_cards_played_this_turn(&state), 3);
    assert_eq!(fool::started_this_turn(&state, CardTag::Wah), 2);
    state.round = 2;
    assert_eq!(fool::melodies_this_turn(&state), 0);
    assert!(!fool::played_this_turn(&state, CardTag::Wah));
}

// ── Fool relics (`sts2_content::relics::fool`) ─────────────────────────

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: std::num::NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn aim(state: &CombatState) -> i32 {
    fool::player_power_amount(state, PowerKind::Aim)
}

fn hand_ids(state: &CombatState) -> Vec<CardId> {
    state
        .player
        .hand
        .iter()
        .map(|u| state.cards[u.get() as usize - 1].card_id)
        .collect()
}

/// `SongBow` / `Storyteller` `AfterRoomEntered`: 6 / 12 Aim; Storyteller
/// adds 3 more at the owner's `BeforeSideTurnEnd` only.
#[test]
fn song_bow_and_storyteller_grant_aim() {
    use sts2_core::relic_effects::{dispatch_before_combat_start, dispatch_before_turn_end};
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::SongBow);
    let _ = dispatch_before_combat_start(&mut state);
    assert_eq!(aim(&state), 6);

    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::Storyteller);
    let _ = dispatch_before_combat_start(&mut state);
    assert_eq!(aim(&state), 12);
    let _ = dispatch_before_turn_end(&mut state, Side::Enemy);
    assert_eq!(aim(&state), 12);
    let _ = dispatch_before_turn_end(&mut state, Side::Player);
    assert_eq!(aim(&state), 15);
}

/// `ExerciseMat.AfterSideTurnStart`: Stretch 6 on turn 1 only.
#[test]
fn exercise_mat_stretches_on_turn_one_only() {
    use sts2_core::relic_effects::dispatch_after_side_turn_start;
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::ExerciseMat);
    let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
    assert_eq!(fool::player_power_amount(&state, PowerKind::Stretch), 6);
    state.round = 2;
    let _ = dispatch_after_side_turn_start(&mut state, Side::Player);
    assert_eq!(fool::player_power_amount(&state, PowerKind::Stretch), 6);
}

/// `WhittlingKnife.AfterShuffle`: 5 Aim per shuffle.
#[test]
fn whittling_knife_aims_on_shuffle() {
    use sts2_core::relic_effects::{dispatch_after_shuffle_bus, ShuffleCtx};
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::WhittlingKnife);
    let ctx = ShuffleCtx {
        from_reshuffle: true,
    };
    let _ = dispatch_after_shuffle_bus(&mut state, &ctx);
    let _ = dispatch_after_shuffle_bus(&mut state, &ctx);
    assert_eq!(aim(&state), 10);
}

/// `ConductorsBaton.AfterCardPlayed`: once three distinct melodies were
/// played this turn, ConductorsBatonPower(1), once per turn.
#[test]
fn conductors_baton_fires_once_after_three_melodies() {
    use sts2_core::relic_effects::{
        dispatch_after_card_played, dispatch_before_side_turn_start, CardEventCtx,
    };
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::ConductorsBaton);
    let wah = add_card(&mut state, CardId::WahBlast);
    let ki = add_card(&mut state, CardId::KiHop);
    let wah_cho = add_card(&mut state, CardId::WahCho);
    let play = |state: &mut CombatState, uid: CardUid| {
        fool::note_play_started(state, uid);
        fool::note_play_finished(state, uid);
        let card = &state.cards[uid.get() as usize - 1];
        let ctx = CardEventCtx {
            card_id: card.card_id,
            card_uid: Some(uid),
            target: None,
            card_tags: card.tags,
            energy_spent: 0,
        };
        let _ = dispatch_after_card_played(state, &ctx);
    };
    play(&mut state, wah);
    play(&mut state, ki);
    assert_eq!(
        fool::player_power_amount(&state, PowerKind::ConductorsBaton),
        0
    );
    play(&mut state, wah_cho);
    assert_eq!(
        fool::player_power_amount(&state, PowerKind::ConductorsBaton),
        1
    );
    // Still three melodies, but it already fired this turn.
    play(&mut state, ki);
    assert_eq!(
        fool::player_power_amount(&state, PowerKind::ConductorsBaton),
        1
    );
    let relic = state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::ConductorsBaton)
        .expect("baton");
    assert_eq!(relic.counter, 1);
    let _ = dispatch_before_side_turn_start(&mut state, Side::Player);
    let relic = state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::ConductorsBaton)
        .expect("baton");
    assert_eq!(relic.counter, 0);
}

/// `BizarreLamp.BeforeHandDraw` (turn 1): Joy then Sadness into hand.
#[test]
fn bizarre_lamp_adds_joy_and_sadness_on_turn_one() {
    use sts2_core::relic_effects::dispatch_before_hand_draw;
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::BizarreLamp);
    let _ = dispatch_before_hand_draw(&mut state);
    assert_eq!(hand_ids(&state), vec![CardId::Joy, CardId::Sadness]);
    state.round = 2;
    let _ = dispatch_before_hand_draw(&mut state);
    assert_eq!(hand_ids(&state).len(), 2);
}

/// `Fertilizer`: a Sadness the owner generates gives 1 Strength.
#[test]
fn fertilizer_turns_generated_sadness_into_strength() {
    use sts2_core::relic_effects::dispatch_before_hand_draw;
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::BizarreLamp);
    add_relic(&mut state, sts2_core::RelicKind::Fertilizer);
    let _ = dispatch_before_hand_draw(&mut state);
    assert_eq!(fool::player_power_amount(&state, PowerKind::Strength), 1);
}

/// `Ocarina.BeforeHandDraw` (turn 1): one Wah/Cho/Ki card into hand.
#[test]
fn ocarina_adds_one_melody_card() {
    use sts2_core::relic_effects::dispatch_before_hand_draw;
    let pool = sts2_content::cards::fool_pool::combat_melody_pool(None);
    assert!(!pool.is_empty());
    for id in &pool {
        assert!(
            sts2_content::cards::status::helpers::tags_for_card_id(*id).has_melody(),
            "{id:?}"
        );
    }
    let (mut state, _) = fresh();
    add_relic(&mut state, sts2_core::RelicKind::Ocarina);
    let _ = dispatch_before_hand_draw(&mut state);
    let hand = hand_ids(&state);
    assert_eq!(hand.len(), 1);
    assert!(pool.contains(&hand[0]));
}

// ── Fool potions ───────────────────────────────────────────────────────

fn drink(state: &mut CombatState, kind: sts2_core::PotionKind) {
    sts2_content::potions::dispatch::ensure_installed();
    let _ = sts2_core::potion_effects::dispatch(state, kind, None);
}

#[test]
fn precision_potion_grants_21_aim() {
    let (mut state, _) = fresh();
    drink(&mut state, sts2_core::PotionKind::PrecisionPotion);
    assert_eq!(aim(&state), 21);
}

/// `BottledSong`: one Wah, one Cho and one Ki card, each free this turn.
#[test]
fn bottled_song_adds_one_card_per_melody_free() {
    let (mut state, _) = fresh();
    drink(&mut state, sts2_core::PotionKind::BottledSong);
    let hand = state.player.hand.clone();
    assert_eq!(hand.len(), 3);
    for (uid, tag) in hand.iter().zip(CardTag::MELODY) {
        let card = &state.cards[uid.get() as usize - 1];
        assert!(card.tags.contains(tag), "{:?} lacks {tag:?}", card.card_id);
        assert_eq!(card.local_cost(), 0, "{:?} not free", card.card_id);
    }
}

/// `CrystallineEuphoria`: 3 Joy, then Sadness up to a full hand.
#[test]
fn crystalline_euphoria_fills_the_hand() {
    let (mut state, _) = fresh();
    add_card(&mut state, CardId::StrikeFool);
    add_card(&mut state, CardId::StrikeFool);
    drink(&mut state, sts2_core::PotionKind::CrystallineEuphoria);
    let hand = hand_ids(&state);
    assert_eq!(hand.len(), 10);
    assert_eq!(&hand[2..5], &[CardId::Joy; 3]);
    assert!(hand[5..].iter().all(|c| *c == CardId::Sadness));
}
