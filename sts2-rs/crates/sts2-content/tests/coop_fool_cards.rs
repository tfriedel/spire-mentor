//! The Fool's multiplayer-only cards in a co-op fight (GH #1337):
//! `SelfDeprecate.cs` and `CompostPower.cs` act on the teammates, whose
//! hands and energy are simulated.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, CreatureUid, MonsterKind, PlayerCombatState,
    PowerKind,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn mint(state: &mut CombatState, id: CardId, upgraded: bool) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, upgraded).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Us and one teammate seat in a v0.113.0 fight with two Nibbits, our play
/// phase open, 5 energy each.
fn fight() -> (CombatState, CreatureUid) {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: sts2_core::GameVersion::V0_113_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.init_player_creature();
    state.allies.push(PlayerCombatState {
        current_hp: 70,
        max_hp: 70,
        energy: 5,
        ..PlayerCombatState::default()
    });
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    (state, enemy)
}

fn play(state: &mut CombatState, id: CardId, upgraded: bool) {
    let uid = mint(state, id, upgraded);
    state.player.hand.push(uid);
    let hand_index = u8::try_from(state.player.hand.len() - 1).expect("fits");
    let _ = sts2_core::apply(
        state,
        Action::PlayCard {
            hand_index,
            target: None,
        },
    );
    assert!(!state.player.hand.contains(&uid), "{id:?} was played");
}

fn ids(state: &CombatState, hand: &[CardUid]) -> Vec<CardId> {
    hand.iter()
        .map(|u| state.cards[u.get() as usize - 1].card_id)
        .collect()
}

/// `SelfDeprecate.cs`: one Sadness to the owner's hand, then 2 (+1) Joy to
/// each other living player's hand.
#[test]
fn self_deprecate_gives_each_teammate_joy() {
    for (upgraded, joy) in [(false, 2), (true, 3)] {
        let (mut state, _) = fight();
        play(&mut state, CardId::SelfDeprecate, upgraded);
        assert_eq!(ids(&state, &state.player.hand), vec![CardId::Sadness]);
        assert_eq!(
            ids(&state, &state.allies[0].hand),
            vec![CardId::Joy; joy],
            "upgraded={upgraded}"
        );
    }
}

/// A dead teammate gets nothing (`c.IsAlive`).
#[test]
fn self_deprecate_skips_a_dead_teammate() {
    let (mut state, _) = fight();
    state.allies[0].current_hp = 0;
    play(&mut state, CardId::SelfDeprecate, false);
    assert!(state.allies[0].hand.is_empty());
}

/// `CompostPower.AfterDeath`: an enemy's death gives Amount energy to every
/// living player on the owner's side (`GetTeammatesOf(Owner)` includes the
/// owner).
#[test]
fn compost_gives_every_player_energy_on_an_enemy_death() {
    let (mut state, enemy) = fight();
    play(&mut state, CardId::CompostCard, true);
    assert_eq!(state.player_power_amount(PowerKind::Compost), 2);
    let (ours, mates) = (state.player.energy, state.allies[0].energy);
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        500,
        ValueProps::MOVE,
        None,
    );
    assert!(!state.is_over, "the second Nibbit lives");
    assert_eq!(state.player.energy, ours + 2);
    assert_eq!(state.allies[0].energy, mates + 2);
}

/// The same from a teammate's Compost: the hook is the power owner's, the
/// energy still goes to every living player.
#[test]
fn a_teammates_compost_gives_every_player_energy() {
    let (mut state, enemy) = fight();
    sts2_core::coop::with_ally_seated(&mut state, 0, |state| {
        let uid = state.player.creature_uid.expect("the teammate's creature");
        sts2_core::card_effects::apply_power_to_creature(state, uid, PowerKind::Compost, 1);
    });
    let (ours, mates) = (state.player.energy, state.allies[0].energy);
    sts2_core::damage::apply_damage(
        &mut state,
        CreatureRef::Player,
        enemy,
        500,
        ValueProps::MOVE,
        None,
    );
    assert_eq!(state.player.energy, ours + 1);
    assert_eq!(state.allies[0].energy, mates + 1);
}
