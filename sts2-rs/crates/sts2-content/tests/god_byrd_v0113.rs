//! v0.113.0 God Byrd (`Cards/GodByrd.cs`, `Powers/GodByrdPower.cs`): the card
//! applies `GodByrdPower(Damage.BaseValue)` — 30, 40 upgraded. The power fires
//! in `AfterPlayerTurnStart` only when `AmountOnTurnStart != 0`, so it waits
//! for the NEXT player turn, then deals `Amount` Unpowered to every hittable
//! enemy and removes itself.

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::CardId;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, Action, CardUid, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, PlayerCombatState, PowerKind,
};

fn setup(upgraded: bool) -> CombatState {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let pod = |id, up| sts2_content::cards::spec_index::card_pod_from_id(id, up).expect("pod");
    let mut cards = Vec::new();
    let mut god = pod(CardId::GodByrd, upgraded);
    god.uid = CardUid::new(1).expect("nz");
    cards.push(god);
    let draw: SmallVec<[CardUid; 32]> = (2..10)
        .map(|i| {
            let mut c = pod(CardId::DefendIronclad, false);
            c.uid = CardUid::new(i).expect("nz");
            cards.push(c);
            c.uid
        })
        .collect();
    let enemy = CreatureUid::new(1).expect("nz");
    let creatures = vec![Creature {
        id: enemy.0,
        current_hp: 400,
        max_hp: 400,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::ShrinkerBeetle),
        monster_id: MonsterId(MonsterKind::ShrinkerBeetle.python_name()),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: Some(build_ai(
            MonsterKind::ShrinkerBeetle,
            MonsterSpawnArgs::None,
            SpawnPosition {
                alone: true,
                front: true,
                back: true,
                index: 0,
            },
        )),
    }];
    let player = PlayerCombatState {
        draw: draw.clone(),
        max_energy: 3,
        current_hp: 400,
        max_hp: 400,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards: cards.into(),
        player,
        enemy_uids: SmallVec::from_slice(&[enemy]),
        game_version: GameVersion::V0_113_0,
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.draw = draw;
    state.player.hand.push(CardUid::new(1).expect("nz"));
    state.player.energy = 3;
    state
}

fn enemy_hp(state: &CombatState) -> i32 {
    state.creatures[0].current_hp
}

fn god_byrd_power(state: &CombatState) -> i32 {
    state
        .player_power_uids()
        .into_iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == PowerKind::GodByrd)
        .map(|p| p.amount)
        .sum()
}

#[test]
#[serial_test::serial(global_registry)]
fn god_byrd_hits_every_enemy_at_the_next_turn_start() {
    for (upgraded, dmg) in [(false, 30), (true, 40)] {
        let mut state = setup(upgraded);
        let _ = apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        );
        assert_eq!(
            god_byrd_power(&state),
            dmg,
            "power amount = Damage.BaseValue"
        );
        assert_eq!(enemy_hp(&state), 400, "no damage on the turn it is played");
        let _ = apply(&mut state, Action::EndTurn);
        assert_eq!(
            enemy_hp(&state),
            400 - dmg,
            "next player turn start deals Amount Unpowered"
        );
        assert_eq!(god_byrd_power(&state), 0, "the power removes itself");
    }
}
