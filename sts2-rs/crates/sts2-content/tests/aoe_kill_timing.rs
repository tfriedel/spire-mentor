//! Kill timing of multi-target damage.
//!
//! C# `CreatureCmd.Damage(targets, ...)` (CreatureCmd.cs) takes every
//! target's HP first, runs each result's after-hit hooks in hit order, and
//! only then calls `Kill(killedCreatures)`, which kills them one by one in hit
//! order and checks the win/loss condition once at the end.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::{
    Action, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, StepResult,
};

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion, player_hp: i32) -> CombatState {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = player_hp;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.player.potion_slots = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn add_enemy(state: &mut CombatState, hp: i32) -> CreatureUid {
    let raw = u32::try_from(state.creatures.len() + 1).expect("u32");
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: 40,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    let uid = CreatureUid(id);
    state.enemy_uids.push(uid);
    uid
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn drink_first_potion(state: &mut CombatState) -> StepResult {
    sts2_core::apply(
        state,
        Action::DrinkPotion {
            potion_slot: 0,
            target: None,
        },
    )
}

/// Foul Potion hits the player too: `CreatureCmd.Damage(Creatures.Where(!IsPet))`
/// (FoulPotion.cs), one call whose targets list the player before the enemies
/// (`CombatState.Creatures` is allies then enemies). When it brings both the
/// player and the last enemy to 0 HP, `Kill(killedCreatures)` kills the
/// player first and every player is dead: the combat is lost.
#[test]
fn foul_potion_that_fells_the_player_and_the_last_enemy_loses() {
    for v in BOTH {
        let mut state = fresh(v, 10);
        add_enemy(&mut state, 5);
        assert!(state
            .player
            .add_potion_first_free_slot(PotionKind::FoulPotion));

        let result = drink_first_potion(&mut state);

        assert_eq!(state.player.current_hp, 0, "{v:?}: the player took 12");
        assert_eq!(result, StepResult::CombatLost, "{v:?}");
    }
}

/// The player's after-hit hooks run before any enemy the potion felled dies.
/// In the one Foul Potion call the player's result comes first, so Centennial
/// Puzzle's draw (CentennialPuzzle.cs `AfterDamageReceived`) pulls the Void
/// at 0 energy, which loses nothing, and only then does `Kill` run Gremlin
/// Horn's `AfterDeath` (+1 energy, draw 1). Killing the enemy at its hit gave
/// the energy first and the Void then took it back.
#[test]
fn foul_potion_player_hooks_run_before_the_felled_enemy_dies() {
    for v in BOTH {
        let mut state = fresh(v, 60);
        state.player.energy = 0;
        add_enemy(&mut state, 5);
        add_enemy(&mut state, 40);
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let void = new_card(&mut state, CardId::Void);
        state.player.draw.push(void);
        for _ in 0..4 {
            let strike = new_card(&mut state, CardId::StrikeIronclad);
            state.player.draw.push(strike);
        }
        assert!(state
            .player
            .add_potion_first_free_slot(PotionKind::FoulPotion));

        let result = drink_first_potion(&mut state);

        assert_eq!(result, StepResult::Advanced, "{v:?}");
        assert_eq!(state.player.hand.len(), 4, "{v:?}: drew 3 + 1");
        assert!(
            state.player.hand.contains(&void),
            "{v:?}: the Void was drawn"
        );
        assert_eq!(
            state.player.energy, 1,
            "{v:?}: the Void drew at 0 energy, then Gremlin Horn gave 1"
        );
    }
}

/// When the potion fells the last enemy, the combat is already ending by the
/// time the player's hooks run (`IsCombatEnding` reads `IsAlive`, i.e. HP), so
/// Centennial Puzzle draws nothing: no card leaves the draw pile and an empty
/// one is not reshuffled from the discard.
#[test]
fn foul_potion_that_fells_the_last_enemy_draws_nothing() {
    for v in BOTH {
        let mut state = fresh(v, 60);
        add_enemy(&mut state, 5);
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        for _ in 0..3 {
            let strike = new_card(&mut state, CardId::StrikeIronclad);
            state.player.discard.push(strike);
        }
        let shuffles = state.run_rng.shuffle.counter();
        assert!(state
            .player
            .add_potion_first_free_slot(PotionKind::FoulPotion));

        let result = drink_first_potion(&mut state);

        assert_eq!(result, StepResult::CombatWon, "{v:?}");
        assert_eq!(state.player.current_hp, 48, "{v:?}: the player took 12");
        assert!(state.player.hand.is_empty(), "{v:?}: nothing drawn");
        assert_eq!(
            state.run_rng.shuffle.counter(),
            shuffles,
            "{v:?}: no reshuffle"
        );
    }
}
