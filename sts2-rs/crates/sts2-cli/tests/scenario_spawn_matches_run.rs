//! GH #762: the scenario builder spawns a fight's monsters exactly as a real
//! run does.
//!
//! The builder used to give every monster slot 0's AI and drop the
//! encounter's spawn args, roll HP off `Rng::new(seed)` with no uniqueness,
//! and apply on-spawn powers before the player creature existed, so a
//! monster's spawn-time debuff on the player was lost. Both paths now go
//! through `run_bridge::spawn_roster`; these tests pin the builder's start
//! state to the run path's.

use sts2_cli::sim::scenario::starter::build_state_with_ascension;
use sts2_cli::sim::scenario::{MonsterAiArgs, MonsterSpec, Scenario};
use sts2_cli::sim::{build_starter_state_from_monsters, PlayerClass};
use sts2_content::encounters::EncounterDef;
use sts2_content::encounters::{act1_overgrowth, act1_underdocks, act2_hive, act3_glory};
use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_content::run_bridge::enter_combat_from_roster_in_slots;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::monsters::roll_unique_max_hps;
use sts2_core::parity::combat_state_to_bridge_state;
use sts2_core::powers::PowerKind;
use sts2_core::run_rng_set::RunRngSet;
use sts2_core::runstate::{RoomType, RunState};
use sts2_core::{CombatState, MonsterKind};

/// Each enemy's opening move id, in roster order.
fn openers(state: &CombatState) -> Vec<&'static str> {
    state
        .enemy_uids
        .iter()
        .map(|uid| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .and_then(|c| c.ai.as_ref())
                .map_or("?", |ai| ai.current_move().id)
        })
        .collect()
}

/// The player's powers as `(kind, amount)`, sorted.
fn player_powers(state: &CombatState) -> Vec<(PowerKind, i32)> {
    let mut v: Vec<(PowerKind, i32)> = state
        .player_power_uids()
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .map(|p| (p.kind, p.amount))
        .collect();
    v.sort_by_key(|&(k, a)| (format!("{k:?}"), a));
    v
}

/// Two Toadpoles, as `ToadpolesWeak.GenerateMonsters` hands them: the first is
/// the front one and opens SPIKEN, the second opens WHIRL (`Toadpole.cs:79-80`,
/// `ToadpolesWeak.cs:17-20`). Before the fix both opened SPIKEN.
#[test]
fn two_toadpoles_open_spiken_then_whirl() {
    sts2_content::cards::dispatch::ensure_installed();
    let mut state = build_starter_state_from_monsters(
        7,
        vec![
            MonsterSpec::new(MonsterKind::Toadpole),
            MonsterSpec::new(MonsterKind::Toadpole),
        ],
        PlayerClass::Ironclad,
        None,
        None,
        None,
    );
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);
    assert_eq!(openers(&state), ["SPIKEN_MOVE", "WHIRL_MOVE"]);
}

/// Rocket debuffs its opponents on spawn (`Rocket.cs:76`, Surrounded 1). The
/// builder used to apply spawn powers before the player creature existed, so
/// the debuff never landed.
#[test]
fn rocket_surrounds_the_player_at_spawn() {
    sts2_content::cards::dispatch::ensure_installed();
    let state = build_starter_state_from_monsters(
        7,
        vec![
            MonsterSpec::new(MonsterKind::Crusher),
            MonsterSpec::new(MonsterKind::Rocket),
        ],
        PlayerClass::Ironclad,
        None,
        None,
        None,
    );
    assert_eq!(state.player_power_amount(PowerKind::Surrounded), 1);
}

/// Every encounter of every act, several seeds and ascensions: the builder's
/// monsters equal the run path's (`enter_combat_from_roster` with HP rolled by
/// `roll_unique_max_hps` off `niche`, as `run_callbacks::build_combat_for_room`
/// does) in kind, HP, block, powers and opening move, the player's powers
/// (spawn debuffs) match, and the `niche`, `monster_ai` and legacy combat RNG
/// streams sit at the same cursors.
#[test]
fn builder_monsters_equal_run_path_monsters() {
    sts2_content::cards::dispatch::ensure_installed();
    let tables: [&[EncounterDef]; 4] = [
        act1_overgrowth::ACT1_ENCOUNTERS,
        act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS,
        act2_hive::ACT2_ENCOUNTERS,
        act3_glory::ACT3_ENCOUNTERS,
    ];
    let mut checked = 0;
    let mut surrounded = 0;
    for enc in tables.into_iter().flatten() {
        for seed in 1..=6_u64 {
            for ascension in [0_u8, 10] {
                let mut enc_rng = sts2_core::dotnet_rng::Rng::new(seed as i32);
                let roster = enc.roster(&mut enc_rng);
                let slots = enc.member_slots();
                let specs: Vec<MonsterSpec> = roster
                    .iter()
                    .enumerate()
                    .map(|(i, &(kind, args))| MonsterSpec {
                        kind,
                        hp_override: None,
                        ai_args: MonsterAiArgs::default(),
                        args,
                        slot: slots.get(i).copied(),
                    })
                    .collect();

                let mut built = build_state_with_ascension(
                    &Scenario::from_monsters(specs),
                    seed,
                    PlayerClass::Ironclad,
                    None,
                    None,
                    ascension,
                );
                let _ = start_combat(&mut built);
                let _ = start_player_turn(&mut built);

                let mut rs = RunState {
                    current_hp: 80,
                    max_hp: 80,
                    ascension_level: ascension,
                    rng: RunRngSet::new(seed as i32),
                    ..RunState::default()
                };
                let kinds: Vec<MonsterKind> = roster.iter().map(|&(k, _)| k).collect();
                let hps = roll_unique_max_hps(&kinds, &mut rs.rng.niche, ascension);
                let run_roster: Vec<(MonsterKind, MonsterSpawnArgs, i32)> = roster
                    .iter()
                    .zip(hps)
                    .map(|(&(k, a), hp)| (k, a, hp))
                    .collect();
                let run =
                    enter_combat_from_roster_in_slots(&rs, &run_roster, slots, RoomType::Monster);

                let ctx = format!("{} seed {seed} A{ascension}", enc.id);
                let b = combat_state_to_bridge_state(&built);
                let r = combat_state_to_bridge_state(&run);
                assert_eq!(b["enemies"], r["enemies"], "{ctx}: enemies");
                // Through the mirror creature: the bridge's `player.powers`
                // reads only the legacy list, where spawn debuffs never land.
                assert_eq!(
                    player_powers(&built),
                    player_powers(&run),
                    "{ctx}: player powers"
                );
                assert_eq!(openers(&built), openers(&run), "{ctx}: opening moves");
                let slots_of = |s: &CombatState| -> Vec<Option<u8>> {
                    s.enemy_uids.iter().map(|u| s.enemy_slot(*u)).collect()
                };
                assert_eq!(slots_of(&built), slots_of(&run), "{ctx}: slots");
                // The shuffle stream is left out: the run here has no deck.
                assert_eq!(
                    built.run_rng.monster_ai.counter(),
                    run.run_rng.monster_ai.counter(),
                    "{ctx}: monster_ai"
                );
                assert_eq!(
                    built.run_rng.niche.counter(),
                    run.run_rng.niche.counter(),
                    "{ctx}: niche"
                );
                assert_eq!(
                    (built.rng.seed(), built.rng.counter()),
                    (run.rng.seed(), run.rng.counter()),
                    "{ctx}: legacy combat rng"
                );
                checked += 1;
                surrounded += usize::from(built.player_power_amount(PowerKind::Surrounded) > 0);
            }
        }
    }
    assert!(checked > 500, "checked only {checked} start states");
    // KAISER_CRAB_BOSS (Crusher + Rocket) must reach the spawn-debuff path.
    assert!(surrounded > 0, "no start state carried Rocket's Surrounded");
}
