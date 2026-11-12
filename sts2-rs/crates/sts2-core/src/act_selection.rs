//! Seed-driven act-variant selection — the port of C# `ActModel.GetRandomList`
//! (`MegaCrit.Sts2.Core.Models/ActModel.cs:558`, v0.107.1).
//!
//! # Why this module exists
//!
//! STS2 ships **two** Act-1 models, both at `Index => 0`:
//!
//! | Act model     | Index | `IsDefault` | Boss pool                                     |
//! |---------------|-------|-------------|-----------------------------------------------|
//! | `Overgrowth`  | 0     | `true`      | Vantom, Ceremonial Beast, The Kin             |
//! | `Underdocks`  | 0     | `false`     | Waterfall Giant, Soul Fysh, Lagavulin Matriarch |
//! | `Hive`        | 1     | `true`      | The Insatiable, Knowledge Demon, Kaiser Crab  |
//! | `Glory`       | 2     | `true`      | Queen, Test Subject, Aeonglass                |
//!
//! Which Act 1 a run gets is rolled at run start from a **dedicated named
//! stream** — `new Rng((uint)GetDeterministicHashCode(seed), "act_selection")`
//! in `StartRunLobby.BeginRunLocally` (`StartRunLobby.cs:459`) — and *not*
//! from `up_front`. The roll lives in the lobby, so any harness that builds a
//! run through `RunState.CreateForTest` / `RunManager` never reaches it and
//! silently gets `ActModel.GetDefaultList()` — i.e. Overgrowth, always.
//!
//! That blind spot is exactly how GH #544 went unnoticed: the
//! headless C# oracle omits the `acts:` argument, so every act-1 parity number
//! this repo has ever produced compared Overgrowth against Overgrowth.
//!
//! # Where the answer goes
//!
//! Underdocks content is ported (`run_room_gen`'s `ACT1_UNDERDOCKS` and
//! `sts2-content::encounters::act1_underdocks`), so the variant this module
//! picks is a real engine input: [`crate::run_manager::RunManager::with_act1`]
//! stores it on [`crate::runstate::RunState::act1`], and
//! `run_room_gen::generate_all_rooms` builds act 1 from it.
//!
//! **The run-facing defaults roll; the engine core does not.** The PyO3
//! `RunManager` (`act1="roll"`), the arena (`Act1Choice::Roll`), `PpoRunEnv`
//! and the C# host all default to the seed-driven pick, as the real game
//! does with everything unlocked and discovered. Win rates measured before
//! that change were Overgrowth-only. Inside the engine,
//! `RunState::act1` still starts as [`ActVariant::Overgrowth`]: a bare
//! [`crate::run_manager::RunManager::new`] is a building block, and the
//! callers above apply the roll with
//! [`crate::run_manager::RunManager::with_rolled_act1`].
//!
//! # Save-state dependence (the reason a seed alone is not enough)
//!
//! `GetRandomList` is not a pure function of the seed. Two escape hatches
//! bypass or precede the RNG:
//!
//! 1. **First-discovery force.** An unlocked non-default act the local save
//!    has never *discovered* is taken unconditionally, `break`ing before the
//!    `NextItem` call — so that act index consumes **zero** reads.
//! 2. **Lobby override.** `list[0] = GetAct(Act1) ?? list[0]`
//!    (`StartRunLobby.cs:460`) lets an explicit `"overgrowth"` / `"underdocks"`
//!    lobby key replace the rolled act outright, after the stream has advanced.
//!
//! Both are guarded by `TestMode.IsOff`, and the headless oracle sets
//! `TestMode.IsOn` — so under the oracle the force branch never fires and the
//! roll *is* purely seed-driven. [`ActSelectionUnlocks::test_mode_on`] models
//! that; [`ActSelectionUnlocks::live_game`] models a real save.

use crate::dotnet_rng::{canonicalize_seed, deterministic_hash_code, Rng};

/// One act model, identified by its C# `ModelId.Entry` slug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActVariant {
    /// Act 1 default (`Overgrowth.cs`) — `IsDefault`, and what a run walks
    /// unless it is told otherwise.
    Overgrowth,
    /// Act 1 alternate (`Underdocks.cs`), gated on `UnderdocksEpoch`.
    Underdocks,
    /// Act 2 (`Hive.cs`) — sole occupant of index 1.
    Hive,
    /// Act 3 (`Glory.cs`) — sole occupant of index 2.
    Glory,
}

impl ActVariant {
    /// The C# `ModelId.Entry` slug (`StringHelper.Slugify(type.Name)`).
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Overgrowth => "OVERGROWTH",
            Self::Underdocks => "UNDERDOCKS",
            Self::Hive => "HIVE",
            Self::Glory => "GLORY",
        }
    }

    /// `ActModel.IsDefault` — whether this act is taken by
    /// `GetDefaultList()` for its index.
    #[must_use]
    pub const fn is_default(self) -> bool {
        !matches!(self, Self::Underdocks)
    }
}

/// `ModelDb.ActsByIndex` — acts grouped by `Index`, in `ModelDb.Acts` order
/// (`ModelDb.cs:264-285`: Overgrowth, Underdocks, Hive, Glory). The order
/// inside a group is load-bearing: it is the `NextItem` pool order.
const ACTS_BY_INDEX: [&[ActVariant]; 3] = [
    &[ActVariant::Overgrowth, ActVariant::Underdocks],
    &[ActVariant::Hive],
    &[ActVariant::Glory],
];

/// The save/session state `GetRandomList` consults alongside the RNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActSelectionUnlocks {
    /// `Underdocks.IsUnlocked` — `UnlockState.IsEpochRevealed<UnderdocksEpoch>()`.
    /// While false the act-1 pool is a singleton and the roll is a formality
    /// (it still consumes one read).
    pub underdocks_unlocked: bool,
    /// `SaveManager.Instance.Progress.DiscoveredActs.Contains(Underdocks)`.
    /// Only consulted when the force branch is live.
    pub underdocks_discovered: bool,
    /// `TestMode.IsOff && !isMultiplayer` — whether the first-discovery force
    /// branch can fire at all. False under the headless oracle and in
    /// multiplayer; true in a real singleplayer session.
    pub force_undiscovered: bool,
}

impl ActSelectionUnlocks {
    /// A headless-oracle session: `TestMode.IsOn`, so the force branch is dead
    /// and the roll is a pure function of the seed. `underdocks_unlocked` is
    /// the only knob that matters.
    #[must_use]
    pub const fn test_mode_on(underdocks_unlocked: bool) -> Self {
        Self {
            underdocks_unlocked,
            underdocks_discovered: false,
            force_undiscovered: false,
        }
    }

    /// A real singleplayer session, where an unlocked-but-undiscovered
    /// Underdocks is forced without consuming a read.
    #[must_use]
    pub const fn live_game(underdocks_unlocked: bool, underdocks_discovered: bool) -> Self {
        Self {
            underdocks_unlocked,
            underdocks_discovered,
            force_undiscovered: true,
        }
    }
}

impl Default for ActSelectionUnlocks {
    /// Underdocks locked, so Act 1 is always Overgrowth. This is the
    /// conservative default for a caller that has not decided; the engine's
    /// own default comes from `RunState::act1`, not from here.
    fn default() -> Self {
        Self::test_mode_on(false)
    }
}

/// `ActModel.IsUnlocked` for the four shipped acts. Only Underdocks is gated.
const fn is_unlocked(act: ActVariant, unlocks: &ActSelectionUnlocks) -> bool {
    match act {
        ActVariant::Underdocks => unlocks.underdocks_unlocked,
        _ => true,
    }
}

/// `DiscoveredActs.Contains(act.Id)`. Only Underdocks can be undiscovered —
/// the defaults are discovered by construction (they are never force-picked,
/// because the branch requires `!IsDefault`).
const fn is_discovered(act: ActVariant, unlocks: &ActSelectionUnlocks) -> bool {
    match act {
        ActVariant::Underdocks => unlocks.underdocks_discovered,
        _ => true,
    }
}

/// Roll the run's act list from an **already-hashed** seed, mirroring
/// `ActModel.GetRandomList` read-for-read.
///
/// `seed_hash` is `deterministic_hash_code(canonicalize_seed(display_seed))` —
/// the same single hash `RunRngSet` takes. Use [`roll_acts_for_seed`] to go
/// straight from the display seed.
///
/// Returns the act chosen for index 0, 1, 2 in order.
#[must_use]
pub fn roll_acts(seed_hash: i32, unlocks: &ActSelectionUnlocks) -> [ActVariant; 3] {
    roll_acts_for(
        crate::GameVersion::V0_107_1,
        u64::from(seed_hash as u32),
        unlocks,
    )
}

/// [`roll_acts`] for game build `version`, from the hashed seed as a `u64`
/// (GH #788). v0.111.0 `StartRunLobby.cs:468`:
///
/// [decompiled C# omitted — see the cited path]
///
/// i.e. XxHash64 of the seed string, then `+ hash("act_selection")` in
/// `ulong` (v0.107.1 `StartRunLobby.cs:461` is the same in `uint`). Note the
/// hash here is the PLAIN `GetDeterministicHashCode` — for every non-migrated
/// seed it equals `RunRngSet.Seed`, which is what callers pass.
#[must_use]
pub fn roll_acts_for(
    version: crate::GameVersion,
    seed_hash: u64,
    unlocks: &ActSelectionUnlocks,
) -> [ActVariant; 3] {
    let mut rng = Rng::with_name_for(version, seed_hash, "act_selection");
    roll_acts_with(&mut rng, unlocks)
}

/// [`roll_acts`] against a caller-owned stream, so a test can assert the
/// exact number of reads consumed.
///
/// Faithful to `ActModel.cs:558-586`: per index, walk the group in order;
/// skip locked acts; take the first unlocked non-default undiscovered act
/// **without** drawing; otherwise `NextItem` over everything unlocked —
/// which draws even for a one-element pool, because `NextItem` calls
/// `NextInt(0, count)` unconditionally (`Rng.cs:264`).
pub fn roll_acts_with(rng: &mut Rng, unlocks: &ActSelectionUnlocks) -> [ActVariant; 3] {
    let mut out = [ActVariant::Overgrowth; 3];
    for (i, group) in ACTS_BY_INDEX.iter().enumerate() {
        let mut forced = None;
        let mut candidates: Vec<ActVariant> = Vec::with_capacity(group.len());
        for &act in *group {
            if !is_unlocked(act, unlocks) {
                continue;
            }
            if !act.is_default() && unlocks.force_undiscovered && !is_discovered(act, unlocks) {
                forced = Some(act);
                break;
            }
            candidates.push(act);
        }
        out[i] = if let Some(act) = forced {
            act
        } else {
            assert!(
                !candidates.is_empty(),
                "No unlocked acts for index {i}! (C# throws InvalidOperationException here)"
            );
            let idx = rng.next_int_exclusive(0, candidates.len() as i32) as usize;
            candidates[idx]
        };
    }
    out
}

/// [`roll_acts`] from a human-typed display seed (`"C9EXV1KTPH"`), applying
/// the game's canonicalisation + hash first.
#[must_use]
pub fn roll_acts_for_seed(display_seed: &str, unlocks: &ActSelectionUnlocks) -> [ActVariant; 3] {
    roll_acts(
        deterministic_hash_code(&canonicalize_seed(display_seed)),
        unlocks,
    )
}

/// [`roll_acts_for_seed`] for game build `version` (GH #788): canonicalise,
/// hash with that build's `GetDeterministicHashCode`, roll.
#[must_use]
pub fn roll_acts_for_seed_for(
    version: crate::GameVersion,
    display_seed: &str,
    unlocks: &ActSelectionUnlocks,
) -> [ActVariant; 3] {
    roll_acts_for(
        version,
        version.seed_hash(&canonicalize_seed(display_seed)),
        unlocks,
    )
}

/// The Act-1 variant a seed rolls — the single number the act-parity gate
/// compares against the C# oracle.
#[must_use]
pub fn act1_for_seed(display_seed: &str, unlocks: &ActSelectionUnlocks) -> ActVariant {
    roll_acts_for_seed(display_seed, unlocks)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With Underdocks locked, act 1 is Overgrowth for every seed — the world
    /// `run_room_gen` currently assumes.
    #[test]
    fn locked_underdocks_always_yields_overgrowth() {
        let unlocks = ActSelectionUnlocks::test_mode_on(false);
        for i in 0..200 {
            let acts = roll_acts_for_seed(&format!("SEED{i}"), &unlocks);
            assert_eq!(
                acts,
                [ActVariant::Overgrowth, ActVariant::Hive, ActVariant::Glory]
            );
        }
    }

    /// Unlocking Underdocks turns act 1 into a real coin flip. This is the
    /// assertion that fails the moment someone "simplifies" the pool away.
    #[test]
    fn unlocked_underdocks_splits_act1_roughly_evenly() {
        let unlocks = ActSelectionUnlocks::test_mode_on(true);
        let n = 2000;
        let under = (0..n)
            .filter(|i| act1_for_seed(&format!("SEED{i}"), &unlocks) == ActVariant::Underdocks)
            .count();
        assert!(
            (n * 2 / 5..=n * 3 / 5).contains(&under),
            "expected a ~50/50 act-1 split, got {under}/{n} Underdocks"
        );
    }

    /// Acts 2 and 3 are singletons but still consume one `NextItem` read each,
    /// because `Rng.NextItem` calls `NextInt(0, count)` even for `count == 1`.
    /// Getting this wrong desyncs nothing today (the stream is used once) but
    /// would the moment anything else reads `act_selection`.
    #[test]
    fn every_index_consumes_exactly_one_read_when_nothing_is_forced() {
        let mut rng = Rng::with_name(12345, "act_selection");
        let _ = roll_acts_with(&mut rng, &ActSelectionUnlocks::test_mode_on(true));
        assert_eq!(rng.counter(), 3, "one NextItem per act index");
    }

    /// The first-discovery force skips the draw for that index, so act 1
    /// costs zero reads and acts 2/3 land on a *different* stream position
    /// than the rolled path. A harness that models the force branch wrongly
    /// would still get act 1 right and acts 2/3 right by luck (they are
    /// singletons) — so assert the read count, not just the acts.
    #[test]
    fn first_discovery_force_consumes_no_read_for_act1() {
        let mut rng = Rng::with_name(12345, "act_selection");
        let acts = roll_acts_with(
            &mut rng,
            &ActSelectionUnlocks::live_game(true, /* discovered */ false),
        );
        assert_eq!(
            acts[0],
            ActVariant::Underdocks,
            "undiscovered act is forced"
        );
        assert_eq!(rng.counter(), 2, "act 1 forced ⇒ only acts 2 and 3 draw");
    }

    /// Once discovered, the force branch is inert and the same save falls
    /// back to the seeded coin flip.
    #[test]
    fn discovered_underdocks_falls_back_to_the_roll() {
        let live = ActSelectionUnlocks::live_game(true, true);
        let test = ActSelectionUnlocks::test_mode_on(true);
        for i in 0..50 {
            let seed = format!("SEED{i}");
            assert_eq!(
                roll_acts_for_seed(&seed, &live),
                roll_acts_for_seed(&seed, &test),
                "discovered+live must match test-mode for {seed}"
            );
        }
    }

    /// Seed canonicalisation applies before the hash, exactly as
    /// `RunRngSet` does it — `O`→`0`, `I`→`1`, upper, trim.
    #[test]
    fn display_seed_is_canonicalised_before_hashing() {
        let unlocks = ActSelectionUnlocks::test_mode_on(true);
        assert_eq!(
            roll_acts_for_seed(" c9exvIktph ", &unlocks),
            roll_acts_for_seed("C9EXV1KTPH", &unlocks)
        );
    }

    /// Act-1 rolls captured from the C# headless oracle (`roll_acts` command,
    /// `UnlockState.all` so Underdocks is unlocked, `TestMode.IsOn` so the
    /// first-discovery force branch is dead and the roll is purely seeded).
    ///
    /// Regenerate from `sts2-pycli/`:
    /// ```text
    /// PYTHONPATH=python uv run python -c "
    /// from sts2_client import Game
    /// g = Game()
    /// for s in ['SEED{}'.format(i) for i in range(40)]:
    ///     print(s, g.send({'cmd': 'roll_acts', 'seed': s})['acts'][0])
    /// g.close()"
    /// ```
    ///
    /// `C9EXV1KTPH` is the seed from the report that opened GH #544 — the real
    /// game showed Lagavulin Matriarch where the sim showed Vantom. The oracle
    /// agrees it rolls Underdocks, which is the whole diagnosis in one row.
    const CSHARP_ACT1_GOLDENS: &[(&str, ActVariant)] = &[
        ("C9EXV1KTPH", ActVariant::Underdocks),
        ("SEED0", ActVariant::Overgrowth),
        ("SEED1", ActVariant::Underdocks),
        ("SEED2", ActVariant::Underdocks),
        ("SEED3", ActVariant::Underdocks),
        ("SEED4", ActVariant::Overgrowth),
        ("SEED5", ActVariant::Underdocks),
        ("SEED6", ActVariant::Underdocks),
        ("SEED7", ActVariant::Overgrowth),
        ("SEED8", ActVariant::Overgrowth),
        ("SEED9", ActVariant::Overgrowth),
        ("SEED10", ActVariant::Overgrowth),
        ("SEED11", ActVariant::Overgrowth),
        ("SEED12", ActVariant::Underdocks),
        ("SEED13", ActVariant::Underdocks),
        ("SEED14", ActVariant::Overgrowth),
        ("SEED15", ActVariant::Overgrowth),
        ("SEED16", ActVariant::Overgrowth),
        ("SEED17", ActVariant::Overgrowth),
        ("SEED18", ActVariant::Underdocks),
        ("SEED19", ActVariant::Overgrowth),
        ("SEED20", ActVariant::Underdocks),
        ("SEED21", ActVariant::Underdocks),
        ("SEED22", ActVariant::Overgrowth),
        ("SEED23", ActVariant::Overgrowth),
        ("SEED24", ActVariant::Overgrowth),
        ("SEED25", ActVariant::Underdocks),
        ("SEED26", ActVariant::Overgrowth),
        ("SEED27", ActVariant::Overgrowth),
        ("SEED28", ActVariant::Overgrowth),
        ("SEED29", ActVariant::Underdocks),
        ("SEED30", ActVariant::Overgrowth),
        ("SEED31", ActVariant::Underdocks),
        ("SEED32", ActVariant::Overgrowth),
        ("SEED33", ActVariant::Underdocks),
        ("SEED34", ActVariant::Overgrowth),
        ("SEED35", ActVariant::Overgrowth),
        ("SEED36", ActVariant::Overgrowth),
        ("SEED37", ActVariant::Underdocks),
        ("SEED38", ActVariant::Underdocks),
        ("SEED39", ActVariant::Overgrowth),
    ];

    /// The cross-engine pin: this port reproduces the C# `act_selection`
    /// stream seed-for-seed.
    #[test]
    fn act1_roll_matches_csharp_oracle_goldens() {
        let unlocks = ActSelectionUnlocks::test_mode_on(true);
        for &(seed, want) in CSHARP_ACT1_GOLDENS {
            assert_eq!(
                act1_for_seed(seed, &unlocks),
                want,
                "act-1 roll diverged from the C# oracle for seed {seed}"
            );
        }
    }

    /// Guards the goldens themselves. A table that had drifted to a single
    /// variant would keep `act1_roll_matches_csharp_oracle_goldens` green
    /// while asserting nothing — the exact shape of the blind spot that hid
    /// GH #544 for months.
    #[test]
    fn goldens_cover_both_act1_variants() {
        let n = CSHARP_ACT1_GOLDENS.len();
        let under = CSHARP_ACT1_GOLDENS
            .iter()
            .filter(|(_, a)| *a == ActVariant::Underdocks)
            .count();
        assert!(
            under > 0 && under < n,
            "goldens must exercise BOTH act-1 variants, got {under}/{n} Underdocks"
        );
    }

    /// Ids match the C# `ModelId.Entry` slugs the oracle reports.
    #[test]
    fn act_ids_match_csharp_slugs() {
        assert_eq!(ActVariant::Overgrowth.id(), "OVERGROWTH");
        assert_eq!(ActVariant::Underdocks.id(), "UNDERDOCKS");
        assert_eq!(ActVariant::Hive.id(), "HIVE");
        assert_eq!(ActVariant::Glory.id(), "GLORY");
        assert!(!ActVariant::Underdocks.is_default());
        assert!(ActVariant::Overgrowth.is_default());
    }
}
