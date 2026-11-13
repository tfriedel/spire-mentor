//! Multi-stream RNG bundle for a single run.
//!
//! Ports `sts2-rl-agent@01c094d:sts2_env/run/run_state.py:1340-1370`
//! (`class RunRngSet`). Each named stream is an independent
//! [`crate::dotnet_rng::Rng`] seeded by mixing a per-stream salt into
//! the run's master seed via [`crate::dotnet_rng::deterministic_hash_code`],
//! exactly as Python does. Streams are *not* derived from one another —
//! advancing `rewards` never perturbs `encounter`, so a fixture that
//! shifts which stream rolls first stays parity-stable.
//!
//! ## Why this matters for parity
//!
//! Python's `Rng(seed, name)` calls `deterministic_hash_code(name)` to
//! mix the salt into `seed`, producing an independent sample sequence
//! per stream. If `sts2-rs` collapses every roll into a single stream
//! (the old `rewards_rng_seed: u64` Xoshiro shim), every fixture that
//! mixes a `card_reward` and an `encounter` roll diverges from Python
//! the moment one stream draws and the other consumes a stale tail.
//! Bead `sts2-rs-ewm6b` (P9.1.g) lands the bundle; downstream beads
//! migrate individual call sites off `rewards_rng_seed` as they need
//! per-stream determinism.
//!
//! ## `combat_potion` alias
//!
//! Python sets `self.combat_potion = self.combat_potion_generation`
//! (a reference aliased to the same `Rng` object). Rust can't share an
//! `Rng` between two struct fields without indirection, so we expose
//! [`RunRngSet::combat_potion`] / [`RunRngSet::combat_potion_mut`] as
//! accessors that route to `combat_potion_generation`. Callers that
//! were `rs.rng.combat_potion.next_int(...)` in Python become
//! `rs.rng.combat_potion_mut().next_int(...)` in Rust.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::dotnet_rng::deterministic_hash_code;
use crate::dotnet_rng::Rng;

/// Mirrors Python `RUN_MANAGER_RNG_SEED_OFFSET = 9999` at
/// `sts2-rl-agent@01c094d:sts2_env/run/run_manager.py:108`. Added to
/// the master seed (no salt-hash mixing) to seed [`RunRngSet::run_manager`].
pub const RUN_MANAGER_RNG_SEED_OFFSET: i32 = 9999;

/// Number of combat-relevant RNG streams that chance-node MCTS detects on
/// and reseeds.
///
/// **Adding/removing a combat stream is a four-place edit — keep them in
/// lockstep or chance-node detection/reseeding silently breaks:**
///   1. this constant,
///   2. [`COMBAT_STREAM_SALTS`] (one distinct salt per stream),
///   3. [`RunRngSet::combat_streams_mut`] (the reseed order),
///   4. [`RunRngSet::combat_counter_fingerprint`] (the detection order).
///
/// The `[_; COMBAT_STREAM_COUNT]` return/array types tie 1↔2↔3↔4 by length
/// at compile time; the `fingerprint_and_reseed_share_one_canonical_order`
/// test pins that 3 and 4 also agree slot-for-slot (so a salt isn't applied
/// to the wrong stream). Neither can prove a *newly added* combat-consuming
/// engine stream was listed here — that judgement stays manual.
pub const COMBAT_STREAM_COUNT: usize = 9;

/// Per-stream salts mixed into the outcome-seed derivation so each combat
/// stream draws an independent realization for a given outcome ordinal.
/// Distinct odd constants (any distinct values would do — the splitmix64
/// finalizer decorrelates them); order matches [`RunRngSet::combat_streams_mut`].
const COMBAT_STREAM_SALTS: [u64; COMBAT_STREAM_COUNT] = [
    0x1111_1111_1111_1111,
    0x2222_2222_2222_2223,
    0x3333_3333_3333_3333,
    0x4444_4444_4444_4445,
    0x5555_5555_5555_5555,
    0x6666_6666_6666_6667,
    0x7777_7777_7777_7777,
    0x8888_8888_8888_8889,
    0x9999_9999_9999_9999,
];

/// SplitMix64 finalizer — a strong avalanche mix used to derive
/// well-distributed seeds from structured `(base, n, salt)` inputs.
#[must_use]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive an independent stream seed for outcome `n` of a chance event from
/// the chance node's `base`, the outcome ordinal `n`, and a per-stream
/// `salt`. Public so chance-node MCTS can reuse the same mixing for the
/// legacy `state.rng` reseed (passing a dedicated salt).
#[must_use]
pub fn outcome_stream_seed(base: u64, n: u32, salt: u64) -> i32 {
    #[allow(clippy::cast_possible_truncation)]
    {
        outcome_stream_seed_u64(base, n, salt) as i32
    }
}

/// Full-width sibling of [`outcome_stream_seed`]. The search-side fast reseed
/// ([`crate::dotnet_rng::Rng::from_search_seed`]) consumes a full `u64`, so it
/// keeps all 64 bits of the `splitmix64` mix rather than truncating to `i32`.
#[must_use]
pub fn outcome_stream_seed_u64(base: u64, n: u32, salt: u64) -> u64 {
    let input = base
        ^ u64::from(n)
            .rotate_left(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    splitmix64(input)
}

/// Salt for reseeding the legacy `state.rng` (the unnamed `DotNetRandom`)
/// alongside the [`RunRngSet`] combat streams in chance-node outcome
/// generation. Distinct from every [`COMBAT_STREAM_SALTS`] entry.
pub const LEGACY_RNG_SALT: u64 = 0xABCD_EF01_2345_6789;

/// Number of run-level "draw" streams resampled as chance outcomes by the
/// outer-loop planner (card-reward, shop, deck-transform, unknown map-point
/// reveal, treasure-room relic roll). The run-level twin of
/// [`COMBAT_STREAM_COUNT`]; membership stays a manual judgement (validated
/// against real callbacks by the planning integration tests — see
/// `sts2-runai/tests/planning_run_draw_streams.rs`'s "Test #7" membership
/// tripwire, which drives a real `PlanningRun` and asserts the aggregate
/// fingerprint moves iff a covered stream did). AlphaZero M3 Task 2 (bead
/// `sts2-rs-0lr87`) widened this from 3 to 5, adding `unknown_map_point`
/// and `treasure_room` — the M2 final review flagged these as run-scoped
/// stochastic streams the twin didn't cover.
pub const RUN_DRAW_STREAM_COUNT: usize = 5;

/// Per-stream salts for run-draw outcome reseeding (distinct odd constants,
/// disjoint from [`COMBAT_STREAM_SALTS`] and [`LEGACY_RNG_SALT`]); order
/// matches [`RunRngSet::run_draw_streams_mut`].
const RUN_DRAW_STREAM_SALTS: [u64; RUN_DRAW_STREAM_COUNT] = [
    0xAAAA_AAAA_AAAA_AAA9,
    0xBBBB_BBBB_BBBB_BBBB,
    0xCCCC_CCCC_CCCC_CCCD,
    0xDDDD_DDDD_DDDD_DDDD,
    0xEEEE_EEEE_EEEE_EEE9,
];

/// Independent RNG streams keyed by purpose. Mirrors Python's
/// `RunRngSet`.
///
/// The struct is constructed once at run-creation from the master seed
/// and lives on [`crate::runstate::RunState`] for the run's duration.
/// Streams are mutable: every `next_*` call on a stream advances *only*
/// that stream's counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRngSet {
    /// C# `RunRngSet.Seed` — the hashed master seed all per-stream salts
    /// mix into. Stored so per-act map RNGs (and per-content RNGs, see
    /// [`Self::named`]) can be created lazily. A `uint` (≤ `u32::MAX`) in
    /// v0.107.1, a `ulong` in v0.111.0 (GH #788).
    pub seed: u64,

    /// The game build whose seed arithmetic [`Self::seed`] and every derived
    /// stream follow (GH #788). Set at construction; mirrors
    /// `RunState::game_version`.
    pub game_version: crate::GameVersion,

    // --- Run-scoped, derived from `seed` -------------------------------
    pub up_front: Rng,
    pub shuffle: Rng,
    pub unknown_map_point: Rng,
    pub combat_card_generation: Rng,
    pub combat_potion_generation: Rng,
    pub combat_card_selection: Rng,
    pub combat_energy_costs: Rng,
    pub combat_targets: Rng,
    pub monster_ai: Rng,
    pub niche: Rng,
    pub combat_orbs: Rng,
    /// Note: Python names the salt `"treasure_room_relics"` (NOT
    /// `"treasure_room"` — the attribute is short for ergonomics but
    /// the underlying salt string is the longer form). Parity demands
    /// the longer salt.
    pub treasure_room: Rng,

    /// Run-manager-level RNG. Mirrors Python
    /// `RunManager.__init__: self._rng = Rng(seed + RUN_MANAGER_RNG_SEED_OFFSET)`
    /// at `sts2-rl-agent@01c094d:sts2_env/run/run_manager.py:214` with
    /// `RUN_MANAGER_RNG_SEED_OFFSET = 9999` (line 108). Unlike the
    /// per-stream salts above, this one is seeded by raw integer
    /// offset, NOT by hashing a salt name — Python uses `Rng(seed + N)`
    /// directly. Drives encounter-pool selection, the per-combat
    /// `combat_seed`, gold rolls, and any future room-side rolls that
    /// in Python were attributed to `self._rng`. Bead `sts2-rs-lakuh`.
    pub run_manager: Rng,

    // --- Player-scoped, derived from `seed + GetPlayerSlotIndex` --------
    /// C# `PlayerRngSet` seed = `hash(seed) + <player term>`
    /// (`Player.InitializeSeed`). The player term is **0** in single-player:
    /// v0.107.1 (`Player.cs:329`) uses `GetPlayerSlotIndex(this) ==
    /// Players.IndexOf(this) == 0`; v0.103.2 (`Player.cs:243`) used
    /// `NetId == 1`. So the player-scoped streams (`rewards`, `shops`,
    /// `transformations`) seed off the SAME base as the run-scoped
    /// streams — the stale `+1` (NetId) desynced every reward/shop/
    /// transform roll vs the v0.107.1 oracle (first combat gold +N,
    /// deck-composition, card-rarity). Bead `sts2-rs-ms8mm` family.
    pub rewards: Rng,
    pub shops: Rng,
    pub transformations: Rng,
    /// C# `PlayerRng.NonCombatGeneration` (v0.113.0, `PlayerRngType.cs:7`,
    /// `PlayerRngSet.cs:34`): out-of-combat card/potion picks that used to
    /// roll on `Rng.Niche` / `CombatPotionGeneration` / `PlayerRng.Rewards`
    /// (Fishing Rod, Phial Holster, Neow's Bones, War Paint, ...). Created
    /// for every build — `new Rng(playerSeed, "non_combat_generation")` —
    /// but only v0.113.0+ content draws from it, so older builds never
    /// advance it.
    pub non_combat_generation: Rng,

    /// Per-act map RNGs, lazily created via [`Self::map_rng`]. Mirrors
    /// Python `self.map_rngs: dict[int, Rng]`.
    pub map_rngs: BTreeMap<i32, Rng>,
}

impl RunRngSet {
    /// The stream a combat card pick that used `Rng.Shuffle` reads in game
    /// build `version`: v0.113.0 moved Beat Down, Uproar, Catastrophe and
    /// Stampede to `Rng.CombatCardSelection` (decomp BeatDown.cs:26,
    /// Uproar.cs:34,38, Catastrophe.cs:25,29, StampedePower.cs:28).
    pub fn shuffle_or_card_selection(&mut self, version: crate::GameVersion) -> &mut Rng {
        if version >= crate::GameVersion::V0_113_0 {
            &mut self.combat_card_selection
        } else {
            &mut self.shuffle
        }
    }

    /// Construct a fresh `RunRngSet` from `master_seed`.
    ///
    /// The hash applied to `str(master_seed)` matches Python's
    /// `deterministic_hash_code(str(master_seed))`, so a Rust caller
    /// passing `0_i32` and a Python caller passing `0` produce
    /// byte-identical opening sequences on every named stream.
    #[must_use]
    pub fn new(master_seed: i32) -> Self {
        Self::new_for(crate::GameVersion::V0_107_1, u64::from(master_seed as u32))
    }

    /// [`Self::new`] for game build `version`, from the already-hashed
    /// master seed (`RunRngSet.Seed`; see
    /// [`crate::GameVersion::run_seed_hash`] for going from the seed string).
    /// The seed is first brought into the build's range
    /// ([`crate::GameVersion::normalize_seed`]); every stream is then
    /// `new Rng(Seed, snake_case(stream))` in that build's arithmetic.
    #[must_use]
    pub fn new_for(version: crate::GameVersion, master_seed: u64) -> Self {
        // C# `RunRngSet(string seed)` (v0.103.2 RunRngSet.cs:54-63) does
        // ONE hash from the string seed: `Seed = (uint)GetDeterministicHashCode(seed)`.
        // The string entrypoint takes the canonicalised seed (e.g.
        // "BK9KTUB5Z1") and the integer seed is the result of that single
        // hash. The Python pin's `RunRngSet.__init__` re-hashes via
        // `deterministic_hash_code(str(master_seed))` — a SECOND hash on
        // the integer-stringified seed — which produces a totally
        // different stream and prevented corpus map-gen parity with C#.
        //
        // The Rust port previously mirrored Python verbatim. Bead
        // `sts2-rs-5hsqp` corpus parity investigation 2026-05-23 traced
        // this as the smoking gun: PathGenerate produced correct start
        // cols against Python (which has the same double-hash bug) but
        // diverged from the C# scraped corpus. Treat the i32 argument
        // here as already-hashed-once (i.e. the result of
        // `deterministic_hash_code(canonical_string_seed)`).
        let v = version;
        let seed = v.normalize_seed(master_seed);
        // v0.107.1 `Player.cs:329`: PlayerRngSet seed = hash(seed) +
        // GetPlayerSlotIndex(this), and GetPlayerSlotIndex == 0 in
        // single-player. v0.103.2 used `+ NetId` (== 1); the stale `+1`
        // desynced the player reward streams. Bead `sts2-rs-ms8mm` family.
        // v0.111.0 `Player.cs:332` is the same sum in `ulong`:
        // `new PlayerRngSet(GetDeterministicHashCode(seed) + (ulong)slot)`.
        // (For an `"old"`-prefixed migrated seed C# hashes the WHOLE string
        // here with XxHash64 while `RunRngSet` uses the old hash of the
        // remainder; the engine only ever sees the hashed master seed, so it
        // models the non-migrated case where the two agree.)
        let player_seed = v.seed_offset(seed, 0);
        let named = |base: u64, name: &str| Rng::with_name_for(v, base, name);
        Self {
            seed,
            game_version: v,
            up_front: named(seed, "up_front"),
            shuffle: named(seed, "shuffle"),
            unknown_map_point: named(seed, "unknown_map_point"),
            combat_card_generation: named(seed, "combat_card_generation"),
            combat_potion_generation: named(seed, "combat_potion_generation"),
            combat_card_selection: named(seed, "combat_card_selection"),
            combat_energy_costs: named(seed, "combat_energy_costs"),
            combat_targets: named(seed, "combat_targets"),
            monster_ai: named(seed, "monster_ai"),
            niche: named(seed, "niche"),
            combat_orbs: named(seed, "combat_orbs"),
            treasure_room: named(seed, "treasure_room_relics"),
            // Python `self._rng = Rng(seed + 9999)` — direct int offset,
            // no salt-hash mixing. Bead `sts2-rs-lakuh`. No C# analog; the
            // offset is added in the build's seed width like every other
            // derivation.
            run_manager: Rng::from_seed(
                v.seed_offset(seed, i64::from(RUN_MANAGER_RNG_SEED_OFFSET)),
            ),
            rewards: named(player_seed, "rewards"),
            shops: named(player_seed, "shops"),
            transformations: named(player_seed, "transformations"),
            non_combat_generation: named(player_seed, "non_combat_generation"),
            map_rngs: BTreeMap::new(),
        }
    }

    /// A per-content RNG off the run's master seed: C# `new Rng(RunState.Rng.Seed,
    /// name)` and its single-player equivalents `EventModel.BeginEvent`'s
    /// `Rng(Seed + slot + hash(Id.Entry))` and `Rng(player, id, 0)` (slot 0).
    /// Computed in this set's game build ([`crate::GameVersion::content_seed`]).
    #[must_use]
    pub fn named(&self, name: &str) -> Rng {
        self.named_at(0, name)
    }

    /// [`Self::named`] with a signed `int` offset folded into the master seed
    /// first: `Rng(Seed + offset + hash(name))` — e.g. the per-encounter roster
    /// RNG, whose offset is `TotalFloor` (`EncounterModel.GenerateMonstersWithSlots`).
    #[must_use]
    pub fn named_at(&self, offset: i64, name: &str) -> Rng {
        let seed = self.game_version.content_seed(self.seed, offset, name);
        if crate::dotnet_rng::rng_trace_enabled() {
            eprintln!("RNGMAP seed={seed} name={name}");
        }
        Rng::from_seed(seed)
    }

    /// Whether this build has the v0.113.0 RNG-stream moves (out-of-combat
    /// generation on `PlayerRng.NonCombatGeneration`, see the "RNG stream
    /// changes" table in `docs/plans/2026-11-12-v0.113.0-port-inventory.md`).
    #[must_use]
    pub fn has_v0113_streams(&self) -> bool {
        self.game_version >= crate::GameVersion::V0_113_0
    }

    /// `PlayerRng.NonCombatGeneration` from v0.113.0 on, else `Rng.Niche`
    /// (the pre-v0.113.0 stream of Beautiful Bracelet, Distinguished Cape,
    /// Fishing Rod, Fragrant Mushroom, Kaleidoscope, Neow's Bones' curse,
    /// Royal Stamp, Sand Castle, War Hammer, War Paint, Whetstone).
    pub fn non_combat_or_niche(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.non_combat_generation
        } else {
            &mut self.niche
        }
    }

    /// `PlayerRng.NonCombatGeneration` from v0.113.0 on, else
    /// `Rng.CombatPotionGeneration` (Alchemical Coffer, Delicate Frond,
    /// Entropic Brew, Phial Holster).
    pub fn non_combat_or_combat_potion(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.non_combat_generation
        } else {
            &mut self.combat_potion_generation
        }
    }

    /// `PlayerRng.NonCombatGeneration` from v0.113.0 on, else
    /// `PlayerRng.Rewards` (Neow's Bones' bundle shuffle).
    pub fn non_combat_or_rewards(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.non_combat_generation
        } else {
            &mut self.rewards
        }
    }

    /// `PlayerRng.Transformations` from v0.113.0 on, else `Rng.Niche`
    /// (Astrolabe, New Leaf, Pandora's Box transforms).
    pub fn transformations_or_niche(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.transformations
        } else {
            &mut self.niche
        }
    }

    /// `PlayerRng.Shops` from v0.113.0 on, else `PlayerRng.Rewards`: the
    /// merchant's card-rarity / upgrade rolls (`CardFactory.CreateForMerchant`)
    /// and relic-rarity rolls (`MerchantInventory`, `MerchantRelicEntry`
    /// restock), which v0.113.0 moved off the reward stream.
    pub fn shops_or_rewards(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.shops
        } else {
            &mut self.rewards
        }
    }

    /// `PlayerRng.Rewards` from v0.113.0 on, else `Rng.Niche` (Wing Charm).
    pub fn rewards_or_niche(&mut self) -> &mut Rng {
        if self.has_v0113_streams() {
            &mut self.rewards
        } else {
            &mut self.niche
        }
    }

    /// [`Self::seed`] as the wire spells `$.rng.run_seed`: the same integer.
    /// (Kept as a method so producers do not each re-derive the width.)
    #[must_use]
    pub const fn wire_seed(&self) -> u64 {
        self.seed
    }

    /// Borrow the `combat_potion` alias. Mirrors Python's
    /// `self.combat_potion = self.combat_potion_generation` aliasing
    /// (Python returns the same `Rng` object; Rust returns a borrow
    /// of the underlying field).
    #[must_use]
    pub const fn combat_potion(&self) -> &Rng {
        &self.combat_potion_generation
    }

    /// Mutable borrow of the `combat_potion` alias. Use when calling
    /// `next_*` on the alias.
    pub const fn combat_potion_mut(&mut self) -> &mut Rng {
        &mut self.combat_potion_generation
    }

    /// The combat-relevant streams, in a fixed canonical order, as a
    /// snapshot of mutable borrows. These are exactly the streams that
    /// `apply()` can consume during a combat transition (verified by
    /// grepping `state.run_rng.<stream>` consumers in core): chance-node
    /// MCTS uses this set for (a) stochastic-transition *detection* via
    /// counter deltas and (b) per-outcome *reseeding*. `niche` is included
    /// because a handful of card effects roll on it (e.g. card_effects.rs).
    ///
    /// Run-/map-scoped streams (rewards, shops, map_rngs, up_front, …) are
    /// deliberately excluded — they are not consumed inside a combat step,
    /// so folding them in would only add noise to detection.
    fn combat_streams_mut(&mut self) -> [&mut Rng; COMBAT_STREAM_COUNT] {
        [
            &mut self.shuffle,
            &mut self.monster_ai,
            &mut self.combat_targets,
            &mut self.combat_card_selection,
            &mut self.combat_energy_costs,
            &mut self.combat_orbs,
            &mut self.combat_card_generation,
            &mut self.combat_potion_generation,
            &mut self.niche,
        ]
    }

    /// Counters of every combat-relevant stream, in the canonical order of
    /// [`Self::combat_streams_mut`]. Two fingerprints differ iff at least
    /// one combat stream advanced between the snapshots — the signal
    /// chance-node MCTS uses to decide whether an applied action was
    /// stochastic. Counters are monotone (`fast_forward_counter` asserts
    /// non-decreasing), so "fingerprint unchanged" ⟺ "no combat randomness
    /// consumed".
    #[must_use]
    pub fn combat_counter_fingerprint(&self) -> [u32; COMBAT_STREAM_COUNT] {
        [
            self.shuffle.counter(),
            self.monster_ai.counter(),
            self.combat_targets.counter(),
            self.combat_card_selection.counter(),
            self.combat_energy_costs.counter(),
            self.combat_orbs.counter(),
            self.combat_card_generation.counter(),
            self.combat_potion_generation.counter(),
            self.niche.counter(),
        ]
    }

    /// Reseed every combat-relevant stream to an independent fresh seed
    /// derived from `(base, n, per-stream salt)`, resetting each cursor to
    /// 0. This realizes **outcome `n`** of a stochastic transition for
    /// chance-node MCTS: starting from the parent's stored pre-action
    /// state, reseed then re-execute the stochastic action to draw one
    /// i.i.d. sample of the joint outcome distribution.
    ///
    /// Independence: the streams are independent by construction (see the
    /// module docs / `streams_are_independent` test), and the splitmix64
    /// finalizer over distinct `(base, n, salt)` inputs yields
    /// well-distributed, mutually-decorrelated seeds — so different `n`
    /// draw different realizations and different streams draw independently.
    /// `base` decorrelates distinct chance nodes from each other.
    ///
    /// Note: this intentionally abandons the real RNG timeline. Inside the
    /// search we model an information set, not the engine's ground-truth
    /// cursor; reroot reinstalls the real cursors when a move is committed.
    pub fn reseed_combat_streams(&mut self, base: u64, n: u32) {
        for (idx, stream) in self.combat_streams_mut().into_iter().enumerate() {
            let salt = COMBAT_STREAM_SALTS[idx];
            // Search-side reseed: this stream only samples nature for MCTS
            // rollouts, so use the fast non-parity seeder (the .NET
            // `System.Random` init here was ~7% of search instructions —
            // callgrind 2026-06-22).
            *stream = Rng::from_search_seed(outcome_stream_seed_u64(base, n, salt));
        }
    }

    /// Borrow (or lazily create) the per-act map RNG. Mirrors
    /// Python's `get_map_rng(act_index)` exactly — including the
    /// 1-based salt name (`act_1_map`, `act_2_map`, …) computed from
    /// the 0-based `act_index` argument.
    pub fn map_rng(&mut self, act_index: i32) -> &mut Rng {
        let (seed, version) = (self.seed, self.game_version);
        self.map_rngs.entry(act_index).or_insert_with(|| {
            Rng::with_name_for(version, seed, &format!("act_{}_map", act_index + 1))
        })
    }

    /// The run-level draw streams, in canonical order: card-reward rarity /
    /// out-of-combat potion rolls (`rewards`), shop stock/pricing (`shops`),
    /// deck transforms (`transformations`), unknown-map-point reveals
    /// (`unknown_map_point`), and treasure-room relic rolls
    /// (`treasure_room`). These are the streams a run-level *decision* draws
    /// from; combat streams are excluded (combat has its own twin). Map
    /// *topology* (`map_rng`) stays excluded too — it's deterministic per
    /// seed by design, unlike the point-reveal roll on `unknown_map_point`.
    fn run_draw_streams_mut(&mut self) -> [&mut Rng; RUN_DRAW_STREAM_COUNT] {
        [
            &mut self.rewards,
            &mut self.shops,
            &mut self.transformations,
            &mut self.unknown_map_point,
            &mut self.treasure_room,
        ]
    }

    /// Counters of every run-draw stream, canonical order of
    /// [`Self::run_draw_streams_mut`]. Two fingerprints differ iff at least
    /// one draw stream advanced — the signal the outer-loop planner uses to
    /// decide whether an applied run action was a stochastic draw.
    #[must_use]
    pub fn run_draw_counter_fingerprint(&self) -> [u32; RUN_DRAW_STREAM_COUNT] {
        [
            self.rewards.counter(),
            self.shops.counter(),
            self.transformations.counter(),
            self.unknown_map_point.counter(),
            self.treasure_room.counter(),
        ]
    }

    /// Reseed every run-draw stream to an independent fresh seed derived
    /// from `(base, n, per-stream salt)`, resetting each cursor to 0 —
    /// realizes **outcome `n`** of a stochastic draw for the outer-loop
    /// planner. The run-level sibling of [`Self::reseed_combat_streams`];
    /// same information-set semantics (abandons the real RNG timeline for
    /// the search subtree).
    pub fn reseed_run_draw_streams(&mut self, base: u64, n: u32) {
        for (idx, stream) in self.run_draw_streams_mut().into_iter().enumerate() {
            let salt = RUN_DRAW_STREAM_SALTS[idx];
            *stream = Rng::from_search_seed(outcome_stream_seed_u64(base, n, salt));
        }
    }

    /// Mutable access to one named stream, or `None` for a name this
    /// engine does not carry.
    ///
    /// The single name→stream table every *restore* path shares: the
    /// combat wire's `$.rng.counters` (`sts2-wire`'s `state_from_dict`)
    /// and the run-level live-state import
    /// (`PyRunManager::debug_set_rng_counters`). Names are the snake_case
    /// forms SpireMentor emits (`SimStateExtractor.RngState`) and
    /// `state_to_dict` re-emits, so producer and consumer line up
    /// field-for-field. Two copies of this match are exactly how a
    /// silently half-restored RNG lineage happens, so there is one.
    ///
    /// [`Self::run_manager`] is deliberately absent: it is a Python-pin
    /// stream with no C# analog, so no producer can carry a cursor for
    /// it. [`Self::map_rngs`] is absent for the same reason — those are
    /// lazily derived per act, not read off the world.
    pub fn stream_mut(&mut self, name: &str) -> Option<&mut Rng> {
        Some(match name {
            "up_front" => &mut self.up_front,
            "shuffle" => &mut self.shuffle,
            "unknown_map_point" => &mut self.unknown_map_point,
            "combat_card_generation" => &mut self.combat_card_generation,
            "combat_potion_generation" => &mut self.combat_potion_generation,
            "combat_card_selection" => &mut self.combat_card_selection,
            "combat_energy_costs" => &mut self.combat_energy_costs,
            "combat_targets" => &mut self.combat_targets,
            "monster_ai" => &mut self.monster_ai,
            "niche" => &mut self.niche,
            "combat_orbs" => &mut self.combat_orbs,
            "treasure_room" => &mut self.treasure_room,
            "rewards" => &mut self.rewards,
            "shops" => &mut self.shops,
            "transformations" => &mut self.transformations,
            "non_combat_generation" => &mut self.non_combat_generation,
            _ => return None,
        })
    }
}

impl Default for RunRngSet {
    /// Default to `RunRngSet::new(0)` — deterministic golden-test seed.
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_mut_names_every_stream_a_producer_can_carry() {
        let mut set = RunRngSet::new(7);
        // The names SpireMentor emits (`SimStateExtractor.RngState`) plus the
        // two the engine's own `state_to_dict` adds. Every one must resolve,
        // or a restore silently drops that stream's lineage.
        for name in [
            "up_front",
            "shuffle",
            "unknown_map_point",
            "combat_card_generation",
            "combat_potion_generation",
            "combat_card_selection",
            "combat_energy_costs",
            "combat_targets",
            "monster_ai",
            "niche",
            "combat_orbs",
            "treasure_room",
            "rewards",
            "shops",
            "transformations",
            "non_combat_generation",
        ] {
            assert!(
                set.stream_mut(name).is_some(),
                "{name} must resolve to a stream"
            );
        }
        // `run_manager` is a Python-pin stream with no C# analog: no producer
        // can carry a cursor for it, so naming it is a vocabulary mismatch,
        // not a stream to restore.
        assert!(set.stream_mut("run_manager").is_none());
        assert!(set.stream_mut("not_a_stream").is_none());
    }

    #[test]
    fn stream_mut_hands_back_the_stream_itself() {
        let mut set = RunRngSet::new(7);
        set.stream_mut("shuffle").unwrap().fast_forward_counter(5);
        assert_eq!(set.shuffle.counter(), 5);
    }

    #[test]
    fn seed_is_master_directly() {
        // After bead `sts2-rs-5hsqp` parity fix, `RunRngSet::new(i)` no
        // longer re-hashes `str(i)`. The caller is responsible for
        // hashing the canonical string seed before passing it in.
        let set = RunRngSet::new(0);
        assert_eq!(set.seed, 0);
    }

    #[test]
    fn run_scoped_streams_share_seed_base() {
        // All run-scoped streams should mix `seed` (not `seed + 1`)
        // with their salt — verify by reconstructing the expected seed
        // for one stream and checking the underlying Rng's seed.
        let set = RunRngSet::new(0);
        let expected =
            u64::from((set.seed as u32).wrapping_add(deterministic_hash_code("up_front") as u32));
        assert_eq!(set.up_front.seed(), expected);
    }

    #[test]
    fn player_scoped_streams_use_player_seed() {
        // v0.107.1: PlayerRngSet seed = hash(seed) + GetPlayerSlotIndex,
        // which is 0 in single-player, so the player streams share the
        // run-seed base (NO `+1` NetId offset). Bead `sts2-rs-ms8mm` family.
        let set = RunRngSet::new(0);
        let player_seed = set.seed;
        let expected =
            u64::from((player_seed as u32).wrapping_add(deterministic_hash_code("rewards") as u32));
        assert_eq!(set.rewards.seed(), expected);
    }

    #[test]
    fn treasure_room_uses_long_salt() {
        // The Python attribute is `treasure_room` but the salt string
        // is `"treasure_room_relics"`. Parity demands the long form.
        let set = RunRngSet::new(0);
        let expected = u64::from(
            (set.seed as u32).wrapping_add(deterministic_hash_code("treasure_room_relics") as u32),
        );
        assert_eq!(set.treasure_room.seed(), expected);
    }

    #[test]
    fn streams_are_independent() {
        let mut set = RunRngSet::new(42);
        let r0 = set.rewards.next_int(0, 1_000);
        // Drawing from `encounter`-equivalent streams must NOT shift
        // the `rewards` sequence.
        for _ in 0..32 {
            let _ = set.combat_card_generation.next_int(0, 1_000);
            let _ = set.monster_ai.next_int(0, 1_000);
        }
        let r1 = set.rewards.next_int(0, 1_000);
        // Two consecutive draws from the same stream are nearly always
        // different at this granularity; the point is r1 matches the
        // value rewards.next_int would have produced regardless of any
        // intervening draws from other streams.
        let mut fresh = RunRngSet::new(42);
        let f0 = fresh.rewards.next_int(0, 1_000);
        let f1 = fresh.rewards.next_int(0, 1_000);
        assert_eq!(r0, f0);
        assert_eq!(r1, f1);
    }

    #[test]
    fn combat_potion_aliases_combat_potion_generation() {
        let mut set = RunRngSet::new(7);
        let alias_seed = set.combat_potion().seed();
        let direct_seed = set.combat_potion_generation.seed();
        assert_eq!(alias_seed, direct_seed);

        // Mutating via the alias is observable on the field.
        let v = set.combat_potion_mut().next_int(0, 100);
        let _ = set; // shut up unused
        let mut set2 = RunRngSet::new(7);
        let direct_v = set2.combat_potion_generation.next_int(0, 100);
        assert_eq!(v, direct_v);
    }

    #[test]
    fn map_rng_is_lazy_and_per_act() {
        let mut set = RunRngSet::new(0);
        assert!(set.map_rngs.is_empty());

        let act0_seed = set.map_rng(0).seed();
        assert_eq!(set.map_rngs.len(), 1);

        // Reusing the same act_index returns the same Rng (no
        // re-creation), so its counter advances rather than resets.
        let _ = set.map_rng(0).next_int(0, 100);
        let counter_after = set.map_rng(0).counter();
        assert!(counter_after > 0);

        // Different act_index seeds a new stream with a different salt.
        let act1_seed = set.map_rng(1).seed();
        assert_ne!(act0_seed, act1_seed);
        assert_eq!(set.map_rngs.len(), 2);
    }

    #[test]
    fn map_rng_salt_uses_one_based_act_number() {
        // Python: `f"act_{act_index + 1}_map"` — index 0 → "act_1_map".
        let mut set = RunRngSet::new(0);
        let master_seed = set.seed;
        let r = set.map_rng(0);
        let expected = u64::from(
            (master_seed as u32).wrapping_add(deterministic_hash_code("act_1_map") as u32),
        );
        assert_eq!(r.seed(), expected);
    }

    #[test]
    fn default_matches_new_zero() {
        let a = RunRngSet::default();
        let b = RunRngSet::new(0);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.rewards.seed(), b.rewards.seed());
    }

    #[test]
    fn combat_fingerprint_tracks_only_combat_streams() {
        let mut set = RunRngSet::new(7);
        let fp0 = set.combat_counter_fingerprint();
        // Advancing a non-combat stream must NOT change the fingerprint.
        let _ = set.rewards.next_int(0, 100);
        let _ = set.shops.next_int(0, 100);
        assert_eq!(
            set.combat_counter_fingerprint(),
            fp0,
            "run-scoped streams are outside the combat fingerprint"
        );
        // Advancing a combat stream MUST change it.
        let _ = set.shuffle.next_int(0, 100);
        assert_ne!(
            set.combat_counter_fingerprint(),
            fp0,
            "a combat-stream draw must change the fingerprint"
        );
    }

    #[test]
    fn reseed_resets_combat_cursors_and_varies_by_n() {
        let mut a = RunRngSet::new(0);
        // Advance some combat streams, then reseed: cursors must reset.
        let _ = a.shuffle.next_int(0, 100);
        let _ = a.monster_ai.next_int(0, 100);
        a.reseed_combat_streams(0xDEAD_BEEF, 0);
        assert_eq!(
            a.combat_counter_fingerprint(),
            [0; COMBAT_STREAM_COUNT],
            "reseed resets every combat cursor to 0"
        );

        // Same (base, n) reseed is reproducible; different n diverges.
        let mut b = RunRngSet::new(999);
        b.reseed_combat_streams(0xDEAD_BEEF, 0);
        assert_eq!(
            a.shuffle.seed(),
            b.shuffle.seed(),
            "same (base,n) yields identical stream seeds regardless of master seed"
        );
        let mut c = RunRngSet::new(0);
        c.reseed_combat_streams(0xDEAD_BEEF, 1);
        assert_ne!(
            a.shuffle.seed(),
            c.shuffle.seed(),
            "different outcome ordinal n yields a different stream seed"
        );
    }

    #[test]
    fn outcome_seed_streams_are_mutually_distinct() {
        // For a fixed (base, n), the per-stream salts must produce distinct
        // seeds so streams draw independently within one outcome.
        let base = 0x1234_5678_9ABC_DEF0;
        let n = 3;
        let mut seeds: Vec<i32> = COMBAT_STREAM_SALTS
            .iter()
            .map(|&salt| outcome_stream_seed(base, n, salt))
            .collect();
        let len = seeds.len();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(
            seeds.len(),
            len,
            "per-stream outcome seeds must be distinct"
        );
    }

    #[test]
    fn fingerprint_and_reseed_share_one_canonical_order() {
        // `combat_streams_mut` (reseed order, indexed against
        // COMBAT_STREAM_SALTS) and `combat_counter_fingerprint` (detection
        // order) are two hand-maintained lists that MUST refer to the same
        // physical stream at every index — otherwise a salt is applied to
        // the wrong stream and/or a stream's advance is read from the wrong
        // fingerprint slot. Advance exactly the idx-th reseed-order stream
        // and assert exactly the idx-th fingerprint slot moves.
        for idx in 0..COMBAT_STREAM_COUNT {
            let mut set = RunRngSet::new(0);
            let fp0 = set.combat_counter_fingerprint();
            {
                let streams = set.combat_streams_mut();
                let _ = streams[idx].next_int(0, 2);
            }
            let fp1 = set.combat_counter_fingerprint();
            for j in 0..COMBAT_STREAM_COUNT {
                if j == idx {
                    assert_ne!(
                        fp1[j], fp0[j],
                        "advancing reseed-order stream {idx} must move fingerprint slot {idx}"
                    );
                } else {
                    assert_eq!(
                        fp1[j], fp0[j],
                        "advancing reseed-order stream {idx} must not move fingerprint slot {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn clone_preserves_stream_state() {
        let mut a = RunRngSet::new(123);
        let _ = a.rewards.next_int(0, 1_000);
        let _ = a.rewards.next_int(0, 1_000);
        let b = a.clone();
        // Next draw from the clone matches the next draw from `a`.
        let an = a.rewards.next_int(0, 1_000);
        let mut b_mut = b;
        let bn = b_mut.rewards.next_int(0, 1_000);
        assert_eq!(an, bn);
    }

    #[test]
    fn run_draw_fingerprint_moves_only_when_a_draw_stream_advances() {
        let mut set = RunRngSet::new(42);
        let fp0 = set.run_draw_counter_fingerprint();
        // Advancing a NON-draw stream (up_front) must not change the fingerprint.
        let _ = set.up_front.next_int(0, 100);
        assert_eq!(
            set.run_draw_counter_fingerprint(),
            fp0,
            "non-draw stream leaked into fingerprint"
        );
        // Advancing a draw stream (rewards) must change it.
        let _ = set.rewards.next_int(0, 100);
        assert_ne!(
            set.run_draw_counter_fingerprint(),
            fp0,
            "rewards draw not detected"
        );
    }

    #[test]
    fn reseed_run_draw_streams_is_deterministic_and_n_varies() {
        let mut a = RunRngSet::new(7);
        let mut b = RunRngSet::new(7);
        a.reseed_run_draw_streams(0xABCD, 1);
        b.reseed_run_draw_streams(0xABCD, 1);
        assert_eq!(
            a.rewards.next_int(0, 1_000_000),
            b.rewards.next_int(0, 1_000_000),
            "same (base,n) must reproduce"
        );
        // AZ M3 Task 2 widened the twin to 5 streams; the two additions
        // (`unknown_map_point`, `treasure_room`) must reseed deterministically
        // too, not just the original `rewards`/`shops` pair.
        assert_eq!(
            a.unknown_map_point.next_int(0, 1_000_000),
            b.unknown_map_point.next_int(0, 1_000_000),
            "same (base,n) must reproduce for unknown_map_point"
        );
        assert_eq!(
            a.treasure_room.next_int(0, 1_000_000),
            b.treasure_room.next_int(0, 1_000_000),
            "same (base,n) must reproduce for treasure_room"
        );
        let mut c = RunRngSet::new(7);
        c.reseed_run_draw_streams(0xABCD, 2);
        // Different n → (almost surely) a different first draw.
        assert_ne!(
            a.shops.next_int(0, 1_000_000),
            c.shops.next_int(0, 1_000_000),
            "different n must differ"
        );
        assert_ne!(
            a.unknown_map_point.next_int(0, 1_000_000),
            c.unknown_map_point.next_int(0, 1_000_000),
            "different n must differ for unknown_map_point"
        );
        assert_ne!(
            a.treasure_room.next_int(0, 1_000_000),
            c.treasure_room.next_int(0, 1_000_000),
            "different n must differ for treasure_room"
        );
    }

    #[test]
    fn run_draw_fingerprint_and_reseed_share_one_canonical_order() {
        // `run_draw_streams_mut` (reseed order, indexed against
        // RUN_DRAW_STREAM_SALTS) and `run_draw_counter_fingerprint` (detection
        // order) are two hand-maintained lists that MUST refer to the same
        // physical stream at every index — otherwise a salt is applied to
        // the wrong stream and/or a stream's advance is read from the wrong
        // fingerprint slot. Advance exactly the idx-th reseed-order stream
        // and assert exactly the idx-th fingerprint slot moves. Mirrors the
        // combat-side `fingerprint_and_reseed_share_one_canonical_order` proof.
        for idx in 0..RUN_DRAW_STREAM_COUNT {
            let mut set = RunRngSet::new(0);
            let fp0 = set.run_draw_counter_fingerprint();
            {
                let streams = set.run_draw_streams_mut();
                let _ = streams[idx].next_int(0, 2);
            }
            let fp1 = set.run_draw_counter_fingerprint();
            for j in 0..RUN_DRAW_STREAM_COUNT {
                if j == idx {
                    assert_ne!(
                        fp1[j], fp0[j],
                        "advancing reseed-order stream {idx} must move fingerprint slot {idx}"
                    );
                } else {
                    assert_eq!(
                        fp1[j], fp0[j],
                        "advancing reseed-order stream {idx} must not move fingerprint slot {j}"
                    );
                }
            }
        }
    }
}
