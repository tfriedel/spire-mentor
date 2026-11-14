//! Which build of the game the engine is simulating.
//!
//! One engine runs every supported game build; the build is a runtime value
//! threaded exactly like ascension: [`crate::run_manager::RunManager::game_version`],
//! [`crate::runstate::RunState::game_version`] and
//! [`crate::CombatState::game_version`]. Behaviour that differs between builds
//! branches on it (design: `docs/plans/2026-10-17-multi-version-v0.111.md`).
//!
//! [`GameVersion::V0_107_1`] is the default and must stay bit-identical to
//! the engine as it was before the multi-version port: every golden, digest
//! and parity fixture predates this enum and is a v0.107.1 artefact.

use std::fmt;
use std::str::FromStr;

/// A supported game build.
///
/// Ordered by release, so `version >= GameVersion::V0_111_0` reads as
/// "from v0.111.0 on".
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum GameVersion {
    /// v0.107.1 — the stable Steam build ("Major Update #2"). Default.
    #[default]
    #[serde(rename = "v0.107.1")]
    V0_107_1,
    /// v0.111.0 — the Steam `public-beta` build (commit `41cef1ea`).
    #[serde(rename = "v0.111.0")]
    V0_111_0,
    /// v0.113.0 — the Steam `public-beta` build (commit `16ac0fe3`). Same
    /// seeding as v0.111.0; content differences branch on `>= V0_113_0`.
    #[serde(rename = "v0.113.0")]
    V0_113_0,
}

impl GameVersion {
    /// Every supported build, oldest first.
    pub const ALL: [GameVersion; 3] = [
        GameVersion::V0_107_1,
        GameVersion::V0_111_0,
        GameVersion::V0_113_0,
    ];

    /// The canonical `vX.Y.Z` spelling (what [`fmt::Display`] prints).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            GameVersion::V0_107_1 => "v0.107.1",
            GameVersion::V0_111_0 => "v0.111.0",
            GameVersion::V0_113_0 => "v0.113.0",
        }
    }

    /// The build a NEW run gets when the caller names none: a workbench
    /// start request, the launcher's first pick. Moves when a newer build
    /// becomes the one worth playing (2026-11-14: v0.111.0).
    ///
    /// Not [`GameVersion::default()`], which is v0.107.1 and never moves: that
    /// is what a stored artefact with no `game_version` IS, and what every
    /// golden, digest and fixture predating this enum was recorded on.
    pub const NEW_RUN_DEFAULT: GameVersion = GameVersion::V0_111_0;

    /// Whether this is the default build ([`GameVersion::V0_107_1`]).
    ///
    /// Digests and wire emitters use it to leave the default build's bytes
    /// untouched: the version is folded in / emitted only when it is not
    /// the default, so every pre-existing v0.107.1 golden stays valid.
    #[must_use]
    pub fn is_default(self) -> bool {
        self == GameVersion::default()
    }

    // ── Seed derivation (GH #788) ───────────────────────────────────────
    //
    // Every derived RNG seed in the game is `hash(string)` and/or a sum of
    // integers, in this build's integer width. The two primitives below are
    // the only place that width lives; every seed-derivation site routes
    // through them (directly or via `Rng::with_name_for` /
    // `RunRngSet::named`), so a v0.107.1 seed is computed exactly as before
    // and a v0.111.0 seed exactly as the v0.111.0 C# computes it.

    /// `StringHelper.GetDeterministicHashCode(text)` in this build, widened
    /// to `u64`.
    ///
    /// * v0.107.1 — the 32-bit DJB-style hash ([`deterministic_hash_code`])
    ///   cast `(uint)`, then zero-extended. Every v0.107.1 call site casts
    ///   the `int` hash to `uint` before mixing it in.
    /// * v0.111.0 — XxHash64 of the UTF-8 bytes with seed 0
    ///   (`MegaCrit.Sts2.Core.Helpers/StringHelper.cs:140-153`):
    ///
    /// [decompiled C# omitted — see the cited path]
    ///
    /// This is the per-name hash (`Rng(seed, name)`, `Id.Entry`, the act
    /// selection seed). The run's master seed has one extra rule — the
    /// `"old"` prefix — and goes through [`Self::run_seed_hash`] instead.
    ///
    /// [`deterministic_hash_code`]: crate::dotnet_rng::deterministic_hash_code
    #[must_use]
    pub fn seed_hash(self, text: &str) -> u64 {
        match self {
            GameVersion::V0_107_1 => {
                u64::from(crate::dotnet_rng::deterministic_hash_code(text) as u32)
            }
            GameVersion::V0_111_0 | GameVersion::V0_113_0 => {
                xxhash_rust::xxh64::xxh64(text.as_bytes(), 0)
            }
        }
    }

    /// `RunRngSet.Seed` for the seed string `seed` — the run's master seed.
    ///
    /// v0.107.1 is `(uint)GetDeterministicHashCode(seed)`, i.e.
    /// [`Self::seed_hash`]. v0.111.0 adds a save-migration escape hatch
    /// (`MegaCrit.Sts2.Core.Runs/RunRngSet.cs:104-116`): a seed that
    /// `SerializableRunV18ToV19` rewrote to `"old" + <seed>` keeps the OLD
    /// 32-bit hash of the remainder, so a migrated run replays its old
    /// streams.
    ///
    /// [decompiled C# omitted — see the cited path]
    ///
    /// Pass the string exactly as C# hands it to `RunRngSet` (a human-typed
    /// seed is canonicalised first — which upper-cases it, so a typed
    /// `"old…"` can never reach the escape hatch). The prefix test is an
    /// ordinal ASCII match; C#'s culture-sensitive `StartsWith` agrees on
    /// every seed the game produces.
    #[must_use]
    pub fn run_seed_hash(self, seed: &str) -> u64 {
        match self {
            GameVersion::V0_107_1 => self.seed_hash(seed),
            GameVersion::V0_111_0 | GameVersion::V0_113_0 => match seed.strip_prefix("old") {
                Some(rest) => u64::from(crate::dotnet_rng::deterministic_hash_code(rest) as u32),
                None => self.seed_hash(seed),
            },
        }
    }

    /// `a + b` in this build's seed width.
    ///
    /// * v0.107.1 — seeds are `uint` and every derivation is
    ///   `(uint)((int)a + b + …)` or `uint + uint`: 32-bit two's-complement
    ///   arithmetic, so both operands are truncated to `u32`, added with
    ///   wrap-around, and the result zero-extended — bit-for-bit what the
    ///   engine did before this enum existed.
    /// * v0.111.0 — seeds are `ulong` and derivations are spelled
    ///   `(ulong)((long)a + (long)b) + c` (e.g. `Rng.cs:50`,
    ///   `EncounterModel.cs:264`, `EventModel.cs:234`). In C#'s default
    ///   unchecked context a `ulong`↔`long` cast reinterprets the same 64
    ///   bits, and `long` addition is two's-complement addition mod 2^64 —
    ///   the identical bit operation as unsigned addition mod 2^64. So
    ///   `(ulong)((long)a + (long)b)` == `a.wrapping_add(b)` for every pair
    ///   of 64-bit patterns, and so is the plain `ulong + ulong`. A signed
    ///   C# `int` operand (a floor, a slot index) is sign-extended by the
    ///   `(long)` cast; pass it through [`Self::seed_offset`], which does
    ///   the same `i64 → u64` reinterpretation.
    ///
    /// The v0.107.1 truncation is the same argument at 32 bits, which is why
    /// a negative `int` offset also works there.
    #[must_use]
    pub fn seed_add(self, a: u64, b: u64) -> u64 {
        match self {
            GameVersion::V0_107_1 => u64::from((a as u32).wrapping_add(b as u32)),
            GameVersion::V0_111_0 | GameVersion::V0_113_0 => a.wrapping_add(b),
        }
    }

    /// `seed + offset` for a signed C# `int` offset (floor, player slot,
    /// fixed mixin) — [`Self::seed_add`] with the `(long)` sign extension.
    #[must_use]
    pub fn seed_offset(self, seed: u64, offset: i64) -> u64 {
        self.seed_add(seed, offset as u64)
    }

    /// `seed + offset + hash(name)` — the shape every per-content RNG seed
    /// takes: `EventModel.BeginEvent` (offset = player slot, 0 in
    /// single-player), `EncounterModel.GenerateMonstersWithSlots` (offset =
    /// `TotalFloor`), `Rng(player, id, mixin)` (offset = slot, then + mixin),
    /// and `Rng(seed, name)` (offset 0).
    ///
    /// v0.111.0 (`EncounterModel.cs:264`, `EventModel.cs:234`):
    ///
    /// [decompiled C# omitted — see the cited path]
    ///
    /// v0.107.1 (`EncounterModel.cs:271`, `EventModel.cs:238`) is the same
    /// sum in `uint`.
    #[must_use]
    pub fn content_seed(self, seed: u64, offset: i64, name: &str) -> u64 {
        self.seed_add(self.seed_offset(seed, offset), self.seed_hash(name))
    }

    /// Bring an externally supplied master seed into this build's range:
    /// v0.107.1 keeps the low 32 bits (`RunRngSet.Seed` is a `uint`), v0.111.0
    /// keeps all 64.
    #[must_use]
    pub fn normalize_seed(self, seed: u64) -> u64 {
        match self {
            GameVersion::V0_107_1 => u64::from(seed as u32),
            GameVersion::V0_111_0 | GameVersion::V0_113_0 => seed,
        }
    }
}

impl fmt::Display for GameVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error from parsing an unknown game-version string.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown game version {0:?} (expected one of: v0.107.1, v0.111.0)")]
pub struct ParseGameVersionError(pub String);

impl FromStr for GameVersion {
    type Err = ParseGameVersionError;

    /// Accepts `v0.111.0` and `0.111.0` (and the 0.107.1 forms); leading /
    /// trailing whitespace and a capital `V` are tolerated.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let bare = trimmed
            .strip_prefix('v')
            .or_else(|| trimmed.strip_prefix('V'))
            .unwrap_or(trimmed);
        match bare {
            "0.107.1" => Ok(GameVersion::V0_107_1),
            "0.111.0" => Ok(GameVersion::V0_111_0),
            "0.113.0" => Ok(GameVersion::V0_113_0),
            _ => Err(ParseGameVersionError(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_v0107() {
        assert_eq!(GameVersion::default(), GameVersion::V0_107_1);
        assert!(GameVersion::V0_107_1.is_default());
        assert!(!GameVersion::V0_111_0.is_default());
    }

    #[test]
    fn display_parse_round_trip() {
        for v in GameVersion::ALL {
            let s = v.to_string();
            assert!(s.starts_with('v'), "{s}");
            assert_eq!(s.parse::<GameVersion>().unwrap(), v);
            assert_eq!(s[1..].parse::<GameVersion>().unwrap(), v, "bare form");
        }
        assert_eq!(GameVersion::V0_111_0.to_string(), "v0.111.0");
        assert_eq!(GameVersion::V0_107_1.to_string(), "v0.107.1");
    }

    #[test]
    fn parse_accepts_documented_forms() {
        assert_eq!("v0.111.0".parse::<GameVersion>(), Ok(GameVersion::V0_111_0));
        assert_eq!("0.111.0".parse::<GameVersion>(), Ok(GameVersion::V0_111_0));
        assert_eq!(
            " V0.107.1 ".parse::<GameVersion>(),
            Ok(GameVersion::V0_107_1)
        );
        assert_eq!("0.107.1".parse::<GameVersion>(), Ok(GameVersion::V0_107_1));
    }

    #[test]
    fn parse_rejects_unknown() {
        for bad in ["", "v0.103.2", "0.111", "latest", "v0.111.0.1"] {
            assert!(bad.parse::<GameVersion>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn ordered_by_release() {
        assert!(GameVersion::V0_107_1 < GameVersion::V0_111_0);
        let mut sorted = GameVersion::ALL;
        sorted.sort();
        assert_eq!(sorted, GameVersion::ALL);
    }

    #[test]
    fn serde_uses_canonical_spelling() {
        let json = serde_json::to_string(&GameVersion::V0_111_0).unwrap();
        assert_eq!(json, "\"v0.111.0\"");
        let back: GameVersion = serde_json::from_str("\"v0.107.1\"").unwrap();
        assert_eq!(back, GameVersion::V0_107_1);
    }

    // ── Seed derivation (GH #788) ───────────────────────────────────────

    use crate::dotnet_rng::{deterministic_hash_code, Rng};
    use crate::RunRngSet;

    const V107: GameVersion = GameVersion::V0_107_1;
    const V111: GameVersion = GameVersion::V0_111_0;

    /// XxHash64 reference vectors (seed 0) from the reference `xxhsum -H1`
    /// tool, not from the crate under test.
    #[test]
    fn xxh64_known_vectors() {
        for (text, want) in [
            ("", 0xEF46_DB37_51D8_E999_u64),
            ("up_front", 0x1A8A_7B60_DA24_3E80),
            ("NEOW", 0x1998_9CBA_53B0_5E6D),
            ("act_selection", 0x79A5_D529_9DC6_2D50),
            ("3ACCFREXKR", 0xCCD0_8772_EAB0_03D2),
            ("SLIMES_WEAK", 0xE112_62E7_3FC9_E82E),
            ("rewards", 0x20F3_BF55_0414_6BB3),
        ] {
            assert_eq!(V111.seed_hash(text), want, "xxh64({text:?})");
        }
    }

    #[test]
    fn v0107_seed_hash_is_the_old_hash_as_uint() {
        for text in ["", "up_front", "NEOW", "3ACCFREXKR", "ünïcödé"] {
            assert_eq!(
                V107.seed_hash(text),
                u64::from(deterministic_hash_code(text) as u32),
                "{text:?}"
            );
            assert_eq!(V107.run_seed_hash(text), V107.seed_hash(text));
        }
    }

    /// `RunRngSet.cs:107-115`: an `"old"`-prefixed (save-migrated) seed keeps
    /// the OLD 32-bit hash of the remainder; anything else is XxHash64.
    #[test]
    fn v0111_old_prefix_keeps_the_old_hash_of_the_remainder() {
        assert_eq!(
            V111.run_seed_hash("old3ACCFREXKR"),
            u64::from(deterministic_hash_code("3ACCFREXKR") as u32)
        );
        assert_eq!(V111.run_seed_hash("3ACCFREXKR"), 0xCCD0_8772_EAB0_03D2);
        // Case-sensitive, prefix-only.
        assert_eq!(V111.run_seed_hash("OLDX"), V111.seed_hash("OLDX"));
        assert_eq!(V111.run_seed_hash("xold"), V111.seed_hash("xold"));
        // The per-name hash never applies the escape hatch.
        assert_eq!(
            V111.seed_hash("old3ACCFREXKR"),
            xxhash_rust::xxh64::xxh64(b"old3ACCFREXKR", 0)
        );
        // v0.107.1 has no escape hatch at all.
        assert_eq!(
            V107.run_seed_hash("old3ACCFREXKR"),
            u64::from(deterministic_hash_code("old3ACCFREXKR") as u32)
        );
    }

    #[test]
    fn seed_add_wraps_in_the_builds_width() {
        assert_eq!(V107.seed_add(0xFFFF_FFFF, 2), 1);
        assert_eq!(V107.seed_add(0x1_0000_0005, 1), 6, "operands truncated");
        assert_eq!(V107.seed_offset(0, -1), 0xFFFF_FFFF);
        assert_eq!(V111.seed_add(u64::MAX, 2), 1);
        assert_eq!(V111.seed_add(0xFFFF_FFFF, 2), 0x1_0000_0001);
        // `(long)` sign extension of a negative int offset.
        assert_eq!(V111.seed_offset(5, -1), 4);
        assert_eq!(V111.seed_offset(0, -1), u64::MAX);
        assert_eq!(V107.normalize_seed(0x1_2345_6789), 0x2345_6789);
        assert_eq!(V111.normalize_seed(0x1_2345_6789), 0x1_2345_6789);
    }

    /// The v0.107.1 primitives reproduce the pre-#788 arithmetic exactly:
    /// `Rng::with_name(i32, name)` and `RunRngSet::new(i32)` are unchanged.
    #[test]
    fn v0107_derivations_are_bit_identical_to_the_legacy_formulas() {
        for seed in [0_i32, 1, -1, 12_345, i32::MIN, i32::MAX, -987_654_321] {
            let legacy = (seed as u32).wrapping_add(deterministic_hash_code("up_front") as u32);
            let set = RunRngSet::new(seed);
            assert_eq!(set.seed, u64::from(seed as u32));
            assert_eq!(set.up_front.seed(), u64::from(legacy));
            assert_eq!(
                set,
                RunRngSet::new_for(V107, u64::from(seed as u32)),
                "{seed}"
            );
            assert_eq!(
                Rng::with_name(seed, "NEOW"),
                set.named("NEOW"),
                "EventModel rng, seed {seed}"
            );
            let floor_legacy = Rng::with_name(seed.wrapping_add(3), "SLIMES_WEAK");
            assert_eq!(floor_legacy, set.named_at(3, "SLIMES_WEAK"), "{seed}");
            assert_eq!(
                set.run_manager,
                Rng::new(seed.wrapping_add(9999)),
                "run_manager stream, seed {seed}"
            );
        }
    }

    // v0.111.0 derived seeds, each computed BY HAND from the C# formula with
    // the master seed `M = xxh64("3ACCFREXKR") = 0xCCD08772EAB003D2` and the
    // `xxhsum` vectors above (64-bit wrapping sums done outside this crate).
    const M: u64 = 0xCCD0_8772_EAB0_03D2;

    /// `RunRngSet.CreateRng`: `new Rng(Seed, "up_front")` = M + xxh64("up_front").
    #[test]
    fn v0111_run_stream_seed_by_hand() {
        let set = RunRngSet::new_for(V111, M);
        assert_eq!(set.seed, M);
        assert_eq!(set.up_front.seed(), 0xE75B_02D3_C4D4_4252);
        assert_eq!(set.map_rngs.len(), 0);
        let mut set = set;
        // `StandardActMap`: `new Rng(Seed, "act_1_map")`.
        assert_eq!(set.map_rng(0).seed(), M.wrapping_add(0x0FFB_9C39_6E72_BE02));
        // Python-pin `run_manager` stream: M + 9999 in u64.
        assert_eq!(set.run_manager.seed(), 0xCCD0_8772_EAB0_2AE1);
    }

    /// `EventModel.cs:234`: `(ulong)((long)Seed + (long)0) + xxh64("NEOW")`.
    #[test]
    fn v0111_per_event_seed_by_hand() {
        let set = RunRngSet::new_for(V111, M);
        assert_eq!(set.named("NEOW").seed(), 0xE669_242D_3E60_623F);
    }

    /// `EncounterModel.cs:264`:
    /// `(ulong)((long)Seed + (long)TotalFloor) + xxh64(Id.Entry)`, floor 3.
    #[test]
    fn v0111_per_encounter_seed_by_hand() {
        let set = RunRngSet::new_for(V111, M);
        assert_eq!(set.named_at(3, "SLIMES_WEAK").seed(), 0xADE2_EA5A_2A79_EC03);
    }

    /// `Player.cs:332` + `PlayerRngSet.CreateRng`:
    /// `PlayerRngSet(hash(seed) + (ulong)slot)` then `new Rng(Seed, "rewards")`,
    /// slot 0 = M + 0 + xxh64("rewards").
    #[test]
    fn v0111_player_seed_by_hand() {
        let set = RunRngSet::new_for(V111, M);
        assert_eq!(set.rewards.seed(), 0xEDC4_46C7_EEC4_6F85);
    }

    /// `StartRunLobby.cs:468`: `new Rng(xxh64(seed), "act_selection")`.
    #[test]
    fn v0111_act_selection_seed_by_hand() {
        let rng = Rng::with_name_for(V111, M, "act_selection");
        assert_eq!(rng.seed(), 0x4676_5C9C_8876_3122);
    }

    /// A v0.111.0 stream is NOT a v0.107.1 stream: the widened seed reaches
    /// `MegaRandom` unchanged, so the draws differ.
    #[test]
    fn v0111_streams_differ_from_v0107() {
        let mut a = RunRngSet::new_for(V107, M);
        let mut b = RunRngSet::new_for(V111, M);
        assert_ne!(a.up_front.seed(), b.up_front.seed());
        let da: Vec<i32> = (0..8).map(|_| a.up_front.next_int(0, 1_000_000)).collect();
        let db: Vec<i32> = (0..8).map(|_| b.up_front.next_int(0, 1_000_000)).collect();
        assert_ne!(da, db);
        // ...and the v0.111.0 one is MegaRandom seeded with the full u64.
        let mut direct = Rng::from_seed(0xE75B_02D3_C4D4_4252);
        let mut fresh = RunRngSet::new_for(V111, M);
        for _ in 0..8 {
            assert_eq!(direct.next_int(0, 99), fresh.up_front.next_int(0, 99));
        }
    }
}
