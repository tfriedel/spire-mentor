//! Standard act-map generator — Rust port of the Python source-of-truth.
//!
//! Bead `sts2-rs-wn4wc` (P9.2.b). Port pin:
//! `sts2-rl-agent@01c094d:sts2_env/map/generator.py`.
//!
//! Mirrors the C# `MegaCrit.Sts2.Core.Map/StandardActMap.cs` algorithm:
//! 7-column grid, 7 random paths from row 1 to the top room row,
//! crossover prevention on lateral moves, room-type assignment with
//! all five placement constraints (lower-map rest/elite ban, upper-map
//! rest ban, parent/child/sibling adjacency uniqueness), plus the
//! spoils-variant centered toward a fixed treasure node.
//!
//! Parity tests against Python seed snapshots live in
//! `tests/map_generator_parity.rs` (integration test); the unit tests
//! below cover structural invariants that don't require RNG-byte
//! parity (graph connectivity, placement-rule validity, _center_grid
//! shifts).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::dotnet_rng::Rng;
use crate::map::{ActMap, MapCoord, MapPointIdx, MapPointType};
use std::collections::BTreeMap;

/// `MAP_WIDTH` from `generator.py:15` — every act-map grid is 7
/// columns wide.
pub const MAP_WIDTH: i32 = 7;

/// `MAP_ITERATIONS` from `generator.py:16` — every act spawns 7
/// independent random paths; the 2nd is guaranteed to start at a
/// different column from the 1st (≤100 attempts).
pub const MAP_ITERATIONS: usize = 7;

// -----------------------------------------------------------------------------
// Golden-path map: a 1-column spine used by tutorials / fixed slices.
// -----------------------------------------------------------------------------

/// `GOLDEN_PATH_POINT_TYPES` from `generator.py:17-34`. Fixed sequence
/// of room types used by `generate_golden_path_map` (the tutorial /
/// thin-slice spine). 16 entries for single-player; index 2 (the
/// second `Monster`) is removed for multiplayer (skip).
const GOLDEN_PATH_POINT_TYPES: [MapPointType; 16] = [
    MapPointType::Monster,
    MapPointType::Unknown,
    MapPointType::Monster,
    MapPointType::RestSite,
    MapPointType::Monster,
    MapPointType::RestSite,
    MapPointType::Unknown,
    MapPointType::Treasure,
    MapPointType::Unknown,
    MapPointType::Treasure,
    MapPointType::Unknown,
    MapPointType::Shop,
    MapPointType::Elite,
    MapPointType::RestSite,
    MapPointType::Elite,
    MapPointType::RestSite,
];

/// Build a fixed 1-column "golden path" map at column 3 (the grid's
/// center). Used by the thin-slice tutorial / `generate_golden_path_map`
/// callers. `player_count > 1` drops the second `Monster` (index 2).
///
/// Port pin: `generator.py:72-92`.
#[must_use]
pub fn generate_golden_path_map(player_count: u32) -> ActMap {
    let mut point_types: Vec<MapPointType> = GOLDEN_PATH_POINT_TYPES.to_vec();
    if player_count > 1 {
        point_types.remove(2);
    }
    let mut map = ActMap::new(point_types.len() as i32);
    let start = map.get_or_create(3, 0);
    map.point_mut(start).point_type = MapPointType::Ancient;
    map.start = Some(start);
    let boss_row = (point_types.len() as i32) + 1;
    let boss = map.get_or_create(3, boss_row);
    map.point_mut(boss).point_type = MapPointType::Boss;
    map.boss = Some(boss);

    let mut previous = start;
    for (i, &pt) in point_types.iter().enumerate() {
        let row = (i as i32) + 1;
        let idx = map.get_or_create(3, row);
        map.point_mut(idx).point_type = pt;
        map.add_child(previous, idx);
        previous = idx;
    }
    map.add_child(previous, boss);
    map
}

// -----------------------------------------------------------------------------
// Standard act map: 7 random paths from row 1 → top row + room-type roll.
// -----------------------------------------------------------------------------

/// Check whether the directed edge `current → (target_col, current.row+1)`
/// would cross an existing edge. Mirrors the `_has_invalid_crossover`
/// helper in `generator.py:95-115` — straight-up moves never cross;
/// otherwise look at the "blocker" point at `(target_col, current.row)`
/// and reject if any of its children moves in the opposite lateral
/// direction.
fn has_invalid_crossover(map: &ActMap, current_idx: MapPointIdx, target_col: i32) -> bool {
    let current = map.point(current_idx);
    let direction = target_col - current.col();
    if direction == 0 {
        return false;
    }
    let Some(blocker_idx) = map
        .grid
        .get(&MapCoord::new(target_col, current.row()))
        .copied()
    else {
        return false;
    };
    let blocker_col = map.point(blocker_idx).col();
    for &child_idx in &map.point(blocker_idx).children {
        let child_dir = map.point(child_idx).col() - blocker_col;
        if child_dir == -direction {
            return true;
        }
    }
    false
}

/// Pick the next `(col, row)` for a random walk from `current_idx`.
/// Tries `[-1, 0, +1]` after a shuffle; if all three would cause a
/// crossover, falls back to "straight up" (the Python fallback).
fn generate_next_coord(
    map: &ActMap,
    current_idx: MapPointIdx,
    rng: &mut Rng,
    version: crate::GameVersion,
) -> (i32, i32) {
    let current = map.point(current_idx);
    let col = current.col();
    let min_col = col.max(1) - 1;
    let max_col = (col + 1).min(MAP_WIDTH - 1);
    let mut directions: Vec<i32> = vec![-1, 0, 1];
    rng.shuffle_for(version, &mut directions);
    for d in directions {
        let target_col = match d {
            -1 => min_col,
            1 => max_col,
            _ => col,
        };
        let target_row = current.row() + 1;
        if !has_invalid_crossover(map, current_idx, target_col) {
            return (target_col, target_row);
        }
    }
    (col, current.row() + 1)
}

/// Walk a random path from `start_idx` upward until it reaches the
/// top room row (just below the boss). Each step calls
/// [`generate_next_coord`] and wires the chosen child via
/// `map.add_child`. Mirrors Python's `_path_generate` in
/// `generator.py:142-149`.
fn path_generate(
    map: &mut ActMap,
    start_idx: MapPointIdx,
    rng: &mut Rng,
    version: crate::GameVersion,
) {
    let mut current = start_idx;
    while map.point(current).row() < map.map_length - 1 {
        let (col, row) = generate_next_coord(map, current, rng, version);
        let next_idx = map.get_or_create(col, row);
        map.add_child(current, next_idx);
        current = next_idx;
    }
}

// -----------------------------------------------------------------------------
// Room-type assignment with the five placement constraints.
// -----------------------------------------------------------------------------

const PLACEMENT_RESTRICTED: &[MapPointType] = &[
    MapPointType::Elite,
    MapPointType::RestSite,
    MapPointType::Treasure,
    MapPointType::Shop,
];

const SIBLING_RESTRICTED: &[MapPointType] = &[
    MapPointType::RestSite,
    MapPointType::Monster,
    MapPointType::Unknown,
    MapPointType::Elite,
    MapPointType::Shop,
];

/// Read-only check of the five placement constraints. Pinned to C#
/// `StandardActMap.IsValidPointType` (`StandardActMap.cs:382-456`),
/// NOT the Python pin — the Python pin uses `row < 5` for the lower
/// bound while C# uses `row < 6` (one row deeper). The Python pin's
/// bound was followed verbatim by this port until bead `sts2-rs-5hsqp`
/// flagged Acts where row-5 RestSite/Elite appeared in Rust but never
/// in scraped C# corpus runs. Switched to C# semantics:
///
/// 1. Lower map (row < 6): no `RestSite` or `Elite`. v0.113.0 lowers the
///    RestSite floor one row (`StandardActMap.cs:31`, `_lowestRowByPointType
///    = { RestSite: 5, Elite: 6 }`, checked by `IsValidForLower` at `:445`).
/// 2. Upper map (row ≥ `map_length - 3`): no `RestSite`.
/// 3. Parent adjacency: `Elite`/`RestSite`/`Treasure`/`Shop` can't
///    share type with any parent.
/// 4. Child adjacency: same restricted set can't share with a child.
/// 5. Sibling uniqueness: `RestSite`/`Monster`/`Unknown`/`Elite`/`Shop`
///    can't share with any sibling (points sharing a parent).
pub(crate) fn is_type_valid_for_point_v(
    map: &ActMap,
    point_idx: MapPointIdx,
    candidate: MapPointType,
    map_length: i32,
    version: crate::GameVersion,
) -> bool {
    let point = map.point(point_idx);
    let row = point.row();
    let lowest_rest_row = if version >= crate::GameVersion::V0_113_0 {
        5
    } else {
        6
    };
    if (candidate == MapPointType::RestSite && row < lowest_rest_row)
        || (candidate == MapPointType::Elite && row < 6)
    {
        return false;
    }
    if row >= map_length - 3 && candidate == MapPointType::RestSite {
        return false;
    }
    if PLACEMENT_RESTRICTED.contains(&candidate) {
        for &parent_idx in &point.parents {
            if map.point(parent_idx).point_type == candidate {
                return false;
            }
        }
        for &child_idx in &point.children {
            if map.point(child_idx).point_type == candidate {
                return false;
            }
        }
    }
    if SIBLING_RESTRICTED.contains(&candidate) {
        // Inline sibling computation: ActMap::siblings isn't on the
        // landed API (only the underlying parents/children fields).
        for &parent_idx in &point.parents {
            for &sib_idx in &map.point(parent_idx).children {
                if sib_idx != point_idx && map.point(sib_idx).point_type == candidate {
                    return false;
                }
            }
        }
    }
    true
}

/// Try every type currently in `pool` (in order) — if a valid one is
/// found, consume it and return; otherwise re-enqueue and keep
/// cycling. If the entire pool is exhausted with no valid pick,
/// returns `Unassigned` (caller falls back to `Monster`).
///
/// Port pin: `_get_next_valid_type_from_queue` in `generator.py:262-277`.
fn get_next_valid_type_from_queue(
    pool: &mut std::collections::VecDeque<MapPointType>,
    map: &ActMap,
    point_idx: MapPointIdx,
    map_length: i32,
    version: crate::GameVersion,
) -> MapPointType {
    if version >= crate::GameVersion::V0_113_0 {
        // v0.113.0 `GetNextValidPointType` (`StandardActMap.cs:396-417`,
        // `SpoilsActMap.cs:425-446`): take the FIRST valid type and remove
        // only it, leaving the rest of the queue in order (0.111 rotated
        // every rejected type to the back). No valid type leaves the queue
        // untouched (0.111's full rotation was also a no-op).
        let Some(pos) = pool
            .iter()
            .position(|&t| is_type_valid_for_point_v(map, point_idx, t, map_length, version))
        else {
            return MapPointType::Unassigned;
        };
        return pool.remove(pos).expect("position is in range");
    }
    let len = pool.len();
    for _ in 0..len {
        let candidate = pool.pop_front().expect("len > 0 inside loop");
        if is_type_valid_for_point_v(map, point_idx, candidate, map_length, version) {
            return candidate;
        }
        pool.push_back(candidate);
    }
    MapPointType::Unassigned
}

/// Counts struct used by [`assign_room_types`] and the spoils variant.
#[derive(Debug, Clone, Copy)]
pub struct PoolCounts {
    pub num_elites: i32,
    pub num_shops: i32,
    pub num_unknowns: i32,
    pub num_rests: i32,
}

/// Assign room types to every non-meta node. Fixed rows first
/// (top row = `RestSite`, treasure row = `Treasure` or `Elite` per
/// `replace_treasure_with_elite`, bottom room row = `Monster`), then
/// drain a shuffled pool of the remaining types via the cycling
/// queue, fall back to `Monster` for leftovers, finally re-stamp
/// `Boss` + `Ancient` on the meta nodes.
///
/// Port pin: `_assign_room_types` in `generator.py:194-259`.
fn assign_room_types(
    map: &mut ActMap,
    rng: &mut Rng,
    counts: PoolCounts,
    replace_treasure_with_elite: bool,
    version: crate::GameVersion,
) {
    let map_length = map.map_length;
    let row_count = map_length;

    // Fixed: top room row (row_count - 1) = RestSite
    let top_room_coords: Vec<MapCoord> = map
        .get_row(row_count - 1)
        .into_iter()
        .map(|p| p.coord)
        .collect();
    for c in top_room_coords {
        let idx = map.grid[&c];
        map.point_mut(idx).point_type = MapPointType::RestSite;
    }

    // Fixed: treasure row = (row_count - 7), iff > 0
    let treasure_row = row_count - 7;
    if treasure_row > 0 {
        let treasure_coords: Vec<MapCoord> = map
            .get_row(treasure_row)
            .into_iter()
            .map(|p| p.coord)
            .collect();
        let stamped = if replace_treasure_with_elite {
            MapPointType::Elite
        } else {
            MapPointType::Treasure
        };
        for c in treasure_coords {
            let idx = map.grid[&c];
            map.point_mut(idx).point_type = stamped;
        }
    }

    // Fixed: row 1 = Monster
    let row1_coords: Vec<MapCoord> = map.get_row(1).into_iter().map(|p| p.coord).collect();
    for c in row1_coords {
        let idx = map.grid[&c];
        map.point_mut(idx).point_type = MapPointType::Monster;
    }

    // Build the pool queue.
    let mut pool: std::collections::VecDeque<MapPointType> = std::collections::VecDeque::new();
    for _ in 0..counts.num_rests {
        pool.push_back(MapPointType::RestSite);
    }
    for _ in 0..counts.num_shops {
        pool.push_back(MapPointType::Shop);
    }
    for _ in 0..counts.num_elites {
        pool.push_back(MapPointType::Elite);
    }
    for _ in 0..counts.num_unknowns {
        pool.push_back(MapPointType::Unknown);
    }

    // Three-iteration pool drain — mirrors C# `StandardActMap.AssignRemaining
    // TypesToRandomPoints` (`StandardActMap.cs:341-362`) which loops
    // `for (int i = 0; i < 3; i++)`, each iteration re-collecting unassigned
    // points + `StableShuffle` + queue-cycle drain. The Python pin
    // `01c094d` only did this once, but C# v0.103.2 (the canonical game)
    // does three passes. The extra passes both consume more RNG (via the
    // shuffle) and fill more nodes from the pool, leaving fewer Monster
    // fallbacks.
    //
    // Bead: sts2-rs-genyg (R3.2 sub-bead under sts2-rs-5hsqp).
    let start = map.start;
    let boss = map.boss;
    let trace = std::env::var_os("MAP_ASSIGN_TRACE").is_some();
    if trace {
        eprintln!(
            "[ASN] pool: {} rests, {} shops, {} elites, {} unknowns",
            counts.num_rests, counts.num_shops, counts.num_elites, counts.num_unknowns
        );
    }
    for iter_idx in 0..3 {
        if pool.is_empty() {
            if trace {
                eprintln!("[ASN] iter {iter_idx}: pool empty, exit");
            }
            break;
        }
        // C# `StandardActMap.AssignRemainingTypesToRandomPoints` collects
        // unassigned via `GetAllMapPoints()` (column-major Grid walk:
        // outer c, inner r; see `ActMap.cs:74-96`), then calls
        // `StableShuffle` which sorts by `MapPoint.CompareTo` —
        // delegating to `MapCoord.CompareTo` = lexicographic `(col, row)`
        // (see `MapCoord.cs:67-70`). The sort step makes the result
        // canonical-by-coord regardless of GetAllMapPoints enumeration
        // order.
        //
        // The previous Rust port collected `MapPointIdx` (a `u16`
        // insertion-order alias) and let `stable_shuffle` sort by
        // integer value — i.e. by insertion order, NOT by (col, row).
        // After Fisher-Yates the indices mapped to a DIFFERENT
        // permutation of coords than the C# code, which desync'd every
        // downstream position assignment. Sort by coord here to mirror
        // C# `MapPoint.CompareTo` exactly.
        let mut unassigned: Vec<(MapCoord, MapPointIdx)> = map
            .grid
            .iter()
            .filter(|(_, &i)| Some(i) != start && Some(i) != boss)
            .filter(|(_, &i)| map.point(i).point_type == MapPointType::Unassigned)
            .map(|(c, &i)| (*c, i))
            .collect();
        // Equivalent to C# `list2.Sort()` step inside `StableShuffle`:
        // tuple key sorts by col then row (matches `MapCoord.CompareTo`).
        unassigned.sort_by_key(|(c, _)| (c.col, c.row));
        // Fisher-Yates on the coord-sorted list. We cannot reuse
        // `Rng::stable_shuffle` here because it would re-sort by
        // `(MapCoord, MapPointIdx)` lexicographic which is identical to
        // the explicit sort above; instead call the bare `shuffle`
        // (== C# `UnstableShuffle`) after the deliberate sort.
        rng.shuffle_for(version, &mut unassigned);
        if trace {
            let order: Vec<String> = unassigned
                .iter()
                .map(|(c, _)| format!("({},{})", c.col, c.row))
                .collect();
            eprintln!(
                "[ASN] iter {iter_idx} ({} nodes) order: {}",
                unassigned.len(),
                order.join(",")
            );
        }
        for (coord, point_idx) in &unassigned {
            if pool.is_empty() {
                break;
            }
            let assigned =
                get_next_valid_type_from_queue(&mut pool, map, *point_idx, map_length, version);
            if trace {
                eprintln!(
                    "[ASN]   ({},{}) <- {:?} pool_remaining={}",
                    coord.col,
                    coord.row,
                    assigned,
                    pool.len()
                );
            }
            map.point_mut(*point_idx).point_type = assigned;
        }
    }

    // Any remaining `Unassigned` → `Monster`.
    let all_room_coords: Vec<MapCoord> = map
        .grid
        .iter()
        .filter(|(_, &i)| Some(i) != start && Some(i) != boss)
        .map(|(c, _)| *c)
        .collect();
    for c in all_room_coords {
        let idx = map.grid[&c];
        if map.point(idx).point_type == MapPointType::Unassigned {
            map.point_mut(idx).point_type = MapPointType::Monster;
        }
    }

    if let Some(boss_idx) = map.boss {
        map.point_mut(boss_idx).point_type = MapPointType::Boss;
    }
    if let Some(start_idx) = map.start {
        map.point_mut(start_idx).point_type = MapPointType::Ancient;
    }
}

/// Roll the four pool counts for a given act/ascension.
///
/// **Pinned to C# v0.103.2** per-act `GetMapPointTypes(Rng)` overrides
/// (`Underdocks.cs:103-108`, `Hive.cs:108-113`, `Glory.cs:97-102`,
/// `Overgrowth.cs:125-130`); the Python pin's `_map_point_type_counts`
/// uses a clone-pre-roll dance that does NOT match real game behavior
/// for Acts 1/2/3 (bead `sts2-rs-5hsqp` fixed Act 1, `sts2-rs-ldrer`
/// fixed Acts 2/3).
///
/// - Act 0 / Act 1 (Underdocks): rests gauss(7, 1, 6, 7); unknowns
///   gauss(12, 1, 10, 14). 2 gauss rolls on main rng.
///
/// **Both Act-1 variants share this branch, and that is correct at
/// v0.107.1.** `Overgrowth.GetMapPointTypes` and
/// `Underdocks.GetMapPointTypes` are line-for-line identical
/// (`Overgrowth.cs:125-130`, `Underdocks.cs:110-115`), so porting Underdocks
/// (GH #544) needs no change here — the act-1 map skeleton is variant-blind
/// even though everything filling it is not. `map_point_type_counts` takes no
/// [`crate::act_selection::ActVariant`] for that reason; pinned by
/// `act1_variants_share_one_map_point_type_roll`.
/// - Act 2 (Hive): rests gauss(6, 1, 6, 7); unknowns gauss(12, 1, 10, 14) − 1.
///   2 gauss rolls on main rng, rest BEFORE unknown.
/// - Act 3 (Glory): rests NextInt(5, 7) (half-open → {5, 6}); unknowns
///   gauss(12, 1, 10, 14) − 1. 1 NextInt + 1 gauss on main rng.
/// - Other (e.g. Overgrowth Act 4): same as Act 0/1.
///
/// Elites = round(5 × 1.6 if SwarmingElites else 5 × 1); shops = 3.
/// Both come from the `MapPointTypeCounts` initializer defaults in C#
/// and do NOT draw from the RNG.
pub fn map_point_type_counts(rng: &mut Rng, ascension_level: i32, act_index: i32) -> PoolCounts {
    // C# `MapPointTypeCounts` (v0.103.2) only touches ASC 1
    // (`SwarmingElites`, scales elites ×1.6). There is NO Gloom/A6
    // reduction of rests in C# — `AscensionLevel.Inflation` is the A6
    // ascension and it affects gold/costs, not the map. Python's
    // `has_gloom` was fictional. Removing it lifted parity from 23/39
    // to 33/39+ (bead `sts2-rs-5hsqp`).
    let has_swarming = ascension_level >= 1;

    let base_elites = |swarm: bool| -> i32 {
        let scale = if swarm { 1.6 } else { 1.0 };
        (5.0_f64 * scale).round_ties_even() as i32
    };

    if act_index == 0 || act_index == 1 {
        // C# `Underdocks.GetMapPointTypes` (v0.103.2 Underdocks.cs:103-108):
        //   restCount    = mapRng.NextGaussianInt(7, 1, 6, 7);
        //   unknownCount = MapPointTypeCounts.StandardRandomUnknownCount(mapRng);
        //                = mapRng.NextGaussianInt(12, 1, 10, 14);
        // Only TWO gaussian rolls — elites/shops come from the
        // `MapPointTypeCounts` initializer defaults (NO RNG draw).
        //
        // The Python pin's `base_counts` helper does a third throwaway
        // `next_gaussian_int(5, 1, 3, 6)` for rests, which the Rust
        // port previously mirrored faithfully. That extra roll desyncs
        // the RNG counter by one full gaussian (~10 next_double calls)
        // versus the real C# game and prevents corpus parity. Drop it
        // for Act 0/1 (bead `sts2-rs-5hsqp`).
        let num_rests = rng.next_gaussian_int(7.0, 1.0, 6, 7);
        let num_unknowns = rng.next_gaussian_int(12.0, 1.0, 10, 14);
        return PoolCounts {
            num_elites: base_elites(has_swarming),
            num_shops: 3,
            num_unknowns,
            num_rests,
        };
    }

    if act_index == 2 {
        // C# `Hive.GetMapPointTypes` (v0.103.2 Hive.cs:108-113):
        //   restCount    = mapRng.NextGaussianInt(6, 1, 6, 7);   // mean=6 (not 7)
        //   unknownCount = MapPointTypeCounts.StandardRandomUnknownCount(mapRng) - 1
        //                = mapRng.NextGaussianInt(12, 1, 10, 14) - 1;
        // Two gauss rolls on main rng, rest BEFORE unknown. Elites
        // and shops come from the `MapPointTypeCounts` initializer
        // defaults (no RNG draw).
        //
        // The prior implementation mirrored the Python pin's
        // `_map_point_type_counts` clone-pre-roll dance which does NOT
        // match real game RNG order/count and blocks Act-2 corpus
        // parity. Replaced for bead `sts2-rs-ldrer`.
        let num_rests = rng.next_gaussian_int(6.0, 1.0, 6, 7);
        let num_unknowns = rng.next_gaussian_int(12.0, 1.0, 10, 14) - 1;
        return PoolCounts {
            num_elites: base_elites(has_swarming),
            num_shops: 3,
            num_unknowns,
            num_rests,
        };
    }

    if act_index == 3 {
        // C# `Glory.GetMapPointTypes` (v0.103.2 Glory.cs:97-102):
        //   restCount    = mapRng.NextInt(5, 7);                 // half-open [5, 7) → {5, 6}
        //   unknownCount = MapPointTypeCounts.StandardRandomUnknownCount(mapRng) - 1;
        // Our `next_int_exclusive(5, 7)` matches C# `NextInt(5, 7)`
        // (verified against the dotnet_rng vectors). One NextInt + one
        // gauss on main rng, rest BEFORE unknown.
        //
        // Previously fell through to the gauss(5,1,3,6) default, which
        // diverged from C# in both distribution and counter footprint.
        // Bead `sts2-rs-ldrer`.
        let num_rests = rng.next_int_exclusive(5, 7);
        let num_unknowns = rng.next_gaussian_int(12.0, 1.0, 10, 14) - 1;
        return PoolCounts {
            num_elites: base_elites(has_swarming),
            num_shops: 3,
            num_unknowns,
            num_rests,
        };
    }

    // Fallback for unrecognized acts (Overgrowth Act 4 et al.) —
    // mirror Underdocks shape: two gauss rolls on main rng, no -1 on
    // unknowns. Matches `Overgrowth.cs:125-130`. Other acts beyond 4
    // are not in v0.103.2; revisit if/when they ship.
    let num_rests = rng.next_gaussian_int(7.0, 1.0, 6, 7);
    let num_unknowns = rng.next_gaussian_int(12.0, 1.0, 10, 14);
    PoolCounts {
        num_elites: base_elites(has_swarming),
        num_shops: 3,
        num_unknowns,
        num_rests,
    }
}

/// If only the left two columns OR only the right two columns are
/// empty, slide the entire grid by ±1 to recenter it. No-op if both
/// (or neither) sides are empty. Mirrors `_center_grid` in
/// `generator.py:280-303`.
fn center_grid(map: &mut ActMap) {
    if map.points.is_empty() {
        return;
    }
    // Emptiness must be read from the LIVE grid, not the `points` arena:
    // `MapPathPruning::remove_point` drops a node's grid slot but keeps its
    // `MapPoint` in `points` (so index handles stay valid). Reading `points`
    // here counts pruned-away nodes as still occupying their column, so a
    // boundary column emptied by pruning looks occupied and the recenter
    // shift is skipped — diverging from C# `MapPostProcessing.IsColumnEmpty`,
    // which checks `Grid[col, row]`. (Manifested as a uniform +1 column
    // offset vs C#, e.g. RP131.)
    let cols_used: std::collections::BTreeSet<i32> = map.grid.keys().map(|c| c.col).collect();
    let left_empty = !cols_used.contains(&0) && !cols_used.contains(&1);
    let right_empty =
        !cols_used.contains(&(MAP_WIDTH - 2)) && !cols_used.contains(&(MAP_WIDTH - 1));
    let shift = if left_empty && !right_empty {
        -1
    } else if right_empty && !left_empty {
        1
    } else {
        return;
    };
    let mut new_grid: BTreeMap<MapCoord, MapPointIdx> = BTreeMap::new();
    // Collect existing (coord, idx) pairs from the OLD grid, then
    // re-stamp into the new grid + each point's coord field.
    let old_entries: Vec<(MapCoord, MapPointIdx)> =
        map.grid.iter().map(|(c, &i)| (*c, i)).collect();
    for (coord, idx) in old_entries {
        // C# `CenterGrid` shifts only the path `Grid`; the boss and start
        // meta-nodes are separate `BossMapPoint`/`StartingMapPoint` props
        // pinned at the center column (`GetColumnCount()/2`) and are NOT in
        // the grid array, so they never move. Keep ours fixed too — else the
        // boss drifts off its center anchor when the grid recenters.
        let is_meta = map.start == Some(idx) || map.boss == Some(idx);
        let new_col = if is_meta {
            coord.col
        } else {
            (coord.col + shift).clamp(0, MAP_WIDTH - 1)
        };
        let new_coord = MapCoord::new(new_col, coord.row);
        map.point_mut(idx).coord = new_coord;
        new_grid.insert(new_coord, idx);
    }
    map.grid = new_grid;
}

/// Port of C# `MapPostProcessing.SpreadAdjacentMapPoints`
/// (v0.103.2 MapPostProcessing.cs:155-205). For each row, repeatedly
/// move each node to a column slot (within its parent/child adjacency
/// constraints) that maximizes its minimum distance to any other node
/// in that row. RNG-free.
fn spread_adjacent_map_points(map: &mut ActMap) {
    let row_count = map.map_length + 1; // rows 0..=map_length inclusive
    for row in 0..row_count {
        // Snapshot the live row nodes (idx + current col).
        let row_nodes: Vec<MapPointIdx> = map
            .grid
            .iter()
            .filter_map(|(c, &i)| if c.row == row { Some(i) } else { None })
            .collect();
        if row_nodes.len() < 2 {
            continue;
        }
        // C# iterates `list` (insertion order from grid scan). The
        // outer do-while repeats until no node moved in a full pass.
        loop {
            let mut moved_any = false;
            for &idx in &row_nodes {
                let cur_col = map.point(idx).col();
                let allowed = compute_allowed_positions(map, idx);
                // Compute current gap (min distance to other row nodes).
                let cur_gap = compute_gap(map, cur_col, &row_nodes, idx);
                let mut best_col = cur_col;
                let mut best_gap = cur_gap;
                for &cand in &allowed {
                    if cand == cur_col {
                        continue;
                    }
                    // C#: `grid[cand, row] == null || grid[cand, row] == current`.
                    // Since we're moving `current`, the slot is OK if empty
                    // or contains `current` itself.
                    let occupant = map.grid.get(&MapCoord::new(cand, row)).copied();
                    if let Some(occ_idx) = occupant {
                        if occ_idx != idx {
                            continue;
                        }
                    }
                    let g = compute_gap(map, cand, &row_nodes, idx);
                    if g > best_gap {
                        best_col = cand;
                        best_gap = g;
                    }
                }
                if best_col != cur_col {
                    // Move idx from cur_col → best_col.
                    map.grid.remove(&MapCoord::new(cur_col, row));
                    let new_coord = MapCoord::new(best_col, row);
                    map.point_mut(idx).coord = new_coord;
                    map.grid.insert(new_coord, idx);
                    moved_any = true;
                }
            }
            if !moved_any {
                break;
            }
            // C# `SpreadAdjacentMapPoints` does NOT re-snapshot the row
            // list inside the do-while — it reuses the original list
            // reference (same MapPoint objects, now with mutated cols).
        }
    }
}

/// `GetNeighborAllowedPositions(col, total)` — columns in `[col-1, col+1]`
/// clipped to `[0, total)`.
fn neighbor_allowed_positions(col: i32, total: i32) -> std::collections::BTreeSet<i32> {
    let mut out = std::collections::BTreeSet::new();
    for d in -1..=1 {
        let n = col + d;
        if n >= 0 && n < total {
            out.insert(n);
        }
    }
    out
}

/// `GetAllowedPositions(node, total)` — intersection of neighbor sets
/// for every parent and every child.
fn compute_allowed_positions(map: &ActMap, idx: MapPointIdx) -> std::collections::BTreeSet<i32> {
    let mut allowed: std::collections::BTreeSet<i32> = (0..MAP_WIDTH).collect();
    let p = map.point(idx);
    let parents = p.parents.clone();
    let children = p.children.clone();
    for &par in &parents {
        let pc = map.point(par).col();
        let n = neighbor_allowed_positions(pc, MAP_WIDTH);
        allowed = allowed.intersection(&n).copied().collect();
    }
    for &ch in &children {
        let cc = map.point(ch).col();
        let n = neighbor_allowed_positions(cc, MAP_WIDTH);
        allowed = allowed.intersection(&n).copied().collect();
    }
    allowed
}

/// `ComputeGap` — minimum |col diff| from `candidate_col` to any other
/// node in `row_nodes` (excluding `self_idx`).
fn compute_gap(
    map: &ActMap,
    candidate_col: i32,
    row_nodes: &[MapPointIdx],
    self_idx: MapPointIdx,
) -> i32 {
    let mut min_d = i32::MAX;
    for &idx in row_nodes {
        if idx == self_idx {
            continue;
        }
        let d = (candidate_col - map.point(idx).col()).abs();
        if d < min_d {
            min_d = d;
        }
    }
    min_d
}

/// Port of C# `MapPostProcessing.StraightenPaths`
/// (v0.103.2 MapPostProcessing.cs:71-112). For each node with exactly
/// one parent + one child, if both are to the same side, nudge the
/// node one column toward them when the slot is free. RNG-free.
fn straighten_paths(map: &mut ActMap) {
    let row_count = map.map_length + 1;
    for row in 0..row_count {
        let row_nodes: Vec<MapPointIdx> = map
            .grid
            .iter()
            .filter_map(|(c, &i)| if c.row == row { Some(i) } else { None })
            .collect();
        for idx in row_nodes {
            let p = map.point(idx);
            if p.parents.len() != 1 || p.children.len() != 1 {
                continue;
            }
            let par = p.parents[0];
            let ch = p.children[0];
            let col = p.col();
            let par_col = map.point(par).col();
            let ch_col = map.point(ch).col();
            let go_right = col < ch_col && col < par_col;
            let go_left = col > ch_col && col > par_col;
            if go_right && col < MAP_WIDTH - 1 {
                let target = col + 1;
                if !map.grid.contains_key(&MapCoord::new(target, row)) {
                    map.grid.remove(&MapCoord::new(col, row));
                    let new_coord = MapCoord::new(target, row);
                    map.point_mut(idx).coord = new_coord;
                    map.grid.insert(new_coord, idx);
                }
            } else if go_left && col > 0 {
                let target = col - 1;
                if !map.grid.contains_key(&MapCoord::new(target, row)) {
                    map.grid.remove(&MapCoord::new(col, row));
                    let new_coord = MapCoord::new(target, row);
                    map.point_mut(idx).coord = new_coord;
                    map.grid.insert(new_coord, idx);
                }
            }
        }
    }
}

/// Generate a complete standard act map.
///
/// 1. Allocate `ActMap::new(num_rooms)`.
/// 2. Pin entrance at `(3, 0)` and boss at `(3, map_length)`.
/// 3. Spawn 7 random paths from row 1 to the top room row; the
///    second path is guaranteed to start at a different column from
///    the first (≤100 attempts).
/// 4. Connect every top-row room → boss and entrance → every row-1
///    room.
/// 5. Roll pool counts via [`map_point_type_counts`] (override elites
///    if `num_elites_override` is `Some`).
/// 6. Run [`assign_room_types`].
/// 7. [`center_grid`].
///
/// Port pin: `generate_act_map` in `generator.py:562-634`.
#[must_use]
pub fn generate_act_map(
    num_rooms: i32,
    rng: &mut Rng,
    ascension_level: i32,
    replace_treasure_with_elite: bool,
    act_index: i32,
    num_elites_override: Option<i32>,
) -> ActMap {
    generate_act_map_for(
        crate::GameVersion::V0_107_1,
        num_rooms,
        rng,
        ascension_level,
        replace_treasure_with_elite,
        act_index,
        num_elites_override,
    )
}

/// [`generate_act_map`] for a given game build (v0.113.0 changed point-type
/// placement and the post-prune repair; see [`get_next_valid_type_from_queue`]
/// and [`crate::map_path_pruning::prune_and_repair`]).
#[must_use]
pub fn generate_act_map_for(
    version: crate::GameVersion,
    num_rooms: i32,
    rng: &mut Rng,
    ascension_level: i32,
    replace_treasure_with_elite: bool,
    act_index: i32,
    num_elites_override: Option<i32>,
) -> ActMap {
    let mut map = ActMap::new(num_rooms);
    let start = map.get_or_create(3, 0);
    map.start = Some(start);
    let boss = map.get_or_create(3, map.map_length);
    map.boss = Some(boss);

    // C# `StandardActMap` constructor (StandardActMap.cs:70-92) rolls
    // `_pointTypeCounts = actModel.GetMapPointTypes(mapRng)` BEFORE
    // calling `GenerateMap()`. Each gaussian roll consumes 2 underlying
    // `NextDouble` samples (Box-Muller), so doing this AFTER PathGenerate
    // — as the Python pin's `generate_act_map` does — shifts every
    // subsequent RNG draw by 2 doubles relative to C#. The Python sim's
    // order was wrong; this is the fix that aligns Rust's start-column
    // sequence with the actual C# game (bead `sts2-rs-5hsqp`).
    let mut counts = map_point_type_counts(rng, ascension_level, act_index);
    if let Some(override_val) = num_elites_override {
        counts.num_elites = override_val;
    }

    let mut start_points: Vec<MapPointIdx> = Vec::with_capacity(MAP_ITERATIONS);
    for i in 0..MAP_ITERATIONS {
        let mut start_col = rng.next_int(0, MAP_WIDTH - 1);
        let mut path_start = map.get_or_create(start_col, 1);
        if i == 1 {
            let mut attempts = 0;
            while start_points.contains(&path_start) && attempts < 100 {
                start_col = rng.next_int(0, MAP_WIDTH - 1);
                path_start = map.get_or_create(start_col, 1);
                attempts += 1;
            }
        }
        start_points.push(path_start);
        path_generate(&mut map, path_start, rng, version);
    }

    // Connect top row → boss.
    let top_row_coords: Vec<MapCoord> = map
        .get_row(map.map_length - 1)
        .into_iter()
        .map(|p| p.coord)
        .collect();
    for c in top_row_coords {
        let idx = map.grid[&c];
        map.add_child(idx, boss);
    }
    // Connect entrance → row 1.
    let row1_coords: Vec<MapCoord> = map.get_row(1).into_iter().map(|p| p.coord).collect();
    for c in row1_coords {
        let idx = map.grid[&c];
        map.add_child(start, idx);
    }

    assign_room_types(&mut map, rng, counts, replace_treasure_with_elite, version);
    // Bead `sts2-rs-l8hcp` — port of C# `MapPathPruning.PruneAndRepair`
    // (`StandardActMap.cs:87` runs it BEFORE `CenterGrid`). Removes
    // duplicate path segments + refills type counts from Monster
    // fallbacks. The Python sim pin `01c094d` had no pruning at all,
    // so this step is C#-only.
    if std::env::var_os("STS2_DISABLE_PRUNING").is_none() {
        crate::map_path_pruning::prune_and_repair(&mut map, rng, counts, version);
    }
    // C# `StandardActMap` constructor (StandardActMap.cs:89-91) runs
    // these three post-processing passes in order. CenterGrid +
    // SpreadAdjacentMapPoints + StraightenPaths all mutate node columns
    // (no RNG); skipping the latter two left Rust maps with nodes in
    // pre-spread positions, which prevented scraped-corpus parity for
    // any seed whose nodes got rearranged by Spread/Straighten.
    // Bead `sts2-rs-d9fwi` (port).
    if std::env::var_os("STS2_DISABLE_POSTPROC").is_none() {
        center_grid(&mut map);
        if std::env::var_os("STS2_DISABLE_SPREAD").is_none() {
            spread_adjacent_map_points(&mut map);
        }
        if std::env::var_os("STS2_DISABLE_STRAIGHTEN").is_none() {
            straighten_paths(&mut map);
        }
    }
    map
}

// -----------------------------------------------------------------------------
// Spoils variant — bonus-act map centered on a single treasure node.
// -----------------------------------------------------------------------------

/// Columns reachable on `row` for the spoils variant. Spread shrinks
/// as we approach `treasure_row` (which is always at the center) and
/// then re-expands going up. Mirrors `_spoils_allowed_columns` in
/// `generator.py:347-353`.
fn spoils_allowed_columns(row: i32, treasure_row: i32, map_length: i32) -> (i32, i32) {
    let center = MAP_WIDTH / 2;
    let distance_to_treasure = (row - treasure_row).abs();
    let rows_to_top = map_length - 1 - row;
    let top_width = center.min(rows_to_top.max(0) + 1);
    let spread = center.min(distance_to_treasure.min(top_width));
    (
        (center - spread).max(0),
        (center + spread).min(MAP_WIDTH - 1),
    )
}

/// Direction priority for the spoils path when within ≤3 rows of the
/// treasure: bias toward the center column. Returns the four-element
/// direction-attempt vector. Mirrors `_spoils_centered_directions` in
/// `generator.py:356-373`.
fn spoils_centered_directions(current_col: i32) -> Vec<i32> {
    let center = MAP_WIDTH / 2;
    let direction = match center.cmp(&current_col) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    };
    let mut directions: Vec<i32> = Vec::new();
    if direction != 0 {
        directions.push(direction);
    }
    directions.push(0);
    if direction != 0 {
        directions.push(-direction);
    }
    if !directions.contains(&-1) {
        directions.push(-1);
    }
    if !directions.contains(&1) {
        directions.push(1);
    }
    directions
}

fn spoils_next_column(current_col: i32, direction: i32) -> i32 {
    match direction.cmp(&0) {
        std::cmp::Ordering::Less => (current_col - 1).max(0),
        std::cmp::Ordering::Greater => (current_col + 1).min(MAP_WIDTH - 1),
        std::cmp::Ordering::Equal => current_col,
    }
}

fn spoils_generate_next_coord(
    map: &ActMap,
    current_idx: MapPointIdx,
    rng: &mut Rng,
    treasure_row: i32,
    version: crate::GameVersion,
) -> (i32, i32) {
    let current = map.point(current_idx).clone();
    let row = current.row() + 1;
    let (min_col, max_col) = spoils_allowed_columns(row, treasure_row, map.map_length);
    let distance_to_treasure = treasure_row - current.row();
    let mut directions: Vec<i32> = if distance_to_treasure > 3 {
        let mut d = vec![-1, 0, 1];
        rng.shuffle_for(version, &mut d);
        d
    } else if distance_to_treasure > 0 {
        spoils_centered_directions(current.col())
    } else {
        let mut d = vec![-1, 0, 1];
        rng.shuffle_for(version, &mut d);
        d
    };
    // Remove the original .into_iter() shadow with an immutable view.
    while let Some(direction) = directions.pop_front_or_none() {
        let next_col = spoils_next_column(current.col(), direction);
        if next_col < min_col || next_col > max_col {
            continue;
        }
        if has_invalid_crossover(map, current_idx, next_col) {
            continue;
        }
        let point_at_target = map.grid.get(&MapCoord::new(next_col, row)).copied();
        let parent_ok = match point_at_target {
            Some(pt_idx) => {
                let pt = map.point(pt_idx);
                pt.parents.contains(&current_idx) || pt.parents.len() < 3
            }
            None => true,
        };
        let child_ok = if Some(current_idx) == map.start || current.children.len() < 3 {
            true
        } else {
            match point_at_target {
                Some(pt_idx) => current.children.contains(&pt_idx),
                None => false,
            }
        };
        if parent_ok && child_ok {
            return (next_col, row);
        }
    }
    // Fallback path mirroring the Python source.
    let center = MAP_WIDTH / 2;
    let mut fallback = center.clamp(min_col, max_col);
    if (fallback - current.col()).abs() > 1 {
        let step = if fallback > current.col() { 1 } else { -1 };
        fallback = (current.col() + step).clamp(min_col, max_col);
    }
    if has_invalid_crossover(map, current_idx, fallback) {
        fallback = current.col().clamp(min_col, max_col);
    }
    (fallback, row)
}

/// Tiny extension trait so the spoils-next-coord loop can pop from
/// the front of a `Vec`. We avoid `VecDeque` here to keep allocation
/// patterns identical to Python's transient `list`.
trait PopFrontVec<T> {
    fn pop_front_or_none(&mut self) -> Option<T>;
}

impl<T> PopFrontVec<T> for Vec<T> {
    fn pop_front_or_none(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(0))
        }
    }
}

fn spoils_path_generate(
    map: &mut ActMap,
    start_idx: MapPointIdx,
    rng: &mut Rng,
    treasure_row: i32,
    version: crate::GameVersion,
) {
    let mut current = start_idx;
    while map.point(current).row() < map.map_length - 1 {
        let (col, row) = spoils_generate_next_coord(map, current, rng, treasure_row, version);
        let next_idx = map.get_or_create(col, row);
        map.add_child(current, next_idx);
        current = next_idx;
    }
}

fn redirect_to_spoils_treasure(
    map: &mut ActMap,
    stray_idx: MapPointIdx,
    treasure_idx: MapPointIdx,
) {
    let parents: Vec<MapPointIdx> = map.point(stray_idx).parents.clone();
    for parent in parents {
        // remove_child: detach stray from parent.children + parent from stray.parents
        map.point_mut(parent).children.retain(|c| *c != stray_idx);
        map.point_mut(stray_idx).parents.retain(|p| *p != parent);
        map.add_child(parent, treasure_idx);
    }
    let children: Vec<MapPointIdx> = map.point(stray_idx).children.clone();
    for child in children {
        map.point_mut(stray_idx).children.retain(|c| *c != child);
        map.point_mut(child).parents.retain(|p| *p != stray_idx);
        map.add_child(treasure_idx, child);
    }
    let coord = map.point(stray_idx).coord;
    map.grid.remove(&coord);
}

fn is_spoils_type_valid(
    map: &ActMap,
    point_idx: MapPointIdx,
    candidate: MapPointType,
    map_length: i32,
    version: crate::GameVersion,
) -> bool {
    // Same constraints as standard, with a slightly different sibling
    // check shape (still parents-of-parents). Reuse the standard impl
    // for clarity — the lists differ only in being literal `set`s in
    // Python vs slices here.
    is_type_valid_for_point_v(map, point_idx, candidate, map_length, version)
}

fn assign_spoils_room_types(
    map: &mut ActMap,
    rng: &mut Rng,
    counts: PoolCounts,
    version: crate::GameVersion,
) {
    let map_length = map.map_length;

    let top_room_coords: Vec<MapCoord> = map
        .get_row(map_length - 1)
        .into_iter()
        .map(|p| p.coord)
        .collect();
    for c in top_room_coords {
        let idx = map.grid[&c];
        map.point_mut(idx).point_type = MapPointType::RestSite;
    }
    let row1_coords: Vec<MapCoord> = map.get_row(1).into_iter().map(|p| p.coord).collect();
    for c in row1_coords {
        let idx = map.grid[&c];
        if map.point(idx).point_type == MapPointType::Unassigned {
            map.point_mut(idx).point_type = MapPointType::Monster;
        }
    }

    let mut pool: std::collections::VecDeque<MapPointType> = std::collections::VecDeque::new();
    for _ in 0..counts.num_rests {
        pool.push_back(MapPointType::RestSite);
    }
    for _ in 0..counts.num_shops {
        pool.push_back(MapPointType::Shop);
    }
    for _ in 0..counts.num_elites {
        pool.push_back(MapPointType::Elite);
    }
    for _ in 0..counts.num_unknowns {
        pool.push_back(MapPointType::Unknown);
    }

    // Three-iteration pool drain — mirrors C# `SpoilsActMap.
    // AssignRemainingTypesToRandomPoints` (`SpoilsActMap.cs`), which loops
    // `for (int i = 0; i < 3; i++)`, each pass re-collecting the *Unassigned*
    // non-meta points, coord-sorted `StableShuffle`, then a queue-cycle drain.
    //
    // The previous port did a SINGLE pass over ALL non-meta points in
    // insertion order: it shuffled the wrong set (every point, not just the
    // unassigned ones) in the wrong order (insertion vs (col,row)-sorted) and
    // ran only one pass — so it both consumed a different number of `NextInt`
    // draws and produced a different permutation, fully desyncing spoils maps.
    // This is the same fix bead `sts2-rs-genyg` applied to the standard variant
    // (`assign_room_types`); the spoils variant was missed.
    let start = map.start;
    let boss = map.boss;
    for _ in 0..3 {
        if pool.is_empty() {
            break;
        }
        // C# collects `Unassigned` points (excl. boss/start) via
        // `GetAllMapPoints()` then `StableShuffle` (sort by `MapPoint.CompareTo`
        // = `MapCoord.CompareTo` = `(col,row)`, then `UnstableShuffle`). Sort
        // by coord first so the Fisher-Yates input is canonical regardless of
        // grid enumeration order.
        let mut unassigned: Vec<(MapCoord, MapPointIdx)> = map
            .grid
            .iter()
            .filter(|(_, &i)| Some(i) != start && Some(i) != boss)
            .filter(|(_, &i)| map.point(i).point_type == MapPointType::Unassigned)
            .map(|(c, &i)| (*c, i))
            .collect();
        unassigned.sort_by_key(|(c, _)| (c.col, c.row));
        rng.shuffle_for(version, &mut unassigned);
        for (_coord, point_idx) in &unassigned {
            if pool.is_empty() {
                break;
            }
            if version >= crate::GameVersion::V0_113_0 {
                // Order-preserving queue pick (`SpoilsActMap.cs:425-446`).
                if let Some(pos) = pool
                    .iter()
                    .position(|&t| is_spoils_type_valid(map, *point_idx, t, map_length, version))
                {
                    let candidate = pool.remove(pos).expect("position is in range");
                    map.point_mut(*point_idx).point_type = candidate;
                }
                continue;
            }
            let len = pool.len();
            for _ in 0..len {
                let candidate = pool.pop_front().expect("len > 0");
                if is_spoils_type_valid(map, *point_idx, candidate, map_length, version) {
                    map.point_mut(*point_idx).point_type = candidate;
                    break;
                }
                pool.push_back(candidate);
            }
        }
    }

    let leftover_coords: Vec<MapCoord> = map
        .grid
        .iter()
        .filter(|(_, &i)| Some(i) != start && Some(i) != boss)
        .map(|(c, _)| *c)
        .collect();
    for c in leftover_coords {
        let idx = map.grid[&c];
        if map.point(idx).point_type == MapPointType::Unassigned {
            map.point_mut(idx).point_type = MapPointType::Monster;
        }
    }
    if let Some(boss_idx) = map.boss {
        map.point_mut(boss_idx).point_type = MapPointType::Boss;
    }
    if let Some(start_idx) = map.start {
        map.point_mut(start_idx).point_type = MapPointType::Ancient;
    }
}

/// Generate a spoils-variant act map (bonus act). The treasure node
/// sits at the grid center on row `map_length - 7`; all stray points
/// on that row are redirected onto it.
///
/// Port pin: `generate_spoils_act_map` in `generator.py:513-559`.
///
/// # Panics
///
/// Panics if `num_rooms < 7` (treasure row would be ≤0).
#[must_use]
pub fn generate_spoils_act_map(
    num_rooms: i32,
    rng: &mut Rng,
    ascension_level: i32,
    act_index: i32,
) -> ActMap {
    generate_spoils_act_map_for(
        crate::GameVersion::V0_107_1,
        num_rooms,
        rng,
        ascension_level,
        act_index,
    )
}

/// [`generate_spoils_act_map`] for a given game build.
///
/// # Panics
///
/// Panics if `num_rooms < 7` (treasure row would be ≤0).
#[must_use]
pub fn generate_spoils_act_map_for(
    version: crate::GameVersion,
    num_rooms: i32,
    rng: &mut Rng,
    ascension_level: i32,
    act_index: i32,
) -> ActMap {
    let counts = map_point_type_counts(rng, ascension_level, act_index);
    let mut map = ActMap::new(num_rooms);
    let center = MAP_WIDTH / 2;
    let start_idx = map.get_or_create(center, 0);
    map.start = Some(start_idx);
    let boss_idx = map.get_or_create(center, map.map_length);
    map.boss = Some(boss_idx);
    let treasure_row = map.map_length - 7;
    assert!(
        treasure_row > 0 && treasure_row < map.map_length,
        "treasure row {treasure_row} out of bounds for spoils map (map_length={})",
        map.map_length
    );

    let mut start_points: Vec<MapPointIdx> = Vec::with_capacity(MAP_ITERATIONS);
    for i in 0..MAP_ITERATIONS {
        let mut s = map.get_or_create(rng.next_int(0, MAP_WIDTH - 1), 1);
        if i == 1 {
            while start_points.contains(&s) {
                s = map.get_or_create(rng.next_int(0, MAP_WIDTH - 1), 1);
            }
        }
        start_points.push(s);
        spoils_path_generate(&mut map, s, rng, treasure_row, version);
    }

    let treasure_idx = map.get_or_create(center, treasure_row);
    map.point_mut(treasure_idx).point_type = MapPointType::Treasure;
    let treasure_row_coords: Vec<MapCoord> = map
        .get_row(treasure_row)
        .into_iter()
        .map(|p| p.coord)
        .collect();
    for c in treasure_row_coords {
        let idx = map.grid[&c];
        if idx != treasure_idx {
            redirect_to_spoils_treasure(&mut map, idx, treasure_idx);
        }
    }

    // Connect top row → boss; row 1 → start.
    let top_row_coords: Vec<MapCoord> = map
        .get_row(map.map_length - 1)
        .into_iter()
        .map(|p| p.coord)
        .collect();
    for c in top_row_coords {
        let idx = map.grid[&c];
        map.add_child(idx, boss_idx);
    }
    let row1_coords: Vec<MapCoord> = map.get_row(1).into_iter().map(|p| p.coord).collect();
    for c in row1_coords {
        let idx = map.grid[&c];
        map.add_child(start_idx, idx);
    }

    assign_spoils_room_types(&mut map, rng, counts, version);
    // C# `SpoilsActMap` ctor (`SpoilsActMap.cs:77`) runs
    // `MapPathPruning.PruneAndRepair(...)` after `AssignPointTypes()` — the
    // SAME variant-agnostic pass the standard map uses. The prior port skipped
    // it for spoils, leaving duplicate path segments uncollapsed, the pool-type
    // deficit unrepaired, and the pruning shuffles never advancing the RNG.
    // (Spoils correctly omits CenterGrid/Spread/Straighten, so none is added.)
    crate::map_path_pruning::prune_and_repair(&mut map, rng, counts, version);
    map
}

// -----------------------------------------------------------------------------
// Unit tests — structural invariants only (no RNG-byte parity).
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn count_with_type(map: &ActMap, t: MapPointType) -> usize {
        map.points.iter().filter(|p| p.point_type == t).count()
    }

    /// GH #544 listed "`Underdocks.GetMapPointTypes` differs from
    /// Overgrowth's" as work to do. At the v0.107.1 pin it does not — the two
    /// overrides are identical, so act 1's rest/unknown counts are the same
    /// whichever variant the run rolled and this function rightly takes no
    /// variant argument.
    ///
    /// The test guards the *reason*, not just the current numbers: it re-runs
    /// the act-0 branch against the constants the decomp gives for BOTH
    /// classes. If a future pin makes them diverge, the act-1 branch has to
    /// split — and this is where that shows up, instead of as a silent
    /// map-parity drift on half of all runs.
    #[test]
    fn act1_variants_share_one_map_point_type_roll() {
        // `NextGaussianInt(7, 1, 6, 7)` + `StandardRandomUnknownCount` —
        // verbatim from Overgrowth.cs:127-129 AND Underdocks.cs:112-114.
        for seed in [1, 7, 12345, -99] {
            for asc in [0, 1, 10] {
                let mut expected_rng = crate::dotnet_rng::Rng::with_name(seed, "map");
                let rests = expected_rng.next_gaussian_int(7.0, 1.0, 6, 7);
                let unknowns = expected_rng.next_gaussian_int(12.0, 1.0, 10, 14);

                let mut rng = crate::dotnet_rng::Rng::with_name(seed, "map");
                let got = map_point_type_counts(&mut rng, asc, 0);
                assert_eq!((got.num_rests, got.num_unknowns), (rests, unknowns));
                assert_eq!(
                    rng.counter(),
                    expected_rng.counter(),
                    "act 1 must consume exactly the two gaussian rolls"
                );
            }
        }
    }

    #[test]
    fn golden_path_has_expected_shape() {
        let m = generate_golden_path_map(1);
        assert_eq!(m.map_length, 17, "16 rooms + boss row 17");
        assert_eq!(m.points.len(), 18, "start + 16 rooms + boss");
        // Start is Ancient; boss is Boss; chain goes through every entry.
        assert_eq!(
            m.point(m.start.expect("start set")).point_type,
            MapPointType::Ancient
        );
        assert_eq!(
            m.point(m.boss.expect("boss set")).point_type,
            MapPointType::Boss
        );
        // Single spine: every non-meta node has exactly one parent and
        // one child.
        for &idx in m.grid.values() {
            if Some(idx) == m.start || Some(idx) == m.boss {
                continue;
            }
            assert_eq!(m.point(idx).parents.len(), 1, "spine: one parent each");
            assert_eq!(m.point(idx).children.len(), 1, "spine: one child each");
        }
    }

    #[test]
    fn standard_act_map_smoke() {
        let mut rng = Rng::new(2026);
        let m = generate_act_map(15, &mut rng, 0, false, 0, None);
        assert_eq!(m.map_length, 16);
        assert!(m.start.is_some());
        assert!(m.boss.is_some());
        assert!(
            m.points.len() > 16,
            "expected at least one node per row plus extras"
        );
        // Boss is reachable: there must be at least one path from
        // start (row 0) to boss (row 16) — but rather than tracing
        // the whole graph, sanity-check that the top room row has at
        // least one node-in-grid and connects to boss.
        //
        // `get_row` walks `m.points` directly, which still contains
        // stale entries for nodes that pruning has detached (the
        // `remove_point` path clears grid + edges but leaves the
        // points Vec untouched). Filter to live grid entries before
        // asserting child-edges.
        let boss_idx = m.boss.expect("boss");
        let map_length = m.map_length;
        // Iterate live grid entries at row = top, not the points Vec
        // (which retains pruned ghosts whose coord may collide with
        // post-Spread-moved live nodes).
        let top: Vec<MapPointIdx> = m
            .grid
            .iter()
            .filter_map(|(c, &i)| {
                if c.row == map_length - 1 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        assert!(!top.is_empty(), "top room row non-empty");
        for idx in top {
            let p = m.point(idx);
            assert!(
                p.children.contains(&boss_idx),
                "top-row node at col={} should connect to boss",
                p.col()
            );
        }
        let start_idx = m.start.expect("start");
        let bottom: Vec<MapPointIdx> = m
            .grid
            .iter()
            .filter_map(|(c, &i)| if c.row == 1 { Some(i) } else { None })
            .collect();
        assert!(!bottom.is_empty());
        for idx in bottom {
            let p = m.point(idx);
            assert!(p.parents.contains(&start_idx));
        }
    }

    #[test]
    fn replace_treasure_with_elite_swaps_treasure_row() {
        let mut rng = Rng::new(7);
        let m = generate_act_map(15, &mut rng, 0, true, 0, None);
        let treasure_row = m.map_length - 7;
        for p in m.get_row(treasure_row) {
            assert_ne!(
                p.point_type,
                MapPointType::Treasure,
                "treasure replaced with elite"
            );
            // It's actually Elite (modulo the room-type pass which
            // may overwrite). Treasure should not appear at all on
            // this row.
        }
        // Treasure shouldn't appear globally (the only fixed treasure
        // source is the treasure row, which we replaced).
        assert_eq!(
            count_with_type(&m, MapPointType::Treasure),
            0,
            "no treasure when replace_treasure_with_elite=true"
        );
    }

    #[test]
    fn act_2_count_path_runs_without_panic() {
        // Hive branch: gauss(6,1,6,7) rests + gauss(12,1,10,14)-1 unknowns.
        let mut rng = Rng::new(99);
        let m = generate_act_map(15, &mut rng, 0, false, 2, None);
        assert!(m.points.len() > 16);
    }

    #[test]
    fn act_3_count_path_runs_without_panic() {
        // Glory branch: next_int_exclusive(5,7) rests + gauss(12,1,10,14)-1 unknowns.
        let mut rng = Rng::new(99);
        let m = generate_act_map(15, &mut rng, 0, false, 3, None);
        assert!(m.points.len() > 16);
    }

    #[test]
    fn act_2_counts_match_hive_pin() {
        // Hive: rests ∈ [6, 7], unknowns ∈ [9, 13] (gauss [10,14] − 1).
        // Sample a handful of seeds and assert the per-call ranges hold.
        for seed in [1i32, 7, 42, 99, 314, 2026] {
            let mut rng = Rng::new(seed);
            let c = map_point_type_counts(&mut rng, 0, 2);
            assert!(
                (6..=7).contains(&c.num_rests),
                "seed {seed}: act-2 rests out of range: {}",
                c.num_rests,
            );
            assert!(
                (9..=13).contains(&c.num_unknowns),
                "seed {seed}: act-2 unknowns out of range: {}",
                c.num_unknowns,
            );
            assert_eq!(c.num_shops, 3);
            assert_eq!(c.num_elites, 5, "asc 0 elites is base default");
        }
    }

    #[test]
    fn act_3_counts_match_glory_pin() {
        // Glory: rests ∈ {5, 6} (NextInt half-open), unknowns ∈ [9, 13].
        for seed in [1i32, 7, 42, 99, 314, 2026] {
            let mut rng = Rng::new(seed);
            let c = map_point_type_counts(&mut rng, 0, 3);
            assert!(
                (5..=6).contains(&c.num_rests),
                "seed {seed}: act-3 rests out of range: {}",
                c.num_rests,
            );
            assert!(
                (9..=13).contains(&c.num_unknowns),
                "seed {seed}: act-3 unknowns out of range: {}",
                c.num_unknowns,
            );
            assert_eq!(c.num_shops, 3);
            assert_eq!(c.num_elites, 5);
        }
    }

    #[test]
    fn act_2_rng_footprint_is_two_gauss() {
        // Hive takes exactly two next_gaussian_int calls on the main rng.
        // We can't observe the counter directly here, but we can assert
        // that running act_2 then act_2 again on a fresh seed yields the
        // same first-result and a (likely) different second-result, and
        // that the counter advance equals 2 gauss rolls' worth (each
        // gauss = ~10 next_double calls). Use the secondary check: run
        // act_2 once and confirm a parallel rng advanced by 2 explicit
        // gaussians lands in the same state.
        for seed in [1i32, 99, 2026] {
            let mut rng_a = Rng::new(seed);
            let _ = map_point_type_counts(&mut rng_a, 0, 2);
            let mut rng_b = Rng::new(seed);
            let _ = rng_b.next_gaussian_int(6.0, 1.0, 6, 7);
            let _ = rng_b.next_gaussian_int(12.0, 1.0, 10, 14);
            assert_eq!(
                rng_a.next_int(0, 1_000_000),
                rng_b.next_int(0, 1_000_000),
                "seed {seed}: act-2 rng footprint should be 2 gauss rolls",
            );
        }
    }

    #[test]
    fn act_3_rng_footprint_is_one_int_one_gauss() {
        // Glory: one NextInt + one gauss on main rng.
        for seed in [1i32, 99, 2026] {
            let mut rng_a = Rng::new(seed);
            let _ = map_point_type_counts(&mut rng_a, 0, 3);
            let mut rng_b = Rng::new(seed);
            let _ = rng_b.next_int_exclusive(5, 7);
            let _ = rng_b.next_gaussian_int(12.0, 1.0, 10, 14);
            assert_eq!(
                rng_a.next_int(0, 1_000_000),
                rng_b.next_int(0, 1_000_000),
                "seed {seed}: act-3 rng footprint should be 1 NextInt + 1 gauss",
            );
        }
    }

    #[test]
    fn num_elites_override_takes_effect_when_pool_room_allows() {
        let mut rng = Rng::new(5);
        // Override to a small number; verify elites count ≤ override.
        let m = generate_act_map(15, &mut rng, 0, false, 0, Some(2));
        let elites = count_with_type(&m, MapPointType::Elite);
        assert!(
            elites <= 2,
            "override elites=2 should cap actual count, got {elites}"
        );
    }

    #[test]
    fn ascension_swarming_boosts_elite_count() {
        let mut rng_lo = Rng::new(2026);
        let counts_lo = map_point_type_counts(&mut rng_lo, 0, 0);
        let mut rng_hi = Rng::new(2026);
        let counts_hi = map_point_type_counts(&mut rng_hi, 1, 0);
        assert!(
            counts_hi.num_elites > counts_lo.num_elites,
            "swarming (asc ≥1) bumps elites: lo={} hi={}",
            counts_lo.num_elites,
            counts_hi.num_elites,
        );
    }

    #[test]
    fn ascension_a6_does_not_change_rests() {
        // C# AscensionLevel.Inflation (A6) doesn't affect map counts —
        // Python's `has_gloom` was fictional. See bead `sts2-rs-5hsqp`
        // close commit for the fix that lifted A6+ parity 2/6 → 6/6.
        let mut rng_lo = Rng::new(2026);
        let counts_lo = map_point_type_counts(&mut rng_lo, 5, 0);
        let mut rng_hi = Rng::new(2026);
        let counts_hi = map_point_type_counts(&mut rng_hi, 6, 0);
        assert_eq!(counts_lo.num_rests, counts_hi.num_rests);
    }

    #[test]
    fn center_grid_shifts_left_when_only_right_columns_used() {
        let mut m = ActMap::new(5);
        // Place nodes on columns 4, 5, 6 only (right-half empty? no
        // — left half empty, columns 0-1 absent).
        m.get_or_create(4, 1);
        m.get_or_create(5, 2);
        m.get_or_create(6, 3);
        center_grid(&mut m);
        // After shift -1: columns become 3, 4, 5.
        let cols: std::collections::BTreeSet<i32> =
            m.points.iter().map(crate::map::MapPoint::col).collect();
        assert_eq!(
            cols,
            [3, 4, 5].iter().copied().collect(),
            "left-empty triggers shift -1"
        );
    }

    #[test]
    fn center_grid_shifts_right_when_only_left_columns_used() {
        let mut m = ActMap::new(5);
        m.get_or_create(0, 1);
        m.get_or_create(1, 2);
        m.get_or_create(2, 3);
        center_grid(&mut m);
        let cols: std::collections::BTreeSet<i32> =
            m.points.iter().map(crate::map::MapPoint::col).collect();
        assert_eq!(
            cols,
            [1, 2, 3].iter().copied().collect(),
            "right-empty triggers shift +1"
        );
    }

    #[test]
    fn center_grid_noop_when_both_sides_populated() {
        let mut m = ActMap::new(5);
        m.get_or_create(0, 1);
        m.get_or_create(6, 1);
        m.get_or_create(3, 2);
        center_grid(&mut m);
        let cols: std::collections::BTreeSet<i32> =
            m.points.iter().map(crate::map::MapPoint::col).collect();
        assert_eq!(cols, [0, 3, 6].iter().copied().collect(), "no shift");
    }

    #[test]
    fn three_iteration_drain_reduces_monster_fallbacks() {
        // Sanity check for the genyg fix: across a bag of seeds, the
        // mean Monster-room count should not be wildly above what C#
        // v0.103.2 produces. We don't have a C# ground truth here, but
        // we DO know that if the loop were truly running zero extra
        // passes the count would be unchanged — so this test only
        // exists to fail loudly if someone reverts the 3-iter loop
        // back to a single drain without rebaselining downstream
        // expectations. The actual assertion is weak by design.
        let mut monster_counts = Vec::new();
        for seed in [0_i32, 1, 7, 42, 99, 256, 2026, 12345] {
            let mut rng = Rng::new(seed);
            let m = generate_act_map(15, &mut rng, 0, false, 0, None);
            let monsters = m
                .points
                .iter()
                .filter(|p| p.point_type == MapPointType::Monster)
                .count();
            monster_counts.push(monsters);
        }
        let total_monsters: usize = monster_counts.iter().sum();
        // Empirically the 8-seed sum sits in the 200-280 band; a single
        // drain would inflate it markedly higher because more nodes fall
        // through to the Monster fallback. Allow generous headroom.
        assert!(
            total_monsters < 400,
            "Monster fallback count {total_monsters} suspiciously high \
             across 8 seeds — has the 3-iter drain in assign_room_types \
             been reverted? Counts: {monster_counts:?}"
        );
    }

    #[test]
    fn placement_constraints_lower_map_excludes_rest_and_elite() {
        let mut rng = Rng::new(2026);
        let m = generate_act_map(15, &mut rng, 0, false, 0, None);
        for &idx in m.grid.values() {
            let p = m.point(idx);
            // C# `IsValidForLower` uses `row < 6` (StandardActMap.cs:409),
            // not the Python pin's `row < 5`. Switched to C# semantics
            // per bead `sts2-rs-5hsqp`.
            if p.row() < 6 {
                assert_ne!(
                    p.point_type,
                    MapPointType::Elite,
                    "no elite at row {} (lower-map rule)",
                    p.row()
                );
                assert_ne!(
                    p.point_type,
                    MapPointType::RestSite,
                    "no rest at row {} (lower-map rule)",
                    p.row()
                );
            }
        }
    }

    #[test]
    fn spoils_smoke() {
        let mut rng = Rng::new(2026);
        // Spoils needs at least 7 rooms.
        let m = generate_spoils_act_map(12, &mut rng, 0, 1);
        assert!(m.points.len() > 12);
        // Treasure row has exactly one (redirected) treasure node.
        let treasure_row = m.map_length - 7;
        let row = m.get_row(treasure_row);
        assert_eq!(row.len(), 1, "all stray treasure-row nodes redirect to one");
        assert_eq!(row[0].point_type, MapPointType::Treasure);
    }
}
