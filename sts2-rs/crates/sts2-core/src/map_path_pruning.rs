//! Port of C# `MegaCrit.Sts2.Core.Map.MapPathPruning` (v0.103.2,
//! `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Map/MapPathPruning.cs`).
//!
//! The Python sim pin `01c094d` has **no pruning** — this whole
//! subsystem is C#-only. Without it, Rust act-maps include many
//! duplicate path segments that the canonical game would have
//! collapsed; the type-sequence parity harness `r3_map_parity.py`
//! reports 0/N matches against scraped runs until pruning lands.
//!
//! Bead: `sts2-rs-l8hcp` (R3.2 sub-bead).
//!
//! # Algorithm
//!
//! `prune_and_repair` loops up to 3 times:
//!
//! 1. **Prune duplicate segments**: find segments of the path-DAG
//!    that occur ≥2 times (same endpoints + same point-type
//!    sequence), keep the last occurrence and either delete the
//!    interior nodes (`prune_segment`) or break a parent/child edge
//!    (`break_parent_child_relationship`). Wrapped in a 50-iter
//!    safety throw.
//! 2. **Repair pruned point types**: for each of Shop, Elite,
//!    RestSite, Unknown in that order, count current placements; if
//!    below target, take Monster room-points still on the map,
//!    `stable_shuffle`, re-stamp as the target type until the count
//!    matches.
//!
//! Iteration stops early when no repairs happen.
//!
//! # Performance (GH #468)
//!
//! This pass runs once per act in the real game but dominates
//! run-construction-heavy simulator workloads (~45 ms/map ≈ 99.6% of
//! `RunManager::new` at `7981d198b`, where C#'s
//! `FindMatchingSegments` — faithfully re-run from scratch after
//! every single prune — enumerated every root→Boss path and
//! `format!`ed a `String` key per sub-segment). The port therefore
//! deviates from the C# *implementation* — never its observable
//! behavior — in two layers:
//!
//! - **Incremental refresh**: a [`SegmentIndex`] is built once per
//!   `prune_duplicate_segments` call and each of the ≤50 prune
//!   iterations refreshes it in O(affected), which is exact because
//!   pruning only removes graph elements (the equivalence argument
//!   lives on `SegmentIndex`).
//! - **Combinatorial paths**: root→Boss paths are never materialized.
//!   Each distinct segment run is enumerated once (a run *is* a
//!   route), and the paths containing it are rank intervals derived
//!   from per-node DFS arrival bases and subtree counts; retirement
//!   is bitset range-clearing. Segment keys are the same ASCII bytes
//!   the C# key strings held, deduplicated through a deterministic
//!   [`HashIndex`] with C#'s `SortedDictionary` ordinal ordering
//!   applied post-hoc.
//!
//! Combined: ~36× (40.8 → ~1.1 ms/map, `map_gen_bench`). Every
//! shortcut is argued inline at its definition; the rank arithmetic
//! is cross-checked against a brute-force path enumeration in this
//! module's tests, and the multi-seed digest gate
//! (`tests/map_gen_digest.rs` vs `tests/parity/map_gen_digests_v1.txt`,
//! recorded pre-optimization) pins byte-identical maps + RNG draw
//! counts across 2520 seed×config combinations.

use crate::dotnet_rng::Rng;
use crate::map::{ActMap, MapPointIdx, MapPointType};
use crate::map_generator::{is_type_valid_for_point_v, PoolCounts};

fn trace_enabled() -> bool {
    std::env::var_os("MAP_PRUNE_TRACE").is_some()
}

fn coord(map: &ActMap, idx: MapPointIdx) -> String {
    let p = map.point(idx);
    format!("({},{})", p.col(), p.row())
}

/// Entry point — mirrors C# `MapPathPruning.PruneAndRepair`
/// (`MapPathPruning.cs:13-23`). Runs up to three passes of
/// `prune_duplicate_segments` + `repair_pruned_point_types`; stops
/// early when a repair pass returns `false` (nothing to fix).
pub fn prune_and_repair(
    map: &mut ActMap,
    rng: &mut Rng,
    counts: PoolCounts,
    version: crate::GameVersion,
) {
    let map_length = map.map_length;
    for _ in 0..3 {
        prune_duplicate_segments(map, rng, version);
        if !repair_pruned_point_types(map, rng, counts, map_length, version) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 2 — Point-type repair after pruning.
// -----------------------------------------------------------------------------

/// Re-stamp Monster room-points to fill any pool-type deficits.
/// Mirrors C# `RepairPrunedPointTypes` (`MapPathPruning.cs:25-32`).
/// Returns `true` iff any change was made (used by `prune_and_repair`
/// to short-circuit the outer 3-iter loop).
fn repair_pruned_point_types(
    map: &mut ActMap,
    rng: &mut Rng,
    counts: PoolCounts,
    map_length: i32,
    version: crate::GameVersion,
) -> bool {
    let mut changed = false;
    // Order matters — matches the C# call sequence exactly.
    changed |= repair_point_type(
        map,
        MapPointType::Shop,
        counts.num_shops,
        rng,
        map_length,
        version,
    );
    changed |= repair_point_type(
        map,
        MapPointType::Elite,
        counts.num_elites,
        rng,
        map_length,
        version,
    );
    changed |= repair_point_type(
        map,
        MapPointType::RestSite,
        counts.num_rests,
        rng,
        map_length,
        version,
    );
    changed |= repair_point_type(
        map,
        MapPointType::Unknown,
        counts.num_unknowns,
        rng,
        map_length,
        version,
    );
    changed
}

/// Repair a single pool type. If `target_count - current_count > 0`,
/// `stable_shuffle` live Monster points and re-stamp them as `kind`
/// until the deficit is closed or no valid candidate remains. Mirrors
/// `RepairPointType` (`MapPathPruning.cs:34-61`).
fn repair_point_type(
    map: &mut ActMap,
    kind: MapPointType,
    target_count: i32,
    rng: &mut Rng,
    map_length: i32,
    version: crate::GameVersion,
) -> bool {
    // v0.113.0 counts only modifiable points (`MapPathPruning.cs:54`
    // `p.PointType == type && p.CanBeModified`): the pinned top-row rest
    // sites / treasure-row elites no longer count toward the target.
    let v113 = version >= crate::GameVersion::V0_113_0;
    let current: i32 = map
        .points
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            map.is_in_map(*i as MapPointIdx)
                && p.point_type == kind
                && !(v113 && is_pinned_row(p.row(), map_length))
        })
        .count() as i32;
    let mut deficit = target_count - current;
    if trace_enabled() {
        eprintln!("[REP] {kind:?} current={current} target={target_count} deficit={deficit}");
    }
    if deficit <= 0 {
        return false;
    }
    // Eligible candidates: Monster nodes still in the map. C# uses
    // `p.CanBeModified`; our row-1/treasure-row/top-row stamps don't
    // carry a CanBeModified flag, so guard those rows explicitly here.
    //
    // C# `MapPathPruning.RepairPointType` (`MapPathPruning.cs:43-46`)
    // collects via `GetAllMapPoints()` then `StableShuffle`-sorts by
    // `MapPoint.CompareTo` (= `MapCoord.CompareTo` = lexicographic
    // `(col, row)`). Collecting as `(coord, idx)` here mirrors that:
    // the sort step makes the order canonical-by-coord before
    // Fisher-Yates, regardless of the order in which `points` was
    // populated by PathGenerate.
    let mut candidates: Vec<(crate::map::MapCoord, MapPointIdx)> = map
        .points
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            map.is_in_map(*i as MapPointIdx)
                && p.point_type == MapPointType::Monster
                && !is_pinned_row(p.row(), map_length)
        })
        .map(|(i, p)| (p.coord, i as MapPointIdx))
        .collect();
    candidates.sort_by_key(|(c, _)| (c.col, c.row));
    rng.shuffle_for(version, &mut candidates);
    // v0.113.0 (`MapPathPruning.cs:65-72`): a non-Unknown deficit may also
    // take a modifiable Unknown point, tried after every Monster point —
    // a second `StableShuffle` on the same rng.
    if v113 && kind != MapPointType::Unknown {
        let mut unknowns: Vec<(crate::map::MapCoord, MapPointIdx)> = map
            .points
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                map.is_in_map(*i as MapPointIdx)
                    && p.point_type == MapPointType::Unknown
                    && !is_pinned_row(p.row(), map_length)
            })
            .map(|(i, p)| (p.coord, i as MapPointIdx))
            .collect();
        unknowns.sort_by_key(|(c, _)| (c.col, c.row));
        rng.shuffle_for(version, &mut unknowns);
        candidates.extend(unknowns);
    }

    let mut changed = false;
    for (coord, idx) in candidates {
        if deficit == 0 {
            break;
        }
        if is_type_valid_for_point_v(map, idx, kind, map_length, version) {
            if trace_enabled() {
                eprintln!("[REP]   ({},{}) <- {:?}", coord.col, coord.row, kind);
            }
            map.point_mut(idx).point_type = kind;
            deficit -= 1;
            changed = true;
        }
    }
    changed
}

/// Row-1 (Monster), treasure row (`map_length - 7`, Treasure/Elite),
/// and the top room row (`map_length - 1`, RestSite) are pinned by
/// `assign_room_types` and must never be re-stamped. Mirrors the C#
/// `CanBeModified = false` flags set at those rows in `StandardActMap`.
const fn is_pinned_row(row: i32, map_length: i32) -> bool {
    row == 1 || row == map_length - 7 || row == map_length - 1
}

// -----------------------------------------------------------------------------
// Phase 1 — Duplicate-segment pruning.
// -----------------------------------------------------------------------------

/// Outer pruning loop. Mirrors `PruneDuplicateSegments`
/// (`MapPathPruning.cs:63-76`): the matching-segments catalogue is
/// refreshed after every successful prune; aborts after 50 iterations
/// (matches the C# guard exactly).
///
/// C# recomputes the catalogue from scratch each iteration. This port
/// builds a [`SegmentIndex`] once and *incrementally* retires paths
/// killed by each prune (GH #468 fix idea 3) — see the `SegmentIndex`
/// docs for the argument that each refresh yields byte-identical
/// buckets, in the same order, to a from-scratch recompute.
fn prune_duplicate_segments(map: &mut ActMap, rng: &mut Rng, version: crate::GameVersion) {
    let Some(start_idx) = map.start else {
        return;
    };
    let mut iters = 0;
    let mut index = SegmentIndex::build(map, start_idx);
    let mut log = PruneLog::default();
    let mut matching = index.current_matching(map);
    if trace_enabled() {
        let sizes: Vec<usize> = matching.iter().map(std::vec::Vec::len).collect();
        eprintln!(
            "[prune] initial matching buckets: {} (sizes: {:?})",
            matching.len(),
            sizes
        );
    }
    while prune_paths(map, rng, &matching, &mut log, version) {
        iters += 1;
        assert!(
            iters <= 50,
            "MapPathPruning: unable to prune matching segments in {iters} iterations \
             (matches C# MapPathPruning.PruneDuplicateSegments throw at 50)"
        );
        index.apply_removals(&log);
        log.clear();
        matching = index.current_matching(map);
        if trace_enabled() {
            let sizes: Vec<usize> = matching.iter().map(std::vec::Vec::len).collect();
            eprintln!(
                "[prune] iter {} matching {} (sizes: {:?}) rng_cnt={}",
                iters,
                matching.len(),
                sizes,
                rng.counter()
            );
        }
    }
}

/// Node / edge removals performed by one round of [`prune_paths`],
/// consumed by [`SegmentIndex::apply_removals`]. A removed node
/// implies the removal of all its incident edges (`ActMap::remove_point`
/// drops them), so those edges are not logged separately — any path
/// through them contains the node itself.
#[derive(Debug, Default)]
struct PruneLog {
    removed_nodes: Vec<MapPointIdx>,
    removed_edges: Vec<(MapPointIdx, MapPointIdx)>,
}

impl PruneLog {
    fn clear(&mut self) {
        self.removed_nodes.clear();
        self.removed_edges.clear();
    }
}

/// A "matching segment" is a sequence of map-points that occurs in
/// ≥2 root→leaf paths with identical (endpoint coords + point-type
/// sequence). We store segments as owned `Vec<MapPointIdx>` chains
/// because they're treated as values in the C# algorithm — the
/// dictionary holds copies of the path arrays, not references.
type Segment = Vec<MapPointIdx>;

/// Dictionary key: the raw ASCII bytes of the C# segment-key string
/// (see [`write_endpoints_prefix`]). `Vec<u8>`'s `Ord` is byte-wise
/// lexicographic — exactly `SortedDictionary<string, …>` with
/// `StringComparer.Ordinal` over ASCII, and exactly what the original
/// `String`-keyed `BTreeMap` did — so prune order (which feeds
/// `map_rng` consumption) is unchanged. Byte keys let
/// [`SegmentIndex::build`]'s candidate loop probe the key index from a
/// reused buffer without allocating per candidate (GH #468: `format!`
/// + `join` churn here was ~99.6% of full-run construction cost).
type SegKey = Vec<u8>;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// One FNV-1a step folding a full 64-bit word. Hash quality only
/// affects probe cost, never behavior — every hash consumer below
/// resolves same-hash entries with exact comparisons.
const fn fnv1a_u64(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(FNV_PRIME)
}

/// Deterministic open-addressing hash index: u64 hash → u32 payload
/// id, linear probing, power-of-two capacity, grown at 50% load.
/// Callers resolve same-hash slots with exact comparisons, so behavior
/// never depends on hash quality or probe order (at most one payload
/// can satisfy the caller's equality check). Deliberately hand-rolled:
/// `sts2-core` bans std `HashMap` for determinism, and the per-
/// candidate probes here are too hot for `BTreeMap` pointer-chasing
/// (callgrind showed the tree probes as the post-opt1 floor).
struct HashIndex {
    /// `(hash, payload)`; payload `u32::MAX` marks an empty slot.
    slots: Vec<(u64, u32)>,
    len: u32,
}

const EMPTY_SLOT: u32 = u32::MAX;

impl HashIndex {
    fn new() -> Self {
        Self {
            slots: vec![(0, EMPTY_SLOT); 64],
            len: 0,
        }
    }

    /// First payload in `hash`'s probe cluster satisfying `eq`.
    fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut i = (hash as usize) & mask;
        loop {
            let (h, p) = self.slots[i];
            if p == EMPTY_SLOT {
                return None;
            }
            if h == hash && eq(p) {
                return Some(p);
            }
            i = (i + 1) & mask;
        }
    }

    /// Insert a payload the caller has verified to be absent.
    fn insert(&mut self, hash: u64, payload: u32) {
        if (self.len as usize + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = (hash as usize) & mask;
        while self.slots[i].1 != EMPTY_SLOT {
            i = (i + 1) & mask;
        }
        self.slots[i] = (hash, payload);
        self.len += 1;
    }

    fn grow(&mut self) {
        let old = std::mem::replace(&mut self.slots, vec![(0, EMPTY_SLOT); 0]);
        self.slots = vec![(0, EMPTY_SLOT); old.len() * 2];
        let mask = self.slots.len() - 1;
        for (h, p) in old {
            if p == EMPTY_SLOT {
                continue;
            }
            let mut i = (h as usize) & mask;
            while self.slots[i].1 != EMPTY_SLOT {
                i = (i + 1) & mask;
            }
            self.slots[i] = (h, p);
        }
    }
}

impl std::fmt::Debug for HashIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashIndex(len={}, cap={})", self.len, self.slots.len())
    }
}

/// One distinct segment run — a node-index chain that occurs in ≥1
/// root→Boss path — stored as a range into [`SegmentIndex::run_arena`].
///
/// The paths containing a run are NOT stored as a list. In DFS path
/// order, every path through one *arrival* of the run's start node is
/// contiguous (rank block `[base, base + cnt(start))`), and within
/// that block the paths that continue along the run's fixed route are
/// the contiguous sub-block `[base + rho_off, base + rho_off +
/// cnt_end)`. So the run's full occurrence set is the ordered disjoint
/// interval union over `bases(start)` — O(1) state per run, resolved
/// against the aliveness bitset on demand.
#[derive(Debug)]
struct RunData {
    /// Range into [`SegmentIndex::run_arena`].
    off: u32,
    len: u32,
    /// Rank offset of the run's route within its start node's arrival
    /// blocks: the sum, along the route, of the path-counts of each
    /// earlier sibling child (see [`SegmentIndex::child_offsets`]).
    rho_off: u32,
    /// `cnt(end)` — number of end→Boss suffixes, i.e. the length of
    /// each occurrence interval.
    cnt_end: u32,
    /// Memoized first surviving occurrence rank (`u32::MAX` once every
    /// occurrence is dead). Aliveness only ever decreases, so this
    /// value is monotonically non-decreasing: a refresh re-verifies it
    /// with one bit test and rescans forward only when it has died.
    first_hint: u32,
}

/// [`RunData::first_hint`] sentinel: no surviving occurrence.
const RUN_DEAD: u32 = u32::MAX;

/// One dictionary key with the distinct runs that map to it (order
/// irrelevant — runs are re-ordered per refresh by first surviving
/// occurrence, which is unique per key).
#[derive(Debug)]
struct KeyData {
    bytes: SegKey,
    runs: Vec<u32>,
}

/// Incremental matching-segments catalogue (GH #468 fix idea 3, with
/// the follow-up combinatorial index replacing materialized paths).
///
/// C# recomputes `FindMatchingSegments` from scratch after every
/// single prune (≤50× per call). This index is built once per
/// [`prune_duplicate_segments`] call and refreshed in O(affected),
/// which is exact — not approximate — because pruning only ever
/// *removes* graph elements:
///
/// - **Paths.** No edge is ever added, so the recomputed path set at
///   any iteration is exactly the subset of build-time paths whose
///   nodes/edges all survive — in the same DFS order
///   (`ActMap::remove_child` uses `retain`, preserving child order, so
///   a from-scratch DFS visits survivors in unchanged relative order).
///   Build-time paths are identified by their DFS rank but never
///   materialized: rank arithmetic replaces them. Each arrival of a
///   node `v` (one distinct root→v route) owns the contiguous rank
///   block `[base, base + cnt(v))`, where `cnt(v)` is the number of
///   v→Boss suffixes and `base` accumulates earlier siblings' counts
///   along the route — precisely the order `FindAllPaths`' recursion
///   emits paths in.
/// - **Retirement.** A path dies iff it contains a removed node or
///   traverses a removed edge. "Paths through node `n`" is the block
///   union over `bases(n)` with length `cnt(n)`; "paths through edge
///   `a→b`" is the union over `bases(a)` of the sub-block at `b`'s
///   build-time child offset with length `cnt(b)`. [`Self::apply_removals`]
///   clears exactly those rank ranges in the aliveness bitset.
/// - **Candidates.** A from-scratch recompute enumerates a run iff
///   some current path contains it AND its head/tail pass
///   `is_valid_segment_start`/`_end` on the current graph. Degrees
///   only decrease, so validity only flips OFF and the build-time run
///   catalogue is a superset of every later candidate set;
///   [`Self::current_matching`] re-checks validity against the live
///   map and drops runs whose occurrence intervals hold no live rank.
///   (A run's every occurrence contains all its route nodes, so
///   removing any of them clears the whole occurrence set — node
///   deaths inside a route need no special handling.) Distinct runs
///   are enumerated once each by construction: a run *is* a route,
///   and the build's route-DFS from each valid start walks each route
///   exactly once — no content hashing, no repeat probes.
/// - **Bucket contents & order.** Within one path, two distinct
///   candidates can never share a key (same key ⇒ same endpoint
///   coords ⇒ same nodes of that path ⇒ same run), so a bucket's
///   insertion order under from-scratch recompute is fully determined
///   by each run's first surviving occurrence rank, and its contents
///   by the same greedy overlap-accept scan C# performs in that
///   order. `current_matching` reproduces exactly that, and emits
///   buckets in ordinal key order (C#'s `SortedDictionary` iteration
///   order), which is stable because the key set only shrinks.
/// - **Types.** Point types are only re-stamped by the *repair* phase,
///   which runs between `prune_duplicate_segments` calls — never
///   inside the ≤50-iteration loop — so keys are stable for the life
///   of one index.
///
/// The multi-seed digest gate (`tests/map_gen_digest.rs`) pins all of
/// this against the pre-optimization from-scratch build.
struct SegmentIndex {
    /// Per map-point: the global rank base of each DFS arrival, in
    /// ascending order. Empty ⇔ the node lies on no root→Boss path.
    bases: Vec<Vec<u32>>,
    /// Per map-point: number of node→Boss suffix paths (`0` = cannot
    /// reach the Boss). Build-time snapshot.
    cnt: Vec<u32>,
    /// Per map-point: build-time `(child, rank offset)` pairs, where
    /// the offset is the sum of earlier siblings' `cnt` in the
    /// build-time child order.
    child_offsets: Vec<Vec<(MapPointIdx, u32)>>,
    /// Distinct runs' node sequences, back-to-back.
    run_arena: Vec<MapPointIdx>,
    runs: Vec<RunData>,
    keys: Vec<KeyData>,
    /// Ids of keys holding ≥2 distinct runs, in ordinal byte order —
    /// the bucket iteration order. Keys with a single distinct run can
    /// never emit a bucket (the run set is fixed at build time and a
    /// bucket needs ≥2 accepted entries), so they are excluded here
    /// once; skipping them preserves the relative ordinal order of the
    /// survivors.
    sorted_key_ids: Vec<u32>,
    /// Aliveness of the `total_paths` build-time paths, one bit per
    /// rank, little-endian within `u64` words.
    alive: Vec<u64>,
    total_paths: u32,
}

impl std::fmt::Debug for SegmentIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SegmentIndex(paths={}, runs={}, keys={})",
            self.total_paths,
            self.runs.len(),
            self.keys.len()
        )
    }
}

/// Route-DFS state for [`SegmentIndex::build`]: walks every distinct
/// route from one valid start node and registers a run at each valid
/// end at depth ≥2 (segment length ≥3, C#'s `j ∈ [2, …)`).
struct RouteWalker<'m> {
    map: &'m ActMap,
    cnt: Vec<u32>,
    child_offsets: Vec<Vec<(MapPointIdx, u32)>>,
    stack: Vec<MapPointIdx>,
    /// `bases[start][0]` for the start currently being explored —
    /// seeds each new run's `first_hint`.
    first_base_of_start: u32,
    key_buf: Vec<u8>,
    key_index: HashIndex,
    run_arena: Vec<MapPointIdx>,
    runs: Vec<RunData>,
    keys: Vec<KeyData>,
}

impl RouteWalker<'_> {
    /// Continue the route currently in `stack` (start .. v) from `v`,
    /// which sits at rank offset `rho_off` within the start node's
    /// arrival blocks.
    fn explore(&mut self, v: MapPointIdx, rho_off: u32) {
        if self.stack.len() >= 3 && is_valid_segment_end(self.map, v) && self.cnt[v as usize] > 0 {
            self.register_run(rho_off, self.cnt[v as usize]);
        }
        // Routes never continue past the Boss — `FindAllPaths` stops
        // there, so no path contains a node after it.
        if self.map.point(v).point_type == MapPointType::Boss {
            return;
        }
        for k in 0..self.child_offsets[v as usize].len() {
            let (c, coff) = self.child_offsets[v as usize][k];
            // A cnt-0 subtree holds no path to the Boss, and every
            // node below a cnt-0 node is itself cnt-0, so no run can
            // ever end below here.
            if self.cnt[c as usize] == 0 {
                continue;
            }
            self.stack.push(c);
            self.explore(c, rho_off + coff);
            self.stack.pop();
        }
    }

    /// Register the run currently in `stack` under its segment key.
    fn register_run(&mut self, rho_off: u32, cnt_end: u32) {
        self.key_buf.clear();
        write_endpoints_prefix(self.map, &self.stack, &mut self.key_buf);
        for (k, &idx) in self.stack.iter().enumerate() {
            if k > 0 {
                self.key_buf.push(b',');
            }
            self.key_buf
                .push(b'0' + csharp_type_code(self.map.point(idx).point_type));
        }
        let key_hash = fnv1a_bytes(&self.key_buf);
        let existing_kid = self
            .key_index
            .find(key_hash, |p| self.keys[p as usize].bytes == self.key_buf);
        let kid = existing_kid.unwrap_or_else(|| {
            let kid = u32::try_from(self.keys.len()).expect("key count fits u32");
            self.key_index.insert(key_hash, kid);
            self.keys.push(KeyData {
                bytes: self.key_buf.clone(),
                runs: Vec::new(),
            });
            kid
        });
        let rid = u32::try_from(self.runs.len()).expect("run count fits u32");
        let off = u32::try_from(self.run_arena.len()).expect("run arena fits u32");
        self.run_arena.extend_from_slice(&self.stack);
        self.runs.push(RunData {
            off,
            len: u32::try_from(self.stack.len()).expect("run length fits u32"),
            rho_off,
            cnt_end,
            // All paths are alive at build, so the first occurrence is
            // the first interval's first rank. `bases` for this start
            // is non-empty (the walk only launches from reachable
            // starts) and `cnt_end ≥ 1`, so this is a real rank.
            first_hint: self.first_base_of_start + rho_off,
        });
        self.keys[kid as usize].runs.push(rid);
    }
}

/// Memoized `cnt(v)` — the number of v→Boss suffix paths, exactly as
/// `FindAllPaths`' recursion counts them: 1 at a Boss node (it stops
/// there regardless of children), the sum over children otherwise
/// (0 at a non-Boss dead end).
fn count_suffixes(map: &ActMap, v: MapPointIdx, memo: &mut [Option<u64>]) -> u64 {
    if let Some(c) = memo[v as usize] {
        return c;
    }
    let point = map.point(v);
    let total = if point.point_type == MapPointType::Boss {
        1
    } else {
        // Snapshot avoids holding the `point` borrow across recursion.
        let children: Vec<MapPointIdx> = point.children.clone();
        children.iter().map(|&c| count_suffixes(map, c, memo)).sum()
    };
    memo[v as usize] = Some(total);
    total
}

/// Record every DFS arrival's rank base. One recursive call per
/// distinct root→v route with a live suffix — the same recursion tree
/// `FindAllPaths` walks, minus dead subtrees.
fn assign_bases(map: &ActMap, v: MapPointIdx, base: u32, cnt: &[u32], bases: &mut [Vec<u32>]) {
    bases[v as usize].push(base);
    if map.point(v).point_type == MapPointType::Boss {
        return;
    }
    let mut off = base;
    let children: Vec<MapPointIdx> = map.point(v).children.clone();
    for c in children {
        let c_cnt = cnt[c as usize];
        if c_cnt > 0 {
            assign_bases(map, c, off, cnt, bases);
        }
        off += c_cnt;
    }
}

impl SegmentIndex {
    fn build(map: &ActMap, start_idx: MapPointIdx) -> Self {
        let n_points = map.points.len();
        // 1. Suffix counts (u64 internally so a pathological braid
        //    fails loudly instead of wrapping).
        let mut memo: Vec<Option<u64>> = vec![None; n_points];
        for v in 0..n_points {
            count_suffixes(
                map,
                u16::try_from(v).expect("point idx fits u16"),
                &mut memo,
            );
        }
        let cnt: Vec<u32> = memo
            .iter()
            .map(|c| u32::try_from(c.expect("memo filled")).expect("path count fits u32"))
            .collect();
        let total_paths = cnt[start_idx as usize];

        // 2. Build-time child order + cumulative rank offsets.
        let child_offsets: Vec<Vec<(MapPointIdx, u32)>> = (0..n_points)
            .map(|v| {
                if map.points[v].point_type == MapPointType::Boss {
                    // Paths never continue past the Boss, so its
                    // outgoing edges carry no rank blocks.
                    return Vec::new();
                }
                let mut off = 0u32;
                map.points[v]
                    .children
                    .iter()
                    .map(|&c| {
                        let pair = (c, off);
                        off += cnt[c as usize];
                        pair
                    })
                    .collect()
            })
            .collect();

        // 3. Arrival bases (ascending per node by DFS order).
        let mut bases: Vec<Vec<u32>> = vec![Vec::new(); n_points];
        if total_paths > 0 {
            assign_bases(map, start_idx, 0, &cnt, &mut bases);
        }

        // 4. Distinct runs via route-DFS from every build-time-valid
        //    start that lies on some path. Enumeration order is
        //    irrelevant to the output (buckets are canonicalized by
        //    ordinal key + first-occurrence rank), but node-index
        //    order keeps it deterministic.
        let mut walker = RouteWalker {
            map,
            cnt,
            child_offsets,
            stack: Vec::with_capacity(32),
            first_base_of_start: 0,
            key_buf: Vec::with_capacity(64),
            key_index: HashIndex::new(),
            run_arena: Vec::new(),
            runs: Vec::new(),
            keys: Vec::new(),
        };
        for w in 0..n_points {
            let w = u16::try_from(w).expect("point idx fits u16");
            if bases[w as usize].is_empty() || !is_valid_segment_start(map, w) {
                continue;
            }
            walker.first_base_of_start = bases[w as usize][0];
            walker.stack.clear();
            walker.stack.push(w);
            walker.explore(w, 0);
        }
        let RouteWalker {
            cnt,
            child_offsets,
            run_arena,
            runs,
            keys,
            ..
        } = walker;

        if std::env::var_os("MAP_PRUNE_STATS").is_some() {
            let arrivals: usize = bases.iter().map(Vec::len).sum();
            eprintln!(
                "[stats] paths={} arrivals={} distinct_runs={} keys={}",
                total_paths,
                arrivals,
                runs.len(),
                keys.len()
            );
        }
        let mut sorted_key_ids: Vec<u32> = (0..u32::try_from(keys.len())
            .expect("key count fits u32"))
            .filter(|&k| keys[k as usize].runs.len() >= 2)
            .collect();
        sorted_key_ids
            .sort_unstable_by(|&a, &b| keys[a as usize].bytes.cmp(&keys[b as usize].bytes));

        // 5. Aliveness bitset over the T build-time path ranks.
        let words = (total_paths as usize).div_ceil(64);
        let mut alive = vec![u64::MAX; words];
        let tail_bits = (total_paths as usize) % 64;
        if tail_bits != 0 {
            if let Some(last) = alive.last_mut() {
                *last = (1u64 << tail_bits) - 1;
            }
        }

        Self {
            bases,
            cnt,
            child_offsets,
            run_arena,
            runs,
            keys,
            sorted_key_ids,
            alive,
            total_paths,
        }
    }

    fn run_slice(&self, rid: u32) -> &[MapPointIdx] {
        let r = &self.runs[rid as usize];
        &self.run_arena[r.off as usize..(r.off + r.len) as usize]
    }

    /// Clear alive bits in `[lo, hi)`.
    fn clear_range(&mut self, lo: u32, hi: u32) {
        let (lo, hi) = (lo as usize, hi as usize);
        let mut w = lo / 64;
        let end_w = hi.div_ceil(64);
        while w < end_w && w < self.alive.len() {
            let word_lo = w * 64;
            let from = lo.max(word_lo) - word_lo;
            let to = hi.min(word_lo + 64) - word_lo;
            let mask = if to - from == 64 {
                u64::MAX
            } else {
                ((1u64 << (to - from)) - 1) << from
            };
            self.alive[w] &= !mask;
            w += 1;
        }
    }

    /// The run's first surviving occurrence rank, memoized in
    /// `first_hint`. Aliveness only decreases, so the true value is
    /// monotonically non-decreasing: a still-alive hint is already the
    /// answer (occurrence intervals are disjoint and ascending, so no
    /// smaller alive occurrence can exist below a previously-minimal
    /// one), and a dead hint means rescanning may skip every interval
    /// that ends at or before it.
    fn first_occurrence(&mut self, rid: u32) -> Option<u32> {
        let run = &self.runs[rid as usize];
        let hint = run.first_hint;
        if hint == RUN_DEAD {
            return None;
        }
        if self.alive[hint as usize / 64] & (1u64 << (hint as usize % 64)) != 0 {
            return Some(hint);
        }
        let (rho_off, cnt_end) = (run.rho_off, run.cnt_end);
        let start = self.run_arena[run.off as usize];
        let first = self.bases[start as usize]
            .iter()
            .filter(|&&base| base + rho_off + cnt_end > hint)
            .find_map(|&base| {
                self.first_alive_in((base + rho_off).max(hint), base + rho_off + cnt_end)
            });
        self.runs[rid as usize].first_hint = first.unwrap_or(RUN_DEAD);
        first
    }

    /// Lowest alive rank in `[lo, hi)`, if any.
    fn first_alive_in(&self, lo: u32, hi: u32) -> Option<u32> {
        let (lo, hi) = (lo as usize, hi as usize);
        let mut w = lo / 64;
        let end_w = hi.div_ceil(64);
        while w < end_w && w < self.alive.len() {
            let word_lo = w * 64;
            let from = lo.max(word_lo) - word_lo;
            let to = hi.min(word_lo + 64) - word_lo;
            let mask = if to - from == 64 {
                u64::MAX
            } else {
                ((1u64 << (to - from)) - 1) << from
            };
            let bits = self.alive[w] & mask;
            if bits != 0 {
                return Some(
                    u32::try_from(word_lo + bits.trailing_zeros() as usize).expect("rank fits u32"),
                );
            }
            w += 1;
        }
        None
    }

    /// Retire every still-alive path invalidated by the logged
    /// removals: paths through a removed node `n` occupy the rank
    /// blocks `[base, base + cnt(n))` over `bases(n)`; paths through a
    /// removed edge `a→b` occupy the sub-blocks at `b`'s build-time
    /// child offset with length `cnt(b)`. Exactly the paths a
    /// from-scratch DFS over the mutated graph would no longer
    /// produce. Re-clearing already-dead ranks is a harmless no-op, so
    /// overlapping removals need no ordering care.
    fn apply_removals(&mut self, log: &PruneLog) {
        for &n in &log.removed_nodes {
            let span = self.cnt[n as usize];
            if span == 0 {
                continue;
            }
            for k in 0..self.bases[n as usize].len() {
                let base = self.bases[n as usize][k];
                self.clear_range(base, base + span);
            }
        }
        for &(a, b) in &log.removed_edges {
            let span = self.cnt[b as usize];
            if span == 0 {
                continue;
            }
            let Some(&(_, coff)) = self.child_offsets[a as usize]
                .iter()
                .find(|&&(c, _)| c == b)
            else {
                continue;
            };
            for k in 0..self.bases[a as usize].len() {
                let base = self.bases[a as usize][k];
                self.clear_range(base + coff, base + coff + span);
            }
        }
    }

    /// The current matching buckets — byte-identical, in the same
    /// order, to what a from-scratch `FindMatchingSegments` over the
    /// mutated map would return (see the type-level docs): per key in
    /// ordinal order, gather the runs that still occur and whose
    /// endpoints are still valid, order them by first surviving
    /// occurrence rank, greedy-accept non-overlapping ones, and keep
    /// buckets with ≥2 entries (`GetDuplicateSegments` filter).
    fn current_matching(&mut self, map: &ActMap) -> Vec<Vec<Segment>> {
        let mut out: Vec<Vec<Segment>> = Vec::new();
        let mut eligible: Vec<(u32, u32)> = Vec::new();
        let mut accepted: Vec<u32> = Vec::new();
        for kid_pos in 0..self.sorted_key_ids.len() {
            let kid = self.sorted_key_ids[kid_pos];
            eligible.clear();
            for run_pos in 0..self.keys[kid as usize].runs.len() {
                let rid = self.keys[kid as usize].runs[run_pos];
                let Some(first_rank) = self.first_occurrence(rid) else {
                    continue;
                };
                let seg = self.run_slice(rid);
                if !is_valid_segment_start(map, seg[0]) {
                    continue;
                }
                if !is_valid_segment_end(map, seg[seg.len() - 1]) {
                    continue;
                }
                eligible.push((first_rank, rid));
            }
            if eligible.len() < 2 {
                continue;
            }
            // First-encounter order. Ranks are unique per key: two
            // same-key runs in one path would be the same run.
            eligible.sort_unstable();
            accepted.clear();
            for &(_, rid) in &eligible {
                let seg = self.run_slice(rid);
                if !accepted
                    .iter()
                    .any(|&a| overlapping_segment(self.run_slice(a), seg))
                {
                    accepted.push(rid);
                }
            }
            if accepted.len() > 1 {
                if trace_enabled() {
                    eprintln!(
                        "[key] '{}' (n={})",
                        String::from_utf8_lossy(&self.keys[kid as usize].bytes),
                        accepted.len()
                    );
                }
                out.push(
                    accepted
                        .iter()
                        .map(|&rid| self.run_slice(rid).to_vec())
                        .collect(),
                );
            }
        }
        out
    }
}

/// FNV-1a over a byte string — key hashing at run-registration scale
/// (a few thousand per build). Behavior never depends on hash quality;
/// see [`HashIndex`].
fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = fnv1a_u64(h, u64::from(b));
    }
    h
}

/// A node is a valid segment-start iff it branches (≥2 children) OR
/// it's the row-0 starting node. Mirrors `IsValidSegmentStartMapPoint`
/// (`MapPathPruning.cs:156-163`).
fn is_valid_segment_start(map: &ActMap, idx: MapPointIdx) -> bool {
    let p = map.point(idx);
    if p.children.len() <= 1 {
        return p.row() == 0;
    }
    true
}

/// A node is a valid segment-end iff ≥2 parents converge on it.
/// Mirrors `IsValidSegmentEndMapPoint` (`MapPathPruning.cs:165-168`).
fn is_valid_segment_end(map: &ActMap, idx: MapPointIdx) -> bool {
    map.point(idx).parents.len() >= 2
}

/// C# `MapPointType` enum order (v0.103.2 MapPointType.cs):
///   Unassigned=0, Unknown=1, Shop=2, Treasure=3, RestSite=4,
///   Monster=5, Elite=6, Boss=7, Ancient=8.
/// Rust enum order differs (Monster=2, Elite=3, Boss=4, Shop=5,
/// RestSite=6, Treasure=7, Ancient=8). Match C# int values here
/// so the byte keys sort identically to C#'s SortedDictionary —
/// different iteration order would change the bucket order in
/// `prune_paths` and divergently affect pruning. Every code is a
/// single decimal digit, which `add_segments_to_dictionary`'s
/// precomputed types buffer relies on.
const fn csharp_type_code(pt: MapPointType) -> u8 {
    match pt {
        MapPointType::Unassigned => 0,
        MapPointType::Unknown => 1,
        MapPointType::Shop => 2,
        MapPointType::Treasure => 3,
        MapPointType::RestSite => 4,
        MapPointType::Monster => 5,
        MapPointType::Elite => 6,
        MapPointType::Boss => 7,
        MapPointType::Ancient => 8,
    }
}

/// Append `v`'s decimal ASCII form to `buf` — byte-identical to what
/// `format!("{v}")` produces, without the fmt machinery.
fn push_i32(buf: &mut Vec<u8>, v: i32) {
    if v < 0 {
        buf.push(b'-');
    }
    // i64 so `abs` can't overflow on i32::MIN.
    let mut n = i64::from(v).unsigned_abs();
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[i..]);
}

/// Write the endpoint part of a segment key — `{head}-{tail.col},{tail.row}-`
/// with the special row-0 head form. Mirrors the first half of
/// `GenerateSegmentKey` (`MapPathPruning.cs:170-205`); the caller
/// appends the comma-joined type-code sequence.
fn write_endpoints_prefix(map: &ActMap, segment: &[MapPointIdx], buf: &mut Vec<u8>) {
    let head = map.point(segment[0]);
    let tail = map.point(segment[segment.len() - 1]);
    if head.row() == 0 {
        // `{head.row}-{tail.col},{tail.row}-`
        push_i32(buf, head.row());
    } else {
        // `{head.col},{head.row}-{tail.col},{tail.row}-`
        push_i32(buf, head.col());
        buf.push(b',');
        push_i32(buf, head.row());
    }
    buf.push(b'-');
    push_i32(buf, tail.col());
    buf.push(b',');
    push_i32(buf, tail.row());
    buf.push(b'-');
}

/// Build the canonical key for a segment as a `String` — endpoint
/// coords followed by the comma-joined integer point-type sequence,
/// mirroring `GenerateSegmentKey` (`MapPathPruning.cs:170-205`).
/// Test-only companion to the buffer-reusing hot path in
/// `add_segments_to_dictionary` (which produces these exact bytes).
#[cfg(test)]
fn generate_segment_key(map: &ActMap, segment: &[MapPointIdx]) -> String {
    let mut buf = Vec::new();
    write_endpoints_prefix(map, segment, &mut buf);
    for (k, &idx) in segment.iter().enumerate() {
        if k > 0 {
            buf.push(b',');
        }
        buf.push(b'0' + csharp_type_code(map.point(idx).point_type));
    }
    String::from_utf8(buf).expect("segment key is pure ASCII")
}

/// Two segments overlap iff they share an *interior* node (positions
/// `1..len-1`); endpoints can match without counting as overlap.
/// Mirrors `OverlappingSegment` (`MapPathPruning.cs:213-227`).
fn overlapping_segment(a: &[MapPointIdx], b: &[MapPointIdx]) -> bool {
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    // C# iterates `i ∈ [1, a.Count - 2]` inclusive.
    for i in 1..=(a.len() - 2) {
        if i >= b.len() {
            break;
        }
        if a[i] == b[i] {
            return true;
        }
    }
    false
}

/// Try to prune one matching-segment bucket per call. Returns
/// `true` iff any node/edge was removed (signals the outer loop to
/// retry). Every removal is recorded in `log` so
/// [`SegmentIndex::apply_removals`] can retire the affected paths.
/// Mirrors `PrunePaths` (`MapPathPruning.cs:234-249`).
fn prune_paths(
    map: &mut ActMap,
    rng: &mut Rng,
    matching: &[Vec<Segment>],
    log: &mut PruneLog,
    version: crate::GameVersion,
) -> bool {
    for bucket in matching {
        // C# `UnstableShuffle` on the bucket. Our `Rng::shuffle` is
        // exactly UnstableShuffle (see `dotnet_rng.rs` doc comment).
        // Shuffling by reference draws the same RNG sequence as
        // shuffling owned clones — consumption depends only on length.
        let mut shuffled: Vec<&Segment> = bucket.iter().collect();
        rng.shuffle_for(version, &mut shuffled);
        if prune_all_but_last(map, &shuffled, log) > 0 {
            return true;
        }
        if break_a_parent_child_relationship_in_any_segment(map, &shuffled, log) {
            return true;
        }
    }
    false
}

/// Delete all but the last matching segment in `matches`. Returns the
/// number of segments that were successfully pruned. C# semantics:
/// the final segment is kept as-is (early-return when index ==
/// matches.Count - 1), prior segments are attempted via
/// `prune_segment`. Mirrors `PruneAllButLast`
/// (`MapPathPruning.cs:251-266`).
fn prune_all_but_last(map: &mut ActMap, matches: &[&Segment], log: &mut PruneLog) -> usize {
    // C# guards on `num` (success count), not on the iteration index.
    // When the first segment in a 2-entry bucket fails to prune, C#
    // still tries the second. The previous index-based guard
    // (`if i == last_idx { return }`) silently skipped the second
    // segment whenever the first failed, which underprunes duplicate
    // buckets and shifts every downstream RNG draw.
    // Mirrors C# `PruneAllButLast` (MapPathPruning.cs:251-266).
    let mut count = 0;
    let target = matches.len().saturating_sub(1);
    for seg in matches {
        if count == target {
            return count;
        }
        if prune_segment(map, seg, log) {
            count += 1;
        }
    }
    count
}

/// Walk the segment's interior nodes and remove any that are "safely
/// removable": single in-degree, single out-degree, and pruning won't
/// orphan a downstream node. Mirrors `PruneSegment`
/// (`MapPathPruning.cs:268-299`).
fn prune_segment(map: &mut ActMap, segment: &[MapPointIdx], log: &mut PruneLog) -> bool {
    let mut changed = false;
    // C# iterates `i ∈ [0, segment.Length - 2]` (skips the final endpoint).
    if segment.len() < 2 {
        return false;
    }
    if trace_enabled() {
        let coords: Vec<String> = segment.iter().map(|&i| coord(map, i)).collect();
        eprintln!("[PS] start segment={}", coords.join(">"));
    }
    for i in 0..segment.len() - 1 {
        let cur = segment[i];
        if !map.is_in_map(cur) {
            if trace_enabled() {
                eprintln!("[PS]   i={i} cur=(removed) early-return");
            }
            return true;
        }
        let p = map.point(cur);
        let children_count = p.children.len();
        let parents_count = p.parents.len();
        if trace_enabled() {
            eprintln!(
                "[PS]   i={} cur={} children={} parents={}",
                i,
                coord(map, cur),
                children_count,
                parents_count
            );
        }
        if children_count > 1 || parents_count > 1 {
            if trace_enabled() {
                eprintln!("[PS]   skip: branching/junction/parent");
            }
            continue;
        }
        // Any parent that has exactly one child and is still on the
        // map — leave alone (we'd orphan it).
        //
        // C# uses `!IsRemoved(grid, n)` here — a STRICT grid-presence
        // check that does NOT exempt Ancient/Boss meta-nodes
        // (MapPathPruning.cs:324-327). Critically, the C# code
        // constructs StartingMapPoint/BossMapPoint as bare MapPoints
        // and does NOT add them to Grid, so `grid[3, 0] == null` for
        // Ancient and `!IsRemoved` is FALSE there. Our Rust port adds
        // Ancient + Boss via `get_or_create` so they DO appear in our
        // grid — we have to short-circuit them here to match C#
        // semantics (treat Ancient/Boss as "removed" for this check).
        let parent_one_child_in_map = p.parents.iter().any(|&pp| {
            let pt = map.point(pp).point_type;
            if matches!(pt, MapPointType::Ancient | MapPointType::Boss) {
                return false;
            }
            map.point(pp).children.len() == 1 && map.grid.get(&map.point(pp).coord) == Some(&pp)
        });
        if parent_one_child_in_map {
            if trace_enabled() {
                eprintln!("[PS]   skip: parent_one_child_in_map");
            }
            continue;
        }
        let rest_has_strict_branch = segment[i..].iter().any(|&n| {
            let pp = map.point(n);
            pp.children.len() > 1 && pp.parents.len() == 1
        });
        if trace_enabled() {
            eprintln!(
                "[PS]   source.Any(strict_branch)={}",
                if rest_has_strict_branch {
                    "True"
                } else {
                    "False"
                }
            );
        }
        if rest_has_strict_branch {
            continue;
        }
        // If the tail endpoint has a single parent, the segment is
        // ineligible — C# `return false` regardless of whether earlier
        // nodes in this segment were removed (the mutation persists but
        // PruneAllButLast doesn't count this as a success). Returning
        // `changed` here was a long-standing Rust bug that prevented
        // pruning from propagating to subsequent buckets in some seeds
        // (bead `sts2-rs-o4mnu`).
        let tail = segment[segment.len() - 1];
        if map.point(tail).parents.len() == 1 {
            if trace_enabled() {
                eprintln!("[PS]   abort: tail parents==1");
            }
            return false;
        }
        // If any external child of `cur` (not in this segment) has
        // exactly one parent, removing `cur` would orphan it. Skip.
        // (Segments are ≤ map_length nodes, so a linear `contains`
        // beats building a set.)
        let any_orphan_risk = map
            .point(cur)
            .children
            .iter()
            .filter(|c| !segment.contains(c))
            .any(|&c| map.point(c).parents.len() == 1);
        if any_orphan_risk {
            if trace_enabled() {
                eprintln!("[PS]   skip: any_orphan_risk");
            }
            continue;
        }
        // Safe to remove.
        if trace_enabled() {
            eprintln!("[PS] removed {}", coord(map, cur));
        }
        map.remove_point(cur);
        log.removed_nodes.push(cur);
        changed = true;
    }
    changed
}

/// If `prune_segment` can't make progress, try to break a single
/// parent→child edge inside one of the segments. Mirrors
/// `BreakAParentChildRelationshipInAnySegment`
/// (`MapPathPruning.cs:329-339`).
fn break_a_parent_child_relationship_in_any_segment(
    map: &mut ActMap,
    matches: &[&Segment],
    log: &mut PruneLog,
) -> bool {
    for seg in matches {
        if break_a_parent_child_relationship_in_segment(map, seg, log) {
            return true;
        }
    }
    false
}

/// Walk consecutive pairs in `segment`; if the parent has ≥2
/// children AND the child has ≥2 parents, drop the parent→child
/// edge. Returns `true` on any successful edge break. Mirrors
/// `BreakAParentChildRelationshipInSegment`
/// (`MapPathPruning.cs:341-358`).
fn break_a_parent_child_relationship_in_segment(
    map: &mut ActMap,
    segment: &[MapPointIdx],
    log: &mut PruneLog,
) -> bool {
    let mut changed = false;
    if segment.len() < 2 {
        return false;
    }
    if trace_enabled() {
        let coords: Vec<String> = segment.iter().map(|&i| coord(map, i)).collect();
        eprintln!("[BR] start segment={}", coords.join(">"));
    }
    for i in 0..segment.len() - 1 {
        let parent = segment[i];
        let pc = map.point(parent).children.len();
        if pc < 2 {
            if trace_enabled() {
                eprintln!(
                    "[BR]   i={} parent={} children={} skip",
                    i,
                    coord(map, parent),
                    pc
                );
            }
            continue;
        }
        let child = segment[i + 1];
        let cp = map.point(child).parents.len();
        if cp != 1 {
            if trace_enabled() {
                eprintln!(
                    "[BR]   i={} parent={} → child={} parents={} REMOVE_EDGE",
                    i,
                    coord(map, parent),
                    coord(map, child),
                    cp
                );
            }
            map.remove_child(parent, child);
            log.removed_edges.push((parent, child));
            changed = true;
        } else if trace_enabled() {
            eprintln!(
                "[BR]   i={} parent={} → child={} parents={} skip(==1)",
                i,
                coord(map, parent),
                coord(map, child),
                cp
            );
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map_generator::generate_act_map;

    #[test]
    fn prune_and_repair_smoke() {
        // Make a real map, run pruning, assert the post-prune graph
        // still has Boss reachable from Start (no segment-pruning bug
        // disconnected the DAG).
        let mut rng = Rng::new(2026);
        let mut map = generate_act_map(15, &mut rng, 0, false, 0, None);
        let counts = PoolCounts {
            num_elites: 5,
            num_shops: 3,
            num_unknowns: 12,
            num_rests: 5,
        };
        let mut rng2 = Rng::new(2026);
        prune_and_repair(&mut map, &mut rng2, counts, crate::GameVersion::V0_107_1);

        // Boss reachable from start.
        let start = map.start.expect("start");
        let boss = map.boss.expect("boss");
        let mut visited: std::collections::BTreeSet<MapPointIdx> =
            std::collections::BTreeSet::new();
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            for &c in &map.point(n).children {
                stack.push(c);
            }
        }
        assert!(
            visited.contains(&boss),
            "post-prune graph disconnected start→boss"
        );
    }

    /// Brute-force root→Boss path enumeration in `FindAllPaths`' DFS
    /// order — the reference the combinatorial index's rank arithmetic
    /// must reproduce.
    fn brute_force_paths(map: &ActMap, start: MapPointIdx) -> Vec<Vec<MapPointIdx>> {
        fn walk(
            map: &ActMap,
            v: MapPointIdx,
            prefix: &mut Vec<MapPointIdx>,
            out: &mut Vec<Vec<MapPointIdx>>,
        ) {
            prefix.push(v);
            if map.point(v).point_type == MapPointType::Boss {
                out.push(prefix.clone());
            } else {
                let children = map.point(v).children.clone();
                for c in children {
                    walk(map, c, prefix, out);
                }
            }
            prefix.pop();
        }
        let mut out = Vec::new();
        walk(map, start, &mut Vec::new(), &mut out);
        out
    }

    #[test]
    fn segment_index_rank_arithmetic_matches_brute_force() {
        // The SegmentIndex never materializes paths; it identifies
        // them by DFS rank via (arrival base, child offset, suffix
        // count) arithmetic. Cross-check every derived quantity
        // against a brute-force enumeration on real generated maps.
        for seed in [7, 42, 2026] {
            let mut rng = Rng::new(seed);
            let map = generate_act_map(15, &mut rng, 0, false, 0, None);
            let start = map.start.expect("start");
            let paths = brute_force_paths(&map, start);
            assert!(!paths.is_empty());
            for p in &paths {
                assert_eq!(
                    map.point(*p.last().expect("non-empty")).point_type,
                    MapPointType::Boss,
                    "every path ends at Boss"
                );
            }

            let index = SegmentIndex::build(&map, start);
            assert_eq!(index.total_paths as usize, paths.len(), "seed {seed}");

            // Node blocks: paths through node n == expansion of
            // bases(n) × cnt(n).
            for n in 0..map.points.len() {
                let n = u16::try_from(n).expect("idx");
                let want: Vec<u32> = paths
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.contains(&n))
                    .map(|(r, _)| u32::try_from(r).expect("rank"))
                    .collect();
                let got: Vec<u32> = index.bases[n as usize]
                    .iter()
                    .flat_map(|&b| b..b + index.cnt[n as usize])
                    .collect();
                assert_eq!(got, want, "node {n} blocks, seed {seed}");
            }

            // Run intervals: paths containing each distinct run ==
            // expansion of bases(start) offset by (rho_off, cnt_end).
            for rid in 0..u32::try_from(index.runs.len()).expect("rid") {
                let seg = index.run_slice(rid);
                let want: Vec<u32> = paths
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.windows(seg.len()).any(|w| w == seg))
                    .map(|(r, _)| u32::try_from(r).expect("rank"))
                    .collect();
                let run = &index.runs[rid as usize];
                let got: Vec<u32> = index.bases[seg[0] as usize]
                    .iter()
                    .flat_map(|&b| b + run.rho_off..b + run.rho_off + run.cnt_end)
                    .collect();
                assert_eq!(got, want, "run {rid} occurrences, seed {seed}");
            }
        }
    }

    #[test]
    fn bitset_clear_and_first_alive() {
        // The bitset ops don't depend on the graph — hand-build an
        // index with a 150-path bitset (word-boundary coverage).
        let t: u32 = 150;
        let mut alive = vec![u64::MAX; 3];
        alive[2] = (1u64 << (t as usize - 128)) - 1;
        let mut index = SegmentIndex {
            bases: Vec::new(),
            cnt: Vec::new(),
            child_offsets: Vec::new(),
            run_arena: Vec::new(),
            runs: Vec::new(),
            keys: Vec::new(),
            sorted_key_ids: Vec::new(),
            alive,
            total_paths: t,
        };
        assert_eq!(index.first_alive_in(0, t), Some(0));
        // Clear a range crossing a word boundary.
        index.clear_range(60, 70);
        assert_eq!(index.first_alive_in(60, 70), None);
        assert_eq!(index.first_alive_in(58, 70), Some(58));
        assert_eq!(index.first_alive_in(60, 71), Some(70));
        // Tail bits beyond total_paths are never reported alive.
        assert_eq!(index.first_alive_in(t, t), None);
        index.clear_range(0, t);
        assert_eq!(index.first_alive_in(0, t), None);
    }

    #[test]
    fn overlapping_segment_endpoint_match_is_not_overlap() {
        // Two segments with the same start and end but different
        // interiors should NOT be flagged as overlapping (C# only
        // counts interior positions).
        let a: Vec<MapPointIdx> = vec![0, 1, 2, 3, 4];
        let b: Vec<MapPointIdx> = vec![0, 5, 6, 7, 4];
        assert!(!overlapping_segment(&a, &b));
    }

    #[test]
    fn overlapping_segment_interior_match_is_overlap() {
        let a: Vec<MapPointIdx> = vec![0, 1, 2, 3, 4];
        let b: Vec<MapPointIdx> = vec![0, 1, 99, 3, 4]; // index 1 matches
        assert!(overlapping_segment(&a, &b));
    }

    #[test]
    fn segment_key_row_zero_uses_short_form() {
        // Hand-built tiny ActMap: row-0 start → row-1 child → row-2 boss
        let mut map = ActMap::new(2);
        let start = map.get_or_create(3, 0);
        let mid = map.get_or_create(3, 1);
        let boss = map.get_or_create(3, 2);
        map.start = Some(start);
        map.boss = Some(boss);
        map.add_child(start, mid);
        map.add_child(mid, boss);
        map.point_mut(start).point_type = MapPointType::Ancient;
        map.point_mut(mid).point_type = MapPointType::Monster;
        map.point_mut(boss).point_type = MapPointType::Boss;

        let seg: Segment = vec![start, mid, boss];
        let key = generate_segment_key(&map, &seg);
        // head.row == 0 → "{0}-{tail.col},{tail.row}-{types}"
        assert!(key.starts_with("0-3,2-"));
    }
}
