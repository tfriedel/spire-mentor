//! Texel corpus collector for MCTS self-play.
//!
//! Bead `sts2-rs-ulbl` (vsqc.1). Sister to `sts2-rs-uxyf`
//! (`eval_features` / `EVAL_FEATURE_NAMES` PyO3 surface) and parent
//! `sts2-rs-vsqc` (Texel weight-fitting). Records mid-combat
//! positions during self-play and labels them with the eventual
//! combat outcome — the standard Texel sampling pattern for
//! supervised tuning of static evaluators.
//!
//! ## Pipeline
//!
//! 1. The driver (`bin/sts2-corpus-gen.rs`) loops over diverse
//!    encounters and runs MCTS-driven self-play.
//! 2. Each player-decision step calls [`CorpusCollector::record_position`]
//!    with the *current* state — these are precisely the "leaf-eval
//!    consumer" positions whose MCTS search would have hit the
//!    rollout depth cap on most branches (the depth_cap_hit signal
//!    is implicit at the search level — the actual leaf-eval cost
//!    centers comes via `evaluate_in_progress`).
//! 3. When the combat ends, [`CorpusCollector::mark_combat_outcome`]
//!    backfills the binary outcome (1 = won, 0 = lost) for every
//!    position recorded in this combat.
//! 4. [`CorpusCollector::filter_round_at_least`] drops positions
//!    with `round < min_round` (default 2) — first-turn bias is
//!    suppressed since the player's options before any draw event
//!    are dominated by the starting hand.
//! 5. [`CorpusCollector::into_arrays`] yields the contiguous
//!    `features: Vec<f32>` (row-major `[N, 12]`) and `outcomes:
//!    Vec<u8>` arrays the NPZ writer ([`crate::npz`]) consumes.
//!
//! ## Column order
//!
//! Columns mirror `sts2_core::EVAL_FEATURE_NAMES` exactly (bead
//! `uxyf` pins the 12-column layout). Re-ordering MUST flow back
//! through `uxyf` + the Texel harness (`vsqc.3`) in lockstep.

use sts2_core::cards::{CardTag, CardType, TargetType};
use sts2_core::{
    eval_features, eval_features_wide, legal_actions, Action, Card, CombatState, PowerKind,
    WIDE_FEATURE_DIM,
};

use crate::action_index::{
    action_to_key, action_to_keyed_slot, action_to_slot, ACTION_KEY_DIM, CANONICAL_ACTION_DIM,
};

/// Which action-slot scheme the policy-target vectors use. `Canonical`
/// (640, card-id only) is the byte-stable default; `Keyed` (4672,
/// card × 8 target classes) distinguishes targets (sts2-rs-1bcx follow-up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyActionSpace {
    Canonical,
    Keyed,
}

impl PolicyActionSpace {
    pub fn dim(self) -> usize {
        match self {
            Self::Canonical => CANONICAL_ACTION_DIM,
            Self::Keyed => ACTION_KEY_DIM,
        }
    }
    pub fn slot(self, state: &CombatState, action: &Action) -> Option<usize> {
        match self {
            Self::Canonical => action_to_slot(state, action),
            Self::Keyed => action_to_keyed_slot(state, action),
        }
    }
}

/// Per-pile / per-slot id-array dimensions for the Wave-6 generalist
/// embedding extension (bead `sts2-rs-uqnh`). Pinned by
/// `docs/plans/2026-05-17-wave6-generalist-architecture.md` —
/// Python-side trainer (`scripts/value_policy_net_generalist.py`)
/// reads these exact widths off the NPZ header, so reordering or
/// widening MUST be reflected on both sides in lockstep.
///
/// All ids are emitted as `(variant as u16) + 1`; `0` is reserved
/// for NULL (empty slot / padding) so the Python embedding tables
/// can keep NULL at row 0 across all three vocabularies.
pub const HAND_SLOT_MAX: usize = 12;
pub const PILE_SLOT_MAX: usize = 80;
pub const ENEMY_SLOT_MAX: usize = 5;
pub const RELIC_SLOT_MAX: usize = 32;

/// Per-card static-feature width (Lever A, stage (b)). Layout:
/// `cost(1) | card_type one-hot(5) | target_type one-hot(5) |
/// upgraded(1) | base_damage(1) | base_block(1) | tags multi-hot(5)`.
/// Pinned to the `sts2_core` enum cardinalities (CardType=5 — Quest shares
/// the Skill slot —
/// TargetType=5, CardTag=5); bump here + in `card_feat_vec` +
/// `value_policy_net_generalist.py::CARD_FEAT_DIM` together if any
/// enum grows. Stage (b) of `docs/perf/2026-05-26-embedding-inference-
/// path-design.md`.
pub const CARD_FEAT_DIM: usize = 19;

/// Per-card static-feature vector for the embedding net. Deterministic
/// function of the live [`Card`] (cost/type/target/upgrade/damage/block/
/// tags). Summed over each pile by [`EmbeddingIds::from_state`] and at
/// inference by `EmbeddingValueNet` — the two MUST agree, so this is the
/// single source of truth. NULL/empty slots contribute nothing (they are
/// never passed here).
#[must_use]
pub fn card_feat_vec(card: &Card) -> [f32; CARD_FEAT_DIM] {
    let mut f = [0.0_f32; CARD_FEAT_DIM];
    // cost: X-cost (negative) emitted as -1.0; else the local cost / 3.0.
    f[0] = if card.base_cost() < 0 {
        -1.0
    } else {
        card.local_cost().max(0) as f32 / 3.0
    };
    // card_type one-hot [1..6]. Quest (GH #1111) keeps the Skill slot it
    // had while it was folded into Skill, so the width and trained nets hold.
    let ty = if card.card_type == CardType::Quest {
        CardType::Skill
    } else {
        card.card_type
    };
    f[1 + ty.index()] = 1.0;
    // target_type one-hot [6..11].
    let t = match card.target_type {
        TargetType::Enemy => 0,
        TargetType::AllEnemies => 1,
        TargetType::Self_ => 2,
        TargetType::None => 3,
        TargetType::Random => 4,
    };
    f[6 + t] = 1.0;
    f[11] = f32::from(card.upgraded);
    f[12] = card.base_damage as f32 / 20.0;
    f[13] = card.base_block as f32 / 20.0;
    // tags multi-hot [14..19] (CardTag discriminants 0..5).
    for (i, tag) in [
        CardTag::Strike,
        CardTag::Defend,
        CardTag::Minion,
        CardTag::OstyAttack,
        CardTag::Shiv,
    ]
    .into_iter()
    .enumerate()
    {
        if card.tags.contains(tag) {
            f[14 + i] = 1.0;
        }
    }
    f
}

/// Encounter-tier code emitted into the wide-NPZ `encounter_tier`
/// column (bead `sts2-rs-acbj`, step 1). The 4-tier `EncounterTier`
/// enum collapses to 3 buckets for downstream stratification: Weak
/// + Normal fold into `TIER_COMMON`, Elite → `TIER_ELITE`, Boss →
/// `TIER_BOSS`. `TIER_UNKNOWN` is the sentinel for fixtures whose
/// `[provenance].encounter` doesn't match any `ACT*_ENCOUNTERS`
/// row (defensive — synthetic-path combats always have a known
/// tier; toml-path with stale or corrupt provenance can hit this).
pub const TIER_COMMON: u8 = 0;
pub const TIER_ELITE: u8 = 1;
pub const TIER_BOSS: u8 = 2;
pub const TIER_UNKNOWN: u8 = 0xFF;

/// Deterministic FNV-1a 64-bit hash of a byte slice. Used to fold
/// `[provenance].run_id` (a UUID-string per .run scrape) into a
/// stable `u64` for NPZ emission + group-by-run_id splits in the
/// trainer. Stdlib `DefaultHasher` is randomized per-process, so
/// it can't satisfy the cross-run determinism the corpus pipeline
/// requires. Inline FNV-1a is small enough to keep here without a
/// new dep. Bead `sts2-rs-acbj`.
#[must_use]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Per-state sparse id arrays consumed by the Wave-6 generalist net
/// (bead `sts2-rs-uqnh`). All entries are `0` for "NULL / empty
/// slot"; non-null entries are `(rust_enum_discriminant as u16) + 1`.
/// Boxed because the inline footprint is `2 × (12 + 80*3 + 5 + 32)
/// = 538` bytes — non-trivial to pay per `CorpusEntryWide` when the
/// `--with-embeddings` flag is absent.
#[derive(Debug, Clone)]
pub struct EmbeddingIds {
    pub card_ids_hand: [u16; HAND_SLOT_MAX],
    pub card_ids_draw: [u16; PILE_SLOT_MAX],
    pub card_ids_discard: [u16; PILE_SLOT_MAX],
    pub card_ids_exhaust: [u16; PILE_SLOT_MAX],
    pub monster_slot_ids: [u16; ENEMY_SLOT_MAX],
    pub relic_set_ids: [u16; RELIC_SLOT_MAX],
    /// Per-pile **sum** of [`card_feat_vec`] over the slotted cards
    /// (Lever A, stage (b)). Summing is exact for the trainer's
    /// sum-pool: `sum_j concat(emb_j, feat_j) = concat(sum emb, sum
    /// feat)`, so emitting the pooled sum is equivalent to per-slot
    /// feats at a fraction of the footprint. NULL slots contribute
    /// nothing (no card → not summed).
    pub card_feats_hand: [f32; CARD_FEAT_DIM],
    pub card_feats_draw: [f32; CARD_FEAT_DIM],
    pub card_feats_discard: [f32; CARD_FEAT_DIM],
    pub card_feats_exhaust: [f32; CARD_FEAT_DIM],
}

impl EmbeddingIds {
    /// Build a fresh all-NULL id payload — every slot 0.
    #[must_use]
    pub fn null() -> Self {
        Self {
            card_ids_hand: [0; HAND_SLOT_MAX],
            card_ids_draw: [0; PILE_SLOT_MAX],
            card_ids_discard: [0; PILE_SLOT_MAX],
            card_ids_exhaust: [0; PILE_SLOT_MAX],
            monster_slot_ids: [0; ENEMY_SLOT_MAX],
            relic_set_ids: [0; RELIC_SLOT_MAX],
            card_feats_hand: [0.0; CARD_FEAT_DIM],
            card_feats_draw: [0.0; CARD_FEAT_DIM],
            card_feats_discard: [0.0; CARD_FEAT_DIM],
            card_feats_exhaust: [0.0; CARD_FEAT_DIM],
        }
    }

    /// Extract per-slot ids from `state`. Slots beyond capacity are
    /// silently truncated (the H/P/E/R_MAX constants are
    /// over-provisioned for the current content cardinality; a
    /// truncation here is a corpus-spec bug, not a runtime concern).
    /// Minion enemies are filtered out so the monster slot array
    /// reflects the same enemy set the leaf-eval `threat` /
    /// `time_pressure` terms see.
    #[must_use]
    pub fn from_state(state: &CombatState) -> Self {
        let mut ids = Self::null();
        let fill_cards = |slots: &mut [u16],
                          feat_sum: &mut [f32; CARD_FEAT_DIM],
                          pile: &[sts2_core::CardUid]| {
            // `zip` truncates at `slots.len()` (= the pile's SLOT_MAX), so
            // the feat-sum covers exactly the cards whose ids the net pools.
            for (slot, uid) in slots.iter_mut().zip(pile.iter()) {
                if let Some(card) = state.cards.iter().find(|c| c.uid == *uid) {
                    *slot = (card.card_id as u16).saturating_add(1);
                    let fv = card_feat_vec(card);
                    for (acc, v) in feat_sum.iter_mut().zip(fv.iter()) {
                        *acc += *v;
                    }
                }
            }
        };
        fill_cards(
            &mut ids.card_ids_hand,
            &mut ids.card_feats_hand,
            &state.player.hand,
        );
        fill_cards(
            &mut ids.card_ids_draw,
            &mut ids.card_feats_draw,
            &state.player.draw,
        );
        fill_cards(
            &mut ids.card_ids_discard,
            &mut ids.card_feats_discard,
            &state.player.discard,
        );
        fill_cards(
            &mut ids.card_ids_exhaust,
            &mut ids.card_feats_exhaust,
            &state.player.exhaust,
        );

        let mut enemy_slot = 0usize;
        for uid in &state.enemy_uids {
            if enemy_slot >= ENEMY_SLOT_MAX {
                break;
            }
            let Some(c) = state.creatures.iter().find(|c| c.id == uid.0) else {
                continue;
            };
            if c.current_hp <= 0 {
                continue;
            }
            let is_minion = c.powers.iter().any(|puid| {
                state
                    .powers
                    .iter()
                    .any(|p| p.id == puid.0 && p.kind == PowerKind::Minion)
            });
            if is_minion {
                continue;
            }
            if let Some(species) = c.species {
                ids.monster_slot_ids[enemy_slot] = (species as u16).saturating_add(1);
                enemy_slot += 1;
            }
        }

        for (slot, relic) in ids.relic_set_ids.iter_mut().zip(state.relics.iter()) {
            *slot = (relic.kind as u16).saturating_add(1);
        }

        ids
    }
}

/// One Texel-corpus position: the 14-column feature vector at a
/// mid-combat decision point, plus the round number and the
/// eventual win/loss outcome of the combat that contained it.
///
/// Bumped 12 → 13 by bead `sts2-rs-ns1s` (added `playable_cards`
/// mobility column); bumped 13 → 14 by bead `sts2-rs-5l4y` (added
/// `survival_horizon` term); bumped 14 → 15 by bead `sts2-rs-gzbh`
/// (added `curse_in_hand` term). Downstream NPZ readers must update
/// their expected column count + header.
#[derive(Debug, Clone, Copy)]
pub struct CorpusEntry {
    /// Feature column order pinned by
    /// [`sts2_core::EVAL_FEATURE_NAMES`] (bead `uxyf`; +1 col
    /// `playable_cards` from `ns1s`; +1 col `survival_horizon`
    /// from `5l4y`; +1 col `curse_in_hand` from `gzbh`; +1 col
    /// `time_pressure` from `ot78`; +1 col `time_to_lethal` from
    /// `m0f2`; +1 col `wasted_block` from `adulv`; +1 col
    /// `enemy_slippery_stacks` from the `ldd1c` follow-up).
    pub features: [f32; 29],
    /// `state.round` at the moment of capture. Round-1 positions
    /// are typically filtered out (default `min_round = 2`).
    pub round: u16,
    /// Eventual combat outcome — 1 = player won, 0 = lost.
    /// [`u8::MAX`] is a sentinel for "unlabeled yet"; the corpus
    /// generator panics if any sentinel survives into
    /// [`CorpusCollector::into_arrays`].
    pub outcome: u8,
    /// Per-row group key for train/val/test splits (bead
    /// `sts2-rs-e2bjk` — narrow-path port of the wide-path
    /// `sts2-rs-acbj` step 1 plumbing). `0` is "unset" — the
    /// trainer's `--group-split` flag treats an all-zero column
    /// as the absence of group identifiers and falls back to
    /// per-row random splits. FNV-1a 64 of `[provenance].run_id`
    /// for toml-source combats; `combat_idx | 1` for synthetic.
    pub run_id_u64: u64,
    /// 3-bucket encounter-tier code (bead `sts2-rs-e2bjk`). One
    /// of [`TIER_COMMON`] / [`TIER_ELITE`] / [`TIER_BOSS`] /
    /// [`TIER_UNKNOWN`].
    pub encounter_tier_u8: u8,
}

const OUTCOME_UNLABELED: u8 = u8::MAX;

impl CorpusEntry {
    /// Capture the 12 feature scalars at this state. `outcome` is
    /// filled in later via [`CorpusCollector::mark_combat_outcome`].
    /// Provenance defaults to (0, TIER_UNKNOWN) — overwritten by
    /// the collector when [`CorpusCollector::record_position`]
    /// runs under a pending-provenance window.
    pub fn from_state(state: &CombatState) -> Self {
        Self {
            features: eval_features(state),
            round: state.round,
            outcome: OUTCOME_UNLABELED,
            run_id_u64: 0,
            encounter_tier_u8: TIER_UNKNOWN,
        }
    }
}

/// Append-only collector for [`CorpusEntry`] rows, with combat-
/// boundary tracking so outcomes can be backfilled on combat end.
#[derive(Debug)]
pub struct CorpusCollector {
    entries: Vec<CorpusEntry>,
    /// Half-open `[start, end)` index ranges per recorded combat,
    /// used by [`Self::mark_combat_outcome`] to label every entry
    /// from the current combat with the same outcome scalar.
    pending_combat_start: usize,
    /// Per-combat group key applied to every entry from
    /// [`Self::record_position`] until the next
    /// [`Self::set_pending_combat_provenance`] call. `0` means
    /// "unset". Bead `sts2-rs-e2bjk` (narrow-path mirror of the
    /// wide collector's `acbj` step 1 plumbing).
    pending_run_id: u64,
    /// Per-combat encounter-tier code applied to every entry from
    /// [`Self::record_position`] until the next
    /// [`Self::set_pending_combat_provenance`] call. Bead
    /// `sts2-rs-e2bjk`.
    pending_tier: u8,
}

impl Default for CorpusCollector {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            pending_combat_start: 0,
            pending_run_id: 0,
            pending_tier: TIER_UNKNOWN,
        }
    }
}

impl CorpusCollector {
    /// Fresh empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-combat provenance (group key + tier) that every
    /// subsequent [`Self::record_position`] call will stamp onto
    /// the captured entry. Call once per combat *before* the first
    /// `record_position` of that combat. Bead `sts2-rs-e2bjk`.
    ///
    /// - `run_id_u64`: FNV-1a 64 of `[provenance].run_id` for
    ///   toml-source combats; `combat_idx | 1` (stable per-combat
    ///   id) for synthetic-path combats. `0` is reserved as the
    ///   "unset" sentinel.
    /// - `tier`: one of [`TIER_COMMON`] / [`TIER_ELITE`] /
    ///   [`TIER_BOSS`] / [`TIER_UNKNOWN`].
    pub fn set_pending_combat_provenance(&mut self, run_id_u64: u64, tier: u8) {
        self.pending_run_id = run_id_u64;
        self.pending_tier = tier;
    }

    /// Capture one mid-combat position. Must be paired with a
    /// matching [`Self::mark_combat_outcome`] call once the combat
    /// resolves.
    pub fn record_position(&mut self, state: &CombatState) {
        let mut entry = CorpusEntry::from_state(state);
        entry.run_id_u64 = self.pending_run_id;
        entry.encounter_tier_u8 = self.pending_tier;
        self.entries.push(entry);
    }

    /// Label every entry recorded since the last call to this
    /// method with `outcome` (1 = won, 0 = lost) and start a new
    /// pending-combat window. Idempotent on empty windows.
    pub fn mark_combat_outcome(&mut self, won: bool) {
        let outcome = u8::from(won);
        for entry in &mut self.entries[self.pending_combat_start..] {
            entry.outcome = outcome;
        }
        self.pending_combat_start = self.entries.len();
    }

    /// Drop every entry whose `round < min_round`. Returns the
    /// number of entries removed. Texel-tuning default is
    /// `min_round = 2`.
    pub fn filter_round_at_least(&mut self, min_round: u16) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.round >= min_round);
        before - self.entries.len()
    }

    /// Truncate to at most `target_len` entries. Used by the
    /// parallel corpus generator (`sts2-rs-3b0p`) to clip an
    /// oversampled merge down to the requested row count
    /// deterministically.
    pub fn truncate(&mut self, target_len: usize) {
        if self.entries.len() > target_len {
            self.entries.truncate(target_len);
            self.pending_combat_start = self.pending_combat_start.min(self.entries.len());
        }
    }

    /// Append every entry from `other` into `self` in order. Used by
    /// the rayon-parallel corpus generator (`sts2-rs-3b0p`):
    /// per-worker collectors are merged into a single driver
    /// collector once each worker has fully labeled its combats.
    /// Panics if `other` still has an open (unlabeled)
    /// pending-combat window, since merging would lose the boundary
    /// between labeled and unlabeled rows.
    pub fn merge(&mut self, mut other: Self) {
        assert_eq!(
            other.pending_combat_start,
            other.entries.len(),
            "CorpusCollector::merge: refusing to merge a collector with an open pending-combat \
             window; call mark_combat_outcome on every recorded combat before merging.",
        );
        self.entries.append(&mut other.entries);
        self.pending_combat_start = self.entries.len();
    }

    /// Number of entries currently retained.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the collector holds any entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Materialize the row-major feature matrix and the outcome
    /// vector. Panics if any entry is still unlabeled — every
    /// `record_position` must be followed by a `mark_combat_outcome`
    /// before serialization.
    pub fn into_arrays(self) -> (Vec<f32>, Vec<u8>) {
        let bundle = self.into_arrays_with_provenance();
        (bundle.features, bundle.outcomes)
    }

    /// Materialize features + outcomes plus two per-row provenance
    /// arrays (`run_id_u64`, `encounter_tier_u8`). Mirrors
    /// [`CorpusCollectorWide::into_arrays_with_provenance`] for the
    /// narrow 27-col schema so the Texel tuner's `--group-split`
    /// flag has a real group identifier to partition on. Bead
    /// `sts2-rs-e2bjk`.
    pub fn into_arrays_with_provenance(self) -> NarrowArraysWithProvenance {
        let n = self.entries.len();
        let mut features = Vec::with_capacity(n * 27);
        let mut outcomes = Vec::with_capacity(n);
        let mut run_ids: Vec<u64> = Vec::with_capacity(n);
        let mut tiers: Vec<u8> = Vec::with_capacity(n);
        for entry in self.entries {
            assert!(
                entry.outcome != OUTCOME_UNLABELED,
                "CorpusCollector::into_arrays called with unlabeled entry — \
                 every record_position must be followed by mark_combat_outcome"
            );
            features.extend_from_slice(&entry.features);
            outcomes.push(entry.outcome);
            run_ids.push(entry.run_id_u64);
            tiers.push(entry.encounter_tier_u8);
        }
        NarrowArraysWithProvenance {
            features,
            outcomes,
            run_id_u64: run_ids,
            encounter_tier_u8: tiers,
        }
    }
}

/// Output bundle from [`CorpusCollector::into_arrays_with_provenance`].
/// Bead `sts2-rs-e2bjk` — narrow-path mirror of
/// [`WideArraysWithProvenance`].
#[derive(Debug)]
pub struct NarrowArraysWithProvenance {
    pub features: Vec<f32>,
    pub outcomes: Vec<u8>,
    pub run_id_u64: Vec<u64>,
    pub encounter_tier_u8: Vec<u8>,
}

// ── Wide-schema sister type ────────────────────────────────────────────────

/// Wide-schema counterpart to [`CorpusEntry`]. Captures the
/// `[f32; WIDE_FEATURE_DIM]` (80-column) feature vector pinned by
/// [`sts2_core::EVAL_FEATURE_NAMES_WIDE`] PLUS the two qctu.6.x
/// auxiliary-target columns:
///
/// - `final_hp_norm` — `final_player_hp / max_hp` at combat
///   resolution, clamped to `[0, 1]`. KataGo's auxiliary head
///   target (§4.7.1 of the leaf-eval plan).
/// - `visit_distribution` — softmax-normalized MCTS root-visit
///   ratios bucketed into [`CANONICAL_ACTION_DIM`] slots per the
///   `action_index` module. Boxed because 640 f32 × 100k entries
///   inline would be ~256 MiB stack — heap-allocate per row.
///
/// Both fields default to "unlabeled" sentinels so the collector
/// can pre-allocate at `record_position` time and the corpus
/// generator can backfill them later (HP at combat end; visit
/// distribution immediately after the per-position MCTS search
/// completes).
///
/// Beads: `sts2-rs-4vo6` (initial 80-col schema), `sts2-rs-quuu`
/// (this bead — aux+policy column extension).
#[derive(Debug, Clone)]
pub struct CorpusEntryWide {
    pub features: [f32; WIDE_FEATURE_DIM],
    pub round: u16,
    pub outcome: u8,
    /// `f32::NAN` when unlabeled — backfilled at combat end via
    /// [`CorpusCollectorWide::mark_combat_outcome_with_hp`].
    pub final_hp_norm: f32,
    /// Dense visit-distribution vector (boxed). All-zeros when
    /// unrecorded; the corpus generator overwrites it via
    /// [`CorpusCollectorWide::record_position_with_visits`].
    pub visit_distribution: Box<[f32]>, // len == policy action-space dim
    /// Wave-6 generalist-net embedding ids (bead `sts2-rs-uqnh`).
    /// `None` when the collector wasn't configured with
    /// [`CorpusCollectorWide::with_embeddings`] — the
    /// [`Self::into_arrays_with_embeddings`] surface asserts every
    /// entry carries a payload before emitting the id arrays.
    pub embedding_ids: Option<Box<EmbeddingIds>>,
    /// Per-row group key for train/val/test splits (bead
    /// `sts2-rs-acbj`, step 1). `0` is the default and means "no
    /// run-id provenance set for this combat"; the trainer's
    /// `--group-col` flag flags an all-zero column as a misuse
    /// (would degenerate group-split into "one giant group").
    /// Source: FNV-1a 64 of `[provenance].run_id` for toml-source
    /// combats, `combat_idx` (synthetic-path stable id) otherwise.
    pub run_id_u64: u64,
    /// Per-row 3-bucket encounter-tier code (bead `sts2-rs-acbj`,
    /// step 1). One of [`TIER_COMMON`] / [`TIER_ELITE`] /
    /// [`TIER_BOSS`] / [`TIER_UNKNOWN`].
    pub encounter_tier_u8: u8,
    /// MCTS root Q at this position, range `[-1.0, 1.0]`. `f32::NAN`
    /// when no Q was available (the recorder lacked search context, or
    /// the root had no expanded children). Paired with
    /// [`Self::mcts_q_valid`] — the trainer's mixed-target loss
    /// (bead `sts2-rs-wrip2`) skips the TD term where the mask is 0.
    pub mcts_root_q: f32,
    /// 0 / 1 mask companion to [`Self::mcts_root_q`]. `1` when the
    /// Q value is from a real MCTS search; `0` otherwise.
    pub mcts_q_valid: u8,
    /// Dense legal-action mask (bead `sts2-rs-8bb9i` / svstv.2).
    /// `1` at slot `i` iff at least one entry in
    /// [`sts2_core::legal_actions`] at record time projects onto slot
    /// `i` via [`action_to_slot`]. All-zeros sentinel when no MCTS
    /// search was driven (the [`Self::from_state`] default) — distinct
    /// from "search ran but found no legal moves" (still all-zeros
    /// but `outcome` will be backfilled as a terminal row).
    ///
    /// Drives the trainer-side legal-mass aux loss (svstv.2 step 2):
    /// `||softmax(logits) * (1 - mask)||_2^2` weighted by
    /// `--lambda-illegal-kl`. Replaces the
    /// `visit_distribution > 0` proxy used in svstv.0 which
    /// under-counts legal-but-not-visited slots.
    pub legal_action_mask: Box<[u8]>, // len == policy action-space dim
    /// Per-row raw Texel-tuned `evaluate_in_progress(state)` score
    /// (bead `sts2-rs-aa09l` / 37vzt.1). Drives the trainer-side
    /// composed value head (`--texel-compose` in
    /// `scripts/value_policy_net.py`), where the network learns a
    /// residual on top of the Texel baseline rather than predicting
    /// outcome from scratch. `f32::NAN` is reserved as a "not
    /// captured" sentinel; the trainer fails fast if `--texel-compose`
    /// sees NaNs in the loaded column.
    pub raw_texel_score: f32,
}

impl CorpusEntryWide {
    /// Capture the feature-only payload — aux + policy targets
    /// stay at their unlabeled sentinels. Used when the corpus
    /// generator doesn't have MCTS-search context handy (e.g. a
    /// non-MCTS recorder, or a smoke test).
    pub fn from_state(state: &CombatState) -> Self {
        Self {
            features: eval_features_wide(state),
            round: state.round,
            outcome: OUTCOME_UNLABELED,
            final_hp_norm: f32::NAN,
            visit_distribution: vec![0.0_f32; CANONICAL_ACTION_DIM].into_boxed_slice(),
            embedding_ids: None,
            run_id_u64: 0,
            encounter_tier_u8: TIER_UNKNOWN,
            mcts_root_q: f32::NAN,
            mcts_q_valid: 0,
            legal_action_mask: vec![0u8; CANONICAL_ACTION_DIM].into_boxed_slice(),
            // Bead `sts2-rs-aa09l`: snapshot the Texel-tuned HCE score
            // at record time. Drives the 37vzt composed-value-head
            // trainer; safe at this pin to call on any combat state
            // (`evaluate_in_progress` returns the in-progress branch
            // even for terminal rows — the trainer doesn't use texel
            // for terminal positions anyway).
            raw_texel_score: sts2_core::evaluate_in_progress(state),
        }
    }
}

/// Wide-schema corpus collector. API mirror of [`CorpusCollector`]
/// — same `record_position` / `mark_combat_outcome` /
/// `filter_round_at_least` / `into_arrays` shape — but the row
/// width is [`WIDE_FEATURE_DIM`] (80) instead of 12. The
/// [`Self::into_arrays`] return shape is still
/// `(Vec<f32>, Vec<u8>)`, with the feature row stride implicit at
/// 80; callers (the corpus generator binary + downstream NPZ
/// writer) take the column count via [`WIDE_FEATURE_DIM`] or
/// [`sts2_core::EVAL_FEATURE_NAMES_WIDE::len()`].
#[derive(Debug)]
pub struct CorpusCollectorWide {
    entries: Vec<CorpusEntryWide>,
    pending_combat_start: usize,
    /// When set, every `record_position*` call also captures the
    /// per-state [`EmbeddingIds`] payload (bead `sts2-rs-uqnh`).
    /// Off by default — the id capture allocates a 538-byte box per
    /// row, ~50 MiB on a 100k-row corpus, which would silently
    /// double the existing wide-NPZ footprint if it were always-on.
    with_embeddings: bool,
    /// Per-combat group key applied to every entry from
    /// [`Self::record_position`] / [`Self::record_position_with_visits`]
    /// until the next [`Self::set_pending_combat_provenance`] call.
    /// `0` means "unset" — the trainer's `--group-col` mode flags
    /// an all-zero column as a misuse. Bead `sts2-rs-acbj`, step 1.
    pending_run_id: u64,
    /// Per-combat encounter-tier code applied to every entry from
    /// [`Self::record_position`] / [`Self::record_position_with_visits`]
    /// until the next [`Self::set_pending_combat_provenance`] call.
    /// One of [`TIER_COMMON`] / [`TIER_ELITE`] / [`TIER_BOSS`] /
    /// [`TIER_UNKNOWN`]. Bead `sts2-rs-acbj`, step 1.
    pending_tier: u8,
    /// Running policy-projection quality metrics over every
    /// `record_position_with_visits` call. GH #24 / bead
    /// `sts2-rs-9c7vr` — surfaces the legal-mass / entropy /
    /// collision / uniform-fallback rate the issue's acceptance
    /// criteria asks for, without touching the on-disk NPZ schema.
    policy_metrics: PolicyMetrics,
    /// Which action-slot scheme to use for policy-target vectors.
    /// Defaults to [`PolicyActionSpace::Canonical`] (640, byte-stable);
    /// set to [`PolicyActionSpace::Keyed`] (4672) via
    /// [`Self::with_policy_action_space`] to distinguish targets.
    /// All positions in a collector share the same scheme.
    policy_action_space: PolicyActionSpace,
}

/// Aggregate policy-projection quality metrics tracked during
/// `record_position_with_visits`. Each accumulator is summed over
/// every recorded position; divide by `count` to get the mean.
///
/// Fields:
/// - `count`: number of positions sampled.
/// - `sum_legal_mass`: total visit-mass fraction that survives the
///   `action_to_slot` projection before any renormalization (mass
///   from actions whose hand_index resolves and whose CardId is in
///   `CardId::all()`). Mean → "legal-mass-before-renorm" — the
///   issue's lead metric.
/// - `sum_entropy`: sum of Shannon entropy of the per-position
///   normalized visit distribution (base-e nats). Mean → policy
///   entropy. Low entropy on uniform-rollout corpora indicates the
///   visit counts are concentrating on a single action despite the
///   uniform rollout signal.
/// - `slot_collision_count`: number of positions where ≥2 distinct
///   legal actions projected onto the same `action_to_slot` slot
///   (i.e. same CardId, different target / instance). Mean → slot
///   collision rate.
/// - `key_collision_count`: same count under the new
///   `action_to_keyed_slot` projection (should be zero or near-zero
///   if the keyed projection actually resolves target ambiguity).
///   The delta between these two is the headline number for GH #24.
/// - `uniform_fallback_count`: positions where `total visits == 0`
///   (the recorder fell back to an all-zeros distribution — the
///   trainer's "uniform fallback" path will kick in for these).
#[derive(Debug, Clone, Default)]
pub struct PolicyMetrics {
    pub count: u64,
    pub sum_legal_mass: f64,
    pub sum_entropy: f64,
    pub slot_collision_count: u64,
    pub key_collision_count: u64,
    pub uniform_fallback_count: u64,
}

impl PolicyMetrics {
    /// Merge `other` into `self` (used by the rayon-parallel combat
    /// loop in corpus_gen to fan-in per-worker metrics).
    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum_legal_mass += other.sum_legal_mass;
        self.sum_entropy += other.sum_entropy;
        self.slot_collision_count += other.slot_collision_count;
        self.key_collision_count += other.key_collision_count;
        self.uniform_fallback_count += other.uniform_fallback_count;
    }

    /// Mean legal-mass-before-renorm in `[0, 1]`. Returns `0.0` for
    /// the empty-corpus case.
    pub fn mean_legal_mass(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_legal_mass / (self.count as f64)
        }
    }

    pub fn mean_entropy(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_entropy / (self.count as f64)
        }
    }

    pub fn slot_collision_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.slot_collision_count as f64) / (self.count as f64)
        }
    }

    pub fn key_collision_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.key_collision_count as f64) / (self.count as f64)
        }
    }

    pub fn uniform_fallback_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.uniform_fallback_count as f64) / (self.count as f64)
        }
    }
}

impl Default for CorpusCollectorWide {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            pending_combat_start: 0,
            with_embeddings: false,
            pending_run_id: 0,
            pending_tier: TIER_UNKNOWN,
            policy_metrics: PolicyMetrics::default(),
            policy_action_space: PolicyActionSpace::Canonical,
        }
    }
}

impl CorpusCollectorWide {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-combat provenance (group key + tier) that every
    /// subsequent [`Self::record_position`] /
    /// [`Self::record_position_with_visits`] call will stamp onto
    /// the captured entry. Call once per combat *before* the first
    /// `record_position` of that combat. Bead `sts2-rs-acbj`,
    /// step 1.
    ///
    /// - `run_id_u64`: FNV-1a 64 of `[provenance].run_id` for
    ///   toml-source combats; `combat_idx` (stable per-combat id)
    ///   for synthetic-path combats. `0` is reserved as the
    ///   "unset" sentinel — callers SHOULD pass a non-zero value
    ///   (FNV-1a of empty string is `0xcbf2_9ce4_8422_2325`, not
    ///   zero, so a fixture with an empty `run_id` won't collide).
    /// - `tier`: one of [`TIER_COMMON`] / [`TIER_ELITE`] /
    ///   [`TIER_BOSS`] / [`TIER_UNKNOWN`].
    pub fn set_pending_combat_provenance(&mut self, run_id_u64: u64, tier: u8) {
        self.pending_run_id = run_id_u64;
        self.pending_tier = tier;
    }

    /// Enable Wave-6 embedding-id capture for every subsequent
    /// `record_position*` call. Idempotent — re-calling with the
    /// flag already set is a no-op. Bead `sts2-rs-uqnh`.
    #[must_use]
    pub fn with_embeddings(mut self) -> Self {
        self.with_embeddings = true;
        self
    }

    /// Whether this collector is set to capture the Wave-6
    /// embedding-id payload alongside the handcrafted wide features.
    #[must_use]
    pub fn captures_embeddings(&self) -> bool {
        self.with_embeddings
    }

    /// Builder: select the policy action-space scheme used for
    /// `visit_distribution` / `legal_action_mask` targets.
    ///
    /// - [`PolicyActionSpace::Canonical`] (640, default) — card-id
    ///   only; byte-identical to all prior NPZ corpora.
    /// - [`PolicyActionSpace::Keyed`] (4672) — card × 8 target
    ///   classes via [`action_to_keyed_slot`]; no target collisions.
    ///
    /// The chosen scheme applies to every position in this collector.
    /// Mixing schemes across a corpus file is not supported — create
    /// separate collectors and concatenate NPZ arrays if needed.
    #[must_use]
    pub fn with_policy_action_space(mut self, space: PolicyActionSpace) -> Self {
        self.policy_action_space = space;
        self
    }

    /// Return the active [`PolicyActionSpace`] for this collector.
    #[must_use]
    pub fn policy_action_space(&self) -> PolicyActionSpace {
        self.policy_action_space
    }

    /// Record a position WITHOUT MCTS visit context. Useful for
    /// non-MCTS recorders or feature-only smoke tests — aux + policy
    /// columns stay at their unlabeled sentinels.
    pub fn record_position(&mut self, state: &CombatState) {
        let mut entry = CorpusEntryWide::from_state(state);
        if self.with_embeddings {
            entry.embedding_ids = Some(Box::new(EmbeddingIds::from_state(state)));
        }
        entry.run_id_u64 = self.pending_run_id;
        entry.encounter_tier_u8 = self.pending_tier;
        // `from_state` defaults to the 640-wide canonical sentinel;
        // resize to the collector's chosen scheme so a Keyed collector
        // never pushes a mixed-width row (silent NPZ corruption).
        let dim = self.policy_action_space.dim();
        entry.visit_distribution = vec![0.0_f32; dim].into_boxed_slice();
        entry.legal_action_mask = vec![0u8; dim].into_boxed_slice();
        self.entries.push(entry);
    }

    /// Record a position with the MCTS root-visit distribution
    /// from the search that just completed at this state.
    /// `visits` is the `(action, visit_count)` slice from
    /// [`crate::Mcts::root_outcomes`]; the helper resolves each
    /// action to a canonical slot via [`action_to_slot`] and
    /// emits a softmax-normalized dense vector. Bead `sts2-rs-quuu`.
    ///
    /// `root_q` is the visit-weighted mean Q across root children
    /// from [`crate::Mcts::root_q`], range `[-1.0, 1.0]`. `None` is
    /// stamped onto the row as `mcts_root_q=NaN, mcts_q_valid=0` so
    /// the trainer's mixed-target loss (bead `sts2-rs-wrip2`) can
    /// skip the TD term for that row.
    pub fn record_position_with_visits(
        &mut self,
        state: &CombatState,
        visits: &[(Action, u32)],
        root_q: Option<f32>,
    ) {
        let mut entry = CorpusEntryWide::from_state(state);
        if self.with_embeddings {
            entry.embedding_ids = Some(Box::new(EmbeddingIds::from_state(state)));
        }
        entry.run_id_u64 = self.pending_run_id;
        entry.encounter_tier_u8 = self.pending_tier;
        if let Some(q) = root_q {
            entry.mcts_root_q = q;
            entry.mcts_q_valid = 1;
        }
        let space = self.policy_action_space;
        let dim = space.dim();
        let mut dist = vec![0.0_f32; dim];
        let total: u64 = visits.iter().map(|(_, v)| u64::from(*v)).sum();

        // GH #24 / bead `sts2-rs-9c7vr`: track policy-projection
        // quality metrics. `slot_seen` / `key_seen` flag whether
        // each projection's destination already has visit mass at
        // it, so we can count collisions (≥2 distinct actions
        // landing on the same slot).
        // NOTE: `slot_seen` is sized to the CANONICAL dim for the
        // collision metric (always tracks the canonical projection,
        // regardless of which scheme writes the target vector) — the
        // diagnostic is anchored to the 640-slot baseline for GH #24.
        self.policy_metrics.count += 1;
        let mut slot_seen = vec![false; CANONICAL_ACTION_DIM];
        let mut key_seen: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(visits.len());
        let mut slot_collision = false;
        let mut key_collision = false;
        let mut legal_mass: f64 = 0.0;

        if total > 0 {
            let total_f = total as f32;
            let total_f64 = total as f64;
            for (action, count) in visits {
                let mass = f64::from(*count) / total_f64;
                // Write to target vector using the selected scheme.
                if let Some(slot) = space.slot(state, action) {
                    dist[slot] += (*count as f32) / total_f;
                }
                // Collision metric always uses canonical (slot_seen sized to 640).
                if let Some(slot) = action_to_slot(state, action) {
                    if slot_seen[slot] {
                        slot_collision = true;
                    } else {
                        slot_seen[slot] = true;
                    }
                    legal_mass += mass;
                }
                if let Some(keyed) =
                    action_to_key(state, action).and_then(crate::action_index::action_key_to_slot)
                {
                    if !key_seen.insert(keyed) {
                        key_collision = true;
                    }
                } else {
                    // The keyed projection has the same resolvability
                    // contract as the slot one — both paths read
                    // `state.player.hand` + `CardId::all()`. Reaching
                    // this branch means slot projection also failed,
                    // and the visit mass was already excluded from
                    // legal_mass above.
                }
            }

            // Shannon entropy of the (normalized-by-total) visit
            // distribution. Build it from `visits` directly rather
            // than `dist` so we measure the raw search output, not
            // the projected one. Zero-count actions contribute 0.
            let mut entropy = 0.0_f64;
            for (_, count) in visits {
                if *count == 0 {
                    continue;
                }
                let p = f64::from(*count) / total_f64;
                entropy -= p * p.ln();
            }
            self.policy_metrics.sum_entropy += entropy;
            self.policy_metrics.sum_legal_mass += legal_mass;
            if slot_collision {
                self.policy_metrics.slot_collision_count += 1;
            }
            if key_collision {
                self.policy_metrics.key_collision_count += 1;
            }
        } else {
            // total == 0 — the search produced no visits for this
            // position (legal_actions empty or all branches pruned).
            // The trainer's uniform-fallback path will kick in.
            self.policy_metrics.uniform_fallback_count += 1;
        }
        entry.visit_distribution = dist.into_boxed_slice();
        // Bead `sts2-rs-8bb9i` (svstv.2): dense legal-action mask
        // sourced from `legal_actions(state)` and projected through
        // the same slot scheme the visit distribution uses. Distinct
        // from `visit_distribution > 0` because MCTS may not visit
        // every legal slot at low simulation budgets — the trainer's
        // aux loss needs the full legal support, not just the visited
        // subset.
        let mut mask = vec![0u8; dim];
        for action in legal_actions(state) {
            if let Some(slot) = space.slot(state, &action) {
                mask[slot] = 1;
            }
        }
        entry.legal_action_mask = mask.into_boxed_slice();
        self.entries.push(entry);
    }

    /// Accessor for the running policy-projection quality metrics
    /// tracked across every `record_position_with_visits` call. GH
    /// #24 / bead `sts2-rs-9c7vr`.
    #[must_use]
    pub fn policy_metrics(&self) -> &PolicyMetrics {
        &self.policy_metrics
    }

    /// Backfill the outcome only. Kept for back-compat with the
    /// qctu.2 corpus pipeline that doesn't capture final HP.
    /// `final_hp_norm` stays NaN — downstream consumers MUST tolerate
    /// NaN in that column when calling
    /// [`Self::into_arrays_with_extras`] without the
    /// `--write-aux-hp` flag (the corpus generator gates this).
    pub fn mark_combat_outcome(&mut self, won: bool) {
        let outcome = u8::from(won);
        for entry in &mut self.entries[self.pending_combat_start..] {
            entry.outcome = outcome;
        }
        self.pending_combat_start = self.entries.len();
    }

    /// Backfill BOTH outcome and `final_hp_norm` for every entry
    /// recorded since the last `mark_combat_outcome*` call. The HP
    /// fraction is clamped to `[0, 1]` (defensive — `current_hp`
    /// can theoretically exceed `max_hp` via overhealing, and
    /// negative HPs are reported as `0` from the engine's
    /// `terminal_value`). Bead `sts2-rs-quuu`.
    pub fn mark_combat_outcome_with_hp(&mut self, won: bool, final_hp_norm: f32) {
        let outcome = u8::from(won);
        let clamped = final_hp_norm.clamp(0.0, 1.0);
        for entry in &mut self.entries[self.pending_combat_start..] {
            entry.outcome = outcome;
            entry.final_hp_norm = clamped;
        }
        self.pending_combat_start = self.entries.len();
    }

    pub fn filter_round_at_least(&mut self, min_round: u16) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.round >= min_round);
        before - self.entries.len()
    }

    /// See [`CorpusCollector::truncate`].
    pub fn truncate(&mut self, target_len: usize) {
        if self.entries.len() > target_len {
            self.entries.truncate(target_len);
            self.pending_combat_start = self.pending_combat_start.min(self.entries.len());
        }
    }

    /// See [`CorpusCollector::merge`].
    pub fn merge(&mut self, mut other: Self) {
        assert_eq!(
            self.policy_action_space, other.policy_action_space,
            "CorpusCollectorWide::merge: mismatched policy_action_space ({:?} vs {:?})",
            self.policy_action_space, other.policy_action_space,
        );
        assert_eq!(
            other.pending_combat_start,
            other.entries.len(),
            "CorpusCollectorWide::merge: refusing to merge a collector with an open \
             pending-combat window; call mark_combat_outcome on every recorded combat \
             before merging.",
        );
        self.entries.append(&mut other.entries);
        self.pending_combat_start = self.entries.len();
        self.policy_metrics.merge(&other.policy_metrics);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn into_arrays(self) -> (Vec<f32>, Vec<u8>) {
        let (features, outcomes, _, _) = self.into_arrays_with_extras();
        (features, outcomes)
    }

    /// Materialize all four arrays consumed by the qctu.3 trainer:
    /// `features` (row-major `[N, 80]`), `outcomes` (`[N]`),
    /// `final_hp_norm` (`[N]`), `visit_distribution` (row-major
    /// `[N, CANONICAL_ACTION_DIM]`). NaN HP entries pass through —
    /// the NPZ writer / downstream trainer is expected to mask
    /// them if the aux head isn't active for this corpus run.
    /// Bead `sts2-rs-quuu`.
    pub fn into_arrays_with_extras(self) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>) {
        let bundle = self.into_arrays_with_provenance();
        (
            bundle.features,
            bundle.outcomes,
            bundle.final_hp_norm,
            bundle.visit_distribution,
        )
    }

    /// Wave-acbj emit (bead `sts2-rs-acbj`, step 1). Materializes
    /// the four base arrays from [`Self::into_arrays_with_extras`]
    /// plus two per-row provenance arrays:
    ///
    /// - `run_id_u64` `[N]` — FNV-1a 64 of `[provenance].run_id`
    ///   (toml path) or stable `combat_idx` (synthetic path). Used
    ///   by the trainer's `--group-col` flag to enforce disjoint
    ///   train/val/test partitions over runs (no same-deck-different-
    ///   floor leakage).
    /// - `encounter_tier_u8` `[N]` — 3-bucket tier code, one of
    ///   [`TIER_COMMON`] / [`TIER_ELITE`] / [`TIER_BOSS`] /
    ///   [`TIER_UNKNOWN`]. Used by per-tier loss weighting + per-
    ///   tier A/B reporting.
    pub fn into_arrays_with_provenance(self) -> WideArraysWithProvenance {
        let n = self.entries.len();
        let dim = self.policy_action_space.dim();
        let mut features = Vec::with_capacity(n * WIDE_FEATURE_DIM);
        let mut outcomes = Vec::with_capacity(n);
        let mut final_hp = Vec::with_capacity(n);
        let mut visits = Vec::with_capacity(n * dim);
        let mut run_ids: Vec<u64> = Vec::with_capacity(n);
        let mut tiers: Vec<u8> = Vec::with_capacity(n);
        let mut q_vals: Vec<f32> = Vec::with_capacity(n);
        let mut q_valid: Vec<u8> = Vec::with_capacity(n);
        let mut legal_mask: Vec<u8> = Vec::with_capacity(n * dim);
        let mut raw_texel: Vec<f32> = Vec::with_capacity(n);
        for entry in self.entries {
            assert!(
                entry.outcome != OUTCOME_UNLABELED,
                "CorpusCollectorWide::into_arrays called with unlabeled entry — \
                 every record_position must be followed by mark_combat_outcome"
            );
            features.extend_from_slice(&entry.features);
            outcomes.push(entry.outcome);
            final_hp.push(entry.final_hp_norm);
            visits.extend_from_slice(entry.visit_distribution.as_ref());
            run_ids.push(entry.run_id_u64);
            tiers.push(entry.encounter_tier_u8);
            q_vals.push(entry.mcts_root_q);
            q_valid.push(entry.mcts_q_valid);
            legal_mask.extend_from_slice(entry.legal_action_mask.as_ref());
            raw_texel.push(entry.raw_texel_score);
        }
        WideArraysWithProvenance {
            features,
            outcomes,
            final_hp_norm: final_hp,
            visit_distribution: visits,
            run_id_u64: run_ids,
            encounter_tier_u8: tiers,
            mcts_root_q: q_vals,
            mcts_q_valid: q_valid,
            legal_action_mask: legal_mask,
            raw_texel_score: raw_texel,
        }
    }

    /// Wave-6 generalist emit (bead `sts2-rs-uqnh`). Materializes
    /// the four base arrays from [`Self::into_arrays_with_extras`]
    /// plus six per-slot id arrays (row-major):
    ///
    /// - `card_ids_hand`     `[N, HAND_SLOT_MAX]`
    /// - `card_ids_draw`     `[N, PILE_SLOT_MAX]`
    /// - `card_ids_discard`  `[N, PILE_SLOT_MAX]`
    /// - `card_ids_exhaust`  `[N, PILE_SLOT_MAX]`
    /// - `monster_slot_ids`  `[N, ENEMY_SLOT_MAX]`
    /// - `relic_set_ids`     `[N, RELIC_SLOT_MAX]`
    ///
    /// Panics if the collector wasn't built with
    /// [`Self::with_embeddings`] (the id payload would be missing on
    /// every row — flunk loudly so the corpus generator can't
    /// silently emit an all-NULL id NPZ).
    pub fn into_arrays_with_embeddings(self) -> WideArraysWithEmbeddings {
        assert!(
            self.with_embeddings,
            "CorpusCollectorWide::into_arrays_with_embeddings called on a \
             collector built without `.with_embeddings()` — id payload \
             would be all-NULL. Build with `CorpusCollectorWide::new().with_embeddings()`."
        );
        let n = self.entries.len();
        let dim = self.policy_action_space.dim();
        let mut features = Vec::with_capacity(n * WIDE_FEATURE_DIM);
        let mut outcomes = Vec::with_capacity(n);
        let mut final_hp = Vec::with_capacity(n);
        let mut visits = Vec::with_capacity(n * dim);
        let mut hand = Vec::with_capacity(n * HAND_SLOT_MAX);
        let mut draw = Vec::with_capacity(n * PILE_SLOT_MAX);
        let mut discard = Vec::with_capacity(n * PILE_SLOT_MAX);
        let mut exhaust = Vec::with_capacity(n * PILE_SLOT_MAX);
        let mut enemies = Vec::with_capacity(n * ENEMY_SLOT_MAX);
        let mut relics = Vec::with_capacity(n * RELIC_SLOT_MAX);
        let mut feats_hand = Vec::with_capacity(n * CARD_FEAT_DIM);
        let mut feats_draw = Vec::with_capacity(n * CARD_FEAT_DIM);
        let mut feats_discard = Vec::with_capacity(n * CARD_FEAT_DIM);
        let mut feats_exhaust = Vec::with_capacity(n * CARD_FEAT_DIM);
        let mut q_vals: Vec<f32> = Vec::with_capacity(n);
        let mut q_valid: Vec<u8> = Vec::with_capacity(n);
        let mut legal_mask: Vec<u8> = Vec::with_capacity(n * dim);
        // Bead `sts2-rs` Phase 2: the embeddings emit now also carries the
        // per-row provenance so the generalist trainer + texel value tuner can
        // do leak-free group splits by run_id (previously only the
        // non-embeddings provenance writer emitted these).
        let mut run_ids: Vec<u64> = Vec::with_capacity(n);
        let mut tiers: Vec<u8> = Vec::with_capacity(n);
        for entry in self.entries {
            assert!(
                entry.outcome != OUTCOME_UNLABELED,
                "CorpusCollectorWide::into_arrays_with_embeddings called with \
                 unlabeled entry — every record_position must be followed by \
                 mark_combat_outcome"
            );
            let ids = entry.embedding_ids.expect(
                "with_embeddings=true but row has no EmbeddingIds payload — \
                 collector state corruption",
            );
            features.extend_from_slice(&entry.features);
            outcomes.push(entry.outcome);
            final_hp.push(entry.final_hp_norm);
            visits.extend_from_slice(entry.visit_distribution.as_ref());
            hand.extend_from_slice(&ids.card_ids_hand);
            draw.extend_from_slice(&ids.card_ids_draw);
            discard.extend_from_slice(&ids.card_ids_discard);
            exhaust.extend_from_slice(&ids.card_ids_exhaust);
            enemies.extend_from_slice(&ids.monster_slot_ids);
            relics.extend_from_slice(&ids.relic_set_ids);
            feats_hand.extend_from_slice(&ids.card_feats_hand);
            feats_draw.extend_from_slice(&ids.card_feats_draw);
            feats_discard.extend_from_slice(&ids.card_feats_discard);
            feats_exhaust.extend_from_slice(&ids.card_feats_exhaust);
            q_vals.push(entry.mcts_root_q);
            q_valid.push(entry.mcts_q_valid);
            legal_mask.extend_from_slice(entry.legal_action_mask.as_ref());
            run_ids.push(entry.run_id_u64);
            tiers.push(entry.encounter_tier_u8);
        }
        WideArraysWithEmbeddings {
            features,
            outcomes,
            final_hp_norm: final_hp,
            visit_distribution: visits,
            card_ids_hand: hand,
            card_ids_draw: draw,
            card_ids_discard: discard,
            card_ids_exhaust: exhaust,
            monster_slot_ids: enemies,
            relic_set_ids: relics,
            card_feats_hand: feats_hand,
            card_feats_draw: feats_draw,
            card_feats_discard: feats_discard,
            card_feats_exhaust: feats_exhaust,
            mcts_root_q: q_vals,
            mcts_q_valid: q_valid,
            legal_action_mask: legal_mask,
            run_id_u64: run_ids,
            encounter_tier_u8: tiers,
        }
    }
}

/// Output bundle from [`CorpusCollectorWide::into_arrays_with_provenance`].
/// Same four base arrays as [`CorpusCollectorWide::into_arrays_with_extras`]
/// plus per-row `run_id_u64` + `encounter_tier_u8`. Bead
/// `sts2-rs-acbj`, step 1.
#[derive(Debug)]
pub struct WideArraysWithProvenance {
    pub features: Vec<f32>,
    pub outcomes: Vec<u8>,
    pub final_hp_norm: Vec<f32>,
    pub visit_distribution: Vec<f32>,
    pub run_id_u64: Vec<u64>,
    pub encounter_tier_u8: Vec<u8>,
    /// Per-row MCTS root Q in `[-1.0, 1.0]`; `f32::NAN` where the
    /// recorder lacked search context. Bead `sts2-rs-wrip2`.
    pub mcts_root_q: Vec<f32>,
    /// 0/1 mask companion to [`Self::mcts_root_q`]. Bead `sts2-rs-wrip2`.
    pub mcts_q_valid: Vec<u8>,
    /// Dense legal-action mask, row-major `[N, dim]` where `dim` is the
    /// per-corpus policy-action-space width (640 canonical / 4672 keyed,
    /// per [`CorpusCollectorWide::policy_action_space`]). `u8` with `1`
    /// at slot `i` iff the corresponding action was legal at record time
    /// (per [`sts2_core::legal_actions`]). Bead `sts2-rs-8bb9i`
    /// (svstv.2) — used by the trainer's `--lambda-illegal-kl` aux loss
    /// to penalize policy-head mass on illegal slots without the
    /// under-counting `visit > 0` proxy.
    pub legal_action_mask: Vec<u8>,
    /// Per-row raw Texel-tuned `evaluate_in_progress(state)` score.
    /// Bead `sts2-rs-aa09l` (37vzt.1) — drives the trainer-side
    /// composed value head (`scripts/value_policy_net.py
    /// --texel-compose`).
    pub raw_texel_score: Vec<f32>,
}

/// Output bundle from [`CorpusCollectorWide::into_arrays_with_embeddings`].
/// Each id vector is row-major; the per-row stride is the
/// corresponding `*_SLOT_MAX` constant. Bead `sts2-rs-uqnh`.
#[derive(Debug)]
pub struct WideArraysWithEmbeddings {
    pub features: Vec<f32>,
    pub outcomes: Vec<u8>,
    pub final_hp_norm: Vec<f32>,
    pub visit_distribution: Vec<f32>,
    pub card_ids_hand: Vec<u16>,
    pub card_ids_draw: Vec<u16>,
    pub card_ids_discard: Vec<u16>,
    pub card_ids_exhaust: Vec<u16>,
    pub monster_slot_ids: Vec<u16>,
    pub relic_set_ids: Vec<u16>,
    /// Per-pile summed [`card_feat_vec`], row-major `[N, CARD_FEAT_DIM]`
    /// each. Lever A stage (b). See [`EmbeddingIds::card_feats_hand`].
    pub card_feats_hand: Vec<f32>,
    pub card_feats_draw: Vec<f32>,
    pub card_feats_discard: Vec<f32>,
    pub card_feats_exhaust: Vec<f32>,
    /// Per-row MCTS root Q in `[-1.0, 1.0]`; `f32::NAN` where the
    /// recorder lacked search context. Bead `sts2-rs-wrip2`.
    pub mcts_root_q: Vec<f32>,
    /// 0/1 mask companion to [`Self::mcts_root_q`]. Bead `sts2-rs-wrip2`.
    pub mcts_q_valid: Vec<u8>,
    /// Dense legal-action mask, row-major `[N, dim]` where `dim` is the
    /// per-corpus policy-action-space width (640 canonical / 4672
    /// keyed), `u8`. Bead `sts2-rs-8bb9i` (svstv.2). See the parallel
    /// [`WideArraysWithProvenance::legal_action_mask`] doc.
    pub legal_action_mask: Vec<u8>,
    /// Per-row FNV-1a 64 of `[provenance].run_id` for leak-free group
    /// splits in the trainer / value tuner. Phase 2: previously emitted
    /// only by the non-embeddings provenance writer; the embeddings emit
    /// now carries it too so the DeepSets generalist trainer can split.
    pub run_id_u64: Vec<u64>,
    /// Per-row encounter tier (0=common, 1=elite, 2=boss). Phase 2.
    pub encounter_tier_u8: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_state() -> CombatState {
        CombatState::default()
    }

    #[test]
    fn fresh_collector_is_empty() {
        let c = CorpusCollector::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn embedding_ids_null_for_empty_state() {
        // Wave-6 (bead sts2-rs-uqnh): an empty state has no cards,
        // no enemies, no relics → every id slot stays at the NULL
        // sentinel (0).
        let ids = EmbeddingIds::from_state(&empty_state());
        assert!(ids.card_ids_hand.iter().all(|&v| v == 0));
        assert!(ids.card_ids_draw.iter().all(|&v| v == 0));
        assert!(ids.card_ids_discard.iter().all(|&v| v == 0));
        assert!(ids.card_ids_exhaust.iter().all(|&v| v == 0));
        assert!(ids.monster_slot_ids.iter().all(|&v| v == 0));
        assert!(ids.relic_set_ids.iter().all(|&v| v == 0));
    }

    #[test]
    fn wide_collector_with_embeddings_emits_id_arrays() {
        // Bead sts2-rs-uqnh: --with-embeddings round-trip.
        let mut c = CorpusCollectorWide::new().with_embeddings();
        assert!(c.captures_embeddings());
        let state = empty_state();
        c.record_position(&state);
        c.record_position(&state);
        c.mark_combat_outcome_with_hp(true, 0.5);
        let bundle = c.into_arrays_with_embeddings();
        assert_eq!(bundle.outcomes, vec![1, 1]);
        assert_eq!(bundle.card_ids_hand.len(), 2 * HAND_SLOT_MAX);
        assert_eq!(bundle.card_ids_draw.len(), 2 * PILE_SLOT_MAX);
        assert_eq!(bundle.card_ids_discard.len(), 2 * PILE_SLOT_MAX);
        assert_eq!(bundle.card_ids_exhaust.len(), 2 * PILE_SLOT_MAX);
        assert_eq!(bundle.monster_slot_ids.len(), 2 * ENEMY_SLOT_MAX);
        assert_eq!(bundle.relic_set_ids.len(), 2 * RELIC_SLOT_MAX);
        // Empty state ⇒ every id stays NULL.
        assert!(bundle.card_ids_hand.iter().all(|&v| v == 0));
        assert!(bundle.monster_slot_ids.iter().all(|&v| v == 0));
        assert!(bundle.relic_set_ids.iter().all(|&v| v == 0));
    }

    #[test]
    fn narrow_collector_provenance_stamped_per_combat() {
        // Bead sts2-rs-e2bjk: narrow-path mirror of the wide
        // provenance test below — set_pending_combat_provenance
        // stamps every subsequent record_position with (run_id,
        // tier) until the next call. Untouched collectors emit
        // the (0, TIER_UNKNOWN) sentinel.
        let mut c = CorpusCollector::new();
        let state = empty_state();

        c.set_pending_combat_provenance(0x1111_2222_3333_4444, TIER_BOSS);
        c.record_position(&state);
        c.record_position(&state);
        c.mark_combat_outcome(true);

        c.set_pending_combat_provenance(0xdead_beef_dead_beef, TIER_ELITE);
        c.record_position(&state);
        c.mark_combat_outcome(false);

        let bundle = c.into_arrays_with_provenance();
        assert_eq!(
            bundle.run_id_u64,
            vec![
                0x1111_2222_3333_4444,
                0x1111_2222_3333_4444,
                0xdead_beef_dead_beef,
            ],
        );
        assert_eq!(
            bundle.encounter_tier_u8,
            vec![TIER_BOSS, TIER_BOSS, TIER_ELITE],
        );
        assert_eq!(bundle.outcomes, vec![1, 1, 0]);
    }

    #[test]
    fn narrow_collector_unset_provenance_is_zero_unknown() {
        // Bead sts2-rs-e2bjk: without a provenance call, every row
        // emits the sentinel pair.
        let mut c = CorpusCollector::new();
        let state = empty_state();
        c.record_position(&state);
        c.mark_combat_outcome(true);
        let bundle = c.into_arrays_with_provenance();
        assert_eq!(bundle.run_id_u64, vec![0]);
        assert_eq!(bundle.encounter_tier_u8, vec![TIER_UNKNOWN]);
    }

    #[test]
    fn legal_action_mask_default_is_all_zeros() {
        // Bead `sts2-rs-8bb9i` (svstv.2): a wide entry constructed via
        // `from_state` (no MCTS search) carries the all-zeros sentinel.
        // Distinct from "mask carried through into_arrays" which only
        // happens via `record_position_with_visits`.
        let entry = CorpusEntryWide::from_state(&empty_state());
        assert_eq!(entry.legal_action_mask.len(), CANONICAL_ACTION_DIM);
        assert!(entry.legal_action_mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn legal_action_mask_round_trips_through_provenance_bundle() {
        // Bead `sts2-rs-8bb9i` (svstv.2): the provenance bundle MUST
        // carry one (N, CANONICAL_ACTION_DIM) u8 row per recorded
        // position. Verified independently of MCTS-search visits — we
        // populate the mask manually then assert round-trip.
        let mut c = CorpusCollectorWide::new();
        c.record_position(&empty_state());
        c.record_position(&empty_state());
        // Stamp a sentinel into both rows so we can detect "all-zeros
        // sentinel survived into the bundle" vs "mask was populated".
        for entry in &mut c.entries {
            entry.legal_action_mask[0] = 1;
            entry.legal_action_mask[CANONICAL_ACTION_DIM - 1] = 1;
        }
        c.mark_combat_outcome_with_hp(true, 0.5);
        let bundle = c.into_arrays_with_provenance();
        assert_eq!(bundle.legal_action_mask.len(), 2 * CANONICAL_ACTION_DIM);
        // First row, slot 0 → 1; row stride is CANONICAL_ACTION_DIM.
        assert_eq!(bundle.legal_action_mask[0], 1);
        assert_eq!(bundle.legal_action_mask[CANONICAL_ACTION_DIM - 1], 1);
        // Second row mirrors the same shape.
        assert_eq!(bundle.legal_action_mask[CANONICAL_ACTION_DIM], 1);
        assert_eq!(bundle.legal_action_mask[2 * CANONICAL_ACTION_DIM - 1], 1);
        // Interior slots in either row remain at sentinel zero.
        assert_eq!(bundle.legal_action_mask[1], 0);
    }

    #[test]
    fn legal_action_mask_round_trips_through_embeddings_bundle() {
        // Bead `sts2-rs-8bb9i` (svstv.2): the embedding bundle MUST
        // also carry the mask. Mirror of the provenance round-trip
        // test above, against the --with-embeddings path.
        let mut c = CorpusCollectorWide::new().with_embeddings();
        c.record_position(&empty_state());
        for entry in &mut c.entries {
            entry.legal_action_mask[7] = 1;
        }
        c.mark_combat_outcome_with_hp(false, 0.0);
        let bundle = c.into_arrays_with_embeddings();
        assert_eq!(bundle.legal_action_mask.len(), CANONICAL_ACTION_DIM);
        assert_eq!(bundle.legal_action_mask[7], 1);
        assert_eq!(bundle.legal_action_mask[6], 0);
    }

    #[test]
    fn wide_collector_provenance_stamped_per_combat() {
        // Bead sts2-rs-acbj (step 1): set_pending_combat_provenance
        // stamps every subsequent record_position with the (run_id,
        // tier) pair until the next call. Untouched collectors
        // emit the (0, TIER_UNKNOWN) sentinel.
        let mut c = CorpusCollectorWide::new();
        let state = empty_state();

        c.set_pending_combat_provenance(0x1111_2222_3333_4444, TIER_BOSS);
        c.record_position(&state); // combat 1: 2 positions
        c.record_position(&state);
        c.mark_combat_outcome_with_hp(true, 0.8);

        c.set_pending_combat_provenance(0xdead_beef_dead_beef, TIER_ELITE);
        c.record_position(&state); // combat 2: 1 position
        c.mark_combat_outcome_with_hp(false, 0.0);

        let bundle = c.into_arrays_with_provenance();
        assert_eq!(
            bundle.run_id_u64,
            vec![
                0x1111_2222_3333_4444,
                0x1111_2222_3333_4444,
                0xdead_beef_dead_beef,
            ],
        );
        assert_eq!(
            bundle.encounter_tier_u8,
            vec![TIER_BOSS, TIER_BOSS, TIER_ELITE],
        );
        assert_eq!(bundle.outcomes, vec![1, 1, 0]);
    }

    #[test]
    fn fnv1a_64_deterministic_and_distinct() {
        // Bead sts2-rs-acbj: FNV-1a is the cross-run-stable hash
        // that group-by-run_id splits need. The two requirements
        // we lean on are (a) determinism (same bytes → same u64
        // across processes) and (b) distinct inputs hash distinct
        // outputs on the toml run_id alphabet.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325); // FNV offset basis — confirms the empty-string sentinel doesn't collide with 0
        let a = fnv1a_64(b"00e4b4d4-fc6c-42ad-a363-fea1dfe39c87");
        let b = fnv1a_64(b"00e4b4d4-fc6c-42ad-a363-fea1dfe39c88");
        assert_ne!(a, b);
        assert_eq!(a, fnv1a_64(b"00e4b4d4-fc6c-42ad-a363-fea1dfe39c87"));
    }

    #[test]
    #[should_panic(expected = "without `.with_embeddings()`")]
    fn into_arrays_with_embeddings_panics_when_capture_disabled() {
        let mut c = CorpusCollectorWide::new();
        c.record_position(&empty_state());
        c.mark_combat_outcome_with_hp(true, 1.0);
        let _ = c.into_arrays_with_embeddings();
    }

    #[test]
    fn record_then_mark_labels_all_pending_entries() {
        let mut c = CorpusCollector::new();
        let state = empty_state();
        c.record_position(&state);
        c.record_position(&state);
        c.mark_combat_outcome(true);
        let (features, outcomes) = c.into_arrays();
        // EVAL_FEATURE_NAMES length grows by 1 per leaf-eval column
        // bead; sourced via the canonical constant so future bumps
        // don't desync this pin (bead `sts2-rs-adulv` was the last
        // bump at the time of writing — 27 → 28).
        assert_eq!(features.len(), 2 * sts2_core::EVAL_FEATURE_NAMES.len());
        assert_eq!(outcomes, vec![1, 1]);
    }

    #[test]
    fn multiple_combats_keep_per_combat_outcomes() {
        let mut c = CorpusCollector::new();
        let state = empty_state();
        c.record_position(&state); // combat 1
        c.mark_combat_outcome(true);
        c.record_position(&state); // combat 2
        c.record_position(&state);
        c.mark_combat_outcome(false);
        let (_, outcomes) = c.into_arrays();
        assert_eq!(outcomes, vec![1, 0, 0]);
    }

    #[test]
    fn filter_round_at_least_drops_low_rounds() {
        let mut c = CorpusCollector::new();
        let mut state = empty_state();
        state.round = 1;
        c.record_position(&state);
        state.round = 3;
        c.record_position(&state);
        state.round = 5;
        c.record_position(&state);
        c.mark_combat_outcome(true);
        let dropped = c.filter_round_at_least(2);
        assert_eq!(dropped, 1);
        assert_eq!(c.len(), 2);
    }

    #[test]
    #[should_panic(expected = "unlabeled entry")]
    fn unlabeled_entries_panic_on_into_arrays() {
        let mut c = CorpusCollector::new();
        let state = empty_state();
        c.record_position(&state); // never marked!
        let _ = c.into_arrays();
    }

    #[test]
    fn wide_collector_emits_80_column_rows() {
        let mut c = CorpusCollectorWide::new();
        let state = empty_state();
        c.record_position(&state);
        c.record_position(&state);
        c.mark_combat_outcome(true);
        let (features, outcomes) = c.into_arrays();
        assert_eq!(features.len(), 2 * WIDE_FEATURE_DIM);
        assert_eq!(outcomes, vec![1, 1]);
    }

    #[test]
    fn wide_collector_filters_low_rounds() {
        let mut c = CorpusCollectorWide::new();
        let mut state = empty_state();
        state.round = 1;
        c.record_position(&state);
        state.round = 4;
        c.record_position(&state);
        c.mark_combat_outcome(false);
        let dropped = c.filter_round_at_least(2);
        assert_eq!(dropped, 1);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn wide_collector_record_with_visits_normalizes_distribution() {
        let mut c = CorpusCollectorWide::new();
        let state = empty_state();
        // 30 + 70 visits across EndTurn and Potion → 0.3 + 0.7.
        let visits = vec![
            (Action::EndTurn, 30u32),
            (
                Action::DrinkPotion {
                    potion_slot: 0,
                    target: None,
                },
                70u32,
            ),
        ];
        c.record_position_with_visits(&state, &visits, Some(0.42));
        c.mark_combat_outcome_with_hp(true, 0.75);
        let (_, _, hp, vd) = c.into_arrays_with_extras();
        assert_eq!(hp.len(), 1);
        assert!((hp[0] - 0.75).abs() < 1e-6);
        assert_eq!(vd.len(), CANONICAL_ACTION_DIM);
        // EndTurn slot (579) → 0.3.
        assert!((vd[crate::action_index::SLOT_END_TURN] - 0.3).abs() < 1e-6);
        // Potion slot (580) → 0.7.
        assert!((vd[crate::action_index::SLOT_POTION] - 0.7).abs() < 1e-6);
        // All other slots → 0.
        let mut other_sum = 0.0f32;
        for (i, x) in vd.iter().enumerate() {
            if i != crate::action_index::SLOT_END_TURN && i != crate::action_index::SLOT_POTION {
                other_sum += x;
            }
        }
        assert!(other_sum.abs() < 1e-6);
    }

    #[test]
    fn wide_collector_hp_backfill_clamps_to_unit_range() {
        let mut c = CorpusCollectorWide::new();
        let state = empty_state();
        c.record_position(&state);
        c.mark_combat_outcome_with_hp(true, 1.5); // out-of-range — clamp
        let (_, _, hp, _) = c.into_arrays_with_extras();
        assert!((hp[0] - 1.0).abs() < 1e-6);

        let mut c2 = CorpusCollectorWide::new();
        c2.record_position(&state);
        c2.mark_combat_outcome_with_hp(false, -0.2);
        let (_, _, hp2, _) = c2.into_arrays_with_extras();
        assert!(hp2[0].abs() < 1e-6);
    }

    #[test]
    fn wide_collector_into_arrays_backcompat_skips_extras() {
        // The legacy two-array path still works — qctu.2-era callers
        // that don't ingest aux/policy targets stay byte-stable.
        let mut c = CorpusCollectorWide::new();
        let state = empty_state();
        c.record_position(&state);
        c.mark_combat_outcome(true);
        let (features, outcomes) = c.into_arrays();
        assert_eq!(features.len(), WIDE_FEATURE_DIM);
        assert_eq!(outcomes, vec![1]);
    }
}
