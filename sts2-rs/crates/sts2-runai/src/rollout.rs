//! Tier-2/3 rollout grounding: `fight_delta` — paired synthetic-seed
//! combat probes of a candidate deck/relic mutation (full-run-AI plan
//! §5/M2, §2.5 rule 3; bead `sts2-rs-40ikw.6`).
//!
//! # What a probe is
//!
//! Clone the [`RunManager`], apply a [`CandidateMutation`] to its
//! `run_state`, then simulate the next elite and the act boss as
//! **standalone combats** against the candidate deck/relics/HP, K
//! synthetic seeds each, MCTS at a probe budget per move, truncated at
//! T player turns. The aggregate [`DeltaStats`] is the rollout score a
//! tier-2/3 escalation or the offline labeler ranks candidates by.
//!
//! # Combat construction path (design decision)
//!
//! The probe builds combats the same way the live run loop does
//! (`run_callbacks::build_combat_for_room`), via
//! [`sts2_content::run_bridge::enter_combat_from_roster`] — the one
//! seam that wires deck → draw pile, relic mirror, potion belt, act
//! snapshot, ascension and the Setup→PlayerPlayPhase advance. The only
//! differences from the live path are deliberate (§2.5 rule 3):
//!
//! 1. **Synthetic RNG.** The cloned `RunState`'s `rng` is replaced with
//!    `RunRngSet::new(synthetic_seed)` and the roster randomizer rolls
//!    off `Rng::with_name(synthetic_seed, enc.id)` — never the run's
//!    actual upcoming streams, so the agent can't become clairvoyant
//!    about its real next fight.
//! 2. **Encounter identity** comes from an [`EncounterPlan`] computed
//!    once per decision: the elite is sampled uniformly from the act's
//!    elite pool (player-visible information only), the boss is the
//!    map-pinned boss. The same plan is shared by every candidate at a
//!    decision.
//!
//! # Seed pairing (design decision)
//!
//! Probe seed k for encounter E is `synthetic_seed(E.id, k)` — a pure
//! function of the encounter identity and the probe index. It does NOT
//! depend on the candidate (pairing across candidates, §2.5 rule 3),
//! the decision, or the run seed (a *fixed panel* of K seeds, which
//! also makes the fight-strength cache sound: a cache hit returns
//! stats computed under exactly the seeds a recompute would use).
//!
//! # Fight-strength cache
//!
//! Keyed `(deck_hash, relics_hash, hp_bucket, encounter, budget)`
//! (plan §5/M2) behind a `Mutex<LruCache>` with bounded capacity. HP is
//! bucketed ([`HP_BUCKET_WIDTH`]) and the probe runs at the bucket's
//! representative HP, so identical keys always mean identical probe
//! inputs (determinism survives cache hits *and* misses).

use std::num::NonZeroUsize;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use sts2_content::encounters::{EncounterDef, EncounterTier};
use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_content::run_bridge::enter_combat_from_roster_in_slots;
use sts2_core::cards::CardId;
use sts2_core::dotnet_rng::Rng;
use sts2_core::map::MapPointType;
use sts2_core::potions::PotionKind;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::run_rng_set::RunRngSet;
use sts2_core::runstate::RunState;
use sts2_core::{legal_actions, MonsterKind, RelicId};
use sts2_mcts::{Mcts, MctsConfig};

use crate::policy::LeafKind;
use crate::view::VisibleRunView;

/// HP-bucket width for the fight-strength cache key. Probes run at the
/// bucket's representative HP (midpoint, clamped to `[1, max_hp]`), so
/// two states in the same bucket share both the key and the probe
/// inputs — the plan's intended approximation, made deterministic.
pub const HP_BUCKET_WIDTH: i32 = 5;

/// Hard cap on moves per probe so a pathological combat can't spin the
/// labeler (loops *inside* one `apply` are the separate 65mbx bug).
const MAX_PROBE_MOVES: u32 = 400;

/// One candidate mutation of the run state — the unit `fight_delta`
/// scores. Mirrors M1's pure `VisibleRunView::with_*` helpers, but M2
/// applies it to a CLONED `RunManager`'s `run_state` so the probe
/// fights the real mutated deck.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateMutation {
    /// No-op (the "skip" / status-quo arm every comparison needs).
    Baseline,
    /// Append `card_id` to the master deck tail (C# deck-adds append).
    AddCard { card_id: CardId, upgraded: bool },
    /// Remove master-deck index `deck_idx` (also the transform-target
    /// stand-in: transform = remove + unknown replacement).
    RemoveCard { deck_idx: usize },
    /// Upgrade master-deck index `deck_idx` in place.
    UpgradeCard { deck_idx: usize },
    /// Append `relic_id` to the relic list.
    AddRelic { relic_id: RelicId },
    /// Adjust gold by `delta` (clamped at 0). Combat-inert — exists so
    /// gold-shaped options share the probe surface (and as the pairing
    /// test's control arm).
    GoldDelta { delta: i32 },
    /// Append a potion to the belt.
    AddPotion { kind: PotionKind },
    /// Heal `amount` HP (clamped at max HP) — the rest-site heal arm
    /// (M2.5 `rest_delta`); combat-relevant only via the probe's
    /// starting HP bucket.
    HealHp { amount: i32 },
}

impl CandidateMutation {
    /// Apply to a (cloned) manager's `run_state`. Returns `false` when
    /// the mutation couldn't apply (bad index / unknown card spec).
    pub fn apply(&self, rm: &mut RunManager) -> bool {
        let rs = &mut rm.run_state;
        match *self {
            CandidateMutation::Baseline => true,
            CandidateMutation::AddCard { card_id, upgraded } => {
                match sts2_content::cards::v0111::card_pod_for(card_id, upgraded, rs.game_version) {
                    Some(card) => {
                        rs.master_deck.push(card);
                        true
                    }
                    None => false,
                }
            }
            CandidateMutation::RemoveCard { deck_idx } => {
                if deck_idx < rs.master_deck.len() {
                    rs.master_deck.remove(deck_idx);
                    true
                } else {
                    false
                }
            }
            CandidateMutation::UpgradeCard { deck_idx } => match rs.master_deck.get_mut(deck_idx) {
                Some(card) => {
                    let _ = card.upgrade_for(rs.game_version);
                    true
                }
                None => false,
            },
            CandidateMutation::AddRelic { relic_id } => {
                rs.relics.push(relic_id);
                true
            }
            CandidateMutation::GoldDelta { delta } => {
                rs.gold = (rs.gold + delta).max(0);
                true
            }
            CandidateMutation::AddPotion { kind } => {
                rs.potions.push(kind);
                true
            }
            CandidateMutation::HealHp { amount } => {
                rs.current_hp = (rs.current_hp + amount).min(rs.max_hp);
                true
            }
        }
    }

    /// Apply to a [`VisibleRunView`] (feature extraction for labels —
    /// same semantics as [`Self::apply`], via the pure M1 helpers).
    #[must_use]
    pub fn apply_to_view(&self, view: &VisibleRunView) -> VisibleRunView {
        match *self {
            CandidateMutation::Baseline => view.clone(),
            CandidateMutation::AddCard { card_id, upgraded } => {
                view.with_card_added(card_id, upgraded)
            }
            CandidateMutation::RemoveCard { deck_idx } => view.with_deck_card_removed(deck_idx),
            CandidateMutation::UpgradeCard { deck_idx } => view.with_deck_card_upgraded(deck_idx),
            CandidateMutation::AddRelic { relic_id } => view.with_relic_added(relic_id),
            CandidateMutation::GoldDelta { delta } => view.with_gold_delta(delta),
            CandidateMutation::AddPotion { kind } => view.with_potion_added(kind),
            CandidateMutation::HealHp { amount } => view.with_hp(view.hp + amount),
        }
    }

    /// Stable human-readable description for label rows.
    #[must_use]
    pub fn desc(&self) -> String {
        match *self {
            CandidateMutation::Baseline => "baseline".to_string(),
            CandidateMutation::AddCard { card_id, upgraded } => {
                format!("add:{card_id:?}{}", if upgraded { "+" } else { "" })
            }
            CandidateMutation::RemoveCard { deck_idx } => format!("remove:{deck_idx}"),
            CandidateMutation::UpgradeCard { deck_idx } => format!("upgrade:{deck_idx}"),
            CandidateMutation::AddRelic { relic_id } => format!("relic:{relic_id:?}"),
            CandidateMutation::GoldDelta { delta } => format!("gold:{delta:+}"),
            CandidateMutation::AddPotion { kind } => format!("potion:{kind:?}"),
            CandidateMutation::HealHp { amount } => format!("heal:+{amount}"),
        }
    }
}

/// Which combat horizon a labeling campaign scores card options against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabelHorizon {
    /// Two independent fights: sampled elite + pinned boss (legacy `fight_delta`).
    #[default]
    ElitePlusBoss,
    /// Chained `[normal×chain_normals, elite, boss]` with HP carryover (`chain_delta`).
    ChainedAct,
}

/// Probe search budget (100-sim MCTS per move, K=16 paired seeds, T=8
/// player-turn truncation). The horizon default is **8**, not the plan's
/// original §5/M2 T=5: the E5 calibration (`docs/experiments/2026-06-12-
/// full-run-e0-baselines.md`) refuted T=5 — it flips 28% of top card-picks
/// vs full-fight scoring — and adopted T=8 as the shortest horizon with
/// Kendall τ ≥ 0.85 against full fights. (`chain_delta` goes further and
/// runs full fights, `t_turns = 0`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeBudget {
    /// MCTS simulations per probe move.
    pub sims: u32,
    /// Paired synthetic seeds per encounter.
    pub k: u32,
    /// Player-turn truncation horizon; `0` = full fight.
    pub t_turns: u32,
    /// Probe leaf evaluator.
    pub leaf: LeafKind,
    /// Labeling horizon (which probe the labeler uses). Default = legacy.
    pub horizon: LabelHorizon,
    /// `ChainedAct`: sampled normal-tier fights prepended before elite+boss.
    pub chain_normals: usize,
}

impl Default for ProbeBudget {
    fn default() -> Self {
        Self {
            sims: 100,
            k: 16,
            t_turns: 8, // E5-calibrated horizon (T=5 refuted; see struct docs)
            leaf: LeafKind::TexelWide,
            horizon: LabelHorizon::default(),
            chain_normals: 0,
        }
    }
}

impl ProbeBudget {
    /// Cache-key form.
    fn key(&self) -> (u32, u32, u32, u8, u8, u32) {
        let leaf = match self.leaf {
            LeafKind::Texel => 0,
            LeafKind::TexelWide => 1,
        };
        let horizon = match self.horizon {
            LabelHorizon::ElitePlusBoss => 0,
            LabelHorizon::ChainedAct => 1,
        };
        (
            self.sims,
            self.k,
            self.t_turns,
            leaf,
            horizon,
            self.chain_normals as u32,
        )
    }
}

/// The encounters one decision's candidates are probed against —
/// computed ONCE per decision and shared by every candidate (§2.5
/// rule 3: identical encounter identities across candidates).
#[derive(Clone, Copy, Debug)]
pub struct EncounterPlan {
    /// Sampled elite identity (uniform over the act's elite pool).
    pub elite: Option<&'static EncounterDef>,
    /// Map-pinned act boss (player-visible from the map screen).
    pub boss: Option<&'static EncounterDef>,
}

/// Per-act encounter table (1-based act index, the `RunState::act_index`
/// convention). Mirrors `run_callbacks::act_encounter_table`.
#[must_use]
pub fn act_encounter_table(
    act_index: u8,
    act1: sts2_core::act_selection::ActVariant,
) -> Option<&'static [EncounterDef]> {
    match act_index {
        // GH #544: act 1 is two different acts. This planner projects what the
        // agent is likely to FIGHT next, so on an Underdocks run the Overgrowth
        // table made every projection about encounters the run cannot meet.
        1 => Some(sts2_content::encounters::act1_encounters(act1)),
        2 => Some(sts2_content::encounters::act2_hive::ACT2_ENCOUNTERS),
        3 => Some(sts2_content::encounters::act3_glory::ACT3_ENCOUNTERS),
        _ => None,
    }
}

impl EncounterPlan {
    /// Build the plan for one decision. `decision_key` seeds the elite
    /// identity sample (so two decisions don't always probe the same
    /// elite) — it must be the same value for every candidate at the
    /// decision and MUST NOT feed the combat seeds (those are the fixed
    /// global panel, see the module docs).
    ///
    /// Elite pool: the act's `EncounterTier::Elite` rows, uniform — the
    /// player-visible "remaining pool" approximation (fought-elite
    /// identities aren't tracked per-id on `RunState`; the generated
    /// elite list is cursor-hidden information we must not read). Act 1
    /// has no Elite tier at this pin → fall back to Normal-tier rows,
    /// mirroring the engine's own act-1 fallback. Boss identity: the
    /// generated `boss_id` (pinned + displayed on the map screen), with
    /// the `pinned_act_boss.encounter_index` fallback for raw-core runs.
    #[must_use]
    pub fn for_decision(rm: &RunManager, decision_key: u64) -> Self {
        let rs = &rm.run_state;
        let act = rs.act_index;
        let Some(table) = act_encounter_table(act, rs.act1) else {
            return Self {
                elite: None,
                boss: None,
            };
        };

        let elite_pool: Vec<&'static EncounterDef> = {
            let elites: Vec<&'static EncounterDef> = table
                .iter()
                .filter(|e| e.tier == EncounterTier::Elite)
                .collect();
            if elites.is_empty() {
                table
                    .iter()
                    .filter(|e| e.tier == EncounterTier::Normal)
                    .collect()
            } else {
                elites
            }
        };
        let elite = if elite_pool.is_empty() {
            None
        } else {
            let idx =
                (splitmix64(decision_key ^ 0xE11E_0000_0000_0001) as usize) % elite_pool.len();
            Some(elite_pool[idx])
        };

        // Boss: at-second-boss handling deliberately omitted (A10 probe
        // fidelity follow-up); the first generated boss is the pin shown
        // on the map.
        let gen_boss = (act as usize)
            .checked_sub(1)
            .and_then(|i| rs.generated_rooms.get(i))
            .and_then(|g| sts2_content::encounters::find(g.boss_id));
        let boss = gen_boss.or_else(|| {
            rs.pinned_act_boss.as_ref().and_then(|pin| {
                table
                    .get(pin.encounter_index)
                    .filter(|e| e.tier == EncounterTier::Boss)
            })
        });

        // Experiment flag (2026-06-25 proxy-fidelity study): drop the elite
        // leg so the panel is boss-only. The MI study found the elite probe
        // adds ≈0 marginal conditional MI about run outcome at ~half the
        // panel cost; this gates the A/B that tests whether dropping it
        // moves win rate. See docs/plans/2026-06-25-proxy-fidelity-mi-study.md.
        let elite = if std::env::var_os("STS2_RUNAI_PROXY_BOSS_ONLY").is_some() {
            None
        } else {
            elite
        };

        Self { elite, boss }
    }
}

/// Chained-horizon plan: sampled normal fights prepended to the
/// elite+boss `EncounterPlan` (used by [`chain_delta`]). Normals are
/// sampled under the same §2.5 information discipline as the elite —
/// room types are player-visible, identities are sampled per decision
/// and shared across candidates, never reading the run's cursor.
#[derive(Clone, Debug)]
pub struct ChainPlan {
    pub normals: Vec<&'static EncounterDef>,
    pub base: EncounterPlan,
}

impl ChainPlan {
    #[must_use]
    pub fn for_decision(rm: &RunManager, decision_key: u64, n_normals: usize) -> Self {
        let base = EncounterPlan::for_decision(rm, decision_key);
        let act = rm.run_state.act_index;
        let normals: Vec<&'static EncounterDef> = match act_encounter_table(act, rm.run_state.act1)
        {
            None => Vec::new(),
            Some(table) => {
                let pool: Vec<&'static EncounterDef> = table
                    .iter()
                    .filter(|e| e.tier == EncounterTier::Normal)
                    .collect();
                if pool.is_empty() {
                    Vec::new()
                } else {
                    (0..n_normals)
                        .map(|i| {
                            let salt = 0x510E_0000_0000_0000
                                ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            let idx = (splitmix64(decision_key ^ salt) as usize) % pool.len();
                            pool[idx]
                        })
                        .collect()
                }
            }
        };
        Self { normals, base }
    }

    #[must_use]
    pub fn sequence(&self) -> Vec<&'static EncounterDef> {
        let mut seq = self.normals.clone();
        seq.extend(self.base.elite);
        seq.extend(self.base.boss);
        seq
    }
}

/// Truncated-horizon probe aggregate over one candidate's elite + boss
/// fights (plan §5/M2). `win_rate` is over probes that reached a
/// terminal state within the horizon; the means are over all probes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeltaStats {
    /// Wins / terminal probes (0.0 when no probe terminated).
    pub win_rate: f64,
    /// Mean player HP lost per probe (can go negative with in-combat
    /// healing).
    pub hp_lost_mean: f64,
    /// Mean enemy HP destroyed per probe within the horizon.
    pub dmg_dealt_t5_mean: f64,
    /// Mean final combat round per probe.
    pub turns_mean: f64,
    /// Total probes run (fights × K).
    pub probes: u32,
    /// Probes that reached a terminal combat state within the horizon.
    pub terminals: u32,
    /// Probes lost (player died) — kept so `win_rate` over truncated
    /// probes can't hide deaths.
    pub losses: u32,
    /// Population standard deviation of per-probe HP lost. 0.0 for a single
    /// probe. This is the distributional signal a surrogate needs in order to
    /// model catastrophes rather than averages — see `FightStats::hp_lost_sq_sum`.
    pub hp_lost_std: f64,
}

impl DeltaStats {
    /// Scalar ranking blend for tier-2/3 candidate comparison and the
    /// E3/E4/E5 calibration rankings. Hand-set M2 blend: winning
    /// dominates, then HP economy, then damage tempo. Tuned later with
    /// the E-experiments — keep deterministic and monotone.
    #[must_use]
    pub fn score(&self) -> f64 {
        let loss_rate = if self.probes == 0 {
            0.0
        } else {
            f64::from(self.losses) / f64::from(self.probes)
        };
        self.win_rate - loss_rate + 0.002 * self.dmg_dealt_t5_mean - 0.004 * self.hp_lost_mean
    }
}

/// Per-encounter probe aggregate (the cache value).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FightStats {
    pub probes: u32,
    pub terminals: u32,
    pub wins: u32,
    pub losses: u32,
    pub hp_lost_sum: f64,
    /// Sum of `hp_lost²` across probes, so the SPREAD of outcomes survives
    /// aggregation instead of collapsing into the mean.
    ///
    /// A mean is the wrong summary for an absorbing process. Death is a tail
    /// event, so a surrogate fit to `hp_lost_sum / probes` reproduces average
    /// damage and no catastrophes: measured 2026-08-13, a mean-trained
    /// surrogate gave 0.68 run survival against a 0.29 reference while its
    /// per-fight MAE looked fine. Stored as a raw second moment rather than a
    /// variance so `merge` stays associative and the struct stays `Copy`.
    pub hp_lost_sq_sum: f64,
    pub dmg_dealt_sum: f64,
    pub turns_sum: f64,
}

impl FightStats {
    fn add(&mut self, o: &ProbeOutcome) {
        self.probes += 1;
        if o.terminal {
            self.terminals += 1;
            if o.won {
                self.wins += 1;
            } else {
                self.losses += 1;
            }
        }
        self.hp_lost_sum += f64::from(o.hp_lost);
        self.hp_lost_sq_sum += f64::from(o.hp_lost) * f64::from(o.hp_lost);
        self.dmg_dealt_sum += f64::from(o.dmg_dealt);
        self.turns_sum += f64::from(o.turns);
    }

    fn merge(&mut self, other: &FightStats) {
        self.probes += other.probes;
        self.terminals += other.terminals;
        self.wins += other.wins;
        self.losses += other.losses;
        self.hp_lost_sum += other.hp_lost_sum;
        self.hp_lost_sq_sum += other.hp_lost_sq_sum;
        self.dmg_dealt_sum += other.dmg_dealt_sum;
        self.turns_sum += other.turns_sum;
    }

    fn into_delta(self) -> DeltaStats {
        let probes = f64::from(self.probes.max(1));
        DeltaStats {
            win_rate: if self.terminals == 0 {
                0.0
            } else {
                f64::from(self.wins) / f64::from(self.terminals)
            },
            hp_lost_mean: self.hp_lost_sum / probes,
            dmg_dealt_t5_mean: self.dmg_dealt_sum / probes,
            turns_mean: self.turns_sum / probes,
            probes: self.probes,
            terminals: self.terminals,
            losses: self.losses,
            // Var = E[x²] - E[x]². Clamped at 0: the identity is exact in
            // reals but floating-point cancellation can make it slightly
            // negative when every probe lands on the same value.
            hp_lost_std: {
                let mean = self.hp_lost_sum / probes;
                (self.hp_lost_sq_sum / probes - mean * mean).max(0.0).sqrt()
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ProbeOutcome {
    terminal: bool,
    won: bool,
    hp_lost: i32,
    dmg_dealt: i32,
    turns: u16,
    /// Player HP at probe end (for HP-carryover chaining in `chain_delta`).
    exit_hp: i32,
}

/// Fight-strength cache key (plan §5/M2): most candidates at a decision
/// share everything but one card, and the same post-pick state recurs
/// across the labeler/calibrator — hash the mutated state, not the
/// candidate.
///
/// `context_hash` extends the plan's `(deck, relics, hp_bucket,
/// encounter, budget)` sketch with everything ELSE
/// `enter_combat_from_roster` reads from the run state (max HP, potion
/// belt, relic counters, ascension, character, act). Without it, two
/// decision points sharing a deck/relics/HP-bucket but differing in
/// e.g. potions poisoned each other's entries — caught by the
/// calibrate smoke as a jobs-dependent τ<1 on the identical-budget
/// control arm.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FightKey {
    deck_hash: u64,
    relics_hash: u64,
    context_hash: u64,
    hp_bucket: i32,
    encounter: &'static str,
    budget: (u32, u32, u32, u8, u8, u32),
}

/// Bounded LRU fight-strength cache (`lru` crate behind a `Mutex` —
/// plan §5/M2: "dashmap alone has no eviction").
#[derive(Debug)]
pub struct FightCache {
    inner: Mutex<lru::LruCache<FightKey, FightStats>>,
}

impl Default for FightCache {
    fn default() -> Self {
        Self::with_capacity(4096)
    }
}

impl FightCache {
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(cap.max(1)).expect("max(1) is non-zero"),
            )),
        }
    }

    /// Entries currently cached (test/diagnostic surface).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("fight cache lock").len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: &FightKey) -> Option<FightStats> {
        self.inner
            .lock()
            .expect("fight cache lock")
            .get(key)
            .copied()
    }

    fn put(&self, key: FightKey, stats: FightStats) {
        self.inner.lock().expect("fight cache lock").put(key, stats);
    }
}

/// splitmix64 finalizer (same constants as `policy::splitmix64`; local
/// copy keeps the module self-contained for seed-derivation review).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// FNV-1a over the encounter id (stable across builds, unlike
/// `DefaultHasher`).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The fixed synthetic seed panel (§2.5 rule 3): seed k for encounter
/// `enc_id`. A pure function of `(encounter, k)` — never of the
/// candidate (pairing), the decision, or the run seed (cache
/// soundness).
#[must_use]
pub fn synthetic_seed(enc_id: &str, k: u32) -> i32 {
    // Fixed panel salt ("seed-of-paired-panel").
    let mixed = splitmix64(fnv1a(enc_id) ^ (u64::from(k) << 32) ^ 0x5EED_0FAB_1E5A_17E0);
    // Truncate to i32 (the engine's seed width).
    mixed as u32 as i32
}

/// Deck hash: order-sensitive (master-deck order feeds the combat-start
/// shuffle, so order is behaviorally significant) and FULL-POD (two
/// cards with the same `(card_id, upgraded)` can carry different
/// behavior-relevant pod fields — e.g. the run-manager pick path and
/// `spec_index::card_pod_from_id` produce different
/// `target_type`/`base_damage` pods for Conflagration, observably
/// changing probe outcomes; an id-only hash aliased those decks and
/// poisoned the cache). The volatile `uid` is neutralized so logically
/// identical decks built at different times still share entries.
fn deck_hash(rs: &RunState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for card in &rs.master_deck {
        let mut c = *card;
        c.uid = sts2_core::CardUid::new(1).expect("non-zero");
        h = splitmix64(h ^ fnv1a(&format!("{c:?}")));
    }
    h
}

fn relics_hash(rs: &RunState) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    for r in &rs.relics {
        h = splitmix64(h ^ fnv1a(&format!("{r:?}")));
    }
    h
}

/// Everything else the bridge carries into a probe combat: max HP, the
/// potion belt (drinkable in-probe!), per-relic counters, potion-slot
/// cap, ascension, character class, act snapshot. See `FightKey` docs.
fn context_hash(rs: &RunState) -> u64 {
    let mut h: u64 = 0x6a09_e667_f3bc_c909;
    h = splitmix64(h ^ u64::from(rs.max_hp.unsigned_abs()));
    h = splitmix64(h ^ u64::from(rs.ascension_level));
    h = splitmix64(h ^ fnv1a(&format!("{:?}", rs.character_id)));
    h = splitmix64(h ^ u64::from(rs.act_index));
    h = splitmix64(h ^ u64::from(rs.max_potion_slots.unsigned_abs()));
    for p in &rs.potions {
        h = splitmix64(h ^ fnv1a(&format!("{p:?}")));
    }
    // Relic counters in owned-relic order (cross-combat countdown /
    // accumulator state observable in the probe — EmberTea, Nunchaku…).
    for r in &rs.relics {
        let counter = rs.relic_counters.get(r).copied().unwrap_or(i32::MIN);
        h = splitmix64(h ^ u64::from(counter.unsigned_abs()) ^ (u64::from(counter < 0) << 63));
    }
    h
}

/// Install the process-global content registries the probe combats
/// need (idempotent; mirrors the run-manager constructors + the
/// MCTS-search test in `run_bridge.rs`).
///
/// PUBLIC + called up-front by every arena/label/export cell (bead
/// `sts2-rs-40ikw.7`): the run constructors install only the card
/// dispatch, so before this fix the FIRST `fight_delta` probe flipped
/// the potion/relic registries on MID-RUN and perturbed every run
/// already in flight (observed as a labeling run diverging from its
/// own replay at the next shop). Installing everything before the run
/// starts makes trajectories reproducible.
pub fn ensure_content_installed() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        sts2_content::cards::dispatch::ensure_installed();
        sts2_content::relics::dispatch::ensure_installed();
        sts2_content::potions::dispatch::ensure_installed();
        sts2_core::relic_effects::ensure_installed();
    });
}

/// Snap HP to its cache bucket's representative value (midpoint,
/// clamped to `[1, max_hp]`).
fn bucket_and_rep_hp(hp: i32, max_hp: i32) -> (i32, i32) {
    let bucket = hp.max(1) / HP_BUCKET_WIDTH;
    let rep = (bucket * HP_BUCKET_WIDTH + HP_BUCKET_WIDTH / 2).clamp(1, max_hp.max(1));
    (bucket, rep)
}

// Enemy `current_hp` at or above `SENTINEL_HP_MIN` is a beaten sentinel (the
// exploding Waterfall Giant's `999_999_999`; `Creature::is_beaten_sentinel`)
// — no real enemy approaches it. Excluded from probe damage accounting so it
// can't inflate `DeltaStats::dmg_dealt_t5_mean`.
use sts2_core::SENTINEL_HP_MIN;

/// Enemy HP counted toward probe damage: clamps negatives to 0 and
/// drops sentinel/Door creatures (>= `SENTINEL_HP_MIN`) entirely.
fn live_enemy_hp(current_hp: i32) -> i32 {
    let h = current_hp.max(0);
    if h >= SENTINEL_HP_MIN {
        0
    } else {
        h
    }
}

/// Run one truncated synthetic-seed combat probe.
fn probe_one(
    rs_base: &RunState,
    enc: &'static EncounterDef,
    entry_hp: i32,
    seed: i32,
    budget: &ProbeBudget,
) -> ProbeOutcome {
    ensure_content_installed();

    // §2.5 rule 3: replace the run's RNG wholesale with the synthetic
    // set — the probe must not read (or advance a clone of) the run's
    // actual upcoming streams.
    let mut rs = rs_base.clone();
    rs.rng = RunRngSet::new(seed);
    rs.current_hp = entry_hp.clamp(1, rs.max_hp.max(1));

    // Roster identity + composition off the synthetic streams, the
    // same two-stream shape as the live builder
    // (`run_callbacks::build_combat_for_room`).
    let mut enc_rng = Rng::with_name(seed, enc.id);
    let roster_kinds = enc.roster(&mut enc_rng);
    let kinds: Vec<MonsterKind> = roster_kinds.iter().map(|&(k, _)| k).collect();
    let hps = sts2_core::monsters::roll_unique_max_hps_for(
        &kinds,
        &mut rs.rng.niche,
        rs.ascension_level,
        rs.game_version,
    );
    let roster: Vec<(MonsterKind, MonsterSpawnArgs, i32)> = roster_kinds
        .iter()
        .zip(hps)
        .map(|(&(kind, args), hp)| (kind, args, hp))
        .collect();

    // Synthetic what-if probe: no map/room context, so the room-type flags
    // stay unset (the prior `..default()` behavior).
    // The encounter's slot model rides in like the live builder's, so a
    // summon (also one inside entry) lands where C# puts it.
    let mut combat = enter_combat_from_roster_in_slots(
        &rs,
        &roster,
        enc.member_slots(),
        sts2_core::runstate::RoomType::Unknown,
    );
    let start_hp = combat.player.current_hp;
    let enemy_total = |c: &sts2_core::CombatState| -> i32 {
        c.enemy_uids
            .iter()
            .filter_map(|uid| c.creatures.iter().find(|cr| cr.id == uid.0))
            .map(|cr| live_enemy_hp(cr.current_hp))
            .sum()
    };
    let start_enemy = enemy_total(&combat);

    let mut moves: u32 = 0;
    while !combat.is_over {
        if budget.t_turns > 0 && u32::from(combat.round) > budget.t_turns {
            break;
        }
        if moves >= MAX_PROBE_MOVES {
            break;
        }
        let acts = legal_actions(&combat);
        if acts.is_empty() {
            break;
        }
        let action = if acts.len() == 1 {
            acts[0].clone()
        } else {
            // Per-move search seed: deterministic in (probe seed, move
            // index) — still candidate-independent.
            let search_seed = splitmix64(
                u64::from(u32::from_ne_bytes(seed.to_ne_bytes()))
                    ^ (u64::from(moves + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let config = MctsConfig {
                max_simulations: budget.sims,
                seed: search_seed,
                max_rollout_depth: crate::policy::COMBAT_ROLLOUT_DEPTH,
                ..MctsConfig::default()
            };
            let mut mcts = Mcts::new(combat.clone(), config)
                .with_leaf_eval(crate::policy::resolve_leaf_eval(budget.leaf));
            mcts.search_n(budget.sims)
        };
        let _ = sts2_core::apply(&mut combat, action);
        moves += 1;
    }

    ProbeOutcome {
        terminal: combat.is_over,
        won: combat.is_over && combat.player_won,
        hp_lost: start_hp - combat.player.current_hp,
        dmg_dealt: (start_enemy - enemy_total(&combat)).max(0),
        turns: combat.round,
        exit_hp: combat.player.current_hp,
    }
}

/// Probe one encounter K times on the fixed synthetic seed panel.
fn probe_encounter(rs: &RunState, enc: &'static EncounterDef, budget: &ProbeBudget) -> FightStats {
    let mut stats = FightStats::default();
    for k in 0..budget.k {
        let outcome = probe_one(rs, enc, rs.current_hp, synthetic_seed(enc.id, k), budget);
        stats.add(&outcome);
    }
    stats
}

/// Tier-2/3 rollout evaluation of one candidate (plan §5/M2).
///
/// Clones `rm`, applies `candidate` to the clone's `run_state`, snaps
/// HP to its cache bucket, then probes the plan's elite + boss as
/// standalone synthetic-seed combats (K paired seeds each, MCTS at
/// `budget.sims` per move, truncated at `budget.t_turns` player turns)
/// and aggregates into [`DeltaStats`]. Deterministic given
/// `(rm state, candidate, plan, budget)`; paired across candidates by
/// construction (see module docs).
#[must_use]
pub fn fight_delta(
    rm: &RunManager,
    candidate: &CandidateMutation,
    encounters: &EncounterPlan,
    budget: &ProbeBudget,
    cache: &FightCache,
) -> DeltaStats {
    let mut probe_rm = rm.clone();
    let _ = candidate.apply(&mut probe_rm);
    // Probes are standalone fights — never the live combat.
    probe_rm.combat = None;

    let (hp_bucket, rep_hp) =
        bucket_and_rep_hp(probe_rm.run_state.current_hp, probe_rm.run_state.max_hp);
    probe_rm.run_state.current_hp = rep_hp;

    let rs = &probe_rm.run_state;
    let dh = deck_hash(rs);
    let rh = relics_hash(rs);
    let ch = context_hash(rs);

    let mut total = FightStats::default();
    for enc in [encounters.elite, encounters.boss].into_iter().flatten() {
        let key = FightKey {
            deck_hash: dh,
            relics_hash: rh,
            context_hash: ch,
            hp_bucket,
            encounter: enc.id,
            budget: budget.key(),
        };
        let stats = if let Some(hit) = cache.get(&key) {
            hit
        } else {
            let computed = probe_encounter(rs, enc, budget);
            cache.put(key, computed);
            computed
        };
        total.merge(&stats);
    }
    total.into_delta()
}

/// Per-COMBAT surrogate label: probe exactly ONE named encounter against
/// the (mutated) run state and return its single-fight [`DeltaStats`].
///
/// Unlike [`fight_delta`] (elite + boss) and [`chain_delta`] (the act
/// chain), this scores a single encounter — the `(pre-combat state,
/// encounter) → outcome` unit the learned combat surrogate regresses on.
/// Probes run as FULL fights (`t_turns = 0`) so `win_rate` and
/// `hp_lost_mean` are decided combat outcomes, not horizon-truncated.
/// Clones `rm`, applies `candidate`, snaps start HP to its cache bucket
/// (the same approximation `fight_delta` uses), then probes `enc` for the
/// K paired synthetic seeds. Deterministic given `(rm state, candidate,
/// enc, budget)`.
#[must_use]
pub fn encounter_delta(
    rm: &RunManager,
    candidate: &CandidateMutation,
    enc: &'static EncounterDef,
    budget: &ProbeBudget,
) -> DeltaStats {
    let mut probe_rm = rm.clone();
    let _ = candidate.apply(&mut probe_rm);
    probe_rm.combat = None;
    let (_bucket, rep_hp) =
        bucket_and_rep_hp(probe_rm.run_state.current_hp, probe_rm.run_state.max_hp);
    probe_rm.run_state.current_hp = rep_hp;
    let full = ProbeBudget {
        t_turns: 0, // full fight: outcome must be decided, not truncated
        ..*budget
    };
    probe_encounter(&probe_rm.run_state, enc, &full).into_delta()
}

/// Chained-attrition rollout (chained-attrition relabel design). Clones
/// `rm`, applies `candidate`, snaps to the bucket-rep start HP, then for
/// each of K paired seeds walks `plan.sequence()` threading surviving HP
/// forward as full fights. Death or a non-terminal link ends that chain
/// as a non-survival; surviving the boss is a survival. One chain = one
/// probe; aggregates into [`DeltaStats`] where `win_rate` is the
/// sequence-survival rate.
///
/// SCALE NOTE: `hp_lost`/`dmg_dealt` accumulate across ALL links of a
/// chain, so [`DeltaStats::hp_lost_mean`]/`dmg_dealt_t5_mean` here are
/// per-CHAIN totals — roughly `(chain_normals + 2)`× the per-FIGHT means
/// `fight_delta` produces. [`DeltaStats::score`]'s tempo coefficients
/// (the `0.002`/`0.004` terms) were hand-tuned for the per-fight scale,
/// so under a chained horizon the tempo terms carry more relative weight
/// the longer the chain; `win_rate` (survival) still dominates. The Task-7
/// pilot logs the score-component breakdown to confirm any re-ranking is
/// survival-driven, not a tempo-scale artifact.
#[must_use]
pub fn chain_delta(
    rm: &RunManager,
    candidate: &CandidateMutation,
    plan: &ChainPlan,
    budget: &ProbeBudget,
) -> DeltaStats {
    let mut probe_rm = rm.clone();
    let _ = candidate.apply(&mut probe_rm);
    probe_rm.combat = None;
    let (_bucket, rep_hp) =
        bucket_and_rep_hp(probe_rm.run_state.current_hp, probe_rm.run_state.max_hp);

    let seq = plan.sequence();
    let link_budget = ProbeBudget {
        t_turns: 0,
        ..*budget
    }; // full fights so links terminate
    let rs = &probe_rm.run_state;

    let mut stats = FightStats::default();
    for k in 0..budget.k {
        let mut entry_hp = rep_hp;
        let mut hp_lost_total: i32 = 0;
        let mut dmg_total: i32 = 0;
        let mut turns_total: u32 = 0;
        let mut survived = true;
        for enc in &seq {
            let out = probe_one(rs, enc, entry_hp, synthetic_seed(enc.id, k), &link_budget);
            hp_lost_total += out.hp_lost.max(0);
            dmg_total += out.dmg_dealt;
            turns_total += u32::from(out.turns);
            if !out.terminal {
                // hit move cap — counts as non-survival (penalised, not excluded)
                survived = false;
                break;
            }
            if !out.won {
                survived = false; // died this fight
                break;
            }
            entry_hp = out.exit_hp;
        }
        // Every chain is a decided binary outcome: survived the full sequence
        // or didn't (death or move-cap). Always terminal so win_rate = wins/probes.
        stats.add(&ProbeOutcome {
            terminal: true,
            won: survived,
            hp_lost: hp_lost_total,
            dmg_dealt: dmg_total,
            turns: turns_total.min(u32::from(u16::MAX)) as u16,
            exit_hp: entry_hp,
        });
    }
    stats.into_delta()
}

// ---------------------------------------------------------------------------
// Meta-move probe: `path_delta` (plan §2 "pathing" + M2.5). Pathing is the
// E2 importance leg's #1 frontier (+36) yet `fight_delta` can't rank `Move`
// options — it scores card-shaped candidates against a FIXED elite+boss
// horizon, whereas distinct map edges lead into DIFFERENT next rooms. This
// probe grounds the one edge that actually commits by simulating that
// edge's own next encounter; the projection tail (`projection::score_path`)
// covers the rest of the path.
// ---------------------------------------------------------------------------

/// Sample the encounter a candidate map edge leads into, under §2.5
/// information discipline. The room *type* is player-visible on the map,
/// but the encounter *identity* for combat/elite rooms is cursor-hidden,
/// so it is sampled deterministically from the act's pool keyed on
/// `decision_key` — the SAME value for every edge at one decision, so the
/// sample is paired across candidates (§2.5 rule 3) and never reads the
/// run's upcoming encounter cursor. Non-combat rooms (shop / rest /
/// treasure / event / unknown) carry no grounded fight and return `None`;
/// the projection tail scores those edges instead.
#[must_use]
pub fn next_edge_encounter(
    rm: &RunManager,
    room: MapPointType,
    decision_key: u64,
) -> Option<&'static EncounterDef> {
    let table = act_encounter_table(rm.run_state.act_index, rm.run_state.act1)?;
    let sample = |pool: &[&'static EncounterDef], salt: u64| -> Option<&'static EncounterDef> {
        if pool.is_empty() {
            None
        } else {
            Some(pool[(splitmix64(decision_key ^ salt) as usize) % pool.len()])
        }
    };
    match room {
        MapPointType::Monster => {
            let pool: Vec<&'static EncounterDef> = table
                .iter()
                .filter(|e| e.tier == EncounterTier::Normal)
                .collect();
            sample(&pool, 0x_0000_0000_004D_0001)
        }
        MapPointType::Elite => {
            // Mirror EncounterPlan's elite pool (act 1 has no Elite tier at
            // this pin → fall back to Normal-tier rows, like the engine).
            let elites: Vec<&'static EncounterDef> = table
                .iter()
                .filter(|e| e.tier == EncounterTier::Elite)
                .collect();
            let pool = if elites.is_empty() {
                table
                    .iter()
                    .filter(|e| e.tier == EncounterTier::Normal)
                    .collect()
            } else {
                elites
            };
            sample(&pool, 0x_0000_0000_00E1_0001)
        }
        MapPointType::Boss => EncounterPlan::for_decision(rm, decision_key).boss,
        MapPointType::Unassigned
        | MapPointType::Unknown
        | MapPointType::Shop
        | MapPointType::RestSite
        | MapPointType::Treasure
        | MapPointType::Ancient => None,
    }
}

/// Rollout-grounded delta for one candidate map edge (plan §2/M2.5
/// `path_delta`). Probes the edge's NEXT room as a standalone
/// synthetic-seed combat — the part of a path the projection model
/// approximates worst — and returns the [`DeltaStats`] the policy adds to
/// the tier-1 projected-path score to re-rank edges. Non-combat edges
/// (shop / rest / treasure / event / unknown) carry no fight and return a
/// zero delta, deferring entirely to the projection tail. Deterministic
/// given `(rm state, room, decision_key, budget)` and paired across edges
/// by `decision_key`; shares `FightCache` with `fight_delta` (same
/// `FightKey` space — a Normal encounter probed here and as a boss-horizon
/// fight cache-hit identically).
#[must_use]
pub fn path_delta(
    rm: &RunManager,
    room: MapPointType,
    decision_key: u64,
    budget: &ProbeBudget,
    cache: &FightCache,
) -> DeltaStats {
    let Some(enc) = next_edge_encounter(rm, room, decision_key) else {
        return DeltaStats::default();
    };

    let mut probe_rm = rm.clone();
    // Probes are standalone fights — never the live combat.
    probe_rm.combat = None;
    let (hp_bucket, rep_hp) =
        bucket_and_rep_hp(probe_rm.run_state.current_hp, probe_rm.run_state.max_hp);
    probe_rm.run_state.current_hp = rep_hp;

    let rs = &probe_rm.run_state;
    let key = FightKey {
        deck_hash: deck_hash(rs),
        relics_hash: relics_hash(rs),
        context_hash: context_hash(rs),
        hp_bucket,
        encounter: enc.id,
        budget: budget.key(),
    };
    let stats = if let Some(hit) = cache.get(&key) {
        hit
    } else {
        let computed = probe_encounter(rs, enc, budget);
        cache.put(key, computed);
        computed
    };
    stats.into_delta()
}

/// Rollout-grounded rest-site evaluation (plan §2 "rest" + M2.5
/// `rest_delta`). Rest — E2's #2 frontier (+24) — is currently a tier-1
/// HP-term heuristic; this grounds the heal-vs-smith tradeoff in simulated
/// fights. Each option becomes a [`CandidateMutation`] scored against the
/// SHARED upcoming horizon (`plan`, the act's elite + boss, identical
/// across options — §2.5 rule 3): heal raises HP
/// ([`CandidateMutation::HealHp`]), smith upgrades the policy-chosen card
/// ([`CandidateMutation::UpgradeCard`]). Returns `(heal_delta,
/// smith_delta?)`; `smith_delta` is `None` when no card is upgradeable
/// (fully-upgraded deck). Deterministic and paired by construction
/// (delegates to [`fight_delta`]); shares its `FightCache`.
#[must_use]
pub fn rest_delta(
    rm: &RunManager,
    heal_amount: i32,
    smith_deck_idx: Option<usize>,
    plan: &EncounterPlan,
    budget: &ProbeBudget,
    cache: &FightCache,
) -> (DeltaStats, Option<DeltaStats>) {
    let heal = fight_delta(
        rm,
        &CandidateMutation::HealHp {
            amount: heal_amount,
        },
        plan,
        budget,
        cache,
    );
    let smith = smith_deck_idx.map(|deck_idx| {
        fight_delta(
            rm,
            &CandidateMutation::UpgradeCard { deck_idx },
            plan,
            budget,
            cache,
        )
    });
    (heal, smith)
}

// ---------------------------------------------------------------------------
// Card-shaped surface detection + candidate enumeration (shared by
// TieredPolicy and the offline labeler).
// ---------------------------------------------------------------------------

/// The card-shaped decision surfaces tier-2/3 and the labeler cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardSurface {
    CardReward,
    BossRelic,
    RemoveDeck,
    UpgradeDeck,
    TransformDeck,
    Smith,
}

impl CardSurface {
    /// Stable token for escalation-count maps / audit output.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            CardSurface::CardReward => "card_reward",
            CardSurface::BossRelic => "boss_relic",
            CardSurface::RemoveDeck => "remove_deck",
            CardSurface::UpgradeDeck => "upgrade_deck",
            CardSurface::TransformDeck => "transform_deck",
            CardSurface::Smith => "smith",
        }
    }
}

/// Detect whether the current decision is card-shaped (same dispatch
/// shape as `HeuristicPolicy::choose`). Deck-mutation preempts are
/// recognized by the homogeneous action list; phases by `rm.phase()`.
#[must_use]
pub fn detect_card_surface(rm: &RunManager, actions: &[RunAction]) -> Option<CardSurface> {
    match actions.first()? {
        RunAction::RemoveDeckCard { .. } => return Some(CardSurface::RemoveDeck),
        RunAction::UpgradeDeckCard { .. } => return Some(CardSurface::UpgradeDeck),
        RunAction::TransformDeckCard { .. } => return Some(CardSurface::TransformDeck),
        // Claws' select with its cap reached: only the confirm is left.
        RunAction::ConfirmDeckSelection => return None,
        _ => {}
    }
    match rm.phase() {
        Phase::CardReward => Some(CardSurface::CardReward),
        Phase::BossRelic => Some(CardSurface::BossRelic),
        Phase::RestSite if matches!(actions.first(), Some(RunAction::SmithCard { .. })) => {
            Some(CardSurface::Smith)
        }
        _ => None,
    }
}

/// Enumerate `(candidate, action index)` pairs for a card-shaped
/// decision, in the same option order the tier-1 surface scores them
/// (so tier-1 score vectors align by position). Candidates whose
/// `RunAction` twin isn't in `actions` are dropped.
#[must_use]
pub fn candidates_for_surface(
    rm: &RunManager,
    view: &VisibleRunView,
    surface: CardSurface,
    actions: &[RunAction],
) -> Vec<(CandidateMutation, usize)> {
    let pos = |wanted: &RunAction| actions.iter().position(|a| a == wanted);
    let mut out = Vec::new();
    match surface {
        CardSurface::CardReward => {
            for (i, offer) in view.offers.card_rewards.iter().enumerate() {
                if let Some(p) = pos(&RunAction::PickCardReward { idx: Some(i) }) {
                    out.push((
                        CandidateMutation::AddCard {
                            card_id: offer.card_id,
                            upgraded: offer.upgraded,
                        },
                        p,
                    ));
                }
            }
            if let Some(p) = pos(&RunAction::PickCardReward { idx: None }) {
                out.push((CandidateMutation::Baseline, p));
            }
        }
        CardSurface::BossRelic => {
            for (i, &relic_id) in view.offers.boss_relics.iter().enumerate() {
                if let Some(p) = pos(&RunAction::PickBossRelic { idx: i }) {
                    out.push((CandidateMutation::AddRelic { relic_id }, p));
                }
            }
        }
        CardSurface::RemoveDeck => {
            // Action index = position in the engine's removal offer order
            // (curses sorted to the front — `RunState::removal_offer_order`),
            // which the engine maps back to a master-deck index. Probe the
            // mutation that removes that master-deck card, tagged with the
            // offer index, so the candidate the policy commits matches the
            // fight it was scored by.
            for (i, &deck_idx) in rm.run_state.removal_offer_order().iter().enumerate() {
                if i >= actions.len() {
                    break;
                }
                out.push((CandidateMutation::RemoveCard { deck_idx }, i));
            }
        }
        CardSurface::UpgradeDeck => {
            // Action index = position in the engine's `is_upgradable()` subset
            // (`run_manager.rs:954-968` / `do_upgrade_deck_card_at`), NOT the
            // `!upgraded` one — those diverge as soon as the deck holds a
            // Curse/Status/Quest card, and the probe would then score
            // upgrading a different card than the action it tags. Same
            // engine-derived-index-space rule as the two arms around it
            // (`removal_offer_order` / `transform_eligible_deck_indices`).
            // GH #399 review.
            let eligible = crate::policy::surfaces::picks::upgradable_deck_indices(rm);
            for (j, &deck_idx) in eligible.iter().enumerate() {
                if j >= actions.len() {
                    break;
                }
                out.push((CandidateMutation::UpgradeCard { deck_idx }, j));
            }
        }
        CardSurface::TransformDeck => {
            // Transform ≈ remove + unknown replacement (the M1 framing).
            let eligible = rm.transform_eligible_deck_indices();
            for (j, &deck_idx) in eligible.iter().enumerate() {
                if j >= actions.len() {
                    break;
                }
                out.push((CandidateMutation::RemoveCard { deck_idx }, j));
            }
        }
        CardSurface::Smith => {
            // Same option→deck-copy mapping as HeuristicPolicy::choose_smith.
            let mut used: Vec<usize> = Vec::new();
            for (j, opt) in view.offers.smith_options.iter().enumerate() {
                if j >= actions.len() {
                    break;
                }
                let deck_idx = view.deck.iter().enumerate().position(|(i, c)| {
                    c.card_id == opt.card_id && !c.upgraded && !used.contains(&i)
                });
                if let Some(deck_idx) = deck_idx {
                    used.push(deck_idx);
                    out.push((CandidateMutation::UpgradeCard { deck_idx }, j));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::potions::CharacterId;

    /// The tier-2/3 probe tags each candidate with the ACTION index it will
    /// commit, so `UpgradeCard { deck_idx }` must be the card that action
    /// actually upgrades — the engine's `is_upgradable()` subset, not the
    /// `!upgraded` one. With a curse in the deck the two differ and the probe
    /// would score upgrading a card the commit never touches. Same defect as
    /// `HeuristicPolicy::choose_deck_mutation`'s upgrade arm (GH #399 review).
    #[test]
    fn upgrade_deck_candidates_pair_actions_with_the_card_the_engine_upgrades() {
        use sts2_core::cards::CardId;
        let mut rm = new_run_manager_for_character(41, CharacterId::Ironclad);
        let curse = sts2_content::cards::spec_index::card_pod_from_id(CardId::Decay, false)
            .expect("curse pod");
        rm.run_state.master_deck.insert(0, curse);
        rm.run_state.pending_deck_upgrades = 1;

        let actions = rm.get_available_actions();
        let view = crate::view::VisibleRunView::from_run_manager(&rm);
        let cands = candidates_for_surface(&rm, &view, CardSurface::UpgradeDeck, &actions);
        assert_eq!(
            cands.len(),
            actions.len(),
            "one candidate per offered action"
        );

        for (cand, action_idx) in cands {
            let CandidateMutation::UpgradeCard { deck_idx } = cand else {
                panic!("expected UpgradeCard, got {cand:?}");
            };
            // Drive the engine itself: applying `actions[action_idx]` to a
            // clone must upgrade exactly `deck_idx`.
            let mut probe = rm.clone();
            let before: Vec<bool> = probe
                .run_state
                .master_deck
                .iter()
                .map(|c| c.upgraded)
                .collect();
            let _ = probe.take_action(actions[action_idx]);
            let flipped: Vec<usize> = probe
                .run_state
                .master_deck
                .iter()
                .enumerate()
                .filter(|(i, c)| c.upgraded && !before[*i])
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                flipped,
                vec![deck_idx],
                "action {action_idx} upgrades {flipped:?}, candidate claims deck_idx {deck_idx}"
            );
        }
    }

    fn tiny_budget() -> ProbeBudget {
        ProbeBudget {
            sims: 4,
            k: 2,
            t_turns: 2,
            leaf: LeafKind::TexelWide,
            horizon: LabelHorizon::ElitePlusBoss,
            chain_normals: 0,
        }
    }

    /// `ProbeBudget::key()` separates cache keys across different horizons and chain lengths.
    #[test]
    fn probe_budget_key_separates_horizon_and_chain_len() {
        let base = ProbeBudget::default();
        assert_eq!(base.horizon, LabelHorizon::ElitePlusBoss);
        assert_eq!(base.chain_normals, 0);

        let chained = ProbeBudget {
            horizon: LabelHorizon::ChainedAct,
            chain_normals: 2,
            ..base
        };
        let chained_3 = ProbeBudget {
            chain_normals: 3,
            ..chained
        };
        assert_ne!(
            base.key(),
            chained.key(),
            "horizon must change the cache key"
        );
        assert_ne!(
            chained.key(),
            chained_3.key(),
            "chain length must change the cache key"
        );
    }

    /// Same decision + candidate + budget → byte-identical DeltaStats
    /// (M2 determinism acceptance).
    #[test]
    fn fight_delta_is_deterministic() {
        let rm = new_run_manager_for_character(17, CharacterId::Ironclad);
        let plan = EncounterPlan::for_decision(&rm, 99);
        assert!(plan.elite.is_some(), "act-1 plan must sample an encounter");
        assert!(plan.boss.is_some(), "act-1 boss must be pinned");
        let budget = tiny_budget();
        let cand = CandidateMutation::AddCard {
            card_id: CardId::Inflame,
            upgraded: false,
        };
        let a = fight_delta(&rm, &cand, &plan, &budget, &FightCache::default());
        let b = fight_delta(&rm, &cand, &plan, &budget, &FightCache::default());
        assert_eq!(a, b, "fight_delta must be deterministic");
        assert!(a.probes > 0);
    }

    /// `encounter_delta` scores a single encounter deterministically and
    /// runs full (non-truncated) fights — the combat-surrogate label unit.
    #[test]
    fn encounter_delta_is_deterministic_single_fight() {
        let rm = new_run_manager_for_character(17, CharacterId::Ironclad);
        let enc = next_edge_encounter(&rm, MapPointType::Monster, 7)
            .expect("act-1 monster room samples an encounter");
        let budget = tiny_budget();
        let cand = CandidateMutation::Baseline;
        let a = encounter_delta(&rm, &cand, enc, &budget);
        let b = encounter_delta(&rm, &cand, enc, &budget);
        assert_eq!(a, b, "encounter_delta must be deterministic");
        assert_eq!(a.probes, budget.k, "one encounter × K seeds = K probes");
        // Full fights are always decided, so every probe terminates.
        assert_eq!(a.terminals, a.probes, "full fights must all terminate");
    }

    /// `next_edge_encounter` maps combat room types to a sampled
    /// encounter and non-combat rooms to `None`, paired by `decision_key`.
    #[test]
    fn next_edge_encounter_maps_rooms_and_pairs_by_key() {
        let rm = new_run_manager_for_character(17, CharacterId::Ironclad);
        // Combat rooms resolve to an encounter; non-combat rooms don't.
        assert!(next_edge_encounter(&rm, MapPointType::Monster, 7).is_some());
        assert!(next_edge_encounter(&rm, MapPointType::Elite, 7).is_some());
        assert!(next_edge_encounter(&rm, MapPointType::Boss, 7).is_some());
        for room in [
            MapPointType::Shop,
            MapPointType::RestSite,
            MapPointType::Treasure,
            MapPointType::Ancient,
            MapPointType::Unknown,
        ] {
            assert!(
                next_edge_encounter(&rm, room, 7).is_none(),
                "{room:?} is not a grounded fight"
            );
        }
        // §2.5 rule 3: identity is a pure function of decision_key, so two
        // edges sharing the key sample the same monster encounter.
        assert_eq!(
            next_edge_encounter(&rm, MapPointType::Monster, 42).map(|e| e.id),
            next_edge_encounter(&rm, MapPointType::Monster, 42).map(|e| e.id),
        );
    }

    /// `path_delta` is deterministic, runs probes on a combat edge, and
    /// returns a zero (no-probe) delta on a non-combat edge.
    #[test]
    fn path_delta_is_deterministic_and_skips_noncombat() {
        let rm = new_run_manager_for_character(17, CharacterId::Ironclad);
        let budget = tiny_budget();
        let a = path_delta(
            &rm,
            MapPointType::Monster,
            5,
            &budget,
            &FightCache::default(),
        );
        let b = path_delta(
            &rm,
            MapPointType::Monster,
            5,
            &budget,
            &FightCache::default(),
        );
        assert_eq!(a, b, "path_delta must be deterministic");
        assert!(a.probes > 0, "a combat edge must run probes");

        let rest = path_delta(
            &rm,
            MapPointType::RestSite,
            5,
            &budget,
            &FightCache::default(),
        );
        assert_eq!(rest.probes, 0, "non-combat edge runs no probe");
        assert_eq!(rest, DeltaStats::default());
    }

    /// `rest_delta` grounds heal vs smith against the shared horizon:
    /// both arms run probes, smith is `None` only when no card is given,
    /// and a HealHp mutation lifts the probe's starting HP (clamped).
    #[test]
    fn rest_delta_grounds_heal_and_smith() {
        let rm = new_run_manager_for_character(17, CharacterId::Ironclad);
        let plan = EncounterPlan::for_decision(&rm, 11);
        let budget = tiny_budget();
        let cache = FightCache::default();

        let (heal, smith) = rest_delta(&rm, 20, Some(0), &plan, &budget, &cache);
        assert!(heal.probes > 0, "heal arm must probe the horizon");
        let smith = smith.expect("smith arm present when a deck idx is given");
        assert!(smith.probes > 0, "smith arm must probe the horizon");

        // No upgradeable card → no smith arm.
        let (_, none_smith) = rest_delta(&rm, 20, None, &plan, &budget, &cache);
        assert!(none_smith.is_none());

        // HealHp clamps at max and lifts current HP.
        let mut probe = rm.clone();
        assert!(CandidateMutation::HealHp { amount: 1_000_000 }.apply(&mut probe));
        assert_eq!(probe.run_state.current_hp, probe.run_state.max_hp);
    }

    /// Pairing: the synthetic seed panel is a pure function of
    /// (encounter, k) — candidate-independent — and a combat-inert
    /// candidate (gold delta) therefore produces byte-identical fight
    /// outcomes to the baseline.
    #[test]
    fn probe_seeds_are_paired_across_candidates() {
        for k in 0..4 {
            assert_eq!(
                synthetic_seed("VANTOM_BOSS", k),
                synthetic_seed("VANTOM_BOSS", k)
            );
        }
        assert_ne!(
            synthetic_seed("VANTOM_BOSS", 0),
            synthetic_seed("VANTOM_BOSS", 1),
            "panel seeds must differ across k"
        );

        let rm = new_run_manager_for_character(23, CharacterId::Ironclad);
        let plan = EncounterPlan::for_decision(&rm, 7);
        let budget = tiny_budget();
        let cache = FightCache::default();
        let base = fight_delta(&rm, &CandidateMutation::Baseline, &plan, &budget, &cache);
        // Distinct cache (gold doesn't change the key's deck/relics/hp,
        // so a shared cache would short-circuit the comparison).
        let gold = fight_delta(
            &rm,
            &CandidateMutation::GoldDelta { delta: 250 },
            &plan,
            &budget,
            &FightCache::default(),
        );
        assert_eq!(
            base, gold,
            "combat-inert candidate must see identical paired fights"
        );
    }

    /// Cache: a planted sentinel value is returned on hit; a miss
    /// populates the cache.
    #[test]
    fn fight_cache_hit_and_miss_behavior() {
        let rm = new_run_manager_for_character(29, CharacterId::Ironclad);
        let plan = EncounterPlan {
            elite: None,
            boss: EncounterPlan::for_decision(&rm, 0).boss,
        };
        let budget = tiny_budget();
        let cache = FightCache::default();
        assert!(cache.is_empty());
        let first = fight_delta(&rm, &CandidateMutation::Baseline, &plan, &budget, &cache);
        assert_eq!(cache.len(), 1, "miss populates the cache");

        // Hit: plant a sentinel under the same key and observe it.
        let enc = plan.boss.expect("boss");
        let mut probe_rm = rm.clone();
        let (hp_bucket, rep) =
            bucket_and_rep_hp(probe_rm.run_state.current_hp, probe_rm.run_state.max_hp);
        probe_rm.run_state.current_hp = rep;
        let key = FightKey {
            deck_hash: deck_hash(&probe_rm.run_state),
            relics_hash: relics_hash(&probe_rm.run_state),
            context_hash: context_hash(&probe_rm.run_state),
            hp_bucket,
            encounter: enc.id,
            budget: budget.key(),
        };
        let sentinel = FightStats {
            probes: 2,
            terminals: 2,
            wins: 2,
            losses: 0,
            hp_lost_sum: 84.0,
            // Both probes lost 42, so the second moment is 2 * 42² and the
            // spread is exactly 0 — keeping the sentinel's "identical probes"
            // story intact now that FightStats carries a second moment.
            hp_lost_sq_sum: 2.0 * 42.0 * 42.0,
            dmg_dealt_sum: 4242.0,
            turns_sum: 2.0,
        };
        cache.put(key, sentinel);
        let hit = fight_delta(&rm, &CandidateMutation::Baseline, &plan, &budget, &cache);
        assert!(
            (hit.dmg_dealt_t5_mean - 2121.0).abs() < 1e-9,
            "hit must return the planted stats, got {hit:?}"
        );
        assert_ne!(hit, first, "sentinel differs from the computed stats");
    }

    /// Candidate application mutates the CLONE only.
    #[test]
    fn fight_delta_leaves_original_manager_untouched() {
        let rm = new_run_manager_for_character(31, CharacterId::Ironclad);
        let deck_before = rm.run_state.master_deck.len();
        let plan = EncounterPlan {
            elite: EncounterPlan::for_decision(&rm, 1).elite,
            boss: None,
        };
        let _ = fight_delta(
            &rm,
            &CandidateMutation::AddCard {
                card_id: CardId::Inflame,
                upgraded: true,
            },
            &plan,
            &tiny_budget(),
            &FightCache::default(),
        );
        assert_eq!(rm.run_state.master_deck.len(), deck_before);
    }

    /// EncounterPlan is a pure function of (rm, decision_key) — the
    /// "identical across candidates" requirement holds by construction,
    /// and different keys can sample different elites.
    #[test]
    fn encounter_plan_is_deterministic_per_decision() {
        let rm = new_run_manager_for_character(37, CharacterId::Ironclad);
        let a = EncounterPlan::for_decision(&rm, 5);
        let b = EncounterPlan::for_decision(&rm, 5);
        assert_eq!(a.elite.map(|e| e.id), b.elite.map(|e| e.id));
        assert_eq!(a.boss.map(|e| e.id), b.boss.map(|e| e.id));
    }

    #[test]
    fn probe_one_respects_entry_hp_and_reports_exit() {
        crate::rollout::ensure_content_installed();
        let mut rm = sts2_content::run_callbacks::new_run_manager_for_character(
            7,
            sts2_core::potions::CharacterId::Ironclad,
        );
        sts2_content::run_callbacks::initialize_run_setup(&mut rm);
        let plan = EncounterPlan::for_decision(&rm, 42);
        let enc = plan.elite.or(plan.boss).expect("act-1 has an encounter");
        let budget = ProbeBudget {
            t_turns: 0,
            ..ProbeBudget::default()
        };
        let out = probe_one(&rm.run_state, enc, 8, synthetic_seed(enc.id, 0), &budget);
        // entry capped at 8 HP; exit can't exceed entry (no probe healing past start here)
        assert!(
            out.exit_hp <= 8,
            "exit_hp {} should be <= entry 8",
            out.exit_hp
        );
        if out.terminal && !out.won {
            assert!(out.exit_hp <= 0, "a lost terminal fight ends at <=0 HP");
        }
    }

    #[test]
    fn chain_delta_is_paired_and_bounded() {
        crate::rollout::ensure_content_installed();
        let mut rm = sts2_content::run_callbacks::new_run_manager_for_character(
            5,
            sts2_core::potions::CharacterId::Ironclad,
        );
        sts2_content::run_callbacks::initialize_run_setup(&mut rm);
        let plan = ChainPlan::for_decision(&rm, 7, 2);
        let budget = ProbeBudget {
            k: 4,
            sims: 30,
            chain_normals: 2,
            horizon: LabelHorizon::ChainedAct,
            ..ProbeBudget::default()
        };
        let a = chain_delta(&rm, &CandidateMutation::Baseline, &plan, &budget);
        let b = chain_delta(&rm, &CandidateMutation::Baseline, &plan, &budget);
        assert_eq!(a, b, "deterministic given (state, candidate, plan, budget)");
        assert!((0.0..=1.0).contains(&a.win_rate));
        assert_eq!(a.probes, 4, "one chain per paired seed");
        assert_eq!(
            a.terminals, a.probes,
            "every chain resolves to a decided survive/not-survive outcome"
        );
    }

    #[test]
    fn live_enemy_hp_excludes_sentinel_and_clamps() {
        assert_eq!(live_enemy_hp(50), 50);
        assert_eq!(live_enemy_hp(800), 800); // real bosses are well under the threshold
        assert_eq!(live_enemy_hp(0), 0);
        assert_eq!(live_enemy_hp(-7), 0); // negative clamps to 0
        assert_eq!(live_enemy_hp(999_999_999), 0); // Door sentinel excluded
        assert_eq!(live_enemy_hp(SENTINEL_HP_MIN), 0);
        assert_eq!(live_enemy_hp(SENTINEL_HP_MIN - 1), SENTINEL_HP_MIN - 1);
    }

    #[test]
    fn chain_plan_samples_normals_paired_and_ordered() {
        crate::rollout::ensure_content_installed();
        let mut rm =
            sts2_content::run_callbacks::new_run_manager_for_character(11, CharacterId::Ironclad);
        sts2_content::run_callbacks::initialize_run_setup(&mut rm);
        let a = ChainPlan::for_decision(&rm, 99, 2);
        let b = ChainPlan::for_decision(&rm, 99, 2);
        // Same decision_key => identical sampled identities (paired across candidates).
        let ids_a: Vec<&str> = a.sequence().iter().map(|e| e.id).collect();
        let ids_b: Vec<&str> = b.sequence().iter().map(|e| e.id).collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(a.normals.len(), 2);
        assert!(a.normals.iter().all(|e| e.tier == EncounterTier::Normal));
        // sequence = normals then elite then boss
        let seq = a.sequence();
        assert!(seq.len() >= 3);
        assert_eq!(seq.last().map(|e| e.tier), a.base.boss.map(|e| e.tier));
    }
}
