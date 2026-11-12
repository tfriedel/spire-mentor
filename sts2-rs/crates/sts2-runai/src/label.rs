//! Offline rollout labeler (full-run-AI plan §5/M2, §7 `label.rs`;
//! bead `sts2-rs-40ikw.6`).
//!
//! [`LabelingPolicy`] wraps a driving policy: the run proceeds with the
//! wrapped policy's choices (meta = heuristic, combat = eval-cheap MCTS
//! when composed under `MctsCombatPolicy`), and at every card-shaped
//! decision it computes [`fight_delta`] for **all** options with paired
//! synthetic seeds and emits one [`LabelRow`] per option. The label
//! rows are the M2 retune corpus (`retune_v.py`).
//!
//! Each row carries the option's tier-1 V score and the post-candidate
//! `HeuristicV::feature_map` vector — the "expose the feature vector
//! via the labeler" arm of the plan's retune design (recomputing
//! features in Python would duplicate `card_stats`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use sts2_core::full_run::{drive_full_run, FullRunConfig, RunPolicy};
use sts2_core::run_manager::{RunAction, RunManager};

use crate::arena::{character_token, classify_outcome, ArenaRow, CellSpec};
use crate::deck_archetype::classify_archetype;
use crate::eval::HeuristicV;
use crate::policy::{
    AuditRecord, AuditingPolicy, CombatBudget, HeuristicPolicy, MctsCombatPolicy, VScoreSource,
};
use crate::rollout::{
    candidates_for_surface, detect_card_surface, fight_delta, DeltaStats, EncounterPlan,
    FightCache, ProbeBudget,
};
use crate::view::VisibleRunView;

/// Engine identity stamp for label rows. The content-addressed source
/// hash of `sts2-core` (strictly stronger than a git SHA: it changes
/// with any engine source edit, committed or not) — the plan's
/// "label files stamped with engine git SHA" freshness guard.
#[must_use]
pub fn engine_identity() -> &'static str {
    sts2_core::SRC_HASH
}

/// Where in a run a label was taken (plan §7 `labels.jsonl` schema).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRef {
    pub seed: i32,
    /// `RunState::total_floor` at decision time.
    pub floor: i32,
    /// `Phase` Debug form.
    pub phase: String,
    /// 0-based decision index within the run — aligned with the
    /// decisions.jsonl `step` (the labeler counts every decision the
    /// same way `AuditingPolicy` does).
    pub step: u64,
}

/// Probe budget in serialized form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetCfg {
    pub sims: u32,
    pub k: u32,
    /// Player-turn truncation for the legacy `"elite-boss"` horizon.
    /// IGNORED under `"chained-act"`: chain links always run as full
    /// fights (`t_turns = 0`) so HP can thread, regardless of this value.
    pub t_turns: u32,
    pub leaf: String,
    /// Labeling horizon: `"elite-boss"` (legacy) or `"chained-act"`.
    pub horizon: String,
    /// Number of normal-tier fights prepended in the `"chained-act"` horizon.
    pub chain_normals: usize,
}

impl From<&ProbeBudget> for BudgetCfg {
    fn from(b: &ProbeBudget) -> Self {
        Self {
            sims: b.sims,
            k: b.k,
            t_turns: b.t_turns,
            leaf: format!("{:?}", b.leaf),
            horizon: match b.horizon {
                crate::rollout::LabelHorizon::ElitePlusBoss => "elite-boss".to_string(),
                crate::rollout::LabelHorizon::ChainedAct => "chained-act".to_string(),
            },
            chain_normals: b.chain_normals,
        }
    }
}

/// One labels.jsonl row: one option of one card-shaped decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelRow {
    pub decision_ref: DecisionRef,
    /// Card-shaped surface token (`rollout::CardSurface::token`).
    pub surface: String,
    /// Option description (`CandidateMutation::desc`).
    pub option: String,
    /// Index of this option in the surfaced action list.
    pub action_idx: usize,
    /// `true` when the driving policy actually picked this option.
    pub chosen: bool,
    /// Tier-1 V score of the candidate-applied view.
    pub v_score: f64,
    /// Rollout-derived score (`DeltaStats::score()` — serialized so
    /// Python consumers don't re-implement the blend).
    pub rollout_score: f64,
    /// Post-candidate `HeuristicV::feature_map` (the retune inputs).
    pub features: BTreeMap<String, f64>,
    pub delta_stats: DeltaStats,
    pub budget_cfg: BudgetCfg,
    pub k: u32,
    /// Engine identity (content-addressed source hash; see
    /// [`engine_identity`]).
    pub engine_git_sha: String,
}

/// Run-policy wrapper that labels every card-shaped decision while the
/// wrapped policy drives the run. Sits INSIDE `AuditingPolicy` and
/// OUTSIDE `MctsCombatPolicy` so its per-decision counter matches the
/// audit log's `step` field.
#[derive(Debug)]
pub struct LabelingPolicy<P> {
    inner: P,
    probe: ProbeBudget,
    cache: FightCache,
    v: HeuristicV,
    rows: Vec<LabelRow>,
    seed: i32,
    step: u64,
}

impl<P: RunPolicy> LabelingPolicy<P> {
    #[must_use]
    pub fn new(inner: P, seed: i32, probe: ProbeBudget) -> Self {
        Self {
            inner,
            probe,
            cache: FightCache::default(),
            v: HeuristicV::default(),
            rows: Vec::new(),
            seed,
            step: 0,
        }
    }

    #[must_use]
    pub fn into_rows(self) -> Vec<LabelRow> {
        self.rows
    }
}

impl<P: RunPolicy> RunPolicy for LabelingPolicy<P> {
    fn prepare(&mut self, rm: &mut RunManager) {
        // Forward the per-decision hook so a wrapped policy that materializes
        // state before observing (e.g. PpoPolicy's event materialization) still
        // runs. Default `prepare` is a no-op, so this is inert for every other
        // inner policy.
        self.inner.prepare(rm);
    }

    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let step = self.step;
        self.step += 1;

        let chosen_idx = self.inner.choose(rm, actions)?;

        if let Some(surface) = detect_card_surface(rm, actions) {
            let view = VisibleRunView::from_run_manager(rm);
            let candidates = candidates_for_surface(rm, &view, surface, actions);
            if candidates.len() >= 2 {
                let key = step ^ (u64::from(rm.seed as u32) << 24);
                let plan = EncounterPlan::for_decision(rm, key);
                if plan.elite.is_some() || plan.boss.is_some() {
                    let chain = match self.probe.horizon {
                        crate::rollout::LabelHorizon::ChainedAct => {
                            Some(crate::rollout::ChainPlan::for_decision(
                                rm,
                                key,
                                self.probe.chain_normals,
                            ))
                        }
                        crate::rollout::LabelHorizon::ElitePlusBoss => None,
                    };
                    for (candidate, action_idx) in &candidates {
                        let delta = match &chain {
                            Some(cp) => crate::rollout::chain_delta(rm, candidate, cp, &self.probe),
                            None => fight_delta(rm, candidate, &plan, &self.probe, &self.cache),
                        };
                        let cand_view = candidate.apply_to_view(&view);
                        let features: BTreeMap<String, f64> = self
                            .v
                            .feature_map(&cand_view)
                            .into_iter()
                            .map(|(k, v)| (k.to_string(), v))
                            .collect();
                        self.rows.push(LabelRow {
                            decision_ref: DecisionRef {
                                seed: self.seed,
                                floor: rm.run_state.total_floor,
                                phase: format!("{:?}", rm.phase()),
                                step,
                            },
                            surface: surface.token().to_string(),
                            option: candidate.desc(),
                            action_idx: *action_idx,
                            chosen: *action_idx == chosen_idx,
                            v_score: self.v.score(&cand_view),
                            rollout_score: delta.score(),
                            features,
                            delta_stats: delta,
                            budget_cfg: BudgetCfg::from(&self.probe),
                            k: self.probe.k,
                            engine_git_sha: engine_identity().to_string(),
                        });
                    }
                }
            }
        }

        Some(chosen_idx)
    }
}

impl<P: VScoreSource> VScoreSource for LabelingPolicy<P> {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        self.inner.take_v_scores()
    }
}

/// The labeler's driving policy: heuristic meta, optionally bridged
/// through the combat MCTS (the campaign default is eval-cheap).
#[derive(Debug)]
enum LabelDriver {
    Plain(HeuristicPolicy),
    Mcts(MctsCombatPolicy<HeuristicPolicy>),
}

impl RunPolicy for LabelDriver {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        match self {
            LabelDriver::Plain(p) => p.choose(rm, actions),
            LabelDriver::Mcts(p) => p.choose(rm, actions),
        }
    }
}

impl VScoreSource for LabelDriver {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        match self {
            LabelDriver::Plain(p) => p.take_v_scores(),
            LabelDriver::Mcts(p) => p.take_v_scores(),
        }
    }
}

/// Run one labeling cell: a full self-play run with the heuristic
/// driver, labeling every card-shaped decision. Returns the arena row
/// (sweep bookkeeping), the label rows, and the §3.6 audit records.
#[must_use]
pub fn run_label_cell(
    spec: &CellSpec,
    combat: Option<CombatBudget>,
    probe: &ProbeBudget,
) -> (ArenaRow, Vec<LabelRow>, Vec<AuditRecord>) {
    // Uniform registry state before the run starts (reproducible
    // labeling trajectories — bead sts2-rs-40ikw.7; see
    // rollout::ensure_content_installed).
    crate::rollout::ensure_content_installed();
    let label = format!("label+{}", crate::arena::policy_label(spec));
    let cfg = FullRunConfig {
        max_steps: spec.max_steps,
        livelock_stall_limit: spec.livelock_stall_limit,
        invincible: false,
        record_trace: false,
    };
    let started = std::time::Instant::now();

    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        spec.seed,
        spec.character,
        spec.ascension,
    )
    .with_neow_start();
    // C#-faithful run setup (relic bags + room generation). Without this the
    // labeled training data is relic-less / static-event — the same harness bug
    // the arena had (Addendum 23). Must match the arena's construction exactly
    // or replayed feature trajectories diverge.
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    let driver = match combat {
        Some(budget) => LabelDriver::Mcts(MctsCombatPolicy::new(
            HeuristicPolicy::new(),
            budget,
            spec.seed,
        )),
        None => LabelDriver::Plain(HeuristicPolicy::new()),
    };
    let labeling = LabelingPolicy::new(driver, spec.seed, *probe);
    let mut policy = AuditingPolicy::new(labeling, spec.seed, label.clone());
    let report = drive_full_run(&mut rm, &mut policy, &cfg);
    let (labeling, records) = policy.into_parts();
    let rows = labeling.into_rows();

    let (outcome, won, bin) = classify_outcome(&report.outcome, report.final_phase);
    let final_view = VisibleRunView::from_run_manager(&rm);
    let final_deck: Vec<String> = final_view
        .deck
        .iter()
        .map(|dc| {
            if dc.upgraded {
                format!("{:?}+", dc.card_id)
            } else {
                format!("{:?}", dc.card_id)
            }
        })
        .collect();
    let final_archetype = classify_archetype(&final_view).token().to_string();
    let sample_seed: u64 = std::env::var("STS2_RUNAI_DECK_SAMPLE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let row = ArenaRow {
        seed: spec.seed,
        character: character_token(spec.character).to_string(),
        ascension: spec.ascension,
        act1: spec.act1.resolved_token(spec.seed),
        policy: label,
        combat_budget: spec.combat.token().to_string(),
        outcome,
        won,
        steps: report.steps,
        final_phase: format!("{:?}", report.final_phase),
        final_hp: report.final_hp,
        final_gold: report.final_gold,
        act_index: report.act_index,
        final_relics: rm.run_state.relics.len(),
        wall_ms: started.elapsed().as_millis() as u64,
        noncompletion_bin: bin,
        escalated_t2: 0,
        escalated_t3: 0,
        combat_decisions: 0,
        combat_decisions_elite: 0,
        combat_decisions_boss: 0,
        combat_decisions_motif: 0,
        combat_sims_spent: 0,
        combat_fallback_untranslatable: 0,
        combat_fallback_no_outcome: 0,
        combat_fallback_not_surfaced: 0,
        final_deck,
        final_archetype,
        sample_seed,
        potions_overflow_dropped: rm.run_state.potions_overflow_dropped,
        // Label/proxy harnesses never run the PPO net and never bin a
        // panic row (they drive inside the caller's unwind scope).
        ppo_peak_act_slots: 0,
        ppo_peak_tokens: 0,
        reached_second_boss: crate::arena::reached_second_boss(&rm),
        panic_message: None,
    };
    (row, rows, records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{CombatBudgetKind, PolicyKind};
    use crate::policy::LeafKind;
    use sts2_core::full_run::DEFAULT_LIVELOCK_STALL_LIMIT;
    use sts2_core::potions::CharacterId;

    fn spec(seed: i32) -> CellSpec {
        CellSpec {
            seed,
            character: CharacterId::Ironclad,
            ascension: 0,
            policy: PolicyKind::Heuristic,
            combat: CombatBudgetKind::None,
            sims_override: None,
            boss_sims_override: None,
            max_steps: 3000,
            livelock_stall_limit: DEFAULT_LIVELOCK_STALL_LIMIT,
            act1: crate::arena::Act1Choice::Overgrowth,
            audit: true,
            ablate: None,
        }
    }

    /// 2-seed campaign smoke (M2 acceptance): well-formed rows, every
    /// labeled decision covers ≥2 options, exactly one chosen flag per
    /// (decision, surface), audit log non-empty.
    #[test]
    fn labeler_emits_well_formed_rows_on_two_seed_campaign() {
        let probe = ProbeBudget {
            sims: 2,
            k: 2,
            t_turns: 1,
            leaf: LeafKind::TexelWide,
            horizon: crate::rollout::LabelHorizon::default(),
            chain_normals: 0,
        };
        let mut all_rows = Vec::new();
        for seed in [3, 5] {
            let (row, rows, records) = run_label_cell(&spec(seed), None, &probe);
            assert!(!row.outcome.is_empty());
            assert!(!records.is_empty(), "audit log must be written");
            all_rows.extend(rows);
        }
        assert!(
            !all_rows.is_empty(),
            "two seeds must produce at least one labeled decision"
        );
        let mut by_decision: std::collections::BTreeMap<(i32, u64), Vec<&LabelRow>> =
            std::collections::BTreeMap::new();
        for row in &all_rows {
            assert!(!row.option.is_empty());
            assert_eq!(row.k, 2);
            assert_eq!(row.budget_cfg.sims, 2);
            assert_eq!(row.budget_cfg.horizon, "elite-boss");
            assert_eq!(row.budget_cfg.chain_normals, 0);
            assert_eq!(row.engine_git_sha, engine_identity());
            assert!(row.delta_stats.probes > 0, "probes ran: {row:?}");
            assert!(!row.features.is_empty(), "feature vector exported");
            by_decision
                .entry((row.decision_ref.seed, row.decision_ref.step))
                .or_default()
                .push(row);
        }
        for ((seed, step), rows) in &by_decision {
            assert!(
                rows.len() >= 2,
                "decision ({seed},{step}) labeled with <2 options"
            );
            assert_eq!(
                rows.iter().filter(|r| r.chosen).count(),
                1,
                "decision ({seed},{step}) must mark exactly one chosen option"
            );
        }
        // JSONL round-trip. Floats are compared with tolerance —
        // serde_json's default float PARSING is fast-path (can be a
        // ULP off), so exact equality after a round-trip is too strict.
        let line = serde_json::to_string(&all_rows[0]).expect("serializes");
        let back: LabelRow = serde_json::from_str(&line).expect("parses");
        assert_eq!(back.decision_ref, all_rows[0].decision_ref);
        assert_eq!(back.surface, all_rows[0].surface);
        assert_eq!(back.option, all_rows[0].option);
        assert_eq!(back.chosen, all_rows[0].chosen);
        assert_eq!(back.budget_cfg, all_rows[0].budget_cfg);
        assert_eq!(back.engine_git_sha, all_rows[0].engine_git_sha);
        assert!((back.v_score - all_rows[0].v_score).abs() < 1e-9);
        assert!((back.rollout_score - all_rows[0].rollout_score).abs() < 1e-9);
        assert_eq!(back.features.len(), all_rows[0].features.len());
    }

    /// Chained-act horizon labels card decisions and stamps `"chained-act"` in `budget_cfg`.
    #[test]
    fn chained_horizon_emits_rows_with_chain_score() {
        let probe = ProbeBudget {
            sims: 20,
            k: 2,
            t_turns: 0,
            leaf: LeafKind::TexelWide,
            horizon: crate::rollout::LabelHorizon::ChainedAct,
            chain_normals: 1,
        };
        let (_row, rows, _audit) = run_label_cell(&spec(3), None, &probe);
        assert!(
            !rows.is_empty(),
            "chained campaign still labels card decisions"
        );
        assert!(
            rows.iter().any(|r| r.budget_cfg.horizon == "chained-act"),
            "at least one row must carry horizon=chained-act"
        );
    }
}
