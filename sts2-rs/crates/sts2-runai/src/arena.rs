//! Arena sweep harness — drive panel seeds through full runs and report
//! per-seed rows with completeness binning (plan §3.2, §5-M0).
//!
//! Bead `sts2-rs-40ikw.4`. Modeled on `sts2-cli`'s `fuzz` subcommand
//! (rayon over cells, per-cell `catch_unwind`, panic-hook suppression),
//! plus: named policies ([`PolicyKind`] × [`CombatBudgetKind`]), JSONL
//! row output, the §3.2 non-completion bins, and the §3.6 audit log.
//!
//! The *per-seed wall-clock timeout* (process-per-seed) lives in the
//! `run_arena` binary, not here — a known engine bug (bead
//! `sts2-rs-65mbx`) loops forever inside a single `take_action`, which
//! no in-process watchdog can interrupt. [`run_cell`] is the in-process
//! unit both modes share.

use std::panic::{catch_unwind, AssertUnwindSafe};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use sts2_content::run_callbacks::new_run_manager_for_character_with_ascension;
use sts2_core::act_selection::{ActSelectionUnlocks, ActVariant};
use sts2_core::full_run::{
    drive_full_run, FullRunConfig, FullRunReport, RandomPolicy, RunOutcome, RunPolicy,
};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction};

use crate::deck_archetype::classify_archetype;
use crate::policy::{
    AblationPolicy, AblationSurface, AuditRecord, AuditingPolicy, CombatBudget, CombatSearchRecord,
    DecisionTelemetry, FallbackCounts, FirstOptionPolicy, HeuristicPolicy, MctsCombatPolicy,
    MetaEvalRecord, SampledDeckPolicy, TieredPolicy, VScoreSource, ValuePolicy,
};
use crate::trajectory::{self, Detail, Meta, RecordingPolicy};
use crate::view::VisibleRunView;

pub use sts2_core::full_run::{DEFAULT_LIVELOCK_STALL_LIMIT, DEFAULT_MAX_STEPS};

/// Meta policy selector (non-combat phases; also combat when the budget
/// is `none`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    Random,
    FirstOption,
    /// M1 tier-1 heuristic agent (plan §5/M1, bead `sts2-rs-40ikw.5`).
    Heuristic,
    /// M2 tiered agent: heuristic + margin-gated tier-2/3 rollout
    /// escalation (plan §5/M2, bead `sts2-rs-40ikw.6`).
    Tiered,
    /// M4 value agent (E9): learned run-value net scores card picks by
    /// `V(resulting state)`, heuristic everywhere else (plan §5/M4, bead
    /// `sts2-rs-40ikw.8`). Requires `STS2_RUNAI_VALUE=<weights.json>`.
    ValueNet,
    /// Deck-oracle sampling agent: samples card-reward choices via softmax
    /// over heuristic-V scores + archetype bias; delegates all other phases
    /// to the embedded heuristic. Sample parameters are seeded by
    /// `(run_seed, STS2_RUNAI_DECK_SAMPLE_SEED)` (default seed = 0).
    SampledDeck,
    /// PPO run-policy net (2026-07-07 ppo-rust-inference plan, Task 8):
    /// tract-onnx inference over the exported `RunPolicyNet` for every
    /// non-combat decision. Requires a combat budget (never plays combat
    /// itself — `(Ppo, CombatBudgetKind::None)` is a hard CLI error) and
    /// `STS2_RUNAI_PPO=<model.onnx>` (or the sibling `sts2-pycli`
    /// checkout's committed snapshot).
    Ppo,
}

impl PolicyKind {
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            PolicyKind::Random => "random",
            PolicyKind::FirstOption => "first",
            PolicyKind::Heuristic => "heuristic",
            PolicyKind::Tiered => "tiered",
            PolicyKind::ValueNet => "value-net",
            PolicyKind::SampledDeck => "sampled-deck",
            PolicyKind::Ppo => "ppo",
        }
    }

    /// Parse the CLI token
    /// (`random` | `first` | `heuristic` | `tiered` | `value-net` | `ppo`).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(PolicyKind::Random),
            "first" | "first-option" => Ok(PolicyKind::FirstOption),
            "heuristic" => Ok(PolicyKind::Heuristic),
            "tiered" => Ok(PolicyKind::Tiered),
            "value-net" | "value" => Ok(PolicyKind::ValueNet),
            "sampled-deck" => Ok(PolicyKind::SampledDeck),
            "ppo" => Ok(PolicyKind::Ppo),
            other => Err(format!(
                "unknown policy '{other}' (random|first|heuristic|tiered|value-net|sampled-deck|ppo)"
            )),
        }
    }
}

/// `(elite decisions, boss decisions, motif-tagged common decisions,
/// simulations requested)` over one run
/// — [`MctsCombatPolicy::combat_budget_spent`]'s shape, carried into
/// [`ArenaRow`] so a budget arm's compute neutrality is measured (GH #446).
pub type BudgetSpent = (u64, u64, u64, u64);

/// Combat budget selector (plan §3.3): the two frozen configs plus
/// `none` (meta policy plays combat too — the fuzz behavior), plus the two
/// GH #446 compute-neutral reallocation arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombatBudgetKind {
    None,
    EvalCheap,
    EvalStrong,
    /// GH #446 reallocation arm: `eval-cheap` compute, elite escalated,
    /// common trimmed. See [`CombatBudget::elite_focus`].
    EliteFocus,
    /// GH #446 reallocation arm: `eval-cheap` compute, common decisions
    /// escalated only on budget-sensitive motifs. See
    /// [`CombatBudget::motif_focus`].
    MotifFocus,
}

impl CombatBudgetKind {
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            CombatBudgetKind::None => "none",
            CombatBudgetKind::EvalCheap => "eval-cheap",
            CombatBudgetKind::EvalStrong => "eval-strong",
            CombatBudgetKind::EliteFocus => "elite-focus",
            CombatBudgetKind::MotifFocus => "motif-focus",
        }
    }

    /// Parse the CLI token (`none` | `eval-cheap` | `eval-strong` |
    /// `elite-focus` | `motif-focus`).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(CombatBudgetKind::None),
            "eval-cheap" => Ok(CombatBudgetKind::EvalCheap),
            "eval-strong" => Ok(CombatBudgetKind::EvalStrong),
            "elite-focus" => Ok(CombatBudgetKind::EliteFocus),
            "motif-focus" => Ok(CombatBudgetKind::MotifFocus),
            other => Err(format!(
                "unknown combat budget '{other}' \
                 (none|eval-cheap|eval-strong|elite-focus|motif-focus)"
            )),
        }
    }

    /// Resolve to a concrete [`CombatBudget`] (with optional debug
    /// sims overrides applied). `None` for [`CombatBudgetKind::None`].
    #[must_use]
    pub fn budget(
        self,
        sims_override: Option<u32>,
        boss_sims_override: Option<u32>,
    ) -> Option<CombatBudget> {
        let mut budget = match self {
            CombatBudgetKind::None => return None,
            CombatBudgetKind::EvalCheap => CombatBudget::eval_cheap(),
            CombatBudgetKind::EvalStrong => CombatBudget::eval_strong(),
            CombatBudgetKind::EliteFocus => CombatBudget::elite_focus(),
            CombatBudgetKind::MotifFocus => CombatBudget::motif_focus(),
        };
        if let Some(s) = sims_override {
            budget.sims = s;
        }
        if let Some(b) = boss_sims_override {
            budget.boss_sims = b;
        }
        Some(budget)
    }
}

/// Lowercase character token (panel/report form) → [`CharacterId`].
pub fn parse_character(s: &str) -> Result<CharacterId, String> {
    match s {
        "ironclad" => Ok(CharacterId::Ironclad),
        "silent" => Ok(CharacterId::Silent),
        "defect" => Ok(CharacterId::Defect),
        "regent" => Ok(CharacterId::Regent),
        "necrobinder" => Ok(CharacterId::Necrobinder),
        other => Err(format!("unknown character '{other}'")),
    }
}

/// [`CharacterId`] → lowercase token.
#[must_use]
pub fn character_token(c: CharacterId) -> &'static str {
    match c {
        CharacterId::Ironclad => "ironclad",
        CharacterId::Silent => "silent",
        CharacterId::Defect => "defect",
        CharacterId::Regent => "regent",
        CharacterId::Necrobinder => "necrobinder",
        CharacterId::Fool => "fool",
    }
}

/// Which Act-1 model an arena cell walks (GH #544).
///
/// The token vocabulary is the PyO3 binding's, verbatim (the `act1` kwarg in
/// `sts2-py/src/run_manager.rs`): `overgrowth` | `underdocks` | `roll`. One
/// vocabulary across Rust and Python means a panel measured through
/// `run_arena` and a run built from `ppo/env.py` name the same thing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Act1Choice {
    /// The single Act 1 this engine modelled before the Underdocks port.
    ///
    /// This was the default until act choice became seed-driven everywhere.
    /// Win-rate numbers measured before that change, with no explicit act,
    /// describe this act alone; compare them against an `overgrowth` arm.
    Overgrowth,
    /// The Act 1 ported in #544/#553.
    Underdocks,
    /// Roll from the seed the way the real game does at
    /// `StartRunLobby.BeginRunLocally`.
    ///
    /// **Not paired with a fixed-act arm.** The act becomes a function of the
    /// seed, so seed 7 under `roll` may be a different act than seed 7 under
    /// `overgrowth` — a per-seed diff across those two arms mixes the act
    /// change with everything downstream of it. Pair `overgrowth` against
    /// `underdocks` when you want a paired difficulty delta; use `roll` to
    /// measure the *deployed* mixture.
    ///
    /// The default: the real game picks the act from the seed, and the C#
    /// host, the PyO3 `RunManager` and `PpoRunEnv` all default to it too.
    #[default]
    Roll,
}

impl Act1Choice {
    /// Parse a CLI/config token. Mirrors the PyO3 binding's accepted set.
    ///
    /// # Errors
    /// Returns the unknown token.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "overgrowth" => Ok(Self::Overgrowth),
            "underdocks" => Ok(Self::Underdocks),
            "roll" => Ok(Self::Roll),
            other => Err(format!(
                "unknown act1 '{other}' (expected 'overgrowth', 'underdocks' or 'roll')"
            )),
        }
    }

    /// The token this choice parses back from.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Overgrowth => "overgrowth",
            Self::Underdocks => "underdocks",
            Self::Roll => "roll",
        }
    }

    /// The act this choice actually produces for `seed`.
    ///
    /// `Roll` reproduces `RunManager::with_rolled_act1` — same stream, same
    /// `test_mode_on(true)` unlock assumption — so a row can report the act a
    /// cell really walked without holding the `RunManager`.
    #[must_use]
    pub fn resolve(self, seed: i32) -> ActVariant {
        match self {
            Self::Overgrowth => ActVariant::Overgrowth,
            Self::Underdocks => ActVariant::Underdocks,
            Self::Roll => {
                sts2_core::act_selection::roll_acts(seed, &ActSelectionUnlocks::test_mode_on(true))
                    [0]
            }
        }
    }

    /// The resolved act as a report token (`"overgrowth"` / `"underdocks"`).
    ///
    /// Spelled out rather than derived from `Debug`, so renaming a variant
    /// cannot silently change the schema of every report on disk. Hive and
    /// Glory are act-2/3 models and cannot come out of index 0; they are
    /// mapped rather than panicked on so a report is never unwritable.
    #[must_use]
    pub fn resolved_token(self, seed: i32) -> String {
        match self.resolve(seed) {
            ActVariant::Overgrowth => "overgrowth",
            ActVariant::Underdocks => "underdocks",
            ActVariant::Hive => "hive",
            ActVariant::Glory => "glory",
        }
        .to_string()
    }
}

/// Everything needed to run one `(seed, character, ascension)` cell —
/// shared by the in-process rayon path and the `one-cell` child process.
#[derive(Clone, Copy, Debug)]
pub struct CellSpec {
    pub seed: i32,
    pub character: CharacterId,
    pub ascension: u8,
    pub policy: PolicyKind,
    pub combat: CombatBudgetKind,
    /// Debug overrides for the frozen budgets (CLI `--sims`/`--boss-sims`).
    pub sims_override: Option<u32>,
    pub boss_sims_override: Option<u32>,
    pub max_steps: usize,
    pub livelock_stall_limit: usize,
    /// Which Act-1 model this cell walks (GH #544). Defaults to
    /// [`Act1Choice::Roll`], the real game's seed-driven pick.
    pub act1: Act1Choice,
    /// Collect §3.6 audit records for this cell.
    pub audit: bool,
    /// E2 importance leg (plan §3.5): when set, this surface's decisions
    /// are routed to a uniform-random pick (all others stay on the
    /// heuristic base policy; Addendum 6/24).
    pub ablate: Option<AblationSurface>,
}

/// The act of a report row that predates the `act1` field. Every such run
/// walked Overgrowth, so this stays `overgrowth` although the arena now
/// defaults to `roll`.
fn default_act1_token() -> String {
    Act1Choice::Overgrowth.token().to_string()
}

/// One output row of a sweep (JSONL line in `report.jsonl`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaRow {
    pub seed: i32,
    pub character: String,
    pub ascension: u8,
    /// The Act-1 model this run actually walked — RESOLVED, so a `roll` cell
    /// reports `overgrowth`/`underdocks` rather than `roll`. Without it a
    /// mixed report is uninterpretable. `#[serde(default)]` keeps every
    /// pre-#544 report loadable (they were all Overgrowth).
    #[serde(default = "default_act1_token")]
    pub act1: String,
    pub policy: String,
    pub combat_budget: String,
    /// `won|lost|max_steps|livelock|empty_actions|not_implemented|rejected|policy_halt|panic|timeout`.
    pub outcome: String,
    pub won: bool,
    pub steps: usize,
    pub final_phase: String,
    pub final_hp: i32,
    pub final_gold: i32,
    pub act_index: u8,
    /// Held relics at run end — the realism anchor for guardrail #3 (a
    /// relic-less agent, the 2026-06-14 bug, shows up as ~2 here vs a healthy
    /// ~8-12). `#[serde(default)]` keeps pre-2026-06-14 reports loadable.
    #[serde(default)]
    pub final_relics: usize,
    /// Wall-clock per cell. NOT deterministic — zero it (`--no-wall-ms`)
    /// for byte-identical re-run comparisons.
    pub wall_ms: u64,
    /// `None` iff the run completed (outcome `won`/`lost`); otherwise
    /// the §3.2 bin: `not_implemented:<phase>:<action-variant>`, or
    /// `<outcome>:<phase>` for livelock/empty_actions/rejected/
    /// max_steps/policy_halt, or bare `panic`/`timeout`.
    pub noncompletion_bin: Option<String>,
    /// Decisions escalated to tier-2 rollout probes (plan §3.3:
    /// escalation counts reported per sweep; 0 for non-tiered
    /// policies). Bead `sts2-rs-40ikw.6`.
    #[serde(default)]
    pub escalated_t2: u64,
    /// Decisions escalated to tier-3 (boss-relic) probes.
    #[serde(default)]
    pub escalated_t3: u64,
    /// Combat decisions for which the bridge consulted the MCTS — the
    /// denominator for the fallback rate (issue #164; 0 for `none`-budget
    /// cells that never bridge combat). `#[serde(default)]` keeps
    /// pre-#164 reports loadable.
    #[serde(default)]
    pub combat_decisions: u64,
    /// Of [`Self::combat_decisions`], those made in elite / boss combats,
    /// and the simulations the budget requested across all of them. A
    /// budget-reallocation arm (GH #446 `elite-focus` / `motif-focus`) is
    /// compute-neutral iff its `combat_sims_spent` matches `eval-cheap`'s
    /// on the same seeds. `#[serde(default)]` keeps older reports loadable.
    #[serde(default)]
    pub combat_decisions_elite: u64,
    #[serde(default)]
    pub combat_decisions_boss: u64,
    /// Common-tier decisions whose root carried a budget-sensitive motif
    /// (potion available / several enemies) — the `motif-focus` arm's
    /// escalation set, counted on every budget.
    #[serde(default)]
    pub combat_decisions_motif: u64,
    #[serde(default)]
    pub combat_sims_spent: u64,
    /// Combat MCTS picks handed back to the meta policy because the chosen
    /// action had no `RunAction` twin — Channel / Evoke / Discover /
    /// unmodeled multi-pick selections. **The #164 gap made measurable.**
    #[serde(default)]
    pub combat_fallback_untranslatable: u64,
    /// Combat MCTS picks that fell back because the search tree had no
    /// root outcome (rare; not the #164 gap).
    #[serde(default)]
    pub combat_fallback_no_outcome: u64,
    /// Combat MCTS picks that translated but were absent from the surfaced
    /// action list (translation drift, not a missing surface).
    #[serde(default)]
    pub combat_fallback_not_surfaced: u64,
    /// Cards in the master deck at run end, in master-deck order.
    /// Each entry is `"<CardId>"` (base) or `"<CardId>+"` (upgraded).
    /// Empty when the run did not complete (panic / timeout).
    /// `#[serde(default)]` keeps pre-task-2.1 reports loadable.
    #[serde(default)]
    pub final_deck: Vec<String>,
    /// Coarse archetype bucket of the final deck (task 2.1).
    /// One of `thin|scaling|block|exhaust|aggro|generic`.
    /// `"generic"` is also the fallback for panic/timeout rows.
    #[serde(default)]
    pub final_archetype: String,
    /// `STS2_RUNAI_DECK_SAMPLE_SEED` env var at the time of this run
    /// (default 0). Lets generate/validate/fidelity join on `(seed, sample_seed)`.
    #[serde(default)]
    pub sample_seed: u64,
    /// Potions irrecoverably lost to a full belt this run (silent drops
    /// plus declined swap decisions) — the overflow-waste measure for the
    /// potion-economics experiments (GH #252/#253). `#[serde(default)]`
    /// keeps pre-#252 reports loadable.
    #[serde(default)]
    pub potions_overflow_dropped: u32,
    /// Largest candidate-action count the PPO net was asked to score this
    /// run — the `A_MAX` headroom signal (GH #315). `0` for non-PPO cells
    /// and for runs whose net never scored a decision.
    ///
    /// The export budget is a hard cliff (overflow panics rather than
    /// truncating), so the only always-on warning that a sweep is
    /// approaching it is how close its worst decision got. Compare
    /// against the loaded champion's `a_max` (160 since #315).
    #[serde(default)]
    pub ppo_peak_act_slots: u64,
    /// Largest `obs + act` token count the PPO net was asked to score
    /// this run — the `T_MAX` headroom signal. See
    /// [`Self::ppo_peak_act_slots`]; compare against `t_max` (320).
    #[serde(default)]
    pub ppo_peak_tokens: u64,
    /// A10 only: the run ENTERED the final act's second boss room.
    ///
    /// A10 is A9 plus one extra Boss node attached directly above the
    /// act-3 boss ([`sts2_core::map::ActMap::attach_second_boss`]), so a
    /// run can only stand in that room after killing the boss that would
    /// have ENDED an A9 run. This flag is therefore the A9-equivalent win
    /// marker: an A10 sweep yields the paired A9 win rate without a second
    /// sweep. Always `false` below A10 (no such node exists) and for
    /// panic/timeout rows.
    ///
    /// Read off `RunState::visited_map_coords`, which is act-scoped and so
    /// still holds the final act's path when the run ends.
    #[serde(default)]
    pub reached_second_boss: bool,
    /// Panic payload for `outcome == "panic"` rows, `None` otherwise.
    ///
    /// Before GH #315 the payload was discarded, so a panicking cell was
    /// binned as `panic` with no way to name what failed unless the sweep
    /// happened to run under `STS2_ARENA_SHOW_PANICS=1` — which is why
    /// #315's failing surface had to be inferred from the outside. The
    /// audit-log path can't cover this: it drops its records on panic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panic_message: Option<String>,
}

impl ArenaRow {
    /// Stable sort key for deterministic report files.
    #[must_use]
    pub fn sort_key(&self) -> (i32, String, u8) {
        (self.seed, self.character.clone(), self.ascension)
    }
}

/// Stable variant name of a [`RunAction`] for binning (`non_exhaustive`
/// enum — unknown future variants bin as `"other"`).
#[must_use]
pub fn run_action_variant_name(action: &RunAction) -> &'static str {
    match action {
        RunAction::Move { .. } => "Move",
        RunAction::PlayCard { .. } => "PlayCard",
        RunAction::EndTurn => "EndTurn",
        RunAction::CombatResolveChoice { .. } => "CombatResolveChoice",
        RunAction::DrinkPotion { .. } => "DrinkPotion",
        RunAction::PotionSwap { .. } => "PotionSwap",
        RunAction::PickCardReward { .. } => "PickCardReward",
        RunAction::PickRelicReward => "PickRelicReward",
        RunAction::SkipRelic => "SkipRelic",
        RunAction::PickBossRelic { .. } => "PickBossRelic",
        RunAction::LeaveShop => "LeaveShop",
        RunAction::ShopBuyCard { .. } => "ShopBuyCard",
        RunAction::ShopBuyRelic { .. } => "ShopBuyRelic",
        RunAction::ShopBuyPotion { .. } => "ShopBuyPotion",
        RunAction::ShopRemoveCard { .. } => "ShopRemoveCard",
        RunAction::RestOption { .. } => "RestOption",
        RunAction::SmithCard { .. } => "SmithCard",
        RunAction::EventChoice { .. } => "EventChoice",
        RunAction::CollectTreasure => "CollectTreasure",
        RunAction::SacrificeCardReward => "SacrificeCardReward",
        RunAction::RemoveDeckCard { .. } => "RemoveDeckCard",
        RunAction::UpgradeDeckCard { .. } => "UpgradeDeckCard",
        _ => "other",
    }
}

fn phase_name(phase: Phase) -> String {
    format!("{phase:?}")
}

/// Map a finished run's outcome to `(outcome token, won, bin)`.
/// Completion (plan §3.2): a run "completes" iff `Won`/`Lost`; every
/// other outcome gets a bin keyed by `(phase, action-variant)` for
/// `NotImplemented` and by phase otherwise.
#[must_use]
pub fn classify_outcome(
    outcome: &RunOutcome,
    final_phase: Phase,
) -> (String, bool, Option<String>) {
    match outcome {
        RunOutcome::Won => ("won".into(), true, None),
        RunOutcome::Lost => ("lost".into(), false, None),
        RunOutcome::MaxSteps => (
            "max_steps".into(),
            false,
            Some(format!("max_steps:{}", phase_name(final_phase))),
        ),
        RunOutcome::Livelock { phase, .. } => (
            "livelock".into(),
            false,
            Some(format!("livelock:{}", phase_name(*phase))),
        ),
        RunOutcome::EmptyActions { phase, .. } => (
            "empty_actions".into(),
            false,
            Some(format!("empty_actions:{}", phase_name(*phase))),
        ),
        RunOutcome::NotImplemented { phase, action, .. } => (
            "not_implemented".into(),
            false,
            Some(format!(
                "not_implemented:{}:{}",
                phase_name(*phase),
                run_action_variant_name(action)
            )),
        ),
        RunOutcome::Rejected { phase, .. } => (
            "rejected".into(),
            false,
            Some(format!("rejected:{}", phase_name(*phase))),
        ),
        RunOutcome::PolicyHalt { phase, .. } => (
            "policy_halt".into(),
            false,
            Some(format!("policy_halt:{}", phase_name(*phase))),
        ),
    }
}

/// The combined policy a cell runs: meta policy, optionally bridged
/// through the combat MCTS. (An enum so [`run_cell`] stays object-safe
/// without boxing.)
#[derive(Debug)]
pub(crate) enum CellPolicy {
    Random(RandomPolicy),
    First(FirstOptionPolicy),
    Heuristic(HeuristicPolicy),
    Tiered(TieredPolicy),
    Value(ValuePolicy),
    MctsRandom(MctsCombatPolicy<RandomPolicy>),
    MctsFirst(MctsCombatPolicy<FirstOptionPolicy>),
    MctsHeuristic(MctsCombatPolicy<HeuristicPolicy>),
    MctsTiered(MctsCombatPolicy<TieredPolicy>),
    MctsValue(MctsCombatPolicy<ValuePolicy>),
    /// E2 importance leg (plan §3.5): heuristic base with one surface
    /// ablated to random. With/without the combat bridge.
    AblatedHeuristic(AblationPolicy<HeuristicPolicy>),
    MctsAblatedHeuristic(MctsCombatPolicy<AblationPolicy<HeuristicPolicy>>),
    /// Deck-oracle sampling agent (task 1.3/1.4): samples card-reward
    /// picks via softmax; delegates all other phases to heuristic.
    Sampled(SampledDeckPolicy),
    MctsSampled(MctsCombatPolicy<SampledDeckPolicy>),
    /// PPO run-policy net (2026-07-07 ppo-rust-inference plan, Task 8).
    /// Always combat-bridged — `(Ppo, CombatBudgetKind::None)` is a hard
    /// CLI error in [`build_policy`], so there is no bare `Ppo` variant.
    #[cfg(feature = "ppo")]
    MctsPpo(MctsCombatPolicy<crate::ppo::policy::PpoPolicy>),
}

impl CellPolicy {
    /// `(tier-2, tier-3)` escalation totals for tiered cells, `(0, 0)`
    /// otherwise (plan §3.3 sweep reporting; bead `sts2-rs-40ikw.6`).
    fn escalations(&self) -> (u64, u64) {
        match self {
            CellPolicy::Tiered(p) => (p.counts().tier2, p.counts().tier3),
            CellPolicy::MctsTiered(p) => (p.meta().counts().tier2, p.meta().counts().tier3),
            _ => (0, 0),
        }
    }

    /// Combat MCTS→meta fallback tally and the combat-decision
    /// denominator (issue #164). Only the `Mcts*` cells bridge combat
    /// through the search; every other cell plays combat with the meta
    /// policy directly and so reports `(default, 0)`.
    fn combat_fallbacks(&self) -> (FallbackCounts, u64, BudgetSpent) {
        match self {
            CellPolicy::MctsRandom(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsFirst(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsHeuristic(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsTiered(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsValue(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsAblatedHeuristic(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            CellPolicy::MctsSampled(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => {
                (p.fallbacks(), p.combat_decisions(), p.combat_budget_spent())
            }
            _ => (FallbackCounts::default(), 0, (0, 0, 0, 0)),
        }
    }

    /// `(peak action slots, peak obs+act tokens)` the PPO net was asked
    /// to score — the export-budget headroom signal (GH #315). `(0, 0)`
    /// for every non-PPO cell, which never loads a net.
    fn ppo_peak(&self) -> (u64, u64) {
        match self {
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => {
                let (slots, tokens) = p.meta().peak();
                (slots as u64, tokens as u64)
            }
            _ => (0, 0),
        }
    }
}

/// Per-decision telemetry ([`DecisionTelemetry`]) forwarded to whichever
/// variant this cell is running, so `--record-runs` can explain every
/// pick regardless of the `--policy` it was swept under.
impl DecisionTelemetry for CellPolicy {
    fn combat_search(&self) -> Option<&CombatSearchRecord> {
        match self {
            CellPolicy::MctsRandom(p) => p.combat_search(),
            CellPolicy::MctsFirst(p) => p.combat_search(),
            CellPolicy::MctsHeuristic(p) => p.combat_search(),
            CellPolicy::MctsTiered(p) => p.combat_search(),
            CellPolicy::MctsValue(p) => p.combat_search(),
            CellPolicy::MctsAblatedHeuristic(p) => p.combat_search(),
            CellPolicy::MctsSampled(p) => p.combat_search(),
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => p.combat_search(),
            _ => None,
        }
    }

    fn take_meta_eval(&mut self) -> Option<MetaEvalRecord> {
        match self {
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => p.take_meta_eval(),
            _ => None,
        }
    }
}

impl RunPolicy for CellPolicy {
    fn choose(
        &mut self,
        rm: &sts2_core::run_manager::RunManager,
        actions: &[RunAction],
    ) -> Option<usize> {
        match self {
            CellPolicy::Random(p) => p.choose(rm, actions),
            CellPolicy::First(p) => p.choose(rm, actions),
            CellPolicy::Heuristic(p) => p.choose(rm, actions),
            CellPolicy::Tiered(p) => p.choose(rm, actions),
            CellPolicy::Value(p) => p.choose(rm, actions),
            CellPolicy::MctsRandom(p) => p.choose(rm, actions),
            CellPolicy::MctsFirst(p) => p.choose(rm, actions),
            CellPolicy::MctsHeuristic(p) => p.choose(rm, actions),
            CellPolicy::MctsTiered(p) => p.choose(rm, actions),
            CellPolicy::MctsValue(p) => p.choose(rm, actions),
            CellPolicy::AblatedHeuristic(p) => p.choose(rm, actions),
            CellPolicy::MctsAblatedHeuristic(p) => p.choose(rm, actions),
            CellPolicy::Sampled(p) => p.choose(rm, actions),
            CellPolicy::MctsSampled(p) => p.choose(rm, actions),
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => p.choose(rm, actions),
        }
    }

    fn prepare(&mut self, rm: &mut sts2_core::run_manager::RunManager) {
        match self {
            CellPolicy::Random(p) => p.prepare(rm),
            CellPolicy::First(p) => p.prepare(rm),
            CellPolicy::Heuristic(p) => p.prepare(rm),
            CellPolicy::Tiered(p) => p.prepare(rm),
            CellPolicy::Value(p) => p.prepare(rm),
            CellPolicy::MctsRandom(p) => p.prepare(rm),
            CellPolicy::MctsFirst(p) => p.prepare(rm),
            CellPolicy::MctsHeuristic(p) => p.prepare(rm),
            CellPolicy::MctsTiered(p) => p.prepare(rm),
            CellPolicy::MctsValue(p) => p.prepare(rm),
            CellPolicy::AblatedHeuristic(p) => p.prepare(rm),
            CellPolicy::MctsAblatedHeuristic(p) => p.prepare(rm),
            CellPolicy::Sampled(p) => p.prepare(rm),
            CellPolicy::MctsSampled(p) => p.prepare(rm),
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => p.prepare(rm),
        }
    }
}

impl VScoreSource for CellPolicy {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        match self {
            CellPolicy::Random(p) => p.take_v_scores(),
            CellPolicy::First(p) => p.take_v_scores(),
            CellPolicy::Heuristic(p) => p.take_v_scores(),
            CellPolicy::Tiered(p) => p.take_v_scores(),
            CellPolicy::Value(p) => p.take_v_scores(),
            CellPolicy::MctsRandom(p) => p.take_v_scores(),
            CellPolicy::MctsFirst(p) => p.take_v_scores(),
            CellPolicy::MctsHeuristic(p) => p.take_v_scores(),
            CellPolicy::MctsTiered(p) => p.take_v_scores(),
            CellPolicy::MctsValue(p) => p.take_v_scores(),
            CellPolicy::AblatedHeuristic(p) => p.take_v_scores(),
            CellPolicy::MctsAblatedHeuristic(p) => p.take_v_scores(),
            CellPolicy::Sampled(p) => p.take_v_scores(),
            CellPolicy::MctsSampled(p) => p.take_v_scores(),
            #[cfg(feature = "ppo")]
            CellPolicy::MctsPpo(p) => p.take_v_scores(),
        }
    }
}

/// Heuristic-V weights for this sweep: the embedded defaults, or the TOML
/// named by `STS2_RUNAI_WEIGHTS` (retune candidates — `retune_v.py`). Env
/// var rather than a `CellSpec` field so it flows through the
/// process-per-seed `one-cell` children for free; loaded once. A bad
/// path/file panics loudly on the first cell rather than silently
/// benchmarking the defaults.
fn sweep_weights() -> &'static (crate::eval::Weights, Option<String>) {
    static WEIGHTS: std::sync::OnceLock<(crate::eval::Weights, Option<String>)> =
        std::sync::OnceLock::new();
    WEIGHTS.get_or_init(|| match std::env::var("STS2_RUNAI_WEIGHTS") {
        Ok(path) => {
            let p = std::path::PathBuf::from(&path);
            let w = crate::eval::Weights::load(&p)
                .unwrap_or_else(|e| panic!("STS2_RUNAI_WEIGHTS: {e}"));
            let base = p
                .file_name()
                .map_or(path.clone(), |f| f.to_string_lossy().into_owned());
            (w, Some(base))
        }
        Err(_) => (crate::eval::Weights::default(), None),
    })
}

fn build_tiered() -> TieredPolicy {
    TieredPolicy::with_weights(sweep_weights().0.clone())
}

/// Value net for `--policy value-net` sweeps (M4 E9): loaded once from
/// `STS2_RUNAI_VALUE` (same env-var pattern as [`sweep_weights`]). Bead
/// `sts2-rs-40ikw.8`.
fn sweep_value() -> Option<&'static (std::sync::Arc<crate::eval::run_value::ValueNet>, String)> {
    static NET: std::sync::OnceLock<
        Option<(std::sync::Arc<crate::eval::run_value::ValueNet>, String)>,
    > = std::sync::OnceLock::new();
    NET.get_or_init(|| match std::env::var("STS2_RUNAI_VALUE") {
        Ok(path) => {
            let p = std::path::PathBuf::from(&path);
            let net = crate::eval::run_value::ValueNet::from_weights_path(&p)
                .unwrap_or_else(|e| panic!("STS2_RUNAI_VALUE: {e}"));
            let base = p
                .file_name()
                .map_or(path.clone(), |f| f.to_string_lossy().into_owned());
            Some((std::sync::Arc::new(net), base))
        }
        Err(_) => None,
    })
    .as_ref()
}

fn build_value() -> ValuePolicy {
    let Some((net, _)) = sweep_value() else {
        panic!("--policy value-net requires STS2_RUNAI_VALUE=<weights.json>")
    };
    ValuePolicy::with_weights(net.clone(), sweep_weights().0.clone())
}

fn build_sampled(spec: &CellSpec) -> SampledDeckPolicy {
    let sample_seed: u64 = std::env::var("STS2_RUNAI_DECK_SAMPLE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    SampledDeckPolicy::new(sweep_weights().0.clone(), spec.seed, sample_seed)
}

pub(crate) fn build_policy(spec: &CellSpec) -> CellPolicy {
    let budget = spec
        .combat
        .budget(spec.sims_override, spec.boss_sims_override);
    let w = || sweep_weights().0.clone();
    // E2 importance leg (plan §3.5): one surface → random over the
    // heuristic base policy (Addendum 6/24). Other `--policy` values +
    // ablation are rejected at the CLI.
    if let Some(surface) = spec.ablate {
        return match budget {
            None => CellPolicy::AblatedHeuristic(AblationPolicy::new(
                HeuristicPolicy::with_weights(w()),
                vec![surface],
                spec.seed,
            )),
            Some(b) => CellPolicy::MctsAblatedHeuristic(MctsCombatPolicy::new(
                AblationPolicy::new(HeuristicPolicy::with_weights(w()), vec![surface], spec.seed),
                b,
                spec.seed,
            )),
        };
    }
    match (spec.policy, budget) {
        (PolicyKind::Random, None) => CellPolicy::Random(RandomPolicy::from_run_seed(spec.seed)),
        (PolicyKind::FirstOption, None) => CellPolicy::First(FirstOptionPolicy),
        (PolicyKind::Heuristic, None) => CellPolicy::Heuristic(HeuristicPolicy::with_weights(w())),
        (PolicyKind::Tiered, None) => CellPolicy::Tiered(build_tiered()),
        (PolicyKind::ValueNet, None) => CellPolicy::Value(build_value()),
        (PolicyKind::SampledDeck, None) => CellPolicy::Sampled(build_sampled(spec)),
        (PolicyKind::Random, Some(b)) => CellPolicy::MctsRandom(MctsCombatPolicy::new(
            RandomPolicy::from_run_seed(spec.seed),
            b,
            spec.seed,
        )),
        (PolicyKind::FirstOption, Some(b)) => {
            CellPolicy::MctsFirst(MctsCombatPolicy::new(FirstOptionPolicy, b, spec.seed))
        }
        (PolicyKind::Heuristic, Some(b)) => CellPolicy::MctsHeuristic(MctsCombatPolicy::new(
            HeuristicPolicy::with_weights(w()),
            b,
            spec.seed,
        )),
        (PolicyKind::Tiered, Some(b)) => {
            CellPolicy::MctsTiered(MctsCombatPolicy::new(build_tiered(), b, spec.seed))
        }
        (PolicyKind::ValueNet, Some(b)) => {
            CellPolicy::MctsValue(MctsCombatPolicy::new(build_value(), b, spec.seed))
        }
        (PolicyKind::SampledDeck, Some(b)) => {
            CellPolicy::MctsSampled(MctsCombatPolicy::new(build_sampled(spec), b, spec.seed))
        }
        (PolicyKind::Ppo, None) => {
            panic!("--policy ppo requires --combat eval-cheap|eval-strong (the net never plays combat)")
        }
        #[cfg(feature = "ppo")]
        (PolicyKind::Ppo, Some(b)) => CellPolicy::MctsPpo(MctsCombatPolicy::new(
            crate::ppo::policy::PpoPolicy::from_env_for(spec.character),
            b,
            spec.seed,
        )),
        #[cfg(not(feature = "ppo"))]
        (PolicyKind::Ppo, Some(_)) => {
            panic!("--policy ppo requires the `ppo` cargo feature of sts2-runai (tract-onnx inference is compiled out of this build)")
        }
    }
}

/// Policy label written into rows + audit records, e.g.
/// `"random+eval-cheap"` (plus any debug sims override, so a report can
/// never silently mix overridden and frozen budgets).
#[must_use]
pub fn policy_label(spec: &CellSpec) -> String {
    // E2 ablation arms taint the label so a report can never silently mix
    // an ablated surface with the unablated base.
    if let Some(surface) = spec.ablate {
        let mut label = format!(
            "heuristic:ablate={}+{}",
            surface.token(),
            spec.combat.token()
        );
        if let (_, Some(name)) = sweep_weights() {
            label.push_str(&format!(":weights={name}"));
        }
        return label;
    }
    let mut label = format!("{}+{}", spec.policy.token(), spec.combat.token());
    if spec.combat != CombatBudgetKind::None {
        if let Some(s) = spec.sims_override {
            label.push_str(&format!(":sims={s}"));
        }
        if let Some(b) = spec.boss_sims_override {
            label.push_str(&format!(":boss-sims={b}"));
        }
    }
    // Retune-candidate weights taint the label the same way sims
    // overrides do — a report can never silently mix weight files.
    if matches!(spec.policy, PolicyKind::Heuristic | PolicyKind::Tiered) {
        if let (_, Some(name)) = sweep_weights() {
            label.push_str(&format!(":weights={name}"));
        }
    }
    label
}

/// Guardrail #1 (2026-06-14): assert the run is validly initialized before it
/// is driven. Catches the "agent silently crippled by an empty pool / missing
/// init" class the relic bug belonged to — the failure mode that survived all
/// of M0–M5 because every experiment was *relative* and the broken baseline
/// hit every arm equally. A hard error here (→ caught as an `outcome="panic"`
/// row) turns that silent degradation into a loud, visible failure.
fn check_run_setup_valid(rm: &sts2_core::run_manager::RunManager) -> Result<(), String> {
    let rs = &rm.run_state;
    if rs.master_deck.is_empty() {
        return Err("empty master deck after setup".into());
    }
    if rs.relics.is_empty() {
        return Err("no starter relic after setup".into());
    }
    if rs.relic_grab_bag_by_rarity.values().all(Vec::is_empty) {
        return Err("relic grab bag empty (initialize_run / stock_relic_pools not run?)".into());
    }
    if rs.boss_relic_pool.is_empty() {
        return Err("boss relic pool empty after setup".into());
    }
    Ok(())
}

/// Parse one injection token into `(name, upgraded)`, stripping an optional
/// `+N` / `+` upgrade suffix.
fn parse_inject_token(token: &str) -> (&str, bool) {
    let t = token.trim();
    if let Some(rest) = t.strip_suffix('+') {
        return (rest.trim_end(), true);
    }
    if let Some(pos) = t.rfind('+') {
        let (name, digits) = t.split_at(pos);
        let digits = &digits[1..];
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            return (name.trim_end(), digits != "0");
        }
    }
    (t, false)
}

/// Force-add a comma-separated card package to the run deck (spec §8 injection
/// A/B). Returns the number of cards added; `Err` on the first unknown token so
/// the caller aborts the arm loudly — never a silent partial injection.
fn apply_card_injection(
    rm: &mut sts2_core::run_manager::RunManager,
    raw: &str,
) -> Result<usize, String> {
    let mut added = 0;
    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (name, upgraded) = parse_inject_token(token);
        let card_id = sts2_content::names::card_id_from_str(name)
            .ok_or_else(|| format!("STS2_RUNAI_INJECT_CARDS: unknown card token '{token}'"))?;
        let pod = sts2_content::cards::v0111::card_pod_for(card_id, upgraded, rm.game_version)
            .ok_or_else(|| format!("STS2_RUNAI_INJECT_CARDS: no spec for '{token}'"))?;
        rm.run_state.master_deck.push(pod);
        added += 1;
    }
    Ok(added)
}

/// Apply `STS2_RUNAI_INJECT_CARDS` if set; panic (abort the arm) on a bad token.
/// Unset / empty leaves the deployed run byte-identical.
fn inject_cards_from_env(rm: &mut sts2_core::run_manager::RunManager) {
    if let Ok(raw) = std::env::var("STS2_RUNAI_INJECT_CARDS") {
        if !raw.trim().is_empty() {
            match apply_card_injection(rm, &raw) {
                Ok(n) => eprintln!("[inject] added {n} card(s) to the run deck"),
                Err(e) => panic!("{e}"),
            }
        }
    }
}

/// Run one cell in-process: build the content-wired run manager, drive
/// it, classify, and (optionally) collect audit records. Panics are
/// caught and reported as `outcome = "panic"` rows.
#[must_use]
/// Build the reproducibility [`Meta`] for a recorded run: the frozen
/// combat budget, the policy model path + its SHA-256, the engine sha,
/// and the `STS2_RUNAI_*` env snapshot.
fn build_meta(spec: &CellSpec, label: &str, detail: Detail, report: &FullRunReport) -> Meta {
    let budget = spec
        .combat
        .budget(spec.sims_override, spec.boss_sims_override);
    let (sims, boss_sims) = budget.map_or((0, 0), |b| (b.sims, b.boss_sims));
    let build_id = std::env::var("STS2_BUILD_ID").unwrap_or_else(|_| "v0.103.2".to_string());
    // The active policy model, if any (PPO ONNX).
    let model_path = std::env::var("STS2_RUNAI_PPO")
        .ok()
        .filter(|s| !s.is_empty());
    let model_sha256 = model_path.as_deref().and_then(trajectory::file_sha256);
    let (outcome, _won, _bin) = classify_outcome(&report.outcome, report.final_phase);
    Meta {
        seed_int: spec.seed,
        character: character_token(spec.character).to_string(),
        ascension: spec.ascension,
        policy: format!("{:?}", spec.policy).to_lowercase(),
        policy_label: label.to_string(),
        combat_budget: spec.combat.token().to_string(),
        sims,
        boss_sims,
        model_path,
        model_sha256,
        engine_git_sha: std::env::var(trajectory::ENGINE_SHA_ENV)
            .ok()
            .filter(|s| !s.is_empty()),
        env: trajectory::repro_env(&build_id),
        detail: detail.token().to_string(),
        outcome,
        final_hp: report.final_hp,
        final_gold: report.final_gold,
        steps: report.steps,
    }
}

/// The parity-pinned run construction every arena cell starts from.
///
/// **This must stay effect-identical to `PpoRunEnv.__post_init__`**
/// (`sts2-pycli/python/ppo/env.py`): gate G1
/// (`sts2-pycli/tests/test_ppo_env_parity.py`) pins the training env's expert
/// mode to `run_cell`'s per-seed outcomes, so a flag flipped on one side and
/// not the other diverges the two worlds. The statement order mirrors the PyO3
/// `RunManager::new` binding exactly (`sts2-py/src/run_manager.rs:596-676`):
/// factory → `with_neow_start` → interactive flags → `initialize_run_setup`.
///
/// GH #399: the four deck-mutation flags are ON. Without them the engine
/// silently auto-resolved every removal / upgrade / transform / enchant
/// (front-of-deck removal, random upgrade, …), so the run policy had never
/// seen a deck-mutation decision — the same invisibility class as the #302
/// event bug. `interactive_potion_swap` (GH #252/#379) is set here too, but
/// from the shared `STS2_RUNAI_POTION_SWAP` resolver rather than hardcoded,
/// so this helper — not the caller — is the ONE place any construction path
/// gets it from. See `potion_swap_flag_is_assigned_exactly_once_in_this_file`
/// for why that is a pinned invariant and not just a convention.
// `pub(crate)` so `fork_diagnosis::new_run` replays the corpus with the
// EXACT construction the sweep used — it re-implemented this by hand and
// silently fell behind when the interactive deck-mutation flags were added
// (every seed of a deckmut-champion corpus then failed replay fidelity by
// a few steps, 2026-08-19).
pub(crate) fn new_arena_run_manager(
    seed: i32,
    character: CharacterId,
    ascension: u8,
    act1: Act1Choice,
) -> sts2_core::run_manager::RunManager {
    // Start at the Neow event like the real game (C# floor 1 = Neow;
    // skipping it both forfeits the opening bonus and shifts the act-1
    // event cursor) — 2026-06-12 anomaly triage on the M1 sweeps.
    let mut rm =
        new_run_manager_for_character_with_ascension(seed, character, ascension).with_neow_start();
    rm.run_state.set_agent_decisions();
    // GH #252 / #447 acquisition follow-up: full-belt potion rewards
    // surface a discard-to-claim swap decision (priced by the OC table in
    // the run policy) instead of silently dropping. Env-gated OFF — the
    // deployed default and every parity/training construction stay
    // byte-identical; the resolver panics unless STS2_RUNAI_POTION_OC is
    // also set, so an unpriced swap surface cannot ship into an arm.
    rm.run_state.interactive_potion_swap = crate::policy::resolve_potion_swap_flag_from_env();
    // GH #544: which of the two Act-1 models this run walks. Set HERE, in the
    // same slot the PyO3 binding sets it (after the interactive flags, before
    // `initialize_run_setup`) — the setup call is what pre-generates every
    // act's rooms off `up_front`, so an act chosen afterwards changes nothing
    // about the rooms already rolled.
    let mut rm = match act1 {
        Act1Choice::Overgrowth => rm.with_act1(ActVariant::Overgrowth),
        Act1Choice::Underdocks => rm.with_act1(ActVariant::Underdocks),
        Act1Choice::Roll => rm.with_rolled_act1(),
    };
    // Bug fix (2026-06-14, Addendum 23): the run-AI path never ran the
    // C#-faithful run setup, so relic grab bags were empty (agent played
    // relic-less) and events served in fixed unshuffled order. Call the one
    // shared init — same code the PyO3/production path uses — after
    // `with_neow_start`, matching the PyO3 order.
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    rm
}

/// Did this run enter the A10 second-boss room? See
/// [`ArenaRow::reached_second_boss`].
pub(crate) fn reached_second_boss(rm: &sts2_core::RunManager) -> bool {
    let Some(map) = rm.run_state.current_act_map.as_ref() else {
        return false;
    };
    let Some(sb) = map.second_boss else {
        return false;
    };
    let coord = map.point(sb).coord;
    rm.run_state.visited_map_coords.contains(&coord)
}

pub fn run_cell(spec: &CellSpec) -> (ArenaRow, Vec<AuditRecord>) {
    // Uniform registry state BEFORE the run starts — without this the
    // first tier-2 probe (or another cell's probe) flips the
    // potion/relic registries on mid-run and perturbs in-flight
    // trajectories (bead sts2-rs-40ikw.7; see
    // rollout::ensure_content_installed).
    crate::rollout::ensure_content_installed();
    let label = policy_label(spec);
    let cfg = FullRunConfig {
        max_steps: spec.max_steps,
        livelock_stall_limit: spec.livelock_stall_limit,
        invincible: false,
        record_trace: false,
    };
    let started = std::time::Instant::now();

    let spec_copy = *spec;
    let label_for_run = label.clone();
    let result = catch_unwind(AssertUnwindSafe(move || {
        // Neow start + deck-mutation flags + shared run setup, in the exact
        // PyO3 order the Python training env mirrors (G1). See
        // `new_arena_run_manager`.
        let mut rm = new_arena_run_manager(
            spec_copy.seed,
            spec_copy.character,
            spec_copy.ascension,
            spec_copy.act1,
        );
        // NOTE: `interactive_potion_swap` (GH #252/#379) is deliberately NOT
        // set here — `new_arena_run_manager` already resolved it from
        // `STS2_RUNAI_POTION_SWAP` (default OFF). #563 set it a second time
        // right here with an inlined, default-ON expression; nothing caught
        // the disagreement, because G1 (`test_ppo_env_parity.py`) compares
        // the decisions the two sides *yield* and no tested seed happened to
        // reach a full belt. One assignment, in the constructor, is what
        // makes this path, `fork_diagnosis` and `ppo/env.py` agree by
        // construction instead of by coincidence.
        // Guardrail #1: fail loud (→ outcome="panic" row) if the run isn't
        // validly initialized, instead of silently degrading like the relic bug.
        if let Err(e) = check_run_setup_valid(&rm) {
            panic!("run setup invalid: {e}");
        }
        // Spec §8: run-level card-injection A/B. Opt-in via STS2_RUNAI_INJECT_CARDS
        // (unset → byte-identical deployed run). The agent then plays the whole
        // run normally with the injected package — no acquisition-policy confound.
        inject_cards_from_env(&mut rm);
        let inner = build_policy(&spec_copy);
        // Champion-run fixture harvest (bead `sts2-rs-cws0i`): always
        // wraps so the assembled type stays static regardless of whether
        // `--harvest-fixtures` was passed — `HarvestPolicy::choose` does a
        // single cheap `OnceLock::get()` and forwards unchanged when no
        // hook is registered, so this is a no-op at every existing call
        // site (byte-stability: flag unset ⇒ zero behavior change).
        let inner = crate::harvest::HarvestPolicy::wrap(
            inner,
            spec_copy.seed,
            spec_copy.character,
            spec_copy.ascension,
        );
        if let Some((_, detail)) = trajectory::record_config() {
            // Recording path: wrap the policy in the trajectory recorder,
            // drive, then project the native `.run` (written outside the
            // catch_unwind, in the Ok arm).
            let mut policy = RecordingPolicy::new(inner, detail);
            let report = drive_full_run(&mut rm, &mut policy, &cfg);
            let escalations = policy.inner().inner().escalations();
            let fallbacks = policy.inner().inner().combat_fallbacks();
            let ppo_peak = policy.inner().inner().ppo_peak();
            let final_relics = rm.run_state.relics.len();
            let potions_overflow = rm.run_state.potions_overflow_dropped;
            let final_view = VisibleRunView::from_run_manager(&rm);
            let second_boss = reached_second_boss(&rm);
            let meta = build_meta(&spec_copy, &label_for_run, detail, &report);
            let run_file = policy.into_run_file(&rm, &report, meta);
            (
                report,
                Vec::new(),
                escalations,
                fallbacks,
                ppo_peak,
                final_relics,
                potions_overflow,
                final_view,
                second_boss,
                Some(run_file),
            )
        } else if spec_copy.audit {
            let mut policy = AuditingPolicy::new(inner, spec_copy.seed, label_for_run);
            let report = drive_full_run(&mut rm, &mut policy, &cfg);
            let escalations = policy.inner().inner().escalations();
            let fallbacks = policy.inner().inner().combat_fallbacks();
            let ppo_peak = policy.inner().inner().ppo_peak();
            let final_relics = rm.run_state.relics.len();
            let potions_overflow = rm.run_state.potions_overflow_dropped;
            let final_view = VisibleRunView::from_run_manager(&rm);
            let second_boss = reached_second_boss(&rm);
            (
                report,
                policy.into_records(),
                escalations,
                fallbacks,
                ppo_peak,
                final_relics,
                potions_overflow,
                final_view,
                second_boss,
                None,
            )
        } else {
            let mut policy = inner;
            let report = drive_full_run(&mut rm, &mut policy, &cfg);
            let escalations = policy.inner().escalations();
            let fallbacks = policy.inner().combat_fallbacks();
            let ppo_peak = policy.inner().ppo_peak();
            let final_relics = rm.run_state.relics.len();
            let potions_overflow = rm.run_state.potions_overflow_dropped;
            let final_view = VisibleRunView::from_run_manager(&rm);
            let second_boss = reached_second_boss(&rm);
            (
                report,
                Vec::new(),
                escalations,
                fallbacks,
                ppo_peak,
                final_relics,
                potions_overflow,
                final_view,
                second_boss,
                None,
            )
        }
    }));

    let wall_ms = started.elapsed().as_millis() as u64;
    let sample_seed: u64 = std::env::var("STS2_RUNAI_DECK_SAMPLE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    match result {
        Ok((
            report,
            records,
            (escalated_t2, escalated_t3),
            (fallbacks, combat_decisions, budget_spent),
            (ppo_peak_act_slots, ppo_peak_tokens),
            final_relics,
            potions_overflow_dropped,
            final_view,
            second_boss,
            run_file,
        )) => {
            // Write the recorded `.run.gz` (best-effort — an I/O error must
            // not lose the report row).
            if let (Some(run), Some((dir, _))) = (run_file, trajectory::record_config()) {
                if let Err(e) = trajectory::write_run_file(&dir, spec.seed, &run) {
                    eprintln!(
                        "run_arena: failed to write .run for seed {}: {e}",
                        spec.seed
                    );
                }
            }
            let (outcome, won, bin) = classify_outcome(&report.outcome, report.final_phase);
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
            (
                ArenaRow {
                    seed: spec.seed,
                    character: character_token(spec.character).to_string(),
                    ascension: spec.ascension,
                    act1: spec.act1.resolved_token(spec.seed),
                    policy: label,
                    combat_budget: spec.combat.token().to_string(),
                    outcome,
                    won,
                    steps: report.steps,
                    final_phase: phase_name(report.final_phase),
                    final_hp: report.final_hp,
                    final_gold: report.final_gold,
                    act_index: report.act_index,
                    final_relics,
                    wall_ms,
                    noncompletion_bin: bin,
                    escalated_t2,
                    escalated_t3,
                    combat_decisions,
                    combat_decisions_elite: budget_spent.0,
                    combat_decisions_boss: budget_spent.1,
                    combat_decisions_motif: budget_spent.2,
                    combat_sims_spent: budget_spent.3,
                    combat_fallback_untranslatable: fallbacks.untranslatable,
                    combat_fallback_no_outcome: fallbacks.no_outcome,
                    combat_fallback_not_surfaced: fallbacks.not_surfaced,
                    final_deck,
                    final_archetype,
                    sample_seed,
                    potions_overflow_dropped,
                    ppo_peak_act_slots,
                    ppo_peak_tokens,
                    reached_second_boss: second_boss,
                    panic_message: None,
                },
                records,
            )
        }
        Err(payload) => (
            ArenaRow {
                seed: spec.seed,
                character: character_token(spec.character).to_string(),
                ascension: spec.ascension,
                act1: spec.act1.resolved_token(spec.seed),
                policy: label,
                combat_budget: spec.combat.token().to_string(),
                outcome: "panic".into(),
                won: false,
                steps: 0,
                final_phase: "Unknown".into(),
                final_hp: 0,
                final_gold: 0,
                act_index: 0,
                final_relics: 0,
                wall_ms,
                noncompletion_bin: Some("panic".into()),
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
                final_deck: vec![],
                final_archetype: "generic".to_string(),
                sample_seed,
                potions_overflow_dropped: 0,
                // The policy (and its peak counters) died with the
                // unwind — a panicking cell reports no headroom. The
                // payload below is what names the failure instead.
                reached_second_boss: false,
                ppo_peak_act_slots: 0,
                ppo_peak_tokens: 0,
                panic_message: Some(panic_payload_string(&*payload)),
            },
            Vec::new(),
        ),
    }
}

/// Recover the panic message from a [`catch_unwind`] payload.
///
/// `panic!` payloads are `&'static str` for a literal message and
/// `String` for a formatted one; anything else (a `panic_any`) has no
/// text to recover. GH #315: the arena used to discard this entirely,
/// so a panicking cell was binned as `panic` and nothing else — the
/// error text the policy went to the trouble of formatting never
/// reached the report.
fn panic_payload_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Run a batch of cells in-process with rayon (`jobs == 0` → all
/// cores). The default panic hook is suppressed for the duration so
/// per-cell `catch_unwind` panics don't spam stderr (the fuzz.rs
/// pattern). Rows come back sorted by [`ArenaRow::sort_key`].
#[must_use]
pub fn run_panel(cells: &[CellSpec], jobs: usize) -> Vec<(ArenaRow, Vec<AuditRecord>)> {
    run_panel_streaming(cells, jobs, &|_| {})
}

/// [`run_panel`], calling `on_row` with each row as its cell finishes
/// (completion order, from the worker thread), so a caller can persist
/// progress before the whole panel ends.
pub fn run_panel_streaming(
    cells: &[CellSpec],
    jobs: usize,
    on_row: &(dyn Fn(&ArenaRow) + Sync),
) -> Vec<(ArenaRow, Vec<AuditRecord>)> {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let run_all = || -> Vec<(ArenaRow, Vec<AuditRecord>)> {
        cells
            .par_iter()
            .map(|cell| {
                let result = run_cell(cell);
                on_row(&result.0);
                result
            })
            .collect()
    };
    let mut results = if jobs == 0 {
        run_all()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .expect("rayon pool");
        pool.install(run_all)
    };

    std::panic::set_hook(prev_hook);
    results.sort_by_key(|(row, _)| row.sort_key());
    results
}

/// Sweep summary: completeness % (plan §3.2) + sorted non-completion
/// bins + win rate.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub total: usize,
    /// Rows whose outcome is `won`/`lost` (a genuine run end).
    pub completed: usize,
    pub completeness_pct: f64,
    pub won: usize,
    /// Wins over *all* rows (non-completions count as not-won).
    pub win_rate_pct: f64,
    /// `(bin, count)` sorted by count desc, then bin asc.
    pub bins: Vec<(String, usize)>,
    /// Tier-2 escalations summed over all rows (plan §3.3 reporting).
    pub escalated_t2: u64,
    /// Tier-3 escalations summed over all rows.
    pub escalated_t3: u64,
    /// Combat decisions the bridge consulted the MCTS for, summed over
    /// all rows — the denominator for the #164 fallback rate.
    pub combat_decisions: u64,
    /// Simulations requested across every combat decision, summed over all
    /// rows. Two arms on the same seeds are compute-neutral iff these match
    /// (GH #446 budget reallocation).
    pub combat_sims_spent: u64,
    /// Combat MCTS→meta fallbacks on untranslatable actions (the #164
    /// gap), summed over all rows. Expected ~0 for Ironclad; a non-zero
    /// total flags reachable Channel/Evoke/Discover/multi-pick surfaces.
    pub combat_fallback_untranslatable: u64,
    /// Combat fallbacks from an empty search tree, summed over all rows.
    pub combat_fallback_no_outcome: u64,
    /// Combat fallbacks from translation drift, summed over all rows.
    pub combat_fallback_not_surfaced: u64,
}

/// Aggregate rows into a [`Summary`].
#[must_use]
pub fn summarize(rows: &[ArenaRow]) -> Summary {
    let total = rows.len();
    let completed = rows
        .iter()
        .filter(|r| r.noncompletion_bin.is_none())
        .count();
    let won = rows.iter().filter(|r| r.won).count();
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for row in rows {
        if let Some(bin) = &row.noncompletion_bin {
            *counts.entry(bin.as_str()).or_insert(0) += 1;
        }
    }
    let mut bins: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    bins.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Summary {
        total,
        completed,
        completeness_pct: if total == 0 {
            0.0
        } else {
            100.0 * completed as f64 / total as f64
        },
        won,
        win_rate_pct: if total == 0 {
            0.0
        } else {
            100.0 * won as f64 / total as f64
        },
        bins,
        escalated_t2: rows.iter().map(|r| r.escalated_t2).sum(),
        escalated_t3: rows.iter().map(|r| r.escalated_t3).sum(),
        combat_decisions: rows.iter().map(|r| r.combat_decisions).sum(),
        combat_sims_spent: rows.iter().map(|r| r.combat_sims_spent).sum(),
        combat_fallback_untranslatable: rows.iter().map(|r| r.combat_fallback_untranslatable).sum(),
        combat_fallback_no_outcome: rows.iter().map(|r| r.combat_fallback_no_outcome).sum(),
        combat_fallback_not_surfaced: rows.iter().map(|r| r.combat_fallback_not_surfaced).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_card_injection_appends_and_rejects_unknown() {
        sts2_content::cards::dispatch::ensure_installed();
        let mut rm = new_arena_run_manager(1, CharacterId::Ironclad, 0, Act1Choice::default());
        let before = rm.run_state.master_deck.len();

        let n = apply_card_injection(&mut rm, "INFLAME, BASH").expect("known tokens resolve");
        assert_eq!(n, 2);
        assert_eq!(rm.run_state.master_deck.len(), before + 2);

        // Empty / whitespace-only entries are skipped, not errors.
        assert_eq!(apply_card_injection(&mut rm, " , ").unwrap(), 0);

        // Unknown token → Err (caller aborts the arm).
        let err = apply_card_injection(&mut rm, "NOT_A_REAL_CARD_XYZ").unwrap_err();
        assert!(err.contains("NOT_A_REAL_CARD_XYZ"));
    }

    #[test]
    fn parse_inject_token_handles_upgrade_suffixes() {
        assert_eq!(parse_inject_token("INFLAME"), ("INFLAME", false));
        assert_eq!(parse_inject_token("INFLAME+"), ("INFLAME", true));
        assert_eq!(parse_inject_token("INFLAME+1"), ("INFLAME", true));
        assert_eq!(parse_inject_token("INFLAME+0"), ("INFLAME", false));
        assert_eq!(parse_inject_token("BODY_SLAM"), ("BODY_SLAM", false));
    }

    fn spec(seed: i32) -> CellSpec {
        CellSpec {
            seed,
            character: CharacterId::Ironclad,
            ascension: 0,
            policy: PolicyKind::FirstOption,
            combat: CombatBudgetKind::None,
            sims_override: None,
            boss_sims_override: None,
            max_steps: 3000,
            livelock_stall_limit: DEFAULT_LIVELOCK_STALL_LIMIT,
            act1: Act1Choice::default(),
            audit: false,
            ablate: None,
        }
    }

    /// End-to-end: with the record env set, `run_cell` writes a
    /// `<seed>.run.gz` that gunzips to native `RunHistory` JSON carrying our
    /// `sts2sim` block, and re-running the same seed is byte-identical
    /// (deterministic engine ⇒ reproducible file).
    #[test]
    fn run_cell_writes_reproducible_run_file() {
        use std::io::Read;

        let dir = std::env::temp_dir().join("sts2_rec_test_777");
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var(crate::trajectory::RECORD_DIR_ENV, &dir);
        std::env::set_var(crate::trajectory::RECORD_DETAIL_ENV, "full");

        let read_gz = |seed: i32| -> Vec<u8> {
            let path = dir.join(format!("{seed}.run.gz"));
            let raw = std::fs::read(&path).expect("run file written");
            let mut dec = flate2::read::GzDecoder::new(&raw[..]);
            let mut out = Vec::new();
            dec.read_to_end(&mut out).expect("valid gzip");
            out
        };

        let s = spec(777);
        let _ = run_cell(&s);
        let first = read_gz(777);
        let v: serde_json::Value = serde_json::from_slice(&first).expect("valid json");
        // Native keys present.
        assert_eq!(v["schema_version"], 9);
        assert!(v["players"][0]["deck"].is_array());
        assert!(v.get("map_point_history").is_some());
        assert!(v["players"][0]["character"]
            .as_str()
            .unwrap()
            .starts_with("CHARACTER."));
        // Extension block present + repro metadata.
        assert_eq!(v["sts2sim"]["meta"]["seed_int"], 777);
        assert_eq!(v["sts2sim"]["meta"]["detail"], "full");
        assert!(v["sts2sim"]["decisions"].is_array());

        // Byte-identical re-run.
        let _ = std::fs::remove_dir_all(&dir);
        let _ = run_cell(&s);
        let second = read_gz(777);
        assert_eq!(first, second, "same seed must reproduce identical run file");

        std::env::remove_var(crate::trajectory::RECORD_DIR_ENV);
        std::env::remove_var(crate::trajectory::RECORD_DETAIL_ENV);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Guardrail #4 (Addendum 23): the arena's `run_cell` construction must
    /// produce a fully-initialized run-state — the same shape the production /
    /// PyO3 path yields — because both now call the single shared
    /// `initialize_run_setup`. Mirrors `run_cell`'s exact sequence (factory →
    /// neow start → init) and asserts the setup invariant the relic-less bug
    /// violated. If a future refactor drops the init call from `run_cell`, this
    /// fails loudly instead of the agent silently playing relic-less.
    #[test]
    fn run_cell_construction_yields_initialized_run_state() {
        crate::rollout::ensure_content_installed();
        let s = spec(123);
        let rm = new_arena_run_manager(s.seed, s.character, s.ascension, s.act1);
        // The exact guardrail `run_cell` enforces at runtime.
        check_run_setup_valid(&rm).expect("arena construction must pass setup validation");
        // Relics are actually drawable (player + shared bags + boss pool).
        assert!(
            rm.run_state
                .relic_grab_bag_by_rarity
                .values()
                .any(|b| !b.is_empty()),
            "arena run starts with a stocked relic grab bag"
        );
        assert_eq!(
            rm.run_state.generated_rooms.len(),
            3,
            "arena run pre-generates all three acts"
        );
    }

    /// The #563 regression this file now carries a note about was a SECOND
    /// assignment to `interactive_potion_swap`, inside `run_cell`, with an
    /// inlined default-ON expression. No behavioural test caught it: the
    /// constructor assert below only pins `new_arena_run_manager`, and
    /// reaching a full potion belt takes a run, not a unit test.
    ///
    /// So pin the invariant directly — the flag is assigned in exactly ONE
    /// place in this file, and that place calls the shared resolver. Any
    /// second assignment (in `run_cell`, `fork_diagnosis`'s helper, anywhere)
    /// fails here, whatever value it inlines.
    #[test]
    fn potion_swap_flag_is_assigned_exactly_once_in_this_file() {
        // Scan only the non-test half — otherwise this test's own needles
        // and message strings count as assignment sites.
        let src = include_str!("arena.rs");
        let production = &src[..src.find("#[cfg(test)]").expect("test module")];
        // No trailing space in the needle: rustfmt may wrap the value onto
        // the next line, which is exactly the shape #563 shipped.
        let needle = concat!("interactive_potion_swap", " =");
        let stmts: Vec<&str> = production
            .match_indices(needle)
            .map(|(i, _)| {
                let rest = &production[i..];
                &rest[..rest.find(';').map_or(rest.len(), |e| e + 1)]
            })
            .collect();
        assert_eq!(
            stmts.len(),
            1,
            "expected one assignment site, found {}: {stmts:?}",
            stmts.len()
        );
        assert!(
            stmts[0].contains("resolve_potion_swap_flag_from_env()"),
            "the one assignment must read the shared resolver `ppo/env.py` \
             also reads, not an inlined default; found: {}",
            stmts[0]
        );
    }

    /// The default is the seed-driven roll, the same as the PyO3 `RunManager`,
    /// `PpoRunEnv` and the C# host. A report row without an `act1` field
    /// predates the roll and so still reads as Overgrowth.
    #[test]
    fn the_default_act1_is_the_seed_roll() {
        assert_eq!(Act1Choice::default(), Act1Choice::Roll);
        assert_eq!(spec(1).act1, Act1Choice::Roll);
        assert_eq!(default_act1_token(), "overgrowth");
    }

    /// Tokens round-trip, and the accepted set is exactly the PyO3 binding's
    /// (`sts2-py/src/run_manager.rs`). A token accepted on one side and not
    /// the other means a panel and a Python-built run can silently disagree
    /// about which act they walked.
    #[test]
    fn act1_tokens_round_trip_and_reject_anything_else() {
        for c in [
            Act1Choice::Overgrowth,
            Act1Choice::Underdocks,
            Act1Choice::Roll,
        ] {
            assert_eq!(Act1Choice::parse(c.token()), Ok(c));
        }
        assert!(Act1Choice::parse("hive").is_err());
        assert!(Act1Choice::parse("Overgrowth").is_err());
    }

    /// The fixed choices resolve to themselves; `roll` reproduces
    /// `RunManager::with_rolled_act1` for the same seed. If these two ever
    /// diverge, a `roll` sweep reports an act it did not actually walk.
    #[test]
    fn roll_resolves_to_the_act_the_run_manager_actually_builds() {
        crate::rollout::ensure_content_installed();
        assert_eq!(
            Act1Choice::Overgrowth.resolve(7),
            sts2_core::act_selection::ActVariant::Overgrowth
        );
        assert_eq!(
            Act1Choice::Underdocks.resolve(7),
            sts2_core::act_selection::ActVariant::Underdocks
        );
        // Both acts have to actually occur across the sample, or this test
        // would pass just as well against a `resolve` hardwired to one act.
        let mut seen = std::collections::HashSet::new();
        for seed in 0..40i32 {
            let resolved = Act1Choice::Roll.resolve(seed);
            seen.insert(resolved);
            let rm = new_arena_run_manager(seed, CharacterId::Ironclad, 0, Act1Choice::Roll);
            assert_eq!(
                rm.run_state.act1, resolved,
                "seed {seed}: reported act disagrees with the built run"
            );
        }
        assert_eq!(seen.len(), 2, "the sample must cover both act-1 models");
    }

    /// Each choice reaches the engine — the wiring test. `Roll` is covered
    /// separately above because its expected value depends on the seed.
    #[test]
    fn a_fixed_act1_choice_reaches_the_run_state() {
        crate::rollout::ensure_content_installed();
        for (choice, want) in [
            (
                Act1Choice::Overgrowth,
                sts2_core::act_selection::ActVariant::Overgrowth,
            ),
            (
                Act1Choice::Underdocks,
                sts2_core::act_selection::ActVariant::Underdocks,
            ),
        ] {
            let rm = new_arena_run_manager(4242, CharacterId::Ironclad, 0, choice);
            assert_eq!(rm.run_state.act1, want, "{}", choice.token());
            // The act must be set BEFORE `initialize_run_setup` pre-generates
            // the rooms, so assert the rooms exist and the act agrees — a
            // late `with_act1` would leave these rolled off the other act.
            assert_eq!(
                rm.run_state.generated_rooms.len(),
                3,
                "{}: rooms pre-generated",
                choice.token()
            );
        }
    }

    /// A row reports the act it RAN, not the choice that was asked for —
    /// otherwise a `roll` report cannot be split by act at analysis time.
    #[test]
    fn a_rolled_cell_reports_a_concrete_act_not_the_word_roll() {
        for seed in 0..20i32 {
            let token = Act1Choice::Roll.resolved_token(seed);
            assert!(
                token == "overgrowth" || token == "underdocks",
                "seed {seed} reported {token:?}"
            );
        }
        assert_eq!(Act1Choice::Underdocks.resolved_token(1), "underdocks");
    }

    /// GH #399 cross-language pin. G1 (`sts2-pycli/tests/test_ppo_env_parity.py`)
    /// asserts `PpoRunEnv` expert mode reproduces `run_cell` per seed, so the
    /// two constructions must set the SAME flags. `PpoRunEnv.__post_init__`
    /// (`sts2-pycli/python/ppo/env.py`) passes exactly these four to
    /// `sts2_rs.RunManager(...)`; if this test fails after an arena edit, fix
    /// the arena — or flip Python in the same commit. Flipping one side alone
    /// re-opens the invisibility bug this campaign closed (the policy never
    /// seeing a deck-mutation decision).
    #[test]
    fn run_cell_construction_enables_deck_mutation_surfaces() {
        crate::rollout::ensure_content_installed();
        let s = spec(123);
        let rm = new_arena_run_manager(s.seed, s.character, s.ascension, s.act1);
        assert!(rm.run_state.interactive_card_removal, "removal flag");
        assert!(rm.run_state.interactive_card_upgrade, "upgrade flag");
        // One flag, two queues: there is no `interactive_card_enchant` — every
        // `pending_deck_enchants` push site is gated on
        // `interactive_card_transform` (sts2-content
        // `relics/shop_event.rs:2998`, `events/shared_wave_b.rs:256`, …).
        assert!(
            rm.run_state.interactive_card_transform,
            "transform flag (also gates the enchant queue)"
        );
        assert!(rm.run_state.direct_event_relic_grant, "direct relic grant");
        // The fifth flag, `interactive_potion_swap` (GH #252/#379), is
        // env-resolved rather than always-on, so pin it against the resolver
        // instead of against `true`. This assert is the gate that was
        // missing when #563 set the flag a SECOND time inside `run_cell`
        // with an inlined default-ON expression: the arena then surfaced a
        // decision `ppo/env.py` did not, and G1 stayed green because no
        // tested seed reached a full potion belt.
        assert_eq!(
            rm.run_state.interactive_potion_swap,
            crate::policy::resolve_potion_swap_flag_from_env(),
            "potion-swap flag must come from the shared resolver, not an \
             inlined default — `ppo/env.py` reads the same one"
        );
    }

    /// 3-seed in-process smoke: every cell yields a row, rows come back
    /// sorted, and a re-run is identical modulo `wall_ms`.
    #[test]
    fn arena_smoke_three_seeds_first_option() {
        let cells: Vec<CellSpec> = [11, 5, 7].into_iter().map(spec).collect();
        let strip = |rows: Vec<(ArenaRow, Vec<AuditRecord>)>| -> Vec<ArenaRow> {
            rows.into_iter()
                .map(|(mut r, _)| {
                    r.wall_ms = 0;
                    r
                })
                .collect()
        };
        let a = strip(run_panel(&cells, 2));
        assert_eq!(a.len(), 3);
        assert_eq!(
            a.iter().map(|r| r.seed).collect::<Vec<_>>(),
            vec![5, 7, 11],
            "rows must be sorted by seed"
        );
        for row in &a {
            assert!(!row.outcome.is_empty());
            assert_eq!(row.character, "ironclad");
            // Completion ⇔ no bin.
            assert_eq!(
                row.noncompletion_bin.is_none(),
                row.outcome == "won" || row.outcome == "lost",
                "bin/outcome mismatch: {row:?}"
            );
        }
        // The streaming variant reports every row once, as it finishes.
        let streamed = std::sync::Mutex::new(Vec::new());
        let b = strip(run_panel_streaming(&cells, 1, &|row| {
            streamed.lock().unwrap().push(row.seed);
        }));
        assert_eq!(a, b, "re-run must be identical modulo wall_ms");
        let mut streamed = streamed.into_inner().unwrap();
        streamed.sort_unstable();
        assert_eq!(streamed, vec![5, 7, 11]);
    }

    #[test]
    fn summarize_counts_bins_and_completeness() {
        let cells: Vec<CellSpec> = (0..3).map(spec).collect();
        let rows: Vec<ArenaRow> = run_panel(&cells, 1).into_iter().map(|(r, _)| r).collect();
        let summary = summarize(&rows);
        assert_eq!(summary.total, 3);
        assert_eq!(
            summary.completed + summary.bins.iter().map(|(_, n)| n).sum::<usize>(),
            3
        );
    }

    #[test]
    fn parse_round_trips() {
        for kind in [
            PolicyKind::Random,
            PolicyKind::FirstOption,
            PolicyKind::Heuristic,
            PolicyKind::Tiered,
        ] {
            assert_eq!(PolicyKind::parse(kind.token()).unwrap(), kind);
        }
        for kind in [
            CombatBudgetKind::None,
            CombatBudgetKind::EvalCheap,
            CombatBudgetKind::EvalStrong,
        ] {
            assert_eq!(CombatBudgetKind::parse(kind.token()).unwrap(), kind);
        }
        assert_eq!(parse_character("ironclad").unwrap(), CharacterId::Ironclad);
        assert!(parse_character("nope").is_err());
    }

    /// `PolicyKind::parse("sampled-deck")` → `Ok(SampledDeck)` and the
    /// token round-trips (task 1.5 TDD).
    #[test]
    fn sampled_deck_parse_and_token_round_trip() {
        let kind = PolicyKind::parse("sampled-deck").expect("sampled-deck must parse");
        assert_eq!(kind, PolicyKind::SampledDeck);
        assert_eq!(kind.token(), "sampled-deck");
        // Parsing the token must be the inverse.
        assert_eq!(
            PolicyKind::parse(kind.token()).unwrap(),
            PolicyKind::SampledDeck
        );
    }

    /// One-seed in-process smoke: `run_cell` with `--policy sampled-deck`
    /// and `--combat none` returns an `ArenaRow` without panicking (task 1.5).
    #[test]
    fn sampled_deck_run_cell_no_panic() {
        let s = CellSpec {
            policy: PolicyKind::SampledDeck,
            combat: CombatBudgetKind::None,
            ..spec(42)
        };
        let (row, _records) = run_cell(&s);
        // Outcome must be one of the valid strings (not empty, not "panic").
        assert_ne!(row.outcome, "panic", "run_cell panicked: {row:?}");
        assert!(!row.outcome.is_empty());
        assert_eq!(row.character, "ironclad");
    }

    /// Task 2.1 TDD: `run_cell` for a completed run must populate
    /// `final_deck` (non-empty), `final_archetype` (known token), and
    /// the JSONL output must contain `"final_deck"` and `"sample_seed"`.
    #[test]
    fn arena_row_captures_final_deck_archetype_sample_seed() {
        let s = CellSpec {
            policy: PolicyKind::FirstOption,
            combat: CombatBudgetKind::None,
            ..spec(42)
        };
        let (row, _records) = run_cell(&s);
        // The run must complete (won or lost), not panic/timeout.
        assert!(
            row.outcome == "won" || row.outcome == "lost",
            "expected completed run, got: {}",
            row.outcome
        );
        // final_deck must be non-empty for a completed run.
        assert!(
            !row.final_deck.is_empty(),
            "final_deck must be non-empty for a completed run"
        );
        // Every entry must be a non-empty string (card id, possibly with trailing "+").
        for entry in &row.final_deck {
            assert!(!entry.is_empty(), "deck entry must be non-empty: {entry:?}");
        }
        // final_archetype must be one of the known tokens.
        let known = ["thin", "scaling", "block", "exhaust", "aggro", "generic"];
        assert!(
            known.contains(&row.final_archetype.as_str()),
            "unknown archetype token: {}",
            row.final_archetype
        );
        // sample_seed field must be present in the JSONL.
        let json = serde_json::to_string(&row).expect("ArenaRow must serialize");
        assert!(
            json.contains("\"final_deck\""),
            "JSON missing final_deck: {json}"
        );
        assert!(
            json.contains("\"sample_seed\""),
            "JSON missing sample_seed: {json}"
        );
    }

    /// `reached_second_boss` is the A9-equivalent win marker: it is
    /// `false` while the run has not stood in the A10 second-boss room,
    /// `true` once it has, and structurally impossible below A10 (no
    /// second-boss node is ever attached).
    #[test]
    fn reached_second_boss_tracks_the_a10_second_boss_room() {
        let mut rm = new_arena_run_manager(
            7,
            sts2_core::CharacterId::Ironclad,
            10,
            Act1Choice::default(),
        );
        rm.run_state.act_index = 3;
        rm.run_state.generate_map();
        let coord = {
            let map = rm.run_state.current_act_map.as_ref().expect("act-3 map");
            let sb = map.second_boss.expect("A10 attaches a second boss");
            map.point(sb).coord
        };
        assert!(
            !reached_second_boss(&rm),
            "not visited yet → A9-equivalent loss"
        );
        rm.run_state.visited_map_coords.push(coord);
        assert!(reached_second_boss(&rm), "visited → A9-equivalent win");
    }

    /// Below A10 there is no second-boss node, so the flag can never fire
    /// — an A9 sweep's rows are all `false` by construction.
    #[test]
    fn reached_second_boss_is_false_below_a10() {
        let mut rm = new_arena_run_manager(
            7,
            sts2_core::CharacterId::Ironclad,
            9,
            Act1Choice::default(),
        );
        rm.run_state.act_index = 3;
        rm.run_state.generate_map();
        assert!(rm
            .run_state
            .current_act_map
            .as_ref()
            .expect("act-3 map")
            .second_boss
            .is_none());
        assert!(!reached_second_boss(&rm));
    }
}
