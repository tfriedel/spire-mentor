//! `run_arena` — full-run arena sweep harness (full-run-AI plan §5-M0).
//!
//! Bead `sts2-rs-40ikw.4`. Thin clap wrapper over [`sts2_runai::arena`]:
//! drives N panel seeds through `drive_full_run` with a named policy ×
//! combat budget, rayon across seeds (throughput mode), and writes one
//! JSONL row per seed plus a completeness summary (plan §3.2).
//!
//! ## Per-seed wall-clock timeout (process-per-seed)
//!
//! A known engine bug (bead `sts2-rs-65mbx`) loops forever *inside a
//! single* `take_action`, so no in-process step cap or livelock watchdog
//! can catch it. When `--timeout-ms > 0` (the default, 600 s) every cell
//! runs in a child process: the parent re-execs `current_exe()` with the
//! hidden `one-cell` subcommand, polls `try_wait`, kills on deadline, and
//! bins the cell as `timeout`. One stuck seed never blocks the panel.
//! `--timeout-ms 0` disables the child layer and runs cells in-process
//! (the fuzz.rs path — faster per cell, no hang protection).
//!
//! ## Act 1 (`--act1`, GH #544)
//!
//! Act 1 has two models since #553: `overgrowth` (the default, and the only
//! one this engine had before the port) and `underdocks`. `roll` picks per
//! seed the way the real game does. Every row records the act it ACTUALLY
//! walked, so a rolled report can be split by act at analysis time.
//!
//! To measure act difficulty, run TWO fixed-act arms over the same panel and
//! `compare` them (`python/run_arena.py compare`). Comparing a fixed arm
//! against a rolled one is not a paired act comparison — the delta blends the
//! act change with per-seed difficulty; `compare` detects that and warns.
//!
//! ## Determinism
//!
//! Output rows are sorted by `(seed, character, ascension)`. Everything
//! in a row is a pure function of the cell spec **except `wall_ms`**;
//! pass `--no-wall-ms` to zero it so a re-run of the same config is
//! byte-identical (the M0 acceptance check).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;

use sts2_runai::arena::{
    character_token, parse_character, policy_label, run_cell, summarize, Act1Choice, ArenaRow,
    CellSpec, CombatBudgetKind, PolicyKind, Summary, DEFAULT_LIVELOCK_STALL_LIMIT,
    DEFAULT_MAX_STEPS,
};
use sts2_runai::calibrate::{calibrate, collect_decision_points, CalibrationConfig};
use sts2_runai::harvest::{set_harvest_hook, HarvestEntry, HarvestTierSet};
use sts2_runai::label::run_label_cell;
use sts2_runai::panel::{write_panels, Panel, PanelRow};
use sts2_runai::policy::{AblationSurface, LeafKind};
use sts2_runai::rollout::{LabelHorizon, ProbeBudget};

/// clap value-parser for `--ablate <surface>` (E2 importance leg).
fn parse_ablation(s: &str) -> Result<AblationSurface, String> {
    AblationSurface::from_token(s)
        .ok_or_else(|| format!("unknown surface '{s}' (path|card-pick|event|shop|rest)"))
}

/// clap value-parser for `--horizon <label-horizon>`.
fn parse_horizon(s: &str) -> Result<LabelHorizon, String> {
    match s {
        "elite-boss" => Ok(LabelHorizon::ElitePlusBoss),
        "chained-act" => Ok(LabelHorizon::ChainedAct),
        other => Err(format!(
            "unknown horizon '{other}' (elite-boss|chained-act)"
        )),
    }
}

/// `[player].class` PascalCase string for a champion-run fixture (the
/// shared writer's `class_name` parameter) — the `--harvest-fixtures`
/// counterpart of `corpus_gen::CliCharacter::fixture_class_name`.
fn character_class_name(c: sts2_core::potions::CharacterId) -> &'static str {
    use sts2_core::potions::CharacterId;
    match c {
        CharacterId::Ironclad => "Ironclad",
        CharacterId::Silent => "Silent",
        CharacterId::Defect => "Defect",
        CharacterId::Regent => "Regent",
        CharacterId::Necrobinder => "Necrobinder",
    }
}

/// Shared, thread-safe state for `--harvest-fixtures`: a globally-unique
/// `state_id` counter (`AtomicU32` — thread IDs must stay unique across
/// the rayon pool per the Task 10 brief) and a `Mutex`-guarded corpus
/// entry list, rewritten to `<dir>/corpus.json` after every fixture (the
/// bead `sts2-rs-19oq0` lesson: end-only writes lose everything on a
/// hang/kill).
struct HarvestState {
    dir: PathBuf,
    next_id: std::sync::atomic::AtomicU32,
    entries: std::sync::Mutex<Vec<String>>,
}

impl HarvestState {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            next_id: std::sync::atomic::AtomicU32::new(0),
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The harvest hook body: write one `ScenarioFile` TOML fixture +
    /// rewrite `corpus.json` with the new entry appended. Best-effort on
    /// I/O errors (mirrors `trajectory::write_run_file`'s "must not lose
    /// the report row" stance) — a write failure is logged, not fatal;
    /// it must not take down the sweep or lose the rest of the harvest.
    fn write_fixture(&self, rm: &sts2_core::run_manager::RunManager, entry: HarvestEntry) {
        let state_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tier_name = entry.tier.token();
        let class_name = character_class_name(entry.character);
        let snapshot_hp = rm.run_state.current_hp;
        let toml = sts2_cli::sim::scenario::export::render_run_entry_toml(
            rm,
            snapshot_hp,
            class_name,
            entry.ascension,
            &rm.run_state.potions,
        );
        let fname = format!("harvest_{state_id:05}_{tier_name}.toml");
        let fpath = self.dir.join(&fname);
        if let Err(e) = std::fs::write(&fpath, &toml) {
            eprintln!(
                "[run_arena] harvest: write {} failed ({e})",
                fpath.display()
            );
            return;
        }
        let entry_json = format!(
            "{{\"state_id\": {state_id}, \"tier\": \"{tier_name}\", \"fixture_path\": {:?}, \
             \"floor\": {}, \"seed\": {}, \"character\": {:?}, \"ascension\": {}}}",
            fpath.display().to_string(),
            entry.floor,
            entry.seed,
            character_token(entry.character),
            entry.ascension,
        );
        // The corpus.json write MUST happen while still holding `lines`:
        // releasing the lock first lets two threads' open(O_TRUNC)+write
        // sequences interleave, so a writer with MORE entries can lose the
        // disk race to one with FEWER (silent under-reporting) or leave the
        // file torn mid-write (unparseable, including at the final state).
        // Serializing the whole read-modify-write under one lock plus a
        // temp-file+rename swap (POSIX `rename` is an atomic replace on the
        // same filesystem) means every reader — including a `jq` watching
        // corpus.json mid-sweep — always sees either the old or the new
        // complete file, never a partial one.
        let mut lines = match self.entries.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        lines.push(entry_json);
        let corpus = format!(
            "{{\n  \"version\": 2,\n  \"git_sha\": \"champion-run-harvest\",\n  \
             \"generator\": \"run_arena sweep --harvest-fixtures (sts2-rs-cws0i)\",\n  \
             \"bucket_globs\": {{}},\n  \"entries\": [\n    {}\n  ]\n}}\n",
            lines.join(",\n    "),
        );
        let corpus_path = self.dir.join("corpus.json");
        let tmp_path = self.dir.join(format!("corpus.json.tmp-{state_id}"));
        if let Err(e) = std::fs::write(&tmp_path, &corpus) {
            eprintln!(
                "[run_arena] harvest: write {} failed ({e})",
                tmp_path.display()
            );
            drop(lines);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &corpus_path) {
            eprintln!(
                "[run_arena] harvest: rename {} -> {} failed ({e})",
                tmp_path.display(),
                corpus_path.display()
            );
        }
        drop(lines);
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "run_arena",
    about = "Full-run arena sweeps over committed seed panels (full-run-AI M0, bead sts2-rs-40ikw.4).",
    long_about = "Full-run arena sweeps over committed seed panels.\n\n\
        Determinism: report rows are sorted by (seed, character, ascension) and are a pure\n\
        function of the cell spec except `wall_ms`. Use --no-wall-ms to zero wall_ms so a\n\
        re-run of the same config produces a byte-identical report file."
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Sweep a seed panel (or a contiguous seed range) and write report.jsonl.
    Sweep(SweepArgs),
    /// Generate + write the three committed seed panels (dev-200 / main-500 / release-2000).
    GenPanels(GenPanelsArgs),
    /// Internal: run exactly one cell and print its JSON row on stdout
    /// (the process-per-seed child mode).
    #[command(hide = true)]
    OneCell(OneCellArgs),
    /// Offline rollout labeling campaign (plan §5/M2, bead
    /// sts2-rs-40ikw.6): N heuristic self-play runs, every card-shaped
    /// decision's options labeled via paired fight_delta probes.
    /// Writes labels.jsonl + decisions.jsonl.
    LabelCampaign(LabelCampaignArgs),
    /// E3/E4/E5 calibration (plan §6): collect decision points from
    /// self-play, score them under multiple probe budgets / K / T vs a
    /// reference, report Kendall τ + flip rates.
    Calibrate(CalibrateArgs),
    /// Proxy-fidelity / mutual-information study
    /// (docs/plans/2026-06-25-proxy-fidelity-mi-study.md): N self-play
    /// runs; at every card-shaped decision, evaluate a battery of
    /// deck-quality proxies (boss_only / elite_only / boss_elite /
    /// chain_k2 / random3 / nextact_boss) on the current deck and stamp
    /// the run's eventual outcome. Writes proxy.jsonl for proxy_mi_study.py.
    ProxyStudy(ProxyStudyArgs),
    /// Fork-at-the-fatal-fight loss diagnosis (spec
    /// docs/superpowers/specs/2026-07-17-boss-loss-diagnosis-design.md):
    /// replay losing corpus seeds, snapshot at the fatal fight, re-play it
    /// --reps times under one arm. ONE ARM PER PROCESS (honest env toggle).
    ForkFatal(ForkFatalArgs),
    ForkLabel(ForkLabelArgs),
}

#[derive(Debug, Args)]
struct ForkFatalArgs {
    /// Recorded honest corpus dir (source of losing seeds + fidelity truth).
    #[arg(long)]
    runs_dir: PathBuf,
    /// Arm: deployed|honest10x|clair|clair10x. Sets STS2_RUNAI_HONEST
    /// process-wide, so ONE arm per process.
    #[arg(long, value_parser = sts2_runai::fork_diagnosis::ForkArm::parse)]
    arm: sts2_runai::fork_diagnosis::ForkArm,
    /// Reps per fatal fight.
    #[arg(long, default_value_t = 8)]
    reps: u32,
    /// One seed per line; defaults to ALL corpus losses when omitted.
    #[arg(long)]
    seeds_file: Option<PathBuf>,
    /// Append-only JSONL output (rep + fidelity + error rows; resumable —
    /// completed (seed, arm, rep) rows are skipped on rerun).
    #[arg(long)]
    out: PathBuf,
    /// Rayon parallelism across seeds (0 = all cores).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    /// Cap the number of seeds (pilot runs).
    #[arg(long)]
    max_seeds: Option<usize>,
    /// Only replay prefixes + write fidelity rows; no fork reps.
    #[arg(long, default_value_t = false)]
    verify_only: bool,
    /// Record a per-decision combat trace on every fork rep (adds a
    /// `trace` array to each rep row; ~0.3 KB per decision).
    #[arg(long, default_value_t = false)]
    trace: bool,
    /// Arrival-state counterfactual arm: mutate every snapshot once
    /// between the phases (heal|potions|decurse|all). Prefix replays
    /// and the fidelity gate are unaffected.
    #[arg(long, value_parser = sts2_runai::fork_diagnosis::EntryMutation::parse)]
    mutate: Option<sts2_runai::fork_diagnosis::EntryMutation>,
}

#[derive(Debug, Args)]
struct ForkLabelArgs {
    /// Recorded honest corpus dir (seeds + fidelity truth).
    #[arg(long)]
    runs_dir: PathBuf,
    /// Shard output dir; one `<seed>.jsonl.gz` per seed. An existing
    /// shard = seed done (resume unit); partial work writes to a `.tmp`
    /// sibling and renames on completion.
    #[arg(long)]
    out_dir: PathBuf,
    /// Reps per NON-fatal fight (coverage stratum).
    #[arg(long, default_value_t = 4)]
    reps_regular: u32,
    /// Reps per fatal fight of a losing seed (hard-example stratum).
    #[arg(long, default_value_t = 16)]
    reps_fatal: u32,
    /// One seed per line; defaults to ALL corpus seeds when omitted.
    #[arg(long)]
    seeds_file: Option<PathBuf>,
    /// Rayon parallelism across seeds (0 = all cores).
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    /// Cap the number of seeds (pilot runs).
    #[arg(long)]
    max_seeds: Option<usize>,
}

#[derive(Debug, Args)]
struct ProxyStudyArgs {
    /// Number of self-play seeds (`seed_start..seed_start+N`).
    #[arg(long, default_value_t = 300)]
    seeds: u32,

    #[arg(long, default_value_t = 0)]
    seed_start: u32,

    #[arg(long, default_value = "ironclad", value_parser = parse_character)]
    character: sts2_core::potions::CharacterId,

    #[arg(long, default_value_t = 0)]
    ascension: u8,

    /// Trajectory-driving meta policy — the deployed agent. `ppo`
    /// (default) runs the exported PPO run-policy net
    /// (`STS2_RUNAI_PPO=<model.onnx>` or the sibling sts2-pycli snapshot);
    /// the study must be ON-POLICY (a weak heuristic driver wins ~0 and
    /// biases the outcome labels).
    #[arg(long, default_value = "ppo", value_parser = PolicyKind::parse)]
    policy: PolicyKind,

    /// Combat budget driving the trajectory. `eval-cheap` (default) runs
    /// the deployed MCTS combat; `none` = no MCTS (much weaker, ~0 wins).
    #[arg(long, default_value = "eval-cheap", value_parser = CombatBudgetKind::parse)]
    combat: CombatBudgetKind,

    /// Per-proxy MCTS probe sims-per-move.
    #[arg(long, default_value_t = 64)]
    probe_sims: u32,

    /// Per-proxy paired synthetic seeds (K).
    #[arg(long, default_value_t = 8)]
    probe_k: u32,

    /// Per-proxy truncation horizon (0 = full fight; fight_delta default 8).
    #[arg(long, default_value_t = 8)]
    probe_t: u32,

    /// Evaluate the battery on every Nth eligible decision (1 = all).
    #[arg(long, default_value_t = 1)]
    snapshot_every: u64,

    /// Worker parallelism (rayon across runs). 0 = all cores.
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// proxy.jsonl output.
    #[arg(long, default_value = "proxy.jsonl")]
    out: PathBuf,

    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    max_steps: usize,

    #[arg(long, default_value_t = DEFAULT_LIVELOCK_STALL_LIMIT)]
    stall_limit: usize,
}

#[derive(Debug, Args)]
struct SweepArgs {
    /// Seed-panel JSON file (rows {seed, ascension, character}); see `gen-panels`.
    #[arg(long, conflicts_with = "seeds")]
    panel: Option<PathBuf>,

    /// Ad-hoc sweep over integer seeds `seed_start .. seed_start+N`, at A0 and
    /// `--character` (default Ironclad). Use `--panel` for mixed rows.
    #[arg(long)]
    seeds: Option<u32>,

    /// First seed of the ad-hoc `--seeds` range.
    #[arg(long, default_value_t = 0)]
    seed_start: u32,

    /// Character for the ad-hoc `--seeds` range. Ignored with `--panel`, whose
    /// rows carry their own. Defaulted to Ironclad for backwards compatibility
    /// — it was previously HARDCODED here, so a `--seeds` sweep silently ran
    /// Ironclad content under any other character's net (see #428: a Silent
    /// headroom check measured `character: "ironclad"` rows and would have
    /// passed a budget it never tested).
    #[arg(long, default_value = "ironclad", value_parser = parse_character)]
    character: sts2_core::potions::CharacterId,

    /// Meta policy: `random` | `first` | `heuristic` | `tiered`
    /// | `value-net` | `sampled-deck` | `ppo`. `ppo` runs the exported PPO
    /// run-policy net via tract (`STS2_RUNAI_PPO=<model.onnx>` or the
    /// sibling sts2-pycli snapshot) and requires a combat budget.
    #[arg(long, default_value = "random", value_parser = PolicyKind::parse)]
    policy: PolicyKind,

    /// Combat budget (plan §3.3): `none` | `eval-cheap` (1000/4000 texel-wide)
    /// | `eval-strong` (5000/15000 texel) | `elite-focus` | `motif-focus`
    /// (GH #446 compute-neutral reallocations of `eval-cheap`; the report
    /// rows carry `combat_sims_spent` so neutrality is checked).
    #[arg(long, default_value = "none", value_parser = CombatBudgetKind::parse)]
    combat: CombatBudgetKind,

    /// Override the panel rows' ascension (paired-by-seed asc sweeps).
    #[arg(long)]
    ascension: Option<u8>,

    /// Which Act-1 model to walk (GH #544): `overgrowth` | `underdocks` |
    /// `roll` (default).
    ///
    /// Same token set and default as the PyO3 `act1` kwarg: `roll` picks the
    /// act from the seed, as the real game does. Pass `overgrowth` to compare
    /// against a number measured before the default changed (those were all
    /// Overgrowth). NOTE `overgrowth` vs `underdocks` is a PAIRED
    /// comparison over a fixed panel; `roll` is not paired with either,
    /// because the act then depends on the seed.
    #[arg(long, default_value = "roll", value_parser = Act1Choice::parse)]
    act1: Act1Choice,

    /// Debug override: non-boss sims (taints the frozen budget — recorded in the policy label).
    #[arg(long)]
    sims: Option<u32>,

    /// Debug override: boss sims.
    #[arg(long)]
    boss_sims: Option<u32>,

    /// Per-seed wall-clock timeout in ms. >0 → process-per-seed (hang-proof,
    /// the default); 0 → in-process rayon (no timeout protection).
    #[arg(long, default_value_t = 600_000)]
    timeout_ms: u64,

    /// Worker parallelism (concurrent child processes / rayon threads). 0 = all cores.
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Report JSONL output path. Parent directories are created as needed.
    ///
    /// The default deliberately lives under `target/`, which every `.gitignore`
    /// in this monorepo already covers (`**/target/` at the root, `/target` in
    /// `sts2-rs/`). It used to be a bare `report.jsonl`, which dropped scratch
    /// into whichever directory the sweep happened to run from — one such file
    /// got committed by accident in 303b73508 and every later sweep then showed
    /// up as spurious dirt in `git status`. A sweep you want to keep should pass
    /// an explicit `--out`; the resolved path is printed either way.
    #[arg(long, default_value = "target/run-arena/report.jsonl")]
    out: PathBuf,

    /// Optional decision audit log (JSONL, one record per decision — plan §3.6).
    #[arg(long)]
    audit_log: Option<PathBuf>,

    /// Hard cap on run steps (in-action hangs need --timeout-ms, not this).
    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    max_steps: usize,

    /// Livelock window (consecutive no-progress steps; 0 disables).
    #[arg(long, default_value_t = DEFAULT_LIVELOCK_STALL_LIMIT)]
    stall_limit: usize,

    /// Zero `wall_ms` in every row so re-runs are byte-identical.
    #[arg(long, default_value_t = false)]
    no_wall_ms: bool,

    /// Record a reproducible per-seed trajectory to `<dir>/<seed>.run.gz`
    /// (native `RunHistory` v9 schema + an `sts2sim` block with repro
    /// metadata, the decision stream, and combat detail). Parseable by
    /// `run_files.load_run` and the spire-codex/sts2replays corpus tooling.
    #[arg(long)]
    record_runs: Option<PathBuf>,

    /// Detail level for `--record-runs`: `summary` (native only) |
    /// `decisions` (+ off-combat choices) | `full` (+ per-ply combat).
    #[arg(long, default_value = "full")]
    record_detail: String,

    /// E2 importance leg (plan §3.5): ablate ONE surface to random while
    /// the heuristic base drives all others. `importance(C) = base − this`.
    /// `path | card-pick | event | shop | rest`. Forces the heuristic base.
    #[arg(long, value_parser = parse_ablation)]
    ablate: Option<AblationSurface>,

    /// Champion-run boss/elite/common fixture harvest (bead
    /// `sts2-rs-cws0i`): at every combat ENTRY (before any action) whose
    /// room tier is in `--harvest-tiers`, snapshot the live `RunManager`
    /// and write one `ScenarioFile` TOML into this directory, plus an
    /// incrementally-updated `<dir>/corpus.json` (flushed after every
    /// fixture). Forces the in-process sweep path regardless of
    /// `--timeout-ms` — the corpus.json writer is a single in-process
    /// `Mutex` (`unsafe_code = "deny"` workspace-wide rules out an
    /// `flock`-style cross-process lock for the default process-per-seed
    /// path), so per-seed hang protection is unavailable while
    /// harvesting.
    #[arg(long)]
    harvest_fixtures: Option<PathBuf>,

    /// Tiers to harvest: `all`, or a comma list of `common|elite|boss`.
    #[arg(long, default_value = "boss")]
    harvest_tiers: String,
}

#[derive(Debug, Args)]
struct GenPanelsArgs {
    /// Output directory for the panel JSON files.
    #[arg(long, default_value = "data/seed-panels")]
    dir: PathBuf,
}

#[derive(Debug, Args)]
struct OneCellArgs {
    #[arg(long)]
    seed: i32,
    #[arg(long, value_parser = parse_character)]
    character: sts2_core::potions::CharacterId,
    #[arg(long, default_value_t = 0)]
    ascension: u8,
    #[arg(long, default_value = "random", value_parser = PolicyKind::parse)]
    policy: PolicyKind,
    #[arg(long, default_value = "none", value_parser = CombatBudgetKind::parse)]
    combat: CombatBudgetKind,
    /// Act-1 model (threaded from the parent sweep — see `SweepArgs::act1`).
    #[arg(long, default_value = "roll", value_parser = Act1Choice::parse)]
    act1: Act1Choice,
    #[arg(long)]
    sims: Option<u32>,
    #[arg(long)]
    boss_sims: Option<u32>,
    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    max_steps: usize,
    #[arg(long, default_value_t = DEFAULT_LIVELOCK_STALL_LIMIT)]
    stall_limit: usize,
    /// Where this child writes its own audit JSONL (parent concatenates).
    #[arg(long)]
    audit_log: Option<PathBuf>,
    /// E2 ablation surface (threaded from the parent sweep).
    #[arg(long, value_parser = parse_ablation)]
    ablate: Option<AblationSurface>,
}

#[derive(Debug, Args)]
struct LabelCampaignArgs {
    /// Seed-panel JSON file (rows {seed, ascension, character}).
    #[arg(long, conflicts_with = "seeds")]
    panel: Option<PathBuf>,

    /// Ad-hoc campaign over integer seeds `seed_start .. seed_start+N`
    /// (Ironclad, A0).
    #[arg(long)]
    seeds: Option<u32>,

    #[arg(long, default_value_t = 0)]
    seed_start: u32,

    /// Combat budget the self-play runs use (plan §5/M2 default:
    /// eval-cheap).
    #[arg(long, default_value = "eval-cheap", value_parser = CombatBudgetKind::parse)]
    combat: CombatBudgetKind,

    /// Self-play driving policy: only `heuristic` is valid here.
    #[arg(long, default_value = "heuristic", value_parser = PolicyKind::parse)]
    policy: PolicyKind,

    /// Debug override: non-boss sims.
    #[arg(long)]
    sims: Option<u32>,

    /// Debug override: boss sims.
    #[arg(long)]
    boss_sims: Option<u32>,

    /// Probe budget: MCTS sims per probe move (plan default 100).
    #[arg(long, default_value_t = 100)]
    probe_sims: u32,

    /// Probe budget: paired synthetic seeds per encounter (default K=16).
    #[arg(long, default_value_t = 16)]
    k: u32,

    /// Probe budget: player-turn truncation (default T=8, the E5-calibrated
    /// horizon; T=5 was refuted — see rollout::ProbeBudget docs; 0 = full fight).
    #[arg(long, default_value_t = 8)]
    t_turns: u32,

    /// Worker parallelism (rayon across runs; combat stays
    /// single-threaded — throughput mode). 0 = all cores.
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Labels JSONL output.
    #[arg(long, default_value = "labels.jsonl")]
    out_labels: PathBuf,

    /// Decision audit-log JSONL output (§3.6).
    #[arg(long, default_value = "decisions.jsonl")]
    out_decisions: PathBuf,

    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    max_steps: usize,

    #[arg(long, default_value_t = DEFAULT_LIVELOCK_STALL_LIMIT)]
    stall_limit: usize,

    /// Labeling horizon: `elite-boss` (legacy) or `chained-act` (HP-threaded sequence).
    #[arg(long, default_value = "elite-boss", value_parser = parse_horizon)]
    horizon: LabelHorizon,

    /// `chained-act`: number of sampled normal fights before elite+boss.
    #[arg(long, default_value_t = 2)]
    chain_normals: usize,
}

#[derive(Debug, Args)]
struct CalibrateArgs {
    /// Number of collection seeds (`seed_start..seed_start+N`).
    #[arg(long, default_value_t = 64)]
    seeds: u32,

    #[arg(long, default_value_t = 0)]
    seed_start: u32,

    /// Decision points to collect (plan E3 full run: 300).
    #[arg(long, default_value_t = 300)]
    points: usize,

    /// Combat budget for the COLLECTION runs (`none` keeps collection
    /// cheap for smokes; the full calibration should use eval-cheap).
    #[arg(long, default_value = "none", value_parser = CombatBudgetKind::parse)]
    collect_combat: CombatBudgetKind,

    /// Debug override for the collection combat sims.
    #[arg(long)]
    collect_sims: Option<u32>,
    #[arg(long)]
    collect_boss_sims: Option<u32>,

    /// Base probe config the arms perturb: sims / K / T.
    #[arg(long, default_value_t = 100)]
    base_sims: u32,
    #[arg(long, default_value_t = 16)]
    base_k: u32,
    /// E5-calibrated horizon (T=5 refuted — see rollout::ProbeBudget docs).
    #[arg(long, default_value_t = 8)]
    base_t: u32,

    /// E3 sim-budget arms (comma-separated).
    #[arg(long, default_value = "1,100,1000", value_delimiter = ',')]
    budgets: Vec<u32>,
    /// E3 reference sims (full fight).
    #[arg(long, default_value_t = 5000)]
    ref_sims: u32,
    /// E3 reference K.
    #[arg(long, default_value_t = 32)]
    ref_k: u32,

    /// E4 K arms.
    #[arg(long, default_value = "8,16,32,64", value_delimiter = ',')]
    ks: Vec<u32>,
    /// E4 reference K.
    #[arg(long, default_value_t = 160)]
    kref: u32,

    /// E5 truncation arms (0 = full fight).
    #[arg(long, default_value = "3,5,8,0", value_delimiter = ',')]
    ts: Vec<u32>,

    /// Worker parallelism for the scoring phase. 0 = all cores.
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Optional JSON report output.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Cmd::Sweep(args) => sweep(&args),
        Cmd::GenPanels(args) => gen_panels(&args),
        Cmd::OneCell(args) => one_cell(&args),
        Cmd::LabelCampaign(args) => label_campaign(&args),
        Cmd::Calibrate(args) => run_calibrate(&args),
        Cmd::ProxyStudy(args) => proxy_study(&args),
        Cmd::ForkFatal(args) => cmd_fork_fatal(&args),
        Cmd::ForkLabel(args) => cmd_fork_label(&args),
    }
}

/// Fork-at-the-fatal-fight diagnosis driver (see [`ForkFatalArgs`]).
fn cmd_fork_fatal(args: &ForkFatalArgs) -> std::io::Result<()> {
    use std::io::{BufRead, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use sts2_runai::fork_diagnosis as fd;

    // Model guard: refuse to run against the wrong net.
    let ppo = std::env::var("STS2_RUNAI_PPO").map_err(|_| {
        std::io::Error::other("STS2_RUNAI_PPO must point at the corpus champion onnx")
    })?;
    eprintln!("fork-fatal: arm={} model={ppo}", args.arm.token());

    // The honest toggle is process-global and read per decision, so the
    // arm's clairvoyance must NOT be applied yet: the prefix replay below
    // must reproduce the (honest) corpus exactly. Phase A replays every
    // prefix with the env unset; the env flips to the arm's mode only
    // between the phases, when no search is in flight.
    std::env::remove_var("STS2_RUNAI_HONEST");
    // Same two-phase contract for the learned combat net (campaign #3
    // G-panel): the corpus was recorded with the TexelWide leaf, so a
    // net in phase A drifts every prefix (verified: seed 1 replays 194
    // vs 198 steps). Stash the env now, restore it between the phases —
    // `combat_net_from_env`'s cache re-reads the env per call.
    let combat_net = std::env::var("STS2_RUNAI_COMBAT_NET").ok();
    std::env::remove_var("STS2_RUNAI_COMBAT_NET");
    if let Some(p) = &combat_net {
        eprintln!("fork-fatal: combat net {p} deferred to phase B");
    }
    // Same deferral for the rollout-depth ablation knob (campaign #3
    // Task 6b): prefixes must replay at the deployed depth 200.
    let rollout_depth = std::env::var("STS2_RUNAI_ROLLOUT_DEPTH").ok();
    std::env::remove_var("STS2_RUNAI_ROLLOUT_DEPTH");
    if let Some(d) = &rollout_depth {
        eprintln!("fork-fatal: rollout depth {d} deferred to phase B");
    }

    let corpus = fd::load_corpus_summary(&args.runs_dir)?;
    let losses: std::collections::HashMap<i32, fd::CorpusRun> = corpus
        .into_iter()
        .filter(|r| !r.won)
        .map(|r| (r.seed, r))
        .collect();

    let mut seeds: Vec<i32> = if let Some(p) = &args.seeds_file {
        std::io::BufReader::new(std::fs::File::open(p)?)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| l.trim().parse().ok())
            .collect()
    } else {
        let mut s: Vec<i32> = losses.keys().copied().collect();
        s.sort_unstable();
        s
    };
    seeds.retain(|s| losses.contains_key(s));
    if let Some(m) = args.max_seeds {
        seeds.truncate(m);
    }

    // Resumability: skip (seed, arm, rep) triples already in `out`.
    let done: std::collections::HashSet<(i32, String, u32)> = std::fs::read_to_string(&args.out)
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .filter(|v| v["kind"] == "rep")
                .filter_map(|v| serde_json::from_value::<fd::ForkRepRow>(v).ok())
                .map(|r| (r.seed, r.arm, r.rep))
                .collect()
        })
        .unwrap_or_default();

    let out = std::sync::Mutex::new(std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.out)?,
    ));
    let emit = |mut v: serde_json::Value, kind: &str| {
        v["kind"] = kind.into();
        let mut w = out.lock().expect("fork-fatal: out lock");
        writeln!(w, "{v}")
            .and_then(|()| w.flush())
            .expect("fork-fatal: write out");
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()
        .expect("rayon pool");
    let arm = args.arm;
    let reps = args.reps;
    let verify_only = args.verify_only;
    let n_seeds = seeds.len();
    eprintln!("fork-fatal: {n_seeds} seeds, reps={reps}, verify_only={verify_only}");

    // ── Phase A: honest prefix replays → in-memory snapshots ────────────
    let snapshots = std::sync::Mutex::new(Vec::<(i32, fd::FatalSnapshot)>::new());
    pool.install(|| {
        seeds.par_iter().for_each(|&seed| {
            // A fully-completed seed needs no prefix replay at all.
            if !verify_only && (0..reps).all(|r| done.contains(&(seed, arm.token().into(), r))) {
                return;
            }
            let truth = &losses[&seed];
            let res = catch_unwind(|| fd::replay_to_fatal(seed, truth.ascension));
            let (snap, fidelity) = match res {
                Ok(Ok(x)) => x,
                Ok(Err(e)) => {
                    eprintln!("fork-fatal: seed {seed}: {e}");
                    emit(serde_json::json!({"seed": seed, "error": e}), "error");
                    return;
                }
                Err(_) => {
                    emit(serde_json::json!({"seed": seed, "error": "panic"}), "error");
                    return;
                }
            };
            let frow = fd::FidelityRow {
                seed,
                ok: !fidelity.won
                    && fidelity.killed_by == truth.killed_by
                    && fidelity.steps == truth.steps,
                corpus_won: truth.won,
                replay_won: fidelity.won,
                corpus_killed_by: truth.killed_by.clone(),
                replay_killed_by: fidelity.killed_by.clone(),
                corpus_steps: truth.steps,
                replay_steps: fidelity.steps,
            };
            let ok = frow.ok;
            emit(
                serde_json::to_value(&frow).expect("fidelity row json"),
                "fidelity",
            );
            if !verify_only && ok {
                snapshots
                    .lock()
                    .expect("fork-fatal: snapshots lock")
                    .push((seed, snap));
            }
        });
    });
    if verify_only {
        return Ok(());
    }

    // ── Phase B: arm forks (env flipped once, no search in flight) ──────
    if arm.clairvoyant() {
        std::env::set_var("STS2_RUNAI_HONEST", "0");
    }
    if let Some(p) = &combat_net {
        std::env::set_var("STS2_RUNAI_COMBAT_NET", p);
    }
    if let Some(d) = &rollout_depth {
        std::env::set_var("STS2_RUNAI_ROLLOUT_DEPTH", d);
    }
    let mut snapshots = snapshots.into_inner().expect("snapshots lock poisoned");
    if let Some(m) = args.mutate {
        for (_, snap) in &mut snapshots {
            fd::apply_entry_mutation(snap, m);
        }
        eprintln!(
            "fork-fatal: applied entry mutation '{}' to {} snapshots",
            m.token(),
            snapshots.len()
        );
    }
    eprintln!(
        "fork-fatal: phase B — {} snapshots x {} reps under arm {}",
        snapshots.len(),
        reps,
        arm.token()
    );
    pool.install(|| {
        snapshots.par_iter().for_each(|(seed, snap)| {
            let seed = *seed;
            for rep in 0..reps {
                if done.contains(&(seed, arm.token().to_string(), rep)) {
                    continue;
                }
                let row = catch_unwind(AssertUnwindSafe(|| {
                    fd::fork_fight(snap, arm, rep, seed, args.trace)
                }));
                match row {
                    Ok(row) => emit(serde_json::to_value(&row).expect("rep row json"), "rep"),
                    Err(_) => emit(
                        serde_json::json!({"seed": seed, "rep": rep, "error": "fork panic"}),
                        "error",
                    ),
                }
            }
        });
    });
    Ok(())
}

/// Hard-example training-row factory (see [`ForkLabelArgs`]).
fn cmd_fork_label(args: &ForkLabelArgs) -> std::io::Result<()> {
    use std::io::{BufRead, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use sts2_runai::fork_diagnosis as fd;
    use sts2_runai::fork_label as fl;

    // Model guard: the prefix replay must run the corpus champion.
    let ppo = std::env::var("STS2_RUNAI_PPO").map_err(|_| {
        std::io::Error::other("STS2_RUNAI_PPO must point at the corpus champion onnx")
    })?;
    // Label guard: rows must come from the DEPLOYED TexelWide search —
    // labeling with a learned net in the loop would bootstrap circularly.
    if std::env::var_os("STS2_RUNAI_COMBAT_NET").is_some() {
        return Err(std::io::Error::other(
            "unset STS2_RUNAI_COMBAT_NET: fork-label must label with the deployed Texel search",
        ));
    }
    // Honest-default, same contract as fork-fatal's phase A.
    std::env::remove_var("STS2_RUNAI_HONEST");
    eprintln!("fork-label: model={ppo}");

    let corpus: std::collections::HashMap<i32, fd::CorpusRun> =
        fd::load_corpus_summary(&args.runs_dir)?
            .into_iter()
            .map(|r| (r.seed, r))
            .collect();

    let mut seeds: Vec<i32> = if let Some(p) = &args.seeds_file {
        std::io::BufReader::new(std::fs::File::open(p)?)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| l.trim().parse().ok())
            .collect()
    } else {
        let mut s: Vec<i32> = corpus.keys().copied().collect();
        s.sort_unstable();
        s
    };
    seeds.retain(|s| corpus.contains_key(s));
    if let Some(m) = args.max_seeds {
        seeds.truncate(m);
    }
    std::fs::create_dir_all(&args.out_dir)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()
        .expect("rayon pool");
    let n_seeds = seeds.len();
    let done_count = std::sync::atomic::AtomicUsize::new(0);
    eprintln!(
        "fork-label: {n_seeds} seeds, reps regular/fatal = {}/{}",
        args.reps_regular, args.reps_fatal
    );

    pool.install(|| {
        seeds.par_iter().for_each(|&seed| {
            let shard = args.out_dir.join(format!("{seed}.jsonl.gz"));
            let progress = |note: &str| {
                let i = done_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!("fork-label: seed {seed} {note} ({i}/{n_seeds})");
            };
            if shard.exists() {
                progress("already done");
                return;
            }
            let tmp = args.out_dir.join(format!("{seed}.jsonl.gz.tmp"));
            let file = std::fs::File::create(&tmp).expect("fork-label: create tmp shard");
            let mut w = flate2::write::GzEncoder::new(
                std::io::BufWriter::new(file),
                flate2::Compression::fast(),
            );
            let mut emit = |v: serde_json::Value| {
                writeln!(w, "{v}").expect("fork-label: write shard");
            };
            // Any terminal per-seed failure still finalizes the shard so
            // the seed is not retried forever on resume (deterministic
            // failures stay failed; the error row records why).
            let finalize = |w: flate2::write::GzEncoder<_>| {
                w.finish()
                    .and_then(|b: std::io::BufWriter<std::fs::File>| {
                        b.into_inner().map_err(std::io::Error::other)
                    })
                    .expect("fork-label: finish shard");
                std::fs::rename(&tmp, &shard).expect("fork-label: rename shard");
            };

            let truth = &corpus[&seed];
            let replay = catch_unwind(|| fd::replay_all_fights(seed, truth.ascension));
            let (fights, fidelity) = match replay {
                Ok(Ok(x)) => x,
                Ok(Err(e)) => {
                    emit(serde_json::json!({"kind": "error", "seed": seed, "error": e}));
                    finalize(w);
                    progress("replay error");
                    return;
                }
                Err(_) => {
                    emit(
                        serde_json::json!({"kind": "error", "seed": seed, "error": "replay panic"}),
                    );
                    finalize(w);
                    progress("replay panic");
                    return;
                }
            };
            let ok = fidelity.won == truth.won
                && fidelity.steps == truth.steps
                && (fidelity.won || fidelity.killed_by == truth.killed_by);
            emit(serde_json::json!({
                "kind": "fidelity", "seed": seed, "ok": ok,
                "corpus_won": truth.won, "replay_won": fidelity.won,
                "corpus_steps": truth.steps, "replay_steps": fidelity.steps,
            }));
            if !ok {
                finalize(w);
                progress("fidelity drift — skipped");
                return;
            }

            let n_fights = fights.len();
            let mut rows = 0usize;
            for fight in &fights {
                let is_fatal = !truth.won && fight.fight_idx + 1 == n_fights;
                let reps = if is_fatal {
                    args.reps_fatal
                } else {
                    args.reps_regular
                };
                for rep in 0..reps {
                    let out = catch_unwind(AssertUnwindSafe(|| {
                        fl::label_fight(fight, rep, seed, is_fatal)
                    }));
                    match out {
                        Ok(fight_rows) => {
                            rows += fight_rows.len();
                            for row in &fight_rows {
                                emit(serde_json::to_value(row).expect("label row json"));
                            }
                        }
                        Err(_) => emit(serde_json::json!({
                            "kind": "error", "seed": seed,
                            "fight_idx": fight.fight_idx, "rep": rep,
                            "error": "label_fight panic",
                        })),
                    }
                }
            }
            finalize(w);
            progress(&format!("done: fights={n_fights} rows={rows}"));
        });
    });
    Ok(())
}

/// Drive N self-play runs and write proxy.jsonl (the proxy-fidelity study).
fn proxy_study(args: &ProxyStudyArgs) -> std::io::Result<()> {
    let probe = ProbeBudget {
        sims: args.probe_sims,
        k: args.probe_k,
        t_turns: args.probe_t,
        leaf: LeafKind::TexelWide,
        horizon: LabelHorizon::default(),
        chain_normals: 2,
    };
    let cells: Vec<CellSpec> = (0..args.seeds)
        .map(|i| CellSpec {
            seed: (args.seed_start + i) as i32,
            character: args.character,
            ascension: args.ascension,
            policy: args.policy,
            combat: args.combat,
            sims_override: None,
            boss_sims_override: None,
            max_steps: args.max_steps,
            livelock_stall_limit: args.stall_limit,
            // No `--act1` on `proxy-study`: it is an offline label-quality
            // probe, not a win-rate measurement, so it stays on the act every
            // existing probe was recorded against. Add the flag here only if
            // a probe is ever re-run per act.
            act1: Act1Choice::Overgrowth,
            audit: false,
            ablate: None,
        })
        .collect();

    eprintln!(
        "run_arena proxy-study: {} runs, policy={}, combat={}, probe(sims={} k={} t={}), snapshot_every={}, jobs={}",
        cells.len(),
        args.policy.token(),
        args.combat.token(),
        args.probe_sims,
        args.probe_k,
        args.probe_t,
        args.snapshot_every,
        if args.jobs == 0 { num_threads() } else { args.jobs },
    );
    let started = Instant::now();

    let snapshot_every = args.snapshot_every;
    let mut results = rayon_run(args.jobs, || {
        cells
            .par_iter()
            .map(|cell| sts2_runai::proxy_study::run_proxy_cell(cell, &probe, snapshot_every))
            .collect()
    });
    results.sort_by_key(|(row, _)| row.sort_key());

    let mut out = std::fs::File::create(&args.out)?;
    let mut n_rows = 0usize;
    let mut arena_rows = Vec::new();
    for (row, proxy_rows) in results {
        for pr in &proxy_rows {
            writeln!(out, "{}", serde_json::to_string(pr)?)?;
        }
        n_rows += proxy_rows.len();
        arena_rows.push(row);
    }
    print_summary(&summarize(&arena_rows), started.elapsed());
    print_realism_check(&arena_rows);
    println!(
        "proxy: {} decision rows over {} seeds → {}",
        n_rows,
        cells.len(),
        args.out.display()
    );
    Ok(())
}

fn gen_panels(args: &GenPanelsArgs) -> std::io::Result<()> {
    let paths = write_panels(&args.dir)?;
    for p in paths {
        println!("wrote {}", p.display());
    }
    Ok(())
}

/// Child mode: run one cell in-process, write the audit JSONL (if asked),
/// print exactly one JSON row on stdout.
fn one_cell(args: &OneCellArgs) -> std::io::Result<()> {
    // Suppress the default panic hook: run_cell catch_unwinds and the
    // row already carries the outcome; stderr noise would interleave
    // with the parent's terminal. `STS2_ARENA_SHOW_PANICS=1` keeps the
    // default hook so a single cell can be re-run to get the message +
    // backtrace behind the `panic` bin.
    if !std::env::var("STS2_ARENA_SHOW_PANICS").is_ok_and(|v| v == "1") {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let spec = CellSpec {
        seed: args.seed,
        character: args.character,
        ascension: args.ascension,
        policy: args.policy,
        combat: args.combat,
        sims_override: args.sims,
        boss_sims_override: args.boss_sims,
        max_steps: args.max_steps,
        livelock_stall_limit: args.stall_limit,
        act1: args.act1,
        audit: args.audit_log.is_some(),
        ablate: args.ablate,
    };
    let (row, records) = run_cell(&spec);
    if let Some(path) = &args.audit_log {
        let mut f = std::fs::File::create(path)?;
        for rec in &records {
            writeln!(f, "{}", serde_json::to_string(rec)?)?;
        }
    }
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}

/// Resolve the sweep's cell list from `--panel` / `--seeds`.
fn build_cells(args: &SweepArgs) -> std::io::Result<Vec<CellSpec>> {
    let rows: Vec<PanelRow> = if let Some(path) = &args.panel {
        Panel::load(path)?.rows
    } else if let Some(n) = args.seeds {
        (0..n)
            .map(|s| PanelRow {
                seed: (args.seed_start + s) as i32,
                ascension: 0,
                character: character_token(args.character).to_string(),
            })
            .collect()
    } else {
        return Err(std::io::Error::other("pass --panel <file> or --seeds <N>"));
    };

    rows.iter()
        .map(|row| {
            let character = parse_character(&row.character).map_err(std::io::Error::other)?;
            Ok(CellSpec {
                seed: row.seed,
                character,
                ascension: args.ascension.unwrap_or(row.ascension),
                policy: args.policy,
                combat: args.combat,
                sims_override: args.sims,
                boss_sims_override: args.boss_sims,
                max_steps: args.max_steps,
                livelock_stall_limit: args.stall_limit,
                act1: args.act1,
                audit: args.audit_log.is_some(),
                ablate: args.ablate,
            })
        })
        .collect()
}

fn sweep(args: &SweepArgs) -> std::io::Result<()> {
    // Trajectory recording is driven through the environment so it reaches
    // both the in-process rayon path and the process-per-seed children
    // (which inherit the parent env). Validate the detail token up front.
    if let Some(dir) = &args.record_runs {
        sts2_runai::trajectory::Detail::parse(&args.record_detail)
            .map_err(std::io::Error::other)?;
        std::env::set_var(sts2_runai::trajectory::RECORD_DIR_ENV, dir);
        std::env::set_var(
            sts2_runai::trajectory::RECORD_DETAIL_ENV,
            &args.record_detail,
        );
        // Stamp the engine git sha once so every recorded run carries it.
        if std::env::var(sts2_runai::trajectory::ENGINE_SHA_ENV).is_err() {
            if let Some(sha) = engine_git_sha() {
                std::env::set_var(sts2_runai::trajectory::ENGINE_SHA_ENV, sha);
            }
        }
        std::fs::create_dir_all(dir)?;
    }
    // Champion-run fixture harvest (bead `sts2-rs-cws0i`): register the
    // process-wide hook once, before any cell runs. See `HarvestState`
    // and the `--harvest-fixtures` flag doc for the in-process-only
    // rationale (a single `Mutex`, not a cross-process file lock).
    if let Some(dir) = &args.harvest_fixtures {
        let tiers = HarvestTierSet::parse(&args.harvest_tiers).map_err(std::io::Error::other)?;
        std::fs::create_dir_all(dir)?;
        let state = std::sync::Arc::new(HarvestState::new(dir.clone()));
        set_harvest_hook(
            std::sync::Arc::new(
                move |rm: &sts2_core::run_manager::RunManager, entry: HarvestEntry| {
                    state.write_fixture(rm, entry);
                },
            ),
            tiers,
        );
        if args.timeout_ms > 0 {
            eprintln!(
                "run_arena sweep: --harvest-fixtures forces the in-process path \
                 (per-seed hang protection is unavailable while harvesting — sts2-rs-cws0i)"
            );
        }
    }
    // E2 ablation wraps a base policy (the wrapper randomizes one surface and
    // delegates the rest). Only the heuristic importance leg remains —
    // reject other bases rather than silently ignoring them.
    if args.ablate.is_some() && args.policy != PolicyKind::Heuristic {
        return Err(std::io::Error::other(
            "--ablate requires --policy heuristic (the E2 importance base)",
        ));
    }
    // Fail fast (and loudly) BEFORE spawning cells: a missing/broken
    // weights file would otherwise surface as one opaque `panic` bin per
    // child process.
    if args.policy == PolicyKind::ValueNet {
        let path = std::env::var("STS2_RUNAI_VALUE").map_err(|_| {
            std::io::Error::other("--policy value-net requires STS2_RUNAI_VALUE=<weights.json>")
        })?;
        sts2_runai::eval::run_value::ValueNet::from_weights_path(std::path::Path::new(&path))
            .map_err(|e| std::io::Error::other(format!("STS2_RUNAI_VALUE: {e}")))?;
    }
    let cells = build_cells(args)?;
    eprintln!(
        "run_arena sweep: {} cells, policy={}, combat={}, act1={}, timeout_ms={}, jobs={}",
        cells.len(),
        args.policy.token(),
        args.combat.token(),
        args.act1.token(),
        args.timeout_ms,
        if args.jobs == 0 {
            num_threads()
        } else {
            args.jobs
        },
    );

    // Each row also goes to `<out>.partial` the moment its cell finishes, so
    // a long sweep shows progress and a stopped one keeps what it finished.
    // The sorted report replaces it at the end.
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let partial_path = partial_report_path(&args.out);
    let partial = std::sync::Mutex::new(std::fs::File::create(&partial_path)?);
    eprintln!("run_arena sweep: rows stream to {}", partial_path.display());
    let on_row = |row: &ArenaRow| {
        let line = serde_json::to_string(row).expect("arena row serializes");
        let mut f = partial
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Best effort: a full disk must not kill the sweep; the final
        // report is still written from memory.
        let _ = writeln!(f, "{line}").and_then(|()| f.flush());
    };

    let started = Instant::now();
    let mut results: Vec<(ArenaRow, Vec<String>)> =
        if args.timeout_ms > 0 && args.harvest_fixtures.is_none() {
            sweep_process_per_seed(args, &cells, &on_row)?
        } else {
            sts2_runai::arena::run_panel_streaming(&cells, args.jobs, &on_row)
                .into_iter()
                .map(|(row, records)| {
                    let lines = records
                        .iter()
                        .map(|r| serde_json::to_string(r).expect("audit record serializes"))
                        .collect();
                    (row, lines)
                })
                .collect()
        };

    // Deterministic output order (rows already sorted by run_panel in the
    // in-process path; the child path sorts here).
    results.sort_by_key(|(row, _)| row.sort_key());
    if args.no_wall_ms {
        for (row, _) in &mut results {
            row.wall_ms = 0;
        }
    }

    // The report (its directory was created for the partial file above).
    let mut out = std::fs::File::create(&args.out)?;
    for (row, _) in &results {
        writeln!(out, "{}", serde_json::to_string(row)?)?;
    }
    drop(partial);
    let _ = std::fs::remove_file(&partial_path);
    println!("report: {}", args.out.display());

    // audit log (concatenated in row order so re-runs are stable)
    if let Some(path) = &args.audit_log {
        let mut f = std::fs::File::create(path)?;
        for (_, lines) in &results {
            for line in lines {
                writeln!(f, "{line}")?;
            }
        }
    }

    let rows: Vec<ArenaRow> = results.into_iter().map(|(row, _)| row).collect();
    print_summary(&summarize(&rows), started.elapsed());
    print_realism_check(&rows);
    println!("report: {}", args.out.display());
    Ok(())
}

/// `report.jsonl` → `report.jsonl.partial`: where a sweep streams rows
/// before the sorted report exists.
fn partial_report_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    out.with_file_name(name)
}

fn num_threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// `git rev-parse HEAD` for the trajectory repro metadata (best-effort).
fn engine_git_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

/// Resolve `(--panel | --seeds N)` into panel rows (Ironclad/A0 for the
/// ad-hoc range), shared by `label-campaign` and `sweep`.
fn resolve_rows(
    panel: Option<&PathBuf>,
    seeds: Option<u32>,
    seed_start: u32,
) -> std::io::Result<Vec<PanelRow>> {
    if let Some(path) = panel {
        Ok(Panel::load(path)?.rows)
    } else if let Some(n) = seeds {
        Ok((0..n)
            .map(|s| PanelRow {
                seed: (seed_start + s) as i32,
                ascension: 0,
                character: "ironclad".to_string(),
            })
            .collect())
    } else {
        Err(std::io::Error::other("pass --panel <file> or --seeds <N>"))
    }
}

fn rayon_run<T: Send>(jobs: usize, f: impl Fn() -> Vec<T> + Send + Sync) -> Vec<T> {
    if jobs == 0 {
        f()
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .expect("rayon pool")
            .install(f)
    }
}

/// Offline labeling campaign (plan §5/M2; in-process rayon across runs,
/// single-thread combat — throughput mode). NOTE: no per-seed
/// wall-clock timeout layer here (the 65mbx in-action-hang risk is
/// accepted for campaigns; re-run minus the wedged seed if it bites).
fn label_campaign(args: &LabelCampaignArgs) -> std::io::Result<()> {
    let rows = resolve_rows(args.panel.as_ref(), args.seeds, args.seed_start)?;
    let probe = ProbeBudget {
        sims: args.probe_sims,
        k: args.k,
        t_turns: args.t_turns,
        leaf: LeafKind::TexelWide,
        horizon: args.horizon,
        chain_normals: args.chain_normals,
    };
    let combat = args.combat.budget(args.sims, args.boss_sims);

    // Only the heuristic self-play driver remains.
    if args.policy != PolicyKind::Heuristic {
        return Err(std::io::Error::other(format!(
            "label-campaign --policy must be heuristic, got {}",
            args.policy.token()
        )));
    }

    let cells: Vec<CellSpec> = rows
        .iter()
        .map(|row| {
            let character = parse_character(&row.character).map_err(std::io::Error::other)?;
            Ok(CellSpec {
                seed: row.seed,
                character,
                ascension: row.ascension,
                policy: args.policy,
                combat: args.combat,
                sims_override: args.sims,
                boss_sims_override: args.boss_sims,
                max_steps: args.max_steps,
                livelock_stall_limit: args.stall_limit,
                // No `--act1` on `label-campaign` either: every existing
                // label corpus is Overgrowth, and mixing acts inside one
                // corpus would silently blend two room distributions.
                act1: Act1Choice::Overgrowth,
                audit: true,
                ablate: None,
            })
        })
        .collect::<std::io::Result<_>>()?;

    eprintln!(
        "run_arena label-campaign: {} runs, self-play={}, combat={}, probe(sims={}, K={}, T={}), jobs={}",
        cells.len(),
        args.policy.token(),
        args.combat.token(),
        probe.sims,
        probe.k,
        probe.t_turns,
        if args.jobs == 0 {
            num_threads()
        } else {
            args.jobs
        },
    );
    let started = Instant::now();

    // Per-seed fault isolation + progress reporting. A single seed hitting an
    // engine assertion (e.g. the chance-node "leaf has realized state" panic,
    // sts2-mcts lib.rs, GH #98) must NOT unwind the whole rayon par_iter and
    // discard every other seed's labels — that lost a ~10h campaign on
    // 2026-06-22. On panic, drop just that seed and keep going (panic = "unwind"
    // so catch_unwind is effective; each cell builds its own MCTS tree, so there
    // is no shared state to corrupt). Replaces the old chunk+retry isolation
    // without paying the per-chunk slow-seed tail.
    let total = cells.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let labels_seen = std::sync::atomic::AtomicUsize::new(0);
    let panicked = std::sync::atomic::AtomicUsize::new(0);
    let collected: Vec<Option<_>> = rayon_run(args.jobs, || {
        cells
            .par_iter()
            .map(|cell| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_label_cell(cell, combat, &probe)
                }));
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let elapsed = started.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed.max(1e-9);
                let eta = (total - n) as f64 / rate.max(1e-9);
                if let Ok(out) = result {
                    let lab = labels_seen
                        .fetch_add(out.1.len(), std::sync::atomic::Ordering::Relaxed)
                        + out.1.len();
                    eprintln!(
                        "[progress] {n}/{total} runs ({:.1}%) | {lab} labels | {:.1} min elapsed | ETA {:.1} min",
                        100.0 * n as f64 / total as f64,
                        elapsed / 60.0,
                        eta / 60.0,
                    );
                    Some(out)
                } else {
                    let p = panicked.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    eprintln!(
                        "[panic] seed={} skipped ({p} skipped total) at {n}/{total} | {:.1} min elapsed",
                        cell.seed,
                        elapsed / 60.0,
                    );
                    None
                }
            })
            .collect()
    });
    let mut results: Vec<_> = collected.into_iter().flatten().collect();
    let n_panicked = panicked.load(std::sync::atomic::Ordering::Relaxed);
    if n_panicked > 0 {
        eprintln!(
            "WARNING: {n_panicked}/{total} seeds panicked and were skipped; labels from {} seeds",
            results.len()
        );
    }
    results.sort_by_key(|(row, _, _)| row.sort_key());

    let mut labels_out = std::fs::File::create(&args.out_labels)?;
    let mut decisions_out = std::fs::File::create(&args.out_decisions)?;
    let mut n_labels = 0usize;
    let mut n_decisions = 0usize;
    let mut arena_rows = Vec::new();
    for (row, labels, records) in results {
        for label in &labels {
            writeln!(labels_out, "{}", serde_json::to_string(label)?)?;
        }
        n_labels += labels.len();
        let mut labeled_decisions: Vec<u64> = labels.iter().map(|l| l.decision_ref.step).collect();
        labeled_decisions.dedup();
        n_decisions += labeled_decisions.len();
        for rec in &records {
            writeln!(decisions_out, "{}", serde_json::to_string(rec)?)?;
        }
        arena_rows.push(row);
    }

    print_summary(&summarize(&arena_rows), started.elapsed());
    print_realism_check(&arena_rows);
    println!(
        "labels: {} rows over {} labeled decisions → {}",
        n_labels,
        n_decisions,
        args.out_labels.display()
    );
    println!("decisions: {}", args.out_decisions.display());
    Ok(())
}

/// E3/E4/E5 calibration (plan §6).
fn run_calibrate(args: &CalibrateArgs) -> std::io::Result<()> {
    let seeds: Vec<i32> = (0..args.seeds)
        .map(|s| (args.seed_start + s) as i32)
        .collect();
    let collect_budget = args
        .collect_combat
        .budget(args.collect_sims, args.collect_boss_sims);

    eprintln!(
        "run_arena calibrate: collecting up to {} points from {} seeds (collect combat={})",
        args.points,
        seeds.len(),
        args.collect_combat.token()
    );
    let started = Instant::now();
    let points = collect_decision_points(
        &seeds,
        sts2_core::potions::CharacterId::Ironclad,
        0,
        collect_budget,
        args.points,
    );
    eprintln!(
        "collected {} decision points in {:.1}s",
        points.len(),
        started.elapsed().as_secs_f64()
    );
    if points.is_empty() {
        return Err(std::io::Error::other(
            "no card-shaped decision points collected — widen --seeds",
        ));
    }

    let cfg = CalibrationConfig {
        base: ProbeBudget {
            sims: args.base_sims,
            k: args.base_k,
            t_turns: args.base_t,
            leaf: LeafKind::TexelWide,
            horizon: sts2_runai::rollout::LabelHorizon::default(),
            chain_normals: 0,
        },
        e3_sims: args.budgets.clone(),
        e3_ref_sims: args.ref_sims,
        e3_ref_k: args.ref_k,
        e4_ks: args.ks.clone(),
        e4_ref_k: args.kref,
        e5_ts: args.ts.clone(),
    };
    let report = rayon_run(args.jobs, || vec![calibrate(&points, &cfg)])
        .pop()
        .expect("one report");

    println!("── calibration report ({} points) ──", report.points);
    println!(
        "  E3 (budget vs ref sims={} K={} T=full):",
        args.ref_sims, args.ref_k
    );
    for arm in &report.e3_budgets {
        println!(
            "    {:<10} τ={:+.3} (n={})  flip={:.1}% (n={})",
            arm.arm,
            arm.mean_tau,
            arm.tau_points,
            100.0 * arm.flip_rate,
            arm.flip_points
        );
    }
    println!("  E4 (K vs ref K={}):", args.kref);
    for arm in &report.e4_ks {
        println!(
            "    {:<10} τ={:+.3} (n={})  flip={:.1}% (n={})",
            arm.arm,
            arm.mean_tau,
            arm.tau_points,
            100.0 * arm.flip_rate,
            arm.flip_points
        );
    }
    println!("  E5 (T vs ref T=full):");
    for arm in &report.e5_ts {
        println!(
            "    {:<10} τ={:+.3} (n={})  flip={:.1}% (n={})",
            arm.arm,
            arm.mean_tau,
            arm.tau_points,
            100.0 * arm.flip_rate,
            arm.flip_points
        );
    }
    println!("  wall: {:.1}s", started.elapsed().as_secs_f64());

    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("report: {}", path.display());
    }
    Ok(())
}

/// Process-per-seed sweep: each cell is a `one-cell` child, killed (and
/// binned `timeout`) when it exceeds `--timeout-ms` wall clock.
fn sweep_process_per_seed(
    args: &SweepArgs,
    cells: &[CellSpec],
    on_row: &(dyn Fn(&ArenaRow) + Sync),
) -> std::io::Result<Vec<(ArenaRow, Vec<String>)>> {
    let exe = std::env::current_exe()?;
    let timeout = Duration::from_millis(args.timeout_ms);
    // Children write their audit slices to per-cell temp files next to
    // the final audit log; the caller concatenates in sorted row order.
    let audit_tmp_dir: Option<PathBuf> = match &args.audit_log {
        Some(path) => {
            let dir = path.with_extension("tmpdir");
            std::fs::create_dir_all(&dir)?;
            Some(dir)
        }
        None => None,
    };

    let run_all = || -> Vec<(ArenaRow, Vec<String>)> {
        cells
            .par_iter()
            .map(|cell| {
                let result = run_cell_in_child(&exe, cell, timeout, audit_tmp_dir.as_deref());
                on_row(&result.0);
                result
            })
            .collect()
    };
    let results = if args.jobs == 0 {
        run_all()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build()
            .expect("rayon pool");
        pool.install(run_all)
    };

    if let Some(dir) = audit_tmp_dir {
        let _ = std::fs::remove_dir_all(dir);
    }
    Ok(results)
}

/// Spawn one `one-cell` child, poll with `try_wait`, kill on deadline.
fn run_cell_in_child(
    exe: &Path,
    cell: &CellSpec,
    timeout: Duration,
    audit_tmp_dir: Option<&Path>,
) -> (ArenaRow, Vec<String>) {
    let mut cmd = Command::new(exe);
    cmd.arg("one-cell")
        .arg("--seed")
        .arg(cell.seed.to_string())
        .arg("--character")
        .arg(character_token(cell.character))
        .arg("--ascension")
        .arg(cell.ascension.to_string())
        .arg("--policy")
        .arg(cell.policy.token())
        .arg("--combat")
        .arg(cell.combat.token())
        // Thread the act to the child, or a process-per-seed sweep would
        // silently run Overgrowth (the child's own default) while the parent
        // reported whatever `--act1` asked for.
        .arg("--act1")
        .arg(cell.act1.token())
        .arg("--max-steps")
        .arg(cell.max_steps.to_string())
        .arg("--stall-limit")
        .arg(cell.livelock_stall_limit.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(s) = cell.sims_override {
        cmd.arg("--sims").arg(s.to_string());
    }
    if let Some(b) = cell.boss_sims_override {
        cmd.arg("--boss-sims").arg(b.to_string());
    }
    if let Some(surface) = cell.ablate {
        cmd.arg("--ablate").arg(surface.token());
    }
    let audit_path = audit_tmp_dir.map(|dir| {
        dir.join(format!(
            "cell-{}-{}-{}.jsonl",
            cell.seed,
            character_token(cell.character),
            cell.ascension
        ))
    });
    if let Some(p) = &audit_path {
        cmd.arg("--audit-log").arg(p);
    }

    let started = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return (
                failure_row(cell, "panic", &format!("spawn failed: {e}"), 0),
                Vec::new(),
            )
        }
    };

    // Poll-with-deadline (the M0 watchdog): try_wait + sleep, kill on
    // timeout. A one-row stdout payload can't fill the pipe buffer, so
    // reading after exit is safe.
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(_)) => break false,
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break true;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break false,
        }
    };
    let wall_ms = started.elapsed().as_millis() as u64;

    if timed_out {
        return (failure_row(cell, "timeout", "timeout", wall_ms), Vec::new());
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            return (
                failure_row(cell, "panic", &format!("wait failed: {e}"), wall_ms),
                Vec::new(),
            )
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let row = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .and_then(|line| serde_json::from_str::<ArenaRow>(line).ok());
    let Some(row) = row else {
        // Hard abort (stack overflow / SIGKILL / bad exit) — the child
        // never printed its row.
        return (failure_row(cell, "panic", "panic", wall_ms), Vec::new());
    };

    let audit_lines = audit_path
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|text| text.lines().map(str::to_string).collect())
        .unwrap_or_default();
    (row, audit_lines)
}

/// Synthesize a parent-side failure row (timeout / spawn failure / child
/// hard-abort). `bin` doubles as the outcome token for `timeout`.
fn failure_row(cell: &CellSpec, outcome: &str, bin: &str, wall_ms: u64) -> ArenaRow {
    ArenaRow {
        seed: cell.seed,
        character: character_token(cell.character).to_string(),
        ascension: cell.ascension,
        act1: cell.act1.resolved_token(cell.seed),
        policy: policy_label(cell),
        combat_budget: cell.combat.token().to_string(),
        outcome: outcome.to_string(),
        won: false,
        steps: 0,
        final_phase: "Unknown".to_string(),
        final_hp: 0,
        final_gold: 0,
        act_index: 0,
        final_relics: 0,
        wall_ms,
        noncompletion_bin: Some(bin.to_string()),
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
        // Key the (seed, sample_seed) join correctly for k>0 sweeps: the parent
        // run_arena process is launched with STS2_RUNAI_DECK_SAMPLE_SEED=k by
        // generate.sh, so read it here (mirrors arena.rs's panic-fallback).
        sample_seed: std::env::var("STS2_RUNAI_DECK_SAMPLE_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        potions_overflow_dropped: 0,
        ppo_peak_act_slots: 0,
        ppo_peak_tokens: 0,
        // Parent-side failure: the child was killed or never printed a
        // row, so there is no payload to recover. A child that panicked
        // *and* survived to print carries its own `panic_message`
        // through the JSON — this arm is the harder abort.
        reached_second_boss: false,
        panic_message: Some(format!("{bin} (no child row recovered)")),
    }
}

/// Guardrail #3 (2026-06-14): corpus-anchored realism check. The human
/// spire-codex reference (Addendum 22) shows winning Ironclad runs hold ~12
/// relics by act-3 entry; a healthy agent should average well above the
/// relic-less floor that hid the 2026-06-14 empty-grab-bag bug (~2). Warn
/// loudly when the mean held-relic count among runs that reached act 2+ is
/// implausibly low — the absolute, ground-truth-anchored signal that would
/// have caught the bug at the FIRST sweep (every prior experiment was purely
/// relative, so a confound hitting all arms equally stayed invisible).
fn print_realism_check(rows: &[ArenaRow]) {
    // `act_index` is 1-based; act 2+ means the run lived long enough to collect
    // relics from at least one act of elites / shops / treasure.
    let deep: Vec<&ArenaRow> = rows.iter().filter(|r| r.act_index >= 2).collect();
    if deep.is_empty() {
        return;
    }
    let mean_relics = deep.iter().map(|r| r.final_relics as f64).sum::<f64>() / deep.len() as f64;
    const FLOOR: f64 = 5.0; // well below human ~12, well above the relic-less ~2
    if mean_relics < FLOOR {
        println!(
            "  ⚠ REALISM   : mean held relics among act-2+ runs = {mean_relics:.1} \
             (human winners ~12; floor {FLOOR:.0}). Agent may be under-acquiring relics — \
             check run init / pathing / shop (guardrail #3)."
        );
    } else {
        println!(
            "  realism      : mean held relics (act-2+) = {mean_relics:.1}  [ok, floor {FLOOR:.0}]"
        );
    }
}

fn print_summary(summary: &Summary, elapsed: Duration) {
    println!("── arena summary ──");
    println!("  cells        : {}", summary.total);
    println!(
        "  completed    : {} ({:.1}%)  [outcome won|lost — plan §3.2 gate]",
        summary.completed, summary.completeness_pct
    );
    println!(
        "  won          : {} ({:.1}% of all cells)",
        summary.won, summary.win_rate_pct
    );
    println!("  wall         : {:.1}s", elapsed.as_secs_f64());
    if summary.escalated_t2 + summary.escalated_t3 > 0 {
        println!(
            "  escalations  : tier-2 {} / tier-3 {} (plan §3.3)",
            summary.escalated_t2, summary.escalated_t3
        );
    }
    // Combat MCTS→meta fallbacks (issue #164). Always print when the bridge
    // ran (combat_decisions > 0) so a clean "0 untranslatable" is an
    // affirmative signal, not silence.
    if summary.combat_decisions > 0 {
        let total = summary.combat_fallback_untranslatable
            + summary.combat_fallback_no_outcome
            + summary.combat_fallback_not_surfaced;
        let pct = 100.0 * total as f64 / summary.combat_decisions as f64;
        println!(
            "  combat fallbk: {total}/{} ({pct:.2}%) → meta  [untranslatable {} / no-outcome {} / not-surfaced {}] (issue #164)",
            summary.combat_decisions,
            summary.combat_fallback_untranslatable,
            summary.combat_fallback_no_outcome,
            summary.combat_fallback_not_surfaced,
        );
    }
    if summary.bins.is_empty() {
        println!("  non-completion bins: none");
    } else {
        println!("  non-completion bins:");
        for (bin, count) in &summary.bins {
            println!("    {count:>6}  {bin}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_campaign_parses_horizon() {
        let cli = Cli::try_parse_from([
            "run_arena",
            "label-campaign",
            "--seeds",
            "2",
            "--horizon",
            "chained-act",
            "--chain-normals",
            "3",
        ])
        .unwrap();
        match cli.command {
            Cmd::LabelCampaign(a) => {
                assert_eq!(a.chain_normals, 3);
                assert!(matches!(a.horizon, LabelHorizon::ChainedAct));
            }
            _ => panic!("expected label-campaign"),
        }
    }

    /// Fix round 1 (sts2-rs-cws0i): `HarvestState::write_fixture` used to
    /// release the `entries` lock BEFORE writing `corpus.json`, so two
    /// threads' open(O_TRUNC)+write sequences could interleave — a writer
    /// with MORE entries could lose the disk race to one with FEWER
    /// (silent under-reporting), or leave the file torn mid-write. The
    /// 8-seed smoke never exercised enough concurrency to catch it.
    ///
    /// Hammers one `HarvestState` with many threads × many fixtures each
    /// and asserts the final `corpus.json` parses and lists EXACTLY
    /// `THREADS * PER_THREAD` entries with unique `state_id`s — the
    /// invariant a torn write or a lost update would break.
    #[test]
    fn write_fixture_is_safe_under_concurrent_writers() {
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const PER_THREAD: usize = 25;

        let dir = std::env::temp_dir().join(format!(
            "sts2-cws0i-harvest-concurrency-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch harvest dir");
        let state = Arc::new(HarvestState::new(dir.clone()));

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    let rm =
                        sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
                            1000 + t as i32,
                            sts2_core::potions::CharacterId::Ironclad,
                            0,
                        );
                    for i in 0..PER_THREAD {
                        let entry = HarvestEntry {
                            tier: sts2_runai::harvest::HarvestTier::Boss,
                            floor: i as i32,
                            seed: 1000 + t as i32,
                            character: sts2_core::potions::CharacterId::Ironclad,
                            ascension: 0,
                        };
                        state.write_fixture(&rm, entry);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("writer thread must not panic");
        }

        let corpus_path = dir.join("corpus.json");
        let text = std::fs::read_to_string(&corpus_path).expect("corpus.json must be readable");
        let parsed: serde_json::Value =
            serde_json::from_str(&text).expect("corpus.json must parse (no torn write)");
        let entries = parsed["entries"]
            .as_array()
            .expect("corpus.json must have an entries array");
        assert_eq!(
            entries.len(),
            THREADS * PER_THREAD,
            "every concurrent write must land in the final corpus.json — a lost update means \
             a writer with more entries lost the disk race to one with fewer"
        );
        let mut ids: Vec<u64> = entries
            .iter()
            .map(|e| e["state_id"].as_u64().expect("state_id is a number"))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            THREADS * PER_THREAD,
            "state_ids must stay unique across the rayon/thread pool"
        );

        // Every fixture file the entries reference must also exist on disk
        // (temp+rename never leaves a corpus.json entry pointing at a file
        // that didn't make it).
        for e in entries {
            let path = e["fixture_path"]
                .as_str()
                .expect("fixture_path is a string");
            assert!(
                std::path::Path::new(path).is_file(),
                "corpus.json references {path} but it does not exist on disk"
            );
        }

        // No leftover `corpus.json.tmp-*` files: every rename either
        // succeeded (consuming the temp file) or, on failure, would have
        // logged loudly — either way none should survive a clean run.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("read scratch dir")
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("corpus.json.tmp-")
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "leftover temp files after a clean concurrent run: {leftovers:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
