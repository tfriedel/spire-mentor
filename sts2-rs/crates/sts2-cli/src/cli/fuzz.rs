//! `sts2-cli fuzz` — seeded full-run crash/livelock sweep.
//!
//! Bead `sts2-rs` (AutoSlay-inspired full-run fuzzer, B7): the headless,
//! multi-seed driver around [`sts2_core::full_run::drive_full_run`]. For
//! each `(seed, character)` it builds a content-wired run, drives it with
//! the uniform-random [`RandomPolicy`], and classifies the outcome. A
//! panic (caught via [`std::panic::catch_unwind`] so one bad seed doesn't
//! abort the sweep) or an anomalous [`RunOutcome`] (livelock / wedged
//! state machine / surfaced-then-rejected action) is recorded as a
//! failing-seed JSON under `--record-dir` for one-line reproduction.
//!
//! `NotImplemented` outcomes are **not** failures — they are unported P9
//! phases (a clean stop), counted separately.
//!
//! The full per-step parity-fixture trace (Tier-2) is a follow-up (B6);
//! this records the seed + outcome + panic message, which is all that's
//! needed to reproduce and triage a Tier-1 crash/livelock.

use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use clap::{Args, ValueEnum};
use rayon::prelude::*;

use sts2_content::run_callbacks::new_run_manager_for_character;
use sts2_core::full_run::{
    drive_full_run, FullRunConfig, RandomPolicy, RunOutcome, DEFAULT_LIVELOCK_STALL_LIMIT,
    DEFAULT_MAX_STEPS,
};
use sts2_core::potions::CharacterId;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FuzzCharacter {
    All,
    Ironclad,
    Silent,
    Defect,
    Regent,
    Necrobinder,
}

impl FuzzCharacter {
    fn expand(self) -> Vec<CharacterId> {
        match self {
            FuzzCharacter::All => vec![
                CharacterId::Ironclad,
                CharacterId::Silent,
                CharacterId::Defect,
                CharacterId::Regent,
                CharacterId::Necrobinder,
            ],
            FuzzCharacter::Ironclad => vec![CharacterId::Ironclad],
            FuzzCharacter::Silent => vec![CharacterId::Silent],
            FuzzCharacter::Defect => vec![CharacterId::Defect],
            FuzzCharacter::Regent => vec![CharacterId::Regent],
            FuzzCharacter::Necrobinder => vec![CharacterId::Necrobinder],
        }
    }
}

#[derive(Debug, Args)]
pub struct FuzzArgs {
    /// Number of seeds to sweep (integer seeds `seed_start .. seed_start+N`).
    #[arg(long, default_value_t = 100)]
    pub seeds: u32,

    /// First integer seed in the sweep range.
    #[arg(long, default_value_t = 0)]
    pub seed_start: u32,

    /// Character(s) to fuzz (`all` expands to every class).
    #[arg(long, value_enum, default_value_t = FuzzCharacter::Ironclad)]
    pub character: FuzzCharacter,

    /// Ascension level (0-20).
    #[arg(long, default_value_t = 0)]
    pub ascension: u8,

    /// Hard cap on actions per run.
    #[arg(long, default_value_t = DEFAULT_MAX_STEPS)]
    pub max_steps: usize,

    /// Livelock window: consecutive no-progress actions before bailing
    /// (`0` disables livelock detection).
    #[arg(long, default_value_t = DEFAULT_LIVELOCK_STALL_LIMIT)]
    pub stall_limit: usize,

    /// Coverage mode: pin the player to huge HP at combat start so random
    /// play penetrates deep into the run instead of dying early.
    #[arg(long, default_value_t = false)]
    pub invincible: bool,

    /// Game build to simulate — filled from the global `--game-version`
    /// flag by `main` (GH #787), not a flag of its own.
    #[arg(skip)]
    pub game_version: sts2_core::GameVersion,

    /// Worker threads (`0` = all available cores).
    #[arg(long, default_value_t = 0)]
    pub jobs: usize,

    /// Directory to write failing-seed JSON records into (created if
    /// absent). Omit to only print the summary.
    #[arg(long)]
    pub record_dir: Option<PathBuf>,
}

/// One sweep cell's classification.
#[derive(Debug, Clone)]
enum Verdict {
    /// Reached `RunOver` (won or lost) — no bug.
    Finished { won: bool },
    /// Hit the step cap — no bug, just didn't finish.
    MaxSteps,
    /// Stopped on an unported P9 phase (or a policy halt) — expected,
    /// not a bug.
    NotImplemented,
    /// Anomalous outcome (livelock / empty-actions / rejected) — a bug.
    Anomaly { detail: String, step: usize },
    /// The driver panicked — a crash.
    Panic { message: String },
}

impl Verdict {
    fn is_failure(&self) -> bool {
        matches!(self, Verdict::Anomaly { .. } | Verdict::Panic { .. })
    }

    fn label(&self) -> &'static str {
        match self {
            Verdict::Finished { won: true } => "won",
            Verdict::Finished { won: false } => "lost",
            Verdict::MaxSteps => "max_steps",
            Verdict::NotImplemented => "not_implemented",
            Verdict::Anomaly { .. } => "anomaly",
            Verdict::Panic { .. } => "panic",
        }
    }
}

#[derive(Debug, Clone)]
struct Cell {
    seed: i32,
    character: CharacterId,
    verdict: Verdict,
    steps: usize,
}

/// Run one `(seed, character)` cell, catching any panic.
fn run_cell(
    seed: i32,
    character: CharacterId,
    game_version: sts2_core::GameVersion,
    cfg: &FullRunConfig,
) -> Cell {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut rm = new_run_manager_for_character(seed, character).with_game_version(game_version);
        // NOTE: ascension is identity at this pin (CLAUDE.md); the run
        // manager is built at A0. An ascension-aware constructor can be
        // threaded here once the run-side scaling lands.
        let mut policy = RandomPolicy::from_run_seed(seed);
        drive_full_run(&mut rm, &mut policy, cfg)
    }));

    match result {
        Ok(report) => {
            let verdict = match &report.outcome {
                RunOutcome::Won => Verdict::Finished { won: true },
                RunOutcome::Lost => Verdict::Finished { won: false },
                RunOutcome::MaxSteps => Verdict::MaxSteps,
                RunOutcome::NotImplemented { .. } | RunOutcome::PolicyHalt { .. } => {
                    Verdict::NotImplemented
                }
                anomaly @ (RunOutcome::Livelock { step, .. }
                | RunOutcome::EmptyActions { step, .. }
                | RunOutcome::Rejected { step, .. }) => Verdict::Anomaly {
                    detail: format!("{anomaly:?}"),
                    step: *step,
                },
            };
            Cell {
                seed,
                character,
                verdict,
                steps: report.steps,
            }
        }
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            Cell {
                seed,
                character,
                verdict: Verdict::Panic { message },
                steps: 0,
            }
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

fn char_token(c: CharacterId) -> &'static str {
    match c {
        CharacterId::Ironclad => "ironclad",
        CharacterId::Silent => "silent",
        CharacterId::Defect => "defect",
        CharacterId::Regent => "regent",
        CharacterId::Necrobinder => "necrobinder",
        CharacterId::Fool => "fool",
    }
}

/// `sts2-cli fuzz` entry point. Returns the number of failing runs
/// (panics + anomalies) so the caller can set a non-zero exit code.
///
/// # Errors
/// Returns an `io::Error` if writing the summary or a failure record fails.
pub fn run<W: Write>(args: FuzzArgs, out: &mut W) -> io::Result<usize> {
    let cfg = FullRunConfig {
        max_steps: args.max_steps,
        livelock_stall_limit: args.stall_limit,
        invincible: args.invincible,
        record_trace: false,
    };

    let game_version = args.game_version;
    // Build the work-list: every (seed, character) cell.
    let characters = args.character.expand();
    let mut cells_in: Vec<(i32, CharacterId)> = Vec::new();
    for s in 0..args.seeds {
        let seed = (args.seed_start + s) as i32;
        for &c in &characters {
            cells_in.push((seed, c));
        }
    }

    writeln!(
        out,
        "fuzz: {} seeds × {} character(s) = {} runs (max_steps={}, stall_limit={}, invincible={})",
        args.seeds,
        characters.len(),
        cells_in.len(),
        args.max_steps,
        args.stall_limit,
        args.invincible,
    )?;

    // Suppress the default panic hook so caught panics don't spam stderr;
    // the message is captured from the unwind payload instead.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let run_all = || -> Vec<Cell> {
        cells_in
            .par_iter()
            .map(|&(seed, c)| run_cell(seed, c, game_version, &cfg))
            .collect()
    };
    let cells: Vec<Cell> = if args.jobs == 0 {
        run_all()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build()
            .expect("rayon pool");
        pool.install(run_all)
    };

    std::panic::set_hook(prev_hook);

    // Tally + emit failure records.
    let mut counts: [usize; 6] = [0; 6]; // won, lost, max_steps, not_impl, anomaly, panic
    let mut failures: Vec<&Cell> = Vec::new();
    for cell in &cells {
        let bucket = match cell.verdict {
            Verdict::Finished { won: true } => 0,
            Verdict::Finished { won: false } => 1,
            Verdict::MaxSteps => 2,
            Verdict::NotImplemented => 3,
            Verdict::Anomaly { .. } => 4,
            Verdict::Panic { .. } => 5,
        };
        counts[bucket] += 1;
        if cell.verdict.is_failure() {
            failures.push(cell);
        }
    }

    if let Some(dir) = &args.record_dir {
        std::fs::create_dir_all(dir)?;
        for cell in &failures {
            write_failure_record(dir, cell, &args)?;
        }
    }

    writeln!(out, "── results ──")?;
    writeln!(out, "  won            : {}", counts[0])?;
    writeln!(out, "  lost           : {}", counts[1])?;
    writeln!(out, "  max_steps      : {}", counts[2])?;
    writeln!(out, "  not_implemented: {}", counts[3])?;
    writeln!(out, "  anomaly        : {}", counts[4])?;
    writeln!(out, "  panic          : {}", counts[5])?;

    if failures.is_empty() {
        writeln!(out, "✔ no crashes or anomalies across {} runs", cells.len())?;
    } else {
        writeln!(out, "✘ {} failing run(s):", failures.len())?;
        for cell in &failures {
            writeln!(
                out,
                "    seed={} char={} {} ({})",
                cell.seed,
                char_token(cell.character),
                cell.verdict.label(),
                failure_detail(&cell.verdict),
            )?;
        }
        if args.record_dir.is_none() {
            writeln!(out, "  (pass --record-dir <path> to save repro records)")?;
        }
    }

    Ok(failures.len())
}

fn failure_detail(v: &Verdict) -> String {
    match v {
        Verdict::Anomaly { detail, step } => format!("step={step} {detail}"),
        Verdict::Panic { message } => message.clone(),
        _ => String::new(),
    }
}

fn write_failure_record(dir: &std::path::Path, cell: &Cell, args: &FuzzArgs) -> io::Result<()> {
    let json = serde_json::json!({
        "seed": cell.seed,
        "character": char_token(cell.character),
        "ascension": args.ascension,
        "policy": "random",
        "outcome": cell.verdict.label(),
        "detail": failure_detail(&cell.verdict),
        "steps": cell.steps,
        "config": {
            "max_steps": args.max_steps,
            "stall_limit": args.stall_limit,
            "invincible": args.invincible,
        },
    });
    let fname = format!(
        "seed-{}-{}-{}.json",
        cell.seed,
        char_token(cell.character),
        cell.verdict.label()
    );
    std::fs::write(dir.join(fname), serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;

    #[test]
    fn panic_message_extracts_str_and_string() {
        let p1 = catch_unwind(|| panic!("boom-str")).unwrap_err();
        assert!(panic_message(p1.as_ref()).contains("boom-str"));

        let p2 = catch_unwind(|| panic!("{}", String::from("boom-string"))).unwrap_err();
        assert!(panic_message(p2.as_ref()).contains("boom-string"));
    }

    /// End-to-end: a small Ironclad sweep runs through `run`, catches no
    /// panics, and prints a results summary. Exercises the rayon +
    /// catch_unwind path without needing an actual engine crash.
    #[test]
    fn small_sweep_runs_and_summarizes() {
        let args = FuzzArgs {
            seeds: 5,
            seed_start: 0,
            character: FuzzCharacter::Ironclad,
            ascension: 0,
            max_steps: 2000,
            stall_limit: DEFAULT_LIVELOCK_STALL_LIMIT,
            invincible: false,
            jobs: 1,
            record_dir: None,
            game_version: sts2_core::GameVersion::default(),
        };
        let mut out: Vec<u8> = Vec::new();
        run(args, &mut out).expect("sweep run ok");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("results"), "missing results header: {text}");
        assert!(text.contains("5 runs") || text.contains("= 5"), "{text}");
    }
}
