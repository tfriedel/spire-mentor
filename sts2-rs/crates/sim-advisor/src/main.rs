//! `SimAdvisor`: the live-game advisor the Spire Mentor mod launches
//! (GH #1426). The flags are `realgame_advisor`'s; the mod passes `--port`,
//! `--control-port`, `--bug-report-site`, `--workbench`, `--log-dir`,
//! `--sims`, `--boss-sims` and `--threads`, and relies on the LAST
//! occurrence of a flag winning (its settings-page command line comes
//! first).
//!
//! `--replay CORPUS` / `--check GOLDEN` run a recorded session offline
//! (`realgame_advisor_replay.py`'s twin).

mod allocator;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use sim_advisor::{
    compare_replies, data_root, read_jsonl, replay, repo_root, write_replies, ReplayOptions,
};
use sts2_advisor::serve::LogLevel;

#[derive(Parser, Debug)]
#[command(
    name = "SimAdvisor",
    about = "Spire Mentor's live-game advisor",
    args_override_self = true
)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 9002)]
    port: u16,
    /// Combat search budget per decision (one tree's sims split over --threads).
    #[arg(long, default_value_t = 4000)]
    sims: u32,
    #[arg(long, default_value_t = 16000)]
    boss_sims: u32,
    /// Root-parallel search trees (clamped to the core count).
    #[arg(long, default_value_t = 4)]
    threads: u32,
    /// The advice/event JSONL log directory.
    #[arg(long)]
    log_dir: Option<PathBuf>,
    /// Seconds of socket silence before the bridge counts as dead.
    #[arg(long, default_value_t = 3600.0)]
    idle_timeout: f64,
    /// Workbench-schema run logs; empty disables.
    #[arg(long)]
    workbench_logs: Option<String>,
    /// Append every raw message from the mod to DIR/requests-<time>.jsonl.
    #[arg(long, default_value = "")]
    record_requests: String,
    /// Instinct's .e2ew weights (default: data/e2e/ under the data root).
    #[arg(long, default_value = "")]
    instinct_weights: String,
    /// Log a pruned companion search tree of SIMS sims with each combat advice.
    #[arg(long, default_value_t = 0)]
    log_search_tree: u32,
    /// `auto` spawned the dev workbench; the native advisor never does.
    #[arg(long, default_value = "off", value_parser = ["off", "auto"])]
    workbench: String,
    #[arg(long, default_value_t = 8079)]
    control_port: u16,
    /// Page origin allowed to call the control plane from a browser (repeatable).
    #[arg(long)]
    control_allow_origin: Vec<String>,
    /// Workbench site whose /intake/bugreport stores bug reports.
    #[arg(long)]
    bug_report_site: Option<String>,
    /// The browser-only workbench build the control plane hosts at /.
    #[arg(long)]
    workbench_dir: Option<PathBuf>,
    /// The data root (localization_eng/, data/ppo/, data/e2e/).
    #[arg(long)]
    data_root: Option<PathBuf>,
    /// Do not carry a per-fight engine mirror.
    #[arg(long)]
    no_mirror: bool,
    /// Exit once the process that started this one is gone (Unix; the mod
    /// passes it). Windows ties the advisor to the game with a job object.
    #[arg(long)]
    exit_with_parent: bool,
    #[arg(long, default_value_t = 2000)]
    coach_catchup_sims: u32,
    #[arg(long, default_value_t = 6.0)]
    coach_budget_s: f64,
    /// Seconds the turn plan may spend re-searching (`inf`: no cap, which a
    /// test that compares replies wants: a clock cut depends on the load).
    #[arg(long, default_value_t = sts2_advisor::turn_plan::PLAN_BUDGET_S)]
    plan_budget_s: f64,
    /// Replay a recorded corpus offline instead of serving.
    #[arg(long)]
    replay: Option<PathBuf>,
    /// With --replay: write the replies here (JSONL, .gz ok).
    #[arg(long)]
    out: Option<PathBuf>,
    /// With --replay: recorded | search | instinct.
    #[arg(long, default_value = "recorded")]
    brain: String,
    /// With --replay: stop after N advice requests (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// With --replay: no live coach.
    #[arg(long)]
    no_coach: bool,
    /// Run the packaged binary's self-check (no game needed) and exit.
    #[arg(long)]
    smoke: bool,
    /// Replay a golden's corpus with its options and compare.
    #[arg(long)]
    check: Option<PathBuf>,
}

fn main() -> ExitCode {
    allocator::never_purge();
    let args = Args::parse();
    let log = sim_advisor::stderr_logger(LogLevel::Info);
    let root = data_root(args.data_root.as_deref());
    if args.smoke {
        let weights = if args.instinct_weights.is_empty() {
            root.join(sts2_advisor::instinct_advisor::WEIGHTS_RELPATH)
        } else {
            PathBuf::from(&args.instinct_weights)
        };
        return if sim_advisor::smoke::smoke(&root, &weights) {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    if let Some(golden) = &args.check {
        return check(golden, &root, log);
    }
    if let Some(corpus) = &args.replay {
        let opts = ReplayOptions {
            brain: args.brain.clone(),
            sims: args.sims,
            boss_sims: args.boss_sims,
            threads: 1,
            coach: !args.no_coach,
            limit: args.limit,
        };
        let replies = match replay(corpus, &root, &opts, log) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("replay failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let errors = replies
            .iter()
            .filter(|r| r.pointer("/advice/hand_back").is_some())
            .count();
        println!("{} replies, {errors} advisor errors", replies.len());
        if let Some(out) = &args.out {
            let corpus = std::fs::canonicalize(corpus).unwrap_or_else(|_| corpus.clone());
            let rel = std::fs::canonicalize(repo_root())
                .ok()
                .and_then(|r| corpus.strip_prefix(r).ok().map(PathBuf::from))
                .unwrap_or(corpus);
            let meta = serde_json::json!({"corpus": rel.to_string_lossy(), "options": opts.meta()});
            if let Err(e) = write_replies(out, &meta, &replies) {
                eprintln!("could not write {}: {e}", out.display());
                return ExitCode::FAILURE;
            }
        }
        return ExitCode::SUCCESS;
    }
    serve(&args, &root, &log)
}

/// `runtime_paths.app_data_dir`: the per-user writable dir the mod shares
/// (`%LOCALAPPDATA%\SpireMentor`).
fn app_data_dir() -> PathBuf {
    sim_advisor::local_app_data().join("SpireMentor")
}

/// The browser workbench the control plane hosts by default: the mod zip's
/// `workbench/`, the sibling of the `SimAdvisor/` dir this binary lives in;
/// a dev checkout's `web/dist`.
fn default_workbench_dir(root: &std::path::Path) -> PathBuf {
    let packaged = std::env::current_exe().ok().and_then(|e| {
        e.parent()
            .and_then(|d| d.parent())
            .map(|d| d.join("workbench"))
    });
    packaged
        .filter(|d| d.is_dir())
        .unwrap_or_else(|| root.join("web").join("dist"))
}

/// The localhost control plane the workbench's live views talk to, hosting
/// the bundled workbench (and its bug reports) when there is one.
fn start_control(
    args: &Args,
    root: &std::path::Path,
    log_dir: &std::path::Path,
    advisor: &sts2_advisor::serve::Advisor,
    log: &sts2_advisor::serve::Logger,
) {
    use sim_advisor::control_http::{self, BugReportForward, Plane, StaticSite};

    let wb_dir = args
        .workbench_dir
        .clone()
        .unwrap_or_else(|| default_workbench_dir(root));
    let site = (!wb_dir.as_os_str().is_empty() && StaticSite::is_static_build(&wb_dir))
        .then(|| StaticSite::new(&wb_dir, Some(&root.join("data").join("ppo"))));
    if site.is_some() {
        log(
            LogLevel::Info,
            "realgame_advisor",
            &format!("hosting the browser workbench from {}", wb_dir.display()),
        );
    }
    // The bundled page's Report button: forwarded to the site the mod
    // passes, else saved beside the logs.
    let bug_reports = site.as_ref().map(|_| {
        BugReportForward::new(
            args.bug_report_site.as_deref(),
            log_dir
                .parent()
                .map_or_else(|| log_dir.to_path_buf(), std::path::Path::to_path_buf)
                .join("bugreports"),
        )
    });
    let last = advisor.last_combat.clone();
    let search_tree: control_http::SearchTreeFn = std::sync::Arc::new(move |sims| {
        let cached = last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(c) = cached else {
            return Err((
                409,
                serde_json::json!({"error": "no_combat_state", "detail":
                    "no combat decision cached (not in a fight, or no advice yet)"}),
            ));
        };
        let sims = sims
            .filter(|v| sts2_advisor::py::truthy(v))
            .map_or(2000, sts2_advisor::py::int_of);
        // A companion tree for a browser view: bounded, whatever was asked.
        let sims = u32::try_from(sims.clamp(1, 20_000)).unwrap_or(20_000);
        let mut tree =
            sts2_advisor::serve::companion_search_tree(&c.fight, &c.cfg, sims).map_err(|e| {
                (
                    500,
                    serde_json::json!({"error": "search_failed", "detail": e}),
                )
            })?;
        tree["state_id"] = c.state_id;
        Ok(tree)
    });
    control_http::start(
        Plane {
            control: advisor.control.clone(),
            allow_origins: args
                .control_allow_origin
                .iter()
                .map(|o| o.trim().trim_end_matches('/').to_string())
                .filter(|o| !o.is_empty())
                .collect(),
            site,
            bug_reports,
            search_tree: Some(search_tree),
            log: log.clone(),
        },
        args.control_port,
    );
}

/// Serve the mod: connect, answer until the connection drops, reconnect.
fn serve(args: &Args, root: &std::path::Path, log: &sts2_advisor::serve::Logger) -> ExitCode {
    use sts2_advisor::serve::{AdvisorLog, RequestRecorder, Settings};

    if args.exit_with_parent {
        sim_advisor::exit_with_parent(log.clone());
    }
    let cores =
        std::thread::available_parallelism().map_or(1, |n| u32::try_from(n.get()).unwrap_or(1));
    let mut threads = args.threads.max(1);
    if threads > cores {
        log(
            LogLevel::Warning,
            "realgame_advisor",
            &format!("--threads {threads} exceeds {cores} available cores; using {cores}"),
        );
        threads = cores;
    }
    log(
        LogLevel::Info,
        "realgame_advisor",
        &format!(
            "combat search: sims={} boss_sims={} threads={threads} ({cores} cores, {} sims/tree)",
            args.sims,
            args.boss_sims,
            args.sims / threads.max(1)
        ),
    );
    let log_dir = args
        .log_dir
        .clone()
        .unwrap_or_else(|| app_data_dir().join("logs").join("realgame_advisor"));
    let workbench_logs = args
        .workbench_logs
        .clone()
        .unwrap_or_else(|| app_data_dir().join("logs").to_string_lossy().into_owned());
    let settings = Settings {
        sims: args.sims,
        boss_sims: args.boss_sims,
        threads,
        log_search_tree: args.log_search_tree,
        coach_catchup_sims: args.coach_catchup_sims,
        coach_budget_s: args.coach_budget_s,
        plan_budget_s: args.plan_budget_s,
        no_mirror: args.no_mirror,
        inline_coach: false,
        divergence_log: Some(
            log_dir
                .parent()
                .map_or_else(|| log_dir.clone(), std::path::Path::to_path_buf)
                .join("diag")
                .join("mirror_divergences.jsonl"),
        ),
    };
    let weights =
        (!args.instinct_weights.is_empty()).then(|| PathBuf::from(&args.instinct_weights));
    let mut advisor = sim_advisor::build_advisor(
        root,
        settings,
        weights,
        Some(workbench_logs.as_str()),
        log.clone(),
    );
    advisor
        .control
        .set_logs_dir(Some(std::path::Path::new(&workbench_logs)));
    match AdvisorLog::new(&log_dir) {
        Ok(a) => {
            log(
                LogLevel::Info,
                "realgame_advisor",
                &format!("advisor log: {}", a.path.display()),
            );
            advisor.alog = Some(a);
        }
        Err(e) => log(
            LogLevel::Warning,
            "realgame_advisor",
            &format!("advisor log unavailable ({}): {e}", log_dir.display()),
        ),
    }
    if !args.record_requests.is_empty() {
        match RequestRecorder::new(std::path::Path::new(&args.record_requests)) {
            Ok(r) => {
                log(
                    LogLevel::Info,
                    "realgame_advisor",
                    &format!("recording raw requests to {}", r.path.display()),
                );
                advisor.recorder = Some(r);
            }
            Err(e) => log(
                LogLevel::Warning,
                "realgame_advisor",
                &format!("request recorder: {e}"),
            ),
        }
    }
    start_control(args, root, &log_dir, &advisor, log);
    let warn = |m: &str| log(LogLevel::Warning, "realgame_bridge", m);
    // A human is allowed to stare at a screen: silence is not a dead game.
    let idle = std::time::Duration::from_secs_f64(args.idle_timeout.max(1.0));
    loop {
        let bridge = match sim_advisor::bridge::Bridge::connect(
            &args.host,
            args.port,
            30,
            std::time::Duration::from_secs(2),
            &warn,
        ) {
            Ok(b) => b,
            Err(e) => {
                log(
                    LogLevel::Warning,
                    "realgame_advisor",
                    &format!("connection lost/failed: {e}; retrying in 3s"),
                );
                std::thread::sleep(std::time::Duration::from_secs(3));
                continue;
            }
        };
        log(
            LogLevel::Info,
            "realgame_advisor",
            &format!("connected to SpireMentor at {}:{}", args.host, args.port),
        );
        advisor.connect(bridge.clone());
        let why = loop {
            match bridge.receive(idle, &warn) {
                Ok(msg) => {
                    if let Err(e) = advisor.on_message(&msg) {
                        break format!("Lost connection while sending action: {e}");
                    }
                }
                Err(e) => break e.to_string(),
            }
        };
        log(
            LogLevel::Warning,
            "realgame_advisor",
            &format!("connection lost/failed: {why}; retrying in 3s"),
        );
        advisor.disconnect(true);
        bridge.close();
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}

fn check(
    golden: &std::path::Path,
    root: &std::path::Path,
    log: sts2_advisor::serve::Logger,
) -> ExitCode {
    let rows = match read_jsonl(golden) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            eprintln!("{}: empty golden", golden.display());
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{}: {e}", golden.display());
            return ExitCode::FAILURE;
        }
    };
    let meta = &rows[0]["meta"];
    let corpus = repo_root().join(meta["corpus"].as_str().unwrap_or_default());
    let opts = ReplayOptions::from_meta(&meta["options"]);
    let got = match replay(&corpus, root, &opts, log) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("replay failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let diffs = compare_replies(&got, &rows[1..], 1e-5, 1.5e-4, 20);
    if diffs.is_empty() {
        println!("{}: {} replies match", golden.display(), got.len());
        ExitCode::SUCCESS
    } else {
        for d in &diffs {
            println!("{d}");
        }
        ExitCode::FAILURE
    }
}
