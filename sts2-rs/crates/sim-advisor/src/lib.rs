//! SimAdvisor, the live-game advisor as one native binary (GH #1426 step 4).
//!
//! The advice itself is `sts2_advisor` ([`sts2_advisor::serve::Advisor`]);
//! this crate is the process around it: where its data lives, where its log
//! lines go, and the transports that feed it (a recorded corpus here, the
//! mod's socket in the binary).
//!
//! **Replay.** [`replay`] feeds a `realgame_advisor --record-requests`
//! corpus through the whole advisor and returns every reply the mod would
//! have received, as `realgame_advisor_replay.py` does for the Python
//! advisor. The goldens under `sts2-pycli/tests/fixtures/advisor_replay/`
//! were recorded from that one; [`compare_replies`] is its comparison.

pub mod bridge;
pub mod control_http;
pub mod smoke;

use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use sts2_advisor::control::Link;
use sts2_advisor::serve::{Advisor, LogLevel, Logger, Parts, Settings};

/// The data root: `localization_eng/`, `data/ppo/`, `data/e2e/`.
///
/// The packaged layout puts them next to the binary; a dev build finds them
/// in the `sts2-pycli` checkout beside this crate.
#[must_use]
pub fn data_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        if dir.join("data").join("ppo").is_dir() {
            return dir;
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../sts2-pycli")
}

/// The monorepo root of a dev checkout (goldens name their corpus relative
/// to it).
#[must_use]
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn level_name(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Debug => "DEBUG",
        LogLevel::Info => "INFO",
        LogLevel::Warning => "WARNING",
        LogLevel::Error => "ERROR",
    }
}

/// `logging.basicConfig(level=INFO, format="%(asctime)s %(levelname)s
/// %(name)s: %(message)s")` on stderr (UTC: std has no local time).
#[must_use]
pub fn stderr_logger(min: LogLevel) -> Logger {
    let rank = |l: LogLevel| match l {
        LogLevel::Debug => 0,
        LogLevel::Info => 1,
        LogLevel::Warning => 2,
        LogLevel::Error => 3,
    };
    Arc::new(move |level: LogLevel, name: &str, msg: &str| {
        if rank(level) < rank(min) {
            return;
        }
        let stamp = sts2_advisor::serve::utc_stamp();
        let _ = writeln!(
            std::io::stderr().lock(),
            "{stamp} {} {name}: {}",
            level_name(level),
            ascii_escaped(msg)
        );
    })
}

/// `msg` with every non-ASCII char as `\uXXXX` (a surrogate pair above the
/// BMP), as the Python advisor logged it. The mod decodes the console with
/// the Windows code page, so raw UTF-8 came out double-encoded in
/// `advisor_stdout.log` (`→` as `â†’`).
#[must_use]
pub fn ascii_escaped(msg: &str) -> std::borrow::Cow<'_, str> {
    if msg.is_ascii() {
        return std::borrow::Cow::Borrowed(msg);
    }
    let mut out = String::with_capacity(msg.len() + 16);
    for c in msg.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{u:04x}");
            }
        }
    }
    std::borrow::Cow::Owned(out)
}

/// .NET's `Environment.SpecialFolder.LocalApplicationData`, which the mod
/// keeps its `SpireMentor` folder (logs, config) under, so both sides agree:
/// `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS,
/// `$XDG_DATA_HOME` or `~/.local/share` elsewhere.
#[must_use]
pub fn local_app_data() -> PathBuf {
    let home = || {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map_or_else(PathBuf::new, PathBuf::from)
    };
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map_or_else(|| home().join("AppData").join("Local"), PathBuf::from)
    } else if cfg!(target_os = "macos") {
        home().join("Library").join("Application Support")
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map_or_else(|| home().join(".local").join("share"), PathBuf::from)
    }
}

/// `--exit-with-parent`: a thread that ends the process once its parent is
/// gone, so the advisor dies with the game on Linux and macOS (Windows uses
/// the mod's job object). An orphan is re-parented (to init or a subreaper),
/// so a changed parent pid is the signal; polled once a second. Not
/// `PR_SET_PDEATHSIG`: it fires when the parent *thread* that forked exits,
/// and .NET may start the child from a pool thread that ends long before the
/// game does.
pub fn exit_with_parent(log: Logger) {
    #[cfg(unix)]
    {
        let parent = std::os::unix::process::parent_id();
        let _ = std::thread::Builder::new()
            .name("parent-watch".into())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                let now = std::os::unix::process::parent_id();
                if now != parent {
                    log(
                        LogLevel::Info,
                        "realgame_advisor",
                        &format!("parent process {parent} is gone (now {now}); exiting"),
                    );
                    std::process::exit(0);
                }
            });
    }
    #[cfg(not(unix))]
    let _ = log;
}

/// A logger that drops everything.
#[must_use]
pub fn quiet_logger() -> Logger {
    Arc::new(|_, _, _| {})
}

/// Every line of a JSONL file (`.gz` too), parsed.
///
/// # Errors
///
/// The file cannot be read, or a line is not JSON.
pub fn read_jsonl(path: &Path) -> std::io::Result<Vec<Value>> {
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn std::io::BufRead> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(file)))
    } else {
        Box::new(std::io::BufReader::new(file))
    };
    reader
        .lines()
        .filter(|l| l.as_ref().map_or(true, |l| !l.trim().is_empty()))
        .map(|l| l.and_then(|l| serde_json::from_str(&l).map_err(std::io::Error::other)))
        .collect()
}

/// Write `meta` then `replies` as JSONL (`.gz` too).
///
/// # Errors
///
/// The file cannot be written.
pub fn write_replies(path: &Path, meta: &Value, replies: &[Value]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out: Box<dyn std::io::Write> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::default(),
        ))
    } else {
        Box::new(std::io::BufWriter::new(file))
    };
    writeln!(out, "{}", json!({"meta": meta}))?;
    for r in replies {
        writeln!(out, "{r}")?;
    }
    out.flush()
}

/// `_scrub`'s `\b\d+ ms\b` -> `<t> ms`: Instinct's note says how long its
/// net took, the one wall-clock value in a reply.
#[must_use]
pub fn scrub(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(scrub_ms(s)),
        Value::Array(a) => Value::Array(a.iter().map(scrub).collect()),
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), scrub(v))).collect()),
        other => other.clone(),
    }
}

fn scrub_ms(s: &str) -> String {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let starts = chars[i].is_ascii_digit() && (i == 0 || !word(chars[i - 1]));
        if starts {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let tail: String = chars[j..].iter().take(3).collect();
            let after = chars.get(j + 3);
            if tail == " ms" && after.is_none_or(|c| !word(*c)) {
                out.push_str("<t> ms");
                i = j + 3;
                continue;
            }
            out.extend(&chars[i..j]);
            i = j;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `compare_replies`: where `got` differs from `want`: numbers to `rel` or
/// `abs_tol` (probabilities are rounded to 4 decimals, so float noise across
/// CPUs can land one unit apart), everything else exactly.
#[must_use]
pub fn compare_replies(
    got: &[Value],
    want: &[Value],
    rel: f64,
    abs_tol: f64,
    max_diffs: usize,
) -> Vec<String> {
    fn walk(
        left: &Value,
        right: &Value,
        at: &str,
        rel: f64,
        abs_tol: f64,
        max: usize,
        out: &mut Vec<String>,
    ) {
        if out.len() >= max {
            return;
        }
        match (left, right) {
            (Value::Number(l), Value::Number(r)) => {
                let (lf, rf) = (l.as_f64().unwrap_or(0.0), r.as_f64().unwrap_or(0.0));
                if (lf - rf).abs() > (rel * lf.abs().max(rf.abs())).max(abs_tol) {
                    out.push(format!("{at}: {lf} != {rf}"));
                }
            }
            (Value::Object(lm), Value::Object(rm)) => {
                let mut diff: Vec<&String> = lm
                    .keys()
                    .filter(|k| !rm.contains_key(*k))
                    .chain(rm.keys().filter(|k| !lm.contains_key(*k)))
                    .collect();
                if !diff.is_empty() {
                    diff.sort();
                    out.push(format!("{at}: keys {diff:?} differ"));
                }
                for (k, v) in lm {
                    if let Some(w) = rm.get(k) {
                        walk(v, w, &format!("{at}.{k}"), rel, abs_tol, max, out);
                    }
                }
            }
            (Value::Array(la), Value::Array(ra)) => {
                if la.len() != ra.len() {
                    out.push(format!("{at}: {} items != {}", la.len(), ra.len()));
                }
                for (i, (v, w)) in la.iter().zip(ra).enumerate() {
                    walk(v, w, &format!("{at}[{i}]"), rel, abs_tol, max, out);
                }
            }
            (l, r) if l != r => {
                let s = |v: &Value| v.to_string().chars().take(120).collect::<String>();
                out.push(format!("{at}: {} != {}", s(l), s(r)));
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    if got.len() != want.len() {
        out.push(format!("{} replies != {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let at = format!(
            "#{i} {} {}",
            sts2_advisor::py::str_of(sts2_advisor::py::get(w, "action")),
            sts2_advisor::py::str_of(sts2_advisor::py::get(w, "request_id"))
        );
        walk(g, w, &at, rel, abs_tol, max_diffs, &mut out);
    }
    out
}

/// A [`Link`] that keeps every reply (a replay's transport).
#[derive(Debug, Default)]
pub struct Collect {
    pub replies: Mutex<Vec<Value>>,
}

impl Link for Collect {
    fn send(&self, msg: &Value, _echo_state_id: bool) -> std::io::Result<()> {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(msg.clone());
        Ok(())
    }

    fn newer_advice_request_pending(&self, _advice_for: &Value) -> bool {
        false
    }
}

/// How a corpus is replayed (the golden's `meta.options`).
#[derive(Debug, Clone)]
pub struct ReplayOptions {
    /// `recorded`, `search` (drop the recorded `brain`) or `instinct`.
    pub brain: String,
    pub sims: u32,
    pub boss_sims: u32,
    pub threads: u32,
    pub coach: bool,
    /// Stop after this many advice requests (0 = all).
    pub limit: usize,
}

impl ReplayOptions {
    #[must_use]
    pub fn meta(&self) -> Value {
        json!({
            "boss_sims": self.boss_sims,
            "brain": self.brain,
            "coach": self.coach,
            "limit": self.limit,
            "sims": self.sims,
            "threads": self.threads,
        })
    }

    /// From a golden's `meta.options`.
    #[must_use]
    pub fn from_meta(o: &Value) -> Self {
        let n = |k: &str, d: u32| {
            o.get(k)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(d)
        };
        Self {
            brain: o
                .get("brain")
                .and_then(Value::as_str)
                .unwrap_or("recorded")
                .to_string(),
            sims: n("sims", 4000),
            boss_sims: n("boss_sims", 16000),
            threads: n("threads", 1),
            coach: o.get("coach").and_then(Value::as_bool).unwrap_or(true),
            limit: usize::try_from(n("limit", 0)).unwrap_or(0),
        }
    }
}

/// An advisor over `data_root` with `settings`, logging to `log`.
#[must_use]
pub fn build_advisor(
    data_root: &Path,
    settings: Settings,
    instinct_weights: Option<PathBuf>,
    workbench_logs: Option<&str>,
    log: Logger,
) -> Advisor {
    use sts2_advisor::{display_names, instinct_advisor, policy::PolicyCache, workbench_log};
    let wl = log.clone();
    let (wi, ww) = (log.clone(), log.clone());
    let wlog = workbench_log::WorkbenchRunLogger::new(
        workbench_logs,
        workbench_log::Hooks {
            clock: Arc::new(sts2_advisor::serve::local_iso_now),
            info: Arc::new(move |m: &str| wi(LogLevel::Info, "realgame_workbench_log", m)),
            warn: Arc::new(move |m: &str| ww(LogLevel::Warning, "realgame_workbench_log", m)),
        },
    );
    let _ = wl;
    let weights =
        instinct_weights.unwrap_or_else(|| data_root.join(instinct_advisor::WEIGHTS_RELPATH));
    Advisor::new(Parts {
        settings,
        names: Arc::new(display_names::load(data_root)),
        policies: Arc::new(PolicyCache::new(data_root.to_path_buf())),
        // The Fool's own net sits beside the all-character one (GH #1535).
        instinct: Arc::new(
            instinct_advisor::InstinctAdvisor::new(weights)
                .with_fool(data_root.join(instinct_advisor::FOOL_WEIGHTS_RELPATH)),
        ),
        wlog: Arc::new(wlog),
        alog: None,
        recorder: None,
        log,
    })
}

/// Feed `corpus` through a fresh advisor; every reply, in order, scrubbed.
///
/// # Errors
///
/// The corpus cannot be read.
pub fn replay(
    corpus: &Path,
    data_root: &Path,
    opts: &ReplayOptions,
    log: Logger,
) -> std::io::Result<Vec<Value>> {
    let settings = Settings {
        sims: opts.sims,
        boss_sims: opts.boss_sims,
        threads: opts.threads,
        inline_coach: true,
        // No wall-clock cuts: a plan or a coach turn the clock ends early
        // depends on the machine's load, so the replies would too (GH #1784).
        plan_budget_s: f64::INFINITY,
        coach_budget_s: f64::INFINITY,
        ..Settings::default()
    };
    let mut advisor = build_advisor(data_root, settings, None, None, log);
    let link = Arc::new(Collect::default());
    if opts.coach {
        advisor.connect(link.clone());
    } else {
        // No coach: only the advice goes over the link.
        advisor.connect(link.clone());
        advisor.drop_coach();
    }
    let mut requests = 0usize;
    for mut msg in read_jsonl(corpus)? {
        if msg.get("type").and_then(Value::as_str) == Some("advice_request") {
            if opts.limit > 0 && requests >= opts.limit {
                break;
            }
            requests += 1;
            if requests.is_multiple_of(100) {
                eprintln!("replay: {requests} requests");
            }
            if let Some(m) = msg.as_object_mut() {
                match opts.brain.as_str() {
                    "search" => {
                        m.remove("brain");
                    }
                    "instinct" => {
                        m.insert("brain".into(), json!("instinct"));
                    }
                    _ => {}
                }
            }
        }
        advisor.on_message(&msg)?;
    }
    advisor.disconnect(false);
    let replies = link
        .replies
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(replies.iter().map(scrub).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_console_log_is_ascii_with_python_style_escapes() {
        assert_eq!(ascii_escaped("plain"), "plain");
        assert_eq!(
            ascii_escaped("a \u{b7} b \u{2192} c"),
            "a \\u00b7 b \\u2192 c"
        );
        assert_eq!(ascii_escaped("\u{1f600}"), "\\ud83d\\ude00");
    }

    #[test]
    fn scrub_masks_only_timings() {
        assert_eq!(
            scrub_ms("Instinct · 87% · 12 ms"),
            "Instinct · 87% · <t> ms"
        );
        assert_eq!(scrub_ms("x12 ms 3 msx 4 ms"), "x12 ms 3 msx <t> ms");
    }

    #[test]
    fn numbers_compare_with_tolerance_and_the_rest_exactly() {
        let want =
            vec![json!({"action": "advice", "advice": {"p": 0.652_304_232, "best": {"index": 0}}})];
        let noise =
            vec![json!({"action": "advice", "advice": {"p": 0.652_304_172, "best": {"index": 0}}})];
        let other =
            vec![json!({"action": "advice", "advice": {"p": 0.652_304_232, "best": {"index": 1}}})];
        assert!(compare_replies(&noise, &want, 1e-5, 1.5e-4, 20).is_empty());
        let rounded = vec![json!({"action": "advice", "advice": {"prob": 0.2678}})];
        let want2 = vec![json!({"action": "advice", "advice": {"prob": 0.2679}})];
        assert!(compare_replies(&rounded, &want2, 1e-5, 1.5e-4, 20).is_empty());
        assert!(!compare_replies(&other, &want, 1e-5, 1.5e-4, 20).is_empty());
        assert!(!compare_replies(&[], &want, 1e-5, 1.5e-4, 20).is_empty());
    }
}
