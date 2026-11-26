//! The advisor's message loop (GH #1426 step 4; was `realgame_advisor`'s
//! `serve` / `_serve_connected` / `_handle_advice_request`).
//!
//! [`Advisor`] owns everything a session carries: the fight mirror, the
//! per-run scratch, Instinct, the run-policy champions, the live run tree,
//! the logs and, per connection, the live coach. [`Advisor::on_message`]
//! handles one message from the mod and answers over the [`Link`]; the
//! transport (the TCP socket, or a recorded corpus) is the caller's.

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde_json::{json, Map, Value};
use sts2_core::game_version::GameVersion;

use crate::advisor_mirror::{AdvisorMirror, Level};
use crate::coach_grader::{
    CoachEngine, CoachGrader, Entry, EntryKind, Executor, GraderConfig, GraderThread,
    InlineExecutor,
};
use crate::combat_mirror::Mirrored;
use crate::control::{ControlState, Link, BRIDGE_MESSAGE_TYPES};
use crate::display_names::Names;
use crate::i18n::{is_zh, tr, trf};
use crate::instinct_advisor::{Context, InstinctAdvisor, InstinctError};
use crate::livetree::LiveRunTree;
use crate::policy::PolicyCache;
use crate::py::{self, get, truthy};
use crate::search::RecommendConfig;
use crate::workbench_log::WorkbenchRunLogger;

/// A log line's level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl From<Level> for LogLevel {
    fn from(l: Level) -> Self {
        match l {
            Level::Debug => Self::Debug,
            Level::Info => Self::Info,
            Level::Warning => Self::Warning,
        }
    }
}

/// Where the advisor's log lines go: `(level, logger name, message)`.
pub type Logger = Arc<dyn Fn(LogLevel, &str, &str) + Send + Sync>;

/// The honest search: what the mod's player can know (no peeking at the
/// draw order), at the run agent's constants.
pub const HONEST_HIDDEN_INFO: &str = "beliefdraw";
pub const HONEST_C_PUCT: f64 = 2.5;
pub const HONEST_DRAW_WIDENING_C: f64 = 2.0;

/// Acts in a standard run; the fallback for a mod too old to send
/// `run_state.is_last_act`.
pub const STANDARD_ACT_COUNT: i64 = 3;

/// What this advisor advertises on every `advice` reply (PROTOCOL.md v5).
#[must_use]
pub fn advisor_capabilities_base() -> Map<String, Value> {
    let Value::Object(m) = json!({"coach": 2, "potion_swap": 1, "crystal_sphere": 1}) else {
        unreachable!()
    };
    m
}

/// The `advice_for` surfaces answered by a screen policy.
pub const SCREEN_ADVICE_FOR: [&str; 12] = [
    "card_reward",
    "card_select",
    "map_select",
    "shop",
    "event",
    "rest_site",
    "boss_relic",
    "treasure",
    "card_bundle",
    "potion_swap",
    "crystal_sphere",
    "reward_screen",
];

/// Characters that exist only from a game build on, by id: `(id, display
/// name, first build)`. On that build and later the engine simulates them;
/// a board naming one on an older build is refused, since the older engine
/// has no rules for them.
const BUILD_GATED_CHARACTERS: [(&str, &str, GameVersion); 1] =
    [("FOOL", "The Fool", GameVersion::V0_113_0)];

/// `check_supported_characters`: the first character on the board (its own
/// or an ally's) that the reported build has no rules for, as `(raw id,
/// player-facing note)`. An unknown build is not judged here: the
/// game-version refusal answers it first.
#[must_use]
pub fn unsupported_character(msg: &Value) -> Option<(String, String)> {
    let build = match get(msg, "game_version") {
        Value::String(s) if !s.trim().is_empty() => s.parse::<GameVersion>().ok()?,
        Value::Null | Value::String(_) => GameVersion::V0_107_1,
        _ => return None,
    };
    let mut ids = vec![get(get(msg, "run_state"), "character")];
    ids.extend(
        py::list(msg, "allies")
            .iter()
            .filter(|a| a.is_object())
            .map(|a| get(a, "character")),
    );
    for cid in ids {
        let Some(s) = cid.as_str() else { continue };
        let up = s.trim().to_uppercase();
        if let Some((_, name, first)) = BUILD_GATED_CHARACTERS
            .iter()
            .find(|(id, _, first)| *id == up && build < *first)
        {
            let note = trf(
                "{name} needs game version {first} or newer; this game reports {build}, so there is no advice for this run.",
                &[
                    ("name", name),
                    ("first", &first.as_str()),
                    ("build", &build.as_str()),
                ],
            );
            return Some((up, note));
        }
    }
    None
}

/// A `game_over` the player lost: only a death gets the coach's autopsy.
#[must_use]
pub fn is_death(msg: &Value) -> bool {
    get(msg, "type").as_str() == Some("game_over")
        && (get(msg, "result").as_str() == Some("defeat") || get(msg, "victory") == &json!(false))
}

/// The search's advice, answered in Instinct's place: say so, and why.
#[must_use]
pub fn mark_instinct_unavailable(mut advice: Value, reason: &str) -> Value {
    let note = get(&advice, "note").clone();
    advice["brain"] = json!("search");
    advice["instinct_unavailable"] = json!(reason);
    let extra = if truthy(&note) {
        format!(" · {}", py::str_of(&note))
    } else {
        String::new()
    };
    advice["note"] = json!(trf(
        "Search (Instinct: {reason}){extra}",
        &[("reason", &reason), ("extra", &extra)],
    ));
    advice
}

/// The fallback reason when Instinct failed only because the engine lacks a
/// card / relic / potion on the screen (`engine_missing_content`).
#[must_use]
pub fn engine_missing_content(exc_type: &str, message: &str) -> Option<String> {
    // Every board error the Python advisor raised as a `ValueError`.
    if !matches!(
        exc_type,
        "ValueError" | "UnsupportedStateError" | "UnsupportedGameVersionError"
    ) {
        return None;
    }
    // `unknown (card|relic|potion) id "([A-Za-z0-9_]+)"`: the first match.
    let mut best: Option<(usize, String)> = None;
    for kind in ["card", "relic", "potion"] {
        let pat = format!("unknown {kind} id \"");
        let mut from = 0;
        while let Some(i) = message[from..].find(&pat).map(|i| i + from) {
            let rest = &message[i + pat.len()..];
            let id: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !id.is_empty() && rest[id.len()..].starts_with('"') {
                if best.as_ref().is_none_or(|(at, _)| i < *at) {
                    best = Some((
                        i,
                        if is_zh() {
                            let kind = match kind {
                                "card" => "卡牌",
                                "relic" => "遗物",
                                _ => "药水",
                            };
                            format!("引擎还没有{kind} {id}")
                        } else {
                            format!("the engine has no {kind} {id} yet")
                        },
                    ));
                }
                break;
            }
            from = i + 1;
        }
    }
    best.map(|(_, m)| m)
}

/// `fight_search_cfg`: `cfg` with the boss win value the run agent uses for
/// THIS fight (on the run's last boss nothing carries forward).
#[must_use]
pub fn fight_search_cfg(cfg: &RecommendConfig, state: &Value) -> RecommendConfig {
    if get(state, "room_type").as_str() != Some("Boss") {
        return cfg.clone();
    }
    let run_state = get(state, "run_state");
    let ascension = py::int_of(get(run_state, "ascension"));
    let is_last_act = match get(run_state, "is_last_act") {
        Value::Null => py::int_of(get(state, "act")) >= STANDARD_ACT_COUNT,
        v => truthy(v),
    };
    let awaiting_second = match get(run_state, "awaiting_second_boss") {
        Value::Null => {
            if ascension >= 10 && is_last_act {
                return cfg.clone();
            }
            false
        }
        v => truthy(v),
    };
    if !is_last_act || awaiting_second {
        return cfg.clone();
    }
    let asc = u8::try_from(ascension.clamp(0, 255)).unwrap_or(0);
    let Some((hp, potion)) = sts2_mcts::boss_win_carry(true, true, false, asc) else {
        return cfg.clone();
    };
    let mut out = cfg.clone();
    out.win_hp_carry = Some(f64::from(hp));
    out.win_potion_carry = Some(f64::from(potion));
    out
}

/// Install what the engine needs before any state is built: the card, relic
/// and potion dispatch registries (without them a card play or a relic
/// silently does nothing — Burning Blood never heals in a rollout) and the
/// co-op teammate policy. What `sts2_rs`'s module init does; idempotent.
pub fn install_engine() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::coop::register_teammate_policy(sts2_mcts::expert::teammate_policy);
}

/// Seconds since the epoch.
fn now_epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `datetime.now(UTC).strftime("%Y%m%d-%H%M%S")`.
#[must_use]
pub fn utc_stamp() -> String {
    utc_stamp_at(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    )
}

/// [`utc_stamp`] of `secs` since the epoch.
#[must_use]
pub fn utc_stamp_at(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// `datetime.now().isoformat()` for the run log's records and file stamp,
/// in UTC: std has no local time zone.
#[must_use]
pub fn local_iso_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let stamp = utc_stamp_at(now.as_secs());
    // `YYYYmmdd-HHMMSS` -> `YYYY-mm-ddTHH:MM:SS.ffffff`.
    format!(
        "{}-{}-{}T{}:{}:{}.{:06}",
        &stamp[0..4],
        &stamp[4..6],
        &stamp[6..8],
        &stamp[9..11],
        &stamp[11..13],
        &stamp[13..15],
        now.subsec_micros()
    )
}

/// Append-only JSONL event log (advice + human actions), one file/session.
#[derive(Debug)]
pub struct AdvisorLog {
    pub path: PathBuf,
    step: u64,
}

impl AdvisorLog {
    /// # Errors
    ///
    /// The directory cannot be created.
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            path: dir.join(format!("advisor-{}.jsonl", utc_stamp())),
            step: 0,
        })
    }

    pub fn write(&mut self, kind: &str, data: Value, log: &Logger) {
        let record = json!({"step": self.step, "ts": now_epoch(), "type": kind, "data": data});
        let res = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| writeln!(f, "{record}"));
        match res {
            Ok(()) => self.step += 1,
            Err(e) => log(
                LogLevel::Warning,
                "realgame_advisor",
                &format!("advisor log write failed: {e}"),
            ),
        }
    }
}

/// Every raw message from the mod, one JSON line each (`--record-requests`).
#[derive(Debug)]
pub struct RequestRecorder {
    pub path: PathBuf,
}

impl RequestRecorder {
    /// # Errors
    ///
    /// The directory cannot be created.
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            path: dir.join(format!("requests-{}.jsonl", utc_stamp())),
        })
    }

    fn write(&self, msg: &Value, log: &Logger) {
        let res = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| writeln!(f, "{msg}"));
        if let Err(e) = res {
            log(
                LogLevel::Warning,
                "realgame_advisor",
                &format!("request recorder: {e}"),
            );
        }
    }
}

/// The most recent combat decision's native search state, for companion
/// searches (the control plane's `/search_tree`).
#[derive(Debug, Clone)]
pub struct LastCombat {
    pub state_id: Value,
    pub fight: Mirrored,
    pub cfg: RecommendConfig,
}

/// The advisor's knobs (`realgame_advisor`'s command line).
#[derive(Debug, Clone)]
pub struct Settings {
    pub sims: u32,
    pub boss_sims: u32,
    pub threads: u32,
    /// Sims of the companion tree logged with each combat advice; 0 = off.
    pub log_search_tree: u32,
    pub coach_catchup_sims: u32,
    pub coach_budget_s: f64,
    /// Wall-clock cap on the turn plan's re-searches (`f64::INFINITY`: none,
    /// as a replay wants, GH #1784).
    pub plan_budget_s: f64,
    pub no_mirror: bool,
    /// Run the coach's jobs inline (a replay) instead of on its thread.
    pub inline_coach: bool,
    /// Where the mirror's divergence records go (`None`: nowhere).
    pub divergence_log: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sims: 4000,
            boss_sims: 16000,
            threads: 4,
            log_search_tree: 0,
            coach_catchup_sims: 2000,
            coach_budget_s: 6.0,
            plan_budget_s: crate::turn_plan::PLAN_BUDGET_S,
            no_mirror: false,
            inline_coach: false,
            divergence_log: None,
        }
    }
}

/// GH #1917: the engine keeps `run_state.current_hp` at its pre-fight value
/// for the whole fight (a fight's damage folds into run HP when it ends; the
/// fight's own HP is the combat player's), so the net trained on RUNCTX
/// `R_HP` = the HP the fight STARTED on. The mod sends the LIVE run HP on
/// every in-fight decision; this rewrites `state.run_state.{hp,max_hp}` to the
/// fight-entry value. `room_hp` is the last out-of-fight `(hp, max_hp)`,
/// `fight_hp` the frozen value (cleared outside a fight). Only `run_state` is
/// touched; the `combat` block keeps the live fight HP. Matches the bench's
/// `_fight_start_hp` (`instinct_policy.py`): the fight-entry run HP is the
/// last out-of-fight value plus what heals on entering the room
/// (`room_entry_heal`). A reconnect with no prior screen falls back to the
/// message's own value, already past any such heal.
fn freeze_fight_run_hp(
    room_hp: &mut Option<(i64, i64)>,
    fight_hp: &mut Option<(i64, i64)>,
    state: &mut Value,
) {
    let in_fight = get(state, "combat").is_object();
    let live = {
        let rs = get(state, "run_state");
        get(rs, "hp").as_i64().zip(get(rs, "max_hp").as_i64())
    };
    if !in_fight {
        *fight_hp = None;
        if live.is_some() {
            *room_hp = live;
        }
        return;
    }
    if fight_hp.is_none() {
        *fight_hp = match *room_hp {
            Some((hp, max_hp)) => Some(((hp + room_entry_heal(state)).min(max_hp), max_hp)),
            None => live,
        };
    }
    if let Some((hp, max_hp)) = *fight_hp {
        if let Some(rs) = state.get_mut("run_state").and_then(Value::as_object_mut) {
            rs.insert("hp".into(), json!(hp));
            rs.insert("max_hp".into(), json!(max_hp));
        }
    }
}

/// HP that heals on entering the fight's room, after the last out-of-fight
/// screen and before the first decision (the bench's `_fight_start_hp`
/// `entry`). Pantograph heals 25 entering a boss fight
/// (`Pantograph.BeforeCombatStart`); a boss is never event-started, so no page
/// before it already shows the heal and the pre-fight `room_hp` is always
/// below it. Planisphere's +5 on a `?` room is deliberately left out: that
/// room's own event page may already carry the heal, so adding it
/// unconditionally would double-count — the conservative choice is to omit it
/// (minor residual on #1917; a `?`-room fight with Planisphere reads 5 low).
fn room_entry_heal(state: &Value) -> i64 {
    let is_boss = get(state, "room_type")
        .as_str()
        .is_some_and(|r| r.eq_ignore_ascii_case("boss"));
    if !is_boss {
        return 0;
    }
    let holds_unmelted_pantograph = py::list(get(state, "run_state"), "relics").iter().any(|r| {
        get(r, "id").as_str().is_some_and(|id| {
            id.strip_prefix("RELIC.")
                .unwrap_or(id)
                .eq_ignore_ascii_case("PANTOGRAPH")
        }) && !truthy(get(r, "is_melted"))
    });
    if holds_unmelted_pantograph {
        25
    } else {
        0
    }
}

/// Everything one advisor process carries across connections.
pub struct Advisor {
    pub settings: Settings,
    pub cfg: RecommendConfig,
    pub names: Arc<Names>,
    pub policies: Arc<PolicyCache>,
    pub instinct: Arc<InstinctAdvisor>,
    pub mirror: AdvisorMirror,
    /// `(hp, max_hp)` of the last out-of-fight screen (GH #1917).
    room_hp: Option<(i64, i64)>,
    /// `(hp, max_hp)` the run held when the current fight began (GH #1917).
    fight_hp: Option<(i64, i64)>,
    pub session: Arc<Mutex<Map<String, Value>>>,
    pub tree: Arc<LiveRunTree>,
    pub control: Arc<ControlState>,
    pub wlog: Arc<WorkbenchRunLogger>,
    pub alog: Option<AdvisorLog>,
    pub recorder: Option<RequestRecorder>,
    pub last_combat: Arc<Mutex<Option<LastCombat>>>,
    pub log: Logger,
    coach: Option<CoachGrader<Mirrored, RecommendConfig>>,
    coach_executor: Option<Arc<dyn Executor>>,
    link: Option<Arc<dyn Link>>,
    warmed: Arc<Mutex<BTreeSet<String>>>,
}

impl std::fmt::Debug for Advisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Advisor")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The parts an [`Advisor`] is built from.
#[allow(missing_debug_implementations)]
pub struct Parts {
    pub settings: Settings,
    pub names: Arc<Names>,
    pub policies: Arc<PolicyCache>,
    pub instinct: Arc<InstinctAdvisor>,
    pub wlog: Arc<WorkbenchRunLogger>,
    pub alog: Option<AdvisorLog>,
    pub recorder: Option<RequestRecorder>,
    pub log: Logger,
}

impl Advisor {
    #[must_use]
    pub fn new(parts: Parts) -> Self {
        let Parts {
            settings,
            names,
            policies,
            instinct,
            wlog,
            alog,
            recorder,
            log,
        } = parts;
        install_engine();
        let cfg = RecommendConfig {
            sims: settings.sims,
            threads: settings.threads,
            hidden_info: HONEST_HIDDEN_INFO.into(),
            c_puct: Some(HONEST_C_PUCT),
            draw_widening_c: Some(HONEST_DRAW_WIDENING_C),
            ..RecommendConfig::default()
        };
        let tree = Arc::new(LiveRunTree::new());
        let damage_trace = std::env::var("STS2_MIRROR_DAMAGE_TRACE").is_ok_and(|v| v == "1");
        Self {
            mirror: AdvisorMirror::new(!settings.no_mirror, damage_trace),
            room_hp: None,
            fight_hp: None,
            settings,
            cfg,
            names,
            policies,
            instinct,
            session: Arc::new(Mutex::new(Map::new())),
            control: Arc::new(ControlState::new(tree.clone())),
            tree,
            wlog,
            alog,
            recorder,
            last_combat: Arc::new(Mutex::new(None)),
            log,
            coach: None,
            coach_executor: None,
            link: None,
            warmed: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }

    fn info(&self, msg: &str) {
        (self.log)(LogLevel::Info, "realgame_advisor", msg);
    }

    fn warning(&self, msg: &str) {
        (self.log)(LogLevel::Warning, "realgame_advisor", msg);
    }

    fn alog_write(&mut self, kind: &str, data: Value) {
        if let Some(a) = self.alog.as_mut() {
            a.write(kind, data, &self.log);
        }
    }

    /// Write what the mirror queued: its log lines and its divergence records.
    fn flush_mirror(&mut self) {
        for (level, message) in self.mirror.take_log() {
            (self.log)(level.into(), "realgame_advisor_mirror", &message);
        }
        let divergences = self.mirror.take_divergences();
        let Some(path) = &self.settings.divergence_log else {
            return;
        };
        for d in divergences {
            let mut record = Map::new();
            record.insert("ts".into(), json!(now_epoch()));
            if let Value::Object(r) = d.report {
                record.extend(r);
            }
            let res = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                })
                .and_then(|mut f| writeln!(f, "{}", Value::Object(record)));
            if let Err(e) = res {
                self.warning(&format!("mirror divergence log: {e}"));
            }
        }
    }

    fn emit(&self, log: Vec<(&'static str, String)>, name: &str) {
        for (level, line) in log {
            let level = match level {
                "debug" => LogLevel::Debug,
                "warning" => LogLevel::Warning,
                "error" => LogLevel::Error,
                _ => LogLevel::Info,
            };
            (self.log)(level, name, &line);
        }
    }

    /// `capabilities` on every reply: the coach level, plus what Instinct plays.
    fn capabilities(&self) -> Value {
        let mut caps = advisor_capabilities_base();
        let mut log = Vec::new();
        caps.insert("instinct".into(), self.instinct.capabilities(&mut log));
        self.emit(log, "realgame_instinct");
        Value::Object(caps)
    }

    // -- the connection --------------------------------------------------

    /// A connection to the mod is up: its coach starts.
    pub fn connect(&mut self, link: Arc<dyn Link>) {
        self.alog_write(
            "meta",
            json!({
                "event": "connected",
                "sims": self.settings.sims,
                "hidden_info": HONEST_HIDDEN_INFO,
                "c_puct": HONEST_C_PUCT,
                "draw_widening_c": HONEST_DRAW_WIDENING_C,
            }),
        );
        self.control.set_link(Some(link.clone()));
        let executor: Arc<dyn Executor> = if self.settings.inline_coach {
            Arc::new(InlineExecutor::default())
        } else {
            Arc::new(GraderThread::new())
        };
        self.coach = Some(self.build_coach(link.clone(), executor.clone()));
        self.coach_executor = Some(executor);
        self.link = Some(link);
    }

    /// The connection dropped (`mid_run`: it had been up, so the mod's
    /// unread requests went with it).
    pub fn disconnect(&mut self, mid_run: bool) {
        if mid_run {
            self.wlog.note_reconnect();
        }
        if let Some(ex) = self.coach_executor.take() {
            ex.stop();
        }
        self.coach = None;
        self.control.set_link(None);
        self.link = None;
    }

    /// Run this connection without a coach (a replay with `coach: false`).
    pub fn drop_coach(&mut self) {
        if let Some(ex) = self.coach_executor.take() {
            ex.stop();
        }
        self.coach = None;
    }

    /// The connection's live coach, wired to this link and run log.
    fn build_coach(
        &self,
        link: Arc<dyn Link>,
        executor: Arc<dyn Executor>,
    ) -> CoachGrader<Mirrored, RecommendConfig> {
        let names = self.names.clone();
        let send_log = self.log.clone();
        let send = Box::new(move |record: &Value| {
            // The same no-raw-ids guard as the advice.
            let mut record = record.clone();
            names.sanitize_coach(&mut record);
            let reply =
                json!({"action": "coach", "request_id": null, "state_id": null, "coach": record});
            if let Err(e) = link.send(&reply, true) {
                send_log(
                    LogLevel::Warning,
                    "realgame_coach",
                    &format!("coach send failed: {e}"),
                );
            }
        });
        let (wlog, log) = (self.wlog.clone(), self.log.clone());
        let log_sink = Box::new(move |record: &Value| {
            let id = get(record, "id");
            let id = if truthy(id) { id } else { get(record, "ids") };
            let verdict = record
                .as_object()
                .and_then(|m| m.get("verdict"))
                .map_or_else(String::new, py::str_of);
            log(
                LogLevel::Info,
                "realgame_advisor",
                &format!(
                    "coach {} {}: {verdict}",
                    py::str_of(get(record, "kind")),
                    py::str_of(id)
                ),
            );
            wlog.log_coach(record);
        });
        let warn_log = self.log.clone();
        let (tree, wlog) = (self.tree.clone(), self.wlog.clone());
        let locate = Box::new(move |node: &Value| {
            let idx = tree.log_idx(node).as_i64();
            (wlog.log_name(), idx)
        });
        let (session, policies) = (self.session.clone(), self.policies.clone());
        let map_path = Box::new(move |msg: &Value, chosen: usize| -> Option<Vec<Value>> {
            let mut state = msg.as_object()?.clone();
            state.insert("type".into(), json!("map_select"));
            let mut state = Value::Object(state);
            if let Some(map) = crate::session::view_with_session_context(&state, &lock(&session)) {
                state["map"] = map;
            }
            let character =
                crate::headless::character_display_name(get(get(&state, "run_state"), "character"))
                    .unwrap_or("Ironclad");
            let net = policies.get(character).0.ok()?;
            let mut score = |obs: &sts2_runai::ppo::obs::PpoObs,
                             actions: &[sts2_runai::ppo::obs::ActionDesc]|
             -> Option<Vec<f64>> {
                let feat = sts2_runai::ppo::encoder::featurize_with(obs, actions, net.features);
                if feat.obs_cat.len() + feat.n_actions > net.net.t_max()
                    || feat.n_actions > net.net.a_max()
                {
                    return None;
                }
                let (probs, _value) = net.net.infer(&feat).ok()?;
                Some(probs.into_iter().map(f64::from).collect())
            };
            Some(crate::screen_advice::project_map_path(
                &state, chosen, &mut score,
            ))
        });
        let t0 = Instant::now();
        CoachGrader::new(GraderConfig {
            send,
            log: log_sink,
            warn: Box::new(move |m: &str| warn_log(LogLevel::Warning, "realgame_coach", m)),
            engine: Box::new(CoachEngine),
            executor,
            names: self.names.clone(),
            catchup_sims: self.settings.coach_catchup_sims,
            budget_s: self.settings.coach_budget_s,
            map_path: Some(map_path),
            clock: Box::new(move || t0.elapsed().as_secs_f64()),
            locate: Some(locate),
            cache: Arc::default(),
        })
    }

    // -- the loop --------------------------------------------------------

    /// Load the run-policy net for `character` on a background thread, so a
    /// screen never pays the load inline.
    fn warm_policy(&self, character: &Value) {
        if !truthy(character) {
            return;
        }
        let name = py::str_of(character);
        let key = name.trim().to_lowercase();
        if key.is_empty() || !lock(&self.warmed).insert(key.clone()) {
            return;
        }
        let (policies, log) = (self.policies.clone(), self.log.clone());
        let display =
            crate::headless::character_display_name(character).map_or(name.clone(), str::to_string);
        let _ = std::thread::Builder::new()
            .name(format!("warm-{key}"))
            .spawn(move || {
                let t0 = Instant::now();
                match policies.get(&display).0 {
                    Ok(_) => log(
                        LogLevel::Info,
                        "realgame_advisor",
                        &format!(
                            "run-policy net ready for {name} ({:.1}s)",
                            t0.elapsed().as_secs_f64()
                        ),
                    ),
                    Err(e) => log(
                        LogLevel::Warning,
                        "realgame_advisor",
                        &format!("run-policy net unavailable for {name}: {e}"),
                    ),
                }
            });
    }

    /// Handle one message from the mod.
    ///
    /// # Errors
    ///
    /// Sending a reply failed: the connection is gone.
    pub fn on_message(&mut self, msg: &Value) -> std::io::Result<()> {
        let mtype = get(msg, "type").as_str().unwrap_or_default().to_string();
        if let Some(r) = &self.recorder {
            r.write(msg, &self.log);
        }
        self.warm_policy(get(get(msg, "run_state"), "character"));
        self.control.note_message(msg);

        if mtype == "advice_request"
            && self
                .link
                .as_ref()
                .is_some_and(|l| l.newer_advice_request_pending(get(msg, "advice_for")))
        {
            // A board that no longer exists: the mod already asked again.
            self.info(&format!(
                "advice: skipped request {} ({}), a newer one is queued",
                py::str_of(get(msg, "request_id")),
                py::str_of(get(msg, "advice_for"))
            ));
            return Ok(());
        }
        if mtype == "advice_request" {
            // The player's language (PROTOCOL.md v5.4): this request's
            // answer and the coach records until the next request.
            crate::i18n::set_language(get(msg, "language").as_str());
        }
        match mtype.as_str() {
            "advice_request" => self.on_advice_request(msg)?,
            "human_action" => self.on_human_action(msg),
            t if BRIDGE_MESSAGE_TYPES.contains(&t) => {
                (self.log)(LogLevel::Debug, "realgame_advisor", &format!("{t}: {msg}"));
                let rewind = matches!(t, "snapshot_restored" | "combat_undone");
                if rewind {
                    // Everything carried for the old position is stale.
                    self.mirror.on_rewind(get(msg, "type"));
                    self.flush_mirror();
                    if let Some(line) = self.instinct.reset(t) {
                        self.info(&line);
                    }
                }
                self.control.on_bridge_message(msg);
                if rewind {
                    if let Some(coach) = &self.coach {
                        let path: Vec<Value> = self
                            .tree
                            .active_path()
                            .into_iter()
                            .map(|n| json!(n))
                            .collect();
                        coach.on_rewind(msg, &path);
                    }
                }
            }
            "pong" => (self.log)(LogLevel::Debug, "realgame_advisor", &format!("pong: {msg}")),
            "combat_ended" => {
                self.info(&format!(
                    "combat_ended: {}",
                    py::repr(&pick(msg, &["victory", "hp", "round"]))
                ));
                self.alog_write("meta", with_event(&mtype, msg));
                if let Some(coach) = &self.coach {
                    coach.on_combat_ended(msg);
                }
            }
            "game_over" | "run_complete" => self.on_run_over(&mtype, msg),
            other => self.warning(&format!(
                "ignoring non-advisor message type={} (advisor never drives)",
                py::repr(&json!(other))
            )),
        }
        // AFTER the handler: the file this message opened is the one served.
        if let Some(path) = self.wlog.path() {
            self.control.set_log_path(Some(&path));
        }
        Ok(())
    }

    fn on_human_action(&mut self, msg: &Value) {
        // The position the action was taken FROM, for the coach's catch-up
        // search: cloned before the mirror follows the move.
        let pre_state = if self.coach.is_some() {
            self.mirror.live().and_then(|m| m.mirrored()).cloned()
        } else {
            None
        };
        // Follow the world BEFORE logging.
        self.mirror.on_world_action(msg);
        self.flush_mirror();
        self.instinct.on_human_action(msg);
        let detail = get(msg, "detail");
        let what = if truthy(detail) {
            detail
        } else {
            get(msg, "action")
        };
        self.info(&format!("human: {}", py::str_of(what)));
        self.alog_write("action", msg.clone());
        self.wlog.log_human_action(msg);
        let node_id = self
            .tree
            .on_human_action(msg)
            .map_or(Value::Null, |n| json!(n));
        if let Some(coach) = &self.coach {
            coach.on_human_action(msg, pre_state, &node_id);
        }
    }

    fn on_run_over(&mut self, mtype: &str, msg: &Value) {
        self.info(&format!(
            "{mtype}: {}",
            py::repr(&pick(msg, &["victory", "floor", "act"]))
        ));
        self.alog_write("meta", with_event(mtype, msg));
        self.wlog.log_run_end(msg);
        self.tree.on_run_end(msg);
        *lock(&self.last_combat) = None;
        self.mirror.drop_fight("run over");
        self.flush_mirror();
        if let Some(line) = self.instinct.reset("run over") {
            self.info(&line);
        }
        let note = self.mirror.note();
        self.info(&format!("advisor {note}"));
        lock(&self.session).clear();
        if let Some(coach) = &self.coach {
            if is_death(msg) {
                // The autopsy is appended to the run log this just closed.
                coach.on_game_over(msg);
            } else {
                coach.on_run_end();
            }
        }
    }

    /// Track the game build across requests; a change invalidates carried
    /// state built for the previous build's engine.
    fn note_game_version(&mut self, version: &str) {
        let previous = lock(&self.session).get("game_version").cloned();
        if previous.as_ref().and_then(Value::as_str) == Some(version) {
            return;
        }
        match &previous {
            None => self.info(&format!("game build: {version}")),
            Some(p) => {
                self.warning(&format!(
                    "game build changed {} -> {version}; dropping carried state",
                    py::str_of(p)
                ));
                *lock(&self.last_combat) = None;
                self.mirror.drop_fight(&format!(
                    "game build changed {} -> {version}",
                    py::str_of(p)
                ));
                self.flush_mirror();
                if let Some(line) = self.instinct.reset("game build changed") {
                    self.info(&line);
                }
            }
        }
        lock(&self.session).insert("game_version".into(), json!(version));
    }

    #[allow(clippy::too_many_lines)]
    fn on_advice_request(&mut self, msg: &Value) -> std::io::Result<()> {
        let request_id = get(msg, "request_id").clone();
        let state_id = get(msg, "state_id").clone();
        let advice_for = {
            let a = get(msg, "advice_for");
            if truthy(a) {
                py::str_of(a)
            } else {
                "combat_action".to_string()
            }
        };
        if let Some(coach) = &self.coach {
            coach.on_advice_request(msg);
        }
        // State first, advice second: the mcts record attaches to the most
        // recent state line, and the state survives a failed search.
        let logged_step = self.wlog.log_decision(msg, &advice_for);
        if let Some(step) = logged_step {
            self.tree
                .on_decision(msg, &advice_for, i64::try_from(step).ok());
        }
        let mut state = msg.clone();
        let mut search_tree_log: Option<Value> = None;
        let mut combat: Option<(Vec<Value>, Mirrored, RecommendConfig)> = None;
        let t0 = Instant::now();
        let advice = match self.advise(
            msg,
            &advice_for,
            &mut state,
            &mut combat,
            &mut search_tree_log,
            logged_step,
        ) {
            Ok(a) => a,
            Err(Refusal::GameVersion(note, raw)) => {
                (self.log)(
                    LogLevel::Error,
                    "realgame_advisor",
                    &format!("advice_request refused: {note}"),
                );
                json!({"note": note, "unsupported_game_version": raw})
            }
            Err(Refusal::Character(raw, note)) => {
                self.warning(&format!("advice_request refused: {note}"));
                json!({"note": note, "unsupported_character": raw})
            }
            Err(Refusal::Error(e)) => {
                (self.log)(
                    LogLevel::Error,
                    "realgame_advisor",
                    &format!("advice_request failed: {e}"),
                );
                json!({
                    "note": trf("advisor error: {e}", &[("e", &e)]),
                    "hand_back": tr("the advisor hit an error"),
                })
            }
        };
        // The wire boundary: no raw game id reaches the HUD.
        let mut advice = advice;
        self.names
            .sanitize_advice(&mut advice, Some(&advice_for), msg);
        let elapsed = t0.elapsed().as_secs_f64();
        let reply = json!({
            "action": "advice",
            "request_id": request_id,
            "state_id": state_id,
            "advice": advice,
            "capabilities": self.capabilities(),
        });
        if let Some(link) = &self.link {
            link.send(&reply, true)?;
        }
        if let Some(coach) = &self.coach {
            // AFTER the reply: the coach's bookkeeping never delays advice.
            let entry = if advice_for == "combat_action" {
                combat.map(|(rows, fight, cfg)| Entry {
                    request_id: request_id.clone(),
                    state_id: state_id.clone(),
                    kind: EntryKind::Combat,
                    rows: Some(rows),
                    options: None,
                    cs: Some(fight),
                    cfg: Some(cfg),
                    advice: advice.clone(),
                    pv_labels: py::list(&advice, "pv").iter().map(py::str_of).collect(),
                    msg: state.clone(),
                })
            } else if SCREEN_ADVICE_FOR.contains(&advice_for.as_str())
                && truthy(get(&advice, "options"))
            {
                Some(Entry {
                    request_id: request_id.clone(),
                    state_id: state_id.clone(),
                    kind: EntryKind::Screen,
                    rows: None,
                    options: Some(py::list(&advice, "options").to_vec()),
                    cs: None,
                    cfg: None,
                    advice: advice.clone(),
                    pv_labels: Vec::new(),
                    msg: state.clone(),
                })
            } else {
                None
            };
            if let Some(entry) = entry {
                let stored = coach.cache_put(entry);
                coach.on_advice(stored.as_deref());
            }
        }
        self.alog_write(
            "mcts",
            json!({
                "request_id": request_id,
                "state_id": state_id,
                "elapsed_s": py::round(elapsed, 2),
                "advice": advice,
            }),
        );
        if logged_step.is_some_and(|s| s > 0) {
            self.wlog.log_advice(
                msg,
                &advice,
                elapsed,
                search_tree_log.as_ref().unwrap_or(&Value::Null),
            );
        }
        let best_label = get(get(&advice, "best"), "label");
        let note = get(&advice, "note");
        self.info(&format!(
            "advice #{}: best={} p_win={} ({elapsed:.1}s){}",
            py::str_of(&request_id),
            py::str_of(best_label),
            py::str_of(get(&advice, "p_win")),
            if truthy(note) {
                format!(" [{}]", py::str_of(note))
            } else {
                String::new()
            }
        ));
        Ok(())
    }

    /// The advice for one request, or why there is none.
    fn advise(
        &mut self,
        msg: &Value,
        advice_for: &str,
        state: &mut Value,
        combat: &mut Option<(Vec<Value>, Mirrored, RecommendConfig)>,
        search_tree_log: &mut Option<Value>,
        logged_step: Option<usize>,
    ) -> Result<Value, Refusal> {
        // The build the mod reports; an unknown build is refused first.
        let version = crate::headless::game_version(msg)
            .map_err(|e| Refusal::GameVersion(e.to_string(), get(msg, "game_version").clone()))?;
        self.note_game_version(version.unwrap_or("v0.107.1"));
        if let Some((raw, note)) = unsupported_character(msg) {
            return Err(Refusal::Character(raw, note));
        }
        if let Some(m) = state.as_object_mut() {
            m.insert("type".into(), json!(advice_for));
        }
        freeze_fight_run_hp(&mut self.room_hp, &mut self.fight_hp, state);
        let is_boss = get(state, "room_type").as_str() == Some("Boss");
        let sims = if is_boss {
            self.settings.boss_sims
        } else {
            self.settings.sims
        };
        let fight_cfg = fight_search_cfg(&self.cfg, state);
        // The player's brain: Instinct answers when it can read the
        // position; otherwise the search answers and says why.
        let mut instinct_advice = None;
        let mut instinct_reason = None;
        if get(msg, "brain").as_str() == Some("instinct") {
            let mut log = Vec::new();
            let out = {
                let mut session = lock(&self.session);
                self.instinct.advise(
                    state,
                    advice_for,
                    Context {
                        names: &self.names,
                        cfg: Some(&fight_cfg),
                        mirror: Some(&mut self.mirror),
                        session: Some(&mut session),
                        sims: Some(sims),
                    },
                    &mut log,
                )
            };
            self.flush_mirror();
            self.emit(log, "realgame_instinct");
            match out {
                Ok(a) => instinct_advice = Some(a),
                Err(InstinctError::Unavailable(why)) => {
                    self.info(&format!("instinct unavailable ({advice_for}): {why}"));
                    instinct_reason = Some(why);
                }
                Err(InstinctError::Failed(ty, m)) => {
                    if let Some(missing) = engine_missing_content(&ty, &m) {
                        self.info(&format!("instinct unavailable ({advice_for}): {missing}"));
                        instinct_reason = Some(missing);
                    } else {
                        (self.log)(
                            LogLevel::Error,
                            "realgame_advisor",
                            &format!("instinct failed: {m}"),
                        );
                        instinct_reason = Some(tr("Instinct hit an error").into());
                    }
                }
            }
        }
        let mut advice = if let Some(a) = instinct_advice {
            if advice_for == "combat_action" {
                crate::coop::apply_ally_wait(state, a)
            } else {
                a
            }
        } else if SCREEN_ADVICE_FOR.contains(&advice_for) {
            self.screen_advice(state, advice_for, &fight_cfg, sims)?
        } else if advice_for == "combat_action" {
            let mut log = Vec::new();
            let out = crate::advise_combat::compute_combat_advice(
                state,
                &fight_cfg,
                sims,
                self.settings.plan_budget_s,
                Some(&mut self.mirror),
                &self.names,
                &mut log,
            );
            self.flush_mirror();
            for (level, line) in log {
                (self.log)(level.into(), "realgame_advisor", &line);
            }
            let out = out.map_err(|e| match e {
                crate::advise_combat::AdviseError::Headless(
                    crate::headless::HeadlessError::GameVersion(m),
                ) => Refusal::GameVersion(m, get(msg, "game_version").clone()),
                e => Refusal::Error(e.to_string()),
            })?;
            let advice = crate::coop::apply_ally_wait(state, out.advice);
            // GH #1918: `best.target_index` is already in board space.
            self.mirror
                .set_advice_target(get(get(&advice, "best"), "target_index").clone());
            *lock(&self.last_combat) = Some(LastCombat {
                state_id: get(msg, "state_id").clone(),
                fight: out.fight.clone(),
                cfg: out.cfg.clone(),
            });
            if self.settings.log_search_tree > 0 && logged_step.is_some_and(|s| s > 0) {
                match companion_search_tree(&out.fight, &out.cfg, self.settings.log_search_tree) {
                    Ok(t) => *search_tree_log = Some(t),
                    Err(e) => self.warning(&format!("search-tree logging failed: {e}")),
                }
            }
            *combat = Some((out.rows, out.fight, out.cfg));
            advice
        } else {
            return Err(Refusal::Error(format!(
                "unsupported advice_for={}",
                py::repr(&json!(advice_for))
            )));
        };
        if let Some(reason) = instinct_reason {
            advice = mark_instinct_unavailable(advice, &reason);
        }
        Ok(advice)
    }

    fn screen_advice(
        &mut self,
        state: &Value,
        advice_for: &str,
        cfg: &RecommendConfig,
        sims: u32,
    ) -> Result<Value, Refusal> {
        use crate::screen_advice::{compute_screen_advice, Outcome, ScreenEnv};
        let policies = self.policies.clone();
        let mut misses = Vec::new();
        let mut policy_for = |character: &str| policies.lookup(character, &mut misses);
        let mut wire_for = crate::board::wire_for;
        let (out, log) = {
            let mut session = lock(&self.session);
            let mut env = ScreenEnv {
                session: Some(&mut session),
                mirror: Some(&mut self.mirror),
                cfg: Some(cfg),
                sims: Some(sims),
                names: &self.names,
                wire_for: &mut wire_for,
                policy_for: &mut policy_for,
                log: Vec::new(),
            };
            let out = compute_screen_advice(state, advice_for, &mut env);
            (out, std::mem::take(&mut env.log))
        };
        self.flush_mirror();
        for line in misses {
            (self.log)(LogLevel::Error, "realgame_ppo", &line);
        }
        for (level, name, line) in log {
            (self.log)(level.into(), name, &line);
        }
        match out {
            Outcome::Advice(a) => Ok(a),
            Outcome::OtherBackend(_) => Err(Refusal::Error(
                "no Rust run-policy net for this character".into(),
            )),
        }
    }
}

/// Why a request got no ranking.
#[derive(Debug)]
enum Refusal {
    /// A build the engine has no rules for: the note and the raw build.
    GameVersion(String, Value),
    /// A character the engine does not simulate: the raw id and the note.
    Character(String, String),
    /// Anything else (the HUD shows "advisor error").
    Error(String),
}

/// `{k: msg.get(k) for k in keys}`.
fn pick(msg: &Value, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .map(|k| ((*k).to_string(), get(msg, k).clone()))
            .collect(),
    )
}

/// `{"event": kind, **msg}`.
fn with_event(kind: &str, msg: &Value) -> Value {
    let mut out = Map::new();
    out.insert("event".into(), json!(kind));
    if let Some(m) = msg.as_object() {
        out.extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    Value::Object(out)
}

/// A single-tree search over `fight` in the workbench's `search_tree`
/// payload shape, pruned to keep log records bounded. A COMPANION search,
/// not the advice's own beliefdraw tree.
///
/// # Errors
///
/// The search refused its config.
pub fn companion_search_tree(
    fight: &Mirrored,
    cfg: &RecommendConfig,
    sims: u32,
) -> Result<Value, String> {
    #[allow(clippy::cast_possible_truncation)] // the engine's knobs are f32
    let c_puct = cfg.resolved_c_puct() as f32;
    let rollout_policy = cfg
        .rollout_policy
        .as_deref()
        .map(|p| sts2_wire::emit::mcts::parse_rollout_policy(p, "mcts_search_tree"))
        .transpose()?;
    // `mcts_search_tree`'s defaults: the honest mode, its widening constant.
    let params = sts2_wire::emit::mcts::RecommendParams {
        sims,
        depth: cfg.depth,
        c_puct,
        seed: cfg.seed,
        rollout_policy,
        hidden_info: sts2_mcts::HiddenInfoMode::BeliefDraw,
        draw_widening_c: Some(sts2_mcts::BELIEF_DRAW_DRAW_WIDENING_C),
        ..Default::default()
    };
    let tree = sts2_wire::emit::mcts::search_tree_value(&fight.state, &params);
    Ok(prune_tree(&tree, 2, 2000))
}

/// Drop leaf noise so a logged tree stays a record, not a payload bomb:
/// keep the root and any node with `visits >= min_visits`, cap to the
/// `max_nodes` most visited, and re-add the parents of kept nodes.
#[must_use]
pub fn prune_tree(tree: &Value, min_visits: i64, max_nodes: usize) -> Value {
    let nodes = py::list(tree, "nodes");
    if nodes.is_empty() {
        return tree.clone();
    }
    let root = get(tree, "root");
    let visits = |n: &Value| py::int_of(get(n, "visits"));
    let is_root = |n: &Value| py::eq(get(n, "id"), root);
    let mut kept: Vec<&Value> = nodes
        .iter()
        .filter(|n| is_root(n) || visits(n) >= min_visits)
        .collect();
    kept.sort_by_key(|n| (!is_root(n), -visits(n)));
    kept.truncate(max_nodes);
    let mut ids: Vec<&Value> = kept.iter().map(|n| get(n, "id")).collect();
    let mut i = 0;
    let n_kept = kept.len();
    while i < n_kept {
        let mut cur = kept[i];
        loop {
            let pid = get(cur, "parent");
            if pid.is_null() || ids.iter().any(|x| py::eq(x, pid)) {
                break;
            }
            let Some(parent) = nodes.iter().find(|n| py::eq(get(n, "id"), pid)) else {
                break;
            };
            kept.push(parent);
            ids.push(pid);
            cur = parent;
        }
        i += 1;
    }
    let mut out = tree.clone();
    out["nodes"] = Value::Array(kept.into_iter().cloned().collect());
    out["pruned_from"] = json!(nodes.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fight_msg(hp: i64) -> Value {
        json!({"run_state": {"hp": hp, "max_hp": 80}, "combat": {"player": {"hp": hp}}})
    }

    #[test]
    fn run_hp_is_frozen_at_fight_entry() {
        // (a) an out-of-fight screen records room_hp and clears fight_hp.
        let (mut room, mut fight) = (None, Some((1, 1)));
        let mut shop = json!({"run_state": {"hp": 70, "max_hp": 80}});
        freeze_fight_run_hp(&mut room, &mut fight, &mut shop);
        assert_eq!((room, fight), (Some((70, 80)), None));
        // (b) the first in-fight decision: fight_hp == room_hp, run_state
        // rewritten, combat untouched.
        let mut first = fight_msg(70);
        freeze_fight_run_hp(&mut room, &mut fight, &mut first);
        assert_eq!(fight, room);
        assert_eq!(first["run_state"], json!({"hp": 70, "max_hp": 80}));
        // (c) later decisions keep it as the live run HP falls.
        let mut later = fight_msg(61);
        freeze_fight_run_hp(&mut room, &mut fight, &mut later);
        assert_eq!(later["run_state"], json!({"hp": 70, "max_hp": 80}));
        assert_eq!(later["combat"], json!({"player": {"hp": 61}}));
        // The next fight starts from the new room_hp.
        let mut rest = json!({"run_state": {"hp": 61, "max_hp": 80}});
        freeze_fight_run_hp(&mut room, &mut fight, &mut rest);
        let mut next = fight_msg(55);
        freeze_fight_run_hp(&mut room, &mut fight, &mut next);
        assert_eq!(next["run_state"]["hp"], 61);
    }

    #[test]
    fn a_fight_with_no_prior_screen_falls_back_to_its_own_run_hp() {
        let (mut room, mut fight) = (None, None);
        let mut m = fight_msg(64);
        freeze_fight_run_hp(&mut room, &mut fight, &mut m);
        assert_eq!(fight, Some((64, 80)));
        let mut m2 = fight_msg(50);
        freeze_fight_run_hp(&mut room, &mut fight, &mut m2);
        assert_eq!(m2["run_state"]["hp"], 64);
    }

    fn boss_msg(hp: i64, relics: Value) -> Value {
        json!({"room_type": "Boss",
               "run_state": {"hp": hp, "max_hp": 80, "relics": relics},
               "combat": {"player": {"hp": hp}}})
    }

    #[test]
    fn a_boss_fight_with_pantograph_adds_its_entry_heal() {
        // Entering a boss room heals 25 (`Pantograph::after_room_entered`),
        // after the last out-of-fight screen's HP: fight_hp = room_hp + 25.
        let (mut room, mut fight) = (Some((50, 80)), None);
        let mut boss = boss_msg(50, json!([{"id": "PANTOGRAPH", "is_melted": false}]));
        freeze_fight_run_hp(&mut room, &mut fight, &mut boss);
        assert_eq!(fight, Some((75, 80)));
        assert_eq!(boss["run_state"]["hp"], 75);
        // The heal is capped at max HP.
        let (mut room2, mut fight2) = (Some((70, 80)), None);
        let mut boss2 = boss_msg(70, json!([{"id": "RELIC.PANTOGRAPH"}]));
        freeze_fight_run_hp(&mut room2, &mut fight2, &mut boss2);
        assert_eq!(fight2, Some((80, 80)));
    }

    #[test]
    fn no_entry_heal_off_a_boss_or_from_a_melted_pantograph() {
        // A non-boss room: no Pantograph heal.
        let (mut room, mut fight) = (Some((50, 80)), None);
        let mut monster = fight_msg(50);
        monster["run_state"]["relics"] = json!([{"id": "PANTOGRAPH"}]);
        freeze_fight_run_hp(&mut room, &mut fight, &mut monster);
        assert_eq!(fight, Some((50, 80)));
        // A melted Pantograph does nothing.
        let (mut room2, mut fight2) = (Some((50, 80)), None);
        let mut boss = boss_msg(50, json!([{"id": "PANTOGRAPH", "is_melted": true}]));
        freeze_fight_run_hp(&mut room2, &mut fight2, &mut boss);
        assert_eq!(fight2, Some((50, 80)));
        // No Pantograph held.
        let (mut room3, mut fight3) = (Some((50, 80)), None);
        let mut boss2 = boss_msg(50, json!([{"id": "BURNING_BLOOD"}]));
        freeze_fight_run_hp(&mut room3, &mut fight3, &mut boss2);
        assert_eq!(fight3, Some((50, 80)));
    }

    #[test]
    fn the_fool_is_advised_from_v0_113_and_refused_before_it() {
        // v0.113.0 and its hotfixes simulate the Fool, own seat or ally's.
        for build in ["v0.113.0", "v0.113.1"] {
            let fool = json!({"game_version": build, "run_state": {"character": "fool "}});
            assert_eq!(unsupported_character(&fool), None, "{build}");
            let ally = json!({"game_version": build, "run_state": {"character": "IRONCLAD"},
                              "allies": [{"character": "FOOL"}]});
            assert_eq!(unsupported_character(&ally), None, "{build}");
        }
        // Older builds have no Fool rules: refused, saying which build it needs.
        let old = json!({"game_version": "v0.111.0", "run_state": {"character": "fool "}});
        let (raw, note) = unsupported_character(&old).unwrap();
        assert_eq!(raw, "FOOL");
        assert_eq!(
            note,
            "The Fool needs game version v0.113.0 or newer; this game reports v0.111.0, so \
             there is no advice for this run."
        );
        let legacy_ally =
            json!({"run_state": {"character": "IRONCLAD"}, "allies": [{"character": "FOOL"}]});
        let (_, note) = unsupported_character(&legacy_ally).unwrap();
        assert!(note.contains("reports v0.107.1"), "{note}");
        // Other characters, a modded id and an unknown build are not judged here.
        assert!(unsupported_character(&json!({"run_state": {"character": "MODDED"}})).is_none());
        assert!(unsupported_character(
            &json!({"game_version": "v0.111.0", "run_state": {"character": "SILENT"}})
        )
        .is_none());
        assert!(unsupported_character(
            &json!({"game_version": "v9.9.9", "run_state": {"character": "FOOL"}})
        )
        .is_none());
    }

    #[test]
    fn only_a_lost_game_is_a_death() {
        assert!(is_death(&json!({"type": "game_over", "victory": false})));
        assert!(is_death(&json!({"type": "game_over", "result": "defeat"})));
        assert!(!is_death(&json!({"type": "game_over", "victory": true})));
        assert!(!is_death(
            &json!({"type": "run_complete", "victory": false})
        ));
    }

    #[test]
    fn a_missing_card_names_itself() {
        assert_eq!(
            engine_missing_content(
                "UnsupportedStateError",
                "$.hand[0]: unknown card id \"FOOLS_GOLD\""
            )
            .as_deref(),
            Some("the engine has no card FOOLS_GOLD yet")
        );
        assert_eq!(
            engine_missing_content("OverflowError", "unknown card id \"X\""),
            None
        );
    }

    #[test]
    fn a_pruned_tree_stays_connected() {
        let tree = json!({"root": 0, "nodes": [
            {"id": 0, "visits": 10},
            {"id": 1, "parent": 0, "visits": 1},
            {"id": 2, "parent": 1, "visits": 5},
            {"id": 3, "parent": 0, "visits": 0},
        ]});
        let out = prune_tree(&tree, 2, 2000);
        let ids: Vec<i64> = py::list(&out, "nodes")
            .iter()
            .map(|n| py::int_of(get(n, "id")))
            .collect();
        assert_eq!(ids, vec![0, 2, 1]);
        assert_eq!(out["pruned_from"], 4);
    }

    #[test]
    fn utc_stamps_are_calendar_dates() {
        assert_eq!(utc_stamp_at(0), "19700101-000000");
        assert_eq!(utc_stamp_at(1_795_000_000), "20261118-110640");
        assert_eq!(local_iso_now().len(), 26);
    }
}
