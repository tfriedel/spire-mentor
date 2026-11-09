//! The route dispatcher: the `/api/*` contract of
//! `sts2-pycli/python/workbench/server.py`, JSON in / JSON out.
//!
//! [`Workbench::call`] takes the route and the request body and returns the
//! response body the FastAPI handler returned, or an [`ApiError`] carrying
//! the HTTP status FastAPI answered with (400 no live run / illegal verb,
//! 404 unknown node or save, 409 replay diverged, 501 not in this build).
//! The host (the Web Worker) wraps that in a `Response` so the SPA's
//! `api.ts` keeps its error banners.
//!
//! Saves are split: the host owns the store, this side builds and restores
//! payloads ([`Workbench::build_save`] / [`Workbench::restore_save`]).

use serde_json::{json, Value};

use crate::advisors;
use crate::assets::AssetResolver;
use crate::enrich::EnrichCtx;
use crate::localization::embedded_monster_hp_ranges_for;
use crate::localization::Loc;
use crate::ppo;
use crate::replay;
use crate::runlog::{self, ReplayStore, RunLog};
use crate::saves;
use crate::session::{Session, SessionError};
use crate::step::live_step;
use sts2_core::GameVersion;

/// A failed route: `status` is the HTTP status, `detail` the message
/// (FastAPI's `{"detail": ...}` body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub detail: String,
}

/// Whether the e2e student advises the live run's hand-played moves
/// (`advisor: "e2e"`): the hint, the auto-action, the policy panel and the
/// choice ranker then answer with its evaluation (`session_e2e.rs`).
fn advises_e2e(sess: &Session) -> bool {
    cfg!(feature = "e2e") && sess.advisor_is_e2e()
}

/// One advisor surface the e2e student answers in the search's shape.
#[derive(Clone, Copy, Debug)]
enum E2eSurface {
    Hint,
    RankChoice,
    Recommended,
    PolicyPanel,
}

/// The e2e student's answer on `which` (`None` = the route's 409).
fn e2e_surface(sess: &mut Session, which: E2eSurface) -> Result<Option<Value>, ApiError> {
    #[cfg(feature = "e2e")]
    {
        Ok(match which {
            E2eSurface::Hint => sess.e2e_hint()?,
            E2eSurface::RankChoice => sess.e2e_rank_choice()?,
            E2eSurface::Recommended => sess.e2e_recommended()?,
            E2eSurface::PolicyPanel => sess.e2e_policy_panel()?,
        })
    }
    #[cfg(not(feature = "e2e"))]
    {
        let _ = (sess, which);
        Ok(None)
    }
}

/// `body["config"]` of an advisor request (`HintRequest.config`, default `{}`).
fn config_of(body: &Value) -> Value {
    body.get("config")
        .filter(|c| c.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}))
}

/// `{"type": "result", **result}`: the terminal frame of an advisor stream.
fn result_frame(mut result: Value) -> Value {
    if let Some(m) = result.as_object_mut() {
        m.insert("type".to_string(), json!("result"));
        result
    } else {
        json!({"type": "result", "value": result})
    }
}

/// `RecommendRequest.idx` (an integer, 422 otherwise — pydantic's answer).
fn idx_of(body: &Value) -> Result<i64, ApiError> {
    body.get("idx")
        .and_then(Value::as_i64)
        .ok_or_else(|| ApiError::new(422, "idx must be an integer"))
}

/// `ImportStateRequest.payload` — the pasted text (422 otherwise).
fn payload_of(body: &Value) -> Result<&str, ApiError> {
    body.get("payload")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::new(422, "payload must be a string"))
}

/// The optional `game_version` of a start / start-faithful / import-state
/// request (GH #801): `"v0.107.1"` (the default when absent) or
/// `"v0.111.0"`; anything else is a 400 naming the supported builds.
fn body_game_version(body: &Value) -> Result<GameVersion, ApiError> {
    crate::session::parse_game_version(body.get("game_version")).map_err(|e| ApiError::new(400, e))
}

/// `{}`: the default `args` of an action request.
fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

impl ApiError {
    #[must_use]
    pub fn new(status: u16, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: detail.into(),
        }
    }

    /// The `{"detail": ...}` body.
    #[must_use]
    pub fn body(&self) -> Value {
        json!({"detail": self.detail})
    }
}

impl From<SessionError> for ApiError {
    fn from(e: SessionError) -> Self {
        Self::new(e.status, e.message)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.status, self.detail)
    }
}

impl std::error::Error for ApiError {}

/// Host-supplied configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Where art is served from (`STS2_ASSETS_BASE_URL`); no trailing slash.
    pub assets_base: String,
    /// A millisecond clock for the `wpa-verify` deadline (`Date.now` in the
    /// browser). `None` = no wall-clock bound on the verify arms (native
    /// builds fall back to `Instant`).
    pub now_ms: Option<fn() -> f64>,
    /// A high-resolution millisecond clock (`performance.now()` in the
    /// browser) the e2e brain times its decisions with. `None` = the crate's
    /// fallback (`Instant` natively).
    pub perf_ms: Option<fn() -> f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            assets_base: "/assets".to_string(),
            now_ms: None,
            perf_ms: None,
        }
    }
}

/// The workbench: one live session plus the static data every route reads.
/// The display data (loc, art, HP ranges) is per game build (GH #801): a
/// live route reads the live session's build, a replay route the loaded
/// recording's.
#[derive(Debug)]
pub struct Workbench {
    config: Config,
    live: Option<Session>,
    /// The loaded replay (`Session.store`), from `load_run`.
    store: Option<ReplayStore>,
    /// Run logs written but not yet collected by the host (`take_run_log`),
    /// oldest first: a `quit`'s, and the one a start / import / resume /
    /// save-load writes for the live run it replaces.
    pending_logs: std::collections::VecDeque<RunLog>,
    /// Loaded PPO champions by lowercase character name (`load_ppo`), and
    /// the host's reason a load was not attempted / failed (`set_ppo_error`).
    ppo: std::collections::BTreeMap<String, ppo::Net>,
    ppo_errors: std::collections::BTreeMap<String, String>,
    /// The e2e Ironclad student (`load_e2e`) and why it is not loaded
    /// (`set_e2e_error`).
    #[cfg(feature = "e2e")]
    e2e: Option<crate::e2e::Net>,
    #[cfg(feature = "e2e")]
    e2e_error: Option<String>,
    /// The host's advisor preference (`POST /api/advisor`), applied to every
    /// run installed after it — start, save-load, import, resume — that the
    /// advisor can read; `None` until the host says (a resumed run keeps the
    /// advisor it was persisted with).
    advisor_pref: Option<String>,
    /// The SimLab distribution sweeps (GH #707), stepped by the host
    /// (`sweep_tick`).
    sweeps: crate::simlab_sweep::SweepRegistry,
}

impl Workbench {
    /// Build with the embedded localization / asset data.
    #[must_use]
    pub fn new(config: Config) -> Self {
        let mut config = config;
        while config.assets_base.ends_with('/') {
            config.assets_base.pop();
        }
        Self {
            config,
            live: None,
            store: None,
            pending_logs: std::collections::VecDeque::new(),
            ppo: std::collections::BTreeMap::new(),
            ppo_errors: std::collections::BTreeMap::new(),
            #[cfg(feature = "e2e")]
            e2e: None,
            #[cfg(feature = "e2e")]
            e2e_error: None,
            advisor_pref: None,
            sweeps: crate::simlab_sweep::SweepRegistry::default(),
        }
    }

    /// Epoch milliseconds: the host clock (`Config::now_ms`, `Date.now` in
    /// the browser) or `SystemTime` natively.
    fn now_ms(&self) -> f64 {
        if let Some(f) = self.config.now_ms {
            return f();
        }
        native_epoch_ms()
    }

    // ── SimLab sweeps (host-stepped; GH #707) ──────────────────────────

    /// Play up to `max_trials` trials of the oldest running sweep. Answers
    /// `{"active": <another tick is needed>, "finished": [ids that ended]}`
    /// so the host can persist what ended and decide whether to call again.
    pub fn sweep_tick(&mut self, max_trials: usize) -> Value {
        let clock: Box<dyn Fn() -> f64> = match self.config.now_ms {
            Some(f) => Box::new(f),
            None => Box::new(native_epoch_ms),
        };
        let finished = self.sweeps.tick(max_trials.max(1), &*clock);
        json!({"active": self.sweeps.has_running(), "finished": finished})
    }

    /// Whether a sweep still has trials to play.
    #[must_use]
    pub fn has_running_sweep(&self) -> bool {
        self.sweeps.has_running()
    }

    /// A sweep as the host stores it (`None` for an unknown id).
    #[must_use]
    pub fn export_sweep(&self, id: &str) -> Option<Value> {
        self.sweeps
            .get(id)
            .map(crate::simlab_sweep::SweepJob::export)
    }

    /// Restore a stored sweep (a reload); an id already live is left alone.
    pub fn import_sweep(&mut self, doc: &Value) -> Result<Value, ApiError> {
        self.sweeps
            .import(doc)
            .map_err(|e| ApiError::new(422, format!("bad stored sweep: {e}")))?;
        Ok(json!({"ok": true}))
    }

    /// `POST /api/bridge/render-tree {tree}`: the control plane's `/tree`
    /// with every node's `combat_log_raw` (+ `player_id`) rendered into
    /// `combat_log` — `server.py::bridge_tree`, the same formatter a
    /// simulated run goes through. Anything that is not a node passes
    /// through untouched.
    fn bridge_render_tree(body: &Value) -> Value {
        let mut tree = body.get("tree").cloned().unwrap_or_else(|| json!({}));
        // The bridge tree describes the REAL running game, not the simulated
        // session, so it is named in the default build like `server.py`.
        let names = crate::enrich::log_names(Loc::embedded());
        if let Some(nodes) = tree.get_mut("nodes").and_then(Value::as_array_mut) {
            for node in nodes.iter_mut().filter_map(Value::as_object_mut) {
                let raw = node.remove("combat_log_raw");
                let player_id = node
                    .remove("player_id")
                    .and_then(|v| v.as_str().map(str::to_string))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "PLAYER".to_string());
                if let Some(rows) = raw
                    .as_ref()
                    .and_then(Value::as_array)
                    .filter(|r| !r.is_empty())
                {
                    let rendered = crate::combatlog::format_combat_log(
                        rows,
                        Some(&names),
                        false,
                        &[],
                        &player_id,
                    );
                    node.insert("combat_log".to_string(), Value::Array(rendered));
                }
            }
        }
        tree
    }

    /// `POST /api/sweeps`: validate and enqueue; `{sweep_id}`.
    fn sweep_submit(&mut self, body: &Value) -> Result<Value, ApiError> {
        let spec = crate::simlab_sweep::SweepSpec::from_json(body)
            .map_err(|e| ApiError::new(422, format!("bad sweep spec: {e}")))?;
        let cells = spec
            .validate()
            .map_err(|e| ApiError::new(422, format!("bad sweep spec: {e}")))?;
        let now = self.now_ms();
        let id = self.sweeps.submit(spec, cells, now);
        Ok(json!({"sweep_id": id}))
    }

    /// The `/api/sweeps/{id}[/stop|/results]` routes.
    fn sweep_route(&mut self, method: &str, path: &str, query: &str) -> Result<Value, ApiError> {
        let rest = path.trim_start_matches("/api/sweeps/");
        let (id, tail) = rest.split_once('/').unwrap_or((rest, ""));
        match (method, tail) {
            ("POST", "stop") => {
                let stopped = self
                    .sweeps
                    .get_mut(id)
                    .is_some_and(crate::simlab_sweep::SweepJob::stop);
                if !stopped {
                    return Err(ApiError::new(404, format!("{id} not running")));
                }
                Ok(json!({"status": "stopping"}))
            }
            ("GET", "") => self
                .sweeps
                .get(id)
                .map(crate::simlab_sweep::SweepJob::progress)
                .ok_or_else(|| ApiError::new(404, id)),
            ("GET", "results") => {
                let job = self
                    .sweeps
                    .get(id)
                    .ok_or_else(|| ApiError::new(404, "no rows yet"))?;
                if job.rows().is_empty() {
                    return Err(ApiError::new(404, "no rows yet"));
                }
                let param = |k: &str| {
                    query
                        .split('&')
                        .filter_map(|kv| kv.split_once('='))
                        .find(|(key, _)| *key == k)
                        .map(|(_, v)| percent_decode(v))
                        .unwrap_or_default()
                };
                crate::simlab_sweep::results(
                    job.rows(),
                    &param("baseline"),
                    &param("baseline_deck"),
                )
                .map_err(|(status, detail)| ApiError::new(status, detail))
            }
            _ => Err(ApiError::new(404, format!("unknown route {method} {path}"))),
        }
    }

    // ── run logs (host-stored) ─────────────────────────────────────────

    /// `POST /api/load {path}` with the log's text the host fetched:
    /// parse it into the replay store. 422 `invalid JSONL: …` on a bad line.
    /// Answers `{count, resume}`.
    pub fn load_run(&mut self, path: &str, text: &str) -> Result<Value, ApiError> {
        let store = ReplayStore::parse(path, text)
            .map_err(|e| ApiError::new(422, format!("invalid JSONL: {e}")))?;
        let out = json!({"count": store.len(), "resume": runlog::resume_capability(&store)});
        self.store = Some(store);
        Ok(out)
    }

    /// The oldest run log not yet collected, once; `None` when every log
    /// has been taken. A host calls it until `None` after any route that
    /// can end a run, and stores each under `filename`.
    pub fn take_run_log(&mut self) -> Option<RunLog> {
        self.pending_logs.pop_front()
    }

    /// `GET /api/mcts-runs` over the host's `(filename, text)` logs.
    #[must_use]
    pub fn mcts_runs(logs: &[(String, String)]) -> Value {
        runlog::mcts_runs(logs)
    }

    /// `GET /api/mcts-runs/aggregate` over the host's logs.
    #[must_use]
    pub fn aggregate_runs(logs: &[(String, String)]) -> Value {
        runlog::aggregate_runs(logs)
    }

    fn store(&self) -> Result<&ReplayStore, ApiError> {
        self.store
            .as_ref()
            .ok_or_else(|| ApiError::new(400, "no run loaded"))
    }

    fn store_mut(&mut self) -> Result<&mut ReplayStore, ApiError> {
        self.store
            .as_mut()
            .ok_or_else(|| ApiError::new(400, "no run loaded"))
    }

    // ── PPO champions (host-loaded) ────────────────────────────────────

    /// The champion ONNX filename the host should fetch for `character`
    /// (400 for an unknown character).
    pub fn ppo_champion_file(character: &str) -> Result<&'static str, ApiError> {
        ppo::champion_file(character).map_err(|e| ApiError::new(400, e))
    }

    /// Load `character`'s champion from the ONNX bytes and attach it to the
    /// live run if it is that character's. 400 with tract's error text when
    /// the bytes are not a loadable export.
    pub fn load_ppo(&mut self, character: &str, bytes: &[u8]) -> Result<Value, ApiError> {
        let key = character.to_lowercase();
        let net = ppo::load(character, bytes).map_err(|e| ApiError::new(400, e))?;
        self.ppo.insert(key.clone(), net);
        self.ppo_errors.remove(&key);
        self.attach_ppo();
        Ok(json!({"ok": true, "character": character}))
    }

    /// Record why the host could not load `character`'s champion (the
    /// panel and the brain report it); clears on a later `load_ppo`.
    pub fn set_ppo_error(&mut self, character: &str, reason: &str) {
        self.ppo_errors
            .insert(character.to_lowercase(), reason.to_string());
        self.attach_ppo();
    }

    /// Whether `character`'s champion is loaded.
    #[must_use]
    pub fn has_ppo(&self, character: &str) -> bool {
        self.ppo.contains_key(&character.to_lowercase())
    }

    // ── the e2e student (host-loaded, on demand) ───────────────────────

    /// The e2e weights filename the host serves under `/e2e/`.
    #[must_use]
    pub fn e2e_file() -> &'static str {
        #[cfg(feature = "e2e")]
        {
            crate::e2e::E2E_FILE
        }
        #[cfg(not(feature = "e2e"))]
        {
            ""
        }
    }

    /// Load the e2e student from the `.e2ew` bytes and attach it to the live
    /// run. 400 with the loader's error when the bytes are not a weights file
    /// of the build the student plays.
    pub fn load_e2e(&mut self, bytes: &[u8]) -> Result<Value, ApiError> {
        #[cfg(feature = "e2e")]
        {
            let net = crate::e2e::load(bytes).map_err(|e| ApiError::new(400, e))?;
            let header = json!({
                "source": net.header().source,
                "source_sha256": net.header().source_sha256,
                "game_version": net.header().game_version,
            });
            self.e2e = Some(net);
            self.e2e_error = None;
            self.attach_ppo();
            Ok(json!({"ok": true, "file": crate::e2e::E2E_FILE, "header": header}))
        }
        #[cfg(not(feature = "e2e"))]
        {
            let _ = bytes;
            Err(ApiError::new(400, "this build has no e2e feature"))
        }
    }

    /// Record why the host could not load the e2e student; clears on a later
    /// `load_e2e`.
    pub fn set_e2e_error(&mut self, reason: &str) {
        #[cfg(feature = "e2e")]
        {
            self.e2e_error = Some(reason.to_string());
            self.attach_ppo();
        }
        #[cfg(not(feature = "e2e"))]
        let _ = reason;
    }

    /// Whether the e2e student is loaded.
    #[must_use]
    pub fn has_e2e(&self) -> bool {
        #[cfg(feature = "e2e")]
        {
            self.e2e.is_some()
        }
        #[cfg(not(feature = "e2e"))]
        {
            false
        }
    }

    /// `POST /api/e2e/selfplay {seed, ascension?, max_decisions?}`: the
    /// student plays a fresh campaign run with no UI (speed + parity probe).
    fn e2e_selfplay(&mut self, body: &Value) -> Result<Value, ApiError> {
        #[cfg(feature = "e2e")]
        {
            let net = self.e2e.clone().ok_or_else(|| {
                ApiError::new(
                    crate::e2e::NO_NET_STATUS,
                    crate::e2e::missing_net_error(self.e2e_error.as_deref()),
                )
            })?;
            let seed = match body.get("seed") {
                Some(Value::Number(n)) => n.as_i64(),
                Some(Value::String(s)) => Some(i64::from(crate::runmode::hash_seed(s))),
                _ => None,
            }
            .filter(|s| (0..1_i64 << 31).contains(s))
            .ok_or_else(|| {
                ApiError::new(
                    400,
                    "seed must be an integer in [0, 2**31) or a run_N string",
                )
            })?;
            let ascension = body.get("ascension").and_then(Value::as_u64).map_or(
                Ok(crate::e2e::DEFAULT_ASCENSION),
                |a| {
                    u8::try_from(a)
                        .ok()
                        .filter(|a| *a <= 10)
                        .ok_or_else(|| ApiError::new(400, format!("ascension {a} out of range")))
                },
            )?;
            let max_decisions = body
                .get("max_decisions")
                .and_then(Value::as_u64)
                .map_or(usize::MAX, |m| usize::try_from(m).unwrap_or(usize::MAX));
            let clock = self.config.perf_ms.unwrap_or(crate::e2e::fallback_ms);
            Ok(crate::e2e::selfplay(
                &net,
                seed,
                ascension,
                max_decisions,
                clock,
            ))
        }
        #[cfg(not(feature = "e2e"))]
        {
            let _ = body;
            Err(ApiError::new(404, "this build has no e2e feature"))
        }
    }

    /// `POST /api/seedcheck/{preview,attempt,summary,log}` (the seed
    /// checker, `seedcheck.html`):
    ///
    /// - `preview {seed}` — Neow offers, act 1 and bosses of a game seed;
    /// - `attempt {seed, seq, k, max_decisions?}` — one attempt
    ///   (`e2e::attempt`), 424 without the net;
    /// - `summary {seed, classes}` — the attempt's named run-level choices;
    /// - `log {seed, classes, seq, k, now}` — the attempt as a run log
    ///   `{path, text}` for the host to store (424 without the net).
    ///
    /// A bad seed is a 400 naming the character.
    fn seedcheck(&mut self, body: &Value, what: &str) -> Result<Value, ApiError> {
        #[cfg(feature = "e2e")]
        {
            use crate::e2e::Seq;
            let bad = |e: String| ApiError::new(400, e);
            let seed = body
                .get("seed")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::new(400, "seed must be a string"))?;
            let seq = || {
                Seq::parse(body.get("seq").and_then(Value::as_str).unwrap_or("check")).map_err(bad)
            };
            let k = || {
                body.get("k")
                    .and_then(Value::as_u64)
                    .and_then(|k| u32::try_from(k).ok())
                    .ok_or_else(|| ApiError::new(400, "k must be a non-negative integer"))
            };
            let classes = || -> Result<Vec<usize>, ApiError> {
                body.get("classes")
                    .and_then(Value::as_array)
                    .ok_or_else(|| ApiError::new(400, "classes must be a list"))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|c| usize::try_from(c).ok())
                            .ok_or_else(|| ApiError::new(400, "classes must be integers"))
                    })
                    .collect()
            };
            // The run Instinct plays: Ironclad on v0.111.0 only (the page
            // shows the other characters and builds, grayed out), at any
            // ascension 0-10 (it trained at A10).
            let character = body
                .get("character")
                .and_then(Value::as_str)
                .unwrap_or(crate::e2e::CHARACTER);
            if !character.eq_ignore_ascii_case(crate::e2e::CHARACTER) {
                return Err(bad(format!(
                    "the seed checker only plays {} for now, not {character}",
                    crate::e2e::CHARACTER
                )));
            }
            let version = body
                .get("game_version")
                .and_then(Value::as_str)
                .unwrap_or(crate::e2e::GAME_VERSION);
            if version != crate::e2e::GAME_VERSION {
                return Err(bad(format!(
                    "the seed checker only plays {} for now, not {version}",
                    crate::e2e::GAME_VERSION
                )));
            }
            let ascension = match body.get("ascension") {
                None | Some(Value::Null) => crate::e2e::DEFAULT_ASCENSION,
                Some(v) => v
                    .as_u64()
                    .and_then(|a| u8::try_from(a).ok())
                    .filter(|a| *a <= 10)
                    .ok_or_else(|| {
                        ApiError::new(400, format!("ascension must be 0-10, not {v}"))
                    })?,
            };
            let net = || {
                self.e2e.clone().ok_or_else(|| {
                    ApiError::new(
                        crate::e2e::NO_NET_STATUS,
                        crate::e2e::missing_net_error(self.e2e_error.as_deref()),
                    )
                })
            };
            match what {
                "preview" => crate::seedcheck::preview(seed, ascension).map_err(bad),
                "summary" => {
                    crate::seedcheck::line_summary(seed, ascension, &classes()?).map_err(bad)
                }
                "attempt" => {
                    sts2_e2e::env::canonical_game_seed(seed).map_err(bad)?;
                    let net = net()?;
                    let max_decisions = body
                        .get("max_decisions")
                        .and_then(Value::as_u64)
                        .map_or(20_000, |m| usize::try_from(m).unwrap_or(usize::MAX));
                    let clock = self.config.perf_ms.unwrap_or(crate::e2e::fallback_ms);
                    let a = crate::e2e::attempt(
                        &net,
                        seed,
                        ascension,
                        seq()?,
                        k()?,
                        max_decisions,
                        clock,
                    )
                    .map_err(bad)?;
                    serde_json::to_value(a).map_err(|e| ApiError::new(500, e.to_string()))
                }
                "log" => {
                    sts2_e2e::env::canonical_game_seed(seed).map_err(bad)?;
                    net()?;
                    let now = body.get("now").and_then(Value::as_str).unwrap_or("");
                    let mut sess = self.new_session(crate::e2e::CHARACTER, GameVersion::V0_111_0);
                    let log = crate::seedcheck::run_log(
                        &mut sess,
                        seed,
                        ascension,
                        &classes()?,
                        seq()?,
                        k()?,
                        now,
                    )
                    .map_err(|e| ApiError::new(e.status, e.message))?;
                    Ok(json!({"path": log.filename, "text": log.text}))
                }
                _ => Err(ApiError::new(404, "unknown seedcheck route")),
            }
        }
        #[cfg(not(feature = "e2e"))]
        {
            let _ = (body, what);
            Err(ApiError::new(404, "this build has no e2e feature"))
        }
    }

    /// Point the live session at its character's net (or error).
    fn attach_ppo(&mut self) {
        let Some(sess) = self.live.as_mut() else {
            return;
        };
        sess.set_perf_clock(self.config.perf_ms);
        #[cfg(feature = "e2e")]
        sess.set_e2e(self.e2e.clone(), self.e2e_error.clone());
        let key = sess.character.to_lowercase();
        let net = self.ppo.get(&key).cloned();
        let error = self
            .ppo_errors
            .get(&key)
            .map(|r| ppo::missing_net_error(&sess.character, Some(r)));
        sess.set_ppo(net, error);
    }

    /// Install `sess` as the live run, quitting the one it replaces as
    /// `/api/live/quit` would: that run's log is written, stamped `now`, for
    /// the host to collect. Python's `LiveRuns.replace` and `import-state`
    /// quit the prior run the same way, so starting over, importing,
    /// resuming a recording or loading a save keeps the run that was being
    /// played instead of dropping it.
    fn replace_live(&mut self, sess: Session, now: &str) {
        if let Some(prior) = self.live.as_mut() {
            let log = runlog::write_run_log(prior, now);
            prior.quit(log.as_ref().map(|l| l.filename.clone()));
            self.pending_logs.extend(log);
        }
        self.install(sess);
    }

    /// Install `sess` as the live run (discarding the old one) with its
    /// champion attached.
    fn install(&mut self, sess: Session) {
        if let Some(old) = self.live.as_mut() {
            old.discard();
        }
        self.live = Some(sess);
        self.attach_ppo();
        self.apply_advisor_pref();
    }

    /// Put the host's advisor preference on the live run: the e2e advisor
    /// where the student can read the run, the search everywhere else.
    fn apply_advisor_pref(&mut self) {
        let (Some(pref), Some(sess)) = (self.advisor_pref.as_deref(), self.live.as_mut()) else {
            return;
        };
        if sess.set_advisor(pref).is_err() {
            sess.advisor = crate::session::ADVISOR_SEARCH.to_string();
        }
    }

    /// `POST /api/advisor {advisor}`: the host's advisor preference
    /// (`search` | `e2e`), kept for the runs installed after it and put on
    /// the live run where the student can read it. `{advisor, live}`: the
    /// preference and the live run's advisor (`null` without a run). 400 for
    /// an unknown name or `e2e` in a build without the e2e feature.
    fn set_advisor_pref(&mut self, body: &Value) -> Result<Value, ApiError> {
        let a = body
            .get("advisor")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::new(400, "advisor must be a string"))?;
        match a {
            crate::session::ADVISOR_SEARCH => {}
            crate::session::ADVISOR_E2E if cfg!(feature = "e2e") => {}
            crate::session::ADVISOR_E2E => {
                return Err(ApiError::new(400, "this build has no e2e feature"));
            }
            other => {
                return Err(ApiError::new(
                    400,
                    format!("advisor must be search|e2e, got '{other}'"),
                ));
            }
        }
        self.advisor_pref = Some(a.to_string());
        self.apply_advisor_pref();
        Ok(json!({
            "advisor": a,
            "live": self.live.as_ref().map(|s| s.advisor.clone()),
        }))
    }

    /// A fresh session in game build `version`, pre-wired to `character`'s
    /// champion, so the root node (and everything after) is scored.
    fn new_session(&self, character: &str, version: GameVersion) -> Session {
        let mut sess = Session::new_for(version);
        let key = character.to_lowercase();
        let error = self
            .ppo_errors
            .get(&key)
            .map(|r| ppo::missing_net_error(character, Some(r)));
        sess.set_ppo(self.ppo.get(&key).cloned(), error);
        sess.set_perf_clock(self.config.perf_ms);
        #[cfg(feature = "e2e")]
        sess.set_e2e(self.e2e.clone(), self.e2e_error.clone());
        sess
    }

    /// The enrichment context of the live session's build (the default
    /// build when no run is live).
    fn ctx(&self) -> EnrichCtx<'_> {
        self.ctx_for(self.live_version())
    }

    /// The live session's build (the default when no run is live).
    fn live_version(&self) -> GameVersion {
        self.live
            .as_ref()
            .map_or_else(GameVersion::default, |s| s.game_version)
    }

    /// The enrichment context of the loaded recording's build.
    fn store_ctx(&self) -> EnrichCtx<'_> {
        self.ctx_for(self.store_version())
    }

    /// The loaded recording's build (the default when none is loaded).
    fn store_version(&self) -> GameVersion {
        self.store
            .as_ref()
            .map_or_else(GameVersion::default, ReplayStore::game_version)
    }

    fn ctx_for(&self, version: GameVersion) -> EnrichCtx<'_> {
        EnrichCtx {
            assets: AssetResolver::embedded_for(version),
            loc: Loc::embedded_for(version),
            hp_ranges: Some(embedded_monster_hp_ranges_for(version)),
            assets_base: &self.config.assets_base,
        }
    }

    /// The live session, if any.
    #[must_use]
    pub fn session(&self) -> Option<&Session> {
        self.live.as_ref()
    }

    fn live(&mut self) -> Result<&mut Session, ApiError> {
        self.live
            .as_mut()
            .ok_or_else(|| ApiError::new(400, "no live run"))
    }

    /// Dispatch one route. `method` is `"GET"` or `"POST"`; `body` is the
    /// JSON request body (ignored for GET). Unknown routes are 404.
    pub fn call(&mut self, method: &str, route: &str, body: &Value) -> Result<Value, ApiError> {
        // Strip a query string: the SPA never sends one on these routes.
        let path = route.split('?').next().unwrap_or(route);
        match (method, path) {
            ("GET", "/api/health") => Ok(json!({"ok": true})),
            ("GET", "/api/capabilities") => Ok(json!({
                "backends": ["rust"],
                // The engine has no upload target for bug reports (the
                // static host may: Cloudflare Pages runs a Function the SPA
                // probes for, web/src/bugreport.ts). State import rebuilds
                // both lanes — the run position and, since GH #708, the exact
                // fight (`combat_import.rs`).
                "features": {
                    "advisors": true,
                    "run_mode": true,
                    "saves": true,
                    "replay": true,
                    "bug_reports": false,
                    "state_import": true,
                    // The SimLab position sandbox (edit a board, per-move MCTS)
                    // runs in the wasm, and since GH #707 so do the distribution
                    // SWEEPS: a host-stepped job (`sweep_tick`) instead of the
                    // desktop's process pool, see `simlab_sweep.rs`.
                    "simlab": true,
                    "simlab_sweeps": true,
                    // The `e2e` brain (the e2e Ironclad student, every
                    // decision, no search); its net is fetched on demand.
                    "e2e": cfg!(feature = "e2e"),
                },
                // GH #801: the game builds a run can be started in (the
                // `game_version` of start / start-faithful / import-state) —
                // only those whose display bundle this binary embeds.
                "game_versions": crate::data::embedded_versions()
                    .into_iter()
                    .map(GameVersion::as_str)
                    .collect::<Vec<_>>(),
                "default_game_version": GameVersion::default().as_str(),
            })),
            ("GET", "/api/meta") => Ok(crate::simlab_catalog::meta()),
            ("GET", "/api/simlab/catalog") => {
                // `?game_version=…` names the build whose loc names the
                // catalog (GH #877); absent = the default build.
                let query = route.split_once('?').map_or("", |(_, q)| q);
                let raw = query
                    .split('&')
                    .filter_map(|kv| kv.split_once('='))
                    .find(|(key, _)| *key == "game_version")
                    .map(|(_, v)| json!(percent_decode(v)));
                let version = crate::session::parse_game_version(raw.as_ref())
                    .map_err(|e| ApiError::new(400, e))?;
                Ok(crate::simlab_catalog::catalog_for(version))
            }
            ("POST", "/api/simlab/position/build") => Self::simlab_build(body),
            ("POST", "/api/simlab/position/new") => Self::simlab_new(body),
            ("POST", "/api/simlab/position/import") => Self::simlab_import(body),
            ("POST", "/api/simlab/position/moves") => Self::simlab_moves(body),
            ("POST", "/api/simlab/position/mcts") => Self::simlab_mcts(body),
            // Live-game bridge (GH #709): the browser talks to the advisor's
            // control plane directly; the one proxy transform the Python
            // server did — rendering each node's raw combat-log rows with the
            // loc map — is this route.
            ("POST", "/api/bridge/render-tree") => Ok(Self::bridge_render_tree(body)),
            ("POST", "/api/sweeps") => self.sweep_submit(body),
            ("GET", "/api/sweeps") => Ok(self.sweeps.list()),
            (m, p) if p.starts_with("/api/sweeps/") => {
                let query = route.split_once('?').map_or("", |(_, q)| q);
                self.sweep_route(m, p, query)
            }
            ("POST", "/api/live/start") => self.live_start(body),
            ("POST", "/api/live/start-faithful") => self.live_start_faithful(body),
            ("POST", "/api/live/import-state/preview") => Self::live_import_preview(body),
            ("POST", "/api/live/import-state") => self.live_import_state(body),
            ("GET", "/api/live/state") => {
                let state = self.live()?.state()?;
                Ok(live_step(state, &self.ctx()))
            }
            ("GET", "/api/live/map") => self.live_map(),
            ("POST", "/api/live/action") => self.live_action(body),
            ("GET", "/api/live/timeline") => Ok(self.live()?.timeline()),
            ("POST", "/api/live/seek") => {
                let node_id = body
                    .get("node_id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| ApiError::new(400, "node_id must be an integer"))?;
                let state = self.live()?.seek(node_id)?;
                Ok(json!({"step": live_step(state, &self.ctx())}))
            }
            ("POST", "/api/bench") => self.bench(body),
            // ── advisors (M2) ────────────────────────────────────────
            ("POST", "/api/live/hint") => {
                let sess = self.live()?;
                let out = if advises_e2e(sess) {
                    e2e_surface(sess, E2eSurface::Hint)?
                } else {
                    advisors::hint_remembered(sess, &config_of(body))?
                };
                out.ok_or_else(|| ApiError::new(409, "hint only available during combat"))
            }
            ("POST", "/api/live/e2e-advice") => self.live_e2e_advice(),
            ("POST", "/api/live/search_tree") => {
                advisors::live_search_tree(self.live()?, &config_of(body))?
                    .ok_or_else(|| ApiError::new(409, "search tree only available during combat"))
            }
            ("POST", "/api/live/choice-ranker") => {
                let sess = self.live()?;
                let out = if advises_e2e(sess) {
                    e2e_surface(sess, E2eSurface::RankChoice)?
                } else {
                    advisors::rank_choice(sess, &config_of(body))?
                };
                out.ok_or_else(|| {
                    ApiError::new(
                        409,
                        "choice ranker only available at an in-combat card-select",
                    )
                })
            }
            ("POST", "/api/live/auto-action") => {
                let sess = self.live()?;
                let out = if advises_e2e(sess) {
                    e2e_surface(sess, E2eSurface::Recommended)?
                } else {
                    sess.recommended_action()
                };
                out.ok_or_else(|| ApiError::new(409, "no recommendation here"))
            }
            ("POST", "/api/live/last-turn-grade") => {
                let sess = self.live()?;
                // The per-turn coach grades against the combat search; under
                // the e2e advisor it stays silent rather than contradict the
                // student's hints with the search's line.
                let out = if advises_e2e(sess) {
                    None
                } else {
                    advisors::last_turn_grade(sess, &config_of(body))?
                };
                out.ok_or_else(|| ApiError::new(409, "no gradable turn just ended"))
            }
            ("POST", "/api/live/last-choice-grade") => self
                .live()?
                .last_choice_grade()?
                .ok_or_else(|| ApiError::new(409, "no prior run-level choice to grade")),
            ("POST", "/api/live/regret-review") => {
                Ok(advisors::review_run(self.live()?, &config_of(body))?)
            }
            ("POST", "/api/live/card-advisor") => {
                advisors::advise_card_reward(self.live()?, &config_of(body), None)?.ok_or_else(
                    || ApiError::new(409, "card advisor only available at a card reward"),
                )
            }
            ("POST", "/api/live/decision-advisor") => {
                advisors::advise_decision(self.live()?, &config_of(body), None)?.ok_or_else(|| {
                    ApiError::new(
                        409,
                        "decision advisor only available at a shop, event, or choice",
                    )
                })
            }
            // ── run mode / autobattle / PPO / WPA (M3) ──────────────
            ("POST", "/api/live/run-mode") => self.live_run_mode(body),
            ("POST", "/api/advisor") => self.set_advisor_pref(body),
            ("POST", "/api/e2e/selfplay") => self.e2e_selfplay(body),
            ("POST", "/api/seedcheck/preview") => self.seedcheck(body, "preview"),
            ("POST", "/api/seedcheck/attempt") => self.seedcheck(body, "attempt"),
            ("POST", "/api/seedcheck/summary") => self.seedcheck(body, "summary"),
            ("POST", "/api/seedcheck/log") => self.seedcheck(body, "log"),
            ("POST", "/api/live/run-mode-step" | "/api/live/faithful-step") => {
                let sess = self.live()?;
                let out = sess.run_mode_step()?;
                let done = out["done"].as_bool().unwrap_or(false) || !out["error"].is_null();
                Ok(json!({
                    "step": live_step(out["state"].clone(), &self.ctx()),
                    "done": done,
                    "over": out["over"],
                    "victory": out["victory"],
                    "steps": out["steps"],
                    "error": out["error"],
                    "debug": out["debug"],
                    "search": out["search"],
                }))
            }
            ("POST", "/api/live/play-to-end") => {
                let sess = self.live()?;
                let out = sess.play_to_end(20_000)?;
                let timeline = sess.timeline();
                Ok(json!({
                    "step": live_step(out["state"].clone(), &self.ctx()),
                    "steps": out["steps"],
                    "victory": out["victory"],
                    "over": out["over"],
                    "error": out["error"],
                    "timeline": timeline,
                }))
            }
            ("POST", "/api/live/autobattle/advance") => {
                let out = self.live()?.autobattle_advance(500)?;
                Ok(json!({
                    "step": live_step(out["state"].clone(), &self.ctx()),
                    "diff": out["diff"],
                    "autoplayed": out["autoplayed"],
                    "stopped": out["stopped"],
                    "over": out["over"],
                    "won": out["won"],
                    "error": out["error"],
                }))
            }
            ("POST", "/api/live/autobattle/step") => {
                let out = self.live()?.autobattle_step()?;
                Ok(json!({
                    "step": live_step(out["state"].clone(), &self.ctx()),
                    "diff": out["diff"],
                    "played": out["played"],
                    "stopped": out["stopped"],
                    "over": out["over"],
                    "won": out["won"],
                    "error": out["error"],
                    "search": out["search"],
                    "debug": out["debug"],
                }))
            }
            ("GET", "/api/live/autobattle/verdict") => Ok(self.live()?.autobattle_verdict()?),
            ("POST", "/api/live/ppo-policy") => {
                let sess = self.live()?;
                let out = if advises_e2e(sess) {
                    e2e_surface(sess, E2eSurface::PolicyPanel)?
                } else {
                    sess.ppo_policy_panel()
                };
                out.ok_or_else(|| {
                    ApiError::new(409, "ppo policy only available at a non-combat decision")
                })
            }
            ("POST", "/api/live/wpa-review") => Ok(self.live()?.wpa_review(&config_of(body))?),
            ("POST", "/api/live/wpa-verify") => self.live_wpa_verify(body),
            ("POST", "/api/live/quit") => {
                let now = host_now(body);
                let sess = self.live()?;
                let log = runlog::write_run_log(sess, &now);
                let out = sess.quit(log.as_ref().map(|l| l.filename.clone()));
                self.pending_logs.extend(log);
                self.live = None;
                Ok(out)
            }
            ("POST", "/api/live/save" | "/api/saves/load") | ("GET", "/api/saves") => {
                Err(ApiError::new(
                    501,
                    "saves are handled by the host (build_save / restore_save)",
                ))
            }
            // ── the replay viewer (M4) ───────────────────────────────
            ("POST", "/api/load")
            | (
                "GET",
                "/api/runs" | "/api/runs/stat" | "/api/mcts-runs" | "/api/mcts-runs/aggregate",
            ) => Err(ApiError::new(
                501,
                "run logs are handled by the host (load_run / mcts_runs)",
            )),
            ("GET", "/api/timeline") => Ok(replay::timeline(
                self.store()?,
                Loc::embedded_for(self.store_version()),
            )),
            ("GET", "/api/map") => Ok(replay::map(self.store()?, &self.store_ctx())),
            ("GET", "/api/coach") => Ok(self
                .store
                .as_ref()
                .map_or_else(runlog::empty_coach, ReplayStore::coach)),
            ("GET", p) if p.starts_with("/api/step/") => {
                let rest = &p["/api/step/".len()..];
                let (idx_s, sub) = rest.split_once('/').unwrap_or((rest, ""));
                let idx: i64 = idx_s.parse().map_err(|_| {
                    ApiError::new(422, format!("step index must be an integer, got {idx_s:?}"))
                })?;
                match sub {
                    "" => replay::step(self.store()?, idx, &self.store_ctx()),
                    "deck-analysis" => replay::step_deck_analysis(self.store()?, idx),
                    _ => Err(ApiError::new(
                        404,
                        format!("{method} {path} is not available in the browser workbench"),
                    )),
                }
            }
            ("GET", "/api/deck/trend") => Ok(replay::deck_trend(self.store()?)),
            ("POST", "/api/deck/reward-preview") => {
                replay::reward_preview(body).map_err(|e| ApiError::new(400, e))
            }
            ("GET", "/api/live/deck-analysis") => {
                let state = self.live()?.state()?;
                Ok(replay::live_deck_analysis(&state, self.live_version()))
            }
            ("POST", "/api/recommend") => {
                let idx = idx_of(body)?;
                replay::recommend(self.store_mut()?, idx, &config_of(body))
            }
            ("POST", "/api/search_tree") => {
                let idx = idx_of(body)?;
                replay::search_tree(self.store()?, idx, &config_of(body))
            }
            ("POST", "/api/regret-review") => Ok(replay::regret_review(
                self.store()?,
                &config_of(body),
                Loc::embedded_for(self.store_version()),
            )),
            ("POST", "/api/wpa-review") => Ok(replay::wpa_review(self.store()?, &config_of(body))),
            ("POST", "/api/live/from_replay") => self.live_from_replay(body),
            _ => Err(ApiError::new(
                if path.starts_with("/api/") { 501 } else { 404 },
                format!("{method} {path} is not available in the browser workbench"),
            )),
        }
    }

    /// `POST /api/bench {sims?, playouts?, seed?}`: the M1 phone-speed
    /// probe over the live combat (400 outside combat). The host times it.
    /// The streaming advisor routes (`…/card-advisor/stream`,
    /// `…/decision-advisor/stream`): every NDJSON frame goes to `sink` —
    /// `{"type": "start", …}`, one `{"type": "tick", …}` per playout, then
    /// exactly one terminal `{"type": "result", …}` or `{"type": "error",
    /// "error"}`. Never fails: an HTTP status cannot change once the body
    /// has started, so "not advisable" and "no live run" are `error` frames
    /// (what `workbench/advice_stream.py` does). A non-stream route through
    /// here answers its plain reply as one `result` / `error` frame.
    pub fn call_stream(
        &mut self,
        method: &str,
        route: &str,
        body: &Value,
        sink: &mut dyn FnMut(&Value),
    ) {
        type Advisor = fn(
            &Session,
            &Value,
            Option<advisors::Progress<'_>>,
        ) -> Result<Option<Value>, SessionError>;
        let path = route.split('?').next().unwrap_or(route);
        let (unavailable, run): (&str, Advisor) = match (method, path) {
            ("POST", "/api/live/card-advisor/stream") => {
                ("not at a card reward", advisors::advise_card_reward)
            }
            ("POST", "/api/live/decision-advisor/stream") => {
                ("not at an advisable decision", advisors::advise_decision)
            }
            ("POST", "/api/live/last-choice-check/stream") => {
                ("nothing to check", crate::session_choice::last_choice_check)
            }
            _ => {
                match self.call(method, route, body) {
                    Ok(v) => sink(&result_frame(v)),
                    Err(e) => sink(&json!({"type": "error", "error": e.detail})),
                }
                return;
            }
        };
        let sess = match self.live() {
            Ok(s) => s,
            Err(e) => {
                sink(&json!({"type": "error", "error": e.detail}));
                return;
            }
        };
        let frame = match run(sess, &config_of(body), Some(&mut *sink)) {
            Ok(Some(v)) => result_frame(v),
            Ok(None) => json!({"type": "error", "error": unavailable}),
            Err(e) => json!({"type": "error", "error": e.message}),
        };
        sink(&frame);
    }

    fn bench(&mut self, body: &Value) -> Result<Value, ApiError> {
        let sims = u32::try_from(body.get("sims").and_then(Value::as_u64).unwrap_or(200))
            .unwrap_or(u32::MAX);
        let playouts = u32::try_from(body.get("playouts").and_then(Value::as_u64).unwrap_or(0))
            .unwrap_or(u32::MAX);
        let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
        let sess = self.live()?;
        let cs = sess
            .game()
            .and_then(|g| g.combat_state())
            .ok_or_else(|| ApiError::new(400, "bench needs a live combat"))?;
        Ok(crate::bench::run(cs, sims, playouts, seed))
    }

    fn live_start(&mut self, body: &Value) -> Result<Value, ApiError> {
        let s = |k: &str, default: &str| {
            body.get(k)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        };
        let character = s("character", "Ironclad");
        let seed = s("seed", "test");
        let backend = s("backend", "rust");
        let act1 = s("act1", "roll");
        let ascension = body.get("ascension").and_then(Value::as_i64).unwrap_or(0);
        let ascension = u8::try_from(ascension)
            .map_err(|_| ApiError::new(400, format!("ascension {ascension} out of range")))?;
        let version = body_game_version(body)?;
        if backend != "rust" {
            return Err(ApiError::new(
                400,
                format!(
                    "could not start the {backend:?} engine: only the rust engine exists in the browser workbench"
                ),
            ));
        }
        // Build the new run BEFORE touching the slot, so a rejected start never
        // costs the client the run they already had.
        let mut sess = self.new_session(&character, version);
        let state = match sess.start(&character, &seed, ascension, &act1) {
            Ok(v) => v,
            Err(e) => {
                sess.discard();
                return Err(ApiError::new(
                    400,
                    format!("could not start the {backend:?} engine: {}", e.message),
                ));
            }
        };
        self.replace_live(sess, &host_now(body));
        Ok(live_step(state, &self.ctx()))
    }

    /// `POST /api/live/from_replay {idx}`: take the loaded recording over at
    /// step `idx`, replacing any current live run — the whole run rebuilt
    /// from its seed along the recorded line.
    /// `POST /api/live/import-state/preview`: parse a pasted live-game state
    /// and describe it WITHOUT touching the current run — which run it is,
    /// where it is, and what the import will and will not reproduce. 422 with
    /// a user-facing message when the paste cannot be read.
    fn live_import_preview(body: &Value) -> Result<Value, ApiError> {
        let payload = payload_of(body)?;
        let imported = crate::state_import::parse_pasted_state(payload)
            .map_err(|e| ApiError::new(422, e.0))?;
        Ok(imported.preview())
    }

    // ── SimLab sandbox (stateless; the position travels in the request) ──────
    // Ports of `simlab/web/app.py`'s `/api/simlab/position/*` routes. A shape
    // the engine could never accept is a 422 (`PositionError`), matching the
    // desktop `_position_or_422`. See `crate::simlab`.

    /// `POST /api/simlab/position/build`: round-trip a position through the
    /// engine; 422 names what it refused.
    fn simlab_build(body: &Value) -> Result<Value, ApiError> {
        let empty = json!({});
        let pos = body.get("position").unwrap_or(&empty);
        let built = crate::simlab::build_position(pos).map_err(|e| ApiError::new(422, e.0))?;
        Ok(json!({"position": built.position, "wire": built.wire, "notes": built.notes}))
    }

    /// `POST /api/simlab/position/new`: a fresh fight from turn 1 — encounter
    /// (or lineup) × character × seed.
    fn simlab_new(body: &Value) -> Result<Value, ApiError> {
        let str_list = |key: &str| -> Option<Vec<String>> {
            body.get(key).and_then(Value::as_array).map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
        };
        let character = body
            .get("character")
            .and_then(Value::as_str)
            .unwrap_or("Ironclad");
        let (starter_hp, starter_deck, starter_relic, _) =
            crate::simlab_catalog::starter_for(character);
        // The selector's build, like `/api/live/start` (GH #877).
        let version = body_game_version(body)?;
        let encounter = body.get("encounter").and_then(Value::as_str);
        let monsters = str_list("monsters");
        let relics = str_list("relics");
        let deck = str_list("deck");
        let potions = str_list("potions").unwrap_or_default();
        let built = crate::simlab::new_position(
            version,
            character,
            encounter,
            monsters.as_deref(),
            body.get("ascension").and_then(Value::as_i64).unwrap_or(0),
            body.get("seed").and_then(Value::as_i64).unwrap_or(1),
            body.get("hp").and_then(Value::as_i64),
            body.get("maxHp").and_then(Value::as_i64),
            relics.as_deref(),
            &potions,
            deck.as_deref(),
            starter_hp,
            &starter_deck,
            Some(starter_relic),
        )
        .map_err(|e| ApiError::new(422, e.0))?;
        Ok(json!({"position": built.position, "wire": built.wire, "notes": built.notes}))
    }

    /// `POST /api/simlab/position/import`: a pasted position / engine wire /
    /// SpireMentor Copy-state → position.
    fn simlab_import(body: &Value) -> Result<Value, ApiError> {
        let text = body.get("text").and_then(Value::as_str).unwrap_or("");
        let (position, notes, source) =
            crate::simlab::position_from_paste(text).map_err(|e| ApiError::new(422, e.0))?;
        Ok(json!({"position": position, "notes": notes, "source": source}))
    }

    /// `POST /api/simlab/position/moves`: every legal first move and the
    /// position it leads to.
    fn simlab_moves(body: &Value) -> Result<Value, ApiError> {
        let empty = json!({});
        let pos = body.get("position").unwrap_or(&empty);
        let built = crate::simlab::build_position(pos).map_err(|e| ApiError::new(422, e.0))?;
        let version =
            crate::simlab::position_game_version(pos).map_err(|e| ApiError::new(422, e.0))?;
        let name_of =
            |kind: &str, ident: &str| crate::simlab_catalog::name_of_for(version, kind, ident);
        let moves = crate::simlab::first_moves(&built.wire, &name_of)
            .map_err(|e| ApiError::new(422, e.0))?;
        Ok(json!({"moves": moves, "wire": built.wire}))
    }

    /// `POST /api/simlab/position/mcts`: the per-first-move MCTS verdict (the
    /// same search the workbench hint runs).
    fn simlab_mcts(body: &Value) -> Result<Value, ApiError> {
        let empty = json!({});
        let pos = body.get("position").unwrap_or(&empty);
        let built = crate::simlab::build_position(pos).map_err(|e| ApiError::new(422, e.0))?;
        let sims = body
            .get("sims")
            .and_then(Value::as_i64)
            .unwrap_or(1000)
            .clamp(1, 20000);
        let config = json!({
            "sims": sims,
            "seed": body.get("seed").and_then(Value::as_i64).unwrap_or(0),
            "depth": body.get("depth").and_then(Value::as_i64).unwrap_or(200),
            "threads": body.get("threads").and_then(Value::as_i64).unwrap_or(1).max(1),
            "hidden_info": body.get("hidden_info").and_then(Value::as_str).unwrap_or("beliefdraw"),
        });
        let version =
            crate::simlab::position_game_version(pos).map_err(|e| ApiError::new(422, e.0))?;
        let name_of =
            |kind: &str, ident: &str| crate::simlab_catalog::name_of_for(version, kind, ident);
        crate::simlab::mcts_eval(&built.wire, &config, &name_of)
            .map_err(|e| ApiError::new(422, e.0))
    }

    /// `POST /api/live/import-state`: rebuild the run behind a pasted state,
    /// replacing the live run. The browser rebuilds the RUN position only (the
    /// exact-fight resume is desktop-only); a fight paste parks on the map and
    /// says so in the notes. Everything that can fail is settled BEFORE the
    /// current run is torn down. 422 when the paste cannot be read, 400 when
    /// the engine cannot rebuild it.
    fn live_import_state(&mut self, body: &Value) -> Result<Value, ApiError> {
        let payload = payload_of(body)?;
        let version = body_game_version(body)?;
        let imported = crate::state_import::parse_pasted_state(payload)
            .map_err(|e| ApiError::new(422, e.0))?;
        // `_import_lane`: which lane to rebuild, and why the other is not
        // available. "auto" prefers the fight.
        let lane = match body.get("lane").and_then(Value::as_str).unwrap_or("auto") {
            "auto" => {
                if imported.has_combat() {
                    "combat"
                } else {
                    "run"
                }
            }
            "combat" if !imported.has_combat() => {
                return Err(ApiError::new(
                    409,
                    format!(
                        "this paste carries no fight — it was copied on the {} screen. Import it as a run instead.",
                        imported.surface
                    ),
                ));
            }
            l @ ("combat" | "run") => l,
            other => {
                return Err(ApiError::new(
                    400,
                    format!("unknown lane {other:?} (want auto, combat or run)"),
                ))
            }
        };
        // Build in a fresh session first; only swap it in once it succeeds, so
        // a rejected paste never costs the user the run they already had.
        let mut sess = self.new_session(&imported.run.character, version);
        let mut snapshot = imported.run.clone();
        if lane == "run" {
            // The run position only: the fight stays out of the graft, and
            // out of what a seek / reload / save rebuilds.
            snapshot.combat = None;
        }
        let state = match sess.start_imported(snapshot) {
            Ok(v) => v,
            Err(e) => {
                sess.discard();
                // A coord off this seed's map, a character with no port: name
                // it with the engine's words rather than a bare 500.
                return Err(ApiError::new(
                    400,
                    format!("could not rebuild this state: {}", e.message),
                ));
            }
        };
        let notes = sess.import_notes();
        self.replace_live(sess, &host_now(body));
        let mut step = live_step(state, &self.ctx());
        // The graft's warnings ride alongside the parser's: the user sees one
        // list.
        let mut warnings = imported.warnings.clone();
        warnings.extend(notes);
        if let Some(obj) = step.as_object_mut() {
            let mut fidelity = imported.fidelity();
            if let Some(f) = fidelity.as_object_mut() {
                f.insert("lane".into(), json!(lane));
                f.insert("warnings".into(), json!(warnings));
            }
            obj.insert("import".into(), fidelity);
        }
        Ok(step)
    }

    fn live_from_replay(&mut self, body: &Value) -> Result<Value, ApiError> {
        let idx = idx_of(body)?;
        let (ident, line) = replay::resume_plan(self.store()?, idx)?;
        let mut sess = self.new_session(&ident.character, ident.game_version);
        sess.set_e2e_setup(ident.run_mode && ident.e2e_setup);
        let state = match sess.load_line(
            &ident.character,
            &ident.seed,
            ident.ascension,
            &line,
            &ident.act1,
            ident.run_mode,
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                sess.discard();
                return Err(ApiError::new(
                    409,
                    format!("could not rebuild this run: {}", e.message),
                ));
            }
        };
        self.replace_live(sess, &host_now(body));
        Ok(live_step(state, &self.ctx()))
    }

    /// `POST /api/live/start-faithful`: a run-mode run under the heuristic
    /// brain at the given budget (`seed` is a run integer as a string or a
    /// `run_N` label). 400 on a bad seed / config.
    fn live_start_faithful(&mut self, body: &Value) -> Result<Value, ApiError> {
        let s = |k: &str, d: &str| body.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        let u = |k: &str, d: u32| -> Result<u32, ApiError> {
            match body.get(k) {
                None | Some(Value::Null) => Ok(d),
                Some(v) => v
                    .as_i64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| {
                        ApiError::new(400, format!("{k} must be a non-negative integer"))
                    }),
            }
        };
        // `e2e: true`: the e2e student's training run (Ironclad, v0.111.0,
        // act 1 rolled, potion swap on; ascension default 10) driven by the
        // `e2e` brain. The run's character and build are forced.
        let e2e = body.get("e2e").and_then(Value::as_bool).unwrap_or(false);
        if e2e && !cfg!(feature = "e2e") {
            return Err(ApiError::new(400, "this build has no e2e feature"));
        }
        let character = if e2e {
            "Ironclad".to_string()
        } else {
            s("character", "Ironclad")
        };
        let seed = s("seed", "run_1");
        let default_asc = if e2e { 10 } else { 0 };
        let ascension = body
            .get("ascension")
            .and_then(Value::as_i64)
            .unwrap_or(default_asc);
        let ascension = u8::try_from(ascension)
            .map_err(|_| ApiError::new(400, format!("ascension {ascension} out of range")))?;
        let sims = u("sims", 1000)?;
        let boss_sims = u("boss_sims", 4000)?;
        let threads = u("threads", 1)?;
        let version = if e2e {
            GameVersion::V0_111_0
        } else {
            body_game_version(body)?
        };
        let seed_int = crate::runmode::hash_seed(&seed);
        let mut sess = self.new_session(&character, version);
        sess.set_e2e_setup(e2e);
        let state = match sess.start_run_mode(
            &character,
            seed_int,
            ascension,
            if e2e { "e2e" } else { "heuristic" },
            "rust",
            sims,
            boss_sims,
            threads,
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                sess.discard();
                return Err(ApiError::new(400, e.message));
            }
        };
        self.replace_live(sess, &host_now(body));
        Ok(live_step(state, &self.ctx()))
    }

    /// `POST /api/live/run-mode`: update any subset of the two-axis run
    /// mode; the resolved config comes back.
    fn live_run_mode(&mut self, body: &Value) -> Result<Value, ApiError> {
        let sess = self.live()?;
        let str_of = |k: &str| body.get(k).and_then(Value::as_str);
        let int_of = |k: &str| body.get(k).and_then(Value::as_i64);
        if let Some(a) = str_of("advisor") {
            sess.set_advisor(a)?;
        }
        sess.set_run_mode(
            str_of("off_combat_brain"),
            str_of("ppo_impl"),
            int_of("combat_sims"),
            int_of("combat_boss_sims"),
            int_of("combat_threads"),
            int_of("reward_sims"),
        )?;
        Ok(sess.run_mode_config())
    }

    /// `POST /api/live/e2e-advice`: the e2e student's evaluation of the live
    /// node (`Session::e2e_advice`); 424 until its net is loaded.
    fn live_e2e_advice(&mut self) -> Result<Value, ApiError> {
        #[cfg(feature = "e2e")]
        {
            Ok(self.live()?.e2e_advice()?)
        }
        #[cfg(not(feature = "e2e"))]
        {
            Err(ApiError::new(404, "this build has no e2e feature"))
        }
    }

    /// `POST /api/live/wpa-verify {node_id, max_steps?, max_seconds?}`.
    fn live_wpa_verify(&mut self, body: &Value) -> Result<Value, ApiError> {
        let node_id = body
            .get("node_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| ApiError::new(400, "node_id must be an integer"))?;
        let max_steps = body
            .get("max_steps")
            .and_then(Value::as_u64)
            .unwrap_or(2_000);
        let max_seconds = body
            .get("max_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(120.0);
        let boxed: Option<Box<dyn Fn() -> f64>> = match self.config.now_ms {
            Some(f) => Some(Box::new(f)),
            None => native_clock(),
        };
        let sess = self.live()?;
        Ok(sess.wpa_verify(node_id, max_steps, max_seconds, boxed.as_deref())?)
    }

    fn live_map(&mut self) -> Result<Value, ApiError> {
        let sess = self.live()?;
        let state = sess.state()?;
        let full_map = sess.map()?;
        let mut graphs = Vec::new();
        if let Some(full_map) = full_map {
            graphs.push(json!({
                "idx": 0,
                "act": state.get("act").cloned().unwrap_or(Value::Null),
                "act_name": state.get("act_name").cloned().unwrap_or(Value::Null),
                "floor": state.get("floor").cloned().unwrap_or(Value::Null),
                "full_map": crate::enrich::enrich_map_nodes(full_map, self.ctx().assets, &self.config.assets_base),
            }));
        }
        Ok(json!({"rows": [], "graphs": graphs}))
    }

    fn live_action(&mut self, body: &Value) -> Result<Value, ApiError> {
        let action = body
            .get("action")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::new(400, "action must be a string"))?
            .to_string();
        let args = body
            .get("args")
            .cloned()
            .filter(Value::is_object)
            .unwrap_or_else(empty_object);
        let sess = self.live()?;
        sess.capture_choice();
        let result = sess.apply(&action, &args)?;
        Ok(json!({
            "step": live_step(result.state, &self.ctx()),
            "error": result.error,
            "label": result.label,
            "details": result.details,
        }))
    }

    // ── saves (host-owned storage) ─────────────────────────────────────

    /// `POST /api/live/save`'s payload: the active line under `name`.
    /// `saved_at` is the host's clock (ISO-8601, `Z`). 400 if no run is
    /// active or the name is blank.
    pub fn build_save(&self, name: &str, saved_at: &str) -> Result<Value, ApiError> {
        let sess = self
            .live
            .as_ref()
            .ok_or_else(|| ApiError::new(400, "no live run"))?;
        if name.trim().is_empty() {
            return Err(ApiError::new(400, "save name must not be blank"));
        }
        let (line, progress) = sess.active_line()?;
        let values = sess.active_node_values()?;
        let mut out = saves::build_save(
            name,
            saved_at,
            &sess.character,
            &sess.seed,
            sess.ascension,
            sess.backend(),
            sess.run_mode(),
            &sess.act1,
            sess.game_version,
            &progress,
            &line,
            &values,
        );
        // Named only for an e2e start (every other save keeps its bytes).
        if sess.e2e_setup() {
            out["e2e_setup"] = json!(true);
        }
        Ok(out)
    }

    /// `POST /api/saves/load`: restore a save payload into a fresh session,
    /// replacing any current one (whose run log is written, stamped `now`;
    /// see [`Self::replace_live`]). 404 for an invalid payload, 409 if replay
    /// diverges.
    pub fn restore_save(&mut self, payload: &Value, now: &str) -> Result<Value, ApiError> {
        if !saves::is_valid(payload) {
            return Err(ApiError::new(404, "unknown save"));
        }
        // An unknown build is refused outright rather than replayed as
        // v0.107.1 (`RunIdentity` reads it as the default).
        crate::session::game_version_of(payload)
            .map_err(|e| ApiError::new(409, format!("cannot restore this save: {e}")))?;
        let ident = saves::RunIdentity::from_save(payload);
        if ident.backend != "rust" {
            return Err(ApiError::new(
                409,
                format!(
                    "save was recorded on the {:?} backend; the browser workbench replays rust saves only",
                    ident.backend
                ),
            ));
        }
        let empty = Vec::new();
        let line = payload
            .get("line")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        let node_values: Option<Vec<Option<f64>>> = payload
            .get("node_values")
            .and_then(Value::as_array)
            .map(|vs| vs.iter().map(Value::as_f64).collect());
        let mut failures = Vec::new();
        for run_mode in ident.restore_attempts() {
            let mut sess = self.new_session(&ident.character, ident.game_version);
            sess.set_e2e_setup(run_mode && ident.e2e_setup);
            match sess.load_line(
                &ident.character,
                &ident.seed,
                ident.ascension,
                line,
                &ident.act1,
                run_mode,
                node_values.clone(),
            ) {
                Ok(state) => {
                    self.replace_live(sess, now);
                    return Ok(live_step(state, &self.ctx()));
                }
                Err(e) => {
                    sess.discard();
                    let tag = if run_mode { "True" } else { "False" };
                    failures.push(format!("[run_mode={tag}] {}", e.message));
                }
            }
        }
        Err(ApiError::new(409, failures.join("; ")))
    }

    /// The save-list header for a stored payload.
    #[must_use]
    pub fn save_header(payload: &Value) -> Value {
        saves::header(payload)
    }

    // ── session persistence (reload / process death) ───────────────────

    /// The live session's tree without states, or `null`.
    #[must_use]
    pub fn export_persist(&self) -> Value {
        self.live
            .as_ref()
            .map_or(Value::Null, Session::export_persist)
    }

    /// Rebuild a session from [`Self::export_persist`]; returns the
    /// enriched active step.
    pub fn import_persist(
        &mut self,
        payload: &Value,
        progress: impl FnMut(usize, usize),
    ) -> Result<Value, ApiError> {
        let character = payload
            .get("character")
            .and_then(Value::as_str)
            .unwrap_or("");
        let version = crate::session::game_version_of(payload)
            .map_err(|e| ApiError::new(409, format!("cannot restore this session: {e}")))?;
        let mut sess = self.new_session(character, version);
        let state = match sess.import_persist(payload, progress) {
            Ok(v) => v,
            Err(e) => {
                sess.discard();
                return Err(e.into());
            }
        };
        self.install(sess);
        Ok(live_step(state, &self.ctx()))
    }
}

/// The host's clock from a request body (`now`, ISO-8601) for the run logs a
/// route writes; a host without one (the golden replay) gets the dump's
/// pinned instant.
fn host_now(body: &Value) -> String {
    body.get("now")
        .and_then(Value::as_str)
        .unwrap_or(runlog::DEFAULT_NOW_ISO)
        .to_string()
}

/// `Instant`-based clock for native hosts (none on wasm32, where the host
/// supplies `Config::now_ms`).
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::unnecessary_wraps)]
fn native_clock() -> Option<Box<dyn Fn() -> f64>> {
    let start = std::time::Instant::now();
    Some(Box::new(move || start.elapsed().as_secs_f64() * 1000.0))
}

#[cfg(target_arch = "wasm32")]
fn native_clock() -> Option<Box<dyn Fn() -> f64>> {
    None
}

/// Epoch milliseconds off `SystemTime` (native hosts without `Config::now_ms`).
#[cfg(not(target_arch = "wasm32"))]
fn native_epoch_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

/// No clock on wasm without the host's `now_ms` (the shell always sets it).
#[cfg(target_arch = "wasm32")]
fn native_epoch_ms() -> f64 {
    0.0
}

/// `%XX` → byte, `+` → space: the query-string decoding of a `baseline` value.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = &s[i + 1..i + 3];
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 2;
                } else {
                    out.push(b'%');
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
