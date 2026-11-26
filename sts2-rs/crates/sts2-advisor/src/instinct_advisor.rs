//! `InstinctAdvisor`: the Instinct student, loaded once, plus its per-run
//! state, answering one `advice_request` at a time (GH #1426 step 4; was
//! `realgame_instinct.InstinctAdvisor` + `realgame_instinct_screens`'
//! dispatch).
//!
//! Thread-safe: requests arrive on the advisor's socket loop, but the
//! control plane may ask from its HTTP threads.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Map, Value};
use sts2_core::run_manager::RunManager;
use sts2_e2e::env::Driver;
use sts2_e2e::net::E2eNet;
use sts2_e2e::tokens::TokCache;

use crate::advisor_mirror::AdvisorMirror;
use crate::graft::{self, GraftError};
use crate::i18n::{tr, trf};
use crate::instinct::{self, Pick};
use crate::instinct_combat::{self, InstinctCombatError, PickerError};
use crate::instinct_screens::{self as screens, CardKey, Evaluator};
use crate::py::{self, get, truthy};
use crate::search::RecommendConfig;

/// The characters the student plays (GH #1197: one net for all five).
pub const INSTINCT_CHARACTERS: [&str; 5] =
    ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"];
/// The build the student was trained on, and a run's fallback build.
pub const INSTINCT_GAME_VERSION: &str = "v0.111.0";
/// The builds Instinct plays, each on its own rules.
pub const INSTINCT_GAME_VERSIONS: [&str; 2] = ["v0.111.0", "v0.113.0"];
/// The weights, relative to the data root.
pub const WEIGHTS_RELPATH: &str = "data/e2e/allchar_e2e_d4500_ema0999.e2ew";
/// The Fool's own student (GH #1535), relative to the data root: a
/// v0.113.0 net, so it answers only a v0.113.0 game.
pub const FOOL_WEIGHTS_RELPATH: &str = "data/e2e/fool_e2e_d850_ema0996.e2ew";
/// The Fool's net's build: the Fool exists only from v0.113.0 on.
pub const FOOL_GAME_VERSION: &str = "v0.113.0";

/// Whether the board's character is the Fool (whose net is
/// [`FOOL_WEIGHTS_RELPATH`]).
fn is_fool(state: &Value) -> bool {
    norm_char(get(get(state, "run_state"), "character")) == "fool"
}

/// Why Instinct did not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstinctError {
    /// Instinct cannot answer this position (`InstinctUnavailable`): the
    /// message is player-facing, and the search answers instead.
    Unavailable(String),
    /// A failure: `(python exception type, message)` as the Python advisor
    /// raised it (`ValueError` for a board the engine cannot model).
    Failed(String, String),
}

impl std::fmt::Display for InstinctError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(m) | Self::Failed(_, m) => f.write_str(m),
        }
    }
}

impl From<GraftError> for InstinctError {
    fn from(e: GraftError) -> Self {
        match e {
            GraftError::Unavailable(m) => Self::Unavailable(m),
            GraftError::Value(m) => Self::Failed("ValueError".into(), m),
        }
    }
}

/// The build a live run is rebuilt on: the one the game is playing when
/// Instinct plays it, else the training build (`engine_game_version`).
#[must_use]
pub fn engine_game_version(state: &Value) -> &'static str {
    match crate::headless::game_version(state) {
        Ok(v) => {
            let v = v.unwrap_or("v0.107.1");
            INSTINCT_GAME_VERSIONS
                .into_iter()
                .find(|g| *g == v)
                .unwrap_or(INSTINCT_GAME_VERSION)
        }
        Err(_) => INSTINCT_GAME_VERSION,
    }
}

/// `_norm_char`: a character id or name as one lowercase key.
fn norm_char(raw: &Value) -> String {
    crate::headless::character_display_name(raw)
        .map_or_else(
            || {
                if truthy(raw) {
                    py::str_of(raw)
                } else {
                    String::new()
                }
            },
            str::to_string,
        )
        .trim()
        .to_lowercase()
}

/// A multiplayer run (`player_count` > 1).
#[must_use]
pub fn is_coop(state: &Value) -> bool {
    let n = get(state, "player_count");
    let n = if truthy(n) { n } else { &json!(1) };
    match n {
        Value::Bool(b) => i64::from(*b) > 1,
        #[allow(clippy::cast_possible_truncation)]
        Value::Number(x) => x
            .as_i64()
            .or_else(|| x.as_f64().map(|f| f as i64))
            .is_some_and(|v| v > 1),
        Value::String(s) => s.trim().parse::<i64>().is_ok_and(|v| v > 1),
        _ => false,
    }
}

/// `_room_kind`: the fight's room as the card reward after it reads it.
fn room_name(state: &Value) -> &'static str {
    match instinct_combat::room_kind(state) {
        sts2_core::runstate::RoomType::Elite => "elite",
        sts2_core::runstate::RoomType::Boss => "boss",
        _ => "monster",
    }
}

/// The run a driver belongs to: `(seed, character)`.
type RunKey = (Value, String);

/// One net, loaded on first use.
#[derive(Debug, Default)]
struct Slot {
    loaded: bool,
    student: Option<Arc<E2eNet>>,
    error: Option<String>,
}

#[derive(Debug, Default)]
struct Inner {
    /// The all-character student.
    main: Slot,
    /// The Fool's student.
    fool: Slot,
    run_key: Option<RunKey>,
    driver: Option<(Driver, TokCache)>,
    /// The last fight's room (`monster` / `elite` / `boss`): the card reward
    /// after it is that room's.
    last_room: Option<String>,
    /// The `advice_for` of the previous request: what opened a card screen.
    last_screen: Option<String>,
    /// The deck card Instinct removes in the shop, asked on the next screen.
    shop_removal: Option<CardKey>,
    /// The current event's trail of taken options.
    event_track: Option<Value>,
    /// The win probe's estimate of the request's position (GH #1222).
    first_estimate: Option<Value>,
    /// The last position a screen handler asked the student about.
    last_pick: Option<Pick>,
}

impl Inner {
    /// This run's driver, fresh when the request is another run's.
    fn driver_for(&mut self, state: &Value, seed: i64) -> &mut (Driver, TokCache) {
        let key = (
            get(state, "seed").clone(),
            norm_char(get(get(state, "run_state"), "character")),
        );
        if self.run_key.as_ref() != Some(&key) || self.driver.is_none() {
            self.run_key = Some(key);
            self.driver = Some((Driver::new(seed), TokCache::new()));
        }
        self.driver.as_mut().expect("just set")
    }
}

/// A copy of the run's driver for a probe, after the run's own driver has
/// seen the position: a round that opens on an in-fight choice logs its
/// enemies' intents there, as the native env does, not at the first turn
/// decision after it (GH #1764).
fn observed_copy(slot: &mut (Driver, TokCache), rm: &RunManager) -> (Driver, TokCache) {
    // A failed observation only costs the intent log a snapshot.
    let _ = slot.0.observe_combat_position(rm);
    slot.clone()
}

/// Read `net`'s tables on this thread while the guard lives (the Fool's
/// v0.113.0 net reads v0.113.0's; `sts2_e2e::build::Scope`).
fn tables(net: Option<&E2eNet>) -> Option<sts2_e2e::build::Scope> {
    net.and_then(|n| n.build().ok())
        .map(sts2_e2e::build::Scope::enter)
}

impl InstinctAdvisor {
    /// The already-loaded student for `state`'s character (no load).
    fn loaded_for(t: &Inner, state: &Value) -> Option<Arc<E2eNet>> {
        if is_fool(state) {
            t.fool.student.clone()
        } else {
            t.main.student.clone()
        }
    }
}

/// The student behind a screen handler, on this run's driver.
struct RunEvaluator<'a> {
    net: Option<&'a E2eNet>,
    inner: &'a mut Inner,
    state: &'a Value,
    grant_picked: Option<i64>,
}

impl Evaluator for RunEvaluator<'_> {
    fn set_grant_picked(&mut self, n: i64) {
        self.grant_picked = Some(n);
    }

    fn evaluate(&mut self, rm: &mut RunManager, expect: &[&str]) -> Result<Pick, String> {
        let Some(net) = self.net else {
            return Err(tr("no Instinct net loaded").to_string());
        };
        let expect: Vec<String> = expect.iter().map(|s| (*s).to_string()).collect();
        let (driver, cache) = self.inner.driver_for(self.state, i64::from(rm.seed));
        if let Some(n) = self.grant_picked.take() {
            driver.set_grant_picked(n);
        }
        let ev = instinct::evaluate(net, driver, cache, rm, &expect)?;
        if self.inner.first_estimate.is_none() {
            self.inner.first_estimate.clone_from(&ev.estimate);
        }
        self.inner.last_pick = Some(ev.pick.clone());
        Ok(ev.pick)
    }
}

/// The search's context for a request, which some Instinct answers stand on.
#[derive(Debug)]
pub struct Context<'a> {
    /// The display-name tables (shop and potion labels).
    pub names: &'a crate::display_names::Names,
    pub cfg: Option<&'a RecommendConfig>,
    pub mirror: Option<&'a mut AdvisorMirror>,
    pub session: Option<&'a mut Map<String, Value>>,
    pub sims: Option<u32>,
}

/// The students (the all-character net and the Fool's), each loaded once,
/// plus the per-run state.
#[derive(Debug)]
pub struct InstinctAdvisor {
    weights: PathBuf,
    fool_weights: Option<PathBuf>,
    inner: Mutex<Inner>,
}

impl InstinctAdvisor {
    /// The all-character student at `weights`, no Fool net.
    #[must_use]
    pub fn new(weights: PathBuf) -> Self {
        Self {
            weights,
            fool_weights: None,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// The Fool's student at `weights` too (v0.113.0 games only).
    #[must_use]
    pub fn with_fool(mut self, weights: PathBuf) -> Self {
        self.fool_weights = Some(weights);
        self
    }

    /// The student that plays `state`'s character, loading it on first use,
    /// with why it is not loaded.
    fn net_for(
        &self,
        t: &mut Inner,
        state: &Value,
        log: &mut Vec<(&'static str, String)>,
    ) -> (Option<Arc<E2eNet>>, Option<String>) {
        let fool = is_fool(state);
        let slot = if fool { &mut t.fool } else { &mut t.main };
        match (fool, self.fool_weights.as_deref()) {
            (true, None) => (None, Some(tr("no Instinct net for the Fool").into())),
            (true, Some(w)) => {
                Self::load_slot(slot, w, FOOL_GAME_VERSION, log);
                (slot.student.clone(), slot.error.clone())
            }
            (false, _) => {
                Self::load_slot(slot, &self.weights, sts2_e2e::env::GAME_VERSION, log);
                (slot.student.clone(), slot.error.clone())
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Load `weights` into `slot` on first use (refusing a net stamped for
    /// another build than `build`); the log line it leaves, if any.
    fn load_slot(
        slot: &mut Slot,
        weights: &std::path::Path,
        build: &str,
        log: &mut Vec<(&'static str, String)>,
    ) {
        if slot.loaded {
            return;
        }
        slot.loaded = true;
        if !weights.is_file() {
            slot.error = Some(trf(
                "no Instinct net at {path}",
                &[("path", &weights.display())],
            ));
            return;
        }
        let loaded = std::fs::read(weights)
            .map_err(|e| e.to_string())
            .and_then(|b| E2eNet::from_bytes(&b))
            .and_then(|net| match net.header().game_version.as_deref() {
                Some(v) if v != build => Err(format!(
                    "e2e weights are stamped for {v}, the student plays {build}"
                )),
                _ => Ok(net),
            });
        match loaded {
            Ok(net) => {
                let name = weights
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                log.push(("info", format!("instinct: loaded {name}")));
                slot.student = Some(Arc::new(net));
            }
            Err(e) => {
                let error = trf("Instinct net failed to load: {e}", &[("e", &e)]);
                log.push(("warning", format!("instinct: {error}")));
                slot.error = Some(error);
            }
        }
    }

    /// Whether the net loaded (loading it on first ask), with the load's
    /// log lines.
    pub fn available(&self, log: &mut Vec<(&'static str, String)>) -> bool {
        let mut t = self.lock();
        Self::load_slot(&mut t.main, &self.weights, sts2_e2e::env::GAME_VERSION, log);
        t.main.student.is_some()
    }

    /// Whether the Fool's net loaded (loading it on first ask).
    pub fn fool_available(&self, log: &mut Vec<(&'static str, String)>) -> bool {
        let Some(w) = self.fool_weights.as_deref() else {
            return false;
        };
        let mut t = self.lock();
        Self::load_slot(&mut t.fool, w, FOOL_GAME_VERSION, log);
        t.fool.student.is_some()
    }

    /// `capabilities.instinct`: what this advisor's Instinct plays.
    pub fn capabilities(&self, log: &mut Vec<(&'static str, String)>) -> Value {
        let mut out: Vec<Value> = Vec::new();
        if self.available(log) {
            out.extend(INSTINCT_CHARACTERS.iter().flat_map(|c| {
                INSTINCT_GAME_VERSIONS
                    .iter()
                    .map(move |v| json!({"character": c, "game_version": v}))
            }));
        }
        if self.fool_available(log) {
            out.push(json!({"character": "Fool", "game_version": FOOL_GAME_VERSION}));
        }
        Value::Array(out)
    }

    /// Forget the per-run state (run over, rewind, new build); the log line
    /// when there was a driver to drop.
    pub fn reset(&self, reason: &str) -> Option<String> {
        let mut t = self.lock();
        let line = t
            .driver
            .is_some()
            .then(|| format!("instinct: state dropped ({reason})"));
        t.run_key = None;
        t.driver = None;
        t.last_room = None;
        t.last_screen = None;
        t.shop_removal = None;
        t.event_track = None;
        line
    }

    /// Follow the live fight's enemies through a rebuild that renamed their
    /// uids (`(uid then, uid now)`, from the fight mirror, GH #1651).
    pub fn remap_enemies(&self, pairs: &[(u32, u32)]) {
        if let Some((driver, _)) = self.lock().driver.as_mut() {
            driver.remap_enemies(pairs);
        }
    }

    /// Follow a move the player made: an event page's choice, so the next
    /// page of that event is replayed right.
    pub fn on_human_action(&self, msg: &Value) {
        if get(msg, "decision_type").as_str() != Some("event") {
            return;
        }
        let mut t = self.lock();
        if let Some(track) = &t.event_track {
            t.event_track = Some(screens::note_event_choice(track, msg));
        }
    }

    /// One screen handler on this run's state (`InstinctAdvisor.rust_screen`).
    /// Without a net only the screens answered by rule (Claws, the Blitz
    /// merchant) answer.
    ///
    /// # Errors
    ///
    /// As [`Self::advise`].
    pub fn screen(
        &self,
        screen: &str,
        state: &Value,
        names: &crate::display_names::Names,
    ) -> Result<Value, InstinctError> {
        let mut guard = self.lock();
        let t = &mut *guard;
        let (net, _) = self.net_for(t, state, &mut Vec::new());
        let _tables = tables(net.as_deref());
        Self::screen_on(net.as_deref(), t, screen, state, names)
    }

    /// One request's dispatch without `advise`'s eligibility check, net load
    /// or bookkeeping (`InstinctAdvisor._dispatch`): what answers without a
    /// net (a remembered pick, a confirm, a rule) still answers.
    ///
    /// # Errors
    ///
    /// As [`Self::advise`].
    pub fn dispatch_unloaded(
        &self,
        state: &Value,
        advice_for: &str,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        let mut guard = self.lock();
        let t = &mut *guard;
        let net = Self::loaded_for(t, state);
        let _tables = tables(net.as_deref());
        Self::dispatch(net.as_deref(), t, state, advice_for, ctx, log)
    }

    /// An in-combat `card_select` on the mirror's fork, without a net load
    /// (`InstinctAdvisor._advise_combat_select`).
    ///
    /// # Errors
    ///
    /// As [`Self::advise`].
    pub fn combat_select_unloaded(
        &self,
        state: &Value,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        let mut guard = self.lock();
        let t = &mut *guard;
        let net = Self::loaded_for(t, state);
        let _tables = tables(net.as_deref());
        Self::advise_combat_select(net.as_deref(), t, state, ctx, log)
    }

    /// Peek `rm` with this run's driver state and run the student
    /// (`InstinctAdvisor.evaluate`).
    ///
    /// # Errors
    ///
    /// The player-facing reason Instinct cannot answer here.
    pub fn evaluate(
        &self,
        state: &Value,
        rm: &mut RunManager,
        expect: &[String],
    ) -> Result<instinct::Evaluated, String> {
        let mut guard = self.lock();
        let t = &mut *guard;
        let net = self
            .net_for(t, state, &mut Vec::new())
            .0
            .ok_or(tr("no Instinct net loaded"))?;
        let _tables = tables(Some(&net));
        let (driver, cache) = t.driver_for(state, i64::from(rm.seed));
        let ev = instinct::evaluate(&net, driver, cache, rm, expect)?;
        if t.first_estimate.is_none() {
            t.first_estimate.clone_from(&ev.estimate);
        }
        Ok(ev)
    }

    /// The student's picks on the in-fight choice `rm` stands on (a run
    /// grafted from `state` with the fight entered), on a copy of this
    /// run's driver: the native run's COMBAT_CHOICE decisions.
    ///
    /// # Errors
    ///
    /// No net loaded, a move the engine refuses, a position the student
    /// cannot evaluate.
    pub fn choice_picks(
        &self,
        state: &Value,
        rm: &mut RunManager,
        max_select: i64,
    ) -> Result<Vec<i64>, String> {
        let mut guard = self.lock();
        let t = &mut *guard;
        let net = self
            .net_for(t, state, &mut Vec::new())
            .0
            .ok_or(tr("no Instinct net loaded"))?;
        let _tables = tables(Some(&net));
        let (mut probe, mut cache) = observed_copy(t.driver_for(state, i64::from(rm.seed)), rm);
        let mut out = instinct_combat::ChoicePicks::default();
        instinct_combat::student_choice_picks(
            &net, &mut probe, &mut cache, rm, max_select, &mut out,
        )
        .map_err(|(_, m)| m)
    }

    #[must_use]
    pub fn last_room(&self) -> Option<String> {
        self.lock().last_room.clone()
    }

    pub fn set_last_room(&self, v: Option<String>) {
        self.lock().last_room = v;
    }

    #[must_use]
    pub fn last_screen(&self) -> Option<String> {
        self.lock().last_screen.clone()
    }

    pub fn set_last_screen(&self, v: Option<String>) {
        self.lock().last_screen = v;
    }

    #[must_use]
    pub fn shop_removal(&self) -> Option<CardKey> {
        self.lock().shop_removal.clone()
    }

    pub fn set_shop_removal(&self, v: Option<CardKey>) {
        self.lock().shop_removal = v;
    }

    #[must_use]
    pub fn event_track(&self) -> Option<Value> {
        self.lock().event_track.clone()
    }

    pub fn set_event_track(&self, v: Option<Value>) {
        self.lock().event_track = v;
    }

    #[must_use]
    pub fn last_pick(&self) -> Option<Pick> {
        self.lock().last_pick.clone()
    }

    /// Why Instinct cannot play this run, or `None` when it can.
    #[must_use]
    pub fn eligibility(state: &Value) -> Option<String> {
        let character = norm_char(get(get(state, "run_state"), "character"));
        let version = match crate::headless::game_version(state) {
            Ok(v) => v.unwrap_or("v0.107.1").to_string(),
            Err(_) => "None".to_string(),
        };
        if character == "fool" {
            // The Fool's own net (v0.113.0); the Fool does not exist earlier.
            return (version != FOOL_GAME_VERSION).then(|| {
                trf(
                    "Instinct plays the Fool on {build} only (this game is {version})",
                    &[("build", &FOOL_GAME_VERSION), ("version", &version)],
                )
            });
        }
        if !INSTINCT_CHARACTERS
            .iter()
            .any(|c| c.to_lowercase() == character)
        {
            return Some(trf(
                "Instinct plays {characters} and the Fool only",
                &[("characters", &INSTINCT_CHARACTERS.join(", "))],
            ));
        }
        if !INSTINCT_GAME_VERSIONS.contains(&version.as_str()) {
            return Some(trf(
                "Instinct plays {builds} only (this game is {version})",
                &[
                    (
                        "builds",
                        &crate::i18n::join_list(
                            &INSTINCT_GAME_VERSIONS.map(str::to_string),
                            " and ",
                        ),
                    ),
                    ("version", &version),
                ],
            ));
        }
        None
    }

    /// Instinct's `advice` for one `advice_request` (`state`: its payload,
    /// typed as the `advice_for`). `log` collects `(level, line)` pairs.
    ///
    /// # Errors
    ///
    /// [`InstinctError::Unavailable`] when Instinct cannot answer (the
    /// caller answers with the search), [`InstinctError::Failed`] on a board
    /// the engine cannot model.
    pub fn advise(
        &self,
        state: &Value,
        advice_for: &str,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        if let Some(why) = Self::eligibility(state) {
            return Err(InstinctError::Unavailable(why));
        }
        let mut guard = self.lock();
        let t = &mut *guard;
        let (net, error) = self.net_for(t, state, log);
        let Some(net) = net else {
            return Err(InstinctError::Unavailable(
                error.unwrap_or_else(|| tr("no Instinct net loaded").into()),
            ));
        };
        let _tables = tables(Some(&net));
        t.first_estimate = None;
        let out = Self::dispatch(Some(&net), t, state, advice_for, ctx, log);
        t.last_screen = Some(advice_for.to_string());
        let estimate = t.first_estimate.clone();
        drop(guard);
        let mut advice = out?;
        advice["brain"] = json!("instinct");
        if let Some(e) = estimate {
            // GH #1222: the position's own estimate (the first one this
            // request evaluated; a turn plan evaluates later ones too).
            advice["estimate"] = e;
        }
        if is_coop(state) {
            let note = advice
                .as_object()
                .and_then(|m| m.get("note"))
                .map_or_else(String::new, py::str_of);
            advice["note"] = json!(trf(
                "{note} · co-op: Instinct trained single-player",
                &[("note", &note)],
            ));
        }
        Ok(advice)
    }

    fn dispatch(
        net: Option<&E2eNet>,
        t: &mut Inner,
        state: &Value,
        advice_for: &str,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        match advice_for {
            "combat_action" => return Self::advise_combat(net, t, state, ctx, log),
            "map_select" => {
                // The session's cached act graph, when the message omits it.
                let patched = ctx
                    .session
                    .and_then(|session| crate::session::apply_session_context(state, session))
                    .and_then(|map| {
                        let mut out = state.as_object()?.clone();
                        out.insert("map".into(), map);
                        Some(Value::Object(out))
                    });
                return Self::screen_on(
                    net,
                    t,
                    "map_select",
                    patched.as_ref().unwrap_or(state),
                    ctx.names,
                );
            }
            "card_select" if !get(state, "combat").is_null() => {
                return Self::advise_combat_select(net, t, state, ctx, log);
            }
            "card_reward" if crate::screen_advice::is_fight_offer(state) => {
                // A card generated into the fight (Discovery, a Skill
                // Potion's offer): the in-combat selection it is.
                let select = crate::screen_advice::fight_offer_as_select(state);
                return Self::advise_combat_select(net, t, &select, ctx, log);
            }
            "treasure" => {
                let offered: Vec<&Value> = py::list(state, "options")
                    .iter()
                    .filter(|o| {
                        o.is_object()
                            && o.as_object()
                                .and_then(|m| m.get("enabled"))
                                .is_none_or(truthy)
                    })
                    .collect();
                if let [only] = offered.as_slice() {
                    // Single player's chest: one relic, taken, not chosen.
                    let mut advice = instinct::lone_option(only);
                    advice["note"] =
                        json!(tr("Instinct (e2e net, no search) · the chest's one relic"));
                    return Ok(advice);
                }
                return Err(InstinctError::Unavailable(
                    tr("the chest relic is taken, not chosen, in Instinct's game").into(),
                ));
            }
            _ => {}
        }
        match advice_for {
            "card_reward" | "card_bundle" | "rest_site" | "card_select" | "shop" | "event"
            | "boss_relic" | "potion_swap" => Self::screen_on(net, t, advice_for, state, ctx.names),
            other => Err(InstinctError::Unavailable(trf(
                "Instinct does not read {surface} yet",
                &[(
                    "surface",
                    &if crate::i18n::is_zh() {
                        crate::i18n::surface(other)
                    } else {
                        other.replace('_', " ")
                    },
                )],
            ))),
        }
    }

    /// A screen whose handler runs in `instinct_screens`.
    fn screen_on(
        net: Option<&E2eNet>,
        t: &mut Inner,
        screen: &str,
        state: &Value,
        names: &crate::display_names::Names,
    ) -> Result<Value, InstinctError> {
        let version = engine_game_version(state);
        let last_room = t.last_room.clone();
        let last_screen = t.last_screen.clone();
        let shop_removal = t.shop_removal.clone();
        let track_in = t.event_track.clone();
        let mut ev = RunEvaluator {
            net,
            inner: t,
            state,
            grant_picked: None,
        };
        let out = match screen {
            "card_reward" => screens::card_reward(&mut ev, state, version, last_room.as_deref()),
            "card_bundle" => screens::card_bundle(&mut ev, state, version),
            "rest_site" => screens::rest_site(&mut ev, state, version),
            "map_select" => screens::map(&mut ev, state, version),
            "boss_relic" => screens::boss_relic(&mut ev, state, version),
            "potion_swap" => screens::potion_swap(&mut ev, state, version, names),
            "card_select" => {
                let mut with_event = state.clone();
                if let (Some(id), Some(obj)) = (
                    screen_event(last_screen.as_deref(), track_in.as_ref(), state),
                    with_event.as_object_mut(),
                ) {
                    obj.entry("event_id").or_insert(id);
                }
                screens::deck_select(
                    &mut ev,
                    &with_event,
                    version,
                    last_screen.as_deref(),
                    shop_removal.as_ref(),
                )
            }
            "event" => {
                let (out, track) = screens::event(&mut ev, state, version, track_in.as_ref());
                // Not past a malformed page (the Python binding raised).
                if !matches!(out, Err(GraftError::Value(_))) {
                    ev.inner.event_track = track;
                }
                out
            }
            "shop" => screens::shop(&mut ev, state, version, names).map(|(advice, removal)| {
                ev.inner.shop_removal = removal;
                advice
            }),
            other => {
                return Err(InstinctError::Failed(
                    "ValueError".into(),
                    format!("no Rust Instinct screen {other:?}"),
                ))
            }
        };
        Ok(out?)
    }

    /// A `combat_action` board: the student stands in the fight the search
    /// would build on the grafted run; its greedy rollout is the plan.
    fn advise_combat(
        net: Option<&E2eNet>,
        t: &mut Inner,
        state: &Value,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        let (mut rm, _notes) = graft::graft_run(state, engine_game_version(state))?;
        let Some(net) = net else {
            return Err(InstinctError::Unavailable(
                tr("no Instinct net loaded").into(),
            ));
        };
        let default_cfg = RecommendConfig::default();
        let cfg = ctx.cfg.unwrap_or(&default_cfg);
        let mut lines = Vec::new();
        let (driver, cache) = t.driver_for(state, i64::from(rm.seed));
        let out = instinct_combat::advise_combat(
            net, driver, cache, &mut rm, state, cfg, ctx.mirror, &mut lines,
        );
        let out = match out {
            Ok(o) => o,
            Err(InstinctCombatError::Unavailable(why)) => {
                t.last_room = Some(room_name(state).into());
                log.extend(lines.into_iter().map(|l| ("info", l)));
                return Err(InstinctError::Unavailable(why));
            }
            Err(InstinctCombatError::Board(e)) => {
                return Err(InstinctError::Failed(e.py_type().into(), e.to_string()))
            }
            Err(InstinctCombatError::State(e)) => {
                return Err(InstinctError::Failed("ValueError".into(), e.to_string()))
            }
        };
        t.last_room = Some(room_name(state).into());
        log.extend(lines.into_iter().map(|l| ("info", l)));
        if out.estimate.is_some() && t.first_estimate.is_none() {
            t.first_estimate.clone_from(&out.estimate);
        }
        if let Some(failed) = &out.plan_failed {
            log.push(("warning", format!("instinct: turn plan failed: {failed}")));
        }
        Ok(out.advice)
    }

    /// An in-combat `card_select` (Burning Pact, a Skill Potion's offer):
    /// Instinct picks on the fight mirror's fork, one pick at a time.
    fn advise_combat_select(
        net: Option<&E2eNet>,
        t: &mut Inner,
        state: &Value,
        ctx: Context<'_>,
        log: &mut Vec<(&'static str, String)>,
    ) -> Result<Value, InstinctError> {
        let (Some(mirror), Some(cfg)) = (ctx.mirror, ctx.cfg) else {
            return Err(InstinctError::Unavailable(
                tr("no fight mirror to stand on this choice").into(),
            ));
        };
        let version = engine_game_version(state);
        let mut driver_of = |seed: i64| -> Result<(Driver, TokCache), PickerError> {
            Ok(t.driver_for(state, seed).clone())
        };
        let mut wire_of = crate::board::wire_for;
        let mut lines = Vec::new();
        let out = instinct_combat::advise_choice(
            net,
            &mut driver_of,
            mirror,
            state,
            cfg,
            ctx.sims,
            version,
            &mut wire_of,
            &mut lines,
        );
        log.extend(lines.into_iter().map(|l| ("info", l)));
        out.map_err(InstinctError::Unavailable)
    }
}

/// The event that opened a card screen, for its own rules: the last
/// screen was its page, or an earlier pick of this screen on the event's
/// floor (`track` is the event trail, keyed `[seed, floor, event]`).
fn screen_event(last_screen: Option<&str>, track: Option<&Value>, state: &Value) -> Option<Value> {
    let key = get(track?, "key");
    let opened_by_event = match last_screen {
        Some("event") => true,
        Some("card_select") => py::eq(&key[1], get(state, "floor")),
        _ => false,
    };
    opened_by_event.then(|| key[2].clone())
}

#[cfg(test)]
mod tests {

    /// GH #1764: the run's driver sees a choice's position before a probe
    /// copies it.
    #[test]
    fn a_choice_probe_logs_the_rounds_intents_first() {
        let msg = crate::graft::tests::seedless_message(
            &crate::graft::tests::screen_map(&crate::graft::tests::reference_run(1)),
            1,
        );
        let (mut rm, _) = crate::graft::graft_run(&msg, "v0.111.0").expect("graft");
        let fight = sts2_wire::scenario::build_scenario_state_for_version(
            sts2_core::GameVersion::V0_111_0,
            7,
            "Ironclad",
            &vec!["STRIKE_IRONCLAD".to_owned(); 10],
            80,
            80,
            &[],
            &[],
            "AXEBOTS_NORMAL",
            10,
            None,
            None,
        )
        .expect("scenario");
        rm.debug_enter_combat(fight, sts2_core::runstate::RoomType::Monster);
        let mut slot = (Driver::new(7), TokCache::new());
        let _copy = observed_copy(&mut slot, &rm);
        let logged = slot.0.intent_log().and_then(|l| l.last_round);
        assert!(logged.is_some(), "the run's driver logged no round");
    }

    /// GH #1747: Amalgamator's second pick still knows its event.
    #[test]
    fn a_card_screens_later_pick_keeps_its_event() {
        let track = json!({"key": [null, 22, "AMALGAMATOR"]});
        let on = |floor: i64| json!({"floor": floor});
        let ev = Some(json!("AMALGAMATOR"));
        assert_eq!(screen_event(Some("event"), Some(&track), &on(22)), ev);
        assert_eq!(screen_event(Some("card_select"), Some(&track), &on(22)), ev);
        assert_eq!(
            screen_event(Some("card_select"), Some(&track), &on(25)),
            None
        );
        assert_eq!(screen_event(Some("shop"), Some(&track), &on(22)), None);
    }
    use super::*;

    fn names() -> crate::display_names::Names {
        crate::display_names::Names::new(std::iter::empty())
    }

    fn ctx(names: &crate::display_names::Names) -> Context<'_> {
        Context {
            names,
            cfg: None,
            mirror: None,
            session: None,
            sims: None,
        }
    }

    #[test]
    fn eligibility_names_the_characters_and_builds_it_plays() {
        let ok = json!({"run_state": {"character": "SILENT"}, "game_version": "v0.113.0"});
        assert_eq!(InstinctAdvisor::eligibility(&ok), None);
        let fool = json!({"run_state": {"character": "FOOL"}, "game_version": "v0.111.0"});
        assert_eq!(
            InstinctAdvisor::eligibility(&fool).as_deref(),
            Some("Instinct plays the Fool on v0.113.0 only (this game is v0.111.0)")
        );
        let fool_new = json!({"run_state": {"character": "FOOL"}, "game_version": "v0.113.0"});
        assert_eq!(InstinctAdvisor::eligibility(&fool_new), None);
        let other = json!({"run_state": {"character": "WATCHER"}, "game_version": "v0.113.0"});
        assert!(InstinctAdvisor::eligibility(&other)
            .unwrap()
            .starts_with("Instinct plays Ironclad, Silent"));
        let legacy = json!({"run_state": {"character": "IRONCLAD"}});
        assert_eq!(
            InstinctAdvisor::eligibility(&legacy).as_deref(),
            Some("Instinct plays v0.111.0 and v0.113.0 only (this game is v0.107.1)")
        );
        assert_eq!(engine_game_version(&legacy), INSTINCT_GAME_VERSION);
        assert_eq!(engine_game_version(&ok), "v0.113.0");
    }

    /// v0.114.0 has no engine build of its own (`game_version::PLAYS_AS`):
    /// the advisor reads it as v0.113.0, and Instinct plays it there, the
    /// Fool on the v0.113.0 Fool net.
    #[test]
    fn a_plays_as_build_is_advised_on_its_engine_build() {
        let silent = json!({"run_state": {"character": "SILENT"}, "game_version": "v0.114.0"});
        assert_eq!(
            crate::headless::game_version(&silent).unwrap(),
            Some("v0.113.0")
        );
        assert_eq!(InstinctAdvisor::eligibility(&silent), None);
        assert_eq!(engine_game_version(&silent), "v0.113.0");
        let fool = json!({"run_state": {"character": "FOOL"}, "game_version": "v0.114.0"});
        assert_eq!(InstinctAdvisor::eligibility(&fool), None);
        let unknown = json!({"run_state": {"character": "SILENT"}, "game_version": "v0.115.0"});
        assert!(crate::headless::game_version(&unknown).is_err());
    }

    #[test]
    fn a_coop_run_is_more_than_one_player() {
        assert!(is_coop(&json!({"player_count": 2})));
        assert!(!is_coop(&json!({"player_count": null})));
        assert!(!is_coop(&json!({"player_count": "x"})));
    }

    #[test]
    fn a_chest_with_one_relic_is_taken_without_a_net() {
        let adv = InstinctAdvisor::new(PathBuf::from("/nonexistent.e2ew"));
        let names = names();
        let one =
            json!({"options": [{"index": 0, "label": "Anchor"}, {"index": 1, "enabled": false}]});
        let advice = adv
            .dispatch_unloaded(&one, "treasure", ctx(&names), &mut Vec::new())
            .unwrap();
        assert_eq!(
            get(&advice, "note"),
            "Instinct (e2e net, no search) · the chest's one relic"
        );
        let two = json!({"options": [{"index": 0}, {"index": 1}]});
        assert!(matches!(
            adv.dispatch_unloaded(&two, "treasure", ctx(&names), &mut Vec::new()),
            Err(InstinctError::Unavailable(_))
        ));
        let mut log = Vec::new();
        assert!(!adv.available(&mut log));
        assert_eq!(
            adv.advise(
                &json!({"run_state": {"character": "IRONCLAD"}, "game_version": "v0.111.0"}),
                "map_select",
                ctx(&names),
                &mut log
            ),
            Err(InstinctError::Unavailable(
                "no Instinct net at /nonexistent.e2ew".into()
            ))
        );
        assert_eq!(
            adv.last_screen(),
            None,
            "no net: the request never dispatched"
        );
    }
}
